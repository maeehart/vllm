# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MORI (Modular RDMA Interface) integration for vLLM MoE expert parallelism.

MORI provides high-performance dispatch/combine kernels for MoE expert
parallelism using RDMA and GPU-direct communication on AMD ROCm.

Based on the reference implementation from:
https://github.com/alexsun07/vllm/tree/mori_ep
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Lazy import mori
try:
    import mori
    MORI_AVAILABLE = True
except ImportError:
    mori = None  # type: ignore
    MORI_AVAILABLE = False


def is_mori_available() -> bool:
    """Check if MORI is available."""
    return MORI_AVAILABLE


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MORI dispatch/combine kernels.
    
    MORI provides optimized all-to-all communication for MoE expert
    parallelism with support for:
    - Intra-node XGMI communication
    - Inter-node RDMA communication
    - FP8 dispatch for reduced bandwidth
    - BF16 combine for accuracy
    """

    def __init__(
        self,
        mori_op,  # mori.ops.EpDispatchCombineOp
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriPrepareAndFinalize but not installed. "
                "Please install mori from https://github.com/ROCm/mori"
            )
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        """MORI combine produces reduced output."""
        return True

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        """MORI requires int32 topk indices."""
        return torch.int32

    def supports_async(self) -> bool:
        """MORI does not currently support async dispatch."""
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """
        Prepare inputs for MoE computation using MORI dispatch.
        
        Returns a tuple of:
        - quantized + dispatched a1
        - Optional quantized + dispatched a1_scales
        - Optional ExpertTokensMetadata
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weights
        """
        assert not apply_router_weight_on_input, (
            "MORI does not support apply_router_weight_on_input=True"
        )
        
        scale = None
        if self.use_fp8_dispatch:
            from aiter import QuantType, get_hip_quant
            
            if quant_config.is_block_quantized:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())

        # Call MORI dispatch
        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, 
            expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        """
        Finalize MoE output using MORI combine.
        
        Combines expert outputs from all ranks back to the original
        token ordering using MORI's optimized combine kernel.
        """
        num_token = output.shape[0]
        result = self.mori_op.combine(
            fused_expert_output,
            None,  # No scale for combine
            topk_ids,
        )[0]
        output.copy_(result[:num_token])
