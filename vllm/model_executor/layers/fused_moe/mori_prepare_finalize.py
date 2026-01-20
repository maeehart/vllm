# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MORI-EP Prepare and Finalize implementation for Expert Parallelism.

This module provides the MoriPrepareAndFinalize class that uses MORI-EP
dispatch/combine APIs for All-to-All communication in MoE layers.
Designed to pair with AiterExperts for maximum AMD performance on MI300X.

MORI-EP supports:
- EP8, EP16, EP32 configurations
- FP8 dispatch + BF16 combine
- XGMI (intra-node) and RDMA (inter-node)

Reference: https://github.com/ROCm/mori
"""
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

if TYPE_CHECKING:
    from mori.ops import EpDispatchCombineOp

logger = init_logger(__name__)

# Try to import MORI-EP
try:
    from mori.ops import (
        EpDispatchCombineOp as _EpDispatchCombineOp,
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
    )

    MORI_EP_AVAILABLE = True
    logger.info("MORI-EP is available")
except ImportError:
    MORI_EP_AVAILABLE = False
    _EpDispatchCombineOp = None  # type: ignore
    EpDispatchCombineConfig = None  # type: ignore
    EpDispatchCombineKernelType = None  # type: ignore
    logger.warning(
        "MORI-EP is not available. Install from https://github.com/ROCm/mori"
    )


def is_mori_ep_available() -> bool:
    """Check if MORI-EP is available."""
    return MORI_EP_AVAILABLE


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Expert Parallelism prepare/finalize using MORI dispatch/combine.
    Designed to pair with AiterExperts for maximum AMD performance.

    MORI-EP provides:
    - Optimized dispatch/combine kernels for MoE token routing
    - FP8 dispatch + BF16 combine support
    - XGMI (intra-node) and RDMA (inter-node) paths
    - EP8, EP16, EP32 configurations

    Performance (from MORI benchmarks, 8× MI300X):
    - EP8 Dispatch: 307 GB/s (XGMI), 35µs latency (128 tokens)
    - EP8 Combine: 330 GB/s (XGMI), 47µs latency (128 tokens)
    """

    def __init__(
        self,
        ep_op: "EpDispatchCombineOp",
        num_local_experts: int,
        rank_expert_offset: int,
        ep_size: int,
        dp_size: int = 1,
        use_fp8_dispatch: bool | None = None,
    ):
        """
        Initialize MoriPrepareAndFinalize.

        Args:
            ep_op: MORI EpDispatchCombineOp for dispatch/combine operations.
            num_local_experts: Number of experts on this GPU
                (e.g., 32 for EP8 with 256 total experts).
            rank_expert_offset: Starting global expert ID for this rank
                (e.g., rank * num_local_experts).
            ep_size: Number of EP ranks (e.g., 8 for EP8).
            dp_size: Data parallel size (default: 1).
            use_fp8_dispatch: Whether to use FP8 quantization before dispatch
                for 2x bandwidth savings. If None, uses VLLM_MORI_EP_USE_FP8_DISPATCH.
        """
        super().__init__()
        assert MORI_EP_AVAILABLE, (
            "MORI-EP package not installed. "
            "Install from https://github.com/ROCm/mori"
        )

        self.ep_op = ep_op
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self.ep_size = ep_size
        self.dp_size = dp_size

        # Use environment variable if not explicitly set
        if use_fp8_dispatch is None:
            use_fp8_dispatch = envs.VLLM_MORI_EP_USE_FP8_DISPATCH
        self.use_fp8_dispatch = use_fp8_dispatch

        # Store dispatch output for combine
        self._dispatch_output: Any = None

        logger.info(
            "Initialized MoriPrepareAndFinalize: "
            "ep_size=%d, num_local_experts=%d, rank_expert_offset=%d, "
            "use_fp8_dispatch=%s",
            ep_size,
            num_local_experts,
            rank_expert_offset,
            use_fp8_dispatch,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # REQUIRED PROPERTIES
    # ─────────────────────────────────────────────────────────────────────────

    def output_is_reduced(self) -> bool:
        """MORI combine produces fully reduced output."""
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        """AITER uses Standard format [N, H], not BatchedExperts."""
        return mk.FusedMoEActivationFormat.Standard

    def num_dispatchers(self) -> int:
        """Return the number of EP ranks."""
        return self.ep_size

    def max_num_tokens_per_rank(self) -> int | None:
        """No fixed limit on tokens per rank."""
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        """MORI expects int32 for topk indices."""
        return torch.int32

    def supports_async(self) -> bool:
        """MORI supports async dispatch for compute/comm overlap."""
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # PREPARE: Dispatch tokens to expert owners
    # ─────────────────────────────────────────────────────────────────────────

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
        Dispatch tokens to GPUs that own the selected experts.

        Args:
            a1: [M, H] input activations
            topk_weights: [M, K] router weights
            topk_ids: [M, K] selected expert IDs (global)
            num_experts: Total experts (e.g., 256)
            expert_map: Not used with MORI dispatch
            apply_router_weight_on_input: Apply weights before dispatch
            quant_config: Quantization configuration

        Returns:
            PrepareResultType: (expert_x, a1q_scale, expert_tokens_meta,
                               expert_topk_ids, expert_topk_weights)

        Note: a1q_scale and expert_tokens_meta are None because AITER
        handles quantization and token counting internally.
        """
        # Step 1: Optional weight application (for topk=1 models)
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, "apply_router_weight_on_input only for topk=1"
            a1 = a1 * topk_weights.to(a1.dtype)

        # Step 2: Optional FP8 quantization before dispatch
        # MORI supports FP8 dispatch for 2x bandwidth savings
        a1_to_dispatch = a1
        dispatch_scale: torch.Tensor | None = None

        if self.use_fp8_dispatch and quant_config.is_block_quantized:
            # Block quantization: quantize before dispatch
            a1_to_dispatch, dispatch_scale = moe_kernel_quantize_input(
                a1,
                a1_scale=None,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )

        # Step 3: MORI dispatch - send tokens to expert owners
        # dispatch(input, weights, scales, indices, block_num, warp_per_block)
        # Returns: (recv_x, recv_weights, recv_x_scale, recv_topk_ids, recv_src_pos)
        dispatch_result = self.ep_op.dispatch(
            input=a1_to_dispatch,
            weights=topk_weights,
            scales=dispatch_scale,
            indices=topk_ids.to(torch.int32),
        )

        # Unpack dispatch result
        # Based on mori/python/mori/ops/dispatch_combine.py:
        # Returns tuple from launch_dispatch
        recv_x = dispatch_result[0]
        recv_weights = dispatch_result[1] if len(dispatch_result) > 1 else topk_weights
        recv_scale = dispatch_result[2] if len(dispatch_result) > 2 else None
        recv_topk_ids = dispatch_result[3] if len(dispatch_result) > 3 else topk_ids
        recv_src_pos = dispatch_result[4] if len(dispatch_result) > 4 else None

        # Store for combine
        self._dispatch_output = {
            "recv_src_pos": recv_src_pos,
            "original_topk_ids": topk_ids,
        }

        # Step 4: Remap global expert IDs to local
        # Global ID 64 on GPU 2 (EP=8) → Local ID 0
        expert_topk_ids = recv_topk_ids.to(torch.int64) - self.rank_expert_offset

        # Step 5: Return PrepareResultType
        # AITER expects: a1q_scale=None, expert_tokens_meta=None
        return (
            recv_x,  # Dispatched activations
            recv_scale,  # a1q_scale - may be None
            None,  # expert_tokens_meta - AITER computes internally
            expert_topk_ids,  # Local expert IDs
            recv_weights,  # Router weights
        )

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.ReceiverType:
        """
        Async version of prepare for compute/comm overlap.

        Returns a callable that when invoked returns the PrepareResultType.
        """
        # For now, use synchronous implementation
        # TODO: Implement true async when MORI provides async_dispatch API
        result = self.prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        return lambda: result

    # ─────────────────────────────────────────────────────────────────────────
    # FINALIZE: Combine results back to original token owners
    # ─────────────────────────────────────────────────────────────────────────

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
        Combine expert outputs back to original token owners.

        MORI combine handles:
        - All-to-All communication to return results
        - Weight application (if not applied on input)
        - Reduction across topk selections

        Args:
            output: [M, H] output buffer (written in-place)
            fused_expert_output: Expert computation result
            topk_weights: [M, K] original router weights
            topk_ids: [M, K] original expert IDs
            apply_router_weight_on_input: Whether weights were applied on input
            weight_and_reduce_impl: Weight/reduce implementation (delegated to MORI)
        """
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the MORI combine kernel."
        )

        # MORI combine - returns results to original token owners
        combine_weights = topk_weights
        if apply_router_weight_on_input:
            # Weights have already been applied
            combine_weights = torch.ones_like(topk_weights)

        # combine(input, weights, indices, block_num, warp_per_block, call_reset)
        # Returns: (output, output_scale)
        combine_result = self.ep_op.combine(
            input=fused_expert_output,
            weights=combine_weights,
            indices=topk_ids.to(torch.int32),
            call_reset=True,  # Reset for next iteration
        )

        # Copy result to output buffer
        combined_output = combine_result[0]
        output.copy_(combined_output)

        # Clear dispatch output after use
        self._dispatch_output = None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        """
        Async version of finalize for compute/comm overlap.

        Returns a callable that when invoked completes the finalization.
        """
        # For now, use synchronous implementation wrapped in a callable
        # TODO: Implement true async when MORI provides async_combine API
        def _receiver():
            self.finalize(
                output,
                fused_expert_output,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                weight_and_reduce_impl,
            )

        return _receiver
