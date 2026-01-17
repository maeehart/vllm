# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MORI (Modular RDMA Interface) integration for vLLM MoE expert parallelism.

MORI provides high-performance dispatch/combine kernels for MoE expert
parallelism using RDMA and GPU-direct communication on AMD ROCm.

Based on the reference implementation from:
https://github.com/alexsun07/vllm/tree/mori_ep
"""

from typing import Optional, Tuple

import torch
from torch.library import Library

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


# ============================================================================
# torch.library registration for CUDA graph compatibility
# ============================================================================
# 
# MORI dispatch/combine ops need to be registered with torch.library to be
# compatible with CUDA graphs and torch.compile. This allows the ops to be
# traced and captured properly.
#
# The key requirements for CUDA graph compatibility:
# 1. Fixed output tensor shapes (use max_tokens_per_rank for sizing)
# 2. No dynamic memory allocation during forward pass
# 3. Proper fake tensor implementations for tracing

# Create a library for MORI ops
mori_lib = Library("mori", "FRAGMENT")

# Global registry to store MORI op handles (needed for dispatch/combine pairing)
_mori_op_registry: dict[int, "mori.ops.EpDispatchCombineOp"] = {}


def _register_mori_op(op_id: int, mori_op) -> None:
    """Register a MORI op handle for later retrieval."""
    _mori_op_registry[op_id] = mori_op


def _get_mori_op(op_id: int):
    """Get a registered MORI op handle."""
    return _mori_op_registry.get(op_id)


if MORI_AVAILABLE:
    # Register MORI dispatch op
    def mori_dispatch_impl(
        mori_op_id: int,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: Optional[torch.Tensor],
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], 
               torch.Tensor, torch.Tensor]:
        """
        MORI dispatch implementation.
        
        Args:
            mori_op_id: ID to look up the MORI op handle
            input: Input hidden states [num_tokens, hidden_dim]
            weights: Router weights [num_tokens, topk]
            scales: Optional FP8 scales
            indices: Expert indices [num_tokens, topk]
            
        Returns:
            Tuple of (dispatched_input, dispatched_weights, dispatched_scales,
                     dispatched_indices, recv_token_counts)
        """
        mori_op = _get_mori_op(mori_op_id)
        return mori_op.dispatch(input, weights, scales, indices)

    def mori_dispatch_fake(
        mori_op_id: int,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: Optional[torch.Tensor],
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               torch.Tensor, torch.Tensor]:
        """
        Fake implementation for torch.compile tracing.
        
        Returns tensors with shapes based on max_tokens configuration.
        
        For CUDA graph compatibility, the output shapes must be deterministic.
        MORI dispatch expands tokens to (max_tokens * world_size, hidden_dim)
        where max_tokens is configured via max_num_inp_token_per_rank.
        
        The fake implementation returns tensors matching the input shape since
        the actual expanded size depends on runtime configuration. When
        use_cudagraph_compatible=True, the caller should ensure consistent
        batch sizes (e.g., via cudagraph_capture_sizes padding).
        """
        num_tokens, hidden_dim = input.shape
        topk = indices.shape[1]
        
        # For CUDA graph compatibility, we need fixed shapes
        # The dispatch output expands tokens based on EP world size
        # For tracing, we use input shape as the output shape
        # The actual runtime will handle the expansion
        dispatched_input = input.new_empty(input.shape)
        dispatched_weights = weights.new_empty(weights.shape)
        dispatched_scales = scales.new_empty(scales.shape) if scales is not None else None
        dispatched_indices = indices.new_empty(indices.shape)
        # recv_token_counts: number of tokens received per local expert
        # Shape depends on num_local_experts which we don't have here
        # Use a placeholder that will be refined at runtime
        recv_token_counts = indices.new_empty((indices.shape[0],), dtype=torch.int32)
        
        return (dispatched_input, dispatched_weights, dispatched_scales,
                dispatched_indices, recv_token_counts)

    # Register dispatch op
    mori_lib.define(
        "dispatch(int mori_op_id, Tensor input, Tensor weights, "
        "Tensor? scales, Tensor indices) -> "
        "(Tensor, Tensor, Tensor?, Tensor, Tensor)"
    )
    mori_lib.impl("dispatch", mori_dispatch_impl, dispatch_key="CUDA")
    mori_lib._register_fake("dispatch", mori_dispatch_fake)

    # Register MORI combine op
    def mori_combine_impl(
        mori_op_id: int,
        expert_output: torch.Tensor,
        scales: Optional[torch.Tensor],
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        MORI combine implementation.
        
        Args:
            mori_op_id: ID to look up the MORI op handle
            expert_output: Expert computation output
            scales: Optional scales (not used in combine)
            indices: Expert indices for routing back
            
        Returns:
            Combined output tensor
        """
        mori_op = _get_mori_op(mori_op_id)
        result = mori_op.combine(expert_output, scales, indices)
        return result[0]

    def mori_combine_fake(
        mori_op_id: int,
        expert_output: torch.Tensor,
        scales: Optional[torch.Tensor],
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Fake implementation for torch.compile tracing."""
        # Combine reduces back to original token count
        # For fake impl, return same shape as expert_output
        return expert_output.new_empty(expert_output.shape)

    # Register combine op
    mori_lib.define(
        "combine(int mori_op_id, Tensor expert_output, "
        "Tensor? scales, Tensor indices) -> Tensor"
    )
    mori_lib.impl("combine", mori_combine_impl, dispatch_key="CUDA")
    mori_lib._register_fake("combine", mori_combine_fake)


# Global counter for MORI op IDs
_mori_op_id_counter = 0


def _get_next_mori_op_id() -> int:
    """Get a unique ID for a MORI op instance."""
    global _mori_op_id_counter
    _mori_op_id_counter += 1
    return _mori_op_id_counter


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MORI dispatch/combine kernels.
    
    MORI provides optimized all-to-all communication for MoE expert
    parallelism with support for:
    - Intra-node XGMI communication
    - Inter-node RDMA communication
    - FP8 dispatch for reduced bandwidth
    - BF16 combine for accuracy
    
    For CUDA graph compatibility, this class registers MORI ops with
    torch.library and uses fixed-size output tensors based on
    max_tokens_per_rank.
    """

    def __init__(
        self,
        mori_op,  # mori.ops.EpDispatchCombineOp
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        use_cudagraph_compatible: bool = False,
        fixed_output_size: Optional[int] = None,
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
        self.use_cudagraph_compatible = use_cudagraph_compatible
        # For CUDA graph compatibility: truncate dispatch output to this size
        # Set to batch_size * topk * world_size for decode with fixed batch sizes
        self.fixed_output_size = fixed_output_size
        
        # Register this op instance for torch.library ops
        self.mori_op_id = _get_next_mori_op_id()
        _register_mori_op(self.mori_op_id, mori_op)

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

        # Call MORI dispatch - use registered op for CUDA graph compatibility
        if self.use_cudagraph_compatible:
            (
                dispatch_a1,
                dispatch_weights,
                dispatch_scale,
                dispatch_ids,
                dispatch_recv_token_num,
            ) = torch.ops.mori.dispatch(
                self.mori_op_id, a1, topk_weights, scale, topk_ids
            )
        else:
            (
                dispatch_a1,
                dispatch_weights,
                dispatch_scale,
                dispatch_ids,
                dispatch_recv_token_num,
            ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        # For CUDA graph compatibility, truncate outputs to fixed size
        # MORI dispatch expands to (max_tokens * world_size, hidden_dim)
        # but for decode with fixed batch sizes, we only need
        # (batch_size * topk * world_size) tokens
        # This optimization is similar to ATOM's graph_bs handling
        if self.use_cudagraph_compatible and self.fixed_output_size is not None:
            fixed_size = self.fixed_output_size
            if fixed_size < dispatch_a1.shape[0]:
                dispatch_a1 = dispatch_a1[:fixed_size]
                dispatch_ids = dispatch_ids[:fixed_size]
                dispatch_weights = dispatch_weights[:fixed_size]
                if dispatch_scale is not None:
                    dispatch_scale = dispatch_scale[:fixed_size]

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
        
        # Call MORI combine - use registered op for CUDA graph compatibility
        if self.use_cudagraph_compatible:
            result = torch.ops.mori.combine(
                self.mori_op_id,
                fused_expert_output,
                None,  # No scale for combine
                topk_ids,
            )
        else:
            result = self.mori_op.combine(
                fused_expert_output,
                None,  # No scale for combine
                topk_ids,
            )[0]
        
        output.copy_(result[:num_token])
