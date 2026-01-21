# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MORI-EP Prepare and Finalize implementation for Expert Parallelism.

This module provides the MoriPrepareAndFinalize class that uses MORI-EP
dispatch/combine APIs for All-to-All communication in MoE layers.
Designed to pair with AiterExperts for maximum AMD performance on MI300X.

MORI-EP supports:
- EP8, EP16, EP32 configurations
- FP8 dispatch + BF16 combine (Strategy A)
- BF16 dispatch + post-dispatch quant (Strategy B)
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
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

# DBO (Disaggregated Batched Operations) support for microbatching
try:
    from vllm.v1.worker.ubatching import (
        dbo_current_ubatch_id,
        dbo_enabled,
    )
    DBO_AVAILABLE = True
except ImportError:
    DBO_AVAILABLE = False

    def dbo_current_ubatch_id() -> int:
        return 0

    def dbo_enabled() -> bool:
        return False


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

    Quantization Strategies:
    - Strategy A (FP8 dispatch): Quantize before dispatch for 2x bandwidth savings
    - Strategy B (BF16 dispatch): Dispatch BF16, quantize after receive

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
        num_experts: int,
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
            num_experts: Total number of experts globally.
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
        self.num_experts = num_experts
        self.dp_size = dp_size

        # Use environment variable if not explicitly set
        if use_fp8_dispatch is None:
            use_fp8_dispatch = envs.VLLM_MORI_EP_USE_FP8_DISPATCH
        self.use_fp8_dispatch = use_fp8_dispatch

        # Handle storage for DBO microbatching
        # Under DBO microbatching we must track one handle per
        # micro-batch to avoid races between threads.
        self.handles: list[Any] = [None, None]

        # Store dispatch metadata for combine
        self._dispatch_metadata: list[dict[str, Any]] = [{}, {}]
        
        # Deduplication state (set during _receiver, used during _finalize_impl)
        self._dedup_inverse_indices: torch.Tensor | None = None
        self._dedup_num_unique: int | None = None

        logger.info(
            "Initialized MoriPrepareAndFinalize: "
            "ep_size=%d, num_local_experts=%d, rank_expert_offset=%d, "
            "num_experts=%d, use_fp8_dispatch=%s",
            ep_size,
            num_local_experts,
            rank_expert_offset,
            num_experts,
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

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
    ) -> Callable[[], mk.PrepareResultType]:
        """
        Internal dispatch implementation.

        Args:
            tokens: Input tokens (may be FP8 quantized or BF16)
            token_scales: Scales if tokens are FP8 quantized
            topk_ids: Selected expert IDs (global)
            topk_weights: Router weights
            num_experts: Total number of experts
            a1_scale: Activation scale for post-dispatch quantization
            quant_config: Quantization configuration

        Returns:
            Callable that returns PrepareResultType when invoked
        """
        has_scales = token_scales is not None

        # MORI dispatch - send tokens to expert owners
        # dispatch(input, weights, scales, indices)
        # Returns: (recv_x, recv_weights, recv_scale, recv_topk_ids, recv_src_pos)
        dispatch_result = self.ep_op.dispatch(
            input=tokens,
            weights=topk_weights,
            scales=token_scales,
            indices=topk_ids.to(torch.int32),
        )

        # Record the handle/metadata for this ubatch
        ubatch_idx = dbo_current_ubatch_id()
        self._dispatch_metadata[ubatch_idx] = {
            "dispatch_result": dispatch_result,
            "original_topk_ids": topk_ids,
            "original_topk_weights": topk_weights,
        }

        return lambda: self._receiver(
            dispatch_result=dispatch_result,
            has_scales=has_scales,
            num_experts=num_experts,
            a1_scale=a1_scale,
            quant_config=quant_config,
        )

    def _receiver(
        self,
        dispatch_result: tuple,
        has_scales: bool,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """
        Process dispatch results and prepare for expert computation.

        This handles:
        1. Unpacking dispatch results
        2. Expert ID remapping (global → local with -1 handling)
        3. Post-dispatch quantization (Strategy B)
        4. ExpertTokensMetadata creation
        """
        # Unpack dispatch result
        # Based on mori/python/mori/ops/dispatch_combine.py:
        # Returns tuple from launch_dispatch:
        #   (out_tokens, out_weights, out_scales, out_indices, total_recv_token_num)
        #
        # MORI returns FIXED-SIZE buffers [max_num_tokens, ...] (e.g., [8192, 7168])
        # Only positions 0..total_recv_tokens-1 contain valid data!
        #
        # [TRIED] Slicing with total_recv_tokens.item() - fails during CUDA graph
        #         capture because .item() transfers GPU->CPU
        #
        # [TRIED] Slicing with batch*topk upper bound - CAUSES GARBAGE OUTPUT!
        #         The upper bound can exceed actual_recv_tokens, including
        #         uninitialized buffer data in expert computation.
        #
        # SOLUTION: Use GPU tensor masking! We can compare indices to
        # total_recv_tokens (a GPU scalar tensor) without .item().
        # This is CUDA-graph safe because all operations stay on GPU.
        recv_x = dispatch_result[0]
        recv_weights = dispatch_result[1] if len(dispatch_result) > 1 else None
        recv_scale = dispatch_result[2] if len(dispatch_result) > 2 else None
        recv_topk_ids = dispatch_result[3] if len(dispatch_result) > 3 else None
        total_recv_tokens = dispatch_result[4] if len(dispatch_result) > 4 else None

        if has_scales:
            expert_x = recv_x
            expert_x_scale = recv_scale
        else:
            expert_x = recv_x
            expert_x_scale = None

        # CRITICAL: Slice buffers to only valid tokens!
        # MORI returns fixed-size buffers [max_num_tokens, ...] but only
        # positions 0..total_recv_tokens-1 contain valid data.
        # Passing garbage data to expert computation corrupts output.
        #
        # NOTE: This uses .item() which breaks CUDA graph capture.
        # For CUDA graph support, we'll need a different approach.
        num_valid = None
        if total_recv_tokens is not None:
            num_valid = int(total_recv_tokens.item())
            if num_valid < expert_x.shape[0]:
                expert_x = expert_x[:num_valid]
                if expert_x_scale is not None:
                    expert_x_scale = expert_x_scale[:num_valid]
                if recv_weights is not None:
                    recv_weights = recv_weights[:num_valid]
                if recv_topk_ids is not None:
                    recv_topk_ids = recv_topk_ids[:num_valid]
        
        # CRITICAL FIX: Deduplicate received entries by source token!
        #
        # MORI dispatches per (token, expert) pair - so if token T has 2 local
        # experts, rank R receives 2 entries for T (both with same topk_ids).
        # If we process each entry with AITER, each entry computes ALL local
        # experts, causing 2x over-computation and garbage output.
        #
        # Solution: Keep only unique tokens, process once, then expand for combine.
        src_token_pos = self.ep_op.get_dispatch_src_token_pos()
        if src_token_pos is not None and src_token_pos.numel() > 0:
            # Find unique source positions and their indices
            unique_pos, inverse_indices = torch.unique(
                src_token_pos, return_inverse=True
            )
            num_unique = unique_pos.numel()
            
            import os
            if os.environ.get("VLLM_MORI_DEBUG"):
                ep_rank = self.rank_expert_offset // self.num_local_experts
                print(f"[MORI DEDUP] ep_rank={ep_rank}, total_recv={num_valid}, unique_tokens={num_unique}")
            
            # Get indices of first occurrence of each unique token
            # (arange creates indices 0..N-1, then scatter_min finds first occurrence)
            first_indices = torch.empty(num_unique, dtype=torch.long, device=src_token_pos.device)
            first_indices.fill_(src_token_pos.numel())  # Initialize with max
            arange = torch.arange(src_token_pos.numel(), device=src_token_pos.device)
            first_indices.scatter_reduce_(0, inverse_indices, arange, reduce='amin')
            
            # Keep only first occurrence of each unique token
            expert_x = expert_x[first_indices]
            if expert_x_scale is not None:
                expert_x_scale = expert_x_scale[first_indices]
            if recv_weights is not None:
                recv_weights = recv_weights[first_indices]
            if recv_topk_ids is not None:
                recv_topk_ids = recv_topk_ids[first_indices]
            
            # Store info needed to expand results back for combine
            self._dedup_inverse_indices = inverse_indices
            self._dedup_num_unique = num_unique
        else:
            self._dedup_inverse_indices = None
            self._dedup_num_unique = None

        # Expert ID handling: Convert GLOBAL IDs to LOCAL IDs
        #
        # MORI dispatch returns GLOBAL expert IDs (0-255), same as router output.
        # But after MORI dispatch, each rank ONLY has tokens for its local experts.
        # We need LOCAL IDs (0-31) for AITER when expert_map=None.
        #
        # Conversion: local_id = global_id - rank_expert_offset
        # Example: Rank 2 (offset=64), global ID 70 → local ID 6
        #
        # IMPORTANT: recv_topk_ids has shape [N_recv, topk] with ALL original
        # expert IDs. But after MORI routing, only IDs for THIS rank are valid.
        # We convert all IDs to local and let AITER use expert_map to filter.
        if recv_topk_ids is not None:
            # Debug: print shapes and values
            import os
            if os.environ.get("VLLM_MORI_DEBUG"):
                ep_rank = self.rank_expert_offset // self.num_local_experts
                print(f"[MORI DEBUG] ep_rank={ep_rank}, recv_topk_ids shape={recv_topk_ids.shape}")
                if recv_topk_ids.numel() > 0:
                    print(f"[MORI DEBUG] recv_topk_ids[:3]={recv_topk_ids[:3].tolist()}")
                print(f"[MORI DEBUG] num_valid={num_valid if 'num_valid' in dir() else 'N/A'}")
                print(f"[MORI DEBUG] rank_expert_offset={self.rank_expert_offset}, num_local_experts={self.num_local_experts}")
            
            # Keep GLOBAL IDs - let expert_map handle the conversion
            # MORI copies full topk_ids (all 8 experts) for each received token.
            # expert_map[global_id] = local_id (or -1 if not on this rank)
            # AITER will filter out experts where expert_map gives -1
            expert_topk_ids = recv_topk_ids.to(torch.int64)
            
            # Debug: Show expert_x shape to understand received data
            if os.environ.get("VLLM_MORI_DEBUG"):
                print(f"[MORI DEBUG] expert_x shape after slice={expert_x.shape}")
                # Check a sample of topk_ids to see local expert coverage
                if recv_topk_ids.numel() > 0:
                    sample_ids = recv_topk_ids[0].tolist()
                    local_experts = [eid for eid in sample_ids 
                                     if self.rank_expert_offset <= eid < self.rank_expert_offset + self.num_local_experts]
                    print(f"[MORI DEBUG] Sample topk_ids={sample_ids}, local_experts={local_experts}")
        else:
            expert_topk_ids = None

        # Strategy B: BF16 dispatch, quantize after receive
        # MORI kernels support block-quantized dispatch (Strategy A)
        # For non-block quantization, we dispatch BF16 and quantize here
        if not quant_config.is_block_quantized:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                )

        # Create ExpertTokensMetadata from MORI's token distribution
        # AITER computes this internally, so we pass None
        # If we had expert_num_tokens_per_expert_list from MORI, we'd use:
        # expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
        #     expert_num_tokens_per_expert_list, device=expert_x.device
        # )
        expert_tokens_meta = None

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            recv_weights,
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
        Async prepare: Dispatch tokens to GPUs that own selected experts.

        WITH MORI: expert_map is NOT used - MORI handles routing internally
        WITHOUT MORI: expert_map would be used to remap global→local expert IDs

        Args:
            a1: [M, H] input activations
            topk_weights: [M, K] router weights
            topk_ids: [M, K] selected expert IDs (global)
            num_experts: Total experts (e.g., 256)
            expert_map: Not used with MORI dispatch (MORI handles routing)
            apply_router_weight_on_input: Apply weights before dispatch
            quant_config: Quantization configuration

        Returns:
            Callable that returns PrepareResultType when invoked
        """
        # Step 1: Optional weight application (for topk=1 models)
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Step 2: Quantization strategy selection
        if quant_config.is_block_quantized and self.use_fp8_dispatch:
            # Strategy A: FP8 dispatch - Quantize before dispatch for 2x BW savings
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            # Strategy B: BF16 dispatch - Dispatch BF16, quantize after receive
            a1q = a1
            a1q_scale = None
            a1_post_scale = quant_config.a1_scale

        # Step 3: Execute dispatch
        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
        )

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
        """Synchronous prepare - calls async version and waits."""
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        return receiver()

    # ─────────────────────────────────────────────────────────────────────────
    # FINALIZE: Combine results back to original token owners
    # ─────────────────────────────────────────────────────────────────────────

    def _finalize_impl(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        """
        Internal finalize implementation.

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
            weight_and_reduce_impl: Weight/reduce implementation
            do_async: Whether to return async callable

        Returns:
            Callable for async mode, None for sync mode
        """
        ubatch_idx = dbo_current_ubatch_id()

        # Retrieve ORIGINAL topk_ids from dispatch metadata
        # CRITICAL: Combine needs the ORIGINAL indices [M, 8] (this rank's tokens)
        # to know what results to receive, NOT the received indices [N_recv, 8]!
        dispatch_meta = self._dispatch_metadata[ubatch_idx]
        original_topk_ids = dispatch_meta.get("original_topk_ids", topk_ids)
        
        # Debug: print shapes to understand the data flow
        import os
        if os.environ.get("VLLM_MORI_DEBUG"):
            print(f"[MORI COMBINE DEBUG] fused_expert_output shape={fused_expert_output.shape}")
            print(f"[MORI COMBINE DEBUG] original_topk_ids shape={original_topk_ids.shape}")
            print(f"[MORI COMBINE DEBUG] topk_ids (received) shape={topk_ids.shape}")
            print(f"[MORI COMBINE DEBUG] output shape={output.shape}")

        # fused_expert_output can have 0 tokens - This happens when none of the
        # tokens from the all2all reach this EP rank.
        if fused_expert_output.numel() != 0:
            # Apply weights before combine if using delegate
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            
            # CRITICAL: Expand deduplicated results back to full size for combine
            # We processed unique tokens only, but combine expects one result per
            # dispatched entry (including duplicates).
            if self._dedup_inverse_indices is not None:
                import os
                if os.environ.get("VLLM_MORI_DEBUG"):
                    print(f"[MORI EXPAND] Expanding {fused_expert_output.shape[0]} -> {self._dedup_inverse_indices.numel()}")
                fused_expert_output = fused_expert_output[self._dedup_inverse_indices]

        # MORI combine expects BF16
        assert fused_expert_output.dtype == torch.bfloat16, (
            f"Expected fused_expert_output bfloat16, got {fused_expert_output.dtype}"
        )

        # MORI combine - returns results to original token owners
        # combine(input, weights, indices, block_num, warp_per_block, call_reset)
        # Returns: (output, output_scale)
        #
        # NOTE: We pass weights=None because AITER fused_moe already applies
        # topk_weights during expert computation. Passing weights here would
        # cause MORI to accumulate them unnecessarily (wasted bandwidth).
        #
        # CRITICAL: Use ORIGINAL topk_ids (this rank's tokens), NOT received!
        # Combine uses original indices to:
        # 1. Route expert results BACK to original token owners
        # 2. Know what results THIS rank expects to receive
        # - fused_expert_output: [N_recv, H] - expert outputs for received tokens
        # - original_topk_ids: [M, K] - THIS rank's original tokens' expert choices
        combine_result = self.ep_op.combine(
            input=fused_expert_output,
            weights=None,  # AITER already applied weights
            indices=original_topk_ids.to(torch.int32),
            call_reset=True,  # Reset for next iteration
        )

        combined_x = combine_result[0]

        # MORI combine returns a fixed-size buffer [max_num_tokens, hidden_dim]
        # but the actual batch may be smaller. Slice to match output shape.
        num_tokens = output.shape[0]
        if combined_x.shape[0] != num_tokens:
            combined_x = combined_x[:num_tokens]

        # Clear dispatch metadata for this ubatch
        self._dispatch_metadata[ubatch_idx] = {}

        if do_async:
            def _receiver():
                # Respect inplace outputs
                output.copy_(combined_x, non_blocking=True)

            return _receiver
        else:
            # Synchronous: copy immediately
            output.copy_(combined_x, non_blocking=True)
            return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        """Async finalize - returns callable that completes finalization."""
        receiver = self._finalize_impl(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        """Synchronous finalize - completes immediately."""
        self._finalize_impl(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )
