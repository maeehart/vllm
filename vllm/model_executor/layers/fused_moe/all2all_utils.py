# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.distributed import (
    get_ep_group,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep, has_pplx

# Check for MORI/Smart routing availability on ROCm
if current_platform.is_rocm():
    from .mori_prepare_finalize import (
        MoriPrepareAndFinalize,
        is_mori_available,
    )
    from .smart_routing_prepare_finalize import (
        SmartRoutingPrepareAndFinalize,
    )
else:
    def is_mori_available():
        return False

if current_platform.is_cuda_alike():
    if has_pplx():
        from .pplx_prepare_finalize import (
            PplxPrepareAndFinalize,
            pplx_hidden_dim_scale_bytes,
        )
    if has_deep_ep():
        from .deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalize
        from .deepep_ll_prepare_finalize import (
            DEEPEP_QUANT_BLOCK_SHAPE,
            DeepEPLLPrepareAndFinalize,
        )


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs
        and all2all backend.
        Original hidden size otherwise.
    """
    if moe_parallel_config.use_deepep_ht_kernels:
        hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype
        )

    if moe_parallel_config.use_deepep_ll_kernels:
        hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size
        )

    return hidden_size


def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> FusedMoEPrepareAndFinalize | None:
    # Check for smart_routing and MORI with pure EP (uses ep_size > 1, not dp_size > 1)
    use_smart_routing = moe.moe_parallel_config.use_smart_routing_kernels
    use_mori = moe.moe_parallel_config.use_mori_kernels
    use_all2all = moe.moe_parallel_config.use_all2all_kernels
    
    print(f"[DEBUG] maybe_make_prepare_finalize: backend={moe.moe_parallel_config.all2all_backend}, "
          f"ep_size={moe.moe_parallel_config.ep_size}, dp_size={moe.moe_parallel_config.dp_size}, "
          f"use_ep={moe.moe_parallel_config.use_ep}, use_all2all_kernels={use_all2all}, "
          f"use_mori={use_mori}, use_smart_routing={use_smart_routing}")
    
    # For most backends, use_all2all_kernels requires dp_size > 1
    # For smart_routing and MORI, we use ep_size > 1 instead (pure EP)
    if not use_all2all and not use_smart_routing and not use_mori:
        print("[DEBUG] maybe_make_prepare_finalize: returning None (no all2all needed)")
        return None

    all2all_manager = get_ep_group().device_communicator.all2all_manager
    print(f"[DEBUG] maybe_make_prepare_finalize: all2all_manager={all2all_manager}")
    if all2all_manager is None:
        print(f"[DEBUG] all2all_manager is None! ep_size={moe.moe_parallel_config.ep_size}, "
              f"dp_size={moe.moe_parallel_config.dp_size}")
        return None

    prepare_finalize: FusedMoEPrepareAndFinalize | None = None

    # TODO(rob): update this as part of the MoE refactor.
    assert not moe.use_flashinfer_cutlass_kernels, (
        "Must be created in modelopt.py or fp8.py"
    )

    if moe.use_pplx_kernels:
        assert quant_config is not None

        hidden_dim_bytes, hidden_scale_bytes = pplx_hidden_dim_scale_bytes(
            moe.max_num_tokens,
            moe.hidden_dim,
            moe.in_dtype,
            quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )

        all_to_all_args = dict(
            max_num_tokens=moe.max_num_tokens,
            num_experts=moe.num_experts,
            experts_per_token=moe.experts_per_token,  # topk
            rank=all2all_manager.rank,
            world_size=all2all_manager.world_size,
            # dp_size actually means tp_size, bug in pplx kernels
            dp_size=all2all_manager.tp_group.world_size,
            hidden_dim=moe.hidden_dim,
            hidden_dim_bytes=hidden_dim_bytes,
            hidden_dim_scale_bytes=hidden_scale_bytes,
        )

        num_dispatchers = (
            all2all_manager.world_size // all2all_manager.tp_group.world_size
        )

        # Intranode pplx a2a takes a group name while internode does not.
        if not all2all_manager.internode:
            all_to_all_args["group_name"] = all2all_manager.cpu_group.group_name

        handle = all2all_manager.get_handle(all_to_all_args)

        prepare_finalize = PplxPrepareAndFinalize(
            handle,
            max_num_tokens=moe.max_num_tokens,
            num_local_experts=moe.num_local_experts,
            num_dispatchers=num_dispatchers,
        )
    elif moe.use_deepep_ht_kernels:
        assert moe.dp_size == all2all_manager.dp_world_size

        all_to_all_args = dict()
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = DeepEPHTPrepareAndFinalize(
            handle,
            num_dispatchers=all2all_manager.world_size,
            dp_size=all2all_manager.dp_world_size,
            rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
        )

    elif moe.use_deepep_ll_kernels:
        assert quant_config is not None
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_ep_ranks=all2all_manager.world_size,
            num_global_experts=moe.num_experts,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
        )

        prepare_finalize = DeepEPLLPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )

    elif moe.use_mori_kernels:
        # MORI dispatch/combine for ROCm expert parallelism
        # Based on: https://github.com/alexsun07/vllm/tree/mori_ep
        assert current_platform.is_rocm(), "MORI is only available on ROCm"
        assert is_mori_available(), (
            "MORI is required but not installed. "
            "Please install from https://github.com/ROCm/mori"
        )
        assert quant_config is not None
        
        # For PTPC (per token per channel) quant, the scale dim is 1
        # For 1x128 quant, the scale dim is hidden_dim // 128
        scale_dim = 1 if quant_config.is_per_act_token else moe.hidden_dim // 128
        
        # For single-node EP, gpu_per_node == ep_size (all EP GPUs on one node)
        # For multi-node, this would be the local EP size per node
        gpu_per_node = moe.moe_parallel_config.ep_size
        
        # Use max_buffer_tokens from config for proper buffer sizing during
        # profiling. This comes from scheduler_config.max_num_batched_tokens.
        # Falls back to max_num_tokens if not set.
        #
        # MORI memory requirements (per GPU):
        #   recv_buffer = max_tokens × ep_size × hidden_dim × dtype_size
        #   send_buffer = max_tokens × hidden_dim × dtype_size  
        #   overhead    = ~50% for scales, indices, weights
        #
        # For DeepSeek-R1 (hidden=7168, ep=8, bf16):
        #   4096 tokens  → ~0.8GB  (fits in default 2GB heap)
        #   8192 tokens  → ~1.6GB  (needs ~3GB heap)
        #   16384 tokens → ~3.2GB  (needs ~5GB heap)
        #
        # By default, we cap at 4096 tokens to fit in MORI's default 2GB heap.
        # For larger batches, increase heap and reduce vLLM's memory:
        #   export MORI_SHMEM_HEAP_SIZE=6G
        #   --gpu-memory-utilization 0.92
        import os
        mori_max_override = os.environ.get("VLLM_MORI_MAX_TOKENS")
        if mori_max_override:
            mori_max_tokens = int(mori_max_override)
        else:
            # Default: cap at 4096 to fit in default 2GB MORI heap
            mori_max_tokens = min(moe.max_buffer_tokens, 4096)
        
        all_to_all_args = dict(
            rank=all2all_manager.rank,
            num_ep_ranks=all2all_manager.world_size,
            quant_dtype=quant_config.quant_dtype,
            token_hidden_size=moe.hidden_dim,
            scale_dim=scale_dim,
            scale_type_size=torch.float32.itemsize,
            max_num_tokens_per_dp_rank=mori_max_tokens,
            input_dtype=moe.in_dtype,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
            num_experts_per_token=moe.experts_per_token,
            gpu_per_node=gpu_per_node,
        )
        handle = all2all_manager.get_handle(all_to_all_args)
        
        # Use FP8 dispatch to reduce data movement
        use_fp8_dispatch = (
            quant_config.is_per_act_token or quant_config.is_block_quantized
        )
        
        # Estimate MORI memory requirement
        mori_mem_gb = (mori_max_tokens * moe.moe_parallel_config.ep_size * 
                       moe.hidden_dim * 2 * 1.5) / 1e9  # 1.5x for overhead
        
        logger.info(
            "Creating MoriPrepareAndFinalize: max_tokens=%d, estimated_heap=%.1fGB. "
            "Ensure MORI_SHMEM_HEAP_SIZE >= %.0fG and --gpu-memory-utilization leaves room.",
            mori_max_tokens, mori_mem_gb, mori_mem_gb + 1
        )
        
        prepare_finalize = MoriPrepareAndFinalize(
            handle,
            max_tokens_per_rank=mori_max_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
        )

    elif use_smart_routing:
        # Smart routing: send TOKENS (not token-expert pairs) only to GPUs 
        # with their selected experts.
        #
        # This enables L2 cache benefits:
        # - Each GPU receives ~1/ep_size of total tokens (if balanced)
        # - Each GPU iterates through only local experts (32 instead of 256)
        # - Better cache utilization = higher throughput for large batches
        #
        # Key insight: Route tokens, not (token, expert) pairs.
        # Each token goes to each GPU AT MOST ONCE.
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        
        ep_group = get_ep_group().device_group
        
        logger.info(
            "Creating SmartRoutingPrepareAndFinalize: num_experts=%d, "
            "max_num_tokens=%d, hidden_dim=%d, ep_size=%d",
            moe.num_experts, moe.max_num_tokens, moe.hidden_dim,
            moe.moe_parallel_config.ep_size
        )
        
        prepare_finalize = SmartRoutingPrepareAndFinalize(
            ep_group=ep_group,
            num_experts=moe.num_experts,
            max_num_tokens=moe.max_num_tokens,
            hidden_dim=moe.hidden_dim,
            use_fp8_dispatch=False,  # TODO: add FP8 support
        )

    return prepare_finalize
