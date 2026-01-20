# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MORI buffer initialization utilities for Expert Parallelism.

This module provides utilities for creating and managing MORI EP
dispatch/combine operations in MoE layers.

Reference: https://github.com/ROCm/mori
"""
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from mori.ops import EpDispatchCombineOp

logger = init_logger(__name__)

# Try to import MORI
try:
    from mori.ops import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp as _EpDispatchCombineOp,
    )
    from mori import shmem

    MORI_EP_AVAILABLE = True
except ImportError:
    MORI_EP_AVAILABLE = False
    EpDispatchCombineConfig = None  # type: ignore
    EpDispatchCombineKernelType = None  # type: ignore
    _EpDispatchCombineOp = None  # type: ignore
    shmem = None  # type: ignore

# Track if shmem has been initialized
_MORI_SHMEM_INITIALIZED = False
_MORI_SHMEM_LOCK = threading.Lock()

# Global cache for MORI EP operators - shared across all MoE layers
# Key is a tuple of (rank, world_size, hidden_dim, max_num_tokens, topk, kernel_type)
_MORI_EP_OP_CACHE: dict[tuple, "_EpDispatchCombineOp"] = {}
_MORI_EP_OP_CACHE_LOCK = threading.Lock()


def is_mori_ep_available() -> bool:
    """Check if MORI-EP is available."""
    return MORI_EP_AVAILABLE


@dataclass
class MoriEpConfig:
    """Configuration for MORI EP dispatch/combine operations."""

    # Core configuration
    rank: int
    world_size: int  # EP size
    hidden_dim: int
    max_num_tokens: int
    num_experts: int
    topk: int

    # Data types
    dtype: torch.dtype = torch.bfloat16
    use_fp8_dispatch: bool = False

    # Kernel type (IntraNode for XGMI, InterNode for RDMA)
    kernel_type: str = "IntraNode"

    # Performance tuning
    gpu_per_node: int = 8
    warp_num_per_block: int = 8
    block_num: int = 80
    rdma_block_num: int = 0
    num_qp_per_pe: int = 1

    @property
    def num_experts_per_rank(self) -> int:
        """Number of experts per EP rank."""
        return self.num_experts // self.world_size

    @property
    def scale_dim(self) -> int:
        """Scale dimension for FP8 (hidden_dim // 128)."""
        return self.hidden_dim // 128 if self.use_fp8_dispatch else 0

    @property
    def scale_type_size(self) -> int:
        """Size of scale type in bytes."""
        return 4 if self.use_fp8_dispatch else 0  # float32 scales

    @property
    def max_token_type_size(self) -> int:
        """Size of token data type in bytes."""
        if self.dtype == torch.float8_e4m3fn:
            return 1
        elif self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            return 2
        elif self.dtype == torch.float32:
            return 4
        else:
            return 2  # Default to BF16


def get_kernel_type(kernel_type_str: str) -> Any:
    """Convert kernel type string to MORI enum."""
    if not MORI_EP_AVAILABLE:
        return None

    mapping = {
        "IntraNode": EpDispatchCombineKernelType.IntraNode,
        "InterNode": EpDispatchCombineKernelType.InterNode,
        "InterNodeV1": EpDispatchCombineKernelType.InterNodeV1,
        "InterNodeV1LL": EpDispatchCombineKernelType.InterNodeV1LL,
    }
    return mapping.get(kernel_type_str, EpDispatchCombineKernelType.IntraNode)


def _ensure_mori_shmem_initialized() -> None:
    """
    Ensure MORI shared memory is initialized.
    
    This must be called before creating any MORI EP operators.
    Initializes MORI's shared memory layer using a unique ID broadcast
    from rank 0 (similar to NCCL initialization pattern).
    
    Thread-safe - uses a lock to prevent multiple initializations.
    """
    global _MORI_SHMEM_INITIALIZED
    
    # Fast path - already initialized
    if _MORI_SHMEM_INITIALIZED:
        return
    
    with _MORI_SHMEM_LOCK:
        # Double-check after acquiring lock
        if _MORI_SHMEM_INITIALIZED:
            return
        
        if not MORI_EP_AVAILABLE:
            raise RuntimeError("MORI-EP not available for shmem initialization")
        
        import torch.distributed as dist
        from vllm.distributed import get_world_group
        
        if not dist.is_initialized():
            raise RuntimeError(
                "Cannot initialize MORI shmem: PyTorch distributed not initialized"
            )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        logger.info(
            "Initializing MORI shmem: rank=%d, world_size=%d", rank, world_size
        )
        
        try:
            # Get unique ID from rank 0 and broadcast to all ranks
            # (similar to NCCL initialization pattern)
            if rank == 0:
                unique_id = shmem.shmem_get_unique_id()
                logger.info("Rank 0 generated MORI unique ID (%d bytes)", len(unique_id))
            else:
                unique_id = bytes(128)  # Placeholder for broadcast
            
            # Broadcast unique ID using vLLM's world group (has CPU backend)
            world_group = get_world_group()
            unique_id_list = [list(unique_id)]
            world_group.broadcast_object_list(unique_id_list, src=0)
            unique_id = bytes(unique_id_list[0])
            
            logger.info("Rank %d received MORI unique ID, calling shmem_init_attr...", rank)
            
            # Initialize shmem with the unique ID
            result = shmem.shmem_init_attr(
                shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
                rank,
                world_size,
                unique_id,
            )
            
            if result != 0:
                raise RuntimeError(f"MORI shmem_init_attr returned error code: {result}")
            
            logger.info("MORI shmem initialized successfully on rank %d", rank)
            _MORI_SHMEM_INITIALIZED = True
            
        except Exception as e:
            raise RuntimeError(f"MORI shmem initialization failed on rank {rank}: {e}") from e


def _make_cache_key(config: MoriEpConfig) -> tuple:
    """
    Create a cache key for the MORI EP operator.
    
    Operators with the same configuration can be shared across MoE layers.
    """
    return (
        config.rank,
        config.world_size,
        config.hidden_dim,
        config.max_num_tokens,
        config.topk,
        config.num_experts_per_rank,
        config.kernel_type,
        config.dtype,
        config.use_fp8_dispatch,
    )


def create_mori_ep_op(config: MoriEpConfig) -> "EpDispatchCombineOp":
    """
    Create or retrieve a cached MORI EP dispatch/combine operator.
    
    IMPORTANT: MORI EP operators are shared across all MoE layers to avoid
    exhausting the symmetric heap memory. Operators with the same configuration
    are cached and reused.

    Args:
        config: MoriEpConfig with all required parameters.

    Returns:
        EpDispatchCombineOp: MORI EP operator handle (may be shared).

    Raises:
        RuntimeError: If MORI-EP is not installed or operator creation fails.
    """
    if not MORI_EP_AVAILABLE:
        raise RuntimeError(
            "MORI-EP not installed. Install from https://github.com/ROCm/mori"
        )

    # Ensure shmem is initialized before creating operator
    _ensure_mori_shmem_initialized()
    
    # Check cache first
    cache_key = _make_cache_key(config)
    
    with _MORI_EP_OP_CACHE_LOCK:
        if cache_key in _MORI_EP_OP_CACHE:
            logger.debug(
                "Reusing cached MORI EP operator for rank %d/%d "
                "(hidden=%d, max_tokens=%d, topk=%d)",
                config.rank,
                config.world_size,
                config.hidden_dim,
                config.max_num_tokens,
                config.topk,
            )
            return _MORI_EP_OP_CACHE[cache_key]
        
        # Not in cache - create new operator
        logger.info(
            "Creating MORI EP operator (will be shared across layers): "
            "world_size=%d, rank=%d, max_tokens=%d, hidden_dim=%d, topk=%d, "
            "num_experts_per_rank=%d, kernel_type=%s",
            config.world_size,
            config.rank,
            config.max_num_tokens,
            config.hidden_dim,
            config.topk,
            config.num_experts_per_rank,
            config.kernel_type,
        )

        try:
            # Create MORI EP configuration
            mori_config = EpDispatchCombineConfig(
                data_type=config.dtype,
                rank=config.rank,
                world_size=config.world_size,
                hidden_dim=config.hidden_dim,
                scale_dim=config.scale_dim,
                scale_type_size=config.scale_type_size,
                max_token_type_size=config.max_token_type_size,
                max_num_inp_token_per_rank=config.max_num_tokens,
                num_experts_per_rank=config.num_experts_per_rank,
                num_experts_per_token=config.topk,
                warp_num_per_block=config.warp_num_per_block,
                block_num=config.block_num,
                use_external_inp_buf=True,
                kernel_type=get_kernel_type(config.kernel_type),
                gpu_per_node=config.gpu_per_node,
                rdma_block_num=config.rdma_block_num,
                num_qp_per_pe=config.num_qp_per_pe,
            )

            # Create the operator
            op = _EpDispatchCombineOp(mori_config)
            
            # Cache it for reuse by other layers
            _MORI_EP_OP_CACHE[cache_key] = op

            logger.info(
                "Successfully created and cached MORI EP operator for rank %d/%d",
                config.rank,
                config.world_size,
            )

            return op
        except Exception as e:
            raise RuntimeError(
                f"MORI EP operator creation failed on rank {config.rank}: {e}. "
                f"Config: world_size={config.world_size}, hidden_dim={config.hidden_dim}, "
                f"max_tokens={config.max_num_tokens}, kernel_type={config.kernel_type}"
            ) from e


def init_mori_shmem_from_process_group(group_name: str = "default") -> int:
    """
    Initialize MORI shmem from PyTorch process group.

    This should be called once during model initialization.

    Args:
        group_name: Name of the PyTorch distributed process group.

    Returns:
        int: Status code (0 for success).
    """
    assert MORI_EP_AVAILABLE, "MORI-EP not installed"
    return shmem.shmem_torch_process_group_init(group_name)


def mori_shmem_barrier() -> None:
    """Global barrier synchronization for MORI shmem."""
    assert MORI_EP_AVAILABLE, "MORI-EP not installed"
    shmem.shmem_barrier_all()


def get_mori_ep_config_for_model(
    model_config,
    parallel_config,
    max_num_tokens: int,
) -> MoriEpConfig:
    """
    Get MORI EP configuration parameters for a given model.

    Args:
        model_config: vLLM model configuration.
        parallel_config: vLLM parallel configuration.
        max_num_tokens: Maximum number of tokens per batch.

    Returns:
        MoriEpConfig: Configuration for create_mori_ep_op.
    """
    import torch.distributed as dist

    # Extract MoE configuration from model config
    hf_config = model_config.hf_config

    # Try to get MoE-specific parameters
    num_experts = getattr(hf_config, "n_routed_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_experts", 256)

    topk = getattr(hf_config, "num_experts_per_tok", None)
    if topk is None:
        topk = getattr(hf_config, "moe_top_k", 8)

    hidden_size = getattr(hf_config, "hidden_size", 7168)

    # Get EP size and rank
    ep_size = parallel_config.tensor_parallel_size
    rank = dist.get_rank() if dist.is_initialized() else 0

    return MoriEpConfig(
        rank=rank,
        world_size=ep_size,
        hidden_dim=hidden_size,
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        topk=topk,
    )


def compute_num_local_experts(
    num_experts: int,
    ep_size: int,
) -> int:
    """
    Compute the number of local experts for a given EP configuration.

    Args:
        num_experts: Total number of experts.
        ep_size: Number of EP ranks.

    Returns:
        int: Number of experts per rank.
    """
    assert num_experts % ep_size == 0, (
        f"Number of experts ({num_experts}) must be divisible by "
        f"EP size ({ep_size})"
    )
    return num_experts // ep_size


def compute_rank_expert_offset(
    ep_rank: int,
    num_local_experts: int,
) -> int:
    """
    Compute the starting global expert ID for a given rank.

    Args:
        ep_rank: Current EP rank.
        num_local_experts: Number of experts per rank.

    Returns:
        int: Starting global expert ID for this rank.
    """
    return ep_rank * num_local_experts
