# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MORI buffer initialization utilities for Expert Parallelism.

This module provides utilities for creating and managing MORI EP
dispatch/combine operations in MoE layers.

Reference: https://github.com/ROCm/mori
"""
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


def create_mori_ep_op(config: MoriEpConfig) -> "EpDispatchCombineOp":
    """
    Create MORI EP dispatch/combine operator.

    Args:
        config: MoriEpConfig with all required parameters.

    Returns:
        EpDispatchCombineOp: MORI EP operator handle.

    Raises:
        AssertionError: If MORI-EP is not installed.
    """
    assert MORI_EP_AVAILABLE, (
        "MORI-EP not installed. Install from https://github.com/ROCm/mori"
    )

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

    logger.info(
        "Created MORI EP operator: world_size=%d, rank=%d, max_tokens=%d, "
        "hidden_dim=%d, topk=%d, num_experts=%d, kernel_type=%s",
        config.world_size,
        config.rank,
        config.max_num_tokens,
        config.hidden_dim,
        config.topk,
        config.num_experts,
        config.kernel_type,
    )

    return op


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
