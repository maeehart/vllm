# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CUDA graph-compatible All2All manager with pre-allocated padded buffers.

This module provides an All2All implementation that uses fixed-size buffers
to enable CUDA graph capture. The key insight (from ATOM) is:

1. All DP ranks pad their tensors to the same max size before communication
2. This allows CUDA graphs to capture the communication operations
3. The padding is removed after reduce_scatter

This is particularly important for MoE layers in prefill, where batch sizes
can vary significantly across DP ranks and iterations.
"""

import torch
from typing import Optional, Tuple

from vllm.distributed.device_communicators.all2all import All2AllManagerBase
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


class PaddedAgRsAll2AllManager(All2AllManagerBase):
    """
    CUDA graph-compatible All2All using padded all-gather and reduce-scatter.
    
    Key differences from AgRsAll2AllManager:
    1. Uses fixed-size buffers instead of variable-size all_gatherv
    2. Pre-allocates buffers at initialization for CUDA graph compatibility
    3. Pads all tensors to max_tokens_per_rank before communication
    """
    
    def __init__(
        self,
        cpu_group,
        max_tokens_per_rank: int,
        hidden_dim: int,
        num_experts: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(cpu_group)
        
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.dtype = dtype
        
        # Pre-allocate buffers for CUDA graph compatibility
        self._init_buffers()
        
        logger.info(
            "Initialized PaddedAgRsAll2AllManager: max_tokens=%d, hidden=%d, "
            "num_experts=%d, dp_size=%d",
            max_tokens_per_rank, hidden_dim, num_experts, self.dp_world_size
        )
    
    def _init_buffers(self):
        """Initialize pre-allocated communication buffers."""
        device = torch.cuda.current_device()
        total_tokens = self.max_tokens_per_rank * self.dp_world_size
        
        # Hidden states buffers
        self.hidden_padded = torch.zeros(
            (self.max_tokens_per_rank, self.hidden_dim),
            dtype=self.dtype,
            device=device,
        )
        self.hidden_gathered = torch.zeros(
            (total_tokens, self.hidden_dim),
            dtype=self.dtype,
            device=device,
        )
        
        # Router logits buffers  
        self.router_padded = torch.zeros(
            (self.max_tokens_per_rank, self.num_experts),
            dtype=self.dtype,
            device=device,
        )
        self.router_gathered = torch.zeros(
            (total_tokens, self.num_experts),
            dtype=self.dtype,
            device=device,
        )
        
        # Store original batch sizes for unpadding
        self._original_batch_size: Optional[int] = None
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch hidden states and router logits using padded all-gather.
        
        Args:
            hidden_states: (num_tokens, hidden_dim)
            router_logits: (num_tokens, num_experts)
            is_sequence_parallel: Whether using sequence parallelism
            extra_tensors: Not supported in padded mode
            
        Returns:
            Tuple of (gathered_hidden, gathered_router)
        """
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for PaddedAgRsAll2AllManager"
            )
        
        # Save original batch size for unpadding in combine()
        self._original_batch_size = hidden_states.shape[0]
        
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        
        # Pad hidden states to fixed size
        self.hidden_padded.zero_()
        self.hidden_padded[:self._original_batch_size].copy_(hidden_states)
        
        # Pad router logits to fixed size
        self.router_padded.zero_()
        self.router_padded[:self._original_batch_size].copy_(router_logits)
        
        # Fixed-size all-gather for hidden states
        dist_group.all_gather(
            self.hidden_gathered,
            self.hidden_padded,
            dim=0,
        )
        
        # Fixed-size all-gather for router logits
        dist_group.all_gather(
            self.router_gathered,
            self.router_padded,
            dim=0,
        )
        
        return self.hidden_gathered, self.router_gathered
    
    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """
        Combine hidden states using padded reduce-scatter.
        
        Args:
            hidden_states: (total_tokens, hidden_dim) from MoE computation
            is_sequence_parallel: Whether using sequence parallelism
            
        Returns:
            Output tensor of shape (original_batch_size, hidden_dim)
        """
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        
        # Fixed-size reduce-scatter
        scattered = dist_group.reduce_scatter(hidden_states, dim=0)
        
        # Remove padding - slice to original size
        if self._original_batch_size is not None:
            if scattered.shape[0] > self._original_batch_size:
                scattered = scattered[:self._original_batch_size].contiguous()
        
        return scattered
    
    def destroy(self):
        """Clean up buffers."""
        del self.hidden_padded
        del self.hidden_gathered
        del self.router_padded
        del self.router_gathered


def create_padded_all2all_manager(
    cpu_group,
    max_tokens_per_rank: int,
    hidden_dim: int,
    num_experts: int,
    dtype: torch.dtype = torch.bfloat16,
) -> PaddedAgRsAll2AllManager:
    """
    Factory function to create a PaddedAgRsAll2AllManager.
    
    Args:
        cpu_group: Process group for CPU coordination
        max_tokens_per_rank: Maximum tokens per DP rank (for CUDA graph compatibility)
        hidden_dim: Hidden dimension size
        num_experts: Number of MoE experts
        dtype: Data type for buffers
        
    Returns:
        Initialized PaddedAgRsAll2AllManager
    """
    return PaddedAgRsAll2AllManager(
        cpu_group=cpu_group,
        max_tokens_per_rank=max_tokens_per_rank,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        dtype=dtype,
    )



