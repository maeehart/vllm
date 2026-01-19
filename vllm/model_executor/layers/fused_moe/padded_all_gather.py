# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Pre-allocated padded buffers for MoE all_gather/reduce_scatter operations.

This module provides CUDA graph-compatible communication operations by:
1. Pre-allocating buffers at initialization time
2. Using fixed tensor sizes for all_gather operations
3. Avoiding dynamic memory allocation during forward pass

Key insight from ATOM:
- All DP ranks pad their tensors to the same max size before communication
- This allows CUDA graphs to capture the communication operations
- The padding is removed after reduce_scatter
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple

from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


class PaddedCommunicationBuffers:
    """
    Pre-allocated buffers for padded all_gather and reduce_scatter operations.
    
    This class manages fixed-size buffers that enable CUDA graph capture
    by avoiding dynamic tensor allocation during forward pass.
    """
    
    def __init__(
        self,
        max_tokens_per_rank: int,
        hidden_dim: int,
        dp_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initialize pre-allocated communication buffers.
        
        Args:
            max_tokens_per_rank: Maximum number of tokens any single DP rank can have
            hidden_dim: Hidden dimension of the model
            dp_size: Data parallel world size
            dtype: Data type for buffers
            device: Device to allocate buffers on
        """
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden_dim = hidden_dim
        self.dp_size = dp_size
        self.dtype = dtype
        self.device = device
        
        # Pre-allocate padded input buffer (for single rank's padded data)
        self.padded_input_buffer = torch.zeros(
            (max_tokens_per_rank, hidden_dim),
            dtype=dtype,
            device=device,
        )
        
        # Pre-allocate gathered output buffer (for all ranks' data)
        self.gathered_buffer = torch.zeros(
            (max_tokens_per_rank * dp_size, hidden_dim),
            dtype=dtype,
            device=device,
        )
        
        # Pre-allocate router logits buffers
        # Note: num_experts is set later when we know the model config
        self.router_padded_buffer: Optional[torch.Tensor] = None
        self.router_gathered_buffer: Optional[torch.Tensor] = None
        
        logger.debug(
            "Initialized PaddedCommunicationBuffers: max_tokens=%d, hidden=%d, dp_size=%d",
            max_tokens_per_rank, hidden_dim, dp_size
        )
    
    def init_router_buffers(self, num_experts: int):
        """Initialize router logits buffers once we know num_experts."""
        if self.router_padded_buffer is None:
            self.router_padded_buffer = torch.zeros(
                (self.max_tokens_per_rank, num_experts),
                dtype=self.dtype,
                device=self.device,
            )
            self.router_gathered_buffer = torch.zeros(
                (self.max_tokens_per_rank * self.dp_size, num_experts),
                dtype=self.dtype,
                device=self.device,
            )


def all_gather_with_fixed_padding(
    x: torch.Tensor,
    padded_size: int,
    padded_buffer: Optional[torch.Tensor] = None,
    gathered_buffer: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """
    All-gather with fixed padding size for CUDA graph compatibility.
    
    Args:
        x: Input tensor of shape (num_tokens, hidden_dim)
        padded_size: Fixed size to pad to (per rank)
        padded_buffer: Optional pre-allocated buffer for padded input
        gathered_buffer: Optional pre-allocated buffer for gathered output
    
    Returns:
        Tuple of (gathered tensor, original batch size)
    """
    dp_group = get_dp_group()
    original_batch_size = x.shape[0]
    hidden_dim = x.shape[1]
    dp_size = dp_group.world_size
    
    # Pad input to fixed size
    if padded_buffer is not None and padded_buffer.shape[0] >= padded_size:
        # Use pre-allocated buffer - zero it first
        padded_x = padded_buffer[:padded_size]
        padded_x.zero_()
        padded_x[:original_batch_size].copy_(x)
    else:
        # Allocate new buffer if pre-allocated one doesn't exist or is too small
        padded_x = torch.zeros(
            (padded_size, hidden_dim),
            dtype=x.dtype,
            device=x.device,
        )
        padded_x[:original_batch_size].copy_(x)
    
    # All-gather with fixed-size tensors
    if gathered_buffer is not None and gathered_buffer.shape[0] >= padded_size * dp_size:
        gathered_output = gathered_buffer[:padded_size * dp_size]
    else:
        gathered_output = torch.empty(
            (padded_size * dp_size, hidden_dim),
            dtype=x.dtype,
            device=x.device,
        )
    
    # Use the DP group's all_gather
    dp_group.all_gather(gathered_output, padded_x, dim=0)
    
    return gathered_output, original_batch_size


def reduce_scatter_with_unpadding(
    x: torch.Tensor,
    original_batch_size: int,
    output_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reduce-scatter and remove padding to restore original size.
    
    Args:
        x: Input tensor from MoE computation (dp_size * padded_size, hidden_dim)
        original_batch_size: Original number of tokens for this rank
        output_buffer: Optional pre-allocated output buffer
    
    Returns:
        Output tensor of shape (original_batch_size, hidden_dim)
    """
    dp_group = get_dp_group()
    
    # Reduce scatter
    scattered_output = dp_group.reduce_scatter(x, dim=0)
    
    # Remove padding - slice to original size
    if scattered_output.shape[0] > original_batch_size:
        scattered_output = scattered_output[:original_batch_size].contiguous()
    
    return scattered_output


def sync_dp_ranks_before_prefill(is_prefill: bool):
    """
    Synchronize all DP ranks before starting prefill.
    
    This ensures all GPUs hit the collective communication at the same time,
    reducing wait time for stragglers. This is especially important for prefill
    where batch sizes can vary significantly across DP ranks.
    
    Args:
        is_prefill: Whether this is a prefill batch
    """
    if not is_prefill:
        return
    
    dp_group = get_dp_group()
    if dp_group.world_size <= 1:
        return
    
    # Use a barrier to synchronize all DP ranks
    # This is cheaper than all_reduce for synchronization only
    torch.cuda.synchronize()
    dist.barrier(group=dp_group.device_group)


class MoEPaddedDispatcher:
    """
    Dispatcher for MoE layers with pre-allocated padded buffers.
    
    This provides CUDA graph-compatible dispatch and combine operations
    using fixed-size tensors.
    """
    
    def __init__(
        self,
        max_tokens_per_rank: int,
        hidden_dim: int,
        num_experts: int,
        dp_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.dp_size = dp_size
        self.dtype = dtype
        self.device = device
        
        # Pre-allocate all buffers
        self._init_buffers()
    
    def _init_buffers(self):
        """Initialize all pre-allocated buffers."""
        total_tokens = self.max_tokens_per_rank * self.dp_size
        
        # Hidden states buffers
        self.hidden_padded = torch.zeros(
            (self.max_tokens_per_rank, self.hidden_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.hidden_gathered = torch.zeros(
            (total_tokens, self.hidden_dim),
            dtype=self.dtype,
            device=self.device,
        )
        
        # Router logits buffers
        self.router_padded = torch.zeros(
            (self.max_tokens_per_rank, self.num_experts),
            dtype=self.dtype,
            device=self.device,
        )
        self.router_gathered = torch.zeros(
            (total_tokens, self.num_experts),
            dtype=self.dtype,
            device=self.device,
        )
        
        # Output buffer for reduce_scatter result
        self.output_buffer = torch.zeros(
            (self.max_tokens_per_rank, self.hidden_dim),
            dtype=self.dtype,
            device=self.device,
        )
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Dispatch hidden states and router logits to all DP ranks.
        
        Args:
            hidden_states: (num_tokens, hidden_dim)
            router_logits: (num_tokens, num_experts)
        
        Returns:
            Tuple of (gathered_hidden, gathered_router, original_batch_size)
        """
        original_batch_size = hidden_states.shape[0]
        
        # Pad hidden states
        self.hidden_padded.zero_()
        self.hidden_padded[:original_batch_size].copy_(hidden_states)
        
        # Pad router logits
        self.router_padded.zero_()
        self.router_padded[:original_batch_size].copy_(router_logits)
        
        dp_group = get_dp_group()
        
        # All-gather both tensors
        dp_group.all_gather(
            self.hidden_gathered, 
            self.hidden_padded[:self.max_tokens_per_rank], 
            dim=0
        )
        dp_group.all_gather(
            self.router_gathered,
            self.router_padded[:self.max_tokens_per_rank],
            dim=0
        )
        
        return self.hidden_gathered, self.router_gathered, original_batch_size
    
    def combine(
        self,
        moe_output: torch.Tensor,
        original_batch_size: int,
    ) -> torch.Tensor:
        """
        Combine MoE outputs via reduce-scatter and remove padding.
        
        Args:
            moe_output: (total_tokens, hidden_dim) from MoE computation
            original_batch_size: Original number of tokens for this rank
        
        Returns:
            Output tensor of shape (original_batch_size, hidden_dim)
        """
        dp_group = get_dp_group()
        
        # Reduce scatter
        scattered = dp_group.reduce_scatter(moe_output, dim=0)
        
        # Copy to output buffer and slice to original size
        actual_size = min(scattered.shape[0], original_batch_size)
        self.output_buffer[:actual_size].copy_(scattered[:actual_size])
        
        return self.output_buffer[:original_batch_size].contiguous()



