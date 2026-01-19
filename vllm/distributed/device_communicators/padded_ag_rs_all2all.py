# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CUDA graph-compatible All2All manager using padded all_gather/reduce_scatter.

This module provides an EP communication implementation that uses fixed-size
buffers with padding to enable CUDA graph capture. The key insight (from ATOM):

1. All EP ranks pad their tensors to the same max size before communication
2. This allows CUDA graphs to capture the communication operations  
3. The padding is removed after reduce_scatter

This is critical for MoE layers in prefill where batch sizes vary across ranks
and iterations, but we want CUDA graphs for reduced kernel launch overhead.
"""

import torch

from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from .all2all import AgRsAll2AllManager

logger = init_logger(__name__)


class PaddedAgRsAll2AllManager(AgRsAll2AllManager):
    """
    All2All communication using padded all_gather and reduce_scatter.
    
    This extends AgRsAll2AllManager with fixed-size buffer allocation
    for CUDA graph compatibility. Instead of using all_gatherv (variable sizes),
    it pads all inputs to max_num_tokens and uses regular all_gather.
    """

    def __init__(self, cpu_group, max_num_tokens: int):
        """
        Initialize padded all2all manager.
        
        Args:
            cpu_group: The CPU process group for communication
            max_num_tokens: Maximum number of tokens per rank (for buffer sizing)
        """
        super().__init__(cpu_group)
        self.max_num_tokens = max_num_tokens
        
        # Pre-allocated buffers will be lazily initialized based on hidden dim
        self._hidden_buffer: torch.Tensor | None = None
        self._router_buffer: torch.Tensor | None = None
        self._gathered_hidden: torch.Tensor | None = None
        self._gathered_router: torch.Tensor | None = None
        
        logger.info(
            "PaddedAgRsAll2AllManager initialized: max_tokens=%d, world_size=%d",
            max_num_tokens, self.world_size
        )

    def _ensure_buffers(
        self,
        hidden_dim: int,
        num_experts: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Lazily initialize pre-allocated buffers based on model dimensions."""
        if self._hidden_buffer is not None:
            # Check if buffers are compatible
            if (self._hidden_buffer.shape[1] == hidden_dim and 
                self._hidden_buffer.dtype == dtype and
                self._router_buffer is not None and
                self._router_buffer.shape[1] == num_experts):
                return
        
        # Allocate padded input buffers (single rank)
        self._hidden_buffer = torch.zeros(
            (self.max_num_tokens, hidden_dim),
            dtype=dtype,
            device=device,
        )
        self._router_buffer = torch.zeros(
            (self.max_num_tokens, num_experts),
            dtype=dtype,
            device=device,
        )
        
        # Allocate gathered buffers (all ranks)
        total_tokens = self.max_num_tokens * self.world_size
        self._gathered_hidden = torch.zeros(
            (total_tokens, hidden_dim),
            dtype=dtype,
            device=device,
        )
        self._gathered_router = torch.zeros(
            (total_tokens, num_experts),
            dtype=dtype,
            device=device,
        )
        
        logger.debug(
            "Allocated EP communication buffers: hidden=%s, router=%s",
            self._hidden_buffer.shape, self._router_buffer.shape
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch hidden_states and router_logits using padded all_gather.
        
        This pads inputs to max_num_tokens, performs all_gather with fixed sizes,
        then returns the gathered result.
        """
        if extra_tensors is not None:
            # Fall back to base implementation for extra tensors
            return super().dispatch(
                hidden_states, router_logits, is_sequence_parallel, extra_tensors
            )
        
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = router_logits.shape[1]
        
        # Ensure buffers are allocated
        self._ensure_buffers(
            hidden_dim, num_experts, 
            hidden_states.dtype, hidden_states.device
        )
        
        assert self._hidden_buffer is not None
        assert self._router_buffer is not None
        assert self._gathered_hidden is not None
        assert self._gathered_router is not None
        
        # Pad hidden states to fixed size
        padded_hidden = self._hidden_buffer[:self.max_num_tokens]
        padded_hidden.zero_()
        padded_hidden[:num_tokens].copy_(hidden_states)
        
        # Pad router logits to fixed size
        padded_router = self._router_buffer[:self.max_num_tokens]
        padded_router.zero_()
        padded_router[:num_tokens].copy_(router_logits)
        
        # All-gather with fixed sizes (CUDA graph compatible)
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        
        # Use the group's all_gather with fixed buffer sizes
        dist_group.all_gather(
            self._gathered_hidden[:self.max_num_tokens * self.world_size],
            padded_hidden,
            dim=0
        )
        dist_group.all_gather(
            self._gathered_router[:self.max_num_tokens * self.world_size],
            padded_router,
            dim=0
        )
        
        # Return gathered tensors (including padding)
        # The MoE kernel will process all tokens, but padding contributes zero
        return (
            self._gathered_hidden[:self.max_num_tokens * self.world_size].clone(),
            self._gathered_router[:self.max_num_tokens * self.world_size].clone(),
        )

    def combine(
        self, 
        hidden_states: torch.Tensor, 
        is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine hidden_states using reduce_scatter.
        
        Since all ranks padded to the same size, we can use reduce_scatter
        with fixed sizes.
        """
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        
        # Get actual token count from forward context
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank
        actual_tokens = sizes[ep_rank]
        
        # Allocate output buffer if needed
        output_dim = hidden_states.shape[1]
        output = torch.zeros(
            (self.max_num_tokens, output_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        
        # Reduce-scatter with fixed sizes
        # Note: This assumes hidden_states has shape (max_tokens * world_size, dim)
        dist_group.reduce_scatter(output, hidden_states, dim=0)
        
        # Return only the actual tokens (remove padding)
        return output[:actual_tokens].clone()

    def destroy(self):
        """Clean up allocated buffers."""
        self._hidden_buffer = None
        self._router_buffer = None
        self._gathered_hidden = None
        self._gathered_router = None
        super().destroy()



