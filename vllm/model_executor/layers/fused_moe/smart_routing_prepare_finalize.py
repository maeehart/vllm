# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Smart Routing for MoE Expert Parallelism.

KEY INSIGHT: Route TOKENS, not (token, expert) pairs.

With 16k tokens, topk=8, EP=8:
- WRONG: Route 16k × 8 = 128k items (massive data expansion)
- RIGHT: Route 16k tokens, each to 1-8 GPUs based on which experts it needs

Each token is sent AT MOST ONCE to each GPU. On arrival, the GPU processes
the token through ALL local experts that the token selected.

Example with round_robin expert placement:
- Token selects experts [0, 8, 16, 32, 40, 48, 64, 72]
- With round_robin: expert_i is on GPU (i % 8)
- Experts 0,8,16 → GPU0; expert 32,40,48 → GPU0; expert 64,72 → GPU0
- Wait, that's wrong! Let me recalculate...
- Expert 0 % 8 = 0 → GPU0
- Expert 8 % 8 = 0 → GPU0  
- Expert 16 % 8 = 0 → GPU0
- Expert 32 % 8 = 0 → GPU0
- All go to GPU0! This is because DeepSeek selects from expert "groups"

For DeepSeek's grouped_topk with 8 groups:
- Linear placement: GPU_i owns group_i entirely → token goes to 4 GPUs (topk_group=4)
- Round-robin: spreads groups across GPUs → more balanced

The L2 cache benefit comes from:
1. Fewer tokens per GPU (if well distributed)
2. Fewer expert iterations (32 vs 256)
3. Each expert's weights stay hot longer
"""

from typing import Optional

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

logger = init_logger(__name__)


class SmartRoutingPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Smart routing for MoE Expert Parallelism.
    
    Routes TOKENS (not token-expert pairs) to GPUs based on expert ownership.
    Each token goes to each GPU at most once.
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        max_num_tokens: int,
        hidden_dim: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_size = dist.get_world_size(ep_group)
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.max_num_tokens = max_num_tokens
        self.hidden_dim = hidden_dim
        self.use_fp8_dispatch = use_fp8_dispatch
        
        # Store dispatch metadata for combine phase
        self._dispatch_info: dict = {}
        
        logger.info(
            f"SmartRoutingPrepareAndFinalize initialized: "
            f"rank={self.ep_rank}, size={self.ep_size}, "
            f"num_experts={num_experts}, local_experts={self.num_local_experts}, "
            f"max_tokens={max_num_tokens}"
        )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self) -> int:
        return self.ep_size

    def max_num_tokens_per_rank(self) -> int | None:
        # Each GPU could receive all tokens in worst case
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def _get_expert_to_rank(self, expert_id: int) -> int:
        """Map expert ID to the rank that owns it."""
        return expert_id // self.num_local_experts

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
        Route tokens to GPUs based on expert selection.
        
        Each TOKEN is sent to each GPU at most once. The GPU then processes
        the token through all local experts that the token selected.
        """
        assert not apply_router_weight_on_input, (
            "Smart routing does not support apply_router_weight_on_input=True"
        )
        
        num_tokens, hidden_dim = a1.shape
        topk = topk_ids.shape[1]
        device = a1.device
        dtype = a1.dtype
        
        logger.debug(f"[SmartRouting] prepare: num_tokens={num_tokens}, topk={topk}, "
                     f"rank={self.ep_rank}")
        
        # Step 1: For each token, determine which GPUs it needs to visit
        # expert_ranks[i, k] = rank that owns expert topk_ids[i, k]
        expert_ranks = topk_ids // self.num_local_experts  # [num_tokens, topk]
        
        # Step 2: Create a mask of which tokens go to which GPUs
        # token_to_gpu[i, r] = True if token i needs to go to GPU r
        token_to_gpu = torch.zeros(num_tokens, self.ep_size, dtype=torch.bool, device=device)
        for k in range(topk):
            # For each topk selection, mark the destination GPU
            token_to_gpu.scatter_(1, expert_ranks[:, k:k+1], True)
        
        # Step 3: Count tokens going to each GPU
        send_counts = token_to_gpu.sum(dim=0).to(torch.int64)  # [ep_size]
        
        # Step 4: Exchange counts with all GPUs
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        total_send = sum(send_counts_list)
        total_recv = sum(recv_counts_list)
        
        logger.debug(f"[SmartRouting] send_counts={send_counts_list}, recv_counts={recv_counts_list}")
        logger.debug(f"[SmartRouting] total_send={total_send}, total_recv={total_recv}")
        
        # Step 5: Gather tokens for each destination GPU
        # We need to pack: hidden_states, token_id, selected_experts_mask
        
        # For each GPU, gather the tokens that need to go there
        send_hidden = []
        send_token_ids = []
        send_expert_masks = []  # Which of my local experts this token selected
        send_weights_list = []  # Router weights for selected experts
        
        for dest_rank in range(self.ep_size):
            # Get tokens going to this rank
            mask = token_to_gpu[:, dest_rank]  # [num_tokens]
            token_indices = mask.nonzero(as_tuple=True)[0]  # indices of tokens
            
            if len(token_indices) > 0:
                send_hidden.append(a1[token_indices])  # [n, hidden_dim]
                send_token_ids.append(token_indices)    # [n]
                
                # For each token, which experts on dest_rank did it select?
                # Local expert range for dest_rank: [dest_rank * num_local, (dest_rank+1) * num_local)
                local_start = dest_rank * self.num_local_experts
                local_end = local_start + self.num_local_experts
                
                # Get the expert IDs and weights for these tokens
                selected_experts = topk_ids[token_indices]  # [n, topk]
                selected_weights = topk_weights[token_indices]  # [n, topk]
                
                # Create mask of which selections are local to dest_rank
                is_local = (selected_experts >= local_start) & (selected_experts < local_end)
                
                # Store local expert IDs (relative to dest_rank's local experts)
                local_expert_ids = (selected_experts - local_start) * is_local + \
                                   (-1) * (~is_local)  # -1 for non-local
                
                send_expert_masks.append(local_expert_ids)  # [n, topk]
                send_weights_list.append(selected_weights * is_local.float())  # Zero out non-local weights
            else:
                # Empty tensors for this destination
                send_hidden.append(torch.empty(0, hidden_dim, dtype=dtype, device=device))
                send_token_ids.append(torch.empty(0, dtype=torch.int64, device=device))
                send_expert_masks.append(torch.empty(0, topk, dtype=torch.int64, device=device))
                send_weights_list.append(torch.empty(0, topk, dtype=topk_weights.dtype, device=device))
        
        # Concatenate all sends
        send_hidden_cat = torch.cat(send_hidden, dim=0)  # [total_send, hidden_dim]
        send_token_ids_cat = torch.cat(send_token_ids, dim=0)  # [total_send]
        send_expert_masks_cat = torch.cat(send_expert_masks, dim=0)  # [total_send, topk]
        send_weights_cat = torch.cat(send_weights_list, dim=0)  # [total_send, topk]
        
        # Step 6: All-to-all exchange
        recv_hidden = torch.empty(total_recv, hidden_dim, dtype=dtype, device=device)
        recv_token_ids = torch.empty(total_recv, dtype=torch.int64, device=device)
        recv_expert_masks = torch.empty(total_recv, topk, dtype=torch.int64, device=device)
        recv_weights = torch.empty(total_recv, topk, dtype=topk_weights.dtype, device=device)
        
        # Exchange hidden states
        dist.all_to_all_single(
            recv_hidden.view(-1),
            send_hidden_cat.contiguous().view(-1),
            output_split_sizes=[c * hidden_dim for c in recv_counts_list],
            input_split_sizes=[c * hidden_dim for c in send_counts_list],
            group=self.ep_group,
        )
        
        # Exchange token IDs
        dist.all_to_all_single(
            recv_token_ids,
            send_token_ids_cat.contiguous(),
            output_split_sizes=recv_counts_list,
            input_split_sizes=send_counts_list,
            group=self.ep_group,
        )
        
        # Exchange expert masks
        dist.all_to_all_single(
            recv_expert_masks.view(-1),
            send_expert_masks_cat.contiguous().view(-1),
            output_split_sizes=[c * topk for c in recv_counts_list],
            input_split_sizes=[c * topk for c in send_counts_list],
            group=self.ep_group,
        )
        
        # Exchange weights
        dist.all_to_all_single(
            recv_weights.view(-1),
            send_weights_cat.contiguous().view(-1),
            output_split_sizes=[c * topk for c in recv_counts_list],
            input_split_sizes=[c * topk for c in send_counts_list],
            group=self.ep_group,
        )
        
        # Step 7: Expand received tokens for local expert processing
        # Each token may need to go through multiple local experts
        # We need to create (token, expert) pairs for local processing
        
        # Find valid (token, local_expert) pairs
        valid_mask = recv_expert_masks >= 0  # [total_recv, topk]
        
        # Expand to create individual pairs
        recv_idx, topk_idx = valid_mask.nonzero(as_tuple=True)
        local_expert_ids = recv_expert_masks[recv_idx, topk_idx]  # Local expert IDs
        local_weights = recv_weights[recv_idx, topk_idx]  # Weights for these pairs
        local_hidden = recv_hidden[recv_idx]  # Hidden states (may be repeated)
        local_token_ids = recv_token_ids[recv_idx]  # Original token IDs
        
        num_local_pairs = len(recv_idx)
        
        logger.debug(f"[SmartRouting] recv_tokens={total_recv}, local_pairs={num_local_pairs}")
        
        # Store info for combine phase
        self._dispatch_info = {
            'num_tokens': num_tokens,
            'send_counts': send_counts_list,
            'recv_counts': recv_counts_list,
            'recv_idx': recv_idx,  # Which received token each pair came from
            'local_token_ids': local_token_ids,  # Original token IDs
            'local_weights': local_weights,  # Weights for weighting outputs
            'total_recv': total_recv,
            'recv_hidden': recv_hidden,  # Keep for indexing in combine
            'hidden_dtype': dtype,
        }
        
        # Convert local expert IDs to global for the MoE kernel
        global_expert_ids = local_expert_ids + (self.ep_rank * self.num_local_experts)
        
        # Step 8: Quantize activations after routing (if FP8 is enabled)
        # This is critical: we route bf16 activations, then quantize on destination GPU
        # so that activation scales match the received tokens, not original layout.
        if quant_config.quant_dtype is not None and num_local_pairs > 0:
            local_hidden_q, a1_scale = moe_kernel_quantize_input(
                local_hidden,
                quant_config.a1_scale,  # May be None for dynamic quantization
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
            )
            logger.debug(f"[SmartRouting] Quantized activations: {local_hidden.shape} -> {local_hidden_q.shape}, scale={a1_scale.shape if a1_scale is not None else None}")
        else:
            local_hidden_q = local_hidden
            a1_scale = None
        
        # Return format expected by MoE kernel
        # Note: We're returning expanded pairs, each with topk=1
        return (
            local_hidden_q,  # [num_local_pairs, hidden_dim] - quantized if FP8
            a1_scale,  # Per-token activation scale (if quantized)
            None,  # expert_tokens_meta
            global_expert_ids.unsqueeze(1).to(torch.int32),  # [num_local_pairs, 1]
            local_weights.unsqueeze(1),  # [num_local_pairs, 1]
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
        Combine expert outputs back to original token owners.
        """
        info = self._dispatch_info
        num_tokens = info['num_tokens']
        send_counts = info['recv_counts']  # Reversed for combine
        recv_counts = info['send_counts']
        local_token_ids = info['local_token_ids']
        local_weights = info['local_weights']
        hidden_dtype = info['hidden_dtype']
        
        device = output.device
        hidden_dim = output.shape[1]
        
        # fused_expert_output is [num_local_pairs, hidden_dim]
        # Apply router weights
        weighted_output = fused_expert_output * local_weights.unsqueeze(1).to(fused_expert_output.dtype)
        
        # We need to aggregate by local_token_ids first (multiple pairs per received token)
        # Then send back to original owners
        
        # First, aggregate pairs that came from the same received token
        total_recv = info['total_recv']
        recv_idx = info['recv_idx']
        
        # Aggregate weighted outputs by received token index
        recv_aggregated = torch.zeros(total_recv, hidden_dim, dtype=fused_expert_output.dtype, device=device)
        recv_aggregated.index_add_(0, recv_idx, weighted_output)
        
        # Now send back to original token owners via reverse all-to-all
        # Also need to send back the original token IDs
        
        # Get token IDs for each received token (not expanded pairs)
        recv_token_ids = info['local_token_ids']
        # Actually we need the token IDs per received token, not per pair
        # Let's reconstruct from recv_hidden indices
        # Hmm, we stored recv_idx which maps pairs to received tokens
        # We need to get unique token IDs per received token
        
        # Actually, we stored recv_token_ids for the received tokens before expansion
        # Let me re-read the prepare method...
        # We have local_token_ids which is indexed by recv_idx
        # So local_token_ids[i] is the original token ID for pair i
        # We need token IDs for each received token (before expansion)
        
        # Let's collect unique token IDs per received token
        # recv_idx tells us which received token each pair came from
        # For each received token index 0..total_recv-1, get any token ID
        
        recv_token_ids_unique = torch.zeros(total_recv, dtype=torch.int64, device=device)
        # Use scatter to get one token ID per received token
        recv_token_ids_unique.scatter_(0, recv_idx, local_token_ids)
        
        # All-to-all to send aggregated results back
        total_send_back = sum(recv_counts)
        send_back_buffer = torch.empty(total_send_back, hidden_dim, dtype=hidden_dtype, device=device)
        send_back_token_ids = torch.empty(total_send_back, dtype=torch.int64, device=device)
        
        dist.all_to_all_single(
            send_back_buffer.view(-1),
            recv_aggregated.to(hidden_dtype).contiguous().view(-1),
            output_split_sizes=[c * hidden_dim for c in recv_counts],
            input_split_sizes=[c * hidden_dim for c in send_counts],
            group=self.ep_group,
        )
        
        dist.all_to_all_single(
            send_back_token_ids,
            recv_token_ids_unique.contiguous(),
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group,
        )
        
        # Aggregate by original token ID
        output.zero_()
        output.index_add_(0, send_back_token_ids, send_back_buffer.to(output.dtype))


def create_smart_routing_manager(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    max_num_tokens: int,
    hidden_dim: int,
    use_fp8_dispatch: bool = False,
) -> SmartRoutingPrepareAndFinalize:
    """Factory function to create SmartRoutingPrepareAndFinalize."""
    return SmartRoutingPrepareAndFinalize(
        ep_group=ep_group,
        num_experts=num_experts,
        max_num_tokens=max_num_tokens,
        hidden_dim=hidden_dim,
        use_fp8_dispatch=use_fp8_dispatch,
    )
