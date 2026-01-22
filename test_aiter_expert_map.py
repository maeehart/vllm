#!/usr/bin/env python3
"""
Test script to understand AITER's expert_mask/expert_map interface.

This test uses vLLM's FusedMoE layer to ensure correct weight setup,
then tests different expert_map configurations.
"""

import torch
import os

os.environ["VLLM_ROCM_USE_AITER"] = "1"
os.environ["VLLM_ROCM_USE_AITER_MOE"] = "1"


def test_expert_map_interface():
    """Test what AITER expects for expert_map parameter."""
    print("=" * 80)
    print("Testing AITER expert_map interface via vLLM FusedMoE")
    print("=" * 80)
    
    # Import necessary modules
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    
    # Create a minimal config
    vllm_config = VllmConfig.from_cli([
        "--model", "dummy",  # Won't actually load
        "--enforce-eager",
    ])
    
    # Test parameters (smaller for speed)
    num_experts = 32
    hidden_size = 256  # Smaller for testing
    intermediate_size = 512
    topk = 8
    num_tokens = 4
    dtype = torch.bfloat16
    
    print(f"\nConfig: {num_experts} experts, hidden={hidden_size}, intermediate={intermediate_size}")
    print(f"        topk={topk}, num_tokens={num_tokens}")
    
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        # Create a FusedMoE layer
        moe = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=dtype,
            tp_size=1,
            dp_size=1,
        ).cuda()
        
        print(f"\nFusedMoE layer created")
        print(f"  w13_weight shape: {moe.experts.w13_weight.shape}")
        print(f"  w2_weight shape: {moe.experts.w2_weight.shape}")
        print(f"  expert_map: {moe.expert_map}")
        print(f"  expert_mask: {moe.expert_mask}")
        print(f"  rocm_aiter_fmoe_enabled: {moe.rocm_aiter_fmoe_enabled}")
        
        # Initialize weights with small random values
        moe.experts.w13_weight.data.normal_(0, 0.01)
        moe.experts.w2_weight.data.normal_(0, 0.01)
        moe.gate.weight.data.normal_(0, 0.1)
        
        # Process weights (may transpose for AITER)
        moe.experts.quant_method.process_weights_after_loading(moe.experts)
        
        # Create input
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * 0.1
        router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
        
        print(f"\nInput hidden_states: shape={hidden_states.shape}, std={hidden_states.float().std():.4f}")
        
        # Run forward pass
        print("\n" + "=" * 80)
        print("Running FusedMoE forward pass")
        print("=" * 80)
        try:
            output = moe.forward(hidden_states, router_logits)
            print(f"✅ Output shape: {output.shape}")
            print(f"   Output std: {output.float().std():.6f}")
            print(f"   Output mean: {output.float().mean():.6f}")
            print(f"   Output[0,:5]: {[f'{v:.4f}' for v in output[0,:5].float().tolist()]}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def test_expert_map_property():
    """Specifically test the expert_map property behavior."""
    print("\n" + "=" * 80)
    print("Testing expert_map property in EP mode")
    print("=" * 80)
    
    from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
    
    # Simulate EP mode: 8 ranks, 256 global experts, 32 local each
    num_global_experts = 256
    num_local_experts = 32
    ep_size = 8
    
    for ep_rank in [0, 1, 7]:
        num_local, expert_map, expert_mask = determine_expert_map(
            global_num_experts=num_global_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            return_expert_mask=True,
        )
        
        print(f"\nEP rank {ep_rank}:")
        start_idx = ep_rank * num_local_experts
        end_idx = start_idx + num_local_experts
        print(f"  Should own experts {start_idx}-{end_idx-1}")
        print(f"  expert_map[{start_idx}:{start_idx+3}] = {expert_map[start_idx:start_idx+3].tolist()}")
        print(f"  expert_mask[{start_idx}:{start_idx+3}] = {expert_mask[start_idx:start_idx+3].tolist()}")
        print(f"  expert_map[{end_idx-1}:{end_idx+2}] = {expert_map[end_idx-1:end_idx+2].tolist()}")
        print(f"  expert_mask[{end_idx-1}:{end_idx+2}] = {expert_mask[end_idx-1:end_idx+2].tolist()}")
        
        # Check the key difference
        print(f"\n  KEY DIFFERENCE:")
        print(f"    expert_map[{start_idx+5}] = {expert_map[start_idx+5].item()} (should be 5 = local index)")
        print(f"    expert_mask[{start_idx+5}] = {expert_mask[start_idx+5].item()} (just 1 = binary)")
        
        if expert_map[start_idx+5].item() != expert_mask[start_idx+5].item():
            print(f"    ⚠️ Different values! AITER might interpret 1 as local_index=1, not local_index={expert_map[start_idx+5].item()}")


def show_the_bug():
    """Demonstrate the actual bug in the code."""
    print("\n" + "=" * 80)
    print("THE BUG: layer.py expert_map property")
    print("=" * 80)
    
    print("""
In vllm/model_executor/layers/fused_moe/layer.py, lines 1594-1598:

    @property
    def expert_map(self) -> torch.Tensor | None:
        return (
            self._expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask
        )

When rocm_aiter_fmoe_enabled=True (i.e., using AITER for MoE):
- It returns expert_mask (binary: 0 or 1)
- But AITER expects expert_map (mapping: global_id -> local_id or -1)

Example for EP rank 0 (owns experts 0-31):
- expert_map[5] = 5   (global expert 5 -> local index 5)
- expert_mask[5] = 1  (just says "expert 5 is local")

When AITER receives expert_mask[5]=1, it might:
- Use local_index=1 instead of local_index=5
- Access wrong expert weights
- Produce garbage output

FIX: The property should always return _expert_map, or we need
     to handle the MORI EP case specially.
""")


if __name__ == "__main__":
    # Show what the bug is
    show_the_bug()
    
    # Test the expert_map property behavior
    test_expert_map_property()
    
    # Try running a basic forward pass (may still crash due to GPU issues)
    # test_expert_map_interface()
