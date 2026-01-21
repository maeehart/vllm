# MORI Dispatch/Combine Analysis - Understanding the Bug

## Key Observation from Logs

```
recv_topk_ids[:3] = [
  [79, 81, 108, 120, 161, 179, 3, 30],  # Entry 0 - SAME token!
  [79, 81, 108, 120, 161, 179, 3, 30],  # Entry 1 - SAME token!  
  [79, 81, 108, 120, 161, 179, 3, 30],  # Entry 2 - SAME token!
]
local_experts = [161, 179]  # Only 2 of 8 experts are on rank 5
total_recv = 65536 = 8192 tokens × 8 experts
```

## The Problem: MORI Dispatches at (Token, Expert) Granularity

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MORI DISPATCH GRANULARITY                                     │
│                                                                                  │
│  Original Token T on Rank 0:                                                     │
│    hidden_states: [H]                                                           │
│    topk_ids: [79, 81, 108, 120, 161, 179, 3, 30]  (8 experts)                   │
│                                                                                  │
│  Expert → Rank mapping (32 experts per rank):                                    │
│    Experts 0-31   → Rank 0                                                       │
│    Experts 32-63  → Rank 1                                                       │
│    Experts 64-95  → Rank 2                                                       │
│    Experts 96-127 → Rank 3                                                       │
│    Experts 128-159→ Rank 4                                                       │
│    Experts 160-191→ Rank 5  ← owns experts 161, 179                             │
│    Experts 192-223→ Rank 6                                                       │
│    Experts 224-255→ Rank 7                                                       │
│                                                                                  │
│  Token T's experts map to ranks:                                                 │
│    79  → Rank 2                                                                  │
│    81  → Rank 2                                                                  │
│    108 → Rank 3                                                                  │
│    120 → Rank 3                                                                  │
│    161 → Rank 5  ✓                                                               │
│    179 → Rank 5  ✓                                                               │
│    3   → Rank 0                                                                  │
│    30  → Rank 0                                                                  │
│                                                                                  │
│  MORI sends 2 COPIES of T to Rank 5:                                            │
│    Copy 1: For expert 161                                                        │
│    Copy 2: For expert 179                                                        │
│    Both copies carry the FULL topk_ids: [79, 81, 108, 120, 161, 179, 3, 30]     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Current BROKEN Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT BROKEN FLOW                                    │
│                                                                                  │
│  Step 1: MORI Dispatch                                                           │
│    Input:  8192 tokens × 8 experts = 65536 (token, expert) pairs                │
│    Output: expert_x[65536, 7168], recv_topk_ids[65536, 8]                        │
│                                                                                  │
│  Step 2: Pass to AITER fused_moe (WRONG!)                                       │
│    hidden_states: [65536, 7168]  ← 65536 entries!                               │
│    topk_ids: [65536, 8]          ← Each with ALL 8 experts!                     │
│                                                                                  │
│  Step 3: AITER computes                                                          │
│    65536 entries × 8 experts each = 524,288 expert computations!                │
│    Should be: ~8192 unique tokens × ~1 local expert = ~8192 computations        │
│                                                                                  │
│    WASTE FACTOR: 524,288 / 8,192 = 64× !!!                                      │
│                                                                                  │
│  Step 4: Results are garbage because:                                            │
│    - Same token computed multiple times                                          │
│    - Each copy computes ALL 8 experts (including non-local ones!)               │
│    - Results don't aggregate correctly                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## What MORI Actually Sends

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MORI DISPATCH OUTPUT STRUCTURE                              │
│                                                                                  │
│  For each (token, expert) pair dispatched to this rank:                         │
│                                                                                  │
│    expert_x[i]       = hidden_states of source token                            │
│    recv_topk_ids[i]  = ALL 8 expert IDs of source token (global)                │
│    recv_weights[i]   = ALL 8 weights of source token                            │
│                                                                                  │
│  CRITICAL: recv_topk_ids[i] does NOT tell us WHICH expert this entry is for!    │
│                                                                                  │
│  Example on Rank 5 (owns experts 160-191):                                       │
│    Entry 0: Token T, recv_topk_ids=[79, 81, 108, 120, 161, 179, 3, 30]          │
│             → This entry is for expert 161 OR expert 179, but which one?        │
│    Entry 1: Token T, recv_topk_ids=[79, 81, 108, 120, 161, 179, 3, 30]          │
│             → This is the OTHER expert (179 or 161)                              │
│                                                                                  │
│  We need: get_dispatch_expert_idx() or similar to know target expert!           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## What We SHOULD Do

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORRECT FLOW (OPTION A)                                │
│                    Deduplicate + Filter to Local Experts                         │
│                                                                                  │
│  Step 1: Get source token positions                                              │
│    src_token_pos = ep_op.get_dispatch_src_token_pos()                           │
│    → Tells us which original token each dispatched entry came from              │
│                                                                                  │
│  Step 2: Deduplicate by source token                                            │
│    unique_tokens, inverse = torch.unique(src_token_pos, return_inverse=True)    │
│    → Get unique token hidden states                                              │
│                                                                                  │
│  Step 3: For each unique token, filter topk_ids to LOCAL experts only           │
│    For token with topk_ids=[79, 81, 108, 120, 161, 179, 3, 30]:                 │
│    On Rank 5: local_mask = (topk_ids >= 160) & (topk_ids < 192)                 │
│    local_topk_ids = [161, 179]  (only 2 experts)                                │
│    local_weights = [w5, w6]     (corresponding weights)                         │
│                                                                                  │
│  Step 4: Call AITER with filtered data                                           │
│    hidden_states: [N_unique, 7168]                                              │
│    topk_ids: [N_unique, K_local]  ← Variable K_local per token!                 │
│                                                                                  │
│  Problem: AITER expects fixed K per token!                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORRECT FLOW (OPTION B)                                │
│                    Process Each Entry as Single Expert                           │
│                                                                                  │
│  Key insight: Each dispatched entry is for ONE specific expert.                  │
│  We need to know WHICH expert each entry is for.                                 │
│                                                                                  │
│  If MORI provides get_dispatch_target_expert():                                  │
│    target_expert[i] = which expert entry i is meant for                         │
│                                                                                  │
│  Then:                                                                           │
│    For each entry i:                                                             │
│      expert_id = target_expert[i]           # e.g., 161                         │
│      local_id = expert_id - rank_offset     # 161 - 160 = 1                     │
│      weight = recv_weights[i, expert_pos]   # weight for this expert            │
│      output[i] = w2[local_id] @ act(w1[local_id] @ x[i]) * weight               │
│                                                                                  │
│  This is the CORRECT per-entry computation!                                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## MORI Test Code Reference

Looking at MORI's own test (`/workspace/mori/examples/ops/dispatch_combine/test_dispatch_combine.py`):

```python
# Dispatch
(dispatch_output, dispatch_weights, dispatch_scales, 
 dispatch_indices, dispatch_recv_num_token) = op.dispatch(
    input, weights, scales, indices, ...)

# The test then does custom expert computation (not using AITER's fused kernel)
# Each rank only computes for its LOCAL experts

# Combine  
combine_output, combine_output_weight = op.combine(
    combine_input,    # Expert outputs
    combine_weight,   # Weights
    indices,          # ORIGINAL indices (from THIS rank's tokens)
    ...)
```

The test doesn't use a fused MoE kernel - it does per-expert computation manually.

## Root Cause Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ROOT CAUSE                                          │
│                                                                                  │
│  1. MORI dispatches at (token, expert) granularity                              │
│     → Each entry is for ONE specific expert, but we don't know which!           │
│                                                                                  │
│  2. We pass ALL entries with ALL 8 expert IDs to AITER                          │
│     → AITER computes 8 experts per entry = 64× over-computation                 │
│                                                                                  │
│  3. We don't deduplicate by source token                                         │
│     → Same token processed multiple times                                        │
│                                                                                  │
│  4. Expert ID remapping is correct (global → local) but irrelevant              │
│     → The real issue is we're computing TOO MANY experts!                        │
│                                                                                  │
│  FIX NEEDED:                                                                     │
│    Either:                                                                       │
│    A) Find which expert each entry is for, compute only that one                │
│    B) Deduplicate by token, filter to local experts, call AITER once            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Questions to Investigate

1. Does MORI provide `get_dispatch_target_expert()` or similar?
2. Can we infer target expert from `get_dispatch_src_token_pos()`?
3. Should we use a different expert computation approach (not AITER fused)?

