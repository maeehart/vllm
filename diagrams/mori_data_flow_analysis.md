# MORI Data Flow Analysis - Root Cause Investigation

## Key Log Observations

```
[MORI DEDUP] ep_rank=5, total_recv=65536, unique_tokens=65536
[MORI DEBUG] recv_topk_ids[:3]=[[79, 81, 108, 120, 161, 179, 3, 30], 
                                [79, 81, 108, 120, 161, 179, 3, 30], 
                                [79, 81, 108, 120, 161, 179, 3, 30]]
[MORI DEBUG] Sample topk_ids=[79, 81, 108, 120, 161, 179, 3, 30], local_experts=[161, 179]
[MORI COMBINE DEBUG] fused_expert_output shape=torch.Size([65536, 7168])
[MORI COMBINE DEBUG] original_topk_ids shape=torch.Size([8192, 8])
[MORI COMBINE DEBUG] output shape=torch.Size([8192, 7168])
```

## Analysis

### 1. Number of Received Entries

```
total_recv = 65536 = 8 ranks × 8192 tokens_per_rank
unique_tokens = 65536 (same as total_recv)
```

**Interpretation**: MORI receives tokens from ALL ranks. Each received entry is from 
a UNIQUE source token (no duplicates). This is expected for expert parallelism.

### 2. Why Multiple Entries Have Same topk_ids

Three entries have identical topk_ids `[79, 81, 108, 120, 161, 179, 3, 30]`:
- These are THREE DIFFERENT source tokens
- They just happen to have the same router output (selected same experts)
- This is normal - multiple tokens can select the same expert combination

### 3. Expert Computation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     WHAT HAPPENS ON RANK 5                                       │
│                                                                                  │
│  Input: 65536 received tokens from all ranks                                    │
│  Each token has topk_ids [8 experts] (GLOBAL IDs)                               │
│                                                                                  │
│  AITER fused_moe receives:                                                       │
│    - hidden_states: [65536, 7168]                                               │
│    - topk_ids: [65536, 8] (GLOBAL IDs: 0-255)                                   │
│    - expert_mask: [257] with 1s at positions 160-191                            │
│                                                                                  │
│  AITER processing:                                                               │
│    For each of 65536 tokens:                                                     │
│      For each expert in token's topk_ids:                                        │
│        IF expert_mask[expert] == 1:                                             │
│          Compute: output += weight * W2 @ act(W1 @ input)                       │
│                                                                                  │
│  Result: [65536, 7168] - each token's contribution from LOCAL experts only      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. The Combine Step (CRITICAL)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MORI COMBINE CALL                                        │
│                                                                                  │
│  ep_op.combine(                                                                  │
│    input=fused_expert_output,        # [65536, 7168] - expert outputs           │
│    weights=None,                      # No weights (AITER already applied)       │
│    indices=original_topk_ids,         # [8192, 8] - THIS rank's tokens          │
│  )                                                                               │
│                                                                                  │
│  Expected behavior:                                                              │
│    1. Route 65536 results back to their source tokens                           │
│    2. Receive results from other ranks for this rank's 8192 tokens              │
│    3. Sum all contributions for each token                                       │
│    4. Return [8192, 7168] final output                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## POTENTIAL ISSUE: AITER's Output vs What MORI Expects

### What AITER Produces

```python
# AITER fused_moe output for a single token with topk_ids [79, 81, 108, 120, 161, 179, 3, 30]:
#
# On Rank 5 (owns experts 160-191):
#   - Only experts 161 and 179 have expert_mask=1
#   - AITER computes:
#       output = w161 * W2[1] @ act(W1[1] @ x) +   # local ID 1
#                w179 * W2[19] @ act(W1[19] @ x)   # local ID 19
#   - This is the WEIGHTED SUM of local expert contributions
```

### What MORI Combine Expects

Looking at MORI test code:
```python
# In test, combine_input is just a copy of dispatch_output (raw hidden states)
# The test verifies: combine_output ≈ input * unique_pes
# This suggests MORI combine does SUMMATION of inputs
```

### The Critical Question

**Does MORI combine expect**:
- A) Raw expert outputs (one per dispatched entry)?
- B) Already-weighted sums (what AITER produces)?

If MORI expects (A), then our pipeline is wrong because AITER produces (B).

## Investigation Points

1. **Check MORI combine semantics**: Does it do weighted summation internally?
2. **Check if weights=None is handled correctly**: Maybe MORI needs weights to function
3. **Verify topk_ids usage in AITER**: Are we passing correct local/global IDs?

## Shape Analysis

```
Dispatch:
  - Input tokens: [8192, 7168] per rank
  - Total dispatched: 8 ranks × 8192 = 65536 tokens
  - Each rank receives: ~65536 (depends on expert distribution)

Expert Compute (AITER):
  - Input: [65536, 7168]
  - topk_ids: [65536, 8] (GLOBAL IDs)
  - expert_mask: [257] (binary, 1s at 160-191 for rank 5)
  - Output: [65536, 7168] (weighted sum of local expert outputs)

Combine:
  - Input to combine: [65536, 7168] (AITER outputs)
  - Original indices: [8192, 8] (this rank's expert selections)
  - Output: [8192, 7168] (final token outputs for this rank)
```

## Possible Fixes

### Option 1: Don't Use AITER's Full fused_moe

If MORI expects raw expert outputs, we need to:
- Compute each expert separately: `expert_out[i] = W2[local_id] @ act(W1[local_id] @ x)`
- NOT apply weights in expert computation
- Let MORI combine apply weights and sum

### Option 2: Verify expert_mask Filtering

Check if AITER is correctly filtering experts. Maybe the mask isn't being applied correctly?

### Option 3: Debug MORI Combine

Add logging to see what combine is actually doing:
- What's in the combine output?
- Is it all zeros? NaN? Random?

## Next Steps

1. Add debug prints to see actual values (not just shapes)
2. Test with a simple case (1 token, 1 expert)
3. Compare combine output with expected values

