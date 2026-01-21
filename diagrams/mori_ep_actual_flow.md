# MORI-EP Actual Data Flow (One MoE Layer During Decode)

## Context
- **Model**: DeepSeek-R1 (256 experts, topk=8)
- **Setup**: 8 GPUs, each owns 32 experts
- **Mode**: Decode (1 token per rank)
- **Rank 0** owns experts **0-31**

---

## OBSERVED FLOW (from logs)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DISPATCH (on Rank 0)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                         │
│    tokens shape = [1, 7168]           ← 1 token from this rank                  │
│    tokens mean = -0.0061                                                        │
│    topk_ids shape = [1, 8]            ← 8 selected experts                      │
│                                                                                 │
│  MORI ep_op.dispatch() sends this token to all 8 ranks that own selected       │
│  experts. Each rank receives entries for experts it owns.                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
                                   ALL-TO-ALL
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RECEIVE (on Rank 0)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  🔴 KEY OBSERVATION: Rank 0 receives 8 entries!                                 │
│                                                                                 │
│  recv_topk_ids shape = [8, 8]                                                   │
│  recv_topk_ids[:3] = [[8, 21, 30, 177, 211, 214, 246, 13],                      │
│                       [8, 21, 30, 177, 211, 214, 246, 13],   ← ALL IDENTICAL!   │
│                       [8, 21, 30, 177, 211, 214, 246, 13]]                      │
│                                                                                 │
│  recv_weights shape = [8, 8], sum = 20.0                                        │
│  expert_x shape = [8, 7168]           ← 8 received token embeddings             │
│                                                                                 │
│  DEDUP: total_recv=8, unique_tokens=8                                           │
│         (8 unique SOURCE tokens, all with same expert IDs!)                     │
│                                                                                 │
│  local_experts = [8, 21, 30, 13]      ← 4 experts in range [0-31]              │
│                  (177, 211, 214, 246 are on OTHER ranks)                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          AITER EXPERT COMPUTATION                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INPUT TO AITER:                                                                │
│    expert_x shape = [8, 7168]                                                   │
│    expert_topk_ids shape = [8, 8]  (GLOBAL IDs, same for all 8 entries!)       │
│    topk_weights shape = [8, 8]                                                  │
│                                                                                 │
│  🔴 PROBLEM: AITER thinks it has 8 tokens, each needing 8 experts computed!    │
│                                                                                 │
│  What AITER DOES (if expert_map filters correctly):                             │
│    For each of 8 entries × each of 8 expert IDs:                                │
│      - expert_map[8] = 8 (local) → compute                                      │
│      - expert_map[21] = 21 (local) → compute                                    │
│      - expert_map[30] = 30 (local) → compute                                    │
│      - expert_map[177] = -1 (not local) → skip/zero                             │
│      - expert_map[211] = -1 (not local) → skip/zero                             │
│      - expert_map[214] = -1 (not local) → skip/zero                             │
│      - expert_map[246] = -1 (not local) → skip/zero                             │
│      - expert_map[13] = 13 (local) → compute                                    │
│                                                                                 │
│  🔴 Each entry computes 4 experts, but should compute only 1!                  │
│     Entry 0 from Token X should use ONLY expert 8 (or whichever it was sent for)│
│     Instead, Entry 0 computes experts 8, 21, 30, AND 13!                        │
│                                                                                 │
│  EXPECTED: 8 entries × 1 expert = 8 expert computations                         │
│  ACTUAL:   8 entries × 4 experts = 32 expert computations (4x waste)           │
│                                                                                 │
│  OUTPUT:                                                                        │
│    fused_expert_output shape = [8, 7168]                                        │
│    fused_expert_output mean = -0.0100, std = 0.6530                             │
│                                                                                 │
│  🔴 BIGGER PROBLEM: Each entry's output is sum of 4 expert outputs!            │
│     Entry 0 = expert_8(x) + expert_21(x) + expert_30(x) + expert_13(x)         │
│     But it should be ONLY expert_K(x) where K is the specific expert for that  │
│     dispatch entry!                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            WEIGHT & REDUCE                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                         │
│    fused_expert_output shape = [8, 7168]                                        │
│    topk_weights shape = [8, 8]        ← RECEIVED weights!                       │
│    topk_ids shape = [8, 8]                                                      │
│                                                                                 │
│  🔴 PROBLEM: weight_and_reduce_impl.apply() multiplies by weights              │
│     But the fused_expert_output is already wrong (sum of 4 experts per entry)  │
│     AND the weights don't match (we have 8 weights but used 4 experts)         │
│                                                                                 │
│  EXPAND: 8 -> 8 (no change since unique_tokens=total_recv)                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MORI COMBINE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                         │
│    fused_expert_output shape = [8, 7168]                                        │
│    original_topk_ids shape = [1, 8]   ← THIS rank's token's expert choices     │
│                                                                                 │
│  ep_op.combine() sends results back to original token owners                    │
│                                                                                 │
│  OUTPUT:                                                                        │
│    combined_x shape = [8192, 7168]    ← Fixed-size buffer                       │
│    combined_x[0] mean = -0.0155, std = 0.8994                                   │
│    combined_x (full) mean = -0.0094, std = 8.1340                               │
│                                                                                 │
│  SLICE to output shape [1, 7168]:                                               │
│    output = combined_x[:1]                                                      │
│    output mean = -0.0155                                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

---

## THE ROOT CAUSE

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         🔴 FUNDAMENTAL MISMATCH 🔴                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  MORI dispatches: (token, expert) pairs                                         │
│    - 1 token with topk=8 → 8 dispatch entries (one per expert)                 │
│    - Each entry goes to the rank that owns that specific expert                 │
│    - Rank 0 receives entries for experts 8, 21, 30, 13 (4 entries)             │
│                                                                                 │
│  MORI returns: Full topk_ids and weights for each entry                         │
│    - Each entry has [8, 21, 30, 177, 211, 214, 246, 13] (all 8 experts)        │
│    - NOT which specific expert this entry is for!                               │
│                                                                                 │
│  AITER expects: Tokens that each need ALL their experts computed                │
│    - Input token → compute ALL topk experts → weighted sum → output             │
│    - Uses expert_map to filter to only local experts                            │
│                                                                                 │
│  THE PROBLEM:                                                                   │
│    - MORI gives 4 entries, each is for 1 specific expert                        │
│    - AITER treats each entry as a full token needing 4 local experts            │
│    - Result: 4 entries × 4 experts = 16 computations (should be 4!)            │
│    - Output: Each entry is sum of 4 expert outputs (should be 1!)              │
│                                                                                 │
│  WHY OUTPUT IS GARBAGE:                                                         │
│    Entry for expert 8:   output = expert_8(x) + expert_21(x) + expert_30(x) +  │
│                                   expert_13(x)  ← WRONG!                        │
│    Should be:            output = expert_8(x)   ← CORRECT                       │
│                                                                                 │
│    The weights applied don't fix this because they're applied to the wrong     │
│    sum. And even if weights were [1, 0, 0, ...], the other expert outputs      │
│    are still added to the result!                                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## SOLUTION OPTIONS

### Option 1: Per-Entry Expert Filtering
For each received entry, ONLY compute the single expert it was dispatched for.
Requires knowing which expert each entry is for (MORI doesn't provide this directly).

### Option 2: Aggregate Before AITER
Group all received entries by source token, then call AITER once per unique source
token with its hidden state. AITER then correctly computes all local experts.
This is what dedup tries to do, but it keeps ALL entries as "unique".

### Option 3: Custom Expert Kernel
Write a kernel that takes (token, single_expert_id) pairs and computes only that
one expert per entry. This is more efficient but requires new kernel development.

### Option 4: Use MORI's per-entry expert info
Check if MORI provides which specific expert each dispatch entry is for.
If so, create a per-entry expert_map or filter.


