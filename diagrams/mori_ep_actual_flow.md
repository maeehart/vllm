# MORI-EP Data Flow (MoE Layer with TP + EP)

## Context
- **Model**: DeepSeek-R1 (256 experts, topk=8)
- **Setup**: 8 GPUs with TP=8 + EP=8, each owns 32 experts
- **Mode**: Decode (1 token per rank, but with TP=8, ALL ranks have SAME token!)
- **Rank 0** owns experts **0-31**

---

## ⚠️ KEY INSIGHT: TP + EP Interaction

With **Tensor Parallelism (TP=8)**, all 8 GPUs process the **SAME token**!
- All 8 ranks dispatch the **identical** token with **identical** routing
- Each expert-owning rank receives the same token **8 times** from 8 different source ranks
- `src_token_pos = src_rank × max_tokens + local_token_idx`
- Same `local_token_idx` across ranks = same logical token

---

## CURRENT IMPLEMENTATION FLOW

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STEP 1: DISPATCH (on each rank)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT:                                                                         │
│    tokens shape = [M, 7168]             ← M tokens from this rank               │
│    topk_ids shape = [M, 8]              ← 8 selected experts per token          │
│    topk_weights shape = [M, 8]          ← router weights                        │
│                                                                                 │
│  MORI ep_op.dispatch() routes tokens to expert-owning ranks:                    │
│    - Token T with topk=[E0, E1, ..., E7] → sent to ranks owning these experts   │
│    - Each rank receives entries for its local experts only                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
                                   ALL-TO-ALL
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STEP 2: RECEIVE & SLICE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  MORI returns fixed-size buffers; slice to valid tokens:                        │
│                                                                                 │
│  total_recv_tokens = dispatch_result[4]   ← GPU tensor with actual count        │
│  num_valid = total_recv_tokens.item()     ← e.g., 8 for decode, 65536 prefill   │
│                                                                                 │
│  expert_x = recv_x[:num_valid]            ← [N_recv, H] token embeddings        │
│  recv_topk_ids = recv_topk_ids[:num_valid]← [N_recv, 8] all 8 expert IDs        │
│  recv_weights = recv_weights[:num_valid]  ← [N_recv, 8] all 8 weights           │
│                                                                                 │
│  src_token_pos = ep_op.get_dispatch_src_token_pos()[:num_valid]                 │
│    ← Format: src_rank × max_tokens_per_rank + local_token_idx                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: DEDUP BY LOCAL TOKEN INDEX ✅                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  🔴 WHY DEDUP IS NEEDED:                                                        │
│  With TP=8, same logical token is dispatched from 8 source ranks.               │
│  If token has 2 local experts here, we receive 8 × 2 = 16 entries!              │
│  Without dedup: AITER processes 16 times → 16× the correct value!              │
│                                                                                 │
│  ✅ THE FIX: Dedup by local_token_idx                                           │
│                                                                                 │
│  # Infer max_tokens_per_rank from position spacing                              │
│  sorted_pos, _ = src_token_pos.sort()                                           │
│  diffs = sorted_pos[1:] - sorted_pos[:-1]                                       │
│  max_diff = diffs.max().item()                                                  │
│  max_tokens_per_rank = max_diff if max_diff > 1000 else 8192                    │
│                                                                                 │
│  # Compute local token index                                                    │
│  local_token_idx = src_token_pos % max_tokens_per_rank                          │
│                                                                                 │
│  # Dedup by local index (merges TP copies)                                      │
│  unique_local_idx, inverse_indices = torch.unique(local_token_idx, ...)         │
│  num_unique = unique_local_idx.shape[0]                                         │
│                                                                                 │
│  EXAMPLE (Decode with TP=8):                                                    │
│    src_token_pos = [100, 8292, 16484, 24676, 32868, 41060, 49252, 57444]        │
│                  = [0×8192+100, 1×8192+100, 2×8192+100, ...]                    │
│    local_token_idx = [100, 100, 100, 100, 100, 100, 100, 100]  ← ALL SAME!     │
│    unique_local_idx = [100]  → num_unique = 1                                   │
│    inverse_indices = [0, 0, 0, 0, 0, 0, 0, 0]                                   │
│                                                                                 │
│  if num_unique < num_valid:                                                     │
│    # Find first occurrence of each unique local token                           │
│    first_indices = scatter_reduce(arange, inverse_indices, reduce='amin')       │
│                                                                                 │
│    # Keep only unique entries                                                   │
│    expert_x = expert_x[first_indices]          ← [1, 7168] after dedup          │
│    recv_weights = recv_weights[first_indices]  ← [1, 8]                         │
│    recv_topk_ids = recv_topk_ids[first_indices]← [1, 8]                         │
│                                                                                 │
│    # Store for later expansion                                                  │
│    self._dedup_inverse_indices = inverse_indices                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 4: AITER EXPERT COMPUTATION                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT TO AITER (after dedup):                                                  │
│    hidden_states = expert_x        ← [N_unique, H]                              │
│    topk_ids = recv_topk_ids        ← [N_unique, 8] GLOBAL expert IDs            │
│    topk_weights = recv_weights     ← [N_unique, 8]                              │
│    expert_map[global_id] → local_id or -1 (filters to local experts)            │
│                                                                                 │
│  AITER COMPUTES (per unique token):                                             │
│    For token with topk_ids = [E0, E1, ..., E7]:                                 │
│      output = Σ expert_map[Ei] != -1 ? expert_Ei(x) * weight[i] : 0             │
│                                                                                 │
│    Example: topk_ids=[79, 81, 108, 120, 161, 179, 3, 30], rank_offset=160       │
│      local_experts = [161, 179] (experts 160-191 on this rank)                  │
│      output = expert_161(x) * w[4] + expert_179(x) * w[5]                       │
│                                                                                 │
│  OUTPUT:                                                                        │
│    fused_expert_output = [N_unique, H]  ← weighted sum of LOCAL experts         │
│                                                                                 │
│  ✅ CORRECT: Each unique token processed ONCE with all its local experts!      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STEP 5: WEIGHT & REDUCE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  weight_and_reduce_impl.apply():                                                │
│    - For AITER: weights already applied during expert computation               │
│    - Returns fused_expert_output unchanged (or with minor adjustments)          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STEP 6: EXPAND FOR COMBINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  If dedup was applied, expand results back to original count:                   │
│                                                                                 │
│  if self._dedup_inverse_indices is not None:                                    │
│    fused_expert_output = fused_expert_output[inverse_indices]                   │
│    # [N_unique, H] → [N_recv, H]                                                │
│    # Example: [1, 7168] → [8, 7168] (8 copies of same result)                   │
│                                                                                 │
│  ✅ All copies have identical results (same token, same computation)            │
│  Combine will route each back to its source rank correctly.                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            STEP 7: MORI COMBINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  combine_result = ep_op.combine(                                                │
│    input=fused_expert_output,     ← [N_recv, H] expert outputs                  │
│    weights=None,                  ← AITER already applied weights               │
│    indices=original_topk_ids,     ← [M_orig, 8] THIS rank's original tokens     │
│  )                                                                              │
│                                                                                 │
│  COMBINE DOES:                                                                  │
│    - Routes expert outputs back to original token owners                        │
│    - Each source rank receives partial results from destination ranks           │
│    - Sums contributions from all expert-owning ranks                            │
│                                                                                 │
│  OUTPUT:                                                                        │
│    combined_x = [max_tokens, H]   ← Fixed-size buffer                           │
│    output = combined_x[:M_orig]   ← Slice to actual token count                 │
│                                                                                 │
│  RESULT: Each original token gets correct weighted sum of all 8 experts!        │
│    output[i] = Σ(j=0..7) expert_topk[j](x[i]) * weight[i,j]                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## DEDUP EXAMPLES

### Example 1: Decode with TP=8, 1 token per rank

```
BEFORE DEDUP:
  num_valid = 8  (8 entries from 8 source ranks, all same logical token)
  src_token_pos = [100, 8292, 16484, 24676, 32868, 41060, 49252, 57444]
  local_token_idx = src_token_pos % 8192 = [100, 100, 100, 100, 100, 100, 100, 100]
  
AFTER DEDUP:
  num_unique = 1
  expert_x shape: [8, 7168] → [1, 7168]
  
AFTER EXPAND:
  fused_expert_output shape: [1, 7168] → [8, 7168] (8 identical copies)
```

### Example 2: Prefill with TP=8, 8192 tokens per rank

```
BEFORE DEDUP:
  num_valid = 65536  (8 ranks × 8192 tokens, all TP-replicated)
  Each unique token appears 8 times (once per source rank)
  local_token_idx has 8192 unique values, each appearing 8 times
  
AFTER DEDUP:
  num_unique = 8192
  expert_x shape: [65536, 7168] → [8192, 7168]
  
AFTER EXPAND:
  fused_expert_output shape: [8192, 7168] → [65536, 7168]
```

### Example 3: Non-TP mode (EP only)

```
BEFORE DEDUP:
  num_valid = 8192  (8192 unique tokens from various ranks)
  src_token_pos all different AND local_token_idx all different
  
AFTER DEDUP:
  num_unique = 8192  (no dedup, all unique)
  No expansion needed
```

---

## CODE REFERENCE

File: `vllm/model_executor/layers/fused_moe/mori_prepare_finalize.py`

### Key Variables

```python
# Source token position: src_rank × max_tokens + local_idx
src_token_pos = self.ep_op.get_dispatch_src_token_pos()

# Infer max_tokens_per_rank from position spacing
sorted_pos, _ = src_token_pos_valid.sort()
diffs = sorted_pos[1:] - sorted_pos[:-1]
max_diff = diffs.max().item()
max_tokens_per_rank = max_diff if max_diff > 1000 else 8192

# Compute local token index for dedup
local_token_idx = src_token_pos_valid % max_tokens_per_rank

# Dedup by local index
unique_local_idx, inverse_indices = torch.unique(local_token_idx, return_inverse=True)

# First occurrence indices
first_indices = torch.empty(num_unique, dtype=torch.long, device=device)
first_indices.fill_(num_valid)
first_indices.scatter_reduce_(0, inverse_indices, arange, reduce='amin')

# Keep unique entries only
expert_x = expert_x[first_indices]
recv_weights = recv_weights[first_indices]
recv_topk_ids = recv_topk_ids[first_indices]

# Store for expansion later
self._dedup_inverse_indices = inverse_indices
```

### Expansion in finalize

```python
# In _finalize_impl, after AITER computation:
if self._dedup_inverse_indices is not None:
    fused_expert_output = fused_expert_output[self._dedup_inverse_indices]
```

---

## WHY THIS WORKS

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORRECTNESS ARGUMENT                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. TP means all ranks have IDENTICAL tokens at each position                   │
│     → Same token dispatched to same experts with same hidden states             │
│                                                                                 │
│  2. Dedup by local_token_idx groups TP copies of same logical token             │
│     → Process once instead of 8× (or N× for TP=N)                               │
│                                                                                 │
│  3. AITER correctly computes weighted sum of LOCAL experts                      │
│     → Each destination rank contributes its partial sum                         │
│                                                                                 │
│  4. Expand creates copies for combine routing                                   │
│     → Same result sent back to all 8 source ranks                               │
│                                                                                 │
│  5. Combine sums partial results from all destination ranks                     │
│     → Each source gets: Σ(all experts) expert(x) * weight                       │
│                                                                                 │
│  RESULT: Correct MoE output with 8× less redundant computation!                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## DEBUG ENVIRONMENT VARIABLE

Set `VLLM_MORI_DEBUG=1` to enable detailed logging:

```bash
VLLM_MORI_DEBUG=1 python ...
```

Output includes:
- `[MORI SRC_POS DEBUG]` - Source token positions
- `[MORI DEDUP]` - Dedup statistics (total_recv, unique_local_tokens, max_tokens_per_rank)
- `[MORI COMBINE DEBUG]` - Combine input/output shapes and statistics
