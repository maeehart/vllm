# MORI EP + AITER Debug Notes

## Current Status

**Branch:** `mori-ep-aiter-fix`  
**Issue:** Output is still garbage after multiple fix attempts.

### What We've Tried

1. **Dedup by topk_ids** → Wrong (different tokens can have same experts)
2. **Dedup by src_token_pos** → Still garbage (TP replication issue)
3. **Dedup by local_token_idx** → Still garbage
4. **Pass expert_map instead of expert_mask** → Crash (AITER can't handle -1)
5. **Convert global→local IDs + zero non-local weights** → Current fix (v6), needs testing

### Key Files Modified

- `vllm/model_executor/layers/fused_moe/mori_prepare_finalize.py` - MORI dispatch/combine logic
- `vllm/model_executor/layers/fused_moe/layer.py` - expert_map property
- `test_aiter_expert_map.py` - Test script for AITER interface

---

## Commands

### Start Server (MORI EP + AITER)

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MOE=1 \
MORI_SHMEM_HEAP_SIZE=12G \
vllm serve deepseek-ai/DeepSeek-R1 \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 72000 \
  --trust-remote-code \
  --enable-expert-parallel \
  --enforce-eager \
  --all2all-backend mori_ep \
  --kv-cache-dtype fp8 \
  2>&1 | tee mori_ep_test.log
```

### Start Server with Debug Logging

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MOE=1 \
MORI_SHMEM_HEAP_SIZE=12G \
VLLM_MORI_DEBUG=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 72000 \
  --trust-remote-code \
  --enable-expert-parallel \
  --enforce-eager \
  --all2all-backend mori_ep \
  --kv-cache-dtype fp8 \
  2>&1 | tee mori_ep_debug.log
```

### Start Server with Value Tracing (Rank 0 only)

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MOE=1 \
MORI_SHMEM_HEAP_SIZE=12G \
VLLM_MORI_TRACE=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 72000 \
  --trust-remote-code \
  --enable-expert-parallel \
  --enforce-eager \
  --all2all-backend mori_ep \
  --kv-cache-dtype fp8 \
  2>&1 | tee mori_ep_trace.log
```

### Test Inference

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-R1", "prompt": "What is 2+2?", "max_tokens": 20, "temperature": 0}'
```

### Reference Server (Non-EP, should work correctly)

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 72000 \
  --trust-remote-code \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  2>&1 | tee reference_no_ep.log
```

---

## Benchmarks

### MORI EP Kernel Benchmark

```bash
cd /workspace/vllm
python benchmarks/kernels/test_mori_ep_kernel_markus.py \
  --num-tokens 8192 \
  --hidden-size 7168 \
  --num-experts 256 \
  --topk 8 \
  --num-iters 100
```

### MoE Benchmark (General)

```bash
python benchmarks/kernels/benchmark_moe.py \
  --model deepseek-ai/DeepSeek-R1 \
  --batch-size 8192 \
  --tp-size 8
```

---

## Debug Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MORI_DEBUG=1` | Enable debug logging from all ranks |
| `VLLM_MORI_TRACE=1` | Enable detailed value tracing (Rank 0 only, 10 steps) |
| `VLLM_ROCM_USE_AITER=1` | Use AITER for optimized ops |
| `VLLM_ROCM_USE_AITER_MOE=1` | Use AITER for MoE computation |
| `MORI_SHMEM_HEAP_SIZE=12G` | MORI shared memory heap size |

---

## Key Findings

### The Root Cause (Identified but not fully fixed)

1. **AITER expects `expert_mask` (binary)** but uses it incorrectly for EP:
   - `expert_mask[global_id] = 1` for local experts
   - AITER interprets the VALUE (1) as the local index
   - For expert 5: mask[5]=1 → AITER uses w1[1] instead of w1[5]

2. **AITER can't handle -1 in expert_map**:
   - Triton uses `expert_map[global_id] = local_id` or `-1` for non-local
   - AITER crashes with memory access fault when it sees -1

3. **Current fix (v6)** tries to work around by:
   - Converting global IDs to local IDs in MORI
   - Zeroing weights for non-local experts
   - Passing `expert_map=None` to AITER

### Trace Analysis Points

Look for these in trace logs:
- **STEP 1**: Dispatch input (should have reasonable std ~0.1)
- **STEP 2**: After receive (65536 tokens for 8192 input × 8 topk)
- **STEP 3**: After dedup (should be 8192 unique tokens)
- **STEP 4**: Output to AITER
- **STEP 5**: AITER output (if std << input std, AITER is broken)
- **STEP 6-10**: Weight reduction, expansion, combine
- **STEP 11**: Final output

---

## Git Commands

```bash
# Push changes to fork
git add -A
git status  # Check for large files!
git commit -m "description"
git push myfork mori-ep-aiter-fix

# Check for large files in history
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort -rnk2 | head -10
```

---

## Next Steps

1. Test current fix (v6) with inference
2. If still broken, add more tracing inside AITER call
3. Consider modifying AITER wrapper to handle EP differently
4. Or bypass AITER entirely for MORI EP (use Triton instead)

