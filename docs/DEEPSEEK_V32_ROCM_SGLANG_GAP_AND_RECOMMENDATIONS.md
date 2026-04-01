# DeepSeek V3.2 on ROCm: Gap Analysis vs SGLang/ATOM and Recommendations

**Repository:** [maeehart/vllm](https://github.com/maeehart/vllm) (`maeehart/deepseek-v32-fusion-fix`)  
**Audience:** Engineers prioritizing AMD MI355X / ROCm work relative to SGLang’s DeepSeek V3.2 and MLA stack  
**Date:** April 2026  

---

## 1. Executive summary

- **SGLang** has invested heavily in **NSA (Native Sparse Attention)** on AMD via **TileLang** and **AITER** backends, plus **FP8 prefill** paths integrated with **prefix (radix) cache**, **context parallel**, and **MoE router** tuning.  
- **vLLM** (including this fork) does **not** ship **TileLang**; treating TileLang parity as out of scope avoids a multi-month compiler/dependency program.  
- This fork already lands several **ROCm-specific performance and correctness** changes for **DeepSeek V3.2** and **persistent AITER MLA decode with FP8 scales**.  
- The largest **remaining** wins that are **not** TileLang-dependent are: **MoEGate `tgemm.mm`-style routing**, **AITER FP8 MLA prefill** (upstream vLLM gates FP8 prefill query quantization to GB200), **context parallel for NSA on AMD**, and **continued correctness** (KV scales, graph padding, CK shape fallbacks where applicable).  
- **“FP8 prefill + prefix cache”** is **not** a small extension of **persistent FP8 MLA decode** alone: prefill today uses `MLACommonImpl` (FlashInfer / FA / etc.), while persistent FP8 is wired to `**mla_decode_fwd`**.

---

## 2. Scope and exclusions


| In scope                                        | Out of scope                                                            |
| ----------------------------------------------- | ----------------------------------------------------------------------- |
| AMD ROCm, MI300/MI355-class, AITER-backed paths | **TileLang** NSA kernels (no TileLang in vLLM)                          |
| MLA, sparse MLA (DeepSeek V3.2), FP8 KV         | Porting SGLang’s TileLang `sparse_fwd` / default `tilelang` NSA backend |
| Prefix caching, chunked prefill, decode         | Feature parity with every SGLang env flag                               |


---

## 3. What SGLang and ATOM do (TileLang-free summary)

Reference searches (public PRs):

- **SGLang (AMD + DeepSeek / MLA):** MoEGate router via `**tgemm.mm`** ([#21657](https://github.com/sgl-project/sglang/pull/21657)), **FP8 prefill + radix cache** for DeepSeek-style models ([#20187](https://github.com/sgl-project/sglang/pull/20187)), **context parallel** for DSV3.2 ([#19975](https://github.com/sgl-project/sglang/pull/19975)), **MI355 accuracy** workaround for a bad CK GEMM shape ([#20840](https://github.com/sgl-project/sglang/pull/20840)), NSA metadata fixes with FP8 KV ([#20606](https://github.com/sgl-project/sglang/pull/20606)), MLA `**nhead < 16`** via head-repeat ([#21213](https://github.com/sgl-project/sglang/pull/21213)).  
- **ATOM:** **Sparse prefill MLA** + indexer rope fixes ([#109](https://github.com/ROCm/ATOM/pull/109)); GLM-5 / shared DeepSeek-style paths ([#289](https://github.com/ROCm/ATOM/pull/289)).

**TileLang** in SGLang is a **separate** NSA backend (`--nsa-prefill-backend tilelang`); it is **not** implemented through AITER. AITER and TileLang are alternative backends.

---

## 4. maeehart-vllm fork: relevant changes (from `git log`)

The branch `maeehart/deepseek-v32-fusion-fix` tracks **github.com/maeehart/vllm** and includes upstream vLLM plus ROCm-focused commits. Representative fork-specific work (newest first among custom commits):


| Commit                | Theme                                                                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `d71fbe9`             | ROCm: guard `concat_mla_q.cuh` includes for HIP (`cuda_bf16.h` fix)                                                                           |
| `34d3434`             | ROCm: early AITER Gluon env, DeepSeek V3.2 Docker base + test helpers                                                                         |
| `2c5982a`             | **[ROCm][Perf] Enable persistent MLA kernel with fp8 support** — `_aiter_ops.py`, `rocm_aiter_mla.py`                                         |
| `9ba96bb`             | [ROCm][Perf] Replace `moe_sum` custom op with `torch.sum` for Inductor fusion                                                                 |
| `3f3f9f9`             | [ROCm] Pre-build AITER JIT kernels during Docker image build                                                                                  |
| `1890ac0`             | [ROCm] Enable CUDA graph support for sparse attention indexer on ROCm                                                                         |
| `63088e9`             | [ROCm] Pin base image to ROCm 7.2 for MI355X                                                                                                  |
| `dab511c`             | [ROCm] Upgrade AITER from source with gfx950 MoE fix in `Dockerfile.rocm`                                                                     |
| `66a6108` / `f14f07d` | [Bugfix][ROCm] Paged MQA logits / `lru_cache` guards                                                                                          |
| `c466eb7`             | **[ROCm][Perf] Apply DSV3.2 sparse MLA optimizations for MI355X** — `deepseek_v2.py`, `rocm_aiter_mla_sparse.py`, `rocm_aiter_mla_sparse` ops |
| `46f28e0`             | [ROCm] `mla_asm.csv` fix and build script for MI355X                                                                                          |
| `9f04b04`             | [ROCm][Perf] RMSNorm+Quant fusion for gfx950 (MI355X) with DSV3.2                                                                             |


Upstream merges on the same branch also include **FP8 MLA / KV scale fixes** (`577df69`), **zero-init MLA buffers for graph padding** (`ef2c4f7`), etc.

**Net:** This fork is already optimized for **ROCm 7.2**, **AITER** (including gfx950 MoE), **persistent MLA decode + FP8 scales**, **sparse MLA / indexer** on ROCm, and **DeepSeek V3.2**-specific fusion and build scripts.

---

## 5. Architectural findings in this codebase

### 5.1 Persistent FP8 MLA = decode path

- `AiterMLAImpl` implements `**forward_mqa`** only for **decode**, calling `rocm_aiter_ops.mla_decode_fwd` with optional **persistent** work buffers (`work_metadata`, `work_indptr`, …) and `**q_scale` / `kv_scale`** when AITER’s `mla_decode_fwd` supports them (`_check_aiter_mla_fp8_support()` in `vllm/_aiter_ops.py`).  
- **Prefill** is not implemented in `rocm_aiter_mla.py` beyond inheriting `**MLACommonImpl`**, which selects **TRT-LLM ragged**, **FlashInfer**, **cuDNN**, or **FlashAttention varlen** (typical ROCm path: FA via `aiter`).

### 5.2 FP8 prefill query quantization is effectively off on ROCm

- `backend_supports_prefill_query_quantization()` in `mla_attention.py` requires **device capability 100 (GB200)** and FlashInfer or TRT-LLM ragged prefill.  
- Therefore `**determine_prefill_query_data_type`** does not set FP8 prefill for **ROCm** today, even with `--attention-config.use_prefill_query_quantization` and FP8 KV.  
- Code paths such as `**_compute_prefill_context`** (prefix chunks, `cp_gather_cache`, FP8 casts) exist but are gated by `**use_fp8_prefill**`, which stays false under current rules.

### 5.3 Prefix / chunked prefill structure already exists

- FlashInfer-style `**kv_indptr` vs `qo_indptr**` for main vs context chunks and chunked prefill metadata are already in `**MLACommonMetadataBuilder**` / `**_build_fi_prefill_wrappers**`.  
- **SGLang’s “radix + FP8 prefill”** work is **not** a duplicate of that plumbing; it assumes an **AITER FP8 prefill** kernel and correct metadata (`kv_indptr`-style) for batched extends. This fork **does not** yet expose an AITER `**mla_fp8_prefill`**-class API in `_aiter_ops.py` (only `**mla_decode_fwd**` is wrapped for MLA).

---

## 6. Gap table (no TileLang): SGLang ideas vs this fork


| Idea (SGLang / ecosystem)            | In this fork / vLLM?                         | Gap severity             |
| ------------------------------------ | -------------------------------------------- | ------------------------ |
| TileLang NSA default / sparse_fwd    | No                                           | **Excluded**             |
| Persistent MLA decode + FP8 scales   | Yes (`2c5982a`, `_aiter_ops`)                | **Closed**               |
| DSV3.2 sparse MLA MI355X tuning      | Yes (`c466eb7` and related)                  | **Reduced**              |
| FP8 **prefill** MLA + radix (AITER)  | No (prefill not AITER FP8; gating on GB200)  | **High**                 |
| MoEGate `**tgemm.mm`** / tuned GEMM  | Unclear without audit; not in listed commits | **Medium–high**          |
| NSA context parallel (AMD)           | Not in fork-specific commits                 | **High** (large feature) |
| CK GEMM shape workaround (7168×2304) | Audit `fp8`/CK paths on gfx950               | **Medium** if same bug   |
| Triton MLA FP8 KV (portable)         | Partially overlaps upstream; ROCm caveats    | **Medium**               |


---

## 7. Prioritized recommendations (difficulty vs impact)

**Tier A — High ROI, lower effort**

1. **MoEGate router GEMM via AITER tuned GEMM (`tgemm.mm`)**
  - Aligns with SGLang [#21657](https://github.com/sgl-project/sglang/pull/21657). Small surface in DeepSeek MoE routing; large win if profiling shows router-bound layers.
2. **Audit gfx950 CK fallbacks for known-bad shapes**
  - If the same (N,K) as SGLang [#20840](https://github.com/sgl-project/sglang/pull/20840) appears in your FP8 GEMM dispatch, add an explicit Triton (or safe) path.
3. **Keep merging upstream ROCm fixes**
  - KV scale consistency, zero-init MLA buffers for graphs, etc., already present on branch—stay rebased.

**Tier B — High impact, medium effort**

1. **Enable FP8 MLA prefill on ROCm (non–TileLang)**
  - Extend `**backend_supports_prefill_query_quantization()`** (or add a dedicated flag) for `**ROCM_AITER_MLA**` when AITER exposes `**mla_fp8_prefill**` (or equivalent).  
  - Implement `**_run_prefill_new_tokens_***` / context chunk entry points that call the AITER FP8 prefill kernel; then validate **prefix chunks** with the same `**kv_indptr`** semantics FlashInfer already uses.  
  - This is **not** automatic from “persistent FP8 decode”; it is new prefill integration.
2. **NSA context parallel on AMD**
  - Matches SGLang’s direction (long-context throughput). Large scheduler/parallelism surface area.

**Tier C — Medium impact, higher effort or dependency-heavy**

1. **MXFP4 / aggressive KV compression on MI355**
  - Overlaps open upstream work; validate accuracy and allocator pressure.
2. **Triton MLA FP8 KV improvements**
  - Cross-check with upstream Triton MLA PRs; on ROCm, watch PDL / CUDA graph limits.

---

## 8. FP8 prefill + prefix cache: how it applies here


| Question                                             | Answer for maeehart-vllm                                                                                                                                                 |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Does persistent FP8 MLA decode imply FP8 prefill?    | **No.** Decode uses `mla_decode_fwd`; prefill uses `MLACommonImpl` (FA/FI/etc.).                                                                                         |
| Is prefix metadata already modeled?                  | **Yes** (chunked context, `kv_indptr` for context chunks in FlashInfer path).                                                                                            |
| What is missing for SGLang-like FP8 prefill + radix? | **(1)** Enable FP8 prefill dtype path on ROCm. **(2)** Wire prefill to **AITER FP8 prefill** (or equivalent). **(3)** Re-validate prefix chunk metadata for that kernel. |


---

## 9. References

- SGLang PRs (search): [DeepSeek V3.2 + AMD](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+deepseek+v3.2+AMD), [MLA + AMD](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+MLA+AMD+)  
- ATOM: [DeepSeek v3.2 PRs](https://github.com/ROCm/ATOM/pulls?q=is%3Apr++deepseek+v3.2+)  
- Fork remote: `https://github.com/maeehart/vllm.git`

---

## 10. Document history


| Version | Notes                                                                                                               |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| 1.0     | Initial write: gap analysis, TileLang exclusion, fork commit review, FP8 prefill vs persistent decode clarification |


