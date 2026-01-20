# MORI-EP: Expert Parallelism Dispatch/Combine for AMD

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                           MORI-EP DISPATCH/COMBINE ARCHITECTURE                                   ║
║                              AMD MI300X Expert Parallelism                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝

                                    FORWARD PASS FLOW
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                             │
    │   INPUT: [M tokens, H hidden]          Router Output: [M, num_experts]                     │
    │         ────────────────────                ────────────────────────────                    │
    │                │                                      │                                    │
    │                │                                      │ topk selection                     │
    │                ▼                                      ▼                                    │
    │   ┌─────────────────────────────────────────────────────────────────────┐                  │
    │   │                     MORI DISPATCH (All-to-All)                       │                  │
    │   │   ─────────────────────────────────────────────────                  │                  │
    │   │   GPU 0         GPU 1         GPU 2         ...  GPU 7              │                  │
    │   │   experts       experts       experts            experts            │                  │
    │   │   [0-31]        [32-63]       [64-95]            [224-255]          │                  │
    │   │                                                                      │                  │
    │   │   Each GPU sends tokens to GPU that owns the selected expert         │                  │
    │   │   FP8 quantization option: 2x bandwidth savings                     │                  │
    │   │   Kernel types: IntraNode (XGMI) | InterNode (RDMA)                 │                  │
    │   └─────────────────────────────────────────────────────────────────────┘                  │
    │                                      │                                                      │
    │                                      ▼                                                      │
    │   ┌─────────────────────────────────────────────────────────────────────┐                  │
    │   │                     AITER EXPERT COMPUTE                             │                  │
    │   │   ─────────────────────────────────────────                          │                  │
    │   │   Each GPU runs MoE FFN on LOCAL experts only:                       │                  │
    │   │                                                                      │                  │
    │   │   recv_x ──► W1 (gate+up) ──► SiLU ──► W2 (down) ──► expert_out     │                  │
    │   │              [E, 2*I, H]              [E, H, I]                      │                  │
    │   │                                                                      │                  │
    │   │   E = 32 local experts (for EP8 with 256 total)                     │                  │
    │   │   Shuffled weight format for AITER ASM kernels                      │                  │
    │   └─────────────────────────────────────────────────────────────────────┘                  │
    │                                      │                                                      │
    │                                      ▼                                                      │
    │   ┌─────────────────────────────────────────────────────────────────────┐                  │
    │   │                     MORI COMBINE (All-to-All)                        │                  │
    │   │   ─────────────────────────────────────────────                      │                  │
    │   │   Each GPU sends expert outputs back to original token owner         │                  │
    │   │   Weighted sum: output += weight[k] * expert_out[k]                 │                  │
    │   │   BF16 output (always, even if FP8 dispatch)                        │                  │
    │   └─────────────────────────────────────────────────────────────────────┘                  │
    │                                      │                                                      │
    │                                      ▼                                                      │
    │   OUTPUT: [M tokens, H hidden]    (reduced across topk experts)                            │
    │                                                                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘


                              MORI API STRUCTURE
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                             │
    │  from mori.ops import EpDispatchCombineOp, EpDispatchCombineConfig, EpDispatchCombineKernelType
    │                                                                                             │
    │  # Configuration                                                                            │
    │  config = EpDispatchCombineConfig(                                                          │
    │      data_type=torch.bfloat16,                                                             │
    │      rank=0,                          # Current EP rank                                    │
    │      world_size=8,                    # EP size (must be power of 2)                       │
    │      hidden_dim=7168,                 # DeepSeek-R1 hidden size                            │
    │      max_num_inp_token_per_rank=4096, # Max tokens per dispatch                            │
    │      num_experts_per_rank=32,         # 256 / 8 = 32 local experts                         │
    │      num_experts_per_token=8,         # topk=8                                             │
    │      kernel_type=EpDispatchCombineKernelType.IntraNode,  # XGMI for single-node            │
    │      gpu_per_node=8,                  # MI300X: 8 GPUs per node                            │
    │  )                                                                                         │
    │                                                                                             │
    │  # Create operator                                                                         │
    │  ep_op = EpDispatchCombineOp(config)                                                       │
    │                                                                                             │
    │  # Dispatch                                                                                │
    │  result = ep_op.dispatch(input, weights, scales, indices)                                  │
    │  recv_x, recv_weights, recv_scale, recv_ids, recv_src_pos = result                        │
    │                                                                                             │
    │  # Expert compute (AITER)                                                                  │
    │  expert_out = aiter_experts(recv_x, w1, w2, recv_ids, recv_weights)                       │
    │                                                                                             │
    │  # Combine                                                                                 │
    │  output, output_scale = ep_op.combine(expert_out, weights, indices, call_reset=True)      │
    │                                                                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘


                              KERNEL TYPE SELECTION
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                             │
    │   EpDispatchCombineKernelType     Use Case                     Transport                   │
    │   ─────────────────────────────   ────────────────────────     ─────────────────           │
    │   IntraNode                       Single node (8× MI300X)      XGMI (800 GB/s)             │
    │   InterNode                       Multi-node                   RDMA (400 Gb/s IB)          │
    │   InterNodeV1                     Multi-node (alt impl)        RDMA                        │
    │   InterNodeV1LL                   Multi-node low-latency       RDMA optimized              │
    │                                                                                             │
    │   Selection logic in vLLM:                                                                 │
    │   - EP8 single node: IntraNode (XGMI)                                                      │
    │   - EP16+ multi-node: InterNode (RDMA)                                                     │
    │                                                                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘


                              PERFORMANCE CHARACTERISTICS
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                             │
    │   From MORI benchmarks (8× MI300X, XGMI):                                                  │
    │                                                                                             │
    │   Dispatch:                                                                                │
    │   - 128 tokens: ~35 µs, 307 GB/s effective bandwidth                                       │
    │   - 4096 tokens: ~180 µs, scales linearly                                                  │
    │                                                                                             │
    │   Combine:                                                                                 │
    │   - 128 tokens: ~47 µs, 330 GB/s effective bandwidth                                       │
    │   - 4096 tokens: ~220 µs, scales linearly                                                  │
    │                                                                                             │
    │   Total MoE layer (dispatch + compute + combine):                                          │
    │   - Compute dominates for large expert sizes (2048 intermediate)                           │
    │   - Communication ~10-15% overhead at high batch sizes                                     │
    │                                                                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘


                              vLLM INTEGRATION POINTS
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                             │
    │   Files:                                                                                   │
    │   ├── vllm/model_executor/layers/fused_moe/                                                │
    │   │   ├── mori_utils.py              # MoriEpConfig, create_mori_ep_op()                  │
    │   │   ├── mori_prepare_finalize.py   # MoriPrepareAndFinalize class                       │
    │   │   └── oracle/unquantized.py      # AITER_MORI_EP backend selection                    │
    │   ├── vllm/envs.py                   # VLLM_MORI_EP_USE_FP8_DISPATCH                       │
    │   └── benchmarks/kernels/            # benchmark_mori_ep_moe.py                           │
    │                                                                                             │
    │   Backend selection:                                                                       │
    │   - UnquantizedMoeBackend.AITER_MORI_EP                                                    │
    │   - FusedMoEParallelConfig.use_mori_ep_kernels                                             │
    │   - all2all_backend="mori_ep"                                                             │
    │                                                                                             │
    │   Dependencies:                                                                            │
    │   - pip install mori (from https://github.com/ROCm/mori)                                  │
    │   - apt install libopenmpi-dev libpci-dev                                                 │
    │   - LD_LIBRARY_PATH must include torch lib path                                           │
    │                                                                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Notes

### Build Requirements
- `libopenmpi-dev` - MPI for multi-GPU communication
- `libpci-dev` - PCIe topology detection
- `--no-build-isolation` flag needed for pip install

### Runtime Requirements
- `LD_LIBRARY_PATH` must include PyTorch lib path
- MORI shmem must be initialized before creating EP operators
- world_size must be power of 2 and divisible by gpu_per_node

### Known Limitations
- Single-GPU testing not supported (requires EP8+)
- Must run with `torchrun --nproc_per_node=8` or equivalent
- Assertion: `IsPowerOf2(config.gpuPerNode) && (config.worldSize % config.gpuPerNode == 0)`

### Performance Tuning
- `warp_num_per_block`: Default 8, tune for occupancy
- `block_num`: Default 80, tune for GPU utilization
- `use_fp8_dispatch`: Enable for 2x bandwidth savings (requires FP8 support)

### Related Backends Comparison
| Backend | Transport | Platform | Notes |
|---------|-----------|----------|-------|
| MORI-EP | XGMI/RDMA | AMD MI300X | True EP, AITER compute |
| DeepEP-HT | NVLink | NVIDIA | High-throughput mode |
| DeepEP-LL | NVLink | NVIDIA | Low-latency mode |
| PPLX | NVLink | NVIDIA | PPLX kernels |

