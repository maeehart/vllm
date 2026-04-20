# SPDX-License-Identifier: Apache-2.0
"""Optimized W2 MoE GEMM kernel for MXFP4 weights + FP8 activations.

Provides _moe_stage2_weighted: a Triton kernel that performs the W2 GEMM
and applies routing weights (gammas) inline, avoiding a separate multiply.

moe_w2_stage2_weighted() is a drop-in replacement for the W2 moe_gemm_a8w4
call inside triton_kernel_fused_mxfp4_w4a8_experts.  It keeps the same
inputs/outputs contract so that routing, W1, and all non-MoE code paths
remain completely untouched.

Enable with VLLM_USE_FUSED_MOE=1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _unswizzle_mx_scale_cdna4(
    x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
    N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR,
                  MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x


@triton.jit
def _moe_stage2_weighted(
    Y,                # [split_k, M_sorted, N2] output, expert-sorted
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,                # [M_sorted, K2] intermediate, FP8
    stride_x_m,
    stride_x_k,
    W,                # [E, K2_packed, N2] weight, MXFP4
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,         # [E, SK*32, N2//32] scale, CDNA4
    stride_ws_e,
    stride_ws_k,
    stride_ws_n,
    X_static_scale,   # scalar fp32
    Gammas,           # [M_sorted] routing weights, bf16
    ExptHist,
    ExptOffs,
    ExptData,
    N, K,
    grid_m, grid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
):
    """Stage2 GEMM with routing weights applied inline.

    Writes to expert-sorted output buffer (no atomics).
    Compatible with CUDA graph capture.
    Uses reduce_grouped afterwards for topk reduction.
    """
    MX_PACK: tl.constexpr = 32
    MX_SBK: tl.constexpr = BLOCK_K // MX_PACK
    WKD: tl.constexpr = 2
    PBK: tl.constexpr = BLOCK_K // WKD
    NKP: tl.constexpr = 32
    PMB: tl.constexpr = MX_SBK * NKP
    SBN: tl.constexpr = BLOCK_N // NKP

    pid = tl.program_id(0)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M_expert = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)

    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_m_wrap = offs_m % M_expert
    mask_m = offs_m < M_expert

    offs_x_m = (start_m + offs_m_wrap).to(tl.int64)
    offs_x_k = tl.arange(0, BLOCK_K)
    XPtrs = X + offs_x_m[:, None] * stride_x_m + offs_x_k[None, :].to(tl.int64) * stride_x_k

    W_e = W + expt_id.to(tl.int64) * stride_w_e
    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n_wrap = offs_w_n % N
    offs_w_k = tl.arange(0, PBK)
    WPtrs = (W_e + offs_w_k[:, None].to(tl.int64) * stride_w_k
             + offs_w_n_wrap[None, :].to(tl.int64) * stride_w_n)

    WS_e = WMxScale + expt_id.to(tl.int64) * stride_ws_e
    offs_ws_n = (pid_n * SBN + tl.arange(0, SBN)) % N
    offs_ws_k = tl.arange(0, PMB)
    WSPtrs = (WS_e + offs_ws_k[None, :].to(tl.int64) * stride_ws_k
              + offs_ws_n[:, None].to(tl.int64) * stride_ws_n)

    x_scales = tl.full((BLOCK_M, MX_SBK), 127, dtype=tl.uint8)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iter = tl.cdiv(K, BLOCK_K)
    if not EVEN_K:
        num_k_iter -= 1

    for ki in range(num_k_iter):
        x = tl.load(XPtrs, mask=mask_m[:, None])
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        ws = _unswizzle_mx_scale_cdna4(
            tl.load(WSPtrs, cache_modifier=W_CACHE_MODIFIER),
            BLOCK_N, MX_SBK)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w, ws, "e2m1",
                            acc=acc, fast_math=True)
        XPtrs += BLOCK_K * stride_x_k
        WPtrs += PBK * stride_w_k
        WSPtrs += PMB * stride_ws_k

    if not EVEN_K:
        mask_k = offs_x_k < MASK_K_LIMIT
        mask_wk = offs_w_k < (MASK_K_LIMIT // WKD)
        x = tl.load(XPtrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(WPtrs, mask=mask_wk[:, None], other=0,
                    cache_modifier=W_CACHE_MODIFIER)
        ws = _unswizzle_mx_scale_cdna4(
            tl.load(WSPtrs, cache_modifier=W_CACHE_MODIFIER),
            BLOCK_N, MX_SBK)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w, ws, "e2m1",
                            acc=acc, fast_math=True)

    if X_static_scale is not None:
        acc = acc * tl.load(X_static_scale)

    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        acc *= gammas[:, None]

    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_out_n < N
    Y_ptrs = (Y + (start_m + offs_m).to(tl.int64)[:, None] * stride_y_m
              + offs_out_n[None, :].to(tl.int64) * stride_y_n)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(Y_ptrs, acc, mask=mask)


def moe_w2_stage2_weighted(
    intermediate: torch.Tensor,
    w2_data: torch.Tensor,
    w2_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    gammas,
    M: int,
    output_dtype: torch.dtype,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    unpadded_N=None,
    unpadded_K=None,
) -> torch.Tensor:
    """Drop-in replacement for the W2 moe_gemm_a8w4 call.

    Performs W2 GEMM with inline routing weights via _moe_stage2_weighted,
    then scatter-reduces via reduce_grouped.  Returns final output in the
    same shape/dtype as moe_gemm_a8w4 with scatter_indx.
    """
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        get_kernel_config, reduce_grouped,
    )

    N2_padded = w2_data.shape[-1]
    K2_padded = intermediate.shape[-1]
    N2 = unpadded_N if unpadded_N is not None else N2_padded
    K2 = unpadded_K if unpadded_K is not None else K2_padded

    M_route = gather_indx.shape[0] if gather_indx is not None else intermediate.shape[0]
    topk = M_route // M

    config = get_kernel_config(M, N2, K2, routing_data)
    BLOCK_M = config["block_m"]
    BLOCK_N = config["block_n"]
    BLOCK_K = config["block_k"]

    grid_m = routing_data.n_blocks(M_route, BLOCK_M)
    grid_n = triton.cdiv(N2, BLOCK_N)

    # Use padded N for output buffers to match stock moe_gemm_a8w4 output shape
    y_stage2 = torch.zeros(
        (1, M_route, N2_padded), device=intermediate.device, dtype=torch.bfloat16)

    expt_data = routing_data.expt_data

    _moe_stage2_weighted[(grid_m * grid_n,)](
        y_stage2,
        y_stage2.stride(0), y_stage2.stride(1), y_stage2.stride(2),
        intermediate, intermediate.stride(0), intermediate.stride(1),
        w2_data, w2_data.stride(0), w2_data.stride(1), w2_data.stride(2),
        w2_scale, w2_scale.stride(0), w2_scale.stride(1), w2_scale.stride(2),
        a2_scale,
        gammas,
        expt_data.hist,
        expt_data.token_offs_raw,
        expt_data.block_pid_map,
        N2, K2,
        grid_m=grid_m, grid_n=grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=config["group_m"],
        EVEN_K=(K2 % BLOCK_K == 0),
        MASK_K_LIMIT=(K2 % BLOCK_K),
        W_CACHE_MODIFIER=config["w_cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    y_final = torch.empty(
        (M, N2_padded), device=intermediate.device, dtype=output_dtype)
    group_indx = scatter_indx.view(-1, topk)

    y_final = reduce_grouped(
        y_stage2, group_indx, y_final,
        False, swiglu_alpha, swiglu_limit, 1,
        out_dtype=output_dtype,
    )

    return y_final
