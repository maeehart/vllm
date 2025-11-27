# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr, value_ptr,
    key_cache_ptr, value_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr, v_scale_ptr,
    # RoPE arguments
    positions_ptr,
    cos_sin_cache_ptr,
    stride_cs_n, stride_cs_d,
    rot_dim: tl.constexpr,
    is_neox: tl.constexpr,
    has_rope: tl.constexpr,
    # Query fusion args
    query_ptr,
    stride_query_n, stride_query_h, stride_query_d,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    # Norm args
    k_norm_weight_ptr,
    q_norm_weight_ptr,
    rms_norm_eps,
    has_k_norm: tl.constexpr,
    has_q_norm: tl.constexpr,
    # Strides
    stride_key_n, stride_key_h, stride_key_d,
    stride_val_n, stride_val_h, stride_val_d,
    stride_cache_b, stride_cache_s, stride_cache_h, stride_cache_d,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_TYPE: tl.constexpr
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    block_idx = slot // BLOCK_SIZE
    block_off = slot % BLOCK_SIZE
    
    # Offsets
    src_offset = token_idx * stride_key_n + head_idx * stride_key_h
    dst_offset = block_idx * stride_cache_b + block_off * stride_cache_s + head_idx * stride_cache_h
    
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Load Value (always full head)
    v_val = tl.load(value_ptr + src_offset + offs_d * stride_val_d)
    
    # Handle Key with optional RoPE
    if has_rope:
        # Apply RMSNorm to Key if needed
        if has_k_norm:
            k_val = tl.load(key_ptr + src_offset + offs_d * stride_key_d)
            k_float = k_val.to(tl.float32)
            k_sq = k_float * k_float
            k_var = tl.sum(k_sq, axis=0) / HEAD_DIM
            k_rsqrt = tl.rsqrt(k_var + rms_norm_eps)
            
            k_norm_w = tl.load(k_norm_weight_ptr + offs_d)
            k_norm_w = k_norm_w.to(tl.float32)
            
            k_normed = k_float * k_rsqrt * (1.0 + k_norm_w)
            k_val = k_normed.to(k_val.dtype)
            tl.store(key_ptr + src_offset + offs_d * stride_key_d, k_val)

        pos = tl.load(positions_ptr + token_idx)
        half_rot = rot_dim // 2
        
        # Load cos/sin
        # cos_sin_cache is [max_pos, rot_dim]
        # cos is [0..half_rot], sin is [half_rot..rot_dim]
        offs_rot_half = tl.arange(0, half_rot)
        cs_row_ptr = cos_sin_cache_ptr + pos * stride_cs_n
        cos = tl.load(cs_row_ptr + offs_rot_half * stride_cs_d)
        sin = tl.load(cs_row_ptr + (half_rot + offs_rot_half) * stride_cs_d)
        
        # Load and rotate Key
        if is_neox:
            # Neox style: split in halves
            offs_1 = offs_rot_half
            offs_2 = half_rot + offs_rot_half
        else:
            # GPT-J style: interleave
            offs_1 = offs_rot_half * 2
            offs_2 = offs_rot_half * 2 + 1
            
        k1 = tl.load(key_ptr + src_offset + offs_1 * stride_key_d)
        k2 = tl.load(key_ptr + src_offset + offs_2 * stride_key_d)
        
        # Apply RoPE to Key
        new_k1 = k1 * cos - k2 * sin
        new_k2 = k2 * cos + k1 * sin
        
        # Process rotated parts
        if IS_FP8:
            k_scale = tl.load(k_scale_ptr)
            new_k1 = new_k1 / k_scale
            new_k2 = new_k2 / k_scale
            
            if FP8_TYPE == 1: # e4m3
                new_k1 = new_k1.to(tl.float8e4b8)
                new_k2 = new_k2.to(tl.float8e4b8)
            elif FP8_TYPE == 2: # e5m2
                new_k1 = new_k1.to(tl.float8e5)
                new_k2 = new_k2.to(tl.float8e5)
                
            new_k1 = new_k1.to(tl.int8, bitcast=True)
            new_k2 = new_k2.to(tl.int8, bitcast=True)
            
        tl.store(key_cache_ptr + dst_offset + offs_1 * stride_cache_d, new_k1)
        tl.store(key_cache_ptr + dst_offset + offs_2 * stride_cache_d, new_k2)
        
        # Handle remainder if rot_dim < HEAD_DIM
        if rot_dim < HEAD_DIM:
            offs_rest = tl.arange(rot_dim, HEAD_DIM)
            k_rest = tl.load(key_ptr + src_offset + offs_rest * stride_key_d)
            
            if IS_FP8:
                k_scale = tl.load(k_scale_ptr) # Reload or reuse? Reuse is fine
                k_rest = k_rest / k_scale
                if FP8_TYPE == 1:
                    k_rest = k_rest.to(tl.float8e4b8)
                elif FP8_TYPE == 2:
                    k_rest = k_rest.to(tl.float8e5)
                k_rest = k_rest.to(tl.int8, bitcast=True)
                
            tl.store(key_cache_ptr + dst_offset + offs_rest * stride_cache_d, k_rest)

        # --- Fused Query Rotation ---
        # If query_ptr is provided (not null/dummy), we rotate Q here.
        # We iterate over the Q heads corresponding to this KV head.
        if query_ptr is not None:
            # Calculate ratio of Q heads to KV heads
            # We assume NUM_Q_HEADS % NUM_KV_HEADS == 0
            ratio = NUM_Q_HEADS // NUM_KV_HEADS
            
            for i in range(ratio):
                q_head_idx = head_idx * ratio + i
                q_offset = token_idx * stride_query_n + q_head_idx * stride_query_h
                
                # Apply RMSNorm to Query if needed
                if has_q_norm:
                    q_val = tl.load(query_ptr + q_offset + offs_d * stride_query_d)
                    q_float = q_val.to(tl.float32)
                    q_sq = q_float * q_float
                    q_var = tl.sum(q_sq, axis=0) / HEAD_DIM
                    q_rsqrt = tl.rsqrt(q_var + rms_norm_eps)
                    
                    q_norm_w = tl.load(q_norm_weight_ptr + offs_d)
                    q_norm_w = q_norm_w.to(tl.float32)
                    
                    q_normed = q_float * q_rsqrt * (1.0 + q_norm_w)
                    q_val = q_normed.to(q_val.dtype)
                    tl.store(query_ptr + q_offset + offs_d * stride_query_d, q_val)

                q1 = tl.load(query_ptr + q_offset + offs_1 * stride_query_d)
                q2 = tl.load(query_ptr + q_offset + offs_2 * stride_query_d)
                
                # Apply RoPE to Query
                new_q1 = q1 * cos - q2 * sin
                new_q2 = q2 * cos + q1 * sin
                
                # Store back to Query (in-place)
                tl.store(query_ptr + q_offset + offs_1 * stride_query_d, new_q1)
                tl.store(query_ptr + q_offset + offs_2 * stride_query_d, new_q2)
            
    else:
        # No RoPE - Standard Load
        k_val = tl.load(key_ptr + src_offset + offs_d * stride_key_d)
        
        if IS_FP8:
            k_scale = tl.load(k_scale_ptr)
            k_val = k_val / k_scale
            
            if FP8_TYPE == 1: # e4m3
                k_val = k_val.to(tl.float8e4b8)
            elif FP8_TYPE == 2: # e5m2
                k_val = k_val.to(tl.float8e5)
                
            k_val = k_val.to(tl.int8, bitcast=True)

        tl.store(key_cache_ptr + dst_offset + offs_d * stride_cache_d, k_val)

    # Process Value (always same)
    if IS_FP8:
        v_scale = tl.load(v_scale_ptr)
        v_val = v_val / v_scale
        
        if FP8_TYPE == 1: # e4m3
            v_val = v_val.to(tl.float8e4b8)
        elif FP8_TYPE == 2: # e5m2
            v_val = v_val.to(tl.float8e5)
            
        v_val = v_val.to(tl.int8, bitcast=True)

    tl.store(value_cache_ptr + dst_offset + offs_d * stride_cache_d, v_val)


def triton_reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    # [num_blocks, block_size, num_heads, head_size]
    key_cache: torch.Tensor,
    # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
    # Optional RoPE args
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    rot_dim: int = 0,
    is_neox_style: bool = True,
    # Optional Query for fusion
    query: torch.Tensor | None = None,
    # Optional Norm args
    k_norm_weight: torch.Tensor | None = None,
    q_norm_weight: torch.Tensor | None = None,
    rms_norm_eps: float = 1e-6,
):
    if k_scale.device != key.device:
        k_scale = k_scale.to(key.device)
    if v_scale.device != key.device:
        v_scale = v_scale.to(key.device)

    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    
    # Check if we can use the Triton kernel
    # We need head_size to be a power of 2 or handle it
    # My kernel uses next_power_of_2(head_dim)
    
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if kv_cache_dtype.startswith("fp8")
        else key_cache.dtype
    )

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")
    fp8_type_code = 0
    if FP8_KV_CACHE:
        if "e4m3" in str(kv_cache_torch_dtype):
            fp8_type_code = 1
        elif "e5m2" in str(kv_cache_torch_dtype):
            fp8_type_code = 2

    has_rope = positions is not None and cos_sin_cache is not None
    
    # Dummy values for RoPE args if not present
    if not has_rope:
        positions = key # dummy
        cos_sin_cache = key # dummy
        stride_cs_n = 0
        stride_cs_d = 0
    else:
        # print(f"DEBUG: triton_reshape_and_cache_flash called with RoPE! rot_dim={rot_dim}")
        stride_cs_n = cos_sin_cache.stride(0)
        stride_cs_d = cos_sin_cache.stride(1)

    # Query handling
    if query is not None:
        stride_query_n = query.stride(0)
        stride_query_h = query.stride(1)
        stride_query_d = query.stride(2)
        num_q_heads = query.shape[1]
    else:
        stride_query_n = 0
        stride_query_h = 0
        stride_query_d = 0
        num_q_heads = 0

    has_k_norm = k_norm_weight is not None
    has_q_norm = q_norm_weight is not None
    
    if not has_k_norm:
        k_norm_weight = key # dummy
    if not has_q_norm:
        q_norm_weight = key # dummy

    num_tokens = key.shape[0]
    grid = (num_tokens, num_heads)
    BLOCK_D = triton.next_power_of_2(head_size)
    
    reshape_and_cache_kernel_flash[grid](
        key, value,
        key_cache, value_cache,
        slot_mapping,
        k_scale, v_scale,
        # RoPE
        positions, cos_sin_cache,
        stride_cs_n, stride_cs_d,
        rot_dim=rot_dim,
        is_neox=is_neox_style,
        has_rope=has_rope,
        # Query Fusion
        query_ptr=query,
        stride_query_n=stride_query_n, stride_query_h=stride_query_h, stride_query_d=stride_query_d,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_heads,
        # Norm args
        k_norm_weight_ptr=k_norm_weight,
        q_norm_weight_ptr=q_norm_weight,
        rms_norm_eps=rms_norm_eps,
        has_k_norm=has_k_norm,
        has_q_norm=has_q_norm,
        # Strides
        stride_key_n=key.stride(0), stride_key_h=key.stride(1), stride_key_d=key.stride(2),
        stride_val_n=value.stride(0), stride_val_h=value.stride(1), stride_val_d=value.stride(2),
        stride_cache_b=key_cache.stride(0), stride_cache_s=key_cache.stride(1), stride_cache_h=key_cache.stride(2), stride_cache_d=key_cache.stride(3),
        BLOCK_SIZE=block_size,
        HEAD_DIM=BLOCK_D,
        IS_FP8=FP8_KV_CACHE,
        FP8_TYPE=fp8_type_code
    )

