# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    QueryLenSupport,
    get_mla_dims,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLABackend,
    AiterMLADecodeMetadata,
    AiterMLAHelper,
    AiterMLAImpl,
    AiterMLAMetadata,
    AiterMLAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    cu_seqlens_ptr,  # int32 [num_tokens + 1]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load cumulative sequence lengths to get starting index of this request
    seq_start = tl.load(cu_seqlens_ptr + token_id)
    seq_end = tl.load(cu_seqlens_ptr + token_id + 1)

    if tile_id * BLOCK_N + seq_start >= seq_end:
        return

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # # If token == -1 OR block_id OOB, output 0; else base * BLOCK_SIZE + offset
    out_val = tl.where(
        is_invalid_tok | (~valid_block), 0, base * BLOCK_SIZE + inblock_off
    )
    out_ptr_ij = out_ptr + seq_start + indice_id
    out_ptr_ij_mask = (seq_start + indice_id) < seq_end

    # store the results with mask
    tl.store(out_ptr_ij, out_val, mask=out_ptr_ij_mask)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    cu_seqlens: torch.Tensor,  # int32 [num_tokens + 1]
    paged_kv_indices: torch.Tensor,  # int32 [num_tokens * topk] out_buffer
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible byBLOCK_N ({BLOCK_N})"
    )
    # print("req_id: ", req_id, flush=True)
    num_tokens = req_id.shape[0]
    _, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        cu_seqlens,
        paged_kv_indices,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
    )
    return


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,  # [num_seq, topk]
    cumsum_ptr,  # [num_seq + 1]
    out_tensor_ptr,  # [max_num_seq * topk]
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
):
    num_tokens = in_tensor.size(0)
    block_size = 64
    num_block_per_row = triton.cdiv(topk, block_size)
    grid = (
        num_tokens,
        num_block_per_row,
    )
    fetch_id_to_ragged_kernel[grid](
        in_tensor, cumsum, out_tensor, in_tensor.stride(0), topk, num_tokens, block_size
    )


@triton.jit
def generate_sparse_seqlen_kernel(
    seq_len_ptr,
    cu_query_lens_ptr,
    out_ptr,
    topk_token: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    query_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    query_start = tl.load(cu_query_lens_ptr + seq_id)
    query_end = tl.load(cu_query_lens_ptr + seq_id + 1)
    if query_start + tl.program_id(1) * BLOCK_SIZE > query_end:
        return
    query_len = query_end - query_start
    query_mask = query_offset + query_start < query_end
    seq_len = tl.load(seq_len_ptr + seq_id)
    if seq_len == 0:
        return
    context_start_point = seq_len - query_len
    sparse_seqlen = context_start_point + query_offset
    sparse_seqlen_masked = tl.where(
        sparse_seqlen + 1 < topk_token, sparse_seqlen + 1, topk_token
    )
    tl.store(
        out_ptr + query_start + query_offset, sparse_seqlen_masked, mask=query_mask
    )


def generate_sparse_seqlen_triton(
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    topk_token: int,
    num_tokens: int,
    max_query_len: int,
):
    num_seqs = seq_lens.size(0)
    out = torch.zeros([num_tokens], dtype=torch.int32, device=seq_lens.device)
    block_size = 64
    num_block_per_row = triton.cdiv(max_query_len, block_size)
    grid = (num_seqs, num_block_per_row)
    generate_sparse_seqlen_kernel[grid](
        seq_lens,
        cu_query_lens,
        out,
        topk_token,
        block_size,
    )
    return out


@dataclass
class ROCMAiterMLASparseMetadata(AiterMLAMetadata):
    """Extends AiterMLAMetadata with sparse-specific fields for decode."""

    # Sparse decode fields
    sparse_req_id_per_token: torch.Tensor | None = None
    sparse_topk_tokens: int = 2048
    sparse_qo_indptr: torch.Tensor | None = None
    sparse_paged_kv_last_page_len: torch.Tensor | None = None
    sparse_paged_kv_indices: torch.Tensor | None = None
    sparse_paged_kv_indptr: torch.Tensor | None = None


class ROCMAiterMLASparseBackend(AiterMLABackend):
    """Sparse MLA backend that inherits prefill (MHA) from AiterMLABackend
    and uses sparse decode via mla_decode_fwd with topk index selection."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, 64]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["ROCMAiterMLASparseMetadata"]:
        return ROCMAiterMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["ROCMAiterMLASparseMetadataBuilder"]:
        return ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["ROCMAiterMLASparseImpl"]:
        return ROCMAiterMLASparseImpl

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class ROCMAiterMLASparseMetadataBuilder(AiterMLAMetadataBuilder):
    """Metadata builder that inherits prefill + decode building from
    AiterMLAMetadataBuilder and adds sparse-specific fields for decode."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Override metadata class so parent builder creates our sparse metadata
        self.metadata_cls = ROCMAiterMLASparseMetadata

        # AiterMLAMetadataBuilder.__init__ allocates paged_kv_indices sized for
        # max_num_reqs * max_model_len pages, assuming kernel_block_size=1 (one
        # page per token).  With kernel_block_size=64 that is 64× more entries
        # than needed — and our _build_decode override never uses this buffer at
        # all.  Releasing it saves ~52 MB per attention layer (≈3 GB total for
        # DeepSeek-V3.2's ~60 MLA layers per TP rank), which is critical budget
        # for the indexer's prefill logits allocation.
        self.paged_kv_indices = torch.zeros(1, dtype=torch.int32, device=device)

        # Sparse-specific fields
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

        # Sparse decode uses separate paged_kv tracking for topk indices
        self.sparse_paged_kv_indices = torch.zeros(
            [max_num_batched_tokens * self.topk_tokens],
            dtype=torch.int32,
            device=device,
        )
        self.sparse_paged_kv_indptr = torch.zeros(
            [max_num_batched_tokens + 1], dtype=torch.int32, device=device
        )
        self.sparse_qo_indptr = torch.arange(
            0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.sparse_paged_kv_last_page_len = torch.ones(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )

        # ----- Persistent (work-stealing) MLA decode metadata buffers -----
        # Mirrors the approach landed in #41990 for the (now-replaced) sparse
        # backend: when the aiter sparse decode kernel is given precomputed
        # work-splitting metadata it takes the persistent path, which load-
        # balances better across CUs for heterogeneous decode batches.
        # The parent's AiterMLAMetadataBuilder.__init__ already sized the
        # persistent buffers via get_mla_metadata_info_v1(...) for the
        # *dense* MLA case (qseqlen <= 1, no is_sparse flag); sparse decode
        # uses uni_seqlen_qo=1 with is_sparse=True, which produces
        # different buffer sizes — so we re-allocate here, overriding the
        # parent's tensors.
        from aiter import dtypes as _aiter_dtypes
        from aiter import get_mla_metadata_info_v1

        self._sparse_num_attention_heads = max(16, self.num_heads)
        _q_dtype = self.model_config.dtype
        _cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if _cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            _cache_dtype_str = "fp8"
        else:
            _cache_dtype_str = "bf16"
        _kv_dtype = _aiter_dtypes.d_dtypes.get(_cache_dtype_str, _aiter_dtypes.bf16)

        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_set_size, work_info_set_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = get_mla_metadata_info_v1(
            max_num_batched_tokens,
            1,
            self._sparse_num_attention_heads,
            _q_dtype,
            _kv_dtype,
            is_sparse=True,
            fast_mode=True,
        )
        self._mla_work_meta_data = torch.empty(
            work_metadata_size, dtype=work_metadata_dtype, device=device
        )
        self._mla_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_dtype, device=device
        )
        self._mla_work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_dtype, device=device
        )
        self._mla_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
        )
        self._mla_reduce_final_map = torch.empty(
            reduce_final_map_size,
            dtype=reduce_final_map_dtype,
            device=device,
        )
        self._mla_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_dtype,
            device=device,
        )

    def _build_sparse_fields(self, common_attn_metadata):
        """Build sparse-specific metadata fields for decode."""
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        self.paged_kv_indices.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        self.sparse_paged_kv_indices.fill_(0)
        self.sparse_paged_kv_indptr.fill_(0)

        seq_lens = common_attn_metadata.seq_lens
        sparse_seqlen = generate_sparse_seqlen_triton(
            seq_lens,
            common_attn_metadata.query_start_loc,
            self.topk_tokens,
            num_tokens,
            common_attn_metadata.max_query_len,
        )
        torch.cumsum(
            sparse_seqlen, dim=0, out=self.sparse_paged_kv_indptr[1 : num_tokens + 1]
        )
        self.sparse_paged_kv_indptr[num_tokens + 1 :].fill_(
            self.sparse_paged_kv_indptr[num_tokens]
        )

        return {
            "sparse_req_id_per_token": self.req_id_per_token_buffer[:num_tokens],
            "sparse_topk_tokens": self.topk_tokens,
            "sparse_qo_indptr": self.sparse_qo_indptr[: num_tokens + 1],
            "sparse_paged_kv_last_page_len": (
                self.sparse_paged_kv_last_page_len[:num_tokens]
            ),
            "sparse_paged_kv_indices": (
                self.sparse_paged_kv_indices[: num_tokens * self.topk_tokens]
            ),
            "sparse_paged_kv_indptr": (self.sparse_paged_kv_indptr[: num_tokens + 1]),
        }

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        # The parent's _build_decode calls _copy_page_indices_kernel which
        # iterates seq_len times per row in block_table.  With kernel_block_size
        # = 64 each row has only ceil(seq_len/64) valid entries → OOB GPU read.
        # Our forward_mqa only reads attn_metadata.decode.block_table directly;
        # it never uses paged_kv_indices or the AITER work metadata.
        # Return a minimal decode metadata with just the fields we need.
        # attn_out_dtype is the model dtype (e.g. bf16): the dispatcher may
        # have pre-quantised q to fp8 before forward_mqa, but AITER mla_reduce_v1
        # only emits bf16/fp16, so we cannot use q.dtype as the output buffer
        # dtype in _forward_sparse_mla.
        return AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            attn_out_dtype=self.decode_attn_out_dtype,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # Build sparse fields first
        sparse_fields = self._build_sparse_fields(common_attn_metadata)

        # Build standard prefill + decode metadata via parent
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        # Attach sparse fields to metadata
        for key, value in sparse_fields.items():
            setattr(metadata, key, value)

        # ----- Populate persistent (work-stealing) MLA decode metadata -----
        # The aiter sparse decode kernel uses qseqlen=1 (each query token is
        # its own batch entry), so the persistent path is always applicable.
        # Mirrors #41990's approach but keyed off the sparse_* indptr buffers
        # produced above. On any failure we leave work_meta_data=None and the
        # kernel falls back to its non-persistent path.
        #
        # Pass DECODE-ONLY indptrs to the populator.  The dispatcher places
        # decode tokens at q[:num_decode_tokens] and only those go through
        # forward_mqa -> mla_decode_fwd; the persistent populator uses the
        # indptr length as the batch size for work-splitting, so passing
        # the full prefill+decode indptr would schedule work for prefill
        # positions whose KV indices are populated as -1 by the indexer.
        num_decode_tokens = getattr(metadata, "num_decode_tokens", 0)
        if num_decode_tokens > 0:
            try:
                from aiter import get_mla_metadata_v1

                get_mla_metadata_v1(
                    metadata.sparse_qo_indptr[: num_decode_tokens + 1],
                    metadata.sparse_paged_kv_indptr[: num_decode_tokens + 1],
                    metadata.sparse_paged_kv_last_page_len[:num_decode_tokens],
                    self._sparse_num_attention_heads,
                    1,
                    True,
                    self._mla_work_meta_data,
                    self._mla_work_info_set,
                    self._mla_work_indptr,
                    self._mla_reduce_indptr,
                    self._mla_reduce_final_map,
                    self._mla_reduce_partial_map,
                    page_size=1,
                    kv_granularity=16,
                    max_seqlen_qo=1,
                    uni_seqlen_qo=1,
                    fast_mode=True,
                    # Buffer sizes were declared with is_sparse=True via
                    # get_mla_metadata_info_v1; the populator must write
                    # the matching sparse layout, otherwise the persistent
                    # sparse decode kernel reads bogus work-info entries
                    # and faults with a GPU memory access fault.  topk
                    # defaults to -1 (dense layout) on the populator
                    # side, so we must pass it explicitly to keep the
                    # populator and the buffer sizing consistent.
                    topk=self.topk_tokens,
                )
                metadata.work_meta_data = self._mla_work_meta_data
                metadata.work_indptr = self._mla_work_indptr
                metadata.work_info_set = self._mla_work_info_set
                metadata.reduce_indptr = self._mla_reduce_indptr
                metadata.reduce_final_map = self._mla_reduce_final_map
                metadata.reduce_partial_map = self._mla_reduce_partial_map
            except Exception as exc:  # noqa: BLE001
                logger.warning_once(
                    "ROCMAiterMLASparseMetadataBuilder: persistent MLA "
                    "metadata population failed (%s); decode will fall "
                    "back to the non-persistent kernel path.",
                    exc,
                )

        return metadata


# Take from
# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_prefill.py#L72
def reference_mla_sparse_prefill(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    skv = kv.shape[0]
    sq = q.shape[0]
    topk = indices.shape[-1]
    dqk = q.shape[-1]
    indices = indices[:, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= skv)
    indices[invalid_indices_mask] = 0
    qs = q  # [s_q, h_q, d_qk]
    kvs = kv[:, 0, :][indices].view(sq, topk, dqk)  # [s_q, topk, d_qk]

    attn_score = (qs @ kvs.transpose(1, 2)).float()  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score.to(q.dtype) @ kvs[:, :, :d_v]
    return (result, lse)


class ROCMAiterMLASparseImpl(AiterMLAImpl):
    """Sparse MLA impl that inherits forward_mha (compute-bound prefill via
    flash_attn_varlen_func) from AiterMLAImpl/MLACommonImpl, and overrides
    forward_mqa for sparse decode via mla_decode_fwd with topk indices."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            indexer=indexer,
            **mla_args,
        )
        # Sparse-specific: get the topk indices buffer from the indexer
        assert indexer is not None
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer
        self._decode_out: torch.Tensor | None = None

    def _forward_sparse_mla(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        # The dispatcher in mla_attention.py may have pre-quantised q to fp8
        # before forward_mqa (when self.supports_quant_query_input is true
        # and the cache dtype is fp8), so q.dtype is unsafe as the output
        # buffer dtype: AITER's mla_reduce_v1 only emits bf16/fp16.  Use the
        # model dtype the builder cached on decode metadata instead.
        attn_out_dtype = attn_metadata.decode.attn_out_dtype
        # q may have been padded by AiterMLAHelper.get_mla_padded_q in the
        # caller (forward_mqa) so num_heads >= 16 for the AITER kernel.
        # Size the output buffer to whatever shape q was actually called
        # with; the caller is responsible for unpadding via
        # get_mla_unpadded_o.
        kernel_num_heads = q.shape[1]

        is_fp8_kv = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv:
            fp8_dtype = current_platform.fp8_dtype()
            q_scale = layer._q_scale if layer is not None else None
            k_scale = layer._k_scale if layer is not None else None
            q = q.to(fp8_dtype)
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(fp8_dtype)

        # mla_decode_fwd uses page_size=1 (per-token paging) internally.
        # When kernel_block_size > 1, the KV cache shape is
        # [num_pages, block_size, head_size].  Flatten to
        # [num_pages * block_size, 1, head_size] so that the flat token
        # indices in sparse_paged_kv_indices correctly address each token.
        if kv_c_and_k_pe_cache.shape[1] != 1:
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.reshape(
                -1, 1, kv_c_and_k_pe_cache.shape[-1]
            )

        # Slice sparse CSR fields to num_tokens (= num_decode_tokens).
        # _build_sparse_fields builds these for num_actual_tokens (prefill+decode),
        # but forward_mqa only handles decode tokens (q.shape[0]).  In a mixed
        # prefill+decode batch, passing the full qo_indptr/kv_indptr would set
        # bs = num_actual_tokens while o has num_decode_tokens rows, causing the
        # stage2 kernel to write o[cur_qo] for cur_qo >= total_s → GPU OOB fault.
        qo_indptr = attn_metadata.sparse_qo_indptr[: num_tokens + 1]
        kv_indptr = attn_metadata.sparse_paged_kv_indptr[: num_tokens + 1]
        kv_last_page_len = attn_metadata.sparse_paged_kv_last_page_len[:num_tokens]

        if (
            self._decode_out is None
            or self._decode_out.shape[0] < num_tokens
            or self._decode_out.shape[1] != kernel_num_heads
            or self._decode_out.dtype != attn_out_dtype
        ):
            self._decode_out = torch.zeros(
                [num_tokens, kernel_num_heads, self.kv_lora_rank],
                dtype=attn_out_dtype,
                device=q.device,
            )
        output = self._decode_out[:num_tokens]

        # Forward persistent (work-stealing) decode metadata when the
        # builder populated it.  The kernel falls back to its non-
        # persistent path automatically when work_meta_data is absent.
        mla_kwargs: dict = {}
        if attn_metadata.work_meta_data is not None:
            mla_kwargs.update(
                work_meta_data=attn_metadata.work_meta_data,
                work_indptr=attn_metadata.work_indptr,
                work_info_set=attn_metadata.work_info_set,
                reduce_indptr=attn_metadata.reduce_indptr,
                reduce_final_map=attn_metadata.reduce_final_map,
                reduce_partial_map=attn_metadata.reduce_partial_map,
            )

        # NOTE: q_scale / kv_scale MUST be passed as keyword arguments.
        # rocm_aiter_ops.mla_decode_fwd has positional slot 10 =
        # logit_cap (float, default 0.0); passing q_scale / k_scale
        # positionally would bind a Tensor(1.0) to logit_cap and
        # trigger AITER's "logit_cap=1.0 is not support yet" error.
        if is_fp8_kv:
            rocm_aiter_ops.mla_decode_fwd(
                q,
                kv_c_and_k_pe_cache,
                output,
                self.scale,
                qo_indptr,
                1,
                kv_indptr,
                attn_metadata.sparse_paged_kv_indices,
                kv_last_page_len,
                q_scale=q_scale,
                kv_scale=k_scale,
                **mla_kwargs,
            )
        else:
            rocm_aiter_ops.mla_decode_fwd(
                q,
                kv_c_and_k_pe_cache,
                output,
                self.scale,
                qo_indptr,
                1,
                kv_indptr,
                attn_metadata.sparse_paged_kv_indices,
                kv_last_page_len,
                **mla_kwargs,
            )

        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # For sparse decode, use MQA 576/512 approach with topk indices.
        #
        # The MLA dispatcher (vllm/model_executor/layers/attention/mla_attention.py
        # forward()) places decode tokens at the front of the batch and passes
        # mqa_q = q[:num_decode_tokens] here, so q.shape[0] == num_decode_tokens.
        # Everything below MUST be decode-sliced — using
        # attn_metadata.num_actual_tokens reads into prefill positions where
        # topk_indices_buffer is uninitialised (the indexer's prefill skip)
        # and where req_id_per_token would index past the end of
        # decode.block_table (it only has rows for decode requests).
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        # AITER MLA decode kernels require num_heads >= 16 (= AITER's MLA
        # head-tile size).  For configs with fewer heads (e.g. heavily
        # tensor-parallelised small models), the helper repeat-interleaves
        # along dim=1 to bring heads up to the tile size; for DSV3.2 with
        # 128 heads this is a no-op.  Match the idiom used by the dense
        # AiterMLAImpl.forward_mqa we inherit so the sparse decode path
        # is safe in both regimes.
        q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)

        num_decode_tokens = q.shape[0]
        if num_decode_tokens == 0:
            return q.new_empty((0, self.num_heads, self.kv_lora_rank)), None

        # Get topk indices for the decode portion only.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_decode_tokens]

        assert attn_metadata.decode is not None

        # Convert per-request topk token indices to global flat KV-cache
        # indices, writing the RAGGED output directly into
        # sparse_paged_kv_indices (which mla_decode_fwd consumes).
        triton_convert_req_index_to_global_index(
            attn_metadata.sparse_req_id_per_token[:num_decode_tokens],
            attn_metadata.decode.block_table,
            topk_indices,
            attn_metadata.sparse_paged_kv_indptr[: num_decode_tokens + 1],
            attn_metadata.sparse_paged_kv_indices,
            BLOCK_SIZE=64,  # block_size=64 for this backend
            NUM_TOPK_TOKENS=attn_metadata.sparse_topk_tokens,
        )

        attn_out = self._forward_sparse_mla(
            layer, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
        )

        # Unpad heads if get_mla_padded_q replicated them above.  For
        # num_heads >= 16 (DSV3.2) this is a no-op.
        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, attn_out), None
