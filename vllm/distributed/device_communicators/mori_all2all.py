# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MORI (Modular RDMA Interface) All2All Manager for ROCm EP MoE.

This provides efficient dispatch/combine communication for MoE expert
parallelism on AMD GPUs using MORI's optimized kernels.

Based on the reference implementation from:
https://github.com/alexsun07/vllm/tree/mori_ep
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_mori

from .base_device_communicator import All2AllManagerBase, Cache

logger = init_logger(__name__)


class MoriAll2AllManager(All2AllManagerBase):
    """
    All2All manager using MORI dispatch/combine for ROCm EP MoE.
    
    MORI provides optimized all-to-all communication for MoE expert
    parallelism with support for:
    - Intra-node XGMI communication
    - Inter-node RDMA communication  
    - FP8 dispatch for reduced bandwidth
    """
    
    def __init__(self, cpu_group):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )
        import mori
        
        super().__init__(cpu_group)
        self.handle_cache = Cache()
        
        # Register process group with MORI and initialize shared memory
        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")
        
        logger.info(
            "MoriAll2AllManager initialized: rank=%d, world_size=%d, internode=%s",
            self.rank, self.world_size, self.internode
        )
    
    def _make_all2all_kwargs(
        self,
        rank: int,
        num_ep_ranks: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        token_hidden_size: int,
        scale_dim: int,
        scale_type_size: int,
        max_num_tokens_per_dp_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
        gpu_per_node: int,
    ):
        """Create MORI configuration kwargs based on node topology."""
        import mori
        
        if not self.internode:
            # Single node - use XGMI
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            warp_num_per_block = 16
            block_num = 80
            rdma_block_num = 0
        else:
            # Multi node - use RDMA
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
            warp_num_per_block = 16
            block_num = 32
            rdma_block_num = 16
        
        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            data_type=quant_dtype,
            hidden_dim=token_hidden_size,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=input_dtype.itemsize,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            kernel_type=kernel_type,
            rdma_block_num=rdma_block_num,
            gpu_per_node=gpu_per_node,
        )
    
    def _make_handle(self, **kwargs):
        """Create MORI EpDispatchCombineOp handle."""
        import mori
        
        mori_config = mori.ops.EpDispatchCombineConfig(**kwargs)
        handle = mori.ops.EpDispatchCombineOp(mori_config)
        return handle
    
    def get_handle(self, kwargs: dict[str, Any]):
        """
        Get or create a MORI EpDispatchCombineOp handle.
        
        The kwargs should contain:
        - rank: Current rank
        - num_ep_ranks: EP world size
        - input_dtype: Input data type
        - quant_dtype: Quantization dtype for communication
        - token_hidden_size: Hidden dimension
        - scale_dim: Scale dimension (1 for per-token, hidden_dim//128 for block)
        - scale_type_size: Size of scale type (4 for float32)
        - max_num_tokens_per_dp_rank: Max tokens per rank
        - num_local_experts: Number of local experts
        - num_experts_per_token: topk
        - gpu_per_node: GPUs per node
        """
        import mori
        
        mori_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("MoRI all2all args %s", mori_kwargs)
        handle: mori.ops.EpDispatchCombineOp = self.handle_cache.get_or_create(
            mori_kwargs, self._make_handle
        )
        return handle
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        """
        MORI dispatch is handled via the modular kernel interface.
        This method is not used directly.
        """
        raise NotImplementedError(
            "MoriAll2AllManager.dispatch() is not used directly. "
            "Use MoriPrepareAndFinalize via the modular kernel interface."
        )
    
    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ):
        """
        MORI combine is handled via the modular kernel interface.
        This method is not used directly.
        """
        raise NotImplementedError(
            "MoriAll2AllManager.combine() is not used directly. "
            "Use MoriPrepareAndFinalize via the modular kernel interface."
        )
    
    def destroy(self):
        """Clean up MORI handles."""
        with self.handle_cache._lock:
            self.handle_cache._cache.clear()
