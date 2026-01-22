# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

_SHARED_DEBUG = os.environ.get("VLLM_SHARED_EXPERT_DEBUG", "0") == "1"
_debug_logged = False


def _log_shared_debug(msg: str):
    """Log debug message once per rank."""
    global _debug_logged
    if _SHARED_DEBUG and not _debug_logged:
        rank = get_tensor_model_parallel_rank()
        print(f"[SharedFusedMoE DEBUG rank={rank}] {msg}", flush=True)


# TODO(bnell): Add shared + fused combo function? e.g. +
class SharedFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of shared experts.
    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """

    def __init__(
        self,
        shared_experts: torch.nn.Module | None,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts

        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothint to gain
        #   - we are using marlin kernels
        backend = self.moe_parallel_config.all2all_backend
        self.use_overlapped = (
            use_overlapped
            and not (
                (self.enable_eplb and backend != "allgather_reducescatter")
                or (self.moe_config.use_flashinfer_cutlass_kernels and self.dp_size > 1)
            )
            and self._shared_experts is not None
        )

        self._gate = gate

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return self._shared_experts if self.use_overlapped else None

    @property
    def gate(self) -> torch.nn.Module | None:
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        return self.gate is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _debug_logged
        
        if not self.use_overlapped:
            _log_shared_debug(f"use_overlapped=False branch")
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # Reduce shared expert outputs if necessary, since the MLP
                # should have been created with reduce_results=False.
                # When must_reduce_shared_expert_outputs()=True OR MORI-EP is used,
                # the main all-reduce is SKIPPED, so we MUST reduce shared
                # output here regardless of reduce_results setting.
                tp_size = get_tensor_model_parallel_world_size()
                must_reduce = self.must_reduce_shared_expert_outputs()
                uses_mori_ep = (
                    self.moe_parallel_config is not None
                    and self.moe_parallel_config.all2all_backend == "mori_ep"
                )
                should_reduce = must_reduce or uses_mori_ep
                _log_shared_debug(f"tp_size={tp_size}, should_reduce={should_reduce}")
                if tp_size > 1 and should_reduce:
                    before_mean = shared_out.float().mean().item()
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
                    after_mean = shared_out.float().mean().item()
                    _log_shared_debug(f"REDUCED shared_out: before={before_mean:.6f}, after={after_mean:.6f}")
                    _debug_logged = True
            else:
                shared_out = None

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            hs_mean = hidden_states.float().mean().item()
            hs_std = hidden_states.float().std().item()
            _log_shared_debug(f"use_overlapped=True | hidden_states: mean={hs_mean:.6f}, std={hs_std:.6f}")
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            # Ensure early TP reduction of shared expert outputs when required.
            # When must_reduce_shared_expert_outputs()=True (MORI-EP, etc.),
            # the combine kernel reduces ROUTED expert output, but NOT shared
            # expert output. So we MUST reduce shared output here regardless
            # of reduce_results setting (which controls the final all-reduce
            # that is SKIPPED when must_reduce_shared_expert_outputs()=True).
            tp_size = get_tensor_model_parallel_world_size()
            
            # Check if we need to reduce shared output:
            # 1. Standard check via must_reduce_shared_expert_outputs()
            # 2. OR if MORI-EP is used (which reduces routed output but not shared)
            must_reduce = self.must_reduce_shared_expert_outputs()
            uses_mori_ep = (
                self.moe_parallel_config is not None
                and self.moe_parallel_config.all2all_backend == "mori_ep"
            )
            should_reduce = must_reduce or uses_mori_ep
            
            _log_shared_debug(
                f"tp_size={tp_size}, must_reduce={must_reduce}, uses_mori_ep={uses_mori_ep}, "
                f"should_reduce={should_reduce}, shared_out is None: {shared_out is None}"
            )
            
            if (
                shared_out is not None
                and tp_size > 1
                and should_reduce
            ):
                before_mean = shared_out.float().mean().item()
                before_std = shared_out.float().std().item()
                shared_out = tensor_model_parallel_all_reduce(shared_out)
                after_mean = shared_out.float().mean().item()
                after_std = shared_out.float().std().item()
                fused_mean = fused_out.float().mean().item()
                fused_std = fused_out.float().std().item()
                _log_shared_debug(
                    f"REDUCED shared: before_mean={before_mean:.6f}, before_std={before_std:.6f}, "
                    f"after_mean={after_mean:.6f}, after_std={after_std:.6f} | "
                    f"fused_out: mean={fused_mean:.6f}, std={fused_std:.6f}"
                )
                _debug_logged = True
            elif shared_out is None:
                _log_shared_debug(f"WARNING: shared_out is None!")
                _debug_logged = True
        return shared_out, fused_out
