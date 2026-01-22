# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


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
        if not self.use_overlapped:
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
                if tp_size > 1 and should_reduce:
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
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
            
            if (
                shared_out is not None
                and tp_size > 1
                and should_reduce
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)

        return shared_out, fused_out
