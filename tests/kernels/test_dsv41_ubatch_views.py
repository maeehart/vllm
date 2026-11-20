# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 step buffers under dual batch overlap: each microbatch must
see its own rows of the shared top-k buffer and its own index caches, so one
microbatch's index source layer cannot overwrite what the other still reads."""

import torch
from torch import nn

from vllm.models.deepseek_v41.amd import ubatch_views


class _Layer(nn.Module):
    def __init__(self, buf: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = buf
        self._topk_ragged_cache: dict = {}


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.topk_indices_buffer = torch.zeros(10, 2, dtype=torch.int32)
        self.layers = nn.ModuleList(_Layer(self.topk_indices_buffer) for _ in range(2))
        self.other = nn.Linear(2, 2)


def test_views_follow_the_calling_microbatch(monkeypatch):
    ubatch = {"id": 0, "offset": 0}
    monkeypatch.setattr(ubatch_views, "dbo_current_ubatch_id", lambda: ubatch["id"])
    monkeypatch.setattr(
        ubatch_views, "dbo_current_ubatch_token_offset", lambda: ubatch["offset"]
    )
    model = _Model()
    full = model.topk_indices_buffer
    # The model and both layers hold the buffer, the Linear does not.
    assert ubatch_views.install_ubatch_views(model) == 3
    layer = model.layers[0]
    assert isinstance(layer, _Layer)
    assert layer.topk_indices_buffer.data_ptr() == full.data_ptr()

    # Microbatch 1 starts at token 6: its writes land in rows 6 and up.
    ubatch.update(id=1, offset=6)
    layer.topk_indices_buffer[:3] = 7
    layer._topk_ragged_cache = {"r": 1}
    assert full[6:9].eq(7).all() and full[:6].eq(0).all()

    # Microbatch 0 sees its own rows and an empty cache.
    ubatch.update(id=0, offset=0)
    assert layer.topk_indices_buffer[:6].eq(0).all()
    assert layer._topk_ragged_cache == {}
    layer._topk_ragged_cache["r"] = 0
    ubatch.update(id=1, offset=6)
    assert layer._topk_ragged_cache == {"r": 1}
