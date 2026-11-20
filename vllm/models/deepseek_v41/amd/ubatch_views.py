# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-microbatch views of the step buffers that DeepSeek V4.1 layers share.

All layers share one top-k index buffer and one candidate block buffer, and
each index source layer caches the ragged form of its indices for the layers
below it. Under dual batch overlap the two microbatches take turns at each TP
all-reduce, so microbatch 1 can run an index source layer while microbatch 0
still has layers to run that read the same rows. Each microbatch therefore
uses the buffer rows at its own token offset (the two microbatches together
never hold more tokens than the step, which the buffers are sized for), and
its own cache dicts. Outside a microbatch the offset is 0 and the id is 0, so
nothing changes.
"""

from typing import Any

import torch

from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_current_ubatch_token_offset,
)


def _rows(full: torch.Tensor | None) -> torch.Tensor | None:
    if full is None:
        return None
    start = dbo_current_ubatch_token_offset()
    return full if start == 0 else full[start:]


class UbatchRowsBuffer:
    """A data descriptor for a shared step buffer attribute. Reads return the
    calling microbatch's rows, writes store the whole buffer."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.key = f"_ubatch_full_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return _rows(obj.__dict__.get(self.key))

    def __set__(self, obj: Any, value: torch.Tensor | None) -> None:
        obj.__dict__[self.key] = value


class UbatchDict:
    """A data descriptor for a per-step cache dict: one dict per microbatch."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.key = f"_ubatch_dicts_{name}"

    def _dicts(self, obj: Any) -> dict[int, dict]:
        return obj.__dict__.setdefault(self.key, {})

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return self._dicts(obj).setdefault(dbo_current_ubatch_id(), {})

    def __set__(self, obj: Any, value: dict) -> None:
        self._dicts(obj)[dbo_current_ubatch_id()] = value


# Attributes that hold a shared step buffer, and attributes that hold a
# per-step cache dict.
_ROW_ATTRS = ("topk_indices_buffer", "candidate_block_buffer", "candidate_blocks")
_DICT_ATTRS = ("_topk_ragged_cache", "_prefill_topk_ragged_cache")
_VIEW_CLASSES: dict[type, type] = {}


def install_ubatch_views(model: torch.nn.Module) -> int:
    """Give every module of `model` that holds a shared step buffer or index
    cache per-microbatch views of it, by moving the module to a subclass with
    the descriptors above. Only for dual batch overlap. Returns the number of
    modules changed."""
    changed = 0
    for module in model.modules():
        rows = [
            a for a in _ROW_ATTRS if isinstance(module.__dict__.get(a), torch.Tensor)
        ]
        dicts = [a for a in _DICT_ATTRS if isinstance(module.__dict__.get(a), dict)]
        if not rows and not dicts:
            continue
        cls = type(module)
        view_cls = _VIEW_CLASSES.get(cls)
        if view_cls is None:
            attrs: dict[str, Any] = {a: UbatchRowsBuffer() for a in _ROW_ATTRS}
            attrs.update({a: UbatchDict() for a in _DICT_ATTRS})
            view_cls = type(cls.__name__, (cls,), attrs)
            for name, descriptor in attrs.items():
                descriptor.__set_name__(view_cls, name)
            _VIEW_CLASSES[cls] = view_cls
        saved = {a: module.__dict__.pop(a) for a in (*rows, *dicts)}
        module.__class__ = view_cls
        for a, value in saved.items():
            setattr(module, a, value)
        changed += 1
    return changed
