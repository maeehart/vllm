# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.envs import disable_envs_cache
from vllm.model_executor.models.config import _verify_glm52_cpx_config
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def _disable_env_cache():
    disable_envs_cache()
    yield
    disable_envs_cache()


def _config(**overrides):
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_size=6144,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=1,
        quantization_config={"quant_method": "quark"},
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=16,
        enable_expert_parallel=False,
    )
    for name, value in overrides.items():
        target, field = name.split("__", 1)
        setattr(
            hf_config if target == "model" else parallel_config,
            field,
            value,
        )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        parallel_config=parallel_config,
    )


def _enable(monkeypatch: pytest.MonkeyPatch, tp_size: int = 16) -> None:
    monkeypatch.setenv("VLLM_ROCM_GLM52_CPX_TP16", "1" if tp_size == 16 else "0")
    monkeypatch.setenv("VLLM_ROCM_GLM52_CPX_TP32", "1" if tp_size == 32 else "0")
    monkeypatch.setenv("VLLM_ROCM_GLM52_CPX_TP64", "1" if tp_size == 64 else "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    disable_envs_cache()


@pytest.mark.parametrize("tp_size", [16, 32, 64])
def test_glm52_cpx_config_accepts_validated_shape(monkeypatch, tp_size):
    _enable(monkeypatch, tp_size)
    _verify_glm52_cpx_config(_config(parallel__tensor_parallel_size=tp_size))


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"parallel__tensor_parallel_size": 8}, "tensor_parallel_size=8"),
        ({"model__moe_intermediate_size": 4096}, "moe_intermediate_size=4096"),
        ({"model__n_routed_experts": 128}, "n_routed_experts=128"),
        ({"parallel__enable_expert_parallel": True}, "enable_expert_parallel=True"),
    ],
)
def test_glm52_cpx_tp16_config_rejects_drift(
    monkeypatch,
    override,
    expected_error,
):
    _enable(monkeypatch)
    with pytest.raises(ValueError, match=expected_error):
        _verify_glm52_cpx_config(_config(**override))


def test_glm52_cpx_tp16_config_requires_aiter(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    with pytest.raises(ValueError, match="VLLM_ROCM_USE_AITER=False"):
        _verify_glm52_cpx_config(_config())


def test_glm52_cpx_tp16_config_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_GLM52_CPX_TP16", raising=False)
    monkeypatch.delenv("VLLM_ROCM_GLM52_CPX_TP32", raising=False)
    monkeypatch.delenv("VLLM_ROCM_GLM52_CPX_TP64", raising=False)
    _verify_glm52_cpx_config(
        _config(
            model__n_routed_experts=128,
            parallel__tensor_parallel_size=8,
        )
    )


def test_glm52_cpx_config_rejects_multiple_topology_flags(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("VLLM_ROCM_GLM52_CPX_TP64", "1")
    with pytest.raises(ValueError, match="Only one GLM-5.2 CPX topology"):
        _verify_glm52_cpx_config(_config())


def test_glm52_cpx_keeps_aiter_a4w4_intermediate_at_128(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        mxfp4_round_up_hidden_size_and_intermediate_size,
    )

    _enable(monkeypatch, 16)
    assert mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        6144,
        128,
    ) == (6144, 128)


def test_glm52_cpx_tp64_keeps_native_aiter_a4w4_intermediate(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        mxfp4_round_up_hidden_size_and_intermediate_size,
    )

    _enable(monkeypatch, 64)
    assert mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        6144,
        32,
    ) == (6144, 32)
