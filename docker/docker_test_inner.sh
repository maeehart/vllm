#!/bin/bash
# Invoked inside the container by run_deepseek_v32_tests.sh (do not run directly on host).
set -euo pipefail
cd /src/vllm

export AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1
export VLLM_ROCM_USE_AITER=1

if [[ "${VLLM_DOCKER_TEST_EDITABLE:-0}" == "1" ]]; then
  # Rebuilds extensions; needs free disk and time. Uses image ROCm torch.
  python3 -m pip install -q -e . --no-build-isolation
elif [[ -d tests/vllm_test_utils ]]; then
  python3 -m pip install -q -e tests/vllm_test_utils 2>/dev/null || true
fi

exec python3 -m pytest "$@"
