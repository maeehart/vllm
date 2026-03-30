#!/bin/bash
# Run ROCm-relevant tests inside an existing DeepSeek V3.2 vLLM image.
# Mounts the repo so pytest uses current sources (including local patches).
#
# Requires: Docker on Linux with ROCm (e.g. MI300/MI355X). Passes through
# /dev/kfd and each /dev/dri/card* and /dev/dri/renderD* (required; /dev/dri
# alone is a directory and cannot be used with --device).
#
# Multi-GPU (e.g. 8x MI355X): use HIP_VISIBLE_DEVICES to pick GPU(s) for tests.
#   HIP_VISIBLE_DEVICES=0 ./docker/run_deepseek_v32_tests.sh IMAGE ...
#
# VLLM_TARGET_DEVICE defaults to rocm so editable installs do not hit setup.py's
# CUDA/NVCC path (PyTorch ROCm may still report torch.version.cuda).
#   HIP_VISIBLE_DEVICES=0,1 ./docker/run_deepseek_v32_tests.sh IMAGE ...
#
# Optional: VLLM_DOCKER_ROCM_IPC=1 adds --ipc=host (sometimes helps ROCm in containers).
#
# Usage:
#   ./docker/run_deepseek_v32_tests.sh [IMAGE] [PYTEST_ARGS...]
#
# Example:
#   ./docker/run_deepseek_v32_tests.sh vllm-dsv32:rocm60326_full_fix \
#     tests/kernels/attention/test_deepgemm_attention.py -v --tb=short

set -euo pipefail

IMAGE="${1:-vllm-deepseek-v32:latest}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -eq 0 ]]; then
  set -- tests/kernels/attention/test_deepgemm_attention.py -v --tb=short
fi

echo "=== Running tests in $IMAGE ==="
echo "  Repo: $REPO_ROOT"
echo "  HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-0}"
echo "  Args: $*"
echo ""

EXTRA_DEVICES=()
if [[ -e /dev/kfd ]]; then EXTRA_DEVICES+=(--device /dev/kfd); fi
# ROCm needs each DRM device node, not the /dev/dri directory
shopt -s nullglob
for f in /dev/dri/card* /dev/dri/renderD*; do
  EXTRA_DEVICES+=(--device "$f")
done
shopt -u nullglob

GROUP_ADD=(--group-add video)
if getent group render >/dev/null 2>&1; then
  GROUP_ADD+=(--group-add render)
fi

IPC_ARGS=()
if [[ "${VLLM_DOCKER_ROCM_IPC:-}" == "1" ]]; then
  IPC_ARGS+=(--ipc=host)
fi

docker run --rm -i \
  "${IPC_ARGS[@]}" \
  "${EXTRA_DEVICES[@]}" \
  "${GROUP_ADD[@]}" \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  -e VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}" \
  -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-}" \
  -v "$REPO_ROOT:/src/vllm" \
  -w /src/vllm \
  "$IMAGE" \
  bash -c 'set -euo pipefail
    export AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1
    export VLLM_ROCM_USE_AITER=1
    # Use image ROCm PyTorch (HIP); isolated build env has CPU torch → unknown device in setup.py.
    python3 -m pip install -q -e . --no-build-isolation
    exec python3 -m pytest "$@"
  ' bash "$@"
