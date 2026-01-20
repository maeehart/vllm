#!/bin/bash
# ========================================================================
# WRAPPER: benchmark_mori_ep_markus.sh
# WRAPS: benchmarks/kernels/benchmark_mori_ep_markus.py
# PURPOSE: Run MORI-EP + AITER MoE benchmarks with standard configurations
# USAGE: ./scripts/benchmark_mori_ep_markus.sh [--config CONFIG] [--tokens N]
# ARGUMENTS:
#   --config CONFIG   Benchmark configuration: deepseek-r1, small, custom
#   --tokens N        Number of tokens (default: 4096)
#   --ep-size N       Expert parallelism size (default: 8)
#   --iters N         Number of iterations (default: 100)
#   --backends LIST   Comma-separated backends (default: all available)
# EXAMPLES:
#   ./scripts/benchmark_mori_ep_markus.sh --config deepseek-r1
#   ./scripts/benchmark_mori_ep_markus.sh --tokens 128 --ep-size 8
#   ./scripts/benchmark_mori_ep_markus.sh --backends triton,aiter,mori_ep
# DEPENDENCIES: python, torch, vllm
# ========================================================================

set -e

# Enable AITER for ROCm
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

# Default values
CONFIG="deepseek-r1"
NUM_TOKENS=4096
EP_SIZE=8
NUM_ITERS=100
WARMUP_ITERS=10
BACKENDS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --tokens)
            NUM_TOKENS="$2"
            shift 2
            ;;
        --ep-size)
            EP_SIZE="$2"
            shift 2
            ;;
        --iters)
            NUM_ITERS="$2"
            shift 2
            ;;
        --backends)
            BACKENDS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--config CONFIG] [--tokens N] [--ep-size N] [--iters N] [--backends LIST]"
            echo ""
            echo "Configurations:"
            echo "  deepseek-r1  DeepSeek-R1 style (256 experts, topk=8, hidden=7168)"
            echo "  small        Small test config (64 experts, topk=4, hidden=4096)"
            echo "  custom       Use command-line arguments for all parameters"
            echo ""
            echo "Examples:"
            echo "  $0 --config deepseek-r1"
            echo "  $0 --config small --tokens 128"
            echo "  $0 --backends triton,aiter,mori_ep"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set configuration based on preset
case $CONFIG in
    deepseek-r1)
        HIDDEN_SIZE=7168
        NUM_EXPERTS=256
        TOPK=8
        ;;
    small)
        HIDDEN_SIZE=4096
        NUM_EXPERTS=64
        TOPK=4
        ;;
    custom)
        # Use defaults or command-line overrides
        HIDDEN_SIZE=${HIDDEN_SIZE:-7168}
        NUM_EXPERTS=${NUM_EXPERTS:-256}
        TOPK=${TOPK:-8}
        ;;
    *)
        echo "Unknown config: $CONFIG"
        exit 1
        ;;
esac

# Build command
CMD="python benchmarks/kernels/benchmark_mori_ep_markus.py"
CMD="$CMD --num-tokens $NUM_TOKENS"
CMD="$CMD --hidden-size $HIDDEN_SIZE"
CMD="$CMD --num-experts $NUM_EXPERTS"
CMD="$CMD --topk $TOPK"
CMD="$CMD --ep-size $EP_SIZE"
CMD="$CMD --num-iters $NUM_ITERS"
CMD="$CMD --warmup-iters $WARMUP_ITERS"

if [[ -n "$BACKENDS" ]]; then
    CMD="$CMD --backends $BACKENDS"
fi

echo "=========================================="
echo "MORI-EP + AITER MoE Benchmark"
echo "=========================================="
echo "Configuration: $CONFIG"
echo "Tokens: $NUM_TOKENS"
echo "Hidden size: $HIDDEN_SIZE"
echo "Num experts: $NUM_EXPERTS"
echo "Top-k: $TOPK"
echo "EP size: $EP_SIZE"
echo "Iterations: $NUM_ITERS"
echo "=========================================="
echo ""
echo "Running: $CMD"
echo ""

# Run benchmark
cd "$(dirname "$0")/.."
exec $CMD

