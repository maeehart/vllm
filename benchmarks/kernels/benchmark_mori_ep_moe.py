#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ========================================================================
# TOOL: benchmark_mori_ep_moe.py
# PURPOSE: Benchmark MORI-EP + AITER MoE backend vs alternatives
# USAGE: python benchmarks/kernels/benchmark_mori_ep_moe.py [options]
# ARGUMENTS:
#   --model: Model name or path (default: deepseek-ai/DeepSeek-R1)
#   --num-tokens: Number of tokens to benchmark (default: 4096)
#   --hidden-size: Hidden dimension (default: 7168)
#   --num-experts: Total number of experts (default: 256)
#   --topk: Number of experts per token (default: 8)
#   --ep-size: Expert parallelism size (default: 8)
#   --num-iters: Number of iterations (default: 100)
#   --warmup-iters: Number of warmup iterations (default: 10)
#   --backends: Backends to benchmark (default: all available)
# EXAMPLES:
#   python benchmarks/kernels/benchmark_mori_ep_moe.py --ep-size 8
#   python benchmarks/kernels/benchmark_mori_ep_moe.py --num-tokens 128 --topk 8
# DEPENDENCIES: torch, vllm, mori (optional), aiter (optional)
# ========================================================================
"""
Benchmark script for MORI-EP + AITER MoE backend.

This script benchmarks the MORI-EP dispatch/combine + AITER compute
combination against other MoE backends available in vLLM.

Target workload: DeepSeek-R1 style
- 256 experts
- topk=8
- hidden_size=7168
- ISL 70k, OSL 300, concurrency 10
"""

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    backend: str
    num_tokens: int
    hidden_size: int
    num_experts: int
    topk: int
    ep_size: int
    latency_us: float
    throughput_tokens_per_sec: float
    dispatch_latency_us: float | None = None
    compute_latency_us: float | None = None
    combine_latency_us: float | None = None


def check_backend_availability() -> dict[str, bool]:
    """Check which backends are available."""
    available = {
        "triton": True,  # Always available
        "aiter": False,
        "mori_ep": False,
        "deepep_ht": False,
        "deepep_ll": False,
        "pplx": False,
    }

    # Check AITER
    try:
        from vllm._aiter_ops import rocm_aiter_ops

        available["aiter"] = rocm_aiter_ops.is_fused_moe_enabled()
    except ImportError:
        pass

    # Check MORI-EP
    try:
        from vllm.model_executor.layers.fused_moe.mori_prepare_finalize import (
            is_mori_ep_available,
        )

        available["mori_ep"] = is_mori_ep_available()
    except ImportError:
        pass

    # Check DeepEP
    try:
        import deep_ep

        available["deepep_ht"] = True
        available["deepep_ll"] = True
    except ImportError:
        pass

    # Check PPLX
    try:
        import pplx_kernels

        available["pplx"] = True
    except ImportError:
        pass

    return available


def create_test_data(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test data for MoE benchmark."""
    # Input activations
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Router output (gating scores)
    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    # Compute topk
    topk_weights, topk_ids = torch.topk(gating_output, topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    # Expert weights (simplified - actual weights would be loaded from model)
    intermediate_size = hidden_size * 4  # Typical FFN expansion
    w1 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
    )
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )

    return x, topk_weights, topk_ids, w1, w2


def benchmark_triton_moe(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_iters: int,
    warmup_iters: int,
) -> float:
    """Benchmark Triton MoE kernel."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # Warmup
    for _ in range(warmup_iters):
        _ = fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        _ = fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
        )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return (elapsed_ms / num_iters) * 1000  # Convert to microseconds


def benchmark_aiter_moe(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_iters: int,
    warmup_iters: int,
) -> float:
    """Benchmark AITER MoE kernel."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        rocm_aiter_fused_experts,
    )

    # Warmup
    for _ in range(warmup_iters):
        _ = rocm_aiter_fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        _ = rocm_aiter_fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
        )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return (elapsed_ms / num_iters) * 1000  # Convert to microseconds


def benchmark_mori_ep_dispatch_combine(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_size: int,
    num_experts: int,
    num_iters: int,
    warmup_iters: int,
) -> tuple[float, float]:
    """
    Benchmark MORI-EP dispatch and combine operations separately.

    Note: This requires MORI shmem to be initialized and running on EP8+
    configuration. For single-GPU testing, use benchmark_mori_ep_ops_mock().

    Returns:
        Tuple of (dispatch_latency_us, combine_latency_us)
    """
    try:
        from vllm.model_executor.layers.fused_moe.mori_utils import (
            MoriEpConfig,
            compute_num_local_experts,
            create_mori_ep_op,
        )
    except ImportError:
        logger.warning("MORI-EP not available for dispatch/combine benchmark")
        return 0.0, 0.0

    num_tokens, hidden_size = x.shape
    num_local_experts = compute_num_local_experts(num_experts, ep_size)
    topk = topk_ids.shape[1]

    # Create MORI EP config and operator
    # Note: This will fail on single-GPU as MORI requires EP8+
    try:
        config = MoriEpConfig(
            rank=0,
            world_size=ep_size,
            hidden_dim=hidden_size,
            max_num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            dtype=x.dtype,
        )
        ep_op = create_mori_ep_op(config)
    except Exception as e:
        logger.warning(
            "Failed to create MORI EP operator (requires EP8+ with shmem init): %s", e
        )
        return 0.0, 0.0

    # Warmup dispatch
    # dispatch(input, weights, scales, indices, block_num, warp_per_block)
    for _ in range(warmup_iters):
        dispatch_result = ep_op.dispatch(
            input=x,
            weights=topk_weights,
            scales=None,
            indices=topk_ids.to(torch.int32),
        )
    torch.cuda.synchronize()

    # Benchmark dispatch
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        dispatch_result = ep_op.dispatch(
            input=x,
            weights=topk_weights,
            scales=None,
            indices=topk_ids.to(torch.int32),
        )
    end_event.record()
    torch.cuda.synchronize()

    dispatch_ms = start_event.elapsed_time(end_event)
    dispatch_latency_us = (dispatch_ms / num_iters) * 1000

    # Create dummy expert output for combine benchmark
    recv_x = dispatch_result[0]
    expert_output = torch.randn_like(recv_x)

    # Warmup combine
    # combine(input, weights, indices, block_num, warp_per_block, call_reset)
    for _ in range(warmup_iters):
        combine_result = ep_op.combine(
            input=expert_output,
            weights=topk_weights,
            indices=topk_ids.to(torch.int32),
            call_reset=True,
        )
    torch.cuda.synchronize()

    # Benchmark combine
    start_event.record()
    for _ in range(num_iters):
        combine_result = ep_op.combine(
            input=expert_output,
            weights=topk_weights,
            indices=topk_ids.to(torch.int32),
            call_reset=True,
        )
    end_event.record()
    torch.cuda.synchronize()

    combine_ms = start_event.elapsed_time(end_event)
    combine_latency_us = (combine_ms / num_iters) * 1000

    return dispatch_latency_us, combine_latency_us


def run_benchmark(
    backend: str,
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    ep_size: int,
    num_iters: int,
    warmup_iters: int,
    dtype: torch.dtype = torch.bfloat16,
) -> BenchmarkResult | None:
    """Run benchmark for a specific backend."""
    logger.info(
        "Benchmarking %s: tokens=%d, hidden=%d, experts=%d, topk=%d, ep=%d",
        backend,
        num_tokens,
        hidden_size,
        num_experts,
        topk,
        ep_size,
    )

    # Create test data
    x, topk_weights, topk_ids, w1, w2 = create_test_data(
        num_tokens, hidden_size, num_experts, topk, dtype
    )

    latency_us = 0.0
    dispatch_latency_us = None
    compute_latency_us = None
    combine_latency_us = None

    try:
        if backend == "triton":
            latency_us = benchmark_triton_moe(
                x, topk_weights, topk_ids, w1, w2, num_iters, warmup_iters
            )
        elif backend == "aiter":
            latency_us = benchmark_aiter_moe(
                x, topk_weights, topk_ids, w1, w2, num_iters, warmup_iters
            )
        elif backend == "mori_ep":
            # Benchmark dispatch/combine separately
            dispatch_latency_us, combine_latency_us = benchmark_mori_ep_dispatch_combine(
                x, topk_weights, topk_ids, ep_size, num_experts, num_iters, warmup_iters
            )
            # Also benchmark AITER compute
            compute_latency_us = benchmark_aiter_moe(
                x, topk_weights, topk_ids, w1, w2, num_iters, warmup_iters
            )
            # Total latency is sum of all phases
            latency_us = (
                (dispatch_latency_us or 0)
                + (compute_latency_us or 0)
                + (combine_latency_us or 0)
            )
        else:
            logger.warning("Unknown backend: %s", backend)
            return None
    except Exception as e:
        logger.error("Error benchmarking %s: %s", backend, e)
        return None

    # Calculate throughput
    throughput = (num_tokens / latency_us) * 1e6  # tokens per second

    # Clean up
    del x, topk_weights, topk_ids, w1, w2
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult(
        backend=backend,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts=num_experts,
        topk=topk,
        ep_size=ep_size,
        latency_us=latency_us,
        throughput_tokens_per_sec=throughput,
        dispatch_latency_us=dispatch_latency_us,
        compute_latency_us=compute_latency_us,
        combine_latency_us=combine_latency_us,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("MORI-EP + AITER MoE Benchmark Results")
    print("=" * 100)

    # Header
    print(
        f"{'Backend':<20} {'Tokens':<10} {'Latency (µs)':<15} "
        f"{'Throughput (tok/s)':<20} {'Dispatch (µs)':<15} "
        f"{'Compute (µs)':<15} {'Combine (µs)':<15}"
    )
    print("-" * 100)

    for r in results:
        dispatch = f"{r.dispatch_latency_us:.1f}" if r.dispatch_latency_us else "N/A"
        compute = f"{r.compute_latency_us:.1f}" if r.compute_latency_us else "N/A"
        combine = f"{r.combine_latency_us:.1f}" if r.combine_latency_us else "N/A"

        print(
            f"{r.backend:<20} {r.num_tokens:<10} {r.latency_us:<15.1f} "
            f"{r.throughput_tokens_per_sec:<20.0f} {dispatch:<15} "
            f"{compute:<15} {combine:<15}"
        )

    print("=" * 100)


def main(args: argparse.Namespace) -> None:
    """Main benchmark function."""
    print(f"Running MORI-EP + AITER MoE Benchmark")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Num tokens: {args.num_tokens}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num experts: {args.num_experts}")
    print(f"  Top-k: {args.topk}")
    print(f"  EP size: {args.ep_size}")
    print(f"  Iterations: {args.num_iters}")
    print(f"  Warmup iterations: {args.warmup_iters}")

    # Check backend availability
    available = check_backend_availability()
    print(f"\nBackend availability:")
    for backend, is_available in available.items():
        status = "✓" if is_available else "✗"
        print(f"  {backend}: {status}")

    # Determine which backends to benchmark
    if args.backends:
        backends = args.backends.split(",")
    else:
        # Default: benchmark all available backends
        backends = [b for b, avail in available.items() if avail]

    # Filter to only available backends
    backends = [b for b in backends if available.get(b, False)]

    if not backends:
        print("\nNo backends available for benchmarking!")
        return

    print(f"\nBenchmarking backends: {', '.join(backends)}")

    # Run benchmarks
    results = []
    for backend in backends:
        result = run_benchmark(
            backend=backend,
            num_tokens=args.num_tokens,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            topk=args.topk,
            ep_size=args.ep_size,
            num_iters=args.num_iters,
            warmup_iters=args.warmup_iters,
        )
        if result:
            results.append(result)

    # Print results
    if results:
        print_results(results)
    else:
        print("\nNo benchmark results to display.")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark MORI-EP + AITER MoE backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1",
        help="Model name or path",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4096,
        help="Number of tokens to benchmark",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=7168,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Total number of experts",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Number of experts per token",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=8,
        help="Expert parallelism size",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="Comma-separated list of backends to benchmark (default: all available)",
    )

    args = parser.parse_args()
    main(args)

