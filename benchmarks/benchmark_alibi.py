"""
Benchmark script for optimized ALiBi (Attention with Linear Biases) kernel.

Compares performance against PyTorch reference implementation.
"""

import torch
import torch.nn as nn
import time
import statistics
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trinix.kernels.alibi_kernel import TritonALiBiKernel

TRITON_ALIBI_AVAILABLE = True


class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    def __init__(self, batch: int, num_heads: int, seq_len: int):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len

    def __repr__(self):
        return f"Batch={self.batch}, Heads={self.num_heads}, SeqLen={self.seq_len}"


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each attention head."""
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
            : num_heads - closest_power_of_2
        ]
        return torch.tensor(slopes + extra_slopes)


def benchmark_forward(
    model_fn, slopes, batch_size, num_heads, seq_len, num_warmup: int = 10, num_iterations: int = 100
) -> float:
    """Benchmark forward pass."""
    for _ in range(num_warmup):
        _ = model_fn(slopes, batch_size, num_heads, seq_len)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model_fn(slopes, batch_size, num_heads, seq_len)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return statistics.median(times)


def compute_alibi_bias_pytorch(slopes, batch_size, num_heads, seq_len):
    """PyTorch reference implementation of ALiBi bias."""
    # Create position indices
    i = torch.arange(seq_len, device=slopes.device).unsqueeze(1)  # (seq_len, 1)
    j = torch.arange(seq_len, device=slopes.device).unsqueeze(0)  # (1, seq_len)
    
    # Compute distances: |i - j|
    distances = torch.abs(i - j).float()  # (seq_len, seq_len)
    
    # Apply slopes: -slope * |i - j|
    slopes = slopes.view(num_heads, 1, 1)  # (num_heads, 1, 1)
    bias = -slopes * distances.unsqueeze(0)  # (num_heads, seq_len, seq_len)
    
    # Expand for batch
    bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch, num_heads, seq_len, seq_len)
    
    return bias


def test_alibi_correctness():
    """Test ALiBi correctness."""
    if not TRITON_ALIBI_AVAILABLE:
        print("Skipping ALiBi correctness test (Triton not available)")
        return

    print("Running ALiBi correctness test...")
    device = torch.device("cuda")

    test_configs = [
        (2, 8, 128),
        (1, 12, 512),
        (4, 16, 1024),
    ]

    for batch, num_heads, seq_len in test_configs:
        slopes = get_alibi_slopes(num_heads).to(device).to(torch.float32)

        # PyTorch reference
        bias_ref = compute_alibi_bias_pytorch(slopes, batch, num_heads, seq_len)

        # Triton
        bias_tri = TritonALiBiKernel.apply(slopes, batch, num_heads, seq_len)

        forward_match = torch.allclose(bias_ref, bias_tri, rtol=1e-3, atol=1e-3)

        status = "✓" if forward_match else "✗"
        print(
            f"  {status} Shape ({batch}, {num_heads}, {seq_len}, {seq_len}): "
            f"Forward={'✓' if forward_match else '✗'}"
        )

    print()


def benchmark_alibi():
    """Main ALiBi benchmark function."""
    print("=" * 100)
    print(" ALiBi (Attention with Linear Biases) Performance Benchmark: PyTorch vs Triton")
    print("=" * 100)
    print()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        return

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"Triton Available: {TRITON_ALIBI_AVAILABLE}")
    print()

    test_alibi_correctness()

    configs = [
        # Standard transformer sizes
        BenchmarkConfig(batch=1, num_heads=12, seq_len=2048),   # GPT-2 small
        BenchmarkConfig(batch=1, num_heads=16, seq_len=4096),   # GPT-2 medium
        BenchmarkConfig(batch=1, num_heads=20, seq_len=8192),   # GPT-2 large
        BenchmarkConfig(batch=1, num_heads=32, seq_len=16384),  # GPT-3
    ]

    device = torch.device("cuda")
    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"Running benchmark {i}/{len(configs)}: {config}")

        slopes = get_alibi_slopes(config.num_heads).to(device).to(torch.float16)

        results = {"config": config, "pytorch": {}, "triton": {}}

        print(f"  Benchmarking PyTorch...")
        results["pytorch"]["forward"] = benchmark_forward(
            compute_alibi_bias_pytorch, slopes, config.batch, config.num_heads, config.seq_len, 20, 100
        )

        if TRITON_ALIBI_AVAILABLE:
            print(f"  Benchmarking Triton...")
            results["triton"]["forward"] = benchmark_forward(
                TritonALiBiKernel.apply, slopes, config.batch, config.num_heads, config.seq_len, 20, 100
            )

        all_results.append(results)
        print()

    # Print results
    for results in all_results:
        config = results["config"]
        print("=" * 100)
        print(f" Configuration: {config}")
        print("=" * 100)
        print(
            f"{'Operation':<20} {'Pass':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}"
        )
        print("-" * 100)

        if results["triton"]:
            pytorch_time = results["pytorch"]["forward"]
            triton_time = results["triton"]["forward"]
            speedup = pytorch_time / triton_time
            print(
                f"{'ALiBi':<20} {'Forward':<10} "
                f"{pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:.2f}x"
            )
        print()

    # Summary and comparison
    if TRITON_ALIBI_AVAILABLE and all_results:
        print("=" * 100)
        print(" Summary Statistics")
        print("=" * 100)

        forward_speedups = [
            r["pytorch"]["forward"] / r["triton"]["forward"] for r in all_results if r["triton"]
        ]

        print(
            f"Forward:  Avg={statistics.mean(forward_speedups):.2f}x, "
            f"Median={statistics.median(forward_speedups):.2f}x"
        )
        print()

        print("=" * 100)
        print(" ALiBi Analysis")
        print("=" * 100)
        print()
        print("ALiBi (Attention with Linear Biases) adds position information:")
        print("  • Adds bias to attention scores: bias[i,j] = -slope * |i - j|")
        print("  • No learnable parameters - uses fixed slopes per head")
        print("  • Linear penalty based on distance between positions")
        print("  • Used in BLOOM, MPT, and other efficient transformers")
        print()
        print("Key properties:")
        print("  • Extrapolates to longer sequences than training")
        print("  • No position embeddings needed (saves parameters)")
        print("  • Different slopes per head provide diversity")
        print("  • Efficient: O(n²) bias computation, but can be fused with attention")
        print()
        print("Optimizations applied (inspired by LayerNorm patterns):")
        print()
        print("  1. Memory Access Optimizations:")
        print("     • Coalesced memory access for bias computation")
        print("     • Efficient stride handling for multi-dimensional tensors")
        print("     • Single load for slope values")
        print()
        print("  2. Computational Optimizations:")
        print("     • Fused distance computation and bias calculation")
        print("     • Vectorized operations for key positions")
        print("     • Efficient absolute value computation")
        print()
        print("  3. Parallelization:")
        print("     • 3D grid parallelism (batch, heads, query positions)")
        print("     • Optimal block sizes for sequence length")
        print("     • Reused kernel configuration")
        print()
        print("Why ALiBi benefits from Triton:")
        print("  • Fused distance and bias computation (2 ops → 1 kernel)")
        print("  • Coalesced memory access patterns")
        print("  • No intermediate tensors for distances")
        print("  • Efficient 3D parallelism across batch/heads/positions")
        print("  • Shared slope values reduce memory bandwidth")
        print()
        print("Note: ALiBi is typically fused with attention computation in practice,")
        print("      but this benchmark measures standalone bias generation.")


if __name__ == "__main__":
    benchmark_alibi()
