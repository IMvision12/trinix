"""
Benchmark script for optimized RoPE (Rotary Position Embedding) kernel.

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

from trinix.kernels.rope_kernel import TritonRoPEKernel

TRITON_ROPE_AVAILABLE = True


class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    def __init__(self, batch: int, seq_len: int, num_heads: int, head_dim: int):
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __repr__(self):
        return f"Batch={self.batch}, SeqLen={self.seq_len}, Heads={self.num_heads}, HeadDim={self.head_dim}"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute cosine and sine frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def benchmark_forward(
    model_fn, q, k, cos, sin, num_warmup: int = 10, num_iterations: int = 100
) -> float:
    """Benchmark forward pass."""
    for _ in range(num_warmup):
        _ = model_fn(q, k, cos, sin)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model_fn(q, k, cos, sin)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return statistics.median(times)


def benchmark_backward(
    model_fn, q, k, cos, sin, num_warmup: int = 10, num_iterations: int = 100
) -> float:
    """Benchmark backward pass."""
    for _ in range(num_warmup):
        q.grad = None
        k.grad = None
        q_rot, k_rot = model_fn(q, k, cos, sin)
        (q_rot.sum() + k_rot.sum()).backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        q.grad = None
        k.grad = None
        q_rot, k_rot = model_fn(q, k, cos, sin)
        torch.cuda.synchronize()
        start = time.perf_counter()
        (q_rot.sum() + k_rot.sum()).backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return statistics.median(times)


def benchmark_total(
    model_fn, q, k, cos, sin, num_warmup: int = 10, num_iterations: int = 100
) -> float:
    """Benchmark forward + backward pass."""
    for _ in range(num_warmup):
        q.grad = None
        k.grad = None
        q_rot, k_rot = model_fn(q, k, cos, sin)
        (q_rot.sum() + k_rot.sum()).backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        q.grad = None
        k.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        q_rot, k_rot = model_fn(q, k, cos, sin)
        (q_rot.sum() + k_rot.sum()).backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return statistics.median(times)


def apply_rotary_emb_pytorch(q, k, cos, sin):
    """PyTorch reference implementation of RoPE.
    
    Matches the Triton kernel implementation:
    - Split features into two halves: x1 = first half, x2 = second half
    - Apply rotation: x1' = x1*cos - x2*sin, x2' = x1*sin + x2*cos
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    half_dim = head_dim // 2
    
    # Split into two halves
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    
    # Expand cos and sin to match tensor dimensions
    # cos, sin shape: (seq_len, head_dim//2)
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(2)
    
    # Apply rotation matrix
    q1_rot = q1 * cos - q2 * sin
    q2_rot = q1 * sin + q2 * cos
    
    k1_rot = k1 * cos - k2 * sin
    k2_rot = k1 * sin + k2 * cos
    
    # Concatenate back
    q_rot = torch.cat([q1_rot, q2_rot], dim=-1)
    k_rot = torch.cat([k1_rot, k2_rot], dim=-1)
    
    return q_rot, k_rot


def test_rope_correctness():
    """Test RoPE correctness."""
    if not TRITON_ROPE_AVAILABLE:
        print("Skipping RoPE correctness test (Triton not available)")
        return

    print("Running RoPE correctness test...")
    device = torch.device("cuda")

    test_configs = [
        (2, 128, 8, 64),
        (1, 512, 12, 64),
        (4, 1024, 16, 128),
    ]

    for batch, seq_len, num_heads, head_dim in test_configs:
        q = torch.randn(
            batch, seq_len, num_heads, head_dim, dtype=torch.float32, device=device, requires_grad=True
        )
        k = torch.randn(
            batch, seq_len, num_heads, head_dim, dtype=torch.float32, device=device, requires_grad=True
        )
        
        cos, sin = precompute_freqs_cis(head_dim, seq_len)
        cos = cos.to(device)
        sin = sin.to(device)

        # PyTorch reference
        q_ref, k_ref = apply_rotary_emb_pytorch(q, k, cos, sin)

        # Triton
        q_triton = q.clone().detach().requires_grad_(True)
        k_triton = k.clone().detach().requires_grad_(True)
        q_tri, k_tri = TritonRoPEKernel.apply(q_triton, k_triton, cos, sin)

        forward_match = torch.allclose(q_ref, q_tri, rtol=1e-3, atol=1e-3) and torch.allclose(
            k_ref, k_tri, rtol=1e-3, atol=1e-3
        )
        
        # Debug: print differences if not matching
        if not forward_match:
            q_diff = (q_ref - q_tri).abs().max().item()
            k_diff = (k_ref - k_tri).abs().max().item()
            print(f"    Forward diff - Q: {q_diff:.6f}, K: {k_diff:.6f}")

        # Test backward
        grad_q = torch.randn_like(q_ref)
        grad_k = torch.randn_like(k_ref)
        
        # PyTorch backward
        loss_ref = (q_ref * grad_q).sum() + (k_ref * grad_k).sum()
        loss_ref.backward()
        
        # Triton backward
        loss_tri = (q_tri * grad_q).sum() + (k_tri * grad_k).sum()
        loss_tri.backward()

        backward_match = torch.allclose(q.grad, q_triton.grad, rtol=1e-3, atol=1e-3) and torch.allclose(
            k.grad, k_triton.grad, rtol=1e-3, atol=1e-3
        )
        
        # Debug: print differences if not matching
        if not backward_match:
            q_grad_diff = (q.grad - q_triton.grad).abs().max().item()
            k_grad_diff = (k.grad - k_triton.grad).abs().max().item()
            print(f"    Backward diff - Q: {q_grad_diff:.6f}, K: {k_grad_diff:.6f}")

        status = "✓" if (forward_match and backward_match) else "✗"
        print(
            f"  {status} Shape ({batch}, {seq_len}, {num_heads}, {head_dim}): "
            f"Forward={'✓' if forward_match else '✗'}, Backward={'✓' if backward_match else '✗'}"
        )

    print()


def benchmark_rope():
    """Main RoPE benchmark function."""
    print("=" * 100)
    print(" RoPE (Rotary Position Embedding) Performance Benchmark: PyTorch vs Triton")
    print("=" * 100)
    print()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        return

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"Triton Available: {TRITON_ROPE_AVAILABLE}")
    print()

    test_rope_correctness()

    configs = [
        # Standard transformer sizes
        BenchmarkConfig(batch=1, seq_len=2048, num_heads=12, head_dim=64),   # GPT-2 small
        BenchmarkConfig(batch=1, seq_len=4096, num_heads=16, head_dim=64),   # GPT-2 medium
        BenchmarkConfig(batch=1, seq_len=8192, num_heads=20, head_dim=128),  # GPT-2 large
        BenchmarkConfig(batch=1, seq_len=16384, num_heads=32, head_dim=128), # GPT-3
    ]

    device = torch.device("cuda")
    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"Running benchmark {i}/{len(configs)}: {config}")

        q = torch.randn(
            config.batch,
            config.seq_len,
            config.num_heads,
            config.head_dim,
            dtype=torch.float16,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            config.batch,
            config.seq_len,
            config.num_heads,
            config.head_dim,
            dtype=torch.float16,
            device=device,
            requires_grad=True,
        )
        
        cos, sin = precompute_freqs_cis(config.head_dim, config.seq_len)
        cos = cos.to(device).to(torch.float16)
        sin = sin.to(device).to(torch.float16)

        results = {"config": config, "pytorch": {}, "triton": {}}

        print(f"  Benchmarking PyTorch...")
        results["pytorch"]["forward"] = benchmark_forward(
            apply_rotary_emb_pytorch, q, k, cos, sin, 20, 100
        )
        results["pytorch"]["backward"] = benchmark_backward(
            apply_rotary_emb_pytorch, q, k, cos, sin, 20, 100
        )
        results["pytorch"]["total"] = benchmark_total(
            apply_rotary_emb_pytorch, q, k, cos, sin, 20, 100
        )

        if TRITON_ROPE_AVAILABLE:
            print(f"  Benchmarking Triton...")
            results["triton"]["forward"] = benchmark_forward(
                TritonRoPEKernel.apply, q, k, cos, sin, 20, 100
            )
            results["triton"]["backward"] = benchmark_backward(
                TritonRoPEKernel.apply, q, k, cos, sin, 20, 100
            )
            results["triton"]["total"] = benchmark_total(
                TritonRoPEKernel.apply, q, k, cos, sin, 20, 100
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
            for pass_type in ["forward", "backward", "total"]:
                pytorch_time = results["pytorch"][pass_type]
                triton_time = results["triton"][pass_type]
                speedup = pytorch_time / triton_time
                print(
                    f"{'RoPE':<20} {pass_type.capitalize():<10} "
                    f"{pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:.2f}x"
                )
        print()

    # Summary and comparison
    if TRITON_ROPE_AVAILABLE and all_results:
        print("=" * 100)
        print(" Summary Statistics")
        print("=" * 100)

        forward_speedups = [
            r["pytorch"]["forward"] / r["triton"]["forward"] for r in all_results if r["triton"]
        ]
        backward_speedups = [
            r["pytorch"]["backward"] / r["triton"]["backward"]
            for r in all_results
            if r["triton"]
        ]
        total_speedups = [
            r["pytorch"]["total"] / r["triton"]["total"] for r in all_results if r["triton"]
        ]

        print(
            f"Forward:  Avg={statistics.mean(forward_speedups):.2f}x, "
            f"Median={statistics.median(forward_speedups):.2f}x"
        )
        print(
            f"Backward: Avg={statistics.mean(backward_speedups):.2f}x, "
            f"Median={statistics.median(backward_speedups):.2f}x"
        )
        print(
            f"Total:    Avg={statistics.mean(total_speedups):.2f}x, "
            f"Median={statistics.median(total_speedups):.2f}x"
        )
        print()

        print("=" * 100)
        print(" RoPE Analysis")
        print("=" * 100)
        print()
        print("RoPE (Rotary Position Embedding) encodes position information:")
        print("  • Applies 2D rotation to feature pairs based on position")
        print("  • No learnable parameters - uses precomputed cos/sin values")
        print("  • Rotation angle increases with position and decreases with dimension")
        print("  • Used in GPT-Neo, GPT-J, LLaMA, and many modern LLMs")
        print()
        print("Key properties:")
        print("  • Relative position encoding (rotation difference encodes distance)")
        print("  • Extrapolates to longer sequences than training")
        print("  • Efficient: O(d) per position vs O(n²) for learned embeddings")
        print()
        print("Optimizations applied (inspired by LayerNorm patterns):")
        print()
        print("  1. Memory Access Optimizations:")
        print("     • Coalesced memory access for paired features")
        print("     • Efficient stride handling for multi-dimensional tensors")
        print("     • Minimal redundant loads")
        print()
        print("  2. Computational Optimizations:")
        print("     • Fused rotation computation for feature pairs")
        print("     • Reused cos/sin values across batch and heads")
        print("     • Efficient inverse rotation for backward pass")
        print()
        print("  3. Parallelization:")
        print("     • 3D grid parallelism (batch, sequence, heads)")
        print("     • Optimal block sizes for head dimension")
        print("     • Reused kernel configuration")
        print()
        print("Why RoPE benefits from Triton:")
        print("  • Fused rotation matrix application (4 ops → 1 kernel)")
        print("  • Coalesced memory access for feature pairs")
        print("  • No intermediate tensors for rotation")
        print("  • Efficient 3D parallelism across batch/sequence/heads")
        print("  • Shared cos/sin values reduce memory bandwidth")


if __name__ == "__main__":
    benchmark_rope()
