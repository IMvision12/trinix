"""Example usage of Triton attention kernel in FlashLM."""

import torch

from flashlm.layers.attention import FastMultiHeadSelfAttention


def example_basic_usage():
    """Basic usage of Triton attention kernel."""
    print("=" * 60)
    print("Example 1: Basic Triton Attention")
    print("=" * 60)

    # Configuration
    batch_size = 4
    seq_len = 256
    embed_dim = 512
    num_heads = 8

    # Create attention layer with Triton kernel
    attention = FastMultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        kernel_type="triton",  # Use Triton kernel
        dropout=0.1,
        causal=False,
    )

    if torch.cuda.is_available():
        attention = attention.cuda()
        hidden_states = torch.randn(batch_size, seq_len, embed_dim).cuda()
        print(f"Input shape: {hidden_states.shape}")

        # Forward pass
        output = attention(hidden_states)
        print(f"Output shape: {output.shape}")
        print("✓ Basic attention completed successfully!\n")
    else:
        print("CUDA not available, skipping example\n")


def example_causal_attention():
    """Example with causal masking for autoregressive models."""
    print("=" * 60)
    print("Example 2: Causal Attention (for GPT-like models)")
    print("=" * 60)

    # Configuration for GPT-style model
    batch_size = 2
    seq_len = 512
    embed_dim = 768
    num_heads = 12

    # Create causal attention layer
    attention = FastMultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        kernel_type="triton",
        causal=True,  # Enable causal masking
        dropout=0.1,
    )

    if torch.cuda.is_available():
        attention = attention.cuda()
        hidden_states = torch.randn(batch_size, seq_len, embed_dim).cuda()
        print(f"Input shape: {hidden_states.shape}")

        # Forward pass
        output = attention(hidden_states)
        print(f"Output shape: {output.shape}")
        print("✓ Causal attention completed successfully!\n")
    else:
        print("CUDA not available, skipping example\n")


def example_with_rope():
    """Example with RoPE position embeddings."""
    print("=" * 60)
    print("Example 3: Triton Attention + RoPE")
    print("=" * 60)

    # Configuration
    batch_size = 2
    seq_len = 1024
    embed_dim = 512
    num_heads = 8

    # Create attention with RoPE
    attention = FastMultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        kernel_type="triton",
        position_method="rope",  # Use RoPE position embeddings
        max_seq_len=2048,
        rope_base=10000.0,
        causal=True,
    )

    if torch.cuda.is_available():
        attention = attention.cuda()
        hidden_states = torch.randn(batch_size, seq_len, embed_dim).cuda()
        print(f"Input shape: {hidden_states.shape}")

        # Forward pass
        output = attention(hidden_states)
        print(f"Output shape: {output.shape}")
        print("✓ Attention with RoPE completed successfully!\n")
    else:
        print("CUDA not available, skipping example\n")


def example_comparison():
    """Compare different kernel types."""
    print("=" * 60)
    print("Example 4: Kernel Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping example\n")
        return

    # Configuration
    batch_size = 2
    seq_len = 128
    embed_dim = 256
    num_heads = 8

    hidden_states = torch.randn(batch_size, seq_len, embed_dim).cuda()

    # Test different kernel types
    kernel_types = ["pytorch", "triton"]

    for kernel_type in kernel_types:
        print(f"\nTesting {kernel_type} kernel...")
        try:
            attention = FastMultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kernel_type=kernel_type,
                causal=False,
            ).cuda()

            # Warmup
            _ = attention(hidden_states)

            # Timed run
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                output = attention(hidden_states)
                end.record()

                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)

                print(f"  Output shape: {output.shape}")
                print(f"  Time: {elapsed_time:.3f} ms")
                print(f"  ✓ {kernel_type} kernel works!")
        except Exception as e:
            print(f"  ✗ {kernel_type} kernel failed: {e}")

    print()


def example_with_kv_cache():
    """Example with KV caching for inference."""
    print("=" * 60)
    print("Example 5: Triton Attention with KV Cache")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping example\n")
        return

    # Configuration
    batch_size = 1
    seq_len = 10
    embed_dim = 512
    num_heads = 8

    attention = FastMultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        kernel_type="triton",
        causal=True,
    ).cuda()

    # First forward pass (no cache)
    hidden_states = torch.randn(batch_size, seq_len, embed_dim).cuda()
    output, kv_cache = attention(hidden_states, use_cache=True)
    print(f"Initial sequence length: {seq_len}")
    print(f"Output shape: {output.shape}")
    print(f"KV cache shapes: K={kv_cache[0].shape}, V={kv_cache[1].shape}")

    # Second forward pass (with cache)
    new_token = torch.randn(batch_size, 1, embed_dim).cuda()
    output, kv_cache = attention(new_token, past_key_value=kv_cache, use_cache=True)
    print(f"\nNew token added")
    print(f"Output shape: {output.shape}")
    print(f"Updated KV cache shapes: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    print("✓ KV caching works!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FlashLM Triton Attention Examples")
    print("=" * 60 + "\n")

    example_basic_usage()
    example_causal_attention()
    example_with_rope()
    example_comparison()
    example_with_kv_cache()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
