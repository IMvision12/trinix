from typing import Optional, Tuple

import torch

from ...kernels.attention_kernel import TritonAttentionKernel


def triton_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional interface for Triton-accelerated Flash Attention.

    Computes scaled dot-product attention using memory-efficient Triton kernels with support
    for various attention patterns including causal masking, sliding window, custom masks,
    ALiBi biases, and dropout. This is a functional API similar to PyTorch's scaled_dot_product_attention.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, num_heads, head_dim).
            Must be a CUDA tensor.
        k (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, num_heads, head_dim).
            Must be a CUDA tensor with same batch_size, num_heads, and head_dim as q.
        v (torch.Tensor): Value tensor of shape (batch_size, seq_len_k, num_heads, head_dim).
            Must be a CUDA tensor with same shape as k.
        attn_mask (torch.Tensor, optional): Attention mask to add to attention scores.
            Can be 2D (seq_len_q, seq_len_k), 3D (batch, seq_len_q, seq_len_k), or
            4D (batch, num_heads, seq_len_q, seq_len_k). Values are added to attention scores
            before softmax. Use -inf to mask out positions. Defaults to None.
        dropout_p (float, optional): Dropout probability for attention weights.
            Only applied during training. Defaults to 0.0.
        softmax_scale (float, optional): Scaling factor for attention scores before softmax.
            If None, uses 1/sqrt(head_dim). Defaults to None.
        causal (bool, optional): Whether to apply causal masking (for autoregressive models).
            When True, each position can only attend to previous positions. Defaults to False.
        window_size (Tuple[int, int], optional): Sliding window attention size as
            (window_left, window_right). Each position can only attend to positions within
            the window. Useful for long sequences. Defaults to None.
        alibi_slopes (torch.Tensor, optional): Per-head slope values for ALiBi position biases.
            Shape should be (num_heads,) or (1, num_heads). Defaults to None.

    Returns:
        torch.Tensor: Attention output of shape (batch_size, seq_len_q, num_heads, head_dim).

    Raises:
        ValueError: If input tensors are not CUDA tensors or have incorrect shapes.
        RuntimeError: If Triton attention kernel is not available.

    Shape:
        - q: (batch_size, seq_len_q, num_heads, head_dim)
        - k: (batch_size, seq_len_k, num_heads, head_dim)
        - v: (batch_size, seq_len_k, num_heads, head_dim)
        - attn_mask: (seq_len_q, seq_len_k) or (batch, seq_len_q, seq_len_k) or
                     (batch, num_heads, seq_len_q, seq_len_k)
        - alibi_slopes: (num_heads,) or (1, num_heads)
        - output: (batch_size, seq_len_q, num_heads, head_dim)

    Examples:
        >>> # Standard attention
        >>> q = k = v = torch.randn(4, 128, 8, 64, device='cuda')
        >>> output = triton_attn_func(q, k, v)

        >>> # Causal attention with dropout (for GPT-style models)
        >>> output = triton_attn_func(q, k, v, causal=True, dropout_p=0.1)

        >>> # Attention with custom mask
        >>> mask = torch.zeros(128, 128, device='cuda')
        >>> mask[:, :64] = float('-inf')  # Mask out first 64 positions
        >>> output = triton_attn_func(q, k, v, attn_mask=mask)

        >>> # Sliding window attention (for long sequences)
        >>> output = triton_attn_func(q, k, v, window_size=(256, 256))

        >>> # Attention with ALiBi position biases
        >>> slopes = torch.randn(8, device='cuda')  # One slope per head
        >>> output = triton_attn_func(q, k, v, alibi_slopes=slopes)

        >>> # Combining multiple features
        >>> output = triton_attn_func(
        ...     q, k, v,
        ...     causal=True,
        ...     dropout_p=0.1,
        ...     window_size=(512, 512),
        ...     alibi_slopes=slopes
        ... )

    Notes:
        - All input tensors must be CUDA tensors
        - Requires Triton to be installed and CUDA to be available
        - Uses Flash Attention algorithm for memory efficiency
        - Supports combining multiple attention patterns (causal + window + ALiBi)
        - More memory efficient than standard PyTorch attention for long sequences
        - Dropout is only applied during training (when dropout_p > 0)
    """
    if not q.is_cuda:
        raise ValueError("triton_attn_func only supports CUDA tensors")

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"Expected 4D tensors (batch, seqlen, nheads, headdim), "
            f"got q: {q.shape}, k: {k.shape}, v: {v.shape}"
        )

    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, _, _ = k.shape

    if k.shape != (batch, seqlen_k, nheads, headdim):
        raise ValueError(f"k shape {k.shape} doesn't match expected shape")

    if v.shape != (batch, seqlen_k, nheads, headdim):
        raise ValueError(f"v shape {v.shape} doesn't match expected shape")

    if not TritonAttentionKernel.is_available():
        raise RuntimeError(
            "Triton attention kernel not available. "
            "Make sure Triton is installed and CUDA is available."
        )

    if softmax_scale is None:
        softmax_scale = headdim**-0.5

    out = TritonAttentionKernel.apply(
        q,
        k,
        v,
        attn_mask=attn_mask,
        causal=causal,
        scale=softmax_scale,
        dropout_p=dropout_p,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )

    return out
