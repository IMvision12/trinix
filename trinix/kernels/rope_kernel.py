from typing import Tuple

import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def _rope_rotation_kernel(
    tensor_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    stride_tensor_batch,
    stride_tensor_seq,
    stride_tensor_head,
    stride_tensor_dim,
    stride_cos_seq,
    stride_cos_dim,
    stride_sin_seq,
    stride_sin_dim,
    is_backward: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RoPE (Rotary Position Embedding) rotation kernel.

    Applies rotary position embeddings by rotating pairs of features using
    position-dependent rotation matrices. For forward pass, applies rotation;
    for backward pass, applies inverse rotation.

    Args:
        tensor_ptr: Pointer to input tensor (query or key).
        cos_ptr: Pointer to cosine values for rotation.
        sin_ptr: Pointer to sine values for rotation.
        output_ptr: Pointer to output tensor.
        batch_size: Number of sequences in the batch.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        stride_tensor_batch: Stride for batch dimension in tensor.
        stride_tensor_seq: Stride for sequence dimension in tensor.
        stride_tensor_head: Stride for head dimension in tensor.
        stride_tensor_dim: Stride for feature dimension in tensor.
        stride_cos_seq: Stride for sequence dimension in cosine tensor.
        stride_cos_dim: Stride for feature dimension in cosine tensor.
        stride_sin_seq: Stride for sequence dimension in sine tensor.
        stride_sin_dim: Stride for feature dimension in sine tensor.
        is_backward: Whether this is backward pass (applies inverse rotation).
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    half_dim = head_dim // 2
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    mask = dim_offsets < half_dim
    
    tensor_base = (
        batch_idx * stride_tensor_batch
        + seq_idx * stride_tensor_seq
        + head_idx * stride_tensor_head
    )
    cos_base = seq_idx * stride_cos_seq
    sin_base = seq_idx * stride_sin_seq
    
    x1 = tl.load(tensor_ptr + tensor_base + dim_offsets, mask=mask, other=0.0)
    x2 = tl.load(
        tensor_ptr + tensor_base + dim_offsets + half_dim, mask=mask, other=0.0
    )
    
    cos_vals = tl.load(cos_ptr + cos_base + dim_offsets, mask=mask, other=0.0)
    sin_vals = tl.load(sin_ptr + sin_base + dim_offsets, mask=mask, other=0.0)
    
    if is_backward:
        sin_vals = -sin_vals
    
    rotated_x1 = x1 * cos_vals - x2 * sin_vals
    rotated_x2 = x1 * sin_vals + x2 * cos_vals
    
    tl.store(output_ptr + tensor_base + dim_offsets, rotated_x1, mask=mask)
    tl.store(output_ptr + tensor_base + dim_offsets + half_dim, rotated_x2, mask=mask)


def _apply_rope_kernel(tensor, cos, sin, is_backward=False):
    """Apply RoPE rotation kernel to a tensor.

    Args:
        tensor: Input tensor to rotate.
        cos: Cosine values for rotation.
        sin: Sine values for rotation.
        is_backward: Whether to apply inverse rotation (for backward pass).

    Returns:
        Rotated output tensor.
    """
    batch_size, seq_len, num_heads, head_dim = tensor.shape    
    output = torch.empty_like(tensor)    
    BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(head_dim // 2)
    grid = (batch_size, seq_len, num_heads)
    
    _rope_rotation_kernel[grid](
        tensor,
        cos,
        sin,
        output,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(2),
        tensor.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        is_backward=is_backward,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output


class TritonRoPEFunction(torch.autograd.Function):
    """Autograd function for RoPE (Rotary Position Embedding).

    This function wraps the RoPE kernel for automatic differentiation.

    Methods:
        forward(ctx, q, k, cos, sin):
            Applies rotary position embeddings to query and key tensors.

            Parameters:
                ctx: Autograd context for saving tensors needed in backward pass.
                q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
                k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
                cos (torch.Tensor): Cosine values for rotation of shape (seq_len, head_dim // 2).
                sin (torch.Tensor): Sine values for rotation of shape (seq_len, head_dim // 2).

            Returns:
                tuple: (rotated_q, rotated_k) - Both tensors with same shapes as inputs.

        backward(ctx, grad_q, grad_k):
            Backward pass for RoPE using inverse rotation.

            Parameters:
                ctx: Autograd context containing saved cos and sin tensors.
                grad_q: Gradient of loss with respect to query output.
                grad_k: Gradient of loss with respect to key output.

            Returns:
                tuple: (grad_q_input, grad_k_input, None, None) - Gradients for q and k inputs,
                    None for cos and sin (no gradients needed for rotation matrices).
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out = _apply_rope_kernel(q, cos, sin, is_backward=False)
        k_out = _apply_rope_kernel(k, cos, sin, is_backward=False)
        ctx.save_for_backward(cos, sin)
        return (q_out, k_out)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        grad_q_out = _apply_rope_kernel(grad_q, cos, sin, is_backward=True)
        grad_k_out = _apply_rope_kernel(grad_k, cos, sin, is_backward=True)
        return (grad_q_out, grad_k_out, None, None)


class TritonRoPEKernel:
    """Triton-accelerated RoPE (Rotary Position Embedding) kernel wrapper.

    Provides a high-level interface for applying rotary position embeddings to
    query and key tensors. RoPE encodes position information by rotating pairs
    of features using position-dependent rotation matrices.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(q, k, cos, sin):
            Applies rotary position embeddings to query and key tensors.

            Parameters:
                q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
                k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
                cos (torch.Tensor): Cosine values for rotation of shape (seq_len, head_dim // 2).
                    Precomputed cosine values for each position and dimension pair.
                sin (torch.Tensor): Sine values for rotation of shape (seq_len, head_dim // 2).
                    Precomputed sine values for each position and dimension pair.

            Returns:
                tuple: (rotated_q, rotated_k) - Both tensors with same shapes as inputs.
                    RoPE rotates pairs of features using 2D rotation matrices, encoding
                    position information without adding extra parameters. The rotation is
                    applied independently to each pair of consecutive dimensions.
    """

    @staticmethod
    def apply(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return TritonRoPEFunction.apply(q, k, cos, sin)

    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False
