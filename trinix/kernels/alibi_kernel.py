import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def alibi_bias_kernel(
    bias_ptr,
    slopes_ptr,
    batch_size,
    num_heads,
    seq_len,
    stride_bias_batch,
    stride_bias_head,
    stride_bias_i,
    stride_bias_j,
    BLOCK_SIZE_J: tl.constexpr,
):
    """ALiBi (Attention with Linear Biases) bias computation kernel.

    Computes position-dependent bias values for attention mechanisms using linear distance-based penalties.
    ALiBi adds a bias to attention scores based on the distance between query and key positions.

    Args:
        bias_ptr: Pointer to output bias tensor.
        slopes_ptr: Pointer to per-head slope values.
        batch_size: Number of sequences in the batch.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        stride_bias_batch: Stride for batch dimension in bias tensor.
        stride_bias_head: Stride for head dimension in bias tensor.
        stride_bias_i: Stride for query position dimension in bias tensor.
        stride_bias_j: Stride for key position dimension in bias tensor.
        BLOCK_SIZE_J: Triton block size for key position dimension.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    i_idx = tl.program_id(2)

    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    j_mask = j_offsets < seq_len

    slope = tl.load(slopes_ptr + head_idx)

    distances = tl.abs(i_idx - j_offsets).to(tl.float32)
    bias_values = -slope * distances

    bias_offset = (
        batch_idx * stride_bias_batch
        + head_idx * stride_bias_head
        + i_idx * stride_bias_i
    )

    tl.store(bias_ptr + bias_offset + j_offsets, bias_values, mask=j_mask)


class TritonALiBiFunction(torch.autograd.Function):
    """Autograd function for ALiBi bias computation.

    This function wraps the ALiBi kernel for automatic differentiation.

    Methods:
        forward(ctx, slopes, batch_size, num_heads, seq_len):
            Computes ALiBi position bias tensor using the Triton kernel.

            Parameters:
                ctx: Autograd context for saving tensors needed in backward pass.
                slopes (torch.Tensor): Per-head slope values of shape (num_heads,).
                batch_size (int): Number of sequences in the batch.
                num_heads (int): Number of attention heads.
                seq_len (int): Sequence length.

            Returns:
                torch.Tensor: Bias tensor of shape (batch_size, num_heads, seq_len, seq_len)
                    where bias[b, h, i, j] = -slopes[h] * |i - j|.

        backward(ctx, grad_output):
            Backward pass for ALiBi bias computation.

            Parameters:
                ctx: Autograd context (unused as no tensors were saved).
                grad_output: Gradient of loss with respect to the output bias tensor.

            Returns:
                tuple: (None, None, None, None) - No gradients needed for any inputs
                    since ALiBi biases are deterministic position-based values that
                    don't depend on learnable parameters.
    """

    @staticmethod
    def forward(ctx, slopes, batch_size, num_heads, seq_len):
        bias = torch.empty(
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            device=slopes.device,
            dtype=slopes.dtype,
        )

        BLOCK_SIZE_J, num_warps = calculate_triton_kernel_configuration(seq_len)
        grid = (batch_size, num_heads, seq_len)
        alibi_bias_kernel[grid](
            bias,
            slopes,
            batch_size,
            num_heads,
            seq_len,
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
            BLOCK_SIZE_J=BLOCK_SIZE_J,
            num_warps=num_warps,
        )

        return bias

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None, None)


class TritonALiBiKernel:
    """Triton-accelerated ALiBi (Attention with Linear Biases) kernel wrapper.

    Provides a high-level interface for computing position-dependent attention biases
    using the ALiBi method, which adds linear distance-based penalties to attention scores.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(slopes, batch_size, num_heads, seq_len):
            Computes ALiBi bias tensor for attention mechanisms.

            Parameters:
                slopes (torch.Tensor): Per-head slope values tensor of shape (num_heads,).
                batch_size (int): Number of sequences in the batch.
                num_heads (int): Number of attention heads.
                seq_len (int): Sequence length.

            Returns:
                torch.Tensor: Bias tensor of shape (batch_size, num_heads, seq_len, seq_len)
                    containing position-dependent bias values based on distance between positions.
    """

    @staticmethod
    def apply(
        slopes: torch.Tensor, batch_size: int, num_heads: int, seq_len: int
    ) -> torch.Tensor:
        return TritonALiBiFunction.apply(slopes, batch_size, num_heads, seq_len)

    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False
