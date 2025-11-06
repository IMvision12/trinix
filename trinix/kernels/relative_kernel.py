import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def relative_pos_bias_kernel(
    embeddings_ptr,
    positions_ptr,
    output_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    stride_emb_vocab,
    stride_emb_dim,
    stride_pos_i,
    stride_pos_j,
    stride_out_batch,
    stride_out_head,
    stride_out_i,
    stride_out_j,
    stride_out_dim,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    """Relative position bias kernel.

    Computes position-dependent bias embeddings for attention mechanisms using learned
    relative position embeddings indexed by position differences.

    Args:
        embeddings_ptr: Pointer to position embedding table.
        positions_ptr: Pointer to position indices tensor.
        output_ptr: Pointer to output bias tensor.
        batch_size: Number of sequences in the batch.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        head_dim: Dimension of each attention head.
        stride_emb_vocab: Stride for vocabulary dimension in embeddings.
        stride_emb_dim: Stride for dimension in embeddings.
        stride_pos_i: Stride for query position dimension in positions.
        stride_pos_j: Stride for key position dimension in positions.
        stride_out_batch: Stride for batch dimension in output.
        stride_out_head: Stride for head dimension in output.
        stride_out_i: Stride for query position dimension in output.
        stride_out_j: Stride for key position dimension in output.
        stride_out_dim: Stride for embedding dimension in output.
        BLOCK_SIZE_DIM: Triton block size for embedding dimension.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    i_idx = tl.program_id(2)
    for j_idx in range(seq_len):
        pos_idx = tl.load(positions_ptr + i_idx * stride_pos_i + j_idx * stride_pos_j)
        dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
        dim_mask = dim_offsets < head_dim
        emb_offset = pos_idx * stride_emb_vocab
        emb_values = tl.load(
            embeddings_ptr + emb_offset + dim_offsets, mask=dim_mask, other=0.0
        )
        out_offset = (
            batch_idx * stride_out_batch
            + head_idx * stride_out_head
            + i_idx * stride_out_i
            + j_idx * stride_out_j
        )
        tl.store(output_ptr + out_offset + dim_offsets, emb_values, mask=dim_mask)


class TritonRelativeFunction(torch.autograd.Function):
    """Autograd function for relative position bias computation.

    This function wraps the relative position bias kernel for automatic differentiation.

    Methods:
        forward(ctx, embeddings, positions, batch_size, num_heads):
            Computes position-dependent bias embeddings using learned relative position embeddings.

            Parameters:
                ctx: Autograd context for saving tensors needed in backward pass.
                embeddings (torch.Tensor): Position embedding table of shape (vocab_size, head_dim).
                positions (torch.Tensor): Position indices tensor of shape (seq_len, seq_len).
                batch_size (int): Number of sequences in the batch.
                num_heads (int): Number of attention heads.

            Returns:
                torch.Tensor: Bias tensor of shape (batch_size, num_heads, seq_len, seq_len, head_dim).

        backward(ctx, grad_output):
            Backward pass for relative position bias.

            Parameters:
                ctx: Autograd context containing saved position indices.
                grad_output: Gradient of loss with respect to the output.

            Returns:
                tuple: (grad_embeddings, None, None, None) - Gradient for embeddings only,
                    None for positions, batch_size, and num_heads.
    """

    @staticmethod
    def forward(ctx, embeddings, positions, batch_size, num_heads):
        seq_len, _ = positions.shape
        vocab_size, head_dim = embeddings.shape
        output = torch.empty(
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            head_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        BLOCK_SIZE_DIM, num_warps = calculate_triton_kernel_configuration(head_dim)
        grid = (batch_size, num_heads, seq_len)
        relative_pos_bias_kernel[grid](
            embeddings,
            positions,
            output,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            embeddings.stride(0),
            embeddings.stride(1),
            positions.stride(0),
            positions.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            output.stride(4),
            BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
            num_warps=num_warps,
        )
        ctx.save_for_backward(positions)
        ctx.shape_info = (batch_size, num_heads, seq_len, head_dim, vocab_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (positions,) = ctx.saved_tensors
        batch_size, num_heads, seq_len, head_dim, vocab_size = ctx.shape_info
        grad_embeddings = torch.zeros(
            vocab_size, head_dim, device=grad_output.device, dtype=grad_output.dtype
        )
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len):
                    for j in range(seq_len):
                        pos_idx = positions[i, j].item()
                        grad_embeddings[pos_idx] += grad_output[b, h, i, j]
        return (grad_embeddings, None, None, None)


class TritonRelativeKernel:
    """Triton-accelerated relative position bias kernel wrapper.

    Provides a high-level interface for computing position-dependent attention biases
    using learned relative position embeddings.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(embeddings, positions, batch_size, num_heads):
            Computes relative position bias embeddings for attention mechanisms.

            Parameters:
                embeddings (torch.Tensor): Position embedding table of shape (vocab_size, head_dim)
                    containing learned embeddings for relative position differences.
                positions (torch.Tensor): Position indices tensor of shape (seq_len, seq_len)
                    where positions[i, j] contains the index into the embedding table for
                    the relative position between query position i and key position j.
                batch_size (int): Number of sequences in the batch.
                num_heads (int): Number of attention heads.

            Returns:
                torch.Tensor: Bias tensor of shape (batch_size, num_heads, seq_len, seq_len, head_dim)
                    containing position-dependent bias embeddings that can be added to attention scores.
    """

    @staticmethod
    def apply(
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        return TritonRelativeFunction.apply(
            embeddings, positions, batch_size, num_heads
        )

    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False
