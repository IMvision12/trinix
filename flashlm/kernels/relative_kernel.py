import torch
import triton
import triton.language as tl


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
        BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim)
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
