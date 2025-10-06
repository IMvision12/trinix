from typing import Tuple

import torch
import triton
import triton.language as tl


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
    batch_size, seq_len, num_heads, head_dim = tensor.shape
    output = torch.empty_like(tensor)
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
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
    )
    return output


class TritonRoPEFunction(torch.autograd.Function):
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
