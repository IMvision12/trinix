import torch
import triton
import triton.language as tl


@triton.jit
def alibi_bias_kernel(
    bias_ptr, slopes_ptr,
    batch_size, num_heads, seq_len,
    stride_bias_batch, stride_bias_head, stride_bias_i, stride_bias_j,
    BLOCK_SIZE_J: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    i_idx = tl.program_id(2)
    
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    j_mask = j_offsets < seq_len
    
    slope = tl.load(slopes_ptr + head_idx)
    
    distances = tl.abs(i_idx - j_offsets).to(tl.float32)
    bias_values = -slope * distances
    
    bias_offset = (batch_idx * stride_bias_batch + 
                  head_idx * stride_bias_head + 
                  i_idx * stride_bias_i)
    
    tl.store(bias_ptr + bias_offset + j_offsets, bias_values, mask=j_mask)


class TritonALiBiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slopes, batch_size, num_heads, seq_len):
        bias = torch.empty(batch_size, num_heads, seq_len, seq_len, 
                          device=slopes.device, dtype=slopes.dtype)
        
        BLOCK_SIZE_J = triton.next_power_of_2(seq_len)
        grid = (batch_size, num_heads, seq_len)
        
        alibi_bias_kernel[grid](
            bias, slopes,
            batch_size, num_heads, seq_len,
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
            BLOCK_SIZE_J=BLOCK_SIZE_J,
        )
        
        return bias
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None


class TritonALiBiKernel:
    @staticmethod
    def apply(slopes: torch.Tensor, batch_size: int, num_heads: int, seq_len: int) -> torch.Tensor:
        return TritonALiBiFunction.apply(slopes, batch_size, num_heads, seq_len)
    
    @staticmethod
    def is_available() -> bool:
        try:
            import triton
            return torch.cuda.is_available()
        except ImportError:
            return False