import torch
import triton
import triton.language as tl
from typing import Tuple

from .utils import calculate_triton_kernel_configuration


@triton.jit
def quantize_rowwise_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    M, K,
    BLOCK_SIZE_K: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    abs_max = 0.0
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        x = tl.load(x_ptr + row_idx * K + offs_k, mask=mask_k, other=0.0)
        block_max = tl.max(tl.abs(x), axis=0)
        abs_max = tl.maximum(abs_max, block_max)
    
    scale = abs_max / 127.0
    scale = tl.maximum(scale, 1e-8)
    tl.store(scale_ptr + row_idx, scale)
    
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        x = tl.load(x_ptr + row_idx * K + offs_k, mask=mask_k, other=0.0)
        q = x / scale
        q = tl.extra.cuda.libdevice.round(q)
        q = tl.maximum(tl.minimum(q, 127.0), -128.0)
        q = q.to(tl.int8)
        tl.store(out_ptr + row_idx * K + offs_k, q, mask=mask_k)


@triton.jit
def quantize_colwise_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    N, K,
    BLOCK_SIZE_N: tl.constexpr,
):
    col_idx = tl.program_id(0)
    
    abs_max = 0.0
    for n_block in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        x = tl.load(x_ptr + offs_n * K + col_idx, mask=mask_n, other=0.0)
        block_max = tl.max(tl.abs(x), axis=0)
        abs_max = tl.maximum(abs_max, block_max)
    
    scale = abs_max / 127.0
    scale = tl.maximum(scale, 1e-8)
    tl.store(scale_ptr + col_idx, scale)
    
    for n_block in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        x = tl.load(x_ptr + offs_n * K + col_idx, mask=mask_n, other=0.0)
        q = x / scale
        q = tl.extra.cuda.libdevice.round(q)
        q = tl.maximum(tl.minimum(q, 127.0), -128.0)
        q = q.to(tl.int8)
        tl.store(out_ptr + offs_n * K + col_idx, q, mask=mask_n)


@triton.jit
def rescale_kernel(
    c_int32_ptr,
    out_ptr,
    scale_a_ptr,
    scale_b_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    c_int32 = tl.load(
        c_int32_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=mask,
        other=0
    )
    
    scale_a = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=1.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=1.0)
    
    c_fp32 = c_int32.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
    c_fp16 = c_fp32.to(out_ptr.dtype.element_ty)
    
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], c_fp16, mask=mask)


def quantize_activation_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    x_int8 = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE_K, num_warps = calculate_triton_kernel_configuration(K)
    grid = (M,)
    
    quantize_rowwise_kernel[grid](
        x, x_int8, scales,
        M, K,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps,
    )
    
    return x_int8, scales


def quantize_weight_colwise(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    N, K = weight.shape
    weight_int8 = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(K, device=weight.device, dtype=torch.float32)
    
    BLOCK_SIZE_N, num_warps = calculate_triton_kernel_configuration(N)
    grid = (K,)
    
    quantize_colwise_kernel[grid](
        weight, weight_int8, scales,
        N, K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )
    
    return weight_int8, scales


def rescale_output(
    c_int32: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: torch.dtype
) -> torch.Tensor:
    M, N = c_int32.shape
    output = torch.empty((M, N), device=c_int32.device, dtype=output_dtype)
    
    BLOCK_SIZE_M, num_warps_m = calculate_triton_kernel_configuration(M)
    BLOCK_SIZE_N, num_warps_n = calculate_triton_kernel_configuration(N)
    
    num_warps = max(num_warps_m, num_warps_n)
    
    BLOCK_SIZE_M = min(BLOCK_SIZE_M, 128)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 128)
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    rescale_kernel[grid](
        c_int32, output,
        scale_a, scale_b,
        M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )
    
    return output
