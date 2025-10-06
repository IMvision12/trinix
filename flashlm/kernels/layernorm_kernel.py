from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from .utils import calculate_triton_kernel_configuration

@triton.jit
def layernorm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride, W_ptr, b_ptr, rstd_ptr, mean_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    rstd_ptr += row_idx
    mean_ptr += row_idx
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_X = tl.sum(X_row, axis=0) / n_cols
    X_centered = tl.where(mask, X_row - mean_X, 0.0)
    row_var = tl.sum(X_centered * X_centered, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(rstd_ptr, inv_var)
    tl.store(mean_ptr, mean_X)
    output = X_centered * inv_var * W_row + b_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)

@triton.jit
def layernorm_backward_kernel_fused(dY_ptr, dY_row_stride, X_ptr, X_row_stride, W_ptr, rstd_ptr, mean_ptr, dX_ptr, dX_row_stride, dW_ptr, db_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    dX_ptr += row_idx * dX_row_stride
    rstd_ptr += row_idx
    mean_ptr += row_idx
    dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    inv_var = tl.load(rstd_ptr).to(tl.float32)
    mean = tl.load(mean_ptr).to(tl.float32)
    X_row_f32 = X_row.to(tl.float32)
    normed_f32 = (X_row_f32 - mean) * inv_var
    normed = normed_f32.to(dY_row.dtype)
    dW_vals = dY_row * normed
    db_vals = dY_row
    tl.atomic_add(dW_ptr + col_offsets, dW_vals, mask=mask)
    tl.atomic_add(db_ptr + col_offsets, db_vals, mask=mask)
    dY_row_f32 = dY_row.to(tl.float32)
    W_row_f32 = W_row.to(tl.float32)
    dY_W = dY_row_f32 * W_row_f32
    sum_dY_W = tl.sum(dY_W, axis=0)
    sum_dY_W_normed = tl.sum(dY_W * normed_f32, axis=0)
    dX_row = dY_W - sum_dY_W / n_cols - normed_f32 * sum_dY_W_normed / n_cols
    dX_row = dX_row * inv_var
    tl.store(dX_ptr + col_offsets, dX_row, mask=mask)

@triton.jit
def layernorm_backward_kernel_welford(dY_ptr, dY_row_stride, X_ptr, X_row_stride, W_ptr, rstd_ptr, mean_ptr, dX_ptr, dX_row_stride, dW_ptr, db_ptr, Lock_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    dX_ptr += row_idx * dX_row_stride
    dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    inv_var = tl.load(rstd_ptr + row_idx).to(tl.float32)
    mean = tl.load(mean_ptr + row_idx).to(tl.float32)
    X_row_f32 = X_row.to(tl.float32)
    normed_f32 = (X_row_f32 - mean) * inv_var
    normed = normed_f32.to(dY_row.dtype)
    dW_vals = dY_row * normed
    db_vals = dY_row
    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass
    dW_current = tl.load(dW_ptr + col_offsets, mask=mask, other=0.0)
    db_current = tl.load(db_ptr + col_offsets, mask=mask, other=0.0)
    tl.store(dW_ptr + col_offsets, dW_current + dW_vals, mask=mask)
    tl.store(db_ptr + col_offsets, db_current + db_vals, mask=mask)
    tl.atomic_xchg(Lock_ptr, 0)
    dY_row_f32 = dY_row.to(tl.float32)
    W_row_f32 = W_row.to(tl.float32)
    dY_W = dY_row_f32 * W_row_f32
    sum_dY_W = tl.sum(dY_W, axis=0)
    sum_dY_W_normed = tl.sum(dY_W * normed_f32, axis=0)
    dX_row = dY_W - sum_dY_W / n_cols - normed_f32 * sum_dY_W_normed / n_cols
    dX_row = dX_row * inv_var
    tl.store(dX_ptr + col_offsets, dX_row, mask=mask)

class TritonLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_cols)
        device = X.device
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=device)
        mean = torch.empty(n_rows, dtype=torch.float32, device=device)
        layernorm_forward_kernel[n_rows,](Y, Y.stride(0), X, X.stride(0), W, b, rstd, mean, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, rstd, mean)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, rstd, mean = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        device = dY.device
        dX = torch.empty_like(X)
        dW = torch.zeros(n_cols, dtype=X.dtype, device=device)
        db = torch.zeros(n_cols, dtype=X.dtype, device=device)
        if n_rows <= 32:
            Lock = torch.zeros(1, dtype=torch.int32, device=device)
            layernorm_backward_kernel_welford[n_rows,](dY, dY.stride(0), X, X.stride(0), W, rstd, mean, dX, dX.stride(0), dW, db, Lock, n_cols, ctx.eps, BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
        else:
            layernorm_backward_kernel_fused[n_rows,](dY, dY.stride(0), X, X.stride(0), W, rstd, mean, dX, dX.stride(0), dW, db, n_cols, ctx.eps, BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
        return (dX.view(*shape), dW, db, None)

class OptimizedLayerNormKernel:

    @staticmethod
    def apply(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float=1e-05) -> torch.Tensor:
        return OptimizedLayerNormFunction.apply(x, weight, bias, eps)

    @staticmethod
    def is_available() -> bool:
        try:
            import triton
            return torch.cuda.is_available()
        except ImportError:
            return False