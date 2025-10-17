import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def rmsnorm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    rstd_ptr += row_idx

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    X_squared = tl.where(mask, X_row * X_row, 0.0)
    variance = tl.sum(X_squared, axis=0) / n_cols

    rstd = tl.math.rsqrt(variance + eps)

    tl.store(rstd_ptr, rstd)
    output = X_row * rstd * W_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)


@triton.jit
def rmsnorm_backward_kernel_fused(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    rstd_ptr,
    dX_ptr,
    dX_row_stride,
    dW_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    dX_ptr += row_idx * dX_row_stride
    rstd_ptr += row_idx

    dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr).to(tl.float32)

    X_row_f32 = X_row.to(tl.float32)
    dY_row_f32 = dY_row.to(tl.float32)
    W_row_f32 = W_row.to(tl.float32)

    normed = X_row_f32 * rstd

    dW_vals = dY_row * normed
    tl.atomic_add(dW_ptr + col_offsets, dW_vals, mask=mask)
    dY_W = dY_row_f32 * W_row_f32

    dY_W_normed = dY_W * normed
    mean_dY_W_normed = tl.sum(dY_W_normed, axis=0) / n_cols

    dX_row = (dY_W - mean_dY_W_normed * normed) * rstd

    tl.store(dX_ptr + col_offsets, dX_row, mask=mask)


@triton.jit
def rmsnorm_backward_kernel_welford(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    rstd_ptr,
    dX_ptr,
    dX_row_stride,
    dW_ptr,
    Lock_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    dX_ptr += row_idx * dX_row_stride

    dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

    X_row_f32 = X_row.to(tl.float32)
    dY_row_f32 = dY_row.to(tl.float32)
    W_row_f32 = W_row.to(tl.float32)

    normed = X_row_f32 * rstd

    dW_vals = dY_row * normed

    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass

    dW_current = tl.load(dW_ptr + col_offsets, mask=mask, other=0.0)
    tl.store(dW_ptr + col_offsets, dW_current + dW_vals, mask=mask)

    tl.atomic_xchg(Lock_ptr, 0)

    dY_W = dY_row_f32 * W_row_f32
    dY_W_normed = dY_W * normed
    mean_dY_W_normed = tl.sum(dY_W_normed, axis=0) / n_cols

    dX_row = (dY_W - mean_dY_W_normed * normed) * rstd

    tl.store(dX_ptr + col_offsets, dX_row, mask=mask)


class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_cols)

        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=device)

        rmsnorm_forward_kernel[n_rows,](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            rstd,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, rstd)

        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)

        X, W, rstd = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        device = dY.device

        dX = torch.empty_like(X)
        dW = torch.zeros(n_cols, dtype=X.dtype, device=device)

        if n_rows <= 32:
            Lock = torch.zeros(1, dtype=torch.int32, device=device)
            rmsnorm_backward_kernel_welford[n_rows,](
                dY,
                dY.stride(0),
                X,
                X.stride(0),
                W,
                rstd,
                dX,
                dX.stride(0),
                dW,
                Lock,
                n_cols,
                ctx.eps,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )
        else:
            rmsnorm_backward_kernel_fused[n_rows,](
                dY,
                dY.stride(0),
                X,
                X.stride(0),
                W,
                rstd,
                dX,
                dX.stride(0),
                dW,
                n_cols,
                ctx.eps,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )

        return dX.view(*shape), dW, None


class TritonRMSNormKernel:
    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def apply(X: torch.Tensor, W: torch.Tensor, eps: float) -> torch.Tensor:
        return TritonRMSNormFunction.apply(X, W, eps)
