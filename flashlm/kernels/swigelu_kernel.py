import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def swiglu_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    x1 = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    x2 = tl.load(X_ptr + col_offsets + n_cols, mask=mask, other=0.0)
    x1_f32 = x1.to(tl.float32)
    x2_f32 = x2.to(tl.float32)
    sigmoid_x2 = tl.sigmoid(x2_f32)
    silu_x2 = x2_f32 * sigmoid_x2
    output = x1_f32 * silu_x2
    tl.store(Y_ptr + col_offsets, output.to(x1.dtype), mask=mask)


@triton.jit
def swiglu_backward_kernel(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    dX_ptr,
    dX_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    dX_ptr += row_idx * dX_row_stride
    dY = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
    x1 = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    x2 = tl.load(X_ptr + col_offsets + n_cols, mask=mask, other=0.0)
    dY_f32 = dY.to(tl.float32)
    x1_f32 = x1.to(tl.float32)
    x2_f32 = x2.to(tl.float32)
    sigmoid_x2 = tl.sigmoid(x2_f32)
    silu_x2 = x2_f32 * sigmoid_x2
    dX1 = dY_f32 * silu_x2
    dsilu_dx2 = sigmoid_x2 * (1.0 + x2_f32 * (1.0 - sigmoid_x2))
    dX2 = dY_f32 * x1_f32 * dsilu_dx2
    tl.store(dX_ptr + col_offsets, dX1.to(x1.dtype), mask=mask)
    tl.store(dX_ptr + col_offsets + n_cols, dX2.to(x2.dtype), mask=mask)


class TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        shape = X.shape
        dim = shape[-1]
        assert dim % 2 == 0, "Last dimension must be even for SwiGLU"
        hidden_dim = dim // 2
        X = X.view(-1, dim)
        n_rows, n_cols_full = X.shape
        n_cols = hidden_dim
        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_cols)
        device = X.device
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        swiglu_forward_kernel[n_rows,](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X)
        output_shape = shape[:-1] + (hidden_dim,)
        return Y.view(*output_shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        hidden_dim = shape[-1]
        dY = dY.view(-1, hidden_dim)
        (X,) = ctx.saved_tensors
        n_rows, n_cols_full = X.shape
        n_cols = hidden_dim
        device = dY.device
        dX = torch.empty_like(X)
        swiglu_backward_kernel[n_rows,](
            dY,
            dY.stride(0),
            X,
            X.stride(0),
            dX,
            dX.stride(0),
            n_cols,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        input_shape = shape[:-1] + (n_cols_full,)
        return dX.view(*input_shape)


class TritonSwiGLUKernel:
    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def apply(X: torch.Tensor) -> torch.Tensor:
        return TritonSwiGLUFunction.apply(X)
