import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def reglu_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ReGLU (Gated ReLU) forward kernel.

    Computes ReGLU activation: x1 * ReLU(x2), where the input is split into two halves.

    Args:
        Y_ptr: Pointer to output tensor.
        Y_row_stride: Stride for row dimension in output tensor.
        X_ptr: Pointer to input tensor (last dimension must be even).
        X_row_stride: Stride for row dimension in input tensor.
        n_cols: Number of columns (half of input dimension).
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride

    x1 = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X_ptr + col_offsets + n_cols, mask=mask, other=0.0).to(tl.float32)

    relu_x2 = tl.maximum(x2, 0.0)
    output = x1 * relu_x2

    tl.store(Y_ptr + col_offsets, output, mask=mask)


@triton.jit
def reglu_backward_kernel(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    dX_ptr,
    dX_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ReGLU backward kernel.

    Computes gradients for ReGLU activation with respect to both input halves.
    Gradients: dL/dx1 = dL/dy * ReLU(x2), dL/dx2 = dL/dy * x1 * (x2 > 0)

    Args:
        dY_ptr: Pointer to output gradient tensor.
        dY_row_stride: Stride for row dimension in output gradient tensor.
        X_ptr: Pointer to input tensor from forward pass.
        X_row_stride: Stride for row dimension in input tensor.
        dX_ptr: Pointer to input gradient tensor.
        dX_row_stride: Stride for row dimension in input gradient tensor.
        n_cols: Number of columns (half of input dimension).
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    dX_ptr += row_idx * dX_row_stride

    dy = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X_ptr + col_offsets + n_cols, mask=mask, other=0.0).to(tl.float32)

    relu_x2 = tl.maximum(x2, 0.0)
    dx1 = dy * relu_x2

    dx2 = tl.where(x2 > 0.0, dy * x1, 0.0)

    tl.store(dX_ptr + col_offsets, dx1, mask=mask)
    tl.store(dX_ptr + col_offsets + n_cols, dx2, mask=mask)


class TritonReGLUFunction(torch.autograd.Function):
    """Autograd function for ReGLU activation.

    This function wraps the ReGLU kernel for automatic differentiation.

    Methods:
        forward(ctx, X):
            Computes ReGLU activation: x1 * ReLU(x2) where input is split in half.

            Parameters:
                ctx: Autograd context for saving tensors needed in backward pass.
                X (torch.Tensor): Input tensor with even last dimension.

            Returns:
                torch.Tensor: Output tensor with last dimension halved.

        backward(ctx, dY):
            Backward pass for ReGLU activation.

            Parameters:
                ctx: Autograd context containing saved input tensor.
                dY: Gradient of loss with respect to the output.

            Returns:
                torch.Tensor: Gradient of loss with respect to the input.
    """

    @staticmethod
    def forward(ctx, X):
        shape = X.shape
        dim = shape[-1]
        assert dim % 2 == 0, "Last dimension must be even for ReGLU"
        hidden_dim = dim // 2

        X = X.view(-1, dim)
        n_rows, n_cols_full = X.shape
        n_cols = hidden_dim

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)

        reglu_forward_kernel[n_rows,](
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

        reglu_backward_kernel[n_rows,](
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


class TritonReGLUKernel:
    """Triton-accelerated ReGLU (Gated ReLU) activation kernel wrapper.

    Provides a high-level interface for applying ReGLU activation: x1 * ReLU(x2),
    where the input is split into two halves along the last dimension.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(X):
            Applies ReGLU activation to the input tensor.

            Parameters:
                X (torch.Tensor): Input tensor with even last dimension. The tensor is split
                    into two halves along the last dimension: x1 and x2.

            Returns:
                torch.Tensor: Output tensor with last dimension halved, computed as x1 * ReLU(x2).
                    ReLU is applied to the second half before element-wise multiplication with the first half.
    """

    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def apply(X: torch.Tensor) -> torch.Tensor:
        return TritonReGLUFunction.apply(X)
