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
    """SwiGLU (Gated SiLU/Swish) forward kernel.

    Computes SwiGLU activation: x1 * SiLU(x2) = x1 * (x2 * sigmoid(x2)),
    where the input is split into two halves.

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
    """SwiGLU backward kernel.

    Computes gradients for SwiGLU activation with respect to both input halves.

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
    """Autograd function for SwiGLU activation.

    This function wraps the SwiGLU kernel for automatic differentiation.

    Methods:
        forward(ctx, X):
            Computes SwiGLU activation: x1 * SiLU(x2) where input is split in half.

            Parameters:
                ctx: Autograd context for saving tensors needed in backward pass.
                X (torch.Tensor): Input tensor with even last dimension.

            Returns:
                torch.Tensor: Output tensor with last dimension halved.

        backward(ctx, dY):
            Backward pass for SwiGLU activation.

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
    """Triton-accelerated SwiGLU (Gated SiLU/Swish) activation kernel wrapper.

    Provides a high-level interface for applying SwiGLU activation: x1 * SiLU(x2),
    where the input is split into two halves along the last dimension.
    SiLU (Swish) is x * sigmoid(x).

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(X):
            Applies SwiGLU activation to the input tensor.

            Parameters:
                X (torch.Tensor): Input tensor with even last dimension. The tensor is split
                    into two halves along the last dimension: x1 and x2.

            Returns:
                torch.Tensor: Output tensor with last dimension halved, computed as x1 * SiLU(x2).
                    SiLU (also known as Swish) is computed as x2 * sigmoid(x2).
                    SwiGLU is commonly used in transformer feed-forward networks and has been
                    shown to improve performance over standard activations.
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
        return TritonSwiGLUFunction.apply(X)
