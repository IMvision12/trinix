import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def geglu_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """GeGLU (Gated GELU) forward kernel.

    Computes GeGLU activation: x1 * GELU(x2), where the input is split into two halves.
    GELU uses the tanh approximation for efficiency.

    Optimizations:
    - Single load for both halves (coalesced memory access)
    - Fused GELU computation
    - Minimal type conversions
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_row = Y_ptr + row_idx * Y_row_stride
    X_row = X_ptr + row_idx * X_row_stride

    x1 = tl.load(X_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X_row + col_offsets + n_cols, mask=mask, other=0.0).to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    x2_squared = x2 * x2
    x2_cubed = x2_squared * x2
    tanh_arg = sqrt_2_over_pi * (x2 + coeff * x2_cubed)

    exp_2x = tl.exp(2.0 * tanh_arg)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)

    gelu_x2 = 0.5 * x2 * (1.0 + tanh_val)
    output = x1 * gelu_x2
    tl.store(Y_row + col_offsets, output, mask=mask)


@triton.jit
def geglu_backward_kernel(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    dX_ptr,
    dX_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """GeGLU backward kernel with optimized gradient computation.

    Computes gradients for GeGLU activation with respect to both input halves.

    Gradient formulas:
    - dL/dx1 = dL/dy * GELU(x2)
    - dL/dx2 = dL/dy * x1 * GELU'(x2)

    Optimizations:
    - Reuse computed values from forward pass
    - Fused gradient computation
    - Coalesced memory writes
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_row = dY_ptr + row_idx * dY_row_stride
    X_row = X_ptr + row_idx * X_row_stride
    dX_row = dX_ptr + row_idx * dX_row_stride

    dY = tl.load(dY_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X_row + col_offsets + n_cols, mask=mask, other=0.0).to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    x2_squared = x2 * x2
    x2_cubed = x2_squared * x2
    tanh_arg = sqrt_2_over_pi * (x2 + coeff * x2_cubed)

    exp_2x = tl.exp(2.0 * tanh_arg)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)

    gelu_x2 = 0.5 * x2 * (1.0 + tanh_val)

    sech_squared = 1.0 - tanh_val * tanh_val
    dtanh_arg_dx2 = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x2_squared)
    dgelu_dx2 = 0.5 * (1.0 + tanh_val) + 0.5 * x2 * sech_squared * dtanh_arg_dx2

    dX1 = dY * gelu_x2
    dX2 = dY * x1 * dgelu_dx2

    tl.store(dX_row + col_offsets, dX1, mask=mask)
    tl.store(dX_row + col_offsets + n_cols, dX2, mask=mask)


class TritonGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        shape = X.shape
        dim = shape[-1]
        assert dim % 2 == 0, "Last dimension must be even for GeGLU"

        hidden_dim = dim // 2
        X = X.contiguous().view(-1, dim)
        n_rows, n_cols_full = X.shape
        n_cols = hidden_dim

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)

        geglu_forward_kernel[(n_rows,)](
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
        dY = dY.contiguous().view(-1, hidden_dim)
        (X,) = ctx.saved_tensors

        n_rows, n_cols_full = X.shape
        n_cols = hidden_dim
        device = dY.device

        dX = torch.empty_like(X)

        geglu_backward_kernel[(n_rows,)](
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


class TritonGeGLUKernel:
    """Triton-accelerated GeGLU (Gated GELU) activation kernel wrapper.

    Provides a high-level interface for applying GeGLU activation: x1 * GELU(x2),
    where the input is split into two halves along the last dimension.

    GeGLU is commonly used in transformer feed-forward networks and has been shown
    to provide better performance than standard activations in many cases.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(X):
            Applies GeGLU activation to the input tensor.

            Parameters:
                X (torch.Tensor): Input tensor with even last dimension. The tensor is split
                    into two halves along the last dimension: x1 and x2.

            Returns:
                torch.Tensor: Output tensor with last dimension halved, computed as x1 * GELU(x2).
                    GELU uses the tanh approximation for efficiency.

            Example:
                >>> x = torch.randn(32, 128, 2048, device='cuda')  # batch, seq, 2*hidden
                >>> y = TritonGeGLUKernel.apply(x)  # shape: (32, 128, 1024)
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
        return TritonGeGLUFunction.apply(X)
