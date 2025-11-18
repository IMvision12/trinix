import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def mish_forward_kernel(
    Y_ptr,
    X_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Mish activation forward kernel.

    Computes Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    Mish is a smooth, non-monotonic activation function.

    Args:
        Y_ptr: Pointer to output tensor.
        X_ptr: Pointer to input tensor.
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    softplus = tl.log(1.0 + tl.exp(x))

    exp_2softplus = tl.exp(2.0 * softplus)
    tanh_softplus = (exp_2softplus - 1.0) / (exp_2softplus + 1.0)

    output = x * tanh_softplus
    tl.store(Y_ptr + offsets, output, mask=mask)


@triton.jit
def mish_backward_kernel(
    dX_ptr,
    dY_ptr,
    X_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Mish activation backward kernel.

    Computes gradient of Mish activation with respect to input.
    Gradient: d/dx[x * tanh(softplus(x))] = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)

    Args:
        dX_ptr: Pointer to input gradient tensor.
        dY_ptr: Pointer to output gradient tensor.
        X_ptr: Pointer to input tensor from forward pass.
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dY_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    exp_x = tl.exp(x)
    softplus = tl.log(1.0 + exp_x)

    exp_2softplus = tl.exp(2.0 * softplus)
    tanh_softplus = (exp_2softplus - 1.0) / (exp_2softplus + 1.0)

    sech2_softplus = 1.0 - tanh_softplus * tanh_softplus
    sigmoid_x = exp_x / (1.0 + exp_x)

    dx = dy * (tanh_softplus + x * sech2_softplus * sigmoid_x)
    tl.store(dX_ptr + offsets, dx, mask=mask)


class TritonMishFunction(torch.autograd.Function):
    """Autograd function for Mish activation.

    This function wraps the Mish kernel for automatic differentiation.

    Methods:
        forward(ctx, X):
            Computes Mish activation: x * tanh(softplus(x)).

            Parameters:
                ctx: Autograd context for saving tensors needed in backward pass.
                X (torch.Tensor): Input tensor of any shape.

            Returns:
                torch.Tensor: Output tensor with Mish activation applied, same shape as input.

        backward(ctx, dY):
            Backward pass for Mish activation.

            Parameters:
                ctx: Autograd context containing saved input tensor.
                dY: Gradient of loss with respect to the output.

            Returns:
                torch.Tensor: Gradient of loss with respect to the input.
    """

    @staticmethod
    def forward(ctx, X):
        shape = X.shape
        X_flat = X.contiguous().view(-1)
        n_elements = X_flat.numel()

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_elements)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        Y = torch.empty_like(X_flat)

        mish_forward_kernel[grid](
            Y,
            X_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(X_flat)
        ctx.n_elements = n_elements
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dY_flat = dY.contiguous().view(-1)
        (X_flat,) = ctx.saved_tensors

        grid = lambda meta: (triton.cdiv(ctx.n_elements, meta["BLOCK_SIZE"]),)
        dX = torch.empty_like(X_flat)
        mish_backward_kernel[grid](
            dX,
            dY_flat,
            X_flat,
            ctx.n_elements,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        return dX.view(*shape)


class TritonMishKernel:
    """Triton-accelerated Mish activation kernel wrapper.

    Provides a high-level interface for applying Mish activation function,
    which is a smooth, non-monotonic activation: x * tanh(softplus(x)).

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(X):
            Applies Mish activation to the input tensor.

            Parameters:
                X (torch.Tensor): Input tensor of any shape.

            Returns:
                torch.Tensor: Output tensor with Mish activation applied, same shape as input.
                    Computed as: x * tanh(ln(1 + exp(x))).
                    Mish is a smooth, self-regularized non-monotonic activation function.
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
        return TritonMishFunction.apply(X)
