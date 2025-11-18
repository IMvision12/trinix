import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def squared_relu_forward_kernel(
    Y_ptr,
    X_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Squared ReLU activation forward kernel.

    Computes Squared ReLU activation: (max(0, x))^2.

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
    relu_x = tl.maximum(x, 0.0)
    output = relu_x * relu_x

    tl.store(Y_ptr + offsets, output, mask=mask)


@triton.jit
def squared_relu_backward_kernel(
    dX_ptr,
    dY_ptr,
    X_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Squared ReLU activation backward kernel.

    Computes gradient of Squared ReLU activation with respect to input.
    Gradient: d/dx[(max(0,x))^2] = 2 * max(0, x) when x > 0, else 0

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

    relu_x = tl.maximum(x, 0.0)
    dx = 2.0 * relu_x * dy

    tl.store(dX_ptr + offsets, dx, mask=mask)


class TritonSquaredReLUFunction(torch.autograd.Function):
    """Autograd function for Squared ReLU activation.

    This function wraps the Squared ReLU kernel for automatic differentiation.

    Methods:
        forward(ctx, X):
            Computes Squared ReLU activation: (max(0, x))^2.

            Parameters:
                ctx: Autograd context for saving tensors needed in backward pass.
                X (torch.Tensor): Input tensor of any shape.

            Returns:
                torch.Tensor: Output tensor with Squared ReLU activation applied, same shape as input.

        backward(ctx, dY):
            Backward pass for Squared ReLU activation.

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
        squared_relu_forward_kernel[grid](
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
        squared_relu_backward_kernel[grid](
            dX,
            dY_flat,
            X_flat,
            ctx.n_elements,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        return dX.view(*shape)


class TritonSquaredReLUKernel:
    """Triton-accelerated Squared ReLU activation kernel wrapper.

    Provides a high-level interface for applying Squared ReLU activation function,
    which squares the output of ReLU: (max(0, x))^2.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(X):
            Applies Squared ReLU activation to the input tensor.

            Parameters:
                X (torch.Tensor): Input tensor of any shape.

            Returns:
                torch.Tensor: Output tensor with Squared ReLU activation applied, same shape as input.
                    Computed as: (max(0, x))^2.
                    This activation provides stronger non-linearity than standard ReLU and
                    can help with gradient flow in deep networks.
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
        return TritonSquaredReLUFunction.apply(X)
