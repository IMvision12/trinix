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
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    relu_x = tl.maximum(x_f32, 0.0)
    output = relu_x * relu_x
    tl.store(Y_ptr + offsets, output.to(x.dtype), mask=mask)


@triton.jit
def squared_relu_backward_kernel(
    dX_ptr,
    dY_ptr,
    X_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dY = tl.load(dY_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    dY_f32 = dY.to(tl.float32)
    x_f32 = x.to(tl.float32)
    relu_x = tl.maximum(x_f32, 0.0)
    drelu_dx = tl.where(x_f32 > 0.0, 1.0, 0.0)
    dX = dY_f32 * 2.0 * relu_x * drelu_dx
    tl.store(dX_ptr + offsets, dX.to(x.dtype), mask=mask)


class TritonSquaredReLUFunction(torch.autograd.Function):
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
