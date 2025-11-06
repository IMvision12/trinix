import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def fused_lion_kernel(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    lr,
    beta1,
    beta2,
    weight_decay,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Lion optimizer kernel.

    Performs a single Lion optimization step using sign-based momentum updates.
    Lion computes the sign of the interpolated momentum and uses it for parameter updates.

    Args:
        params_ptr: Pointer to parameter tensor.
        grads_ptr: Pointer to gradient tensor.
        exp_avg_ptr: Pointer to exponential moving average of gradients (momentum).
        lr: Learning rate.
        beta1: Interpolation parameter for momentum computation (typically 0.9).
        beta2: Exponential decay rate for momentum update (typically 0.99).
        weight_decay: Weight decay coefficient.
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

    c_t = beta1 * exp_avg + (1.0 - beta1) * grads

    update = tl.where(c_t > 0.0, 1.0, tl.where(c_t < 0.0, -1.0, 0.0))
    params = params - lr * (update + weight_decay * params)

    exp_avg = beta2 * exp_avg + (1.0 - beta2) * grads

    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)


@triton.jit
def fused_lion_kernel_with_grad_scale(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    lr,
    beta1,
    beta2,
    weight_decay,
    grad_scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Lion optimizer kernel with gradient scaling.

    Performs a single Lion optimization step with gradient scaling using sign-based momentum updates.
    This variant is useful for mixed precision training where gradients need to be unscaled.

    Args:
        params_ptr: Pointer to parameter tensor.
        grads_ptr: Pointer to gradient tensor.
        exp_avg_ptr: Pointer to exponential moving average of gradients (momentum).
        lr: Learning rate.
        beta1: Interpolation parameter for momentum computation (typically 0.9).
        beta2: Exponential decay rate for momentum update (typically 0.99).
        weight_decay: Weight decay coefficient.
        grad_scale: Gradient scaling factor to unscale gradients.
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

    grads = grads / grad_scale

    c_t = beta1 * exp_avg + (1.0 - beta1) * grads

    update = tl.where(c_t > 0.0, 1.0, tl.where(c_t < 0.0, -1.0, 0.0))
    params = params - lr * (update + weight_decay * params)

    exp_avg = beta2 * exp_avg + (1.0 - beta2) * grads

    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)


class TritonLionKernel:
    """Triton-accelerated Lion optimizer kernel wrapper.

    Provides a high-level interface for applying fused Lion optimization steps using Triton kernels.
    Lion (EvoLved Sign Momentum) uses sign-based updates with momentum, requiring only first-order
    moments (no second moment like Adam/AdamW). This class automatically selects the appropriate
    kernel based on whether gradient scaling is needed.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(params, grads, exp_avg, lr, beta1, beta2, weight_decay, grad_scale=1.0):
            Applies a fused Lion optimization step to update parameters in-place.

            Parameters:
                params (torch.Tensor): Parameter tensor to update (must be CUDA tensor).
                grads (torch.Tensor): Gradient tensor.
                exp_avg (torch.Tensor): Exponential moving average of gradients (momentum).
                lr (float): Learning rate.
                beta1 (float): Interpolation parameter for momentum computation (typically 0.9).
                beta2 (float): Exponential decay rate for momentum update (typically 0.99).
                weight_decay (float): Weight decay coefficient.
                grad_scale (float, optional): Gradient scaling factor for mixed precision training.
                    Defaults to 1.0. When not 1.0, uses the gradient scaling kernel variant.

            The method automatically selects the appropriate kernel (with or without gradient
            scaling) based on the grad_scale parameter. Unlike Adam/AdamW, Lion does not use
            bias correction or second moment estimates.
    """

    @staticmethod
    def is_available():
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def apply(
        params: torch.Tensor,
        grads: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        grad_scale: float = 1.0,
    ):
        assert params.is_cuda, "Triton kernels require CUDA tensors"
        assert params.shape == grads.shape == exp_avg.shape

        n_elements = params.numel()

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_elements)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if grad_scale != 1.0:
            fused_lion_kernel_with_grad_scale[grid](
                params,
                grads,
                exp_avg,
                lr,
                beta1,
                beta2,
                weight_decay,
                grad_scale,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            fused_lion_kernel[grid](
                params,
                grads,
                exp_avg,
                lr,
                beta1,
                beta2,
                weight_decay,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
