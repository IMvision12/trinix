import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def fused_adam_kernel(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    bias_correction1,
    bias_correction2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Adam optimizer kernel.

    Performs a single Adam optimization step with bias correction and optional weight decay.

    Args:
        params_ptr: Pointer to parameter tensor.
        grads_ptr: Pointer to gradient tensor.
        exp_avg_ptr: Pointer to exponential moving average of gradients (first moment).
        exp_avg_sq_ptr: Pointer to exponential moving average of squared gradients (second moment).
        lr: Learning rate.
        beta1: Exponential decay rate for first moment estimates.
        beta2: Exponential decay rate for second moment estimates.
        eps: Small constant for numerical stability.
        weight_decay: Weight decay coefficient (L2 penalty).
        step: Current optimization step number.
        bias_correction1: Bias correction factor for first moment (1 - beta1^step).
        bias_correction2: Bias correction factor for second moment (1 - beta2^step).
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    grads = tl.load(grads_ptr + offsets, mask=mask, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)

    grads = tl.where(weight_decay != 0.0, grads + weight_decay * params, grads)

    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2

    exp_avg = tl.fma(beta1, exp_avg, one_minus_beta1 * grads)

    exp_avg_sq = tl.fma(beta2, exp_avg_sq, one_minus_beta2 * grads * grads)

    exp_avg_corrected = exp_avg / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2

    denom = tl.sqrt(exp_avg_sq_corrected) + eps

    params = params - lr * exp_avg_corrected / denom

    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


@triton.jit
def fused_adam_kernel_with_grad_scale(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    bias_correction1,
    bias_correction2,
    grad_scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Adam optimizer kernel with gradient scaling.

    Performs a single Adam optimization step with gradient scaling, bias correction, and optional weight decay.
    This variant is useful for mixed precision training where gradients need to be unscaled.

    Args:
        params_ptr: Pointer to parameter tensor.
        grads_ptr: Pointer to gradient tensor.
        exp_avg_ptr: Pointer to exponential moving average of gradients (first moment).
        exp_avg_sq_ptr: Pointer to exponential moving average of squared gradients (second moment).
        lr: Learning rate.
        beta1: Exponential decay rate for first moment estimates.
        beta2: Exponential decay rate for second moment estimates.
        eps: Small constant for numerical stability.
        weight_decay: Weight decay coefficient (L2 penalty).
        step: Current optimization step number.
        bias_correction1: Bias correction factor for first moment (1 - beta1^step).
        bias_correction2: Bias correction factor for second moment (1 - beta2^step).
        grad_scale: Gradient scaling factor to unscale gradients.
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(params_ptr + offsets, mask=mask, other=0.0)
    grads = tl.load(grads_ptr + offsets, mask=mask, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)

    grads = grads / grad_scale

    grads = tl.where(weight_decay != 0.0, grads + weight_decay * params, grads)

    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2

    exp_avg = tl.fma(beta1, exp_avg, one_minus_beta1 * grads)

    exp_avg_sq = tl.fma(beta2, exp_avg_sq, one_minus_beta2 * grads * grads)

    exp_avg_corrected = exp_avg / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2

    denom = tl.sqrt(exp_avg_sq_corrected) + eps

    params = params - lr * exp_avg_corrected / denom

    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


class TritonAdamKernel:
    """Triton-accelerated Adam optimizer kernel wrapper.

    Provides a high-level interface for applying fused Adam optimization steps using Triton kernels.
    This class automatically selects the appropriate kernel based on whether gradient scaling is needed.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, grad_scale=1.0):
            Applies a fused Adam optimization step to update parameters in-place.
            
            Parameters:
                params (torch.Tensor): Parameter tensor to update (must be CUDA tensor).
                grads (torch.Tensor): Gradient tensor.
                exp_avg (torch.Tensor): Exponential moving average of gradients (first moment).
                exp_avg_sq (torch.Tensor): Exponential moving average of squared gradients (second moment).
                lr (float): Learning rate.
                beta1 (float): Exponential decay rate for first moment estimates (typically 0.9).
                beta2 (float): Exponential decay rate for second moment estimates (typically 0.999).
                eps (float): Small constant for numerical stability (typically 1e-8).
                weight_decay (float): Weight decay coefficient (L2 penalty).
                step (int): Current optimization step number (1-indexed).
                grad_scale (float, optional): Gradient scaling factor for mixed precision training. 
                    Defaults to 1.0. When not 1.0, uses the gradient scaling kernel variant.
            
            The method automatically computes bias corrections and selects the appropriate kernel
            (with or without gradient scaling) based on the grad_scale parameter.
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
        exp_avg_sq: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
        grad_scale: float = 1.0,
    ):
        assert params.is_cuda, "Triton kernels require CUDA tensors"
        assert params.shape == grads.shape == exp_avg.shape == exp_avg_sq.shape

        n_elements = params.numel()

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_elements)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if grad_scale != 1.0:
            fused_adam_kernel_with_grad_scale[grid](
                params,
                grads,
                exp_avg,
                exp_avg_sq,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                bias_correction1,
                bias_correction2,
                grad_scale,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            fused_adam_kernel[grid](
                params,
                grads,
                exp_avg,
                exp_avg_sq,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                bias_correction1,
                bias_correction2,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
