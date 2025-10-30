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
