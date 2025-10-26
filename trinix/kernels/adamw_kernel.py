import torch
import triton
import triton.language as tl


@triton.jit
def fused_adamw_kernel(
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

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    params = params * (1.0 - lr * weight_decay)
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grads
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grads * grads
    exp_avg_corrected = exp_avg / bias_correction1

    exp_avg_sq_corrected = exp_avg_sq / bias_correction2

    denom = tl.sqrt(exp_avg_sq_corrected) + eps

    params = params - lr * exp_avg_corrected / denom

    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


@triton.jit
def fused_adamw_kernel_with_grad_scale(
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

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    grads = grads / grad_scale
    params = params * (1.0 - lr * weight_decay)
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grads
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grads * grads
    exp_avg_corrected = exp_avg / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2

    denom = tl.sqrt(exp_avg_sq_corrected) + eps
    params = params - lr * exp_avg_corrected / denom

    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


class TritonAdamWKernel:
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

        BLOCK_SIZE = 1024

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if grad_scale != 1.0:
            fused_adamw_kernel_with_grad_scale[grid](
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
            )
        else:
            fused_adamw_kernel[grid](
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
            )
