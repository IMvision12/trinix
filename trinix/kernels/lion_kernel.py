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
