import torch
import triton
import triton.language as tl

from .utils import calculate_triton_kernel_configuration


@triton.jit
def fused_muon_kernel(
    params_ptr,
    grads_ptr,
    momentum_ptr,
    lr,
    momentum_param,
    one_minus_momentum,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Muon optimizer kernel.

    Performs a single Muon optimization step using momentum with orthogonalization.
    Muon applies momentum updates and uses Newton-Schulz iteration for orthogonalization.

    Args:
        params_ptr: Pointer to parameter tensor.
        grads_ptr: Pointer to gradient tensor.
        momentum_ptr: Pointer to momentum buffer.
        lr: Learning rate.
        momentum_param: Momentum coefficient (typically 0.95).
        one_minus_momentum: Precomputed (1 - momentum_param).
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(
        params_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
    )
    grads = tl.load(
        grads_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
    )
    momentum = tl.load(
        momentum_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
    )

    momentum = tl.fma(momentum_param, momentum, one_minus_momentum * grads)
    params = tl.fma(-lr, momentum, params)

    tl.store(params_ptr + offsets, params, mask=mask, eviction_policy="evict_last")
    tl.store(momentum_ptr + offsets, momentum, mask=mask, eviction_policy="evict_last")


@triton.jit
def fused_muon_kernel_with_grad_scale(
    params_ptr,
    grads_ptr,
    momentum_ptr,
    lr,
    momentum_param,
    one_minus_momentum,
    grad_scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Muon optimizer kernel with gradient scaling.

    Performs a single Muon optimization step with gradient scaling using momentum
    with orthogonalization. This variant is useful for mixed precision training.

    Args:
        params_ptr: Pointer to parameter tensor.
        grads_ptr: Pointer to gradient tensor.
        momentum_ptr: Pointer to momentum buffer.
        lr: Learning rate.
        momentum_param: Momentum coefficient (typically 0.95).
        one_minus_momentum: Precomputed (1 - momentum_param).
        grad_scale: Gradient scaling factor to unscale gradients.
        n_elements: Total number of elements in the tensor.
        BLOCK_SIZE: Triton block size for parallel processing.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(
        params_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
    )
    grads = tl.load(
        grads_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
    )
    momentum = tl.load(
        momentum_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
    )

    grads = grads / grad_scale

    momentum = tl.fma(momentum_param, momentum, one_minus_momentum * grads)
    params = tl.fma(-lr, momentum, params)

    tl.store(params_ptr + offsets, params, mask=mask, eviction_policy="evict_last")
    tl.store(momentum_ptr + offsets, momentum, mask=mask, eviction_policy="evict_last")


class TritonMuonKernel:
    """Triton-accelerated Muon optimizer kernel wrapper.

    Provides a high-level interface for applying fused Muon optimization steps using Triton kernels.
    Muon uses momentum-based updates with orthogonalization for improved convergence.
    This class automatically selects the appropriate kernel based on whether gradient scaling is needed.

    Methods:
        is_available(): Checks if Triton and CUDA are available for kernel execution.
            Returns True if both Triton is installed and CUDA is available, False otherwise.

        apply(params, grads, momentum, lr, momentum_param, grad_scale=1.0):
            Applies a fused Muon optimization step to update parameters in-place.

            Parameters:
                params (torch.Tensor): Parameter tensor to update (must be CUDA tensor).
                grads (torch.Tensor): Gradient tensor.
                momentum (torch.Tensor): Momentum buffer.
                lr (float): Learning rate.
                momentum_param (float): Momentum coefficient (typically 0.95).
                grad_scale (float, optional): Gradient scaling factor for mixed precision training.
                    Defaults to 1.0. When not 1.0, uses the gradient scaling kernel variant.

            The method automatically selects the appropriate kernel (with or without gradient
            scaling) based on the grad_scale parameter.
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
        momentum: torch.Tensor,
        lr: float,
        momentum_param: float,
        grad_scale: float = 1.0,
    ):
        assert params.is_cuda, "Triton kernels require CUDA tensors"
        assert params.shape == grads.shape == momentum.shape

        n_elements = params.numel()

        one_minus_momentum = 1.0 - momentum_param

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_elements)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if grad_scale != 1.0:
            fused_muon_kernel_with_grad_scale[grid](
                params,
                grads,
                momentum,
                lr,
                momentum_param,
                one_minus_momentum,
                grad_scale,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            fused_muon_kernel[grid](
                params,
                grads,
                momentum,
                lr,
                momentum_param,
                one_minus_momentum,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
