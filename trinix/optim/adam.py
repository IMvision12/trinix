from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from ..kernels.adam_kernel import TritonAdamKernel


class FastAdam(Optimizer):
    """Fast Adam optimizer with automatic Triton/PyTorch backend selection.

    Implements Adam algorithm with adaptive learning rates. Automatically uses Triton kernels
    for CUDA tensors when available, falling back to PyTorch implementation otherwise.
    Adam maintains exponential moving averages of both gradients (first moment) and
    squared gradients (second moment) with bias correction.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): Coefficients for computing running averages
            of gradient and its square. Defaults to (0.9, 0.999).
        eps (float, optional): Term added to denominator for numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty) coefficient. Defaults to 0.0.
        use_triton (bool, optional): Whether to enable Triton kernels for CUDA tensors.
            Defaults to True.

    Examples:
        >>> optimizer = FastAdam(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

        >>> # With mixed precision training
        >>> scaler = torch.cuda.amp.GradScaler()
        >>> optimizer.step(grad_scaler=scaler.get_scale())
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_triton: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(FastAdam, self).__init__(params, defaults)
        self.use_triton = use_triton

    def _check_triton_availability(self, param: torch.Tensor) -> bool:
        if not self.use_triton:
            return False
        if not TritonAdamKernel.is_available():
            return False
        if not param.is_cuda:
            return False
        return True

    def _triton_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
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
        p_flat = param.view(-1)
        grad_flat = grad.view(-1)
        exp_avg_flat = exp_avg.view(-1)
        exp_avg_sq_flat = exp_avg_sq.view(-1)

        TritonAdamKernel.apply(
            p_flat,
            grad_flat,
            exp_avg_flat,
            exp_avg_sq_flat,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            grad_scale,
        )

    def _pytorch_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
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
        if grad_scale != 1.0:
            grad = grad / grad_scale

        if weight_decay != 0.0:
            grad = grad.add(param, alpha=weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1
        denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable] = None, grad_scaler: Optional[float] = None
    ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_scale = grad_scaler if grad_scaler is not None else 1.0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FastAdam does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if self._check_triton_availability(p):
                    self._triton_step(
                        p,
                        grad,
                        exp_avg,
                        exp_avg_sq,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        grad_scale,
                    )
                else:
                    self._pytorch_step(
                        p,
                        grad,
                        exp_avg,
                        exp_avg_sq,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        grad_scale,
                    )

        return loss
