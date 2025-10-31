from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from ..kernels.lion_kernel import TritonLionKernel


class FastLion(Optimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        super(FastLion, self).__init__(params, defaults)
        self.use_triton = use_triton

    def _check_triton_availability(self, param: torch.Tensor) -> bool:
        if not self.use_triton:
            return False
        if not TritonLionKernel.is_available():
            return False
        if not param.is_cuda:
            return False
        return True

    def _triton_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        grad_scale: float = 1.0,
    ):
        p_flat = param.view(-1)
        grad_flat = grad.view(-1)
        exp_avg_flat = exp_avg.view(-1)

        TritonLionKernel.apply(
            p_flat,
            grad_flat,
            exp_avg_flat,
            lr,
            beta1,
            beta2,
            weight_decay,
            grad_scale,
        )

    def _pytorch_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        grad_scale: float = 1.0,
    ):
        if grad_scale != 1.0:
            grad = grad / grad_scale
        c_t = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
        param.add_(c_t.sign_().add_(param, alpha=weight_decay), alpha=-lr)
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FastLion does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                state["step"] += 1

                if self._check_triton_availability(p):
                    self._triton_step(
                        p,
                        grad,
                        exp_avg,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        grad_scale,
                    )
                else:
                    self._pytorch_step(
                        p,
                        grad,
                        exp_avg,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        grad_scale,
                    )

        return loss

    def __repr__(self):
        return f"FastLion(lr={self.defaults['lr']}, betas={self.defaults['betas']}, weight_decay={self.defaults['weight_decay']}, use_triton={self.use_triton})"
