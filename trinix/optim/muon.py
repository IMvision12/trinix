from typing import Callable, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer

from ..kernels.muon_kernel import TritonMuonKernel


class FastMuon(Optimizer):
    """Fast Muon optimizer with automatic Triton/PyTorch backend selection.

    Implements Muon algorithm which uses momentum-based updates with orthogonalization
    for improved convergence. Muon is designed to be memory-efficient and provides
    stable training dynamics. Automatically uses Triton kernels for CUDA tensors when
    available, falling back to PyTorch implementation otherwise.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Defaults to 2e-2.
        momentum (float, optional): Momentum coefficient. Defaults to 0.95.
        use_triton (bool, optional): Whether to enable Triton kernels for CUDA tensors.
            Defaults to True.

    Examples:
        >>> optimizer = FastMuon(model.parameters(), lr=2e-2, momentum=0.95)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

        >>> # Muon works well with larger learning rates
        >>> optimizer = FastMuon(model.parameters(), lr=3e-2)
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 2e-2,
        momentum: float = 0.95,
        use_triton: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
        )
        super(FastMuon, self).__init__(params, defaults)
        self.use_triton = use_triton

    def _check_triton_availability(self, param: torch.Tensor) -> bool:
        if not self.use_triton:
            return False
        if not TritonMuonKernel.is_available():
            return False
        if not param.is_cuda:
            return False
        return True

    def _triton_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        lr: float,
        momentum: float,
        grad_scale: float = 1.0,
    ):
        p_flat = param.view(-1)
        grad_flat = grad.view(-1)
        momentum_flat = momentum_buffer.view(-1)

        TritonMuonKernel.apply(
            p_flat,
            grad_flat,
            momentum_flat,
            lr,
            momentum,
            grad_scale,
        )

    def _pytorch_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        lr: float,
        momentum: float,
        grad_scale: float = 1.0,
    ):
        if grad_scale != 1.0:
            grad = grad / grad_scale

        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)
        param.add_(momentum_buffer, alpha=-lr)

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
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FastMuon does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                momentum_buffer = state["momentum_buffer"]
                state["step"] += 1

                if self._check_triton_availability(p):
                    self._triton_step(
                        p,
                        grad,
                        momentum_buffer,
                        lr,
                        momentum,
                        grad_scale,
                    )
                else:
                    self._pytorch_step(
                        p,
                        grad,
                        momentum_buffer,
                        lr,
                        momentum,
                        grad_scale,
                    )

        return loss

    def __repr__(self):
        return f"FastMuon(lr={self.defaults['lr']}, momentum={self.defaults['momentum']}, use_triton={self.use_triton})"
