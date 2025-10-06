import torch
import torch.nn as nn

from ...kernels import TritonRMSNormKernel


class FastRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        use_triton: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_triton = use_triton

        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _check_triton_availability(self) -> bool:
        if not self.use_triton:
            return False
        if not TritonRMSNormKernel.is_available():
            return False
        return self.hidden_size >= 2048

    def _reshape_for_triton(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        if hidden_states.dim() > 2:
            batch_size = hidden_states.numel() // self.hidden_size
            hidden_states = hidden_states.view(batch_size, self.hidden_size)
        return hidden_states, original_shape

    def _triton_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_2d, original_shape = self._reshape_for_triton(hidden_states)
        output = TritonRMSNormKernel.apply(
            hidden_states_2d,
            self.weight,
            self.eps,
        )
        return output.view(original_shape)

    def _pytorch_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_states with last dimension {self.hidden_size}, "
                f"but got shape {hidden_states.shape}"
            )

        if self._check_triton_availability():
            return self._triton_forward(hidden_states)
        else:
            return self._pytorch_forward(hidden_states)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}, use_triton={self.use_triton}"
