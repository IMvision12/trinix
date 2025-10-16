import torch
import torch.nn as nn

from ...kernels.geglu_kernel import TritonGeGLUKernel


class FastGeGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        use_triton: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_triton = use_triton

    def _check_triton_availability(self) -> bool:
        if not self.use_triton:
            return False
        if not TritonGeGLUKernel.is_available():
            return False
        return self.hidden_size >= 512

    def _triton_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return TritonGeGLUKernel.apply(hidden_states)

    def _pytorch_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dim = hidden_states.shape[-1]
        assert dim % 2 == 0, "Last dimension must be even for GeGLU"
        hidden_dim = dim // 2
        x1, x2 = hidden_states.split(hidden_dim, dim=-1)
        return x1 * torch.nn.functional.gelu(x2, approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] % 2 != 0:
            raise ValueError(
                f"Expected hidden_states with even last dimension, "
                f"but got shape {hidden_states.shape}"
            )

        if self._check_triton_availability():
            return self._triton_forward(hidden_states)
        else:
            return self._pytorch_forward(hidden_states)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, use_triton={self.use_triton}"
