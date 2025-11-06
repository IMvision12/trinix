import torch
import torch.nn as nn

from ...kernels.geglu_kernel import TritonGeGLUKernel


class FastGeGLU(nn.Module):
    """Fast GeGLU (Gated GELU) activation layer with automatic Triton/PyTorch backend selection.

    GeGLU splits the input into two halves and computes: x1 * GELU(x2).
    Automatically uses Triton kernels for large tensors (hidden_size >= 512) when available,
    falling back to PyTorch implementation otherwise.

    Args:
        hidden_size (int): Size of the hidden dimension. Used to determine whether to use Triton.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - Input: (*, hidden_size * 2) where * means any number of dimensions
        - Output: (*, hidden_size)

    Examples:
        >>> layer = FastGeGLU(hidden_size=1024)
        >>> x = torch.randn(32, 2048)  # batch_size=32, hidden_size*2=2048
        >>> output = layer(x)  # shape: (32, 1024)
    """

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
