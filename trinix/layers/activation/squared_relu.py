import torch
import torch.nn as nn

from ...kernels import TritonSquaredReLUKernel


class FastSquaredReLU(nn.Module):
    """Fast Squared ReLU activation layer with automatic Triton/PyTorch backend selection.

    Squared ReLU computes: (max(0, x))^2, providing stronger non-linearity than standard ReLU.
    Automatically uses Triton kernels for large tensors (hidden_size >= 512) when available,
    falling back to PyTorch implementation otherwise.

    Args:
        hidden_size (int): Size of the hidden dimension. Used to determine whether to use Triton.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - Input: (*, hidden_size) where * means any number of dimensions
        - Output: (*, hidden_size)

    Examples:
        >>> layer = FastSquaredReLU(hidden_size=1024)
        >>> x = torch.randn(32, 1024)
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
        if not TritonSquaredReLUKernel.is_available():
            return False
        return self.hidden_size >= 512

    def _triton_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return TritonSquaredReLUKernel.apply(hidden_states)

    def _pytorch_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.relu(hidden_states) ** 2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._check_triton_availability():
            return self._triton_forward(hidden_states)
        else:
            return self._pytorch_forward(hidden_states)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, use_triton={self.use_triton}"
