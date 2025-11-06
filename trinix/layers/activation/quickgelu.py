import torch
import torch.nn as nn

from ...kernels import TritonQuickGELUKernel


class FastQuickGELU(nn.Module):
    """Fast QuickGELU activation layer with automatic Triton/PyTorch backend selection.

    QuickGELU is a faster approximation of GELU: x * sigmoid(1.702 * x).
    Automatically uses Triton kernels for large tensors (hidden_size >= 512) when available,
    falling back to PyTorch implementation otherwise.

    Args:
        hidden_size (int): Size of the hidden dimension. Used to determine whether to use Triton.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - Input: (*, hidden_size) where * means any number of dimensions
        - Output: (*, hidden_size)

    Examples:
        >>> layer = FastQuickGELU(hidden_size=1024)
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
        if not TritonQuickGELUKernel.is_available():
            return False
        return self.hidden_size >= 512

    def _triton_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return TritonQuickGELUKernel.apply(hidden_states)

    def _pytorch_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * torch.sigmoid(1.702 * hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._check_triton_availability():
            return self._triton_forward(hidden_states)
        else:
            return self._pytorch_forward(hidden_states)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, use_triton={self.use_triton}"
