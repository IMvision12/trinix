import torch
import torch.nn as nn

from ...kernels import TritonRMSNormKernel


class FastRMSNorm(nn.Module):
    """Fast RMS Normalization with automatic Triton/PyTorch backend selection.

    RMS Normalization is a simpler alternative to Layer Normalization that normalizes using
    only the root mean square (RMS) without centering (no mean subtraction). This makes it
    computationally more efficient while maintaining similar performance.
    Automatically uses Triton kernels for large tensors (hidden_size > 2048) when available,
    falling back to PyTorch implementation otherwise.

    Args:
        hidden_size (int): Size of the hidden dimension to normalize.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - Input: (*, hidden_size) where * means any number of dimensions
        - Output: (*, hidden_size) same shape as input

    Examples:
        >>> rms_norm = FastRMSNorm(hidden_size=768)
        >>> x = torch.randn(32, 128, 768)  # (batch, seq_len, hidden_size)
        >>> output = rms_norm(x)  # shape: (32, 128, 768)

        >>> # RMSNorm is commonly used in modern LLMs like LLaMA
        >>> rms_norm = FastRMSNorm(4096, eps=1e-5)
    """

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
        return self.hidden_size > 2048

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
        backend = "triton" if self._check_triton_availability() else "pytorch"
        return f"hidden_size={self.hidden_size}, eps={self.eps}, backend={backend}"
