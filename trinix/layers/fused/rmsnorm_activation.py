import torch
import torch.nn as nn

from ...kernels.fused import TritonFusedRMSNormActivationKernel

_SUPPORTED_ACTIVATIONS = {"mish", "quickgelu", "squared_relu"}


class FastFusedRMSNormActivation(nn.Module):
    """Fast fused RMSNorm + Activation with automatic Triton/PyTorch backend selection.

    Combines RMS normalization and activation into a single kernel launch,
    eliminating the memory round-trip between normalization and activation.

    Computation:
        normed = (X / sqrt(mean(X^2) + eps)) * W
        Y = activation(normed)

    Supported activations: mish, quickgelu, squared_relu.

    Args:
        hidden_size (int): Size of the hidden dimension to normalize.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        activation (str, optional): Activation function name. Defaults to "mish".
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - Input: (*, hidden_size) where * means any number of dimensions
        - Output: (*, hidden_size) same shape as input

    Examples:
        >>> layer = FastFusedRMSNormActivation(4096, activation="mish")
        >>> output = layer(x)  # single kernel: normalize + activate
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "mish",
        use_triton: bool = True,
    ):
        super().__init__()
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Supported: {sorted(_SUPPORTED_ACTIVATIONS)}"
            )
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        self.use_triton = use_triton

        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _check_triton_availability(self) -> bool:
        if not self.use_triton:
            return False
        if not TritonFusedRMSNormActivationKernel.is_available():
            return False
        return self.hidden_size > 2048

    def _triton_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() > 2:
            batch_size = hidden_states.numel() // self.hidden_size
            hidden_states = hidden_states.view(batch_size, self.hidden_size)
        output = TritonFusedRMSNormActivationKernel.apply(
            hidden_states, self.weight, self.eps, self.activation
        )
        return output.view(original_shape)

    def _pytorch_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        normed = (self.weight * normed.to(input_dtype)).to(torch.float32)

        if self.activation == "mish":
            output = normed * torch.tanh(torch.nn.functional.softplus(normed))
        elif self.activation == "quickgelu":
            output = normed * torch.sigmoid(1.702 * normed)
        elif self.activation == "squared_relu":
            output = torch.relu(normed) ** 2

        return output.to(input_dtype)

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
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, "
            f"activation={self.activation}, backend={backend}"
        )
