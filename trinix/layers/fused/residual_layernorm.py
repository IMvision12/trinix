import torch
import torch.nn as nn

from ...kernels.fused import TritonFusedResidualLayerNormKernel


class FastFusedResidualLayerNorm(nn.Module):
    """Fast fused Residual + Layer Normalization with automatic Triton/PyTorch backend selection.

    Combines residual addition and layer normalization into a single kernel launch.

    Computation:
        SumOut = X + Residual
        Y = ((SumOut - mean(SumOut)) / sqrt(var(SumOut) + eps)) * W + b

    Args:
        hidden_size (int): Size of the hidden dimension to normalize.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - X: (*, hidden_size) where * means any number of dimensions
        - Residual: (*, hidden_size) same shape as X
        - Output: (*, hidden_size)

    Examples:
        >>> fused_norm = FastFusedResidualLayerNorm(hidden_size=4096)
        >>> output = fused_norm(attention_output, residual)  # single kernel
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        use_triton: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_triton = use_triton

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def _check_triton_availability(self) -> bool:
        if not self.use_triton:
            return False
        if not TritonFusedResidualLayerNormKernel.is_available():
            return False
        return self.hidden_size > 4096

    def _triton_forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() > 2:
            batch_size = hidden_states.numel() // self.hidden_size
            hidden_states = hidden_states.view(batch_size, self.hidden_size)
            residual = residual.view(batch_size, self.hidden_size)
        output = TritonFusedResidualLayerNormKernel.apply(
            hidden_states, residual, self.weight, self.bias, self.eps
        )
        return output.view(original_shape)

    def _pytorch_forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        sum_out = hidden_states.to(torch.float32) + residual.to(torch.float32)
        mean = sum_out.mean(-1, keepdim=True)
        variance = (sum_out - mean).pow(2).mean(-1, keepdim=True)
        sum_out = (sum_out - mean) * torch.rsqrt(variance + self.eps)
        return (self.weight * sum_out.to(input_dtype)) + self.bias

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_states with last dimension {self.hidden_size}, "
                f"but got shape {hidden_states.shape}"
            )

        if self._check_triton_availability():
            return self._triton_forward(hidden_states, residual)
        else:
            return self._pytorch_forward(hidden_states, residual)

    def extra_repr(self) -> str:
        backend = "triton" if self._check_triton_availability() else "pytorch"
        return f"hidden_size={self.hidden_size}, eps={self.eps}, backend={backend}"
