import torch
import torch.nn as nn

from ...kernels import TritonLayerNormKernel


class FastLayerNorm(nn.Module):
    """Fast Layer Normalization with automatic Triton/PyTorch backend selection.

    Layer Normalization normalizes inputs across the feature dimension by subtracting the mean
    and dividing by the standard deviation, then applies learned affine transformation.
    Automatically uses Triton kernels for large tensors (hidden_size >= 4096) when available,
    falling back to PyTorch's native implementation otherwise.

    Args:
        normalized_shape (int or tuple): Shape of the input to be normalized (typically hidden_size).
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        elementwise_affine (bool, optional): Whether to learn affine parameters (weight and bias).
            Defaults to True.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - Input: (*, normalized_shape) where * means any number of dimensions
        - Output: (*, normalized_shape) same shape as input

    Examples:
        >>> layer_norm = FastLayerNorm(normalized_shape=768)
        >>> x = torch.randn(32, 128, 768)  # (batch, seq_len, hidden_size)
        >>> output = layer_norm(x)  # shape: (32, 128, 768)

        >>> # Without affine transformation
        >>> layer_norm = FastLayerNorm(768, elementwise_affine=False)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_triton = use_triton
        self.pytorch_layernorm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        if self.elementwise_affine:
            self.weight = self.pytorch_layernorm.weight
            self.bias = self.pytorch_layernorm.bias
        else:
            self.weight = None
            self.bias = None

    def _check_triton_availability(self) -> bool:
        if not self.use_triton:
            return False
        if not TritonLayerNormKernel.is_available():
            return False
        if len(self.normalized_shape) != 1:
            return False
        hidden_size = self.normalized_shape[0]
        return hidden_size >= 4096

    def _reshape_for_triton(self, input: torch.Tensor):
        original_shape = input.shape
        if input.dim() > 2:
            batch_size = input.numel() // self.normalized_shape[0]
            input = input.view(batch_size, self.normalized_shape[0])
        return (input, original_shape)

    def _triton_forward(self, input: torch.Tensor) -> torch.Tensor:
        input_2d, original_shape = self._reshape_for_triton(input)
        output = TritonLayerNormKernel.apply(
            input_2d,
            self.pytorch_layernorm.weight if self.elementwise_affine else None,
            self.pytorch_layernorm.bias if self.elementwise_affine else None,
            self.eps,
        )
        return output.view(original_shape)

    def _pytorch_forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pytorch_layernorm(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-len(self.normalized_shape) :] != self.normalized_shape:
            raise ValueError(
                f"Expected input shape ending with {self.normalized_shape}, but got {input.shape}"
            )
        if self._check_triton_availability():
            return self._triton_forward(input)
        else:
            return self._pytorch_forward(input)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, use_triton={self.use_triton}"
