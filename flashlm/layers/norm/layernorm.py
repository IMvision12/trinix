import torch
import torch.nn as nn

from ...kernels import TritonLayerNormKernel


class FastLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        triton: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.triton = triton
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
        if not self.triton:
            return False
        if not TritonLayerNormKernel.is_available():
            return False
        if len(self.normalized_shape) != 1:
            return False
        hidden_size = self.normalized_shape[0]
        return hidden_size >= 2048

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
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, triton={self.triton}"
