import warnings

import torch
import torch.nn as nn


class FusedElementwiseChain(nn.Module):
    """Fallback fused elementwise chain using torch.compile.

    Wraps an arbitrary chain of nn.Module layers with torch.compile for
    automatic kernel fusion when no pre-built fused kernel is available.

    This is used by the `fuse()` API when no registered fusion pattern matches.
    It provides automatic fusion through PyTorch's compiler rather than
    hand-written Triton kernels.

    Args:
        *modules: Variable number of nn.Module instances to chain together.

    Shape:
        - Input: Any shape accepted by the first module
        - Output: Shape produced by the last module

    Examples:
        >>> chain = FusedElementwiseChain(nn.LayerNorm(768), nn.ReLU())
        >>> output = chain(x)  # torch.compile optimized
    """

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.chain = nn.Sequential(*modules)
        self._compiled = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._compiled and torch.cuda.is_available():
            try:
                self.chain = torch.compile(self.chain)
                self._compiled = True
            except Exception:
                warnings.warn(
                    "torch.compile failed for FusedElementwiseChain, "
                    "falling back to eager execution.",
                    stacklevel=2,
                )
                self._compiled = True  # Don't retry
        return self.chain(hidden_states)

    def extra_repr(self) -> str:
        compiled = "compiled" if self._compiled else "eager"
        return f"n_modules={len(self.chain)}, mode={compiled}"
