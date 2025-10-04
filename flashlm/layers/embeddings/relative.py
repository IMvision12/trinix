import torch
import torch.nn as nn

try:
    from ...kernels import TritonRelativeKernel

    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TRITON_AVAILABLE = False
    TritonRelativeKernel = None


class FastRelativePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_relative_position: int = 128,
        use_triton: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position
        self.use_triton = use_triton and TRITON_AVAILABLE

        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, head_dim
        )

    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(
            seq_len
        ).unsqueeze(1)
        positions = torch.clamp(
            positions, -self.max_relative_position, self.max_relative_position
        )
        positions = positions + self.max_relative_position
        return positions

    def forward(self, seq_len: int, batch_size: int = 1) -> torch.Tensor:
        relative_positions = self._get_relative_positions(seq_len)

        if (
            self.use_triton
            and TRITON_AVAILABLE
            and TritonRelativeKernel is not None
            and TritonRelativeKernel.is_available()
            and self.relative_position_embeddings.weight.is_cuda
        ):
            return TritonRelativeKernel.apply(
                self.relative_position_embeddings.weight,
                relative_positions,
                batch_size,
                self.num_heads,
            )
        else:
            return self._compute_relative_pytorch(relative_positions, batch_size)

    def _compute_relative_pytorch(
        self, relative_positions: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        relative_embeddings = self.relative_position_embeddings(relative_positions)

        relative_embeddings = relative_embeddings.permute(2, 0, 1).unsqueeze(0)
        relative_embeddings = relative_embeddings.expand(self.num_heads, -1, -1, -1)

        if batch_size > 1:
            relative_embeddings = relative_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1, -1, -1
            )

        return relative_embeddings
