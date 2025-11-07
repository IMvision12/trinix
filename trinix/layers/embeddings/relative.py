import torch
import torch.nn as nn

try:
    from ...kernels import TritonRelativeKernel

    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TRITON_AVAILABLE = False
    TritonRelativeKernel = None


class FastRelativePositionEmbedding(nn.Module):
    """Fast relative position embedding layer with learned position embeddings.

    Computes position-dependent embeddings based on relative distances between query and key positions.
    Uses learned embeddings for each relative position within a maximum range.
    Automatically uses Triton kernels when available, falling back to PyTorch implementation.

    Args:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        max_relative_position (int, optional): Maximum relative position distance to consider.
            Positions beyond this range are clamped. Defaults to 128.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - Output: (batch_size, num_heads, seq_len, seq_len, head_dim) containing
          learned embeddings for each relative position pair

    Examples:
        >>> rel_pos = FastRelativePositionEmbedding(num_heads=8, head_dim=64)
        >>> embeddings = rel_pos(seq_len=128, batch_size=4)  # shape: (4, 8, 128, 128, 64)
        >>> # Use in attention computation with relative position information
    """

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

    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(
            seq_len, device=device
        ).unsqueeze(1)
        positions = torch.clamp(
            positions, -self.max_relative_position, self.max_relative_position
        )
        positions = positions + self.max_relative_position
        return positions

    def forward(self, seq_len: int, batch_size: int = 1) -> torch.Tensor:
        device = self.relative_position_embeddings.weight.device
        relative_positions = self._get_relative_positions(seq_len, device)
        if (
            self.use_triton
            and TRITON_AVAILABLE
            and (TritonRelativeKernel is not None)
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
