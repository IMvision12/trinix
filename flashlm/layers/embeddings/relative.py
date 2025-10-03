import torch
import torch.nn as nn


class RelativePositionEmbedding(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, max_relative_position: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position
        
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, head_dim
        )
        
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        positions = torch.clamp(positions, -self.max_relative_position, self.max_relative_position)
        positions = positions + self.max_relative_position
        return positions
    
    def forward(self, seq_len: int) -> torch.Tensor:
        relative_positions = self._get_relative_positions(seq_len)
        relative_embeddings = self.relative_position_embeddings(relative_positions)
        
        relative_embeddings = relative_embeddings.permute(2, 0, 1).unsqueeze(0)
        relative_embeddings = relative_embeddings.expand(self.num_heads, -1, -1, -1)
        
        return relative_embeddings