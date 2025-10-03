import torch
import torch.nn as nn
import math


class ALiBiPositionEmbedding(nn.Module):
    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes, persistent=False)
        
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2])
        
        return torch.tensor(slopes, dtype=torch.float32)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        positions = positions.abs()
        
        bias = positions.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
        bias = -bias
        
        return bias