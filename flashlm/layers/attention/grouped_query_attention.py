import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import FastBaseAttention


class FastGroupedQueryAttention(FastBaseAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kernel_type: str = "flash",
        causal: bool = False,
        head_dim: Optional[int] = None,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, kernel_type, causal)
        
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)
    
    def _rope_rotation(self, tensor):
        dim = tensor.shape[-1]
        half_dim = dim // 2
        left = tensor[..., :half_dim]
        right = tensor[..., half_dim:]
        return torch.cat([-right, left], dim=-1)
    
    def _apply_positional_encoding(self, q, k, cos_table, sin_table, axis=1):
        cos_expanded = cos_table.unsqueeze(axis)
        sin_expanded = sin_table.unsqueeze(axis)
        
        q_rotated = q * cos_expanded + self._rope_rotation(q) * sin_expanded
        k_rotated = k * cos_expanded + self._rope_rotation(k) * sin_expanded
        
        return q_rotated, k_rotated
    
    def _tile_kv_heads(self, tensor: torch.Tensor, factor: int) -> torch.Tensor:
        if factor == 1:
            return tensor
        
        bs, heads, seq, dim = tensor.shape
        expanded = tensor[:, :, None, :, :].expand(bs, heads, factor, seq, dim)
        return expanded.reshape(bs, heads * factor, seq, dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:

        bs, seq, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(bs, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if position_embeddings is not None:
            cos_table, sin_table = position_embeddings
            q, k = self._apply_positional_encoding(q, k, cos_table, sin_table)
        
        k = self._tile_kv_heads(k, self.num_key_value_groups)
        v = self._tile_kv_heads(v, self.num_key_value_groups)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = self.forward_attention(q, k, v, attention_mask)
        out = out.reshape(bs, seq, -1)
        out = self.o_proj(out)
        
        return out, None