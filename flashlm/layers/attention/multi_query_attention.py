import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import FastBaseAttention


class FastMultiQueryAttention(FastBaseAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kernel_type: str = "flash",
        causal: bool = False,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, kernel_type, causal)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(batch_size, seq_len, self.num_heads, head_dim)
        return x
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, embed_dim = query.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        attn_output = self.forward_attention(q, k, v, attn_mask)
        
        attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output, None