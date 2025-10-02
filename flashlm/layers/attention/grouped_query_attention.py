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
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, kernel_type, causal)
        
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_head_dim = embed_dim // num_kv_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        
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
        batch_size, seq_len, num_kv_heads, head_dim = x.shape
        if self.num_queries_per_kv == 1:
            return x
        
        x = x.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, self.num_queries_per_kv, head_dim)
        return x.contiguous().view(batch_size, seq_len, self.num_heads, head_dim)
    
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
        
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        attn_output = self.forward_attention(q, k, v, attn_mask)
        
        attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output, None