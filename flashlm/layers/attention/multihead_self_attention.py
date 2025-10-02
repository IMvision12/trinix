import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import FastBaseAttention


class FastMultiHeadSelfAttention(FastBaseAttention):
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
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, embed_dim = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        attn_output = self.forward_attention(q, k, v, attn_mask)
        
        attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output, None