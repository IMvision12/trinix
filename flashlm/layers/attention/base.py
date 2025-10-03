import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("Flash Attention not available. Install with: pip install flash-attn")

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    warnings.warn("Triton not available. Install with: pip install triton")


class FastBaseAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kernel_type: str = "flash",
        causal: bool = False,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.kernel_type = kernel_type
        
        if kernel_type == "flash" and not FLASH_ATTN_AVAILABLE:
            warnings.warn("Flash Attention not available, falling back to PyTorch")
            self.kernel_type = "pytorch"
        elif kernel_type == "triton" and not TRITON_AVAILABLE:
            warnings.warn("Triton not available, falling back to PyTorch")
            self.kernel_type = "pytorch"
        
        self.scale = self.head_dim ** -0.5
        
    def _apply_flash_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not FLASH_ATTN_AVAILABLE:
            return self._apply_pytorch_attention(q, k, v, attn_mask)
        
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=self.causal,
        )
        
        return out
    
    def _apply_triton_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        warnings.warn("Triton kernel not implemented yet, falling back to PyTorch")
        return self._apply_pytorch_attention(q, k, v, attn_mask)
    
    def _apply_pytorch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        q = q.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
        k = k.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
        v = v.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
        
        attn_weights = torch.bmm(q, k.transpose(-2, -1)) * self.scale
        
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                attn_mask = attn_mask.reshape(batch_size * num_heads, seq_len, seq_len)
            elif attn_mask.dim() == 3:
                pass
            elif attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
            
            attn_weights += attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        out = torch.bmm(attn_weights, v)
        
        out = out.reshape(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        
        return out
    
    def forward_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.kernel_type == "flash":
            return self._apply_flash_attention(q, k, v, attn_mask)
        elif self.kernel_type == "triton":
            return self._apply_triton_attention(q, k, v, attn_mask)
        else:
            return self._apply_pytorch_attention(q, k, v, attn_mask)