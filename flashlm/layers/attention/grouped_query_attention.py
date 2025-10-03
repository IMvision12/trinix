import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .base import FastBaseAttention
from ..embeddings import FastRoPEPositionEmbedding, FastALiBiPositionEmbedding, FastRelativePositionEmbedding


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
        position_method: Union[str, nn.Module] = "none",
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        max_relative_position: int = 128,
        use_sliding_window: bool = False,
        sliding_window_size: Optional[int] = None,
        qk_layer_norm: bool = False,
        use_triton_embeddings: bool = True,

    ):
        super().__init__(embed_dim, num_heads, dropout, bias, kernel_type, causal)
        
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        if isinstance(position_method, str):
            assert position_method in ["rope", "alibi", "relative", "none"] or position_method.startswith("custom"), f"Invalid position_method: {position_method}. Use 'rope', 'alibi', 'relative', 'none', or provide a custom nn.Module."
        elif isinstance(position_method, nn.Module):
            pass
        else:
            raise TypeError(f"position_method must be a string or nn.Module, got {type(position_method)}")
        
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.position_method = position_method
        self.max_seq_len = max_seq_len
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.qk_layer_norm = qk_layer_norm
        
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        
        if isinstance(position_method, nn.Module):
            self.position_embedding = position_method
            self.position_method = "custom"
        elif position_method == "rope":
            self.position_embedding = FastRoPEPositionEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_seq_len,
                base=rope_base,
                use_triton=use_triton_embeddings
            )
        elif position_method == "alibi":
            self.position_embedding = FastALiBiPositionEmbedding(
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                use_triton=use_triton_embeddings
            )
        elif position_method == "relative":
            self.position_embedding = FastRelativePositionEmbedding(
                num_heads=num_heads,
                head_dim=self.head_dim,
                max_relative_position=max_relative_position,
                use_triton=use_triton_embeddings
            )
        else:
            self.position_embedding = None
        
        if qk_layer_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        else:
            self.q_norm = None
            self.k_norm = None
        
        self._init_weights()
    
    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)
    
    def _apply_sliding_window_mask(self, attention_mask: Optional[torch.Tensor], seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_sliding_window or self.sliding_window_size is None:
            return attention_mask
        
        window_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=-self.sliding_window_size
        )
        window_mask = torch.tril(window_mask, diagonal=0)
        
        window_mask = (1 - window_mask) * float('-inf')
        
        if attention_mask is not None:
            attention_mask = attention_mask + window_mask
        else:
            attention_mask = window_mask
            
        return attention_mask
    
    def _apply_position_embedding(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        seq_len: int,
        batch_size: int,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        position_bias = None
        
        if self.position_method == "rope" and self.position_embedding is not None:
            cos, sin = self.position_embedding(q, seq_len)
            q, k = self.position_embedding.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.position_method == "alibi" and self.position_embedding is not None:
            position_bias = self.position_embedding(seq_len, batch_size)
        elif self.position_method == "relative" and self.position_embedding is not None:
            position_bias = self.position_embedding(seq_len, batch_size)
        elif self.position_method == "custom" and self.position_embedding is not None:
            # For custom position embeddings, try to call with standard parameters
            # The custom implementation should handle the specific interface
            try:
                # Try the standard interface first (returns modified q, k and optional bias)
                result = self.position_embedding(q, k, seq_len, batch_size, position_ids)
                if isinstance(result, tuple) and len(result) == 3:
                    q, k, position_bias = result
                elif isinstance(result, tuple) and len(result) == 2:
                    q, k = result
                else:
                    # If single return value, assume it's position_bias
                    position_bias = result
            except TypeError:
                # Fallback: try with just q, k
                try:
                    result = self.position_embedding(q, k)
                    if isinstance(result, tuple):
                        q, k = result
                    else:
                        position_bias = result
                except:
                    # Last resort: call with no args and assume bias return
                    position_bias = self.position_embedding()
        
        return q, k, position_bias
    
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
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bs, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(bs, seq_len, self.num_kv_heads, self.head_dim)
        
        q, k, position_bias = self._apply_position_embedding(q, k, seq_len, bs, position_ids)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.qk_layer_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            seq_len = k.shape[-2]
        
        k = self._tile_kv_heads(k, self.num_key_value_groups)
        v = self._tile_kv_heads(v, self.num_key_value_groups)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_sliding_window:
            attention_mask = self._apply_sliding_window_mask(attention_mask, seq_len, hidden_states.device)
        
        if position_bias is not None:
            if attention_mask is not None:
                attention_mask = attention_mask + position_bias
            else:
                attention_mask = position_bias
        
        if attention_mask is not None and self.kernel_type == "flash":
            original_kernel = self.kernel_type
            self.kernel_type = "pytorch"
            out = self.forward_attention(q, k, v, attention_mask)
            self.kernel_type = original_kernel
        else:
            out = self.forward_attention(q, k, v, attention_mask)
        out = out.reshape(bs, seq_len, -1)
        out = self.o_proj(out)
        
        present_key_value = None
        if use_cache:
            k_cache = k.transpose(1, 2)[:, :self.num_kv_heads]
            v_cache = v.transpose(1, 2)[:, :self.num_kv_heads]
            present_key_value = (k_cache, v_cache)
        
        return out, present_key_value