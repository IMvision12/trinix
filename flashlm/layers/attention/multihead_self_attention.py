from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings import (
    FastALiBiPositionEmbedding,
    FastRelativePositionEmbedding,
    FastRoPEPositionEmbedding,
)
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
        head_dim: Optional[int] = None,
        position_method: Union[str, nn.Module] = "none",
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        max_relative_position: int = 128,
        use_sliding_window: bool = False,
        sliding_window_size: Optional[int] = None,
        qk_layer_norm: bool = False,
        use_triton_embeddings: bool = True,
        add_zero_attn: bool = False,
        batch_first: bool = True,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, kernel_type, causal)
        assert num_heads > 0, f"num_heads must be positive, got {num_heads}"
        if isinstance(position_method, str):
            assert position_method in [
                "rope",
                "alibi",
                "relative",
                "none",
            ] or position_method.startswith("custom"), (
                f"Invalid position_method: {position_method}. Use 'rope', 'alibi', 'relative', 'none', or provide a custom nn.Module."
            )
        elif isinstance(position_method, nn.Module):
            pass
        else:
            raise TypeError(
                f"position_method must be a string or nn.Module, got {type(position_method)}"
            )
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.scaling = self.head_dim ** (-0.5)
        self.position_method = position_method
        self.max_seq_len = max_seq_len
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.qk_layer_norm = qk_layer_norm
        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first
        self.qkv_proj = nn.Linear(embed_dim, 3 * num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        if isinstance(position_method, nn.Module):
            self.position_embedding = position_method
            self.position_method = "custom"
        elif position_method == "rope":
            self.position_embedding = FastRoPEPositionEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_seq_len,
                base=rope_base,
                use_triton=use_triton_embeddings,
            )
        elif position_method == "alibi":
            self.position_embedding = FastALiBiPositionEmbedding(
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                use_triton=use_triton_embeddings,
            )
        elif position_method == "relative":
            self.position_embedding = FastRelativePositionEmbedding(
                num_heads=num_heads,
                head_dim=self.head_dim,
                max_relative_position=max_relative_position,
                use_triton=use_triton_embeddings,
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
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0.0)
        if self.o_proj.bias is not None:
            nn.init.constant_(self.o_proj.bias, 0.0)

    def _apply_sliding_window_mask(
        self, attention_mask: Optional[torch.Tensor], seq_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if not self.use_sliding_window or self.sliding_window_size is None:
            return attention_mask
        window_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=-self.sliding_window_size,
        )
        window_mask = torch.tril(window_mask, diagonal=0)
        window_mask = (1 - window_mask) * float("-inf")
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
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        position_bias = None
        if self.position_method == "rope" and self.position_embedding is not None:
            cos, sin = self.position_embedding(q, seq_len)
            q, k = self.position_embedding.apply_rotary_pos_emb(
                q, k, cos, sin, position_ids
            )
        elif self.position_method == "alibi" and self.position_embedding is not None:
            position_bias = self.position_embedding(seq_len, batch_size)
        elif self.position_method == "relative" and self.position_embedding is not None:
            position_bias = self.position_embedding(seq_len, batch_size)
        elif self.position_method == "custom" and self.position_embedding is not None:
            try:
                result = self.position_embedding(
                    q, k, seq_len, batch_size, position_ids
                )
                if isinstance(result, tuple) and len(result) == 3:
                    q, k, position_bias = result
                elif isinstance(result, tuple) and len(result) == 2:
                    q, k = result
                else:
                    position_bias = result
            except TypeError:
                try:
                    result = self.position_embedding(q, k)
                    if isinstance(result, tuple):
                        q, k = result
                    else:
                        position_bias = result
                except:
                    position_bias = self.position_embedding()
        return (q, k, position_bias)

    def _add_zero_attention(
        self, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.add_zero_attn:
            return (k, v, attention_mask)
        bs, num_heads, seq_len, head_dim = k.shape
        zero_k = torch.zeros(bs, num_heads, 1, head_dim, dtype=k.dtype, device=k.device)
        zero_v = torch.zeros(bs, num_heads, 1, head_dim, dtype=v.dtype, device=v.device)
        k = torch.cat([k, zero_k], dim=2)
        v = torch.cat([v, zero_v], dim=2)
        if attention_mask is not None:
            zero_mask = torch.zeros(
                bs,
                num_heads,
                attention_mask.shape[2],
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, zero_mask], dim=-1)
        return (k, v, attention_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]
    ]:
        if not self.batch_first:
            hidden_states = hidden_states.transpose(0, 1)
        bs, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bs, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, position_bias = self._apply_position_embedding(
            q, k, seq_len, bs, position_ids
        )
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
        k, v, attention_mask = self._add_zero_attention(k, v, attention_mask)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_sliding_window:
            attention_mask = self._apply_sliding_window_mask(
                attention_mask, seq_len, hidden_states.device
            )
        if position_bias is not None:
            if attention_mask is not None:
                attention_mask = attention_mask + position_bias
            else:
                attention_mask = position_bias
        if self.kernel_type == "flash" and attention_mask is not None:
            original_kernel = self.kernel_type
            self.kernel_type = "pytorch"
            out = self.forward_attention(q, k, v, attention_mask)
            self.kernel_type = original_kernel
        else:
            out = self.forward_attention(q, k, v, attention_mask)
        out = out.reshape(bs, seq_len, -1)
        out = self.o_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        present_key_value = None
        if use_cache:
            present_key_value = (k.transpose(1, 2), v.transpose(1, 2))
        if use_cache:
            return (out, present_key_value)
        else:
            return out
