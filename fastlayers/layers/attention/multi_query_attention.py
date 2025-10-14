from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if head_dim is not None:
            temp_embed = num_heads * head_dim
            super().__init__(
                temp_embed,
                num_heads,
                dropout,
                bias,
                kernel_type,
                causal,
                position_method,
                max_seq_len,
                rope_base,
                max_relative_position,
                use_sliding_window,
                sliding_window_size,
                qk_layer_norm,
                use_triton_embeddings,
                False,
            )
            self.head_dim = head_dim
        else:
            super().__init__(
                embed_dim,
                num_heads,
                dropout,
                bias,
                kernel_type,
                causal,
                position_method,
                max_seq_len,
                rope_base,
                max_relative_position,
                use_sliding_window,
                sliding_window_size,
                qk_layer_norm,
                use_triton_embeddings,
                False,
            )

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)  # Single head!
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)  # Single head!
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)

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
        bs, seq_len_q, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = q.view(bs, seq_len_q, self.num_heads, self.head_dim)

        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        k = k.view(bs, seq_len_q, 1, self.head_dim)
        v = v.view(bs, seq_len_q, 1, self.head_dim)

        q, k, position_bias = self._apply_position_embedding(
            q, k, seq_len_q, seq_len_q, bs, position_ids
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_layer_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        seq_len_kv = seq_len_q
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            seq_len_kv = k.shape[-2]

        present_key_value = None
        if use_cache:
            present_key_value = (k, v)

        k = k.expand(bs, self.num_heads, seq_len_kv, self.head_dim)
        v = v.expand(bs, self.num_heads, seq_len_kv, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_sliding_window:
            attention_mask = self._apply_sliding_window_mask(
                attention_mask, seq_len_kv, hidden_states.device
            )

        attention_mask = self._merge_position_bias(attention_mask, position_bias)

        out = self.forward_attention(q, k, v, attention_mask)

        out = out.reshape(bs, seq_len_q, -1)
        out = self.o_proj(out)

        if use_cache:
            return (out, present_key_value)
        else:
            return out
