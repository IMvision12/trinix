from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FastBaseAttention


class FastMultiHeadLatentAttention(FastBaseAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kernel_type: str = "flash",
        causal: bool = False,
        latent_dim: Optional[int] = None,
        position_method: Union[str, nn.Module] = "none",
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        max_relative_position: int = 128,
        use_sliding_window: bool = False,
        sliding_window_size: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_type: str = "rmsnorm",
        use_triton_norm: bool = True,
        use_triton_embeddings: bool = True,
        add_zero_attn: bool = False,
        batch_first: bool = True,
    ):
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
            qk_norm,
            qk_norm_type,
            use_triton_norm,
            use_triton_embeddings,
            add_zero_attn,
        )

        self.batch_first = batch_first
        self.latent_dim = (
            latent_dim if latent_dim is not None else max(16, embed_dim // 8)
        )

        self.q_down_proj = nn.Linear(embed_dim, self.latent_dim, bias=bias)
        self.q_up_proj = nn.Linear(
            self.latent_dim, num_heads * self.head_dim, bias=bias
        )

        self.kv_down_proj = nn.Linear(embed_dim, self.latent_dim, bias=bias)

        self.k_up_proj = nn.Linear(
            self.latent_dim, num_heads * self.head_dim, bias=bias
        )
        self.v_up_proj = nn.Linear(
            self.latent_dim, num_heads * self.head_dim, bias=bias
        )

        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.register_buffer("cache_latent_kv", None, persistent=False)
        self.cache_position = 0

        self._init_weights()

    def _init_weights(self):
        projs = [
            self.q_down_proj,
            self.q_up_proj,
            self.kv_down_proj,
            self.k_up_proj,
            self.v_up_proj,
            self.o_proj,
        ]
        for proj in projs:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)

    def reset_cache(self):
        self.cache_latent_kv = None
        self.cache_position = 0

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.batch_first:
            query = query.transpose(0, 1)
            if key is not None:
                key = key.transpose(0, 1)
            if value is not None:
                value = value.transpose(0, 1)

        if key is None:
            key = query
        if value is None:
            value = key

        bs, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        q_latent = self.q_down_proj(query)
        q = self.q_up_proj(q_latent)

        latent_new = self.kv_down_proj(key)

        if use_cache:
            if self.cache_latent_kv is None:
                latent_total = latent_new
            else:
                latent_total = torch.cat([self.cache_latent_kv, latent_new], dim=1)
            self.cache_latent_kv = latent_total
        else:
            latent_total = latent_new
            self.cache_position = 0

        seq_len_latent = latent_total.shape[1]

        k = self.k_up_proj(latent_total)
        v = self.v_up_proj(latent_total)

        q = q.view(bs, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len_latent, self.num_heads, self.head_dim)
        v = v.view(bs, seq_len_latent, self.num_heads, self.head_dim)

        q, k, position_bias = self._apply_position_embedding(
            q, k, seq_len_q, seq_len_latent, bs, position_ids
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k, v, attention_mask = self._add_zero_attention(k, v, attention_mask)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_sliding_window:
            attention_mask = self._apply_sliding_window_mask(
                attention_mask, seq_len_latent, query.device
            )

        if self.causal and use_cache:
            q_positions = torch.arange(
                self.cache_position,
                self.cache_position + seq_len_q,
                device=query.device,
                dtype=torch.long,
            )
            k_positions = torch.arange(
                seq_len_latent, device=query.device, dtype=torch.long
            )
            causal_mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)
            causal_mask = (
                causal_mask.unsqueeze(0).unsqueeze(0).expand(bs, self.num_heads, -1, -1)
            )
            causal_mask = causal_mask.float() * float("-inf")

            if attention_mask is not None:
                attention_mask = attention_mask + causal_mask
            else:
                attention_mask = causal_mask

            self.cache_position += seq_len_q

        attention_mask = self._merge_position_bias(attention_mask, position_bias)

        out = self.forward_attention(q, k, v, attention_mask)

        out = out.reshape(bs, seq_len_q, -1)
        out = self.o_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        if use_cache:
            return (out, self.cache_latent_kv)
        else:
            return out
