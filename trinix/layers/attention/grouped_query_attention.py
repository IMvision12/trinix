from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FastBaseAttention


class FastGroupedQueryAttention(FastBaseAttention):
    """Fast Grouped Query Attention (GQA) layer.

    GQA is a memory-efficient variant where multiple query heads share the same key-value heads.
    This reduces the KV cache size while maintaining most of the performance of standard MHA.
    Used in modern LLMs like LLaMA 2, Mistral, and others for efficient inference.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of query heads (must be divisible by num_kv_heads).
        num_kv_heads (int): Number of key-value heads. Each KV head is shared by
            num_heads // num_kv_heads query heads.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in projections. Defaults to True.
        kernel_type (str, optional): Attention backend ('flash', 'triton', 'pytorch'). Defaults to 'flash'.
        causal (bool, optional): Whether to apply causal masking. Defaults to False.
        head_dim (int, optional): Dimension per head. If None, uses embed_dim // num_heads. Defaults to None.
        position_method (str or nn.Module, optional): Position encoding method. Defaults to 'none'.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 2048.
        rope_base (float, optional): Base for RoPE frequencies. Defaults to 10000.0.
        use_sliding_window (bool, optional): Enable sliding window attention. Defaults to False.
        sliding_window_size (int, optional): Sliding window size. Defaults to None.
        qk_norm (bool, optional): Normalize queries and keys. Defaults to False.
        qk_norm_type (str, optional): Normalization type ('rmsnorm' or 'layernorm'). Defaults to 'rmsnorm'.
        use_triton_norm (bool, optional): Use Triton for normalization. Defaults to True.
        use_triton_embeddings (bool, optional): Use Triton for position embeddings. Defaults to True.

    Examples:
        >>> # GQA with 32 query heads and 8 KV heads (4x reduction in KV cache)
        >>> attn = FastGroupedQueryAttention(
        ...     embed_dim=4096, num_heads=32, num_kv_heads=8
        ... )
        >>> x = torch.randn(4, 128, 4096)
        >>> output, kv_cache = attn(x, use_cache=True)

        >>> # LLaMA 2 style GQA with RoPE
        >>> attn = FastGroupedQueryAttention(
        ...     embed_dim=4096, num_heads=32, num_kv_heads=8,
        ...     causal=True, position_method='rope'
        ... )

    Notes:
        - num_heads must be divisible by num_kv_heads
        - When num_kv_heads = num_heads, this is equivalent to standard MHA
        - When num_kv_heads = 1, this is equivalent to Multi-Query Attention (MQA)
        - Reduces KV cache size by factor of (num_heads / num_kv_heads)
    """

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
        use_sliding_window: bool = False,
        sliding_window_size: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_type: str = "rmsnorm",
        use_triton_norm: bool = True,
        use_triton_embeddings: bool = True,
    ):
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

        # Override head_dim if provided
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
                use_sliding_window,
                sliding_window_size,
                qk_norm,
                qk_norm_type,
                use_triton_norm,
                use_triton_embeddings,
                False,  # GQA doesn't use add_zero_attn
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
                use_sliding_window,
                sliding_window_size,
                qk_norm,
                qk_norm_type,
                use_triton_norm,
                use_triton_embeddings,
                False,  # GQA doesn't use add_zero_attn
            )

        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads

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

        q, k, position_bias = self._apply_position_embedding(
            q, k, seq_len, seq_len, bs, position_ids
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            seq_len = k.shape[-2]

        present_key_value = None
        if use_cache:
            present_key_value = (k, v)

        k = self._tile_kv_heads(k, self.num_key_value_groups)
        v = self._tile_kv_heads(v, self.num_key_value_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_sliding_window:
            attention_mask = self._apply_sliding_window_mask(
                attention_mask, seq_len, hidden_states.device
            )

        attention_mask = self._merge_position_bias(attention_mask, position_bias)

        out = self.forward_attention(q, k, v, attention_mask)
        out = out.reshape(bs, seq_len, -1)
        out = self.o_proj(out)

        return (out, present_key_value)
