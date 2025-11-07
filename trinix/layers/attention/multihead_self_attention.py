from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FastBaseAttention


class FastMultiHeadSelfAttention(FastBaseAttention):
    """Fast Multi-Head Self-Attention layer with fused QKV projection.

    Optimized self-attention layer that computes Q, K, V from the same input using a single
    fused projection. This is more efficient than separate projections and is commonly used
    in transformer encoder blocks and decoder-only models.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
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
        add_zero_attn (bool, optional): Add zero attention token. Defaults to False.
        batch_first (bool, optional): If True, input shape is (batch, seq, feature). Defaults to True.

    Examples:
        >>> # Standard self-attention for transformer encoder
        >>> attn = FastMultiHeadSelfAttention(embed_dim=768, num_heads=12)
        >>> x = torch.randn(4, 128, 768)
        >>> output = attn(x)

        >>> # Causal self-attention for GPT-style decoder
        >>> attn = FastMultiHeadSelfAttention(
        ...     embed_dim=768, num_heads=12, causal=True,
        ...     position_method='rope'
        ... )
        >>> x = torch.randn(4, 128, 768)
        >>> output, kv_cache = attn(x, use_cache=True)

    Notes:
        - Uses fused QKV projection for efficiency (single matrix multiply)
        - Only supports self-attention (query, key, value from same input)
        - For cross-attention, use FastMultiHeadAttention instead
        - Commonly used in BERT, GPT, and other transformer models
    """

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
        use_sliding_window: bool = False,
        sliding_window_size: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_type: str = "rmsnorm",
        use_triton_norm: bool = True,
        use_triton_embeddings: bool = True,
        add_zero_attn: bool = False,
        batch_first: bool = True,
    ):
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
                add_zero_attn,
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
                add_zero_attn,
            )

        self.batch_first = batch_first

        self.qkv_proj = nn.Linear(embed_dim, 3 * num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0.0)
        if self.o_proj.bias is not None:
            nn.init.constant_(self.o_proj.bias, 0.0)

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

        k, v, attention_mask = self._add_zero_attention(k, v, attention_mask)

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
        if not self.batch_first:
            out = out.transpose(0, 1)
        present_key_value = None
        if use_cache:
            present_key_value = (k.transpose(1, 2), v.transpose(1, 2))
        if use_cache:
            return (out, present_key_value)
        else:
            return out
