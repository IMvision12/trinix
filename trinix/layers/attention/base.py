import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm import FastLayerNorm, FastRMSNorm

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
    """Fast base attention layer with multiple backend and position encoding options.

    A flexible attention implementation that supports multiple computational backends
    (Flash Attention, Triton, PyTorch) and various position encoding methods (RoPE, ALiBi,
    Relative). Automatically selects the best available backend and provides extensive
    configuration options for different attention patterns.

    Args:
        embed_dim (int): Total dimension of the model (must be divisible by num_heads).
        num_heads (int): Number of parallel attention heads.
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in projections. Defaults to True.
        kernel_type (str, optional): Attention backend to use. Options: 'flash', 'triton', 'pytorch'.
            Defaults to 'flash'. Falls back to PyTorch if selected backend is unavailable.
        causal (bool, optional): Whether to apply causal masking (for autoregressive models).
            Defaults to False.
        position_method (str or nn.Module, optional): Position encoding method. Options:
            - 'rope': Rotary Position Embedding
            - 'alibi': Attention with Linear Biases
            - 'relative': Learned relative position embeddings
            - 'none': No position encoding
            - Custom nn.Module for custom position encoding
            Defaults to 'none'.
        max_seq_len (int, optional): Maximum sequence length for position encodings.
            Defaults to 2048.
        rope_base (float, optional): Base for RoPE frequency computation. Defaults to 10000.0.
        max_relative_position (int, optional): Maximum relative position for relative embeddings.
            Defaults to 128.
        use_sliding_window (bool, optional): Whether to use sliding window attention.
            Defaults to False.
        sliding_window_size (int, optional): Size of the sliding window. Required if
            use_sliding_window is True. Defaults to None.
        qk_norm (bool, optional): Whether to normalize queries and keys before attention.
            Helps stabilize training. Defaults to False.
        qk_norm_type (str, optional): Type of normalization for Q/K. Options: 'rmsnorm', 'layernorm'.
            Defaults to 'rmsnorm'.
        use_triton_norm (bool, optional): Whether to use Triton kernels for Q/K normalization.
            Defaults to True.
        use_triton_embeddings (bool, optional): Whether to use Triton kernels for position embeddings.
            Defaults to True.
        add_zero_attn (bool, optional): Whether to add a zero attention token (useful for padding).
            Defaults to False.

    Shape:
        - q: (batch_size, seq_len, num_heads, head_dim)
        - k: (batch_size, seq_len_k, num_heads, head_dim)
        - v: (batch_size, seq_len_k, num_heads, head_dim)
        - output: (batch_size, seq_len, num_heads, head_dim)

    Examples:
        >>> # Standard multi-head attention with Flash Attention
        >>> attn = FastBaseAttention(embed_dim=768, num_heads=12)
        >>> q = k = v = torch.randn(4, 128, 12, 64)  # (batch, seq_len, heads, head_dim)
        >>> output = attn.forward_attention(q, k, v)

        >>> # Causal attention with RoPE (for GPT-style models)
        >>> attn = FastBaseAttention(
        ...     embed_dim=768, num_heads=12, causal=True,
        ...     position_method='rope', kernel_type='flash'
        ... )

        >>> # Attention with ALiBi and sliding window
        >>> attn = FastBaseAttention(
        ...     embed_dim=768, num_heads=12,
        ...     position_method='alibi',
        ...     use_sliding_window=True,
        ...     sliding_window_size=256
        ... )

        >>> # With Q/K normalization for training stability
        >>> attn = FastBaseAttention(
        ...     embed_dim=768, num_heads=12,
        ...     qk_norm=True, qk_norm_type='rmsnorm'
        ... )

    Notes:
        - Flash Attention requires fp16 or bf16 tensors and doesn't support custom masks
        - Triton backend supports all features including custom masks and position encodings
        - PyTorch backend is the most compatible but may be slower
        - The layer automatically falls back to available backends if the requested one is unavailable
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kernel_type: str = "flash",
        causal: bool = False,
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
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert num_heads > 0, f"num_heads must be positive, got {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.kernel_type = kernel_type
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type
        self.add_zero_attn = add_zero_attn
        self.max_seq_len = max_seq_len
        self.scale = self.head_dim ** (-0.5)

        if kernel_type == "flash" and (not FLASH_ATTN_AVAILABLE):
            warnings.warn("Flash Attention not available, falling back to PyTorch")
            self.kernel_type = "pytorch"
        elif kernel_type == "triton" and (not TRITON_AVAILABLE):
            warnings.warn("Triton not available, falling back to PyTorch")
            self.kernel_type = "pytorch"

        self._validate_position_method(position_method)
        self.position_method = (
            position_method if isinstance(position_method, str) else "custom"
        )
        self._setup_position_embedding(
            position_method,
            max_seq_len,
            rope_base,
            max_relative_position,
            use_triton_embeddings,
        )

        if qk_norm:
            if qk_norm_type == "rmsnorm":
                self.q_norm = FastRMSNorm(self.head_dim, use_triton=use_triton_norm)
                self.k_norm = FastRMSNorm(self.head_dim, use_triton=use_triton_norm)
            elif qk_norm_type == "layernorm":
                self.q_norm = FastLayerNorm(self.head_dim, use_triton=use_triton_norm)
                self.k_norm = FastLayerNorm(self.head_dim, use_triton=use_triton_norm)
            else:
                raise ValueError(
                    f"Invalid qk_norm_type: {qk_norm_type}. Use 'rmsnorm' or 'layernorm'"
                )
        else:
            self.q_norm = None
            self.k_norm = None

    def _validate_position_method(self, position_method: Union[str, nn.Module]):
        if isinstance(position_method, str):
            assert position_method in [
                "rope",
                "alibi",
                "relative",
                "none",
            ] or position_method.startswith("custom"), (
                f"Invalid position_method: {position_method}. Use 'rope', 'alibi', 'relative', 'none', or provide a custom nn.Module."
            )
        elif not isinstance(position_method, nn.Module):
            raise TypeError(
                f"position_method must be a string or nn.Module, got {type(position_method)}"
            )

    def _setup_position_embedding(
        self,
        position_method: Union[str, nn.Module],
        max_seq_len: int,
        rope_base: float,
        max_relative_position: int,
        use_triton_embeddings: bool,
    ):
        from ..embeddings import (
            FastALiBiPositionEmbedding,
            FastRelativePositionEmbedding,
            FastRoPEPositionEmbedding,
        )

        if isinstance(position_method, nn.Module):
            self.position_embedding = position_method
        elif position_method == "rope":
            self.position_embedding = FastRoPEPositionEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_seq_len,
                base=rope_base,
                use_triton=use_triton_embeddings,
            )
        elif position_method == "alibi":
            self.position_embedding = FastALiBiPositionEmbedding(
                num_heads=self.num_heads,
                max_seq_len=max_seq_len,
                use_triton=use_triton_embeddings,
            )
        elif position_method == "relative":
            self.position_embedding = FastRelativePositionEmbedding(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_relative_position=max_relative_position,
                use_triton=use_triton_embeddings,
            )
        else:
            self.position_embedding = None

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
        seq_len_q: int,
        seq_len_k: Optional[int] = None,
        batch_size: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if seq_len_k is None:
            seq_len_k = seq_len_q
        if batch_size is None:
            batch_size = q.shape[0]

        position_bias = None
        max_seq_len = max(seq_len_q, seq_len_k)

        if self.position_method == "rope" and self.position_embedding is not None:
            cos, sin = self.position_embedding(q, max_seq_len)
            q, k = self.position_embedding.apply_rotary_pos_emb(
                q, k, cos, sin, position_ids
            )
        elif self.position_method == "alibi" and self.position_embedding is not None:
            position_bias = self.position_embedding(max_seq_len, batch_size)
            if seq_len_q != seq_len_k:
                position_bias = position_bias[:, :, :seq_len_q, :seq_len_k]
        elif self.position_method == "relative" and self.position_embedding is not None:
            position_bias = self.position_embedding(max_seq_len, batch_size)
            if seq_len_q != seq_len_k:
                position_bias = position_bias[:, :, :seq_len_q, :seq_len_k]
        elif self.position_method == "custom" and self.position_embedding is not None:
            try:
                result = self.position_embedding(
                    q, k, seq_len_q, seq_len_k, batch_size, position_ids
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

    def _merge_position_bias(
        self,
        attention_mask: Optional[torch.Tensor],
        position_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if position_bias is not None:
            if attention_mask is not None:
                attention_mask = attention_mask + position_bias
            else:
                attention_mask = position_bias
        return attention_mask

    def _apply_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not FLASH_ATTN_AVAILABLE:
            raise RuntimeError("Flash Attention not available")

        if q.dtype not in [torch.float16, torch.bfloat16]:
            raise TypeError(
                f"Flash Attention only supports fp16 and bf16 data types, got {q.dtype}. "
                f"Please convert your model to fp16 or bf16: model.half() or model.to(torch.bfloat16)"
            )

        batch_size, seq_len, num_heads, head_dim = q.shape
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=self.causal,
        )
        return out

    def _apply_pytorch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        seq_len_k = k.shape[1]

        q = (
            q.transpose(1, 2)
            .contiguous()
            .view(batch_size * num_heads, seq_len, head_dim)
        )
        k = (
            k.transpose(1, 2)
            .contiguous()
            .view(batch_size * num_heads, seq_len_k, head_dim)
        )
        v = (
            v.transpose(1, 2)
            .contiguous()
            .view(batch_size * num_heads, seq_len_k, head_dim)
        )

        attn_weights = torch.bmm(q, k.transpose(-2, -1)) * self.scale

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len_k, device=q.device), diagonal=1
            ).bool()
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dim() == 4:
                attn_mask = attn_mask.reshape(
                    batch_size * num_heads, seq_len, seq_len_k
                )
            elif attn_mask.dim() == 3:
                pass
            elif attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).expand(
                    batch_size * num_heads, -1, -1
                )
            attn_weights += attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.bmm(attn_weights, v)
        out = out.reshape(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

        return out

    def _apply_triton_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")

        from .functional import triton_attn_func

        try:
            alibi_slopes = None
            if self.position_method == "alibi" and self.position_embedding is not None:
                alibi_slopes = self.position_embedding.slopes

            window_size = None
            if self.use_sliding_window and self.sliding_window_size is not None:
                window_size = (self.sliding_window_size, self.sliding_window_size)

            out = triton_attn_func(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
            )
            return out
        except Exception as e:
            warnings.warn(
                f"Triton attention kernel failed: {e}, falling back to PyTorch"
            )
            return self._apply_pytorch_attention(q, k, v, attn_mask)

    def forward_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.kernel_type == "flash" and attn_mask is not None:
            warnings.warn(
                "Flash Attention doesn't support custom masks, falling back to PyTorch"
            )
            return self._apply_pytorch_attention(q, k, v, attn_mask)

        if self.kernel_type == "flash":
            return self._apply_flash_attention(q, k, v, attn_mask)
        elif self.kernel_type == "triton":
            return self._apply_triton_attention(q, k, v, attn_mask)
        else:
            return self._apply_pytorch_attention(q, k, v, attn_mask)
