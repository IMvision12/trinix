from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from ...kernels import TritonRoPEKernel

    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TRITON_AVAILABLE = False
    TritonRoPEKernel = None


class FastRoPEPositionEmbedding(nn.Module):
    """Fast RoPE (Rotary Position Embedding) layer.

    RoPE encodes position information by rotating pairs of features using position-dependent
    rotation matrices. This allows the model to naturally incorporate relative position information
    without adding extra parameters. Automatically uses Triton kernels for larger models (hidden_size > 2048)
    with sufficient sequence length (seq_len > 512), falling back to PyTorch otherwise.

    Args:
        dim (int): Dimension of the embeddings (typically head_dim).
        max_position_embeddings (int, optional): Maximum sequence length. Defaults to 2048.
        base (float, optional): Base for computing rotation frequencies. Defaults to 10000.0.
        use_triton (bool, optional): Whether to enable Triton kernels. Defaults to True.

    Shape:
        - forward output: Tuple of (cos, sin) tensors, each of shape (seq_len, dim)
        - apply_rotary_pos_emb output: Tuple of (rotated_q, rotated_k), each of shape
          (batch_size, seq_len, num_heads, head_dim)

    Examples:
        >>> rope = FastRoPEPositionEmbedding(dim=64)
        >>> q = torch.randn(4, 128, 8, 64)  # (batch, seq_len, num_heads, head_dim)
        >>> k = torch.randn(4, 128, 8, 64)
        >>> cos, sin = rope(q, seq_len=128)
        >>> q_rot, k_rot = rope.apply_rotary_pos_emb(q, k, cos, sin)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        use_triton: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.use_triton = use_triton and TRITON_AVAILABLE
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return (cos, sin)

    def _check_triton_availability(self, q: torch.Tensor) -> bool:
        """Check if Triton should be used based on model size and sequence length."""
        if not self.use_triton or not TRITON_AVAILABLE:
            return False
        if TritonRoPEKernel is None or not TritonRoPEKernel.is_available():
            return False
        if not q.is_cuda:
            return False
        # q shape: (batch, seq_len, num_heads, head_dim)
        # Triton is faster for larger models (hidden > 2048) with longer sequences (seq_len > 512)
        if q.dim() >= 3:
            seq_len = q.shape[1]
            num_heads = q.shape[2]
            head_dim = q.shape[3]
            hidden_size = num_heads * head_dim
            # Use Triton when BOTH conditions are met
            return hidden_size > 2048 and seq_len > 512
        return False

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is not None:
            return self._apply_rope_with_position_ids(q, k, cos, sin, position_ids)
        elif self._check_triton_availability(q):
            return TritonRoPEKernel.apply(q, k, cos, sin)
        else:
            return self._apply_rope_pytorch(q, k, cos, sin)

    def _apply_rope_pytorch(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin
        return (q_embed, k_embed)

    def _apply_rope_with_position_ids(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.squeeze(1).squeeze(0) if cos.dim() > 2 else cos
        sin = sin.squeeze(1).squeeze(0) if sin.dim() > 2 else sin
        cos = cos[position_ids].unsqueeze(2)
        sin = sin[position_ids].unsqueeze(2)
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin
        return (q_embed, k_embed)
