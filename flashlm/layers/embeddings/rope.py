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

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.use_triton
            and TRITON_AVAILABLE
            and (TritonRoPEKernel is not None)
            and TritonRoPEKernel.is_available()
            and q.is_cuda
            and (position_ids is None)
        ):
            return TritonRoPEKernel.apply(q, k, cos, sin)
        elif position_ids is not None:
            return self._apply_rope_with_position_ids(q, k, cos, sin, position_ids)
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
