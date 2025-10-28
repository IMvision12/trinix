from typing import Optional, Tuple

import torch

from ...kernels.attention_kernel import TritonAttentionKernel


def triton_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not q.is_cuda:
        raise ValueError("triton_attn_func only supports CUDA tensors")
    
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"Expected 4D tensors (batch, seqlen, nheads, headdim), "
            f"got q: {q.shape}, k: {k.shape}, v: {v.shape}"
        )
    
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, _, _ = k.shape
    
    if k.shape != (batch, seqlen_k, nheads, headdim):
        raise ValueError(f"k shape {k.shape} doesn't match expected shape")
    
    if v.shape != (batch, seqlen_k, nheads, headdim):
        raise ValueError(f"v shape {v.shape} doesn't match expected shape")
    
    if not TritonAttentionKernel.is_available():
        raise RuntimeError(
            "Triton attention kernel not available. "
            "Make sure Triton is installed and CUDA is available."
        )
    
    if softmax_scale is None:
        softmax_scale = headdim ** -0.5
    
    out = TritonAttentionKernel.apply(
        q,
        k,
        v,
        attn_mask=None,
        causal=causal,
        scale=softmax_scale,
        dropout_p=dropout_p,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )
    
    return out