import torch
import triton
import triton.language as tl

from .utils import calculate_attention_block_sizes


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    ALiBi,
    L,
    M,
    philox_seed,
    philox_offset,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_ab,
    stride_ah,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    dropout_p,
    window_left,
    window_right,
    has_alibi: tl.constexpr,
    has_dropout: tl.constexpr,
    has_window: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    if has_alibi:
        alibi_slope = tl.load(ALiBi + pid_h * stride_ah)
    else:
        alibi_slope = 0.0

    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        if causal:
            mask_n = offs_n_curr[None, :] <= offs_m[:, None]
        else:
            mask_n = offs_n_curr[None, :] < seq_len

        if has_window:
            pos_diff = offs_n_curr[None, :] - offs_m[:, None]
            window_mask = (pos_diff >= -window_left) & (pos_diff <= window_right)
            mask_n = mask_n & window_mask

        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n_curr[None, :] * stride_ks
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + offs_n_curr[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(
            k_ptrs,
            mask=(offs_n_curr[None, :] < seq_len) & (offs_d[:, None] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_ptrs,
            mask=(offs_n_curr[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= scale

        if has_alibi:
            pos_diff = tl.abs(offs_m[:, None] - offs_n_curr[None, :])
            alibi_bias = -alibi_slope * pos_diff.to(tl.float32)
            qk += alibi_bias

        qk = tl.where(mask_n & (offs_m[:, None] < seq_len), qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])

        if has_dropout:
            philox_offset_curr = philox_offset + pid_b * num_heads * seq_len * seq_len
            philox_offset_curr += pid_h * seq_len * seq_len
            philox_offset_curr += pid_m * BLOCK_M * seq_len + start_n

            rand = tl.rand(
                philox_seed,
                philox_offset_curr + offs_m[:, None] * seq_len + offs_n_curr[None, :],
            )
            keep_mask = rand > dropout_p
            p = tl.where(keep_mask, p / (1.0 - dropout_p), 0.0)

        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]

    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )
    tl.store(
        out_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )

    l_ptrs = L + pid_b * num_heads * seq_len + pid_h * seq_len + offs_m
    m_ptrs = M + pid_b * num_heads * seq_len + pid_h * seq_len + offs_m
    tl.store(l_ptrs, l_i, mask=offs_m < seq_len)
    tl.store(m_ptrs, m_i, mask=offs_m < seq_len)


def _prepare_mask(attn_mask, batch_size, num_heads, seq_len, device):
    if attn_mask is None:
        return None

    if attn_mask.dim() == 2:
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        attn_mask = attn_mask.expand(batch_size, num_heads, -1, -1)
    elif attn_mask.dim() == 3:
        attn_mask = attn_mask.unsqueeze(1)
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)
    elif attn_mask.dim() == 4:
        if attn_mask.shape[1] == 1:
            attn_mask = attn_mask.expand(-1, num_heads, -1, -1)
    else:
        raise ValueError(f"Unsupported mask shape: {attn_mask.shape}")

    attn_mask = attn_mask.contiguous()
    return attn_mask


class TritonAttentionKernel:
    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def apply(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        causal: bool = False,
        scale: float = None,
        dropout_p: float = 0.0,
        window_size: tuple = None,
        alibi_slopes: torch.Tensor = None,
    ) -> torch.Tensor:
        assert q.is_cuda, "Triton kernels require CUDA tensors"
        assert q.shape == k.shape == v.shape

        batch_size, seq_len, num_heads, head_dim = q.shape

        if scale is None:
            scale = head_dim**-0.5

        out = torch.empty_like(q)

        L = torch.empty(
            (batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device
        )
        M = torch.empty(
            (batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device
        )

        has_dropout = dropout_p > 0.0
        if has_dropout:
            philox_seed = torch.randint(0, 2**31 - 1, (1,), device=q.device).item()
            philox_offset = 0
        else:
            philox_seed = 0
            philox_offset = 0

        has_window = window_size is not None
        if has_window:
            window_left, window_right = window_size
        else:
            window_left, window_right = 0, 0

        has_alibi = alibi_slopes is not None
        if has_alibi:
            if alibi_slopes.dim() == 1:
                alibi_slopes = alibi_slopes.unsqueeze(0)
            assert alibi_slopes.shape[-1] == num_heads
            alibi_slopes = alibi_slopes.contiguous()
        else:
            alibi_slopes = torch.zeros(1, 1, device=q.device, dtype=q.dtype)

        BLOCK_M, BLOCK_N, BLOCK_DMODEL = calculate_attention_block_sizes(
            head_dim, seq_len
        )

        grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))

        _fwd_kernel[grid](
            q,
            k,
            v,
            out,
            alibi_slopes,
            L,
            M,
            philox_seed,
            philox_offset,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            out.stride(0),
            out.stride(2),
            out.stride(1),
            out.stride(3),
            alibi_slopes.stride(0) if has_alibi else 0,
            alibi_slopes.stride(1) if has_alibi else 0,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            dropout_p,
            window_left,
            window_right,
            has_alibi,
            has_dropout,
            has_window,
            causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )

        return out
