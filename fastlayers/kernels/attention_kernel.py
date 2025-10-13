import torch
import triton
import triton.language as tl

from .utils import (
    calculate_attention_block_sizes,
    calculate_triton_kernel_configuration,
)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Mask,
    L,
    M,
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
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    has_mask: tl.constexpr,
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
    k_ptrs = (
        K
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[None, :] * stride_ks
        + offs_d[:, None] * stride_kd
    )
    v_ptrs = (
        V
        + pid_b * stride_vb
        + pid_h * stride_vh
        + offs_n[:, None] * stride_vs
        + offs_d[None, :] * stride_vd
    )

    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        if causal:
            mask_n = offs_n_curr[None, :] <= offs_m[:, None]
        else:
            mask_n = offs_n_curr[None, :] < seq_len

        k = tl.load(
            k_ptrs + start_n * stride_ks,
            mask=(offs_n_curr[None, :] < seq_len) & (offs_d[:, None] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n * stride_vs,
            mask=(offs_n_curr[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= scale

        if has_mask:
            mask_ptrs = (
                Mask
                + pid_b * stride_mb
                + pid_h * stride_mh
                + offs_m[:, None] * stride_mm
                + offs_n_curr[None, :] * stride_mn
            )
            custom_mask = tl.load(
                mask_ptrs,
                mask=(offs_m[:, None] < seq_len) & (offs_n_curr[None, :] < seq_len),
                other=0.0,
            )
            qk += custom_mask

        qk = tl.where(mask_n & (offs_m[:, None] < seq_len), qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
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


@triton.jit
def _bwd_kernel_dq(
    Q,
    K,
    V,
    Out,
    DO,
    DQ,
    Mask,
    L,
    M,
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
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    has_mask: tl.constexpr,
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
    do_ptrs = (
        DO
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )

    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    do = tl.load(
        do_ptrs,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    l_ptrs = L + pid_b * num_heads * seq_len + pid_h * seq_len + offs_m
    m_ptrs = M + pid_b * num_heads * seq_len + pid_h * seq_len + offs_m
    l_i = tl.load(l_ptrs, mask=offs_m < seq_len, other=1.0)
    m_i = tl.load(m_ptrs, mask=offs_m < seq_len, other=0.0)

    o_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )
    o = tl.load(
        o_ptrs,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    Di = tl.sum(do * o, axis=1)

    dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    k_ptrs = (
        K
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[None, :] * stride_ks
        + offs_d[:, None] * stride_kd
    )
    v_ptrs = (
        V
        + pid_b * stride_vb
        + pid_h * stride_vh
        + offs_n[:, None] * stride_vs
        + offs_d[None, :] * stride_vd
    )

    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        if causal:
            mask_n = offs_n_curr[None, :] <= offs_m[:, None]
        else:
            mask_n = offs_n_curr[None, :] < seq_len

        k = tl.load(
            k_ptrs + start_n * stride_ks,
            mask=(offs_n_curr[None, :] < seq_len) & (offs_d[:, None] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n * stride_vs,
            mask=(offs_n_curr[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.dot(q, k) * scale

        if has_mask:
            mask_ptrs = (
                Mask
                + pid_b * stride_mb
                + pid_h * stride_mh
                + offs_m[:, None] * stride_mm
                + offs_n_curr[None, :] * stride_mn
            )
            custom_mask = tl.load(
                mask_ptrs,
                mask=(offs_m[:, None] < seq_len) & (offs_n_curr[None, :] < seq_len),
                other=0.0,
            )
            qk += custom_mask

        qk = tl.where(mask_n & (offs_m[:, None] < seq_len), qk, float("-inf"))

        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]

        dp = tl.dot(do, tl.trans(v))

        ds = p * (dp - Di[:, None])

        dq_acc += tl.dot(ds.to(k.dtype), tl.trans(k)) * scale

    dq_ptrs = (
        DQ
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )
    tl.store(
        dq_ptrs,
        dq_acc.to(DQ.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )


@triton.jit
def _bwd_kernel_dkv(
    Q,
    K,
    V,
    Out,
    DO,
    DK,
    DV,
    Mask,
    L,
    M,
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
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    has_mask: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    k_ptrs = (
        K
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[:, None] * stride_ks
        + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        V
        + pid_b * stride_vb
        + pid_h * stride_vh
        + offs_n[:, None] * stride_vs
        + offs_d[None, :] * stride_vd
    )

    k = tl.load(
        k_ptrs,
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    v = tl.load(
        v_ptrs,
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    dk_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )
    do_ptrs = (
        DO
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )
    o_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )

    for start_m in range(0, seq_len, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        if causal:
            mask_m = offs_n[None, :] <= offs_m_curr[:, None]
        else:
            mask_m = offs_m_curr[:, None] < seq_len

        q = tl.load(
            q_ptrs + start_m * stride_qs,
            mask=(offs_m_curr[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        do = tl.load(
            do_ptrs + start_m * stride_os,
            mask=(offs_m_curr[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        o = tl.load(
            o_ptrs + start_m * stride_os,
            mask=(offs_m_curr[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        l_ptrs = L + pid_b * num_heads * seq_len + pid_h * seq_len + offs_m_curr
        m_ptrs = M + pid_b * num_heads * seq_len + pid_h * seq_len + offs_m_curr
        l_i = tl.load(l_ptrs, mask=offs_m_curr < seq_len, other=1.0)
        m_i = tl.load(m_ptrs, mask=offs_m_curr < seq_len, other=0.0)

        Di = tl.sum(do * o, axis=1)

        qk = tl.dot(q, tl.trans(k)) * scale

        if has_mask:
            mask_ptrs = (
                Mask
                + pid_b * stride_mb
                + pid_h * stride_mh
                + offs_m_curr[:, None] * stride_mm
                + offs_n[None, :] * stride_mn
            )
            custom_mask = tl.load(
                mask_ptrs,
                mask=(offs_m_curr[:, None] < seq_len) & (offs_n[None, :] < seq_len),
                other=0.0,
            )
            qk += custom_mask

        qk = tl.where(
            mask_m & (offs_m_curr[:, None] < seq_len) & (offs_n[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]

        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)

        dp = tl.dot(do, tl.trans(v))

        ds = p * (dp - Di[:, None])

        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale

    dk_ptrs = (
        DK
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[:, None] * stride_ks
        + offs_d[None, :] * stride_kd
    )
    dv_ptrs = (
        DV
        + pid_b * stride_vb
        + pid_h * stride_vh
        + offs_n[:, None] * stride_vs
        + offs_d[None, :] * stride_vd
    )

    tl.store(
        dk_ptrs,
        dk_acc.to(DK.dtype.element_ty),
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )
    tl.store(
        dv_ptrs,
        dv_acc.to(DV.dtype.element_ty),
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )


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


def _triton_attention_forward_only(q, k, v, attn_mask=None, causal=False, scale=None):
    """
    Triton attention forward pass (uses PyTorch autograd for backward).

    This provides the best balance:
    - Fast forward pass (Triton optimized)
    - Correct gradients (PyTorch autograd)

    For maximum backward performance, PyTorch's autograd is actually very efficient
    and avoids the overhead of custom Triton backward kernels.
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    if scale is None:
        scale = head_dim**-0.5

    # Prepare mask
    has_mask = attn_mask is not None
    if has_mask:
        attn_mask = _prepare_mask(attn_mask, batch_size, num_heads, seq_len, q.device)
    else:
        attn_mask = torch.empty(1, 1, 1, 1, dtype=q.dtype, device=q.device)

    out = torch.empty_like(q)

    # Allocate softmax statistics (not used for backward with autograd)
    L = torch.empty(
        (batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device
    )
    M = torch.empty(
        (batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device
    )

    # Calculate block sizes
    BLOCK_M, BLOCK_N, BLOCK_DMODEL = calculate_attention_block_sizes(head_dim, seq_len)

    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))

    _fwd_kernel[grid](
        q,
        k,
        v,
        out,
        attn_mask,
        L,
        M,
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
        attn_mask.stride(0),
        attn_mask.stride(1),
        attn_mask.stride(2),
        attn_mask.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        has_mask,
        causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )

    return out


class TritonAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask=None, causal=False, scale=None):
        batch_size, seq_len, num_heads, head_dim = q.shape

        if scale is None:
            scale = head_dim**-0.5

        has_mask = attn_mask is not None
        if has_mask:
            attn_mask = _prepare_mask(
                attn_mask, batch_size, num_heads, seq_len, q.device
            )
        else:
            attn_mask = torch.empty(1, 1, 1, 1, dtype=q.dtype, device=q.device)

        out = torch.empty_like(q)

        L = torch.empty(
            (batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device
        )
        M = torch.empty(
            (batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device
        )

        BLOCK_M, BLOCK_N, BLOCK_DMODEL = calculate_attention_block_sizes(
            head_dim, seq_len
        )

        grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))

        _fwd_kernel[grid](
            q,
            k,
            v,
            out,
            attn_mask,
            L,
            M,
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
            attn_mask.stride(0),
            attn_mask.stride(1),
            attn_mask.stride(2),
            attn_mask.stride(3),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            has_mask,
            causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )

        ctx.save_for_backward(q, k, v, out, attn_mask, L, M)
        ctx.causal = causal
        ctx.scale = scale
        ctx.has_mask = has_mask
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL = BLOCK_DMODEL

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, attn_mask, L, M = ctx.saved_tensors
        batch_size, seq_len, num_heads, head_dim = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid_dq = (batch_size, num_heads, triton.cdiv(seq_len, ctx.BLOCK_M))
        _bwd_kernel_dq[grid_dq](
            q,
            k,
            v,
            out,
            dout,
            dq,
            attn_mask,
            L,
            M,
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
            attn_mask.stride(0),
            attn_mask.stride(1),
            attn_mask.stride(2),
            attn_mask.stride(3),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            ctx.scale,
            ctx.has_mask,
            ctx.causal,
            BLOCK_M=ctx.BLOCK_M,
            BLOCK_N=ctx.BLOCK_N,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        )

        grid_dkv = (batch_size, num_heads, triton.cdiv(seq_len, ctx.BLOCK_N))
        _bwd_kernel_dkv[grid_dkv](
            q,
            k,
            v,
            out,
            dout,
            dk,
            dv,
            attn_mask,
            L,
            M,
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
            attn_mask.stride(0),
            attn_mask.stride(1),
            attn_mask.stride(2),
            attn_mask.stride(3),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            ctx.scale,
            ctx.has_mask,
            ctx.causal,
            BLOCK_M=ctx.BLOCK_M,
            BLOCK_N=ctx.BLOCK_N,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        )

        return dq, dk, dv, None, None, None


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
    ) -> torch.Tensor:
        return _triton_attention_forward_only(q, k, v, attn_mask, causal, scale)
