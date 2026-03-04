import torch
import triton
import triton.language as tl

from ..utils import calculate_triton_kernel_configuration


@triton.jit
def fused_residual_layernorm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    SumOut_ptr,
    SumOut_row_stride,
    X_ptr,
    X_row_stride,
    Residual_ptr,
    Residual_row_stride,
    W_ptr,
    b_ptr,
    rstd_ptr,
    mean_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Residual + Layer Normalization forward pass kernel.

    Computes residual addition and layer normalization in a single kernel:
        SumOut = X + Residual
        Y = ((SumOut - mean(SumOut)) / sqrt(var(SumOut) + eps)) * W + b

    Args:
        Y_ptr: Output tensor pointer (normalized result)
        Y_row_stride: Stride between rows in output
        SumOut_ptr: Intermediate sum output pointer (saved for backward)
        SumOut_row_stride: Stride between rows in SumOut
        X_ptr: Input tensor pointer
        X_row_stride: Stride between rows in input
        Residual_ptr: Residual tensor pointer
        Residual_row_stride: Stride between rows in residual
        W_ptr: Weight tensor pointer (learned scale)
        b_ptr: Bias tensor pointer (learned shift)
        rstd_ptr: Reciprocal standard deviation output pointer (for backward)
        mean_ptr: Mean output pointer (for backward)
        n_cols: Number of columns (features) per row
        eps: Small constant for numerical stability
        BLOCK_SIZE: Triton block size for parallel processing

    Grid: (n_rows,) - one program per row
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_idx * X_row_stride
    Residual_ptr += row_idx * Residual_row_stride
    Y_ptr += row_idx * Y_row_stride
    SumOut_ptr += row_idx * SumOut_row_stride
    rstd_ptr += row_idx
    mean_ptr += row_idx

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    Res_row = tl.load(Residual_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused: add residual then normalize
    sum_out = X_row + Res_row
    tl.store(SumOut_ptr + col_offsets, sum_out, mask=mask)

    mean_val = tl.sum(sum_out, axis=0) / n_cols
    centered = tl.where(mask, sum_out - mean_val, 0.0)
    row_var = tl.sum(centered * centered, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)

    tl.store(rstd_ptr, inv_var)
    tl.store(mean_ptr, mean_val)
    output = centered * inv_var * W_row + b_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)


@triton.jit
def fused_residual_layernorm_backward_dx_fused(
    DX_ptr,
    DResidual_ptr,
    DY_ptr,
    DW_ptr,
    DB_ptr,
    SumOut_ptr,
    W_ptr,
    Mean_ptr,
    Rstd_ptr,
    Lock_ptr,
    stride: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused Residual + Layer Normalization backward pass kernel.

    Computes gradients for both inputs (dX, dResidual) and accumulates partial
    weight/bias gradients (dW, dB).

    Gradient formulas:
        dSumOut = (dY * W - mean(dY * W * SumOut_hat) * SumOut_hat - mean(dY * W)) * rstd
        dX = dSumOut
        dResidual = dSumOut
        dW = sum(dY * SumOut_hat) over all rows
        dB = sum(dY) over all rows

    where SumOut_hat = (SumOut - mean) * rstd

    Args:
        DX_ptr: Output gradient w.r.t. X pointer
        DResidual_ptr: Output gradient w.r.t. Residual pointer
        DY_ptr: Input gradient w.r.t. output pointer
        DW_ptr: Partial weight gradient accumulator pointer
        DB_ptr: Partial bias gradient accumulator pointer
        SumOut_ptr: Saved SumOut from forward pass
        W_ptr: Weight tensor pointer
        Mean_ptr: Row means from forward pass pointer
        Rstd_ptr: Reciprocal standard deviations from forward pass pointer
        Lock_ptr: Lock array for atomic accumulation
        stride: Row stride in tensors
        N: Number of columns (features)
        GROUP_SIZE_M: Number of lock groups
        BLOCK_SIZE_N: Block size for column processing

    Grid: (n_rows,) - one program per row
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    lock_id = row % GROUP_SIZE_M
    Lock = Lock_ptr + lock_id
    Count = Lock_ptr + GROUP_SIZE_M + lock_id
    DW = DW_ptr + lock_id * N + cols
    DB = DB_ptr + lock_id * N + cols

    sum_out = tl.load(SumOut_ptr + row * stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    dy = tl.load(DY_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row)
    rstd = tl.load(Rstd_ptr + row)

    xhat = (sum_out - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    d_sumout = (wdy - (xhat * c1 + c2)) * rstd

    # dX = dResidual = dSumOut
    tl.store(DX_ptr + row * stride + cols, d_sumout, mask=mask)
    tl.store(DResidual_ptr + row * stride + cols, d_sumout, mask=mask)

    partial_dw = dy * xhat
    partial_db = dy

    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass

    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask, other=0.0)
        partial_db += tl.load(DB, mask=mask, other=0.0)

    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    tl.atomic_xchg(Lock, 0)


@triton.jit
def fused_residual_layernorm_backward_dwdb(
    DW_ptr,
    DB_ptr,
    FINAL_DW_ptr,
    FINAL_DB_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused Residual + Layer Normalization backward - final dW/dB reduction.

    Reduces partial weight and bias gradients from lock groups into final gradients.

    Args:
        DW_ptr: Partial weight gradients pointer, shape (M, N)
        DB_ptr: Partial bias gradients pointer, shape (M, N)
        FINAL_DW_ptr: Final weight gradient output pointer, shape (N,)
        FINAL_DB_ptr: Final bias gradient output pointer, shape (N,)
        M: Number of lock groups
        N: Number of columns (features)
        BLOCK_SIZE_M: Block size for row reduction
        BLOCK_SIZE_N: Block size for column processing

    Grid: (cdiv(N, BLOCK_SIZE_N),) - one program per column block
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        db += tl.load(DB_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW_ptr + cols, sum_dw, mask=col_mask)
    tl.store(FINAL_DB_ptr + cols, sum_db, mask=col_mask)


class TritonFusedResidualLayerNormFunction(torch.autograd.Function):
    """Autograd function for fused Residual + Layer Normalization.

    Forward: SumOut = X + Residual; Y = LayerNorm(SumOut, W, b, eps)
    Backward: Two-stage parallel reduction, returns (dX, dResidual, dW, dB, None)
    """

    @staticmethod
    def forward(ctx, X, Residual, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.contiguous().view(-1, dim)
        Residual = Residual.contiguous().view(-1, dim)
        n_rows, n_cols = X.shape

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        SumOut = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=device)
        mean = torch.empty(n_rows, dtype=torch.float32, device=device)

        fused_residual_layernorm_forward_kernel[(n_rows,)](
            Y,
            Y.stride(0),
            SumOut,
            SumOut.stride(0),
            X,
            X.stride(0),
            Residual,
            Residual.stride(0),
            W,
            b,
            rstd,
            mean,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(SumOut, W, rstd, mean)

        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.contiguous().view(-1, dim)
        SumOut, W, rstd, mean = ctx.saved_tensors
        SumOut = SumOut.contiguous().view(-1, dim)
        n_rows, n_cols = dY.shape
        device = dY.device

        GROUP_SIZE_M = 64
        if n_cols <= 8192:
            GROUP_SIZE_M = 96
        if n_cols <= 4096:
            GROUP_SIZE_M = 128
        if n_cols <= 1024:
            GROUP_SIZE_M = 256

        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=device)
        _dw = torch.zeros((GROUP_SIZE_M, n_cols), dtype=dY.dtype, device=device)
        _db = torch.zeros((GROUP_SIZE_M, n_cols), dtype=dY.dtype, device=device)
        dW = torch.empty(n_cols, dtype=W.dtype, device=device)
        db = torch.empty(n_cols, dtype=W.dtype, device=device)
        dX = torch.empty_like(dY)
        dResidual = torch.empty_like(dY)

        fused_residual_layernorm_backward_dx_fused[(n_rows,)](
            dX,
            dResidual,
            dY,
            _dw,
            _db,
            SumOut,
            W,
            mean,
            rstd,
            locks,
            SumOut.stride(0),
            n_cols,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        def grid(meta):
            return (triton.cdiv(n_cols, meta["BLOCK_SIZE_N"]),)

        fused_residual_layernorm_backward_dwdb[grid](
            _dw,
            _db,
            dW,
            db,
            min(GROUP_SIZE_M, n_rows),
            n_cols,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
        )

        return (dX.view(*shape), dResidual.view(*shape), dW, db, None)


class TritonFusedResidualLayerNormKernel:
    """Triton-accelerated fused Residual + Layer Normalization kernel wrapper.

    Combines residual addition and layer normalization into a single kernel launch.

    Algorithm:
        Forward:  SumOut = X + Residual; Y = ((SumOut - mean) / sqrt(var + eps)) * W + b
        Backward: Two-stage parallel reduction with atomic locks

    Methods:
        is_available() -> bool:
            Check if Triton and CUDA are available.

        apply(X, Residual, W, b, eps) -> torch.Tensor:
            Apply fused residual addition + layer normalization.

            Args:
                X: Input tensor of any shape (*, N)
                Residual: Residual tensor, same shape as X
                W: Weight tensor, shape (N,)
                b: Bias tensor, shape (N,)
                eps: Small constant for numerical stability
    """

    @staticmethod
    def is_available() -> bool:
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def apply(
        X: torch.Tensor,
        Residual: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return TritonFusedResidualLayerNormFunction.apply(X, Residual, W, b, eps)
