import torch
import triton
import triton.language as tl

from ..utils import calculate_triton_kernel_configuration


@triton.jit
def fused_rmsnorm_activation_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm + Activation forward pass kernel.

    Computes RMS normalization followed by activation in a single kernel:
        normed = (X / sqrt(mean(X^2) + eps)) * W
        Y = activation(normed)

    ACTIVATION flag: 0=mish, 1=quickgelu, 2=squared_relu

    Args:
        Y_ptr: Output tensor pointer
        Y_row_stride: Stride between rows in output
        X_ptr: Input tensor pointer
        X_row_stride: Stride between rows in input
        W_ptr: Weight tensor pointer
        rstd_ptr: Reciprocal RMS output pointer (for backward)
        n_cols: Number of columns per row
        eps: Numerical stability constant
        ACTIVATION: Activation type (0=mish, 1=quickgelu, 2=squared_relu)
        BLOCK_SIZE: Triton block size

    Grid: (n_rows,) - one program per row
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    rstd_ptr += row_idx

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm
    X_squared = tl.where(mask, X_row * X_row, 0.0)
    variance = tl.sum(X_squared, axis=0) / n_cols
    rstd = tl.math.rsqrt(variance + eps)
    tl.store(rstd_ptr, rstd)

    normed = X_row * rstd * W_row

    # Activation
    if ACTIVATION == 0:
        # Mish: x * tanh(softplus(x))
        softplus = tl.log(1.0 + tl.exp(normed))
        exp_2sp = tl.exp(2.0 * softplus)
        tanh_sp = (exp_2sp - 1.0) / (exp_2sp + 1.0)
        output = normed * tanh_sp
    elif ACTIVATION == 1:
        # QuickGELU: x * sigmoid(1.702 * x)
        output = normed * tl.sigmoid(1.702 * normed)
    elif ACTIVATION == 2:
        # Squared ReLU: (max(0, x))^2
        relu_x = tl.maximum(normed, 0.0)
        output = relu_x * relu_x

    tl.store(Y_ptr + col_offsets, output, mask=mask)


@triton.jit
def fused_rmsnorm_activation_backward_dx_fused(
    DX_ptr,
    DY_ptr,
    DW_ptr,
    X_ptr,
    W_ptr,
    Rstd_ptr,
    Lock_ptr,
    stride: tl.constexpr,
    N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused RMSNorm + Activation backward pass kernel.

    Recomputes normed values from X, W, rstd to avoid saving extra tensor.
    Chains activation derivative into norm backward.

    Args:
        DX_ptr: Output gradient w.r.t. input pointer
        DY_ptr: Input gradient w.r.t. output pointer
        DW_ptr: Partial weight gradient accumulator pointer
        X_ptr: Original input tensor pointer
        W_ptr: Weight tensor pointer
        Rstd_ptr: Reciprocal RMS from forward pass pointer
        Lock_ptr: Lock array for atomic accumulation
        stride: Row stride in tensors
        N: Number of columns
        ACTIVATION: Activation type (0=mish, 1=quickgelu, 2=squared_relu)
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

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row)

    # Recompute normed values
    xhat = x * rstd
    normed = xhat * w

    # Compute activation derivative: d_activation/d_normed
    if ACTIVATION == 0:
        # Mish derivative: tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
        exp_n = tl.exp(normed)
        softplus = tl.log(1.0 + exp_n)
        exp_2sp = tl.exp(2.0 * softplus)
        tanh_sp = (exp_2sp - 1.0) / (exp_2sp + 1.0)
        sech2_sp = 1.0 - tanh_sp * tanh_sp
        sigmoid_n = exp_n / (1.0 + exp_n)
        d_act = tanh_sp + normed * sech2_sp * sigmoid_n
    elif ACTIVATION == 1:
        # QuickGELU derivative: sigmoid(1.702*x) + x * sigmoid'(1.702*x) * 1.702
        alpha = 1.702
        sig = tl.sigmoid(alpha * normed)
        dsig = sig * (1.0 - sig)
        d_act = sig + normed * dsig * alpha
    elif ACTIVATION == 2:
        # Squared ReLU derivative: 2 * max(0, x)
        relu_n = tl.maximum(normed, 0.0)
        d_act = 2.0 * relu_n

    # Chain: dY * d_activation/d_normed gives gradient w.r.t normed output
    # normed = xhat * w, so d_normed/d_xhat = w, d_normed/d_w = xhat
    dy_act = dy * d_act  # gradient w.r.t. normed
    wdy = dy_act * w  # gradient w.r.t. xhat (= dy_act * d_normed/d_xhat)

    wdy = tl.where(mask, wdy, 0.0)
    xhat = tl.where(mask, xhat, 0.0)

    # RMSNorm backward for dX
    c1 = tl.sum(wdy * xhat, axis=0) / N
    dx = (wdy - c1 * xhat) * rstd

    tl.store(DX_ptr + row * stride + cols, dx, mask=mask)

    # Partial dW: d_loss/d_w = dy_act * xhat (chain through activation and norm)
    partial_dw = dy_act * xhat

    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass

    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask, other=0.0)

    tl.store(DW, partial_dw, mask=mask)
    tl.atomic_xchg(Lock, 0)


@triton.jit
def fused_rmsnorm_activation_backward_dw(
    DW_ptr,
    FINAL_DW_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Final dW reduction for fused RMSNorm + Activation backward.

    Same two-stage pattern as standalone RMSNorm.

    Args:
        DW_ptr: Partial weight gradients pointer, shape (M, N)
        FINAL_DW_ptr: Final weight gradient output pointer, shape (N,)
        M: Number of lock groups
        N: Number of columns
        BLOCK_SIZE_M: Block size for row reduction
        BLOCK_SIZE_N: Block size for column processing

    Grid: (cdiv(N, BLOCK_SIZE_N),)
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sum_dw = tl.sum(dw, axis=0)
    mask = cols < N
    tl.store(FINAL_DW_ptr + cols, sum_dw, mask=mask)


# Activation type constants
ACTIVATION_MISH = 0
ACTIVATION_QUICKGELU = 1
ACTIVATION_SQUARED_RELU = 2

_ACTIVATION_MAP = {
    "mish": ACTIVATION_MISH,
    "quickgelu": ACTIVATION_QUICKGELU,
    "squared_relu": ACTIVATION_SQUARED_RELU,
}


class TritonFusedRMSNormActivationFunction(torch.autograd.Function):
    """Autograd function for fused RMSNorm + Activation.

    Forward: Y = activation(RMSNorm(X, W, eps))
    Backward: Chains activation derivative into norm backward.
              Recomputes normed values to avoid saving extra tensor.
    """

    @staticmethod
    def forward(ctx, X, W, eps, activation_type):
        shape = X.shape
        dim = shape[-1]
        X = X.contiguous().view(-1, dim)
        n_rows, n_cols = X.shape

        BLOCK_SIZE, num_warps = calculate_triton_kernel_configuration(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=device)

        fused_rmsnorm_activation_forward_kernel[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            rstd,
            n_cols,
            eps,
            ACTIVATION=activation_type,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.activation_type = activation_type
        ctx.save_for_backward(X, W, rstd)

        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.contiguous().view(-1, dim)
        X, W, rstd = ctx.saved_tensors
        X = X.contiguous().view(-1, dim)
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
        dW = torch.empty(n_cols, dtype=W.dtype, device=device)
        dX = torch.empty_like(dY)

        fused_rmsnorm_activation_backward_dx_fused[(n_rows,)](
            dX,
            dY,
            _dw,
            X,
            W,
            rstd,
            locks,
            X.stride(0),
            n_cols,
            ACTIVATION=ctx.activation_type,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        def grid(meta):
            return (triton.cdiv(n_cols, meta["BLOCK_SIZE_N"]),)

        fused_rmsnorm_activation_backward_dw[grid](
            _dw,
            dW,
            min(GROUP_SIZE_M, n_rows),
            n_cols,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
        )

        return (dX.view(*shape), dW, None, None)


class TritonFusedRMSNormActivationKernel:
    """Triton-accelerated fused RMSNorm + Activation kernel wrapper.

    Combines RMS normalization and activation into a single kernel launch.
    Supports mish, quickgelu, and squared_relu activations.

    Methods:
        is_available() -> bool:
            Check if Triton and CUDA are available.

        apply(X, W, eps, activation) -> torch.Tensor:
            Apply fused RMSNorm + activation.

            Args:
                X: Input tensor of any shape (*, N)
                W: Weight tensor, shape (N,)
                eps: Numerical stability constant
                activation: Activation name ("mish", "quickgelu", "squared_relu")
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
        X: torch.Tensor, W: torch.Tensor, eps: float, activation: str
    ) -> torch.Tensor:
        activation_type = _ACTIVATION_MAP[activation]
        return TritonFusedRMSNormActivationFunction.apply(X, W, eps, activation_type)
