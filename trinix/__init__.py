from .kernels import (
    TritonAdamWKernel,
    TritonALiBiKernel,
    TritonFusedResidualLayerNormKernel,
    TritonFusedResidualRMSNormKernel,
    TritonFusedRMSNormActivationKernel,
    TritonGeGLUKernel,
    TritonLayerNormKernel,
    TritonRMSNormKernel,
    TritonRoPEKernel,
    TritonSwiGLUKernel,
)

try:
    from .kernels import (
        calculate_triton_kernel_configuration,
        get_cuda_compute_capability,
    )
except ImportError:
    calculate_triton_kernel_configuration = None
    get_cuda_compute_capability = None

from .fusion import fuse
from .layers.activation import FastGeGLU, FastSwiGLU
from .layers.attention import (
    FastBaseAttention,
    FastGroupedQueryAttention,
    FastMultiHeadAttention,
    FastMultiHeadSelfAttention,
    triton_attn_func,
)
from .layers.embeddings import (
    FastALiBiPositionEmbedding,
    FastRoPEPositionEmbedding,
)
from .layers.fused import (
    FastFusedResidualLayerNorm,
    FastFusedResidualRMSNorm,
    FastFusedRMSNormActivation,
)
from .layers.norm import FastLayerNorm, FastRMSNorm
from .optim import FastAdamW, FastMuon

__all__ = [
    # Fusion API
    "fuse",
    # Layers
    "FastBaseAttention",
    "FastGroupedQueryAttention",
    "FastMultiHeadAttention",
    "FastMultiHeadSelfAttention",
    "FastRoPEPositionEmbedding",
    "FastALiBiPositionEmbedding",
    "FastLayerNorm",
    "FastRMSNorm",
    "FastSwiGLU",
    "FastGeGLU",
    # Fused Layers
    "FastFusedResidualRMSNorm",
    "FastFusedResidualLayerNorm",
    "FastFusedRMSNormActivation",
    # Optimizers
    "FastAdamW",
    "FastMuon",
    # Kernels
    "TritonAdamWKernel",
    "TritonRoPEKernel",
    "TritonALiBiKernel",
    "TritonLayerNormKernel",
    "TritonRMSNormKernel",
    "TritonSwiGLUKernel",
    "TritonGeGLUKernel",
    "TritonFusedResidualRMSNormKernel",
    "TritonFusedResidualLayerNormKernel",
    "TritonFusedRMSNormActivationKernel",
    "triton_attn_func",
    # Utilities
    "calculate_triton_kernel_configuration",
    "get_cuda_compute_capability",
]


from .version import version

__version__ = "0.1.4"
