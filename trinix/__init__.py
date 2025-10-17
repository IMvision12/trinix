from . import kernels, layers
from .kernels import (
    TritonALiBiKernel,
    TritonGeGLUKernel,
    TritonLayerNormKernel,
    TritonRelativeKernel,
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

from .layers.activation import FastGeGLU, FastSwiGLU
from .layers.attention import (
    FastBaseAttention,
    FastGroupedQueryAttention,
    FastMultiHeadAttention,
    FastMultiHeadSelfAttention,
)
from .layers.embeddings import (
    FastALiBiPositionEmbedding,
    FastRelativePositionEmbedding,
    FastRoPEPositionEmbedding,
)

from .layers.norm import FastLayerNorm, FastRMSNorm

__version__ = "0.1.0"
__all__ = [
    "layers",
    "kernels",
    "FastBaseAttention",
    "FastGroupedQueryAttention",
    "FastMultiHeadAttention",
    "FastMultiHeadSelfAttention",
    "FastRoPEPositionEmbedding",
    "FastALiBiPositionEmbedding",
    "FastRelativePositionEmbedding",
    "FastLayerNorm",
    "FastRMSNorm",
    "FastSwiGLU",
    "FastGeGLU",
    "TritonRoPEKernel",
    "TritonALiBiKernel",
    "TritonRelativeKernel",
    "TritonLayerNormKernel",
    "TritonRMSNormKernel",
    "TritonSwiGLUKernel",
    "TritonGeGLUKernel",
    "calculate_triton_kernel_configuration",
    "get_cuda_compute_capability",
]
