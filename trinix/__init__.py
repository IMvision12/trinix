from .kernels import (
    TritonAdamWKernel,
    TritonALiBiKernel,
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

from .aug import Compose
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
from .layers.norm import FastLayerNorm, FastRMSNorm
from .optim import FastAdamW, FastMuon

__all__ = [
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
    "FastAdamW",
    "FastMuon",
    "TritonAdamWKernel",
    "TritonRoPEKernel",
    "TritonALiBiKernel",
    "TritonLayerNormKernel",
    "TritonRMSNormKernel",
    "TritonSwiGLUKernel",
    "TritonGeGLUKernel",
    "triton_attn_func",
    "calculate_triton_kernel_configuration",
    "get_cuda_compute_capability",
    "Compose",
    "OneOf",
    "Sequential",
]


from .version import version

__version__ = "0.1.1"
