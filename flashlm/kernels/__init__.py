import warnings

try:
    import triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    warnings.warn("Triton not available. Triton kernels will not be available.")

if TRITON_AVAILABLE:
    from .alibi_kernel import TritonALiBiKernel
    from .layernorm_kernel import TritonLayerNormFunction, TritonLayerNormKernel
    from .relative_kernel import TritonRelativeKernel
    from .rmsnorm_kernel import TritonRMSNormKernel
    from .rope_kernel import TritonRoPEKernel
    from .utils import (
        calculate_triton_kernel_configuration,
        get_cuda_compute_capability,
    )

    __all__ = [
        "TritonRoPEKernel",
        "TritonALiBiKernel",
        "TritonRelativeKernel",
        "TritonLayerNormKernel",
        "TritonLayerNormFunction",
        "TritonRMSNormKernel",
        "calculate_triton_kernel_configuration",
        "get_cuda_compute_capability",
    ]
else:

    class TritonALiBiKernel:
        @staticmethod
        def is_available():
            return False

    class TritonRelativeKernel:
        @staticmethod
        def is_available():
            return False

    class TritonRMSNormKernel:
        @staticmethod
        def is_available():
            return False

    class TritonRoPEKernel:
        @staticmethod
        def is_available():
            return False

    class TritonLayerNormKernel:
        @staticmethod
        def is_available():
            return False

    __all__ = [
        "TritonRoPEKernel",
        "TritonALiBiKernel",
        "TritonRelativeKernel",
        "TritonLayerNormKernel",
        "TritonRMSNormKernel",
    ]
