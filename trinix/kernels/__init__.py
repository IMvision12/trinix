import warnings

try:
    import triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    warnings.warn("Triton not available. Triton kernels will not be available.")

if TRITON_AVAILABLE:
    from .adamw_kernel import TritonAdamWKernel
    from .alibi_kernel import TritonALiBiKernel
    from .attention_kernel import TritonAttentionKernel
    from .geglu_kernel import TritonGeGLUKernel
    from .layernorm_kernel import TritonLayerNormFunction, TritonLayerNormKernel
    from .lion_kernel import TritonLionKernel
    from .mish_kernel import TritonMishKernel
    from .quickgelu_kernel import TritonQuickGELUKernel
    from .reglu_kernel import TritonReGLUKernel
    from .rmsnorm_kernel import TritonRMSNormKernel
    from .rope_kernel import TritonRoPEKernel
    from .squared_relu_kernel import TritonSquaredReLUKernel
    from .swigelu_kernel import TritonSwiGLUKernel
    from .utils import (
        calculate_attention_block_sizes,
        calculate_triton_kernel_configuration,
        get_cuda_compute_capability,
        get_gpu_shared_memory_limit,
    )

    __all__ = [
        "TritonAdamWKernel",
        "TritonLionKernel",
        "TritonRoPEKernel",
        "TritonALiBiKernel",
        "TritonLayerNormKernel",
        "TritonLayerNormFunction",
        "TritonRMSNormKernel",
        "TritonSwiGLUKernel",
        "TritonGeGLUKernel",
        "TritonQuickGELUKernel",
        "TritonReGLUKernel",
        "TritonSquaredReLUKernel",
        "TritonMishKernel",
        "TritonAttentionKernel",
        "calculate_triton_kernel_configuration",
        "calculate_attention_block_sizes",
        "get_cuda_compute_capability",
        "get_gpu_shared_memory_limit",
    ]
else:

    class TritonAdamWKernel:
        @staticmethod
        def is_available():
            return False

    class TritonLionKernel:
        @staticmethod
        def is_available():
            return False

    class TritonALiBiKernel:
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

    class TritonSwiGLUKernel:
        @staticmethod
        def is_available():
            return False

    class TritonGeGLUKernel:
        @staticmethod
        def is_available():
            return False

    class TritonQuickGELUKernel:
        @staticmethod
        def is_available():
            return False

    class TritonReGLUKernel:
        @staticmethod
        def is_available():
            return False

    class TritonSquaredReLUKernel:
        @staticmethod
        def is_available():
            return False

    class TritonMishKernel:
        @staticmethod
        def is_available():
            return False

    class TritonAttentionKernel:
        @staticmethod
        def is_available():
            return False

    __all__ = [
        "TritonAdamWKernel",
        "TritonLionKernel",
        "TritonRoPEKernel",
        "TritonALiBiKernel",
        "TritonLayerNormKernel",
        "TritonRMSNormKernel",
        "TritonSwiGLUKernel",
        "TritonGeGLUKernel",
        "TritonQuickGELUKernel",
        "TritonReGLUKernel",
        "TritonSquaredReLUKernel",
        "TritonMishKernel",
        "TritonAttentionKernel",
    ]
