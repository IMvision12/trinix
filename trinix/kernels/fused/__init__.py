import warnings

try:
    import triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    from .residual_rmsnorm_kernel import TritonFusedResidualRMSNormKernel
    from .residual_layernorm_kernel import TritonFusedResidualLayerNormKernel
    from .rmsnorm_activation_kernel import TritonFusedRMSNormActivationKernel

    __all__ = [
        "TritonFusedResidualRMSNormKernel",
        "TritonFusedResidualLayerNormKernel",
        "TritonFusedRMSNormActivationKernel",
    ]
else:

    class TritonFusedResidualRMSNormKernel:
        @staticmethod
        def is_available():
            return False

    class TritonFusedResidualLayerNormKernel:
        @staticmethod
        def is_available():
            return False

    class TritonFusedRMSNormActivationKernel:
        @staticmethod
        def is_available():
            return False

    __all__ = [
        "TritonFusedResidualRMSNormKernel",
        "TritonFusedResidualLayerNormKernel",
        "TritonFusedRMSNormActivationKernel",
    ]
