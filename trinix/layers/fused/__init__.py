from .elementwise_chain import FusedElementwiseChain
from .residual_layernorm import FastFusedResidualLayerNorm
from .residual_rmsnorm import FastFusedResidualRMSNorm
from .rmsnorm_activation import FastFusedRMSNormActivation

__all__ = [
    "FastFusedResidualRMSNorm",
    "FastFusedResidualLayerNorm",
    "FastFusedRMSNormActivation",
    "FusedElementwiseChain",
]


def _register_fusion_patterns():
    """Register built-in fusion patterns with the fusion registry.

    Called lazily by trinix.fusion to avoid circular imports.
    """
    from ...fusion import register_fusion
    from ..norm.rmsnorm import FastRMSNorm
    from ..activation.mish import FastMish
    from ..activation.quickgelu import FastQuickGELU
    from ..activation.squared_relu import FastSquaredReLU

    def _make_rmsnorm_mish(modules):
        norm = modules[0]
        return FastFusedRMSNormActivation(
            hidden_size=norm.hidden_size,
            eps=norm.eps,
            activation="mish",
            use_triton=norm.use_triton,
        )

    def _make_rmsnorm_quickgelu(modules):
        norm = modules[0]
        return FastFusedRMSNormActivation(
            hidden_size=norm.hidden_size,
            eps=norm.eps,
            activation="quickgelu",
            use_triton=norm.use_triton,
        )

    def _make_rmsnorm_squared_relu(modules):
        norm = modules[0]
        return FastFusedRMSNormActivation(
            hidden_size=norm.hidden_size,
            eps=norm.eps,
            activation="squared_relu",
            use_triton=norm.use_triton,
        )

    register_fusion((FastRMSNorm, FastMish), _make_rmsnorm_mish)
    register_fusion((FastRMSNorm, FastQuickGELU), _make_rmsnorm_quickgelu)
    register_fusion((FastRMSNorm, FastSquaredReLU), _make_rmsnorm_squared_relu)
