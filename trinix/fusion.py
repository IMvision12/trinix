import warnings

import torch.nn as nn


# Registry: maps tuple of module types → factory function
_FUSION_REGISTRY = {}
_PATTERNS_REGISTERED = False


def register_fusion(pattern_key, factory_fn):
    """Register a fusion pattern.

    Args:
        pattern_key: Tuple of module types, e.g. (FastRMSNorm, FastMish).
        factory_fn: Callable that takes a list of module instances and returns
                    a fused module. Signature: factory_fn(modules) -> nn.Module.
    """
    _FUSION_REGISTRY[pattern_key] = factory_fn


def _ensure_patterns_registered():
    """Lazily register built-in fusion patterns to avoid circular imports."""
    global _PATTERNS_REGISTERED
    if not _PATTERNS_REGISTERED:
        _PATTERNS_REGISTERED = True
        from .layers.fused import _register_fusion_patterns

        _register_fusion_patterns()


def fuse(*modules):
    """Fuse a chain of modules into a single optimized module.

    Looks up the types of the given modules in the fusion registry. If a
    matching pre-built fused kernel exists, returns the fused module. Otherwise
    falls back to a torch.compile-wrapped nn.Sequential with a warning.

    This API works for single-input chains. For multi-input fusions (like
    residual + norm), use the fused modules directly:
        fused_norm = FastFusedResidualRMSNorm(dim)
        output = fused_norm(x, residual)

    Args:
        *modules: nn.Module instances to fuse together. Must be at least 2.

    Returns:
        nn.Module: A fused module (pre-built or torch.compile fallback).

    Examples:
        >>> from trinix import FastRMSNorm, FastMish, fuse
        >>> fused = fuse(FastRMSNorm(4096), FastMish(4096))  # single kernel
        >>> output = fused(x)

        >>> # Unknown pattern falls back to nn.Sequential
        >>> fused = fuse(FastRMSNorm(4096), nn.Linear(4096, 4096))
        >>> # UserWarning: No fused kernel available...
    """
    if len(modules) < 2:
        raise ValueError("fuse() requires at least 2 modules")

    _ensure_patterns_registered()

    pattern_key = tuple(type(m) for m in modules)

    if pattern_key in _FUSION_REGISTRY:
        return _FUSION_REGISTRY[pattern_key](list(modules))

    warnings.warn(
        f"No fused kernel available for pattern {pattern_key}, "
        f"falling back to sequential execution with torch.compile.",
        UserWarning,
        stacklevel=2,
    )

    from .layers.fused.elementwise_chain import FusedElementwiseChain

    return FusedElementwiseChain(*modules)
