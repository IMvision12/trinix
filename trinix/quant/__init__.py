"""
INT8 Quantization module for Trinix
"""
from .utils import torch_int8_available, check_backend_availability
from ..kernels.int8_quant_kernel import (
    quantize_activation_rowwise,
    quantize_weight_colwise,
    rescale_output,
)

from .int8_quant import Int8Linear, int8_model, int8_matmul

__all__ = [
    # Operations
    'quantize_activation_rowwise',
    'quantize_weight_colwise',
    'rescale_output',
    'int8_matmul',
    # Layers
    'Int8Linear',
    'int8_model',
    # Utils
    'check_backend_availability',
    'torch_int8_available',
]
