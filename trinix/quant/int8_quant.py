import torch
import torch.nn as nn
from typing import Optional

from ..kernels.int8_quant_kernel import (
    quantize_activation_rowwise,
    quantize_weight_colwise,
    rescale_output,
)
from .utils import torch_int8_available


def int8_matmul(
    a: torch.Tensor,
    b_int8: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = a.dtype
    
    M, K = a.shape
    K_b, N = b_int8.shape
    assert K == K_b, f"Dimension mismatch: {K} vs {K_b}"
    
    a_int8, scale_a = quantize_activation_rowwise(a)
    
    if torch_int8_available():
        c_int32 = torch._int_mm(a_int8, b_int8)
    else:
        c_int32 = torch.matmul(a_int8.to(torch.int32), b_int8.to(torch.int32))
    
    output = rescale_output(c_int32, scale_a, scale_b, output_dtype)
    
    return output


class Int8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer(
            'weight_int8',
            torch.empty((out_features, in_features), dtype=torch.int8, device=device)
        )
        self.register_buffer(
            'weight_scales',
            torch.empty(in_features, dtype=torch.float32, device=device)
        )
        
        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_buffer('bias', None)
    
    @classmethod
    def from_float(cls, module: nn.Linear):
        int8_module = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        
        weight_int8, scales = quantize_weight_colwise(module.weight.data)
        int8_module.weight_int8.copy_(weight_int8)
        int8_module.weight_scales.copy_(scales)
        
        if module.bias is not None:
            int8_module.bias.copy_(module.bias.data)
        
        return int8_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        
        output = int8_matmul(
            x,
            self.weight_int8.t(),
            self.weight_scales,
            output_dtype=x.dtype,
        )
        
        if self.bias is not None:
            output += self.bias
        
        return output.view(*original_shape[:-1], self.out_features)


def int8_model(
    model: nn.Module,
    inplace: bool = True
):
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            int8_layer = Int8Linear.from_float(module)
            setattr(model, name, int8_layer)
            print(f"Converted {name} to INT8 layer")
        else:
            int8_model(module, inplace=True)
    
    return model
