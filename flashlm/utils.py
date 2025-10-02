import torch
import warnings
from typing import Dict, Any


def get_available_kernels() -> Dict[str, bool]:
    kernels = {
        "flash": False,
        "triton": False,
        "pytorch": True,
    }
    
    try:
        import flash_attn
        kernels["flash"] = True
    except ImportError:
        pass
    
    try:
        import triton
        kernels["triton"] = True
    except ImportError:
        pass
    
    return kernels


def validate_kernel_type(kernel_type: str) -> str:
    available = get_available_kernels()
    
    if kernel_type not in available:
        warnings.warn(f"Unknown kernel type '{kernel_type}', falling back to 'pytorch'")
        return "pytorch"
    
    if not available[kernel_type]:
        warnings.warn(f"Kernel '{kernel_type}' not available, falling back to 'pytorch'")
        return "pytorch"
    
    return kernel_type


def print_kernel_info():
    kernels = get_available_kernels()
    
    print("FlashLM Kernel Availability:")
    print("-" * 30)
    for kernel, available in kernels.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{kernel.capitalize():10}: {status}")
    
    if kernels["flash"]:
        try:
            import flash_attn
            print(f"\nFlash Attention version: {flash_attn.__version__}")
        except:
            pass
    
    if kernels["triton"]:
        try:
            import triton
            print(f"Triton version: {triton.__version__}")
        except:
            pass


def benchmark_kernels(
    embed_dim: int = 512,
    num_heads: int = 8,
    seq_len: int = 1024,
    batch_size: int = 4,
    num_runs: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        warnings.warn("CUDA not available, benchmarking on CPU")
    
    from .layers.attention import FastMultiHeadSelfAttention
    
    results = {}
    available_kernels = get_available_kernels()
    
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    for kernel_type in ["pytorch", "flash", "triton"]:
        if not available_kernels[kernel_type]:
            continue
        
        attn = FastMultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_type=kernel_type
        ).to(device)
        
        with torch.no_grad():
            for _ in range(10):
                _ = attn(x)
        
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if device == "cuda":
            start_time.record()
        else:
            import time
            start = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = attn(x)
        
        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            elapsed_time = time.time() - start
        
        results[kernel_type] = elapsed_time / num_runs
    
    return results