import torch


def torch_int8_available():
    return hasattr(torch, '_int_mm')


def check_backend_availability():
    backends = {
        'torch_int8': torch_int8_available(),
        'cuda': torch.cuda.is_available(),
    }
    
    return backends
