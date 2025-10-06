import torch
import triton

def get_cuda_compute_capability():
    if not torch.cuda.is_available():
        return 75
    device_properties = torch.cuda.get_device_properties(0)
    return device_properties.major * 10 + device_properties.minor

def calculate_triton_kernel_configuration(input_vector_length):
    optimal_block_size = triton.next_power_of_2(input_vector_length)
    gpu_compute_capability = get_cuda_compute_capability()
    has_modern_architecture = gpu_compute_capability >= 80
    if has_modern_architecture:
        optimal_block_size = max(64, min(optimal_block_size, 16384))
        if optimal_block_size >= 16384:
            optimal_warp_count = 32
        elif optimal_block_size >= 8192:
            optimal_warp_count = 16
        elif optimal_block_size >= 2048:
            optimal_warp_count = 8
        elif optimal_block_size >= 512:
            optimal_warp_count = 4
        else:
            optimal_warp_count = 2
    else:
        optimal_block_size = max(128, min(optimal_block_size, 8192))
        if optimal_block_size >= 4096:
            optimal_warp_count = 16
        elif optimal_block_size >= 2048:
            optimal_warp_count = 8
        elif optimal_block_size >= 1024:
            optimal_warp_count = 4
        else:
            optimal_warp_count = 2
    return (optimal_block_size, optimal_warp_count)