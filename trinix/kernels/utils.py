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


def get_gpu_shared_memory_limit():
    if not torch.cuda.is_available():
        return 48 * 1024

    device_properties = torch.cuda.get_device_properties(0)
    
    if hasattr(device_properties, 'shared_memory_per_block'):
        return device_properties.shared_memory_per_block
    elif hasattr(device_properties, 'max_shared_memory_per_block_optin'):
        return device_properties.max_shared_memory_per_block_optin
    else:
        compute_capability = device_properties.major * 10 + device_properties.minor
        if compute_capability >= 80:
            return 164 * 1024
        elif compute_capability >= 70:
            return 96 * 1024
        else:
            return 48 * 1024


def calculate_attention_block_sizes(head_dim, seq_len=None):
    gpu_compute_capability = get_cuda_compute_capability()
    shared_mem_limit = get_gpu_shared_memory_limit()

    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    def estimate_memory(block_m, block_n, block_d):
        q_mem = block_m * block_d * 4  # Q matrix
        k_mem = block_d * block_n * 4  # K matrix
        v_mem = block_n * block_d * 4  # V matrix
        acc_mem = block_m * block_d * 4  # Accumulator
        scores_mem = block_m * block_n * 4  # Attention scores
        overhead = 1024  # Other variables
        return q_mem + k_mem + v_mem + acc_mem + scores_mem + overhead

    if gpu_compute_capability >= 90:
        default_block_m = 64
        default_block_n = 64
    elif gpu_compute_capability >= 80:
        default_block_m = 64
        default_block_n = 64
    elif gpu_compute_capability >= 75:
        default_block_m = 64
        default_block_n = 64
    elif gpu_compute_capability >= 70:
        default_block_m = 64
        default_block_n = 64
    else:
        default_block_m = 32
        default_block_n = 32

    estimated_mem = estimate_memory(default_block_m, default_block_n, BLOCK_DMODEL)

    if estimated_mem > shared_mem_limit * 0.9:
        if default_block_m == 64:
            block_m, block_n = 32, 32
            estimated_mem = estimate_memory(block_m, block_n, BLOCK_DMODEL)

            if estimated_mem > shared_mem_limit * 0.9:
                block_m, block_n = 16, 16
        else:
            block_m, block_n = 16, 16
    else:
        block_m, block_n = default_block_m, default_block_n

    if head_dim >= 128 and block_m > 16:
        estimated_mem = estimate_memory(block_m, block_n, BLOCK_DMODEL)
        if estimated_mem > shared_mem_limit * 0.9:
            block_m = max(16, block_m // 2)
            block_n = max(16, block_n // 2)

    return (block_m, block_n, BLOCK_DMODEL)
