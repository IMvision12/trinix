# Position Embeddings Benchmark

## Test Environment

- **GPU**: NVIDIA A100-SXM4-40GB
- **CUDA**: 12.6
- **PyTorch**: 2.8.0+cu126

---

## Summary

| Metric | Value |
|--------|-------|
| **Average Speedup (Total)** | 2.49x |
| **Maximum Speedup (Total)** | 2.92x |
| **Minimum Speedup (Total)** | 1.83x |
| **Forward Pass Avg** | 4.77x |
| **Backward Pass Avg** | 2.07x |
| **Best Method** | RoPE (2.90-2.92x total), ALiBi (5.88x forward) |
| **Triton Threshold** | All configurations benefit |

---

## RoPE (Rotary Position Embedding)

### Configuration 1: Batch=1, SeqLen=2048, Hidden=4096, Heads=32

| Pass | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| Forward | 0.6478 | 0.3088 | 2.10x |
| Backward | 1.4525 | 0.8365 | 1.74x |
| **Total** | **2.1003** | **1.1452** | **1.83x** |

---

### Configuration 2: Batch=1, SeqLen=4096, Hidden=5120, Heads=40

| Pass | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| Forward | 1.3783 | 0.3320 | 4.15x |
| Backward | 3.2459 | 1.2610 | 2.57x |
| **Total** | **4.6242** | **1.5930** | **2.90x** |

---

### Configuration 3: Batch=1, SeqLen=4096, Hidden=8192, Heads=64

| Pass | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| Forward | 2.1571 | 0.4510 | 4.78x |
| Backward | 5.0667 | 2.0260 | 2.50x |
| **Total** | **7.2238** | **2.4770** | **2.92x** |

---

### Configuration 4: Batch=1, SeqLen=8192, Hidden=8192, Heads=64

| Pass | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| Forward | 4.2430 | 0.8874 | 4.78x |
| Backward | 9.9618 | 4.0190 | 2.48x |
| **Total** | **14.2048** | **4.9064** | **2.90x** |

---

## ALiBi (Attention with Linear Biases)

### Configuration 1: Batch=1, SeqLen=2048, Hidden=4096, Heads=32

| Pass | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| Forward | 2.0943 | 0.3562 | 5.88x |
| Backward | 4.0558 | 2.3222 | 1.75x |
| **Total** | **6.1501** | **2.6784** | **2.30x** |

---

### Configuration 2: Batch=1, SeqLen=4096, Hidden=5120, Heads=40

| Pass | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| Forward | 10.3026 | 1.7508 | 5.88x |
| Backward | 20.0000 | 11.4433 | 1.75x |
| **Total** | **30.3027** | **13.1942** | **2.30x** |

---

### Configuration 3: Batch=1, SeqLen=4096, Hidden=8192, Heads=64

| Pass | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| Forward | 16.2847 | 2.7867 | 5.84x |
| Backward | 31.7695 | 18.2708 | 1.74x |
| **Total** | **48.0542** | **21.0575** | **2.28x** |

---