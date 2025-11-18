# Normalization Layers Benchmark

## Test Environment

- **GPU**: NVIDIA A100-SXM4-80GB
- **CUDA**: 12.6
- **PyTorch**: 2.8.0+cu126

---

## Summary

| Metric | Value |
|--------|-------|
| **Average Speedup** | **2.38x** |
| **Maximum Speedup** | **3.78x** (RMSNorm, Large config) |
| **Minimum Speedup** | **1.31x** (RMSNorm, Small config) |
| **Total Tests** | 12 |
| **Triton Available** | 10/12 |

---

## Detailed Results

### Small Configuration
**Batch=4, SeqLen=512, Hidden=4096**

| Layer | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-------|--------------|-------------|---------|-------------|
| LayerNorm | 0.0651 | N/A* | - | - |
| RMSNorm | 0.1826 | 0.1390 | **1.31x** | - |

*Falls back to PyTorch (hidden_size ≤ 4096 threshold)

---

### Medium Configuration
**Batch=1, SeqLen=2048, Hidden=4096**

| Layer | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-------|--------------|-------------|---------|-------------|
| LayerNorm | 0.0646 | N/A* | - | - |
| RMSNorm | 0.1827 | 0.1312 | **1.39x** | - |

*Falls back to PyTorch (hidden_size ≤ 4096 threshold)

---

### Large Configuration
**Batch=1, SeqLen=4096, Hidden=8192**

| Layer | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-------|--------------|-------------|---------|-------------|
| LayerNorm | 0.2573 | 0.1620 | **1.59x** | - |
| RMSNorm | 0.6094 | 0.1611 | **3.78x** | - |

---

### XL Configuration
**Batch=1, SeqLen=16384, Hidden=8192**

| Layer | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-------|--------------|-------------|---------|-------------|
| LayerNorm | 0.9568 | 0.6207 | **1.54x** | - |
| RMSNorm | 2.2912 | 0.6186 | **3.70x** | - |

---

### XXL Configuration
**Batch=1, SeqLen=32768, Hidden=12288**

| Layer | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-------|--------------|-------------|---------|-------------|
| LayerNorm | 2.8812 | 1.8516 | **1.56x** | - |
| RMSNorm | 6.7995 | 1.8303 | **3.72x** | - |

---

### XXXL Configuration
**Batch=1, SeqLen=131072, Hidden=16384**

| Layer | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-------|--------------|-------------|---------|-------------|
| LayerNorm | 15.2244 | 9.8518 | **1.55x** | - |
| RMSNorm | 36.2859 | 9.9628 | **3.64x** | - |

---