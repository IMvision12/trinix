# Optimizer Benchmark

## Test Environment

- **GPU**: NVIDIA A100-SXM4-80GB
- **CUDA**: 12.6
- **PyTorch**: 2.8.0+cu126

---

## Summary

| Optimizer | Average Speedup | Median Speedup | Best Speedup | Worst Speedup |
|-----------|-----------------|----------------|--------------|---------------|
| **Lion** | **4.17x** | **4.35x** | 4.43x | 3.52x |
| **Adam** | **3.02x** | **3.00x** | 3.07x | 2.98x |
| **AdamW** | **2.86x** | **2.84x** | 2.95x | 2.82x |
| **Muon** | **1.14x** | **1.23x** | 1.23x | 0.87x |

---

## Adam Optimizer

### Performance by Model Size

| Configuration | Parameters | PyTorch (ms) | Triton (ms) | Speedup |
|---------------|------------|--------------|-------------|---------|
| Small Model | 10M | 0.8757 | 0.2852 | **3.07x** |
| Medium Model | 100M | 6.7885 | 2.2756 | **2.98x** |
| Large Model | 500M | 33.3744 | 11.0852 | **3.01x** |
| XL Model | 1B | 66.3009 | 22.1176 | **3.00x** |

**Summary**: Average 3.02x, Median 3.00x
---

## AdamW Optimizer

### Performance by Model Size

| Configuration | Parameters | PyTorch (ms) | Triton (ms) | Speedup |
|---------------|------------|--------------|-------------|---------|
| Small Model | 10M | 0.8490 | 0.2879 | **2.95x** |
| Medium Model | 100M | 6.4645 | 2.2956 | **2.82x** |
| Large Model | 500M | 31.7683 | 11.2013 | **2.84x** |
| XL Model | 1B | 63.3144 | 22.3291 | **2.84x** |

**Summary**: Average 2.86x, Median 2.84x

---

## Lion Optimizer

### Performance by Model Size

| Configuration | Parameters | PyTorch (ms) | Triton (ms) | Speedup |
|---------------|------------|--------------|-------------|---------|
| Small Model | 10M | 0.7011 | 0.1990 | **3.52x** |
| Medium Model | 100M | 6.4627 | 1.5063 | **4.29x** |
| Large Model | 500M | 32.0407 | 7.2520 | **4.42x** |
| XL Model | 1B | 64.0803 | 14.4508 | **4.43x** |

**Summary**: Average 4.17x, Median 4.35x

---

## Muon Optimizer

### Performance by Model Size

| Configuration | Parameters | PyTorch (ms) | Triton (ms) | Speedup |
|---------------|------------|--------------|-------------|---------|
| Small Model | 10M | 0.4314 | 0.4983 | **0.87x** |
| Medium Model | 100M | 3.6574 | 2.9743 | **1.23x** |
| Large Model | 500M | 18.0263 | 14.7394 | **1.22x** |
| XL Model | 1B | 36.0196 | 29.3948 | **1.23x** |

**Summary**: Average 1.14x, Median 1.23x

**Note**: Muon shows modest speedups compared to other optimizers due to its simpler update rule (momentum-only, no second moment). The optimizer excels in memory efficiency (50% less than Adam/AdamW) while providing stable 1.2x speedup for models ≥100M parameters.

---