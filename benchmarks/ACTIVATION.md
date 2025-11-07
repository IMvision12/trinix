# Activation Functions Benchmark

Performance comparison of GLU variants and activation functions with PyTorch vs Triton backends.

## Test Environment

- **GPU**: NVIDIA A100-SXM4-80GB
- **CUDA**: 12.6
- **PyTorch**: 2.8.0+cu126
- **Precision**: FP16
- **Warmup**: 10 iterations
- **Measurement**: 100 iterations

---

## Summary

| Metric | Value |
|--------|-------|
| **Average Speedup** | TBD |
| **Maximum Speedup** | TBD |
| **Best Activation** | TBD |
| **Triton Threshold** | hidden_size ≥ 512 |

---

## GLU Variants (Gated Linear Units)

### SwiGLU (Swish-Gated Linear Unit)
**Used in: LLaMA, PaLM, GLM**

| Config | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|--------|--------------|-------------|---------|-------------|
| H=512, S=2048, B=4 | TBD | TBD | TBD | TBD |
| H=1024, S=4096, B=2 | TBD | TBD | TBD | TBD |
| H=2048, S=8192, B=1 | TBD | TBD | TBD | TBD |
| H=4096, S=16384, B=1 | TBD | TBD | TBD | TBD |

**Formula**: `SwiGLU(x) = Swish(xW) ⊗ xV` where `Swish(x) = x · σ(x)`

---

### GeGLU (GELU-Gated Linear Unit)
**Used in: T5, Switch Transformer**

| Config | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|--------|--------------|-------------|---------|-------------|
| H=512, S=2048, B=4 | TBD | TBD | TBD | TBD |
| H=1024, S=4096, B=2 | TBD | TBD | TBD | TBD |
| H=2048, S=8192, B=1 | TBD | TBD | TBD | TBD |
| H=4096, S=16384, B=1 | TBD | TBD | TBD | TBD |

**Formula**: `GeGLU(x) = GELU(xW) ⊗ xV`

---

### ReGLU (ReLU-Gated Linear Unit)
**Used in: Some efficient transformers**

| Config | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|--------|--------------|-------------|---------|-------------|
| H=512, S=2048, B=4 | TBD | TBD | TBD | TBD |
| H=1024, S=4096, B=2 | TBD | TBD | TBD | TBD |
| H=2048, S=8192, B=1 | TBD | TBD | TBD | TBD |
| H=4096, S=16384, B=1 | TBD | TBD | TBD | TBD |

**Formula**: `ReGLU(x) = ReLU(xW) ⊗ xV`

---

## Other Activation Functions

### QuickGELU
**Fast approximation of GELU**

| Config | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|--------|--------------|-------------|---------|-------------|
| H=512, S=2048, B=4 | TBD | TBD | TBD | TBD |
| H=1024, S=4096, B=2 | TBD | TBD | TBD | TBD |
| H=2048, S=8192, B=1 | TBD | TBD | TBD | TBD |
| H=4096, S=16384, B=1 | TBD | TBD | TBD | TBD |

**Formula**: `QuickGELU(x) = x · σ(1.702x)`

---

### SquaredReLU
**Used in: Primer, some efficient architectures**

| Config | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|--------|--------------|-------------|---------|-------------|
| H=512, S=2048, B=4 | TBD | TBD | TBD | TBD |
| H=1024, S=4096, B=2 | TBD | TBD | TBD | TBD |
| H=2048, S=8192, B=1 | TBD | TBD | TBD | TBD |
| H=4096, S=16384, B=1 | TBD | TBD | TBD | TBD |

**Formula**: `SquaredReLU(x) = (ReLU(x))²`

---

### Mish
**Smooth activation function**

| Config | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|--------|--------------|-------------|---------|-------------|
| H=512, S=2048, B=4 | TBD | TBD | TBD | TBD |
| H=1024, S=4096, B=2 | TBD | TBD | TBD | TBD |
| H=2048, S=8192, B=1 | TBD | TBD | TBD | TBD |
| H=4096, S=16384, B=1 | TBD | TBD | TBD | TBD |

**Formula**: `Mish(x) = x · tanh(softplus(x))`

---

## Key Insights (Expected)

### GLU Variants
- ✅ **SwiGLU**: Best for modern LLMs (LLaMA, PaLM)
- ✅ **GeGLU**: Good for encoder-decoder models (T5)
- ✅ **ReGLU**: Fastest but less expressive
- ✅ **Triton speedup**: Expected 1.5-2x for hidden_size ≥ 512

### Activation Functions
- ✅ **QuickGELU**: Faster than standard GELU
- ✅ **SquaredReLU**: Simple and efficient
- ✅ **Mish**: Smooth but computationally expensive
- ✅ **Triton benefit**: More pronounced for complex activations

---

## Recommendations

### For LLM Training
```python
from trinix import FastSwiGLU

# SwiGLU is the standard for modern LLMs
activation = FastSwiGLU(
    input_dim=4096,
    hidden_dim=11008,  # Typical 2.7x expansion
    bias=False,
    use_triton=True  # Expected 1.7x speedup
)
```

### For Encoder-Decoder Models
```python
from trinix import FastGeGLU

# GeGLU works well with T5-style models
activation = FastGeGLU(
    input_dim=768,
    hidden_dim=3072,  # 4x expansion
    use_triton=True
)
```

### For Efficient Inference
```python
from trinix import FastReGLU

# ReGLU is fastest GLU variant
activation = FastReGLU(
    input_dim=2048,
    hidden_dim=8192,
    use_triton=True
)
```

---

## Configuration Guidelines

| Use Case | Activation | Hidden Dim | Expected Speedup |
|----------|------------|------------|------------------|
| **LLaMA-style** | SwiGLU | 11008 (2.7x) | 1.7x |
| **T5-style** | GeGLU | 3072 (4x) | 1.6x |
| **Efficient** | ReGLU | 8192 (4x) | 1.8x |
| **Fast GELU** | QuickGELU | Any | 1.5x |
| **Smooth** | Mish | Any | 1.4x |

---

## Performance Scaling (Expected)

### By Hidden Dimension
| Hidden | Expected Speedup |
|--------|------------------|
| 512 | 1.3-1.5x |
| 1024 | 1.5-1.7x |
| 2048 | 1.6-1.8x |
| 4096 | 1.7-2.0x |

### By Sequence Length
| SeqLen | Expected Speedup |
|--------|------------------|
| 2048 | 1.5x |
| 4096 | 1.6x |
| 8192 | 1.7x |
| 16384 | 1.8x |

---

## Memory Efficiency (Expected)

GLU variants typically show:
- **10-15% lower memory** usage with Triton
- **Better memory access patterns**
- **Reduced intermediate allocations**

---

## Reproduce These Results

```bash
# Install dependencies
pip install trinix torch triton

# Run benchmark
python benchmark_activation.py

# Custom configuration
python benchmark_activation.py --hidden_size 4096 --seq_len 8192
```

---

## Implementation Notes

### SwiGLU Architecture
```
Input: [batch, seq_len, input_dim]
  ↓
Split into two paths:
  Path 1: Linear(input_dim → hidden_dim)
  Path 2: Linear(input_dim → hidden_dim)
  ↓
Path 1: Apply Swish activation
  ↓
Element-wise multiply: Path1 ⊗ Path2
  ↓
Output: [batch, seq_len, hidden_dim]
```

### Triton Optimization
- **Fused operations**: Activation + gating in single kernel
- **Memory coalescing**: Optimized memory access patterns
- **Reduced overhead**: Fewer kernel launches

---

## Related Benchmarks

- [Attention Layers](ATTENTION.md)
- [Normalization Layers](NORMALIZATION.md)
- [Position Embeddings](EMBEDDINGS.md)
- [Complete Results](BENCHMARK_RESULTS.md)

---

*Last updated: 2025-01-07*
*Benchmarked on NVIDIA A100-SXM4-80GB*
*Results pending - run benchmark_activation.py to populate*
