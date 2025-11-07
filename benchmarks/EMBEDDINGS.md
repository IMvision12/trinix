# Position Embeddings Benchmark

Performance comparison of position encoding methods with PyTorch vs Triton backends.

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
| **Best Method** | TBD |
| **Triton Threshold** | hidden_size > 2048, seq_len > 512 |

---

## RoPE (Rotary Position Embedding)
**Used in: LLaMA, Qwen, Gemma, GPT-NeoX**

### Small Configuration
**Heads=32, HeadDim=64, SeqLen=512, Batch=4**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate cos/sin | TBD | TBD | TBD | TBD |
| Apply rotation | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

### Medium Configuration
**Heads=64, HeadDim=128, SeqLen=2048, Batch=2**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate cos/sin | TBD | TBD | TBD | TBD |
| Apply rotation | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

### Large Configuration (LLaMA 3 70B)
**Heads=64, HeadDim=128, SeqLen=8192, Batch=1**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate cos/sin | TBD | TBD | TBD | TBD |
| Apply rotation | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

### XL Configuration (Long Context)
**Heads=128, HeadDim=128, SeqLen=32768, Batch=1**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate cos/sin | TBD | TBD | TBD | TBD |
| Apply rotation | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

## ALiBi (Attention with Linear Biases)
**Used in: BLOOM, MPT, some efficient models**

### Small Configuration
**Heads=32, SeqLen=512, Batch=4**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate slopes | TBD | TBD | TBD | TBD |
| Compute biases | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

### Medium Configuration
**Heads=64, SeqLen=2048, Batch=2**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate slopes | TBD | TBD | TBD | TBD |
| Compute biases | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

### Large Configuration
**Heads=128, SeqLen=8192, Batch=1**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate slopes | TBD | TBD | TBD | TBD |
| Compute biases | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

### XL Configuration (Long Context)
**Heads=128, SeqLen=32768, Batch=1**

| Operation | PyTorch (ms) | Triton (ms) | Speedup | Memory (MB) |
|-----------|--------------|-------------|---------|-------------|
| Generate slopes | TBD | TBD | TBD | TBD |
| Compute biases | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

---

## Key Insights (Expected)

### RoPE Performance
- ✅ **Triton activation**: hidden_size > 2048 AND seq_len > 512
- ✅ **Expected speedup**: 1.8-2.1x for large models
- ✅ **Best for**: LLaMA-style models with long context
- ✅ **Memory efficient**: No learned parameters

### ALiBi Performance
- ✅ **Simpler computation** than RoPE
- ✅ **Expected speedup**: 1.5-1.8x
- ✅ **Best for**: Length extrapolation
- ✅ **No rotation**: Just bias addition

### Comparison
| Feature | RoPE | ALiBi |
|---------|------|-------|
| **Computation** | Rotation (complex) | Bias (simple) |
| **Parameters** | None | None |
| **Extrapolation** | Good | Excellent |
| **Speed** | Moderate | Fast |
| **Memory** | Low | Very Low |
| **Used in** | LLaMA, Qwen | BLOOM, MPT |

---

## Recommendations

### For LLaMA-style Models
```python
from trinix import FastRoPEPositionEmbedding

# RoPE is standard for modern LLMs
rope = FastRoPEPositionEmbedding(
    dim=128,  # head_dim
    max_position_embeddings=32768,
    base=10000.0,
    use_triton=True  # Expected 2.1x speedup
)

# Usage
q = torch.randn(batch, seq_len, num_heads, head_dim)
k = torch.randn(batch, seq_len, num_heads, head_dim)
cos, sin = rope(q, seq_len)
q_rot, k_rot = rope.apply_rotary_pos_emb(q, k, cos, sin)
```

### For Length Extrapolation
```python
from trinix import FastALiBiPositionEmbedding

# ALiBi for better length extrapolation
alibi = FastALiBiPositionEmbedding(
    num_heads=64,
    max_seq_len=32768,
    use_triton=True  # Expected 1.8x speedup
)

# Usage
bias = alibi(seq_len, batch_size)
# Add bias to attention scores before softmax
```

### For Efficient Inference
```python
# RoPE with caching for inference
rope = FastRoPEPositionEmbedding(
    dim=128,
    max_position_embeddings=8192,
    use_triton=True
)

# Pre-compute for inference
cos_cached, sin_cached = rope.forward(
    torch.zeros(1, max_seq_len, 1, head_dim),
    seq_len=max_seq_len
)
```

---

## Configuration Guidelines

| Use Case | Method | Config | Expected Speedup |
|----------|--------|--------|------------------|
| **LLaMA Training** | RoPE | H=128, S=8192 | 2.1x |
| **Long Context** | RoPE | H=128, S=32768 | 2.0x |
| **Extrapolation** | ALiBi | Any | 1.8x |
| **Efficient** | ALiBi | S=2048 | 1.5x |
| **Small Models** | Either | H<64, S<512 | Fallback |

---

## Performance Scaling (Expected)

### RoPE by Configuration
| Config | Expected Speedup | Reason |
|--------|------------------|--------|
| Small (H=64, S=512) | Fallback | Below threshold |
| Medium (H=128, S=2048) | 1.8x | At threshold |
| Large (H=128, S=8192) | 2.1x | Optimal |
| XL (H=128, S=32768) | 2.0x | Memory bound |

### ALiBi by Sequence Length
| SeqLen | Expected Speedup |
|--------|------------------|
| 512 | 1.4x |
| 2048 | 1.5x |
| 8192 | 1.7x |
| 32768 | 1.8x |

---

## Memory Efficiency (Expected)

### RoPE
- **No learned parameters**: Zero parameter memory
- **Cached cos/sin**: Reusable across batches
- **Triton benefit**: 10-15% less temporary memory

### ALiBi
- **Minimal memory**: Just slope values
- **No caching needed**: Computed on-the-fly
- **Very efficient**: Best for memory-constrained scenarios

---

## Implementation Details

### RoPE Algorithm
```
1. Generate position indices: [0, 1, 2, ..., seq_len-1]
2. Compute frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
3. Compute angles: position * frequency
4. Generate cos(angles) and sin(angles)
5. Apply rotation to query and key:
   - Split features into pairs
   - Rotate each pair by corresponding angle
```

### ALiBi Algorithm
```
1. Generate slopes: 2^(-8i/num_heads) for i in [1, num_heads]
2. Compute position differences: |i - j| for all positions
3. Compute biases: -slope * position_difference
4. Add biases to attention scores before softmax
```

---

## Triton Optimization Details

### RoPE Optimization
- **Fused rotation**: Combine cos/sin computation with rotation
- **Vectorized operations**: Process multiple heads in parallel
- **Memory coalescing**: Optimized access patterns

### ALiBi Optimization
- **Fused bias computation**: Generate and apply biases in one kernel
- **Efficient broadcasting**: Minimize memory transfers
- **Cached slopes**: Reuse across sequences

---

## Reproduce These Results

```bash
# Install dependencies
pip install trinix torch triton

# Run benchmark
python benchmark_embeddings.py

# Custom configuration
python benchmark_embeddings.py --num_heads 64 --head_dim 128 --seq_len 8192
```

---

## Related Benchmarks

- [Attention Layers](ATTENTION.md)
- [Normalization Layers](NORMALIZATION.md)
- [Activation Functions](ACTIVATION.md)
- [Complete Results](BENCHMARK_RESULTS.md)

---

*Last updated: 2025-01-07*
*Benchmarked on NVIDIA A100-SXM4-80GB*
*Results pending - run benchmark_embeddings.py to populate*
