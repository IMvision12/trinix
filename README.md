# Trinix 🚀

High-performance PyTorch layers library providing optimized implementations in both Triton and PyTorch. Trinix offers drop-in replacements for essential deep learning components including attention mechanisms, normalization layers, activation functions, and optimizers with automatic backend selection for maximum performance.

Attention layers intelligently choose between Flash Attention, Triton kernels, and PyTorch implementations, while other layers select between Triton and PyTorch backends based on your hardware capabilities and workload characteristics. When GPU acceleration isn't available, Trinix gracefully falls back to PyTorch implementations, ensuring your code runs everywhere while getting the best performance where possible.

## 🚀 Quickstart

**Requirements:**
- Python >= 3.9, < 3.14
- PyTorch >= 2.0.0
- NumPy >= 1.20.0

**Optional (for GPU acceleration):**
- CUDA-capable GPU
- Triton >= 2.0.0 (for Triton kernels)
- Flash Attention >= 2.0.0 (for optimized attention)

**Installation:**

```bash
# Basic installation (CPU/PyTorch backend only)
pip install trinix

# With GPU acceleration (Triton + Flash Attention)
pip install trinix[gpu]

# With all optional dependencies
pip install trinix[all]
```

**Install from Source:**

```bash
# Basic installation
pip install -U git+https://github.com/IMvision12/trinix

# With GPU support
pip install -U "trinix[gpu] @ git+https://github.com/IMvision12/trinix"
```

**Basic Usage:**

```python
import torch
from trinix import (
    FastMultiHeadAttention,
    FastRoPEPositionEmbedding,
    FastLayerNorm,
    FastAdamW,
)

# Create model components with automatic backend selection
attention = FastMultiHeadAttention(
    embed_dim=768,
    num_heads=12,
    kernel_type='flash'  # Options: 'flash', 'triton', 'pytorch'
)

rope = FastRoPEPositionEmbedding(dim=64, use_triton=True)
layernorm = FastLayerNorm(768, use_triton=True)

# Use in your model
x = torch.randn(4, 128, 768, device='cuda')
attn_output = attention(x, x, x)
normalized = layernorm(attn_output)

# Optimize with FastAdamW
optimizer = FastAdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

## 🛠️ Components

### 👁️ Attention Layers

Trinix provides multiple attention mechanisms with Flash Attention support. All attention layers support:

**Backend Selection** via `kernel_type` parameter:
- `'flash'`: Flash Attention (fastest, requires fp16/bf16)
- `'triton'`: Triton kernels (full feature support)
- `'pytorch'`: Standard PyTorch (most compatible)

**Position Encoding** via `position_method` parameter:
- `'rope'`: Rotary Position Embedding (used in LLaMA, Qwen, Gemma)
- `'alibi'`: Attention with Linear Biases (for length extrapolation)
- `'none'`: No position encoding (default)
- Custom `nn.Module`: Provide your own position encoding

**Available Attention Layers:**
- **FastMultiHeadAttention**: Standard multi-head attention
- **FastMultiHeadSelfAttention**: Optimized self-attention
- **FastGroupedQueryAttention**: Grouped-query attention (GQA) for efficient inference
- **FastMultiQueryAttention**: Multi-query attention (MQA)
- **FastMultiHeadLatentAttention**: Latent attention mechanisms

```python
from trinix import FastGroupedQueryAttention

# Grouped Query Attention (used in LLaMA 2, Mistral)
gqa = FastGroupedQueryAttention(
    embed_dim=4096,
    num_heads=32,
    num_kv_heads=8,  # Fewer KV heads for efficiency
    dropout=0.1,
    kernel_type='flash',  # Backend selection
    position_method='rope',  # Built-in RoPE support
    causal=True  # Causal masking for autoregressive models
)
```

### 🔧 Functional API

Direct access to Triton attention kernels:

```python
from trinix import triton_attn_func

# Functional Flash Attention interface
q = k = v = torch.randn(4, 128, 8, 64, device='cuda')

# Standard attention
output = triton_attn_func(q, k, v)

# Causal attention (for autoregressive models)
output = triton_attn_func(q, k, v, causal=True, dropout_p=0.1)

# Sliding window attention (for long sequences)
output = triton_attn_func(q, k, v, window_size=(256, 256))

# With ALiBi position biases
slopes = torch.randn(8, device='cuda')
output = triton_attn_func(q, k, v, alibi_slopes=slopes)

# Custom attention masks
mask = torch.zeros(128, 128, device='cuda')
mask[:, :64] = float('-inf')  # Mask out first 64 positions
output = triton_attn_func(q, k, v, attn_mask=mask)
```

### 📍 Position Embeddings

Position embeddings support Triton acceleration via `use_triton` parameter:

- **FastRoPEPositionEmbedding**: Rotary Position Embedding (RoPE) used in LLaMA, Qwen, Gemma
- **FastALiBiPositionEmbedding**: Attention with Linear Biases (ALiBi) for length extrapolation

```python
from trinix import FastRoPEPositionEmbedding, FastALiBiPositionEmbedding

# RoPE for rotary position encoding
rope = FastRoPEPositionEmbedding(
    dim=64,  # head_dim
    max_position_embeddings=2048,
    base=10000.0,
    use_triton=True  # Enable Triton acceleration
)

q = torch.randn(4, 1024, 8, 64, device='cuda')
k = torch.randn(4, 1024, 8, 64, device='cuda')
cos, sin = rope(q, seq_len=1024)
q_rot, k_rot = rope.apply_rotary_pos_emb(q, k, cos, sin)

# ALiBi for position biases
alibi = FastALiBiPositionEmbedding(
    num_heads=12,
    use_triton=True  # Enable Triton acceleration
)
bias = alibi(seq_len=512, batch_size=4)
```

### 📊 Normalization Layers

Normalization layers support Triton acceleration via `use_triton` parameter:

- **FastLayerNorm**: Layer normalization with Triton acceleration
- **FastRMSNorm**: Root Mean Square normalization (used in LLaMA, Mistral)

```python
from trinix import FastLayerNorm, FastRMSNorm

# Layer Normalization
ln = FastLayerNorm(
    768,
    eps=1e-5,
    use_triton=True  # Enable Triton acceleration
)

# RMS Normalization (more efficient, used in modern LLMs)
rms = FastRMSNorm(
    768,
    eps=1e-6,
    use_triton=True  # Enable Triton acceleration
)
```

### 🎨 Activation Functions

Optimized gated linear unit (GLU) variants with Triton acceleration via `use_triton` parameter:

- **FastSwiGLU**: SwiGLU activation (used in LLaMA, PaLM)
- **FastGeGLU**: GeGLU activation
- **FastReGLU**: ReGLU activation
- **FastQuickGELU**: Fast approximation of GELU
- **FastSquaredReLU**: Squared ReLU activation
- **FastMish**: Mish activation function

```python
from trinix import FastSwiGLU, FastGeGLU

# SwiGLU (Swish-Gated Linear Unit) - used in LLaMA
swiglu = FastSwiGLU(
    input_dim=768,
    hidden_dim=3072,
    bias=False,
    use_triton=True  # Enable Triton acceleration
)

# GeGLU (GELU-Gated Linear Unit) - used in T5
geglu = FastGeGLU(
    input_dim=768,
    hidden_dim=3072,
    use_triton=True  # Enable Triton acceleration
)
```

### 🎯 Optimizers

- **FastAdamW**: AdamW with decoupled weight decay and Triton acceleration
- **FastAdam**: Standard Adam optimizer with Triton kernels
- **FastLion**: Lion optimizer (evolved sign momentum)

```python
from trinix import FastAdamW

optimizer = FastAdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    use_triton=True
)
```

### ⚙️ Backend Selection

Layers support explicit backend control with different parameters:

**Attention Layers** use `kernel_type`:
```python
# Flash Attention (fastest, recommended for fp16/bf16)
attn = FastMultiHeadAttention(embed_dim=768, num_heads=12, kernel_type='flash')

# Triton kernels (supports custom masks and all features)
attn = FastMultiHeadAttention(embed_dim=768, num_heads=12, kernel_type='triton')

# PyTorch (most compatible, CPU-friendly)
attn = FastMultiHeadAttention(embed_dim=768, num_heads=12, kernel_type='pytorch')
```

**Other Layers** use `use_triton`:
```python
# Enable Triton acceleration (auto-fallback to PyTorch if unavailable)
layer = FastLayerNorm(768, use_triton=True)

# Force PyTorch backend
layer = FastLayerNorm(768, use_triton=False)

# Automatic selection (default, recommended)
rope = FastRoPEPositionEmbedding(dim=64)  # Chooses best backend automatically
```

## 🥇 Performance Benchmarking

Comprehensive benchmarks on **NVIDIA A100** (40GB/80GB) with **CUDA 12.6** and **PyTorch 2.8.0**:

### 🎯 Attention Mechanisms

**Training (Forward + Backward):**

| Attention Type | Best Backend | Total Speedup | Backward Speedup | Use Case |
|----------------|--------------|---------------|------------------|----------|
| **Self-Attention** | Triton | **2.50-3.86x** | **7.47-17.15x** 🔥 | GPT-style models |
| **Multi-Head (MHA)** | Triton | **2.52-3.47x** | **7.72-17.02x** 🔥 | Standard transformers |
| **Grouped Query (GQA)** | Triton | **2.34-4.84x** | **7.59-15.59x** | LLaMA 2, Mistral |
| **Multi-Query (MQA)** | Triton | **2.42-3.64x** | **7.57-15.43x** | PaLM, StarCoder |
| **Latent Attention** | Triton/Flash | **1.66-2.11x** | **4.43-10.31x** | Long context |

**Inference (Forward Only):**

| Attention Type | Flash Speedup | Best Configuration |
|----------------|---------------|-------------------|
| Self-Attention | **1.76-3.20x** | SeqLen=2048, Heads=12 |
| GQA | **1.92-3.27x** | SeqLen=2048, 32 heads, 8 KV heads |
| MQA | **2.19-3.17x** | SeqLen=4096, 32 heads |

**Key Insight**: Triton dominates training with exceptional backward pass speedup (up to **17x**). Flash Attention excels for inference.

[📊 Full Attention Benchmarks](benchmarks/ATTENTION.md)

---

### 🎨 Activation Functions

| Activation | Total Speedup | Forward Speedup | Use Case |
|------------|---------------|-----------------|----------|
| **Mish** | **2.82-3.01x** | **3.36-3.41x** 🔥 | Smooth activation |
| **QuickGELU** | **2.83-2.93x** | **3.34-3.41x** | Fast GELU approximation |
| **SquaredReLU** | **1.92-1.98x** | **1.95-1.97x** | Efficient, used in Primer |
| **SwiGLU** | **1.44-1.88x** | **1.88-1.90x** | LLaMA, PaLM (standard) |
| **GeGLU** | **1.60-1.87x** | **1.78-1.93x** | T5, Switch Transformer |
| **ReGLU** | **1.45-2.17x** | **1.85-1.87x** | Efficient GLU variant |

**Summary**: Average **2.22x** speedup across 24 tests. Non-GLU activations (Mish, QuickGELU) show best speedups.

[📊 Full Activation Benchmarks](benchmarks/ACTIVATION.md)

---

### 📍 Position Embeddings

| Method | Total Speedup | Forward Speedup | Memory Efficiency |
|--------|---------------|-----------------|-------------------|
| **RoPE** | **1.83-2.92x** | **2.10-4.78x** | ✅ Excellent (handles 8K+ seq) |
| **ALiBi** | **2.28-2.30x** | **5.84-5.88x** 🔥 | ⚠️ High (OOM at 8K seq) |

**Key Insight**: RoPE scales better for long sequences. ALiBi has outstanding forward speedup but high memory usage.

[📊 Full Embedding Benchmarks](benchmarks/EMBEDDINGS.md)

---

### 📊 Normalization Layers

| Layer | Speedup | Best Configuration | Use Case |
|-------|---------|-------------------|----------|
| **RMSNorm** | **3.64-3.78x** 🔥 | SeqLen≥4096, Hidden≥8192 | LLaMA, Mistral, Qwen |
| **LayerNorm** | **1.54-1.59x** | Hidden>4096 | Standard transformers |

**Summary**: RMSNorm shows **3.7x** consistent speedup at scale. LayerNorm falls back to PyTorch for hidden_size ≤ 4096.

[📊 Full Normalization Benchmarks](benchmarks/NORMALIZATION.md)

---

### 🎯 Optimizers

| Optimizer | Average Speedup | Best Speedup | Memory Benefit |
|-----------|-----------------|--------------|----------------|
| **Lion** | **4.17x** 🔥 | **4.43x** (1B params) | High (33% less than Adam) |
| **Adam** | **3.02x** | **3.07x** (10M params) | Medium |
| **AdamW** | **2.86x** | **2.95x** (10M params) | Medium |

**Key Insight**: Lion optimizer shows best speedup and scales excellently with model size. Ideal for large-scale training.

[📊 Full Optimizer Benchmarks](benchmarks/OPTIMIZER.md)

---

## 🔖 License

Trinix Code: This repository is licensed under the Apache 2.0 License.

## 📚 Citation

You can cite the Trinix repo as follows:

```bibtex
@software{trinix2024,
  author = {Gitesh Chawda},
  title = {Trinix},
  year = {2025},
  url = {https://github.com/IMvision12/trinix}
}
```

## 🌟 Acknowledgments

Trinix builds upon the following projects:

1. **[Triton](https://github.com/openai/triton)** - GPU kernel compilation and optimization framework
2. **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** - Memory-efficient attention implementation
3. **[PyTorch](https://github.com/pytorch/pytorch)** - Deep learning framework and tensor operations
