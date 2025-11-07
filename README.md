# Trinix 🚀

High-performance PyTorch library providing drop-in replacements for neural network layers with automatic backend selection between optimized Triton kernels and PyTorch implementations. Accelerate training and inference of large language models (LLMs) and transformer architectures without code changes.

Trinix intelligently selects between Flash Attention, Triton kernels, and PyTorch backends based on hardware capabilities and workload characteristics, with graceful fallback for maximum compatibility.

## 🚀 Quickstart

**Requirements:**
- Python >= 3.9, < 3.14
- PyTorch >= 2.0.0
- CUDA-capable GPU (for Triton acceleration)
- Triton >= 2.0.0 (optional, for GPU acceleration)
- Flash Attention >= 2.0.0 (optional, for optimized attention)

**Installation:**

```bash
pip install trinix
```

**Install from Source:**

```bash
pip install -U git+https://github.com/IMvision12/trinix
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

Trinix automatically selects the optimal backend based on:

- **Hardware**: CUDA availability and compute capability
- **Tensor Size**: Larger tensors benefit more from Triton
- **Sequence Length**: Longer sequences see greater speedups
- **Model Scale**: Optimized for LLM-scale workloads

Example speedups on NVIDIA A100 (40GB):

| Component | Configuration | Speedup |
|-----------|--------------|---------|
| RoPE | LLaMA-7B scale (4096 hidden, 2048 seq) | 2.1x |
| ALiBi | 32 heads, 2048 sequence | 1.8x |
| LayerNorm | 8192 hidden dim | 1.5x |
| SwiGLU | 4096→11008 expansion | 1.7x |

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
