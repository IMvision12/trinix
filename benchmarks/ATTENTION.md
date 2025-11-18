# Attention Layers Benchmark


## Test Environment

- **GPU**: NVIDIA A100-SXM4-40GB
- **CUDA**: 12.6
- **PyTorch**: 2.8.0+cu126

---

## Summary

| Metric | Value |
|--------|-------|
| **Best Backend** | Flash Attention (forward), Triton (backward) |
| **Flash Forward Speedup** | 1.76-3.77x |
| **Triton Backward Speedup** | 3.37-41.99x |
| **Best Overall** | Flash Attention for forward-heavy workloads |
| **Total Tests** | 45 configurations across 5 attention types |

---

## Multi-Head Self-Attention (Fused QKV)

### Configuration 1: Batch=2, SeqLen=2048, Heads=12, HeadDim=64

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 1.5461 | 0.9185 | **0.4836** | **3.20x** (Flash) |
| Backward | 2.7444 | **0.3675** | 1.3318 | **7.47x** (Triton) |
| Total | 3.7505 | **0.9707** | 1.5866 | **3.86x** (Triton) |

### Configuration 2: Batch=1, SeqLen=4096, Heads=32, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 9.1440 | 8.9353 | **3.6391** | **2.51x** (Flash) |
| Backward | 15.6006 | **0.9097** | 9.1439 | **17.15x** (Triton) |
| Total | 24.5154 | **9.6654** | 12.5642 | **2.54x** (Triton) |

### Configuration 3: Batch=1, SeqLen=2048, Heads=40, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 3.8319 | 3.8639 | **2.1802** | **1.76x** (Flash) |
| Backward | 7.5638 | **0.8141** | 5.5167 | **9.29x** (Triton) |
| Total | 11.1777 | **4.4747** | 7.5320 | **2.50x** (Triton) |

---

## Multi-Head Attention (Separate Q/K/V)

### Configuration 1: Batch=2, SeqLen=2048, Heads=12, HeadDim=64

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 1.6297 | 0.9850 | **0.5485** | **2.97x** (Flash) |
| Backward | 2.9216 | **0.3786** | 1.4171 | **7.72x** (Triton) |
| Total | 3.9123 | **1.1282** | 1.7229 | **3.47x** (Triton) |

### Configuration 2: Batch=1, SeqLen=4096, Heads=32, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 9.2263 | 9.0310 | **3.6937** | **2.50x** (Flash) |
| Backward | 15.5443 | **0.9135** | 9.0987 | **17.02x** (Triton) |
| Total | 24.5741 | **9.6974** | 12.4975 | **2.53x** (Triton) |

### Configuration 3: Batch=1, SeqLen=2048, Heads=40, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 3.9236 | 4.8992 | **2.2899** | **1.71x** (Flash) |
| Backward | 7.7086 | **0.8647** | 5.7080 | **8.91x** (Triton) |
| Total | 11.4532 | **4.5519** | 7.8258 | **2.52x** (Triton) |

---

## Grouped Query Attention (GQA)

### Configuration 1: Batch=2, SeqLen=2048, Heads=32, KVHeads=8, HeadDim=64

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 3.9803 | 2.1496 | **1.2159** | **3.27x** (Flash) |
| Backward | 6.5527 | **0.4368** | 3.0534 | **15.00x** (Triton) |
| Total | 9.8826 | **2.0408** | 3.3544 | **4.84x** (Triton) |

### Configuration 2: Batch=1, SeqLen=4096, Heads=32, KVHeads=8, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 8.5682 | 8.5239 | **3.0899** | **2.77x** (Flash) |
| Backward | 14.2498 | **0.9140** | 7.7724 | **15.59x** (Triton) |
| Total | 22.5969 | **9.2463** | 10.6118 | **2.44x** (Triton) |

### Configuration 3: Batch=1, SeqLen=2048, Heads=40, KVHeads=8, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 3.3243 | 3.4601 | **1.7338** | **1.92x** (Flash) |
| Backward | 6.4576 | **0.8509** | 4.3455 | **7.59x** (Triton) |
| Total | 9.5391 | **4.0725** | 5.8773 | **2.34x** (Triton) |

---

## Multi-Query Attention (MQA)

### Configuration 1: Batch=2, SeqLen=2048, Heads=16, HeadDim=64

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 2.0782 | 1.2295 | **0.6677** | **3.11x** (Flash) |
| Backward | 3.4066 | **0.3761** | 1.5692 | **9.06x** (Triton) |
| Total | 4.9639 | **1.3632** | 2.0765 | **3.64x** (Triton) |

### Configuration 2: Batch=1, SeqLen=4096, Heads=32, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 8.2511 | 7.6663 | **2.6063** | **3.17x** (Flash) |
| Backward | 13.7692 | **0.8924** | 6.8854 | **15.43x** (Triton) |
| Total | 21.6910 | **8.3465** | 9.2544 | **2.60x** (Triton) |

### Configuration 3: Batch=1, SeqLen=2048, Heads=40, HeadDim=128

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 3.1056 | 3.0593 | **1.4199** | **2.19x** (Flash) |
| Backward | 6.0071 | **0.7934** | 3.8087 | **7.57x** (Triton) |
| Total | 8.9028 | **3.6751** | 5.0659 | **2.42x** (Triton) |

---

## Multi-Head Latent Attention

### Configuration 1: Batch=2, SeqLen=2048, Heads=32, HeadDim=128, LatentDim=512

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 4.4957 | 5.4143 | **1.7666** | **2.54x** (Flash) |
| Backward | 8.3772 | **0.9107** | 4.6803 | **9.20x** (Triton) |
| Total | 12.5536 | **6.1230** | 6.2137 | **2.05x** (Triton) |

### Configuration 2: Batch=1, SeqLen=4096, Heads=64, HeadDim=128, LatentDim=1024

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 17.8650 | 25.1916 | **6.6530** | **2.68x** (Flash) |
| Backward | 29.0787 | **2.8200** | 15.8775 | **10.31x** (Triton) |
| Total | 46.5593 | **27.6567** | 22.0984 | **2.11x** (Flash) |

### Configuration 3: Batch=1, SeqLen=2048, Heads=128, HeadDim=128, LatentDim=2048

| Pass | PyTorch (ms) | Triton (ms) | Flash (ms) | Best Speedup |
|------|--------------|-------------|------------|--------------|
| Forward | 13.8854 | 18.6263 | **9.0415** | **1.54x** (Flash) |
| Backward | 27.4436 | **6.2005** | 21.2844 | **4.43x** (Triton) |
| Total | 40.6924 | **24.5216** | 29.8241 | **1.66x** (Triton) |

---