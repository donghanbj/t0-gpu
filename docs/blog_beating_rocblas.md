# How I Beat AMD's rocBLAS with 600 Lines of Rust — and Zero Dependencies

*A bare-metal GPU programming story on the RX 7900 XTX*

---

## TL;DR

I wrote a parameterized GEMM kernel generator in **600 lines of Rust** that beats AMD's official rocBLAS library on **3 out of 10 matrix sizes** — by up to **42%**. It runs on bare metal through the Linux KFD driver, with **zero external dependencies**. No HIP. No ROCm runtime. Just `/dev/kfd` and a dream.

| Matrix | My Code | rocBLAS | Mine / rocBLAS |
|--------|---------|---------|----------------|
| **1024³** | **34.5 TFLOPS** | **27.9 TFLOPS** | **🏆 124%** |
| **2048³** | **44.1 TFLOPS** | **36.7 TFLOPS** | **🏆 120%** |
| 512×1024×4096 | 44.3 TFLOPS | 45.9 TFLOPS | 97% |
| **1024×1024×4096** | **42.5 TFLOPS** | **29.9 TFLOPS** | **🏆 142%** |

The code is open source: [**github.com/GeisYaO/t0-gpu**](https://github.com/GeisYaO/t0-gpu)

---

## Why Would Anyone Do This?

I'm building a custom neural network training system for my research. The standard path would be: install ROCm (2+ GB), use HIP, call rocBLAS, done. But I ran into three problems:

1. **Dispatch latency**: HIP's kernel launch takes ~20μs synchronous. When you're launching hundreds of tiny kernels per training step, that overhead adds up to seconds.
2. **Dependency hell**: ROCm versions break constantly. My Ubuntu 24.04 setup needed specific ROCm 7.x builds, and the driver/runtime version matrix was a nightmare.
3. **Black box**: When my GPU hangs (which happens a lot in bare-metal development), HIP gives you nothing. With KFD, I can at least see the AQL queue state.

So I decided: what if I just... talk to the GPU directly?

## The Stack

The whole system is three layers, all in Rust:

```
┌─────────────────────────────────────────┐
│  gemm_gen.rs (600 LOC)                  │
│  Parameterized GEMM kernel generator    │
│  auto_select(M, K, N) → optimal config  │
├─────────────────────────────────────────┤
│  T0 Compiler (compile.rs + asm_emitter) │
│  IR → GFX1100 ISA → AMD HSA ELF        │
├─────────────────────────────────────────┤
│  KFD Runtime (kfd/mod.rs)               │
│  /dev/kfd → AQL Queue → GPU             │
│  Dispatch latency: 2.26μs (vs HIP 2.6) │
└─────────────────────────────────────────┘
```

**No CUDA. No HIP. No ROCm userspace. No Python. No C++.**

Just Rust, LLVM's assembler (`llvm-mc`), and the Linux kernel.

## The Three Optimizations That Made the Difference

### 1. Coalesced Memory Loading (the biggest win)

The single most impactful optimization was fixing memory access patterns. GPU memory works in 128-byte cache lines. If 32 threads each load from random locations, you get 32 cache line fetches. If they load from adjacent addresses, you get 1.

My trick: each thread computes its load address as `(thread_id * 16) / row_stride` for the row and `(thread_id * 16) % row_stride` for the column. This guarantees adjacent threads hit adjacent 16-byte chunks.

**Result**: 2048×2048 went from 28 → 38 TFLOPS (+35%).

### 2. Single-Dispatch Split-K (the clever one)

Small matrices (256³, 512³) don't have enough tiles to fill all 96 Compute Units on the 7900 XTX. With a 64×64 tile, 256³ only generates 4×4 = 16 workgroups for 96 CUs.

Split-K solves this by dividing the K dimension: instead of one workgroup computing the full dot product, 4 workgroups each compute 1/4 and the results are summed.

The key insight: **do it in a single dispatch**. I pack `tile_col` and `split_k_id` into `TGID.y`:

```
TGID.y = tile_col * split_k + split_k_id
tile_col = TGID.y >> log2(split_k)      // compile-time shift
split_k_id = TGID.y & (split_k - 1)     // compile-time mask
```

When `split_k = 1`, the shift is 0, so `tile_col = TGID.y` — zero overhead for non-split cases.

**Result**: 512³ went from 6.2 → 10.1 TFLOPS (+63%).

### 3. Parameterized Generation (the scalable one)

Instead of hand-writing 6 different GEMM kernels (which I actually did initially — 1800 lines of copy-paste), I wrote a single `generate()` function that takes a `GemmConfig`:

```rust
let cfg = GemmConfig {
    tile_m: 64, tile_n: 64, tile_k: 32,
    wg_size: 64,
    use_lds: true, double_buffer: true,
    split_k: Some(4),
};
let kernel = generate(&cfg);
let elf = kernel.compile(Target::GFX1100)?;
```

600 lines cover **every** tile size (32×32 to 128×64), K-unroll (16 or 32), and split-K factor (1/2/4/8). The auto-select function picks the best config for each matrix shape:

```rust
let cfg = auto_select(m, k, n);
// 256³  → split_k=4, k32 (fill CUs)
// 1024³ → 64×64_k16 (balanced)
// 4096³ → 128×64_k32 (compute density)
```

## Why Does It Beat rocBLAS?

Honestly, I'm not 100% sure. My best theories:

1. **rocBLAS uses generic configs**: Tensile (rocBLAS's kernel generator) optimizes for the general case across many GPUs. My code targets exactly one GPU with hand-tuned heuristics.
2. **Dispatch overhead**: rocBLAS goes through the HIP runtime, which adds overhead. My AQL queue writes are direct to hardware.
3. **Swizzled grid mapping**: I swap TGID.x and TGID.y to iterate M-dimension first, keeping the X matrix hot in L2 cache. This specifically helps rectangular matrices where rocBLAS might not have the best mapping.

## What I Learned

- **GPU programming is 90% memory access patterns**. The compute units are fast enough; the bottleneck is almost always feeding them data.
- **LLVM-MC is incredibly useful**. I verify every ISA instruction encoding with `echo 'v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[0:7]' | llvm-mc -mcpu=gfx1100 --show-encoding`. This saved me from at least 10 GPU hard-hangs.
- **GPU hard-hangs are terrifying**. When you write to the wrong GPU address, the entire system freezes. No graceful error. No log. Just a black screen and a power button.
- **The AMD KFD interface is actually well-designed**. Despite the lack of documentation, the ioctl API is clean and the AQL queue spec is elegant.

## Try It Yourself

```bash
git clone https://github.com/GeisYaO/t0-gpu.git
cd t0-gpu
cargo run --example bench_gemm_sweep --features rocm --release
```

Requirements: AMD RDNA3 GPU, Linux kernel 5.15+, Rust, LLVM 17+. That's it.

---

*I built this while unemployed, working from home. If you're hiring for GPU compiler/runtime work, my email is in my GitHub profile.*

*If this resonates with you, star the repo ⭐ — it helps more than you'd think.*

---

**Discussion**: [Hacker News](#) | [Reddit r/rust](#) | [知乎](#)

**Code**: [github.com/GeisYaO/t0-gpu](https://github.com/GeisYaO/t0-gpu)
