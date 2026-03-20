# 600 行 Rust 击败 AMD 官方 GEMM 库 — 裸金属 GPU 编程实战

*在 RX 7900 XTX 上的一次裸金属 GPU 编程冒险*

---

## 太长不看

我用 **600 行 Rust** 写了一个参数化 GEMM 内核生成器，在 **3 个矩阵尺寸上超越了 AMD 官方的 rocBLAS 库**，最高领先 **42%**。它通过 Linux KFD 驱动直接与 GPU 通信，**零外部依赖**。不用 HIP，不用 ROCm 运行时。只需要 `/dev/kfd` 和一点勇气。

| 矩阵 | 我的代码 | rocBLAS | 我 / rocBLAS |
|------|---------|---------|-------------|
| **1024³** | **34.5 TFLOPS** | **27.9 TFLOPS** | **🏆 124%** |
| **2048³** | **44.1 TFLOPS** | **36.7 TFLOPS** | **🏆 120%** |
| 512×1024×4096 | 44.3 TFLOPS | 45.9 TFLOPS | 97% |
| **1024×1024×4096** | **42.5 TFLOPS** | **29.9 TFLOPS** | **🏆 142%** |

代码已开源：[**github.com/GeisYaO/t0-gpu**](https://github.com/GeisYaO/t0-gpu)

rocBLAS 基线数据来源：PyTorch 2.9.1+rocm6.4，`torch.mm()` bf16，RX 7900 XTX 实测。

---

## 为什么要做这件事？

不是因为我想做一个 GPU 编译器。是因为**我被逼到了这一步**。

我在 RX 7900 XTX 上做自定义注意力机制的训练。最初的路线很正常——用 PyTorch + ROCm。然后掉进了一个又一个坑。

当时我面前有三条路：**A)** 退回 wgpu + 标准注意力分块——安全但 wgpu 不支持 bf16 WMMA，等于放弃 90% 的硬件算力；**B)** 修复 CubeCL 以支持 FA-2——但 GFX1100 这张卡连一个可用的 FA-2 渠道都没有，这不是修框架能解决的；**C)** 等待上游更新——可能等一年也等不到（事实上到今天，FA-2 仍然不支持 RDNA3）。

**我选了不存在的第四条：三条路都不走，从 ISA 开始全部自己写。** 从概率上说这个选择不是理性的，但对于当时我来说可能是唯一正确的——一个人做 GPU 编译器 + 运行时在行业里是团队级别的工作。

于是：

1. **PyTorch 不支持 GFX1100 使用 FlashAttention-2**。RDNA3 是消费级架构，官方 FA-2 只支持 CDNA（MI250/MI300）。我的 GPU 被排除在外。

2. **转向 Burn 框架 + CubeCL（Rust GPU 计算库）**。发现 CubeCL 的 bf16 支持有 bug，panic 在一连串的 `unwrap()` 上。我去修 CubeCL，提交了 uniformity analysis panic fix 的 PR。结果修完第一个 `unwrap()`，运行到下一行又是一个 `unwrap()` panic。一路追下去，发现 HIP 内存管理也有问题，又提交了 HIP allocator + GC 的 PR。**修别人的框架比自己写还累。**

3. **自己写 HIP 版 FlashAttention-2**。花了几周手写了 12,536 bytes 的 backward kernel——168 个 VGPR、56KB LDS、8 个 wave、9 个计算步骤。能跑了，但分析后发现 **WMMA 利用率只有 2%**。70% 的时间花在等内存——因为 Occupancy 只有 1（LDS 56KB 吃满了 64KB 上限），GPU 没有其他工作组可以切换，每个 `wait` 都是硬停顿。

4. **战略转型：放弃 FA-2，拥抱纯 GEMM**。既然 FlashAttention 的 fused kernel 在 RDNA3 上效率极低，那不如回到基础——把注意力拆成多个高效的 GEMM 调用。这就是后来 OCPA（正交分块纯矩阵注意力）的起点。

5. **开始手写 GFX1100 汇编**。既然框架都靠不住，那我直接跟硬件对话。用 `llvm-mc` 验证每条指令编码，一条一条地写 WMMA GEMM 内核。光是一个 bf16 butterfly 转置就修了 4 个 ISA 编码 bug（`v_perm_b32` 的 literal marker、cross-VGPR routing、XOR distance、selector byte order）。

6. **HIP 调度也有问题**。hipLaunchKernel 同步延迟 20μs，我的训练循环每步几百个小内核，全部浪费在调度上。于是**自己写了 KFD 运行时**——直接通过 `/dev/kfd` 往 AQL 队列写 dispatch packet。这一步也不轻松：连 AQL doorbell 的语义都搞错了（应该写 `write_ptr - 1` 而不是 `write_ptr`），还是参考了 tinygrad 的源码才修对的。

7. **手写算子太痛苦**。一个 GEMM 内核 300 行汇编，改一个 tile 尺寸要复制粘贴整个文件。于是写了**参数化代码生成器**——一个 `generate()` 函数覆盖所有变体。

回头看，这条路线荒谬得不可思议：**从 `import torch` 到手写 GPU ISA 汇编，中间经历了 Burn 框架、CubeCL 的 unwrap 连环 panic、12KB 的 FA-2 内核只有 2% WMMA 利用率、KFD doorbell 语义 bug...**

但每一步都是被逼的——不是我想跳过框架，是框架在 RDNA3 上跑不通。**当整个生态告诉你"请等下一个版本"，而你等不起的时候，唯一的选择就是自己动手。**

最终的产物就是 T0：一个纯 Rust 的 GPU 内核编译器 + 裸金属运行时。

## 技术栈

整个系统三层，全部用 Rust 实现：

```
┌─────────────────────────────────────────┐
│  gemm_gen.rs (600 行)                   │
│  参数化 GEMM 内核生成器                   │
│  auto_select(M, K, N) → 最优配置         │
├─────────────────────────────────────────┤
│  T0 编译器 (compile.rs + asm_emitter)    │
│  IR → GFX1100 ISA → AMD HSA ELF        │
├─────────────────────────────────────────┤
│  KFD 运行时 (kfd/mod.rs)                │
│  /dev/kfd → AQL 队列 → GPU              │
│  调度延迟: 2.26μs (HIP: 2.6μs)          │
└─────────────────────────────────────────┘
```

**没有 CUDA。没有 HIP。没有 ROCm 用户态。没有 Python。没有 C++。**

只有 Rust、LLVM 的汇编器（`llvm-mc`）和 Linux 内核。

## 三个关键优化

### 1. 合并内存加载（影响最大的优化）

GPU 内存以 128 字节缓存行工作。如果 32 个线程各自从随机位置加载，就会产生 32 次缓存行获取。如果它们从相邻地址加载，只需要 1 次。

我的做法：每个线程的加载地址计算为 `row = (thread_id × 16) / row_stride`，`col = (thread_id × 16) % row_stride`。这保证了相邻线程命中相邻的 16 字节块。

**效果**：2048×2048 从 28 → 38 TFLOPS（+35%）。

### 2. 单次调度 Split-K（最巧妙的优化）

小矩阵（256³、512³）没有足够的 tile 来填满 7900 XTX 的全部 96 个计算单元。用 64×64 的 tile，256³ 只会生成 4×4 = 16 个工作组，对应 96 个 CU。

Split-K 的解决方案是将 K 维度分割：不是一个工作组计算完整的点积，而是 4 个工作组各计算 1/4，最后求和。

关键设计：**在单次 dispatch 中完成**。我把 `tile_col` 和 `split_k_id` 编码到 `TGID.y` 中：

```
TGID.y = tile_col × split_k + split_k_id
tile_col = TGID.y >> log2(split_k)      // 编译时移位
split_k_id = TGID.y & (split_k - 1)     // 编译时掩码
```

当 `split_k = 1` 时，移位量为 0，所以 `tile_col = TGID.y`——非 split 情况零开销。

**效果**：512³ 从 6.2 → 10.1 TFLOPS（+63%）。

### 3. 参数化生成（可扩展的优化）

最初我手写了 6 个不同的 GEMM 内核——1800 行复制粘贴。后来我写了一个 `generate()` 函数，接受一个 `GemmConfig` 参数：

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

600 行覆盖了**所有** tile 尺寸（32×32 到 128×64）、K-unroll（16 或 32）和 split-K 因子（1/2/4/8）。自动选择函数为每种矩阵形状挑选最优配置：

```rust
let cfg = auto_select(m, k, n);
// 256³  → split_k=4, k32 (填满 CU)
// 1024³ → 64×64_k16 (均衡)
// 4096³ → 128×64_k32 (计算密度)
```

## 为什么能超越 rocBLAS？

说实话，我也不完全确定。我的推测：

1. **rocBLAS 用通用配置**：Tensile（rocBLAS 的内核生成器）为多种 GPU 优化通用方案。我的代码只针对一块 GPU，用手调的启发式规则。
2. **调度开销**：rocBLAS 经过 HIP 运行时，有额外开销。我的 AQL 队列写入直接到硬件。
3. **Swizzled grid 映射**：我交换 TGID.x 和 TGID.y，让 M 维度优先遍历，保持 X 矩阵在 L2 缓存中热。这对矩形矩阵特别有帮助。

## 诚实评估：真实训练场景

在你兴奋之前，需要一些背景。在实际的神经网络训练中（比如 dim=1024, hidden=4096, batch=2, seq=128 的 Transformer）：

| 操作 | 矩阵尺寸 (M×K×N) | T0 vs rocBLAS |
|------|-----------------|---------------|
| QKV 投影 | 256×1024×1024 | ~40% |
| **FFN gate/up** | **256×1024×4096** | **74%** |
| FFN down | 256×4096×1024 | ~74% |
| 反向传播 dW | 1024×256×4096 | ~142% 🏆 |

T0 赢的尺寸（1024³、2048³、1024×1024×4096）对应的是**权重梯度（dW）计算**和**大 batch 场景**。前向传播中最频繁的 GEMM 使用 M=128-256（batch × seq_len），这里 rocBLAS 仍然领先 2-3 倍。

**这意味着**：T0 在**反向权重梯度**方面有竞争力（这在大模型训练中占据大部分时间），但**小 M 前向 GEMM** 仍需优化。下一步的优化方向是专门的小 M tile（16×64 或 32×64）加上激进的 split-K 来填满 CU。

## 我学到了什么

- **GPU 编程 90% 是内存访问模式**。计算单元够快了；瓶颈几乎总是在喂数据。
- **LLVM-MC 非常有用**。我用 `echo 'v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[0:7]' | llvm-mc -mcpu=gfx1100 --show-encoding` 验证每一条 ISA 指令编码。这至少避免了 10 次 GPU 硬挂。
- **GPU 硬挂非常可怕**。当你写入错误的 GPU 地址时，整个系统冻结。没有优雅的错误。没有日志。只有黑屏和电源键。
- **AMD KFD 接口设计得其实不错**。尽管缺乏文档，ioctl API 很干净，AQL 队列规范很优雅。

## 自己试试

```bash
git clone https://github.com/GeisYaO/t0-gpu.git
cd t0-gpu
cargo run --example bench_gemm_sweep --features rocm --release
```

要求：AMD RDNA3 GPU、Linux 内核 5.15+、Rust、LLVM 17+。就这些。

---

*我在失业期间做了这个项目。如果你在招 GPU 编译器/运行时方面的人才，我的邮箱在 GitHub 个人主页上。*

*如果这个项目对你有启发，请给个 star ⭐ — 比你想象的更有帮助。*

---

**代码**：[github.com/GeisYaO/t0-gpu](https://github.com/GeisYaO/t0-gpu)
