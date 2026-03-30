# 从零到 79 TFLOPS：一个人用 Rust 写了一个裸金属 GPU 编译器，在 AMD 上匹敌 Triton

> *一周密集开发实录 — 从 GPU 每天硬挂 6 次到稳定输出 79.2 TFLOPS*

---

## 引言

如果我告诉你，一个人用纯 Rust 写的 GPU 编译器，在 AMD RX 7900 XTX 上跑出了 79.2 TFLOPS 的 bf16 GEMM 性能——**和 Triton on AMD 基本持平**——你会相信吗？

更关键的是：**零外部依赖**。不需要 HIP，不需要 ROCm 用户态库，不需要 Python，不需要 LLVM IR。整个系统只依赖 Linux 内核的 `/dev/kfd` 接口，直接和 GPU 硬件对话。

这篇文章记录了 T0-GPU 编译器在 2026 年 3 月 23 日至 29 日这一周里，从"每天 GPU 硬挂 6 次"到"稳定 79.2 TFLOPS"的完整历程。每一个 Bug 都是用硬挂和强制重启换来的教训。

---

## 一、为什么要造这个轮子

### 1.1 AMD GPU 上的困境

在 NVIDIA 的世界里，CUDA 生态成熟得令人羡慕。但在 AMD 这边，你的选项是：

- **ROCm/HIP**: 官方方案，但依赖栈极重（libhip、libhsakmt、ROCr、LLVM），版本兼容性是噩梦
- **Triton on AMD**: 性能不错，但背后仍是完整的 Python + LLVM + ROCm 栈，编译一次 kernel 要几秒
- **rocBLAS**: AMD 的手写 BLAS 库，性能最强，但是闭源的、预编译的、无法定制的

我想要的是一个**轻量、快速、可定制**的 GPU 编程方案。于是 T0-GPU 诞生了。

### 1.2 核心设计决策

T0 做了几个激进的决定：

1. **直接生成机器码** — 不经过 LLVM IR，直接输出 GFX1100 ISA 编码
2. **手工构建 ELF** — 不依赖 LLVM linker，自己构建 AMD HSA Code Object
3. **KFD 裸金属调度** — 绕过 HIP 运行时，直接写 AQL packet 到硬件队列
4. **纯 Rust 实现** — 零 C/C++ 依赖，零 FFI（除了 libc 的 mmap/ioctl）

这意味着从 Rust DSL 到 GPU 执行，整个路径上没有任何第三方库。dispatch 延迟低至 2.26μs（HIP 是 2.6μs）。

---

## 二、3 月 23 日：起点

### 2.1 已有基础

3 月 23 日时，T0 已经具备了基本能力：
- ISA 编码器覆盖了 GFX1100 的主要指令集（VOP1/VOP2/VOP3/SMEM/FLAT/WMMA/DS）
- KFD 运行时可以分配 VRAM、创建 AQL 队列、dispatch 内核
- 参数化 GEMM 生成器（gemm_gen.rs）已经在 WGP Mode + Split-K 下跑出了 67.3 TFLOPS 的峰值

在 **7/9 个矩阵尺寸上超越了 rocBLAS**，最高领先 97%（1024×1024×4096 上，T0 58.84 TF vs rocBLAS 29.94 TF）。

听起来很美好。但这些都是在特定配置下、用 WGP Mode 跑出来的。而 WGP Mode 后来被证明有严重的硬件兼容性问题。

### 2.2 T0 vs Triton：差距在哪？

3 月 24 日，我仔细分析了 T0 和 Triton on AMD 的差距。结论是：

| 维度 | Triton | T0 |
|------|--------|----|
| 依赖栈 | Python + LLVM + ROCm (数 GB) | 纯 Rust (~50K LOC) |
| 编译延迟 | 秒级 | 毫秒级 |
| GEMM 性能 | ~80 TFLOPS | ~67 TFLOPS (当时) |
| 通用性 | 广泛的 kernel DSL | 编译器还在建设中 |

性能差距约 15-20%。但 T0 的优势在于**极低的 dispatch 延迟**和**零依赖部署**。对于需要频繁调度小 kernel 的场景（如训练循环中的逐元素操作），T0 的 2μs dispatch 比 HIP 的 20μs 有巨大优势。

这时候，我决定全力推进编译器的 SSA 管线——从手写模板迁移到编译器自动代码生成。

---

## 三、3 月 25 日：地狱日

### 3.1 GPU 硬挂 × 6

3 月 25 日上午是整个项目最黑暗的时刻。

我启用了新写的 SSA 编译管线来编译 GEMM 内核，然后——

```
[73843.071840] sq_intr: error, type 2, sh 0, priv 1
[73843.072029] [gfxhub] page fault (ring:40 vmid:8)
[73845.072581] MES might be in unrecoverable state
[73848.122299] MODE1 reset
[73848.629535] VRAM is lost due to GPU reset!
```

GPU page fault → MODE1 reset → VRAM 丢失 → 强制重启。

一上午重启了 **6 次**。每次重启大约需要 3-5 分钟（等 GPU 恢复 + Linux 重新初始化 KFD）。

### 3.2 根因 #1：LICM 把指令插到了死区

第一个 Bug 在 LICM (Loop-Invariant Code Motion) 优化 Pass 中。

LICM 负责把循环不变量提升到循环外。但它用的是 `extend`——把指令追加到 block 末尾。问题是，block 末尾是 `s_branch` 终结器。所以被提升的指令落在了 `s_branch` **之后**，成了不可达的死代码。

但 GPU 不知道这是"死代码"——它的指令缓存会预取这些字节，如果其中包含非法的内存地址（比如 0x0），就会触发 page fault。

修复很简单，但代价是 6 次硬挂：

```rust
// BEFORE (broken): 插到末尾
block.ops.extend(hoisted_ops);

// AFTER (fixed): 插到 terminator 之前
block.ops.insert(block.ops.len() - 1, hoisted_op);
```

### 3.3 根因 #2：raw_asm 绕过了寄存器分配

第二个 Bug 更隐蔽。右移操作 `Shr` 的实现走了一条 `raw_asm` 路径，直接在生成的汇编中使用**虚拟寄存器号**而不是物理寄存器号。

虚拟寄存器号（比如 v153）在 regalloc 之后应该被映射到物理寄存器（比如 v12）。但 `raw_asm` 跳过了这个映射。结果就是 GPU 试图读取 v153——一个根本不存在的寄存器——然后 page fault。

**教训: raw_asm 是绕过 regalloc 的定时炸弹。**

### 3.4 KFD 运行时的三层防御

修完 Bug 后，我给 KFD 运行时加了三层防御，防止类似的灾难：

1. **SIGPIPE 忽略**: `cargo test | head -N` 中，`head` 关闭 stdin 会发 SIGPIPE 杀进程。进程死时 Drop 没跑，GPU 队列没清理，下次启动就挂
2. **KFD open 5 次重试**: GPU Reset 后 `/dev/kfd` 可能暂时不可用，需要等待恢复
3. **GPU 健康探针**: 分配 GTT buffer → 写入 `0xDEADBEEFCAFEF00D` → 读回验证。如果不一致，GPU 还没恢复

---

## 四、3 月 25-26 日：SSA 管线的 9 个根因 Bug

### 4.1 CSE 的 5 个坑

CSE (Common Subexpression Elimination) 是编译器优化中最基础的 Pass 之一。但在 GPU 编译器中，它有意想不到的陷阱。

**陷阱 1: barrier 不是透明的。** CSE 把 barrier 两侧的相同表达式合并了。但 barrier 意味着 LDS 内容可能已经被其他线程改写。合并之后读到的是旧数据。

**陷阱 2: 内联常量必须编码进 CSE key。** `v_add_f32 v0, v1, 0x42` 和 `v_add_f32 v0, v1, 0x43` 被当成了同一个表达式（因为 inline int 没编码进 key）。

**陷阱 3: MVal(u32::MAX) 是哨兵值。** SSA IR 中用 `u32::MAX` 表示"无值"，但 CSE 把它当普通值处理了。

我花了整个下午修这 5 个 Bug。每个都是"修完 → 跑测试 → 新的 err=inf → 再调试"的循环。

### 4.2 DCE 删掉了 K-loop 的指针步进

Dead Code Elimination 删掉了 K-loop 中 `ptr += stride` 的指针步进操作。

原因：SSA 形式中，`ptr_new = ptr_old + stride` 生成了一个新 MVal。但 DCE 看了看，发现这个新 MVal 在当前 block 中没有被 use（它被 use 是在下一次循环迭代中）。所以 DCE 把它标记为"死代码"删除了。

K-loop 永远用第一个 tile 的数据计算，GEMM 结果自然错得离谱。

修复：检测后向分支（backward branch = 循环），标记循环体内所有 VReg defs 为 root-live。

### 4.3 最终战果

经过两天 11+ 个会话的密集调试：

| 指标 | 3/25 开始 | 3/26 结束 |
|------|-----------|-----------|
| GPU 硬挂 | 6+ 次/天 | **0 次** |
| 测试通过率 | 0/9 | **10/10** |
| 最佳精度 | N/A | **err=3.81e-6** |
| 性能 (1024³) | N/A | **14.73 TF** |
| 优化等级 | 0 (全禁用) | **4 (全启用)** |

---

## 五、3 月 27-28 日：性能追击

### 5.1 Triton/rocBLAS ISA 抓包

为了理解 T0 和 rocBLAS 之间 ~25% 的性能差距（79 TF vs 105 TF），我用 `rocprofv2` 抓取了 Triton 和 rocBLAS 生成的 ISA 汇编。

关键发现：

- **rocBLAS 用 WGP Mode** — 2 个 CU 共享 128KB LDS，K-tile 可以更深
- **Triton 也用 WGP Mode** — 但它的 LLVM 后端在 AMD 上的代码质量不如 rocBLAS
- **T0 在 CU Mode** — 每个 CU 只有 64KB LDS，K-tile 深度受限

这就是差距的根源：不是编译器质量，而是**调度模式**的差异。

### 5.2 K-loop 回归修复

3 月 28 日遇到了一个经典的"修一个 Bug 引入另一个 Bug"的情况。

TileIR 的 K-loop 有两个分支：streaming mode（n_col_tiles > 4）和 bulk-load mode（n_col_tiles ≤ 4）。streaming mode 包含 `emit_interleaved_store`，但 bulk-load mode **完全没有 LDS store 逻辑**。

更糟的是，`lower_gemm` 末尾强制设置 `opt_level=4`，覆盖了用户的 `T0_OPT_LEVEL=0` 环境变量。DCE 正确地识别出 buffer_load 的结果没被写入 LDS（因为根本没有 store），于是删除了所有 load。K-loop 只读 prologue 的 stale LDS 数据。结果：16384/16384 个元素全错。

因果链：**缺失 store → VGPR 死亡 → DCE 删除 load → stale LDS → err=inf**

这个 Bug 告诉我一个铁律：**tile_ir 手工优化的 K-loop 必须 skip_optimize**。SSA 优化 Pass 对这种精心编排的内存访问模式有破坏性副作用。

---

## 六、3 月 29 日：三个突破

### 6.1 Gap Reclaim：15 个 VGPR 的胜利

SSA 寄存器分配器在分配 WMMA fragment（需要 8-VGPR 对齐）时，会跳过一些间隙。比如 HWM 在 v13，需要 8-aligned，就跳到 v16，v13-v15 就浪费了。

Gap Reclaim 把这些间隙 VGPRs 回收到 FreePool：

```rust
let gap_start = hwm;
let aligned = (hwm + 7) & !7;  // 8-aligned
for v in gap_start..aligned {
    free_pool.push(v);  // 回收间隙
}
```

效果：128×128 k32 从 254 VGPRs 降到 239，节省 15 个。k64 配置从 64 spills 降到 0。

这 15 个 VGPR 将用于后续的 ILP 优化（store interleave、prefetch）。

### 6.2 k48 Hang：一个 Bug 三行代码

128×128 tile_k=48 的配置无论如何都会 GPU hang，即使 0 spill。

根因出奇简单：cooperative load 的 tid→(row, col) 分解用了 bitwise AND/shift：

```rust
row = tid >> cpr_shift     // tid / chunks_per_row
col = tid & (cpr - 1)      // tid % chunks_per_row
```

**这只在 chunks_per_row 是 2 的幂时正确！**

k48: `chunks_per_row = 48×2/16 = 6`。6 不是 2 的幂。`6.trailing_zeros() = 1`。`tid=2 >> 1 = 1`，但正确答案是 `2/6 = 0`。

**tid=2 就已经算错了**。所有 128 个线程访问了错误的 GMEM 地址，LDS 数据完全混乱。

修复：一行 assert。

```rust
assert!(chunks_per_row.is_power_of_two());
```

### 6.3 WGP Mode：CWSR 的诅咒

为了追平 rocBLAS 的 WGP 性能，我尝试启用 WGP Mode。写了一个最小 probe 内核——只有 barrier + store，没有任何 GEMM 逻辑。

结果：GPU page fault → MODE1 reset → VRAM 丢失。

dmesg 显示：
```
sq_intr: error, type 2, priv 1  ← CWSR 特权 wave 错误
Faulty UTCL2 client: TCP (0x8)  ← 纹理缓存访问
PERMISSION_FAULTS: 0x5          ← 读写权限全拒
```

根因：**CWSR (Context Wave Save/Restore) 与 WGP Mode 不兼容**。CWSR buffer 的大小按 CU mode 计算（3072 waves × 12B），WGP mode 下 wave 布局不同，导致 CWSR 地址越界。

这是一个**驱动层面的问题**，不是编译器能解决的。我选择放弃 WGP mode，全力优化 CU mode。

---

## 七、最终成绩

### 7.1 性能对比

| 引擎 | 4096³ bf16 GEMM | 依赖栈 | Dispatch 延迟 |
|------|:---------------:|--------|:------------:|
| **T0** | **79.2 TFLOPS** | Rust only | **2.26 μs** |
| Triton | ~80 TFLOPS | Python+LLVM+ROCm | ~20 μs |
| rocBLAS | ~105 TFLOPS | ROCm (WGP mode) | ~20 μs |
| 理论峰值 | 123 TFLOPS | — | — |

T0 在 CU mode 下达到理论峰值的 **64%**，和 Triton 基本持平。

rocBLAS 的额外 25% 来自 WGP mode（2 CU 共享 128KB LDS，可以用更深的 K-tile）。如果 CWSR 兼容性问题解决，T0 也有望达到 90+ TFLOPS。

### 7.2 项目规模

```
T0-GPU: ~50,000 LOC Rust
├── T0 编译器 (34 文件): ~38,000 LOC
│   ├── SSA 管线: BlockDSL → SSA IR → 6-pass 优化 → RegAlloc
│   ├── GEMM 管线: TileIR → TileSSA → Lower → T0Kernel
│   └── 诊断工具: ISA 验证器 + HW Probe + K-loop 模拟器
├── ISA 编码器: 3,100 LOC (GFX1100 全指令集)
├── ELF 生成器: 1,400 LOC (AMD HSA Code Object)
└── KFD 运行时: 3,000 LOC (裸金属 GPU 控制)
```

### 7.3 13 条铁律

这一周的调试提炼出了 13 条铁律，每一条都是用 GPU 硬挂换来的：

1. LICM 必须 insert(len-1) — hoisted 指令放在 terminator 前
2. CSE key 必须含 opcode + MVal + inline 常量
3. CSE 必须在 barrier 处清空 seen table
4. DCE 必须处理 loop-carried deps
5. Scheduling 必须在 regalloc 之后
6. max_vgprs 不要人为限制 — GEMM 需要 200+
7. raw_asm 是绕过 regalloc 的定时炸弹
8. KFD VA 复用必须用 BufferPool
9. tile_ir 内核必须 skip_optimize
10. coop load chunks_per_row 必须是 2^n
11. CWSR 与 WGP mode 不兼容 (RX 7900 XTX, Linux 6.17)
12. VGPR 上限 254 — 255/256 触发 CWSR hang
13. SIGPIPE 必须 ignore — 防管道杀进程泄漏队列

---

## 八、反思与展望

### 8.1 为什么值得做

有人会问：rocBLAS 已经有了，Triton 也有了，为什么还要从零造一个？

三个原因：

1. **依赖栈的噩梦**。在 AMD 上，ROCm 版本更新经常破坏兼容性。一个只依赖 `/dev/kfd` 的方案免疫了所有这些问题。

2. **调度延迟**。对于 OCPA 注意力这种需要调度几十个小 kernel 的场景，2μs vs 20μs 的差距是 10 倍。乘以 100 次 dispatch，就是 0.2ms vs 2ms 的差距。在训练吞吐量上可能意味着 10-30% 的提升。

3. **可定制性**。需要一个特殊的 GEMM 变体（比如 causal mask 融合）？用 T0 的 BlockDSL 写一个就好。不需要等 AMD 把它加到 rocBLAS 里。

### 8.2 下一步

- **ILP 优化**: 利用 Gap Reclaim 回收的 15 个 VGPRs 做 store interleave 和 prefetch
- **Software Pipelining**: VMEM load 与 WMMA 计算完美重叠
- **WMMA 双链 ILP**: 2 条独立 WMMA 链交替执行，预期 +20%
- **目标**: CU mode 下冲击 90-100 TFLOPS

### 8.3 开源

项目开源在 GitHub:

🔗 **[github.com/GeisYaO/t0-gpu](https://github.com/GeisYaO/t0-gpu)**

MIT / Apache-2.0 双许可。

---

## 附录：一周时间线

| 日期 | 事件 | 成果 |
|------|------|------|
| 3/23 | 初始 GEMM 基线 | 67.3 TF peak, 7/9 超越 rocBLAS |
| 3/24 | T0 vs Triton 差距分析 | 确认可比性，定位为轻量 JIT |
| 3/25 AM | GPU 硬挂 6 次 | LICM + raw_asm + KFD 三层防御 |
| 3/25 PM | SSA RegAlloc + CSE 修复 | 5/10 → 8/10 → 10/10 PASS |
| 3/26 | 边界 Masking + BufferPool | 任意维度 GEMM, err=3.81e-6 |
| 3/27 | Triton/rocBLAS ISA 抓包 | 差距模型: CU 79 vs WGP 105 |
| 3/28 | K-loop 回归修复 | err=inf → 正确, skip_optimize 铁律 |
| 3/29 | Gap Reclaim + WGP + k48 | **79.2 TF**, CWSR 不兼容确认 |

---

*写于 2026 年 3 月 29 日，GPU 已经连续 4 天没有硬挂了。*
