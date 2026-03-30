# T0-GPU × Ignis 全景状态报告

> 📅 2026-03-23 审计

## 两个项目的关系

```
Ignis (自动微分框架)                T0-GPU (内核编译器)
┌─────────────────────┐           ┌─────────────────────┐
│ Tensor + Tape       │           │ DSL 前端 (Block IR)  │
│ ops/ (高层算子)      │──调用──→  │ Lowering (IR→内核)   │
│ nn/ (模型层)         │           │ math.rs (ISA 内核)   │
│ 训练循环 (data/opt)  │           │ compile (寄存器分配)  │
└────────┬────────────┘           │ asm_emitter (编码)   │
         │                        │ Code Object (ELF)   │
         └──── GpuRuntime ───────→│ KFD dispatch        │
                                  └─────────────────────┘
```

**Ignis** = "用什么算子、怎么组合、怎么训练"（PyTorch 对标）
**T0-GPU** = "算子内部的 GPU 代码怎么生成"（Triton 对标）

---

## T0-GPU 状态

**14,179 行 · 14 文件 · 66 测试**

### 编译器 pipeline

| 层 | 文件 | 行数 | 状态 |
|----|------|------|------|
| **DSL 前端** | `block_ir.rs` | 606 | ✅ PhantomData typed handles |
| **旧 DSL** | `dsl.rs` + `dsl_lower.rs` | 643 | ✅ Op 枚举路径，Ignis 生产使用 |
| **IR** | `ir.rs` | 662 | ✅ Op/VReg/SReg/Target |
| **Lowering** | `block_lower.rs` | 594 | ✅ elementwise/GEMM/reduce/cast |
| **JIT 缓存** | `block_jit.rs` | 140 | ✅ GraphSignature 去重 |
| **内核模板** | `math.rs` | 7,328 | ✅ 最大文件，所有 ISA 内核 |
| **GEMM 生成** | `gemm_gen.rs` | 1,125 | ✅ 参数化 WMMA tile |
| **编译器核心** | `compile.rs` | 1,007 | ✅ T0Kernel + Op→ISA |
| **汇编发射** | `asm_emitter.rs` | 833 | ✅ ISA→机器码字节 |
| **寄存器分配** | `regalloc.rs` | 329 | ✅ 线性扫描 |
| **调度特性** | `schedule.rs` | 376 | ✅ GFX1100Schedule |

### math.rs 中已实现的 ISA 内核

| 内核 | 用途 | GPU ✅ |
|------|------|--------|
| `elementwise_binary` | add/mul/sub | ✅ |
| `elementwise_unary` | neg/abs/relu | ✅ |
| `fused_elementwise` | 任意融合链 (scale→exp→...) | ✅ |
| `t0_reduce_scalar` | 全局 sum reduction | ✅ |
| `t0_silu_mul` | SiLU(gate) × up | ✅ |
| `t0_residual_add` | y += x (in-place) | ✅ |
| `t0_psi_inplace` | ψ 激活 | ✅ |
| `t0_rmsnorm_forward` | RMSNorm 前向 | ✅ |
| `t0_f32_to_bf16` / `bf16_to_f32` | 类型转换 | ✅ |
| `gemm_gen::build_gemm` | WMMA GEMM (多 tile) | ✅ |

### ❌ T0 缺失功能

| 功能 | 说明 |
|------|------|
| Row-wise reduction | Softmax/RMSNorm 的基础，只有 scalar reduce |
| GEMM epilogue 融合 | ReLU(X@W+b) 需要 2 个 kernel |
| 多架构后端 | 只有 GFX1100 |

---

## Ignis 状态

**5,452 行 · 28 文件 · 28 测试**

### 核心基础设施

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| **Tensor** | `tensor.rs` | 483 | ✅ shape/dtype/grad/buffer |
| **Tape** (自动微分) | `tape.rs` | 457 | ✅ forward 记录 + backward 遍历 |
| **GpuRuntime** | `gpu_context.rs` | 482 | ✅ KFD 封装 + 内核缓存 + type-safe dispatch |
| **测试套件** | `tests.rs` | 613 | ✅ 28 个端到端测试 |

### 算子完成度

| 算子 | 文件 | Forward | Backward | 实现方式 |
|------|------|---------|----------|---------|
| **add** | `add.rs` | ✅ GPU | ✅ GPU (identity grad) | T0 DSL |
| **scale** | `add.rs` | ✅ GPU | ✅ CPU | T0 DSL |
| **mul** | `add.rs` | ✅ GPU | ✅ CPU | T0 DSL |
| **sum** | `add.rs` | ⚠️ CPU | ✅ CPU (broadcast) | CPU fallback |
| **BF16 GEMM** | `bf16_matmul.rs` | ✅ GPU | ✅ CPU | T0 GEMM |
| **SiLU gate** | `silu.rs` | ✅ GPU | ✅ CPU | 手写 ISA |
| **RMSNorm** | `rmsnorm.rs` | ✅ GPU | ⚠️ CPU | 手写 ISA (fwd) |
| **Embedding** | `embedding.rs` | ✅ GPU | ✅ GPU | 手写 ISA |
| **Cross-Entropy** | `cross_entropy.rs` | ⚠️ CPU | ⚠️ CPU | CPU 全量 |
| **Softmax** | (无独立文件) | ❌ | ❌ | — |
| **ψ activation** | `psi_activation.rs` | ✅ GPU | ❌ | 手写 ISA |
| **OCPA attention** | `ocpa_attention.rs` | ✅ GPU | 🚧 | 手写 ISA |
| **shape ops** | `shape_ops.rs` | ✅ (view/cat) | ✅ | 零拷贝 |
| **Reshape/View** | `shape_ops.rs` | ✅ | ✅ | 元数据操作 |

### nn/ 高层模块

| 模块 | 文件 | 状态 |
|------|------|------|
| `Linear` | `nn/linear.rs` | ✅ forward (GEMM) + backward |
| `Embedding` | `nn/embedding.rs` | ✅ lookup + backward |
| `Model` | `nn/model.rs` | ✅ 参数收集 |
| `Transformer` | `nn/transformer.rs` | 🚧 结构定义，待完善 |

### 训练基础设施

| 模块 | 文件 | 状态 |
|------|------|------|
| LR Scheduler | `lr_scheduler.rs` | ✅ cosine + warmup |
| Grad Clipping | `grad_clip.rs` | ✅ |
| Loss Scaler | `loss_scaler.rs` | ✅ mixed precision |
| Buffer Pool | `buffer_pool.rs` | ✅ VRAM 池化 |
| Data Loader | `data_loader.rs` | ✅ tokenized 数据 |
| Tokenizer | `tokenizer.rs` | ✅ BPE |

---

## 测试覆盖

```
T0 单元测试:     29 (math.rs)
T0 编译测试:      3 (compile.rs)
T0 调度测试:      5 (schedule.rs)
T0 Block IR:    10 (block_ir.rs)
T0 Block Lower:  5 (block_lower.rs)
T0 Block JIT:    2 (block_jit.rs)
T0 GPU 端到端:   11 (gpu_tests.rs) [1 ignored]
T0 GEMM:         1 (gemm_gen.rs)
Ignis:          28 (tests.rs)
ISA 编码器:     11 (rdna3_asm.rs)
Code Object:     2 (rdna3_code_object.rs)
──────────────────
总计:          107 测试
```

---

## 总代码量

| 项目 | 行数 | 说明 |
|------|------|------|
| T0-GPU | 14,179 | 内核编译器 |
| Ignis | 5,452 | 自动微分框架 |
| ISA + Runtime | ~4,000 | rdna3_asm + kfd (根目录) |
| **总计** | **~23,600** | 纯 Rust |
