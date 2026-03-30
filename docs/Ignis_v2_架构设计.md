# Ignis v2：T0 之上的训推一体框架 — 修订版

> **定位**：T0 编译器/运行时之上的 PyTorch-API 层。
> 训练与推理共享同一模型定义。不生成 ISA——全部委托 T0。
> GPU 计算全 BF16 + F32 累加，CPU 验证用 F32。

---

## 历史沿革

```
原始 Ignis (严重丢失)
  ├── GpuRuntime → ✅ 已被 T0 吸收
  ├── ISA 内核生成 → ✅ 已被 T0 DSL 编译器替代
  ├── Ops (matmul, OCPA, silu...) → ❌ 严重丢失
  └── 启发: PyTorch 动态图 + vLLM 推理架构

当前 Ignis ⊂ T0 项目
  ├── Tensor + Tape (核心 autodiff) ✅
  ├── 4 个基础 Op (add, mul, scale, relu) ✅
  └── 缺失: matmul, CE, silu, OCPA, 优化器
```

## 分层架构

```
用户代码:  model.forward(&x) → loss.backward() → opt.step()
─────────────────── Ignis ──────────────────────────
 Tensor(bf16/f32)  │  Tape  │  Module/Optim  │  InferEngine
─────────────────── T0 接口 ─────────────────────────
 ensure_kernel_t0() → dispatch() → wait()
─────────────────── T0 编译器 ───────────────────────
 DSL/IR → RegAlloc → ISA → ELF    │  math.rs  │  gemm_gen.rs
─────────────────── KFD 裸金属 ──────────────────────
 /dev/kfd + AQL Queue + VRAM mmap
```

---

## BF16 策略

| 场景 | 精度 | 说明 |
|------|------|------|
| **GPU compute** | BF16 input + F32 accumulation | WMMA 天然模式 `v_wmma_f32_16x16x16_bf16` |
| **Master weight** | F32 | 参数更新精度 |
| **梯度** | F32 (WMMA output) | 反向用 F32 梯度 |
| **CPU 验证** | 全 F32 | 测试正确性基准 |

Tensor 需支持：
- `DType::F32` — master weight / 梯度 / CPU 验证
- `DType::BF16` — GEMM compute 输入
- `ensure_bf16_cache()` — forward 时 F32 → BF16 (T0 `f32_to_bf16` 内核)

---

## Op 体系与内核来源

### T0 已有内核 (math.rs + gemm_gen.rs)

| T0 内核 | Ignis Op | 状态 |
|---------|---------|------|
| `elementwise_binary(Add)` | add(a, b) | ✅ 已接入 |
| `elementwise_mul` (build) | mul(a, b) | ✅ 已接入 |
| `t0_residual_add` | grad accumulate | ✅ 已接入 |
| RMSNorm fwd+bwd | rmsnorm(x, γ) | ✅ 已接入 |
| transpose f32/bf16 | .t() | ✅ 已接入 |
| f32⇆bf16 | 类型转换 | ✅ 可用 |
| **gemm_gen (Y=X@W^T)** | **matmul forward** | **需扩展 backward** |
| elementwise(SiLU) | silu(x) | 需包装 |
| softmax_ce_loss | cross_entropy | 需从 examples/ 包装 |

### 需要扩展 gemm_gen.rs

当前 `gemm_gen.rs` 只支持 `Y = X @ W^T` (一种转置组合)。
GEMM backward 需要：

| 反向计算 | 数学公式 | 转置需求 |
|---------|---------|---------|
| dX | `dX = dY @ W` | NN (都不转置) |
| dW | `dW = X^T @ dY` | TN (左转置) |

> [!IMPORTANT]
> 扩展 `gemm_gen.rs` 支持 `trans_a` + `trans_b` 参数组合：
> - **NT** (当前): Y = X @ W^T
> - **NN** (新增): dX = dY @ W
> - **TN** (新增): dW = X^T @ dY

### 保留的手写内核 (examples/kernels/)

| 文件 | 内容 | 状态 |
|------|------|------|
| `ocpa_forward_intra.rs` (27KB) | chunk 内前向 | 可直接包装 |
| `ocpa_backward_intra.rs` (22KB) | chunk 内后向 | 可直接包装 |
| `ocpa_state_update.rs` (11KB) | S = S + K^T V | 可直接包装 |
| `softmax_ce_loss.rs` (15KB) | Softmax+CE | 可直接包装 |

### OCPA 缺失的内核 (需后续重建)

OCPA 完整 9-step pipeline 缺失：
- `ocpa_forward_inter` — O_inter = Q @ S
- `ocpa_prefix_sum` — S 前缀和（scan）
- `ocpa_backward_inter_dq/dk/dv` — 跨块反向
- `ocpa_reverse_prefix_sum` — dS 逆前缀和
- `ocpa_denom_norm` — 输出归一化

> 数学公式完整保留在 `docs/OCPA_正交分块纯矩阵注意力架构.md`

---

## 实施路线（修订）

### Phase 1：GEMM backward + E2E Training (当前优先)

**目标**：Embedding → Linear → ReLU → CE → SGD，50步 loss 下降

1. 扩展 `gemm_gen.rs` 支持 NN/TN 转置组合
2. 实现 `matmul()` Op — forward (NT GEMM) + backward (NN + TN GEMM)
3. 包装 `softmax_ce_loss` 为 `cross_entropy()` Op
4. 实现 SGD optimizer (CPU 回退)
5. E2E 测试验证

### Phase 2：Transformer 层

1. SiLU gate Op (T0 elementwise SiLU)
2. Module trait + Linear + Embedding
3. TransformerLayer (RMSNorm → Linear-Attn → FFN)
4. AdamW optimizer (T0 `adamw_1d` 内核)

### Phase 3：OCPA 完整管线

1. 包装保留的 4 个手写内核
2. 用 T0 DSL 重建缺失的 5 个内核
3. OCPA attention Op（原子 9-step）
4. Ada-GLAM optimizer

### Phase 4：推理引擎

1. KV cache 管理
2. 自回归 generate()
3. 利用 KFD ~1μs 调度延迟

---

## 已有资产清单

### 设计文档 (docs/)
- `OCPA_正交分块纯矩阵注意力架构.md` — 完整前向+反向公式
- `纯GEMM注意力与状态空间模型数学原理.md` — 数学基础
- `ignis_architecture.md` — 原始框架架构
- `ignis_completeness_report.md` — 完整性审查

### T0 编译器+运行时
- `gemm_gen.rs` — WMMA GEMM 生成器 (67.3 TFLOPS, 7/9 超 rocBLAS)
- `math.rs` — ~7000 行内核库
- `kfd/mod.rs` — KFD 裸金属运行时

### 当前 Ignis (opensource/src/ignis/)
- `tensor.rs` — Tensor + grad + tape_node
- `tape.rs` — Tape-based autodiff + gpu_accumulate_grad
- `gpu_context.rs` — GpuRuntime + ensure_kernel_t0 + dispatch
- `ops/add.rs` — add, mul, scale, sum, relu
- `ops/rmsnorm.rs` — RMSNorm fwd+bwd
- `nn/linear.rs` — Linear 层
- `nn/embedding.rs` — Embedding 层
- `tests.rs` — 15/17 通过
