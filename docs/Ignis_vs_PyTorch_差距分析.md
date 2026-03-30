# Ignis vs PyTorch 完整性对照分析

> 目标：系统梳理 PyTorch 的核心能力，逐项对照 Ignis 现状，明确缺什么、有什么、优先补什么。

---

## 一、torch.Tensor — 张量核心

### PyTorch 能力

| 功能 | PyTorch | 说明 |
|------|---------|------|
| 多维数组 | `torch.Tensor` | shape, dtype, device, stride |
| DType | float32, float16, bfloat16, int8, int32, int64, bool... | 20+ 种 |
| Device | CPU / CUDA / MPS | 透明设备切换 |
| 内存布局 | contiguous / stride-based views | 零拷贝 reshape/transpose |
| 运算符重载 | +, -, *, /, @, **, //, %, 比较, 索引 | 完整 Python 运算符 |
| 方法语法 | `.matmul()`, `.sum()`, `.mean()`, `.t()`, `.reshape()`, `.view()`, `.contiguous()`, `.to()`, `.clone()`, `.detach()` | 100+ 方法 |
| 广播 | 自动形状广播 | NumPy 规则 |
| 索引与切片 | `x[0, :, 2:5]`, 高级索引, mask 索引 | 完整 |
| 原地操作 | `.add_()`, `.mul_()`, `.zero_()` | 尾缀 `_` |
| 梯度 | `.requires_grad`, `.grad`, `.grad_fn`, `.backward()` | autograd 核心 |
| 序列化 | `torch.save()`, `torch.load()` | checkpoint |

### Ignis 现状 (tensor.rs, 14KB)

| 功能 | 状态 | 说明 |
|------|------|------|
| 多维数组 | ✅ | `Arc<GpuBuffer>` + shape |
| DType | ⚠️ 部分 | F32, BF16 两种 |
| Device | ✅ 仅 GPU | KFD 裸金属，无 CPU tensor |
| 内存布局 | ❌ | 无 stride，reshape 可能需拷贝 |
| 运算符重载 | ❌ 缺失 | 无 `+`, `*`, `@` 重载 |
| 方法语法 | ❌ 缺失 | 无 `.matmul()`, `.sum()` 等 |
| 广播 | ❌ | 要求形状完全匹配 |
| 索引与切片 | ⚠️ 基础 | shape_ops 有 slice，无高级索引 |
| 原地操作 | ⚠️ 部分 | `t0_residual_add` 做梯度累加 |
| 梯度 | ✅ | `grad`, `requires_grad`, `tape_node` |
| 序列化 | ❌ | 无 save/load |

### 🔴 关键缺失
- **运算符重载**：用户不能写 `a + b`，必须 `ops::add::add(&a, &b, &dev)`
- **广播**：标量 + 张量、不同维度张量运算
- **stride-based view**：transpose 需要实际拷贝数据

---

## 二、torch.autograd — 自动微分

### PyTorch 能力

| 功能 | 说明 |
|------|------|
| 动态图 | 每次 forward 构建新图 |
| `backward()` | 反向传播，支持 `retain_graph` |
| `torch.no_grad()` | 推理模式，不记录 |
| `grad_fn` 链 | 每个 tensor 知道自己的创建 op |
| 高阶梯度 | `torch.autograd.grad()` 支持二阶导 |
| 梯度累加 | `loss.backward()` 累加到 `.grad` |
| `detach()` | 从计算图脱离 |
| 自定义函数 | `torch.autograd.Function` |
| 梯度检查 | `torch.autograd.gradcheck()` |

### Ignis 现状 (tape.rs, 17KB)

| 功能 | 状态 | 说明 |
|------|------|------|
| 动态图 | ✅ | Tape-based，每次 forward 新图 |
| `backward()` | ✅ | `Tape::backward()` 反向遍历 |
| `no_grad()` | ✅ | `Tape::no_grad()` → `NoGradGuard` |
| grad_fn 链 | ✅ | `TapeNode` 存储 backward_fn |
| 高阶梯度 | ❌ | 不支持 |
| 梯度累加 | ✅ | `gpu_accumulate_grad()` via `t0_residual_add` |
| `detach()` | ✅ | `tensor.detach()` |
| 自定义函数 | ✅ | 闭包 `Box<dyn FnOnce(...)>` |
| 梯度检查 | ⚠️ | `test_numerical_gradient_check` 存在但有 bug |

### ✅ 基本完整
autograd 是 Ignis 最完整的模块。主要缺失：高阶梯度（非必要）。

---

## 三、torch.nn.functional — 算子层

### PyTorch 核心算子 vs Ignis

#### 3.1 逐元素操作

| PyTorch | Ignis ops | 状态 |
|---------|-----------|------|
| `F.relu` | shape_ops.rs (316行) | ✅ 有 relu |
| `F.silu` / `F.gelu` | silu.rs (130行) | ✅ SiLU+gate |
| `F.sigmoid` | shape_ops.rs | ⚠️ 需检查 |
| `F.softmax` | shape_ops.rs | ✅ 有 softmax |
| `torch.neg` / `torch.sub` | shape_ops.rs | ✅ neg, sub |
| `torch.mean` | shape_ops.rs | ✅ mean |
| `torch.add` | add.rs (362行) | ✅ |
| `torch.mul` | add.rs | ✅ elementwise_mul |
| `torch.sum` | add.rs | ✅ (CPU fallback) |
| `torch.scale` / `*scalar` | add.rs | ✅ (CPU fallback) |
| `F.dropout` | ❌ | **缺失** |
| `torch.exp` / `torch.log` | ❌ | **缺失** |
| `torch.clamp` | ❌ | **缺失** |
| `torch.pow` | ❌ | **缺失** |
| `torch.sqrt` | ❌ | **缺失** |
| `torch.abs` | ❌ | **缺失** |
| `torch.where` / 条件选择 | ❌ | **缺失** |

#### 3.2 线性代数 / 矩阵操作

| PyTorch | Ignis ops | 状态 |
|---------|-----------|------|
| `F.linear` / `torch.mm` / `torch.matmul` | bf16_matmul.rs (333行) | ⚠️ **存在但需验证** |
| `torch.transpose` | shape_ops.rs | ✅ |
| `torch.reshape` / `torch.view` | shape_ops.rs | ✅ |
| `torch.cat` / `torch.stack` | ❌ | **缺失** |
| `torch.split` / `torch.chunk` | ❌ | **缺失** |
| `torch.einsum` | ❌ | **缺失** |
| `torch.bmm` (batch matmul) | ❌ | **缺失** |

#### 3.3 归一化

| PyTorch | Ignis ops | 状态 |
|---------|-----------|------|
| `F.rms_norm` | rmsnorm.rs (146行) | ✅ fwd+bwd |
| `F.layer_norm` | ❌ | 缺失（可用 RMSNorm 替代） |
| `F.batch_norm` | ❌ | 缺失 |
| `F.group_norm` | ❌ | 缺失 |

#### 3.4 损失函数

| PyTorch | Ignis ops | 状态 |
|---------|-----------|------|
| `F.cross_entropy` | cross_entropy.rs (189行) | ⚠️ **存在但需验证** |
| `F.mse_loss` | ❌ | **缺失** |
| `F.l1_loss` | ❌ | **缺失** |
| `F.nll_loss` | ❌ | 缺失 |
| `F.binary_cross_entropy` | ❌ | 缺失 |

#### 3.5 注意力

| PyTorch | Ignis ops | 状态 |
|---------|-----------|------|
| `F.scaled_dot_product_attention` | ❌ | 缺失 (有 OCPA 替代) |
| OCPA 注意力 | ocpa_attention.rs (585行) | ⚠️ **存在但内核严重缺失** |

#### 3.6 嵌入

| PyTorch | Ignis ops | 状态 |
|---------|-----------|------|
| `F.embedding` | embedding.rs (104行) | ✅ gather+scatter_add |

---

## 四、torch.nn — 模型层

### PyTorch 核心层 vs Ignis

| PyTorch | Ignis nn/ | 状态 |
|---------|-----------|------|
| `nn.Module` | mod.rs (Module trait) | ✅ |
| `nn.Parameter` | mod.rs (Parameter struct) | ✅ |
| `nn.Linear` | linear.rs (86行) | ✅ f32 master + bf16 WMMA |
| `nn.Embedding` | embedding.rs (91行) | ✅ |
| `nn.RMSNorm` | rmsnorm op wrapping | ✅ |
| `nn.LayerNorm` | ❌ | 缺失 |
| `nn.TransformerEncoder` | transformer.rs (150行) | ✅ TransformerLayer |
| 完整语言模型 | model.rs (108行) | ✅ LanguageModel |
| `nn.Sequential` | ❌ | **缺失** |
| `nn.ModuleList` | ❌ | **缺失** |
| `nn.Dropout` | ❌ | **缺失** |
| `nn.Conv1d/2d` | ❌ | 缺失（LLM 不需要） |
| `nn.LSTM/GRU` | ❌ | 缺失 |

---

## 五、torch.optim — 优化器

| PyTorch | Ignis | 状态 |
|---------|-------|------|
| `optim.SGD` | ❌ | **缺失**（原有 CPU fallback 丢失） |
| `optim.Adam` | ❌ | **缺失** |
| `optim.AdamW` | ❌ | **缺失**（T0 有 `adamw_1d` 内核） |
| `optim.lr_scheduler` | lr_scheduler.rs (87行) | ✅ CosineWarmup + ConstantLR |
| Ada-GLAM (自研) | ❌ | **缺失**（T0 有内核） |

---

## 六、训练基础设施

| PyTorch | Ignis | 状态 |
|---------|-------|------|
| `DataLoader` | data_loader.rs (3.7KB) | ✅ .bin + text, shuffle |
| `GradScaler` (AMP) | loss_scaler.rs (3.4KB) | ✅ 动态 loss scaling |
| 梯度裁剪 | grad_clip.rs (2.5KB) | ✅ L2 全局裁剪 |
| `BufferPool` | buffer_pool.rs (2.1KB) | ⚠️ 有 bug |
| Tokenizer | tokenizer.rs (5.8KB) | ✅ BPE + Vocab |
| Checkpoint (save/load) | checkpoint.rs (stub) | ❌ **仅占位** |

---

## 七、PyTorch 有但 Ignis 不需要的

| PyTorch 功能 | 为什么 Ignis 不需要 |
|-------------|-------------------|
| CPU Tensor | 裸金属项目，GPU only |
| CUDA streams | KFD AQL Queue 替代 |
| torch.compile / JIT | T0 编译器替代 |
| torch.distributed | 单卡项目 |
| Conv/Pool 层 | LLM 不需要 |
| ONNX 导出 | 自有格式 |
| C++ 扩展 | 纯 Rust |

---

## 八、总结矩阵

| 模块 | PyTorch 功能数 | Ignis ✅ | Ignis ⚠️需验证 | Ignis ❌缺失 |
|------|------------|---------|-------------|-----------|
| **Tensor** | ~15 | 5 | 2 | 8 |
| **Autograd** | ~10 | 7 | 1 | 2 |
| **逐元素 Op** | ~17 | 9 | 1 | 7 |
| **线性代数 Op** | ~7 | 3 | 1 | 3 |
| **归一化** | ~4 | 1 | 0 | 3 |
| **损失函数** | ~5 | 0 | 1 | 4 |
| **NN 层** | ~12 | 5 | 0 | 7 |
| **优化器** | ~5 | 1 | 0 | 4 |
| **训练Infra** | ~6 | 4 | 1 | 1 |

---

## 九、重建优先级

### P0 — 不修就不能训练

| 缺失项 | 阻断影响 | 修复方式 |
|--------|---------|---------|
| 运算符重载 (+, *, @) | 用户 API 不可用 | 实现 `std::ops::Add/Mul` for Tensor |
| matmul fwd+bwd **验证** | E2E 训练必须 | 验证 bf16_matmul.rs 是否可用 |
| cross_entropy **验证** | 损失函数必须 | 验证 cross_entropy.rs 是否可用 |
| SGD/AdamW 优化器 | 参数更新必须 | 最简 CPU SGD + T0 AdamW 内核 |
| scale/sum **GPU 路径修复** | CPU fallback 太慢 | 修复 T0 ISA 内核 |

### P1 — 训练能跑但不好用

| 缺失项 | 影响 | 修复方式 |
|--------|------|---------|
| dropout | 正则化 | 掩码 + scale 内核 |
| Sequential / ModuleList | 模型构建便利 | Rust Vec<Box<dyn Module>> |
| checkpoint save/load | 断点续训 | 序列化 f32 参数 |
| mse_loss | 回归任务 | GPU elementwise |
| cat / stack | 张量拼接 | 内存拷贝 |
| 广播 | 便利性 | 自动 expand |

### P2 — 完善体验

| 缺失项 | 影响 |
|--------|------|
| exp/log/sqrt/abs/pow | 数学完整性 |
| stride-based view | 零拷贝性能 |
| 高级索引 | API 完整性 |
| OCPA 完整 9-step | OCPA 训练 |
| 推理引擎 | 部署 |
