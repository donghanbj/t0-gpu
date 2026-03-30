# GPU Buffer Pool (LRU Cache) 实验记录

## 日期
2026-03-26

## 目标
消除连续 GPU dispatch 间的 KFD VA 复用竞态，解决全套测试顺序运行时的硬挂和数值错误。

## 方法

### 根因分析

通过逐步隔离实验确认：

| 实验 | 结果 | 结论 |
|------|------|------|
| 100×300×200 单独 | ✅ err=1.53e-5 | 内核本身正确 |
| 10-case combo（原始） | ❌ 第9个 err=19.7 | 顺序执行有问题 |
| 禁用 bench loop | ❌ 仍然 err=19.7 | 不是 bench loop 累积 |
| c128→c32 2-case | ❌ 失败 | wg_size 不同触发 |
| c32→c32 2-case | ✅ 通过 | 少量 VA 回收不触发 |
| 反转顺序 c32→c128 | ✅ 通过 | 方向性问题 |
| 128³→100×300×200 | ❌ 任意 c128 前置都触发 | 不是特定维度 |

**根因**：`GpuBuffer::Drop` 调用 `UNMAP_MEMORY + FREE_MEMORY` 释放 VA 后，KFD driver 缺少全局 L2/TLB invalidation。下次 `ALLOC_MEMORY` 复用同一 VA 时，GPU 在新 dispatch 中读到旧的 L2 缓存数据。

### 修复：BufferPool (LRU Cache)

参考 tinygrad 的 `LRUAllocator` 方案：

- **buffer "释放"** → 放入 `HashMap<usize, Vec<GpuBuffer>>` 缓存（按 aligned_size 分组）
- **buffer 分配** → 先查 cache，有则 pop 复用；无则走 `device.alloc_vram()`
- **VA 映射保持不变** → 完全避免 UNMAP/FREE → 无 VA 竞态

## 结果

### 修复前

```
[combo] 100×300×200: err=1.97e1 bad=711  ← 第9个位置必错
[combo] 1000×768×512: 未到达（前面已失败）
```

### 修复后

```
[combo] 128³: err=3.81e-6 bad=0       ✅
[combo] 256³: err=9.54e-6 bad=0       ✅
[combo] 128×1024×512: err=9.54e-5 bad=0 ✅
[combo] 256×512×128: err=2.48e-5 bad=0 ✅
[combo] 512³: err=3.81e-5 bad=0       ✅
[combo] 1024³: err=9.92e-5 bad=0      ✅
[combo] 33×100×50: err=2.86e-6 bad=0  ✅
[combo] 129×65×17: err=9.54e-7 bad=0  ✅
[combo] 100×300×200: err=1.53e-5 bad=0 ✅
[combo] 1000×768×512: err=6.10e-5 bad=0 ✅
```

**10/10 全通过，1.62 秒完成。零硬挂。**

## 结论

1. **KFD VA 复用是裸金属 GPU 开发的重大陷阱** — 标准 HIP runtime 通过 AMDGPUMalloc 的内部缓存避免了这个问题
2. **Buffer Pool 是零成本修复** — 不增加任何 GPU 开销，反而减少 syscall 次数
3. **bench loop 可以安全恢复** — 之前因为 VA 竞态被禁用

## 后续

1. 全套独立测试的顺序运行仍需验证（独立测试不使用 recycle，buffer 仍然 Drop）
2. 考虑在 `GpuBuffer::Drop` 中自动检测是否有关联的 BufferPool，自动回收
3. 在 OCPA 训练循环中也使用 BufferPool
