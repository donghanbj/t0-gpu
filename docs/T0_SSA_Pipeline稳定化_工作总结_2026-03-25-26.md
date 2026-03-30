# T0 编译器 3 月 25-26 日完整工作小结

> 横跨 **11+ 会话**，两天密集开发。从"GPU 反复硬挂" → "10/10 GEMM 全通过、零硬挂"。

---

## 一、总体战果

| 指标 | 3.25 开始时 | 3.26 结束时 |
|------|-----------|-----------|
| GPU hard hang | 频繁（6+ 次/天） | **0 次** |
| test 通过率 | 0/9（全 hang） | **10/10 PASS** |
| 最佳精度 | N/A | **err=3.81e-6** |
| 1024³ 性能 | N/A | **14.73 TF** |
| opt_level | 0（全禁用） | **4（全启用）** |
| 任意维度 | ❌ 仅对齐 | ✅ 33×100×50 ~ 1000×768×512 |
| 测试稳定性 | 每次跑都可能挂 | **combo 1.57s 稳定复现** |

---

## 二、修复时间线

### 3 月 25 日上午 — GPU 硬挂根治

#### 修复 #1：LICM 死代码 + Shr raw_asm
- **现象**：`test_gpu_gemm_tn` → GPU page fault at 0x0 → 6 次强制重启
- **根因 A**：LICM 将 hoisted 指令 `extend` 到 block 末尾 → 放在 `s_branch` 之后 → 不可达死代码
- **根因 B**：`get_vreg(rhs)` 覆写 `val_map` → Shr 走 `raw_asm` 路径 → 虚拟寄存器号绕过 regalloc
- **修复**：`insert(len-1)` + 缓存 `rhs_val_precheck`

#### 修复 #2：KFD 运行时三层防御
- SIGPIPE 忽略（防管道杀进程）
- KFD open 5 次重试（覆盖 MODE1 reset 恢复窗口）
- GPU 健康探针（GTT buffer 写入→读回验证）

### 3 月 25 日下午 — SSA 基础设施

#### 修复 #3：SSA Regalloc Liveness
- **根因**：`max_vgprs=128` 硬编码 + In-place op 的 MVal 合并缺失
- **修复**：max_vgprs=255 + 同 VReg 多 MVal interval 合并
- **结果**：干涉冲突 26→0，通过率 5/10→8/10

#### 修复 #4：CSE 正确性（5 个根因）

| # | 根因 | 修复 |
|---|------|------|
| 1 | `MVal(u32::MAX)` 哨兵被用作 CSE key | 跳过含哨兵 MVal |
| 2 | InlineInt 常量未编码进 CSE key | 编码为 `MVal(0xFE000000\|bits)` |
| 3 | CSE 替换 VMov 用 VReg(0) | 用 `mval_to_vreg` 查找正确 VReg |
| 4 | 不清空 barrier 处的 seen table | `block_seen.clear()` at barrier |
| 5 | `lower_from_ssa` 不做 VReg 重映射 | 完整 `rename_op_uses/defs` |

### 3 月 26 日上午 — 优化 Pass 修复

#### 修复 #5：DCE Loop-Carried Liveness
- **根因**：K-loop 指针步进 `v153 += 32` 的新 MVal 无 use → DCE 删除 → K 不递增
- **修复**：backward branch 检测 → loop 内 VReg defs 标记为 root-live

#### 修复 #6：Phase D Post-Regalloc Scheduling
- **根因**：Pre-regalloc scheduling 改变指令顺序 → liveness 变化 → 物理寄存器分配不同
- **修复**：实现 `post_regalloc_schedule`（在 lower_from_ssa 之后，物理 VReg 已确定）

### 3 月 26 日下午 — 边界 Masking + 稳定化

#### 修复 #7：任意维度 GEMM（边界 Masking）
- **策略**：Clamp-and-Discard（`v_min_u32`，+2 VGPR）+ Host K/N padding
- **结果**：33×100×50 ~ 1000×768×512 全通过，对齐测试零回归

#### 修复 #8：BufferPool (LRU Cache)
- **根因**：KFD `FREE_MEMORY→ALLOC_MEMORY` 复用 VA 时 GPU L2/TLB 未 invalidate
- **修复**：buffer "释放" 入池缓存 → 零 syscall 复用 → VA 映射永不释放
- **方向性实验**：c128→c32 ❌ / c32→c128 ✅ / c32→c32(2) ✅ / 10-case ❌ → 确认 VA 竞态

#### 修复 #9：Combo 测试体系
- 14 个独立测试标 `#[ignore]`（防跨 runtime VA 竞态）
- combo 测试 10 组维度，单一 runtime，buffer pool 保护
- bench loop 移除（防 dispatch 积累 → 大矩阵 5s timeout）

---

## 三、未解决问题

| 问题 | 优先级 | 状态 |
|------|--------|------|
| `emit_store_phase_masked` | 低 | 两种重写均 err=13.1；推测 VGPR 压力/regalloc 冲突；padded Y 方案已完全可用 |
| `buffer_load` 迁移 | 中 | 释放 VGPR，提升占用率 |
| split-K | 低 | 大 K 维度的并行归约 |

---

## 四、铁律总结

1. **LICM**：hoisted 指令必须插入 terminator **之前**
2. **CSE key**：必须包含 opcode + MVal + inline 常量
3. **DCE**：必须处理 loop-carried dependencies（backward branch → root-live）
4. **Scheduling**：必须在 regalloc **之后**
5. **max_vgprs**：不要人为限制，GEMM 需要大量 VGPR
6. **raw_asm**：绕过 regalloc 的定时炸弹
7. **KFD VA 复用**：必须用 BufferPool 避免 FREE→ALLOC 竞态
8. **硬挂恢复**：SIGPIPE 忽略 + KFD 重试 + 健康探针

---

## 五、关键文件索引

| 文件 | 角色 |
|------|------|
| `ignis/gpu_context.rs` | BufferPool + GpuRuntime |
| `t0/tile_ir.rs` | GEMM 内核生成（边界 Masking + store phase） |
| `t0/ssa_ir.rs` | SSA IR + LICM/DCE/Regalloc |
| `t0/opt_passes.rs` | CSE + post_regalloc_schedule |
| `t0/test_tile_gemm_suite.rs` | combo 测试体系 |
| `kfd/mod.rs` | KFD 运行时（SIGPIPE/重试/探针） |
