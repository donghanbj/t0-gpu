# SSA Regalloc Liveness 根因分析与修复

## 日期
2026-03-25

## 目标
修复 SSA regalloc 的 liveness analysis bug，消除 `ssa_regalloc(false)` workaround，让 T0 编译器能用 SSA pipeline 编译任意 tile 配置的 GEMM 内核。

## 方法

### 诊断流程
1. 添加 SSA vs Legacy regalloc 交叉对比 → 发现 200 个映射差异
2. 添加精确干涉检查器（per-instruction use-use / def-clobbers-use）→ 发现 26 个物理寄存器冲突
3. 追踪冲突到 `compute_live_intervals` 的 MVal → VReg 映射逻辑

### 根因 1: max_vgprs 硬编码
- `compile.rs:847` 硬编码 `max_vgprs=128`
- GEMM 内核需要 ~163 VGPRs → 35 个被 spill 到 LDS
- Spill/reload 代码破坏寄存器映射

### 根因 2: MVal-VReg 多对一映射（核心 bug）
- In-place op (`v_add_co(v5, v5, v3)`) 为同一 VReg 创建多个 SSA MVal
- 每个 MVal 有独立 live interval → 可能被分配不同物理寄存器
- `to_legacy_regalloc` 只能给每个 VReg 保留一个映射（后来的覆盖先前的）
- 结果：某些 VReg 使用点引用了错误的物理寄存器

### 修复
1. `max_vgprs` 128 → 255（GFX1100 硬件上限）
2. `compute_live_intervals` 增加 Step 6：合并同 VReg 的所有 MVal intervals
   - `def_point = min(all defs)`, `last_use = max(all uses)`
   - 等效于 SSA "coalescing"，强制同 VReg 所有版本使用同一物理寄存器

## 结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 测试通过率 | 5/10 | **8/10** |
| 干涉冲突 | 26 个 | **0 个** |
| Spills | 大量 | **0** |
| 32x64 tile | 12866 bad | **0 bad** ✅ |
| SSA regalloc | 禁用(workaround) | **正常使用** |
| VGPRs | 163 (legacy) | 169 (SSA, 合并后) |
| 1024³ 性能 | 14.8 TF | 14.7 TF |

## 结论（铁律）

1. **SSA regalloc 必须合并同 VReg 的 MVal intervals**。
   - 如果 to_legacy_regalloc 是唯一 backend，就不能让同一 VReg 的不同 SSA 版本占用不同物理寄存器。
   - 替代方案：拉高到完全 SSA（插入 copy/phi），但成本更高。

2. **max_vgprs 不要随意限制**。
   - GEMM 等 compute-intensive 内核天然需要大量 VGPRs。
   - 人为限制到 128 会触发 spill，spill 代码的正确性是另一个 attack surface。

3. **skip_optimize 对 GEMM 仍然必要**。
   - 优化 passes (DCE/LICM/scheduling) 会重排 barrier/LDS 协作序列 → GPU 硬挂。
   - 这是独立于 regalloc 的问题，需要 barrier-aware 优化 pass。

## 后续
- 64x64 tile 仍有精度问题 — tile_ir 的 cooperative load 地址计算可能对此配置不正确
- 考虑实现 barrier-aware 优化 pass，让 GEMM 也能享受 DCE/调度优化
