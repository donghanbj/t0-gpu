# SSA RegAlloc 碎片化调查实验记录

## 日期
2026-03-29

## 目标
调查 64×128 k32 配置出现 28 spills 的反直觉现象

## 发现

### 1. ACC 误解澄清
64×128 k32 的 per-wave ACC 和 128×128 k32 完全相同（128 VGPRs/wave）

### 2. WT GMEM 加载压差
wt_loads_per_thread: 8 vs 4 → GMEM VGPRs 多 16

### 3. 碎片化根因
spill 发生时 free_pool=0 — alignment gaps 沉默浪费 ~12-15 VGPRs

### 4. FreePool::free() 合并 bug
修复为全量排序+合并（安全，无回归）

### 5. Gap 回收导致 page fault（已禁用）
VGPR 大幅降低但 dispatch page fault，需进一步调查

## 结论
- 64×128 k32 的 spill 是真实压力（253 peak active）
- coalesce 修复安全有效
- Gap 回收是最大优化机会（省 15 VGPRs），有正确性问题待解
