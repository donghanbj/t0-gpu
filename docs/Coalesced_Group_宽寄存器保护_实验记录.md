# Coalesced Group：宽寄存器操作的通用 Opt Pass 保护

## 日期
2026-03-26

## 目标
用通用方案替代 `skip_optimize(true)` + `set_ssa_regalloc(false)` 临时 workaround，让 WMMA 等宽操作在不禁用优化的前提下保持多寄存器物理连续性。

## 方法

### 根因分析
CopyProp 将 SplatFragment 的 8 个 `VMov{dst:v3+j, src:v2}` 替换为直接使用 v2 → WMMA 读 `v[2:9]` 而非 `v[8:15]`。与 SSA Regalloc MVal 合并 bug 属同一类：**SSA 标量语义不理解硬件宽操作的物理连续性约束**。

### 设计
- `MachInst` 添加 `coalesced_group: Option<u32>` 字段
- `T0Kernel` 添加 `CoalescedGroup` 注册表 + `mark_coalesced_group(base, count)` 方法
- `annotate_coalesced_groups()` 后处理函数：扫描 MachFunc，根据 VReg 范围标记 MachInst
- CopyProp：跳过 `coalesced_group.is_some()` 的 VMov
- DCE：`coalesced_group.is_some()` → root-live

### 关键设计决策
**不修改 `Op` enum**（ISA 层纯净），标记仅存在于 SSA 层的 `MachInst` 中。

## 修改文件

| 文件 | 修改 |
|------|------|
| `ssa_ir.rs` | MachInst 字段 + 6 处构造点 + annotate 函数 + CopyProp/DCE 保护 |
| `compile.rs` | CoalescedGroup 结构体 + mark_coalesced_group + 传参 |
| `opt_passes.rs` | optimize() 接收 groups + annotate 调用 |
| `tile_ssa_lower.rs` | 3 处替换为 mark_coalesced_group |

## 结果

| 指标 | 临时方案 | 通用方案 |
|------|---------|---------|
| WMMA 精度 | err=0 ✅ | err=0 ✅ |
| Opt passes | ❌ 全部禁用 | ✅ 仅跳过 coalesced VMov |
| SSA regalloc | ❌ 强制 legacy | ✅ 正常 SSA |
| GEMM combo | 10/10 | 10/10 |
| 9 GPU 测试 | 9/9 | 9/9 |

## 结论

1. `coalesced_group` 是正确的抽象层级——在 MachInst（SSA 层）而非 Op（ISA 层）上标记
2. 通用方案让 WMMA 内核也能享受 ConstFold/AlgSimp/CSE 等安全优化
3. 未来 128-bit load/store、64-bit addr 等宽操作也可复用此机制

## 后续
- 为 `DsLoadB128`/`GlobalLoad B128` 等 4-VGPR 操作注册 coalesced group
- 考虑在 SSA regalloc 中也尊重 coalesced_group（当前仍用 alloc_vreg_array Align8）
