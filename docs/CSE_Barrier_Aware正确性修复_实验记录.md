# CSE Barrier-Aware + 正确性修复 实验记录

## 日期
2026-03-25

## 目标
修复 CSE (Common Subexpression Elimination) 导致 GEMM 内核 GPU hard hang 和错误输出的问题，使 CSE 能安全用于 GEMM 内核。

## 方法

### 诊断过程
1. **T0_DUMP_ASM CSE 诊断日志**：在 `cse_mach_func_domtree` 中添加诊断打印，显示每次 CSE 合并的 tag、VReg、MVal key
2. **lower_from_ssa remap 统计**：添加 USE/DEF remap 计数器
3. **ISA verifier**：检查 regalloc interference conflicts
4. **逐步排除法**：逐个启用/禁用 pass 定位 root cause

### 发现的 5 个 Root Cause

#### RC1: MVal(u32::MAX) 哨兵值用作 CSE key
- `lift_to_ssa` 对未追踪的 VReg 使用 `MVal(u32::MAX)` 作为哨兵值
- CSE 将两个不同的 AND_B32 操作合并，因为它们都使用 `MVal(u32::MAX)` 作为 key
- 修复：`if key_uses.iter().any(|m| m.0 == u32::MAX) { continue; }`

#### RC2: Inline 常量未编码进 CSE key
- `v_add_u32(d, s, 64)` 和 `v_add_u32(d, s, 128)` 的 MVal uses 相同（只有 VReg 源被追踪）
- CSE 将两个不同常量偏移的地址计算合并 → store 写入错误位置
- 修复：将 InlineInt/InlineFloat 编码为 `MVal(0xFE/FD000000 | position | bits)` 加入 key

#### RC3: CSE 替换 VMov 用 VReg(0)（WORKITEM_ID_X）
- `lower_from_ssa` 直接克隆 Op 不做 VReg 重映射
- CSE 创建的 `VMov{src:VReg(0)}` 原封不动输出到最终 ISA → 读线程 ID 而非正确值
- 修复：用 `build_mval_to_vreg` 查找 `prev_mval` 对应的正确 VReg

#### RC4: CSE 不清空 barrier 处的 seen table
- `cse_mach_func_domtree` 在遇到 `SBarrier` 时跳过但不清空 seen table
- 导致跨 barrier 的表达式被合并（LDS 数据在 barrier 后被其他 wave 修改）
- 修复：`if matches!(inst.op, Op::Barrier | Op::SBarrier) { block_seen.clear(); continue; }`

#### RC5: lower_from_ssa 不做 VReg 重映射
- SSA pass（CSE、CopyProp）修改 MVal uses 后，Op 中的 VReg 不更新
- 这是所有 SSA pass 的通用问题，不只是 CSE
- 修复：重构 `lower_from_ssa` 用 `build_mval_to_vreg` + `rename_op_uses`/`rename_op_defs`

## 结果

| 阶段 | 效果 |
|------|------|
| 初始 | GPU HARD HANG（多次） |
| + lower_from_ssa 重映射 | err=inf（不再 hang） |
| + MVal(u32::MAX) 修复 | err=18.3 |
| + Inline 常量编码 | **err=3.81e-6** ✅ |
| 完整 9-test 套件 | **9/10 PASS** ✅ |

最终精度：
- 128³: err=3.81e-6
- 1024³: err=9.92e-5, 15.0 TF

## 结论

1. CSE 的核心问题不在 barrier-aware（只是辅助修复），而在于 **key 构建不完整**和 **lower_from_ssa 不做 VReg 映射**
2. `lower_from_ssa` 重构是最关键的基础设施改进——它使所有 SSA pass 的输出都能正确转回 Vec<Op>
3. 之前"全套 9 test 失败"是因为多次 hard hang 导致 GPU 设备状态 poison，不是 CSE bug
4. **铁律**：CSE key 必须包含所有决定表达式值的因素（操作码 + MVal uses + inline 常量）

## 后续

- Phase C/D（DCE、scheduling、waitcnt 优化）仍需独立调查
- LICM 有独立 bug（`rename_op_uses` 覆盖不全），暂时禁用
- 边界 masking（非 tile 对齐矩阵）缺失
