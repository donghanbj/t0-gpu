# GPU 硬挂修复：LICM 死代码 + Shr/Shl raw_asm 虚拟寄存器号

## 日期
2026-03-25

## 目标
修复 `test_gpu_gemm_tn` 内核执行时触发 GPU page fault at 0x0 → 硬挂（当天已累计 6 次强制重启）。

## 故障现象
- `amdgpu` 内核日志：`SQC(inst)` 在地址 0x0 处发生 page fault
- 进程：`t0::block_dsl::gpu_tests::test_gpu_gemm_tn`（`--test-threads=1` 串行模式）
- KFD 运行时：`wait_read_ptr TIMEOUT (5s): GPU hung!`

## 根因分析

### Bug #1：LICM 将 hoisted 指令放到 Branch 之后（硬挂直接原因）

**路径**：`ssa_ir.rs::licm_mach_func()` L1333-1335

`opt_passes::optimize()` 中的 LICM pass 正确识别了循环体（body block）中的地址初始化指令为循环不变量（v10-v15 = A_ptr/B_ptr GPU VA），但将它们 `extend` 到 preheader block 的 `insts` 末尾。由于 `MachBlock` 没有单独的 terminator 字段（Branch op 是 `insts` 的最后一条），hoisted 指令被放在 `s_branch .Lbb1` **之后**，成为不可达死代码。

ISA dump 证据（L43-54）：
```
43: s_mov_b32 s15, 0      ; 循环变量初始化
44: s_branch .Lbb1        ; 跳到循环头 ← hoisted 指令在此之后
45: v_add_nc_u32 v8, ...  ; 🚨 不可达！
...
52: v_mov_b32 v10, s6     ; A_ptr low（从未执行）
53: v_lshlrev_b32 v8, ... ; 
54: .Lbb1:                ; 循环头（GPU 从这里开始）
```

**修复**：`insert(len-1)` 代替 `extend`，在 terminator 之前插入。

### Bug #2：get_vreg 覆写 val_map 导致 Shr 走 raw_asm 路径

**路径**：`tile_ssa_lower.rs::lower_tile_op()` L621-689

BinOp 向量路径先调用 `get_vreg(rhs)` 将 `InlineInt(4)` 物化为 VReg 并覆写 `val_map`，然后 `Shr` 分支调用 `get_val(rhs)` 时返回 `VReg` 而非 `InlineInt`，走进 `raw_asm` 路径。`raw_asm` 使用虚拟寄存器号 `.0` 而非物理号，绕过了 regalloc。

ISA 证据（L20）：
```
20: v_lshrrev_b32 v5, v4, v1   ; v4 未初始化！应为 v_lshrrev_b32 v5, 4, v1
```

**修复**：在 `get_vreg` 之前缓存 `rhs_val_precheck = get_val(val_map, *rhs).ok()`。

## 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `src/t0/ssa_ir.rs` | LICM：`insert(pos)` 代替 `extend` |
| `src/t0/tile_ssa_lower.rs` | 缓存 `rhs_val_precheck`，移除 `raw_asm` 路径 |

## 结论

1. **两个 bug 都是"逻辑正确但接入方式有缺陷"**：SSA regalloc 本身工作正常，但 LICM 的 insert 位置和 `get_vreg` 的副作用引入了边界 bug。
2. **LICM 是更危险的**：它在 opt_passes 中默认启用，且只在含循环的内核中触发，不易被简单测试覆盖。
3. **raw_asm 是反模式**：任何使用 `raw_asm` 的路径都绕过了 regalloc，是潜在的定时炸弹。

## 验证
- `cargo build --release --features rocm --lib` → ✅ 通过
- GPU 测试需谨慎执行（建议先用 `T0_DUMP_ASM=1` 检查 ISA）

## 后续
- 为 LICM 添加 insert 位置的单元测试（验证 hoisted 指令在 Branch 之前）
- 考虑在 MachBlock 中添加显式 `terminator` 字段，避免类似 bug
- 逐步消除所有 `raw_asm` 使用
