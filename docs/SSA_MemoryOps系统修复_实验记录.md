# T0 SSA Memory Ops 系统修复

## 日期
2026-03-28

## 目标
修复 SSA 优化管道对 BufferLoad/BufferStore 内存操作的错误处理，使 tile_ir GEMM 内核能在**启用所有优化 pass** 的情况下正确运行。

## 最终结果

**所有优化 pass 全部启用，零环境变量 workaround，10/10 测试通过** ✅

```
[combo] 128³:           err=3.81e-6  bad=0  ✓
[combo] 256³:           err=9.54e-6  bad=0  ✓
[combo] 128×1024×512:   err=9.54e-5  bad=0  ✓
[combo] 256×512×128:    err=2.48e-5  bad=0  ✓
[combo] 512³:           err=3.81e-5  bad=0  ✓
[combo] 1024³:          err=9.92e-5  bad=0  ✓
[combo] 33×100×50:      err=2.86e-6  bad=0  ✓
[combo] 129×65×17:      err=9.54e-7  bad=0  ✓
[combo] 100×300×200:    err=1.53e-5  bad=0  ✓
[combo] 1000×768×512:   err=6.10e-5  bad=0  ✓
```

## 根因分析与修复

### Bug #1: `has_side_effects()` 缺失 BufferLoad/BufferStore（最严重）

**文件**: `ir.rs` L979  
**影响**: DCE 删除 buffer_load/buffer_store → err=inf

```diff
 Op::GlobalStore { .. } | Op::LdsStore { .. } |
+Op::BufferLoad { .. } | Op::BufferStore { .. } |
 Op::DsStoreB16 { .. } | Op::DsStoreB32 { .. } |
```

### Bug #2: LICM 提升 BufferLoad 循环中的指令

**文件**: `ssa_ir.rs`  
**影响**: LICM 将 K-loop 的地址计算提升到 preheader → err=1e38

**根因**: `loop_inst_count > 100` 保护未生效——SSA 将 K-loop 拆分为多个基本块后，每个块 < 100 指令。LICM 误将 VAddU32/VMov（双缓冲地址计算）判定为"循环不变"并提升。

**修复**: 添加 BufferLoad 循环检测——包含 BufferLoad/BufferStore 的循环体直接跳过 LICM：
```rust
let has_buffer_ops = lp.body.iter().any(|&b| {
    func.blocks[b as usize].insts.iter().any(|&idx| {
        matches!(&func.insts[idx].op,
            Op::BufferLoad { .. } | Op::BufferStore { .. })
    })
});
if has_buffer_ops { continue; }
```

### Bug #3: Scheduler 重排 BufferLoad 块中的指令

**文件**: `opt_passes.rs`  
**影响**: scheduler 将 K-loop 基本块中的 VALU/SALU 指令移到 ds_store 和 waitcnt 之间 → graduated waitcnt 计数混乱 → 256³ err=18.97（128³ 不受影响因为工作组数较少）

**修复**: scheduler 跳过包含 BufferLoad/BufferStore 的基本块：
```rust
let has_buffer_ops = block.iter().any(|op| matches!(op,
    Op::BufferLoad { .. } | Op::BufferStore { .. }));
if has_buffer_ops { continue; }
```

### 额外修复（防御性）

- LICM 排除列表添加 BufferLoad/BufferStore（`ssa_ir.rs`）
- Scheduler VMEM 检测添加 BufferLoad（`ssa_ir.rs`）
- Scheduler SReg 依赖追踪（`opt_passes.rs`）——之前只追踪 VReg，忽略 SReg

## 诊断二分法过程

| 配置 | 结果 | 结论 |
|------|------|------|
| `skip_optimize=true` | ✅ 10/10 | 优化管道有 bug |
| `opt_level=4`（全部优化） | ❌ err=inf | DCE 删除 buffer_load |
| 修 has_side_effects + `DISABLE_PASS=licm SKIP_SCHED=1` | ✅ 10/10 | DCE 修好了 |
| 修 has_side_effects + `DISABLE_PASS=licm` | ❌ 256³ err=18.97 | scheduler bug |
| 修 has_side_effects + `SKIP_SCHED=1` | ✅ 10/10 | LICM+scheduler 有问题 |
| LICM enabled（无 BufferLoad skip），scheduler disabled | ❌ 128³ err=1e38 | LICM 仍破坏 |
| `T0_LICM_DEBUG=1` | LICM 提升 6 条 VAddU32/VMov | 罪魁确认 |
| 添加 BufferLoad-loop skip | ✅ 10/10（scheduler off) | LICM 修好 |
| 添加 BufferLoad-block skip to scheduler | ✅ 10/10（全部启用） | scheduler 修好 |

## 铁律

1. **新增 Op 变体必须同步更新 `has_side_effects()`** — 所有内存操作都必须返回 true
2. **LICM 必须跳过 BufferLoad 循环** — 手动调度的 GEMM K-loop 不兼容 LICM
3. **Scheduler 必须跳过 BufferLoad 块** — graduated waitcnt 模式不容许指令重排
4. **T0 的 phi-less SSA 模型对循环变量的建模不完整** — LICM 的 MVal 不变性分析在没有 phi 节点的情况下不可靠

## 修改文件汇总

| 文件 | 修改 |
|------|------|
| `ir.rs` | `has_side_effects()` 添加 BufferLoad/BufferStore |
| `ssa_ir.rs` | LICM 排除列表 + BufferLoad 循环跳过 + Scheduler VMEM 检测 |
| `opt_passes.rs` | Scheduler SReg 追踪 + BufferLoad 块跳过 |
| `tile_ir.rs` | `set_opt_level(4)` 全优化，无 workaround |
