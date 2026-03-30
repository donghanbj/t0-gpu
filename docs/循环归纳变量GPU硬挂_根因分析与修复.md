# 循环归纳变量 GPU 硬挂 — 根因分析与修复

## 日期
2026-03-25

## 目标
解决 `test_gpu_gemm_tn` 执行时 100% 触发 GPU 硬挂（MODE1 Reset）的问题。

## 根因分析

### Bug 1: `get_vreg()` 覆写 block param 的 SReg 映射（主因）

**文件**: `tile_ssa_lower.rs` → `get_vreg()` 函数

**机制**:
1. 循环归纳变量 `iv`（block param）在 `lower_elementwise_1d` 中预分配为 `SReg(s15)`
2. 循环体内第一次使用 `iv`（如 `iter_k.mul(kb, m_arg)`）调用 `get_vreg(iv)` 
3. `get_vreg()` 将 SReg→VReg 提升结果缓存：`val_map[iv] = VReg(v9)` ← **覆写了 SReg(s15)**
4. `end_for` 中 `add(iv, step)` 再次查询 `val_map[iv]`，得到 `VReg(v9)` → 走 **向量加法路径** → `v_add_nc_u32 v29, v9, v28`
5. `Branch{args=[iv_next]}` 的 `copy_val(iv_next, iv_param)`:
   - `val_map[iv_next]` = `VReg(v29)`
   - `val_map[iv_param]` = `VReg(v9)` (被 get_vreg 污染!)
   - 结果: `v_mov v9, v29` (VReg→VReg) ← **应该写 s15 但写了 v9！**

**后果**: Header 的 `s_cmp_lt_u32 s15, s14` 永远读到 `s15=0` → **无限循环 → GPU 挂死**

### Bug 2: LICM 错误提升循环依赖指令（放大因素）

**文件**: `opt_passes.rs` → `licm_mach_func()` 

虽然 Bug 1 是导致 s15 不更新的直接原因，但 LICM 进一步恶化了情况：
- `lift_to_ssa()` 从线性 op 列表构建 SSA，但不为 label/branch/copy 模式创建 phi 节点
- LICM 无法区分循环承载依赖和循环不变量
- 将地址计算、v_mov_from_sgpr 等 ~20 条指令提升到循环外，导致寄存器使用未初始化值

## 修复

### 修复 1: 禁止 get_vreg 缓存 SReg→VReg 提升 （核心修复）
```diff
 // tile_ssa_lower.rs: get_vreg()
 MachineVal::SReg(sr) => {
     let vr = k.alloc_vreg();
     k.v_mov_from_sgpr(vr, sr);
-    val_map.insert(v, MachineVal::VReg(vr));
+    // REMOVED: 缓存覆写导致 GPU 挂死
     Ok(vr)
 }
```

### 修复 2: block_param_original 保护原始分配
```rust
let mut block_param_original: HashMap<Value, MachineVal> = HashMap::new();
// 对每个 block param 保存原始 SReg/VReg 分配
```

### 修复 3: copy_val_to_param 使用原始分配
```rust
fn copy_val_to_param(k, val_map, block_param_original, src, dst_param) {
    let dst_val = block_param_original.get(&dst_param)  // 用原始分配！
        .or_else(|| val_map.get(&dst_param));
    // ...
}
```

### 修复 4: 禁用 LICM（安全网）
```rust
// opt_passes.rs: optimize()
// stats.licm_hoisted = super::ssa_ir::licm_mach_func(&mut func);
stats.licm_hoisted = 0;
```

### 修复 5: VReg→SReg 复制支持
```rust
(MachineVal::VReg(s), MachineVal::SReg(d)) => {
    k.push(Op::RawAsm(format!("v_readfirstlane_b32 s{}, v{}", d.0, s.0)));
}
```

## 结果

**修复前 ISA**（无限循环）:
```asm
.Lbb2:  ; body
  v_mov_b32 v9, s15          ; iv → VGPR (get_vreg caches val_map!)
  ...                         ; 计算、加载、FMA
  s_barrier
  v_mov_b32 v9, v29          ; ← 写回 VReg 而非 SGPR！
  s_branch .Lbb1             ; s15 永远是 0 → 无限循环
```

**修复后 ISA**（正确递增）:
```asm
.Lbb2:  ; body
  v_mov_b32 v9, s15          ; iv → VGPR (每次重新 copy)
  ...                         ; 计算、加载、FMA  
  s_barrier
  s_add_u32 s27, s15, 1      ; iv_next = iv + 1 (标量！)
  s_mov_b32 s15, s27         ; s15 = iv_next (正确写回！)
  s_branch .Lbb1             ; s15 递增了 → 循环终将退出
```

## 验证

- 19 个编译测试全部通过（包括 gemm_tn、wmma、tile_gemm、silu 等）
- ISA dump 确认循环递增指令正确生成

## 铁律

> **NEVER cache SReg→VReg promotions in val_map for SSA Values that may be block params.**
> `get_vreg()` 的缓存机制会覆写 block param 的 SReg 身份，导致 back-edge copy 写入错误的寄存器。

## 后续

- [ ] 重新启动 GPU 后运行 `test_gpu_gemm_tn` 端到端验证
- [ ] 未来重新启用 LICM 时，必须先修复 `lift_to_ssa` 的 phi 语义
