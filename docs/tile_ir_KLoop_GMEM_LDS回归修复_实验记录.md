# tile_ir K-loop GMEM→LDS Pipeline 回归修复

## 日期
2026-03-28

## 目标
修复 tile_ir 生成的 GEMM 内核 K-loop 中 buffer_load/ds_store 被 SSA 优化管道消除的回归 bug。

## 症状
所有 `test_tg_*` 测试返回 `err=inf`（16384/16384 元素全部错误）。

## 根因分析（3 层穿透）

### 第 1 层：缺失的 GMEM→LDS 存储
`emit_lds_read_and_wmma` 有两个分支：
- **streaming mode** (`n_col_tiles > 4`): 包含 `emit_interleaved_store`，正确地在 WMMA 之间插入 GMEM→LDS 存储
- **bulk-load mode** (`n_col_tiles ≤ 4`, 即 128×64 等常用配置): **完全没有存储逻辑**

结果：`emit_coop_load_buffer` 加载的 GMEM 数据存入 VGPRs，但 VGPRs 从未被 `ds_store` 写入 LDS。

### 第 2 层：优化等级覆盖
`lower_gemm` 末尾调用 `k.set_opt_level(4)`，该方法通过 `std::env::set_var("T0_OPT_LEVEL", "4")` **覆盖**用户的 `T0_OPT_LEVEL=0` 环境变量。因此 Phase C DCE 始终运行。

### 第 3 层：DCE 正确消除"死代码"
DCE 正确识别 buffer_load 的 VGPR 结果无人使用（因为第 1 层的存储缺失），将其标记为死代码删除。

**因果链**：缺失存储 → VGPR 无人使用 → DCE 删除 buffer_load → K-loop 只读 prologue 的 stale LDS 数据 → err=inf

## 修复方案

### 方案选择过程

| # | 方案 | 结果 | 失败原因 |
|---|------|------|----------|
| 1 | WMMA 内 interleaved store | err=15.29 | ds_store 污染 lgkmcnt graduated waits |
| 2 | 全部 store 在 WMMA 后 | GPU hang | gmem VGPR 活跃范围过长 → regalloc 冲突 |
| 3 | Column 边界 store + lgkmcnt 补偿 | GPU hang | 同上 regalloc 冲突 |
| **4** | **Prologue 模式 + skip_optimize** | **✅ 通过** | 复用验证过的存储模式 |

### 最终修复（方案 4）

1. **K-loop 存储**：在 `emit_lds_read_and_wmma` 返回后，调用 `emit_lds_store_graduated`（与 prologue 完全相同的模式）
2. **跳过 SSA 优化**：`k.set_skip_optimize(true)` 替代 `k.set_opt_level(4)`
   - SSA passes (LICM, CSE, DCE) 对手工优化的 K-loop 代码有副作用
   - tile_ir 的计算核心（WMMA + graduated LDS waits）已是手工最优

### 代码变更

```rust
// tile_ir.rs: K-loop Phase A（Phase B 对称）

// BEFORE (broken):
emit_coop_load_buffer(&mut k, &gmem_x_1, ...);  // buffer_load → VGPRs
emit_lds_read_and_wmma(..., Some(&sched_a));     // WMMAs only, no stores
k.wait_lgkmcnt(0); k.s_barrier();

// AFTER (fixed):
emit_coop_load_buffer(&mut k, &gmem_x_1, ...);  // buffer_load → VGPRs
emit_lds_read_and_wmma(..., None);               // WMMAs only (explicit)
emit_lds_store_graduated(&mut k, &gmem_x_1, ...buf1...);  // VGPR→LDS ★
emit_lds_store_graduated(&mut k, &gmem_wt_1, ...buf1...); // VGPR→LDS ★
k.wait_lgkmcnt(0); k.s_barrier();
```

## 验证结果

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

## 铁律

1. **tile_ir 内核必须 `skip_optimize`** — SSA passes (LICM/CSE/DCE) 对手工优化的 K-loop 有破坏性副作用
2. **不能在 graduated wait 循环中插入 ds_store** — lgkmcnt 不区分 load/store，会破坏计数
3. **`set_opt_level(4)` 覆盖 env var** — `T0_OPT_LEVEL=0` 对 tile_ir 无效
4. **post-WMMA store 活跃范围过长** — 12 VGPRs 跨 8 WMMAs 触发 regalloc 冲突 → hang
5. **Prologue 模式最安全** — `emit_lds_store_graduated` 已在 prologue 验证，应始终复用

## 后续
1. 修复 SSA passes 对 memory side-effect 指令的处理
2. 修复 regalloc 干涉冲突（根本原因）
3. 性能优化：VMEM latency hiding（K-sub 增大、三级流水线）
