# T0 SSA Pipeline 安全编码规范

> 本文档总结 T0 编译器 SSA pipeline 中反复出现的 bug 模式，建立铁律级编码规范。
> 每条规范都有对应的历史 bug 案例。**违反任何一条都可能导致 GPU 硬 hang（系统重启）。**

---

## 一、地址计算铁律

### 1.1 禁止偏移量 double-count

**铁律**：一个偏移量只能在一个位置加到地址上。

```rust
// ❌ WRONG: k_start_bytes 被加了两次
gmem_base += k_start_bytes;     // 第一次（L607）
k_byte_off = k_start_bytes;     // 初始化为 k_start
addr = gmem_base + k_byte_off;  // 第二次（emit_coop_load L763）
// 实际偏移 = 2 × k_start_bytes → OOB → GPU page fault → 硬 hang

// ✅ CORRECT: gmem_base 已含偏移，k_byte_off 从 0 开始
gmem_base += k_start_bytes;     // 唯一的加法
k_byte_off = 0;                 // K-loop 迭代偏移从 0 开始
addr = gmem_base + k_byte_off;  // 正确
```

**历史案例**：tile_ir 512×512×512 split_k=4 → `k_start_bytes` double-count → split_k_id=3 偏移=1536 > row_size=1024 → OOB → 硬 hang。

### 1.2 64-bit 地址加法必须用 v_add_co + v_add_co_ci

**铁律**：GPU VA 是 64-bit。任何地址偏移加法都必须用 carry chain。

```rust
// ❌ WRONG: 32-bit add 没有 carry → 低 32 位溢出时地址损坏
k.v_add_u32(y_base, y_base, col_bytes);

// ✅ CORRECT: 64-bit add with carry
k.clear_vcc();
k.v_add_co(y_base, y_base, col_bytes);
k.v_add_co_ci(VReg(y_base.0 + 1), VReg(y_base.0 + 1));
```

**注意**：当前 emit_store_phase L974 仍用 `v_add_u32` 加 col_bytes。对小偏移（<2KB）暂时安全，但对 N>16K 的矩阵会 hang。

### 1.3 Buffer 分配必须匹配 kernel 写入范围

**铁律**：`alloc_zero(M*N*4)` 不够，split_k 时需要 `M*N*4*split_k`。

```rust
// ❌ WRONG
let y_buf = rt.alloc_zero((m * n * 4) as usize);

// ✅ CORRECT
let sk = spec.split_k;
let y_buf = rt.alloc_zero((m * n * 4 * sk) as usize);
```

---

## 二、SSA 优化 Pass 铁律

### 2.1 手写 GEMM kernel 必须 skip_optimize

**铁律**：cooperative load + LDS double-buffer + WMMA 的指令序列不可被 DCE/CSE 优化。

```rust
// ❌ WRONG: opt pass 会错误删除"看似冗余"的 load/store
k.set_opt_level(4);

// ✅ CORRECT: 手写 kernel 跳过优化
k.set_skip_optimize(true);
```

**历史案例**：tile_ir 启用 `opt_level=4` → DCE 删除关键指令 → 输出垃圾（max_err=6e31）。同类 gemm_gen 用 `skip_optimize=true` 不受影响。

### 2.2 每次 pass 分配全新 VReg

**铁律**：cooperative load 的每次 pass 必须用新 VReg pair。否则 SSA/DCE 认为"已赋值"而删除后续 pass。

```rust
// ❌ WRONG: 复用 cur_addr → SSA 认为第二次赋值是 dead code
k.v_add_co(cur_addr, cur_addr, stride);

// ✅ CORRECT: 每次 pass 分配新 VReg pair
let next_addr = k.alloc_vreg_array(2, Alignment::Align2);
k.v_add_co(next_addr, cur_addr, stride);
cur_addr = next_addr;
```

---

## 三、VCC / SCC 铁律

### 3.1 v_add_co 序列前必须 clear_vcc

**铁律**：`v_add_co_ci` 读取 VCC carry-in。如果 VCC 被之前的 `v_cmp_*` 污染 → 地址计算错误 → OOB → hang。

```rust
k.clear_vcc();                    // ← 必须
k.v_add_co(dst_lo, a_lo, b_lo);
k.v_add_co_ci(dst_hi, a_hi);
```

**检测**：ISA verifier 的 `vcc_dirty` 检查（`isa_verifier.rs` L146-157）。

### 3.2 循环体内 VCC 必须隔离

**铁律**：K-loop 体内的 `v_cmp_*`（如 boundary check）会污染 VCC。循环计数器的 `s_cmp_*` 使用 SCC（安全），但地址的 `v_add_co` 使用 VCC（危险）。

---

## 四、EXEC Mask 铁律

### 4.1 SaveExec / RestoreExec 必须配对

**铁律**：每个 `SaveExec` 必须有且仅有一个 `RestoreExec`。不平衡 → 后续 wave 执行错误线程 → hang。

**检测**：ISA verifier `exec_save_depth` 检查。

### 4.2 s_barrier 前必须 RestoreExec

**铁律**：`s_barrier` 需要 WG 内所有 wave 参与。如果 EXEC mask 关闭了某些 lane，barrier 仍然正常（等待所有 wave，不是所有 lane）。但如果 SaveExec 导致整个 wave 的 EXEC=0，该 wave 仍会到达 barrier → 安全。

---

## 五、Barrier / 同步铁律

### 5.1 s_barrier 前所有 LDS store 必须完成

```rust
// ❌ WRONG: barrier 前没有等 LDS store
k.ds_store(...);
k.s_barrier();   // 其他 wave 可能读到未完成的 LDS store

// ✅ CORRECT
k.ds_store(...);
k.wait_lgkmcnt(0);
k.s_barrier();
```

### 5.2 s_endpgm 前必须 drain 所有 pending ops

```rust
k.wait_vmcnt(0);    // drain global loads
k.wait_vscnt(0);    // drain global stores
k.wait_lgkmcnt(0);  // drain LDS / scalar loads
k.endpgm();
```

---

## 六、Kernel 描述符铁律

### 6.1 VGPR/SGPR 声明必须 ≥ 实际使用

**铁律**：kernel descriptor 中声明的 VGPR/SGPR 数量必须 ≥ 编译后实际使用。否则 GPU 分配不足 → 寄存器踩踏 → hang。

**检测**：regalloc 输出 `[T0 SSA RegAlloc] 219 VGPRs, 43 SGPRs` 必须与 descriptor 一致。

### 6.2 WGP mode 必须在 descriptor 和 config 中同步

如果 config 设置 `wgp_mode=true`，descriptor 必须设置 `.amdhsa_workgroup_processor_mode 1`。否则 barrier 跨 CU 死锁。

---

## 七、测试规范

### 7.1 新内核必须先 compile-only 验证

```bash
# 先跑 compile-only（不 dispatch GPU）— 零 hang 风险
cargo test -- compile_tests::test_compile_all_sizes --nocapture
```

### 7.2 只在 compile-only 通过后才单 dispatch

```bash
# 单次小矩阵 dispatch
cargo test -- test_tile_ir_gpu_gemm_128x64 --nocapture --test-threads=1
```

### 7.3 多 dispatch benchmark 最后运行

```bash
# 多尺寸 benchmark — 有 hang 风险
timeout 120 cargo test -- test_safe_benchmark --nocapture --test-threads=1
```

---

## 附录：Bug 模式速查表

| 症状 | 可能原因 | 检查点 |
|------|----------|--------|
| **硬 hang（系统重启）** | OOB read/write | 地址偏移 double-count、buffer 分配不足 |
| **硬 hang + timeout 5s** | 同上，或 barrier 死锁 | WGP flag、K-loop 终止条件 |
| **输出全部垃圾** | opt pass 破坏 ISA | 关闭优化重试、dump ISA 对比 |
| **输出数量级正确但错位** | store 偏移 bug | 检查 row/col 计算、WMMA output mapping |
| **部分元素错误** | VCC carry 残留 | 检查 clear_vcc 位置 |
| **结果全 0** | EXEC mask 不平衡 | 检查 SaveExec/RestoreExec 配对 |
| **间歇性 hang** | 地址溢出（条件性 OOB） | 检查 v_add_u32 vs v_add_co |
