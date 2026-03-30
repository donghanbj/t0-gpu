# tile_ir vs gemm_gen 性能对比 + GPU Hang 根因分析

## 日期
2026-03-23

## 目标
集成 tile_ir GEMM 编译器到 T0 block_ir 管线，验证稳定性并与 gemm_gen 对比性能。

## 性能结果

| M | K | N | tile_ir | TF | gemm_gen | TF | 比率 | 胜者 |
|--:|--:|--:|--------:|---:|---------:|---:|-----:|------|
| 64 | 64 | 64 | 29.9μs | 0.02 | 38.4μs | 0.01 | 1.28x | tile_ir |
| 128 | 64 | 64 | 44.1μs | 0.02 | 40.4μs | 0.03 | 0.92x | gemm_gen |
| 128 | 128 | 128 | 43.1μs | 0.10 | 28.0μs | 0.15 | 0.65x | gemm_gen |
| 256 | 256 | 256 | 38.5μs | 0.87 | 32.1μs | 1.04 | 0.83x | gemm_gen |
| 512 | 512 | 512 | 52.4μs | 5.13 | 45.1μs | 5.95 | 0.86x | gemm_gen |
| **1024** | **1024** | **1024** | **75.4μs** | **28.48** | 121.0μs | 17.75 | **1.60x** | **tile_ir** |
| 2048 | 2048 | 2048 | 1078μs | 15.93 | 612.8μs | 28.04 | 0.57x | gemm_gen |
| 4096 | 4096 | 4096 | 2630μs | 52.25 | 2104μs | 65.32 | 0.80x | gemm_gen |
| **128** | **1024** | **4096** | **49.3μs** | **21.78** | 67.5μs | 15.91 | **1.37x** | **tile_ir** |
| 256 | 1024 | 4096 | 127.8μs | 16.80 | 97.8μs | 21.96 | 0.77x | gemm_gen |
| 512 | 1024 | 4096 | 356.0μs | 12.06 | 203.9μs | 21.06 | 0.57x | gemm_gen |
| 1024 | 1024 | 4096 | 714.5μs | 12.02 | 316.4μs | 27.15 | 0.44x | gemm_gen |

### 分析

**tile_ir 优势场景**（比率 > 1.0x）：
- 1024³ 方阵：28.48 TF（gemm_gen 的 1.60x），因为 gemm_gen 的 auto_select 没有在此尺寸启用 WGP
- 128×1024×4096 瘦矩阵：21.78 TF（1.37x），tile_ir 的 128×64 tile 天然匹配

**tile_ir 劣势根因**：
- **无 WGP 模式**：gemm_gen 在大矩阵启用 WGP（2 CU 共享 dispatch），tile_ir 尚未支持
- **无 split-K**：gemm_gen 用 split_k=4/8 增加并行度，tile_ir 仅 split_k=1
- **这两个缺失导致大矩阵时 CU 利用率低**

## GPU Hang 根因分析

### Bug 1：split-K Output Buffer 溢出（已修复）

**现象**：benchmark 第 54 次 dispatch 后 GPU hang（`read=53, target=54`）需要硬件重启。

**根因**：`gemm_gen::auto_select(64,64,64)` 选择 `split_k=4`，kernel 需要写 `4 × M×N×4` 字节到 output buffer，但 benchmark 只分配了 `1 × M×N×4` 字节。越界写损坏了邻近的 GPU 内存（可能是 AQL 队列状态或其他 buffer）。

**修复**：`bench_tile_ir.rs` 中为 gemm_gen 的 Y buffer 分配 `split_k × M×N×4` 字节。

**铁律**：**任何使用 split_k 的 GEMM dispatch，output buffer 必须分配 `split_k × M×N×element_size` 字节**。

### Bug 2：64×64 Tile 2-Wave Multi-Dispatch Hang（已绕过，根因待定）

**现象**：tile_ir 64×64 kernel 第一次 dispatch 正确返回结果，第二次 dispatch 永久 hang。

**排除项**：
- ✅ LDS 地址范围内（最大 8176+16=8192，恰好等于上限）
- ✅ `set_wg_size(64)` 正确调用
- ✅ `wait_vmcnt(0)`, `wait_vscnt(0)`, `wait_lgkmcnt(0)` 均在 `s_endpgm` 前
- ✅ `s_barrier` 不是问题（gemm_gen 也用 barrier，multi-dispatch 正常）
- ✅ GMEM store carry chain 逻辑正确

**唯一区别**：64×64 tile `n_waves=2, wg_size=64`。32×64（n_waves=1）和 128×64（n_waves=4）均 multi-dispatch 稳定。

**绕过**：tile 选择逻辑跳过 64×64，用 32×64 替代。

**怀疑方向**：可能是 RDNA3 硬件对 2-wave workgroup + barrier 有特殊行为，或者 wave 调度器在 2-wave 场景下留下未清理的 barrier 状态。

## 后续计划
1. **tile_ir WGP 模式**：启用 `.amdhsa_workgroup_processor_mode 1`，翻倍 CU 利用率
2. **tile_ir split-K**：实现 K 维度分块并行，提升大 K 矩阵性能
3. **64×64 tile 根因调查**：转储二进制机器码，用 `umr` 或类似工具对比 32×64 和 64×64 的 KD 差异
