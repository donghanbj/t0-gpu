# tile_ir GPU 硬 Hang 根因分析

## 日期
2026-03-27

## 目标
排查 tile_ir kernel 100% 触发 GPU 硬 hang（MODE1 reset）的根因

## 症状
- `test_safe_benchmark` 100% 触发硬 hang 重启
- 最后一条终端输出精确定位 hang 点：
  ```
  512×512×512  54.0  4.971  ✅ HANG [KFD] wait_read_ptr TIMEOUT (5s): read=55, target=56
  ```
- gemm_gen 同尺寸通过（4.97 TFLOPS），**tile_ir 单次 dispatch 就 hang**
- 前 3 个尺寸的 tile_ir dispatch 成功，512³ 失败

## 根因分析

### Bug 1: split-K Y 缓冲区 OOB 写入
- `test_safe_benchmark` 中 Y buffer 只分配 `M*N*4` 字节
- split_k>1 时，split_k_id>0 的 WG 写入 `Y + id*M*N*4` → 超出缓冲区
- **修复**：Y buffer 分配 `M*N*4*split_k` 字节

### Bug 2: optimization pass 破坏 tile_ir ISA
- `opt_level=4` 启用 DCE/CSE 等优化，错误删除 tile_ir 关键指令
- 导致输出全部错乱（max_err=6e31），不直接导致 hang
- gemm_gen 使用 `skip_optimize=true` 不受影响
- **修复**：tile_ir 也使用 `skip_optimize=true`（TODO: 修复优化 pass）

### Bug 3: k_start_bytes double-count（主要 hang 根因）
- **位置**：`tile_ir.rs` L607-616 + L623-624 + L763
- **机制**：
  1. L607-616：`k_start_bytes` 正确加到 `x_gmem_base` / `wt_gmem_base`
  2. L624：`k_byte_off = k_start_bytes`（应该为 0！）
  3. L763：`emit_coop_load` 再次把 `k_byte_off` 加到地址上
  4. 结果：实际偏移 = `k_start_bytes × 2`
- **数值验证**（512×512×512, split_k=4）：
  - split_k_id=3: k_start_bytes = 3×128×2 = 768
  - double-counted: 768×2 = 1536
  - X row size = K×2 = 1024
  - **1536 > 1024 → OOB read → GPU page fault → 硬 hang**
- **为什么 128×64 不 hang**：split_k=1，k_start_bytes=0，double-count 无影响
- **修复**：`k_byte_off` 初始化为 0（K-loop 迭代偏移从 0 开始）

## 验证
- 128×64 单 dispatch: max_err=0.000000 ✅
- safe_benchmark: 待用户验证

## 后续
- [ ] 修复 optimization pass（目前临时禁用）
- [ ] 给 ISA verifier 添加 address double-count 检测
- [ ] split_k 需要正确的 y_split_stride + reduction kernel
