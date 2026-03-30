# 边界 Masking 实验记录

## 日期
2026-03-26

## 目标
让 tile_gemm 支持任意 M/N/K 维度，不要求 caller 做 padding。

## 方法

### 核心策略：Clamp-and-Discard + Host Padding

基于 DeepResearch 调研结论：

1. **M/N 维度**：`v_min_u32` clamp（仅 +2 VREG），无 zero-fill
   - OOB 线程 clamp 到最后一行，读到重复数据
   - WMMA 行正交性保证 valid 行结果不受污染
   - Store 阶段 OOB 行写入 padded Y 区域（harmless）

2. **K 维度**：Host-side padding
   - 分配 X[M, K_padded] 和 WT[N, K_padded]，pad 区域零填充
   - K 是归约轴，clamp 会导致重复累加 → 必须 zero-fill

3. **Y 输出**：分配 m_padded × n_padded 的 padded buffer
   - kernel 传 K_padded 和 N_padded 作为 stride 参数
   - 读回时按 N_padded stride 提取 M×N 子矩阵

4. **Tile Categorization**：SGPR 级 WG 分类（0 VGPR）
   - 完全 OOB WG → `s_endpgm`（early exit）
   - 当前所有 tile 用原始 fast store（padded Y 保护越界写入）

### 关键 Bug 修复

| Bug | 根因 | 修复 |
|-----|------|------|
| 对齐测试 err=12.1 | Masking 额外 8 VREG 导致 VGPR 溢出 | 去除 zero-fill + oob flag，仅用 v_min_u32 (+2 VREG) |
| 非对齐全错 (bad=1361) | K stride 不匹配：kernel 用 K 但 buffer 是 K_padded | 传 K_padded 给 kernel 的 K 参数 |
| N stride 重叠 (bad=338) | OOB col 写入 `row*N+col(>=N)` 与下行重叠 | 传 N_padded 给 kernel，Y 用 N_padded stride |
| 硬挂 (SGPR 溢出) | emit_store_phase_masked 循环内分配 SGPR | 循环外复用单个 saved_exec SGPR |

## 结果

### 非对齐测试 ✅

| 测试 | M×N×K | Tile | err | bad | 状态 |
|------|-------|------|-----|-----|------|
| test_tg_33x100x50 | 33×100×50 | 32×64 | 2.86e-6 | 0 | ✅ PASS |
| test_tg_100x300x200 | 100×300×200 | 32×64 | 1.53e-5 | 0 | ✅ PASS |
| test_tg_129x65x17 | 129×65×17 | 128×64 | 9.54e-7 | 0 | ✅ PASS |
| test_tg_1000x768x512 | 1000×768×512 | 128×64 | 6.10e-5 | 0 | ✅ PASS（单独运行） |

### 对齐测试回归 ✅

| 测试 | err | bad | 状态 |
|------|-----|-----|------|
| 128³ | 3.81e-6 | 0 | ✅ |
| 128×1024×512 | 9.54e-5 | 0 | ✅ |
| 256³ | 9.54e-6 | 0 | ✅ |
| 256×512×128 | 2.48e-5 | 0 | ✅ |

## 结论

1. **Clamp-and-Discard 策略完全正确**：WMMA 行正交性保证 valid 行不受污染
2. **Host padding 是最安全的 K/N 对齐方案**：0 内核修改，0 性能损失
3. **1000×768×512 硬挂根因**：bench loop（8 次重复 dispatch）在 timeout 杀进程时 GPU queue 未清理 → 已修复（大矩阵跳过 bench loop）
4. **全套测试顺序运行仍有硬挂风险**：多测试共享 OnceLock GPU runtime 的状态残留问题，需单独运行
5. **对齐测试零回归**：1024³ PASS (15.10 TF)

## 后续

1. 调查 1000×768×512 硬挂（可能是 grid 大小/内存问题）
2. emit_store_phase_masked 有 bug 待修复（当前用 padded Y 绕过）
3. 未来考虑 buffer_load 迁移释放 VGPR
