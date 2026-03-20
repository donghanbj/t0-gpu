# 参数化 GEMM 代码生成器实验记录

## 日期
2026-03-20

## 目标
开发参数化 GEMM 内核生成器，系统性优化不同矩阵尺寸的 GEMM 性能，超越 rocBLAS。

## 最终性能表 (10 个变体 Auto-Select)

| 矩阵 | 最优变体 | TFLOPS | vs rocBLAS | 总提升 |
|------|---------|--------|-----------|--------|
| 256³ | k16 split4 | 1.36 TF | 36% | +33% |
| 512³ | k16 split4 | 9.74 TF | **75%** | +71% |
| 1024³ | 64×64 k16 | 31.77 TF | **92%** | +168% |
| **2048³** | k32 split2 | 43.01 TF | **🏆 122%** | +66% |
| 4096³ | 128×64 k32 | 46.96 TF | 79% | +10% |
| 8192³ | 128×64 k32 | 47.53 TF | 78% | — |
| 128×1024×4096 | k16 split8 | 30.33 TF | 52% | — |
| **256×1024×4096** | k16 split8 | 40.22 TF | **🏆 111%** | +96% |
| **512×1024×4096** | k16 split8 | 44.01 TF | **🏆 102%** | +127% |
| **1024×1024×4096** | k32 split4 | 43.85 TF | **🏆 182%** | +100% |

## 优化技术总结

### Phase 1: 参数化生成器 (`gemm_gen.rs`)
- 600行统一代码替代5个手写变体（-67%代码量）
- `GemmConfig`: tile_m/n/k, wg_size, split_k

### Phase 2: 合并内存加载 (最大单项影响)
- 相邻线程访问相邻16-byte chunk
- `row=(t*16)/row_bytes, col=(t*16)%row_bytes`
- 2048³ +15%, 1024×1024×4096 +22%

### Phase 3: Swizzled Grid
- 交换 TGID.x↔TGID.y，M-first WG 调度
- 改善 X 数据 L2 缓存命中率

### Phase 4: Single-Dispatch Split-K
- 编译时 SALU 逻辑提取 tile_col + split_k_id
- Grid Y 编码: `TGID.y = tile_col * split_k + split_k_id`
- 512³ +63%, 512×1024×4096 +38%, 256×1024×4096 +42%

### Phase 5: 128×64 大 tile
- tile_m=128, wg_size=128: WT 数据共享跨 2× 更多行
- Compute/WT-load 比率提高 33%
- 4096³ 和 8192³ 新最优

## 最优配置映射

| 尺寸特征 | 最优配置 | 原因 |
|---------|---------|------|
| tiny (≤512) | split_k=4 k16 | CU 填充 |
| medium (1024) | 64×64 k16 | 平衡 |
| large square | 128×64 k32 | 高计算密度 |
| rect, small M | split_k=8 k16 | 大量并行 |
| rect, large M | split_k=4 k32 | 计算密度+并行 |

## 瓶颈分析

| 尺寸 | 瓶颈 | 可能解决方案 |
|------|------|------------|
| 256³ (36%) | 仅16 WGs for 96 CU | split_k=16 或 persistent kernel |
| 4096³ (79%) | 1 wave/SIMD occupancy | 128×128 tile 或更低 VGPR 变体 |
| 128×1024×4096 (52%) | M太小，CU利用率低 | 16×64 tile + split_k |
