# GEMM 自动选择与边界保护优化实验记录

## 日期
2026-03-21

## 目标
1. 修复反复出现的 GPU 硬 hang 问题
2. 优化 128×1024×4096 瘦矩阵性能
3. 更新 `auto_select()` 使训练管线自动使用最优配置

## 方法

### 边界保护（修复 GPU Hang 根因）

**根因**: `GpuBuffer::drop()` 无 GPU 同步 — 直接 unmap VRAM 而 GPU L2 writeback 仍在进行，导致 use-after-free → page fault → 硬 hang。

**修复**:
- `wait_read_ptr` 超时 120s→10s（快速检测 hang，exit(99) 阻止系统崩溃）
- 新增 `queue.synchronize()` 方法（wait_idle + 内存屏障 + 10μs L2 drain）
- `bench_gemm_sweep` 预分配 Y workspace（消除 per-config alloc/drop 竞争）
- `compute_grid`/`compute_grid_split_k` 添加维度验证 assert

### 128×4096 优化探索

| 实验 | 结果 | 结论 |
|------|------|------|
| Deep-K (k32/k64) | 无提升 | LDS 翻倍但 GMEM 延迟也翻倍，抵消 |
| 高 split-K (sk=16) | 128×4096 更差 | 每 WG K-loop 仅 4 次，流水线不够 |
| CU vs WGP mode | CU 冷启动更好 | 热运行 WGP 仍优，差距来自 GPU 时钟 |

### auto_select() 重写

旧版无 WGP、无 M-on-X、无 K-divisibility 检查。新版：
- `clamp_sk()` 闭包自动钳制 split-K 保证 K%(tile_k*sk)==0
- tile_m 回退链 128→64→32 处理任意 M 值
- 6 个分支映射到 benchmark 验证的最优配置

### Stream-K 可行性评估

结论：**GFX1100 上 ROI 为负**。
- 铁律 #63: Atomic+WMMA = +209% 吞吐退化
- 铁律 #27: 96 WG 全 CU 竞争 = 7.3× 延迟
- 铁律 #23: GDS/GWS 不可用，只能用 global atomic

## 结果

### 最终 Sweep 数据（7/12 超 rocBLAS）

| 尺寸 | TFLOPS | vs rocBLAS |
|------|--------|-----------|
| 512² | 12.72 | **102%** 🏆 |
| 1024² | 45.40 | **163%** 🏆 |
| 2048² | 55.86 | **152%** 🏆 |
| 4096² | 67.30 | **115%** 🏆 |
| 128×4096 | 35.55 | 70% |
| 256×4096 | 44.65 | **101%** 🏆 |
| 512×4096 | 50.12 | **109%** 🏆 |
| 1024×4096 | 58.84 | **197%** 🏆 |

峰值 67.30 TF = 理论 123 TF 的 54.7%。

## 结论

1. **GPU hang 根因是 buffer use-after-free**，10s timeout + exit(99) 防护生效
2. **128×4096 的 70% 是静态 split-K 的天花板**，突破需 stream-K 但 GFX1100 不适合
3. **auto_select 已自动选最优配置**，训练管线无需手动指定
4. **下一步优化方向**: WMMA 双链 ILP (+20%) 和 graph 级算子融合

## 后续
- [ ] WMMA 双链 ILP（铁律 #35: 1.24× 吞吐提升）
- [ ] K-loop 软件流水线（铁律 #40: VMMA+VMEM 完美重叠）
- [ ] Graph 级融合（GEMM+Bias+RMSNorm）
