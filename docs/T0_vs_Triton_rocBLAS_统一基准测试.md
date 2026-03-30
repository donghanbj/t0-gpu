# T0 vs Triton+rocBLAS 统一基准测试

## 日期
2026-03-25

## 目标
在 README.md 的 9 个标准矩阵尺寸上，对比 T0（bench_gemm_sweep 最优配置）vs Triton+rocBLAS 三方性能。

## 环境
- GPU: AMD Radeon RX 7900 XTX (GFX1100, 96 CU)
- PyTorch: 2.9.1+rocm6.4
- Triton: 3.6.0
- rocBLAS: via PyTorch torch.mm
- T0: gemm_gen 参数化生成器（legacy regalloc，48 配置穷举）

## 🐛 修复的 Bug

**SSA RegAlloc LDS Spill 冲突 → GPU Hang**

`gemm_gen::generate()` 使用 `skip_optimize=true` 保留手工指令序列，但未关闭 SSA regalloc（全局默认启用）。
SSA regalloc 在 128×128 = 256 VGPR 的 tile 上产生 48 次 spill，spill 区域起始于 `lds_size`（= 16384），
与 GEMM 双缓冲末尾精确重叠，导致双缓冲数据被覆盖 → GPU hang/timeout。

修复：`generate()` 增加 `k.set_ssa_regalloc(false)`。

## 结果

### Triton+rocBLAS Baseline

| Matrix | rocBLAS (TF) | Triton Fixed (TF) | Triton AutoTuned (TF) |
|---|---:|---:|---:|
| 256³ | 3.05 | 1.41 | 2.26 |
| 512³ | 13.29 | 6.53 | 17.63 |
| 1024³ | 51.96 | 52.49 | 54.21 |
| 2048³ | 71.46 | 76.82 | 76.33 |
| 4096³ | 90.78 | 86.71 | 77.30 |
| 128×1024×4096 | 48.47 | 27.22 | 39.93 |
| 256×1024×4096 | 51.47 | 53.44 | 56.17 |
| 512×1024×4096 | 74.41 | 76.97 | 58.91 |
| 1024×1024×4096 | 64.40 | 75.57 | 74.41 |

### T0 vs Best Opponent

| Matrix | T0 Best (TF) | Best Config | Best Opponent (TF) | T0/Best |
|---|---:|---|---:|---:|
| 256³ | 1.91 | 128x64_k32_sk2_mg | 3.05 (rocBLAS) | 63% |
| 512³ | 12.04 | 128x64_k16_sk2_mg_wgp | 17.63 (Tri-AT) | 68% |
| 1024³ | 41.54 | 128x64_k16_sk8_wgp | 54.21 (Tri-AT) | 77% |
| 2048³ | 56.27 | 128x64_k16_sk8_wgp | 76.82 (Triton) | 73% |
| 4096³ | 66.73 | 128x64_k16_sk8_wgp | 90.78 (rocBLAS) | 74% |
| 128×1024×4096 | 34.30 | 128x64_k16_sk4_mg_wgp | 48.47 (rocBLAS) | 71% |
| 256×1024×4096 | 46.38 | 128x64_k16_sk2_mg_wgp | 56.17 (Tri-AT) | 83% |
| 512×1024×4096 | 48.48 | 128x64_k16_sk8_mg_wgp | 76.97 (Triton) | 63% |
| 1024×1024×4096 | 58.82 | 128x64_k16_sk4_wgp | 75.57 (Triton) | 78% |

## 结论

1. **T0 当前峰值 66.73 TF vs rocBLAS 90.78 TF**（4096³），达到 74%
2. **新版 rocBLAS+Triton 性能远超 README 旧数据**，rocBLAS 1024³ 从 27.89→51.96 TF（+86%）
3. **关键瓶颈**：WMMA 单链 ILP、K-loop 缺少软件流水线、小尺寸占用率不足
4. **优化优先级**：WMMA 双链 ILP (+20%) → 软件流水线 → LDS 访问优化

## 后续
- 实现 WMMA 双链 ILP（铁律 #35: v_wmma 延迟=4 VALU-norm，双链可完美隐藏）
- 实现 K-loop 软件流水线（铁律 #40: VMEM+WMMA 串行但可 overlap scheduled）
- 更新 README rocBLAS 基线数据
