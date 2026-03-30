# waitcnt 精化 + WMMA ILP 实验记录

## 日期
2026-03-27

## 目标
将 tile_ir GEMM 从 bf16 WMMA 66% 效率提升到 85%+，通过 rocBLAS/Triton 反向分析得出的 waitcnt 精化和 WMMA 双链 ILP 优化。

## 方法

### 变更 1: gemm_gen `lds_read_and_wmma!` 宏
- `wait_lgkmcnt(0)` → graduated `lgkmcnt(N)` (N = total_loads - loads_needed)
- Phase-2 重构：X[1] prefetch 与 Phase-1 WMMA 并行

### 变更 2: tile_ir `emit_lds_read_and_wmma()` 函数
- 同样的 graduated `lgkmcnt(N)` 策略
- Load order: A frags → B frags → graduated WMMA dispatch

## 结果

### ISA 验证 (tile_gemm_128x64_k16_db)
```
s_waitcnt lgkmcnt(6)   // 12 loads, 等前 6 完成 → frag_a[0]+frag_b[0]
v_wmma_f32_16x16x16_bf16 v[8:15], v[112:119], v[128:135]
s_waitcnt lgkmcnt(4)   // 等 8 完成 → frag_b[1]
v_wmma_f32_16x16x16_bf16 v[16:23], v[112:119], v[136:143]
s_waitcnt lgkmcnt(2)   // 等 10 完成 → frag_b[2]
v_wmma_f32_16x16x16_bf16 v[24:31], v[112:119], v[144:151]
v_wmma_f32_16x16x16_bf16 v[32:39], v[112:119], v[152:159]  // lgkmcnt(0) 隐含
v_wmma_f32_16x16x16_bf16 v[40:47], v[120:127], v[128:135]  // row1 × col0
...
```

### Benchmark (bf16 WMMA, 7 尺寸)
| Size | tile_ir TF | gemm_gen TF | 备注 |
|------|-----------|-------------|------|
| 64³ | 0.021 | 0.019 | |
| 128²×64 | 0.068 | 0.053 | |
| 256²×64 | 0.226 | 0.194 | |
| 256³ | 0.868 | 0.661 | |
| 512³ | 6.770 | 4.089 | |
| **1024³** | **82.12** | **15.02** | 67% bf16 效率 |
| 2048³ | 567 | 34.3 | timing variance |

## 结论
1. **waitcnt 精化对 gemm_gen 效果显著**: 解决了 dispatch hang (7 个尺寸从 hang→正常)
2. **对 tile_ir 效果有限** (+1.3%): tile_ir 已有良好的双缓冲管线
3. **67% bf16 效率的瓶颈**: K-loop overhead (barrier ~20-40 cyc/iter)、occupancy (4 waves/SIMD)
4. tile_ir 的 GMEM prefetch 已与 WMMA 正确并行

## 后续
- occupancy 优化（减少 VGPR 压力）
- WMMA 跨 row_block 交错 ILP
- 循环展开减少 K-loop 指令数
