# T0 GEMM 系统性优化实验记录

## 日期
2026-03-21

## 目标
突破 GEMM 性能瓶颈（之前卡在 ~50 TF，41% 峰值利用率）

## 方法与结果

### 1. 发现并修复 LDS padding 正确性 bug
- **问题**：协作加载用 padded LDS stride 做线程分解，只加载一半 tile 行
- **影响**：之前报的 71 TF (4096³) 是错误数据的计算速度
- **修复**：GMEM 分解用 `tile_k*2`，LDS 存储用 padded stride
- **结论**：LDS padding 的"收益"全部是 artifact

### 2. 正确性测试建立
- 39 个测试（8 configs × 6 sizes），随机 bf16 数据
- 相对误差 ~1e-7，远优于 bf16 理论上限
- 所有优化后立即验证

### 3. Roofline 分析（修正版）

**关键公式**：Per-tile AI = tile_m × tile_n / (tile_m + tile_n)
- 64×64 tile: AI = 32 FLOP/byte（远低于 128 转折点）
- tile_k 不影响 AI！
- 128×128 / 64×128 超 VGPR 上限无法编译

**指令分析**：WMMA 仅占 K-loop 指令的 14.5%
但 RDNA3 VALU+WMMA 双发射可隐藏 VALU 开销，真正瓶颈是内存带宽。

### 4. Grid Swizzle (L2 Tiling)
- TGID.x=N-tiles(快)，TGID.y=M-tiles(慢)
- 连续 WG 共享 X 行数据在 L2 cache

| 尺寸 | Before | After | 提升 |
|------|--------|-------|------|
| 512² | 11.6 TF | 12.2 TF | +5% |
| 1024² | 35.2 TF | 38.1 TF | **+8%** |
| 4096² | 50.8 TF | 53.8 TF | **+6%** |

矩形 M<<N 退化（split-K 打包不平衡）。

## 结论/铁律
1. **永远先跑正确性测试** — 性能数据无意义如果计算是错的
2. **Per-tile AI 由 tile 尺寸决定，与 tile_k 无关**
3. **VGPR 256 上限是 tile 增大的硬墙** — 无法 128×128
4. **Grid 轴顺序影响 L2 命中率 ~6-8%**

## 后续
- LDS XOR swizzle 消除 bank conflict
- 更深层的内循环优化（减少地址计算指令）
- 探索 persistent kernel / stream-K 方法
