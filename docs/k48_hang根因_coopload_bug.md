# k48 GPU Hang 根因分析：Cooperative Load 地址分解 Bug

## 日期
2026-03-29

## 目标
找出 k48 (128×128 tile_k=48) 即使 0 spill 也会 GPU hang 的根因

## 发现

### 核心 Bug

`lower_gemm` 中使用 **bitwise AND/shift** 分解 tid→(row, col)：

```rust
let x_cpr_shift = chunks_per_row_x.trailing_zeros() as u8;
x_row_in_tile = tid >> x_cpr_shift   // 应该是 tid / chunks_per_row
x_col_chunk = tid & (chunks_per_row - 1)  // 应该是 tid % chunks_per_row
```

这只在 `chunks_per_row` 是 **2 的幂** 时正确！

### 各配置的 chunks_per_row

| 配置 | tile_k | chunks_per_row | 是 2^n？ | 正确？ |
|------|--------|:-------:|:---:|:---:|
| k16 | 16 | 2 | ✅ | ✅ |
| k32 | 32 | 4 | ✅ | ✅ |
| **k48** | **48** | **6** | **❌** | **❌** |
| k64 | 64 | 8 | ✅ | ✅ |

### k48 的具体错误

- `chunks_per_row_x = 48*2/16 = 6`
- `6.trailing_zeros() = 1` (因为 6=0b110)
- `x_cpr_shift = 1`

实际分解 vs 正确分解：

| tid | `tid >> 1` (错误 row) | `tid / 6` (正确 row) | `tid & 5` (错误 col) | `tid % 6` (正确 col) |
|-----|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 1 |
| 2 | 1 | 0 | 0 | 2 |
| 3 | 1 | 0 | 1 | 3 |
| 4 | 2 | 0 | 4 | 4 |
| 5 | 2 | 0 | 5 | 5 |
| 6 | 3 | 1 | 2 | 0 |
| 7 | 3 | 1 | 3 | 1 |

tid=2 就已经错误！row=1 应该是 0，col=0 应该是 2。

### 后果

所有 128 个线程的 GMEM 读取地址和 LDS 写入地址都是错的。LDS 数据完全混乱，
WMMA 读到垃圾 → 可能产生 NaN/Inf → 最终导致非法 global_store 地址 → GPU page fault → hard hang。

## 结论

**k48 hang 与寄存器分配、spills、gap reclaim 完全无关。**
**根因是 cooperative load 的 tid→(row,col) 分解假设 chunks_per_row 是 2^n，
但 k48 的 chunks_per_row=6 打破了这个假设。**

## 修复方案

1. **添加 assert**：`assert!(chunks_per_row.is_power_of_two())`
2. **k48 需要特殊处理**：使用真除法 `v_mul_hi_u32` 或限制 tile_k 只能是 16 的倍数且使得 chunks_per_row 是 2 的幂（16, 32, 64, 128...）
3. **简单方案**：删除 k48 配置，因为 k32 (79 TF) 已经很好，且 k64 是 power-of-2

## 后续

1. 验证 k64 是否安全（chunks_per_row=8，是 2的幂）
2. k64 在无 gap reclaim 下有 64 spills → 需要 gap reclaim 或其他方法消除
3. gap reclaim k64 = 0 spills → 可以安全 benchmark（k64 的 coop load 是正确的）
