# Triton + rocBLAS GEMM 抓包分析

## 日期
2026-03-27

## 性能对比

| Size | **T0 tile_ir** | **Triton** | **rocBLAS** |
|------|:-:|:-:|:-:|
| 1024³ | 16.63 TF | 51.23 TF | 52.00 TF |
| 2048³ | 20.76 TF | 73.20 TF | 93.98 TF |
| 4096³ | **48.53 TF** | **80.11 TF** | **96.17 TF** |

T0 vs rocBLAS = **50.5%**，vs Triton = **60.6%**

## Triton Autotune 最优配置

| Size | BLOCK_M | BLOCK_N | BLOCK_K | warps | stages |
|------|:---:|:---:|:---:|:---:|:---:|
| 256³ | 64 | 32 | 32 | 2 | 5 |
| 512³ | 32 | 64 | 32 | 2 | 5 |
| 1024³ | 64 | 128 | 32 | 4 | 4 |
| **2048³** | **128** | **128** | **64** | **8** | **2** |
| **4096³** | **128** | **128** | **64** | **8** | **2** |

大矩阵 Triton 选择 **128×128 k64, 8 warps, 2 stages, WGP mode**

## Triton ISA 关键指标 (4096³ kernel)

| 指标 | Triton | T0 tile_ir |
|------|:---:|:---:|
| WMMA 指令 | **64** | 8 |
| LDS loads (ds_load) | **288** | 12 |
| s_barrier | **3** | ~256 (每 K 迭代 2 个) |
| s_waitcnt | 105 | ~512 |
| WGP mode | **✅** | ❌ |
| tile_k (K sub-steps) | 64 (4 sub-steps) | 16 (1 sub-step) |
| tile M×N | 128×128 | 128×64 |
| WMMA format | f16→f32 | bf16→f32 |

## T0 vs Triton 差距根因

### 1. tile 大小 (最大因素)
- Triton: **128×128** = 16384 输出元素 → 每元素 GMEM 写入开销摊薄
- T0: **128×64** = 8192 输出元素 → 2× 更高每元素开销
- 影响：~30% 性能差距

### 2. K 维度深度
- Triton: **k64 = 4 K sub-steps** → 64 WMMAs/iter, 更多计算/同步比
- T0: **k16 = 1 sub-step** → 8 WMMAs/iter, K-loop 控制开销大
- 影响：~20% 性能差距

### 3. s_barrier 频率
- Triton: **整个 kernel 仅 3 个 s_barrier** — 可能合并 K 迭代或用其他同步方式
- T0: 每 K 迭代 2 个 s_barrier（Phase A/B 各一个）→ K=4096, iter=256, 512 barriers
- 影响：~15% 性能差距（barrier 延迟累积）

### 4. WGP Mode
- Triton: **WGP enabled** (8 warps 跨 2 CU)
- T0: CU mode (4 waves 单 CU)
- 影响：~5-10%（取决于 VGPR 压力）

## 优化路线图

| 优先级 | 优化项 | 预期收益 | 难度 |
|:---:|--------|:---:|:---:|
| 1 | **128×128 tile** | +30% | 中 |
| 2 | **k32/k64 + 交错** | +20% | 高 |
| 3 | **减少 s_barrier** | +15% | 高 |
| 4 | WGP mode | +5-10% | 低 (已验证) |
