# T0 编译器 GEMM Boundary Masking 技术调研

## 日期
2026-03-26

## 一、背景

### 1.1 项目概述

T0 是面向 AMD RDNA3 (GFX1100, RX 7900 XTX) 的裸金属 GPU 编译器。核心场景是生成高性能 GEMM 内核。

### 1.2 当前状态

- 性能：1024³ GEMM 达 14.73 TFLOPS
- 限制：M/N/K 必须是 tile 尺寸整数倍

### 1.3 编译模型

Tile 尺寸编译时固定，M/N/K 通过 kernarg 运行时传入。同一 kernel binary 复用于各种维度。

### 1.4 硬件约束

| 资源 | 限制 | 当前使用 |
|------|------|---------|
| VGPR | 256 个/wave | 128×64 kernel ≈ 240+ |
| SGPR | 106 个/wave | ≈ 40-60 |
| global_load_b128 | 尊重 EXEC mask | 不自动 zero-fill |

---

## 二、DeepResearch 调研结论（2026-03-26）

### 2.1 M/N 维度：Clamp-and-Discard ✅

**结论**：v_min_u32 clamp + store EXEC mask = 0 额外 VGPR（load 阶段）。

- WMMA 的行/列严格正交，OOB 行的垃圾结果不会跨通道污染 valid 行
- Store phase 用 EXEC mask 拦截掉越界写入即可
- **不需要 zero-fill**

### 2.2 Store Phase：Tile Categorization（0 VGPR 开销）

在 Kernel Prologue 用 SGPR 对 WG 分类：

```asm
// 1. 完全越界 WG → 直接 endpgm
s_cmp_ge_u32 s_tile_base_m, s_M
s_cbranch_scc1 .L_ENDPGM

// 2. 分类为 Internal / Boundary
s_add_i32 s_tile_end_m, s_tile_base_m, tile_m
s_cmp_le_u32 s_tile_end_m, s_M
s_cselect_b32 s_is_internal_m, 1, 0
```

- **Internal Tile (>90%)**：极速通路，无边界判断，全速 store
- **Boundary Tile**：慢速通路，复用已死亡的 load VGPR 做 masking

**关键洞察**：进入 Store Phase 时，K-loop 的 load VGPR 全部死亡。寄存器分配器可以 alias 这些 VGPR 作为 masking 临时变量 → **峰值 VGPR 水位不增加**。

### 2.3 K 维度：必须 Zero-fill

K 是归约轴。Clamp 会导致最后列重复累加 → 计算错误。

- **首选**：Host/Allocator pad K 到 tile_k 对齐（0 kernel 修改）
- **次选**：Loop Peeling — 主循环全速跑 `K / tile_k` 次，尾部单独生成一段 tail block，用原位 pre-zero + EXEC mask

### 2.4 buffer_load 架构改造（未来）

从 `global_load`（2 VGPR/地址）切换到 `buffer_load`（1 VGPR/地址 + SGPR 基址）：
- Double-buffer 下可释放 **十几个 VGPR**
- `OOB_SELECT=3` 可让硬件自动 zero-fill 越界访问（1D 线性检查）
- 这是更大的架构改造，但收益巨大

---

## 三、新实现计划

### Phase 1：Clamp-and-Discard（最小 VGPR）

1. `v_min_u32` clamp M/N（已有，+2 VGPR）
2. Tile Categorization prologue（SGPR only，0 VGPR）
3. 双 store path：internal=极速 / boundary=EXEC mask（复用死亡 VGPR，0 额外峰值 VGPR）
4. K 维度：host pad（0 kernel 修改）

### Phase 2：buffer_load 迁移（释放 VGPR）

1. 添加 `buffer_load_b128` ISA 支持
2. 用 SGPR Buffer Descriptor 替代 VGPR 64-bit 地址
3. 启用 `OOB_SELECT=3` 硬件 zero-fill
