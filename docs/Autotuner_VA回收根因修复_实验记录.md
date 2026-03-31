# Autotuner GPU Hang 根因修复：code_buf VA 回收

## 日期
2026-03-31

## 目标
解决 tile_ir autotuner 在同一队列上顺序 dispatch 不同内核时导致 GPU 硬挂的问题。

## 症状
- k64 单独 dispatch ✅（101 TF）
- k64 → k32 顺序 dispatch：k32 的第一个 warmup 硬挂
  - `wait_read_ptr TIMEOUT: read=30 target=31 write=31 pending=1`
- 原始 tile_ir benchmark（使用 `ensure_kernel_t0`）从不 hang

## 排除的假说

### ❌ WGP Mode 切换假说
- 强制所有候选 `wgp_mode=false` → 仍然 hang
- 实际观察：两个内核的 RSRC1 仅 VGPR_COUNT 不同（248 vs 216），WGP/SGPR/LDS 完全一致
- KFD log 显示两者都是 `WGP mode enabled: KCP=0x0408`（tile_ir 编译器默认行为）

### ❌ 异步 dispatch 残留假说
- k64 的 wait_idle() 成功完成后才开始 k32
- read == write 确认所有 k64 packet 已处理

## 真正的根因：code_buf VA 回收

### 旧代码路径
```
benchmark_tile_ir_one(k64):
  GpuKernel::load → code_buf 分配 VA=0x...22C0 (VRAM)
  dispatch × 30 → 成功
  函数返回 → GpuKernel drop → code_buf 被 munmap + KFD FREE
  
benchmark_tile_ir_one(k32):
  GpuKernel::load → code_buf 分配 VA=0x...32C0 (仅差 0x1000!)
  dispatch → GPU TLB 可能仍缓存旧 VA→物理页映射 → HANG
```

### 为什么原始 benchmark 不 hang
`ensure_kernel_t0` 将所有 GpuKernel 缓存在 HashMap 中，**code_buf 永远不释放**。
没有 VA 回收 → 没有 TLB stale entry → 不 hang。

### 证据
- k64 desc_va=0x7C3F3585**2**2C0
- k32 desc_va=0x7C3F3585**3**2C0
- 差值 = 0x1000（一个页面），说明 Linux mmap 回收了相邻 VA

## 修复方案
**预编译所有候选，将 GpuKernel 保持在 Vec 中直到整个 tune 会话结束。**

具体改动：
1. 遍历所有候选 → 编译 + GpuKernel::load → 保存到 `Vec<CompiledCandidate>`
2. 共享一组 A/B/C buffer（不再每次 alloc/recycle）
3. 逐个 benchmark（10 warmup + 20 async + wait_idle）
4. 全部完成后 Vec drop → 所有 code_buf 同时释放

## 修复结果

13 个候选全部成功 benchmark，零 hang！

| 配置 | TFLOPS | VGPRs | Waves |
|------|--------|-------|-------|
| 128x128_k32 | **99.2** | 216 | 2 |
| 128x64_k32 | **99.2** | 160 | 4 |
| 128x128_k64 | 97.4 | 248 | 2 |
| 128x64_k64 | 93.0 | 184 | 4 |
| 64x128_k32 | 90.7 | 232 | 2 |
| 64x64_k64 | 88.3 | 200 | 2 |
| 64x64_k32 | 87.3 | 168 | 4 |
| 128x128_k16 | 65.4 | 200 | 2 |
| 128x64_k16 | 57.6 | 152 | 4 |
| 64x128_k16 | 49.9 | 208 | 2 |
| 64x64_k16 | 45.8 | 160 | 4 |
| 32x64_k16 | 28.1 | 152 | 4 |
| 64x128_k64 | 2.3 | 244+spill | 2 |

总耗时：2.89s（含编译 + benchmark）

## 关键发现

1. **k32 ≈ k64 性能**：128x128_k32 (99.2 TF) ≈ 128x128_k64 (97.4 TF)
   - k32 VGPRs 更少（216 vs 248），寄存器压力低
   - 对于 4096³ 矩阵，K 维度足够大，k32 的更高 ALU 利用率抵消了 k 步数增加
   
2. **Spill = 废**：64x128_k64 需要 32 spills → 2.3 TF（正常的 1/40）
   - 244 VGPRs + spill 到 LDS = 每次 WMMA 前后都要 load/store LDS
   
3. **非方形 tile 也能跑高**：128x64_k32 = 99.2 TF（与 128x128 并列）

## 铁律

> **铁律 #1：GPU 内核在 KFD 裸金属环境中，同一队列上不同 GpuKernel 之间切换时，
> 必须确保所有 code_buf 同时存活。提前释放 code_buf 会导致 VA 回收 → GPU TLB 
> stale entry → CP 读取错误代码 → 硬挂。**

## 后续
- [ ] 将 autotuner 结果整合到 tile_auto_select 的默认配置中
- [ ] 修复 64x128_k64 的寄存器溢出（或从候选中排除）
- [ ] 探索更多非方形 tile 配置
