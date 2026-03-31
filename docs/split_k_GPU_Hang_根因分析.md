# Split-K tile_ir GPU Hang 根因分析

## 日期
2026-03-30

## 现象
`tile_gemm_128x128_k16_db_sk2` (split_k=2) dispatch 后 GPU 硬挂

## 根因链（3 个 bug）

### Bug 1: y_split_stride 永远传 0
```rust
// build_kernargs_m L2631:
let y_split_stride: u32 = 0;  // ← 所有 partition 写入同一地址！
```

### Bug 2: Y 缓冲区大小不足
benchmark 只分配 `M * N * 4`，split_k=2 应该分配 `2 * M * N * 4`

### Bug 3: 缺少 split-K reduction kernel
需要将 `split_k` 份 partial sum 累加，当前无此内核

## Hang 原因
多个 split-K partition 同时向同一 Y 地址 buffer_store_b128，
导致 VRAM controller write conflict → GPU hang

## 短期修复
在 can_use_tile_ir() 中过滤 split_k > 1
