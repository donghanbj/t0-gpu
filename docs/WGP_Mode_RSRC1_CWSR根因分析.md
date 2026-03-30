# WGP Mode 完整根因分析 — RSRC1 修复 + CWSR 64KB 限制

## 日期
2026-03-30

## 目标
找到 WGP mode 下 >64KB LDS hang 的根因并修复

## 方法与发现

### 第一层：地址对连续性 Bug
- GlobalStore 的 v[addr_lo:addr_lo+1] 假设连续物理寄存器
- alloc_addr_pair() 修复后 CU mode 彻底稳定

### 第二层：RSRC1 WGP_MODE 未设置
- LLVM .amdhsa_workgroup_processor_mode 只设 KCP bit 10，不设 RSRC1 bit 27
- Triton 也是同样的行为（RSRC1 WGP=0, KCP WGP=1）
- KFD runtime 需要手动从 KCP 传播到 RSRC1
- 修复后 rsrc1=0xE8BF0000 (bit 27=1)

### 第三层：128KB LDS — KFD 内核模块硬限制

#### 证据链
1. `/sys/class/kfd/.../properties` 报告 `lds_size_in_kb 64`
2. `kfd_queue.c` 中 `kfd_get_lds_size_per_cu()` 硬编码 0x10000 (64KB)
3. GFX1100 不在动态 LDS 列表中（只有 GFX905x 和 GFX1250x）
4. CWSR save area = 96 CU × (VGPR + SGPR + LDS_64KB + misc)
5. CWSR size 必须精确匹配驱动计算值（ctl_stack_size != 则 EINVAL）
6. 增大 CWSR buffer → CREATE_QUEUE EINVAL
7. dmesg: "Freeing queue vital buffer, queue evicted"

#### 根因
当内核请求 >64KB LDS 时，CWSR 无法保存完整 LDS 状态。
当系统调度器触发 preemption 时，save 失败，队列被 evict。

## 结果

| 阶段 | 配置 | 状态 |
|------|------|------|
| Stage 1 | CU + barrier | ✅ PASS（地址对修复） |
| Stage 2 | WGP, no LDS | ✅ PASS（RSRC1 修复） |
| Stage 3 | WGP, 4KB LDS | ✅ PASS |
| Stage 4 | WGP, 80KB LDS | ❌ CWSR 限制（KFD 内核模块） |

## 铁律

1. **WGP mode + LDS ≤ 64KB 安全可用** — 立即可在 tile_ir 中启用
2. **128KB LDS 不可用** — 需要 KFD 内核模块补丁
3. **LLVM GFX11 不设 RSRC1.WGP_MODE** — runtime 必须从 KCP 传播
4. **Triton 也不用 >64KB LDS** — 这不是我们的特有限制

## 后续
- [ ] 在 tile_ir 生产路径启用 WGP mode
- [ ] 提交 upstream KFD 补丁（添加 GFX1100 到动态 LDS 列表）
- [ ] 测试 WGP mode 对 GEMM 性能的实际影响
