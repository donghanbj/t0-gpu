# WGP Mode Page Fault 根因分析

## 日期
2026-03-29

## 目标
定位 WGP 模式 GEMM 内核在 GFX1100 (RX 7900 XTX) 上的 GPU hang 根因。

## 方法
1. 编写最小 WGP barrier probe 内核（无 GEMM，纯 barrier + store）
2. 运行后触发 GPU hang → MODE1 reset
3. 分析 dmesg 日志定位硬件级错误

## dmesg 关键证据

### 错误链
```
[73843.071840] sq_intr: error, type 2, sh 0, priv 1, wave_id 0, simd_id 0, wgp_id 0
   ← SQ (Shader Processor) 中断：priv=1 表示特权 wave（CWSR context save wave）
   
[73843.072029] [gfxhub] page fault (ring:40 vmid:8 pasid:32780)
   GCVM_L2_PROTECTION_FAULT_STATUS: 0x00841050
   Faulty UTCL2 client ID: TCP (0x8)  ← Texture Cache 数据访问
   PERMISSION_FAULTS: 0x5            ← 权限全拒（read+write+exec）

[73843.072750] [gfxhub] page fault (ring:88 vmid:8 pasid:32780)
   GCVM_L2_PROTECTION_FAULT_STATUS: 0x008012B0
   Faulty UTCL2 client ID: SQC (inst) (0x9)  ← 指令缓存取指
   PERMISSION_FAULTS: 0xb                    ← 执行权限拒绝

[73845.072581] MES might be in unrecoverable state, issue a GPU reset
[73848.122299] MODE1 reset
[73848.629535] VRAM is lost due to GPU reset!
[73849.102641] GPU reset(8) succeeded!
[73849.116519] device wedged, but recovered through reset
```

### 后续 dispatch 继续失败
```
GPU reset 后新的 queue dispatch 也触发 page fault:
[74753.312155] Faulty UTCL2 client ID: CPF (0x4) ← Command Processor Fetch
   ← CP 无法从 ring buffer 读取 AQL packet
   ← 原因：GPU reset 导致 VA 映射丢失，新 queue 的 mapping 未生效
```

## 根因分析

### 直接原因
WGP 模式内核触发了 **CWSR (Context Wave Save/Restore) page fault**：
- `sq_intr: error, type 2, priv 1` = CWSR 特权 wave 执行错误
- CWSR 试图在 WGP mode 下保存 wave context，但 CWSR buffer 的布局
  假设 CU mode 的 wave 数量和地址空间

### 根本原因假说（3 个候选）

1. **CWSR+WGP 不兼容**（最可能）
   - CWSR buffer 大小按 CU mode 计算（3072 waves × 12B control stack）
   - WGP mode 下每个 WGP 管理的 wave 数量和布局不同
   - CWSR context save 操作使用了不正确的地址 → page fault
   
2. **TLB 失效竞争**
   - WGP mode 下两个 CU 共享 TLB
   - 首次 dispatch 后 TLB 失效窗口期，CWSR preempt 尝试触发了 fault
   
3. **Kernel 驱动 bug**
   - Linux 6.17 的 MES (Micro Engine Scheduler) 可能对 WGP mode 的
     wave preemption 处理有 bug
   - `MES might be in unrecoverable state` 支持此假说

### GPU Reset 后的二次损坏
- `VRAM is lost` → 所有已映射 buffer 的 GPU VA 失效
- 新的 `KfdDevice::open()` + `create_queue()` 分配的 buffer 
  可能映射到了无效的 VMID → CPF page fault
- **结论**：GPU reset 后必须完全重启系统

## 已验证的事实

| 项目 | 结果 |
|------|------|
| CU 模式 GEMM (128×128 k32) | ✅ 79.4 TFLOPS，稳定 |
| CU 模式 barrier + LDS | ✅ 正常 |
| WGP 模式最小 probe（无 LDS） | ❌ GPU hang + page fault |
| WGP ASM 编码（离线验证） | ✅ 正确（bit 29=1, MEM_ORDERED=1, FWD_PROGRESS=1）|
| AQL packet 构建 | ✅ descriptor_va 正确 |
| GPU reset 后恢复 | ❌ 需要完全重启 |

## 解决方案

### 方案 A（推荐）：放弃 WGP，用 CU mode 128×128 k32
- 当前最高性能：79.4 TFLOPS (38% peak)
- 稳定可靠，已在训练中验证
- 通过其他优化路径（ping-pong LDS、更好的调度）提升到 85-90 TF

### 方案 B：kernel 启动参数禁用 CWSR
- `amdgpu.cwsr_enable=0` 添加到 GRUB
- **风险**：GPU 无法做 wave preemption，可能影响多进程 GPU 共享
- 需要重启验证

### 方案 C：升级/降级 Linux kernel
- 尝试 6.14 LTS（已知 RDNA3 稳定性较好）
- 或等待 6.18 修复 MES reissue bug

## 后续
1. ⏳ 需要重启系统清除 GPU reset 状态
2. 重启后运行 CU mode benchmark 确认恢复
3. 评估方案 B（cwsr_enable=0）的可行性
4. 继续 CU mode 优化路径
