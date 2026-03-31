# KCP bit10 / RSRC1 WGP Mode 根因分析

## 日期
2026-03-31

## 目标
调查为什么 `.amdhsa_workgroup_processor_mode 0` 没有改变 KCP bit 10，以及 KFD loader 对 RSRC1 的 WGP bit 使用了错误的位号。

## 方法

### 受控 LLVM 实验
使用最小汇编文件，分别测试三种情况：
1. 无 `.amdhsa_workgroup_processor_mode` 行（LLVM 默认）
2. `.amdhsa_workgroup_processor_mode 1`
3. `.amdhsa_workgroup_processor_mode 0`

对比生成的 ELF 中 KD（Kernel Descriptor）的关键字段。

### KD 布局确认
```
0x00: GROUP_SEGMENT_FIXED_SIZE       (u32)
0x04: PRIVATE_SEGMENT_FIXED_SIZE     (u32)
0x08: KERNARG_SIZE                   (u32)
0x0C: reserved
0x10: KERNEL_CODE_ENTRY_BYTE_OFFSET  (i64)
0x18-0x2B: reserved
0x2C: COMPUTE_PGM_RSRC3              (u32)
0x30: COMPUTE_PGM_RSRC1              (u32)  ← WGP 在这里
0x34: COMPUTE_PGM_RSRC2              (u32)
0x38: KERNEL_CODE_PROPERTIES          (u16) ← KCP
0x3A: KERNARG_PRELOAD_LENGTH          (u16)
0x3C: reserved
```

## 结果

### COMPUTE_PGM_RSRC1 bit 映射（GFX10+ 正确值）
| bit | 字段 | 说明 |
|-----|------|------|
| [5:0] | VGPR_GRANULATED | VGPR 数量编码 |
| [9:6] | SGPR_GRANULATED | SGPR 数量编码 |
| [11:10] | PRIORITY | 队列优先级 |
| [19:12] | FLOAT_MODE | 浮点模式 |
| 20 | PRIV | KFD 需要（CWSR） |
| **29** | **ENABLE_WGP_MODE** | **WGP 模式** |
| 30 | MEM_ORDERED | 内存有序 |
| 31 | FWD_PROGRESS | 前向进度保证 |

### KERNEL_CODE_PROPERTIES bit 映射（Code Object V5）
| bit | 字段 |
|-----|------|
| 3 | ENABLE_SGPR_KERNARG_SEGMENT_PTR |
| 9 | ENABLE_WAVEFRONT_SIZE32 |
| **10** | **USES_DYNAMIC_STACK**（不是 WGP！） |

### 实验数据
| Case | RSRC1@0x30 | bit29(WGP) | KCP@0x38 | KCP bit10 |
|------|-----------|-----------|---------|-----------|
| Default | 0xE0AF0000 | **1** | 0x0400 | 1 (dyn_stack) |
| WGP=1 | 0xE0AF0000 | **1** | 0x0400 | 1 |
| WGP=0 | 0xC0AF0000 | **0** | 0x0400 | 1 |

## 结论

### Bug 1: KFD loader 读取了错误的 KCP bit
KFD loader 读取 KCP bit 10 作为 WGP mode，但实际上 KCP bit 10 是 `USES_DYNAMIC_STACK`（Code Object V5）。WGP mode 不在 KCP 中——它由 LLVM 直接写入 RSRC1 bit 29。

### Bug 2: KFD loader patch 了错误的 RSRC1 bit
KFD 将 WGP flag 写入 RSRC1 bit 27（无效 bit），应该是 bit 29。但实际上根本不需要 KFD 来 patch——LLVM 已经正确设置了 RSRC1 bit 29。

### 修复
1. **移除** KCP→RSRC1 WGP 传播逻辑
2. **保留** RSRC1 bit 20（PRIV）patch（KFD CWSR 需要）
3. **信任** LLVM 的 `.amdhsa_workgroup_processor_mode` 指令

### 性能影响
修复前 RSRC1=0xC8BF..（bit27 被错误设置），修复后 RSRC1=0xC0BF..（干净）。
性能无变化：103.4 TF vs 103.7 TF（正常波动）。

## 铁律

> **COMPUTE_PGM_RSRC1 bit 映射铁律（GFX10+）**：WGP_MODE=bit29，MEM_ORDERED=bit30，FWD_PROGRESS=bit31。不是 bit 27/28/29。LLVM SIDefines.h 是唯一权威来源。

> **KCP 不含 WGP 铁律**：`.amdhsa_workgroup_processor_mode` 直接设置 RSRC1 bit 29，不通过 KERNEL_CODE_PROPERTIES 中转。KCP bit 10 是 USES_DYNAMIC_STACK（V5）。
