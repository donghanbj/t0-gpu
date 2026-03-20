# GFX1100 (RDNA3) 开发笔记

从实战中总结的 GFX1100 ISA 开发铁律。

## WMMA 指令

```
v_wmma_f32_16x16x16_bf16  D[v0:v7], A[v8:v15], B[v16:v23], C[v0:v7]
```

- Wave32：每个 wavefront 32 线程
- A/B fragment：各 8 VGPR（16×16 bf16 矩阵，每 VGPR 存 2 个 bf16）
- C/D accumulator：8 VGPR（16×16 f32 矩阵）
- 语义：D += A × B^T（注意 B 是转置的！）
- 每 lane 持有矩阵的一行

## 寄存器编码

| 操作数类型 | 编码 |
|-----------|------|
| VGPR v0-v255 | 在 SRC0 位置：`256 + vgpr_num` |
| SGPR s0-s127 | 直接使用 `0..127` |
| 行内常量 0-64 | `128 + value` |
| 负数行内常量 | `0xC0 + |value|` |
| 浮点常量 0.5 | `0xF0` |
| 浮点常量 1.0 | `0xF2` |
| 浮点常量 2.0 | `0xF4` |

## GFX10 vs GFX11 陷阱

**global_load/store 编码在 GFX11 变了！** 必须用 LLVM 验证：

```bash
echo 'global_load_b32 v0, v[0:1], off' | \
  llvm-mc -mcpu=gfx1100 --show-encoding -triple=amdgcn-amd-amdhsa
```

## 内存访问

- `global_load_b128`：加载 16 bytes（4 DWORD），最高效
- `global_load_b32`：加载 4 bytes
- `global_store_b128`：存储 16 bytes
- 全局内存偏移：13-bit 有符号（-4096 ~ +4095）
- `s_waitcnt vmcnt(0)`：等待所有 VMEM 操作完成

## SMEM 对齐

**SBASE 必须偶数寄存器对齐**：s[0:1], s[2:3], s[4:5]...

错误：`s_load_b64 s[1:2], s[0:1], 0x0` ← s1 不对齐！

## ds_swizzle XOR 模式

用于 wave 内跨 lane 数据交换：

| 步长 | 编码 |
|------|------|
| 16 lanes | `0x401F` |
| 8 lanes | `0x201F` |
| 4 lanes | `0x101F` |
| 2 lanes | `0x081F` |
| 1 lane | `0x041F` |

## 超越函数

- `v_exp_f32`：计算 **2^x**（不是 e^x！）
- `v_log_f32`：计算 **log₂(x)**（不是 ln(x)！）
- e^x = 2^(x / ln2) = v_exp_f32(x * 1.4426950408)
- ln(x) = log₂(x) * ln2 = v_log_f32(x) * 0.6931471805

## EXEC mask

分支执行后必须恢复 EXEC：
```
s_and_saveexec_b32  s_tmp, vcc_lo    // 保存 EXEC 到 s_tmp
// ... 条件执行代码 ...
s_mov_b32  exec_lo, s_tmp             // 恢复 EXEC
```
