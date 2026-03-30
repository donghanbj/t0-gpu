# tile_ir SSA VReg 自更新 Bug 修复实验记录

## 日期
2026-03-27

## 目标
修复 tile_ir GEMM 输出全零/垃圾值问题，恢复正确计算

## 方法
通过正确性验证框架（GPU output → CPU reference 对比）发现问题，
然后 ISA dump 分析定位 3 个 SSA coalescing bug

## 结果

### 修复前
| 测试 | 结果 |
|------|------|
| 128×64 GEMM | 931-6492/8192 bad, max_err=8e36 |
| WGP/split_k | 全零输出 |
| benchmark 82T | **虚假** |

### 修复后
| 测试 | 结果 |
|------|------|
| 128×64 GEMM | **max_err=0.044, 0 bad** ✅ |
| 间歇性 | 1/3 runs GPU state race |

### Root Cause
SSA pipeline 对 `v_add(x, x, imm)` 形式做了错误的 dead-code elimination

## 结论
1. in-place VReg 自更新是 SSA pipeline 已知缺陷
2. 解决方案：每次迭代使用 fresh VReg
3. 正确性验证应作为所有 benchmark 的前置检查

## 后续
- 修复 SSA pipeline root cause（而非逐个 workaround）
- 运行带验证的 benchmark 获取真实 TFLOPS
