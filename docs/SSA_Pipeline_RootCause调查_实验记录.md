# SSA Pipeline VReg 自更新 Root Cause 调查实验记录

## 日期
2026-03-27

## 目标
确认 SSA pipeline 是否有导致 `v_add(x, x, stride)` 被 DCE/CSE 消除的系统性 bug

## 方法
1. 阅读 `ssa_ir.rs` 全部 SSA 管线代码
2. 分析 `lift_to_ssa` use→MVal / def→MVal 解析顺序
3. 分析 `dce_mach_func` Step 2b loop-carried liveness 
4. 编写 3 个最小复现测试（DCE-only / full-pipeline / per-pass bisect）
5. 逐 pass 验证 stride add 是否保留

## 结果
- **SSA pipeline 在隔离测试中完全正确**
- `lift_to_ssa`: use 先解析到旧 MVal，def 再分配新 MVal — 顺序正确
- `dce_mach_func`: 因 `ds_store` 是 side-effect op → root alive → 传播到其 use 的 MVal → stride add 的 def MVal 被保留
- `cse_mach_func`: 不同 use MVals → 不同 CSE key → 不合并
- Full optimize: DCE=0, CSE=0, CopyProp=0 — 零消除

## 结论
1. SSA pipeline core 逻辑无系统性 bug
2. tile_ir 的指令消除是 context-specific（大量 VReg 交互触发边缘情况）
3. Fresh VReg per pass 是**正确的编码规范**，不仅是 workaround
4. 已添加 3 个回归测试防止未来退化

## 后续
- 保持 fresh VReg per pass 作为永久做法
- 文档化 SSA 安全编码规范
- 如果未来遇到类似问题，用 T0_DUMP_ASM=1 + per-pass bisect 方法论隔离
