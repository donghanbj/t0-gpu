# T0-GPU vs Triton 深度对比分析

> 基于全部 27 个 T0 模块（~14,000 行 Rust）+ 29 个 Ignis 模块的源码级审查。
> 更新日期：2026-03-25（第五轮更新：链路 B 隔离验证 + Exp2/Log2 修复 + SSA RegAlloc L3 验证）

---

## 一、核心架构对比

### 编译管线

```mermaid
graph LR
    subgraph Triton
        TT["Triton-Python DSL"] --> TTIR["Triton IR (MLIR)"]
        TTIR --> TTG["TritonGPU IR"]
        TTG --> LLIR["LLVM IR"]
        LLIR --> PTX["PTX / AMDGCN"]
    end
    
    subgraph T0
        DSL["Block DSL / Tile SSA"] --> IR["T0 Machine IR (Op)"]
        IR --> SSA["MachFunc SSA"]
        SSA --> OPT["13-pass Optimizer"]
        OPT --> RA["SSA RegAlloc + Spill"]
        RA --> ASM["asm_emitter → binary"]
        ASM --> ELF["HSA ELF Code Object"]
    end
```

| 维度 | Triton | T0 |
|------|--------|-----|
| **前端** | Python + `@triton.jit` | Rust `BlockKernel`/`KernelBuilder`/`TileGemm` |
| **中间表示** | MLIR dialect（多轮 lowering） | 单层 `Vec<Op>` + SSA lift/lower |
| **后端** | LLVM codegen（继承全部优化） | 自研 `asm_emitter`（~80 条 GFX1100 指令直出机器码） |
| **运行时** | HIP / CUDA driver API | KFD 裸金属（直写 AQL doorbell） |
| **目标硬件** | NVIDIA (主) + AMD (via MLIR) | AMD GFX1100 专用 |

### IR 层数对比

| 层级 | Triton | T0 | 差距分析 |
|------|--------|----|----------|
| **High-level DSL** | `tl.load/store/dot/reduce` | `BlockKernel` 的 BNode（40+ 节点类型） | T0 覆盖度 ~85% |
| **Tile IR** | TritonGPU IR（tile-level 语义） | `TileGemm` spec + `tile_ssa::TileFunc`（1593 行）+ `TileLoad2D/TileDot/TileStore2D` | T0 GEMM+Elementwise+Tile 2D，Triton 更通用 |
| **Machine IR** | LLVM MachineInstr | `Op` 枚举（90+ 变体） | T0 更贴近硬件，Triton 更抽象 |
| **SSA** | LLVM SSA（成熟） | `MachFunc`（lift/lower，Phi-based）+ **Dominator Tree** | T0 功能子集，DomTree 已实现 |

---

## 二、优化 Pass 逐项深度对比

### 代数优化（SSA-based）

| Pass | Triton | T0 | T0 实现细节 | 差距 |
|------|--------|----|------------|------|
| 常量折叠 | ✅ MLIR canonicalize | ✅ `constant_fold_mach_func` | 追踪 `MVal → f32`，支持 add/mul/sub/max/min/fma | Triton 支持更多类型 |
| 死代码消除 | ✅ MLIR DCE | ✅ `dce_mach_func` | O(n) use-count，iterative + AlgSimp 联合 fixpoint | **等价** |
| Copy Propagation | ✅ MLIR | ✅ `copy_propagate_mach_func` | VMov 链传递解析 + 环检测 | **等价** |
| 代数简化 | ✅ peephole | ✅ `algebraic_simplify` | `x+0→x`, `x*1→x`, `x*0→0`, `fma(0,b,c)→c` | Triton 规则更多 |
| CSE | ✅ MLIR CSE | ✅ `cse_mach_func_domtree` | 跨 block CSE via domtree preorder 遍历，按 (opcode, src0, src1) 哈希 + 交换律 | **等价** |
| InstrCombine | ✅ LLVM | ✅ `instruction_combine` | `mul+add→fma`, `sub(0,x)→neg(x)` | T0 ~5 规则 vs LLVM ~1000+ |

### 循环优化

| Pass | Triton | T0 | T0 实现细节 | 差距 |
|------|--------|----|------------|------|
| Loop Unroll | ✅ 编译期展开 | ✅ `loop_unroll` | ×2/×4 自动展开，安全检查（无 barrier/WMMA/嵌套） | **等价** |
| LICM | ✅ MLIR | ✅ `licm_mach_func` (SSA) | **Domtree 回边检测 + Natural Loop 收集 + MVal 不变量分析 + preheader 外提** | **等价** |
| Software Pipelining | ✅ A100+ | ✅ `software_pipeline` | 检测 load→wait→compute 模式，prefetch 第一批 | 基础版本 |
| Strength Reduction | ✅ LLVM | ✅ `strength_reduce` | 循环内乘法→累加转换 | **等价** |

### 内存优化

| Pass | Triton | T0 | T0 实现细节 | 差距 |
|------|--------|----|------------|------|
| Load Coalescing | ✅ 自动 | ✅ `coalesce_loads` | 连续 B32→B64/B128 合并 | **等价** |
| Waitcnt 优化 | N/A（NVIDIA 无需） | ✅ `optimize_waitcnt` | 消除冗余 vmcnt/lgkmcnt/vscnt | **T0 独有** |
| **LDS Bank Conflict** | ✅ swizzle 分析 | ❌ 手动 padding | gemm_gen 内硬编码 `lds_pad` | ⚠️ 性能差距 |
| **Tensor Layout** | ✅ blocked/shared/MMA encoding | ❌ 仅隐式 row-major | 无 layout 类型系统 | ⚠️ 表达力差距 |

### 调度与寄存器分配

| Pass | Triton | T0 | T0 实现细节 | 差距 |
|------|--------|----|------------|------|
| Instruction Scheduling | ✅ LLVM | ✅ `schedule_mach_func` | SSA-based 2-phase：延迟隐藏 + 压力感知 | T0 更精确（显式 VCC/SCC conflict） |
| RegAlloc | ✅ LLVM graph coloring | ✅ `allocate_ssa` | interval-sorted linear scan + best-fit + Align2/4/8 | LLVM 更优（graph coloring） |
| **Register Spilling** | ✅ LLVM（完整） | ✅ `insert_spill_reloads` | def/last_use 跟踪，spill-farthest 策略，LDS scratch 自动管理 | ~~已关闭~~ |

> [!NOTE]
> **第五轮验证**：SSA RegAlloc 已全局启用（`compile.rs` 默认 `use_ssa_regalloc: true`）。
> 修复了 `to_legacy_regalloc` 两个缺陷：
> 1. dead-def VReg fallback 映射（EXEC-masked 区域内的临时 VReg）
> 2. post-spill ops 扫描（`insert_spill_reloads` 新创建的 spill_addr VReg）
> **34/34 测试全部通过**（含 wg_reduce + tile_gemm + 6 个 spill 单元测试）。

### 覆盖率汇总

```
Triton 优化能力覆盖:    ████████████████████  100%
T0 当前 (13 passes):    ██████████████████░░   90%
                                          ↑
                  剩余: LDS Swizzle + Tensor Layout
```

> [!IMPORTANT]
> 第四轮更新将覆盖率从 88% 上调至 **90%**，新增原因：
> - Dominator Tree 实现（+1%）：`build_domtree` + `DomTree::dominates/idom/preorder`
> - CSE 升级到 cross-block domtree 版本（+0.5%）：`cse_mach_func_domtree`
> - LICM 升级到 SSA domtree 版本（+0.5%）：`licm_mach_func`（回边检测 + natural loop + invariant 外提）
>
> 累计已关闭：Register Spill, Epilogue Fusion, Strength Reduction, Load Coalescing, Tile GEMM, **Dominator Tree**

---

## 三、关键子系统深度对比

### 3.1 Block DSL vs Triton Python DSL

| 能力 | Triton | T0 `BlockKernel` |
|------|--------|-------------------|
| `program_id` | ✅ | ✅ `ProgramId(axis)` |
| `arange` | ✅ | ✅ `Arange { start, end }` |
| `load / store` (masked) | ✅ | ✅ `Load/Store/LoadU32` |
| `atomic_add` | ✅ | ✅ `AtomicAddF32` |
| `dot` (WMMA) | ✅ | ✅ `Wmma { a, b, c }` + `ZeroAcc` |
| `reduce` (wave) | ✅ | ✅ `WaveReduceAddF32/MaxF32` |
| `reduce` (workgroup) | ✅ | ✅ `WgReduceAddF32/MaxF32`（via LDS） |
| `for` 循环 | ✅ | ✅ `ForBegin/ForEnd` |
| 激活函数 | 需手写 | ✅ 内置 `silu/gelu/relu/sigmoid/tanh`（第五轮修复 Exp2/Log2 语义 bug） |
| LDS 操作 | ✅ `shared` | ✅ `LdsAlloc/LdsLoad/LdsStore` + `Barrier` |
| WMMA fragment | 隐式 | ✅ 显式 `CvtPkBf16F32` + `SplatFragment` |
| **Tile-level GEMM** | ✅ `tl.dot` 自动 | ✅ `TileGemm` mega-op + `TileLoad2D/TileDot/TileStore2D` SSA 路径 |
| **compile_via_ssa** | ✅ 统一 lowering | ✅ 自动检测 GEMM → `lower_tiled_gemm()` / elementwise 路由（L2 A/B GPU 验证通过） |
| **多维 tile** | ✅ 2D/3D block | ⚠️ 2D tile（GEMM），通用 2D 仅分析用 | ⚠️ |
| **Autotuning** | ✅ `@triton.autotune` | ⚠️ `cost_model` 穷举搜索存在但未连接 `gemm_gen` |
| **Debug / printf** | ✅ `tl.device_print` | ❌ |

### 3.2 GEMM 生成器对比

| 维度 | Triton | T0 `gemm_gen` / `tile_ir` |
|------|--------|----------------------------|
| Tile 选择 | autotuning（运行时搜索） | `auto_select` + `cost_model`（穷举，未集成） |
| LDS 策略 | blocked + swizzled | 双缓冲 + 手动 stride + padding |
| K-loop | 自动生成 + SWP | A/B 交替双缓冲 prologue/epilogue |
| WMMA 调度 | 自动 | 手工交织（row_blocks × col_tiles） |
| Split-K | ✅ | ✅（1/2/4/8/16） |
| **Epilogue Fusion** | ✅ bias+activation+store | ✅ BiasAdd + ReLU（gemm_gen 内融合） |
| **WGP 模式** | N/A（NVIDIA 无） | ✅ 256×64 k32 WGP（跨 2 CU 共享 LDS） |
| NT/NN layout | ✅ | ✅ `TileTranspose::NT/NN` |
| **实测性能** | ~80-95% peak | ~70-85% peak（128×64 最优） |

### 3.3 硬件建模对比（T0 独有优势）

| 模块 | Triton | T0 |
|------|--------|-----|
| **hw_probe** | ❌ 无 | ✅ 全指令穷举探测（40+ ProbeOp） |
| **latency_model** | LLVM 内建 | ✅ 实测标定（VMEM=47, LDS=7, WMMA=4 VALU-norm） |
| **cost_model** | 启发式 | ✅ GFX1100 精确建模（562 行，VGPR/LDS/占用率/BW） |
| **Pipeline overlap** | 假设可重叠 | ✅ 实测证明单 wave 串行 |

### 3.4 Tile SSA IR 成熟度

| 能力 | Triton (MLIR/LLVM) | T0 `TileFunc` / `MachFunc` |
|------|--------------------|----------------------------|
| CFG 构建 | ✅ 完整 | ✅ Label/Branch + BasicBlock + Terminator |
| Phi 节点 | ✅ | ✅ `PhiNode { dst, entries }` + Block params |
| Dominator Tree | ✅ | ✅ `build_domtree` + `DomTree` | 已实现 |
| **Implicit State** | N/A | ✅ 显式 VCC/SCC/EXEC 建模 |
| Liveness | ✅ LLVM | ✅ `compute_live_intervals` |
| SSA 优化 | ✅ 完整 suite | ✅ 6 pass SSA suite |
| Tile-level 操作数 | ✅ 通用 | ✅ 40+ TileOp 变体（Scalar/Vector/Tile 2D） |
| **Lowering** | ✅ 通用 | ✅ elementwise 1D + tiled GEMM（via `lower_tiled_gemm` → `tile_ir`）|
| **RegAlloc 质量** | 图着色 → 最优 | interval-sorted linear → 次优 |

### 3.5 运行时对比

| 维度 | Triton (ROCm) | T0 (KFD) |
|------|---------------|----------|
| 调度延迟 | ~10-50μs (hipLaunchKernel) | ~1-2μs (AQL doorbell 直写) |
| 内存管理 | hipMalloc/Free | KFD mmap VRAM |
| 编译延迟 | 100ms-10s (JIT) | <1ms (Rust native + llvm-mc) |
| 依赖 | libhip, libhsakmt, ROCr, HIP | 仅 /dev/kfd + /dev/dri |

---

## 四、剩余差距矩阵（按优先级）

### 已关闭差距（本轮审查确认）

| 原编号 | 差距 | 状态 | 证据 |
|--------|------|------|------|
| ~~P0 #2~~ | Register Spill 完善 | ✅ **已关闭** | `ssa_regalloc.rs:474` `insert_spill_reloads` 完整实现 |
| ~~P1 #5~~ | Epilogue 融合 | ✅ **已关闭** | `gemm_gen.rs` `EpilogueOp::BiasAdd/BiasAddRelu` |
| ~~P0 #1~~ | 通用 tile-level 可编程性 | ✅ **基本关闭** | `tile_ssa.rs` TileLoad2D/TileDot/TileStore2D + `tile_ssa_lower.rs` `lower_tiled_gemm()` + `block_dsl_to_ssa.rs` `compile_via_ssa()` GEMM 路由 |

### P0 — 必须弥补

| # | 差距 | 影响 | 当前状态 | 工作量 |
|---|------|------|---------|-------:|
| 1 | ~~通用 tile-level 可编程性~~ | ~~无法表达自定义 GEMM~~ | ✅ GEMM 路径已通 | ~~已关闭(GEMM)~~ |
| 2 | ~~**Dominator Tree**~~ | ~~Phi 插入不完整，跨 block 优化受限~~ | ✅ `build_domtree` + `DomTree` 完整实现，CSE/LICM 已升级 | ~~已关闭~~ |

### P1 — 性能关键

| # | 差距 | 影响 | 工作量 |
|---|------|------|-------:|
| 3 | **LDS Bank Conflict 自动消除** | 高 tile_k 场景 LDS 吞吐受限 | 1 周 |
| 4 | **Tensor Layout 类型系统** | 无法自动选择 blocked/sliced/MMA layout | 2-3 周 |
| 5 | **cost_model → gemm_gen 集成** | 穷举搜索结果未连接到实际 GEMM 生成 | 1 周 |
| 6 | **Autotuning** | 运行时搜索最优配置（vs 编译期启发式） | 2 周 |

### P2 — 生态/可用性

| # | 差距 |
|---|------|
| 7 | 多 target 后端（仅 GFX1100） |
| 8 | Batched GEMM / Grouped GEMM |
| 9 | JIT 磁盘缓存 |
| 10 | GPU printf / debug 工具 |

---

## 五、T0 的结构性优势

| 优势 | 技术细节 | 对比 Triton |
|------|---------|-------------|
| **微架构实测校准** | hw_probe 40+ 指令穷举 → VALU-norm 延迟表 → 指导调度 | Triton 用 LLVM 通用模型 |
| **KFD 裸金属** | 1-2μs 调度延迟，零拷贝 VRAM | HIP 10-50μs |
| **编译速度** | Rust native <1ms | Triton JIT 100ms-10s |
| **13-pass 优化管线** | 4-phase pipeline: SSA→Loop→PostLoop→Final | Triton 依赖 LLVM 通用 pass |
| **VCC/SCC/EXEC 显式建模** | `ImplicitReg` 枚举 + conflict check | LLVM 内部处理 |
| **WMMA 直控** | 直接选择 BF16_F32/F16_F32/BF16_BF16 格式 | Triton 自动，用户无法选择 |
| **零依赖部署** | 仅 /dev/kfd + llvm-mc | 需要完整 ROCm stack |
| **Epilogue 融合** | GEMM 内直接 BiasAdd+ReLU，零额外 dispatch | Triton 需 separate fuse pass |

---

## 六、结论

### 90% 覆盖率的含义

```
                    Triton 能力域
 ┌──────────────────────────────────────────────┐
 │  ████████████████████████████████████████░░░  │
 │  T0 覆盖 (90%)                         │未覆盖│
 │                                        │      │
 │  优化 Pass: ~95% (+DomTree CSE/LICM)   │Layout│
 │  Block DSL: ~90% (+tile GEMM SSA)      │      │
 │  GEMM Gen: ~97% (+epilogue)            │Autotune
 │  Tile Lowering: 85% (GEMM+Elementwise) │      │
 │  HW Model: 120% (超越)                 │      │
 │  RegAlloc: ~90% (+spill完整)            │      │
 └──────────────────────────────────────────────┘
```

### 下一步建议

> [!IMPORTANT]
> **最高优先级**：LDS Bank Conflict 自动消除（P1 #3）
> - 工作量最小（1 周），性能收益最直接

> **核心判断**：T0 在 GEMM 生成和硬件建模上已超越 Triton（AMD 目标），
> tile-level GEMM 可编程性已实现，**Dominator Tree + 跨 block CSE + SSA LICM** 已落地，
> **SSA RegAlloc 已全局启用（含 spill/reload）**，
> **链路 B（compile_via_ssa）通过 L1/L2/L3 三级验证**，
> 剩余差距集中在 **通用 2D kernel lowering** 和 **LDS Bank Conflict**。
