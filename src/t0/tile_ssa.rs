//! T0 Tile SSA IR — Tile-level Static Single Assignment 中间表示
//!
//! 位于 block_dsl（前端 DSL）和 ir.rs（机器指令）之间的核心编译层。
//!
//! # 设计原则
//!
//! 1. **SSA 形式**：每个 Value 只定义一次，依赖关系显式清晰
//! 2. **Tile 语义**：操作的是 2D tile（如 [128, 64] 的矩阵块），不是标量
//! 3. **Shape 自动推导**：每个 Value 携带 dtype + shape，op 创建时自动推导
//! 4. **无硬件细节**：不暴露 VCC/SCC/EXEC/寄存器，下降到 ir.rs 时才出现
//! 5. **BasicBlock 控制流**：支持 for loop、if/else、phi 节点
//!
//! # 架构位置
//!
//! ```text
//! block_dsl.rs (BNode)  ──build──→  tile_ssa.rs (TileSSA)
//!                                        │
//!                                    ┌───▼───┐
//!                                    │ lower  │  (tile_ir.rs 重构)
//!                                    └───┬───┘
//!                                        ▼
//!                                   ir.rs (Vec<Op>)
//!                                        ▼
//!                                   compile.rs → ELF
//! ```
//!
//! # 使用示例
//!
//! ```ignore
//! use t0_gpu::t0::tile_ssa::*;
//!
//! let mut f = TileFunc::new("my_kernel");
//!
//! // 声明 kernel 参数
//! let x_ptr = f.arg_ptr("X");
//! let y_ptr = f.arg_ptr("Y");
//! let n = f.arg_u32("N");
//!
//! // 计算 tile 坐标
//! let pid = f.program_id(0);                         // → Scalar u32
//! let offs = f.arange(0, 128);                       // → Vector [128] u32
//! let idx = f.add(f.splat(f.mul(pid, f.const_u32(128)), 128), offs);
//!
//! // Tile load + 计算 + store
//! let x = f.load(x_ptr, idx, TileDType::F32);        // → Tile [128] f32
//! let two = f.splat(f.const_f32(2.0), 128);
//! let y = f.mul(x, two);                              // → Tile [128] f32
//! f.store(y_ptr, idx, y);
//! ```

use std::fmt;
use std::collections::HashMap;

// ============================================================================
// 类型系统
// ============================================================================

/// 标量数据类型
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarDType {
    F32,
    BF16,
    F16,
    U32,
    I32,
    Bool,
}

impl ScalarDType {
    pub fn bytes(&self) -> u32 {
        match self {
            ScalarDType::F32 | ScalarDType::U32 | ScalarDType::I32 => 4,
            ScalarDType::BF16 | ScalarDType::F16 => 2,
            ScalarDType::Bool => 1,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, ScalarDType::F32 | ScalarDType::BF16 | ScalarDType::F16)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, ScalarDType::U32 | ScalarDType::I32)
    }
}

// ============================================================================
// Tensor Layout — 数据在 thread/wave/WG 间的分布方式
// ============================================================================

/// Tensor 在 workgroup 线程间的分布方式。
///
/// 对应 Triton 的 layout encoding (#blocked, #shared, #mma)。
/// 每种 layout 告诉 lowering pass：谁拥有哪些元素、如何协作加载/存储。
///
/// # Layout 含义
///
/// - `Blocked`: 1D 分块，每线程连续拥有 `elems_per_thread` 个元素
/// - `Blocked2D`: 2D 分块，线程沿 (row, col) 分工
/// - `Shared`: 数据在 LDS 中，所有线程可访问（WMMA 操作数）
/// - `MmaAccumulator`: 数据在 VGPR 中，按 WMMA 16×16 output layout
/// - `Scalar`: 无分布（标量/指针）
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    /// 1D blocked: Thread t 拥有 [t*ept, (t+1)*ept) 范围
    /// 对应 Triton `#blocked{sizePerThread=[ept], threadsPerWarp=[32]}`
    Blocked {
        elems_per_thread: u32,
    },

    /// 2D blocked: Thread (tx, ty) 拥有 tile 的一个子区域
    /// tx = flat_tid & (block_x - 1), ty = flat_tid >> log2(block_x)
    Blocked2D {
        rows_per_thread: u32,
        cols_per_thread: u32,
        block_x: u32,  // X 轴线程数 (= tile_cols / cols_per_thread)
    },

    /// LDS 共享存储: 数据在 LDS 中，所有线程可访问
    /// 对应 Triton `#shared` encoding (用于 WMMA 操作数)
    Shared {
        pad_bytes: u32,     // 行尾 padding (消除 bank conflict)
        swizzle: bool,      // 是否使用 swizzle 模式
    },

    /// WMMA 累加器: 数据在寄存器中，按 WMMA 16×16 output layout
    /// 对应 Triton `#mma` encoding
    MmaAccumulator {
        m_per_wave: u32,    // 每个 wave 负责的 M 行数 (16 的倍数)
        n_tiles: u32,       // N 方向的 16-col tile 数
    },

    /// 标量/指针: 无分布
    Scalar,
}

impl TensorLayout {
    /// Default layout for 1D vectors: 1 element per thread
    pub fn blocked_default() -> Self {
        TensorLayout::Blocked { elems_per_thread: 1 }
    }

    /// Default layout for 2D tiles: deferred (lowering decides)
    pub fn blocked2d_default(cols: u32) -> Self {
        TensorLayout::Blocked2D {
            rows_per_thread: 1,
            cols_per_thread: 1,
            block_x: cols.min(32),
        }
    }

    /// Shared layout with optional padding
    pub fn shared(pad_bytes: u32) -> Self {
        TensorLayout::Shared { pad_bytes, swizzle: false }
    }

    /// MMA accumulator layout
    pub fn mma(m_per_wave: u32, n_tiles: u32) -> Self {
        TensorLayout::MmaAccumulator { m_per_wave, n_tiles }
    }

    pub fn is_shared(&self) -> bool { matches!(self, TensorLayout::Shared { .. }) }
    pub fn is_mma(&self) -> bool { matches!(self, TensorLayout::MmaAccumulator { .. }) }
    pub fn is_blocked(&self) -> bool { matches!(self, TensorLayout::Blocked { .. } | TensorLayout::Blocked2D { .. }) }
}

impl fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorLayout::Blocked { elems_per_thread } =>
                write!(f, "#blocked{{{}}}", elems_per_thread),
            TensorLayout::Blocked2D { rows_per_thread, cols_per_thread, block_x } =>
                write!(f, "#blocked2d{{{},{}; bx={}}}", rows_per_thread, cols_per_thread, block_x),
            TensorLayout::Shared { pad_bytes, swizzle } =>
                write!(f, "#shared{{pad={}, swz={}}}", pad_bytes, swizzle),
            TensorLayout::MmaAccumulator { m_per_wave, n_tiles } =>
                write!(f, "#mma{{m={}, n={}}}", m_per_wave, n_tiles),
            TensorLayout::Scalar =>
                write!(f, "#scalar"),
        }
    }
}

/// SSA Value 的类型：标量、向量（1D tile）、矩阵（2D tile）、指针
///
/// 每个 tile 类型携带 layout 信息，描述数据如何分布在线程间。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileType {
    /// 标量值（单个元素）
    Scalar(ScalarDType),
    /// 1D Tile（向量），如 [128] f32 #blocked{1}
    Vector { len: u32, dtype: ScalarDType, layout: TensorLayout },
    /// 2D Tile（矩阵块），如 [16, 64] bf16 #shared{pad=4}
    Tile { rows: u32, cols: u32, dtype: ScalarDType, layout: TensorLayout },
    /// 指针（指向全局内存）
    Ptr,
}

impl TileType {
    pub fn scalar_f32() -> Self { TileType::Scalar(ScalarDType::F32) }
    pub fn scalar_u32() -> Self { TileType::Scalar(ScalarDType::U32) }
    pub fn scalar_bool() -> Self { TileType::Scalar(ScalarDType::Bool) }

    /// Create 1D vector with default Blocked layout (1 elem/thread)
    pub fn vector(len: u32, dtype: ScalarDType) -> Self {
        TileType::Vector { len, dtype, layout: TensorLayout::blocked_default() }
    }

    /// Create 1D vector with explicit layout
    pub fn vector_with_layout(len: u32, dtype: ScalarDType, layout: TensorLayout) -> Self {
        TileType::Vector { len, dtype, layout }
    }

    /// Create 2D tile with default Blocked2D layout
    pub fn tile(rows: u32, cols: u32, dtype: ScalarDType) -> Self {
        TileType::Tile { rows, cols, dtype, layout: TensorLayout::blocked2d_default(cols) }
    }

    /// Create 2D tile with explicit layout (e.g. Shared for LDS, MmaAccumulator for WMMA)
    pub fn tile_with_layout(rows: u32, cols: u32, dtype: ScalarDType, layout: TensorLayout) -> Self {
        TileType::Tile { rows, cols, dtype, layout }
    }

    /// 获取标量 dtype（如果存在）
    pub fn dtype(&self) -> Option<ScalarDType> {
        match self {
            TileType::Scalar(d) => Some(*d),
            TileType::Vector { dtype, .. } => Some(*dtype),
            TileType::Tile { dtype, .. } => Some(*dtype),
            TileType::Ptr => None,
        }
    }

    /// 获取元素总数
    pub fn numel(&self) -> u32 {
        match self {
            TileType::Scalar(_) => 1,
            TileType::Vector { len, .. } => *len,
            TileType::Tile { rows, cols, .. } => rows * cols,
            TileType::Ptr => 1,
        }
    }

    /// 获取 layout
    pub fn layout(&self) -> &TensorLayout {
        match self {
            TileType::Vector { layout, .. } => layout,
            TileType::Tile { layout, .. } => layout,
            TileType::Scalar(_) | TileType::Ptr => &TensorLayout::Scalar,
        }
    }

    /// 返回相同 shape/dtype 但不同 layout 的新类型
    pub fn with_layout(&self, new_layout: TensorLayout) -> Self {
        match self {
            TileType::Vector { len, dtype, .. } =>
                TileType::Vector { len: *len, dtype: *dtype, layout: new_layout },
            TileType::Tile { rows, cols, dtype, .. } =>
                TileType::Tile { rows: *rows, cols: *cols, dtype: *dtype, layout: new_layout },
            other => other.clone(),
        }
    }

    /// 是否为标量
    pub fn is_scalar(&self) -> bool { matches!(self, TileType::Scalar(_)) }

    /// 是否为 tile（向量或矩阵）
    pub fn is_tile(&self) -> bool {
        matches!(self, TileType::Vector { .. } | TileType::Tile { .. })
    }
}

impl fmt::Display for TileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileType::Scalar(d) => write!(f, "{:?}", d),
            TileType::Vector { len, dtype, layout } =>
                write!(f, "[{}]{:?} {}", len, dtype, layout),
            TileType::Tile { rows, cols, dtype, layout } =>
                write!(f, "[{},{}]{:?} {}", rows, cols, dtype, layout),
            TileType::Ptr => write!(f, "ptr"),
        }
    }
}

// ============================================================================
// SSA Value — 唯一标识的值引用
// ============================================================================

/// SSA 值引用。每个值只被定义一次（SSA 性质）。
///
/// Value 是轻量 handle（u32 索引），实际数据存储在 TileFunc 中。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub u32);

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Value 的定义信息
#[derive(Clone, Debug)]
pub struct ValueDef {
    /// 唯一 ID
    pub id: Value,
    /// 类型（含 shape + dtype）
    pub ty: TileType,
    /// 定义该值的指令（None = 函数参数）
    pub defining_op: Option<usize>,
    /// 所在的 BasicBlock
    pub block: BlockId,
    /// 可选名称（用于调试）
    pub name: Option<String>,
}

// ============================================================================
// BasicBlock — 控制流基本块
// ============================================================================

/// 基本块标识
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// 基本块——指令的线性序列 + 终结指令
#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub id: BlockId,
    /// 块内指令索引列表（指向 TileFunc.ops）
    pub ops: Vec<usize>,
    /// 终结指令（控制流转移）
    pub terminator: Option<Terminator>,
    /// 块参数（用于 phi 节点替代）
    pub params: Vec<Value>,
}

/// 基本块终结指令
#[derive(Clone, Debug)]
pub enum Terminator {
    /// 无条件跳转
    Branch { target: BlockId, args: Vec<Value> },
    /// 条件跳转
    CondBranch {
        cond: Value,
        true_bb: BlockId, true_args: Vec<Value>,
        false_bb: BlockId, false_args: Vec<Value>,
    },
    /// 函数返回
    Return,
}

/// For 循环句柄（由 `for_range` 返回）
#[derive(Clone, Debug)]
pub struct ForLoop {
    pub header: BlockId,
    pub body: BlockId,
    pub exit: BlockId,
    /// 循环归纳变量（i）
    pub iv: Value,
    /// 步长
    pub step: u32,
}

/// 带累加器的 For 循环句柄（由 `for_range_with_acc` 返回）
#[derive(Clone, Debug)]
pub struct ForLoopAcc {
    pub header: BlockId,
    pub body: BlockId,
    pub exit: BlockId,
    /// 循环归纳变量（i）
    pub iv: Value,
    /// 循环体内的累加器（读取用）
    pub acc: Value,
    /// exit block 的累加器结果（循环结束后使用）
    pub result: Value,
    /// 步长
    pub step: u32,
}

// ============================================================================
// TileOp — Tile-level SSA 操作
// ============================================================================

/// Tile-level SSA 操作。
///
/// 每个操作定义零个或一个 Value（SSA 结果）。
/// 操作的语义是 tile-level 的——load/store 操作整个 tile，dot 是矩阵乘。
#[derive(Clone, Debug)]
pub enum TileOp {
    // ── 常量 ──
    /// 标量整数常量
    ConstU32 { result: Value, value: u32 },
    /// 标量浮点常量
    ConstF32 { result: Value, value: f32 },

    // ── 索引 ──
    /// 获取 program ID（workgroup ID）沿指定轴
    ProgramId { result: Value, axis: u8 },
    /// 生成连续索引向量 [start, start+1, ..., start+len-1]
    Arange { result: Value, start: u32, len: u32 },
    /// 2D workgroup thread X index: flat_tid & (block_x - 1)
    ThreadIdX2D { result: Value, block_x: u32 },
    /// 2D workgroup thread Y index: flat_tid >> log2(block_x)
    ThreadIdY2D { result: Value, block_x: u32 },

    // ── Shape 操作 ──
    /// 标量广播为向量/tile
    Splat { result: Value, src: Value, shape: Vec<u32> },
    /// reshape（元素数量不变）
    Reshape { result: Value, src: Value, shape: Vec<u32> },
    /// 扩展维度
    ExpandDims { result: Value, src: Value, axis: u32 },

    // ── 内存操作 ──
    /// Tile 加载：从全局内存加载一个 tile
    /// ptr[idx] → tile, 带可选 mask 和其他值(other)
    Load {
        result: Value,
        ptr: Value,
        indices: Value,
        mask: Option<Value>,
        other: Option<Value>,
        dtype: ScalarDType,
    },
    /// Tile 存储：将 tile 写入全局内存
    Store {
        ptr: Value,
        indices: Value,
        val: Value,
        mask: Option<Value>,
    },

    // ── 逐元素算术 ──
    /// 二元算术操作（add/sub/mul/div 等）
    BinOp { result: Value, op: BinOpKind, lhs: Value, rhs: Value },
    /// 一元操作（neg/exp/log/sqrt/rcp 等）
    UnaryOp { result: Value, op: UnaryOpKind, src: Value },
    /// FMA: result = a * b + c
    Fma { result: Value, a: Value, b: Value, c: Value },
    /// 类型转换
    Cast { result: Value, src: Value, to: ScalarDType },

    // ── 比较 ──
    /// 比较操作，返回 bool tile
    Cmp { result: Value, op: CmpOpKind, lhs: Value, rhs: Value },
    /// 条件选择: result = cond ? true_val : false_val
    Select { result: Value, cond: Value, true_val: Value, false_val: Value },

    // ── 归约 ──
    /// 沿指定轴归约
    Reduce { result: Value, src: Value, axis: u32, op: ReduceKind },

    // ── Tile 线性代数 ──
    /// 矩阵乘: result = dot(a, b)
    /// a: [M, K], b: [K, N] → result: [M, N]
    Dot { result: Value, a: Value, b: Value },

    // ── 同步 ──
    /// Workgroup barrier
    Barrier,

    // ── LDS（共享内存）──
    /// 分配 LDS 空间，返回基地址 (u32 byte offset)
    LdsAlloc { result: Value, size_bytes: u32 },
    /// 从 LDS 加载 f32: val = lds[base + offset * 4]
    LdsLoad { result: Value, base: Value, offset: Value },
    /// 写入 LDS: lds[base + offset * 4] = val
    LdsStore { base: Value, offset: Value, val: Value },

    // ── 原子操作 ──
    /// global_atomic_add_f32: ptr[indices] += val (with optional mask)
    AtomicAddF32 { ptr: Value, indices: Value, val: Value, mask: Option<Value> },

    // ── WMMA / BF16 ──
    /// 8×f32 零初始化累加器 (WMMA accumulator)
    ZeroAcc { result: Value },
    /// f32×2 → bf16x2 打包: pack(lo, hi) → u32
    CvtPkBf16F32 { result: Value, lo: Value, hi: Value },
    /// WMMA: result = a × b + c (16×16×16 bf16 → f32)
    /// a, b: bf16 fragment (8×u32), c: f32 accumulator (8×f32)
    WmmaF32 { result: Value, a: Value, b: Value, c: Value },
    /// 从 8×f32 累加器提取第 idx 个元素
    ExtractF32 { result: Value, src: Value, idx: u32 },
    /// 将单个 u32/f32 广播到8×VGPR fragment
    SplatFragment { result: Value, src: Value },

    // ── WG 级归约 ──
    /// Workgroup-level sum reduction (跨 wave，通过 LDS)
    WgReduceAdd { result: Value, src: Value, block_size: u32 },
    /// Workgroup-level max reduction (跨 wave，通过 LDS)
    WgReduceMax { result: Value, src: Value, block_size: u32 },

    // ── Tile-Level 2D 操作（Triton 语义）──
    //
    // 用户用 2D tile 思考，编译器自动处理：
    //   - per-thread 协作加载分工
    //   - LDS 分配 + bank-conflict-free padding
    //   - WMMA fragment 读取 + 调度
    //   - 双缓冲（K-loop 模式检测后自动启用）

    /// 2D Tile 加载：WG 协作加载 [rows, cols] tile 到 LDS
    /// 等价于 Triton: a = tl.load(ptr + offs_m[:, None] * stride + offs_k[None, :])
    /// 结果是一个 opaque [rows, cols] tile handle（lowering 时解析为 LDS 区域）
    TileLoad2D {
        result: Value,
        ptr: Value,          // 全局基地址 (Ptr)
        row_offset: Value,   // tile 起始行 (Scalar u32)
        col_offset: Value,   // tile 起始列/K偏移 (Scalar u32)
        stride: Value,       // 行间距，元素数 (Scalar u32)
        rows: u32,           // tile 行数 (编译期常量)
        cols: u32,           // tile 列数 (编译期常量)
        dtype: ScalarDType,  // 元素类型 (BF16 for WMMA)
    },

    /// Tile 矩阵乘累加：acc += dot(a, b)
    /// 等价于 Triton: acc = tl.dot(a, b, acc)
    /// a: [M, K] tile, b: [K, N] tile → acc: [M, N] f32
    /// 编译器自动处理 LDS→fragment 读取 + WMMA 16x16x16 调度
    TileDot {
        result: Value,       // 更新后的 [M, N] f32 accumulator
        a: Value,            // [M, K] tile (来自 TileLoad2D)
        b: Value,            // [K, N] tile (来自 TileLoad2D)
        acc: Value,          // [M, N] f32 accumulator (初始来自 tile_zeros)
    },

    /// 2D Tile 存储：将 [M, N] f32 tile 写入全局内存
    /// 等价于 Triton: tl.store(ptr + offs_m[:, None] * stride + offs_n[None, :], acc)
    /// 编译器自动处理 WMMA 输出布局解码 + coalesced global stores
    TileStore2D {
        ptr: Value,          // 全局基地址 (Ptr)
        row_offset: Value,   // tile 起始行 (Scalar u32)
        col_offset: Value,   // tile 起始列 (Scalar u32)
        stride: Value,       // 行间距，元素数 (Scalar u32)
        val: Value,          // [M, N] f32 tile (accumulator)
    },

    // ── EXEC Mask（条件执行）──
    /// Push EXEC mask: saved = EXEC; EXEC &= mask
    /// 仅 mask!=0 的 lane 后续指令生效。支持嵌套（栈式保存）。
    /// 等价于 IfMask — maps to: v_cmp + s_and_saveexec_b32
    ExecMaskPush { mask: Value },
    /// Flip EXEC mask: EXEC ^= saved (切换到 else 分支的 lane)
    /// 等价于 ElseMask — maps to: s_xor_b32 exec_lo, exec_lo, saved
    ExecMaskFlip,
    /// Pop EXEC mask: EXEC = saved (恢复原始 mask)
    /// 等价于 EndIf — maps to: s_mov_b32 exec_lo, saved
    ExecMaskPop,
}

/// 二元算术操作类型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOpKind {
    Add, Sub, Mul, Div,
    Rem,
    And, Or, Xor,
    Shl, Shr,
    Max, Min,
}

/// 一元操作类型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOpKind {
    Neg, Exp, Log, Sqrt, Rcp, Rsqrt, Abs,
    Sigmoid, Relu, Silu,
    Sin, Cos,
    /// Raw hardware 2^x (v_exp_f32), no log2(e) scaling
    Exp2,
    /// Raw hardware log₂(x) (v_log_f32), no ln(2) scaling
    Log2,
}

/// 比较操作类型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOpKind {
    Eq, Ne, Lt, Le, Gt, Ge,
}

/// 归约操作类型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceKind {
    Sum, Max, Min, Prod,
}

// ============================================================================
// TileFunc — SSA 函数（一个 GPU kernel）
// ============================================================================

/// 一个 Tile-level SSA 函数，对应一个 GPU kernel。
///
/// 包含所有 Value 定义、BasicBlock、TileOp 指令。
/// 提供 builder API 用于构建 SSA 程序。
pub struct TileFunc {
    pub name: String,
    /// 所有 Value 的定义
    values: Vec<ValueDef>,
    /// 所有指令
    ops: Vec<TileOp>,
    /// 所有基本块
    blocks: Vec<BasicBlock>,
    /// 函数参数（按声明顺序）
    pub args: Vec<Value>,
    /// 当前正在构建的基本块
    current_block: BlockId,
    /// 下一个 Value ID
    next_value: u32,
}

impl TileFunc {
    /// 创建新的 SSA 函数
    pub fn new(name: &str) -> Self {
        let entry_block = BasicBlock {
            id: BlockId(0),
            ops: Vec::new(),
            terminator: None,
            params: Vec::new(),
        };
        TileFunc {
            name: name.to_string(),
            values: Vec::new(),
            ops: Vec::new(),
            blocks: vec![entry_block],
            args: Vec::new(),
            current_block: BlockId(0),
            next_value: 0,
        }
    }

    // ── 内部 helpers ──

    pub fn alloc_value(&mut self, ty: TileType, name: Option<&str>) -> Value {
        let id = Value(self.next_value);
        self.next_value += 1;
        self.values.push(ValueDef {
            id,
            ty,
            defining_op: None,
            block: self.current_block,
            name: name.map(|s| s.to_string()),
        });
        id
    }

    pub fn push_op(&mut self, op: TileOp) -> usize {
        let idx = self.ops.len();
        // 设置 result 的 defining_op
        if let Some(result) = op_result(&op) {
            self.values[result.0 as usize].defining_op = Some(idx);
        }
        self.blocks[self.current_block.0 as usize].ops.push(idx);
        self.ops.push(op);
        idx
    }

    // ── 查询 API ──

    /// 获取 Value 的类型
    pub fn value_type(&self, v: Value) -> &TileType {
        &self.values[v.0 as usize].ty
    }

    /// 获取 Value 的 dtype（如果存在）
    pub fn value_dtype(&self, v: Value) -> Option<ScalarDType> {
        self.values[v.0 as usize].ty.dtype()
    }

    /// 获取所有指令
    pub fn all_ops(&self) -> &[TileOp] { &self.ops }

    /// 获取所有基本块
    pub fn all_blocks(&self) -> &[BasicBlock] { &self.blocks }

    /// 获取所有 Value 定义
    pub fn all_values(&self) -> &[ValueDef] { &self.values }

    // ════════════════════════════════════════════════════════
    // Builder API — 构建 SSA 程序
    // ════════════════════════════════════════════════════════

    // ── 参数声明 ──

    /// 声明指针参数
    pub fn arg_ptr(&mut self, name: &str) -> Value {
        let v = self.alloc_value(TileType::Ptr, Some(name));
        self.args.push(v);
        v
    }

    /// 声明 u32 标量参数
    pub fn arg_u32(&mut self, name: &str) -> Value {
        let v = self.alloc_value(TileType::scalar_u32(), Some(name));
        self.args.push(v);
        v
    }

    /// 声明 f32 标量参数
    pub fn arg_f32(&mut self, name: &str) -> Value {
        let v = self.alloc_value(TileType::scalar_f32(), Some(name));
        self.args.push(v);
        v
    }

    // ── 常量 ──

    pub fn const_u32(&mut self, value: u32) -> Value {
        let v = self.alloc_value(TileType::scalar_u32(), None);
        self.push_op(TileOp::ConstU32 { result: v, value });
        v
    }

    pub fn const_f32(&mut self, value: f32) -> Value {
        let v = self.alloc_value(TileType::scalar_f32(), None);
        self.push_op(TileOp::ConstF32 { result: v, value });
        v
    }

    // ── 索引 ──

    /// 获取 program_id (workgroup id) 沿指定轴
    pub fn program_id(&mut self, axis: u8) -> Value {
        let v = self.alloc_value(TileType::scalar_u32(), None);
        self.push_op(TileOp::ProgramId { result: v, axis });
        v
    }

    /// 生成连续索引向量 [start, start+1, ..., start+len-1]
    pub fn arange(&mut self, start: u32, len: u32) -> Value {
        let v = self.alloc_value(TileType::vector(len, ScalarDType::U32), None);
        self.push_op(TileOp::Arange { result: v, start, len });
        v
    }

    // ── Shape 操作 ──

    /// 标量 → 向量/tile 广播
    pub fn splat(&mut self, src: Value, len: u32) -> Value {
        let src_ty = self.value_type(src).clone();
        let dtype = src_ty.dtype().expect("splat: source must have dtype");
        assert!(src_ty.is_scalar(), "splat: source must be scalar, got {}", src_ty);
        let v = self.alloc_value(TileType::vector(len, dtype), None);
        self.push_op(TileOp::Splat { result: v, src, shape: vec![len] });
        v
    }

    /// 2D splat: 标量 → [rows, cols] tile
    pub fn splat_2d(&mut self, src: Value, rows: u32, cols: u32) -> Value {
        let src_ty = self.value_type(src).clone();
        let dtype = src_ty.dtype().expect("splat_2d: source must have dtype");
        assert!(src_ty.is_scalar(), "splat_2d: source must be scalar, got {}", src_ty);
        let v = self.alloc_value(TileType::tile(rows, cols, dtype), None);
        self.push_op(TileOp::Splat { result: v, src, shape: vec![rows, cols] });
        v
    }

    // ── 内存操作 ──

    /// Tile load：ptr[indices] → tile
    pub fn load(&mut self, ptr: Value, indices: Value, dtype: ScalarDType) -> Value {
        assert_eq!(self.value_type(ptr), &TileType::Ptr, "load: first arg must be ptr");
        let idx_ty = self.value_type(indices).clone();
        let result_ty = match &idx_ty {
            TileType::Vector { len, .. } => TileType::vector(*len, dtype),
            TileType::Tile { rows, cols, .. } => TileType::tile(*rows, *cols, dtype),
            TileType::Scalar(_) => TileType::Scalar(dtype),
            _ => panic!("load: indices must be scalar/vector/tile, got {}", idx_ty),
        };
        let v = self.alloc_value(result_ty, None);
        self.push_op(TileOp::Load { result: v, ptr, indices, mask: None, other: None, dtype });
        v
    }

    /// Tile load with mask：ptr[indices] where mask → tile
    pub fn load_masked(&mut self, ptr: Value, indices: Value, mask: Value,
                       other: Value, dtype: ScalarDType) -> Value {
        assert_eq!(self.value_type(ptr), &TileType::Ptr, "load: first arg must be ptr");
        let idx_ty = self.value_type(indices).clone();
        let result_ty = match &idx_ty {
            TileType::Vector { len, .. } => TileType::vector(*len, dtype),
            TileType::Tile { rows, cols, .. } => TileType::tile(*rows, *cols, dtype),
            _ => panic!("load_masked: indices must be vector/tile, got {}", idx_ty),
        };
        let v = self.alloc_value(result_ty, None);
        self.push_op(TileOp::Load {
            result: v, ptr, indices, mask: Some(mask), other: Some(other), dtype
        });
        v
    }

    /// Tile store：val → ptr[indices]
    pub fn store(&mut self, ptr: Value, indices: Value, val: Value) {
        assert_eq!(self.value_type(ptr), &TileType::Ptr, "store: first arg must be ptr");
        self.push_op(TileOp::Store { ptr, indices, val, mask: None });
    }

    /// Tile store with mask：val → ptr[indices] where mask
    pub fn store_masked(&mut self, ptr: Value, indices: Value, val: Value, mask: Value) {
        assert_eq!(self.value_type(ptr), &TileType::Ptr, "store_masked: first arg must be ptr");
        self.push_op(TileOp::Store { ptr, indices, val, mask: Some(mask) });
    }

    /// Reduce (sum) along axis 0
    pub fn reduce_add(&mut self, src: Value) -> Value {
        let src_ty = self.value_type(src).clone();
        let dtype = src_ty.dtype().expect("reduce_add: src must have dtype");
        let v = self.alloc_value(TileType::Scalar(dtype), None);
        self.push_op(TileOp::Reduce { result: v, src, axis: 0, op: ReduceKind::Sum });
        v
    }

    // ── 算术操作 ──

    fn binop(&mut self, op: BinOpKind, lhs: Value, rhs: Value) -> Value {
        let lty = self.value_type(lhs).clone();
        let rty = self.value_type(rhs).clone();
        let result_ty = infer_binop_type(&lty, &rty, op);
        let v = self.alloc_value(result_ty, None);
        self.push_op(TileOp::BinOp { result: v, op, lhs, rhs });
        v
    }

    /// Public binop with explicit BinOpKind (for translator use)
    pub fn binop_raw(&mut self, op: BinOpKind, lhs: Value, rhs: Value) -> Value {
        self.binop(op, lhs, rhs)
    }

    pub fn add(&mut self, lhs: Value, rhs: Value) -> Value { self.binop(BinOpKind::Add, lhs, rhs) }
    pub fn sub(&mut self, lhs: Value, rhs: Value) -> Value { self.binop(BinOpKind::Sub, lhs, rhs) }
    pub fn mul(&mut self, lhs: Value, rhs: Value) -> Value { self.binop(BinOpKind::Mul, lhs, rhs) }
    pub fn div(&mut self, lhs: Value, rhs: Value) -> Value { self.binop(BinOpKind::Div, lhs, rhs) }
    pub fn max(&mut self, lhs: Value, rhs: Value) -> Value { self.binop(BinOpKind::Max, lhs, rhs) }
    pub fn min(&mut self, lhs: Value, rhs: Value) -> Value { self.binop(BinOpKind::Min, lhs, rhs) }

    fn unaryop(&mut self, op: UnaryOpKind, src: Value) -> Value {
        let src_ty = self.value_type(src).clone();
        let v = self.alloc_value(src_ty, None);
        self.push_op(TileOp::UnaryOp { result: v, op, src });
        v
    }

    pub fn neg(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Neg, src) }
    pub fn exp(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Exp, src) }
    pub fn log(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Log, src) }
    pub fn sqrt(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Sqrt, src) }
    pub fn rcp(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Rcp, src) }
    pub fn rsqrt(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Rsqrt, src) }
    pub fn abs(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Abs, src) }
    pub fn sigmoid(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Sigmoid, src) }
    pub fn relu(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Relu, src) }
    pub fn silu(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Silu, src) }
    pub fn sin(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Sin, src) }
    pub fn cos(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Cos, src) }
    /// Raw hardware 2^x (maps to v_exp_f32 without log2e scaling)
    pub fn exp2(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Exp2, src) }
    /// Raw hardware log₂(x) (maps to v_log_f32 without ln2 scaling)
    pub fn log2(&mut self, src: Value) -> Value { self.unaryop(UnaryOpKind::Log2, src) }

    /// FMA: result = a * b + c
    pub fn fma(&mut self, a: Value, b: Value, c: Value) -> Value {
        let ty = self.value_type(a).clone();
        let v = self.alloc_value(ty, None);
        self.push_op(TileOp::Fma { result: v, a, b, c });
        v
    }

    /// 类型转换
    pub fn cast(&mut self, src: Value, to: ScalarDType) -> Value {
        let src_ty = self.value_type(src).clone();
        let result_ty = match &src_ty {
            TileType::Scalar(_) => TileType::Scalar(to),
            TileType::Vector { len, .. } => TileType::vector(*len, to),
            TileType::Tile { rows, cols, .. } => TileType::tile(*rows, *cols, to),
            TileType::Ptr => panic!("cast: cannot cast ptr"),
        };
        let v = self.alloc_value(result_ty, None);
        self.push_op(TileOp::Cast { result: v, src, to });
        v
    }

    // ── 比较 ──

    pub fn cmp(&mut self, op: CmpOpKind, lhs: Value, rhs: Value) -> Value {
        let lty = self.value_type(lhs).clone();
        let result_ty = match &lty {
            TileType::Scalar(_) => TileType::scalar_bool(),
            TileType::Vector { len, .. } => TileType::vector(*len, ScalarDType::Bool),
            TileType::Tile { rows, cols, .. } => TileType::tile(*rows, *cols, ScalarDType::Bool),
            _ => panic!("cmp: unsupported type {}", lty),
        };
        let v = self.alloc_value(result_ty, None);
        self.push_op(TileOp::Cmp { result: v, op, lhs, rhs });
        v
    }

    pub fn cmp_lt(&mut self, lhs: Value, rhs: Value) -> Value {
        self.cmp(CmpOpKind::Lt, lhs, rhs)
    }

    pub fn cmp_ge(&mut self, lhs: Value, rhs: Value) -> Value {
        self.cmp(CmpOpKind::Ge, lhs, rhs)
    }

    /// 条件选择: cond ? true_val : false_val
    pub fn select(&mut self, cond: Value, true_val: Value, false_val: Value) -> Value {
        let ty = self.value_type(true_val).clone();
        let v = self.alloc_value(ty, None);
        self.push_op(TileOp::Select { result: v, cond, true_val, false_val });
        v
    }

    // ── 归约 ──

    /// 沿指定轴求和归约
    pub fn sum(&mut self, src: Value, axis: u32) -> Value {
        self.reduce(src, axis, ReduceKind::Sum)
    }

    /// 沿指定轴取最大值归约
    pub fn reduce_max(&mut self, src: Value, axis: u32) -> Value {
        self.reduce(src, axis, ReduceKind::Max)
    }

    fn reduce(&mut self, src: Value, axis: u32, op: ReduceKind) -> Value {
        let src_ty = self.value_type(src).clone();
        let result_ty = infer_reduce_type(&src_ty, axis);
        let v = self.alloc_value(result_ty, None);
        self.push_op(TileOp::Reduce { result: v, src, axis, op });
        v
    }

    // ── Dot ──

    /// 矩阵乘: a[M,K] @ b[K,N] → [M,N]
    pub fn dot(&mut self, a: Value, b: Value) -> Value {
        let aty = self.value_type(a).clone();
        let bty = self.value_type(b).clone();
        let result_ty = infer_dot_type(&aty, &bty);
        let v = self.alloc_value(result_ty, None);
        self.push_op(TileOp::Dot { result: v, a, b });
        v
    }

    // ── 同步 ──

    pub fn barrier(&mut self) {
        self.push_op(TileOp::Barrier);
    }

    // ── LDS ──

    /// 分配 LDS 空间，返回基地址 handle
    pub fn lds_alloc(&mut self, size_bytes: u32) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::U32), None);
        self.push_op(TileOp::LdsAlloc { result: v, size_bytes });
        v
    }

    /// LDS load: val = lds[base + offset * 4]
    pub fn lds_load(&mut self, base: Value, offset: Value) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::F32), None);
        self.push_op(TileOp::LdsLoad { result: v, base, offset });
        v
    }

    /// LDS store: lds[base + offset * 4] = val
    pub fn lds_store(&mut self, base: Value, offset: Value, val: Value) {
        self.push_op(TileOp::LdsStore { base, offset, val });
    }

    // ── Atomic ──

    /// global_atomic_add_f32: ptr[indices] += val (masked)
    pub fn atomic_add_f32(&mut self, ptr: Value, indices: Value, val: Value, mask: Option<Value>) {
        self.push_op(TileOp::AtomicAddF32 { ptr, indices, val, mask });
    }

    // ── WMMA / BF16 ──

    /// 8×f32 零初始化累加器
    pub fn zero_acc(&mut self) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::F32), None);
        self.push_op(TileOp::ZeroAcc { result: v });
        v
    }

    /// f32×2 → bf16x2 打包
    pub fn cvt_pk_bf16_f32(&mut self, lo: Value, hi: Value) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::U32), None);
        self.push_op(TileOp::CvtPkBf16F32 { result: v, lo, hi });
        v
    }

    /// WMMA: result = a × b + c
    pub fn wmma_f32(&mut self, a: Value, b: Value, c: Value) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::F32), None);
        self.push_op(TileOp::WmmaF32 { result: v, a, b, c });
        v
    }

    /// 从 8×f32 acc 提取第 idx 个元素
    pub fn extract_f32(&mut self, src: Value, idx: u32) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::F32), None);
        self.push_op(TileOp::ExtractF32 { result: v, src, idx });
        v
    }

    /// 广播到 8×VGPR fragment
    pub fn splat_fragment(&mut self, src: Value) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::U32), None);
        self.push_op(TileOp::SplatFragment { result: v, src });
        v
    }

    // ── WG 级归约 ──

    /// Workgroup-level sum reduction
    pub fn wg_reduce_add(&mut self, src: Value, block_size: u32) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::F32), None);
        self.push_op(TileOp::WgReduceAdd { result: v, src, block_size });
        v
    }

    /// Workgroup-level max reduction
    pub fn wg_reduce_max(&mut self, src: Value, block_size: u32) -> Value {
        let v = self.alloc_value(TileType::Scalar(ScalarDType::F32), None);
        self.push_op(TileOp::WgReduceMax { result: v, src, block_size });
        v
    }

    // ════════════════════════════════════════════════════════
    // Tile-Level 2D 操作（Triton 语义 API）
    // ════════════════════════════════════════════════════════

    /// 创建 [rows, cols] 的全零 f32 tile（GEMM 累加器初始值）
    /// 等价于 Triton: acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    /// Layout: MmaAccumulator (数据将存在 WMMA 寄存器中)
    pub fn tile_zeros(&mut self, rows: u32, cols: u32) -> Value {
        let zero = self.const_f32(0.0);
        // MMA accumulator: m_per_wave=rows (each wave handles all rows),
        // n_tiles = cols/16 (WMMA 16-col tiles)
        let layout = TensorLayout::mma(rows, cols / 16);
        let v = self.alloc_value(
            TileType::tile_with_layout(rows, cols, ScalarDType::F32, layout), None
        );
        self.push_op(TileOp::Splat { result: v, src: zero, shape: vec![rows, cols] });
        v
    }

    /// 2D Tile 加载：WG 协作加载 [rows, cols] tile
    /// 等价于 Triton: a = tl.load(a_ptr + offs_m[:, None] * stride + offs_k[None, :])
    ///
    /// # 参数
    /// - `ptr`: 全局内存基地址
    /// - `row_offset`: tile 起始行号（绝对）
    /// - `col_offset`: tile 起始列号（绝对）
    /// - `stride`: 行间距（元素数，不是字节数）
    /// - `rows`, `cols`: tile 形状（编译期常量）
    /// - `dtype`: 元素类型（BF16 for WMMA input）
    ///
    /// # 返回
    /// Opaque [rows, cols] tile handle，lowering 时解析为 LDS 区域
    pub fn tile_load_2d(
        &mut self, ptr: Value, row_offset: Value, col_offset: Value,
        stride: Value, rows: u32, cols: u32, dtype: ScalarDType,
    ) -> Value {
        assert_eq!(self.value_type(ptr), &TileType::Ptr, "tile_load_2d: ptr must be Ptr");
        // TileLoad2D 结果在 LDS 中（Shared layout）—— WG 协作加载到 LDS
        let layout = TensorLayout::shared(0);  // pad_bytes=0 default, lowering may override
        let v = self.alloc_value(
            TileType::tile_with_layout(rows, cols, dtype, layout), None
        );
        self.push_op(TileOp::TileLoad2D {
            result: v, ptr, row_offset, col_offset, stride, rows, cols, dtype,
        });
        v
    }

    /// Tile 矩阵乘累加：acc += dot(a, b)
    /// 等价于 Triton: acc = tl.dot(a, b, acc)
    ///
    /// # 参数
    /// - `a`: [M, K] tile（来自 tile_load_2d）
    /// - `b`: [K, N] tile（来自 tile_load_2d）
    /// - `acc`: [M, N] f32 accumulator
    ///
    /// # 返回
    /// 更新后的 [M, N] f32 accumulator
    pub fn tile_dot(&mut self, a: Value, b: Value, acc: Value) -> Value {
        let aty = self.value_type(a).clone();
        let bty = self.value_type(b).clone();
        let acc_ty = self.value_type(acc).clone();
        // 类型检查
        let (m, k1) = match &aty {
            TileType::Tile { rows, cols, .. } => (*rows, *cols),
            _ => panic!("tile_dot: 'a' must be 2D tile, got {}", aty),
        };
        let (k2, n) = match &bty {
            TileType::Tile { rows, cols, .. } => (*rows, *cols),
            _ => panic!("tile_dot: 'b' must be 2D tile, got {}", bty),
        };
        assert_eq!(k1, k2, "tile_dot: K mismatch {} vs {}", k1, k2);
        // acc 必须是 [M, N] f32
        match &acc_ty {
            TileType::Tile { rows, cols, dtype, .. } => {
                assert_eq!(*rows, m, "tile_dot: acc rows {} != M {}", rows, m);
                assert_eq!(*cols, n, "tile_dot: acc cols {} != N {}", cols, n);
                assert_eq!(*dtype, ScalarDType::F32, "tile_dot: acc must be f32");
            }
            _ => panic!("tile_dot: acc must be [M,N] f32 tile, got {}", acc_ty),
        }
        // tile_dot 结果是 MMA 累加器 layout
        let layout = TensorLayout::mma(m, n / 16);
        let v = self.alloc_value(
            TileType::tile_with_layout(m, n, ScalarDType::F32, layout), None
        );
        self.push_op(TileOp::TileDot { result: v, a, b, acc });
        v
    }

    /// 2D Tile 存储：将 [M, N] f32 tile 写入全局内存
    /// 等价于 Triton: tl.store(c_ptr + offs_m[:, None] * stride + offs_n[None, :], acc)
    pub fn tile_store_2d(
        &mut self, ptr: Value, row_offset: Value, col_offset: Value,
        stride: Value, val: Value,
    ) {
        assert_eq!(self.value_type(ptr), &TileType::Ptr, "tile_store_2d: ptr must be Ptr");
        assert!(
            matches!(self.value_type(val), TileType::Tile { .. }),
            "tile_store_2d: val must be 2D tile, got {}", self.value_type(val)
        );
        self.push_op(TileOp::TileStore2D { ptr, row_offset, col_offset, stride, val });
    }

    // ── 控制流 ──

    /// 创建新的基本块，返回其 ID
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock {
            id,
            ops: Vec::new(),
            terminator: None,
            params: Vec::new(),
        });
        id
    }

    /// 切换到指定基本块
    pub fn switch_to_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    /// 无条件跳转到目标块
    pub fn branch(&mut self, target: BlockId, args: Vec<Value>) {
        self.blocks[self.current_block.0 as usize].terminator =
            Some(Terminator::Branch { target, args });
    }

    /// 条件分支
    pub fn cond_branch(&mut self, cond: Value,
                       true_bb: BlockId, true_args: Vec<Value>,
                       false_bb: BlockId, false_args: Vec<Value>) {
        self.blocks[self.current_block.0 as usize].terminator =
            Some(Terminator::CondBranch { cond, true_bb, true_args, false_bb, false_args });
    }

    /// 添加块参数（类似 phi 节点）
    pub fn add_block_param(&mut self, block: BlockId, ty: TileType) -> Value {
        let v = self.alloc_value(ty, None);
        self.blocks[block.0 as usize].params.push(v);
        v
    }

    /// 返回
    pub fn return_(&mut self) {
        self.blocks[self.current_block.0 as usize].terminator =
            Some(Terminator::Return);
    }

    // ── 高层控制流 helper ──

    /// 创建一个 for range 循环: for i in [start, end) { body }
    ///
    /// 返回 ForLoop 结构，包含循环归纳变量和 body block ID。
    /// 使用模式：
    /// ```ignore
    /// let lp = f.for_range(0, n);
    /// // 在 body 中使用 lp.iv 作为循环变量
    /// let val = f.load(ptr, lp.iv, ScalarDType::F32);
    /// // ... 更多 body 操作 ...
    /// f.end_for(&lp);          // 关闭循环
    /// // 现在在 exit block 中
    /// ```
    ///
    /// 生成的 CFG:
    /// ```text
    /// entry:
    ///   branch → header(start)
    ///
    /// header(iv):            ; iv = block param
    ///   cond = iv < end
    ///   cond_br cond → body, exit
    ///
    /// body:                  ; body ops go here
    ///   iv_next = iv + 1
    ///   branch → header(iv_next)
    ///
    /// exit:
    ///   ... continue ...
    /// ```
    pub fn for_range(&mut self, start: u32, end: u32) -> ForLoop {
        let header = self.new_block();
        let body = self.new_block();
        let exit = self.new_block();

        // Entry → header(start)
        let start_val = self.const_u32(start);
        self.branch(header, vec![start_val]);

        // Header: iv = block param
        self.switch_to_block(header);
        let iv = self.add_block_param(header, TileType::Scalar(ScalarDType::U32));

        // Compare: iv < end
        let end_val = self.const_u32(end);
        let cond = self.cmp_lt(iv, end_val);
        self.cond_branch(cond, body, vec![], exit, vec![]);

        // Switch to body for user to add ops
        self.switch_to_block(body);

        ForLoop { header, body, exit, iv, step: 1 }
    }

    /// 结束 for 循环：发射 iv += step 和 branch 回 header
    pub fn end_for(&mut self, lp: &ForLoop) {
        let step_val = self.const_u32(lp.step);
        let iv_next = self.add(lp.iv, step_val);
        self.branch(lp.header, vec![iv_next]);

        // 切换到 exit block
        self.switch_to_block(lp.exit);
    }

    /// 创建 runtime-bounds for 循环: for i in [start, end) step step { body }
    ///
    /// 与 `for_range` 相同的 CFG 结构，但 start/end 是 SSA Value（运行时值）。
    /// start/end 必须是标量 U32 类型。
    ///
    /// 生成的 CFG:
    /// ```text
    /// current_block:
    ///   branch → header(start)
    ///
    /// header(iv):           ; iv = block param
    ///   cond = iv < end
    ///   cond_br cond → body, exit
    ///
    /// body:                 ; body ops go here
    ///   iv_next = iv + step
    ///   branch → header(iv_next)
    ///
    /// exit:
    ///   ... continue ...
    /// ```
    pub fn for_range_runtime(&mut self, start: Value, end: Value, step: u32) -> ForLoop {
        let header = self.new_block();
        let body = self.new_block();
        let exit = self.new_block();

        // Current block → header(start)
        self.branch(header, vec![start]);

        // Header: iv = block param
        self.switch_to_block(header);
        let iv = self.add_block_param(header, TileType::Scalar(ScalarDType::U32));

        // Compare: iv < end (runtime value)
        let cond = self.cmp_lt(iv, end);
        self.cond_branch(cond, body, vec![], exit, vec![]);

        // Switch to body for caller to add ops
        self.switch_to_block(body);

        ForLoop { header, body, exit, iv, step }
    }


    /// 创建带累加器的 for range: for i in [start, end) { acc = body(acc, i) }
    ///
    /// 返回 ForLoopAcc，包含循环变量 iv 和累加器 acc。
    /// 累加器的初始值由 init_val 提供，类型由 acc_ty 指定。
    ///
    /// ```ignore
    /// let zero = f.const_f32(0.0);
    /// let lp = f.for_range_with_acc(0, n, zero, TileType::Scalar(ScalarDType::F32));
    /// // lp.iv = 当前循环索引
    /// // lp.acc = 当前累加值
    /// let val = f.load(ptr, lp.iv, ScalarDType::F32);
    /// let new_acc = f.add(lp.acc, val);
    /// f.end_for_acc(&lp, new_acc);
    /// // 现在在 exit block，lp.result 包含最终累加值
    /// ```
    pub fn for_range_with_acc(&mut self, start: u32, end: u32,
                              init_val: Value, acc_ty: TileType) -> ForLoopAcc {
        let header = self.new_block();
        let body = self.new_block();
        let exit = self.new_block();

        // Entry → header(start, init_val)
        let start_val = self.const_u32(start);
        self.branch(header, vec![start_val, init_val]);

        // Header: iv + acc as block params
        self.switch_to_block(header);
        let iv = self.add_block_param(header, TileType::Scalar(ScalarDType::U32));
        let acc = self.add_block_param(header, acc_ty.clone());

        // Compare: iv < end
        let end_val = self.const_u32(end);
        let cond = self.cmp_lt(iv, end_val);

        // exit gets the final accumulator value
        let result = self.add_block_param(exit, acc_ty);
        self.cond_branch(cond, body, vec![], exit, vec![acc]);

        self.switch_to_block(body);

        ForLoopAcc { header, body, exit, iv, acc, result, step: 1 }
    }

    /// 结束带累加器的 for 循环
    pub fn end_for_acc(&mut self, lp: &ForLoopAcc, new_acc: Value) {
        let step_val = self.const_u32(lp.step);
        let iv_next = self.add(lp.iv, step_val);
        self.branch(lp.header, vec![iv_next, new_acc]);

        self.switch_to_block(lp.exit);
    }

    /// 创建运行时 bounds 的带累加器 for 循环:
    /// for i in [start, end) step step { acc = body(acc, i) }
    ///
    /// 与 `for_range_with_acc` 相同的 CFG，但 start/end 是 SSA Value。
    ///
    /// ```ignore
    /// let zero = f.const_f32(0.0);
    /// let lp = f.for_range_with_acc_runtime(start, end, 1, zero,
    ///              TileType::Scalar(ScalarDType::F32));
    /// let val = f.load(ptr, lp.iv, ScalarDType::F32);
    /// let new_acc = f.add(lp.acc, val);
    /// f.end_for_acc(&lp, new_acc);
    /// // lp.result = final sum
    /// ```
    ///
    /// CFG:
    /// ```text
    /// current_block:
    ///   branch → header(start, init_val)
    ///
    /// header(iv, acc):
    ///   cond = iv < end
    ///   cond_br cond → body, exit(acc)
    ///
    /// body:
    ///   ... user ops ...
    ///   iv_next = iv + step
    ///   branch → header(iv_next, new_acc)
    ///
    /// exit(result):
    ///   ... continue with result ...
    /// ```
    pub fn for_range_with_acc_runtime(
        &mut self, start: Value, end: Value, step: u32,
        init_val: Value, acc_ty: TileType,
    ) -> ForLoopAcc {
        let header = self.new_block();
        let body = self.new_block();
        let exit = self.new_block();

        // Current block → header(start, init_val)
        self.branch(header, vec![start, init_val]);

        // Header: iv + acc as block params
        self.switch_to_block(header);
        let iv = self.add_block_param(header, TileType::Scalar(ScalarDType::U32));
        let acc = self.add_block_param(header, acc_ty.clone());

        // Compare: iv < end (runtime value)
        let cond = self.cmp_lt(iv, end);

        // exit gets the final accumulator value
        let result = self.add_block_param(exit, acc_ty);
        self.cond_branch(cond, body, vec![], exit, vec![acc]);

        // Switch to body for caller to add ops
        self.switch_to_block(body);

        ForLoopAcc { header, body, exit, iv, acc, result, step }
    }

    // ════════════════════════════════════════════════════════
    // 打印 / Debug
    // ════════════════════════════════════════════════════════

    /// 打印可读的 SSA IR 文本
    pub fn dump(&self) -> String {
        let mut s = format!("func @{}(", self.name);
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 { s.push_str(", "); }
            let def = &self.values[arg.0 as usize];
            s.push_str(&format!("{}: {}", arg, def.ty));
            if let Some(name) = &def.name {
                s.push_str(&format!(" /*{}*/", name));
            }
        }
        s.push_str(") {\n");

        for block in &self.blocks {
            s.push_str(&format!("  bb{}(", block.id.0));
            for (i, p) in block.params.iter().enumerate() {
                if i > 0 { s.push_str(", "); }
                s.push_str(&format!("{}: {}", p, self.values[p.0 as usize].ty));
            }
            s.push_str("):\n");

            for &op_idx in &block.ops {
                s.push_str(&format!("    {}\n", self.format_op(&self.ops[op_idx])));
            }

            if let Some(term) = &block.terminator {
                s.push_str(&format!("    {}\n", self.format_terminator(term)));
            }
        }
        s.push_str("}\n");
        s
    }

    fn format_op(&self, op: &TileOp) -> String {
        match op {
            TileOp::ConstU32 { result, value } =>
                format!("{}: {} = const {}", result, self.values[result.0 as usize].ty, value),
            TileOp::ConstF32 { result, value } =>
                format!("{}: {} = const {:.6}", result, self.values[result.0 as usize].ty, value),
            TileOp::ProgramId { result, axis } =>
                format!("{}: U32 = program_id({})", result, axis),
            TileOp::Arange { result, start, len } =>
                format!("{}: [{}]U32 = arange({}, {})", result, len, start, start + len),
            TileOp::ThreadIdX2D { result, block_x } =>
                format!("{}: [{}]U32 = thread_id_x_2d(bx={})", result, block_x, block_x),
            TileOp::ThreadIdY2D { result, block_x } =>
                format!("{}: U32 = thread_id_y_2d(bx={})", result, block_x),
            TileOp::Splat { result, src, shape } =>
                format!("{}: {} = splat({}, {:?})", result, self.values[result.0 as usize].ty, src, shape),
            TileOp::Load { result, ptr, indices, mask, .. } => {
                let m = if mask.is_some() { ", masked" } else { "" };
                format!("{}: {} = load({}, {}{})", result, self.values[result.0 as usize].ty, ptr, indices, m)
            }
            TileOp::Store { ptr, indices, val, .. } =>
                format!("store({}, {}, {})", ptr, indices, val),
            TileOp::BinOp { result, op, lhs, rhs } =>
                format!("{}: {} = {:?}({}, {})", result, self.values[result.0 as usize].ty, op, lhs, rhs),
            TileOp::UnaryOp { result, op, src } =>
                format!("{}: {} = {:?}({})", result, self.values[result.0 as usize].ty, op, src),
            TileOp::Dot { result, a, b } =>
                format!("{}: {} = dot({}, {})", result, self.values[result.0 as usize].ty, a, b),
            TileOp::Reduce { result, src, axis, op } =>
                format!("{}: {} = {:?}({}, axis={})", result, self.values[result.0 as usize].ty, op, src, axis),
            TileOp::Cast { result, src, to } =>
                format!("{}: {} = cast({}, {:?})", result, self.values[result.0 as usize].ty, src, to),
            TileOp::Barrier => "barrier".into(),
            TileOp::LdsAlloc { result, size_bytes } =>
                format!("{}: U32 = lds_alloc({})", result, size_bytes),
            TileOp::LdsLoad { result, base, offset } =>
                format!("{}: F32 = lds_load({}, {})", result, base, offset),
            TileOp::LdsStore { base, offset, val } =>
                format!("lds_store({}, {}, {})", base, offset, val),
            TileOp::AtomicAddF32 { ptr, indices, val, mask } => {
                let m = if mask.is_some() { ", masked" } else { "" };
                format!("atomic_add_f32({}, {}, {}{})", ptr, indices, val, m)
            }
            TileOp::ZeroAcc { result } =>
                format!("{}: F32x8 = zero_acc()", result),
            TileOp::CvtPkBf16F32 { result, lo, hi } =>
                format!("{}: U32 = cvt_pk_bf16_f32({}, {})", result, lo, hi),
            TileOp::WmmaF32 { result, a, b, c } =>
                format!("{}: F32x8 = wmma({}, {}, {})", result, a, b, c),
            TileOp::ExtractF32 { result, src, idx } =>
                format!("{}: F32 = extract({}, {})", result, src, idx),
            TileOp::SplatFragment { result, src } =>
                format!("{}: U32x8 = splat_fragment({})", result, src),
            TileOp::WgReduceAdd { result, src, block_size } =>
                format!("{}: F32 = wg_reduce_add({}, bs={})", result, src, block_size),
            TileOp::WgReduceMax { result, src, block_size } =>
                format!("{}: F32 = wg_reduce_max({}, bs={})", result, src, block_size),
            TileOp::Cmp { result, op, lhs, rhs } =>
                format!("{}: Bool = {:?}({}, {})", result, op, lhs, rhs),
            TileOp::Select { result, cond, true_val, false_val } =>
                format!("{}: {} = select({}, {}, {})", result, self.values[result.0 as usize].ty, cond, true_val, false_val),
            TileOp::Fma { result, a, b, c } =>
                format!("{}: {} = fma({}, {}, {})", result, self.values[result.0 as usize].ty, a, b, c),
            TileOp::Reshape { result, src, shape } =>
                format!("{}: {} = reshape({}, {:?})", result, self.values[result.0 as usize].ty, src, shape),
            TileOp::ExpandDims { result, src, axis } =>
                format!("{}: {} = expand_dims({}, {})", result, self.values[result.0 as usize].ty, src, axis),
            // ── Tile-Level 2D ops ──
            TileOp::TileLoad2D { result, ptr, row_offset, col_offset, stride, rows, cols, dtype } =>
                format!("{}: {} = tile_load_2d({}, row={}, col={}, stride={}, {}x{} {:?})",
                    result, self.values[result.0 as usize].ty, ptr, row_offset, col_offset, stride, rows, cols, dtype),
            TileOp::TileDot { result, a, b, acc } =>
                format!("{}: {} = tile_dot({}, {}, {})",
                    result, self.values[result.0 as usize].ty, a, b, acc),
            TileOp::TileStore2D { ptr, row_offset, col_offset, stride, val } =>
                format!("tile_store_2d({}, row={}, col={}, stride={}, {})",
                    ptr, row_offset, col_offset, stride, val),
            // ── EXEC Mask ──
            TileOp::ExecMaskPush { mask } =>
                format!("exec_mask_push({})", mask),
            TileOp::ExecMaskFlip =>
                "exec_mask_flip".into(),
            TileOp::ExecMaskPop =>
                "exec_mask_pop".into(),
        }
    }

    fn format_terminator(&self, t: &Terminator) -> String {
        match t {
            Terminator::Branch { target, args } =>
                format!("br bb{}({})", target.0, args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(", ")),
            Terminator::CondBranch { cond, true_bb, true_args, false_bb, false_args } =>
                format!("cond_br {}, bb{}({}), bb{}({})", cond,
                    true_bb.0, true_args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(", "),
                    false_bb.0, false_args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(", ")),
            Terminator::Return => "return".into(),
        }
    }
}

// ============================================================================
// Shape 推导
// ============================================================================

/// 推导二元操作的输出类型（自动广播）
fn infer_binop_type(lty: &TileType, rty: &TileType, _op: BinOpKind) -> TileType {
    match (lty, rty) {
        // 相同类型 → 相同输出
        (a, b) if a == b => a.clone(),

        // 标量 + tile → tile（广播）
        (TileType::Scalar(d), TileType::Vector { len, dtype, .. }) |
        (TileType::Vector { len, dtype, .. }, TileType::Scalar(d)) => {
            assert_eq!(d, dtype, "binop: dtype mismatch {:?} vs {:?}", d, dtype);
            TileType::vector(*len, *dtype)
        }
        (TileType::Scalar(d), TileType::Tile { rows, cols, dtype, .. }) |
        (TileType::Tile { rows, cols, dtype, .. }, TileType::Scalar(d)) => {
            assert_eq!(d, dtype, "binop: dtype mismatch {:?} vs {:?}", d, dtype);
            TileType::tile(*rows, *cols, *dtype)
        }

        _ => panic!("binop: incompatible types {} and {}", lty, rty),
    }
}

/// 推导归约操作的输出类型
fn infer_reduce_type(src_ty: &TileType, axis: u32) -> TileType {
    match src_ty {
        TileType::Vector { dtype, .. } => {
            assert_eq!(axis, 0, "reduce: vector only supports axis=0");
            TileType::Scalar(*dtype)
        }
        TileType::Tile { rows, cols, dtype, .. } => {
            match axis {
                0 => TileType::vector(*cols, *dtype),  // 沿行归约 → 列向量
                1 => TileType::vector(*rows, *dtype),  // 沿列归约 → 行向量
                _ => panic!("reduce: axis {} out of range for 2D tile", axis),
            }
        }
        _ => panic!("reduce: source must be vector or tile, got {}", src_ty),
    }
}

/// 推导 dot 操作的输出类型
fn infer_dot_type(aty: &TileType, bty: &TileType) -> TileType {
    match (aty, bty) {
        (TileType::Tile { rows: m, cols: k1, dtype: da, .. },
         TileType::Tile { rows: k2, cols: n, dtype: db, .. }) => {
            assert_eq!(k1, k2, "dot: K dimension mismatch {} vs {}", k1, k2);
            // WMMA: bf16 × bf16 → f32 accumulator
            let out_dtype = match (*da, *db) {
                (ScalarDType::BF16, ScalarDType::BF16) => ScalarDType::F32,
                (ScalarDType::F16, ScalarDType::F16) => ScalarDType::F32,
                (ScalarDType::F32, ScalarDType::F32) => ScalarDType::F32,
                _ => panic!("dot: unsupported dtype pair {:?} × {:?}", da, db),
            };
            TileType::tile(*m, *n, out_dtype)
        }
        _ => panic!("dot: both operands must be 2D tiles, got {} and {}", aty, bty),
    }
}

/// 获取 TileOp 的 result Value（如果有）
fn op_result(op: &TileOp) -> Option<Value> {
    match op {
        TileOp::ConstU32 { result, .. } | TileOp::ConstF32 { result, .. } |
        TileOp::ProgramId { result, .. } | TileOp::Arange { result, .. } |
        TileOp::ThreadIdX2D { result, .. } | TileOp::ThreadIdY2D { result, .. } |
        TileOp::Splat { result, .. } | TileOp::Reshape { result, .. } |
        TileOp::ExpandDims { result, .. } |
        TileOp::Load { result, .. } |
        TileOp::BinOp { result, .. } | TileOp::UnaryOp { result, .. } |
        TileOp::Fma { result, .. } | TileOp::Cast { result, .. } |
        TileOp::Cmp { result, .. } | TileOp::Select { result, .. } |
        TileOp::Reduce { result, .. } | TileOp::Dot { result, .. } => Some(*result),
        TileOp::Store { .. } | TileOp::Barrier |
        TileOp::LdsStore { .. } | TileOp::AtomicAddF32 { .. } |
        TileOp::TileStore2D { .. } |
        TileOp::ExecMaskPush { .. } | TileOp::ExecMaskFlip | TileOp::ExecMaskPop => None,
        TileOp::LdsAlloc { result, .. } | TileOp::LdsLoad { result, .. } |
        TileOp::ZeroAcc { result, .. } | TileOp::CvtPkBf16F32 { result, .. } |
        TileOp::WmmaF32 { result, .. } | TileOp::ExtractF32 { result, .. } |
        TileOp::SplatFragment { result, .. } |
        TileOp::WgReduceAdd { result, .. } | TileOp::WgReduceMax { result, .. } |
        TileOp::TileLoad2D { result, .. } | TileOp::TileDot { result, .. } => Some(*result),
    }
}

// ============================================================================
// TileCfg — CfgProvider for TileFunc (Dominator Tree integration)
// ============================================================================

use super::domtree::{CfgProvider, DomTree};

/// Cached CFG view of a TileFunc — computes preds/succs from Terminators.
///
/// Created via `TileFunc::build_cfg()` or `TileFunc::build_domtree()`.
pub struct TileCfg {
    preds: Vec<Vec<u32>>,
    succs: Vec<Vec<u32>>,
}

impl TileCfg {
    /// Build CFG from TileFunc's blocks and terminators.
    pub fn from_tile_func(func: &TileFunc) -> Self {
        let n = func.blocks.len();
        let mut preds = vec![vec![]; n];
        let mut succs = vec![vec![]; n];

        for (i, block) in func.blocks.iter().enumerate() {
            let src = i as u32;
            match &block.terminator {
                Some(Terminator::Branch { target, .. }) => {
                    let dst = target.0;
                    succs[src as usize].push(dst);
                    preds[dst as usize].push(src);
                }
                Some(Terminator::CondBranch { true_bb, false_bb, .. }) => {
                    let t = true_bb.0;
                    let f = false_bb.0;
                    succs[src as usize].push(t);
                    preds[t as usize].push(src);
                    if f != t {
                        succs[src as usize].push(f);
                        preds[f as usize].push(src);
                    }
                }
                Some(Terminator::Return) | None => {}
            }
        }

        TileCfg { preds, succs }
    }
}

impl CfgProvider for TileCfg {
    fn num_blocks(&self) -> usize { self.succs.len() }
    fn entry(&self) -> u32 { 0 }
    fn preds(&self, block: u32) -> &[u32] { &self.preds[block as usize] }
    fn succs(&self, block: u32) -> &[u32] { &self.succs[block as usize] }
}

impl TileFunc {
    /// Build a CFG view (preds/succs computed from terminators).
    pub fn build_cfg(&self) -> TileCfg {
        TileCfg::from_tile_func(self)
    }

    /// Build Dominator Tree from the current TileFunc CFG.
    pub fn build_domtree(&self) -> DomTree {
        let cfg = self.build_cfg();
        DomTree::build(&cfg)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add_kernel() {
        // 等价于 Triton 的:
        //   @triton.jit
        //   def add_kernel(x_ptr, y_ptr, out_ptr, n):
        //       pid = tl.program_id(0)
        //       offs = tl.arange(0, 128)
        //       idx = pid * 128 + offs
        //       x = tl.load(x_ptr + idx)
        //       y = tl.load(y_ptr + idx)
        //       out = x + y
        //       tl.store(out_ptr + idx, out)
        let mut f = TileFunc::new("vector_add");
        let x_ptr = f.arg_ptr("x_ptr");
        let y_ptr = f.arg_ptr("y_ptr");
        let out_ptr = f.arg_ptr("out_ptr");
        let _n = f.arg_u32("n");

        let pid = f.program_id(0);
        let block_size = f.const_u32(128);
        let base = f.mul(pid, block_size);
        let offs = f.arange(0, 128);
        let base_v = f.splat(base, 128);
        let idx = f.add(base_v, offs);

        let x = f.load(x_ptr, idx, ScalarDType::F32);
        let y = f.load(y_ptr, idx, ScalarDType::F32);
        let out = f.add(x, y);
        f.store(out_ptr, idx, out);
        f.return_();

        let ir_text = f.dump();
        eprintln!("{}", ir_text);

        // 验证 SSA 性质: 4 args + 9 value-producing ops = 13 values, 10 total ops (9 + store)
        assert_eq!(f.all_values().len(), 13);
        assert_eq!(f.all_ops().len(), 10);

        // 验证 shape 推导
        assert_eq!(f.value_type(pid), &TileType::scalar_u32());
        assert_eq!(f.value_type(offs), &TileType::vector(128, ScalarDType::U32));
        assert_eq!(f.value_type(idx), &TileType::vector(128, ScalarDType::U32));
        assert_eq!(f.value_type(x), &TileType::vector(128, ScalarDType::F32));
        assert_eq!(f.value_type(out), &TileType::vector(128, ScalarDType::F32));
    }

    #[test]
    fn test_softmax_kernel() {
        // Triton softmax 模式
        let mut f = TileFunc::new("softmax");
        let x_ptr = f.arg_ptr("x_ptr");
        let out_ptr = f.arg_ptr("out_ptr");
        let n_cols = f.arg_u32("n_cols");

        let pid = f.program_id(0);
        let offs = f.arange(0, 128);

        // row_start = pid * n_cols
        let row_start = f.mul(pid, n_cols);
        let row_start_v = f.splat(row_start, 128);
        let idx = f.add(row_start_v, offs);

        // x = load
        let x = f.load(x_ptr, idx, ScalarDType::F32);

        // row_max = reduce_max(x, axis=0)
        let row_max = f.reduce_max(x, 0);

        // x_shifted = x - broadcast(row_max)
        let row_max_v = f.splat(row_max, 128);
        let x_shifted = f.sub(x, row_max_v);

        // exp_x = exp(x_shifted)
        let exp_x = f.exp(x_shifted);

        // row_sum = sum(exp_x, axis=0)
        let row_sum = f.sum(exp_x, 0);

        // result = exp_x / broadcast(row_sum)
        let row_sum_v = f.splat(row_sum, 128);
        let result = f.div(exp_x, row_sum_v);

        // store
        f.store(out_ptr, idx, result);
        f.return_();

        let ir_text = f.dump();
        eprintln!("{}", ir_text);

        // 验证 reduce 的 shape 推导
        assert_eq!(f.value_type(row_max), &TileType::scalar_f32());
        assert_eq!(f.value_type(row_sum), &TileType::scalar_f32());
        assert_eq!(f.value_type(result), &TileType::vector(128, ScalarDType::F32));
    }

    #[test]
    fn test_dot_shape_inference() {
        let mut f = TileFunc::new("matmul_tile");
        let a_ptr = f.arg_ptr("A");
        let _b_ptr = f.arg_ptr("B");

        // 假设已经 load 了 tile
        let _idx_m = f.arange(0, 16);
        let _idx_k = f.arange(0, 64);

        // 模拟 2D tile（实际使用时会有 expand_dims + broadcast）
        let _a = f.load(a_ptr, _idx_m, ScalarDType::BF16);  // [16] bf16 (简化)

        // 验证 dot 需要 2D tile
        // 这里我们手动构建 2D 类型来测试
        let a_tile = f.alloc_value(TileType::tile(16, 64, ScalarDType::BF16), Some("A_tile"));
        let b_tile = f.alloc_value(TileType::tile(64, 32, ScalarDType::BF16), Some("B_tile"));
        let c = f.dot(a_tile, b_tile);

        // bf16 × bf16 → f32
        assert_eq!(f.value_type(c), &TileType::tile(16, 32, ScalarDType::F32));
    }

    #[test]
    fn test_ir_dump_readable() {
        let mut f = TileFunc::new("simple");
        let x = f.arg_ptr("x");
        let _n = f.arg_u32("n");
        let pid = f.program_id(0);
        let offs = f.arange(0, 64);
        let c64 = f.const_u32(64);
        let base = f.mul(pid, c64);
        let base_v = f.splat(base, 64);
        let idx = f.add(base_v, offs);
        let data = f.load(x, idx, ScalarDType::F32);
        let doubled = f.add(data, data);
        f.store(x, idx, doubled);
        f.return_();

        let text = f.dump();
        assert!(text.contains("func @simple"));
        assert!(text.contains("program_id(0)"));
        assert!(text.contains("arange(0, 64)"));
        assert!(text.contains("load"));
        assert!(text.contains("store"));
        assert!(text.contains("return"));
        eprintln!("{}", text);
    }

    #[test]
    fn test_tile_level_gemm_builder() {
        // 等价于 Triton GEMM 教程：
        //   acc = tl.zeros((128, 64), dtype=tl.float32)
        //   for k in range(0, K, 16):
        //       a = tl.load(a_ptr + ...)   # [128, 16] bf16
        //       b = tl.load(b_ptr + ...)   # [16, 64] bf16
        //       acc = tl.dot(a, b, acc)
        //   tl.store(c_ptr + ..., acc)
        let mut f = TileFunc::new("tiled_gemm");
        let x_ptr = f.arg_ptr("X");   // [M, K] bf16
        let w_ptr = f.arg_ptr("W");   // [N, K] bf16 (NT layout)
        let y_ptr = f.arg_ptr("Y");   // [M, N] f32
        let k_dim = f.arg_u32("K");
        let n_dim = f.arg_u32("N");

        // tile 坐标
        let pid_m = f.program_id(0);
        let pid_n = f.program_id(1);
        let c128 = f.const_u32(128);
        let c64 = f.const_u32(64);
        let row_off = f.mul(pid_m, c128);  // tile 起始行
        let col_off = f.mul(pid_n, c64);   // tile 起始列

        // 累加器初始化: [128, 64] f32 全零
        let acc = f.tile_zeros(128, 64);
        // tile_zeros returns MmaAccumulator layout
        assert!(f.value_type(acc).layout().is_mma(),
            "tile_zeros should be MmaAccumulator, got {}", f.value_type(acc));

        // K-loop: 模拟加载 + dot
        let k_start = f.const_u32(0);
        let a = f.tile_load_2d(x_ptr, row_off, k_start, k_dim, 128, 16, ScalarDType::BF16);
        // tile_load_2d returns Shared layout (data in LDS)
        assert!(f.value_type(a).layout().is_shared(),
            "tile_load_2d should be Shared, got {}", f.value_type(a));

        let b = f.tile_load_2d(w_ptr, col_off, k_start, k_dim, 16, 64, ScalarDType::BF16);
        assert!(f.value_type(b).layout().is_shared(),
            "tile_load_2d should be Shared, got {}", f.value_type(b));

        let acc2 = f.tile_dot(a, b, acc);
        // tile_dot returns MmaAccumulator layout
        assert!(f.value_type(acc2).layout().is_mma(),
            "tile_dot should be MmaAccumulator, got {}", f.value_type(acc2));
        assert_eq!(f.value_type(acc2).numel(), 128 * 64);

        // Store
        f.tile_store_2d(y_ptr, row_off, col_off, n_dim, acc2);

        f.return_();

        // Dump and verify
        let ir_text = f.dump();
        eprintln!("{}", ir_text);

        assert!(ir_text.contains("tile_load_2d"));
        assert!(ir_text.contains("tile_dot"));
        assert!(ir_text.contains("tile_store_2d"));
        assert!(ir_text.contains("128x16"));
        assert!(ir_text.contains("16x64"));
    }

    // ═══════════════════════════════════════════════════════════
    // TensorLayout 类型系统测试
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_layout_default_blocked() {
        // Vector 默认 Blocked{1}
        let v = TileType::vector(128, ScalarDType::F32);
        assert_eq!(v.layout(), &TensorLayout::Blocked { elems_per_thread: 1 });
        assert!(v.layout().is_blocked());
        assert!(!v.layout().is_shared());
        assert!(!v.layout().is_mma());
    }

    #[test]
    fn test_layout_default_blocked2d() {
        // Tile 默认 Blocked2D
        let t = TileType::tile(16, 64, ScalarDType::BF16);
        match t.layout() {
            TensorLayout::Blocked2D { rows_per_thread, cols_per_thread, block_x } => {
                assert_eq!(*rows_per_thread, 1);
                assert_eq!(*cols_per_thread, 1);
                assert_eq!(*block_x, 32);
            }
            _ => panic!("tile default layout should be Blocked2D, got {:?}", t.layout()),
        }
    }

    #[test]
    fn test_layout_with_shared() {
        let t = TileType::tile_with_layout(16, 64, ScalarDType::BF16,
            TensorLayout::shared(4));
        assert!(t.layout().is_shared());
        match t.layout() {
            TensorLayout::Shared { pad_bytes, swizzle } => {
                assert_eq!(*pad_bytes, 4);
                assert!(!swizzle);
            }
            _ => panic!("expected Shared layout"),
        }
    }

    #[test]
    fn test_layout_with_mma() {
        let t = TileType::tile_with_layout(128, 64, ScalarDType::F32,
            TensorLayout::mma(16, 4));
        assert!(t.layout().is_mma());
        match t.layout() {
            TensorLayout::MmaAccumulator { m_per_wave, n_tiles } => {
                assert_eq!(*m_per_wave, 16);
                assert_eq!(*n_tiles, 4);
            }
            _ => panic!("expected MmaAccumulator layout"),
        }
    }

    #[test]
    fn test_layout_scalar_is_scalar() {
        let s = TileType::scalar_f32();
        assert_eq!(s.layout(), &TensorLayout::Scalar);
        let p = TileType::Ptr;
        assert_eq!(p.layout(), &TensorLayout::Scalar);
    }

    #[test]
    fn test_with_layout_preserves_shape() {
        let original = TileType::tile(32, 64, ScalarDType::F32);
        let shared = original.with_layout(TensorLayout::shared(8));
        // Shape and dtype preserved, layout changed
        assert_eq!(shared.numel(), 32 * 64);
        assert_eq!(shared.dtype(), Some(ScalarDType::F32));
        assert!(shared.layout().is_shared());
    }

    #[test]
    fn test_layout_display_format() {
        let v = TileType::vector(128, ScalarDType::F32);
        let display = format!("{}", v);
        assert!(display.contains("#blocked{1}"), "got: {}", display);

        let t = TileType::tile_with_layout(16, 64, ScalarDType::BF16,
            TensorLayout::shared(4));
        let display = format!("{}", t);
        assert!(display.contains("#shared{pad=4"), "got: {}", display);

        let m = TileType::tile_with_layout(128, 64, ScalarDType::F32,
            TensorLayout::mma(16, 4));
        let display = format!("{}", m);
        assert!(display.contains("#mma{m=16"), "got: {}", display);
    }

    #[test]
    fn test_layout_equality() {
        // Same shape + same layout → equal
        let a = TileType::vector(128, ScalarDType::F32);
        let b = TileType::vector(128, ScalarDType::F32);
        assert_eq!(a, b);

        // Same shape + different layout → not equal
        let c = TileType::vector_with_layout(128, ScalarDType::F32,
            TensorLayout::Blocked { elems_per_thread: 4 });
        assert_ne!(a, c);
    }

    #[test]
    fn test_layout_inference_in_builder() {
        // Builder API should auto-set default layouts
        let mut f = TileFunc::new("layout_test");
        let x_ptr = f.arg_ptr("X");
        let idx = f.arange(0, 128);
        // arange → Vector with default Blocked{1}
        assert_eq!(f.value_type(idx).layout(), &TensorLayout::blocked_default());
        // load → inherits layout from indices
        let x = f.load(x_ptr, idx, ScalarDType::F32);
        assert!(f.value_type(x).layout().is_blocked());
    }

    #[test]
    fn test_for_range_with_acc_runtime() {
        let mut f = TileFunc::new("acc_test");
        let ptr = f.arg_ptr("data");
        let n = f.arg_u32("n");
        let zero = f.const_f32(0.0);
        let start = f.const_u32(0);

        // Create loop: sum = 0; for i in [0, n) { sum += load(data[i]) }
        let lp = f.for_range_with_acc_runtime(start, n, 1, zero,
            TileType::Scalar(ScalarDType::F32));

        // Verify lp fields
        assert_ne!(lp.iv, lp.acc, "iv and acc should be different Values");
        assert_ne!(lp.acc, lp.result, "acc (body) and result (exit) should differ");

        // Verify types
        assert_eq!(f.value_type(lp.iv), &TileType::Scalar(ScalarDType::U32));
        assert_eq!(f.value_type(lp.acc), &TileType::Scalar(ScalarDType::F32));
        assert_eq!(f.value_type(lp.result), &TileType::Scalar(ScalarDType::F32));

        // Body: load + add
        let val = f.load(ptr, lp.iv, ScalarDType::F32);
        let new_acc = f.add(lp.acc, val);
        f.end_for_acc(&lp, new_acc);

        // After exit block, result is available
        let _out = lp.result;

        f.return_();

        let ir = f.dump();
        eprintln!("=== ForAcc SSA IR ===\n{}", ir);

        // Verify CFG structure
        // Header block should have 2 params (iv: U32, acc: F32)
        let blocks = f.all_blocks();
        let header = &blocks[lp.header.0 as usize];
        assert_eq!(header.params.len(), 2, "Header should have 2 block params (iv, acc)");

        // Exit block should have 1 param (result: F32)
        let exit = &blocks[lp.exit.0 as usize];
        assert_eq!(exit.params.len(), 1, "Exit should have 1 block param (result)");

        eprintln!("✓ for_range_with_acc_runtime: CFG verified");
    }
}
