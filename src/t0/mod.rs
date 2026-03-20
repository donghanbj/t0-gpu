//! T0 内核编译器 — 多层 GPU 内核代码生成框架
//!
//! 从高级数学表达式自动生成 GPU 机器码:
//!   T0-high (数学层) → T0-mid (调度层) → T0-low (代码生成层) → LLVM → ELF
//!
//! ## 使用示例
//! ```ignore
//! use t0::{T0Kernel, Target, Width, WmmaFormat, Alignment};
//!
//! let mut k = T0Kernel::new("my_kernel");
//! let ptr = k.arg_ptr("input");
//! let n = k.arg_u32("n_elems");
//! let val = k.alloc_vreg();
//! let addr = k.thread_global_addr(ptr);
//! k.global_load(val, addr, Width::B32, 0);
//! k.v_mul_f32(val, val, val);  // square
//! k.global_store(addr, val, Width::B32, 0);
//! k.endpgm();
//! let elf = k.compile(Target::GFX1100)?;
//! ```

pub mod ir;
pub mod regalloc;
pub mod asm_emitter;
pub mod compile;
pub mod schedule;
pub mod math;
// gpu_tests omitted in open-source build

pub use ir::*;
pub use compile::T0Kernel;
pub use schedule::{Schedule, GFX1100Schedule};
