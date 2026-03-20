//! # T0-GPU
//!
//! RDNA3 (GFX1100) 裸金属 GPU 内核编译器 & KFD 运行时。
//!
//! - **T0 编译器**: 数学 IR → GFX1100 ISA → AMD HSA ELF
//! - **KFD 运行时**: 直接通过 /dev/kfd 驱动接口与 GPU 通信
//!
//! ## 示例
//! ```ignore
//! use t0_gpu::t0::{T0Kernel, Target, GFX1100Schedule};
//! use t0_gpu::t0::math;
//!
//! // 编译一个 bf16 GEMM 内核
//! let kernel = math::matmul_direct(&GFX1100Schedule {});
//! let elf = kernel.compile(Target::GFX1100).unwrap();
//! ```

// ── T0 编译器 ──
pub mod t0;

// ── ISA 编码器 ──
pub mod rdna3_asm;

// ── Code Object (ELF) 生成器 ──
pub mod rdna3_code_object;

// ── KFD 裸金属运行时 ──
#[cfg(feature = "rocm")]
pub mod kfd;
