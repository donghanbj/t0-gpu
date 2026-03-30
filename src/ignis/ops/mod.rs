//! Ignis ops — GPU-accelerated operations with autodiff tape recording.
//!
//! Each op records its forward computation on the tape and provides
//! a backward closure that dispatches GPU kernels for gradient computation.

pub mod add;
pub mod bf16_matmul;
pub mod rmsnorm;
pub mod silu;
pub mod cross_entropy;
pub mod embedding;
pub mod ocpa_attention;
pub mod shape_ops;
pub mod checkpoint;
pub mod psi_activation;
pub mod gemm_autotune;
pub mod fusion;
