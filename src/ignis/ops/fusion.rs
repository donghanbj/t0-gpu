//! Auto-fusion: compose multiple elementwise ops into a single GPU kernel.
//!
//! ## Example
//!
//! ```rust
//! // Without fusion: 3 kernel dispatches
//! let sum = ops::add::add(&a, &b, &device)?;
//! let scaled = ops::add::scale(&sum, 2.0, &device)?;
//! let out = ops::psi_activation::psi_inplace(&scaled, &rt)?;
//!
//! // With fusion: 1 kernel dispatch
//! let out = FusedOp::binary(&rt, &a, &b, "add_scale_psi", |kb, va, vb| {
//!     let sum = va.add(kb, vb);
//!     let scaled = sum.mul(kb, kb.const_f32(2.0));
//!     let sig = scaled.sigmoid(kb);
//!     let two = kb.const_f32(2.0);
//!     let one = kb.const_f32(1.0);
//!     two.mul(kb, sig).add(kb, one)  // psi = 1 + 2σ(x)
//! })?;
//! ```

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use crate::t0::block_dsl::{BlockKernel, BVal};

/// Builder for fused elementwise operations.
///
/// Composes multiple elementwise ops into a single GPU kernel dispatch.
/// Supports unary (1 input), binary (2 inputs), and ternary (3 inputs) patterns.
#[cfg(feature = "rocm")]
pub struct FusedOp;

#[cfg(feature = "rocm")]
impl FusedOp {
    /// Fused unary op: out = f(a)
    ///
    /// Single kernel: load a → compute → store out
    pub fn unary<F>(
        runtime: &Arc<GpuRuntime>,
        a: &Tensor,
        name: &str,
        f: F,
    ) -> Result<Tensor, String>
    where
        F: FnOnce(&mut BlockKernel, BVal) -> BVal,
    {
        let n = a.numel();
        let shape = a.shape().to_vec();

        let kernel_name = format!("bdsl_fused_{}", name);
        let kernel = {
            let cached = runtime.get_kernel(&kernel_name);
            if let Some(k) = cached {
                k
            } else {
                use crate::t0::ir::Target;
                let mut kb = BlockKernel::new(&kernel_name, 256);
                let a_ptr = kb.arg_ptr("a");
                let out_ptr = kb.arg_ptr("out");
                let n_arg = kb.arg_u32("n");

                let pid = kb.program_id(0);
                let bs = kb.const_u32(256);
                let base = pid.mul(&mut kb, bs);
                let tid = kb.arange(0, 256);
                let off = tid.add(&mut kb, base);

                let va = kb.load_checked(a_ptr, off, n_arg);
                let result = f(&mut kb, va);
                kb.store_checked(out_ptr, off, result, n_arg);

                let compiled = kb.compile(Target::GFX1100)?;
                runtime.compile_dsl(compiled)?
            }
        };

        let out_buf = runtime.alloc_f32(n)?;
        let grid_x = ((n as u32 + 255) / 256) * 256;
        let ka = crate::kernargs![
            a.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            n as u32 => u32
        ];
        runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

        Ok(Tensor::from_buffer(
            Arc::new(out_buf),
            runtime,
            &shape,
            DType::F32,
            "fused_out",
        ))
    }

    /// Fused binary op: out = f(a, b)
    ///
    /// Single kernel: load a, b → compute → store out
    pub fn binary<F>(
        runtime: &Arc<GpuRuntime>,
        a: &Tensor,
        b: &Tensor,
        name: &str,
        f: F,
    ) -> Result<Tensor, String>
    where
        F: FnOnce(&mut BlockKernel, BVal, BVal) -> BVal,
    {
        assert_eq!(a.numel(), b.numel(), "FusedOp::binary: shape mismatch");
        let n = a.numel();
        let shape = a.shape().to_vec();

        let kernel_name = format!("bdsl_fused_{}", name);
        let kernel = {
            let cached = runtime.get_kernel(&kernel_name);
            if let Some(k) = cached {
                k
            } else {
                use crate::t0::ir::Target;
                let mut kb = BlockKernel::new(&kernel_name, 256);
                let a_ptr = kb.arg_ptr("a");
                let b_ptr = kb.arg_ptr("b");
                let out_ptr = kb.arg_ptr("out");
                let n_arg = kb.arg_u32("n");

                let pid = kb.program_id(0);
                let bs = kb.const_u32(256);
                let base = pid.mul(&mut kb, bs);
                let tid = kb.arange(0, 256);
                let off = tid.add(&mut kb, base);

                let va = kb.load_checked(a_ptr, off, n_arg);
                let vb = kb.load_checked(b_ptr, off, n_arg);
                let result = f(&mut kb, va, vb);
                kb.store_checked(out_ptr, off, result, n_arg);

                let compiled = kb.compile(Target::GFX1100)?;
                runtime.compile_dsl(compiled)?
            }
        };

        let out_buf = runtime.alloc_f32(n)?;
        let grid_x = ((n as u32 + 255) / 256) * 256;
        let ka = crate::kernargs![
            a.gpu_addr() => u64,
            b.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            n as u32 => u32
        ];
        runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

        Ok(Tensor::from_buffer(
            Arc::new(out_buf),
            runtime,
            &shape,
            DType::F32,
            "fused_out",
        ))
    }

    /// Fused ternary op: out = f(a, b, c)
    ///
    /// Single kernel: load a, b, c → compute → store out
    pub fn ternary<F>(
        runtime: &Arc<GpuRuntime>,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        name: &str,
        f: F,
    ) -> Result<Tensor, String>
    where
        F: FnOnce(&mut BlockKernel, BVal, BVal, BVal) -> BVal,
    {
        assert_eq!(a.numel(), b.numel(), "FusedOp::ternary: a/b shape mismatch");
        assert_eq!(a.numel(), c.numel(), "FusedOp::ternary: a/c shape mismatch");
        let n = a.numel();
        let shape = a.shape().to_vec();

        let kernel_name = format!("bdsl_fused_{}", name);
        let kernel = {
            let cached = runtime.get_kernel(&kernel_name);
            if let Some(k) = cached {
                k
            } else {
                use crate::t0::ir::Target;
                let mut kb = BlockKernel::new(&kernel_name, 256);
                let a_ptr = kb.arg_ptr("a");
                let b_ptr = kb.arg_ptr("b");
                let c_ptr = kb.arg_ptr("c");
                let out_ptr = kb.arg_ptr("out");
                let n_arg = kb.arg_u32("n");

                let pid = kb.program_id(0);
                let bs = kb.const_u32(256);
                let base = pid.mul(&mut kb, bs);
                let tid = kb.arange(0, 256);
                let off = tid.add(&mut kb, base);

                let va = kb.load_checked(a_ptr, off, n_arg);
                let vb = kb.load_checked(b_ptr, off, n_arg);
                let vc = kb.load_checked(c_ptr, off, n_arg);
                let result = f(&mut kb, va, vb, vc);
                kb.store_checked(out_ptr, off, result, n_arg);

                let compiled = kb.compile(Target::GFX1100)?;
                runtime.compile_dsl(compiled)?
            }
        };

        let out_buf = runtime.alloc_f32(n)?;
        let grid_x = ((n as u32 + 255) / 256) * 256;
        let ka = crate::kernargs![
            a.gpu_addr() => u64,
            b.gpu_addr() => u64,
            c.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            n as u32 => u32
        ];
        runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

        Ok(Tensor::from_buffer(
            Arc::new(out_buf),
            runtime,
            &shape,
            DType::F32,
            "fused_out",
        ))
    }
}
