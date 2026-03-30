//! Psi activation — custom ψ activation function.
//!
//! Partial implementation: forward dispatch only, no backward or tape recording.

#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use std::sync::Arc;

/// Apply ψ activation in-place on an f32 tensor.
///
/// Dispatches `f32_psi_inplace` kernel. No tape recording (gradients
/// will not flow through this op).
#[cfg(feature = "rocm")]
pub fn psi_inplace(
    tensor: &Tensor,
    runtime: &Arc<GpuRuntime>,
) -> Result<(), String> {
    let n = tensor.numel();

    // Ψ(x) = 1 + 2·σ(x) via block_dsl (in-place)
    let kernel = {
        let cached = runtime.get_kernel("bdsl_psi_inplace");
        if let Some(k) = cached {
            k
        } else {
            use crate::t0::block_dsl::BlockKernel;
            use crate::t0::ir::Target;
            let mut kb = BlockKernel::new("bdsl_psi_inplace", 256);
            let ptr = kb.arg_ptr("ptr");
            let n_arg = kb.arg_u32("n");

            let pid = kb.program_id(0);
            let bs = kb.const_u32(256);
            let base = pid.mul(&mut kb, bs);
            let tid = kb.arange(0, 256);
            let off = tid.add(&mut kb, base);

            let x = kb.load_checked(ptr, off, n_arg);
            let sig = x.sigmoid(&mut kb);
            let two = kb.const_f32(2.0);
            let one = kb.const_f32(1.0);
            let result = two.mul(&mut kb, sig).add(&mut kb, one); // 1 + 2σ(x)
            kb.store_checked(ptr, off, result, n_arg);

            let compiled = kb.compile(Target::GFX1100)?;
            runtime.compile_dsl(compiled)?
        }
    };

    let grid_x = ((n as u32 + 255) / 256) * 256;
    let ka = crate::kernargs![
        tensor.gpu_addr() => u64,
        n as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)
}
