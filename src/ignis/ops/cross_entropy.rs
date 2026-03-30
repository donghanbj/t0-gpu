//! Cross-entropy loss — softmax + negative log-likelihood with autodiff.
//!
//! Forward: loss = -sum(one_hot(targets) * log(softmax(logits))) / batch
//! Backward: d_logits = (softmax(logits) - one_hot(targets)) / batch
//!
//! Uses build_softmax_ce_loss() ISA kernel when available.
//! Kernarg (40 bytes):
//!   [0:8]   logits_ptr
//!   [8:16]  targets_ptr
//!   [16:24] loss_ptr  
//!   [24:28] vocab_size
//!   [28:32] batch_size (= rows)
//!   [32:40] grad_ptr

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::kfd::GpuBuffer;
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::tape::Tape;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Softmax Cross-Entropy Loss (fused forward + backward gradient pre-computation)
///
/// Computes:
///   probs = softmax(logits)    per row
///   loss  = -mean(log(probs[target]))
///   grad  = (probs - one_hot(target)) / batch_size   (saved for backward)
///
/// # Arguments
/// - logits: [batch, vocab_size] f32
/// - targets: [batch] u32 (token indices, stored as f32-cast or raw u32)
/// - vocab_size: vocabulary dimension
///
/// # Returns
/// - Scalar loss tensor [1]
#[cfg(feature = "rocm")]
pub fn cross_entropy(
    logits: &Tensor,
    targets_buf: &GpuBuffer,
    vocab_size: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    let batch = logits.numel() / vocab_size;
    let cols = vocab_size;
    let grid = batch as u32 * 32;

    // ═══ Step 1: GPU Softmax via composed kernels ═══
    // 1a. row_max
    let max_buf = runtime.alloc_f32(batch)?;
    let k_max = runtime.ensure_kernel_t0("ce_row_max",
        || crate::t0::math::t0_row_reduce_max(), [32, 1, 1], 0)?;
    let ka1 = crate::kernargs![
        logits.gpu_addr() => u64, max_buf.gpu_addr() => u64, cols as u32 => u32
    ];
    runtime.dispatch(&k_max, [grid, 1, 1], &ka1)?;

    // 1b. exp(logits - max) [fused]
    let probs_buf = runtime.alloc_f32(batch * cols)?;
    let k_exp = runtime.ensure_kernel_t0("ce_exp_sub",
        || crate::t0::math::t0_row_broadcast_exp_sub(), [32, 1, 1], 0)?;
    let ka2 = crate::kernargs![
        logits.gpu_addr() => u64, max_buf.gpu_addr() => u64,
        probs_buf.gpu_addr() => u64, cols as u32 => u32
    ];
    runtime.dispatch(&k_exp, [grid, 1, 1], &ka2)?;

    // 1c. row_sum(exp)
    let sum_buf = runtime.alloc_f32(batch)?;
    let k_sum = runtime.ensure_kernel_t0("ce_row_sum",
        || crate::t0::math::t0_row_reduce_sum(), [32, 1, 1], 0)?;
    let ka3 = crate::kernargs![
        probs_buf.gpu_addr() => u64, sum_buf.gpu_addr() => u64, cols as u32 => u32
    ];
    runtime.dispatch(&k_sum, [grid, 1, 1], &ka3)?;

    // 1d. probs = exp / sum [broadcast div, in-place on probs_buf]
    let k_div = runtime.ensure_kernel_t0("ce_bcast_div",
        || crate::t0::math::t0_row_broadcast_div(), [32, 1, 1], 0)?;
    let probs_out = runtime.alloc_f32(batch * cols)?;
    let ka4 = crate::kernargs![
        probs_buf.gpu_addr() => u64, sum_buf.gpu_addr() => u64,
        probs_out.gpu_addr() => u64, cols as u32 => u32
    ];
    runtime.dispatch(&k_div, [grid, 1, 1], &ka4)?;

    // ═══ Step 2: CPU loss + grad (batch-level, tiny) ═══
    runtime.wait_idle()?;
    let probs = read_f32_vec(&probs_out, batch * cols);

    let mut targets_bytes = vec![0u8; batch * 4];
    targets_buf.read(&mut targets_bytes);
    let targets: Vec<u32> = targets_bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Loss: -mean(log(probs[target]))
    let mut total_loss = 0f32;
    for b in 0..batch {
        let target = targets[b] as usize;
        if target < cols {
            total_loss += -probs[b * cols + target].max(1e-10).ln();
        }
    }
    total_loss /= batch as f32;

    let loss = Tensor::from_f32(runtime, &[total_loss], &[1], "ce_loss")?;

    if Tape::is_recording() && logits.requires_grad() {
        let logits_id = Some(logits.id());
        let v = cols;
        let bs = batch;

        // Grad: (softmax - one_hot) / batch → upload to GPU
        let mut grad_data = probs;
        for b in 0..bs {
            let target = targets[b] as usize;
            if target < v {
                grad_data[b * v + target] -= 1.0;
            }
            for c in 0..v {
                grad_data[b * v + c] /= bs as f32;
            }
        }
        let grad_buf = runtime.alloc_f32(bs * v)?;
        write_f32(&grad_buf, &grad_data);
        let grad_arc = Arc::new(grad_buf);

        let node_id = Tape::record(
            "cross_entropy",
            loss.id(),
            vec![logits_id],
            vec![true],
            vec![grad_arc],
            Box::new(move |_grad_output, saved, _runtime| {
                Ok(vec![Some(saved[0].clone())])
            }),
        );
        loss.set_tape_node(node_id);
    }

    Ok(loss)
}

#[cfg(feature = "rocm")]
fn read_f32_vec(buf: &GpuBuffer, n: usize) -> Vec<f32> {
    let mut data = vec![0f32; n];
    buf.read(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) });
    data
}

#[cfg(feature = "rocm")]
fn write_f32(buf: &GpuBuffer, data: &[f32]) {
    buf.write(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) });
}
