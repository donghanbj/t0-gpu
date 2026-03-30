//! Checkpoint — Gradient checkpointing (activation recompute).
//!
//! Stub implementation — will be completed in Phase 4.

#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use std::sync::Arc;

/// Checkpoint a computation: during forward, only store inputs (not activations).
/// During backward, re-run forward to recompute activations.
///
/// This trades compute for memory — essential for training large models.
#[cfg(feature = "rocm")]
pub fn checkpoint<F>(
    _inputs: &[&Tensor],
    _forward_fn: F,
    _runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String>
where
    F: FnOnce(&[&Tensor]) -> Tensor,
{
    Err("checkpoint: not yet implemented — Phase 4 target".to_string())
}
