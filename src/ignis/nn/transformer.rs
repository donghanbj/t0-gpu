//! TransformerLayer — OCPA attention + FFN with RMSNorm residual connections.
//!
//! Architecture:
//!   x → RMSNorm → Q/K/V projections → OCPA → output projection → residual
//!   x → RMSNorm → gate/up projections → SiLU gate → down projection → residual

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::Module;
#[cfg(feature = "rocm")]
use super::linear::Linear;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use super::super::ops;

/// Single Transformer layer with OCPA attention.
///
/// 9 weight matrices + 2 RMSNorm gammas = 11 parameter tensors.
#[cfg(feature = "rocm")]
pub struct TransformerLayer {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub attn_norm_gamma: Tensor,
    pub ffn_norm_gamma: Tensor,
    pub dim: usize,
    pub d_head: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    runtime: Arc<GpuRuntime>,
}

#[cfg(feature = "rocm")]
impl TransformerLayer {
    pub fn new(
        runtime: &Arc<GpuRuntime>,
        dim: usize,
        n_heads: usize,
        ffn_mult: usize, // typically 4
        layer_idx: usize,
    ) -> Result<Self, String> {
        let d_head = dim / n_heads;
        let ffn_dim = dim * ffn_mult;
        let prefix = format!("L{}", layer_idx);

        Ok(Self {
            wq: Linear::new(runtime, dim, dim, &format!("{}_wq", prefix))?,
            wk: Linear::new(runtime, dim, dim, &format!("{}_wk", prefix))?,
            wv: Linear::new(runtime, dim, dim, &format!("{}_wv", prefix))?,
            wo: Linear::new(runtime, dim, dim, &format!("{}_wo", prefix))?,
            w_gate: Linear::new(runtime, dim, ffn_dim, &format!("{}_gate", prefix))?,
            w_up: Linear::new(runtime, dim, ffn_dim, &format!("{}_up", prefix))?,
            w_down: Linear::new(runtime, ffn_dim, dim, &format!("{}_down", prefix))?,
            attn_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; dim], &[dim],
                    &format!("{}_attn_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            ffn_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; dim], &[dim],
                    &format!("{}_ffn_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            dim,
            d_head,
            n_heads,
            ffn_dim,
            runtime: runtime.clone(),
        })
    }

    /// Simple forward (no OCPA — standard matmul attention for testing).
    pub fn forward_simple(&self, x: &Tensor) -> Result<Tensor, String> {
        let device = &self.runtime.device;

        // Attention sub-layer
        let h = ops::rmsnorm::rmsnorm(x, &self.attn_norm_gamma, device)?;
        let q = self.wq.forward(&h)?;
        let _k = self.wk.forward(&h)?;
        let _v = self.wv.forward(&h)?;

        // Simplified attention: just Q @ K^T @ V → Wo → residual
        let attn_out = ops::bf16_matmul::matmul(&q, &self.wo.weight, device)?;
        let x2 = ops::add::add(x, &attn_out, device)?;

        // FFN sub-layer
        let h2 = ops::rmsnorm::rmsnorm(&x2, &self.ffn_norm_gamma, device)?;
        let gate = self.w_gate.forward(&h2)?;
        let up = self.w_up.forward(&h2)?;
        let silu_out = ops::silu::silu_gate(&gate, &up, device)?;
        let ffn_out = self.w_down.forward(&silu_out)?;

        ops::add::add(&x2, &ffn_out, device)
    }

    /// Full forward with OCPA attention.
    pub fn forward_ocpa(&self, x: &Tensor, config: &ops::ocpa_attention::OcpaConfig) -> Result<Tensor, String> {
        let device = &self.runtime.device;

        let h = ops::rmsnorm::rmsnorm(x, &self.attn_norm_gamma, device)?;
        let q = self.wq.forward(&h)?;
        let k = self.wk.forward(&h)?;
        let v = self.wv.forward(&h)?;

        let attn_out = ops::ocpa_attention::ocpa_forward(&q, &k, &v, config, &self.runtime)?;
        let proj_out = self.wo.forward(&attn_out)?;
        let x2 = ops::add::add(x, &proj_out, device)?;

        let h2 = ops::rmsnorm::rmsnorm(&x2, &self.ffn_norm_gamma, device)?;
        let gate = self.w_gate.forward(&h2)?;
        let up = self.w_up.forward(&h2)?;
        let silu_out = ops::silu::silu_gate(&gate, &up, device)?;
        let ffn_out = self.w_down.forward(&silu_out)?;

        ops::add::add(&x2, &ffn_out, device)
    }
}

#[cfg(feature = "rocm")]
impl Module for TransformerLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        self.forward_simple(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.wq.weight, &self.wk.weight, &self.wv.weight, &self.wo.weight,
            &self.w_gate.weight, &self.w_up.weight, &self.w_down.weight,
            &self.attn_norm_gamma, &self.ffn_norm_gamma,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.wq.weight, &mut self.wk.weight, &mut self.wv.weight, &mut self.wo.weight,
            &mut self.w_gate.weight, &mut self.w_up.weight, &mut self.w_down.weight,
            &mut self.attn_norm_gamma, &mut self.ffn_norm_gamma,
        ]
    }
}
