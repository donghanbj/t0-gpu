//! LanguageModel — Full model: Embedding → N×TransformerLayer → RMSNorm → LM Head.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::Module;
#[cfg(feature = "rocm")]
use super::linear::Linear;
#[cfg(feature = "rocm")]
use super::embedding::Embedding;
#[cfg(feature = "rocm")]
use super::transformer::TransformerLayer;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use super::super::ops;

/// Complete language model.
#[cfg(feature = "rocm")]
pub struct LanguageModel {
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub final_norm_gamma: Tensor,
    pub lm_head: Linear,
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    runtime: Arc<GpuRuntime>,
}

#[cfg(feature = "rocm")]
impl LanguageModel {
    pub fn new(
        runtime: &Arc<GpuRuntime>,
        vocab_size: usize,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        ffn_mult: usize,
    ) -> Result<Self, String> {
        let embedding = Embedding::new(runtime, vocab_size, dim, "emb")?;

        let mut layers = Vec::new();
        for i in 0..n_layers {
            layers.push(TransformerLayer::new(runtime, dim, n_heads, ffn_mult, i)?);
        }

        let mut final_norm_gamma = Tensor::from_f32(
            runtime, &vec![1.0f32; dim], &[dim], "final_norm",
        )?;
        final_norm_gamma.set_requires_grad(true);

        let lm_head = Linear::new(runtime, dim, vocab_size, "lm_head")?;

        Ok(Self {
            embedding, layers, final_norm_gamma, lm_head,
            dim, n_layers, vocab_size, runtime: runtime.clone(),
        })
    }

    /// Forward pass: token_ids → logits
    pub fn forward_ids(&self, ids: &[u32]) -> Result<Tensor, String> {
        let device = &self.runtime.device;

        // Embedding
        let mut h = self.embedding.forward_cpu(ids)?;

        // Transformer layers
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        // Final RMSNorm
        h = ops::rmsnorm::rmsnorm(&h, &self.final_norm_gamma, device)?;

        // LM head → logits
        self.lm_head.forward(&h)
    }

    /// Get all parameters for optimizer.
    pub fn all_parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.embedding.weight];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.push(&self.final_norm_gamma);
        params.push(&self.lm_head.weight);
        params
    }

    /// Get all mutable parameters.
    pub fn all_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = vec![&mut self.embedding.weight];
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params.push(&mut self.final_norm_gamma);
        params.push(&mut self.lm_head.weight);
        params
    }

    /// Total number of parameters.
    pub fn param_count(&self) -> usize {
        self.all_parameters().iter().map(|t| t.numel()).sum()
    }
}
