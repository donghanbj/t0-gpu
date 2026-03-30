//! NN module — Parameter, Module trait, and layer exports.

#[cfg(feature = "rocm")]
use super::tensor::Tensor;

pub mod linear;
pub mod embedding;
pub mod transformer;
pub mod model;

/// A trainable parameter — wrapper around a Tensor with requires_grad=true.
#[cfg(feature = "rocm")]
pub struct Parameter {
    pub tensor: Tensor,
    pub name: String,
}

#[cfg(feature = "rocm")]
impl Parameter {
    pub fn new(tensor: Tensor, name: &str) -> Self {
        let mut t = tensor;
        t.set_requires_grad(true);
        Self { tensor: t, name: name.to_string() }
    }
}

/// Module trait — defines the interface for neural network layers.
#[cfg(feature = "rocm")]
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor, String>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
}
