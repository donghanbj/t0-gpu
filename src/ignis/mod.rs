//! Ignis — GPU-native autodiff framework for RDNA3 bare-metal training.
//!
//! Built on top of the KFD runtime, Ignis provides:
//! - Tape-based reverse-mode automatic differentiation
//! - GPU tensor ops with ISA kernel dispatch
//! - NN layers (Linear, Embedding, Transformer)
//! - Training infrastructure (DataLoader, Tokenizer, LR scheduler, etc.)

// Core autodiff engine
pub mod tape;
pub mod tensor;
pub mod gpu_context;

// GPU operations with tape recording
pub mod ops;

// Neural network layers
pub mod nn;

// Training infrastructure
pub mod buffer_pool;
pub mod data_loader;
pub mod tokenizer;
pub mod lr_scheduler;
pub mod grad_clip;
pub mod loss_scaler;

// Tests
#[cfg(test)]
pub mod tests;
