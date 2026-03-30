//! Learning rate schedulers — CosineWarmup and ConstantLR.

/// Learning rate scheduler trait.
pub trait LrScheduler {
    fn get_lr(&self, step: usize) -> f32;
}

/// Cosine annealing with linear warmup.
///
/// - Steps 0..warmup: linear ramp from 0 to max_lr
/// - Steps warmup..total: cosine decay from max_lr to min_lr
pub struct CosineWarmupScheduler {
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl CosineWarmupScheduler {
    pub fn new(max_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self { max_lr, min_lr, warmup_steps, total_steps }
    }
}

impl LrScheduler for CosineWarmupScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.max_lr * (step as f32 / self.warmup_steps as f32)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
            self.min_lr + (self.max_lr - self.min_lr) * cosine
        }
    }
}

/// Constant learning rate.
pub struct ConstantLR {
    pub lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self { Self { lr } }
}

impl LrScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 { self.lr }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_warmup() {
        let sched = CosineWarmupScheduler::new(1e-3, 1e-5, 100, 1000);
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-8);
        assert!((sched.get_lr(50) - 5e-4).abs() < 1e-6);
        assert!((sched.get_lr(100) - 1e-3).abs() < 1e-6);
        assert!(sched.get_lr(500) < 1e-3);
        assert!(sched.get_lr(1000) <= 1e-5 + 1e-6);
    }

    #[test]
    fn test_constant_lr() {
        let sched = ConstantLR::new(0.01);
        assert_eq!(sched.get_lr(0), 0.01);
        assert_eq!(sched.get_lr(1000), 0.01);
    }
}
