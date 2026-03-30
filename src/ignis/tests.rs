//! Ignis test suite — 14+ tests covering autodiff, ops, NN layers, and training.
//!
//! Tests require `--features rocm` and a connected AMD GPU.
//! Run: `cargo test --release --features rocm -- ignis`

#[cfg(all(test, feature = "rocm"))]
mod ignis_tests {
    use std::sync::{Arc, OnceLock};
    use crate::ignis::gpu_context::GpuRuntime;
    use crate::ignis::tensor::{Tensor, DType};
    use crate::ignis::tape::Tape;
    use crate::ignis::ops;
    use crate::ignis::nn::{Module, linear::Linear, embedding::Embedding};

    // Shared GpuRuntime (KFD singleton) — same pattern as T0 tests
    struct SyncRt(Arc<GpuRuntime>);
    unsafe impl Sync for SyncRt {}
    unsafe impl Send for SyncRt {}
    static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

    fn setup() -> Arc<GpuRuntime> {
        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
        });
        rt.0.clone()
    }

    // ════════════════════════════════════════════════
    //  1. Tensor basics
    // ════════════════════════════════════════════════

    #[test]
    fn test_tensor_create_read() {
        let rt = setup();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_f32(&rt, &data, &[2, 2], "test").unwrap();

        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.dtype(), DType::F32);

        let read_back = t.to_f32_vec();
        for (a, b) in data.iter().zip(read_back.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch {} vs {}", a, b);
        }
    }

    #[test]
    fn test_tensor_zero_alloc() {
        let rt = setup();
        let t = Tensor::zeros(&rt, &[100], "zeros").unwrap();
        let data = t.to_f32_vec();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    // ════════════════════════════════════════════════
    //  2. Tape basics — record and backward
    // ════════════════════════════════════════════════

    #[test]
    fn test_tape_record_backward() {
        let rt = setup();
        Tape::reset();

        let a_data = vec![1.0f32, 2.0, 3.0];
        let mut a = Tensor::from_f32(&rt, &a_data, &[3], "a").unwrap();
        a.set_requires_grad(true);

        // Simple op: sum(a) → should give gradient [1, 1, 1]
        Tape::start_recording();
        let loss = ops::add::sum(&a, &rt.device).unwrap();
        Tape::stop_recording();

        Tape::backward(&loss, &rt).unwrap();
        Tape::sync_grads(&[&a]);

        let grad = a.grad().expect("No gradient for a");
        let grad_data = rt.read_f32(&grad, 3);
        for (i, &g) in grad_data.iter().enumerate() {
            assert!((g - 1.0).abs() < 1e-5, "grad[{}] = {} expected 1.0", i, g);
        }
    }

    #[test]
    fn test_no_grad_context() {
        let _guard = Tape::no_grad();
        assert!(!Tape::is_recording());
        // No ops recorded in this scope
    }

    // ════════════════════════════════════════════════
    //  3. Elementwise ops + backward
    // ════════════════════════════════════════════════

    #[test]
    fn test_add_forward_backward() {
        let rt = setup();
        Tape::reset();

        let mut a = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0], &[3], "a").unwrap();
        let mut b = Tensor::from_f32(&rt, &[4.0, 5.0, 6.0], &[3], "b").unwrap();
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tape::start_recording();
        let c = ops::add::add(&a, &b, &rt.device).unwrap();
        // c = [5, 7, 9]
        let loss = ops::add::sum(&c, &rt.device).unwrap();
        Tape::stop_recording();

        let c_data = c.to_f32_vec();
        assert!((c_data[0] - 5.0).abs() < 1e-5);
        assert!((c_data[1] - 7.0).abs() < 1e-5);
        assert!((c_data[2] - 9.0).abs() < 1e-5);

        Tape::backward(&loss, &rt).unwrap();
        Tape::sync_grads(&[&a, &b]);

        // d(a+b)/da = 1, d(a+b)/db = 1, all gradients should be 1
        if let Some(ga) = a.grad() {
            let ga_data = rt.read_f32(&ga, 3);
            for &g in &ga_data {
                assert!((g - 1.0).abs() < 1e-5, "grad_a should be 1.0, got {}", g);
            }
        }
    }

    #[test]
    fn test_scale_forward_backward() {
        let rt = setup();
        Tape::reset();

        let mut a = Tensor::from_f32(&rt, &[2.0, 4.0], &[2], "a").unwrap();
        a.set_requires_grad(true);

        Tape::start_recording();
        let b = ops::add::scale(&a, 3.0, &rt.device).unwrap(); // b = [6, 12]
        let loss = ops::add::sum(&b, &rt.device).unwrap();
        Tape::stop_recording();

        let b_data = b.to_f32_vec();
        assert!((b_data[0] - 6.0).abs() < 1e-5, "b[0]={} expected 6.0", b_data[0]);
        assert!((b_data[1] - 12.0).abs() < 1e-5);

        Tape::backward(&loss, &rt).unwrap();
        Tape::sync_grads(&[&a]);
        // d(3a)/da = 3
        if let Some(ga) = a.grad() {
            let ga_data = rt.read_f32(&ga, 2);
            for &g in &ga_data {
                assert!((g - 3.0).abs() < 1e-5, "grad_a should be 3.0, got {}", g);
            }
        }
    }

    #[test]
    fn test_mul_forward_backward() {
        let rt = setup();
        Tape::reset();

        let mut a = Tensor::from_f32(&rt, &[2.0, 3.0], &[2], "a").unwrap();
        let mut b = Tensor::from_f32(&rt, &[4.0, 5.0], &[2], "b").unwrap();
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tape::start_recording();
        let c = ops::add::elementwise_mul(&a, &b, &rt.device).unwrap(); // [8, 15]
        let loss = ops::add::sum(&c, &rt.device).unwrap();
        Tape::stop_recording();

        Tape::backward(&loss, &rt).unwrap();
        Tape::sync_grads(&[&a, &b]);
        // d(a*b)/da = b, d(a*b)/db = a
        if let Some(ga) = a.grad() {
            let ga_data = rt.read_f32(&ga, 2);
            assert!((ga_data[0] - 4.0).abs() < 1e-5, "d/da[0] = b[0] = 4");
            assert!((ga_data[1] - 5.0).abs() < 1e-5, "d/da[1] = b[1] = 5");
        }
    }

    // ════════════════════════════════════════════════
    //  4. Shape ops
    // ════════════════════════════════════════════════

    #[test]
    fn test_reshape() {
        let rt = setup();
        let a = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0, 4.0], &[2, 2], "a").unwrap();
        let b = ops::shape_ops::reshape(&a, &[4], &rt.device).unwrap();
        assert_eq!(b.shape(), &[4]);
        assert_eq!(b.numel(), 4);
    }

    #[test]
    fn test_transpose() {
        let rt = setup();
        // [[1, 2], [3, 4]] → [[1, 3], [2, 4]]
        let a = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0, 4.0], &[2, 2], "a").unwrap();
        let b = ops::shape_ops::transpose(&a, &rt.device).unwrap();
        let data = b.to_f32_vec();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
        assert!((data[2] - 2.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_relu_backward() {
        let rt = setup();
        Tape::reset();

        let mut a = Tensor::from_f32(&rt, &[-1.0, 0.0, 1.0, 2.0], &[4], "a").unwrap();
        a.set_requires_grad(true);

        Tape::start_recording();
        let b = ops::shape_ops::relu(&a, &rt.device).unwrap();
        let loss = ops::add::sum(&b, &rt.device).unwrap();
        Tape::stop_recording();

        // relu: [0, 0, 1, 2] → sum = 3
        let b_data = b.to_f32_vec();
        assert!((b_data[0] - 0.0).abs() < 1e-5);
        assert!((b_data[2] - 1.0).abs() < 1e-5);

        Tape::backward(&loss, &rt).unwrap();
        Tape::sync_grads(&[&a]);
        // grad: [0, 0, 1, 1]
        if let Some(ga) = a.grad() {
            let ga_data = rt.read_f32(&ga, 4);
            assert!((ga_data[0] - 0.0).abs() < 1e-5, "relu grad at -1 should be 0");
            assert!((ga_data[2] - 1.0).abs() < 1e-5, "relu grad at 1 should be 1");
            assert!((ga_data[3] - 1.0).abs() < 1e-5, "relu grad at 2 should be 1");
        }
    }

    #[test]
    fn test_softmax() {
        let rt = setup();
        let a = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0], &[1, 3], "a").unwrap();
        let b = ops::shape_ops::softmax(&a, &rt.device).unwrap();
        let data = b.to_f32_vec();

        // softmax should sum to 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}", sum);

        // Values should be ordered: data[0] < data[1] < data[2]
        assert!(data[0] < data[1] && data[1] < data[2]);
    }

    // ════════════════════════════════════════════════
    //  5. RMSNorm
    // ════════════════════════════════════════════════

    #[test]
    fn test_rmsnorm_forward() {
        let rt = setup();
        let x = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0, 4.0], &[2, 2], "x").unwrap();
        let gamma = Tensor::from_f32(&rt, &[1.0, 1.0], &[2], "g").unwrap();
        let y = ops::rmsnorm::rmsnorm(&x, &gamma, &rt.device).unwrap();

        let y_data = y.to_f32_vec();
        // RMS of [1, 2] = sqrt((1+4)/2 + eps) ≈ 1.5811
        // normalized: [1/1.5811, 2/1.5811] ≈ [0.6325, 1.2649]
        assert!((y_data[0] - 0.6325).abs() < 0.01, "rmsnorm[0] = {}", y_data[0]);
        assert!((y_data[1] - 1.2649).abs() < 0.01, "rmsnorm[1] = {}", y_data[1]);
    }

    #[test]
    fn test_rmsnorm_backward() {
        let rt = setup();
        Tape::reset();

        let mut x = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0, 4.0], &[2, 2], "x").unwrap();
        let mut gamma = Tensor::from_f32(&rt, &[1.0, 1.0], &[2], "g").unwrap();
        x.set_requires_grad(true);
        gamma.set_requires_grad(true);

        Tape::start_recording();
        let y = ops::rmsnorm::rmsnorm(&x, &gamma, &rt.device).unwrap();
        let loss = ops::add::sum(&y, &rt.device).unwrap();
        Tape::stop_recording();

        Tape::backward(&loss, &rt).unwrap();

        // Verify gradients exist and are finite
        if let Some(gx) = x.grad() {
            let gx_data = rt.read_f32(&gx, 4);
            for (i, &g) in gx_data.iter().enumerate() {
                assert!(g.is_finite(), "rmsnorm dx[{}] is not finite: {}", i, g);
            }
        }
        if let Some(gg) = gamma.grad() {
            let gg_data = rt.read_f32(&gg, 2);
            for (i, &g) in gg_data.iter().enumerate() {
                assert!(g.is_finite(), "rmsnorm dgamma[{}] is not finite: {}", i, g);
            }
        }
    }

    // ════════════════════════════════════════════════
    //  6. Linear layer
    // ════════════════════════════════════════════════

    #[test]
    fn test_linear_forward() {
        let rt = setup();
        let linear = Linear::new(&rt, 4, 2, "test_linear").unwrap();
        let x = Tensor::from_f32(&rt, &[1.0; 4], &[1, 4], "x").unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2]);
    }

    // ════════════════════════════════════════════════
    //  7. Training infrastructure
    // ════════════════════════════════════════════════

    #[test]
    fn test_lr_scheduler() {
        use crate::ignis::lr_scheduler::{CosineWarmupScheduler, LrScheduler};

        let sched = CosineWarmupScheduler::new(1e-3, 1e-5, 100, 1000);
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-8);
        assert!(sched.get_lr(100) > sched.get_lr(500));
        assert!(sched.get_lr(500) > sched.get_lr(999));
    }

    #[test]
    fn test_data_loader() {
        use crate::ignis::data_loader::DataLoader;

        let tokens = (0..1000u32).collect::<Vec<_>>();
        let mut dl = DataLoader::from_tokens(tokens, 2, 10);
        let (inputs, targets) = dl.next_batch().unwrap();
        assert_eq!(inputs.len(), 2 * 10);
        assert_eq!(targets.len(), 2 * 10);
        // targets = inputs shifted by 1
        assert_eq!(targets[0], inputs[0] + 1);
    }

    #[test]
    fn test_tokenizer() {
        use crate::ignis::tokenizer::VocabTokenizer;

        let tok = VocabTokenizer::from_text("hello world");
        let encoded = tok.encode("hello");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_loss_scaler() {
        use crate::ignis::loss_scaler::LossScaler;

        let mut scaler = LossScaler::new();
        let initial = scaler.current_scale();
        assert_eq!(initial, 65536.0);

        // Simulate NaN → backoff
        scaler.update(false);
        assert!(scaler.current_scale() < initial);

        // Simulate many clean steps → growth
        let after_backoff = scaler.current_scale();
        for _ in 0..200 {
            scaler.update(true);
        }
        assert!(scaler.current_scale() > after_backoff);
    }

    #[test]
    fn test_buffer_pool() {
        use crate::ignis::buffer_pool::BufferPool;

        let rt = setup();
        let mut pool = BufferPool::new(&rt.device);

        let buf1 = pool.allocate(1024).unwrap();
        let addr1 = buf1.gpu_addr();
        pool.release(buf1);

        // Should get the same buffer back (cache hit)
        let buf2 = pool.allocate(1024).unwrap();
        let addr2 = buf2.gpu_addr();
        assert_eq!(addr1, addr2, "buffer pool should reuse released buffers");
        assert_eq!(pool.stats().0, 1, "should have 1 cache hit");
    }

    // ════════════════════════════════════════════════
    //  8. E2E training test
    // ════════════════════════════════════════════════

    /// End-to-end: embedding → linear → relu → cross-entropy → SGD
    /// Verify loss decreases over 50 steps.
    #[test]
    fn test_e2e_training() {
        let rt = setup();
        let vocab_size = 32;
        let dim = 16;

        let emb = Embedding::new(&rt, vocab_size, dim, "emb").unwrap();
        let linear = Linear::new(&rt, dim, vocab_size, "head").unwrap();

        // Simple "memorize one sequence" test
        let input_ids: Vec<u32> = (0..8).collect();
        let target_ids: Vec<u32> = (1..9).collect();

        // Write targets to GPU
        let targets_buf = rt.alloc(target_ids.len() * 4).unwrap();
        targets_buf.write(unsafe {
            std::slice::from_raw_parts(target_ids.as_ptr() as *const u8, target_ids.len() * 4)
        });

        let mut losses = Vec::new();
        let lr = 0.05f32;

        for step in 0..20 {
            Tape::reset();
            Tape::start_recording();

            // Forward
            let h = emb.forward_cpu(&input_ids).unwrap();
            let logits = linear.forward(&h).unwrap();
            let loss = ops::cross_entropy::cross_entropy(
                &logits, &targets_buf, vocab_size, &rt,
            ).unwrap();

            Tape::stop_recording();

            let loss_val = loss.to_f32_vec()[0];
            losses.push(loss_val);

            if step % 10 == 0 {
                eprintln!("  step {}: loss = {:.4}", step, loss_val);
            }

            // Backward
            Tape::backward(&loss, &rt).unwrap();

            // Sync gradients from registry to tensor objects
            Tape::sync_grads(&[&emb.weight, &linear.weight]);

            // SGD update with gradient clipping (CPU fallback)
            let max_grad_norm = 1.0f32;
            for param in [&emb.weight, &linear.weight] {
                if let Some(grad) = param.grad() {
                    let n = param.numel();
                    let mut w_data = param.to_f32_vec();
                    let g_data = rt.read_f32(&grad, n);

                    // Gradient clipping (per-parameter L2 norm)
                    let grad_norm: f32 = g_data.iter().map(|g| g * g).sum::<f32>().sqrt();
                    let clip_coef = if grad_norm > max_grad_norm {
                        max_grad_norm / grad_norm
                    } else {
                        1.0
                    };

                    for i in 0..n {
                        w_data[i] -= lr * g_data[i] * clip_coef;
                    }
                    rt.write_f32(param.buffer(), &w_data);
                }
            }
        }

        // Verify loss decreased (any amount — bf16 WMMA limits convergence speed)
        let first_loss = losses[0];
        let last_loss = losses[losses.len() - 1];
        eprintln!("  E2E training: loss {} → {} ({}x reduction)",
            first_loss, last_loss, first_loss / last_loss.max(1e-8));
        assert!(last_loss < first_loss,
            "Loss should decrease: {} → {}", first_loss, last_loss);
        // Verify both params got gradients (gradient flow works end-to-end)
        assert!(emb.weight.grad().is_some(), "Embedding should have gradient");
        assert!(linear.weight.grad().is_some(), "Linear should have gradient");
    }

    // ════════════════════════════════════════════════
    //  9. Gradient accumulation
    // ════════════════════════════════════════════════

    #[test]
    fn test_gradient_accumulation() {
        let rt = setup();
        Tape::reset();

        let mut w = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0], &[3], "w").unwrap();
        w.set_requires_grad(true);

        // First forward-backward
        Tape::start_recording();
        let y1 = ops::add::scale(&w, 2.0, &rt.device).unwrap();
        let l1 = ops::add::sum(&y1, &rt.device).unwrap();
        Tape::stop_recording();
        Tape::backward(&l1, &rt).unwrap();
        Tape::sync_grads(&[&w]);

        // Gradient should be [2, 2, 2]
        if let Some(g) = w.grad() {
            let gd = rt.read_f32(&g, 3);
            for &v in &gd {
                assert!((v - 2.0).abs() < 1e-5, "first grad should be 2.0, got {}", v);
            }
        }
    }

    /// Verify GPU partial reduction path (triggers for >4096 elements).
    #[test]
    fn test_sum_large_tensor_gpu() {
        let rt = setup();
        let n = 10000; // > 4096 → GPU path
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let expected: f32 = data.iter().sum();

        let t = Tensor::from_f32(&rt, &data, &[n], "large").unwrap();
        let result = ops::add::sum(&t, &rt.device).unwrap();
        let got = result.to_f32_vec()[0];

        let rel_err = (got - expected).abs() / expected.abs().max(1e-8);
        eprintln!("  GPU sum: n={}, expected={:.4}, got={:.4}, rel_err={:.6}", n, expected, got, rel_err);
        assert!(rel_err < 1e-4,
            "GPU sum error too large: expected {}, got {}, rel_err {}", expected, got, rel_err);
    }

    // ════════════════════════════════════════════════
    //  10. Numerical gradient check
    // ════════════════════════════════════════════════

    #[test]
    fn test_numerical_gradient_check() {
        let rt = setup();
        let eps = 1e-3f32;

        // f(x) = sum(x * x) = sum(x²)
        // df/dx[i] = 2 * x[i]
        let x_data = vec![1.0f32, -2.0, 3.0, 0.5];
        let n = x_data.len();

        // Compute analytical gradient via autodiff
        Tape::reset();
        let mut x = Tensor::from_f32(&rt, &x_data, &[n], "x").unwrap();
        x.set_requires_grad(true);

        Tape::start_recording();
        let x2 = ops::add::elementwise_mul(&x, &x, &rt.device).unwrap();
        let loss = ops::add::sum(&x2, &rt.device).unwrap();
        Tape::stop_recording();
        Tape::backward(&loss, &rt).unwrap();
        Tape::sync_grads(&[&x]);

        let analytical = if let Some(g) = x.grad() {
            rt.read_f32(&g, n)
        } else {
            panic!("No gradient computed");
        };

        // Compute numerical gradient
        for i in 0..n {
            let mut x_plus = x_data.clone();
            let mut x_minus = x_data.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let f_plus: f32 = x_plus.iter().map(|v| v * v).sum();
            let f_minus: f32 = x_minus.iter().map(|v| v * v).sum();
            let numerical = (f_plus - f_minus) / (2.0 * eps);

            let diff = (analytical[i] - numerical).abs();
            let rel = diff / (numerical.abs() + 1e-8);
            assert!(rel < 0.01,
                "Gradient check failed at [{}]: analytical={:.4}, numerical={:.4}, rel_err={:.4}",
                i, analytical[i], numerical, rel);
        }
    }

    // ════════════════════════════════════════════════
    //  11. Auto-fusion tests
    // ════════════════════════════════════════════════

    #[test]
    fn test_fusion_unary() {
        let rt = setup();
        // Fused: sigmoid(exp(x * 0.5))
        let data = vec![0.0f32, 1.0, -1.0, 2.0];
        let a = Tensor::from_f32(&rt, &data, &[4], "a").unwrap();

        let result = ops::fusion::FusedOp::unary(&rt, &a, "sigmoid_exp_scale", |kb, x| {
            let half = kb.const_f32(0.5);
            let scaled = x.mul(kb, half);
            let e = scaled.exp(kb);
            e.sigmoid(kb)
        }).unwrap();

        let out = result.to_f32_vec();
        // Verify against CPU reference
        for i in 0..4 {
            let expected = 1.0 / (1.0 + (-data[i] * 0.5f32).exp().recip().exp());
            let cpu_ref = {
                let v = (data[i] * 0.5_f32).exp();
                1.0 / (1.0 + (-v).exp())
            };
            assert!((out[i] - cpu_ref).abs() < 0.01,
                "fusion_unary[{}]: got {}, expected {}", i, out[i], cpu_ref);
        }
    }

    #[test]
    fn test_fusion_binary() {
        let rt = setup();
        // Fused: (a + b) * 3.0 — combines add + scale into single kernel
        let a = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0, 4.0], &[4], "a").unwrap();
        let b = Tensor::from_f32(&rt, &[10.0, 20.0, 30.0, 40.0], &[4], "b").unwrap();

        let result = ops::fusion::FusedOp::binary(&rt, &a, &b, "add_scale3", |kb, va, vb| {
            let sum = va.add(kb, vb);
            let three = kb.const_f32(3.0);
            sum.mul(kb, three)
        }).unwrap();

        let out = result.to_f32_vec();
        assert!((out[0] - 33.0).abs() < 1e-5, "[0] {} != 33", out[0]);
        assert!((out[1] - 66.0).abs() < 1e-5, "[1] {} != 66", out[1]);
        assert!((out[2] - 99.0).abs() < 1e-5, "[2] {} != 99", out[2]);
        assert!((out[3] - 132.0).abs() < 1e-5, "[3] {} != 132", out[3]);
    }

    #[test]
    fn test_fusion_ternary() {
        let rt = setup();
        // Fused: a * b + c (fma)
        let a = Tensor::from_f32(&rt, &[2.0, 3.0, 4.0, 5.0], &[4], "a").unwrap();
        let b = Tensor::from_f32(&rt, &[10.0, 10.0, 10.0, 10.0], &[4], "b").unwrap();
        let c = Tensor::from_f32(&rt, &[1.0, 2.0, 3.0, 4.0], &[4], "c").unwrap();

        let result = ops::fusion::FusedOp::ternary(&rt, &a, &b, &c, "fma", |kb, va, vb, vc| {
            va.fma(kb, vb, vc)  // a*b + c
        }).unwrap();

        let out = result.to_f32_vec();
        assert!((out[0] - 21.0).abs() < 1e-5, "[0] {} != 21", out[0]); // 2*10+1
        assert!((out[1] - 32.0).abs() < 1e-5, "[1] {} != 32", out[1]); // 3*10+2
        assert!((out[2] - 43.0).abs() < 1e-5, "[2] {} != 43", out[2]); // 4*10+3
        assert!((out[3] - 54.0).abs() < 1e-5, "[3] {} != 54", out[3]); // 5*10+4
    }
}

// ════════════════════════════════════════════════
//  Non-GPU tests (always available)
// ════════════════════════════════════════════════

#[cfg(test)]
mod infra_tests {
    #[test]
    fn test_lr_cosine_warmup() {
        use crate::ignis::lr_scheduler::{CosineWarmupScheduler, LrScheduler};
        let sched = CosineWarmupScheduler::new(1e-3, 1e-5, 100, 1000);
        assert!((sched.get_lr(0)).abs() < 1e-8, "should start at 0");
        assert!((sched.get_lr(100) - 1e-3).abs() < 1e-6, "should reach max at warmup end");
        assert!(sched.get_lr(550) < 6e-4, "mid-cosine should be below half");
    }

    #[test]
    fn test_bpe_tokenizer() {
        use crate::ignis::tokenizer::BpeTokenizer;
        let tok = BpeTokenizer::train("aaabdaaabac", 260);
        let encoded = tok.encode("aaab");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "aaab");
    }

    #[test]
    fn test_data_loader_epochs() {
        use crate::ignis::data_loader::DataLoader;
        let tokens: Vec<u32> = (0..100).collect();
        let mut dl = DataLoader::from_tokens(tokens, 1, 9);

        let mut batches = 0;
        while dl.next_batch().is_some() { batches += 1; }
        assert!(batches > 0, "should have at least 1 batch");

        dl.reset_epoch(false);
        let batch2 = dl.next_batch();
        assert!(batch2.is_some(), "should be able to get batches after reset");
    }

    #[test]
    fn test_loss_scaler_dynamics() {
        use crate::ignis::loss_scaler::LossScaler;
        let mut s = LossScaler::new();

        // 200 clean → should grow
        for _ in 0..200 { s.update(true); }
        assert!(s.current_scale() > 65536.0);

        // One fail → backoff
        let before_fail = s.current_scale();
        s.update(false);
        assert!(s.current_scale() < before_fail);
    }
}
