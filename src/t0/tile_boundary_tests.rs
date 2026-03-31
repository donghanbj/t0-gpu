//! Tile GEMM 边界测试 — 系统性验证所有 tile 配置的 M/N boundary 处理
//!
//! 对每种 tile 配置，测试 M < tile_M 的情况：
//! 1. 输出 buffer 预填充标记值 (SENTINEL)
//! 2. 执行 GEMM（kernel 使用 N 作为 stride）
//! 3. 检查有效区域精度 + M 溢出行的标记值未被覆盖
//!
//! 运行:
//!   cargo test --release --lib --features rocm -- test_tile_boundary --ignored --nocapture --test-threads=1

#[cfg(all(test, feature = "rocm"))]
mod boundary_tests {
    use crate::ignis::gpu_context::GpuRuntime;
    use crate::t0::tile_ir::{self, TileGemm};

    const SENTINEL: f32 = -9999.0;

    fn bf16_to_f32(v: u16) -> f32 {
        f32::from_bits((v as u32) << 16)
    }

    fn f32_to_bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// CPU reference GEMM NT: Y[m,n] = X[m,k] @ WT[n,k]^T
    fn cpu_gemm_nt_bf16(x: &[u16], wt: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += bf16_to_f32(x[i * k + kk]) * bf16_to_f32(wt[j * k + kk]);
                }
                y[i * n + j] = acc;
            }
        }
        y
    }

    /// 简单 LCG 随机 BF16 数据
    fn rand_bf16(n: usize, seed: u64) -> Vec<u16> {
        let mut rng = seed;
        (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            f32_to_bf16(f * 0.5)
        }).collect()
    }

    struct TestCase {
        name: &'static str,
        spec_fn: fn() -> TileGemm,
        m: u32,
        k: u32,
        n: u32,
    }

    /// Run a single boundary test case
    fn run_case(rt: &GpuRuntime, case: &TestCase) -> Result<(), String> {
        let spec = (case.spec_fn)();
        let m = case.m as usize;
        let k = case.k as usize;
        let n = case.n as usize;

        // GEMM kernel writes Y with stride=N (the kernel param N, not padded).
        // We allocate extra rows beyond M to detect OOB writes.
        let padded_m = ((m as u32 + spec.tile_m - 1) / spec.tile_m * spec.tile_m) as usize;
        let total_elems = padded_m * n;

        // Generate random inputs
        let x_data = rand_bf16(m * k, 42 + case.m as u64);
        let wt_data = rand_bf16(n * k, 137 + case.n as u64);
        let ref_y = cpu_gemm_nt_bf16(&x_data, &wt_data, m, k, n);

        // Upload
        let x_bytes: Vec<u8> = x_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wt_bytes: Vec<u8> = wt_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_buf = rt.alloc(x_bytes.len()).map_err(|e| format!("alloc X: {}", e))?;
        x_buf.write(&x_bytes);
        let wt_buf = rt.alloc(wt_bytes.len()).map_err(|e| format!("alloc WT: {}", e))?;
        wt_buf.write(&wt_bytes);

        // Output buffer: padded rows, fill with sentinel
        let y_buf = rt.alloc_f32(total_elems).map_err(|e| format!("alloc Y: {}", e))?;
        let sentinel_bytes: Vec<u8> = vec![SENTINEL; total_elems]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        y_buf.write(&sentinel_bytes);

        // Compile kernel
        let kernel_name = format!("boundary_{}", spec.name());
        let kernel = rt.ensure_kernel_t0(
            &kernel_name,
            || tile_ir::lower_gemm(&spec),
            [spec.wg_size(), 1, 1],
            spec.lds_total(),
        ).map_err(|e| format!("compile: {}", e))?;

        // Dispatch — pass actual M, not padded M
        let ka = tile_ir::build_kernargs_m(
            x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
            case.k, case.n, case.m, &spec,
        );
        // Grid covers padded dimensions (ceil to tile size)
        let grid = tile_ir::compute_grid(&spec, case.m, case.n);
        rt.dispatch(&kernel, grid, &ka).map_err(|e| format!("dispatch: {}", e))?;

        // Read back
        let gpu_y = rt.read_f32(&y_buf, total_elems);

        // Check 1: Valid region [0..m, 0..n] — accuracy
        let mut max_err = 0.0f64;
        let mut n_bad = 0usize;
        let mut first_bad = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let gpu_val = gpu_y[i * n + j];
                let ref_val = ref_y[i * n + j];
                let err = (gpu_val as f64 - ref_val as f64).abs();
                if err > max_err { max_err = err; }
                if err > 2.0 {
                    n_bad += 1;
                    if first_bad.len() < 5 {
                        first_bad.push(format!("[{},{}] gpu={:.4} ref={:.4} Δ={:.4e}", i, j, gpu_val, ref_val, err));
                    }
                }
            }
        }

        // Check 2: OOB rows [m..padded_m] — should still be sentinel
        let mut n_oob = 0usize;
        for i in m..padded_m {
            for j in 0..n {
                if gpu_y[i * n + j] != SENTINEL {
                    n_oob += 1;
                }
            }
        }

        if n_bad > 0 {
            let diag = first_bad.join(", ");
            return Err(format!("valid region: {} bad values (max_err={:.4e}) first: {}", n_bad, max_err, diag));
        }
        if n_oob > 0 {
            return Err(format!("OOB writes: {} values in rows [{},{})", n_oob, m, padded_m));
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_tile_boundary_all_configs() {
        let rt = GpuRuntime::new().expect("GPU init failed");

        eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
        eprintln!("║  Tile GEMM M 边界测试 — M < tile_M 系统性验证                    ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

        let cases = vec![
            // ── M < tile_M tests ──
            TestCase { name: "128x64_k16  M=64",  spec_fn: TileGemm::tile_128x64_k16, m: 64, k: 256, n: 64 },
            TestCase { name: "128x64_k16  M=96",  spec_fn: TileGemm::tile_128x64_k16, m: 96, k: 256, n: 64 },
            TestCase { name: "128x64_k32  M=64",  spec_fn: TileGemm::tile_128x64_k32, m: 64, k: 256, n: 64 },
            TestCase { name: "128x64_k64  M=64",  spec_fn: TileGemm::tile_128x64_k64, m: 64, k: 256, n: 64 },
            TestCase { name: "64x64_k16   M=32",  spec_fn: TileGemm::tile_64x64_k16,  m: 32, k: 256, n: 64 },
            TestCase { name: "64x64_k32   M=32",  spec_fn: TileGemm::tile_64x64_k32,  m: 32, k: 256, n: 64 },
            TestCase { name: "64x64_k64   M=32",  spec_fn: TileGemm::tile_64x64_k64,  m: 32, k: 256, n: 64 },
            TestCase { name: "32x64_k16   M=16",  spec_fn: TileGemm::tile_32x64_k16,  m: 16, k: 256, n: 64 },
            TestCase { name: "128x128_k16 M=64",  spec_fn: TileGemm::tile_128x128_k16, m: 64, k: 256, n: 128 },
            TestCase { name: "128x128_k32 M=64",  spec_fn: TileGemm::tile_128x128_k32, m: 64, k: 256, n: 128 },
            TestCase { name: "64x128_k16  M=32",  spec_fn: TileGemm::tile_64x128_k16, m: 32, k: 256, n: 128 },
            TestCase { name: "64x128_k32  M=32",  spec_fn: TileGemm::tile_64x128_k32, m: 32, k: 256, n: 128 },

            // ── M = tile_M (no boundary, regression check) ──
            TestCase { name: "128x64_k16  M=128", spec_fn: TileGemm::tile_128x64_k16, m: 128, k: 256, n: 64 },
            TestCase { name: "64x64_k16   M=64",  spec_fn: TileGemm::tile_64x64_k16,  m: 64, k: 256, n: 64 },

            // ── acc_swap tiles (TODO: need separate boundary fix in emit_store_phase_swap) ──
            // TestCase { name: "128x128_k16_sw M=64", spec_fn: TileGemm::tile_128x128_k16_swap, m: 64, k: 256, n: 128 },
            // TestCase { name: "128x128_k32_sw M=64", spec_fn: TileGemm::tile_128x128_k32_swap, m: 64, k: 256, n: 128 },

            // ── M=1 tile row (extreme) ──
            TestCase { name: "128x64_k16  M=16",  spec_fn: TileGemm::tile_128x64_k16, m: 16, k: 256, n: 64 },
            TestCase { name: "128x128_k16 M=16",  spec_fn: TileGemm::tile_128x128_k16, m: 16, k: 256, n: 128 },
        ];

        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut errors: Vec<String> = Vec::new();

        for case in &cases {
            match run_case(&rt, case) {
                Ok(()) => {
                    eprintln!("  ✅ {}", case.name);
                    passed += 1;
                }
                Err(e) => {
                    eprintln!("  ❌ {} — {}", case.name, e);
                    failed += 1;
                    errors.push(format!("{}: {}", case.name, e));
                }
            }
        }

        eprintln!("\n  ════════════════════════════════════════════════════");
        eprintln!("  结果: {} passed, {} failed (total {})", passed, failed, cases.len());
        if !errors.is_empty() {
            eprintln!("  失败详情:");
            for e in &errors { eprintln!("    ❌ {}", e); }
        }
        eprintln!("  ════════════════════════════════════════════════════\n");

        assert_eq!(failed, 0, "Tile boundary tests failed");
    }
}
