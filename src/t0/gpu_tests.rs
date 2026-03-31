//! T0 GPU 正确性测试
//!
//! 测试 T0 编译器生成的 GPU 内核在 RX 7900 XTX 上的运行正确性。
//! 所有 GPU 测试通过统一的 GpuRuntime 访问 GPU（不再使用 T0Context）。
//!
//! 运行: cargo test --features rocm -- t0_gpu_tests --test-threads=1

#[cfg(all(test, feature = "rocm"))]
mod t0_gpu_tests {
    use crate::t0::compile::T0Kernel;
    use crate::t0::ir::*;
    use crate::ignis::gpu_context::GpuRuntime;
    use std::sync::{Arc, OnceLock};

    // ── Shared GpuRuntime (KFD singleton) ──
    // GpuRuntime is !Sync (DispatchPool uses RefCell), but with --test-threads=1
    // only one test runs at a time. Use an unsafe Sync wrapper.
    struct SyncRt(Arc<GpuRuntime>);
    unsafe impl Sync for SyncRt {}
    unsafe impl Send for SyncRt {}

    static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

    fn with_rt<F, R>(f: F) -> R
    where F: FnOnce(&GpuRuntime) -> R {
        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
        });
        // Pre-sync: ensure queue is idle from any previous test
        let _ = rt.0.wait_idle();
        let result = f(&rt.0);
        // Post-sync: ensure all GPU work completes before next test
        let _ = rt.0.wait_idle();
        result
    }

    // ═══════════════════════════════════════════
    //  Test 1: T0Kernel compile + assembly smoke
    // ═══════════════════════════════════════════

    #[test]
    fn test_t0kernel_compile_smoke() {
        let mut k = T0Kernel::new("smoke_test");
        let ptr = k.arg_ptr("data");
        let n = k.arg_u32("n");
        k.emit_arg_loads();

        let gid = k.compute_global_id_x(256);
        let n_vreg = k.alloc_vreg();
        k.v_mov_from_sgpr(n_vreg, n);
        let saved = k.bounds_check_begin(gid, n_vreg);

        let offset = k.alloc_vreg();
        k.v_lshlrev_b32(offset, 2, gid);

        let (addr_lo, addr_hi) = k.alloc_addr_pair();
        let val = k.alloc_vreg();
        k.v_mov_from_sgpr(addr_lo, SReg(ptr.0));
        k.v_mov_from_sgpr(addr_hi, SReg(ptr.0 + 1));
        k.addr64_add(addr_lo, addr_hi, offset);
        k.global_load(val, addr_lo, Width::B32, 0);
        k.wait_vmcnt(0);

        let one = k.alloc_vreg();
        k.v_mov_imm(one, 1.0f32.to_bits() as i32);
        k.v_add_f32(val, val, one);
        k.global_store(addr_lo, val, Width::B32, 0);

        k.bounds_check_end(saved);
        k.endpgm();

        // Verify assembly
        let asm = k.to_assembly(Target::GFX1100).expect("to_assembly failed");
        assert!(asm.contains("s_endpgm"), "Missing s_endpgm");
        assert!(asm.contains("global_load"), "Missing global_load");

        // Verify ELF
        let elf = k.compile(Target::GFX1100).expect("ELF compile failed");
        assert!(elf.len() > 100, "ELF too small: {} bytes", elf.len());

        // Verify kernarg_size auto-tracking
        assert_eq!(k.kernarg_size(), 12, "kernarg_size should be 8 (ptr) + 4 (u32) = 12");

        eprintln!("[PASS] test_t0kernel_compile_smoke: asm={} bytes, ELF={} bytes, ka={}", asm.len(), elf.len(), k.kernarg_size());
    }

    // ═══════════════════════════════════════════
    //  Test 2: validate pass — missing endpgm
    // ═══════════════════════════════════════════

    #[test]
    fn test_validate_missing_endpgm() {
        let mut k = T0Kernel::new("no_endpgm");
        k.arg_u32("n");
        k.emit_arg_loads();

        let result = k.to_assembly(Target::GFX1100);
        assert!(result.is_err(), "Should fail validation without endpgm");
        let err = result.unwrap_err();
        assert!(err.contains("missing s_endpgm"), "Error should mention endpgm: {}", err);
        eprintln!("[PASS] test_validate_missing_endpgm: correctly caught");
    }

    // ═══════════════════════════════════════════
    //  Test 3: validate pass — unbalanced exec mask
    // ═══════════════════════════════════════════

    #[test]
    fn test_validate_unbalanced_exec() {
        let mut k = T0Kernel::new("bad_exec");
        let n = k.arg_u32("n");
        k.emit_arg_loads();
        let gid = k.compute_global_id_x(32);
        let nv = k.alloc_vreg();
        k.v_mov_from_sgpr(nv, n);
        let _saved = k.bounds_check_begin(gid, nv);
        k.endpgm();

        let result = k.to_assembly(Target::GFX1100);
        assert!(result.is_err(), "Should fail validation with unbalanced exec");
        let err = result.unwrap_err();
        assert!(err.contains("SaveExec"), "Error should mention SaveExec: {}", err);
        eprintln!("[PASS] test_validate_unbalanced_exec: correctly caught");
    }

    // ═══════════════════════════════════════════
    //  Test 4: DSL lower — Scale
    // ═══════════════════════════════════════════

    #[test]
    fn test_gpu_row_reduce_sum() {
        with_rt(|rt| {
            for &(rows, cols) in &[(1usize, 32usize), (1, 1), (4, 128), (2, 33)] {
                let data: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.01 - 1.0)).collect();
                eprintln!("  testing {}x{}, first 8 elems: {:?}", rows, cols, &data[..8.min(data.len())]);

                let in_buf = rt.upload_f32(&data).unwrap();
                let out_buf = rt.alloc_f32(rows).unwrap();

                let kernel = rt.ensure_kernel_t0(
                    "row_reduce_sum",
                    || crate::t0::math::t0_row_reduce_sum(),
                    [32, 1, 1],
                    0,
                ).unwrap();

                let ka = crate::kernargs![
                    in_buf.gpu_addr() => u64,
                    out_buf.gpu_addr() => u64,
                    cols as u32 => u32
                ];
                rt.dispatch(&kernel, [rows as u32 * 32, 1, 1], &ka).unwrap();

                let result = rt.read_f32(&out_buf, rows);

                // CPU reference
                for r in 0..rows {
                    let expected: f32 = data[r * cols..(r + 1) * cols].iter().sum();
                    let tol = expected.abs() * 1e-4 + 1e-4;
                    assert!(
                        (result[r] - expected).abs() < tol,
                        "row_sum mismatch at row {}: expected {}, got {} (rows={}, cols={})",
                        r, expected, result[r], rows, cols,
                    );
                }
                eprintln!("[PASS] test_gpu_row_reduce_sum: {}x{} verified", rows, cols);
            }
        });
    }

    #[test]
    fn test_gpu_row_reduce_max() {
        with_rt(|rt| {
            for &(rows, cols) in &[(4usize, 128usize), (8, 1024), (2, 33), (1, 7)] {
                let data: Vec<f32> = (0..rows * cols).map(|i| {
                    ((i as f32) * 0.037 - 5.0).sin()  // values in [-1, 1]
                }).collect();

                let in_buf = rt.upload_f32(&data).unwrap();
                let out_buf = rt.alloc_f32(rows).unwrap();

                let kernel = rt.ensure_kernel_t0(
                    "row_reduce_max",
                    || crate::t0::math::t0_row_reduce_max(),
                    [32, 1, 1],
                    0,
                ).unwrap();

                let ka = crate::kernargs![
                    in_buf.gpu_addr() => u64,
                    out_buf.gpu_addr() => u64,
                    cols as u32 => u32
                ];
                rt.dispatch(&kernel, [rows as u32 * 32, 1, 1], &ka).unwrap();

                let result = rt.read_f32(&out_buf, rows);

                for r in 0..rows {
                    let expected = data[r * cols..(r + 1) * cols]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let tol = 1e-6;
                    assert!(
                        (result[r] - expected).abs() < tol,
                        "row_max mismatch at row {}: expected {}, got {} (rows={}, cols={})",
                        r, expected, result[r], rows, cols,
                    );
                }
                eprintln!("[PASS] test_gpu_row_reduce_max: {}x{} verified", rows, cols);
            }
        });
    }

    // ═══════════════════════════════════════════
    //  Row-wise broadcast ops GPU tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_gpu_row_broadcast_sub() {
        with_rt(|rt| {
            let rows = 2usize;
            let cols = 64usize;
            let x_data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
            let vec_data: Vec<f32> = vec![1.0, 2.0]; // subtract 1.0 from row 0, 2.0 from row 1

            let x_buf = rt.upload_f32(&x_data).unwrap();
            let vec_buf = rt.upload_f32(&vec_data).unwrap();
            let out_buf = rt.alloc_f32(rows * cols).unwrap();

            let kernel = rt.ensure_kernel_t0("bcast_sub_test",
                || crate::t0::math::t0_row_broadcast_sub(), [32, 1, 1], 0).unwrap();

            let ka = crate::kernargs![
                x_buf.gpu_addr() => u64, vec_buf.gpu_addr() => u64,
                out_buf.gpu_addr() => u64, cols as u32 => u32
            ];
            rt.dispatch(&kernel, [rows as u32 * 32, 1, 1], &ka).unwrap();

            let result = rt.read_f32(&out_buf, rows * cols);

            for r in 0..rows {
                for c in 0..cols {
                    let expected = x_data[r * cols + c] - vec_data[r];
                    let got = result[r * cols + c];
                    assert!(
                        (got - expected).abs() < 1e-4,
                        "bcast_sub mismatch [{},{}]: expected {}, got {}", r, c, expected, got,
                    );
                }
            }
            eprintln!("[PASS] test_gpu_row_broadcast_sub: {}x{} verified", rows, cols);
        });
    }

    // ═══════════════════════════════════════════

    #[test]
    fn test_gpu_softmax() {
        with_rt(|rt| {
            for &(rows, cols) in &[(2usize, 64usize), (4, 128), (1, 33), (8, 256)] {
                // Random-ish input data
                let data: Vec<f32> = (0..rows * cols).map(|i| {
                    ((i as f32) * 0.037 - 3.0).sin() * 2.0
                }).collect();

                let x_buf = rt.upload_f32(&data).unwrap();
                let max_buf = rt.alloc_f32(rows).unwrap();
                let exp_buf = rt.alloc_f32(rows * cols).unwrap();
                let sum_buf = rt.alloc_f32(rows).unwrap();
                let out_buf = rt.alloc_f32(rows * cols).unwrap();
                let grid = rows as u32 * 32;

                // Step 1: row_max
                let k_max = rt.ensure_kernel_t0("softmax_row_max",
                    || crate::t0::math::t0_row_reduce_max(), [32, 1, 1], 0).unwrap();
                let ka1 = crate::kernargs![
                    x_buf.gpu_addr() => u64, max_buf.gpu_addr() => u64,
                    cols as u32 => u32
                ];
                rt.dispatch(&k_max, [grid, 1, 1], &ka1).unwrap();

                // Step 2: exp(x - max)  [fused exp_sub]
                let k_exp = rt.ensure_kernel_t0("softmax_exp_sub",
                    || crate::t0::math::t0_row_broadcast_exp_sub(), [32, 1, 1], 0).unwrap();
                let ka2 = crate::kernargs![
                    x_buf.gpu_addr() => u64, max_buf.gpu_addr() => u64,
                    exp_buf.gpu_addr() => u64, cols as u32 => u32
                ];
                rt.dispatch(&k_exp, [grid, 1, 1], &ka2).unwrap();

                // Step 3: row_sum(exp)
                let k_sum = rt.ensure_kernel_t0("softmax_row_sum",
                    || crate::t0::math::t0_row_reduce_sum(), [32, 1, 1], 0).unwrap();
                let ka3 = crate::kernargs![
                    exp_buf.gpu_addr() => u64, sum_buf.gpu_addr() => u64,
                    cols as u32 => u32
                ];
                rt.dispatch(&k_sum, [grid, 1, 1], &ka3).unwrap();

                // Step 4: out = exp / sum  [broadcast div]
                let k_div = rt.ensure_kernel_t0("softmax_bcast_div",
                    || crate::t0::math::t0_row_broadcast_div(), [32, 1, 1], 0).unwrap();
                let ka4 = crate::kernargs![
                    exp_buf.gpu_addr() => u64, sum_buf.gpu_addr() => u64,
                    out_buf.gpu_addr() => u64, cols as u32 => u32
                ];
                rt.dispatch(&k_div, [grid, 1, 1], &ka4).unwrap();

                // Read result
                let result = rt.read_f32(&out_buf, rows * cols);

                // CPU reference softmax
                let mut max_err = 0f32;
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let row_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_vals: Vec<f32> = row.iter().map(|&x| (x - row_max).exp()).collect();
                    let exp_sum: f32 = exp_vals.iter().sum();
                    for c in 0..cols {
                        let expected = exp_vals[c] / exp_sum;
                        let got = result[r * cols + c];
                        let err = (got - expected).abs();
                        max_err = max_err.max(err);
                        assert!(
                            err < 1e-3,
                            "softmax mismatch at [{},{}]: expected {:.6}, got {:.6} ({}x{})",
                            r, c, expected, got, rows, cols,
                        );
                    }
                    // Verify row sums to ~1.0
                    let row_sum: f32 = result[r * cols..(r + 1) * cols].iter().sum();
                    assert!(
                        (row_sum - 1.0).abs() < 1e-3,
                        "softmax row {} sum = {} (expected 1.0), {}x{}", r, row_sum, rows, cols,
                    );
                }
                eprintln!("[PASS] test_gpu_softmax: {}x{} verified (max_err={:.6})", rows, cols, max_err);
            }
        });
    }
}
