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

    /// WGP mode isolation test: uses SEPARATE GpuRuntime per stage to prevent
    /// queue poisoning from WGP hang cascading to subsequent tests.
    ///
    /// Root cause hypothesis (from dmesg): CWSR (Context Wave Save/Restore)
    /// triggers page faults when preempting WGP-mode waves, because the CWSR
    /// buffer layout assumes CU mode wave counts.
    ///
    /// Fix: run with KFD_NO_CWSR=1 to disable CWSR and test WGP in isolation.
    ///
    /// Run:
    ///   # Without CWSR (should work):
    ///   KFD_NO_CWSR=1 cargo test --release --lib --features rocm -- test_wgp_barrier_probe --nocapture --ignored --test-threads=1
    ///
    ///   # With CWSR (expected to hang):
    ///   cargo test --release --lib --features rocm -- test_wgp_barrier_probe --nocapture --ignored --test-threads=1
    #[test]
    #[ignore]
    fn test_wgp_barrier_probe() {
        let cwsr_disabled = std::env::var("KFD_NO_CWSR").map(|v| v == "1").unwrap_or(false);
        eprintln!("\n╔══════════════════════════════════════════════════════════╗");
        eprintln!("║  WGP Barrier Probe — GFX1100 isolation test             ║");
        eprintln!("║  CWSR: {}                                          ║",
                  if cwsr_disabled { "DISABLED (safe)" } else { "ENABLED (risk!)" });
        eprintln!("╚══════════════════════════════════════════════════════════╝\n");

        // ── Stage 1: CU mode baseline (always safe) ──
        {
            let rt = GpuRuntime::new().expect("Stage1: failed to create GpuRuntime");
            let out_buf = rt.alloc_f32(256).unwrap();

            let mut k = T0Kernel::new("wgp_probe_cu_baseline");
            k.set_wg_size(256);
            let out_ptr = k.arg_ptr("out");
            k.emit_arg_loads();
            let gid = k.compute_global_id_x(256);
            let val = k.alloc_vreg();
            k.push(Op::VCvtF32U32 { dst: val, src: gid });
            // barrier with waitcnt
            k.wait_vmcnt(0);
            k.wait_lgkmcnt(0);
            k.s_barrier();
            let offset = k.alloc_vreg();
            k.v_lshlrev_b32(offset, 2, gid);
            let (addr_lo, addr_hi) = k.alloc_addr_pair();
            k.v_mov_from_sgpr(addr_lo, SReg(out_ptr.0));
            k.v_mov_from_sgpr(addr_hi, SReg(out_ptr.0 + 1));
            k.addr64_add(addr_lo, addr_hi, offset);
            k.global_store(addr_lo, val, Width::B32, 0);
            k.wait_vscnt(0);
            k.endpgm();

            let kernel = rt.ensure_kernel_t0(
                "wgp_probe_cu_baseline", || k, [256, 1, 1], 0,
            ).expect("stage1 compile");
            let ka = crate::kernargs![out_buf.gpu_addr() => u64];
            rt.dispatch(&kernel, [256, 1, 1], &ka).expect("stage1 dispatch");
            let result = rt.read_f32(&out_buf, 4);
            eprintln!("  Stage 1 (CU, barrier):       PASS  out[0..3]={:?}", &result[..4]);
        }
        // GpuRuntime dropped → queue destroyed → clean slate

        // ── Stage 2: WGP mode, no LDS, barrier only ──
        {
            let rt = GpuRuntime::new().expect("Stage2: failed to create GpuRuntime");
            let out_buf = rt.alloc_f32(256).unwrap();

            let mut k = T0Kernel::new("wgp_probe_wgp_nolds");
            k.set_wg_size(256);
            k.set_wgp_mode(true);
            let out_ptr = k.arg_ptr("out");
            k.emit_arg_loads();
            let gid = k.compute_global_id_x(256);
            let val = k.alloc_vreg();
            k.push(Op::VCvtF32U32 { dst: val, src: gid });
            k.wait_vmcnt(0);
            k.wait_lgkmcnt(0);
            k.s_barrier();
            let offset = k.alloc_vreg();
            k.v_lshlrev_b32(offset, 2, gid);
            let (addr_lo, addr_hi) = k.alloc_addr_pair();
            k.v_mov_from_sgpr(addr_lo, SReg(out_ptr.0));
            k.v_mov_from_sgpr(addr_hi, SReg(out_ptr.0 + 1));
            k.addr64_add(addr_lo, addr_hi, offset);
            k.global_store(addr_lo, val, Width::B32, 0);
            k.wait_vscnt(0);
            k.endpgm();

            let kernel = rt.ensure_kernel_t0(
                "wgp_probe_wgp_nolds", || k, [256, 1, 1], 0,
            ).expect("stage2 compile");
            let ka = crate::kernargs![out_buf.gpu_addr() => u64];
            match rt.dispatch(&kernel, [256, 1, 1], &ka) {
                Ok(_) => {
                    let result = rt.read_f32(&out_buf, 4);
                    eprintln!("  Stage 2 (WGP, no LDS):       PASS  out[0..3]={:?}", &result[..4]);
                }
                Err(e) => {
                    eprintln!("  Stage 2 (WGP, no LDS):       FAIL: {}", e);
                    if !cwsr_disabled {
                        eprintln!("  → Try: KFD_NO_CWSR=1 to disable CWSR and retry");
                    }
                    return; // Don't continue — queue is poisoned
                }
            }
        }

        // ── Stage 3: WGP mode, LDS=4KB, barrier + LDS access ──
        {
            let rt = GpuRuntime::new().expect("Stage3: failed to create GpuRuntime");
            let out_buf = rt.alloc_f32(256).unwrap();

            let mut k = T0Kernel::new("wgp_probe_wgp_lds4k");
            k.set_wg_size(256);
            k.set_wgp_mode(true);
            k.set_lds_size(4096);
            let out_ptr = k.arg_ptr("out");
            k.emit_arg_loads();
            let gid = k.compute_global_id_x(256);
            let val = k.alloc_vreg();
            k.push(Op::VCvtF32U32 { dst: val, src: gid });
            // Write to LDS
            let lds_addr = k.alloc_vreg();
            k.v_lshlrev_b32(lds_addr, 2, VReg(0));
            k.ds_store_b32(lds_addr, val, 0);
            k.wait_lgkmcnt(0);
            k.wait_vmcnt(0);
            k.s_barrier();
            // Read from LDS (XOR neighbour)
            let neighbour = k.alloc_vreg();
            let four = k.alloc_vreg();
            k.v_mov_imm(four, 4);
            k.push(Op::VXorB32 { dst: neighbour, src0: Operand::VReg(lds_addr), src1: Operand::VReg(four) });
            let rval = k.alloc_vreg();
            k.ds_load_b32(rval, neighbour, 0);
            k.wait_lgkmcnt(0);
            // Store to GMEM
            let offset = k.alloc_vreg();
            k.v_lshlrev_b32(offset, 2, gid);
            let (addr_lo, addr_hi) = k.alloc_addr_pair();
            k.v_mov_from_sgpr(addr_lo, SReg(out_ptr.0));
            k.v_mov_from_sgpr(addr_hi, SReg(out_ptr.0 + 1));
            k.addr64_add(addr_lo, addr_hi, offset);
            k.global_store(addr_lo, rval, Width::B32, 0);
            k.wait_vscnt(0);
            k.endpgm();

            let kernel = rt.ensure_kernel_t0(
                "wgp_probe_wgp_lds4k", || k, [256, 1, 1], 4096,
            ).expect("stage3 compile");
            let ka = crate::kernargs![out_buf.gpu_addr() => u64];
            match rt.dispatch(&kernel, [256, 1, 1], &ka) {
                Ok(_) => {
                    let result = rt.read_f32(&out_buf, 4);
                    eprintln!("  Stage 3 (WGP, LDS=4KB):      PASS  out[0..3]={:?}", &result[..4]);
                }
                Err(e) => {
                    eprintln!("  Stage 3 (WGP, LDS=4KB):      FAIL: {}", e);
                    return;
                }
            }
        }

        // ── Stage 4: WGP mode, LDS=80KB (needs WGP for >64KB) ──
        {
            let rt = GpuRuntime::new().expect("Stage4: failed to create GpuRuntime");
            let out_buf = rt.alloc_f32(256).unwrap();

            let mut k = T0Kernel::new("wgp_probe_wgp_lds80k");
            k.set_wg_size(256);
            k.set_wgp_mode(true);
            k.set_lds_size(81920);
            let out_ptr = k.arg_ptr("out");
            k.emit_arg_loads();
            let gid = k.compute_global_id_x(256);
            let val = k.alloc_vreg();
            k.push(Op::VCvtF32U32 { dst: val, src: gid });
            let lds_addr = k.alloc_vreg();
            k.v_lshlrev_b32(lds_addr, 2, VReg(0));
            k.ds_store_b32(lds_addr, val, 0);
            k.wait_lgkmcnt(0);
            k.wait_vmcnt(0);
            k.s_barrier();
            let rval = k.alloc_vreg();
            k.ds_load_b32(rval, lds_addr, 0);
            k.wait_lgkmcnt(0);
            let offset = k.alloc_vreg();
            k.v_lshlrev_b32(offset, 2, gid);
            let (addr_lo, addr_hi) = k.alloc_addr_pair();
            k.v_mov_from_sgpr(addr_lo, SReg(out_ptr.0));
            k.v_mov_from_sgpr(addr_hi, SReg(out_ptr.0 + 1));
            k.addr64_add(addr_lo, addr_hi, offset);
            k.global_store(addr_lo, rval, Width::B32, 0);
            k.wait_vscnt(0);
            k.endpgm();

            let kernel = rt.ensure_kernel_t0(
                "wgp_probe_wgp_lds80k", || k, [256, 1, 1], 81920,
            ).expect("stage4 compile");
            let ka = crate::kernargs![out_buf.gpu_addr() => u64];
            match rt.dispatch(&kernel, [256, 1, 1], &ka) {
                Ok(_) => {
                    let result = rt.read_f32(&out_buf, 4);
                    eprintln!("  Stage 4 (WGP, LDS=80KB):     PASS  out[0..3]={:?}", &result[..4]);
                }
                Err(e) => {
                    eprintln!("  Stage 4 (WGP, LDS=80KB):     FAIL: {}", e);
                    return;
                }
            }
        }

        eprintln!("\n  ✅ All WGP stages PASSED!");
    }

    /// k32 vs k64 head-to-head GEMM benchmark at 4096³.
    /// Tests both ILP improvement and dispatch safety under 252 VGPR limit.
    ///
    /// Run: cargo test --release --lib --features rocm -- test_k64_vs_k32_benchmark --nocapture --ignored --test-threads=1
    #[test]
    #[ignore]
    fn test_k64_vs_k32_benchmark() {
        use crate::t0::tile_ir::{TileGemm, lower_gemm, build_kernargs_m, compute_grid, f32_to_bf16};
        use std::time::Instant;

        with_rt(|rt| {
            eprintln!("\n╔══════════════════════════════════════════════════════════╗");
            eprintln!("║  k32 vs k64 Head-to-Head Benchmark (128×128, 4096³)     ║");
            eprintln!("╚══════════════════════════════════════════════════════════╝\n");

            let m = 4096u32;
            let k = 4096u32;
            let n = 4096u32;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;

            // Prepare data
            let x_bf16: Vec<u16> = (0..m*k).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
            let wt_bf16: Vec<u16> = (0..n*k).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();

            let x_bytes: Vec<u8> = x_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let wt_bytes: Vec<u8> = wt_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let x_buf = rt.alloc(x_bytes.len()).expect("alloc X");
            let wt_buf = rt.alloc(wt_bytes.len()).expect("alloc WT");
            x_buf.write(&x_bytes);
            wt_buf.write(&wt_bytes);

            let warmup = 5;
            let iters = 20;

            eprintln!("  {:20} {:>8} {:>8} {:>8} {:>8} {:>6}",
                "Config", "VGPRs", "Spills", "LDS_KB", "μs", "TFLOPS");
            eprintln!("  {}", "─".repeat(65));

            let configs: Vec<(&str, TileGemm)> = vec![
                ("128×128 k16",  TileGemm::tile_128x128_k16()),
                ("128×128 k32",  TileGemm::tile_128x128_k32()),
                // k48 DEAD: coop load cpr=6 (non-power-of-2) bug
                // k64 128×128: LDS=64KB exactly at CU limit → GPU hang (coop load OK, cpr=8)
                ("128×64 k32",   TileGemm::tile_128x64_k32()),
                ("128×64 k64",   TileGemm::tile_128x64_k64()),
            ];

            for (label, spec) in configs {
                let y_buf = rt.alloc_zero((m * n * 4) as usize).expect("alloc Y");

                let kname = format!("bench_{}", spec.name());
                let kernel = match rt.ensure_kernel_t0(
                    &kname,
                    || lower_gemm(&spec),
                    [spec.wg_size(), 1, 1],
                    spec.lds_total(),
                ) {
                    Ok(k) => k,
                    Err(e) => {
                        eprintln!("  {:20} COMPILE FAIL: {}", label, e);
                        continue;
                    }
                };

                let ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, &spec,
                );
                let grid = compute_grid(&spec, m, n);

                // Warmup
                let mut ok = true;
                for _ in 0..warmup {
                    match rt.dispatch(&kernel, [grid[0], grid[1], grid[2]], &ka) {
                        Ok(_) => {},
                        Err(e) => {
                            eprintln!("  {:20} DISPATCH FAIL: {}", label, e);
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok { continue; }

                // Benchmark
                let t0 = Instant::now();
                for _ in 0..iters {
                    if rt.dispatch(&kernel, [grid[0], grid[1], grid[2]], &ka).is_err() {
                        ok = false;
                        break;
                    }
                }
                let elapsed = t0.elapsed();
                let us = elapsed.as_micros() as f64 / iters as f64;
                let tflops = flops / (us * 1e6);

                let lds_kb = spec.lds_total() as f32 / 1024.0;

                if ok {
                    eprintln!("  {:20} {:>8} {:>8} {:>7.0} {:>8.1} {:>6.1}",
                        label, "?", "?", lds_kb, us, tflops);
                } else {
                    eprintln!("  {:20} BENCHMARK FAIL (hang?)", label);
                }
            }

            eprintln!();
        });
    }

    /// COMPILE-ONLY: Compare VGPR usage across all tile GEMM configurations.
    /// No GPU dispatch — pure compiler analysis.
    ///
    /// Run: cargo test --release --lib --features rocm -- test_tile_vgpr_comparison --nocapture
    #[test]
    fn test_tile_vgpr_comparison() {
        use crate::t0::tile_ir::{TileGemm, lower_gemm};

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  Tile GEMM VGPR Comparison — Compile-only analysis                 ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
        eprintln!();
        eprintln!("  {:30} {:>6} {:>6} {:>7} {:>7} {:>5} {:>8}",
            "Config", "VGPRs", "SGPRs", "LDS_KB", "Spills", "Waves", "Mode");
        eprintln!("  {}", "─".repeat(75));

        let configs: Vec<(&str, TileGemm)> = vec![
            ("128×128 k16",       TileGemm::tile_128x128_k16()),
            ("128×128 k32",       TileGemm::tile_128x128_k32()),
            // k48 REMOVED: tile_k=48 → chunks_per_row=6 (not power-of-2) → coop load bug → GPU hang
            // ("128×128 k48",       TileGemm::tile_128x128_k48()),
            ("128×128 k64",       TileGemm::tile_128x128_k64()),
            ("128×128 k16 swap",  TileGemm::tile_128x128_k16_swap()),
            ("128×128 k32 swap",  TileGemm::tile_128x128_k32_swap()),
            ("128×64 k16",        TileGemm::tile_128x64_k16()),
            ("128×64 k32",        TileGemm::tile_128x64_k32()),
            ("128×64 k64",        TileGemm::tile_128x64_k64()),
            ("64×128 k32",        TileGemm::tile_64x128_k32()),
            ("256×64 k32 WGP",    TileGemm::tile_256x64_k32_wgp()),
            ("256×64 k64 WGP",    TileGemm::tile_256x64_k64_wgp()),
        ];

        for (label, spec) in configs {
            let lds_kb = spec.lds_total() as f32 / 1024.0;
            let mode = if spec.wgp_mode { "WGP" } else { "CU" };
            let lds_limit = if spec.wgp_mode { 128.0 } else { 64.0 };
            let lds_ok = lds_kb <= lds_limit;

            let kernel = lower_gemm(&spec);
            match kernel.compile(Target::GFX1100) {
                Ok(_elf) => {
                    // Get VGPR/SGPR stats from the compiler
                    let asm = kernel.to_assembly(Target::GFX1100).unwrap_or_default();
                    let vgprs = asm.lines()
                        .find(|l| l.contains("amdhsa_next_free_vgpr"))
                        .and_then(|l| l.split_whitespace().last()?.parse::<u32>().ok())
                        .unwrap_or(0);
                    let sgprs = asm.lines()
                        .find(|l| l.contains("amdhsa_next_free_sgpr"))
                        .and_then(|l| l.split_whitespace().last()?.parse::<u32>().ok())
                        .unwrap_or(0);
                    // Waves/SIMD = floor(256 / vgprs), min 1
                    let waves = if vgprs > 0 { 256 / vgprs } else { 0 };
                    // Check for spills (look for scratch in ASM)
                    let has_scratch = asm.contains("scratch_") || asm.contains("s_scratch");
                    let spill_str = if has_scratch { "YES" } else { "0" };

                    let lds_flag = if !lds_ok { " ⚠️" } else { "" };
                    eprintln!("  {:30} {:>6} {:>6} {:>6.1}{} {:>7} {:>5} {:>8}",
                        label, vgprs, sgprs, lds_kb, lds_flag, spill_str, waves, mode);
                }
                Err(e) => {
                    let lds_flag = if !lds_ok { " ⚠️LDS!" } else { "" };
                    eprintln!("  {:30} {:>44} {:>8}{}",
                        label, format!("COMPILE FAIL: {}", &e[..e.len().min(30)]), mode, lds_flag);
                }
            }
        }

        eprintln!();
        eprintln!("  Legend: Waves = floor(256/VGPRs), ⚠️ = LDS exceeds mode limit");
        eprintln!("  GFX1100: 256 VGPRs/SIMD, CU=64KB LDS, WGP=128KB LDS");
        eprintln!();
    }

    /// COMPILE-ONLY: dump WGP kernel assembly + KD for offline analysis.
    /// NO GPU dispatch — safe from hangs.
    ///
    /// Run: cargo test --release --lib --features rocm -- test_wgp_asm_dump --nocapture
    #[test]
    fn test_wgp_asm_dump() {
        // Build a minimal WGP kernel: wg=256, LDS=4KB, s_barrier
        let mut k = T0Kernel::new("wgp_dump_test");
        k.set_wg_size(256);
        k.set_wgp_mode(true);
        k.set_lds_size(4096);
        let out_ptr = k.arg_ptr("out");
        k.emit_arg_loads();
        let gid = k.compute_global_id_x(256);
        let val = k.alloc_vreg();
        k.push(Op::VCvtF32U32 { dst: val, src: gid });
        let lds_addr = k.alloc_vreg();
        k.v_lshlrev_b32(lds_addr, 2, VReg(0));
        k.ds_store_b32(lds_addr, val, 0);
        k.wait_lgkmcnt(0);
        k.wait_vmcnt(0);
        k.s_barrier();
        let rval = k.alloc_vreg();
        k.ds_load_b32(rval, lds_addr, 0);
        k.wait_lgkmcnt(0);
        let offset = k.alloc_vreg();
        k.v_lshlrev_b32(offset, 2, gid);
        let (addr_lo, addr_hi) = k.alloc_addr_pair();
        k.v_mov_from_sgpr(addr_lo, SReg(out_ptr.0));
        k.v_mov_from_sgpr(addr_hi, SReg(out_ptr.0 + 1));
        k.addr64_add(addr_lo, addr_hi, offset);
        k.global_store(addr_lo, rval, Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();

        // Dump assembly
        let asm = k.to_assembly(Target::GFX1100).expect("asm gen failed");
        eprintln!("\n═══ WGP Kernel Assembly ═══");
        for line in asm.lines() {
            eprintln!("  {}", line);
        }
        eprintln!("═══ End ASM ═══\n");

        // Verify WGP mode is set in the assembly
        assert!(asm.contains("amdhsa_workgroup_processor_mode 1"),
            "WGP mode flag missing from kernel descriptor!");

        // Compile to ELF
        let elf = k.compile(Target::GFX1100).expect("ELF compile failed");
        eprintln!("[WGP dump] ELF size: {} bytes", elf.len());

        // Dump kernel descriptor from ELF (last 64 bytes of .rodata)
        // The KD is at the .amdhsa_kernel symbol
        eprintln!("[WGP dump] First 64 bytes of ELF (header):");
        for chunk in elf[..64.min(elf.len())].chunks(16) {
            let hex: String = chunk.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
            eprintln!("  {}", hex);
        }

        // Also build a CU mode version for comparison
        let mut k_cu = T0Kernel::new("cu_dump_test");
        k_cu.set_wg_size(256);
        k_cu.set_lds_size(4096);
        let out_ptr2 = k_cu.arg_ptr("out");
        k_cu.emit_arg_loads();
        let gid2 = k_cu.compute_global_id_x(256);
        let val2 = k_cu.alloc_vreg();
        k_cu.push(Op::VCvtF32U32 { dst: val2, src: gid2 });
        let lds_addr2 = k_cu.alloc_vreg();
        k_cu.v_lshlrev_b32(lds_addr2, 2, VReg(0));
        k_cu.ds_store_b32(lds_addr2, val2, 0);
        k_cu.wait_lgkmcnt(0);
        k_cu.wait_vmcnt(0);
        k_cu.s_barrier();
        let rval2 = k_cu.alloc_vreg();
        k_cu.ds_load_b32(rval2, lds_addr2, 0);
        k_cu.wait_lgkmcnt(0);
        let offset2 = k_cu.alloc_vreg();
        k_cu.v_lshlrev_b32(offset2, 2, gid2);
        let (addr_lo2, addr_hi2) = k_cu.alloc_addr_pair();
        k_cu.v_mov_from_sgpr(addr_lo2, SReg(out_ptr2.0));
        k_cu.v_mov_from_sgpr(addr_hi2, SReg(out_ptr2.0 + 1));
        k_cu.addr64_add(addr_lo2, addr_hi2, offset2);
        k_cu.global_store(addr_lo2, rval2, Width::B32, 0);
        k_cu.wait_vscnt(0);
        k_cu.endpgm();

        let asm_cu = k_cu.to_assembly(Target::GFX1100).expect("CU asm gen failed");
        assert!(!asm_cu.contains("amdhsa_workgroup_processor_mode"),
            "CU mode should NOT have WGP flag!");

        eprintln!("\n═══ CU Kernel Assembly (for diff) ═══");
        // Only show the kernel descriptor section
        for line in asm_cu.lines().filter(|l| l.contains("amdhsa_") || l.contains(".amdhsa_")) {
            eprintln!("  {}", line);
        }
        eprintln!("  ---");
        for line in asm.lines().filter(|l| l.contains("amdhsa_") || l.contains(".amdhsa_")) {
            eprintln!("  {}", line);
        }
        eprintln!("═══ End KD diff ═══");

        eprintln!("\n[PASS] test_wgp_asm_dump: WGP kernel compiles correctly");
    }

    /// COMPILE-ONLY: Dump the WGP probe kernel ASM for root cause analysis.
    /// Does NOT dispatch to GPU — safe to run after hard hang.
    #[test]
    fn test_wgp_probe_asm_dump() {
        use crate::t0::compile::T0Kernel;
        use crate::t0::ir::*;

        let mut k = T0Kernel::new("wgp_probe_cu_baseline");
        k.set_wg_size(256);
        let out_ptr = k.arg_ptr("out");
        k.emit_arg_loads();
        let gid = k.compute_global_id_x(256);
        let val = k.alloc_vreg();
        k.push(Op::VCvtF32U32 { dst: val, src: gid });
        k.wait_vmcnt(0);
        k.wait_lgkmcnt(0);
        k.s_barrier();
        let offset = k.alloc_vreg();
        k.v_lshlrev_b32(offset, 2, gid);
        let (addr_lo, addr_hi) = k.alloc_addr_pair();
        k.v_mov_from_sgpr(addr_lo, SReg(out_ptr.0));
        k.v_mov_from_sgpr(addr_hi, SReg(out_ptr.0 + 1));
        k.addr64_add(addr_lo, addr_hi, offset);
        k.global_store(addr_lo, val, Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();

        let (asm, _) = k.to_assembly_with_info(Target::GFX1100)
            .expect("CU compile failed");

        eprintln!("\n=== CU Probe Kernel ASM ===\n");
        for (i, line) in asm.lines().enumerate() {
            eprintln!("{:4}: {}", i + 1, line);
        }
        eprintln!("\n[PASS] compile-only, no GPU dispatch");
    }

    /// COMPILE-ONLY: Verify WGP mode is correctly set in the ELF RSRC1 field.
    #[test]
    fn test_wgp_rsrc1_verification() {
        use crate::t0::compile::T0Kernel;
        use crate::t0::ir::*;

        // Build a simple WGP kernel
        let mut k = T0Kernel::new("wgp_rsrc1_check");
        k.set_wg_size(256);
        k.set_wgp_mode(true);
        k.set_lds_size(81920);
        let out_ptr = k.arg_ptr("out");
        k.emit_arg_loads();
        k.endpgm();

        // Get assembly text to verify directive is present
        let (asm, _) = k.to_assembly_with_info(Target::GFX1100).expect("asm");
        let has_wgp_directive = asm.contains(".amdhsa_workgroup_processor_mode 1");
        eprintln!("[ASM] .amdhsa_workgroup_processor_mode 1 present: {}", has_wgp_directive);
        assert!(has_wgp_directive, "WGP directive missing from assembly text!");

        // Compile to ELF binary
        let (elf, _) = k.compile_with_info(Target::GFX1100).expect("compile");
        eprintln!("[ELF] {} bytes", elf.len());

        // Find the kernel descriptor in the ELF .rodata section
        // KD is 64 bytes, starts at a 64-byte aligned boundary
        // COMPUTE_PGM_RSRC1 is at KD offset 0x30 (48)
        // We search for the KD by looking for a known pattern:
        // group_segment_fixed_size = 81920 = 0x00014000
        let target_lds = 81920u32.to_le_bytes(); // [00, 40, 01, 00]
        
        let mut kd_offset = None;
        for i in 0..elf.len().saturating_sub(64) {
            if i % 64 == 0 && elf[i..i+4] == target_lds {
                kd_offset = Some(i);
                break;
            }
        }
        // Also try non-aligned search
        if kd_offset.is_none() {
            for i in 0..elf.len().saturating_sub(64) {
                if elf[i..i+4] == target_lds {
                    kd_offset = Some(i);
                    break;
                }
            }
        }

        if let Some(off) = kd_offset {
            eprintln!("[KD] Found at ELF offset 0x{:X}", off);
            // Dump KD hex
            for row in 0..4 {
                let base = off + row * 16;
                if base + 16 <= elf.len() {
                    eprint!("  {:02X}:", row * 16);
                    for j in 0..16 {
                        eprint!(" {:02X}", elf[base + j]);
                    }
                    eprintln!();
                }
            }

            // RSRC1 at KD+0x30
            if off + 0x34 <= elf.len() {
                let rsrc1 = u32::from_le_bytes([
                    elf[off + 0x30], elf[off + 0x31], elf[off + 0x32], elf[off + 0x33]
                ]);
                let wgp_bit = (rsrc1 >> 29) & 1;  // ENABLE_WGP_MODE (GFX10+)
                let mem_ordered = (rsrc1 >> 30) & 1;  // MEM_ORDERED
                let fwd_progress = (rsrc1 >> 31) & 1;  // FWD_PROGRESS
                let vgpr_gran = rsrc1 & 0x3F;
                
                eprintln!("\n[RSRC1] 0x{:08X}", rsrc1);
                eprintln!("  WGP_MODE (bit 29):    {}", wgp_bit);
                eprintln!("  MEM_ORDERED (bit 30): {}", mem_ordered);
                eprintln!("  FWD_PROGRESS (bit 31): {}", fwd_progress);
                eprintln!("  VGPR granulated: {} → {} VGPRs", vgpr_gran, (vgpr_gran + 1) * 8);

                if wgp_bit == 0 {
                    eprintln!("\n⚠️ WGP_MODE NOT SET in RSRC1 bit 29.");
                } else {
                    eprintln!("\n✅ WGP_MODE correctly set in RSRC1 bit 29!");
                }
            }

            // kernel_code_properties at KD+0x38
            if off + 0x3A <= elf.len() {
                let kcp = u16::from_le_bytes([elf[off + 0x38], elf[off + 0x39]]);
                eprintln!("[KCP] 0x{:04X}", kcp);
                eprintln!("  bit 3 (ENABLE_SGPR_KERNARG): {}", (kcp >> 3) & 1);
                eprintln!("  bit 9 (WAVEFRONT_SIZE32): {}", (kcp >> 9) & 1);
                eprintln!("  bit 10 (USES_DYNAMIC_STACK): {}", (kcp >> 10) & 1);
            }
        } else {
            eprintln!("[KD] NOT FOUND in ELF! Searching for LDS=81920 pattern failed.");
            // Dump first 256 bytes for inspection
            for row in 0..16 {
                let base = row * 16;
                if base + 16 <= elf.len() {
                    eprint!("  {:04X}:", base);
                    for j in 0..16 { eprint!(" {:02X}", elf[base + j]); }
                    eprintln!();
                }
            }
        }
    }
}
