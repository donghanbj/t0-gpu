//! T0 vs PyTorch 逐层精度对比测试
//!
//! 加载 `benchmarks/gen_torch_ref.py` 生成的参考数据，
//! 用 T0 编译器的 GEMM + SwiGLU 内核执行同样的链路，
//! 逐层对比精度。
//!
//! 运行:
//!   1. python3 benchmarks/gen_torch_ref.py
//!   2. cargo test --release --lib --features rocm -- test_precision_vs_torch --ignored --nocapture --test-threads=1

#[cfg(all(test, feature = "rocm"))]
mod precision_tests {
    use std::path::Path;

    /// Load BF16 binary file as Vec<u16>
    fn load_bf16(path: &str) -> Vec<u16> {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("Cannot read {}: {}", path, e));
        data.chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    /// Load F32 binary file as Vec<f32>
    fn load_f32(path: &str) -> Vec<f32> {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("Cannot read {}: {}", path, e));
        data.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Load metadata as key-value map
    fn load_meta(path: &str) -> std::collections::HashMap<String, String> {
        let text = std::fs::read_to_string(path).unwrap();
        text.lines()
            .filter_map(|l| {
                let mut parts = l.splitn(2, '=');
                Some((parts.next()?.to_string(), parts.next()?.to_string()))
            })
            .collect()
    }

    /// Compare two f32 slices and print statistics
    fn compare(label: &str, gpu: &[f32], ref_data: &[f32], rows: usize, cols: usize) -> f64 {
        assert_eq!(gpu.len(), ref_data.len(), "{}: length mismatch", label);
        let n = gpu.len();
        let mut max_err = 0f64;
        let mut sum_err = 0f64;
        let mut max_rel = 0f64;
        let mut n_bad = 0usize;
        let mut worst_idx = 0;

        for i in 0..n {
            let err = (gpu[i] as f64 - ref_data[i] as f64).abs();
            let rel = if ref_data[i].abs() > 1e-8 {
                err / ref_data[i].abs() as f64
            } else {
                err
            };
            sum_err += err;
            if err > max_err {
                max_err = err;
                worst_idx = i;
            }
            if rel > max_rel { max_rel = rel; }
            if err > 0.1 { n_bad += 1; }
        }
        let mean_err = sum_err / n as f64;
        let worst_row = worst_idx / cols;
        let worst_col = worst_idx % cols;

        eprintln!("  {:<25} max_err={:.6e}  mean_err={:.6e}  max_rel={:.4e}  bad(>0.1)={}",
            label, max_err, mean_err, max_rel, n_bad);
        if max_err > 1e-3 {
            eprintln!("    ⚠️  worst at [{},{}]: T0={:.6} torch={:.6} Δ={:.6e}",
                worst_row, worst_col, gpu[worst_idx], ref_data[worst_idx], max_err);
        }
        max_err
    }

    #[test]
    #[ignore]
    fn test_precision_vs_torch() {
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::t0::block_dsl::*;
        use crate::t0::ir::Target;
        use crate::t0::tile_ir;
        use crate::t0::tile_ssa;

        let ref_dir = "benchmarks/torch_ref";
        if !Path::new(&format!("{}/meta.txt", ref_dir)).exists() {
            eprintln!("⚠️  参考数据不存在，请先运行: python3 benchmarks/gen_torch_ref.py");
            return;
        }

        let meta = load_meta(&format!("{}/meta.txt", ref_dir));
        let m: u32 = meta["M"].parse().unwrap();
        let k: u32 = meta["K"].parse().unwrap();
        let n: u32 = meta["N"].parse().unwrap();

        eprintln!("\n╔══════════════════════════════════════════════════════════╗");
        eprintln!("║  T0 vs PyTorch — GEMM + SwiGLU + GEMM 逐层精度对比      ║");
        eprintln!("║  M={}, K={}, N={}  torch={}              ║",
            m, k, n, meta.get("torch_version").unwrap_or(&"?".into()));
        eprintln!("╚══════════════════════════════════════════════════════════╝\n");

        // Load inputs (BF16)
        let x_bf16 = load_bf16(&format!("{}/x.bin", ref_dir));
        let w_gate_bf16 = load_bf16(&format!("{}/w_gate.bin", ref_dir));
        let w_up_bf16 = load_bf16(&format!("{}/w_up.bin", ref_dir));
        let w_down_bf16 = load_bf16(&format!("{}/w_down.bin", ref_dir));

        // Load torch references (F32)
        let torch_gate = load_f32(&format!("{}/h_gate_bf16acc.bin", ref_dir));
        let torch_up = load_f32(&format!("{}/h_up_bf16acc.bin", ref_dir));
        let torch_swiglu = load_f32(&format!("{}/swiglu_bf16chain.bin", ref_dir));
        let torch_y = load_f32(&format!("{}/y_bf16chain.bin", ref_dir));

        assert_eq!(x_bf16.len(), (m * k) as usize, "X shape mismatch");
        assert_eq!(w_gate_bf16.len(), (n * k) as usize, "W_gate shape mismatch");

        eprintln!("  Input shapes: X=[{},{}] W_gate=[{},{}] W_up=[{},{}] W_down=[{},{}]\n",
            m, k, n, k, n, k, k, n);

        // ═══════════════════════════════════════════════════
        // GPU runtime
        // ═══════════════════════════════════════════════════
        let rt = GpuRuntime::new().expect("GPU init failed");

        // Upload BF16 inputs to VRAM
        let x_bytes: Vec<u8> = x_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wg_bytes: Vec<u8> = w_gate_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wu_bytes: Vec<u8> = w_up_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wd_bytes: Vec<u8> = w_down_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let x_buf = rt.alloc(x_bytes.len()).unwrap(); x_buf.write(&x_bytes);
        let wg_buf = rt.alloc(wg_bytes.len()).unwrap(); wg_buf.write(&wg_bytes);
        let wu_buf = rt.alloc(wu_bytes.len()).unwrap(); wu_buf.write(&wu_bytes);
        let wd_buf = rt.alloc(wd_bytes.len()).unwrap(); wd_buf.write(&wd_bytes);

        // ═══════════════════════════════════════════════════
        // Stage 1: GEMM (gate projection) — X @ W_gate.T
        // ═══════════════════════════════════════════════════
        eprintln!("  ── Stage 1: GEMM gate/up projections ──");
        let h_gate_buf = rt.alloc_f32((m * n) as usize).unwrap();
        let h_up_buf = rt.alloc_f32((m * n) as usize).unwrap();

        // Auto-select tile spec: tile_M must <= M to avoid boundary overflow
        // (tile_128x64 on M=64 causes 17619 bad elements — M boundary not clamped)
        let spec = if m >= 128 {
            tile_ir::TileGemm::tile_128x64_k16()
        } else {
            tile_ir::TileGemm::tile_64x64_k16()
        };
        let gemm_kernel = rt.ensure_kernel_t0(
            "precision_gemm",
            || tile_ir::lower_gemm(&spec),
            [spec.wg_size(), 1, 1],
            spec.lds_total(),
        ).expect("GEMM compile failed");

        let ka_gate = tile_ir::build_kernargs_m(
            x_buf.gpu_addr(), wg_buf.gpu_addr(), h_gate_buf.gpu_addr(),
            k, n, m, &spec,
        );
        let grid = tile_ir::compute_grid(&spec, m, n);
        rt.dispatch(&gemm_kernel, grid, &ka_gate).expect("GEMM gate dispatch");

        let ka_up = tile_ir::build_kernargs_m(
            x_buf.gpu_addr(), wu_buf.gpu_addr(), h_up_buf.gpu_addr(),
            k, n, m, &spec,
        );
        rt.dispatch(&gemm_kernel, grid, &ka_up).expect("GEMM up dispatch");

        let h_gate_gpu = rt.read_f32(&h_gate_buf, (m * n) as usize);
        let h_up_gpu = rt.read_f32(&h_up_buf, (m * n) as usize);
        let e1g = compare("GEMM gate (T0 vs torch)", &h_gate_gpu, &torch_gate, m as usize, n as usize);
        let e1u = compare("GEMM up (T0 vs torch)", &h_up_gpu, &torch_up, m as usize, n as usize);

        // ═══════════════════════════════════════════════════
        // Stage 2: SwiGLU — out[i] = silu(gate[i]) * up[i]
        // ═══════════════════════════════════════════════════
        eprintln!("\n  ── Stage 2: SwiGLU (fused silu_mul) ──");
        let swiglu_buf = rt.alloc_f32((m * n) as usize).unwrap();

        let swiglu_func = tile_ssa::ElemChain::swiglu(256);
        let lowered = crate::t0::tile_ssa_lower::lower_elementwise_1d(&swiglu_func, 256, 1)
            .expect("SwiGLU lower failed");
        let swiglu_elf = lowered.kernel.compile(Target::GFX1100)
            .expect("SwiGLU compile failed");

        // Build CompiledKernel manually for compile_dsl
        let swiglu_args: Vec<crate::t0::dsl::KernArgMeta> = lowered.kernel.args().iter().map(|a| {
            crate::t0::dsl::KernArgMeta {
                name: a.name.clone(),
                kind: match a.kind {
                    crate::t0::ir::ArgKind::Ptr => crate::t0::dsl::KernArgType::Ptr,
                    crate::t0::ir::ArgKind::U32 => crate::t0::dsl::KernArgType::U32,
                    crate::t0::ir::ArgKind::F32 => crate::t0::dsl::KernArgType::F32,
                },
                offset: a.offset as usize,
            }
        }).collect();
        let swiglu_ck = crate::t0::dsl::CompiledKernel {
            elf: swiglu_elf,
            kernarg_size: lowered.kernel.kernarg_size() as usize,
            workgroup_size: [256, 1, 1],
            lds_size: lowered.kernel.lds_size(),
            name: "t0_swiglu".to_string(),
            args: swiglu_args,
        };
        let swiglu_kernel = rt.compile_dsl(swiglu_ck).expect("SwiGLU load failed");

        let n_elems = (m * n) as u32;
        let ka_swiglu = crate::kernargs![
            h_gate_buf.gpu_addr() => u64,
            h_up_buf.gpu_addr() => u64,
            swiglu_buf.gpu_addr() => u64,
            n_elems => u32
        ];
        let swiglu_grid_x = ((n_elems + 255) / 256) * 256;
        rt.dispatch(&swiglu_kernel, [swiglu_grid_x, 1, 1], &ka_swiglu)
            .expect("SwiGLU dispatch");

        let swiglu_gpu = rt.read_f32(&swiglu_buf, (m * n) as usize);
        let e2 = compare("SwiGLU (T0 vs torch)", &swiglu_gpu, &torch_swiglu, m as usize, n as usize);

        // ═══════════════════════════════════════════════════
        // Stage 3: F32 → BF16 conversion + GEMM2 (down projection)
        //   Uses new BlockDSL store_bf16_checked for clean F32→BF16 truncation
        // ═══════════════════════════════════════════════════
        eprintln!("\n  ── Stage 3: GEMM2 down projection ──");

        // F32 → BF16 conversion kernel using BlockDSL store_bf16
        let swiglu_bf16_buf = rt.alloc((m * n * 2) as usize).unwrap();
        {
            let mut kb = BlockKernel::new("f32_to_bf16_cvt", 256);
            let src = kb.arg_ptr("src");
            let dst = kb.arg_ptr("dst");
            let count = kb.arg_u32("count");
            let offsets = kb.arange(0, 256);
            let pid = kb.program_id(0);
            let c256 = kb.const_u32(256);
            let base = pid.mul(&mut kb, c256);
            let idx = offsets.add(&mut kb, base);

            // Load F32 with bounds check
            let val = kb.load_checked(src, idx, count);
            // Store as BF16 with bounds check
            kb.store_bf16_checked(dst, idx, val, count);

            let ck = kb.compile_via_ssa(Target::GFX1100).expect("f32→bf16 compile");
            let cvt_kernel = rt.compile_dsl(ck).expect("f32→bf16 load");
            let ka = crate::kernargs![
                swiglu_buf.gpu_addr() => u64,
                swiglu_bf16_buf.gpu_addr() => u64,
                n_elems => u32
            ];
            let grid_x = ((n_elems + 255) / 256) * 256;
            rt.dispatch(&cvt_kernel, [grid_x, 1, 1], &ka).expect("f32→bf16 dispatch");
        }

        // GEMM2: swiglu_bf16 [M, N] @ W_down [K, N].T → y [M, K]
        let y_buf = rt.alloc_f32((m * k) as usize).unwrap();

        let spec2 = if m >= 128 {
            tile_ir::TileGemm::tile_128x64_k16()
        } else {
            tile_ir::TileGemm::tile_64x64_k16()
        };
        let gemm2_kernel = rt.ensure_kernel_t0(
            "precision_gemm2",
            || tile_ir::lower_gemm(&spec2),
            [spec2.wg_size(), 1, 1],
            spec2.lds_total(),
        ).expect("GEMM2 compile");

        let ka_down = tile_ir::build_kernargs_m(
            swiglu_bf16_buf.gpu_addr(), wd_buf.gpu_addr(), y_buf.gpu_addr(),
            n, k, m, &spec2,
        );
        let grid2 = tile_ir::compute_grid(&spec2, m, k);
        rt.dispatch(&gemm2_kernel, grid2, &ka_down).expect("GEMM2 dispatch");

        let y_gpu = rt.read_f32(&y_buf, (m * k) as usize);
        let e3 = compare("GEMM2 output (T0 vs torch)", &y_gpu, &torch_y, m as usize, k as usize);

        // ═══════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════
        eprintln!("\n  ════════════════════════════════════════════════════");
        eprintln!("  精度总结 (T0 vs PyTorch {}):", meta.get("torch_version").unwrap_or(&"?".into()));
        eprintln!("    GEMM gate:    max_err = {:.6e}  {}", e1g,
            if e1g < 1e-2 { "✅" } else { "⚠️" });
        eprintln!("    GEMM up:      max_err = {:.6e}  {}", e1u,
            if e1u < 1e-2 { "✅" } else { "⚠️" });
        eprintln!("    SwiGLU:       max_err = {:.6e}  {}", e2,
            if e2 < 1e-3 { "✅" } else { "⚠️" });
        eprintln!("    GEMM2 final:  max_err = {:.6e}  {}", e3,
            if e3 < 1e-2 { "✅" } else { "⚠️" });
        eprintln!("  ════════════════════════════════════════════════════");

        assert!(e1g < 0.05, "GEMM gate precision regression: {:.6e}", e1g);
        assert!(e1u < 0.05, "GEMM up precision regression: {:.6e}", e1u);
        assert!(e2 < 0.01, "SwiGLU precision regression: {:.6e}", e2);
        assert!(e3 < 0.1, "GEMM2 end-to-end precision regression: {:.6e}", e3);

        eprintln!("\n  ✅ T0 精度验证通过！");
    }
}
