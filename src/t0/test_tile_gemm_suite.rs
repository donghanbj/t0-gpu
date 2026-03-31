//! Comprehensive GPU test suite for tile_gemm (tile_ir compilation path)
//!
//! Validates correctness, precision, and robustness across matrix sizes and tile configs.

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use crate::t0::block_dsl::*;
    use crate::t0::gemm_gen::{self, GemmConfig};
    use crate::t0::ir::Target;
    use crate::ignis::gpu_context::GpuRuntime;
    use std::sync::Arc;

    fn setup() -> Arc<GpuRuntime> {
        GpuRuntime::new().expect("GPU init failed")
    }

    fn make_bf16(n: usize, seed: u64) -> Vec<u16> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let f = (((s >> 33) as u32 % 200) as f32 - 100.0) * 0.01;
            (f.to_bits() >> 16) as u16
        }).collect()
    }

    fn cpu_gemm_nt(x: &[u16], w: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
        let bf = |b: u16| f32::from_bits((b as u32) << 16);
        let mut y = vec![0f32; m * n];
        for mi in 0..m { for ni in 0..n {
            let mut s = 0f32;
            for ki in 0..k { s += bf(x[mi*k+ki]) * bf(w[ni*k+ki]); }
            y[mi*n+ni] = s;
        }}
        y
    }

    /// Core test runner: compile tile_gemm, dispatch, compare with CPU ref
    fn run(rt: &GpuRuntime, m: u32, k: u32, n: u32, cfg: TileGemmConfig)
        -> Result<(f32, usize, f64), String>
    {
        // Build and compile kernel
        let gcfg = GemmConfig {
            tile_m: cfg.tile_m, tile_n: cfg.tile_n, tile_k: cfg.tile_k,
            wg_size: cfg.tile_m / 32 * 32, // n_waves * 32
            use_lds: true, double_buffer: true,
            split_k: if cfg.split_k > 1 { Some(cfg.split_k) } else { None },
            lds_pad: 0, n_col_passes: 1,
            swap_grid: cfg.swap_grid, wgp_mode: cfg.wgp_mode,
            transpose: gemm_gen::GemmTranspose::NT,
            epilogue: gemm_gen::EpilogueOp::StoreF32,
        };
        let name = format!("tgm_{}x{}x{}_{}x{}", m, k, n, cfg.tile_m, cfg.tile_n);
        let mut kb = BlockKernel::new(&name, gcfg.wg_size);
        let _x = kb.arg_ptr("X"); let _w = kb.arg_ptr("WT"); let _y = kb.arg_ptr("Y");
        let _k = kb.arg_u32("K"); let _n = kb.arg_u32("N");
        let _sks = kb.arg_u32("split_k_shift"); let _yss = kb.arg_u32("y_split_stride");
        let _m = kb.arg_u32("M");
        let tile_k = cfg.tile_k;
        kb.tile_gemm(_x, _w, _y, _k, _n, cfg);
        let compiled = kb.compile(Target::GFX1100)?;

        // Prepare data — pad K to tile_k alignment for safe K-loop ceiling
        let k_padded = (k + tile_k - 1) & !(tile_k - 1);  // ceil(K, tile_k)
        let xb = make_bf16((m*k) as usize, 42+m as u64);
        let wb = make_bf16((n*k) as usize, 137+n as u64);
        // Allocate padded buffers (K_padded >= K), zero-fill so OOB K elements are 0
        let xbuf = rt.alloc((m * k_padded * 2) as usize).map_err(|e| e.to_string())?;
        xbuf.zero();
        // Copy row-by-row: each row is K elements, padded to K_padded
        if k == k_padded {
            xbuf.write(&xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        } else {
            let mut padded = vec![0u8; (m * k_padded * 2) as usize];
            for row in 0..m as usize {
                let src_off = row * k as usize;
                let dst_off = row * k_padded as usize;
                for col in 0..k as usize {
                    padded[(dst_off + col) * 2..(dst_off + col) * 2 + 2]
                        .copy_from_slice(&xb[src_off + col].to_le_bytes());
                }
            }
            xbuf.write(&padded);
        }
        let wbuf = rt.alloc((n * k_padded * 2) as usize).map_err(|e| e.to_string())?;
        wbuf.zero();
        if k == k_padded {
            wbuf.write(&wb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        } else {
            let mut padded = vec![0u8; (n * k_padded * 2) as usize];
            for row in 0..n as usize {
                let src_off = row * k as usize;
                let dst_off = row * k_padded as usize;
                for col in 0..k as usize {
                    padded[(dst_off + col) * 2..(dst_off + col) * 2 + 2]
                        .copy_from_slice(&wb[src_off + col].to_le_bytes());
                }
            }
            wbuf.write(&padded);
        }
        // Allocate Y buffer padded to tile-aligned dimensions.
        // Boundary WGs store OOB rows/cols to the padded area (harmless).
        let m_padded = (m + gcfg.tile_m - 1) & !(gcfg.tile_m - 1);
        let n_padded = (n + gcfg.tile_n - 1) & !(gcfg.tile_n - 1);
        let ybuf = rt.alloc_f32((m_padded * n_padded) as usize).map_err(|e| e.to_string())?;
        ybuf.zero();

        // Build 44-byte kernargs: [X:u64, WT:u64, Y:u64, K:u32, N:u32, sks:u32, yss:u32, M:u32]
        // CRITICAL: use k_padded for K because cooperative load stride is row * K * 2
        let sk = gcfg.split_k.unwrap_or(1);
        let y_stride = if sk > 1 { m * n * 4 } else { 0u32 };
        let mut ka = [0u8; 44];
        ka[0..8].copy_from_slice(&xbuf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&wbuf.gpu_addr().to_le_bytes());
        ka[16..24].copy_from_slice(&ybuf.gpu_addr().to_le_bytes());
        ka[24..28].copy_from_slice(&k_padded.to_le_bytes());  // K_padded for stride
        ka[28..32].copy_from_slice(&n_padded.to_le_bytes());  // N_padded for Y stride
        ka[32..36].copy_from_slice(&0u32.to_le_bytes());
        ka[36..40].copy_from_slice(&y_stride.to_le_bytes());
        ka[40..44].copy_from_slice(&m.to_le_bytes());  // M dimension

        // Grid: ceil(M/tile_m) × ceil(N/tile_n)
        let tiles_m = (m + gcfg.tile_m - 1) / gcfg.tile_m;
        let tiles_n = (n + gcfg.tile_n - 1) / gcfg.tile_n;
        let (gx, gy) = if gcfg.swap_grid {
            (tiles_n * gcfg.wg_size, tiles_m)
        } else {
            (tiles_m * gcfg.wg_size, tiles_n)
        };
        let gk = rt.compile_dsl(compiled).map_err(|e| e.to_string())?;
        rt.dispatch(&gk, [gx, gy, 1], &ka)?;

        // Compare
        let yref = cpu_gemm_nt(&xb, &wb, m as usize, k as usize, n as usize);
        // Read back: kernel stored with N_padded stride, extract valid M×N submatrix
        let ygpu_padded = rt.read_f32(&ybuf, (m_padded * n_padded) as usize);
        let mut ygpu = vec![0f32; (m*n) as usize];
        for row in 0..m as usize {
            for col in 0..n as usize {
                ygpu[row * n as usize + col] = ygpu_padded[row * n_padded as usize + col];
            }
        }
        let mut me = 0f32; let mut nb = 0usize;
        for i in 0..(m*n) as usize {
            let e = (ygpu[i]-yref[i]).abs();
            if e > me { me = e; } if e > 1.0 { nb += 1; }
        }

        // Perf bench loop — DISABLED in combo to prevent dispatch accumulation.
        // After many dispatches, large matrix kernels can exceed 5s timeout.
        // Use individual #[ignore] tests for perf measurement instead.
        let us = 0.0;

        // Recycle buffers to pool instead of dropping (avoids KFD VA reuse race)
        rt.recycle(xbuf);
        rt.recycle(wbuf);
        rt.recycle(ybuf);

        Ok((me, nb, us))
    }

    fn c128() -> TileGemmConfig {
        TileGemmConfig { tile_m: 128, tile_n: 64, tile_k: 16,
            wgp_mode: false, split_k: 1, swap_grid: true }
    }






    fn c32() -> TileGemmConfig {
        TileGemmConfig { tile_m: 32, tile_n: 64, tile_k: 16,
            wgp_mode: false, split_k: 1, swap_grid: true }
    }

    /// Sequential combo: runs multiple GEMM dimensions within ONE test function.
    /// This avoids inter-test GpuBuffer Drop VA-reuse race conditions
    /// that cause GPU page faults in KFD bare-metal dispatch.
    #[test] fn test_tg_combo_sequential() {
        let rt = setup();
        let cases: Vec<(u32, u32, u32, TileGemmConfig, &str, f32)> = vec![
            // Aligned tests
            (128, 128, 128, c128(), "128³", 0.01),
            (256, 256, 256, c32(), "256³", 0.01),
            (128, 1024, 512, c128(), "128×1024×512", 0.5),
            (256, 512, 128, c128(), "256×512×128", 0.01),
            (512, 512, 512, c128(), "512³", 0.5),
            (1024, 1024, 1024, c128(), "1024³", 0.5),
            // Non-aligned tests
            (33, 100, 50, c32(), "33×100×50", 1.0),
            (129, 65, 17, c128(), "129×65×17", 0.5),
            (100, 300, 200, c32(), "100×300×200", 5.0),
            (1000, 768, 512, c128(), "1000×768×512", 5.0),
        ];
        for (m, k, n, cfg, label, tol) in cases {
            match run(&rt, m, k, n, cfg) {
                Ok((e, b, _)) => {
                    eprintln!("[combo] {}: err={:.2e} bad={}", label, e, b);
                    assert!(e < tol, "{} err={}", label, e);
                    assert_eq!(b, 0, "{} bad={}", label, b);
                }
                Err(e) => {
                    eprintln!("[combo] {} FAILED: {}", label, e);
                    panic!("{} failed: {}", label, e);
                }
            }
        }
    }
}
