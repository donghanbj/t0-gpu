//! GFX1100 Cost Model for Auto-Scheduling
//!
//! Models hardware constraints of AMD RX 7900 XTX (RDNA3, GFX1100)
//! and provides cost estimation for GEMM tile parameter selection.
//!
//! # Usage
//! ```rust
//! let best = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
//! // Returns optimal tile_m, tile_n, tile_k, workgroup_size
//! ```

// ============================================================================
// GFX1100 Hardware Limits
// ============================================================================

/// Hardware specifications for GFX1100 (Navi 31, RDNA3).
#[derive(Clone, Debug)]
pub struct GFX1100Limits {
    /// VGPRs per SIMD unit (Wave32 mode)
    pub max_vgprs: u32,
    /// SGPRs per wavefront  
    pub max_sgprs: u32,
    /// LDS per workgroup (bytes)
    pub lds_per_wg: u32,
    /// Total LDS per CU (bytes, shared by all WGs on CU)
    pub lds_per_cu: u32,
    /// Number of CUs on die
    pub n_cus: u32,
    /// WMMA throughput: VALU-normalized cycles per v_wmma instruction
    /// (empirical: ~43 shader cycles / 18 ≈ 2.4, rounded to 4)
    pub wmma_cycles: u32,
    /// WMMA tile size: always 16×16×16 on GFX11
    pub wmma_mn: u32,
    pub wmma_k: u32,
    /// Peak VMEM bandwidth (GB/s)
    pub vmem_bandwidth_gbps: f64,
    /// Peak LDS bandwidth (GB/s)
    pub lds_bandwidth_gbps: f64,
    /// GPU clock (GHz, for cycle estimation)
    pub clock_ghz: f64,
    /// WGP mode: each WGP = 2 CUs sharing LDS.
    /// In WGP mode, a workgroup can use both SIMDs in the WGP,
    /// effectively doubling the concurrent wave capacity.
    pub wgp_mode: bool,
    /// SIMDs per CU (RDNA3 = 2)
    pub simds_per_cu: u32,
    /// L2 cache size in bytes (RX 7900 XTX = 6 MB)
    pub l2_cache_bytes: u64,
}

impl Default for GFX1100Limits {
    fn default() -> Self {
        GFX1100Limits {
            max_vgprs: 256,
            max_sgprs: 106,
            lds_per_wg: 65536,    // 64 KB
            lds_per_cu: 131072,   // 128 KB (shared across WGP in WGP mode)
            n_cus: 96,
            wmma_cycles: 4,       // probe-calibrated: ~36 shader cycles ≈ 3.4→4 VALU-norm
            wmma_mn: 16,
            wmma_k: 16,
            vmem_bandwidth_gbps: 960.0,
            lds_bandwidth_gbps: 3700.0,
            clock_ghz: 2.5,
            wgp_mode: true,       // enable WGP mode by default on GFX1100
            simds_per_cu: 2,
            l2_cache_bytes: 6 * 1024 * 1024, // 6 MB L2 on RX 7900 XTX
        }
    }
}

// ============================================================================
// Data format
// ============================================================================

/// Data format for GEMM operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataFormat {
    BF16,   // 2 bytes, WMMA native
    F16,    // 2 bytes, WMMA native
    F32,    // 4 bytes, no WMMA (scalar mul-add)
}

impl DataFormat {
    pub fn bytes(&self) -> u32 {
        match self {
            DataFormat::BF16 | DataFormat::F16 => 2,
            DataFormat::F32 => 4,
        }
    }

    pub fn has_wmma(&self) -> bool {
        matches!(self, DataFormat::BF16 | DataFormat::F16)
    }
}

// ============================================================================
// Tile configuration
// ============================================================================

/// A candidate tile configuration for GEMM.
#[derive(Clone, Debug)]
pub struct TileConfig {
    pub tile_m: u32,        // rows per workgroup
    pub tile_n: u32,        // cols per workgroup  
    pub tile_k: u32,        // K-dimension per loop iteration
    pub waves_per_wg: u32,  // waves in workgroup (1, 2, or 4)
    pub use_lds: bool,      // whether to use LDS for tile staging
    pub split_k: u32,       // split-K factor (1 = no split, 2/4/8/16)
    pub wgp_mode: bool,     // WGP mode (2 CUs share LDS)
    pub swap_grid: bool,    // true=N-on-X (L2 friendly), false=M-on-X (rectangular)
    pub lds_pad: u32,       // LDS row padding in bytes (0/4/8, eliminates bank conflicts)
}

impl TileConfig {
    /// Workgroup size in threads.
    pub fn wg_threads(&self) -> u32 {
        self.waves_per_wg * 32 // Wave32
    }

    /// Number of WMMA tiles in N direction.
    pub fn n_wmma_tiles(&self) -> u32 {
        self.tile_n / 16
    }

    /// Number of WMMA tiles in M direction (per wave).
    pub fn m_wmma_per_wave(&self) -> u32 {
        (self.tile_m / self.waves_per_wg) / 16
    }

    /// Short name for display.
    pub fn name(&self) -> String {
        let sk = if self.split_k > 1 { format!("_sk{}", self.split_k) } else { String::new() };
        let wgp = if self.wgp_mode { "_wgp" } else { "" };
        let mg = if !self.swap_grid { "_mg" } else { "" };
        let pad = if self.lds_pad > 0 { format!("_pad{}", self.lds_pad) } else { String::new() };
        format!("{}x{}_k{}{}{}{}{}", self.tile_m, self.tile_n, self.tile_k, sk, mg, wgp, pad)
    }
}

/// Cost estimation result.
#[derive(Clone, Debug)]
pub struct TileCost {
    pub config: TileConfig,
    /// Estimated VGPRs needed per wave
    pub vgprs: u32,
    /// Estimated SGPRs needed
    pub sgprs: u32,
    /// LDS bytes needed per workgroup
    pub lds_bytes: u32,
    /// Occupancy: waves per SIMD unit
    pub occupancy: u32,
    /// Number of workgroups to cover the full M×N output
    pub n_wgs: u32,
    /// Whether this config fits in hardware limits
    pub feasible: bool,
    /// Performance score (higher is better): TFLOPS estimate
    pub score: f64,
    /// Bottleneck description
    pub bottleneck: &'static str,
}

// ============================================================================
// Cost estimation
// ============================================================================

/// Estimate cost for a given tile configuration on GFX1100.
pub fn estimate_tile_cost(
    config: &TileConfig,
    m: u32, n: u32, k: u32,
    fmt: DataFormat,
    hw: &GFX1100Limits,
) -> TileCost {
    let tile_m = config.tile_m;
    let tile_n = config.tile_n;
    let tile_k = config.tile_k;

    // ── VGPR estimation ──
    // Accumulators: n_wmma_tiles × m_wmma_per_wave × 8 VGPRs (f32 accumulator)
    let n_wmma = config.n_wmma_tiles();
    let m_wmma = config.m_wmma_per_wave();
    let acc_vgprs = n_wmma * m_wmma * 8;

    // Fragments: A fragment (8 VGPRs) + B fragments (n_wmma × 8 VGPRs)
    let frag_a_vgprs = m_wmma * 8;
    let frag_b_vgprs = n_wmma * 8;

    // Address registers: ~6 (x_base, wt_base, k_byte_off, etc.)
    let addr_vgprs = 6u32;

    // Thread ID / lane decomposition: ~4
    let misc_vgprs = 4u32;

    let vgprs = acc_vgprs + frag_a_vgprs + frag_b_vgprs + addr_vgprs + misc_vgprs;

    // ── SGPR estimation ──
    // Kernargs (5 pointers/dims × 2) + TGID (3) + loop counter + misc
    let sgprs = 5u32 + 10 + 5; // ~20 SGPRs typical

    // ── LDS estimation (with bank conflict padding) ──
    let lds_pad = config.lds_pad;
    let lds_bytes = if config.use_lds {
        // A tile: tile_m rows × (tile_k × elem_bytes + lds_pad) per row
        // B tile: tile_n rows × (tile_k × elem_bytes + lds_pad) per row
        let a_bytes = tile_m * (tile_k * fmt.bytes() + lds_pad);
        let b_bytes = tile_n * (tile_k * fmt.bytes() + lds_pad);
        // Double-buffered: × 2
        (a_bytes + b_bytes) * 2
    } else {
        0u32
    };

    // ── Feasibility check ──
    let feasible = vgprs <= hw.max_vgprs
        && sgprs <= hw.max_sgprs
        && lds_bytes <= hw.lds_per_wg;

    // ── Occupancy ──
    // RDNA3 Wave32: 256 VGPRs / SIMD, max 16 waves per SIMD
    let occ_per_simd = if vgprs == 0 { 0 } else {
        let by_vgpr = if vgprs <= 64 { 16u32 }
            else if vgprs <= 96 { 10 }
            else if vgprs <= 128 { 8 }
            else if vgprs <= 192 { 4 }
            else if vgprs <= 256 { 2 }
            else { 0 };
        let by_lds = if lds_bytes == 0 { 16 }
            else { (hw.lds_per_cu / lds_bytes.max(1)).min(16) };
        by_vgpr.min(by_lds)
    };

    // WGP mode: per-config (not hw-global).
    // Each WGP = 2 CUs with simds_per_cu SIMDs each.
    let simds_per_wgp = if config.wgp_mode { hw.simds_per_cu * 2 } else { hw.simds_per_cu };
    let max_waves_per_wgp = occ_per_simd * simds_per_wgp;

    let occupancy = occ_per_simd;

    // Hard constraint: WG's waves must fit in one WGP's capacity.
    let feasible = feasible && config.waves_per_wg <= max_waves_per_wgp;

    // ── Workgroup count (with split-K) ──
    let n_wgs_m = (m + tile_m - 1) / tile_m;
    let n_wgs_n = (n + tile_n - 1) / tile_n;
    let n_wgs = n_wgs_m * n_wgs_n * config.split_k; // split-K multiplies WGs
    let k_per_chunk = k / config.split_k; // each WG processes K/split_k elements

    // Feasibility: K must be divisible by split_k * tile_k
    let feasible = feasible && (k % (config.split_k * tile_k) == 0);

    // ── Performance model (with latency_model integration) ──
    let k_iters = (k_per_chunk + tile_k - 1) / tile_k; // K-iters per WG (split-K aware)

    // Output tiles per WG = (tile_m/16) × (tile_n/16)
    let output_wmma_tiles = (tile_m / 16) * (tile_n / 16);
    let total_wmma_per_wg = output_wmma_tiles * k_iters;

    // Total FLOPs = M × N × K × 2 (fixed for problem size)
    let total_flops: f64 = 2.0 * m as f64 * n as f64 * k as f64;

    // ── Compute cycle estimation ──
    // Try instruction-level analysis first (accurate but requires code generation).
    // Fall back to macro-level model if kernel generation fails.
    let compute_cycles_per_wg: u32;
    
    // Use a thread-local cache to avoid regenerating kernels for the same config
    let kloop = analyze_kloop(config);
    
    if let Some(ref analysis) = kloop {
        // Instruction-level model: use refined cycles from actual K-loop analysis.
        // cycles_per_iter accounts for WMMA pipeline depth and LDS/VALU overlap.
        let cycles_per_iter = analysis.cycles_per_iter;
        // The K-loop body contains TWO phases (double-buffer ping-pong),
        // so each "iteration" of the outer loop covers 2 K-tiles.
        // k_iters counts individual K-tiles, so adjust:
        let effective_iters = (k_iters + 1) / 2; // outer loop iterations
        compute_cycles_per_wg = (cycles_per_iter * effective_iters as f64) as u32;
    } else {
        // Fallback: macro-level model (original formula)
        let wmma_per_wave = total_wmma_per_wg / config.waves_per_wg;
        let wmma_cycles = wmma_per_wave * hw.wmma_cycles;
        let loads_per_wave_per_iter = (tile_m / config.waves_per_wg + tile_n + 31) / 32;
        let overhead_per_iter = loads_per_wave_per_iter + 6u32;
        let overhead_cycles = overhead_per_iter * k_iters;
        compute_cycles_per_wg = wmma_cycles + overhead_cycles / 2;
    }

    // Effective scheduling units
    let n_scheduling_units = if hw.wgp_mode { hw.n_cus / 2 } else { hw.n_cus };
    let wgs_per_unit = max_waves_per_wgp / config.waves_per_wg;
    let total_concurrent_wgs = n_scheduling_units * wgs_per_unit;
    let compute_time_cycles = if total_concurrent_wgs > 0 {
        let batches = (n_wgs as f64 / total_concurrent_wgs as f64).ceil();
        compute_cycles_per_wg as f64 * batches
    } else {
        f64::MAX
    };

    // ── Memory bound: Roofline Model ──
    //
    // Per-WG arithmetic intensity (DRAM-level):
    //   AI = 2 × tile_m × tile_n / ((tile_m + tile_n) × bytes)
    //   This tells us the FLOPs/byte ratio at the DRAM level.
    //
    // Note: AI is independent of K — this is the data reuse property of tiling.
    // The roofline predicts a LOWER BOUND on performance from DRAM BW alone.
    // In practice, L2 cache provides additional reuse that we model as
    // effective_bw = DRAM_bw × (1 + l2_boost_factor).
    //
    // GFX1100 L2: 6 MB across 6 shader arrays. For large GEMMs, L2 provides
    // ~50-70% effective bandwidth boost due to tile-strip sharing.
    let ai_wg = 2.0 * tile_m as f64 * tile_n as f64
        / ((tile_m + tile_n) as f64 * fmt.bytes() as f64);
    
    // Occupancy-dependent DRAM BW efficiency + L2 cache boost
    let dram_bw_eff = match occupancy {
        0 => 0.0,
        1 => 0.55,
        2 => 0.75,
        3..=4 => 0.85,
        5..=8 => 0.92,
        _ => 0.95,
    };
    
    // L2 boost: larger tiles and higher occupancy get more L2 reuse.
    // Empirically, L2 can boost effective BW by 50-80% for large GEMMs on GFX1100.
    // Model: l2_boost increases with tile area (more data sharing per WG)
    let tile_area = tile_m * tile_n;
    let l2_boost = if tile_area >= 128 * 128 {
        0.7  // large tiles: substantial L2 reuse
    } else if tile_area >= 64 * 128 {
        0.5  // medium tiles
    } else if tile_area >= 64 * 64 {
        0.3  // small tiles
    } else {
        0.1  // tiny tiles: minimal reuse
    };
    
    let effective_bw = hw.vmem_bandwidth_gbps * dram_bw_eff * (1.0 + l2_boost); // GB/s
    
    // Roofline ceiling: achievable TFLOPS from memory bandwidth
    let mem_tflops = ai_wg * effective_bw / 1000.0;
    
    // Peak compute ceiling (from WMMA throughput)
    // 96 CUs × 2 SIMDs × 1 WMMA per 4 cycles × 8192 FLOPs per WMMA × 2.5 GHz / 1e12
    // = 96 × 2 × 2.5e9 / 4 × 8192 / 1e12 ≈ 98.3 TFLOPS theoretical
    // Published peak: 123 TFLOPS (AMD spec)
    let peak_tflops = 123.0;
    
    // Also account for compute time from K-loop analysis
    let compute_tflops = if compute_time_cycles > 0.0 {
        total_flops / (compute_time_cycles / (hw.clock_ghz * 1e9)) / 1e12
    } else {
        peak_tflops
    };
    
    // Effective TFLOPS = min(compute, memory roofline)
    let effective_tflops = compute_tflops.min(mem_tflops).min(peak_tflops);
    
    // Convert back to time for bottleneck analysis
    let time_sec = if effective_tflops > 0.0 {
        total_flops / (effective_tflops * 1e12)
    } else {
        f64::MAX
    };
    let _mem_time_cycles = time_sec * hw.clock_ghz * 1e9;

    // Bottleneck: which ceiling is binding?
    let bottleneck = if effective_tflops >= peak_tflops - 0.1 {
        "peak"
    } else if mem_tflops < compute_tflops {
        "memory"
    } else {
        "compute"
    };

    // Score = predicted achievable TFLOPS (from roofline)
    let score = if feasible { effective_tflops } else { 0.0 };

    TileCost {
        config: config.clone(),
        vgprs,
        sgprs,
        lds_bytes,
        occupancy,
        n_wgs,
        feasible,
        score,
        bottleneck,
    }
}

// ============================================================================
// Auto-scheduling: exhaustive search
// ============================================================================

/// Auto-schedule GEMM: search tile space and return the best configuration.
///
/// Searches over:
/// - tile_m ∈ {16, 32, 64, 128}
/// - tile_n ∈ {32, 64, 128}
/// - tile_k ∈ {16, 32}
/// - waves_per_wg ∈ {1, 2, 4, 8}
/// - split_k ∈ {1, 2, 4, 8}
/// - wgp_mode ∈ {false, true}
/// - swap_grid ∈ {false, true}
///
/// Returns ordered list of feasible configs, best first.
pub fn auto_schedule_gemm(
    m: u32, n: u32, k: u32,
    fmt: DataFormat,
) -> Vec<TileCost> {
    let hw = GFX1100Limits::default();

    let tile_m_candidates = [32u32, 64, 128];
    let tile_n_candidates = [64u32]; // 128 maps to 64+2pass (scoring mismatch); keep single-pass only
    let tile_k_candidates = [16u32, 32];
    let waves_candidates = [2u32, 4, 8];
    let split_k_candidates = [1u32, 2, 4, 8];

    let mut results = Vec::new();
    let lds_pad_candidates = [0u32, 4, 8];

    for &tile_m in &tile_m_candidates {
        for &tile_n in &tile_n_candidates {
            for &tile_k in &tile_k_candidates {
                for &waves in &waves_candidates {
                    // Constraint: tile_m must be divisible by waves × 16 (WMMA rows)
                    if tile_m < waves * 16 { continue; }
                    if tile_m % (waves * 16) != 0 { continue; }
                    if tile_n % 16 != 0 { continue; }
                    if fmt.has_wmma() && tile_k < hw.wmma_k { continue; }

                    for &sk in &split_k_candidates {
                        // K must be divisible by split_k * tile_k
                        if k % (sk * tile_k) != 0 { continue; }
                        if sk > k / tile_k { continue; }

                        for &wgp in &[false, true] {
                            for &swap in &[true, false] {
                                for &pad in &lds_pad_candidates {
                                    let config = TileConfig {
                                        tile_m, tile_n, tile_k,
                                        waves_per_wg: waves,
                                        use_lds: true,
                                        split_k: sk,
                                        wgp_mode: wgp,
                                        swap_grid: swap,
                                        lds_pad: pad,
                                    };
                                    let cost = estimate_tile_cost(&config, m, n, k, fmt, &hw);
                                    if cost.feasible {
                                        // Verify final GemmConfig fits in 256 VGPRs
                                        // (to_gemm_config may split tile_n into multi-pass)
                                        let gc = config.to_gemm_config();
                                        if !gc.is_feasible() { continue; }
                                        results.push(cost);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort by score descending (best first)
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Get the single best GEMM tile configuration.
/// Returns None if no feasible configuration exists.
pub fn best_gemm_config(m: u32, n: u32, k: u32, fmt: DataFormat) -> Option<TileCost> {
    auto_schedule_gemm(m, n, k, fmt).into_iter().next()
}

/// Print a comparison table of top N tile configurations.
pub fn print_schedule_report(m: u32, n: u32, k: u32, fmt: DataFormat, top_n: usize) {
    let results = auto_schedule_gemm(m, n, k, fmt);

    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Auto-schedule GEMM: M={}, N={}, K={}, {:?}{}║",
        m, n, k, fmt, " ".repeat(64usize.saturating_sub(48 + format!("{m}{n}{k}").len())));
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!("║ {:>3} {:>3} {:>3} {:>4}  {:>4} {:>4} {:>5} {:>4} {:>6} {:>8} ║",
        "tM", "tN", "tK", "WPG", "VGPR", "LDS", "Occ", "WGs", "Score", "Bound");
    eprintln!("╟──────────────────────────────────────────────────────────────────╢");

    for (i, cost) in results.iter().take(top_n).enumerate() {
        let c = &cost.config;
        eprintln!("║ {:>3} {:>3} {:>3} {:>3}w sk{:<2} {:>4} {:>4} {:>4}w {:>5} {:>5.1}T {:>8} ║{}",
            c.tile_m, c.tile_n, c.tile_k, c.waves_per_wg, c.split_k,
            cost.vgprs,
            if cost.lds_bytes > 0 { format!("{}K", cost.lds_bytes / 1024) } else { "-".into() },
            cost.occupancy, cost.n_wgs,
            cost.score, cost.bottleneck,
            if i == 0 { format!(" ← BEST ({})", c.name()) } else { String::new() });
    }

    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
    eprintln!("Searched {} feasible configs", results.len());
}

// ============================================================================
// TileConfig → GemmConfig conversion (Autotune bridge)
// ============================================================================

impl TileConfig {
    /// Convert cost-model TileConfig to gemm_gen GemmConfig.
    ///
    /// This bridges the gap between the cost_model exhaustive search and
    /// the gemm_gen code generator.
    pub fn to_gemm_config(&self) -> super::gemm_gen::GemmConfig {
        let wg_size = self.waves_per_wg * 32;
        // When tile_n > 64, use multi-pass (n_col_passes=2) to stay within
        // GFX1100's 256-VGPR limit. Single-pass tile_n=128 needs ~273 VGPRs
        // (acc=128 + wt_frag=64 + x_frag=16 + gmem=16 + temps=49).
        // Multi-pass: tile_n=64 per pass → acc=64, wt_frag=32 → ~177 VGPRs.
        let (effective_tile_n, n_col_passes) = if self.tile_n > 64 {
            (64, (self.tile_n / 64) as u32)
        } else {
            (self.tile_n, 1)
        };
        super::gemm_gen::GemmConfig {
            tile_m: self.tile_m,
            tile_n: effective_tile_n,
            tile_k: self.tile_k,
            wg_size,
            use_lds: self.use_lds,
            double_buffer: self.use_lds,  // always double-buffer when using LDS
            split_k: if self.split_k > 1 { Some(self.split_k) } else { None },
            lds_pad: self.lds_pad,
            n_col_passes,
            swap_grid: self.swap_grid,
            wgp_mode: self.wgp_mode,
            transpose: super::gemm_gen::GemmTranspose::NT,  // default
            epilogue: super::gemm_gen::EpilogueOp::StoreF32,
        }
    }
}

/// CPU-only: predict the best GemmConfig for given dimensions using cost model.
///
/// This is the primary autotune entry point — no GPU needed.
/// Internally runs `auto_schedule_gemm` and converts the top result.
///
/// # Example
/// ```rust,ignore
/// let cfg = cost_model::predict_best(1024, 1024, 1024);
/// let kernel = gemm_gen::generate(&cfg);
/// ```
pub fn predict_best(m: u32, k: u32, n: u32) -> super::gemm_gen::GemmConfig {
    let results = auto_schedule_gemm(m, n, k, DataFormat::BF16);
    if let Some(best) = results.first() {
        best.config.to_gemm_config()
    } else {
        // Fallback: use gemm_gen's hand-tuned heuristic
        super::gemm_gen::auto_select_legacy(m, k, n)
    }
}

/// Override cost model scoring with actual measured data from PGO.
///
/// Takes a `TuneResult` from `ProfileTuner` and replaces the theoretical
/// cost model's scoring with the measured wall-clock nanoseconds.
/// Returns the best config based on actual hardware measurements.
///
/// # Integration Pattern
/// ```rust,ignore
/// // 1. First run: use theoretical model for initial candidate set
/// let candidates = auto_schedule_gemm(m, n, k, DataFormat::BF16);
///
/// // 2. Profile top candidates on actual hardware
/// let tune_result = tuner.tune_workgroup_size("gemm_1024", &rt, &builder, &[64, 128, 256], n);
///
/// // 3. Override: select best based on measured cycles
/// let best = with_measured_override(&candidates, &tune_result);
/// ```
pub fn with_measured_override(
    costs: &[TileCost],
    measured: &super::profile_guided::TuneResult,
) -> Option<TileCost> {
    // Build a map from wg_size → measured_cycles
    let mut measured_map: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for &(wg, cycles) in &measured.measurements {
        measured_map.insert(wg, cycles);
    }

    // Find the cost with matching WG size that has the lowest measured cycles
    let mut best: Option<(TileCost, u64)> = None;

    for cost in costs {
        let wg = cost.config.wg_threads();
        if let Some(&cycles) = measured_map.get(&wg) {
            match &best {
                None => best = Some((cost.clone(), cycles)),
                Some((_, best_cycles)) => {
                    if cycles < *best_cycles {
                        best = Some((cost.clone(), cycles));
                    }
                }
            }
        }
    }

    best.map(|(mut cost, cycles)| {
        // Override score with measured performance
        let time_sec = cycles as f64 / 1e9; // ns → sec
        let m = 1024.0; // placeholder — actual dims would be passed in practice
        let total_flops = 2.0 * m * m * m;
        cost.score = total_flops / time_sec / 1e12;
        cost.bottleneck = "measured";
        cost
    })
}

// ============================================================================
// Instruction-level K-loop Analysis (insn_latency integration)
// ============================================================================

/// Instruction-level K-loop statistics from actual code generation.
///
/// Unlike the macro-level `estimate_tile_cost`, this generates the REAL
/// GEMM kernel for a given config and analyzes the K-loop body instruction
/// by instruction using the `insn_latency` model.
#[derive(Clone, Debug)]
pub struct KLoopAnalysis {
    pub total_ops: usize,
    pub valu: u32,
    pub vtrans: u32,
    pub wmma: u32,
    pub vmem_loads: u32,
    pub vmem_stores: u32,
    pub lds: u32,
    pub salu: u32,
    pub ctrl: u32,
    /// Total issue cycles (sum of all instruction issue costs)
    pub issue_cycles: u32,
    /// Estimated critical path through the K-loop body
    pub critical_path: u32,
    /// ILP potential: fraction of WMMA latency that can be hidden [0.0, 1.0]
    pub ilp_potential: f32,
    /// Primary bottleneck
    pub bottleneck: &'static str,
    /// Refined cycles per K-loop iteration (accounting for overlap)
    pub cycles_per_iter: f64,
}

/// Generate a GEMM kernel from a TileConfig and analyze its K-loop body.
///
/// Returns None if the kernel cannot be generated or no K-loop is found.
pub fn analyze_kloop(config: &TileConfig) -> Option<KLoopAnalysis> {
    use super::insn_latency;
    use super::ir::Op;

    let gemm_cfg = config.to_gemm_config();
    // Skip configs that exceed VGPR limit (would panic in generate)
    if !gemm_cfg.is_feasible() { return None; }
    let kernel = super::gemm_gen::generate(&gemm_cfg);
    let ops = kernel.ops();

    // Find the K-loop body between "ggen_loop" label and back-edge branch
    let mut loop_start = None;
    let mut loop_end = None;
    for (i, op) in ops.iter().enumerate() {
        if let Op::Label(name) = op {
            if name.starts_with("ggen_loop") { loop_start = Some(i + 1); }
        }
        if let Op::Branch(target) = op {
            if target.starts_with("ggen_loop") { loop_end = Some(i); }
        }
        if let Op::BranchScc1(target) = op {
            if target.starts_with("ggen_loop") { loop_end = Some(i); }
        }
    }

    let (start, end) = match (loop_start, loop_end) {
        (Some(s), Some(e)) if s < e => (s, e),
        _ => return None,
    };

    let loop_body = &ops[start..end];
    let stats = insn_latency::analyze_block(loop_body);
    let (ilp, bottleneck) = insn_latency::ilp_potential(&stats);

    // ── RDNA3 K-loop Pipeline Model ──
    //
    // RDNA3 CU has 3 concurrent execution pipelines for GEMM:
    //   1. WMMA/Matrix unit: issue rate = 1 per 4 cycles (shared with VALU pipe)
    //   2. Vector ALU (VALU+LDS): issue rate = 1 per cycle
    //   3. VMEM (global loads): issue rate = 1 per cycle
    //
    // In a well-scheduled double-buffered K-loop:
    //   Phase A: {LDS read buf0 → WMMA compute} overlapped with {VMEM prefetch buf1}
    //   Phase B: {LDS read buf1 → WMMA compute} overlapped with {VMEM prefetch buf0}
    //
    // The critical path per K-loop iteration is:
    //   max(WMMA_issue_time, VALU_issue_time + LDS_issue_time, VMEM_issue_time)
    //
    // WMMA issue: 4 cycles per WMMA op (they share the VALU pipe but have 4x throughput)
    let wmma_issue = stats.wmma_count as f64 * 4.0;
    
    // LDS issue: 1 cycle per ds_read_b128 operation
    let lds_issue = stats.lds_count as f64;
    
    // VALU + SALU issue: 1 cycle per op (VALU and SALU share same issue slot)
    let alu_issue = stats.valu_count as f64 + stats.salu_count as f64;
    
    // VMEM issue: 1 cycle per global_load_b128 (fully overlapped with compute)
    let vmem_issue = (stats.vmem_load_count + stats.vmem_store_count) as f64;
    
    // Control flow overhead (waitcnt, barriers, loop counter): ~1 cycle each
    let ctrl_issue = stats.ctrl_count as f64;
    
    // WMMA and VALU share the vector issue port — they CANNOT overlap.
    // The total vector pipe demand = WMMA_issue + VALU_issue
    // LDS uses a separate pipe but depends on VALU for address computation.
    // Model: vector_pipe = WMMA_issue + max(LDS_issue, ALU_issue)
    // because LDS reads and VALU interleave in the same phase.
    let vector_pipe = wmma_issue + lds_issue.max(alu_issue);
    
    // VMEM prefetches overlap with compute (double-buffering).
    // Only add VMEM issue cycles if they EXCEED the compute time.
    let compute_time = vector_pipe + ctrl_issue;
    let cycles_per_iter = compute_time.max(vmem_issue);

    Some(KLoopAnalysis {
        total_ops: loop_body.len(),
        valu: stats.valu_count,
        vtrans: stats.vtrans_count,
        wmma: stats.wmma_count,
        vmem_loads: stats.vmem_load_count,
        vmem_stores: stats.vmem_store_count,
        lds: stats.lds_count,
        salu: stats.salu_count,
        ctrl: stats.ctrl_count,
        issue_cycles: stats.total_issue_cycles,
        critical_path: stats.estimated_critical_path,
        ilp_potential: ilp,
        bottleneck,
        cycles_per_iter,
    })
}

/// Print K-loop analysis report for a TileConfig.
pub fn print_kloop_analysis(config: &TileConfig) {
    if let Some(a) = analyze_kloop(config) {
        eprintln!("╔═══════════════════════════════════════════════════╗");
        eprintln!("║  K-loop insn analysis: {:>28}  ║", config.name());
        eprintln!("╠═══════════════════════════════════════════════════╣");
        eprintln!("║  Total: {} ops ({} issue cycles)             ", a.total_ops, a.issue_cycles);
        eprintln!("║  VALU:{:>3}  VTRANS:{:>3}  WMMA:{:>3}              ", a.valu, a.vtrans, a.wmma);
        eprintln!("║  VMEM:{:>3} ld + {:>3} st  LDS:{:>3}               ", a.vmem_loads, a.vmem_stores, a.lds);
        eprintln!("║  SALU:{:>3}  CTRL:{:>3}                           ", a.salu, a.ctrl);
        eprintln!("║  ILP potential: {:.1}% ({})                      ", a.ilp_potential * 100.0, a.bottleneck);
        eprintln!("║  Cycles/iter (refined): {:.0}                    ", a.cycles_per_iter);
        eprintln!("╚═══════════════════════════════════════════════════╝");
    } else {
        eprintln!("[kloop] No K-loop found for {}", config.name());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_limits_default() {
        let hw = GFX1100Limits::default();
        assert_eq!(hw.max_vgprs, 256);
        assert_eq!(hw.n_cus, 96);
        assert_eq!(hw.lds_per_wg, 65536);
    }

    #[test]
    fn test_tile_config_derived() {
        let cfg = TileConfig {
            tile_m: 32, tile_n: 64, tile_k: 16, waves_per_wg: 2, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        assert_eq!(cfg.wg_threads(), 64);
        assert_eq!(cfg.n_wmma_tiles(), 4);
        assert_eq!(cfg.m_wmma_per_wave(), 1);
    }

    #[test]
    fn test_current_default_is_feasible() {
        let config = TileConfig {
            tile_m: 32, tile_n: 64, tile_k: 16, waves_per_wg: 2, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        let hw = GFX1100Limits::default();
        let cost = estimate_tile_cost(&config, 4096, 4096, 512, DataFormat::BF16, &hw);
        assert!(cost.feasible, "hand-tuned config should be feasible");
        assert!(cost.vgprs <= 256, "VGPRs should fit: {}", cost.vgprs);
        assert!(cost.occupancy >= 2, "should have at least 2 waves: {}", cost.occupancy);
        assert!(cost.score > 0.0, "should have positive score: {}", cost.score);
        eprintln!("Hand-tuned config: {} VGPRs, {} occ, {:.1} TFLOPS, {}",
            cost.vgprs, cost.occupancy, cost.score, cost.bottleneck);
    }

    #[test]
    fn test_auto_schedule_returns_results() {
        let results = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
        assert!(!results.is_empty(), "should find at least 1 feasible config");
        let best = &results[0];
        assert!(best.feasible);
        assert!(best.score > 0.0);
        eprintln!("Best: {} waves={} {:.1} TFLOPS ({})",
            best.config.name(), best.config.waves_per_wg, best.score, best.bottleneck);
    }

    #[test]
    fn test_auto_schedule_respects_limits() {
        let results = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
        let hw = GFX1100Limits::default();

        for cost in &results {
            assert!(cost.vgprs <= hw.max_vgprs, "VGPRs {} exceeds {}", cost.vgprs, hw.max_vgprs);
            assert!(cost.lds_bytes <= hw.lds_per_wg, "LDS {} exceeds {}", cost.lds_bytes, hw.lds_per_wg);
        }
    }

    #[test]
    fn test_auto_schedule_small_gemm() {
        // Small GEMM should still find configs
        let results = auto_schedule_gemm(128, 128, 64, DataFormat::BF16);
        assert!(!results.is_empty());

        let best = &results[0];
        eprintln!("Small GEMM best: tile_m={}, tile_n={}, tile_k={}, {:.1} TFLOPS",
            best.config.tile_m, best.config.tile_n, best.config.tile_k, best.score);
    }

    #[test]
    fn test_print_schedule_report() {
        // Smoke test: just ensure it doesn't panic
        print_schedule_report(2048, 2048, 512, DataFormat::BF16, 5);
    }

    #[test]
    fn test_infeasible_config() {
        let config = TileConfig {
            tile_m: 64, tile_n: 128, tile_k: 32, waves_per_wg: 1, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        let hw = GFX1100Limits::default();
        let cost = estimate_tile_cost(&config, 4096, 4096, 512, DataFormat::BF16, &hw);
        if cost.feasible {
            assert!(cost.occupancy <= 4, "large tile should have low occupancy: {}", cost.occupancy);
        }
        eprintln!("Large tile: feasible={}, {} VGPRs, {} occ", cost.feasible, cost.vgprs, cost.occupancy);
    }

    // ── P1-B: LDS Bank Conflict tests ──

    #[test]
    fn test_lds_padding_increases_lds_usage() {
        let hw = GFX1100Limits::default();
        let base = TileConfig {
            tile_m: 128, tile_n: 64, tile_k: 16, waves_per_wg: 4, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        let padded = TileConfig { lds_pad: 4, ..base.clone() };

        let cost_base = estimate_tile_cost(&base, 4096, 4096, 512, DataFormat::BF16, &hw);
        let cost_pad = estimate_tile_cost(&padded, 4096, 4096, 512, DataFormat::BF16, &hw);

        assert!(cost_pad.lds_bytes > cost_base.lds_bytes,
            "padded LDS ({}) should be larger than unpadded ({})",
            cost_pad.lds_bytes, cost_base.lds_bytes);
        eprintln!("LDS: base={} pad4={} (delta=+{})",
            cost_base.lds_bytes, cost_pad.lds_bytes, cost_pad.lds_bytes - cost_base.lds_bytes);
    }

    #[test]
    fn test_search_includes_lds_padded_configs() {
        let results = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
        let has_padded = results.iter().any(|r| r.config.lds_pad > 0);
        assert!(has_padded, "search should include lds_pad > 0 configs");

        let has_pad4 = results.iter().any(|r| r.config.lds_pad == 4);
        let has_pad8 = results.iter().any(|r| r.config.lds_pad == 8);
        eprintln!("Search: total={}, pad4={}, pad8={}", results.len(), has_pad4, has_pad8);
    }

    // ── P1-A: Autotune tests ──

    #[test]
    fn test_tile_config_to_gemm_config() {
        let tc = TileConfig {
            tile_m: 128, tile_n: 64, tile_k: 16, waves_per_wg: 4, use_lds: true,
            split_k: 4, wgp_mode: true, swap_grid: false, lds_pad: 4,
        };
        let gc = tc.to_gemm_config();
        assert_eq!(gc.tile_m, 128);
        assert_eq!(gc.tile_n, 64);
        assert_eq!(gc.tile_k, 16);
        assert_eq!(gc.wg_size, 128);
        assert_eq!(gc.split_k, Some(4));
        assert_eq!(gc.lds_pad, 4);
        assert!(gc.wgp_mode);
        assert!(!gc.swap_grid);
        assert!(gc.use_lds);
        assert!(gc.double_buffer);
    }

    #[test]
    fn test_predict_best_returns_valid_config() {
        let cfg = predict_best(1024, 1024, 1024);
        assert!(cfg.tile_m >= 32, "tile_m should be >= 32: {}", cfg.tile_m);
        assert!(cfg.tile_n >= 32, "tile_n should be >= 32: {}", cfg.tile_n);
        assert!(cfg.tile_k >= 16, "tile_k should be >= 16: {}", cfg.tile_k);
        assert!(cfg.wg_size >= 32, "wg_size should be >= 32: {}", cfg.wg_size);
        eprintln!("predict_best(1024,1024,1024): {}", cfg.name());
    }

    #[test]
    fn test_predict_best_various_sizes() {
        for &(m, k, n) in &[(128, 256, 64), (512, 512, 512), (2048, 2048, 2048), (4096, 4096, 512)] {
            let cfg = predict_best(m, k, n);
            eprintln!("predict_best({},{},{}): {}", m, k, n, cfg.name());
            assert!(cfg.tile_m > 0 && cfg.tile_n > 0 && cfg.tile_k > 0);
        }
    }
}
