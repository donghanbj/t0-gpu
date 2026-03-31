//! Autotune Oracle: rocBLAS/Triton performance reference data
//!
//! Provides known-good performance targets from industry-standard libraries
//! (rocBLAS and Triton) as a calibration reference for the T0 cost model.
//!
//! Uses:
//! 1. **Benchmarking target**: know how fast we SHOULD be for a given shape
//! 2. **Cost model calibration**: if T0's predicted TFLOPS diverges too far
//!    from the oracle, the cost model needs recalibration
//! 3. **Tile selection override**: for well-known shapes, use the oracle's
//!    recommended tile configuration instead of the cost model's prediction

use super::cost_model::TileConfig;

/// A known performance data point from rocBLAS or Triton.
#[derive(Clone, Debug)]
pub struct OracleEntry {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    /// Best measured TFLOPS across all libraries
    pub best_tflops: f64,
    /// Which library achieved the best result
    pub best_source: &'static str,
    /// rocBLAS TFLOPS (0 if not measured)
    pub rocblas_tflops: f64,
    /// Triton best TFLOPS (0 if not measured)
    pub triton_tflops: f64,
    /// Recommended tile configuration (inferred from performance data)
    pub recommended_tile: TileConfig,
}

/// Oracle database: hardcoded from benchmarks/triton_rocblas_results.csv
///
/// Data source: RX 7900 XTX, ROCm 7.1, Triton + rocBLAS benchmarks.
/// Measured with bf16 inputs, f32 accumulator.
static ORACLE_ENTRIES: &[OracleData] = &[
    // 256³ — small, rocBLAS wins (3.05 TFLOPS)
    OracleData { m: 256, k: 256, n: 256,
        rocblas: 3.05, triton: 2.26, best: 3.05, source: "rocBLAS",
        tile_m: 32, tile_n: 64, tile_k: 16, waves: 2, split_k: 1 },
    // 512³ — Triton-AT wins (17.63 TFLOPS)
    OracleData { m: 512, k: 512, n: 512,
        rocblas: 13.29, triton: 17.63, best: 17.63, source: "Triton-AT",
        tile_m: 64, tile_n: 64, tile_k: 16, waves: 2, split_k: 2 },
    // 1024³ — Triton-AT wins (54.21 TFLOPS)
    OracleData { m: 1024, k: 1024, n: 1024,
        rocblas: 51.96, triton: 54.21, best: 54.21, source: "Triton-AT",
        tile_m: 128, tile_n: 64, tile_k: 16, waves: 4, split_k: 1 },
    // 2048³ — Triton wins (76.82 TFLOPS)
    OracleData { m: 2048, k: 2048, n: 2048,
        rocblas: 71.46, triton: 76.82, best: 76.82, source: "Triton",
        tile_m: 128, tile_n: 128, tile_k: 16, waves: 4, split_k: 1 },
    // 4096³ — rocBLAS wins (90.78 TFLOPS, 73% peak)
    OracleData { m: 4096, k: 4096, n: 4096,
        rocblas: 90.78, triton: 86.71, best: 90.78, source: "rocBLAS",
        tile_m: 128, tile_n: 128, tile_k: 32, waves: 8, split_k: 1 },
    // 128×1024×4096 — tall-skinny, rocBLAS wins (48.47 TFLOPS)
    OracleData { m: 128, k: 1024, n: 4096,
        rocblas: 48.47, triton: 39.93, best: 48.47, source: "rocBLAS",
        tile_m: 128, tile_n: 64, tile_k: 16, waves: 4, split_k: 1 },
    // 256×1024×4096 — Triton-AT wins (56.17 TFLOPS)
    OracleData { m: 256, k: 1024, n: 4096,
        rocblas: 51.47, triton: 56.17, best: 56.17, source: "Triton-AT",
        tile_m: 128, tile_n: 64, tile_k: 16, waves: 4, split_k: 1 },
    // 512×1024×4096 — Triton wins (76.97 TFLOPS)
    OracleData { m: 512, k: 1024, n: 4096,
        rocblas: 74.41, triton: 76.97, best: 76.97, source: "Triton",
        tile_m: 128, tile_n: 128, tile_k: 16, waves: 4, split_k: 1 },
    // 1024×1024×4096 — Triton wins (75.57 TFLOPS)
    OracleData { m: 1024, k: 1024, n: 4096,
        rocblas: 64.40, triton: 75.57, best: 75.57, source: "Triton",
        tile_m: 128, tile_n: 128, tile_k: 16, waves: 4, split_k: 1 },
];

/// Internal data structure for static oracle entries.
struct OracleData {
    m: u32, k: u32, n: u32,
    rocblas: f64, triton: f64, best: f64, source: &'static str,
    tile_m: u32, tile_n: u32, tile_k: u32, waves: u32, split_k: u32,
}

impl OracleData {
    fn to_entry(&self) -> OracleEntry {
        OracleEntry {
            m: self.m, k: self.k, n: self.n,
            best_tflops: self.best,
            best_source: self.source,
            rocblas_tflops: self.rocblas,
            triton_tflops: self.triton,
            recommended_tile: TileConfig {
                tile_m: self.tile_m, tile_n: self.tile_n, tile_k: self.tile_k,
                waves_per_wg: self.waves, use_lds: true,
                split_k: self.split_k, wgp_mode: true,
                swap_grid: true, lds_pad: 0,
            },
        }
    }
}

/// Look up the oracle for an exact shape match.
pub fn oracle_lookup(m: u32, k: u32, n: u32) -> Option<OracleEntry> {
    ORACLE_ENTRIES.iter()
        .find(|e| e.m == m && e.k == k && e.n == n)
        .map(|e| e.to_entry())
}

/// Find the closest oracle entry by shape similarity.
/// Uses geometric mean distance in log-space.
pub fn oracle_nearest(m: u32, k: u32, n: u32) -> OracleEntry {
    let log_target = ((m as f64).ln(), (k as f64).ln(), (n as f64).ln());
    
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    
    for (i, e) in ORACLE_ENTRIES.iter().enumerate() {
        let log_e = ((e.m as f64).ln(), (e.k as f64).ln(), (e.n as f64).ln());
        let dist = (log_target.0 - log_e.0).powi(2)
            + (log_target.1 - log_e.1).powi(2)
            + (log_target.2 - log_e.2).powi(2);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    ORACLE_ENTRIES[best_idx].to_entry()
}

/// Compute the theoretical peak TFLOPS for GFX1100.
///
/// RX 7900 XTX: 96 CUs × 2 SIMDs/CU × 2.5 GHz × 16×16×16 WMMA ops
/// Each WMMA: 16×16×16×2 FLOPs = 8192 FLOPs per op
/// Peak = 96 × 2 × 2.5e9 × (8192 / 36) ≈ 109 TFLOPS (with WMMA pipeline)
///
/// Practical peak: ~123 TFLOPS (AMD spec)
pub fn gfx1100_peak_tflops() -> f64 {
    123.0 // AMD published peak for bf16 GEMM on RX 7900 XTX
}

/// Compute the efficiency of T0's predicted TFLOPS vs the oracle.
///
/// Returns (efficiency, gap_tflops) where:
/// - efficiency = t0_tflops / oracle_best_tflops (0.0 - 1.0+)
/// - gap_tflops = oracle_best - t0_predicted (positive = we're behind)
pub fn compare_with_oracle(m: u32, k: u32, n: u32, t0_predicted_tflops: f64) -> (f64, f64) {
    let oracle = oracle_nearest(m, k, n);
    let efficiency = if oracle.best_tflops > 0.0 {
        t0_predicted_tflops / oracle.best_tflops
    } else {
        1.0
    };
    let gap = oracle.best_tflops - t0_predicted_tflops;
    (efficiency, gap)
}

/// Print a full oracle comparison table.
pub fn print_oracle_comparison(t0_results: &[(u32, u32, u32, f64)]) {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 vs rocBLAS/Triton Oracle                                        ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>6} {:>8} ║",
        "M", "K", "N", "T0", "Oracle", "Best", "Eff%", "Source");
    eprintln!("╟───────────────────────────────────────────────────────────────────────╢");

    for &(m, k, n, t0_tflops) in t0_results {
        let oracle = oracle_nearest(m, k, n);
        let (eff, _gap) = compare_with_oracle(m, k, n, t0_tflops);
        eprintln!("║ {:>6} {:>6} {:>6} {:>7.1}T {:>7.1}T {:>8} {:>5.0}% {:>8} ║",
            m, k, n, t0_tflops, oracle.best_tflops, oracle.best_source,
            eff * 100.0, oracle.recommended_tile.name());
    }

    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_exact_lookup() {
        let entry = oracle_lookup(4096, 4096, 4096);
        assert!(entry.is_some(), "should find 4096³ in oracle");
        let e = entry.unwrap();
        assert_eq!(e.best_source, "rocBLAS");
        assert!((e.best_tflops - 90.78).abs() < 0.1);
        assert_eq!(e.recommended_tile.tile_m, 128);
    }

    #[test]
    fn test_oracle_nearest() {
        // 3000³ should find 2048³ or 4096³ as nearest
        let e = oracle_nearest(3000, 3000, 3000);
        assert!(e.m == 2048 || e.m == 4096,
            "nearest to 3000³ should be 2048³ or 4096³, got {}³", e.m);
        assert!(e.best_tflops > 70.0);
    }

    #[test]
    fn test_oracle_missing_returns_none() {
        let entry = oracle_lookup(999, 999, 999);
        assert!(entry.is_none(), "999³ should not be in oracle");
    }

    #[test]
    fn test_compare_efficiency() {
        let (eff, gap) = compare_with_oracle(4096, 4096, 4096, 72.0);
        assert!(eff > 0.5 && eff < 1.0, "72 vs 90.78 should be ~79%: got {:.0}%", eff * 100.0);
        assert!(gap > 0.0, "gap should be positive (we're behind)");
    }

    #[test]
    fn test_peak_tflops() {
        assert!(gfx1100_peak_tflops() > 100.0);
    }

    #[test]
    fn test_oracle_all_entries_valid() {
        for data in ORACLE_ENTRIES {
            let e = data.to_entry();
            assert!(e.best_tflops > 0.0, "entry {}×{}×{} has zero TFLOPS", e.m, e.k, e.n);
            assert!(e.recommended_tile.tile_m >= 32);
            assert!(e.recommended_tile.tile_n >= 32);
            assert!(e.recommended_tile.tile_k >= 16);
        }
    }

    #[test]
    fn test_cost_model_vs_oracle() {
        // Compare cost model predictions against oracle for key shapes
        use super::super::cost_model;
        
        let shapes = [(1024u32, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)];
        for (m, k, n) in shapes {
            let best = cost_model::best_gemm_config(m, n, k, cost_model::DataFormat::BF16);
            let oracle = oracle_nearest(m, k, n);
            if let Some(cost) = best {
                let (eff, _) = compare_with_oracle(m, k, n, cost.score);
                eprintln!("[oracle] {}×{}×{}: T0={:.1}T oracle={:.1}T ({}) eff={:.0}%",
                    m, k, n, cost.score, oracle.best_tflops, oracle.best_source, eff * 100.0);
                // Don't assert on exact efficiency — cost model is theoretical
            }
        }
    }
}
