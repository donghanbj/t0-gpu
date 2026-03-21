//! Parameterized GEMM kernel generator (T0-Tensile)
//!
//! A single `generate()` function replaces all hand-written GEMM variants
//! by accepting a `GemmConfig` that controls tile sizes, K-unroll factor,
//! LDS strategy, and pipeline depth.
//!
//! # Example
//! ```rust,no_run
//! use t0_gpu::t0::gemm_gen::{GemmConfig, generate};
//! let config = GemmConfig::tile_64x64_k16();
//! let kernel = generate(&config);
//! let elf = kernel.compile(t0_gpu::t0::ir::Target::GFX1100).unwrap();
//! ```

use super::ir::*;
use super::compile::T0Kernel;

// ============================================================================
// Configuration
// ============================================================================

/// GEMM kernel configuration — each combination produces a different optimized kernel.
///
/// Y[M,N] = X[M,K] @ WT[N,K]^T  (bf16 in, f32 accumulate, f32 out)
#[derive(Clone, Debug)]
pub struct GemmConfig {
    /// Output tile rows per workgroup (must be multiple of 32).
    pub tile_m: u32,
    /// Output tile columns per workgroup (must be multiple of 64).
    pub tile_n: u32,
    /// K-dimension tile size per loop iteration (16 or 32).
    pub tile_k: u32,
    /// Workgroup size (threads). Must be 64 × (tile_m / 32).
    pub wg_size: u32,
    /// Use LDS cooperative loading (true) or direct GMEM (false).
    pub use_lds: bool,
    /// Double-buffer LDS (requires use_lds=true).
    pub double_buffer: bool,
    /// Split-K factor (None or Some(1) = no split, Some(2/4/8) = split K dimension)
    pub split_k: Option<u32>,
}

impl GemmConfig {
    /// 16×64, K=16, LDS double-buffered (small-M: 1 wave, max M-parallelism)
    pub fn tile_16x64_k16() -> Self {
        Self { tile_m: 16, tile_n: 64, tile_k: 16, wg_size: 32, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 32×64, K=16, no LDS (equivalent to `matmul`)
    pub fn tile_32x64_direct() -> Self {
        Self { tile_m: 32, tile_n: 64, tile_k: 16, wg_size: 64, use_lds: false, double_buffer: false, split_k: None }
    }
    /// 32×64, K=16, LDS double-buffered (equivalent to `matmul_lds_db`)
    pub fn tile_32x64_k16() -> Self {
        Self { tile_m: 32, tile_n: 64, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 32×64, K=32, LDS double-buffered (small-M optimized with deeper K unroll)
    pub fn tile_32x64_k32() -> Self {
        Self { tile_m: 32, tile_n: 64, tile_k: 32, wg_size: 64, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 32×128, K=16, LDS double-buffered (small-M: wide N for more compute per WG)
    pub fn tile_32x128_k16() -> Self {
        Self { tile_m: 32, tile_n: 128, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 64×64, K=16, LDS double-buffered (equivalent to `matmul_64x64_lds_db`)
    pub fn tile_64x64_k16() -> Self {
        Self { tile_m: 64, tile_n: 64, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 64×64, K=32, LDS double-buffered (equivalent to `matmul_64x64_k32`)
    pub fn tile_64x64_k32() -> Self {
        Self { tile_m: 64, tile_n: 64, tile_k: 32, wg_size: 64, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 128×64, K=16, LDS double-buffered (higher compute density)
    pub fn tile_128x64_k16() -> Self {
        Self { tile_m: 128, tile_n: 64, tile_k: 16, wg_size: 128, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 64×128, K=16, LDS double-buffered (wider N tiles)
    pub fn tile_64x128_k16() -> Self {
        Self { tile_m: 64, tile_n: 128, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None }
    }
    /// 128×64, K=32 (max compute density)
    pub fn tile_128x64_k32() -> Self {
        Self { tile_m: 128, tile_n: 64, tile_k: 32, wg_size: 128, use_lds: true, double_buffer: true, split_k: None }
    }

    /// Number of WMMA column tiles = tile_n / 16
    pub fn n_col_tiles(&self) -> usize { (self.tile_n / 16) as usize }
    /// Number of row blocks per wave = (tile_m / n_waves) / 16
    pub fn n_row_blocks(&self) -> usize {
        let n_waves = self.wg_size / 32;
        let rows_per_wave = self.tile_m / n_waves;
        (rows_per_wave / 16) as usize
    }
    /// Number of waves per workgroup
    pub fn n_waves(&self) -> u32 { self.wg_size / 32 }
    /// Rows per wave = tile_m / n_waves (must be multiple of 16)
    pub fn rows_per_wave(&self) -> u32 { self.tile_m / self.n_waves() }
    /// K sub-steps per tile_k (each WMMA handles K=16)
    pub fn k_sub_steps(&self) -> u32 { self.tile_k / 16 }
    /// LDS bytes per buffer (X + WT region)
    pub fn lds_per_buffer(&self) -> u32 {
        let x_bytes = self.tile_m * self.tile_k * 2;  // M rows × K cols × bf16
        let wt_bytes = self.tile_n * self.tile_k * 2;
        x_bytes + wt_bytes
    }
    /// Total LDS bytes (single or double buffered)
    pub fn lds_total(&self) -> u32 {
        if self.double_buffer { self.lds_per_buffer() * 2 } else { self.lds_per_buffer() }
    }
    /// LDS X region size per buffer
    pub fn lds_x_size(&self) -> u32 { self.tile_m * self.tile_k * 2 }
    /// LDS WT region size per buffer
    pub fn lds_wt_size(&self) -> u32 { self.tile_n * self.tile_k * 2 }
    /// LDS row stride in bytes for X (tile_k × 2)
    pub fn lds_x_row_stride(&self) -> u32 { self.tile_k * 2 }
    /// LDS row stride in bytes for WT (tile_k × 2)
    pub fn lds_wt_row_stride(&self) -> u32 { self.tile_k * 2 }
    /// Bytes per global load per thread for X
    pub fn x_bytes_per_thread(&self) -> u32 { self.tile_m * self.tile_k * 2 / self.wg_size }
    /// Bytes per global load per thread for WT
    pub fn wt_bytes_per_thread(&self) -> u32 { self.tile_n * self.tile_k * 2 / self.wg_size }
    /// Total WMMA operations per K-tile
    pub fn wmma_per_k_tile(&self) -> usize {
        self.n_row_blocks() * self.n_col_tiles() * self.k_sub_steps() as usize
    }
    /// Descriptive name
    pub fn name(&self) -> String {
        let lds_tag = if self.use_lds {
            if self.double_buffer { "_ldsdb" } else { "_lds" }
        } else { "_direct" };
        let sk_tag = match self.split_k {
            Some(sk) if sk > 1 => format!("_sk{}", sk),
            _ => String::new(),
        };
        format!("t0_gemm_{}x{}_k{}{}{}", self.tile_m, self.tile_n, self.tile_k, lds_tag, sk_tag)
    }
}

/// Predefined sweep search space — configurations to benchmark
pub fn sweep_configs() -> Vec<GemmConfig> {
    vec![
        GemmConfig::tile_16x64_k16(),
        GemmConfig::tile_32x64_k16(),
        GemmConfig::tile_32x64_k32(),
        GemmConfig::tile_32x128_k16(),
        GemmConfig::tile_64x64_k16(),
        GemmConfig::tile_64x64_k32(),
        GemmConfig::tile_128x64_k16(),
        GemmConfig::tile_64x128_k16(),
        GemmConfig::tile_128x64_k32(),
    ]
}

// ============================================================================
// Generator
// ============================================================================

/// Generate a GEMM kernel from configuration.
///
/// Returns (kernel, lds_size, workgroup_size, grid_fn) where grid_fn
/// computes grid dimensions for a given (M, N).
pub fn generate(cfg: &GemmConfig) -> T0Kernel {
    if cfg.use_lds && cfg.double_buffer {
        generate_lds_db(cfg)
    } else {
        generate_direct(cfg)
    }
}

/// Returns (grid_x, grid_y) for a given (M, N) based on this config.
pub fn compute_grid(cfg: &GemmConfig, m: u32, n: u32) -> (u32, u32) {
    let tiles_m = m / cfg.tile_m;
    let tiles_n = n / cfg.tile_n;
    // Swizzled: TGID.x → tile_row, TGID.y → tile_col
    // Grid X iterates M-rows first for L2 locality on X data
    (tiles_m * cfg.wg_size, tiles_n)
}

/// Grid with split-K: Y dimension = tiles_n * split_k
pub fn compute_grid_split_k(cfg: &GemmConfig, m: u32, n: u32, split_k: u32) -> (u32, u32) {
    let tiles_m = m / cfg.tile_m;
    let tiles_n = n / cfg.tile_n;
    (tiles_m * cfg.wg_size, tiles_n * split_k)
}

// ============================================================================
// Auto-Select: Pick optimal GemmConfig for given matrix dimensions
// ============================================================================

/// Select the optimal GEMM kernel configuration for given matrix dimensions.
///
/// Based on empirical sweep data (RX 7900 XTX, GFX1100):
/// - Tiny squares (≤512): split_k=4 k32 for CU fill
/// - Medium squares (1024): 64×64 k16
/// - Large squares (≥2048): k32 split2 or 128×64 k32
/// - Rectangular (small M, big K): split_k=8 k16
pub fn auto_select(m: u32, k: u32, n: u32) -> GemmConfig {
    let mn = (m as u64) * (n as u64);

    // Small M with large K/N: use smaller tile_m for more M-tiles → more WGs
    if m <= 64 && k >= 512 {
        // Tiny M: 16×64 tile, each WG = 1 wave (32 threads)
        // M=64 → 4 M-tiles × (N/64) N-tiles → plenty of WGs without split-K
        let tiles = (m / 16) as u64 * (n / 64) as u64;
        if tiles < 96 {
            GemmConfig { split_k: Some(4), ..GemmConfig::tile_16x64_k16() }
        } else {
            GemmConfig::tile_16x64_k16()
        }
    } else if m <= 256 && k >= 1024 {
        // Small M: 32×64 tile with k32 for deeper unroll
        // M=128 → 4 M-tiles, M=256 → 8 M-tiles
        let tiles = (m / 32) as u64 * (n / 64) as u64;
        if tiles >= 96 {
            // Enough WGs to fill CUs without split-K
            GemmConfig::tile_32x64_k32()
        } else {
            // Need split-K to fill CUs
            GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k32() }
        }
    } else if mn <= 512 * 512 {
        // Tiny square: need more WGs to fill 96 CUs
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k32() }
    } else if mn <= 1024 * 1024 {
        // Medium: balanced config
        GemmConfig::tile_64x64_k16()
    } else if mn <= 2048 * 2048 {
        // Large-medium: k32 with split2
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_64x64_k32() }
    } else if k >= 4096 && m >= 512 {
        // Rectangular large M, big K: split_k=4 k16
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() }
    } else {
        // Very large: 128×64 k32 for best compute density
        GemmConfig::tile_128x64_k32()
    }
}

/// Standard set of pre-selected configs covering all common use cases.
pub fn standard_configs() -> Vec<GemmConfig> {
    vec![
        GemmConfig::tile_16x64_k16(),
        GemmConfig::tile_32x64_k16(),
        GemmConfig::tile_32x64_k32(),
        GemmConfig::tile_32x128_k16(),
        GemmConfig::tile_64x64_k16(),
        GemmConfig::tile_64x64_k32(),
        GemmConfig::tile_128x64_k32(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k32() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_16x64_k16() },
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_64x64_k32() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k32() },
        GemmConfig { split_k: Some(8), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_128x64_k32() },
    ]
}

/// Compute grid dimensions for a config, handling split-K automatically.
pub fn compute_grid_auto(cfg: &GemmConfig, m: u32, n: u32) -> (u32, u32) {
    let sk = cfg.split_k.unwrap_or(1);
    if sk > 1 {
        compute_grid_split_k(cfg, m, n, sk)
    } else {
        compute_grid(cfg, m, n)
    }
}

/// Build 40-byte kernarg buffer for a GEMM dispatch.
///
/// Layout: [X:u64, WT:u64, Y:u64, K:u32, N:u32, split_k_shift:u32, y_split_stride:u32]
pub fn build_kernargs(
    x_addr: u64, wt_addr: u64, y_addr: u64,
    k: u32, n: u32, m: u32,
    cfg: &GemmConfig,
) -> [u8; 40] {
    let sk = cfg.split_k.unwrap_or(1);
    let y_stride = if sk > 1 { m * n * 4 } else { 0 };
    let mut ka = [0u8; 40];
    ka[0..8].copy_from_slice(&x_addr.to_le_bytes());
    ka[8..16].copy_from_slice(&wt_addr.to_le_bytes());
    ka[16..24].copy_from_slice(&y_addr.to_le_bytes());
    ka[24..28].copy_from_slice(&k.to_le_bytes());
    ka[28..32].copy_from_slice(&n.to_le_bytes());
    ka[32..36].copy_from_slice(&0u32.to_le_bytes());
    ka[36..40].copy_from_slice(&y_stride.to_le_bytes());
    ka
}

#[cfg(test)]
mod auto_select_tests {
    use super::*;

    #[test]
    fn test_auto_select_sizes() {
        // Small square → split_k
        let c = auto_select(256, 256, 256);
        assert!(c.split_k.unwrap_or(1) > 1, "small should use split-K");

        // Medium → 64×64 k16
        let c = auto_select(1024, 1024, 1024);
        assert_eq!(c.tile_m, 64);
        assert_eq!(c.tile_k, 16);

        // Very large → split_k=4 k16 (k>=4096, m>=512)
        let c = auto_select(8192, 8192, 8192);
        assert!(c.split_k.is_some(), "large should use split-K");

        // Rectangular small M → 32×64 k32 (enough tiles for CU fill)
        let c = auto_select(128, 4096, 4096);
        assert_eq!(c.tile_m, 32);
    }
}

// ============================================================================
// LDS Double-Buffered Generator (handles any tile_m × tile_n × tile_k)
// ============================================================================

fn generate_lds_db(cfg: &GemmConfig) -> T0Kernel {
    let mut k = T0Kernel::new(&cfg.name());
    let n_col_tiles = cfg.n_col_tiles();
    let n_row_blocks = cfg.n_row_blocks();
    let n_waves = cfg.n_waves();
    let rows_per_wave = cfg.rows_per_wave();

    let lds_x = cfg.lds_x_size();
    let lds_wt = cfg.lds_wt_size();
    let lds_buf = lds_x + lds_wt;  // per buffer
    k.set_lds_size(lds_buf * 2);   // double buffer

    let x_row_stride = cfg.lds_x_row_stride();   // tile_k * 2 bytes
    let wt_row_stride = cfg.lds_wt_row_stride();

    // How many b128 loads per thread for X and WT
    let x_loads_per_thread = cfg.x_bytes_per_thread() / 16;  // 16 bytes per b128
    let wt_loads_per_thread = cfg.wt_bytes_per_thread() / 16;

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    let split_k_shift_arg = k.arg_u32("split_k_shift");  // reserved (unused, kept for layout)
    let y_split_stride_arg = k.arg_u32("y_split_stride"); // M*N*4 for split, 0 for normal
    k.emit_arg_loads();

    // ── TGIDs with split-K support ──
    // Swizzled grid: TGID.x → tile_row, TGID.y → tile_col (possibly packed with split_k_id)
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_row_s);

    let tgid_y_s = k.alloc_sreg();
    k.capture_tgid_y(tgid_y_s);

    // Extract tile_col and split_k_id from TGID.y using compile-time shift
    let split_k = cfg.split_k.unwrap_or(1);
    let split_k_shift: u8 = match split_k { 1 => 0, 2 => 1, 4 => 2, 8 => 3, _ => panic!("unsupported split_k") };

    let tile_col_s = k.alloc_sreg();
    let split_k_id_s = k.alloc_sreg();
    if split_k <= 1 {
        // No split: tile_col = TGID.y, split_k_id = 0
        k.push(Op::SAddU32 { dst: tile_col_s, src0: tgid_y_s, src1: SOperand::InlineInt(0) });
        k.s_mov_imm(split_k_id_s, 0);
    } else {
        // tile_col = TGID.y >> split_k_shift
        k.s_lshr_b32(tile_col_s, tgid_y_s, split_k_shift);
        // split_k_id = TGID.y & (split_k - 1)
        let mask_s = k.alloc_sreg();
        k.s_mov_imm(mask_s, (split_k - 1) as i32);
        k.s_and_b32(split_k_id_s, tgid_y_s, mask_s);
    }

    // k_end = K >> split_k_shift (= K when no split)
    let k_end_s = k.alloc_sreg();
    if split_k <= 1 {
        k.push(Op::SAddU32 { dst: k_end_s, src0: SReg(k_dim.0), src1: SOperand::InlineInt(0) });
    } else {
        k.s_lshr_b32(k_end_s, SReg(k_dim.0), split_k_shift);
    }

    // k_start_bytes = split_k_id * k_end * 2
    let k_start_bytes_s = k.alloc_sreg();
    if split_k <= 1 {
        k.s_mov_imm(k_start_bytes_s, 0);
    } else {
        let s_tmp = k.alloc_sreg();
        k.s_mul_i32(s_tmp, split_k_id_s, k_end_s);
        k.s_lshl_b32(k_start_bytes_s, s_tmp, 1); // * 2 for bf16
    }

    // Y offset = split_k_id * y_split_stride
    let y_offset_s = k.alloc_sreg();
    if split_k <= 1 {
        k.s_mov_imm(y_offset_s, 0);
    } else {
        k.s_mul_i32(y_offset_s, split_k_id_s, SReg(y_split_stride_arg.0));
    }

    // ── Thread decomposition ──
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators: n_row_blocks × n_col_tiles × 8 VGPRs ──
    let mut acc = Vec::new();
    for _r in 0..n_row_blocks {
        for _c in 0..n_col_tiles {
            acc.push(k.alloc_vreg_array(8, Alignment::Align8));
        }
    }
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── Store phase constants ──
    // s_row_base[r] for each row block
    let tile_m_shift = match cfg.tile_m {
        16 => 4, 32 => 5, 64 => 6, 128 => 7, _ => panic!("unsupported tile_m"),
    };
    let rpw_shift = match rows_per_wave {
        16 => 4, 32 => 5, 64 => 6, _ => panic!("unsupported rows_per_wave"),
    };
    let mut s_row_bases = Vec::new();
    let s_row_base0 = k.alloc_sreg();
    k.s_lshl_b32(s_row_base0, tile_row_s, tile_m_shift);
    let s_tmp = k.alloc_sreg();
    k.s_lshl_b32(s_tmp, wave_id_s, rpw_shift);
    k.push(Op::SAddU32 { dst: s_row_base0, src0: s_row_base0, src1: SOperand::SReg(s_tmp) });
    s_row_bases.push(s_row_base0);
    for r in 1..n_row_blocks {
        let rb = k.alloc_sreg();
        k.push(Op::SAddU32 {
            dst: rb, src0: s_row_base0,
            src1: SOperand::InlineInt((r * 16) as i32),
        });
        s_row_bases.push(rb);
    }

    let base_n_s = k.alloc_sreg();
    let tile_n_shift = match cfg.tile_n {
        32 => 5, 64 => 6, 128 => 7, _ => panic!("unsupported tile_n"),
    };
    k.s_lshl_b32(base_n_s, tile_col_s, tile_n_shift);

    // ══════════════════════════════════════════════════════════════════
    // COOPERATIVE LOAD ADDRESSES
    // ══════════════════════════════════════════════════════════════════

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));

    // ── X cooperative load address (COALESCED) ──
    let xrs_shift = match x_row_stride { 32 => 5, 64 => 6, 128 => 7, _ => panic!("xrs") };
    let wrs_shift = match wt_row_stride { 32 => 5, 64 => 6, 128 => 7, _ => panic!("wrs") };
    // Thread t loads from byte offset (t * 16) within the tile's GMEM footprint.
    // With x_row_stride bytes per row, thread t corresponds to:
    //   row_in_tile = (t * 16) / x_row_stride
    //   col_byte    = (t * 16) % x_row_stride
    // Adjacent threads access adjacent 16-byte chunks → coalesced within cache lines!
    //
    // For tile_k=16, x_row_stride=32: threads 2t,2t+1 share the same row.
    // For tile_k=32, x_row_stride=64: threads 4t..4t+3 share the same row.
    let chunks_per_row_x = x_row_stride / 16;  // 2 for k16, 4 for k32
    let x_cpr_shift = match chunks_per_row_x { 2 => 1, 4 => 2, 8 => 3, _ => panic!("cpr_x") };

    let x_row_in_tile = k.alloc_vreg();
    k.v_lshrrev_b32(x_row_in_tile, x_cpr_shift, tid);  // tid / chunks_per_row
    let x_col_chunk = k.alloc_vreg();
    k.v_and_b32_imm(x_col_chunk, tid, chunks_per_row_x - 1);  // tid % chunks_per_row

    // GMEM address: X_ptr + (tile_row * tile_m + x_row_in_tile) * K * 2 + k_byte_off + x_col_chunk * 16
    let x_abs_row = k.alloc_vreg();
    let s_xbase = k.alloc_sreg();
    k.s_lshl_b32(s_xbase, tile_row_s, tile_m_shift);
    k.v_mov_from_sgpr(x_abs_row, s_xbase);
    k.v_add_u32(x_abs_row, x_abs_row, x_row_in_tile);

    let x_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_byte, x_abs_row, k_vreg);
    k.v_lshlrev_b32(x_row_byte, 1, x_row_byte);
    // Add column offset: x_col_chunk * 16
    let x_col_byte = k.alloc_vreg();
    k.v_lshlrev_b32(x_col_byte, 4, x_col_chunk);  // col_chunk * 16
    k.v_add_u32(x_row_byte, x_row_byte, x_col_byte);

    let x_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_gmem_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_gmem_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_gmem_base, x_gmem_base, x_row_byte);
    k.v_add_co_ci(VReg(x_gmem_base.0 + 1), VReg(x_gmem_base.0 + 1));

    // X LDS store addr: row_in_tile * x_row_stride + col_chunk * 16
    let x_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_lds_off, xrs_shift, x_row_in_tile);  // row * row_stride
    k.v_add_u32(x_lds_off, x_lds_off, x_col_byte);          // + col_chunk * 16

    // ── WT cooperative load address (COALESCED) ──
    let chunks_per_row_wt = wt_row_stride / 16;
    let wt_cpr_shift = match chunks_per_row_wt { 2 => 1, 4 => 2, 8 => 3, _ => panic!("cpr_wt") };

    let wt_row_in_tile = k.alloc_vreg();
    k.v_lshrrev_b32(wt_row_in_tile, wt_cpr_shift, tid);
    let wt_col_chunk = k.alloc_vreg();
    k.v_and_b32_imm(wt_col_chunk, tid, chunks_per_row_wt - 1);

    let wt_abs_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_abs_row, base_n_s);
    k.v_add_u32(wt_abs_row, wt_abs_row, wt_row_in_tile);

    let wt_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_byte, wt_abs_row, k_vreg);
    k.v_lshlrev_b32(wt_row_byte, 1, wt_row_byte);
    let wt_col_byte = k.alloc_vreg();
    k.v_lshlrev_b32(wt_col_byte, 4, wt_col_chunk);
    k.v_add_u32(wt_row_byte, wt_row_byte, wt_col_byte);

    let wt_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_gmem_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_gmem_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_gmem_base, wt_gmem_base, wt_row_byte);
    k.v_add_co_ci(VReg(wt_gmem_base.0 + 1), VReg(wt_gmem_base.0 + 1));

    // WT LDS store addr: lds_x + row_in_tile * wt_row_stride + col_chunk * 16
    let wt_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_off, wrs_shift, wt_row_in_tile);
    k.v_add_u32(wt_lds_off, wt_lds_off, wt_col_byte);
    k.push(Op::VAddU32 {
        dst: wt_lds_off, src0: Operand::VReg(wt_lds_off),
        src1: Operand::InlineInt(lds_x as i32),
    });

    // ── LDS read addresses for WMMA fragments ──
    // X frag[r]: (wave_id * rows_per_wave + r*16 + lane_row) * x_row_stride
    let lane_row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(lane_row_stride, xrs_shift, lane_row);

    let s_wave_x_off = k.alloc_sreg();
    let s_wave_stride = k.alloc_sreg();
    k.s_mov_imm(s_wave_stride, (rows_per_wave * x_row_stride) as i32);
    k.s_mul_i32(s_wave_x_off, wave_id_s, s_wave_stride);

    let mut x_lds_reads = Vec::new();
    for r in 0..n_row_blocks {
        let xr = k.alloc_vreg();
        k.v_mov_from_sgpr(xr, s_wave_x_off);
        k.v_add_u32(xr, xr, lane_row_stride);
        if r > 0 {
            k.push(Op::VAddU32 {
                dst: xr, src0: Operand::VReg(xr),
                src1: Operand::InlineInt((r as i32) * 16 * (x_row_stride as i32)),
            });
        }
        x_lds_reads.push(xr);
    }

    // WT: lane_row * wt_row_stride (tile offset as LDS_X + col*16*wt_row_stride)
    let wt_lds_read_base = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_read_base, wrs_shift, lane_row);

    // ── Temp VGPRs (pre-allocated, reused) ──
    let n_x_gmem_regs = x_loads_per_thread as usize;
    let n_wt_gmem_regs = wt_loads_per_thread as usize;
    let gmem_x: Vec<VReg> = (0..n_x_gmem_regs)
        .map(|_| k.alloc_vreg_array(4, Alignment::Align4))
        .collect();
    let gmem_wt: Vec<VReg> = (0..n_wt_gmem_regs)
        .map(|_| k.alloc_vreg_array(4, Alignment::Align4))
        .collect();

    // WMMA fragment registers
    let x_frags: Vec<VReg> = (0..n_row_blocks)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    let wt_frags: Vec<VReg> = (0..n_col_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // Pre-allocated temp addresses (reuse across all macro invocations!)
    let tmp_xa = k.alloc_vreg_array(2, Alignment::Align2);
    let tmp_wa = k.alloc_vreg_array(2, Alignment::Align2);
    let tmp_lds_x = k.alloc_vreg();
    let tmp_lds_w = k.alloc_vreg();
    let tmp_frag_addr = k.alloc_vreg();  // reused for all fragment reads

    // ── GMEM multi-pass strides (coalesced loading) ──
    let x_rows_per_pass = cfg.wg_size / chunks_per_row_x;
    let wt_rows_per_pass = cfg.wg_size / chunks_per_row_wt;
    let x_lds_stride = (x_rows_per_pass * x_row_stride) as i32;
    let wt_lds_stride = (wt_rows_per_pass * wt_row_stride) as i32;

    // GMEM stride = rows_per_pass * K * 2 (K is runtime, so compute as vreg)
    let x_gmem_stride = k.alloc_vreg();
    {
        let tmp = k.alloc_vreg();
        let s_factor = k.alloc_sreg();
        k.s_mov_imm(s_factor, (x_rows_per_pass * 2) as i32);
        k.v_mov_from_sgpr(tmp, s_factor);
        k.v_mul_lo_u32(x_gmem_stride, k_vreg, tmp);
    }
    let wt_gmem_stride = k.alloc_vreg();
    {
        let tmp = k.alloc_vreg();
        let s_factor = k.alloc_sreg();
        k.s_mov_imm(s_factor, (wt_rows_per_pass * 2) as i32);
        k.v_mov_from_sgpr(tmp, s_factor);
        k.v_mul_lo_u32(wt_gmem_stride, k_vreg, tmp);
    }

    // ── K-loop state (with split-K support) ──
    // k_byte_off starts at k_start_bytes (0 for non-split)
    let k_byte_off = k.alloc_vreg();
    k.v_mov_from_sgpr(k_byte_off, k_start_bytes_s);  // start at k_start_bytes
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let k_step = (cfg.tile_k * 2) as i32;

    // k_end_s already computed above from split-K logic

    let buf0_off = k.alloc_vreg();
    k.v_mov_imm(buf0_off, 0);
    let buf1_off = k.alloc_vreg();
    k.v_mov_imm(buf1_off, lds_buf as i32);

    // ══════════════════════════════════════════════════════════════════
    // HELPER CLOSURES (via macros to avoid borrow issues)
    // ══════════════════════════════════════════════════════════════════

    macro_rules! coop_gmem_load {
        ($k:expr, $koff:expr) => {{
            // X loads (coalesced, multi-pass)
            $k.v_mov(tmp_xa, x_gmem_base);
            $k.v_mov(VReg(tmp_xa.0 + 1), VReg(x_gmem_base.0 + 1));
            $k.v_add_co(tmp_xa, tmp_xa, $koff);
            $k.v_add_co_ci(VReg(tmp_xa.0 + 1), VReg(tmp_xa.0 + 1));
            for i in 0..n_x_gmem_regs {
                $k.global_load(gmem_x[i], tmp_xa, Width::B128, 0);
                if i + 1 < n_x_gmem_regs {
                    // Advance to next row block
                    $k.v_add_co(tmp_xa, tmp_xa, x_gmem_stride);
                    $k.v_add_co_ci(VReg(tmp_xa.0 + 1), VReg(tmp_xa.0 + 1));
                }
            }
            // WT loads (coalesced, multi-pass)
            $k.v_mov(tmp_wa, wt_gmem_base);
            $k.v_mov(VReg(tmp_wa.0 + 1), VReg(wt_gmem_base.0 + 1));
            $k.v_add_co(tmp_wa, tmp_wa, $koff);
            $k.v_add_co_ci(VReg(tmp_wa.0 + 1), VReg(tmp_wa.0 + 1));
            for i in 0..n_wt_gmem_regs {
                $k.global_load(gmem_wt[i], tmp_wa, Width::B128, 0);
                if i + 1 < n_wt_gmem_regs {
                    $k.v_add_co(tmp_wa, tmp_wa, wt_gmem_stride);
                    $k.v_add_co_ci(VReg(tmp_wa.0 + 1), VReg(tmp_wa.0 + 1));
                }
            }
        }};
    }

    macro_rules! coop_lds_store {
        ($k:expr, $buf_off:expr) => {{
            $k.v_add_u32(tmp_lds_x, x_lds_off, $buf_off);
            for i in 0..n_x_gmem_regs {
                $k.ds_store_b128(tmp_lds_x, gmem_x[i], 0);
                if i + 1 < n_x_gmem_regs {
                    $k.push(Op::VAddU32 {
                        dst: tmp_lds_x, src0: Operand::VReg(tmp_lds_x),
                        src1: Operand::InlineInt(x_lds_stride),
                    });
                }
            }
            $k.v_add_u32(tmp_lds_w, wt_lds_off, $buf_off);
            for i in 0..n_wt_gmem_regs {
                $k.ds_store_b128(tmp_lds_w, gmem_wt[i], 0);
                if i + 1 < n_wt_gmem_regs {
                    $k.push(Op::VAddU32 {
                        dst: tmp_lds_w, src0: Operand::VReg(tmp_lds_w),
                        src1: Operand::InlineInt(wt_lds_stride),
                    });
                }
            }
        }};
    }

    macro_rules! lds_read_and_wmma {
        ($k:expr, $buf_off:expr) => {{
            for ks in 0..cfg.k_sub_steps() as usize {
                let k_byte_within = (ks * 32) as u16;

                if n_row_blocks >= 2 {
                    // ── INTERLEAVED SCHEDULE ──
                    // Phase 1: Load X[0] + all WT frags
                    $k.v_add_u32(tmp_frag_addr, x_lds_reads[0], $buf_off);
                    $k.ds_load_b128(x_frags[0], tmp_frag_addr, k_byte_within);
                    $k.ds_load_b128(VReg(x_frags[0].0 + 4), tmp_frag_addr, k_byte_within + 16);
                    for c in 0..n_col_tiles {
                        $k.v_add_u32(tmp_frag_addr, wt_lds_read_base, $buf_off);
                        let base_off: u16 = (lds_x + (c as u32) * 16 * wt_row_stride) as u16;
                        $k.ds_load_b128(wt_frags[c], tmp_frag_addr, base_off + k_byte_within);
                        $k.ds_load_b128(VReg(wt_frags[c].0 + 4), tmp_frag_addr, base_off + k_byte_within + 16);
                    }
                    $k.wait_lgkmcnt(0);

                    // Phase 2: WMMA X[0]×WT[all] while loading X[1]
                    // Issue X[1] load BEFORE starting WMMA — hides LDS latency behind compute
                    $k.v_add_u32(tmp_frag_addr, x_lds_reads[1], $buf_off);
                    $k.ds_load_b128(x_frags[1], tmp_frag_addr, k_byte_within);
                    $k.ds_load_b128(VReg(x_frags[1].0 + 4), tmp_frag_addr, k_byte_within + 16);
                    // WMMA with X[0] executes concurrently with X[1] LDS loads
                    for c in 0..n_col_tiles {
                        let a_idx = 0 * n_col_tiles + c;
                        $k.wmma_bf16_f32(acc[a_idx], x_frags[0], wt_frags[c], acc[a_idx]);
                    }
                    $k.wait_lgkmcnt(0);

                    // Phase 3: remaining row blocks
                    for r in 1..n_row_blocks {
                        if r + 1 < n_row_blocks {
                            // Prefetch next X frag
                            $k.v_add_u32(tmp_frag_addr, x_lds_reads[r + 1], $buf_off);
                            $k.ds_load_b128(x_frags[r + 1], tmp_frag_addr, k_byte_within);
                            $k.ds_load_b128(VReg(x_frags[r + 1].0 + 4), tmp_frag_addr, k_byte_within + 16);
                        }
                        for c in 0..n_col_tiles {
                            let a_idx = r * n_col_tiles + c;
                            $k.wmma_bf16_f32(acc[a_idx], x_frags[r], wt_frags[c], acc[a_idx]);
                        }
                        if r + 1 < n_row_blocks {
                            $k.wait_lgkmcnt(0);
                        }
                    }
                } else {
                    // ── SIMPLE SCHEDULE (n_row_blocks == 1) ──
                    $k.v_add_u32(tmp_frag_addr, x_lds_reads[0], $buf_off);
                    $k.ds_load_b128(x_frags[0], tmp_frag_addr, k_byte_within);
                    $k.ds_load_b128(VReg(x_frags[0].0 + 4), tmp_frag_addr, k_byte_within + 16);
                    for c in 0..n_col_tiles {
                        $k.v_add_u32(tmp_frag_addr, wt_lds_read_base, $buf_off);
                        let base_off: u16 = (lds_x + (c as u32) * 16 * wt_row_stride) as u16;
                        $k.ds_load_b128(wt_frags[c], tmp_frag_addr, base_off + k_byte_within);
                        $k.ds_load_b128(VReg(wt_frags[c].0 + 4), tmp_frag_addr, base_off + k_byte_within + 16);
                    }
                    $k.wait_lgkmcnt(0);
                    for c in 0..n_col_tiles {
                        $k.wmma_bf16_f32(acc[0 * n_col_tiles + c], x_frags[0], wt_frags[c], acc[0 * n_col_tiles + c]);
                    }
                }
            }
        }};
    }



    // ══════════════════════════════════════════════════════════════════
    // PROLOGUE
    // ══════════════════════════════════════════════════════════════════
    coop_gmem_load!(k, k_byte_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, cfg.tile_k as i32);

    // ══════════════════════════════════════════════════════════════════
    // MAIN LOOP
    // ══════════════════════════════════════════════════════════════════
    let loop_label = k.make_label("ggen_loop");
    k.label(&loop_label);
    k.s_cmp_ge_u32(k_iter_s, k_end_s);
    let epilog_a = k.make_label("ggen_ea");
    k.branch_scc1(&epilog_a);

    // Phase A: prefetch→buf1, compute buf0
    coop_gmem_load!(k, k_byte_off);
    lds_read_and_wmma!(k, buf0_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf1_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, cfg.tile_k as i32);

    k.s_cmp_ge_u32(k_iter_s, k_end_s);
    let epilog_b = k.make_label("ggen_eb");
    k.branch_scc1(&epilog_b);

    // Phase B: prefetch→buf0, compute buf1
    coop_gmem_load!(k, k_byte_off);
    lds_read_and_wmma!(k, buf1_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, cfg.tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, k_end_s);
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // EPILOGUES
    // ══════════════════════════════════════════════════════════════════
    k.label(&epilog_a);
    lds_read_and_wmma!(k, buf0_off);
    let store_label = k.make_label("ggen_store");
    k.s_mov_imm(k_iter_s, 0);
    k.s_cmp_eq_u32_imm(k_iter_s, 0);
    k.branch_scc1(&store_label);

    k.label(&epilog_b);
    lds_read_and_wmma!(k, buf1_off);

    // ══════════════════════════════════════════════════════════════════
    // STORE PHASE: f32 → global memory
    // ══════════════════════════════════════════════════════════════════
    k.label(&store_label);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);  // N * 8 (2 rows × 4 bytes)

    let col_base_v = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_v, base_n_s);
    k.v_add_u32(col_base_v, col_base_v, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base_v);  // col * 4 bytes

    for r in 0..n_row_blocks {
        let base_row_v = k.alloc_vreg();
        k.v_mov_from_sgpr(base_row_v, s_row_bases[r]);
        k.v_add_u32(base_row_v, base_row_v, lane_half);

        let row_bytes = k.alloc_vreg();
        k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
        k.v_lshlrev_b32(row_bytes, 2, row_bytes);

        for c in 0..n_col_tiles {
            let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
            k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
            k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
            // Add split-K workspace offset (0 when no split)
            {
                let v_yoff = k.alloc_vreg();
                k.v_mov_from_sgpr(v_yoff, y_offset_s);
                k.v_add_co(y_addr, y_addr, v_yoff);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            }
            k.v_add_co(y_addr, y_addr, row_bytes);
            k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            k.v_add_u32(y_addr, y_addr, col_bytes);
            if c > 0 {
                k.push(Op::VAddU32 {
                    dst: y_addr, src0: Operand::VReg(y_addr),
                    src1: Operand::InlineInt((c * 64) as i32),
                });
            }
            let a_idx = r * n_col_tiles + c;
            for vk in 0..8u32 {
                k.global_store(y_addr, VReg(acc[a_idx].0 + vk), Width::B32, 0);
                if vk < 7 {
                    k.v_add_co(y_addr, y_addr, row_stride);
                    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
                }
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Direct (no-LDS) Generator — for reference/small matrices
// ============================================================================

fn generate_direct(cfg: &GemmConfig) -> T0Kernel {
    // Delegate to the LDS version with single-buffer for now
    // TODO: implement zero-LDS variant
    let mut lds_cfg = cfg.clone();
    lds_cfg.use_lds = true;
    lds_cfg.double_buffer = true;
    generate_lds_db(&lds_cfg)
}
