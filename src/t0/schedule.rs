//! T0-mid: Scheduling Layer
//!
//! Defines hardware-specific scheduling parameters and kernel generation templates.
//! The `Schedule` trait encodes all optimization decisions for a target GPU,
//! and template functions use these parameters to generate T0Kernel objects.
//!
//! # Architecture
//! ```text
//! Schedule (hardware params) + Template (algorithm) → T0Kernel → compile() → ELF
//! ```

use super::ir::*;
use super::compile::T0Kernel;

// ============================================================================
// Schedule trait
// ============================================================================

/// Memory access strategy for tile loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileLoadStrategy {
    /// Direct global memory load (no LDS). Each thread loads its own data.
    DirectGlobal,
    /// Load to LDS first, then read from LDS. Better for shared patterns.
    ViaLds,
}

/// Scheduling parameters for a specific GPU target.
/// Encodes all optimization decisions that differ between architectures.
pub trait Schedule: std::fmt::Debug {
    /// Target name for display.
    fn name(&self) -> &'static str;

    // ── GEMM parameters ──
    /// Output tile size per workgroup: (M_tile, N_tile).
    fn gemm_tile_mn(&self) -> (usize, usize);
    /// K-dimension tile size (reduction dimension per iteration).
    fn gemm_tile_k(&self) -> usize;
    /// Number of WMMA tiles in N-direction = N_tile / 16.
    fn gemm_n_wmma_tiles(&self) -> usize {
        let (_, n) = self.gemm_tile_mn();
        n / 16
    }
    /// Whether to use WMMA instructions.
    fn use_wmma(&self) -> bool;
    /// WMMA format.
    fn wmma_format(&self) -> WmmaFormat;
    /// Tile loading strategy for A operand.
    fn a_load_strategy(&self) -> TileLoadStrategy;
    /// Tile loading strategy for B operand.
    fn b_load_strategy(&self) -> TileLoadStrategy;

    // ── Workgroup ──
    /// Workgroup size (threads per workgroup).
    fn workgroup_size(&self) -> (u16, u16, u16);
    /// Number of Wave32 waves per workgroup.
    fn waves_per_wg(&self) -> u32 {
        let (x, y, z) = self.workgroup_size();
        ((x as u32 * y as u32 * z as u32) + 31) / 32
    }

    // ── Elementwise ──
    /// Elements processed per thread in elementwise kernels.
    fn elems_per_thread(&self) -> usize;

    // ── LDS ──
    /// LDS size budget in bytes.
    fn lds_budget(&self) -> u32;

    // ── Target ──
    fn target(&self) -> Target;
}

// ============================================================================
// GFX1100 Schedule — RDNA3 / Navi 31
// ============================================================================

/// Scheduling parameters for AMD RX 7900 XTX (GFX1100, RDNA3).
/// Values match the proven hand-tuned kernels in `gemm_forward.rs`.
#[derive(Clone, Debug)]
pub struct GFX1100Schedule;

impl Schedule for GFX1100Schedule {
    fn name(&self) -> &'static str { "GFX1100 (RDNA3)" }
    fn gemm_tile_mn(&self) -> (usize, usize) { (32, 64) }
    fn gemm_tile_k(&self) -> usize { 16 }
    fn use_wmma(&self) -> bool { true }
    fn wmma_format(&self) -> WmmaFormat { WmmaFormat::BF16_F32 }
    fn a_load_strategy(&self) -> TileLoadStrategy { TileLoadStrategy::DirectGlobal }
    fn b_load_strategy(&self) -> TileLoadStrategy { TileLoadStrategy::DirectGlobal }
    fn workgroup_size(&self) -> (u16, u16, u16) { (64, 1, 1) }
    fn elems_per_thread(&self) -> usize { 4 }
    fn lds_budget(&self) -> u32 { 65536 }
    fn target(&self) -> Target { Target::GFX1100 }
}

// ============================================================================
// Kernel Template: Elementwise Scale
// ============================================================================

/// Generate: y[i] = x[i] * scale
///
/// Grid: [ceil(n / wg_size), 1, 1]
/// Kernargs: x_ptr(0), y_ptr(8), scale(16:f32), n(20:u32)
pub fn build_elementwise_scale(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_elementwise_scale");
    let (wg_x, _, _) = sched.workgroup_size();

    // ── Args ──
    let x_ptr = k.arg_ptr("x");
    let y_ptr = k.arg_ptr("y");
    let scale_arg = k.arg_f32("scale");
    let _n_arg = k.arg_u32("n");
    k.emit_arg_loads();

    // ── Global thread ID ──
    let global_id = k.compute_global_id_x(wg_x as u32);

    // ── Load x[global_id] ──
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, global_id);  // * 4 bytes

    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_addr, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_addr.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_addr, x_addr, byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));

    let val = k.alloc_vreg();
    k.global_load(val, x_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // ── Scale ──
    let scale_v = k.alloc_vreg();
    k.v_mov_from_sgpr(scale_v, SReg(scale_arg.0));
    k.v_mul_f32(val, val, scale_v);

    // ── Store y[global_id] ──
    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
    k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
    k.v_add_co(y_addr, y_addr, byte_off);
    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
    k.global_store(y_addr, val, Width::B32, 0);

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Kernel Template: GEMM Forward  Y = X @ W^T
// ============================================================================

/// Generate GEMM forward kernel using schedule's tile/WMMA parameters.
///
/// Grid: [ceil(N/tile_n) * wg_size, ceil(M/tile_m), 1]
/// Kernargs: X_ptr(0), WT_ptr(8), Y_ptr(16), K(24:u32), N(28:u32)
pub fn build_gemm_forward(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_gemm_forward");
    let (tile_m, tile_n) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();  // tile_n / 16

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── Capture TGIDs ──
    // 2D grid: TGID.x = tile_col, TGID.y = tile_row
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    // ── Thread decomposition ──
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);  // lane within wave

    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, VReg(0));   // wave index (0 or 1)

    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });

    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);  // 0..15

    // ── Accumulator allocation ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // Zero accumulators
    for a in &acc {
        for i in 0..8u32 {
            k.v_mov_imm(VReg(a.0 + i), 0);
        }
    }

    // ── X base address ──
    // x_row = tile_row*32 + wave_id*16 + lane_row
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);     // tile_row * 32
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);      // wave_id * 16
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    let x_row = k.alloc_vreg();
    k.v_mov_from_sgpr(x_row, s_tmp1);
    k.v_add_u32(x_row, x_row, lane_row);

    // x_row_bytes = x_row * K * 2 (bf16)
    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));
    let x_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_off, x_row, k_vreg);
    k.v_lshlrev_b32(x_row_off, 1, x_row_off);  // * 2

    let x_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_base, x_base, x_row_off);
    k.v_add_co_ci(VReg(x_base.0 + 1), VReg(x_base.0 + 1));

    // ── WT base offset ──
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);  // tile_col * 64
    let base_n = k.alloc_vreg();
    k.v_mov_from_sgpr(base_n, base_n_s);
    k.v_add_u32(base_n, base_n, lane_row);

    let wt_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_off, base_n, k_vreg);
    k.v_lshlrev_b32(wt_row_off, 1, wt_row_off);

    // ── WMMA fragments ──
    let x_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // ── K-loop ──
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);

    let loop_label = k.make_label("k_loop");
    k.label(&loop_label);

    // Load X fragment: 2× global_load_b128 (8 bf16 values = 16 bytes each)
    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(x_addr, x_base);
    k.v_mov(VReg(x_addr.0 + 1), VReg(x_base.0 + 1));
    k.v_add_co(x_addr, x_addr, k_byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));
    k.global_load(x_frag, x_addr, Width::B128, 0);
    k.global_load(VReg(x_frag.0 + 4), x_addr, Width::B128, 16);

    // Load WT fragments: n_tiles × 2× global_load_b128
    let wt_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_addr, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_addr.0 + 1), SReg(wt_ptr.0 + 1));
    let wt_total_off = k.alloc_vreg();
    k.v_add_u32(wt_total_off, wt_row_off, k_byte_off);
    k.v_add_co(wt_addr, wt_addr, wt_total_off);
    k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));

    // Tile stride = 16 rows * K * 2 bytes
    let tile_stride = k.alloc_vreg();
    k.v_mov_from_sgpr(tile_stride, SReg(k_dim.0));
    k.v_lshlrev_b32(tile_stride, 5, tile_stride);  // K * 32

    for t in 0..n_tiles {
        k.global_load(wt_frags[t], wt_addr, Width::B128, 0);
        k.global_load(VReg(wt_frags[t].0 + 4), wt_addr, Width::B128, 16);
        if t + 1 < n_tiles {
            k.v_add_co(wt_addr, wt_addr, tile_stride);
            k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));
        }
    }

    // Wait for loads
    k.wait_vmcnt(0);

    // WMMA: acc[t] += X_frag * WT_frag[t]
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }

    // K-loop advance
    k.push(Op::VAddU32 {
        dst: k_byte_off,
        src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(tile_k as i32 * 2),  // bf16 bytes
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ── Store results (simplified — full version needs bf16 conversion) ──
    // For now, endpgm. Full store phase will be added in T0-high.
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gfx1100_schedule_params() {
        let sched = GFX1100Schedule;
        assert_eq!(sched.gemm_tile_mn(), (32, 64));
        assert_eq!(sched.gemm_tile_k(), 16);
        assert_eq!(sched.gemm_n_wmma_tiles(), 4);
        assert!(sched.use_wmma());
        assert_eq!(sched.workgroup_size(), (64, 1, 1));
        assert_eq!(sched.waves_per_wg(), 2);
    }

    #[test]
    fn test_build_elementwise_scale() {
        let sched = GFX1100Schedule;
        let kernel = build_elementwise_scale(&sched);
        let asm = kernel.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("global_load_b32"));
        assert!(asm.contains("v_mul_f32"));
        assert!(asm.contains("global_store_b32"));
        assert!(asm.contains("s_endpgm"));
        eprintln!("--- Elementwise scale (T0-mid) ---\n{}", asm);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_elementwise_scale_elf() {
        let sched = GFX1100Schedule;
        let kernel = build_elementwise_scale(&sched);
        let elf = kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("Elementwise scale ELF: {} bytes", elf.len());
    }

    #[test]
    fn test_build_gemm_forward() {
        let sched = GFX1100Schedule;
        let kernel = build_gemm_forward(&sched);
        let asm = kernel.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_wmma_f32_16x16x16_bf16"));
        assert!(asm.contains("s_cbranch_scc1"));
        assert!(asm.contains("s_mov_b32"));  // TGID capture
        eprintln!("--- GEMM Forward (T0-mid) ---\n{}", asm);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_gemm_forward_elf() {
        let sched = GFX1100Schedule;
        let kernel = build_gemm_forward(&sched);
        let elf = kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("GEMM Forward ELF: {} bytes", elf.len());
    }
}
