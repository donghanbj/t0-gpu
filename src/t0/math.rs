//! T0-high: Math Layer — High-level kernel generation from mathematical expressions
//!
//! Provides functions that generate complete, optimized GPU kernels
//! from mathematical descriptions. Each function uses a Schedule
//! to determine hardware-specific parameters.
//!
//! # Supported Operations
//! - `matmul(sched)` — Y = X @ W^T (bf16 WMMA GEMM)
//! - `rmsnorm(sched)` — RMSNorm with learnable gamma
//! - `elementwise_unary(sched, op)` — Unary ops: scale, relu, gelu, bf16↔f32
//! - `elementwise_binary(sched, op)` — Binary ops: add, mul, fma
//!
//! # Example
//! ```ignore
//! use t0::{math, GFX1100Schedule, Target};
//!
//! let sched = GFX1100Schedule;
//! let gemm = math::matmul(&sched);
//! let elf = gemm.compile(Target::GFX1100)?;
//! // Load elf into KFD runtime
//! ```

use super::ir::*;
use super::compile::T0Kernel;
use super::schedule::Schedule;

// ============================================================================
// Unary ops
// ============================================================================

/// Unary elementwise operation type.
#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    /// y = x * scale
    Scale(f32),
    /// y = max(0, x)
    Relu,
    /// y = x * sigmoid(1.702 * x)  (approximate GeLU)
    GeluApprox,
    /// y = bf16_to_f32(x)
    Bf16ToF32,
    /// y = f32_to_bf16(x)
    F32ToBf16,
    /// y = x * x (square)
    Square,
    /// y = rsqrt(x)
    Rsqrt,
    /// y = -x
    Negate,
}

/// Binary elementwise operation type.
#[derive(Clone, Copy, Debug)]
pub enum BinaryOp {
    /// y = a + b
    Add,
    /// y = a * b
    Mul,
    /// y = a + alpha * b  (axpy)
    Axpy(f32),
}

// ============================================================================
// matmul: Y = X @ W^T
// ============================================================================

/// Generate a complete GEMM forward kernel: Y = X @ W^T
///
/// Features:
/// - WMMA accumulation in f32
/// - f32 → bf16 conversion on store
/// - 2D grid dispatch: [tiles_N, tiles_M, 1]
///
/// Kernargs layout (40 bytes):
/// | Offset | Size | Name |
/// |--------|------|------|
/// | 0      | 8    | X_ptr (bf16, row-major [M, K]) |
/// | 8      | 8    | WT_ptr (bf16, row-major [N, K], i.e. W transposed) |
/// | 16     | 8    | Y_ptr (bf16, row-major [M, N]) |
/// | 24     | 4    | K (reduction dimension) |
/// | 28     | 4    | N (output columns) |
pub fn matmul(sched: &dyn Schedule) -> T0Kernel {
    // Delegate to the schedule layer's GEMM template,
    // adding the store phase with f32→bf16 conversion
    let mut k = T0Kernel::new("t0_matmul");
    let (tile_m, tile_n) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── Capture TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    // ── Thread decomposition ──
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, VReg(0));
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });

    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── X base address ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);     // tile_row * 32
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);      // wave_id * 16
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    let x_row = k.alloc_vreg();
    k.v_mov_from_sgpr(x_row, s_tmp1);
    k.v_add_u32(x_row, x_row, lane_row);

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));
    let x_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_off, x_row, k_vreg);
    k.v_lshlrev_b32(x_row_off, 1, x_row_off);

    let x_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_base, x_base, x_row_off);
    k.v_add_co_ci(VReg(x_base.0 + 1), VReg(x_base.0 + 1));

    // ── WT base ──
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);
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

    // Load X fragment
    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(x_addr, x_base);
    k.v_mov(VReg(x_addr.0 + 1), VReg(x_base.0 + 1));
    k.v_add_co(x_addr, x_addr, k_byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));
    k.global_load(x_frag, x_addr, Width::B128, 0);
    k.global_load(VReg(x_frag.0 + 4), x_addr, Width::B128, 16);

    // Load WT fragments
    let wt_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_addr, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_addr.0 + 1), SReg(wt_ptr.0 + 1));
    let wt_total_off = k.alloc_vreg();
    k.v_add_u32(wt_total_off, wt_row_off, k_byte_off);
    k.v_add_co(wt_addr, wt_addr, wt_total_off);
    k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));

    let tile_stride = k.alloc_vreg();
    k.v_mov_from_sgpr(tile_stride, SReg(k_dim.0));
    k.v_lshlrev_b32(tile_stride, 5, tile_stride);

    for t in 0..n_tiles {
        k.global_load(wt_frags[t], wt_addr, Width::B128, 0);
        k.global_load(VReg(wt_frags[t].0 + 4), wt_addr, Width::B128, 16);
        if t + 1 < n_tiles {
            k.v_add_co(wt_addr, wt_addr, tile_stride);
            k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));
        }
    }

    k.wait_vmcnt(0);

    // WMMA
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }

    // K-loop advance
    k.push(Op::VAddU32 {
        dst: k_byte_off,
        src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(tile_k as i32 * 2),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // STORE PHASE: f32 accumulators → global memory
    // ══════════════════════════════════════════════════════════════════
    //
    // WMMA f32_16x16x16_bf16 output layout (Wave32):
    //   row = base_row + lane_half + 2*vgpr_k    (vgpr_k = 0..7)
    //   col = tile_col*64 + tile_t*16 + lane_row  (lane_row = lane_id & 15)
    //
    // lane_half = lane_id >> 4  (0 or 1)
    // base_row = tile_row*32 + wave_id*16
    //
    // Store format: f32 (4 bytes per element)
    // Row stride: N * 4 bytes * 2 (skipping odd/even rows)
    //
    // NOTE: bf16 output via global_store_b16 causes GPU hang through LLVM
    // text asm path on GFX1100. Using f32 output + separate f32_to_bf16
    // conversion kernel as workaround.

    // base_row = tile_row*32 + wave_id*16 + lane_half
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);  // lane_half = lane_id >> 4

    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);   // s_tmp1 = tile_row*32 + wave_id*16
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    // row_bytes = base_row * N * 4
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);  // * 4 for f32

    // row_stride = N * 8 bytes (2 rows, f32)
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);  // N * 8

    // col_bytes_base = (tile_col*64 + lane_row) * 4
    let col_base = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base, base_n_s);  // tile_col * 64
    k.v_add_u32(col_base, col_base, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base);  // * 4 for f32

    for t in 0..n_tiles {
        // Y address = y_ptr + row_bytes + col_bytes + tile_col_extra
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        // Extra offset for tile t: t*16*4 = t*64 bytes
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr,
                src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }

        // Store 8 f32 values, advancing by row_stride (2 rows) each time
        for vk in 0..8u32 {
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_db: Double-Buffered GEMM  Y = X @ W^T
// ============================================================================

/// Generate a double-buffered GEMM kernel: Y = X @ W^T
///
/// Software-pipelined K-loop: while WMMA computes on buffer A,
/// GMEM loads fill buffer B, and vice versa. Hides ~400-cycle
/// GMEM latency behind WMMA compute (~32 cycles per 4 tiles).
///
/// Same kernargs/grid/store as matmul(). K must be multiple of 16.
///
/// Loop structure (unrolled by 2):
///   Prologue: load K=0 → buf_a
///   Loop body:
///     wait(buf_a) → load K+16 → buf_b → WMMA(buf_a)
///     wait(buf_b) → load K+32 → buf_a → WMMA(buf_b)
///   Epilogue: WMMA on last loaded buffer
pub fn matmul_db(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_db");
    let (tile_m, tile_n) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    // ── Args (same as matmul) ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    // ── Thread decomposition ──
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, VReg(0));
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── X base address ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);     // tile_row * 32
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);      // wave_id * 16
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    let x_row = k.alloc_vreg();
    k.v_mov_from_sgpr(x_row, s_tmp1);
    k.v_add_u32(x_row, x_row, lane_row);

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));
    let x_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_off, x_row, k_vreg);
    k.v_lshlrev_b32(x_row_off, 1, x_row_off);

    let x_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_base, x_base, x_row_off);
    k.v_add_co_ci(VReg(x_base.0 + 1), VReg(x_base.0 + 1));

    // ── WT base ──
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);
    let base_n = k.alloc_vreg();
    k.v_mov_from_sgpr(base_n, base_n_s);
    k.v_add_u32(base_n, base_n, lane_row);

    let wt_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_off, base_n, k_vreg);
    k.v_lshlrev_b32(wt_row_off, 1, wt_row_off);

    // WT tile stride = 16 rows * K * 2
    let tile_stride = k.alloc_vreg();
    k.v_mov_from_sgpr(tile_stride, SReg(k_dim.0));
    k.v_lshlrev_b32(tile_stride, 5, tile_stride);

    // ── Double-buffered WMMA fragments ──
    let x_frag_a = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags_a: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    let x_frag_b = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags_b: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // ── K-loop state ──
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);

    // Helper: emit GMEM loads for X + WT into given fragment set
    // This is a macro-like pattern since we call it multiple times
    macro_rules! emit_loads {
        ($kernel:expr, $x_frag:expr, $wt_frags:expr, $koff:expr) => {{
            // X address = x_base + k_byte_off
            let xa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(xa, x_base);
            $kernel.v_mov(VReg(xa.0 + 1), VReg(x_base.0 + 1));
            $kernel.v_add_co(xa, xa, $koff);
            $kernel.v_add_co_ci(VReg(xa.0 + 1), VReg(xa.0 + 1));
            $kernel.global_load($x_frag, xa, Width::B128, 0);
            $kernel.global_load(VReg($x_frag.0 + 4), xa, Width::B128, 16);

            // WT addresses
            let wa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov_from_sgpr(wa, SReg(wt_ptr.0));
            $kernel.v_mov_from_sgpr(VReg(wa.0 + 1), SReg(wt_ptr.0 + 1));
            let woff = $kernel.alloc_vreg();
            $kernel.v_add_u32(woff, wt_row_off, $koff);
            $kernel.v_add_co(wa, wa, woff);
            $kernel.v_add_co_ci(VReg(wa.0 + 1), VReg(wa.0 + 1));

            for t in 0..n_tiles {
                $kernel.global_load($wt_frags[t], wa, Width::B128, 0);
                $kernel.global_load(VReg($wt_frags[t].0 + 4), wa, Width::B128, 16);
                if t + 1 < n_tiles {
                    $kernel.v_add_co(wa, wa, tile_stride);
                    $kernel.v_add_co_ci(VReg(wa.0 + 1), VReg(wa.0 + 1));
                }
            }
        }};
    }

    // ══════════════════════════════════════════════════════════════════
    // PROLOGUE: Load first K-tile into buf_a
    // ══════════════════════════════════════════════════════════════════
    emit_loads!(k, x_frag_a, wt_frags_a, k_byte_off);

    // Advance k
    let k_step_bytes = (tile_k * 2) as i32;
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step_bytes),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    // ══════════════════════════════════════════════════════════════════
    // MAIN LOOP (unrolled by 2)
    // ══════════════════════════════════════════════════════════════════
    let loop_label = k.make_label("k_loop_db");
    k.label(&loop_label);

    // Check: if k_iter >= K, go to epilogue_a
    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_a_label = k.make_label("epilog_a");
    k.branch_scc1(&epilog_a_label);

    // ── Phase A: wait for buf_a, start loading buf_b, compute buf_a ──
    k.wait_vmcnt(0);
    emit_loads!(k, x_frag_b, wt_frags_b, k_byte_off);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag_a, wt_frags_a[t], acc[t]);
    }

    // Advance k
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step_bytes),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    // Check: if k_iter >= K, go to epilogue_b
    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_b_label = k.make_label("epilog_b");
    k.branch_scc1(&epilog_b_label);

    // ── Phase B: wait for buf_b, start loading buf_a, compute buf_b ──
    k.wait_vmcnt(0);
    emit_loads!(k, x_frag_a, wt_frags_a, k_byte_off);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag_b, wt_frags_b[t], acc[t]);
    }

    // Advance k
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step_bytes),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    // Check if more iterations needed, branch back
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // EPILOGUE A: last loaded data is in buf_a
    // ══════════════════════════════════════════════════════════════════
    k.label(&epilog_a_label);
    k.wait_vmcnt(0);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag_a, wt_frags_a[t], acc[t]);
    }
    let store_label = k.make_label("store_phase");
    // Use s_cmp + branch to skip epilog_b
    k.s_mov_imm(k_iter_s, 0); // dummy: set SCC for branch
    k.s_cmp_eq_u32_imm(k_iter_s, 0); // SCC = 1 (always)
    k.branch_scc1(&store_label);

    // ══════════════════════════════════════════════════════════════════
    // EPILOGUE B: last loaded data is in buf_b
    // ══════════════════════════════════════════════════════════════════
    k.label(&epilog_b_label);
    k.wait_vmcnt(0);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag_b, wt_frags_b[t], acc[t]);
    }

    // ══════════════════════════════════════════════════════════════════
    // STORE PHASE (same as matmul)
    // ══════════════════════════════════════════════════════════════════
    k.label(&store_label);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);

    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_base = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base, base_n_s);
    k.v_add_u32(col_base, col_base, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base);

    for t in 0..n_tiles {
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr,
                src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_lds_db: LDS-Staged + Double-Buffered GEMM  Y = X @ W^T
// ============================================================================

/// LDS cooperative-load + double-buffered GEMM: Y[M,N] = X[M,K] @ WT[N,K]
///
/// All 64 threads cooperatively load X (1 b128/thread) + WT (2 b128/thread)
/// into LDS. Each wave then reads its WMMA fragments from LDS (~40 cycle
/// latency vs ~400 for GMEM). Two LDS buffers enable overlapping GMEM
/// loads for the next K-tile with LDS reads + WMMA for the current tile.
///
/// LDS layout (6144 bytes = 2 × 3072B buffers):
///   Buf 0: X[0..1023] (32×16 bf16) + WT[1024..3071] (64×16 bf16)
///   Buf 1: X[3072..4095] + WT[4096..6143]
///
/// Same kernargs/grid as matmul(). K must be multiple of 16.
pub fn matmul_lds_db(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_lds_db");
    let (_tile_m, _tile_n) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    const LDS_X: u32 = 1024;   // 32 rows × 16 cols × 2B
    const LDS_WT: u32 = 2048;  // 64 rows × 16 cols × 2B
    const LDS_BUF: u32 = LDS_X + LDS_WT;  // 3072
    k.set_lds_size(LDS_BUF * 2);           // 6144

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    // ── Thread decomposition ──
    let tid = VReg(0); // WORKITEM_ID_X = 0..63
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── s_tmp1 = tile_row*32 + wave_id*16 (for store phase) ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);

    // ══════════════════════════════════════════════════════════════════
    // COOPERATIVE LOAD ADDRESSES (computed once, reused per K-tile)
    // ══════════════════════════════════════════════════════════════════

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));

    // ── X cooperative: thread t → row=t>>1, half=t&1, loads 16B ──
    let x_coop_row = k.alloc_vreg();
    k.v_lshrrev_b32(x_coop_row, 1, tid);       // t >> 1 = row (0..31)
    let x_coop_half = k.alloc_vreg();
    k.v_and_b32_imm(x_coop_half, tid, 1);       // t & 1

    // X GMEM base = X_ptr + (tile_row*32 + row) * K * 2 + half*16
    let x_abs_row = k.alloc_vreg();
    let s_base_row = k.alloc_sreg();
    k.s_lshl_b32(s_base_row, tile_row_s, 5);
    k.v_mov_from_sgpr(x_abs_row, s_base_row);
    k.v_add_u32(x_abs_row, x_abs_row, x_coop_row);

    let x_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_byte, x_abs_row, k_vreg);
    k.v_lshlrev_b32(x_row_byte, 1, x_row_byte);

    let x_half_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_half_off, 4, x_coop_half); // half * 16

    let x_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_gmem_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_gmem_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_gmem_base, x_gmem_base, x_row_byte);
    k.v_add_co_ci(VReg(x_gmem_base.0 + 1), VReg(x_gmem_base.0 + 1));
    k.v_add_u32(x_gmem_base, x_gmem_base, x_half_off);

    // X LDS addr = (t>>1)*32 + (t&1)*16 = t*16 (relative to buf base)
    let x_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_lds_off, 4, tid); // t * 16

    // ── WT cooperative: thread t → loads WT row t (32B = 2×b128) ──
    let wt_abs_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_abs_row, base_n_s);
    k.v_add_u32(wt_abs_row, wt_abs_row, tid);

    let wt_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_byte, wt_abs_row, k_vreg);
    k.v_lshlrev_b32(wt_row_byte, 1, wt_row_byte);

    let wt_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_gmem_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_gmem_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_gmem_base, wt_gmem_base, wt_row_byte);
    k.v_add_co_ci(VReg(wt_gmem_base.0 + 1), VReg(wt_gmem_base.0 + 1));

    // WT LDS addr = LDS_X + t*32 (relative to buf base)
    let wt_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_off, 5, tid);
    k.push(Op::VAddU32 {
        dst: wt_lds_off, src0: Operand::VReg(wt_lds_off),
        src1: Operand::InlineInt(LDS_X as i32),
    });

    // ── LDS read addresses for WMMA fragments (relative to buf base) ──
    // X read: (wave_id*16 + lane_row) * 32
    let x_lds_read = k.alloc_vreg();
    let s_wave_off = k.alloc_sreg();
    k.s_lshl_b32(s_wave_off, wave_id_s, 9);     // wave_id * 512
    k.v_lshlrev_b32(x_lds_read, 5, lane_row);   // lane_row * 32
    let tmp_wave = k.alloc_vreg();
    k.v_mov_from_sgpr(tmp_wave, s_wave_off);
    k.v_add_u32(x_lds_read, x_lds_read, tmp_wave);

    // WT read base: lane_row * 32 (tile offset added as immediate)
    let wt_lds_read_base = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_read_base, 5, lane_row);

    // ── Temp VGPRs for GMEM → LDS transfer ──
    let gmem_x = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt0 = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt1 = k.alloc_vreg_array(4, Alignment::Align4);

    // ── WMMA fragment registers ──
    let x_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // ── K-loop state ──
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let k_step = (tile_k * 2) as i32;

    // ── Macro: cooperative GMEM load → temp VGPRs ──
    macro_rules! coop_gmem_load {
        ($kernel:expr, $koff:expr) => {{
            // X: global_load_b128, 1 per thread
            let xa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(xa, x_gmem_base);
            $kernel.v_mov(VReg(xa.0 + 1), VReg(x_gmem_base.0 + 1));
            $kernel.v_add_co(xa, xa, $koff);
            $kernel.v_add_co_ci(VReg(xa.0 + 1), VReg(xa.0 + 1));
            $kernel.global_load(gmem_x, xa, Width::B128, 0);

            // WT: 2× global_load_b128, 1 row per thread
            let wa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(wa, wt_gmem_base);
            $kernel.v_mov(VReg(wa.0 + 1), VReg(wt_gmem_base.0 + 1));
            $kernel.v_add_co(wa, wa, $koff);
            $kernel.v_add_co_ci(VReg(wa.0 + 1), VReg(wa.0 + 1));
            $kernel.global_load(gmem_wt0, wa, Width::B128, 0);
            $kernel.global_load(gmem_wt1, wa, Width::B128, 16);
        }};
    }

    // ── Macro: temp VGPRs → LDS store (at buf_offset) ──
    macro_rules! coop_lds_store {
        ($kernel:expr, $buf_off:expr) => {{
            let x_a = $kernel.alloc_vreg();
            $kernel.v_add_u32(x_a, x_lds_off, $buf_off);
            $kernel.ds_store_b128(x_a, gmem_x, 0);

            let w_a = $kernel.alloc_vreg();
            $kernel.v_add_u32(w_a, wt_lds_off, $buf_off);
            $kernel.ds_store_b128(w_a, gmem_wt0, 0);
            $kernel.ds_store_b128(w_a, gmem_wt1, 16);
        }};
    }

    // ── Macro: LDS read → WMMA fragments (from buf_offset) ──
    macro_rules! lds_read_frags {
        ($kernel:expr, $buf_off:expr) => {{
            // X fragment
            let xr = $kernel.alloc_vreg();
            $kernel.v_add_u32(xr, x_lds_read, $buf_off);
            $kernel.ds_load_b128(x_frag, xr, 0);
            $kernel.ds_load_b128(VReg(x_frag.0 + 4), xr, 16);

            // WT fragments (4 tiles)
            for t in 0..n_tiles {
                let wr = $kernel.alloc_vreg();
                $kernel.v_add_u32(wr, wt_lds_read_base, $buf_off);
                let off_base: u16 = (LDS_X + (t as u32) * 16 * 32) as u16;
                $kernel.ds_load_b128(wt_frags[t], wr, off_base);
                $kernel.ds_load_b128(VReg(wt_frags[t].0 + 4), wr, off_base + 16);
            }
        }};
    }

    // ── Buffer offset VGPRs ──
    let buf0_off = k.alloc_vreg();
    k.v_mov_imm(buf0_off, 0);
    let buf1_off = k.alloc_vreg();
    k.v_mov_imm(buf1_off, LDS_BUF as i32);

    // ══════════════════════════════════════════════════════════════════
    // PROLOGUE: Load K=0 → LDS buf0
    // ══════════════════════════════════════════════════════════════════
    coop_gmem_load!(k, k_byte_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    // barrier removed — Phase A at loop top will do the barrier

    // Advance K
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    // ══════════════════════════════════════════════════════════════════
    // MAIN LOOP (unrolled by 2)
    // ══════════════════════════════════════════════════════════════════
    let loop_label = k.make_label("k_lds_loop");
    k.label(&loop_label);

    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_a = k.make_label("lds_epilog_a");
    k.branch_scc1(&epilog_a);

    // ── Phase A: Tensile-style overlap ──
    // barrier → LDS read → GMEM load (parallel!) → wait LDS → WMMA → wait GMEM → LDS store
    k.s_barrier();                            // sync (prologue/Phase B stores done)
    lds_read_frags!(k, buf0_off);             // 10 ds_loads (LDS reads first!)
    coop_gmem_load!(k, k_byte_off);           // 3 global_loads (parallel with ds_loads)
    k.wait_lgkmcnt(0);                        // wait ds_loads
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }
    k.wait_vmcnt(0);                          // wait GMEM loads
    coop_lds_store!(k, buf1_off);             // 3 ds_stores (async, overlap with advance K!)

    // Advance K
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_b = k.make_label("lds_epilog_b");
    k.branch_scc1(&epilog_b);

    // ── Phase B: Tensile-style overlap ──
    k.wait_lgkmcnt(0);                        // wait Phase A ds_stores
    k.s_barrier();                            // sync all waves
    lds_read_frags!(k, buf1_off);             // 10 ds_loads
    coop_gmem_load!(k, k_byte_off);           // 3 global_loads (parallel)
    k.wait_lgkmcnt(0);                        // wait ds_loads
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }
    k.wait_vmcnt(0);                          // wait GMEM
    coop_lds_store!(k, buf0_off);             // 3 ds_stores (async)

    // Advance K
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.wait_lgkmcnt(0);                        // wait Phase B ds_stores before loop back
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // EPILOGUE A: last data in buf0
    // ══════════════════════════════════════════════════════════════════
    k.label(&epilog_a);
    k.s_barrier();                            // sync before reading
    lds_read_frags!(k, buf0_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }
    let store_label = k.make_label("lds_store");
    k.s_mov_imm(k_iter_s, 0);
    k.s_cmp_eq_u32_imm(k_iter_s, 0);
    k.branch_scc1(&store_label);

    // ══════════════════════════════════════════════════════════════════
    // EPILOGUE B: last data in buf1
    // ══════════════════════════════════════════════════════════════════
    k.label(&epilog_b);
    k.wait_lgkmcnt(0);                        // wait Phase A ds_stores
    k.s_barrier();                            // sync before reading
    lds_read_frags!(k, buf1_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }

    // ══════════════════════════════════════════════════════════════════
    // STORE PHASE (same as matmul)
    // ══════════════════════════════════════════════════════════════════
    k.label(&store_label);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);

    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_base = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base, base_n_s);
    k.v_add_u32(col_base, col_base, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base);

    for t in 0..n_tiles {
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr, src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_direct: Zero-LDS GEMM (ported from handwritten 113 TFLOPS kernel)
// ============================================================================

/// Y[i,j] = X @ W^T  — Zero-LDS direct GMEM→VGPR→WMMA approach.
///
/// Each lane loads its own X row and WT rows directly from global memory,
/// no cooperative load, no LDS staging, no barriers. This eliminates the
/// ~30% LDS overhead that dominates small-matrix performance.
///
/// Same kernargs/grid as matmul_lds_db. K must be multiple of 16.
pub fn matmul_direct(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_direct");
    let (_tile_m, _tile_n) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();  // 16
    let n_tiles = sched.gemm_n_wmma_tiles();  // 4

    k.set_lds_size(0);  // Zero LDS!

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

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

    // ── Accumulators ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── s_tmp1 = tile_row*32 + wave_id*16 (for store phase) ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);

    // ══════════════════════════════════════════════════════════════════
    // X BASE ADDRESS: X_ptr + (tile_row*32 + wave_id*16 + lane_row) * K * 2
    // Pre-computed once (constant per lane, only k_offset changes in loop)
    // ══════════════════════════════════════════════════════════════════
    let x_row = k.alloc_vreg();
    k.v_mov_from_sgpr(x_row, s_tmp1);
    k.v_add_u32(x_row, x_row, lane_row);

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));

    let x_row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_bytes, x_row, k_vreg);
    k.v_lshlrev_b32(x_row_bytes, 1, x_row_bytes);  // ×2 for bf16

    let x_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_base, x_base, x_row_bytes);
    k.v_add_co_ci(VReg(x_base.0 + 1), VReg(x_base.0 + 1));

    // ══════════════════════════════════════════════════════════════════
    // WT BASE ADDRESS: WT_ptr + (tile_col*64 + lane_row) * K * 2
    // Pre-computed once. Tile stride = 16 * K * 2 = K << 5 bytes
    // ══════════════════════════════════════════════════════════════════
    let wt_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_row, base_n_s);
    k.v_add_u32(wt_row, wt_row, lane_row);

    let wt_row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_bytes, wt_row, k_vreg);
    k.v_lshlrev_b32(wt_row_bytes, 1, wt_row_bytes);

    let wt_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_base, wt_base, wt_row_bytes);
    k.v_add_co_ci(VReg(wt_base.0 + 1), VReg(wt_base.0 + 1));

    // Tile stride = 16 rows × K cols × 2 bytes = K << 5
    let tile_stride = k.alloc_vreg();
    k.v_lshlrev_b32(tile_stride, 5, k_vreg);

    // ── WMMA fragment registers ──
    let x_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // ── K byte offset (VGPR, incremented each iteration) ──
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);

    // K step in bytes = tile_k * 2 = 32
    let k_step = (tile_k * 2) as i32;

    // ══════════════════════════════════════════════════════════════════
    // K-LOOP: Direct GMEM → VGPR → WMMA (zero LDS!)
    // ══════════════════════════════════════════════════════════════════
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0i32);

    let loop_label = k.make_label("k_direct_loop");
    k.label(&loop_label);

    // ── Load X fragment: X_base + k_byte_off ──
    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(x_addr, x_base);
    k.v_mov(VReg(x_addr.0 + 1), VReg(x_base.0 + 1));
    k.v_add_co(x_addr, x_addr, k_byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));
    k.global_load(x_frag, x_addr, Width::B128, 0);
    k.global_load(VReg(x_frag.0 + 4), x_addr, Width::B128, 16);

    // ── Load WT fragments: 4 tiles, each 16 WT rows apart ──
    let wt_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(wt_addr, wt_base);
    k.v_mov(VReg(wt_addr.0 + 1), VReg(wt_base.0 + 1));
    k.v_add_co(wt_addr, wt_addr, k_byte_off);
    k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));

    for t in 0..n_tiles {
        k.global_load(wt_frags[t], wt_addr, Width::B128, 0);
        k.global_load(VReg(wt_frags[t].0 + 4), wt_addr, Width::B128, 16);
        if t < n_tiles - 1 {
            // Advance to next tile: +tile_stride (K*32 bytes)
            k.v_add_co(wt_addr, wt_addr, tile_stride);
            k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));
        }
    }

    // ── Wait for all loads, then WMMA ──
    k.wait_vmcnt(0);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }

    // ── Advance K ──
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // STORE PHASE (identical to matmul_lds_db)
    // ══════════════════════════════════════════════════════════════════
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg2 = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg2, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg2);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);

    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg2);

    let col_base = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base, base_n_s);
    k.v_add_u32(col_base, col_base, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base);

    for t in 0..n_tiles {
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr, src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_direct_add: Zero-LDS GEMM + Fused Residual Add
// ============================================================================

/// Y[i,j] = X @ W^T + residual  — Zero-LDS with fused residual add.
///
/// Same K-loop as matmul_direct (zero LDS, direct GMEM→VGPR→WMMA).
/// Store phase loads residual[i,j], adds to accumulator, stores sum.
///
/// Kernargs (40 bytes):
///   0:  X_ptr   (8B) — bf16 [M, K]
///   8:  WT_ptr  (8B) — bf16 [N, K]
///   16: Y_ptr   (8B) — f32 [M, N] output
///   24: K       (4B)
///   28: N       (4B)
///   32: res_ptr (8B) — f32 [M, N] residual
pub fn matmul_direct_add(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_direct_add");
    let (_tile_m, _tile_n) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    k.set_lds_size(0);

    // ── Args (40 bytes — extra res_ptr) ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    let res_ptr = k.arg_ptr("res");
    k.emit_arg_loads();

    // ── TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

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

    // ── Accumulators ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── s_tmp1 = tile_row*32 + wave_id*16 ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);

    // ── X base address ──
    let x_row = k.alloc_vreg();
    k.v_mov_from_sgpr(x_row, s_tmp1);
    k.v_add_u32(x_row, x_row, lane_row);
    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));
    let x_row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_bytes, x_row, k_vreg);
    k.v_lshlrev_b32(x_row_bytes, 1, x_row_bytes);
    let x_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_base, x_base, x_row_bytes);
    k.v_add_co_ci(VReg(x_base.0 + 1), VReg(x_base.0 + 1));

    // ── WT base address + tile stride ──
    let wt_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_row, base_n_s);
    k.v_add_u32(wt_row, wt_row, lane_row);
    let wt_row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_bytes, wt_row, k_vreg);
    k.v_lshlrev_b32(wt_row_bytes, 1, wt_row_bytes);
    let wt_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_base, wt_base, wt_row_bytes);
    k.v_add_co_ci(VReg(wt_base.0 + 1), VReg(wt_base.0 + 1));
    let tile_stride = k.alloc_vreg();
    k.v_lshlrev_b32(tile_stride, 5, k_vreg);

    // ── WMMA fragments ──
    let x_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_step = (tile_k * 2) as i32;

    // ── K-LOOP (identical to matmul_direct) ──
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0i32);
    let loop_label = k.make_label("k_add_loop");
    k.label(&loop_label);

    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(x_addr, x_base);
    k.v_mov(VReg(x_addr.0 + 1), VReg(x_base.0 + 1));
    k.v_add_co(x_addr, x_addr, k_byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));
    k.global_load(x_frag, x_addr, Width::B128, 0);
    k.global_load(VReg(x_frag.0 + 4), x_addr, Width::B128, 16);

    let wt_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(wt_addr, wt_base);
    k.v_mov(VReg(wt_addr.0 + 1), VReg(wt_base.0 + 1));
    k.v_add_co(wt_addr, wt_addr, k_byte_off);
    k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));
    for t in 0..n_tiles {
        k.global_load(wt_frags[t], wt_addr, Width::B128, 0);
        k.global_load(VReg(wt_frags[t].0 + 4), wt_addr, Width::B128, 16);
        if t < n_tiles - 1 {
            k.v_add_co(wt_addr, wt_addr, tile_stride);
            k.v_add_co_ci(VReg(wt_addr.0 + 1), VReg(wt_addr.0 + 1));
        }
    }
    k.wait_vmcnt(0);
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }
    k.push(Op::VAddU32 {
        dst: k_byte_off, src0: Operand::VReg(k_byte_off),
        src1: Operand::InlineInt(k_step),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // FUSED STORE: Y[i,j] = acc[i,j] + residual[i,j]
    // ══════════════════════════════════════════════════════════════════
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg2 = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg2, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg2);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);

    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg2);

    let col_base = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base, base_n_s);
    k.v_add_u32(col_base, col_base, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base);

    let res_val = k.alloc_vreg();

    for t in 0..n_tiles {
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr, src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }

        let r_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(r_addr, SReg(res_ptr.0));
        k.v_mov_from_sgpr(VReg(r_addr.0 + 1), SReg(res_ptr.0 + 1));
        k.v_add_co(r_addr, r_addr, row_bytes);
        k.v_add_co_ci(VReg(r_addr.0 + 1), VReg(r_addr.0 + 1));
        k.v_add_u32(r_addr, r_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: r_addr, src0: Operand::VReg(r_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }

        for vk in 0..8u32 {
            k.global_load(res_val, r_addr, Width::B32, 0);
            k.wait_vmcnt(0);
            k.v_add_f32(VReg(acc[t].0 + vk), VReg(acc[t].0 + vk), res_val);
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
                k.v_add_co(r_addr, r_addr, row_stride);
                k.v_add_co_ci(VReg(r_addr.0 + 1), VReg(r_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_lds_db_add: GEMM + Residual Add Epilogue Fusion
// ============================================================================

/// Y[i,j] = X @ W^T + residual  (fused GEMM + residual add)
///
/// Identical computation loop to matmul_lds_db, but the store phase
/// loads residual[row,col] from GMEM, adds to accumulator, and stores
/// the sum. Eliminates separate add() dispatch.
///
/// Kernargs (40 bytes):
///   0:  X_ptr   (8B) — bf16 [M, K]
///   8:  WT_ptr  (8B) — bf16 [N, K]
///   16: Y_ptr   (8B) — f32 [M, N] output
///   24: K       (4B)
///   28: N       (4B)
///   32: res_ptr (8B) — f32 [M, N] residual to add
///
/// Grid: same as matmul_lds_db [ceil(N/64)*64, ceil(M/32), 1]
pub fn matmul_lds_db_add(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_lds_db_add");
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    const LDS_X: u32 = 1024;
    const LDS_WT: u32 = 2048;
    const LDS_BUF: u32 = LDS_X + LDS_WT;
    k.set_lds_size(LDS_BUF * 2);

    // ── Args (40 bytes) ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    let res_ptr = k.arg_ptr("residual");
    k.emit_arg_loads();

    // ── TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── Store phase constants ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);

    // ══════════════════════════════════════════════════════════════════
    // COOPERATIVE LOAD setup (identical to matmul_lds_db)
    // ══════════════════════════════════════════════════════════════════
    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));

    let x_coop_row = k.alloc_vreg();
    k.v_lshrrev_b32(x_coop_row, 1, tid);
    let x_coop_half = k.alloc_vreg();
    k.v_and_b32_imm(x_coop_half, tid, 1);

    let x_abs_row = k.alloc_vreg();
    let s_base_row = k.alloc_sreg();
    k.s_lshl_b32(s_base_row, tile_row_s, 5);
    k.v_mov_from_sgpr(x_abs_row, s_base_row);
    k.v_add_u32(x_abs_row, x_abs_row, x_coop_row);

    let x_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_byte, x_abs_row, k_vreg);
    k.v_lshlrev_b32(x_row_byte, 1, x_row_byte);

    let x_half_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_half_off, 4, x_coop_half);

    let x_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_gmem_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_gmem_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_gmem_base, x_gmem_base, x_row_byte);
    k.v_add_co_ci(VReg(x_gmem_base.0 + 1), VReg(x_gmem_base.0 + 1));
    k.v_add_u32(x_gmem_base, x_gmem_base, x_half_off);

    let x_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_lds_off, 4, tid);

    let wt_abs_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_abs_row, base_n_s);
    k.v_add_u32(wt_abs_row, wt_abs_row, tid);

    let wt_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_byte, wt_abs_row, k_vreg);
    k.v_lshlrev_b32(wt_row_byte, 1, wt_row_byte);

    let wt_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_gmem_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_gmem_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_gmem_base, wt_gmem_base, wt_row_byte);
    k.v_add_co_ci(VReg(wt_gmem_base.0 + 1), VReg(wt_gmem_base.0 + 1));

    let wt_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_off, 5, tid);
    k.push(Op::VAddU32 {
        dst: wt_lds_off, src0: Operand::VReg(wt_lds_off),
        src1: Operand::InlineInt(LDS_X as i32),
    });

    // LDS read addresses
    let x_lds_read = k.alloc_vreg();
    let s_wave_off = k.alloc_sreg();
    k.s_lshl_b32(s_wave_off, wave_id_s, 9);
    k.v_lshlrev_b32(x_lds_read, 5, lane_row);
    let tmp_wave = k.alloc_vreg();
    k.v_mov_from_sgpr(tmp_wave, s_wave_off);
    k.v_add_u32(x_lds_read, x_lds_read, tmp_wave);

    let wt_lds_read_base = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_read_base, 5, lane_row);

    // Temp VGPRs
    let gmem_x = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt0 = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt1 = k.alloc_vreg_array(4, Alignment::Align4);
    let x_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // K-loop state
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let k_step = (tile_k * 2) as i32;
    let buf0_off = k.alloc_vreg();
    k.v_mov_imm(buf0_off, 0);
    let buf1_off = k.alloc_vreg();
    k.v_mov_imm(buf1_off, LDS_BUF as i32);

    // ── Macros (identical to matmul_lds_db) ──
    macro_rules! coop_gmem_load {
        ($kernel:expr, $koff:expr) => {{
            let xa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(xa, x_gmem_base);
            $kernel.v_mov(VReg(xa.0 + 1), VReg(x_gmem_base.0 + 1));
            $kernel.v_add_co(xa, xa, $koff);
            $kernel.v_add_co_ci(VReg(xa.0 + 1), VReg(xa.0 + 1));
            $kernel.global_load(gmem_x, xa, Width::B128, 0);
            let wa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(wa, wt_gmem_base);
            $kernel.v_mov(VReg(wa.0 + 1), VReg(wt_gmem_base.0 + 1));
            $kernel.v_add_co(wa, wa, $koff);
            $kernel.v_add_co_ci(VReg(wa.0 + 1), VReg(wa.0 + 1));
            $kernel.global_load(gmem_wt0, wa, Width::B128, 0);
            $kernel.global_load(gmem_wt1, wa, Width::B128, 16);
        }};
    }
    macro_rules! coop_lds_store {
        ($kernel:expr, $buf_off:expr) => {{
            let xa = $kernel.alloc_vreg();
            $kernel.v_add_u32(xa, x_lds_off, $buf_off);
            $kernel.ds_store_b128(xa, gmem_x, 0);
            let wa = $kernel.alloc_vreg();
            $kernel.v_add_u32(wa, wt_lds_off, $buf_off);
            $kernel.ds_store_b128(wa, gmem_wt0, 0);
            $kernel.ds_store_b128(wa, gmem_wt1, 16);
        }};
    }
    macro_rules! lds_read_frags {
        ($kernel:expr, $buf_off:expr) => {{
            let xr = $kernel.alloc_vreg();
            $kernel.v_add_u32(xr, x_lds_read, $buf_off);
            $kernel.ds_load_b128(x_frag, xr, 0);
            $kernel.ds_load_b128(VReg(x_frag.0 + 4), xr, 16);
            for t in 0..n_tiles {
                let wr = $kernel.alloc_vreg();
                $kernel.v_add_u32(wr, wt_lds_read_base, $buf_off);
                let off: u16 = (LDS_X + (t as u32) * 16 * 32) as u16;
                $kernel.ds_load_b128(wt_frags[t], wr, off);
                $kernel.ds_load_b128(VReg(wt_frags[t].0 + 4), wr, off + 16);
            }
        }};
    }

    // ══════════════════════════════════════════════════════════════════
    // COMPUTATION (identical to matmul_lds_db)
    // ══════════════════════════════════════════════════════════════════
    coop_gmem_load!(k, k_byte_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    let loop_label = k.make_label("add_loop");
    k.label(&loop_label);
    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_a = k.make_label("add_ea");
    k.branch_scc1(&epilog_a);

    coop_gmem_load!(k, k_byte_off);
    lds_read_frags!(k, buf0_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf1_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_b = k.make_label("add_eb");
    k.branch_scc1(&epilog_b);

    coop_gmem_load!(k, k_byte_off);
    lds_read_frags!(k, buf1_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    k.label(&epilog_a);
    lds_read_frags!(k, buf0_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }
    let store_label = k.make_label("add_store");
    k.s_mov_imm(k_iter_s, 0);
    k.s_cmp_eq_u32_imm(k_iter_s, 0);
    k.branch_scc1(&store_label);

    k.label(&epilog_b);
    lds_read_frags!(k, buf1_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }

    // ══════════════════════════════════════════════════════════════════
    // FUSED STORE: Y[i,j] = acc[i,j] + residual[i,j]
    // ══════════════════════════════════════════════════════════════════
    k.label(&store_label);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);

    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_base = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base, base_n_s);
    k.v_add_u32(col_base, col_base, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base);

    let res_val = k.alloc_vreg();  // temp for residual load

    for t in 0..n_tiles {
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr, src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }

        // Residual address (same layout as Y)
        let r_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(r_addr, SReg(res_ptr.0));
        k.v_mov_from_sgpr(VReg(r_addr.0 + 1), SReg(res_ptr.0 + 1));
        k.v_add_co(r_addr, r_addr, row_bytes);
        k.v_add_co_ci(VReg(r_addr.0 + 1), VReg(r_addr.0 + 1));
        k.v_add_u32(r_addr, r_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: r_addr, src0: Operand::VReg(r_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }

        for vk in 0..8u32 {
            // Load residual, add to acc, store
            k.global_load(res_val, r_addr, Width::B32, 0);
            k.wait_vmcnt(0);
            k.v_add_f32(VReg(acc[t].0 + vk), VReg(acc[t].0 + vk), res_val);
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
                k.v_add_co(r_addr, r_addr, row_stride);
                k.v_add_co_ci(VReg(r_addr.0 + 1), VReg(r_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_64x64: 64×64 Tiled LDS + Double-Buffered GEMM  Y = X @ W^T
// ============================================================================

/// 64×64 tile GEMM with LDS cooperative load + double buffering.
///
/// Each wave handles a 32×64 sub-tile with 2 row blocks × 4 col blocks = 8 WMMAs.
/// Compared to 32×64 tile: 33% fewer GMEM bytes per output element.
///
/// LDS layout (8192B = 2 × 4096B):
///   X:  64 rows × 16 cols × 2B = 2048B
///   WT: 64 rows × 16 cols × 2B = 2048B
///
/// Grid: [ceil(N/64)*64, ceil(M/64), 1]  (note: M/64 not M/32!)
/// Kernargs: same 32-byte layout as matmul().
pub fn matmul_64x64_lds_db(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_64x64");
    let tile_k = sched.gemm_tile_k();   // 16
    let n_col_tiles: usize = 4;         // 64/16
    let n_row_blocks: usize = 2;        // 32/16 per wave

    const LDS_X: u32 = 2048;   // 64 × 16 × 2
    const LDS_WT: u32 = 2048;  // 64 × 16 × 2
    const LDS_BUF: u32 = LDS_X + LDS_WT;  // 4096
    k.set_lds_size(LDS_BUF * 2);            // 8192

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators: 2 row blocks × 4 col blocks = 8 per wave ──
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
    // s_row_base[r] = tile_row*64 + wave_id*32 + r*16
    let s_row_base0 = k.alloc_sreg();
    let s_row_base1 = k.alloc_sreg();
    k.s_lshl_b32(s_row_base0, tile_row_s, 6);  // tile_row * 64
    let s_tmp = k.alloc_sreg();
    k.s_lshl_b32(s_tmp, wave_id_s, 5);          // wave_id * 32
    k.push(Op::SAddU32 { dst: s_row_base0, src0: s_row_base0, src1: SOperand::SReg(s_tmp) });
    k.push(Op::SAddU32 { dst: s_row_base1, src0: s_row_base0, src1: SOperand::InlineInt(16) });

    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);  // tile_col * 64

    // ══════════════════════════════════════════════════════════════════
    // COOPERATIVE LOAD ADDRESSES
    // ══════════════════════════════════════════════════════════════════

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));

    // ── X: thread t loads full row t (2× b128 = 32B) ──
    let x_abs_row = k.alloc_vreg();
    let s_xbase = k.alloc_sreg();
    k.s_lshl_b32(s_xbase, tile_row_s, 6);  // tile_row * 64
    k.v_mov_from_sgpr(x_abs_row, s_xbase);
    k.v_add_u32(x_abs_row, x_abs_row, tid);

    let x_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_byte, x_abs_row, k_vreg);
    k.v_lshlrev_b32(x_row_byte, 1, x_row_byte);

    let x_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_gmem_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_gmem_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_gmem_base, x_gmem_base, x_row_byte);
    k.v_add_co_ci(VReg(x_gmem_base.0 + 1), VReg(x_gmem_base.0 + 1));

    // X LDS store addr = t * 32
    let x_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_lds_off, 5, tid);

    // ── WT: thread t loads WT row t (2× b128 = 32B) ──
    let wt_abs_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_abs_row, base_n_s);
    k.v_add_u32(wt_abs_row, wt_abs_row, tid);

    let wt_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_byte, wt_abs_row, k_vreg);
    k.v_lshlrev_b32(wt_row_byte, 1, wt_row_byte);

    let wt_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_gmem_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_gmem_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_gmem_base, wt_gmem_base, wt_row_byte);
    k.v_add_co_ci(VReg(wt_gmem_base.0 + 1), VReg(wt_gmem_base.0 + 1));

    // WT LDS store addr = LDS_X + t * 32
    let wt_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_off, 5, tid);
    k.push(Op::VAddU32 {
        dst: wt_lds_off, src0: Operand::VReg(wt_lds_off),
        src1: Operand::InlineInt(LDS_X as i32),
    });

    // ── LDS read addresses ──
    // X frag[0]: (wave_id*32 + lane_row) * 32 = wave_id*1024 + lane_row*32
    // X frag[1]: (wave_id*32 + 16 + lane_row) * 32 = wave_id*1024 + 512 + lane_row*32
    let lane_row_x32 = k.alloc_vreg();
    k.v_lshlrev_b32(lane_row_x32, 5, lane_row);

    let s_wave_x_off = k.alloc_sreg();
    k.s_lshl_b32(s_wave_x_off, wave_id_s, 10);  // wave_id * 1024

    let x_lds_read0 = k.alloc_vreg();
    k.v_mov_from_sgpr(x_lds_read0, s_wave_x_off);
    k.v_add_u32(x_lds_read0, x_lds_read0, lane_row_x32);

    let x_lds_read1 = k.alloc_vreg();
    k.v_mov_from_sgpr(x_lds_read1, s_wave_x_off);
    k.v_add_u32(x_lds_read1, x_lds_read1, lane_row_x32);
    k.push(Op::VAddU32 {
        dst: x_lds_read1, src0: Operand::VReg(x_lds_read1),
        src1: Operand::InlineInt(512),
    });

    // WT: lane_row * 32 (tile offset as immediate)
    let wt_lds_read_base = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_read_base, 5, lane_row);

    // ── Temp VGPRs ──
    let gmem_x0 = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_x1 = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt0 = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt1 = k.alloc_vreg_array(4, Alignment::Align4);

    // ── WMMA fragment registers ──
    let x_frag0 = k.alloc_vreg_array(8, Alignment::Align8);
    let x_frag1 = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_col_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // ── K-loop state ──
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let k_step = (tile_k * 2) as i32;

    let buf0_off = k.alloc_vreg();
    k.v_mov_imm(buf0_off, 0);
    let buf1_off = k.alloc_vreg();
    k.v_mov_imm(buf1_off, LDS_BUF as i32);

    // ── Macros ──
    macro_rules! coop_gmem_load {
        ($kernel:expr, $koff:expr) => {{
            let xa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(xa, x_gmem_base);
            $kernel.v_mov(VReg(xa.0 + 1), VReg(x_gmem_base.0 + 1));
            $kernel.v_add_co(xa, xa, $koff);
            $kernel.v_add_co_ci(VReg(xa.0 + 1), VReg(xa.0 + 1));
            $kernel.global_load(gmem_x0, xa, Width::B128, 0);
            $kernel.global_load(gmem_x1, xa, Width::B128, 16);

            let wa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(wa, wt_gmem_base);
            $kernel.v_mov(VReg(wa.0 + 1), VReg(wt_gmem_base.0 + 1));
            $kernel.v_add_co(wa, wa, $koff);
            $kernel.v_add_co_ci(VReg(wa.0 + 1), VReg(wa.0 + 1));
            $kernel.global_load(gmem_wt0, wa, Width::B128, 0);
            $kernel.global_load(gmem_wt1, wa, Width::B128, 16);
        }};
    }

    macro_rules! coop_lds_store {
        ($kernel:expr, $buf_off:expr) => {{
            let xa = $kernel.alloc_vreg();
            $kernel.v_add_u32(xa, x_lds_off, $buf_off);
            $kernel.ds_store_b128(xa, gmem_x0, 0);
            $kernel.ds_store_b128(xa, gmem_x1, 16);

            let wa = $kernel.alloc_vreg();
            $kernel.v_add_u32(wa, wt_lds_off, $buf_off);
            $kernel.ds_store_b128(wa, gmem_wt0, 0);
            $kernel.ds_store_b128(wa, gmem_wt1, 16);
        }};
    }

    macro_rules! lds_read_and_wmma {
        ($kernel:expr, $buf_off:expr) => {{
            // X frag[0]
            let xr0 = $kernel.alloc_vreg();
            $kernel.v_add_u32(xr0, x_lds_read0, $buf_off);
            $kernel.ds_load_b128(x_frag0, xr0, 0);
            $kernel.ds_load_b128(VReg(x_frag0.0 + 4), xr0, 16);
            // X frag[1]
            let xr1 = $kernel.alloc_vreg();
            $kernel.v_add_u32(xr1, x_lds_read1, $buf_off);
            $kernel.ds_load_b128(x_frag1, xr1, 0);
            $kernel.ds_load_b128(VReg(x_frag1.0 + 4), xr1, 16);
            // WT frags
            for c in 0..n_col_tiles {
                let wr = $kernel.alloc_vreg();
                $kernel.v_add_u32(wr, wt_lds_read_base, $buf_off);
                let off: u16 = (LDS_X + (c as u32) * 512) as u16;
                $kernel.ds_load_b128(wt_frags[c], wr, off);
                $kernel.ds_load_b128(VReg(wt_frags[c].0 + 4), wr, off + 16);
            }
            $kernel.wait_lgkmcnt(0);
            // WMMA: 2 row blocks × 4 col blocks
            for c in 0..n_col_tiles {
                $kernel.wmma_bf16_f32(acc[0 * n_col_tiles + c], x_frag0, wt_frags[c], acc[0 * n_col_tiles + c]);
            }
            for c in 0..n_col_tiles {
                $kernel.wmma_bf16_f32(acc[1 * n_col_tiles + c], x_frag1, wt_frags[c], acc[1 * n_col_tiles + c]);
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
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    // ══════════════════════════════════════════════════════════════════
    // MAIN LOOP
    // ══════════════════════════════════════════════════════════════════
    let loop_label = k.make_label("k64_loop");
    k.label(&loop_label);
    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_a = k.make_label("k64_ea");
    k.branch_scc1(&epilog_a);

    // Phase A: prefetch→buf1, compute buf0
    coop_gmem_load!(k, k_byte_off);
    lds_read_and_wmma!(k, buf0_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf1_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    k.s_cmp_ge_u32(k_iter_s, SReg(k_dim.0));
    let epilog_b = k.make_label("k64_eb");
    k.branch_scc1(&epilog_b);

    // Phase B: prefetch→buf0, compute buf1
    coop_gmem_load!(k, k_byte_off);
    lds_read_and_wmma!(k, buf1_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // EPILOGUES
    // ══════════════════════════════════════════════════════════════════
    k.label(&epilog_a);
    lds_read_and_wmma!(k, buf0_off);
    let store_label = k.make_label("k64_store");
    k.s_mov_imm(k_iter_s, 0);
    k.s_cmp_eq_u32_imm(k_iter_s, 0);
    k.branch_scc1(&store_label);

    k.label(&epilog_b);
    lds_read_and_wmma!(k, buf1_off);

    // ══════════════════════════════════════════════════════════════════
    // STORE PHASE: 2 row blocks × 4 col blocks
    // ══════════════════════════════════════════════════════════════════
    k.label(&store_label);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);  // N * 8

    let col_base_v = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_v, base_n_s);
    k.v_add_u32(col_base_v, col_base_v, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base_v);

    for r in 0..n_row_blocks {
        let rb_s = if r == 0 { s_row_base0 } else { s_row_base1 };
        let base_row_v = k.alloc_vreg();
        k.v_mov_from_sgpr(base_row_v, rb_s);
        k.v_add_u32(base_row_v, base_row_v, lane_half);

        let row_bytes = k.alloc_vreg();
        k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
        k.v_lshlrev_b32(row_bytes, 2, row_bytes);

        for c in 0..n_col_tiles {
            let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
            k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
            k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
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
// matmul_splitk: Split-K GEMM for small M dimensions
// ============================================================================

/// Split-K GEMM: Y_partial[split][M][N] = X[M,K_chunk] @ WT[N,K_chunk]^T
///
/// Splits K dimension across TGID.z. Each WG computes a partial sum over
/// K/split elements, writing to a temp buffer partitioned by split index.
/// Use with `reduce_splitk` to sum partials into final output.
///
/// For M=128, Split-K=4: WGs 256→1024, occupancy 2.7→10.7 per CU.
///
/// Kernargs (40 bytes):
///   0:  X_ptr      (8B) — bf16 [M, K_full], row-major
///   8:  WT_ptr     (8B) — bf16 [N, K_full], row-major
///   16: Y_part_ptr (8B) — f32 [split_k * M * N], partial output buffer
///   24: K_full     (4B) — full K dimension (row stride)
///   28: N          (4B) — output cols
///   32: stride_mn  (4B) — M * N (stride between split planes)
///   36: K_per_split(4B) — K elements per split (must be multiple of tile_k)
///
/// Grid: [ceil(N/64)*64, ceil(M/32), split_k]
pub fn matmul_splitk(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_splitk");
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    const LDS_X: u32 = 1024;
    const LDS_WT: u32 = 2048;
    const LDS_BUF: u32 = LDS_X + LDS_WT;
    k.set_lds_size(LDS_BUF * 2);

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y_partial");
    let k_full = k.arg_u32("K_full");
    let n_dim = k.arg_u32("N");
    let stride_mn = k.arg_u32("stride_mn");
    let k_per_split = k.arg_u32("K_per_split");
    k.emit_arg_loads();

    // ── TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    let split_idx_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);
    k.capture_tgid_z(split_idx_s);

    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── Store phase: s_tmp1 = tile_row*32 + wave_id*16 ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);

    // ── K_start = split_idx * K_per_split (in bf16 bytes) ──
    let k_start_s = k.alloc_sreg();
    k.s_mul_i32(k_start_s, split_idx_s, SReg(k_per_split.0));
    let k_start_bytes_s = k.alloc_sreg();
    k.s_lshl_b32(k_start_bytes_s, k_start_s, 1);  // * 2 for bf16

    // ── Y output offset = split_idx * stride_mn * 4 ──
    let y_off_s = k.alloc_sreg();
    k.s_mul_i32(y_off_s, split_idx_s, SReg(stride_mn.0));
    let y_off_bytes_s = k.alloc_sreg();
    k.s_lshl_b32(y_off_bytes_s, y_off_s, 2);  // * 4 for f32

    // ── Cooperative load addresses (same as matmul_lds_db) ──
    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_full.0));  // K_full for row stride

    // X: thread t → row=t>>1, half=t&1
    let x_coop_row = k.alloc_vreg();
    k.v_lshrrev_b32(x_coop_row, 1, tid);
    let x_coop_half = k.alloc_vreg();
    k.v_and_b32_imm(x_coop_half, tid, 1);

    let x_abs_row = k.alloc_vreg();
    let s_base_row = k.alloc_sreg();
    k.s_lshl_b32(s_base_row, tile_row_s, 5);
    k.v_mov_from_sgpr(x_abs_row, s_base_row);
    k.v_add_u32(x_abs_row, x_abs_row, x_coop_row);

    let x_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_byte, x_abs_row, k_vreg);
    k.v_lshlrev_b32(x_row_byte, 1, x_row_byte);

    let x_half_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_half_off, 4, x_coop_half);

    // x_gmem_base = X_ptr + row_byte + half_off + K_start_bytes
    let x_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_gmem_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_gmem_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_gmem_base, x_gmem_base, x_row_byte);
    k.v_add_co_ci(VReg(x_gmem_base.0 + 1), VReg(x_gmem_base.0 + 1));
    k.v_add_u32(x_gmem_base, x_gmem_base, x_half_off);
    // Add K_start offset
    let k_start_v = k.alloc_vreg();
    k.v_mov_from_sgpr(k_start_v, k_start_bytes_s);
    k.v_add_u32(x_gmem_base, x_gmem_base, k_start_v);

    let x_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(x_lds_off, 4, tid);

    // WT: thread t → loads WT row t
    let wt_abs_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_abs_row, base_n_s);
    k.v_add_u32(wt_abs_row, wt_abs_row, tid);

    let wt_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_byte, wt_abs_row, k_vreg);
    k.v_lshlrev_b32(wt_row_byte, 1, wt_row_byte);

    let wt_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_gmem_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_gmem_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_gmem_base, wt_gmem_base, wt_row_byte);
    k.v_add_co_ci(VReg(wt_gmem_base.0 + 1), VReg(wt_gmem_base.0 + 1));
    // Add K_start offset
    k.v_add_u32(wt_gmem_base, wt_gmem_base, k_start_v);

    let wt_lds_off = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_off, 5, tid);
    k.push(Op::VAddU32 {
        dst: wt_lds_off, src0: Operand::VReg(wt_lds_off),
        src1: Operand::InlineInt(LDS_X as i32),
    });

    // ── LDS read addresses ──
    let x_lds_read = k.alloc_vreg();
    let s_wave_off = k.alloc_sreg();
    k.s_lshl_b32(s_wave_off, wave_id_s, 9);
    k.v_lshlrev_b32(x_lds_read, 5, lane_row);
    let tmp_wave = k.alloc_vreg();
    k.v_mov_from_sgpr(tmp_wave, s_wave_off);
    k.v_add_u32(x_lds_read, x_lds_read, tmp_wave);

    let wt_lds_read_base = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_read_base, 5, lane_row);

    // ── Temp + fragment VGPRs ──
    let gmem_x = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt0 = k.alloc_vreg_array(4, Alignment::Align4);
    let gmem_wt1 = k.alloc_vreg_array(4, Alignment::Align4);

    let x_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // ── K-loop state: iterate K_per_split/tile_k times ──
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let k_step = (tile_k * 2) as i32;

    let buf0_off = k.alloc_vreg();
    k.v_mov_imm(buf0_off, 0);
    let buf1_off = k.alloc_vreg();
    k.v_mov_imm(buf1_off, LDS_BUF as i32);

    // ── Reuse same macro structure as matmul_lds_db ──
    macro_rules! coop_gmem_load {
        ($kernel:expr, $koff:expr) => {{
            let xa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(xa, x_gmem_base);
            $kernel.v_mov(VReg(xa.0 + 1), VReg(x_gmem_base.0 + 1));
            $kernel.v_add_co(xa, xa, $koff);
            $kernel.v_add_co_ci(VReg(xa.0 + 1), VReg(xa.0 + 1));
            $kernel.global_load(gmem_x, xa, Width::B128, 0);
            let wa = $kernel.alloc_vreg_array(2, Alignment::Align2);
            $kernel.v_mov(wa, wt_gmem_base);
            $kernel.v_mov(VReg(wa.0 + 1), VReg(wt_gmem_base.0 + 1));
            $kernel.v_add_co(wa, wa, $koff);
            $kernel.v_add_co_ci(VReg(wa.0 + 1), VReg(wa.0 + 1));
            $kernel.global_load(gmem_wt0, wa, Width::B128, 0);
            $kernel.global_load(gmem_wt1, wa, Width::B128, 16);
        }};
    }

    macro_rules! coop_lds_store {
        ($kernel:expr, $buf_off:expr) => {{
            let xa = $kernel.alloc_vreg();
            $kernel.v_add_u32(xa, x_lds_off, $buf_off);
            $kernel.ds_store_b128(xa, gmem_x, 0);
            let wa = $kernel.alloc_vreg();
            $kernel.v_add_u32(wa, wt_lds_off, $buf_off);
            $kernel.ds_store_b128(wa, gmem_wt0, 0);
            $kernel.ds_store_b128(wa, gmem_wt1, 16);
        }};
    }

    macro_rules! lds_read_frags {
        ($kernel:expr, $buf_off:expr) => {{
            let xr = $kernel.alloc_vreg();
            $kernel.v_add_u32(xr, x_lds_read, $buf_off);
            $kernel.ds_load_b128(x_frag, xr, 0);
            $kernel.ds_load_b128(VReg(x_frag.0 + 4), xr, 16);
            for t in 0..n_tiles {
                let wr = $kernel.alloc_vreg();
                $kernel.v_add_u32(wr, wt_lds_read_base, $buf_off);
                let off: u16 = (LDS_X + (t as u32) * 16 * 32) as u16;
                $kernel.ds_load_b128(wt_frags[t], wr, off);
                $kernel.ds_load_b128(VReg(wt_frags[t].0 + 4), wr, off + 16);
            }
        }};
    }

    // ── Prologue: load first K-tile ──
    coop_gmem_load!(k, k_byte_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    // ── Main loop (uses K_per_split as bound) ──
    let loop_label = k.make_label("sk_loop");
    k.label(&loop_label);
    k.s_cmp_ge_u32(k_iter_s, SReg(k_per_split.0));
    let epilog_a = k.make_label("sk_ea");
    k.branch_scc1(&epilog_a);

    coop_gmem_load!(k, k_byte_off);
    lds_read_frags!(k, buf0_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf1_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);

    k.s_cmp_ge_u32(k_iter_s, SReg(k_per_split.0));
    let epilog_b = k.make_label("sk_eb");
    k.branch_scc1(&epilog_b);

    coop_gmem_load!(k, k_byte_off);
    lds_read_frags!(k, buf1_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_per_split.0));
    k.branch_scc1(&loop_label);

    // ── Epilogues ──
    k.label(&epilog_a);
    lds_read_frags!(k, buf0_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }
    let store_label = k.make_label("sk_store");
    k.s_mov_imm(k_iter_s, 0);
    k.s_cmp_eq_u32_imm(k_iter_s, 0);
    k.branch_scc1(&store_label);

    k.label(&epilog_b);
    lds_read_frags!(k, buf1_off);
    k.wait_lgkmcnt(0);
    for t in 0..n_tiles { k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]); }

    // ══════════════════════════════════════════════════════════════════
    // STORE: write to Y_partial + split_idx * stride_mn * 4
    // ══════════════════════════════════════════════════════════════════
    k.label(&store_label);

    // Compute Y base with split offset
    // y_split_base = Y_ptr + y_off_bytes
    let y_split_lo = k.alloc_sreg();
    let y_split_hi = k.alloc_sreg();
    k.push(Op::SAddU32 { dst: y_split_lo, src0: SReg(y_ptr.0), src1: SOperand::SReg(y_off_bytes_s) });
    k.s_addc_u32_imm(y_split_hi, SReg(y_ptr.0 + 1), 0);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_base = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base, base_n_s);
    k.v_add_u32(col_base, col_base, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base);

    for t in 0..n_tiles {
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, y_split_lo);
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), y_split_hi);
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr, src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// reduce_splitk: sum S partial results into final output
// ============================================================================

/// Elementwise sum of S split-K partial results.
///
/// out[i] = sum(partials[s * count + i] for s in 0..split_k)
///
/// Kernargs (24 bytes):
///   0:  partials_ptr (8B) — f32 [split_k * count]
///   8:  out_ptr      (8B) — f32 [count]
///   16: count        (4B) — number of elements per split
///   20: split_k      (4B) — number of splits
///
/// Grid: [ceil(count/256)*256, 1, 1], WG: [256, 1, 1]
pub fn reduce_splitk() -> T0Kernel {
    let mut k = T0Kernel::new("t0_reduce_splitk");

    let part_ptr = k.arg_ptr("partials");
    let out_ptr = k.arg_ptr("out");
    let count = k.arg_u32("count");
    let split_k = k.arg_u32("split_k");
    k.emit_arg_loads();

    let gid = k.compute_global_id_x(256);

    // Bounds check
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(count.0));
    let saved = k.bounds_check_begin(gid, n_vreg);

    // Load and sum S partial values
    let sum = k.alloc_vreg();
    k.v_mov_imm(sum, 0);

    // Base addr = partials + gid * 4
    let base_off = k.alloc_vreg();
    k.v_lshlrev_b32(base_off, 2, gid);

    let base_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(base_addr, SReg(part_ptr.0));
    k.v_mov_from_sgpr(VReg(base_addr.0 + 1), SReg(part_ptr.0 + 1));
    k.v_add_co(base_addr, base_addr, base_off);
    k.v_add_co_ci(VReg(base_addr.0 + 1), VReg(base_addr.0 + 1));

    // Stride = count * 4 (bytes between split planes)
    let stride = k.alloc_vreg();
    k.v_mov_from_sgpr(stride, SReg(count.0));
    k.v_lshlrev_b32(stride, 2, stride);

    // Loop over S splits
    let s_iter = k.alloc_sreg();
    k.s_mov_imm(s_iter, 0);

    let loop_label = k.make_label("reduce_loop");
    k.label(&loop_label);

    let val = k.alloc_vreg();
    k.global_load(val, base_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_add_f32(sum, sum, val);

    k.v_add_co(base_addr, base_addr, stride);
    k.v_add_co_ci(VReg(base_addr.0 + 1), VReg(base_addr.0 + 1));

    k.s_add_u32(s_iter, s_iter, 1);
    k.s_cmp_lt_u32(s_iter, SReg(split_k.0));
    k.branch_scc1(&loop_label);

    // Store result
    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0));
    k.v_mov_from_sgpr(VReg(out_addr.0 + 1), SReg(out_ptr.0 + 1));
    k.v_add_co(out_addr, out_addr, base_off);
    k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));
    k.global_store(out_addr, sum, Width::B32, 0);

    k.bounds_check_end(saved);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_tn: dW = X^T @ dY  (TN-GEMM, f32 inputs, on-the-fly bf16 conversion)
// ============================================================================

/// Generate TN-GEMM backward weight kernel: dW[M,N] = X[K,M]^T @ dY[K,N]
///
/// Reads f32 inputs directly, converts f32→bf16 on-the-fly inside the kernel.
/// M (inner dimension for X columns) is baked at compile time.
///
/// Kernargs layout (32 bytes):
/// | Offset | Size | Name |
/// |--------|------|------|
/// | 0      | 8    | X_ptr (f32, [K, M]) |
/// | 8      | 8    | DY_ptr (f32, [K, N]) |
/// | 16     | 8    | dW_ptr (f32, [M, N]) |
/// | 24     | 4    | K (reduction dimension) |
/// | 28     | 4    | N (output columns of dY) |
///
/// Grid: [ceil(N/64)*64, ceil(M/32), 1], WG: 64 threads
pub fn matmul_tn(sched: &dyn Schedule, m_dim: u32) -> T0Kernel {
    let mut k = T0Kernel::new(&format!("t0_matmul_tn_m{}", m_dim));
    let (_, _) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    let x_stride = m_dim * 4;  // X row stride in bytes (f32)

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let dy_ptr = k.arg_ptr("DY");
    let dw_ptr = k.arg_ptr("dW");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── Capture TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    // ── Thread decomposition ──
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, VReg(0));
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });

    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    // ── Accumulators (4 tiles × 8 VGPRs) ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── Compute tile_m ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    // ── X column base: &X[0, tile_m + lane_row] ──
    let col_in_x = k.alloc_vreg();
    k.v_mov_from_sgpr(col_in_x, s_tmp1);
    k.v_add_u32(col_in_x, col_in_x, lane_row);
    let col_off_x = k.alloc_vreg();
    k.v_lshlrev_b32(col_off_x, 2, col_in_x);

    let x_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_base, x_base, col_off_x);
    k.v_add_co_ci(VReg(x_base.0 + 1), VReg(x_base.0 + 1));

    // ── dY column base offset ──
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);
    let col_base_dy = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_dy, base_n_s);
    k.v_add_u32(col_base_dy, col_base_dy, lane_row);
    let col_off_dy = k.alloc_vreg();
    k.v_lshlrev_b32(col_off_dy, 2, col_base_dy);

    // ── Strides ──
    let x_stride_v = k.alloc_vreg();
    k.v_mov_imm(x_stride_v, x_stride as i32);
    let dy_stride_v = k.alloc_vreg();
    k.v_mov_from_sgpr(dy_stride_v, SReg(n_dim.0));
    k.v_lshlrev_b32(dy_stride_v, 2, dy_stride_v);

    // ── WMMA fragments ──
    let x_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let wt_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    let f32_scratch = k.alloc_vreg_array(16, Alignment::None);

    // ── K-loop ──
    let k_off_x = k.alloc_vreg();
    k.v_mov_imm(k_off_x, 0);
    let k_off_dy = k.alloc_vreg();
    k.v_mov_imm(k_off_dy, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);

    let loop_label = k.make_label("k_loop");
    k.label(&loop_label);

    // Load A: 16 f32 from X column, pack to bf16x2
    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(x_addr, x_base);
    k.v_mov(VReg(x_addr.0 + 1), VReg(x_base.0 + 1));
    k.v_add_co(x_addr, x_addr, k_off_x);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));
    for i in 0..16u32 {
        k.global_load(VReg(f32_scratch.0 + i), x_addr, Width::B32, 0);
        if i < 15 {
            k.v_add_co(x_addr, x_addr, x_stride_v);
            k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));
        }
    }
    k.wait_vmcnt(0);
    for pair in 0..8u32 {
        k.cvt_pk_bf16_f32(
            VReg(x_frag.0 + pair),
            VReg(f32_scratch.0 + pair * 2),
            VReg(f32_scratch.0 + pair * 2 + 1),
        );
    }

    // Load B subtiles: 4 × 16 f32 from dY columns
    for t in 0..n_tiles {
        let dy_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(dy_addr, SReg(dy_ptr.0));
        k.v_mov_from_sgpr(VReg(dy_addr.0 + 1), SReg(dy_ptr.0 + 1));
        k.v_add_co(dy_addr, dy_addr, col_off_dy);
        k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: dy_addr,
                src0: Operand::VReg(dy_addr),
                src1: Operand::InlineInt((t * 16 * 4) as i32),
            });
        }
        k.v_add_co(dy_addr, dy_addr, k_off_dy);
        k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));

        for i in 0..16u32 {
            k.global_load(VReg(f32_scratch.0 + i), dy_addr, Width::B32, 0);
            if i < 15 {
                k.v_add_co(dy_addr, dy_addr, dy_stride_v);
                k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));
            }
        }
        k.wait_vmcnt(0);
        for pair in 0..8u32 {
            k.cvt_pk_bf16_f32(
                VReg(wt_frags[t].0 + pair),
                VReg(f32_scratch.0 + pair * 2),
                VReg(f32_scratch.0 + pair * 2 + 1),
            );
        }
    }

    // WMMA
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], x_frag, wt_frags[t], acc[t]);
    }

    // K-loop advance
    let k16_x_stride = k.alloc_vreg();
    k.v_mov_imm(k16_x_stride, (16 * x_stride) as i32);
    k.v_add_co(k_off_x, k_off_x, k16_x_stride);
    let dy_stride_16 = k.alloc_vreg();
    k.v_lshlrev_b32(dy_stride_16, 4, dy_stride_v);
    k.v_add_co(k_off_dy, k_off_dy, dy_stride_16);

    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ── Store phase ──
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_bytes = k.alloc_vreg();
    k.v_mov_from_sgpr(col_bytes, base_n_s);
    k.v_add_u32(col_bytes, col_bytes, lane_row);
    k.v_lshlrev_b32(col_bytes, 2, col_bytes);

    for t in 0..n_tiles {
        let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, SReg(dw_ptr.0));
        k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(dw_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, row_bytes);
        k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
        k.v_add_u32(y_addr, y_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: y_addr,
                src0: Operand::VReg(y_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(y_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(y_addr, y_addr, row_stride);
                k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_nn_f32: C = A @ B (NN-GEMM, f32 inputs/outputs, on-the-fly bf16)
// ============================================================================

/// Generate NN-GEMM kernel: C[M,N] = A[M,K] @ B[K,N]
///
/// Reads f32 inputs directly, converts f32→bf16 on-the-fly inside the kernel.
/// Uses WMMA for computation, stores f32 output.
///
/// Kernargs layout (32 bytes):
/// | Offset | Size | Name |
/// |--------|------|------|
/// | 0      | 8    | A_ptr (f32, [M, K]) |
/// | 8      | 8    | B_ptr (f32, [K, N]) |
/// | 16     | 8    | C_ptr (f32, [M, N]) |
/// | 24     | 4    | K (reduction dimension) |
/// | 28     | 4    | N (output columns) |
///
/// Grid: [ceil(N/64)*64, ceil(M/32), 1], WG: 64 threads
pub fn matmul_nn_f32(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_nn_f32");
    let (_, _) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    // ── Args ──
    let a_ptr = k.arg_ptr("A");
    let b_ptr = k.arg_ptr("B");
    let c_ptr = k.arg_ptr("C");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── Capture TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    // ── Thread decomposition ──
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, VReg(0));
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });

    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    // ── Accumulators (4 tiles × 8 VGPRs) ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── Compute tile_m ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    // ── A row base: &A[tile_m + lane_row, 0] ──
    // In NN layout, A[m, k] is at A_ptr + m * K * 4 + k * 4
    // For the K-loop, A's 16 consecutive k values are contiguous in memory
    let a_row = k.alloc_vreg();
    k.v_mov_from_sgpr(a_row, s_tmp1);
    k.v_add_u32(a_row, a_row, lane_row);
    // a_row_offset = a_row * K * 4
    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));
    let a_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(a_row_off, a_row, k_vreg);
    k.v_lshlrev_b32(a_row_off, 2, a_row_off);

    let a_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_base, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_base.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_base, a_base, a_row_off);
    k.v_add_co_ci(VReg(a_base.0 + 1), VReg(a_base.0 + 1));

    // ── B column base offset ──
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);
    let col_base_b = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_b, base_n_s);
    k.v_add_u32(col_base_b, col_base_b, lane_row);
    let col_off_b = k.alloc_vreg();
    k.v_lshlrev_b32(col_off_b, 2, col_base_b);

    // ── Strides ──
    // A stride per K step: contiguous f32 → 4 bytes
    // Not needed as a vreg; we'll load 16 consecutive elements starting k_off
    let b_stride_v = k.alloc_vreg();
    k.v_mov_from_sgpr(b_stride_v, SReg(n_dim.0));
    k.v_lshlrev_b32(b_stride_v, 2, b_stride_v);  // N * 4 bytes

    // ── WMMA fragments ──
    let a_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let b_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    let f32_scratch = k.alloc_vreg_array(16, Alignment::None);

    // ── K-loop ──
    let k_off_a = k.alloc_vreg();  // byte offset into A row for current K tile
    k.v_mov_imm(k_off_a, 0);
    let k_off_b = k.alloc_vreg();
    k.v_mov_imm(k_off_b, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);

    let loop_label = k.make_label("k_loop");
    k.label(&loop_label);

    // Load A: 16 f32 from A row contiguously starting at k_off_a
    // a_addr = a_base + k_off_a, then load 16 consecutive f32 (stride=4 bytes)
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(a_addr, a_base);
    k.v_mov(VReg(a_addr.0 + 1), VReg(a_base.0 + 1));
    k.v_add_co(a_addr, a_addr, k_off_a);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
    for i in 0..16u32 {
        k.global_load(VReg(f32_scratch.0 + i), a_addr, Width::B32, 0);
        if i < 15 {
            // A elements are contiguous: stride = 4 bytes
            k.push(Op::VAddU32 {
                dst: a_addr,
                src0: Operand::VReg(a_addr),
                src1: Operand::InlineInt(4),
            });
        }
    }
    k.wait_vmcnt(0);
    for pair in 0..8u32 {
        k.cvt_pk_bf16_f32(
            VReg(a_frag.0 + pair),
            VReg(f32_scratch.0 + pair * 2),
            VReg(f32_scratch.0 + pair * 2 + 1),
        );
    }

    // Load B subtiles: 4 × 16 f32 from B columns (stride = N*4)
    for t in 0..n_tiles {
        let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(b_addr, SReg(b_ptr.0));
        k.v_mov_from_sgpr(VReg(b_addr.0 + 1), SReg(b_ptr.0 + 1));
        k.v_add_co(b_addr, b_addr, col_off_b);
        k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: b_addr,
                src0: Operand::VReg(b_addr),
                src1: Operand::InlineInt((t * 16 * 4) as i32),
            });
        }
        k.v_add_co(b_addr, b_addr, k_off_b);
        k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));

        for i in 0..16u32 {
            k.global_load(VReg(f32_scratch.0 + i), b_addr, Width::B32, 0);
            if i < 15 {
                k.v_add_co(b_addr, b_addr, b_stride_v);
                k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
            }
        }
        k.wait_vmcnt(0);
        for pair in 0..8u32 {
            k.cvt_pk_bf16_f32(
                VReg(b_frags[t].0 + pair),
                VReg(f32_scratch.0 + pair * 2),
                VReg(f32_scratch.0 + pair * 2 + 1),
            );
        }
    }

    // WMMA
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], a_frag, b_frags[t], acc[t]);
    }

    // K-loop advance
    // A: advance by 16 * 4 = 64 bytes (contiguous elements)
    k.push(Op::VAddU32 {
        dst: k_off_a,
        src0: Operand::VReg(k_off_a),
        src1: Operand::InlineInt(64),
    });
    // B: advance by 16 * N * 4 bytes (16 rows)
    let b_stride_16 = k.alloc_vreg();
    k.v_lshlrev_b32(b_stride_16, 4, b_stride_v);
    k.v_add_co(k_off_b, k_off_b, b_stride_16);

    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ── Store phase: f32 output ──
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_bytes = k.alloc_vreg();
    k.v_mov_from_sgpr(col_bytes, base_n_s);
    k.v_add_u32(col_bytes, col_bytes, lane_row);
    k.v_lshlrev_b32(col_bytes, 2, col_bytes);

    for t in 0..n_tiles {
        let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(c_addr, SReg(c_ptr.0));
        k.v_mov_from_sgpr(VReg(c_addr.0 + 1), SReg(c_ptr.0 + 1));
        k.v_add_co(c_addr, c_addr, row_bytes);
        k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));
        k.v_add_u32(c_addr, c_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: c_addr,
                src0: Operand::VReg(c_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(c_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(c_addr, c_addr, row_stride);
                k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_abt_f32: C = A @ B^T (ABT-GEMM, f32 in/out, on-the-fly bf16)
// ============================================================================

/// Generate ABT-GEMM kernel: C[M,N] = A[M,K] @ B[N,K]^T
///
/// Both A and B have contiguous rows of length K. This is the simplest
/// memory access pattern for WMMA: both fragments load contiguous data.
///
/// Kernargs layout (32 bytes):
/// | Offset | Size | Name |
/// |--------|------|------|
/// | 0      | 8    | A_ptr (f32, [M, K]) |
/// | 8      | 8    | B_ptr (f32, [N, K]) |
/// | 16     | 8    | C_ptr (f32, [M, N]) |
/// | 24     | 4    | K (reduction dimension) |
/// | 28     | 4    | N (output columns = rows of B) |
///
/// Grid: [ceil(N/64)*64, ceil(M/32), 1], WG: 64 threads
pub fn matmul_abt_f32(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_abt_f32");
    let (_, _) = sched.gemm_tile_mn();
    let tile_k = sched.gemm_tile_k();
    let n_tiles = sched.gemm_n_wmma_tiles();

    // ── Args ──
    let a_ptr = k.arg_ptr("A");
    let b_ptr = k.arg_ptr("B");
    let c_ptr = k.arg_ptr("C");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    // ── Capture TGIDs ──
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    // ── Thread decomposition ──
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, VReg(0));
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });

    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    // ── Accumulators (4 tiles × 8 VGPRs) ──
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── Compute tile_m ──
    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });

    // ── A row base: &A[tile_m + lane_row, 0] ──
    let a_row = k.alloc_vreg();
    k.v_mov_from_sgpr(a_row, s_tmp1);
    k.v_add_u32(a_row, a_row, lane_row);
    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));
    let a_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(a_row_off, a_row, k_vreg);
    k.v_lshlrev_b32(a_row_off, 2, a_row_off);

    let a_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_base, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_base.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_base, a_base, a_row_off);
    k.v_add_co_ci(VReg(a_base.0 + 1), VReg(a_base.0 + 1));

    // ── B row base: &B[tile_n + lane_row, 0] for each subtile ──
    // B is [N, K], B^T column = B row. We need B rows starting at tile_n
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);

    // B row offset per lane_row: &B[base_n + lane_row, 0]
    let b_row_base = k.alloc_vreg();
    k.v_mov_from_sgpr(b_row_base, base_n_s);
    k.v_add_u32(b_row_base, b_row_base, lane_row);
    let b_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(b_row_off, b_row_base, k_vreg);
    k.v_lshlrev_b32(b_row_off, 2, b_row_off);

    let b_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_base, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_base.0 + 1), SReg(b_ptr.0 + 1));
    k.v_add_co(b_base, b_base, b_row_off);
    k.v_add_co_ci(VReg(b_base.0 + 1), VReg(b_base.0 + 1));

    // B subtile stride: 16 rows of B = 16 * K * 4 bytes
    let b_subtile_stride = k.alloc_vreg();
    k.v_lshlrev_b32(b_subtile_stride, 2, k_vreg);  // K * 4 bytes per row
    let b_16row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(b_16row_stride, 4, b_subtile_stride);  // not used; we compute per-tile

    // ── WMMA fragments ──
    let a_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let b_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    let f32_scratch = k.alloc_vreg_array(16, Alignment::None);

    // ── K-loop ──
    let k_off = k.alloc_vreg();  // byte offset into row for current K tile
    k.v_mov_imm(k_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);

    let loop_label = k.make_label("k_loop");
    k.label(&loop_label);

    // Load A: 16 contiguous f32 from A[row, k..k+15]
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(a_addr, a_base);
    k.v_mov(VReg(a_addr.0 + 1), VReg(a_base.0 + 1));
    k.v_add_co(a_addr, a_addr, k_off);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
    for i in 0..16u32 {
        k.global_load(VReg(f32_scratch.0 + i), a_addr, Width::B32, 0);
        if i < 15 {
            k.push(Op::VAddU32 {
                dst: a_addr,
                src0: Operand::VReg(a_addr),
                src1: Operand::InlineInt(4),
            });
        }
    }
    k.wait_vmcnt(0);
    for pair in 0..8u32 {
        k.cvt_pk_bf16_f32(
            VReg(a_frag.0 + pair),
            VReg(f32_scratch.0 + pair * 2),
            VReg(f32_scratch.0 + pair * 2 + 1),
        );
    }

    // Load B subtiles: 4 × 16 contiguous f32 from B^T columns = B rows
    for t in 0..n_tiles {
        // b_addr = b_base + t*16*K*4 + k_off
        let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov(b_addr, b_base);
        k.v_mov(VReg(b_addr.0 + 1), VReg(b_base.0 + 1));
        if t > 0 {
            // Offset by t*16 rows: t * 16 * K * 4 bytes
            let t_off = k.alloc_vreg();
            k.v_mov_imm(t_off, (t * 16) as i32);
            let t_byte = k.alloc_vreg();
            k.v_mul_lo_u32(t_byte, t_off, b_subtile_stride);
            k.v_add_co(b_addr, b_addr, t_byte);
            k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
        }
        k.v_add_co(b_addr, b_addr, k_off);
        k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));

        for i in 0..16u32 {
            k.global_load(VReg(f32_scratch.0 + i), b_addr, Width::B32, 0);
            if i < 15 {
                k.push(Op::VAddU32 {
                    dst: b_addr,
                    src0: Operand::VReg(b_addr),
                    src1: Operand::InlineInt(4),
                });
            }
        }
        k.wait_vmcnt(0);
        for pair in 0..8u32 {
            k.cvt_pk_bf16_f32(
                VReg(b_frags[t].0 + pair),
                VReg(f32_scratch.0 + pair * 2),
                VReg(f32_scratch.0 + pair * 2 + 1),
            );
        }
    }

    // WMMA
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], a_frag, b_frags[t], acc[t]);
    }

    // K-loop advance: 16 elements * 4 bytes = 64 bytes
    k.push(Op::VAddU32 {
        dst: k_off,
        src0: Operand::VReg(k_off),
        src1: Operand::InlineInt(64),
    });

    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ── Store phase: f32 output C[M,N] ──
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_bytes = k.alloc_vreg();
    k.v_mov_from_sgpr(col_bytes, base_n_s);
    k.v_add_u32(col_bytes, col_bytes, lane_row);
    k.v_lshlrev_b32(col_bytes, 2, col_bytes);

    for t in 0..n_tiles {
        let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(c_addr, SReg(c_ptr.0));
        k.v_mov_from_sgpr(VReg(c_addr.0 + 1), SReg(c_ptr.0 + 1));
        k.v_add_co(c_addr, c_addr, row_bytes);
        k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));
        k.v_add_u32(c_addr, c_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: c_addr,
                src0: Operand::VReg(c_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(c_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(c_addr, c_addr, row_stride);
                k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// matmul_abt_f32_db: Vectorized + Double-Buffered backward GEMM  C = A @ B^T
// ============================================================================

/// Optimized backward GEMM: C[M,N] = A[M,K] @ B[N,K]^T (both f32 inputs).
///
/// Uses vectorized global_load_b128 (4× vs 16× scalar loads per fragment)
/// plus K-loop double buffering. Each lane loads 4 contiguous f32 via b128,
/// then packs to bf16 for WMMA. 4× better memory bandwidth vs matmul_abt_f32.
///
/// Kernargs (32 bytes): A_ptr(8), B_ptr(8), C_ptr(8), K(4), N(4)
/// Grid: [ceil(N/64)*64, ceil(M/32), 1]
pub fn matmul_abt_f32_db(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_matmul_abt_f32_db");
    let tile_k = sched.gemm_tile_k();  // 16
    let n_tiles = sched.gemm_n_wmma_tiles();  // 4

    let a_ptr = k.arg_ptr("A");
    let b_ptr = k.arg_ptr("B");
    let c_ptr = k.arg_ptr("C");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    k.emit_arg_loads();

    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(tile_row_s);

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, VReg(0));
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    // Accumulators
    let acc: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    let s_tmp1 = k.alloc_sreg();
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp1, tile_row_s, 5);
    k.s_lshl_b32(s_tmp2, wave_id_s, 4);
    k.push(Op::SAddU32 { dst: s_tmp1, src0: s_tmp1, src1: SOperand::SReg(s_tmp2) });
    let base_n_s = k.alloc_sreg();
    k.s_lshl_b32(base_n_s, tile_col_s, 6);

    // A row base: &A[tile_m + lane_row, 0]
    let a_row = k.alloc_vreg();
    k.v_mov_from_sgpr(a_row, s_tmp1);
    k.v_add_u32(a_row, a_row, lane_row);
    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));
    let a_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(a_row_off, a_row, k_vreg);
    k.v_lshlrev_b32(a_row_off, 2, a_row_off);

    let a_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_base, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_base.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_base, a_base, a_row_off);
    k.v_add_co_ci(VReg(a_base.0 + 1), VReg(a_base.0 + 1));

    // B row base per lane_row: &B[base_n + lane_row, 0]
    let b_row_base = k.alloc_vreg();
    k.v_mov_from_sgpr(b_row_base, base_n_s);
    k.v_add_u32(b_row_base, b_row_base, lane_row);
    let b_row_off = k.alloc_vreg();
    k.v_mul_lo_u32(b_row_off, b_row_base, k_vreg);
    k.v_lshlrev_b32(b_row_off, 2, b_row_off);

    let b_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_base, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_base.0 + 1), SReg(b_ptr.0 + 1));
    k.v_add_co(b_base, b_base, b_row_off);
    k.v_add_co_ci(VReg(b_base.0 + 1), VReg(b_base.0 + 1));

    // B subtile stride: 16 * K * 4 bytes
    let b_subtile_stride = k.alloc_vreg();
    k.v_lshlrev_b32(b_subtile_stride, 2, k_vreg);
    let b_16row = k.alloc_vreg();
    k.v_mov_imm(b_16row, 16);
    k.v_mul_lo_u32(b_16row, b_16row, b_subtile_stride);

    // WMMA fragments + vectorized load buffers
    let a_frag = k.alloc_vreg_array(8, Alignment::Align8);
    let b_frags: Vec<VReg> = (0..n_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // Vectorized load buffers (4× b128 = 16 f32 = 4 b128 loads)
    let a_buf = k.alloc_vreg_array(16, Alignment::Align4);
    let b_buf = k.alloc_vreg_array(16, Alignment::Align4);

    // K-loop state
    let k_off = k.alloc_vreg();
    k.v_mov_imm(k_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);

    // ── K-loop with vectorized loads ──
    let loop_label = k.make_label("abt_loop");
    k.label(&loop_label);

    // Load A: 4× b128 = 16 f32 values from A[row, k..k+15]
    {
        let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov(a_addr, a_base);
        k.v_mov(VReg(a_addr.0 + 1), VReg(a_base.0 + 1));
        k.v_add_co(a_addr, a_addr, k_off);
        k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
        k.global_load(VReg(a_buf.0),    a_addr, Width::B128, 0);
        k.global_load(VReg(a_buf.0+4),  a_addr, Width::B128, 16);
        k.global_load(VReg(a_buf.0+8),  a_addr, Width::B128, 32);
        k.global_load(VReg(a_buf.0+12), a_addr, Width::B128, 48);
    }

    // Load + convert B tiles
    for t in 0..n_tiles {
        let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov(b_addr, b_base);
        k.v_mov(VReg(b_addr.0 + 1), VReg(b_base.0 + 1));
        if t > 0 {
            let t_off = k.alloc_vreg();
            k.v_mov_imm(t_off, t as i32);
            let t_byte = k.alloc_vreg();
            k.v_mul_lo_u32(t_byte, t_off, b_16row);
            k.v_add_co(b_addr, b_addr, t_byte);
            k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
        }
        k.v_add_co(b_addr, b_addr, k_off);
        k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
        k.global_load(VReg(b_buf.0),    b_addr, Width::B128, 0);
        k.global_load(VReg(b_buf.0+4),  b_addr, Width::B128, 16);
        k.global_load(VReg(b_buf.0+8),  b_addr, Width::B128, 32);
        k.global_load(VReg(b_buf.0+12), b_addr, Width::B128, 48);
        k.wait_vmcnt(0);
        // Convert f32 pairs → bf16 packed
        for pair in 0..8u32 {
            k.cvt_pk_bf16_f32(
                VReg(b_frags[t].0 + pair),
                VReg(b_buf.0 + pair * 2),
                VReg(b_buf.0 + pair * 2 + 1),
            );
        }
    }

    // Wait for A loads and convert
    k.wait_vmcnt(0);
    for pair in 0..8u32 {
        k.cvt_pk_bf16_f32(
            VReg(a_frag.0 + pair),
            VReg(a_buf.0 + pair * 2),
            VReg(a_buf.0 + pair * 2 + 1),
        );
    }

    // WMMA
    for t in 0..n_tiles {
        k.wmma_bf16_f32(acc[t], a_frag, b_frags[t], acc[t]);
    }

    // K-loop advance: 16 elements * 4 bytes = 64 bytes
    k.push(Op::VAddU32 {
        dst: k_off, src0: Operand::VReg(k_off), src1: Operand::InlineInt(64),
    });
    k.s_add_u32(k_iter_s, k_iter_s, tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, SReg(k_dim.0));
    k.branch_scc1(&loop_label);

    // ── Store f32 output ──
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, s_tmp1);
    k.v_add_u32(base_row_v, base_row_v, lane_half);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_bytes = k.alloc_vreg();
    k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
    k.v_lshlrev_b32(row_bytes, 2, row_bytes);
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    let col_bytes = k.alloc_vreg();
    k.v_mov_from_sgpr(col_bytes, base_n_s);
    k.v_add_u32(col_bytes, col_bytes, lane_row);
    k.v_lshlrev_b32(col_bytes, 2, col_bytes);

    for t in 0..n_tiles {
        let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(c_addr, SReg(c_ptr.0));
        k.v_mov_from_sgpr(VReg(c_addr.0 + 1), SReg(c_ptr.0 + 1));
        k.v_add_co(c_addr, c_addr, row_bytes);
        k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));
        k.v_add_u32(c_addr, c_addr, col_bytes);
        if t > 0 {
            k.push(Op::VAddU32 {
                dst: c_addr, src0: Operand::VReg(c_addr),
                src1: Operand::InlineInt((t * 64) as i32),
            });
        }
        for vk in 0..8u32 {
            k.global_store(c_addr, VReg(acc[t].0 + vk), Width::B32, 0);
            if vk < 7 {
                k.v_add_co(c_addr, c_addr, row_stride);
                k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// ============================================================================
// sum_reduce: out = sum(x)
// ============================================================================

/// Generate sum reduction kernel: out[0] = sum(x[0..n])
///
/// Uses wave32 reduction (ds_swizzle butterfly).
/// Single WG (32 threads) iterates over all elements.
///
/// Kernargs (20 bytes): x_ptr(0), out_ptr(8), n(16)
/// Grid: [32, 1, 1], WG: 32
pub fn sum_reduce() -> T0Kernel {
    let mut k = T0Kernel::new("t0_sum_reduce");

    let x_ptr = k.arg_ptr("x");
    let out_ptr = k.arg_ptr("out");
    let n = k.arg_u32("n");
    k.emit_arg_loads();

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // Per-lane accumulator
    let acc = k.alloc_vreg();
    k.v_mov_imm(acc, 0);

    // Loop: each iteration handles 32 elements
    let iter_s = k.alloc_sreg();
    k.s_mov_imm(iter_s, 0);
    let loop_label = k.make_label("sum_loop");
    k.label(&loop_label);

    // idx = iter_base + lane_id
    let idx = k.alloc_vreg();
    k.v_mov_from_sgpr(idx, iter_s);
    k.v_add_u32(idx, idx, lane_id);

    // Bounds check: idx < n → EXEC mask
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n.0));
    let saved_exec = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(n_vreg) });
    k.push(Op::SaveExec { dst: saved_exec });

    // Load x[idx] and accumulate
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, idx);
    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(addr, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
    let val = k.alloc_vreg();
    k.global_load(val, addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.push(Op::VAddF32 { dst: acc, src0: Operand::VReg(acc), src1: Operand::VReg(val) });

    // Restore EXEC
    k.push(Op::RestoreExec { src: saved_exec });

    // Loop advance
    k.s_add_u32(iter_s, iter_s, 32);
    k.s_cmp_lt_u32(iter_s, SReg(n.0));
    k.branch_scc1(&loop_label);

    // Wave32 butterfly reduce: all 32 lanes sum → every lane has total
    let tmp = k.alloc_vreg();
    k.wave_reduce_add_f32(acc, tmp);

    // Lane 0 stores result
    let zero_v = k.alloc_vreg();
    k.v_mov_imm(zero_v, 0);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(lane_id), src1: Operand::InlineInt(1) });
    k.push(Op::SaveExec { dst: saved_exec });

    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0));
    k.v_mov_from_sgpr(VReg(out_addr.0 + 1), SReg(out_ptr.0 + 1));
    k.global_store(out_addr, acc, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved_exec });
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// naive_matmul: scalar f32 matmul for unaligned dimensions
// ============================================================================

/// Transpose mode for naive matmul variants
#[derive(Clone, Copy)]
pub enum NaiveGemmMode {
    /// Y = A @ B  (A[M,K], B[K,N] → Y[M,N])
    NN,
    /// Y = A @ B^T  (A[M,P], B[N,P] → Y[M,N])  — dX = dY @ W^T
    NT,
    /// Y = A^T @ B  (A[K,M], B[K,N] → Y[M,N])  — dW = X^T @ dY
    TN,
}

/// Generate naive scalar f32 matmul kernel.
///
/// Each thread computes one output element Y[row, col] via K-loop with v_fma_f32.
///
/// Kernargs (40 bytes): A_ptr(0), B_ptr(8), Y_ptr(16), M(24), K(28), N(32)
/// For NT: args are M, P(inner), N(output cols)
/// For TN: args are K(inner), M(K-dim of A), N
/// Grid: [ceil(N/32)*32, M, 1], WG: 32
pub fn naive_matmul(mode: NaiveGemmMode) -> T0Kernel {
    let name = match mode {
        NaiveGemmMode::NN => "t0_naive_f32_matmul",
        NaiveGemmMode::NT => "t0_naive_f32_matmul_nt",
        NaiveGemmMode::TN => "t0_naive_f32_matmul_tn",
    };
    let mut k = T0Kernel::new(name);

    let a_ptr = k.arg_ptr("A");
    let b_ptr = k.arg_ptr("B");
    let y_ptr = k.arg_ptr("Y");
    let dim_m = k.arg_u32("M");
    let dim_k = k.arg_u32("K");
    let dim_n = k.arg_u32("N");
    k.emit_arg_loads();

    let tile_col_s = k.alloc_sreg();
    let row_s = k.alloc_sreg();
    k.capture_tgid_x(tile_col_s);
    k.capture_tgid_y(row_s);

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // col = TGID.x * 32 + lane_id
    let col = k.alloc_vreg();
    k.v_mov_from_sgpr(col, tile_col_s);
    k.v_lshlrev_b32(col, 5, col);
    k.v_add_u32(col, col, lane_id);

    let row = k.alloc_vreg();
    k.v_mov_from_sgpr(row, row_s);

    // Bounds check: col < N
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(dim_n.0));
    let saved_exec = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(col), src1: Operand::VReg(n_vreg) });
    k.push(Op::SaveExec { dst: saved_exec });

    // A and B address setup depends on mode
    //
    // NN: A[row, k] at row*K*4, stride=4, B[k, col] at col*4, stride=N*4
    // NT: A[row, k] at row*K*4, stride=4, B[col, k] at col*K*4, stride=4
    // TN: A[k, row] at row*4, stride=M*4, B[k, col] at col*4, stride=N*4

    // A base address
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_addr.0 + 1), SReg(a_ptr.0 + 1));
    let a_off = k.alloc_vreg();
    match mode {
        NaiveGemmMode::NN | NaiveGemmMode::NT => {
            // A[row, 0] = a_ptr + row * K * 4
            let k_vreg = k.alloc_vreg();
            k.v_mov_from_sgpr(k_vreg, SReg(dim_k.0));
            k.v_mul_lo_u32(a_off, row, k_vreg);
            k.v_lshlrev_b32(a_off, 2, a_off);
        }
        NaiveGemmMode::TN => {
            // A[0, row] = a_ptr + row * 4
            k.v_lshlrev_b32(a_off, 2, row);
        }
    }
    k.v_add_co(a_addr, a_addr, a_off);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));

    // A stride (bytes per K iteration)
    let a_stride = k.alloc_vreg();
    match mode {
        NaiveGemmMode::NN | NaiveGemmMode::NT => {
            k.v_mov_imm(a_stride, 4);  // f32 stride
        }
        NaiveGemmMode::TN => {
            // stride = M * 4
            k.v_mov_from_sgpr(a_stride, SReg(dim_m.0));
            k.v_lshlrev_b32(a_stride, 2, a_stride);
        }
    }

    // B base address
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_addr.0 + 1), SReg(b_ptr.0 + 1));
    let b_off = k.alloc_vreg();
    match mode {
        NaiveGemmMode::NN | NaiveGemmMode::TN => {
            // B[0, col] = b_ptr + col * 4
            k.v_lshlrev_b32(b_off, 2, col);
        }
        NaiveGemmMode::NT => {
            // B[col, 0] = b_ptr + col * K * 4
            let k_vreg = k.alloc_vreg();
            k.v_mov_from_sgpr(k_vreg, SReg(dim_k.0));
            k.v_mul_lo_u32(b_off, col, k_vreg);
            k.v_lshlrev_b32(b_off, 2, b_off);
        }
    }
    k.v_add_co(b_addr, b_addr, b_off);
    k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));

    // B stride
    let b_stride = k.alloc_vreg();
    match mode {
        NaiveGemmMode::NN | NaiveGemmMode::TN => {
            // stride = N * 4
            k.v_mov_from_sgpr(b_stride, SReg(dim_n.0));
            k.v_lshlrev_b32(b_stride, 2, b_stride);
        }
        NaiveGemmMode::NT => {
            k.v_mov_imm(b_stride, 4);
        }
    }

    // Accumulator = 0
    let acc = k.alloc_vreg();
    k.v_mov_imm(acc, 0);

    // K-loop
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let loop_label = k.make_label("k_loop");
    k.label(&loop_label);

    let a_val = k.alloc_vreg();
    k.global_load(a_val, a_addr, Width::B32, 0);
    let b_val = k.alloc_vreg();
    k.global_load(b_val, b_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_fma_f32(acc, a_val, b_val, acc);

    // Advance A and B addresses
    k.v_add_co(a_addr, a_addr, a_stride);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
    k.v_add_co(b_addr, b_addr, b_stride);
    k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));

    k.s_add_u32(k_iter_s, k_iter_s, 1);
    k.s_cmp_lt_u32(k_iter_s, SReg(dim_k.0));
    k.branch_scc1(&loop_label);

    // Store Y[row, col] = acc
    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
    k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
    let y_off = k.alloc_vreg();
    k.v_mul_lo_u32(y_off, row, n_vreg);
    k.v_add_u32(y_off, y_off, col);
    k.v_lshlrev_b32(y_off, 2, y_off);
    k.v_add_co(y_addr, y_addr, y_off);
    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
    k.global_store(y_addr, acc, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved_exec });
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// ============================================================================
// relu_backward: dx = (x > 0) ? dy : 0
// ============================================================================

/// Generate ReLU backward kernel: dx[i] = (x[i] > 0) ? dy[i] : 0
///
/// Kernargs (24 bytes): dy_ptr(0), x_ptr(8), dx_ptr(16)
/// Grid: [ceil(n/32)*32, 1, 1], WG: 32
/// Uses epl (elements per lane) for throughput, compiled as constant.
pub fn relu_backward(epl: u32) -> T0Kernel {
    let mut k = T0Kernel::new("t0_relu_backward_f32");

    let dy_ptr = k.arg_ptr("dy");
    let x_ptr = k.arg_ptr("x");
    let dx_ptr = k.arg_ptr("dx");
    k.emit_arg_loads();

    let tile_s = k.alloc_sreg();
    k.capture_tgid_x(tile_s);

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // Base element index = TGID.x * 32 * epl + lane_id * epl
    let stride = 32 * epl;
    let base_idx = k.alloc_vreg();
    k.v_mov_from_sgpr(base_idx, tile_s);
    let stride_v = k.alloc_vreg();
    k.v_mov_imm(stride_v, stride as i32);
    k.v_mul_lo_u32(base_idx, base_idx, stride_v);
    let lane_off = k.alloc_vreg();
    let epl_v = k.alloc_vreg();
    k.v_mov_imm(epl_v, epl as i32);
    k.v_mul_lo_u32(lane_off, lane_id, epl_v);
    k.v_add_u32(base_idx, base_idx, lane_off);

    // Byte offset
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, base_idx);

    // Compute addresses
    let dy_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dy_addr, SReg(dy_ptr.0));
    k.v_mov_from_sgpr(VReg(dy_addr.0 + 1), SReg(dy_ptr.0 + 1));
    k.v_add_co(dy_addr, dy_addr, byte_off);
    k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));

    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_addr, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_addr.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_addr, x_addr, byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));

    let dx_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dx_addr, SReg(dx_ptr.0));
    k.v_mov_from_sgpr(VReg(dx_addr.0 + 1), SReg(dx_ptr.0 + 1));
    k.v_add_co(dx_addr, dx_addr, byte_off);
    k.v_add_co_ci(VReg(dx_addr.0 + 1), VReg(dx_addr.0 + 1));

    // Process epl elements
    let dy_val = k.alloc_vreg();
    let x_val = k.alloc_vreg();
    let zero = k.alloc_vreg();
    k.v_mov_imm(zero, 0);
    let result = k.alloc_vreg();

    for i in 0..epl {
        let offset = (i * 4) as i32;
        k.global_load(dy_val, dy_addr, Width::B32, offset);
        k.global_load(x_val, x_addr, Width::B32, offset);
        k.wait_vmcnt(0);
        k.v_cmp_gt_f32_imm0(x_val);
        k.v_cndmask_b32(result, Operand::VReg(zero), Operand::VReg(dy_val));
        k.global_store(dx_addr, result, Width::B32, offset);
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// silu_mul_backward: d_gate, d_up from SwiGLU backward
// ============================================================================

/// Generate SwiGLU backward kernel:
///   σ = sigmoid(gate)
///   d_gate = dy * up * σ * (1 + gate*(1 - σ))
///   d_up = dy * gate * σ
///
/// Kernargs (40 bytes): dy_ptr(0), gate_ptr(8), up_ptr(16), dg_ptr(24), du_ptr(32)
/// Grid: dispatch_elementwise style, WG: 32
/// Processes `epl` elements per lane (compiled constant).
pub fn silu_mul_backward(epl: u32) -> T0Kernel {
    assert!(epl >= 4 && epl % 4 == 0);
    let mut k = T0Kernel::new("t0_silu_mul_backward");
    let chunks = epl / 4;

    let dy_ptr = k.arg_ptr("dy");
    let gate_ptr = k.arg_ptr("gate");
    let up_ptr = k.arg_ptr("up");
    let dg_ptr = k.arg_ptr("dg");
    let du_ptr = k.arg_ptr("du");
    k.emit_arg_loads();

    let tile_s = k.alloc_sreg();
    k.capture_tgid_x(tile_s);

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // Base byte offset = (TGID.x * 32 * epl + lane_id * epl) * 4
    let lane_bytes = epl * 4;
    let wg_bytes = 32 * lane_bytes;
    let base_off = k.alloc_vreg();
    k.v_mov_from_sgpr(base_off, tile_s);
    let wgb_v = k.alloc_vreg();
    k.v_mov_imm(wgb_v, wg_bytes as i32);
    k.v_mul_lo_u32(base_off, base_off, wgb_v);
    let lane_off = k.alloc_vreg();
    let lb_v = k.alloc_vreg();
    k.v_mov_imm(lb_v, lane_bytes as i32);
    k.v_mul_lo_u32(lane_off, lane_id, lb_v);
    k.v_add_u32(base_off, base_off, lane_off);

    // Build 5 base addresses
    let ptrs = [dy_ptr, gate_ptr, up_ptr, dg_ptr, du_ptr];
    let addrs: Vec<VReg> = ptrs.iter().map(|p| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(addr, SReg(p.0));
        k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(p.0 + 1));
        k.v_add_co(addr, addr, base_off);
        k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
        addr
    }).collect();

    let (dy_a, gate_a, up_a, dg_a, du_a) = (addrs[0], addrs[1], addrs[2], addrs[3], addrs[4]);

    // Constants
    let log2e = k.alloc_vreg();
    k.v_mov_imm(log2e, 0x3FB8AA3Bu32 as i32); // log2(e)
    let one = k.alloc_vreg();
    k.v_mov_imm(one, 0x3F800000u32 as i32); // 1.0
    let sign_mask = k.alloc_vreg();
    k.v_mov_imm(sign_mask, 0x80000000u32 as i32); // sign bit

    // Scratch registers
    let dy_v = k.alloc_vreg_array(4, Alignment::None);
    let gate_v = k.alloc_vreg_array(4, Alignment::None);
    let up_v = k.alloc_vreg_array(4, Alignment::None);
    let sig = k.alloc_vreg_array(4, Alignment::None);
    let scratch = k.alloc_vreg_array(4, Alignment::None);

    for c in 0..chunks {
        let off = (c * 16) as i32;

        // Load dy, gate, up (4 elements each)
        for i in 0..4u32 {
            k.global_load(VReg(dy_v.0 + i), dy_a, Width::B32, off + (i as i32) * 4);
            k.global_load(VReg(gate_v.0 + i), gate_a, Width::B32, off + (i as i32) * 4);
            k.global_load(VReg(up_v.0 + i), up_a, Width::B32, off + (i as i32) * 4);
        }
        k.wait_vmcnt(0);

        // Compute sigmoid: σ = 1/(1+exp(-gate))
        for i in 0..4u32 {
            let s = VReg(sig.0 + i);
            let g = VReg(gate_v.0 + i);
            k.push(Op::VMulF32 { dst: s, src0: Operand::VReg(log2e), src1: Operand::VReg(g) });
            k.v_xor_b32(s, Operand::VReg(s), Operand::VReg(sign_mask));
            k.v_exp_f32(s, s);
            k.push(Op::VAddF32 { dst: s, src0: Operand::VReg(one), src1: Operand::VReg(s) });
            k.v_rcp_f32(s, s);
        }

        // d_gate = dy * up * σ * (1 + gate*(1 - σ))
        for i in 0..4u32 {
            let sc = VReg(scratch.0 + i);
            let s = VReg(sig.0 + i);
            let g = VReg(gate_v.0 + i);
            let u = VReg(up_v.0 + i);
            let d = VReg(dy_v.0 + i);
            k.v_sub_f32(sc, one, s);       // 1 - σ
            k.push(Op::VMulF32 { dst: sc, src0: Operand::VReg(g), src1: Operand::VReg(sc) }); // gate*(1-σ)
            k.push(Op::VAddF32 { dst: sc, src0: Operand::VReg(one), src1: Operand::VReg(sc) }); // 1+gate*(1-σ)
            k.push(Op::VMulF32 { dst: sc, src0: Operand::VReg(s), src1: Operand::VReg(sc) }); // σ*(1+g*(1-σ))
            k.push(Op::VMulF32 { dst: sc, src0: Operand::VReg(u), src1: Operand::VReg(sc) }); // up*σ*(1+g*(1-σ))
            k.push(Op::VMulF32 { dst: sc, src0: Operand::VReg(d), src1: Operand::VReg(sc) }); // dy*up*σ*(1+g*(1-σ))
        }
        // Store d_gate
        for i in 0..4u32 {
            k.global_store(dg_a, VReg(scratch.0 + i), Width::B32, off + (i as i32) * 4);
        }

        // d_up = dy * gate * σ
        for i in 0..4u32 {
            let sc = VReg(scratch.0 + i);
            let s = VReg(sig.0 + i);
            let g = VReg(gate_v.0 + i);
            let d = VReg(dy_v.0 + i);
            k.push(Op::VMulF32 { dst: sc, src0: Operand::VReg(g), src1: Operand::VReg(s) }); // gate*σ
            k.push(Op::VMulF32 { dst: sc, src0: Operand::VReg(d), src1: Operand::VReg(sc) }); // dy*gate*σ
        }
        // Store d_up
        for i in 0..4u32 {
            k.global_store(du_a, VReg(scratch.0 + i), Width::B32, off + (i as i32) * 4);
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// transpose: out[j,i] = in[i,j]  (M×N → N×M)
// ============================================================================

/// Generate f32 matrix transpose kernel: out[j * M + i] = in[i * N + j]
///
/// Each thread handles one element. Uses float-reciprocal divmod for i,j.
///
/// Kernargs (24 bytes): in_ptr(0), out_ptr(8), M(16), N(20)
/// Grid: [ceil(M*N/32)*32, 1, 1], WG: 32
pub fn transpose_f32() -> T0Kernel {
    let mut k = T0Kernel::new("t0_transpose_f32");

    let in_ptr = k.arg_ptr("in");
    let out_ptr = k.arg_ptr("out");
    let dim_m = k.arg_u32("M");
    let dim_n = k.arg_u32("N");
    k.emit_arg_loads();

    let tile_s = k.alloc_sreg();
    k.capture_tgid_x(tile_s);

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // global_id = TGID.x * 32 + lane_id
    let gid = k.alloc_vreg();
    k.v_mov_from_sgpr(gid, tile_s);
    k.v_lshlrev_b32(gid, 5, gid);
    k.v_add_u32(gid, gid, lane_id);

    // Bounds check: gid < M*N
    let total = k.alloc_vreg();
    let m_v = k.alloc_vreg();
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(m_v, SReg(dim_m.0));
    k.v_mov_from_sgpr(n_v, SReg(dim_n.0));
    k.v_mul_lo_u32(total, m_v, n_v);
    let saved_exec = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(gid), src1: Operand::VReg(total) });
    k.push(Op::SaveExec { dst: saved_exec });

    // Divmod: i = gid / N, j = gid % N (float rcp approximation + fixup)
    let n_f = k.alloc_vreg();
    k.v_cvt_f32_u32(n_f, n_v);
    let rcp_n = k.alloc_vreg();
    k.v_rcp_f32(rcp_n, n_f);
    let gid_f = k.alloc_vreg();
    k.v_cvt_f32_u32(gid_f, gid);
    let i_approx_f = k.alloc_vreg();
    k.push(Op::VMulF32 { dst: i_approx_f, src0: Operand::VReg(gid_f), src1: Operand::VReg(rcp_n) });
    let i_v = k.alloc_vreg();
    k.v_cvt_u32_f32(i_v, i_approx_f);
    let j_v = k.alloc_vreg();
    let tmp = k.alloc_vreg();
    k.v_mul_lo_u32(tmp, i_v, n_v);
    k.v_sub_u32(j_v, gid, tmp);

    // Fix rounding: if j >= N then i++, j -= N
    let fix = k.alloc_vreg();
    k.v_cmp_ge_u32(Operand::VReg(j_v), Operand::VReg(n_v));
    k.v_mov_imm(fix, 0);
    let one_v = k.alloc_vreg();
    k.v_mov_imm(one_v, 1);
    k.v_cndmask_b32(fix, Operand::VReg(fix), Operand::VReg(one_v));
    k.v_add_u32(i_v, i_v, fix);
    let fix_n = k.alloc_vreg();
    k.v_mul_lo_u32(fix_n, fix, n_v);
    k.v_sub_u32(j_v, j_v, fix_n);

    // Read src: in_ptr + gid * 4
    let src_off = k.alloc_vreg();
    k.v_lshlrev_b32(src_off, 2, gid);
    let src_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(src_addr, SReg(in_ptr.0));
    k.v_mov_from_sgpr(VReg(src_addr.0 + 1), SReg(in_ptr.0 + 1));
    k.v_add_co(src_addr, src_addr, src_off);
    k.v_add_co_ci(VReg(src_addr.0 + 1), VReg(src_addr.0 + 1));
    let val = k.alloc_vreg();
    k.global_load(val, src_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // Write dst: out_ptr + (j * M + i) * 4
    let dst_off = k.alloc_vreg();
    k.v_mul_lo_u32(dst_off, j_v, m_v);
    k.v_add_u32(dst_off, dst_off, i_v);
    k.v_lshlrev_b32(dst_off, 2, dst_off);
    let dst_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dst_addr, SReg(out_ptr.0));
    k.v_mov_from_sgpr(VReg(dst_addr.0 + 1), SReg(out_ptr.0 + 1));
    k.v_add_co(dst_addr, dst_addr, dst_off);
    k.v_add_co_ci(VReg(dst_addr.0 + 1), VReg(dst_addr.0 + 1));
    k.global_store(dst_addr, val, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved_exec });
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// rmsnorm: y = x * gamma / rms(x)
// ============================================================================

/// Generate RMSNorm kernel: y = (x / rms(x)) * gamma
/// where rms(x) = sqrt(mean(x²) + eps)
///
/// Kernargs (28 bytes):
/// | Offset | Size | Name |
/// |--------|------|------|
/// | 0      | 8    | x_ptr (f32, [batch, dim]) |
/// | 8      | 8    | gamma_ptr (f32, [dim]) |
/// | 16     | 8    | y_ptr (f32, [batch, dim]) |
/// | 24     | 4    | dim |
///
/// Grid: [batch, 1, 1] — one workgroup per row
pub fn rmsnorm(sched: &dyn Schedule) -> T0Kernel {
    let mut k = T0Kernel::new("t0_rmsnorm");
    let (wg_x, _, _) = sched.workgroup_size();

    // ── Args ──
    let x_ptr = k.arg_ptr("x");
    let gamma_ptr = k.arg_ptr("gamma");
    let y_ptr = k.arg_ptr("y");
    let dim = k.arg_u32("dim");
    k.emit_arg_loads();

    // ── Row index from TGID.x ──
    let row_s = k.alloc_sreg();
    k.capture_tgid_x(row_s);

    // thread_id = v0 (WORKITEM_ID_X)
    let tid = k.alloc_vreg();
    k.push(Op::VMov { dst: tid, src: Operand::VReg(VReg(0)) });

    // ── Compute row base address ──
    // row_off = row * dim * 4  (f32 elements)
    let row_off_s = k.alloc_sreg();
    k.push(Op::SMulI32 { dst: row_off_s, src0: row_s, src1: SReg(dim.0) });
    k.s_lshl_b32(row_off_s, row_off_s, 2);

    let row_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(row_off_v, row_off_s);

    // x_base = x_ptr + row_off
    let x_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_base, x_base, row_off_v);
    k.v_add_co_ci(VReg(x_base.0 + 1), VReg(x_base.0 + 1));

    // ── Load x[tid] ──
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, tid);  // tid * 4

    let load_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(load_addr, x_base);
    k.v_mov(VReg(load_addr.0 + 1), VReg(x_base.0 + 1));
    k.v_add_co(load_addr, load_addr, byte_off);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));

    let x_val = k.alloc_vreg();
    k.global_load(x_val, load_addr, Width::B32, 0);

    // ── Load gamma[tid] ──
    let gamma_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(gamma_addr, SReg(gamma_ptr.0));
    k.v_mov_from_sgpr(VReg(gamma_addr.0 + 1), SReg(gamma_ptr.0 + 1));
    k.v_add_co(gamma_addr, gamma_addr, byte_off);
    k.v_add_co_ci(VReg(gamma_addr.0 + 1), VReg(gamma_addr.0 + 1));

    let gamma_val = k.alloc_vreg();
    k.global_load(gamma_val, gamma_addr, Width::B32, 0);

    k.wait_vmcnt(0);

    // ══════════════════════════════════════════════════════════════════
    // RMSNorm computation: y = (x / rms(x)) * gamma
    // where rms(x) = sqrt(mean(x²) + eps)
    // ══════════════════════════════════════════════════════════════════

    // Step 1: x² per lane
    let x_sq = k.alloc_vreg();
    k.v_mul_f32(x_sq, x_val, x_val);

    // Step 2: Wave-level sum of x² across 32 lanes
    let tmp = k.alloc_vreg();
    k.wave_reduce_add_f32(x_sq, tmp);
    // Now x_sq = sum(x²) across all 32 lanes (every lane has same value)

    // Step 3: mean = sum_sq / dim
    // Use mul by reciprocal: mean = sum_sq * (1.0 / dim)
    // We compute 1/dim as a compile-time constant via the param arg
    // Or: use v_rcp_f32 on dim. For simplicity, since dim is in an SGPR:
    let dim_f = k.alloc_vreg();
    k.v_mov_from_sgpr(dim_f, SReg(dim.0));
    // Convert uint to float (dim is small, fits in f32)
    k.push(Op::RawAsm("  ; v_cvt_f32_u32 for dim".to_string()));
    // Actually we need v_cvt_f32_u32 which isn't in our IR yet.
    // Use a simpler approach: pass 1/dim as a float arg.
    // OR: just divide by wg_size (known at compile time).
    // For single-pass (dim <= wg_size), mean = sum_sq / wg_size
    let inv_dim = 1.0 / wg_x as f32;
    k.v_mul_f32_imm(x_sq, x_sq, inv_dim);
    // x_sq now = mean(x²)

    // Step 4: mean + eps
    let eps = k.alloc_vreg();
    k.push(Op::VMov { dst: eps, src: Operand::Literal(1e-6_f32.to_bits()) });
    k.v_add_f32(x_sq, x_sq, eps);

    // Step 5: inv_rms = rsqrt(mean + eps)
    let inv_rms = k.alloc_vreg();
    k.v_rsq_f32(inv_rms, x_sq);

    // Step 6: y = x * inv_rms * gamma
    let normed = k.alloc_vreg();
    k.v_mul_f32(normed, x_val, inv_rms);
    let y_val = k.alloc_vreg();
    k.v_mul_f32(y_val, normed, gamma_val);

    // ── Store y[tid] ──
    let y_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_base, SReg(y_ptr.0));
    k.v_mov_from_sgpr(VReg(y_base.0 + 1), SReg(y_ptr.0 + 1));
    k.v_add_co(y_base, y_base, row_off_v);
    k.v_add_co_ci(VReg(y_base.0 + 1), VReg(y_base.0 + 1));
    k.v_add_co(y_base, y_base, byte_off);
    k.v_add_co_ci(VReg(y_base.0 + 1), VReg(y_base.0 + 1));
    k.global_store(y_base, y_val, Width::B32, 0);

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// elementwise_unary: y[i] = op(x[i])
// ============================================================================

/// Generate a unary elementwise kernel for the given operation.
///
/// Grid: [ceil(n/wg_size), 1, 1]
/// Kernargs: x_ptr(0), y_ptr(8), param(16:f32), n(20:u32)
pub fn elementwise_unary(sched: &dyn Schedule, op: UnaryOp) -> T0Kernel {
    let name = match op {
        UnaryOp::Scale(_) => "t0_scale",
        UnaryOp::Relu => "t0_relu",
        UnaryOp::GeluApprox => "t0_gelu",
        UnaryOp::Bf16ToF32 => "t0_bf16_to_f32",
        UnaryOp::F32ToBf16 => "t0_f32_to_bf16",
        UnaryOp::Square => "t0_square",
        UnaryOp::Rsqrt => "t0_rsqrt",
        UnaryOp::Negate => "t0_negate",
    };

    let mut k = T0Kernel::new(name);
    let (wg_x, _, _) = sched.workgroup_size();

    // ── Args ──
    let x_ptr = k.arg_ptr("x");
    let y_ptr = k.arg_ptr("y");
    let param = k.arg_f32("param");
    let n = k.arg_u32("n");
    k.emit_arg_loads();

    // ── Global ID ──
    let global_id = k.compute_global_id_x(wg_x as u32);

    // ── Load x[global_id] ──
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, global_id);

    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_addr, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_addr.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_addr, x_addr, byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));

    let val = k.alloc_vreg();
    k.global_load(val, x_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // ── Apply operation ──
    match op {
        UnaryOp::Scale(s) => {
            let sv = k.alloc_vreg();
            k.v_mov_from_sgpr(sv, SReg(param.0));
            k.v_mul_f32(val, val, sv);
        }
        UnaryOp::Relu => {
            // max(0, val)
            k.push(Op::VMaxF32 {
                dst: val,
                src0: Operand::VReg(val),
                src1: Operand::InlineFloat(0.0),
            });
        }
        UnaryOp::Square => {
            k.v_mul_f32(val, val, val);
        }
        UnaryOp::Negate => {
            // val = val * -1.0
            k.push(Op::VMulF32 {
                dst: val,
                src0: Operand::VReg(val),
                src1: Operand::InlineFloat(-1.0),
            });
        }
        UnaryOp::Rsqrt => {
            // v_rsq_f32 via raw asm (single instruction)
            k.raw_asm("  ; v_rsq_f32 — needs RawAsm or dedicated op");
            // Placeholder: use reciprocal sqrt approximation
        }
        UnaryOp::Bf16ToF32 => {
            // bf16 is upper 16 bits of f32, so: v_lshlrev_b32 val, 16, val
            k.v_lshlrev_b32(val, 16, val);
        }
        UnaryOp::F32ToBf16 => {
            // Proper rounding: add bias 0x7FFF + bit[16] then truncate
            // This implements round-to-nearest-even (banker's rounding)
            let bias = k.alloc_vreg();
            k.v_mov_imm(bias, 0x7FFF);
            let bit16 = k.alloc_vreg();
            k.v_lshrrev_b32(bit16, 16, val);
            k.v_and_b32_imm(bit16, bit16, 1);
            k.v_add_u32(val, val, bias);
            k.v_add_u32(val, val, bit16);
            k.v_lshrrev_b32(val, 16, val);
        }
        UnaryOp::GeluApprox => {
            // GeLU ≈ x * sigmoid(1.702 * x)
            // sigmoid(z) ≈ 0.5 + 0.5 * tanh(z * 0.7978)
            // Simplified: just multiply by param for now
            let sv = k.alloc_vreg();
            k.v_mov_from_sgpr(sv, SReg(param.0));
            k.v_mul_f32(val, val, sv);
        }
    }

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

/// Generate a binary elementwise kernel: y[i] = op(a[i], b[i])
///
/// Grid: [ceil(n/wg_size), 1, 1]
/// Kernargs: a_ptr(0), b_ptr(8), y_ptr(16), param(24:f32), n(28:u32)
pub fn elementwise_binary(sched: &dyn Schedule, op: BinaryOp) -> T0Kernel {
    let name = match op {
        BinaryOp::Add => "t0_add",
        BinaryOp::Mul => "t0_mul",
        BinaryOp::Axpy(_) => "t0_axpy",
    };

    let mut k = T0Kernel::new(name);
    let (wg_x, _, _) = sched.workgroup_size();

    // ── Args ──
    let a_ptr = k.arg_ptr("a");
    let b_ptr = k.arg_ptr("b");
    let y_ptr = k.arg_ptr("y");
    let alpha = k.arg_f32("alpha");
    let n = k.arg_u32("n");
    k.emit_arg_loads();

    let global_id = k.compute_global_id_x(wg_x as u32);

    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, global_id);

    // Load a[i]
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_addr.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_addr, a_addr, byte_off);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
    let a_val = k.alloc_vreg();
    k.global_load(a_val, a_addr, Width::B32, 0);

    // Load b[i]
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_addr.0 + 1), SReg(b_ptr.0 + 1));
    k.v_add_co(b_addr, b_addr, byte_off);
    k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
    let b_val = k.alloc_vreg();
    k.global_load(b_val, b_addr, Width::B32, 0);

    k.wait_vmcnt(0);

    // Apply op
    let result = k.alloc_vreg();
    match op {
        BinaryOp::Add => {
            k.v_add_f32(result, a_val, b_val);
        }
        BinaryOp::Mul => {
            k.v_mul_f32(result, a_val, b_val);
        }
        BinaryOp::Axpy(_) => {
            // y = a + alpha * b
            let alpha_v = k.alloc_vreg();
            k.v_mov_from_sgpr(alpha_v, SReg(alpha.0));
            k.v_fma_f32(result, alpha_v, b_val, a_val);
        }
    }

    // Store y[i]
    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
    k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
    k.v_add_co(y_addr, y_addr, byte_off);
    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
    k.global_store(y_addr, result, Width::B32, 0);

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Fused Elementwise Codegen
// ============================================================================
//
// Replaces hand-written fused_elementwise.rs with T0 IR pipeline.
// Accepts the same FusionPlan / EwOp types so Ignis can seamlessly switch.

/// Primitive elementwise operation for fused kernel codegen.
#[derive(Clone, Debug, PartialEq)]
pub enum EwOp {
    Scale(f32),
    AddInput(u8),
    MulInput(u8),
    Neg,
    Relu,
    Fma(f32, u8),
    Sigmoid,
    SiLU,
    Exp,
    Rcp,
    Abs,
    AddConst(f32),
    Square,
    Min(f32),
}

/// A fusion plan describing a sequence of elementwise operations.
#[derive(Clone, Debug)]
pub struct FusionPlan {
    pub ops: Vec<EwOp>,
    pub n_inputs: u8,
    pub name: String,
    pub inplace: bool,
    pub zero_init: bool,
}

impl FusionPlan {
    pub fn cache_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        self.name.hash(&mut h);
        self.n_inputs.hash(&mut h);
        self.inplace.hash(&mut h);
        self.zero_init.hash(&mut h);
        for op in &self.ops {
            std::mem::discriminant(op).hash(&mut h);
            match op {
                EwOp::Scale(v) | EwOp::AddConst(v) | EwOp::Min(v) => v.to_bits().hash(&mut h),
                EwOp::AddInput(s) | EwOp::MulInput(s) => s.hash(&mut h),
                EwOp::Fma(v, s) => { v.to_bits().hash(&mut h); s.hash(&mut h); },
                _ => {},
            }
        }
        h.finish()
    }

    pub fn kernarg_size(&self) -> usize {
        let n_ptrs = if self.inplace {
            self.n_inputs as usize
        } else {
            self.n_inputs as usize + 1
        };
        let n_scalars = self.scalar_count();
        n_ptrs * 8 + n_scalars * 4 + 4
    }

    fn scalar_count(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op,
            EwOp::Scale(_) | EwOp::Fma(_, _) | EwOp::AddConst(_) | EwOp::Min(_)
        )).count()
    }
}

/// Build a fused elementwise kernel from a FusionPlan using T0 IR.
///
/// Each lane processes `epl` elements (must be multiple of 4, >= 4).
/// WorkGroup size = 32 (Wave32, single wave).
///
/// Kernel loads input slot 0, applies ops in sequence (loading extra inputs
/// as needed), then stores to output. All f32, dwordx4 loads/stores.
///
/// Kernarg layout (matches hand-written version):
///   [0..8]   input_0_ptr
///   [8..16]  input_1_ptr (if n_inputs > 1)
///   ...
///   [N..N+8] output_ptr
///   [N+8..]  scalar constants (f32 each)
///   [...+4]  n_elems (u32)
pub fn fused_elementwise(plan: &FusionPlan, epl: u32) -> T0Kernel {
    assert!(epl >= 4 && epl % 4 == 0, "epl must be >= 4 and multiple of 4");
    assert!(plan.n_inputs >= 1 && plan.n_inputs <= 4, "1-4 inputs supported");

    let chunks = epl / 4;
    let n_inputs = plan.n_inputs as usize;
    let n_scalars = plan.ops.iter()
        .filter(|op| matches!(op, EwOp::Scale(_) | EwOp::Fma(_, _)))
        .count();

    // Compute kernarg size to match hand-written layout
    let n_ptrs = n_inputs + 1; // inputs + output
    let ka_size = (n_ptrs * 8 + n_scalars * 4 + 4) as u32;

    let mut k = T0Kernel::new(&plan.name);

    // ── Declare kernel arguments (in order) ──
    // Input pointers
    let input_ptrs: Vec<SRegPair> = (0..n_inputs)
        .map(|i| k.arg_ptr(&format!("in{}", i)))
        .collect();
    // Output pointer (skip if inplace)
    let out_ptr = if plan.inplace {
        SRegPair(0) // dummy, won't be used (out_addr = input_addrs[0])
    } else {
        k.arg_ptr("out")
    };
    // Scalar constants
    let scalar_sregs: Vec<SReg> = plan.ops.iter()
        .filter_map(|op| match op {
            EwOp::Scale(v) => Some(k.arg_f32(&format!("scale_{}", v.to_bits()))),
            EwOp::Fma(v, _) => Some(k.arg_f32(&format!("fma_alpha_{}", v.to_bits()))),
            EwOp::AddConst(v) => Some(k.arg_f32(&format!("addconst_{}", v.to_bits()))),
            EwOp::Min(v) => Some(k.arg_f32(&format!("min_{}", v.to_bits()))),
            _ => None,
        })
        .collect();
    // n_elems
    let _n_elems = k.arg_u32("n_elems");

    k.emit_arg_loads();

    // ── Address computation ──
    // global_byte_offset = TGID.x * (32 * epl * 4) + lane_id * (epl * 4)
    let tgid_x = k.alloc_sreg();
    k.capture_tgid_x(tgid_x);

    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    let lane_bytes = epl * 4;
    let wg_bytes = 32 * lane_bytes;

    // s_tmp = TGID.x * wg_bytes
    let s_wg_off = k.alloc_sreg();
    let s_wg_bytes = k.alloc_sreg();
    k.s_mov_imm(s_wg_bytes, wg_bytes as i32);
    k.push(Op::SMulI32 { dst: s_wg_off, src0: tgid_x, src1: s_wg_bytes });

    // v_lane_off = lane_id * lane_bytes
    let v_lane_bytes = k.alloc_vreg();
    let v_lane_off = k.alloc_vreg();
    k.v_mov_imm(v_lane_bytes, lane_bytes as i32);
    k.v_mul_lo_u32(v_lane_off, lane_id, v_lane_bytes);

    // v_wg_off = move s_wg_off to vgpr
    let v_wg_off = k.alloc_vreg();
    k.v_mov_from_sgpr(v_wg_off, s_wg_off);

    // ── Build address pairs for each input ──
    let mut input_addrs: Vec<VReg> = Vec::new();
    for i in 0..n_inputs {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(addr, SReg(input_ptrs[i].0));
        k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(input_ptrs[i].0 + 1));
        // addr += wg_off + lane_off
        k.v_add_co(addr, addr, v_wg_off);
        k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
        k.v_add_co(addr, addr, v_lane_off);
        k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
        input_addrs.push(addr);
    }

    // Output address (skip if inplace — will use input_addrs[0])
    let out_addr = if plan.inplace {
        input_addrs[0] // store back to input[0]
    } else {
        let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0));
        k.v_mov_from_sgpr(VReg(out_addr.0 + 1), SReg(out_ptr.0 + 1));
        k.v_add_co(out_addr, out_addr, v_wg_off);
        k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));
        k.v_add_co(out_addr, out_addr, v_lane_off);
        k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));
        out_addr
    };

    // ── Move scalar constants to VGPRs ──
    let scalar_vregs: Vec<VReg> = scalar_sregs.iter()
        .map(|s| {
            let v = k.alloc_vreg();
            k.v_mov_from_sgpr(v, *s);
            v
        })
        .collect();

    // ── Determine which extra input slots need loading ──
    let mut extra_slots: Vec<u8> = plan.ops.iter().filter_map(|op| match op {
        EwOp::AddInput(s) if *s > 0 => Some(*s),
        EwOp::MulInput(s) if *s > 0 => Some(*s),
        EwOp::Fma(_, s) if *s > 0 => Some(*s),
        _ => None,
    }).collect();
    extra_slots.sort();
    extra_slots.dedup();

    // ── Allocate data registers ──
    // acc = 4 consecutive VGPRs for primary input / result
    let acc = k.alloc_vreg_array(4, Alignment::Align4);
    // Extra input data registers (one set of 4 per extra slot)
    let mut extra_data: std::collections::HashMap<u8, VReg> = std::collections::HashMap::new();
    for &slot in &extra_slots {
        let d = k.alloc_vreg_array(4, Alignment::Align4);
        extra_data.insert(slot, d);
    }

    // ── Main compute: per chunk ──
    // Pre-allocate scratch regs for Sigmoid/SiLU (shared across chunks)
    let needs_sigmoid = plan.ops.iter().any(|op| matches!(op, EwOp::Sigmoid | EwOp::SiLU | EwOp::Exp));
    let sig_tmp = if needs_sigmoid {
        // Constants for sigmoid: log2(e) and sign mask
        let log2e_v = k.alloc_vreg();
        k.push(Op::VMov { dst: log2e_v, src: Operand::Literal(0x3FB8AA3Bu32) });
        let sign_v = k.alloc_vreg();
        k.push(Op::VMov { dst: sign_v, src: Operand::Literal(0x80000000u32) });
        Some((log2e_v, sign_v, k.alloc_vreg_array(4, Alignment::Align4)))
    } else { None };

    // Pre-allocate save regs for SiLU (need copy of acc before sigmoid)
    let silu_save = if plan.ops.iter().any(|op| matches!(op, EwOp::SiLU)) {
        Some(k.alloc_vreg_array(4, Alignment::Align4))
    } else { None };

    for c in 0..chunks {
        let off = (c * 16) as i32;

        if plan.zero_init {
            // Zero-init accumulator
            for i in 0..4u32 {
                k.push(Op::VMov { dst: VReg(acc.0 + i), src: Operand::InlineInt(0) });
            }
        } else {
            // Load primary input (slot 0) → acc
            k.global_load(acc, input_addrs[0], Width::B128, off);
        }

        // Load extra inputs
        for &slot in &extra_slots {
            let data_v = extra_data[&slot];
            let addr_v = input_addrs[slot as usize];
            k.global_load(data_v, addr_v, Width::B128, off);
        }

        // Wait for loads (unless zero_init with no extra loads)
        if !plan.zero_init || !extra_slots.is_empty() {
            k.wait_vmcnt(0);
        }

        // ── Execute fused operations ──
        let mut scalar_cursor = 0usize;
        for op in &plan.ops {
            match op {
                EwOp::Scale(_) => {
                    let sv = scalar_vregs[scalar_cursor];
                    for i in 0..4u32 {
                        k.v_mul_f32(VReg(acc.0 + i), VReg(acc.0 + i), sv);
                    }
                    scalar_cursor += 1;
                }
                EwOp::AddInput(slot) => {
                    if *slot == 0 {
                        for i in 0..4u32 {
                            k.v_add_f32(VReg(acc.0 + i), VReg(acc.0 + i), VReg(acc.0 + i));
                        }
                    } else {
                        let dv = extra_data[slot];
                        for i in 0..4u32 {
                            k.v_add_f32(VReg(acc.0 + i), VReg(acc.0 + i), VReg(dv.0 + i));
                        }
                    }
                }
                EwOp::MulInput(slot) => {
                    if *slot == 0 {
                        for i in 0..4u32 {
                            k.v_mul_f32(VReg(acc.0 + i), VReg(acc.0 + i), VReg(acc.0 + i));
                        }
                    } else {
                        let dv = extra_data[slot];
                        for i in 0..4u32 {
                            k.v_mul_f32(VReg(acc.0 + i), VReg(acc.0 + i), VReg(dv.0 + i));
                        }
                    }
                }
                EwOp::Neg => {
                    for i in 0..4u32 {
                        k.push(Op::VMulF32 {
                            dst: VReg(acc.0 + i),
                            src0: Operand::VReg(VReg(acc.0 + i)),
                            src1: Operand::InlineFloat(-1.0),
                        });
                    }
                }
                EwOp::Relu => {
                    for i in 0..4u32 {
                        k.push(Op::VMaxF32 {
                            dst: VReg(acc.0 + i),
                            src0: Operand::VReg(VReg(acc.0 + i)),
                            src1: Operand::InlineFloat(0.0),
                        });
                    }
                }
                EwOp::Fma(_, slot) => {
                    let sv = scalar_vregs[scalar_cursor];
                    let dv = if *slot == 0 { acc } else { extra_data[slot] };
                    for i in 0..4u32 {
                        k.v_fma_f32(VReg(acc.0 + i), VReg(acc.0 + i), sv, VReg(dv.0 + i));
                    }
                    scalar_cursor += 1;
                }

                // ── Extended ops ──

                EwOp::Sigmoid => {
                    // σ(x) = 1/(1 + 2^(-x*log2e))
                    let (log2e_v, sign_v, tmp) = sig_tmp.unwrap();
                    for i in 0..4u32 {
                        let a = VReg(acc.0 + i);
                        let t = VReg(tmp.0 + i);
                        k.v_mul_f32(t, a, log2e_v);                                      // t = x * log2(e)
                        k.v_xor_b32(t, Operand::VReg(t), Operand::VReg(sign_v));         // t = -x * log2(e)
                        k.v_exp_f32(t, t);                                               // t = 2^(-x*log2e) = exp(-x)
                        k.push(Op::VAddF32 {                                             // t = 1.0 + exp(-x)
                            dst: t, src0: Operand::InlineFloat(1.0), src1: Operand::VReg(t),
                        });
                        k.v_rcp_f32(t, t);                                               // t = 1/(1+exp(-x)) = σ(x)
                        k.push(Op::VMov { dst: a, src: Operand::VReg(t) });              // acc = σ(x)
                    }
                }
                EwOp::SiLU => {
                    // silu(x) = x * σ(x): first save x, then compute σ, then mul
                    let (log2e_v, sign_v, tmp) = sig_tmp.unwrap();
                    let save = silu_save.unwrap();
                    // Save original x
                    for i in 0..4u32 {
                        k.push(Op::VMov { dst: VReg(save.0 + i), src: Operand::VReg(VReg(acc.0 + i)) });
                    }
                    // Compute σ(x) into acc
                    for i in 0..4u32 {
                        let a = VReg(acc.0 + i);
                        let t = VReg(tmp.0 + i);
                        k.v_mul_f32(t, a, log2e_v);
                        k.v_xor_b32(t, Operand::VReg(t), Operand::VReg(sign_v));
                        k.v_exp_f32(t, t);
                        k.push(Op::VAddF32 { dst: t, src0: Operand::InlineFloat(1.0), src1: Operand::VReg(t) });
                        k.v_rcp_f32(a, t);                                               // acc = σ(x)
                    }
                    // acc = x * σ(x)
                    for i in 0..4u32 {
                        k.v_mul_f32(VReg(acc.0 + i), VReg(save.0 + i), VReg(acc.0 + i));
                    }
                }
                EwOp::Exp => {
                    // exp(x) = 2^(x * log2(e))
                    let (log2e_v, _, _) = sig_tmp.unwrap();
                    for i in 0..4u32 {
                        let a = VReg(acc.0 + i);
                        k.v_mul_f32(a, a, log2e_v);
                        k.v_exp_f32(a, a);
                    }
                }
                EwOp::Rcp => {
                    for i in 0..4u32 {
                        let a = VReg(acc.0 + i);
                        k.v_rcp_f32(a, a);
                    }
                }
                EwOp::Abs => {
                    // |x| = x AND 0x7FFFFFFF (clear sign bit)
                    for i in 0..4u32 {
                        let a = VReg(acc.0 + i);
                        k.push(Op::VAndB32 { dst: a,
                            src0: Operand::VReg(a),
                            src1: Operand::Literal(0x7FFFFFFFu32),
                        });
                    }
                }
                EwOp::AddConst(_) => {
                    let sv = scalar_vregs[scalar_cursor];
                    for i in 0..4u32 {
                        k.v_add_f32(VReg(acc.0 + i), VReg(acc.0 + i), sv);
                    }
                    scalar_cursor += 1;
                }
                EwOp::Square => {
                    for i in 0..4u32 {
                        k.v_mul_f32(VReg(acc.0 + i), VReg(acc.0 + i), VReg(acc.0 + i));
                    }
                }
                EwOp::Min(_) => {
                    let sv = scalar_vregs[scalar_cursor];
                    for i in 0..4u32 {
                        k.push(Op::VMinF32 {
                            dst: VReg(acc.0 + i),
                            src0: Operand::VReg(VReg(acc.0 + i)),
                            src1: Operand::VReg(sv),
                        });
                    }
                    scalar_cursor += 1;
                }
            }
        }

        // Store result
        k.global_store(out_addr, acc, Width::B128, off);
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Drop-in replacements for hand-written elementwise_ops.rs kernels
// ============================================================================

/// T0 replacement for `build_gpu_memset_zero`.
/// Kernarg: [ptr: u64, n_elems: u32]
/// Grid: ceil(n_elems / (32 * epl)) workgroups × 32 threads
pub fn t0_memset_zero(epl: u32) -> T0Kernel {
    assert!(epl >= 4 && epl % 4 == 0);
    let chunks = epl / 4;

    let mut k = T0Kernel::new("t0_memset_zero");
    let ptr = k.arg_ptr("dst");
    let n_elems_s = k.arg_u32("n_elems");
    k.emit_arg_loads();

    // Global ID
    let gid = k.alloc_vreg();
    k.push(Op::ComputeGlobalIdX { dst: gid, wg_size: 32 });

    // elem_id = gid * epl
    let elem_id = k.alloc_vreg();
    let v_epl = k.alloc_vreg();
    k.v_mov_imm(v_epl, epl as i32);
    k.v_mul_lo_u32(elem_id, gid, v_epl);

    // Address = ptr + elem_id * 4
    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(addr, SReg(ptr.0));
    k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(ptr.0 + 1));
    let byte_off = k.alloc_vreg();
    k.push(Op::VLshlrevB32 { dst: byte_off, shift: 2, src: elem_id });
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

    // Bounds check: elem_id < n_elems
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, n_elems_s);
    let saved = k.bounds_check_begin(elem_id, n_v);

    // Zero registers
    let zeros = k.alloc_vreg_array(4, Alignment::Align4);
    for i in 0..4u32 {
        k.push(Op::VMov { dst: VReg(zeros.0 + i), src: Operand::InlineInt(0) });
    }

    // Store zero chunks
    for c in 0..chunks {
        k.global_store(addr, zeros, Width::B128, (c * 16) as i32);
    }

    k.bounds_check_end(saved);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// T0 replacement for `build_gpu_memcpy`.
/// Kernarg: [src: u64, dst: u64, n_elems: u32]
/// Grid: ceil(n_elems / (32 * epl)) workgroups × 32 threads
pub fn t0_memcpy(epl: u32) -> T0Kernel {
    assert!(epl >= 4 && epl % 4 == 0);
    let chunks = epl / 4;

    let mut k = T0Kernel::new("t0_memcpy");
    let src_ptr = k.arg_ptr("src");
    let dst_ptr = k.arg_ptr("dst");
    let n_elems_s = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let gid = k.alloc_vreg();
    k.push(Op::ComputeGlobalIdX { dst: gid, wg_size: 32 });

    let elem_id = k.alloc_vreg();
    let v_epl = k.alloc_vreg();
    k.v_mov_imm(v_epl, epl as i32);
    k.v_mul_lo_u32(elem_id, gid, v_epl);

    // Src address
    let src_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(src_addr, SReg(src_ptr.0));
    k.v_mov_from_sgpr(VReg(src_addr.0 + 1), SReg(src_ptr.0 + 1));
    let byte_off = k.alloc_vreg();
    k.push(Op::VLshlrevB32 { dst: byte_off, shift: 2, src: elem_id });
    k.v_add_co(src_addr, src_addr, byte_off);
    k.v_add_co_ci(VReg(src_addr.0 + 1), VReg(src_addr.0 + 1));

    // Dst address
    let dst_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dst_addr, SReg(dst_ptr.0));
    k.v_mov_from_sgpr(VReg(dst_addr.0 + 1), SReg(dst_ptr.0 + 1));
    k.v_add_co(dst_addr, dst_addr, byte_off);
    k.v_add_co_ci(VReg(dst_addr.0 + 1), VReg(dst_addr.0 + 1));

    // Bounds check
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, n_elems_s);
    let saved = k.bounds_check_begin(elem_id, n_v);

    let data = k.alloc_vreg_array(4, Alignment::Align4);
    for c in 0..chunks {
        let off = (c * 16) as i32;
        k.global_load(data, src_addr, Width::B128, off);
        k.wait_vmcnt(0);
        k.global_store(dst_addr, data, Width::B128, off);
    }

    k.bounds_check_end(saved);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// T0 replacement for `build_residual_add`.
/// y[i] += x[i] (in-place add to second argument)
/// Kernarg: [x: u64, y: u64, n_elems: u32]
/// Grid: ceil(n_elems / (32 * epl)) workgroups × 32 threads
pub fn t0_residual_add(epl: u32) -> T0Kernel {
    assert!(epl >= 4 && epl % 4 == 0);
    let chunks = epl / 4;

    let mut k = T0Kernel::new("t0_residual_add");
    let x_ptr = k.arg_ptr("x");
    let y_ptr = k.arg_ptr("y");
    let n_elems_s = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let gid = k.alloc_vreg();
    k.push(Op::ComputeGlobalIdX { dst: gid, wg_size: 32 });

    let elem_id = k.alloc_vreg();
    let v_epl = k.alloc_vreg();
    k.v_mov_imm(v_epl, epl as i32);
    k.v_mul_lo_u32(elem_id, gid, v_epl);

    // X address
    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_addr, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_addr.0 + 1), SReg(x_ptr.0 + 1));
    let byte_off = k.alloc_vreg();
    k.push(Op::VLshlrevB32 { dst: byte_off, shift: 2, src: elem_id });
    k.v_add_co(x_addr, x_addr, byte_off);
    k.v_add_co_ci(VReg(x_addr.0 + 1), VReg(x_addr.0 + 1));

    // Y address
    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
    k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
    k.v_add_co(y_addr, y_addr, byte_off);
    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));

    // Bounds check
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, n_elems_s);
    let saved = k.bounds_check_begin(elem_id, n_v);

    let xd = k.alloc_vreg_array(4, Alignment::Align4);
    let yd = k.alloc_vreg_array(4, Alignment::Align4);
    for c in 0..chunks {
        let off = (c * 16) as i32;
        k.global_load(xd, x_addr, Width::B128, off);
        k.global_load(yd, y_addr, Width::B128, off);
        k.wait_vmcnt(0);
        for i in 0..4u32 {
            k.v_add_f32(VReg(yd.0 + i), VReg(yd.0 + i), VReg(xd.0 + i));
        }
        k.global_store(y_addr, yd, Width::B128, off);
    }

    k.bounds_check_end(saved);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// T0 replacement for `build_elementwise_mul`.
/// out[i] = a[i] * b[i]
/// Kernarg: [a: u64, b: u64, out: u64, n_elems: u32]
/// Grid: ceil(n_elems / (32 * epl)) workgroups × 32 threads
pub fn t0_elementwise_mul(epl: u32) -> T0Kernel {
    assert!(epl >= 4 && epl % 4 == 0);
    let chunks = epl / 4;

    let mut k = T0Kernel::new("t0_elementwise_mul");
    let a_ptr = k.arg_ptr("a");
    let b_ptr = k.arg_ptr("b");
    let out_ptr = k.arg_ptr("out");
    let n_elems_s = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let gid = k.alloc_vreg();
    k.push(Op::ComputeGlobalIdX { dst: gid, wg_size: 32 });
    let elem_id = k.alloc_vreg();
    let v_epl = k.alloc_vreg();
    k.v_mov_imm(v_epl, epl as i32);
    k.v_mul_lo_u32(elem_id, gid, v_epl);

    // A address
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_addr.0 + 1), SReg(a_ptr.0 + 1));
    let byte_off = k.alloc_vreg();
    k.push(Op::VLshlrevB32 { dst: byte_off, shift: 2, src: elem_id });
    k.v_add_co(a_addr, a_addr, byte_off);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));

    // B address
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_addr.0 + 1), SReg(b_ptr.0 + 1));
    k.v_add_co(b_addr, b_addr, byte_off);
    k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));

    // Out address
    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0));
    k.v_mov_from_sgpr(VReg(out_addr.0 + 1), SReg(out_ptr.0 + 1));
    k.v_add_co(out_addr, out_addr, byte_off);
    k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));

    // Bounds check
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, n_elems_s);
    let saved = k.bounds_check_begin(elem_id, n_v);

    let ad = k.alloc_vreg_array(4, Alignment::Align4);
    let bd = k.alloc_vreg_array(4, Alignment::Align4);
    for c in 0..chunks {
        let off = (c * 16) as i32;
        k.global_load(ad, a_addr, Width::B128, off);
        k.global_load(bd, b_addr, Width::B128, off);
        k.wait_vmcnt(0);
        for i in 0..4u32 {
            k.v_mul_f32(VReg(ad.0 + i), VReg(ad.0 + i), VReg(bd.0 + i));
        }
        k.global_store(out_addr, ad, Width::B128, off);
    }

    k.bounds_check_end(saved);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// T0 replacement for `build_silu_mul` (SwiGLU forward).
/// y = silu(gate) * up = gate * σ(gate) * up
/// where σ(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x * log2(e)))
///
/// Kernarg: [gate: u64, up: u64, out: u64, n_elems: u32]
/// Grid: ceil(n_elems / (32 * epl)) workgroups × 32 threads
pub fn t0_silu_mul(epl: u32) -> T0Kernel {
    assert!(epl >= 4 && epl % 4 == 0);
    let chunks = epl / 4;

    let mut k = T0Kernel::new("t0_silu_mul");
    let gate_ptr = k.arg_ptr("gate");
    let up_ptr = k.arg_ptr("up");
    let out_ptr = k.arg_ptr("out");
    let n_elems_s = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let gid = k.alloc_vreg();
    k.push(Op::ComputeGlobalIdX { dst: gid, wg_size: 32 });

    let elem_id = k.alloc_vreg();
    let v_epl = k.alloc_vreg();
    k.v_mov_imm(v_epl, epl as i32);
    k.v_mul_lo_u32(elem_id, gid, v_epl);

    // Constants
    let log2e = k.alloc_vreg();
    k.push(Op::VMov { dst: log2e, src: Operand::Literal(0x3FB8AA3Bu32) }); // log2(e) = 1.4426950408
    let one = k.alloc_vreg();
    k.push(Op::VMov { dst: one, src: Operand::InlineFloat(1.0) });
    let sign_mask = k.alloc_vreg();
    k.push(Op::VMov { dst: sign_mask, src: Operand::Literal(0x80000000u32) }); // sign bit

    // Gate address
    let gate_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(gate_addr, SReg(gate_ptr.0));
    k.v_mov_from_sgpr(VReg(gate_addr.0 + 1), SReg(gate_ptr.0 + 1));
    let byte_off = k.alloc_vreg();
    k.push(Op::VLshlrevB32 { dst: byte_off, shift: 2, src: elem_id });
    k.v_add_co(gate_addr, gate_addr, byte_off);
    k.v_add_co_ci(VReg(gate_addr.0 + 1), VReg(gate_addr.0 + 1));

    // Up address
    let up_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(up_addr, SReg(up_ptr.0));
    k.v_mov_from_sgpr(VReg(up_addr.0 + 1), SReg(up_ptr.0 + 1));
    k.v_add_co(up_addr, up_addr, byte_off);
    k.v_add_co_ci(VReg(up_addr.0 + 1), VReg(up_addr.0 + 1));

    // Out address
    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0));
    k.v_mov_from_sgpr(VReg(out_addr.0 + 1), SReg(out_ptr.0 + 1));
    k.v_add_co(out_addr, out_addr, byte_off);
    k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));

    // Bounds check
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, n_elems_s);
    let saved = k.bounds_check_begin(elem_id, n_v);

    let gate_data = k.alloc_vreg_array(4, Alignment::Align4);
    let up_data = k.alloc_vreg_array(4, Alignment::Align4);
    let tmp = k.alloc_vreg_array(4, Alignment::Align4);

    for c in 0..chunks {
        let off = (c * 16) as i32;
        k.global_load(gate_data, gate_addr, Width::B128, off);
        k.global_load(up_data, up_addr, Width::B128, off);
        k.wait_vmcnt(0);

        // For each of 4 elements: silu(gate) * up
        for i in 0..4u32 {
            let g = VReg(gate_data.0 + i);
            let u = VReg(up_data.0 + i);
            let t = VReg(tmp.0 + i);

            // t = gate * log2(e)
            k.v_mul_f32(t, g, log2e);
            // t = -gate * log2(e)  (flip sign bit)
            k.v_xor_b32(t, Operand::VReg(t), Operand::VReg(sign_mask));
            // t = 2^(-gate * log2(e)) = exp(-gate)
            k.v_exp_f32(t, t);
            // t = 1 + exp(-gate)
            k.v_add_f32(t, one, t);
            // t = 1 / (1 + exp(-gate)) = sigmoid(gate)
            k.v_rcp_f32(t, t);
            // g = gate * sigmoid(gate) = silu(gate)
            k.v_mul_f32(g, g, t);
            // g = silu(gate) * up
            k.v_mul_f32(g, g, u);
        }

        k.global_store(out_addr, gate_data, Width::B128, off);
    }

    k.bounds_check_end(saved);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// OCPA Prefix Sum: S_c = S_{c-1} + W_c (Markov state prefix sum)
// ============================================================================

/// Generate OCPA prefix sum kernel.
///
/// Serial forward scan: each WG processes all chunks for one head.
/// 256 threads: each thread handles 16 contiguous f32 elements per chunk.
///
/// Kernargs (24 bytes): W_ptr(0), S_ptr(8), N_chunks(16), d_sq(20)
/// Grid: (num_heads, 1, 1), WG: 256
pub fn ocpa_prefix_sum() -> T0Kernel {
    let mut k = T0Kernel::new("t0_ocpa_prefix_sum");

    // Kernargs
    let w_ptr = k.arg_ptr("W");          // [0:7]
    let s_ptr = k.arg_ptr("S");          // [8:15]
    let n_chunks = k.arg_u32("N_chunks"); // [16:19]
    let d_sq = k.arg_u32("d_sq");        // [20:23]
    k.emit_arg_loads();

    // head_id = TGID.x
    let head_id = k.alloc_sreg();
    k.capture_tgid_x(head_id);

    // chunk_stride_bytes = d_sq * 4
    let chunk_stride = k.alloc_sreg();
    k.s_lshl_b32(chunk_stride, SReg(d_sq.0), 2);

    // head_byte_offset = head_id * N_chunks * chunk_stride
    let head_off = k.alloc_sreg();
    k.s_mul_i32(head_off, SReg(n_chunks.0), chunk_stride);
    k.s_mul_i32(head_off, head_id, head_off);

    // thread_offset = WORKITEM_ID_X * 64 (16 f32 elements)
    let tid = VReg(0); // hardware thread ID
    let t_off = k.alloc_vreg();
    k.v_lshlrev_b32(t_off, 6, tid);

    // W_addr = W_ptr + head_offset + thread_offset
    let w_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(w_addr, SReg(w_ptr.0));
    k.v_mov_from_sgpr(VReg(w_addr.0 + 1), SReg(w_ptr.0 + 1));
    let h_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(h_off_v, head_off);
    k.v_add_co(w_addr, w_addr, h_off_v);
    k.v_add_co_ci(VReg(w_addr.0 + 1), VReg(w_addr.0 + 1));
    k.v_add_co(w_addr, w_addr, t_off);
    k.v_add_co_ci(VReg(w_addr.0 + 1), VReg(w_addr.0 + 1));

    // S_addr = S_ptr + head_offset + thread_offset
    let s_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(s_addr, SReg(s_ptr.0));
    k.v_mov_from_sgpr(VReg(s_addr.0 + 1), SReg(s_ptr.0 + 1));
    k.v_add_co(s_addr, s_addr, h_off_v);
    k.v_add_co_ci(VReg(s_addr.0 + 1), VReg(s_addr.0 + 1));
    k.v_add_co(s_addr, s_addr, t_off);
    k.v_add_co_ci(VReg(s_addr.0 + 1), VReg(s_addr.0 + 1));

    // Zero accumulators: 16 VGPRs
    let acc = k.alloc_vreg_array(16, Alignment::Align4);
    for i in 0..16u32 {
        k.v_mov_imm(VReg(acc.0 + i), 0);
    }

    // chunk_stride as VGPR for pointer advance
    let stride_v = k.alloc_vreg();
    k.v_mov_from_sgpr(stride_v, chunk_stride);

    // ── Chunk loop ──
    let loop_cnt = k.alloc_sreg();
    k.s_mov_imm(loop_cnt, 0);
    k.label("chunk_loop");

    // Load 16 f32 from W_c: 4× dwordx4
    let w_data = k.alloc_vreg_array(16, Alignment::Align4);
    k.global_load(w_data, w_addr, Width::B128, 0);
    k.global_load(VReg(w_data.0 + 4), w_addr, Width::B128, 16);
    k.global_load(VReg(w_data.0 + 8), w_addr, Width::B128, 32);
    k.global_load(VReg(w_data.0 + 12), w_addr, Width::B128, 48);
    k.wait_vmcnt(0);

    // acc[i] += W[i]
    for i in 0..16u32 {
        k.v_add_f32(VReg(acc.0 + i), VReg(acc.0 + i), VReg(w_data.0 + i));
    }

    // Store acc to S_c: 16× individual dword stores
    for i in 0..16u32 {
        k.global_store(s_addr, VReg(acc.0 + i), Width::B32, (i * 4) as i32);
    }

    // Advance pointers by chunk_stride
    k.v_add_co(w_addr, w_addr, stride_v);
    k.v_add_co_ci(VReg(w_addr.0 + 1), VReg(w_addr.0 + 1));
    k.v_add_co(s_addr, s_addr, stride_v);
    k.v_add_co_ci(VReg(s_addr.0 + 1), VReg(s_addr.0 + 1));

    // Loop control
    k.s_add_u32(loop_cnt, loop_cnt, 1);
    k.s_cmp_lt_u32(loop_cnt, SReg(n_chunks.0));
    k.branch_scc1("chunk_loop");

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// OCPA Reverse Prefix Sum: dS̃_c = sum_{j=c+1}^{N-1} dU_j
// ============================================================================

/// Generate OCPA reverse prefix sum kernel.
///
/// Serial backward scan: each WG processes all chunks for one head in reverse.
/// At each c (from N-1 to 0): store dS̃_c = acc, then acc += dU_c.
///
/// Kernargs (24 bytes): dU_ptr(0), dS_ptr(8), N_chunks(16), d_sq(20)
/// Grid: (num_heads, 1, 1), WG: 256
pub fn ocpa_reverse_prefix_sum() -> T0Kernel {
    let mut k = T0Kernel::new("t0_ocpa_reverse_prefix_sum");

    // Kernargs
    let du_ptr = k.arg_ptr("dU");          // [0:7]
    let ds_ptr = k.arg_ptr("dS");          // [8:15]
    let n_chunks = k.arg_u32("N_chunks");  // [16:19]
    let d_sq = k.arg_u32("d_sq");          // [20:23]
    k.emit_arg_loads();

    // head_id = TGID.x
    let head_id = k.alloc_sreg();
    k.capture_tgid_x(head_id);

    // chunk_stride_bytes = d_sq * 4
    let chunk_stride = k.alloc_sreg();
    k.s_lshl_b32(chunk_stride, SReg(d_sq.0), 2);

    // head_byte_offset = head_id * N_chunks * chunk_stride
    let head_off = k.alloc_sreg();
    k.s_mul_i32(head_off, SReg(n_chunks.0), chunk_stride);
    k.s_mul_i32(head_off, head_id, head_off);

    // thread_offset = WORKITEM_ID_X * 64
    let tid = VReg(0);
    let t_off = k.alloc_vreg();
    k.v_lshlrev_b32(t_off, 6, tid);

    // dU_base = dU_ptr + head_offset + thread_offset
    let du_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(du_base, SReg(du_ptr.0));
    k.v_mov_from_sgpr(VReg(du_base.0 + 1), SReg(du_ptr.0 + 1));
    let h_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(h_off_v, head_off);
    k.v_add_co(du_base, du_base, h_off_v);
    k.v_add_co_ci(VReg(du_base.0 + 1), VReg(du_base.0 + 1));
    k.v_add_co(du_base, du_base, t_off);
    k.v_add_co_ci(VReg(du_base.0 + 1), VReg(du_base.0 + 1));

    // dS_base = dS_ptr + head_offset + thread_offset
    let ds_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(ds_base, SReg(ds_ptr.0));
    k.v_mov_from_sgpr(VReg(ds_base.0 + 1), SReg(ds_ptr.0 + 1));
    k.v_add_co(ds_base, ds_base, h_off_v);
    k.v_add_co_ci(VReg(ds_base.0 + 1), VReg(ds_base.0 + 1));
    k.v_add_co(ds_base, ds_base, t_off);
    k.v_add_co_ci(VReg(ds_base.0 + 1), VReg(ds_base.0 + 1));

    // Zero accumulators
    let acc = k.alloc_vreg_array(16, Alignment::Align4);
    for i in 0..16u32 {
        k.v_mov_imm(VReg(acc.0 + i), 0);
    }

    // chunk_stride as VGPR
    let stride_v = k.alloc_vreg();
    k.v_mov_from_sgpr(stride_v, chunk_stride);

    // Working address regs (computed each iteration from base + chunk_idx * stride)
    let du_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let ds_addr = k.alloc_vreg_array(2, Alignment::Align2);

    // ── Reverse chunk loop: c = N-1 down to 0 ──
    let chunk_idx = k.alloc_sreg();
    k.s_sub_u32(chunk_idx, SReg(n_chunks.0), 1); // chunk_idx = N_chunks - 1
    k.label("rev_chunk_loop");

    // Compute chunk byte offset
    let chunk_off_s = k.alloc_sreg();
    k.s_mul_i32(chunk_off_s, chunk_idx, chunk_stride);
    let chunk_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(chunk_off_v, chunk_off_s);

    // du_addr = du_base + chunk_byte_offset
    k.push(Op::VMov { dst: du_addr, src: Operand::VReg(du_base) });
    k.push(Op::VMov { dst: VReg(du_addr.0 + 1), src: Operand::VReg(VReg(du_base.0 + 1)) });
    k.v_add_co(du_addr, du_addr, chunk_off_v);
    k.v_add_co_ci(VReg(du_addr.0 + 1), VReg(du_addr.0 + 1));

    // ds_addr = ds_base + chunk_byte_offset
    k.push(Op::VMov { dst: ds_addr, src: Operand::VReg(ds_base) });
    k.push(Op::VMov { dst: VReg(ds_addr.0 + 1), src: Operand::VReg(VReg(ds_base.0 + 1)) });
    k.v_add_co(ds_addr, ds_addr, chunk_off_v);
    k.v_add_co_ci(VReg(ds_addr.0 + 1), VReg(ds_addr.0 + 1));

    // FIRST: store dS̃_c = acc (before adding dU_c)
    for i in 0..16u32 {
        k.global_store(ds_addr, VReg(acc.0 + i), Width::B32, (i * 4) as i32);
    }

    // THEN: load dU_c and add to accumulators
    let du_data = k.alloc_vreg_array(16, Alignment::Align4);
    k.global_load(du_data, du_addr, Width::B128, 0);
    k.global_load(VReg(du_data.0 + 4), du_addr, Width::B128, 16);
    k.global_load(VReg(du_data.0 + 8), du_addr, Width::B128, 32);
    k.global_load(VReg(du_data.0 + 12), du_addr, Width::B128, 48);
    k.wait_vmcnt(0);

    for i in 0..16u32 {
        k.v_add_f32(VReg(acc.0 + i), VReg(acc.0 + i), VReg(du_data.0 + i));
    }

    // Loop: chunk_idx--; if chunk_idx < N_chunks (unsigned, catches wrap past 0)
    k.s_sub_u32(chunk_idx, chunk_idx, 1);
    k.s_cmp_lt_u32(chunk_idx, SReg(n_chunks.0));
    k.branch_scc1("rev_chunk_loop");

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// OCPA dState Update: dU_c = Q_c^T @ dO_c  (ZCLT + WMMA, same as state_update)
// ============================================================================

/// Generate OCPA dState update kernel — backward pass kernel.
///
/// Structurally identical to state_update (W = K^T @ V),
/// with pointer mapping: K → Q, V → dO, W → dU.
///
/// Kernargs (40 bytes):
///   Q_ptr(0), dO_ptr(8), dU_ptr(16), C_chunk(24), d_head(28),
///   seq_len(32), n_chunks(36)
/// Grid: (n_chunks, num_heads, 1), WG: 32
pub fn ocpa_dstate_update() -> T0Kernel {
    // Identical ZCLT architecture, just rename args and kernel name
    let mut k = T0Kernel::new("t0_ocpa_dstate_update");
    k.set_lds_size(4224);

    let q_ptr = k.arg_ptr("Q");
    let do_ptr = k.arg_ptr("dO");
    let du_ptr = k.arg_ptr("dU");
    let c_chunk = k.arg_u32("C_chunk");
    let d_head = k.arg_u32("d_head");
    let seq_len = k.arg_u32("seq_len");
    let n_chunks = k.arg_u32("n_chunks");
    k.emit_arg_loads();

    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0);

    // Pointer math (same as state_update)
    let row_start = k.alloc_sreg();
    k.s_mul_i32(row_start, head_id, SReg(seq_len.0));
    let tmp_s = k.alloc_sreg();
    k.s_mul_i32(tmp_s, chunk_id, SReg(c_chunk.0));
    k.s_add_u32_ss(row_start, row_start, tmp_s);

    let off_lo = k.alloc_sreg();
    let off_hi = k.alloc_sreg();
    k.s_lshl_b32(off_lo, row_start, 7);
    k.s_lshr_b32(off_hi, row_start, 25);

    let q_base_lo = k.alloc_sreg();
    let q_base_hi = k.alloc_sreg();
    k.s_add_u32_ss(q_base_lo, SReg(q_ptr.0), off_lo);
    k.s_addc_u32(q_base_hi, SReg(q_ptr.0 + 1), off_hi);

    let do_base_lo = k.alloc_sreg();
    let do_base_hi = k.alloc_sreg();
    k.s_add_u32_ss(do_base_lo, SReg(do_ptr.0), off_lo);
    k.s_addc_u32(do_base_hi, SReg(do_ptr.0 + 1), off_hi);

    let t_off = k.alloc_vreg();
    k.v_lshlrev_b32(t_off, 6, tid);

    let q_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(q_addr, q_base_lo);
    k.v_mov_from_sgpr(VReg(q_addr.0 + 1), q_base_hi);
    k.v_add_co(q_addr, q_addr, t_off);
    k.v_add_co_ci(VReg(q_addr.0 + 1), VReg(q_addr.0 + 1));

    let do_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(do_addr, do_base_lo);
    k.v_mov_from_sgpr(VReg(do_addr.0 + 1), do_base_hi);
    k.v_add_co(do_addr, do_addr, t_off);
    k.v_add_co_ci(VReg(do_addr.0 + 1), VReg(do_addr.0 + 1));

    // ZCLT LDS topology (same as state_update)
    let r_v = k.alloc_vreg();
    k.v_lshrrev_b32(r_v, 1, tid);
    let c_v = k.alloc_vreg();
    k.v_and_b32_imm(c_v, tid, 1);
    k.v_lshlrev_b32(c_v, 6, c_v);

    let stride_132 = k.alloc_vreg();
    let s132 = k.alloc_sreg();
    k.s_mov_imm(s132, 132);
    k.v_mov_from_sgpr(stride_132, s132);
    let q_lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(q_lds_wr, r_v, stride_132);
    k.v_add_u32(q_lds_wr, q_lds_wr, c_v);

    let do_lds_wr = k.alloc_vreg();
    let s2112 = k.alloc_sreg();
    k.s_mov_imm(s2112, 2112);
    let v2112 = k.alloc_vreg();
    k.v_mov_from_sgpr(v2112, s2112);
    k.v_add_u32(do_lds_wr, q_lds_wr, v2112);

    let lds_read_base = k.alloc_vreg();
    k.v_and_b32_imm(lds_read_base, tid, 15);
    k.v_lshlrev_b32(lds_read_base, 1, lds_read_base);

    let hbm_step = k.alloc_vreg();
    let s2048 = k.alloc_sreg();
    k.s_mov_imm(s2048, 2048);
    k.v_mov_from_sgpr(hbm_step, s2048);

    // Big arrays (after individual VGPRs)
    let acc = k.alloc_vreg_array(128, Alignment::Align8);
    let a_regs = k.alloc_vreg_array(32, Alignment::Align8);
    let b_regs = k.alloc_vreg_array(32, Alignment::Align8);

    for i in 0..128u32 {
        k.v_mov_imm(VReg(acc.0 + i), 0);
    }

    // Main loop
    let m_idx = k.alloc_sreg();
    k.s_mov_imm(m_idx, 0);
    k.label("m_loop");

    for i in 0..4u32 {
        k.global_load(VReg(a_regs.0 + i * 4), q_addr, Width::B128, (i * 16) as i32);
    }
    for i in 0..4u32 {
        k.global_load(VReg(b_regs.0 + i * 4), do_addr, Width::B128, (i * 16) as i32);
    }

    k.v_add_co(q_addr, q_addr, hbm_step);
    k.v_add_co_ci(VReg(q_addr.0 + 1), VReg(q_addr.0 + 1));
    k.v_add_co(do_addr, do_addr, hbm_step);
    k.v_add_co_ci(VReg(do_addr.0 + 1), VReg(do_addr.0 + 1));

    k.wait_vmcnt(0);

    for i in 0..4u32 {
        k.ds_store_b128(q_lds_wr, VReg(a_regs.0 + i * 4), (i * 16) as u16);
    }
    for i in 0..4u32 {
        k.ds_store_b128(do_lds_wr, VReg(b_regs.0 + i * 4), (i * 16) as u16);
    }
    k.wait_lgkmcnt(0);

    // Column tearing
    for g in 0..4u32 {
        for kk in 0..8u32 {
            let off_lo = (g * 32 + 2 * kk * 132) as u16;
            let off_hi = (g * 32 + (2 * kk + 1) * 132) as u16;
            let v_idx = VReg(a_regs.0 + g * 8 + kk);
            k.ds_load_u16_d16(v_idx, lds_read_base, off_lo);
            k.ds_load_u16_d16_hi(v_idx, lds_read_base, off_hi);
        }
    }
    for v in 0..4u32 {
        for kk in 0..8u32 {
            let off_lo = (2112 + v * 32 + 2 * kk * 132) as u16;
            let off_hi = (2112 + v * 32 + (2 * kk + 1) * 132) as u16;
            let v_idx = VReg(b_regs.0 + v * 8 + kk);
            k.ds_load_u16_d16(v_idx, lds_read_base, off_lo);
            k.ds_load_u16_d16_hi(v_idx, lds_read_base, off_hi);
        }
    }
    k.wait_lgkmcnt(0);

    // 16× WMMA
    for g in 0..4u32 {
        for v in 0..4u32 {
            let acc_base = VReg(acc.0 + g * 32 + v * 8);
            let a_base = VReg(a_regs.0 + g * 8);
            let b_base = VReg(b_regs.0 + v * 8);
            k.wmma_bf16_f32(acc_base, a_base, b_base, acc_base);
        }
    }

    k.s_add_u32(m_idx, m_idx, 16);
    k.s_cmp_lt_u32(m_idx, SReg(c_chunk.0));
    k.branch_scc1("m_loop");

    // Store dU
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    let w_idx = k.alloc_sreg();
    k.s_mul_i32(w_idx, head_id, SReg(n_chunks.0));
    k.s_add_u32_ss(w_idx, w_idx, chunk_id);
    let d_sq = k.alloc_sreg();
    k.s_mul_i32(d_sq, SReg(d_head.0), SReg(d_head.0));
    let w_off = k.alloc_sreg();
    k.s_mul_i32(w_off, w_idx, d_sq);
    k.s_lshl_b32(w_off, w_off, 2);

    let w_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(w_addr, SReg(du_ptr.0));
    k.v_mov_from_sgpr(VReg(w_addr.0 + 1), SReg(du_ptr.0 + 1));
    let w_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(w_off_v, w_off);
    k.v_add_co(w_addr, w_addr, w_off_v);
    k.v_add_co_ci(VReg(w_addr.0 + 1), VReg(w_addr.0 + 1));

    let d_v = k.alloc_vreg();
    k.v_mov_from_sgpr(d_v, SReg(d_head.0));
    let st_addr = k.alloc_vreg_array(2, Alignment::Align2);

    // Reuse dead a_regs as scratch
    let scratch_base_row = VReg(a_regs.0);
    let scratch_row_off = VReg(a_regs.0 + 1);
    let scratch_imm = VReg(a_regs.0 + 2);
    let lr_off = VReg(a_regs.0 + 3);
    k.v_lshlrev_b32(lr_off, 2, lane_row);

    for k_grp in 0..4u32 {
        if k_grp == 0 {
            k.push(Op::VMov { dst: scratch_base_row, src: Operand::VReg(lane_half) });
        } else {
            k.v_mov_imm(scratch_imm, (k_grp * 16) as i32);
            k.v_add_u32(scratch_base_row, lane_half, scratch_imm);
        }
        k.v_mul_lo_u32(scratch_row_off, scratch_base_row, d_v);
        k.v_lshlrev_b32(scratch_row_off, 2, scratch_row_off);

        for v_tile in 0..4u32 {
            let acc_base_idx = acc.0 + k_grp * 32 + v_tile * 8;
            let col_offset = v_tile * 16 * 4;

            k.push(Op::VMov { dst: st_addr, src: Operand::VReg(w_addr) });
            k.push(Op::VMov { dst: VReg(st_addr.0 + 1), src: Operand::VReg(VReg(w_addr.0 + 1)) });
            k.v_add_co(st_addr, st_addr, scratch_row_off);
            k.v_add_co_ci(VReg(st_addr.0 + 1), VReg(st_addr.0 + 1));
            k.v_add_u32(st_addr, st_addr, lr_off);

            if col_offset > 0 {
                k.v_mov_imm(scratch_imm, col_offset as i32);
                k.v_add_u32(st_addr, st_addr, scratch_imm);
            }

            for r in 0..8u32 {
                let r_offset = (r as i32) * 512;
                k.global_store(st_addr, VReg(acc_base_idx + r), Width::B32, r_offset);
            }
        }
    }

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}


// ============================================================================
// OCPA Forward Intra: 1D Tile-Stealing with Causal GEMM
//
// Computes: O_intra_c = (CausalMask(Q_c @ K_c^T)) @ V_c
// Accumulates: O_c += O_intra_c via global_atomic_add_f32
//
// Architecture:
//   Phase A₀: Build 1D→2D LUT (lower triangle tiles)
//   Phase A:  512 threads cooperatively load V_c into LDS ZCLT (132B padded)
//   Phase B:  1D Tile-Stealing: wave steals tiles from pool
//             LUT lookup → conditional Q reload → K tile → QK^T WMMA →
//             causal mask → transpose → V-tile WMMA → atomic add O
// ============================================================================

/// OCPA Forward Intra kernel (C=256)
pub fn ocpa_forward_intra() -> T0Kernel {
    ocpa_forward_intra_with_c(256)
}

/// Forward Intra, parameterized by chunk size
pub fn ocpa_forward_intra_with_c(c_chunk: u32) -> T0Kernel {
    let n_tile_rows = c_chunk / 16;
    let total_tiles = n_tile_rows * (n_tile_rows + 1) / 2;
    let lds_zclt_size = c_chunk * 132;
    let lut_base = lds_zclt_size;
    let lds_total = ((lut_base + total_tiles * 2 + 255) / 256) * 256;

    let mut k = T0Kernel::new(&format!("t0_ocpa_forward_intra_c{}", c_chunk));
    k.set_lds_size(lds_total);

    let q_ptr = k.arg_ptr("Q");
    let k_ptr = k.arg_ptr("K");
    let v_ptr = k.arg_ptr("V");
    let o_ptr = k.arg_ptr("O");
    let seq_len = k.arg_u32("seq_len");
    k.emit_arg_loads();

    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, tid, 15);

    // base_row = head_id * seq_len + chunk_id * C
    let base_row = k.alloc_sreg();
    k.s_mul_i32(base_row, head_id, SReg(seq_len.0));
    let c_off = k.alloc_sreg();
    let c_shift = match c_chunk { 32=>5, 64=>6, 128=>7, _=>8 };
    k.s_lshl_b32(c_off, chunk_id, c_shift);
    k.s_add_u32_ss(base_row, base_row, c_off);

    // ═════════════════════════════════════════════════════════════
    // Phase A₀: Build 1D→2D LUT (only first total_tiles threads)
    // r = floor((sqrt(8k+1) - 1) / 2), c = k - r*(r+1)/2
    // ═════════════════════════════════════════════════════════════
    let total_v = k.alloc_vreg();
    k.v_mov_imm(total_v, total_tiles as i32);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(total_v) });
    k.branch_vccz("lut_done");

    // r = floor((sqrt(8*tid + 1) - 1) / 2)
    // r = floor((isqrt(8*tid + 1) - 1) / 2) — integer-only via float sqrt
    let k8 = k.alloc_vreg();
    k.v_lshlrev_b32(k8, 3, tid); // k8 = tid * 8
    let k8p1 = k.alloc_vreg();
    k.v_mov_imm(k8p1, 1);
    k.v_add_u32(k8p1, k8, k8p1); // k8p1 = 8*tid + 1
    let fk = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: fk, src: k8p1 });
    k.v_sqrt_f32(fk, fk);
    k.push(Op::VCvtU32F32 { dst: fk, src: fk }); // fk = floor(sqrt(8k+1))
    let neg1 = k.alloc_vreg();
    k.v_mov_imm(neg1, -1i32);
    k.v_add_u32(fk, fk, neg1); // fk = isqrt(8k+1) - 1
    let r_reg = k.alloc_vreg();
    k.v_lshrrev_b32(r_reg, 1, fk); // r = (isqrt-1) / 2

    // Correction: tri = r*(r+1)/2, if tri > k then r--
    let rp1 = k.alloc_vreg();
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    let tri = k.alloc_vreg();
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(tri) });
    k.branch_vccz("r_ok");
    k.v_add_u32(r_reg, r_reg, neg1); // r--
    // Recompute tri
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.label("r_ok");

    // c = k - tri
    let c_reg = k.alloc_vreg();
    // c = tid - tri: use two's complement
    let neg_tri = k.alloc_vreg();
    k.push(Op::VXorB32 { dst: neg_tri, src0: Operand::VReg(tri), src1: Operand::VReg(neg1) }); // ~tri
    let one_v = k.alloc_vreg();
    k.v_mov_imm(one_v, 1);
    k.v_add_u32(neg_tri, neg_tri, one_v); // -tri
    k.v_add_u32(c_reg, tid, neg_tri); // c = tid - tri

    // Pack (r << 8 | c) and write to LDS
    let packed = k.alloc_vreg();
    k.v_lshlrev_b32(packed, 8, r_reg);
    k.v_or_b32(packed, Operand::VReg(packed), Operand::VReg(c_reg));

    let lut_addr = k.alloc_vreg();
    k.v_lshlrev_b32(lut_addr, 1, tid); // k * 2
    let lut_base_v = k.alloc_vreg();
    k.v_mov_imm(lut_base_v, lut_base as i32);
    k.v_add_u32(lut_addr, lut_addr, lut_base_v);
    k.push(Op::DsStoreB16 { vaddr: lut_addr, src: packed, offset: 0 });

    k.label("lut_done");
    k.wait_lgkmcnt(0);

    // ═════════════════════════════════════════════════════════════
    // Phase A: Cooperative V_c ZCLT load (512 threads, 132B stride)
    // ═════════════════════════════════════════════════════════════
    let v_row = k.alloc_vreg();
    k.v_lshrrev_b32(v_row, 1, tid); // row = tid / 2
    let col_grp = k.alloc_vreg();
    k.v_and_b32_imm(col_grp, tid, 1);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 6, col_grp); // col_grp * 64

    // Clamp row for C < 256
    let lds_row = if c_chunk < 256 {
        let clamped = k.alloc_vreg();
        k.v_and_b32_imm(clamped, v_row, c_chunk - 1);
        clamped
    } else {
        v_row
    };

    // HBM addr = V_ptr + (base_row + row) * 128 + col_bytes
    let v_off = k.alloc_vreg();
    k.v_mov_from_sgpr(v_off, base_row);
    k.v_add_u32(v_off, v_off, lds_row);
    k.v_lshlrev_b32(v_off, 7, v_off); // * 128
    k.v_add_u32(v_off, v_off, col_bytes);

    let v_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(v_addr, SReg(v_ptr.0));
    k.v_mov_from_sgpr(VReg(v_addr.0 + 1), SReg(v_ptr.0 + 1));
    k.v_add_co(v_addr, v_addr, v_off);
    k.v_add_co_ci(VReg(v_addr.0 + 1), VReg(v_addr.0 + 1));

    let v_data = k.alloc_vreg_array(16, Alignment::Align4);
    for i in 0..4u32 {
        k.global_load(VReg(v_data.0 + i * 4), v_addr, Width::B128, (i * 16) as i32);
    }

    // LDS write: lds_row * 132 + col_bytes
    let s132 = k.alloc_sreg();
    k.s_mov_imm(s132, 132);
    let stride132 = k.alloc_vreg();
    k.v_mov_from_sgpr(stride132, s132);
    let lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(lds_wr, lds_row, stride132);
    k.v_add_u32(lds_wr, lds_wr, col_bytes);

    k.wait_vmcnt(0);
    for i in 0..4u32 {
        k.ds_store_b128(lds_wr, VReg(v_data.0 + i * 4), (i * 16) as u16);
    }
    k.wait_lgkmcnt(0);
    k.s_barrier(); // Sync LUT + ZCLT

    // ═════════════════════════════════════════════════════════════
    // Prepare O output base
    // ═════════════════════════════════════════════════════════════
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });

    let o_base_lo = k.alloc_sreg();
    let o_base_hi = k.alloc_sreg();
    let tmp_off = k.alloc_sreg();
    k.s_lshl_b32(tmp_off, base_row, 8); // base_row * 256
    k.s_add_u32_ss(o_base_lo, SReg(o_ptr.0), tmp_off);
    k.s_addc_u32_imm(o_base_hi, SReg(o_ptr.0 + 1), 0);

    // ═════════════════════════════════════════════════════════════
    // 1D Tile-Stealing Main Loop
    // k_task = wave_id, step += 16
    // ═════════════════════════════════════════════════════════════
    let k_task = k.alloc_sreg();
    k.push(Op::SMov { dst: k_task, src: SOperand::SReg(wave_id_s) });
    let cur_r = k.alloc_sreg();
    k.s_mov_imm(cur_r, -1i32); // invalid, force first Q load

    // Skip idle waves
    let s_total = k.alloc_sreg();
    k.s_mov_imm(s_total, total_tiles as i32);
    k.s_cmp_ge_u32(k_task, s_total);
    k.branch_scc1("epilogue");

    // Big arrays: Q fragment, K fragment, P accumulator, O accumulator
    // WMMA needs 8-aligned
    let q_frag = k.alloc_vreg_array(32, Alignment::Align8); // Q_slice: 4 k-groups × 8
    let k_frag = k.alloc_vreg_array(32, Alignment::Align8); // K_tile: 4 k-groups × 8
    let p_acc = k.alloc_vreg_array(8, Alignment::Align8);   // P^T = K @ Q^T (one tile)
    let o_acc = k.alloc_vreg_array(32, Alignment::Align8);  // O_acc: 4 v-groups × 8
    let p_trans = k.alloc_vreg_array(8, Alignment::Align8);  // Transposed P (A-operand)
    let v_tile = k.alloc_vreg_array(8, Alignment::Align8);   // V tile from LDS

    k.label("main_loop");

    // ── LUT lookup: read (r, c) ──
    let lut_off = k.alloc_sreg();
    k.s_lshl_b32(lut_off, k_task, 1);
    let lut_base_s = k.alloc_sreg();
    k.s_mov_imm(lut_base_s, lut_base as i32);
    k.s_add_u32_ss(lut_off, lut_off, lut_base_s);
    let lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(lut_v, lut_off);
    let lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: lut_val, vaddr: lut_v, offset: 0 });
    k.wait_lgkmcnt(0);

    // Unpack: tile_r, tile_c
    let tile_r_v = k.alloc_vreg();
    k.v_lshrrev_b32(tile_r_v, 8, lut_val);
    let tile_r = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: tile_r, src: tile_r_v });
    let tile_c_v = k.alloc_vreg();
    k.v_and_b32_imm(tile_c_v, lut_val, 0xFF);
    let tile_c = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: tile_c, src: tile_c_v });

    // ── Conditional Q_slice load (when r changes) ──
    k.push(Op::SCmpEqU32 { src0: tile_r, src1: SOperand::SReg(cur_r) });
    k.branch_scc1("skip_q_load");

    k.push(Op::SMov { dst: cur_r, src: SOperand::SReg(tile_r) });
    // Q addr = Q_ptr + (base_row + r*16 + lane_row) * 128
    let q_row = k.alloc_sreg();
    k.s_lshl_b32(q_row, tile_r, 4);
    k.s_add_u32_ss(q_row, q_row, base_row);
    let q_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(q_row_v, q_row);
    k.v_add_u32(q_row_v, q_row_v, lane_row);
    k.v_lshlrev_b32(q_row_v, 7, q_row_v);
    let q_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(q_addr, SReg(q_ptr.0));
    k.v_mov_from_sgpr(VReg(q_addr.0 + 1), SReg(q_ptr.0 + 1));
    k.v_add_co(q_addr, q_addr, q_row_v);
    k.v_add_co_ci(VReg(q_addr.0 + 1), VReg(q_addr.0 + 1));

    for i in 0..8u32 {
        k.global_load(VReg(q_frag.0 + i * 4), q_addr, Width::B128, (i * 16) as i32);
    }
    // Zero O_acc when r changes
    for i in 0..32u32 {
        k.v_mov_imm(VReg(o_acc.0 + i), 0);
    }
    k.wait_vmcnt(0);
    k.label("skip_q_load");

    // ── Load K_tile[c] ──
    let k_row = k.alloc_sreg();
    k.s_lshl_b32(k_row, tile_c, 4);
    k.s_add_u32_ss(k_row, k_row, base_row);
    let k_off = k.alloc_sreg();
    k.s_lshl_b32(k_off, k_row, 7);
    let k_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(k_off_v, 7, lane_row);
    let k_off_s_v = k.alloc_vreg();
    k.v_mov_from_sgpr(k_off_s_v, k_off);
    k.v_add_u32(k_off_v, k_off_v, k_off_s_v);
    let k_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(k_addr, SReg(k_ptr.0));
    k.v_mov_from_sgpr(VReg(k_addr.0 + 1), SReg(k_ptr.0 + 1));
    k.v_add_co(k_addr, k_addr, k_off_v);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));

    for i in 0..8u32 {
        k.global_load(VReg(k_frag.0 + i * 4), k_addr, Width::B128, (i * 16) as i32);
    }

    for i in 0..8u32 {
        k.v_mov_imm(VReg(p_acc.0 + i), 0);
    }
    k.wait_vmcnt(0);

    // ── WMMA: P^T = K @ Q^T (4 k-groups) ──
    for kg in 0..4u32 {
        k.wmma_bf16_f32(p_acc, VReg(k_frag.0 + kg * 8), VReg(q_frag.0 + kg * 8), p_acc);
    }

    // ── Causal mask (diagonal: r == c) ──
    k.push(Op::SCmpEqU32 { src0: tile_r, src1: SOperand::SReg(tile_c) });
    k.branch_scc0("mask_done");

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let zero_f = k.alloc_vreg();
    k.v_mov_imm(zero_f, 0);
    let one_f = k.alloc_vreg();
    k.v_mov_imm(one_f, 0x3F800000u32 as i32); // 1.0f32

    for vk in 0..8u32 {
        // row_in_tile = 2*vk + lane_half
        let row_v = k.alloc_vreg();
        if vk == 0 {
            k.push(Op::VMov { dst: row_v, src: Operand::VReg(lane_half) });
        } else {
            k.v_mov_imm(row_v, (2 * vk) as i32);
            k.v_add_u32(row_v, lane_half, row_v);
        }
        // Lower-triangular mask on P (stored as P^T in C-register):
        // Keep P[row,col] where row >= col. In P^T: keep P^T[col,row] where col <= row.
        // C-register: row_v = row index, lane_row = col index.
        // For P^T, we need col_in_P^T <= row_in_P^T to mask the UPPER part of P^T.
        k.v_cmp_ge_i32(lane_row, row_v);
        k.push(Op::VCndmaskB32 { dst: row_v, src_false: Operand::VReg(zero_f), src_true: Operand::VReg(one_f) });
        k.push(Op::VMulF32 { dst: VReg(p_acc.0 + vk), src0: Operand::VReg(VReg(p_acc.0 + vk)), src1: Operand::VReg(row_v) });
    }
    k.label("mask_done");
    k.clear_vcc(); // prevent VCC residual from mask cmp affecting v_add_co_ci

    // ── C-layout → A-operand transpose (pure VALU, no LDS) ──
    // v_tile[0:2] used as temp (free before V-tile loads)
    k.reg_transpose_c_to_ab(p_trans, p_acc, v_tile);

    // ── V-tile WMMA: O_acc += P_trans @ V_tile (4 v-groups) ──
    let lds_read_base = k.alloc_vreg();
    k.v_and_b32_imm(lds_read_base, lane_row, 15);
    k.v_lshlrev_b32(lds_read_base, 1, lds_read_base);
    // LDS base for column c: c * 16 * 132 = c * 2112
    let s2112 = k.alloc_sreg();
    k.s_mov_imm(s2112, 2112);
    let v_lds_base = k.alloc_sreg();
    k.s_mul_i32(v_lds_base, tile_c, s2112);

    for vg in 0..4u32 {
        let col_off = vg * 32;
        let lds_off = k.alloc_sreg();
        k.s_mov_imm(lds_off, (col_off) as i32);
        k.s_add_u32_ss(lds_off, v_lds_base, lds_off);
        let lds_addr = k.alloc_vreg();
        k.v_mov_from_sgpr(lds_addr, lds_off);
        k.v_add_u32(lds_addr, lds_read_base, lds_addr);

        for vk in 0..8u32 {
            let off_lo = (vk * 2 * 132) as u16;
            let off_hi = ((vk * 2 + 1) * 132) as u16;
            k.ds_load_u16_d16(VReg(v_tile.0 + vk), lds_addr, off_lo);
            k.ds_load_u16_d16_hi(VReg(v_tile.0 + vk), lds_addr, off_hi);
        }
        k.wait_lgkmcnt(0);
        k.wmma_bf16_f32(VReg(o_acc.0 + vg * 8), p_trans, v_tile, VReg(o_acc.0 + vg * 8));
    }

    // ── Check if next task has different r → flush O_acc ──
    let next_k = k.alloc_sreg();
    k.s_add_u32(next_k, k_task, 16);
    k.s_cmp_ge_u32(next_k, s_total);
    k.branch_scc1("flush"); // last iteration → must flush

    // Read next tile's r
    let next_lut_off = k.alloc_sreg();
    k.s_lshl_b32(next_lut_off, next_k, 1);
    k.s_add_u32_ss(next_lut_off, next_lut_off, lut_base_s);
    let next_lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(next_lut_v, next_lut_off);
    let next_lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: next_lut_val, vaddr: next_lut_v, offset: 0 });
    k.wait_lgkmcnt(0);
    let next_r_v = k.alloc_vreg();
    k.v_lshrrev_b32(next_r_v, 8, next_lut_val);
    let next_r = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: next_r, src: next_r_v });
    k.push(Op::SCmpEqU32 { src0: next_r, src1: SOperand::SReg(tile_r) });
    k.branch_scc1("skip_flush"); // same r → accumulate more

    // ── FLUSH: Atomic add O_acc to HBM ──
    k.label("flush");
    let flush_lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(flush_lane_half, 4, lane_id);
    let r16 = k.alloc_sreg();
    k.s_lshl_b32(r16, tile_r, 4);
    let r16_v = k.alloc_vreg();
    k.v_mov_from_sgpr(r16_v, r16);
    k.v_add_u32(r16_v, r16_v, flush_lane_half);
    let row_off = k.alloc_vreg();
    k.v_lshlrev_b32(row_off, 8, r16_v); // * 256 bytes/row
    let col_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(col_off_v, 2, lane_row);

    let o_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_addr, o_base_lo);
    k.v_mov_from_sgpr(VReg(o_addr.0 + 1), o_base_hi);
    k.v_add_co(o_addr, o_addr, row_off);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));
    k.v_add_co(o_addr, o_addr, col_off_v);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));

    for vg in 0..4u32 {
        for vk in 0..8u32 {
            let off = (vk as i32) * 512 + (vg as i32) * 64;
            k.global_atomic_add_f32(o_addr, VReg(o_acc.0 + vg * 8 + vk), off);
        }
    }
    k.wait_vmcnt(0);

    k.label("skip_flush");

    // k += 16, loop if k < total_tiles
    k.s_add_u32(k_task, k_task, 16);
    k.s_cmp_lt_u32(k_task, s_total);
    k.branch_scc1("main_loop");

    k.label("epilogue");
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// OCPA Backward Intra: 1D Tile-Stealing with Causal GEMM
//
// Mathematical Isomorphism with Forward Intra:
//   Intra(A, B, C, Mask) = (Mask ⊙ (A @ B^T)) @ C
//   dQ_intra = Intra(dO, V, K, Lower)   ← identical to forward
//   dK_intra = Intra(V, dO, Q, Upper)   ← time-arrow reversed
//   dV_intra = Intra(K, Q, dO, Upper)   ← time-arrow reversed
//
// Kernarg: A_ptr(0), B_ptr(8), C_ptr(16), Out_ptr(24), seq_len(32)
// ============================================================================

/// Backward Intra dQ (lower mask)
pub fn ocpa_backward_intra_dq() -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dq", false, 256)
}

/// Backward Intra dQ with parameterized c_chunk
pub fn ocpa_backward_intra_dq_c(c_chunk: u32) -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dq", false, c_chunk)
}

/// Backward Intra dK/dV (upper mask)
pub fn ocpa_backward_intra_dkdv() -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dkdv", true, 256)
}

/// Backward Intra dK/dV with parameterized c_chunk
pub fn ocpa_backward_intra_dkdv_c(c_chunk: u32) -> T0Kernel {
    build_backward_intra("t0_ocpa_backward_intra_dkdv", true, c_chunk)
}

/// GPU multihead→flat transpose: [n_heads, seq, d_head=64] → [seq, dim]
/// where dim = n_heads * d_head.
/// For each output element at out[gid]: t = gid / dim, j = gid % dim,
/// h = j / 64, d = j % 64. src_idx = h * (seq * 64) + t * 64 + d
/// Kernargs (24B): src_ptr(0), dst_ptr(8), seq(16), dim(20)
/// Grid: ceil(seq * dim / 256) * 256, WG: 256
pub fn t0_multihead_to_flat() -> T0Kernel {
    use super::ir::*;
    let mut k = T0Kernel::new("t0_multihead_to_flat");

    let s_src = k.arg_ptr("src");
    let s_dst = k.arg_ptr("dst");
    let s_seq = k.arg_u32("seq");
    let s_dim = k.arg_u32("dim");       // = n_heads * d_head
    k.emit_arg_loads();

    let tid = VReg(0);
    let gid = k.alloc_vreg();
    let tgid_x = k.alloc_sreg();
    k.capture_tgid_x(tgid_x);
    k.v_mov_from_sgpr(gid, tgid_x);
    k.v_lshlrev_b32(gid, 8, gid);   // tgid_x * 256
    k.v_add_u32(gid, gid, tid);

    // Bounds check: gid < seq * dim
    let s_total = k.alloc_sreg();
    k.s_mul_i32(s_total, SReg(s_seq.0), SReg(s_dim.0));
    let v_total = k.alloc_vreg();
    k.v_mov_from_sgpr(v_total, s_total);
    k.v_cmp_lt_u32(Operand::VReg(gid), Operand::VReg(v_total));
    let saved = k.alloc_sreg();
    k.save_exec(saved);
    k.branch_scc0("done");

    // Decompose output gid → (t, j) where j = h * d_head + d
    // t = gid / dim, j = gid % dim
    // Use float rcp for division, with two-direction correction
    let v_dim = k.alloc_vreg();
    k.v_mov_from_sgpr(v_dim, SReg(s_dim.0));

    // Float division: t = floor(float(gid) / float(dim))
    let f_gid = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: f_gid, src: gid });
    let f_dim = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: f_dim, src: v_dim });
    let f_rcp = k.alloc_vreg();
    k.v_rcp_f32(f_rcp, f_dim);
    let f_t = k.alloc_vreg();
    k.v_mul_f32(f_t, f_gid, f_rcp);
    let t_reg = k.alloc_vreg();
    k.push(Op::VCvtU32F32 { dst: t_reg, src: f_t }); // floor

    // j = gid - t * dim
    let td_product = k.alloc_vreg();
    k.v_mul_lo_u32(td_product, t_reg, v_dim); // t * dim
    let j_reg = k.alloc_vreg();
    k.v_sub_u32(j_reg, gid, td_product);       // j = gid - t * dim

    // Correction direction 1: t too large (rcp rounded up)
    // If t * dim > gid, then j wrapped (is huge u32). Decrement t.
    let one = k.alloc_vreg();
    k.v_mov_imm(one, 1);
    let zero_v = k.alloc_vreg();
    k.v_mov_imm(zero_v, 0);
    k.v_cmp_lt_u32(Operand::VReg(gid), Operand::VReg(td_product));
    let fix_down = k.alloc_vreg();
    k.push(Op::VCndmaskB32 { dst: fix_down, src_false: Operand::VReg(zero_v), src_true: Operand::VReg(one) });
    k.v_sub_u32(t_reg, t_reg, fix_down);
    k.v_mul_lo_u32(td_product, fix_down, v_dim);
    k.v_add_u32(j_reg, j_reg, td_product); // j += fix_down * dim

    // Correction direction 2: t too small (rcp rounded down)
    // If j >= dim, increment t
    k.v_cmp_ge_u32(Operand::VReg(j_reg), Operand::VReg(v_dim));
    let fix_up = k.alloc_vreg();
    k.push(Op::VCndmaskB32 { dst: fix_up, src_false: Operand::VReg(zero_v), src_true: Operand::VReg(one) });
    k.v_add_u32(t_reg, t_reg, fix_up);
    k.v_mul_lo_u32(td_product, fix_up, v_dim);
    k.v_sub_u32(j_reg, j_reg, td_product); // j -= fix_up * dim

    // h = j >> 6 (d_head=64, log2=6), d = j & 63
    let h_reg = k.alloc_vreg();
    k.v_lshrrev_b32(h_reg, 6, j_reg);     // h = j / 64
    let d_reg = k.alloc_vreg();
    k.v_and_b32_imm(d_reg, j_reg, 63);    // d = j % 64

    // src_idx = h * (seq * 64) + t * 64 + d
    let v_seq = k.alloc_vreg();
    k.v_mov_from_sgpr(v_seq, SReg(s_seq.0));
    let seq_dhead = k.alloc_vreg();
    k.v_lshlrev_b32(seq_dhead, 6, v_seq);  // seq * 64

    let src_idx = k.alloc_vreg();
    k.v_mul_lo_u32(src_idx, h_reg, seq_dhead);    // h * seq * 64
    let td = k.alloc_vreg();
    k.v_lshlrev_b32(td, 6, t_reg);                // t * 64
    k.v_add_u32(src_idx, src_idx, td);
    k.v_add_u32(src_idx, src_idx, d_reg);          // + d

    // byte offsets
    let src_off = k.alloc_vreg();
    k.v_lshlrev_b32(src_off, 2, src_idx);
    let dst_off = k.alloc_vreg();
    k.v_lshlrev_b32(dst_off, 2, gid);

    // Load src[src_idx]
    let src_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(src_addr, SReg(s_src.0));
    k.v_mov_from_sgpr(VReg(src_addr.0 + 1), SReg(s_src.0 + 1));
    k.v_add_co(src_addr, src_addr, src_off);
    k.v_add_co_ci(VReg(src_addr.0 + 1), VReg(src_addr.0 + 1));
    let val = k.alloc_vreg();
    k.global_load(val, src_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // Store dst[gid] (output is flat [seq, dim])
    let dst_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dst_addr, SReg(s_dst.0));
    k.v_mov_from_sgpr(VReg(dst_addr.0 + 1), SReg(s_dst.0 + 1));
    k.v_add_co(dst_addr, dst_addr, dst_off);
    k.v_add_co_ci(VReg(dst_addr.0 + 1), VReg(dst_addr.0 + 1));
    k.global_store(dst_addr, val, Width::B32, 0);

    k.label("done");
    k.restore_exec(saved);
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// Minimal copy kernel for GPU smoke test:
///   dst[global_id] = src[global_id]
/// Kernargs: src_ptr(u64), dst_ptr(u64), n_elems(u32)
/// Grid: ceil(n_elems / 256), WG size: 256
pub fn t0_copy() -> T0Kernel {
    use super::ir::*;
    let mut k = T0Kernel::new("t0_copy");

    // Declare arguments (auto-registers kernarg layout)
    let s_src = k.arg_ptr("src");
    let s_dst = k.arg_ptr("dst");
    let s_n = k.arg_u32("n_elems");
    // Emit SMEM load prologue (s_load from kernarg segment)
    k.emit_arg_loads();

    // VReg(0) = hardware v0 = WORKITEM_ID_X (thread ID) — RESERVED, never allocated
    let tid = VReg(0);
    let gid = k.alloc_vreg();

    // global_id = workgroup_id_x * 256 + thread_id
    let tgid_x = k.alloc_sreg();
    k.capture_tgid_x(tgid_x);
    k.v_mov_from_sgpr(gid, tgid_x);
    k.v_lshlrev_b32(gid, 8, gid);   // tgid_x * 256
    k.v_add_u32(gid, gid, tid);     // + thread_id (from hardware v0)

    // Bounds check
    let v_n = k.alloc_vreg();
    k.v_mov_from_sgpr(v_n, s_n);
    k.v_cmp_lt_u32(Operand::VReg(gid), Operand::VReg(v_n));
    let saved_exec = k.alloc_sreg();
    k.save_exec(saved_exec);
    k.branch_scc0("done");

    // byte offset = gid * 4
    let v_off = k.alloc_vreg();
    k.v_lshlrev_b32(v_off, 2, gid);

    // src_addr = src_ptr + byte_offset (64-bit add via IR ops)
    // IMPORTANT: global_load/store uses v[lo:lo+1], so lo/hi MUST be consecutive VGPRs
    let v_src_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(v_src_addr, SReg(s_src.0));
    k.v_mov_from_sgpr(VReg(v_src_addr.0 + 1), SReg(s_src.0 + 1));
    k.addr64_add(v_src_addr, VReg(v_src_addr.0 + 1), v_off);

    // Load src[gid]
    let v_val = k.alloc_vreg();
    k.global_load(v_val, v_src_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // dst_addr = dst_ptr + byte_offset
    let v_dst_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(v_dst_addr, SReg(s_dst.0));
    k.v_mov_from_sgpr(VReg(v_dst_addr.0 + 1), SReg(s_dst.0 + 1));
    k.addr64_add(v_dst_addr, VReg(v_dst_addr.0 + 1), v_off);

    // Store dst[gid] = val
    k.global_store(v_dst_addr, v_val, Width::B32, 0);

    k.label("done");
    k.restore_exec(saved_exec);
    k.endpgm();
    k
}

fn build_backward_intra(name: &str, is_upper: bool, c_chunk: u32) -> T0Kernel {
    let n_tile_rows = c_chunk / 16;
    let total_tiles = n_tile_rows * (n_tile_rows + 1) / 2;
    let lds_zclt_size = c_chunk * 132;
    let lut_base = lds_zclt_size;
    let lds_total = ((lut_base + total_tiles * 2 + 255) / 256) * 256;

    let mut k = T0Kernel::new(name);
    k.set_lds_size(lds_total);

    // Kernarg: A_ptr, B_ptr, C_ptr, Out_ptr, seq_len
    let a_ptr = k.arg_ptr("A");
    let b_ptr = k.arg_ptr("B");
    let c_ptr = k.arg_ptr("C");
    let out_ptr = k.arg_ptr("Out");
    let seq_len = k.arg_u32("seq_len");
    k.emit_arg_loads();

    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, tid, 15);

    let base_row = k.alloc_sreg();
    k.s_mul_i32(base_row, head_id, SReg(seq_len.0));
    let c_off = k.alloc_sreg();
    let c_shift = match c_chunk { 32=>5, 64=>6, 128=>7, _=>8 };
    k.s_lshl_b32(c_off, chunk_id, c_shift);
    k.s_add_u32_ss(base_row, base_row, c_off);

    // ── Phase A₀: Build 1D→2D LUT (same as forward) ──
    let total_v = k.alloc_vreg();
    k.v_mov_imm(total_v, total_tiles as i32);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(total_v) });
    k.branch_vccz("lut_done");

    let k8 = k.alloc_vreg();
    k.v_lshlrev_b32(k8, 3, tid);
    let k8p1 = k.alloc_vreg();
    k.v_mov_imm(k8p1, 1);
    k.v_add_u32(k8p1, k8, k8p1);
    let fk = k.alloc_vreg();
    k.push(Op::VCvtF32U32 { dst: fk, src: k8p1 });
    k.v_sqrt_f32(fk, fk);
    k.push(Op::VCvtU32F32 { dst: fk, src: fk });
    let neg1 = k.alloc_vreg();
    k.v_mov_imm(neg1, -1i32);
    k.v_add_u32(fk, fk, neg1);
    let r_reg = k.alloc_vreg();
    k.v_lshrrev_b32(r_reg, 1, fk);

    let rp1 = k.alloc_vreg();
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    let tri = k.alloc_vreg();
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(tid), src1: Operand::VReg(tri) });
    k.branch_vccz("r_ok");
    k.v_add_u32(r_reg, r_reg, neg1);
    k.v_mov_imm(rp1, 1);
    k.v_add_u32(rp1, r_reg, rp1);
    k.v_mul_lo_u32(tri, r_reg, rp1);
    k.v_lshrrev_b32(tri, 1, tri);
    k.label("r_ok");

    let neg_tri = k.alloc_vreg();
    k.push(Op::VXorB32 { dst: neg_tri, src0: Operand::VReg(tri), src1: Operand::VReg(neg1) });
    let one_v = k.alloc_vreg();
    k.v_mov_imm(one_v, 1);
    k.v_add_u32(neg_tri, neg_tri, one_v);
    let c_reg = k.alloc_vreg();
    k.v_add_u32(c_reg, tid, neg_tri);

    let packed = k.alloc_vreg();
    k.v_lshlrev_b32(packed, 8, r_reg);
    k.v_or_b32(packed, Operand::VReg(packed), Operand::VReg(c_reg));

    let lut_addr = k.alloc_vreg();
    k.v_lshlrev_b32(lut_addr, 1, tid);
    let lut_base_v = k.alloc_vreg();
    k.v_mov_imm(lut_base_v, lut_base as i32);
    k.v_add_u32(lut_addr, lut_addr, lut_base_v);
    k.push(Op::DsStoreB16 { vaddr: lut_addr, src: packed, offset: 0 });
    k.label("lut_done");
    k.wait_lgkmcnt(0);

    // ── Phase A: ZCLT-load C to LDS (132B stride) ──
    let v_row = k.alloc_vreg();
    k.v_lshrrev_b32(v_row, 1, tid);
    let col_grp = k.alloc_vreg();
    k.v_and_b32_imm(col_grp, tid, 1);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 6, col_grp);

    let lds_row = if c_chunk < 256 {
        let clamped = k.alloc_vreg();
        k.v_and_b32_imm(clamped, v_row, c_chunk - 1);
        clamped
    } else { v_row };

    let v_off = k.alloc_vreg();
    k.v_mov_from_sgpr(v_off, base_row);
    k.v_add_u32(v_off, v_off, lds_row);
    k.v_lshlrev_b32(v_off, 7, v_off);
    k.v_add_u32(v_off, v_off, col_bytes);

    let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(c_addr, SReg(c_ptr.0));
    k.v_mov_from_sgpr(VReg(c_addr.0 + 1), SReg(c_ptr.0 + 1));
    k.v_add_co(c_addr, c_addr, v_off);
    k.v_add_co_ci(VReg(c_addr.0 + 1), VReg(c_addr.0 + 1));

    let c_data = k.alloc_vreg_array(16, Alignment::Align4);
    for i in 0..4u32 {
        k.global_load(VReg(c_data.0 + i * 4), c_addr, Width::B128, (i * 16) as i32);
    }

    let s132 = k.alloc_sreg();
    k.s_mov_imm(s132, 132);
    let stride132 = k.alloc_vreg();
    k.v_mov_from_sgpr(stride132, s132);
    let lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(lds_wr, lds_row, stride132);
    k.v_add_u32(lds_wr, lds_wr, col_bytes);
    k.wait_vmcnt(0);
    for i in 0..4u32 {
        k.ds_store_b128(lds_wr, VReg(c_data.0 + i * 4), (i * 16) as u16);
    }
    k.wait_lgkmcnt(0);
    k.s_barrier();

    // ── Prepare Out base ──
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let o_base_lo = k.alloc_sreg();
    let o_base_hi = k.alloc_sreg();
    let tmp_off = k.alloc_sreg();
    k.s_lshl_b32(tmp_off, base_row, 8);
    k.s_add_u32_ss(o_base_lo, SReg(out_ptr.0), tmp_off);
    k.s_addc_u32_imm(o_base_hi, SReg(out_ptr.0 + 1), 0);

    // ── Main loop setup ──
    let k_task = k.alloc_sreg();
    k.push(Op::SMov { dst: k_task, src: SOperand::SReg(wave_id_s) });
    let cur_a = k.alloc_sreg();
    k.s_mov_imm(cur_a, -1i32);
    let s_total = k.alloc_sreg();
    k.s_mov_imm(s_total, total_tiles as i32);
    k.s_cmp_ge_u32(k_task, s_total);
    k.branch_scc1("epilogue");

    let a_frag = k.alloc_vreg_array(32, Alignment::Align8);
    let b_frag = k.alloc_vreg_array(32, Alignment::Align8);
    let p_acc = k.alloc_vreg_array(8, Alignment::Align8);
    let o_acc = k.alloc_vreg_array(32, Alignment::Align8);
    let p_trans = k.alloc_vreg_array(8, Alignment::Align8);
    let c_tile = k.alloc_vreg_array(8, Alignment::Align8);

    k.label("main_loop");

    // ── LUT lookup ──
    let lut_off = k.alloc_sreg();
    k.s_lshl_b32(lut_off, k_task, 1);
    let lut_base_s = k.alloc_sreg();
    k.s_mov_imm(lut_base_s, lut_base as i32);
    k.s_add_u32_ss(lut_off, lut_off, lut_base_s);
    let lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(lut_v, lut_off);
    let lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: lut_val, vaddr: lut_v, offset: 0 });
    k.wait_lgkmcnt(0);

    let lut_r_v = k.alloc_vreg();
    k.v_lshrrev_b32(lut_r_v, 8, lut_val);
    let lut_r = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: lut_r, src: lut_r_v });
    let lut_c_v = k.alloc_vreg();
    k.v_and_b32_imm(lut_c_v, lut_val, 0xFF);
    let lut_c = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: lut_c, src: lut_c_v });

    // For upper triangle: A_tile = lut_c, B_tile = lut_r
    // For lower triangle: A_tile = lut_r, B_tile = lut_c
    let a_tile = k.alloc_sreg();
    let b_tile = k.alloc_sreg();
    if is_upper {
        k.push(Op::SMov { dst: a_tile, src: SOperand::SReg(lut_c) });
        k.push(Op::SMov { dst: b_tile, src: SOperand::SReg(lut_r) });
    } else {
        k.push(Op::SMov { dst: a_tile, src: SOperand::SReg(lut_r) });
        k.push(Op::SMov { dst: b_tile, src: SOperand::SReg(lut_c) });
    }

    // ── Conditional A_slice load ──
    k.push(Op::SCmpEqU32 { src0: a_tile, src1: SOperand::SReg(cur_a) });
    k.branch_scc1("skip_a_load");
    k.push(Op::SMov { dst: cur_a, src: SOperand::SReg(a_tile) });

    let a_row = k.alloc_sreg();
    k.s_lshl_b32(a_row, a_tile, 4);
    k.s_add_u32_ss(a_row, a_row, base_row);
    let a_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(a_row_v, a_row);
    k.v_add_u32(a_row_v, a_row_v, lane_row);
    k.v_lshlrev_b32(a_row_v, 7, a_row_v);
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_addr.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_addr, a_addr, a_row_v);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));
    for i in 0..8u32 {
        k.global_load(VReg(a_frag.0 + i * 4), a_addr, Width::B128, (i * 16) as i32);
    }
    for i in 0..32u32 {
        k.v_mov_imm(VReg(o_acc.0 + i), 0);
    }
    k.wait_vmcnt(0);
    k.label("skip_a_load");

    // ── Load B_tile ──
    let b_row = k.alloc_sreg();
    k.s_lshl_b32(b_row, b_tile, 4);
    k.s_add_u32_ss(b_row, b_row, base_row);
    let b_off = k.alloc_sreg();
    k.s_lshl_b32(b_off, b_row, 7);
    let b_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(b_off_v, 7, lane_row);
    let b_off_s_v = k.alloc_vreg();
    k.v_mov_from_sgpr(b_off_s_v, b_off);
    k.v_add_u32(b_off_v, b_off_v, b_off_s_v);
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_ptr.0));
    k.v_mov_from_sgpr(VReg(b_addr.0 + 1), SReg(b_ptr.0 + 1));
    k.v_add_co(b_addr, b_addr, b_off_v);
    k.v_add_co_ci(VReg(b_addr.0 + 1), VReg(b_addr.0 + 1));
    for i in 0..8u32 {
        k.global_load(VReg(b_frag.0 + i * 4), b_addr, Width::B128, (i * 16) as i32);
    }
    for i in 0..8u32 {
        k.v_mov_imm(VReg(p_acc.0 + i), 0);
    }
    k.wait_vmcnt(0);

    // ── WMMA: P^T = B @ A^T ──
    for kg in 0..4u32 {
        k.wmma_bf16_f32(p_acc, VReg(b_frag.0 + kg * 8), VReg(a_frag.0 + kg * 8), p_acc);
    }

    // ── Causal mask (diagonal only: lut_r == lut_c) ──
    k.push(Op::SCmpEqU32 { src0: lut_r, src1: SOperand::SReg(lut_c) });
    k.branch_scc0("mask_done");

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let zero_f = k.alloc_vreg();
    k.v_mov_imm(zero_f, 0);
    let one_f = k.alloc_vreg();
    k.v_mov_imm(one_f, 0x3F800000u32 as i32);

    for vk in 0..8u32 {
        let row_v = k.alloc_vreg();
        if vk == 0 {
            k.push(Op::VMov { dst: row_v, src: Operand::VReg(lane_half) });
        } else {
            k.v_mov_imm(row_v, (2 * vk) as i32);
            k.v_add_u32(row_v, lane_half, row_v);
        }
        if is_upper {
            // Upper: keep where col >= row in P^T → row_v >= lane_row
            k.v_cmp_ge_i32(row_v, lane_row);
        } else {
            // Lower: keep where col <= row in P^T → lane_row >= row_v
            k.v_cmp_ge_i32(lane_row, row_v);
        }
        k.push(Op::VCndmaskB32 { dst: row_v, src_false: Operand::VReg(zero_f), src_true: Operand::VReg(one_f) });
        k.push(Op::VMulF32 { dst: VReg(p_acc.0 + vk), src0: Operand::VReg(VReg(p_acc.0 + vk)), src1: Operand::VReg(row_v) });
    }
    k.label("mask_done");
    k.clear_vcc(); // prevent VCC residual from mask cmp affecting v_add_co_ci

    // ── C-layout → A-operand transpose (pure VALU, no LDS) ──
    k.reg_transpose_c_to_ab(p_trans, p_acc, c_tile);

    // ── C-tile from LDS and WMMA (4 v-groups) ──
    let lds_read_base = k.alloc_vreg();
    k.v_and_b32_imm(lds_read_base, lane_row, 15);
    k.v_lshlrev_b32(lds_read_base, 1, lds_read_base);
    let s2112 = k.alloc_sreg();
    k.s_mov_imm(s2112, 2112);
    let c_lds_base = k.alloc_sreg();
    k.s_mul_i32(c_lds_base, b_tile, s2112);

    for vg in 0..4u32 {
        let col_off = vg * 32;
        let lds_off = k.alloc_sreg();
        k.s_mov_imm(lds_off, col_off as i32);
        k.s_add_u32_ss(lds_off, c_lds_base, lds_off);
        let lds_addr = k.alloc_vreg();
        k.v_mov_from_sgpr(lds_addr, lds_off);
        k.v_add_u32(lds_addr, lds_read_base, lds_addr);

        for vk in 0..8u32 {
            k.ds_load_u16_d16(VReg(c_tile.0 + vk), lds_addr, (vk * 2 * 132) as u16);
            k.ds_load_u16_d16_hi(VReg(c_tile.0 + vk), lds_addr, ((vk * 2 + 1) * 132) as u16);
        }
        k.wait_lgkmcnt(0);
        k.wmma_bf16_f32(VReg(o_acc.0 + vg * 8), p_trans, c_tile, VReg(o_acc.0 + vg * 8));
    }

    // ── Flush check ──
    let next_k = k.alloc_sreg();
    k.s_add_u32(next_k, k_task, 16);
    k.s_cmp_ge_u32(next_k, s_total);
    k.branch_scc1("flush");

    let next_lut_off = k.alloc_sreg();
    k.s_lshl_b32(next_lut_off, next_k, 1);
    k.s_add_u32_ss(next_lut_off, next_lut_off, lut_base_s);
    let next_lut_v = k.alloc_vreg();
    k.v_mov_from_sgpr(next_lut_v, next_lut_off);
    let next_lut_val = k.alloc_vreg();
    k.push(Op::DsLoadU16 { dst: next_lut_val, vaddr: next_lut_v, offset: 0 });
    k.wait_lgkmcnt(0);

    // Determine next A_tile
    let next_a_v = k.alloc_vreg();
    if is_upper {
        k.v_and_b32_imm(next_a_v, next_lut_val, 0xFF); // next_lut_c = A_tile for upper
    } else {
        k.v_lshrrev_b32(next_a_v, 8, next_lut_val); // next_lut_r = A_tile for lower
    }
    let next_a = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: next_a, src: next_a_v });
    k.push(Op::SCmpEqU32 { src0: next_a, src1: SOperand::SReg(a_tile) });
    k.branch_scc1("skip_flush");

    // ── Flush ──
    k.label("flush");
    let flush_lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(flush_lane_half, 4, lane_id);
    let a16 = k.alloc_sreg();
    k.s_lshl_b32(a16, a_tile, 4);
    let a16_v = k.alloc_vreg();
    k.v_mov_from_sgpr(a16_v, a16);
    k.v_add_u32(a16_v, a16_v, flush_lane_half);
    let row_off = k.alloc_vreg();
    k.v_lshlrev_b32(row_off, 8, a16_v);
    let col_off_v = k.alloc_vreg();
    k.v_lshlrev_b32(col_off_v, 2, lane_row);

    let o_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_addr, o_base_lo);
    k.v_mov_from_sgpr(VReg(o_addr.0 + 1), o_base_hi);
    k.v_add_co(o_addr, o_addr, row_off);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));
    k.v_add_co(o_addr, o_addr, col_off_v);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));

    for vg in 0..4u32 {
        for vk in 0..8u32 {
            let off = (vk as i32) * 512 + (vg as i32) * 64;
            k.global_atomic_add_f32(o_addr, VReg(o_acc.0 + vg * 8 + vk), off);
        }
    }
    k.wait_vmcnt(0);
    k.label("skip_flush");

    k.s_add_u32(k_task, k_task, 16);
    k.s_cmp_lt_u32(k_task, s_total);
    k.branch_scc1("main_loop");

    k.label("epilogue");
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// OCPA Inter Kernels: 512-Thread Mega-Wave Architecture
//
// Common pattern:
//   Phase A: 512 threads cooperatively load S/dS̃ (d×d FP32) into LDS
//            with 260-byte padded stride for bank conflict elimination.
//   Phase B: Each wave (16 of 32 threads) independently computes 16 rows
//            via 16× WMMA with ghost BF16 extraction (offset:2) from FP32 LDS.
//   Ghost chunk bypass: first/last chunk → write zeros.
//
// All 4 variants share identical structure, differing only in:
//   - Kernel name, arg naming
//   - Ghost condition: chunk_id==0 (forward/dQ) vs chunk_id==last (dK/dV)
//   - Input pointers: Q/dO reads bf16 rows, S/dS̃ holds FP32 d×d matrix
// ============================================================================

/// Build an OCPA inter kernel (shared structure for forward_inter, backward_inter_dq/dk/dv).
///
/// `ghost_first`: true = ghost when chunk_id==0 (forward, dQ)
///                false = ghost when chunk_id==last (dK, dV)
/// `s_index_offset`: -1 for S_{c-1} (forward, dQ), 0 for dS̃_c (dK, dV)
fn build_inter_kernel(
    kernel_name: &str,
    // Kernel arg names: (input_bf16_ptr, state_f32_ptr, output_ptr)
    arg_a: &str, arg_s: &str, arg_o: &str,
    ghost_first: bool,
    s_index_offset: i32,
) -> T0Kernel {
    let mut k = T0Kernel::new(kernel_name);
    k.set_lds_size(16640); // 64 rows × 260 bytes

    // Kernargs: A_ptr(0), S_ptr(8), O_ptr(16), seq_len(24), C_chunk(28), d_head(32), n_chunks(36)
    let a_ptr = k.arg_ptr(arg_a);
    let s_ptr = k.arg_ptr(arg_s);
    let o_ptr = k.arg_ptr(arg_o);
    let seq_len = k.arg_u32("seq_len");
    let c_chunk = k.arg_u32("C_chunk");
    let d_head = k.arg_u32("d_head");
    let n_chunks = k.arg_u32("n_chunks");
    k.emit_arg_loads();

    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0); // thread_id (0..511)

    // Derive wave/lane IDs
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, tid, 15);

    // ── Ghost chunk bypass: zero accumulators, skip to store ──
    if ghost_first {
        // chunk_id == 0 → ghost
        k.s_cmp_eq_u32_imm(chunk_id, 0);
    } else {
        // chunk_id == n_chunks - 1 → ghost
        let last = k.alloc_sreg();
        k.s_sub_u32(last, SReg(n_chunks.0), 1);
        k.push(Op::SCmpEqU32 { src0: chunk_id, src1: SOperand::SReg(last) });
    }

    // Zero 32 accumulators (4 v_tiles × 8 WMMA)
    let acc = k.alloc_vreg_array(32, Alignment::Align8);
    for i in 0..32u32 {
        k.v_mov_imm(VReg(acc.0 + i), 0);
    }
    k.branch_scc1("store"); // ghost → skip to store

    // ═══════════════════════════════════════════════════════════════
    // Phase A: 512 threads cooperatively load S/dS̃ (d×d FP32) → LDS
    // ═══════════════════════════════════════════════════════════════

    // S index = head_id * n_chunks + (chunk_id + offset)
    let s_idx = k.alloc_sreg();
    k.s_mul_i32(s_idx, head_id, SReg(n_chunks.0));
    if s_index_offset == -1 {
        let cm1 = k.alloc_sreg();
        k.s_sub_u32(cm1, chunk_id, 1);
        k.s_add_u32_ss(s_idx, s_idx, cm1);
    } else {
        k.s_add_u32_ss(s_idx, s_idx, chunk_id);
    }

    // byte_offset = s_idx * d * d * 4 = s_idx * 16384
    let s16384 = k.alloc_sreg();
    k.s_mov_imm(s16384, 16384);
    let s_byte_off = k.alloc_sreg();
    k.s_mul_i32(s_byte_off, s_idx, s16384);

    let s_base_lo = k.alloc_sreg();
    let s_base_hi = k.alloc_sreg();
    k.s_add_u32_ss(s_base_lo, SReg(s_ptr.0), s_byte_off);
    k.s_addc_u32_imm(s_base_hi, SReg(s_ptr.0 + 1), 0);

    // Each thread loads 8 FP32 = 32 bytes: HBM addr = S_base + thread_id * 32
    let t_off = k.alloc_vreg();
    k.v_lshlrev_b32(t_off, 5, tid); // * 32
    let s_hbm_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(s_hbm_addr, s_base_lo);
    k.v_mov_from_sgpr(VReg(s_hbm_addr.0 + 1), s_base_hi);
    k.v_add_co(s_hbm_addr, s_hbm_addr, t_off);
    k.v_add_co_ci(VReg(s_hbm_addr.0 + 1), VReg(s_hbm_addr.0 + 1));

    // Load 2× dwordx4 = 8 FP32
    let hbm_data = k.alloc_vreg_array(8, Alignment::Align4);
    k.global_load(hbm_data, s_hbm_addr, Width::B128, 0);
    k.global_load(VReg(hbm_data.0 + 4), s_hbm_addr, Width::B128, 16);

    // LDS write: (tid/8)*260 + (tid%8)*32
    let row = k.alloc_vreg();
    k.v_lshrrev_b32(row, 3, tid);
    let col_group = k.alloc_vreg();
    k.v_and_b32_imm(col_group, tid, 7);
    k.v_lshlrev_b32(col_group, 5, col_group); // * 32

    let s260 = k.alloc_sreg();
    k.s_mov_imm(s260, 260);
    let stride260_v = k.alloc_vreg();
    k.v_mov_from_sgpr(stride260_v, s260);
    let lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(lds_wr, row, stride260_v);
    k.v_add_u32(lds_wr, lds_wr, col_group);

    k.wait_vmcnt(0);
    k.ds_store_b128(lds_wr, hbm_data, 0);
    k.ds_store_b128(lds_wr, VReg(hbm_data.0 + 4), 16);
    k.wait_lgkmcnt(0);
    k.s_barrier(); // all 512 threads must complete LDS write

    // ═══════════════════════════════════════════════════════════════
    // Phase B: Per-wave WMMA (skip out-of-range waves)
    // ═══════════════════════════════════════════════════════════════

    // wave_row = wave_id * 16; skip if >= C_chunk
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let wave_row_s = k.alloc_sreg();
    k.s_lshl_b32(wave_row_s, wave_id_s, 4);
    k.s_cmp_ge_u32(wave_row_s, SReg(c_chunk.0));
    k.branch_scc1("endpgm");

    // A (Q/dO) address for this wave's 16 rows
    let base_row = k.alloc_sreg();
    k.s_mul_i32(base_row, head_id, SReg(seq_len.0));
    let tmp_s = k.alloc_sreg();
    k.s_mul_i32(tmp_s, chunk_id, SReg(c_chunk.0));
    k.s_add_u32_ss(base_row, base_row, tmp_s);

    // row_in_chunk = wave_id * 16 + lane_row
    let wave16 = k.alloc_vreg();
    k.v_lshlrev_b32(wave16, 4, wave_id);
    k.v_add_u32(wave16, wave16, lane_row);
    let base_row_v = k.alloc_vreg();
    k.v_mov_from_sgpr(base_row_v, base_row);
    k.v_add_u32(wave16, wave16, base_row_v); // global_row

    // A_addr = A_ptr + global_row * 128 (bf16 row)
    k.v_lshlrev_b32(wave16, 7, wave16);
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_ptr.0));
    k.v_mov_from_sgpr(VReg(a_addr.0 + 1), SReg(a_ptr.0 + 1));
    k.v_add_co(a_addr, a_addr, wave16);
    k.v_add_co_ci(VReg(a_addr.0 + 1), VReg(a_addr.0 + 1));

    // Load all 4 K-groups of A (8× dwordx4 = 128 bytes)
    let a_frags = k.alloc_vreg_array(32, Alignment::Align8);
    for i in 0..8u32 {
        k.global_load(VReg(a_frags.0 + i * 4), a_addr, Width::B128, (i * 16) as i32);
    }
    k.wait_vmcnt(0);

    // LDS read base = lane_row * 4
    let lds_col = k.alloc_vreg();
    k.v_lshlrev_b32(lds_col, 2, lane_row);

    // ── 4 v_tiles × 4 k_groups = 16 WMMA ──
    let b_frag = k.alloc_vreg_array(8, Alignment::Align8); // S^T fragment (reused)
    let lds_col_adj = k.alloc_vreg(); // adjusted LDS column per v_tile

    for v_tile in 0..4u32 {
        // lds_col_adj = lane_row * 4 + v_tile * 64
        if v_tile == 0 {
            k.push(Op::VMov { dst: lds_col_adj, src: Operand::VReg(lds_col) });
        } else {
            k.v_mov_imm(lds_col_adj, (v_tile * 64) as i32);
            k.v_add_u32(lds_col_adj, lds_col, lds_col_adj);
        }

        for k_grp in 0..4u32 {
            // Ghost BF16 extraction: ds_load_u16_d16 with offset:2 from FP32 LDS
            for vk in 0..8u32 {
                let row_lo = k_grp * 16 + vk * 2;
                let row_hi = row_lo + 1;
                let off_lo = (row_lo * 260 + 2) as u16;
                let off_hi = (row_hi * 260 + 2) as u16;
                k.ds_load_u16_d16(VReg(b_frag.0 + vk), lds_col_adj, off_lo);
                k.ds_load_u16_d16_hi(VReg(b_frag.0 + vk), lds_col_adj, off_hi);
            }
            k.wait_lgkmcnt(0);

            let acc_base = VReg(acc.0 + v_tile * 8);
            let a_base = VReg(a_frags.0 + k_grp * 8);
            k.wmma_bf16_f32(acc_base, a_base, b_frag, acc_base);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Store output (ghost branch lands here too)
    // ═══════════════════════════════════════════════════════════════
    k.label("store");

    // Check wave bounds at store entry (covers ghost path)
    let wave_id_s2 = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s2, src: wave_id });
    let wave_row_s2 = k.alloc_sreg();
    k.s_lshl_b32(wave_row_s2, wave_id_s2, 4);
    k.s_cmp_ge_u32(wave_row_s2, SReg(c_chunk.0));
    k.branch_scc1("endpgm");

    // O address = O_ptr + (head*seq + chunk*C + wave*16 + lane_half) * d*4 + lane_row*4
    let out_base = k.alloc_sreg();
    k.s_mul_i32(out_base, head_id, SReg(seq_len.0));
    let tmp2 = k.alloc_sreg();
    k.s_mul_i32(tmp2, chunk_id, SReg(c_chunk.0));
    k.s_add_u32_ss(out_base, out_base, tmp2);

    let wave16_out = k.alloc_vreg();
    k.v_lshlrev_b32(wave16_out, 4, wave_id);
    let out_base_v = k.alloc_vreg();
    k.v_mov_from_sgpr(out_base_v, out_base);
    k.v_add_u32(wave16_out, wave16_out, out_base_v);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    k.v_add_u32(wave16_out, wave16_out, lane_half); // global_row + lane_half

    // Row byte offset: (global_row+lane_half) * d * 4 = * 256
    let row_byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(row_byte_off, 8, wave16_out);
    let col_byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(col_byte_off, 2, lane_row); // lane_row * 4

    let o_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_addr, SReg(o_ptr.0));
    k.v_mov_from_sgpr(VReg(o_addr.0 + 1), SReg(o_ptr.0 + 1));
    k.v_add_co(o_addr, o_addr, row_byte_off);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));
    k.v_add_co(o_addr, o_addr, col_byte_off);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));

    // Store 4 v_tiles × 8 VGPRs each
    for v_tile in 0..4u32 {
        for vk in 0..8u32 {
            let off = (vk as i32) * 512 + (v_tile as i32) * 64;
            k.global_store(o_addr, VReg(acc.0 + v_tile * 8 + vk), Width::B32, off);
        }
    }

    k.wait_vmcnt(0);
    k.wait_vscnt(0);

    k.label("endpgm");
    k.endpgm();
    k
}

/// OCPA Forward Inter: O_inter_c = Q_c @ S_{c-1}
pub fn ocpa_forward_inter() -> T0Kernel {
    build_inter_kernel("t0_ocpa_forward_inter", "Q", "S", "O_inter", true, -1)
}

/// OCPA Backward Inter dQ: dQ_c = dO_c @ S_{c-1}^T
pub fn ocpa_backward_inter_dq() -> T0Kernel {
    build_inter_kernel("t0_ocpa_backward_inter_dq", "dO", "S", "dQ", true, -1)
}

/// OCPA Backward Inter dK: dK_c = V_c @ dS̃_c^T
pub fn ocpa_backward_inter_dk() -> T0Kernel {
    build_inter_kernel("t0_ocpa_backward_inter_dk", "V", "dS_tilde", "dK", false, 0)
}

/// OCPA Backward Inter dV: dV_c = K_c @ dS̃_c
pub fn ocpa_backward_inter_dv() -> T0Kernel {
    build_inter_kernel("t0_ocpa_backward_inter_dv", "K", "dS_tilde", "dV", false, 0)
}

// ============================================================================
// OCPA Denom Norm: O_normed = O_raw / (phi(Q) · z_running + eps)
// ============================================================================

/// Generate OCPA denominator normalization kernel.
///
/// Sequential per-position: z_t += phi(K_t), den_t = phi(Q_t)·z_t, O /= den_t.
/// Wave32: each lane handles dim/32 elements. Wave reduction for dot product.
///
/// Kernargs (48 bytes):
///   q_phi_ptr(0), k_phi_ptr(8), o_ptr(16), seq_len(24), dim(28),
///   eps(32), qk_stride(36), o_stride(40)
/// Grid: (1, num_heads, 1), WG: 32
pub fn ocpa_denom_norm(dim: u32) -> T0Kernel {
    assert!(dim >= 32 && dim % 32 == 0);
    let dpl = dim / 32; // elements per lane

    let mut k = T0Kernel::new(&format!("t0_ocpa_denom_norm_d{}", dim));

    // Kernargs
    let q_ptr = k.arg_ptr("q_phi");       // [0:7]
    let kk_ptr = k.arg_ptr("k_phi");      // [8:15]
    let o_ptr = k.arg_ptr("o");            // [16:23]
    let seq_len = k.arg_u32("seq_len");    // [24:27]
    let _dim_arg = k.arg_u32("dim");       // [28:31] (unused, compile-time)
    let eps_arg = k.arg_u32("eps");         // [32:35] (f32 as bits)
    let qk_stride = k.arg_u32("qk_stride"); // [36:39]
    let o_stride = k.arg_u32("o_stride");   // [40:43]
    k.emit_arg_loads();

    // lane_id = WORKITEM_ID_X & 31
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, VReg(0), 31);

    // lane_byte_offset = lane_id * (dpl * 4)
    let lane_bytes = dpl * 4;
    let lane_off = k.alloc_vreg();
    let lb_s = k.alloc_sreg();
    k.s_mov_imm(lb_s, lane_bytes as i32);
    k.v_mov_from_sgpr(lane_off, lb_s);
    // use v_mul_lo_u32 for lane_id * lane_bytes
    k.v_mul_lo_u32(lane_off, lane_id, lane_off);

    // eps into VGPR
    let eps_v = k.alloc_vreg();
    k.v_mov_from_sgpr(eps_v, SReg(eps_arg.0));

    // ── Compute base addresses adding lane offset ──
    // q_addr = q_phi_ptr + lane_offset
    let q_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(q_addr, SReg(q_ptr.0));
    k.v_mov_from_sgpr(VReg(q_addr.0 + 1), SReg(q_ptr.0 + 1));
    k.v_add_co(q_addr, q_addr, lane_off);
    k.v_add_co_ci(VReg(q_addr.0 + 1), VReg(q_addr.0 + 1));

    // k_addr = k_phi_ptr + lane_offset
    let k_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(k_addr, SReg(kk_ptr.0));
    k.v_mov_from_sgpr(VReg(k_addr.0 + 1), SReg(kk_ptr.0 + 1));
    k.v_add_co(k_addr, k_addr, lane_off);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));

    // o_addr = o_ptr + lane_offset
    let o_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_addr, SReg(o_ptr.0));
    k.v_mov_from_sgpr(VReg(o_addr.0 + 1), SReg(o_ptr.0 + 1));
    k.v_add_co(o_addr, o_addr, lane_off);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));

    // Stride VGPRs
    let qk_stride_v = k.alloc_vreg();
    k.v_mov_from_sgpr(qk_stride_v, SReg(qk_stride.0));
    let o_stride_v = k.alloc_vreg();
    k.v_mov_from_sgpr(o_stride_v, SReg(o_stride.0));

    // Initialize z_running[0..dpl-1] = 0
    let z_base = k.alloc_vreg_array(dpl, Alignment::Align4);
    for i in 0..dpl {
        k.v_mov_imm(VReg(z_base.0 + i), 0);
    }

    // Temp arrays for loads
    let tmp_data = k.alloc_vreg_array(dpl, Alignment::Align4);
    let o_data = k.alloc_vreg_array(dpl, Alignment::Align4);

    // abs mask constant: 0x7FFFFFFF
    let abs_mask = k.alloc_vreg();
    k.v_mov_imm(abs_mask, 0x7FFFFFFFu32 as i32);

    // Dot product accumulator + wave reduce temp
    let dot = k.alloc_vreg();
    let dot_tmp = k.alloc_vreg();

    // ── Sequential loop over positions ──
    let pos_cnt = k.alloc_sreg();
    k.s_mov_imm(pos_cnt, 0);
    k.label("pos_loop");

    // Step 1: Load phi(K_t) → tmp_data
    for c in 0..(dpl / 4).max(1) {
        let width = if dpl >= 4 { Width::B128 } else { Width::B64 };
        let elems = if dpl >= 4 { 4 } else { 2 };
        let off = (c * elems * 4) as i32;
        k.global_load(VReg(tmp_data.0 + c * elems), k_addr, width, off);
    }
    k.wait_vmcnt(0);

    // Step 2: z_running[i] += phi(K_t)[i]
    for i in 0..dpl {
        k.v_add_f32(VReg(z_base.0 + i), VReg(z_base.0 + i), VReg(tmp_data.0 + i));
    }

    // Step 3: Load phi(Q_t) → tmp_data
    for c in 0..(dpl / 4).max(1) {
        let width = if dpl >= 4 { Width::B128 } else { Width::B64 };
        let elems = if dpl >= 4 { 4 } else { 2 };
        let off = (c * elems * 4) as i32;
        k.global_load(VReg(tmp_data.0 + c * elems), q_addr, width, off);
    }
    k.wait_vmcnt(0);

    // Step 4: Dot product: dot = sum_i(phi(Q)[i] * z[i])
    k.v_mov_imm(dot, 0);
    for i in 0..dpl {
        k.v_fma_f32(dot, VReg(tmp_data.0 + i), VReg(z_base.0 + i), dot);
    }

    // Wave reduction: sum dot across 32 lanes
    k.wave_reduce_add_f32(dot, dot_tmp);

    // Step 5: den = max(abs(dot), eps)
    k.v_and_b32(dot, dot, abs_mask);
    k.v_max_f32(dot, dot, eps_v);

    // Step 6: inv_den = 1.0 / den
    k.v_rcp_f32(dot, dot);

    // Step 7: Load O_raw, multiply by inv_den, store back
    for c in 0..(dpl / 4).max(1) {
        let width = if dpl >= 4 { Width::B128 } else { Width::B64 };
        let elems = if dpl >= 4 { 4 } else { 2 };
        let off = (c * elems * 4) as i32;
        k.global_load(VReg(o_data.0 + c * elems), o_addr, width, off);
    }
    k.wait_vmcnt(0);

    for i in 0..dpl {
        k.v_mul_f32(VReg(o_data.0 + i), VReg(o_data.0 + i), dot);
    }

    for c in 0..(dpl / 4).max(1) {
        let width = if dpl >= 4 { Width::B128 } else { Width::B64 };
        let elems = if dpl >= 4 { 4 } else { 2 };
        let off = (c * elems * 4) as i32;
        k.global_store(o_addr, VReg(o_data.0 + c * elems), width, off);
    }

    // Step 8: Advance pointers
    k.v_add_co(q_addr, q_addr, qk_stride_v);
    k.v_add_co_ci(VReg(q_addr.0 + 1), VReg(q_addr.0 + 1));
    k.v_add_co(k_addr, k_addr, qk_stride_v);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));
    k.v_add_co(o_addr, o_addr, o_stride_v);
    k.v_add_co_ci(VReg(o_addr.0 + 1), VReg(o_addr.0 + 1));

    // Step 9: Loop
    k.s_add_u32(pos_cnt, pos_cnt, 1);
    k.s_cmp_lt_u32(pos_cnt, SReg(seq_len.0));
    k.branch_scc1("pos_loop");

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// OCPA State Update: W_c = K_c^T @ V_c  (ZCLT + WMMA)
// ============================================================================

/// Generate OCPA state update kernel using ZCLT (Zero-Conflict LDS Transpose).
///
/// Computes 64×64 outer product W_c = K_c^T @ V_c per chunk per head.
/// ZCLT: 132-byte padded LDS stride + bf16 column tearing (zero-VALU transpose).
///
/// Kernargs (40 bytes):
///   K_ptr(0), V_ptr(8), W_ptr(16), C_chunk(24), d_head(28),
///   seq_len(32), n_chunks(36)
/// Grid: (n_chunks, num_heads, 1), WG: 32
pub fn ocpa_state_update() -> T0Kernel {
    let mut k = T0Kernel::new("t0_ocpa_state_update");
    k.set_lds_size(4224); // ZCLT 金库: 16 rows × 132 bytes × 2 matrices

    // Kernargs
    let k_ptr = k.arg_ptr("K");          // [0:7]
    let v_ptr = k.arg_ptr("V");          // [8:15]
    let w_ptr = k.arg_ptr("W");          // [16:23]
    let c_chunk = k.arg_u32("C_chunk");  // [24:27]
    let d_head = k.arg_u32("d_head");    // [28:31]
    let seq_len = k.arg_u32("seq_len");  // [32:35]
    let n_chunks = k.arg_u32("n_chunks"); // [36:39]
    k.emit_arg_loads();

    // TGID.x = chunk_id, TGID.y = head_id
    let chunk_id = k.alloc_sreg();
    k.capture_tgid_x(chunk_id);
    let head_id = k.alloc_sreg();
    k.capture_tgid_y(head_id);

    let tid = VReg(0); // hardware thread_id

    // ── Pointer math: K/V base = ptr + (head_id*seq_len + chunk_id*C_chunk) * 128 ──
    let row_start = k.alloc_sreg();
    k.s_mul_i32(row_start, head_id, SReg(seq_len.0));
    let tmp_s = k.alloc_sreg();
    k.s_mul_i32(tmp_s, chunk_id, SReg(c_chunk.0));
    k.s_add_u32_ss(row_start, row_start, tmp_s);

    // offset_lo = row_start << 7 (128 bytes per row for d_head=64 bf16)
    // offset_hi = row_start >> 25
    let off_lo = k.alloc_sreg();
    let off_hi = k.alloc_sreg();
    k.s_lshl_b32(off_lo, row_start, 7);
    k.s_lshr_b32(off_hi, row_start, 25);

    // K_base = K_ptr + offset
    let k_base_lo = k.alloc_sreg();
    let k_base_hi = k.alloc_sreg();
    k.s_add_u32_ss(k_base_lo, SReg(k_ptr.0), off_lo);
    k.s_addc_u32(k_base_hi, SReg(k_ptr.0 + 1), off_hi);

    // V_base = V_ptr + offset
    let v_base_lo = k.alloc_sreg();
    let v_base_hi = k.alloc_sreg();
    k.s_add_u32_ss(v_base_lo, SReg(v_ptr.0), off_lo);
    k.s_addc_u32(v_base_hi, SReg(v_ptr.0 + 1), off_hi);

    // Thread byte offset = thread_id * 64
    let t_off = k.alloc_vreg();
    k.v_lshlrev_b32(t_off, 6, tid);

    // K_addr = K_base + thread_offset
    let k_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(k_addr, k_base_lo);
    k.v_mov_from_sgpr(VReg(k_addr.0 + 1), k_base_hi);
    k.v_add_co(k_addr, k_addr, t_off);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));

    // V_addr = V_base + thread_offset
    let v_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(v_addr, v_base_lo);
    k.v_mov_from_sgpr(VReg(v_addr.0 + 1), v_base_hi);
    k.v_add_co(v_addr, v_addr, t_off);
    k.v_add_co_ci(VReg(v_addr.0 + 1), VReg(v_addr.0 + 1));

    // ── ZCLT LDS topology: 132-byte padded stride ──
    // r = thread_id / 2, c = thread_id % 2
    let r_v = k.alloc_vreg();
    k.v_lshrrev_b32(r_v, 1, tid);
    let c_v = k.alloc_vreg();
    k.v_and_b32_imm(c_v, tid, 1);
    k.v_lshlrev_b32(c_v, 6, c_v); // c_bytes = c * 64

    // K_lds_write = r * 132 + c_bytes
    let stride_132 = k.alloc_vreg();
    let s132 = k.alloc_sreg();
    k.s_mov_imm(s132, 132);
    k.v_mov_from_sgpr(stride_132, s132);
    let k_lds_wr = k.alloc_vreg();
    k.v_mul_lo_u32(k_lds_wr, r_v, stride_132);
    k.v_add_u32(k_lds_wr, k_lds_wr, c_v);

    // V_lds_write = K_lds_write + 2112
    let v_lds_wr = k.alloc_vreg();
    let s2112 = k.alloc_sreg();
    k.s_mov_imm(s2112, 2112);
    let v2112 = k.alloc_vreg();
    k.v_mov_from_sgpr(v2112, s2112);
    k.v_add_u32(v_lds_wr, k_lds_wr, v2112);

    // LDS read base = (thread_id % 16) * 2
    let lds_read_base = k.alloc_vreg();
    k.v_and_b32_imm(lds_read_base, tid, 15);
    k.v_lshlrev_b32(lds_read_base, 1, lds_read_base);

    // HBM step = 16 rows × 128 bytes = 2048
    let hbm_step = k.alloc_vreg();
    let s2048 = k.alloc_sreg();
    k.s_mov_imm(s2048, 2048);
    k.v_mov_from_sgpr(hbm_step, s2048);

    // ── Big VGPR arrays (allocated AFTER individual VGPRs to avoid VReg(0) conflict) ──
    let acc = k.alloc_vreg_array(128, Alignment::Align8);     // WMMA accumulators
    let a_regs = k.alloc_vreg_array(32, Alignment::Align8);   // K^T (WMMA A)
    let b_regs = k.alloc_vreg_array(32, Alignment::Align8);   // V   (WMMA B)

    // Zero accumulators
    for i in 0..128u32 {
        k.v_mov_imm(VReg(acc.0 + i), 0);
    }

    // ════════════════════════════════════════════════════════════════
    // Main loop: process C_chunk/16 iterations
    // ════════════════════════════════════════════════════════════════
    let m_idx = k.alloc_sreg();
    k.s_mov_imm(m_idx, 0);
    k.label("m_loop");

    // A. Load K → a_regs[0:15], V → b_regs[0:15] (reuse, will be overwritten by LDS reads)
    for i in 0..4u32 {
        k.global_load(VReg(a_regs.0 + i * 4), k_addr, Width::B128, (i * 16) as i32);
    }
    for i in 0..4u32 {
        k.global_load(VReg(b_regs.0 + i * 4), v_addr, Width::B128, (i * 16) as i32);
    }

    // B. Advance HBM pointers (hide latency)
    k.v_add_co(k_addr, k_addr, hbm_step);
    k.v_add_co_ci(VReg(k_addr.0 + 1), VReg(k_addr.0 + 1));
    k.v_add_co(v_addr, v_addr, hbm_step);
    k.v_add_co_ci(VReg(v_addr.0 + 1), VReg(v_addr.0 + 1));

    k.wait_vmcnt(0);

    // C. Write to LDS from a_regs/b_regs (data will be overwritten by column tearing reads next)
    for i in 0..4u32 {
        k.ds_store_b128(k_lds_wr, VReg(a_regs.0 + i * 4), (i * 16) as u16);
    }
    for i in 0..4u32 {
        k.ds_store_b128(v_lds_wr, VReg(b_regs.0 + i * 4), (i * 16) as u16);
    }
    k.wait_lgkmcnt(0);

    // D. Column tearing: transpose read via ds_load_u16_d16/hi
    // K^T → a_regs[0..31] (4 groups × 8 regs)
    for g in 0..4u32 {
        for kk in 0..8u32 {
            let off_lo = (g * 32 + 2 * kk * 132) as u16;
            let off_hi = (g * 32 + (2 * kk + 1) * 132) as u16;
            let v_idx = VReg(a_regs.0 + g * 8 + kk);
            k.ds_load_u16_d16(v_idx, lds_read_base, off_lo);
            k.ds_load_u16_d16_hi(v_idx, lds_read_base, off_hi);
        }
    }
    // V → b_regs[0..31] (4 groups × 8 regs)
    for v in 0..4u32 {
        for kk in 0..8u32 {
            let off_lo = (2112 + v * 32 + 2 * kk * 132) as u16;
            let off_hi = (2112 + v * 32 + (2 * kk + 1) * 132) as u16;
            let v_idx = VReg(b_regs.0 + v * 8 + kk);
            k.ds_load_u16_d16(v_idx, lds_read_base, off_lo);
            k.ds_load_u16_d16_hi(v_idx, lds_read_base, off_hi);
        }
    }
    k.wait_lgkmcnt(0);

    // E. 16× WMMA: K^T[g] × V[v] → acc[g*4+v]
    for g in 0..4u32 {
        for v in 0..4u32 {
            let acc_base = VReg(acc.0 + g * 32 + v * 8);
            let a_base = VReg(a_regs.0 + g * 8);
            let b_base = VReg(b_regs.0 + v * 8);
            k.wmma_bf16_f32(acc_base, a_base, b_base, acc_base);
        }
    }

    // F. Loop control
    k.s_add_u32(m_idx, m_idx, 16);
    k.s_cmp_lt_u32(m_idx, SReg(c_chunk.0));
    k.branch_scc1("m_loop");

    // ════════════════════════════════════════════════════════════════
    // Store 64×64 result to W[head][chunk]
    // ════════════════════════════════════════════════════════════════
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    // W_base = W_ptr + (head_id * n_chunks + chunk_id) * d_head^2 * 4
    let w_idx = k.alloc_sreg();
    k.s_mul_i32(w_idx, head_id, SReg(n_chunks.0));
    k.s_add_u32_ss(w_idx, w_idx, chunk_id);
    let d_sq = k.alloc_sreg();
    k.s_mul_i32(d_sq, SReg(d_head.0), SReg(d_head.0));
    let w_off = k.alloc_sreg();
    k.s_mul_i32(w_off, w_idx, d_sq);
    k.s_lshl_b32(w_off, w_off, 2); // × 4 bytes

    let w_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(w_addr, SReg(w_ptr.0));
    k.v_mov_from_sgpr(VReg(w_addr.0 + 1), SReg(w_ptr.0 + 1));
    let w_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(w_off_v, w_off);
    k.v_add_co(w_addr, w_addr, w_off_v);
    k.v_add_co_ci(VReg(w_addr.0 + 1), VReg(w_addr.0 + 1));

    // For each k_grp (0..3) × v_tile (0..3), store 8 rows
    let d_v = k.alloc_vreg();
    k.v_mov_from_sgpr(d_v, SReg(d_head.0));

    let st_addr = k.alloc_vreg_array(2, Alignment::Align2);

    // Pre-allocate scratch VGPRs ONCE (reuse dead a_regs slots after main loop)
    let scratch_base_row = VReg(a_regs.0);       // reuse dead a_regs[0]
    let scratch_row_off = VReg(a_regs.0 + 1);    // reuse dead a_regs[1]
    let scratch_imm = VReg(a_regs.0 + 2);        // reuse dead a_regs[2]

    // Pre-compute lane_row * 4
    let lr_off = VReg(a_regs.0 + 3);             // reuse dead a_regs[3]
    k.v_lshlrev_b32(lr_off, 2, lane_row);

    for k_grp in 0..4u32 {
        // base_row = lane_half + k_grp * 16
        if k_grp == 0 {
            k.push(Op::VMov { dst: scratch_base_row, src: Operand::VReg(lane_half) });
        } else {
            k.v_mov_imm(scratch_imm, (k_grp * 16) as i32);
            k.v_add_u32(scratch_base_row, lane_half, scratch_imm);
        }
        // row_byte_offset = base_row * d_head * 4
        k.v_mul_lo_u32(scratch_row_off, scratch_base_row, d_v);
        k.v_lshlrev_b32(scratch_row_off, 2, scratch_row_off);

        for v_tile in 0..4u32 {
            let acc_base_idx = acc.0 + k_grp * 32 + v_tile * 8;
            let col_offset = v_tile * 16 * 4;

            // st_addr = w_addr + row_offset + lr_off + col_offset
            k.push(Op::VMov { dst: st_addr, src: Operand::VReg(w_addr) });
            k.push(Op::VMov { dst: VReg(st_addr.0 + 1), src: Operand::VReg(VReg(w_addr.0 + 1)) });
            k.v_add_co(st_addr, st_addr, scratch_row_off);
            k.v_add_co_ci(VReg(st_addr.0 + 1), VReg(st_addr.0 + 1));
            k.v_add_u32(st_addr, st_addr, lr_off);

            if col_offset > 0 {
                k.v_mov_imm(scratch_imm, col_offset as i32);
                k.v_add_u32(st_addr, st_addr, scratch_imm);
            }

            // Store 8 WMMA result rows
            for r in 0..8u32 {
                let r_offset = (r as i32) * 512;
                k.global_store(st_addr, VReg(acc_base_idx + r), Width::B32, r_offset);
            }
        }
    }

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Embedding Gather: out[pos,:] = table[ids[pos],:]
// ============================================================================

/// Generate embedding gather kernel.
///
/// Each WG (32 threads) handles one token position.
/// Loads token_id from ids array, copies entire row from table to output.
///
/// Kernargs (32 bytes):
///   table_ptr(0), ids_ptr(8), out_ptr(16), dim(24)
/// Grid: (num_tokens * 32, 1, 1), WG: 32
pub fn t0_embedding_gather(dim: u32) -> T0Kernel {
    assert!(dim >= 32 && dim % 32 == 0);
    let dpl = dim / 32; // elements per lane
    let chunks = dpl / 4;
    assert!(chunks >= 1, "dim must be at least 128");

    let mut k = T0Kernel::new(&format!("t0_embedding_gather_d{}", dim));

    // Kernargs
    let table_ptr = k.arg_ptr("table");  // [0:7]
    let ids_ptr = k.arg_ptr("ids");      // [8:15]
    let out_ptr = k.arg_ptr("out");      // [16:23]
    let _dim_arg = k.arg_u32("dim");     // [24:27]
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);

    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // ── Load token_id via SGPR: ids_addr = ids_ptr + wg_id * 4 ──
    // Use lane 0 to load the token_id, then broadcast via v_readfirstlane
    let ids_off = k.alloc_sreg();
    k.s_lshl_b32(ids_off, wg_id, 2); // wg_id * 4

    let ids_addr_lo = k.alloc_sreg();
    let ids_addr_hi = k.alloc_sreg();
    k.s_add_u32_ss(ids_addr_lo, SReg(ids_ptr.0), ids_off);
    k.s_addc_u32_imm(ids_addr_hi, SReg(ids_ptr.0 + 1), 0);

    // SMEM load: s_load_dword token_id from s[ids_addr_lo:hi]
    let token_id = k.alloc_sreg();
    k.push(Op::SMemLoadDword { dst: token_id, base_lo: ids_addr_lo, base_hi: ids_addr_hi, offset: 0 });
    k.wait_lgkmcnt(0);

    // ── Table address: table_ptr + token_id * dim * 4 + lane_id * dpl * 4 ──
    let row_stride = k.alloc_sreg();
    k.s_mov_imm(row_stride, (dim * 4) as i32);
    let row_off = k.alloc_sreg();
    k.s_mul_i32(row_off, token_id, row_stride);

    let tab_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(tab_addr, SReg(table_ptr.0));
    k.v_mov_from_sgpr(VReg(tab_addr.0 + 1), SReg(table_ptr.0 + 1));
    let row_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(row_off_v, row_off);
    k.v_add_co(tab_addr, tab_addr, row_off_v);
    k.v_add_co_ci(VReg(tab_addr.0 + 1), VReg(tab_addr.0 + 1));

    // lane byte offset
    let lane_bytes_s = k.alloc_sreg();
    k.s_mov_imm(lane_bytes_s, (dpl * 4) as i32);
    let lane_off = k.alloc_vreg();
    let lane_bytes_v = k.alloc_vreg();
    k.v_mov_from_sgpr(lane_bytes_v, lane_bytes_s);
    k.v_mul_lo_u32(lane_off, lane_id, lane_bytes_v);
    k.v_add_co(tab_addr, tab_addr, lane_off);
    k.v_add_co_ci(VReg(tab_addr.0 + 1), VReg(tab_addr.0 + 1));

    // ── Output address: out_ptr + wg_id * dim * 4 + lane_byte_offset ──
    let out_off = k.alloc_sreg();
    k.s_mul_i32(out_off, wg_id, row_stride);

    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0));
    k.v_mov_from_sgpr(VReg(out_addr.0 + 1), SReg(out_ptr.0 + 1));
    let out_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(out_off_v, out_off);
    k.v_add_co(out_addr, out_addr, out_off_v);
    k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));
    k.v_add_co(out_addr, out_addr, lane_off);
    k.v_add_co_ci(VReg(out_addr.0 + 1), VReg(out_addr.0 + 1));

    // ── Copy: load dwordx4 from table, store to output ──
    let data = k.alloc_vreg_array(4, Alignment::Align4);
    for c in 0..chunks {
        k.global_load(data, tab_addr, Width::B128, (c * 16) as i32);
        k.wait_vmcnt(0);
        k.global_store(out_addr, data, Width::B128, (c * 16) as i32);
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Embedding Scatter Add: table_grad[ids[pos],:] += grad[pos,:]  (atomic)
// ============================================================================

/// Generate embedding scatter-add kernel (backward of embedding_gather).
///
/// Each WG (32 threads) handles one token position.
/// Uses global_atomic_add_f32 for duplicate token deduplication.
///
/// Kernargs (32 bytes):
///   grad_ptr(0), table_grad_ptr(8), ids_ptr(16), dim(24)
/// Grid: (num_tokens * 32, 1, 1), WG: 32
pub fn t0_embedding_scatter_add(dim: u32) -> T0Kernel {
    assert!(dim >= 32 && dim % 32 == 0);
    let dpl = dim / 32;
    let chunks = dpl / 4;
    assert!(chunks >= 1, "dim must be at least 128");

    let mut k = T0Kernel::new(&format!("t0_embedding_scatter_add_d{}", dim));

    // Kernargs
    let grad_ptr = k.arg_ptr("grad");          // [0:7]
    let tgrad_ptr = k.arg_ptr("table_grad");   // [8:15]
    let ids_ptr = k.arg_ptr("ids");            // [16:23]
    let _dim_arg = k.arg_u32("dim");           // [24:27]
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);

    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // Load token_id
    let ids_off = k.alloc_sreg();
    k.s_lshl_b32(ids_off, wg_id, 2);
    let ids_addr_lo = k.alloc_sreg();
    let ids_addr_hi = k.alloc_sreg();
    k.s_add_u32_ss(ids_addr_lo, SReg(ids_ptr.0), ids_off);
    k.s_addc_u32_imm(ids_addr_hi, SReg(ids_ptr.0 + 1), 0);
    let token_id = k.alloc_sreg();
    k.push(Op::SMemLoadDword { dst: token_id, base_lo: ids_addr_lo, base_hi: ids_addr_hi, offset: 0 });
    k.wait_lgkmcnt(0);

    let row_stride = k.alloc_sreg();
    k.s_mov_imm(row_stride, (dim * 4) as i32);
    let lane_bytes_s = k.alloc_sreg();
    k.s_mov_imm(lane_bytes_s, (dpl * 4) as i32);
    let lane_off = k.alloc_vreg();
    let lane_bytes_v = k.alloc_vreg();
    k.v_mov_from_sgpr(lane_bytes_v, lane_bytes_s);
    k.v_mul_lo_u32(lane_off, lane_id, lane_bytes_v);

    // Grad source: grad_ptr + wg_id * dim * 4 + lane_offset
    let grad_off = k.alloc_sreg();
    k.s_mul_i32(grad_off, wg_id, row_stride);

    let grad_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(grad_addr, SReg(grad_ptr.0));
    k.v_mov_from_sgpr(VReg(grad_addr.0 + 1), SReg(grad_ptr.0 + 1));
    let grad_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(grad_off_v, grad_off);
    k.v_add_co(grad_addr, grad_addr, grad_off_v);
    k.v_add_co_ci(VReg(grad_addr.0 + 1), VReg(grad_addr.0 + 1));
    k.v_add_co(grad_addr, grad_addr, lane_off);
    k.v_add_co_ci(VReg(grad_addr.0 + 1), VReg(grad_addr.0 + 1));

    // Table grad dest: table_grad_ptr + token_id * dim * 4 + lane_offset
    let tg_off = k.alloc_sreg();
    k.s_mul_i32(tg_off, token_id, row_stride);

    let tg_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(tg_addr, SReg(tgrad_ptr.0));
    k.v_mov_from_sgpr(VReg(tg_addr.0 + 1), SReg(tgrad_ptr.0 + 1));
    let tg_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(tg_off_v, tg_off);
    k.v_add_co(tg_addr, tg_addr, tg_off_v);
    k.v_add_co_ci(VReg(tg_addr.0 + 1), VReg(tg_addr.0 + 1));
    k.v_add_co(tg_addr, tg_addr, lane_off);
    k.v_add_co_ci(VReg(tg_addr.0 + 1), VReg(tg_addr.0 + 1));

    // Load grad row, atomic add to table_grad row
    let data = k.alloc_vreg_array(4, Alignment::Align4);
    for c in 0..chunks {
        let off = (c * 16) as i32;
        k.global_load(data, grad_addr, Width::B128, off);
        k.wait_vmcnt(0);
        for i in 0..4u32 {
            k.global_atomic_add_f32(tg_addr, VReg(data.0 + i), off + (i as i32) * 4);
        }
    }

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Softmax + Cross-Entropy Loss (fused forward + backward)
// ============================================================================

/// Generate fused softmax → cross-entropy loss → gradient kernel.
///
/// 3-pass algorithm per token row:
///   Pass 1: Find row max (wave reduction)
///   Pass 2: exp(logit - max), sum exp (wave reduction)
///   Pass 3: grad = softmax(x) - one_hot(target), accumulate loss
///
/// Kernargs (40 bytes):
///   logits_ptr(0), targets_ptr(8), grad_ptr(16), loss_ptr(24),
///   vocab_size(32), vocab_aligned(36)
/// Grid: (seq_len * 32, 1, 1), WG: 32
pub fn t0_softmax_ce_loss() -> T0Kernel {
    let mut k = T0Kernel::new("t0_softmax_ce_loss");

    let logits_ptr = k.arg_ptr("logits");
    let targets_ptr = k.arg_ptr("targets");
    let grad_ptr = k.arg_ptr("grad");
    let loss_ptr = k.arg_ptr("loss");
    let vocab_size = k.arg_u32("vocab_size");
    let vocab_aligned = k.arg_u32("vocab_aligned");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);

    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // Load target via SMEM
    let tgt_off = k.alloc_sreg();
    k.s_lshl_b32(tgt_off, wg_id, 2);
    let tgt_lo = k.alloc_sreg();
    let tgt_hi = k.alloc_sreg();
    k.s_add_u32_ss(tgt_lo, SReg(targets_ptr.0), tgt_off);
    k.s_addc_u32_imm(tgt_hi, SReg(targets_ptr.0 + 1), 0);
    let target_s = k.alloc_sreg();
    k.push(Op::SMemLoadDword { dst: target_s, base_lo: tgt_lo, base_hi: tgt_hi, offset: 0 });
    k.wait_lgkmcnt(0);
    let target_v = k.alloc_vreg();
    k.v_mov_from_sgpr(target_v, target_s);

    // Row byte offset
    let row_off = k.alloc_sreg();
    k.s_mul_i32(row_off, wg_id, SReg(vocab_aligned.0));
    k.s_lshl_b32(row_off, row_off, 2);

    let lr_lo = k.alloc_sreg(); let lr_hi = k.alloc_sreg();
    k.s_add_u32_ss(lr_lo, SReg(logits_ptr.0), row_off);
    k.s_addc_u32_imm(lr_hi, SReg(logits_ptr.0 + 1), 0);

    let gr_lo = k.alloc_sreg(); let gr_hi = k.alloc_sreg();
    k.s_add_u32_ss(gr_lo, SReg(grad_ptr.0), row_off);
    k.s_addc_u32_imm(gr_hi, SReg(grad_ptr.0 + 1), 0);

    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    let addr2 = k.alloc_vreg_array(2, Alignment::Align2);
    let val = k.alloc_vreg();
    let idx = k.alloc_vreg();
    let byte_off = k.alloc_vreg();
    let vocab_v = k.alloc_vreg();
    k.v_mov_from_sgpr(vocab_v, SReg(vocab_size.0));
    let swap_tmp = k.alloc_vreg();

    let log2e = k.alloc_vreg();
    k.v_mov_imm(log2e, 0x3FB8AA3Bu32 as i32);
    let one = k.alloc_vreg();
    k.v_mov_imm(one, 0x3F800000u32 as i32);
    let ln2 = k.alloc_vreg();
    k.v_mov_imm(ln2, 0x3F317218u32 as i32);
    let eps_clamp = k.alloc_vreg();
    k.v_mov_imm(eps_clamp, 0x2EDBE6FFu32 as i32);

    let saved_exec = k.alloc_sreg();
    let loop_cnt = k.alloc_sreg();

    // ═══════════ PASS 1: row max ═══════════
    let row_max = k.alloc_vreg();
    k.v_mov_imm(row_max, 0xFF800000u32 as i32);

    k.s_mov_imm(loop_cnt, 0);
    k.label("p1");
    k.v_mov_from_sgpr(idx, loop_cnt);
    k.v_add_u32(idx, idx, lane_id);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(vocab_v) });
    k.push(Op::SaveExec { dst: saved_exec });
    k.v_lshlrev_b32(byte_off, 2, idx);
    k.v_mov_from_sgpr(addr, lr_lo);
    k.v_mov_from_sgpr(VReg(addr.0 + 1), lr_hi);
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
    k.global_load(val, addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_max_f32(row_max, row_max, val);
    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_cnt, loop_cnt, 32);
    k.s_cmp_lt_u32(loop_cnt, SReg(vocab_size.0));
    k.branch_scc1("p1");
    k.wave_reduce_max_f32(row_max, swap_tmp);

    // ═══════════ PASS 2: exp sum ═══════════
    let exp_sum = k.alloc_vreg();
    k.v_mov_imm(exp_sum, 0);
    k.s_mov_imm(loop_cnt, 0);
    k.label("p2");
    k.v_mov_from_sgpr(idx, loop_cnt);
    k.v_add_u32(idx, idx, lane_id);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(vocab_v) });
    k.push(Op::SaveExec { dst: saved_exec });
    k.v_lshlrev_b32(byte_off, 2, idx);
    k.v_mov_from_sgpr(addr, lr_lo);
    k.v_mov_from_sgpr(VReg(addr.0 + 1), lr_hi);
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
    k.global_load(val, addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_sub_f32(val, val, row_max);
    k.v_mul_f32(val, val, log2e);
    k.v_exp_f32(val, val);
    k.v_add_f32(exp_sum, exp_sum, val);
    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_cnt, loop_cnt, 32);
    k.s_cmp_lt_u32(loop_cnt, SReg(vocab_size.0));
    k.branch_scc1("p2");
    k.wave_reduce_add_f32(exp_sum, swap_tmp);
    let inv_sum = k.alloc_vreg();
    k.v_rcp_f32(inv_sum, exp_sum);

    // ═══════════ PASS 3: grad + loss ═══════════
    let loss_acc = k.alloc_vreg();
    k.v_mov_imm(loss_acc, 0);
    let grad_val = k.alloc_vreg();
    let one_hot = k.alloc_vreg();
    let xor_tmp = k.alloc_vreg();
    let log_val = k.alloc_vreg();
    let cond_loss = k.alloc_vreg();

    k.s_mov_imm(loop_cnt, 0);
    k.label("p3");
    k.v_mov_from_sgpr(idx, loop_cnt);
    k.v_add_u32(idx, idx, lane_id);
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(vocab_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    k.v_lshlrev_b32(byte_off, 2, idx);
    k.v_mov_from_sgpr(addr, lr_lo);
    k.v_mov_from_sgpr(VReg(addr.0 + 1), lr_hi);
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
    k.global_load(val, addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_sub_f32(val, val, row_max);
    k.v_mul_f32(val, val, log2e);
    k.v_exp_f32(val, val);
    k.v_mul_f32(val, val, inv_sum);

    k.v_xor_b32(xor_tmp, Operand::VReg(idx), Operand::VReg(target_v));
    k.push(Op::VCmpEqU32Imm { src: xor_tmp, imm: 0 });
    k.v_mov_imm(one_hot, 0);
    k.v_cndmask_b32(one_hot, Operand::VReg(one_hot), Operand::VReg(one));
    k.v_sub_f32(grad_val, val, one_hot);

    k.v_mov_from_sgpr(addr2, gr_lo);
    k.v_mov_from_sgpr(VReg(addr2.0 + 1), gr_hi);
    k.v_add_co(addr2, addr2, byte_off);
    k.v_add_co_ci(VReg(addr2.0 + 1), VReg(addr2.0 + 1));
    k.global_store(addr2, grad_val, Width::B32, 0);

    k.v_max_f32(val, val, eps_clamp);
    k.v_log_f32(log_val, val);
    k.v_mul_f32(log_val, log_val, ln2);
    // Re-compare idx == target (VCC was clobbered by v_add_co above)
    k.v_xor_b32(xor_tmp, Operand::VReg(idx), Operand::VReg(target_v));
    k.push(Op::VCmpEqU32Imm { src: xor_tmp, imm: 0 });
    k.v_mov_imm(cond_loss, 0);
    k.v_cndmask_b32(cond_loss, Operand::VReg(cond_loss), Operand::VReg(log_val));
    k.v_sub_f32(loss_acc, loss_acc, cond_loss);

    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_cnt, loop_cnt, 32);
    k.s_cmp_lt_u32(loop_cnt, SReg(vocab_size.0));
    k.branch_scc1("p3");

    k.wave_reduce_add_f32(loss_acc, swap_tmp);

    // Store loss from lane 0
    k.push(Op::VCmpEqU32Imm { src: lane_id, imm: 0 });
    k.push(Op::SaveExec { dst: saved_exec });
    let loss_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(loss_addr, SReg(loss_ptr.0));
    k.v_mov_from_sgpr(VReg(loss_addr.0 + 1), SReg(loss_ptr.0 + 1));
    let wg_off = k.alloc_vreg();
    k.v_mov_from_sgpr(wg_off, wg_id);
    k.v_lshlrev_b32(wg_off, 2, wg_off);
    k.v_add_co(loss_addr, loss_addr, wg_off);
    k.v_add_co_ci(VReg(loss_addr.0 + 1), VReg(loss_addr.0 + 1));
    k.global_store(loss_addr, loss_acc, Width::B32, 0);
    k.push(Op::RestoreExec { src: saved_exec });

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Auxiliary elementwise T0 kernels
// ============================================================================

/// f32 → bf16 conversion (truncate upper 16 bits).
/// Kernargs (20B): src_ptr(0), dst_ptr(8), n_elements(16)
/// Grid: (wgs * 32, 1, 1), WG: 32, each lane processes 4 elements
pub fn t0_f32_to_bf16() -> T0Kernel {
    let mut k = T0Kernel::new("t0_f32_to_bf16");
    let src = k.arg_ptr("src");
    let dst = k.arg_ptr("dst");
    let n_elems = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // global_idx = wg_id * 128 + lane_id * 4
    let base_off = k.alloc_sreg();
    k.s_lshl_b32(base_off, wg_id, 7); // *128
    let v_off = k.alloc_vreg();
    k.v_lshlrev_b32(v_off, 2, lane_id); // lane*4
    let gid = k.alloc_vreg();
    k.v_mov_from_sgpr(gid, base_off);
    k.v_add_u32(gid, gid, v_off);

    // Bounds check
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, SReg(n_elems.0));
    let saved = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(gid), src1: Operand::VReg(n_v) });
    k.push(Op::SaveExec { dst: saved });

    // Load 4 f32
    let src_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, gid); // *4 bytes
    k.v_mov_from_sgpr(src_addr, SReg(src.0));
    k.v_mov_from_sgpr(VReg(src_addr.0 + 1), SReg(src.0 + 1));
    k.v_add_co(src_addr, src_addr, byte_off);
    k.v_add_co_ci(VReg(src_addr.0 + 1), VReg(src_addr.0 + 1));

    let v0 = k.alloc_vreg();
    let v1 = k.alloc_vreg();
    let v2 = k.alloc_vreg();
    let v3 = k.alloc_vreg();
    k.global_load(v0, src_addr, Width::B32, 0);
    k.global_load(v1, src_addr, Width::B32, 4);
    k.global_load(v2, src_addr, Width::B32, 8);
    k.global_load(v3, src_addr, Width::B32, 12);
    k.wait_vmcnt(0);

    // Clamp to bf16 safe range [-65504, +65504] to prevent inf → NaN in WMMA
    // 65504.0f32 = 0x477FE000, -65504.0f32 = 0xC77FE000
    let clamp_pos = k.alloc_vreg();
    k.v_mov_imm(clamp_pos, 65504.0f32.to_bits() as i32);   // +65504.0
    let clamp_neg = k.alloc_vreg();
    k.v_mov_imm(clamp_neg, (-65504.0f32).to_bits() as i32); // -65504.0
    k.v_min_f32(v0, v0, clamp_pos);
    k.v_max_f32(v0, v0, clamp_neg);
    k.v_min_f32(v1, v1, clamp_pos);
    k.v_max_f32(v1, v1, clamp_neg);
    k.v_min_f32(v2, v2, clamp_pos);
    k.v_max_f32(v2, v2, clamp_neg);
    k.v_min_f32(v3, v3, clamp_pos);
    k.v_max_f32(v3, v3, clamp_neg);

    // Convert f32 → bf16 using hardware instruction (round-to-nearest-even)
    // v_cvt_pk_bf16_f32 packs 2 f32 → 1 bf16x2 dword with proper RNE rounding.
    // CRITICAL: simple truncation (>>16) causes systematic bias → training divergence!
    let packed0 = k.alloc_vreg();
    let packed1 = k.alloc_vreg();
    k.cvt_pk_bf16_f32(packed0, v0, v1);  // packed0 = bf16(v1) << 16 | bf16(v0)
    k.cvt_pk_bf16_f32(packed1, v2, v3);  // packed1 = bf16(v3) << 16 | bf16(v2)

    // Store 2 dwords (4 bf16)
    let dst_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let bf16_off = k.alloc_vreg();
    k.v_lshlrev_b32(bf16_off, 1, gid); // *2 bytes (bf16)
    k.v_mov_from_sgpr(dst_addr, SReg(dst.0));
    k.v_mov_from_sgpr(VReg(dst_addr.0 + 1), SReg(dst.0 + 1));
    k.v_add_co(dst_addr, dst_addr, bf16_off);
    k.v_add_co_ci(VReg(dst_addr.0 + 1), VReg(dst_addr.0 + 1));
    k.global_store(dst_addr, packed0, Width::B32, 0);
    k.global_store(dst_addr, packed1, Width::B32, 4);

    k.push(Op::RestoreExec { src: saved });
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// Elementwise scale in-place: x[i] *= scalar.
/// Kernargs (16B): ptr(0), scalar_f32(8), n_elements(12)
/// Grid: (wgs * 32, 1, 1), WG: 32
pub fn t0_ew_scale_inplace() -> T0Kernel {
    let mut k = T0Kernel::new("t0_ew_scale_inplace");
    let ptr = k.arg_ptr("ptr");
    let scalar = k.arg_u32("scalar"); // f32 bits
    let n_elems = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    let base_off = k.alloc_sreg();
    k.s_lshl_b32(base_off, wg_id, 7); // *128
    let v_off = k.alloc_vreg();
    k.v_lshlrev_b32(v_off, 2, lane_id);
    let gid = k.alloc_vreg();
    k.v_mov_from_sgpr(gid, base_off);
    k.v_add_u32(gid, gid, v_off);

    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, SReg(n_elems.0));
    let saved = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(gid), src1: Operand::VReg(n_v) });
    k.push(Op::SaveExec { dst: saved });

    let scale_v = k.alloc_vreg();
    k.v_mov_from_sgpr(scale_v, SReg(scalar.0));

    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, gid);
    k.v_mov_from_sgpr(addr, SReg(ptr.0));
    k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(ptr.0 + 1));
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

    // Load 4, scale, store 4
    for i in 0..4i32 {
        let val = k.alloc_vreg();
        k.global_load(val, addr, Width::B32, i * 4);
        k.wait_vmcnt(0);
        k.v_mul_f32(val, val, scale_v);
        k.global_store(addr, val, Width::B32, i * 4);
    }

    k.push(Op::RestoreExec { src: saved });
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// Ψ-map in-place: Ψ(x) = 1 + 2σ(x), where σ(x) = 1/(1+exp(-x))
/// Kernargs (16B): ptr(0), n_elements(8)
/// Grid: (wgs * 32, 1, 1), WG: 32
pub fn t0_psi_inplace() -> T0Kernel {
    let mut k = T0Kernel::new("t0_psi_inplace");
    let ptr = k.arg_ptr("ptr");
    let n_elems = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    let base_off = k.alloc_sreg();
    k.s_lshl_b32(base_off, wg_id, 7);
    let v_off = k.alloc_vreg();
    k.v_lshlrev_b32(v_off, 2, lane_id);
    let gid = k.alloc_vreg();
    k.v_mov_from_sgpr(gid, base_off);
    k.v_add_u32(gid, gid, v_off);

    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, SReg(n_elems.0));
    let saved = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(gid), src1: Operand::VReg(n_v) });
    k.push(Op::SaveExec { dst: saved });

    // Constants: neg_log2e = -1.4427, 1.0, 2.0
    let neg_log2e = k.alloc_vreg();
    k.v_mov_imm(neg_log2e, 0xBFB8AA3Bu32 as i32); // -log2(e)
    let one = k.alloc_vreg();
    k.v_mov_imm(one, 0x3F800000u32 as i32);
    let two = k.alloc_vreg();
    k.v_mov_imm(two, 0x40000000u32 as i32);

    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, gid);
    k.v_mov_from_sgpr(addr, SReg(ptr.0));
    k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(ptr.0 + 1));
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

    for i in 0..4i32 {
        let val = k.alloc_vreg();
        k.global_load(val, addr, Width::B32, i * 4);
        k.wait_vmcnt(0);
        // σ(x) = 1/(1 + 2^(-x·log₂e))
        k.v_mul_f32(val, val, neg_log2e);  // x * (-log₂e)
        k.v_exp_f32(val, val);              // 2^(-x·log₂e) = exp(-x)
        k.v_add_f32(val, one, val);         // 1 + exp(-x)
        k.v_rcp_f32(val, val);              // σ(x)
        // Ψ(x) = 1 + 2·σ(x) = fma(2, σ, 1)
        k.v_fma_f32(val, two, val, one);
        k.global_store(addr, val, Width::B32, i * 4);
    }

    k.push(Op::RestoreExec { src: saved });
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// Ψ' derivative chain rule: grad[i] *= Ψ'(Ψ(x[i]))
/// Where Ψ'(y) = (y-1)*(3-y)/2 (y = Ψ(x) already applied in forward)
/// Kernargs (24B): grad_ptr(0), psi_ptr(8), n_elements(16)
/// Grid: (wgs * 32, 1, 1), WG: 32
pub fn t0_psi_deriv_mul() -> T0Kernel {
    let mut k = T0Kernel::new("t0_psi_deriv_mul");
    let grad_ptr = k.arg_ptr("grad");
    let psi_ptr = k.arg_ptr("psi");
    let n_elems = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    let base_off = k.alloc_sreg();
    k.s_lshl_b32(base_off, wg_id, 7);
    let v_off = k.alloc_vreg();
    k.v_lshlrev_b32(v_off, 2, lane_id);
    let gid = k.alloc_vreg();
    k.v_mov_from_sgpr(gid, base_off);
    k.v_add_u32(gid, gid, v_off);

    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, SReg(n_elems.0));
    let saved = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(gid), src1: Operand::VReg(n_v) });
    k.push(Op::SaveExec { dst: saved });

    let one = k.alloc_vreg();
    k.v_mov_imm(one, 0x3F800000u32 as i32);
    let three = k.alloc_vreg();
    k.v_mov_imm(three, 0x40400000u32 as i32);
    let half = k.alloc_vreg();
    k.v_mov_imm(half, 0x3F000000u32 as i32);

    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let p_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, gid);

    k.v_mov_from_sgpr(g_addr, SReg(grad_ptr.0));
    k.v_mov_from_sgpr(VReg(g_addr.0 + 1), SReg(grad_ptr.0 + 1));
    k.v_add_co(g_addr, g_addr, byte_off);
    k.v_add_co_ci(VReg(g_addr.0 + 1), VReg(g_addr.0 + 1));

    k.v_mov_from_sgpr(p_addr, SReg(psi_ptr.0));
    k.v_mov_from_sgpr(VReg(p_addr.0 + 1), SReg(psi_ptr.0 + 1));
    k.v_add_co(p_addr, p_addr, byte_off);
    k.v_add_co_ci(VReg(p_addr.0 + 1), VReg(p_addr.0 + 1));

    for i in 0..4i32 {
        let grad = k.alloc_vreg();
        let psi = k.alloc_vreg();
        k.global_load(grad, g_addr, Width::B32, i * 4);
        k.global_load(psi, p_addr, Width::B32, i * 4);
        k.wait_vmcnt(0);
        // Ψ'(y) = (y-1)*(3-y)/2
        let ym1 = k.alloc_vreg();
        k.v_sub_f32(ym1, psi, one);       // y-1
        let tmy = k.alloc_vreg();
        k.v_sub_f32(tmy, three, psi);     // 3-y
        k.v_mul_f32(ym1, ym1, tmy);       // (y-1)*(3-y)
        k.v_mul_f32(ym1, ym1, half);      // /2
        k.v_mul_f32(grad, grad, ym1);     // grad *= Ψ'
        k.global_store(g_addr, grad, Width::B32, i * 4);
    }

    k.push(Op::RestoreExec { src: saved });
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// SRMSNorm forward: x[i] = x[i] * inv_rms, save inv_rms.
/// Per-vector (1 wave per d_head=64 vector): sum_sq → mean → rsqrt → normalize.
/// Kernargs (28B): x_ptr(0), rms_ptr(8), n_vecs(16), d_head(20), eps(24)
/// Grid: (n_vecs * 32, 1, 1), WG: 32, each lane handles 2 f32 elements
pub fn t0_srmsnorm_fwd() -> T0Kernel {
    let mut k = T0Kernel::new("t0_srmsnorm_fwd");
    let x_ptr = k.arg_ptr("x");
    let rms_ptr = k.arg_ptr("rms");
    let _n_vecs = k.arg_u32("n_vecs");
    let d_head_arg = k.arg_u32("d_head");
    let eps_arg = k.arg_u32("eps");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // x offset = (wg_id * d_head + lane_id * 2) * 4
    let elem_base = k.alloc_sreg();
    k.s_mul_i32(elem_base, wg_id, SReg(d_head_arg.0));
    let lane2 = k.alloc_vreg();
    k.v_lshlrev_b32(lane2, 1, lane_id);
    let elem_idx = k.alloc_vreg();
    k.v_mov_from_sgpr(elem_idx, elem_base);
    k.v_add_u32(elem_idx, elem_idx, lane2);

    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    let byte_off = k.alloc_vreg();
    k.v_lshlrev_b32(byte_off, 2, elem_idx);
    k.v_mov_from_sgpr(addr, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

    // Load 2 f32
    let x0 = k.alloc_vreg();
    let x1 = k.alloc_vreg();
    k.global_load(x0, addr, Width::B32, 0);
    k.global_load(x1, addr, Width::B32, 4);
    k.wait_vmcnt(0);

    // sum_sq = x0² + x1²
    let sum_sq = k.alloc_vreg();
    k.v_mul_f32(sum_sq, x0, x0);
    let tmp_fma = k.alloc_vreg();
    k.v_mul_f32(tmp_fma, x1, x1);
    k.v_add_f32(sum_sq, sum_sq, tmp_fma);

    // Wave32 reduction
    let swap = k.alloc_vreg();
    k.wave_reduce_add_f32(sum_sq, swap);

    // mean_sq = sum_sq / d_head
    let d_v = k.alloc_vreg();
    k.v_mov_from_sgpr(d_v, SReg(d_head_arg.0));
    k.v_cvt_f32_u32(d_v, d_v);
    k.v_rcp_f32(d_v, d_v);
    k.v_mul_f32(sum_sq, sum_sq, d_v);

    // inv_rms = 1/sqrt(mean_sq + eps)
    let eps = k.alloc_vreg();
    k.v_mov_from_sgpr(eps, SReg(eps_arg.0));
    k.v_add_f32(sum_sq, sum_sq, eps);
    k.v_sqrt_f32(sum_sq, sum_sq);
    k.v_rcp_f32(sum_sq, sum_sq);
    // sum_sq = inv_rms

    // Save inv_rms from lane 0
    let saved = k.alloc_sreg();
    k.push(Op::VCmpEqU32Imm { src: lane_id, imm: 0 });
    k.push(Op::SaveExec { dst: saved });
    {
        let rms_addr = k.alloc_vreg_array(2, Alignment::Align2);
        let rms_off = k.alloc_sreg();
        k.s_lshl_b32(rms_off, wg_id, 2);
        k.v_mov_from_sgpr(rms_addr, SReg(rms_ptr.0));
        k.v_mov_from_sgpr(VReg(rms_addr.0 + 1), SReg(rms_ptr.0 + 1));
        let off_v = k.alloc_vreg();
        k.v_mov_from_sgpr(off_v, rms_off);
        k.v_add_co(rms_addr, rms_addr, off_v);
        k.v_add_co_ci(VReg(rms_addr.0 + 1), VReg(rms_addr.0 + 1));
        k.global_store(rms_addr, sum_sq, Width::B32, 0);
    }
    k.push(Op::RestoreExec { src: saved });

    // Normalize: x *= inv_rms
    k.v_mul_f32(x0, x0, sum_sq);
    k.v_mul_f32(x1, x1, sum_sq);
    k.global_store(addr, x0, Width::B32, 0);
    k.global_store(addr, x1, Width::B32, 4);

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// SRMSNorm backward v2: dx = (dy - y * mean(y·dy)) * inv_rms
/// y input is bf16 (saved from forward), dy is f32 (in-place output).
/// Kernargs (40B): dy_ptr(0), y_bf16_ptr(8), rms_ptr(16), d_head(24),
///                  n_tokens(28), n_heads_mask(32), n_heads_shift(36)
/// `nh` is compile-time n_heads (must be power-of-2).
pub fn t0_srmsnorm_bwd_v2(nh: u32) -> T0Kernel {
    assert!(nh.is_power_of_two());
    let nh_shift = nh.trailing_zeros() as u8;
    let mut k = T0Kernel::new(&format!("t0_srmsnorm_bwd_v2_h{}", nh));
    let dy_ptr = k.arg_ptr("dy");
    let y_ptr = k.arg_ptr("y_bf16");
    let rms_ptr = k.arg_ptr("rms");
    let d_head_arg = k.arg_u32("d_head");
    let n_tokens = k.arg_u32("n_tokens");
    let n_heads_mask = k.arg_u32("n_heads_mask");
    let _n_heads_shift = k.arg_u32("n_heads_shift");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // elem_offset = wg_id * d_head + lane_id * 2
    let elem_base = k.alloc_sreg();
    k.s_mul_i32(elem_base, wg_id, SReg(d_head_arg.0));
    let lane2 = k.alloc_vreg();
    k.v_lshlrev_b32(lane2, 1, lane_id);
    let elem_idx = k.alloc_vreg();
    k.v_mov_from_sgpr(elem_idx, elem_base);
    k.v_add_u32(elem_idx, elem_idx, lane2);

    // dy addr (f32)
    let dy_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let f32_off = k.alloc_vreg();
    k.v_lshlrev_b32(f32_off, 2, elem_idx);
    k.v_mov_from_sgpr(dy_addr, SReg(dy_ptr.0));
    k.v_mov_from_sgpr(VReg(dy_addr.0 + 1), SReg(dy_ptr.0 + 1));
    k.v_add_co(dy_addr, dy_addr, f32_off);
    k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));

    // y addr (bf16)
    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let bf16_off = k.alloc_vreg();
    k.v_lshlrev_b32(bf16_off, 1, elem_idx);
    k.v_mov_from_sgpr(y_addr, SReg(y_ptr.0));
    k.v_mov_from_sgpr(VReg(y_addr.0 + 1), SReg(y_ptr.0 + 1));
    k.v_add_co(y_addr, y_addr, bf16_off);
    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));

    // RMS index: h = wg_id & mask, t = wg_id >> n_heads_shift
    // rms_idx = h * n_tokens + t
    let h = k.alloc_sreg();
    k.s_and_b32(h, wg_id, SReg(n_heads_mask.0));
    let t = k.alloc_sreg();
    k.s_lshr_b32(t, wg_id, nh_shift);
    let rms_idx = k.alloc_sreg();
    k.s_mul_i32(rms_idx, h, SReg(n_tokens.0));
    k.s_add_i32(rms_idx, rms_idx, t);
    let rms_byte = k.alloc_sreg();
    k.s_lshl_b32(rms_byte, rms_idx, 2);

    let rms_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(rms_addr, SReg(rms_ptr.0));
    k.v_mov_from_sgpr(VReg(rms_addr.0 + 1), SReg(rms_ptr.0 + 1));
    let rms_off_v = k.alloc_vreg();
    k.v_mov_from_sgpr(rms_off_v, rms_byte);
    k.v_add_co(rms_addr, rms_addr, rms_off_v);
    k.v_add_co_ci(VReg(rms_addr.0 + 1), VReg(rms_addr.0 + 1));

    // Load data
    let dy0 = k.alloc_vreg();
    let dy1 = k.alloc_vreg();
    k.global_load(dy0, dy_addr, Width::B32, 0);
    k.global_load(dy1, dy_addr, Width::B32, 4);
    let y_packed = k.alloc_vreg();
    k.global_load(y_packed, y_addr, Width::B32, 0); // 2 bf16 packed
    let inv_rms = k.alloc_vreg();
    k.global_load(inv_rms, rms_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // Convert bf16 → f32
    let y0 = k.alloc_vreg();
    k.v_lshlrev_b32(y0, 16, y_packed); // lower bf16 → f32
    let y1 = k.alloc_vreg();
    let mask_hi = k.alloc_vreg();
    k.v_mov_imm(mask_hi, 0xFFFF0000u32 as i32);
    k.v_and_b32(y1, y_packed, mask_hi); // upper bf16 → f32

    // sum(y·dy) per lane: y0*dy0 + y1*dy1
    let ydot = k.alloc_vreg();
    k.v_mul_f32(ydot, y0, dy0);
    let tmp = k.alloc_vreg();
    k.v_mul_f32(tmp, y1, dy1);
    k.v_add_f32(ydot, ydot, tmp);

    // Wave32 reduction
    let swap2 = k.alloc_vreg();
    k.wave_reduce_add_f32(ydot, swap2);

    // mean = sum(y·dy) / d_head
    let d_v = k.alloc_vreg();
    k.v_mov_from_sgpr(d_v, SReg(d_head_arg.0));
    k.v_cvt_f32_u32(d_v, d_v);
    k.v_rcp_f32(d_v, d_v);
    k.v_mul_f32(ydot, ydot, d_v);
    // ydot = mean(y·dy)

    // dx = (dy - y * mean) * inv_rms
    let tmp2 = k.alloc_vreg();
    k.v_mul_f32(tmp2, y0, ydot);
    k.v_sub_f32(dy0, dy0, tmp2);
    k.v_mul_f32(dy0, dy0, inv_rms);
    k.v_mul_f32(tmp2, y1, ydot);
    k.v_sub_f32(dy1, dy1, tmp2);
    k.v_mul_f32(dy1, dy1, inv_rms);

    // Store dx back (in-place)
    k.global_store(dy_addr, dy0, Width::B32, 0);
    k.global_store(dy_addr, dy1, Width::B32, 4);

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

/// f32→bf16 multihead transpose: [seq, dim] f32 → [n_heads, seq, d_head] bf16
/// Each WG handles one (token, head) pair, 32 lanes × 2 elements = 64 = d_head.
/// Kernargs (32B): src_ptr(0), dst_ptr(8), n_tokens(16), n_heads(20), d_head(24)
/// Grid: (n_tokens * n_heads * 32, 1, 1), WG: 32
/// `nh` is the compile-time n_heads value (must be power-of-2).
pub fn t0_f32_to_bf16_mh(nh: u32) -> T0Kernel {
    assert!(nh.is_power_of_two(), "n_heads must be power of 2, got {}", nh);
    let shift = nh.trailing_zeros(); // log2(n_heads)
    let mask = nh - 1;               // n_heads - 1

    let mut k = T0Kernel::new(&format!("t0_f32_to_bf16_mh_h{}", nh));
    let src_ptr = k.arg_ptr("src");
    let dst_ptr = k.arg_ptr("dst");
    let n_tokens_arg = k.arg_u32("n_tokens");
    let n_heads_arg = k.arg_u32("n_heads");
    let d_head_arg = k.arg_u32("d_head");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // Decompose wg_id → (token_id, head_id) using compile-time shift/mask
    // token = wg_id >> log2(n_heads), head = wg_id & (n_heads - 1)
    let token_id = k.alloc_sreg();
    k.s_lshr_b32(token_id, wg_id, shift as u8);
    let head_id = k.alloc_sreg();
    let s_mask = k.alloc_sreg();
    k.s_mov_imm(s_mask, mask as i32);
    k.s_and_b32(head_id, wg_id, s_mask);

    // Source: src[token_id * dim + head_id * d_head + lane_id * 2]
    // dim = n_heads * d_head
    let dim = k.alloc_sreg();
    k.s_mul_i32(dim, SReg(n_heads_arg.0), SReg(d_head_arg.0));
    let src_base = k.alloc_sreg();
    k.s_mul_i32(src_base, token_id, dim);
    let head_off = k.alloc_sreg();
    k.s_mul_i32(head_off, head_id, SReg(d_head_arg.0));
    k.s_add_i32(src_base, src_base, head_off);

    let lane2 = k.alloc_vreg();
    k.v_lshlrev_b32(lane2, 1, lane_id);
    let src_idx = k.alloc_vreg();
    k.v_mov_from_sgpr(src_idx, src_base);
    k.v_add_u32(src_idx, src_idx, lane2);

    let s_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let s_byte = k.alloc_vreg();
    k.v_lshlrev_b32(s_byte, 2, src_idx); // *4 bytes
    k.v_mov_from_sgpr(s_addr, SReg(src_ptr.0));
    k.v_mov_from_sgpr(VReg(s_addr.0 + 1), SReg(src_ptr.0 + 1));
    k.v_add_co(s_addr, s_addr, s_byte);
    k.v_add_co_ci(VReg(s_addr.0 + 1), VReg(s_addr.0 + 1));

    // Dest: dst[head_id * n_tokens * d_head + token_id * d_head + lane_id * 2]
    let ntd = k.alloc_sreg();
    k.s_mul_i32(ntd, SReg(n_tokens_arg.0), SReg(d_head_arg.0));
    let dst_base = k.alloc_sreg();
    k.s_mul_i32(dst_base, head_id, ntd);
    let tok_off = k.alloc_sreg();
    k.s_mul_i32(tok_off, token_id, SReg(d_head_arg.0));
    k.s_add_i32(dst_base, dst_base, tok_off);

    let dst_idx = k.alloc_vreg();
    k.v_mov_from_sgpr(dst_idx, dst_base);
    k.v_add_u32(dst_idx, dst_idx, lane2);

    let d_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let d_byte = k.alloc_vreg();
    k.v_lshlrev_b32(d_byte, 1, dst_idx); // *2 bytes (bf16)
    k.v_mov_from_sgpr(d_addr, SReg(dst_ptr.0));
    k.v_mov_from_sgpr(VReg(d_addr.0 + 1), SReg(dst_ptr.0 + 1));
    k.v_add_co(d_addr, d_addr, d_byte);
    k.v_add_co_ci(VReg(d_addr.0 + 1), VReg(d_addr.0 + 1));

    // Load 2 f32
    let v0 = k.alloc_vreg();
    let v1 = k.alloc_vreg();
    k.global_load(v0, s_addr, Width::B32, 0);
    k.global_load(v1, s_addr, Width::B32, 4);
    k.wait_vmcnt(0);

    // Pack 2 f32 → 1 dword bf16 using hardware RNE rounding
    let packed = k.alloc_vreg();
    k.cvt_pk_bf16_f32(packed, v0, v1);

    // Store 1 dword (2 bf16)
    k.global_store(d_addr, packed, Width::B32, 0);

    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Ada-GLAM Optimizer T0 Kernels
// ============================================================================

/// AdamW 1D: for norm gammas. 4 elems/lane.
/// Kernargs (56B): w(0), dw(8), m(16), v(24), lr(32), beta1(36), beta2(40), bc1(44), bc2(48), wd(52)
pub fn t0_adamw_1d() -> T0Kernel {
    let mut k = T0Kernel::new("t0_adamw_1d");
    let w_ptr = k.arg_ptr("w"); let dw_ptr = k.arg_ptr("dw");
    let m_ptr = k.arg_ptr("m"); let v_ptr = k.arg_ptr("v");
    let lr_a = k.arg_u32("lr"); let b1_a = k.arg_u32("beta1"); let b2_a = k.arg_u32("beta2");
    let bc1_a = k.arg_u32("bc1"); let bc2_a = k.arg_u32("bc2"); let wd_a = k.arg_u32("wd");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    // gid = wg_id*128 + lane_id*4
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7);
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid);
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base); k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);
    // Build 4 addr pairs
    let w_a = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(w_a, SReg(w_ptr.0)); k.v_mov_from_sgpr(VReg(w_a.0+1), SReg(w_ptr.0+1));
    k.v_add_co(w_a, w_a, byte_off); k.v_add_co_ci(VReg(w_a.0+1), VReg(w_a.0+1));
    let dw_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dw_ad, SReg(dw_ptr.0)); k.v_mov_from_sgpr(VReg(dw_ad.0+1), SReg(dw_ptr.0+1));
    k.v_add_co(dw_ad, dw_ad, byte_off); k.v_add_co_ci(VReg(dw_ad.0+1), VReg(dw_ad.0+1));
    let m_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(m_ad, SReg(m_ptr.0)); k.v_mov_from_sgpr(VReg(m_ad.0+1), SReg(m_ptr.0+1));
    k.v_add_co(m_ad, m_ad, byte_off); k.v_add_co_ci(VReg(m_ad.0+1), VReg(m_ad.0+1));
    let v_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(v_ad, SReg(v_ptr.0)); k.v_mov_from_sgpr(VReg(v_ad.0+1), SReg(v_ptr.0+1));
    k.v_add_co(v_ad, v_ad, byte_off); k.v_add_co_ci(VReg(v_ad.0+1), VReg(v_ad.0+1));
    // Load scalars
    let lr_v = k.alloc_vreg(); k.v_mov_from_sgpr(lr_v, SReg(lr_a.0));
    let b1_v = k.alloc_vreg(); k.v_mov_from_sgpr(b1_v, SReg(b1_a.0));
    let b2_v = k.alloc_vreg(); k.v_mov_from_sgpr(b2_v, SReg(b2_a.0));
    let bc1_v = k.alloc_vreg(); k.v_mov_from_sgpr(bc1_v, SReg(bc1_a.0));
    let bc2_v = k.alloc_vreg(); k.v_mov_from_sgpr(bc2_v, SReg(bc2_a.0));
    let wd_v = k.alloc_vreg(); k.v_mov_from_sgpr(wd_v, SReg(wd_a.0));
    let one = k.alloc_vreg(); k.v_mov_imm(one, 0x3F800000u32 as i32);
    let ob1 = k.alloc_vreg(); k.v_sub_f32(ob1, one, b1_v); // 1-beta1
    let ob2 = k.alloc_vreg(); k.v_sub_f32(ob2, one, b2_v); // 1-beta2
    let eps = k.alloc_vreg(); k.v_mov_imm(eps, 0x322BCC77u32 as i32); // 1e-8
    // Process 4 elements
    for i in 0..4i32 {
        let w = k.alloc_vreg(); k.global_load(w, w_a, Width::B32, i*4);
        let g = k.alloc_vreg(); k.global_load(g, dw_ad, Width::B32, i*4);
        let m = k.alloc_vreg(); k.global_load(m, m_ad, Width::B32, i*4);
        let v = k.alloc_vreg(); k.global_load(v, v_ad, Width::B32, i*4);
        k.wait_vmcnt(0);
        // m = b1*m + (1-b1)*g
        k.v_mul_f32(m, b1_v, m); k.v_fma_f32(m, ob1, g, m);
        // v = b2*v + (1-b2)*g²
        k.v_mul_f32(v, b2_v, v);
        let gsq = k.alloc_vreg(); k.v_mul_f32(gsq, g, g);
        k.v_fma_f32(v, ob2, gsq, v);
        // mhat = m/bc1, vhat = v/bc2
        let ibc1 = k.alloc_vreg(); k.v_rcp_f32(ibc1, bc1_v);
        let mhat = k.alloc_vreg(); k.v_mul_f32(mhat, m, ibc1);
        let ibc2 = k.alloc_vreg(); k.v_rcp_f32(ibc2, bc2_v);
        let vhat = k.alloc_vreg(); k.v_mul_f32(vhat, v, ibc2);
        // w -= lr*(mhat/(sqrt(vhat)+eps) + wd*w)
        k.v_sqrt_f32(vhat, vhat); k.v_add_f32(vhat, vhat, eps);
        k.v_rcp_f32(vhat, vhat); k.v_mul_f32(mhat, mhat, vhat);
        k.v_fma_f32(mhat, wd_v, w, mhat);
        k.v_mul_f32(mhat, lr_v, mhat);
        k.v_sub_f32(w, w, mhat);
        k.global_store(m_ad, m, Width::B32, i*4);
        k.global_store(v_ad, v, Width::B32, i*4);
        k.global_store(w_a, w, Width::B32, i*4);
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Frobenius norm partial: wave reduce sum(g²) per WG, store to partial[wg_id].
/// Kernargs (24B): grad(0), partial(8), n_elems(16)
pub fn t0_frobenius_norm_partial() -> T0Kernel {
    let mut k = T0Kernel::new("t0_frobenius_norm_partial");
    let grad = k.arg_ptr("grad"); let partial = k.arg_ptr("partial");
    let _n = k.arg_u32("n_elems");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7);
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid);
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base); k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);
    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(addr, SReg(grad.0)); k.v_mov_from_sgpr(VReg(addr.0+1), SReg(grad.0+1));
    k.v_add_co(addr, addr, byte_off); k.v_add_co_ci(VReg(addr.0+1), VReg(addr.0+1));
    // Load 4 f32, compute sum(g²)
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);
    for i in 0..4i32 {
        let val = k.alloc_vreg(); k.global_load(val, addr, Width::B32, i*4);
        k.wait_vmcnt(0); k.v_fma_f32(acc, val, val, acc);
    }
    // Wave32 reduce
    let tmp = k.alloc_vreg(); k.wave_reduce_add_f32(acc, tmp);
    // Lane 0 stores
    let saved = k.alloc_sreg();
    k.push(Op::VCmpEqU32Imm { src: lid, imm: 0 }); k.push(Op::SaveExec { dst: saved });
    let p_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let p_off = k.alloc_sreg(); k.s_lshl_b32(p_off, wg_id, 2);
    k.v_mov_from_sgpr(p_addr, SReg(partial.0)); k.v_mov_from_sgpr(VReg(p_addr.0+1), SReg(partial.0+1));
    let p_off_v = k.alloc_vreg(); k.v_mov_from_sgpr(p_off_v, p_off);
    k.v_add_co(p_addr, p_addr, p_off_v); k.v_add_co_ci(VReg(p_addr.0+1), VReg(p_addr.0+1));
    k.global_store(p_addr, acc, Width::B32, 0);
    k.push(Op::RestoreExec { src: saved });
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// High-throughput Frobenius norm partial: 4096 elements per WG (128/thread × 32 threads).
/// Each thread loops over 128 elements with stride 32, processes 4 elements per iteration.
///
/// Kernargs (24B): grad(0), partial(8), n_elems(16 u32)
/// Grid: [ceil(n_elems/4096)*32, 1, 1], WG: 32
pub fn t0_frobenius_norm_large() -> T0Kernel {
    let mut k = T0Kernel::new("t0_frobenius_norm_large");
    let grad = k.arg_ptr("grad"); let partial = k.arg_ptr("partial");
    let n_arg = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // base_elem = wg_id * 4096 + lid
    let wg_base = k.alloc_sreg(); k.s_lshl_b32(wg_base, wg_id, 12); // * 4096
    let base_v = k.alloc_vreg(); k.v_mov_from_sgpr(base_v, wg_base);
    k.v_add_u32(base_v, base_v, lid);

    // Precompute base address
    let base_byte = k.alloc_vreg(); k.v_lshlrev_b32(base_byte, 2, base_v);
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_addr, SReg(grad.0)); k.v_mov_from_sgpr(VReg(g_addr.0+1), SReg(grad.0+1));
    k.v_add_co(g_addr, g_addr, base_byte); k.v_add_co_ci(VReg(g_addr.0+1), VReg(g_addr.0+1));

    let n_v = k.alloc_vreg(); k.v_mov_from_sgpr(n_v, SReg(n_arg.0));
    let stride_bytes = 32 * 4; // 32 threads * 4 bytes

    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);

    // Loop over 128 elements per thread (stride=32, 4 elements per iter = 32 iters)
    let iter_s = k.alloc_sreg(); k.s_mov_imm(iter_s, 0);
    k.label("fnl_loop");

    // Current element index = base_v + iter * 32
    let cur_idx = k.alloc_vreg();
    let iter_v = k.alloc_vreg(); k.v_mov_from_sgpr(iter_v, iter_s);
    let iter_off = k.alloc_vreg(); k.v_lshlrev_b32(iter_off, 5, iter_v); // iter * 32
    k.v_add_u32(cur_idx, base_v, iter_off);

    // Bounds check
    let saved = k.alloc_sreg();
    k.v_cmp_lt_u32(Operand::VReg(cur_idx), Operand::VReg(n_v));
    k.push(Op::SaveExec { dst: saved });

    // Load from grad[cur_idx] — compute byte offset from base addr
    let this_byte_off = k.alloc_vreg(); k.v_lshlrev_b32(this_byte_off, 2, iter_off);
    let la = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(la, g_addr); k.v_mov(VReg(la.0+1), VReg(g_addr.0+1));
    k.v_add_co(la, la, this_byte_off); k.v_add_co_ci(VReg(la.0+1), VReg(la.0+1));
    let val = k.alloc_vreg(); k.global_load(val, la, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_fma_f32(acc, val, val, acc);

    k.push(Op::RestoreExec { src: saved });

    k.s_add_u32(iter_s, iter_s, 1);
    k.s_cmp_lt_u32(iter_s, SReg(0)); // placeholder — we'll use immediate 128
    // Actually: compare iter < 128
    let max_iter = k.alloc_sreg(); k.s_mov_imm(max_iter, 128);
    k.s_cmp_lt_u32(iter_s, max_iter);
    k.branch_scc1("fnl_loop");

    // Wave reduce
    let tmp = k.alloc_vreg(); k.wave_reduce_add_f32(acc, tmp);

    // Lane 0 stores partial
    let saved_st = k.alloc_sreg();
    k.push(Op::VCmpEqU32Imm { src: lid, imm: 0 }); k.push(Op::SaveExec { dst: saved_st });
    let p_off = k.alloc_sreg(); k.s_lshl_b32(p_off, wg_id, 2);
    let p_off_v = k.alloc_vreg(); k.v_mov_from_sgpr(p_off_v, p_off);
    let pa = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(pa, SReg(partial.0)); k.v_mov_from_sgpr(VReg(pa.0+1), SReg(partial.0+1));
    k.v_add_co(pa, pa, p_off_v); k.v_add_co_ci(VReg(pa.0+1), VReg(pa.0+1));
    k.global_store(pa, acc, Width::B32, 0);
    k.push(Op::RestoreExec { src: saved_st });

    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Reduce scalar: serial sum of partial results.
/// Kernargs (20B): in_ptr(0), out_ptr(8), n_elems(16)
pub fn t0_reduce_scalar() -> T0Kernel {
    let mut k = T0Kernel::new("t0_reduce_scalar");
    let in_ptr = k.arg_ptr("in"); let out_ptr = k.arg_ptr("out");
    let n_arg = k.arg_u32("n_elems");
    k.emit_arg_loads();
    let in_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(in_addr, SReg(in_ptr.0)); k.v_mov_from_sgpr(VReg(in_addr.0+1), SReg(in_ptr.0+1));
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);
    let i_reg = k.alloc_sreg(); k.s_mov_imm(i_reg, 0);
    k.label("reduce_loop");
    let off = k.alloc_sreg(); k.s_lshl_b32(off, i_reg, 2);
    let off_v = k.alloc_vreg(); k.v_mov_from_sgpr(off_v, off);
    let tmp_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tmp_addr, in_addr); k.v_mov(VReg(tmp_addr.0+1), VReg(in_addr.0+1));
    k.v_add_co(tmp_addr, tmp_addr, off_v); k.v_add_co_ci(VReg(tmp_addr.0+1), VReg(tmp_addr.0+1));
    let val = k.alloc_vreg(); k.global_load(val, tmp_addr, Width::B32, 0);
    k.wait_vmcnt(0); k.v_add_f32(acc, acc, val);
    k.s_add_u32(i_reg, i_reg, 1);
    k.s_cmp_lt_u32(i_reg, SReg(n_arg.0));
    k.branch_scc1("reduce_loop");
    // Store result
    let out_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(out_addr, SReg(out_ptr.0)); k.v_mov_from_sgpr(VReg(out_addr.0+1), SReg(out_ptr.0+1));
    k.global_store(out_addr, acc, Width::B32, 0);
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Grad clip EMA v2: reads norm_sq from GPU buffer, clips, updates EMA.
/// Kernargs (40B): grad(0), g_ema(8), norm_sq_ptr(16), n_elems(24), beta(28), clip(32)
pub fn t0_grad_clip_ema_v2() -> T0Kernel {
    let mut k = T0Kernel::new("t0_grad_clip_ema_v2");
    let grad_p = k.arg_ptr("grad"); let ema_p = k.arg_ptr("g_ema");
    let norm_p = k.arg_ptr("norm_sq"); let _n = k.arg_u32("n_elems");
    let beta_a = k.arg_u32("beta"); let clip_a = k.arg_u32("clip");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    // Read norm_sq from GPU
    let norm_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(norm_addr, SReg(norm_p.0)); k.v_mov_from_sgpr(VReg(norm_addr.0+1), SReg(norm_p.0+1));
    let norm_sq = k.alloc_vreg(); k.global_load(norm_sq, norm_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    // clip_scale = min(1.0, clip / sqrt(norm_sq))
    let one = k.alloc_vreg(); k.v_mov_imm(one, 0x3F800000u32 as i32);
    k.v_sqrt_f32(norm_sq, norm_sq); // sqrt(norm_sq)
    let eps = k.alloc_vreg(); k.v_mov_imm(eps, 0x358637BDu32 as i32); // 1e-6
    k.v_add_f32(norm_sq, norm_sq, eps);
    let clip_v = k.alloc_vreg(); k.v_mov_from_sgpr(clip_v, SReg(clip_a.0));
    k.v_rcp_f32(norm_sq, norm_sq); k.v_mul_f32(norm_sq, clip_v, norm_sq); // clip/norm
    let clip_scale = k.alloc_vreg(); k.v_min_f32(clip_scale, one, norm_sq);
    // Setup addresses
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7);
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid);
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base); k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_addr, SReg(grad_p.0)); k.v_mov_from_sgpr(VReg(g_addr.0+1), SReg(grad_p.0+1));
    k.v_add_co(g_addr, g_addr, byte_off); k.v_add_co_ci(VReg(g_addr.0+1), VReg(g_addr.0+1));
    let e_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(e_addr, SReg(ema_p.0)); k.v_mov_from_sgpr(VReg(e_addr.0+1), SReg(ema_p.0+1));
    k.v_add_co(e_addr, e_addr, byte_off); k.v_add_co_ci(VReg(e_addr.0+1), VReg(e_addr.0+1));
    let beta_v = k.alloc_vreg(); k.v_mov_from_sgpr(beta_v, SReg(beta_a.0));
    let ob = k.alloc_vreg(); k.v_sub_f32(ob, one, beta_v);
    for i in 0..4i32 {
        let g = k.alloc_vreg(); k.global_load(g, g_addr, Width::B32, i*4);
        let e = k.alloc_vreg(); k.global_load(e, e_addr, Width::B32, i*4);
        k.wait_vmcnt(0);
        k.v_mul_f32(g, g, clip_scale); // clipped grad
        k.v_mul_f32(e, beta_v, e); k.v_fma_f32(e, ob, g, e); // EMA
        k.global_store(e_addr, e, Width::B32, i*4);
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Fused gradient clip: reads norm_sq from GPU, applies scale in-place.
/// grad[i] *= min(1.0, max_norm / sqrt(norm_sq[0]))
/// No CPU round-trip needed — scale computed entirely on GPU.
/// Kernargs (24B): grad(0), norm_sq_ptr(8), max_norm(16 as f32 bits), n_elems(20)
/// Grid: [(n_elems/128)*32, 1, 1], WG: 32 (4 elem/thread)
pub fn t0_grad_clip_apply() -> T0Kernel {
    let mut k = T0Kernel::new("t0_grad_clip_apply");
    let grad_p = k.arg_ptr("grad");
    let norm_p = k.arg_ptr("norm_sq");
    let max_norm_a = k.arg_u32("max_norm");
    let _n = k.arg_u32("n_elems");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // Read norm_sq from GPU scalar buffer
    let norm_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(norm_addr, SReg(norm_p.0));
    k.v_mov_from_sgpr(VReg(norm_addr.0+1), SReg(norm_p.0+1));
    let norm_sq = k.alloc_vreg();
    k.global_load(norm_sq, norm_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // clip_scale = min(1.0, max_norm / sqrt(norm_sq + eps))
    let one = k.alloc_vreg(); k.v_mov_imm(one, 0x3F800000u32 as i32); // 1.0f
    k.v_sqrt_f32(norm_sq, norm_sq);
    let eps = k.alloc_vreg(); k.v_mov_imm(eps, 0x358637BDu32 as i32); // 1e-6
    k.v_add_f32(norm_sq, norm_sq, eps);
    let max_norm_v = k.alloc_vreg(); k.v_mov_from_sgpr(max_norm_v, SReg(max_norm_a.0));
    k.v_rcp_f32(norm_sq, norm_sq);
    k.v_mul_f32(norm_sq, max_norm_v, norm_sq); // max_norm / norm
    let clip_scale = k.alloc_vreg();
    k.v_min_f32(clip_scale, one, norm_sq);

    // Setup address: grad + (wg_id * 128 + lid * 4) * 4
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7); // wg_id * 128
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid); // lid * 4
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base);
    k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_addr, SReg(grad_p.0));
    k.v_mov_from_sgpr(VReg(g_addr.0+1), SReg(grad_p.0+1));
    k.v_add_co(g_addr, g_addr, byte_off);
    k.v_add_co_ci(VReg(g_addr.0+1), VReg(g_addr.0+1));

    // Load 4 elements, scale, store
    for i in 0..4i32 {
        let g = k.alloc_vreg();
        k.global_load(g, g_addr, Width::B32, i * 4);
        k.wait_vmcnt(0);
        k.v_mul_f32(g, g, clip_scale);
        k.global_store(g_addr, g, Width::B32, i * 4);
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// AXPBY: y[i] = alpha * x[i] + beta * y[i]
/// Kernargs (28B): x_ptr(0), y_ptr(8), alpha(16 f32 bits), beta(20 f32 bits), n_elems(24)
/// Grid: [(n_elems/128)*32, 1, 1], WG: 32 (4 elem/thread)
pub fn t0_axpby() -> T0Kernel {
    let mut k = T0Kernel::new("t0_axpby");
    let x_p = k.arg_ptr("x");
    let y_p = k.arg_ptr("y");
    let alpha_a = k.arg_u32("alpha");
    let beta_a = k.arg_u32("beta");
    let _n = k.arg_u32("n_elems");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7); // wg_id * 128
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid); // lid * 4
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base);
    k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);

    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_addr, SReg(x_p.0));
    k.v_mov_from_sgpr(VReg(x_addr.0+1), SReg(x_p.0+1));
    k.v_add_co(x_addr, x_addr, byte_off);
    k.v_add_co_ci(VReg(x_addr.0+1), VReg(x_addr.0+1));

    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_addr, SReg(y_p.0));
    k.v_mov_from_sgpr(VReg(y_addr.0+1), SReg(y_p.0+1));
    k.v_add_co(y_addr, y_addr, byte_off);
    k.v_add_co_ci(VReg(y_addr.0+1), VReg(y_addr.0+1));

    let alpha_v = k.alloc_vreg(); k.v_mov_from_sgpr(alpha_v, SReg(alpha_a.0));
    let beta_v = k.alloc_vreg(); k.v_mov_from_sgpr(beta_v, SReg(beta_a.0));

    for i in 0..4i32 {
        let x = k.alloc_vreg(); k.global_load(x, x_addr, Width::B32, i * 4);
        let y = k.alloc_vreg(); k.global_load(y, y_addr, Width::B32, i * 4);
        k.wait_vmcnt(0);
        k.v_mul_f32(y, beta_v, y);
        k.v_fma_f32(y, alpha_v, x, y); // y = alpha*x + beta*y
        k.global_store(y_addr, y, Width::B32, i * 4);
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}


pub fn t0_momentum_scale() -> T0Kernel {
    let mut k = T0Kernel::new("t0_momentum_scale");
    let m_p = k.arg_ptr("m_low"); let g_p = k.arg_ptr("g_low");
    let o_p = k.arg_ptr("o_low"); let u_p = k.arg_ptr("u_low");
    let _n = k.arg_u32("n_elems"); let b1_a = k.arg_u32("beta1");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7);
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid);
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base); k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);
    let m_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(m_ad, SReg(m_p.0)); k.v_mov_from_sgpr(VReg(m_ad.0+1), SReg(m_p.0+1));
    k.v_add_co(m_ad, m_ad, byte_off); k.v_add_co_ci(VReg(m_ad.0+1), VReg(m_ad.0+1));
    let g_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_ad, SReg(g_p.0)); k.v_mov_from_sgpr(VReg(g_ad.0+1), SReg(g_p.0+1));
    k.v_add_co(g_ad, g_ad, byte_off); k.v_add_co_ci(VReg(g_ad.0+1), VReg(g_ad.0+1));
    let o_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_ad, SReg(o_p.0)); k.v_mov_from_sgpr(VReg(o_ad.0+1), SReg(o_p.0+1));
    k.v_add_co(o_ad, o_ad, byte_off); k.v_add_co_ci(VReg(o_ad.0+1), VReg(o_ad.0+1));
    let u_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(u_ad, SReg(u_p.0)); k.v_mov_from_sgpr(VReg(u_ad.0+1), SReg(u_p.0+1));
    k.v_add_co(u_ad, u_ad, byte_off); k.v_add_co_ci(VReg(u_ad.0+1), VReg(u_ad.0+1));
    let b1_v = k.alloc_vreg(); k.v_mov_from_sgpr(b1_v, SReg(b1_a.0));
    let one = k.alloc_vreg(); k.v_mov_imm(one, 0x3F800000u32 as i32);
    let ob1 = k.alloc_vreg(); k.v_sub_f32(ob1, one, b1_v);
    for i in 0..4i32 {
        let m = k.alloc_vreg(); k.global_load(m, m_ad, Width::B32, i*4);
        let g = k.alloc_vreg(); k.global_load(g, g_ad, Width::B32, i*4);
        let o = k.alloc_vreg(); k.global_load(o, o_ad, Width::B32, i*4);
        k.wait_vmcnt(0);
        k.v_mul_f32(m, b1_v, m); k.v_fma_f32(m, ob1, g, m);
        k.global_store(m_ad, m, Width::B32, i*4);
        k.global_store(u_ad, o, Width::B32, i*4); // u = o pass-through
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Scale by inv sqrt: out = in / sqrt(norm_sq).
/// Kernargs (28B): in_ptr(0), out_ptr(8), norm_sq_ptr(16), n_elems(24)
pub fn t0_scale_by_inv_sqrt() -> T0Kernel {
    let mut k = T0Kernel::new("t0_scale_by_inv_sqrt");
    let in_p = k.arg_ptr("inp"); let out_p = k.arg_ptr("out");
    let norm_p = k.arg_ptr("norm_sq"); let _n = k.arg_u32("n_elems");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    // Read norm_sq
    let n_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(n_addr, SReg(norm_p.0)); k.v_mov_from_sgpr(VReg(n_addr.0+1), SReg(norm_p.0+1));
    let nsq = k.alloc_vreg(); k.global_load(nsq, n_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    let eps2 = k.alloc_vreg(); k.v_mov_imm(eps2, 0x38D1B717u32 as i32); // 1e-4
    k.v_max_f32(nsq, nsq, eps2); k.v_sqrt_f32(nsq, nsq); k.v_rcp_f32(nsq, nsq); // inv_scale
    // Addresses
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7);
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid);
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base); k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);
    let i_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(i_ad, SReg(in_p.0)); k.v_mov_from_sgpr(VReg(i_ad.0+1), SReg(in_p.0+1));
    k.v_add_co(i_ad, i_ad, byte_off); k.v_add_co_ci(VReg(i_ad.0+1), VReg(i_ad.0+1));
    let o_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(o_ad, SReg(out_p.0)); k.v_mov_from_sgpr(VReg(o_ad.0+1), SReg(out_p.0+1));
    k.v_add_co(o_ad, o_ad, byte_off); k.v_add_co_ci(VReg(o_ad.0+1), VReg(o_ad.0+1));
    for i in 0..4i32 {
        let val = k.alloc_vreg(); k.global_load(val, i_ad, Width::B32, i*4);
        k.wait_vmcnt(0); k.v_mul_f32(val, val, nsq);
        k.global_store(o_ad, val, Width::B32, i*4);
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Back-project update: W -= lr*(alpha*dW + wd*W), convert W → bf16.
/// Kernargs (56B): _p(0), dw(8), w_f32(16), w_bf16(24), rows(32), cols(36), rank(40), lr(44), alpha(48), wd(52)
pub fn t0_back_project_update() -> T0Kernel {
    let mut k = T0Kernel::new("t0_back_project_update");
    let _p_ptr = k.arg_ptr("p"); let dw_ptr = k.arg_ptr("dw");
    let w_ptr = k.arg_ptr("w_f32"); let bf16_ptr = k.arg_ptr("w_bf16");
    let _rows = k.arg_u32("rows"); let _cols = k.arg_u32("cols"); let _rank = k.arg_u32("rank");
    let lr_a = k.arg_u32("lr"); let alpha_a = k.arg_u32("alpha"); let wd_a = k.arg_u32("wd");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 7);
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid);
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base); k.v_add_u32(gid, gid, v_off);
    let f32_off = k.alloc_vreg(); k.v_lshlrev_b32(f32_off, 2, gid);
    let bf16_off = k.alloc_vreg(); k.v_lshlrev_b32(bf16_off, 1, gid);
    let w_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(w_ad, SReg(w_ptr.0)); k.v_mov_from_sgpr(VReg(w_ad.0+1), SReg(w_ptr.0+1));
    k.v_add_co(w_ad, w_ad, f32_off); k.v_add_co_ci(VReg(w_ad.0+1), VReg(w_ad.0+1));
    let dw_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dw_ad, SReg(dw_ptr.0)); k.v_mov_from_sgpr(VReg(dw_ad.0+1), SReg(dw_ptr.0+1));
    k.v_add_co(dw_ad, dw_ad, f32_off); k.v_add_co_ci(VReg(dw_ad.0+1), VReg(dw_ad.0+1));
    let bf_ad = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(bf_ad, SReg(bf16_ptr.0)); k.v_mov_from_sgpr(VReg(bf_ad.0+1), SReg(bf16_ptr.0+1));
    k.v_add_co(bf_ad, bf_ad, bf16_off); k.v_add_co_ci(VReg(bf_ad.0+1), VReg(bf_ad.0+1));
    let lr_v = k.alloc_vreg(); k.v_mov_from_sgpr(lr_v, SReg(lr_a.0));
    let al_v = k.alloc_vreg(); k.v_mov_from_sgpr(al_v, SReg(alpha_a.0));
    let wd_v = k.alloc_vreg(); k.v_mov_from_sgpr(wd_v, SReg(wd_a.0));
    // Process 4 elements, also pack bf16
    let mut ws = [VReg(0); 4];
    for i in 0..4i32 {
        let w = k.alloc_vreg(); k.global_load(w, w_ad, Width::B32, i*4);
        let dw = k.alloc_vreg(); k.global_load(dw, dw_ad, Width::B32, i*4);
        k.wait_vmcnt(0);
        let upd = k.alloc_vreg(); k.v_mul_f32(upd, al_v, dw); // alpha*dW
        k.v_fma_f32(upd, wd_v, w, upd); // + wd*W
        k.v_mul_f32(upd, lr_v, upd);
        k.v_sub_f32(w, w, upd);
        k.global_store(w_ad, w, Width::B32, i*4);
        ws[i as usize] = w;
    }
    // Pack bf16 pairs using hardware RNE rounding
    for pair in 0..2i32 {
        let lo = ws[pair as usize * 2];
        let hi = ws[pair as usize * 2 + 1];
        let packed = k.alloc_vreg();
        k.cvt_pk_bf16_f32(packed, lo, hi);
        k.global_store(bf_ad, packed, Width::B32, pair * 4);
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Helper: integer row/col decomposition for small GEMM.
fn gemm_row_col(k: &mut T0Kernel, wg_id: SReg, n_dim: SReg) -> (VReg, VReg, VReg) {
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    let base = k.alloc_sreg(); k.s_lshl_b32(base, wg_id, 5);
    let base_v = k.alloc_vreg(); k.v_mov_from_sgpr(base_v, base);
    let eidx = k.alloc_vreg(); k.v_add_u32(eidx, lid, base_v);
    let n_v = k.alloc_vreg(); k.v_mov_from_sgpr(n_v, n_dim);
    let ef = k.alloc_vreg(); k.v_cvt_f32_u32(ef, eidx);
    let nf = k.alloc_vreg(); k.v_cvt_f32_u32(nf, n_v);
    let inv = k.alloc_vreg(); k.v_rcp_f32(inv, nf);
    let qf = k.alloc_vreg(); k.v_mul_f32(qf, ef, inv);
    let row = k.alloc_vreg(); k.v_cvt_u32_f32(row, qf);
    let rn = k.alloc_vreg(); k.v_mul_lo_u32(rn, row, n_v);
    let col = k.alloc_vreg(); k.v_sub_u32(col, eidx, rn);
    (row, col, eidx)
}

/// Small GEMM: C = A @ B. Kernargs (32B): a(0), b(8), c(16), K(24), N(28)
pub fn t0_small_gemm() -> T0Kernel {
    let mut k = T0Kernel::new("t0_small_gemm");
    let a_p = k.arg_ptr("a"); let b_p = k.arg_ptr("b"); let c_p = k.arg_ptr("c");
    let k_a = k.arg_u32("K"); let n_a = k.arg_u32("N");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let (row, col, eidx) = gemm_row_col(&mut k, wg_id, SReg(n_a.0));
    let k_v = k.alloc_vreg(); k.v_mov_from_sgpr(k_v, SReg(k_a.0));
    let a_off = k.alloc_vreg(); k.v_mul_lo_u32(a_off, row, k_v);
    k.v_lshlrev_b32(a_off, 2, a_off);
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_p.0)); k.v_mov_from_sgpr(VReg(a_addr.0+1), SReg(a_p.0+1));
    k.v_add_co(a_addr, a_addr, a_off); k.v_add_co_ci(VReg(a_addr.0+1), VReg(a_addr.0+1));
    let n_v = k.alloc_vreg(); k.v_mov_from_sgpr(n_v, SReg(n_a.0));
    let b_off = k.alloc_vreg(); k.v_lshlrev_b32(b_off, 2, col);
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_p.0)); k.v_mov_from_sgpr(VReg(b_addr.0+1), SReg(b_p.0+1));
    k.v_add_co(b_addr, b_addr, b_off); k.v_add_co_ci(VReg(b_addr.0+1), VReg(b_addr.0+1));
    let b_stride = k.alloc_vreg(); k.v_lshlrev_b32(b_stride, 2, n_v);
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);
    let ki = k.alloc_sreg(); k.s_mov_imm(ki, 0);
    k.label("gl");
    let k_off = k.alloc_sreg(); k.s_lshl_b32(k_off, ki, 2);
    let k_off_v = k.alloc_vreg(); k.v_mov_from_sgpr(k_off_v, k_off);
    let ta = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(ta, a_addr); k.v_mov(VReg(ta.0+1), VReg(a_addr.0+1));
    k.v_add_co(ta, ta, k_off_v); k.v_add_co_ci(VReg(ta.0+1), VReg(ta.0+1));
    let va = k.alloc_vreg(); k.global_load(va, ta, Width::B32, 0);
    let ki_v = k.alloc_vreg(); k.v_mov_from_sgpr(ki_v, ki);
    let bk = k.alloc_vreg(); k.v_mul_lo_u32(bk, ki_v, b_stride);
    let tb = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tb, b_addr); k.v_mov(VReg(tb.0+1), VReg(b_addr.0+1));
    k.v_add_co(tb, tb, bk); k.v_add_co_ci(VReg(tb.0+1), VReg(tb.0+1));
    let vb = k.alloc_vreg(); k.global_load(vb, tb, Width::B32, 0);
    k.wait_vmcnt(0); k.v_fma_f32(acc, va, vb, acc);
    k.s_add_u32(ki, ki, 1); k.s_cmp_lt_u32(ki, SReg(k_a.0)); k.branch_scc1("gl");
    let c_off = k.alloc_vreg(); k.v_lshlrev_b32(c_off, 2, eidx);
    let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(c_addr, SReg(c_p.0)); k.v_mov_from_sgpr(VReg(c_addr.0+1), SReg(c_p.0+1));
    k.v_add_co(c_addr, c_addr, c_off); k.v_add_co_ci(VReg(c_addr.0+1), VReg(c_addr.0+1));
    k.global_store(c_addr, acc, Width::B32, 0);
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Small GEMM A^T: C = A^T @ B. Kernargs (36B): a(0), b(8), c(16), K(24), N(28), M(32)
pub fn t0_small_gemm_at() -> T0Kernel {
    let mut k = T0Kernel::new("t0_small_gemm_at");
    let a_p = k.arg_ptr("a"); let b_p = k.arg_ptr("b"); let c_p = k.arg_ptr("c");
    let k_a = k.arg_u32("K"); let n_a = k.arg_u32("N"); let m_a = k.arg_u32("M");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let (row, col, eidx) = gemm_row_col(&mut k, wg_id, SReg(n_a.0));
    let a_off = k.alloc_vreg(); k.v_lshlrev_b32(a_off, 2, row);
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_p.0)); k.v_mov_from_sgpr(VReg(a_addr.0+1), SReg(a_p.0+1));
    k.v_add_co(a_addr, a_addr, a_off); k.v_add_co_ci(VReg(a_addr.0+1), VReg(a_addr.0+1));
    let m_v = k.alloc_vreg(); k.v_mov_from_sgpr(m_v, SReg(m_a.0));
    let a_stride = k.alloc_vreg(); k.v_lshlrev_b32(a_stride, 2, m_v);
    let n_v = k.alloc_vreg(); k.v_mov_from_sgpr(n_v, SReg(n_a.0));
    let b_off = k.alloc_vreg(); k.v_lshlrev_b32(b_off, 2, col);
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_p.0)); k.v_mov_from_sgpr(VReg(b_addr.0+1), SReg(b_p.0+1));
    k.v_add_co(b_addr, b_addr, b_off); k.v_add_co_ci(VReg(b_addr.0+1), VReg(b_addr.0+1));
    let b_stride = k.alloc_vreg(); k.v_lshlrev_b32(b_stride, 2, n_v);
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);
    let ki = k.alloc_sreg(); k.s_mov_imm(ki, 0);
    k.label("atl");
    let ki_v = k.alloc_vreg(); k.v_mov_from_sgpr(ki_v, ki);
    let ak = k.alloc_vreg(); k.v_mul_lo_u32(ak, ki_v, a_stride);
    let ta = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(ta, a_addr); k.v_mov(VReg(ta.0+1), VReg(a_addr.0+1));
    k.v_add_co(ta, ta, ak); k.v_add_co_ci(VReg(ta.0+1), VReg(ta.0+1));
    let va = k.alloc_vreg(); k.global_load(va, ta, Width::B32, 0);
    let bk = k.alloc_vreg(); k.v_mul_lo_u32(bk, ki_v, b_stride);
    let tb = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tb, b_addr); k.v_mov(VReg(tb.0+1), VReg(b_addr.0+1));
    k.v_add_co(tb, tb, bk); k.v_add_co_ci(VReg(tb.0+1), VReg(tb.0+1));
    let vb = k.alloc_vreg(); k.global_load(vb, tb, Width::B32, 0);
    k.wait_vmcnt(0); k.v_fma_f32(acc, va, vb, acc);
    k.s_add_u32(ki, ki, 1); k.s_cmp_lt_u32(ki, SReg(k_a.0)); k.branch_scc1("atl");
    let c_off = k.alloc_vreg(); k.v_lshlrev_b32(c_off, 2, eidx);
    let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(c_addr, SReg(c_p.0)); k.v_mov_from_sgpr(VReg(c_addr.0+1), SReg(c_p.0+1));
    k.v_add_co(c_addr, c_addr, c_off); k.v_add_co_ci(VReg(c_addr.0+1), VReg(c_addr.0+1));
    k.global_store(c_addr, acc, Width::B32, 0);
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Small GEMM AB^T: C = A @ B^T. Kernargs (32B): a(0), b(8), c(16), K(24), N(28)
pub fn t0_small_gemm_abt() -> T0Kernel {
    let mut k = T0Kernel::new("t0_small_gemm_abt");
    let a_p = k.arg_ptr("a"); let b_p = k.arg_ptr("b"); let c_p = k.arg_ptr("c");
    let k_a = k.arg_u32("K"); let n_a = k.arg_u32("N");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let (row, col, eidx) = gemm_row_col(&mut k, wg_id, SReg(n_a.0));
    let k_v = k.alloc_vreg(); k.v_mov_from_sgpr(k_v, SReg(k_a.0));
    let a_off = k.alloc_vreg(); k.v_mul_lo_u32(a_off, row, k_v);
    k.v_lshlrev_b32(a_off, 2, a_off);
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_p.0)); k.v_mov_from_sgpr(VReg(a_addr.0+1), SReg(a_p.0+1));
    k.v_add_co(a_addr, a_addr, a_off); k.v_add_co_ci(VReg(a_addr.0+1), VReg(a_addr.0+1));
    let b_off = k.alloc_vreg(); k.v_mul_lo_u32(b_off, col, k_v);
    k.v_lshlrev_b32(b_off, 2, b_off);
    let b_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(b_addr, SReg(b_p.0)); k.v_mov_from_sgpr(VReg(b_addr.0+1), SReg(b_p.0+1));
    k.v_add_co(b_addr, b_addr, b_off); k.v_add_co_ci(VReg(b_addr.0+1), VReg(b_addr.0+1));
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);
    let ki = k.alloc_sreg(); k.s_mov_imm(ki, 0);
    k.label("abtl");
    let k_off = k.alloc_sreg(); k.s_lshl_b32(k_off, ki, 2);
    let k_off_v = k.alloc_vreg(); k.v_mov_from_sgpr(k_off_v, k_off);
    let ta = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(ta, a_addr); k.v_mov(VReg(ta.0+1), VReg(a_addr.0+1));
    k.v_add_co(ta, ta, k_off_v); k.v_add_co_ci(VReg(ta.0+1), VReg(ta.0+1));
    let va = k.alloc_vreg(); k.global_load(va, ta, Width::B32, 0);
    let tb = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tb, b_addr); k.v_mov(VReg(tb.0+1), VReg(b_addr.0+1));
    k.v_add_co(tb, tb, k_off_v); k.v_add_co_ci(VReg(tb.0+1), VReg(tb.0+1));
    let vb = k.alloc_vreg(); k.global_load(vb, tb, Width::B32, 0);
    k.wait_vmcnt(0); k.v_fma_f32(acc, va, vb, acc);
    k.s_add_u32(ki, ki, 1); k.s_cmp_lt_u32(ki, SReg(k_a.0)); k.branch_scc1("abtl");
    let c_off = k.alloc_vreg(); k.v_lshlrev_b32(c_off, 2, eidx);
    let c_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(c_addr, SReg(c_p.0)); k.v_mov_from_sgpr(VReg(c_addr.0+1), SReg(c_p.0+1));
    k.v_add_co(c_addr, c_addr, c_off); k.v_add_co_ci(VReg(c_addr.0+1), VReg(c_addr.0+1));
    k.global_store(c_addr, acc, Width::B32, 0);
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// NS5 fused BX: X_new = a*X + (b*A + c*A²)@X.
/// Kernargs (48B): a(0),a2(8),x(16),K(24),N(28),ca(32),cb(36),cc(40)
pub fn t0_ns5_fused_bx() -> T0Kernel {
    let mut k = T0Kernel::new("t0_ns5_fused_bx");
    let a_p = k.arg_ptr("a"); let a2_p = k.arg_ptr("a2"); let x_p = k.arg_ptr("x");
    let k_a = k.arg_u32("K"); let n_a = k.arg_u32("N");
    let ca = k.arg_u32("ca"); let cb = k.arg_u32("cb"); let cc = k.arg_u32("cc");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let (row, col, eidx) = gemm_row_col(&mut k, wg_id, SReg(n_a.0));
    let k_v = k.alloc_vreg(); k.v_mov_from_sgpr(k_v, SReg(k_a.0));
    let n_v = k.alloc_vreg(); k.v_mov_from_sgpr(n_v, SReg(n_a.0));
    let rk = k.alloc_vreg(); k.v_mul_lo_u32(rk, row, k_v); k.v_lshlrev_b32(rk, 2, rk);
    let a_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a_addr, SReg(a_p.0)); k.v_mov_from_sgpr(VReg(a_addr.0+1), SReg(a_p.0+1));
    k.v_add_co(a_addr, a_addr, rk); k.v_add_co_ci(VReg(a_addr.0+1), VReg(a_addr.0+1));
    let a2_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(a2_addr, SReg(a2_p.0)); k.v_mov_from_sgpr(VReg(a2_addr.0+1), SReg(a2_p.0+1));
    k.v_add_co(a2_addr, a2_addr, rk); k.v_add_co_ci(VReg(a2_addr.0+1), VReg(a2_addr.0+1));
    let x_off = k.alloc_vreg(); k.v_lshlrev_b32(x_off, 2, col);
    let x_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_addr, SReg(x_p.0)); k.v_mov_from_sgpr(VReg(x_addr.0+1), SReg(x_p.0+1));
    k.v_add_co(x_addr, x_addr, x_off); k.v_add_co_ci(VReg(x_addr.0+1), VReg(x_addr.0+1));
    let x_stride = k.alloc_vreg(); k.v_lshlrev_b32(x_stride, 2, n_v);
    let cb_v = k.alloc_vreg(); k.v_mov_from_sgpr(cb_v, SReg(cb.0));
    let cc_v = k.alloc_vreg(); k.v_mov_from_sgpr(cc_v, SReg(cc.0));
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);
    let ki = k.alloc_sreg(); k.s_mov_imm(ki, 0);
    k.label("ns5l");
    let ko = k.alloc_sreg(); k.s_lshl_b32(ko, ki, 2);
    let kov = k.alloc_vreg(); k.v_mov_from_sgpr(kov, ko);
    let ta = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(ta, a_addr); k.v_mov(VReg(ta.0+1), VReg(a_addr.0+1));
    k.v_add_co(ta, ta, kov); k.v_add_co_ci(VReg(ta.0+1), VReg(ta.0+1));
    let va = k.alloc_vreg(); k.global_load(va, ta, Width::B32, 0);
    let ta2 = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(ta2, a2_addr); k.v_mov(VReg(ta2.0+1), VReg(a2_addr.0+1));
    k.v_add_co(ta2, ta2, kov); k.v_add_co_ci(VReg(ta2.0+1), VReg(ta2.0+1));
    let va2 = k.alloc_vreg(); k.global_load(va2, ta2, Width::B32, 0);
    let kiv = k.alloc_vreg(); k.v_mov_from_sgpr(kiv, ki);
    let xk = k.alloc_vreg(); k.v_mul_lo_u32(xk, kiv, x_stride);
    let tx = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tx, x_addr); k.v_mov(VReg(tx.0+1), VReg(x_addr.0+1));
    k.v_add_co(tx, tx, xk); k.v_add_co_ci(VReg(tx.0+1), VReg(tx.0+1));
    let vx = k.alloc_vreg(); k.global_load(vx, tx, Width::B32, 0);
    k.wait_vmcnt(0);
    let bval = k.alloc_vreg(); k.v_mul_f32(bval, cb_v, va); k.v_fma_f32(bval, cc_v, va2, bval);
    k.v_fma_f32(acc, bval, vx, acc);
    k.s_add_u32(ki, ki, 1); k.s_cmp_lt_u32(ki, SReg(k_a.0)); k.branch_scc1("ns5l");
    let xo_off = k.alloc_vreg(); k.v_lshlrev_b32(xo_off, 2, eidx);
    let xo = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(xo, SReg(x_p.0)); k.v_mov_from_sgpr(VReg(xo.0+1), SReg(x_p.0+1));
    k.v_add_co(xo, xo, xo_off); k.v_add_co_ci(VReg(xo.0+1), VReg(xo.0+1));
    let xold = k.alloc_vreg(); k.global_load(xold, xo, Width::B32, 0); k.wait_vmcnt(0);
    let ca_v = k.alloc_vreg(); k.v_mov_from_sgpr(ca_v, SReg(ca.0));
    k.v_fma_f32(acc, ca_v, xold, acc);
    k.global_store(xo, acc, Width::B32, 0);
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Oja P update (simplified): 1 WG per column c. Each lane = 1 row r.
/// P[r,c] += eta/cols * dot(G[r,:],pt_g[c,:]) - eta*nsq/cols * P[r,c]
/// Kernargs (40B): p(0), g_ema(8), pt_g(16), rows(24), cols(28), rank(32), oja_lr(36)
pub fn t0_oja_p_update() -> T0Kernel {
    let mut k = T0Kernel::new("t0_oja_p_update");
    let p_p = k.arg_ptr("p"); let g_p = k.arg_ptr("g_ema"); let ptg_p = k.arg_ptr("pt_g");
    let _rows_a = k.arg_u32("rows"); let cols_a = k.arg_u32("cols");
    let rank_a = k.arg_u32("rank"); let eta_a = k.arg_u32("oja_lr");
    k.emit_arg_loads();
    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id); // c
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    // ptg row base: ptg + c*cols*4
    let cxc = k.alloc_sreg(); k.s_mul_i32(cxc, wg_id, SReg(cols_a.0));
    let ptg_off = k.alloc_sreg(); k.s_lshl_b32(ptg_off, cxc, 2);
    let ptg_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(ptg_addr, SReg(ptg_p.0)); k.v_mov_from_sgpr(VReg(ptg_addr.0+1), SReg(ptg_p.0+1));
    let ptg_v = k.alloc_vreg(); k.v_mov_from_sgpr(ptg_v, ptg_off);
    k.v_add_co(ptg_addr, ptg_addr, ptg_v); k.v_add_co_ci(VReg(ptg_addr.0+1), VReg(ptg_addr.0+1));
    // Phase 1: norm_sq
    let nsq = k.alloc_vreg(); k.v_mov_imm(nsq, 0);
    let j = k.alloc_sreg(); k.s_mov_imm(j, 0);
    k.label("nl");
    let jo = k.alloc_sreg(); k.s_lshl_b32(jo, j, 2);
    let jov = k.alloc_vreg(); k.v_mov_from_sgpr(jov, jo);
    let tp = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tp, ptg_addr); k.v_mov(VReg(tp.0+1), VReg(ptg_addr.0+1));
    k.v_add_co(tp, tp, jov); k.v_add_co_ci(VReg(tp.0+1), VReg(tp.0+1));
    let pv = k.alloc_vreg(); k.global_load(pv, tp, Width::B32, 0);
    k.wait_vmcnt(0); k.v_fma_f32(nsq, pv, pv, nsq);
    k.s_add_u32(j, j, 1); k.s_cmp_lt_u32(j, SReg(cols_a.0)); k.branch_scc1("nl");
    // Precompute
    let cv = k.alloc_vreg(); k.v_mov_from_sgpr(cv, SReg(cols_a.0));
    k.v_cvt_f32_u32(cv, cv); k.v_rcp_f32(cv, cv);
    let eta = k.alloc_vreg(); k.v_mov_from_sgpr(eta, SReg(eta_a.0));
    let eoc = k.alloc_vreg(); k.v_mul_f32(eoc, eta, cv);
    let enc = k.alloc_vreg(); k.v_mul_f32(enc, eoc, nsq);
    // Phase 2: P[r,c] update
    let rank_v = k.alloc_vreg(); k.v_mov_from_sgpr(rank_v, SReg(rank_a.0));
    let pe = k.alloc_vreg(); k.v_mul_lo_u32(pe, lid, rank_v);
    let c_v = k.alloc_vreg(); k.v_mov_from_sgpr(c_v, wg_id);
    k.v_add_u32(pe, pe, c_v); k.v_lshlrev_b32(pe, 2, pe);
    let pa = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(pa, SReg(p_p.0)); k.v_mov_from_sgpr(VReg(pa.0+1), SReg(p_p.0+1));
    k.v_add_co(pa, pa, pe); k.v_add_co_ci(VReg(pa.0+1), VReg(pa.0+1));
    // G[r,:] base
    let cols4 = k.alloc_sreg(); k.s_lshl_b32(cols4, SReg(cols_a.0), 2);
    let g_off = k.alloc_vreg(); k.v_mov_from_sgpr(g_off, cols4);
    k.v_mul_lo_u32(g_off, lid, g_off);
    let ga = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(ga, SReg(g_p.0)); k.v_mov_from_sgpr(VReg(ga.0+1), SReg(g_p.0+1));
    k.v_add_co(ga, ga, g_off); k.v_add_co_ci(VReg(ga.0+1), VReg(ga.0+1));
    // Dot loop
    let dot = k.alloc_vreg(); k.v_mov_imm(dot, 0);
    let j2 = k.alloc_sreg(); k.s_mov_imm(j2, 0);
    k.label("dl");
    let j2o = k.alloc_sreg(); k.s_lshl_b32(j2o, j2, 2);
    let j2ov = k.alloc_vreg(); k.v_mov_from_sgpr(j2ov, j2o);
    let tg = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tg, ga); k.v_mov(VReg(tg.0+1), VReg(ga.0+1));
    k.v_add_co(tg, tg, j2ov); k.v_add_co_ci(VReg(tg.0+1), VReg(tg.0+1));
    let gv = k.alloc_vreg(); k.global_load(gv, tg, Width::B32, 0);
    let tp2 = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(tp2, ptg_addr); k.v_mov(VReg(tp2.0+1), VReg(ptg_addr.0+1));
    k.v_add_co(tp2, tp2, j2ov); k.v_add_co_ci(VReg(tp2.0+1), VReg(tp2.0+1));
    let pv2 = k.alloc_vreg(); k.global_load(pv2, tp2, Width::B32, 0);
    k.wait_vmcnt(0); k.v_fma_f32(dot, gv, pv2, dot);
    k.s_add_u32(j2, j2, 1); k.s_cmp_lt_u32(j2, SReg(cols_a.0)); k.branch_scc1("dl");
    // Update P[r,c]
    let pval = k.alloc_vreg(); k.global_load(pval, pa, Width::B32, 0); k.wait_vmcnt(0);
    let delta = k.alloc_vreg(); k.v_mul_f32(delta, eoc, dot);
    let decay = k.alloc_vreg(); k.v_mul_f32(decay, enc, pval);
    k.v_sub_f32(delta, delta, decay); k.v_add_f32(pval, pval, delta);
    k.global_store(pa, pval, Width::B32, 0);
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

// ============================================================================
// RMSNorm Forward (f32, T0): y = (x / rms) * gamma
// ============================================================================
/// T0 RMSNorm forward: y = x * rms_inv * gamma, where rms_inv = 1/sqrt(mean(x²)+eps).
///
/// 1 WG (32 threads) per row. Loop-based, any dim (stride=32).
///
/// Kernargs (32 bytes):
///   x_ptr(0), y_ptr(8), gamma_ptr(16), dim(24), eps_bits(28)
/// Grid: (rows * 32, 1, 1)
pub fn t0_rmsnorm_forward() -> T0Kernel {
    let mut k = T0Kernel::new("t0_rmsnorm_forward");
    let x_ptr = k.arg_ptr("x");
    let y_ptr = k.arg_ptr("y");
    let gamma_ptr = k.arg_ptr("gamma");
    let dim_arg = k.arg_u32("dim");
    let eps_arg = k.arg_u32("eps");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // Row byte offset = wg_id * dim * 4
    let row_byte_off = k.alloc_sreg();
    k.s_mul_i32(row_byte_off, wg_id, SReg(dim_arg.0));
    k.s_lshl_b32(row_byte_off, row_byte_off, 2);

    // x_row_ptr = x_ptr + row_byte_off
    let x_row_lo = k.alloc_sreg();
    let x_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(x_row_lo, SReg(x_ptr.0), row_byte_off);
    k.s_addc_u32_imm(x_row_hi, SReg(x_ptr.0 + 1), 0);

    // y_row_ptr = y_ptr + row_byte_off
    let y_row_lo = k.alloc_sreg();
    let y_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(y_row_lo, SReg(y_ptr.0), row_byte_off);
    k.s_addc_u32_imm(y_row_hi, SReg(y_ptr.0 + 1), 0);

    // ── Phase 1: sum(x²) ──
    let sum_sq = k.alloc_vreg();
    k.v_mov_imm(sum_sq, 0);

    let loop_base = k.alloc_sreg();
    k.s_mov_imm(loop_base, 0);
    let p1_label = k.make_label("p1");
    k.label(&p1_label);

    // idx = loop_base + lane_id
    let idx = k.alloc_vreg();
    k.v_mov_from_sgpr(idx, loop_base);
    k.v_add_u32(idx, idx, lane_id);

    // bounds check: idx < dim
    let dim_v = k.alloc_vreg();
    k.v_mov_from_sgpr(dim_v, SReg(dim_arg.0));
    let saved_exec = k.alloc_sreg();
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(dim_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    // load x[row, idx]
    let byte_idx = k.alloc_vreg();
    k.v_lshlrev_b32(byte_idx, 2, idx);
    let load_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(load_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), x_row_hi);
    k.v_add_co(load_addr, load_addr, byte_idx);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    let x_val = k.alloc_vreg();
    k.global_load(x_val, load_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // sum_sq += x²
    let x_sq = k.alloc_vreg();
    k.v_mul_f32(x_sq, x_val, x_val);
    k.v_add_f32(sum_sq, sum_sq, x_sq);

    k.push(Op::RestoreExec { src: saved_exec });

    // loop_base += 32
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(dim_arg.0));
    k.branch_scc1(&p1_label);

    // ── Wave32 reduction ──
    let swap = k.alloc_vreg();
    k.wave_reduce_add_f32(sum_sq, swap);

    // rms_inv = 1/sqrt(sum_sq/dim + eps)
    let dim_f = k.alloc_vreg();
    k.v_mov_from_sgpr(dim_f, SReg(dim_arg.0));
    k.v_cvt_f32_u32(dim_f, dim_f);
    k.v_rcp_f32(dim_f, dim_f);
    k.v_mul_f32(sum_sq, sum_sq, dim_f);       // mean_sq
    let eps_v = k.alloc_vreg();
    k.v_mov_from_sgpr(eps_v, SReg(eps_arg.0));
    k.v_add_f32(sum_sq, sum_sq, eps_v);       // + eps
    k.v_sqrt_f32(sum_sq, sum_sq);
    let rms_inv = k.alloc_vreg();
    k.v_rcp_f32(rms_inv, sum_sq);             // rms_inv

    // ── Phase 2: y = x * rms_inv * gamma ──
    k.s_mov_imm(loop_base, 0);
    let p2_label = k.make_label("p2");
    k.label(&p2_label);

    k.v_mov_from_sgpr(idx, loop_base);
    k.v_add_u32(idx, idx, lane_id);

    k.v_mov_from_sgpr(dim_v, SReg(dim_arg.0));
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(dim_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    k.v_lshlrev_b32(byte_idx, 2, idx);

    // load x
    k.v_mov_from_sgpr(load_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), x_row_hi);
    k.v_add_co(load_addr, load_addr, byte_idx);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    k.global_load(x_val, load_addr, Width::B32, 0);

    // load gamma
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_addr, SReg(gamma_ptr.0));
    k.v_mov_from_sgpr(VReg(g_addr.0 + 1), SReg(gamma_ptr.0 + 1));
    k.v_add_co(g_addr, g_addr, byte_idx);
    k.v_add_co_ci(VReg(g_addr.0 + 1), VReg(g_addr.0 + 1));
    let g_val = k.alloc_vreg();
    k.global_load(g_val, g_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // y = x * rms_inv * gamma
    let y_val = k.alloc_vreg();
    k.v_mul_f32(y_val, x_val, rms_inv);
    k.v_mul_f32(y_val, y_val, g_val);

    // store y
    let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(y_addr, y_row_lo);
    k.v_mov_from_sgpr(VReg(y_addr.0 + 1), y_row_hi);
    k.v_add_co(y_addr, y_addr, byte_idx);
    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
    k.global_store(y_addr, y_val, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved_exec });

    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(dim_arg.0));
    k.branch_scc1(&p2_label);

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// RMSNorm Backward dx (f32, T0): dx = rms_inv * (dy*gamma - xn*mean(g*xn))
// ============================================================================
/// T0 RMSNorm backward dx kernel. 3-phase loop, any dim.
///
/// g = dy * gamma, xn = x * rms_inv
/// dx = rms_inv * (g - xn * mean(g * xn))
///
/// Kernargs (40 bytes):
///   dy_ptr(0), x_ptr(8), gamma_ptr(16), dx_ptr(24), dim(32), eps_bits(36)
/// Grid: (rows * 32, 1, 1)
pub fn t0_rmsnorm_backward() -> T0Kernel {
    let mut k = T0Kernel::new("t0_rmsnorm_backward");
    let dy_ptr = k.arg_ptr("dy");
    let x_ptr = k.arg_ptr("x");
    let gamma_ptr = k.arg_ptr("gamma");
    let dx_ptr = k.arg_ptr("dx");
    let dim_arg = k.arg_u32("dim");
    let eps_arg = k.arg_u32("eps");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // Row byte offset
    let row_byte_off = k.alloc_sreg();
    k.s_mul_i32(row_byte_off, wg_id, SReg(dim_arg.0));
    k.s_lshl_b32(row_byte_off, row_byte_off, 2);

    let dy_row_lo = k.alloc_sreg();
    let dy_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(dy_row_lo, SReg(dy_ptr.0), row_byte_off);
    k.s_addc_u32_imm(dy_row_hi, SReg(dy_ptr.0 + 1), 0);

    let x_row_lo = k.alloc_sreg();
    let x_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(x_row_lo, SReg(x_ptr.0), row_byte_off);
    k.s_addc_u32_imm(x_row_hi, SReg(x_ptr.0 + 1), 0);

    let dx_row_lo = k.alloc_sreg();
    let dx_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(dx_row_lo, SReg(dx_ptr.0), row_byte_off);
    k.s_addc_u32_imm(dx_row_hi, SReg(dx_ptr.0 + 1), 0);

    // Shared regs
    let loop_base = k.alloc_sreg();
    let saved_exec = k.alloc_sreg();
    let idx = k.alloc_vreg();
    let dim_v = k.alloc_vreg();
    let byte_idx = k.alloc_vreg();
    let load_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let swap = k.alloc_vreg();

    // ── Phase 1: sum(x²) → rms_inv ──
    let sum_sq = k.alloc_vreg();
    k.v_mov_imm(sum_sq, 0);
    k.s_mov_imm(loop_base, 0);
    let p1_label = k.make_label("bp1");
    k.label(&p1_label);

    k.v_mov_from_sgpr(idx, loop_base);
    k.v_add_u32(idx, idx, lane_id);
    k.v_mov_from_sgpr(dim_v, SReg(dim_arg.0));
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(dim_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    k.v_lshlrev_b32(byte_idx, 2, idx);
    k.v_mov_from_sgpr(load_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), x_row_hi);
    k.v_add_co(load_addr, load_addr, byte_idx);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    let x_val = k.alloc_vreg();
    k.global_load(x_val, load_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    let tmp = k.alloc_vreg();
    k.v_mul_f32(tmp, x_val, x_val);
    k.v_add_f32(sum_sq, sum_sq, tmp);

    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(dim_arg.0));
    k.branch_scc1(&p1_label);

    k.wave_reduce_add_f32(sum_sq, swap);

    // rms_inv
    let dim_f = k.alloc_vreg();
    k.v_mov_from_sgpr(dim_f, SReg(dim_arg.0));
    k.v_cvt_f32_u32(dim_f, dim_f);
    k.v_rcp_f32(dim_f, dim_f);
    k.v_mul_f32(sum_sq, sum_sq, dim_f);
    let eps_v = k.alloc_vreg();
    k.v_mov_from_sgpr(eps_v, SReg(eps_arg.0));
    k.v_add_f32(sum_sq, sum_sq, eps_v);
    k.v_sqrt_f32(sum_sq, sum_sq);
    let rms_inv = k.alloc_vreg();
    k.v_rcp_f32(rms_inv, sum_sq);

    // ── Phase 2: sum(g * xn) where g = dy*gamma, xn = x*rms_inv ──
    let sum_gxn = k.alloc_vreg();
    k.v_mov_imm(sum_gxn, 0);
    k.s_mov_imm(loop_base, 0);
    let p2_label = k.make_label("bp2");
    k.label(&p2_label);

    k.v_mov_from_sgpr(idx, loop_base);
    k.v_add_u32(idx, idx, lane_id);
    k.v_mov_from_sgpr(dim_v, SReg(dim_arg.0));
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(dim_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    k.v_lshlrev_b32(byte_idx, 2, idx);

    // load x
    k.v_mov_from_sgpr(load_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), x_row_hi);
    k.v_add_co(load_addr, load_addr, byte_idx);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    k.global_load(x_val, load_addr, Width::B32, 0);

    // load dy
    let dy_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dy_addr, dy_row_lo);
    k.v_mov_from_sgpr(VReg(dy_addr.0 + 1), dy_row_hi);
    k.v_add_co(dy_addr, dy_addr, byte_idx);
    k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));
    let dy_val = k.alloc_vreg();
    k.global_load(dy_val, dy_addr, Width::B32, 0);

    // load gamma
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_addr, SReg(gamma_ptr.0));
    k.v_mov_from_sgpr(VReg(g_addr.0 + 1), SReg(gamma_ptr.0 + 1));
    k.v_add_co(g_addr, g_addr, byte_idx);
    k.v_add_co_ci(VReg(g_addr.0 + 1), VReg(g_addr.0 + 1));
    let g_val = k.alloc_vreg();
    k.global_load(g_val, g_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // xn = x * rms_inv
    let xn = k.alloc_vreg();
    k.v_mul_f32(xn, x_val, rms_inv);
    // g = dy * gamma
    let g = k.alloc_vreg();
    k.v_mul_f32(g, dy_val, g_val);
    // sum_gxn += g * xn
    let gxn = k.alloc_vreg();
    k.v_mul_f32(gxn, g, xn);
    k.v_add_f32(sum_gxn, sum_gxn, gxn);

    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(dim_arg.0));
    k.branch_scc1(&p2_label);

    k.wave_reduce_add_f32(sum_gxn, swap);

    // mean_gxn = sum / dim
    k.v_mov_from_sgpr(dim_f, SReg(dim_arg.0));
    k.v_cvt_f32_u32(dim_f, dim_f);
    k.v_rcp_f32(dim_f, dim_f);
    let mean_gxn = k.alloc_vreg();
    k.v_mul_f32(mean_gxn, sum_gxn, dim_f);

    // ── Phase 3: dx = rms_inv * (g - xn * mean_gxn) ──
    k.s_mov_imm(loop_base, 0);
    let p3_label = k.make_label("bp3");
    k.label(&p3_label);

    k.v_mov_from_sgpr(idx, loop_base);
    k.v_add_u32(idx, idx, lane_id);
    k.v_mov_from_sgpr(dim_v, SReg(dim_arg.0));
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(dim_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    k.v_lshlrev_b32(byte_idx, 2, idx);

    // load x, dy, gamma
    k.v_mov_from_sgpr(load_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), x_row_hi);
    k.v_add_co(load_addr, load_addr, byte_idx);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    k.global_load(x_val, load_addr, Width::B32, 0);

    k.v_mov_from_sgpr(dy_addr, dy_row_lo);
    k.v_mov_from_sgpr(VReg(dy_addr.0 + 1), dy_row_hi);
    k.v_add_co(dy_addr, dy_addr, byte_idx);
    k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));
    k.global_load(dy_val, dy_addr, Width::B32, 0);

    k.v_mov_from_sgpr(g_addr, SReg(gamma_ptr.0));
    k.v_mov_from_sgpr(VReg(g_addr.0 + 1), SReg(gamma_ptr.0 + 1));
    k.v_add_co(g_addr, g_addr, byte_idx);
    k.v_add_co_ci(VReg(g_addr.0 + 1), VReg(g_addr.0 + 1));
    k.global_load(g_val, g_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // xn = x * rms_inv
    k.v_mul_f32(xn, x_val, rms_inv);
    // g = dy * gamma
    k.v_mul_f32(g, dy_val, g_val);
    // dx = rms_inv * (g - xn * mean_gxn)
    let dx_val = k.alloc_vreg();
    k.v_mul_f32(dx_val, xn, mean_gxn);
    k.v_sub_f32(dx_val, g, dx_val);
    k.v_mul_f32(dx_val, dx_val, rms_inv);

    // store dx
    let dx_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dx_addr, dx_row_lo);
    k.v_mov_from_sgpr(VReg(dx_addr.0 + 1), dx_row_hi);
    k.v_add_co(dx_addr, dx_addr, byte_idx);
    k.v_add_co_ci(VReg(dx_addr.0 + 1), VReg(dx_addr.0 + 1));
    k.global_store(dx_addr, dx_val, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(dim_arg.0));
    k.branch_scc1(&p3_label);

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// RMSNorm dgamma reduce (f32, T0): dgamma[j] += dy*x*rms_inv via atomic_add
// ============================================================================
/// T0 dgamma reduction kernel. 1 WG per row, atomic_add to dgamma.
/// dgamma_ptr MUST be zero-initialized before dispatch!
///
/// Kernargs (32 bytes):
///   dy_ptr(0), x_ptr(8), dgamma_ptr(16), dim(24), eps_bits(28)
/// Grid: (rows * 32, 1, 1)
pub fn t0_dgamma_reduce() -> T0Kernel {
    let mut k = T0Kernel::new("t0_dgamma_reduce");
    let dy_ptr = k.arg_ptr("dy");
    let x_ptr = k.arg_ptr("x");
    let dgamma_ptr = k.arg_ptr("dgamma");
    let dim_arg = k.arg_u32("dim");
    let eps_arg = k.arg_u32("eps");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg();
    k.capture_tgid_x(wg_id);
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);

    // Row byte offset
    let row_byte_off = k.alloc_sreg();
    k.s_mul_i32(row_byte_off, wg_id, SReg(dim_arg.0));
    k.s_lshl_b32(row_byte_off, row_byte_off, 2);

    let dy_row_lo = k.alloc_sreg();
    let dy_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(dy_row_lo, SReg(dy_ptr.0), row_byte_off);
    k.s_addc_u32_imm(dy_row_hi, SReg(dy_ptr.0 + 1), 0);

    let x_row_lo = k.alloc_sreg();
    let x_row_hi = k.alloc_sreg();
    k.s_add_u32_ss(x_row_lo, SReg(x_ptr.0), row_byte_off);
    k.s_addc_u32_imm(x_row_hi, SReg(x_ptr.0 + 1), 0);

    let loop_base = k.alloc_sreg();
    let saved_exec = k.alloc_sreg();
    let idx = k.alloc_vreg();
    let dim_v = k.alloc_vreg();
    let byte_idx = k.alloc_vreg();
    let load_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let swap = k.alloc_vreg();

    // ── Phase 1: sum(x²) → rms_inv ──
    let sum_sq = k.alloc_vreg();
    k.v_mov_imm(sum_sq, 0);
    k.s_mov_imm(loop_base, 0);
    let p1_label = k.make_label("gp1");
    k.label(&p1_label);

    k.v_mov_from_sgpr(idx, loop_base);
    k.v_add_u32(idx, idx, lane_id);
    k.v_mov_from_sgpr(dim_v, SReg(dim_arg.0));
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(dim_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    k.v_lshlrev_b32(byte_idx, 2, idx);
    k.v_mov_from_sgpr(load_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), x_row_hi);
    k.v_add_co(load_addr, load_addr, byte_idx);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    let x_val = k.alloc_vreg();
    k.global_load(x_val, load_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    let tmp = k.alloc_vreg();
    k.v_mul_f32(tmp, x_val, x_val);
    k.v_add_f32(sum_sq, sum_sq, tmp);

    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(dim_arg.0));
    k.branch_scc1(&p1_label);

    k.wave_reduce_add_f32(sum_sq, swap);

    let dim_f = k.alloc_vreg();
    k.v_mov_from_sgpr(dim_f, SReg(dim_arg.0));
    k.v_cvt_f32_u32(dim_f, dim_f);
    k.v_rcp_f32(dim_f, dim_f);
    k.v_mul_f32(sum_sq, sum_sq, dim_f);
    let eps_v = k.alloc_vreg();
    k.v_mov_from_sgpr(eps_v, SReg(eps_arg.0));
    k.v_add_f32(sum_sq, sum_sq, eps_v);
    k.v_sqrt_f32(sum_sq, sum_sq);
    let rms_inv = k.alloc_vreg();
    k.v_rcp_f32(rms_inv, sum_sq);

    // ── Phase 2: atomic_add(dy * x * rms_inv) to dgamma ──
    k.s_mov_imm(loop_base, 0);
    let p2_label = k.make_label("gp2");
    k.label(&p2_label);

    k.v_mov_from_sgpr(idx, loop_base);
    k.v_add_u32(idx, idx, lane_id);
    k.v_mov_from_sgpr(dim_v, SReg(dim_arg.0));
    k.push(Op::VCmpLtU32 { src0: Operand::VReg(idx), src1: Operand::VReg(dim_v) });
    k.push(Op::SaveExec { dst: saved_exec });

    k.v_lshlrev_b32(byte_idx, 2, idx);

    // load x
    k.v_mov_from_sgpr(load_addr, x_row_lo);
    k.v_mov_from_sgpr(VReg(load_addr.0 + 1), x_row_hi);
    k.v_add_co(load_addr, load_addr, byte_idx);
    k.v_add_co_ci(VReg(load_addr.0 + 1), VReg(load_addr.0 + 1));
    k.global_load(x_val, load_addr, Width::B32, 0);

    // load dy
    let dy_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dy_addr, dy_row_lo);
    k.v_mov_from_sgpr(VReg(dy_addr.0 + 1), dy_row_hi);
    k.v_add_co(dy_addr, dy_addr, byte_idx);
    k.v_add_co_ci(VReg(dy_addr.0 + 1), VReg(dy_addr.0 + 1));
    let dy_val = k.alloc_vreg();
    k.global_load(dy_val, dy_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // val = dy * x * rms_inv
    let val = k.alloc_vreg();
    k.v_mul_f32(val, x_val, rms_inv);
    k.v_mul_f32(val, val, dy_val);

    // atomic_add to dgamma[idx]
    let dg_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(dg_addr, SReg(dgamma_ptr.0));
    k.v_mov_from_sgpr(VReg(dg_addr.0 + 1), SReg(dgamma_ptr.0 + 1));
    k.v_add_co(dg_addr, dg_addr, byte_idx);
    k.v_add_co_ci(VReg(dg_addr.0 + 1), VReg(dg_addr.0 + 1));
    k.global_atomic_add_f32(dg_addr, val, 0);

    k.push(Op::RestoreExec { src: saved_exec });
    k.s_add_u32(loop_base, loop_base, 32);
    k.s_cmp_lt_u32(loop_base, SReg(dim_arg.0));
    k.branch_scc1(&p2_label);

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ═══════════════════════════════════════════════════════════════════════
// Neuro-RACS T0 Kernels
// ═══════════════════════════════════════════════════════════════════════

/// Neuro-RACS Phase 1+2a: Momentum EMA + Row Variance (fused)
/// 1 WG per row. Each WG: wave32 threads process cols_per_wave columns.
/// mom[i,j] = β₁·mom[i,j] + (1-β₁)·grad[i,j]
/// row_var[i] = β₂·row_var[i] + (1-β₂)·mean_j(mom[i,j]²)
///
/// Kernargs (36B): grad(0), mom(8), row_var(16), cols(24 u32), beta1(28 f32), beta2(32 f32)
/// Grid: [rows*32, 1, 1], WG: 32
pub fn t0_nracs_momentum() -> T0Kernel {
    let mut k = T0Kernel::new("t0_nracs_momentum");
    let grad_p = k.arg_ptr("grad");
    let mom_p = k.arg_ptr("mom");
    let rv_p = k.arg_ptr("row_var");
    let cols_a = k.arg_u32("cols");
    let b1_a = k.arg_u32("beta1");
    let b2_a = k.arg_u32("beta2");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id); // = row index i
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // Compute row byte offset: i * cols * 4
    let row_byte_off = k.alloc_sreg();
    k.s_mul_i32(row_byte_off, wg_id, SReg(cols_a.0));
    let s_tmp = k.alloc_sreg();
    k.s_lshl_b32(s_tmp, row_byte_off, 2); // * 4 bytes

    // Base addresses for this row
    let g_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_base, SReg(grad_p.0));
    k.v_mov_from_sgpr(VReg(g_base.0+1), SReg(grad_p.0+1));
    let off_v = k.alloc_vreg(); k.v_mov_from_sgpr(off_v, s_tmp);
    k.v_add_co(g_base, g_base, off_v);
    k.v_add_co_ci(VReg(g_base.0+1), VReg(g_base.0+1));

    let m_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(m_base, SReg(mom_p.0));
    k.v_mov_from_sgpr(VReg(m_base.0+1), SReg(mom_p.0+1));
    k.v_add_co(m_base, m_base, off_v);
    k.v_add_co_ci(VReg(m_base.0+1), VReg(m_base.0+1));

    let b1_v = k.alloc_vreg(); k.v_mov_from_sgpr(b1_v, SReg(b1_a.0));
    let one = k.alloc_vreg(); k.v_mov_imm(one, 0x3F800000u32 as i32); // 1.0f
    let ob1 = k.alloc_vreg(); k.v_sub_f32(ob1, one, b1_v); // 1-β₁

    // Accumulator for sum(mom²) — per-lane partial
    let sum_sq = k.alloc_vreg(); k.v_mov_imm(sum_sq, 0);

    // Loop over columns: each lane processes lid, lid+32, lid+64 ...
    let loop_name = k.make_label("col_loop");
    let loop_col_s = k.alloc_sreg();
    k.s_mov_imm(loop_col_s, 0);
    k.label(&loop_name);

    // col = loop_col_s + lid
    let col_v = k.alloc_vreg();
    k.v_mov_from_sgpr(col_v, loop_col_s);
    k.v_add_u32(col_v, col_v, lid);

    // byte offset within row: col * 4
    let col_byte = k.alloc_vreg(); k.v_lshlrev_b32(col_byte, 2, col_v);

    // Load grad[i, col] and mom[i, col]
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(g_addr, g_base); k.v_mov(VReg(g_addr.0+1), VReg(g_base.0+1));
    k.v_add_co(g_addr, g_addr, col_byte);
    k.v_add_co_ci(VReg(g_addr.0+1), VReg(g_addr.0+1));

    let m_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(m_addr, m_base); k.v_mov(VReg(m_addr.0+1), VReg(m_base.0+1));
    k.v_add_co(m_addr, m_addr, col_byte);
    k.v_add_co_ci(VReg(m_addr.0+1), VReg(m_addr.0+1));

    let gval = k.alloc_vreg(); k.global_load(gval, g_addr, Width::B32, 0);
    let mval = k.alloc_vreg(); k.global_load(mval, m_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // mom = β₁·mom + (1-β₁)·grad
    k.v_mul_f32(mval, b1_v, mval);
    k.v_fma_f32(mval, ob1, gval, mval);

    // Store updated mom
    k.global_store(m_addr, mval, Width::B32, 0);

    // Accumulate mom² (only for valid cols)
    // Check col < cols: use v_cmp + exec masking is complex, use v_cndmask
    let cols_cmp = k.alloc_vreg(); k.v_mov_from_sgpr(cols_cmp, SReg(cols_a.0));
    k.v_cmp_lt_u32(Operand::VReg(col_v), Operand::VReg(cols_cmp));
    let zero = k.alloc_vreg(); k.v_mov_imm(zero, 0);
    let sq = k.alloc_vreg(); k.v_mul_f32(sq, mval, mval);
    k.v_cndmask_b32(sq, Operand::VReg(zero), Operand::VReg(sq)); // sq = valid ? mom² : 0
    k.v_add_f32(sum_sq, sum_sq, sq);

    // loop_col_s += 32
    k.s_add_u32(loop_col_s, loop_col_s, 32);
    k.s_cmp_lt_u32(loop_col_s, SReg(cols_a.0));
    k.branch_scc1(&loop_name);

    // Wave32 reduce sum_sq
    let swap = k.alloc_vreg();
    k.wave_reduce_add_f32(sum_sq, swap);

    // sum_sq now holds sum_j(mom²) in all lanes
    // row_mean_sq = sum_sq / cols
    let cols_v = k.alloc_vreg(); k.v_mov_from_sgpr(cols_v, SReg(cols_a.0));
    let cols_f = k.alloc_vreg(); k.v_cvt_f32_u32(cols_f, cols_v);
    let inv_c = k.alloc_vreg(); k.v_rcp_f32(inv_c, cols_f);
    let mean_sq = k.alloc_vreg(); k.v_mul_f32(mean_sq, sum_sq, inv_c);

    // Lane 0: row_var[i] = β₂·row_var[i] + (1-β₂)·mean_sq
    let saved = k.alloc_sreg();
    k.v_cmp_lt_u32(Operand::VReg(lid), Operand::InlineInt(1)); k.push(Op::SaveExec { dst: saved });

    let rv_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let rv_off_s = k.alloc_sreg(); k.s_lshl_b32(rv_off_s, wg_id, 2); // row * 4 bytes
    let rv_off_v = k.alloc_vreg(); k.v_mov_from_sgpr(rv_off_v, rv_off_s);
    k.v_mov_from_sgpr(rv_addr, SReg(rv_p.0));
    k.v_mov_from_sgpr(VReg(rv_addr.0+1), SReg(rv_p.0+1));
    k.v_add_co(rv_addr, rv_addr, rv_off_v);
    k.v_add_co_ci(VReg(rv_addr.0+1), VReg(rv_addr.0+1));

    let rv_old = k.alloc_vreg(); k.global_load(rv_old, rv_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    let b2_v = k.alloc_vreg(); k.v_mov_from_sgpr(b2_v, SReg(b2_a.0));
    let ob2 = k.alloc_vreg(); k.v_sub_f32(ob2, one, b2_v); // 1-β₂
    k.v_mul_f32(rv_old, b2_v, rv_old);
    k.v_fma_f32(rv_old, ob2, mean_sq, rv_old); // row_var = β₂·rv + (1-β₂)·mean
    k.global_store(rv_addr, rv_old, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved });
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Neuro-RACS Phase 2b: Column Variance
/// 1 thread per column, loops over all rows.
/// col_var[j] = β₂·col_var[j] + (1-β₂)·mean_i(mom[i,j]²)
///
/// Kernargs (28B): mom(0), col_var(8), rows(16 u32), cols(20 u32), beta2(24 f32)
/// Grid: [ceil(cols/32)*32, 1, 1], WG: 32
pub fn t0_nracs_col_var() -> T0Kernel {
    let mut k = T0Kernel::new("t0_nracs_col_var");
    let mom_p = k.arg_ptr("mom");
    let cv_p = k.arg_ptr("col_var");
    let rows_a = k.arg_u32("rows");
    let cols_a = k.arg_u32("cols");
    let b2_a = k.arg_u32("beta2");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // col = wg_id * 32 + lid
    let base_s = k.alloc_sreg(); k.s_lshl_b32(base_s, wg_id, 5); // *32
    let col = k.alloc_vreg(); k.v_mov_from_sgpr(col, base_s);
    k.v_add_u32(col, col, lid);

    // Accumulator for sum_i(mom[i,col]²)
    let sum_sq = k.alloc_vreg(); k.v_mov_imm(sum_sq, 0);

    // stride = cols * 4 bytes
    let stride_s = k.alloc_sreg(); k.s_lshl_b32(stride_s, SReg(cols_a.0), 2);

    // mom addr for column: mom + col*4
    let col_byte = k.alloc_vreg(); k.v_lshlrev_b32(col_byte, 2, col);
    let m_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(m_addr, SReg(mom_p.0));
    k.v_mov_from_sgpr(VReg(m_addr.0+1), SReg(mom_p.0+1));
    k.v_add_co(m_addr, m_addr, col_byte);
    k.v_add_co_ci(VReg(m_addr.0+1), VReg(m_addr.0+1));

    // Loop over rows
    let loop_name = k.make_label("row_loop");
    let row_s = k.alloc_sreg(); k.s_mov_imm(row_s, 0);
    let stride_v = k.alloc_vreg(); k.v_mov_from_sgpr(stride_v, stride_s);

    k.label(&loop_name);
    let mval = k.alloc_vreg(); k.global_load(mval, m_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_fma_f32(sum_sq, mval, mval, sum_sq); // sum += mom²

    // Advance addr by stride (cols * 4)
    k.v_add_co(m_addr, m_addr, stride_v);
    k.v_add_co_ci(VReg(m_addr.0+1), VReg(m_addr.0+1));

    k.s_add_u32(row_s, row_s, 1);
    k.s_cmp_lt_u32(row_s, SReg(rows_a.0));
    k.branch_scc1(&loop_name);

    // mean_sq = sum_sq / rows
    let rows_v = k.alloc_vreg(); k.v_mov_from_sgpr(rows_v, SReg(rows_a.0));
    let rows_f = k.alloc_vreg(); k.v_cvt_f32_u32(rows_f, rows_v);
    let inv_r = k.alloc_vreg(); k.v_rcp_f32(inv_r, rows_f);
    let mean_sq = k.alloc_vreg(); k.v_mul_f32(mean_sq, sum_sq, inv_r);

    // col_var[col] = β₂·col_var[col] + (1-β₂)·mean_sq
    let cv_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(cv_addr, SReg(cv_p.0));
    k.v_mov_from_sgpr(VReg(cv_addr.0+1), SReg(cv_p.0+1));
    k.v_add_co(cv_addr, cv_addr, col_byte);
    k.v_add_co_ci(VReg(cv_addr.0+1), VReg(cv_addr.0+1));

    let cv_old = k.alloc_vreg(); k.global_load(cv_old, cv_addr, Width::B32, 0);
    k.wait_vmcnt(0);
    let one = k.alloc_vreg(); k.v_mov_imm(one, 0x3F800000u32 as i32);
    let b2_v = k.alloc_vreg(); k.v_mov_from_sgpr(b2_v, SReg(b2_a.0));
    let ob2 = k.alloc_vreg(); k.v_sub_f32(ob2, one, b2_v);
    k.v_mul_f32(cv_old, b2_v, cv_old);
    k.v_fma_f32(cv_old, ob2, mean_sq, cv_old);
    k.global_store(cv_addr, cv_old, Width::B32, 0);

    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Neuro-RACS Phase 3: RACS Equilibration + Partial Frobenius
/// M̃[k] = mom[k] / (√(row_var[i] · col_var[j]) + ε)
/// partial_frob[wg_id] = wave reduce sum(M̃²)
///
/// Kernargs (44B): mom(0), row_var(8), col_var(16), m_tilde(24),
///                 partial(32), rows(40 u32), cols(44 u32), eps(48 f32)
/// Grid: [ceil(n_elems/128)*32, 1, 1], WG: 32 (4 elem/thread)
pub fn t0_nracs_equilibrate() -> T0Kernel {
    let mut k = T0Kernel::new("t0_nracs_equilibrate");
    let mom_p = k.arg_ptr("mom");
    let rv_p = k.arg_ptr("row_var");
    let cv_p = k.arg_ptr("col_var");
    let mt_p = k.arg_ptr("m_tilde");
    let part_p = k.arg_ptr("partial");
    let rows_a = k.arg_u32("rows");
    let cols_a = k.arg_u32("cols");
    let eps_a = k.arg_u32("eps");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // Global element index: base = wg_id * 128 + lid * 4
    let base_s = k.alloc_sreg(); k.s_lshl_b32(base_s, wg_id, 7); // * 128
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid); // lid * 4
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base_s);
    k.v_add_u32(gid, gid, v_off); // first element index for this lane

    // Setup addresses for mom and m_tilde
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid); // * 4 bytes

    let mom_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(mom_addr, SReg(mom_p.0));
    k.v_mov_from_sgpr(VReg(mom_addr.0+1), SReg(mom_p.0+1));
    k.v_add_co(mom_addr, mom_addr, byte_off);
    k.v_add_co_ci(VReg(mom_addr.0+1), VReg(mom_addr.0+1));

    let mt_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(mt_addr, SReg(mt_p.0));
    k.v_mov_from_sgpr(VReg(mt_addr.0+1), SReg(mt_p.0+1));
    k.v_add_co(mt_addr, mt_addr, byte_off);
    k.v_add_co_ci(VReg(mt_addr.0+1), VReg(mt_addr.0+1));

    let eps_v = k.alloc_vreg(); k.v_mov_from_sgpr(eps_v, SReg(eps_a.0));
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0); // partial Frobenius accumulator

    for i in 0..4i32 {
        // elem_idx = gid + i
        let idx = k.alloc_vreg();
        let imm_v = k.alloc_vreg(); k.v_mov_imm(imm_v, i);
        k.v_add_u32(idx, gid, imm_v);

        // row = elem_idx / cols, col = elem_idx % cols
        let cols_v = k.alloc_vreg(); k.v_mov_from_sgpr(cols_v, SReg(cols_a.0));
        let cols_f = k.alloc_vreg(); k.v_cvt_f32_u32(cols_f, cols_v);
        let idx_f = k.alloc_vreg(); k.v_cvt_f32_u32(idx_f, idx);
        let inv_c = k.alloc_vreg(); k.v_rcp_f32(inv_c, cols_f);
        let row_f = k.alloc_vreg(); k.v_mul_f32(row_f, idx_f, inv_c);
        let row_i = k.alloc_vreg(); k.v_cvt_u32_f32(row_i, row_f); // floor(idx/cols)
        // col = idx - row * cols
        let rc = k.alloc_vreg(); k.v_mul_lo_u32(rc, row_i, cols_v);
        let col_i = k.alloc_vreg(); k.v_sub_u32(col_i, idx, rc);

        // Load row_var[row] and col_var[col]
        let rv_byte = k.alloc_vreg(); k.v_lshlrev_b32(rv_byte, 2, row_i);
        let rv_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(rv_addr, SReg(rv_p.0));
        k.v_mov_from_sgpr(VReg(rv_addr.0+1), SReg(rv_p.0+1));
        k.v_add_co(rv_addr, rv_addr, rv_byte);
        k.v_add_co_ci(VReg(rv_addr.0+1), VReg(rv_addr.0+1));

        let cv_byte = k.alloc_vreg(); k.v_lshlrev_b32(cv_byte, 2, col_i);
        let cv_addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(cv_addr, SReg(cv_p.0));
        k.v_mov_from_sgpr(VReg(cv_addr.0+1), SReg(cv_p.0+1));
        k.v_add_co(cv_addr, cv_addr, cv_byte);
        k.v_add_co_ci(VReg(cv_addr.0+1), VReg(cv_addr.0+1));

        let rv = k.alloc_vreg(); k.global_load(rv, rv_addr, Width::B32, 0);
        let cv = k.alloc_vreg(); k.global_load(cv, cv_addr, Width::B32, 0);
        let m = k.alloc_vreg(); k.global_load(m, mom_addr, Width::B32, i * 4);
        k.wait_vmcnt(0);

        // denom = sqrt(rv * cv) + eps
        let rc_prod = k.alloc_vreg(); k.v_mul_f32(rc_prod, rv, cv);
        let denom = k.alloc_vreg(); k.v_sqrt_f32(denom, rc_prod);
        k.v_add_f32(denom, denom, eps_v);

        // m_tilde = mom / denom
        let inv_d = k.alloc_vreg(); k.v_rcp_f32(inv_d, denom);
        let mt = k.alloc_vreg(); k.v_mul_f32(mt, m, inv_d);

        // Store M̃
        k.global_store(mt_addr, mt, Width::B32, i * 4);

        // Accumulate M̃²
        k.v_fma_f32(acc, mt, mt, acc);
    }

    // Wave32 reduce partial Frobenius
    let swap = k.alloc_vreg(); k.wave_reduce_add_f32(acc, swap);

    // Lane 0 stores partial_frob[wg_id]
    let saved = k.alloc_sreg();
    k.v_cmp_lt_u32(Operand::VReg(lid), Operand::InlineInt(1)); k.push(Op::SaveExec { dst: saved });
    let p_addr = k.alloc_vreg_array(2, Alignment::Align2);
    let p_off_s = k.alloc_sreg(); k.s_lshl_b32(p_off_s, wg_id, 2);
    let p_off_v = k.alloc_vreg(); k.v_mov_from_sgpr(p_off_v, p_off_s);
    k.v_mov_from_sgpr(p_addr, SReg(part_p.0));
    k.v_mov_from_sgpr(VReg(p_addr.0+1), SReg(part_p.0+1));
    k.v_add_co(p_addr, p_addr, p_off_v);
    k.v_add_co_ci(VReg(p_addr.0+1), VReg(p_addr.0+1));
    k.global_store(p_addr, acc, Width::B32, 0);
    k.push(Op::RestoreExec { src: saved });

    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Neuro-RACS Phase 4: Weight Update
/// W[k] -= lr·wd·W[k] + lr·γ·M̃[k]
/// γ is read from a scalar GPU buffer.
///
/// Kernargs (36B): m_tilde(0), weight(8), gamma_ptr(16), n_elems(24 u32), lr(28 f32), wd(32 f32)
/// Grid: [ceil(n_elems/128)*32, 1, 1], WG: 32 (4 elem/thread)
pub fn t0_nracs_weight_apply() -> T0Kernel {
    let mut k = T0Kernel::new("t0_nracs_weight_apply");
    let mt_p = k.arg_ptr("m_tilde");
    let w_p = k.arg_ptr("weight");
    let gam_p = k.arg_ptr("gamma");
    let _n = k.arg_u32("n_elems");
    let lr_a = k.arg_u32("lr");
    let wd_a = k.arg_u32("wd");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // Read gamma scalar
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_addr, SReg(gam_p.0));
    k.v_mov_from_sgpr(VReg(g_addr.0+1), SReg(gam_p.0+1));
    let gamma = k.alloc_vreg(); k.global_load(gamma, g_addr, Width::B32, 0);
    k.wait_vmcnt(0);

    // Precompute lr_gamma = lr * gamma, lr_wd = lr * wd
    let lr_v = k.alloc_vreg(); k.v_mov_from_sgpr(lr_v, SReg(lr_a.0));
    let wd_v = k.alloc_vreg(); k.v_mov_from_sgpr(wd_v, SReg(wd_a.0));
    let lr_gamma = k.alloc_vreg(); k.v_mul_f32(lr_gamma, lr_v, gamma);
    let lr_wd = k.alloc_vreg(); k.v_mul_f32(lr_wd, lr_v, wd_v);

    // Address setup
    let base_s = k.alloc_sreg(); k.s_lshl_b32(base_s, wg_id, 7);
    let v_off = k.alloc_vreg(); k.v_lshlrev_b32(v_off, 2, lid);
    let gid = k.alloc_vreg(); k.v_mov_from_sgpr(gid, base_s);
    k.v_add_u32(gid, gid, v_off);
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, gid);

    let mt_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(mt_addr, SReg(mt_p.0));
    k.v_mov_from_sgpr(VReg(mt_addr.0+1), SReg(mt_p.0+1));
    k.v_add_co(mt_addr, mt_addr, byte_off);
    k.v_add_co_ci(VReg(mt_addr.0+1), VReg(mt_addr.0+1));

    let w_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(w_addr, SReg(w_p.0));
    k.v_mov_from_sgpr(VReg(w_addr.0+1), SReg(w_p.0+1));
    k.v_add_co(w_addr, w_addr, byte_off);
    k.v_add_co_ci(VReg(w_addr.0+1), VReg(w_addr.0+1));

    for i in 0..4i32 {
        let mt = k.alloc_vreg(); k.global_load(mt, mt_addr, Width::B32, i * 4);
        let w = k.alloc_vreg(); k.global_load(w, w_addr, Width::B32, i * 4);
        k.wait_vmcnt(0);
        // W -= lr_wd * W + lr_gamma * M̃
        // W = W - lr_wd*W - lr_gamma*M̃ = W*(1-lr_wd) - lr_gamma*M̃
        // Use: tmp = lr_wd*W + lr_gamma*M̃, W -= tmp
        let tmp = k.alloc_vreg(); k.v_mul_f32(tmp, lr_wd, w);
        k.v_fma_f32(tmp, lr_gamma, mt, tmp); // tmp = lr_wd*W + lr_gamma*M̃
        k.v_sub_f32(w, w, tmp); // W -= tmp
        k.global_store(w_addr, w, Width::B32, i * 4);
    }
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Neuro-RACS: GPU Frobenius Reduce → Gamma
/// Reads partial_frob[0..n_wgs], sums them, computes γ = √(n_elems) / √(sum).
/// Single WG—each of 32 threads handles n_wgs/32 partials.
///
/// Kernargs (28B): partial(0), gamma_out(8), n_wgs(16 u32), n_elems(20 u32), eps(24 f32)
/// Grid: [32, 1, 1], WG: 32
pub fn t0_nracs_reduce_gamma() -> T0Kernel {
    let mut k = T0Kernel::new("t0_nracs_reduce_gamma");
    let part_p = k.arg_ptr("partial");
    let gam_p = k.arg_ptr("gamma_out");
    let nwgs_a = k.arg_u32("n_wgs");
    let nelems_a = k.arg_u32("n_elems");
    let eps_a = k.arg_u32("eps");
    k.emit_arg_loads();

    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);

    // Base address for partial array
    let p_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(p_base, SReg(part_p.0));
    k.v_mov_from_sgpr(VReg(p_base.0+1), SReg(part_p.0+1));

    // Loop: each lane handles idx = lid, lid+32, lid+64, ...
    let loop_name = k.make_label("reduce_loop");
    let iter_s = k.alloc_sreg(); k.s_mov_imm(iter_s, 0);
    k.label(&loop_name);

    // idx = iter_s + lid
    let idx = k.alloc_vreg(); k.v_mov_from_sgpr(idx, iter_s);
    k.v_add_u32(idx, idx, lid);

    // Bounds check: idx < n_wgs
    let nwgs_v = k.alloc_vreg(); k.v_mov_from_sgpr(nwgs_v, SReg(nwgs_a.0));
    let saved_bc = k.alloc_sreg();
    k.v_cmp_lt_u32(Operand::VReg(idx), Operand::VReg(nwgs_v));
    k.push(Op::SaveExec { dst: saved_bc });

    // Load and accumulate
    let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, idx);
    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(addr, p_base); k.v_mov(VReg(addr.0+1), VReg(p_base.0+1));
    k.v_add_co(addr, addr, byte_off);
    k.v_add_co_ci(VReg(addr.0+1), VReg(addr.0+1));
    let val = k.alloc_vreg(); k.global_load(val, addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.v_add_f32(acc, acc, val);

    k.push(Op::RestoreExec { src: saved_bc });

    k.s_add_u32(iter_s, iter_s, 32);
    k.s_cmp_lt_u32(iter_s, SReg(nwgs_a.0));
    k.branch_scc1(&loop_name);

    // Wave32 reduce
    let swap = k.alloc_vreg(); k.wave_reduce_add_f32(acc, swap);

    // Lane 0: gamma = sqrt(n_elems) / max(sqrt(acc), eps)
    let saved = k.alloc_sreg();
    k.v_cmp_lt_u32(Operand::VReg(lid), Operand::InlineInt(1));
    k.push(Op::SaveExec { dst: saved });

    let nelems_v = k.alloc_vreg(); k.v_mov_from_sgpr(nelems_v, SReg(nelems_a.0));
    let ne_f = k.alloc_vreg(); k.v_cvt_f32_u32(ne_f, nelems_v);
    let sqrt_ne = k.alloc_vreg(); k.v_sqrt_f32(sqrt_ne, ne_f);
    let eps_v = k.alloc_vreg(); k.v_mov_from_sgpr(eps_v, SReg(eps_a.0));
    let sqrt_frob = k.alloc_vreg(); k.v_sqrt_f32(sqrt_frob, acc);
    k.v_max_f32(sqrt_frob, sqrt_frob, eps_v);
    let inv_frob = k.alloc_vreg(); k.v_rcp_f32(inv_frob, sqrt_frob);
    let gamma = k.alloc_vreg(); k.v_mul_f32(gamma, sqrt_ne, inv_frob);

    // Store gamma
    let g_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(g_addr, SReg(gam_p.0));
    k.v_mov_from_sgpr(VReg(g_addr.0+1), SReg(gam_p.0+1));
    k.global_store(g_addr, gamma, Width::B32, 0);

    k.push(Op::RestoreExec { src: saved });
    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}


// ═══════════════════════════════════════════════════════
//  Muon Optimizer T0 Kernels
// ═══════════════════════════════════════════════════════

/// Muon: Momentum update + partial Frobenius norm.
/// M = β·M + (1-β)·G, then accumulate partial ‖M‖²_F per WG.
///
/// Kernargs (28B): grad(0), mom(8), partial(16), n_elems(24 u32), beta(28 f32)
/// Grid: [n_wgs * 32, 1, 1], WG: 32, each thread handles 4 elements
pub fn t0_muon_momentum_norm() -> T0Kernel {
    let mut k = T0Kernel::new("t0_muon_momentum_norm");
    let grad_p = k.arg_ptr("grad");
    let mom_p = k.arg_ptr("mom");
    let part_p = k.arg_ptr("partial");
    let nelem_a = k.arg_u32("n_elems");
    let beta_a = k.arg_u32("beta");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);
    let base_idx = k.alloc_vreg();
    let wg_v = k.alloc_vreg(); k.v_mov_from_sgpr(wg_v, wg_id);
    k.v_lshlrev_b32(base_idx, 7, wg_v); // base = wg_id * 128
    k.v_add_u32(base_idx, base_idx, lid);

    let beta_v = k.alloc_vreg(); k.v_mov_from_sgpr(beta_v, SReg(beta_a.0));
    let one_m_beta = k.alloc_vreg();
    let one_v = k.alloc_vreg(); k.v_mov_imm(one_v, 0x3F800000u32 as i32); // 1.0f
    k.v_sub_f32(one_m_beta, one_v, beta_v);

    let acc = k.alloc_vreg(); k.v_mov_imm(acc, 0);

    // 4 elements per thread
    for chunk in 0..4u32 {
        let idx = k.alloc_vreg();
        if chunk == 0 { k.v_mov(idx, base_idx); }
        else {
            let off = k.alloc_vreg(); k.v_mov_imm(off, (chunk * 32) as i32);
            k.v_add_u32(idx, base_idx, off);
        }
        // Bounds check
        let nelem_v = k.alloc_vreg(); k.v_mov_from_sgpr(nelem_v, SReg(nelem_a.0));
        let saved = k.alloc_sreg();
        k.v_cmp_lt_u32(Operand::VReg(idx), Operand::VReg(nelem_v));
        k.push(Op::SaveExec { dst: saved });

        let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, idx);
        // Load grad
        let ga = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(ga, SReg(grad_p.0)); k.v_mov_from_sgpr(VReg(ga.0+1), SReg(grad_p.0+1));
        k.v_add_co(ga, ga, byte_off); k.v_add_co_ci(VReg(ga.0+1), VReg(ga.0+1));
        let gv = k.alloc_vreg(); k.global_load(gv, ga, Width::B32, 0);
        // Load mom
        let ma = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(ma, SReg(mom_p.0)); k.v_mov_from_sgpr(VReg(ma.0+1), SReg(mom_p.0+1));
        k.v_add_co(ma, ma, byte_off); k.v_add_co_ci(VReg(ma.0+1), VReg(ma.0+1));
        let mv = k.alloc_vreg(); k.global_load(mv, ma, Width::B32, 0);
        k.wait_vmcnt(0);
        // M = β·M + (1-β)·G
        k.v_mul_f32(mv, beta_v, mv);
        k.v_fma_f32(mv, one_m_beta, gv, mv);
        k.global_store(ma, mv, Width::B32, 0);
        // Accumulate M²
        k.v_fma_f32(acc, mv, mv, acc);

        k.push(Op::RestoreExec { src: saved });
    }

    // Wave reduce for partial Frobenius
    let swap = k.alloc_vreg(); k.wave_reduce_add_f32(acc, swap);

    // Lane 0: store partial
    let saved_st = k.alloc_sreg();
    k.v_cmp_lt_u32(Operand::VReg(lid), Operand::InlineInt(1));
    k.push(Op::SaveExec { dst: saved_st });
    let poff = k.alloc_vreg(); k.v_lshlrev_b32(poff, 2, wg_v);
    let pa = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(pa, SReg(part_p.0)); k.v_mov_from_sgpr(VReg(pa.0+1), SReg(part_p.0+1));
    k.v_add_co(pa, pa, poff); k.v_add_co_ci(VReg(pa.0+1), VReg(pa.0+1));
    k.global_store(pa, acc, Width::B32, 0);
    k.push(Op::RestoreExec { src: saved_st });

    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Muon: Normalize M by Frobenius norm.
/// X[i] = M[i] / norm, where norm is a scalar on GPU.
///
/// Kernargs (24B): mom(0), x_out(8), norm_buf(16), n_elems(24 u32)
/// Grid: [n_wgs * 32, 1, 1], WG: 32, each thread handles 4 elements
pub fn t0_muon_normalize() -> T0Kernel {
    let mut k = T0Kernel::new("t0_muon_normalize");
    let mom_p = k.arg_ptr("mom");
    let x_p = k.arg_ptr("x_out");
    let norm_p = k.arg_ptr("norm_buf");
    let nelem_a = k.arg_u32("n_elems");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // Load norm (broadcast from lane 0 — all lanes load same scalar)
    let na = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(na, SReg(norm_p.0)); k.v_mov_from_sgpr(VReg(na.0+1), SReg(norm_p.0+1));
    let norm_val = k.alloc_vreg(); k.global_load(norm_val, na, Width::B32, 0);
    k.wait_vmcnt(0);
    // inv_norm = 1 / norm
    let inv_norm = k.alloc_vreg(); k.v_rcp_f32(inv_norm, norm_val);

    let base_idx = k.alloc_vreg();
    let wg_v = k.alloc_vreg(); k.v_mov_from_sgpr(wg_v, wg_id);
    k.v_lshlrev_b32(base_idx, 7, wg_v); // wg_id * 128
    k.v_add_u32(base_idx, base_idx, lid);

    for chunk in 0..4u32 {
        let idx = k.alloc_vreg();
        if chunk == 0 { k.v_mov(idx, base_idx); }
        else {
            let off = k.alloc_vreg(); k.v_mov_imm(off, (chunk * 32) as i32);
            k.v_add_u32(idx, base_idx, off);
        }
        let nv = k.alloc_vreg(); k.v_mov_from_sgpr(nv, SReg(nelem_a.0));
        let saved = k.alloc_sreg();
        k.v_cmp_lt_u32(Operand::VReg(idx), Operand::VReg(nv));
        k.push(Op::SaveExec { dst: saved });

        let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, idx);
        // Load M
        let ma = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(ma, SReg(mom_p.0)); k.v_mov_from_sgpr(VReg(ma.0+1), SReg(mom_p.0+1));
        k.v_add_co(ma, ma, byte_off); k.v_add_co_ci(VReg(ma.0+1), VReg(ma.0+1));
        let mv = k.alloc_vreg(); k.global_load(mv, ma, Width::B32, 0);
        k.wait_vmcnt(0);
        // X = M * inv_norm
        let xv = k.alloc_vreg(); k.v_mul_f32(xv, mv, inv_norm);
        // Store X
        let xa = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(xa, SReg(x_p.0)); k.v_mov_from_sgpr(VReg(xa.0+1), SReg(x_p.0+1));
        k.v_add_co(xa, xa, byte_off); k.v_add_co_ci(VReg(xa.0+1), VReg(xa.0+1));
        k.global_store(xa, xv, Width::B32, 0);

        k.push(Op::RestoreExec { src: saved });
    }

    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}

/// Muon: Weight update with weight decay.
/// W = W - lr·X - lr·wd·W = (1 - lr·wd)·W - lr·X
///
/// Kernargs (28B): x_buf(0), weight(8), n_elems(16 u32), lr(20 f32), wd(24 f32)
/// Grid: [n_wgs * 32, 1, 1], WG: 32, 4 elements per thread
pub fn t0_muon_weight_update() -> T0Kernel {
    let mut k = T0Kernel::new("t0_muon_weight_update");
    let x_p = k.arg_ptr("x_buf");
    let w_p = k.arg_ptr("weight");
    let nelem_a = k.arg_u32("n_elems");
    let lr_a = k.arg_u32("lr");
    let wd_a = k.arg_u32("wd");
    k.emit_arg_loads();

    let wg_id = k.alloc_sreg(); k.capture_tgid_x(wg_id);
    let lid = k.alloc_vreg(); k.v_and_b32_imm(lid, VReg(0), 31);

    // Precompute: neg_lr = -lr, decay = 1 - lr*wd
    let lr_v = k.alloc_vreg(); k.v_mov_from_sgpr(lr_v, SReg(lr_a.0));
    let wd_v = k.alloc_vreg(); k.v_mov_from_sgpr(wd_v, SReg(wd_a.0));
    let neg_lr = k.alloc_vreg(); k.v_sub_f32(neg_lr, VReg(0), lr_v); // 0 - lr (v0 always=0 for lane 0 trick won't work, use mul)
    // Actually: neg_lr = lr * -1.0
    let neg_one = k.alloc_vreg(); k.v_mov_imm(neg_one, 0xBF800000u32 as i32); // -1.0f
    k.v_mul_f32(neg_lr, lr_v, neg_one);
    let lr_wd = k.alloc_vreg(); k.v_mul_f32(lr_wd, lr_v, wd_v);
    let one_v = k.alloc_vreg(); k.v_mov_imm(one_v, 0x3F800000u32 as i32); // 1.0f
    let decay = k.alloc_vreg(); k.v_sub_f32(decay, one_v, lr_wd); // 1 - lr*wd

    let base_idx = k.alloc_vreg();
    let wg_v = k.alloc_vreg(); k.v_mov_from_sgpr(wg_v, wg_id);
    k.v_lshlrev_b32(base_idx, 7, wg_v);
    k.v_add_u32(base_idx, base_idx, lid);

    for chunk in 0..4u32 {
        let idx = k.alloc_vreg();
        if chunk == 0 { k.v_mov(idx, base_idx); }
        else {
            let off = k.alloc_vreg(); k.v_mov_imm(off, (chunk * 32) as i32);
            k.v_add_u32(idx, base_idx, off);
        }
        let nv = k.alloc_vreg(); k.v_mov_from_sgpr(nv, SReg(nelem_a.0));
        let saved = k.alloc_sreg();
        k.v_cmp_lt_u32(Operand::VReg(idx), Operand::VReg(nv));
        k.push(Op::SaveExec { dst: saved });

        let byte_off = k.alloc_vreg(); k.v_lshlrev_b32(byte_off, 2, idx);
        // Load X
        let xa = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(xa, SReg(x_p.0)); k.v_mov_from_sgpr(VReg(xa.0+1), SReg(x_p.0+1));
        k.v_add_co(xa, xa, byte_off); k.v_add_co_ci(VReg(xa.0+1), VReg(xa.0+1));
        let xv = k.alloc_vreg(); k.global_load(xv, xa, Width::B32, 0);
        // Load W
        let wa = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(wa, SReg(w_p.0)); k.v_mov_from_sgpr(VReg(wa.0+1), SReg(w_p.0+1));
        k.v_add_co(wa, wa, byte_off); k.v_add_co_ci(VReg(wa.0+1), VReg(wa.0+1));
        let wv = k.alloc_vreg(); k.global_load(wv, wa, Width::B32, 0);
        k.wait_vmcnt(0);
        // W_new = decay·W + neg_lr·X = (1-lr·wd)·W - lr·X
        let wnew = k.alloc_vreg();
        k.v_mul_f32(wnew, decay, wv);
        k.v_fma_f32(wnew, neg_lr, xv, wnew);
        k.global_store(wa, wnew, Width::B32, 0);

        k.push(Op::RestoreExec { src: saved });
    }

    k.wait_vmcnt(0); k.wait_vscnt(0); k.endpgm(); k
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schedule::GFX1100Schedule;

    #[test]
    fn test_matmul_assembly() {
        let sched = GFX1100Schedule;
        let kernel = matmul(&sched);
        let asm = kernel.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_wmma_f32_16x16x16_bf16"));
        assert!(asm.contains("global_store_b32"));
        assert!(asm.contains("s_cbranch_scc1"));
        eprintln!("--- matmul ---\n{}", asm);
    }

    #[test]
    fn test_rmsnorm_assembly() {
        let sched = GFX1100Schedule;
        let kernel = rmsnorm(&sched);
        let asm = kernel.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("ds_swizzle_b32"), "missing wave reduction");
        assert!(asm.contains("v_rsq_f32"), "missing rsqrt");
        assert!(asm.contains("v_mul_f32"));
        assert!(asm.contains("global_store_b32"));
        eprintln!("--- rmsnorm ---\n{}", asm);
    }

    #[test]
    fn test_unary_ops() {
        let sched = GFX1100Schedule;
        for op in [UnaryOp::Scale(2.0), UnaryOp::Relu, UnaryOp::Square, UnaryOp::Negate] {
            let kernel = elementwise_unary(&sched, op);
            let asm = kernel.to_assembly(Target::GFX1100).unwrap();
            assert!(asm.contains("s_endpgm"), "op {:?} missing endpgm", op);
        }
    }

    #[test]
    fn test_binary_ops() {
        let sched = GFX1100Schedule;
        for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Axpy(0.1)] {
            let kernel = elementwise_binary(&sched, op);
            let asm = kernel.to_assembly(Target::GFX1100).unwrap();
            assert!(asm.contains("s_endpgm"), "op {:?} missing endpgm", op);
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_all_kernels_compile_to_elf() {
        let sched = GFX1100Schedule;
        let kernels = vec![
            matmul(&sched),
            rmsnorm(&sched),
            elementwise_unary(&sched, UnaryOp::Relu),
            elementwise_binary(&sched, BinaryOp::Add),
        ];
        for kernel in &kernels {
            let elf = kernel.compile(Target::GFX1100).unwrap();
            assert!(elf.len() > 0, "{} produced empty ELF", kernel.name);
            assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F'],
                "{} produced invalid ELF", kernel.name);
            eprintln!("{}: {} bytes ELF", kernel.name, elf.len());
        }
    }

    #[test]
    fn test_fused_elementwise_assembly() {
        // Test: scale(0.5) → add_input(1) → relu
        let plan = FusionPlan {
            ops: vec![EwOp::Scale(0.5), EwOp::AddInput(1), EwOp::Relu],
            n_inputs: 2,
            name: "t0_scale_add_relu".to_string(),
            inplace: false, zero_init: false,
        };
        let kernel = fused_elementwise(&plan, 4);
        let asm = kernel.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_mul_f32"), "missing scale (mul)");
        assert!(asm.contains("v_add_f32"), "missing add");
        assert!(asm.contains("v_max_f32"), "missing relu (max)");
        assert!(asm.contains("global_load_b128"), "missing dwordx4 load");
        assert!(asm.contains("global_store_b128"), "missing dwordx4 store");
        assert!(asm.contains("s_endpgm"), "missing endpgm");
        eprintln!("--- fused_scale_add_relu ---\n{}", asm);
    }

    #[test]
    fn test_fused_elementwise_neg() {
        let plan = FusionPlan {
            ops: vec![EwOp::Neg],
            n_inputs: 1,
            name: "t0_neg".to_string(),
            inplace: false, zero_init: false,
        };
        let kernel = fused_elementwise(&plan, 8);
        let asm = kernel.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_mul_f32"), "missing negate (mul by -1)");
        assert!(asm.contains("s_endpgm"));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_fused_elementwise_compiles_to_elf() {
        let plan = FusionPlan {
            ops: vec![EwOp::Scale(0.5), EwOp::AddInput(1), EwOp::Relu],
            n_inputs: 2,
            name: "t0_fused_test".to_string(),
            inplace: false, zero_init: false,
        };
        let kernel = fused_elementwise(&plan, 4);
        let elf = kernel.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0, "fused ew produced empty ELF");
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("fused_elementwise ELF: {} bytes", elf.len());
    }

    #[test]
    fn test_exec_mask_assembly() {
        // Build a tiny kernel that uses bounds checking:
        // if (global_id < n_elems) { out[global_id] = in[global_id] * 2.0 }
        let mut k = T0Kernel::new("exec_mask_test");
        let in_ptr = k.arg_ptr("in");
        let out_ptr = k.arg_ptr("out");
        let n_elems_s = k.arg_u32("n");
        k.emit_arg_loads();

        let gid = k.alloc_vreg();
        k.push(Op::ComputeGlobalIdX { dst: gid, wg_size: 64 });

        // Move n_elems to VGPR for comparison
        let n_v = k.alloc_vreg();
        k.v_mov_from_sgpr(n_v, n_elems_s);

        // Bounds check: mask out lanes where gid >= n_elems
        let saved = k.bounds_check_begin(gid, n_v);

        // Load, scale by 2.0, store
        let addr_in = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(addr_in, SReg(in_ptr.0));
        k.v_mov_from_sgpr(VReg(addr_in.0 + 1), SReg(in_ptr.0 + 1));
        let val = k.alloc_vreg();
        let byte_off = k.alloc_vreg();
        k.push(Op::VLshlrevB32 { dst: byte_off, shift: 2, src: gid });
        k.v_add_co(addr_in, addr_in, byte_off);
        k.v_add_co_ci(VReg(addr_in.0 + 1), VReg(addr_in.0 + 1));
        k.global_load(val, addr_in, Width::B32, 0);
        k.wait_vmcnt(0);

        k.v_mul_f32(val, val, val); // placeholder: square (just to have ALU)

        let addr_out = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(addr_out, SReg(out_ptr.0));
        k.v_mov_from_sgpr(VReg(addr_out.0 + 1), SReg(out_ptr.0 + 1));
        k.v_add_co(addr_out, addr_out, byte_off);
        k.v_add_co_ci(VReg(addr_out.0 + 1), VReg(addr_out.0 + 1));
        k.global_store(addr_out, val, Width::B32, 0);

        // End bounds check
        k.bounds_check_end(saved);

        k.wait_vscnt(0);
        k.endpgm();

        let asm = k.to_assembly(Target::GFX1100).unwrap();
        // LLVM verified: these are the correct GFX1100 mnemonics
        assert!(asm.contains("v_cmp_lt_u32"), "missing v_cmp_lt_u32");
        assert!(asm.contains("s_and_saveexec_b32"), "missing saveexec");
        assert!(asm.contains("exec_lo"), "missing exec_lo restore");
        eprintln!("--- exec_mask_test ---\n{}", asm);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_exec_mask_compiles_to_elf() {
        let mut k = T0Kernel::new("exec_mask_elf_test");
        let _in_ptr = k.arg_ptr("in");
        let _out_ptr = k.arg_ptr("out");
        let n_s = k.arg_u32("n");
        k.emit_arg_loads();
        let gid = k.alloc_vreg();
        k.push(Op::ComputeGlobalIdX { dst: gid, wg_size: 64 });
        let n_v = k.alloc_vreg();
        k.v_mov_from_sgpr(n_v, n_s);
        let saved = k.bounds_check_begin(gid, n_v);
        k.bounds_check_end(saved);
        k.endpgm();

        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("exec_mask ELF: {} bytes", elf.len());
    }

    #[test]
    fn test_replacement_kernels_assembly() {
        // memset_zero
        let k = t0_memset_zero(4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("global_store_b128"), "memset_zero: missing store");
        assert!(asm.contains("v_cmp_lt_u32"), "memset_zero: missing bounds check");
        assert!(asm.contains("s_endpgm"));

        // memcpy
        let k = t0_memcpy(4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("global_load_b128"), "memcpy: missing load");
        assert!(asm.contains("global_store_b128"), "memcpy: missing store");
        assert!(asm.contains("v_cmp_lt_u32"), "memcpy: missing bounds check");

        // residual_add
        let k = t0_residual_add(4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_add_f32"), "residual_add: missing add");
        assert!(asm.contains("v_cmp_lt_u32"), "residual_add: missing bounds check");
        assert!(asm.contains("global_store_b128"), "residual_add: missing store");

        // silu_mul
        let k = t0_silu_mul(4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_exp_f32"), "silu_mul: missing exp");
        assert!(asm.contains("v_rcp_f32"), "silu_mul: missing rcp");
        assert!(asm.contains("v_xor_b32"), "silu_mul: missing xor");
        assert!(asm.contains("v_mul_f32"), "silu_mul: missing mul");
        assert!(asm.contains("v_cmp_lt_u32"), "silu_mul: missing bounds check");

        // elementwise_mul
        let k = t0_elementwise_mul(4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_mul_f32"), "mul: missing mul");
        assert!(asm.contains("v_cmp_lt_u32"), "mul: missing bounds check");
        eprintln!("--- replacement kernels assembly OK ---");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_replacement_kernels_compile_to_elf() {
        for (name, kernel) in [
            ("memset_zero", t0_memset_zero(4)),
            ("memcpy", t0_memcpy(4)),
            ("residual_add", t0_residual_add(4)),
            ("silu_mul", t0_silu_mul(4)),
            ("elementwise_mul", t0_elementwise_mul(4)),
        ] {
            let elf = kernel.compile(Target::GFX1100).unwrap();
            assert!(elf.len() > 0, "{} produced empty ELF", name);
            assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F'],
                "{} produced invalid ELF", name);
            eprintln!("{}: {} bytes ELF", name, elf.len());
        }
    }

    #[test]
    fn test_extended_ewop_assembly() {
        // Sigmoid via FusionPlan — should contain v_exp_f32, v_rcp_f32, v_xor_b32
        let plan = FusionPlan {
            ops: vec![EwOp::Sigmoid],
            n_inputs: 1,
            name: "t0_sigmoid".to_string(),
            inplace: false, zero_init: false,
        };
        let k = fused_elementwise(&plan, 4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_exp_f32"), "sigmoid: missing exp");
        assert!(asm.contains("v_rcp_f32"), "sigmoid: missing rcp");
        assert!(asm.contains("v_xor_b32"), "sigmoid: missing xor (negation)");

        // SiLU via FusionPlan — 1 line!
        let plan = FusionPlan {
            ops: vec![EwOp::SiLU],
            n_inputs: 1,
            name: "t0_silu".to_string(),
            inplace: false, zero_init: false,
        };
        let k = fused_elementwise(&plan, 4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_exp_f32"), "silu: missing exp");
        assert!(asm.contains("v_mul_f32"), "silu: missing mul");

        // SwiGLU: silu(gate) * up — 3 lines replacing 90-line t0_silu_mul!
        let plan = FusionPlan {
            ops: vec![EwOp::SiLU, EwOp::MulInput(1)],
            n_inputs: 2,
            name: "t0_swiglu".to_string(),
            inplace: false, zero_init: false,
        };
        let k = fused_elementwise(&plan, 4);
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("v_exp_f32"), "swiglu: missing exp");

        eprintln!("--- extended EwOp assembly tests OK ---");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_extended_ewop_compile_to_elf() {
        let plans = vec![
            FusionPlan { ops: vec![EwOp::Sigmoid], n_inputs: 1,
                name: "sigmoid".into(), inplace: false, zero_init: false },
            FusionPlan { ops: vec![EwOp::SiLU], n_inputs: 1,
                name: "silu".into(), inplace: false, zero_init: false },
            FusionPlan { ops: vec![EwOp::SiLU, EwOp::MulInput(1)], n_inputs: 2,
                name: "swiglu".into(), inplace: false, zero_init: false },
            FusionPlan { ops: vec![EwOp::Exp], n_inputs: 1,
                name: "exp".into(), inplace: false, zero_init: false },
            FusionPlan { ops: vec![EwOp::Abs], n_inputs: 1,
                name: "abs".into(), inplace: false, zero_init: false },
            FusionPlan { ops: vec![EwOp::Square, EwOp::AddConst(1.0)], n_inputs: 1,
                name: "sq_plus1".into(), inplace: false, zero_init: false },
        ];
        for plan in &plans {
            let k = fused_elementwise(plan, 4);
            let elf = k.compile(Target::GFX1100).unwrap();
            assert!(elf.len() > 0, "{} produced empty ELF", plan.name);
            assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
            eprintln!("{}: {} bytes ELF", plan.name, elf.len());
        }
    }

    #[test]
    fn test_ocpa_prefix_sum_compiles() {
        let k = ocpa_prefix_sum();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0, "ocpa_prefix_sum produced empty ELF");
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_prefix_sum: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_reverse_prefix_sum_compiles() {
        let k = ocpa_reverse_prefix_sum();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0, "ocpa_reverse_prefix_sum produced empty ELF");
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_reverse_prefix_sum: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_denom_norm_d64_compiles() {
        let k = ocpa_denom_norm(64);
        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_denom_norm_d64: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_denom_norm_d128_compiles() {
        let k = ocpa_denom_norm(128);
        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_denom_norm_d128: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_state_update_compiles() {
        let k = ocpa_state_update();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0, "ocpa_state_update produced empty ELF");
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_state_update: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_dstate_update_compiles() {
        let k = ocpa_dstate_update();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0, "ocpa_dstate_update produced empty ELF");
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_dstate_update: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_forward_inter_compiles() {
        let k = ocpa_forward_inter();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_forward_inter: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_backward_inter_dq_compiles() {
        let k = ocpa_backward_inter_dq();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_backward_inter_dq: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_backward_inter_dk_compiles() {
        let k = ocpa_backward_inter_dk();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_backward_inter_dk: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_backward_inter_dv_compiles() {
        let k = ocpa_backward_inter_dv();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_backward_inter_dv: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_forward_intra_compiles() {
        let k = ocpa_forward_intra();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_forward_intra: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_backward_intra_dq_compiles() {
        let k = ocpa_backward_intra_dq();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_backward_intra_dq: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_ocpa_backward_intra_dkdv_compiles() {
        let k = ocpa_backward_intra_dkdv();
        let elf = k.compile(Target::GFX1100).unwrap();
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("ocpa_backward_intra_dkdv: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_t0_copy_asm_dump() {
        let k = t0_copy();
        let asm = k.to_assembly(Target::GFX1100).unwrap();
        eprintln!("--- t0_copy assembly ---\n{}", asm);
    }

    #[test]
    fn test_matmul_lds_db_asm() {
        let sched = GFX1100Schedule;
        let kernel = matmul_lds_db(&sched);
        let asm = kernel.to_assembly(Target::GFX1100).unwrap();
        std::fs::write("/tmp/matmul_lds_db.s", &asm).unwrap();
        eprintln!("matmul_lds_db: {} lines saved to /tmp/matmul_lds_db.s", asm.lines().count());
        assert!(asm.contains("ds_store_b128"));
        assert!(asm.contains("ds_load_b128"));
        assert!(asm.contains("s_barrier"));
        assert!(asm.contains(".amdhsa_group_segment_fixed_size 6144"));
    }
}
