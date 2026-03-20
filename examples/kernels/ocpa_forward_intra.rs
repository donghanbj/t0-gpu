//! OCPA Forward Intra Kernel for GFX11 (RDNA3) — 1D Tile-Stealing Architecture
//!
//! Computes: O_intra_c = (CausalMask(Q_c @ K_c^T)) @ V_c
//! Accumulates into: O_c += O_intra_c via global_atomic_add_f32 (Zero-Atomic)
//!
//! KEY OPTIMIZATION: 1D Tile-Stealing eliminates triangular load imbalance.
//!   Old: Wave W loops t=0..=W → Wave 0 does 1 iter, Wave 15 does 16 → 46.8% waste
//!   New: 136 valid tiles in lower triangle mapped to 1D pool, 16 waves steal linearly
//!        Each wave does 8 or 9 iterations → near-perfect load balance
//!
//! Architecture:
//!   Phase A₀: 136 threads build (r,c) LUT in LDS (272 bytes, one-time cost)
//!   Phase A:  512 threads cooperatively load V_c into LDS ZCLT (132-byte padded)
//!   Phase B:  Each wave loads its first Q_slice
//!   Phase C:  1D Tile-Stealing loop: k = wave_id, k < 136, k += 16
//!             - LUT lookup → conditional Q_slice reload → K_tile load → WMMA
//!             - Causal mask on diagonal (r==c) → transpose → V-tile WMMA 
//!             - Atomic add O_acc to output row
//!
//! VGPR Budget: ≤ 168 (7 groups → 9 waves/SIMD → 2 WG per CU)
//!
//! Grid: (N_chunks, num_heads, 1)
//! WG:   (512, 1, 1) = 16 Waves
//!
//! Kernarg layout:
//!   Q_ptr(0), K_ptr(8), V_ptr(16), O_ptr(24), seq_len(32)

use crate::rdna3_asm::{Rdna3Assembler, gfx11};
use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

const fn lds_zclt_size(c: u32) -> u32 { c * 132 }
const fn lds_lut_base(c: u32) -> u32 { lds_zclt_size(c) }
const fn n_tile_rows(c: u32) -> u32 { c / 16 }
const fn n_tiles(c: u32) -> u32 { let r = n_tile_rows(c); r * (r + 1) / 2 }
const fn c_shift(c: u32) -> u32 {
    // log2(c) for power-of-2 chunk sizes
    match c { 32 => 5, 64 => 6, 128 => 7, 256 => 8, _ => 8 }
}

pub fn build_ocpa_forward_intra() -> AmdGpuCodeObject {
    build_ocpa_forward_intra_with_c(256)
}

pub fn build_ocpa_forward_intra_c64() -> AmdGpuCodeObject {
    build_ocpa_forward_intra_with_c(64)
}

pub fn build_ocpa_forward_intra_c32() -> AmdGpuCodeObject {
    build_ocpa_forward_intra_with_c(32)
}

pub fn build_ocpa_forward_intra_c128() -> AmdGpuCodeObject {
    build_ocpa_forward_intra_with_c(128)
}

fn build_ocpa_forward_intra_with_c(c_chunk: u32) -> AmdGpuCodeObject {
    let mut asm = Rdna3Assembler::new();

    // ========================================================================
    // 1. System parameter capture & Mega-Wave topology
    // ========================================================================
    asm.emit(gfx11::s_mov_b32(20, 2));  // s20 = chunk_id (TGID.x)
    asm.emit(gfx11::s_mov_b32(21, 3));  // s21 = head_id  (TGID.y)

    asm.emit2(gfx11::s_load_dwordx2(2, 0, 0));    // s[2:3]   = Q_ptr
    asm.emit2(gfx11::s_load_dwordx2(4, 0, 8));    // s[4:5]   = K_ptr
    asm.emit2(gfx11::s_load_dwordx2(6, 0, 16));   // s[6:7]   = V_ptr
    asm.emit2(gfx11::s_load_dwordx2(8, 0, 24));   // s[8:9]   = O_ptr
    asm.emit2(gfx11::s_load_dword(10, 0, 32));    // s10      = seq_len
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    asm.emit(gfx11::v_mov_b32(32, 0));             // v32 = thread_id (0..511)
    asm.emit(gfx11::v_lshrrev_b32(33, 5, 32));     // v33 = wave_id (0..15)
    asm.emit(gfx11::v_and_b32_imm(34, 32, 31));    // v34 = lane_id (0..31)
    asm.emit(gfx11::v_and_b32_imm(35, 32, 15));    // v35 = L_mod_16 = lane_row

    // base_row = head_id * seq_len + chunk_id * C
    asm.emit(gfx11::s_mul_i32(11, 21, 10));        // head_id * seq_len
    asm.emit(gfx11::s_lshl_b32(12, 20, c_shift(c_chunk) as u8)); // chunk_id * C
    asm.emit(gfx11::s_add_u32(13, 11, 12));        // s13 = base_row

    // ========================================================================
    // 2. Phase A₀: Build 1D→2D LUT (136 entries, only first 136 threads work)
    //    k → (r, c) using: r = floor((sqrt(8k+1) - 1) / 2), c = k - r*(r+1)/2
    //    Packed as u16: (r << 8) | c, stored at LDS[LUT_BASE + k*2]
    // ========================================================================
    // Inline constants for VOP3:
    // 1.0 = 0xF2, -1.0 = 0xF3, 0.5 = 0xF0, 8.0 = 0x41000000 (literal)
    
    // Check if thread_id < n_tiles(C)
    let total_tiles = n_tiles(c_chunk);
    if total_tiles <= 64 {
        asm.emit(gfx11::v_mov_b32_imm(0, total_tiles as i32));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
        asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));
    }
    asm.emit(gfx11::v_cmp_lt_u32(32, 0));           // VCC = (tid < n_tiles)
    let skip_lut_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_vccz(0));              // skip if tid >= 136

    // r = floor((sqrt(8*k + 1) - 1) / 2)
    asm.emit(gfx11::v_cvt_f32_u32(0, 32));           // v0 = float(k) where k = tid
    asm.emit3(gfx11::v_mul_f32_e64_literal(0, 0, 0x41000000)); // v0 *= 8.0
    asm.emit2(gfx11::v_add_f32_e64_inline(0, 0, 0xF2));  // v0 += 1.0
    asm.emit(gfx11::v_sqrt_f32(0, 0));                // v0 = sqrt(8k+1)
    asm.emit2(gfx11::v_add_f32_e64_inline(0, 0, 0xF3));  // v0 += -1.0 (subtract 1)
    asm.emit2(gfx11::v_mul_f32_e64_inline(0, 0, 0xF0));  // v0 *= 0.5
    asm.emit(gfx11::v_cvt_u32_f32(1, 0));             // v1 = r = floor(v0)

    // Correction: check if r*(r+1)/2 > k, if so r--
    // tri = r*(r+1)/2
    asm.emit(gfx11::v_add_u32_imm(2, 1, 1));          // v2 = r + 1
    asm.emit2(gfx11::v_mul_lo_u32(2, 1, 2));          // v2 = r * (r+1)
    asm.emit(gfx11::v_lshrrev_b32(2, 1, 2));          // v2 = r*(r+1)/2 = tri
    
    // if tri > k: r--  (floating point rounding fix)
    asm.emit(gfx11::v_cmp_lt_u32(32, 2));              // VCC = (k < tri) → r was too big
    let skip_fix_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_vccz(0));                // skip if k >= tri (r is correct)
    asm.emit(gfx11::v_add_u32_imm(1, 1, 0));          // v1 = r (no-op, we need r-1)
    // Actually: we need to subtract 1. v_add_u32 with -1? Use v_sub_u32 or literal
    // v_add_u32 doesn't have -1 inline. Use the simple approach: recalculate
    asm.emit2(gfx11::s_mov_b32_literal(14, 0xFFFFFFFF));
    asm.emit(gfx11::v_mov_b32_from_sgpr(3, 14));
    asm.emit(gfx11::v_add_u32(1, 1, 3));               // v1 = r - 1

    // Recalculate tri for corrected r
    asm.emit(gfx11::v_add_u32_imm(2, 1, 1));
    asm.emit2(gfx11::v_mul_lo_u32(2, 1, 2));
    asm.emit(gfx11::v_lshrrev_b32(2, 1, 2));          // v2 = new tri

    let fix_done_pc = asm.current_pc();
    asm.patch_branch(skip_fix_pc, fix_done_pc);
    asm.emit(gfx11::s_mov_b32_imm(106, 0));           // clear VCC

    // c = k - tri
    asm.emit(gfx11::v_sub_f32(3, 32, 2));              // Oops, this is float sub, use integer
    // Actually need integer sub: v_sub_u32 doesn't exist as VOP2 in our assembler
    // Use: v3 = k - tri via v_add_u32(v3, v_tid, -tri) ... but -tri isn't simple
    // Better: compute v3 = k + (~tri + 1) = k - tri using two's complement
    // Simplest: use v_sub_nc_u32 which we don't have. Let me use VOP3 path.
    // Actually we can use: v3 = v32 - v2 via flip: v_add_u32(v3, v32, -v2)
    // But we don't have negate for integers. Let's just do it verbosely:
    // v0 = ~v2 (bitwise not via v_xor with -1)
    asm.emit2(gfx11::s_mov_b32_literal(14, 0xFFFFFFFF));
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));
    asm.emit(gfx11::v_xor_b32(0, 2, 0));               // v0 = ~tri
    asm.emit(gfx11::v_add_u32_imm(0, 0, 1));           // v0 = -tri (two's complement)
    asm.emit(gfx11::v_add_u32(3, 32, 0));               // v3 = k + (-tri) = k - tri = c

    // Pack (r << 8 | c) into v1
    asm.emit(gfx11::v_lshlrev_b32(1, 8, 1));           // v1 = r << 8
    asm.emit(gfx11::v_or_b32(1, 1, 3));                // v1 = (r << 8) | c

    // Write to LDS: LUT_BASE + k * 2
    asm.emit(gfx11::v_lshlrev_b32(0, 1, 32));           // v0 = k * 2
    let lut_base = lds_lut_base(c_chunk);
    if lut_base <= 64 {
        asm.emit(gfx11::v_add_u32_imm(0, 0, lut_base));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(14, lut_base));
        asm.emit(gfx11::v_mov_b32_from_sgpr(3, 14));
        asm.emit(gfx11::v_add_u32(0, 0, 3));               // v0 = LUT_BASE + k*2
    }
    asm.emit2(gfx11::ds_store_b16(0, 1, 0));            // LDS[v0] = packed u16

    let lut_done_pc = asm.current_pc();
    asm.patch_branch(skip_lut_pc, lut_done_pc);
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
    asm.emit(gfx11::s_mov_b32_imm(106, 0));            // clear VCC
    // Note: barrier will be after Phase A ZCLT load below

    // ========================================================================
    // 3. Phase A: 512 threads cooperatively ZCLT-load V_c to LDS (132-byte Padding)
    //    For C<256: out-of-range threads (row>=C) clamp their LDS write address
    //    to row & (C-1) to avoid out-of-bounds writes. HBM reads may be garbage
    //    but LDS is safe, and intra kernel never accesses those clamped rows.
    // ========================================================================
    // row = tid / 2, col_grp = tid % 2
    asm.emit(gfx11::v_lshrrev_b32(152, 1, 32));    // v152 = row = tid / 2
    asm.emit(gfx11::v_and_b32_imm(153, 32, 1));    // v153 = col_grp = tid % 2
    asm.emit(gfx11::v_lshlrev_b32(154, 6, 153));   // v154 = col_bytes = col_grp * 64

    // For C<256: clamp row to row & (C-1) for LDS write (address safety)
    let lds_row = if c_chunk < 256 {
        // v155 = row & (C-1) for LDS write
        // v155 = row & (C-1) for LDS write
        if c_chunk - 1 <= 64 {
            asm.emit(gfx11::v_and_b32_imm(155, 152, c_chunk - 1));
        } else {
            asm.emit2(gfx11::s_mov_b32_literal(14, c_chunk - 1));
            asm.emit(gfx11::v_mov_b32_from_sgpr(155, 14));
            asm.emit(gfx11::v_and_b32(155, 152, 155));
        }
        155u8
    } else {
        152u8  // use row directly
    };

    // HBM addr = V_ptr + (base_row + clamped_row) * 128 + col_bytes
    // Using clamped row for both HBM read and LDS write to avoid page faults.
    asm.emit(gfx11::v_mov_b32_from_sgpr(156, 13));  // base_row
    asm.emit(gfx11::v_add_u32(156, 156, lds_row));  // base_row + clamped_row
    asm.emit(gfx11::v_lshlrev_b32(156, 7, 156));    // * 128 bytes (bf16 row)
    asm.emit(gfx11::v_add_u32(156, 156, 154));      // + col_bytes

    asm.emit(gfx11::v_mov_b32_from_sgpr(157, 6));   // V_ptr_lo  (was 156, shift down by 1)
    asm.emit(gfx11::v_mov_b32_from_sgpr(158, 7));   // V_ptr_hi  (use 158)
    asm.emit2(gfx11::v_add_co_u32_vcc(157, 157, 156));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(158, 158));

    asm.emit2(gfx11::global_load_dwordx4(72, 157, 0));
    asm.emit2(gfx11::global_load_dwordx4(76, 157, 16));
    asm.emit2(gfx11::global_load_dwordx4(80, 157, 32));
    asm.emit2(gfx11::global_load_dwordx4(84, 157, 48));

    // LDS write addr = clamped_row * 132 + col_bytes
    asm.emit2(gfx11::s_mov_b32_literal(14, 132));
    asm.emit(gfx11::v_mov_b32_from_sgpr(159, 14));  // use v159 (was 158)
    asm.emit2(gfx11::v_mul_lo_u32(159, lds_row, 159)); // clamped_row * 132
    asm.emit(gfx11::v_add_u32(159, 159, 154));     // + col_bytes

    asm.emit(gfx11::s_waitcnt_vmcnt(0));
    asm.emit2(gfx11::ds_store_b128(159, 72, 0));
    asm.emit2(gfx11::ds_store_b128(159, 76, 16));
    asm.emit2(gfx11::ds_store_b128(159, 80, 32));
    asm.emit2(gfx11::ds_store_b128(159, 84, 48));

    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
    asm.barrier();  // Synchronize LUT + ZCLT

    // ========================================================================
    // 4. Prepare O output base address (for atomic adds later)
    //    O layout: [num_heads, seq_len, d_head] FP32 (256 bytes/row)
    // ========================================================================
    // v36 = wave_id (scalar copy for initial Q row)
    asm.emit(gfx11::v_readfirstlane(16, 33));   // s16 = wave_id

    // Precompute O_ptr base = O_ptr + base_row * 256
    // We'll add the actual row offset inside the loop
    asm.emit(gfx11::s_lshl_b32(17, 13, 8));     // s17 = base_row * 256 (bytes)
    asm.emit(gfx11::s_add_u32(18, 8, 17));       // s18 = O_ptr_lo + base_row_offset
    asm.emit(gfx11::s_addc_u32(19, 9, 0));       // s19 = O_ptr_hi

    // ========================================================================
    // 5. 1D Tile-Stealing Main Loop
    //    k = wave_id (initial task), k < 136, k += 16
    // ========================================================================
    asm.emit(gfx11::s_mov_b32(15, 16));            // s15 = k = wave_id (initial task)
    asm.emit2(gfx11::s_mov_b32_literal(22, 0xFFFFFFFF)); // s22 = current_r (invalid, force first Q load)

    // Check if wave_id >= n_tiles (skip main loop entirely for idle waves)
    let total_tiles = n_tiles(c_chunk);
    if total_tiles < 136 {
        if total_tiles <= 64 {
            asm.emit(gfx11::s_cmp_ge_u32(15, (0x80 + total_tiles) as u8));
        } else {
            asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
            asm.emit(gfx11::s_cmp_ge_u32(15, 14));
        }
    }
    let skip_loop_pc = if total_tiles < 136 {
        let pc = asm.current_pc();
        asm.emit(gfx11::s_cbranch_scc1(0)); // skip if wave_id >= n_tiles
        Some(pc)
    } else { None };

    let main_loop = asm.current_pc();

    // --- 5a. LUT lookup: read (r, c) from LDS ---
    // LDS addr = LUT_BASE + k * 2 — SALU computes uniform addr (铁律 #83: free)
    asm.emit(gfx11::s_lshl_b32(14, 15, 1));          // s14 = k * 2
    let lut_base = lds_lut_base(c_chunk);
    if lut_base <= 64 {
        asm.emit(gfx11::s_add_u32_imm(14, 14, lut_base as u8));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(27, lut_base));
        asm.emit(gfx11::s_add_u32(14, 14, 27));           // s14 = LUT_BASE + k*2
    }
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));      // v0 = s14 (1 VALU only)
    asm.emit2(gfx11::ds_load_u16(1, 0, 0));         // v1 = packed (r<<8 | c)
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
    
    // Unpack: s23 = r, s24 = c
    asm.emit(gfx11::v_lshrrev_b32(0, 8, 1));        // v0 = r
    asm.emit(gfx11::v_readfirstlane(23, 0));         // s23 = r
    asm.emit2(gfx11::v_and_b32_literal(0, 1, 0xFF)); // v0 = c (low 8 bits)
    asm.emit(gfx11::v_readfirstlane(24, 0));         // s24 = c

    // --- 5b. Conditional Q_slice load (only when r changes) ---
    asm.emit(gfx11::s_cmp_eq_u32(23, 22));           // SCC = (r == current_r)?
    let skip_q_load = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1(0));              // skip Q load if same r

    // Load Q_slice for row r
    asm.emit(gfx11::s_mov_b32(22, 23));              // current_r = r
    // Q addr = Q_ptr + (base_row + r*16 + lane_row) * 128
    asm.emit(gfx11::s_lshl_b32(25, 23, 4));         // s25 = r * 16
    asm.emit(gfx11::s_add_u32(25, 25, 13));          // s25 = base_row + r*16
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 25));
    asm.emit(gfx11::v_add_u32(0, 0, 35));            // v0 = base_row + r*16 + lane_row
    asm.emit(gfx11::v_lshlrev_b32(0, 7, 0));         // v0 = * 128 bytes
    asm.emit(gfx11::v_mov_b32_from_sgpr(1, 2));      // Q_ptr_lo
    asm.emit(gfx11::v_mov_b32_from_sgpr(2, 3));      // Q_ptr_hi  -- NOTE: s2/s3 still Q_ptr
    // Wait, I'm overwriting v2 which has Q_ptr_hi... let me use a temp pattern
    // Actually s2:s3 = Q_ptr. Let me be more careful with VGPR usage.
    asm.emit(gfx11::v_mov_b32_from_sgpr(36, 2));     // v36 = Q_ptr_lo
    asm.emit(gfx11::v_mov_b32_from_sgpr(37, 3));     // v37 = Q_ptr_hi
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 0));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));

    // Q_slice → v[40:71]
    asm.emit2(gfx11::global_load_dwordx4(40, 36, 0));
    asm.emit2(gfx11::global_load_dwordx4(44, 36, 16));
    asm.emit2(gfx11::global_load_dwordx4(48, 36, 32));
    asm.emit2(gfx11::global_load_dwordx4(52, 36, 48));
    asm.emit2(gfx11::global_load_dwordx4(56, 36, 64));
    asm.emit2(gfx11::global_load_dwordx4(60, 36, 80));
    asm.emit2(gfx11::global_load_dwordx4(64, 36, 96));
    asm.emit2(gfx11::global_load_dwordx4(68, 36, 112));

    // Also zero O_acc when r changes (new output row group)
    for i in 120..152u8 {
        asm.emit(gfx11::v_mov_b32_imm(i, 0));
    }

    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    let q_load_done = asm.current_pc();
    asm.patch_branch(skip_q_load, q_load_done);

    // --- 5c. Load K_tile[c] ---
    // K addr = K_ptr + (base_row + c*16 + lane_row) * 128
    // SALU: compute uniform part (base_row + c*16) * 128 (铁律 #83: free)
    asm.emit(gfx11::s_lshl_b32(25, 24, 4));         // s25 = c * 16
    asm.emit(gfx11::s_add_u32(25, 25, 13));          // s25 = base_row + c*16
    asm.emit(gfx11::s_lshl_b32(14, 25, 7));          // s14 = (base_row + c*16) * 128
    // Per-lane: lane_row * 128 + uniform offset
    asm.emit(gfx11::v_lshlrev_b32(0, 7, 35));        // v0 = lane_row * 128
    asm.emit(gfx11::v_mov_b32_from_sgpr(1, 14));
    asm.emit(gfx11::v_add_u32(0, 0, 1));             // v0 = total byte offset
    asm.emit(gfx11::v_mov_b32_from_sgpr(1, 4));      // K_ptr_lo
    asm.emit(gfx11::v_mov_b32_from_sgpr(2, 5));      // K_ptr_hi
    asm.emit2(gfx11::v_add_co_u32_vcc(1, 1, 0));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(2, 2));

    // K_tile → v[72:103]
    asm.emit2(gfx11::global_load_dwordx4(72, 1, 0));
    asm.emit2(gfx11::global_load_dwordx4(76, 1, 16));
    asm.emit2(gfx11::global_load_dwordx4(80, 1, 32));
    asm.emit2(gfx11::global_load_dwordx4(84, 1, 48));
    asm.emit2(gfx11::global_load_dwordx4(88, 1, 64));
    asm.emit2(gfx11::global_load_dwordx4(92, 1, 80));
    asm.emit2(gfx11::global_load_dwordx4(96, 1, 96));
    asm.emit2(gfx11::global_load_dwordx4(100, 1, 112));

    // Zero P_acc: v[104:111]
    for i in 104..112u8 {
        asm.emit(gfx11::v_mov_b32_imm(i, 0));
    }
    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    // --- 5d. WMMA: P_sub^T = K_tile @ Q_slice^T (4 k-groups) ---
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 72, 40, 104));
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 80, 48, 104));
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 88, 56, 104));
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 96, 64, 104));
    asm.emit(gfx11::s_mov_b32_imm(106, 0));  // clear VCC

    // --- 5e. Causal mask (only on diagonal tile: r == c) ---
    asm.emit(gfx11::s_cmp_eq_u32(23, 24));          // SCC = (r == c)
    let skip_mask_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc0(0));             // skip mask if r != c

    asm.emit(gfx11::v_lshrrev_b32(1, 4, 34));       // v1 = lane_half
    asm.emit(gfx11::v_mov_b32_imm(0, 0));            // v0 = 0.0f32
    asm.emit2(gfx11::s_mov_b32_literal(14, 0x3F800000u32)); // s14 = 1.0f32
    asm.emit(gfx11::v_mov_b32_from_sgpr(3, 14));     // v3 = 1.0f32
    for k in 0..8u8 {
        if k == 0 {
            asm.emit(gfx11::v_mov_b32(2, 1));
        } else {
            asm.emit(gfx11::v_add_u32_imm(2, 1, 2 * k as u32));
        }
        asm.emit(gfx11::v_cmp_ge_i32(35, 2));
        asm.emit(gfx11::v_cndmask_b32(4, 0, 3));
        asm.emit(gfx11::s_mov_b32_imm(106, 0));
        asm.emit(gfx11::v_mul_f32(104 + k, 104 + k, 4));
    }

    let mask_done_pc = asm.current_pc();
    asm.patch_branch(skip_mask_pc, mask_done_pc);

    // --- 5f. P^T C-layout → A-operand BF16x2 via ds_swizzle(SWAP16) ---
    asm.reg_transpose_c_to_ab(112, 104, 1, 34);
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // --- 5g. V_tile^T extraction from LDS and WMMA (per tile c) ---
    // V_tile for column c: LDS base = c * 16 * 132 = c * 2112
    // SALU: precompute tile base and per-v_grp offsets (铁律 #83: free)
    asm.emit2(gfx11::s_mov_b32_literal(25, 2112));
    asm.emit(gfx11::s_mul_i32(25, 24, 25));         // s25 = c * 2112

    // OPTIMIZATION: Ping-pong 2-group LDS prefetch to halve waitcnt barriers
    // Group pair 0+1: load both into v[152:159] and v[160:167], then 2 WMMAs
    // Group pair 2+3: reload v[152:159] and v[160:167], then 2 WMMAs
    // Result: 2 waitcnt barriers instead of 4 → ~-50% LDS stall exposure
    for pair in 0..2u8 {
        let grp_a = pair * 2;       // e.g. pair=0: grp_a=0, pair=1: grp_a=2
        let grp_b = pair * 2 + 1;  // e.g. pair=0: grp_b=1, pair=1: grp_b=3

        // Issue Group A loads (v[152:159])
        {
            let col_byte_offset_a = (grp_a as u32) * 32;
            if col_byte_offset_a == 0 {
                asm.emit(gfx11::s_mov_b32(14, 25));
            } else if col_byte_offset_a <= 64 {
                asm.emit(gfx11::s_add_u32_imm(14, 25, col_byte_offset_a as u8));
            } else {
                asm.emit2(gfx11::s_mov_b32_literal(14, col_byte_offset_a));
                asm.emit(gfx11::s_add_u32(14, 25, 14));
            }
            asm.emit(gfx11::v_lshlrev_b32(2, 1, 35));
            asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));
            asm.emit(gfx11::v_add_u32(2, 0, 2));
            for k in 0..8u8 {
                let r_even = (k as u32) * 2;
                let r_odd  = r_even + 1;
                asm.emit2(gfx11::ds_load_u16_d16(152 + k, 2, (r_even * 132) as u16));
                asm.emit2(gfx11::ds_load_u16_d16_hi(152 + k, 2, (r_odd * 132) as u16));
            }
        }

        // Issue Group B loads (v[160:167]) while Group A is in flight
        {
            let col_byte_offset_b = (grp_b as u32) * 32;
            if col_byte_offset_b <= 64 {
                asm.emit(gfx11::s_add_u32_imm(14, 25, col_byte_offset_b as u8));
            } else {
                asm.emit2(gfx11::s_mov_b32_literal(14, col_byte_offset_b));
                asm.emit(gfx11::s_add_u32(14, 25, 14));
            }
            asm.emit(gfx11::v_lshlrev_b32(2, 1, 35));
            asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));
            asm.emit(gfx11::v_add_u32(2, 0, 2));
            for k in 0..8u8 {
                let r_even = (k as u32) * 2;
                let r_odd  = r_even + 1;
                asm.emit2(gfx11::ds_load_u16_d16(160 + k, 2, (r_even * 132) as u16));
                asm.emit2(gfx11::ds_load_u16_d16_hi(160 + k, 2, (r_odd * 132) as u16));
            }
        }

        // ONE waitcnt for both groups (previously 2 separate waits)
        asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

        // WMMA Group A
        let acc_a = 120 + grp_a * 8;
        asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(acc_a, 112, 152, acc_a));
        // WMMA Group B
        let acc_b = 120 + grp_b * 8;
        asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(acc_b, 112, 160, acc_b));
    }
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // --- 5h. Check if next task has different r → flush O_acc via atomic add ---
    // Compute next_k = k + 16
    asm.emit(gfx11::s_add_u32_imm(26, 15, 16));     // s26 = next_k
    
    // Check: is next_k >= 136 OR does next tile have different r?
    // If next_k >= 136: this is the last iteration, must flush
    asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
    asm.emit(gfx11::s_cmp_ge_u32(26, 14));           // SCC = (next_k >= n_tiles)?
    let flush_at_end_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1(0));              // → flush

    // Read next tile's r from LUT — SALU addr calc (铁律 #83: free)
    asm.emit(gfx11::s_lshl_b32(14, 26, 1));           // s14 = next_k * 2
    let lut_base2 = lds_lut_base(c_chunk);
    if lut_base2 <= 64 {
        asm.emit(gfx11::s_add_u32_imm(14, 14, lut_base2 as u8));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(27, lut_base2));
        asm.emit(gfx11::s_add_u32(14, 14, 27));            // s14 = LUT_BASE + next_k*2
    }
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));       // v0 = s14 (1 VALU)
    asm.emit2(gfx11::ds_load_u16(1, 0, 0));
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
    asm.emit(gfx11::v_lshrrev_b32(0, 8, 1));         // v0 = next_r
    asm.emit(gfx11::v_readfirstlane(27, 0));          // s27 = next_r
    
    asm.emit(gfx11::s_cmp_eq_u32(27, 23));            // next_r == current_r?
    let skip_flush = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1(0));               // skip flush if same r (will accumulate more)

    // --- FLUSH: Atomic add O_acc to HBM ---
    let flush_label = asm.current_pc();
    asm.patch_branch(flush_at_end_pc, flush_label);

    // O atomic add: row = base_row + r*16 + lane_half + 2*vk
    // col = v_grp*16 + lane_row
    asm.emit(gfx11::v_lshrrev_b32(1, 4, 34));        // v1 = lane_half
    asm.emit(gfx11::s_lshl_b32(25, 23, 4));          // s25 = r * 16
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 25));
    asm.emit(gfx11::v_add_u32(0, 0, 1));              // v0 = r*16 + lane_half
    asm.emit(gfx11::v_lshlrev_b32(0, 8, 0));          // v0 = (r*16+lane_half) * 256 bytes/row

    asm.emit(gfx11::v_lshlrev_b32(3, 2, 35));         // v3 = lane_row * 4

    asm.emit(gfx11::v_mov_b32_from_sgpr(36, 18));     // O_base_lo
    asm.emit(gfx11::v_mov_b32_from_sgpr(37, 19));     // O_base_hi
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 0));    // + row offset
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 3));    // + col offset
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));

    // Atomic add all 32 accumulators (4 v_grps × 8 vk)
    for v_grp in 0..4u8 {
        let col_off_bytes = (v_grp as i32) * 64;
        for k in 0..8u8 {
            let row_off = (k as i32) * 512;   // 2*vk rows * 256 bytes/row
            let total_off = col_off_bytes + row_off;
            let acc = 120 + v_grp * 8 + k;
            asm.emit2(gfx11::global_atomic_add_f32_no_rtn(36, acc, total_off));
        }
    }
    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    let flush_done = asm.current_pc();
    asm.patch_branch(skip_flush, flush_done);
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    asm.emit(gfx11::s_add_u32_imm(15, 15, 16));      // k += 16
    asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
    asm.emit(gfx11::s_cmp_lt_u32(15, 14));            // k < n_tiles?
    let branch_loop = asm.branch_offset(asm.current_pc(), main_loop);
    asm.emit(gfx11::s_cbranch_scc1(branch_loop));

    // ========================================================================
    // 6. Epilogue
    // ========================================================================
    if let Some(pc) = skip_loop_pc {
        let here = asm.current_pc();
        asm.patch_branch(pc, here);
    }
    asm.emit(gfx11::s_waitcnt_vmcnt(0));
    asm.emit(gfx11::s_waitcnt_vscnt(0));
    asm.emit(gfx11::S_ENDPGM);

    let lds_total = lds_lut_base(c_chunk) + n_tiles(c_chunk) * 2;
    // Round up to 256-byte alignment
    let lds_total = (lds_total + 255) & !255;

    AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
        name: format!("ocpa_forward_intra_c{c_chunk}"),
        lds_size: lds_total,
        kernarg_size: 40,
        vgpr_count: 176,
        sgpr_count: 32,
        workgroup_size_x: 512,
        workgroup_size_y: 1,
        workgroup_size_z: 1,
        scratch_size: 0,
    })
}
