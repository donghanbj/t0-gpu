//! OCPA Backward Intra Kernels for GFX11 (RDNA3) — 1D Tile-Stealing Architecture
//!
//! Mathematical Isomorphism with Forward Intra:
//!   Intra(A, B, C, Mask) = (Mask ⊙ (A @ B^T)) @ C
//!
//!   dQ_intra = Intra(dO, V, K, Lower)   ← identical topology to Forward
//!   dK_intra = Intra(V, dO, Q, Upper)   ← time-arrow reversed
//!   dV_intra = Intra(K, Q, dO, Upper)   ← time-arrow reversed
//!
//! 1D TILE-STEALING: 136 tiles mapped to 1D pool, 16 waves steal linearly
//!   Lower triangle LUT: k → (r, c) where r*(r+1)/2 + c = k
//!   Upper triangle: same LUT but swap (r, c) → (c, r)
//!   Output via global_atomic_add_f32 (waves no longer own fixed rows)
//!
//! Grid: (N_chunks, num_heads, 1)
//! WG:   (512, 1, 1) = 16 Waves
//!
//! Kernarg layout:
//!   A_ptr(0), B_ptr(8), C_ptr(16), Out_ptr(24), seq_len(32)

use crate::rdna3_asm::{Rdna3Assembler, gfx11};
use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

const fn lds_zclt_size(c: u32) -> u32 { c * 132 }
const fn lds_lut_base(c: u32) -> u32 { lds_zclt_size(c) }
const fn n_tile_rows(c: u32) -> u32 { c / 16 }
const fn n_tiles(c: u32) -> u32 { let r = n_tile_rows(c); r * (r + 1) / 2 }
const fn c_shift(c: u32) -> u32 { match c { 32 => 5, 64 => 6, 128 => 7, 256 => 8, _ => 8 } }

/// Build backward intra kernel for dQ (Lower causal mask)
pub fn build_ocpa_backward_intra_dq() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dq", false, 256)
}

pub fn build_ocpa_backward_intra_dq_c64() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dq_c64", false, 64)
}

pub fn build_ocpa_backward_intra_dq_c32() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dq_c32", false, 32)
}

/// Build backward intra kernel for dK/dV (Upper causal mask)
pub fn build_ocpa_backward_intra_dkdv() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dkdv", true, 256)
}

pub fn build_ocpa_backward_intra_dkdv_c64() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dkdv_c64", true, 64)
}

pub fn build_ocpa_backward_intra_dkdv_c32() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dkdv_c32", true, 32)
}

pub fn build_ocpa_backward_intra_dq_c128() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dq_c128", false, 128)
}

pub fn build_ocpa_backward_intra_dkdv_c128() -> AmdGpuCodeObject {
    build_intra_kernel("ocpa_backward_intra_dkdv_c128", true, 128)
}

/// Unified Intra kernel builder — `is_upper` controls mask direction.
/// 1D Tile-Stealing: k = wave_id, k < 136, k += 16
fn build_intra_kernel(name: &str, is_upper: bool, c_chunk: u32) -> AmdGpuCodeObject {
    let mut asm = Rdna3Assembler::new();

    // ========================================================================
    // 1. System parameter capture
    // ========================================================================
    asm.emit(gfx11::s_mov_b32(20, 2));
    asm.emit(gfx11::s_mov_b32(21, 3));

    asm.emit2(gfx11::s_load_dwordx2(2, 0, 0));    // s[2:3] = A_ptr
    asm.emit2(gfx11::s_load_dwordx2(4, 0, 8));    // s[4:5] = B_ptr
    asm.emit2(gfx11::s_load_dwordx2(6, 0, 16));   // s[6:7] = C_ptr
    asm.emit2(gfx11::s_load_dwordx2(8, 0, 24));   // s[8:9] = Out_ptr
    asm.emit2(gfx11::s_load_dword(10, 0, 32));    // s10    = seq_len
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    asm.emit(gfx11::v_mov_b32(32, 0));
    asm.emit(gfx11::v_lshrrev_b32(33, 5, 32));
    asm.emit(gfx11::v_and_b32_imm(34, 32, 31));
    asm.emit(gfx11::v_and_b32_imm(35, 32, 15));

    asm.emit(gfx11::s_mul_i32(11, 21, 10));
    asm.emit(gfx11::s_lshl_b32(12, 20, c_shift(c_chunk) as u8));
    asm.emit(gfx11::s_add_u32(13, 11, 12));        // s13 = base_row

    // ========================================================================
    // 2. Phase A₀: Build 1D→2D LUT (136 entries)
    // ========================================================================
    let total_tiles = n_tiles(c_chunk);
    if total_tiles <= 64 {
        asm.emit(gfx11::v_mov_b32_imm(0, total_tiles as i32));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
        asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));
    }
    asm.emit(gfx11::v_cmp_lt_u32(32, 0));
    let skip_lut_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_vccz(0));

    // r = floor((sqrt(8*k + 1) - 1) / 2)
    asm.emit(gfx11::v_cvt_f32_u32(0, 32));
    asm.emit3(gfx11::v_mul_f32_e64_literal(0, 0, 0x41000000)); // * 8.0
    asm.emit2(gfx11::v_add_f32_e64_inline(0, 0, 0xF2));        // + 1.0
    asm.emit(gfx11::v_sqrt_f32(0, 0));
    asm.emit2(gfx11::v_add_f32_e64_inline(0, 0, 0xF3));        // - 1.0
    asm.emit2(gfx11::v_mul_f32_e64_inline(0, 0, 0xF0));        // * 0.5
    asm.emit(gfx11::v_cvt_u32_f32(1, 0));                       // v1 = r

    // Correction: tri = r*(r+1)/2, if tri > k then r--
    asm.emit(gfx11::v_add_u32_imm(2, 1, 1));
    asm.emit2(gfx11::v_mul_lo_u32(2, 1, 2));
    asm.emit(gfx11::v_lshrrev_b32(2, 1, 2));       // v2 = tri

    asm.emit(gfx11::v_cmp_lt_u32(32, 2));            // k < tri → r too big
    let skip_fix_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_vccz(0));

    asm.emit2(gfx11::s_mov_b32_literal(14, 0xFFFFFFFF));
    asm.emit(gfx11::v_mov_b32_from_sgpr(3, 14));
    asm.emit(gfx11::v_add_u32(1, 1, 3));             // r--

    asm.emit(gfx11::v_add_u32_imm(2, 1, 1));
    asm.emit2(gfx11::v_mul_lo_u32(2, 1, 2));
    asm.emit(gfx11::v_lshrrev_b32(2, 1, 2));

    let fix_done_pc = asm.current_pc();
    asm.patch_branch(skip_fix_pc, fix_done_pc);
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // c = k - tri
    asm.emit2(gfx11::s_mov_b32_literal(14, 0xFFFFFFFF));
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));
    asm.emit(gfx11::v_xor_b32(0, 2, 0));             // ~tri
    asm.emit(gfx11::v_add_u32_imm(0, 0, 1));         // -tri
    asm.emit(gfx11::v_add_u32(3, 32, 0));             // v3 = k - tri = c

    // Pack (r << 8 | c) into v1
    asm.emit(gfx11::v_lshlrev_b32(1, 8, 1));
    asm.emit(gfx11::v_or_b32(1, 1, 3));

    // Write to LDS LUT
    asm.emit(gfx11::v_lshlrev_b32(0, 1, 32));
    let lut_base = lds_lut_base(c_chunk);
    if lut_base <= 64 {
        asm.emit(gfx11::v_add_u32_imm(0, 0, lut_base));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(14, lut_base));
        asm.emit(gfx11::v_mov_b32_from_sgpr(3, 14));
        asm.emit(gfx11::v_add_u32(0, 0, 3));
    }
    asm.emit2(gfx11::ds_store_b16(0, 1, 0));

    let lut_done_pc = asm.current_pc();
    asm.patch_branch(skip_lut_pc, lut_done_pc);
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // ========================================================================
    // 3. Phase A: ZCLT-load C to LDS (132-byte stride)
    //    For C<256: clamp LDS write row to row & (C-1) to prevent out-of-bounds.
    // ========================================================================
    asm.emit(gfx11::v_lshrrev_b32(152, 1, 32));
    asm.emit(gfx11::v_and_b32_imm(153, 32, 1));
    asm.emit(gfx11::v_lshlrev_b32(154, 6, 153));

    // Clamp row for LDS write address when C<256
    let lds_row = if c_chunk < 256 {
        if c_chunk - 1 <= 64 {
            asm.emit(gfx11::v_and_b32_imm(155, 152, c_chunk - 1));
        } else {
            asm.emit2(gfx11::s_mov_b32_literal(14, c_chunk - 1));
            asm.emit(gfx11::v_mov_b32_from_sgpr(155, 14));
            asm.emit(gfx11::v_and_b32(155, 152, 155));
        }
        155u8
    } else { 152u8 };

    // HBM addr = C_ptr + (base_row + clamped_row) * 128 + col_bytes
    asm.emit(gfx11::v_mov_b32_from_sgpr(156, 13));
    asm.emit(gfx11::v_add_u32(156, 156, lds_row));  // use clamped row
    asm.emit(gfx11::v_lshlrev_b32(156, 7, 156));
    asm.emit(gfx11::v_add_u32(156, 156, 154));

    asm.emit(gfx11::v_mov_b32_from_sgpr(157, 6));
    asm.emit(gfx11::v_mov_b32_from_sgpr(158, 7));
    asm.emit2(gfx11::v_add_co_u32_vcc(157, 157, 156));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(158, 158));

    asm.emit2(gfx11::global_load_dwordx4(72, 157, 0));
    asm.emit2(gfx11::global_load_dwordx4(76, 157, 16));
    asm.emit2(gfx11::global_load_dwordx4(80, 157, 32));
    asm.emit2(gfx11::global_load_dwordx4(84, 157, 48));

    asm.emit2(gfx11::s_mov_b32_literal(14, 132));
    asm.emit(gfx11::v_mov_b32_from_sgpr(159, 14));
    asm.emit2(gfx11::v_mul_lo_u32(159, lds_row, 159));
    asm.emit(gfx11::v_add_u32(159, 159, 154));

    asm.emit(gfx11::s_waitcnt_vmcnt(0));
    asm.emit2(gfx11::ds_store_b128(159, 72, 0));
    asm.emit2(gfx11::ds_store_b128(159, 76, 16));
    asm.emit2(gfx11::ds_store_b128(159, 80, 32));
    asm.emit2(gfx11::ds_store_b128(159, 84, 48));

    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
    asm.barrier();

    // ========================================================================
    // 4. Prepare Out_ptr base for atomic adds
    // ========================================================================
    asm.emit(gfx11::v_readfirstlane(16, 33));     // s16 = wave_id
    asm.emit(gfx11::s_lshl_b32(17, 13, 8));       // s17 = base_row * 256
    asm.emit(gfx11::s_add_u32(18, 8, 17));         // s18 = Out_ptr_lo + offset
    asm.emit(gfx11::s_addc_u32(19, 9, 0));         // s19 = Out_ptr_hi

    // ========================================================================
    // 5. 1D Tile-Stealing Main Loop: k = wave_id, k < 136, k += 16
    // ========================================================================
    asm.emit(gfx11::s_mov_b32(15, 16));             // s15 = k = wave_id
    asm.emit2(gfx11::s_mov_b32_literal(22, 0xFFFFFFFF)); // s22 = current_r (invalid)

    // Skip loop for idle waves
    let skip_loop_pc = if total_tiles < 136 {
        if total_tiles <= 64 {
            asm.emit(gfx11::s_cmp_ge_u32(15, (0x80 + total_tiles) as u8));
        } else {
            asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
            asm.emit(gfx11::s_cmp_ge_u32(15, 14));
        }
        let pc = asm.current_pc();
        asm.emit(gfx11::s_cbranch_scc1(0));
        Some(pc)
    } else { None };

    let main_loop = asm.current_pc();

    // --- 5a. LUT lookup ---
    // SALU: compute uniform LDS addr (铁律 #83: free)
    asm.emit(gfx11::s_lshl_b32(14, 15, 1));          // s14 = k * 2
    let lut_base = lds_lut_base(c_chunk);
    if lut_base <= 64 {
        asm.emit(gfx11::s_add_u32_imm(14, 14, lut_base as u8));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(29, lut_base));
        asm.emit(gfx11::s_add_u32(14, 14, 29));
    }
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));      // v0 = s14 (1 VALU only)
    asm.emit2(gfx11::ds_load_u16(1, 0, 0));
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    // Unpack into r, c (lower triangle coords)
    asm.emit(gfx11::v_lshrrev_b32(0, 8, 1));
    asm.emit(gfx11::v_readfirstlane(23, 0));        // s23 = lut_r
    asm.emit2(gfx11::v_and_b32_literal(0, 1, 0xFF));
    asm.emit(gfx11::v_readfirstlane(24, 0));        // s24 = lut_c

    // For upper triangle: swap (r, c) → A_row = lut_c, B_col = lut_r
    // For lower triangle: A_row = lut_r, B_col = lut_c
    // s25 = A_tile_idx (which 16-row block of A to load)
    // s26 = B_tile_idx (which 16-row block of B to load)
    if is_upper {
        // Upper: A_tile = c, B_tile = r, C_tile = r (for LDS V extraction)
        asm.emit(gfx11::s_mov_b32(25, 24));  // s25 = A_tile = lut_c
        asm.emit(gfx11::s_mov_b32(26, 23));  // s26 = B_tile = lut_r
        // Diagonal check: lut_r == lut_c
    } else {
        // Lower: A_tile = r, B_tile = c, C_tile = c
        asm.emit(gfx11::s_mov_b32(25, 23));  // s25 = A_tile = lut_r
        asm.emit(gfx11::s_mov_b32(26, 24));  // s26 = B_tile = lut_c
    }

    // --- 5b. Conditional A_slice load (only when A_tile changes) ---
    asm.emit(gfx11::s_cmp_eq_u32(25, 22));
    let skip_a_load = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1(0));

    asm.emit(gfx11::s_mov_b32(22, 25));             // current_a_tile = A_tile

    // A addr = A_ptr + (base_row + A_tile*16 + lane_row) * 128
    asm.emit(gfx11::s_lshl_b32(27, 25, 4));
    asm.emit(gfx11::s_add_u32(27, 27, 13));
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 27));
    asm.emit(gfx11::v_add_u32(0, 0, 35));
    asm.emit(gfx11::v_lshlrev_b32(0, 7, 0));
    asm.emit(gfx11::v_mov_b32_from_sgpr(36, 2));
    asm.emit(gfx11::v_mov_b32_from_sgpr(37, 3));
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 0));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));

    asm.emit2(gfx11::global_load_dwordx4(40, 36, 0));
    asm.emit2(gfx11::global_load_dwordx4(44, 36, 16));
    asm.emit2(gfx11::global_load_dwordx4(48, 36, 32));
    asm.emit2(gfx11::global_load_dwordx4(52, 36, 48));
    asm.emit2(gfx11::global_load_dwordx4(56, 36, 64));
    asm.emit2(gfx11::global_load_dwordx4(60, 36, 80));
    asm.emit2(gfx11::global_load_dwordx4(64, 36, 96));
    asm.emit2(gfx11::global_load_dwordx4(68, 36, 112));

    // Zero O_acc when A_tile changes (new output row group)
    for i in 120..152u8 {
        asm.emit(gfx11::v_mov_b32_imm(i, 0));
    }
    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    let a_load_done = asm.current_pc();
    asm.patch_branch(skip_a_load, a_load_done);

    // --- 5c. Load B_tile ---
    // SALU: compute uniform part (base_row + B_tile*16) * 128 (铁律 #83: free)
    asm.emit(gfx11::s_lshl_b32(27, 26, 4));
    asm.emit(gfx11::s_add_u32(27, 27, 13));
    asm.emit(gfx11::s_lshl_b32(14, 27, 7));          // s14 = (base_row + B_tile*16) * 128
    // Per-lane: lane_row * 128 + uniform offset
    asm.emit(gfx11::v_lshlrev_b32(0, 7, 35));        // v0 = lane_row * 128
    asm.emit(gfx11::v_mov_b32_from_sgpr(1, 14));
    asm.emit(gfx11::v_add_u32(0, 0, 1));             // v0 = total byte offset
    asm.emit(gfx11::v_mov_b32_from_sgpr(1, 4));
    asm.emit(gfx11::v_mov_b32_from_sgpr(2, 5));
    asm.emit2(gfx11::v_add_co_u32_vcc(1, 1, 0));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(2, 2));

    asm.emit2(gfx11::global_load_dwordx4(72, 1, 0));
    asm.emit2(gfx11::global_load_dwordx4(76, 1, 16));
    asm.emit2(gfx11::global_load_dwordx4(80, 1, 32));
    asm.emit2(gfx11::global_load_dwordx4(84, 1, 48));
    asm.emit2(gfx11::global_load_dwordx4(88, 1, 64));
    asm.emit2(gfx11::global_load_dwordx4(92, 1, 80));
    asm.emit2(gfx11::global_load_dwordx4(96, 1, 96));
    asm.emit2(gfx11::global_load_dwordx4(100, 1, 112));

    for i in 104..112u8 {
        asm.emit(gfx11::v_mov_b32_imm(i, 0));
    }
    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    // --- 5d. WMMA: P_sub^T = B_tile @ A_slice^T ---
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 72, 40, 104));
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 80, 48, 104));
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 88, 56, 104));
    asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(104, 96, 64, 104));
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // --- 5e. Causal mask (only on diagonal: lut_r == lut_c) ---
    asm.emit(gfx11::s_cmp_eq_u32(23, 24));
    let skip_mask_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc0(0));

    asm.emit(gfx11::v_lshrrev_b32(1, 4, 34));
    asm.emit(gfx11::v_mov_b32_imm(0, 0));
    asm.emit2(gfx11::s_mov_b32_literal(14, 0x3F800000u32));
    asm.emit(gfx11::v_mov_b32_from_sgpr(3, 14));
    for k in 0..8u8 {
        if k == 0 {
            asm.emit(gfx11::v_mov_b32(2, 1));
        } else {
            asm.emit(gfx11::v_add_u32_imm(2, 1, 2 * k as u32));
        }
        if is_upper {
            asm.emit(gfx11::v_cmp_ge_i32(2, 35));   // K_row >= lane_row → upper
        } else {
            asm.emit(gfx11::v_cmp_ge_i32(35, 2));   // lane_row >= K_row → lower
        }
        asm.emit(gfx11::v_cndmask_b32(4, 0, 3));
        asm.emit(gfx11::s_mov_b32_imm(106, 0));
        asm.emit(gfx11::v_mul_f32(104 + k, 104 + k, 4));
    }

    let mask_done_pc = asm.current_pc();
    asm.patch_branch(skip_mask_pc, mask_done_pc);

    // --- 5f. Transpose P ---
    asm.reg_transpose_c_to_ab(112, 104, 1, 34);
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // --- 5g. C_tile extraction from LDS and WMMA ---
    // C_tile index = B_tile (s26) since C[B_tile*16..] is what we multiply P^T by
    // SALU: precompute tile base + PING-PONG optimization: halve waitcnt barriers
    asm.emit2(gfx11::s_mov_b32_literal(27, 2112));
    asm.emit(gfx11::s_mul_i32(27, 26, 27));         // s27 = B_tile * 2112

    for pair in 0..2u8 {
        let grp_a = pair * 2;
        let grp_b = pair * 2 + 1;

        // Issue Group A loads (v[152:159])
        {
            let col_byte_offset_a = (grp_a as u32) * 32;
            if col_byte_offset_a == 0 {
                asm.emit(gfx11::s_mov_b32(14, 27));
            } else if col_byte_offset_a <= 64 {
                asm.emit(gfx11::s_add_u32_imm(14, 27, col_byte_offset_a as u8));
            } else {
                asm.emit2(gfx11::s_mov_b32_literal(14, col_byte_offset_a));
                asm.emit(gfx11::s_add_u32(14, 27, 14));
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

        // Issue Group B loads (v[160:167]) while A is in flight
        {
            let col_byte_offset_b = (grp_b as u32) * 32;
            if col_byte_offset_b <= 64 {
                asm.emit(gfx11::s_add_u32_imm(14, 27, col_byte_offset_b as u8));
            } else {
                asm.emit2(gfx11::s_mov_b32_literal(14, col_byte_offset_b));
                asm.emit(gfx11::s_add_u32(14, 27, 14));
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

        asm.emit(gfx11::s_waitcnt_lgkmcnt(0)); // ONE wait for both
        let acc_a = 120 + grp_a * 8;
        asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(acc_a, 112, 152, acc_a));
        let acc_b = 120 + grp_b * 8;
        asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(acc_b, 112, 160, acc_b));
    }
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // --- 5h. Check if next task has different A_tile → flush via atomic add ---
    asm.emit(gfx11::s_add_u32_imm(28, 15, 16));    // s28 = next_k

    asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
    asm.emit(gfx11::s_cmp_ge_u32(28, 14));
    let flush_at_end_pc = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1(0));

    // Unpack next_r, next_c — SALU addr calc (铁律 #83: free)
    asm.emit(gfx11::s_lshl_b32(14, 28, 1));
    if lut_base <= 64 {
        asm.emit(gfx11::s_add_u32_imm(14, 14, lut_base as u8));
    } else {
        asm.emit2(gfx11::s_mov_b32_literal(29, lut_base));
        asm.emit(gfx11::s_add_u32(14, 14, 29));
    }
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 14));       // v0 = s14 (1 VALU)
    asm.emit2(gfx11::ds_load_u16(1, 0, 0));
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    // Unpack next_r, next_c
    asm.emit(gfx11::v_lshrrev_b32(0, 8, 1));        // next_lut_r
    asm.emit2(gfx11::v_and_b32_literal(1, 1, 0xFF)); // next_lut_c

    // Determine next_A_tile
    if is_upper {
        // next_A_tile = next_lut_c
        asm.emit(gfx11::v_readfirstlane(29, 1));
    } else {
        // next_A_tile = next_lut_r
        asm.emit(gfx11::v_readfirstlane(29, 0));
    }

    asm.emit(gfx11::s_cmp_eq_u32(29, 25));
    let skip_flush = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1(0));

    // --- FLUSH: Atomic add O_acc to HBM ---
    let flush_label = asm.current_pc();
    asm.patch_branch(flush_at_end_pc, flush_label);

    asm.emit(gfx11::v_lshrrev_b32(1, 4, 34));        // lane_half
    asm.emit(gfx11::s_lshl_b32(27, 25, 4));          // A_tile * 16
    asm.emit(gfx11::v_mov_b32_from_sgpr(0, 27));
    asm.emit(gfx11::v_add_u32(0, 0, 1));              // + lane_half
    asm.emit(gfx11::v_lshlrev_b32(0, 8, 0));          // * 256

    asm.emit(gfx11::v_lshlrev_b32(3, 2, 35));         // lane_row * 4

    asm.emit(gfx11::v_mov_b32_from_sgpr(36, 18));
    asm.emit(gfx11::v_mov_b32_from_sgpr(37, 19));
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 0));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 3));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));

    for v_grp in 0..4u8 {
        let col_off = (v_grp as i32) * 64;
        for k in 0..8u8 {
            let row_off = (k as i32) * 512;
            let total_off = col_off + row_off;
            let acc = 120 + v_grp * 8 + k;
            asm.emit2(gfx11::global_atomic_add_f32_no_rtn(36, acc, total_off));
        }
    }
    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    let flush_done = asm.current_pc();
    asm.patch_branch(skip_flush, flush_done);
    asm.emit(gfx11::s_mov_b32_imm(106, 0));

    // --- 5i. Loop control ---
    asm.emit(gfx11::s_add_u32_imm(15, 15, 16));
    asm.emit2(gfx11::s_mov_b32_literal(14, total_tiles));
    asm.emit(gfx11::s_cmp_lt_u32(15, 14));
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
    let lds_total = (lds_total + 255) & !255;

    AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
        name: name.to_string(),
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
