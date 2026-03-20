//! Softmax + Cross-Entropy Loss GPU Kernel (RDNA3 ISA)
//!
//! Numerically stable per-token softmax → CE loss → gradient computation.
//!
//! Architecture:
//!   - 1 WorkGroup (32 lanes) per token row
//!   - Each lane handles ceil(vocab/32) elements via loop
//!   - 3-pass algorithm:
//!     Pass 1: Find row max (wave reduction via ds_swizzle)
//!     Pass 2: Compute exp(x - max), accumulate sum (wave reduction)
//!     Pass 3: Compute grad = softmax(x) - one_hot(target), store grad, accumulate loss
//!
//! Kernarg (40 bytes):
//!   [0:7]   logits_ptr    [seq, vocab_aligned] f32 (read)
//!   [8:15]  targets_ptr   [seq] u32 (read)
//!   [16:23] grad_ptr      [seq, vocab_aligned] f32 (write)
//!   [24:31] loss_ptr      [seq] f32 (write, per-token loss)
//!   [32:35] vocab_size    u32 (actual vocab, for bounds checking)
//!   [36:39] vocab_aligned u32 (padded to 64 multiple, for stride)
//!
//! Grid: (seq_len * 32, 1, 1)   — seq_len WGs of 32 lanes each
//! WorkGroup: (32, 1, 1)

use crate::rdna3_asm::{Rdna3Assembler, gfx11};
use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

/// Build the softmax + cross-entropy loss kernel.
///
/// Register allocation:
///   s[0:1]  = kernarg_ptr
///   s[2:3]  = logits_ptr
///   s[4:5]  = targets_ptr
///   s[6:7]  = grad_ptr
///   s[8:9]  = loss_ptr
///   s10     = vocab_size
///   s11     = vocab_aligned
///   s12     = scratch
///   s[14:15]= logits_row_ptr
///   s[16:17]= grad_row_ptr
///   s18     = saved exec
///   s20     = WG_ID (token index)
///
///   v0      = loaded value / scratch
///   v1-v5   = scratch
///   v[30:31]= address register pair
///   v[32:33]= address register pair (grad)
///   v40     = lane_id
///   v41     = target token ID
///   v42     = loop index (base + lane_id)
///   v43     = scratch
///   v44     = scratch (byte offset)
///   v50     = row_max (pass 1) / accumulator
///   v51     = swizzle temp
///   v52     = exp_sum (pass 2)
///   v53     = inv_sum (1/sum)
///   v54     = loss accumulator (pass 3)
///   v55     = log2(e) constant
pub fn build_softmax_ce_loss() -> AmdGpuCodeObject {
    let mut asm = Rdna3Assembler::new();

    // ════════════════════════════════════════════════════════════════
    // Prologue: load kernargs
    // ════════════════════════════════════════════════════════════════
    asm.emit(gfx11::s_mov_b32(20, 2));             // s20 = WG_ID (TGID.x in s2)

    asm.emit2(gfx11::s_load_dwordx2(2, 0, 0));    // s[2:3]  = logits_ptr
    asm.emit2(gfx11::s_load_dwordx2(4, 0, 8));    // s[4:5]  = targets_ptr
    asm.emit2(gfx11::s_load_dwordx2(6, 0, 16));   // s[6:7]  = grad_ptr
    asm.emit2(gfx11::s_load_dwordx2(8, 0, 24));   // s[8:9]  = loss_ptr
    asm.emit2(gfx11::s_load_dword(10, 0, 32));    // s10     = vocab_size
    asm.emit2(gfx11::s_load_dword(11, 0, 36));    // s11     = vocab_aligned
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    // v40 = lane_id = v0 & 31
    asm.emit(gfx11::v_and_b32_imm(40, 0, 31));

    // Load target for this token: targets_ptr[wg_id]
    asm.emit(gfx11::v_mov_b32_from_sgpr(30, 4));
    asm.emit(gfx11::v_mov_b32_from_sgpr(31, 5));
    asm.emit(gfx11::v_mov_b32_from_sgpr(44, 20));  // wg_id
    asm.emit(gfx11::v_lshlrev_b32(44, 2, 44));     // wg_id * 4
    asm.emit2(gfx11::v_add_co_u32_vcc(30, 30, 44));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(31, 31));
    asm.emit2(gfx11::global_load_dword(41, 30, 0)); // v41 = target token ID
    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    // Compute row byte offset = wg_id * vocab_aligned * 4
    asm.emit(gfx11::s_mul_i32(12, 20, 11));         // s12 = wg_id * vocab_aligned
    asm.emit(gfx11::s_lshl_b32(12, 12, 2));         // s12 *= 4 (byte offset)

    // logits_row_ptr = logits_ptr + row_byte_offset → s[14:15]
    asm.emit(gfx11::s_add_u32(14, 2, 12));
    asm.emit(gfx11::s_addc_u32(15, 3, 0));

    // grad_row_ptr = grad_ptr + row_byte_offset → s[16:17]
    asm.emit(gfx11::s_add_u32(16, 6, 12));
    asm.emit(gfx11::s_addc_u32(17, 7, 0));

    // ════════════════════════════════════════════════════════════════
    // PASS 1: Find row maximum
    // ════════════════════════════════════════════════════════════════
    // v50 = local_max = -INF
    asm.emit2(gfx11::v_mov_b32_literal(50, 0xFF800000u32)); // -INF

    // SGPR loop counter: s13 = 0, increments by 32
    asm.emit(gfx11::s_mov_b32_imm(13, 0));
    let pass1_start = asm.current_pc();

    // idx = s13 + lane_id → v42
    asm.emit(gfx11::v_mov_b32_from_sgpr(42, 13));
    asm.emit(gfx11::v_add_u32(42, 42, 40));        // v42 = base + lane_id

    // bounds: v42 < vocab_size → VCC
    asm.emit(gfx11::v_mov_b32_from_sgpr(43, 10));
    asm.emit(gfx11::v_cmp_lt_u32(42, 43));         // VCC = (idx < vocab)

    // ⚠️ FIX: saveexec BEFORE addr calc (v_add_co_u32_vcc clobbers VCC!)
    asm.emit(gfx11::s_and_saveexec_b32_vcc(18));

    // addr = logits_row_ptr + idx * 4 (under exec mask)
    asm.emit(gfx11::v_lshlrev_b32(44, 2, 42));
    asm.emit(gfx11::v_mov_b32_from_sgpr(30, 14));
    asm.emit(gfx11::v_mov_b32_from_sgpr(31, 15));
    asm.emit2(gfx11::v_add_co_u32_vcc(30, 30, 44));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(31, 31));

    asm.emit2(gfx11::global_load_dword(0, 30, 0));  // v0 = logits[idx]
    asm.emit(gfx11::s_waitcnt_vmcnt(0));
    asm.emit(gfx11::v_max_f32(50, 50, 0));          // max = max(max, logits[idx])
    asm.emit(gfx11::s_mov_b32_exec_lo_from_sgpr(18)); // restore exec

    // s13 += 32
    asm.emit(gfx11::s_add_u32_imm(13, 13, 32));
    asm.emit(gfx11::s_cmp_lt_u32(13, 10));           // s13 < vocab_size?
    let pass1_end = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1((pass1_start as i32 - pass1_end as i32 - 1) as i16));

    // Wave max reduction: ds_swizzle XOR
    for pattern in [0x401Fu16, 0x201F, 0x101F, 0x081F, 0x041F] {
        asm.emit2(gfx11::ds_swizzle_b32(51, 50, pattern));
        asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
        asm.emit(gfx11::v_max_f32(50, 50, 51));
    }
    // v50 = row_max (broadcast to all lanes via swizzle)

    // ════════════════════════════════════════════════════════════════
    // PASS 2: exp(logit - max), accumulate sum
    // ════════════════════════════════════════════════════════════════
    // v55 = log2(e) = 1.4426950408...
    asm.emit2(gfx11::v_mov_b32_literal(55, 0x3FB8AA3Bu32));
    // v52 = exp_sum = 0
    asm.emit(gfx11::v_xor_b32(52, 52, 52));

    asm.emit(gfx11::s_mov_b32_imm(13, 0));
    let pass2_start = asm.current_pc();

    asm.emit(gfx11::v_mov_b32_from_sgpr(42, 13));
    asm.emit(gfx11::v_add_u32(42, 42, 40));

    asm.emit(gfx11::v_mov_b32_from_sgpr(43, 10));
    asm.emit(gfx11::v_cmp_lt_u32(42, 43));

    // ⚠️ FIX: saveexec BEFORE addr calc
    asm.emit(gfx11::s_and_saveexec_b32_vcc(18));

    asm.emit(gfx11::v_lshlrev_b32(44, 2, 42));
    asm.emit(gfx11::v_mov_b32_from_sgpr(30, 14));
    asm.emit(gfx11::v_mov_b32_from_sgpr(31, 15));
    asm.emit2(gfx11::v_add_co_u32_vcc(30, 30, 44));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(31, 31));

    asm.emit2(gfx11::global_load_dword(0, 30, 0));
    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    // exp_val = 2^((logit - max) * log2(e))
    asm.emit(gfx11::v_sub_f32(0, 0, 50));           // logit - max
    asm.emit(gfx11::v_mul_f32(0, 0, 55));            // * log2(e)
    asm.emit(gfx11::v_exp_f32(0, 0));                // 2^(x) = exp(logit - max)
    asm.emit(gfx11::v_add_f32(52, 52, 0));           // sum += exp_val

    asm.emit(gfx11::s_mov_b32_exec_lo_from_sgpr(18));

    asm.emit(gfx11::s_add_u32_imm(13, 13, 32));
    asm.emit(gfx11::s_cmp_lt_u32(13, 10));
    let pass2_end = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1((pass2_start as i32 - pass2_end as i32 - 1) as i16));

    // Wave sum reduction
    for pattern in [0x401Fu16, 0x201F, 0x101F, 0x081F, 0x041F] {
        asm.emit2(gfx11::ds_swizzle_b32(51, 52, pattern));
        asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
        asm.emit(gfx11::v_add_f32(52, 52, 51));
    }

    // v53 = inv_sum = 1 / exp_sum
    asm.emit(gfx11::v_rcp_f32(53, 52));

    // ════════════════════════════════════════════════════════════════
    // PASS 3: grad = softmax - one_hot(target), store, loss
    // ════════════════════════════════════════════════════════════════
    // v54 = loss_accum = 0
    asm.emit(gfx11::v_xor_b32(54, 54, 54));
    // v56 = 1.0 constant
    asm.emit2(gfx11::v_mov_b32_literal(56, 0x3F800000u32));
    // v57 = ln(2) = 0.6931472
    asm.emit2(gfx11::v_mov_b32_literal(57, 0x3F317218u32));

    asm.emit(gfx11::s_mov_b32_imm(13, 0));
    let pass3_start = asm.current_pc();

    asm.emit(gfx11::v_mov_b32_from_sgpr(42, 13));
    asm.emit(gfx11::v_add_u32(42, 42, 40));

    asm.emit(gfx11::v_mov_b32_from_sgpr(43, 10));
    asm.emit(gfx11::v_cmp_lt_u32(42, 43));

    // ⚠️ FIX: saveexec BEFORE addr calc
    asm.emit(gfx11::s_and_saveexec_b32_vcc(18));

    // logits addr (under exec mask — inactive lanes don't matter)
    asm.emit(gfx11::v_lshlrev_b32(44, 2, 42));
    asm.emit(gfx11::v_mov_b32_from_sgpr(30, 14));
    asm.emit(gfx11::v_mov_b32_from_sgpr(31, 15));
    asm.emit2(gfx11::v_add_co_u32_vcc(30, 30, 44));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(31, 31));
    // grad addr
    asm.emit(gfx11::v_mov_b32_from_sgpr(32, 16));
    asm.emit(gfx11::v_mov_b32_from_sgpr(33, 17));
    asm.emit2(gfx11::v_add_co_u32_vcc(32, 32, 44));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(33, 33));

    // Load logit, recompute softmax prob
    asm.emit2(gfx11::global_load_dword(0, 30, 0));
    asm.emit(gfx11::s_waitcnt_vmcnt(0));
    asm.emit(gfx11::v_sub_f32(0, 0, 50));           // logit - max
    asm.emit(gfx11::v_mul_f32(0, 0, 55));            // * log2(e)
    asm.emit(gfx11::v_exp_f32(0, 0));                // exp(logit - max)
    asm.emit(gfx11::v_mul_f32(0, 0, 53));            // prob = exp / sum

    // Check idx == target: XOR + cmp_eq_imm(0)
    asm.emit(gfx11::v_xor_b32(1, 42, 41));          // v1 = idx XOR target
    asm.emit(gfx11::v_cmp_eq_u32_imm(1, 0));         // VCC = (idx == target)
    // one_hot = VCC ? 1.0 : 0.0
    asm.emit(gfx11::v_mov_b32_imm(2, 0));            // v2 = 0.0
    asm.emit(gfx11::v_cndmask_b32(2, 2, 56));        // v2 = VCC ? 1.0 : 0.0

    // grad = prob - one_hot
    asm.emit(gfx11::v_sub_f32(3, 0, 2));             // v3 = grad

    // Store gradient
    asm.emit2(gfx11::global_store_dword(32, 3, 0));

    // Loss for target: -ln(prob) = -log2(prob) * ln(2)
    // CRITICAL: Clamp prob to min 1e-10 before log to prevent log(0) = -inf
    // which causes inf loss → inf gradients → NaN in backward GEMM (inf * 0 = NaN)
    asm.emit2(gfx11::v_mov_b32_literal(58, 0x2EDBE6FFu32)); // v58 = 1e-10
    asm.emit(gfx11::v_max_f32(0, 0, 58));            // prob = max(prob, 1e-10)
    asm.emit(gfx11::v_log_f32(4, 0));                // v4 = log2(prob)
    asm.emit(gfx11::v_mul_f32(4, 4, 57));             // v4 = ln(prob)
    // Negate and mask: only add if target
    asm.emit(gfx11::v_mov_b32_imm(5, 0));             // v5 = 0
    asm.emit(gfx11::v_cndmask_b32(5, 5, 4));          // v5 = VCC ? ln(prob) : 0
    asm.emit(gfx11::v_sub_f32(54, 54, 5));             // loss -= ln(prob)

    asm.emit(gfx11::s_mov_b32_exec_lo_from_sgpr(18));

    asm.emit(gfx11::s_add_u32_imm(13, 13, 32));
    asm.emit(gfx11::s_cmp_lt_u32(13, 10));
    let pass3_end = asm.current_pc();
    asm.emit(gfx11::s_cbranch_scc1((pass3_start as i32 - pass3_end as i32 - 1) as i16));

    // Wave sum reduction for loss (only 1 lane has nonzero, but reduce for robustness)
    for pattern in [0x401Fu16, 0x201F, 0x101F, 0x081F, 0x041F] {
        asm.emit2(gfx11::ds_swizzle_b32(51, 54, pattern));
        asm.emit(gfx11::s_waitcnt_lgkmcnt(0));
        asm.emit(gfx11::v_add_f32(54, 54, 51));
    }

    // ════════════════════════════════════════════════════════════════
    // Store per-token loss: loss_ptr[wg_id] (lane 0 only)
    // ════════════════════════════════════════════════════════════════
    asm.emit(gfx11::v_cmp_eq_u32_imm(40, 0));       // VCC = (lane_id == 0)
    asm.emit(gfx11::s_and_saveexec_b32_vcc(18));
    {
        asm.emit(gfx11::v_mov_b32_from_sgpr(30, 8));
        asm.emit(gfx11::v_mov_b32_from_sgpr(31, 9));
        asm.emit(gfx11::v_mov_b32_from_sgpr(44, 20)); // wg_id
        asm.emit(gfx11::v_lshlrev_b32(44, 2, 44));
        asm.emit2(gfx11::v_add_co_u32_vcc(30, 30, 44));
        asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(31, 31));
        asm.emit2(gfx11::global_store_dword(30, 54, 0));
    }
    asm.emit(gfx11::s_mov_b32_exec_lo_from_sgpr(18));

    asm.emit(gfx11::s_waitcnt_vmcnt(0));
    asm.emit(gfx11::s_waitcnt_vscnt(0));
    asm.emit(gfx11::S_ENDPGM);

    AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
        name: "softmax_ce_loss".to_string(),
        sgpr_count: 24,
        vgpr_count: 64,
        kernarg_size: 40,
        lds_size: 0,
        workgroup_size_x: 32, workgroup_size_y: 1, workgroup_size_z: 1,
        scratch_size: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_ce_loss_builds() {
        let co = build_softmax_ce_loss();
        let hs = co.to_code_object_llvm();
        assert!(hs.is_ok(), "softmax_ce_loss build failed: {:?}", hs.err());
    }
}
