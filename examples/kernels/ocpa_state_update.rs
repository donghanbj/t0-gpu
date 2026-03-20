//! OCPA State Update Kernel for GFX11 (RDNA3)
//!
//! Computes: W_c = K_c^T @ V_c  (64×64 FP32 output, per chunk per head)
//!
//! FIXED: ZCLT (Zero-Conflict LDS Transpose) Architecture Applied.
//! Maps the sequence dimension (m) perfectly into WMMA's reduction axis (k)
//! via a 132-byte padded LDS stride and zero-VALU ds_load_u16_d16 column tearing.

use crate::rdna3_asm::{Rdna3Assembler, gfx11};
use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

pub fn build_ocpa_state_update() -> AmdGpuCodeObject {
    let mut asm = Rdna3Assembler::new();

    // ========================================================================
    // 1. 系统参数捕获与 SGPR 加载
    // ========================================================================
    asm.emit(gfx11::s_mov_b32(20, 2));  // s20 = chunk_id
    asm.emit(gfx11::s_mov_b32(21, 3));  // s21 = head_id

    asm.emit2(gfx11::s_load_dwordx2(2, 0, 0));    // K_ptr
    asm.emit2(gfx11::s_load_dwordx2(4, 0, 8));    // V_ptr
    asm.emit2(gfx11::s_load_dwordx2(6, 0, 16));   // W_ptr
    asm.emit2(gfx11::s_load_dword(8, 0, 24));     // C_chunk (256)
    asm.emit2(gfx11::s_load_dword(9, 0, 28));     // d_head (64)
    asm.emit2(gfx11::s_load_dword(10, 0, 32));    // seq_len
    asm.emit2(gfx11::s_load_dword(11, 0, 36));    // n_chunks (dedicated field)
    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    asm.emit(gfx11::v_mov_b32(32, 0));  // v32 = thread_id

    // 清零 128 个 VGPR 作为 16 块 WMMA 的累加器 v[40..167]
    for i in 40..168u8 {
        asm.emit(gfx11::v_mov_b32_imm(i, 0));
    }

    // ========================================================================
    // 2. 指针寻址：HBM 宏观推算与单线程合并访存偏移
    // ========================================================================
    asm.emit(gfx11::s_mul_i32(12, 21, 10));        // head_id * seq_len
    asm.emit(gfx11::s_mul_i32(13, 20, 8));         // chunk_id * C_chunk
    asm.emit(gfx11::s_add_u32(14, 12, 13));        // row_start
    
    // row_start * 128 = row_start << 7; high bits = row_start >> 25
    asm.emit(gfx11::s_lshl_b32(22, 14, 7));        // offset_lo = row_start << 7
    asm.emit(gfx11::s_lshr_b32(23, 14, 25));       // offset_hi = row_start >> 25
    
    asm.emit(gfx11::s_add_u32(16, 2, 22));         // s[16:17] = K_base
    asm.emit(gfx11::s_addc_u32(17, 3, 23));
    asm.emit(gfx11::s_add_u32(18, 4, 22));         // s[18:19] = V_base
    asm.emit(gfx11::s_addc_u32(19, 5, 23));

    // 计算当前线程对于 16x64 Block 的线性 HBM 抓取偏移
    asm.emit(gfx11::v_lshlrev_b32(33, 6, 32));     // v33 = thread_id * 64
    
    // 绑定至 VGPR 供 Load 使用
    asm.emit(gfx11::v_mov_b32_from_sgpr(34, 16));  // K_ptr_lo
    asm.emit(gfx11::v_mov_b32_from_sgpr(35, 17));  // K_ptr_hi
    asm.emit2(gfx11::v_add_co_u32_vcc(34, 34, 33));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(35, 35));

    asm.emit(gfx11::v_mov_b32_from_sgpr(36, 18));  // V_ptr_lo
    asm.emit(gfx11::v_mov_b32_from_sgpr(37, 19));  // V_ptr_hi
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 33));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));

    // ========================================================================
    // 3. ZCLT LDS 拓扑映射：132 字节 Padding 摧毁 Bank Conflict
    // ========================================================================
    asm.emit(gfx11::v_lshrrev_b32(1, 1, 32));      // r = thread_id / 2
    asm.emit(gfx11::v_and_b32_imm(2, 32, 1));      // c = thread_id % 2
    asm.emit(gfx11::v_lshlrev_b32(2, 6, 2));       // c_bytes = c * 64
    asm.emit2(gfx11::s_mov_b32_literal(15, 132));
    asm.emit(gfx11::v_mov_b32_from_sgpr(3, 15));
    asm.emit2(gfx11::v_mul_lo_u32(38, 1, 3));      // r * 132
    asm.emit(gfx11::v_add_u32(38, 38, 2));         // v38 = K_LDS_write (Bank-safe!)
    // 铁律 #7: 2112 > 64, must use literal path to avoid v_add_u32_imm overflow
    asm.emit2(gfx11::s_mov_b32_literal(27, 2112));
    asm.emit(gfx11::v_mov_b32_from_sgpr(3, 27));   // reuse v3 (free temp)
    asm.emit(gfx11::v_add_u32(39, 38, 3));          // v39 = V_LDS_write (+2112 bytes)

    // LDS 读基址 (Lane = thread_id % 16)
    asm.emit(gfx11::v_and_b32_imm(232, 32, 15));   
    asm.emit(gfx11::v_lshlrev_b32(232, 1, 232));   // v232 = v_lds_read_base = Lane * 2
    
    asm.emit2(gfx11::s_mov_b32_literal(26, 2048)); // HBM step = 16 rows * 128 bytes
    asm.emit(gfx11::v_mov_b32_from_sgpr(233, 26));

    // ========================================================================
    // 4. 外积引擎主循环 (M 维度，每次推 16 行)
    // ========================================================================
    asm.emit(gfx11::s_mov_b32_imm(25, 0));         // m_idx = 0
    let loop_start = asm.current_pc();

    // A. 从 HBM 猛烈吸入 K/V 到临时寄存器 v[0..31]
    asm.emit2(gfx11::global_load_dwordx4(0, 34, 0));
    asm.emit2(gfx11::global_load_dwordx4(4, 34, 16));
    asm.emit2(gfx11::global_load_dwordx4(8, 34, 32));
    asm.emit2(gfx11::global_load_dwordx4(12, 34, 48));

    asm.emit2(gfx11::global_load_dwordx4(16, 36, 0));
    asm.emit2(gfx11::global_load_dwordx4(20, 36, 16));
    asm.emit2(gfx11::global_load_dwordx4(24, 36, 32));
    asm.emit2(gfx11::global_load_dwordx4(28, 36, 48));

    // B. 提前推进 HBM 指针隐藏延迟
    asm.emit2(gfx11::v_add_co_u32_vcc(34, 34, 233));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(35, 35));
    asm.emit2(gfx11::v_add_co_u32_vcc(36, 36, 233));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(37, 37));

    asm.emit(gfx11::s_waitcnt_vmcnt(0));

    // C. 带 132-byte Padding 写入 LDS 阵列 (使用 LLVM 验证的 ds_store_b128，opcode=0xDB7C0000)
    asm.emit2(gfx11::ds_store_b128(38, 0, 0));
    asm.emit2(gfx11::ds_store_b128(38, 4, 16));
    asm.emit2(gfx11::ds_store_b128(38, 8, 32));
    asm.emit2(gfx11::ds_store_b128(38, 12, 48));

    asm.emit2(gfx11::ds_store_b128(39, 16, 0));
    asm.emit2(gfx11::ds_store_b128(39, 20, 16));
    asm.emit2(gfx11::ds_store_b128(39, 24, 32));
    asm.emit2(gfx11::ds_store_b128(39, 28, 48));

    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    // D. [核心魔法] 跨列抽取转置 (Zero-VALU Tearing)
    // 利用 16-bit 硬件常数 Offset，将 A(K^T) 压入 v[168..199]，B(V^T) 压入 v[200..231]
    for g in 0..4u8 {
        for k in 0..8u8 {
            let off_lo = ((g as i32) * 32 + (2 * k as i32) * 132) as u16;
            let off_hi = ((g as i32) * 32 + (2 * k as i32 + 1) * 132) as u16;
            let v_idx = 168 + g * 8 + k;
            asm.emit2(gfx11::ds_load_u16_d16(v_idx, 232, off_lo));
            asm.emit2(gfx11::ds_load_u16_d16_hi(v_idx, 232, off_hi));
        }
    }
    for v in 0..4u8 {
        for k in 0..8u8 {
            let off_lo = (2112 + (v as i32) * 32 + (2 * k as i32) * 132) as u16;
            let off_hi = (2112 + (v as i32) * 32 + (2 * k as i32 + 1) * 132) as u16;
            let v_idx = 200 + v * 8 + k;
            asm.emit2(gfx11::ds_load_u16_d16(v_idx, 232, off_lo));
            asm.emit2(gfx11::ds_load_u16_d16_hi(v_idx, 232, off_hi));
        }
    }

    asm.emit(gfx11::s_waitcnt_lgkmcnt(0));

    // E. 16 路纯算力轰炸 (K^T * V 全矩阵外积)
    for g in 0..4u8 {
        for v in 0..4u8 {
            let acc = 40 + g * 32 + v * 8;
            let a_reg = 168 + g * 8;
            let b_reg = 200 + v * 8;
            asm.emit2(gfx11::v_wmma_f32_16x16x16_bf16(acc, a_reg, b_reg, acc));
        }
    }

    asm.emit(gfx11::s_mov_b32_imm(106, 0)); // 【硬件铁律】清零被污染的 VCC
    
    // F. Loop Control
    asm.emit(gfx11::s_add_u32_imm(25, 25, 16));
    asm.emit(gfx11::s_cmp_lt_u32(25, 8));   // s8 = C_chunk
    let branch_offset = asm.branch_offset(asm.current_pc(), loop_start);
    asm.emit(gfx11::s_cbranch_scc1(branch_offset));

    // ========================================================================
    // 5. 将 64x64 矩阵精确倒模回 HBM (硬件级 13-bit 偏移量优化)
    // ========================================================================
    asm.emit(gfx11::v_and_b32_imm(211, 32, 31));    // v211 = lane_id
    asm.emit(gfx11::v_and_b32_imm(212, 211, 15));   // v212 = lane_row
    asm.emit(gfx11::v_lshrrev_b32(213, 4, 211));    // v213 = lane_half

    asm.emit(gfx11::s_mov_b32(15, 11));   // s15 = N_chunks (from kernarg)
    asm.emit(gfx11::s_mul_i32(16, 21, 15));  
    asm.emit(gfx11::s_add_u32(16, 16, 20));  // head_id*N_chunks + chunk_id
    asm.emit(gfx11::s_mul_i32(17, 9, 9));    // 64 * 64 = 4096
    asm.emit(gfx11::s_mul_i32(18, 16, 17));  
    asm.emit(gfx11::s_lshl_b32(18, 18, 2));  // byte offset

    asm.emit(gfx11::v_mov_b32_from_sgpr(190, 6));   
    asm.emit(gfx11::v_mov_b32_from_sgpr(191, 7));   
    asm.emit(gfx11::v_mov_b32_from_sgpr(192, 18));
    asm.emit2(gfx11::v_add_co_u32_vcc(190, 190, 192));
    asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(191, 191)); // v[190:191] = W_base

    asm.emit(gfx11::v_mov_b32_from_sgpr(214, 9)); // v214 = 64
    
    for k_grp in 0..4u8 {
        if k_grp == 0 {
            asm.emit(gfx11::v_mov_b32(216, 213)); 
        } else {
            asm.emit(gfx11::v_add_u32_imm(216, 213, (k_grp as u32) * 16));
        }
        asm.emit2(gfx11::v_mul_lo_u32(217, 216, 214));  // base_row * 64
        asm.emit(gfx11::v_lshlrev_b32(217, 2, 217));    // * 4

        for v_tile in 0..4u8 {
            let acc_base = 40 + k_grp * 32 + v_tile * 8;
            let col_offset_bytes = (v_tile as u32) * 16 * 4;

            asm.emit(gfx11::v_mov_b32(218, 190));
            asm.emit(gfx11::v_mov_b32(219, 191));
            asm.emit2(gfx11::v_add_co_u32_vcc(218, 218, 217)); // Add row offset
            asm.emit2(gfx11::v_add_co_ci_u32_zero_vcc(219, 219));

            asm.emit(gfx11::v_lshlrev_b32(220, 2, 212));       // lane_row * 4
            asm.emit(gfx11::v_add_u32(218, 218, 220));

            if col_offset_bytes > 0 && col_offset_bytes <= 64 {
                asm.emit(gfx11::v_add_u32_imm(218, 218, col_offset_bytes));
            } else if col_offset_bytes > 64 {
                asm.emit2(gfx11::v_add_u32_literal(218, 218, col_offset_bytes));
            }

            // 极限优化：借用 13-bit Literal Offset 代替 7 条 v_add
            for r in 0..8u8 {
                let r_offset = (r as i32) * 512;
                asm.emit2(gfx11::global_store_dword(218, acc_base + r, r_offset));
            }
        }
    }

    asm.emit(gfx11::s_waitcnt_vmcnt(0));
    asm.emit(gfx11::s_waitcnt_vscnt(0));
    asm.emit(gfx11::S_ENDPGM);

    AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
        name: "ocpa_state_update".to_string(),
        lds_size: 4224,      // 完美的 4.125 KB ZCLT 金库
        kernarg_size: 40,
        vgpr_count: 234,     // 游刃有余地停留在 256 红线内
        sgpr_count: 32,
        workgroup_size_x: 32,
        workgroup_size_y: 1,
        workgroup_size_z: 1,
            scratch_size: 0,
    })
}
