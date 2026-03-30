//! ISA Probe — 自动指令编码查询 + 代码生成工具
//!
//! 调用 llvm-mc 查询 GFX11 指令编码，解析输出，
//! 生成可直接使用的 Rust 代码（ir.rs Op + asm_emitter.rs match 分支）。
//!
//! # 功能
//! - `probe_instruction()` — 单条指令编码查询
//! - `classify_format()` — 格式分类 (VOP1/VOP2/VOP3/SOP/SMEM/DS...)
//! - `discover_available()` — 发现目标 GPU 全部可用指令
//! - `diff_unused()` — 输出可用但未使用的指令清单
//! - `generate_rust_code()` — 生成 ir.rs + asm_emitter.rs 代码片段

use std::process::Command;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Data types
// ============================================================================

/// Instruction encoding format classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsaFormat {
    VOP1,   // 1-src vector ALU (4 bytes)
    VOP2,   // 2-src vector ALU (4 bytes)
    VOP3,   // 3-src or extended vector ALU (8 bytes)
    VOPC,   // Vector compare (4 bytes, writes VCC)
    SOP1,   // 1-src scalar ALU
    SOP2,   // 2-src scalar ALU
    SOPC,   // Scalar compare
    SOPP,   // Scalar program control
    SOPK,   // Scalar with 16-bit immediate
    SMEM,   // Scalar memory (8 bytes)
    DS,     // Data share / LDS (8 bytes)
    FLAT,   // Flat/Global memory (8 bytes)
    WMMA,   // Wave matrix multiply (8 bytes)
    VINTERP, // Interpolation
    Unknown,
}

impl std::fmt::Display for IsaFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Probed instruction info.
#[derive(Clone, Debug)]
pub struct InstructionInfo {
    /// Input mnemonic (e.g. "v_sin_f32")
    pub mnemonic: String,
    /// Canonical mnemonic from LLVM (e.g. "v_sin_f32_e32")
    pub canonical: String,
    /// Raw encoding bytes
    pub encoding: Vec<u8>,
    /// Detected instruction format
    pub format: IsaFormat,
    /// Number of encoding bytes (4 or 8)
    pub n_bytes: usize,
    /// Operand signature (e.g. "vdst, vsrc0")
    pub operand_sig: String,
}

/// Operand template for probing an instruction.
#[derive(Clone, Debug)]
pub struct OperandTemplate {
    /// Mnemonic prefix (e.g. "v_add_f32")
    pub mnemonic: String,
    /// Full test assembly line (e.g. "v_add_f32 v0, v1, v2")
    pub test_asm: String,
    /// Category for grouping
    pub category: &'static str,
}

// ============================================================================
// Core probing
// ============================================================================

/// Probe a single instruction by calling llvm-mc.
///
/// Returns `Ok(InstructionInfo)` if the instruction is valid,
/// `Err(msg)` if llvm-mc rejects it.
pub fn probe_instruction(test_asm: &str, target: &str) -> Result<InstructionInfo, String> {
    let output = Command::new("llvm-mc")
        .args(&[
            &format!("-mcpu={}", target),
            "--show-encoding",
            "-triple=amdgcn-amd-amdhsa",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                stdin.write_all(test_asm.as_bytes()).ok();
            }
            child.wait_with_output()
        })
        .map_err(|e| format!("llvm-mc failed to run: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() || !stderr.is_empty() {
        return Err(format!("llvm-mc error: {}", stderr.trim()));
    }

    parse_llvm_mc_output(&stdout, test_asm)
}

/// Parse llvm-mc --show-encoding output.
///
/// Example input lines:
/// ```text
///         v_sin_f32_e32 v0, v1                    ; encoding: [0x01,0x6b,0x00,0x7e]
///         v_fma_f32 v0, v1, v2, v3                ; encoding: [0x00,0x00,0x13,0xd6,0x01,0x05,0x0e,0x04]
/// ```
fn parse_llvm_mc_output(output: &str, original_asm: &str) -> Result<InstructionInfo, String> {
    // Find the line with "; encoding:"
    for line in output.lines() {
        let line = line.trim();
        if let Some(enc_pos) = line.find("; encoding:") {
            // Extract canonical mnemonic (everything before ; encoding:)
            let asm_part = line[..enc_pos].trim();
            let canonical = asm_part.split_whitespace().next().unwrap_or("").to_string();

            // Extract mnemonic from original asm
            let mnemonic = original_asm.trim().split_whitespace().next()
                .unwrap_or("").to_string();

            // Extract operand signature
            let operand_sig = if let Some(space_pos) = asm_part.find(' ') {
                asm_part[space_pos..].trim().to_string()
            } else {
                String::new()
            };

            // Parse encoding bytes: [0x01,0x6b,0x00,0x7e]
            let enc_str = &line[enc_pos + "; encoding:".len()..];
            let enc_str = enc_str.trim();
            let encoding = parse_encoding_bytes(enc_str)?;
            let n_bytes = encoding.len();

            let format = classify_format(&encoding, &canonical);

            return Ok(InstructionInfo {
                mnemonic,
                canonical,
                encoding,
                format,
                n_bytes,
                operand_sig,
            });
        }
    }

    Err(format!("No encoding found in llvm-mc output for '{}'", original_asm))
}

/// Parse encoding bytes from "[0x01,0x6b,0x00,0x7e]" format.
fn parse_encoding_bytes(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    let mut bytes = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() { continue; }
        let val = if part.starts_with("0x") || part.starts_with("0X") {
            u8::from_str_radix(&part[2..], 16)
        } else {
            part.parse::<u8>()
        };
        bytes.push(val.map_err(|e| format!("Bad encoding byte '{}': {}", part, e))?);
    }
    Ok(bytes)
}

// ============================================================================
// Format classification
// ============================================================================

/// Classify instruction format from encoding bytes and canonical mnemonic.
///
/// GFX11 encoding reference:
/// - 4 bytes: VOP1 (0x7E), VOP2 (0x00-0x3F), VOPC (0x7C), SOP1, SOP2, SOPC, SOPP
/// - 8 bytes: VOP3 (0xD5-0xD6), SMEM, DS (0xD8), FLAT/Global (0xDC), WMMA (0xCC)
pub fn classify_format(encoding: &[u8], canonical: &str) -> IsaFormat {
    let n = encoding.len();
    if n < 4 { return IsaFormat::Unknown; }

    // Use canonical mnemonic suffix as primary hint
    if canonical.contains("_e32") {
        // 4-byte VOP encoding — distinguish VOP1 vs VOP2
        let top_byte = encoding[3];
        if top_byte == 0x7E || (top_byte & 0xFE) == 0x7E {
            return IsaFormat::VOP1;
        }
        return IsaFormat::VOP2;
    }

    // 8-byte formats: check high byte (byte[3] in little-endian = instruction word high byte)
    if n == 8 {
        let hi_byte = encoding[3]; // bits[31:24] of first dword (little-endian)
        match hi_byte {
            0xD5 | 0xD6 => return IsaFormat::VOP3,
            0xD8 => return IsaFormat::DS,
            0xDC => return IsaFormat::FLAT,
            0xCC => return IsaFormat::WMMA,
            _ => {}
        }
        // SMEM: check prefix
        if canonical.starts_with("s_load") || canonical.starts_with("s_store") ||
           canonical.starts_with("s_buffer") {
            return IsaFormat::SMEM;
        }
    }

    // 4-byte scalar formats
    if n == 4 {
        let top_byte = encoding[3];
        // SOP2: 0x80-0xBE
        if top_byte >= 0x80 && top_byte <= 0xBE {
            return IsaFormat::SOP2;
        }
        // SOPP: 0xBF
        if top_byte == 0xBF {
            return IsaFormat::SOPP;
        }
        // SOP1: bits [31:23] = 10_1011_101 = 0xBE80..0xBEFF range
        // Actually SOP1 high byte is 0xBE for GFX11
        if top_byte == 0xBE {
            return IsaFormat::SOP1;
        }
        // VOPC: starts with v_cmp_
        if canonical.starts_with("v_cmp_") || canonical.starts_with("v_cmpx_") {
            return IsaFormat::VOPC;
        }
        // SOPK
        if canonical.starts_with("s_") && (top_byte >= 0xB0 && top_byte < 0xBE) {
            return IsaFormat::SOPK;
        }
        // SOPC
        if top_byte == 0xBF && canonical.starts_with("s_cmp_") {
            return IsaFormat::SOPC;
        }
    }

    // Mnemonic-based fallbacks
    if canonical.starts_with("v_wmma_") { return IsaFormat::WMMA; }
    if canonical.starts_with("ds_") { return IsaFormat::DS; }
    if canonical.starts_with("global_") { return IsaFormat::FLAT; }
    if canonical.starts_with("flat_") { return IsaFormat::FLAT; }

    IsaFormat::Unknown
}

// ============================================================================
// Instruction database — GFX11 operand templates
// ============================================================================

/// Build full GFX11 instruction probe database.
///
/// Returns a list of (mnemonic, test_asm, category) tuples.
/// Each entry is a valid assembly line that llvm-mc should accept.
pub fn gfx11_instruction_db() -> Vec<OperandTemplate> {
    let mut db = Vec::new();

    macro_rules! add {
        ($cat:expr, $mn:expr, $asm:expr) => {
            db.push(OperandTemplate {
                mnemonic: $mn.to_string(),
                test_asm: $asm.to_string(),
                category: $cat,
            });
        };
    }

    // ── VOP1: single-source vector ops ──
    let vop1_list = [
        "v_mov_b32", "v_cvt_f32_i32", "v_cvt_f32_u32", "v_cvt_u32_f32",
        "v_cvt_i32_f32", "v_cvt_f16_f32", "v_cvt_f32_f16", "v_cvt_f64_f32",
        "v_cvt_f32_f64", "v_cvt_f64_i32", "v_cvt_f64_u32",
        "v_fract_f32", "v_trunc_f32", "v_ceil_f32", "v_rndne_f32", "v_floor_f32",
        "v_exp_f32", "v_log_f32", "v_rcp_f32", "v_rcp_f64",
        "v_rsq_f32", "v_rsq_f64", "v_sqrt_f32", "v_sqrt_f64",
        "v_sin_f32", "v_cos_f32",
        "v_not_b32", "v_bfrev_b32", "v_cls_i32",
        "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2", "v_cvt_f32_ubyte3",
        "v_readfirstlane_b32",
        "v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16",
        "v_sat_pk_u8_i16",
    ];
    // f64 instructions need special operand templates
    let f64_dst_f32_src = ["v_cvt_f64_f32", "v_cvt_f64_i32", "v_cvt_f64_u32"];
    let f32_dst_f64_src = ["v_cvt_f32_f64"];
    let f64_dst_f64_src = ["v_rcp_f64", "v_rsq_f64", "v_sqrt_f64"];

    for mn in &vop1_list {
        if *mn == "v_readfirstlane_b32" {
            add!("VOP1", mn, format!("{} s0, v0", mn));
        } else if f64_dst_f32_src.contains(mn) {
            // dst=f64 v[0:1], src=f32 v2
            add!("VOP1", mn, format!("{} v[0:1], v2", mn));
        } else if f32_dst_f64_src.contains(mn) {
            // dst=f32 v0, src=f64 v[2:3]
            add!("VOP1", mn, format!("{} v0, v[2:3]", mn));
        } else if f64_dst_f64_src.contains(mn) {
            // dst=f64 v[0:1], src=f64 v[2:3]
            add!("VOP1", mn, format!("{} v[0:1], v[2:3]", mn));
        } else {
            add!("VOP1", mn, format!("{} v0, v1", mn));
        }
    }

    // ── VOP2: two-source vector ops ──
    let vop2_list = [
        "v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32",
        "v_max_f32", "v_min_f32",
        "v_add_f16", "v_sub_f16", "v_mul_f16", "v_max_f16", "v_min_f16",
        "v_and_b32", "v_or_b32", "v_xor_b32",
        "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32",
        "v_add_nc_u32", "v_sub_nc_u32",
        "v_mul_lo_u16", "v_mul_hi_u32_u24", "v_mul_hi_i32_i24",
        "v_mul_u32_u24", "v_mul_i32_i24",
        "v_cndmask_b32",
        "v_add_co_ci_u32",
        "v_sub_co_ci_u32",
        "v_add_f64",
        "v_mul_f64",
        "v_fmac_f32",
        "v_pk_add_f16", "v_pk_mul_f16",
    ];
    for mn in &vop2_list {
        if *mn == "v_cndmask_b32" {
            add!("VOP2", mn, format!("{} v0, v1, v2, vcc_lo", mn));
        } else if *mn == "v_add_co_ci_u32" || *mn == "v_sub_co_ci_u32" {
            add!("VOP2", mn, format!("{} v0, vcc_lo, v1, v2, vcc_lo", mn));
        } else if mn.contains("f64") {
            add!("VOP2", mn, format!("{} v[0:1], v[2:3], v[4:5]", mn));
        } else {
            add!("VOP2", mn, format!("{} v0, v1, v2", mn));
        }
    }

    // ── VOP3: three-source or extended ──
    let vop3_list = [
        "v_fma_f32", "v_fma_f64",
        "v_mad_u32_u24", "v_mad_i32_i24",
        "v_med3_f32", "v_med3_i32", "v_med3_u32",
        "v_min3_f32", "v_min3_i32", "v_min3_u32",
        "v_max3_f32", "v_max3_i32", "v_max3_u32",
        "v_mul_lo_u32",
        "v_mul_hi_u32", "v_mul_hi_i32",
        "v_add_co_u32",
        "v_sub_co_u32",
        "v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64",
        "v_and_or_b32", "v_or3_b32",
        "v_bfe_u32", "v_bfe_i32", "v_bfi_b32",
        "v_alignbyte_b32", "v_alignbit_b32",
        "v_perm_b32",
        "v_sad_u32",
        "v_permlanex16_b32",
        "v_permlane16_b32",
    ];
    for mn in &vop3_list {
        if *mn == "v_mul_lo_u32" || *mn == "v_mul_hi_u32" || *mn == "v_mul_hi_i32" {
            add!("VOP3", mn, format!("{} v0, v1, v2", mn));
        } else if *mn == "v_add_co_u32" || *mn == "v_sub_co_u32" {
            add!("VOP3", mn, format!("{} v0, vcc_lo, v1, v2", mn));
        } else if mn.contains("b64") || mn.contains("f64") {
            add!("VOP3", mn, format!("{} v[0:1], v[2:3], v[4:5]", mn));
        } else if *mn == "v_permlanex16_b32" || *mn == "v_permlane16_b32" {
            add!("VOP3", mn, format!("{} v0, v1, s0, s0", mn));
        } else if mn.contains("3_") || mn.contains("or_") || mn.contains("fi_") {
            add!("VOP3", mn, format!("{} v0, v1, v2, v3", mn));
        } else {
            add!("VOP3", mn, format!("{} v0, v1, v2, v3", mn));
        }
    }

    // ── VOPC: vector compare ──
    let vopc_list = [
        "v_cmp_lt_f32", "v_cmp_eq_f32", "v_cmp_le_f32", "v_cmp_gt_f32",
        "v_cmp_ge_f32", "v_cmp_neq_f32",
        "v_cmp_lt_u32", "v_cmp_eq_u32", "v_cmp_le_u32", "v_cmp_gt_u32",
        "v_cmp_ge_u32", "v_cmp_ne_u32",
        "v_cmp_lt_i32", "v_cmp_eq_i32", "v_cmp_le_i32", "v_cmp_gt_i32",
        "v_cmp_ge_i32", "v_cmp_ne_i32",
        "v_cmp_lt_f16", "v_cmp_eq_f16", "v_cmp_le_f16",
        "v_cmp_lt_u16", "v_cmp_eq_u16",
    ];
    for mn in &vopc_list {
        add!("VOPC", mn, format!("{} vcc_lo, v0, v1", mn));
    }

    // ── SOP2: 2-src scalar ──
    let sop2_list = [
        "s_add_u32", "s_sub_u32", "s_addc_u32", "s_subb_u32",
        "s_and_b32", "s_or_b32", "s_xor_b32", "s_andn2_b32", "s_orn2_b32",
        "s_and_b64", "s_or_b64", "s_xor_b64",
        "s_lshl_b32", "s_lshr_b32", "s_ashr_i32",
        "s_lshl_b64", "s_lshr_b64", "s_ashr_i64",
        "s_mul_i32", "s_mul_hi_u32", "s_mul_hi_i32",
        "s_bfe_u32", "s_bfe_i32",
        "s_min_u32", "s_min_i32", "s_max_u32", "s_max_i32",
        "s_cselect_b32",
    ];
    for mn in &sop2_list {
        if mn.contains("b64") || mn.contains("i64") {
            add!("SOP2", mn, format!("{} s[0:1], s[2:3], s[4:5]", mn));
        } else {
            add!("SOP2", mn, format!("{} s0, s1, s2", mn));
        }
    }

    // ── SOP1: 1-src scalar ──
    let sop1_list = [
        "s_mov_b32", "s_mov_b64",
        "s_not_b32", "s_not_b64",
        "s_brev_b32",
        "s_and_saveexec_b32", "s_or_saveexec_b32",
        "s_getpc_b64",
    ];
    for mn in &sop1_list {
        if *mn == "s_and_saveexec_b32" || *mn == "s_or_saveexec_b32" {
            add!("SOP1", mn, format!("{} s0, vcc_lo", mn));
        } else if *mn == "s_getpc_b64" {
            add!("SOP1", mn, format!("{} s[0:1]", mn));
        } else if mn.contains("b64") {
            add!("SOP1", mn, format!("{} s[0:1], s[2:3]", mn));
        } else {
            add!("SOP1", mn, format!("{} s0, s1", mn));
        }
    }

    // ── SOPC: scalar compare ──
    let sopc_list = [
        "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32",
        "s_cmp_lt_u32", "s_cmp_le_u32",
        "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32",
        "s_cmp_lt_i32", "s_cmp_le_i32",
    ];
    for mn in &sopc_list {
        add!("SOPC", mn, format!("{} s0, s1", mn));
    }

    // ── SOPP: scalar program control ──
    let sopp_list = [
        "s_endpgm", "s_barrier", "s_nop",
        "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
        "s_cbranch_vccz", "s_cbranch_vccnz",
        "s_cbranch_execz", "s_cbranch_execnz",
    ];
    for mn in &sopp_list {
        if *mn == "s_endpgm" || *mn == "s_barrier" {
            add!("SOPP", mn, mn.to_string());
        } else if *mn == "s_nop" {
            add!("SOPP", mn, format!("{} 0", mn));
        } else {
            add!("SOPP", mn, format!("{} 0", mn)); // branch offset = 0
        }
    }

    // ── SMEM: scalar memory ──
    let smem_list = [
        "s_load_b32", "s_load_b64", "s_load_b128", "s_load_b256",
    ];
    for mn in &smem_list {
        let dst = match *mn {
            "s_load_b32" => "s0",
            "s_load_b64" => "s[0:1]",
            "s_load_b128" => "s[0:3]",
            "s_load_b256" => "s[0:7]",
            _ => "s0",
        };
        add!("SMEM", mn, format!("{} {}, s[2:3], 0x0", mn, dst));
    }

    // ── DS: data share / LDS ──
    let ds_list = [
        "ds_load_b32", "ds_load_b64", "ds_load_b128",
        "ds_load_u8", "ds_load_u16",
        "ds_store_b8", "ds_store_b16", "ds_store_b32", "ds_store_b64", "ds_store_b128",
        "ds_swizzle_b32",
        "ds_add_u32", "ds_add_f32",
        "ds_min_u32", "ds_max_u32", "ds_min_f32", "ds_max_f32",
        "ds_load_u16_d16", "ds_load_u16_d16_hi",
    ];
    for mn in &ds_list {
        if *mn == "ds_swizzle_b32" {
            add!("DS", mn, format!("{} v0, v1 offset:0x0000", mn));
        } else if mn.starts_with("ds_load") {
            let dst = if mn.contains("b64") { "v[0:1]" }
                else if mn.contains("b128") { "v[0:3]" }
                else { "v0" };
            add!("DS", mn, format!("{} {}, v2", mn, dst));
        } else if mn.starts_with("ds_store") {
            let src = if mn.contains("b64") { "v[1:2]" }
                else if mn.contains("b128") { "v[1:4]" }
                else { "v1" };
            add!("DS", mn, format!("{} v0, {}", mn, src));
        } else {
            // atomics: ds_add_u32 v_addr, v_src
            add!("DS", mn, format!("{} v0, v1", mn));
        }
    }

    // ── FLAT/Global: global memory ──
    let flat_list = [
        "global_load_b32", "global_load_b64", "global_load_b128",
        "global_load_u8", "global_load_u16",
        "global_store_b8", "global_store_b16", "global_store_b32",
        "global_store_b64", "global_store_b128",
        "global_atomic_add_u32", "global_atomic_add_f32",
        "global_atomic_cmpswap_b32",
    ];
    for mn in &flat_list {
        if mn.starts_with("global_load") {
            let dst = if mn.contains("b64") { "v[0:1]" }
                else if mn.contains("b128") { "v[0:3]" }
                else { "v0" };
            add!("FLAT", mn, format!("{} {}, v[2:3], off", mn, dst));
        } else if mn.starts_with("global_store") {
            let src = if mn.contains("b64") { "v[2:3]" }
                else if mn.contains("b128") { "v[2:5]" }
                else { "v2" };
            add!("FLAT", mn, format!("{} v[0:1], {}, off", mn, src));
        } else if mn.contains("cmpswap") {
            add!("FLAT", mn, format!("{} v0, v[1:2], v[3:4], off", mn));
        } else {
            // atomics
            add!("FLAT", mn, format!("{} v[0:1], v2, off", mn));
        }
    }

    // ── WMMA ──
    let wmma_list = [
        "v_wmma_f32_16x16x16_bf16",
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_bf16_16x16x16_bf16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_i32_16x16x16_iu8",
    ];
    for mn in &wmma_list {
        let dst = if mn.contains("f32") || mn.contains("i32") {
            "v[0:7]"
        } else {
            "v[0:7]"
        };
        add!("WMMA", mn, format!("{} {}, v[8:15], v[16:23], v[24:31]", mn, dst));
    }

    // ── Waitcnt (special SOPP) ──
    add!("SOPP", "s_waitcnt", "s_waitcnt vmcnt(0)");
    add!("SOPP", "s_waitcnt_vscnt", "s_waitcnt_vscnt null, 0x0");

    db
}

// ============================================================================
// Already-implemented instructions (extracted from asm_emitter.rs)
// ============================================================================

/// Return the set of mnemonics already implemented in T0's asm_emitter.
///
/// This is a static list extracted from asm_emitter.rs match arms.
pub fn implemented_mnemonics() -> HashSet<String> {
    let list = [
        // Global memory
        "global_load_u16", "global_load_b32", "global_load_b64", "global_load_b128",
        "global_store_b16", "global_store_b32", "global_store_b64", "global_store_b128",
        "global_atomic_add_f32",
        // LDS
        "ds_load_u16", "ds_load_b32", "ds_load_b64", "ds_load_b128",
        "ds_store_b16", "ds_store_b32", "ds_store_b64", "ds_store_b128",
        "ds_load_u16_d16", "ds_load_u16_d16_hi",
        "ds_swizzle_b32",
        // Scalar memory
        "s_load_b32", "s_load_b64", "s_load_b128",
        "s_load_dword", // legacy alias
        // VOP2 / VOP1
        "v_add_f32", "v_mul_f32", "v_max_f32", "v_min_f32",
        "v_mov_b32", "v_add_nc_u32", "v_mul_lo_u32",
        "v_lshlrev_b32", "v_lshrrev_b32",
        "v_and_b32", "v_or_b32", "v_xor_b32",
        "v_sub_f32", "v_sub_u32",
        "v_readfirstlane_b32",
        "v_cndmask_b32",
        // VOP3
        "v_fma_f32", "v_and_or_b32",
        "v_add_co_u32", "v_add_co_ci_u32",
        "v_permlanex16_b32",
        // VOP1 special math
        "v_rsq_f32", "v_exp_f32", "v_rcp_f32", "v_sqrt_f32", "v_log_f32",
        "v_cvt_f32_u32", "v_cvt_u32_f32",
        // VOPC
        "v_cmp_lt_u32", "v_cmp_ge_u32", "v_cmp_gt_f32",
        "v_cmp_gt_u32", "v_cmp_eq_u32", "v_cmp_ge_i32",
        // Scalar ALU
        "s_add_u32", "s_addc_u32", "s_sub_u32",
        "s_and_b32", "s_mul_i32",
        "s_lshl_b32", "s_lshr_b32",
        "s_mov_b32",
        "s_cmp_lt_u32", "s_cmp_eq_u32", "s_cmp_ge_u32",
        "s_and_saveexec_b32",
        // SOPP
        "s_endpgm", "s_barrier",
        "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
        "s_cbranch_vccz",
        "s_waitcnt", "s_waitcnt_vscnt",
        // WMMA
        "v_wmma_f32_16x16x16_bf16", "v_wmma_f32_16x16x16_f16",
        "v_wmma_bf16_16x16x16_bf16",
    ];
    list.iter().map(|s| s.to_string()).collect()
}

// ============================================================================
// Diff: find unused instructions
// ============================================================================

/// Discover all available instructions and diff against implemented set.
///
/// Returns (available, unused) where:
/// - available: all instructions that llvm-mc accepts for the target
/// - unused: available minus already implemented
pub fn discover_and_diff(target: &str) -> (Vec<InstructionInfo>, Vec<InstructionInfo>) {
    let db = gfx11_instruction_db();
    let implemented = implemented_mnemonics();

    let mut available = Vec::new();
    let mut unused = Vec::new();

    for template in &db {
        match probe_instruction(&template.test_asm, target) {
            Ok(info) => {
                let is_implemented = implemented.contains(&template.mnemonic);
                if !is_implemented {
                    unused.push(info.clone());
                }
                available.push(info);
            }
            Err(_) => {
                // Instruction not supported on this target — skip
            }
        }
    }

    (available, unused)
}

// ============================================================================
// Rust code generation
// ============================================================================

/// Generate Rust code snippets for integrating an instruction into T0.
///
/// Returns (ir_op_variant, asm_emitter_match, vreg_refs_match, vreg_defs_match)
pub fn generate_rust_code(info: &InstructionInfo) -> String {
    let mut out = String::new();

    // Determine operand structure from format
    let rust_name = mnemonic_to_rust_name(&info.mnemonic);

    out.push_str(&format!("// ═══════════════════════════════════════════\n"));
    out.push_str(&format!("// {} — llvm-mc verified\n", info.mnemonic));
    out.push_str(&format!("// encoding: {:?} ({} bytes, format: {})\n",
        info.encoding, info.n_bytes, info.format));
    out.push_str(&format!("// canonical: {}\n", info.canonical));
    out.push_str(&format!("// ═══════════════════════════════════════════\n\n"));

    // 1. ir.rs Op variant
    out.push_str("// ── Add to ir.rs Op enum ──\n");
    match info.format {
        IsaFormat::VOP1 => {
            out.push_str(&format!(
                "/// {}: {}\n{} {{ dst: VReg, src: VReg }},\n\n",
                info.mnemonic, info.operand_sig, rust_name
            ));
        }
        IsaFormat::VOP2 => {
            out.push_str(&format!(
                "/// {}: {}\n{} {{ dst: VReg, src0: Operand, src1: Operand }},\n\n",
                info.mnemonic, info.operand_sig, rust_name
            ));
        }
        IsaFormat::VOP3 => {
            if info.operand_sig.matches(',').count() >= 3 {
                out.push_str(&format!(
                    "/// {}: {}\n{} {{ dst: VReg, src0: Operand, src1: Operand, src2: Operand }},\n\n",
                    info.mnemonic, info.operand_sig, rust_name
                ));
            } else {
                out.push_str(&format!(
                    "/// {}: {}\n{} {{ dst: VReg, src0: VReg, src1: VReg }},\n\n",
                    info.mnemonic, info.operand_sig, rust_name
                ));
            }
        }
        _ => {
            out.push_str(&format!(
                "/// {}: {}\n// TODO: determine operand structure for format {}\n{} {{ /* ... */ }},\n\n",
                info.mnemonic, info.operand_sig, info.format, rust_name
            ));
        }
    }

    // 2. asm_emitter.rs match arm
    out.push_str("// ── Add to asm_emitter.rs emit_op() match ──\n");
    match info.format {
        IsaFormat::VOP1 => {
            out.push_str(&format!(
                "Op::{} {{ dst, src }} => {{\n    let vd = a.phys_v(*dst);\n    let vs = a.phys_v(*src);\n    writeln!(self.buf, \"{{}}{}  v{{}}, v{{}}\", self.indent, vd, vs).unwrap();\n}}\n\n",
                rust_name, info.mnemonic
            ));
        }
        IsaFormat::VOP2 => {
            out.push_str(&format!(
                "Op::{} {{ dst, src0, src1 }} => {{\n    let vd = a.phys_v(*dst);\n    writeln!(self.buf, \"{{}}{} v{{}}, {{}}, {{}}\",\n        self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();\n}}\n\n",
                rust_name, info.mnemonic
            ));
        }
        _ => {
            out.push_str(&format!(
                "// TODO: implement asm_emitter match for {} (format: {})\n\n",
                rust_name, info.format
            ));
        }
    }

    // 3. vreg_refs / vreg_defs additions
    out.push_str("// ── Add to vreg_refs() / vreg_defs() in ir.rs ──\n");
    match info.format {
        IsaFormat::VOP1 => {
            out.push_str(&format!(
                "// vreg_refs: Op::{} {{ dst, src }} => vec![*dst, *src],\n",
                rust_name
            ));
            out.push_str(&format!(
                "// vreg_defs: Op::{} {{ dst, .. }} => vec![*dst],\n\n",
                rust_name
            ));
        }
        IsaFormat::VOP2 => {
            out.push_str(&format!(
                "// vreg_refs: add to existing VOP2 match arm\n"
            ));
            out.push_str(&format!(
                "// vreg_defs: add to existing VOP2 match arm\n\n"
            ));
        }
        _ => {
            out.push_str("// TODO: add vreg tracking\n\n");
        }
    }

    out
}

/// Convert mnemonic like "v_sin_f32" to Rust enum name like "VSinF32".
fn mnemonic_to_rust_name(mnemonic: &str) -> String {
    mnemonic
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

// ============================================================================
// Pretty-print helpers
// ============================================================================

/// Format an InstructionInfo for display.
pub fn format_instruction(info: &InstructionInfo) -> String {
    let enc_str: Vec<String> = info.encoding.iter().map(|b| format!("0x{:02x}", b)).collect();
    format!("{:<30} {:>5}  [{}]  {}",
        info.mnemonic, info.format, enc_str.join(","), info.operand_sig)
}

/// Format unused instruction list grouped by category.
pub fn format_unused_report(
    unused: &[InstructionInfo],
    db: &[OperandTemplate],
) -> String {
    let mut out = String::new();
    let mut by_category: HashMap<&str, Vec<&InstructionInfo>> = HashMap::new();

    // Build mnemonic → category map
    let mn_to_cat: HashMap<&str, &str> = db.iter()
        .map(|t| (t.mnemonic.as_str(), t.category))
        .collect();

    for info in unused {
        let cat = mn_to_cat.get(info.mnemonic.as_str()).copied().unwrap_or("Unknown");
        by_category.entry(cat).or_default().push(info);
    }

    // Sort categories
    let mut cats: Vec<&&str> = by_category.keys().collect();
    cats.sort();

    out.push_str(&format!("# 可用但未使用的 GFX11 指令 ({} 条)\n\n", unused.len()));

    for cat in cats {
        let instrs = &by_category[cat];
        out.push_str(&format!("## {} ({} 条)\n", cat, instrs.len()));
        for info in instrs {
            out.push_str(&format!("  {}\n", format_instruction(info)));
        }
        out.push_str("\n");
    }

    out
}

// ============================================================================
// Batch code generation with file output
// ============================================================================

/// Batch probe and generate code for multiple instructions.
///
/// Returns (ir_ops_code, emitter_code, summary_md, errors).
pub fn generate_batch(mnemonics: &[String], target: &str) -> BatchOutput {
    let db = gfx11_instruction_db();
    let mut ir_ops = String::new();
    let mut emitter_code = String::new();
    let mut summary = String::new();
    let mut errors: Vec<String> = Vec::new();
    let mut count = 0;

    ir_ops.push_str("// ═══════════════════════════════════════════════\n");
    ir_ops.push_str("// Auto-generated by isa_probe codegen\n");
    ir_ops.push_str("// Add these variants to the Op enum in ir.rs\n");
    ir_ops.push_str("// ═══════════════════════════════════════════════\n\n");

    emitter_code.push_str("// ═══════════════════════════════════════════════\n");
    emitter_code.push_str("// Auto-generated by isa_probe codegen\n");
    emitter_code.push_str("// Add these match arms to emit_op() in asm_emitter.rs\n");
    emitter_code.push_str("// ═══════════════════════════════════════════════\n\n");

    summary.push_str(&format!("# ISA Codegen 报告 (target: {})\n\n", target));
    summary.push_str("| 指令 | 格式 | 编码 | Rust 名 |\n");
    summary.push_str("|------|------|------|--------|\n");

    for mnemonic in mnemonics {
        // Find template in DB
        let template = db.iter().find(|t| t.mnemonic == *mnemonic);
        let test_asm = if let Some(t) = template {
            t.test_asm.clone()
        } else {
            // Auto-guess operand patterns
            let guesses = [
                format!("{} v0, v1", mnemonic),
                format!("{} v0, v1, v2", mnemonic),
                format!("{} v0, v1, v2, v3", mnemonic),
                format!("{} s0, s1, s2", mnemonic),
                format!("{} s0, s1", mnemonic),
                mnemonic.clone(),
            ];
            let mut found = None;
            for g in &guesses {
                if probe_instruction(g, target).is_ok() {
                    found = Some(g.clone());
                    break;
                }
            }
            match found {
                Some(asm) => asm,
                None => {
                    errors.push(format!("{}: not found on {}", mnemonic, target));
                    continue;
                }
            }
        };

        match probe_instruction(&test_asm, target) {
            Ok(info) => {
                count += 1;
                let rust_name = mnemonic_to_rust_name(&info.mnemonic);
                let enc_hex: Vec<String> = info.encoding.iter()
                    .map(|b| format!("0x{:02X}", b)).collect();

                // IR Op variant
                match info.format {
                    IsaFormat::VOP1 => {
                        ir_ops.push_str(&format!(
                            "/// {} (llvm-mc verified, {})\n{} {{ dst: VReg, src: VReg }},\n",
                            info.mnemonic, info.format, rust_name
                        ));
                    }
                    IsaFormat::VOP2 => {
                        ir_ops.push_str(&format!(
                            "/// {} (llvm-mc verified, {})\n{} {{ dst: VReg, src0: Operand, src1: Operand }},\n",
                            info.mnemonic, info.format, rust_name
                        ));
                    }
                    IsaFormat::VOP3 => {
                        if info.operand_sig.matches(',').count() >= 3 {
                            ir_ops.push_str(&format!(
                                "/// {} (llvm-mc verified, {})\n{} {{ dst: VReg, src0: Operand, src1: Operand, src2: Operand }},\n",
                                info.mnemonic, info.format, rust_name
                            ));
                        } else {
                            ir_ops.push_str(&format!(
                                "/// {} (llvm-mc verified, {})\n{} {{ dst: VReg, src0: VReg, src1: VReg }},\n",
                                info.mnemonic, info.format, rust_name
                            ));
                        }
                    }
                    IsaFormat::VOPC => {
                        ir_ops.push_str(&format!(
                            "/// {} (llvm-mc verified, {})\n{} {{ src0: Operand, src1: Operand }},\n",
                            info.mnemonic, info.format, rust_name
                        ));
                    }
                    _ => {
                        ir_ops.push_str(&format!(
                            "/// {} (llvm-mc verified, {}) — TODO: operand structure\n// {} {{ /* ... */ }},\n",
                            info.mnemonic, info.format, rust_name
                        ));
                    }
                }

                // Asm emitter match arm
                match info.format {
                    IsaFormat::VOP1 => {
                        emitter_code.push_str(&format!(
                            "Op::{} {{ dst, src }} => {{\n    let vd = a.phys_v(*dst);\n    let vs = a.phys_v(*src);\n    writeln!(self.buf, \"{{}}{}  v{{}}, v{{}}\", self.indent, vd, vs).unwrap();\n}}\n",
                            rust_name, info.mnemonic
                        ));
                    }
                    IsaFormat::VOP2 => {
                        emitter_code.push_str(&format!(
                            "Op::{} {{ dst, src0, src1 }} => {{\n    let vd = a.phys_v(*dst);\n    writeln!(self.buf, \"{{}}{} v{{}}, {{}}, {{}}\",\n        self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();\n}}\n",
                            rust_name, info.mnemonic
                        ));
                    }
                    IsaFormat::VOPC => {
                        emitter_code.push_str(&format!(
                            "Op::{} {{ src0, src1 }} => {{\n    writeln!(self.buf, \"{{}}{} vcc_lo, {{}}, {{}}\",\n        self.indent, operand_str(src0, a), operand_str(src1, a)).unwrap();\n}}\n",
                            rust_name, info.mnemonic
                        ));
                    }
                    _ => {
                        emitter_code.push_str(&format!(
                            "// TODO: Op::{} (format: {})\n",
                            rust_name, info.format
                        ));
                    }
                }

                // Summary table row
                summary.push_str(&format!(
                    "| `{}` | {} | `[{}]` | `{}` |\n",
                    info.mnemonic, info.format, enc_hex.join(","), rust_name
                ));
            }
            Err(e) => {
                errors.push(format!("{}: {}", mnemonic, e));
            }
        }
    }

    summary.push_str(&format!("\n共 {} 条指令生成成功", count));
    if !errors.is_empty() {
        summary.push_str(&format!(", {} 条失败:\n", errors.len()));
        for e in &errors {
            summary.push_str(&format!("- {}\n", e));
        }
    }
    summary.push_str("\n");

    BatchOutput { ir_ops, emitter_code, summary, errors, count }
}

/// Output of batch code generation.
pub struct BatchOutput {
    pub ir_ops: String,
    pub emitter_code: String,
    pub summary: String,
    pub errors: Vec<String>,
    pub count: usize,
}

/// Write batch codegen output to a directory.
///
/// Creates:
/// - `<dir>/ir_ops.rs`       — Op enum variants
/// - `<dir>/asm_emitter.rs`  — emit_op match arms
/// - `<dir>/summary.md`      — encoding summary table
pub fn write_codegen_files(output: &BatchOutput, dir: &str) -> Result<(), String> {
    use std::fs;
    use std::path::Path;

    let dir = Path::new(dir);
    fs::create_dir_all(dir)
        .map_err(|e| format!("Failed to create output dir: {}", e))?;

    fs::write(dir.join("ir_ops.rs"), &output.ir_ops)
        .map_err(|e| format!("Failed to write ir_ops.rs: {}", e))?;
    fs::write(dir.join("asm_emitter.rs"), &output.emitter_code)
        .map_err(|e| format!("Failed to write asm_emitter.rs: {}", e))?;
    fs::write(dir.join("summary.md"), &output.summary)
        .map_err(|e| format!("Failed to write summary.md: {}", e))?;

    Ok(())
}
