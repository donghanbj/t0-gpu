//! T0 Intermediate Representation
//!
//! Defines virtual registers and operations for the T0 kernel compiler.
//! All registers are virtual — physical allocation happens in regalloc.rs.

use std::fmt;

// ============================================================================
// Virtual Registers
// ============================================================================

/// Virtual VGPR (vector general-purpose register).
/// Allocated to physical VGPRs by the register allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VReg(pub u32);

/// Virtual SGPR (scalar general-purpose register).
/// Allocated to physical SGPRs by the register allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SReg(pub u32);

/// Virtual SGPR pair (64-bit pointer in two adjacent SGPRs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SRegPair(pub u32);  // refers to SReg(n) and SReg(n+1)

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%v{}", self.0)
    }
}

impl fmt::Display for SReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%s{}", self.0)
    }
}

impl fmt::Display for SRegPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%s[{}:{}]", self.0, self.0 + 1)
    }
}

// ============================================================================
// Alignment constraints
// ============================================================================

/// Alignment constraint for register allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    /// No alignment required.
    None,
    /// Must be 2-aligned (SGPR pairs, dwordx2 loads).
    Align2,
    /// Must be 4-aligned (dwordx4 loads).
    Align4,
    /// Must be 8-aligned (WMMA operands: 8 consecutive VGPRs).
    Align8,
}

// ============================================================================
// Data widths
// ============================================================================

/// Memory access width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    /// 16-bit (bf16/f16)
    B16,
    /// 32-bit (f32/u32)
    B32,
    /// 64-bit (2×f32, pointer)
    B64,
    /// 128-bit (4×f32, dwordx4)
    B128,
}

impl Width {
    /// Number of consecutive VGPRs consumed by this width.
    pub fn vreg_count(&self) -> u32 {
        match self {
            Width::B16 => 1,
            Width::B32 => 1,
            Width::B64 => 2,
            Width::B128 => 4,
        }
    }

    /// Byte count.
    pub fn bytes(&self) -> u32 {
        match self {
            Width::B16 => 2,
            Width::B32 => 4,
            Width::B64 => 8,
            Width::B128 => 16,
        }
    }
}

// ============================================================================
// WMMA format
// ============================================================================

/// WMMA instruction variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmmaFormat {
    /// v_wmma_f32_16x16x16_bf16: BF16 inputs, F32 accumulator (most common)
    BF16_F32,
    /// v_wmma_f32_16x16x16_f16: FP16 inputs, F32 accumulator
    F16_F32,
    /// v_wmma_bf16_16x16x16_bf16: BF16 inputs, BF16 accumulator (saves VGPRs)
    BF16_BF16,
}

// ============================================================================
// Operands
// ============================================================================

/// A vector operand: either a virtual register or an inline constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    /// Virtual VGPR
    VReg(VReg),
    /// Inline integer constant (0..64, or -1..-16)
    InlineInt(i32),
    /// Inline float constant (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0)
    InlineFloat(f32),
    /// 32-bit literal constant (requires extra dword)
    Literal(u32),
}

/// A scalar operand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SOperand {
    /// Virtual SGPR
    SReg(SReg),
    /// Inline integer constant
    InlineInt(i32),
    /// 32-bit literal
    Literal(u32),
}

// ============================================================================
// IR Operations
// ============================================================================

/// GPU target architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    GFX1100,  // RDNA3, Navi 31
    // Future: GFX1030, GFX900, etc.
}

impl Target {
    pub fn mcpu_str(&self) -> &'static str {
        match self {
            Target::GFX1100 => "gfx1100",
        }
    }
}

/// A single IR operation.
#[derive(Clone, Debug)]
pub enum Op {
    // ── Global memory ──
    GlobalLoad {
        dst: VReg,
        addr: VReg, // lo register of 64-bit addr pair (addr, addr+1)
        width: Width,
        offset: i32,
    },
    GlobalStore {
        addr: VReg, // lo register of 64-bit addr pair
        src: VReg,
        width: Width,
        offset: i32,
    },

    // ── LDS (Local Data Share) ──
    LdsLoad {
        dst: VReg,
        addr: VReg,
        width: Width,
        offset: u16,
    },
    LdsStore {
        addr: VReg,
        src: VReg,
        width: Width,
        offset: u16,
    },

    // ── Scalar memory ──
    ScalarLoad {
        dst: SReg,   // destination SGPR (or first of pair/quad)
        base: SRegPair, // base pointer pair
        offset: u32,
        width: Width, // B32, B64, or B128
    },

    // ── Vector ALU ──
    VAddF32 { dst: VReg, src0: Operand, src1: Operand },
    VMulF32 { dst: VReg, src0: Operand, src1: Operand },
    VFmaF32 { dst: VReg, src0: Operand, src1: Operand, src2: Operand },
    VMaxF32 { dst: VReg, src0: Operand, src1: Operand },
    VMinF32 { dst: VReg, src0: Operand, src1: Operand },
    VMov { dst: VReg, src: Operand },
    VMovFromSgpr { dst: VReg, src: SReg },
    VAddU32 { dst: VReg, src0: Operand, src1: Operand },
    VMulLoU32 { dst: VReg, src0: VReg, src1: VReg },
    VLshlrevB32 { dst: VReg, shift: u8, src: VReg },
    VLshrrevB32 { dst: VReg, shift: u8, src: VReg },
    VAndB32 { dst: VReg, src0: Operand, src1: Operand },
    VReadfirstlane { dst: SReg, src: VReg },

    // ── 64-bit address arithmetic ──
    VAddCo { dst: VReg, src0: VReg, src1: VReg },  // add with carry-out to VCC
    VAddCoCi { dst: VReg, src: VReg },              // add carry-in from VCC

    // ── Scalar ALU ──
    SAddU32 { dst: SReg, src0: SReg, src1: SOperand },
    /// s_addc_u32: scalar add with carry from previous s_add_u32
    SAddcU32 { dst: SReg, src0: SReg, src1: SOperand },
    SSubU32 { dst: SReg, src0: SReg, src1: SOperand },
    SAndB32 { dst: SReg, src0: SReg, src1: SOperand },
    SMulI32 { dst: SReg, src0: SReg, src1: SReg },
    SLshlB32 { dst: SReg, src: SReg, shift: u8 },
    SLshrB32 { dst: SReg, src: SReg, shift: u8 },
    SMov { dst: SReg, src: SOperand },
    SCmpLtU32 { src0: SReg, src1: SReg },
    SCmpEqU32 { src0: SReg, src1: SOperand },
    SCmpGeU32 { src0: SReg, src1: SReg },

    // ── WMMA (Wave Matrix Multiply Accumulate) ──
    Wmma {
        dst: VReg,  // first of 8 consecutive VGPRs
        a: VReg,    // first of 8 consecutive VGPRs (A fragment)
        b: VReg,    // first of 8 consecutive VGPRs (B fragment)
        c: VReg,    // first of 8 consecutive VGPRs (accumulator input)
        format: WmmaFormat,
    },

    // ── Control flow ──
    /// Label marker (not an instruction, used for branch targets)
    Label(String),
    /// Conditional branch to label if SCC==1
    BranchScc1(String),
    /// Unconditional branch to label
    Branch(String),

    // ── Synchronization ──
    Barrier,
    WaitVmcnt(u8),
    WaitLgkmcnt(u8),
    WaitVscnt(u8),
    /// Clear VCC (s_mov_b32 vcc_lo, 0) — prevent carry residual from mask ops
    ClearVcc,

    // ── Program structure ──
    Endpgm,

    // ── Hardware register access ──
    /// Copy hardware TGID (workgroup ID) to a virtual SGPR.
    /// axis: 0=X(s2), 1=Y(s3), 2=Z(s4)
    CaptureTgid { dst: SReg, axis: u8 },

    /// Compute global thread ID for 1D dispatch:
    /// dst = TGID.x * wg_size + WORKITEM_ID_X (v0)
    /// Clobbers s2 (TGID.x).
    ComputeGlobalIdX { dst: VReg, wg_size: u32 },

    // ── Cross-lane operations ──
    /// ds_swizzle_b32: cross-lane data exchange within a wave.
    /// offset encodes the swizzle pattern (XOR mode: 0x0000 | xor_mask).
    /// GFX11 XOR patterns: 0x401F(xor16), 0x201F(xor8), 0x101F(xor4), 0x081F(xor2), 0x041F(xor1)
    DsSwizzle { dst: VReg, src: VReg, offset: u16 },

    // ── Special math ──
    /// v_rsq_f32: reciprocal square root (1/sqrt(x))
    VRsqF32 { dst: VReg, src: VReg },

    /// v_exp_f32: compute 2^x (NOT e^x!)
    /// For natural exp: v_mul_f32(x, log2e); v_exp_f32(x)
    VExpF32 { dst: VReg, src: VReg },

    /// v_rcp_f32: reciprocal (1/x)
    VRcpF32 { dst: VReg, src: VReg },

    /// v_xor_b32: bitwise XOR (used for sign bit flip with 0x80000000)
    VXorB32 { dst: VReg, src0: Operand, src1: Operand },

    /// v_sub_f32: floating point subtraction dst = src0 - src1
    VSubF32 { dst: VReg, src0: Operand, src1: Operand },

    /// Wave-level butterfly reduction: sum all 32 lanes.
    /// Emits 5× ds_swizzle + v_add_f32 sequence (xor16, xor8, xor4, xor2, xor1).
    /// Result: every lane has the sum of all 32 lanes.
    WaveReduceAddF32 { val: VReg, tmp: VReg },

    // ── Data type conversion ──
    /// Pack two f32 values into one bf16x2: dst = (bf16(src1) << 16) | bf16(src0)
    /// On GFX11: emitted as v_lshrrev_b32 + v_and_or_b32 (no native instruction)
    CvtPkBf16F32 { dst: VReg, src0: VReg, src1: VReg },

    /// v_cvt_f32_u32: convert unsigned int to float
    VCvtF32U32 { dst: VReg, src: VReg },
    /// v_cvt_u32_f32: convert float to unsigned int (truncate)
    VCvtU32F32 { dst: VReg, src: VReg },
    /// v_sub_u32: unsigned integer subtraction (no carry)
    VSubU32 { dst: VReg, src0: Operand, src1: Operand },

    // ── LDS (Local Data Share) operations ──

    /// ds_store_b16: store 16-bit value to LDS
    /// LDS[vaddr + offset] = src (low 16 bits)
    DsStoreB16 { vaddr: VReg, src: VReg, offset: u16 },
    /// ds_store_b32: store 32-bit value to LDS
    DsStoreB32 { vaddr: VReg, src: VReg, offset: u16 },
    /// ds_store_b64: store 64-bit value (2 consecutive VGPRs) to LDS
    DsStoreB64 { vaddr: VReg, src: VReg, offset: u16 },
    /// ds_store_b128: store 128-bit value (4 consecutive VGPRs) to LDS
    DsStoreB128 { vaddr: VReg, src: VReg, offset: u16 },

    /// ds_load_b32: load 32-bit value from LDS
    DsLoadB32 { dst: VReg, vaddr: VReg, offset: u16 },
    /// ds_load_b64: load 64-bit value from LDS into 2 consecutive VGPRs
    DsLoadB64 { dst: VReg, vaddr: VReg, offset: u16 },
    /// ds_load_b128: load 128-bit value from LDS into 4 consecutive VGPRs
    DsLoadB128 { dst: VReg, vaddr: VReg, offset: u16 },
    /// ds_load_u16: load 16-bit unsigned, zero-extend to 32-bit
    DsLoadU16 { dst: VReg, vaddr: VReg, offset: u16 },
    /// ds_load_u16_d16: load 16-bit into low half of dst (bf16 column tearing)
    DsLoadU16D16 { dst: VReg, vaddr: VReg, offset: u16 },
    /// ds_load_u16_d16_hi: load 16-bit into high half of dst (bf16 column tearing)
    DsLoadU16D16Hi { dst: VReg, vaddr: VReg, offset: u16 },

    /// s_barrier: workgroup barrier — all waves in WG must reach before any proceed
    SBarrier,

    // ── EXEC mask (conditional execution) ──
    /// v_cmp_lt_u32 vcc, src0, src1 — set VCC bitmask where src0 < src1 (unsigned)
    /// Used for bounds checking: v_cmp_lt_u32 vcc, global_id, n_elems
    VCmpLtU32 { src0: Operand, src1: Operand },

    /// v_cmp_ge_u32 vcc, src0, src1 — set VCC where src0 >= src1 (unsigned)
    VCmpGeU32 { src0: Operand, src1: Operand },

    /// v_cmp_gt_f32 vcc, src, 0.0 — set VCC where src > 0.0 (for ReLU mask)
    VCmpGtF32Imm0 { src: VReg },

    /// v_cndmask_b32 dst, src0, src1, vcc — dst = VCC ? src1 : src0
    VCndmaskB32 { dst: VReg, src_false: Operand, src_true: Operand },

    /// s_and_saveexec_b32 dst, vcc_lo — Save current EXEC to dst, then EXEC &= VCC
    /// Lanes where VCC==0 are masked out (no loads/stores/ALU for those lanes)
    SaveExec { dst: SReg },

    /// s_mov_b32 exec_lo, src — Restore EXEC from saved SGPR
    /// Must be called after the conditional block to unmask all lanes
    RestoreExec { src: SReg },

    // ── Additional branch variants ──
    /// s_cbranch_scc0 — branch if SCC == 0
    BranchScc0(String),
    /// s_cbranch_vccz — branch if VCC == 0 (all lanes false)
    BranchVccz(String),

    // ── Additional ALU ops ──
    /// v_or_b32: bitwise OR
    VOrB32 { dst: VReg, src0: Operand, src1: Operand },
    /// v_sqrt_f32: square root
    VSqrtF32 { dst: VReg, src: VReg },
    /// v_cmp_gt_u32 vcc, src, imm — set VCC where src > imm
    VCmpGtU32Imm { src: VReg, imm: u32 },
    /// v_cmp_eq_u32 vcc, src, imm — set VCC where src == imm
    VCmpEqU32Imm { src: VReg, imm: u32 },
    /// v_cmp_ge_i32 vcc, src0, src1 — set VCC where src0 >= src1 (signed)
    VCmpGeI32 { src0: VReg, src1: VReg },
    /// v_log_f32: compute log₂(x) — NOT ln(x)! For ln(x), post-multiply by ln(2)
    VLog2F32 { dst: VReg, src: VReg },

    // ── Lane permute ──
    /// v_permlanex16_b32 vdst, vsrc, s0, s0 — swap lane L with L XOR 16
    VPermlanex16B32 { dst: VReg, src: VReg },

    // ── VOP3 three-source ──
    /// v_and_or_b32 vdst, vsrc0, literal, vsrc2 — vdst = (vsrc0 & literal) | vsrc2
    VAndOrB32 { dst: VReg, src0: VReg, literal: u32, src2: VReg },

    // ── 64-bit address arithmetic ──
    /// v_add_co_u32 vdst, vcc_lo, vsrc0, vsrc1 — add low 32 bits with carry out
    VAddCOU32 { dst: VReg, src0: VReg, src1: VReg },
    /// v_add_co_ci_u32 vdst, vcc_lo, vsrc0, 0, vcc_lo — add high 32 bits with carry in
    VAddCCU32 { dst: VReg, src: VReg },

    // ── Global atomics ──
    /// global_atomic_add_f32 (no return) — fire-and-forget atomic float add
    GlobalAtomicAddF32 { addr: VReg, src: VReg, offset: i32 },

    // ── SMEM scalar load ──
    /// s_load_dword dst, s[base_lo:base_hi], offset
    SMemLoadDword { dst: SReg, base_lo: SReg, base_hi: SReg, offset: i32 },

    // ── Wave reduction (max) ──
    /// Wave32 max reduction via ds_swizzle XOR patterns
    WaveReduceMaxF32 { val: VReg, tmp: VReg },

    // ── Raw assembly passthrough (escape hatch) ──
    RawAsm(String),
}

// Helper: extract VRegs from an Operand
fn operand_vregs(op: &Operand) -> Option<VReg> {
    match op {
        Operand::VReg(v) => Some(*v),
        _ => None,
    }
}

impl Op {
    /// Return all VRegs referenced by this instruction (both def and use).
    /// Used by liveness analysis to compute live intervals.
    pub fn vreg_refs(&self) -> Vec<VReg> {
        match self {
            // Global memory
            Op::GlobalLoad { dst, addr, width, .. } => {
                let n = width.vreg_count();
                let mut v: Vec<VReg> = (0..n).map(|i| VReg(dst.0 + i as u32)).collect();
                v.push(*addr); v.push(VReg(addr.0 + 1));
                v
            }
            Op::GlobalStore { addr, src, width, .. } => {
                let n = width.vreg_count();
                let mut v: Vec<VReg> = (0..n).map(|i| VReg(src.0 + i as u32)).collect();
                v.push(*addr); v.push(VReg(addr.0 + 1));
                v
            }

            // LDS
            Op::LdsLoad { dst, addr, width, .. } => {
                let n = width.vreg_count();
                let mut v: Vec<VReg> = (0..n).map(|i| VReg(dst.0 + i as u32)).collect();
                v.push(*addr);
                v
            }
            Op::LdsStore { addr, src, width, .. } => {
                let n = width.vreg_count();
                let mut v: Vec<VReg> = (0..n).map(|i| VReg(src.0 + i as u32)).collect();
                v.push(*addr);
                v
            }

            // Scalar memory (no VGPRs)
            Op::ScalarLoad { .. } => vec![],

            // Vector ALU (2-src)
            Op::VAddF32 { dst, src0, src1 } |
            Op::VMulF32 { dst, src0, src1 } |
            Op::VMaxF32 { dst, src0, src1 } |
            Op::VMinF32 { dst, src0, src1 } |
            Op::VAddU32 { dst, src0, src1 } |
            Op::VAndB32 { dst, src0, src1 } |
            Op::VXorB32 { dst, src0, src1 } |
            Op::VSubF32 { dst, src0, src1 } |
            Op::VSubU32 { dst, src0, src1 } => {
                let mut v = vec![*dst];
                v.extend(operand_vregs(src0));
                v.extend(operand_vregs(src1));
                v
            }

            // Vector ALU (3-src)
            Op::VFmaF32 { dst, src0, src1, src2 } => {
                let mut v = vec![*dst];
                v.extend(operand_vregs(src0));
                v.extend(operand_vregs(src1));
                v.extend(operand_vregs(src2));
                v
            }

            // Vector move
            Op::VMov { dst, src } => {
                let mut v = vec![*dst];
                v.extend(operand_vregs(src));
                v
            }
            Op::VMovFromSgpr { dst, .. } => vec![*dst],

            // Vector int ops
            Op::VMulLoU32 { dst, src0, src1 } => vec![*dst, *src0, *src1],
            Op::VLshlrevB32 { dst, src, .. } |
            Op::VLshrrevB32 { dst, src, .. } => vec![*dst, *src],

            // Readfirstlane
            Op::VReadfirstlane { src, .. } => vec![*src],

            // 64-bit addr
            Op::VAddCo { dst, src0, src1 } => vec![*dst, *src0, *src1],
            Op::VAddCoCi { dst, src } => vec![*dst, *src],

            // Scalar ALU (no VGPRs)
            Op::SAddU32 { .. } | Op::SAddcU32 { .. } | Op::SSubU32 { .. } | Op::SAndB32 { .. } |
            Op::SMulI32 { .. } | Op::SLshlB32 { .. } | Op::SLshrB32 { .. } |
            Op::SMov { .. } | Op::SCmpLtU32 { .. } |
            Op::SCmpEqU32 { .. } | Op::SCmpGeU32 { .. } => vec![],

            // WMMA: 8 consecutive VGPRs for each of dst, a, b, c
            Op::Wmma { dst, a, b, c, .. } => {
                let mut v = Vec::with_capacity(32);
                for i in 0..8u32 {
                    v.push(VReg(dst.0 + i));
                    v.push(VReg(a.0 + i));
                    v.push(VReg(b.0 + i));
                    v.push(VReg(c.0 + i));
                }
                v
            }

            // Control flow (no VGPRs)
            Op::Label(_) | Op::BranchScc1(_) | Op::Branch(_) => vec![],

            // Sync (no VGPRs)
            Op::Barrier | Op::WaitVmcnt(_) | Op::WaitLgkmcnt(_) | Op::WaitVscnt(_) | Op::ClearVcc
            | Op::SMemLoadDword { .. } => vec![],
            Op::Endpgm => vec![],

            // Hardware
            Op::CaptureTgid { .. } => vec![],
            Op::ComputeGlobalIdX { dst, .. } => vec![*dst],

            // Cross-lane
            Op::DsSwizzle { dst, src, .. } => vec![*dst, *src],

            // Special math
            Op::VRsqF32 { dst, src } |
            Op::VExpF32 { dst, src } |
            Op::VRcpF32 { dst, src } |
            Op::VCvtF32U32 { dst, src } |
            Op::VCvtU32F32 { dst, src } => vec![*dst, *src],

            // Data conversion
            Op::CvtPkBf16F32 { dst, src0, src1 } => vec![*dst, *src0, *src1],

            // LDS ops (new)
            Op::DsStoreB16 { vaddr, src, .. } |
            Op::DsStoreB32 { vaddr, src, .. } => vec![*vaddr, *src],
            Op::DsStoreB64 { vaddr, src, .. } => {
                vec![*vaddr, *src, VReg(src.0 + 1)]
            }
            Op::DsStoreB128 { vaddr, src, .. } => {
                vec![*vaddr, *src, VReg(src.0 + 1), VReg(src.0 + 2), VReg(src.0 + 3)]
            }
            Op::DsLoadB32 { dst, vaddr, .. } |
            Op::DsLoadU16 { dst, vaddr, .. } |
            Op::DsLoadU16D16 { dst, vaddr, .. } |
            Op::DsLoadU16D16Hi { dst, vaddr, .. } => vec![*dst, *vaddr],
            Op::DsLoadB64 { dst, vaddr, .. } => vec![*dst, VReg(dst.0 + 1), *vaddr],
            Op::DsLoadB128 { dst, vaddr, .. } => {
                vec![*dst, VReg(dst.0 + 1), VReg(dst.0 + 2), VReg(dst.0 + 3), *vaddr]
            }

            Op::SBarrier => vec![],

            // Comparisons
            Op::VCmpLtU32 { src0, src1 } |
            Op::VCmpGeU32 { src0, src1 } => {
                let mut v = vec![];
                v.extend(operand_vregs(src0));
                v.extend(operand_vregs(src1));
                v
            }
            Op::VCmpGtF32Imm0 { src } => vec![*src],
            Op::VCndmaskB32 { dst, src_false, src_true } => {
                let mut v = vec![*dst];
                v.extend(operand_vregs(src_false));
                v.extend(operand_vregs(src_true));
                v
            }

            // EXEC mask (no VGPRs)
            Op::SaveExec { .. } | Op::RestoreExec { .. } => vec![],

            // Additional branch variants (no VGPRs)
            Op::BranchScc0(_) | Op::BranchVccz(_) => vec![],

            // Additional ALU
            Op::VOrB32 { dst, src0, src1 } => {
                let mut v = vec![*dst];
                v.extend(operand_vregs(src0));
                v.extend(operand_vregs(src1));
                v
            }
            Op::VSqrtF32 { dst, src } => vec![*dst, *src],
            Op::VLog2F32 { dst, src } => vec![*dst, *src],
            Op::VCmpGtU32Imm { src, .. } | Op::VCmpEqU32Imm { src, .. } => vec![*src],
            Op::VCmpGeI32 { src0, src1 } => vec![*src0, *src1],

            // Lane permute
            Op::VPermlanex16B32 { dst, src } => vec![*dst, *src],
            // VOP3 three-source
            Op::VAndOrB32 { dst, src0, src2, .. } => vec![*dst, *src0, *src2],
            // 64-bit add
            Op::VAddCOU32 { dst, src0, src1 } => vec![*dst, *src0, *src1],
            Op::VAddCCU32 { dst, src } => vec![*dst, *src],

            // Global atomics
            Op::GlobalAtomicAddF32 { addr, src, .. } => {
                vec![*addr, VReg(addr.0 + 1), *src]
            }

            // Wave reduce
            Op::WaveReduceAddF32 { val, tmp } => vec![*val, *tmp],
            Op::WaveReduceMaxF32 { val, tmp } => vec![*val, *tmp],

            // Raw asm (unknown, assume none)
            Op::RawAsm(_) => vec![],
        }
    }
}

// ============================================================================
// Kernel argument metadata
// ============================================================================

/// Kernel argument type.
#[derive(Clone, Debug)]
pub enum ArgKind {
    /// 64-bit pointer (2 SGPRs)
    Ptr,
    /// 32-bit unsigned integer (1 SGPR)
    U32,
    /// 32-bit float (1 SGPR)
    F32,
}

/// Kernel argument descriptor.
#[derive(Clone, Debug)]
pub struct KernArg {
    pub name: String,
    pub kind: ArgKind,
    pub offset: u32, // byte offset in kernarg segment
    pub sreg: SReg,  // first SGPR allocated to this arg
}

// ============================================================================
// Register allocation hints
// ============================================================================

/// Allocation request for virtual registers, with optional constraints.
#[derive(Clone, Debug)]
pub struct VRegAlloc {
    pub vreg: VReg,
    pub count: u32,        // number of consecutive registers (1, 2, 4, 8)
    pub alignment: Alignment,
}

#[derive(Clone, Debug)]
pub struct SRegAlloc {
    pub sreg: SReg,
    pub count: u32,
    pub alignment: Alignment,
}
