//! RDNA3 / GFX1100 Instruction Latency Model
//!
//! Classifies T0 IR ops into hardware pipeline categories and provides
//! empirically-calibrated latency estimates for instruction scheduling.
//!
//! # Measurement methodology
//!
//! All latencies are expressed in **VALU-normalized cycles** — i.e. how many
//! simple VALU operations (v_add_f32) worth of time the instruction takes.
//! These were measured on GFX1100 (Navi 31, Wave32) using `hw_probe`
//! serial-chain microbenchmarks with `s_getreg_b32 HW_REG_SHADER_CYCLES`.
//!
//! # Pipeline overlap (empirical, hw_probe overlap probes 2026-03-24)
//!
//! Single-wave instruction issue is SERIAL across all pipelines:
//!   VALU+VMEM: 0% overlap, VALU+LDS: 0%, VALU+TRANS: 6% (~serial)
//! Multi-wave latency hiding is handled at the cost_model level.
//!
//! # GFX1100 Instruction Latencies (probe-calibrated 2026-03-24)
//!
//! | Pipeline     | Examples                           | Shader cy | VALU-norm |
//! |--------------|------------------------------------|-----------|-----------|
//! | VALU-simple  | add, sub, min, max, and, or, shift |     ~10   |     1     |
//! | VALU-complex | mul, fma, mul_lo_u32, and_or       |     ~19   |     2     |
//! | TRANS        | rcp, rsq, exp, log, sqrt           |     ~11   |     1     |
//! | SALU         | s_add, s_mul, s_lshl               |     ~10   |     1     |
//! | CVT          | cvt_f32_u32, cvt_u32_f32           |     ~20   |     2     |
//! | CVT pack     | v_cvt_pk_bf16_f32                  |     ~40   |     4     |
//! | LDS load     | ds_load_b32/b64/b128               |     ~76   |     7     |
//! | LDS store    | ds_store_b16/b32/b64/b128          |     ~38   |     4     |
//! | ds_swizzle   | lane permute (XOR, no mem)         |     ~33   |     3     |
//! | v_permlane   | v_permlane_x16                     |     ~27   |     3     |
//! | VMEM load    | global_load b32/b64/b128           |    ~500   |    47     |
//! | VMEM store   | global_store b32                   |    ~249   |    24     |
//! | WMMA         | v_wmma_f32_16x16x16_bf16           |     ~36   |     4     |
//! | Wave reduce  | 5×swizzle + 5×add composite        |    ~254   |    24     |
//! | CTRL         | branch, waitcnt, barrier           |       0   |     0     |

use super::ir::*;

// ============================================================================
// Pipeline classification
// ============================================================================

/// Hardware pipeline that executes an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pipeline {
    VALU,       // Vector ALU: add, mul, fma, logic, mov
    TRANS,      // Transcendental: rcp, rsq, exp, log, sqrt, sin, cos
    SALU,       // Scalar ALU: s_add, s_lshl, s_and, s_cmp, s_mov
    VMEM,       // Global memory: global_load, global_store
    LDS,        // Local data share: ds_load, ds_store, ds_swizzle
    WMMA,       // Wave matrix multiply accumulate
    CTRL,       // Control flow: branch, waitcnt, barrier, endpgm, label
}

/// Latency info for a single instruction.
#[derive(Clone, Copy, Debug)]
pub struct LatencyInfo {
    /// Pipeline that executes this instruction.
    pub pipeline: Pipeline,
    /// Use-to-use latency in VALU-normalized cycles.
    /// 1 = same as v_add_f32, 27 = needs 27 simple VALU to hide.
    pub latency: u32,
    /// Reciprocal throughput in VALU-normalized cycles.
    /// 1 = can issue every VALU cycle, 4 = every 4 VALU cycles.
    pub throughput: u32,
    /// Does this instruction access memory (affects waitcnt tracking)?
    pub is_mem: bool,
    /// Counter category for waitcnt (None if not a memory op).
    pub wait_counter: Option<WaitCounter>,
}

/// Wait counter categories for s_waitcnt / s_waitcnt_vscnt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitCounter {
    VMcnt,      // global_load → s_waitcnt vmcnt(N)
    VScnt,      // global_store → s_waitcnt_vscnt null, N
    LGKMcnt,    // LDS / SMEM → s_waitcnt lgkmcnt(N)
}

// ============================================================================
// Latency helpers (private constructors)
// ============================================================================
// All values are VALU-normalized (1 = one simple VALU cycle = ~10 shader cycles)
// Source: hw_probe GPU sweep on GFX1100, 2026-03-24
// Baseline: v_add_f32 lat/op = 10.5 shader cycles

/// Simple VALU: v_add_f32, v_sub, v_min, v_max, v_and, v_or, shifts, mov
/// Measured: ~10 shader cycles = 1 VALU-norm
const fn valu_simple() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::VALU, latency: 1, throughput: 1, is_mem: false, wait_counter: None }
}
/// Complex VALU: v_mul_f32, v_fma_f32, v_mul_lo_u32, v_and_or_b32
/// Measured: ~19 shader cycles = 2 VALU-norm
const fn valu_complex() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::VALU, latency: 2, throughput: 2, is_mem: false, wait_counter: None }
}
/// Transcendental: v_rcp, v_rsq, v_exp, v_log, v_sqrt
/// Measured: ~11 shader cycles ≈ 1 VALU-norm
const fn trans() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::TRANS, latency: 1, throughput: 1, is_mem: false, wait_counter: None }
}
/// SALU: s_add_u32, s_mul_i32, s_lshl, s_and, s_cmp
/// Measured: ~10 shader cycles ≈ 1 VALU-norm
const fn salu() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::SALU, latency: 1, throughput: 1, is_mem: false, wait_counter: None }
}
/// VMEM load: global_load_b32/b64/b128
/// Measured: ~500 shader cycles ≈ 47 VALU-norm  [was 27, corrected 2026-03-24]
const fn vmem_load() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::VMEM, latency: 47, throughput: 1, is_mem: true, wait_counter: Some(WaitCounter::VMcnt) }
}
/// VMEM store: global_store_b32
/// Measured: ~249 shader cycles ≈ 24 VALU-norm  [was 15, corrected 2026-03-24]
const fn vmem_store() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::VMEM, latency: 24, throughput: 1, is_mem: true, wait_counter: Some(WaitCounter::VScnt) }
}
/// LDS load: ds_load_b32/b64/b128 (all widths: same latency)
/// Measured: ~76 shader cycles ≈ 7 VALU-norm
const fn lds_load() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::LDS, latency: 7, throughput: 1, is_mem: true, wait_counter: Some(WaitCounter::LGKMcnt) }
}
/// LDS store: ds_store_b16/b32/b64/b128
/// Measured: ~38 shader cycles ≈ 4 VALU-norm
const fn lds_store() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::LDS, latency: 4, throughput: 1, is_mem: true, wait_counter: Some(WaitCounter::LGKMcnt) }
}
/// CVT pack: v_cvt_pk_bf16_f32
/// Measured: ~40 shader cycles ≈ 4 VALU-norm
const fn cvt_pack() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::VALU, latency: 4, throughput: 4, is_mem: false, wait_counter: None }
}
const fn ctrl() -> LatencyInfo {
    LatencyInfo { pipeline: Pipeline::CTRL, latency: 0, throughput: 1, is_mem: false, wait_counter: None }
}

// ============================================================================
// Latency query
// ============================================================================

/// Get latency info for a T0 IR instruction.
pub fn op_latency(op: &Op) -> LatencyInfo {
    match op {
        // ── VALU simple: 1-cycle (v_add, v_sub, v_min, v_max, logic, shifts, mov) ──
        Op::VAddF32 { .. } | Op::VSubF32 { .. } |
        Op::VMaxF32 { .. } | Op::VMinF32 { .. } | Op::VMinU32 { .. } |
        Op::VMov { .. } | Op::VMovFromSgpr { .. } |
        Op::VAddU32 { .. } | Op::VSubU32 { .. } |
        Op::VLshlrevB32 { .. } | Op::VLshrrevB32 { .. } |
        Op::VAndB32 { .. } | Op::VOrB32 { .. } | Op::VXorB32 { .. } |
        Op::VCndmaskB32 { .. } |
        Op::VAddCo { .. } | Op::VAddCoCi { .. } |
        Op::VAddCOU32 { .. } | Op::VAddCCU32 { .. } |
        Op::VCmpLtU32 { .. } | Op::VCmpGeU32 { .. } |
        Op::VCmpGtF32Imm0 { .. } | Op::VCmpGtU32Imm { .. } |
        Op::VCmpEqU32Imm { .. } | Op::VCmpGeI32 { .. } |
        Op::VReadfirstlane { .. } => valu_simple(),

        // ── VALU complex: 2-cycle (mul, fma, integer mul, cvt, VOP3) ──
        // Measured: ~36 shader cycles = 2× simple VALU
        Op::VMulF32 { .. } | Op::VFmaF32 { .. } |
        Op::VMulLoU32 { .. } |
        Op::VCvtF32U32 { .. } | Op::VCvtU32F32 { .. } |
        Op::VAndOrB32 { .. } => valu_complex(),

        // ── CVT pack: 4-cycle (bf16 pack) ──
        Op::CvtPkBf16F32 { .. } => cvt_pack(),

        // ── TRANS: 1-cycle (same tier as simple VALU on GFX1100!) ──
        Op::VRcpF32 { .. } | Op::VRsqF32 { .. } |
        Op::VExpF32 { .. } | Op::VLog2F32 { .. } |
        Op::VSinF32 { .. } | Op::VCosF32 { .. } |
        Op::VSqrtF32 { .. } => trans(),

        // ── SALU: 1-cycle scalar ALU ──
        Op::SAddU32 { .. } | Op::SSubU32 { .. } | Op::SAddcU32 { .. } |
        Op::SMulI32 { .. } |
        Op::SLshlB32 { .. } | Op::SLshrB32 { .. } |
        Op::SAndB32 { .. } | Op::SMov { .. } |
        Op::SCmpLtU32 { .. } | Op::SCmpGeU32 { .. } | Op::SCmpEqU32 { .. } |
        Op::SaveExec { .. } | Op::RestoreExec { .. } | Op::XorExec { .. } |
        Op::ClearVcc | Op::SMovToVcc { .. } |
        Op::ReadShaderCycles { .. } => salu(),

        // ── VMEM: global memory (load ≈ 27, store ≈ 15 VALU-norm) ──
        Op::GlobalLoad { .. } | Op::BufferLoad { .. } => vmem_load(),
        Op::GlobalStore { .. } | Op::BufferStore { .. } => vmem_store(),
        Op::GlobalAtomicAddF32 { .. } => vmem_load(), // atomic returns via vmcnt
        Op::GlobalAtomicAddU32Rtn { .. } => vmem_load(), // atomic u32 returns via vmcnt

        // ── LDS load: 7 VALU-norm (width-independent: b32=b64=b128) ──
        Op::LdsLoad { .. } |
        Op::DsLoadB32 { .. } | Op::DsLoadB64 { .. } | Op::DsLoadB128 { .. } |
        Op::DsLoadU16 { .. } | Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. } => lds_load(),

        // ── LDS store: 4 VALU-norm ──
        Op::LdsStore { .. } |
        Op::DsStoreB16 { .. } | Op::DsStoreB32 { .. } |
        Op::DsStoreB64 { .. } | Op::DsStoreB128 { .. } => lds_store(),

        // ── ds_swizzle: lane permute, no memory access, 3 VALU-norm ──
        // Measured: ~33 shader cycles ≈ 3.2 VALU-norm, rounded to 3
        Op::DsSwizzle { .. } => LatencyInfo {
            pipeline: Pipeline::LDS, latency: 3, throughput: 2, is_mem: false, wait_counter: None,
        },

        // ── SMEM: scalar memory (same counter as LDS) ──
        Op::ScalarLoad { .. } | Op::SMemLoadDword { .. } => LatencyInfo {
            pipeline: Pipeline::LDS, latency: 7, throughput: 1, is_mem: true,
            wait_counter: Some(WaitCounter::LGKMcnt),
        },

        // ── WMMA: 4 VALU-norm (measured ~36 shader cycles ≈ 3.4×, rounded to 4) ──
        Op::Wmma { .. } => LatencyInfo {
            pipeline: Pipeline::WMMA, latency: 4, throughput: 4, is_mem: false, wait_counter: None,
        },

        // ── CTRL: control flow (0 cycle) ──
        Op::Branch(_) | Op::BranchScc0(_) | Op::BranchScc1(_) | Op::BranchVccz(_) |
        Op::Label(_) |
        Op::WaitVmcnt(_) | Op::WaitLgkmcnt(_) | Op::WaitVscnt(_) |
        Op::Barrier | Op::SBarrier | Op::Endpgm |
        Op::CaptureTgid { .. } | Op::ComputeGlobalIdX { .. } => ctrl(),

        // ── Lane permute: 3 VALU-norm (measured ~27 shader cycles) ──
        Op::VPermlanex16B32 { .. } => LatencyInfo {
            pipeline: Pipeline::VALU, latency: 3, throughput: 2, is_mem: false, wait_counter: None,
        },

        // ── Wave reductions: composite (5×swizzle + 5×add ≈ 24 VALU-norm) ──
        // Measured: ~254 shader cycles ≈ 24.2 VALU-norm
        Op::WaveReduceAddF32 { .. } | Op::WaveReduceMaxF32 { .. } => LatencyInfo {
            pipeline: Pipeline::LDS, latency: 24, throughput: 24, is_mem: false, wait_counter: None,
        },

        // ── Raw asm: unknown, assume simple VALU ──
        Op::RawAsm(_) => valu_simple(),
    }
}

// ============================================================================
// Scheduling helpers
// ============================================================================

/// Check if an instruction can execute in parallel with VMEM loads.
/// Returns true for VALU/SALU/TRANS ops that don't access memory.
pub fn can_overlap_vmem(op: &Op) -> bool {
    let info = op_latency(op);
    matches!(info.pipeline, Pipeline::VALU | Pipeline::SALU | Pipeline::TRANS)
        && !info.is_mem
}

/// Check if an instruction can execute in parallel with LDS ops.
pub fn can_overlap_lds(op: &Op) -> bool {
    let info = op_latency(op);
    matches!(info.pipeline, Pipeline::VALU | Pipeline::SALU | Pipeline::TRANS | Pipeline::VMEM)
        && !info.is_mem
}

/// Estimate total execution cycles for a sequence of ops.
///
/// # Pipeline model (probe-calibrated 2026-03-24)
///
/// Overlap probes confirmed: single-wave instruction issue is SERIAL across
/// all pipelines (VALU+VMEM: 0%, VALU+LDS: 0%, VALU+TRANS: 6%).
/// Therefore we use an ADDITIVE model: total = compute + memory.
///
/// Multi-wave latency hiding (where later waves execute compute while earlier
/// waves wait for memory) is modeled at the cost_model level via occupancy.
///
/// Returns (total_cycles, pipeline_breakdown).
/// All cycle counts are in VALU-normalized units.
pub fn estimate_cycles(ops: &[Op]) -> (u32, PipelineBreakdown) {
    let mut breakdown = PipelineBreakdown::default();

    for op in ops {
        let info = op_latency(op);
        match info.pipeline {
            Pipeline::VALU => breakdown.valu_cycles += info.throughput,
            Pipeline::TRANS => breakdown.trans_cycles += info.throughput,
            Pipeline::SALU => breakdown.salu_cycles += info.throughput,
            Pipeline::VMEM => breakdown.vmem_count += 1,
            Pipeline::LDS => breakdown.lds_count += 1,
            Pipeline::WMMA => breakdown.wmma_cycles += info.throughput,
            Pipeline::CTRL => breakdown.ctrl_count += 1,
        }
    }

    // Compute cycles: these pipelines share the wave's issue slot, so they
    // serialize within a single wave. Use sum, not max.
    let compute_cycles = breakdown.valu_cycles
        + breakdown.trans_cycles
        + breakdown.salu_cycles
        + breakdown.wmma_cycles;

    // Memory latency (VALU-normalized)
    // VMEM: probe measured ~500 shader cycles = 47 VALU-norm per load
    // LDS:  probe measured ~76 shader cycles  = 7 VALU-norm per load
    let vmem_latency = breakdown.vmem_count * 47;
    let lds_latency = breakdown.lds_count * 7;
    let mem_latency = vmem_latency + lds_latency;

    // Total = compute + memory (serial within single wave)
    let total = compute_cycles + mem_latency;

    (total, breakdown)
}

/// Breakdown of cycles by pipeline.
#[derive(Clone, Debug, Default)]
pub struct PipelineBreakdown {
    pub valu_cycles: u32,
    pub trans_cycles: u32,
    pub salu_cycles: u32,
    pub wmma_cycles: u32,
    pub vmem_count: u32,
    pub lds_count: u32,
    pub ctrl_count: u32,
}

impl std::fmt::Display for PipelineBreakdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VALU:{} TRANS:{} SALU:{} WMMA:{} VMEM:×{} LDS:×{} CTRL:×{}",
            self.valu_cycles, self.trans_cycles, self.salu_cycles,
            self.wmma_cycles, self.vmem_count, self.lds_count, self.ctrl_count)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Simple VALU: latency=1, throughput=1 ──
    #[test]
    fn test_valu_simple_latency() {
        let op = Op::VAddF32 {
            dst: VReg(1),
            src0: Operand::VReg(VReg(2)),
            src1: Operand::VReg(VReg(3)),
        };
        let info = op_latency(&op);
        assert_eq!(info.pipeline, Pipeline::VALU);
        assert_eq!(info.latency, 1);
        assert_eq!(info.throughput, 1);
        assert!(!info.is_mem);
    }

    // ── Complex VALU: latency=2 (mul, fma are 2× simple) ──
    #[test]
    fn test_valu_complex_latency() {
        let mul = Op::VMulF32 { dst: VReg(1), src0: Operand::VReg(VReg(2)), src1: Operand::VReg(VReg(3)) };
        let fma = Op::VFmaF32 {
            dst: VReg(1),
            src0: Operand::VReg(VReg(2)),
            src1: Operand::VReg(VReg(3)),
            src2: Operand::VReg(VReg(4)),
        };
        assert_eq!(op_latency(&mul).latency, 2);
        assert_eq!(op_latency(&mul).throughput, 2);
        assert_eq!(op_latency(&fma).latency, 2);
        // mul/fma latency should be strictly greater than add
        let add = Op::VAddF32 { dst: VReg(1), src0: Operand::VReg(VReg(2)), src1: Operand::VReg(VReg(3)) };
        assert!(op_latency(&mul).latency > op_latency(&add).latency,
            "v_mul_f32 should have higher latency than v_add_f32");
    }

    // ── TRANS: same tier as simple VALU on GFX1100 ──
    #[test]
    fn test_trans_latency() {
        let op = Op::VRcpF32 { dst: VReg(1), src: VReg(2) };
        let info = op_latency(&op);
        assert_eq!(info.pipeline, Pipeline::TRANS);
        assert_eq!(info.latency, 1);
        assert_eq!(info.throughput, 1);
    }

    // ── VMEM: load=47, store=24, load > store ──
    #[test]
    fn test_vmem_latency() {
        let load = Op::GlobalLoad { dst: VReg(1), addr: VReg(2), width: Width::B32, offset: 0 };
        let store = Op::GlobalStore { addr: VReg(2), src: VReg(1), width: Width::B32, offset: 0 };
        let load_info = op_latency(&load);
        let store_info = op_latency(&store);
        assert_eq!(load_info.pipeline, Pipeline::VMEM);
        assert_eq!(load_info.latency, 47);
        assert_eq!(store_info.latency, 24);
        assert!(load_info.latency > store_info.latency,
            "VMEM load should have higher latency than store");
        assert!(load_info.is_mem);
        assert_eq!(load_info.wait_counter, Some(WaitCounter::VMcnt));
        assert_eq!(store_info.wait_counter, Some(WaitCounter::VScnt));
    }

    // ── WMMA: 4 VALU-norm ──
    #[test]
    fn test_wmma_latency() {
        let op = Op::Wmma {
            dst: VReg(0), a: VReg(8), b: VReg(16), c: VReg(24),
            format: WmmaFormat::BF16_F32,
        };
        let info = op_latency(&op);
        assert_eq!(info.pipeline, Pipeline::WMMA);
        assert_eq!(info.latency, 4);
        assert_eq!(info.throughput, 4);
    }

    // ── LDS: load=7, store=4, load > store (asymmetric) ──
    #[test]
    fn test_lds_load_store_asymmetric() {
        let load = Op::DsLoadB32 { dst: VReg(1), vaddr: VReg(0), offset: 0 };
        let store = Op::DsStoreB32 { vaddr: VReg(0), src: VReg(1), offset: 0 };
        let load_info = op_latency(&load);
        let store_info = op_latency(&store);
        assert_eq!(load_info.pipeline, Pipeline::LDS);
        assert_eq!(load_info.latency, 7);
        assert_eq!(store_info.latency, 4);
        assert!(load_info.latency > store_info.latency,
            "LDS load should have higher latency than store");
        assert!(load_info.is_mem);
        assert_eq!(load_info.wait_counter, Some(WaitCounter::LGKMcnt));
    }

    // ── CVT pack: bf16 packing is 4× simple VALU ──
    #[test]
    fn test_cvt_pack_latency() {
        let op = Op::CvtPkBf16F32 { dst: VReg(1), src0: VReg(2), src1: VReg(3) };
        let info = op_latency(&op);
        assert_eq!(info.latency, 4);
        assert_eq!(info.throughput, 4);
    }

    #[test]
    fn test_estimate_cycles() {
        let ops = vec![
            Op::GlobalLoad { dst: VReg(1), addr: VReg(2), width: Width::B32, offset: 0 },
            Op::WaitVmcnt(0),
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::InlineFloat(1.0) },
            Op::VMulF32 { dst: VReg(4), src0: Operand::VReg(VReg(3)), src1: Operand::InlineFloat(2.0) },
            Op::GlobalStore { addr: VReg(2), src: VReg(4), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let (total, breakdown) = estimate_cycles(&ops);
        assert_eq!(breakdown.vmem_count, 2); // 1 load + 1 store
        assert_eq!(breakdown.valu_cycles, 3); // add(1) + mul(2)
        assert!(total > 0);
        // Additive model: 3 VALU + 2 VMEM = 3 + (47+24) = 74 + ctrl
        // VMEM: load(47) + store(24) = 71, total = 71 + 3 = 74
        assert!(total >= 70, "VMEM should dominate: {}", total);
        eprintln!("Estimate: {} cycles — {}", total, breakdown);
    }

    #[test]
    fn test_can_overlap() {
        let valu = Op::VAddF32 { dst: VReg(1), src0: Operand::VReg(VReg(2)), src1: Operand::InlineFloat(1.0) };
        let load = Op::GlobalLoad { dst: VReg(1), addr: VReg(2), width: Width::B32, offset: 0 };

        assert!(can_overlap_vmem(&valu));
        assert!(!can_overlap_vmem(&load));
    }
}
