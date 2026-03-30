//! GFX1100 Instruction Latency Model
//!
//! Provides per-instruction latency and throughput data for the RDNA3
//! GFX1100 (Navi 31, RX 7900 XTX) GPU. This is the foundation for:
//!
//! 1. **DAG-based instruction scheduling** — critical path analysis
//! 2. **Cost model refinement** — accurate K-loop iteration estimation
//! 3. **Software pipelining** — knowing how many VALU can overlap with VMEM
//!
//! # Data Sources
//! - AMD RDNA3 ISA Reference (public)
//! - hw_probe microbenchmarks on actual GFX1100 hardware
//! - rocBLAS Tensile scheduling heuristics (reverse-engineered)
//!
//! # Units
//! All latencies are in **shader clock cycles** (not VALU-normalized).
//! GFX1100 shader clock ≈ 2.5 GHz.  
//! VALU-normalized factor: 1 VALU-norm ≈ 18.3 shader cycles (Wave32).

use super::ir::Op;

/// Instruction functional unit classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InsnClass {
    /// Vector ALU (simple): add, mul, fma, mov, shifts, cmp, cndmask
    VALU,
    /// Vector ALU (transcendental): exp, log, sin, cos, rcp, rsq, sqrt
    /// These use the "trans" execution unit (quarter-rate on RDNA3).
    VTRANS,
    /// Vector memory: global_load, global_store
    VMEM,
    /// LDS (Local Data Share): ds_load, ds_store, ds_swizzle
    LDS,
    /// Scalar ALU: s_add, s_mul, s_cmp, s_mov, s_and, s_lshl
    SALU,
    /// Scalar memory: s_load (kernarg loads)
    SMEM,
    /// WMMA (Wave Matrix Multiply-Accumulate): v_wmma_f32_16x16x16_bf16
    WMMA,
    /// Control flow: branch, barrier, waitcnt, endpgm
    CTRL,
    /// Data conversion: cvt_pk_bf16_f32, v_cvt_f32_u32
    CVT,
    /// Cross-lane (non-LDS): ds_swizzle (register-level), readfirstlane, permlane
    XLANE,
}

/// Latency and throughput information for a single instruction.
#[derive(Clone, Copy, Debug)]
pub struct InsnLatency {
    /// Functional unit classification
    pub class: InsnClass,
    /// Issue latency: cycles to dispatch this instruction (occupies the issue port)
    /// Almost all instructions issue in 1 cycle.
    pub issue: u32,
    /// Result latency: cycles until the result register is readable.
    /// For VMEM this is very high (~490sc); for VALU it's typically 1.
    pub result: u32,
    /// Reciprocal throughput: minimum cycles between consecutive instructions
    /// of the same class. 1.0 = one per cycle, 4.0 = one per 4 cycles.
    pub recip_throughput: f32,
}

impl InsnLatency {
    const fn valu() -> Self {
        InsnLatency { class: InsnClass::VALU, issue: 1, result: 1, recip_throughput: 1.0 }
    }
    const fn vtrans(result: u32) -> Self {
        InsnLatency { class: InsnClass::VTRANS, issue: 1, result, recip_throughput: 4.0 }
    }
    const fn vmem_load() -> Self {
        InsnLatency { class: InsnClass::VMEM, issue: 1, result: 490, recip_throughput: 1.0 }
    }
    const fn vmem_store() -> Self {
        // Stores are fire-and-forget: result latency = 0 (no readback needed)
        InsnLatency { class: InsnClass::VMEM, issue: 1, result: 0, recip_throughput: 1.0 }
    }
    const fn lds() -> Self {
        InsnLatency { class: InsnClass::LDS, issue: 1, result: 20, recip_throughput: 1.0 }
    }
    const fn salu() -> Self {
        InsnLatency { class: InsnClass::SALU, issue: 1, result: 1, recip_throughput: 1.0 }
    }
    const fn smem() -> Self {
        InsnLatency { class: InsnClass::SMEM, issue: 1, result: 25, recip_throughput: 1.0 }
    }
    const fn wmma() -> Self {
        // WMMA: ~36 shader cycles result latency, ~4 VALU-normalized.
        // Throughput: 1 per cycle issue, but result takes 36 cycles.
        InsnLatency { class: InsnClass::WMMA, issue: 1, result: 36, recip_throughput: 1.0 }
    }
    const fn ctrl() -> Self {
        InsnLatency { class: InsnClass::CTRL, issue: 1, result: 0, recip_throughput: 1.0 }
    }
    const fn xlane() -> Self {
        InsnLatency { class: InsnClass::XLANE, issue: 1, result: 2, recip_throughput: 1.0 }
    }
}

/// Get the latency model for an instruction.
///
/// # RDNA3 GFX1100 Latency Notes
///
/// The RDNA3 architecture has several execution pipelines:
/// - **VALU pipe**: simple ALU (1 cycle latency, 1 per cycle)
/// - **Trans pipe**: transcendental (v_exp, v_log, v_sin, v_cos, v_rcp, v_rsq, v_sqrt)
///   - Quarter-rate on RDNA3: 4 cycles per op (shared with VALU pipe)
///   - Result latency ~16 shader cycles (due to pipelining)
/// - **VMEM pipe**: global memory (high latency ~490sc, high bandwidth)
/// - **LDS pipe**: local data share (~20sc latency)
/// - **SALU pipe**: scalar ALU (1 cycle, runs in parallel with VALU)
/// - **SMEM pipe**: scalar memory (kernarg loads, ~25sc)
/// - **WMMA pipe**: matrix multiply (36sc result latency)
pub fn op_latency(op: &Op) -> InsnLatency {
    match op {
        // ── VALU: simple arithmetic (1 cycle result latency) ──
        Op::VAddF32 { .. } | Op::VMulF32 { .. } | Op::VSubF32 { .. } |
        Op::VMaxF32 { .. } | Op::VMinF32 { .. } |
        Op::VAddU32 { .. } | Op::VSubU32 { .. } | Op::VMinU32 { .. } |
        Op::VAndB32 { .. } | Op::VXorB32 { .. } | Op::VOrB32 { .. } |
        Op::VFmaF32 { .. } |
        Op::VMov { .. } | Op::VMovFromSgpr { .. } |
        Op::VMulLoU32 { .. } |
        Op::VLshlrevB32 { .. } | Op::VLshrrevB32 { .. } |
        Op::VAddCo { .. } | Op::VAddCoCi { .. } |
        Op::VAddCOU32 { .. } | Op::VAddCCU32 { .. } |
        Op::VCndmaskB32 { .. } |
        Op::CvtPkBf16F32 { .. } |
        Op::VCvtF32U32 { .. } | Op::VCvtU32F32 { .. } |
        Op::VAndOrB32 { .. } => InsnLatency::valu(),

        // ── VALU: transcendental (quarter-rate, ~16 sc result latency) ──
        // These share the VALU pipe but stall it for 4 cycles.
        Op::VExpF32  { .. } => InsnLatency::vtrans(16),
        Op::VRcpF32  { .. } => InsnLatency::vtrans(16),
        Op::VRsqF32  { .. } => InsnLatency::vtrans(16),
        Op::VSqrtF32 { .. } => InsnLatency::vtrans(24), // sqrt is longer
        Op::VLog2F32 { .. } => InsnLatency::vtrans(16),
        Op::VSinF32  { .. } => InsnLatency::vtrans(16),
        Op::VCosF32  { .. } => InsnLatency::vtrans(16),

        // ── VMEM: global memory ──
        Op::GlobalLoad { .. } | Op::BufferLoad { .. } => InsnLatency::vmem_load(),
        Op::GlobalStore { .. } | Op::BufferStore { .. } => InsnLatency::vmem_store(),
        Op::GlobalAtomicAddF32 { .. } |
        Op::GlobalAtomicAddU32Rtn { .. } => InsnLatency {
            class: InsnClass::VMEM, issue: 1, result: 490, recip_throughput: 1.0,
        },

        // ── LDS: local data share ──
        Op::LdsLoad { .. } | Op::LdsStore { .. } |
        Op::DsLoadB32 { .. } | Op::DsLoadU16 { .. } |
        Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. } |
        Op::DsLoadB64 { .. } | Op::DsLoadB128 { .. } |
        Op::DsStoreB16 { .. } | Op::DsStoreB32 { .. } |
        Op::DsStoreB64 { .. } | Op::DsStoreB128 { .. } => InsnLatency::lds(),

        // ── Cross-lane: ds_swizzle, readfirstlane, permlane ──
        Op::DsSwizzle { .. } => InsnLatency {
            class: InsnClass::XLANE, issue: 1, result: 4, recip_throughput: 1.0,
        },
        Op::VReadfirstlane { .. } => InsnLatency::xlane(),
        Op::VPermlanex16B32 { .. } => InsnLatency::xlane(),

        // ── SALU: scalar ALU ──
        Op::SAddU32 { .. } | Op::SSubU32 { .. } | Op::SAddcU32 { .. } |
        Op::SAndB32 { .. } | Op::SMulI32 { .. } |
        Op::SLshlB32 { .. } | Op::SLshrB32 { .. } |
        Op::SMov { .. } |
        Op::SCmpLtU32 { .. } | Op::SCmpEqU32 { .. } | Op::SCmpGeU32 { .. } |
        Op::SMovToVcc { .. } |
        Op::CaptureTgid { .. } | Op::ComputeGlobalIdX { .. } => InsnLatency::salu(),

        // ── SMEM: scalar memory (kernarg loads) ──
        Op::ScalarLoad { .. } | Op::SMemLoadDword { .. } => InsnLatency::smem(),

        // ── WMMA: matrix multiply-accumulate ──
        Op::Wmma { .. } => InsnLatency::wmma(),

        // ── Comparisons: VOPC (1 cycle, sets VCC) ──
        Op::VCmpLtU32 { .. } | Op::VCmpGeU32 { .. } |
        Op::VCmpGtF32Imm0 { .. } |
        Op::VCmpGtU32Imm { .. } | Op::VCmpEqU32Imm { .. } |
        Op::VCmpGeI32 { .. } => InsnLatency::valu(),

        // ── Control flow ──
        Op::Label(_) | Op::Branch(_) | Op::BranchScc1(_) |
        Op::BranchScc0(_) | Op::BranchVccz(_) |
        Op::Endpgm => InsnLatency::ctrl(),

        // ── Synchronization ──
        Op::Barrier | Op::SBarrier => InsnLatency {
            class: InsnClass::CTRL, issue: 1, result: 0, recip_throughput: 1.0,
        },
        Op::WaitVmcnt(_) | Op::WaitLgkmcnt(_) | Op::WaitVscnt(_) => InsnLatency {
            class: InsnClass::CTRL, issue: 1, result: 0, recip_throughput: 1.0,
        },
        Op::ClearVcc => InsnLatency::salu(),

        // ── EXEC mask ops (scalar) ──
        Op::SaveExec { .. } | Op::RestoreExec { .. } | Op::XorExec { .. } => InsnLatency::salu(),

        // ── Wave-level reduce (composite: 5× ds_swizzle + 5× v_add/max) ──
        // Modeled as LDS since ds_swizzle dominates latency.
        Op::WaveReduceAddF32 { .. } | Op::WaveReduceMaxF32 { .. } => InsnLatency {
            class: InsnClass::XLANE, issue: 10, result: 20, recip_throughput: 10.0,
        },

        // ── Shader cycles counter ──
        Op::ReadShaderCycles { .. } => InsnLatency::salu(),

        // ── Raw assembly (unknown latency, assume VALU) ──
        Op::RawAsm(_) => InsnLatency::valu(),
    }
}

/// Compute the critical-path length through a linear block of ops.
///
/// Uses ASAP scheduling to find the minimum number of cycles needed.
/// This is useful for estimating K-loop iteration latency.
pub fn critical_path_cycles(ops: &[Op]) -> u32 {
    if ops.is_empty() { return 0; }

    // For a linear block (no branches), critical path =
    // sum of all result latencies along the longest dependency chain.
    //
    // Simplified model: each op starts 1 cycle after the previous op issues,
    // but must wait for its data dependencies. Without full data-flow analysis,
    // we approximate by computing: max(sum_issue, sum_latency_chain).
    let mut total_issue: u32 = 0;
    let mut max_latency: u32 = 0;

    for op in ops {
        let lat = op_latency(op);
        total_issue += lat.issue;
        max_latency = max_latency.max(total_issue + lat.result);
    }

    max_latency
}

/// Estimate how many VALU ops can be executed while waiting for a VMEM load.
///
/// This is the key metric for VMEM/VALU interleaving.
/// On GFX1100: VMEM latency ≈ 490 shader cycles, VALU issue = 1 cycle/op.
/// → ~490 independent VALU ops can fill the VMEM latency.
pub fn valu_slots_per_vmem() -> u32 {
    // VMEM result latency / VALU issue rate
    490 / 1
}

/// Estimate how many VALU ops can be executed while waiting for a WMMA result.
///
/// On GFX1100: WMMA result latency ≈ 36 shader cycles.
/// → ~36 independent VALU ops (or another WMMA on different accumulators).
pub fn valu_slots_per_wmma() -> u32 {
    36 / 1
}

/// How many VALU-equivalent cycles does a WMMA instruction take?
///
/// VALU-normalized: 36 shader clocks / 18.3 ≈ 2, but empirically ~4
/// (due to pipeline stalls and dependency chains).
pub fn wmma_valu_equivalent() -> u32 {
    4
}

/// Summary statistics for a block of ops.
#[derive(Clone, Debug, Default)]
pub struct BlockStats {
    pub valu_count: u32,
    pub vtrans_count: u32,
    pub vmem_load_count: u32,
    pub vmem_store_count: u32,
    pub lds_count: u32,
    pub salu_count: u32,
    pub wmma_count: u32,
    pub ctrl_count: u32,
    pub total_issue_cycles: u32,
    pub estimated_critical_path: u32,
}

/// Analyze a block of ops and return summary statistics.
pub fn analyze_block(ops: &[Op]) -> BlockStats {
    let mut stats = BlockStats::default();
    for op in ops {
        let lat = op_latency(op);
        stats.total_issue_cycles += lat.issue;
        match lat.class {
            InsnClass::VALU => stats.valu_count += 1,
            InsnClass::VTRANS => stats.vtrans_count += 1,
            InsnClass::VMEM => {
                if matches!(op, Op::GlobalLoad { .. } | Op::BufferLoad { .. }) {
                    stats.vmem_load_count += 1;
                } else {
                    stats.vmem_store_count += 1;
                }
            }
            InsnClass::LDS => stats.lds_count += 1,
            InsnClass::SALU => stats.salu_count += 1,
            InsnClass::WMMA => stats.wmma_count += 1,
            InsnClass::CTRL => stats.ctrl_count += 1,
            InsnClass::CVT => stats.valu_count += 1, // CVT uses VALU pipe
            InsnClass::XLANE => stats.lds_count += 1, // ds_swizzle uses LDS pipe
            InsnClass::SMEM => stats.salu_count += 1,
        }
    }
    stats.estimated_critical_path = critical_path_cycles(ops);
    stats
}

/// Compute the "ILP potential" for a GEMM K-loop body.
///
/// ILP potential = how much of the VMEM/WMMA latency can be hidden
/// by issuing independent instructions in parallel.
///
/// Returns (hidden_fraction, bottleneck) where:
/// - hidden_fraction ∈ [0.0, 1.0] — 1.0 means all latency hidden
/// - bottleneck — which resource is limiting
pub fn ilp_potential(stats: &BlockStats) -> (f32, &'static str) {
    // WMMA latency to hide
    let wmma_latency_total = stats.wmma_count * 36;
    // VMEM latency to hide (loads only — stores are fire-and-forget)
    let vmem_latency_total = stats.vmem_load_count * 490;

    // Available VALU cycles to fill gaps
    let available_valu = stats.valu_count + stats.vtrans_count * 4;

    // Can we hide WMMA latency with VALU?
    let wmma_hidden = if wmma_latency_total > 0 {
        (available_valu as f32 / wmma_latency_total as f32).min(1.0)
    } else {
        1.0
    };

    // Can we hide VMEM latency with WMMA + VALU?
    let vmem_filler = stats.wmma_count * 36 + available_valu;
    let vmem_hidden = if vmem_latency_total > 0 {
        (vmem_filler as f32 / vmem_latency_total as f32).min(1.0)
    } else {
        1.0
    };

    let overall = wmma_hidden.min(vmem_hidden);
    let bottleneck = if wmma_hidden < vmem_hidden { "wmma_latency" }
        else if vmem_hidden < 0.5 { "vmem_latency" }
        else { "compute" };

    (overall, bottleneck)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valu_latency() {
        let op = Op::VAddF32 {
            dst: super::super::ir::VReg(0),
            src0: super::super::ir::Operand::VReg(super::super::ir::VReg(1)),
            src1: super::super::ir::Operand::VReg(super::super::ir::VReg(2)),
        };
        let lat = op_latency(&op);
        assert_eq!(lat.class, InsnClass::VALU);
        assert_eq!(lat.issue, 1);
        assert_eq!(lat.result, 1);
    }

    #[test]
    fn test_vmem_latency() {
        let op = Op::GlobalLoad {
            dst: super::super::ir::VReg(0),
            addr: super::super::ir::VReg(1),
            width: super::super::ir::Width::B32,
            offset: 0,
        };
        let lat = op_latency(&op);
        assert_eq!(lat.class, InsnClass::VMEM);
        assert_eq!(lat.result, 490);
    }

    #[test]
    fn test_wmma_latency() {
        let op = Op::Wmma {
            dst: super::super::ir::VReg(0),
            a: super::super::ir::VReg(8),
            b: super::super::ir::VReg(16),
            c: super::super::ir::VReg(0),
            format: super::super::ir::WmmaFormat::BF16_F32,
        };
        let lat = op_latency(&op);
        assert_eq!(lat.class, InsnClass::WMMA);
        assert_eq!(lat.result, 36);
        assert_eq!(lat.issue, 1);
    }

    #[test]
    fn test_vtrans_rcp() {
        let op = Op::VRcpF32 {
            dst: super::super::ir::VReg(0),
            src: super::super::ir::VReg(1),
        };
        let lat = op_latency(&op);
        assert_eq!(lat.class, InsnClass::VTRANS);
        assert_eq!(lat.result, 16);
        assert_eq!(lat.recip_throughput, 4.0);
    }

    #[test]
    fn test_valu_slots_computation() {
        assert_eq!(valu_slots_per_vmem(), 490);
        assert_eq!(valu_slots_per_wmma(), 36);
    }

    #[test]
    fn test_analyze_empty_block() {
        let stats = analyze_block(&[]);
        assert_eq!(stats.valu_count, 0);
        assert_eq!(stats.total_issue_cycles, 0);
    }

    #[test]
    fn test_critical_path_valu_chain() {
        use super::super::ir::*;
        // 4 VALU ops in a linear chain: critical path = 4 issue + 1 result
        let ops = vec![
            Op::VAddF32 { dst: VReg(1), src0: Operand::VReg(VReg(0)), src1: Operand::InlineFloat(1.0) },
            Op::VAddF32 { dst: VReg(2), src0: Operand::VReg(VReg(1)), src1: Operand::InlineFloat(2.0) },
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(2)), src1: Operand::InlineFloat(3.0) },
            Op::VAddF32 { dst: VReg(4), src0: Operand::VReg(VReg(3)), src1: Operand::InlineFloat(4.0) },
        ];
        let path = critical_path_cycles(&ops);
        assert!(path >= 4, "critical path should be >= 4 for 4 VALU: got {}", path);
    }

    /// Analyze a real GEMM kernel's K-loop body to establish baseline metrics.
    #[test]
    fn test_gemm_kloop_analysis() {
        use super::super::gemm_gen;

        // Generate a typical GEMM kernel
        let cfg = gemm_gen::GemmConfig {
            tile_m: 128, tile_n: 64, tile_k: 16, wg_size: 128,
            use_lds: true, double_buffer: true, split_k: None,
            lds_pad: 0, n_col_passes: 1, swap_grid: true,
            wgp_mode: true,
            transpose: gemm_gen::GemmTranspose::NT,
            epilogue: gemm_gen::EpilogueOp::StoreF32,
        };
        let kernel = gemm_gen::generate(&cfg);

        // Find the K-loop body (between label "ggen_loop" and branch back)
        let mut loop_start = None;
        let mut loop_end = None;
        for (i, op) in kernel.ops().iter().enumerate() {
            if let Op::Label(name) = op {
                if name.starts_with("ggen_loop") { loop_start = Some(i + 1); }
            }
            if let Op::Branch(target) = op {
                if target.starts_with("ggen_loop") { loop_end = Some(i); }
            }
            // Also check BranchScc1 since the loop uses s_cmp_lt + branch_scc1
            if let Op::BranchScc1(target) = op {
                if target.starts_with("ggen_loop") { loop_end = Some(i); }
            }
        }

        let (start, end) = match (loop_start, loop_end) {
            (Some(s), Some(e)) if s < e => (s, e),
            _ => {
                eprintln!("[analysis] No K-loop found in GEMM kernel");
                return;
            }
        };

        let loop_body = &kernel.ops()[start..end];
        let stats = analyze_block(loop_body);
        let (ilp, bottleneck) = ilp_potential(&stats);

        eprintln!("╔═══════════════════════════════════════════════════╗");
        eprintln!("║  GEMM K-loop Analysis: {}                 ║", cfg.name());
        eprintln!("╠═══════════════════════════════════════════════════╣");
        eprintln!("║  Loop body: {} ops ({} issue cycles)            ", loop_body.len(), stats.total_issue_cycles);
        eprintln!("║  VALU:  {:>3}  VTRANS: {:>3}  WMMA: {:>3}             ", stats.valu_count, stats.vtrans_count, stats.wmma_count);
        eprintln!("║  VMEM:  {:>3} ld + {:>3} st  LDS: {:>3}              ", stats.vmem_load_count, stats.vmem_store_count, stats.lds_count);
        eprintln!("║  SALU:  {:>3}  CTRL: {:>3}                          ", stats.salu_count, stats.ctrl_count);
        eprintln!("║  Critical path: {} cycles                        ", stats.estimated_critical_path);
        eprintln!("║  ILP potential: {:.1}% ({})                       ", ilp * 100.0, bottleneck);
        eprintln!("║  WMMA/VMEM ratio: {:.1}                          ",
            if stats.vmem_load_count > 0 { stats.wmma_count as f32 / stats.vmem_load_count as f32 } else { 0.0 });
        eprintln!("╚═══════════════════════════════════════════════════╝");

        // Sanity checks
        assert!(stats.wmma_count > 0, "GEMM should have WMMA instructions");
        assert!(stats.vmem_load_count > 0, "GEMM should have VMEM loads");
        assert!(stats.lds_count > 0, "GEMM should have LDS ops");
    }
}

