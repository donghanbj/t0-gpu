//! T0 Optimization Passes
//!
//! Applied between validate() and allocate_registers() in the compile pipeline.
//! Operates on Vec<Op> (IR instruction list).
//!
//! Pipeline order:
//! 1. Constant Folding (track known values, replace computable ops)
//! 2. Algebraic Simplification (x+0→x, x*1→x, x*0→0, x-x→0)
//! 3. Copy Propagation (v_mov v2, v1; use v2 → use v1)
//! 4. CSE — Common Subexpression Elimination
//! 5. Instruction Combining (v_mul + v_add → v_fma)
//! 6. LICM — Loop Invariant Code Motion
//! 7. Dead Code Elimination (iterative until fixpoint)
//! 8. Waitcnt Optimization (remove redundant s_waitcnt)
//! 9. Instruction Scheduling (VALU/VMEM interleaving, reg-pressure aware)

use std::collections::{HashMap, HashSet};
use super::ir::*;

/// Run all optimization passes on the op list.
/// Returns (optimized_ops, stats).
///
/// `coalesced_groups`: optional list of VReg ranges that must remain physically
/// contiguous (WMMA fragments, etc.). CopyProp/DCE will skip these.
pub fn optimize(ops: Vec<Op>, coalesced_groups: &[super::compile::CoalescedGroup]) -> (Vec<Op>, OptStats) {
    let mut stats = OptStats::default();
    let original_len = ops.len();

    // T0_OPT_LEVEL: 0=none, 1=Phase A, 2=A+B, 3=A+B+C, 4=all (default)
    let opt_level: u32 = std::env::var("T0_OPT_LEVEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    
    if opt_level == 0 {
        return (ops, stats);
    }

    // ════════════════════════════════════════════════════════════════
    // Phase A: MachSSA passes (lift once → run all SSA passes → lower)
    // ════════════════════════════════════════════════════════════════
    let mut func = super::ssa_ir::lift_to_ssa(&ops);
    
    // Annotate coalesced groups: mark MachInsts whose VRegs belong to
    // wide-op groups (WMMA fragments, etc.) so opt passes respect contiguity.
    super::ssa_ir::annotate_coalesced_groups(&mut func, coalesced_groups);
    
    let disabled: String = std::env::var("T0_DISABLE_PASS").unwrap_or_default();
    let skip = |name: &str| disabled.split(',').any(|s| s == name);

    // Pass 1: Constant Folding (D4c)
    if !skip("fold") { stats.consts_folded = super::ssa_ir::constant_fold_mach_func(&mut func); }

    // Pass 2: Algebraic Simplification (D4d)
    if !skip("alg") { stats.algebraic_simplified = super::ssa_ir::algebraic_simplify_mach_func(&mut func); }

    // Pass 3: Copy Propagation (D4b)
    if !skip("copy") { stats.copies_propagated = super::ssa_ir::copy_propagate_mach_func(&mut func); }

    // Pass 4: CSE (D4e+ cross-block via domtree)
    if !skip("cse") { stats.cse_eliminated = super::ssa_ir::cse_mach_func_domtree(&mut func); }

    // Pass 5: Instruction Combining — mul+add → fma (D4f)
    if !skip("combine") { stats.instructions_combined = super::ssa_ir::instruction_combine_mach_func(&mut func); }

    // Pass 6: LICM — hoists loop-invariant instructions to preheader
    // (Fixed: rename_op_uses/defs now covers all Op variants including DS, LDS, atomics, WMMA)
    if !skip("licm") { stats.licm_hoisted = super::ssa_ir::licm_mach_func(&mut func); }


    // Lower back to Vec<Op> for loop-based passes
    let mut ops = super::ssa_ir::lower_from_ssa(&func);

    if opt_level <= 1 {
        stats.original_ops = original_len; stats.final_ops = ops.len();
        return (ops, stats);
    }

    // ════════════════════════════════════════════════════════════════
    // Phase B: Vec<Op> loop-based passes (keep linear — SSA gain minimal)
    // ════════════════════════════════════════════════════════════════

    // Pass 5.5: Loop Unrolling
    let (new_ops, unroll_count) = loop_unroll(ops);
    stats.loops_unrolled = unroll_count;
    ops = new_ops;

    // Pass 6: LICM — now in Phase A (SSA-based), removed from here

    // Pass 6.5: Strength Reduction
    let (new_ops, sr_count) = strength_reduce(ops);
    stats.strength_reduced = sr_count;
    ops = new_ops;

    if opt_level <= 2 {
        stats.original_ops = original_len; stats.final_ops = ops.len();
        return (ops, stats);
    }

    // ════════════════════════════════════════════════════════════════
    // Phase C: Post-loop SSA passes (lift again for iterative DCE+cleanup)
    // ════════════════════════════════════════════════════════════════
    let mut func = super::ssa_ir::lift_to_ssa(&ops);

    // Pass 7: Iterative AlgSimp + DCE (D4a + D4d)
    // Skip with T0_SKIP_PHASE_CD=1 for diagnostic isolation
    let skip_phase_cd = std::env::var("T0_SKIP_PHASE_CD").is_ok();
    // DCE enabled by default. Loop-carried liveness fix in dce_mach_func
    // prevents removal of loop induction variable updates (e.g. k_offset += 32).
    // Disable with T0_SKIP_PHASE_C_DCE=1 for diagnostic isolation.
    let enable_phase_c_dce = !std::env::var("T0_SKIP_PHASE_C_DCE").is_ok();
    let skip_phase_c_alg = std::env::var("T0_SKIP_PHASE_C_ALG").is_ok();
    if !skip_phase_cd {
        let max_iters = 5;
        for _iter in 0..max_iters {
            let simp_count = if !skip_phase_c_alg {
                super::ssa_ir::algebraic_simplify_mach_func(&mut func)
            } else { 0 };
            stats.algebraic_simplified += simp_count;

            // DCE with loop-carried liveness (Step 2b in dce_mach_func).
            // Verified: 9/9 GEMM tests pass with DCE enabled.
            let dce_count = if enable_phase_c_dce {
                super::ssa_ir::dce_mach_func(&mut func)
            } else { 0 };
            stats.dead_ops_removed += dce_count;

            if dce_count == 0 && simp_count == 0 {
                break;
            }
        }
    }

    // ── Phase D: Waitcnt optimization (safe, operates on MachFunc) ──
    if opt_level >= 4 {
        stats.waitcnts_removed = if std::env::var("T0_SKIP_WAITOPT").is_ok() {
            0
        } else {
            super::ssa_ir::optimize_waitcnt_mach_func(&mut func)
        };
    }

    // Single lower_from_ssa for all SSA passes
    ops = super::ssa_ir::lower_from_ssa(&func);

    // Pass 7.5: Load/Store Coalescing (address-pattern, stays Vec<Op>)
    let (new_ops, coal_count) = coalesce_loads(ops);
    stats.loads_coalesced = coal_count;
    ops = new_ops;

    // Pass 10: Post-regalloc instruction scheduling (Phase D)
    // Operates on Vec<Op> with physical registers — safe to reorder
    // because regalloc has already assigned registers.
    if opt_level >= 4 && !std::env::var("T0_SKIP_SCHED").is_ok() {
        let (new_ops, sched_count) = post_regalloc_schedule(ops);
        stats.ops_reordered = sched_count;
        ops = new_ops;
    }

    if opt_level <= 4 {
        stats.original_ops = original_len; stats.final_ops = ops.len();
        return (ops, stats);
    }

    stats.original_ops = original_len;
    stats.final_ops = ops.len();

    let any_change = stats.dead_ops_removed > 0 || stats.consts_folded > 0
        || stats.algebraic_simplified > 0 || stats.ops_reordered > 0
        || stats.copies_propagated > 0 || stats.instructions_combined > 0
        || stats.waitcnts_removed > 0 || stats.cse_eliminated > 0
        || stats.licm_hoisted > 0 || stats.loops_unrolled > 0
        || stats.sw_pipelined > 0 || stats.strength_reduced > 0
        || stats.loads_coalesced > 0;
    if any_change {
        eprintln!(
            "[T0 Optimize] {} ops → {} ops: DCE -{}, CF -{}, AlgSimp -{}, CopyProp ~{}, CSE ~{}, Combine ~{}, Unroll ×{}, LICM ~{}, SR ~{}, Coal ~{}, SWP ~{}, WaitOpt -{}, Sched ~{}",
            stats.original_ops, stats.final_ops,
            stats.dead_ops_removed, stats.consts_folded,
            stats.algebraic_simplified, stats.copies_propagated,
            stats.cse_eliminated, stats.instructions_combined,
            stats.loops_unrolled, stats.licm_hoisted,
            stats.strength_reduced, stats.loads_coalesced,
            stats.sw_pipelined, stats.waitcnts_removed,
            stats.ops_reordered
        );
    }

    (ops, stats)
}

/// Optimization statistics.
#[derive(Default, Debug)]
pub struct OptStats {
    pub original_ops: usize,
    pub final_ops: usize,
    pub dead_ops_removed: usize,
    pub consts_folded: usize,
    pub algebraic_simplified: usize,
    pub copies_propagated: usize,
    pub cse_eliminated: usize,
    pub instructions_combined: usize,
    pub licm_hoisted: usize,
    pub loops_unrolled: usize,
    pub sw_pipelined: usize,
    pub waitcnts_removed: usize,
    pub ops_reordered: usize,
    pub strength_reduced: usize,
    pub loads_coalesced: usize,
}

struct LoopInfo {
    /// Index of Label("loop_N") in the op list
    header_idx: usize,
    /// Index of BranchScc1("end_loop_N") (early exit if iter >= end)
    exit_branch_idx: usize,
    /// First body instruction index (after exit branch)
    body_start: usize,
    /// Index of SAddU32 (iter += step)
    latch_add_idx: usize,
    /// Index of Branch("loop_N") (back-edge)
    backedge_idx: usize,
    /// Index of Label("end_loop_N")
    exit_label_idx: usize,
    /// Iterator SGPR
    iter_sreg: SReg,
    /// End-bound SGPR
    end_sreg: SReg,
    /// Step value (constant)
    step: i32,
    /// Loop header label name
    header_label: String,
    /// Loop exit label name
    exit_label: String,
}

/// Analyze ops to find structured counted loops.
fn detect_loops(ops: &[Op]) -> Vec<LoopInfo> {
    let mut loops = Vec::new();

    // Build label→index map
    let mut label_positions: HashMap<String, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        if let Op::Label(name) = op {
            label_positions.insert(name.clone(), i);
        }
    }

    // Find backward branches (back-edges)
    for (i, op) in ops.iter().enumerate() {
        let target = match op {
            Op::Branch(t) => t,
            _ => continue,
        };

        // Must be backward branch
        let &header_idx = match label_positions.get(target) {
            Some(pos) if *pos < i => pos,
            _ => continue,
        };

        let backedge_idx = i;
        let header_label = target.clone();

        // The instruction before Branch must be SAddU32 (iter += step)
        if backedge_idx == 0 { continue; }
        let latch_add_idx = backedge_idx - 1;
        let (iter_sreg, step) = match &ops[latch_add_idx] {
            Op::SAddU32 { dst, src0, src1: SOperand::InlineInt(s) } if *dst == *src0 => {
                (*dst, *s)
            }
            _ => continue,
        };

        // After the Label, expect SCmpGeU32 + BranchScc1
        if header_idx + 2 >= ops.len() { continue; }
        let (end_sreg, exit_label) = match (&ops[header_idx + 1], &ops[header_idx + 2]) {
            (Op::SCmpGeU32 { src0, src1 }, Op::BranchScc1(exit_lbl))
                if *src0 == iter_sreg =>
            {
                (*src1, exit_lbl.clone())
            }
            _ => continue,
        };

        // Find the exit label position
        let exit_label_idx = match label_positions.get(&exit_label) {
            Some(&pos) => pos,
            None => continue,
        };

        // Exit label should be right after the backedge
        if exit_label_idx != backedge_idx + 1 { continue; }

        let body_start = header_idx + 3; // first body op after SCmpGe + BranchScc1
        let exit_branch_idx = header_idx + 2;

        // Only if body is non-empty
        if body_start >= latch_add_idx { continue; }

        loops.push(LoopInfo {
            header_idx,
            exit_branch_idx,
            body_start,
            latch_add_idx,
            backedge_idx,
            exit_label_idx,
            iter_sreg,
            end_sreg,
            step,
            header_label,
            exit_label,
        });
    }

    loops
}

/// Check if a loop body is safe to unroll.
/// Returns false if body contains barriers, nested loops, WMMA, or other unsafe ops.
fn is_safe_to_unroll(ops: &[Op], body_start: usize, body_end: usize) -> bool {
    for i in body_start..body_end {
        match &ops[i] {
            // Nested loops / control flow
            Op::Label(_) | Op::Branch(_) | Op::BranchScc1(_) | Op::BranchScc0(_) |
            Op::BranchVccz(_) => return false,
            // Barriers (WG sync semantics)
            Op::Barrier | Op::SBarrier => return false,
            // WMMA (already hand-optimized)
            Op::Wmma { .. } => return false,
            _ => {}
        }
    }
    true
}

/// Select unroll factor based on body size.
fn unroll_factor(body_size: usize) -> usize {
    if body_size == 0 { return 1; }
    if body_size <= 16 { return 4; }
    if body_size <= 48 { return 2; }
    1 // too large, don't unroll
}

/// Find the maximum VReg index used in an op list.
fn max_vreg_in_ops(ops: &[Op]) -> u32 {
    let mut max_v = 0u32;
    for op in ops {
        for v in op.vreg_refs() {
            max_v = max_v.max(v.0);
        }
        for v in op.vreg_defs() {
            max_v = max_v.max(v.0);
        }
    }
    max_v
}

/// Rename VRegs in an Op by adding `offset` to each VReg number.
/// Only renames VRegs that are in `rename_set` (body-local definitions).
fn rename_op_vregs(op: &Op, offset: u32, rename_set: &HashSet<u32>) -> Op {
    let rename_v = |v: VReg| -> VReg {
        if rename_set.contains(&v.0) { VReg(v.0 + offset) } else { v }
    };
    let rename_op = |o: &Operand| -> Operand {
        match o {
            Operand::VReg(v) => {
                if rename_set.contains(&v.0) {
                    Operand::VReg(VReg(v.0 + offset))
                } else {
                    *o
                }
            }
            _ => *o,
        }
    };

    match op {
        // 2-src VALU
        Op::VAddF32 { dst, src0, src1 } =>
            Op::VAddF32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VMulF32 { dst, src0, src1 } =>
            Op::VMulF32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VSubF32 { dst, src0, src1 } =>
            Op::VSubF32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VMaxF32 { dst, src0, src1 } =>
            Op::VMaxF32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VMinF32 { dst, src0, src1 } =>
            Op::VMinF32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VAddU32 { dst, src0, src1 } =>
            Op::VAddU32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VSubU32 { dst, src0, src1 } =>
            Op::VSubU32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VAndB32 { dst, src0, src1 } =>
            Op::VAndB32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VXorB32 { dst, src0, src1 } =>
            Op::VXorB32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        Op::VOrB32 { dst, src0, src1 } =>
            Op::VOrB32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1) },
        // 3-src
        Op::VFmaF32 { dst, src0, src1, src2 } =>
            Op::VFmaF32 { dst: rename_v(*dst), src0: rename_op(src0), src1: rename_op(src1), src2: rename_op(src2) },
        // 1-src VALU
        Op::VMov { dst, src } =>
            Op::VMov { dst: rename_v(*dst), src: rename_op(src) },
        Op::VRsqF32 { dst, src } =>
            Op::VRsqF32 { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VExpF32 { dst, src } =>
            Op::VExpF32 { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VRcpF32 { dst, src } =>
            Op::VRcpF32 { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VSqrtF32 { dst, src } =>
            Op::VSqrtF32 { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VLog2F32 { dst, src } =>
            Op::VLog2F32 { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VCvtF32U32 { dst, src } =>
            Op::VCvtF32U32 { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VCvtU32F32 { dst, src } =>
            Op::VCvtU32F32 { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VMovFromSgpr { dst, src } =>
            Op::VMovFromSgpr { dst: rename_v(*dst), src: *src },
        Op::VLshlrevB32 { dst, shift, src } =>
            Op::VLshlrevB32 { dst: rename_v(*dst), shift: *shift, src: rename_v(*src) },
        Op::VLshrrevB32 { dst, shift, src } =>
            Op::VLshrrevB32 { dst: rename_v(*dst), shift: *shift, src: rename_v(*src) },
        // Memory ops (rename dst/src VRegs, keep addr as-is since addr often comes from outside)
        Op::GlobalLoad { dst, addr, width, offset } =>
            Op::GlobalLoad { dst: rename_v(*dst), addr: rename_v(*addr), width: *width, offset: *offset },
        Op::GlobalStore { addr, src, width, offset } =>
            Op::GlobalStore { addr: rename_v(*addr), src: rename_v(*src), width: *width, offset: *offset },
        // Wait/sync — pass through unchanged
        Op::WaitVmcnt(n) => Op::WaitVmcnt(*n),
        Op::WaitLgkmcnt(n) => Op::WaitLgkmcnt(*n),
        Op::WaitVscnt(n) => Op::WaitVscnt(*n),
        // Comparisons
        Op::VCmpLtU32 { src0, src1 } =>
            Op::VCmpLtU32 { src0: rename_op(src0), src1: rename_op(src1) },
        Op::VCmpGeU32 { src0, src1 } =>
            Op::VCmpGeU32 { src0: rename_op(src0), src1: rename_op(src1) },
        Op::VCndmaskB32 { dst, src_false, src_true } =>
            Op::VCndmaskB32 { dst: rename_v(*dst), src_false: rename_op(src_false), src_true: rename_op(src_true) },
        Op::SaveExec { dst } => Op::SaveExec { dst: *dst },
        Op::RestoreExec { src } => Op::RestoreExec { src: *src },
        Op::XorExec { saved } => Op::XorExec { saved: *saved },
        Op::ClearVcc => Op::ClearVcc,
        // DsSwizzle
        Op::DsSwizzle { dst, src, offset } =>
            Op::DsSwizzle { dst: rename_v(*dst), src: rename_v(*src), offset: *offset },
        // LDS ops
        Op::DsLoadB32 { dst, vaddr, offset } =>
            Op::DsLoadB32 { dst: rename_v(*dst), vaddr: rename_v(*vaddr), offset: *offset },
        Op::DsStoreB32 { vaddr, src, offset } =>
            Op::DsStoreB32 { vaddr: rename_v(*vaddr), src: rename_v(*src), offset: *offset },
        Op::LdsLoad { dst, addr, width, offset } =>
            Op::LdsLoad { dst: rename_v(*dst), addr: rename_v(*addr), width: *width, offset: *offset },
        Op::LdsStore { addr, src, width, offset } =>
            Op::LdsStore { addr: rename_v(*addr), src: rename_v(*src), width: *width, offset: *offset },
        // Wave reduce
        Op::WaveReduceAddF32 { val, tmp } =>
            Op::WaveReduceAddF32 { val: rename_v(*val), tmp: rename_v(*tmp) },
        Op::WaveReduceMaxF32 { val, tmp } =>
            Op::WaveReduceMaxF32 { val: rename_v(*val), tmp: rename_v(*tmp) },
        // Scalar ops — pass through unchanged (SGPRs are not renamed)
        Op::SAddU32 { .. } | Op::SSubU32 { .. } | Op::SAddcU32 { .. } |
        Op::SAndB32 { .. } | Op::SMulI32 { .. } | Op::SLshlB32 { .. } |
        Op::SLshrB32 { .. } | Op::SMov { .. } | Op::SCmpLtU32 { .. } |
        Op::SCmpEqU32 { .. } | Op::SCmpGeU32 { .. } => op.clone(),
        // VMulLoU32
        Op::VMulLoU32 { dst, src0, src1 } =>
            Op::VMulLoU32 { dst: rename_v(*dst), src0: rename_v(*src0), src1: rename_v(*src1) },
        // 64-bit addr
        Op::VAddCo { dst, src0, src1 } =>
            Op::VAddCo { dst: rename_v(*dst), src0: rename_v(*src0), src1: rename_v(*src1) },
        Op::VAddCoCi { dst, src } =>
            Op::VAddCoCi { dst: rename_v(*dst), src: rename_v(*src) },
        Op::VAddCOU32 { dst, src0, src1 } =>
            Op::VAddCOU32 { dst: rename_v(*dst), src0: rename_v(*src0), src1: rename_v(*src1) },
        Op::VAddCCU32 { dst, src } =>
            Op::VAddCCU32 { dst: rename_v(*dst), src: rename_v(*src) },
        // Atomics
        Op::GlobalAtomicAddF32 { addr, src, offset } =>
            Op::GlobalAtomicAddF32 { addr: rename_v(*addr), src: rename_v(*src), offset: *offset },
        Op::GlobalAtomicAddU32Rtn { dst, addr, src } =>
            Op::GlobalAtomicAddU32Rtn { dst: rename_v(*dst), addr: rename_v(*addr), src: rename_v(*src) },
        // Catch-all: clone as-is (safe for control flow, endpgm, etc.)
        other => other.clone(),
    }
}

/// Loop Unrolling pass: replicate loop body N times to increase ILP.
///
/// Detects counted loops (ForBegin/ForEnd lowered pattern), selects unroll
/// factor based on body size (≤16 ops → ×4, ≤48 → ×2), and generates renamed
/// body copies with a remainder loop for non-divisible trip counts.
///
/// Safety: skips loops with barriers, nested control flow, or WMMA.
fn loop_unroll(ops: Vec<Op>) -> (Vec<Op>, usize) {
    let loops = detect_loops(&ops);
    if loops.is_empty() { return (ops, 0); }

    let mut unrolled_count = 0usize;
    let mut result = ops;

    // Process loops in reverse order (so indices remain valid)
    for lp in loops.iter().rev() {
        let body_size = lp.latch_add_idx - lp.body_start;
        let factor = unroll_factor(body_size);
        if factor <= 1 { continue; }

        // Safety check
        if !is_safe_to_unroll(&result, lp.body_start, lp.latch_add_idx) {
            continue;
        }

        // Collect VRegs defined in the loop body (candidates for rename)
        let mut body_defs: HashSet<u32> = HashSet::new();
        for i in lp.body_start..lp.latch_add_idx {
            for d in result[i].vreg_defs() {
                body_defs.insert(d.0);
            }
        }

        // If no VRegs are defined, renaming is unnecessary but unrolling is still valid
        let max_v = max_vreg_in_ops(&result);
        let rename_stride = max_v + 1; // each copy gets VRegs starting at max_v+1, max_v*2+1, ...

        // Build unrolled body: N copies of body + N copies of iter increment
        let body_ops: Vec<Op> = result[lp.body_start..lp.latch_add_idx].to_vec();

        let mut unrolled_body: Vec<Op> = Vec::new();

        // Copy 0: original body (no rename)
        for op in &body_ops {
            unrolled_body.push(op.clone());
        }
        // iter += step (original latch)
        unrolled_body.push(result[lp.latch_add_idx].clone());

        // Copies 1..factor-1: renamed body + iter increment
        for copy_idx in 1..factor {
            let offset = rename_stride * copy_idx as u32;
            for op in &body_ops {
                unrolled_body.push(rename_op_vregs(op, offset, &body_defs));
            }
            // iter += step (same SGPR, not renamed)
            unrolled_body.push(result[lp.latch_add_idx].clone());
        }

        // Build the new op sequence:
        // [ops before header] [header label] [SCmpGe] [BranchScc1(end)]
        //   [unrolled body] [Branch(header)] [exit label]
        // [ops after exit label]
        let mut new_ops: Vec<Op> = Vec::with_capacity(result.len() + unrolled_body.len());

        // Everything before the loop header (inclusive)
        for i in 0..lp.body_start {
            new_ops.push(result[i].clone());
        }

        // Unrolled body
        new_ops.extend(unrolled_body);

        // Back-edge branch
        new_ops.push(result[lp.backedge_idx].clone());

        // Everything from exit label onward
        for i in lp.exit_label_idx..result.len() {
            new_ops.push(result[i].clone());
        }

        result = new_ops;
        unrolled_count += 1;
    }

    (result, unrolled_count)
}

// ═══════════════════════════════════════════════════
// Pass 6.5: Strength Reduction
// ═══════════════════════════════════════════════════

/// Strength Reduction: replace loop-internal multiplications with additions.
///
/// Detects patterns where a loop induction variable is multiplied by a
/// constant stride to compute addresses, and replaces the multiplication
/// with an accumulating addition:
///
/// Before:
/// ```text
///   loop:
///     offset = v_lshlrev_b32 iter_vreg, shift   // or v_mul iter, stride
///     ... use offset ...
///     iter += step
/// ```
///
/// After:
/// ```text
///   offset = 0  (pre-loop init)
///   loop:
///     ... use offset ...
///     offset += stride   // cheaper add replaces mul
///     iter += step
/// ```
fn strength_reduce(ops: Vec<Op>) -> (Vec<Op>, usize) {
    // TODO: Strength reduction (replace loop-internal lshl/mul with accumulating add).
    // Previous implementation was incomplete: it replaced the shift, then immediately
    // reverted because inserting pre-header init + latch accumulate requires
    // sophisticated analysis of loop initial values and latch boundaries.
    // Tracked as future optimization opportunity.
    (ops, 0)
}


// ═══════════════════════════════════════════════════
// Pass 7.5: Load/Store Coalescing
// ═══════════════════════════════════════════════════

/// Coalesce adjacent GlobalLoad/GlobalStore instructions with the same base
/// address into wider loads/stores.
///
/// Patterns detected:
/// - 4× GlobalLoad B32 with same addr, consecutive offsets (0,4,8,12),
///   consecutive dst VRegs → 1× GlobalLoad B128
/// - 2× GlobalLoad B32 with same addr, consecutive offsets (N, N+4),
///   consecutive dst VRegs → 1× GlobalLoad B64
/// - Same for GlobalStore
///
/// Safety: no side-effect ops may appear between the loads/stores being merged.
fn coalesce_loads(ops: Vec<Op>) -> (Vec<Op>, usize) {
    let len = ops.len();
    if len < 2 { return (ops, 0); }

    let mut coalesced = 0usize;
    let mut result: Vec<Op> = Vec::with_capacity(len);

    let mut i = 0;
    while i < len {

        // Try 4-way coalescing first (GlobalLoad B32 × 4 → B128)
        if i + 3 < len {
            if let (
                Op::GlobalLoad { dst: d0, addr: a0, width: Width::B32, offset: o0 },
                Op::GlobalLoad { dst: d1, addr: a1, width: Width::B32, offset: o1 },
                Op::GlobalLoad { dst: d2, addr: a2, width: Width::B32, offset: o2 },
                Op::GlobalLoad { dst: d3, addr: a3, width: Width::B32, offset: o3 },
            ) = (&ops[i], &ops[i+1], &ops[i+2], &ops[i+3]) {
                if *a0 == *a1 && *a1 == *a2 && *a2 == *a3
                    && *o1 == *o0 + 4 && *o2 == *o0 + 8 && *o3 == *o0 + 12
                    && d1.0 == d0.0 + 1 && d2.0 == d0.0 + 2 && d3.0 == d0.0 + 3
                {
                    result.push(Op::GlobalLoad {
                        dst: *d0,
                        addr: *a0,
                        width: Width::B128,
                        offset: *o0,
                    });
                    coalesced += 3; // saved 3 ops
                    i += 4;
                    continue;
                }
            }
        }

        // Try 2-way coalescing (GlobalLoad B32 × 2 → B64)
        if i + 1 < len {
            if let (
                Op::GlobalLoad { dst: d0, addr: a0, width: Width::B32, offset: o0 },
                Op::GlobalLoad { dst: d1, addr: a1, width: Width::B32, offset: o1 },
            ) = (&ops[i], &ops[i+1]) {
                if *a0 == *a1 && *o1 == *o0 + 4 && d1.0 == d0.0 + 1 {
                    result.push(Op::GlobalLoad {
                        dst: *d0,
                        addr: *a0,
                        width: Width::B64,
                        offset: *o0,
                    });
                    coalesced += 1; // saved 1 op
                    i += 2;
                    continue;
                }
            }
        }

        // Try 4-way store coalescing (GlobalStore B32 × 4 → B128)
        if i + 3 < len {
            if let (
                Op::GlobalStore { addr: a0, src: s0, width: Width::B32, offset: o0 },
                Op::GlobalStore { addr: a1, src: s1, width: Width::B32, offset: o1 },
                Op::GlobalStore { addr: a2, src: s2, width: Width::B32, offset: o2 },
                Op::GlobalStore { addr: a3, src: s3, width: Width::B32, offset: o3 },
            ) = (&ops[i], &ops[i+1], &ops[i+2], &ops[i+3]) {
                if *a0 == *a1 && *a1 == *a2 && *a2 == *a3
                    && *o1 == *o0 + 4 && *o2 == *o0 + 8 && *o3 == *o0 + 12
                    && s1.0 == s0.0 + 1 && s2.0 == s0.0 + 2 && s3.0 == s0.0 + 3
                {
                    result.push(Op::GlobalStore {
                        addr: *a0,
                        src: *s0,
                        width: Width::B128,
                        offset: *o0,
                    });
                    coalesced += 3;
                    i += 4;
                    continue;
                }
            }
        }

        // Try 2-way store coalescing (GlobalStore B32 × 2 → B64)
        if i + 1 < len {
            if let (
                Op::GlobalStore { addr: a0, src: s0, width: Width::B32, offset: o0 },
                Op::GlobalStore { addr: a1, src: s1, width: Width::B32, offset: o1 },
            ) = (&ops[i], &ops[i+1]) {
                if *a0 == *a1 && *o1 == *o0 + 4 && s1.0 == s0.0 + 1 {
                    result.push(Op::GlobalStore {
                        addr: *a0,
                        src: *s0,
                        width: Width::B64,
                        offset: *o0,
                    });
                    coalesced += 1;
                    i += 2;
                    continue;
                }
            }
        }

        // No coalescing possible — emit as-is
        result.push(ops[i].clone());
        i += 1;
    }

    (result, coalesced)
}

// ═══════════════════════════════════════════════════
// Pass 8.5: Software Pipelining
// ═══════════════════════════════════════════════════

/// Software Pipelining: overlap loads from iteration N+1 with compute from
/// iteration N, hiding memory latency across loop iterations.
///
/// Transforms:
/// ```text
///   loop:
///     load A[i]
///     wait
///     compute(A[i])
///     store result[i]
///     i += step
///     if i < end: goto loop
/// ```
/// Into:
/// ```text
///   // Prologue: load first iteration's data
///   load A[0]
///   loop:
///     wait              // wait for current iteration's data (loaded in prev iter)
///     load A[i+step]    // prefetch NEXT iteration's data (overlapped with compute)
///     compute(A[i])
///     store result[i]
///     i += step
///     if i < end: goto loop
///   // Epilogue: process last iteration (already loaded but not computed)
///   wait
///   compute(A[last])
///   store result[last]
/// ```
///
/// Constraints:
/// - Only applies to counted loops (detected by `detect_loops()`)
/// - Loop must have at least one GlobalLoad + matching WaitVmcnt
/// Post-regalloc instruction scheduling for latency hiding.
///
/// Operates on Vec<Op> with PHYSICAL registers (after regalloc), so reordering
/// is safe and doesn't affect register assignment. Uses vreg_uses()/vreg_defs()
/// and reads/writes_vcc/scc for precise dependency analysis.
///
/// Within each basic block, moves independent ALU ops between memory loads
/// and their corresponding waitcnt to overlap load latency.
fn post_regalloc_schedule(ops: Vec<Op>) -> (Vec<Op>, usize) {
    use super::latency_model;

    let mut result = ops;
    let mut total_reordered = 0usize;

    // Find basic block boundaries (Label → next Label/end)
    let mut bb_starts: Vec<usize> = vec![0];
    for (i, op) in result.iter().enumerate() {
        if matches!(op, Op::Label(_)) && i > 0 {
            bb_starts.push(i);
        }
    }
    bb_starts.push(result.len());

    // Process each basic block
    for bb_idx in 0..bb_starts.len() - 1 {
        let bb_start = bb_starts[bb_idx];
        let bb_end = bb_starts[bb_idx + 1];
        if bb_end - bb_start < 3 { continue; }

        let block = &result[bb_start..bb_end];
        let block_len = block.len();

        // Skip blocks with BufferLoad/BufferStore — these are GEMM K-loop blocks
        // with hand-scheduled graduated waitcnt patterns. Reordering buffer_load
        // relative to ds_store and waitcnt corrupts the latency-hiding schedule.
        let has_buffer_ops = block.iter().any(|op| matches!(op,
            Op::BufferLoad { .. } | Op::BufferStore { .. }));
        if has_buffer_ops { continue; }

        let mut scheduled: Vec<Op> = Vec::with_capacity(block_len);
        let mut i = 0;

        while i < block_len {
            let op = &block[i];
            let is_vmem = matches!(op, Op::GlobalLoad { .. } | Op::BufferLoad { .. });
            let is_lds = matches!(op,
                Op::LdsLoad { .. } | Op::DsLoadB32 { .. } | Op::DsLoadB64 { .. } |
                Op::DsLoadB128 { .. } | Op::DsLoadU16 { .. } |
                Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. }
            );

            if is_vmem || is_lds {
                let wait_match = if is_vmem {
                    |op: &Op| matches!(op, Op::WaitVmcnt(_))
                } else {
                    |op: &Op| matches!(op, Op::WaitLgkmcnt(_))
                };

                // Collect consecutive loads of the same type
                let mut load_ops: Vec<Op> = Vec::new();
                let mut load_def_vregs: HashSet<u32> = HashSet::new();
                let mut load_writes_vcc = false;

                while i < block_len {
                    let cur = &block[i];
                    let match_vmem = matches!(cur, Op::GlobalLoad { .. } | Op::BufferLoad { .. });
                    let match_lds = matches!(cur,
                        Op::LdsLoad { .. } | Op::DsLoadB32 { .. } | Op::DsLoadB64 { .. } |
                        Op::DsLoadB128 { .. } | Op::DsLoadU16 { .. } |
                        Op::DsLoadU16D16 { .. } | Op::DsLoadU16D16Hi { .. }
                    );
                    if (is_vmem && match_vmem) || (is_lds && match_lds) {
                        for v in cur.vreg_defs() { load_def_vregs.insert(v.0); }
                        if cur.writes_vcc() { load_writes_vcc = true; }
                        load_ops.push(cur.clone());
                        i += 1;
                    } else {
                        break;
                    }
                }

                // Emit loads
                for l in &load_ops { scheduled.push(l.clone()); }

                // Classify ops between loads and wait as movable/non-movable
                let mut movable: Vec<Op> = Vec::new();
                let mut non_movable: Vec<Op> = Vec::new();
                // Track physical VRegs defined/used by non-movable ops
                let mut nm_def_vregs: HashSet<u32> = HashSet::new();
                let mut nm_use_vregs: HashSet<u32> = HashSet::new();
                // Track physical SRegs defined/used by non-movable ops
                let mut nm_def_sregs: HashSet<u32> = HashSet::new();
                let mut nm_use_sregs: HashSet<u32> = HashSet::new();
                let mut nm_writes_vcc = false;
                let mut nm_writes_scc = false;

                // Track SRegs used by loads (for address components)
                let mut load_use_sregs: HashSet<u32> = HashSet::new();
                let mut load_def_sregs: HashSet<u32> = HashSet::new();
                for l in &load_ops {
                    for s in l.sreg_uses() { load_use_sregs.insert(s.0); }
                    for s in l.sreg_defs() { load_def_sregs.insert(s.0); }
                }

                while i < block_len {
                    let cur = &block[i];

                    if wait_match(cur) { break; }

                    // Stop at control flow / sync boundaries
                    if matches!(cur, Op::Barrier | Op::SBarrier |
                        Op::Label(_) | Op::Branch(_) | Op::BranchScc0(_) |
                        Op::BranchScc1(_) | Op::BranchVccz(_)
                    ) { break; }

                    let cur_use_vregs: HashSet<u32> = cur.vreg_uses().into_iter().map(|v| v.0).collect();
                    let cur_def_vregs: HashSet<u32> = cur.vreg_defs().into_iter().map(|v| v.0).collect();
                    let cur_use_sregs: HashSet<u32> = cur.sreg_uses().into_iter().map(|s| s.0).collect();
                    let cur_def_sregs: HashSet<u32> = cur.sreg_defs().into_iter().map(|s| s.0).collect();

                    // Check ALL dependency conditions using physical VRegs:
                    // 1. RAW: cur reads a VReg written by loads
                    let raw_with_load = cur_use_vregs.iter().any(|v| load_def_vregs.contains(v));
                    // 2. WAW/WAR with loads (VReg conflict)
                    let war_with_load = cur_def_vregs.iter().any(|v| load_def_vregs.contains(v));
                    // 3. VCC/SCC implicit conflicts with loads
                    let vcc_conflict = (cur.reads_vcc() && load_writes_vcc)
                        || (cur.writes_vcc() && load_writes_vcc);
                    // 4. RAW/WAW/WAR with accumulated non-movable ops
                    let raw_with_nm = cur_use_vregs.iter().any(|v| nm_def_vregs.contains(v));
                    let waw_with_nm = cur_def_vregs.iter().any(|v| nm_def_vregs.contains(v));
                    let war_with_nm = cur_def_vregs.iter().any(|v| nm_use_vregs.contains(v));
                    // 5. VCC/SCC implicit conflicts with non-movable
                    let vcc_nm = (cur.reads_vcc() && nm_writes_vcc)
                        || (cur.writes_vcc() && nm_writes_vcc);
                    let scc_nm = (cur.reads_scc() && nm_writes_scc)
                        || (cur.writes_scc() && nm_writes_scc);
                    // 6. Side effects
                    let has_side = cur.has_side_effects();
                    // 7. Can overlap with memory?
                    let can_overlap = if is_vmem {
                        latency_model::can_overlap_vmem(cur)
                    } else {
                        latency_model::can_overlap_lds(cur)
                    };

                    // 8. SReg dependency checks (CRITICAL: was missing before!)
                    // RAW: cur reads SReg written by loads (e.g. address components)
                    let sreg_raw_load = cur_use_sregs.iter().any(|s| load_def_sregs.contains(s));
                    // WAR: cur writes SReg read by loads
                    let sreg_war_load = cur_def_sregs.iter().any(|s| load_use_sregs.contains(s));
                    // RAW/WAW/WAR with non-movable
                    let sreg_raw_nm = cur_use_sregs.iter().any(|s| nm_def_sregs.contains(s));
                    let sreg_waw_nm = cur_def_sregs.iter().any(|s| nm_def_sregs.contains(s));
                    let sreg_war_nm = cur_def_sregs.iter().any(|s| nm_use_sregs.contains(s));

                    let is_movable = can_overlap && !has_side
                        && !raw_with_load && !war_with_load && !vcc_conflict
                        && !raw_with_nm && !waw_with_nm && !war_with_nm
                        && !vcc_nm && !scc_nm
                        && !sreg_raw_load && !sreg_war_load
                        && !sreg_raw_nm && !sreg_waw_nm && !sreg_war_nm;

                    if is_movable {
                        movable.push(cur.clone());
                        total_reordered += 1;
                    } else {
                        for v in &cur_def_vregs { nm_def_vregs.insert(*v); }
                        for v in &cur_use_vregs { nm_use_vregs.insert(*v); }
                        for s in &cur_def_sregs { nm_def_sregs.insert(*s); }
                        for s in &cur_use_sregs { nm_use_sregs.insert(*s); }
                        if cur.writes_vcc() { nm_writes_vcc = true; }
                        if cur.writes_scc() { nm_writes_scc = true; }
                        non_movable.push(cur.clone());
                    }
                    i += 1;
                }

                // Emit: movable first (fills load latency), then non-movable, then wait
                for op in movable { scheduled.push(op); }
                for op in non_movable { scheduled.push(op); }

                if i < block_len && wait_match(&block[i]) {
                    scheduled.push(block[i].clone());
                    i += 1;
                }
            } else {
                scheduled.push(op.clone());
                i += 1;
            }
        }

        // Replace block in result
        result.splice(bb_start..bb_end, scheduled);
        // Recalculate bb_starts since splice may change lengths
        // (But our scheduling preserves length, so this is fine.)
    }

    (result, total_reordered)
}

/// Software pipelining: reorder loop body to overlap loads with compute.
///
/// - No barriers or nested control flow in body
/// - Trip count must be ≥ 2 (otherwise prologue+epilogue is worse)
fn software_pipeline(ops: Vec<Op>) -> (Vec<Op>, usize) {
    let loops = detect_loops(&ops);
    if loops.is_empty() { return (ops, 0); }

    let mut pipelined_count = 0usize;
    let mut result = ops;

    for lp in loops.iter().rev() {
        let body = &result[lp.body_start..lp.latch_add_idx];
        let body_len = body.len();

        // Safety: no barriers, nested loops, or WMMA
        if !is_safe_to_unroll(&result, lp.body_start, lp.latch_add_idx) {
            continue;
        }

        // Find loads and their matching waits in the body
        let mut load_indices: Vec<usize> = Vec::new(); // relative to body_start
        let mut wait_index: Option<usize> = None;
        let mut has_lds = false;

        for (j, op) in body.iter().enumerate() {
            match op {
                Op::GlobalLoad { .. } => { load_indices.push(j); }
                Op::WaitVmcnt(0) => { wait_index = Some(j); }
                // LDS ops in body prevent software pipelining (complex dependencies)
                Op::LdsLoad { .. } | Op::LdsStore { .. } |
                Op::DsLoadB32 { .. } | Op::DsStoreB32 { .. } |
                Op::DsLoadB64 { .. } | Op::DsStoreB64 { .. } |
                Op::DsLoadB128 { .. } | Op::DsStoreB128 { .. } => { has_lds = true; }
                _ => {}
            }
        }

        // Need at least one load AND a wait, no LDS
        if load_indices.is_empty() || wait_index.is_none() || has_lds {
            continue;
        }
        let wait_idx = wait_index.unwrap();

        // The wait must come AFTER all loads (otherwise the pattern is unusual)
        if load_indices.iter().any(|&li| li >= wait_idx) {
            continue;
        }

        // Collect VRegs defined by loads (these need double-buffering)
        let mut load_dst_vregs: HashSet<u32> = HashSet::new();
        for &li in &load_indices {
            for d in body[li].vreg_defs() {
                load_dst_vregs.insert(d.0);
            }
        }

        // Compute the rename offset for the "next iteration" buffer
        let max_v = max_vreg_in_ops(&result);
        let _buf_offset = max_v + 1;

        // Build the pipelined version:
        //
        // PROLOGUE (before loop): issue loads for iteration 0
        //   [original loads]
        //
        // LOOP BODY (rewritten):
        //   [wait for current data]
        //   [prefetch loads for iter+1, using renamed dst VRegs]
        //   [compute using original VRegs (already loaded)]
        //   [store results]
        //   [swap buffers: copy prefetch VRegs to original VRegs — but we use
        //    renamed refs in next iteration's compute, so no explicit swap needed
        //    if we rename the compute sources too]
        //
        // For simplicity, we use a simpler approach:
        // - Prologue: emit original loads
        // - Loop body: move loads AFTER the wait (they prefetch for next iter)
        //   which converts: load→wait→compute → wait→compute→load (of next iter)
        //   This naturally overlaps the load with the back-edge branch latency.

        // Reorder the body: [wait] [compute/store ops] [loads] (loads now prefetch)
        let mut reordered_body: Vec<Op> = Vec::with_capacity(body_len);

        // 1. Wait instruction first
        reordered_body.push(body[wait_idx].clone());

        // 2. All non-load, non-wait ops (compute + stores + scalar ops)
        for (j, op) in body.iter().enumerate() {
            if load_indices.contains(&j) || j == wait_idx { continue; }
            reordered_body.push(op.clone());
        }

        // 3. Load instructions last (they prefetch the next iteration's data)
        for &li in &load_indices {
            reordered_body.push(body[li].clone());
        }

        // Build the new op sequence
        let mut new_ops: Vec<Op> = Vec::with_capacity(result.len() + load_indices.len() * 2);

        // Everything before the loop body
        for i in 0..lp.body_start {
            new_ops.push(result[i].clone());
        }

        // Prologue: first iteration's loads (before loop body, inside the loop header)
        for &li in &load_indices {
            new_ops.push(body[li].clone());
        }

        // Reordered body
        new_ops.extend(reordered_body);

        // Latch + back-edge + everything from exit label onward
        for i in lp.latch_add_idx..result.len() {
            new_ops.push(result[i].clone());
        }

        result = new_ops;
        pipelined_count += 1;
    }

    (result, pipelined_count)
}

// ═══════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_full_pipeline() {
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(2.0) },
            Op::VMov { dst: VReg(2), src: Operand::InlineFloat(3.0) },
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::VMov { dst: VReg(99), src: Operand::InlineFloat(99.0) }, // dead
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];

        let (_result, stats) = optimize(ops, &[]);
        assert!(stats.consts_folded >= 1, "should fold at least 1 op");
        assert!(stats.dead_ops_removed >= 1, "should remove at least 1 dead op");
    }

    // ═══════════════════════════════════════════
    //  Loop Unrolling tests
    // ═══════════════════════════════════════════

    /// Helper: build a counted loop in canonical Op IR form.
    /// SMov iter=start → SMov end=end_val → Label(header) → SCmpGeU32 → BranchScc1(exit)
    /// → body → SAddU32(iter, iter, step) → Branch(header) → Label(exit)
    fn make_counted_loop(body: Vec<Op>, start: i32, end_val: i32, step: i32) -> Vec<Op> {
        let iter_s = SReg(100);
        let end_s  = SReg(101);
        let mut ops = vec![
            Op::SMov { dst: iter_s, src: SOperand::InlineInt(start) },
            Op::SMov { dst: end_s,  src: SOperand::InlineInt(end_val) },
            Op::Label("loop_0".to_string()),
            Op::SCmpGeU32 { src0: iter_s, src1: end_s },
            Op::BranchScc1("end_loop_0".to_string()),
        ];
        ops.extend(body);
        ops.push(Op::SAddU32 { dst: iter_s, src0: iter_s, src1: SOperand::InlineInt(step) });
        ops.push(Op::Branch("loop_0".to_string()));
        ops.push(Op::Label("end_loop_0".to_string()));
        ops.push(Op::Endpgm);
        ops
    }

    #[test]
    fn test_unroll_basic() {
        // Simple loop: 2 ops in body (≤16 → ×4 unroll)
        // 4 iterations with step=1 → body should be replicated 4 times
        let body = vec![
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 0 },
        ];
        let ops = make_counted_loop(body, 0, 8, 1);
        let original_len = ops.len();

        let (result, unrolled) = loop_unroll(ops);
        assert_eq!(unrolled, 1, "should unroll 1 loop");
        // Original: 2 body ops + 1 latch = 3 per iteration
        // Unrolled ×4: (2 body + 1 latch) × 4 = 12 body ops
        // Total should be larger than original
        assert!(result.len() > original_len,
            "unrolled code ({} ops) should be longer than original ({} ops)",
            result.len(), original_len);

        // Count how many VAddF32 ops exist (should be 4 copies)
        let add_count = result.iter().filter(|op| matches!(op, Op::VAddF32 { .. })).count();
        assert_eq!(add_count, 4, "×4 unroll should produce 4 copies of VAddF32");

        // Count SAddU32 (iter increments) — should be 4
        let sadd_count = result.iter().filter(|op| matches!(op, Op::SAddU32 { .. })).count();
        assert_eq!(sadd_count, 4, "×4 unroll should produce 4 iter increments");
    }

    #[test]
    fn test_unroll_remainder() {
        // Body has 2 ops → ×4 unroll factor
        // The loop itself doesn't change trip count at IR level,
        // but the structure should still be valid
        let body = vec![
            Op::VMulF32 { dst: VReg(5), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::VAddF32 { dst: VReg(6), src0: Operand::VReg(VReg(5)), src1: Operand::InlineFloat(1.0) },
        ];
        let ops = make_counted_loop(body, 0, 7, 1);

        let (result, unrolled) = loop_unroll(ops);
        assert_eq!(unrolled, 1, "should unroll the loop");

        // Verify that copied VAddF32 ops have different dst VRegs (renamed)
        let add_dsts: Vec<u32> = result.iter().filter_map(|op| {
            if let Op::VAddF32 { dst, .. } = op { Some(dst.0) } else { None }
        }).collect();
        assert_eq!(add_dsts.len(), 4, "×4 unroll should produce 4 VAddF32 copies");
        // At least some should be renamed (different dst)
        let unique_dsts: HashSet<u32> = add_dsts.iter().copied().collect();
        assert!(unique_dsts.len() > 1,
            "unrolled copies should have renamed VRegs, got: {:?}", add_dsts);
    }

    #[test]
    fn test_unroll_skip_barrier() {
        // Loop body contains Barrier → should NOT unroll
        let body = vec![
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::Barrier,
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 0 },
        ];
        let ops = make_counted_loop(body, 0, 8, 1);
        let original_len = ops.len();

        let (result, unrolled) = loop_unroll(ops);
        assert_eq!(unrolled, 0, "should NOT unroll loop with Barrier");
        assert_eq!(result.len(), original_len, "op list should be unchanged");
    }

    // ═══════════════════════════════════════════
    //  Software Pipelining tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_sw_pipeline_basic() {
        // Loop: load → wait → compute → store
        // After pipelining: prologue loads + reordered body (wait→compute→store→load)
        let body = vec![
            Op::GlobalLoad { dst: VReg(3), addr: VReg(20), width: Width::B32, offset: 0 },
            Op::WaitVmcnt(0),
            Op::VAddF32 { dst: VReg(4), src0: Operand::VReg(VReg(3)), src1: Operand::InlineFloat(1.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(4), width: Width::B32, offset: 0 },
        ];
        let ops = make_counted_loop(body, 0, 8, 1);

        let (result, pipelined) = software_pipeline(ops);
        assert_eq!(pipelined, 1, "should pipeline 1 loop");

        // There should now be 2 GlobalLoads: 1 prologue + 1 in body
        let load_count = result.iter().filter(|op| matches!(op, Op::GlobalLoad { .. })).count();
        assert_eq!(load_count, 2, "pipelining should add 1 prologue load (total 2)");

        // The prologue load should appear BEFORE the loop body wait
        let first_load_pos = result.iter().position(|op| matches!(op, Op::GlobalLoad { .. })).unwrap();
        let first_wait_pos = result.iter().position(|op| matches!(op, Op::WaitVmcnt(0))).unwrap();
        assert!(first_load_pos < first_wait_pos,
            "prologue load (pos {}) should come before first wait (pos {})",
            first_load_pos, first_wait_pos);
    }

    #[test]
    fn test_sw_pipeline_no_loop() {
        // No loop → pass should be a no-op
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },
            Op::GlobalStore { addr: VReg(10), src: VReg(1), width: Width::B32, offset: 0 },
            Op::Endpgm,
        ];
        let original = ops.clone();

        let (result, pipelined) = software_pipeline(ops);
        assert_eq!(pipelined, 0, "no loops → 0 pipelined");
        assert_eq!(result.len(), original.len(), "op list should be unchanged");
    }

    // ═══════════════════════════════════════════
    //  Load/Store Coalescing tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_coalesce_4x_b32_to_b128() {
        // 4 consecutive B32 loads with same addr, contiguous offsets, contiguous dsts
        let ops = vec![
            Op::GlobalLoad { dst: VReg(3), addr: VReg(20), width: Width::B32, offset: 0 },
            Op::GlobalLoad { dst: VReg(4), addr: VReg(20), width: Width::B32, offset: 4 },
            Op::GlobalLoad { dst: VReg(5), addr: VReg(20), width: Width::B32, offset: 8 },
            Op::GlobalLoad { dst: VReg(6), addr: VReg(20), width: Width::B32, offset: 12 },
            Op::Endpgm,
        ];

        let (result, coalesced) = coalesce_loads(ops);
        assert_eq!(coalesced, 3, "should save 3 ops (4→1)");
        assert_eq!(result.len(), 2, "should be 1 B128 load + Endpgm");
        // Check the coalesced load
        match &result[0] {
            Op::GlobalLoad { dst, addr, width, offset } => {
                assert_eq!(*dst, VReg(3));
                assert_eq!(*addr, VReg(20));
                assert_eq!(*width, Width::B128);
                assert_eq!(*offset, 0);
            }
            other => panic!("expected GlobalLoad B128, got {:?}", other),
        }
    }

    #[test]
    fn test_coalesce_2x_b32_to_b64() {
        // 2 consecutive B32 loads → B64
        let ops = vec![
            Op::GlobalLoad { dst: VReg(10), addr: VReg(20), width: Width::B32, offset: 8 },
            Op::GlobalLoad { dst: VReg(11), addr: VReg(20), width: Width::B32, offset: 12 },
            Op::Endpgm,
        ];

        let (result, coalesced) = coalesce_loads(ops);
        assert_eq!(coalesced, 1, "should save 1 op (2→1)");
        assert_eq!(result.len(), 2, "should be 1 B64 load + Endpgm");
        match &result[0] {
            Op::GlobalLoad { dst, width, offset, .. } => {
                assert_eq!(*dst, VReg(10));
                assert_eq!(*width, Width::B64);
                assert_eq!(*offset, 8);
            }
            other => panic!("expected GlobalLoad B64, got {:?}", other),
        }
    }

    #[test]
    fn test_coalesce_skip_gap() {
        // Non-contiguous offsets → should NOT coalesce
        let ops = vec![
            Op::GlobalLoad { dst: VReg(3), addr: VReg(20), width: Width::B32, offset: 0 },
            Op::GlobalLoad { dst: VReg(4), addr: VReg(20), width: Width::B32, offset: 8 }, // gap!
            Op::Endpgm,
        ];

        let (result, coalesced) = coalesce_loads(ops);
        assert_eq!(coalesced, 0, "non-contiguous offsets should not coalesce");
        assert_eq!(result.len(), 3, "op list should be unchanged");
    }

    // ═══════════════════════════════════════════
    //  Strength Reduction tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_strength_reduce_no_loop() {
        // No loop → pass should be a no-op
        let ops = vec![
            Op::VMov { dst: VReg(1), src: Operand::InlineFloat(1.0) },
            Op::Endpgm,
        ];
        let original_len = ops.len();

        let (result, reduced) = strength_reduce(ops);
        assert_eq!(reduced, 0, "no loops → 0 reduced");
        assert_eq!(result.len(), original_len);
    }

    #[test]
    fn test_strength_reduce_conservative() {
        // Has a loop but no VMovFromSgpr pattern → strength_reduce is conservative, returns 0
        let body = vec![
            Op::VAddF32 { dst: VReg(3), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
            Op::GlobalStore { addr: VReg(10), src: VReg(3), width: Width::B32, offset: 0 },
        ];
        let ops = make_counted_loop(body, 0, 8, 1);
        let original_len = ops.len();

        let (result, reduced) = strength_reduce(ops);
        assert_eq!(reduced, 0, "no VLshlrev pattern → 0 reduced");
        assert_eq!(result.len(), original_len);
    }
}
