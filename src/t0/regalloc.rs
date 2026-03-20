//! T0 Register Allocator — Linear Scan with Liveness Analysis
//!
//! Maps virtual registers (VReg/SReg) to physical GPU registers.
//! Computes live intervals from IR ops and reuses dead physical registers.
//! Handles alignment constraints (WMMA needs 8-aligned VGPRs).

use std::collections::{HashMap, BTreeSet};
use super::ir::*;

// ============================================================================
// Allocation result
// ============================================================================

/// Result of register allocation: maps virtual → physical.
#[derive(Clone, Debug)]
pub struct RegAlloc {
    pub vgpr_map: HashMap<VReg, u8>,   // VReg → physical VGPR number
    pub sgpr_map: HashMap<SReg, u8>,   // SReg → physical SGPR number
    pub total_vgprs: u8,
    pub total_sgprs: u8,
}

impl RegAlloc {
    /// Get physical VGPR for a virtual register.
    pub fn phys_v(&self, v: VReg) -> u8 {
        if let Some(&p) = self.vgpr_map.get(&v) {
            return p;
        }
        panic!("VReg {:?} not allocated! Did you forget alloc_vreg()?", v);
    }

    /// Get physical SGPR for a virtual scalar register.
    pub fn phys_s(&self, s: SReg) -> u8 {
        self.sgpr_map[&s]
    }
}

// ============================================================================
// Live interval
// ============================================================================

/// A live interval for a VReg allocation (possibly multiple consecutive regs).
#[derive(Clone, Debug)]
struct LiveInterval {
    alloc_idx: usize,        // index into vreg_allocs
    vreg_base: VReg,         // first virtual register
    count: u32,              // number of consecutive VGPRs
    alignment: Alignment,
    last_use: usize,         // last instruction index where any VReg in this alloc is used
    phys_base: Option<u8>,   // assigned physical register (set during allocation)
}

// ============================================================================
// Linear scan allocator
// ============================================================================

/// Allocate registers with liveness-based reuse.
///
/// 1. Compute last-use index for each VRegAlloc by scanning ops
/// 2. Handle loops: extend last-use to loop end for any VReg used inside a loop
/// 3. Allocate in declaration order, expiring dead intervals and reusing their physical regs
pub fn allocate(
    vreg_allocs: &[VRegAlloc],
    sreg_allocs: &[SRegAlloc],
    ops: &[Op],
) -> RegAlloc {
    // ── Compute live intervals ──

    // Build VReg → alloc_idx mapping (which allocation does each VReg belong to?)
    let mut vreg_to_alloc: HashMap<VReg, usize> = HashMap::new();
    for (idx, va) in vreg_allocs.iter().enumerate() {
        for i in 0..va.count {
            vreg_to_alloc.insert(VReg(va.vreg.0 + i), idx);
        }
    }

    // Find first-use and last-use instruction index for each allocation
    let mut last_use = vec![0usize; vreg_allocs.len()];
    let mut first_use = vec![usize::MAX; vreg_allocs.len()];
    for (op_idx, op) in ops.iter().enumerate() {
        for vr in op.vreg_refs() {
            if let Some(&alloc_idx) = vreg_to_alloc.get(&vr) {
                if op_idx > last_use[alloc_idx] {
                    last_use[alloc_idx] = op_idx;
                }
                if op_idx < first_use[alloc_idx] {
                    first_use[alloc_idx] = op_idx;
                }
            }
            // VReg(0) = hardware tid, not in vreg_allocs — skip silently
        }
    }

    // Handle loops: find label → branch_scc1 pairs, extend live ranges
    let mut label_positions: HashMap<String, usize> = HashMap::new();
    let mut loop_ranges: Vec<(usize, usize)> = Vec::new(); // (start, end)

    for (op_idx, op) in ops.iter().enumerate() {
        if let Op::Label(name) = op {
            label_positions.insert(name.clone(), op_idx);
        }
    }
    for (op_idx, op) in ops.iter().enumerate() {
        if let Op::BranchScc1(target) = op {
            if let Some(&label_pos) = label_positions.get(target) {
                if label_pos < op_idx {
                    // Backward branch = loop: label_pos..op_idx
                    loop_ranges.push((label_pos, op_idx));
                }
            }
        }
    }

    // Extend last-use for VRegs used inside loops
    for &(loop_start, loop_end) in &loop_ranges {
        for alloc_idx in 0..vreg_allocs.len() {
            // If this alloc is used anywhere inside the loop, extend to loop_end
            if last_use[alloc_idx] >= loop_start && last_use[alloc_idx] <= loop_end {
                last_use[alloc_idx] = loop_end;
            }
        }
    }

    // Build live intervals
    let mut intervals: Vec<LiveInterval> = vreg_allocs.iter().enumerate().map(|(idx, va)| {
        LiveInterval {
            alloc_idx: idx,
            vreg_base: va.vreg,
            count: va.count,
            alignment: va.alignment,
            last_use: last_use[idx],
            phys_base: None,
        }
    }).collect();

    // ── Allocate SGPRs (bump, no liveness needed) ──
    let mut sgpr_map: HashMap<SReg, u8> = HashMap::new();
    let mut next_sgpr: u8 = 5; // s0:s1 = kernarg ptr, s2/s3/s4 = TGID

    for sa in sreg_allocs {
        if sa.count == 1 {
            sgpr_map.insert(sa.sreg, next_sgpr);
            next_sgpr += 1;
        } else if sa.count == 2 {
            let aligned = (next_sgpr + 1) & !1;
            sgpr_map.insert(sa.sreg, aligned);
            sgpr_map.insert(SReg(sa.sreg.0 + 1), aligned + 1);
            next_sgpr = aligned + 2;
        } else {
            let base = next_sgpr;
            for i in 0..sa.count {
                sgpr_map.insert(SReg(sa.sreg.0 + i), base + i as u8);
            }
            next_sgpr = base + sa.count as u8;
        }
        assert!(next_sgpr < 106, "SGPR overflow!");
    }

    // ── Allocate VGPRs with liveness-based reuse ──

    // Free list: available physical register ranges
    // Each entry is (start_phys, count) — a contiguous block of free registers
    let mut free_ranges: Vec<(u8, u32)> = Vec::new();
    let mut max_vgpr: u8 = 1; // v0 is reserved for WORKITEM_ID_X
    let mut vgpr_map: HashMap<VReg, u8> = HashMap::new();
    vgpr_map.insert(VReg(0), 0); // v0 = hardware thread_id

    // Active intervals: sorted by last_use so we can expire efficiently
    let mut active: Vec<usize> = Vec::new(); // indices into intervals

    // Allocate in declaration order
    for idx in 0..intervals.len() {
        // VReg(0) = hardware v0 (WORKITEM_ID_X), already pre-mapped above.
        // CRITICAL: Do NOT reallocate it, and do NOT add to active list
        // so v0 can never be reclaimed via the expire mechanism.
        if intervals[idx].vreg_base == VReg(0) && intervals[idx].count == 1 {
            intervals[idx].phys_base = Some(0);
            continue;
        }

        let current_alloc_idx = intervals[idx].alloc_idx;

        // Expire dead intervals: return their physical regs to free list.
        // An interval is SAFE to expire only if its last_use is BEFORE the
        // first_use of the current allocation. This prevents freeing registers
        // that are still needed between the current alloc's definition and
        // its eventual last use.
        let current_first_use = first_use[current_alloc_idx];
        let mut expired = Vec::new();
        for (active_pos, &active_idx) in active.iter().enumerate() {
            if intervals[active_idx].last_use < current_first_use {
                if let Some(phys) = intervals[active_idx].phys_base {
                    let count = intervals[active_idx].count;
                    free_ranges.push((phys, count));
                    expired.push(active_pos);
                }
            }
        }
        // Remove expired (reverse order to preserve indices)
        expired.sort();
        for &pos in expired.iter().rev() {
            active.remove(pos);
        }

        let count = intervals[idx].count;
        let align = intervals[idx].alignment;

        // Try to find a suitable range in the free list
        let mut found = None;
        for (fi, &(start, fcount)) in free_ranges.iter().enumerate() {
            // Apply alignment
            let aligned = match align {
                Alignment::None => start,
                Alignment::Align2 => (start + 1) & !1,
                Alignment::Align4 => (start + 3) & !3,
                Alignment::Align8 => (start + 7) & !7,
            };
            let waste = (aligned - start) as u32;
            if fcount >= count + waste {
                found = Some((fi, aligned, waste));
                break;
            }
        }

        let phys_base;
        if let Some((fi, aligned, waste)) = found {
            phys_base = aligned;
            let (start, fcount) = free_ranges[fi];
            let used = count + waste;
            if fcount > used {
                // Split: keep remainder
                free_ranges[fi] = (start + used as u8, fcount - used);
                // Also add waste as free (if alignment caused gap)
                if waste > 0 {
                    free_ranges.push((start, waste));
                }
            } else {
                free_ranges.remove(fi);
                if waste > 0 {
                    free_ranges.push((start, waste));
                }
            }
        } else {
            // No suitable free range found — allocate from the end
            let aligned = match align {
                Alignment::None => max_vgpr,
                Alignment::Align2 => (max_vgpr + 1) & !1,
                Alignment::Align4 => (max_vgpr + 3) & !3,
                Alignment::Align8 => (max_vgpr + 7) & !7,
            };
            phys_base = aligned;
            let end = aligned as u32 + count;
            assert!(end <= 255, "VGPR overflow at {}+{}", aligned, count);
            max_vgpr = end as u8;
        }

        // Record allocation
        intervals[idx].phys_base = Some(phys_base);
        for i in 0..count {
            vgpr_map.insert(VReg(intervals[idx].vreg_base.0 + i), phys_base + i as u8);
        }
        if phys_base + count as u8 > max_vgpr {
            max_vgpr = phys_base + count as u8;
        }

        active.push(idx);
    }

    RegAlloc {
        total_vgprs: max_vgpr,
        total_sgprs: next_sgpr,
        vgpr_map,
        sgpr_map,
    }
}
