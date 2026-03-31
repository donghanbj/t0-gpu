//! RDNA3 GFX1100 K-loop Pipeline Simulator (v2)
//!
//! Models 4 concurrent execution pipelines with **full data-dependency tracking**:
//! 1. **VALU/WMMA** — shared issue port. WMMA occupies 4 cycles, VALU 1 cycle.
//! 2. **LDS** — ds_load/ds_store, independent pipe, configurable result latency.
//! 3. **VMEM** — buffer_load/global_store, independent pipe, configurable result latency.
//! 4. **SALU** — scalar ALU + control flow, independent pipe, 1 cycle.
//!
//! ## Data Dependency Tracking
//! Tracks RAW (Read-After-Write) hazards for VRegs by recording the cycle at which
//! each register's value becomes available. Instructions stall if their source
//! operands are not yet ready.
//!
//! ## ASM Parser
//! Includes a lightweight ASM text parser (`parse_asm_kloop`) that converts
//! disassembly text (from `T0_DUMP_ASM=1` or rocBLAS/Triton dumps) into
//! `AsmInsn` sequences for simulation.
//!
//! # Usage
//! ```rust,ignore
//! let kernel = tile_ir::lower_gemm(&spec);
//! let result = kloop_simulator::analyze_tile_gemm(&spec);
//! result.print_report();
//! ```

use std::collections::{HashMap, VecDeque};
use super::ir::{Op, VReg};
use super::insn_latency::{self, InsnClass};

// ============================================================================
// Hardware Parameters (tunable for calibration)
// ============================================================================

/// Tunable hardware timing parameters for GFX1100.
///
/// All latencies in **shader clock cycles** (~2.5 GHz on RX 7900 XTX).
/// Use `HwParams::default()` for calibrated defaults, or tune for experiments.
#[derive(Clone, Debug)]
pub struct HwParams {
    /// VALU simple instruction issue cost (cycles)
    pub valu_issue: u64,
    /// WMMA instruction issue cost (occupies VALU pipe for this many cycles)
    pub wmma_issue: u64,
    /// WMMA result latency (cycles until accumulator readable for dependent WMMA)
    pub wmma_result_latency: u64,
    /// LDS load/store result latency (cycles until data ready in register)
    pub lds_latency: u64,
    /// VMEM load result latency (cycles until data ready; DRAM-dominated)
    pub vmem_latency: u64,
    /// s_barrier synchronization overhead (cycles).
    /// Tunable: set via calibration. Initial default = 30.
    pub barrier_cost: u64,
    /// Minimum s_delay_alu penalty per hint (cycles)
    pub delay_alu_cost: u64,
    /// GPU clock frequency (GHz) for absolute TFLOPS conversion
    pub clock_ghz: f64,
    /// Number of CUs (96 for RX 7900 XTX)
    pub n_cus: u32,
}

impl Default for HwParams {
    fn default() -> Self {
        HwParams {
            valu_issue: 1,
            wmma_issue: 4,         // WMMA occupies VALU pipe for 4 cycles
            wmma_result_latency: 8, // Accumulator readable 8 cycles after issue
            lds_latency: 20,       // ds_load_b128 result latency
            vmem_latency: 300,     // buffer_load: ~300 cycles (tuned; raw DRAM ~500)
            barrier_cost: 30,      // s_barrier: calibrate via hw_probe
            delay_alu_cost: 1,     // s_delay_alu forced stall per hint
            clock_ghz: 2.5,
            n_cus: 96,
        }
    }
}

// ============================================================================
// Data Dependency Tracker
// ============================================================================

/// Tracks per-VReg availability cycle for RAW hazard detection.
#[derive(Clone, Debug, Default)]
struct DepTracker {
    /// Cycle at which each VReg's value becomes readable.
    /// Missing entries imply "available at cycle 0" (pre-loop live-in).
    ready: HashMap<u32, u64>,
}

impl DepTracker {
    /// Get the earliest cycle at which a VReg value is available.
    fn ready_cycle(&self, vreg: VReg) -> u64 {
        self.ready.get(&vreg.0).copied().unwrap_or(0)
    }

    /// Get the earliest cycle at which ALL source regs of an op are available.
    fn sources_ready(&self, op: &Op) -> u64 {
        let uses = op.vreg_uses();
        uses.iter().map(|v| self.ready_cycle(*v)).max().unwrap_or(0)
    }

    /// Record that a set of VRegs will be available at `cycle`.
    fn set_defs(&mut self, op: &Op, ready_cycle: u64) {
        for d in op.vreg_defs() {
            self.ready.insert(d.0, ready_cycle);
        }
    }
}

// ============================================================================
// Pipeline State
// ============================================================================

/// Hardware pipeline state during simulation.
#[derive(Clone, Debug)]
struct PipelineState {
    /// Current global cycle counter
    cycle: u64,
    /// VALU/WMMA pipe: earliest cycle it can accept a new instruction
    valu_ready: u64,
    /// In-flight LDS operations: completion times (sorted ascending)
    lds_inflight: VecDeque<u64>,
    /// In-flight VMEM operations: completion times (sorted ascending)
    vmem_inflight: VecDeque<u64>,
    /// Data dependency tracker
    deps: DepTracker,

    // ── Statistics ──
    valu_busy: u64,
    lds_issue: u64,
    vmem_issue: u64,
    stall_waitcnt: u64,
    stall_barrier: u64,
    stall_delay_alu: u64,
    stall_dep: u64,       // NEW: cycles stalled on data dependencies (RAW)
}

impl PipelineState {
    fn new() -> Self {
        PipelineState {
            cycle: 0,
            valu_ready: 0,
            lds_inflight: VecDeque::new(),
            vmem_inflight: VecDeque::new(),
            deps: DepTracker::default(),
            valu_busy: 0,
            lds_issue: 0,
            vmem_issue: 0,
            stall_waitcnt: 0,
            stall_barrier: 0,
            stall_delay_alu: 0,
            stall_dep: 0,
        }
    }

    fn advance_to(&mut self, target: u64) {
        if target > self.cycle {
            self.cycle = target;
        }
    }

    /// Wait for data dependencies (RAW hazard) and VALU pipe availability.
    fn wait_for_valu_and_deps(&mut self, op: &Op) {
        let dep_ready = self.deps.sources_ready(op);
        let stall_dep = dep_ready.saturating_sub(self.cycle);
        let stall_pipe = self.valu_ready.saturating_sub(self.cycle.max(dep_ready));
        self.stall_dep += stall_dep;
        self.advance_to(dep_ready.max(self.valu_ready));
    }

    fn issue_valu(&mut self, hw: &HwParams, op: &Op) {
        self.wait_for_valu_and_deps(op);
        self.valu_busy += hw.valu_issue;
        let result_ready = self.cycle + hw.valu_issue; // VALU: 1-cycle result
        self.deps.set_defs(op, result_ready);
        self.valu_ready = self.cycle + hw.valu_issue;
        self.cycle += 1;
    }

    fn issue_wmma(&mut self, hw: &HwParams, op: &Op) {
        self.wait_for_valu_and_deps(op);
        self.valu_busy += hw.wmma_issue;
        // WMMA result: accumulator readable after wmma_result_latency cycles
        let result_ready = self.cycle + hw.wmma_result_latency;
        self.deps.set_defs(op, result_ready);
        self.valu_ready = self.cycle + hw.wmma_issue;
        self.cycle += 1;
    }

    fn issue_vtrans(&mut self, hw: &HwParams, op: &Op) {
        self.wait_for_valu_and_deps(op);
        self.valu_busy += 4;
        let result_ready = self.cycle + 16; // transcendental: 16 cycle result
        self.deps.set_defs(op, result_ready);
        self.valu_ready = self.cycle + 4;
        self.cycle += 1;
    }

    fn issue_lds_load(&mut self, hw: &HwParams, op: &Op) {
        // Wait for address registers to be ready (deps check)
        let dep_ready = self.deps.sources_ready(op);
        let stall = dep_ready.saturating_sub(self.cycle);
        self.stall_dep += stall;
        self.advance_to(dep_ready);

        let completion = self.cycle + hw.lds_latency;
        self.lds_inflight.push_back(completion);
        self.deps.set_defs(op, completion); // data available after LDS latency
        self.lds_issue += 1;
        self.cycle += 1;
    }

    fn issue_lds_store(&mut self, hw: &HwParams, op: &Op) {
        let dep_ready = self.deps.sources_ready(op);
        let stall = dep_ready.saturating_sub(self.cycle);
        self.stall_dep += stall;
        self.advance_to(dep_ready);

        let completion = self.cycle + hw.lds_latency;
        self.lds_inflight.push_back(completion);
        self.lds_issue += 1;
        self.cycle += 1;
    }

    fn issue_vmem_load(&mut self, hw: &HwParams, op: &Op) {
        let dep_ready = self.deps.sources_ready(op);
        let stall = dep_ready.saturating_sub(self.cycle);
        self.stall_dep += stall;
        self.advance_to(dep_ready);

        let completion = self.cycle + hw.vmem_latency;
        self.vmem_inflight.push_back(completion);
        self.deps.set_defs(op, completion);
        self.vmem_issue += 1;
        self.cycle += 1;
    }

    fn issue_vmem_store(&mut self, op: &Op) {
        let dep_ready = self.deps.sources_ready(op);
        let stall = dep_ready.saturating_sub(self.cycle);
        self.stall_dep += stall;
        self.advance_to(dep_ready);

        self.vmem_issue += 1;
        self.cycle += 1;
    }

    fn issue_salu(&mut self) {
        self.cycle += 1;
    }

    fn wait_lgkmcnt(&mut self, n: u8) {
        let n = n as usize;
        while self.lds_inflight.len() > n {
            if let Some(&earliest) = self.lds_inflight.front() {
                let stall = earliest.saturating_sub(self.cycle);
                self.stall_waitcnt += stall;
                self.advance_to(earliest);
                self.lds_inflight.pop_front();
            } else { break; }
        }
    }

    fn wait_vmcnt(&mut self, n: u8) {
        let n = n as usize;
        while self.vmem_inflight.len() > n {
            if let Some(&earliest) = self.vmem_inflight.front() {
                let stall = earliest.saturating_sub(self.cycle);
                self.stall_waitcnt += stall;
                self.advance_to(earliest);
                self.vmem_inflight.pop_front();
            } else { break; }
        }
    }

    fn barrier(&mut self, hw: &HwParams) {
        self.stall_barrier += hw.barrier_cost;
        self.cycle += hw.barrier_cost;
    }
}

// ============================================================================
// Simulation Result
// ============================================================================

/// Result of K-loop pipeline simulation.
#[derive(Clone, Debug)]
pub struct SimResult {
    /// Total simulated cycles for the K-loop body
    pub total_cycles: u64,
    /// VALU/WMMA pipe busy cycles (issue-side)
    pub valu_busy: u64,
    /// LDS issue cycles
    pub lds_issue: u64,
    /// VMEM issue cycles
    pub vmem_issue: u64,
    /// Stall: waitcnt blocking cycles
    pub stall_waitcnt: u64,
    /// Stall: barrier synchronization cycles
    pub stall_barrier: u64,
    /// Stall: delay_alu hints
    pub stall_delay_alu: u64,
    /// Stall: RAW data dependency hazards (NEW)
    pub stall_dep: u64,

    // ── Instruction counts ──
    pub n_wmma: u32,
    pub n_valu: u32,
    pub n_lds_load: u32,
    pub n_lds_store: u32,
    pub n_vmem_load: u32,
    pub n_vmem_store: u32,
    pub n_salu: u32,
    pub n_barrier: u32,
    pub n_waitcnt: u32,
    pub n_delay_alu: u32,
    pub n_total: u32,
}

impl SimResult {
    /// VALU/WMMA pipeline utilization (0.0 - 1.0)
    pub fn valu_utilization(&self) -> f64 {
        if self.total_cycles == 0 { return 0.0; }
        self.valu_busy as f64 / self.total_cycles as f64
    }

    /// Predict TFLOPS for a GEMM given problem dimensions and K-loop cycle count.
    ///
    /// The simulation covers ONE full K-loop iteration (Phase A + Phase B = 2 K-tiles).
    /// Each loop iteration processes 2 × tile_k elements along the K dimension.
    ///
    /// ## Prediction Model
    ///
    /// Three ceilings constrain TFLOPS:
    ///
    /// 1. **Compute ceiling**: VALU/WMMA pipe utilization (the absolute maximum)
    /// 2. **LDS/barrier ceiling**: LDS loads + barriers + RAW deps dominate per-WG time.
    ///    VMEM stalls are mostly hidden by the software pipeline (dual-phase double-buffering).
    ///    WLP (occupancy N) hides remaining LDS stalls partially.
    /// 3. **DRAM ceiling**: 960 GB/s bandwidth limits data delivery to LDS.
    ///
    /// The prediction returns `min(compute_ceiling, lds_ceiling, dram_ceiling)`.
    pub fn predict_tflops(
        &self, m: u32, n: u32, k: u32,
        tile_k: u32,
        hw: &HwParams,
        waves_per_wg: u32,
        occupancy: u32,
        n_wgs: u32,
    ) -> f64 {
        if self.total_cycles == 0 || self.valu_busy == 0 { return 0.0; }

        let k_tiles_total = k / tile_k;
        let loop_iterations = (k_tiles_total + 1) / 2;

        // ── VALU/WMMA compute floor ──
        let iter_compute = self.valu_busy;

        // ── Effective WLP factor ──
        // Three levels of latency hiding work together:
        //   1. Intra-wave: WMMA instructions interleaved between ds_loads (simulated above)
        //   2. Inter-wave: occupancy N → N waves per SIMD can overlap LDS/VMEM stalls
        //   3. Software pipeline: double-buffering hides VMEM across K-iterations (~1.5-2×)
        //
        // The simulation already captures level 1 in total_cycles.
        // Levels 2+3 are modeled as a combined WLP factor.
        // Calibration: 910 cycles / (2 × 1.9) = 239 ≈ back-calculated 237 from 66 TF.
        let sw_pipeline_factor = 1.9;  // empirical: double-buffer pipeline hiding
        let wlp = (occupancy as f64 * sw_pipeline_factor).max(1.0);
        let effective_per_iter = (iter_compute as f64).max(self.total_cycles as f64 / wlp);
        let cycles_per_wg = effective_per_iter * loop_iterations as f64;

        // Chip-level scheduling
        let total_concurrent_waves = hw.n_cus as u64 * occupancy as u64;
        let concurrent_wgs = (total_concurrent_waves / waves_per_wg as u64).max(1);
        let batches = (n_wgs as f64 / concurrent_wgs as f64).ceil();
        let total_cycles_chip = cycles_per_wg * batches;

        let total_time_sec = total_cycles_chip / (hw.clock_ghz * 1e9);
        let total_flops = 2.0 * m as f64 * n as f64 * k as f64;
        let compute_tf = total_flops / total_time_sec / 1e12;

        // ── DRAM ceiling: bandwidth-limited TFLOPS ──
        let bytes_read = 2.0 * (m as f64 * k as f64 + k as f64 * n as f64) * 2.0; // bf16 inputs
        let bytes_write = m as f64 * n as f64 * 4.0; // f32 output
        let total_bytes = bytes_read + bytes_write;
        let dram_bw = 960.0e9; // bytes/sec, RX 7900 XTX
        let dram_time = total_bytes / dram_bw;
        let dram_tf = total_flops / dram_time / 1e12;

        // Return the minimum of compute and DRAM ceilings
        compute_tf.min(dram_tf)
    }

    /// Identify the primary bottleneck.
    pub fn bottleneck(&self) -> &'static str {
        let stall_total = self.stall_waitcnt + self.stall_barrier + self.stall_delay_alu;
        if self.stall_dep > stall_total && self.stall_dep > self.valu_busy {
            "RAW dependency"
        } else if stall_total > self.valu_busy && stall_total > self.lds_issue {
            "stall (waitcnt/barrier)"
        } else if self.valu_busy >= self.lds_issue {
            "VALU/WMMA pipe"
        } else {
            "LDS pipe"
        }
    }

    /// Print a human-readable analysis report.
    pub fn print_report(&self) {
        let total = self.total_cycles;
        let pct = |v: u64| if total > 0 { v as f64 / total as f64 * 100.0 } else { 0.0 };
        let bar_fn = |p: f64| {
            let filled = (p / 10.0).min(10.0) as usize;
            format!("[{}{}]", "█".repeat(filled), "░".repeat(10 - filled))
        };

        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  K-Loop Pipeline Simulation v2 (GFX1100, dep-aware)         ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  Total cycles: {:>6}                                        ║", total);
        eprintln!("╟──────────────────────────────────────────────────────────────╢");
        eprintln!("║  Pipeline Utilization:                                       ║");
        eprintln!("║    VALU/WMMA: {:>6} cyc ({:>5.1}%) {}                 ║",
            self.valu_busy, pct(self.valu_busy), bar_fn(pct(self.valu_busy)));
        eprintln!("║    LDS:      {:>6} cyc ({:>5.1}%) {}                 ║",
            self.lds_issue, pct(self.lds_issue), bar_fn(pct(self.lds_issue)));
        eprintln!("║    VMEM:     {:>6} cyc ({:>5.1}%) {}                 ║",
            self.vmem_issue, pct(self.vmem_issue), bar_fn(pct(self.vmem_issue)));
        eprintln!("╟──────────────────────────────────────────────────────────────╢");
        eprintln!("║  Stalls:                                                     ║");
        eprintln!("║    waitcnt:   {:>6} cyc ({:>5.1}%)                           ║",
            self.stall_waitcnt, pct(self.stall_waitcnt));
        eprintln!("║    barrier:   {:>6} cyc ({:>5.1}%)                           ║",
            self.stall_barrier, pct(self.stall_barrier));
        eprintln!("║    RAW dep:   {:>6} cyc ({:>5.1}%)  ← data dependency        ║",
            self.stall_dep, pct(self.stall_dep));
        eprintln!("║    delay_alu: {:>6} cyc ({:>5.1}%)                           ║",
            self.stall_delay_alu, pct(self.stall_delay_alu));
        eprintln!("╟──────────────────────────────────────────────────────────────╢");
        eprintln!("║  Instructions: {:>4} total                                   ║", self.n_total);
        eprintln!("║    WMMA:{:>3}  VALU:{:>3}  LDS_ld:{:>3}  LDS_st:{:>3}              ║",
            self.n_wmma, self.n_valu, self.n_lds_load, self.n_lds_store);
        eprintln!("║    VMEM_ld:{:>3}  VMEM_st:{:>3}  SALU:{:>3}  barrier:{:>2}           ║",
            self.n_vmem_load, self.n_vmem_store, self.n_salu, self.n_barrier);
        eprintln!("║    waitcnt:{:>3}  delay_alu:{:>3}                               ║",
            self.n_waitcnt, self.n_delay_alu);
        eprintln!("╟──────────────────────────────────────────────────────────────╢");
        eprintln!("║  Bottleneck: {:>35}   ║", self.bottleneck());
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }
}

// ============================================================================
// Simulator Core
// ============================================================================

/// Simulate a sequence of IR Ops through the RDNA3 pipeline model with full
/// data-dependency (RAW hazard) tracking.
pub fn simulate(ops: &[Op], hw: &HwParams) -> SimResult {
    let mut s = PipelineState::new();
    let mut r = SimResult {
        total_cycles: 0,
        valu_busy: 0, lds_issue: 0, vmem_issue: 0,
        stall_waitcnt: 0, stall_barrier: 0, stall_delay_alu: 0, stall_dep: 0,
        n_wmma: 0, n_valu: 0, n_lds_load: 0, n_lds_store: 0,
        n_vmem_load: 0, n_vmem_store: 0, n_salu: 0,
        n_barrier: 0, n_waitcnt: 0, n_delay_alu: 0, n_total: 0,
    };

    for op in ops {
        let lat = insn_latency::op_latency(op);
        r.n_total += 1;

        match lat.class {
            InsnClass::WMMA => {
                s.issue_wmma(hw, op);
                r.n_wmma += 1;
            }
            InsnClass::VALU | InsnClass::CVT | InsnClass::XLANE => {
                s.issue_valu(hw, op);
                r.n_valu += 1;
            }
            InsnClass::VTRANS => {
                s.issue_vtrans(hw, op);
                r.n_valu += 1;
            }
            InsnClass::LDS => {
                match op {
                    Op::DsStoreB16 { .. } | Op::DsStoreB32 { .. } |
                    Op::DsStoreB64 { .. } | Op::DsStoreB128 { .. } |
                    Op::LdsStore { .. } => {
                        s.issue_lds_store(hw, op);
                        r.n_lds_store += 1;
                    }
                    _ => {
                        s.issue_lds_load(hw, op);
                        r.n_lds_load += 1;
                    }
                }
            }
            InsnClass::VMEM => {
                match op {
                    Op::GlobalStore { .. } | Op::BufferStore { .. } => {
                        s.issue_vmem_store(op);
                        r.n_vmem_store += 1;
                    }
                    _ => {
                        s.issue_vmem_load(hw, op);
                        r.n_vmem_load += 1;
                    }
                }
            }
            InsnClass::SALU | InsnClass::SMEM => {
                match op {
                    Op::WaitLgkmcnt(n) => { s.wait_lgkmcnt(*n); r.n_waitcnt += 1; }
                    Op::WaitVmcnt(n) => { s.wait_vmcnt(*n); r.n_waitcnt += 1; }
                    Op::WaitVscnt(_) => { r.n_waitcnt += 1; s.cycle += 1; }
                    _ => { s.issue_salu(); r.n_salu += 1; }
                }
            }
            InsnClass::CTRL => {
                match op {
                    Op::Barrier | Op::SBarrier => { s.barrier(hw); r.n_barrier += 1; }
                    Op::WaitLgkmcnt(n) => { s.wait_lgkmcnt(*n); r.n_waitcnt += 1; }
                    Op::WaitVmcnt(n) => { s.wait_vmcnt(*n); r.n_waitcnt += 1; }
                    Op::WaitVscnt(_) => { r.n_waitcnt += 1; s.cycle += 1; }
                    _ => { s.cycle += 1; r.n_salu += 1; }
                }
            }
        }
    }

    r.total_cycles = s.cycle;
    r.valu_busy = s.valu_busy;
    r.lds_issue = s.lds_issue;
    r.vmem_issue = s.vmem_issue;
    r.stall_waitcnt = s.stall_waitcnt;
    r.stall_barrier = s.stall_barrier;
    r.stall_delay_alu = s.stall_delay_alu;
    r.stall_dep = s.stall_dep;
    r
}

// ============================================================================
// tile_ir Integration
// ============================================================================

/// Extract the FULL K-loop body ops (Phase A + Phase B) from a T0Kernel.
///
/// The tile_ir software-pipelined K-loop has this structure:
/// ```text
///   k_loop_N:                    ← header
///     cmp >= end → epilog_a_N    ← exit check
///     [Phase A: load→buf1, compute buf0, store→buf1]
///     cmp >= end → epilog_b_N    ← mid-loop exit check
///     [Phase B: load→buf0, compute buf1, store→buf0]
///     branch k_loop_N            ← backedge
///   epilog_a_N:                  ← after-loop
/// ```
///
/// Returns ops slice indices [start, end) covering A+B (header to backedge exclusive).
pub fn extract_kloop(ops: &[Op]) -> Option<(usize, usize)> {
    let mut loop_start = None;
    let mut loop_label = String::new();
    for (i, op) in ops.iter().enumerate() {
        if let Op::Label(name) = op {
            if name.contains("k_loop") {
                loop_start = Some(i + 1);
                loop_label = name.clone();
            }
        }
    }
    let start = loop_start?;

    // Find the backedge: Branch(loop_label) after the loop start
    for (i, op) in ops.iter().enumerate().skip(start) {
        if let Op::BranchScc1(target) = op {
            if *target == loop_label {
                return Some((start, i + 1));  // include the backedge
            }
        }
        // Also check unconditional branch (s_branch)
        if let Op::Branch(target) = op {
            if *target == loop_label && i > start + 10 {
                return Some((start, i + 1));
            }
        }
    }

    // Fallback: epilog_a boundary (Phase A only)
    for (i, op) in ops.iter().enumerate().skip(start) {
        if let Op::Label(name) = op {
            if name.contains("epilog_a") {
                return Some((start, i));
            }
        }
    }
    None
}

/// Phase boundaries within the full K-loop body.
#[derive(Clone, Debug)]
pub struct KLoopPhases {
    /// Full loop body: [start, end)
    pub full: (usize, usize),
    /// Phase A boundary: [start, phase_a_end)
    pub phase_a_end: usize,
    /// Phase B boundary: [phase_b_start, end)
    pub phase_b_start: usize,
    /// Number of Phase A ops
    pub n_phase_a: usize,
    /// Number of Phase B ops
    pub n_phase_b: usize,
}

/// Extract K-loop with detailed phase boundaries.
pub fn extract_kloop_phases(ops: &[Op]) -> Option<KLoopPhases> {
    let (start, end) = extract_kloop(ops)?;

    // Find the mid-loop exit branch to epilog_b (separates Phase A from Phase B)
    let mut phase_a_end = end;  // default: no split found
    let mut phase_b_start = end;
    for i in start..end {
        if let Op::BranchScc1(target) = &ops[i] {
            if target.contains("epilog_b") {
                phase_a_end = i + 1;  // Phase A ends after this branch
                phase_b_start = i + 1; // Phase B starts right after
                break;
            }
        }
    }

    Some(KLoopPhases {
        full: (start, end),
        phase_a_end,
        phase_b_start,
        n_phase_a: phase_a_end - start,
        n_phase_b: end - phase_b_start,
    })
}

/// Generate a tile_ir GEMM, extract its K-loop, simulate, return result.
pub fn analyze_tile_gemm(spec: &super::tile_ir::TileGemm) -> Option<SimResult> {
    analyze_tile_gemm_with_params(spec, &HwParams::default())
}

/// Same as `analyze_tile_gemm` but with custom hardware params.
pub fn analyze_tile_gemm_with_params(
    spec: &super::tile_ir::TileGemm,
    hw: &HwParams,
) -> Option<SimResult> {
    let kernel = super::tile_ir::lower_gemm(spec);
    let ops = kernel.ops();
    let (start, end) = extract_kloop(ops)?;
    Some(simulate(&ops[start..end], hw))
}

/// Full analysis: simulate + predict TFLOPS for specific problem dimensions.
pub fn predict_gemm_tflops(
    spec: &super::tile_ir::TileGemm,
    m: u32, n: u32, k: u32,
) -> Option<(SimResult, f64)> {
    predict_gemm_tflops_with_params(spec, m, n, k, &HwParams::default())
}

/// Same but with custom params for parameter sweeps / calibration.
pub fn predict_gemm_tflops_with_params(
    spec: &super::tile_ir::TileGemm,
    m: u32, n: u32, k: u32,
    hw: &HwParams,
) -> Option<(SimResult, f64)> {
    predict_gemm_tflops_ex(spec, m, n, k, hw, None, None)
}

/// Extended prediction with optional overrides for VGPR count and split_k.
///
/// - `actual_vgprs`: If Some, use this VGPR count for occupancy calculation
///   instead of estimating. Obtain from RegAlloc.total_vgprs after compilation.
/// - `split_k_override`: If Some, use this split_k for WG count calculation
///   instead of spec.split_k. The 4096³ benchmark uses split_k=8.
pub fn predict_gemm_tflops_ex(
    spec: &super::tile_ir::TileGemm,
    m: u32, n: u32, k: u32,
    hw: &HwParams,
    actual_vgprs: Option<u32>,
    split_k_override: Option<u32>,
) -> Option<(SimResult, f64)> {
    let result = analyze_tile_gemm_with_params(spec, hw)?;

    // Occupancy: use actual VGPR count if provided, otherwise estimate
    let vgprs = actual_vgprs.unwrap_or_else(|| {
        let n_row = spec.n_row_blocks();
        let n_col = spec.n_col_tiles();
        n_row * n_col * 8 + n_row * 8 + n_col * 8 + 40
    });
    let occupancy = gfx1100_occupancy(vgprs);

    let split_k = split_k_override.unwrap_or(spec.split_k);
    let n_wgs_m = (m + spec.tile_m - 1) / spec.tile_m;
    let n_wgs_n = (n + spec.tile_n - 1) / spec.tile_n;
    let n_wgs = n_wgs_m * n_wgs_n * split_k;

    let waves_per_wg = spec.n_waves();
    let tflops = result.predict_tflops(m, n, k, spec.tile_k, hw, waves_per_wg, occupancy, n_wgs);
    Some((result, tflops))
}

/// GFX1100 (RDNA3, Wave32) occupancy tiers.
///
/// Maps VGPR count to waves per SIMD, matching hardware granularity:
///   ≤64  VGPRs → 16 waves/SIMD (max)
///   ≤96  VGPRs → 10 waves/SIMD
///   ≤128 VGPRs →  8 waves/SIMD
///   ≤192 VGPRs →  4 waves/SIMD
///   ≤256 VGPRs →  2 waves/SIMD
pub fn gfx1100_occupancy(vgprs: u32) -> u32 {
    if vgprs <= 64 { 16 }
    else if vgprs <= 96 { 10 }
    else if vgprs <= 128 { 8 }
    else if vgprs <= 192 { 4 }
    else { 2 }
}

// ============================================================================
// ASM Text Parser
// ============================================================================

/// A parsed assembly instruction (from disassembly text).
#[derive(Clone, Debug)]
pub struct AsmInsn {
    pub mnemonic: String,
    pub class: InsnClass,
    /// For waitcnt: the counter value
    pub wait_count: Option<u8>,
    /// For waitcnt: which counter (lgkmcnt, vmcnt, vscnt)
    pub wait_kind: Option<WaitKind>,
    /// Number of destination VGPRs (for width estimation)
    pub n_dst_regs: u32,
    /// Number of source VGPRs
    pub n_src_regs: u32,
}

/// Which hardware wait counter a s_waitcnt instruction targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitKind {
    Lgkmcnt,
    Vmcnt,
    Vscnt,
}

/// Parse a K-loop section from disassembly text into AsmInsn sequence.
///
/// Input: multi-line ASM text (e.g., from T0_DUMP_ASM=1 output).
/// Expects lines like:
///   326:   v_add_nc_u32 v151, v140, v138
///   349:   v_wmma_f32_16x16x16_bf16 v[8:15], v[200:207], v[216:223], v[8:15]
///
/// Returns the parsed instruction sequence.
pub fn parse_asm_text(text: &str) -> Vec<AsmInsn> {
    let mut insns = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with(';') { continue; }

        // Skip labels (lines ending with ':' without instruction)
        if line.ends_with(':') && !line.contains(' ') { continue; }

        // Strip line numbers like "  326:   "
        let stripped = if let Some(idx) = line.find(':') {
            let prefix = &line[..idx];
            if prefix.trim().chars().all(|c| c.is_ascii_digit()) {
                line[idx + 1..].trim()
            } else {
                line
            }
        } else {
            line
        };
        if stripped.is_empty() { continue; }

        // Skip directives
        if stripped.starts_with('.') { continue; }

        // Extract mnemonic (first word)
        let mnemonic = stripped.split_whitespace().next().unwrap_or("").to_string();
        if mnemonic.is_empty() { continue; }

        let (class, wait_count, wait_kind) = classify_asm_mnemonic(&mnemonic, stripped);

        // Count VGPRs from operands
        let (n_dst, n_src) = count_asm_vregs(stripped, &class);

        insns.push(AsmInsn {
            mnemonic, class, wait_count, wait_kind,
            n_dst_regs: n_dst, n_src_regs: n_src,
        });
    }

    insns
}

/// Classify an ASM mnemonic into pipeline class.
fn classify_asm_mnemonic(mnemonic: &str, full_line: &str) -> (InsnClass, Option<u8>, Option<WaitKind>) {
    // WMMA
    if mnemonic.starts_with("v_wmma") { return (InsnClass::WMMA, None, None); }

    // Waitcnt
    if mnemonic == "s_waitcnt" {
        // Parse: s_waitcnt lgkmcnt(2) or s_waitcnt vmcnt(0)
        if let Some(n) = extract_waitcnt_value(full_line, "lgkmcnt") {
            return (InsnClass::CTRL, Some(n), Some(WaitKind::Lgkmcnt));
        }
        if let Some(n) = extract_waitcnt_value(full_line, "vmcnt") {
            return (InsnClass::CTRL, Some(n), Some(WaitKind::Vmcnt));
        }
        return (InsnClass::CTRL, Some(0), None);
    }
    if mnemonic == "s_waitcnt_vscnt" { return (InsnClass::CTRL, Some(0), Some(WaitKind::Vscnt)); }

    // Barrier
    if mnemonic == "s_barrier" { return (InsnClass::CTRL, None, None); }

    // LDS
    if mnemonic.starts_with("ds_load") || mnemonic.starts_with("ds_store") {
        return (InsnClass::LDS, None, None);
    }

    // VMEM
    if mnemonic.starts_with("buffer_load") || mnemonic.starts_with("global_load") {
        return (InsnClass::VMEM, None, None);
    }
    if mnemonic.starts_with("buffer_store") || mnemonic.starts_with("global_store") {
        return (InsnClass::VMEM, None, None);
    }

    // delay_alu
    if mnemonic == "s_delay_alu" { return (InsnClass::SALU, None, None); }

    // Transcendental
    if mnemonic.starts_with("v_rcp_") || mnemonic.starts_with("v_rsq_") ||
       mnemonic.starts_with("v_exp_") || mnemonic.starts_with("v_log_") ||
       mnemonic.starts_with("v_sin_") || mnemonic.starts_with("v_cos_") ||
       mnemonic.starts_with("v_sqrt_") {
        return (InsnClass::VTRANS, None, None);
    }

    // VALU (everything starting with v_)
    if mnemonic.starts_with("v_") { return (InsnClass::VALU, None, None); }

    // SALU
    if mnemonic.starts_with("s_") { return (InsnClass::SALU, None, None); }

    (InsnClass::SALU, None, None) // default
}

/// Extract counter value from waitcnt line (e.g., "lgkmcnt(2)" → 2).
fn extract_waitcnt_value(line: &str, counter: &str) -> Option<u8> {
    if let Some(pos) = line.find(counter) {
        let rest = &line[pos + counter.len()..];
        if rest.starts_with('(') {
            let end = rest.find(')')?;
            let num_str = &rest[1..end];
            return num_str.trim().parse::<u8>().ok();
        }
    }
    None
}

/// Count destination and source VGPRs from operand text.
fn count_asm_vregs(line: &str, class: &InsnClass) -> (u32, u32) {
    // Count v[N:M] ranges and vN references
    let mut total_vregs = 0u32;
    for part in line.split(|c: char| c == ',' || c == ' ') {
        let p = part.trim();
        if p.starts_with("v[") {
            // v[8:15] → 8 regs
            if let (Some(a), Some(b)) = (p.find('['), p.find(']')) {
                let inner = &p[a+1..b];
                if let Some(colon) = inner.find(':') {
                    let lo: u32 = inner[..colon].parse().unwrap_or(0);
                    let hi: u32 = inner[colon+1..].parse().unwrap_or(lo);
                    total_vregs += hi - lo + 1;
                } else {
                    total_vregs += 1;
                }
            }
        } else if p.starts_with('v') && p.len() > 1 && p.as_bytes()[1].is_ascii_digit() {
            total_vregs += 1;
        }
    }

    // For VALU: first operand is dst, rest are src
    match class {
        InsnClass::WMMA => (8, total_vregs.saturating_sub(8)), // dst=8, rest=src
        InsnClass::VALU | InsnClass::VTRANS => (1, total_vregs.saturating_sub(1)),
        InsnClass::LDS => {
            if line.contains("ds_load") { (total_vregs / 2, 1) }
            else { (0, total_vregs) }
        }
        InsnClass::VMEM => {
            if line.contains("load") { (total_vregs / 2, 1) }
            else { (0, total_vregs) }
        }
        _ => (0, 0),
    }
}

/// Simulate a parsed ASM instruction sequence.
///
/// Since ASM insns don't have VReg identity for dependency tracking,
/// this uses a simplified pipeline model (no RAW tracking, only structural hazards).
pub fn simulate_asm(insns: &[AsmInsn], hw: &HwParams) -> SimResult {
    let mut cycle: u64 = 0;
    let mut valu_ready: u64 = 0;
    let mut lds_inflight: VecDeque<u64> = VecDeque::new();
    let mut vmem_inflight: VecDeque<u64> = VecDeque::new();

    let mut r = SimResult {
        total_cycles: 0,
        valu_busy: 0, lds_issue: 0, vmem_issue: 0,
        stall_waitcnt: 0, stall_barrier: 0, stall_delay_alu: 0, stall_dep: 0,
        n_wmma: 0, n_valu: 0, n_lds_load: 0, n_lds_store: 0,
        n_vmem_load: 0, n_vmem_store: 0, n_salu: 0,
        n_barrier: 0, n_waitcnt: 0, n_delay_alu: 0, n_total: 0,
    };

    for insn in insns {
        r.n_total += 1;

        match insn.class {
            InsnClass::WMMA => {
                if valu_ready > cycle { cycle = valu_ready; }
                r.valu_busy += hw.wmma_issue;
                valu_ready = cycle + hw.wmma_issue;
                cycle += 1;
                r.n_wmma += 1;
            }
            InsnClass::VALU | InsnClass::CVT | InsnClass::XLANE => {
                if valu_ready > cycle { cycle = valu_ready; }
                r.valu_busy += hw.valu_issue;
                valu_ready = cycle + hw.valu_issue;
                cycle += 1;
                r.n_valu += 1;
            }
            InsnClass::VTRANS => {
                if valu_ready > cycle { cycle = valu_ready; }
                r.valu_busy += 4;
                valu_ready = cycle + 4;
                cycle += 1;
                r.n_valu += 1;
            }
            InsnClass::LDS => {
                let is_store = insn.mnemonic.contains("store");
                lds_inflight.push_back(cycle + hw.lds_latency);
                r.lds_issue += 1;
                cycle += 1;
                if is_store { r.n_lds_store += 1; } else { r.n_lds_load += 1; }
            }
            InsnClass::VMEM => {
                let is_store = insn.mnemonic.contains("store");
                if !is_store { vmem_inflight.push_back(cycle + hw.vmem_latency); }
                r.vmem_issue += 1;
                cycle += 1;
                if is_store { r.n_vmem_store += 1; } else { r.n_vmem_load += 1; }
            }
            InsnClass::CTRL => {
                if insn.mnemonic == "s_barrier" {
                    r.stall_barrier += hw.barrier_cost;
                    cycle += hw.barrier_cost;
                    r.n_barrier += 1;
                } else if insn.mnemonic.starts_with("s_waitcnt") {
                    let n = insn.wait_count.unwrap_or(0) as usize;
                    match insn.wait_kind {
                        Some(WaitKind::Lgkmcnt) => {
                            while lds_inflight.len() > n {
                                if let Some(&earliest) = lds_inflight.front() {
                                    let stall = earliest.saturating_sub(cycle);
                                    r.stall_waitcnt += stall;
                                    if earliest > cycle { cycle = earliest; }
                                    lds_inflight.pop_front();
                                } else { break; }
                            }
                        }
                        Some(WaitKind::Vmcnt) => {
                            while vmem_inflight.len() > n {
                                if let Some(&earliest) = vmem_inflight.front() {
                                    let stall = earliest.saturating_sub(cycle);
                                    r.stall_waitcnt += stall;
                                    if earliest > cycle { cycle = earliest; }
                                    vmem_inflight.pop_front();
                                } else { break; }
                            }
                        }
                        Some(WaitKind::Vscnt) | None => {
                            // vscnt: just advance 1 cycle (store completion, no stall)
                            cycle += 1;
                        }
                    }
                    r.n_waitcnt += 1;
                } else {
                    cycle += 1;
                    r.n_salu += 1;
                }
            }
            InsnClass::SALU | InsnClass::SMEM => {
                if insn.mnemonic == "s_delay_alu" {
                    r.stall_delay_alu += hw.delay_alu_cost;
                    cycle += hw.delay_alu_cost;
                    r.n_delay_alu += 1;
                } else {
                    cycle += 1;
                    r.n_salu += 1;
                }
            }
        }
    }

    r.total_cycles = cycle;
    r
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ir::*;

    #[test]
    fn test_empty_simulation() {
        let hw = HwParams::default();
        let result = simulate(&[], &hw);
        assert_eq!(result.total_cycles, 0);
        assert_eq!(result.n_total, 0);
    }

    #[test]
    fn test_pure_wmma_simulation() {
        let hw = HwParams::default();
        let ops: Vec<Op> = (0..16).map(|i| Op::Wmma {
            dst: VReg(i * 8),
            a: VReg(128),
            b: VReg(136),
            c: VReg(i * 8),
            format: WmmaFormat::BF16_F32,
        }).collect();
        let result = simulate(&ops, &hw);
        assert_eq!(result.n_wmma, 16);
        assert_eq!(result.valu_busy, 64);
        assert!(result.total_cycles >= 16, "got {} cycles", result.total_cycles);
    }

    #[test]
    fn test_raw_dependency_stall() {
        // v1 = load LDS → needs 20 cycles
        // v2 = v1 + 1.0 → must wait for v1
        let hw = HwParams::default();
        let ops = vec![
            Op::DsLoadB32 { dst: VReg(1), vaddr: VReg(100), offset: 0 },
            Op::WaitLgkmcnt(0),
            Op::VAddF32 { dst: VReg(2), src0: Operand::VReg(VReg(1)), src1: Operand::InlineFloat(1.0) },
        ];
        let result = simulate(&ops, &hw);
        assert!(result.stall_waitcnt > 0 || result.stall_dep > 0,
            "should stall on RAW dep: waitcnt={}, dep={}", result.stall_waitcnt, result.stall_dep);
    }

    #[test]
    fn test_waitcnt_blocking() {
        let hw = HwParams::default();
        let ops = vec![
            Op::DsLoadB128 { dst: VReg(0), vaddr: VReg(100), offset: 0 },
            Op::DsLoadB128 { dst: VReg(4), vaddr: VReg(100), offset: 16 },
            Op::DsLoadB128 { dst: VReg(8), vaddr: VReg(100), offset: 32 },
            Op::DsLoadB128 { dst: VReg(12), vaddr: VReg(100), offset: 48 },
            Op::WaitLgkmcnt(0),
        ];
        let result = simulate(&ops, &hw);
        assert_eq!(result.n_lds_load, 4);
        assert_eq!(result.n_waitcnt, 1);
        assert!(result.stall_waitcnt > 0, "waitcnt should cause stall");
    }

    #[test]
    fn test_barrier_cost_tunable() {
        let mut hw = HwParams::default();
        hw.barrier_cost = 50; // custom tuning
        let ops = vec![Op::SBarrier, Op::SBarrier];
        let result = simulate(&ops, &hw);
        assert_eq!(result.n_barrier, 2);
        assert_eq!(result.stall_barrier, 100); // 2 × 50
    }

    /// Integration test: analyze the actual tile_ir 128×128×k16 K-loop.
    #[test]
    fn test_tile_ir_kloop_analysis() {
        let spec = super::super::tile_ir::TileGemm::tile_128x128_k16();
        let kernel = super::super::tile_ir::lower_gemm(&spec);
        let ops = kernel.ops();

        // Phase boundary analysis
        if let Some(phases) = extract_kloop_phases(ops) {
            eprintln!("\n  ── K-Loop Phase Boundary Analysis ──");
            eprintln!("  Full loop body: ops[{}..{}] ({} instructions)",
                phases.full.0, phases.full.1, phases.full.1 - phases.full.0);
            eprintln!("  Phase A: {} ops (compute buf0, store→buf1)", phases.n_phase_a);
            eprintln!("  Phase B: {} ops (compute buf1, store→buf0)", phases.n_phase_b);
        }

        if let Some(result) = analyze_tile_gemm(&spec) {
            result.print_report();
            // With full A+B extraction, expect 64 WMMA (32 per phase)
            assert!(result.n_wmma >= 32, "K-loop should contain at least 32 WMMA instructions, got {}", result.n_wmma);
            assert!(result.n_lds_load > 0, "K-loop should contain LDS loads");
            assert!(result.total_cycles > 0, "Total cycles should be positive");
            eprintln!("  RAW dep stalls: {} cycles", result.stall_dep);
        } else {
            panic!("Failed to extract K-loop from tile_ir kernel");
        }
    }

    /// Calibration test: compare predicted TFLOPS vs measured.
    #[test]
    fn test_prediction_vs_measured() {
        let spec = super::super::tile_ir::TileGemm::tile_128x128_k16();
        let hw = HwParams::default();
        let m = 4096u32; let n = 4096u32; let k = 4096u32;

        // Actual values from regalloc and benchmark config:
        let actual_vgprs = 228u32;  // from [T0 SSA RegAlloc] output
        let bench_split_k = 8u32;   // from test_tile_ir_benchmark config

        // Predict with corrected parameters
        let (result, predicted_tf) = predict_gemm_tflops_ex(
            &spec, m, n, k, &hw,
            Some(actual_vgprs), Some(bench_split_k),
        ).expect("prediction");

        // Also compute with default (wrong) params for comparison
        let (_, naive_tf) = predict_gemm_tflops(&spec, m, n, k)
            .expect("naive prediction");

        let occupancy = gfx1100_occupancy(actual_vgprs);
        let waves_per_wg = spec.n_waves();
        let n_wgs_m = (m + spec.tile_m - 1) / spec.tile_m;
        let n_wgs_n = (n + spec.tile_n - 1) / spec.tile_n;
        let n_wgs = n_wgs_m * n_wgs_n * bench_split_k;
        let k_tiles = k / spec.tile_k;
        let loop_iters = (k_tiles + 1) / 2;

        eprintln!("\n═══ Calibration (v3 — corrected VGPR + split_k) ═══");
        eprintln!("  K-loop sim:  {} cycles, {} WMMA, {} LDS_ld, {} VMEM_ld",
            result.total_cycles, result.n_wmma, result.n_lds_load, result.n_vmem_load);
        eprintln!("    valu_busy={} lds_issue={} waitcnt={} barrier={} dep={}",
            result.valu_busy, result.lds_issue, result.stall_waitcnt,
            result.stall_barrier, result.stall_dep);
        eprintln!("  Config: {}×{}×{}, 128×128×k16, split_k={}", m, n, k, bench_split_k);
        eprintln!("  VGPRs: {} → {} waves/SIMD, {} waves_per_wg", actual_vgprs, occupancy, waves_per_wg);
        eprintln!("  WGs: {}×{}×{} = {}, k_tiles={}, loop_iters={}",
            n_wgs_m, n_wgs_n, bench_split_k, n_wgs, k_tiles, loop_iters);

        // Show WLP-corrected math
        let compute_part = (result.valu_busy + result.lds_issue + result.stall_barrier + result.stall_dep)
            * loop_iters as u64;
        let stall_part = result.stall_waitcnt * loop_iters as u64;
        let cycles_per_wg = compute_part as f64 + stall_part as f64 / occupancy as f64;
        let conc_waves = hw.n_cus as u64 * occupancy as u64;
        let conc_wgs = (conc_waves / waves_per_wg as u64).max(1);
        let batches = (n_wgs as f64 / conc_wgs as f64).ceil();
        eprintln!("  ── Prediction Math ──");
        eprintln!("  compute_part: {:.0} ({} × {})", compute_part,
            result.valu_busy + result.lds_issue + result.stall_barrier + result.stall_dep, loop_iters);
        eprintln!("  stall_part:   {} / {} occ = {:.0}", stall_part, occupancy, stall_part as f64 / occupancy as f64);
        eprintln!("  cycles/wg:    {:.0}", cycles_per_wg);
        eprintln!("  concurrent:   {} wgs (96 CU × {} occ / {} waves)", conc_wgs, occupancy, waves_per_wg);
        eprintln!("  batches:      {:.0} ({} / {})", batches, n_wgs, conc_wgs);

        let measured_tf = 66.0;
        let error = ((predicted_tf - measured_tf) / measured_tf * 100.0).abs();
        eprintln!("  ═══════════════════════════");
        eprintln!("  Corrected:  {:.1} TF (error: {:.1}%)", predicted_tf, error);
        eprintln!("  Naive:      {:.1} TF (was 14× off due to wrong VGPR/split_k)", naive_tf);
        eprintln!("  Measured:   {:.1} TF", measured_tf);

        result.print_report();
        assert!(predicted_tf > 0.0, "Prediction should be positive");
    }

    /// Test ASM text parser with a realistic K-loop snippet.
    #[test]
    fn test_asm_parser() {
        let asm = r#"
323: .Lk_loop_3:
326:   v_add_nc_u32 v151, v140, v138
327:   buffer_load_b128 v[152:155], v151, s[40:43], 0 offen
340:   ds_load_b128 v[200:203], v144 offset:0
349:   v_wmma_f32_16x16x16_bf16 v[8:15], v[200:207], v[216:223], v[8:15]
357:   ds_store_b128 v151, v[172:175] offset:0
393:   s_barrier
394:   v_add_nc_u32 v138, v138, 32
        "#;
        let insns = parse_asm_text(asm);
        assert!(insns.len() >= 5, "should parse at least 5 insns, got {}", insns.len());

        // Check classifications
        let wmma_count = insns.iter().filter(|i| i.class == InsnClass::WMMA).count();
        let valu_count = insns.iter().filter(|i| i.class == InsnClass::VALU).count();
        let lds_count = insns.iter().filter(|i| i.class == InsnClass::LDS).count();
        let vmem_count = insns.iter().filter(|i| i.class == InsnClass::VMEM).count();

        assert_eq!(wmma_count, 1, "1 WMMA");
        assert_eq!(valu_count, 2, "2 VALU (v_add)");
        assert_eq!(lds_count, 2, "2 LDS (load + store)");
        assert_eq!(vmem_count, 1, "1 VMEM (buffer_load)");

        // Simulate ASM insns
        let hw = HwParams::default();
        let result = simulate_asm(&insns, &hw);
        assert!(result.total_cycles > 0);
        assert_eq!(result.n_wmma, 1);
    }

    /// Test parameter sweep: barrier cost sensitivity.
    #[test]
    fn test_barrier_cost_sensitivity() {
        let spec = super::super::tile_ir::TileGemm::tile_128x128_k16();
        eprintln!("\n═══ Barrier Cost Sensitivity ═══");
        for barrier in [10, 20, 30, 50, 100] {
            let mut hw = HwParams::default();
            hw.barrier_cost = barrier;
            if let Some(result) = analyze_tile_gemm_with_params(&spec, &hw) {
                eprintln!("  barrier={:>3}: total={:>5} cyc, waitcnt={:>5}, barrier_stall={:>4}, dep={:>4}",
                    barrier, result.total_cycles, result.stall_waitcnt, result.stall_barrier, result.stall_dep);
            }
        }
    }

    /// Sweep LDS latency to find the value that best matches measured TFLOPS.
    #[test]
    fn test_lds_latency_sweep() {
        let spec = super::super::tile_ir::TileGemm::tile_128x128_k16();
        let actual_vgprs = 228u32;
        let bench_split_k = 8u32;
        let measured_tf = 66.0;

        eprintln!("\n═══ LDS Latency Sweep (target: {:.0} TF) ═══", measured_tf);
        eprintln!("  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}",
            "lds_lat", "total", "waitcnt", "valu%", "pred_TF", "error%");

        let mut best_lds = 20u64;
        let mut best_err = f64::MAX;

        for lds_lat in [5, 8, 10, 12, 15, 18, 20, 25, 30] {
            let mut hw = HwParams::default();
            hw.lds_latency = lds_lat;

            if let Some((result, tf)) = predict_gemm_tflops_ex(
                &spec, 4096, 4096, 4096, &hw,
                Some(actual_vgprs), Some(bench_split_k),
            ) {
                let err = ((tf - measured_tf) / measured_tf * 100.0).abs();
                let valu_pct = result.valu_utilization() * 100.0;
                eprintln!("  {:>6}  {:>6}  {:>6}  {:>5.1}%  {:>6.1}  {:>5.1}%{}",
                    lds_lat, result.total_cycles, result.stall_waitcnt,
                    valu_pct, tf, err,
                    if err < best_err { " ◄" } else { "" });
                if err < best_err {
                    best_err = err;
                    best_lds = lds_lat;
                }
            }
        }
        eprintln!("  Best: lds_latency={} (error {:.1}%)", best_lds, best_err);
    }

    /// DRAM roofline analysis to validate theoretical maximum TFLOPS.
    #[test]
    fn test_dram_roofline() {
        let m = 4096u32; let n = 4096u32; let k = 4096u32;
        let tile_m = 128u32; let tile_n = 128u32; let tile_k = 16u32;
        let hw = HwParams::default();

        // Arithmetic intensity for GEMM: 2*M*N*K flops / (2*(M*K + K*N + M*N)) bytes
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let bytes_read = 2.0 * (m as f64 * k as f64 + k as f64 * n as f64) * 2.0; // bf16
        let bytes_write = m as f64 * n as f64 * 4.0; // f32
        let total_bytes = bytes_read + bytes_write;
        let ai = flops / total_bytes;

        // RX 7900 XTX: 960 GB/s DRAM BW, 122 TFLOPS peak BF16
        let dram_bw_gbps = 960.0;
        let peak_tflops = 122.0; // BF16 WMMA peak

        // Roofline TFLOPS = min(peak, AI × BW)
        let mem_bound_tf = ai * dram_bw_gbps / 1e3; // convert to TFLOPS
        let roofline_tf = mem_bound_tf.min(peak_tflops);

        // Tile-level: bytes loaded per K-tile for one WG
        let tile_bytes_per_k = (tile_m * tile_k + tile_n * tile_k) * 2; // bf16
        let k_tiles = k / tile_k;
        let flops_per_wg = 2.0 * tile_m as f64 * tile_n as f64 * k as f64;
        let bytes_per_wg = tile_bytes_per_k as f64 * k_tiles as f64;
        let tile_ai = flops_per_wg / bytes_per_wg;

        eprintln!("\n═══ DRAM Roofline Analysis ═══");
        eprintln!("  Problem: {}×{}×{} (bf16)", m, n, k);
        eprintln!("  Total FLOPs:  {:.1}G", flops / 1e9);
        eprintln!("  Total bytes:  {:.1}MB (read) + {:.1}MB (write)",
            bytes_read / 1e6, bytes_write / 1e6);
        eprintln!("  Arithmetic intensity: {:.1} FLOP/byte (global)", ai);
        eprintln!("  Tile-level AI:        {:.1} FLOP/byte", tile_ai);
        eprintln!("  ─────────────────────────");
        eprintln!("  DRAM BW:       {:.0} GB/s", dram_bw_gbps);
        eprintln!("  Peak compute:  {:.0} TF (BF16 WMMA)", peak_tflops);
        eprintln!("  Mem-bound:     {:.1} TF", mem_bound_tf);
        eprintln!("  Roofline:      {:.1} TF", roofline_tf);
        eprintln!("  Measured:      66.0 TF ({:.1}% of peak)", 66.0 / peak_tflops * 100.0);
        eprintln!("  Bottleneck:    {}", if ai < peak_tflops * 1e3 / dram_bw_gbps {
            "MEMORY BOUND" } else { "COMPUTE BOUND" });
    }

    /// Comprehensive K-loop bottleneck diagnostic.
    ///
    /// Separates Phase A and Phase B, shows per-waitcnt stall breakdown.
    #[test]
    fn test_bottleneck_diagnostic() {
        use super::super::tile_ir;

        let hw = HwParams::default();

        // ── tile_ir 128×128 k16 ──
        let spec = tile_ir::TileGemm::tile_128x128_k16();
        let kernel = tile_ir::lower_gemm(&spec);
        let ops = kernel.ops();

        eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  K-Loop Bottleneck Diagnostic — {}", spec.name());
        eprintln!("╠══════════════════════════════════════════════════════════════╣");

        // Full K-loop simulation
        if let Some(phases) = extract_kloop_phases(ops) {
            let (start, end) = phases.full;
            let full_result = simulate(&ops[start..end], &hw);

            eprintln!("║  Full K-loop: [{}, {}) = {} ops", start, end, end - start);
            eprintln!("║  Total: {:>5} cyc | WMMA: {:>3} | LDS_ld: {:>3} | LDS_st: {:>3} | VMEM: {:>3}",
                full_result.total_cycles, full_result.n_wmma, full_result.n_lds_load,
                full_result.n_lds_store, full_result.n_vmem_load);
            let total = full_result.total_cycles;
            eprintln!("║  VALU busy:    {:>5} ({:>5.1}%)", full_result.valu_busy,
                full_result.valu_busy as f64 / total as f64 * 100.0);
            eprintln!("║  waitcnt:      {:>5} ({:>5.1}%)", full_result.stall_waitcnt,
                full_result.stall_waitcnt as f64 / total as f64 * 100.0);
            eprintln!("║  barrier:      {:>5} ({:>5.1}%)", full_result.stall_barrier,
                full_result.stall_barrier as f64 / total as f64 * 100.0);
            eprintln!("║  WMMA util:    {:>5.1}%",
                full_result.valu_busy as f64 / total as f64 * 100.0);

            // Phase A only
            let phase_a_ops = &ops[start..phases.phase_a_end];
            let phase_a_result = simulate(phase_a_ops, &hw);
            eprintln!("║");
            eprintln!("║  Phase A: [{}, {}) = {} ops", start, phases.phase_a_end,
                phases.phase_a_end - start);
            let at = phase_a_result.total_cycles;
            eprintln!("║    Total: {:>5} cyc | WMMA: {:>3} | LDS_ld: {:>3} | LDS_st: {:>3} | VMEM: {:>3}",
                at, phase_a_result.n_wmma, phase_a_result.n_lds_load,
                phase_a_result.n_lds_store, phase_a_result.n_vmem_load);
            eprintln!("║    VALU busy:  {:>5} ({:>5.1}%)", phase_a_result.valu_busy,
                phase_a_result.valu_busy as f64 / at as f64 * 100.0);
            eprintln!("║    waitcnt:    {:>5} ({:>5.1}%)", phase_a_result.stall_waitcnt,
                phase_a_result.stall_waitcnt as f64 / at as f64 * 100.0);
            eprintln!("║    barrier:    {:>5} ({:>5.1}%)", phase_a_result.stall_barrier,
                phase_a_result.stall_barrier as f64 / at as f64 * 100.0);

            // Phase B only
            if phases.phase_b_start < end {
                let phase_b_ops = &ops[phases.phase_b_start..end];
                let phase_b_result = simulate(phase_b_ops, &hw);
                let bt = phase_b_result.total_cycles;
                eprintln!("║");
                eprintln!("║  Phase B: [{}, {}) = {} ops", phases.phase_b_start, end,
                    end - phases.phase_b_start);
                eprintln!("║    Total: {:>5} cyc | WMMA: {:>3} | LDS_ld: {:>3} | LDS_st: {:>3} | VMEM: {:>3}",
                    bt, phase_b_result.n_wmma, phase_b_result.n_lds_load,
                    phase_b_result.n_lds_store, phase_b_result.n_vmem_load);
                eprintln!("║    VALU busy:  {:>5} ({:>5.1}%)", phase_b_result.valu_busy,
                    phase_b_result.valu_busy as f64 / bt as f64 * 100.0);
                eprintln!("║    waitcnt:    {:>5} ({:>5.1}%)", phase_b_result.stall_waitcnt,
                    phase_b_result.stall_waitcnt as f64 / bt as f64 * 100.0);
                eprintln!("║    barrier:    {:>5} ({:>5.1}%)", phase_b_result.stall_barrier,
                    phase_b_result.stall_barrier as f64 / bt as f64 * 100.0);
            }

            // ── Per-waitcnt analysis: trace every WaitLgkmcnt and WaitVmcnt ──
            eprintln!("║");
            eprintln!("╠══════════════════════════════════════════════════════════════╣");
            eprintln!("║  Per-waitcnt stall trace (inside K-loop)                    ║");
            eprintln!("╠══════════════════════════════════════════════════════════════╣");

            // Run a traced simulation: accumulate counters and report at each waitcnt
            let mut cycle: u64 = 0;
            let mut lds_inflight: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
            let mut vmem_inflight: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
            let mut valu_busy: u64 = 0;
            let mut waitcnt_idx = 0u32;
            let mut lds_issued = 0u32;
            let mut vmem_issued = 0u32;
            let mut ds_store_issued = 0u32;

            for op in &ops[start..end] {
                match op {
                    Op::DsLoadB128 { .. } | Op::DsLoadB64 { .. } | Op::DsLoadB32 { .. } |
                    Op::LdsLoad { .. } => {
                        lds_inflight.push_back(cycle + hw.lds_latency);
                        lds_issued += 1;
                        cycle += 1;
                    }
                    Op::DsStoreB128 { .. } | Op::DsStoreB64 { .. } | Op::DsStoreB32 { .. } |
                    Op::DsStoreB16 { .. } => {
                        lds_inflight.push_back(cycle + hw.lds_latency);
                        ds_store_issued += 1;
                        cycle += 1;
                    }
                    Op::BufferLoad { .. } | Op::GlobalLoad { .. } => {
                        vmem_inflight.push_back(cycle + hw.vmem_latency);
                        vmem_issued += 1;
                        cycle += 1;
                    }
                    Op::Wmma { .. } => {
                        cycle += 1;
                        valu_busy += 4; // 4 VALU pipe cycles
                    }
                    Op::WaitLgkmcnt(n) => {
                        let n = *n as usize;
                        let before = cycle;
                        while lds_inflight.len() > n {
                            if let Some(&earliest) = lds_inflight.front() {
                                if earliest > cycle { cycle = earliest; }
                                lds_inflight.pop_front();
                            } else { break; }
                        }
                        let stall = cycle - before;
                        if stall > 0 {
                            eprintln!("║  [{:>3}] wait_lgkmcnt({}) stall={:>4} cyc  (lds_pending={}, ds_store={})",
                                waitcnt_idx, n, stall,
                                lds_issued, ds_store_issued);
                        }
                        waitcnt_idx += 1;
                    }
                    Op::WaitVmcnt(n) => {
                        let n = *n as usize;
                        let before = cycle;
                        while vmem_inflight.len() > n {
                            if let Some(&earliest) = vmem_inflight.front() {
                                if earliest > cycle { cycle = earliest; }
                                vmem_inflight.pop_front();
                            } else { break; }
                        }
                        let stall = cycle - before;
                        if stall > 0 {
                            eprintln!("║  [{:>3}] wait_vmcnt({})  stall={:>4} cyc  (vmem_pending={})",
                                waitcnt_idx, n, stall, vmem_issued);
                        }
                        waitcnt_idx += 1;
                    }
                    Op::SBarrier => {
                        cycle += hw.barrier_cost;
                    }
                    _ => {
                        cycle += 1;
                    }
                }
            }
            eprintln!("║  Total traced cycle: {}", cycle);
        }

        // ── Also test 128×64 for comparison ──
        eprintln!("║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        let spec64 = tile_ir::TileGemm::tile_128x64_k16();
        let k64 = tile_ir::lower_gemm(&spec64);
        let ops64 = k64.ops();
        if let Some(r64) = analyze_tile_gemm_with_params(&spec64, &hw) {
            let t = r64.total_cycles;
            eprintln!("║  128×64 k16 comparison:");
            eprintln!("║    Total: {:>5} | WMMA: {:>3} | waitcnt: {:>5} ({:>5.1}%) | barrier: {:>5}",
                t, r64.n_wmma, r64.stall_waitcnt,
                r64.stall_waitcnt as f64 / t as f64 * 100.0, r64.stall_barrier);
        }

        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }
}

