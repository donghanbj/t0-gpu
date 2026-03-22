//! Instruction mix analysis for T0 GEMM kernels
//! Dumps assembly and counts instruction categories
//! Run: cargo run --example analyze_gemm_isa --release

use t0_gpu::t0::ir::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate};

fn analyze(name: &str, cfg: &GemmConfig) {
    let kernel_ir = generate(cfg);
    let asm = kernel_ir.to_assembly(Target::GFX1100).unwrap();
    
    let mut n_wmma = 0u32;
    let mut n_ds_load = 0u32;
    let mut n_ds_store = 0u32;
    let mut n_global_load = 0u32;
    let mut n_global_store = 0u32;
    let mut n_valu = 0u32;
    let mut n_salu = 0u32;
    let mut n_wait = 0u32;
    let mut n_barrier = 0u32;
    let mut n_branch = 0u32;
    let mut n_total = 0u32;
    
    for line in asm.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('.') || line.starts_with("//") || line.ends_with(':') {
            continue; // directive, comment, or label
        }
        n_total += 1;
        if line.starts_with("v_wmma") {
            n_wmma += 1;
        } else if line.starts_with("ds_load") {
            n_ds_load += 1;
        } else if line.starts_with("ds_store") {
            n_ds_store += 1;
        } else if line.starts_with("global_load") {
            n_global_load += 1;
        } else if line.starts_with("global_store") {
            n_global_store += 1;
        } else if line.starts_with("s_waitcnt") {
            n_wait += 1;
        } else if line.starts_with("s_barrier") {
            n_barrier += 1;
        } else if line.starts_with("s_cbranch") || line.starts_with("s_branch") {
            n_branch += 1;
        } else if line.starts_with("s_") {
            n_salu += 1;
        } else if line.starts_with("v_") {
            n_valu += 1;
        }
    }
    
    eprintln!("\n── {} ──", name);
    eprintln!("  Total instructions: {}", n_total);
    eprintln!("  WMMA:         {:>4} ({:>5.1}%)", n_wmma, n_wmma as f64 / n_total as f64 * 100.0);
    eprintln!("  VALU:         {:>4} ({:>5.1}%)", n_valu, n_valu as f64 / n_total as f64 * 100.0);
    eprintln!("  SALU:         {:>4} ({:>5.1}%)", n_salu, n_salu as f64 / n_total as f64 * 100.0);
    eprintln!("  DS load:      {:>4} ({:>5.1}%)", n_ds_load, n_ds_load as f64 / n_total as f64 * 100.0);
    eprintln!("  DS store:     {:>4} ({:>5.1}%)", n_ds_store, n_ds_store as f64 / n_total as f64 * 100.0);
    eprintln!("  Global load:  {:>4} ({:>5.1}%)", n_global_load, n_global_load as f64 / n_total as f64 * 100.0);
    eprintln!("  Global store: {:>4} ({:>5.1}%)", n_global_store, n_global_store as f64 / n_total as f64 * 100.0);
    eprintln!("  Wait:         {:>4} ({:>5.1}%)", n_wait, n_wait as f64 / n_total as f64 * 100.0);
    eprintln!("  Barrier:      {:>4} ({:>5.1}%)", n_barrier, n_barrier as f64 / n_total as f64 * 100.0);
    eprintln!("  Branch:       {:>4} ({:>5.1}%)", n_branch, n_branch as f64 / n_total as f64 * 100.0);
    
    // Also dump the loop body only (between ggen_loop and ggen_ea labels)
    let mut in_loop = false;
    let mut loop_wmma = 0u32;
    let mut loop_total = 0u32;
    for line in asm.lines() {
        let line = line.trim();
        if line == "ggen_loop_0:" { in_loop = true; continue; }
        if line == "ggen_ea_0:" { in_loop = false; continue; }
        if !in_loop { continue; }
        if line.is_empty() || line.starts_with('.') || line.starts_with("//") || line.ends_with(':') {
            continue;
        }
        loop_total += 1;
        if line.starts_with("v_wmma") { loop_wmma += 1; }
    }
    if loop_total > 0 {
        eprintln!("  ── K-loop body ──");
        eprintln!("  Loop instructions: {}", loop_total);
        eprintln!("  Loop WMMA:     {:>4} ({:>5.1}%)", loop_wmma, loop_wmma as f64 / loop_total as f64 * 100.0);
        eprintln!("  Loop overhead: {:>4} ({:>5.1}%)", loop_total - loop_wmma, (loop_total - loop_wmma) as f64 / loop_total as f64 * 100.0);
    }
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM Instruction Mix Analysis                          ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    
    analyze("64×64_k16", &GemmConfig::tile_64x64_k16());
    analyze("64×64_k32", &GemmConfig::tile_64x64_k32());
    analyze("128×64_k32", &GemmConfig::tile_128x64_k32());
    analyze("32×128_k16", &GemmConfig::tile_32x128_k16());
    analyze("64×64_k64", &GemmConfig::tile_64x64_k64());
}
