use t0_gpu::t0::{GFX1100Schedule, Schedule, Target};
use t0_gpu::t0::math;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate};

fn main() {
    let sched = GFX1100Schedule {};
    let hw = math::matmul_64x64_lds_db(&sched);
    let hw_asm = hw.to_assembly(Target::GFX1100).unwrap();

    let cfg = GemmConfig::tile_64x64_k16();
    let gen = generate(&cfg);
    let gen_asm = gen.to_assembly(Target::GFX1100).unwrap();

    let hw_lines: Vec<&str> = hw_asm.lines().collect();
    let gen_lines: Vec<&str> = gen_asm.lines().collect();

    // Skip first 5 lines (header/label differences)
    // Compare instruction-by-instruction, ignoring register numbers
    let skip = 5;
    let max = hw_lines.len().min(gen_lines.len());
    let mut diffs = 0;
    for i in skip..max {
        let hw_op = hw_lines[i].split_whitespace().next().unwrap_or("");
        let gen_op = gen_lines[i].split_whitespace().next().unwrap_or("");
        if hw_op != gen_op {
            eprintln!("DIFF@{}: HW=[{}]  GEN=[{}]", i, hw_lines[i].trim(), gen_lines[i].trim());
            diffs += 1;
            if diffs > 20 { break; }
        }
    }
    if diffs == 0 {
        eprintln!("No opcode differences found in first {} lines!", max);
    }
    eprintln!("\nTotal: HW={} lines, GEN={} lines, diffs={}", hw_lines.len(), gen_lines.len(), diffs);

    // Dump lines 5-80 side by side
    eprintln!("\n=== SIDE BY SIDE (lines 5-80) ===");
    for i in 5..80.min(max) {
        let hw_s = hw_lines[i].trim();
        let gen_s = gen_lines[i].trim();
        let marker = if hw_s == gen_s { " " } else { "*" };
        eprintln!("{}{:3} HW: {:60} | GEN: {}", marker, i, hw_s, gen_s);
    }
}
