use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate};

fn main() {
    // Generate two GEMM configs and compare their assembly
    let cfg_64 = GemmConfig::tile_64x64_k16();
    let gen_64 = generate(&cfg_64);
    let asm_64 = gen_64.to_assembly(Target::GFX1100).unwrap();

    let cfg_128 = GemmConfig::tile_128x64_k16();
    let gen_128 = generate(&cfg_128);
    let asm_128 = gen_128.to_assembly(Target::GFX1100).unwrap();

    let lines_64: Vec<&str> = asm_64.lines().collect();
    let lines_128: Vec<&str> = asm_128.lines().collect();

    eprintln!("64x64 tile: {} lines", lines_64.len());
    eprintln!("128x64 tile: {} lines", lines_128.len());

    // Dump first 80 lines of 64x64 tile
    eprintln!("\n=== 64x64 tile ASM (first 80 lines) ===");
    for (i, line) in lines_64.iter().enumerate().take(80) {
        eprintln!("{:3} {}", i, line);
    }
}
