//! Benchmark all T0 GEMM variants to find the fastest
//!
//! Run: cargo run --example bench_gemm_variants --features rocm --release

use t0_gpu::t0::{GFX1100Schedule, Schedule, Target};
use t0_gpu::t0::math;

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM Variant Benchmark — RX 7900 XTX                   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    let sched = GFX1100Schedule {};

    // Compile all variants
    let variants: Vec<(&str, _)> = vec![
        ("matmul (basic)", math::matmul(&sched)),
        ("matmul_lds_db (LDS+DB)", math::matmul_lds_db(&sched)),
        ("matmul_direct (zero-LDS)", math::matmul_direct(&sched)),
    ];

    #[cfg(feature = "rocm")]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        let test_sizes: Vec<(u32, u32, u32)> = vec![
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (128, 1024, 4096),
        ];

        for (name, kernel_ir) in &variants {
            let elf = kernel_ir.compile(Target::GFX1100)?;

            let lds_size = if name.contains("lds_db") { 6144 } else { 0 };
            let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
                workgroup_size: [64, 1, 1],
                lds_size,
            })?;

            eprintln!("\n── {} ({} bytes) ──", name, elf.len());
            eprintln!("{:>6} {:>6} {:>6} | {:>8} {:>8}", "M", "K", "N", "Time", "TFLOPS");

            for &(m, k, n) in &test_sizes {
                let x_bytes = (m as usize) * (k as usize) * 2;
                let w_bytes = (n as usize) * (k as usize) * 2;
                let y_bytes = (m as usize) * (n as usize) * 4;

                let x_buf = device.alloc_vram(x_bytes)?;
                let w_buf = device.alloc_vram(w_bytes)?;
                let y_buf = device.alloc_vram(y_bytes)?;

                // Fill bf16 1.0
                let data = vec![0x3F80u16; std::cmp::max(m*k, n*k) as usize];
                x_buf.write(unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, x_bytes)
                });
                w_buf.write(unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, w_bytes)
                });

                let tiles_m = m / 32;
                let tiles_n = n / 64;
                let grid_x = tiles_n * 64;
                let grid_y = tiles_m;

                let mut ka = [0u8; 32];
                ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
                ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
                ka[16..24].copy_from_slice(&y_buf.gpu_addr().to_le_bytes());
                ka[24..28].copy_from_slice(&k.to_le_bytes());
                ka[28..32].copy_from_slice(&n.to_le_bytes());

                // Warmup
                for _ in 0..5 {
                    let ka_buf = pool.write_kernargs(0, &ka);
                    queue.submit(&gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                    queue.wait_idle()?;
                }

                // Timed
                let n_iter = if m * k * n <= 1024u32.pow(3) { 20 } else { 5 };
                let start = std::time::Instant::now();
                for i in 0..n_iter {
                    let ka_buf = pool.write_kernargs(i % 16, &ka);
                    queue.submit(&gpu_kernel, [grid_x, grid_y, 1], ka_buf);
                    queue.wait_idle()?;
                }
                let avg_us = start.elapsed().as_micros() as f64 / n_iter as f64;
                let tflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / (avg_us * 1e6);

                eprintln!("{:>6} {:>6} {:>6} | {:>6.1}μs {:>6.2} TF", m, k, n, avg_us, tflops);

                drop(x_buf); drop(w_buf); drop(y_buf);
            }
        }
    }

    eprintln!("\n══ Done ══");
    Ok(())
}
