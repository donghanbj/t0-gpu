# 示例

## 1. Hello GEMM — 最小 bf16 矩阵乘法

```rust
use t0::math::matmul_direct;
use t0::schedule::GFX1100Schedule;
use t0::ir::Target;

fn main() -> Result<(), String> {
    // 编译 GEMM 内核
    let hsaco = matmul_direct(&GFX1100Schedule)
        .compile(Target::GFX1100)?;

    // 打开 GPU
    let device = KfdDevice::open()?;
    let queue = device.create_queue()?;
    let pool = DispatchPool::new(&device, 0)?;
    let kernel = GpuKernel::load(&device, &hsaco, &KernelLoadConfig {
        workgroup_size: [64, 1, 1],
        lds_size: 0,
    })?;

    // 准备数据: A[32,64] × B^T[128,64] = C[32,128]
    let m = 32; let k = 64; let n = 128;
    let a_buf = device.alloc_vram(m * k * 2)?;  // bf16
    let b_buf = device.alloc_vram(n * k * 2)?;  // bf16 (N×K, 已转置)
    let c_buf = device.alloc_vram(m * n * 4)?;  // f32 输出

    // 填充随机 bf16 数据 (略)
    // ...

    // 调度
    let mut ka = [0u8; 32];
    ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
    ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
    ka[16..24].copy_from_slice(&c_buf.gpu_addr().to_le_bytes());
    ka[24..28].copy_from_slice(&(k as u32).to_le_bytes());
    ka[28..32].copy_from_slice(&(n as u32).to_le_bytes());

    let grid_x = ((n + 63) / 64) as u32 * 64;
    let grid_y = ((m + 31) / 32) as u32;
    let ka_va = pool.write_kernargs(0, &ka);
    queue.submit(&kernel, [grid_x, grid_y, 1], ka_va);
    queue.wait_idle()?;

    // 读取结果
    let mut result = vec![0f32; m * n];
    c_buf.read(unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, m * n * 4)
    });
    println!("C[0,0] = {}", result[0]);
    Ok(())
}
```

## 2. 自定义逐元素内核

```rust
// 用 T0 math 定义: y = x² + 2x + 1
fn square_plus(sched: &dyn Schedule) -> T0Kernel {
    // T0 自动处理：线程映射、全局加载、存储、grid 计算
    t0_elementwise_unary(|x| {
        let x2 = mul(x, x);
        let two_x = mul(x, const_f32(2.0));
        add(add(x2, two_x), const_f32(1.0))
    })
}
```

## 3. 自定义注意力机制

这是 T0 的核心价值 — 定义任意注意力变体：

```rust
// 示例：线性注意力 (不用 softmax)
fn linear_attention() -> T0Kernel {
    // O = (Q @ K^T) @ V
    // 无 softmax，直接矩阵乘
    let qk = wmma_matmul(q, k_t);     // [M, N] 注意力矩阵
    let out = wmma_matmul(qk, v);     // [M, D] 输出
    output_f32("O", out)
}

// 示例：OCPA chunk-wise 注意力
fn ocpa_intra_attention() -> T0Kernel {
    // S_intra = tril(Q @ K^T)        // 因果掩码
    // O_intra = S_intra @ V
    let qk = wmma_matmul(q, k_t);
    let qk_causal = apply_causal_mask(qk);
    let out = wmma_matmul(qk_causal, v);
    output_f32("O", out)
}
```

## 源文件索引

| 现有实现 | 路径 | 说明 |
|---------|------|------|
| matmul_direct | `t0/math.rs` | bf16 WMMA GEMM (NT) |
| OCPA forward intra | `kernels/ocpa_forward_intra.rs` | chunk 内因果注意力 |
| OCPA backward intra | `kernels/ocpa_backward_intra.rs` | chunk 内反向 |
| Softmax + CE Loss | `kernels/softmax_ce_loss.rs` | 融合 softmax+交叉熵 |
| Ada-GLAM 优化器 | `kernels/ada_glam_kernels.rs` | 12 个优化器内核 |
