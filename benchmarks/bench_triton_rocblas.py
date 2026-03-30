#!/usr/bin/env python3
"""
T0 vs Triton+rocBLAS 统一基准测试 — Triton/rocBLAS 端
========================================================

在 README.md 定义的 9 个标准矩阵尺寸上，测量：
  1. PyTorch torch.mm (= rocBLAS)
  2. Triton @triton.jit matmul
  3. Triton @triton.autotune matmul

数据类型：BF16 输入，F32 输出
计算公式：TFLOPS = 2 * M * K * N / (elapsed_us * 1e6)

运行：python3 benchmarks/bench_triton_rocblas.py
"""

import torch
import triton
import triton.language as tl
import time
import csv
import sys
import os

# ─── 标准矩阵尺寸（与 README.md 完全一致） ───
SIZES = [
    (256,   256,   256),
    (512,   512,   512),
    (1024,  1024,  1024),
    (2048,  2048,  2048),
    (4096,  4096,  4096),
    (128,   1024,  4096),
    (256,   1024,  4096),
    (512,   1024,  4096),
    (1024,  1024,  4096),
]

# ─── Triton matmul 内核（基于官方 tutorial 的 RDNA3 优化版） ───
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cn + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# ─── Triton autotuned matmul ───
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cn + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# ─── 测量函数 ───
def bench_pytorch_rocblas(M, K, N, warmup=5, n_iters=50):
    """PyTorch torch.mm (= rocBLAS underneath)"""
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    # Warmup
    for _ in range(warmup):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    # Timed
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed_s = (time.perf_counter() - start) / n_iters
    tflops = 2.0 * M * K * N / (elapsed_s * 1e12)
    return tflops, elapsed_s * 1e6  # tflops, μs

def bench_triton_fixed(M, K, N, warmup=5, n_iters=50):
    """Triton matmul with fixed config (BLOCK=128x128x32)"""
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = lambda meta: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    # Warmup
    for _ in range(warmup):
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=8,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=8,
        )
    torch.cuda.synchronize()
    elapsed_s = (time.perf_counter() - start) / n_iters
    tflops = 2.0 * M * K * N / (elapsed_s * 1e12)
    return tflops, elapsed_s * 1e6

def bench_triton_autotuned(M, K, N, warmup=5, n_iters=50):
    """Triton matmul with @triton.autotune"""
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )

    # Warmup (triggers autotune)
    for _ in range(warmup):
        matmul_kernel_autotuned[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        matmul_kernel_autotuned[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    torch.cuda.synchronize()
    elapsed_s = (time.perf_counter() - start) / n_iters
    tflops = 2.0 * M * K * N / (elapsed_s * 1e12)

    # Correctness check on last result
    ref = torch.mm(a, b).to(torch.float32)
    max_err = (c - ref).abs().max().item()
    if max_err > 1.0:
        print(f"  ⚠️  Correctness warning: max_err={max_err:.4f}")

    return tflops, elapsed_s * 1e6

# ─── 主程序 ───
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Triton + rocBLAS 基准测试 — RX 7900 XTX                      ║")
    print("║  BF16 WMMA GEMM (Y = X × W^T), F32 output                     ║")
    print(f"║  PyTorch {torch.__version__}, Triton {triton.__version__}                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Header
    header = f"{'Matrix':>20s} | {'rocBLAS':>10s} | {'Triton':>10s} | {'Triton-AT':>10s} | {'Best':>10s}"
    print(header)
    print("-" * len(header))

    results = []

    for (M, K, N) in SIZES:
        label = f"{M}×{K}×{N}"

        # 1. PyTorch / rocBLAS
        tf_rb, us_rb = bench_pytorch_rocblas(M, K, N)

        # 2. Triton fixed
        try:
            tf_tr, us_tr = bench_triton_fixed(M, K, N)
        except Exception as e:
            print(f"  Triton fixed failed for {label}: {e}")
            tf_tr, us_tr = 0.0, 0.0

        # 3. Triton autotuned
        try:
            tf_at, us_at = bench_triton_autotuned(M, K, N)
        except Exception as e:
            print(f"  Triton autotuned failed for {label}: {e}")
            tf_at, us_at = 0.0, 0.0

        best_tf = max(tf_rb, tf_tr, tf_at)
        best_name = "rocBLAS" if best_tf == tf_rb else ("Triton" if best_tf == tf_tr else "Triton-AT")

        print(f"{label:>20s} | {tf_rb:>8.2f} TF | {tf_tr:>8.2f} TF | {tf_at:>8.2f} TF | ★ {best_name} {best_tf:.2f}")

        results.append({
            'M': M, 'K': K, 'N': N,
            'rocblas_tflops': tf_rb, 'rocblas_us': us_rb,
            'triton_tflops': tf_tr, 'triton_us': us_tr,
            'triton_at_tflops': tf_at, 'triton_at_us': us_at,
            'best': best_name, 'best_tflops': best_tf,
        })

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'triton_rocblas_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n结果已保存到 / Results saved to: {csv_path}")

    # Summary
    print("\n═══ 汇总 / Summary ═══")
    print(f"{'Matrix':>20s} | {'rocBLAS':>10s} | {'Triton-AT':>10s} | {'Best':>12s}")
    print("-" * 65)
    for r in results:
        label = f"{r['M']}×{r['K']}×{r['N']}"
        best = r['best']
        tf = r['best_tflops']
        print(f"{label:>20s} | {r['rocblas_tflops']:>8.2f} TF | {r['triton_at_tflops']:>8.2f} TF | ★ {best} {tf:.2f} TF")

if __name__ == '__main__':
    main()
