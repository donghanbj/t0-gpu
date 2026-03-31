#!/usr/bin/env python3
"""
T0 vs PyTorch 逐层精度对比 — 数据生成器

生成 GEMM + SwiGLU + GEMM 链路的测试数据和 PyTorch 参考结果。
所有中间张量以原始二进制格式保存，供 Rust 端加载对比。

用法:
    python benchmarks/gen_torch_ref.py [--dim 256] [--hidden 512] [--seq 64]

输出目录: benchmarks/torch_ref/
"""

import argparse
import os
import struct
import numpy as np

def save_bf16(path: str, tensor):
    """Save tensor as BF16 binary (little-endian u16 array)."""
    import torch
    t = tensor.to(torch.bfloat16).contiguous().cpu()
    # BF16 is stored as uint16
    raw = t.view(torch.uint16).numpy().tobytes()
    with open(path, 'wb') as f:
        f.write(raw)

def save_f32(path: str, tensor):
    """Save tensor as F32 binary (little-endian f32 array)."""
    import torch
    t = tensor.to(torch.float32).contiguous().cpu()
    raw = t.numpy().tobytes()
    with open(path, 'wb') as f:
        f.write(raw)

def save_meta(path: str, **kwargs):
    """Save metadata as simple key=value text file."""
    with open(path, 'w') as f:
        for k, v in kwargs.items():
            f.write(f"{k}={v}\n")

def main():
    import torch

    parser = argparse.ArgumentParser(description="Generate torch reference for T0 precision comparison")
    parser.add_argument("--seq", type=int, default=64, help="Sequence length (M dimension)")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension (K dimension)")
    parser.add_argument("--hidden", type=int, default=512, help="FFN hidden dimension (N dimension)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="benchmarks/torch_ref", help="Output directory")
    args = parser.parse_args()

    M, K, N = args.seq, args.dim, args.hidden
    os.makedirs(args.outdir, exist_ok=True)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  T0 vs PyTorch — 参考数据生成                    ║")
    print(f"║  M={M}, K={K}, N={N}, seed={args.seed}          ║")
    print(f"╚══════════════════════════════════════════════════╝")

    torch.manual_seed(args.seed)

    # ══════════════════════════════════════════════════
    # 1. 生成输入 (BF16)
    # ══════════════════════════════════════════════════
    # X: [M, K] — 输入激活
    # W_gate: [N, K] — SwiGLU gate 权重 (stored as N×K, compute X @ W_gate.T)
    # W_up:   [N, K] — SwiGLU up 权重
    # W_down: [K, N] — 下投影权重 (stored as K×N, compute S @ W_down.T)
    
    # Use small values to prevent overflow in BF16
    X_f32 = torch.randn(M, K) * 0.1
    W_gate_f32 = torch.randn(N, K) * 0.1
    W_up_f32 = torch.randn(N, K) * 0.1
    W_down_f32 = torch.randn(K, N) * 0.1

    # Convert to BF16 (this is what T0 GEMM operates on)
    X = X_f32.to(torch.bfloat16)
    W_gate = W_gate_f32.to(torch.bfloat16)
    W_up = W_up_f32.to(torch.bfloat16)
    W_down = W_down_f32.to(torch.bfloat16)

    print(f"\n  Input shapes:")
    print(f"    X:      {list(X.shape)} bf16")
    print(f"    W_gate: {list(W_gate.shape)} bf16")
    print(f"    W_up:   {list(W_up.shape)} bf16")
    print(f"    W_down: {list(W_down.shape)} bf16")

    # Save inputs
    save_bf16(f"{args.outdir}/x.bin", X)
    save_bf16(f"{args.outdir}/w_gate.bin", W_gate)
    save_bf16(f"{args.outdir}/w_up.bin", W_up)
    save_bf16(f"{args.outdir}/w_down.bin", W_down)

    # ══════════════════════════════════════════════════
    # 2. 逐层计算 (with full precision tracking)
    # ══════════════════════════════════════════════════

    # --- Stage 1: GEMM (gate & up projections) ---
    # torch.mm does BF16 matmul with F32 accumulation on ROCm
    h_gate = torch.mm(X.float(), W_gate.float().T)  # [M, N] f32
    h_up = torch.mm(X.float(), W_up.float().T)      # [M, N] f32

    # Also compute BF16-precise version (matches WMMA behavior)
    h_gate_bf16acc = torch.mm(X, W_gate.T).float()  # BF16 matmul → F32
    h_up_bf16acc = torch.mm(X, W_up.T).float()

    print(f"\n  Stage 1: GEMM projections")
    print(f"    h_gate (f32 ref):    range [{h_gate.min():.4f}, {h_gate.max():.4f}]")
    print(f"    h_gate (bf16 acc):   range [{h_gate_bf16acc.min():.4f}, {h_gate_bf16acc.max():.4f}]")
    gemm1_diff = (h_gate - h_gate_bf16acc).abs()
    print(f"    f32 vs bf16acc diff: max={gemm1_diff.max():.6e}, mean={gemm1_diff.mean():.6e}")

    save_f32(f"{args.outdir}/h_gate_f32ref.bin", h_gate)
    save_f32(f"{args.outdir}/h_gate_bf16acc.bin", h_gate_bf16acc)
    save_f32(f"{args.outdir}/h_up_f32ref.bin", h_up)
    save_f32(f"{args.outdir}/h_up_bf16acc.bin", h_up_bf16acc)

    # --- Stage 2: SwiGLU = silu(gate) * up ---
    # silu(x) = x * sigmoid(x)
    silu_gate_f32 = torch.nn.functional.silu(h_gate)          # f32 精度
    swiglu_f32 = silu_gate_f32 * h_up                         # f32 精度

    silu_gate_bf16 = torch.nn.functional.silu(h_gate_bf16acc) # 从 bf16 GEMM 结果
    swiglu_bf16 = silu_gate_bf16 * h_up_bf16acc               # 混合精度

    print(f"\n  Stage 2: SwiGLU")
    print(f"    silu(gate) f32:   range [{silu_gate_f32.min():.4f}, {silu_gate_f32.max():.4f}]")
    print(f"    swiglu f32 ref:   range [{swiglu_f32.min():.4f}, {swiglu_f32.max():.4f}]")
    swiglu_diff = (swiglu_f32 - swiglu_bf16).abs()
    print(f"    f32 vs bf16-chain: max={swiglu_diff.max():.6e}, mean={swiglu_diff.mean():.6e}")

    save_f32(f"{args.outdir}/swiglu_f32ref.bin", swiglu_f32)
    save_f32(f"{args.outdir}/swiglu_bf16chain.bin", swiglu_bf16)

    # --- Stage 3: GEMM2 (down projection) ---
    # Convert SwiGLU output to BF16 for second GEMM (this is the critical truncation)
    swiglu_bf16_trunc = swiglu_bf16.to(torch.bfloat16)
    
    y_f32ref = torch.mm(swiglu_f32.float(), W_down.float().T)           # pure f32 chain
    y_bf16chain = torch.mm(swiglu_bf16_trunc, W_down.T).float()         # bf16 GEMM chain
    y_bf16_f32acc = torch.mm(swiglu_bf16_trunc.float(), W_down.float().T) # bf16 input, f32 matmul

    print(f"\n  Stage 3: GEMM2 (down projection)")
    print(f"    y f32 ref:      range [{y_f32ref.min():.4f}, {y_f32ref.max():.4f}]")
    print(f"    y bf16 chain:   range [{y_bf16chain.min():.4f}, {y_bf16chain.max():.4f}]")
    y_diff = (y_f32ref - y_bf16chain).abs()
    print(f"    f32 vs bf16-chain: max={y_diff.max():.6e}, mean={y_diff.mean():.6e}")

    save_f32(f"{args.outdir}/y_f32ref.bin", y_f32ref)
    save_f32(f"{args.outdir}/y_bf16chain.bin", y_bf16chain)
    save_f32(f"{args.outdir}/y_bf16_f32acc.bin", y_bf16_f32acc)
    save_bf16(f"{args.outdir}/swiglu_bf16_trunc.bin", swiglu_bf16_trunc)

    # --- Stage 4: Summary ---
    print(f"\n  ════════════════════════════════════════")
    print(f"  误差链路分析:")
    e1 = gemm1_diff.max().item()
    e2 = swiglu_diff.max().item()
    e3 = y_diff.max().item()
    print(f"    GEMM1 (bf16 vs f32):    {e1:.6e}")
    print(f"    SwiGLU chain error:     {e2:.6e}")
    print(f"    GEMM2 end-to-end:       {e3:.6e}")
    print(f"  ════════════════════════════════════════")

    # Save metadata
    save_meta(f"{args.outdir}/meta.txt",
        M=M, K=K, N=N, seed=args.seed,
        torch_version=torch.__version__,
        gemm1_max_err=f"{e1:.6e}",
        swiglu_max_err=f"{e2:.6e}",
        gemm2_max_err=f"{e3:.6e}",
    )

    print(f"\n  ✅ 所有参考数据已保存到 {args.outdir}/")
    print(f"     共 {len(os.listdir(args.outdir))} 个文件")
    print(f"\n  下一步: cargo test --release --features rocm -- test_precision_vs_torch --ignored --nocapture")

if __name__ == "__main__":
    main()
