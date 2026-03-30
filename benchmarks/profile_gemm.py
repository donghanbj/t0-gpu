#!/usr/bin/env python3
"""
Triton + rocBLAS GEMM 深度 Profiling — 寄存器/LDS/ISA 级优化分析

抓取：
  1. Triton autotune 配置 (BLOCK_M/N/K, warps, stages)
  2. ISA 级分析: VGPR/SGPR 数、LDS 大小、occupancy、WGP mode
  3. WMMA 调度模式: 双链交错、累加器依赖链
  4. 同步分析: waitcnt 值分布、barrier 频率
  5. 内存访问: global load/store 模式、LDS read/write 模式
  6. 指令混合: VALU/VMEM/LDS/SALU/CTRL 占比
  7. 软件流水线: prefetch 深度、LDS-WMMA 重叠
  8. rocBLAS kernel 名称 + tile 参数 (via rocprof)

用法: python3 profile_gemm.py [--size 4096] [--dump-isa]
"""
import torch
import triton
import triton.language as tl
import time, os, re, subprocess, glob, sys, json
from collections import Counter, defaultdict

# ============================================================================
# Triton GEMM kernel (from official tutorial)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
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
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# ============================================================================
# ISA Deep Analysis
# ============================================================================

def analyze_isa(asm_text: str, label: str = "kernel"):
    """ISA 级深度分析"""
    lines = asm_text.splitlines()
    result = {}

    # ── 1. Kernel Descriptor (VGPR/SGPR/LDS/WGP) ──
    vgpr_match = re.search(r'granulated_workitem_vgpr_count\s*=\s*(\d+)', asm_text) or \
                 re.search(r'\.amdhsa_next_free_vgpr\s+(\d+)', asm_text)
    sgpr_match = re.search(r'granulated_wavefront_sgpr_count\s*=\s*(\d+)', asm_text) or \
                 re.search(r'\.amdhsa_next_free_sgpr\s+(\d+)', asm_text)
    lds_match = re.search(r'group_segment_fixed_size\s*=\s*(\d+)', asm_text) or \
                re.search(r'\.amdhsa_group_segment_fixed_size\s+(\d+)', asm_text)
    wgp_match = re.search(r'workgroup_processor_mode[:\s]*(\d+|true|false)', asm_text) or \
                re.search(r'\.amdhsa_workgroup_processor_mode\s+(\d+)', asm_text)

    if vgpr_match:
        val = int(vgpr_match.group(1))
        # amdhsa_next_free_vgpr is the actual count
        if 'next_free_vgpr' in (vgpr_match.group(0) if vgpr_match else ''):
            result['vgprs'] = val
        else:
            result['vgprs'] = (val + 1) * 8  # granulated: (N+1)*8
    if sgpr_match:
        val = int(sgpr_match.group(1))
        if 'next_free_sgpr' in (sgpr_match.group(0) if sgpr_match else ''):
            result['sgprs'] = val
        else:
            result['sgprs'] = (val + 1) * 8
    if lds_match:
        result['lds_bytes'] = int(lds_match.group(1))
        result['lds_kb'] = result['lds_bytes'] / 1024
    if wgp_match:
        v = wgp_match.group(1)
        result['wgp_mode'] = v in ('1', 'true')

    # Compute occupancy (GFX1100: 256 VGPRs per SIMD, 1024 total per CU)
    if 'vgprs' in result:
        vgprs = result['vgprs']
        if vgprs <= 64:    result['waves_per_simd'] = 8
        elif vgprs <= 96:  result['waves_per_simd'] = 5
        elif vgprs <= 128: result['waves_per_simd'] = 4
        elif vgprs <= 170: result['waves_per_simd'] = 3
        elif vgprs <= 256: result['waves_per_simd'] = 2
        else:              result['waves_per_simd'] = 1

    # ── 2. Instruction Classification ──
    categories = Counter()
    inst_counts = Counter()
    wmma_ops = []
    waitcnt_vals = []
    ds_ops = []
    global_ops = []
    barrier_positions = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(';') or stripped.startswith('.') or stripped.endswith(':'):
            continue

        # WMMA
        if 'v_wmma' in stripped:
            categories['WMMA'] += 1
            wmma_ops.append((i, stripped))
            m = re.search(r'v\[(\d+):(\d+)\],\s*v\[(\d+):(\d+)\],\s*v\[(\d+):(\d+)\],\s*v\[(\d+):(\d+)\]', stripped)
            if m:
                inst_counts[f"wmma_dst_v{m.group(1)}:{m.group(2)}"] += 1
        # LDS
        elif stripped.startswith('ds_') or 'ds_load' in stripped or 'ds_store' in stripped or 'ds_read' in stripped or 'ds_write' in stripped:
            categories['LDS'] += 1
            ds_ops.append((i, stripped))
            if 'b128' in stripped or 'x2' in stripped: inst_counts['ds_128bit'] += 1
            elif 'b64' in stripped: inst_counts['ds_64bit'] += 1
            elif 'b32' in stripped: inst_counts['ds_32bit'] += 1
            elif 'b16' in stripped: inst_counts['ds_16bit'] += 1
            else: inst_counts['ds_other'] += 1
        # Global memory
        elif 'global_' in stripped or 'buffer_' in stripped:
            categories['GMEM'] += 1
            global_ops.append((i, stripped))
            if 'load' in stripped:
                if 'b128' in stripped: inst_counts['gmem_load_128'] += 1
                elif 'b64' in stripped: inst_counts['gmem_load_64'] += 1
                elif 'b32' in stripped: inst_counts['gmem_load_32'] += 1
                else: inst_counts['gmem_load_other'] += 1
            elif 'store' in stripped:
                if 'b128' in stripped: inst_counts['gmem_store_128'] += 1
                elif 'b64' in stripped: inst_counts['gmem_store_64'] += 1
                elif 'b32' in stripped: inst_counts['gmem_store_32'] += 1
                else: inst_counts['gmem_store_other'] += 1
        # Waitcnt
        elif 's_waitcnt' in stripped:
            categories['SYNC'] += 1
            m = re.search(r'vmcnt\((\d+)\)', stripped)
            if m: waitcnt_vals.append(('vmcnt', int(m.group(1))))
            m = re.search(r'lgkmcnt\((\d+)\)', stripped)
            if m: waitcnt_vals.append(('lgkmcnt', int(m.group(1))))
            m = re.search(r'expcnt\((\d+)\)', stripped)
            if m: waitcnt_vals.append(('expcnt', int(m.group(1))))
        # Barrier
        elif 's_barrier' in stripped:
            categories['SYNC'] += 1
            barrier_positions.append(i)
        # VALU (vector ALU not WMMA)
        elif stripped.startswith('v_') and 'v_wmma' not in stripped:
            categories['VALU'] += 1
            # Track specific ops
            op = stripped.split()[0] if stripped.split() else ''
            inst_counts[op] += 1
        # SALU (scalar ALU)
        elif stripped.startswith('s_') and 's_waitcnt' not in stripped and 's_barrier' not in stripped:
            categories['SALU'] += 1
        # Delay hints
        elif 's_delay_alu' in stripped:
            categories['DELAY'] += 1

    result['inst_categories'] = dict(categories)
    result['total_instructions'] = sum(categories.values())

    # ── 3. WMMA Scheduling Analysis ──
    result['wmma_count'] = len(wmma_ops)
    if wmma_ops:
        # Extract accumulator register groups
        acc_groups = []
        for _, op in wmma_ops:
            m = re.search(r'v\[(\d+):(\d+)\],\s*v\[(\d+):(\d+)\],\s*v\[(\d+):(\d+)\],\s*v\[(\d+):(\d+)\]', op)
            if m:
                dst_lo = int(m.group(1))
                a_lo = int(m.group(3))
                b_lo = int(m.group(5))
                c_lo = int(m.group(7))
                acc_groups.append({
                    'dst': f"v[{m.group(1)}:{m.group(2)}]",
                    'a': f"v[{m.group(3)}:{m.group(4)}]",
                    'b': f"v[{m.group(5)}:{m.group(6)}]",
                    'c': f"v[{m.group(7)}:{m.group(8)}]",
                    'dst_lo': dst_lo, 'a_lo': a_lo, 'b_lo': b_lo, 'c_lo': c_lo,
                })

        # Analyze interleaving: consecutive WMMAs with different acc → ILP
        if len(acc_groups) >= 2:
            ilp_pairs = 0
            dep_pairs = 0
            for i in range(len(acc_groups)-1):
                if acc_groups[i]['dst_lo'] != acc_groups[i+1]['dst_lo']:
                    ilp_pairs += 1  # different acc → can execute in parallel
                else:
                    dep_pairs += 1  # same acc → WAW dependency
            result['wmma_ilp_pairs'] = ilp_pairs
            result['wmma_dep_pairs'] = dep_pairs
            result['wmma_ilp_ratio'] = ilp_pairs / max(1, ilp_pairs + dep_pairs)

        # Unique accumulator groups
        unique_acc = set(g['dst'] for g in acc_groups)
        unique_a = set(g['a'] for g in acc_groups)
        unique_b = set(g['b'] for g in acc_groups)
        result['wmma_unique_acc'] = len(unique_acc)
        result['wmma_unique_a_frags'] = len(unique_a)
        result['wmma_unique_b_frags'] = len(unique_b)
        result['wmma_acc_groups'] = sorted(unique_acc)
        result['wmma_a_frags'] = sorted(unique_a)
        result['wmma_b_frags'] = sorted(unique_b)

        # K sub-steps = WMMA / (n_row_blocks × n_col_tiles)
        n_acc = len(unique_acc)
        if n_acc > 0:
            result['wmma_per_acc'] = len(wmma_ops) // n_acc
            # n_row_blocks = unique A frags, n_col_tiles = unique B frags
            result['n_row_blocks'] = len(unique_a)
            result['n_col_tiles'] = len(unique_b)
            result['k_sub_steps'] = len(wmma_ops) // (len(unique_a) * len(unique_b))

    # ── 4. LDS Analysis ──
    result['ds_total'] = len(ds_ops)
    ds_load_count = sum(1 for _, op in ds_ops if 'load' in op or 'read' in op)
    ds_store_count = sum(1 for _, op in ds_ops if 'store' in op or 'write' in op)
    result['ds_loads'] = ds_load_count
    result['ds_stores'] = ds_store_count
    result['ds_width_counts'] = {k: v for k, v in inst_counts.items() if k.startswith('ds_')}

    # ── 5. GMEM Analysis ──
    result['gmem_total'] = len(global_ops)
    gmem_loads = sum(1 for _, op in global_ops if 'load' in op)
    gmem_stores = sum(1 for _, op in global_ops if 'store' in op)
    result['gmem_loads'] = gmem_loads
    result['gmem_stores'] = gmem_stores
    result['gmem_width_counts'] = {k: v for k, v in inst_counts.items() if k.startswith('gmem_')}

    # ── 6. Waitcnt Analysis ──
    result['waitcnt_count'] = sum(1 for k in categories if k == 'SYNC')
    result['barrier_count'] = len(barrier_positions)
    if waitcnt_vals:
        vmcnt_vals = [v for t, v in waitcnt_vals if t == 'vmcnt']
        lgkm_vals = [v for t, v in waitcnt_vals if t == 'lgkmcnt']
        if vmcnt_vals:
            result['vmcnt_distribution'] = dict(Counter(vmcnt_vals).most_common(10))
            result['vmcnt_zero_ratio'] = vmcnt_vals.count(0) / len(vmcnt_vals)
        if lgkm_vals:
            result['lgkmcnt_distribution'] = dict(Counter(lgkm_vals).most_common(10))
            result['lgkmcnt_zero_ratio'] = lgkm_vals.count(0) / len(lgkm_vals)

    # ── 7. Software Pipelining Detection ──
    # Check if GMEM loads are interleaved with WMMA (prefetch pattern)
    if wmma_ops and global_ops:
        wmma_lines = set(i for i, _ in wmma_ops)
        gmem_lines = set(i for i, _ in global_ops)
        # Check if any GMEM loads appear AFTER the first WMMA but BEFORE the last WMMA
        first_wmma = min(wmma_lines)
        last_wmma = max(wmma_lines)
        prefetch_loads = [l for l in gmem_lines if first_wmma < l < last_wmma]
        result['software_pipeline'] = len(prefetch_loads) > 0
        result['prefetch_loads_in_compute'] = len(prefetch_loads)

    # Check LDS-WMMA interleaving
    if wmma_ops and ds_ops:
        ds_lines = set(i for i, _ in ds_ops if 'load' in ds_ops[0][1] or 'read' in ds_ops[0][1])
        wmma_lines_list = sorted(i for i, _ in wmma_ops)
        # Count LDS loads between consecutive WMMAs
        lds_between_wmma = 0
        for j in range(len(wmma_lines_list) - 1):
            start, end = wmma_lines_list[j], wmma_lines_list[j+1]
            lds_between = sum(1 for dl, dop in ds_ops if start < dl < end and ('load' in dop or 'read' in dop))
            lds_between_wmma += lds_between
        result['lds_between_wmma'] = lds_between_wmma

    # ── 8. Top VALU Instructions ──
    valu_ops = {k: v for k, v in inst_counts.items() if k.startswith('v_') and 'wmma' not in k}
    result['top_valu_ops'] = dict(sorted(valu_ops.items(), key=lambda x: -x[1])[:15])

    return result


def format_analysis(result: dict, label: str):
    """格式化输出分析结果"""
    print(f"\n{'='*72}")
    print(f"  {label} — ISA 深度分析")
    print(f"{'='*72}")

    # Kernel descriptor
    print(f"\n  ┌─── Kernel 描述符 ───")
    if 'vgprs' in result:
        print(f"  │ VGPRs:           {result['vgprs']}")
    if 'sgprs' in result:
        print(f"  │ SGPRs:           {result['sgprs']}")
    if 'lds_bytes' in result:
        print(f"  │ LDS:             {result['lds_bytes']} bytes ({result['lds_kb']:.1f} KB)")
    if 'wgp_mode' in result:
        print(f"  │ WGP Mode:        {'✅ ENABLED' if result['wgp_mode'] else '❌ disabled'}")
    if 'waves_per_simd' in result:
        print(f"  │ Occupancy:       {result['waves_per_simd']} waves/SIMD")
    print(f"  └───")

    # Instruction mix
    if 'inst_categories' in result:
        print(f"\n  ┌─── 指令混合 (总计 {result['total_instructions']}) ───")
        cats = result['inst_categories']
        for cat in ['WMMA', 'VALU', 'LDS', 'GMEM', 'SALU', 'SYNC', 'DELAY']:
            if cat in cats:
                pct = cats[cat] / result['total_instructions'] * 100
                bar = '█' * int(pct / 2)
                print(f"  │ {cat:8s} {cats[cat]:5d}  ({pct:5.1f}%)  {bar}")
        print(f"  └───")

    # WMMA scheduling
    if 'wmma_count' in result and result['wmma_count'] > 0:
        print(f"\n  ┌─── WMMA 调度分析 ───")
        print(f"  │ 总 WMMA 指令:     {result['wmma_count']}")
        if 'wmma_unique_acc' in result:
            print(f"  │ 独立 acc 组:     {result['wmma_unique_acc']} 组")
        if 'n_row_blocks' in result:
            print(f"  │ row blocks:      {result['n_row_blocks']}")
        if 'n_col_tiles' in result:
            print(f"  │ col tiles:       {result['n_col_tiles']}")
        if 'k_sub_steps' in result:
            print(f"  │ K sub-steps:     {result['k_sub_steps']}")
        if 'wmma_ilp_ratio' in result:
            print(f"  │ ILP 比率:        {result['wmma_ilp_ratio']:.1%} ({result['wmma_ilp_pairs']} ILP / {result['wmma_dep_pairs']} dep)")
        if 'wmma_per_acc' in result:
            print(f"  │ WMMA/acc 组:     {result['wmma_per_acc']}")
        print(f"  └───")

        # Acc register map
        if 'wmma_acc_groups' in result:
            print(f"\n  ┌─── 累加器寄存器映射 ───")
            for i, acc in enumerate(result['wmma_acc_groups']):
                print(f"  │ acc[{i}] = {acc}")
            print(f"  └───")

    # LDS access pattern
    print(f"\n  ┌─── LDS 访问模式 ───")
    print(f"  │ ds_load:          {result.get('ds_loads', 0)}")
    print(f"  │ ds_store:         {result.get('ds_stores', 0)}")
    if 'ds_width_counts' in result:
        for k, v in sorted(result['ds_width_counts'].items()):
            print(f"  │   {k}: {v}")
    print(f"  └───")

    # GMEM access pattern
    print(f"\n  ┌─── GMEM 访问模式 ───")
    print(f"  │ global_load:      {result.get('gmem_loads', 0)}")
    print(f"  │ global_store:     {result.get('gmem_stores', 0)}")
    if 'gmem_width_counts' in result:
        for k, v in sorted(result['gmem_width_counts'].items()):
            print(f"  │   {k}: {v}")
    print(f"  └───")

    # Synchronization
    print(f"\n  ┌─── 同步与 Waitcnt ───")
    print(f"  │ s_barrier:        {result.get('barrier_count', 0)}")
    if 'vmcnt_distribution' in result:
        print(f"  │ vmcnt 分布:       {result['vmcnt_distribution']}")
        print(f"  │ vmcnt(0) 比率:   {result.get('vmcnt_zero_ratio', 0):.1%}")
    if 'lgkmcnt_distribution' in result:
        print(f"  │ lgkmcnt 分布:     {result['lgkmcnt_distribution']}")
        print(f"  │ lgkmcnt(0) 比率: {result.get('lgkmcnt_zero_ratio', 0):.1%}")
    print(f"  └───")

    # Software pipelining
    print(f"\n  ┌─── 软件流水线 ───")
    if 'software_pipeline' in result:
        print(f"  │ GMEM prefetch 交叠: {'✅' if result['software_pipeline'] else '❌'}")
        print(f"  │ compute 区间内 loads: {result.get('prefetch_loads_in_compute', 0)}")
    if 'lds_between_wmma' in result:
        print(f"  │ WMMA 间 LDS reads:  {result['lds_between_wmma']}")
    print(f"  └───")

    # Top VALU ops
    if 'top_valu_ops' in result and result['top_valu_ops']:
        print(f"\n  ┌─── Top VALU 指令 ───")
        for op, cnt in list(result['top_valu_ops'].items())[:10]:
            print(f"  │ {op:30s} × {cnt}")
        print(f"  └───")

# ============================================================================
# Benchmark helpers
# ============================================================================

def triton_gemm(a, b):
    M, K = a.shape; K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c

def benchmark_fn(fn, *args, warmup=5, rep=20):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn(*args)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    size = int(sys.argv[sys.argv.index('--size') + 1]) if '--size' in sys.argv else 4096
    dump_isa = '--dump-isa' in sys.argv

    print("=" * 72)
    print(f"  GEMM 深度 Profiling — RX 7900 XTX (GFX1100)")
    print(f"  Matrix size: {size}³  |  Triton 3.6.0 + rocBLAS (torch)")
    print("=" * 72)

    device = torch.device('cuda')
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # ── Part 1: Triton Autotune ──
    sizes = [(256,256,256), (512,512,512), (1024,1024,1024), (2048,2048,2048), (4096,4096,4096)]

    print(f"\n{'='*72}")
    print(f"  Part 1: Triton Autotune 配置")
    print(f"{'='*72}")

    for M, N, K in sizes:
        matmul_kernel.cache.clear()
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        _ = triton_gemm(a, b); torch.cuda.synchronize()
        try:
            bc = matmul_kernel.best_config
            cfg = f"BLOCK_M={bc.kwargs.get('BLOCK_M')}, BLOCK_N={bc.kwargs.get('BLOCK_N')}, BLOCK_K={bc.kwargs.get('BLOCK_K')}, warps={bc.num_warps}, stages={bc.num_stages}"
            print(f"  {M:>5d}³: {cfg}")
        except: pass

    # ── Part 2: Performance ──
    print(f"\n{'='*72}")
    print(f"  Part 2: 性能对比")
    print(f"{'='*72}")
    print(f"\n{'Size':>12s}  {'Triton':>10s}  {'rocBLAS':>10s}  {'ratio':>8s}")
    print("-" * 50)

    for M, N, K in sizes:
        flops = 2.0 * M * N * K
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        matmul_kernel.cache.clear()
        _ = triton_gemm(a, b); torch.cuda.synchronize()
        t_ms = benchmark_fn(triton_gemm, a, b)
        r_ms = benchmark_fn(torch.mm, a, b)
        t_tf = flops / (t_ms * 1e-3) / 1e12
        r_tf = flops / (r_ms * 1e-3) / 1e12
        print(f"  {M:>5d}³ {t_tf:9.2f} TF {r_tf:9.2f} TF  {t_tf/r_tf:7.1%}")

    # ── Part 3: Triton ISA 深度分析 ──
    print(f"\n{'='*72}")
    print(f"  Part 3: Triton ISA 深度分析 ({size}³)")
    print(f"{'='*72}")

    # Force compile for target size
    matmul_kernel.cache.clear()
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    _ = triton_gemm(a, b); torch.cuda.synchronize()

    # Find ISA
    cache_dir = os.path.join(os.path.expanduser("~"), ".triton", "cache")
    amdgcn_files = glob.glob(os.path.join(cache_dir, "**", "*.amdgcn"), recursive=True)
    if amdgcn_files:
        amdgcn_files.sort(key=os.path.getmtime, reverse=True)
        isa_path = amdgcn_files[0]
        print(f"\n  ISA file: {isa_path}")
        with open(isa_path) as f:
            isa_text = f.read()
        result = analyze_isa(isa_text, f"Triton {size}³")
        format_analysis(result, f"Triton matmul_kernel ({size}³)")

        if dump_isa:
            out = f"/tmp/triton_gemm_{size}.s"
            with open(out, 'w') as f: f.write(isa_text)
            print(f"\n  Full ISA saved: {out}")

    # ── Part 4: rocBLAS via rocprof ──
    print(f"\n{'='*72}")
    print(f"  Part 4: rocBLAS Kernel 分析 (via rocprof)")
    print(f"{'='*72}")

    rocprof_script = "/tmp/rocblas_profile_deep.py"
    with open(rocprof_script, 'w') as f:
        f.write(f"""import torch
a = torch.randn({size}, {size}, device='cuda', dtype=torch.float16)
b = torch.randn({size}, {size}, device='cuda', dtype=torch.float16)
for _ in range(3): c = torch.mm(a, b)
torch.cuda.synchronize()
for _ in range(5): c = torch.mm(a, b)
torch.cuda.synchronize()
""")

    try:
        result = subprocess.run(
            ["rocprof", "--stats", "--hip-trace", "-o", "/tmp/rocblas_deep.csv",
             "python3", rocprof_script],
            capture_output=True, text=True, timeout=60, cwd="/tmp")

        if os.path.exists("/tmp/rocblas_deep.stats.csv"):
            print(f"\n  rocBLAS stats:")
            with open("/tmp/rocblas_deep.stats.csv") as f:
                for line in f:
                    if 'Cijk' in line or 'gemm' in line.lower() or 'Name' in line:
                        parts = line.strip().split(',')
                        if 'Name' in line:
                            print(f"    {line.strip()}")
                        else:
                            # Extract kernel name and stats
                            name = parts[-1].strip() if len(parts) > 4 else parts[0]
                            print(f"    Kernel: {name[:100]}")
                            if len(parts) > 3:
                                print(f"    Calls={parts[0].strip()}, AvgNs={parts[2].strip()}, TotalNs={parts[1].strip()}")
        # Try to extract rocBLAS kernel ISA
        print(f"\n  To extract rocBLAS ISA, run:")
        print(f"    ROCBLAS_LAYER=4 python3 {rocprof_script} 2>&1 | grep -i kernel")
        print(f"    Or: rocprof --hsa-trace -o /tmp/rocblas_hsa.csv python3 {rocprof_script}")

    except Exception as e:
        print(f"  rocprof error: {e}")

    print(f"\n{'='*72}")
    print(f"  抓包完成!")
    print(f"{'='*72}")
