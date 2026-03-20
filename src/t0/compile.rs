//! T0 Kernel Compiler — Top-level API
//!
//! Provides the `T0Kernel` builder that lets you define GPU kernels
//! using virtual registers, then compiles to ELF via LLVM.
//!
//! ## Example
//! ```ignore
//! let mut k = T0Kernel::new("scale_kernel");
//! let ptr = k.arg_ptr("data");
//! let n = k.arg_u32("n_elems");
//! let val = k.alloc_vreg();
//! // ... add ops ...
//! let elf = k.compile(Target::GFX1100)?;
//! ```

use std::process::Command;
use super::ir::*;
use super::regalloc::{self, RegAlloc};
use super::asm_emitter::AsmEmitter;

// ============================================================================
// T0Kernel: the main builder
// ============================================================================

/// A GPU kernel under construction.
/// Build the kernel by calling methods to add ops, then compile().
pub struct T0Kernel {
    pub name: String,
    ops: Vec<Op>,
    args: Vec<KernArg>,
    vreg_allocs: Vec<VRegAlloc>,
    sreg_allocs: Vec<SRegAlloc>,
    next_vreg: u32,
    next_sreg: u32,
    kernarg_size: u32,
    lds_size: u32,
    label_counter: u32,
}

impl T0Kernel {
    /// Create a new empty kernel.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ops: Vec::new(),
            args: Vec::new(),
            vreg_allocs: Vec::new(),
            sreg_allocs: Vec::new(),
            // v0 = WORKITEM_ID_X (hardware), RESERVED — alloc_vreg starts from VReg(1)
            next_vreg: 1,
            // s0:s1 = kernarg ptr, s2/s3/s4 = TGID (hardware)
            next_sreg: 0,
            kernarg_size: 0,
            lds_size: 0,
            label_counter: 0,
        }
    }

    // ── Kernel arguments ──

    /// Declare a 64-bit pointer argument. Returns the SRegPair for the pointer.
    pub fn arg_ptr(&mut self, name: &str) -> SRegPair {
        let sreg_id = self.next_sreg;
        self.next_sreg += 2;
        let sreg = SReg(sreg_id);
        let offset = self.kernarg_size;
        self.kernarg_size += 8;
        self.sreg_allocs.push(SRegAlloc {
            sreg,
            count: 2,
            alignment: Alignment::Align2,
        });
        self.args.push(KernArg {
            name: name.to_string(),
            kind: ArgKind::Ptr,
            offset,
            sreg,
        });
        SRegPair(sreg_id)
    }

    /// Declare a u32 argument. Returns the SReg.
    pub fn arg_u32(&mut self, name: &str) -> SReg {
        let sreg_id = self.next_sreg;
        self.next_sreg += 1;
        let sreg = SReg(sreg_id);
        let offset = self.kernarg_size;
        self.kernarg_size += 4;
        self.sreg_allocs.push(SRegAlloc {
            sreg,
            count: 1,
            alignment: Alignment::None,
        });
        self.args.push(KernArg {
            name: name.to_string(),
            kind: ArgKind::U32,
            offset,
            sreg,
        });
        sreg
    }

    /// Declare a f32 argument. Returns the SReg.
    pub fn arg_f32(&mut self, name: &str) -> SReg {
        let sreg_id = self.next_sreg;
        self.next_sreg += 1;
        let sreg = SReg(sreg_id);
        let offset = self.kernarg_size;
        self.kernarg_size += 4;
        self.sreg_allocs.push(SRegAlloc {
            sreg,
            count: 1,
            alignment: Alignment::None,
        });
        self.args.push(KernArg {
            name: name.to_string(),
            kind: ArgKind::F32,
            offset,
            sreg,
        });
        sreg
    }

    // ── Register allocation ──

    /// Allocate a single virtual VGPR.
    pub fn alloc_vreg(&mut self) -> VReg {
        let v = VReg(self.next_vreg);
        self.next_vreg += 1;
        self.vreg_allocs.push(VRegAlloc { vreg: v, count: 1, alignment: Alignment::None });
        v
    }

    /// Allocate N consecutive VGPRs with alignment.
    /// Returns VReg of the first register (the rest are VReg(first.0+1), etc.)
    pub fn alloc_vreg_array(&mut self, count: u32, align: Alignment) -> VReg {
        let v = VReg(self.next_vreg);
        self.next_vreg += count;
        self.vreg_allocs.push(VRegAlloc { vreg: v, count, alignment: align });
        v
    }

    /// Allocate a single virtual SGPR.
    pub fn alloc_sreg(&mut self) -> SReg {
        let s = SReg(self.next_sreg);
        self.next_sreg += 1;
        self.sreg_allocs.push(SRegAlloc { sreg: s, count: 1, alignment: Alignment::None });
        s
    }

    /// Allocate an SGPR pair (64-bit, 2-aligned).
    pub fn alloc_sreg_pair(&mut self) -> SRegPair {
        let s = SReg(self.next_sreg);
        self.next_sreg += 2;
        self.sreg_allocs.push(SRegAlloc { sreg: s, count: 2, alignment: Alignment::Align2 });
        SRegPair(s.0)
    }

    /// Set LDS size in bytes.
    pub fn set_lds_size(&mut self, size: u32) {
        self.lds_size = size;
    }

    /// Generate a unique label name.
    pub fn make_label(&mut self, prefix: &str) -> String {
        let id = self.label_counter;
        self.label_counter += 1;
        format!("{}_{}", prefix, id)
    }

    // ── Built-in register accessors ──

    /// Get VReg for hardware WORKITEM_ID_X (always v0).
    pub fn thread_id_x(&self) -> VReg { VReg(0) }

    /// Get SReg for hardware TGID.x (workgroup ID X).
    /// Hardware places this in s2, but we'll emit s_mov to capture it.
    pub fn workgroup_id_x(&self) -> SReg { SReg(u32::MAX - 0) }
    pub fn workgroup_id_y(&self) -> SReg { SReg(u32::MAX - 1) }
    pub fn workgroup_id_z(&self) -> SReg { SReg(u32::MAX - 2) }

    // ── Emitting operations ──

    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    // ── Convenience methods for common ops ──

    pub fn global_load(&mut self, dst: VReg, addr: VReg, width: Width, offset: i32) {
        self.ops.push(Op::GlobalLoad { dst, addr, width, offset });
    }

    pub fn global_store(&mut self, addr: VReg, src: VReg, width: Width, offset: i32) {
        self.ops.push(Op::GlobalStore { addr, src, width, offset });
    }

    pub fn lds_load(&mut self, dst: VReg, addr: VReg, width: Width, offset: u16) {
        self.ops.push(Op::LdsLoad { dst, addr, width, offset });
    }

    pub fn lds_store(&mut self, addr: VReg, src: VReg, width: Width, offset: u16) {
        self.ops.push(Op::LdsStore { addr, src, width, offset });
    }

    pub fn scalar_load(&mut self, dst: SReg, base: SRegPair, offset: u32, width: Width) {
        self.ops.push(Op::ScalarLoad { dst, base, offset, width });
    }

    pub fn v_add_f32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VAddF32 {
            dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1)
        });
    }

    pub fn v_mul_f32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VMulF32 {
            dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1)
        });
    }

    pub fn v_mul_f32_imm(&mut self, dst: VReg, src: VReg, scale: f32) {
        self.ops.push(Op::VMulF32 {
            dst, src0: Operand::VReg(src), src1: Operand::InlineFloat(scale)
        });
    }

    pub fn v_fma_f32(&mut self, dst: VReg, a: VReg, b: VReg, c: VReg) {
        self.ops.push(Op::VFmaF32 {
            dst,
            src0: Operand::VReg(a),
            src1: Operand::VReg(b),
            src2: Operand::VReg(c),
        });
    }

    pub fn v_mov(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VMov { dst, src: Operand::VReg(src) });
    }

    pub fn v_mov_imm(&mut self, dst: VReg, val: i32) {
        self.ops.push(Op::VMov { dst, src: Operand::InlineInt(val) });
    }

    pub fn v_mov_from_sgpr(&mut self, dst: VReg, src: SReg) {
        self.ops.push(Op::VMovFromSgpr { dst, src });
    }

    pub fn v_add_u32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VAddU32 {
            dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1)
        });
    }

    pub fn v_and_b32_imm(&mut self, dst: VReg, src: VReg, mask: u32) {
        self.ops.push(Op::VAndB32 {
            dst, src0: Operand::VReg(src), src1: Operand::InlineInt(mask as i32)
        });
    }

    pub fn v_lshlrev_b32(&mut self, dst: VReg, shift: u8, src: VReg) {
        self.ops.push(Op::VLshlrevB32 { dst, shift, src });
    }

    pub fn v_lshrrev_b32(&mut self, dst: VReg, shift: u8, src: VReg) {
        self.ops.push(Op::VLshrrevB32 { dst, shift, src });
    }

    pub fn v_mul_lo_u32(&mut self, dst: VReg, a: VReg, b: VReg) {
        self.ops.push(Op::VMulLoU32 { dst, src0: a, src1: b });
    }

    pub fn v_add_co(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VAddCo { dst, src0, src1 });
    }

    pub fn v_add_co_ci(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VAddCoCi { dst, src });
    }

    pub fn s_add_u32(&mut self, dst: SReg, src0: SReg, imm: i32) {
        self.ops.push(Op::SAddU32 { dst, src0, src1: SOperand::InlineInt(imm) });
    }

    pub fn s_mov_imm(&mut self, dst: SReg, val: i32) {
        self.ops.push(Op::SMov { dst, src: SOperand::InlineInt(val) });
    }

    pub fn s_lshl_b32(&mut self, dst: SReg, src: SReg, shift: u8) {
        self.ops.push(Op::SLshlB32 { dst, src, shift });
    }

    /// s_lshr_b32: scalar right shift
    pub fn s_lshr_b32(&mut self, dst: SReg, src: SReg, shift: u8) {
        self.ops.push(Op::SLshrB32 { dst, src, shift });
    }

    /// s_add_u32 with SGPR source (not immediate)
    pub fn s_add_u32_ss(&mut self, dst: SReg, src0: SReg, src1: SReg) {
        self.ops.push(Op::SAddU32 { dst, src0, src1: SOperand::SReg(src1) });
    }

    /// s_addc_u32: scalar add with carry (must follow s_add_u32)
    pub fn s_addc_u32(&mut self, dst: SReg, src0: SReg, src1: SReg) {
        self.ops.push(Op::SAddcU32 { dst, src0, src1: SOperand::SReg(src1) });
    }

    /// s_addc_u32 with immediate (for carry + 0)
    pub fn s_addc_u32_imm(&mut self, dst: SReg, src0: SReg, imm: i32) {
        self.ops.push(Op::SAddcU32 { dst, src0, src1: SOperand::InlineInt(imm) });
    }

    pub fn s_cmp_lt_u32(&mut self, src0: SReg, src1: SReg) {
        self.ops.push(Op::SCmpLtU32 { src0, src1 });
    }

    /// s_cmp_eq_u32: compare equal. Sets SCC = (src0 == src1).
    pub fn s_cmp_eq_u32_imm(&mut self, src0: SReg, imm: i32) {
        self.ops.push(Op::SCmpEqU32 { src0, src1: SOperand::InlineInt(imm) });
    }

    /// s_cmp_ge_u32: compare greater-or-equal. Sets SCC = (src0 >= src1).
    pub fn s_cmp_ge_u32(&mut self, src0: SReg, src1: SReg) {
        self.ops.push(Op::SCmpGeU32 { src0, src1 });
    }


    /// s_sub_u32: scalar subtraction dst = src0 - imm
    pub fn s_sub_u32(&mut self, dst: SReg, src0: SReg, imm: i32) {
        self.ops.push(Op::SSubU32 { dst, src0, src1: SOperand::InlineInt(imm) });
    }

    /// s_and_b32: scalar bitwise AND dst = src0 & src1
    pub fn s_and_b32(&mut self, dst: SReg, src0: SReg, src1: SReg) {
        self.ops.push(Op::SAndB32 { dst, src0, src1: SOperand::SReg(src1) });
    }

    /// s_add_i32: alias for s_add_u32 with SGPR sources (same encoding)
    pub fn s_add_i32(&mut self, dst: SReg, src0: SReg, src1: SReg) {
        self.ops.push(Op::SAddU32 { dst, src0, src1: SOperand::SReg(src1) });
    }

    /// s_mul_i32: scalar multiply dst = src0 * src1
    pub fn s_mul_i32(&mut self, dst: SReg, src0: SReg, src1: SReg) {
        self.ops.push(Op::SMulI32 { dst, src0, src1 });
    }

    /// v_max_f32: dst = max(src0, src1)
    pub fn v_max_f32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VMaxF32 { dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1) });
    }

    /// v_min_f32: dst = min(src0, src1)
    pub fn v_min_f32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VMinF32 { dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1) });
    }

    /// v_and_b32: dst = src0 & src1 (VGPR-VGPR)
    pub fn v_and_b32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VAndB32 { dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1) });
    }

    pub fn wmma_bf16_f32(&mut self, dst: VReg, a: VReg, b: VReg, c: VReg) {
        self.ops.push(Op::Wmma { dst, a, b, c, format: WmmaFormat::BF16_F32 });
    }

    pub fn label(&mut self, name: &str) {
        self.ops.push(Op::Label(name.to_string()));
    }

    pub fn branch_scc1(&mut self, target: &str) {
        self.ops.push(Op::BranchScc1(target.to_string()));
    }

    pub fn barrier(&mut self) {
        self.ops.push(Op::Barrier);
    }

    pub fn wait_vmcnt(&mut self, n: u8) {
        self.ops.push(Op::WaitVmcnt(n));
    }

    pub fn wait_lgkmcnt(&mut self, n: u8) {
        self.ops.push(Op::WaitLgkmcnt(n));
    }

    pub fn wait_vscnt(&mut self, n: u8) {
        self.ops.push(Op::WaitVscnt(n));
    }

    /// Clear VCC (s_mov_b32 vcc_lo, 0) — prevents carry residual from mask/cmp ops
    pub fn clear_vcc(&mut self) {
        self.ops.push(Op::ClearVcc);
    }

    pub fn endpgm(&mut self) {
        self.ops.push(Op::Endpgm);
    }

    pub fn raw_asm(&mut self, text: &str) {
        self.ops.push(Op::RawAsm(text.to_string()));
    }

    pub fn ds_swizzle(&mut self, dst: VReg, src: VReg, offset: u16) {
        self.ops.push(Op::DsSwizzle { dst, src, offset });
    }

    pub fn v_rsq_f32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VRsqF32 { dst, src });
    }

    /// Wave-level butterfly sum reduction across all 32 lanes.
    /// `val` is modified in-place to hold the sum. `tmp` is scratch.
    pub fn wave_reduce_add_f32(&mut self, val: VReg, tmp: VReg) {
        self.ops.push(Op::WaveReduceAddF32 { val, tmp });
    }

    /// Wave-level butterfly max reduction across all 32 lanes.
    /// `val` is modified in-place to hold the max. `tmp` is scratch.
    pub fn wave_reduce_max_f32(&mut self, val: VReg, tmp: VReg) {
        self.ops.push(Op::WaveReduceMaxF32 { val, tmp });
    }

    /// Pack two f32 values into bf16x2: dst = (bf16(src1) << 16) | bf16(src0)
    pub fn cvt_pk_bf16_f32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::CvtPkBf16F32 { dst, src0, src1 });
    }

    /// v_exp_f32: compute 2^x (NOT e^x! For e^x, pre-multiply by log2(e))
    pub fn v_exp_f32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VExpF32 { dst, src });
    }

    /// v_log_f32: compute log₂(x) (NOT ln(x)! For ln(x), post-multiply by ln(2))
    pub fn v_log_f32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VLog2F32 { dst, src });
    }

    /// v_rcp_f32: compute 1/x
    pub fn v_rcp_f32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VRcpF32 { dst, src });
    }

    /// v_xor_b32: bitwise XOR (e.g. sign-flip with 0x80000000)
    pub fn v_xor_b32(&mut self, dst: VReg, src0: Operand, src1: Operand) {
        self.ops.push(Op::VXorB32 { dst, src0, src1 });
    }

    /// v_sub_f32: dst = src0 - src1
    pub fn v_sub_f32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VSubF32 { dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1) });
    }

    /// v_cvt_f32_u32: convert unsigned int to float
    pub fn v_cvt_f32_u32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VCvtF32U32 { dst, src });
    }

    /// v_cvt_u32_f32: convert float to unsigned int (truncate)
    pub fn v_cvt_u32_f32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VCvtU32F32 { dst, src });
    }

    /// v_sub_u32: unsigned integer subtraction (no carry)
    pub fn v_sub_u32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VSubU32 { dst, src0: Operand::VReg(src0), src1: Operand::VReg(src1) });
    }

    /// v_cmp_ge_u32: set VCC where src0 >= src1
    pub fn v_cmp_ge_u32(&mut self, src0: Operand, src1: Operand) {
        self.ops.push(Op::VCmpGeU32 { src0, src1 });
    }

    /// v_cmp_gt_f32 vcc, src, 0 — set VCC where src > 0.0 (ReLU mask)
    pub fn v_cmp_gt_f32_imm0(&mut self, src: VReg) {
        self.ops.push(Op::VCmpGtF32Imm0 { src });
    }

    /// v_cndmask_b32: dst = VCC ? src_true : src_false
    pub fn v_cndmask_b32(&mut self, dst: VReg, src_false: Operand, src_true: Operand) {
        self.ops.push(Op::VCndmaskB32 { dst, src_false, src_true });
    }

    // ── LDS (Local Data Share) operations ──

    /// ds_store_b16: store low 16 bits of src to LDS at vaddr + offset
    pub fn ds_store_b16(&mut self, vaddr: VReg, src: VReg, offset: u16) {
        self.ops.push(Op::DsStoreB16 { vaddr, src, offset });
    }
    /// ds_store_b32: store 32-bit src to LDS at vaddr + offset
    pub fn ds_store_b32(&mut self, vaddr: VReg, src: VReg, offset: u16) {
        self.ops.push(Op::DsStoreB32 { vaddr, src, offset });
    }
    /// ds_store_b64: store v[src:src+1] (64-bit) to LDS at vaddr + offset
    pub fn ds_store_b64(&mut self, vaddr: VReg, src: VReg, offset: u16) {
        self.ops.push(Op::DsStoreB64 { vaddr, src, offset });
    }
    /// ds_store_b128: store v[src:src+3] (128-bit) to LDS at vaddr + offset
    pub fn ds_store_b128(&mut self, vaddr: VReg, src: VReg, offset: u16) {
        self.ops.push(Op::DsStoreB128 { vaddr, src, offset });
    }

    /// ds_load_b32: load 32-bit from LDS at vaddr + offset into dst
    pub fn ds_load_b32(&mut self, dst: VReg, vaddr: VReg, offset: u16) {
        self.ops.push(Op::DsLoadB32 { dst, vaddr, offset });
    }
    /// ds_load_b64: load 64-bit from LDS into v[dst:dst+1]
    pub fn ds_load_b64(&mut self, dst: VReg, vaddr: VReg, offset: u16) {
        self.ops.push(Op::DsLoadB64 { dst, vaddr, offset });
    }
    /// ds_load_b128: load 128-bit from LDS into v[dst:dst+3]
    pub fn ds_load_b128(&mut self, dst: VReg, vaddr: VReg, offset: u16) {
        self.ops.push(Op::DsLoadB128 { dst, vaddr, offset });
    }
    /// ds_load_u16: load 16-bit unsigned from LDS, zero-extend to 32-bit
    pub fn ds_load_u16(&mut self, dst: VReg, vaddr: VReg, offset: u16) {
        self.ops.push(Op::DsLoadU16 { dst, vaddr, offset });
    }
    /// ds_load_u16_d16: load 16-bit into low 16 bits of dst (bf16 column tearing)
    pub fn ds_load_u16_d16(&mut self, dst: VReg, vaddr: VReg, offset: u16) {
        self.ops.push(Op::DsLoadU16D16 { dst, vaddr, offset });
    }
    /// ds_load_u16_d16_hi: load 16-bit into high 16 bits of dst (bf16 column tearing)
    pub fn ds_load_u16_d16_hi(&mut self, dst: VReg, vaddr: VReg, offset: u16) {
        self.ops.push(Op::DsLoadU16D16Hi { dst, vaddr, offset });
    }

    /// s_barrier: synchronize all waves in the workgroup
    pub fn s_barrier(&mut self) {
        self.ops.push(Op::SBarrier);
    }

    /// s_cbranch_scc0: branch if SCC == 0
    pub fn branch_scc0(&mut self, target: &str) {
        self.ops.push(Op::BranchScc0(target.to_string()));
    }

    /// s_cbranch_vccz: branch if VCC == 0
    pub fn branch_vccz(&mut self, target: &str) {
        self.ops.push(Op::BranchVccz(target.to_string()));
    }

    /// v_or_b32
    pub fn v_or_b32(&mut self, dst: VReg, src0: Operand, src1: Operand) {
        self.ops.push(Op::VOrB32 { dst, src0, src1 });
    }

    /// v_sqrt_f32
    pub fn v_sqrt_f32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VSqrtF32 { dst, src });
    }

    /// v_cmp_gt_u32 vcc, src, imm
    pub fn v_cmp_gt_u32_imm(&mut self, src: VReg, imm: u32) {
        self.ops.push(Op::VCmpGtU32Imm { src, imm });
    }

    /// v_cmp_ge_i32 vcc, src0, src1
    pub fn v_cmp_ge_i32(&mut self, src0: VReg, src1: VReg) {
        self.ops.push(Op::VCmpGeI32 { src0, src1 });
    }

    /// global_atomic_add_f32 (fire-and-forget, no return)
    pub fn global_atomic_add_f32(&mut self, addr: VReg, src: VReg, offset: i32) {
        self.ops.push(Op::GlobalAtomicAddF32 { addr, src, offset });
    }

    /// v_permlanex16_b32: swap lane L with lane L XOR 16
    pub fn v_permlanex16(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VPermlanex16B32 { dst, src });
    }

    /// v_and_or_b32: dst = (src0 & literal) | src2
    pub fn v_and_or_b32(&mut self, dst: VReg, src0: VReg, literal: u32, src2: VReg) {
        self.ops.push(Op::VAndOrB32 { dst, src0, literal, src2 });
    }

    /// v_add_co_u32: low 32-bit add with carry out to VCC
    pub fn v_add_co_u32(&mut self, dst: VReg, src0: VReg, src1: VReg) {
        self.ops.push(Op::VAddCOU32 { dst, src0, src1 });
    }

    /// v_add_co_ci_u32: high 32-bit add with carry in from VCC
    pub fn v_add_co_ci_u32(&mut self, dst: VReg, src: VReg) {
        self.ops.push(Op::VAddCCU32 { dst, src });
    }

    /// 64-bit address add: addr[lo:hi] += offset (modifies addr in place)
    pub fn addr64_add(&mut self, addr_lo: VReg, addr_hi: VReg, offset: VReg) {
        self.v_add_co_u32(addr_lo, addr_lo, offset);
        self.v_add_co_ci_u32(addr_hi, addr_hi);
    }
    /// C-layout → A-operand transpose (8 WMMA C-regs → 8 bf16x2 A-regs)
    ///
    /// Uses v_permlanex16 (SWAP16) + v_cndmask + bf16 packing.
    /// Pure VALU — no LDS, no barrier.
    ///
    /// - src: 8-aligned VReg block (WMMA C-layout f32)
    /// - dst: 8-aligned VReg block (A-operand bf16x2)
    /// - tmp: 3 temp VRegs (val_even, val_odd, pack_tmp)
    pub fn reg_transpose_c_to_ab(&mut self, dst: VReg, src: VReg, tmp: VReg) {
        // VCC = 1 for lanes 0-15 (lower half of wave)
        // CRITICAL: gfx11 binary v_cmp_gt_u32_imm(vsrc,16) encodes immediate as SRC0:
        //   VCC = (16 > vsrc) = (vsrc < 16) → VCC=1 for lanes 0-15 ✓
        // But T0 text assembly "v_cmp_gt_u32 vcc_lo, v{s}, 16" means:
        //   VCC = (v{s} > 16) → VCC=1 for lanes 17-31 ✗ (INVERTED!)
        // Fix: use v_cmp_lt_u32 which gives VCC = (lane_id < 16) ✓
        // Also: must use lane_id (tid & 31), not full thread ID (0-511).
        let lane_tmp = VReg(tmp.0 + 3);
        self.v_and_b32_imm(lane_tmp, VReg(0), 31); // lane_id = tid & 31
        self.v_cmp_lt_u32(
            Operand::VReg(lane_tmp),
            Operand::InlineInt(16),
        ); // VCC = (lane_id < 16) → VCC=1 for lanes 0-15

        // Phase 1: permlanex16 (swap half-waves)
        for i in 0..8u32 {
            self.v_permlanex16(VReg(dst.0 + i), VReg(src.0 + i));
        }

        // Phase 2: merge + bf16 pack
        let val_e = tmp;
        let val_o = VReg(tmp.0 + 1);
        let pack_tmp_r = VReg(tmp.0 + 2);
        for i in 0..8u32 {
            let s = VReg(src.0 + i);
            let swiz = VReg(dst.0 + i);
            let d = VReg(dst.0 + i);
            self.ops.push(Op::VCndmaskB32 {
                dst: val_e, src_false: Operand::VReg(swiz), src_true: Operand::VReg(s)
            });
            self.ops.push(Op::VCndmaskB32 {
                dst: val_o, src_false: Operand::VReg(s), src_true: Operand::VReg(swiz)
            });
            self.v_lshrrev_b32(pack_tmp_r, 16, val_e);
            self.v_and_or_b32(d, val_o, 0xFFFF0000, pack_tmp_r);
        }
    }

    /// Compare: set VCC bitmask where src0 < src1 (unsigned).
    pub fn v_cmp_lt_u32(&mut self, src0: Operand, src1: Operand) {
        self.ops.push(Op::VCmpLtU32 { src0, src1 });
    }

    /// Save EXEC to dst, then EXEC &= VCC. Returns the saved EXEC register.
    pub fn save_exec(&mut self, dst: SReg) {
        self.ops.push(Op::SaveExec { dst });
    }

    /// Restore EXEC from saved SGPR (unmask all lanes).
    pub fn restore_exec(&mut self, src: SReg) {
        self.ops.push(Op::RestoreExec { src });
    }

    /// Begin a bounds-checked block: mask out lanes where `global_id >= n_elems`.
    /// Returns the saved EXEC register (pass to `bounds_check_end`).
    pub fn bounds_check_begin(&mut self, global_id: VReg, n_elems: VReg) -> SReg {
        let saved = self.alloc_sreg();
        self.v_cmp_lt_u32(
            Operand::VReg(global_id),
            Operand::VReg(n_elems),
        );
        self.save_exec(saved);
        saved
    }

    /// End a bounds-checked block: restore EXEC to unmask all lanes.
    pub fn bounds_check_end(&mut self, saved: SReg) {
        self.restore_exec(saved);
    }

    // ── Hardware register access ──

    /// Copy TGID.x (workgroup ID X) into a virtual SGPR.
    pub fn capture_tgid_x(&mut self, dst: SReg) {
        self.ops.push(Op::CaptureTgid { dst, axis: 0 });
    }

    /// Copy TGID.y (workgroup ID Y) into a virtual SGPR.
    pub fn capture_tgid_y(&mut self, dst: SReg) {
        self.ops.push(Op::CaptureTgid { dst, axis: 1 });
    }

    /// Copy TGID.z (workgroup ID Z) into a virtual SGPR.
    pub fn capture_tgid_z(&mut self, dst: SReg) {
        self.ops.push(Op::CaptureTgid { dst, axis: 2 });
    }

    /// Compute global thread ID = TGID.x * wg_size + WORKITEM_ID_X.
    /// Returns VReg holding the result. Clobbers hardware s2.
    pub fn compute_global_id_x(&mut self, wg_size: u32) -> VReg {
        let global_id = self.alloc_vreg();
        self.ops.push(Op::ComputeGlobalIdX { dst: global_id, wg_size });
        global_id
    }

    // ── Kernarg load prologue helper ──

    /// Emit scalar loads for all declared kernel arguments.
    /// Loads from s[0:1] (kernarg_segment_ptr) at each arg's offset.
    pub fn emit_arg_loads(&mut self) {
        for arg in &self.args {
            let width = match arg.kind {
                ArgKind::Ptr => Width::B64,
                ArgKind::U32 | ArgKind::F32 => Width::B32,
            };
            self.ops.push(Op::ScalarLoad {
                dst: arg.sreg,
                base: SRegPair(u32::MAX - 10), // will be replaced with s[0:1] in emitter
                offset: arg.offset,
                width,
            });
        }
        self.ops.push(Op::WaitLgkmcnt(0));
    }

    // ══════════════════════════════════════════════════════════════════
    // Compilation
    // ══════════════════════════════════════════════════════════════════

    /// Compile this kernel to assembly text (for debugging/inspection).
    pub fn to_assembly(&self, target: Target) -> Result<String, String> {
        let alloc = self.allocate_registers();
        let mut emitter = AsmEmitter::new();
        emitter.emit_kernel(
            &self.name,
            &self.ops,
            &alloc,
            target,
            self.kernarg_size,
            self.lds_size,
        );
        Ok(emitter.finish())
    }

    /// Compile this kernel to a GPU code object (ELF binary).
    pub fn compile(&self, target: Target) -> Result<Vec<u8>, String> {
        let asm_text = self.to_assembly(target)?;
        llvm_assemble(&asm_text, target, &self.name)
    }

    /// Internal: run register allocation.
    fn allocate_registers(&self) -> RegAlloc {
        regalloc::allocate(&self.vreg_allocs, &self.sreg_allocs, &self.ops)
    }
}

// ============================================================================
// LLVM assembly pipeline
// ============================================================================

/// Assemble GCN text → .o → .hsaco via LLVM/clang + ld.lld
fn llvm_assemble(asm_text: &str, target: Target, name: &str) -> Result<Vec<u8>, String> {
    use std::fs;

    let temp_dir = std::env::temp_dir();
    let asm_path = temp_dir.join(format!("t0_{}.s", name));
    let obj_path = temp_dir.join(format!("t0_{}.o", name));
    let co_path = temp_dir.join(format!("t0_{}.hsaco", name));

    // Write assembly
    fs::write(&asm_path, asm_text)
        .map_err(|e| format!("Failed to write .s file: {}", e))?;

    // Find LLVM tools
    let llvm_bin = find_llvm_bin()?;

    // clang: assemble .s → .o
    let clang = format!("{}/clang", llvm_bin);
    let clang_result = Command::new(&clang)
        .args([
            "-x", "assembler",
            "-target", "amdgcn-amd-amdhsa",
            &format!("-mcpu={}", target.mcpu_str()),
            "-c",
            &asm_path.to_string_lossy(),
            "-o",
            &obj_path.to_string_lossy(),
        ])
        .output()
        .map_err(|e| format!("Failed to run clang: {}", e))?;

    if !clang_result.status.success() {
        let stderr = String::from_utf8_lossy(&clang_result.stderr);
        // Include assembly source for debugging
        return Err(format!(
            "clang assembly failed:\n{}\n\n--- Assembly source ---\n{}",
            stderr, asm_text
        ));
    }

    // ld.lld: link .o → .hsaco (shared library)
    let lld = format!("{}/ld.lld", llvm_bin);
    let link_result = Command::new(&lld)
        .args([
            "--shared",
            &obj_path.to_string_lossy(),
            "-o",
            &co_path.to_string_lossy(),
        ])
        .output()
        .map_err(|e| format!("Failed to run ld.lld: {}", e))?;

    if !link_result.status.success() {
        return Err(format!(
            "ld.lld linking failed: {}",
            String::from_utf8_lossy(&link_result.stderr)
        ));
    }

    // Read result
    let co_bytes = fs::read(&co_path)
        .map_err(|e| format!("Failed to read .hsaco: {}", e))?;

    // Cleanup
    let _ = fs::remove_file(&asm_path);
    let _ = fs::remove_file(&obj_path);
    let _ = fs::remove_file(&co_path);

    Ok(co_bytes)
}

/// Find LLVM binary directory from ROCm installation.
fn find_llvm_bin() -> Result<String, String> {
    let candidates = [
        "/opt/rocm-7.1.1/llvm/bin",
        "/opt/rocm-7.1.1/bin",
        "/opt/rocm/llvm/bin",
        "/opt/rocm/bin",
    ];
    for path in &candidates {
        let clang = format!("{}/clang", path);
        if std::path::Path::new(&clang).exists() {
            return Ok(path.to_string());
        }
    }
    Err("LLVM/ROCm installation not found. \
         Checked: /opt/rocm-7.1.1/llvm/bin, /opt/rocm/llvm/bin".to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_kernel_assembly() {
        let mut k = T0Kernel::new("test_nop");
        k.endpgm();

        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("s_endpgm"));
        assert!(asm.contains(".amdhsa_kernel test_nop"));
        assert!(asm.contains("gfx1100"));
        eprintln!("--- Generated assembly ---\n{}", asm);
    }

    #[test]
    fn test_elementwise_scale_kernel() {
        // Build: y[i] = x[i] * scale
        let mut k = T0Kernel::new("t0_scale");

        // Args
        let x_ptr = k.arg_ptr("x");
        let y_ptr = k.arg_ptr("y");
        let scale_arg = k.arg_f32("scale");
        let n_arg = k.arg_u32("n");

        // Load all args from kernarg segment
        k.emit_arg_loads();

        // tid = v0 (WORKITEM_ID_X), compute global_id
        let tid = k.alloc_vreg();
        k.v_and_b32_imm(tid, VReg(0), 31);

        // Load x[tid]
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        let val = k.alloc_vreg();

        // addr = x_ptr + tid * 4
        let offset = k.alloc_vreg();
        k.v_lshlrev_b32(offset, 2, tid);
        k.v_mov_from_sgpr(addr, SReg(x_ptr.0));
        k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(x_ptr.0 + 1));
        k.v_add_co(addr, addr, offset);
        k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));

        k.global_load(val, addr, Width::B32, 0);
        k.wait_vmcnt(0);

        // val *= scale
        let sv = k.alloc_vreg();
        k.v_mov_from_sgpr(sv, SReg(scale_arg.0));
        k.v_mul_f32(val, val, sv);

        // Store to y[tid]
        let yaddr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(yaddr, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(yaddr.0 + 1), SReg(y_ptr.0 + 1));
        k.v_add_co(yaddr, yaddr, offset);
        k.v_add_co_ci(VReg(yaddr.0 + 1), VReg(yaddr.0 + 1));
        k.global_store(yaddr, val, Width::B32, 0);

        k.wait_vscnt(0);
        k.endpgm();

        let asm = k.to_assembly(Target::GFX1100).unwrap();
        assert!(asm.contains("global_load_b32"));
        assert!(asm.contains("v_mul_f32"));
        assert!(asm.contains("global_store_b32"));
        eprintln!("--- Scale kernel assembly ---\n{}", asm);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_compile_to_elf() {
        let mut k = T0Kernel::new("t0_nop_elf");
        k.endpgm();

        let elf = k.compile(Target::GFX1100).unwrap();
        assert!(elf.len() > 0);
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        eprintln!("T0 compiled ELF: {} bytes", elf.len());
    }
}
