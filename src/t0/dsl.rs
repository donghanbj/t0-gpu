//! T0 DSL 类型定义
//!
//! 仅保留 CompiledKernel 和关联类型，供 compile_via_ssa 和 Ignis 使用。
//! 原 dsl.rs 的 Op/KernelBuilder/DType/lower 函数已删除（2026-03-25）。

/// 编译后的内核 — ELF + 元数据
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    /// HSA ELF code object bytes
    pub elf: Vec<u8>,
    /// Total kernarg buffer size in bytes
    pub kernarg_size: usize,
    /// Workgroup size [x, y, z]
    pub workgroup_size: [u32; 3],
    /// LDS size in bytes
    pub lds_size: u32,
    /// Kernel name
    pub name: String,
    /// Kernel argument metadata
    pub args: Vec<KernArgMeta>,
}

/// 内核参数元数据
#[derive(Clone, Debug)]
pub struct KernArgMeta {
    pub name: String,
    pub kind: KernArgType,
    pub offset: usize,
}

/// 参数类型
#[derive(Clone, Debug, PartialEq)]
pub enum KernArgType {
    Ptr,
    U32,
    F32,
}

/// 数据类型
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F32,
    BF16,
    U32,
}

impl DType {
    /// Size in bytes
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::U32 => 4,
        }
    }

    /// Name string
    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::U32 => "u32",
        }
    }
}
