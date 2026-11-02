// Port of CodeGen/RegisterAllocators/{AllocationResult,StackAllocator,RegisterMasks}.cs

use crate::ir::types::OperandType;

/// C# readonly struct AllocationResult  output of register allocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocationResult {
    pub int_used_registers: u32,
    pub vec_used_registers: u32,
    pub spill_region_size: i32,
}

impl AllocationResult {
    pub const fn new(int_used: u32, vec_used: u32, spill: i32) -> Self {
        Self {
            int_used_registers: int_used,
            vec_used_registers: vec_used,
            spill_region_size: spill,
        }
    }
}

/// C# class StackAllocator  sequential bump allocator for stack slots.
#[derive(Debug, Clone, Default)]
pub struct StackAllocator {
    offset: i32,
}

impl StackAllocator {
    pub fn new() -> Self {
        Self { offset: 0 }
    }

    pub fn allocate(&mut self, typ: OperandType) -> i32 {
        self.allocate_bytes(typ.size_bytes() as i32)
    }

    pub fn allocate_bytes(&mut self, size: i32) -> i32 {
        let offs = self.offset;
        self.offset += size;
        offs
    }

    pub fn total_size(&self) -> i32 {
        self.offset
    }
}

/// C# struct RegisterMasks  bitmasks of available/preserved registers.
#[derive(Debug, Clone, Copy)]
pub struct RegisterMasks {
    pub int_available: u32,
    pub vec_available: u32,
    pub int_caller_saved: u32,
    pub vec_caller_saved: u32,
    pub int_callee_saved: u32,
    pub vec_callee_saved: u32,
    pub registers_count: i32,
}

impl RegisterMasks {
    pub fn new(registers_count: i32) -> Self {
        use crate::codegen::x86_64::calling_convention::*;
        Self {
            int_available: get_int_available_registers() as u32,
            vec_available: get_vec_available_registers() as u32,
            int_caller_saved: get_int_caller_saved_registers() as u32,
            vec_caller_saved: get_vec_caller_saved_registers() as u32,
            int_callee_saved: get_int_callee_saved_registers() as u32,
            vec_callee_saved: get_vec_callee_saved_registers() as u32,
            registers_count,
        }
    }
}
