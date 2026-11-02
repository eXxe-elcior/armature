// Port of Instructions/InstEmitMemoryHelper.cs (A64 paths).
// Generic memory operations use fixed-width accessors. Page-table constants carry
// PTC relocation symbols; symbols are inert outside relocatable compilation.

use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::block::BlockFrequency;
use crate::ir::operand::OperandIndex;
use crate::ir::types::{Comparison, OperandType};
use crate::registers::{GpRegisterIndex, VecRegisterIndex};

use super::context::ArmEmitterCtx;
use super::helpers::{get_int_or_zr, get_vec, set_int_or_zr};
use crate::instructions::native_function::NativeFn;

const PAGE_BITS: i32 = 12;

/// ARMATURE_MEM_DEBUG=1 → route host memory translation through dbg_translate.
fn mem_debug_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ARMATURE_MEM_DEBUG").ok().as_deref() == Some("1"))
}
const PAGE_MASK: i32 = (1 << PAGE_BITS) - 1;

// C# Ptc.PageTableSymbol = new(SymbolType.Special, 1).
fn page_table_symbol() -> crate::ir::types::Symbol {
    crate::ir::types::Symbol {
        typ: crate::ir::types::SymbolType::Special,
        value: 1,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Extension {
    Zx,
    Sx32,
    Sx64,
}

// C# Const(x.Type, value)  a constant typed to match another operand.
fn const_typed(ctx: &mut ArmEmitterCtx, typ: OperandType, value: i64) -> OperandIndex {
    match typ {
        OperandType::I32 => ctx.ctx.operands.constant_i32(value as i32),
        OperandType::I64 => ctx.ctx.operands.constant_i64(value),
        _ => panic!("const_typed: unexpected type {typ:?}"),
    }
}

pub fn emit_load_zx(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    emit_load(ctx, address, Extension::Zx, rt, size);
}

pub fn emit_load_sx32(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    emit_load(ctx, address, Extension::Sx32, rt, size);
}

pub fn emit_load_sx64(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    emit_load(ctx, address, Extension::Sx64, rt, size);
}

fn emit_load(ctx: &mut ArmEmitterCtx, address: OperandIndex, ext: Extension, rt: u8, size: i32) {
    let is_simd = is_simd(ctx);

    if size as u32 > if is_simd { 4 } else { 3 } {
        panic!("emit_load: size {size} out of range");
    }

    if is_simd {
        let vector = ctx.vector_zero();
        emit_read_vector(ctx, address, vector, rt, 0, size);
    } else {
        emit_read_int(ctx, address, rt, size);
    }

    if !is_simd {
        let mut value = get_int_or_zr(ctx, GpRegisterIndex::from(rt));

        if matches!(ext, Extension::Sx32 | Extension::Sx64) {
            let dest_type = if ext == Extension::Sx64 {
                OperandType::I64
            } else {
                OperandType::I32
            };

            value = match size {
                0 => ctx.sign_extend8(dest_type, value),
                1 => ctx.sign_extend16(dest_type, value),
                2 => ctx.sign_extend32(dest_type, value),
                _ => value,
            };
        }

        set_int_or_zr(ctx, GpRegisterIndex::from(rt), value);
    }
}

/// Used by the Ld__Vss single-structure loads.
pub fn emit_load_simd(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    vector: OperandIndex,
    rt: u8,
    elem: i32,
    size: i32,
) {
    emit_read_vector(ctx, address, vector, rt, elem, size);
}

pub fn emit_store(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    let is_simd = is_simd(ctx);

    if size as u32 > if is_simd { 4 } else { 3 } {
        panic!("emit_store: size {size} out of range");
    }

    if is_simd {
        emit_write_vector(ctx, address, rt, 0, size);
    } else {
        emit_write_int(ctx, address, rt, size);
    }
}

/// Used by the St__Vss single-structure stores.
pub fn emit_store_simd(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    rt: u8,
    elem: i32,
    size: i32,
) {
    emit_write_vector(ctx, address, rt, elem, size);
}

// C#: CurrOp is IOpCodeSimd and not (OpCodeSimdMemMs or OpCodeSimdMemSs).
// Only the SIMD memory kinds can reach these helpers; Ms/Ss go through
// emit_load_simd/emit_store_simd instead.
fn is_simd(ctx: &ArmEmitterCtx) -> bool {
    matches!(
        ctx.curr_op().kind,
        OpCodeKind::SimdMemImm { .. }
            | OpCodeKind::SimdMemReg { .. }
            | OpCodeKind::SimdMemPair { .. }
            | OpCodeKind::SimdMemLit { .. }
    )
}

fn emit_read_int(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    let lbl_slow_path = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    let phys_addr = emit_pt_pointer_load(ctx, address, Some(lbl_slow_path), false, size);

    let value = match size {
        0 => ctx.load8(phys_addr),
        1 => ctx.load16(phys_addr),
        2 => ctx.load(OperandType::I32, phys_addr),
        3 => ctx.load(OperandType::I64, phys_addr),
        _ => panic!("emit_read_int: size {size} out of range"),
    };

    set_int_or_zr(ctx, GpRegisterIndex::from(rt), value);

    if !ctx.memory.manager_type().is_host_mapped_or_tracked() {
        ctx.branch(lbl_end);

        ctx.mark_label(lbl_slow_path, BlockFrequency::Cold);

        emit_read_int_fallback(ctx, address, rt, size);

        ctx.mark_label(lbl_end, BlockFrequency::Default);
    }
}

pub fn emit_read_int_aligned(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    size: i32,
) -> OperandIndex {
    if size as u32 > 4 {
        panic!("emit_read_int_aligned: size {size} out of range");
    }

    let phys_addr = emit_pt_pointer_load(ctx, address, None, false, size);

    match size {
        0 => ctx.load8(phys_addr),
        1 => ctx.load16(phys_addr),
        2 => ctx.load(OperandType::I32, phys_addr),
        3 => ctx.load(OperandType::I64, phys_addr),
        _ => ctx.load(OperandType::V128, phys_addr),
    }
}

fn emit_read_vector(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    vector: OperandIndex,
    rt: u8,
    elem: i32,
    size: i32,
) {
    let lbl_slow_path = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    let phys_addr = emit_pt_pointer_load(ctx, address, Some(lbl_slow_path), false, size);

    let value = match size {
        0 => {
            let loaded = ctx.load8(phys_addr);
            ctx.vector_insert8(vector, loaded, elem)
        }
        1 => {
            let loaded = ctx.load16(phys_addr);
            ctx.vector_insert16(vector, loaded, elem)
        }
        2 => {
            let loaded = ctx.load(OperandType::I32, phys_addr);
            ctx.vector_insert(vector, loaded, elem)
        }
        3 => {
            let loaded = ctx.load(OperandType::I64, phys_addr);
            ctx.vector_insert(vector, loaded, elem)
        }
        4 => ctx.load(OperandType::V128, phys_addr),
        _ => panic!("emit_read_vector: size {size} out of range"),
    };

    let vec_reg = get_vec(ctx, VecRegisterIndex::from(rt));
    ctx.copy_to(vec_reg, value);

    if !ctx.memory.manager_type().is_host_mapped_or_tracked() {
        ctx.branch(lbl_end);

        ctx.mark_label(lbl_slow_path, BlockFrequency::Cold);

        emit_read_vector_fallback(ctx, address, vector, rt, elem, size);

        ctx.mark_label(lbl_end, BlockFrequency::Default);
    }
}

fn emit_write_int(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    let lbl_slow_path = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    let phys_addr = emit_pt_pointer_load(ctx, address, Some(lbl_slow_path), true, size);

    let mut value = get_int_or_zr(ctx, GpRegisterIndex::from(rt));

    if size < 3 && ctx.ctx.operands.get(value).typ == OperandType::I64 {
        value = ctx.convert_i64_to_i32(value);
    }

    match size {
        0 => ctx.store8(phys_addr, value),
        1 => ctx.store16(phys_addr, value),
        2 | 3 => ctx.store(phys_addr, value),
        _ => panic!("emit_write_int: size {size} out of range"),
    }

    if !ctx.memory.manager_type().is_host_mapped_or_tracked() {
        ctx.branch(lbl_end);

        ctx.mark_label(lbl_slow_path, BlockFrequency::Cold);

        emit_write_int_fallback(ctx, address, rt, size);

        ctx.mark_label(lbl_end, BlockFrequency::Default);
    }
}

pub fn emit_write_int_aligned(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    mut value: OperandIndex,
    size: i32,
) {
    if size as u32 > 4 {
        panic!("emit_write_int_aligned: size {size} out of range");
    }

    let phys_addr = emit_pt_pointer_load(ctx, address, None, true, size);

    if size < 3 && ctx.ctx.operands.get(value).typ == OperandType::I64 {
        value = ctx.convert_i64_to_i32(value);
    }

    if size == 0 {
        ctx.store8(phys_addr, value);
    } else if size == 1 {
        ctx.store16(phys_addr, value);
    } else {
        ctx.store(phys_addr, value);
    }
}

fn emit_write_vector(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, elem: i32, size: i32) {
    let lbl_slow_path = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    let phys_addr = emit_pt_pointer_load(ctx, address, Some(lbl_slow_path), true, size);

    let value = get_vec(ctx, VecRegisterIndex::from(rt));

    match size {
        0 => {
            let extracted = ctx.vector_extract8(value, elem);
            ctx.store8(phys_addr, extracted);
        }
        1 => {
            let extracted = ctx.vector_extract16(value, elem);
            ctx.store16(phys_addr, extracted);
        }
        2 => {
            let extracted = ctx.vector_extract(OperandType::I32, value, elem);
            ctx.store(phys_addr, extracted);
        }
        3 => {
            let extracted = ctx.vector_extract(OperandType::I64, value, elem);
            ctx.store(phys_addr, extracted);
        }
        4 => ctx.store(phys_addr, value),
        _ => panic!("emit_write_vector: size {size} out of range"),
    }

    if !ctx.memory.manager_type().is_host_mapped_or_tracked() {
        ctx.branch(lbl_end);

        ctx.mark_label(lbl_slow_path, BlockFrequency::Cold);

        emit_write_vector_fallback(ctx, address, rt, elem, size);

        ctx.mark_label(lbl_end, BlockFrequency::Default);
    }
}

pub fn emit_pt_pointer_load(
    ctx: &mut ArmEmitterCtx,
    mut address: OperandIndex,
    lbl_slow_path: Option<OperandIndex>,
    write: bool,
    size: i32,
) -> OperandIndex {
    // Debug: route every host-mapped/host-tracked translation through a native
    // logger that recomputes va→host and prints the page-table entry. Lets us
    // see whether a wild store comes from an unmapped (entry=0) page or a bad
    // base. ARMATURE_MEM_DEBUG=1.
    if mem_debug_enabled() && ctx.memory.manager_type().is_host_mapped_or_tracked() {
        let va = if ctx.ctx.operands.get(address).typ == OperandType::I32 {
            ctx.zero_extend32(OperandType::I64, address)
        } else {
            address
        };
        return ctx
            .call(NativeFn::DbgTranslate, &[va])
            .expect("DbgTranslate returns host address");
    }

    if ctx.memory.manager_type().is_host_mapped() {
        return emit_host_mapped_pointer(ctx, address);
    } else if ctx.memory.manager_type().is_host_tracked() {
        if ctx.ctx.operands.get(address).typ == OperandType::I32 {
            address = ctx.zero_extend32(OperandType::I64, address);
        }

        if ctx.memory.manager_type() == crate::memory::MemoryManagerType::HostTracked {
            let mask = ctx
                .ctx
                .operands
                .constant_i64((u64::MAX >> (64 - ctx.memory.address_space_bits())) as i64);
            address = ctx.bitwise_and(address, mask);
        }

        let pt_base = ctx
            .ctx
            .operands
            .constant_i64_with_symbol(ctx.memory.page_table_pointer() as i64, page_table_symbol());

        let page_bits = ctx.ctx.operands.constant_i32(PAGE_BITS);
        let pt_offset = ctx.shift_right_ui(address, page_bits);

        let three = ctx.ctx.operands.constant_i32(3);
        let shifted = ctx.shift_left(pt_offset, three);
        let entry_addr = ctx.add(pt_base, shifted);
        let host_offset = ctx.load(OperandType::I64, entry_addr);
        return ctx.add(address, host_offset);
    }

    let pt_level_bits = ctx.memory.address_space_bits() - PAGE_BITS;
    let pt_level_size: i32 = 1 << pt_level_bits;
    let pt_level_mask = pt_level_size - 1;

    let addr_rotated = if size != 0 {
        let size_op = ctx.ctx.operands.constant_i32(size);
        ctx.rotate_right(address, size_op)
    } else {
        address
    };
    let shift_amount = ctx.ctx.operands.constant_i32(PAGE_BITS - size);
    let mut addr_shifted = ctx.shift_right_ui(addr_rotated, shift_amount);

    let mut pte = ctx
        .ctx
        .operands
        .constant_i64_with_symbol(ctx.memory.page_table_pointer() as i64, page_table_symbol());

    let addr_shifted_type = ctx.ctx.operands.get(addr_shifted).typ;
    let level_mask = const_typed(ctx, addr_shifted_type, pt_level_mask as i64);
    let mut pte_offset = ctx.bitwise_and(addr_shifted, level_mask);

    if ctx.ctx.operands.get(pte_offset).typ == OperandType::I32 {
        pte_offset = ctx.zero_extend32(OperandType::I64, pte_offset);
    }

    let three = ctx.ctx.operands.constant_i32(3);
    let pte_shifted = ctx.shift_left(pte_offset, three);
    let pte_addr = ctx.add(pte, pte_shifted);
    pte = ctx.load(OperandType::I64, pte_addr);

    if ctx.ctx.operands.get(addr_shifted).typ == OperandType::I32 {
        addr_shifted = ctx.zero_extend32(OperandType::I64, addr_shifted);
    }

    // If the VA is out of range, or not aligned to the access size, force
    // the PTE to 0 by masking it.
    let neg_level_size = ctx.ctx.operands.constant_i64(-(pt_level_size as i64));
    let range_check = ctx.add(addr_shifted, neg_level_size);
    let sixty_three = ctx.ctx.operands.constant_i32(63);
    let range_mask = ctx.shift_right_si(range_check, sixty_three);
    pte = ctx.bitwise_and(pte, range_mask);

    if let Some(lbl_slow_path) = lbl_slow_path {
        if write {
            let zero = ctx.ctx.operands.constant_i64(0);
            ctx.branch_if(
                lbl_slow_path,
                pte,
                zero,
                Comparison::LessOrEqual,
                BlockFrequency::Default,
            );
            // Ignore any software protection bits. (they are still used by
            // the host-side memory access)
            let prot_mask = ctx.ctx.operands.constant_i64(0xffffffffffffu64 as i64);
            pte = ctx.bitwise_and(pte, prot_mask);
        } else {
            let one = ctx.ctx.operands.constant_i32(1);
            pte = ctx.shift_left(pte, one);
            let zero = ctx.ctx.operands.constant_i64(0);
            ctx.branch_if(
                lbl_slow_path,
                pte,
                zero,
                Comparison::LessOrEqual,
                BlockFrequency::Default,
            );
            let one = ctx.ctx.operands.constant_i32(1);
            pte = ctx.shift_right_ui(pte, one);
        }
    } else {
        // When no label is provided to jump to a slow path if the address
        // is invalid, we do the validation ourselves, and throw if needed.

        let lbl_not_watched = ctx.ctx.operands.label();

        // Is the page currently being tracked for read/write? If so we
        // need to call SignalMemoryTracking.
        let zero = ctx.ctx.operands.constant_i64(0);
        ctx.branch_if(
            lbl_not_watched,
            pte,
            zero,
            Comparison::GreaterOrEqual,
            BlockFrequency::Cold,
        );

        // Signal memory tracking. Size here doesn't matter as address is
        // assumed to be size aligned here.
        let one_u64 = ctx.ctx.operands.constant_i64(1);
        let write_flag = ctx.ctx.operands.constant_i32(if write { 1 } else { 0 });
        ctx.call(
            NativeFn::SignalMemoryTracking,
            &[address, one_u64, write_flag],
        );
        ctx.mark_label(lbl_not_watched, BlockFrequency::Default);

        // Ignore any software protection bits. (they are still used by
        // the host-side memory access)
        let prot_mask = ctx.ctx.operands.constant_i64(0xffffffffffffu64 as i64);
        pte = ctx.bitwise_and(pte, prot_mask);

        let lbl_non_null = ctx.ctx.operands.label();

        // Skip exception if the PTE address is non-null (not zero).
        let zero = ctx.ctx.operands.const_zero(OperandType::I64);
        ctx.branch_if(
            lbl_non_null,
            pte,
            zero,
            Comparison::NotEqual,
            BlockFrequency::Cold,
        );

        // The call is not expected to return (it should throw).
        ctx.call(NativeFn::ThrowInvalidMemoryAccess, &[address]);
        ctx.mark_label(lbl_non_null, BlockFrequency::Default);
    }

    let address_type = ctx.ctx.operands.get(address).typ;
    let page_mask = const_typed(ctx, address_type, PAGE_MASK as i64);
    let mut page_offset = ctx.bitwise_and(address, page_mask);

    if ctx.ctx.operands.get(page_offset).typ == OperandType::I32 {
        page_offset = ctx.zero_extend32(OperandType::I64, page_offset);
    }

    ctx.add(pte, page_offset)
}

pub fn emit_host_mapped_pointer(
    ctx: &mut ArmEmitterCtx,
    mut address: OperandIndex,
) -> OperandIndex {
    if ctx.ctx.operands.get(address).typ == OperandType::I32 {
        address = ctx.zero_extend32(OperandType::I64, address);
    }

    if ctx.memory.manager_type() == crate::memory::MemoryManagerType::HostMapped {
        let mask = ctx
            .ctx
            .operands
            .constant_i64((u64::MAX >> (64 - ctx.memory.address_space_bits())) as i64);
        address = ctx.bitwise_and(address, mask);
    }

    let base_addr = ctx
        .ctx
        .operands
        .constant_i64_with_symbol(ctx.memory.page_table_pointer() as i64, page_table_symbol());

    ctx.add(base_addr, address)
}

fn emit_read_int_fallback(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    let func = match size {
        0 => NativeFn::ReadByte,
        1 => NativeFn::ReadUInt16,
        2 => NativeFn::ReadUInt32,
        3 => NativeFn::ReadUInt64,
        _ => panic!("emit_read_int_fallback: size {size} out of range"),
    };

    let value = ctx.call(func, &[address]).expect("Read* returns a value");
    set_int_or_zr(ctx, GpRegisterIndex::from(rt), value);
}

fn emit_read_vector_fallback(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    vector: OperandIndex,
    rt: u8,
    elem: i32,
    size: i32,
) {
    let func = match size {
        0 => NativeFn::ReadByte,
        1 => NativeFn::ReadUInt16,
        2 => NativeFn::ReadUInt32,
        3 => NativeFn::ReadUInt64,
        4 => NativeFn::ReadVector128,
        _ => panic!("emit_read_vector_fallback: size {size} out of range"),
    };

    let mut value = ctx.call(func, &[address]).expect("Read* returns a value");

    value = match size {
        0 => ctx.vector_insert8(vector, value, elem),
        1 => ctx.vector_insert16(vector, value, elem),
        2 | 3 => ctx.vector_insert(vector, value, elem),
        _ => value,
    };

    let vec_reg = get_vec(ctx, VecRegisterIndex::from(rt));
    ctx.copy_to(vec_reg, value);
}

fn emit_write_int_fallback(ctx: &mut ArmEmitterCtx, address: OperandIndex, rt: u8, size: i32) {
    let func = match size {
        0 => NativeFn::WriteByte,
        1 => NativeFn::WriteUInt16,
        2 => NativeFn::WriteUInt32,
        3 => NativeFn::WriteUInt64,
        _ => panic!("emit_write_int_fallback: size {size} out of range"),
    };

    let mut value = get_int_or_zr(ctx, GpRegisterIndex::from(rt));

    if size < 3 && ctx.ctx.operands.get(value).typ == OperandType::I64 {
        value = ctx.convert_i64_to_i32(value);
    }

    ctx.call(func, &[address, value]);
}

fn emit_write_vector_fallback(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    rt: u8,
    elem: i32,
    size: i32,
) {
    let func = match size {
        0 => NativeFn::WriteByte,
        1 => NativeFn::WriteUInt16,
        2 => NativeFn::WriteUInt32,
        3 => NativeFn::WriteUInt64,
        4 => NativeFn::WriteVector128,
        _ => panic!("emit_write_vector_fallback: size {size} out of range"),
    };

    let vec_reg = get_vec(ctx, VecRegisterIndex::from(rt));

    let value = if size < 4 {
        match size {
            0 => ctx.vector_extract8(vec_reg, elem),
            1 => ctx.vector_extract16(vec_reg, elem),
            2 => ctx.vector_extract(OperandType::I32, vec_reg, elem),
            _ => ctx.vector_extract(OperandType::I64, vec_reg, elem),
        }
    } else {
        vec_reg
    };

    ctx.call(func, &[address, value]);
}
