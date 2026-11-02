// Port of Instructions/InstEmitMemoryExHelper.cs.
//
// Divergence: the `a32` parameter of EmitStoreExclusive is dropped (A32
// excluded)  SetRs always goes through SetIntOrZR.

use crate::ir::operand::OperandIndex;
use crate::ir::types::OperandType;
use crate::registers::{CpuState, GpRegisterIndex};

use super::context::ArmEmitterCtx;
use super::helpers::set_int_or_zr;
use super::memory_helpers::{emit_pt_pointer_load, emit_read_int_aligned, emit_write_int_aligned};

const ERG_SIZE_LOG2: i32 = 4;

fn get_exclusive_address_mask() -> i64 {
    !((4i64 << ERG_SIZE_LOG2) - 1)
}

pub fn emit_load_exclusive(
    ctx: &mut ArmEmitterCtx,
    address: OperandIndex,
    exclusive: bool,
    size: i32,
) -> OperandIndex {
    if exclusive {
        let value = if size == 4 {
            // Only 128-bit CAS is guaranteed to have an atomic load.
            let phys_addr = emit_pt_pointer_load(ctx, address, None, false, 4);

            let zero = ctx.vector_zero();

            ctx.compare_and_swap(phys_addr, zero, zero)
        } else {
            emit_read_int_aligned(ctx, address, size)
        };

        let arg0 = ctx.load_argument(OperandType::I64, 0);

        let ex_addr_offs = ctx
            .ctx
            .operands
            .constant_i64(CpuState::exclusive_address_offset() as i64);
        let ex_addr_ptr = ctx.add(arg0, ex_addr_offs);
        let ex_value_offs = ctx
            .ctx
            .operands
            .constant_i64(CpuState::exclusive_value_offset() as i64);
        let ex_value_ptr = ctx.add(arg0, ex_value_offs);

        // address is always I64 here (A64); C# Const(address.Type, mask).
        let mask = ctx.ctx.operands.constant_i64(get_exclusive_address_mask());
        let masked_address = ctx.bitwise_and(address, mask);
        ctx.store(ex_addr_ptr, masked_address);

        // Make sure the unused higher bits of the value are cleared.
        if size < 3 {
            let zero64 = ctx.ctx.operands.constant_i64(0);
            ctx.store(ex_value_ptr, zero64);
        }

        if size < 4 {
            let eight = ctx.ctx.operands.constant_i64(8);
            let high_ptr = ctx.add(ex_value_ptr, eight);
            let zero64 = ctx.ctx.operands.constant_i64(0);
            ctx.store(high_ptr, zero64);
        }

        // Store the new exclusive value.
        ctx.store(ex_value_ptr, value);

        value
    } else {
        emit_read_int_aligned(ctx, address, size)
    }
}

pub fn emit_store_exclusive(
    ctx: &mut ArmEmitterCtx,
    mut address: OperandIndex,
    mut value: OperandIndex,
    exclusive: bool,
    size: i32,
    rs: GpRegisterIndex,
) {
    if size < 3 {
        value = ctx.convert_i64_to_i32(value);
    }

    if exclusive {
        // We overwrite one of the registers (Rs), keep a copy of the
        // values to ensure we are working with the correct values.
        address = ctx.copy(address);
        value = ctx.copy(value);

        let arg0 = ctx.load_argument(OperandType::I64, 0);

        let ex_addr_offs = ctx
            .ctx
            .operands
            .constant_i64(CpuState::exclusive_address_offset() as i64);
        let ex_addr_ptr = ctx.add(arg0, ex_addr_offs);
        let address_type = ctx.ctx.operands.get(address).typ;
        let ex_addr = ctx.load(address_type, ex_addr_ptr);

        // STEP 1: Check if we have exclusive access to this memory region.
        // If not, fail and skip the store.
        let mask = ctx.ctx.operands.constant_i64(get_exclusive_address_mask());
        let masked_address = ctx.bitwise_and(address, mask);

        let ex_failed = ctx.icompare_not_equal(ex_addr, masked_address);

        let lbl_exit = ctx.ctx.operands.label();

        let one = ctx.ctx.operands.constant_i32(1);
        set_int_or_zr(ctx, rs, one);

        ctx.branch_if_true(lbl_exit, ex_failed);

        // STEP 2: We have exclusive access and the address is valid,
        // attempt the store using CAS.
        let phys_addr = emit_pt_pointer_load(ctx, address, None, true, size);

        let ex_value_offs = ctx
            .ctx
            .operands
            .constant_i64(CpuState::exclusive_value_offset() as i64);
        let ex_value_ptr = ctx.add(arg0, ex_value_offs);
        let ex_value = match size {
            0 => ctx.load8(ex_value_ptr),
            1 => ctx.load16(ex_value_ptr),
            2 => ctx.load(OperandType::I32, ex_value_ptr),
            3 => ctx.load(OperandType::I64, ex_value_ptr),
            _ => ctx.load(OperandType::V128, ex_value_ptr),
        };

        let curr_value = match size {
            0 => ctx.compare_and_swap8(phys_addr, ex_value, value),
            1 => ctx.compare_and_swap16(phys_addr, ex_value, value),
            _ => ctx.compare_and_swap(phys_addr, ex_value, value),
        };

        // STEP 3: Check if we succeeded by comparing expected and
        // in-memory values.
        let store_failed = if size == 4 {
            let curr_value_low = ctx.vector_extract(OperandType::I64, curr_value, 0);
            let curr_value_high = ctx.vector_extract(OperandType::I64, curr_value, 1);

            let ex_value_low = ctx.vector_extract(OperandType::I64, ex_value, 0);
            let ex_value_high = ctx.vector_extract(OperandType::I64, ex_value, 1);

            let low_ne = ctx.icompare_not_equal(curr_value_low, ex_value_low);
            let high_ne = ctx.icompare_not_equal(curr_value_high, ex_value_high);
            ctx.bitwise_or(low_ne, high_ne)
        } else {
            ctx.icompare_not_equal(curr_value, ex_value)
        };

        set_int_or_zr(ctx, rs, store_failed);

        ctx.mark_label(lbl_exit, crate::ir::block::BlockFrequency::Default);
    } else {
        emit_write_int_aligned(ctx, address, value, size);
    }
}

pub fn emit_clear_exclusive(ctx: &mut ArmEmitterCtx) {
    let arg0 = ctx.load_argument(OperandType::I64, 0);

    let ex_addr_offs = ctx
        .ctx
        .operands
        .constant_i64(CpuState::exclusive_address_offset() as i64);
    let ex_addr_ptr = ctx.add(arg0, ex_addr_offs);

    // We store ULONG max to force any exclusive address checks to fail,
    // since this value is not aligned to the ERG mask.
    let max = ctx.ctx.operands.constant_i64(u64::MAX as i64);
    ctx.store(ex_addr_ptr, max);
}
