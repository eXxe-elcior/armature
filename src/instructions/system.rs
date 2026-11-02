// Port of Instructions/InstEmitSystem.cs.

use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::operand::OperandIndex;
use crate::ir::types::{OperandType, RegisterType};
use crate::registers::{CpuState, GpRegisterIndex};
use crate::state::{PState, fpcr, fpsr};

use super::context::ArmEmitterCtx;
use super::helpers::{get_flag, get_int_or_zr, set_flag, set_int_or_zr};
use super::memory_helpers::{emit_pt_pointer_load, emit_store};
use crate::instructions::native_function::NativeFn;

const DCZ_SIZE_LOG2: i32 = 4; // Log2 size in words
pub const DCZ_SIZE_IN_BYTES: i32 = 4 << DCZ_SIZE_LOG2;

// C# casts CurrOp to OpCodeSystem.
fn system_op(ctx: &ArmEmitterCtx) -> (GpRegisterIndex, u8, u8, u8, u8, u8) {
    match ctx.curr_op().kind {
        OpCodeKind::System {
            rt,
            op0,
            op1,
            op2,
            crn,
            crm,
        } => (rt, op0, op1, op2, crn, crm),
        _ => panic!("expected System opcode, got {:?}", ctx.curr_op().kind),
    }
}

// C# GetPackedId(OpCodeSystem).
fn get_packed_id(op0: u8, op1: u8, op2: u8, crn: u8, crm: u8) -> u32 {
    (op2 as u32) | (crm as u32) << 3 | (crn as u32) << 7 | (op1 as u32) << 11 | (op0 as u32) << 14
}

pub fn emit_isb(_ctx: &mut ArmEmitterCtx) {
    // Execute as no-op.
}

pub fn emit_mrs(ctx: &mut ArmEmitterCtx) {
    let (rt, op0, op1, op2, crn, crm) = system_op(ctx);

    let func = match get_packed_id(op0, op1, op2, crn, crm) {
        0b11_011_0000_0000_001 => NativeFn::GetCtrEl0,
        0b11_011_0000_0000_111 => NativeFn::GetDczidEl0,
        0b11_011_0100_0010_000 => return emit_get_nzcv(ctx),
        0b11_011_0100_0100_000 => return emit_get_fpcr(ctx),
        0b11_011_0100_0100_001 => return emit_get_fpsr(ctx),
        0b11_011_1101_0000_010 => return emit_get_tpidr_el0(ctx),
        0b11_011_1101_0000_011 => return emit_get_tpidrro_el0(ctx),
        0b11_011_1101_0000_101 => return emit_get_tpidr2_el0(ctx),
        0b11_011_1110_0000_000 => NativeFn::GetCntfrqEl0,
        0b11_011_1110_0000_001 => NativeFn::GetCntpctEl0,
        0b11_011_1110_0000_010 => NativeFn::GetCntvctEl0,
        _ => panic!(
            "Unknown MRS 0x{:08X} at 0x{:016X}.",
            ctx.curr_op().raw,
            ctx.curr_op().address
        ),
    };

    let value = ctx.call(func, &[]).expect("MRS helper returns a value");
    set_int_or_zr(ctx, rt, value);
}

pub fn emit_msr(ctx: &mut ArmEmitterCtx) {
    let (_, op0, op1, op2, crn, crm) = system_op(ctx);

    match get_packed_id(op0, op1, op2, crn, crm) {
        0b11_011_0100_0010_000 => emit_set_nzcv(ctx),
        0b11_011_0100_0100_000 => emit_set_fpcr(ctx),
        0b11_011_0100_0100_001 => emit_set_fpsr(ctx),
        0b11_011_1101_0000_010 => emit_set_tpidr_el0(ctx),
        0b11_011_1101_0000_101 => emit_set_tpidr2_el0(ctx),
        _ => panic!(
            "Unknown MSR 0x{:08X} at 0x{:016X}.",
            ctx.curr_op().raw,
            ctx.curr_op().address
        ),
    }
}

pub fn emit_nop(_ctx: &mut ArmEmitterCtx) {
    // Do nothing.
}

pub fn emit_sys(ctx: &mut ArmEmitterCtx) {
    // C#: cache-maintenance ops treated mostly as no-ops (no emulated cache).
    let (rt, op0, op1, op2, crn, crm) = system_op(ctx);

    match get_packed_id(op0, op1, op2, crn, crm) {
        0b11_011_0111_0100_001 => {
            // DC ZVA  zero N bytes (64 by default).
            let t = get_int_or_zr(ctx, rt);

            if ctx.memory.manager_type().is_host_mapped_or_tracked() {
                // Fast path: 4× V128 stores after shared address translation.
                let phys_base = emit_pt_pointer_load(ctx, t, None, true, 3);
                let zero = ctx.vector_zero();
                let mut offset: i64 = 0;
                while offset < DCZ_SIZE_IN_BYTES as i64 {
                    let offs = ctx.ctx.operands.constant_i64(offset);
                    let address = ctx.add(phys_base, offs);
                    ctx.store(address, zero);
                    offset += 16;
                }
            } else {
                // Slow path: 8× I64 stores through full memory helpers.
                let mut offset: i64 = 0;
                while offset < DCZ_SIZE_IN_BYTES as i64 {
                    let offs = ctx.ctx.operands.constant_i64(offset);
                    let address = ctx.add(t, offs);
                    emit_store(ctx, address, 31, 3);
                    offset += 8;
                }
            }
        }

        // No-op
        0b11_011_0111_1110_001 => {} // DC CIVAC

        0b11_011_0111_0101_001 => {
            // IC IVAU
            let target =
                ctx.ctx
                    .operands
                    .register(rt.idx() as u32, RegisterType::Integer, OperandType::I64);
            ctx.call(NativeFn::InvalidateCacheLine, &[target]);
        }

        // C# switch has no default arm  unmatched ids fall through.
        _ => {}
    }
}

fn emit_get_nzcv(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let v = get_flag(PState::VFlag, &mut ctx.ctx.operands);
    let sh = ctx.ctx.operands.constant_i32(PState::VFlag as i32);
    let mut nzcv = ctx.shift_left(v, sh);

    // C# ORs in C, Z, N in that order.
    for flag in [PState::CFlag, PState::ZFlag, PState::NFlag] {
        let f = get_flag(flag, &mut ctx.ctx.operands);
        let sh = ctx.ctx.operands.constant_i32(flag as i32);
        let shifted = ctx.shift_left(f, sh);
        nzcv = ctx.bitwise_or(nzcv, shifted);
    }

    set_int_or_zr(ctx, rt, nzcv);
}

// C# loops flag 0..RegisterConsts.FpFlagsCount and casts the index to
// FPState; the FpFlag register index IS the flag index, so the register
// operand is built directly here.
fn get_fp_flag_idx(ctx: &mut ArmEmitterCtx, flag: u8) -> OperandIndex {
    ctx.ctx
        .operands
        .register(flag as u32, RegisterType::FpFlag, OperandType::I32)
}

// C# SetFpFlag  Copy only, no mark_flag_set (matches helpers::set_fp_flag).
fn set_fp_flag_idx(ctx: &mut ArmEmitterCtx, flag: u8, value: OperandIndex) {
    let flag_op = ctx
        .ctx
        .operands
        .register(flag as u32, RegisterType::FpFlag, OperandType::I32);
    ctx.copy_to(flag_op, value);
}

fn emit_get_fpcr(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let mut value = ctx.ctx.operands.constant_i32(0);

    for flag in 0..32u8 {
        // RegisterConsts.FpFlagsCount
        if fpcr::MASK & (1u32 << flag) != 0 {
            let flag_reg = get_fp_flag_idx(ctx, flag);
            let sh = ctx.ctx.operands.constant_i32(flag as i32);
            let shifted = ctx.shift_left(flag_reg, sh);
            value = ctx.bitwise_or(value, shifted);
        }
    }

    set_int_or_zr(ctx, rt, value);
}

fn emit_get_fpsr(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    ctx.sync_qc_flag();

    let mut value = ctx.ctx.operands.constant_i32(0);

    for flag in 0..32u8 {
        // RegisterConsts.FpFlagsCount
        if fpsr::MASK & (1u32 << flag) != 0 {
            let flag_reg = get_fp_flag_idx(ctx, flag);
            let sh = ctx.ctx.operands.constant_i32(flag as i32);
            let shifted = ctx.shift_left(flag_reg, sh);
            value = ctx.bitwise_or(value, shifted);
        }
    }

    set_int_or_zr(ctx, rt, value);
}

fn emit_get_tpidr_el0(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let native_ctx = ctx.load_argument(OperandType::I64, 0);
    let offs = ctx
        .ctx
        .operands
        .constant_i64(CpuState::tpidr_el0_offset() as i64);
    let addr = ctx.add(native_ctx, offs);
    let result = ctx.load(OperandType::I64, addr);

    set_int_or_zr(ctx, rt, result);
}

fn emit_get_tpidrro_el0(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let native_ctx = ctx.load_argument(OperandType::I64, 0);
    let offs = ctx
        .ctx
        .operands
        .constant_i64(CpuState::tpidrro_el0_offset() as i64);
    let addr = ctx.add(native_ctx, offs);
    let result = ctx.load(OperandType::I64, addr);

    set_int_or_zr(ctx, rt, result);
}

fn emit_get_tpidr2_el0(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let native_ctx = ctx.load_argument(OperandType::I64, 0);
    let offs = ctx
        .ctx
        .operands
        .constant_i64(CpuState::tpidr2_el0_offset() as i64);
    let addr = ctx.add(native_ctx, offs);
    let result = ctx.load(OperandType::I64, addr);

    set_int_or_zr(ctx, rt, result);
}

fn emit_set_nzcv(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let nzcv64 = get_int_or_zr(ctx, rt);
    let nzcv = ctx.convert_i64_to_i32(nzcv64);

    // C# writes the four SetFlag lines explicitly, in V, C, Z, N order.
    for flag in [PState::VFlag, PState::CFlag, PState::ZFlag, PState::NFlag] {
        let sh = ctx.ctx.operands.constant_i32(flag as i32);
        let shifted = ctx.shift_right_ui(nzcv, sh);
        let one = ctx.ctx.operands.constant_i32(1);
        let bit = ctx.bitwise_and(shifted, one);
        set_flag(ctx, flag, bit);
    }
}

fn emit_set_fpcr(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let value64 = get_int_or_zr(ctx, rt);
    let value = ctx.convert_i64_to_i32(value64);

    for flag in 0..32u8 {
        // RegisterConsts.FpFlagsCount
        if fpcr::MASK & (1u32 << flag) != 0 {
            let sh = ctx.ctx.operands.constant_i32(flag as i32);
            let shifted = ctx.shift_right_ui(value, sh);
            let one = ctx.ctx.operands.constant_i32(1);
            let bit = ctx.bitwise_and(shifted, one);
            set_fp_flag_idx(ctx, flag, bit);
        }
    }

    ctx.update_arm_fp_mode();
}

fn emit_set_fpsr(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    ctx.clear_qc_flag_if_modified();

    let value64 = get_int_or_zr(ctx, rt);
    let value = ctx.convert_i64_to_i32(value64);

    for flag in 0..32u8 {
        // RegisterConsts.FpFlagsCount
        if fpsr::MASK & (1u32 << flag) != 0 {
            let sh = ctx.ctx.operands.constant_i32(flag as i32);
            let shifted = ctx.shift_right_ui(value, sh);
            let one = ctx.ctx.operands.constant_i32(1);
            let bit = ctx.bitwise_and(shifted, one);
            set_fp_flag_idx(ctx, flag, bit);
        }
    }

    ctx.update_arm_fp_mode();
}

fn emit_set_tpidr_el0(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let value = get_int_or_zr(ctx, rt);

    let native_ctx = ctx.load_argument(OperandType::I64, 0);
    let offs = ctx
        .ctx
        .operands
        .constant_i64(CpuState::tpidr_el0_offset() as i64);
    let addr = ctx.add(native_ctx, offs);
    ctx.store(addr, value);
}

fn emit_set_tpidr2_el0(ctx: &mut ArmEmitterCtx) {
    let (rt, ..) = system_op(ctx);

    let value = get_int_or_zr(ctx, rt);

    let native_ctx = ctx.load_argument(OperandType::I64, 0);
    let offs = ctx
        .ctx
        .operands
        .constant_i64(CpuState::tpidr2_el0_offset() as i64);
    let addr = ctx.add(native_ctx, offs);
    ctx.store(addr, value);
}
