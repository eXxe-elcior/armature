use crate::decoder::RegisterSize;
use crate::decoder::utilities::IntegerType;
use crate::ir::operand::OperandIndex;
use crate::ir::types::{OperandType, RegisterType};
use crate::registers::{GpRegisterIndex, VecRegisterIndex};
use crate::state::{FPState, PState};

use super::context::ArmEmitterCtx;

// Helpers matching C# InstEmitHelper
pub fn get_operand_type(reg_size: RegisterSize) -> OperandType {
    match reg_size {
        RegisterSize::Int32 => OperandType::I32,
        RegisterSize::Int64 => OperandType::I64,
        RegisterSize::Simd64 | RegisterSize::Simd128 => OperandType::I64,
    }
}

pub fn get_int_or_zr(ctx: &mut ArmEmitterCtx, reg: GpRegisterIndex) -> OperandIndex {
    if reg.is_zr_or_sp() {
        let typ = get_operand_type(ctx.curr_op().reg_size);
        ctx.ctx.operands.const_zero(typ)
    } else {
        get_int_or_sp(ctx, reg)
    }
}

pub fn set_int_or_zr(ctx: &mut ArmEmitterCtx, reg: GpRegisterIndex, value: OperandIndex) {
    if !reg.is_zr_or_sp() {
        set_int_or_sp(ctx, reg, value);
    }
}

pub fn get_int_or_sp(ctx: &mut ArmEmitterCtx, reg: GpRegisterIndex) -> OperandIndex {
    let value =
        ctx.ctx
            .operands
            .register(reg.idx() as u32, RegisterType::Integer, OperandType::I64);
    if ctx.curr_op().reg_size == RegisterSize::Int32 {
        ctx.convert_i64_to_i32(value)
    } else {
        value
    }
}

pub fn set_int_or_sp(ctx: &mut ArmEmitterCtx, reg: GpRegisterIndex, value: OperandIndex) {
    let reg_op =
        ctx.ctx
            .operands
            .register(reg.idx() as u32, RegisterType::Integer, OperandType::I64);
    let val = if ctx.ctx.operands.get(value).typ == OperandType::I32 {
        ctx.zero_extend32(OperandType::I64, value)
    } else {
        value
    };
    ctx.copy_to(reg_op, val);
}

pub fn get_vec(ctx: &mut ArmEmitterCtx, reg: VecRegisterIndex) -> OperandIndex {
    ctx.ctx
        .operands
        .register(reg.idx() as u32, RegisterType::Vector, OperandType::V128)
}

pub fn get_flag(
    state_flag: PState,
    operands: &mut crate::ir::operand::OperandArena,
) -> OperandIndex {
    operands.register(state_flag as u32, RegisterType::Flag, OperandType::I32)
}

pub fn get_fp_flag(
    state_flag: FPState,
    operands: &mut crate::ir::operand::OperandArena,
) -> OperandIndex {
    operands.register(state_flag as u32, RegisterType::FpFlag, OperandType::I32)
}

pub fn set_flag(ctx: &mut ArmEmitterCtx, state_flag: PState, value: OperandIndex) {
    let flag_op =
        ctx.ctx
            .operands
            .register(state_flag as u32, RegisterType::Flag, OperandType::I32);
    ctx.copy_to(flag_op, value);
    ctx.mark_flag_set(state_flag);
}

pub fn set_fp_flag(ctx: &mut ArmEmitterCtx, state_flag: FPState, value: OperandIndex) {
    let flag_op =
        ctx.ctx
            .operands
            .register(state_flag as u32, RegisterType::FpFlag, OperandType::I32);
    ctx.copy_to(flag_op, value);
}

pub fn get_extended_m(
    ctx: &mut ArmEmitterCtx,
    rm: GpRegisterIndex,
    int_type: IntegerType,
) -> OperandIndex {
    let value = get_int_or_zr(ctx, rm);
    let typ = ctx.ctx.operands.get(value).typ;
    match int_type {
        IntegerType::Uxtb => ctx.zero_extend8(typ, value),
        IntegerType::Uxth => ctx.zero_extend16(typ, value),
        IntegerType::Uxtw => ctx.zero_extend32(typ, value),
        IntegerType::Uxtx => value,
        IntegerType::Sxtb => ctx.sign_extend8(typ, value),
        IntegerType::Sxth => ctx.sign_extend16(typ, value),
        IntegerType::Sxtw => ctx.sign_extend32(typ, value),
        IntegerType::Sxtx => value,
    }
}
