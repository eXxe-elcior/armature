// Port of Instructions/InstEmitMul.cs

use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::operand::OperandIndex;
use crate::registers::GpRegisterIndex;

use super::context::ArmEmitterCtx;
use super::helpers::{get_int_or_zr, set_int_or_zr};

fn mul_op(
    ctx: &ArmEmitterCtx,
) -> (
    GpRegisterIndex,
    GpRegisterIndex,
    GpRegisterIndex,
    GpRegisterIndex,
) {
    match ctx.curr_op().kind {
        OpCodeKind::Mul { rd, rn, rm, ra, .. } => (rd, rn, rm, ra),
        _ => panic!("expected Mul opcode, got {:?}", ctx.curr_op().kind),
    }
}

pub fn emit_madd(ctx: &mut ArmEmitterCtx) {
    emit_mul_inner(ctx, true);
}
pub fn emit_msub(ctx: &mut ArmEmitterCtx) {
    emit_mul_inner(ctx, false);
}

fn emit_mul_inner(ctx: &mut ArmEmitterCtx, is_add: bool) {
    let (rd, rn, rm, ra) = mul_op(ctx);

    let a = get_int_or_zr(ctx, ra);
    let n = get_int_or_zr(ctx, rn);
    let m = get_int_or_zr(ctx, rm);

    let res = ctx.multiply(n, m);
    let res = if is_add {
        ctx.add(a, res)
    } else {
        ctx.subtract(a, res)
    };

    set_int_or_zr(ctx, rd, res);
}

pub fn emit_smaddl(ctx: &mut ArmEmitterCtx) {
    emit_mull_inner(ctx, true, true);
}
pub fn emit_smsubl(ctx: &mut ArmEmitterCtx) {
    emit_mull_inner(ctx, true, false);
}
pub fn emit_umaddl(ctx: &mut ArmEmitterCtx) {
    emit_mull_inner(ctx, false, true);
}
pub fn emit_umsubl(ctx: &mut ArmEmitterCtx) {
    emit_mull_inner(ctx, false, false);
}

// C# [Flags] MullFlags { Signed, Add } collapsed into two bools.
fn emit_mull_inner(ctx: &mut ArmEmitterCtx, signed: bool, is_add: bool) {
    let (rd, rn, rm, ra) = mul_op(ctx);

    let a = get_int_or_zr(ctx, ra);

    let n = get_extended_register32(ctx, rn, signed);
    let m = get_extended_register32(ctx, rm, signed);

    let res = ctx.multiply(n, m);
    let res = if is_add {
        ctx.add(a, res)
    } else {
        ctx.subtract(a, res)
    };

    set_int_or_zr(ctx, rd, res);
}

// C# local function GetExtendedRegister32 inside EmitMull.
fn get_extended_register32(
    ctx: &mut ArmEmitterCtx,
    reg: GpRegisterIndex,
    signed: bool,
) -> OperandIndex {
    let value = get_int_or_zr(ctx, reg);
    let typ = ctx.ctx.operands.get(value).typ;
    if signed {
        ctx.sign_extend32(typ, value)
    } else {
        ctx.zero_extend32(typ, value)
    }
}

pub fn emit_smulh(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm, _) = mul_op(ctx);
    let n = get_int_or_zr(ctx, rn);
    let m = get_int_or_zr(ctx, rm);
    let d = ctx.multiply64_high_si(n, m);
    set_int_or_zr(ctx, rd, d);
}

pub fn emit_umulh(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm, _) = mul_op(ctx);
    let n = get_int_or_zr(ctx, rn);
    let m = get_int_or_zr(ctx, rm);
    let d = ctx.multiply64_high_ui(n, m);
    set_int_or_zr(ctx, rd, d);
}
