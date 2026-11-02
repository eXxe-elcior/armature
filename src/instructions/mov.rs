// Port of Instructions/InstEmitMove.cs (module named `mov` because `move`
// is a reserved Rust keyword).

use crate::decoder::opcode_kind::OpCodeKind;
use crate::registers::GpRegisterIndex;

use super::context::ArmEmitterCtx;
use super::helpers::{get_int_or_zr, get_operand_type, set_int_or_zr};

fn mov_op(ctx: &ArmEmitterCtx) -> (GpRegisterIndex, i64, u8) {
    match ctx.curr_op().kind {
        OpCodeKind::Mov { rd, immediate, bit } => (rd, immediate, bit),
        _ => panic!("expected Mov opcode, got {:?}", ctx.curr_op().kind),
    }
}

pub fn emit_movk(ctx: &mut ArmEmitterCtx) {
    let (rd, immediate, bit) = mov_op(ctx);
    let typ = get_operand_type(ctx.curr_op().reg_size);

    let mut res = get_int_or_zr(ctx, rd);

    let mask = ctx.ctx.operands.constant_int(typ, !(0xffffi64 << bit));
    res = ctx.bitwise_and(res, mask);

    let imm = ctx.ctx.operands.constant_int(typ, immediate);
    res = ctx.bitwise_or(res, imm);

    set_int_or_zr(ctx, rd, res);
}

pub fn emit_movn(ctx: &mut ArmEmitterCtx) {
    let (rd, immediate, _) = mov_op(ctx);
    let typ = get_operand_type(ctx.curr_op().reg_size);
    let imm = ctx.ctx.operands.constant_int(typ, !immediate);
    set_int_or_zr(ctx, rd, imm);
}

pub fn emit_movz(ctx: &mut ArmEmitterCtx) {
    let (rd, immediate, _) = mov_op(ctx);
    let typ = get_operand_type(ctx.curr_op().reg_size);
    let imm = ctx.ctx.operands.constant_int(typ, immediate);
    set_int_or_zr(ctx, rd, imm);
}
