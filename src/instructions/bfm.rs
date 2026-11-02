// Port of Instructions/InstEmitBfm.cs

use crate::decoder::opcode_kind::OpCodeKind;
use crate::registers::GpRegisterIndex;

use super::context::ArmEmitterCtx;
use super::helpers::{get_int_or_zr, set_int_or_zr};

struct BfmOp {
    rd: GpRegisterIndex,
    rn: GpRegisterIndex,
    wmask: i64,
    tmask: i64,
    pos: i32,
    shift: i32,
}

fn bfm_op(ctx: &ArmEmitterCtx) -> BfmOp {
    match ctx.curr_op().kind {
        OpCodeKind::Bfm {
            rd,
            rn,
            wmask,
            tmask,
            pos,
            shift,
        } => BfmOp {
            rd,
            rn,
            wmask,
            tmask,
            pos: pos as i32,
            shift: shift as i32,
        },
        _ => panic!("expected Bfm opcode, got {:?}", ctx.curr_op().kind),
    }
}

pub fn emit_bfm(ctx: &mut ArmEmitterCtx) {
    let op = bfm_op(ctx);

    let d = get_int_or_zr(ctx, op.rd);
    let n = get_int_or_zr(ctx, op.rn);
    let typ = ctx.ctx.operands.get(n).typ;

    let res = if op.pos < op.shift {
        // BFI.
        let shift = ctx.curr_op().bits() as i32 - op.shift;
        let width = op.pos + 1;
        let mask = (u64::MAX >> (64 - width)) as i64;

        let mask_op = ctx.ctx.operands.constant_int(typ, mask);
        let and_n = ctx.bitwise_and(n, mask_op);
        let shift_op = ctx.ctx.operands.constant_i32(shift);
        let res = ctx.shift_left(and_n, shift_op);

        let inv_mask_op = ctx.ctx.operands.constant_int(typ, !(mask << shift));
        let and_d = ctx.bitwise_and(d, inv_mask_op);
        ctx.bitwise_or(res, and_d)
    } else {
        // BFXIL.
        let shift = op.shift;
        let width = op.pos - shift + 1;
        let mask = (u64::MAX >> (64 - width)) as i64;

        let shift_op = ctx.ctx.operands.constant_i32(shift);
        let shr_n = ctx.shift_right_ui(n, shift_op);
        let mask_op = ctx.ctx.operands.constant_int(typ, mask);
        let res = ctx.bitwise_and(shr_n, mask_op);

        let inv_mask_op = ctx.ctx.operands.constant_int(typ, !mask);
        let and_d = ctx.bitwise_and(d, inv_mask_op);
        ctx.bitwise_or(res, and_d)
    };

    set_int_or_zr(ctx, op.rd, res);
}

pub fn emit_sbfm(ctx: &mut ArmEmitterCtx) {
    let op = bfm_op(ctx);
    let bits_count = ctx.curr_op().bits() as i32;

    if op.pos + 1 == bits_count {
        emit_bfm_shift(ctx, true);
    } else if op.pos < op.shift {
        emit_bfiz(ctx, true);
    } else if op.pos == 7 && op.shift == 0 {
        let n = get_int_or_zr(ctx, op.rn);
        let typ = ctx.ctx.operands.get(n).typ;
        let res = ctx.sign_extend8(typ, n);
        set_int_or_zr(ctx, op.rd, res);
    } else if op.pos == 15 && op.shift == 0 {
        let n = get_int_or_zr(ctx, op.rn);
        let typ = ctx.ctx.operands.get(n).typ;
        let res = ctx.sign_extend16(typ, n);
        set_int_or_zr(ctx, op.rd, res);
    } else if op.pos == 31 && op.shift == 0 {
        let n = get_int_or_zr(ctx, op.rn);
        let typ = ctx.ctx.operands.get(n).typ;
        let res = ctx.sign_extend32(typ, n);
        set_int_or_zr(ctx, op.rd, res);
    } else {
        let res = get_int_or_zr(ctx, op.rn);

        let shl_op = ctx.ctx.operands.constant_i32(bits_count - 1 - op.pos);
        let res = ctx.shift_left(res, shl_op);
        let shr_op = ctx.ctx.operands.constant_i32(bits_count - 1);
        let res = ctx.shift_right_si(res, shr_op);

        let typ = ctx.ctx.operands.get(res).typ;
        let inv_tmask = ctx.ctx.operands.constant_int(typ, !op.tmask);
        let res = ctx.bitwise_and(res, inv_tmask);

        let n2 = get_bfm_n(ctx);
        let or = ctx.bitwise_or(res, n2);
        set_int_or_zr(ctx, op.rd, or);
    }
}

pub fn emit_ubfm(ctx: &mut ArmEmitterCtx) {
    let op = bfm_op(ctx);
    let bits_count = ctx.curr_op().bits() as i32;

    if op.pos + 1 == bits_count {
        emit_bfm_shift(ctx, false);
    } else if op.pos < op.shift {
        emit_bfiz(ctx, false);
    } else if op.pos + 1 == op.shift {
        emit_bfm_lsl(ctx);
    } else if op.pos == 7 && op.shift == 0 {
        let n = get_int_or_zr(ctx, op.rn);
        let typ = ctx.ctx.operands.get(n).typ;
        let mask = ctx.ctx.operands.constant_int(typ, 0xff);
        let res = ctx.bitwise_and(n, mask);
        set_int_or_zr(ctx, op.rd, res);
    } else if op.pos == 15 && op.shift == 0 {
        let n = get_int_or_zr(ctx, op.rn);
        let typ = ctx.ctx.operands.get(n).typ;
        let mask = ctx.ctx.operands.constant_int(typ, 0xffff);
        let res = ctx.bitwise_and(n, mask);
        set_int_or_zr(ctx, op.rd, res);
    } else {
        let n2 = get_bfm_n(ctx);
        set_int_or_zr(ctx, op.rd, n2);
    }
}

// C# EmitSbfiz/EmitUbfiz → EmitBfiz(signed).
fn emit_bfiz(ctx: &mut ArmEmitterCtx, signed: bool) {
    let op = bfm_op(ctx);
    let width = op.pos + 1;

    let res = get_int_or_zr(ctx, op.rn);

    let shl_op = ctx
        .ctx
        .operands
        .constant_i32(ctx.curr_op().bits() as i32 - width);
    let res = ctx.shift_left(res, shl_op);

    let shr_op = ctx.ctx.operands.constant_i32(op.shift - width);
    let res = if signed {
        ctx.shift_right_si(res, shr_op)
    } else {
        ctx.shift_right_ui(res, shr_op)
    };

    set_int_or_zr(ctx, op.rd, res);
}

// C# EmitSbfmShift/EmitUbfmShift → EmitBfmShift(signed).
fn emit_bfm_shift(ctx: &mut ArmEmitterCtx, signed: bool) {
    let op = bfm_op(ctx);

    let res = get_int_or_zr(ctx, op.rn);

    let shift_op = ctx.ctx.operands.constant_i32(op.shift);
    let res = if signed {
        ctx.shift_right_si(res, shift_op)
    } else {
        ctx.shift_right_ui(res, shift_op)
    };

    set_int_or_zr(ctx, op.rd, res);
}

fn emit_bfm_lsl(ctx: &mut ArmEmitterCtx) {
    let op = bfm_op(ctx);

    let res = get_int_or_zr(ctx, op.rn);

    let shift = ctx.curr_op().bits() as i32 - op.shift;
    let shift_op = ctx.ctx.operands.constant_i32(shift);
    let res = ctx.shift_left(res, shift_op);

    set_int_or_zr(ctx, op.rd, res);
}

fn get_bfm_n(ctx: &mut ArmEmitterCtx) -> crate::ir::operand::OperandIndex {
    let op = bfm_op(ctx);

    let res = get_int_or_zr(ctx, op.rn);

    let mask = op.wmask & op.tmask;

    let shift_op = ctx.ctx.operands.constant_i32(op.shift);
    let ror = ctx.rotate_right(res, shift_op);
    let typ = ctx.ctx.operands.get(res).typ;
    let mask_op = ctx.ctx.operands.constant_int(typ, mask);
    ctx.bitwise_and(ror, mask_op)
}
