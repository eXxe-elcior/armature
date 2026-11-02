// Port of Instructions/InstEmitCsel.cs

use crate::decoder::opcode_kind::OpCodeKind;

use super::context::ArmEmitterCtx;
use super::flow_helpers::get_cond_true;
use super::helpers::{get_int_or_zr, set_int_or_zr};

#[derive(PartialEq)]
enum CselOperation {
    None,
    Increment,
    Invert,
    Negate,
}

pub fn emit_csel(ctx: &mut ArmEmitterCtx) {
    emit_csel_inner(ctx, CselOperation::None);
}
pub fn emit_csinc(ctx: &mut ArmEmitterCtx) {
    emit_csel_inner(ctx, CselOperation::Increment);
}
pub fn emit_csinv(ctx: &mut ArmEmitterCtx) {
    emit_csel_inner(ctx, CselOperation::Invert);
}
pub fn emit_csneg(ctx: &mut ArmEmitterCtx) {
    emit_csel_inner(ctx, CselOperation::Negate);
}

fn emit_csel_inner(ctx: &mut ArmEmitterCtx, csel_op: CselOperation) {
    let (rd, rn, rm, cond) = match ctx.curr_op().kind {
        OpCodeKind::Csel { rd, rn, rm, cond } => (rd, rn, rm, cond),
        _ => panic!("expected Csel opcode, got {:?}", ctx.curr_op().kind),
    };

    let n = get_int_or_zr(ctx, rn);
    let mut m = get_int_or_zr(ctx, rm);

    if csel_op == CselOperation::Increment {
        let typ = ctx.ctx.operands.get(m).typ;
        let one = ctx.ctx.operands.constant_int(typ, 1);
        m = ctx.add(m, one);
    } else if csel_op == CselOperation::Invert {
        m = ctx.bitwise_not(m);
    } else if csel_op == CselOperation::Negate {
        m = ctx.negate(m);
    }

    let cond_true = get_cond_true(ctx, cond);

    let d = ctx.conditional_select(cond_true, n, m);

    set_int_or_zr(ctx, rd, d);
}
