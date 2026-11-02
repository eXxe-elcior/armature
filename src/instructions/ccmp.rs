// Port of Instructions/InstEmitCcmp.cs

use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::block::BlockFrequency;
use crate::state::PState;

use super::alu_helpers::{
    emit_adds_c_check, emit_adds_v_check, emit_nz_flags_check, emit_subs_c_check,
    emit_subs_v_check, get_alu_m, get_alu_n,
};
use super::context::ArmEmitterCtx;
use super::flow_helpers::emit_cond_branch;
use super::helpers::set_flag;

pub fn emit_ccmn(ctx: &mut ArmEmitterCtx) {
    emit_ccmp_inner(ctx, true);
}
pub fn emit_ccmp(ctx: &mut ArmEmitterCtx) {
    emit_ccmp_inner(ctx, false);
}

fn emit_ccmp_inner(ctx: &mut ArmEmitterCtx, is_negated: bool) {
    let (nzcv, cond) = match ctx.curr_op().kind {
        OpCodeKind::Ccmp { nzcv, cond, .. } => (nzcv as i32, cond),
        _ => panic!("expected Ccmp opcode, got {:?}", ctx.curr_op().kind),
    };

    let lbl_true = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    emit_cond_branch(ctx, lbl_true, cond);

    // Condition false: write the immediate NZCV (bit 0=V, 1=C, 2=Z, 3=N).
    let v = ctx.ctx.operands.constant_i32(nzcv & 1);
    set_flag(ctx, PState::VFlag, v);
    let c = ctx.ctx.operands.constant_i32((nzcv >> 1) & 1);
    set_flag(ctx, PState::CFlag, c);
    let z = ctx.ctx.operands.constant_i32((nzcv >> 2) & 1);
    set_flag(ctx, PState::ZFlag, z);
    let n_flag = ctx.ctx.operands.constant_i32((nzcv >> 3) & 1);
    set_flag(ctx, PState::NFlag, n_flag);

    ctx.branch(lbl_end);

    ctx.mark_label(lbl_true, BlockFrequency::Default);

    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);

    if is_negated {
        let d = ctx.add(n, m);

        emit_nz_flags_check(ctx, d);

        emit_adds_c_check(ctx, n, d);
        emit_adds_v_check(ctx, n, m, d);
    } else {
        let d = ctx.subtract(n, m);

        emit_nz_flags_check(ctx, d);

        emit_subs_c_check(ctx, n, m);
        emit_subs_v_check(ctx, n, m, d);
    }

    ctx.mark_label(lbl_end, BlockFrequency::Default);
}
