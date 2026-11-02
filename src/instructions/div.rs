// Port of Instructions/InstEmitDiv.cs

use crate::decoder::RegisterSize;
use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::block::BlockFrequency;
use crate::registers::GpRegisterIndex;

use super::alu_helpers::set_alu_d_or_zr;
use super::context::ArmEmitterCtx;
use super::helpers::{get_int_or_zr, get_operand_type};

pub fn emit_sdiv(ctx: &mut ArmEmitterCtx) {
    emit_div_inner(ctx, false);
}
pub fn emit_udiv(ctx: &mut ArmEmitterCtx) {
    emit_div_inner(ctx, true);
}

fn emit_div_inner(ctx: &mut ArmEmitterCtx, unsigned: bool) {
    let (rn, rm): (GpRegisterIndex, GpRegisterIndex) = match ctx.curr_op().kind {
        OpCodeKind::AluBinary { rn, rm, .. } => (rn, rm),
        _ => panic!("expected AluBinary opcode, got {:?}", ctx.curr_op().kind),
    };

    // If Rm == 0, Rd = 0 (division by zero).
    let n = get_int_or_zr(ctx, rn);
    let m = get_int_or_zr(ctx, rm);

    let m_typ = ctx.ctx.operands.get(m).typ;
    let zero = ctx.ctx.operands.constant_int(m_typ, 0);
    let divisor_is_zero = ctx.icompare_equal(m, zero);

    let lbl_bad_div = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    ctx.branch_if_true(lbl_bad_div, divisor_is_zero);

    if !unsigned {
        // If Rn == INT_MIN && Rm == -1, Rd = INT_MIN (overflow).
        let is_32_bits = ctx.curr_op().reg_size == RegisterSize::Int32;

        let int_min = if is_32_bits {
            ctx.ctx.operands.constant_i32(i32::MIN)
        } else {
            ctx.ctx.operands.constant_i64(i64::MIN)
        };
        let minus1 = if is_32_bits {
            ctx.ctx.operands.constant_i32(-1)
        } else {
            ctx.ctx.operands.constant_i64(-1)
        };

        let n_is_int_min = ctx.icompare_equal(n, int_min);
        let m_is_minus1 = ctx.icompare_equal(m, minus1);

        let lbl_good_div = ctx.ctx.operands.label();

        let both = ctx.bitwise_and(n_is_int_min, m_is_minus1);
        ctx.branch_if_false(lbl_good_div, both);

        set_alu_d_or_zr(ctx, int_min);

        ctx.branch(lbl_end);

        ctx.mark_label(lbl_good_div, BlockFrequency::Default);
    }

    let d = if unsigned {
        ctx.divide_ui(n, m)
    } else {
        ctx.divide(n, m)
    };

    set_alu_d_or_zr(ctx, d);

    ctx.branch(lbl_end);

    ctx.mark_label(lbl_bad_div, BlockFrequency::Default);

    let typ = get_operand_type(ctx.curr_op().reg_size);
    let zero_d = ctx.ctx.operands.constant_int(typ, 0);
    set_alu_d_or_zr(ctx, zero_d);

    ctx.mark_label(lbl_end, BlockFrequency::Default);
}
