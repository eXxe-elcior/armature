use crate::ir::operand::OperandIndex;
use crate::ir::types::OperandType;
use crate::state::PState;

use super::alu_helpers::*;
use super::context::ArmEmitterCtx;
use super::helpers::{get_flag, get_int_or_zr, set_flag};

pub fn emit_adc(ctx: &mut ArmEmitterCtx) {
    emit_adc_inner(ctx, false);
}
pub fn emit_adcs(ctx: &mut ArmEmitterCtx) {
    emit_adc_inner(ctx, true);
}

fn emit_adc_inner(ctx: &mut ArmEmitterCtx, set_flags: bool) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d0 = ctx.add(n, m);
    let carry0 = get_flag(PState::CFlag, &mut ctx.ctx.operands);
    let carry = if ctx.curr_op().reg_size == crate::decoder::RegisterSize::Int64 {
        ctx.zero_extend32(OperandType::I64, carry0)
    } else {
        carry0
    };
    let d = ctx.add(d0, carry);
    if set_flags {
        emit_nz_flags_check(ctx, d);
        emit_adcs_c_check(ctx, n, d);
        emit_adds_v_check(ctx, n, m, d);
    }
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_add(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d = ctx.add(n, m);
    set_alu_d(ctx, d);
}

pub fn emit_adds(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    ctx.mark_comparison(n, m);
    let d = ctx.add(n, m);
    emit_nz_flags_check(ctx, d);
    emit_adds_c_check(ctx, n, d);
    emit_adds_v_check(ctx, n, m, d);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_and(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d = ctx.bitwise_and(n, m);
    set_alu_d(ctx, d);
}

pub fn emit_ands(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d = ctx.bitwise_and(n, m);
    emit_nz_flags_check(ctx, d);
    emit_cv_flags_clear(ctx);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_asrv(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let s = get_alu_m_shift(ctx);
    let d = ctx.shift_right_si(n, s);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_bic(ctx: &mut ArmEmitterCtx) {
    emit_bic_inner(ctx, false);
}
pub fn emit_bics(ctx: &mut ArmEmitterCtx) {
    emit_bic_inner(ctx, true);
}

fn emit_bic_inner(ctx: &mut ArmEmitterCtx, set_flags: bool) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    // BIC = n & ~m = (~m) & n. With BMI1 this is a single ANDN; otherwise the
    // portable NOT + AND. Result is identical either way.
    let d = if crate::optimizations::use_bmi1() {
        ctx.bitwise_and_not(m, n)
    } else {
        let not_m = ctx.bitwise_not(m);
        ctx.bitwise_and(n, not_m)
    };
    if set_flags {
        emit_nz_flags_check(ctx, d);
        emit_cv_flags_clear(ctx);
    }
    set_alu_d(ctx, d);
}

pub fn emit_cls(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Alu { rn, .. } => rn,
        _ => panic!("emit_cls: expected Alu opcode"),
    };
    let n = get_int_or_zr(ctx, rn);
    let c1 = ctx.ctx.operands.constant_i32(1);
    let n_high = ctx.shift_right_ui(n, c1);
    let is_32_bit = ctx.curr_op().reg_size == crate::decoder::RegisterSize::Int32;
    let mask = if is_32_bit {
        ctx.ctx.operands.constant_i32(i32::MAX)
    } else {
        ctx.ctx.operands.constant_i64(i64::MAX)
    };
    let n_low = ctx.bitwise_and(n, mask);
    let xor = ctx.bitwise_xor(n_high, n_low);
    let res0 = ctx.count_leading_zeros(xor);
    // C#: Subtract(res, Const(res.Type, 1))  constant must match res type.
    let one = if is_32_bit {
        ctx.ctx.operands.constant_i32(1)
    } else {
        ctx.ctx.operands.constant_i64(1)
    };
    let res = ctx.subtract(res0, one);
    set_alu_d_or_zr(ctx, res);
}

pub fn emit_clz(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Alu { rn, .. } => rn,
        _ => panic!("emit_clz: expected Alu opcode"),
    };
    let n = get_int_or_zr(ctx, rn);
    let d = ctx.count_leading_zeros(n);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_eon(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let not_m = ctx.bitwise_not(m);
    let d = ctx.bitwise_xor(n, not_m);
    set_alu_d(ctx, d);
}

pub fn emit_eor(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d = ctx.bitwise_xor(n, m);
    set_alu_d(ctx, d);
}

pub fn emit_extr(ctx: &mut ArmEmitterCtx) {
    let (rm, rn, shift_val) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::AluRs {
            rd: _,
            rn,
            rm,
            data_op: _,
            shift,
            shift_type: _,
        } => (rm, rn, shift as u32),
        _ => panic!("emit_extr: expected AluRs opcode"),
    };
    let mut res = get_int_or_zr(ctx, rm);
    if shift_val != 0 {
        let s_op = ctx.ctx.operands.constant_i32(shift_val as i32);
        if rn == rm {
            res = ctx.rotate_right(res, s_op);
        } else {
            res = ctx.shift_right_ui(res, s_op);
            let n = get_int_or_zr(ctx, rn);
            let inv_shift = ctx.curr_op().bits() - shift_val;
            let inv_op = ctx.ctx.operands.constant_i32(inv_shift as i32);
            let shifted = ctx.shift_left(n, inv_op);
            res = ctx.bitwise_or(res, shifted);
        }
    }
    set_alu_d_or_zr(ctx, res);
}

pub fn emit_lslv(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let s = get_alu_m_shift(ctx);
    let d = ctx.shift_left(n, s);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_lsrv(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let s = get_alu_m_shift(ctx);
    let d = ctx.shift_right_ui(n, s);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_orn(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let not_m = ctx.bitwise_not(m);
    let d = ctx.bitwise_or(n, not_m);
    set_alu_d(ctx, d);
}

pub fn emit_orr(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d = ctx.bitwise_or(n, m);
    set_alu_d(ctx, d);
}

pub fn emit_rbit(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Alu { rn, .. } => rn,
        _ => panic!("emit_rbit: expected Alu opcode"),
    };
    let n = get_int_or_zr(ctx, rn);
    let d = if ctx.curr_op().reg_size == crate::decoder::RegisterSize::Int32 {
        emit_reverse_bits_32_op(ctx, n)
    } else {
        emit_reverse_bits_64_op(ctx, n)
    };
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_rev16(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Alu { rn, .. } => rn,
        _ => panic!("emit_rev16: expected Alu opcode"),
    };
    let n = get_int_or_zr(ctx, rn);
    let d = if ctx.curr_op().reg_size == crate::decoder::RegisterSize::Int32 {
        emit_reverse_bytes_16_32_op(ctx, n)
    } else {
        emit_reverse_bytes_16_64_op(ctx, n)
    };
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_rev32(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Alu { rn, .. } => rn,
        _ => panic!("emit_rev32: expected Alu opcode"),
    };
    let n = get_int_or_zr(ctx, rn);
    let d = if ctx.curr_op().reg_size == crate::decoder::RegisterSize::Int32 {
        ctx.byte_swap(n)
    } else {
        emit_reverse_bytes_32_64_op(ctx, n)
    };
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_rev64(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Alu { rn, .. } => rn,
        _ => panic!("emit_rev64: expected Alu opcode"),
    };
    let n = get_int_or_zr(ctx, rn);
    let d = ctx.byte_swap(n);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_rorv(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let s = get_alu_m_shift(ctx);
    let d = ctx.rotate_right(n, s);
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_sbc(ctx: &mut ArmEmitterCtx) {
    emit_sbc_inner(ctx, false);
}
pub fn emit_sbcs(ctx: &mut ArmEmitterCtx) {
    emit_sbc_inner(ctx, true);
}

fn emit_sbc_inner(ctx: &mut ArmEmitterCtx, set_flags: bool) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d0 = ctx.subtract(n, m);
    let flag = get_flag(PState::CFlag, &mut ctx.ctx.operands);
    let c1 = ctx.ctx.operands.constant_i32(1);
    let not_carry = ctx.bitwise_xor(flag, c1);
    let borrow0 = not_carry;
    let borrow = if ctx.curr_op().reg_size == crate::decoder::RegisterSize::Int64 {
        ctx.zero_extend32(OperandType::I64, borrow0)
    } else {
        borrow0
    };
    let d = ctx.subtract(d0, borrow);
    if set_flags {
        emit_nz_flags_check(ctx, d);
        emit_sbcs_c_check(ctx, n, m);
        emit_subs_v_check(ctx, n, m, d);
    }
    set_alu_d_or_zr(ctx, d);
}

pub fn emit_sub(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    let d = ctx.subtract(n, m);
    set_alu_d(ctx, d);
}

pub fn emit_subs(ctx: &mut ArmEmitterCtx) {
    let n = get_alu_n(ctx);
    let m = get_alu_m(ctx);
    ctx.mark_comparison(n, m);
    let d = ctx.subtract(n, m);
    emit_nz_flags_check(ctx, d);
    emit_subs_c_check(ctx, n, m);
    emit_subs_v_check(ctx, n, m, d);
    set_alu_d_or_zr(ctx, d);
}

// Private helpers
fn emit_cv_flags_clear(ctx: &mut ArmEmitterCtx) {
    let cz = ctx.ctx.operands.constant_i32(0);
    set_flag(ctx, PState::CFlag, cz);
    let vz = ctx.ctx.operands.constant_i32(0);
    set_flag(ctx, PState::VFlag, vz);
}

fn emit_reverse_bits_64_op(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    let c1 = ctx.ctx.operands.constant_i64(0xaaaaaaaaaaaaaaaa_u64 as i64);
    let c2 = ctx.ctx.operands.constant_i64(0x5555555555555555_u64 as i64);
    let c1_i = ctx.ctx.operands.constant_i32(1);
    let and1 = ctx.bitwise_and(op, c1);
    let shr1 = ctx.shift_right_ui(and1, c1_i);
    let and2 = ctx.bitwise_and(op, c2);
    let shl1 = ctx.shift_left(and2, c1_i);
    let v = ctx.bitwise_or(shr1, shl1);

    let c3 = ctx.ctx.operands.constant_i64(0xcccccccccccccccc_u64 as i64);
    let c4 = ctx.ctx.operands.constant_i64(0x3333333333333333_u64 as i64);
    let c2_i = ctx.ctx.operands.constant_i32(2);
    let and3 = ctx.bitwise_and(v, c3);
    let shr2 = ctx.shift_right_ui(and3, c2_i);
    let and4 = ctx.bitwise_and(v, c4);
    let shl2 = ctx.shift_left(and4, c2_i);
    let v = ctx.bitwise_or(shr2, shl2);

    let c5 = ctx.ctx.operands.constant_i64(0xf0f0f0f0f0f0f0f0_u64 as i64);
    let c6 = ctx.ctx.operands.constant_i64(0x0f0f0f0f0f0f0f0f_u64 as i64);
    let c4_i = ctx.ctx.operands.constant_i32(4);
    let and5 = ctx.bitwise_and(v, c5);
    let shr3 = ctx.shift_right_ui(and5, c4_i);
    let and6 = ctx.bitwise_and(v, c6);
    let shl3 = ctx.shift_left(and6, c4_i);
    let v = ctx.bitwise_or(shr3, shl3);

    let c7 = ctx.ctx.operands.constant_i64(0xff00ff00ff00ff00_u64 as i64);
    let c8 = ctx.ctx.operands.constant_i64(0x00ff00ff00ff00ff_u64 as i64);
    let c8_i = ctx.ctx.operands.constant_i32(8);
    let and7 = ctx.bitwise_and(v, c7);
    let shr4 = ctx.shift_right_ui(and7, c8_i);
    let and8 = ctx.bitwise_and(v, c8);
    let shl4 = ctx.shift_left(and8, c8_i);
    let v = ctx.bitwise_or(shr4, shl4);

    let c9 = ctx.ctx.operands.constant_i64(0xffff0000ffff0000_u64 as i64);
    let c10 = ctx.ctx.operands.constant_i64(0x0000ffff0000ffff_u64 as i64);
    let c16 = ctx.ctx.operands.constant_i32(16);
    let and9 = ctx.bitwise_and(v, c9);
    let shr5 = ctx.shift_right_ui(and9, c16);
    let and10 = ctx.bitwise_and(v, c10);
    let shl5 = ctx.shift_left(and10, c16);
    let v = ctx.bitwise_or(shr5, shl5);

    let c32 = ctx.ctx.operands.constant_i32(32);
    let shr6 = ctx.shift_right_ui(v, c32);
    let shl6 = ctx.shift_left(v, c32);
    ctx.bitwise_or(shr6, shl6)
}

fn emit_reverse_bytes_32_64_op(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    let v = emit_reverse_bytes_16_64_op(ctx, op);
    let c1 = ctx.ctx.operands.constant_i64(0xffff0000ffff0000_u64 as i64);
    let c2 = ctx.ctx.operands.constant_i64(0x0000ffff0000ffff_u64 as i64);
    let c16 = ctx.ctx.operands.constant_i32(16);
    let and1 = ctx.bitwise_and(v, c1);
    let shr = ctx.shift_right_ui(and1, c16);
    let and2 = ctx.bitwise_and(v, c2);
    let shl = ctx.shift_left(and2, c16);
    ctx.bitwise_or(shr, shl)
}
