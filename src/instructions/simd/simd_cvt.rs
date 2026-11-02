use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::helpers::{get_int_or_zr, get_vec, set_int_or_zr};
use crate::instructions::simd_helper::{
    CmpCondition, emit_sse41_round_to_nearest_with_ties_to_away_op_f, get_imm_shr,
    x86_get_all_elements_i32, x86_get_all_elements_i64, x86_get_round_control, x86_get_scalar_i32,
    x86_get_scalar_i64,
};
use crate::ir::operand::OperandIndex;
use crate::ir::types::{Intrinsic, OperandType};
use crate::optimizations;
use crate::registers::{GpRegisterIndex, VecRegisterIndex};

// Op-kind field extraction helpers
// Extract all needed values upfront to avoid borrow issues with curr_op().

fn op_fields_sim_shimm(
    ctx: &mut ArmEmitterCtx,
) -> (VecRegisterIndex, VecRegisterIndex, i32, i32, bool) {
    let op = ctx.curr_op();
    match &op.kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (
            *rd,
            *rn,
            *size as i32,
            get_imm_shr(*imm as i32, *size as i32),
            true,
        ),
        crate::decoder::OpCodeKind::SimdReg { rd, rn, size, .. } => {
            (*rd, *rn, *size as i32, 0, false)
        }
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => (*rd, *rn, *size as i32, 0, false),
        _ => unreachable!(),
    }
}

fn op_fields_simd_cvt_gp(
    ctx: &mut ArmEmitterCtx,
) -> (VecRegisterIndex, GpRegisterIndex, i32, bool, i32) {
    let op = ctx.curr_op();
    match &op.kind {
        // For float→int (Fcvtzs/Fcvtas/Fcvtms/Fcvtns/Fcvtps):
        //   rn (bits 5-9) = SIMD source, rd (bits 0-4) = GP dest
        crate::decoder::OpCodeKind::SimdCvt {
            rd,
            rn,
            size,
            fbits,
            ..
        } => (
            *rn,
            GpRegisterIndex::from(rd.idx()),
            *size as i32,
            *fbits != 0,
            *fbits as i32,
        ),
        _ => unreachable!(),
    }
}

fn op_fields_simd_cvt_gp_rn_only(
    ctx: &mut ArmEmitterCtx,
) -> (VecRegisterIndex, GpRegisterIndex, i32, i32) {
    let op = ctx.curr_op();
    match &op.kind {
        crate::decoder::OpCodeKind::SimdCvt {
            rd,
            rn,
            size,
            fbits,
            ..
        } => (
            *rd,
            GpRegisterIndex::from(rn.idx()),
            *size as i32,
            *fbits as i32,
        ),
        _ => unreachable!(),
    }
}

// Private helpers
fn emit_fp_convert(
    ctx: &mut ArmEmitterCtx,
    value: OperandIndex,
    size: i32,
    signed: bool,
) -> OperandIndex {
    let typ = if size == 0 {
        OperandType::FP32
    } else {
        OperandType::FP64
    };
    if signed {
        ctx.convert_to_fp(typ, value)
    } else {
        ctx.convert_to_fpui(typ, value)
    }
}

#[allow(dead_code)]
fn emit_f2i_fbits_mul(ctx: &mut ArmEmitterCtx, value: OperandIndex, fbits: i32) -> OperandIndex {
    if fbits == 0 {
        return value;
    }
    let typ = ctx.ctx.operands.get(value).typ;
    let multiplier = match typ {
        OperandType::FP32 => ctx.ctx.operands.constant_f32(2.0f32.powi(fbits)),
        _ => ctx.ctx.operands.constant_f64(2.0f64.powi(fbits)),
    };
    ctx.multiply(value, multiplier)
}

fn emit_i2f_fbits_mul(ctx: &mut ArmEmitterCtx, value: OperandIndex, fbits: i32) -> OperandIndex {
    if fbits == 0 {
        return value;
    }
    let typ = ctx.ctx.operands.get(value).typ;
    let multiplier = match typ {
        OperandType::FP32 => ctx.ctx.operands.constant_f32(1.0f32 / 2.0f32.powi(fbits)),
        _ => ctx.ctx.operands.constant_f64(1.0f64 / 2.0f64.powi(fbits)),
    };
    ctx.multiply(value, multiplier)
}

fn emit_sse2_cvt_double_to_int64_op_f(
    ctx: &mut ArmEmitterCtx,
    op_f: OperandIndex,
    scalar: bool,
    truncate: bool,
) -> OperandIndex {
    let cvt = if truncate {
        Intrinsic::X86Cvttsd2si
    } else {
        Intrinsic::X86Cvtsd2si
    };
    let long_l = ctx.emit_intrinsic_long(cvt, &[op_f]);
    let mut res = ctx.vector_create_scalar(long_l);
    if !scalar {
        let op_fh = ctx.emit_intrinsic(Intrinsic::X86Movhlps, &[res, op_f]);
        let long_h = ctx.emit_intrinsic_long(cvt, &[op_fh]);
        let res_h = ctx.vector_create_scalar(long_h);
        res = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[res, res_h]);
    }
    res
}

fn emit_sse2_cvt_int64_to_double_op(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    scalar: bool,
) -> OperandIndex {
    let long_l = ctx.emit_intrinsic_long(Intrinsic::X86Cvtsi2si, &[op]);
    let zero = ctx.vector_zero();
    let mut res = ctx.emit_intrinsic(Intrinsic::X86Cvtsi2sd, &[zero, long_l]);
    if !scalar {
        let op_h = ctx.emit_intrinsic(Intrinsic::X86Movhlps, &[res, op]);
        let long_h = ctx.emit_intrinsic_long(Intrinsic::X86Cvtsi2si, &[op_h]);
        let res_h = ctx.emit_intrinsic(Intrinsic::X86Cvtsi2sd, &[res, long_h]);
        res = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[res, res_h]);
    }
    res
}

// SSE2 Scvtf (signed int→float)
fn emit_sse2_scvtf_op(ctx: &mut ArmEmitterCtx, scalar: bool) {
    let (rd, rn, size, fbits, is_fixed) = op_fields_sim_shimm(ctx);
    let reg_size = ctx.curr_op().reg_size;
    let n = get_vec(ctx, rn);
    let size_f = size & 1;

    if size_f == 0 {
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Cvtdq2ps, &[n]);
        if is_fixed {
            let fp_scaled = 0x3F800000 - fbits * 0x800000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i32(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i32(ctx, fp_scaled)
            };
            res = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[res, fp_scaled_mask]);
        }
        if scalar {
            res = ctx.vector_zero_upper96(res);
        } else if matches!(reg_size, crate::decoder::RegisterSize::Simd64) {
            res = ctx.vector_zero_upper64(res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        let mut res = emit_sse2_cvt_int64_to_double_op(ctx, n, scalar);
        if is_fixed {
            let fp_scaled: i64 = 0x3FF0000000000000 - fbits as i64 * 0x10000000000000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i64(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i64(ctx, fp_scaled)
            };
            res = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[res, fp_scaled_mask]);
        }
        if scalar {
            res = ctx.vector_zero_upper64(res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    }
}

// SSE2 Ucvtf (unsigned int→float)
fn emit_sse2_ucvtf_op(ctx: &mut ArmEmitterCtx, scalar: bool) {
    let (rd, rn, size, fbits, is_fixed) = op_fields_sim_shimm(ctx);
    let reg_size = ctx.curr_op().reg_size;
    let n = get_vec(ctx, rn);
    let size_f = size & 1;

    if size_f == 0 {
        let mask = if scalar {
            x86_get_scalar_i32(ctx, 0x47800000)
        } else {
            x86_get_all_elements_i32(ctx, 0x47800000)
        };
        let ci16 = ctx.ctx.operands.constant_i32(16);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Psrld, &[n, ci16]);
        res = ctx.emit_intrinsic(Intrinsic::X86Cvtdq2ps, &[res]);
        res = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[res, mask]);

        let mut res2 = ctx.emit_intrinsic(Intrinsic::X86Pslld, &[n, ci16]);
        res2 = ctx.emit_intrinsic(Intrinsic::X86Psrld, &[res2, ci16]);
        res2 = ctx.emit_intrinsic(Intrinsic::X86Cvtdq2ps, &[res2]);

        res = ctx.emit_intrinsic(Intrinsic::X86Addps, &[res, res2]);
        if is_fixed {
            let fp_scaled = 0x3F800000 - fbits * 0x800000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i32(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i32(ctx, fp_scaled)
            };
            res = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[res, fp_scaled_mask]);
        }
        if scalar {
            res = ctx.vector_zero_upper96(res);
        } else if matches!(reg_size, crate::decoder::RegisterSize::Simd64) {
            res = ctx.vector_zero_upper64(res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        let mask = if scalar {
            x86_get_scalar_i64(ctx, 0x41F0000000000000)
        } else {
            x86_get_all_elements_i64(ctx, 0x41F0000000000000)
        };
        let ci32 = ctx.ctx.operands.constant_i32(32);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Psrlq, &[n, ci32]);
        res = emit_sse2_cvt_int64_to_double_op(ctx, res, scalar);
        res = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[res, mask]);

        let mut res2 = ctx.emit_intrinsic(Intrinsic::X86Psllq, &[n, ci32]);
        res2 = ctx.emit_intrinsic(Intrinsic::X86Psrlq, &[res2, ci32]);
        res2 = emit_sse2_cvt_int64_to_double_op(ctx, res2, scalar);

        res = ctx.emit_intrinsic(Intrinsic::X86Addpd, &[res, res2]);
        if is_fixed {
            let fp_scaled: i64 = 0x3FF0000000000000 - fbits as i64 * 0x10000000000000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i64(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i64(ctx, fp_scaled)
            };
            res = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[res, fp_scaled_mask]);
        }
        if scalar {
            res = ctx.vector_zero_upper64(res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    }
}

// SSE4.1 Fcvts (float→signed, vector ops)
fn emit_sse41_fcvts_op_f(ctx: &mut ArmEmitterCtx, round_mode: i32, scalar: bool) {
    let (rd, rn, size, fbits, is_fixed) = op_fields_sim_shimm(ctx);
    let reg_size = ctx.curr_op().reg_size;
    let n = get_vec(ctx, rn);
    let size_f = size & 1;

    let ordered_q = ctx.ctx.operands.constant_i32(CmpCondition::OrderedQ as i32);
    let not_lt = ctx
        .ctx
        .operands
        .constant_i32(CmpCondition::NotLessThan as i32);
    // round_mode 3 = truncate-toward-zero: fold round+convert into a single
    // truncating convert (cvtt*) instead of Roundps/Roundpd + Cvt*.
    let truncate = round_mode == 3;

    if size_f == 0 {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled = 0x3F800000 + fbits * 0x800000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i32(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i32(ctx, fp_scaled)
            };
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = if scalar {
                ctx.emit_intrinsic(Intrinsic::X86Roundss, &[n_res, rc])
            } else {
                ctx.emit_intrinsic(Intrinsic::X86Roundps, &[n_res, rc])
            };
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, scalar, true);
        }
        let n_int = ctx.emit_intrinsic(
            if truncate {
                Intrinsic::X86Cvttps2dq
            } else {
                Intrinsic::X86Cvtps2dq
            },
            &[n_res],
        );
        let fp_max = if scalar {
            x86_get_scalar_i32(ctx, 0x4F000000)
        } else {
            x86_get_all_elements_i32(ctx, 0x4F000000)
        };
        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[n_res, fp_max, not_lt]);
        let mut d_res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n_int, n_res]);
        if scalar {
            d_res = ctx.vector_zero_upper96(d_res);
        } else if matches!(reg_size, crate::decoder::RegisterSize::Simd64) {
            d_res = ctx.vector_zero_upper64(d_res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, d_res);
    } else {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled: i64 = 0x3FF0000000000000 + fbits as i64 * 0x10000000000000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i64(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i64(ctx, fp_scaled)
            };
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = if scalar {
                ctx.emit_intrinsic(Intrinsic::X86Roundsd, &[n_res, rc])
            } else {
                ctx.emit_intrinsic(Intrinsic::X86Roundpd, &[n_res, rc])
            };
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, scalar, false);
        }
        let n_long = emit_sse2_cvt_double_to_int64_op_f(ctx, n_res, scalar, truncate);
        let fp_max = if scalar {
            x86_get_scalar_i64(ctx, 0x43E0000000000000)
        } else {
            x86_get_all_elements_i64(ctx, 0x43E0000000000000)
        };
        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[n_res, fp_max, not_lt]);
        let d_res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n_long, n_res]);
        let d_res = if scalar {
            ctx.vector_zero_upper64(d_res)
        } else {
            d_res
        };
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, d_res);
    }
}

// SSE4.1 Fcvtu (float→unsigned, vector ops)
fn emit_sse41_fcvtu_op_f(ctx: &mut ArmEmitterCtx, round_mode: i32, scalar: bool) {
    let (rd, rn, size, fbits, is_fixed) = op_fields_sim_shimm(ctx);
    let reg_size = ctx.curr_op().reg_size;
    let n = get_vec(ctx, rn);
    let size_f = size & 1;

    let ordered_q = ctx.ctx.operands.constant_i32(CmpCondition::OrderedQ as i32);
    let not_le = ctx
        .ctx
        .operands
        .constant_i32(CmpCondition::NotLessThanOrEqual as i32);
    let not_lt = ctx
        .ctx
        .operands
        .constant_i32(CmpCondition::NotLessThan as i32);
    // round_mode 3 = truncate-toward-zero: fold round+convert into cvtt*.
    let truncate = round_mode == 3;

    if size_f == 0 {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled = 0x3F800000 + fbits * 0x800000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i32(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i32(ctx, fp_scaled)
            };
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = if scalar {
                ctx.emit_intrinsic(Intrinsic::X86Roundss, &[n_res, rc])
            } else {
                ctx.emit_intrinsic(Intrinsic::X86Roundps, &[n_res, rc])
            };
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, scalar, true);
        }
        let cvt = if truncate {
            Intrinsic::X86Cvttps2dq
        } else {
            Intrinsic::X86Cvtps2dq
        };
        let zero = ctx.vector_zero();
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);
        let fp_max = if scalar {
            x86_get_scalar_i32(ctx, 0x4F000000)
        } else {
            x86_get_all_elements_i32(ctx, 0x4F000000)
        };
        let n_int = ctx.emit_intrinsic(cvt, &[n_res]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Subps, &[n_res, fp_max]);
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);
        let n_int2 = ctx.emit_intrinsic(cvt, &[n_res]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[n_res, fp_max, not_lt]);
        let mut d_res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n_int2, n_res]);
        d_res = ctx.emit_intrinsic(Intrinsic::X86Paddd, &[d_res, n_int]);
        if scalar {
            d_res = ctx.vector_zero_upper96(d_res);
        } else if matches!(reg_size, crate::decoder::RegisterSize::Simd64) {
            d_res = ctx.vector_zero_upper64(d_res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, d_res);
    } else {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled: i64 = 0x3FF0000000000000 + fbits as i64 * 0x10000000000000;
            let fp_scaled_mask = if scalar {
                x86_get_scalar_i64(ctx, fp_scaled)
            } else {
                x86_get_all_elements_i64(ctx, fp_scaled)
            };
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = if scalar {
                ctx.emit_intrinsic(Intrinsic::X86Roundsd, &[n_res, rc])
            } else {
                ctx.emit_intrinsic(Intrinsic::X86Roundpd, &[n_res, rc])
            };
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, scalar, false);
        }
        let zero = ctx.vector_zero();
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);
        let fp_max = if scalar {
            x86_get_scalar_i64(ctx, 0x43E0000000000000)
        } else {
            x86_get_all_elements_i64(ctx, 0x43E0000000000000)
        };
        let n_long = emit_sse2_cvt_double_to_int64_op_f(ctx, n_res, scalar, truncate);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Subpd, &[n_res, fp_max]);
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);
        let n_long2 = emit_sse2_cvt_double_to_int64_op_f(ctx, n_res, scalar, truncate);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[n_res, fp_max, not_lt]);
        let d_res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n_long2, n_res]);
        let d_res = ctx.emit_intrinsic(Intrinsic::X86Paddq, &[d_res, n_long]);
        let d_res = if scalar {
            ctx.vector_zero_upper64(d_res)
        } else {
            d_res
        };
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, d_res);
    }
}

// SSE4.1 Fcvts_Gp (float→signed GP register ops)
fn emit_sse41_fcvts_gp(ctx: &mut ArmEmitterCtx, round_mode: i32, is_fixed: bool) {
    let (rd_vec, rn_gpr, size, _, fbits) = op_fields_simd_cvt_gp(ctx);
    let reg_size = ctx.curr_op().reg_size;
    let n = get_vec(ctx, rd_vec);
    let ordered_q = ctx.ctx.operands.constant_i32(CmpCondition::OrderedQ as i32);
    let not_lt = ctx
        .ctx
        .operands
        .constant_i32(CmpCondition::NotLessThan as i32);
    // round_mode 3 = truncate-toward-zero: fold round+convert into cvtt*.
    let truncate = round_mode == 3;

    if size == 0 {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpss, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled = 0x3F800000 + fbits * 0x800000;
            let fp_scaled_mask = x86_get_scalar_i32(ctx, fp_scaled);
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = ctx.emit_intrinsic(Intrinsic::X86Roundss, &[n_res, rc]);
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, true, true);
        }

        let cvt = if truncate {
            Intrinsic::X86Cvttss2si
        } else {
            Intrinsic::X86Cvtss2si
        };
        let n_int_or_long = if reg_size == crate::decoder::RegisterSize::Int32 {
            ctx.emit_intrinsic_int(cvt, &[n_res])
        } else {
            ctx.emit_intrinsic_long(cvt, &[n_res])
        };

        let fp_max_val = if reg_size == crate::decoder::RegisterSize::Int32 {
            0x4F000000i32
        } else {
            0x5F000000i32
        };
        let fp_max_val_mask = x86_get_scalar_i32(ctx, fp_max_val);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpss, &[n_res, fp_max_val_mask, not_lt]);

        let mut n_int = ctx.emit_intrinsic_int(Intrinsic::X86Cvtsi2si, &[n_res]);
        if reg_size == crate::decoder::RegisterSize::Int64 {
            n_int = ctx.sign_extend32(OperandType::I64, n_int);
        }
        let d_res = ctx.bitwise_xor(n_int_or_long, n_int);
        set_int_or_zr(ctx, rn_gpr, d_res);
    } else {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpsd, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled: i64 = 0x3FF0000000000000 + fbits as i64 * 0x10000000000000;
            let fp_scaled_mask = x86_get_scalar_i64(ctx, fp_scaled);
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = ctx.emit_intrinsic(Intrinsic::X86Roundsd, &[n_res, rc]);
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, true, false);
        }

        let cvt = if truncate {
            Intrinsic::X86Cvttsd2si
        } else {
            Intrinsic::X86Cvtsd2si
        };
        let n_int_or_long = if reg_size == crate::decoder::RegisterSize::Int32 {
            ctx.emit_intrinsic_int(cvt, &[n_res])
        } else {
            ctx.emit_intrinsic_long(cvt, &[n_res])
        };

        let fp_max_val = if reg_size == crate::decoder::RegisterSize::Int32 {
            0x41E0000000000000i64
        } else {
            0x43E0000000000000i64
        };
        let fp_max_val_mask = x86_get_scalar_i64(ctx, fp_max_val);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpsd, &[n_res, fp_max_val_mask, not_lt]);

        let mut n_long = ctx.emit_intrinsic_long(Intrinsic::X86Cvtsi2si, &[n_res]);
        if reg_size == crate::decoder::RegisterSize::Int32 {
            n_long = ctx.convert_i64_to_i32(n_long);
        }
        let d_res = ctx.bitwise_xor(n_int_or_long, n_long);
        set_int_or_zr(ctx, rn_gpr, d_res);
    }
}

// SSE4.1 Fcvtu_Gp (float→unsigned GP register ops)
fn emit_sse41_fcvtu_gp(ctx: &mut ArmEmitterCtx, round_mode: i32, is_fixed: bool) {
    let (rd_vec, rn_gpr, size, _, fbits) = op_fields_simd_cvt_gp(ctx);
    let reg_size = ctx.curr_op().reg_size;
    let n = get_vec(ctx, rd_vec);
    let ordered_q = ctx.ctx.operands.constant_i32(CmpCondition::OrderedQ as i32);
    let not_le = ctx
        .ctx
        .operands
        .constant_i32(CmpCondition::NotLessThanOrEqual as i32);
    let not_lt = ctx
        .ctx
        .operands
        .constant_i32(CmpCondition::NotLessThan as i32);
    // round_mode 3 = truncate-toward-zero: fold round+convert into cvtt*.
    let truncate = round_mode == 3;
    if size == 0 {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpss, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled = 0x3F800000 + fbits * 0x800000;
            let fp_scaled_mask = x86_get_scalar_i32(ctx, fp_scaled);
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = ctx.emit_intrinsic(Intrinsic::X86Roundss, &[n_res, rc]);
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, true, true);
        }
        let cvt = if truncate {
            Intrinsic::X86Cvttss2si
        } else {
            Intrinsic::X86Cvtss2si
        };
        let zero = ctx.vector_zero();
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmpss, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);

        let fp_max_val = if reg_size == crate::decoder::RegisterSize::Int32 {
            0x4F000000i32
        } else {
            0x5F000000i32
        };
        let fp_max_val_mask = x86_get_scalar_i32(ctx, fp_max_val);

        let n_int_or_long = if reg_size == crate::decoder::RegisterSize::Int32 {
            ctx.emit_intrinsic_int(cvt, &[n_res])
        } else {
            ctx.emit_intrinsic_long(cvt, &[n_res])
        };

        n_res = ctx.emit_intrinsic(Intrinsic::X86Subss, &[n_res, fp_max_val_mask]);
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmpss, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);

        let n_int_or_long2 = if reg_size == crate::decoder::RegisterSize::Int32 {
            ctx.emit_intrinsic_int(cvt, &[n_res])
        } else {
            ctx.emit_intrinsic_long(cvt, &[n_res])
        };

        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpss, &[n_res, fp_max_val_mask, not_lt]);
        let mut n_int = ctx.emit_intrinsic_int(Intrinsic::X86Cvtsi2si, &[n_res]);
        if reg_size == crate::decoder::RegisterSize::Int64 {
            n_int = ctx.sign_extend32(OperandType::I64, n_int);
        }
        let d_res = ctx.bitwise_xor(n_int_or_long2, n_int);
        let d_res = ctx.add(d_res, n_int_or_long);
        set_int_or_zr(ctx, rn_gpr, d_res);
    } else {
        let mut n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpsd, &[n, n, ordered_q]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n]);
        if is_fixed {
            let fp_scaled: i64 = 0x3FF0000000000000 + fbits as i64 * 0x10000000000000;
            let fp_scaled_mask = x86_get_scalar_i64(ctx, fp_scaled);
            n_res = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n_res, fp_scaled_mask]);
        }
        if round_mode != 4 && !truncate {
            let rc = ctx
                .ctx
                .operands
                .constant_i32(x86_get_round_control(round_mode));
            n_res = ctx.emit_intrinsic(Intrinsic::X86Roundsd, &[n_res, rc]);
        } else if round_mode == 4 {
            n_res = emit_sse41_round_to_nearest_with_ties_to_away_op_f(ctx, n_res, true, false);
        }
        let cvt = if truncate {
            Intrinsic::X86Cvttsd2si
        } else {
            Intrinsic::X86Cvtsd2si
        };
        let zero = ctx.vector_zero();
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmpsd, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);

        let fp_max_val = if reg_size == crate::decoder::RegisterSize::Int32 {
            0x41E0000000000000i64
        } else {
            0x43E0000000000000i64
        };
        let fp_max_val_mask = x86_get_scalar_i64(ctx, fp_max_val);

        let n_int_or_long = if reg_size == crate::decoder::RegisterSize::Int32 {
            ctx.emit_intrinsic_int(cvt, &[n_res])
        } else {
            ctx.emit_intrinsic_long(cvt, &[n_res])
        };

        n_res = ctx.emit_intrinsic(Intrinsic::X86Subsd, &[n_res, fp_max_val_mask]);
        let n_cmp = ctx.emit_intrinsic(Intrinsic::X86Cmpsd, &[n_res, zero, not_le]);
        n_res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n_res, n_cmp]);

        let n_int_or_long2 = if reg_size == crate::decoder::RegisterSize::Int32 {
            ctx.emit_intrinsic_int(cvt, &[n_res])
        } else {
            ctx.emit_intrinsic_long(cvt, &[n_res])
        };

        n_res = ctx.emit_intrinsic(Intrinsic::X86Cmpsd, &[n_res, fp_max_val_mask, not_lt]);
        let mut n_long = ctx.emit_intrinsic_long(Intrinsic::X86Cvtsi2si, &[n_res]);
        if reg_size == crate::decoder::RegisterSize::Int32 {
            n_long = ctx.convert_i64_to_i32(n_long);
        }
        let d_res = ctx.bitwise_xor(n_int_or_long2, n_long);
        let d_res = ctx.add(d_res, n_int_or_long);
        set_int_or_zr(ctx, rn_gpr, d_res);
    }
}

// SoftFallback paths (stubs)
fn emit_fcvt(_ctx: &mut ArmEmitterCtx, _signed: bool, _scalar: bool) {
    todo!("emit_fcvt (SoftFallback) for InstEmitSimdCvt");
}
fn emit_fcvtz(_ctx: &mut ArmEmitterCtx, _signed: bool, _scalar: bool) {
    todo!("emit_fcvtz (SoftFallback) for InstEmitSimdCvt");
}
fn emit_cvtf(_ctx: &mut ArmEmitterCtx, _signed: bool, _scalar: bool) {
    todo!("emit_cvtf (SoftFallback) for InstEmitSimdCvt");
}
fn emit_fcvt_s_gp(_ctx: &mut ArmEmitterCtx, _signed: bool) {
    todo!("emit_fcvt_s_gp (SoftFallback) for InstEmitSimdCvt");
}
fn emit_fcvt_u_gp(_ctx: &mut ArmEmitterCtx, _signed: bool) {
    todo!("emit_fcvt_u_gp (SoftFallback) for InstEmitSimdCvt");
}

// Public: Fcvt_S (half-precision conversions)
pub fn emit_fcvt_s(ctx: &mut ArmEmitterCtx) {
    let (size, opc, rd, rn) = {
        let op = ctx.curr_op();
        match &op.kind {
            crate::decoder::OpCodeKind::SimdReg {
                size, opc, rd, rn, ..
            }
            | crate::decoder::OpCodeKind::Simd {
                size, opc, rd, rn, ..
            } => (*size as i32, *opc as i32, *rd, *rn),
            _ => unreachable!(),
        }
    };

    if size == 0 && opc == 1 {
        let n = get_vec(ctx, rn);
        let zero = ctx.vector_zero();
        if optimizations::use_sse2() {
            let res = ctx.emit_intrinsic(Intrinsic::X86Cvtss2sd, &[zero, n]);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("SoftFloat Fcvt Single->Double");
        }
    } else if size == 1 && opc == 0 {
        let n = get_vec(ctx, rn);
        let zero = ctx.vector_zero();
        if optimizations::use_sse2() {
            let res = ctx.emit_intrinsic(Intrinsic::X86Cvtsd2ss, &[zero, n]);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("SoftFloat Fcvt Double->Single");
        }
    } else if size == 0 && opc == 3 {
        let n = get_vec(ctx, rn);
        if optimizations::use_f16c() {
            let rc = ctx.ctx.operands.constant_i32(x86_get_round_control(0));
            let ci14 = ctx.ctx.operands.constant_i32(14);
            let res = ctx.emit_intrinsic(Intrinsic::X86Vcvtps2ph, &[n, rc]);
            let res = ctx.emit_intrinsic(Intrinsic::X86Pslldq, &[res, ci14]);
            let res = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[res, ci14]);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("SoftFloat32_16 Fcvt Single->Half");
        }
    } else if size == 3 && opc == 0 {
        let n = get_vec(ctx, rn);
        if optimizations::use_f16c() {
            let res = ctx.emit_intrinsic(Intrinsic::X86Vcvtph2ps, &[n]);
            let res = ctx.vector_zero_upper96(res);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("SoftFloat16_32 Fcvt Half->Single");
        }
    } else if size == 1 && opc == 3 {
        let n = get_vec(ctx, rn);
        if optimizations::use_f16c() && optimizations::use_sse2() {
            let zero = ctx.vector_zero();
            let res = ctx.emit_intrinsic(Intrinsic::X86Cvtsd2ss, &[zero, n]);
            let rc = ctx.ctx.operands.constant_i32(x86_get_round_control(0));
            let res = ctx.emit_intrinsic(Intrinsic::X86Vcvtps2ph, &[res, rc]);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("SoftFloat64_16 Fcvt Double->Half");
        }
    } else if size == 3 && opc == 1 {
        let n = get_vec(ctx, rn);
        if optimizations::use_f16c() && optimizations::use_sse2() {
            let zero = ctx.vector_zero();
            let res = ctx.emit_intrinsic(Intrinsic::X86Vcvtph2ps, &[n]);
            let res = ctx.emit_intrinsic(Intrinsic::X86Cvtss2sd, &[zero, res]);
            let res = ctx.vector_zero_upper64(res);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("SoftFloat16_64 Fcvt Half->Double");
        }
    } else {
        panic!("Invalid Fcvt encoding: size={size} opc={opc}");
    }
}

// Public: Fcvtas (float→signed, nearest-away)
pub fn emit_fcvtas_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_gp(ctx, 4, false);
    } else {
        todo!("Fcvtas_Gp");
    }
}
pub fn emit_fcvtas_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 4, true);
    } else {
        todo!("Fcvtas_S");
    }
}
pub fn emit_fcvtas_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 4, false);
    } else {
        todo!("Fcvtas_V");
    }
}

// Public: Fcvtau (float→unsigned, nearest-away)
pub fn emit_fcvtau_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_gp(ctx, 4, false);
    } else {
        todo!("Fcvtau_Gp");
    }
}
pub fn emit_fcvtau_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_op_f(ctx, 4, true);
    } else {
        todo!("Fcvtau_S");
    }
}
pub fn emit_fcvtau_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_op_f(ctx, 4, false);
    } else {
        todo!("Fcvtau_V");
    }
}

// Public: Fcvtl_V (lengthen)
pub fn emit_fcvtl_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = {
        let op = ctx.curr_op();
        match &op.kind {
            crate::decoder::OpCodeKind::Simd { size, rd, rn, .. } => (*rd, *rn, *size as i32),
            _ => unreachable!(),
        }
    };
    let reg_size = ctx.curr_op().reg_size;
    let size_f = size & 1;

    if size_f == 1 {
        if optimizations::use_sse2() {
            let n = get_vec(ctx, rn);
            let res = if reg_size == crate::decoder::RegisterSize::Simd128 {
                ctx.emit_intrinsic(Intrinsic::X86Movhlps, &[n, n])
            } else {
                n
            };
            let res = ctx.emit_intrinsic(Intrinsic::X86Cvtps2pd, &[res]);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("Fcvtl_V f32->f64");
        }
    } else if size_f == 0 {
        if optimizations::use_f16c() {
            let n = get_vec(ctx, rn);
            let res = if reg_size == crate::decoder::RegisterSize::Simd128 {
                ctx.emit_intrinsic(Intrinsic::X86Movhlps, &[n, n])
            } else {
                n
            };
            let res = ctx.emit_intrinsic(Intrinsic::X86Vcvtph2ps, &[res]);
            let rd_op = get_vec(ctx, rd);
            ctx.copy_to(rd_op, res);
        } else {
            todo!("Fcvtl_V f16->f32");
        }
    }
}

// Public: Fcvtms / Fcvtmu (float→signed/unsigned, −∞)
pub fn emit_fcvtms_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_gp(ctx, 2, false);
    } else {
        emit_fcvt_s_gp(ctx, true);
    }
}
pub fn emit_fcvtms_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 2, false);
    } else {
        emit_fcvt(ctx, true, false);
    }
}
pub fn emit_fcvtmu_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_gp(ctx, 2, false);
    } else {
        emit_fcvt_u_gp(ctx, false);
    }
}
pub fn emit_fcvtms_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 2, true);
    } else {
        emit_fcvt(ctx, true, true);
    }
}

// Public: Fcvtn_V (narrow)
pub fn emit_fcvtn_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = {
        let op = ctx.curr_op();
        match &op.kind {
            crate::decoder::OpCodeKind::Simd { size, rd, rn, .. } => (*rd, *rn, *size as i32),
            _ => unreachable!(),
        }
    };
    let reg_size = ctx.curr_op().reg_size;
    let size_f = size & 1;

    if size_f == 1 {
        if optimizations::use_sse2() {
            let d = get_vec(ctx, rd);
            let mov_inst = if reg_size == crate::decoder::RegisterSize::Simd128 {
                Intrinsic::X86Movlhps
            } else {
                Intrinsic::X86Movhlps
            };
            let n = get_vec(ctx, rn);
            let n_int = ctx.emit_intrinsic(Intrinsic::X86Cvtpd2ps, &[n]);
            let n_int = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[n_int, n_int]);
            let res = ctx.vector_zero_upper64(d);
            let res = ctx.emit_intrinsic(mov_inst, &[res, n_int]);
            ctx.copy_to(d, res);
        } else {
            todo!("Fcvtn_V f64->f32");
        }
    } else if size_f == 0 {
        if optimizations::use_f16c() {
            let d = get_vec(ctx, rd);
            let mov_inst = if reg_size == crate::decoder::RegisterSize::Simd128 {
                Intrinsic::X86Movlhps
            } else {
                Intrinsic::X86Movhlps
            };
            let rc = ctx.ctx.operands.constant_i32(x86_get_round_control(0));
            let n = get_vec(ctx, rn);
            let n_int = ctx.emit_intrinsic(Intrinsic::X86Vcvtps2ph, &[n, rc]);
            let n_int = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[n_int, n_int]);
            let res = ctx.vector_zero_upper64(d);
            let res = ctx.emit_intrinsic(mov_inst, &[res, n_int]);
            ctx.copy_to(d, res);
        } else {
            todo!("Fcvtn_V f32->f16");
        }
    }
}

// Public: Fcvtns / Fcvtnu (float→signed/unsigned, nearest-even)
pub fn emit_fcvtns_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_gp(ctx, 0, false);
    } else {
        emit_fcvt_s_gp(ctx, true);
    }
}
pub fn emit_fcvtns_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 0, true);
    } else {
        emit_fcvt(ctx, true, true);
    }
}
pub fn emit_fcvtns_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 0, false);
    } else {
        emit_fcvt(ctx, true, false);
    }
}
pub fn emit_fcvtnu_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_op_f(ctx, 0, true);
    } else {
        emit_fcvt(ctx, false, true);
    }
}
pub fn emit_fcvtnu_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_op_f(ctx, 0, false);
    } else {
        emit_fcvt(ctx, false, false);
    }
}

// Public: Fcvtps / Fcvtpu (float→signed/unsigned, +∞)
pub fn emit_fcvtps_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_gp(ctx, 1, false);
    } else {
        emit_fcvt_s_gp(ctx, true);
    }
}
pub fn emit_fcvtpu_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_gp(ctx, 1, false);
    } else {
        emit_fcvt_u_gp(ctx, false);
    }
}

// Public: Fcvtzs / Fcvtzu (float→signed/unsigned, towards zero)
pub fn emit_fcvtzs_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_gp(ctx, 3, false);
    } else {
        todo!("Fcvtzs_Gp");
    }
}
pub fn emit_fcvtzs_gp_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_gp(ctx, 3, true);
    } else {
        todo!("Fcvtzs_Gp_Fixed");
    }
}
pub fn emit_fcvtzs_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 3, true);
    } else {
        emit_fcvtz(ctx, true, true);
    }
}
pub fn emit_fcvtzs_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 3, false);
    } else {
        emit_fcvtz(ctx, true, false);
    }
}
pub fn emit_fcvtzs_v_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvts_op_f(ctx, 3, false);
    } else {
        emit_fcvtz(ctx, true, false);
    }
}
pub fn emit_fcvtzu_gp(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_gp(ctx, 3, false);
    } else {
        todo!("Fcvtzu_Gp");
    }
}
pub fn emit_fcvtzu_gp_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_gp(ctx, 3, true);
    } else {
        todo!("Fcvtzu_Gp_Fixed");
    }
}
pub fn emit_fcvtzu_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_op_f(ctx, 3, true);
    } else {
        emit_fcvtz(ctx, false, true);
    }
}
pub fn emit_fcvtzu_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_op_f(ctx, 3, false);
    } else {
        emit_fcvtz(ctx, false, false);
    }
}
pub fn emit_fcvtzu_v_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        emit_sse41_fcvtu_op_f(ctx, 3, false);
    } else {
        emit_fcvtz(ctx, false, false);
    }
}

// Public: Scvtf (signed int→float)
pub fn emit_scvtf_gp(ctx: &mut ArmEmitterCtx) {
    let (rd, rn_gpr, size) = {
        let (rd, rn, size, _) = op_fields_simd_cvt_gp_rn_only(ctx);
        (rd, rn, size)
    };
    let reg_size = ctx.curr_op().reg_size;
    let mut res = get_int_or_zr(ctx, rn_gpr);
    if reg_size == crate::decoder::RegisterSize::Int32 {
        res = ctx.sign_extend32(OperandType::I64, res);
    }
    res = emit_fp_convert(ctx, res, size, true);
    let zero = ctx.vector_zero();
    let inserted = ctx.vector_insert(zero, res, 0);
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, inserted);
}

pub fn emit_scvtf_gp_fixed(ctx: &mut ArmEmitterCtx) {
    let (rd, rn_gpr, size, fbits) = op_fields_simd_cvt_gp_rn_only(ctx);
    let reg_size = ctx.curr_op().reg_size;
    let mut res = get_int_or_zr(ctx, rn_gpr);
    if reg_size == crate::decoder::RegisterSize::Int32 {
        res = ctx.sign_extend32(OperandType::I64, res);
    }
    res = emit_fp_convert(ctx, res, size, true);
    res = emit_i2f_fbits_mul(ctx, res, fbits);
    let zero = ctx.vector_zero();
    let inserted = ctx.vector_insert(zero, res, 0);
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, inserted);
}

pub fn emit_scvtf_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_scvtf_op(ctx, true);
    } else {
        emit_cvtf(ctx, true, true);
    }
}
pub fn emit_scvtf_s_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_scvtf_op(ctx, true);
    } else {
        emit_cvtf(ctx, true, true);
    }
}
pub fn emit_scvtf_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_scvtf_op(ctx, false);
    } else {
        emit_cvtf(ctx, true, false);
    }
}
pub fn emit_scvtf_v_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_scvtf_op(ctx, false);
    } else {
        emit_cvtf(ctx, true, false);
    }
}

// Public: Ucvtf (unsigned int→float)
pub fn emit_ucvtf_gp(ctx: &mut ArmEmitterCtx) {
    let (rd, rn_gpr, size) = {
        let (rd, rn, size, _) = op_fields_simd_cvt_gp_rn_only(ctx);
        (rd, rn, size)
    };
    let res = get_int_or_zr(ctx, rn_gpr);
    let res = emit_fp_convert(ctx, res, size, false);
    let zero = ctx.vector_zero();
    let inserted = ctx.vector_insert(zero, res, 0);
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, inserted);
}

pub fn emit_ucvtf_gp_fixed(ctx: &mut ArmEmitterCtx) {
    let (rd, rn_gpr, size, fbits) = op_fields_simd_cvt_gp_rn_only(ctx);
    let mut res = get_int_or_zr(ctx, rn_gpr);
    res = emit_fp_convert(ctx, res, size, false);
    res = emit_i2f_fbits_mul(ctx, res, fbits);
    let zero = ctx.vector_zero();
    let inserted = ctx.vector_insert(zero, res, 0);
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, inserted);
}

pub fn emit_ucvtf_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_ucvtf_op(ctx, true);
    } else {
        emit_cvtf(ctx, false, true);
    }
}
pub fn emit_ucvtf_s_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_ucvtf_op(ctx, true);
    } else {
        emit_cvtf(ctx, false, true);
    }
}
pub fn emit_ucvtf_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_ucvtf_op(ctx, false);
    } else {
        emit_cvtf(ctx, false, false);
    }
}
pub fn emit_ucvtf_v_fixed(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        emit_sse2_ucvtf_op(ctx, false);
    } else {
        emit_cvtf(ctx, false, false);
    }
}
