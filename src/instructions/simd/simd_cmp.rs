use crate::decoder::utilities::RegisterSize;
use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::flow_helpers::get_cond_true;
use crate::instructions::helpers::{get_vec, set_flag};
use crate::instructions::simd_helper::{
    CmpCondition, X86_PCMPEQ_INSTRUCTION, X86_PCMPGT_INSTRUCTION, X86_PMAXU_INSTRUCTION,
    bytes_count, emit_soft_float_call, emit_vector_extract_sx, emit_vector_extract_zx,
    emit_vector_insert, x86_get_all_elements_i32, x86_get_all_elements_i64, x86_get_scalar_i32,
    x86_get_scalar_i64,
};
use crate::ir::block::BlockFrequency;
use crate::ir::operand::OperandIndex;
use crate::ir::types::{Intrinsic, OperandType};
use crate::optimizations;
use crate::registers::VecRegisterIndex;

fn emit_cmp_op(
    ctx: &mut ArmEmitterCtx,
    emit_cmp: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
    scalar: bool,
) {
    let op = ctx.curr_op();
    let (rd, rn, rm_opt, size, reg_size) = match &op.kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (*rd, *rn, Some(*rm), *size as i32, op.reg_size),
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
            (*rd, *rn, None, *size as i32, op.reg_size)
        }
        _ => unreachable!("emit_cmp_op: unexpected opcode"),
    };

    let elems = if !scalar {
        bytes_count(reg_size) >> size
    } else {
        1
    };
    let sz_mask = u64::MAX >> (64 - (8 << size));

    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_sx(ctx, rn, index, size);
        let me = match rm_opt {
            Some(rm) => emit_vector_extract_sx(ctx, rm, index, size),
            None => ctx.ctx.operands.constant_i64(0),
        };
        let is_true = emit_cmp(ctx, ne, me);
        let sz_mask_op = ctx.ctx.operands.constant_i64(sz_mask as i64);
        let zero_op = ctx.ctx.operands.constant_i64(0);
        let mask = ctx.conditional_select(is_true, sz_mask_op, zero_op);
        res = emit_vector_insert(ctx, res, mask, index, size);
    }
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, res);
}

fn emit_cmtst_op(ctx: &mut ArmEmitterCtx, scalar: bool) {
    let op = ctx.curr_op();
    let (rd, rn, rm, size, reg_size) = match &op.kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (*rd, *rn, *rm, *size as i32, op.reg_size),
        _ => unreachable!("emit_cmtst_op: expected SimdReg"),
    };

    let elems = if !scalar {
        bytes_count(reg_size) >> size
    } else {
        1
    };
    let sz_mask = u64::MAX >> (64 - (8 << size));

    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_zx(ctx, rn, index, size);
        let me = emit_vector_extract_zx(ctx, rm, index, size);
        let test = ctx.bitwise_and(ne, me);
        let zero_op = ctx.ctx.operands.constant_i64(0);
        let is_true = ctx.icompare_not_equal(test, zero_op);
        let sz_mask_op = ctx.ctx.operands.constant_i64(sz_mask as i64);
        let mask = ctx.conditional_select(is_true, sz_mask_op, zero_op);
        res = emit_vector_insert(ctx, res, mask, index, size);
    }
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, res);
}

fn emit_sse2_or_avx_cmp_op_f(
    ctx: &mut ArmEmitterCtx,
    cond: CmpCondition,
    scalar: bool,
    absolute: bool,
) {
    let op = ctx.curr_op();
    let (rd, rn, rm_opt, size, reg_size) = match &op.kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (*rd, *rn, Some(*rm), *size as i32, op.reg_size),
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
            (*rd, *rn, None, *size as i32, op.reg_size)
        }
        crate::decoder::OpCodeKind::SimdFcond {
            rd, rn, rm, size, ..
        } => (*rd, *rn, Some(*rm), *size as i32, op.reg_size),
        _ => unreachable!("emit_sse2_or_avx_cmp_op_f: unexpected opcode"),
    };

    let n = get_vec(ctx, rn);
    let m = match rm_opt {
        Some(rm) => get_vec(ctx, rm),
        None => ctx.vector_zero(),
    };

    let size_f = size & 1;

    if size_f == 0 {
        let n = if absolute {
            let mask = if scalar {
                x86_get_scalar_i32(ctx, i32::MAX)
            } else {
                x86_get_all_elements_i32(ctx, i32::MAX)
            };
            ctx.emit_intrinsic(Intrinsic::X86Andps, &[n, mask])
        } else {
            n
        };
        let m = if absolute {
            let mask = if scalar {
                x86_get_scalar_i32(ctx, i32::MAX)
            } else {
                x86_get_all_elements_i32(ctx, i32::MAX)
            };
            ctx.emit_intrinsic(Intrinsic::X86Andps, &[m, mask])
        } else {
            m
        };

        let inst = if scalar {
            Intrinsic::X86Cmpss
        } else {
            Intrinsic::X86Cmpps
        };
        let cond_op = ctx.ctx.operands.constant_i32(cond as i32);
        let mut res = ctx.emit_intrinsic(inst, &[n, m, cond_op]);

        if scalar {
            res = ctx.vector_zero_upper96(res);
        } else if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        let n = if absolute {
            let mask = if scalar {
                x86_get_scalar_i64(ctx, i64::MAX)
            } else {
                x86_get_all_elements_i64(ctx, i64::MAX)
            };
            ctx.emit_intrinsic(Intrinsic::X86Andpd, &[n, mask])
        } else {
            n
        };
        let m = if absolute {
            let mask = if scalar {
                x86_get_scalar_i64(ctx, i64::MAX)
            } else {
                x86_get_all_elements_i64(ctx, i64::MAX)
            };
            ctx.emit_intrinsic(Intrinsic::X86Andpd, &[m, mask])
        } else {
            m
        };

        let inst = if scalar {
            Intrinsic::X86Cmpsd
        } else {
            Intrinsic::X86Cmppd
        };
        let cond_op = ctx.ctx.operands.constant_i32(cond as i32);
        let mut res = ctx.emit_intrinsic(inst, &[n, m, cond_op]);

        if scalar {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    }
}

fn emit_cmp_op_f(ctx: &mut ArmEmitterCtx, name: &str, scalar: bool, absolute: bool) {
    let op = ctx.curr_op();
    let (rd, rn, rm_opt, size, reg_size) = match &op.kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (*rd, *rn, Some(*rm), *size as i32, op.reg_size),
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
            (*rd, *rn, None, *size as i32, op.reg_size)
        }
        _ => unreachable!("emit_cmp_op_f: unexpected opcode"),
    };

    let size_f = size & 1;
    let typ = if size_f != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let elems = if !scalar {
        bytes_count(reg_size) >> (size_f + 2)
    } else {
        1
    };

    let mut res = ctx.vector_zero();
    let vec_rn = get_vec(ctx, rn);
    for index in 0..elems {
        let mut ne = ctx.vector_extract(typ, vec_rn, index);
        let mut me = match rm_opt {
            Some(rm) => {
                let vec_rm = get_vec(ctx, rm);
                ctx.vector_extract(typ, vec_rm, index)
            }
            None => {
                if size_f == 0 {
                    ctx.ctx.operands.constant_f32(0.0)
                } else {
                    ctx.ctx.operands.constant_f64(0.0)
                }
            }
        };

        if absolute {
            ne = emit_soft_float_call(ctx, "Abs", &[ne]);
            me = emit_soft_float_call(ctx, "Abs", &[me]);
        }

        let e = emit_soft_float_call(ctx, name, &[ne, me]);
        res = ctx.vector_insert(res, e, index);
    }

    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, res);
}

fn emit_fcmp_or_fcmpe(ctx: &mut ArmEmitterCtx, signal_na_ns: bool) {
    let op = ctx.curr_op();
    let is_fcond = matches!(&op.kind, crate::decoder::OpCodeKind::SimdFcond { .. });
    let cmp_with_zero = !is_fcond
        && match &op.kind {
            crate::decoder::OpCodeKind::SimdReg { bit3, .. } => *bit3,
            _ => false,
        };

    let (rn, rm_opt, size) = match &op.kind {
        crate::decoder::OpCodeKind::SimdReg { rn, rm, size, .. } => (*rn, Some(*rm), *size as i32),
        crate::decoder::OpCodeKind::SimdFcond { rn, rm, size, .. } => {
            (*rn, Some(*rm), *size as i32)
        }
        _ => (VecRegisterIndex::new(0), None, 0),
    };

    let n = get_vec(ctx, rn);
    let m = if cmp_with_zero {
        ctx.vector_zero()
    } else {
        match rm_opt {
            Some(rm) => get_vec(ctx, rm),
            None => ctx.vector_zero(),
        }
    };

    if optimizations::fast_fp()
        && (if signal_na_ns {
            optimizations::use_avx()
        } else {
            optimizations::use_sse2()
        })
    {
        let cmp_ordered = if signal_na_ns {
            CmpCondition::OrderedS
        } else {
            CmpCondition::OrderedQ
        };

        let lbl_nan = ctx.ctx.operands.label();
        let lbl_end = ctx.ctx.operands.label();

        if size == 0 {
            let ord_q = ctx.ctx.operands.constant_i32(cmp_ordered as i32);
            let ord_mask = ctx.emit_intrinsic(Intrinsic::X86Cmpss, &[n, m, ord_q]);

            let is_ordered = ctx.emit_intrinsic_int(Intrinsic::X86Cvtsi2si, &[ord_mask]);

            ctx.branch_if_false(lbl_nan, is_ordered);

            let n_copy = ctx.copy(n);
            let m_copy = if cmp_with_zero {
                ctx.vector_zero()
            } else {
                ctx.copy(m)
            };

            let cf = ctx.emit_intrinsic_int(Intrinsic::X86Comissge, &[n_copy, m_copy]);
            let zf = ctx.emit_intrinsic_int(Intrinsic::X86Comisseq, &[n_copy, m_copy]);
            let nf = ctx.emit_intrinsic_int(Intrinsic::X86Comisslt, &[n_copy, m_copy]);

            let zero_op = ctx.ctx.operands.constant_i32(0);
            set_flag(ctx, crate::state::PState::VFlag, zero_op);
            set_flag(ctx, crate::state::PState::CFlag, cf);
            set_flag(ctx, crate::state::PState::ZFlag, zf);
            set_flag(ctx, crate::state::PState::NFlag, nf);
        } else {
            let ord_q = ctx.ctx.operands.constant_i32(cmp_ordered as i32);
            let ord_mask = ctx.emit_intrinsic(Intrinsic::X86Cmpsd, &[n, m, ord_q]);

            let is_ordered = ctx.emit_intrinsic_long(Intrinsic::X86Cvtsi2si, &[ord_mask]);

            ctx.branch_if_false(lbl_nan, is_ordered);

            let n_copy = ctx.copy(n);
            let m_copy = if cmp_with_zero {
                ctx.vector_zero()
            } else {
                ctx.copy(m)
            };

            let cf = ctx.emit_intrinsic_int(Intrinsic::X86Comisdge, &[n_copy, m_copy]);
            let zf = ctx.emit_intrinsic_int(Intrinsic::X86Comisdeq, &[n_copy, m_copy]);
            let nf = ctx.emit_intrinsic_int(Intrinsic::X86Comisdlt, &[n_copy, m_copy]);

            let zero_op = ctx.ctx.operands.constant_i32(0);
            set_flag(ctx, crate::state::PState::VFlag, zero_op);
            set_flag(ctx, crate::state::PState::CFlag, cf);
            set_flag(ctx, crate::state::PState::ZFlag, zf);
            set_flag(ctx, crate::state::PState::NFlag, nf);
        }

        ctx.branch(lbl_end);

        ctx.mark_label(lbl_nan, BlockFrequency::Default);

        let one_op = ctx.ctx.operands.constant_i32(1);
        let zero_op = ctx.ctx.operands.constant_i32(0);
        set_flag(ctx, crate::state::PState::VFlag, one_op);
        set_flag(ctx, crate::state::PState::CFlag, one_op);
        set_flag(ctx, crate::state::PState::ZFlag, zero_op);
        set_flag(ctx, crate::state::PState::NFlag, zero_op);

        ctx.mark_label(lbl_end, BlockFrequency::Default);
    } else {
        let op_size = size;
        let typ = if op_size != 0 {
            OperandType::FP64
        } else {
            OperandType::FP32
        };

        let ne = ctx.vector_extract(typ, n, 0);
        let me = if cmp_with_zero {
            if op_size == 0 {
                ctx.ctx.operands.constant_f32(0.0)
            } else {
                ctx.ctx.operands.constant_f64(0.0)
            }
        } else {
            ctx.vector_extract(typ, m, 0)
        };

        let signal_op = ctx.ctx.operands.constant_i32(signal_na_ns as i32);
        let nzcv = emit_soft_float_call(ctx, "FPCompare", &[ne, me, signal_op]);
        emit_set_nzcv(ctx, nzcv);
    }
}

fn extract_bit(ctx: &mut ArmEmitterCtx, value: OperandIndex, bit: i32) -> OperandIndex {
    let v = if bit != 0 {
        let bit_op = ctx.ctx.operands.constant_i32(bit);
        ctx.shift_right_ui(value, bit_op)
    } else {
        value
    };
    let one_op = ctx.ctx.operands.constant_i32(1);
    ctx.bitwise_and(v, one_op)
}

fn emit_set_nzcv(ctx: &mut ArmEmitterCtx, nzcv: OperandIndex) {
    let v = extract_bit(ctx, nzcv, 0);
    let c = extract_bit(ctx, nzcv, 1);
    let z = extract_bit(ctx, nzcv, 2);
    let n = extract_bit(ctx, nzcv, 3);
    set_flag(ctx, crate::state::PState::VFlag, v);
    set_flag(ctx, crate::state::PState::CFlag, c);
    set_flag(ctx, crate::state::PState::ZFlag, z);
    set_flag(ctx, crate::state::PState::NFlag, n);
}

fn emit_set_nzcv_const(ctx: &mut ArmEmitterCtx, nzcv: u8) {
    let extract = |value: u8, bit: i32| -> i32 {
        let v = if bit != 0 { value >> bit } else { value };
        (v & 1) as i32
    };

    let v_op = ctx.ctx.operands.constant_i32(extract(nzcv, 0));
    let c_op = ctx.ctx.operands.constant_i32(extract(nzcv, 1));
    let z_op = ctx.ctx.operands.constant_i32(extract(nzcv, 2));
    let n_op = ctx.ctx.operands.constant_i32(extract(nzcv, 3));

    set_flag(ctx, crate::state::PState::VFlag, v_op);
    set_flag(ctx, crate::state::PState::CFlag, c_op);
    set_flag(ctx, crate::state::PState::ZFlag, z_op);
    set_flag(ctx, crate::state::PState::NFlag, n_op);
}

fn emit_fccmp_or_fccmpe(ctx: &mut ArmEmitterCtx, signal_na_ns: bool) {
    let op = ctx.curr_op();
    let (cond, nzcv) = match &op.kind {
        crate::decoder::OpCodeKind::SimdFcond { cond, nzcv, .. } => (*cond, *nzcv),
        _ => unreachable!("emit_fccmp_or_fccmpe: expected SimdFcond"),
    };

    let lbl_true = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    let cond_true = get_cond_true(ctx, cond);
    ctx.branch_if_true(lbl_true, cond_true);

    emit_set_nzcv_const(ctx, nzcv);

    ctx.branch(lbl_end);

    ctx.mark_label(lbl_true, BlockFrequency::Default);

    emit_fcmp_or_fcmpe(ctx, signal_na_ns);

    ctx.mark_label(lbl_end, BlockFrequency::Default);
}

// Integer compare
pub fn emit_cmeq_s(ctx: &mut ArmEmitterCtx) {
    emit_cmp_op(ctx, |ctx, op1, op2| ctx.icompare_equal(op1, op2), true);
}

pub fn emit_cmeq_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse41() {
        let op = ctx.curr_op();
        let (rd, rn, rm_opt, size, reg_size) = match &op.kind {
            crate::decoder::OpCodeKind::SimdReg {
                rd, rn, rm, size, ..
            } => (*rd, *rn, Some(*rm), *size as i32, op.reg_size),
            crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
                (*rd, *rn, None, *size as i32, op.reg_size)
            }
            _ => unreachable!("emit_cmeq_v: unexpected opcode"),
        };

        let n = get_vec(ctx, rn);
        let m = match rm_opt {
            Some(rm) => get_vec(ctx, rm),
            None => ctx.vector_zero(),
        };

        let cmp_inst = X86_PCMPEQ_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(cmp_inst, &[n, m]);

        if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        emit_cmp_op(ctx, |ctx, op1, op2| ctx.icompare_equal(op1, op2), false);
    }
}

pub fn emit_cmge_s(ctx: &mut ArmEmitterCtx) {
    emit_cmp_op(
        ctx,
        |ctx, op1, op2| ctx.icompare_greater_or_equal(op1, op2),
        true,
    );
}

pub fn emit_cmge_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse42() {
        let op = ctx.curr_op();
        let (rd, rn, rm_opt, size, reg_size) = match &op.kind {
            crate::decoder::OpCodeKind::SimdReg {
                rd, rn, rm, size, ..
            } => (*rd, *rn, Some(*rm), *size as i32, op.reg_size),
            crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
                (*rd, *rn, None, *size as i32, op.reg_size)
            }
            _ => unreachable!("emit_cmge_v: unexpected opcode"),
        };

        let n = get_vec(ctx, rn);
        let m = match rm_opt {
            Some(rm) => get_vec(ctx, rm),
            None => ctx.vector_zero(),
        };

        let cmp_inst = X86_PCMPGT_INSTRUCTION[size as usize];
        let res0 = ctx.emit_intrinsic(cmp_inst, &[m, n]);

        let mask = x86_get_all_elements_i64(ctx, -1i64);
        let res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[res0, mask]);

        let mut res = res;
        if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        emit_cmp_op(
            ctx,
            |ctx, op1, op2| ctx.icompare_greater_or_equal(op1, op2),
            false,
        );
    }
}

pub fn emit_cmgt_s(ctx: &mut ArmEmitterCtx) {
    emit_cmp_op(ctx, |ctx, op1, op2| ctx.icompare_greater(op1, op2), true);
}

pub fn emit_cmgt_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse42() {
        let op = ctx.curr_op();
        let (rd, rn, rm_opt, size, reg_size) = match &op.kind {
            crate::decoder::OpCodeKind::SimdReg {
                rd, rn, rm, size, ..
            } => (*rd, *rn, Some(*rm), *size as i32, op.reg_size),
            crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
                (*rd, *rn, None, *size as i32, op.reg_size)
            }
            _ => unreachable!("emit_cmgt_v: unexpected opcode"),
        };

        let n = get_vec(ctx, rn);
        let m = match rm_opt {
            Some(rm) => get_vec(ctx, rm),
            None => ctx.vector_zero(),
        };

        let cmp_inst = X86_PCMPGT_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(cmp_inst, &[n, m]);

        if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        emit_cmp_op(ctx, |ctx, op1, op2| ctx.icompare_greater(op1, op2), false);
    }
}

pub fn emit_cmhi_s(ctx: &mut ArmEmitterCtx) {
    emit_cmp_op(ctx, |ctx, op1, op2| ctx.icompare_greater_ui(op1, op2), true);
}

pub fn emit_cmhi_v(ctx: &mut ArmEmitterCtx) {
    let op = ctx.curr_op();
    let (rd, rn, rm, size, reg_size) = match &op.kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (*rd, *rn, *rm, *size as i32, op.reg_size),
        _ => unreachable!("emit_cmhi_v: expected SimdReg"),
    };

    if optimizations::use_sse41() && size < 3 {
        let n = get_vec(ctx, rn);
        let m = get_vec(ctx, rm);

        let max_inst = X86_PMAXU_INSTRUCTION[size as usize];
        let res0 = ctx.emit_intrinsic(max_inst, &[m, n]);

        let cmp_inst = X86_PCMPEQ_INSTRUCTION[size as usize];
        let res1 = ctx.emit_intrinsic(cmp_inst, &[res0, m]);

        let mask = x86_get_all_elements_i64(ctx, -1i64);
        let res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[res1, mask]);

        let mut res = res;
        if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        emit_cmp_op(
            ctx,
            |ctx, op1, op2| ctx.icompare_greater_ui(op1, op2),
            false,
        );
    }
}

pub fn emit_cmhs_s(ctx: &mut ArmEmitterCtx) {
    emit_cmp_op(
        ctx,
        |ctx, op1, op2| ctx.icompare_greater_or_equal_ui(op1, op2),
        true,
    );
}

pub fn emit_cmhs_v(ctx: &mut ArmEmitterCtx) {
    let op = ctx.curr_op();
    let (rd, rn, rm, size, reg_size) = match &op.kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (*rd, *rn, *rm, *size as i32, op.reg_size),
        _ => unreachable!("emit_cmhs_v: expected SimdReg"),
    };

    if optimizations::use_sse41() && size < 3 {
        let n = get_vec(ctx, rn);
        let m = get_vec(ctx, rm);

        let max_inst = X86_PMAXU_INSTRUCTION[size as usize];
        let res0 = ctx.emit_intrinsic(max_inst, &[n, m]);

        let cmp_inst = X86_PCMPEQ_INSTRUCTION[size as usize];
        let res = ctx.emit_intrinsic(cmp_inst, &[res0, n]);

        let mut res = res;
        if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        emit_cmp_op(
            ctx,
            |ctx, op1, op2| ctx.icompare_greater_or_equal_ui(op1, op2),
            false,
        );
    }
}

pub fn emit_cmle_s(ctx: &mut ArmEmitterCtx) {
    emit_cmp_op(
        ctx,
        |ctx, op1, op2| ctx.icompare_less_or_equal(op1, op2),
        true,
    );
}

pub fn emit_cmle_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse42() {
        let op = ctx.curr_op();
        let (rd, rn, size, reg_size) = match &op.kind {
            crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
                (*rd, *rn, *size as i32, op.reg_size)
            }
            crate::decoder::OpCodeKind::SimdReg { rd, rn, size, .. } => {
                (*rd, *rn, *size as i32, op.reg_size)
            }
            _ => unreachable!("emit_cmle_v: unexpected opcode"),
        };

        let n = get_vec(ctx, rn);
        let cmp_inst = X86_PCMPGT_INSTRUCTION[size as usize];
        let zero_vec = ctx.vector_zero();
        let res0 = ctx.emit_intrinsic(cmp_inst, &[n, zero_vec]);

        let mask = x86_get_all_elements_i64(ctx, -1i64);
        let res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[res0, mask]);

        let mut res = res;
        if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        emit_cmp_op(
            ctx,
            |ctx, op1, op2| ctx.icompare_less_or_equal(op1, op2),
            false,
        );
    }
}

pub fn emit_cmlt_s(ctx: &mut ArmEmitterCtx) {
    emit_cmp_op(ctx, |ctx, op1, op2| ctx.icompare_less(op1, op2), true);
}

pub fn emit_cmlt_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse42() {
        let op = ctx.curr_op();
        let (rd, rn, size, reg_size) = match &op.kind {
            crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => {
                (*rd, *rn, *size as i32, op.reg_size)
            }
            crate::decoder::OpCodeKind::SimdReg { rd, rn, size, .. } => {
                (*rd, *rn, *size as i32, op.reg_size)
            }
            _ => unreachable!("emit_cmlt_v: unexpected opcode"),
        };

        let cmp_inst = X86_PCMPGT_INSTRUCTION[size as usize];
        let zero_vec = ctx.vector_zero();
        let vec_rn = get_vec(ctx, rn);
        let mut res = ctx.emit_intrinsic(cmp_inst, &[zero_vec, vec_rn]);

        if reg_size == RegisterSize::Simd64 {
            res = ctx.vector_zero_upper64(res);
        }

        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        emit_cmp_op(ctx, |ctx, op1, op2| ctx.icompare_less(op1, op2), false);
    }
}

pub fn emit_cmtst_s(ctx: &mut ArmEmitterCtx) {
    emit_cmtst_op(ctx, true);
}

pub fn emit_cmtst_v(ctx: &mut ArmEmitterCtx) {
    emit_cmtst_op(ctx, false);
}

// Float compare
pub fn emit_facge_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThanOrEqual, true, true);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGE", true, true);
    }
}

pub fn emit_facge_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThanOrEqual, false, true);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGE", false, true);
    }
}

pub fn emit_facgt_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThan, true, true);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGT", true, true);
    }
}

pub fn emit_facgt_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThan, false, true);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGT", false, true);
    }
}

pub fn emit_fccmp_s(ctx: &mut ArmEmitterCtx) {
    emit_fccmp_or_fccmpe(ctx, false);
}

pub fn emit_fccmpe_s(ctx: &mut ArmEmitterCtx) {
    emit_fccmp_or_fccmpe(ctx, true);
}

pub fn emit_fcmeq_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_sse2() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::Equal, true, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareEQ", true, false);
    }
}

pub fn emit_fcmeq_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_sse2() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::Equal, false, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareEQ", false, false);
    }
}

pub fn emit_fcmge_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThanOrEqual, true, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGE", true, false);
    }
}

pub fn emit_fcmge_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThanOrEqual, false, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGE", false, false);
    }
}

pub fn emit_fcmgt_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThan, true, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGT", true, false);
    }
}

pub fn emit_fcmgt_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_avx() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::GreaterThan, false, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareGT", false, false);
    }
}

pub fn emit_fcmle_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_sse2() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::LessThanOrEqual, true, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareLE", true, false);
    }
}

pub fn emit_fcmle_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_sse2() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::LessThanOrEqual, false, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareLE", false, false);
    }
}

pub fn emit_fcmlt_s(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_sse2() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::LessThan, true, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareLT", true, false);
    }
}

pub fn emit_fcmlt_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::fast_fp() && optimizations::use_sse2() {
        emit_sse2_or_avx_cmp_op_f(ctx, CmpCondition::LessThan, false, false);
    } else {
        emit_cmp_op_f(ctx, "FPCompareLT", false, false);
    }
}

pub fn emit_fcmp_s(ctx: &mut ArmEmitterCtx) {
    emit_fcmp_or_fcmpe(ctx, false);
}

pub fn emit_fcmpe_s(ctx: &mut ArmEmitterCtx) {
    emit_fcmp_or_fcmpe(ctx, true);
}
