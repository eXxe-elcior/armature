// Port of InstEmitSimdMove.cs
// Host x86-64 only: AdvSimd paths excluded. SSE paths preferred, IR fallback.

use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::flow_helpers;
use crate::instructions::helpers::{get_int_or_zr, get_vec, set_int_or_zr};
use crate::instructions::simd_helper::*;
use crate::ir::types::{Intrinsic, OperandType};
use crate::registers::{GpRegisterIndex, VecRegisterIndex};

// Masks (from C# _masksE0_Uzp, _masksE1_Uzp)
const MASKS_E0_UZP: [i64; 2] = [0x0d0905010c080400, 0x0b0a030209080100];

const MASKS_E1_UZP: [i64; 2] = [0x0f0b07030e0a0602, 0x0f0e07060d0c0504];

// ══════════════════════════════════════════════════════════════════════════════
// Dup_Gp
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_dup_gp(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdIns { rd, rn, size, .. } => (rd, rn, size as i32),
        _ => panic!("emit_dup_gp: expected SimdIns"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let mut n = get_int_or_zr(ctx, GpRegisterIndex::from(rn.idx()));
    // C# switches on op.Size keeping n.Type (I64 for vector reg sizes),
    // then VectorInsert + Shufps (sizes 0-2) or Movlhps (size 3).
    let n_typ = ctx.ctx.operands.get(n).typ;
    match size {
        0 => {
            n = ctx.zero_extend8(n_typ, n);
            let mul_val = ctx.ctx.operands.constant_int(n_typ, 0x01010101);
            n = ctx.multiply(n, mul_val);
        }
        1 => {
            n = ctx.zero_extend16(n_typ, n);
            let mul_val = ctx.ctx.operands.constant_int(n_typ, 0x00010001);
            n = ctx.multiply(n, mul_val);
        }
        2 => {
            n = ctx.zero_extend32(n_typ, n);
        }
        _ => {}
    }
    let __vz_0 = ctx.vector_zero();
    let mut res = ctx.vector_insert(__vz_0, n, 0);
    if size < 3 {
        let mask = if reg_size.bits() == 64 { 0xf0 } else { 0 };
        let __c_0 = ctx.ctx.operands.constant_i32(mask);
        res = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[res, res, __c_0]);
    } else {
        res = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[res, res]);
    }
    let __gv_0 = get_vec(ctx, rd);
    ctx.copy_to(__gv_0, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Dup_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_dup_s(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, dst_index) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdIns {
            rd,
            rn,
            size,
            dst_index,
            ..
        } => (rd, rn, size as i32, dst_index as i32),
        _ => panic!("emit_dup_s: expected SimdIns"),
    };
    let ne = emit_vector_extract_zx(ctx, rn, dst_index, size);
    let __gv_0 = get_vec(ctx, rd);
    let __vz_0 = ctx.vector_zero();
    let __vi_1 = emit_vector_insert(ctx, __vz_0, ne, 0, size);
    ctx.copy_to(__gv_0, __vi_1);
}

// ══════════════════════════════════════════════════════════════════════════════
// Dup_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_dup_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, dst_index) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdIns {
            rd,
            rn,
            size,
            dst_index,
            ..
        } => (rd, rn, size as i32, dst_index as i32),
        _ => panic!("emit_dup_v: expected SimdIns"),
    };
    let reg_size = ctx.curr_op().reg_size;
    if size == 0 {
        let mut res = get_vec(ctx, rn);
        if dst_index != 0 {
            let __c_0 = ctx.ctx.operands.constant_i32(dst_index);
            res = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[res, __c_0]);
        }
        res = ctx.emit_intrinsic(Intrinsic::X86Punpcklbw, &[res, res]);
        res = ctx.emit_intrinsic(Intrinsic::X86Punpcklwd, &[res, res]);
        let __c_0 = ctx.ctx.operands.constant_i32(0);
        res = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[res, res, __c_0]);
        let mut res = res;
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    } else if size == 1 {
        let mut res = get_vec(ctx, rn);
        if dst_index != 0 {
            let __c_0 = ctx.ctx.operands.constant_i32(dst_index * 2);
            res = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[res, __c_0]);
        }
        res = ctx.emit_intrinsic(Intrinsic::X86Punpcklwd, &[res, res]);
        let __c_0 = ctx.ctx.operands.constant_i32(0);
        res = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[res, res, __c_0]);
        let mut res = res;
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    } else if size == 2 {
        let mask = dst_index * 0b01010101;
        let __gv_0 = get_vec(ctx, rn);
        let __c_1 = ctx.ctx.operands.constant_i32(mask);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[__gv_0, __gv_0, __c_1]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_2 = get_vec(ctx, rd);
        ctx.copy_to(__gv_2, res);
    } else if dst_index == 0 && reg_size.bits() != 64 {
        let __gv_0 = get_vec(ctx, rn);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[__gv_0, __gv_0]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_1 = get_vec(ctx, rd);
        ctx.copy_to(__gv_1, res);
    } else if dst_index == 1 {
        let __gv_0 = get_vec(ctx, rn);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Movhlps, &[__gv_0, __gv_0]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_1 = get_vec(ctx, rd);
        ctx.copy_to(__gv_1, res);
    } else {
        let ne = emit_vector_extract_zx(ctx, rn, dst_index, size);
        let elems = bytes_count(reg_size) >> size;
        let mut res = ctx.vector_zero();
        for index in 0..elems {
            res = emit_vector_insert(ctx, res, ne, index, size);
        }
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Ext_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ext_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm, imm4) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdExt {
            rd, rn, rm, imm4, ..
        } => (rd, rn, rm, imm4 as i32),
        _ => panic!("emit_ext_v: expected SimdExt"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let bytes = bytes_count(reg_size);
    let mut n_shifted = get_vec(ctx, rn);
    if reg_size.bits() == 64 {
        n_shifted = ctx.vector_zero_upper64(n_shifted);
    }
    let __c_0 = ctx.ctx.operands.constant_i32(imm4);
    n_shifted = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n_shifted, __c_0]);
    let mut m_shifted = get_vec(ctx, rm);
    let __c_0 = ctx.ctx.operands.constant_i32(bytes - imm4);
    m_shifted = ctx.emit_intrinsic(Intrinsic::X86Pslldq, &[m_shifted, __c_0]);
    if reg_size.bits() == 64 {
        m_shifted = ctx.vector_zero_upper64(m_shifted);
    }
    let res = ctx.emit_intrinsic(Intrinsic::X86Por, &[n_shifted, m_shifted]);
    let __gv_0 = get_vec(ctx, rd);
    ctx.copy_to(__gv_0, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Fcsel_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fcsel_s(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdFcond { rd, rn, rm, .. } => (rd, rn, rm),
        _ => panic!("emit_fcsel_s: expected SimdFcond"),
    };
    let size = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdFcond { size, .. } => size as i32,
        _ => unreachable!(),
    };
    let cond = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdFcond { cond, .. } => cond,
        _ => unreachable!(),
    };
    let lbl_true = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();
    let is_true = flow_helpers::get_cond_true(ctx, cond);
    ctx.branch_if_true(lbl_true, is_true);
    let fp_type = if size == 0 {
        OperandType::FP32
    } else {
        OperandType::FP64
    };
    let __gv_0 = get_vec(ctx, rm);
    let me = ctx.vector_extract(fp_type, __gv_0, 0);
    let __gv_1 = get_vec(ctx, rd);
    let __vz_0 = ctx.vector_zero();
    let __vi_2 = ctx.vector_insert(__vz_0, me, 0);
    ctx.copy_to(__gv_1, __vi_2);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_true, crate::ir::block::BlockFrequency::Default);
    let __gv_0 = get_vec(ctx, rn);
    let ne = ctx.vector_extract(fp_type, __gv_0, 0);
    let __gv_1 = get_vec(ctx, rd);
    let __vz_0 = ctx.vector_zero();
    let __vi_2 = ctx.vector_insert(__vz_0, ne, 0);
    ctx.copy_to(__gv_1, __vi_2);
    ctx.mark_label(lbl_end, crate::ir::block::BlockFrequency::Default);
}

// ══════════════════════════════════════════════════════════════════════════════
// Fmov_Ftoi
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fmov_ftoi(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => (rd, rn, size as i32),
        _ => panic!("emit_fmov_ftoi: expected Simd"),
    };
    let ne = emit_vector_extract_zx(ctx, rn, 0, size + 2);
    set_int_or_zr(ctx, GpRegisterIndex::from(rd.idx()), ne);
}

// ══════════════════════════════════════════════════════════════════════════════
// Fmov_Ftoi1
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fmov_ftoi1(ctx: &mut ArmEmitterCtx) {
    let (rd, rn) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, .. } => (rd, rn),
        _ => panic!("emit_fmov_ftoi1: expected Simd"),
    };
    let ne = emit_vector_extract_zx(ctx, rn, 1, 3);
    set_int_or_zr(ctx, GpRegisterIndex::from(rd.idx()), ne);
}

// ══════════════════════════════════════════════════════════════════════════════
// Fmov_Itof
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fmov_itof(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => (rd, rn, size as i32),
        _ => panic!("emit_fmov_itof: expected Simd"),
    };
    let n = get_int_or_zr(ctx, GpRegisterIndex::from(rn.idx()));
    let __gv_0 = get_vec(ctx, rd);
    let __vz_0 = ctx.vector_zero();
    let __vi_1 = emit_vector_insert(ctx, __vz_0, n, 0, size + 2);
    ctx.copy_to(__gv_0, __vi_1);
}

// ══════════════════════════════════════════════════════════════════════════════
// Fmov_Itof1
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fmov_itof1(ctx: &mut ArmEmitterCtx) {
    let (rd, rn) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, .. } => (rd, rn),
        _ => panic!("emit_fmov_itof1: expected Simd"),
    };
    let d = get_vec(ctx, rd);
    let n = get_int_or_zr(ctx, GpRegisterIndex::from(rn.idx()));
    let __vi_0 = emit_vector_insert(ctx, d, n, 1, 3);
    ctx.copy_to(d, __vi_0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Fmov_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fmov_s(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => (rd, rn, size as i32),
        _ => panic!("emit_fmov_s: expected Simd"),
    };
    let fp_type = if size == 0 {
        OperandType::FP32
    } else {
        OperandType::FP64
    };
    let __gv_0 = get_vec(ctx, rn);
    let ne = ctx.vector_extract(fp_type, __gv_0, 0);
    let __gv_1 = get_vec(ctx, rd);
    let __vz_0 = ctx.vector_zero();
    let __vi_2 = ctx.vector_insert(__vz_0, ne, 0);
    ctx.copy_to(__gv_1, __vi_2);
}

// ══════════════════════════════════════════════════════════════════════════════
// Fmov_Si
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fmov_si(ctx: &mut ArmEmitterCtx) {
    let (rd, immediate, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdFmov {
            rd,
            immediate,
            size,
        } => (rd, immediate, size as i32),
        _ => panic!("emit_fmov_si: expected SimdFmov"),
    };
    if size == 0 {
        let res = x86_get_scalar_i32(ctx, immediate as i32);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    } else {
        let res = x86_get_scalar_i64(ctx, immediate);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Fmov_Vi
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_fmov_vi(ctx: &mut ArmEmitterCtx) {
    let (rd, immediate) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdImm { rd, immediate, .. } => (rd, immediate),
        _ => panic!("emit_fmov_vi: expected SimdImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    if reg_size.bits() == 128 {
        let res = x86_get_all_elements_i64(ctx, immediate);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    } else {
        let res = x86_get_scalar_i64(ctx, immediate);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Ins_Gp
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ins_gp(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, dst_index) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdIns {
            rd,
            rn,
            size,
            dst_index,
            ..
        } => (rd, rn, size as i32, dst_index as i32),
        _ => panic!("emit_ins_gp: expected SimdIns"),
    };
    let d = get_vec(ctx, rd);
    let n = get_int_or_zr(ctx, GpRegisterIndex::from(rn.idx()));
    let __vi_0 = emit_vector_insert(ctx, d, n, dst_index, size);
    ctx.copy_to(d, __vi_0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Ins_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ins_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, dst_index, src_index) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdIns {
            rd,
            rn,
            size,
            dst_index,
            src_index,
        } => (rd, rn, size as i32, dst_index as i32, src_index as i32),
        _ => panic!("emit_ins_v: expected SimdIns"),
    };
    let d = get_vec(ctx, rd);
    let ne = emit_vector_extract_zx(ctx, rn, src_index, size);
    let __vi_0 = emit_vector_insert(ctx, d, ne, dst_index, size);
    ctx.copy_to(d, __vi_0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Movi_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_movi_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_movi_mvni_op(ctx, false);
}

// ══════════════════════════════════════════════════════════════════════════════
// Mvni_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_mvni_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_movi_mvni_op(ctx, true);
}

// ══════════════════════════════════════════════════════════════════════════════
// Smov_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_smov_s(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, dst_index) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdIns {
            rd,
            rn,
            size,
            dst_index,
            ..
        } => (rd, rn, size as i32, dst_index as i32),
        _ => panic!("emit_smov_s: expected SimdIns"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let mut ne = emit_vector_extract_sx(ctx, rn, dst_index, size);
    if reg_size.bits() == 64 {
        ne = ctx.zero_extend32(OperandType::I64, ne);
    }
    set_int_or_zr(ctx, GpRegisterIndex::from(rd.idx()), ne);
}

// ══════════════════════════════════════════════════════════════════════════════
// Tbl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_tbl_v(ctx: &mut ArmEmitterCtx) {
    emit_table_vector_lookup(ctx, true);
}

// ══════════════════════════════════════════════════════════════════════════════
// Tbx_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_tbx_v(ctx: &mut ArmEmitterCtx) {
    emit_table_vector_lookup(ctx, false);
}

// ══════════════════════════════════════════════════════════════════════════════
// Trn1_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_trn1_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_transpose(ctx, 0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Trn2_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_trn2_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_transpose(ctx, 1);
}

// ══════════════════════════════════════════════════════════════════════════════
// Umov_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_umov_s(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, dst_index) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdIns {
            rd,
            rn,
            size,
            dst_index,
            ..
        } => (rd, rn, size as i32, dst_index as i32),
        _ => panic!("emit_umov_s: expected SimdIns"),
    };
    let ne = emit_vector_extract_zx(ctx, rn, dst_index, size);
    set_int_or_zr(ctx, GpRegisterIndex::from(rd.idx()), ne);
}

// ══════════════════════════════════════════════════════════════════════════════
// Uzp1_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uzp1_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_unzip(ctx, 0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Uzp2_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uzp2_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_unzip(ctx, 1);
}

// ══════════════════════════════════════════════════════════════════════════════
// Xtn_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_xtn_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => (rd, rn, size as i32),
        _ => panic!("emit_xtn_v: expected Simd"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let d = get_vec(ctx, rd);
    let mut res = ctx.vector_zero_upper64(d);
    let mask = x86_get_all_elements_i64(ctx, EVEN_MASKS[size as usize]);
    let __gv_0 = get_vec(ctx, rn);
    let res2 = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[__gv_0, mask]);
    let mov_inst = if reg_size.bits() == 128 {
        Intrinsic::X86Movlhps
    } else {
        Intrinsic::X86Movhlps
    };
    res = ctx.emit_intrinsic(mov_inst, &[res, res2]);
    ctx.copy_to(d, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Zip1_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_zip1_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_zip(ctx, 0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Zip2_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_zip2_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_zip(ctx, 1);
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitSse2VectorMoviMvniOp (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_sse2_vector_movi_mvni_op(ctx: &mut ArmEmitterCtx, not: bool) {
    let (rd, size, immediate) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdImm {
            rd,
            size,
            immediate,
        } => (rd, size as i32, immediate),
        _ => panic!("emit_sse2_vector_movi_mvni_op: expected SimdImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let mut imm = immediate;
    match size {
        0 => imm *= 0x01010101,
        1 => imm *= 0x00010001,
        _ => {}
    }
    if not {
        imm = !imm;
    }
    let mask = if size < 3 {
        x86_get_all_elements_i32(ctx, imm as i32)
    } else {
        x86_get_all_elements_i64(ctx, imm)
    };
    let mut mask = mask;
    if reg_size.bits() == 64 {
        mask = ctx.vector_zero_upper64(mask);
    }
    let __gv_0 = get_vec(ctx, rd);
    ctx.copy_to(__gv_0, mask);
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitTableVectorLookup (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_table_vector_lookup(ctx: &mut ArmEmitterCtx, is_tbl: bool) {
    let (rd, rn, rm, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdTbl {
            rd, rn, rm, size, ..
        } => (rd, rn, rm, size as i32),
        _ => panic!("emit_table_vector_lookup: expected SimdTbl"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let d = get_vec(ctx, rd);
    let m = get_vec(ctx, rm);
    let mask = x86_get_all_elements_i64(ctx, 0x0f0f0f0f0f0f0f0f);
    // Fast path for single register table.
    let n = get_vec(ctx, rn);
    let m_mask = ctx.emit_intrinsic(Intrinsic::X86Pcmpgtb, &[m, mask]);
    let m_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[m_mask, m]);
    let mut res = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n, m_mask]);
    for index in 1..size {
        let ni = get_vec(
            ctx,
            VecRegisterIndex::new((rn.idx() as i32 + index) as u32 & 0x1f),
        );
        let idx_mask = x86_get_all_elements_i64(ctx, 0x1010101010101010 * index as i64);
        let m_sub_mask = ctx.emit_intrinsic(Intrinsic::X86Psubb, &[m, idx_mask]);
        let m_mask = ctx.emit_intrinsic(Intrinsic::X86Pcmpgtb, &[m_sub_mask, mask]);
        let m_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[m_mask, m_sub_mask]);
        let res2 = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[ni, m_mask]);
        res = ctx.emit_intrinsic(Intrinsic::X86Por, &[res, res2]);
    }
    if !is_tbl {
        let idx_mask =
            x86_get_all_elements_i64(ctx, (0x1010101010101010 * size as i64) - 0x0101010101010101);
        let zero_mask = ctx.vector_zero();
        let m_pos_mask = ctx.emit_intrinsic(Intrinsic::X86Pcmpgtb, &[m, idx_mask]);
        let m_neg_mask = ctx.emit_intrinsic(Intrinsic::X86Pcmpgtb, &[zero_mask, m]);
        let m_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[m_pos_mask, m_neg_mask]);
        let d_mask = ctx.emit_intrinsic(Intrinsic::X86Pand, &[d, m_mask]);
        res = ctx.emit_intrinsic(Intrinsic::X86Por, &[res, d_mask]);
    }
    if reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    ctx.copy_to(d, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitVectorTranspose (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_vector_transpose(ctx: &mut ArmEmitterCtx, part: i32) {
    let (rd, rn, rm, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (rd, rn, rm, size as i32),
        _ => panic!("emit_vector_transpose: expected SimdReg"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let mut n = get_vec(ctx, rn);
    let mut m = get_vec(ctx, rm);
    if size < 3 {
        let mask_e0 = EVEN_MASKS[size as usize];
        let mask_e1 = ODD_MASKS[size as usize];
        let mask = x86_get_scalar_i64(ctx, mask_e0);
        let __c_0 = ctx.ctx.operands.constant_i64(mask_e1);
        let mask = emit_vector_insert(ctx, mask, __c_0, 1, 3);
        n = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n, mask]);
        m = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[m, mask]);
    }
    let punpck_inst = if part == 0 {
        X86_PUNPCKL_INSTRUCTION[size as usize]
    } else {
        X86_PUNPCKH_INSTRUCTION[size as usize]
    };
    let mut res = ctx.emit_intrinsic(punpck_inst, &[n, m]);
    if reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let __gv_0 = get_vec(ctx, rd);
    ctx.copy_to(__gv_0, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitVectorUnzip (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_vector_unzip(ctx: &mut ArmEmitterCtx, part: i32) {
    let (rd, rn, rm, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (rd, rn, rm, size as i32),
        _ => panic!("emit_vector_unzip: expected SimdReg"),
    };
    let reg_size = ctx.curr_op().reg_size;
    if reg_size.bits() == 128 {
        let mut n = get_vec(ctx, rn);
        let mut m = get_vec(ctx, rm);
        if size < 3 {
            let mask_e0 = EVEN_MASKS[size as usize];
            let mask_e1 = ODD_MASKS[size as usize];
            let mask = x86_get_scalar_i64(ctx, mask_e0);
            let __c_0 = ctx.ctx.operands.constant_i64(mask_e1);
            let mask = emit_vector_insert(ctx, mask, __c_0, 1, 3);
            n = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n, mask]);
            m = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[m, mask]);
        }
        let punpck_inst = if part == 0 {
            Intrinsic::X86Punpcklqdq
        } else {
            Intrinsic::X86Punpckhqdq
        };
        let res = ctx.emit_intrinsic(punpck_inst, &[n, m]);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    } else {
        let n = get_vec(ctx, rn);
        let m = get_vec(ctx, rm);
        let punpckl_inst = X86_PUNPCKL_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(punpckl_inst, &[n, m]);
        if size < 2 {
            let mask_e0 = MASKS_E0_UZP[size as usize];
            let mask_e1 = MASKS_E1_UZP[size as usize];
            let mask = x86_get_scalar_i64(ctx, mask_e0);
            let __c_0 = ctx.ctx.operands.constant_i64(mask_e1);
            let mask = emit_vector_insert(ctx, mask, __c_0, 1, 3);
            res = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[res, mask]);
        }
        let punpck_inst = if part == 0 {
            Intrinsic::X86Punpcklqdq
        } else {
            Intrinsic::X86Punpckhqdq
        };
        let __vz_0 = ctx.vector_zero();
        res = ctx.emit_intrinsic(punpck_inst, &[res, __vz_0]);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitVectorZip (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_vector_zip(ctx: &mut ArmEmitterCtx, part: i32) {
    let (rd, rn, rm, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (rd, rn, rm, size as i32),
        _ => panic!("emit_vector_zip: expected SimdReg"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let n = get_vec(ctx, rn);
    let m = get_vec(ctx, rm);
    if reg_size.bits() == 128 {
        let punpck_inst = if part == 0 {
            X86_PUNPCKL_INSTRUCTION[size as usize]
        } else {
            X86_PUNPCKH_INSTRUCTION[size as usize]
        };
        let res = ctx.emit_intrinsic(punpck_inst, &[n, m]);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    } else {
        let mut res = ctx.emit_intrinsic(X86_PUNPCKL_INSTRUCTION[size as usize], &[n, m]);
        let punpck_inst = if part == 0 {
            Intrinsic::X86Punpcklqdq
        } else {
            Intrinsic::X86Punpckhqdq
        };
        let __vz_0 = ctx.vector_zero();
        res = ctx.emit_intrinsic(punpck_inst, &[res, __vz_0]);
        let __gv_0 = get_vec(ctx, rd);
        ctx.copy_to(__gv_0, res);
    }
}
