// Port of InstEmitSimdShift.cs
// Host x86-64 only: AdvSimd paths excluded. SSE paths preferred, IR fallback.

use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::helpers::get_vec;
use crate::instructions::simd_helper::*;
use crate::ir::block::BlockFrequency;
use crate::ir::operand::OperandIndex;
use crate::ir::types::{Comparison, Intrinsic, OperandType};

// Masks (from C# _masks_SliSri)
const MASKS_SLI_SRI: [i64; 4] = [
    0x0101010101010101,
    0x0001000100010001,
    0x0000000100000001,
    0x0000000000000001,
];

// ShrImmFlags (C# private enum)
const SHR_SCALAR: u8 = 1 << 0;
const SHR_SIGNED: u8 = 1 << 1;
const SHR_ROUND: u8 = 1 << 2;
const SHR_ACCUMULATE: u8 = 1 << 3;

const SHR_SCALAR_SX: u8 = SHR_SCALAR | SHR_SIGNED;
const SHR_SCALAR_ZX: u8 = SHR_SCALAR;
const SHR_VECTOR_SX: u8 = SHR_SIGNED;
const SHR_VECTOR_ZX: u8 = 0;

// ShrImmSaturatingNarrowFlags (C# private enum)
const SHRN_SCALAR: u8 = 1 << 0;
const SHRN_SIGNED_SRC: u8 = 1 << 1;
const SHRN_SIGNED_DST: u8 = 1 << 2;
const SHRN_ROUND: u8 = 1 << 3;

const SHRN_SCALAR_SX_SX: u8 = SHRN_SCALAR | SHRN_SIGNED_SRC | SHRN_SIGNED_DST;
const SHRN_SCALAR_SX_ZX: u8 = SHRN_SCALAR | SHRN_SIGNED_SRC;
const SHRN_SCALAR_ZX_ZX: u8 = SHRN_SCALAR;
const SHRN_VECTOR_SX_SX: u8 = SHRN_SIGNED_SRC | SHRN_SIGNED_DST;
const SHRN_VECTOR_SX_ZX: u8 = SHRN_SIGNED_SRC;
const SHRN_VECTOR_ZX_ZX: u8 = 0;

// ShlRegFlags (C# private enum)
const SHL_SCALAR: u8 = 1 << 0;
const SHL_SIGNED: u8 = 1 << 1;
const SHL_ROUND: u8 = 1 << 2;
const SHL_SATURATING: u8 = 1 << 3;

// ══════════════════════════════════════════════════════════════════════════════
// Rshrn_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_rshrn_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_rshrn_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    let round_const = ctx.ctx.operands.constant_i64(1i64 << (shift - 1));
    let elems = 8 >> size;
    let part = if reg_size.bits() == 128 { elems } else { 0 };
    let d_op = get_vec(ctx, rd);
    let mut res = if part == 0 {
        ctx.vector_zero()
    } else {
        ctx.copy(d_op)
    };
    for index in 0..elems {
        let e = emit_vector_extract_zx(ctx, rn, index, size + 1);
        let e = ctx.add(e, round_const);
        let __c_0 = ctx.ctx.operands.constant_i32(shift);
        let e = ctx.shift_right_ui(e, __c_0);
        res = emit_vector_insert(ctx, res, e, part + index, size);
    }
    ctx.copy_to(d_op, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Shl_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_shl_s(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_shl_s: expected SimdShImm"),
    };
    let shift = get_imm_shl(imm as i32, size);
    let ne = emit_vector_extract_zx(ctx, rn, 0, size);
    let shift_op = ctx.ctx.operands.constant_i32(shift);
    let res = ctx.shift_left(ne, shift_op);
    let __v0 = ctx.vector_zero();
    let __vi_1 = emit_vector_insert(ctx, __v0, res, 0, size);
    let __gv_2 = get_vec(ctx, rd);
    ctx.copy_to(__gv_2, __vi_1);
}

// ══════════════════════════════════════════════════════════════════════════════
// Shl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_shl_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_shl_v: expected SimdShImm"),
    };
    let shift = get_imm_shl(imm as i32, size);
    let e_size = 8 << size;
    if shift >= e_size {
        if ctx.curr_op().reg_size.bits() == 64 {
            let rd_op = get_vec(ctx, rd);
            let __v1 = ctx.vector_zero_upper64(rd_op);
            ctx.copy_to(rd_op, __v1);
        }
        return;
    }
    if size > 0 {
        let n = get_vec(ctx, rn);
        let shift_op = ctx.ctx.operands.constant_i32(shift);
        let sll_inst = X86_PSLL_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(sll_inst, &[n, shift_op]);
        if ctx.curr_op().reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_3 = get_vec(ctx, rd);
        ctx.copy_to(__gv_3, res);
    } else {
        // size 0 (byte): no psllb. Shift as words then clear each byte's low
        // `shift` bits (spilled in from the lower neighbour). shift < 8 here
        // (shift >= e_size early-returned above).
        let n = get_vec(ctx, rn);
        let shift_op = ctx.ctx.operands.constant_i32(shift);
        let mut res = ctx.emit_intrinsic(X86_PSLL_INSTRUCTION[1], &[n, shift_op]);
        let mb = ((0xFFu32 << shift) & 0xFF) * 0x01010101;
        let mask = x86_get_all_elements_i32(ctx, mb as i32);
        res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[res, mask]);
        if ctx.curr_op().reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Shll_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_shll_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => (rd, rn, size as i32),
        _ => panic!("emit_shll_v: expected Simd"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = 8 << size;
    // C#: UseSse41  all sizes (0-2) take the SSE path.
    if crate::optimizations::use_sse41() {
        let mut n = get_vec(ctx, rn);
        if reg_size.bits() == 128 {
            let __c_5 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, __c_5]);
        }
        let movsx_inst = X86_PMOVSX_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(movsx_inst, &[n]);
        let sll_inst = X86_PSLL_INSTRUCTION[(size + 1) as usize];
        let __c_6 = ctx.ctx.operands.constant_i32(shift);
        res = ctx.emit_intrinsic(sll_inst, &[res, __c_6]);
        let __gv_7 = get_vec(ctx, rd);
        ctx.copy_to(__gv_7, res);
    } else {
        // C# EmitVectorShImmWidenBinaryZx.
        emit_vector_sh_imm_widen_binary_op(ctx, shift, false);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Shrn_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_shrn_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_shrn_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    let d = get_vec(ctx, rd);
    let d_low = ctx.vector_zero_upper64(d);
    let n = get_vec(ctx, rn);
    let srl_inst = X86_PSRL_INSTRUCTION[(size + 1) as usize];
    let __c_8 = ctx.ctx.operands.constant_i32(shift);
    let n_shifted = ctx.emit_intrinsic(srl_inst, &[n, __c_8]);
    let mask = x86_get_all_elements_i64(ctx, EVEN_MASKS[size as usize]);
    let mut res = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n_shifted, mask]);
    let mov_inst = if reg_size.bits() == 128 {
        Intrinsic::X86Movlhps
    } else {
        Intrinsic::X86Movhlps
    };
    res = ctx.emit_intrinsic(mov_inst, &[d_low, res]);
    ctx.copy_to(d, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sli_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sli_s(ctx: &mut ArmEmitterCtx) {
    emit_sli(ctx, true);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sli_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sli_v(ctx: &mut ArmEmitterCtx) {
    emit_sli(ctx, false);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqrshl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqrshl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_SIGNED | SHL_ROUND | SHL_SATURATING);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqrshrn_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqrshrn_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_SCALAR_SX_SX | SHRN_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqrshrn_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqrshrn_v(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_VECTOR_SX_SX | SHRN_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqrshrun_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqrshrun_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_SCALAR_SX_ZX | SHRN_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqrshrun_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqrshrun_v(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_VECTOR_SX_ZX | SHRN_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqshl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqshl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_SIGNED | SHL_SATURATING);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqshl_Si
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqshl_si(ctx: &mut ArmEmitterCtx) {
    emit_shl_imm_op(ctx, true, SHL_SIGNED | SHL_SCALAR | SHL_SATURATING);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqshl_Vi
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqshl_vi(ctx: &mut ArmEmitterCtx) {
    emit_shl_imm_op(ctx, true, SHL_SIGNED | SHL_SATURATING);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqshrn_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqshrn_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_SCALAR_SX_SX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqshrn_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqshrn_v(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_VECTOR_SX_SX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqshrun_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqshrun_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_SCALAR_SX_ZX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sqshrun_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sqshrun_v(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_VECTOR_SX_ZX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sri_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sri_s(ctx: &mut ArmEmitterCtx) {
    emit_sri(ctx, true);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sri_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sri_v(ctx: &mut ArmEmitterCtx) {
    emit_sri(ctx, false);
}

// ══════════════════════════════════════════════════════════════════════════════
// Srshl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_srshl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_SIGNED | SHL_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Srshr_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_srshr_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_SX | SHR_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Srshr_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_srshr_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_srshr_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    let e_size = 8 << size;
    if size > 0 && size < 3 {
        let n = get_vec(ctx, rn);
        let sll_inst = X86_PSLL_INSTRUCTION[size as usize];
        let __c_9 = ctx.ctx.operands.constant_i32(e_size - shift);
        let res_sll = ctx.emit_intrinsic(sll_inst, &[n, __c_9]);
        let srl_inst = X86_PSRL_INSTRUCTION[size as usize];
        let __c_10 = ctx.ctx.operands.constant_i32(e_size - 1);
        let res_round = ctx.emit_intrinsic(srl_inst, &[res_sll, __c_10]);
        let sra_inst = X86_PSRA_INSTRUCTION[size as usize];
        let __c_11 = ctx.ctx.operands.constant_i32(shift);
        let n_sra = ctx.emit_intrinsic(sra_inst, &[n, __c_11]);
        let add_inst = X86_PADD_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(add_inst, &[res_round, n_sra]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_12 = get_vec(ctx, rd);
        ctx.copy_to(__gv_12, res);
    } else {
        emit_shr_imm_op(ctx, SHR_VECTOR_SX | SHR_ROUND);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Srsra_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_srsra_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_SX | SHR_ROUND | SHR_ACCUMULATE);
}

// ══════════════════════════════════════════════════════════════════════════════
// Srsra_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_srsra_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_srsra_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    let e_size = 8 << size;
    if size > 0 && size < 3 {
        let d = get_vec(ctx, rd);
        let n = get_vec(ctx, rn);
        let sll_inst = X86_PSLL_INSTRUCTION[size as usize];
        let __c_13 = ctx.ctx.operands.constant_i32(e_size - shift);
        let res_sll = ctx.emit_intrinsic(sll_inst, &[n, __c_13]);
        let srl_inst = X86_PSRL_INSTRUCTION[size as usize];
        let __c_14 = ctx.ctx.operands.constant_i32(e_size - 1);
        let res_round = ctx.emit_intrinsic(srl_inst, &[res_sll, __c_14]);
        let sra_inst = X86_PSRA_INSTRUCTION[size as usize];
        let __c_15 = ctx.ctx.operands.constant_i32(shift);
        let n_sra = ctx.emit_intrinsic(sra_inst, &[n, __c_15]);
        let add_inst = X86_PADD_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(add_inst, &[res_round, n_sra]);
        res = ctx.emit_intrinsic(add_inst, &[res, d]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        ctx.copy_to(d, res);
    } else {
        emit_shr_imm_op(ctx, SHR_VECTOR_SX | SHR_ROUND | SHR_ACCUMULATE);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Sshl_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sshl_s(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_SCALAR | SHL_SIGNED);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sshl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sshl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_SIGNED);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sshll_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sshll_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_sshll_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shl(imm as i32, size);
    // C#: UseSse41  all sizes (0-2) take the SSE path.
    if crate::optimizations::use_sse41() {
        let mut n = get_vec(ctx, rn);
        if reg_size.bits() == 128 {
            let __c_16 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, __c_16]);
        }
        let movsx_inst = X86_PMOVSX_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(movsx_inst, &[n]);
        if shift != 0 {
            let sll_inst = X86_PSLL_INSTRUCTION[(size + 1) as usize];
            let __c_17 = ctx.ctx.operands.constant_i32(shift);
            res = ctx.emit_intrinsic(sll_inst, &[res, __c_17]);
        }
        let __gv_18 = get_vec(ctx, rd);
        ctx.copy_to(__gv_18, res);
    } else {
        emit_vector_sh_imm_widen_binary_op(ctx, shift, true);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Sshr_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sshr_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_SX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Sshr_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_sshr_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_sshr_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    if size > 0 && size < 3 {
        let n = get_vec(ctx, rn);
        let sra_inst = X86_PSRA_INSTRUCTION[size as usize];
        let __c_19 = ctx.ctx.operands.constant_i32(shift);
        let mut res = ctx.emit_intrinsic(sra_inst, &[n, __c_19]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_20 = get_vec(ctx, rd);
        ctx.copy_to(__gv_20, res);
    } else {
        emit_shr_imm_op(ctx, SHR_VECTOR_SX);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Ssra_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ssra_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_SX | SHR_ACCUMULATE);
}

// ══════════════════════════════════════════════════════════════════════════════
// Ssra_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ssra_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_ssra_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    if size > 0 && size < 3 {
        let d = get_vec(ctx, rd);
        let n = get_vec(ctx, rn);
        let sra_inst = X86_PSRA_INSTRUCTION[size as usize];
        let __c_21 = ctx.ctx.operands.constant_i32(shift);
        let n_sra = ctx.emit_intrinsic(sra_inst, &[n, __c_21]);
        let add_inst = X86_PADD_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(add_inst, &[n_sra, d]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        ctx.copy_to(d, res);
    } else {
        emit_shr_imm_op(ctx, SHR_VECTOR_SX | SHR_ACCUMULATE);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Uqrshl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uqrshl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_ROUND | SHL_SATURATING);
}

// ══════════════════════════════════════════════════════════════════════════════
// Uqrshrn_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uqrshrn_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_SCALAR_ZX_ZX | SHRN_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Uqrshrn_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uqrshrn_v(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_VECTOR_ZX_ZX | SHRN_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Uqshl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uqshl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_SATURATING);
}

// ══════════════════════════════════════════════════════════════════════════════
// Uqshrn_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uqshrn_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_SCALAR_ZX_ZX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Uqshrn_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_uqshrn_v(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_saturating_narrow_op(ctx, SHRN_VECTOR_ZX_ZX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Urshl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_urshl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Urshr_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_urshr_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_ZX | SHR_ROUND);
}

// ══════════════════════════════════════════════════════════════════════════════
// Urshr_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_urshr_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_urshr_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    let e_size = 8 << size;
    if size > 0 {
        let n = get_vec(ctx, rn);
        let sll_inst = X86_PSLL_INSTRUCTION[size as usize];
        let __c_22 = ctx.ctx.operands.constant_i32(e_size - shift);
        let res_sll = ctx.emit_intrinsic(sll_inst, &[n, __c_22]);
        let srl_inst = X86_PSRL_INSTRUCTION[size as usize];
        let __c_23 = ctx.ctx.operands.constant_i32(e_size - 1);
        let res_round = ctx.emit_intrinsic(srl_inst, &[res_sll, __c_23]);
        let __c_24 = ctx.ctx.operands.constant_i32(shift);
        let n_srl = ctx.emit_intrinsic(srl_inst, &[n, __c_24]);
        let add_inst = X86_PADD_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(add_inst, &[res_round, n_srl]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_25 = get_vec(ctx, rd);
        ctx.copy_to(__gv_25, res);
    } else {
        emit_shr_imm_op(ctx, SHR_VECTOR_ZX | SHR_ROUND);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Ursra_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ursra_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_ZX | SHR_ROUND | SHR_ACCUMULATE);
}

// ══════════════════════════════════════════════════════════════════════════════
// Ursra_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ursra_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_ursra_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    let e_size = 8 << size;
    if size > 0 {
        let d = get_vec(ctx, rd);
        let n = get_vec(ctx, rn);
        let sll_inst = X86_PSLL_INSTRUCTION[size as usize];
        let __c_26 = ctx.ctx.operands.constant_i32(e_size - shift);
        let res_sll = ctx.emit_intrinsic(sll_inst, &[n, __c_26]);
        let srl_inst = X86_PSRL_INSTRUCTION[size as usize];
        let __c_27 = ctx.ctx.operands.constant_i32(e_size - 1);
        let res_round = ctx.emit_intrinsic(srl_inst, &[res_sll, __c_27]);
        let __c_28 = ctx.ctx.operands.constant_i32(shift);
        let n_srl = ctx.emit_intrinsic(srl_inst, &[n, __c_28]);
        let add_inst = X86_PADD_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(add_inst, &[res_round, n_srl]);
        res = ctx.emit_intrinsic(add_inst, &[res, d]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        ctx.copy_to(d, res);
    } else {
        emit_shr_imm_op(ctx, SHR_VECTOR_ZX | SHR_ROUND | SHR_ACCUMULATE);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Ushl_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ushl_s(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, SHL_SCALAR);
}

// ══════════════════════════════════════════════════════════════════════════════
// Ushl_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ushl_v(ctx: &mut ArmEmitterCtx) {
    emit_shl_reg_op(ctx, 0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Ushll_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ushll_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_ushll_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shl(imm as i32, size);
    // C#: UseSse41  all sizes (0-2) take the SSE path.
    if crate::optimizations::use_sse41() {
        let mut n = get_vec(ctx, rn);
        if reg_size.bits() == 128 {
            let __c_29 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, __c_29]);
        }
        let movzx_inst = X86_PMOVZX_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(movzx_inst, &[n]);
        if shift != 0 {
            let sll_inst = X86_PSLL_INSTRUCTION[(size + 1) as usize];
            let __c_30 = ctx.ctx.operands.constant_i32(shift);
            res = ctx.emit_intrinsic(sll_inst, &[res, __c_30]);
        }
        let __gv_31 = get_vec(ctx, rd);
        ctx.copy_to(__gv_31, res);
    } else {
        emit_vector_sh_imm_widen_binary_op(ctx, shift, false);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Ushr_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ushr_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_ZX);
}

// ══════════════════════════════════════════════════════════════════════════════
// Ushr_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_ushr_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_ushr_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    if size > 0 {
        let n = get_vec(ctx, rn);
        let srl_inst = X86_PSRL_INSTRUCTION[size as usize];
        let __c_32 = ctx.ctx.operands.constant_i32(shift);
        let mut res = ctx.emit_intrinsic(srl_inst, &[n, __c_32]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let __gv_33 = get_vec(ctx, rd);
        ctx.copy_to(__gv_33, res);
    } else {
        // size 0 (byte): no psrlb. Shift as words then clear each byte's high
        // `shift` bits (spilled in from the higher neighbour). shift == 8 gives
        // mask 0 -> all zero, which is the correct ushr-by-element-size result.
        let n = get_vec(ctx, rn);
        let shift_op = ctx.ctx.operands.constant_i32(shift);
        let mut res = ctx.emit_intrinsic(X86_PSRL_INSTRUCTION[1], &[n, shift_op]);
        let mb = (0xFFu32 >> shift) * 0x01010101;
        let mask = x86_get_all_elements_i32(ctx, mb as i32);
        res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[res, mask]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Usra_S
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_usra_s(ctx: &mut ArmEmitterCtx) {
    emit_shr_imm_op(ctx, SHR_SCALAR_ZX | SHR_ACCUMULATE);
}

// ══════════════════════════════════════════════════════════════════════════════
// Usra_V
// ══════════════════════════════════════════════════════════════════════════════
pub fn emit_usra_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_usra_v: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    if size > 0 {
        let d = get_vec(ctx, rd);
        let n = get_vec(ctx, rn);
        let srl_inst = X86_PSRL_INSTRUCTION[size as usize];
        let __c_34 = ctx.ctx.operands.constant_i32(shift);
        let n_srl = ctx.emit_intrinsic(srl_inst, &[n, __c_34]);
        let add_inst = X86_PADD_INSTRUCTION[size as usize];
        let mut res = ctx.emit_intrinsic(add_inst, &[n_srl, d]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        ctx.copy_to(d, res);
    } else {
        emit_shr_imm_op(ctx, SHR_VECTOR_ZX | SHR_ACCUMULATE);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitShrImmOp (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_shr_imm_op(ctx: &mut ArmEmitterCtx, flags: u8) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_shr_imm_op: expected SimdShImm"),
    };
    let scalar = (flags & SHR_SCALAR) != 0;
    let signed = (flags & SHR_SIGNED) != 0;
    let round = (flags & SHR_ROUND) != 0;
    let accumulate = (flags & SHR_ACCUMULATE) != 0;
    let shift = get_imm_shr(imm as i32, size);
    let round_const = ctx.ctx.operands.constant_i64(1i64 << (shift - 1));
    let reg_size = ctx.curr_op().reg_size;
    let elems = if scalar {
        1
    } else {
        bytes_count(reg_size) >> size
    };
    let shift_op = ctx.ctx.operands.constant_i32(shift);
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let e = emit_vector_extract(ctx, rn, index, size, signed);
        // C#: the round-const add happens only in the size <= 2 branch;
        // EmitShrImm64 takes the raw value and adds the round const itself.
        let e = if size <= 2 {
            let e = if round { ctx.add(e, round_const) } else { e };
            if signed {
                ctx.shift_right_si(e, shift_op)
            } else {
                ctx.shift_right_ui(e, shift_op)
            }
        } else {
            emit_shr_imm64(
                ctx,
                e,
                signed,
                if round { 1i64 << (shift - 1) } else { 0 },
                shift,
            )
        };
        let e = if accumulate {
            let de = emit_vector_extract(ctx, rd, index, size, signed);
            ctx.add(e, de)
        } else {
            e
        };
        res = emit_vector_insert(ctx, res, e, index, size);
    }
    let __gv_35 = get_vec(ctx, rd);
    ctx.copy_to(__gv_35, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitShrImm64 (C#  context.Call(SoftFallback.SignedShrImm64/...))
// ══════════════════════════════════════════════════════════════════════════════
fn emit_shr_imm64(
    ctx: &mut ArmEmitterCtx,
    value: OperandIndex,
    signed: bool,
    round_const: i64,
    shift: i32,
) -> OperandIndex {
    let __c_36 = ctx.ctx.operands.constant_i64(round_const);
    let __c_37 = ctx.ctx.operands.constant_i32(shift);
    let func = if signed {
        crate::instructions::native_function::NativeFn::SignedShrImm64
    } else {
        crate::instructions::native_function::NativeFn::UnsignedShrImm64
    };
    ctx.call(func, &[value, __c_36, __c_37])
        .expect("ShrImm64 helpers return I64")
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitShrImmSaturatingNarrowOp (C# EmitShrImmSaturatingNarrowOp + Round)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_shr_imm_saturating_narrow_op(ctx: &mut ArmEmitterCtx, flags: u8) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_shr_imm_saturating_narrow_op: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let scalar = (flags & SHRN_SCALAR) != 0;
    let signed_src = (flags & SHRN_SIGNED_SRC) != 0;
    let signed_dst = (flags & SHRN_SIGNED_DST) != 0;
    let round = (flags & SHRN_ROUND) != 0;
    let shift = get_imm_shr(imm as i32, size);
    let round_const = ctx.ctx.operands.constant_i64(1i64 << (shift - 1));
    let elems = if scalar { 1 } else { 8 >> size };
    let part = if !scalar && reg_size.bits() == 128 {
        elems
    } else {
        0
    };
    let d_op = get_vec(ctx, rd);
    let shift_op = ctx.ctx.operands.constant_i32(shift);
    let mut res = if part == 0 {
        ctx.vector_zero()
    } else {
        ctx.copy(d_op)
    };
    for index in 0..elems {
        // C#: source elements are double-width (op.Size + 1).
        let e = emit_vector_extract(ctx, rn, index, size + 1, signed_src);
        let e = if size <= 1 || !round {
            let e = if round { ctx.add(e, round_const) } else { e };
            if signed_src {
                ctx.shift_right_si(e, shift_op)
            } else {
                ctx.shift_right_ui(e, shift_op)
            }
        } else {
            emit_shr_imm64(
                ctx,
                e,
                signed_src,
                if round { 1i64 << (shift - 1) } else { 0 },
                shift,
            )
        };
        let e = if signed_src {
            emit_signed_src_sat_q(ctx, e, size, signed_dst)
        } else {
            emit_unsigned_src_sat_q(ctx, e, size, signed_dst)
        };
        res = emit_vector_insert(ctx, res, e, part + index, size);
    }
    ctx.copy_to(d_op, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitVectorShImmWidenBinaryOp (C#)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_vector_sh_imm_widen_binary_op(ctx: &mut ArmEmitterCtx, imm: i32, signed: bool) {
    let (rd, rn, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::Simd { rd, rn, size, .. } => (rd, rn, size as i32),
        _ => panic!("emit_vector_sh_imm_widen_binary_op: expected Simd"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let mut res = ctx.vector_zero();
    let elems = 8 >> size;
    let part = if reg_size.bits() == 128 { elems } else { 0 };
    let imm_op = ctx.ctx.operands.constant_i32(imm);
    for index in 0..elems {
        let ne = emit_vector_extract(ctx, rn, part + index, size, signed);
        let __v3 = ctx.shift_left(ne, imm_op);
        res = emit_vector_insert(ctx, res, __v3, index, size + 1);
    }
    let __gv_38 = get_vec(ctx, rd);
    ctx.copy_to(__gv_38, res);
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitSli (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_sli(ctx: &mut ArmEmitterCtx, scalar: bool) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_sli: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shl(imm as i32, size);
    let e_size = 8 << size;
    let mask_val: u64 = if shift != 0 {
        u64::MAX >> (64 - shift)
    } else {
        0
    };
    if shift >= e_size {
        if reg_size.bits() == 64 || scalar {
            let __gv_39 = get_vec(ctx, rd);
            let __gv_40 = get_vec(ctx, rd);
            let __v4 = ctx.vector_zero_upper64(__gv_40);
            ctx.copy_to(__gv_39, __v4);
        }
        return;
    }
    if size > 0 {
        let d = get_vec(ctx, rd);
        let n = get_vec(ctx, rn);
        let sll_inst = X86_PSLL_INSTRUCTION[size as usize];
        let __c_41 = ctx.ctx.operands.constant_i32(shift);
        let n_shifted = ctx.emit_intrinsic(sll_inst, &[n, __c_41]);
        let d_mask = x86_get_all_elements_i64(
            ctx,
            (mask_val as i64).wrapping_mul(MASKS_SLI_SRI[size as usize]),
        );
        let d_masked = ctx.emit_intrinsic(Intrinsic::X86Pand, &[d, d_mask]);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Por, &[n_shifted, d_masked]);
        if reg_size.bits() == 64 || scalar {
            res = ctx.vector_zero_upper64(res);
        }
        ctx.copy_to(d, res);
    } else {
        let elems = if scalar {
            1
        } else {
            bytes_count(reg_size) >> size
        };
        let mut res = ctx.vector_zero();
        for index in 0..elems {
            let ne = emit_vector_extract_zx(ctx, rn, index, size);
            let __c_42 = ctx.ctx.operands.constant_i32(shift);
            let ne_shifted = ctx.shift_left(ne, __c_42);
            let de = emit_vector_extract_zx(ctx, rd, index, size);
            let __c_43 = ctx.ctx.operands.constant_i64(mask_val as i64);
            let de_masked = ctx.bitwise_and(de, __c_43);
            let e = ctx.bitwise_or(ne_shifted, de_masked);
            res = emit_vector_insert(ctx, res, e, index, size);
        }
        let __gv_44 = get_vec(ctx, rd);
        ctx.copy_to(__gv_44, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitSri (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_sri(ctx: &mut ArmEmitterCtx, scalar: bool) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_sri: expected SimdShImm"),
    };
    let reg_size = ctx.curr_op().reg_size;
    let shift = get_imm_shr(imm as i32, size);
    let e_size = 8 << size;
    let mask_val = (u64::MAX << (e_size - shift)) & (u64::MAX >> (64 - e_size));
    if shift >= e_size {
        if reg_size.bits() == 64 || scalar {
            let __gv_45 = get_vec(ctx, rd);
            let __gv_46 = get_vec(ctx, rd);
            let __v5 = ctx.vector_zero_upper64(__gv_46);
            ctx.copy_to(__gv_45, __v5);
        }
        return;
    }
    if size > 0 {
        let d = get_vec(ctx, rd);
        let n = get_vec(ctx, rn);
        let srl_inst = X86_PSRL_INSTRUCTION[size as usize];
        let __c_47 = ctx.ctx.operands.constant_i32(shift);
        let n_shifted = ctx.emit_intrinsic(srl_inst, &[n, __c_47]);
        let d_mask = x86_get_all_elements_i64(
            ctx,
            (mask_val as i64).wrapping_mul(MASKS_SLI_SRI[size as usize]),
        );
        let d_masked = ctx.emit_intrinsic(Intrinsic::X86Pand, &[d, d_mask]);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Por, &[n_shifted, d_masked]);
        if reg_size.bits() == 64 || scalar {
            res = ctx.vector_zero_upper64(res);
        }
        ctx.copy_to(d, res);
    } else {
        let elems = if scalar {
            1
        } else {
            bytes_count(reg_size) >> size
        };
        let mut res = ctx.vector_zero();
        for index in 0..elems {
            let ne = emit_vector_extract_zx(ctx, rn, index, size);
            let __c_48 = ctx.ctx.operands.constant_i32(shift);
            let ne_shifted = if shift != 64 {
                ctx.shift_right_ui(ne, __c_48)
            } else {
                ctx.ctx.operands.constant_i64(0)
            };
            let de = emit_vector_extract_zx(ctx, rd, index, size);
            let __c_49 = ctx.ctx.operands.constant_i64(mask_val as i64);
            let de_masked = ctx.bitwise_and(de, __c_49);
            let e = ctx.bitwise_or(ne_shifted, de_masked);
            res = emit_vector_insert(ctx, res, e, index, size);
        }
        let __gv_50 = get_vec(ctx, rd);
        ctx.copy_to(__gv_50, res);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitShlImmOp (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_shl_imm_op(ctx: &mut ArmEmitterCtx, signed_dst: bool, flags: u8) {
    let (rd, rn, size, imm) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdShImm {
            rd, rn, size, imm, ..
        } => (rd, rn, size as i32, imm),
        _ => panic!("emit_shl_imm_op: expected SimdShImm"),
    };
    let scalar = (flags & SHL_SCALAR) != 0;
    let signed = (flags & SHL_SIGNED) != 0;
    let saturating = (flags & SHL_SATURATING) != 0;
    let shift = get_imm_shl(imm as i32, size);
    let reg_size = ctx.curr_op().reg_size;
    let elems = if scalar {
        1
    } else {
        bytes_count(reg_size) >> size
    };
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract(ctx, rn, index, size, signed);
        let e = if saturating {
            emit_shl_imm_sat_q(ctx, ne, shift, size, signed, signed_dst)
        } else {
            emit_shl_imm(ctx, ne, shift, size)
        };
        res = emit_vector_insert(ctx, res, e, index, size);
    }
    let __gv_51 = get_vec(ctx, rd);
    ctx.copy_to(__gv_51, res);
}

// C# EmitShlImm  branch sense kept exactly as upstream (only reachable
// through the non-saturating EmitShlImmOp path, which no A64 caller uses).
fn emit_shl_imm(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    shift_lsb: i32,
    size: i32,
) -> OperandIndex {
    let e_size = 8 << size;
    let res = ctx.allocate_local(OperandType::I64);
    if shift_lsb >= e_size {
        let __c_53 = ctx.ctx.operands.constant_i32(shift_lsb);
        let __v6 = ctx.shift_left(op, __c_53);
        ctx.copy_to(res, __v6);
    } else {
        let __c_52 = ctx.ctx.operands.constant_i64(0);
        ctx.copy_to(res, __c_52);
    }
    res
}

fn emit_shl_imm_sat_q(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    shift_lsb: i32,
    size: i32,
    signed_src: bool,
    signed_dst: bool,
) -> OperandIndex {
    let e_size = 8 << size;
    let lbl_end = ctx.ctx.operands.label();
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, op);
    if shift_lsb >= e_size {
        let sat = if signed_src {
            emit_signed_sign_sat_q(ctx, op, size)
        } else {
            emit_unsigned_sign_sat_q(ctx, op, size)
        };
        ctx.copy_to(res, sat);
    } else {
        let __c_54 = ctx.ctx.operands.constant_i32(shift_lsb);
        let shl = ctx.shift_left(op, __c_54);
        if e_size == 64 {
            let sar_or_shr = if signed_src {
                let __c_55 = ctx.ctx.operands.constant_i32(shift_lsb);
                ctx.shift_right_si(shl, __c_55)
            } else {
                let __c_56 = ctx.ctx.operands.constant_i32(shift_lsb);
                ctx.shift_right_ui(shl, __c_56)
            };
            ctx.copy_to(res, shl);
            ctx.branch_if(
                lbl_end,
                sar_or_shr,
                op,
                Comparison::Equal,
                BlockFrequency::Default,
            );
            let sat = if signed_src {
                emit_signed_sign_sat_q(ctx, op, size)
            } else {
                emit_unsigned_sign_sat_q(ctx, op, size)
            };
            ctx.copy_to(res, sat);
        } else {
            let sat = if signed_src {
                emit_signed_src_sat_q(ctx, shl, size, signed_dst)
            } else {
                emit_unsigned_src_sat_q(ctx, shl, size, signed_dst)
            };
            ctx.copy_to(res, sat);
        }
    }
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// ══════════════════════════════════════════════════════════════════════════════
// Private: EmitShlRegOp (C# private static)
// ══════════════════════════════════════════════════════════════════════════════
fn emit_shl_reg_op(ctx: &mut ArmEmitterCtx, flags: u8) {
    let scalar = (flags & SHL_SCALAR) != 0;
    let signed = (flags & SHL_SIGNED) != 0;
    let round = (flags & SHL_ROUND) != 0;
    let saturating = (flags & SHL_SATURATING) != 0;
    let (rd, rn, rm, size) = match ctx.curr_op().kind {
        crate::decoder::OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (rd, rn, rm, size as i32),
        _ => panic!("emit_shl_reg_op: expected SimdReg"),
    };
    let reg_size = ctx.curr_op().reg_size;
    // AVX2 dword (4S/2S) fast path for plain Ushl/Sshl (not rounding/saturating).
    // Per ARM, each lane shifts by the sign-extended low byte of the Rm element:
    // amount >= 0 -> left (>= esize gives 0), amount < 0 -> right by -amount
    // (logical for U, arithmetic for S; >= esize gives 0 / sign). VPSLLVD/VPSRLVD/
    // VPSRAVD clamp counts >= 32 to exactly 0 / sign, so this is correct branchless.
    if !scalar && !round && !saturating && size == 2 && crate::optimizations::use_avx2() {
        let n = get_vec(ctx, rn);
        let m = get_vec(ctx, rm);
        // amount[lane] = sign-extend(low byte of m[lane]) via pslld 24 + psrad 24.
        let c24 = ctx.ctx.operands.constant_i32(24);
        let amt = ctx.emit_intrinsic(X86_PSLL_INSTRUCTION[2], &[m, c24]);
        let amt = ctx.emit_intrinsic(X86_PSRA_INSTRUCTION[2], &[amt, c24]);
        let left = ctx.emit_intrinsic(Intrinsic::X86Psllvd, &[n, amt]);
        let zero = ctx.vector_zero();
        let neg = ctx.emit_intrinsic(X86_PSUB_INSTRUCTION[2], &[zero, amt]);
        let right = if signed {
            ctx.emit_intrinsic(Intrinsic::X86Psravd, &[n, neg])
        } else {
            ctx.emit_intrinsic(Intrinsic::X86Psrlvd, &[n, neg])
        };
        // selmask[lane] = all-ones where amount < 0 (psrad by 31), else 0.
        let c31 = ctx.ctx.operands.constant_i32(31);
        let selmask = ctx.emit_intrinsic(X86_PSRA_INSTRUCTION[2], &[amt, c31]);
        let r_sel = ctx.emit_intrinsic(Intrinsic::X86Pand, &[selmask, right]);
        let l_sel = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[selmask, left]); // ~selmask & left
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Por, &[r_sel, l_sel]);
        if reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
        return;
    }
    let elems = if scalar {
        1
    } else {
        bytes_count(reg_size) >> size
    };
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract(ctx, rn, index, size, signed);
        let me = emit_vector_extract_sx(ctx, rm, index << size, 0);
        let me_i32 = ctx.convert_i64_to_i32(me);
        let e = if saturating {
            emit_shl_reg_sat_q(ctx, ne, me_i32, round, size, signed)
        } else {
            emit_shl_reg_inner(ctx, ne, me_i32, round, size, signed)
        };
        res = emit_vector_insert(ctx, res, e, index, size);
    }
    let __gv_57 = get_vec(ctx, rd);
    ctx.copy_to(__gv_57, res);
}

fn emit_shl_reg_inner(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    shift_lsb: OperandIndex,
    round: bool,
    size: i32,
    signed: bool,
) -> OperandIndex {
    let e_size = 8 << size;
    let lbl1 = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();
    let e_size_op = ctx.ctx.operands.constant_i32(e_size);
    let zero = ctx.ctx.operands.constant_i32(0);
    let zero_l = ctx.ctx.operands.constant_i64(0);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, op);
    ctx.branch_if(
        lbl1,
        shift_lsb,
        zero,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    let neg_shift = ctx.subtract(zero, shift_lsb);
    let shr_val = if signed {
        emit_signed_shr_reg(ctx, op, neg_shift, round, e_size)
    } else {
        emit_unsigned_shr_reg(ctx, op, neg_shift, round, e_size)
    };
    ctx.copy_to(res, shr_val);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl1, BlockFrequency::Default);
    ctx.branch_if(
        lbl_end,
        shift_lsb,
        zero,
        Comparison::LessOrEqual,
        BlockFrequency::Default,
    );
    let shl = ctx.shift_left(op, shift_lsb);
    let is_ge = ctx.icompare_greater_or_equal(shift_lsb, e_size_op);
    let __v7 = ctx.conditional_select(is_ge, zero_l, shl);
    ctx.copy_to(res, __v7);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

fn emit_shl_reg_sat_q(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    shift_lsb: OperandIndex,
    round: bool,
    size: i32,
    signed: bool,
) -> OperandIndex {
    let e_size = 8 << size;
    let lbl1 = ctx.ctx.operands.label();
    let lbl2 = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();
    let e_size_op = ctx.ctx.operands.constant_i32(e_size);
    let zero = ctx.ctx.operands.constant_i32(0);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, op);
    ctx.branch_if(
        lbl1,
        shift_lsb,
        zero,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    let neg_shift = ctx.subtract(zero, shift_lsb);
    let shr_val = if signed {
        emit_signed_shr_reg(ctx, op, neg_shift, round, e_size)
    } else {
        emit_unsigned_shr_reg(ctx, op, neg_shift, round, e_size)
    };
    ctx.copy_to(res, shr_val);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl1, BlockFrequency::Default);
    ctx.branch_if(
        lbl_end,
        shift_lsb,
        zero,
        Comparison::LessOrEqual,
        BlockFrequency::Default,
    );
    ctx.branch_if(
        lbl2,
        shift_lsb,
        e_size_op,
        Comparison::Less,
        BlockFrequency::Default,
    );
    let sat = if signed {
        emit_signed_sign_sat_q(ctx, op, size)
    } else {
        emit_unsigned_sign_sat_q(ctx, op, size)
    };
    ctx.copy_to(res, sat);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl2, BlockFrequency::Default);
    let shl = ctx.shift_left(op, shift_lsb);
    if e_size == 64 {
        let sar_or_shr = if signed {
            ctx.shift_right_si(shl, shift_lsb)
        } else {
            ctx.shift_right_ui(shl, shift_lsb)
        };
        ctx.copy_to(res, shl);
        ctx.branch_if(
            lbl_end,
            sar_or_shr,
            op,
            Comparison::Equal,
            BlockFrequency::Default,
        );
        let sat = if signed {
            emit_signed_sign_sat_q(ctx, op, size)
        } else {
            emit_unsigned_sign_sat_q(ctx, op, size)
        };
        ctx.copy_to(res, sat);
    } else {
        let sat = if signed {
            emit_signed_src_sat_q(ctx, shl, size, true)
        } else {
            emit_unsigned_src_sat_q(ctx, shl, size, false)
        };
        ctx.copy_to(res, sat);
    }
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

fn emit_signed_shr_reg(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    shift: OperandIndex,
    round: bool,
    e_size: i32,
) -> OperandIndex {
    if round {
        let lbl_end = ctx.ctx.operands.label();
        let e_size_op = ctx.ctx.operands.constant_i32(e_size);
        let zero_l = ctx.ctx.operands.constant_i64(0);
        let one = ctx.ctx.operands.constant_i32(1);
        let one_l = ctx.ctx.operands.constant_i64(1);
        let res = ctx.allocate_local(OperandType::I64);
        ctx.copy_to(res, zero_l);
        ctx.branch_if(
            lbl_end,
            shift,
            e_size_op,
            Comparison::GreaterOrEqual,
            BlockFrequency::Default,
        );
        let __v8 = ctx.subtract(shift, one);
        let round_const = ctx.shift_left(one_l, __v8);
        let add = ctx.add(op, round_const);
        let sar = ctx.shift_right_si(add, shift);
        if e_size == 64 {
            let shr = ctx.shift_right_ui(add, shift);
            let __v9 = ctx.negate(op);
            let __v10 = ctx.bitwise_xor(op, add);
            let left = ctx.bitwise_and(__v9, __v10);
            let is_less = ctx.icompare_less(left, zero_l);
            let __v11 = ctx.conditional_select(is_less, shr, sar);
            ctx.copy_to(res, __v11);
        } else {
            ctx.copy_to(res, sar);
        }
        ctx.branch(lbl_end);
        ctx.mark_label(lbl_end, BlockFrequency::Default);
        res
    } else {
        let lbl_end = ctx.ctx.operands.label();
        let e_size_op = ctx.ctx.operands.constant_i32(e_size);
        let zero_l = ctx.ctx.operands.constant_i64(0);
        let neg_one_l = ctx.ctx.operands.constant_i64(-1);
        let sar = ctx.shift_right_si(op, shift);
        let res = ctx.allocate_local(OperandType::I64);
        ctx.copy_to(res, sar);
        ctx.branch_if(
            lbl_end,
            shift,
            e_size_op,
            Comparison::Less,
            BlockFrequency::Default,
        );
        let is_less = ctx.icompare_less(op, zero_l);
        let __v12 = ctx.conditional_select(is_less, neg_one_l, zero_l);
        ctx.copy_to(res, __v12);
        ctx.branch(lbl_end);
        ctx.mark_label(lbl_end, BlockFrequency::Default);
        res
    }
}

fn emit_unsigned_shr_reg(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    shift: OperandIndex,
    round: bool,
    e_size: i32,
) -> OperandIndex {
    if round {
        let lbl_end = ctx.ctx.operands.label();
        let zero_ul = ctx.ctx.operands.constant_i64(0);
        let one = ctx.ctx.operands.constant_i32(1);
        let one_ul = ctx.ctx.operands.constant_i64(1);
        let e_size_max_op = ctx.ctx.operands.constant_i32(64);
        let one_shl63_ul = ctx.ctx.operands.constant_i64((1u64 << 63) as i64);
        let res = ctx.allocate_local(OperandType::I64);
        ctx.copy_to(res, zero_ul);
        ctx.branch_if(
            lbl_end,
            shift,
            e_size_max_op,
            Comparison::Greater,
            BlockFrequency::Default,
        );
        let __v13 = ctx.subtract(shift, one);
        let round_const = ctx.shift_left(one_ul, __v13);
        let add = ctx.add(op, round_const);
        let shr = ctx.shift_right_ui(add, shift);
        let is_equal = ctx.icompare_equal(shift, e_size_max_op);
        let __v14 = ctx.conditional_select(is_equal, zero_ul, shr);
        ctx.copy_to(res, __v14);
        if e_size == 64 {
            ctx.branch_if(
                lbl_end,
                add,
                op,
                Comparison::GreaterOrEqualUI,
                BlockFrequency::Default,
            );
            let __v15 = ctx.subtract(shift, one);
            let __v16 = ctx.shift_right_ui(one_shl63_ul, __v15);
            let right = ctx.bitwise_or(shr, __v16);
            let __v17 = ctx.conditional_select(is_equal, one_ul, right);
            ctx.copy_to(res, __v17);
        }
        ctx.branch(lbl_end);
        ctx.mark_label(lbl_end, BlockFrequency::Default);
        res
    } else {
        let lbl_end = ctx.ctx.operands.label();
        let e_size_op = ctx.ctx.operands.constant_i32(e_size);
        let zero_ul = ctx.ctx.operands.constant_i64(0);
        let shr = ctx.shift_right_ui(op, shift);
        let res = ctx.allocate_local(OperandType::I64);
        ctx.copy_to(res, shr);
        ctx.branch_if(
            lbl_end,
            shift,
            e_size_op,
            Comparison::Less,
            BlockFrequency::Default,
        );
        ctx.copy_to(res, zero_ul);
        ctx.branch(lbl_end);
        ctx.mark_label(lbl_end, BlockFrequency::Default);
        res
    }
}
