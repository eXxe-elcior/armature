// Port of InstEmitSimdHelper.cs  reusable SIMD emission patterns.
// Host is x86-64 only, so the Arm64 AdvSimd paths are excluded entirely.
// Each helper is parameterized on the operation (via a closure or intrinsic)
// and handles element-type/size dispatch.

use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::block::BlockFrequency;
use crate::ir::operand::OperandIndex;
use crate::ir::types::{Comparison, Intrinsic, OperandType};
use crate::registers::VecRegisterIndex;

use super::context::ArmEmitterCtx;
use super::helpers::get_vec;

// Masks (from C# EvenMasks / OddMasks / ZeroMask)
pub const EVEN_MASKS: [i64; 3] = [
    // B: 14,12,10,8,6,4,2,0
    (14i64 << 56)
        | (12 << 48)
        | (10 << 40)
        | (8 << 32)
        | (6 << 24)
        | (4 << 16)
        | (2 << 8)
        | (0 << 0),
    // H: 13,12,9,8,5,4,1,0
    (13i64 << 56)
        | (12 << 48)
        | (9 << 40)
        | (8 << 32)
        | (5 << 24)
        | (4 << 16)
        | (1 << 8)
        | (0 << 0),
    // S: 11,10,9,8,3,2,1,0
    (11i64 << 56)
        | (10 << 48)
        | (9 << 40)
        | (8 << 32)
        | (3 << 24)
        | (2 << 16)
        | (1 << 8)
        | (0 << 0),
];

pub const ODD_MASKS: [i64; 3] = [
    // B: 15,13,11,9,7,5,3,1
    (15i64 << 56)
        | (13 << 48)
        | (11 << 40)
        | (9 << 32)
        | (7 << 24)
        | (5 << 16)
        | (3 << 8)
        | (1 << 0),
    // H: 15,14,11,10,7,6,3,2
    (15i64 << 56)
        | (14 << 48)
        | (11 << 40)
        | (10 << 32)
        | (7 << 24)
        | (6 << 16)
        | (3 << 8)
        | (2 << 0),
    // S: 15,14,13,12,7,6,5,4
    (15i64 << 56)
        | (14 << 48)
        | (13 << 40)
        | (12 << 32)
        | (7 << 24)
        | (6 << 16)
        | (5 << 8)
        | (4 << 0),
];

pub const ZERO_MASK: i64 = (128i64 << 56)
    | (128 << 48)
    | (128 << 40)
    | (128 << 32)
    | (128 << 24)
    | (128 << 16)
    | (128 << 8)
    | (128 << 0);

// X86 intrinsic tables (from C# static readonly arrays)
// Index by size: 0=B, 1=H, 2=S, 3=D (where applicable).

pub const X86_PADD_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Paddb,
    Intrinsic::X86Paddw,
    Intrinsic::X86Paddd,
    Intrinsic::X86Paddq,
];

pub const X86_PCMPEQ_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Pcmpeqb,
    Intrinsic::X86Pcmpeqw,
    Intrinsic::X86Pcmpeqd,
    Intrinsic::X86Pcmpeqq,
];

pub const X86_PCMPGT_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Pcmpgtb,
    Intrinsic::X86Pcmpgtw,
    Intrinsic::X86Pcmpgtd,
    Intrinsic::X86Pcmpgtq,
];

pub const X86_PMAXS_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Pmaxsb,
    Intrinsic::X86Pmaxsw,
    Intrinsic::X86Pmaxsd,
    Intrinsic::X86Pmaxsb, // dummy; size=3 (64-bit) path handled separately
];

pub const X86_PMAXU_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Pmaxub,
    Intrinsic::X86Pmaxuw,
    Intrinsic::X86Pmaxud,
    Intrinsic::X86Pmaxub, // dummy
];

pub const X86_PMINS_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Pminsb,
    Intrinsic::X86Pminsw,
    Intrinsic::X86Pminsd,
    Intrinsic::X86Pminsb, // dummy
];

pub const X86_PMINU_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Pminub,
    Intrinsic::X86Pminuw,
    Intrinsic::X86Pminud,
    Intrinsic::X86Pminub, // dummy
];

pub const X86_PMOVSX_INSTRUCTION: [Intrinsic; 3] = [
    Intrinsic::X86Pmovsxbw,
    Intrinsic::X86Pmovsxwd,
    Intrinsic::X86Pmovsxdq,
];

pub const X86_PMOVZX_INSTRUCTION: [Intrinsic; 3] = [
    Intrinsic::X86Pmovzxbw,
    Intrinsic::X86Pmovzxwd,
    Intrinsic::X86Pmovzxdq,
];

// Index 0 is unused (no 8-bit shift intrinsic).
pub const X86_PSLL_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86None,
    Intrinsic::X86Psllw,
    Intrinsic::X86Pslld,
    Intrinsic::X86Psllq,
];

pub const X86_PSRA_INSTRUCTION: [Intrinsic; 3] =
    [Intrinsic::X86None, Intrinsic::X86Psraw, Intrinsic::X86Psrad];

pub const X86_PSRL_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86None,
    Intrinsic::X86Psrlw,
    Intrinsic::X86Psrld,
    Intrinsic::X86Psrlq,
];

pub const X86_PSUB_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Psubb,
    Intrinsic::X86Psubw,
    Intrinsic::X86Psubd,
    Intrinsic::X86Psubq,
];

pub const X86_PUNPCKH_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Punpckhbw,
    Intrinsic::X86Punpckhwd,
    Intrinsic::X86Punpckhdq,
    Intrinsic::X86Punpckhqdq,
];

pub const X86_PUNPCKL_INSTRUCTION: [Intrinsic; 4] = [
    Intrinsic::X86Punpcklbw,
    Intrinsic::X86Punpcklwd,
    Intrinsic::X86Punpckldq,
    Intrinsic::X86Punpcklqdq,
];

// Scalar helpers (work on element 0)
pub fn emit_scalar_unary_op_f_sse(ctx: &mut ArmEmitterCtx, inst32: Intrinsic, inst64: Intrinsic) {
    let op = decode_simd_op(ctx);
    let n = get_vec(ctx, op.rn);
    let inst = if (op.size & 1) != 0 { inst64 } else { inst32 };
    let res = ctx.emit_intrinsic(inst, &[n]);
    let res = if (op.size & 1) != 0 {
        ctx.vector_zero_upper64(res)
    } else {
        ctx.vector_zero_upper96(res)
    };
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_scalar_binary_op_f_sse(ctx: &mut ArmEmitterCtx, inst32: Intrinsic, inst64: Intrinsic) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let inst = if (op.size & 1) != 0 { inst64 } else { inst32 };
    let res = ctx.emit_intrinsic(inst, &[n, m]);
    let res = if (op.size & 1) != 0 {
        ctx.vector_zero_upper64(res)
    } else {
        ctx.vector_zero_upper96(res)
    };
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Vector helpers (work on all elements)
pub fn emit_vector_unary_op_f_sse(ctx: &mut ArmEmitterCtx, inst32: Intrinsic, inst64: Intrinsic) {
    let op = decode_simd_op(ctx);
    let n = get_vec(ctx, op.rn);
    let inst = if (op.size & 1) != 0 { inst64 } else { inst32 };
    let mut res = ctx.emit_intrinsic(inst, &[n]);
    if op.reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_binary_op_f_sse(ctx: &mut ArmEmitterCtx, inst32: Intrinsic, inst64: Intrinsic) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let inst = if (op.size & 1) != 0 { inst64 } else { inst32 };
    let mut res = ctx.emit_intrinsic(inst, &[n, m]);
    if op.reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Callback-based helpers (soft/fallback path)
// These match C# EmitScalarUnaryOpSx/Zx, EmitVectorBinaryOpSx/Zx, etc.

pub fn emit_scalar_unary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    let n = emit_vector_extract_sx(ctx, op.rn, 0, op.size);
    let emitted = emit(ctx, n);
    let zero = ctx.vector_zero();
    let d = emit_vector_insert(ctx, zero, emitted, 0, op.size);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_unary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    let n = emit_vector_extract_zx(ctx, op.rn, 0, op.size);
    let emitted = emit(ctx, n);
    let zero = ctx.vector_zero();
    let d = emit_vector_insert(ctx, zero, emitted, 0, op.size);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_binary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let n = emit_vector_extract_zx(ctx, op.rn, 0, op.size);
    let m = emit_vector_extract_zx(ctx, op.rm, 0, op.size);
    let emitted = emit(ctx, n, m);
    let zero = ctx.vector_zero();
    let d = emit_vector_insert(ctx, zero, emitted, 0, op.size);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_binary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let n = emit_vector_extract_sx(ctx, op.rn, 0, op.size);
    let m = emit_vector_extract_sx(ctx, op.rm, 0, op.size);
    let emitted = emit(ctx, n, m);
    let zero = ctx.vector_zero();
    let d = emit_vector_insert(ctx, zero, emitted, 0, op.size);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_ternary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let d = emit_vector_extract_zx(ctx, op.rd, 0, op.size);
    let n = emit_vector_extract_zx(ctx, op.rn, 0, op.size);
    let m = emit_vector_extract_zx(ctx, op.rm, 0, op.size);
    let emitted = emit(ctx, d, n, m);
    let zero = ctx.vector_zero();
    let d = emit_vector_insert(ctx, zero, emitted, 0, op.size);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_unary_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    let typ = if (op.size & 1) != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let rn = get_vec(ctx, op.rn);
    let n = ctx.vector_extract(typ, rn, 0);
    let emitted = emit(ctx, n);
    let zero = ctx.vector_zero();
    let d = ctx.vector_insert(zero, emitted, 0);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_binary_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let typ = if (op.size & 1) != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let rn = get_vec(ctx, op.rn);
    let rm = get_vec(ctx, op.rm);
    let n = ctx.vector_extract(typ, rn, 0);
    let m = ctx.vector_extract(typ, rm, 0);
    let emitted = emit(ctx, n, m);
    let zero = ctx.vector_zero();
    let d = ctx.vector_insert(zero, emitted, 0);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_ternary_ra_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let typ = if (op.size & 1) != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let ra = get_vec(ctx, op.ra);
    let rn = get_vec(ctx, op.rn);
    let rm = get_vec(ctx, op.rm);
    let a = ctx.vector_extract(typ, ra, 0);
    let n = ctx.vector_extract(typ, rn, 0);
    let m = ctx.vector_extract(typ, rm, 0);
    let emitted = emit(ctx, a, n, m);
    let zero = ctx.vector_zero();
    let d = ctx.vector_insert(zero, emitted, 0);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_vector_unary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_sx(ctx, op.rn, index, op.size);
        let emitted = emit(ctx, ne);
        res = emit_vector_insert(ctx, res, emitted, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_unary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_zx(ctx, op.rn, index, op.size);
        let emitted = emit(ctx, ne);
        res = emit_vector_insert(ctx, res, emitted, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_binary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_zx(ctx, op.rn, index, op.size);
        let me = emit_vector_extract_zx(ctx, op.rm, index, op.size);
        let emitted = emit(ctx, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_binary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_sx(ctx, op.rn, index, op.size);
        let me = emit_vector_extract_sx(ctx, op.rm, index, op.size);
        let emitted = emit(ctx, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_ternary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let de = emit_vector_extract_zx(ctx, op.rd, index, op.size);
        let ne = emit_vector_extract_zx(ctx, op.rn, index, op.size);
        let me = emit_vector_extract_zx(ctx, op.rm, index, op.size);
        let emitted = emit(ctx, de, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_ternary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let de = emit_vector_extract_sx(ctx, op.rd, index, op.size);
        let ne = emit_vector_extract_sx(ctx, op.rn, index, op.size);
        let me = emit_vector_extract_sx(ctx, op.rm, index, op.size);
        let emitted = emit(ctx, de, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_binary_op_by_elem_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_elem_op(ctx);
    let me = emit_vector_extract_zx(ctx, op.rm, op.index as i32, op.size);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_zx(ctx, op.rn, index as i32, op.size);
        let emitted = emit(ctx, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index as i32, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_binary_op_by_elem_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_elem_op(ctx);
    let me = emit_vector_extract_sx(ctx, op.rm, op.index as i32, op.size);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_sx(ctx, op.rn, index as i32, op.size);
        let emitted = emit(ctx, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index as i32, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_ternary_op_by_elem_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_elem_op(ctx);
    let me = emit_vector_extract_zx(ctx, op.rm, op.index as i32, op.size);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let de = emit_vector_extract_zx(ctx, op.rd, index as i32, op.size);
        let ne = emit_vector_extract_zx(ctx, op.rn, index as i32, op.size);
        let emitted = emit(ctx, de, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index as i32, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_unary_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    let size_f = op.size & 1;
    let typ = if size_f != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let elems = bytes_count(op.reg_size) >> (size_f + 2);
    let rn = get_vec(ctx, op.rn);
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = ctx.vector_extract(typ, rn, index as i32);
        let emitted = emit(ctx, ne);
        res = ctx.vector_insert(res, emitted, index as i32);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_binary_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let size_f = op.size & 1;
    let typ = if size_f != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let elems = bytes_count(op.reg_size) >> (size_f + 2);
    let rn = get_vec(ctx, op.rn);
    let rm = get_vec(ctx, op.rm);
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = ctx.vector_extract(typ, rn, index as i32);
        let me = ctx.vector_extract(typ, rm, index as i32);
        let emitted = emit(ctx, ne, me);
        res = ctx.vector_insert(res, emitted, index as i32);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_ternary_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let size_f = op.size & 1;
    let typ = if size_f != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let elems = bytes_count(op.reg_size) >> (size_f + 2);
    let rd_vec = get_vec(ctx, op.rd);
    let rn = get_vec(ctx, op.rn);
    let rm = get_vec(ctx, op.rm);
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let de = ctx.vector_extract(typ, rd_vec, index as i32);
        let ne = ctx.vector_extract(typ, rn, index as i32);
        let me = ctx.vector_extract(typ, rm, index as i32);
        let emitted = emit(ctx, de, ne, me);
        res = ctx.vector_insert(res, emitted, index as i32);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_binary_op_by_elem_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_elem_f_op(ctx);
    let size_f = op.size & 1;
    let typ = if size_f != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let elems = bytes_count(op.reg_size) >> (size_f + 2);
    let rm = get_vec(ctx, op.rm);
    let rn = get_vec(ctx, op.rn);
    let me = ctx.vector_extract(typ, rm, op.index as i32);
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = ctx.vector_extract(typ, rn, index as i32);
        let emitted = emit(ctx, ne, me);
        res = ctx.vector_insert(res, emitted, index as i32);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_ternary_op_by_elem_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_elem_f_op(ctx);
    let size_f = op.size & 1;
    let typ = if size_f != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let elems = bytes_count(op.reg_size) >> (size_f + 2);
    let rm = get_vec(ctx, op.rm);
    let rd_vec = get_vec(ctx, op.rd);
    let rn = get_vec(ctx, op.rn);
    let me = ctx.vector_extract(typ, rm, op.index as i32);
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let de = ctx.vector_extract(typ, rd_vec, index as i32);
        let ne = ctx.vector_extract(typ, rn, index as i32);
        let emitted = emit(ctx, de, ne, me);
        res = ctx.vector_insert(res, emitted, index as i32);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Vector insert/extract helpers (C# EmitVectorExtractZx/Sx/Insert)
pub fn emit_vector_extract_zx(
    ctx: &mut ArmEmitterCtx,
    reg: VecRegisterIndex,
    index: i32,
    size: i32,
) -> OperandIndex {
    let vec = get_vec(ctx, reg);
    let res = match size {
        0 => ctx.vector_extract8(vec, index),
        1 => ctx.vector_extract16(vec, index),
        2 => ctx.vector_extract(OperandType::I32, vec, index),
        _ => ctx.vector_extract(OperandType::I64, vec, index),
    };
    match size {
        0 => ctx.zero_extend8(OperandType::I64, res),
        1 => ctx.zero_extend16(OperandType::I64, res),
        2 => ctx.zero_extend32(OperandType::I64, res),
        _ => res,
    }
}

pub fn emit_vector_extract_sx(
    ctx: &mut ArmEmitterCtx,
    reg: VecRegisterIndex,
    index: i32,
    size: i32,
) -> OperandIndex {
    let vec = get_vec(ctx, reg);
    let res = match size {
        0 => ctx.vector_extract8(vec, index),
        1 => ctx.vector_extract16(vec, index),
        2 => ctx.vector_extract(OperandType::I32, vec, index),
        _ => ctx.vector_extract(OperandType::I64, vec, index),
    };
    match size {
        0 => ctx.sign_extend8(OperandType::I64, res),
        1 => ctx.sign_extend16(OperandType::I64, res),
        2 => ctx.sign_extend32(OperandType::I64, res),
        _ => res,
    }
}

pub fn emit_vector_insert(
    ctx: &mut ArmEmitterCtx,
    vector: OperandIndex,
    mut value: OperandIndex,
    index: i32,
    size: i32,
) -> OperandIndex {
    if size < 3 && ctx.ctx.operands.get(value).typ == OperandType::I64 {
        value = ctx.convert_i64_to_i32(value);
    }
    match size {
        0 => ctx.vector_insert8(vector, value, index),
        1 => ctx.vector_insert16(vector, value, index),
        _ => ctx.vector_insert(vector, value, index),
    }
}

pub fn emit_vector_extract(
    ctx: &mut ArmEmitterCtx,
    reg: VecRegisterIndex,
    index: i32,
    size: i32,
    signed: bool,
) -> OperandIndex {
    if signed {
        emit_vector_extract_sx(ctx, reg, index, size)
    } else {
        emit_vector_extract_zx(ctx, reg, index, size)
    }
}

// X86 scalar/vector constant helpers
pub fn x86_get_scalar_i32(ctx: &mut ArmEmitterCtx, value: i32) -> OperandIndex {
    let c = ctx.ctx.operands.constant_i32(value);
    ctx.vector_create_scalar(c)
}

pub fn x86_get_scalar_i64(ctx: &mut ArmEmitterCtx, value: i64) -> OperandIndex {
    let c = ctx.ctx.operands.constant_i64(value);
    ctx.vector_create_scalar(c)
}

pub fn x86_get_scalar_f32(ctx: &mut ArmEmitterCtx, value: f32) -> OperandIndex {
    x86_get_scalar_i32(ctx, value.to_bits() as i32)
}

pub fn x86_get_scalar_f64(ctx: &mut ArmEmitterCtx, value: f64) -> OperandIndex {
    x86_get_scalar_i64(ctx, value.to_bits() as i64)
}

pub fn x86_get_all_elements_i32(ctx: &mut ArmEmitterCtx, value: i32) -> OperandIndex {
    let c = ctx.ctx.operands.constant_i32(value);
    let v = ctx.vector_create_scalar(c);
    let shuf = ctx.ctx.operands.constant_i32(0);
    ctx.emit_intrinsic(Intrinsic::X86Shufps, &[v, v, shuf])
}

pub fn x86_get_all_elements_i64(ctx: &mut ArmEmitterCtx, value: i64) -> OperandIndex {
    let c = ctx.ctx.operands.constant_i64(value);
    let v = ctx.vector_create_scalar(c);
    ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[v, v])
}

pub fn x86_get_all_elements_f32(ctx: &mut ArmEmitterCtx, value: f32) -> OperandIndex {
    x86_get_all_elements_i32(ctx, value.to_bits() as i32)
}

pub fn x86_get_all_elements_f64(ctx: &mut ArmEmitterCtx, value: f64) -> OperandIndex {
    x86_get_all_elements_i64(ctx, value.to_bits() as i64)
}

pub fn x86_get_all_elements_short(ctx: &mut ArmEmitterCtx, value: i16) -> OperandIndex {
    let v1 = value as u16 as u64;
    let v2 = (v1 << 16) | v1;
    let v4 = (v2 << 32) | v2;
    x86_get_all_elements_i64(ctx, v4 as i64)
}

pub fn x86_get_elements(ctx: &mut ArmEmitterCtx, e1: i64, e0: i64) -> OperandIndex {
    let c0 = ctx.ctx.operands.constant_i64(e0);
    let v0 = ctx.vector_create_scalar(c0);
    let c1 = ctx.ctx.operands.constant_i64(e1);
    let v1 = ctx.vector_create_scalar(c1);
    ctx.emit_intrinsic(Intrinsic::X86Punpcklqdq, &[v0, v1])
}

// Float abs helper (C# EmitFloatAbs)
// Uses X86Andnps/Andnpd (~mask & value) to clear the sign bit.
// mask = -0f (0x80000000) or -0d (0x8000000000000000), so ~mask clears the sign bit.

pub fn emit_float_abs(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    single: bool,
    vector: bool,
) -> OperandIndex {
    let mask = if single {
        if vector {
            x86_get_all_elements_i32(ctx, i32::MIN)
        } else {
            x86_get_scalar_i32(ctx, i32::MIN)
        }
    } else {
        if vector {
            x86_get_all_elements_i64(ctx, i64::MIN)
        } else {
            x86_get_scalar_i64(ctx, i64::MIN)
        }
    };
    let intrin = if single {
        Intrinsic::X86Andnps
    } else {
        Intrinsic::X86Andnpd
    };
    ctx.emit_intrinsic(intrin, &[mask, op])
}

// SoftFloat call helper (C# EmitSoftFloatCall)
// C# EmitSoftFloatCall dispatches to SoftFloat32/SoftFloat64 by op.Size.
// Size = 0 → single (f32), size & 1 ≠ 0 → double (f64).
pub fn emit_soft_float_call(
    ctx: &mut ArmEmitterCtx,
    name: &str,
    args: &[OperandIndex],
) -> OperandIndex {
    use crate::instructions::native_function::NativeFn;

    let kind = &ctx.curr_op().kind;
    let single = opcode_is_single(kind);

    let native_fn = match (name, single) {
        ("FPAdd", true) => NativeFn::SoftF32FPAdd,
        ("FPAdd", false) => NativeFn::SoftF64FPAdd,
        ("FPSub", true) => NativeFn::SoftF32FPSub,
        ("FPSub", false) => NativeFn::SoftF64FPSub,
        ("FPMul", true) => NativeFn::SoftF32FPMul,
        ("FPMul", false) => NativeFn::SoftF64FPMul,
        ("FPDiv", true) => NativeFn::SoftF32FPDiv,
        ("FPDiv", false) => NativeFn::SoftF64FPDiv,
        ("FPSqrt", true) => NativeFn::SoftF32FPSqrt,
        ("FPSqrt", false) => NativeFn::SoftF64FPSqrt,
        ("FPMax", true) => NativeFn::SoftF32FPMax,
        ("FPMax", false) => NativeFn::SoftF64FPMax,
        ("FPMin", true) => NativeFn::SoftF32FPMin,
        ("FPMin", false) => NativeFn::SoftF64FPMin,
        ("FPMulX", true) => NativeFn::SoftF32FPMulX,
        ("FPMulX", false) => NativeFn::SoftF64FPMulX,
        ("FPRecipEstimate", true) => NativeFn::SoftF32FPRecipEstimate,
        ("FPRecipEstimate", false) => NativeFn::SoftF64FPRecipEstimate,
        ("FPRecpX", true) => NativeFn::SoftF32FPRecpX,
        ("FPRecpX", false) => NativeFn::SoftF64FPRecpX,
        ("FPRSqrtEstimate", true) => NativeFn::SoftF32FPRSqrtEstimate,
        ("FPRSqrtEstimate", false) => NativeFn::SoftF64FPRSqrtEstimate,
        ("FPCompare", true) => NativeFn::SoftF32FPCompare,
        ("FPCompare", false) => NativeFn::SoftF64FPCompare,
        ("FPCompareEQ", true) => NativeFn::SoftF32FPCompareEQ,
        ("FPCompareEQ", false) => NativeFn::SoftF64FPCompareEQ,
        ("FPCompareGE", true) => NativeFn::SoftF32FPCompareGE,
        ("FPCompareGE", false) => NativeFn::SoftF64FPCompareGE,
        ("FPCompareGT", true) => NativeFn::SoftF32FPCompareGT,
        ("FPCompareGT", false) => NativeFn::SoftF64FPCompareGT,
        ("FPCompareLE", true) => NativeFn::SoftF32FPCompareLE,
        ("FPCompareLE", false) => NativeFn::SoftF64FPCompareLE,
        ("FPCompareLT", true) => NativeFn::SoftF32FPCompareLT,
        ("FPCompareLT", false) => NativeFn::SoftF64FPCompareLT,
        ("FPRecipStep", true) => NativeFn::SoftF32FPRecipStep,
        ("FPRecipStep", false) => NativeFn::SoftF64FPRecipStep,
        ("FPRSqrtStep", true) => NativeFn::SoftF32FPRSqrtStep,
        ("FPRSqrtStep", false) => NativeFn::SoftF64FPRSqrtStep,
        ("FPRecipStepFused", true) => NativeFn::SoftF32FPRecipStepFused,
        ("FPRecipStepFused", false) => NativeFn::SoftF64FPRecipStepFused,
        ("FPRSqrtStepFused", true) => NativeFn::SoftF32FPRSqrtStepFused,
        ("FPRSqrtStepFused", false) => NativeFn::SoftF64FPRSqrtStepFused,
        // Fused multiply-add family.
        ("FPMulAdd", true) => NativeFn::SoftF32FPMulAdd,
        ("FPMulAdd", false) => NativeFn::SoftF64FPMulAdd,
        ("FPMulSub", true) => NativeFn::SoftF32FPMulSub,
        ("FPMulSub", false) => NativeFn::SoftF64FPMulSub,
        ("FPNegMulAdd", true) => NativeFn::SoftF32FPNegMulAdd,
        ("FPNegMulAdd", false) => NativeFn::SoftF64FPNegMulAdd,
        ("FPNegMulSub", true) => NativeFn::SoftF32FPNegMulSub,
        ("FPNegMulSub", false) => NativeFn::SoftF64FPNegMulSub,
        // Abs helper (used by Facge/Facgt soft fallback).
        ("Abs", true) => NativeFn::SoftF32Abs,
        ("Abs", false) => NativeFn::SoftF64Abs,
        _ => panic!("emit_soft_float_call: unknown {name} (single={single})"),
    };
    // ctx.call returns Option<OperandIdx>. unwrap: all our SoftFloat fns return a value.
    ctx.call(native_fn, args).unwrap()
}

// C# EmitUnaryMathCall  MathHelperF (single) / MathHelper (double) by op.Size.
pub fn emit_unary_math_call(ctx: &mut ArmEmitterCtx, name: &str, n: OperandIndex) -> OperandIndex {
    use crate::instructions::native_function::NativeFn;

    let single = opcode_is_single(&ctx.curr_op().kind);
    let native_fn = match (name, single) {
        ("Abs", true) => NativeFn::SoftF32Abs,
        ("Abs", false) => NativeFn::SoftF64Abs,
        ("Ceiling", true) => NativeFn::MathF32Ceiling,
        ("Ceiling", false) => NativeFn::MathF64Ceiling,
        ("Floor", true) => NativeFn::MathF32Floor,
        ("Floor", false) => NativeFn::MathF64Floor,
        ("Truncate", true) => NativeFn::MathF32Truncate,
        ("Truncate", false) => NativeFn::MathF64Truncate,
        _ => panic!("emit_unary_math_call: unknown {name} (single={single})"),
    };
    ctx.call(native_fn, &[n]).unwrap()
}

// C# EmitRoundMathCall  MathHelper(F).Round(value, (int)MidpointRounding).
// C# MidpointRounding: ToEven = 0, AwayFromZero = 1.
pub fn emit_round_math_call(
    ctx: &mut ArmEmitterCtx,
    round_mode: i32,
    n: OperandIndex,
) -> OperandIndex {
    use crate::instructions::native_function::NativeFn;

    let single = opcode_is_single(&ctx.curr_op().kind);
    let native_fn = if single {
        NativeFn::MathF32Round
    } else {
        NativeFn::MathF64Round
    };
    let mode = ctx.ctx.operands.constant_i32(round_mode);
    ctx.call(native_fn, &[n, mode]).unwrap()
}

// C# EmitGetRoundingMode  rMode = (RMode1 << 1) | RMode0.
pub fn emit_get_rounding_mode(ctx: &mut ArmEmitterCtx) -> OperandIndex {
    use crate::state::FPState;

    let r1 = super::helpers::get_fp_flag(FPState::RMode1Flag, &mut ctx.ctx.operands);
    let one = ctx.ctx.operands.constant_i32(1);
    let r_mode = ctx.shift_left(r1, one);
    let r0 = super::helpers::get_fp_flag(FPState::RMode0Flag, &mut ctx.ctx.operands);
    ctx.bitwise_or(r_mode, r0)
}

// C# EmitRoundByRMode  branch on the live FPCR rounding mode and round
// via the MathHelper calls (Frinti/Frintx).
pub fn emit_round_by_rmode(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    let typ = ctx.ctx.operands.get(op).typ;
    debug_assert!(matches!(typ, OperandType::FP32 | OperandType::FP64));

    let lbl1 = ctx.ctx.operands.label();
    let lbl2 = ctx.ctx.operands.label();
    let lbl3 = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();

    // FPRoundingMode: ToNearest = 0, TowardsPlusInfinity = 1,
    // TowardsMinusInfinity = 2 (TowardsZero = 3 is the fall-through).
    let r_n = ctx.ctx.operands.constant_i32(0);
    let r_p = ctx.ctx.operands.constant_i32(1);
    let r_m = ctx.ctx.operands.constant_i32(2);

    let res = ctx.allocate_local(typ);

    let r_mode = emit_get_rounding_mode(ctx);

    ctx.branch_if(
        lbl1,
        r_mode,
        r_n,
        Comparison::NotEqual,
        BlockFrequency::Default,
    );
    let v = emit_round_math_call(ctx, 0, op); // MidpointRounding.ToEven
    ctx.copy_to(res, v);
    ctx.branch(lbl_end);

    ctx.mark_label(lbl1, BlockFrequency::Default);
    ctx.branch_if(
        lbl2,
        r_mode,
        r_p,
        Comparison::NotEqual,
        BlockFrequency::Default,
    );
    let v = emit_unary_math_call(ctx, "Ceiling", op);
    ctx.copy_to(res, v);
    ctx.branch(lbl_end);

    ctx.mark_label(lbl2, BlockFrequency::Default);
    ctx.branch_if(
        lbl3,
        r_mode,
        r_m,
        Comparison::NotEqual,
        BlockFrequency::Default,
    );
    let v = emit_unary_math_call(ctx, "Floor", op);
    ctx.copy_to(res, v);
    ctx.branch(lbl_end);

    ctx.mark_label(lbl3, BlockFrequency::Default);
    let v = emit_unary_math_call(ctx, "Truncate", op);
    ctx.copy_to(res, v);
    ctx.branch(lbl_end);

    ctx.mark_label(lbl_end, BlockFrequency::Default);

    res
}

fn opcode_is_single(kind: &OpCodeKind) -> bool {
    let size = match kind {
        OpCodeKind::Simd { size, .. }
        | OpCodeKind::SimdReg { size, .. }
        | OpCodeKind::SimdImm { size, .. }
        | OpCodeKind::SimdFmov { size, .. }
        | OpCodeKind::SimdCvt { size, .. }
        | OpCodeKind::SimdExt { size, .. }
        | OpCodeKind::SimdFcond { size, .. }
        | OpCodeKind::SimdRegElem { size, .. }
        | OpCodeKind::SimdRegElemF { size, .. }
        | OpCodeKind::SimdTbl { size, .. }
        | OpCodeKind::SimdIns { size, .. }
        | OpCodeKind::SimdMemLit { size, .. }
        | OpCodeKind::SimdMemImm { size, .. }
        | OpCodeKind::SimdMemPair { size, .. }
        | OpCodeKind::SimdMemReg { size, .. }
        | OpCodeKind::SimdMemMs { size, .. }
        | OpCodeKind::SimdMemSs { size, .. } => *size,
        OpCodeKind::SimdShImm { size, .. } => *size as u8,
        _ => panic!("emit_soft_float_call: OpCodeKind has no size field"),
    };
    // size: 0 = b/h/8 or f32; 1 = f64; 2+ = not float.
    // For float: size & 1 == 0 → single, size & 1 == 1 → double.
    size & 1 == 0
}

// Decoder opcode extraction (C# (IOpCodeSimd)context.CurrOp)
// These extract the common SIMD opcode fields from the decoded instruction.
// Field types match the decoder's OpCodeKind variants exactly.

use crate::decoder::utilities::RegisterSize;

pub(crate) struct SimdOp {
    pub(crate) rd: VecRegisterIndex,
    pub(crate) rn: VecRegisterIndex,
    pub(crate) size: i32,
    pub(crate) reg_size: RegisterSize,
}

pub(crate) struct SimdRegOp {
    pub(crate) rd: VecRegisterIndex,
    pub(crate) rn: VecRegisterIndex,
    pub(crate) rm: VecRegisterIndex,
    pub(crate) ra: VecRegisterIndex,
    pub(crate) size: i32,
    pub(crate) reg_size: RegisterSize,
}

pub(crate) struct SimdRegElemOp {
    pub(crate) rd: VecRegisterIndex,
    pub(crate) rn: VecRegisterIndex,
    pub(crate) rm: VecRegisterIndex,
    pub(crate) size: i32,
    pub(crate) index: u8,
    pub(crate) reg_size: RegisterSize,
}

pub(crate) fn bytes_count(reg_size: RegisterSize) -> i32 {
    if reg_size.bits() == 128 { 16 } else { 8 }
}

pub(crate) fn decode_simd_op(ctx: &ArmEmitterCtx) -> SimdOp {
    let op = ctx.curr_op();
    match op.kind {
        OpCodeKind::SimdReg { size, rd, rn, .. } => SimdOp {
            rd,
            rn,
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::Simd { size, rd, rn, .. } => SimdOp {
            rd,
            rn,
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::SimdImm { size, rd, .. } => SimdOp {
            rd,
            rn: VecRegisterIndex::new(0),
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::SimdFcond { size, rd, rn, .. } => SimdOp {
            rd,
            rn,
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::SimdRegElemF { size, rd, rn, .. } => SimdOp {
            rd,
            rn,
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::SimdRegElem { size, rd, rn, .. } => SimdOp {
            rd,
            rn,
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::SimdShImm { size, rd, rn, .. } => SimdOp {
            rd,
            rn,
            size: size as i32,
            reg_size: op.reg_size,
        },
        _ => SimdOp {
            rd: VecRegisterIndex::new(0),
            rn: VecRegisterIndex::new(0),
            size: 0,
            reg_size: RegisterSize::Int64,
        },
    }
}

pub(crate) fn decode_simd_reg_op(ctx: &ArmEmitterCtx) -> SimdRegOp {
    let op = ctx.curr_op();
    match op.kind {
        OpCodeKind::SimdReg {
            size,
            rd,
            rn,
            rm,
            ra,
            ..
        } => SimdRegOp {
            rd,
            rn,
            rm,
            ra,
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::SimdFcond {
            size,
            rd,
            rn,
            rm,
            ra,
            ..
        } => SimdRegOp {
            rd,
            rn,
            rm,
            ra,
            size: size as i32,
            reg_size: op.reg_size,
        },
        // C# OpCodeSimdRegElem/OpCodeSimdRegElemF inherit OpCodeSimdReg
        // the (OpCodeSimdReg) cast succeeds for them.
        OpCodeKind::SimdRegElem {
            size, rd, rn, rm, ..
        } => SimdRegOp {
            rd,
            rn,
            rm,
            ra: VecRegisterIndex::new(0),
            size: size as i32,
            reg_size: op.reg_size,
        },
        OpCodeKind::SimdRegElemF {
            size, rd, rn, rm, ..
        } => SimdRegOp {
            rd,
            rn,
            rm,
            ra: VecRegisterIndex::new(0),
            size: size as i32,
            reg_size: op.reg_size,
        },
        _ => panic!("decode_simd_reg_op: unexpected opcode kind {:?}", op.kind),
    }
}

fn decode_simd_reg_elem_op(ctx: &ArmEmitterCtx) -> SimdRegElemOp {
    let op = ctx.curr_op();
    match op.kind {
        OpCodeKind::SimdRegElem {
            size,
            rd,
            rn,
            rm,
            index,
            ..
        } => SimdRegElemOp {
            rd,
            rn,
            rm,
            size: size as i32,
            index,
            reg_size: op.reg_size,
        },
        _ => SimdRegElemOp {
            rd: VecRegisterIndex::new(0),
            rn: VecRegisterIndex::new(0),
            rm: VecRegisterIndex::new(0),
            size: 0,
            index: 0,
            reg_size: RegisterSize::Int64,
        },
    }
}

pub(crate) fn decode_simd_reg_elem_f_op(ctx: &ArmEmitterCtx) -> SimdRegElemOp {
    let op = ctx.curr_op();
    match op.kind {
        OpCodeKind::SimdRegElemF {
            size,
            rd,
            rn,
            rm,
            index,
            ..
        } => SimdRegElemOp {
            rd,
            rn,
            rm,
            size: size as i32,
            index,
            reg_size: op.reg_size,
        },
        _ => SimdRegElemOp {
            rd: VecRegisterIndex::new(0),
            rn: VecRegisterIndex::new(0),
            rm: VecRegisterIndex::new(0),
            size: 0,
            index: 0,
            reg_size: RegisterSize::Int64,
        },
    }
}

// CmpCondition enum (C# CmpCondition)
// Used by SIMD compare instruction emitters.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CmpCondition {
    Equal = 0,
    LessThan = 1,
    LessThanOrEqual = 2,
    UnorderedQ = 3,
    NotLessThan = 5,
    NotLessThanOrEqual = 6,
    OrderedQ = 7,
    GreaterThanOrEqual = 13,
    GreaterThan = 14,
    OrderedS = 23,
}

// EmitSse41RoundToNearestWithTiesToAwayOpF (C#)
// Round ties to away: add (0.5 - epsilon) with matching sign, then truncate.
pub fn emit_sse41_round_to_nearest_with_ties_to_away_op_f(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    scalar: bool,
    is_f32: bool,
) -> OperandIndex {
    let n_copy = ctx.copy(n);
    let rc = ctx.ctx.operands.constant_i32(x86_get_round_control(3)); // TowardsZero

    if is_f32 {
        let sign_mask = if scalar {
            x86_get_scalar_i32(ctx, i32::MIN)
        } else {
            x86_get_all_elements_i32(ctx, i32::MIN)
        };
        let sign_mask = ctx.emit_intrinsic(Intrinsic::X86Pand, &[sign_mask, n_copy]);
        // 0x3EFFFFFF == BitConverter.SingleToInt32Bits(0.5f) - 1
        let value_mask = if scalar {
            x86_get_scalar_i32(ctx, 0x3EFFFFFF)
        } else {
            x86_get_all_elements_i32(ctx, 0x3EFFFFFF)
        };
        let value_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[value_mask, sign_mask]);
        let n_copy = if scalar {
            ctx.emit_intrinsic(Intrinsic::X86Addss, &[n_copy, value_mask])
        } else {
            ctx.emit_intrinsic(Intrinsic::X86Addps, &[n_copy, value_mask])
        };
        if scalar {
            ctx.emit_intrinsic(Intrinsic::X86Roundss, &[n_copy, rc])
        } else {
            ctx.emit_intrinsic(Intrinsic::X86Roundps, &[n_copy, rc])
        }
    } else {
        let sign_mask = if scalar {
            x86_get_scalar_i64(ctx, i64::MIN)
        } else {
            x86_get_all_elements_i64(ctx, i64::MIN)
        };
        let sign_mask = ctx.emit_intrinsic(Intrinsic::X86Pand, &[sign_mask, n_copy]);
        // 0x3FDFFFFFFFFFFFFFL == BitConverter.DoubleToInt64Bits(0.5d) - 1L
        let value_mask = if scalar {
            x86_get_scalar_i64(ctx, 0x3FDFFFFFFFFFFFFF_i64)
        } else {
            x86_get_all_elements_i64(ctx, 0x3FDFFFFFFFFFFFFF_i64)
        };
        let value_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[value_mask, sign_mask]);
        let n_copy = if scalar {
            ctx.emit_intrinsic(Intrinsic::X86Addsd, &[n_copy, value_mask])
        } else {
            ctx.emit_intrinsic(Intrinsic::X86Addpd, &[n_copy, value_mask])
        };
        if scalar {
            ctx.emit_intrinsic(Intrinsic::X86Roundsd, &[n_copy, rc])
        } else {
            ctx.emit_intrinsic(Intrinsic::X86Roundpd, &[n_copy, rc])
        }
    }
}

// SaturatingFlags (C# SaturatingFlags)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SaturatingFlags(pub(crate) u8);

impl SaturatingFlags {
    pub const NONE: Self = Self(0);
    pub const BY_ELEM: Self = Self(1 << 0);
    pub const SCALAR: Self = Self(1 << 1);
    pub const SIGNED: Self = Self(1 << 2);
    pub const ADD: Self = Self(1 << 3);
    pub const SUB: Self = Self(1 << 4);
    pub const ACCUMULATE: Self = Self(1 << 5);
}

// SaturatingNarrowFlags (C# SaturatingNarrowFlags)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SaturatingNarrowFlags(u8);

impl SaturatingNarrowFlags {
    pub const SCALAR: Self = Self(1 << 0);
    pub const SIGNED_SRC: Self = Self(1 << 1);
    pub const SIGNED_DST: Self = Self(1 << 2);
    pub const SCALAR_SX_SX: Self = Self(1 << 0 | 1 << 1 | 1 << 2);
    pub const SCALAR_SX_ZX: Self = Self(1 << 0 | 1 << 1);
    pub const SCALAR_ZX_ZX: Self = Self(1 << 0);
    pub const VECTOR_SX_SX: Self = Self(1 << 1 | 1 << 2);
    pub const VECTOR_SX_ZX: Self = Self(1 << 1);
    pub const VECTOR_ZX_ZX: Self = Self(0);
}

// GetImmShl / GetImmShr (C# GetImmShl / GetImmShr)
pub fn get_imm_shl(imm: i32, size: i32) -> i32 {
    imm - (8 << size)
}

pub fn get_imm_shr(imm: i32, size: i32) -> i32 {
    (8 << (size + 1)) - imm
}

// X86GetRoundControl (C# X86GetRoundControl)
pub fn x86_get_round_control(round_mode: i32) -> i32 {
    match round_mode {
        0 => 8 | 0, // ToNearest (even)
        1 => 8 | 2, // TowardsPlusInfinity
        2 => 8 | 1, // TowardsMinusInfinity
        3 => 8 | 3, // TowardsZero
        _ => panic!("Invalid rounding mode {round_mode}"),
    }
}

// EmitCountSetBits8 (C# EmitCountSetBits8)
pub fn emit_count_set_bits8(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    let typ = ctx.ctx.operands.get(op).typ;
    let one = ctx.ctx.operands.constant_i32(1);
    let c55 = ctx.ctx.operands.constant_int(typ, 0x55);
    let shr = ctx.shift_right_ui(op, one);
    let masked = ctx.bitwise_and(shr, c55);
    let op0 = ctx.subtract(op, masked);
    let c33 = ctx.ctx.operands.constant_int(typ, 0x33);
    let c2 = ctx.ctx.operands.constant_i32(2);
    let c4 = ctx.ctx.operands.constant_i32(4);
    let c0f = ctx.ctx.operands.constant_int(typ, 0x0f);
    let shr2 = ctx.shift_right_ui(op0, c2);
    let and2 = ctx.bitwise_and(shr2, c33);
    let and0 = ctx.bitwise_and(op0, c33);
    let op1 = ctx.add(and2, and0);
    let shr4 = ctx.shift_right_ui(op1, c4);
    let sum = ctx.add(op1, shr4);
    ctx.bitwise_and(sum, c0f)
}

// Scalar by-elem FP helpers (C# EmitScalarBinaryOpByElemF / EmitScalarTernaryOpByElemF)
pub fn emit_scalar_binary_op_by_elem_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_elem_f_op(ctx);
    let typ = if (op.size & 1) != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let rn = get_vec(ctx, op.rn);
    let rm = get_vec(ctx, op.rm);
    let n = ctx.vector_extract(typ, rn, 0);
    let m = ctx.vector_extract(typ, rm, op.index as i32);
    let emitted = emit(ctx, n, m);
    let zero = ctx.vector_zero();
    let d = ctx.vector_insert(zero, emitted, 0);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

pub fn emit_scalar_ternary_op_by_elem_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_elem_f_op(ctx);
    let typ = if (op.size & 1) != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    let rd_vec = get_vec(ctx, op.rd);
    let rn = get_vec(ctx, op.rn);
    let rm = get_vec(ctx, op.rm);
    let d = ctx.vector_extract(typ, rd_vec, 0);
    let n = ctx.vector_extract(typ, rn, 0);
    let m = ctx.vector_extract(typ, rm, op.index as i32);
    let emitted = emit(ctx, d, n, m);
    let zero = ctx.vector_zero();
    let res = ctx.vector_insert(zero, emitted, 0);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Vector imm helpers (C# EmitVectorImmUnaryOp / EmitVectorImmBinaryOp)
pub fn emit_vector_imm_unary_op(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
) {
    let op_kind = ctx.curr_op().kind;
    let reg_size = ctx.curr_op().reg_size;
    let (imm_val, size, rd) = match op_kind {
        OpCodeKind::SimdImm {
            rd,
            size,
            immediate,
            ..
        } => (immediate, size as i32, rd),
        _ => panic!("emit_vector_imm_unary_op: expected SimdImm"),
    };
    let imm = ctx.ctx.operands.constant_i64(imm_val);
    let elems = bytes_count(reg_size) >> size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let emitted = emit(ctx, imm);
        res = emit_vector_insert(ctx, res, emitted, index, size);
    }
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, res);
}

pub fn emit_vector_imm_binary_op(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op_kind = ctx.curr_op().kind;
    let reg_size = ctx.curr_op().reg_size;
    let (rd, size, immediate) = match op_kind {
        OpCodeKind::SimdImm {
            rd,
            size,
            immediate,
            ..
        } => (rd, size as i32, immediate),
        _ => panic!("emit_vector_imm_binary_op: expected SimdImm"),
    };
    let imm = ctx.ctx.operands.constant_i64(immediate);
    let elems = bytes_count(reg_size) >> size;
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let de = emit_vector_extract_zx(ctx, rd, index, size);
        let emitted = emit(ctx, de, imm);
        res = emit_vector_insert(ctx, res, emitted, index, size);
    }
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, res);
}

// Widen helpers (C# EmitVectorWidenRmBinaryOpSx/Zx, etc.)
pub fn emit_vector_widen_rm_binary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_rm_binary_op(ctx, emit, true);
}

pub fn emit_vector_widen_rm_binary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_rm_binary_op(ctx, emit, false);
}

fn emit_vector_widen_rm_binary_op(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
    signed: bool,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = 8 >> op.size;
    let part = if op.reg_size.bits() == 128 { elems } else { 0 };
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract(ctx, op.rn, index, op.size + 1, signed);
        let me = emit_vector_extract(ctx, op.rm, part + index, op.size, signed);
        let emitted = emit(ctx, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index, op.size + 1);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_widen_rn_rm_binary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_rn_rm_binary_op(ctx, emit, true);
}

pub fn emit_vector_widen_rn_rm_binary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_rn_rm_binary_op(ctx, emit, false);
}

fn emit_vector_widen_rn_rm_binary_op(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
    signed: bool,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = 8 >> op.size;
    let part = if op.reg_size.bits() == 128 { elems } else { 0 };
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract(ctx, op.rn, part + index, op.size, signed);
        let me = emit_vector_extract(ctx, op.rm, part + index, op.size, signed);
        let emitted = emit(ctx, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index, op.size + 1);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_widen_rn_rm_ternary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_rn_rm_ternary_op(ctx, emit, true);
}

pub fn emit_vector_widen_rn_rm_ternary_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_rn_rm_ternary_op(ctx, emit, false);
}

fn emit_vector_widen_rn_rm_ternary_op(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
    signed: bool,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = 8 >> op.size;
    let part = if op.reg_size.bits() == 128 { elems } else { 0 };
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let de = emit_vector_extract(ctx, op.rd, index, op.size + 1, signed);
        let ne = emit_vector_extract(ctx, op.rn, part + index, op.size, signed);
        let me = emit_vector_extract(ctx, op.rm, part + index, op.size, signed);
        let emitted = emit(ctx, de, ne, me);
        res = emit_vector_insert(ctx, res, emitted, index, op.size + 1);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_vector_widen_binary_op_by_elem_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_binary_op_by_elem(ctx, emit, true);
}

pub fn emit_vector_widen_binary_op_by_elem_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_binary_op_by_elem(ctx, emit, false);
}

fn emit_vector_widen_binary_op_by_elem(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
    signed: bool,
) {
    let op_kind = ctx.curr_op().kind;
    let reg_size = ctx.curr_op().reg_size;
    let (rd, rn, rm, size, index) = match op_kind {
        OpCodeKind::SimdRegElem {
            rd,
            rn,
            rm,
            size,
            index,
            ..
        } => (rd, rn, rm, size as i32, index as i32),
        _ => panic!("emit_vector_widen_binary_op_by_elem: expected SimdRegElem"),
    };
    let me = emit_vector_extract(ctx, rm, index, size, signed);
    let elems = 8 >> size;
    let part = if reg_size.bits() == 128 { elems } else { 0 };
    let mut res = ctx.vector_zero();
    for idx in 0..elems {
        let ne = emit_vector_extract(ctx, rn, part + idx, size, signed);
        let emitted = emit(ctx, ne, me);
        res = emit_vector_insert(ctx, res, emitted, idx, size + 1);
    }
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, res);
}

pub fn emit_vector_widen_ternary_op_by_elem_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_ternary_op_by_elem(ctx, emit, true);
}

pub fn emit_vector_widen_ternary_op_by_elem_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_widen_ternary_op_by_elem(ctx, emit, false);
}

fn emit_vector_widen_ternary_op_by_elem(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex, OperandIndex) -> OperandIndex,
    signed: bool,
) {
    let op_kind = ctx.curr_op().kind;
    let reg_size = ctx.curr_op().reg_size;
    let (rd, rn, rm, size, index) = match op_kind {
        OpCodeKind::SimdRegElem {
            rd,
            rn,
            rm,
            size,
            index,
            ..
        } => (rd, rn, rm, size as i32, index as i32),
        _ => panic!("emit_vector_widen_ternary_op_by_elem: expected SimdRegElem"),
    };
    let me = emit_vector_extract(ctx, rm, index, size, signed);
    let elems = 8 >> size;
    let part = if reg_size.bits() == 128 { elems } else { 0 };
    let mut res = ctx.vector_zero();
    for idx in 0..elems {
        let de = emit_vector_extract(ctx, rd, idx, size + 1, signed);
        let ne = emit_vector_extract(ctx, rn, part + idx, size, signed);
        let emitted = emit(ctx, de, ne, me);
        res = emit_vector_insert(ctx, res, emitted, idx, size + 1);
    }
    let rd_op = get_vec(ctx, rd);
    ctx.copy_to(rd_op, res);
}

// Pairwise helpers (C# EmitVectorPairwiseOpSx/Zx)
pub fn emit_vector_pairwise_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_pairwise_op(ctx, emit, true);
}

pub fn emit_vector_pairwise_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_pairwise_op(ctx, emit, false);
}

fn emit_vector_pairwise_op(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
    signed: bool,
) {
    let op = decode_simd_reg_op(ctx);
    // C# op.GetPairsCount() >> op.Size, GetPairsCount() = GetBytesCount() >> 1.
    let pairs = (bytes_count(op.reg_size) >> 1) >> op.size;
    let mut res = ctx.vector_zero();
    for index in 0..pairs {
        let pair_index = index << 1;
        let n0 = emit_vector_extract(ctx, op.rn, pair_index, op.size, signed);
        let n1 = emit_vector_extract(ctx, op.rn, pair_index + 1, op.size, signed);
        let m0 = emit_vector_extract(ctx, op.rm, pair_index, op.size, signed);
        let m1 = emit_vector_extract(ctx, op.rm, pair_index + 1, op.size, signed);
        let emitted_n = emit(ctx, n0, n1);
        let emitted_m = emit(ctx, m0, m1);
        res = emit_vector_insert(ctx, res, emitted_n, index, op.size);
        res = emit_vector_insert(ctx, res, emitted_m, pairs + index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Across-vector helpers (C# EmitVectorAcrossVectorOpSx/Zx, etc.)
pub fn emit_vector_across_vector_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_across_vector_op(ctx, emit, true, false);
}

pub fn emit_vector_across_vector_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_across_vector_op(ctx, emit, false, false);
}

pub fn emit_vector_long_across_vector_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_across_vector_op(ctx, emit, true, true);
}

pub fn emit_vector_long_across_vector_op_zx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    emit_vector_across_vector_op(ctx, emit, false, true);
}

fn emit_vector_across_vector_op(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
    signed: bool,
    is_long: bool,
) {
    let op = decode_simd_op(ctx);
    let elems = bytes_count(op.reg_size) >> op.size;
    let mut res = emit_vector_extract(ctx, op.rn, 0, op.size, signed);
    for index in 1..elems {
        let n = emit_vector_extract(ctx, op.rn, index, op.size, signed);
        res = emit(ctx, res, n);
    }
    let out_size = if is_long { op.size + 1 } else { op.size };
    let zero = ctx.vector_zero();
    let d = emit_vector_insert(ctx, zero, res, 0, out_size);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, d);
}

// Saturating helpers (C# EmitSignedSrcSatQ, EmitUnsignedSrcSatQ, etc.)
pub fn emit_signed_src_sat_q(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    size_dst: i32,
    signed_dst: bool,
) -> OperandIndex {
    let e_size_dst = 8 << size_dst;
    let max_t = if signed_dst {
        ctx.ctx
            .operands
            .constant_i64((1i64 << (e_size_dst - 1)) - 1)
    } else {
        ctx.ctx.operands.constant_u64((1u64 << e_size_dst) - 1)
    };
    let min_t = if signed_dst {
        ctx.ctx.operands.constant_i64(-(1i64 << (e_size_dst - 1)))
    } else {
        ctx.ctx.operands.const_zero(OperandType::I64)
    };
    let lbl1 = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, op);
    ctx.branch_if(
        lbl1,
        op,
        max_t,
        Comparison::LessOrEqual,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_t);
    // set Qc flag
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl1, BlockFrequency::Default);
    ctx.branch_if(
        lbl_end,
        op,
        min_t,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, min_t);
    // set Qc flag
    emit_set_qc_flag(ctx);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

pub fn emit_unsigned_src_sat_q(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    size_dst: i32,
    signed_dst: bool,
) -> OperandIndex {
    let e_size_dst = 8 << size_dst;
    let max_t = if signed_dst {
        ctx.ctx
            .operands
            .constant_i64((1i64 << (e_size_dst - 1)) - 1)
    } else {
        ctx.ctx.operands.constant_u64((1u64 << e_size_dst) - 1)
    };
    let lbl_end = ctx.ctx.operands.label();
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, op);
    ctx.branch_if(
        lbl_end,
        op,
        max_t,
        Comparison::LessOrEqualUI,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_t);
    // set Qc flag
    emit_set_qc_flag(ctx);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// Sse2 scalar pairwise op F (C# EmitSse2ScalarPairwiseOpF)
pub fn emit_sse2_scalar_pairwise_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    let n = get_vec(ctx, op.rn);
    let (op0, op1) = if (op.size & 1) == 0 {
        let zero_n = ctx.vector_zero_upper64(n);
        let sm0 = ctx
            .ctx
            .operands
            .constant_i32((2 << 6) | (2 << 4) | (2 << 2) | (0 << 0));
        let sm1 = ctx
            .ctx
            .operands
            .constant_i32((2 << 6) | (2 << 4) | (2 << 2) | (1 << 0));
        (
            ctx.emit_intrinsic(Intrinsic::X86Pshufd, &[zero_n, sm0]),
            ctx.emit_intrinsic(Intrinsic::X86Pshufd, &[zero_n, sm1]),
        )
    } else {
        let zero = ctx.vector_zero();
        (
            ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[n, zero]),
            ctx.emit_intrinsic(Intrinsic::X86Movhlps, &[zero, n]),
        )
    };
    let result = emit(ctx, op0, op1);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, result);
}

// Vector pairwise op F (C# EmitVectorPairwiseOpF)
pub fn emit_vector_pairwise_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let size_f = op.size & 1;
    let typ = if size_f != 0 {
        OperandType::FP64
    } else {
        OperandType::FP32
    };
    // C# op.GetPairsCount() >> sizeF + 2, GetPairsCount() = GetBytesCount() >> 1.
    let pairs = (bytes_count(op.reg_size) >> 1) >> (size_f + 2);
    let mut res = ctx.vector_zero();
    for index in 0..pairs {
        let pair_index = index << 1;
        let rn = get_vec(ctx, op.rn);
        let rm = get_vec(ctx, op.rm);
        let n0 = ctx.vector_extract(typ, rn, pair_index);
        let n1 = ctx.vector_extract(typ, rn, pair_index + 1);
        let m0 = ctx.vector_extract(typ, rm, pair_index);
        let m1 = ctx.vector_extract(typ, rm, pair_index + 1);
        let emitted_n = emit(ctx, n0, n1);
        let emitted_m = emit(ctx, m0, m1);
        res = ctx.vector_insert(res, emitted_n, index);
        res = ctx.vector_insert(res, emitted_m, pairs + index);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Arm FP mode (InstEmitSimdHelper.cs EnterArmFpMode / ExitArmFpMode)
//
// Host is x86-64 only: the C# UseAdvSimd arm is excluded; UseSse2 is the
// only live path (SSE2 is baseline on x86-64, but the knob is honored).

use crate::codegen::x86_64::mxcsr;
use crate::ir::types::Intrinsic as Intrin;
use crate::optimizations;
use crate::state::FPState;

// Register-operand getter  the in-function path (RegisterUsage maps
// FpFlag registers to context loads/stores).
pub fn enter_arm_fp_mode(ctx: &mut ArmEmitterCtx) {
    enter_arm_fp_mode_with(ctx, |ctx, flag| {
        super::helpers::get_fp_flag(flag, &mut ctx.ctx.operands)
    });
}

// C# InstEmitSimdHelper.EnterArmFpMode(context, Func<FPState, Operand>)
// the getter closure abstracts flag access: register operands inside guest
// functions, raw context-memory loads inside TranslatorStubs.
pub fn enter_arm_fp_mode_with(
    ctx: &mut ArmEmitterCtx,
    get_fp_flag: impl Fn(&mut ArmEmitterCtx, FPState) -> OperandIndex,
) {
    if optimizations::use_sse2() {
        let mut mxcsr_v = ctx.emit_intrinsic_int(Intrin::X86Stmxcsr, &[]);

        let fz_true = get_fp_flag(ctx, FPState::FzFlag);
        let r0_true = get_fp_flag(ctx, FPState::RMode0Flag);
        let r1_true = get_fp_flag(ctx, FPState::RMode1Flag);

        let clear = ctx
            .ctx
            .operands
            .constant_i32(!(mxcsr::FTZ | mxcsr::DAZ | mxcsr::RHI | mxcsr::RLO));
        mxcsr_v = ctx.bitwise_and(mxcsr_v, clear);

        let fz_bits = ctx
            .ctx
            .operands
            .constant_i32(mxcsr::FTZ | mxcsr::DAZ | mxcsr::UM | mxcsr::DM);
        let zero = ctx.ctx.operands.constant_i32(0);
        let fz_sel = ctx.conditional_select(fz_true, fz_bits, zero);
        mxcsr_v = ctx.bitwise_or(mxcsr_v, fz_sel);

        // X86 round modes in order: nearest, negative, positive, zero
        // ARM round modes in order: nearest, positive, negative, zero
        // Read the bits backwards to correct this.
        let rhi = ctx.ctx.operands.constant_i32(mxcsr::RHI);
        let zero = ctx.ctx.operands.constant_i32(0);
        let r0_sel = ctx.conditional_select(r0_true, rhi, zero);
        mxcsr_v = ctx.bitwise_or(mxcsr_v, r0_sel);

        let rlo = ctx.ctx.operands.constant_i32(mxcsr::RLO);
        let zero = ctx.ctx.operands.constant_i32(0);
        let r1_sel = ctx.conditional_select(r1_true, rlo, zero);
        mxcsr_v = ctx.bitwise_or(mxcsr_v, r1_sel);

        ctx.emit_intrinsic_no_ret(Intrin::X86Ldmxcsr, &[mxcsr_v]);
    }
}

pub fn exit_arm_fp_mode(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let mut mxcsr_v = ctx.emit_intrinsic_int(Intrin::X86Stmxcsr, &[]);

        // Unset round mode (to nearest) and ftz.
        let clear = ctx
            .ctx
            .operands
            .constant_i32(!(mxcsr::FTZ | mxcsr::DAZ | mxcsr::RHI | mxcsr::RLO));
        mxcsr_v = ctx.bitwise_and(mxcsr_v, clear);

        ctx.emit_intrinsic_no_ret(Intrin::X86Ldmxcsr, &[mxcsr_v]);

        // Status flags would be stored here if they were used.
    }
}

// X86GetScalar / X86GetAllElements (InstEmitSimdHelper.cs)
pub fn x86_get_all_elements_i16(ctx: &mut ArmEmitterCtx, value: i16) -> OperandIndex {
    let value1 = value as u16 as u64;
    let value2 = value1 << 16 | value1;
    let value4 = value2 << 32 | value2;
    x86_get_all_elements_i64(ctx, value4 as i64)
}

// SimdImm decode + EmitVectorImmBinaryOp
pub(crate) struct SimdImmOp {
    pub rd: VecRegisterIndex,
    pub immediate: i64,
    pub size: i32,
    pub reg_size: RegisterSize,
}

pub(crate) fn decode_simd_imm_op(ctx: &ArmEmitterCtx) -> SimdImmOp {
    let op = ctx.curr_op();
    match op.kind {
        OpCodeKind::SimdImm {
            rd,
            immediate,
            size,
        } => SimdImmOp {
            rd,
            immediate,
            size: size as i32,
            reg_size: op.reg_size,
        },
        _ => panic!("expected SimdImm opcode, got {:?}", op.kind),
    }
}

// C# EmitVectorImmBinaryOp.

pub(crate) fn simd_bytes_count(reg_size: RegisterSize) -> i32 {
    bytes_count(reg_size)
}

// C# `SetFpFlag(context, FPState.QcFlag, Const(1))`  the saturating
// helpers store 1 to the QC flag register directly.
fn emit_set_qc_flag(ctx: &mut ArmEmitterCtx) {
    let one = ctx.ctx.operands.constant_i32(1);
    crate::instructions::helpers::set_fp_flag(ctx, FPState::QcFlag, one);
}

// EmitAbs (C# private EmitAbs in SimdArithmetic)
pub fn emit_abs(ctx: &mut ArmEmitterCtx, value: OperandIndex) -> OperandIndex {
    let typ = ctx.ctx.operands.get(value).typ;
    let zero = ctx.ctx.operands.const_zero(typ);
    let is_positive = ctx.icompare_greater_or_equal(value, zero);
    let negated = ctx.negate(value);
    ctx.conditional_select(is_positive, value, negated)
}

// EmitMax64Op / EmitMin64Op (C# private)
pub fn emit_max64_op(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
    signed: bool,
) -> OperandIndex {
    let cmp = if signed {
        ctx.icompare_greater_or_equal(op1, op2)
    } else {
        ctx.icompare_greater_or_equal_ui(op1, op2)
    };
    ctx.conditional_select(cmp, op1, op2)
}

pub fn emit_min64_op(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
    signed: bool,
) -> OperandIndex {
    let cmp = if signed {
        ctx.icompare_less_or_equal(op1, op2)
    } else {
        ctx.icompare_less_or_equal_ui(op1, op2)
    };
    ctx.conditional_select(cmp, op1, op2)
}

// EmitHighNarrow (C# private)
pub fn emit_high_narrow(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
    round: bool,
) {
    let op = decode_simd_reg_op(ctx);
    let elems = 8 >> op.size;
    let e_size = 8 << op.size;
    let part = if op.reg_size.bits() == 128 { elems } else { 0 };
    let d = get_vec(ctx, op.rd);
    let mut res = if part == 0 {
        ctx.vector_zero()
    } else {
        ctx.copy(d)
    };
    let round_const = ctx.ctx.operands.constant_i64(1i64 << (e_size - 1));
    for index in 0..elems {
        let ne = emit_vector_extract_zx(ctx, op.rn, index, op.size + 1);
        let me = emit_vector_extract_zx(ctx, op.rm, index, op.size + 1);
        let mut de = emit(ctx, ne, me);
        if round {
            de = ctx.add(de, round_const);
        }
        let shift = ctx.ctx.operands.constant_i32(e_size);
        de = ctx.shift_right_ui(de, shift);
        res = emit_vector_insert(ctx, res, de, part + index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// EmitAddLongPairwise (C# private)
pub fn emit_add_long_pairwise(ctx: &mut ArmEmitterCtx, signed: bool, accumulate: bool) {
    let op = decode_simd_op(ctx);
    let mut res = ctx.vector_zero();
    let pairs = (bytes_count(op.reg_size) >> op.size) >> 1;
    for index in 0..pairs {
        let pair_index = index << 1;
        let ne0 = emit_vector_extract(ctx, op.rn, pair_index, op.size, signed);
        let ne1 = emit_vector_extract(ctx, op.rn, pair_index + 1, op.size, signed);
        let mut e = ctx.add(ne0, ne1);
        if accumulate {
            let de = emit_vector_extract(ctx, op.rd, index, op.size + 1, signed);
            e = ctx.add(e, de);
        }
        res = emit_vector_insert(ctx, res, e, index, op.size + 1);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// EmitDoublingMultiplyHighHalf (C# private)
pub fn emit_doubling_multiply_high_half(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    m: OperandIndex,
    round: bool,
) -> OperandIndex {
    let op = decode_simd_reg_op(ctx);
    let e_size = 8 << op.size;
    let res = ctx.multiply(n, m);
    if !round {
        let shift = ctx.ctx.operands.constant_i32(e_size - 1);
        ctx.shift_right_si(res, shift)
    } else {
        let round_const = ctx.ctx.operands.constant_i64(1i64 << (e_size - 1));
        let one = ctx.ctx.operands.constant_i32(1);
        let mut r = ctx.shift_left(res, one);
        r = ctx.add(r, round_const);
        let shift = ctx.ctx.operands.constant_i32(e_size);
        r = ctx.shift_right_si(r, shift);
        // C# checks against (long)int.MinValue = -2147483648 = 0xFFFFFFFF80000000
        let int_min_val = ctx.ctx.operands.constant_i64(-2147483648i64);
        let is_intmin = ctx.icompare_equal(r, int_min_val);
        let neg = ctx.negate(r);
        ctx.conditional_select(is_intmin, neg, r)
    }
}

// Unary saturating helpers (C# EmitSignedSignSatQ, EmitUnsignedSignSatQ)
pub fn emit_signed_sign_sat_q(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    size: i32,
) -> OperandIndex {
    let e_size = 8 << size;
    let lbl1 = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();
    let zero = ctx.ctx.operands.constant_i64(0);
    // C# unchecked: for eSize == 64, (1L << 63) - 1 wraps to long.MaxValue
    // and -(1L << 63) stays long.MinValue.
    let max_t = ctx
        .ctx
        .operands
        .constant_i64((1i64 << (e_size - 1)).wrapping_sub(1));
    let min_t = ctx
        .ctx
        .operands
        .constant_i64((1i64 << (e_size - 1)).wrapping_neg());
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, zero);
    ctx.branch_if(
        lbl1,
        op,
        zero,
        Comparison::LessOrEqual,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_t);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl1, BlockFrequency::Default);
    ctx.branch_if(
        lbl_end,
        op,
        zero,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, min_t);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

pub fn emit_unsigned_sign_sat_q(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
    size: i32,
) -> OperandIndex {
    let e_size = 8 << size;
    let lbl_end = ctx.ctx.operands.label();
    let zero = ctx.ctx.operands.constant_u64(0);
    let max_t = ctx.ctx.operands.constant_u64(u64::MAX >> (64 - e_size));
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, zero);
    ctx.branch_if(
        lbl_end,
        op,
        zero,
        Comparison::LessOrEqualUI,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_t);
    emit_set_qc_flag(ctx);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// UnarySignedSatQAbsOrNeg (C# private)
pub fn emit_unary_signed_sat_q_abs_or_neg(
    ctx: &mut ArmEmitterCtx,
    op: OperandIndex,
) -> OperandIndex {
    let lbl_end = ctx.ctx.operands.label();
    let min_l = ctx.ctx.operands.constant_i64(i64::MIN);
    let max_l = ctx.ctx.operands.constant_i64(i64::MAX);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, op);
    ctx.branch_if(
        lbl_end,
        op,
        min_l,
        Comparison::NotEqual,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_l);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// BinarySignedSatQAdd (C# public)
pub fn emit_binary_signed_sat_q_add(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
) -> OperandIndex {
    let lbl_end = ctx.ctx.operands.label();
    let min_l = ctx.ctx.operands.constant_i64(i64::MIN);
    let max_l = ctx.ctx.operands.constant_i64(i64::MAX);
    let zero_l = ctx.ctx.operands.constant_i64(0);
    let add = ctx.add(op1, op2);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, add);
    let xor_ns_op = ctx.bitwise_xor(op1, op2);
    let left = ctx.bitwise_not(xor_ns_op);
    let right = ctx.bitwise_xor(op1, add);
    let test = ctx.bitwise_and(left, right);
    ctx.branch_if(
        lbl_end,
        test,
        zero_l,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    let is_positive = ctx.icompare_greater_or_equal(op1, zero_l);
    let sat = ctx.conditional_select(is_positive, max_l, min_l);
    ctx.copy_to(res, sat);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// BinaryUnsignedSatQAdd (C# public)
pub fn emit_binary_unsigned_sat_q_add(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
) -> OperandIndex {
    let lbl_end = ctx.ctx.operands.label();
    let max_ul = ctx.ctx.operands.constant_u64(u64::MAX);
    let add = ctx.add(op1, op2);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, add);
    ctx.branch_if(
        lbl_end,
        add,
        op1,
        Comparison::GreaterOrEqualUI,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_ul);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// BinarySignedSatQSub (C# public)
pub fn emit_binary_signed_sat_q_sub(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
) -> OperandIndex {
    let lbl_end = ctx.ctx.operands.label();
    let min_l = ctx.ctx.operands.constant_i64(i64::MIN);
    let max_l = ctx.ctx.operands.constant_i64(i64::MAX);
    let zero_l = ctx.ctx.operands.constant_i64(0);
    let sub = ctx.subtract(op1, op2);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, sub);
    let left = ctx.bitwise_xor(op1, op2);
    let right = ctx.bitwise_xor(op1, sub);
    let test = ctx.bitwise_and(left, right);
    ctx.branch_if(
        lbl_end,
        test,
        zero_l,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    let is_positive = ctx.icompare_greater_or_equal(op1, zero_l);
    let sat = ctx.conditional_select(is_positive, max_l, min_l);
    ctx.copy_to(res, sat);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// BinaryUnsignedSatQSub (C# public)
pub fn emit_binary_unsigned_sat_q_sub(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
) -> OperandIndex {
    let lbl_end = ctx.ctx.operands.label();
    let zero_l = ctx.ctx.operands.constant_i64(0);
    let sub = ctx.subtract(op1, op2);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, sub);
    ctx.branch_if(
        lbl_end,
        op1,
        op2,
        Comparison::GreaterOrEqualUI,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, zero_l);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// EmitSaturatingUnaryOpSx (C#)
pub fn emit_saturating_unary_op_sx(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex) -> OperandIndex,
    flags: SaturatingFlags,
) {
    let op = decode_simd_op(ctx);
    let scalar = (flags.0 & SaturatingFlags::SCALAR.0) != 0;
    let elems = if !scalar {
        bytes_count(op.reg_size) >> op.size
    } else {
        1
    };
    let mut res = ctx.vector_zero();
    for index in 0..elems {
        let ne = emit_vector_extract_sx(ctx, op.rn, index, op.size);
        let emitted = emit(ctx, ne);
        let de = if op.size <= 2 {
            emit_signed_src_sat_q(ctx, emitted, op.size, true)
        } else {
            emit_unary_signed_sat_q_abs_or_neg(ctx, emitted)
        };
        res = emit_vector_insert(ctx, res, de, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// EmitSaturatingBinaryOp (C#)
pub fn emit_saturating_binary_op(
    ctx: &mut ArmEmitterCtx,
    emit: Option<fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex>,
    flags: SaturatingFlags,
) {
    let op_kind = ctx.curr_op().kind;
    let reg_size = ctx.curr_op().reg_size;
    let by_elem = (flags.0 & SaturatingFlags::BY_ELEM.0) != 0;
    let scalar = (flags.0 & SaturatingFlags::SCALAR.0) != 0;
    let signed = (flags.0 & SaturatingFlags::SIGNED.0) != 0;
    let add = (flags.0 & SaturatingFlags::ADD.0) != 0;
    let sub = (flags.0 & SaturatingFlags::SUB.0) != 0;
    let accumulate = (flags.0 & SaturatingFlags::ACCUMULATE.0) != 0;
    let (rd_idx, rn_idx, rm_idx, size) = match op_kind {
        OpCodeKind::SimdReg {
            rd, rn, rm, size, ..
        } => (rd, rn, rm, size as i32),
        OpCodeKind::SimdRegElem {
            rd, rn, rm, size, ..
        } => (rd, rn, rm, size as i32),
        OpCodeKind::Simd { rd, rn, size, .. } => (rd, rn, VecRegisterIndex::new(0), size as i32),
        _ => panic!("emit_saturating_binary_op: unexpected opcode kind"),
    };
    let elems = if !scalar {
        bytes_count(reg_size) >> size
    } else {
        1
    };
    let mut res = ctx.vector_zero();
    if add || sub {
        if !scalar && size <= 1 {
            // Packed saturating add/sub for byte/word: one SSE2 instruction over
            // the whole vector instead of the per-lane clamp loop. QC is preserved
            // by detecting saturation as (saturating result != wraparound result)
            // and OR-ing it into FPSR.QC (sticky), matching the scalar path which
            // sets QC whenever a lane clamps.
            let n = get_vec(ctx, rn_idx);
            let m = get_vec(ctx, rm_idx);
            let sat_inst = match (signed, add, size) {
                (true, true, 0) => Intrinsic::X86Paddsb,
                (true, true, _) => Intrinsic::X86Paddsw,
                (false, true, 0) => Intrinsic::X86Paddusb,
                (false, true, _) => Intrinsic::X86Paddusw,
                (true, false, 0) => Intrinsic::X86Psubsb,
                (true, false, _) => Intrinsic::X86Psubsw,
                (false, false, 0) => Intrinsic::X86Psubusb,
                (false, false, _) => Intrinsic::X86Psubusw,
            };
            let sat = ctx.emit_intrinsic(sat_inst, &[n, m]);
            let wrap_inst = if add {
                X86_PADD_INSTRUCTION[size as usize]
            } else {
                X86_PSUB_INSTRUCTION[size as usize]
            };
            let wrap = ctx.emit_intrinsic(wrap_inst, &[n, m]);
            let mut diff = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[sat, wrap]);
            if reg_size.bits() == 64 {
                // The upper half holds unrelated register contents, which would otherwise
                // report a saturation the instruction never performed.
                diff = ctx.vector_zero_upper64(diff);
            }
            // PMOVMSKB only reports each byte's high bit, and a clamped lane's difference is
            // frequently small (0xFFFF ^ 0xFFFE == 1), so testing the raw difference misses
            // most saturations. Turn it into a per-byte all-ones mask against zero first;
            // every byte then reads back as set when nothing saturated.
            let vzero = ctx.vector_zero();
            let eq = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqb, &[diff, vzero]);
            let mask = ctx.emit_intrinsic_int(Intrinsic::X86Pmovmskb, &[eq]);
            let all_eq = ctx.ctx.operands.constant_i32(0xFFFF);
            let saturated = ctx.icompare_not_equal(mask, all_eq);
            let cur_qc =
                crate::instructions::helpers::get_fp_flag(FPState::QcFlag, &mut ctx.ctx.operands);
            let new_qc = ctx.bitwise_or(cur_qc, saturated);
            crate::instructions::helpers::set_fp_flag(ctx, FPState::QcFlag, new_qc);
            let mut res = sat;
            if reg_size.bits() == 64 {
                res = ctx.vector_zero_upper64(res);
            }
            let rd = get_vec(ctx, rd_idx);
            ctx.copy_to(rd, res);
            return;
        }
        for index in 0..elems {
            let ne = emit_vector_extract(ctx, rn_idx, index, size, signed);
            let me = emit_vector_extract(ctx, rm_idx, index, size, signed);
            let de_ns = if add {
                ctx.add(ne, me)
            } else {
                ctx.subtract(ne, me)
            };
            let de = if size <= 2 {
                emit_signed_src_sat_q(ctx, de_ns, size, signed)
            } else {
                if add {
                    if signed {
                        emit_binary_signed_sat_q_add(ctx, ne, me)
                    } else {
                        emit_binary_unsigned_sat_q_add(ctx, ne, me)
                    }
                } else {
                    if signed {
                        emit_binary_signed_sat_q_sub(ctx, ne, me)
                    } else {
                        emit_binary_unsigned_sat_q_sub(ctx, ne, me)
                    }
                }
            };
            res = emit_vector_insert(ctx, res, de, index, size);
        }
    } else if accumulate {
        for index in 0..elems {
            let ne = emit_vector_extract(ctx, rn_idx, index, size, !signed);
            let me = emit_vector_extract(ctx, rd_idx, index, size, signed);
            let de_ns = ctx.add(ne, me);
            let de = if size <= 2 {
                emit_signed_src_sat_q(ctx, de_ns, size, signed)
            } else {
                if signed {
                    emit_binary_signed_sat_q_acc(ctx, ne, me)
                } else {
                    emit_binary_unsigned_sat_q_acc(ctx, ne, me)
                }
            };
            res = emit_vector_insert(ctx, res, de, index, size);
        }
    } else {
        let emit_fn =
            emit.expect("emit_saturating_binary_op: emit required for non-add/sub/accumulate");
        for index in 0..elems {
            let ne = emit_vector_extract(ctx, rn_idx, index, size, signed);
            let me_idx = if by_elem {
                match op_kind {
                    OpCodeKind::SimdRegElem {
                        rm,
                        index: elem_idx,
                        ..
                    } => emit_vector_extract(ctx, rm, elem_idx as i32, size, signed),
                    _ => panic!("emit_saturating_binary_op: expected SimdRegElem for by_elem"),
                }
            } else {
                emit_vector_extract(ctx, rm_idx, index, size, signed)
            };
            let emitted = emit_fn(ctx, ne, me_idx);
            let de = emit_signed_src_sat_q(ctx, emitted, size, signed);
            res = emit_vector_insert(ctx, res, de, index, size);
        }
    }
    let rd = get_vec(ctx, rd_idx);
    ctx.copy_to(rd, res);
}

// BinarySignedSatQAcc / BinaryUnsignedSatQAcc (C# private)
pub fn emit_binary_signed_sat_q_acc(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
) -> OperandIndex {
    let lbl1 = ctx.ctx.operands.label();
    let lbl2 = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();
    let max_l = ctx.ctx.operands.constant_i64(i64::MAX);
    let zero_l = ctx.ctx.operands.constant_i64(0);
    let add = ctx.add(op1, op2);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, add);
    ctx.branch_if(
        lbl1,
        op1,
        max_l,
        Comparison::GreaterUI,
        BlockFrequency::Default,
    );
    let not_op2 = ctx.bitwise_not(op2);
    let not_op2_and_res = ctx.bitwise_and(not_op2, add);
    ctx.branch_if(
        lbl_end,
        not_op2_and_res,
        zero_l,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_l);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl1, BlockFrequency::Default);
    ctx.branch_if(lbl2, op2, zero_l, Comparison::Less, BlockFrequency::Default);
    ctx.copy_to(res, max_l);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl2, BlockFrequency::Default);
    ctx.branch_if(
        lbl_end,
        add,
        max_l,
        Comparison::LessOrEqualUI,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_l);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

pub fn emit_binary_unsigned_sat_q_acc(
    ctx: &mut ArmEmitterCtx,
    op1: OperandIndex,
    op2: OperandIndex,
) -> OperandIndex {
    let lbl1 = ctx.ctx.operands.label();
    let lbl_end = ctx.ctx.operands.label();
    let max_ul = ctx.ctx.operands.constant_u64(u64::MAX);
    let max_l = ctx.ctx.operands.constant_i64(i64::MAX);
    let zero_l = ctx.ctx.operands.constant_i64(0);
    let add = ctx.add(op1, op2);
    let res = ctx.allocate_local(OperandType::I64);
    ctx.copy_to(res, add);
    ctx.branch_if(lbl1, op1, zero_l, Comparison::Less, BlockFrequency::Default);
    ctx.branch_if(
        lbl_end,
        add,
        op1,
        Comparison::GreaterOrEqualUI,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, max_ul);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl1, BlockFrequency::Default);
    ctx.branch_if(
        lbl_end,
        op2,
        max_l,
        Comparison::GreaterUI,
        BlockFrequency::Default,
    );
    ctx.branch_if(
        lbl_end,
        add,
        zero_l,
        Comparison::GreaterOrEqual,
        BlockFrequency::Default,
    );
    ctx.copy_to(res, zero_l);
    emit_set_qc_flag(ctx);
    ctx.branch(lbl_end);
    ctx.mark_label(lbl_end, BlockFrequency::Default);
    res
}

// EmitSaturatingNarrowOp (C#)
pub fn emit_saturating_narrow_op(ctx: &mut ArmEmitterCtx, flags: SaturatingNarrowFlags) {
    let op = decode_simd_op(ctx);
    let scalar = (flags.0 & SaturatingNarrowFlags::SCALAR.0) != 0;
    let signed_src = (flags.0 & SaturatingNarrowFlags::SIGNED_SRC.0) != 0;
    let signed_dst = (flags.0 & SaturatingNarrowFlags::SIGNED_DST.0) != 0;
    let elems = if !scalar { 8 >> op.size } else { 1 };
    let part = if !scalar && op.reg_size.bits() == 128 {
        elems
    } else {
        0
    };
    let d = get_vec(ctx, op.rd);
    let mut res = if part == 0 {
        ctx.vector_zero()
    } else {
        ctx.copy(d)
    };
    for index in 0..elems {
        let ne = emit_vector_extract(ctx, op.rn, index, op.size + 1, signed_src);
        let temp = if signed_src {
            emit_signed_src_sat_q(ctx, ne, op.size, signed_dst)
        } else {
            emit_unsigned_src_sat_q(ctx, ne, op.size, signed_dst)
        };
        res = emit_vector_insert(ctx, res, temp, part + index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// EmitSsse3VectorPairwiseOp (from InstEmitSimdHelper.cs)
pub fn emit_ssse3_vector_pairwise_op(ctx: &mut ArmEmitterCtx, inst: &[Intrinsic; 4]) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 64 {
        let zero_even_mask = x86_get_elements(ctx, ZERO_MASK, EVEN_MASKS[op.size as usize]);
        let zero_odd_mask = x86_get_elements(ctx, ZERO_MASK, ODD_MASKS[op.size as usize]);
        let mn = ctx.emit_intrinsic(Intrinsic::X86Punpcklqdq, &[n, m]);
        let left = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[mn, zero_even_mask]);
        let right = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[mn, zero_odd_mask]);
        let rd = get_vec(ctx, op.rd);
        let result = ctx.emit_intrinsic(inst[op.size as usize], &[left, right]);
        ctx.copy_to(rd, result);
    } else if op.size < 3 {
        let odd_even_mask = x86_get_elements(
            ctx,
            ODD_MASKS[op.size as usize],
            EVEN_MASKS[op.size as usize],
        );
        let odd_even_n = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n, odd_even_mask]);
        let odd_even_m = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[m, odd_even_mask]);
        let left = ctx.emit_intrinsic(Intrinsic::X86Punpcklqdq, &[odd_even_n, odd_even_m]);
        let right = ctx.emit_intrinsic(Intrinsic::X86Punpckhqdq, &[odd_even_n, odd_even_m]);
        let rd = get_vec(ctx, op.rd);
        let result = ctx.emit_intrinsic(inst[op.size as usize], &[left, right]);
        ctx.copy_to(rd, result);
    } else {
        let left = ctx.emit_intrinsic(Intrinsic::X86Punpcklqdq, &[n, m]);
        let right = ctx.emit_intrinsic(Intrinsic::X86Punpckhqdq, &[n, m]);
        let rd = get_vec(ctx, op.rd);
        let result = ctx.emit_intrinsic(inst[3], &[left, right]);
        ctx.copy_to(rd, result);
    }
}

// EmitSse41RoundToNearestWithTiesToAwayOpF (from InstEmitSimdHelper.cs)
pub fn emit_sse41_round_to_nearest_ties_away_op_f(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    scalar: bool,
) -> OperandIndex {
    let op = decode_simd_op(ctx);
    let rc = ctx.ctx.operands.constant_i32(x86_get_round_control(3)); // TowardsZero
    if (op.size & 1) == 0 {
        let sign_mask = if scalar {
            x86_get_scalar_i32(ctx, i32::MIN)
        } else {
            x86_get_all_elements_i32(ctx, i32::MIN)
        };
        let sign_mask = ctx.emit_intrinsic(Intrinsic::X86Pand, &[sign_mask, n]);
        let value_mask = if scalar {
            x86_get_scalar_i32(ctx, 0x3EFFFFFF)
        } else {
            x86_get_all_elements_i32(ctx, 0x3EFFFFFF)
        };
        let value_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[value_mask, sign_mask]);
        let mut nc = ctx.emit_intrinsic(
            if scalar {
                Intrinsic::X86Addss
            } else {
                Intrinsic::X86Addps
            },
            &[n, value_mask],
        );
        nc = ctx.emit_intrinsic(
            if scalar {
                Intrinsic::X86Roundss
            } else {
                Intrinsic::X86Roundps
            },
            &[nc, rc],
        );
        nc
    } else {
        let sign_mask = if scalar {
            x86_get_scalar_i64(ctx, i64::MIN)
        } else {
            x86_get_all_elements_i64(ctx, i64::MIN)
        };
        let sign_mask = ctx.emit_intrinsic(Intrinsic::X86Pand, &[sign_mask, n]);
        let value_mask = if scalar {
            x86_get_scalar_i64(ctx, 0x3FDFFFFFFFFFFFFFi64)
        } else {
            x86_get_all_elements_i64(ctx, 0x3FDFFFFFFFFFFFFFi64)
        };
        let value_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[value_mask, sign_mask]);
        let mut nc = ctx.emit_intrinsic(
            if scalar {
                Intrinsic::X86Addsd
            } else {
                Intrinsic::X86Addpd
            },
            &[n, value_mask],
        );
        nc = ctx.emit_intrinsic(
            if scalar {
                Intrinsic::X86Roundsd
            } else {
                Intrinsic::X86Roundpd
            },
            &[nc, rc],
        );
        nc
    }
}

// EmitSse41ScalarRoundOpF (from SimdArithmetic private)
pub fn emit_sse41_scalar_round_op_f(ctx: &mut ArmEmitterCtx, round_mode: i32) {
    let op = decode_simd_op(ctx);
    let n = get_vec(ctx, op.rn);
    let res = if round_mode != 4 {
        // ToNearestAway=4 needs custom handling
        let inst = if (op.size & 1) != 0 {
            Intrinsic::X86Roundsd
        } else {
            Intrinsic::X86Roundss
        };
        let rc = ctx
            .ctx
            .operands
            .constant_i32(x86_get_round_control(round_mode));
        ctx.emit_intrinsic(inst, &[n, rc])
    } else {
        emit_sse41_round_to_nearest_ties_away_op_f(ctx, n, true)
    };
    let res = if (op.size & 1) != 0 {
        ctx.vector_zero_upper64(res)
    } else {
        ctx.vector_zero_upper96(res)
    };
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_sse41_vector_round_op_f(ctx: &mut ArmEmitterCtx, round_mode: i32) {
    let op = decode_simd_op(ctx);
    let n = get_vec(ctx, op.rn);
    let res = if round_mode != 4 {
        // ToNearestAway=4
        let inst = if (op.size & 1) != 0 {
            Intrinsic::X86Roundpd
        } else {
            Intrinsic::X86Roundps
        };
        let rc = ctx
            .ctx
            .operands
            .constant_i32(x86_get_round_control(round_mode));
        ctx.emit_intrinsic(inst, &[n, rc])
    } else {
        emit_sse41_round_to_nearest_ties_away_op_f(ctx, n, false)
    };
    let res = if op.reg_size.bits() == 64 {
        ctx.vector_zero_upper64(res)
    } else {
        res
    };
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// EmitSse41Round32Exp8OpF (from SimdArithmetic private)
pub fn emit_sse41_round32_exp8_op_f(
    ctx: &mut ArmEmitterCtx,
    value: OperandIndex,
    scalar: bool,
) -> OperandIndex {
    let round_mask = if scalar {
        x86_get_scalar_i32(ctx, 0x4000)
    } else {
        x86_get_all_elements_i32(ctx, 0x4000)
    };
    let trunc_mask = if scalar {
        x86_get_scalar_i32(ctx, -32768i32)
    } else {
        x86_get_all_elements_i32(ctx, -32768i32)
    };
    let exp_mask = if scalar {
        x86_get_scalar_i32(ctx, 0x7F800000)
    } else {
        x86_get_all_elements_i32(ctx, 0x7F800000)
    };
    let o_value = value;
    let masked = ctx.emit_intrinsic(Intrinsic::X86Pand, &[value, exp_mask]);
    let is_nan_inf = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqd, &[masked, exp_mask]);
    let mut v = ctx.emit_intrinsic(Intrinsic::X86Paddd, &[value, round_mask]);
    v = ctx.emit_intrinsic(Intrinsic::X86Pand, &[v, trunc_mask]);
    ctx.emit_intrinsic(Intrinsic::X86Blendvps, &[v, o_value, is_nan_inf])
}

// EmitSse41RecipStepSelectOpF (from SimdArithmetic private)
pub fn emit_sse41_recip_step_select_op_f(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    m: OperandIndex,
    res: OperandIndex,
    mask: OperandIndex,
    scalar: bool,
    size_f: i32,
) -> OperandIndex {
    let zero = ctx.vector_zero();
    if size_f == 0 {
        let exp_val = (0x7F800000u32 << 1) as i32;
        let exp_mask = if scalar {
            x86_get_scalar_i32(ctx, exp_val)
        } else {
            x86_get_all_elements_i32(ctx, exp_val)
        };
        let c1 = ctx.ctx.operands.constant_i32(1);
        let n2 = ctx.emit_intrinsic(Intrinsic::X86Pslld, &[n, c1]);
        let m2 = ctx.emit_intrinsic(Intrinsic::X86Pslld, &[m, c1]);
        let n_zero = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqd, &[n2, zero]);
        let m_zero = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqd, &[m2, zero]);
        let n_inf = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqd, &[n2, exp_mask]);
        let m_inf = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqd, &[m2, exp_mask]);
        let nm_zero = ctx.emit_intrinsic(Intrinsic::X86Por, &[n_zero, m_zero]);
        let nm_inf = ctx.emit_intrinsic(Intrinsic::X86Por, &[n_inf, m_inf]);
        let nm_zero_inf = ctx.emit_intrinsic(Intrinsic::X86Pand, &[nm_zero, nm_inf]);
        ctx.emit_intrinsic(Intrinsic::X86Blendvps, &[res, mask, nm_zero_inf])
    } else {
        let exp_mask = if scalar {
            x86_get_scalar_i64(ctx, (0x7FF0000000000000u64 << 1) as i64)
        } else {
            x86_get_all_elements_i64(ctx, (0x7FF0000000000000u64 << 1) as i64)
        };
        let c1 = ctx.ctx.operands.constant_i32(1);
        let n2 = ctx.emit_intrinsic(Intrinsic::X86Psllq, &[n, c1]);
        let m2 = ctx.emit_intrinsic(Intrinsic::X86Psllq, &[m, c1]);
        let n_zero = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqq, &[n2, zero]);
        let m_zero = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqq, &[m2, zero]);
        let n_inf = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqq, &[n2, exp_mask]);
        let m_inf = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqq, &[m2, exp_mask]);
        let nm_zero = ctx.emit_intrinsic(Intrinsic::X86Por, &[n_zero, m_zero]);
        let nm_inf = ctx.emit_intrinsic(Intrinsic::X86Por, &[n_inf, m_inf]);
        let nm_zero_inf = ctx.emit_intrinsic(Intrinsic::X86Pand, &[nm_zero, nm_inf]);
        ctx.emit_intrinsic(Intrinsic::X86Blendvpd, &[res, mask, nm_zero_inf])
    }
}

// EmitSse2VectorIsNaNOpF (from SimdArithmetic)
pub fn emit_sse2_vector_is_nan_op_f(
    ctx: &mut ArmEmitterCtx,
    op_f: OperandIndex,
) -> (OperandIndex, OperandIndex) {
    let op_size = decode_simd_op(ctx).size;
    if (op_size & 1) == 0 {
        let q_bit = 22;
        let q_mask = x86_get_all_elements_i32(ctx, 1 << q_bit);
        let cmpuoq = ctx
            .ctx
            .operands
            .constant_i32(CmpCondition::UnorderedQ as i32);
        let mask1 = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[op_f, op_f, cmpuoq]);
        let mask2 = ctx.emit_intrinsic(Intrinsic::X86Pand, &[op_f, q_mask]);
        let cmpeq = ctx.ctx.operands.constant_i32(CmpCondition::Equal as i32);
        let mask2 = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[mask2, q_mask, cmpeq]);
        let q_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Andps, &[mask2, mask1]);
        let s_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Andnps, &[mask2, mask1]);
        (q_nan_mask, s_nan_mask)
    } else {
        let q_bit = 51;
        let q_mask = x86_get_all_elements_i64(ctx, 1i64 << q_bit);
        let cmpuoq = ctx
            .ctx
            .operands
            .constant_i32(CmpCondition::UnorderedQ as i32);
        let mask1 = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[op_f, op_f, cmpuoq]);
        let mask2 = ctx.emit_intrinsic(Intrinsic::X86Pand, &[op_f, q_mask]);
        let cmpeq = ctx.ctx.operands.constant_i32(CmpCondition::Equal as i32);
        let mask2 = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[mask2, q_mask, cmpeq]);
        let q_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Andpd, &[mask2, mask1]);
        let s_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Andnpd, &[mask2, mask1]);
        (q_nan_mask, s_nan_mask)
    }
}

// EmitSse2VectorMaxMinOpF (from SimdArithmetic private)
pub fn emit_sse2_vector_max_min_op_f(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    m: OperandIndex,
    is_max: bool,
) -> OperandIndex {
    // C# uses sign-bit mask (-0f = 0x80000000, -0d = 0x8000000000000000) to
    // work around x86 max/min returning -0 instead of +0 (ARM behavior differs).
    let op_size = decode_simd_op(ctx).size;
    if (op_size & 1) == 0 {
        let mask = x86_get_all_elements_i32(ctx, i32::MIN);
        let res = ctx.emit_intrinsic(
            if is_max {
                Intrinsic::X86Maxps
            } else {
                Intrinsic::X86Minps
            },
            &[n, m],
        );
        let res = ctx.emit_intrinsic(Intrinsic::X86Andnps, &[mask, res]);
        let res_sign = if is_max {
            ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, m])
        } else {
            ctx.emit_intrinsic(Intrinsic::X86Por, &[n, m])
        };
        let res_sign = ctx.emit_intrinsic(Intrinsic::X86Andps, &[mask, res_sign]);
        ctx.emit_intrinsic(Intrinsic::X86Por, &[res, res_sign])
    } else {
        let mask = x86_get_all_elements_i64(ctx, i64::MIN);
        let res = ctx.emit_intrinsic(
            if is_max {
                Intrinsic::X86Maxpd
            } else {
                Intrinsic::X86Minpd
            },
            &[n, m],
        );
        let res = ctx.emit_intrinsic(Intrinsic::X86Andnpd, &[mask, res]);
        let res_sign = if is_max {
            ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, m])
        } else {
            ctx.emit_intrinsic(Intrinsic::X86Por, &[n, m])
        };
        let res_sign = ctx.emit_intrinsic(Intrinsic::X86Andpd, &[mask, res_sign]);
        ctx.emit_intrinsic(Intrinsic::X86Por, &[res, res_sign])
    }
}

// EmitSse41ProcessNaNsOpF (from SimdArithmetic)
pub fn emit_sse41_process_nans_op_f<
    F: Fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
>(
    ctx: &mut ArmEmitterCtx,
    emit: F,
    scalar: bool,
    n_opt: Option<OperandIndex>,
    m_opt: Option<OperandIndex>,
) -> OperandIndex {
    // C# only casts CurrOp to OpCodeSimdReg when n/m are defaults  for
    // pairwise/across callers CurrOp is a plain OpCodeSimd and n/m are
    // always provided. Size comes from IOpCodeSimd.
    let op = decode_simd_op(ctx);
    // C#: nCopy = n == default ? context.Copy(GetVec(op.Rn)) : n.
    let n = match n_opt {
        Some(n) => n,
        None => {
            let rn_idx = decode_simd_reg_op(ctx).rn;
            let rn = get_vec(ctx, rn_idx);
            ctx.copy(rn)
        }
    };
    let m = match m_opt {
        Some(m) => m,
        None => {
            let rm_idx = decode_simd_reg_op(ctx).rm;
            let rm = get_vec(ctx, rm_idx);
            ctx.copy(rm)
        }
    };
    let (n_q_nan_mask, n_s_nan_mask) = emit_sse2_vector_is_nan_op_f(ctx, n);
    let (_, m_s_nan_mask) = emit_sse2_vector_is_nan_op_f(ctx, m);
    let size_f = op.size & 1;
    if size_f == 0 {
        let q_mask = if scalar {
            x86_get_scalar_i32(ctx, 1 << 22)
        } else {
            x86_get_all_elements_i32(ctx, 1 << 22)
        };
        let res_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[m_s_nan_mask, n_q_nan_mask]);
        let res_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[res_nan_mask, n_s_nan_mask]);
        let res_nan = ctx.emit_intrinsic(Intrinsic::X86Blendvps, &[m, n, res_nan_mask]);
        let res_nan = ctx.emit_intrinsic(Intrinsic::X86Por, &[res_nan, q_mask]);
        let ordq = ctx.ctx.operands.constant_i32(CmpCondition::OrderedQ as i32);
        let res_mask = ctx.emit_intrinsic(Intrinsic::X86Cmpps, &[n, m, ordq]);
        let emit_result = emit(ctx, n, m);
        let result = ctx.emit_intrinsic(Intrinsic::X86Blendvps, &[res_nan, emit_result, res_mask]);
        if n_opt.is_some() || m_opt.is_some() {
            return result;
        }
        let result = if scalar {
            ctx.vector_zero_upper96(result)
        } else if op.reg_size.bits() == 64 {
            ctx.vector_zero_upper64(result)
        } else {
            result
        };
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, result);
    } else {
        let q_mask = if scalar {
            x86_get_scalar_i64(ctx, 1i64 << 51)
        } else {
            x86_get_all_elements_i64(ctx, 1i64 << 51)
        };
        let res_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[m_s_nan_mask, n_q_nan_mask]);
        let res_nan_mask = ctx.emit_intrinsic(Intrinsic::X86Por, &[res_nan_mask, n_s_nan_mask]);
        let res_nan = ctx.emit_intrinsic(Intrinsic::X86Blendvpd, &[m, n, res_nan_mask]);
        let res_nan = ctx.emit_intrinsic(Intrinsic::X86Por, &[res_nan, q_mask]);
        let ordq = ctx.ctx.operands.constant_i32(CmpCondition::OrderedQ as i32);
        let res_mask = ctx.emit_intrinsic(Intrinsic::X86Cmppd, &[n, m, ordq]);
        let emit_result = emit(ctx, n, m);
        let result = ctx.emit_intrinsic(Intrinsic::X86Blendvpd, &[res_nan, emit_result, res_mask]);
        if n_opt.is_some() || m_opt.is_some() {
            return result;
        }
        let result = if scalar {
            ctx.vector_zero_upper64(result)
        } else {
            result
        };
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, result);
    }
    OperandIndex::from(0) // unreachable, but needed for type
}

// EmitSse41MaxMinNumOpF (from SimdArithmetic private)
pub fn emit_sse41_max_min_num_op_f(
    ctx: &mut ArmEmitterCtx,
    is_max_num: bool,
    scalar: bool,
    n_opt: Option<OperandIndex>,
    m_opt: Option<OperandIndex>,
) -> OperandIndex {
    // See emit_sse41_process_nans_op_f: OpCodeSimdReg fields only when
    // n/m are not provided.
    let op = decode_simd_op(ctx);
    // C#: nCopy = n == default ? context.Copy(GetVec(op.Rn)) : n.
    let n = match n_opt {
        Some(n) => n,
        None => {
            let rn_idx = decode_simd_reg_op(ctx).rn;
            let rn = get_vec(ctx, rn_idx);
            ctx.copy(rn)
        }
    };
    let m = match m_opt {
        Some(m) => m,
        None => {
            let rm_idx = decode_simd_reg_op(ctx).rm;
            let rm = get_vec(ctx, rm_idx);
            ctx.copy(rm)
        }
    };
    let (n_q_nan_mask, _) = emit_sse2_vector_is_nan_op_f(ctx, n);
    let (m_q_nan_mask, _) = emit_sse2_vector_is_nan_op_f(ctx, m);
    let size_f = op.size & 1;
    if size_f == 0 {
        let neg_inf_mask = if scalar {
            x86_get_scalar_f32(
                ctx,
                if is_max_num {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                },
            )
        } else {
            x86_get_all_elements_f32(
                ctx,
                if is_max_num {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                },
            )
        };
        let n_mask = ctx.emit_intrinsic(Intrinsic::X86Andnps, &[m_q_nan_mask, n_q_nan_mask]);
        let m_mask = ctx.emit_intrinsic(Intrinsic::X86Andnps, &[n_q_nan_mask, m_q_nan_mask]);
        let n_copy = ctx.emit_intrinsic(Intrinsic::X86Blendvps, &[n, neg_inf_mask, n_mask]);
        let m_copy = ctx.emit_intrinsic(Intrinsic::X86Blendvps, &[m, neg_inf_mask, m_mask]);
        let res = emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, is_max_num),
            scalar,
            Some(n_copy),
            Some(m_copy),
        );
        if n_opt.is_some() || m_opt.is_some() {
            return res;
        }
        let res = if scalar {
            ctx.vector_zero_upper96(res)
        } else if op.reg_size.bits() == 64 {
            ctx.vector_zero_upper64(res)
        } else {
            res
        };
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let neg_inf_mask = if scalar {
            x86_get_scalar_f64(
                ctx,
                if is_max_num {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                },
            )
        } else {
            x86_get_all_elements_f64(
                ctx,
                if is_max_num {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                },
            )
        };
        let n_mask = ctx.emit_intrinsic(Intrinsic::X86Andnpd, &[m_q_nan_mask, n_q_nan_mask]);
        let m_mask = ctx.emit_intrinsic(Intrinsic::X86Andnpd, &[n_q_nan_mask, m_q_nan_mask]);
        let n_copy = ctx.emit_intrinsic(Intrinsic::X86Blendvpd, &[n, neg_inf_mask, n_mask]);
        let m_copy = ctx.emit_intrinsic(Intrinsic::X86Blendvpd, &[m, neg_inf_mask, m_mask]);
        let res = emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, is_max_num),
            scalar,
            Some(n_copy),
            Some(m_copy),
        );
        if n_opt.is_some() || m_opt.is_some() {
            return res;
        }
        let res = if scalar {
            ctx.vector_zero_upper64(res)
        } else {
            res
        };
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
    OperandIndex::from(0)
}

// EmitSse41VectorMul_AddSub (from SimdArithmetic private)
pub fn emit_sse41_vector_mul_add_sub(ctx: &mut ArmEmitterCtx, add_sub: i32) {
    // add_sub: 0=None, 1=Add, 2=Subtract
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let res = if op.size == 0 {
        let c8 = ctx.ctx.operands.constant_i32(8);
        let ns8 = ctx.emit_intrinsic(Intrinsic::X86Psrlw, &[n, c8]);
        let ms8 = ctx.emit_intrinsic(Intrinsic::X86Psrlw, &[m, c8]);
        let mut r = ctx.emit_intrinsic(Intrinsic::X86Pmullw, &[ns8, ms8]);
        r = ctx.emit_intrinsic(Intrinsic::X86Psllw, &[r, c8]);
        let r2 = ctx.emit_intrinsic(Intrinsic::X86Pmullw, &[n, m]);
        let mask = x86_get_all_elements_i32(ctx, 0x00FF00FF);
        ctx.emit_intrinsic(Intrinsic::X86Pblendvb, &[r, r2, mask])
    } else if op.size == 1 {
        ctx.emit_intrinsic(Intrinsic::X86Pmullw, &[n, m])
    } else {
        ctx.emit_intrinsic(Intrinsic::X86Pmulld, &[n, m])
    };
    let d = get_vec(ctx, op.rd);
    let res = if add_sub == 1 {
        let add_inst = X86_PADD_INSTRUCTION[op.size as usize];
        ctx.emit_intrinsic(add_inst, &[d, res])
    } else if add_sub == 2 {
        let sub_inst = X86_PSUB_INSTRUCTION[op.size as usize];
        ctx.emit_intrinsic(sub_inst, &[d, res])
    } else {
        res
    };
    let res = if op.reg_size.bits() == 64 {
        ctx.vector_zero_upper64(res)
    } else {
        res
    };
    ctx.copy_to(d, res);
}

// EmitSse41VectorSabdOp (from SimdArithmetic private)
pub fn emit_sse41_vector_sabd_op(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    m: OperandIndex,
    is_long: bool,
) {
    let op = decode_simd_reg_op(ctx);
    let size = if is_long { op.size + 1 } else { op.size };
    let cmpgt_inst = X86_PCMPGT_INSTRUCTION[size as usize];
    let cmp_mask = ctx.emit_intrinsic(cmpgt_inst, &[n, m]);
    let sub_inst = X86_PSUB_INSTRUCTION[size as usize];
    let mut res = ctx.emit_intrinsic(sub_inst, &[n, m]);
    res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[cmp_mask, res]);
    let mut res2 = ctx.emit_intrinsic(sub_inst, &[m, n]);
    res2 = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[cmp_mask, res2]);
    let res = ctx.emit_intrinsic(Intrinsic::X86Por, &[res, res2]);
    let rd = get_vec(ctx, op.rd);
    if !is_long && op.reg_size.bits() == 64 {
        let z = ctx.vector_zero_upper64(res);
        ctx.copy_to(rd, z);
    } else {
        ctx.copy_to(rd, res);
    }
}

pub fn emit_sse41_vector_uabd_op(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    m: OperandIndex,
    is_long: bool,
) {
    let op = decode_simd_reg_op(ctx);
    let size = if is_long { op.size + 1 } else { op.size };
    let max_inst = X86_PMAXU_INSTRUCTION[size as usize];
    let max_val = ctx.emit_intrinsic(max_inst, &[m, n]);
    let cmpeq_inst = X86_PCMPEQ_INSTRUCTION[size as usize];
    let mut cmp_mask = ctx.emit_intrinsic(cmpeq_inst, &[max_val, m]);
    let ones_mask = x86_get_all_elements_i64(ctx, -1i64);
    cmp_mask = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[cmp_mask, ones_mask]);
    let sub_inst = X86_PSUB_INSTRUCTION[size as usize];
    let mut res = ctx.emit_intrinsic(sub_inst, &[n, m]);
    let mut res2 = ctx.emit_intrinsic(sub_inst, &[m, n]);
    res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[cmp_mask, res]);
    res2 = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[cmp_mask, res2]);
    let res = ctx.emit_intrinsic(Intrinsic::X86Por, &[res, res2]);
    let rd = get_vec(ctx, op.rd);
    if !is_long && op.reg_size.bits() == 64 {
        let z = ctx.vector_zero_upper64(res);
        ctx.copy_to(rd, z);
    } else {
        ctx.copy_to(rd, res);
    }
}

// EmitSse2Sll_128 (from SimdArithmetic private)
pub fn emit_sse2_sll_128(ctx: &mut ArmEmitterCtx, op: OperandIndex, shift: i32) -> OperandIndex {
    if shift == 0 {
        return op;
    }
    let c8 = ctx.ctx.operands.constant_i32(8);
    let high = ctx.emit_intrinsic(Intrinsic::X86Pslldq, &[op, c8]);
    let c64_minus_shift = ctx.ctx.operands.constant_i32(64 - shift);
    let high = ctx.emit_intrinsic(Intrinsic::X86Psrlq, &[high, c64_minus_shift]);
    let cshift = ctx.ctx.operands.constant_i32(shift);
    let low = ctx.emit_intrinsic(Intrinsic::X86Psllq, &[op, cshift]);
    ctx.emit_intrinsic(Intrinsic::X86Por, &[high, low])
}

// EmitPolynomialMultiply (C# InstEmitSimdHelper32, used by A64 Pmull_V)
// result ^= op2 * (op1 & (1 << i))  the multiply yields op2 << i when the
// bit is set, 0 otherwise.

pub fn emit_polynomial_multiply(
    ctx: &mut ArmEmitterCtx,
    mut op1: OperandIndex,
    mut op2: OperandIndex,
    e_size: i32,
) -> OperandIndex {
    debug_assert!(e_size <= 32);

    let mut result = if e_size == 32 {
        ctx.ctx.operands.constant_i64(0)
    } else {
        ctx.ctx.operands.constant_i32(0)
    };

    if e_size == 32 {
        op1 = ctx.zero_extend32(OperandType::I64, op1);
        op2 = ctx.zero_extend32(OperandType::I64, op2);
    }

    let op1_typ = ctx.ctx.operands.get(op1).typ;

    for i in 0..e_size {
        let bit = ctx.ctx.operands.constant_int(op1_typ, 1i64 << i);
        let mask = ctx.bitwise_and(op1, bit);
        let term = ctx.multiply(op2, mask);
        result = ctx.bitwise_xor(result, term);
    }

    result
}

// EmitSse2VectorPairwiseOpF (C# InstEmitSimdHelper)
pub fn emit_sse2_vector_pairwise_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_reg_op(ctx);
    let rn = get_vec(ctx, op.rn);
    let n_copy = ctx.copy(rn);
    let rm = get_vec(ctx, op.rm);
    let m_copy = ctx.copy(rm);
    let size_f = op.size & 1;
    if size_f == 0 {
        if op.reg_size.bits() == 64 {
            let unpck = ctx.emit_intrinsic(Intrinsic::X86Unpcklps, &[n_copy, m_copy]);
            let zero = ctx.vector_zero();
            let part0 = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[unpck, zero]);
            let part1 = ctx.emit_intrinsic(Intrinsic::X86Movhlps, &[zero, unpck]);
            let result = emit(ctx, part0, part1);
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, result);
        } else {
            const SM0: i32 = 2 << 6 | 0 << 4 | 2 << 2 | 0 << 0;
            const SM1: i32 = 3 << 6 | 1 << 4 | 3 << 2 | 1 << 0;
            let sm0 = ctx.ctx.operands.constant_i32(SM0);
            let part0 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[n_copy, m_copy, sm0]);
            let sm1 = ctx.ctx.operands.constant_i32(SM1);
            let part1 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[n_copy, m_copy, sm1]);
            let result = emit(ctx, part0, part1);
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, result);
        }
    } else {
        let part0 = ctx.emit_intrinsic(Intrinsic::X86Unpcklpd, &[n_copy, m_copy]);
        let part1 = ctx.emit_intrinsic(Intrinsic::X86Unpckhpd, &[n_copy, m_copy]);
        let result = emit(ctx, part0, part1);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, result);
    }
}

// EmitSse2VectorAcrossVectorOpF (C# InstEmitSimdHelper)
// C# asserts (op.Size & 1) == 0 && RegisterSize == Simd128.

pub fn emit_sse2_vector_across_vector_op_f(
    ctx: &mut ArmEmitterCtx,
    emit: fn(&mut ArmEmitterCtx, OperandIndex, OperandIndex) -> OperandIndex,
) {
    let op = decode_simd_op(ctx);
    debug_assert!((op.size & 1) == 0 && op.reg_size.bits() == 128);

    const SM0: i32 = 0 << 6 | 0 << 4 | 0 << 2 | 0 << 0;
    const SM1: i32 = 1 << 6 | 1 << 4 | 1 << 2 | 1 << 0;
    const SM2: i32 = 2 << 6 | 2 << 4 | 2 << 2 | 2 << 0;
    const SM3: i32 = 3 << 6 | 3 << 4 | 3 << 2 | 3 << 0;

    let rn = get_vec(ctx, op.rn);
    let n_copy = ctx.copy(rn);

    let sm0 = ctx.ctx.operands.constant_i32(SM0);
    let part0 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[n_copy, n_copy, sm0]);
    let sm1 = ctx.ctx.operands.constant_i32(SM1);
    let part1 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[n_copy, n_copy, sm1]);
    let sm2 = ctx.ctx.operands.constant_i32(SM2);
    let part2 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[n_copy, n_copy, sm2]);
    let sm3 = ctx.ctx.operands.constant_i32(SM3);
    let part3 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[n_copy, n_copy, sm3]);

    let res01 = emit(ctx, part0, part1);
    let res23 = emit(ctx, part2, part3);
    let res = emit(ctx, res01, res23);

    let res = ctx.vector_zero_upper96(res);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}
