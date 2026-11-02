// Port of Instructions/InstEmitSimdLogical.cs (A64). The UseAdvSimd
// arms are excluded (x86-64 host only); SSE2/SSSE3/AVX512/GFNI paths and
// the scalar fallbacks are ported as-is.

use crate::ir::operand::OperandIndex;
use crate::ir::types::{Intrinsic, OperandType};
use crate::optimizations;
use crate::registers::VecRegisterIndex;

use crate::decoder::RegisterSize;
use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::helpers::get_vec;
use crate::instructions::simd_helper::{
    decode_simd_imm_op, decode_simd_op, decode_simd_reg_op, emit_vector_binary_op_zx,
    emit_vector_extract_zx, emit_vector_imm_binary_op, emit_vector_insert,
    emit_vector_ternary_op_zx, emit_vector_unary_op_zx, simd_bytes_count, x86_get_all_elements_i16,
    x86_get_all_elements_i32, x86_get_all_elements_i64, x86_get_scalar_i64,
};

fn is_simd64(reg_size: RegisterSize) -> bool {
    reg_size == RegisterSize::Simd64
}

pub fn emit_and_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let op = decode_simd_reg_op(ctx);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, m]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_binary_op_zx(ctx, |ctx, op1, op2| ctx.bitwise_and(op1, op2));
    }
}

pub fn emit_bic_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let op = decode_simd_reg_op(ctx);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[m, n]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_binary_op_zx(ctx, |ctx, op1, op2| {
            let not2 = ctx.bitwise_not(op2);
            ctx.bitwise_and(op1, not2)
        });
    }
}

pub fn emit_bic_vi(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let op = decode_simd_imm_op(ctx);

        let e_size = 8 << op.size;

        let d = get_vec(ctx, op.rd);
        let imm = match e_size {
            16 => x86_get_all_elements_i16(ctx, !(op.immediate as i16)),
            32 => x86_get_all_elements_i32(ctx, !(op.immediate as i32)),
            _ => panic!("Invalid element size {e_size}."),
        };

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[d, imm]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_imm_binary_op(ctx, |ctx, op1, op2| {
            let not2 = ctx.bitwise_not(op2);
            ctx.bitwise_and(op1, not2)
        });
    }
}

pub fn emit_bif_v(ctx: &mut ArmEmitterCtx) {
    emit_bif_bit(ctx, true);
}

pub fn emit_bit_v(ctx: &mut ArmEmitterCtx) {
    emit_bif_bit(ctx, false);
}

fn emit_bif_bit(ctx: &mut ArmEmitterCtx, not_rm: bool) {
    let op = decode_simd_reg_op(ctx);

    if optimizations::use_sse2() {
        let d = get_vec(ctx, op.rd);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, d]);

        if not_rm {
            res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[m, res]);
        } else {
            res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[m, res]);
        }

        res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[d, res]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        ctx.copy_to(d, res);
    } else {
        let mut res = ctx.vector_zero();

        let elems = if op.reg_size == RegisterSize::Simd128 {
            2
        } else {
            1
        };

        for index in 0..elems {
            let d = emit_vector_extract_zx(ctx, op.rd, index, 3);
            let n = emit_vector_extract_zx(ctx, op.rn, index, 3);
            let mut m = emit_vector_extract_zx(ctx, op.rm, index, 3);

            if not_rm {
                m = ctx.bitwise_not(m);
            }

            let mut e = ctx.bitwise_xor(d, n);

            e = ctx.bitwise_and(e, m);
            e = ctx.bitwise_xor(e, d);

            res = emit_vector_insert(ctx, res, e, index, 3);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

pub fn emit_bsl_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let op = decode_simd_reg_op(ctx);

        let d = get_vec(ctx, op.rd);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);

        res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[res, d]);
        res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[res, m]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        ctx.copy_to(d, res);
    } else {
        emit_vector_ternary_op_zx(ctx, |ctx, op1, op2, op3| {
            let x = ctx.bitwise_xor(op2, op3);
            let a = ctx.bitwise_and(op1, x);
            ctx.bitwise_xor(a, op3)
        });
    }
}

pub fn emit_eor_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let op = decode_simd_reg_op(ctx);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_binary_op_zx(ctx, |ctx, op1, op2| ctx.bitwise_xor(op1, op2));
    }
}

pub fn emit_not_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_avx512_ortho() {
        let op = decode_simd_op(ctx);
        let n = get_vec(ctx, op.rn);

        let tern = ctx.ctx.operands.constant_i32(!0b10101010);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Vpternlogd, &[n, n, tern]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else if optimizations::use_sse2() {
        let op = decode_simd_op(ctx);
        let n = get_vec(ctx, op.rn);

        let mask = x86_get_all_elements_i64(ctx, -1);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[n, mask]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_unary_op_zx(ctx, |ctx, op1| ctx.bitwise_not(op1));
    }
}

pub fn emit_orn_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_avx512_ortho() {
        let op = decode_simd_reg_op(ctx);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let tern = ctx.ctx.operands.constant_i32(0b11001100 | !0b10101010);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Vpternlogd, &[n, m, tern]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else if optimizations::use_sse2() {
        let op = decode_simd_reg_op(ctx);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let mask = x86_get_all_elements_i64(ctx, -1);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[m, mask]);

        res = ctx.emit_intrinsic(Intrinsic::X86Por, &[res, n]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_binary_op_zx(ctx, |ctx, op1, op2| {
            let not2 = ctx.bitwise_not(op2);
            ctx.bitwise_or(op1, not2)
        });
    }
}

pub fn emit_orr_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let op = decode_simd_reg_op(ctx);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Por, &[n, m]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_binary_op_zx(ctx, |ctx, op1, op2| ctx.bitwise_or(op1, op2));
    }
}

pub fn emit_orr_vi(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_sse2() {
        let op = decode_simd_imm_op(ctx);

        let e_size = 8 << op.size;

        let d = get_vec(ctx, op.rd);
        let imm = match e_size {
            16 => x86_get_all_elements_i16(ctx, op.immediate as i16),
            32 => x86_get_all_elements_i32(ctx, op.immediate as i32),
            _ => panic!("Invalid element size {e_size}."),
        };

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Por, &[d, imm]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_imm_binary_op(ctx, |ctx, op1, op2| ctx.bitwise_or(op1, op2));
    }
}

pub fn emit_rbit_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);

    if optimizations::use_gfni() {
        const BIT_MATRIX: i64 = (0b10000000i64 << 56)
            | (0b01000000i64 << 48)
            | (0b00100000i64 << 40)
            | (0b00010000i64 << 32)
            | (0b00001000i64 << 24)
            | (0b00000100i64 << 16)
            | (0b00000010i64 << 8)
            | 0b00000001i64;

        let v_bit_matrix = x86_get_all_elements_i64(ctx, BIT_MATRIX);

        let n = get_vec(ctx, op.rn);
        let zero = ctx.ctx.operands.constant_i32(0);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Gf2p8affineqb, &[n, v_bit_matrix, zero]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let mut res = ctx.vector_zero();
        let elems = if op.reg_size == RegisterSize::Simd128 {
            16
        } else {
            8
        };

        for index in 0..elems {
            let ne = emit_vector_extract_zx(ctx, op.rn, index, 0);

            let de = emit_reverse_bits8_op(ctx, ne);

            res = emit_vector_insert(ctx, res, de, index, 0);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

fn emit_reverse_bits8_op(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    debug_assert_eq!(ctx.ctx.operands.get(op).typ, OperandType::I64);

    let c_aa = ctx.ctx.operands.constant_i64(0xaa);
    let c_55 = ctx.ctx.operands.constant_i64(0x55);
    let one = ctx.ctx.operands.constant_i32(1);
    let a = ctx.bitwise_and(op, c_aa);
    let a = ctx.shift_right_ui(a, one);
    let one = ctx.ctx.operands.constant_i32(1);
    let b = ctx.bitwise_and(op, c_55);
    let b = ctx.shift_left(b, one);
    let mut val = ctx.bitwise_or(a, b);

    let c_cc = ctx.ctx.operands.constant_i64(0xcc);
    let c_33 = ctx.ctx.operands.constant_i64(0x33);
    let two = ctx.ctx.operands.constant_i32(2);
    let a = ctx.bitwise_and(val, c_cc);
    let a = ctx.shift_right_ui(a, two);
    let two = ctx.ctx.operands.constant_i32(2);
    let b = ctx.bitwise_and(val, c_33);
    let b = ctx.shift_left(b, two);
    val = ctx.bitwise_or(a, b);

    let four = ctx.ctx.operands.constant_i32(4);
    let a = ctx.shift_right_ui(val, four);
    let c_0f = ctx.ctx.operands.constant_i64(0x0f);
    let four = ctx.ctx.operands.constant_i32(4);
    let b = ctx.bitwise_and(val, c_0f);
    let b = ctx.shift_left(b, four);
    ctx.bitwise_or(a, b)
}

pub fn emit_rev16_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_ssse3() {
        let op = decode_simd_op(ctx);
        let n = get_vec(ctx, op.rn);

        const MASK_E0: i64 = 6 << 56 | 7 << 48 | 4 << 40 | 5 << 32 | 2 << 24 | 3 << 16 | 1;
        const MASK_E1: i64 =
            14 << 56 | 15 << 48 | 12 << 40 | 13 << 32 | 10 << 24 | 11 << 16 | 8 << 8 | 9;

        let mut mask = x86_get_scalar_i64(ctx, MASK_E0);

        let e1 = ctx.ctx.operands.constant_i64(MASK_E1);
        mask = emit_vector_insert(ctx, mask, e1, 1, 3);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n, mask]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_rev_v(ctx, 1);
    }
}

pub fn emit_rev32_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_ssse3() {
        let op = decode_simd_op(ctx);
        let n = get_vec(ctx, op.rn);

        let (mask_e0, mask_e1): (i64, i64) = if op.size == 0 {
            (
                4 << 56 | 5 << 48 | 6 << 40 | 7 << 32 | 1 << 16 | 2 << 8 | 3,
                12 << 56 | 13 << 48 | 14 << 40 | 15 << 32 | 8 << 24 | 9 << 16 | 10 << 8 | 11,
            )
        } else {
            (
                5 << 56 | 4 << 48 | 7 << 40 | 6 << 32 | 1 << 24 | 3 << 8 | 2,
                13 << 56 | 12 << 48 | 15 << 40 | 14 << 32 | 9 << 24 | 8 << 16 | 11 << 8 | 10,
            )
        };

        let mut mask = x86_get_scalar_i64(ctx, mask_e0);
        let e1 = ctx.ctx.operands.constant_i64(mask_e1);
        mask = emit_vector_insert(ctx, mask, e1, 1, 3);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n, mask]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_rev_v(ctx, 2);
    }
}

pub fn emit_rev64_v(ctx: &mut ArmEmitterCtx) {
    if optimizations::use_ssse3() {
        let op = decode_simd_op(ctx);
        let n = get_vec(ctx, op.rn);

        let (mask_e0, mask_e1): (i64, i64) = if op.size == 0 {
            (
                1 << 48 | 2 << 40 | 3 << 32 | 4 << 24 | 5 << 16 | 6 << 8 | 7,
                8 << 56 | 9 << 48 | 10 << 40 | 11 << 32 | 12 << 24 | 13 << 16 | 14 << 8 | 15,
            )
        } else if op.size == 1 {
            (
                1 << 56 | 3 << 40 | 2 << 32 | 5 << 24 | 4 << 16 | 7 << 8 | 6,
                9 << 56 | 8 << 48 | 11 << 40 | 10 << 32 | 13 << 24 | 12 << 16 | 15 << 8 | 14,
            )
        } else {
            (
                3 << 56 | 2 << 48 | 1 << 40 | 7 << 24 | 6 << 16 | 5 << 8 | 4,
                11 << 56 | 10 << 48 | 9 << 40 | 8 << 32 | 15 << 24 | 14 << 16 | 13 << 8 | 12,
            )
        };

        let mut mask = x86_get_scalar_i64(ctx, mask_e0);
        let e1 = ctx.ctx.operands.constant_i64(mask_e1);
        mask = emit_vector_insert(ctx, mask, e1, 1, 3);

        let mut res = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[n, mask]);

        if is_simd64(op.reg_size) {
            res = ctx.vector_zero_upper64(res);
        }

        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_rev_v(ctx, 3);
    }
}

// C# EmitRev_V.
fn emit_rev_v(ctx: &mut ArmEmitterCtx, container_size: i32) {
    let op = decode_simd_op(ctx);

    let mut res = ctx.vector_zero();

    let elems = simd_bytes_count(op.reg_size) >> op.size;

    let container_mask = (1 << (container_size - op.size)) - 1;

    for index in 0..elems {
        let rev_index = index ^ container_mask;

        let ne = emit_vector_extract_zx(ctx, op.rn, rev_index, op.size);

        res = emit_vector_insert(ctx, res, ne, index, op.size);
    }

    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Silence the unused warning until more files consume it.
#[allow(unused)]
fn _kind_check(_: VecRegisterIndex) {}
