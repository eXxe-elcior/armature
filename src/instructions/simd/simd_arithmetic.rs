// Port of InstEmitSimdArithmetic.cs
// Host x86-64 only: AdvSimd paths excluded. SSE paths preferred, IR fallback.

use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::helpers::get_vec;
use crate::instructions::native_function::NativeFn;
use crate::instructions::simd_helper::*;
use crate::ir::operand::OperandIndex;
use crate::ir::types::Intrinsic;
use crate::optimizations;

// Abs_S
pub fn emit_abs_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_unary_op_sx(ctx, |ctx, op1| emit_abs(ctx, op1));
}

pub fn emit_abs_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    // SSSE3 PABS does per-element |x| in one instruction for byte/word/dword.
    // Size 3 (2x i64) has no PABSQ before AVX512, so it keeps the scalar loop.
    // PABS(INT_MIN) == INT_MIN, which matches ARM ABS (no saturation).
    if optimizations::use_ssse3() && op.size < 3 {
        let pabs = [
            Intrinsic::X86Pabsb,
            Intrinsic::X86Pabsw,
            Intrinsic::X86Pabsd,
        ][op.size as usize];
        let n = get_vec(ctx, op.rn);
        let mut res = ctx.emit_intrinsic(pabs, &[n]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_unary_op_sx(ctx, |ctx, op1| emit_abs(ctx, op1));
    }
}

// Add_S / Add_V
pub fn emit_add_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_zx(ctx, |ctx, op1, op2| ctx.add(op1, op2));
}

pub fn emit_add_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let add_inst = X86_PADD_INSTRUCTION[op.size as usize];
    let mut res = ctx.emit_intrinsic(add_inst, &[n, m]);
    if op.reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Addhn_V
pub fn emit_addhn_v(ctx: &mut ArmEmitterCtx) {
    emit_high_narrow(ctx, |ctx, op1, op2| ctx.add(op1, op2), false);
}

// Addp_S
pub fn emit_addp_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    let ne0 = emit_vector_extract_zx(ctx, op.rn, 0, op.size);
    let ne1 = emit_vector_extract_zx(ctx, op.rn, 1, op.size);
    let res = ctx.add(ne0, ne1);
    let rd = get_vec(ctx, op.rd);
    let zero = ctx.vector_zero();
    let tmp = emit_vector_insert(ctx, zero, res, 0, op.size);
    ctx.copy_to(rd, tmp);
}

// Addp_V
pub fn emit_addp_v(ctx: &mut ArmEmitterCtx) {
    emit_ssse3_vector_pairwise_op(ctx, &X86_PADD_INSTRUCTION);
}

// Addv_V
pub fn emit_addv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_across_vector_op_zx(ctx, |ctx, op1, op2| ctx.add(op1, op2));
}

// Cls_V
pub fn emit_cls_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    let mut res = ctx.vector_zero();
    let elems = bytes_count(op.reg_size) >> op.size;
    let e_size = 8 << op.size;
    for index in 0..elems {
        let ne = emit_vector_extract_zx(ctx, op.rn, index, op.size);
        let e_size_c = ctx.ctx.operands.constant_i32(e_size);
        let de = ctx
            .call(NativeFn::CountLeadingSigns, &[ne, e_size_c])
            .expect("CountLeadingSigns returns I64");
        res = emit_vector_insert(ctx, res, de, index, op.size);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Clz_V
pub fn emit_clz_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    let e_size = 8 << op.size;
    let rn = get_vec(ctx, op.rn);
    let res = match e_size {
        8 => clz_v_i8(ctx, rn),
        16 => clz_v_i16(ctx, rn),
        32 => clz_v_i32(ctx, rn),
        _ => None,
    };
    if let Some(mut res) = res {
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let elems = bytes_count(op.reg_size) >> op.size;
        let mut res = ctx.vector_zero();
        for index in 0..elems {
            let ne = emit_vector_extract_zx(ctx, op.rn, index, op.size);
            let e_size_c = ctx.ctx.operands.constant_i32(e_size);
            let de = ctx
                .call(NativeFn::CountLeadingZeros, &[ne, e_size_c])
                .expect("CountLeadingZeros returns I64");
            res = emit_vector_insert(ctx, res, de, index, op.size);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

fn clz_v_i8(ctx: &mut ArmEmitterCtx, arg: OperandIndex) -> Option<OperandIndex> {
    if !optimizations::use_ssse3() {
        return None;
    }

    // CLZ nibble table.
    let clz_table = x86_get_scalar_i64(ctx, 0x01_01_01_01_02_02_03_04);

    let mask_low = x86_get_all_elements_i32(ctx, 0x0f_0f_0f_0f);
    let c04 = x86_get_all_elements_i32(ctx, 0x04_04_04_04);

    // CLZ of low 4 bits of elements in arg.
    let lo_clz = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[clz_table, arg]);

    // Get the high 4 bits of elements in arg.
    let four = ctx.ctx.operands.constant_i32(4);
    let hi_arg = ctx.emit_intrinsic(Intrinsic::X86Psrlw, &[arg, four]);
    let hi_arg = ctx.emit_intrinsic(Intrinsic::X86Pand, &[hi_arg, mask_low]);

    // CLZ of high 4 bits of elements in arg.
    let hi_clz = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[clz_table, hi_arg]);

    // If high 4 bits are not all zero, we discard the CLZ of the low 4 bits.
    let mask = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqb, &[hi_clz, c04]);
    let lo_clz = ctx.emit_intrinsic(Intrinsic::X86Pand, &[lo_clz, mask]);

    Some(ctx.emit_intrinsic(Intrinsic::X86Paddb, &[lo_clz, hi_clz]))
}

fn clz_v_i16(ctx: &mut ArmEmitterCtx, arg: OperandIndex) -> Option<OperandIndex> {
    if !optimizations::use_ssse3() {
        return None;
    }

    let mask_swap = x86_get_elements(
        ctx,
        0x80_0f_80_0d_80_0b_80_09u64 as i64,
        0x80_07_80_05_80_03_80_01u64 as i64,
    );
    let mask_low = x86_get_all_elements_i32(ctx, 0x00ff_00ff);
    let c0008 = x86_get_all_elements_i32(ctx, 0x0008_0008);

    // CLZ pair of high 8 and low 8 bits of elements in arg.
    let hilo_clz = clz_v_i8(ctx, arg).expect("use_ssse3 already checked");
    // Get CLZ of low 8 bits in each pair.
    let lo_clz = ctx.emit_intrinsic(Intrinsic::X86Pand, &[hilo_clz, mask_low]);
    // Get CLZ of high 8 bits in each pair.
    let hi_clz = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[hilo_clz, mask_swap]);

    // If high 8 bits are not all zero, we discard the CLZ of the low 8 bits.
    let mask = ctx.emit_intrinsic(Intrinsic::X86Pcmpeqw, &[hi_clz, c0008]);
    let lo_clz = ctx.emit_intrinsic(Intrinsic::X86Pand, &[lo_clz, mask]);

    Some(ctx.emit_intrinsic(Intrinsic::X86Paddw, &[lo_clz, hi_clz]))
}

fn clz_v_i32(ctx: &mut ArmEmitterCtx, arg: OperandIndex) -> Option<OperandIndex> {
    // TODO (upstream): use vplzcntd when AVX-512 is supported.
    if !optimizations::use_sse2() {
        return None;
    }

    let c55555555 = x86_get_all_elements_i32(ctx, 0x55555555);
    let c33333333 = x86_get_all_elements_i32(ctx, 0x33333333);
    let c0f0f0f0f = x86_get_all_elements_i32(ctx, 0x0f0f0f0f);
    let c0000003f = x86_get_all_elements_i32(ctx, 0x0000003f);

    let shr = |ctx: &mut ArmEmitterCtx, op0: OperandIndex, imm8: i32| {
        let c = ctx.ctx.operands.constant_i32(imm8);
        ctx.emit_intrinsic(Intrinsic::X86Psrld, &[op0, c])
    };

    // Set all bits after highest set bit to 1.
    let mut res = shr(ctx, arg, 1);
    res = ctx.emit_intrinsic(Intrinsic::X86Por, &[res, arg]);
    let t = shr(ctx, res, 2);
    res = ctx.emit_intrinsic(Intrinsic::X86Por, &[t, res]);
    let t = shr(ctx, res, 4);
    res = ctx.emit_intrinsic(Intrinsic::X86Por, &[t, res]);
    let t = shr(ctx, res, 8);
    res = ctx.emit_intrinsic(Intrinsic::X86Por, &[t, res]);
    let t = shr(ctx, res, 16);
    res = ctx.emit_intrinsic(Intrinsic::X86Por, &[t, res]);

    // Make leading 0s into leading 1s.
    let ones = ctx.vector_one();
    res = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[res, ones]);

    // Count leading 1s, which is the population count.
    let mut tmp0 = shr(ctx, res, 1);
    tmp0 = ctx.emit_intrinsic(Intrinsic::X86Pand, &[tmp0, c55555555]);
    res = ctx.emit_intrinsic(Intrinsic::X86Psubd, &[res, tmp0]);

    let mut tmp0 = shr(ctx, res, 2);
    tmp0 = ctx.emit_intrinsic(Intrinsic::X86Pand, &[tmp0, c33333333]);
    let tmp1 = ctx.emit_intrinsic(Intrinsic::X86Pand, &[res, c33333333]);
    res = ctx.emit_intrinsic(Intrinsic::X86Paddd, &[tmp0, tmp1]);

    let mut tmp0 = shr(ctx, res, 4);
    tmp0 = ctx.emit_intrinsic(Intrinsic::X86Paddd, &[tmp0, res]);
    res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[tmp0, c0f0f0f0f]);

    let tmp0 = shr(ctx, res, 8);
    res = ctx.emit_intrinsic(Intrinsic::X86Paddd, &[tmp0, res]);

    let tmp0 = shr(ctx, res, 16);
    res = ctx.emit_intrinsic(Intrinsic::X86Paddd, &[tmp0, res]);

    Some(ctx.emit_intrinsic(Intrinsic::X86Pand, &[res, c0000003f]))
}

// Cnt_V
pub fn emit_cnt_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if optimizations::use_ssse3() {
        // Muła nibble-lookup popcount: per-byte popcount of the whole vector in
        // a few packed ops instead of a 16-iteration extract/popcnt/insert loop.
        // table[i] = popcount(i) for i in 0..16, low 8 then high 8 bytes.
        let n = get_vec(ctx, op.rn);
        let table = x86_get_elements(ctx, 0x0403030203020201, 0x0302020102010100);
        let low_mask = x86_get_all_elements_i32(ctx, 0x0F0F0F0Fu32 as i32);
        let lo = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, low_mask]);
        let const_4 = ctx.ctx.operands.constant_i32(4);
        let mut hi = ctx.emit_intrinsic(Intrinsic::X86Psrlw, &[n, const_4]);
        hi = ctx.emit_intrinsic(Intrinsic::X86Pand, &[hi, low_mask]);
        let pc_lo = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[table, lo]);
        let pc_hi = ctx.emit_intrinsic(Intrinsic::X86Pshufb, &[table, hi]);
        let mut res = ctx.emit_intrinsic(X86_PADD_INSTRUCTION[0], &[pc_lo, pc_hi]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let mut res = ctx.vector_zero();
        let elems = if op.reg_size.bits() == 128 { 16 } else { 8 };
        for index in 0..elems {
            let ne = emit_vector_extract_zx(ctx, op.rn, index, 0);
            let de = if optimizations::use_popcnt() {
                ctx.emit_intrinsic_long(Intrinsic::X86Popcnt, &[ne])
            } else {
                emit_count_set_bits8(ctx, ne)
            };
            res = emit_vector_insert(ctx, res, de, index, 0);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Fabd_S / Fabd_V
pub fn emit_fabd_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let size_f = op.size & 1;
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if size_f == 0 {
        let res = ctx.emit_intrinsic(Intrinsic::X86Subss, &[n, m]);
        let res = emit_float_abs(ctx, res, true, false);
        let res = ctx.vector_zero_upper96(res);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let res = ctx.emit_intrinsic(Intrinsic::X86Subsd, &[n, m]);
        let res = emit_float_abs(ctx, res, false, false);
        let res = ctx.vector_zero_upper64(res);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

pub fn emit_fabd_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let size_f = op.size & 1;
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if size_f == 0 {
        let res = ctx.emit_intrinsic(Intrinsic::X86Subps, &[n, m]);
        let res = emit_float_abs(ctx, res, true, true);
        let mut res = res;
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let res = ctx.emit_intrinsic(Intrinsic::X86Subpd, &[n, m]);
        let res = emit_float_abs(ctx, res, false, true);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Fabs_S / Fabs_V
pub fn emit_fabs_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if op.size == 0 {
        let rn = get_vec(ctx, op.rn);
        let res = emit_float_abs(ctx, rn, true, false);
        let res = ctx.vector_zero_upper96(res);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let rn = get_vec(ctx, op.rn);
        let res = emit_float_abs(ctx, rn, false, false);
        let res = ctx.vector_zero_upper64(res);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

pub fn emit_fabs_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    let size_f = op.size & 1;
    if size_f == 0 {
        let rn = get_vec(ctx, op.rn);
        let mut res = emit_float_abs(ctx, rn, true, true);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let rn = get_vec(ctx, op.rn);
        let res = emit_float_abs(ctx, rn, false, true);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Fadd_S / Fadd_V
pub fn emit_fadd_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_f_sse(ctx, Intrinsic::X86Addss, Intrinsic::X86Addsd);
}

pub fn emit_fadd_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_f_sse(ctx, Intrinsic::X86Addps, Intrinsic::X86Addpd);
}

// Faddp_S / Faddp_V
pub fn emit_faddp_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if (op.size & 1) == 0 {
        let rn = get_vec(ctx, op.rn);
        let res = ctx.emit_intrinsic(Intrinsic::X86Haddps, &[rn, rn]);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(rd, tmp);
    } else {
        let rn = get_vec(ctx, op.rn);
        let res = ctx.emit_intrinsic(Intrinsic::X86Haddpd, &[rn, rn]);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper64(res);
        ctx.copy_to(rd, tmp);
    }
}

pub fn emit_faddp_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| {
                let op = decode_simd_reg_op(ctx);
                let add_inst = if (op.size & 1) == 0 {
                    Intrinsic::X86Addps
                } else {
                    Intrinsic::X86Addpd
                };
                ctx.emit_intrinsic(add_inst, &[a, b])
            },
            false,
            Some(op1),
            Some(op2),
        )
    });
}

// Fdiv_S / Fdiv_V
pub fn emit_fdiv_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_f_sse(ctx, Intrinsic::X86Divss, Intrinsic::X86Divsd);
}

pub fn emit_fdiv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_f_sse(ctx, Intrinsic::X86Divps, Intrinsic::X86Divpd);
}

// Fmadd_S (fused)
pub fn emit_fmadd_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let d = get_vec(ctx, op.rd);
    let a = get_vec(ctx, op.ra);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if op.size == 0 {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmadd231ss, &[a, n, m])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Addss, &[a, r])
        };
        let res = ctx.vector_zero_upper96(res);
        ctx.copy_to(d, res);
    } else {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmadd231sd, &[a, n, m])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Addsd, &[a, r])
        };
        let res = ctx.vector_zero_upper64(res);
        ctx.copy_to(d, res);
    }
}

// Fmax_S / Fmax_V
pub fn emit_fmax_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_process_nans_op_f(
        ctx,
        |ctx, op1, op2| emit_sse2_vector_max_min_op_f(ctx, op1, op2, true),
        true,
        None,
        None,
    );
}

pub fn emit_fmax_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_process_nans_op_f(
        ctx,
        |ctx, op1, op2| emit_sse2_vector_max_min_op_f(ctx, op1, op2, true),
        false,
        None,
        None,
    );
}

// Fmaxnm_S / Fmaxnm_V
pub fn emit_fmaxnm_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_max_min_num_op_f(ctx, true, true, None, None);
}

pub fn emit_fmaxnm_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_max_min_num_op_f(ctx, true, false, None, None);
}

// Fmaxnmp_S / Fmaxnmp_V
pub fn emit_fmaxnmp_s(ctx: &mut ArmEmitterCtx) {
    emit_sse2_scalar_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_max_min_num_op_f(ctx, true, true, Some(op1), Some(op2))
    });
}

pub fn emit_fmaxnmp_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_max_min_num_op_f(ctx, true, false, Some(op1), Some(op2))
    });
}

// Fmaxnmv_V
pub fn emit_fmaxnmv_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_across_vector_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_max_min_num_op_f(ctx, true, false, Some(op1), Some(op2))
    });
}

// Fmaxp_S / Fmaxp_V
pub fn emit_fmaxp_s(ctx: &mut ArmEmitterCtx) {
    emit_sse2_scalar_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, true),
            true,
            Some(op1),
            Some(op2),
        )
    });
}

pub fn emit_fmaxp_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, true),
            false,
            Some(op1),
            Some(op2),
        )
    });
}

// Fmaxv_V
pub fn emit_fmaxv_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_across_vector_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, true),
            false,
            Some(op1),
            Some(op2),
        )
    });
}

// Fmin_S / Fmin_V
pub fn emit_fmin_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_process_nans_op_f(
        ctx,
        |ctx, op1, op2| emit_sse2_vector_max_min_op_f(ctx, op1, op2, false),
        true,
        None,
        None,
    );
}

pub fn emit_fmin_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_process_nans_op_f(
        ctx,
        |ctx, op1, op2| emit_sse2_vector_max_min_op_f(ctx, op1, op2, false),
        false,
        None,
        None,
    );
}

// Fminnm_S / Fminnm_V
pub fn emit_fminnm_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_max_min_num_op_f(ctx, false, true, None, None);
}

pub fn emit_fminnm_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_max_min_num_op_f(ctx, false, false, None, None);
}

// Fminnmp_S / Fminnmp_V
pub fn emit_fminnmp_s(ctx: &mut ArmEmitterCtx) {
    emit_sse2_scalar_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_max_min_num_op_f(ctx, false, true, Some(op1), Some(op2))
    });
}

pub fn emit_fminnmp_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_max_min_num_op_f(ctx, false, false, Some(op1), Some(op2))
    });
}

// Fminnmv_V
pub fn emit_fminnmv_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_across_vector_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_max_min_num_op_f(ctx, false, false, Some(op1), Some(op2))
    });
}

// Fminp_S / Fminp_V
pub fn emit_fminp_s(ctx: &mut ArmEmitterCtx) {
    emit_sse2_scalar_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, false),
            true,
            Some(op1),
            Some(op2),
        )
    });
}

pub fn emit_fminp_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_pairwise_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, false),
            false,
            Some(op1),
            Some(op2),
        )
    });
}

// Fminv_V
pub fn emit_fminv_v(ctx: &mut ArmEmitterCtx) {
    emit_sse2_vector_across_vector_op_f(ctx, |ctx, op1, op2| {
        emit_sse41_process_nans_op_f(
            ctx,
            |ctx, a, b| emit_sse2_vector_max_min_op_f(ctx, a, b, false),
            false,
            Some(op1),
            Some(op2),
        )
    });
}

// Fmla_Se / Fmla_V / Fmla_Ve (fused)
pub fn emit_fmla_se(ctx: &mut ArmEmitterCtx) {
    // C# Fmla_Se: SSE path exists only under UseFma; otherwise scalar
    // ternary by-elem (Add(op1, Multiply(op2, op3))).
    if optimizations::use_fma() {
        let op = decode_simd_reg_elem_f_op(ctx);
        let d = get_vec(ctx, op.rd);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        if (op.size & 1) == 0 {
            let shuffle_mask = op.index | (op.index << 2) | (op.index << 4) | (op.index << 6);
            let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
            let res = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[m, m, const_imm]);
            let res = ctx.emit_intrinsic(Intrinsic::X86Vfmadd231ss, &[d, n, res]);
            let tmp = ctx.vector_zero_upper96(res);
            ctx.copy_to(d, tmp);
        } else {
            let shuffle_mask = op.index | (op.index << 1);
            let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
            let res = ctx.emit_intrinsic(Intrinsic::X86Shufpd, &[m, m, const_imm]);
            let res = ctx.emit_intrinsic(Intrinsic::X86Vfmadd231sd, &[d, n, res]);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(d, tmp);
        }
    } else {
        emit_scalar_ternary_op_by_elem_f(ctx, |ctx, op1, op2, op3| {
            let prod = ctx.multiply(op2, op3);
            ctx.add(op1, prod)
        });
    }
}

pub fn emit_fmla_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let d = get_vec(ctx, op.rd);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if (op.size & 1) == 0 {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmadd231ps, &[d, n, m])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Addps, &[d, r])
        };
        if op.reg_size.bits() == 64 {
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(d, tmp);
        } else {
            ctx.copy_to(d, res);
        }
    } else {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmadd231pd, &[d, n, m])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Addpd, &[d, r])
        };
        ctx.copy_to(d, res);
    }
}

pub fn emit_fmla_ve(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_elem_f_op(ctx);
    let d = get_vec(ctx, op.rd);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if (op.size & 1) == 0 {
        let shuffle_mask = op.index | (op.index << 2) | (op.index << 4) | (op.index << 6);
        let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
        let m_shuf = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[m, m, const_imm]);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmadd231ps, &[d, n, m_shuf])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n, m_shuf]);
            ctx.emit_intrinsic(Intrinsic::X86Addps, &[d, r])
        };
        if op.reg_size.bits() == 64 {
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(d, tmp);
        } else {
            ctx.copy_to(d, res);
        }
    } else {
        let shuffle_mask = op.index | (op.index << 1);
        let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
        let m_shuf = ctx.emit_intrinsic(Intrinsic::X86Shufpd, &[m, m, const_imm]);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmadd231pd, &[d, n, m_shuf])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n, m_shuf]);
            ctx.emit_intrinsic(Intrinsic::X86Addpd, &[d, r])
        };
        ctx.copy_to(d, res);
    }
}

// Fmls_Se / Fmls_V / Fmls_Ve (fused)
pub fn emit_fmls_se(ctx: &mut ArmEmitterCtx) {
    // C# Fmls_Se: SSE path exists only under UseFma; otherwise scalar
    // ternary by-elem (Subtract(op1, Multiply(op2, op3))).
    if optimizations::use_fma() {
        let op = decode_simd_reg_elem_f_op(ctx);
        let d = get_vec(ctx, op.rd);
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        if (op.size & 1) == 0 {
            let shuffle_mask = op.index | (op.index << 2) | (op.index << 4) | (op.index << 6);
            let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
            let res = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[m, m, const_imm]);
            let res = ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ss, &[d, n, res]);
            let tmp = ctx.vector_zero_upper96(res);
            ctx.copy_to(d, tmp);
        } else {
            let shuffle_mask = op.index | (op.index << 1);
            let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
            let res = ctx.emit_intrinsic(Intrinsic::X86Shufpd, &[m, m, const_imm]);
            let res = ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231sd, &[d, n, res]);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(d, tmp);
        }
    } else {
        emit_scalar_ternary_op_by_elem_f(ctx, |ctx, op1, op2, op3| {
            let prod = ctx.multiply(op2, op3);
            ctx.subtract(op1, prod)
        });
    }
}

pub fn emit_fmls_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let d = get_vec(ctx, op.rd);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if (op.size & 1) == 0 {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ps, &[d, n, m])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subps, &[d, r])
        };
        if op.reg_size.bits() == 64 {
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(d, tmp);
        } else {
            ctx.copy_to(d, res);
        }
    } else {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231pd, &[d, n, m])
        } else {
            let r = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subpd, &[d, r])
        };
        ctx.copy_to(d, res);
    }
}

pub fn emit_fmls_ve(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_elem_f_op(ctx);
    let d = get_vec(ctx, op.rd);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if (op.size & 1) == 0 {
        let shuffle_mask = op.index | (op.index << 2) | (op.index << 4) | (op.index << 6);
        let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
        let m_shuf = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[m, m, const_imm]);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ps, &[d, n, m_shuf])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n, m_shuf]);
            ctx.emit_intrinsic(Intrinsic::X86Subps, &[d, mul])
        };
        if op.reg_size.bits() == 64 {
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(d, tmp);
        } else {
            ctx.copy_to(d, res);
        }
    } else {
        let shuffle_mask = op.index | (op.index << 1);
        let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
        let m_shuf = ctx.emit_intrinsic(Intrinsic::X86Shufpd, &[m, m, const_imm]);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231pd, &[d, n, m_shuf])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n, m_shuf]);
            ctx.emit_intrinsic(Intrinsic::X86Subpd, &[d, mul])
        };
        ctx.copy_to(d, res);
    }
}

// Fmsub_S (fused)
pub fn emit_fmsub_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let d = get_vec(ctx, op.rd);
    let a = get_vec(ctx, op.ra);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if op.size == 0 {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ss, &[a, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subss, &[a, mul])
        };
        let res = ctx.vector_zero_upper96(res);
        ctx.copy_to(d, res);
    } else {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231sd, &[a, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subsd, &[a, mul])
        };
        let res = ctx.vector_zero_upper64(res);
        ctx.copy_to(d, res);
    }
}

// Fmul_S / Fmul_V / Fmul_Se / Fmul_Ve
pub fn emit_fmul_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_f_sse(ctx, Intrinsic::X86Mulss, Intrinsic::X86Mulsd);
}

pub fn emit_fmul_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_f_sse(ctx, Intrinsic::X86Mulps, Intrinsic::X86Mulpd);
}

pub fn emit_fmul_se(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_by_elem_f(ctx, |ctx, op1, op2| ctx.multiply(op1, op2));
}

pub fn emit_fmul_ve(ctx: &mut ArmEmitterCtx) {
    // C# Fmul_Ve FastFP && UseSse2 path: shuffle Rm element then Mulps/Mulpd.
    let op = decode_simd_reg_elem_f_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if (op.size & 1) == 0 {
        let shuffle_mask = op.index | (op.index << 2) | (op.index << 4) | (op.index << 6);
        let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
        let res = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[m, m, const_imm]);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n, res]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let shuffle_mask = op.index | (op.index << 1);
        let const_imm = ctx.ctx.operands.constant_i32(shuffle_mask as i32);
        let res = ctx.emit_intrinsic(Intrinsic::X86Shufpd, &[m, m, const_imm]);
        let res = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n, res]);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Fmulx_Se / Fmulx_V / Fmulx_Ve
pub fn emit_fmulx_se(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_by_elem_f(ctx, |ctx, op1, op2| {
        emit_soft_float_call(ctx, "FPMulX", &[op1, op2])
    });
}

pub fn emit_fmulx_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_f(ctx, |ctx, op1, op2| {
        emit_soft_float_call(ctx, "FPMulX", &[op1, op2])
    });
}

pub fn emit_fmulx_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_f(ctx, |ctx, op1, op2| {
        emit_soft_float_call(ctx, "FPMulX", &[op1, op2])
    });
}

pub fn emit_fmulx_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_by_elem_f(ctx, |ctx, op1, op2| {
        emit_soft_float_call(ctx, "FPMulX", &[op1, op2])
    });
}

// Fneg_S / Fneg_V
pub fn emit_fneg_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if op.size == 0 {
        let mask = x86_get_scalar_f32(ctx, -0.0f32);
        let rn = get_vec(ctx, op.rn);
        let res = ctx.emit_intrinsic(Intrinsic::X86Xorps, &[mask, rn]);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(rd, tmp);
    } else {
        let mask = x86_get_scalar_f64(ctx, -0.0f64);
        let rn = get_vec(ctx, op.rn);
        let res = ctx.emit_intrinsic(Intrinsic::X86Xorpd, &[mask, rn]);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper64(res);
        ctx.copy_to(rd, tmp);
    }
}

pub fn emit_fneg_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    let size_f = op.size & 1;
    if size_f == 0 {
        let mask = x86_get_all_elements_f32(ctx, -0.0f32);
        let rn = get_vec(ctx, op.rn);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Xorps, &[mask, rn]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        let mask = x86_get_all_elements_f64(ctx, -0.0f64);
        let rn = get_vec(ctx, op.rn);
        let res = ctx.emit_intrinsic(Intrinsic::X86Xorpd, &[mask, rn]);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Fnmadd_S / Fnmsub_S / Fnmul_S (fused)
pub fn emit_fnmadd_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let d = get_vec(ctx, op.rd);
    let a = get_vec(ctx, op.ra);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if op.size == 0 {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmsub231ss, &[a, n, m])
        } else {
            let mask = x86_get_scalar_f32(ctx, -0.0f32);
            let a_neg = ctx.emit_intrinsic(Intrinsic::X86Xorps, &[mask, a]);
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subss, &[a_neg, mul])
        };
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(d, tmp);
    } else {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmsub231sd, &[a, n, m])
        } else {
            let mask = x86_get_scalar_f64(ctx, -0.0f64);
            let a_neg = ctx.emit_intrinsic(Intrinsic::X86Xorpd, &[mask, a]);
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subsd, &[a_neg, mul])
        };
        let tmp = ctx.vector_zero_upper64(res);
        ctx.copy_to(d, tmp);
    }
}

pub fn emit_fnmsub_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let d = get_vec(ctx, op.rd);
    let a = get_vec(ctx, op.ra);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if op.size == 0 {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmsub231ss, &[a, n, m])
        } else {
            let mask = x86_get_scalar_f32(ctx, -0.0f32);
            let a_neg = ctx.emit_intrinsic(Intrinsic::X86Xorps, &[mask, a]);
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Addss, &[a_neg, mul])
        };
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(d, tmp);
    } else {
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfmsub231sd, &[a, n, m])
        } else {
            let mask = x86_get_scalar_f64(ctx, -0.0f64);
            let a_neg = ctx.emit_intrinsic(Intrinsic::X86Xorpd, &[mask, a]);
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Addsd, &[a_neg, mul])
        };
        let tmp = ctx.vector_zero_upper64(res);
        ctx.copy_to(d, tmp);
    }
}

pub fn emit_fnmul_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_f(ctx, |ctx, op1, op2| {
        let prod = ctx.multiply(op1, op2);
        ctx.negate(prod)
    });
}

// Frecpe_S / Frecpe_V
pub fn emit_frecpe_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if (op.size & 1) == 0 {
        let rn = get_vec(ctx, op.rn);
        let res = ctx.emit_intrinsic(Intrinsic::X86Rcpss, &[rn]);
        let res = emit_sse41_round32_exp8_op_f(ctx, res, true);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(rd, tmp);
    } else {
        emit_scalar_unary_op_f(ctx, |ctx, op1| {
            emit_soft_float_call(ctx, "FPRecipEstimate", &[op1])
        });
    }
}

pub fn emit_frecpe_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if (op.size & 1) == 0 {
        let rn = get_vec(ctx, op.rn);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Rcpps, &[rn]);
        res = emit_sse41_round32_exp8_op_f(ctx, res, false);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_unary_op_f(ctx, |ctx, op1| {
            emit_soft_float_call(ctx, "FPRecipEstimate", &[op1])
        });
    }
}

// Frecps_S / Frecps_V (fused)
pub fn emit_frecps_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let size_f = op.size & 1;
    if size_f == 0 {
        let mask = x86_get_scalar_f32(ctx, 2.0f32);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ss, &[mask, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subss, &[mask, mul])
        };
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask, true, size_f);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(rd, tmp);
    } else {
        let mask = x86_get_scalar_f64(ctx, 2.0f64);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231sd, &[mask, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subsd, &[mask, mul])
        };
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask, true, size_f);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper64(res);
        ctx.copy_to(rd, tmp);
    }
}

pub fn emit_frecps_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let size_f = op.size & 1;
    if size_f == 0 {
        let mask = x86_get_all_elements_f32(ctx, 2.0f32);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ps, &[mask, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subps, &[mask, mul])
        };
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask, false, size_f);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        let mask = x86_get_all_elements_f64(ctx, 2.0f64);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231pd, &[mask, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subpd, &[mask, mul])
        };
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask, false, size_f);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Frecpx_S
pub fn emit_frecpx_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_unary_op_f(ctx, |ctx, op1| emit_soft_float_call(ctx, "FPRecpX", &[op1]));
}

// Frinta_S / Frinta_V
pub fn emit_frinta_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_scalar_round_op_f(ctx, 4); // ToNearestAway
}

pub fn emit_frinta_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_round_op_f(ctx, 4); // ToNearestAway
}

// Frinti_S / Frinti_V
pub fn emit_frinti_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_unary_op_f(ctx, |ctx, op1| emit_round_by_rmode(ctx, op1));
}

pub fn emit_frinti_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_unary_op_f(ctx, |ctx, op1| emit_round_by_rmode(ctx, op1));
}

// Frintm_S / Frintm_V
pub fn emit_frintm_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_scalar_round_op_f(ctx, 2); // TowardsMinusInfinity
}

pub fn emit_frintm_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_round_op_f(ctx, 2); // TowardsMinusInfinity
}

// Frintn_S / Frintn_V
pub fn emit_frintn_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_scalar_round_op_f(ctx, 0); // ToNearest
}

pub fn emit_frintn_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_round_op_f(ctx, 0); // ToNearest
}

// Frintp_S / Frintp_V
pub fn emit_frintp_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_scalar_round_op_f(ctx, 1); // TowardsPlusInfinity
}

pub fn emit_frintp_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_round_op_f(ctx, 1); // TowardsPlusInfinity
}

// Frintx_S / Frintx_V
pub fn emit_frintx_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_unary_op_f(ctx, |ctx, op1| emit_round_by_rmode(ctx, op1));
}

pub fn emit_frintx_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_unary_op_f(ctx, |ctx, op1| emit_round_by_rmode(ctx, op1));
}

// Frintz_S / Frintz_V
pub fn emit_frintz_s(ctx: &mut ArmEmitterCtx) {
    emit_sse41_scalar_round_op_f(ctx, 3); // TowardsZero
}

pub fn emit_frintz_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_round_op_f(ctx, 3); // TowardsZero
}

// Frsqrte_S / Frsqrte_V
pub fn emit_frsqrte_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if (op.size & 1) == 0 {
        let rn = get_vec(ctx, op.rn);
        let res = ctx.emit_intrinsic(Intrinsic::X86Sqrtss, &[rn]);
        let res = ctx.emit_intrinsic(Intrinsic::X86Rcpss, &[res]);
        let res = emit_sse41_round32_exp8_op_f(ctx, res, true);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(rd, tmp);
    } else {
        emit_scalar_unary_op_f(ctx, |ctx, op1| {
            emit_soft_float_call(ctx, "FPRSqrtEstimate", &[op1])
        });
    }
}

pub fn emit_frsqrte_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    if (op.size & 1) == 0 {
        let rn = get_vec(ctx, op.rn);
        let mut res = ctx.emit_intrinsic(Intrinsic::X86Sqrtps, &[rn]);
        res = ctx.emit_intrinsic(Intrinsic::X86Rcpps, &[res]);
        res = emit_sse41_round32_exp8_op_f(ctx, res, false);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    } else {
        emit_vector_unary_op_f(ctx, |ctx, op1| {
            emit_soft_float_call(ctx, "FPRSqrtEstimate", &[op1])
        });
    }
}

// Frsqrts_S / Frsqrts_V (fused)
pub fn emit_frsqrts_s(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let size_f = op.size & 1;
    if size_f == 0 {
        let mask_half = x86_get_scalar_f32(ctx, 0.5f32);
        let mask_three = x86_get_scalar_f32(ctx, 3.0f32);
        let mask_one_half = x86_get_scalar_f32(ctx, 1.5f32);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ss, &[mask_three, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subss, &[mask_three, mul])
        };
        let res = ctx.emit_intrinsic(Intrinsic::X86Mulss, &[mask_half, res]);
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask_one_half, true, size_f);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper96(res);
        ctx.copy_to(rd, tmp);
    } else {
        let mask_half = x86_get_scalar_f64(ctx, 0.5f64);
        let mask_three = x86_get_scalar_f64(ctx, 3.0f64);
        let mask_one_half = x86_get_scalar_f64(ctx, 1.5f64);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231sd, &[mask_three, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subsd, &[mask_three, mul])
        };
        let res = ctx.emit_intrinsic(Intrinsic::X86Mulsd, &[mask_half, res]);
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask_one_half, true, size_f);
        let rd = get_vec(ctx, op.rd);
        let tmp = ctx.vector_zero_upper64(res);
        ctx.copy_to(rd, tmp);
    }
}

pub fn emit_frsqrts_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let size_f = op.size & 1;
    if size_f == 0 {
        let mask_half = x86_get_all_elements_f32(ctx, 0.5f32);
        let mask_three = x86_get_all_elements_f32(ctx, 3.0f32);
        let mask_one_half = x86_get_all_elements_f32(ctx, 1.5f32);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231ps, &[mask_three, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subps, &[mask_three, mul])
        };
        let res = ctx.emit_intrinsic(Intrinsic::X86Mulps, &[mask_half, res]);
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask_one_half, false, size_f);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        let mask_half = x86_get_all_elements_f64(ctx, 0.5f64);
        let mask_three = x86_get_all_elements_f64(ctx, 3.0f64);
        let mask_one_half = x86_get_all_elements_f64(ctx, 1.5f64);
        let res = if optimizations::use_fma() {
            ctx.emit_intrinsic(Intrinsic::X86Vfnmadd231pd, &[mask_three, n, m])
        } else {
            let mul = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[n, m]);
            ctx.emit_intrinsic(Intrinsic::X86Subpd, &[mask_three, mul])
        };
        let res = ctx.emit_intrinsic(Intrinsic::X86Mulpd, &[mask_half, res]);
        let res = emit_sse41_recip_step_select_op_f(ctx, n, m, res, mask_one_half, false, size_f);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Fsqrt_S / Fsqrt_V
pub fn emit_fsqrt_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_unary_op_f_sse(ctx, Intrinsic::X86Sqrtss, Intrinsic::X86Sqrtsd);
}

pub fn emit_fsqrt_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_unary_op_f_sse(ctx, Intrinsic::X86Sqrtps, Intrinsic::X86Sqrtpd);
}

// Fsub_S / Fsub_V
pub fn emit_fsub_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_f_sse(ctx, Intrinsic::X86Subss, Intrinsic::X86Subsd);
}

pub fn emit_fsub_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_f_sse(ctx, Intrinsic::X86Subps, Intrinsic::X86Subpd);
}

// Mla_V / Mls_V / Mul_V
pub fn emit_mla_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_mul_add_sub(ctx, 1); // Add
}

pub fn emit_mls_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_mul_add_sub(ctx, 2); // Subtract
}

pub fn emit_mul_v(ctx: &mut ArmEmitterCtx) {
    emit_sse41_vector_mul_add_sub(ctx, 0); // None
}

// Mla_Ve / Mls_Ve / Mul_Ve
pub fn emit_mla_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_ternary_op_by_elem_zx(ctx, |ctx, op1, op2, op3| {
        let prod = ctx.multiply(op2, op3);
        ctx.add(op1, prod)
    });
}

pub fn emit_mls_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_ternary_op_by_elem_zx(ctx, |ctx, op1, op2, op3| {
        let prod = ctx.multiply(op2, op3);
        ctx.subtract(op1, prod)
    });
}

pub fn emit_mul_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_by_elem_zx(ctx, |ctx, op1, op2| ctx.multiply(op1, op2));
}

// Neg_S / Neg_V
pub fn emit_neg_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_unary_op_sx(ctx, |ctx, op1| ctx.negate(op1));
}

pub fn emit_neg_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_op(ctx);
    let sub_inst = X86_PSUB_INSTRUCTION[op.size as usize];
    let zero = ctx.vector_zero();
    let rn = get_vec(ctx, op.rn);
    let mut res = ctx.emit_intrinsic(sub_inst, &[zero, rn]);
    if op.reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Pmull_V
pub fn emit_pmull_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    if optimizations::use_pclmulqdq() && op.size == 3 {
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let imm8 = if op.reg_size.bits() == 64 {
            0b0000_0000
        } else {
            0b0001_0001
        };
        let imm8_c = ctx.ctx.operands.constant_i32(imm8);
        let res = ctx.emit_intrinsic(Intrinsic::X86Pclmulqdq, &[n, m, imm8_c]);
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
        return;
    }
    if !optimizations::use_sse41() {
        // C# falls back to EmitPolynomialMultiply (size 0) /
        // SoftFallback.PolynomialMult64_128 (size 3)  the latter needs
        // V128-returning native calls, not wired yet.
        panic!("emit_pmull_v: non-SSE41 fallback not ported (needs PolynomialMult64_128)");
    }
    let mut n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 64 {
        n = ctx.vector_zero_upper64(n);
        m = ctx.vector_zero_upper64(m);
    } else {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mut res = ctx.vector_zero();
    if op.size == 0 {
        n = ctx.emit_intrinsic(Intrinsic::X86Pmovzxbw, &[n]);
        m = ctx.emit_intrinsic(Intrinsic::X86Pmovzxbw, &[m]);
        for i in 0..8 {
            let shift_15_minus_i = ctx.ctx.operands.constant_i32(15 - i);
            let shift_i = ctx.ctx.operands.constant_i32(i);
            let mask = ctx.emit_intrinsic(Intrinsic::X86Psllw, &[n, shift_15_minus_i]);
            let const_15 = ctx.ctx.operands.constant_i32(15);
            let mask = ctx.emit_intrinsic(Intrinsic::X86Psraw, &[mask, const_15]);
            let tmp = ctx.emit_intrinsic(Intrinsic::X86Psllw, &[m, shift_i]);
            let tmp = ctx.emit_intrinsic(Intrinsic::X86Pand, &[tmp, mask]);
            res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[res, tmp]);
        }
    } else {
        let zero = ctx.vector_zero();
        for i in 0..64 {
            let mask = ctx.emit_intrinsic(Intrinsic::X86Movlhps, &[n, n]);
            let shift = ctx.ctx.operands.constant_i32(63 - i);
            let mask = ctx.emit_intrinsic(Intrinsic::X86Psllq, &[mask, shift]);
            let const_63 = ctx.ctx.operands.constant_i32(63);
            let mask = ctx.emit_intrinsic(Intrinsic::X86Psrlq, &[mask, const_63]);
            let mask = ctx.emit_intrinsic(Intrinsic::X86Psubq, &[zero, mask]);
            let tmp = emit_sse2_sll_128(ctx, m, i);
            let tmp = ctx.emit_intrinsic(Intrinsic::X86Pand, &[tmp, mask]);
            res = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[res, tmp]);
        }
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Raddhn_V / Rsubhn_V
pub fn emit_raddhn_v(ctx: &mut ArmEmitterCtx) {
    emit_high_narrow(ctx, |ctx, op1, op2| ctx.add(op1, op2), true);
}

pub fn emit_rsubhn_v(ctx: &mut ArmEmitterCtx) {
    emit_high_narrow(ctx, |ctx, op1, op2| ctx.subtract(op1, op2), true);
}

// Saba_V / Sabal_V
pub fn emit_saba_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_ternary_op_sx(ctx, |ctx, op1, op2, op3| {
        let diff = ctx.subtract(op2, op3);
        let abs_diff = emit_abs(ctx, diff);
        ctx.add(op1, abs_diff)
    });
}

pub fn emit_sabal_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_rn_rm_ternary_op_sx(ctx, |ctx, op1, op2, op3| {
        let diff = ctx.subtract(op2, op3);
        let abs_diff = emit_abs(ctx, diff);
        ctx.add(op1, abs_diff)
    });
}

// Sabd_V / Sabdl_V
pub fn emit_sabd_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    emit_sse41_vector_sabd_op(ctx, n, m, false);
}

pub fn emit_sabdl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    // C#: UseSse41 && op.Size < 2.
    if optimizations::use_sse41() && op.size < 2 {
        let mut n = get_vec(ctx, op.rn);
        let mut m = get_vec(ctx, op.rm);
        if op.reg_size.bits() == 128 {
            let const_8 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
            m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
        }
        let mov_inst = if op.size == 0 {
            Intrinsic::X86Pmovsxbw
        } else {
            Intrinsic::X86Pmovsxwd
        };
        n = ctx.emit_intrinsic(mov_inst, &[n]);
        m = ctx.emit_intrinsic(mov_inst, &[m]);
        emit_sse41_vector_sabd_op(ctx, n, m, true);
    } else {
        emit_vector_widen_rn_rm_binary_op_sx(ctx, |ctx, op1, op2| {
            let diff = ctx.subtract(op1, op2);
            emit_abs(ctx, diff)
        });
    }
}

// Sadalp_V / Uadalp_V
pub fn emit_sadalp_v(ctx: &mut ArmEmitterCtx) {
    emit_add_long_pairwise(ctx, true, true);
}

pub fn emit_uadalp_v(ctx: &mut ArmEmitterCtx) {
    emit_add_long_pairwise(ctx, false, true);
}

// Saddl_V / Uaddl_V
pub fn emit_saddl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let mut n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVSX_INSTRUCTION[op.size as usize];
    n = ctx.emit_intrinsic(mov_inst, &[n]);
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let add_inst = X86_PADD_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(add_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_uaddl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let mut n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVZX_INSTRUCTION[op.size as usize];
    n = ctx.emit_intrinsic(mov_inst, &[n]);
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let add_inst = X86_PADD_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(add_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Saddlp_V / Uaddlp_V
pub fn emit_saddlp_v(ctx: &mut ArmEmitterCtx) {
    emit_add_long_pairwise(ctx, true, false);
}

pub fn emit_uaddlp_v(ctx: &mut ArmEmitterCtx) {
    emit_add_long_pairwise(ctx, false, false);
}

// Saddlv_V / Uaddlv_V
pub fn emit_saddlv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_long_across_vector_op_sx(ctx, |ctx, op1, op2| ctx.add(op1, op2));
}

pub fn emit_uaddlv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_long_across_vector_op_zx(ctx, |ctx, op1, op2| ctx.add(op1, op2));
}

// Saddw_V / Ssubw_V / Uaddw_V / Usubw_V
pub fn emit_saddw_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVSX_INSTRUCTION[op.size as usize];
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let add_inst = X86_PADD_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(add_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_ssubw_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVSX_INSTRUCTION[op.size as usize];
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let sub_inst = X86_PSUB_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(sub_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_uaddw_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVZX_INSTRUCTION[op.size as usize];
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let add_inst = X86_PADD_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(add_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_usubw_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVZX_INSTRUCTION[op.size as usize];
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let sub_inst = X86_PSUB_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(sub_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Shadd_V / Srhadd_V / Uhadd_V
pub fn emit_shadd_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    if op.size > 0 {
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, m]);
        let mut res2 = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);
        let shift_inst = if op.size == 1 {
            Intrinsic::X86Psraw
        } else {
            Intrinsic::X86Psrad
        };
        let const_1 = ctx.ctx.operands.constant_i32(1);
        res2 = ctx.emit_intrinsic(shift_inst, &[res2, const_1]);
        let add_inst = X86_PADD_INSTRUCTION[op.size as usize];
        let res = ctx.emit_intrinsic(add_inst, &[res, res2]);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        // size 0 (byte): no psrab. (a & b) + ((a ^ b) >>s 1); byte arithmetic
        // shift = word-shift + boundary mask, then restore each byte's sign bit.
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let and = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, m]);
        let xor = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);
        let const_1 = ctx.ctx.operands.constant_i32(1);
        let mut sh = ctx.emit_intrinsic(Intrinsic::X86Psrlw, &[xor, const_1]);
        let mask = x86_get_all_elements_i32(ctx, 0x7F7F7F7Fu32 as i32);
        sh = ctx.emit_intrinsic(Intrinsic::X86Pand, &[sh, mask]);
        let sign_mask = x86_get_all_elements_i32(ctx, 0x80808080u32 as i32);
        let signs = ctx.emit_intrinsic(Intrinsic::X86Pand, &[xor, sign_mask]);
        sh = ctx.emit_intrinsic(Intrinsic::X86Por, &[sh, signs]);
        let mut res = ctx.emit_intrinsic(X86_PADD_INSTRUCTION[0], &[and, sh]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

pub fn emit_srhadd_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    if op.size < 2 {
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let mask_val = if op.size == 0 {
            0x80808080u32
        } else {
            0x80008000u32
        };
        let mask = x86_get_all_elements_i32(ctx, mask_val as i32);
        let sub_inst = X86_PSUB_INSTRUCTION[op.size as usize];
        let n_minus_mask = ctx.emit_intrinsic(sub_inst, &[n, mask]);
        let m_minus_mask = ctx.emit_intrinsic(sub_inst, &[m, mask]);
        let avg_inst = if op.size == 0 {
            Intrinsic::X86Pavgb
        } else {
            Intrinsic::X86Pavgw
        };
        let mut res = ctx.emit_intrinsic(avg_inst, &[n_minus_mask, m_minus_mask]);
        let add_inst = X86_PADD_INSTRUCTION[op.size as usize];
        res = ctx.emit_intrinsic(add_inst, &[res, mask]);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        // size 2 (dword): no pavgd. Signed rounding average:
        // (a | b) - ((a ^ b) >>s 1).
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let or = ctx.emit_intrinsic(Intrinsic::X86Por, &[n, m]);
        let xor = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);
        let const_1 = ctx.ctx.operands.constant_i32(1);
        let sh = ctx.emit_intrinsic(Intrinsic::X86Psrad, &[xor, const_1]);
        let mut res = ctx.emit_intrinsic(X86_PSUB_INSTRUCTION[2], &[or, sh]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

pub fn emit_uhadd_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    if op.size > 0 {
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let res = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, m]);
        let mut res2 = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);
        let shift_inst = if op.size == 1 {
            Intrinsic::X86Psrlw
        } else {
            Intrinsic::X86Psrld
        };
        let const_1 = ctx.ctx.operands.constant_i32(1);
        res2 = ctx.emit_intrinsic(shift_inst, &[res2, const_1]);
        let add_inst = X86_PADD_INSTRUCTION[op.size as usize];
        let res = ctx.emit_intrinsic(add_inst, &[res, res2]);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        // size 0 (byte): no psrlb. (a & b) + ((a ^ b) >>u 1); emulate the byte
        // logical shift with a word shift + mask of the bit that crossed each
        // byte boundary.
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let and = ctx.emit_intrinsic(Intrinsic::X86Pand, &[n, m]);
        let xor = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);
        let const_1 = ctx.ctx.operands.constant_i32(1);
        let mut sh = ctx.emit_intrinsic(Intrinsic::X86Psrlw, &[xor, const_1]);
        let mask = x86_get_all_elements_i32(ctx, 0x7F7F7F7Fu32 as i32);
        sh = ctx.emit_intrinsic(Intrinsic::X86Pand, &[sh, mask]);
        let mut res = ctx.emit_intrinsic(X86_PADD_INSTRUCTION[0], &[and, sh]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Shsub_V / Uhsub_V
pub fn emit_shsub_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    if op.size < 2 {
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let mask_val = if op.size == 0 {
            0x80808080u32
        } else {
            0x80008000u32
        };
        let mask = x86_get_all_elements_i32(ctx, mask_val as i32);
        let add_inst = X86_PADD_INSTRUCTION[op.size as usize];
        let n_plus_mask = ctx.emit_intrinsic(add_inst, &[n, mask]);
        let m_plus_mask = ctx.emit_intrinsic(add_inst, &[m, mask]);
        let avg_inst = if op.size == 0 {
            Intrinsic::X86Pavgb
        } else {
            Intrinsic::X86Pavgw
        };
        let mut res = ctx.emit_intrinsic(avg_inst, &[n_plus_mask, m_plus_mask]);
        let sub_inst = X86_PSUB_INSTRUCTION[op.size as usize];
        res = ctx.emit_intrinsic(sub_inst, &[n_plus_mask, res]);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        // size 2 (dword): (a - b) >>s 1 = ((a ^ b) >>s 1) - (~a & b).
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let xor = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);
        let nb = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[n, m]); // ~n & m
        let const_1 = ctx.ctx.operands.constant_i32(1);
        let sh = ctx.emit_intrinsic(Intrinsic::X86Psrad, &[xor, const_1]);
        let mut res = ctx.emit_intrinsic(X86_PSUB_INSTRUCTION[2], &[sh, nb]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

pub fn emit_uhsub_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    if op.size < 2 {
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let avg_inst = if op.size == 0 {
            Intrinsic::X86Pavgb
        } else {
            Intrinsic::X86Pavgw
        };
        let res = ctx.emit_intrinsic(avg_inst, &[n, m]);
        let sub_inst = X86_PSUB_INSTRUCTION[op.size as usize];
        let res = ctx.emit_intrinsic(sub_inst, &[n, res]);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        // size 2 (dword): (a - b) >>u 1 = ((a ^ b) >>u 1) - (~a & b).
        let n = get_vec(ctx, op.rn);
        let m = get_vec(ctx, op.rm);
        let xor = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[n, m]);
        let nb = ctx.emit_intrinsic(Intrinsic::X86Pandn, &[n, m]); // ~n & m
        let const_1 = ctx.ctx.operands.constant_i32(1);
        let sh = ctx.emit_intrinsic(Intrinsic::X86Psrld, &[xor, const_1]);
        let mut res = ctx.emit_intrinsic(X86_PSUB_INSTRUCTION[2], &[sh, nb]);
        if op.reg_size.bits() == 64 {
            res = ctx.vector_zero_upper64(res);
        }
        let rd = get_vec(ctx, op.rd);
        ctx.copy_to(rd, res);
    }
}

// Smax_V / Smin_V / Umax_V / Umin_V
pub fn emit_smax_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_sx_using_sse41(ctx, &X86_PMAXS_INSTRUCTION);
}

pub fn emit_smin_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_sx_using_sse41(ctx, &X86_PMINS_INSTRUCTION);
}

pub fn emit_umax_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_zx_using_sse41(ctx, &X86_PMAXU_INSTRUCTION);
}

pub fn emit_umin_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_binary_op_zx_using_sse41(ctx, &X86_PMINU_INSTRUCTION);
}

// Helper for the above 4
fn emit_vector_binary_op_sx_using_sse41(ctx: &mut ArmEmitterCtx, inst_tbl: &[Intrinsic; 4]) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let max_inst = inst_tbl[op.size as usize];
    let mut res = ctx.emit_intrinsic(max_inst, &[n, m]);
    if op.reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

fn emit_vector_binary_op_zx_using_sse41(ctx: &mut ArmEmitterCtx, inst_tbl: &[Intrinsic; 4]) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let max_inst = inst_tbl[op.size as usize];
    let mut res = ctx.emit_intrinsic(max_inst, &[n, m]);
    if op.reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Smaxp_V / Sminp_V / Umaxp_V / Uminp_V
pub fn emit_smaxp_v(ctx: &mut ArmEmitterCtx) {
    emit_ssse3_vector_pairwise_op(ctx, &X86_PMAXS_INSTRUCTION);
}

pub fn emit_sminp_v(ctx: &mut ArmEmitterCtx) {
    emit_ssse3_vector_pairwise_op(ctx, &X86_PMINS_INSTRUCTION);
}

pub fn emit_umaxp_v(ctx: &mut ArmEmitterCtx) {
    emit_ssse3_vector_pairwise_op(ctx, &X86_PMAXU_INSTRUCTION);
}

pub fn emit_uminp_v(ctx: &mut ArmEmitterCtx) {
    emit_ssse3_vector_pairwise_op(ctx, &X86_PMINU_INSTRUCTION);
}

// Smaxv_V / Sminv_V / Umaxv_V / Uminv_V
pub fn emit_smaxv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_across_vector_op_sx(ctx, |ctx, op1, op2| emit_max64_op(ctx, op1, op2, true));
}

pub fn emit_sminv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_across_vector_op_sx(ctx, |ctx, op1, op2| emit_min64_op(ctx, op1, op2, true));
}

pub fn emit_umaxv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_across_vector_op_zx(ctx, |ctx, op1, op2| emit_max64_op(ctx, op1, op2, false));
}

pub fn emit_uminv_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_across_vector_op_zx(ctx, |ctx, op1, op2| emit_min64_op(ctx, op1, op2, false));
}

// Smlal_V / Smlsl_V / Smull_V / Umlal_V / Umlsl_V / Umull_V
pub fn emit_smlal_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    // C#: UseSse41 && op.Size < 2 (Pmullw/Pmulld cannot widen-mul 32-bit).
    if optimizations::use_sse41() && op.size < 2 {
        let d = get_vec(ctx, op.rd);
        let mut n = get_vec(ctx, op.rn);
        let mut m = get_vec(ctx, op.rm);
        if op.reg_size.bits() == 128 {
            let const_8 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
            m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
        }
        let mov_inst = X86_PMOVSX_INSTRUCTION[op.size as usize];
        n = ctx.emit_intrinsic(mov_inst, &[n]);
        m = ctx.emit_intrinsic(mov_inst, &[m]);
        let mull_inst = if op.size == 0 {
            Intrinsic::X86Pmullw
        } else {
            Intrinsic::X86Pmulld
        };
        let res = ctx.emit_intrinsic(mull_inst, &[n, m]);
        let add_inst = X86_PADD_INSTRUCTION[(op.size + 1) as usize];
        let res = ctx.emit_intrinsic(add_inst, &[d, res]);
        ctx.copy_to(d, res);
    } else {
        emit_vector_widen_rn_rm_ternary_op_sx(ctx, |ctx, op1, op2, op3| {
            let prod = ctx.multiply(op2, op3);
            ctx.add(op1, prod)
        });
    }
}

pub fn emit_smlsl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    // C#: UseSse41 && op.Size < 2.
    if optimizations::use_sse41() && op.size < 2 {
        let d = get_vec(ctx, op.rd);
        let mut n = get_vec(ctx, op.rn);
        let mut m = get_vec(ctx, op.rm);
        if op.reg_size.bits() == 128 {
            let const_8 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
            m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
        }
        let mov_inst = X86_PMOVSX_INSTRUCTION[op.size as usize];
        n = ctx.emit_intrinsic(mov_inst, &[n]);
        m = ctx.emit_intrinsic(mov_inst, &[m]);
        let mull_inst = if op.size == 0 {
            Intrinsic::X86Pmullw
        } else {
            Intrinsic::X86Pmulld
        };
        let res = ctx.emit_intrinsic(mull_inst, &[n, m]);
        let sub_inst = X86_PSUB_INSTRUCTION[(op.size + 1) as usize];
        let res = ctx.emit_intrinsic(sub_inst, &[d, res]);
        ctx.copy_to(d, res);
    } else {
        emit_vector_widen_rn_rm_ternary_op_sx(ctx, |ctx, op1, op2, op3| {
            let prod = ctx.multiply(op2, op3);
            ctx.subtract(op1, prod)
        });
    }
}

pub fn emit_smull_v(ctx: &mut ArmEmitterCtx) {
    // C# Smull_V has no SSE path  always the widen fallback.
    emit_vector_widen_rn_rm_binary_op_sx(ctx, |ctx, op1, op2| ctx.multiply(op1, op2));
}

pub fn emit_umlal_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    // C#: UseSse41 && op.Size < 2.
    if optimizations::use_sse41() && op.size < 2 {
        let d = get_vec(ctx, op.rd);
        let mut n = get_vec(ctx, op.rn);
        let mut m = get_vec(ctx, op.rm);
        if op.reg_size.bits() == 128 {
            let const_8 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
            m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
        }
        let mov_inst = X86_PMOVZX_INSTRUCTION[op.size as usize];
        n = ctx.emit_intrinsic(mov_inst, &[n]);
        m = ctx.emit_intrinsic(mov_inst, &[m]);
        let mull_inst = if op.size == 0 {
            Intrinsic::X86Pmullw
        } else {
            Intrinsic::X86Pmulld
        };
        let res = ctx.emit_intrinsic(mull_inst, &[n, m]);
        let add_inst = X86_PADD_INSTRUCTION[(op.size + 1) as usize];
        let res = ctx.emit_intrinsic(add_inst, &[d, res]);
        ctx.copy_to(d, res);
    } else {
        emit_vector_widen_rn_rm_ternary_op_zx(ctx, |ctx, op1, op2, op3| {
            let prod = ctx.multiply(op2, op3);
            ctx.add(op1, prod)
        });
    }
}

pub fn emit_umlsl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    // C#: UseSse41 && op.Size < 2.
    if optimizations::use_sse41() && op.size < 2 {
        let d = get_vec(ctx, op.rd);
        let mut n = get_vec(ctx, op.rn);
        let mut m = get_vec(ctx, op.rm);
        if op.reg_size.bits() == 128 {
            let const_8 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
            m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
        }
        let mov_inst = X86_PMOVZX_INSTRUCTION[op.size as usize];
        n = ctx.emit_intrinsic(mov_inst, &[n]);
        m = ctx.emit_intrinsic(mov_inst, &[m]);
        let mull_inst = if op.size == 0 {
            Intrinsic::X86Pmullw
        } else {
            Intrinsic::X86Pmulld
        };
        let res = ctx.emit_intrinsic(mull_inst, &[n, m]);
        let sub_inst = X86_PSUB_INSTRUCTION[(op.size + 1) as usize];
        let res = ctx.emit_intrinsic(sub_inst, &[d, res]);
        ctx.copy_to(d, res);
    } else {
        emit_vector_widen_rn_rm_ternary_op_zx(ctx, |ctx, op1, op2, op3| {
            let prod = ctx.multiply(op2, op3);
            ctx.subtract(op1, prod)
        });
    }
}

pub fn emit_umull_v(ctx: &mut ArmEmitterCtx) {
    // C# Umull_V has no SSE path  always the widen fallback.
    emit_vector_widen_rn_rm_binary_op_zx(ctx, |ctx, op1, op2| ctx.multiply(op1, op2));
}

// Smlal_Ve / Smlsl_Ve / Smull_Ve / Umlal_Ve / Umlsl_Ve / Umull_Ve
pub fn emit_smlal_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_ternary_op_by_elem_sx(ctx, |ctx, op1, op2, op3| {
        let prod = ctx.multiply(op2, op3);
        ctx.add(op1, prod)
    });
}

pub fn emit_smlsl_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_ternary_op_by_elem_sx(ctx, |ctx, op1, op2, op3| {
        let prod = ctx.multiply(op2, op3);
        ctx.subtract(op1, prod)
    });
}

pub fn emit_smull_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_binary_op_by_elem_sx(ctx, |ctx, op1, op2| ctx.multiply(op1, op2));
}

pub fn emit_umlal_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_ternary_op_by_elem_zx(ctx, |ctx, op1, op2, op3| {
        let prod = ctx.multiply(op2, op3);
        ctx.add(op1, prod)
    });
}

pub fn emit_umlsl_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_ternary_op_by_elem_zx(ctx, |ctx, op1, op2, op3| {
        let prod = ctx.multiply(op2, op3);
        ctx.subtract(op1, prod)
    });
}

pub fn emit_umull_ve(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_binary_op_by_elem_zx(ctx, |ctx, op1, op2| ctx.multiply(op1, op2));
}

// Sqabs_S / Sqabs_V / Sqneg_S / Sqneg_V
pub fn emit_sqabs_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_unary_op_sx(
        ctx,
        |ctx, op1| emit_abs(ctx, op1),
        SaturatingFlags(SaturatingFlags::SCALAR.0 | SaturatingFlags::SIGNED.0),
    );
}

pub fn emit_sqabs_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_unary_op_sx(
        ctx,
        |ctx, op1| emit_abs(ctx, op1),
        SaturatingFlags(SaturatingFlags::SIGNED.0),
    );
}

pub fn emit_sqneg_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_unary_op_sx(
        ctx,
        |ctx, op1| ctx.negate(op1),
        SaturatingFlags(SaturatingFlags::SCALAR.0 | SaturatingFlags::SIGNED.0),
    );
}

pub fn emit_sqneg_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_unary_op_sx(
        ctx,
        |ctx, op1| ctx.negate(op1),
        SaturatingFlags(SaturatingFlags::SIGNED.0),
    );
}

// Sqadd_S / Sqadd_V / Sqsub_S / Sqsub_V / Suqadd_S / Suqadd_V / Usqadd_S / Usqadd_V
pub fn emit_sqadd_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(
            SaturatingFlags::SCALAR.0 | SaturatingFlags::SIGNED.0 | SaturatingFlags::ADD.0,
        ),
    );
}

pub fn emit_sqadd_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(SaturatingFlags::SIGNED.0 | SaturatingFlags::ADD.0),
    );
}

pub fn emit_sqsub_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(
            SaturatingFlags::SCALAR.0 | SaturatingFlags::SIGNED.0 | SaturatingFlags::SUB.0,
        ),
    );
}

pub fn emit_sqsub_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(SaturatingFlags::SIGNED.0 | SaturatingFlags::SUB.0),
    );
}

// Suqadd_S / Suqadd_V
// C# Suqadd = EmitScalarSaturatingBinaryOpSx(flags: Accumulate)
// → Scalar | Signed | Accumulate.
pub fn emit_suqadd_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(
            SaturatingFlags::SCALAR.0 | SaturatingFlags::SIGNED.0 | SaturatingFlags::ACCUMULATE.0,
        ),
    );
}

pub fn emit_suqadd_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(SaturatingFlags::SIGNED.0 | SaturatingFlags::ACCUMULATE.0),
    );
}

// Usqadd_S / Usqadd_V
// C# Usqadd = EmitScalarSaturatingBinaryOpZx(SaturatingFlags.Accumulate)
// → Scalar | Accumulate (no Signed).
pub fn emit_usqadd_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(SaturatingFlags::SCALAR.0 | SaturatingFlags::ACCUMULATE.0),
    );
}

pub fn emit_usqadd_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(ctx, None, SaturatingFlags(SaturatingFlags::ACCUMULATE.0));
}

// Sqdmulh_S / Sqdmulh_V / Sqdmulh_Ve / Sqrdmulh_S / Sqrdmulh_V / Sqrdmulh_Ve
pub fn emit_sqdmulh_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        Some(|ctx, op1, op2| emit_doubling_multiply_high_half(ctx, op1, op2, false)),
        SaturatingFlags(SaturatingFlags::SCALAR.0 | SaturatingFlags::SIGNED.0),
    );
}

pub fn emit_sqdmulh_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        Some(|ctx, op1, op2| emit_doubling_multiply_high_half(ctx, op1, op2, false)),
        SaturatingFlags(SaturatingFlags::SIGNED.0),
    );
}

pub fn emit_sqrdmulh_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        Some(|ctx, op1, op2| emit_doubling_multiply_high_half(ctx, op1, op2, true)),
        SaturatingFlags(SaturatingFlags::SCALAR.0 | SaturatingFlags::SIGNED.0),
    );
}

pub fn emit_sqrdmulh_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        Some(|ctx, op1, op2| emit_doubling_multiply_high_half(ctx, op1, op2, true)),
        SaturatingFlags(SaturatingFlags::SIGNED.0),
    );
}

// Sqxtn_S / Sqxtn_V / Sqxtun_S / Sqxtun_V / Uqxtn_S / Uqxtn_V
pub fn emit_sqxtn_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_narrow_op(ctx, SaturatingNarrowFlags::SCALAR_SX_SX);
}

pub fn emit_sqxtn_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_narrow_op(ctx, SaturatingNarrowFlags::VECTOR_SX_SX);
}

pub fn emit_sqxtun_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_narrow_op(ctx, SaturatingNarrowFlags::SCALAR_SX_ZX);
}

pub fn emit_sqxtun_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_narrow_op(ctx, SaturatingNarrowFlags::VECTOR_SX_ZX);
}

pub fn emit_uqxtn_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_narrow_op(ctx, SaturatingNarrowFlags::SCALAR_ZX_ZX);
}

pub fn emit_uqxtn_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_narrow_op(ctx, SaturatingNarrowFlags::VECTOR_ZX_ZX);
}

// Sub_S / Sub_V / Subhn_V
pub fn emit_sub_s(ctx: &mut ArmEmitterCtx) {
    emit_scalar_binary_op_zx(ctx, |ctx, op1, op2| ctx.subtract(op1, op2));
}

pub fn emit_sub_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    let sub_inst = X86_PSUB_INSTRUCTION[op.size as usize];
    let mut res = ctx.emit_intrinsic(sub_inst, &[n, m]);
    if op.reg_size.bits() == 64 {
        res = ctx.vector_zero_upper64(res);
    }
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_subhn_v(ctx: &mut ArmEmitterCtx) {
    emit_high_narrow(ctx, |ctx, op1, op2| ctx.subtract(op1, op2), false);
}

// Ssubl_V / Usubl_V
pub fn emit_ssubl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let mut n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVSX_INSTRUCTION[op.size as usize];
    n = ctx.emit_intrinsic(mov_inst, &[n]);
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let sub_inst = X86_PSUB_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(sub_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

pub fn emit_usubl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let mut n = get_vec(ctx, op.rn);
    let mut m = get_vec(ctx, op.rm);
    if op.reg_size.bits() == 128 {
        let const_8 = ctx.ctx.operands.constant_i32(8);
        n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
        m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
    }
    let mov_inst = X86_PMOVZX_INSTRUCTION[op.size as usize];
    n = ctx.emit_intrinsic(mov_inst, &[n]);
    m = ctx.emit_intrinsic(mov_inst, &[m]);
    let sub_inst = X86_PSUB_INSTRUCTION[(op.size + 1) as usize];
    let res = ctx.emit_intrinsic(sub_inst, &[n, m]);
    let rd = get_vec(ctx, op.rd);
    ctx.copy_to(rd, res);
}

// Uaba_V / Uabal_V
pub fn emit_uaba_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_ternary_op_zx(ctx, |ctx, op1, op2, op3| {
        let diff = ctx.subtract(op2, op3);
        let abs_diff = emit_abs(ctx, diff);
        ctx.add(op1, abs_diff)
    });
}

pub fn emit_uabal_v(ctx: &mut ArmEmitterCtx) {
    emit_vector_widen_rn_rm_ternary_op_zx(ctx, |ctx, op1, op2, op3| {
        let diff = ctx.subtract(op2, op3);
        let abs_diff = emit_abs(ctx, diff);
        ctx.add(op1, abs_diff)
    });
}

// Uabd_V / Uabdl_V
pub fn emit_uabd_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    emit_sse41_vector_uabd_op(ctx, n, m, false);
}

pub fn emit_uabdl_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    // C#: UseSse41 && op.Size < 2.
    if optimizations::use_sse41() && op.size < 2 {
        let mut n = get_vec(ctx, op.rn);
        let mut m = get_vec(ctx, op.rm);
        if op.reg_size.bits() == 128 {
            let const_8 = ctx.ctx.operands.constant_i32(8);
            n = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[n, const_8]);
            m = ctx.emit_intrinsic(Intrinsic::X86Psrldq, &[m, const_8]);
        }
        let mov_inst = if op.size == 0 {
            Intrinsic::X86Pmovzxbw
        } else {
            Intrinsic::X86Pmovzxwd
        };
        n = ctx.emit_intrinsic(mov_inst, &[n]);
        m = ctx.emit_intrinsic(mov_inst, &[m]);
        emit_sse41_vector_uabd_op(ctx, n, m, true);
    } else {
        emit_vector_widen_rn_rm_binary_op_zx(ctx, |ctx, op1, op2| {
            let diff = ctx.subtract(op1, op2);
            emit_abs(ctx, diff)
        });
    }
}

// Urhadd_V
pub fn emit_urhadd_v(ctx: &mut ArmEmitterCtx) {
    let op = decode_simd_reg_op(ctx);
    let n = get_vec(ctx, op.rn);
    let m = get_vec(ctx, op.rm);
    if op.size < 2 {
        let avg_inst = if op.size == 0 {
            Intrinsic::X86Pavgb
        } else {
            Intrinsic::X86Pavgw
        };
        let res = ctx.emit_intrinsic(avg_inst, &[n, m]);
        if op.reg_size.bits() == 64 {
            let rd = get_vec(ctx, op.rd);
            let tmp = ctx.vector_zero_upper64(res);
            ctx.copy_to(rd, tmp);
        } else {
            let rd = get_vec(ctx, op.rd);
            ctx.copy_to(rd, res);
        }
    } else {
        emit_vector_binary_op_zx(ctx, |ctx, op1, op2| {
            let sum = ctx.add(op1, op2);
            let const_1 = ctx.ctx.operands.constant_i64(1);
            let sum = ctx.add(sum, const_1);
            let const_1 = ctx.ctx.operands.constant_i32(1);
            ctx.shift_right_ui(sum, const_1)
        });
    }
}

// Sqdmulh_Ve (by-element saturating doubling multiply high, no round)
pub fn emit_sqdmulh_ve(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        Some(|ctx, n, m| emit_doubling_multiply_high_half(ctx, n, m, false)),
        SaturatingFlags(SaturatingFlags::BY_ELEM.0 | SaturatingFlags::SIGNED.0),
    );
}

// Sqrdmulh_Ve (by-element saturating rounding doubling multiply high)
pub fn emit_sqrdmulh_ve(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        Some(|ctx, n, m| emit_doubling_multiply_high_half(ctx, n, m, true)),
        SaturatingFlags(SaturatingFlags::BY_ELEM.0 | SaturatingFlags::SIGNED.0),
    );
}

// Uqadd_S / Uqadd_V (unsigned saturating add)
pub fn emit_uqadd_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(SaturatingFlags::SCALAR.0 | SaturatingFlags::ADD.0),
    );
}

pub fn emit_uqadd_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(ctx, None, SaturatingFlags(SaturatingFlags::ADD.0));
}

// Uqsub_S / Uqsub_V (unsigned saturating subtract)
pub fn emit_uqsub_s(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(
        ctx,
        None,
        SaturatingFlags(SaturatingFlags::SCALAR.0 | SaturatingFlags::SUB.0),
    );
}

pub fn emit_uqsub_v(ctx: &mut ArmEmitterCtx) {
    emit_saturating_binary_op(ctx, None, SaturatingFlags(SaturatingFlags::SUB.0));
}
