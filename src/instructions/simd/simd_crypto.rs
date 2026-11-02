// Port of InstEmitSimdCrypto.cs for x86-64 hosts.
// AES requires AES-NI; software fallback paths are unimplemented.

use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::helpers::get_vec;
use crate::ir::types::Intrinsic;
use crate::optimizations;

use crate::decoder::opcode_kind::OpCodeKind;

pub fn emit_aesd_v(ctx: &mut ArmEmitterCtx) {
    let (d, n) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::Simd { rd, rn, .. } => (*rd, *rn),
            _ => unreachable!(),
        }
    };

    let d_op = get_vec(ctx, d);
    let n_op = get_vec(ctx, n);

    if optimizations::use_aesni() {
        let xored = ctx.emit_intrinsic(Intrinsic::X86Xorpd, &[d_op, n_op]);
        let zero = ctx.vector_zero();
        let res = ctx.emit_intrinsic(Intrinsic::X86Aesdeclast, &[xored, zero]);
        ctx.copy_to(d_op, res);
    } else {
        todo!("emit_aesd_v (SoftFallback) for InstEmitSimdCrypto");
    }
}

pub fn emit_aese_v(ctx: &mut ArmEmitterCtx) {
    let (d, n) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::Simd { rd, rn, .. } => (*rd, *rn),
            _ => unreachable!(),
        }
    };

    let d_op = get_vec(ctx, d);
    let n_op = get_vec(ctx, n);

    if optimizations::use_aesni() {
        let xored = ctx.emit_intrinsic(Intrinsic::X86Xorpd, &[d_op, n_op]);
        let zero = ctx.vector_zero();
        let res = ctx.emit_intrinsic(Intrinsic::X86Aesenclast, &[xored, zero]);
        ctx.copy_to(d_op, res);
    } else {
        todo!("emit_aese_v (SoftFallback) for InstEmitSimdCrypto");
    }
}

pub fn emit_aesimc_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::Simd { rd, rn, .. } => (*rd, *rn),
            _ => unreachable!(),
        }
    };

    let n_op = get_vec(ctx, rn);

    if optimizations::use_aesni() {
        let res = ctx.emit_intrinsic(Intrinsic::X86Aesimc, &[n_op]);
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        todo!("emit_aesimc_v (SoftFallback) for InstEmitSimdCrypto");
    }
}

pub fn emit_aesmc_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::Simd { rd, rn, .. } => (*rd, *rn),
            _ => unreachable!(),
        }
    };

    let n_op = get_vec(ctx, rn);

    if optimizations::use_aesni() {
        let zero = ctx.vector_zero();
        let res = ctx.emit_intrinsic(Intrinsic::X86Aesdeclast, &[n_op, zero]);
        let res = ctx.emit_intrinsic(Intrinsic::X86Aesenc, &[res, zero]);
        let rd_op = get_vec(ctx, rd);
        ctx.copy_to(rd_op, res);
    } else {
        todo!("emit_aesmc_v (SoftFallback) for InstEmitSimdCrypto");
    }
}
