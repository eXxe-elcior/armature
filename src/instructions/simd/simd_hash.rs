// Port of InstEmitSimdHash.cs and InstEmitSimdHashHelper.cs.
// SHA256 uses SHA-NI; its software fallback is unimplemented. SHA1 uses software helpers.

use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::helpers::get_vec;
use crate::ir::operand::OperandIndex;
use crate::ir::types::Intrinsic;
use crate::optimizations;

use crate::decoder::opcode_kind::OpCodeKind;

// SHA1 (SoftFallback only  upstream has no x86 SHA1 fast path)
use crate::instructions::native_function::NativeFn;
use crate::ir::types::OperandType;

pub fn emit_sha1c_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdReg { rd, rn, rm, .. } => (*rd, *rn, *rm),
            _ => unreachable!(),
        }
    };

    let d = get_vec(ctx, rd);
    let n = get_vec(ctx, rn);
    let ne = ctx.vector_extract(OperandType::I32, n, 0);
    let m = get_vec(ctx, rm);

    let res = ctx.call(NativeFn::HashChoose, &[d, ne, m]).unwrap();

    let d_op = get_vec(ctx, rd);
    ctx.copy_to(d_op, res);
}

pub fn emit_sha1h_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::Simd { rd, rn, .. } => (*rd, *rn),
            _ => unreachable!(),
        }
    };

    let n = get_vec(ctx, rn);
    let ne = ctx.vector_extract(OperandType::I32, n, 0);

    let res = ctx.call(NativeFn::FixedRotate, &[ne]).unwrap();

    let scalar = ctx.vector_create_scalar(res);
    let d_op = get_vec(ctx, rd);
    ctx.copy_to(d_op, scalar);
}

pub fn emit_sha1m_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdReg { rd, rn, rm, .. } => (*rd, *rn, *rm),
            _ => unreachable!(),
        }
    };

    let d = get_vec(ctx, rd);
    let n = get_vec(ctx, rn);
    let ne = ctx.vector_extract(OperandType::I32, n, 0);
    let m = get_vec(ctx, rm);

    let res = ctx.call(NativeFn::HashMajority, &[d, ne, m]).unwrap();

    let d_op = get_vec(ctx, rd);
    ctx.copy_to(d_op, res);
}

pub fn emit_sha1p_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdReg { rd, rn, rm, .. } => (*rd, *rn, *rm),
            _ => unreachable!(),
        }
    };

    let d = get_vec(ctx, rd);
    let n = get_vec(ctx, rn);
    let ne = ctx.vector_extract(OperandType::I32, n, 0);
    let m = get_vec(ctx, rm);

    let res = ctx.call(NativeFn::HashParity, &[d, ne, m]).unwrap();

    let d_op = get_vec(ctx, rd);
    ctx.copy_to(d_op, res);
}

pub fn emit_sha1su0_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdReg { rd, rn, rm, .. } => (*rd, *rn, *rm),
            _ => unreachable!(),
        }
    };

    let d = get_vec(ctx, rd);
    let n = get_vec(ctx, rn);
    let m = get_vec(ctx, rm);

    let res = ctx.call(NativeFn::Sha1SchedulePart1, &[d, n, m]).unwrap();

    let d_op = get_vec(ctx, rd);
    ctx.copy_to(d_op, res);
}

pub fn emit_sha1su1_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::Simd { rd, rn, .. } => (*rd, *rn),
            _ => unreachable!(),
        }
    };

    let d = get_vec(ctx, rd);
    let n = get_vec(ctx, rn);

    let res = ctx.call(NativeFn::Sha1SchedulePart2, &[d, n]).unwrap();

    let d_op = get_vec(ctx, rd);
    ctx.copy_to(d_op, res);
}

// SHA256
pub fn emit_sha256h_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdReg { rd, rn, rm, .. } => (*rd, *rn, *rm),
            _ => unreachable!(),
        }
    };
    let d_op = get_vec(ctx, rd);
    let n_op = get_vec(ctx, rn);
    let m_op = get_vec(ctx, rm);
    let res = emit_sha256h(ctx, d_op, n_op, m_op, false);
    ctx.copy_to(d_op, res);
}

pub fn emit_sha256h2_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdReg { rd, rn, rm, .. } => (*rd, *rn, *rm),
            _ => unreachable!(),
        }
    };
    let d_op = get_vec(ctx, rd);
    let n_op = get_vec(ctx, rn);
    let m_op = get_vec(ctx, rm);
    // C# passes (n, d, m) for part2  args swapped vs Sha256h.
    let res = emit_sha256h(ctx, n_op, d_op, m_op, true);
    ctx.copy_to(d_op, res);
}

pub fn emit_sha256su0_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::Simd { rd, rn, .. } => (*rd, *rn),
            _ => unreachable!(),
        }
    };
    let d_op = get_vec(ctx, rd);
    let n_op = get_vec(ctx, rn);
    let res = emit_sha256su0(ctx, d_op, n_op);
    ctx.copy_to(d_op, res);
}

pub fn emit_sha256su1_v(ctx: &mut ArmEmitterCtx) {
    let (rd, rn, rm) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdReg { rd, rn, rm, .. } => (*rd, *rn, *rm),
            _ => unreachable!(),
        }
    };
    let d_op = get_vec(ctx, rd);
    let n_op = get_vec(ctx, rn);
    let m_op = get_vec(ctx, rm);
    let res = emit_sha256su1(ctx, d_op, n_op, m_op);
    ctx.copy_to(d_op, res);
}

// Private helpers (port of InstEmitSimdHashHelper)
fn emit_sha256h(
    ctx: &mut ArmEmitterCtx,
    x: OperandIndex,
    y: OperandIndex,
    w: OperandIndex,
    part2: bool,
) -> OperandIndex {
    if optimizations::use_sha() {
        let imm_bb = ctx.ctx.operands.constant_i32(0xbb);
        let imm_11 = ctx.ctx.operands.constant_i32(0x11);
        let imm_sel = ctx
            .ctx
            .operands
            .constant_i32(if part2 { 0x11 } else { 0xbb });

        let src1 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[y, x, imm_bb]);
        let src2 = ctx.emit_intrinsic(Intrinsic::X86Shufps, &[y, x, imm_11]);
        let w2 = ctx.emit_intrinsic(Intrinsic::X86Punpckhqdq, &[w, w]);

        let round2 = ctx.emit_intrinsic(Intrinsic::X86Sha256Rnds2, &[src1, src2, w]);
        let round4 = ctx.emit_intrinsic(Intrinsic::X86Sha256Rnds2, &[src2, round2, w2]);

        ctx.emit_intrinsic(Intrinsic::X86Shufps, &[round4, round2, imm_sel])
    } else {
        todo!("emit_sha256h (SoftFallback) for InstEmitSimdHash");
    }
}

fn emit_sha256su0(ctx: &mut ArmEmitterCtx, x: OperandIndex, y: OperandIndex) -> OperandIndex {
    if optimizations::use_sha() {
        ctx.emit_intrinsic(Intrinsic::X86Sha256Msg1, &[x, y])
    } else {
        todo!("emit_sha256su0 (SoftFallback) for InstEmitSimdHash");
    }
}

fn emit_sha256su1(
    ctx: &mut ArmEmitterCtx,
    x: OperandIndex,
    y: OperandIndex,
    z: OperandIndex,
) -> OperandIndex {
    if optimizations::use_sha() && optimizations::use_ssse3() {
        let imm_4 = ctx.ctx.operands.constant_i32(4);
        let extr = ctx.emit_intrinsic(Intrinsic::X86Palignr, &[z, y, imm_4]);
        let tmp = ctx.emit_intrinsic(Intrinsic::X86Paddd, &[extr, x]);
        ctx.emit_intrinsic(Intrinsic::X86Sha256Msg2, &[tmp, z])
    } else {
        todo!("emit_sha256su1 (SoftFallback) for InstEmitSimdHash");
    }
}
