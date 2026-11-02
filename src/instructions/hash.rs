// Port of Instructions/InstEmitHash.cs (A64 half  the A32 region is
// excluded with ARM32).

use crate::decoder::opcode_kind::OpCodeKind;
use crate::registers::GpRegisterIndex;

use super::context::ArmEmitterCtx;
use super::hash_helpers::emit_crc32;
use super::helpers::{get_int_or_zr, set_int_or_zr};

const BYTE_SIZE_LOG2: i32 = 0;
const HWORD_SIZE_LOG2: i32 = 1;
const WORD_SIZE_LOG2: i32 = 2;
const DWORD_SIZE_LOG2: i32 = 3;

pub fn emit_crc32b(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, BYTE_SIZE_LOG2, false);
}

pub fn emit_crc32h(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, HWORD_SIZE_LOG2, false);
}

pub fn emit_crc32w(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, WORD_SIZE_LOG2, false);
}

pub fn emit_crc32x(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, DWORD_SIZE_LOG2, false);
}

pub fn emit_crc32cb(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, BYTE_SIZE_LOG2, true);
}

pub fn emit_crc32ch(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, HWORD_SIZE_LOG2, true);
}

pub fn emit_crc32cw(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, WORD_SIZE_LOG2, true);
}

pub fn emit_crc32cx(ctx: &mut ArmEmitterCtx) {
    emit_crc32_call(ctx, DWORD_SIZE_LOG2, true);
}

// C# casts CurrOp to OpCodeAluBinary.
fn alu_binary_op(ctx: &ArmEmitterCtx) -> (GpRegisterIndex, GpRegisterIndex, GpRegisterIndex) {
    match ctx.curr_op().kind {
        OpCodeKind::AluBinary { rd, rn, rm, .. } => (rd, rn, rm),
        _ => panic!("expected AluBinary opcode, got {:?}", ctx.curr_op().kind),
    }
}

fn emit_crc32_call(ctx: &mut ArmEmitterCtx, size: i32, c: bool) {
    let (rd, rn, rm) = alu_binary_op(ctx);

    let n = get_int_or_zr(ctx, rn);
    let m = get_int_or_zr(ctx, rm);

    let d = emit_crc32(ctx, n, m, size, c);

    set_int_or_zr(ctx, rd, d);
}
