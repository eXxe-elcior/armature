// Port of Instructions/InstEmitHashHelper.cs:
// SSE4.2 X86Crc32 (Castagnoli), PCLMULQDQ folding, SoftFallback calls.
//
// CRC32X divergence from the upstream PCLMULQDQ path:
// upstream's EmitCrc32Optimized64 inserts the crc operand with its
// I64 type for CRC32X (sf=1 decode), leaking Wn's upper 32 bits into
// the computation  against the ARM spec, upstream's own soft fallback
// and its SSE4.2 path. We truncate to 32 bits first (the ConvertI64ToI32
// below), matching spec + soft semantics.

use crate::ir::operand::OperandIndex;
use crate::ir::types::{Intrinsic, OperandKind, OperandType};
use crate::optimizations;

use super::context::ArmEmitterCtx;
use crate::instructions::native_function::NativeFn;

pub const CRC32_REV_POLY: u32 = 0xedb88320;
pub const CRC32C_REV_POLY: u32 = 0x82f63b78;

pub fn emit_crc32(
    ctx: &mut ArmEmitterCtx,
    mut crc: OperandIndex,
    mut value: OperandIndex,
    size: i32,
    castagnoli: bool,
) -> OperandIndex {
    debug_assert!(ctx.ctx.operands.get(crc).typ.is_integer());
    debug_assert!(ctx.ctx.operands.get(value).typ.is_integer());
    debug_assert!((0..4).contains(&size));
    debug_assert!(size < 3 || ctx.ctx.operands.get(value).typ == OperandType::I64);

    if castagnoli && optimizations::use_sse42() {
        // The CRC32 instruction does not have an immediate variant, so
        // ensure both inputs are in registers.
        if ctx.ctx.operands.get(value).kind == OperandKind::Constant {
            value = ctx.copy(value);
        }
        if ctx.ctx.operands.get(crc).kind == OperandKind::Constant {
            crc = ctx.copy(crc);
        }

        let op = match size {
            0 => Intrinsic::X86Crc32_8,
            1 => Intrinsic::X86Crc32_16,
            _ => Intrinsic::X86Crc32,
        };

        if size == 3 {
            let r = ctx.emit_intrinsic_long(op, &[crc, value]);
            ctx.convert_i64_to_i32(r)
        } else {
            ctx.emit_intrinsic_int(op, &[crc, value])
        }
    } else if optimizations::use_pclmulqdq() {
        if size == 3 {
            emit_crc32_optimized64(ctx, crc, value, castagnoli)
        } else {
            emit_crc32_optimized(ctx, crc, value, castagnoli, size)
        }
    } else {
        let func = match (size, castagnoli) {
            (0, false) => NativeFn::Crc32b,
            (1, false) => NativeFn::Crc32h,
            (2, false) => NativeFn::Crc32w,
            (3, false) => NativeFn::Crc32x,
            (0, true) => NativeFn::Crc32cb,
            (1, true) => NativeFn::Crc32ch,
            (2, true) => NativeFn::Crc32cw,
            (3, true) => NativeFn::Crc32cx,
            _ => panic!("emit_crc32: size {size} out of range"),
        };

        ctx.call(func, &[crc, value])
            .expect("Crc32 helpers return I32")
    }
}

// C# EmitCrc32Optimized (sizes 0-2; crc is I32  no truncation issue).
fn emit_crc32_optimized(
    ctx: &mut ArmEmitterCtx,
    crc: OperandIndex,
    data: OperandIndex,
    castagnoli: bool,
    size: i32,
) -> OperandIndex {
    let mu: i64 = if castagnoli { 0x0DEA713F1 } else { 0x1F7011641 }; // mu' = floor(x^64/P(x))'
    let polynomial: i64 = if castagnoli { 0x105EC76F0 } else { 0x1DB710641 }; // P'(x) << 1

    let zero_v = ctx.vector_zero();
    let crc_v = ctx.vector_insert(zero_v, crc, 0);

    let data_v = match size {
        0 => {
            let z = ctx.vector_zero();
            ctx.vector_insert8(z, data, 0)
        }
        1 => {
            let z = ctx.vector_zero();
            ctx.vector_insert16(z, data, 0)
        }
        2 => {
            let z = ctx.vector_zero();
            ctx.vector_insert(z, data, 0)
        }
        _ => unreachable!(),
    };

    let bitsize = 8i32 << size;

    let mut tmp = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[crc_v, data_v]);
    let shift = ctx.ctx.operands.constant_i32(64 - bitsize);
    tmp = ctx.emit_intrinsic(Intrinsic::X86Psllq, &[tmp, shift]);
    let mu_v = x86_get_scalar(ctx, mu);
    let zero_c = ctx.ctx.operands.constant_i32(0);
    tmp = ctx.emit_intrinsic(Intrinsic::X86Pclmulqdq, &[tmp, mu_v, zero_c]);
    let poly_v = x86_get_scalar(ctx, polynomial);
    let zero_c = ctx.ctx.operands.constant_i32(0);
    tmp = ctx.emit_intrinsic(Intrinsic::X86Pclmulqdq, &[tmp, poly_v, zero_c]);

    if bitsize < 32 {
        let shift_bytes = ctx.ctx.operands.constant_i32((64 - bitsize) / 8);
        let crc_shifted = ctx.emit_intrinsic(Intrinsic::X86Pslldq, &[crc_v, shift_bytes]);
        tmp = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[tmp, crc_shifted]);
    }

    ctx.vector_extract(OperandType::I32, tmp, 2)
}

// C# EmitCrc32Optimized64  with the Wn-truncation fix (see header).
fn emit_crc32_optimized64(
    ctx: &mut ArmEmitterCtx,
    crc: OperandIndex,
    data: OperandIndex,
    castagnoli: bool,
) -> OperandIndex {
    let mu: i64 = if castagnoli { 0x0DEA713F1 } else { 0x1F7011641 }; // mu' = floor(x^64/P(x))'
    let polynomial: i64 = if castagnoli { 0x105EC76F0 } else { 0x1DB710641 }; // P'(x) << 1

    // DIVERGENCE FIX: upstream inserts `crc` with its I64 type here,
    // leaking Wn[63:32] into the fold. Truncate to W first.
    let crc = if ctx.ctx.operands.get(crc).typ == OperandType::I64 {
        ctx.convert_i64_to_i32(crc)
    } else {
        crc
    };

    let zero_v = ctx.vector_zero();
    let crc_v = ctx.vector_insert(zero_v, crc, 0);
    let zero_v = ctx.vector_zero();
    let data_v = ctx.vector_insert(zero_v, data, 0);

    let tmp0 = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[crc_v, data_v]);
    let four = ctx.ctx.operands.constant_i32(4);
    let res = ctx.emit_intrinsic(Intrinsic::X86Pslldq, &[tmp0, four]);

    let mu_v = x86_get_scalar(ctx, mu);
    let zero_c = ctx.ctx.operands.constant_i32(0);
    let mut tmp = ctx.emit_intrinsic(Intrinsic::X86Pclmulqdq, &[res, mu_v, zero_c]);
    let poly_v = x86_get_scalar(ctx, polynomial);
    let zero_c = ctx.ctx.operands.constant_i32(0);
    tmp = ctx.emit_intrinsic(Intrinsic::X86Pclmulqdq, &[tmp, poly_v, zero_c]);

    tmp = ctx.emit_intrinsic(Intrinsic::X86Pxor, &[tmp, res]);
    let thirty_two = ctx.ctx.operands.constant_i32(32);
    tmp = ctx.emit_intrinsic(Intrinsic::X86Psllq, &[tmp, thirty_two]);

    let mu_v = x86_get_scalar(ctx, mu);
    let one_c = ctx.ctx.operands.constant_i32(1);
    tmp = ctx.emit_intrinsic(Intrinsic::X86Pclmulqdq, &[tmp, mu_v, one_c]);
    let poly_v = x86_get_scalar(ctx, polynomial);
    let zero_c = ctx.ctx.operands.constant_i32(0);
    tmp = ctx.emit_intrinsic(Intrinsic::X86Pclmulqdq, &[tmp, poly_v, zero_c]);

    ctx.vector_extract(OperandType::I32, tmp, 2)
}

// C# InstEmitSimdHelper.X86GetScalar(context, long)  scalar constant
// into a vector register.
fn x86_get_scalar(ctx: &mut ArmEmitterCtx, value: i64) -> OperandIndex {
    let c = ctx.ctx.operands.constant_i64(value);
    ctx.vector_create_scalar(c)
}
