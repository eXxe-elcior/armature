// Port of Instructions/InstEmitMemory.cs (A64).
//
// C# leans on opcode inheritance here: OpCodeMemPair/OpCodeSimdMemImm/
// OpCodeSimdMemPair all derive OpCodeMem(Imm), so `case OpCodeMemImm` and
// `CurrOp is OpCodeMemImm` catch the whole family. The flat OpCodeKind
// has no inheritance, so every match below spells the family out.

use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::operand::OperandIndex;

use super::context::ArmEmitterCtx;
use super::helpers::{get_extended_m, get_int_or_sp, set_int_or_sp, set_int_or_zr};
use super::memory_helpers::{emit_load_sx32, emit_load_sx64, emit_load_zx, emit_store};

pub fn emit_adr(ctx: &mut ArmEmitterCtx) {
    let (rd, immediate) = adr_op(ctx);
    let address = ctx.curr_op().address;

    let value = ctx
        .ctx
        .operands
        .constant_i64(address.wrapping_add(immediate as u64) as i64);
    set_int_or_zr(ctx, rd, value);
}

pub fn emit_adrp(ctx: &mut ArmEmitterCtx) {
    let (rd, immediate) = adr_op(ctx);
    let address = (ctx.curr_op().address & !0xfffu64).wrapping_add((immediate as u64) << 12);

    let value = ctx.ctx.operands.constant_i64(address as i64);
    set_int_or_zr(ctx, rd, value);
}

fn adr_op(ctx: &ArmEmitterCtx) -> (crate::registers::GpRegisterIndex, i64) {
    match ctx.curr_op().kind {
        OpCodeKind::Adr { rd, immediate } => (rd, immediate),
        _ => panic!("expected Adr opcode, got {:?}", ctx.curr_op().kind),
    }
}

pub fn emit_ldr(ctx: &mut ArmEmitterCtx) {
    emit_ldr_inner(ctx, false);
}

pub fn emit_ldrs(ctx: &mut ArmEmitterCtx) {
    emit_ldr_inner(ctx, true);
}

// C# private EmitLdr(context, signed). OpCodeMem base fields (Rt, Size,
// Extend64) across the kinds that reach Ldr/Str.
fn mem_op_fields(ctx: &ArmEmitterCtx) -> (u8, i32, bool) {
    match ctx.curr_op().kind {
        OpCodeKind::MemImm {
            rt, size, extend64, ..
        } => (rt.idx(), size as i32, extend64),
        OpCodeKind::MemReg {
            rt, size, extend64, ..
        } => (rt.idx(), size as i32, extend64),
        // C# OpCodeSimdMemImm/Reg leave Extend64 = false.
        OpCodeKind::SimdMemImm { rt, size, .. } => (rt.idx(), size as i32, false),
        OpCodeKind::SimdMemReg { rt, size, .. } => (rt.idx(), size as i32, false),
        _ => panic!("expected Mem opcode, got {:?}", ctx.curr_op().kind),
    }
}

fn emit_ldr_inner(ctx: &mut ArmEmitterCtx, signed: bool) {
    let (rt, size, extend64) = mem_op_fields(ctx);

    let address = get_address(ctx, 0);

    if signed && extend64 {
        emit_load_sx64(ctx, address, rt, size);
    } else if signed {
        emit_load_sx32(ctx, address, rt, size);
    } else {
        emit_load_zx(ctx, address, rt, size);
    }

    emit_wback_if_needed(ctx, address);
}

pub fn emit_ldr_literal(ctx: &mut ArmEmitterCtx) {
    // C# IOpCodeLit: OpCodeMemLit + OpCodeSimdMemLit (Signed/Prefetch
    // false on the SIMD variant).
    let (rt, size, immediate, signed, prefetch) = match ctx.curr_op().kind {
        OpCodeKind::MemLit {
            rt,
            size,
            immediate,
            signed,
            prefetch,
        } => (rt.idx(), size as i32, immediate, signed, prefetch),
        OpCodeKind::SimdMemLit {
            rt,
            size,
            immediate,
        } => (rt.idx(), size as i32, immediate, false, false),
        _ => panic!("expected MemLit opcode, got {:?}", ctx.curr_op().kind),
    };

    if prefetch {
        return;
    }

    let address = ctx.ctx.operands.constant_i64(immediate);

    if signed {
        emit_load_sx64(ctx, address, rt, size);
    } else {
        emit_load_zx(ctx, address, rt, size);
    }
}

pub fn emit_ldp(ctx: &mut ArmEmitterCtx) {
    let (rt, rt2, size, extend64) = match ctx.curr_op().kind {
        OpCodeKind::MemPair {
            rt,
            rt2,
            size,
            extend64,
            ..
        } => (rt.idx(), rt2.idx(), size as i32, extend64),
        // C# OpCodeSimdMemPair leaves Extend64 = false.
        OpCodeKind::SimdMemPair { rt, rt2, size, .. } => (rt.idx(), rt2.idx(), size as i32, false),
        _ => panic!("expected MemPair opcode, got {:?}", ctx.curr_op().kind),
    };

    let address = get_address(ctx, 0);
    let address2 = get_address(ctx, 1i64 << size);

    let emit_one = |ctx: &mut ArmEmitterCtx, rt: u8, ld_addr: OperandIndex| {
        if extend64 {
            emit_load_sx64(ctx, ld_addr, rt, size);
        } else {
            emit_load_zx(ctx, ld_addr, rt, size);
        }
    };

    emit_one(ctx, rt, address);
    emit_one(ctx, rt2, address2);

    emit_wback_if_needed(ctx, address);
}

pub fn emit_str(ctx: &mut ArmEmitterCtx) {
    let (rt, size, _) = mem_op_fields(ctx);

    let address = get_address(ctx, 0);

    emit_store(ctx, address, rt, size);

    emit_wback_if_needed(ctx, address);
}

pub fn emit_stp(ctx: &mut ArmEmitterCtx) {
    let (rt, rt2, size) = match ctx.curr_op().kind {
        OpCodeKind::MemPair { rt, rt2, size, .. } => (rt.idx(), rt2.idx(), size as i32),
        OpCodeKind::SimdMemPair { rt, rt2, size, .. } => (rt.idx(), rt2.idx(), size as i32),
        _ => panic!("expected MemPair opcode, got {:?}", ctx.curr_op().kind),
    };

    let address = get_address(ctx, 0);
    let address2 = get_address(ctx, 1i64 << size);

    emit_store(ctx, address, rt, size);
    emit_store(ctx, address2, rt2, size);

    emit_wback_if_needed(ctx, address);
}

// C# private GetAddress(context, addend = 0).
fn get_address(ctx: &mut ArmEmitterCtx, addend: i64) -> OperandIndex {
    match ctx.curr_op().kind {
        // C# `case OpCodeMemImm op`  also catches the derived
        // OpCodeMemPair / OpCodeSimdMemImm / OpCodeSimdMemPair.
        OpCodeKind::MemImm {
            rn,
            immediate,
            post_idx,
            ..
        }
        | OpCodeKind::MemPair {
            rn,
            immediate,
            post_idx,
            ..
        }
        | OpCodeKind::SimdMemImm {
            rn,
            immediate,
            post_idx,
            ..
        }
        | OpCodeKind::SimdMemPair {
            rn,
            immediate,
            post_idx,
            ..
        } => {
            let n = get_int_or_sp(ctx, rn);
            let mut address = ctx.copy(n);

            // Pre-indexing.
            if !post_idx {
                let imm = ctx.ctx.operands.constant_i64(immediate + addend);
                address = ctx.add(address, imm);
            } else if addend != 0 {
                let imm = ctx.ctx.operands.constant_i64(addend);
                address = ctx.add(address, imm);
            }

            address
        }

        // C# `case OpCodeMemReg op`  also catches OpCodeSimdMemReg.
        OpCodeKind::MemReg {
            rn,
            rm,
            shift,
            size,
            int_type,
            ..
        }
        | OpCodeKind::SimdMemReg {
            rn,
            rm,
            shift,
            size,
            int_type,
            ..
        } => {
            let n = get_int_or_sp(ctx, rn);

            let mut m = get_extended_m(ctx, rm, int_type);

            if shift {
                let size_op = ctx.ctx.operands.constant_i32(size as i32);
                m = ctx.shift_left(m, size_op);
            }

            let mut address = ctx.add(n, m);

            if addend != 0 {
                let imm = ctx.ctx.operands.constant_i64(addend);
                address = ctx.add(address, imm);
            }

            address
        }

        _ => panic!("get_address: unexpected opcode {:?}", ctx.curr_op().kind),
    }
}

// C# private EmitWBackIfNeeded: `CurrOp is OpCodeMemImm op && op.WBack`
//  the derived kinds again.
fn emit_wback_if_needed(ctx: &mut ArmEmitterCtx, mut address: OperandIndex) {
    let (rn, immediate, wback, post_idx) = match ctx.curr_op().kind {
        OpCodeKind::MemImm {
            rn,
            immediate,
            wback,
            post_idx,
            ..
        }
        | OpCodeKind::MemPair {
            rn,
            immediate,
            wback,
            post_idx,
            ..
        }
        | OpCodeKind::SimdMemImm {
            rn,
            immediate,
            wback,
            post_idx,
            ..
        }
        | OpCodeKind::SimdMemPair {
            rn,
            immediate,
            wback,
            post_idx,
            ..
        } => (rn, immediate, wback, post_idx),
        _ => return,
    };

    if wback {
        if post_idx {
            let imm = ctx.ctx.operands.constant_i64(immediate);
            address = ctx.add(address, imm);
        }

        set_int_or_sp(ctx, rn, address);
    }
}
