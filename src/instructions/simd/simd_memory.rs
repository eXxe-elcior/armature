// Port of InstEmitSimdMemory.cs
// Host x86-64 only: single-structure SIMD memory operations (LD/ST multiple
// single-element, and LD/ST single structure).  Non-temporal and exclusive
// variants are not part of this file in the C# either.

#![allow(non_snake_case)]

use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::helpers::{get_int_or_sp, get_int_or_zr, get_vec, set_int_or_sp};
use crate::instructions::memory_helpers::{emit_load_simd, emit_store_simd};
use crate::registers::{GpRegisterIndex, VecRegisterIndex};

use crate::decoder::RegisterSize;
use crate::decoder::opcode_kind::OpCodeKind;

// Public entry points
pub fn emit_ld__vms(ctx: &mut ArmEmitterCtx) {
    emit_simd_mem_ms(ctx, true);
}

pub fn emit_ld__vss(ctx: &mut ArmEmitterCtx) {
    emit_simd_mem_ss(ctx, true);
}

pub fn emit_st__vms(ctx: &mut ArmEmitterCtx) {
    emit_simd_mem_ms(ctx, false);
}

pub fn emit_st__vss(ctx: &mut ArmEmitterCtx) {
    emit_simd_mem_ss(ctx, false);
}

// LD/ST multiple single-element structure
fn emit_simd_mem_ms(ctx: &mut ArmEmitterCtx, is_load: bool) {
    let (rt, rn, rm, size, wback, reps, s_elems, elems, reg_size) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdMemMs {
                rt,
                rn,
                rm,
                size,
                wback,
                reps,
                s_elems,
                elems,
            } => (
                *rt,
                *rn,
                *rm,
                *size,
                *wback,
                *reps,
                *s_elems,
                *elems,
                op.reg_size,
            ),
            _ => unreachable!(),
        }
    };

    let n = get_int_or_sp(ctx, rn);
    let mut offset: i64 = 0;

    for rep in 0..reps {
        for elem in 0..elems {
            for s_elem in 0..s_elems {
                let rtt = (rt.idx() + rep + s_elem) & 0x1f;
                let tt = get_vec(ctx, VecRegisterIndex::from(rtt));
                let offset_op = ctx.ctx.operands.constant_i64(offset);
                let address = ctx.add(n, offset_op);

                if is_load {
                    emit_load_simd(ctx, address, tt, rtt, elem as i32, size as i32);
                    if reg_size == RegisterSize::Simd64 && elem == elems - 1 {
                        let zeroed = ctx.vector_zero_upper64(tt);
                        ctx.copy_to(tt, zeroed);
                    }
                } else {
                    emit_store_simd(ctx, address, rtt, elem as i32, size as i32);
                }

                offset += 1i64 << (size as u32);
            }
        }
    }

    if wback {
        emit_simd_mem_wback(ctx, offset, rn, rm);
    }
}

// LD/ST single structure
fn emit_simd_mem_ss(ctx: &mut ArmEmitterCtx, is_load: bool) {
    let (rt, rn, rm, size, wback, s_elems, index, replicate, reg_size, bytes_count) = {
        let op = ctx.curr_op();
        match &op.kind {
            OpCodeKind::SimdMemSs {
                rt,
                rn,
                rm,
                size,
                wback,
                s_elems,
                index,
                replicate,
            } => (
                *rt,
                *rn,
                *rm,
                *size,
                *wback,
                *s_elems,
                *index,
                *replicate,
                op.reg_size,
                op.bytes(),
            ),
            _ => unreachable!(),
        }
    };

    let n = get_int_or_sp(ctx, rn);
    let mut offset: i64 = 0;

    if replicate {
        debug_assert!(is_load, "Replicate mode is not valid for stores.");
        let elems = (bytes_count >> size) as u8;

        for s_elem in 0..s_elems {
            let rt_reg = (rt.idx() + s_elem) & 0x1f;
            let t = get_vec(ctx, VecRegisterIndex::from(rt_reg));
            let offset_op = ctx.ctx.operands.constant_i64(offset);
            let address = ctx.add(n, offset_op);

            for idx in 0..elems {
                emit_load_simd(ctx, address, t, rt_reg, idx as i32, size as i32);
            }

            if reg_size == RegisterSize::Simd64 {
                let zeroed = ctx.vector_zero_upper64(t);
                ctx.copy_to(t, zeroed);
            }

            offset += 1i64 << (size as u32);
        }
    } else {
        for s_elem in 0..s_elems {
            let rt_reg = (rt.idx() + s_elem) & 0x1f;
            let t = get_vec(ctx, VecRegisterIndex::from(rt_reg));
            let offset_op = ctx.ctx.operands.constant_i64(offset);
            let address = ctx.add(n, offset_op);

            if is_load {
                emit_load_simd(ctx, address, t, rt_reg, index as i32, size as i32);
            } else {
                emit_store_simd(ctx, address, rt_reg, index as i32, size as i32);
            }

            offset += 1i64 << (size as u32);
        }
    }

    if wback {
        emit_simd_mem_wback(ctx, offset, rn, rm);
    }
}

// Writeback
fn emit_simd_mem_wback(
    ctx: &mut ArmEmitterCtx,
    offset: i64,
    rn: GpRegisterIndex,
    rm: GpRegisterIndex,
) {
    let n = get_int_or_sp(ctx, rn);
    let m = if rm.idx() != 0x1f {
        get_int_or_zr(ctx, rm)
    } else {
        ctx.ctx.operands.constant_i64(offset)
    };
    let sum = ctx.add(n, m);
    set_int_or_sp(ctx, rn, sum);
}
