// Port of Instructions/InstEmitMemoryEx.cs (A64).

use crate::decoder::opcode_kind::OpCodeKind;
use crate::ir::types::{MemoryBarrierKind, OperandType};
use crate::registers::GpRegisterIndex;

use super::context::ArmEmitterCtx;
use super::helpers::{get_int_or_sp, get_int_or_zr, set_int_or_zr};
use super::memory_ex_helpers::{emit_clear_exclusive, emit_load_exclusive, emit_store_exclusive};

// C# [Flags] enum AccessType.
#[derive(Clone, Copy)]
struct AccessType(u8);

impl AccessType {
    const ORDERED: AccessType = AccessType(1);
    const EXCLUSIVE: AccessType = AccessType(2);
    const ORDERED_EX: AccessType = AccessType(1 | 2);

    fn ordered(self) -> bool {
        self.0 & Self::ORDERED.0 != 0
    }
    fn exclusive(self) -> bool {
        self.0 & Self::EXCLUSIVE.0 != 0
    }
}

pub fn emit_clrex(ctx: &mut ArmEmitterCtx) {
    emit_clear_exclusive(ctx);
}

pub fn emit_csdb(_ctx: &mut ArmEmitterCtx) {
    // Execute as no-op.
}

pub fn emit_dmb(ctx: &mut ArmEmitterCtx) {
    // CRm[1:0] encodes the barrier type per ARM C6.2.150:
    //   0b11 = SY  (full  stores + loads)
    //   0b01 = ST  (stores only)
    //   0b10 = LD  (loads only)
    //   0b00 = reserved (maps to full for forward compat)
    //
    // ST and LD are no-ops on x86-TSO (guaranteed without barrier).
    // On future ARM host, these map to dmb st / dmb ld respectively.
    // Full is always needed on any host  emitted as MFENCE on x86-64.
    let kind = match ctx.curr_op().kind {
        OpCodeKind::System { crm, .. } => match crm & 0b11 {
            0b01 => MemoryBarrierKind::StoreStore, // ST
            0b10 => MemoryBarrierKind::LoadLoad,   // LD
            _ => MemoryBarrierKind::Full,          // SY (0b11) or reserved (0b00)
        },
        _ => MemoryBarrierKind::Full, // Not a System op (shouldn't happen)
    };
    ctx.memory_barrier(kind);
}
pub fn emit_dsb(ctx: &mut ArmEmitterCtx) {
    // Same memory-ordering semantics as DMB; DSB's extra context-synchronization
    // effect is irrelevant in JIT output (no guest pipeline state to drain).
    emit_dmb(ctx);
}

pub fn emit_ldar(ctx: &mut ArmEmitterCtx) {
    emit_load_ex(ctx, AccessType::ORDERED, false);
}
pub fn emit_ldaxr(ctx: &mut ArmEmitterCtx) {
    emit_load_ex(ctx, AccessType::ORDERED_EX, false);
}
pub fn emit_ldxr(ctx: &mut ArmEmitterCtx) {
    emit_load_ex(ctx, AccessType::EXCLUSIVE, false);
}
pub fn emit_ldxp(ctx: &mut ArmEmitterCtx) {
    emit_load_ex(ctx, AccessType::EXCLUSIVE, true);
}
pub fn emit_ldaxp(ctx: &mut ArmEmitterCtx) {
    emit_load_ex(ctx, AccessType::ORDERED_EX, true);
}

fn mem_ex_op(
    ctx: &ArmEmitterCtx,
) -> (
    GpRegisterIndex,
    GpRegisterIndex,
    GpRegisterIndex,
    GpRegisterIndex,
    i32,
) {
    match ctx.curr_op().kind {
        OpCodeKind::MemEx {
            rt,
            rt2,
            rn,
            rs,
            size,
        } => (rt, rt2, rn, rs, size as i32),
        _ => panic!("expected MemEx opcode, got {:?}", ctx.curr_op().kind),
    }
}

// C# private EmitLoadEx (EmitLdr/EmitLdp collapse into the `pair` flag).
fn emit_load_ex(ctx: &mut ArmEmitterCtx, acc_type: AccessType, pair: bool) {
    let (rt, rt2, rn, _, size) = mem_ex_op(ctx);

    let ordered = acc_type.ordered();
    let exclusive = acc_type.exclusive();

    if ordered {
        emit_barrier(ctx);
    }

    let n = get_int_or_sp(ctx, rn);
    let address = ctx.copy(n);

    if pair {
        // Exclusive loads should be atomic. For pairwise loads, we need to
        // read all the data at once. For a 32-bits pairwise load, we do a
        // simple 64-bits load, for a 128-bits load, we need to call a
        // special method to read 128-bits atomically.
        if size == 2 {
            let value = emit_load_exclusive(ctx, address, exclusive, 3);

            let value_low = ctx.convert_i64_to_i32(value);

            let value_low = ctx.zero_extend32(OperandType::I64, value_low);

            let thirty_two = ctx.ctx.operands.constant_i32(32);
            let value_high = ctx.shift_right_ui(value, thirty_two);

            set_int_or_zr(ctx, rt, value_low);
            set_int_or_zr(ctx, rt2, value_high);
        } else if size == 3 {
            let value = emit_load_exclusive(ctx, address, exclusive, 4);

            let value_low = ctx.vector_extract(OperandType::I64, value, 0);
            let value_high = ctx.vector_extract(OperandType::I64, value, 1);

            set_int_or_zr(ctx, rt, value_low);
            set_int_or_zr(ctx, rt2, value_high);
        } else {
            panic!("Invalid load size of {} bytes.", 1 << size);
        }
    } else {
        // 8, 16, 32 or 64-bits (non-pairwise) load.
        let value = emit_load_exclusive(ctx, address, exclusive, size);

        set_int_or_zr(ctx, rt, value);
    }
}

pub fn emit_prfm(_ctx: &mut ArmEmitterCtx) {
    // Memory Prefetch, execute as no-op.
}

pub fn emit_stlr(ctx: &mut ArmEmitterCtx) {
    emit_store_ex(ctx, AccessType::ORDERED, false);
}
pub fn emit_stlxr(ctx: &mut ArmEmitterCtx) {
    emit_store_ex(ctx, AccessType::ORDERED_EX, false);
}
pub fn emit_stxr(ctx: &mut ArmEmitterCtx) {
    emit_store_ex(ctx, AccessType::EXCLUSIVE, false);
}
pub fn emit_stxp(ctx: &mut ArmEmitterCtx) {
    emit_store_ex(ctx, AccessType::EXCLUSIVE, true);
}
pub fn emit_stlxp(ctx: &mut ArmEmitterCtx) {
    emit_store_ex(ctx, AccessType::ORDERED_EX, true);
}

// C# private EmitStoreEx (EmitStr/EmitStp collapse into the `pair` flag).
fn emit_store_ex(ctx: &mut ArmEmitterCtx, acc_type: AccessType, pair: bool) {
    let (rt, rt2, rn, rs, size) = mem_ex_op(ctx);

    let ordered = acc_type.ordered();
    let exclusive = acc_type.exclusive();

    let n = get_int_or_sp(ctx, rn);
    let address = ctx.copy(n);

    let t = get_int_or_zr(ctx, rt);

    if pair {
        debug_assert!(size == 2 || size == 3, "Invalid size for pairwise store.");

        let t2 = get_int_or_zr(ctx, rt2);

        let value = if size == 2 {
            let thirty_two = ctx.ctx.operands.constant_i32(32);
            let t2_shifted = ctx.shift_left(t2, thirty_two);
            ctx.bitwise_or(t, t2_shifted)
        } else
        /* if size == 3 */
        {
            let zero = ctx.vector_zero();
            let value = ctx.vector_insert(zero, t, 0);
            ctx.vector_insert(value, t2, 1)
        };

        emit_store_exclusive(ctx, address, value, exclusive, size + 1, rs);
    } else {
        emit_store_exclusive(ctx, address, t, exclusive, size, rs);
    }

    if ordered {
        emit_barrier(ctx);
    }
}

fn emit_barrier(ctx: &mut ArmEmitterCtx) {
    ctx.memory_barrier(MemoryBarrierKind::Full);
}
