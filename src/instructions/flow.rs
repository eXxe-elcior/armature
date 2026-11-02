// Port of Instructions/InstEmitFlow.cs

use crate::decoder::opcode_kind::OpCodeKind;
use crate::decoder::utilities::Condition;
use crate::ir::operand::OperandIndex;

use super::context::ArmEmitterCtx;
use super::flow_helpers::{
    emit_call, emit_cond_branch, emit_virtual_call, emit_virtual_jump, get_cond_true,
};
use super::helpers::get_int_or_zr;

// C# casts CurrOp to the OpCodeBImm base class to read Immediate.
fn bimm_immediate(ctx: &ArmEmitterCtx) -> i64 {
    match ctx.curr_op().kind {
        OpCodeKind::BImm { immediate, .. } => immediate,
        OpCodeKind::BImmCond { immediate, .. } => immediate,
        OpCodeKind::BImmCmp { immediate, .. } => immediate,
        OpCodeKind::BImmTest { immediate, .. } => immediate,
        _ => panic!("expected BImm-family opcode, got {:?}", ctx.curr_op().kind),
    }
}

fn lr_reg() -> crate::registers::GpRegisterIndex {
    // RegisterAlias.Lr = 30 (C# State/RegisterAlias.cs)
    30u8.into()
}

/// A branch to an address at/below the current PC is a loop back-edge. Bump the
/// rejit counter there (no-op for HighCq) so functions with a hot inner loop but
/// few calls still reach the HighCq promotion threshold via iteration count.
fn count_backedge(ctx: &mut ArmEmitterCtx, target: u64) {
    if target <= ctx.curr_op().address {
        ctx.emit_rejit_count();
    }
}

/// True when `target` is a loop back-edge (at or before the current PC) and
/// this is an LCQ translation (HighCq has no rejit counter, never OSRs).
fn is_osr_backedge(ctx: &ArmEmitterCtx, target: u64) -> bool {
    target <= ctx.curr_op().address && ctx.rejit_counter.is_some()
}

pub fn emit_b(ctx: &mut ArmEmitterCtx) {
    let immediate = bimm_immediate(ctx);
    count_backedge(ctx, immediate as u64);
    // Unconditional back-edge: always taken, so the unconditional OSR bounce
    // (emit_osr_exit) is correct  no loop-exit edge to preserve.
    if is_osr_backedge(ctx, immediate as u64) {
        ctx.emit_osr_exit(immediate as u64);
    }
    let label = ctx.get_label(immediate as u64);
    ctx.branch(label);
}

pub fn emit_b_cond(ctx: &mut ArmEmitterCtx) {
    let cond = match ctx.curr_op().kind {
        OpCodeKind::BImmCond { cond, .. } => cond,
        _ => panic!("expected BImmCond opcode, got {:?}", ctx.curr_op().kind),
    };
    emit_branch_cond(ctx, cond);
}

pub fn emit_cbnz(ctx: &mut ArmEmitterCtx) {
    emit_cb(ctx, true);
}
pub fn emit_cbz(ctx: &mut ArmEmitterCtx) {
    emit_cb(ctx, false);
}

fn emit_cb(ctx: &mut ArmEmitterCtx, on_not_zero: bool) {
    let rt = match ctx.curr_op().kind {
        OpCodeKind::BImmCmp { rt, .. } => rt,
        _ => panic!("expected BImmCmp opcode, got {:?}", ctx.curr_op().kind),
    };
    let value = get_int_or_zr(ctx, rt);
    emit_branch_zero(ctx, value, on_not_zero);
}

pub fn emit_bl(ctx: &mut ArmEmitterCtx) {
    let immediate = match ctx.curr_op().kind {
        OpCodeKind::BImm {
            immediate,
            link: true,
        } => immediate as u64,
        _ => panic!("expected linked BImm opcode, got {:?}", ctx.curr_op().kind),
    };
    let addr = ctx.curr_op().address;
    let lr_val = ctx.ctx.operands.constant_u64(addr + 4);
    let lr_reg_op = get_int_or_zr(ctx, lr_reg());
    ctx.copy_to(lr_reg_op, lr_val);
    emit_call(ctx, immediate);
}

pub fn emit_blr(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        OpCodeKind::BReg { rn } => rn,
        _ => panic!("expected BReg opcode, got {:?}", ctx.curr_op().kind),
    };
    let addr = ctx.curr_op().address;
    // C#: Operand n = context.Copy(GetIntOrZR(op.Rn))  snapshot BEFORE
    // the LR write, or Blr X30 reads the clobbered link register.
    let n_reg = get_int_or_zr(ctx, rn);
    let n = ctx.copy(n_reg);
    let lr_val = ctx.ctx.operands.constant_u64(addr + 4);
    let lr_reg_op = get_int_or_zr(ctx, lr_reg());
    ctx.copy_to(lr_reg_op, lr_val);
    emit_virtual_call(ctx, n);
}

pub fn emit_br(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        OpCodeKind::BReg { rn } => rn,
        _ => panic!("expected BReg opcode, got {:?}", ctx.curr_op().kind),
    };
    let target = get_int_or_zr(ctx, rn);
    let is_return = rn == lr_reg();
    emit_virtual_jump(ctx, target, is_return);
}

pub fn emit_ret(ctx: &mut ArmEmitterCtx) {
    let rn = match ctx.curr_op().kind {
        OpCodeKind::BReg { rn } => rn,
        _ => panic!("expected BReg opcode, got {:?}", ctx.curr_op().kind),
    };
    let n = get_int_or_zr(ctx, rn);
    ctx.return_with(n);
}

pub fn emit_tbnz(ctx: &mut ArmEmitterCtx) {
    emit_tb(ctx, true);
}
pub fn emit_tbz(ctx: &mut ArmEmitterCtx) {
    emit_tb(ctx, false);
}

fn emit_tb(ctx: &mut ArmEmitterCtx, on_not_zero: bool) {
    let (rt, bit) = match ctx.curr_op().kind {
        OpCodeKind::BImmTest { rt, bit, .. } => (rt, bit),
        _ => panic!("expected BImmTest opcode, got {:?}", ctx.curr_op().kind),
    };
    let v = get_int_or_zr(ctx, rt);
    // C#: Const(1L << op.Bit)  always an I64 constant.
    let mask = ctx.ctx.operands.constant_i64(1i64 << bit);
    let value = ctx.bitwise_and(v, mask);
    emit_branch_zero(ctx, value, on_not_zero);
}

// C# private EmitBranch(context, Condition cond).
fn emit_branch_cond(ctx: &mut ArmEmitterCtx, cond: Condition) {
    let immediate = bimm_immediate(ctx);
    count_backedge(ctx, immediate as u64);
    let label = ctx.get_label(immediate as u64);

    // For a conditional back-edge, evaluate the condition once and bounce
    // through the dispatcher only on the taken edge (preserving loop exit).
    if cond != Condition::Al && is_osr_backedge(ctx, immediate as u64) {
        let cond_true = get_cond_true(ctx, cond);
        ctx.emit_osr_taken(immediate as u64, cond_true, true);
        ctx.branch_if_true(label, cond_true);
        return;
    }

    emit_cond_branch(ctx, label, cond);
}

// C# private EmitBranch(context, Operand value, bool onNotZero).
fn emit_branch_zero(ctx: &mut ArmEmitterCtx, value: OperandIndex, on_not_zero: bool) {
    let immediate = bimm_immediate(ctx);
    count_backedge(ctx, immediate as u64);
    let lbl_target = ctx.get_label(immediate as u64);

    // Bounce only on the taken edge: CBNZ takes when value != 0, CBZ when == 0.
    if is_osr_backedge(ctx, immediate as u64) {
        ctx.emit_osr_taken(immediate as u64, value, on_not_zero);
    }

    if on_not_zero {
        ctx.branch_if_true(lbl_target, value);
    } else {
        ctx.branch_if_false(lbl_target, value);
    }
}
