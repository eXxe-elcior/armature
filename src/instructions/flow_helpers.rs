// Port of Instructions/InstEmitFlowHelper.cs

use crate::decoder::utilities::Condition;
use crate::ir::operand::OperandIndex;
use crate::ir::types::OperandType;
use crate::state::PState;

use super::context::ArmEmitterCtx;
use super::helpers::get_flag;

pub fn emit_cond_branch(ctx: &mut ArmEmitterCtx, target: OperandIndex, cond: Condition) {
    if cond != Condition::Al {
        let cond_true = get_cond_true(ctx, cond);
        ctx.branch_if_true(target, cond_true);
    } else {
        ctx.branch(target);
    }
}

// C# local function Inverse: val ^ 1.
fn inverse(ctx: &mut ArmEmitterCtx, val: OperandIndex) -> OperandIndex {
    let one = ctx.ctx.operands.constant_i32(1);
    ctx.bitwise_xor(val, one)
}

pub fn get_cond_true(ctx: &mut ArmEmitterCtx, condition: Condition) -> OperandIndex {
    if let Some(cmp_result) = ctx.try_get_comparison_result(condition) {
        return cmp_result;
    }

    match condition {
        Condition::Eq => get_flag(PState::ZFlag, &mut ctx.ctx.operands),
        Condition::Ne => {
            let z = get_flag(PState::ZFlag, &mut ctx.ctx.operands);
            inverse(ctx, z)
        }
        Condition::GeUn => get_flag(PState::CFlag, &mut ctx.ctx.operands),
        Condition::LtUn => {
            let c = get_flag(PState::CFlag, &mut ctx.ctx.operands);
            inverse(ctx, c)
        }
        Condition::Mi => get_flag(PState::NFlag, &mut ctx.ctx.operands),
        Condition::Pl => {
            let n = get_flag(PState::NFlag, &mut ctx.ctx.operands);
            inverse(ctx, n)
        }
        Condition::Vs => get_flag(PState::VFlag, &mut ctx.ctx.operands),
        Condition::Vc => {
            let v = get_flag(PState::VFlag, &mut ctx.ctx.operands);
            inverse(ctx, v)
        }
        Condition::GtUn => {
            let c = get_flag(PState::CFlag, &mut ctx.ctx.operands);
            let z = get_flag(PState::ZFlag, &mut ctx.ctx.operands);
            let not_z = inverse(ctx, z);
            ctx.bitwise_and(c, not_z)
        }
        Condition::LeUn => {
            let c = get_flag(PState::CFlag, &mut ctx.ctx.operands);
            let z = get_flag(PState::ZFlag, &mut ctx.ctx.operands);
            let not_c = inverse(ctx, c);
            ctx.bitwise_or(not_c, z)
        }
        Condition::Ge => {
            let n = get_flag(PState::NFlag, &mut ctx.ctx.operands);
            let v = get_flag(PState::VFlag, &mut ctx.ctx.operands);
            ctx.icompare_equal(n, v)
        }
        Condition::Lt => {
            let n = get_flag(PState::NFlag, &mut ctx.ctx.operands);
            let v = get_flag(PState::VFlag, &mut ctx.ctx.operands);
            ctx.icompare_not_equal(n, v)
        }
        Condition::Gt => {
            let n = get_flag(PState::NFlag, &mut ctx.ctx.operands);
            let z = get_flag(PState::ZFlag, &mut ctx.ctx.operands);
            let v = get_flag(PState::VFlag, &mut ctx.ctx.operands);
            let not_z = inverse(ctx, z);
            let n_eq_v = ctx.icompare_equal(n, v);
            ctx.bitwise_and(not_z, n_eq_v)
        }
        Condition::Le => {
            let n = get_flag(PState::NFlag, &mut ctx.ctx.operands);
            let z = get_flag(PState::ZFlag, &mut ctx.ctx.operands);
            let v = get_flag(PState::VFlag, &mut ctx.ctx.operands);
            let n_ne_v = ctx.icompare_not_equal(n, v);
            ctx.bitwise_or(z, n_ne_v)
        }
        // C# default: value = Const(1) (Al and Nv fall through the switch).
        Condition::Al | Condition::Nv => ctx.ctx.operands.constant_i32(1),
    }
}

// Call / virtual-call / virtual-jump
/// C# EmitCall: branch to a known guest address (immediate).
/// Recursive calls (same function) → local branch.
/// Non-recursive → EmitTableBranch (or Return in singlestep).
pub fn emit_call(ctx: &mut ArmEmitterCtx, immediate: u64) {
    if ctx.is_single_step {
        let addr = ctx.ctx.operands.constant_u64(immediate);
        ctx.return_with(addr);
        return;
    }

    let is_recursive = immediate == ctx.entry_address;
    if is_recursive {
        let label = ctx.get_label(immediate);
        ctx.branch(label);
    } else {
        let addr = ctx.ctx.operands.constant_u64(immediate);
        emit_table_branch(ctx, addr, false);
    }
}

/// C# EmitVirtualCall: branch to a runtime-computed guest address (register).
pub fn emit_virtual_call(ctx: &mut ArmEmitterCtx, target: OperandIndex) {
    if ctx.is_single_step {
        ctx.return_with(target);
        return;
    }

    emit_table_branch(ctx, target, false);
}

/// C# EmitVirtualJump: branch to a computed address (register), possibly a return.
/// `is_return`: when true (or singlestep), does context.Return(target).
pub fn emit_virtual_jump(ctx: &mut ArmEmitterCtx, target: OperandIndex, is_return: bool) {
    if is_return || ctx.is_single_step {
        ctx.return_with(target);
        return;
    }

    emit_table_branch(ctx, target, true);
}

/// C# EmitTableBranch: store guest address to context, then call/tailcall
/// the dispatch stub.
///
/// Branches use the dispatch stub, which checks the function table before
/// resolving a missing translation through GetFunctionAddress.
fn emit_table_branch(ctx: &mut ArmEmitterCtx, guest_address: OperandIndex, is_jump: bool) {
    use crate::ir::block::BlockFrequency;
    use crate::ir::types::Comparison;

    // C#: StoreToContext()
    ctx.store_to_context();

    // C#: ZeroExtend32(I64) if guest_address.Type == I32
    let guest_addr = if ctx.ctx.operands.get(guest_address).typ == OperandType::I32 {
        ctx.zero_extend32(OperandType::I64, guest_address)
    } else {
        guest_address
    };

    // C#: store the target guest address into the native context; the
    // stub uses it to dispatch into the next translation.
    let native_ctx = ctx.load_argument(OperandType::I64, 0);
    let disp_offset = ctx
        .ctx
        .operands
        .constant_i64(crate::registers::CpuState::dispatch_address_offset() as i64);
    let disp_addr_op = ctx.add(native_ctx, disp_offset);
    ctx.store(disp_addr_op, guest_addr);

    debug_assert!(
        ctx.dispatch_stub != 0,
        "EmitTableBranch with no dispatch stub wired"
    );
    // Symbol the baked dispatch-stub address so RELOCATABLE codegen records a
    // reloc for it  required for PTC (the stub lives at a fresh address each
    // run). Behaviour-neutral when not relocatable (reloc simply not collected).
    // Special,3 = dispatch stub (C# Ptc.DispatchStubSymbol); Special,1 = page table.
    let host_address = ctx.ctx.operands.constant_i64_with_symbol(
        ctx.dispatch_stub as i64,
        crate::ir::types::Symbol {
            typ: crate::ir::types::SymbolType::Special,
            value: 3,
        },
    );

    if is_jump {
        ctx.tailcall(host_address, &[native_ctx]);
    } else {
        // C#: returnAddress = Call(hostAddress, I64, nativeContext)
        let return_address = ctx
            .call_address(host_address, OperandType::I64, &[native_ctx])
            .expect("dispatch stub returns I64");

        ctx.load_from_context();

        // The return value of a translated function is the address
        // execution returned to. If it is our next instruction, continue
        // within this block; otherwise return until the dispatcher sorts
        // it out.
        let op = ctx.curr_op();
        let next_addr_value = op.address + 4;
        let next_addr = ctx.ctx.operands.constant_i64(next_addr_value as i64);

        let lbl_continue = ctx.get_label(next_addr_value);
        ctx.branch_if(
            lbl_continue,
            return_address,
            next_addr,
            Comparison::Equal,
            BlockFrequency::Cold,
        );

        ctx.return_with(return_address);
    }
}
