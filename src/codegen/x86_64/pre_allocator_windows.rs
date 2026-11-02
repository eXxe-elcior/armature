// Port of CodeGen/X86/PreAllocatorWindows.cs.
//
// Same list-surgery mapping as pre_allocator.rs. The Windows ABI passes
// at most 4 args on registers (shared int/vec slots), V128 by pointer,
// and V128 returns through a caller-allocated buffer in the first arg.

use crate::codegen::pre_allocator_common;
use crate::codegen::x86_64::allocation_result::StackAllocator;
use crate::codegen::x86_64::calling_convention;
use crate::ir::block::BlockIndex;
use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, OperandKind, OperandType};
use crate::translation::compiler::CompilerContext;

use super::pre_allocator::{delete, gpr, insert_constant_reg_copies, xmm};
use super::pre_allocator_system_v::prepend_to_entry;

// C# InsertCallCopies(nodes, stackAlloc, node).
pub(crate) fn insert_call_copies(
    ctx: &mut TranslationContext,
    stack_alloc: &mut StackAllocator,
    out: &mut Vec<OperationIdx>,
    afters: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let dest = ctx.ops.dests(node).first().copied();

    // Handle struct arguments.
    let mut ret_args = 0usize;
    let mut stack_alloc_offset: i32 = 0;

    // C# local function AllocateOnStack: reuse space allocated by other
    // calls (the allocator is assumed empty before the first call site),
    // expanding only when this call needs more.
    let mut allocate_on_stack = |stack_alloc: &mut StackAllocator, size: i32| -> i32 {
        let offset = stack_alloc_offset;

        if stack_alloc_offset + size > stack_alloc.total_size() {
            stack_alloc.allocate_bytes((stack_alloc_offset + size) - stack_alloc.total_size());
        }

        stack_alloc_offset += size;

        offset
    };

    let mut arg0_reg: Option<OperandIndex> = None;

    if let Some(dest) = dest {
        if ctx.operands.get(dest).typ == OperandType::V128 {
            let stack_offset =
                allocate_on_stack(stack_alloc, OperandType::V128.size_bytes() as i32);

            let reg = gpr(
                ctx,
                calling_convention::get_int_argument_register(0),
                OperandType::I64,
            );
            arg0_reg = Some(reg);

            let offset = ctx.operands.constant_i32(stack_offset);
            let alloc_op = ctx.make_op(Instruction::StackAlloc, &[reg], &[offset]);
            out.push(alloc_op);

            ret_args = 1;
        }
    }

    let mut args_count = ctx.ops.sources(node).len() - 1;
    let max_args = calling_convention::get_arguments_on_regs_count() as usize - ret_args;

    if args_count > max_args {
        args_count = max_args;
    }

    let mut sources: Vec<OperandIndex> = vec![OperandIndex(u32::MAX); 1 + ret_args + args_count];

    sources[0] = ctx.ops.sources(node)[0];

    if let Some(reg) = arg0_reg {
        sources[1] = reg;
    }

    // V128 arguments are passed by pointer  store them to the stack
    // and replace the source with the address.
    for index in 1..ctx.ops.sources(node).len() {
        let source = ctx.ops.sources(node)[index];

        if ctx.operands.get(source).typ == OperandType::V128 {
            let stack_addr = ctx.temp(OperandType::I64);

            let stack_offset =
                allocate_on_stack(stack_alloc, OperandType::V128.size_bytes() as i32);

            let offset = ctx.operands.constant_i32(stack_offset);
            let alloc_op = ctx.make_op(Instruction::StackAlloc, &[stack_addr], &[offset]);
            out.push(alloc_op);

            let store_op = ctx.make_op(Instruction::Store, &[], &[stack_addr, source]);
            insert_constant_reg_copies(ctx, out, store_op);
            out.push(store_op);

            ctx.ops
                .set_source(node, index, stack_addr, &mut ctx.operands);
        }
    }

    // Handle arguments passed on registers.
    for index in 0..args_count {
        let source = ctx.ops.sources(node)[index + 1];
        let source_typ = ctx.operands.get(source).typ;

        let arg_index = (index + ret_args) as i32;

        let arg_reg = if source_typ.is_integer() {
            gpr(
                ctx,
                calling_convention::get_int_argument_register(arg_index),
                source_typ,
            )
        } else {
            xmm(
                ctx,
                calling_convention::get_vec_argument_register(arg_index),
                source_typ,
            )
        };

        let copy_op = ctx.make_op(Instruction::Copy, &[arg_reg], &[source]);
        insert_constant_reg_copies(ctx, out, copy_op);
        out.push(copy_op);

        sources[1 + ret_args + index] = arg_reg;
    }

    // The remaining arguments (those that are not passed on registers)
    // should be passed on the stack, we write them to the stack with "SpillArg".
    for index in args_count..ctx.ops.sources(node).len() - 1 {
        let source = ctx.ops.sources(node)[index + 1];
        let offset = ctx.operands.constant_i32(((index + ret_args) * 8) as i32);

        let spill_op = ctx.make_op(Instruction::SpillArg, &[], &[offset, source]);
        insert_constant_reg_copies(ctx, out, spill_op);
        out.push(spill_op);
    }

    if let Some(dest) = dest {
        let dest_typ = ctx.operands.get(dest).typ;

        if dest_typ == OperandType::V128 {
            let ret_value_addr = ctx.temp(OperandType::I64);

            let arg0 = arg0_reg.expect("V128 return allocated arg0");
            let copy = ctx.make_op(Instruction::Copy, &[ret_value_addr], &[arg0]);
            out.push(copy);

            let load_op = ctx.make_op(Instruction::Load, &[dest], &[ret_value_addr]);
            afters.push(load_op);

            ctx.ops.set_destinations(node, &[], &mut ctx.operands);
        } else {
            let ret_reg = if dest_typ.is_integer() {
                gpr(ctx, calling_convention::get_int_return_register(), dest_typ)
            } else {
                xmm(ctx, calling_convention::get_vec_return_register(), dest_typ)
            };

            let copy_op = ctx.make_op(Instruction::Copy, &[dest], &[ret_reg]);
            afters.push(copy_op);

            ctx.ops
                .set_destinations(node, &[ret_reg], &mut ctx.operands);
        }
    }

    ctx.ops.set_sources(node, &sources, &mut ctx.operands);
}

// C# InsertTailcallCopies.
pub(crate) fn insert_tailcall_copies(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let args_count = ctx.ops.sources(node).len() - 1;
    let max_args = calling_convention::get_arguments_on_regs_count() as usize;

    if args_count > max_args {
        unimplemented!("Spilling is not currently supported for tail calls. (too many arguments)");
    }

    let mut sources: Vec<OperandIndex> = vec![OperandIndex(u32::MAX); 1 + args_count];

    // Handle arguments passed on registers.
    for index in 0..args_count {
        let source = ctx.ops.sources(node)[1 + index];
        let source_typ = ctx.operands.get(source).typ;

        let arg_reg = if source_typ.is_integer() {
            gpr(
                ctx,
                calling_convention::get_int_argument_register(index as i32),
                source_typ,
            )
        } else {
            xmm(
                ctx,
                calling_convention::get_vec_argument_register(index as i32),
                source_typ,
            )
        };

        let copy_op = ctx.make_op(Instruction::Copy, &[arg_reg], &[source]);
        insert_constant_reg_copies(ctx, out, copy_op);
        out.push(copy_op);

        sources[1 + index] = arg_reg;
    }

    // The target address must be on the return registers, since we
    // don't return anything and it is guaranteed to not be a
    // callee saved register (which would be trashed on the epilogue).
    let ret_reg = gpr(
        ctx,
        calling_convention::get_int_return_register(),
        OperandType::I64,
    );

    let src0 = ctx.ops.sources(node)[0];
    let addr_copy_op = ctx.make_op(Instruction::Copy, &[ret_reg], &[src0]);
    out.push(addr_copy_op);

    sources[0] = ret_reg;

    ctx.ops.set_sources(node, &sources, &mut ctx.operands);
}

// C# InsertLoadArgumentCopy. Returns true when the node was deleted.
pub(crate) fn insert_load_argument_copy(
    cctx: &CompilerContext,
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    preserved_args: &mut [Option<OperandIndex>],
    entry: BlockIndex,
    current_block: BlockIndex,
    node: OperationIdx,
) -> bool {
    let source = ctx.ops.sources(node)[0];

    debug_assert_eq!(
        ctx.operands.get(source).kind,
        OperandKind::Constant,
        "Non-constant LoadArgument source kind."
    );

    let ret_args = if cctx.func_return_type == OperandType::V128 {
        1
    } else {
        0
    };

    let index = ctx.operands.get(source).value as i32 + ret_args;

    if index < calling_convention::get_arguments_on_regs_count() {
        let dest = ctx.ops.dests(node)[0];
        let dest_typ = ctx.operands.get(dest).typ;

        if preserved_args[index as usize].is_none() {
            let (arg_reg, p_arg) = if dest_typ.is_integer() {
                let r = gpr(
                    ctx,
                    calling_convention::get_int_argument_register(index),
                    dest_typ,
                );
                (r, ctx.temp(dest_typ))
            } else if dest_typ == OperandType::V128 {
                // V128 is passed by pointer  preserve the pointer.
                let r = gpr(
                    ctx,
                    calling_convention::get_int_argument_register(index),
                    OperandType::I64,
                );
                (r, ctx.temp(OperandType::I64))
            } else {
                let r = xmm(
                    ctx,
                    calling_convention::get_vec_argument_register(index),
                    dest_typ,
                );
                (r, ctx.temp(dest_typ))
            };

            let copy_op = ctx.make_op(Instruction::Copy, &[p_arg], &[arg_reg]);

            prepend_to_entry(ctx, out, entry, current_block, copy_op);

            preserved_args[index as usize] = Some(p_arg);
        }

        let preserved = preserved_args[index as usize].unwrap();

        if dest_typ != OperandType::V128 && ctx.operands.assigns_count(dest) == 1 {
            // Let's propagate the argument if we can to avoid copies.
            pre_allocator_common::propagate(ctx, dest, preserved);
        } else {
            // V128 destination loads through the preserved pointer.
            let inst = if dest_typ == OperandType::V128 {
                Instruction::Load
            } else {
                Instruction::Copy
            };
            let arg_copy_op = ctx.make_op(inst, &[dest], &[preserved]);
            out.push(arg_copy_op);
        }

        delete(ctx, node);
        true
    } else {
        // TODO: Pass on stack.
        false
    }
}

// C# InsertReturnCopy.
pub(crate) fn insert_return_copy(
    cctx: &CompilerContext,
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    preserved_args: &mut [Option<OperandIndex>],
    entry: BlockIndex,
    current_block: BlockIndex,
    node: OperationIdx,
) {
    let _ = cctx;

    if ctx.ops.sources(node).is_empty() {
        return;
    }

    let source = ctx.ops.sources(node)[0];
    let source_typ = ctx.operands.get(source).typ;

    let ret_reg = if source_typ.is_integer() {
        gpr(
            ctx,
            calling_convention::get_int_return_register(),
            source_typ,
        )
    } else if source_typ == OperandType::V128 {
        if preserved_args[0].is_none() {
            let preserved_arg = ctx.temp(OperandType::I64);
            let arg0 = gpr(
                ctx,
                calling_convention::get_int_argument_register(0),
                OperandType::I64,
            );

            let copy_op = ctx.make_op(Instruction::Copy, &[preserved_arg], &[arg0]);

            prepend_to_entry(ctx, out, entry, current_block, copy_op);

            preserved_args[0] = Some(preserved_arg);
        }

        preserved_args[0].unwrap()
    } else {
        xmm(
            ctx,
            calling_convention::get_vec_return_register(),
            source_typ,
        )
    };

    if source_typ == OperandType::V128 {
        let ret_store_op = ctx.make_op(Instruction::Store, &[], &[ret_reg, source]);
        out.push(ret_store_op);
    } else {
        let ret_copy_op = ctx.make_op(Instruction::Copy, &[ret_reg], &[source]);
        out.push(ret_copy_op);
    }

    ctx.ops.set_sources(node, &[], &mut ctx.operands);
}
