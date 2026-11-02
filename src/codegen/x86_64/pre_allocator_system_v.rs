// Port of CodeGen/X86/PreAllocatorSystemV.cs.
//
// Same list-surgery mapping as pre_allocator.rs: AddBefore → out.push,
// chained AddAfter → afters, Delete → caller drops the node.

use crate::codegen::pre_allocator_common;
use crate::codegen::x86_64::calling_convention;
use crate::ir::block::BlockIndex;
use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, OperandKind, OperandType};
use crate::translation::compiler::CompilerContext;

use super::pre_allocator::{delete, gpr, insert_constant_reg_copies, xmm};

// C# InsertCallCopies.
pub(crate) fn insert_call_copies(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    afters: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let dest = ctx.ops.dests(node).first().copied();

    let mut sources: Vec<OperandIndex> = vec![ctx.ops.sources(node)[0]];

    let args_count = ctx.ops.sources(node).len() - 1;

    let int_max = calling_convention::get_int_arguments_on_regs_count();
    let vec_max = calling_convention::get_vec_arguments_on_regs_count();

    let mut int_count = 0;
    let mut vec_count = 0;

    let mut stack_offset: i32 = 0;

    for index in 0..args_count {
        let source = ctx.ops.sources(node)[index + 1];
        let source_typ = ctx.operands.get(source).typ;

        let pass_on_reg = if source_typ.is_integer() {
            int_count < int_max
        } else if source_typ == OperandType::V128 {
            int_count + 1 < int_max
        } else {
            vec_count < vec_max
        };

        if source_typ == OperandType::V128 && pass_on_reg {
            // V128 is a struct, we pass each half on a GPR if possible.
            let arg_reg = gpr(
                ctx,
                calling_convention::get_int_argument_register(int_count),
                OperandType::I64,
            );
            int_count += 1;
            let arg_reg2 = gpr(
                ctx,
                calling_convention::get_int_argument_register(int_count),
                OperandType::I64,
            );
            int_count += 1;

            let zero = ctx.operands.constant_i32(0);
            let lo = ctx.make_op(Instruction::VectorExtract, &[arg_reg], &[source, zero]);
            out.push(lo);
            let one = ctx.operands.constant_i32(1);
            let hi = ctx.make_op(Instruction::VectorExtract, &[arg_reg2], &[source, one]);
            out.push(hi);

            continue;
        }

        if pass_on_reg {
            let arg_reg = if source_typ.is_integer() {
                let r = gpr(
                    ctx,
                    calling_convention::get_int_argument_register(int_count),
                    source_typ,
                );
                int_count += 1;
                r
            } else {
                let r = xmm(
                    ctx,
                    calling_convention::get_vec_argument_register(vec_count),
                    source_typ,
                );
                vec_count += 1;
                r
            };

            let copy_op = ctx.make_op(Instruction::Copy, &[arg_reg], &[source]);
            // C#: InsertConstantRegCopies(nodes, nodes.AddBefore(node, copyOp))
            //  the lowering copies land before copyOp.
            insert_constant_reg_copies(ctx, out, copy_op);
            out.push(copy_op);

            sources.push(arg_reg);
        } else {
            let offset = ctx.operands.constant_i32(stack_offset);

            let spill_op = ctx.make_op(Instruction::SpillArg, &[], &[offset, source]);
            insert_constant_reg_copies(ctx, out, spill_op);
            out.push(spill_op);

            stack_offset += ctx.operands.get(source).typ.size_bytes() as i32;
        }
    }

    ctx.ops.set_sources(node, &sources, &mut ctx.operands);

    if let Some(dest) = dest {
        let dest_typ = ctx.operands.get(dest).typ;

        if dest_typ == OperandType::V128 {
            let ret_l_reg = gpr(
                ctx,
                calling_convention::get_int_return_register(),
                OperandType::I64,
            );
            let ret_h_reg = gpr(
                ctx,
                calling_convention::get_int_return_register_high(),
                OperandType::I64,
            );

            let vcreate = ctx.make_op(Instruction::VectorCreateScalar, &[dest], &[ret_l_reg]);
            afters.push(vcreate);
            let one = ctx.operands.constant_i32(1);
            let vinsert = ctx.make_op(Instruction::VectorInsert, &[dest], &[dest, ret_h_reg, one]);
            afters.push(vinsert);

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
}

// C# InsertTailcallCopies.
pub(crate) fn insert_tailcall_copies(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let mut sources: Vec<OperandIndex> = vec![ctx.ops.sources(node)[0]];

    let args_count = ctx.ops.sources(node).len() - 1;

    let int_max = calling_convention::get_int_arguments_on_regs_count();
    let vec_max = calling_convention::get_vec_arguments_on_regs_count();

    let mut int_count = 0;
    let mut vec_count = 0;

    // Handle arguments passed on registers.
    for index in 0..args_count {
        let source = ctx.ops.sources(node)[1 + index];
        let source_typ = ctx.operands.get(source).typ;

        let pass_on_reg = if source_typ.is_integer() {
            int_count + 1 < int_max
        } else {
            vec_count < vec_max
        };

        if source_typ == OperandType::V128 && pass_on_reg {
            // V128 is a struct, we pass each half on a GPR if possible.
            let arg_reg = gpr(
                ctx,
                calling_convention::get_int_argument_register(int_count),
                OperandType::I64,
            );
            int_count += 1;
            let arg_reg2 = gpr(
                ctx,
                calling_convention::get_int_argument_register(int_count),
                OperandType::I64,
            );
            int_count += 1;

            let zero = ctx.operands.constant_i32(0);
            let lo = ctx.make_op(Instruction::VectorExtract, &[arg_reg], &[source, zero]);
            out.push(lo);
            let one = ctx.operands.constant_i32(1);
            let hi = ctx.make_op(Instruction::VectorExtract, &[arg_reg2], &[source, one]);
            out.push(hi);

            continue;
        }

        if pass_on_reg {
            let arg_reg = if source_typ.is_integer() {
                let r = gpr(
                    ctx,
                    calling_convention::get_int_argument_register(int_count),
                    source_typ,
                );
                int_count += 1;
                r
            } else {
                let r = xmm(
                    ctx,
                    calling_convention::get_vec_argument_register(vec_count),
                    source_typ,
                );
                vec_count += 1;
                r
            };

            let copy_op = ctx.make_op(Instruction::Copy, &[arg_reg], &[source]);
            insert_constant_reg_copies(ctx, out, copy_op);
            out.push(copy_op);

            sources.push(arg_reg);
        } else {
            unimplemented!(
                "Spilling is not currently supported for tail calls. (too many arguments)"
            );
        }
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

// C# InsertLoadArgumentCopy. Returns true when the node was deleted
// (C# returns nextNode and Deletes; the caller's rebuild loop only needs
// the deletion flag  see pre_allocator.rs header note).
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

    let index = ctx.operands.get(source).value as i32;

    let mut int_count = 0;
    let mut vec_count = 0;

    for c_index in 0..index {
        let arg_type = cctx.func_arg_types[c_index as usize];

        if arg_type.is_integer() {
            int_count += 1;
        } else if arg_type == OperandType::V128 {
            int_count += 2;
        } else {
            vec_count += 1;
        }
    }

    let source_typ = ctx.operands.get(source).typ;
    let pass_on_reg = if source_typ.is_integer() {
        int_count < calling_convention::get_int_arguments_on_regs_count()
    } else if source_typ == OperandType::V128 {
        int_count + 1 < calling_convention::get_int_arguments_on_regs_count()
    } else {
        vec_count < calling_convention::get_vec_arguments_on_regs_count()
    };

    if pass_on_reg {
        let dest = ctx.ops.dests(node)[0];
        let dest_typ = ctx.operands.get(dest).typ;

        if preserved_args[index as usize].is_none() {
            if dest_typ == OperandType::V128 {
                // V128 is a struct, we pass each half on a GPR if possible.
                let p_arg = ctx.temp(OperandType::V128);

                let arg_l_reg = gpr(
                    ctx,
                    calling_convention::get_int_argument_register(int_count),
                    OperandType::I64,
                );
                let arg_h_reg = gpr(
                    ctx,
                    calling_convention::get_int_argument_register(int_count + 1),
                    OperandType::I64,
                );

                let copy_l = ctx.make_op(Instruction::VectorCreateScalar, &[p_arg], &[arg_l_reg]);
                let one = ctx.operands.constant_i32(1);
                let copy_h = ctx.make_op(
                    Instruction::VectorInsert,
                    &[p_arg],
                    &[p_arg, arg_h_reg, one],
                );

                // C#: AddFirst(copyH); AddFirst(copyL) → final order L, H.
                prepend_to_entry(ctx, out, entry, current_block, copy_h);
                prepend_to_entry(ctx, out, entry, current_block, copy_l);

                preserved_args[index as usize] = Some(p_arg);
            } else {
                let p_arg = ctx.temp(dest_typ);

                let arg_reg = if dest_typ.is_integer() {
                    gpr(
                        ctx,
                        calling_convention::get_int_argument_register(int_count),
                        dest_typ,
                    )
                } else {
                    xmm(
                        ctx,
                        calling_convention::get_vec_argument_register(vec_count),
                        dest_typ,
                    )
                };

                let copy_op = ctx.make_op(Instruction::Copy, &[p_arg], &[arg_reg]);

                prepend_to_entry(ctx, out, entry, current_block, copy_op);

                preserved_args[index as usize] = Some(p_arg);
            }
        }

        let preserved = preserved_args[index as usize].unwrap();

        if ctx.operands.assigns_count(dest) == 1 {
            // Let's propagate the argument if we can to avoid copies.
            pre_allocator_common::propagate(ctx, dest, preserved);
        } else {
            let arg_copy_op = ctx.make_op(Instruction::Copy, &[dest], &[preserved]);
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
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    if ctx.ops.sources(node).is_empty() {
        return;
    }

    let source = ctx.ops.sources(node)[0];
    let source_typ = ctx.operands.get(source).typ;

    if source_typ == OperandType::V128 {
        let ret_l_reg = gpr(
            ctx,
            calling_convention::get_int_return_register(),
            OperandType::I64,
        );
        let ret_h_reg = gpr(
            ctx,
            calling_convention::get_int_return_register_high(),
            OperandType::I64,
        );

        let zero = ctx.operands.constant_i32(0);
        let lo = ctx.make_op(Instruction::VectorExtract, &[ret_l_reg], &[source, zero]);
        out.push(lo);
        let one = ctx.operands.constant_i32(1);
        let hi = ctx.make_op(Instruction::VectorExtract, &[ret_h_reg], &[source, one]);
        out.push(hi);
    } else {
        let ret_reg = if source_typ.is_integer() {
            gpr(
                ctx,
                calling_convention::get_int_return_register(),
                source_typ,
            )
        } else {
            xmm(
                ctx,
                calling_convention::get_vec_return_register(),
                source_typ,
            )
        };

        let ret_copy_op = ctx.make_op(Instruction::Copy, &[ret_reg], &[source]);
        out.push(ret_copy_op);
    }
}

// C# `cctx.Cfg.Entry.Operations.AddFirst(op)`. When the entry block is
// the one currently being rebuilt, its ops Vec is taken  prepend to the
// rebuild buffer instead. `_unused` keeps the signature uniform.
pub(crate) fn prepend_to_entry(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    entry: BlockIndex,
    current_block: BlockIndex,
    op: OperationIdx,
) {
    if entry == current_block {
        out.insert(0, op);
    } else {
        ctx.blocks.get_mut(entry).ops.insert(0, op);
    }
}
