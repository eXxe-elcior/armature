use crate::ir::block::{BlockFrequency, BlockIndex};
use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, OperandType};
use crate::translation::compiler::CompilerContext;

pub fn run_pass(cctx: &mut CompilerContext, ctx: &mut TranslationContext) {
    let merged_return = ctx.blocks.alloc();

    let return_value: Option<OperandIndex>;
    let return_op: OperationIdx;

    if cctx.func_return_type == OperandType::None {
        return_value = None;
        return_op = ctx.make_op(Instruction::Return, &[], &[]);
    } else {
        let rv = cctx
            .cfg
            .allocate_local(&mut ctx.operands, cctx.func_return_type);
        return_value = Some(rv);
        return_op = ctx.make_op(Instruction::Return, &[], &[rv]);
    }

    ctx.blocks.get_mut(merged_return).frequency = BlockFrequency::Cold;
    ctx.blocks.append_op(merged_return, return_op);

    let order_snapshot: Vec<BlockIndex> = ctx.blocks.order().to_vec();
    for &block in &order_snapshot {
        let ops_snapshot: Vec<OperationIdx> = ctx.blocks.get(block).ops.clone();
        let last_op = ops_snapshot.last().copied();

        if let Some(op) = last_op {
            if ctx.ops.get(op).instruction == Instruction::Return {
                // Remove the return op from the block.
                {
                    let ops = &mut ctx.blocks.get_mut(block).ops;
                    if let Some(pos) = ops.iter().position(|&o| o == op) {
                        ops.remove(pos);
                    }
                }

                if cctx.func_return_type == OperandType::None {
                    let _prep_block = prepare_merge(ctx, block, merged_return);
                } else {
                    let copy_op = ctx.make_op(
                        Instruction::Copy,
                        &[return_value.unwrap()],
                        &[ctx.ops.sources(op)[0]],
                    );
                    let prep_block = prepare_merge(ctx, block, merged_return);
                    // C#: PrepareMerge(block, mergedReturn).Append(copyOp)
                    // insert before the prep block's terminator (Return /
                    // Tailcall / BranchIf), or at the end if none.
                    let insert_at = {
                        let op_list = &ctx.blocks.get(prep_block).ops;
                        match op_list.last() {
                            Some(&last) => match ctx.ops.get(last).instruction {
                                Instruction::Return
                                | Instruction::Tailcall
                                | Instruction::BranchIf => op_list.len() - 1,
                                _ => op_list.len(),
                            },
                            None => 0,
                        }
                    };
                    ctx.blocks
                        .get_mut(prep_block)
                        .ops
                        .insert(insert_at, copy_op);
                }
            }
        }
    }

    ctx.blocks.push_order(merged_return);
    cctx.cfg.update(&mut ctx.blocks);
}

fn prepare_merge(ctx: &mut TranslationContext, from: BlockIndex, to: BlockIndex) -> BlockIndex {
    let preds: Vec<BlockIndex> = ctx.blocks.get(from).predecessors.clone();
    let from_pred = if preds.len() == 1 {
        Some(preds[0])
    } else {
        None
    };

    if ctx.blocks.get(from).ops.is_empty()
        && from_pred.is_some()
        && ctx.blocks.get(from_pred.unwrap()).successor_count == 1
    {
        let fp = from_pred.unwrap();
        ctx.blocks.get_mut(fp).successors[0] = Some(to);
        fp
    } else {
        ctx.blocks.add_successor(from, to);
        from
    }
}
