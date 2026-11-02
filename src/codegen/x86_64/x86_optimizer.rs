// Port of CodeGen/X86/X86Optimizer.cs
// - Constant materialization for large immediates
// - Address folding into memory operands

use crate::util::hash::FxHashMap as HashMap;

use crate::ir::block::BlockIndex;
use crate::ir::operand::{OperandArena, OperandIndex};
use crate::ir::operation::{OperationArena, OperationIdx};
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, Multiplier, OperandKind, OperandType, Symbol};
use crate::translation::cfg::ControlFlowGraph;

const MAX_CONSTANT_USES: usize = 10000;

fn is_long_const(ops: &OperandArena, idx: OperandIndex) -> bool {
    let op = ops.get(idx);
    let value = match op.typ {
        OperandType::I32 => op.value as i32 as i64,
        _ => op.value as i64,
    };
    value as i32 as i64 != value
}

pub fn run_pass(_cfg: &ControlFlowGraph, ctx: &mut TranslationContext) {
    let mut constants: HashMap<u64, OperandIndex> = HashMap::default();

    let order_snapshot: Vec<BlockIndex> = ctx.blocks.order().to_vec();
    for &block in &order_snapshot {
        constants.clear();

        let ops_snapshot: Vec<OperationIdx> = ctx.blocks.get(block).ops.clone();

        for &node in &ops_snapshot {
            let inst = ctx.ops.get(node).instruction;

            if inst == Instruction::Add {
                let src1 = ctx.ops.sources(node)[0];
                let src2 = ctx.ops.sources(node)[1];

                if ctx.operands.get(src1).kind == OperandKind::Constant
                    && (ctx.operands.get(src1).symbol != Symbol::default()
                        || is_long_const(&ctx.operands, src1))
                {
                    let copy_id = get_constant_copy(ctx, block, node, src1, &mut constants);
                    ctx.ops.set_source(node, 0, copy_id, &mut ctx.operands);
                }

                if ctx.operands.get(src2).kind == OperandKind::Constant
                    && (ctx.operands.get(src2).symbol != Symbol::default()
                        || is_long_const(&ctx.operands, src2))
                {
                    let copy_id = get_constant_copy(ctx, block, node, src2, &mut constants);
                    ctx.ops.set_source(node, 1, copy_id, &mut ctx.operands);
                }
            }

            if is_memory_load_or_store(inst) {
                let typ = if !ctx.ops.dests(node).is_empty() {
                    let d = ctx.ops.dests(node)[0];
                    ctx.operands.get(d).typ
                } else {
                    ctx.operands.get(ctx.ops.sources(node)[1]).typ
                };

                let addr = ctx.ops.sources(node)[0];
                let mem_op = get_memory_operand_or_null(ctx, addr, typ);

                if mem_op != OperandIndex(u32::MAX) {
                    ctx.ops.set_source(node, 0, mem_op, &mut ctx.operands);
                }
            }
        }
    }

    crate::codegen::optimizations::optimizer::remove_unused_nodes(ctx);
}

fn get_constant_copy(
    ctx: &mut TranslationContext,
    block: BlockIndex,
    op: OperationIdx,
    source: OperandIndex,
    constants: &mut HashMap<u64, OperandIndex>,
) -> OperandIndex {
    let val = ctx.operands.get(source).value;
    let entry = constants.get(&val).copied();
    if entry.is_none() || ctx.operands.uses(entry.unwrap()).len() > MAX_CONSTANT_USES {
        let local = ctx.operands.local(ctx.operands.get(source).typ);
        let copy_op = ctx.make_op(Instruction::Copy, &[local], &[source]);
        let ops = &mut ctx.blocks.get_mut(block).ops;
        if let Some(pos) = ops.iter().position(|&o| o == op) {
            ops.insert(pos, copy_op);
        }
        constants.insert(val, local);
        return local;
    }
    entry.unwrap()
}

fn get_memory_operand_or_null(
    ctx: &mut TranslationContext,
    addr: OperandIndex,
    typ: OperandType,
) -> OperandIndex {
    let ops = &ctx.ops;
    let operands = &ctx.operands;

    let mut base_op = addr;

    let imm = get_const_op(ops, operands, &mut base_op);

    let (index_op, scale) = get_index_op(ops, operands, &mut base_op);

    if base_op == addr {
        return OperandIndex(u32::MAX);
    }

    let mut final_imm = imm;
    let mut final_index = index_op;
    let final_scale = scale;

    if final_imm == 0 && scale == Multiplier::x1 && index_op.is_some() {
        let mut idx = index_op.unwrap();
        final_imm = get_const_op(ops, operands, &mut idx);
        final_index = Some(idx);
    }

    ctx.operands
        .memory(typ, Some(base_op), final_index, final_scale, final_imm)
}

fn get_const_op(ops: &OperationArena, operands: &OperandArena, base_op: &mut OperandIndex) -> i32 {
    let operation = get_asg_op_with_inst(ops, operands, *base_op, Instruction::Add);
    let operation = match operation {
        Some(op) => op,
        None => return 0,
    };

    let src1 = ops.sources(operation)[0];
    let src2 = ops.sources(operation)[1];

    let (const_op, other_op) = if operands.get(src1).kind == OperandKind::Constant
        && operands.get(src2).kind == OperandKind::LocalVariable
    {
        (src1, src2)
    } else if operands.get(src1).kind == OperandKind::LocalVariable
        && operands.get(src2).kind == OperandKind::Constant
    {
        (src2, src1)
    } else {
        return 0;
    };

    if is_long_const(operands, const_op) {
        return 0;
    }

    *base_op = other_op;
    operands.get(const_op).value as i32
}

fn get_index_op(
    ops: &OperationArena,
    operands: &OperandArena,
    base_op: &mut OperandIndex,
) -> (Option<OperandIndex>, Multiplier) {
    let mut scale = Multiplier::x1;

    let add_op = get_asg_op_with_inst(ops, operands, *base_op, Instruction::Add);
    let add_op = match add_op {
        Some(op) => op,
        None => return (None, scale),
    };

    let src1 = ops.sources(add_op)[0];
    let src2 = ops.sources(add_op)[1];

    if operands.get(src1).kind != OperandKind::LocalVariable
        || operands.get(src2).kind != OperandKind::LocalVariable
    {
        return (None, scale);
    }

    let mut base_op_new = src1;
    let mut index_op = src2;
    let mut index_on_src2 = false;

    let mut shl_op = get_asg_op_with_inst(ops, operands, src1, Instruction::ShiftLeft);

    if shl_op.is_none() {
        shl_op = get_asg_op_with_inst(ops, operands, src2, Instruction::ShiftLeft);
        index_on_src2 = true;
    }

    if let Some(shl) = shl_op {
        let sh_src = ops.sources(shl)[0];
        let shift = ops.sources(shl)[1];

        if operands.get(sh_src).kind == OperandKind::LocalVariable
            && operands.get(shift).kind == OperandKind::Constant
            && operands.get(shift).value <= 3
        {
            scale = match operands.get(shift).value {
                1 => Multiplier::x2,
                2 => Multiplier::x4,
                3 => Multiplier::x8,
                _ => Multiplier::x1,
            };

            // C#: baseOp = indexOnSrc2 ? src1 : src2  the operand that is
            // NOT the shift result becomes the base.
            base_op_new = if index_on_src2 { src1 } else { src2 };
            index_op = sh_src;
        }
    }

    *base_op = base_op_new;
    (Some(index_op), scale)
}

fn get_asg_op_with_inst(
    ops: &OperationArena,
    operands: &OperandArena,
    op: OperandIndex,
    inst: Instruction,
) -> Option<OperationIdx> {
    if operands.assignments(op).len() != 1 {
        return None;
    }
    let asg_op = operands.assignments(op)[0];
    if ops.get(asg_op).instruction != inst {
        return None;
    }
    Some(asg_op)
}

fn is_memory_load_or_store(inst: Instruction) -> bool {
    matches!(
        inst,
        Instruction::Load
            | Instruction::Load16
            | Instruction::Load8
            | Instruction::Store
            | Instruction::Store16
            | Instruction::Store8
    )
}
