// Port of Translation/RegisterToLocal.cs  the LCQ path's SSA
// replacement: every guest register operand becomes one shared local.
//
// C# keys the map on the Register struct (Index + Type); the Rust key is
// the equivalent (RegisterType, index) pair.

use crate::util::hash::FxHashMap as HashMap;

use crate::ir::operand::OperandIndex;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{OperandKind, RegisterType};
use crate::translation::cfg::ControlFlowGraph;

pub fn rename(cfg: &ControlFlowGraph, ctx: &mut TranslationContext) {
    let mut register_to_local_map: HashMap<(RegisterType, u32), OperandIndex> = HashMap::default();

    // C# local function GetLocal(op).
    let mut get_local = |ctx: &mut TranslationContext, op: OperandIndex| -> OperandIndex {
        let data = ctx.operands.get(op);
        let register = (data.reg_type(), data.reg_index());
        let typ = data.typ;

        *register_to_local_map
            .entry(register)
            .or_insert_with(|| ctx.operands.local(typ))
    };

    let _ = cfg; // C# iterates cfg.Blocks  identical to the arena order.

    for block_pos in 0..ctx.blocks.order().len() {
        let block = ctx.blocks.order()[block_pos];

        for op_pos in 0..ctx.blocks.get(block).ops.len() {
            let node = ctx.blocks.get(block).ops[op_pos];

            if let Some(&dest) = ctx.ops.dests(node).first() {
                if ctx.operands.get(dest).kind == OperandKind::Register {
                    let local = get_local(ctx, dest);
                    let (ops, operands) = (&mut ctx.ops, &mut ctx.operands);
                    ops.set_destination(node, 0, local, operands);
                }
            }

            for index in 0..ctx.ops.sources(node).len() {
                let source = ctx.ops.sources(node)[index];

                if ctx.operands.get(source).kind == OperandKind::Register {
                    let local = get_local(ctx, source);
                    let (ops, operands) = (&mut ctx.ops, &mut ctx.operands);
                    ops.set_source(node, index, local, operands);
                }
            }
        }
    }
}
