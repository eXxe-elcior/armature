use crate::ir::block::BlockIndex;
use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Comparison, Instruction, OperandKind};

pub fn run_pass(ctx: &mut TranslationContext) {
    loop {
        let mut modified = false;

        let order_snapshot: Vec<BlockIndex> = ctx.blocks.order().to_vec();
        for &block in order_snapshot.iter().rev() {
            let ops_snapshot: Vec<OperationIdx> = ctx.blocks.get(block).ops.clone();
            for &node in ops_snapshot.iter().rev() {
                if ctx.ops.get(node).dead {
                    continue;
                }
                if is_unused(ctx, node) {
                    remove_node(ctx, block, node);
                    modified = true;
                    continue;
                } else if ctx.ops.get(node).instruction == Instruction::Phi {
                    continue;
                }

                super::constant_folding::run_pass(ctx, node);
                super::simplification::run_pass(ctx, node);

                if dest_is_single_local_var(ctx, node) {
                    if is_propagable_compare(ctx, node) {
                        let changed = propagate_compare(ctx, node);
                        modified |= changed;
                        if changed && is_unused(ctx, node) {
                            remove_node(ctx, block, node);
                        }
                    } else if is_propagable_copy(ctx, node) {
                        if propagate_copy(ctx, node) {
                            remove_node(ctx, block, node);
                            modified = true;
                        }
                    }
                }
            }
        }

        if cse_pass(ctx) {
            modified = true;
        }

        if licm_pass(ctx) {
            modified = true;
        }

        // Splice out everything marked dead this iteration in one O(n) pass per
        // block (instead of O(n) per individual removal).
        compact_dead(ctx);

        if !modified {
            break;
        }
    }
}

/// Remove all dead-marked operations from every block in one pass.
fn compact_dead(ctx: &mut TranslationContext) {
    let order: Vec<BlockIndex> = ctx.blocks.order().to_vec();
    for block in order {
        // Borrow-split: read dead flags via ctx.ops while mutating block ops.
        let dead: Vec<bool> = ctx
            .blocks
            .get(block)
            .ops
            .iter()
            .map(|&o| ctx.ops.get(o).dead)
            .collect();
        let mut i = 0;
        ctx.blocks.get_mut(block).ops.retain(|_| {
            let keep = !dead[i];
            i += 1;
            keep
        });
    }
}

/// Loop-Invariant Code Motion. Hoists pure, non-trapping operations whose
/// operands are all invariant in a loop out to the end of idom(header)  which
/// runs once before the loop and dominates every loop block, so the value is
/// identical and all (dominated) uses stay valid. Big win for hot loops, which
/// dominate game frame time. Conservative: only LocalVariable/Constant operands
/// (register/flag sources may change in the loop), excludes Divide (trap-ish).
fn licm_pass(ctx: &mut TranslationContext) -> bool {
    use crate::util::hash::FxHashMap;

    let all: Vec<BlockIndex> = ctx.blocks.order().to_vec();
    if all.is_empty() {
        return false;
    }

    // Local predecessor map from successors (don't depend on stale state).
    let mut preds: FxHashMap<BlockIndex, Vec<BlockIndex>> = FxHashMap::default();
    for &b in &all {
        let data = ctx.blocks.get(b);
        for s in 0..data.successor_count as usize {
            if let Some(succ) = data.successors[s] {
                preds.entry(succ).or_default().push(b);
            }
        }
    }

    // operand -> defining block (single-dest local defs only).
    let mut def_block: FxHashMap<u32, BlockIndex> = FxHashMap::default();
    for &b in &all {
        for &node in &ctx.blocks.get(b).ops.clone() {
            let dests = ctx.ops.dests(node);
            if dests.len() == 1 && dests[0].0 != u32::MAX {
                def_block.insert(dests[0].0, b);
            }
        }
    }

    let mut modified = false;

    // Find natural loops: edge B -> H where H dominates B (back-edge).
    for &b in &all {
        let data = ctx.blocks.get(b);
        let succs: Vec<BlockIndex> = (0..data.successor_count as usize)
            .filter_map(|s| data.successors[s])
            .collect();
        for header in succs {
            if !super::loop_analysis::dominates(&ctx.blocks, header, b) {
                continue;
            }
            let Some(preheader) = ctx.blocks.get(header).idom else {
                continue;
            };
            // preheader must be outside the loop (it dominates header, so it is).
            let body = super::loop_analysis::natural_loop(header, b, &preds);
            if body.contains(&preheader) {
                continue;
            }
            modified |= hoist_loop(ctx, &body, preheader, &mut def_block);
        }
    }

    modified
}

fn hoist_loop(
    ctx: &mut TranslationContext,
    body: &crate::util::hash::FxHashSet<BlockIndex>,
    preheader: BlockIndex,
    def_block: &mut crate::util::hash::FxHashMap<u32, BlockIndex>,
) -> bool {
    let mut modified = false;
    loop {
        let mut moved_any = false;
        for &b in body {
            if b == preheader {
                continue;
            }
            for node in ctx.blocks.get(b).ops.clone() {
                if ctx.ops.get(node).dead {
                    continue;
                }
                if !is_licm_hoistable(ctx.ops.get(node).instruction)
                    || !dest_is_single_local_var(ctx, node)
                {
                    continue;
                }
                let srcs: Vec<OperandIndex> = ctx.ops.sources(node).to_vec();
                let invariant = srcs.iter().all(|&s| {
                    if s.0 == u32::MAX {
                        return false;
                    }
                    match ctx.operands.get(s).kind {
                        OperandKind::Constant => true,
                        OperandKind::LocalVariable => def_block
                            .get(&s.0)
                            .map(|db| !body.contains(db))
                            .unwrap_or(false),
                        // Register/flag/memory sources may change in the loop.
                        _ => false,
                    }
                });
                if !invariant {
                    continue;
                }
                // Move node to the end of preheader, before its terminator.
                move_op_before_terminator(ctx, b, node, preheader);
                let dest = ctx.ops.dests(node)[0];
                def_block.insert(dest.0, preheader);
                moved_any = true;
                modified = true;
            }
        }
        if !moved_any {
            break;
        }
    }
    modified
}

/// Pure + deterministic + non-trapping instructions safe to hoist.
fn is_licm_hoistable(inst: Instruction) -> bool {
    use Instruction::*;
    is_cse_pure(inst) && !matches!(inst, Divide | DivideUI)
}

fn move_op_before_terminator(
    ctx: &mut TranslationContext,
    from: BlockIndex,
    node: OperationIdx,
    to: BlockIndex,
) {
    {
        let ops = &mut ctx.blocks.get_mut(from).ops;
        if let Some(pos) = ops.iter().position(|&o| o == node) {
            ops.remove(pos);
        }
    }
    let has_term = {
        let ops = &ctx.blocks.get(to).ops;
        ops.last()
            .map(|&last| is_terminator(ctx.ops.get(last).instruction))
            .unwrap_or(false)
    };
    let ops = &mut ctx.blocks.get_mut(to).ops;
    let insert_at = if has_term { ops.len() - 1 } else { ops.len() };
    ops.insert(insert_at, node);
}

fn is_terminator(inst: Instruction) -> bool {
    matches!(
        inst,
        Instruction::BranchIf | Instruction::Return | Instruction::Tailcall
    )
}

/// Common Subexpression Elimination. Within each block (forward order, so a def
/// dominates later uses), a pure value-producing operation with the same opcode,
/// the same SSA source operands and the same result type computes an identical
/// value  replace later duplicates' uses with the first result and delete them.
/// Removes the redundant address arithmetic, flag setups and integer math that
/// recur heavily in guest code, especially hot loops.
type CseKey = (u16, Vec<u32>, u8);

fn cse_pass(ctx: &mut TranslationContext) -> bool {
    // Global CSE over the dominator tree: a value computed in block A is
    // available in every block A dominates (A always runs first). DFS the
    // dom-tree with a scoped expression map  entries added in a block are
    // visible to its descendants and rolled back on the way out. Blocks with
    // no idom (added after dominance, unreachable) fall back to block-local.
    let all: Vec<BlockIndex> = ctx.blocks.order().to_vec();

    // Build dom-tree children from idom, and find the root (idom == self).
    let mut children: crate::util::hash::FxHashMap<BlockIndex, Vec<BlockIndex>> =
        crate::util::hash::FxHashMap::default();
    let mut root: Option<BlockIndex> = None;
    for &b in &all {
        match ctx.blocks.get(b).idom {
            Some(idom) if idom == b => root = Some(b),
            Some(idom) => children.entry(idom).or_default().push(b),
            None => {}
        }
    }

    let mut seen: crate::util::hash::FxHashMap<CseKey, OperandIndex> =
        crate::util::hash::FxHashMap::default();
    let mut modified = false;

    if let Some(root) = root {
        cse_dom(ctx, root, &children, &mut seen, &mut modified);
    }

    // Block-local sweep for any block the dom DFS didn't reach (idom == None).
    for &b in &all {
        if ctx.blocks.get(b).idom.is_none() {
            let mut local: crate::util::hash::FxHashMap<CseKey, OperandIndex> =
                crate::util::hash::FxHashMap::default();
            cse_block(ctx, b, &mut local, &mut modified, &mut Vec::new());
        }
    }

    modified
}

/// DFS the dominator tree; `seen` holds expressions available from dominators.
fn cse_dom(
    ctx: &mut TranslationContext,
    block: BlockIndex,
    children: &crate::util::hash::FxHashMap<BlockIndex, Vec<BlockIndex>>,
    seen: &mut crate::util::hash::FxHashMap<CseKey, OperandIndex>,
    modified: &mut bool,
) {
    // Keys this block newly inserts  rolled back when leaving its subtree.
    let mut added: Vec<CseKey> = Vec::new();
    cse_block(ctx, block, seen, modified, &mut added);

    if let Some(kids) = children.get(&block) {
        for &kid in &kids.clone() {
            cse_dom(ctx, kid, children, seen, modified);
        }
    }

    for key in added {
        seen.remove(&key);
    }
}

/// CSE one block against `seen`; records newly added keys into `added`.
fn cse_block(
    ctx: &mut TranslationContext,
    block: BlockIndex,
    seen: &mut crate::util::hash::FxHashMap<CseKey, OperandIndex>,
    modified: &mut bool,
    added: &mut Vec<CseKey>,
) {
    let nodes: Vec<OperationIdx> = ctx.blocks.get(block).ops.clone();
    for node in nodes {
        if ctx.ops.get(node).dead {
            continue;
        }
        if ctx.ops.get(node).is_phi() || !is_cse_pure(ctx.ops.get(node).instruction) {
            continue;
        }
        if !dest_is_single_local_var(ctx, node) {
            continue;
        }
        let srcs: Vec<u32> = ctx.ops.sources(node).iter().map(|o| o.0).collect();
        if srcs.iter().any(|&s| s == u32::MAX) {
            continue;
        }
        let dest = ctx.ops.dests(node)[0];
        let key: CseKey = (
            ctx.ops.get(node).instruction as u16,
            srcs,
            ctx.operands.get(dest).typ as u8,
        );
        if let Some(&cached) = seen.get(&key) {
            replace_all_uses(ctx, dest, cached);
            remove_node(ctx, block, node);
            *modified = true;
        } else {
            seen.insert(key.clone(), dest);
            added.push(key);
        }
    }
}

/// Pure, deterministic-from-operands, single-result instructions safe to CSE.
/// Excludes everything touching memory/flags-as-state/calls/control flow.
fn is_cse_pure(inst: Instruction) -> bool {
    use Instruction::*;
    matches!(
        inst,
        Add | Subtract
            | Multiply
            | Multiply64HighSI
            | Multiply64HighUI
            | Divide
            | DivideUI
            | BitwiseAnd
            | BitwiseOr
            | BitwiseExclusiveOr
            | BitwiseNot
            | Negate
            | ShiftLeft
            | ShiftRightSI
            | ShiftRightUI
            | RotateRight
            | SignExtend8
            | SignExtend16
            | SignExtend32
            | ZeroExtend8
            | ZeroExtend16
            | ZeroExtend32
            | ConvertI64ToI32
            | Compare
            | ConditionalSelect
            | CountLeadingZeros
            | ByteSwap
    )
}

/// Repoint every use of `from` to `to` (def-use rewrite; phi-aware).
fn replace_all_uses(ctx: &mut TranslationContext, from: OperandIndex, to: OperandIndex) {
    let uses: Vec<OperationIdx> = ctx.operands.uses(from).to_vec();
    for use_op in uses {
        if ctx.ops.get(use_op).is_phi() {
            let n = ctx.ops.phi_inputs(use_op).len();
            for i in 0..n {
                if ctx.ops.phi_inputs(use_op)[i].value == from {
                    ctx.ops.set_phi_value(use_op, i, to, &mut ctx.operands);
                }
            }
            continue;
        }
        let n = ctx.ops.sources(use_op).len();
        for i in 0..n {
            if ctx.ops.sources(use_op)[i] == from {
                ctx.ops.set_source(use_op, i, to, &mut ctx.operands);
            }
        }
    }
}

pub fn remove_unused_nodes(ctx: &mut TranslationContext) {
    loop {
        let mut modified = false;

        let order_snapshot: Vec<BlockIndex> = ctx.blocks.order().to_vec();
        for &block in order_snapshot.iter().rev() {
            let ops_snapshot: Vec<OperationIdx> = ctx.blocks.get(block).ops.clone();
            for &node in ops_snapshot.iter().rev() {
                if ctx.ops.get(node).dead {
                    continue;
                }
                if is_unused(ctx, node) {
                    remove_node(ctx, block, node);
                    modified = true;
                }
            }
        }

        compact_dead(ctx);

        if !modified {
            break;
        }
    }
}

fn propagate_compare(ctx: &mut TranslationContext, comp_op: OperationIdx) -> bool {
    let dest = ctx.ops.dests(comp_op)[0];
    let srcs = ctx.ops.sources(comp_op);
    let src1 = srcs[0];
    let src2 = srcs[1];
    let comp = srcs[2];
    let comp_type_val = ctx.operands.get(comp).as_i32();

    let comp_type = match comp_type_val as u8 {
        v if v == Comparison::Equal as u8 => Comparison::Equal,
        v if v == Comparison::NotEqual as u8 => Comparison::NotEqual,
        v if v == Comparison::Greater as u8 => Comparison::Greater,
        v if v == Comparison::LessOrEqual as u8 => Comparison::LessOrEqual,
        v if v == Comparison::GreaterUI as u8 => Comparison::GreaterUI,
        v if v == Comparison::LessOrEqualUI as u8 => Comparison::LessOrEqualUI,
        v if v == Comparison::GreaterOrEqual as u8 => Comparison::GreaterOrEqual,
        v if v == Comparison::Less as u8 => Comparison::Less,
        v if v == Comparison::GreaterOrEqualUI as u8 => Comparison::GreaterOrEqualUI,
        v if v == Comparison::LessUI as u8 => Comparison::LessUI,
        _ => return false,
    };

    let mut modified = false;
    let uses: Vec<OperationIdx> = ctx.operands.uses(dest).to_vec();

    for &use_op in &uses {
        if let Some(other_comp_type) = is_zero_branch(ctx, use_op) {
            let prop_comp_type = if other_comp_type == Comparison::NotEqual {
                comp_type
            } else if other_comp_type == Comparison::Equal {
                comp_type.invert()
            } else {
                continue;
            };

            let comp_const = ctx.operands.constant_i32(prop_comp_type as i32);
            ctx.ops.set_source(use_op, 0, src1, &mut ctx.operands);
            ctx.ops.set_source(use_op, 1, src2, &mut ctx.operands);
            ctx.ops.set_source(use_op, 2, comp_const, &mut ctx.operands);

            modified = true;
        }
    }

    modified
}

fn propagate_copy(ctx: &mut TranslationContext, copy_op: OperationIdx) -> bool {
    let dest = ctx.ops.dests(copy_op)[0];
    let source = ctx.ops.sources(copy_op)[0];

    let uses: Vec<OperationIdx> = ctx.operands.uses(dest).to_vec();

    for &use_op in &uses {
        // C# phis keep their values in the regular source list; our IR
        // stores them in phi_inputs, so they need the phi-specific setter.
        if ctx.ops.get(use_op).is_phi() {
            let input_count = ctx.ops.phi_inputs(use_op).len();
            for index in 0..input_count {
                if ctx.ops.phi_inputs(use_op)[index].value == dest {
                    ctx.ops
                        .set_phi_value(use_op, index, source, &mut ctx.operands);
                }
            }
            continue;
        }
        let src_count = ctx.ops.sources(use_op).len();
        for index in 0..src_count {
            if ctx.ops.sources(use_op)[index] == dest {
                ctx.ops.set_source(use_op, index, source, &mut ctx.operands);
            }
        }
    }

    true
}

fn remove_node(ctx: &mut TranslationContext, _block: BlockIndex, node: OperationIdx) {
    // Mark dead instead of splicing out of the block's Vec. The old
    // `iter().position()` + `Vec::remove` was O(n) per call → O(n²) when DCE
    // removes many nodes from a big function. Marking is O(1); dead nodes are
    // skipped by every node loop and compacted out once per pass (compact_dead).
    if ctx.ops.get(node).dead {
        return;
    }
    ctx.ops.get_mut(node).dead = true;

    let src_count = ctx.ops.sources(node).len();
    for index in 0..src_count {
        ctx.ops
            .set_source(node, index, OperandIndex(u32::MAX), &mut ctx.operands);
    }

    // Phi values live in phi_inputs, not the source list  drop their
    // use entries too (C# phis store values as regular sources).
    if ctx.ops.get(node).is_phi() {
        let input_count = ctx.ops.phi_inputs(node).len();
        for index in 0..input_count {
            let value = ctx.ops.phi_inputs(node)[index].value;
            if value.0 != u32::MAX {
                crate::ir::operand::operand_remove_use(&mut ctx.operands, value, node);
                ctx.ops.phi_inputs_mut(node)[index].value = OperandIndex(u32::MAX);
            }
        }
    }

    let dests = ctx.ops.dests(node).to_vec();
    for (i, &dest) in dests.iter().enumerate() {
        if dest.0 != u32::MAX {
            debug_assert!(ctx.operands.uses(dest).is_empty());
            ctx.ops
                .set_destination(node, i, OperandIndex(u32::MAX), &mut ctx.operands);
        }
    }
}

fn is_unused(ctx: &TranslationContext, node: OperationIdx) -> bool {
    dest_is_single_local_var(ctx, node)
        && {
            let dest = ctx.ops.dests(node)[0];
            ctx.operands.uses(dest).is_empty()
        }
        && !has_side_effects(ctx, node)
}

fn dest_is_single_local_var(ctx: &TranslationContext, node: OperationIdx) -> bool {
    let dests = ctx.ops.dests(node);
    dests.len() == 1 && ctx.operands.get(dests[0]).kind == OperandKind::LocalVariable
}

fn has_side_effects(ctx: &TranslationContext, node: OperationIdx) -> bool {
    let inst = ctx.ops.get(node).instruction;
    matches!(
        inst,
        Instruction::Call
            | Instruction::Tailcall
            | Instruction::CompareAndSwap
            | Instruction::CompareAndSwap16
            | Instruction::CompareAndSwap8
    )
}

fn is_propagable_compare(ctx: &TranslationContext, node: OperationIdx) -> bool {
    ctx.ops.get(node).instruction == Instruction::Compare
}

fn is_propagable_copy(ctx: &TranslationContext, node: OperationIdx) -> bool {
    if ctx.ops.get(node).instruction != Instruction::Copy {
        return false;
    }
    let srcs = ctx.ops.sources(node);
    let dests = ctx.ops.dests(node);
    dests.len() == 1
        && srcs.len() == 1
        && ctx.operands.get(dests[0]).typ == ctx.operands.get(srcs[0]).typ
}

fn is_zero_branch(ctx: &TranslationContext, operation: OperationIdx) -> Option<Comparison> {
    if ctx.ops.get(operation).instruction != Instruction::BranchIf {
        return None;
    }

    let srcs = ctx.ops.sources(operation);
    let src1 = srcs[0];
    let src2 = srcs[1];

    let has_zero = (ctx.operands.get(src1).kind == OperandKind::Constant
        && ctx.operands.get(src1).value == 0)
        || (ctx.operands.get(src2).kind == OperandKind::Constant
            && ctx.operands.get(src2).value == 0);

    if !has_zero {
        return None;
    }

    let comp = ctx.operands.get(srcs[2]).as_i32();
    let comp_type = match comp as u8 {
        v if v == Comparison::Equal as u8 => Comparison::Equal,
        v if v == Comparison::NotEqual as u8 => Comparison::NotEqual,
        v if v == Comparison::Greater as u8 => Comparison::Greater,
        v if v == Comparison::LessOrEqual as u8 => Comparison::LessOrEqual,
        v if v == Comparison::GreaterUI as u8 => Comparison::GreaterUI,
        v if v == Comparison::LessOrEqualUI as u8 => Comparison::LessOrEqualUI,
        v if v == Comparison::GreaterOrEqual as u8 => Comparison::GreaterOrEqual,
        v if v == Comparison::Less as u8 => Comparison::Less,
        v if v == Comparison::GreaterOrEqualUI as u8 => Comparison::GreaterOrEqualUI,
        v if v == Comparison::LessUI as u8 => Comparison::LessUI,
        _ => return None,
    };

    Some(comp_type)
}
