use crate::ir::block::{BlockArena, BlockFrequency, BlockIndex};
use crate::ir::operand::OperandArena;
use crate::ir::operation::OperationArena;
use crate::ir::types::{Comparison, Instruction};
use crate::translation::cfg::ControlFlowGraph;
use crate::util::hash::FxHashSet;

pub fn run_pass(
    cfg: &mut ControlFlowGraph,
    blocks: &mut BlockArena,
    ops: &mut OperationArena,
    operands: &mut OperandArena,
) {
    let order_snapshot: Vec<BlockIndex> = blocks.order().to_vec();
    let last_block = order_snapshot[order_snapshot.len() - 1];

    // 1. Move cold blocks to the end
    let mut new_order: Vec<BlockIndex> = Vec::with_capacity(blocks.order().len());
    let mut cold_blocks: Vec<BlockIndex> = Vec::new();

    for &block in &order_snapshot {
        if blocks.get(block).frequency == BlockFrequency::Cold {
            cold_blocks.push(block);
        } else {
            new_order.push(block);
        }
        if block == last_block {
            break;
        }
    }

    // Append remaining blocks that were after the original last block (if any).
    let mut seen_last = false;
    for &block in &order_snapshot {
        if block == last_block {
            seen_last = true;
            continue;
        }
        if seen_last {
            new_order.push(block);
        }
    }

    new_order.extend(cold_blocks);

    // 2. Loop-aware block reordering
    // After cold-block separation, make each natural loop's blocks
    // contiguous. Process innermost loops first so outer loops see inner
    // ones already as a single contiguous segment. Blocks between loop
    // body blocks (e.g. exit blocks emitted mid-loop) are moved after
    // the loop body, improving I-cache locality for hot paths.
    let loops = super::loop_analysis::find_all_loops(blocks);
    if !loops.is_empty() {
        let mut sorted = loops;
        super::loop_analysis::sort_innermost_first(&mut sorted);

        for li in &sorted {
            let loop_set: &FxHashSet<BlockIndex> = &li.body;

            // Find positions of all blocks in this loop.
            let positions: Vec<(usize, BlockIndex)> = new_order
                .iter()
                .enumerate()
                .filter(|(_, b)| loop_set.contains(b))
                .map(|(i, b)| (i, *b))
                .collect();

            if positions.len() <= 1 {
                continue;
            }

            let first = positions[0].0;
            let last = positions.last().unwrap().0;

            // Check if already contiguous (no non-loop blocks in between).
            let contiguous = positions
                .iter()
                .enumerate()
                .all(|(j, (i, _))| *i == first + j);
            if contiguous {
                continue;
            }

            // Collect loop blocks in their current relative order.
            let loop_blocks: Vec<BlockIndex> = positions.iter().map(|(_, b)| *b).collect();

            // Collect non-loop blocks between first and last.
            let between: Vec<BlockIndex> = (first..=last)
                .filter(|&i| !loop_set.contains(&new_order[i]))
                .map(|i| new_order[i])
                .collect();

            // Rebuild: prefix + loop_blocks (hot, contiguous) + between (cold-ish exits, after) + suffix.
            let mut rebuilt = Vec::with_capacity(new_order.len());
            rebuilt.extend_from_slice(&new_order[..first]);
            rebuilt.extend(&loop_blocks);
            rebuilt.extend(&between);
            rebuilt.extend_from_slice(&new_order[(last + 1)..]);

            new_order = rebuilt;
        }

        for (i, &b) in new_order.iter().enumerate() {
            blocks.get_mut(b).index = i as u32;
        }
        *blocks.order_mut() = new_order;
    }

    // 3. Optimize branch conditions for fallthrough
    let mut update = false;
    let order: Vec<BlockIndex> = blocks.order().to_vec();

    for i in 0..order.len() {
        let block = order[i];
        let next_block = if i + 1 < order.len() {
            Some(order[i + 1])
        } else {
            None
        };

        if blocks.get(block).successor_count == 2 {
            let ops_list = blocks.get(block).ops.clone();
            let branch_op = ops_list.last().copied().unwrap();
            debug_assert!(ops.get(branch_op).instruction == Instruction::BranchIf);

            let false_succ = blocks.get(block).successors[0].unwrap();
            let true_succ = blocks.get(block).successors[1].unwrap();

            if Some(true_succ) == next_block {
                let srcs = ops.sources(branch_op);
                let comp_val = operands.get(srcs[2]).as_i32();
                let comp = match comp_val as u8 {
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
                    _ => unreachable!(),
                };
                let comp_inv = comp.invert();
                let comp_const = operands.constant_i32(comp_inv as i32);
                ops.set_source(branch_op, 2, comp_const, operands);

                blocks.get_mut(block).successors[0] = Some(true_succ);
                blocks.get_mut(block).successors[1] = Some(false_succ);

                update = true;
            }
        }
    }

    if update {
        cfg.update(blocks);
    }
}
