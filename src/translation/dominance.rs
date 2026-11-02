// Port of Translation/Dominance.cs.
//
// Cooper/Harvey/Kennedy simple fast dominance algorithm:
// https://www.cs.rice.edu/~keith/EMBED/dom.pdf
//
// Precondition: `blocks.compute_predecessors()` has been called before
// `find_dominators`.
//
// Divergence: C# stores `ImmediateDominator` / `DominanceFrontiers` as
// fields on `BasicBlock`; our `BasicBlockData` already has `idom` and
// `dom_frontiers` for exactly this purpose (see ir/block.rs).

use crate::ir::block::{BlockArena, BlockIndex};
use crate::translation::cfg::ControlFlowGraph;

/// C# Dominance.FindDominators.
pub fn find_dominators(cfg: &ControlFlowGraph, blocks: &mut BlockArena) {
    blocks.get_mut(cfg.entry).idom = Some(cfg.entry);

    let last_idx = match cfg.post_order_blocks.len().checked_sub(1) {
        Some(0) | None => return,
        Some(n) => n,
    };

    let mut modified = true;
    while modified {
        modified = false;
        for blk_index in (0..last_idx).rev() {
            let block = cfg.post_order_blocks[blk_index];

            // Clone predecessors: avoids holding an arena reference across
            // the mutable get_mut call below.
            let preds = blocks.get(block).predecessors.clone();

            let mut new_idom: Option<BlockIndex> = None;
            for &pred in &preds {
                if blocks.get(pred).idom.is_some() {
                    new_idom = Some(match new_idom {
                        Some(curr) => intersect(cfg, blocks, pred, curr),
                        None => pred,
                    });
                }
            }

            if blocks.get(block).idom != new_idom {
                blocks.get_mut(block).idom = new_idom;
                modified = true;
            }
        }
    }
}

/// C# local function Intersect inside FindDominators.
///
/// Compare post-order numbers, not BlockIdx values.
fn intersect(
    cfg: &ControlFlowGraph,
    blocks: &BlockArena,
    mut b1: BlockIndex,
    mut b2: BlockIndex,
) -> BlockIndex {
    while b1 != b2 {
        while cfg.post_order_map[blocks.get(b1).index as usize]
            < cfg.post_order_map[blocks.get(b2).index as usize]
        {
            b1 = blocks.get(b1).idom.expect("idom must be set");
        }
        while cfg.post_order_map[blocks.get(b2).index as usize]
            < cfg.post_order_map[blocks.get(b1).index as usize]
        {
            b2 = blocks.get(b2).idom.expect("idom must be set");
        }
    }
    b1
}

/// C# Dominance.FindDominanceFrontiers.
pub fn find_dominance_frontiers(blocks: &mut BlockArena) {
    // Snapshot order: avoids holding an immutable borrow on `blocks`
    // across `get_mut` calls inside the loop.
    let order: Vec<BlockIndex> = blocks.order().to_vec();
    for &block in &order {
        let pred_count = blocks.get(block).predecessors.len();
        if pred_count < 2 {
            continue;
        }

        // Clone predecessors: avoids borrow issues from mutable push below.
        let preds = blocks.get(block).predecessors.clone();
        let block_idom = blocks
            .get(block)
            .idom
            .expect("find_dominators must run before find_dominance_frontiers");

        for &pred in &preds {
            let mut current = pred;
            while current != block_idom {
                blocks.get_mut(current).dom_frontiers.push(block);
                current = blocks.get(current).idom.expect("idom must be set");
            }
        }
    }
}
