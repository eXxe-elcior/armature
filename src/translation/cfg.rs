// Port of Translation/ControlFlowGraph.cs.
//
// Mapping: C# `IntrusiveList<BasicBlock> Blocks` = `BlockArena::order`
// (block identity is separate from traversal order). The CFG
// struct holds computed data only; it borrows the arenas per call
// instead of owning references (C# holds the list).
//
// SplitEdge is ported here (its only consumer is SSA deconstruction).
// PRECONDITION: blocks.compute_predecessors() has been called so the C#
// `predecessor.GetSuccessor(i) == successor` check and the
// `splitBlock.Predecessors.Count == 0` safety check both work.
//
// POSTCONDITION: blocks.compute_predecessors() must be called after this
// (the predecessor lists are stale after mutating edges).

use crate::util::hash::FxHashSet as HashSet;
use std::collections::VecDeque;

use crate::ir::block::{BlockArena, BlockIndex};
use crate::ir::operand::{OperandArena, OperandIndex};
use crate::ir::types::OperandType;

pub struct ControlFlowGraph {
    pub entry: BlockIndex,
    /// C# PostOrderBlocks.
    pub post_order_blocks: Vec<BlockIndex>,
    /// C# PostOrderMap  indexed by `BasicBlockData::index` (dense after
    /// `update` renumbers).
    pub post_order_map: Vec<u32>,
    pub locals_count: u32,
}

impl ControlFlowGraph {
    pub fn new(blocks: &mut BlockArena, entry: BlockIndex, locals_count: u32) -> Self {
        let mut cfg = Self {
            entry,
            post_order_blocks: Vec::new(),
            post_order_map: Vec::new(),
            locals_count,
        };

        cfg.update(blocks);

        cfg
    }

    /// C# AllocateLocal: numbered local for the optimization passes.
    pub fn allocate_local(
        &mut self,
        operands: &mut OperandArena,
        typ: OperandType,
    ) -> OperandIndex {
        self.locals_count += 1;

        let result = operands.local(typ);
        operands.get_mut(result).number_local(self.locals_count);

        result
    }

    /// C# UpdateEntry. As in C#, the caller is responsible for having set
    /// `new_entry`'s index (C#: `new BasicBlock(cfg.Blocks.Count)`).
    pub fn update_entry(&mut self, blocks: &mut BlockArena, new_entry: BlockIndex) {
        blocks.add_successor(new_entry, self.entry);

        self.entry = new_entry;
        blocks.order_mut().insert(0, new_entry);
        self.update(blocks);
    }

    /// C# Update: drop unreachable blocks, then recompute the post-order
    /// traversal (iterative DFS with the re-push trick).
    ///
    /// Divergence: also recomputes predecessor lists. C# maintains them
    /// eagerly inside AddSuccessor/SetSuccessor; our BlockArena defers to
    /// a batch compute_predecessors(), so Update is the consistency point
    /// for passes that mutate edges (e.g. TailMerge) before Dominance/SSA.
    pub fn update(&mut self, blocks: &mut BlockArena) {
        self.remove_unreachable_blocks(blocks);
        blocks.compute_predecessors();

        let count = blocks.order().len();

        let mut visited: HashSet<BlockIndex> = HashSet::default();
        let mut block_stack: Vec<BlockIndex> = Vec::new();

        self.post_order_blocks.clear();
        self.post_order_blocks.resize(count, BlockIndex(u32::MAX));
        self.post_order_map.clear();
        self.post_order_map.resize(count, u32::MAX);

        visited.insert(self.entry);
        block_stack.push(self.entry);

        let mut index: usize = 0;

        while let Some(block) = block_stack.pop() {
            let mut visited_new = false;

            let data = blocks.get(block);
            for i in 0..data.successor_count as usize {
                let succ = data.successors[i].expect("successor slot within count");

                if visited.insert(succ) {
                    block_stack.push(block);
                    block_stack.push(succ);

                    visited_new = true;

                    break;
                }
            }

            if !visited_new {
                self.post_order_map[blocks.get(block).index as usize] = index as u32;

                self.post_order_blocks[index] = block;
                index += 1;
            }
        }
    }

    // C# RemoveUnreachableBlocks.
    fn remove_unreachable_blocks(&mut self, blocks: &mut BlockArena) {
        let mut visited: HashSet<BlockIndex> = HashSet::default();
        let mut work_queue: VecDeque<BlockIndex> = VecDeque::new();

        visited.insert(self.entry);
        work_queue.push_back(self.entry);

        while let Some(block) = work_queue.pop_front() {
            debug_assert!(blocks.get(block).index != u32::MAX, "Invalid block index.");

            let data = blocks.get(block);
            for i in 0..data.successor_count as usize {
                let succ = data.successors[i].expect("successor slot within count");

                if visited.insert(succ) {
                    work_queue.push_back(succ);
                }
            }
        }

        if visited.len() < blocks.order().len() {
            // Remove unreachable blocks and renumber.
            let mut index: u32 = 0;

            let order: Vec<BlockIndex> = blocks.order().to_vec();
            let mut new_order: Vec<BlockIndex> = Vec::with_capacity(visited.len());

            for block in order {
                if !visited.contains(&block) {
                    let data = blocks.get_mut(block);
                    data.successors = [None, None];
                    data.successor_count = 0;
                } else {
                    blocks.get_mut(block).index = index;
                    index += 1;
                    new_order.push(block);
                }
            }

            *blocks.order_mut() = new_order;
        }
    }

    /// C# SplitEdge. Insert a new block on the edge from `predecessor` to
    /// `successor`. The split block becomes the target of the predecessor's
    /// edge and the source of the edge to the successor.
    ///
    /// PRECONDITION: `blocks.compute_predecessors()` has been called.
    /// POSTCONDITION: `blocks.compute_predecessors()` must be called after.
    /// Static version of split_edge that doesn't require &mut self.
    /// Used by the register allocator which only has a shared ref to cfg.
    pub fn split_edge(
        blocks: &mut BlockArena,
        predecessor: BlockIndex,
        successor: BlockIndex,
    ) -> BlockIndex {
        let has_edge = {
            let d = blocks.get(predecessor);
            (0..d.successor_count as usize).any(|i| d.successors[i] == Some(successor))
        };
        if !has_edge {
            eprintln!(
                "split_edge: pred={:?}(idx={}) succ=[{:?},{:?}] cnt={} no edge to succ={:?}(idx={}) from {}:{}",
                predecessor,
                blocks.get(predecessor).index,
                blocks.get(predecessor).successors[0],
                blocks.get(predecessor).successors[1],
                blocks.get(predecessor).successor_count,
                successor,
                blocks.get(successor).index,
                std::file!(),
                std::line!(),
            );
            panic!("Predecessor and successor are not connected.");
        }

        let split_block = blocks.alloc();
        blocks.get_mut(split_block).index = blocks.order().len() as u32;

        {
            let d = blocks.get_mut(predecessor);
            for i in 0..d.successor_count as usize {
                if d.successors[i] == Some(successor) {
                    d.successors[i] = Some(split_block);
                }
            }
        }

        blocks.add_successor(split_block, successor);

        let succ_pos = blocks
            .order()
            .iter()
            .position(|&b| b == successor)
            .expect("successor must be in order list");
        blocks.order_mut().insert(succ_pos, split_block);

        split_block
    }
}
