use crate::ir::operation::OperationIdx;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockIndex(pub u32);

define_index!(BlockIndex);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockFrequency {
    Default,
    Cold,
}

impl Default for BlockFrequency {
    fn default() -> Self {
        Self::Default
    }
}

/// A basic block in the IR control-flow graph.
///
/// Design notes (matching C# ARMeilleure's BasicBlock):
///
/// - Successors are stored inline (max 2) as a fixed-size array, matching
///   C#'s `_succ0` / `_succ1` fields. AArch64 control flow has at most
///   2 out-edges per block (branch-true + fallthrough).
///
/// - Predecessors are NOT written during emission. C#'s `AddSuccessor`
///   writes both `from.successors` and `to.predecessors` simultaneously,
///   which would require two `&mut` into the same `Vec`  illegal in Rust.
///   Since no code reads `predecessors` during emission (verified by grep),
///   we defer population to `BlockArena::compute_predecessors()`, a single
///   post-pass that runs before SSA construction.
///
/// - `ops` uses simple `Vec` + `push`. C#'s `EmitterContext.Add()` calls
///   `Operations.AddLast()` directly (NOT `BasicBlock.Append()`). The
///   "insert before terminator" logic in `Append()` is for optimization
///   passes, not the emitter hot path.
#[derive(Debug, Clone)]
pub struct BasicBlockData {
    pub index: u32,
    pub frequency: BlockFrequency,

    /// Operations in this block, in order.
    /// Emitted via simple push (matching C# AddLast in the emitter path).
    pub ops: Vec<OperationIdx>,

    /// Successor blocks, stored inline (max 2).
    /// Index 0 = fallthrough, Index 1 = conditional-true (if present).
    /// This matches C#'s `_succ0` / `_succ1` inline fields.
    pub successors: [Option<BlockIndex>; 2],

    /// How many successors are currently set (0, 1, or 2).
    pub successor_count: u8,

    /// Predecessor blocks  populated by `BlockArena::compute_predecessors()`
    /// AFTER all emission is complete. C# writes this eagerly in
    /// `AddSuccessor()`, but since nothing reads it during emission, we
    /// defer to avoid double-`&mut` borrow conflicts.
    pub predecessors: Vec<BlockIndex>,

    pub idom: Option<BlockIndex>,
    pub dom_frontiers: Vec<BlockIndex>,
}

impl Default for BasicBlockData {
    fn default() -> Self {
        Self::new(0)
    }
}

impl BasicBlockData {
    pub fn new(index: u32) -> Self {
        Self {
            index,
            frequency: BlockFrequency::Default,
            ops: Vec::new(),
            successors: [None, None],
            successor_count: 0,
            predecessors: Vec::new(),
            idom: None,
            dom_frontiers: Vec::new(),
        }
    }
}

/// Arena-allocated storage for all basic blocks in a translation unit.
///
/// Identity vs order  C# conflates these, we split them:
///
/// - `blocks` is the arena: `BlockIdx` = slot, assigned at `alloc()`,
///   never moves. This is block *identity* (C#: the object reference).
/// - `order` is the emission-order list (C#: `_irBlocks`). A block enters
///   `order` only when it becomes the current block (`NewNextBlock` /
///   `MarkLabel`), NOT when it is allocated. Blocks created eagerly by
///   `BranchToLabel` sit outside `order` (C#: `new BasicBlock()` with
///   `Index = -1`, not yet in `_irBlocks`) until their label is marked.
///
/// `BasicBlockData::index` is the position in `order` (u32::MAX until
/// assigned), matching C# `BasicBlock.Index`. Codegen layout and CFG
/// construction consume `order`, never raw arena order.
pub struct BlockArena {
    blocks: Vec<BasicBlockData>,
    order: Vec<BlockIndex>,
}

impl BlockArena {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            order: Vec::new(),
        }
    }

    /// Clear all state for arena reuse, retaining capacity.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.order.clear();
    }

    /// Allocate a new block and return its index.
    ///
    /// The block starts UNORDERED (`index == u32::MAX`), matching C#'s
    /// `new BasicBlock()` default of `Index = -1`. Call `push_order()`
    /// when the block becomes current.
    pub fn alloc(&mut self) -> BlockIndex {
        let idx = BlockIndex(self.blocks.len() as u32);
        self.blocks.push(BasicBlockData::new(u32::MAX));
        idx
    }

    /// Append a block to the emission-order list and assign its index.
    ///
    /// Matches C#: `block.Index = _irBlocks.Count; _irBlocks.AddLast(block);`
    /// (both the `NewNextBlock` and `MarkLabel` paths).
    pub fn push_order(&mut self, idx: BlockIndex) {
        self.blocks[idx.0 as usize].index = self.order.len() as u32;
        self.order.push(idx);
    }

    /// Blocks in emission order (C# `_irBlocks`). This is what CFG
    /// construction and codegen layout consume.
    #[inline]
    pub fn order(&self) -> &[BlockIndex] {
        &self.order
    }

    /// Mutable emission order  for ControlFlowGraph only (C#: the CFG
    /// owns and mutates the IntrusiveList directly). Emission code must
    /// keep going through push_order.
    #[inline]
    pub fn order_mut(&mut self) -> &mut Vec<BlockIndex> {
        &mut self.order
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    #[inline]
    pub fn get(&self, idx: BlockIndex) -> &BasicBlockData {
        &self.blocks[idx.0 as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, idx: BlockIndex) -> &mut BasicBlockData {
        &mut self.blocks[idx.0 as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = (BlockIndex, &BasicBlockData)> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (BlockIndex(i as u32), b))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (BlockIndex, &mut BasicBlockData)> {
        self.blocks
            .iter_mut()
            .enumerate()
            .map(|(i, b)| (BlockIndex(i as u32), b))
    }

    /// Add a successor edge DURING EMISSION.
    ///
    /// Only writes the `from` block's successor slot  the `to` block's
    /// `predecessors` is NOT updated here. This avoids the double-`&mut`
    /// problem that would arise from writing both sides simultaneously
    /// (as C#'s `BasicBlock.AddSuccessor` does).
    ///
    /// Call `compute_predecessors()` once after all emission to backfill
    /// the predecessor lists.
    pub fn add_successor(&mut self, from: BlockIndex, to: BlockIndex) {
        let b = &mut self.blocks[from.0 as usize];
        assert!(
            b.successor_count < 2,
            "BasicBlock can only have 2 successors"
        );
        b.successors[b.successor_count as usize] = Some(to);
        b.successor_count += 1;
    }

    /// Backfill all predecessor lists from the stored successor edges.
    ///
    /// C# does this eagerly in `BasicBlock.AddSuccessor()`. We defer it
    /// because no code reads `predecessors` during emission (verified:
    /// all reads are in SsaConstruction, Dominance, RegisterUsage, and
    /// optimization passes). This single pass runs once after emission
    /// and is equivalent in all observable behavior.
    ///
    /// Safety: each loop iteration accesses `self.blocks[i]` (read) and
    /// `self.blocks[to.0 as usize]` (write) where `i != to.0 as usize`
    /// (no self-loops in the CFG). The indices are always distinct, so
    /// Rust's slice indexing produces no aliasing. Multiple blocks may
    /// share the same successor (common merge point)  that's fine,
    /// each `push` happens in a different iteration on the same `Vec`.
    pub fn compute_predecessors(&mut self) {
        // Idempotent: CFG UpdateEntry re-runs this after
        // mutating edges.
        for b in &mut self.blocks {
            b.predecessors.clear();
        }
        for i in 0..self.blocks.len() {
            let from = BlockIndex(i as u32);
            for s in 0..self.blocks[i].successor_count as usize {
                if let Some(to) = self.blocks[i].successors[s] {
                    self.blocks[to.0 as usize].predecessors.push(from);
                }
            }
        }
    }

    /// Append an operation to a block. Simple push to end.
    ///
    /// C#'s `EmitterContext.Add()` calls `Operations.AddLast()` directly,
    /// NOT `BasicBlock.Append()`. The "insert before terminator" logic
    /// in `Append()` is only used by optimization passes. The emitter
    /// path guarantees no ops are added after a terminator because
    /// `_needsNewBlock` creates a fresh block first.
    pub fn append_op(&mut self, block: BlockIndex, op_idx: OperationIdx) {
        self.blocks[block.0 as usize].ops.push(op_idx);
    }

    pub fn into_inner(self) -> Vec<BasicBlockData> {
        self.blocks
    }
}

impl Default for BlockArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<BlockIndex> for BlockArena {
    type Output = BasicBlockData;
    #[inline]
    fn index(&self, idx: BlockIndex) -> &Self::Output {
        &self.blocks[idx.0 as usize]
    }
}

impl IndexMut<BlockIndex> for BlockArena {
    #[inline]
    fn index_mut(&mut self, idx: BlockIndex) -> &mut Self::Output {
        &mut self.blocks[idx.0 as usize]
    }
}
