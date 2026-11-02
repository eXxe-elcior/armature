use crate::ir::block::BlockIndex;
use crate::ir::operand::{
    OperandArena, OperandIndex, operand_add_assignment, operand_add_use, operand_remove_assignment,
    operand_remove_use,
};
use crate::ir::types::{Instruction, Intrinsic};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationIdx(pub u32);

define_index!(OperationIdx);

#[derive(Debug, Clone)]
pub struct OperationData {
    pub instruction: Instruction,
    pub intrinsic: Intrinsic,
    pub dead: bool,
    pub src_start: u32,
    pub src_count: u16,
    pub dst_start: u32,
    pub dst_count: u16,
    pub phi_start: u32,
    pub phi_count: u16,
}

impl OperationData {
    pub fn is_phi(&self) -> bool {
        self.instruction == Instruction::Phi
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhiInput {
    pub block: BlockIndex,
    pub value: OperandIndex,
}

pub struct OperationArena {
    pub(crate) operations: Vec<OperationData>,
    pub(crate) src_pool: Vec<OperandIndex>,
    pub(crate) dst_pool: Vec<OperandIndex>,
    pub(crate) phi_inputs: Vec<PhiInput>,
}

impl OperationArena {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            src_pool: Vec::new(),
            dst_pool: Vec::new(),
            phi_inputs: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            operations: Vec::with_capacity(cap),
            src_pool: Vec::with_capacity(cap * 3),
            dst_pool: Vec::with_capacity(cap),
            phi_inputs: Vec::new(),
        }
    }

    /// Clear all state for arena reuse, retaining capacity.
    pub fn reset(&mut self) {
        self.operations.clear();
        self.src_pool.clear();
        self.dst_pool.clear();
        self.phi_inputs.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    #[inline]
    pub fn get(&self, idx: OperationIdx) -> &OperationData {
        &self.operations[idx.0 as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, idx: OperationIdx) -> &mut OperationData {
        &mut self.operations[idx.0 as usize]
    }

    // Pool accessors
    pub fn sources(&self, idx: OperationIdx) -> &[OperandIndex] {
        let op = &self.operations[idx.0 as usize];
        &self.src_pool[op.src_start as usize..][..op.src_count as usize]
    }

    pub fn dests(&self, idx: OperationIdx) -> &[OperandIndex] {
        let op = &self.operations[idx.0 as usize];
        &self.dst_pool[op.dst_start as usize..][..op.dst_count as usize]
    }

    pub fn phi_inputs(&self, idx: OperationIdx) -> &[PhiInput] {
        let op = &self.operations[idx.0 as usize];
        if op.phi_start == u32::MAX {
            return &[];
        }
        &self.phi_inputs[op.phi_start as usize..][..op.phi_count as usize]
    }

    pub fn phi_inputs_mut(&mut self, idx: OperationIdx) -> &mut [PhiInput] {
        let phi_start = self.operations[idx.0 as usize].phi_start;
        let phi_count = self.operations[idx.0 as usize].phi_count;
        if phi_start == u32::MAX {
            return &mut [];
        }
        &mut self.phi_inputs[phi_start as usize..][..phi_count as usize]
    }

    // Core allocation
    pub fn emit_raw(
        &mut self,
        instruction: Instruction,
        dsts: &[OperandIndex],
        srcs: &[OperandIndex],
        phi_count: usize,
    ) -> OperationIdx {
        let intrinsic = Intrinsic::default(); // non-intrinsic ops get X86None
        self.emit_raw_intrinsic(instruction, intrinsic, dsts, srcs, phi_count)
    }

    pub fn emit_raw_intrinsic(
        &mut self,
        instruction: Instruction,
        intrinsic: Intrinsic,
        dsts: &[OperandIndex],
        srcs: &[OperandIndex],
        phi_count: usize,
    ) -> OperationIdx {
        let src_start = self.src_pool.len() as u32;
        self.src_pool.extend_from_slice(srcs);

        let dst_start = self.dst_pool.len() as u32;
        self.dst_pool.extend_from_slice(dsts);

        let phi_start = if phi_count > 0 {
            let start = self.phi_inputs.len() as u32;
            self.phi_inputs.resize(
                self.phi_inputs.len() + phi_count,
                PhiInput {
                    block: BlockIndex(u32::MAX),
                    value: OperandIndex(u32::MAX),
                },
            );
            start
        } else {
            u32::MAX
        };

        let idx = OperationIdx(self.operations.len() as u32);
        self.operations.push(OperationData {
            instruction,
            intrinsic,
            dead: false,
            src_start,
            src_count: srcs.len() as u16,
            dst_start,
            dst_count: dsts.len() as u16,
            phi_start,
            phi_count: phi_count as u16,
        });
        idx
    }

    pub fn emit_phi_raw(&mut self, dest: OperandIndex, pred_count: u16) -> OperationIdx {
        let dst_start = self.dst_pool.len() as u32;
        self.dst_pool.push(dest);

        let phi_start = self.phi_inputs.len() as u32;
        self.phi_inputs.resize(
            self.phi_inputs.len() + pred_count as usize,
            PhiInput {
                block: BlockIndex(u32::MAX),
                value: OperandIndex(u32::MAX),
            },
        );

        let idx = OperationIdx(self.operations.len() as u32);
        self.operations.push(OperationData {
            instruction: Instruction::Phi,
            intrinsic: Intrinsic::default(),
            dead: false,
            src_start: 0,
            src_count: 0,
            dst_start,
            dst_count: 1,
            phi_start,
            phi_count: pred_count,
        });
        idx
    }

    // Source/dest mutation
    /// Raw slot write  register-allocation only. C# HybridAllocator/
    /// LinearScan write the operand spans directly (`ref Operand` foreach)
    /// without touching def-use; post-RA the chains are dead.
    pub fn set_source_raw(&mut self, op_idx: OperationIdx, i: usize, new_src: OperandIndex) {
        let start = self.operations[op_idx.0 as usize].src_start as usize;
        self.src_pool[start + i] = new_src;
    }

    /// Raw slot write  register-allocation only (see set_source_raw).
    pub fn set_destination_raw(&mut self, op_idx: OperationIdx, i: usize, new_dst: OperandIndex) {
        let start = self.operations[op_idx.0 as usize].dst_start as usize;
        self.dst_pool[start + i] = new_dst;
    }

    /// Replace the whole source list (C# SetSources). The span may change
    /// length, so a fresh span is allocated at the pool end (the old one
    /// is leaked in the pool  arena dies with the translation). Def-use
    /// is maintained: old sources lose the use, new ones gain it.
    pub fn set_sources(
        &mut self,
        op_idx: OperationIdx,
        new_srcs: &[OperandIndex],
        operands: &mut OperandArena,
    ) {
        let (src_start, src_count) = {
            let op = &self.operations[op_idx.0 as usize];
            (op.src_start as usize, op.src_count as usize)
        };

        for i in 0..src_count {
            let old = self.src_pool[src_start + i];
            if old.0 != u32::MAX {
                operand_remove_use(operands, old, op_idx);
            }
        }

        let new_start = self.src_pool.len() as u32;
        self.src_pool.extend_from_slice(new_srcs);

        {
            let op = &mut self.operations[op_idx.0 as usize];
            op.src_start = new_start;
            op.src_count = new_srcs.len() as u16;
        }

        for &src in new_srcs {
            if src.0 != u32::MAX {
                operand_add_use(operands, src, op_idx);
            }
        }
    }

    /// Replace the whole destination list (C# SetDestinations /
    /// Destination setter). Same span-reallocation scheme as set_sources.
    pub fn set_destinations(
        &mut self,
        op_idx: OperationIdx,
        new_dsts: &[OperandIndex],
        operands: &mut OperandArena,
    ) {
        let (dst_start, dst_count) = {
            let op = &self.operations[op_idx.0 as usize];
            (op.dst_start as usize, op.dst_count as usize)
        };

        for i in 0..dst_count {
            let old = self.dst_pool[dst_start + i];
            if old.0 != u32::MAX {
                operand_remove_assignment(operands, old, op_idx);
            }
        }

        let new_start = self.dst_pool.len() as u32;
        self.dst_pool.extend_from_slice(new_dsts);

        {
            let op = &mut self.operations[op_idx.0 as usize];
            op.dst_start = new_start;
            op.dst_count = new_dsts.len() as u16;
        }

        for &dst in new_dsts {
            if dst.0 != u32::MAX {
                operand_add_assignment(operands, dst, op_idx);
            }
        }
    }

    /// Set source at index. Removes old def-use if slot was occupied,
    /// adds def-use for new source. Matches C# SetSource.
    pub fn set_source(
        &mut self,
        op_idx: OperationIdx,
        i: usize,
        new_src: OperandIndex,
        operands: &mut OperandArena,
    ) {
        let (src_start, src_count) = {
            let op = &self.operations[op_idx.0 as usize];
            (op.src_start, op.src_count)
        };

        let slot = (src_start as usize) + i;

        if (i as u16) < src_count {
            let old = self.src_pool[slot];
            if old.0 != u32::MAX {
                operand_remove_use(operands, old, op_idx);
            }
        }

        self.src_pool[slot] = new_src;

        if new_src.0 != u32::MAX {
            operand_add_use(operands, new_src, op_idx);
        }
    }

    /// Set destination at index. Removes old assignment if slot was
    /// occupied, adds assignment for new destination. Matches C#
    /// SetDestination.
    pub fn set_destination(
        &mut self,
        op_idx: OperationIdx,
        i: usize,
        new_dst: OperandIndex,
        operands: &mut OperandArena,
    ) {
        let (dst_start, dst_count) = {
            let op = &self.operations[op_idx.0 as usize];
            (op.dst_start, op.dst_count)
        };

        let slot = (dst_start as usize) + i;

        if (i as u16) < dst_count {
            let old = self.dst_pool[slot];
            if old.0 != u32::MAX {
                operand_remove_assignment(operands, old, op_idx);
            }
        }

        self.dst_pool[slot] = new_dst;

        if new_dst.0 != u32::MAX {
            operand_add_assignment(operands, new_dst, op_idx);
        }
    }

    // Phi mutation
    pub fn set_phi_block(&mut self, op_idx: OperationIdx, i: usize, block: BlockIndex) {
        let phi_start = self.operations[op_idx.0 as usize].phi_start;
        self.phi_inputs[(phi_start as usize) + i].block = block;
    }

    pub fn set_phi_value(
        &mut self,
        op_idx: OperationIdx,
        i: usize,
        value: OperandIndex,
        operands: &mut OperandArena,
    ) {
        let phi_start = self.operations[op_idx.0 as usize].phi_start;
        let slot = &mut self.phi_inputs[(phi_start as usize) + i];
        let old = slot.value;
        if old.0 != u32::MAX {
            operand_remove_use(operands, old, op_idx);
        }
        slot.value = value;
        operand_add_use(operands, value, op_idx);
    }

    // Kill
    /// Mark operation as dead and remove all def-use references.
    /// Does NOT touch the src_pool/dst_pool/phi_inputs entries (they
    /// become unreachable garbage).
    pub fn kill(&mut self, op_idx: OperationIdx, operands: &mut OperandArena) {
        let (src_start, src_count, dst_start, dst_count, phi_start, phi_count) = {
            let op = &self.operations[op_idx.0 as usize];
            (
                op.src_start,
                op.src_count,
                op.dst_start,
                op.dst_count,
                op.phi_start,
                op.phi_count,
            )
        };

        for i in 0..src_count as usize {
            let src = self.src_pool[(src_start as usize) + i];
            if src.0 != u32::MAX {
                operand_remove_use(operands, src, op_idx);
            }
        }

        for i in 0..dst_count as usize {
            let dst = self.dst_pool[(dst_start as usize) + i];
            if dst.0 != u32::MAX {
                operand_remove_assignment(operands, dst, op_idx);
            }
        }

        if phi_start != u32::MAX {
            for i in 0..phi_count as usize {
                let phi = &self.phi_inputs[(phi_start as usize) + i];
                if phi.value.0 != u32::MAX {
                    operand_remove_use(operands, phi.value, op_idx);
                }
            }
        }

        self.operations[op_idx.0 as usize].dead = true;
    }

    // TurnIntoCopy
    /// Matches C# `Operation.TurnIntoCopy(Operand source)`.
    /// Removes ALL old source uses, replaces sources with `source` alone.
    /// Does NOT touch destinations or phi inputs (C# doesn't).
    /// Sets `Instruction = Copy`, `src_count = 1` (or 0 if source is poison).
    pub fn turn_into_copy(
        &mut self,
        op_idx: OperationIdx,
        source: OperandIndex,
        operands: &mut OperandArena,
    ) {
        let (old_src_start, old_src_count) = {
            let op = &self.operations[op_idx.0 as usize];
            (op.src_start, op.src_count)
        };

        // Remove ALL old source def-use (matches C# RemoveOldSources)
        for i in 0..old_src_count as usize {
            let old_src = self.src_pool[(old_src_start as usize) + i];
            if old_src.0 != u32::MAX {
                operand_remove_use(operands, old_src, op_idx);
            }
        }

        // Write new source  reuse slot 0 if it exists, else push
        let new_src_start = if old_src_count > 0 {
            self.src_pool[old_src_start as usize] = source;
            for i in 1..old_src_count as usize {
                self.src_pool[(old_src_start as usize) + i] = OperandIndex(u32::MAX);
            }
            old_src_start
        } else {
            self.src_pool.push(source);
            (self.src_pool.len() - 1) as u32
        };

        // Update operation: only instruction + source metadata (C# doesn't
        // touch DestinationsCount, PhiSourcesCount, etc.)
        {
            let op = &mut self.operations[op_idx.0 as usize];
            op.instruction = Instruction::Copy;
            op.intrinsic = Intrinsic::default();
            op.src_start = new_src_start;
            op.src_count = if source.0 != u32::MAX { 1 } else { 0 };
        }

        // Add def-use for new source (C# AddUse skips default)
        if source.0 != u32::MAX {
            operand_add_use(operands, source, op_idx);
        }
    }

    /// Rewrite an operation into a two-source `inst` (dest unchanged). Used by
    /// strength reduction (e.g. Multiply -> ShiftLeft). Maintains def-use.
    pub fn turn_into_binary_op(
        &mut self,
        op_idx: OperationIdx,
        inst: Instruction,
        x: OperandIndex,
        y: OperandIndex,
        operands: &mut OperandArena,
    ) {
        self.set_sources(op_idx, &[x, y], operands);
        let op = &mut self.operations[op_idx.0 as usize];
        op.instruction = inst;
        op.intrinsic = Intrinsic::default();
    }
}

impl Default for OperationArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<OperationIdx> for OperationArena {
    type Output = OperationData;
    #[inline]
    fn index(&self, idx: OperationIdx) -> &Self::Output {
        &self.operations[idx.0 as usize]
    }
}

impl IndexMut<OperationIdx> for OperationArena {
    #[inline]
    fn index_mut(&mut self, idx: OperationIdx) -> &mut Self::Output {
        &mut self.operations[idx.0 as usize]
    }
}
