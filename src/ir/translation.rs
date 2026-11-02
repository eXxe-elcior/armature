use crate::ir::block::{BlockArena, BlockIndex};
use crate::ir::operand::{OperandArena, OperandIndex, operand_add_assignment, operand_add_use};
use crate::ir::operation::{OperationArena, OperationIdx};
use crate::ir::types::{Instruction, Intrinsic, OperandType};

pub struct TranslationContext {
    pub blocks: BlockArena,
    pub ops: OperationArena,
    pub operands: OperandArena,
}

impl TranslationContext {
    pub fn new() -> Self {
        Self {
            blocks: BlockArena::new(),
            ops: OperationArena::new(),
            operands: OperandArena::new(),
        }
    }

    /// Clear all arenas for reuse across translations (retains capacity).
    /// Must clear every arena field or stale state leaks function-to-function.
    pub fn reset(&mut self) {
        self.blocks.reset();
        self.ops.reset();
        self.operands.reset();
    }

    /// Create a live operation: allocate it and wire def-use chains.
    ///
    /// This is the single choke point for operation creation, mirroring
    /// C# `Operation.Factory.Operation()`, whose constructor calls
    /// `AddAssignment`/`AddUse` for every destination/source.
    ///
    /// Does NOT append to a block  the emitter appends to the current
    /// block (C# `EmitterContext.Add` → `Operations.AddLast`), passes
    /// use terminator-aware insertion.
    pub fn make_op(
        &mut self,
        inst: Instruction,
        dsts: &[OperandIndex],
        srcs: &[OperandIndex],
    ) -> OperationIdx {
        let idx = self.ops.emit_raw(inst, dsts, srcs, 0);
        for &dst in dsts {
            operand_add_assignment(&mut self.operands, dst, idx);
        }
        for &src in srcs {
            operand_add_use(&mut self.operands, src, idx);
        }
        idx
    }

    pub fn make_intrinsic_op(
        &mut self,
        intrinsic: Intrinsic,
        dsts: &[OperandIndex],
        srcs: &[OperandIndex],
    ) -> OperationIdx {
        let idx = self
            .ops
            .emit_raw_intrinsic(Instruction::Extended, intrinsic, dsts, srcs, 0);
        for &dst in dsts {
            operand_add_assignment(&mut self.operands, dst, idx);
        }
        for &src in srcs {
            operand_add_use(&mut self.operands, src, idx);
        }
        idx
    }

    pub fn append_op(&mut self, block: BlockIndex, op_idx: OperationIdx) {
        self.blocks.append_op(block, op_idx);
    }

    #[inline]
    pub fn temp(&mut self, typ: OperandType) -> OperandIndex {
        self.operands.local(typ)
    }

    #[inline]
    pub fn const_zero(&mut self, typ: OperandType) -> OperandIndex {
        self.operands.const_zero(typ)
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}
