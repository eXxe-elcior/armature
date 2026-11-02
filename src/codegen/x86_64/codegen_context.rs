// Port of CodeGen/X86/CodeGenContext.cs.
// Tracks blocks, labels, stack frames, and assembler state.

use crate::util::hash::FxHashMap as HashMap;

use crate::codegen::linking::RelocationInfo;
use crate::codegen::x86_64::allocation_result::AllocationResult;
use crate::codegen::x86_64::assembler::Assembler;
use crate::codegen::x86_64::condition::X86Condition;
use crate::codegen::x86_64::register::X86Register;
use crate::ir::block::BlockIndex;
use crate::ir::operand::{OperandArena, OperandIndex};

pub struct CodeGenContext {
    pub asm: Assembler,
    pub alloc_result: AllocationResult,
    pub call_args_region_size: i32,
    pub xmm_save_region_size: i32,
    pub curr_block: Option<BlockIndex>,
    /// Per-block assembly label operands.
    block_labels: HashMap<BlockIndex, OperandIndex>,
    /// Max call args seen during pre-allocator pass.
    #[allow(dead_code)]
    max_call_args: i32,
}

impl CodeGenContext {
    pub fn new(
        alloc_result: AllocationResult,
        max_call_args: i32,
        blocks_count: usize,
        relocatable: bool,
    ) -> Self {
        let (call_args_region_size, xmm_save_region_size) =
            Self::get_call_args_region_size(&alloc_result, max_call_args);

        Self {
            asm: Assembler::new(relocatable),
            alloc_result,
            call_args_region_size,
            xmm_save_region_size,
            curr_block: None,
            block_labels: HashMap::with_capacity_and_hasher(blocks_count, Default::default()),
            max_call_args,
        }
    }

    /// C# CodeGenContext.GetCallArgsRegionSize  computes stack frame layout.
    fn get_call_args_region_size(
        alloc_result: &AllocationResult,
        max_call_args: i32,
    ) -> (i32, i32) {
        use crate::codegen::x86_64::calling_convention::*;

        let int_mask = get_int_callee_saved_registers() & alloc_result.int_used_registers as i32;
        let vec_mask = get_vec_callee_saved_registers() & alloc_result.vec_used_registers as i32;

        let xmm_save = (vec_mask.count_ones() as i32) * 16;
        let callee_save = (int_mask.count_ones() as i32) * 8 + xmm_save + 8;

        let args_count = if max_call_args < 0 {
            0
        } else if max_call_args < 4 {
            4
        } else {
            max_call_args
        };

        let frame_size = callee_save + alloc_result.spill_region_size;
        let call_args_and_frame = frame_size + args_count * 16;
        let aligned = (call_args_and_frame + 0xf) & !0xf;

        (aligned - frame_size, xmm_save)
    }

    /// C# EnterBlock: mark label for this block, set CurrBlock.
    pub fn enter_block(&mut self, block: BlockIndex, operands: &mut OperandArena) {
        let label = self.get_label(block, operands);
        self.asm.mark_label(label);
        self.curr_block = Some(block);
    }

    /// C# JumpTo: unconditional jump to target block.
    pub fn jump_to(&mut self, target: BlockIndex, operands: &mut OperandArena) {
        let label = self.get_label(target, operands);
        self.asm.jmp(operands, label);
    }

    /// C# JumpTo(condition): conditional jump to target block.
    pub fn jump_to_cond(
        &mut self,
        condition: X86Condition,
        target: BlockIndex,
        operands: &mut OperandArena,
    ) {
        let label = self.get_label(target, operands);
        self.asm.jcc(condition, operands, label);
    }

    pub fn get_label(&mut self, block: BlockIndex, operands: &mut OperandArena) -> OperandIndex {
        *self
            .block_labels
            .entry(block)
            .or_insert_with(|| operands.label())
    }

    /// C# Assembler.GetCode() → (bytes, relocInfo).
    /// Resolve assembler labels and produce the final relocatable bytes.
    pub fn get_code(&mut self, operands: &OperandArena) -> (Vec<u8>, RelocationInfo) {
        self.asm.get_code(operands)
    }

    pub fn stream_offset(&self) -> i64 {
        self.asm.position() as i64
    }
}

impl X86Register {
    pub fn from_index(idx: u32) -> Self {
        match idx {
            0 => X86Register::RAX,
            1 => X86Register::RCX,
            2 => X86Register::RDX,
            3 => X86Register::RBX,
            4 => X86Register::RSP,
            5 => X86Register::RBP,
            6 => X86Register::RSI,
            7 => X86Register::RDI,
            8 => X86Register::R8,
            9 => X86Register::R9,
            10 => X86Register::R10,
            11 => X86Register::R11,
            12 => X86Register::R12,
            13 => X86Register::R13,
            14 => X86Register::R14,
            15 => X86Register::R15,
            _ => panic!("invalid X86Register index: {idx}"),
        }
    }
}
