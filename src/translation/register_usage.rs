// Port of Translation/RegisterUsage.cs (A64 only  ExecutionMode dropped,
// Integer registers are always I64).
//
// Divergences:
//   - C# RegisterMask wraps Vector128<long> with an SSE2 fast path; the
//     Rust port is a plain { int_mask, vec_mask } pair  semantically
//     identical, the SIMD there was a C#-side micro-optimization.
//   - Predecessor lists are deferred in this IR (see ir/block.rs):
//     run_pass calls compute_predecessors itself, and again after
//     UpdateEntry mutates the edges.

use crate::ir::block::BlockIndex;
use crate::ir::operand::OperandIndex;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, OperandKind, OperandType, RegisterType};
use crate::registers::{CpuState, GpRegisterIndex, VecRegisterIndex};
use crate::translation::cfg::ControlFlowGraph;

const REGS_COUNT: i32 = 32;
const REGS_MASK: i32 = REGS_COUNT - 1;

// C# readonly struct RegisterMask (Vector128 storage flattened).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct RegisterMask {
    int_mask: i64,
    vec_mask: i64,
}

impl RegisterMask {
    fn new(int_mask: i64, vec_mask: i64) -> Self {
        Self { int_mask, vec_mask }
    }
}

impl std::ops::BitAnd for RegisterMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self::new(self.int_mask & rhs.int_mask, self.vec_mask & rhs.vec_mask)
    }
}

impl std::ops::BitOr for RegisterMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self::new(self.int_mask | rhs.int_mask, self.vec_mask | rhs.vec_mask)
    }
}

impl std::ops::BitOrAssign for RegisterMask {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl std::ops::BitAndAssign for RegisterMask {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl std::ops::Not for RegisterMask {
    type Output = Self;
    fn not(self) -> Self {
        Self::new(!self.int_mask, !self.vec_mask)
    }
}

pub fn run_pass(cfg: &mut ControlFlowGraph, ctx: &mut TranslationContext) {
    // Predecessors are deferred in this IR  materialize them first.
    ctx.blocks.compute_predecessors();

    if !ctx.blocks.get(cfg.entry).predecessors.is_empty() {
        // We expect the entry block to have no predecessors. This is
        // required because we have an implicit context load at the start
        // of the function, but if there is a jump to the start of the
        // function, the context load would trash the modified values.
        // Here we insert a new entry block that will jump to the existing
        // entry block.
        let new_entry = ctx.blocks.alloc();
        // C#: new BasicBlock(cfg.Blocks.Count).
        ctx.blocks.get_mut(new_entry).index = ctx.blocks.order().len() as u32;

        cfg.update_entry(&mut ctx.blocks, new_entry);

        // Edges changed  refresh the predecessor lists.
        ctx.blocks.compute_predecessors();
    }

    let block_count = ctx.blocks.order().len();

    // Compute local register inputs and outputs used inside blocks.
    let mut local_inputs = vec![RegisterMask::default(); block_count];
    let mut local_outputs = vec![RegisterMask::default(); block_count];

    for &block in ctx.blocks.order() {
        let bi = ctx.blocks.get(block).index as usize;

        for &node in &ctx.blocks.get(block).ops {
            for &source in ctx.ops.sources(node) {
                let src = ctx.operands.get(source);
                if src.kind == OperandKind::Register {
                    let mask = get_mask(src.reg_type(), src.reg_index());
                    local_inputs[bi] |= mask & !local_outputs[bi];
                }
            }

            if let Some(&dest) = ctx.ops.dests(node).first() {
                let d = ctx.operands.get(dest);
                if d.kind == OperandKind::Register {
                    local_outputs[bi] |= get_mask(d.reg_type(), d.reg_index());
                }
            }
        }
    }

    // Compute global register inputs and outputs used across blocks.
    let mut global_cmn_outputs = vec![RegisterMask::default(); block_count];
    let mut global_inputs = vec![RegisterMask::default(); block_count];
    let mut global_outputs = vec![RegisterMask::default(); block_count];

    let mut first_pass = true;

    loop {
        let mut modified = false;

        // Compute register outputs.
        for index in (0..cfg.post_order_blocks.len()).rev() {
            let block = cfg.post_order_blocks[index];
            let bi = ctx.blocks.get(block).index as usize;

            if !ctx.blocks.get(block).predecessors.is_empty() && !has_context_load(ctx, block) {
                let preds = ctx.blocks.get(block).predecessors.clone();

                let p0 = ctx.blocks.get(preds[0]).index as usize;
                let mut cmn_outputs = local_outputs[p0] | global_cmn_outputs[p0];
                let mut outputs = global_outputs[p0];

                for &pred in &preds[1..] {
                    let pi = ctx.blocks.get(pred).index as usize;

                    cmn_outputs &= local_outputs[pi] | global_cmn_outputs[pi];
                    outputs |= global_outputs[pi];
                }

                global_inputs[bi] |= outputs & !cmn_outputs;

                if !first_pass {
                    cmn_outputs &= global_cmn_outputs[bi];
                }

                modified |= exchange(&mut global_cmn_outputs, bi, cmn_outputs);
                outputs |= local_outputs[bi];
                let merged = global_outputs[bi] | outputs;
                modified |= exchange(&mut global_outputs, bi, merged);
            } else {
                modified |= exchange(&mut global_outputs, bi, local_outputs[bi]);
            }
        }

        // Compute register inputs.
        for index in 0..cfg.post_order_blocks.len() {
            let block = cfg.post_order_blocks[index];
            let bi = ctx.blocks.get(block).index as usize;

            let mut inputs = local_inputs[bi];

            let data = ctx.blocks.get(block);
            for i in 0..data.successor_count as usize {
                let succ = data.successors[i].expect("successor slot within count");
                inputs |= global_inputs[ctx.blocks.get(succ).index as usize];
            }

            inputs &= !global_cmn_outputs[bi];

            let merged = global_inputs[bi] | inputs;
            modified |= exchange(&mut global_inputs, bi, merged);
        }

        first_pass = false;

        if !modified {
            break;
        }
    }

    // Insert load and store context instructions where needed.
    for &block in ctx.blocks.order().to_vec().iter() {
        let bi = ctx.blocks.get(block).index as usize;

        let has_context_load = has_context_load(ctx, block);

        if has_context_load {
            ctx.blocks.get_mut(block).ops.remove(0);
        }

        let mut arg: Option<OperandIndex> = None;

        // The only block without any predecessor should be the entry
        // block. It always needs a context load as it is the first block
        // to run.
        if block == cfg.entry || has_context_load {
            let vec_mask = global_inputs[bi].vec_mask;
            let int_mask = global_inputs[bi].int_mask;

            if vec_mask != 0 || int_mask != 0 {
                let a = ctx.temp(OperandType::I64);
                arg = Some(a);

                let zero = ctx.operands.constant_i32(0);
                let load_arg = ctx.make_op(Instruction::LoadArgument, &[a], &[zero]);
                ctx.blocks.get_mut(block).ops.insert(0, load_arg);

                // C# AddAfter(loadArg, …) twice per register: the second
                // insert lands between loadArg and the first  final
                // order is loadArg, Add, Load. insert(1, …) in the same
                // sequence replicates that.
                load_locals(ctx, block, vec_mask, RegisterType::Vector, a);
                load_locals(ctx, block, int_mask, RegisterType::Integer, a);
            }
        }

        let has_context_store = has_context_store(ctx, block);

        if has_context_store {
            ctx.blocks.get_mut(block).ops.pop();
        }

        if ends_with_return(ctx, block) || has_context_store {
            let vec_mask = global_outputs[bi].vec_mask;
            let int_mask = global_outputs[bi].int_mask;

            if vec_mask != 0 || int_mask != 0 {
                let a = match arg {
                    Some(a) => a,
                    None => {
                        let a = ctx.temp(OperandType::I64);
                        let zero = ctx.operands.constant_i32(0);
                        let load_arg = ctx.make_op(Instruction::LoadArgument, &[a], &[zero]);
                        append(ctx, block, load_arg);
                        a
                    }
                };

                store_locals(ctx, block, int_mask, RegisterType::Integer, a);
                store_locals(ctx, block, vec_mask, RegisterType::Vector, a);
            }
        }
    }
}

fn has_context_load(ctx: &TranslationContext, block: BlockIndex) -> bool {
    starts_with(ctx, block, Instruction::LoadFromContext)
        && ctx
            .ops
            .sources(*ctx.blocks.get(block).ops.first().unwrap())
            .is_empty()
}

fn has_context_store(ctx: &TranslationContext, block: BlockIndex) -> bool {
    ends_with(ctx, block, Instruction::StoreToContext)
        && ctx
            .ops
            .sources(*ctx.blocks.get(block).ops.last().unwrap())
            .is_empty()
}

fn starts_with(ctx: &TranslationContext, block: BlockIndex, inst: Instruction) -> bool {
    match ctx.blocks.get(block).ops.first() {
        Some(&first) => ctx.ops.get(first).instruction == inst,
        None => false,
    }
}

fn ends_with(ctx: &TranslationContext, block: BlockIndex, inst: Instruction) -> bool {
    match ctx.blocks.get(block).ops.last() {
        Some(&last) => ctx.ops.get(last).instruction == inst,
        None => false,
    }
}

fn get_mask(reg_type: RegisterType, index: u32) -> RegisterMask {
    let mut int_mask: i64 = 0;
    let mut vec_mask: i64 = 0;

    // C# has a default: no-op arm; RegisterType is exhaustive in Rust.
    match reg_type {
        RegisterType::Flag => int_mask = (1i64 << REGS_COUNT) << index,
        RegisterType::Integer => int_mask = 1i64 << index,
        RegisterType::FpFlag => vec_mask = (1i64 << REGS_COUNT) << index,
        RegisterType::Vector => vec_mask = 1i64 << index,
    }

    RegisterMask::new(int_mask, vec_mask)
}

fn exchange(masks: &mut [RegisterMask], blk_index: usize, value: RegisterMask) -> bool {
    let changed = masks[blk_index] != value;
    masks[blk_index] = value;
    changed
}

// C# LoadLocals  walks bits high→low; each register becomes
// [Add addr, arg, offset] + [Load dest, addr] inserted right after the
// LoadArgument at ops[0].
fn load_locals(
    ctx: &mut TranslationContext,
    block: BlockIndex,
    mut inputs: i64,
    base_type: RegisterType,
    arg: OperandIndex,
) {
    while inputs != 0 {
        let bit = 63 - (inputs as u64).leading_zeros() as i32;

        let dest = get_reg_from_bit(ctx, bit, base_type);
        let dest_data = ctx.operands.get(dest);
        let offset_val = get_register_offset(dest_data.reg_type(), dest_data.reg_index());
        let offset = ctx.operands.constant_i64(offset_val as i64);
        let addr = ctx.temp(OperandType::I64);

        let load = ctx.make_op(Instruction::Load, &[dest], &[addr]);
        ctx.blocks.get_mut(block).ops.insert(1, load);
        let add = ctx.make_op(Instruction::Add, &[addr], &[arg, offset]);
        ctx.blocks.get_mut(block).ops.insert(1, add);

        inputs &= !(1i64 << bit);
    }
}

// C# StoreLocals  walks bits low→high; appends before the terminator.
fn store_locals(
    ctx: &mut TranslationContext,
    block: BlockIndex,
    mut outputs: i64,
    base_type: RegisterType,
    arg: OperandIndex,
) {
    while outputs != 0 {
        let bit = outputs.trailing_zeros() as i32;

        let source = get_reg_from_bit(ctx, bit, base_type);
        let source_data = ctx.operands.get(source);
        let offset_val = get_register_offset(source_data.reg_type(), source_data.reg_index());
        let offset = ctx.operands.constant_i64(offset_val as i64);
        let addr = ctx.temp(OperandType::I64);

        let add = ctx.make_op(Instruction::Add, &[addr], &[arg, offset]);
        append(ctx, block, add);
        let store = ctx.make_op(Instruction::Store, &[], &[addr, source]);
        append(ctx, block, store);

        outputs &= !(1i64 << bit);
    }
}

// C# BasicBlock.Append: insert before the terminal op (Return / Tailcall /
// BranchIf), or at the end if there is none.
fn append(
    ctx: &mut TranslationContext,
    block: BlockIndex,
    node: crate::ir::operation::OperationIdx,
) {
    let ops = &ctx.blocks.get(block).ops;

    let insert_at = match ops.last() {
        Some(&last) => match ctx.ops.get(last).instruction {
            Instruction::Return | Instruction::Tailcall | Instruction::BranchIf => ops.len() - 1,
            _ => ops.len(),
        },
        None => 0,
    };

    ctx.blocks.get_mut(block).ops.insert(insert_at, node);
}

fn get_reg_from_bit(
    ctx: &mut TranslationContext,
    bit: i32,
    base_type: RegisterType,
) -> OperandIndex {
    if bit < REGS_COUNT {
        let typ = get_operand_type(base_type);
        ctx.operands.register(bit as u32, base_type, typ)
    } else if base_type == RegisterType::Integer {
        ctx.operands.register(
            (bit & REGS_MASK) as u32,
            RegisterType::Flag,
            OperandType::I32,
        )
    } else if base_type == RegisterType::Vector {
        ctx.operands.register(
            (bit & REGS_MASK) as u32,
            RegisterType::FpFlag,
            OperandType::I32,
        )
    } else {
        panic!("get_reg_from_bit: bit {bit} out of range for {base_type:?}");
    }
}

// C# GetOperandType (A64: Integer is always I64).
fn get_operand_type(reg_type: RegisterType) -> OperandType {
    match reg_type {
        RegisterType::Flag => OperandType::I32,
        RegisterType::FpFlag => OperandType::I32,
        RegisterType::Integer => OperandType::I64,
        RegisterType::Vector => OperandType::V128,
    }
}

// C# NativeContext.GetRegisterOffset(register).
fn get_register_offset(reg_type: RegisterType, index: u32) -> usize {
    match reg_type {
        RegisterType::Integer => CpuState::gpr_offset(GpRegisterIndex::from(index as u8)),
        RegisterType::Vector => CpuState::vec_offset(VecRegisterIndex::from(index as u8)),
        RegisterType::Flag => CpuState::flag_offset(index as u8),
        RegisterType::FpFlag => CpuState::fp_flag_offset(index as u8),
    }
}

fn ends_with_return(ctx: &TranslationContext, block: BlockIndex) -> bool {
    match ctx.blocks.get(block).ops.last() {
        Some(&last) => ctx.ops.get(last).instruction == Instruction::Return,
        None => false,
    }
}
