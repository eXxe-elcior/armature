// Port of CodeGen/RegisterAllocators/HybridAllocator.cs  the LCQ
// register allocator. Two passes over post-order (reversed):
//
//   1. Collect per-block info (has-call, fixed registers) and per-local
//      info (use counts, first/last block → block-local detection).
//   2. Allocate: block-local locals get free registers; everything else
//      lives on the stack with Fill/Spill ops around each use, using
//      spill-temp registers reserved per block (MaxIROperands of them).
//
// Mappings:
//   - C# stores the local number in the operand value with a visited bit
//     at MSB (VisitedMask)  identical here via OperandData.value.
//   - C# `ref Operand` span writes bypass def-use; set_source_raw /
//     set_destination_raw mirror that (post-RA the chains are dead).
//   - C# IntrusiveList AddBefore/AddAfter/Remove → Vec cursor with
//     index insertion; "node = spillOp" → cursor jumps past the spill.

use crate::codegen::x86_64::allocation_result::{AllocationResult, RegisterMasks, StackAllocator};
use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, OperandKind, OperandType};
use crate::translation::cfg::ControlFlowGraph;

const MAX_IR_OPERANDS: i32 = 4;
// The "visited" state is stored in the MSB of the local's value.
const VISITED_MASK: u64 = 1u64 << 63;

#[derive(Clone, Copy, Default)]
struct BlockInfo {
    has_call: bool,
    int_fixed_registers: u32,
    vec_fixed_registers: u32,
}

#[derive(Clone, Copy)]
struct LocalInfo {
    uses: i32,
    uses_allocated: i32,
    sequence: i32,
    temp: Option<OperandIndex>,
    register: Option<OperandIndex>,
    spill_offset: Option<OperandIndex>,
    typ: OperandType,
    first: i32,
    last: i32,
}

impl LocalInfo {
    fn new(typ: OperandType, uses: i32, blk_index: i32) -> Self {
        let mut info = Self {
            uses,
            uses_allocated: 0,
            sequence: 0,
            temp: None,
            register: None,
            spill_offset: None,
            typ,
            first: -1,
            last: -1,
        };
        info.set_block_index(blk_index);
        info
    }

    fn set_block_index(&mut self, blk_index: i32) {
        if self.first == -1 || blk_index < self.first {
            self.first = blk_index;
        }
        if self.last == -1 || blk_index > self.last {
            self.last = blk_index;
        }
    }

    fn is_block_local(&self) -> bool {
        self.first == self.last
    }
}

fn is_visited(ctx: &TranslationContext, local: OperandIndex) -> bool {
    let d = ctx.operands.get(local);
    debug_assert_eq!(d.kind, OperandKind::LocalVariable);
    d.value & VISITED_MASK != 0
}

fn set_visited(ctx: &mut TranslationContext, local: OperandIndex) {
    let d = ctx.operands.get_mut(local);
    debug_assert_eq!(d.kind, OperandKind::LocalVariable);
    d.value |= VISITED_MASK;
}

fn local_info_index(ctx: &TranslationContext, local: OperandIndex) -> usize {
    let d = ctx.operands.get(local);
    debug_assert_eq!(d.kind, OperandKind::LocalVariable);
    debug_assert!(
        d.value & VISITED_MASK != 0,
        "Local variable not visited. Used before defined?"
    );
    (d.value as u32 - 1) as usize
}

// C# UsesCount(local) = AssignmentsCount + UsesCount.
fn uses_count(ctx: &TranslationContext, local: OperandIndex) -> i32 {
    ctx.operands.assigns_count(local) as i32 + ctx.operands.uses_count(local) as i32
}

pub fn run_pass(
    cfg: &ControlFlowGraph,
    ctx: &mut TranslationContext,
    stack_alloc: &mut StackAllocator,
    reg_masks: &RegisterMasks,
) -> AllocationResult {
    let mut int_used_registers: u32 = 0;
    let mut vec_used_registers: u32 = 0;

    let int_free_registers = reg_masks.int_available;
    let vec_free_registers = reg_masks.vec_available;

    let block_count = ctx.blocks.order().len();
    let mut block_info = vec![BlockInfo::default(); block_count];
    let mut local_info: Vec<LocalInfo> = Vec::with_capacity(block_count * 3);

    // Pass 1: collect block + local info
    for index in (0..cfg.post_order_blocks.len()).rev() {
        let block = cfg.post_order_blocks[index];
        let block_index = ctx.blocks.get(block).index as i32;

        let mut int_fixed_registers: u32 = 0;
        let mut vec_fixed_registers: u32 = 0;
        let mut has_call = false;

        let ops: Vec<OperationIdx> = ctx.blocks.get(block).ops.clone();
        for node in ops {
            if ctx.ops.get(node).instruction == Instruction::Call {
                has_call = true;
            }

            for si in 0..ctx.ops.sources(node).len() {
                let source = ctx.ops.sources(node)[si];
                let kind = ctx.operands.get(source).kind;

                if kind == OperandKind::LocalVariable {
                    let li = local_info_index(ctx, source);
                    local_info[li].set_block_index(block_index);
                } else if kind == OperandKind::Memory {
                    let (base, idx) = {
                        let d = ctx.operands.get(source);
                        (d.mem_base(), d.mem_index())
                    };
                    if let Some(base) = base {
                        let li = local_info_index(ctx, base);
                        local_info[li].set_block_index(block_index);
                    }
                    if let Some(idx) = idx {
                        let li = local_info_index(ctx, idx);
                        local_info[li].set_block_index(block_index);
                    }
                }
            }

            for di in 0..ctx.ops.dests(node).len() {
                let dest = ctx.ops.dests(node)[di];
                let d = ctx.operands.get(dest);

                if d.kind == OperandKind::LocalVariable {
                    if is_visited(ctx, dest) {
                        let li = local_info_index(ctx, dest);
                        local_info[li].set_block_index(block_index);
                    } else {
                        let number = local_info.len() as u32 + 1;
                        let typ = d.typ;
                        ctx.operands.get_mut(dest).number_local(number);
                        set_visited(ctx, dest);
                        let uses = uses_count(ctx, dest);
                        local_info.push(LocalInfo::new(typ, uses, block_index));
                    }
                } else if d.kind == OperandKind::Register {
                    if d.typ.is_integer() {
                        int_fixed_registers |= 1 << d.reg_index();
                    } else {
                        vec_fixed_registers |= 1 << d.reg_index();
                    }
                }
            }
        }

        block_info[block_index as usize] = BlockInfo {
            has_call,
            int_fixed_registers,
            vec_fixed_registers,
        };
    }

    // Pass 2: allocate
    let mut sequence: i32 = 0;

    for index in (0..cfg.post_order_blocks.len()).rev() {
        let block = cfg.post_order_blocks[index];
        let block_index = ctx.blocks.get(block).index as usize;
        let blk_info = block_info[block_index];

        let mut int_local_free_registers = int_free_registers & !blk_info.int_fixed_registers;
        let mut vec_local_free_registers = vec_free_registers & !blk_info.vec_fixed_registers;

        let int_caller_saved_registers = if blk_info.has_call {
            reg_masks.int_caller_saved
        } else {
            0
        };
        let vec_caller_saved_registers = if blk_info.has_call {
            reg_masks.vec_caller_saved
        } else {
            0
        };

        let int_spill_temp_registers = select_spill_temps(
            int_caller_saved_registers & !blk_info.int_fixed_registers,
            int_local_free_registers,
        );
        let vec_spill_temp_registers = select_spill_temps(
            vec_caller_saved_registers & !blk_info.vec_fixed_registers,
            vec_local_free_registers,
        );

        int_local_free_registers &= !(int_spill_temp_registers | int_caller_saved_registers);
        vec_local_free_registers &= !(vec_spill_temp_registers | vec_caller_saved_registers);

        let mut pos: usize = 0;
        while pos < ctx.blocks.get(block).ops.len() {
            let mut node = ctx.blocks.get(block).ops[pos];

            let mut int_local_use: u32 = 0;
            let mut vec_local_use: u32 = 0;

            let mut folded = false;

            // If operation is a copy of a local and that local is living
            // on the stack, we turn the copy into a fill, instead of
            // inserting a fill before it.
            if ctx.ops.get(node).instruction == Instruction::Copy {
                let source = ctx.ops.sources(node)[0];

                if ctx.operands.get(source).kind == OperandKind::LocalVariable {
                    let li = local_info_index(ctx, source);

                    if local_info[li].register.is_none() {
                        let dest = ctx.ops.dests(node)[0];
                        let spill_offset = local_info[li]
                            .spill_offset
                            .expect("spilled local has offset");
                        let fill_op = ctx.make_op(Instruction::Fill, &[dest], &[spill_offset]);

                        // C#: AddBefore(node, fill) + Remove(node) ≡ replace.
                        ctx.blocks.get_mut(block).ops[pos] = fill_op;

                        node = fill_op;
                        folded = true;
                    }
                }
            }

            if !folded {
                let mut pending_fills: Vec<OperationIdx> = Vec::new();

                for si in 0..ctx.ops.sources(node).len() {
                    let source = ctx.ops.sources(node)[si];
                    let kind = ctx.operands.get(source).kind;

                    if kind == OperandKind::LocalVariable {
                        let new = allocate_register(
                            ctx,
                            &mut local_info,
                            source,
                            sequence,
                            int_spill_temp_registers,
                            vec_spill_temp_registers,
                            &mut int_local_free_registers,
                            &mut vec_local_free_registers,
                            &mut int_local_use,
                            &mut vec_local_use,
                            &mut pending_fills,
                        );
                        ctx.ops.set_source_raw(node, si, new);
                    } else if kind == OperandKind::Memory {
                        let (base, idx) = {
                            let d = ctx.operands.get(source);
                            (d.mem_base(), d.mem_index())
                        };
                        if let Some(base) = base {
                            let new = allocate_register(
                                ctx,
                                &mut local_info,
                                base,
                                sequence,
                                int_spill_temp_registers,
                                vec_spill_temp_registers,
                                &mut int_local_free_registers,
                                &mut vec_local_free_registers,
                                &mut int_local_use,
                                &mut vec_local_use,
                                &mut pending_fills,
                            );
                            ctx.operands.get_mut(source).mem_base = new;
                        }
                        if let Some(idx) = idx {
                            let new = allocate_register(
                                ctx,
                                &mut local_info,
                                idx,
                                sequence,
                                int_spill_temp_registers,
                                vec_spill_temp_registers,
                                &mut int_local_free_registers,
                                &mut vec_local_free_registers,
                                &mut int_local_use,
                                &mut vec_local_use,
                                &mut pending_fills,
                            );
                            ctx.operands.get_mut(source).mem_index = new;
                        }
                    }
                }

                // C# inserts each fill immediately before `node`, in call
                // order  splice them before pos, advancing pos past them.
                for fill in pending_fills {
                    ctx.blocks.get_mut(block).ops.insert(pos, fill);
                    pos += 1;
                }
            }

            let mut int_local_asg: u32 = 0;
            let mut vec_local_asg: u32 = 0;

            let mut insert_after_pos = pos;

            for di in 0..ctx.ops.dests(node).len() {
                let dest = ctx.ops.dests(node)[di];

                if ctx.operands.get(dest).kind != OperandKind::LocalVariable {
                    continue;
                }

                let li = local_info_index(ctx, dest);
                let dest_typ = ctx.operands.get(dest).typ;

                if local_info[li].uses_allocated == 0 {
                    let mask = if dest_typ.is_integer() {
                        int_local_free_registers
                    } else {
                        vec_local_free_registers
                    };

                    if local_info[li].is_block_local() && mask != 0 {
                        let selected_reg = mask.trailing_zeros();

                        let info_typ = local_info[li].typ;
                        local_info[li].register = Some(ctx.operands.register(
                            selected_reg,
                            info_typ.to_register_type(),
                            info_typ,
                        ));

                        if dest_typ.is_integer() {
                            int_local_free_registers &= !(1 << selected_reg);
                            int_used_registers |= 1 << selected_reg;
                        } else {
                            vec_local_free_registers &= !(1 << selected_reg);
                            vec_used_registers |= 1 << selected_reg;
                        }
                    } else {
                        local_info[li].register = None;
                        let offset = stack_alloc.allocate_bytes(dest_typ.size_bytes() as i32);
                        local_info[li].spill_offset = Some(ctx.operands.constant_i32(offset));
                    }
                }

                local_info[li].uses_allocated += 1;
                debug_assert!(local_info[li].uses_allocated <= local_info[li].uses);

                if let Some(reg) = local_info[li].register {
                    ctx.ops.set_destination_raw(node, di, reg);
                } else {
                    let temp = match local_info[li].temp {
                        Some(t) if local_info[li].sequence == sequence => t,
                        _ => {
                            let t = if dest_typ.is_integer() {
                                get_spill_temp(
                                    ctx,
                                    dest_typ,
                                    int_spill_temp_registers,
                                    &mut int_local_asg,
                                )
                            } else {
                                get_spill_temp(
                                    ctx,
                                    dest_typ,
                                    vec_spill_temp_registers,
                                    &mut vec_local_asg,
                                )
                            };
                            local_info[li].sequence = sequence;
                            local_info[li].temp = Some(t);
                            t
                        }
                    };

                    ctx.ops.set_destination_raw(node, di, temp);

                    let spill_offset = local_info[li]
                        .spill_offset
                        .expect("spilled local has offset");
                    let spill_op = ctx.make_op(Instruction::Spill, &[], &[spill_offset, temp]);

                    // C#: AddAfter(node, spill); node = spillOp.
                    insert_after_pos += 1;
                    ctx.blocks
                        .get_mut(block)
                        .ops
                        .insert(insert_after_pos, spill_op);
                }
            }

            // C# continues iteration from the last inserted spill.
            pos = insert_after_pos;

            sequence += 1;

            int_used_registers |= int_local_asg | int_local_use;
            vec_used_registers |= vec_local_asg | vec_local_use;

            pos += 1;
        }
    }

    AllocationResult::new(
        int_used_registers,
        vec_used_registers,
        stack_alloc.total_size(),
    )
}

// C# AllocateRegister local function  flattened with explicit state.
#[allow(clippy::too_many_arguments)]
fn allocate_register(
    ctx: &mut TranslationContext,
    local_info: &mut [LocalInfo],
    local: OperandIndex,
    sequence: i32,
    int_spill_temp_registers: u32,
    vec_spill_temp_registers: u32,
    int_local_free_registers: &mut u32,
    vec_local_free_registers: &mut u32,
    int_local_use: &mut u32,
    vec_local_use: &mut u32,
    pending_fills: &mut Vec<OperationIdx>,
) -> OperandIndex {
    let li = local_info_index(ctx, local);
    let local_typ = ctx.operands.get(local).typ;

    local_info[li].uses_allocated += 1;
    debug_assert!(local_info[li].uses_allocated <= local_info[li].uses);

    if let Some(reg) = local_info[li].register {
        if local_info[li].uses_allocated == local_info[li].uses {
            let reg_index = ctx.operands.get(reg).reg_index();
            if local_typ.is_integer() {
                *int_local_free_registers |= 1 << reg_index;
            } else {
                *vec_local_free_registers |= 1 << reg_index;
            }
        }

        reg
    } else {
        let temp = match local_info[li].temp {
            Some(t) if local_info[li].sequence == sequence => t,
            _ => {
                let t = if local_typ.is_integer() {
                    get_spill_temp(ctx, local_typ, int_spill_temp_registers, int_local_use)
                } else {
                    get_spill_temp(ctx, local_typ, vec_spill_temp_registers, vec_local_use)
                };
                local_info[li].sequence = sequence;
                local_info[li].temp = Some(t);
                t
            }
        };

        let spill_offset = local_info[li]
            .spill_offset
            .expect("spilled local has offset");
        let fill_op = ctx.make_op(Instruction::Fill, &[temp], &[spill_offset]);

        pending_fills.push(fill_op);

        temp
    }
}

// C# SelectSpillTemps: pick MaxIROperands registers, preferring mask0
// (caller-saved in calling blocks  they'd be clobbered anyway).
fn select_spill_temps(mut mask0: u32, mut mask1: u32) -> u32 {
    let mut selection: u32 = 0;
    let mut count = 0;

    while count < MAX_IR_OPERANDS && mask0 != 0 {
        let mask = mask0 & mask0.wrapping_neg();
        selection |= mask;
        mask0 &= !mask;
        count += 1;
    }

    while count < MAX_IR_OPERANDS && mask1 != 0 {
        let mask = mask1 & mask1.wrapping_neg();
        selection |= mask;
        mask1 &= !mask;
        count += 1;
    }

    debug_assert!(
        count == MAX_IR_OPERANDS,
        "Not enough registers for spill temps."
    );

    selection
}

fn get_spill_temp(
    ctx: &mut TranslationContext,
    typ: OperandType,
    free_mask: u32,
    use_mask: &mut u32,
) -> OperandIndex {
    let selected_reg = (free_mask & !*use_mask).trailing_zeros();

    *use_mask |= 1 << selected_reg;

    ctx.operands
        .register(selected_reg, typ.to_register_type(), typ)
}
