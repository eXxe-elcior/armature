// Port of CodeGen/PreAllocatorCommon.cs.

use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::OperandKind;

/// C# Propagate(ref buffer, dest, value): replace every use of `dest`
/// with `value`, including memory-operand base/index. The C# buffer is a
/// reusable snapshot of the use list (mutating uses while iterating)
/// the Vec clone serves the same purpose.
pub fn propagate(ctx: &mut TranslationContext, dest: OperandIndex, value: OperandIndex) {
    let uses: Vec<OperationIdx> = ctx.operands.uses(dest).to_vec();

    for use_op in uses {
        // Skip sentinel entries (u32::MAX) that may remain in the use list
        // after pool-based def-use removal (the sentinel fills the vacated
        // slot at the end of the valid range; uses_count excludes it, but
        // the old value may still read through if uses_count was corrupted).
        if use_op.0 == u32::MAX {
            continue;
        }
        // Skip phi operations: their values are stored in phi_inputs, not
        // sources. Calling set_source on a phi writes to src_pool at
        // index 0 (src_start=0 for phi ops), corrupting the first non-phi
        // operation's source. The optimizer handles phi propagation
        // separately via set_phi_value.
        if ctx.ops.get(use_op).is_phi() {
            continue;
        }

        for src_index in 0..ctx.ops.sources(use_op).len() {
            let use_src = ctx.ops.sources(use_op)[src_index];

            if use_src == dest {
                ctx.ops
                    .set_source(use_op, src_index, value, &mut ctx.operands);
            } else if ctx.operands.get(use_src).kind == OperandKind::Memory {
                let (typ, mut base_addr, mut index, scale, disp) = {
                    let d = ctx.operands.get(use_src);
                    (d.typ, d.mem_base(), d.mem_index(), d.mem_scale, d.mem_disp)
                };
                let mut changed = false;

                if base_addr == Some(dest) {
                    base_addr = Some(value);
                    changed = true;
                }

                if index == Some(dest) {
                    index = Some(value);
                    changed = true;
                }

                if changed {
                    let new_mem = ctx.operands.memory(typ, base_addr, index, scale, disp);
                    ctx.ops
                        .set_source(use_op, src_index, new_mem, &mut ctx.operands);
                }
            }
        }
    }
}
