// Port of Translation/SsaConstruction.cs + SsaDeconstruction.cs.
//
// Precondition: find_dominators() + find_dominance_frontiers() have been
// called, so each block has `idom` and `dom_frontiers` populated.

use crate::util::hash::FxHashMap as HashMap;

use crate::ir::block::{BlockArena, BlockIndex};
use crate::ir::operand::{OperandArena, OperandIndex, operand_add_assignment, operand_add_use};
use crate::ir::operation::{OperationArena, OperationIdx};
use crate::ir::types::{Instruction, OperandKind, OperandType, RegisterType};
use crate::translation::cfg::ControlFlowGraph;

// RegisterConsts
const INT_REGS_COUNT: u32 = 32;
const VEC_REGS_COUNT: u32 = 32;
const FLAGS_COUNT: u32 = 32;
const FP_FLAGS_COUNT: u32 = 32;
const TOTAL_COUNT: u32 = INT_REGS_COUNT + VEC_REGS_COUNT + FLAGS_COUNT + FP_FLAGS_COUNT; // 128

// DefMap
//
// Register keys live in a flat array; only the rare local-variable
// overflow goes to a HashMap.

struct DefMap {
    /// Direct-indexed storage for keys in 0..TOTAL_COUNT.
    reg_defs: [Option<OperandIndex>; TOTAL_COUNT as usize],
    /// Overflow for local-variable keys (>= TOTAL_COUNT).
    local_defs: HashMap<u32, OperandIndex>,
    /// Growable bit array for phi presence (same semantics as before).
    phi_mask: Vec<u64>,
}

impl DefMap {
    fn new() -> Self {
        Self {
            reg_defs: std::array::from_fn(|_| None),
            local_defs: HashMap::default(),
            phi_mask: vec![0u64; (TOTAL_COUNT as usize).div_ceil(64)],
        }
    }

    fn try_add_operand(&mut self, key: u32, operand: OperandIndex) -> bool {
        if key < TOTAL_COUNT {
            let slot = &mut self.reg_defs[key as usize];
            if slot.is_some() {
                return false;
            }
            *slot = Some(operand);
            true
        } else {
            use std::collections::hash_map::Entry;
            match self.local_defs.entry(key) {
                Entry::Occupied(_) => false,
                Entry::Vacant(v) => {
                    v.insert(operand);
                    true
                }
            }
        }
    }

    fn try_get_operand(&self, key: u32) -> Option<OperandIndex> {
        if key < TOTAL_COUNT {
            self.reg_defs[key as usize]
        } else {
            self.local_defs.get(&key).copied()
        }
    }

    fn add_phi(&mut self, key: u32) -> bool {
        let word = (key / 64) as usize;
        if word >= self.phi_mask.len() {
            self.phi_mask.resize((word + 1).next_power_of_two(), 0);
        }
        let bit = key % 64;
        let mask = 1u64 << bit;
        let was_set = (self.phi_mask[word] & mask) != 0;
        self.phi_mask[word] |= mask;
        !was_set
    }

    fn has_phi(&self, key: u32) -> bool {
        let word = (key / 64) as usize;
        word < self.phi_mask.len() && (self.phi_mask[word] & (1u64 << (key % 64))) != 0
    }
}

// Key mapping
fn try_get_id(operands: &OperandArena, operand: OperandIndex) -> Option<u32> {
    let data = operands.get(operand);
    if data.kind == OperandKind::Register {
        let rt = data.reg_type();
        let index = data.reg_index();
        Some(match rt {
            RegisterType::Integer => index,
            RegisterType::Vector => INT_REGS_COUNT + index,
            RegisterType::Flag => INT_REGS_COUNT + VEC_REGS_COUNT + index,
            RegisterType::FpFlag => INT_REGS_COUNT + VEC_REGS_COUNT + FLAGS_COUNT + index,
        })
    } else if data.kind == OperandKind::LocalVariable && data.local_number() > 0 {
        Some(TOTAL_COUNT + data.local_number() - 1)
    } else {
        None
    }
}

// Append helper (insert before terminator)
fn append_op(blocks: &mut BlockArena, ops: &OperationArena, block: BlockIndex, node: OperationIdx) {
    let insert_at = {
        let op_list = &blocks.get(block).ops;
        match op_list.last() {
            Some(&last) => match ops.get(last).instruction {
                Instruction::Return | Instruction::Tailcall | Instruction::BranchIf => {
                    op_list.len() - 1
                }
                _ => op_list.len(),
            },
            None => 0,
        }
    };
    blocks.get_mut(block).ops.insert(insert_at, node);
}

// Add-phi helper
//
// O(1) insertion via `phi_counts` cursor.

fn add_phi_to_block(
    blocks: &mut BlockArena,
    block: BlockIndex,
    phi: OperationIdx,
    phi_counts: &mut [usize],
) {
    let idx = blocks.get(block).index as usize;
    let pos = phi_counts[idx];
    blocks.get_mut(block).ops.insert(pos, phi);
    phi_counts[idx] += 1;
}

// FindDef / FindDefOnPred / InsertPhi
fn find_def(
    global_defs: &mut [DefMap],
    blocks: &mut BlockArena,
    ops: &mut OperationArena,
    operands: &mut OperandArena,
    phi_counts: &mut Vec<usize>,
    current: BlockIndex,
    key: u32,
    typ: OperandType,
) -> OperandIndex {
    if global_defs[blocks.get(current).index as usize].has_phi(key) {
        return insert_phi(
            global_defs,
            blocks,
            ops,
            operands,
            phi_counts,
            current,
            key,
            typ,
        );
    }

    let idom = blocks.get(current).idom;
    if let Some(idom) = idom {
        if current != idom {
            return find_def_on_pred(
                global_defs,
                blocks,
                ops,
                operands,
                phi_counts,
                idom,
                key,
                typ,
            );
        }
    }

    operands.undef()
}

fn find_def_on_pred(
    global_defs: &mut [DefMap],
    blocks: &mut BlockArena,
    ops: &mut OperationArena,
    operands: &mut OperandArena,
    phi_counts: &mut Vec<usize>,
    mut current: BlockIndex,
    key: u32,
    typ: OperandType,
) -> OperandIndex {
    loop {
        let idx = blocks.get(current).index as usize;
        let has_phi = global_defs[idx].has_phi(key);
        let def = global_defs[idx].try_get_operand(key);

        if let Some(def) = def {
            return def;
        }

        if has_phi {
            return insert_phi(
                global_defs,
                blocks,
                ops,
                operands,
                phi_counts,
                current,
                key,
                typ,
            );
        }

        let idom = blocks.get(current).idom;
        match idom {
            Some(next) if current != next => current = next,
            _ => break,
        }
    }

    operands.undef()
}

fn insert_phi(
    global_defs: &mut [DefMap],
    blocks: &mut BlockArena,
    ops: &mut OperationArena,
    operands: &mut OperandArena,
    phi_counts: &mut Vec<usize>,
    block: BlockIndex,
    key: u32,
    typ: OperandType,
) -> OperandIndex {
    let local = operands.local(typ);
    let pred_count = blocks.get(block).predecessors.len() as u16;

    let phi_idx = ops.emit_phi_raw(local, pred_count);
    operand_add_assignment(operands, local, phi_idx);

    // Opt #6: O(1) insertion.
    add_phi_to_block(blocks, block, phi_idx, phi_counts);

    global_defs[blocks.get(block).index as usize].try_add_operand(key, local);

    let preds = blocks.get(block).predecessors.clone();
    for (index, &pred) in preds.iter().enumerate() {
        ops.set_phi_block(phi_idx, index, pred);
        let source = find_def_on_pred(
            global_defs,
            blocks,
            ops,
            operands,
            phi_counts,
            pred,
            key,
            typ,
        );
        ops.set_phi_value(phi_idx, index, source, operands);
    }

    local
}

// Public API
/// C# Ssa.Construct(cfg).
pub fn construct(cfg: &ControlFlowGraph, ctx: &mut crate::ir::translation::TranslationContext) {
    #[derive(Copy, Clone)]
    struct PendingSource {
        node: OperationIdx,
        source_index: usize,
        key: u32,
        typ: OperandType,
    }

    #[inline(always)]
    fn bit_test(bits: &[u64], key: usize) -> bool {
        let word = key >> 6;
        let bit = key & 63;
        (bits[word] & (1u64 << bit)) != 0
    }

    #[inline(always)]
    fn bit_set(bits: &mut [u64], key: usize) {
        let word = key >> 6;
        let bit = key & 63;
        bits[word] |= 1u64 << bit;
    }

    let blocks = &mut ctx.blocks;
    let ops = &mut ctx.ops;
    let operands = &mut ctx.operands;

    let block_count = blocks.order().len();
    let order: Vec<BlockIndex> = blocks.order().to_vec();

    let mut global_defs: Vec<DefMap> = (0..block_count).map(|_| DefMap::new()).collect();
    let mut phi_counts: Vec<usize> = vec![0; block_count];

    let local_defs_len = cfg.locals_count as usize + TOTAL_COUNT as usize;
    let mut local_defs: Vec<Option<OperandIndex>> = vec![None; local_defs_len];

    let word_count = (local_defs_len + 63) / 64;

    // Scratch buffers shared across every block.
    let mut dirty = Vec::<usize>::new();
    let mut worklist = Vec::<BlockIndex>::new();
    let mut pending_sources = Vec::<PendingSource>::new();

    // Build successors once from predecessor lists for liveness.
    let mut successors = vec![Vec::<BlockIndex>::new(); block_count];
    for &block in &order {
        for &pred in &blocks.get(block).predecessors {
            let pred_idx = blocks.get(pred).index as usize;
            successors[pred_idx].push(block);
        }
    }

    // First pass: rename registers to locals, propagate phi markers
    for &block in &order {
        dirty.clear();

        let block_index = blocks.get(block).index as usize;
        let op_count = blocks.get(block).ops.len();

        for op_idx in 0..op_count {
            let node = blocks.get(block).ops[op_idx];

            for si in 0..ops.sources(node).len() {
                let src = ops.sources(node)[si];
                if let Some(key) = try_get_id(operands, src) {
                    if let Some(local) = local_defs[key as usize] {
                        ops.set_source(node, si, local, operands);
                    }
                }
            }

            if let Some(&dest) = ops.dests(node).first() {
                if let Some(key) = try_get_id(operands, dest) {
                    let typ = operands.get(dest).typ;
                    let local = operands.local(typ);
                    let idx = key as usize;

                    if local_defs[idx].is_none() {
                        dirty.push(idx);
                    }

                    local_defs[idx] = Some(local);
                    ops.set_destination(node, 0, local, operands);
                }
            }
        }

        for &idx in &dirty {
            let local = local_defs[idx].unwrap();
            let key = idx as u32;

            global_defs[block_index].try_add_operand(key, local);

            worklist.clear();
            worklist.push(block);

            while let Some(df_phi_block) = worklist.pop() {
                let n = blocks.get(df_phi_block).dom_frontiers.len();
                for i in 0..n {
                    let dom_frontier = blocks.get(df_phi_block).dom_frontiers[i];
                    let df_idx = blocks.get(dom_frontier).index as usize;

                    if global_defs[df_idx].add_phi(key) {
                        worklist.push(dom_frontier);
                    }
                }
            }
        }

        for &idx in &dirty {
            local_defs[idx] = None;
        }
    }

    dirty.clear();

    // Compute live-in masks from the renamed IR
    //
    // This is the pruning layer:
    // only keys that are actually live-in to a block are eligible for
    // second-pass resolution in that block.

    let mut block_use = vec![0u64; block_count * word_count];
    let mut block_def = vec![0u64; block_count * word_count];

    for &block in &order {
        let bi = blocks.get(block).index as usize;
        let op_count = blocks.get(block).ops.len();

        for op_idx in 0..op_count {
            let node = blocks.get(block).ops[op_idx];

            for si in 0..ops.sources(node).len() {
                let src = ops.sources(node)[si];
                if let Some(key) = try_get_id(operands, src) {
                    let k = key as usize;
                    let def_row = &block_def[bi * word_count..(bi + 1) * word_count];
                    if !bit_test(def_row, k) {
                        let use_row = &mut block_use[bi * word_count..(bi + 1) * word_count];
                        bit_set(use_row, k);
                    }
                }
            }

            if let Some(&dest) = ops.dests(node).first() {
                if let Some(key) = try_get_id(operands, dest) {
                    let def_row = &mut block_def[bi * word_count..(bi + 1) * word_count];
                    bit_set(def_row, key as usize);
                }
            }
        }
    }

    let mut out = vec![0u64; word_count];
    let mut in_new = vec![0u64; word_count];

    let mut live_in = vec![0u64; block_count * word_count];
    let mut live_out = vec![0u64; block_count * word_count];

    let mut changed = true;
    while changed {
        changed = false;

        for &block in order.iter().rev() {
            let bi = blocks.get(block).index as usize;

            out.fill(0);
            for &succ in &successors[bi] {
                let si = blocks.get(succ).index as usize;
                let src = &live_in[si * word_count..(si + 1) * word_count];
                for w in 0..word_count {
                    out[w] |= src[w];
                }
            }

            let use_row = &block_use[bi * word_count..(bi + 1) * word_count];
            in_new.copy_from_slice(use_row);
            let def_row = &block_def[bi * word_count..(bi + 1) * word_count];
            for w in 0..word_count {
                in_new[w] |= out[w] & !def_row[w];
            }

            let live_in_row = &mut live_in[bi * word_count..(bi + 1) * word_count];
            if in_new.as_slice() != &live_in_row[..] {
                live_in_row.copy_from_slice(&in_new);
                changed = true;
            }

            let live_out_row = &mut live_out[bi * word_count..(bi + 1) * word_count];
            if out.as_slice() != &live_out_row[..] {
                live_out_row.copy_from_slice(&out);
                changed = true;
            }
        }
    }

    dirty.clear();

    // Second pass: resolve cross-block sources, inserting phis as needed
    let mut seen_pending = vec![0u64; word_count];

    for &block in &order {
        dirty.clear();
        pending_sources.clear();
        seen_pending.fill(0);

        let bi = blocks.get(block).index as usize;
        let live = &live_in[bi * word_count..(bi + 1) * word_count];

        // Phase A: collect only live-in unresolved source positions.
        let op_count = blocks.get(block).ops.len();
        for op_idx in 0..op_count {
            let node = blocks.get(block).ops[op_idx];

            for si in 0..ops.sources(node).len() {
                let src = ops.sources(node)[si];
                if let Some(key) = try_get_id(operands, src) {
                    let idx = key as usize;

                    if local_defs[idx].is_none() && bit_test(live, idx) {
                        pending_sources.push(PendingSource {
                            node,
                            source_index: si,
                            key,
                            typ: operands.get(src).typ,
                        });
                    }
                }
            }
        }

        // Phase B: resolve each live missing definition once.
        for pending in &pending_sources {
            let idx = pending.key as usize;

            if local_defs[idx].is_none() {
                let local = find_def(
                    &mut global_defs,
                    blocks,
                    ops,
                    operands,
                    &mut phi_counts,
                    block,
                    pending.key,
                    pending.typ,
                );

                dirty.push(idx);
                local_defs[idx] = Some(local);
            }
        }

        // Phase C: patch only the recorded source sites.
        for pending in &pending_sources {
            if let Some(local) = local_defs[pending.key as usize] {
                ops.set_source(pending.node, pending.source_index, local, operands);
            }
        }

        for &idx in &dirty {
            local_defs[idx] = None;
        }
    }
}

/// C# Ssa.Deconstruct(cfg).
///
/// Replaces phi operations with Copy instructions: each phi input becomes
/// `Copy(temp, source)` appended (before the terminator) to the predecessor
/// block; the phi itself is replaced by `Copy(dest, temp)`.
pub fn deconstruct(
    _cfg: &mut ControlFlowGraph,
    ctx: &mut crate::ir::translation::TranslationContext,
) {
    let blocks = &mut ctx.blocks;
    let ops = &mut ctx.ops;
    let operands = &mut ctx.operands;

    let order: Vec<BlockIndex> = blocks.order().to_vec();

    for &block in &order {
        let phis: Vec<OperationIdx> = blocks
            .get(block)
            .ops
            .iter()
            .copied()
            .take_while(|&op| ops.get(op).is_phi())
            .collect();

        // Phis are at positions 0..n at the start of the loop.
        // Each iteration is a net in-place replace inserting copy_i right
        // before phi_i and then removing phi_i leaves the list the same
        // length and keeps phi_{i+1} at index i+1.  So phi i is always at
        // slot i during the i-th iteration, and we can write directly into
        // `ops[i]` instead of doing insert + remove (which would each scan
        // O(n) for the position).
        for (i, &phi) in phis.iter().enumerate() {
            let dest = ops.dests(phi)[0];
            let typ = operands.get(dest).typ;
            let temp = operands.local(typ);

            let phi_inputs: Vec<_> = ops
                .phi_inputs(phi)
                .iter()
                .map(|input| (input.block, input.value))
                .collect();

            for (pred_block, source) in &phi_inputs {
                let copy = ops.emit_raw(Instruction::Copy, &[temp], &[*source], 0);
                operand_add_assignment(operands, temp, copy);
                operand_add_use(operands, *source, copy);
                append_op(blocks, ops, *pred_block, copy);
            }

            let copy = ops.emit_raw(Instruction::Copy, &[dest], &[temp], 0);
            operand_add_assignment(operands, dest, copy);
            operand_add_use(operands, temp, copy);

            // Direct slot replace.
            blocks.get_mut(block).ops[i] = copy;
            ops.kill(phi, operands);
        }
    }
}
