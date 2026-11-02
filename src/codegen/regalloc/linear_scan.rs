// Port of CodeGen/RegisterAllocators/LinearScanAllocator.cs +
// LiveInterval.cs + LiveRange.cs + UseList.cs + LiveIntervalList.cs +
// CopyResolver.cs.
//
// Based on "Linear Scan Register Allocation for the Java(tm) HotSpot
// Client Compiler" (Wimmer04a)  same algorithm as the C# source.
//
// Representation divergences (semantics identical):
//   - C# LiveInterval/LiveRange are pointers into arena allocators; here
//     intervals and ranges live in Vecs and are addressed by index
//     (IntervalIdx = usize into `intervals`, RangeIdx = u32 into `ranges`,
//     RANGE_NONE = null). No raw pointers.
//   - C# _intervals is one List that is both identity and scan order;
//     insertions shift list positions. Here identity (`intervals` arena,
//     stable indices  also the local numbering) and scan order (`sorted`,
//     = C# _intervals list order) are separate. Active/Inactive bitmaps
//     are keyed by arena index instead of list position; equivalent
//     because C# never shifts a position that is already in a set
//     (children are always inserted after the scan cursor).
//   - C# stores the visited flag in the MSB of the local's value;
//     local_number() reads the low 32 bits, so the same trick works here.
//   - Operand rewriting uses the raw setters: nothing reads def-use
//     information after register allocation.

use crate::util::hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;

use crate::codegen::x86_64::allocation_result::{AllocationResult, RegisterMasks, StackAllocator};
use crate::ir::block::BlockIndex;
use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, OperandKind, OperandType, RegisterType};
use crate::translation::cfg::ControlFlowGraph;

const INSTRUCTION_GAP: i32 = 2;
const INSTRUCTION_GAP_MASK: i32 = INSTRUCTION_GAP - 1;

/// C# LiveInterval.NotFound.
const NOT_FOUND: i32 = -1;

/// C# visited-bit trick in NumberLocals: MSB of the local's value.
const VISITED_MASK: u64 = 1u64 << 63;

// Register (C# Register struct)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Register {
    index: i32,
    reg_type: RegisterType,
}

fn get_register_id(register: Register) -> usize {
    ((register.index as usize) << 1)
        | if register.reg_type == RegisterType::Vector {
            1
        } else {
            0
        }
}

// BitMap (C# ARMeilleure.Common.BitMap  grows on Set)
/// Word-backed bit set (one `u64` per 64 bits). Mirrors C#
/// ARMeilleure.Common.BitMap (`long[]` masks)  grows on Set. Replaces the
/// earlier `Vec<bool>` (one byte per bit) so union/iter run word-wise instead
/// of element-by-element: ~64× fewer iterations in the LSRA liveness fixpoint.
struct BitMap {
    words: Vec<u64>,
}

const BITMAP_BITS: usize = 64;

impl BitMap {
    fn new(size: usize) -> Self {
        Self {
            words: vec![0u64; size.div_ceil(BITMAP_BITS)],
        }
    }
    #[inline]
    fn ensure_word(&mut self, word: usize) {
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
    }
    fn set(&mut self, bit: usize) {
        let w = bit / BITMAP_BITS;
        self.ensure_word(w);
        self.words[w] |= 1u64 << (bit % BITMAP_BITS);
    }
    fn clear(&mut self, bit: usize) {
        let w = bit / BITMAP_BITS;
        if w < self.words.len() {
            self.words[w] &= !(1u64 << (bit % BITMAP_BITS));
        }
    }
    fn is_set(&self, bit: usize) -> bool {
        let w = bit / BITMAP_BITS;
        w < self.words.len() && (self.words[w] >> (bit % BITMAP_BITS)) & 1 != 0
    }
    fn union(&mut self, other: &BitMap) -> bool {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        let mut modified = false;
        for i in 0..other.words.len() {
            let before = self.words[i];
            let after = before | other.words[i];
            if after != before {
                self.words[i] = after;
                modified = true;
            }
        }
        modified
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(wi, &word)| BitWordIter {
                word,
                base: wi * BITMAP_BITS,
            })
    }
}

/// Active/inactive working set. C# holds these as a List iterated O(active)
/// (active ≈ register count). A plain BitMap sized to the interval count made
/// `iter()` walk every word → O(n) per allocation → O(n^2) over the scan.
/// Keep a presence bitmap (O(1) set/clear + dedup) plus a members Vec so
/// iteration is O(active). Only set/clear/iter are needed here  no membership
/// query  so this is behaviourally identical to the old BitMap usage.
struct IntervalSet {
    present: Vec<u64>,
    members: Vec<usize>,
}

impl IntervalSet {
    fn new(size: usize) -> Self {
        Self {
            present: vec![0u64; size.div_ceil(BITMAP_BITS)],
            members: Vec::new(),
        }
    }
    fn set(&mut self, bit: usize) {
        let w = bit / BITMAP_BITS;
        if w >= self.present.len() {
            self.present.resize(w + 1, 0);
        }
        if (self.present[w] >> (bit % BITMAP_BITS)) & 1 == 0 {
            self.present[w] |= 1u64 << (bit % BITMAP_BITS);
            self.members.push(bit);
        }
    }
    fn clear(&mut self, bit: usize) {
        let w = bit / BITMAP_BITS;
        if w < self.present.len() && (self.present[w] >> (bit % BITMAP_BITS)) & 1 != 0 {
            self.present[w] &= !(1u64 << (bit % BITMAP_BITS));
            if let Some(pos) = self.members.iter().position(|&m| m == bit) {
                self.members.swap_remove(pos);
            }
        }
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.members.iter().copied()
    }
}

/// Yields the set-bit indices of a single 64-bit word, lowest first.
struct BitWordIter {
    word: u64,
    base: usize,
}

impl Iterator for BitWordIter {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.word == 0 {
            return None;
        }
        let tz = self.word.trailing_zeros() as usize;
        self.word &= self.word - 1; // clear lowest set bit
        Some(self.base + tz)
    }
}

// LiveRange (C# LiveRange.cs  arena node)
const RANGE_NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct LiveRange {
    start: i32,
    end: i32,
    next: u32, // RANGE_NONE = null
}

impl LiveRange {
    /// C# LiveRange.Overlaps(int position).
    fn overlaps_position(&self, position: i32) -> bool {
        position >= self.start && position < self.end
    }
    /// C# LiveRange.Overlaps(int start, int end).
    fn overlaps_range(&self, start: i32, end: i32) -> bool {
        self.start < end && start < self.end
    }
}

// LiveInterval (C# LiveInterval.cs  arena entry)
struct LiveInterval {
    end: i32,
    spill_offset: i32,

    first_range: u32, // RangeIdx or RANGE_NONE
    prev_range: u32,
    curr_range: u32,

    /// C# Parent  self index for root intervals.
    parent: usize,
    /// C# CopySource (interval index).
    copy_source: Option<usize>,

    /// C# UseList  positions in DESCENDING order.
    uses: Vec<i32>,
    /// C# LiveIntervalList  split children, on the PARENT only, sorted
    /// ascending by start position.
    children: Vec<usize>,

    local: OperandIndex,
    /// C# interval.Local.Type (cached: Spill/GetRegister need it where no
    /// arena access is available).
    local_type: OperandType,
    register: Register,

    is_fixed: bool,
    is_fixed_and_used: bool,
}

impl LiveInterval {
    fn new_local(local: OperandIndex, local_type: OperandType, parent: usize) -> Self {
        Self {
            end: 0,
            spill_offset: -1,
            first_range: RANGE_NONE,
            prev_range: RANGE_NONE,
            curr_range: RANGE_NONE,
            parent,
            copy_source: None,
            uses: Vec::new(),
            children: Vec::new(),
            local,
            local_type,
            register: Register {
                index: 0,
                reg_type: RegisterType::Integer,
            },
            is_fixed: false,
            is_fixed_and_used: false,
        }
    }

    fn new_fixed(register: Register, parent: usize) -> Self {
        let mut interval = Self::new_local(OperandIndex(u32::MAX), OperandType::I64, parent);
        interval.register = register;
        interval.is_fixed = true;
        interval
    }

    fn is_empty(&self) -> bool {
        self.first_range == RANGE_NONE
    }
    fn is_split(&self) -> bool {
        !self.children.is_empty()
    }
    fn is_spilled(&self) -> bool {
        self.spill_offset != -1
    }
    fn uses_count(&self) -> usize {
        self.uses.len()
    }

    /// C# UseList.FirstUse  last element of the descending list.
    fn first_use(&self) -> i32 {
        self.uses.last().copied().unwrap_or(NOT_FOUND)
    }

    /// C# UseList.NextUseIndex.
    fn next_use_index(&self, position: i32) -> i32 {
        let mut i = self.uses.len() as i32 - 1;
        if i == -1 || position > self.uses[0] {
            return NOT_FOUND;
        }
        while i >= 0 && self.uses[i as usize] < position {
            i -= 1;
        }
        i
    }

    /// C# UseList.NextUse.
    fn next_use_after(&self, position: i32) -> i32 {
        let index = self.next_use_index(position);
        if index != NOT_FOUND {
            self.uses[index as usize]
        } else {
            NOT_FOUND
        }
    }

    /// C# UseList.Add  insert keeping descending order.
    fn add_use_position(&mut self, position: i32) {
        let mut i = self.uses.len() as i32 - 1;
        self.uses.push(0);
        while i >= 0 && self.uses[i as usize] < position {
            self.uses[i as usize + 1] = self.uses[i as usize];
            i -= 1;
        }
        self.uses[(i + 1) as usize] = position;
    }
}

// CopyResolver (C# CopyResolver.cs)
struct ParallelCopy {
    copies: Vec<(Register, Register, OperandType)>, // (dest, source, type)
}

impl ParallelCopy {
    fn new() -> Self {
        Self { copies: Vec::new() }
    }

    fn add_copy(&mut self, dest: Register, source: Register, typ: OperandType) {
        self.copies.push((dest, source, typ));
    }

    /// C# ParallelCopy.Sequence  emits copies in dependency order, using
    /// xor-swaps to break cycles.
    fn sequence(&self, ctx: &mut TranslationContext, sequence: &mut Vec<OperationIdx>) {
        let mut locations: HashMap<Register, Register> = HashMap::default();
        let mut sources: HashMap<Register, Register> = HashMap::default();
        let mut types: HashMap<Register, OperandType> = HashMap::default();

        let mut pending_queue: VecDeque<Register> = VecDeque::new();
        let mut ready_queue: VecDeque<Register> = VecDeque::new();

        for &(dest, source, typ) in &self.copies {
            locations.insert(source, source);
            sources.insert(dest, source);
            types.insert(dest, typ);
            pending_queue.push_back(dest);
        }

        for &(dest, _, _) in &self.copies {
            // If the destination is not used anywhere, we can assign it immediately.
            if !locations.contains_key(&dest) {
                ready_queue.push_back(dest);
            }
        }

        while let Some(current) = pending_queue.pop_front() {
            while let Some(copy_dest) = ready_queue.pop_front() {
                let orig_source = sources[&copy_dest];
                let copy_source = locations[&orig_source];
                let typ = types[&copy_dest];

                emit_copy(ctx, sequence, copy_dest, copy_source, typ);

                locations.insert(orig_source, copy_dest);

                if orig_source == copy_source && sources.contains_key(&orig_source) {
                    ready_queue.push_back(orig_source);
                }
            }

            let copy_dest = current;
            let orig_source = sources[&copy_dest];
            let copy_source = locations[&orig_source];

            if copy_dest != copy_source {
                let typ = types[&copy_dest];
                let typ = if typ.is_integer() {
                    OperandType::I64
                } else {
                    OperandType::V128
                };

                emit_xor_swap(ctx, sequence, copy_dest, copy_source, typ);

                locations.insert(orig_source, copy_dest);

                let mut swap_other = copy_source;

                if copy_dest != locations[&sources[&copy_source]] {
                    // Find the other swap destination register: the pending
                    // register whose copy source is the current dest.
                    for &pending in &pending_queue {
                        if copy_dest == locations[&sources[&pending]] {
                            swap_other = pending;
                            break;
                        }
                    }
                }

                // The value previously at copy_dest now lives at copy_source.
                locations.insert(sources[&swap_other], copy_source);
            }
        }
    }
}

fn register_operand(
    ctx: &mut TranslationContext,
    register: Register,
    typ: OperandType,
) -> OperandIndex {
    ctx.operands
        .register(register.index as u32, register.reg_type, typ)
}

fn emit_copy(
    ctx: &mut TranslationContext,
    sequence: &mut Vec<OperationIdx>,
    dest: Register,
    source: Register,
    typ: OperandType,
) {
    let x = register_operand(ctx, dest, typ);
    let y = register_operand(ctx, source, typ);
    sequence.push(ctx.make_op(Instruction::Copy, &[x], &[y]));
}

fn emit_xor_swap(
    ctx: &mut TranslationContext,
    sequence: &mut Vec<OperationIdx>,
    dest: Register,
    source: Register,
    typ: OperandType,
) {
    let x = register_operand(ctx, dest, typ);
    let y = register_operand(ctx, source, typ);
    sequence.push(ctx.make_op(Instruction::BitwiseExclusiveOr, &[x], &[x, y]));
    sequence.push(ctx.make_op(Instruction::BitwiseExclusiveOr, &[y], &[y, x]));
    sequence.push(ctx.make_op(Instruction::BitwiseExclusiveOr, &[x], &[x, y]));
}

#[derive(Default)]
struct CopyResolver {
    fill_queue: Vec<(Register, i32, OperandType)>, // (dest reg, offset, type)
    spill_queue: Vec<(i32, Register, OperandType)>, // (offset, source reg, type)
    parallel_copy: Option<ParallelCopy>,
    has_copy: bool,
}

impl CopyResolver {
    fn new() -> Self {
        Self::default()
    }

    /// C# CopyResolver.AddSplit.
    fn add_split(&mut self, left: &LiveInterval, right: &LiveInterval) {
        assert_eq!(
            left.local, right.local,
            "Intervals of different variables are not allowed."
        );

        let typ = left.local_type;

        if left.is_spilled() && !right.is_spilled() {
            // Move from the stack to a register.
            self.fill_queue
                .push((right.register, left.spill_offset, typ));
            self.has_copy = true;
        } else if !left.is_spilled() && right.is_spilled() {
            // Move from a register to the stack.
            self.spill_queue
                .push((right.spill_offset, left.register, typ));
            self.has_copy = true;
        } else if !left.is_spilled() && !right.is_spilled() && left.register != right.register {
            // Move from one register to another.
            self.parallel_copy
                .get_or_insert_with(ParallelCopy::new)
                .add_copy(right.register, left.register, typ);
            self.has_copy = true;
        } else if left.is_spilled() && right.is_spilled() && left.spill_offset != right.spill_offset
        {
            panic!("Both intervals were spilled.");
        }
    }

    /// C# CopyResolver.Sequence  spills, then parallel copies, then fills.
    fn sequence(&self, ctx: &mut TranslationContext) -> Vec<OperationIdx> {
        let mut sequence: Vec<OperationIdx> = Vec::new();

        for &(offset, source, typ) in &self.spill_queue {
            let offset_op = ctx.operands.constant_i32(offset);
            let source_op = register_operand(ctx, source, typ);
            sequence.push(ctx.make_op(Instruction::Spill, &[], &[offset_op, source_op]));
        }

        if let Some(ref parallel_copy) = self.parallel_copy {
            parallel_copy.sequence(ctx, &mut sequence);
        }

        for &(dest, offset, typ) in &self.fill_queue {
            let dest_op = register_operand(ctx, dest, typ);
            let offset_op = ctx.operands.constant_i32(offset);
            sequence.push(ctx.make_op(Instruction::Fill, &[dest_op], &[offset_op]));
        }

        sequence
    }
}

// AllocationContext (C# private class)
struct AllocationContext {
    stack_alloc: StackAllocator,
    masks: RegisterMasks,

    active: IntervalSet,
    inactive: IntervalSet,

    int_used_registers: u32,
    vec_used_registers: u32,

    int_free_positions: Vec<i32>,
    int_free_positions_count: i32,
    vec_free_positions: Vec<i32>,
    vec_free_positions_count: i32,
}

impl AllocationContext {
    fn new(stack_alloc: StackAllocator, masks: RegisterMasks, intervals_count: usize) -> Self {
        // C# PopulateFreePositions: positions default to 0; only AVAILABLE
        // registers get int.MaxValue.
        fn populate(mask: u32, count: usize) -> (Vec<i32>, i32) {
            let mut positions = vec![0i32; count];
            let mut free_count = 0i32;
            for (i, position) in positions.iter_mut().enumerate() {
                if (mask & (1u32 << i)) != 0 {
                    *position = i32::MAX;
                    free_count += 1;
                }
            }
            (positions, free_count)
        }

        let rc = masks.registers_count as usize;
        let (int_free_positions, int_free_positions_count) = populate(masks.int_available, rc);
        let (vec_free_positions, vec_free_positions_count) = populate(masks.vec_available, rc);

        Self {
            stack_alloc,
            masks,
            active: IntervalSet::new(intervals_count),
            inactive: IntervalSet::new(intervals_count),
            int_used_registers: 0,
            vec_used_registers: 0,
            int_free_positions,
            int_free_positions_count,
            vec_free_positions,
            vec_free_positions_count,
        }
    }

    /// C# GetFreePositions  COPIES the master array; callers mutate the copy.
    fn get_free_positions(&self, reg_type: RegisterType) -> (Vec<i32>, i32) {
        if reg_type == RegisterType::Integer {
            (
                self.int_free_positions.clone(),
                self.int_free_positions_count,
            )
        } else {
            (
                self.vec_free_positions.clone(),
                self.vec_free_positions_count,
            )
        }
    }

    fn move_active_to_inactive(&mut self, bit: usize) {
        self.active.clear(bit);
        self.inactive.set(bit);
    }

    fn move_inactive_to_active(&mut self, bit: usize) {
        self.inactive.clear(bit);
        self.active.set(bit);
    }
}

// LinearScanAllocator
struct LinearScanAllocator {
    block_edges: HashSet<i32>,
    /// C# _blockRanges  (start, end) indexed by block.index.
    block_ranges: Vec<(i32, i32)>,
    /// C# _blockLiveIn  indexed by block.index, bits are interval ids.
    block_live_in: Vec<BitMap>,

    /// Identity arena. Indices are stable; locals are numbered with their
    /// arena index by number_locals.
    intervals: Vec<LiveInterval>,
    /// C# _intervals list ORDER: fixed prefix then sorted by start.
    sorted: Vec<usize>,
    /// C# _parentIntervals  count of intervals at the end of
    /// number_locals (everything before is a parent / fixed interval).
    parent_count: usize,

    /// Range arena (C# LiveRange allocator).
    ranges: Vec<LiveRange>,

    /// C# _operationNodes  (block, op) per position/GAP, in RPO.
    /// OperationIdx(u32::MAX) = empty-block dummy.
    operation_nodes: Vec<(BlockIndex, OperationIdx)>,
    /// C# _operationsCount.
    operations_count: i32,
}

/// Entry point  C# LinearScanAllocator.RunPass.
pub fn run_pass(
    cfg: &ControlFlowGraph,
    ctx: &mut TranslationContext,
    stack_alloc: &mut StackAllocator,
    reg_masks: &RegisterMasks,
) -> AllocationResult {
    let registers_count = reg_masks.registers_count;

    let mut allocator = LinearScanAllocator {
        block_edges: HashSet::default(),
        block_ranges: Vec::new(),
        block_live_in: Vec::new(),
        intervals: Vec::new(),
        sorted: Vec::new(),
        parent_count: 0,
        ranges: Vec::new(),
        operation_nodes: Vec::new(),
        operations_count: 0,
    };

    let _t = std::env::var("ARMATURE_LSRA_TIME").is_ok();
    macro_rules! ph {
        ($s:expr) => {
            if _t {
                Some((std::time::Instant::now(), $s))
            } else {
                None
            }
        };
    }
    macro_rules! end {
        ($g:expr) => {
            if let Some((t, s)) = $g {
                eprintln!("  LSRA {s}: {:?}", t.elapsed());
            }
        };
    }

    let g = ph!("number_locals");
    allocator.number_locals(cfg, ctx, registers_count);
    end!(g);

    let mut context = AllocationContext::new(
        std::mem::replace(stack_alloc, StackAllocator::new()),
        *reg_masks,
        allocator.intervals.len(),
    );

    let g = ph!("build_intervals");
    allocator.build_intervals(cfg, ctx, &context);
    end!(g);
    let g_alloc = ph!("allocate_loop");

    // C#: for (index = 0; index < _intervals.Count; index++)  the list
    // grows during allocation (split children inserted sorted by start)
    // and the loop picks them up.
    let mut index = 0usize;
    while index < allocator.sorted.len() {
        let current = allocator.sorted[index];
        index += 1;

        if allocator.intervals[current].is_empty() {
            continue;
        }

        if allocator.intervals[current].is_fixed {
            context.active.set(current);

            if allocator.intervals[current].is_fixed_and_used {
                let register = allocator.intervals[current].register;
                if register.reg_type == RegisterType::Integer {
                    context.int_used_registers |= 1u32 << register.index;
                } else {
                    context.vec_used_registers |= 1u32 << register.index;
                }
            }

            continue;
        }

        allocator.allocate_interval(&mut context, current, registers_count);
    }
    end!(g_alloc);
    if _t {
        eprintln!(
            "  LSRA intervals={} sorted={}",
            allocator.intervals.len(),
            allocator.sorted.len()
        );
    }

    if std::env::var("ARMATURE_REGALLOC_DUMP").is_ok() {
        eprintln!(
            "== regalloc dump: {} intervals (fixed prefix {}) ==",
            allocator.intervals.len(),
            registers_count * 2
        );
        for id in (registers_count as usize * 2)..allocator.intervals.len() {
            let iv = &allocator.intervals[id];
            let loc = if iv.is_spilled() {
                format!("SPILL@{}", iv.spill_offset)
            } else {
                format!("{:?}{}", iv.register.reg_type, iv.register.index)
            };
            eprintln!(
                "  iv{id} parent={} split={} children={:?} {} start={} uses={:?}",
                iv.parent,
                iv.is_split(),
                iv.children,
                loc,
                if iv.is_empty() {
                    -1
                } else {
                    allocator.iv_get_start(id)
                },
                iv.uses,
            );
        }
    }

    // C#: replace locals on every non-spilled interval past the fixed
    // prefix (parents AND split children).
    for index in (registers_count as usize * 2)..allocator.intervals.len() {
        if !allocator.intervals[index].is_spilled() {
            allocator.replace_local_with_register(ctx, index);
        }
    }

    let g = ph!("insert_split_copies");
    allocator.insert_split_copies(ctx);
    end!(g);
    let g = ph!("insert_split_copies_at_edges");
    allocator.insert_split_copies_at_edges(ctx);
    end!(g);

    *stack_alloc = context.stack_alloc;

    AllocationResult::new(
        context.int_used_registers,
        context.vec_used_registers,
        stack_alloc.total_size(),
    )
}

impl LinearScanAllocator {
    // LiveInterval methods that need the arenas
    /// C# LiveInterval.Reset.
    fn iv_reset(&mut self, i: usize) {
        let interval = &mut self.intervals[i];
        interval.prev_range = RANGE_NONE;
        interval.curr_range = interval.first_range;
    }

    /// C# LiveInterval.Forward.
    fn iv_forward(&mut self, i: usize, position: i32) {
        let mut prev = self.intervals[i].prev_range;
        let mut curr = self.intervals[i].curr_range;

        while curr != RANGE_NONE {
            let range = self.ranges[curr as usize];
            if range.start >= position || range.overlaps_position(position) {
                break;
            }
            prev = curr;
            curr = range.next;
        }

        self.intervals[i].prev_range = prev;
        self.intervals[i].curr_range = curr;
    }

    /// C# LiveInterval.GetStart.
    fn iv_get_start(&self, i: usize) -> i32 {
        let first = self.intervals[i].first_range;
        debug_assert!(
            first != RANGE_NONE,
            "Empty LiveInterval cannot have a start position."
        );
        self.ranges[first as usize].start
    }

    /// C# LiveInterval.SetStart.
    fn iv_set_start(&mut self, i: usize, position: i32) {
        let first = self.intervals[i].first_range;
        if first != RANGE_NONE {
            self.ranges[first as usize].start = position;
        } else {
            let range_idx = self.alloc_range(position, position + 1, RANGE_NONE);
            self.intervals[i].first_range = range_idx;
            self.intervals[i].end = position + 1;
        }
    }

    /// C# LiveInterval.GetEnd.
    fn iv_get_end(&self, i: usize) -> i32 {
        self.intervals[i].end
    }

    fn alloc_range(&mut self, start: i32, end: i32, next: u32) -> u32 {
        self.ranges.push(LiveRange { start, end, next });
        (self.ranges.len() - 1) as u32
    }

    /// C# LiveInterval.AddRange.
    fn iv_add_range(&mut self, i: usize, start: i32, end: i32) {
        debug_assert!(start < end, "Invalid range start position {start}, {end}");

        let first = self.intervals[i].first_range;
        if first != RANGE_NONE {
            let first_range = self.ranges[first as usize];
            // If the new range ends exactly where the first range starts,
            // coalesce together.
            if end == first_range.start {
                self.ranges[first as usize].start = start;
                return;
            } else if first_range.overlaps_range(start, end) {
                let r = &mut self.ranges[first as usize];
                r.start = r.start.min(start);
                r.end = r.end.max(end);
                let new_end = r.end;
                let interval = &mut self.intervals[i];
                interval.end = interval.end.max(new_end);
                return;
            }
        }

        let range_idx = self.alloc_range(start, end, first);
        let interval = &mut self.intervals[i];
        interval.first_range = range_idx;
        interval.end = interval.end.max(end);
    }

    /// C# LiveInterval.Overlaps(int position)  walks from curr_range.
    fn iv_overlaps_position(&self, i: usize, position: i32) -> bool {
        let mut curr = self.intervals[i].curr_range;
        while curr != RANGE_NONE {
            let range = self.ranges[curr as usize];
            if range.start > position {
                break;
            }
            if range.overlaps_position(position) {
                return true;
            }
            curr = range.next;
        }
        false
    }

    /// C# LiveInterval.GetOverlapPosition(other)  parallel walk from the
    /// current cursor of both intervals.
    fn iv_get_overlap_position(&self, i: usize, other: usize) -> i32 {
        let mut a = self.intervals[i].curr_range;
        let mut b = self.intervals[other].curr_range;

        while a != RANGE_NONE {
            let ra = self.ranges[a as usize];

            while b != RANGE_NONE {
                let rb = self.ranges[b as usize];
                if rb.start >= ra.start {
                    break;
                }
                if ra.overlaps_range(rb.start, rb.end) {
                    return ra.start;
                }
                b = rb.next;
            }

            if b == RANGE_NONE {
                break;
            }
            let rb = self.ranges[b as usize];
            if ra.overlaps_range(rb.start, rb.end) {
                return ra.start;
            }

            a = ra.next;
        }

        NOT_FOUND
    }

    /// C# LiveInterval.Overlaps(LiveInterval other).
    fn iv_overlaps(&self, i: usize, other: usize) -> bool {
        self.iv_get_overlap_position(i, other) != NOT_FOUND
    }

    /// C# LiveInterval.Split  cuts at `position`, returns the new child's
    /// arena index. The child is registered on the PARENT's children list
    /// (sorted by start).
    fn iv_split(&mut self, i: usize, position: i32) -> usize {
        let parent = self.intervals[i].parent;
        let local = self.intervals[i].local;
        let local_type = self.intervals[i].local_type;
        let end = self.intervals[i].end;

        let mut result = LiveInterval::new_local(local, local_type, parent);
        // The child allocates from the same register class as the parent.
        result.register = Register {
            index: 0,
            reg_type: self.intervals[i].register.reg_type,
        };
        result.end = end;

        // Walk from the cursor, exactly like C#.
        let mut prev = self.intervals[i].prev_range;
        let mut curr = self.intervals[i].curr_range;
        while curr != RANGE_NONE {
            let range = self.ranges[curr as usize];
            if range.start >= position || range.overlaps_position(position) {
                break;
            }
            prev = curr;
            curr = range.next;
        }

        debug_assert!(
            curr != RANGE_NONE,
            "Split position outside interval ranges."
        );

        if self.ranges[curr as usize].start >= position {
            // C#: prev.Next = default; result.FirstRange = curr; End = prev.End.
            debug_assert!(
                prev != RANGE_NONE,
                "Split would leave the left interval empty."
            );
            self.ranges[prev as usize].next = RANGE_NONE;
            result.first_range = curr;
            self.intervals[i].end = self.ranges[prev as usize].end;
        } else {
            // C#: result.FirstRange = new LiveRange(position, curr.End, curr.Next);
            //     curr.End = position; curr.Next = default; End = curr.End.
            let curr_end = self.ranges[curr as usize].end;
            let curr_next = self.ranges[curr as usize].next;
            let new_first = self.alloc_range(position, curr_end, curr_next);
            result.first_range = new_first;
            self.ranges[curr as usize].end = position;
            self.ranges[curr as usize].next = RANGE_NONE;
            self.intervals[i].end = position;
        }

        // C#: result.Uses = Uses.Split(position)  descending list, the
        // front (positions >= split) goes to the child.
        let use_index = self.intervals[i].next_use_index(position);
        if use_index != NOT_FOUND {
            let split_at = use_index as usize + 1;
            result.uses = self.intervals[i].uses.drain(..split_at).collect();
        }

        // result.Reset().
        result.prev_range = RANGE_NONE;
        result.curr_range = result.first_range;

        debug_assert!(
            !self.intervals[i].is_empty(),
            "Left interval is empty after split."
        );
        debug_assert!(!result.is_empty(), "Right interval is empty after split.");

        let child_idx = self.intervals.len();
        let child_start = self.ranges[result.first_range as usize].start;
        self.intervals.push(result);

        // C# AddSplitChild → Parent.Children.Add (insertion sort by start).
        let children = &self.intervals[parent].children;
        let mut insert_at = children.len();
        while insert_at > 0 {
            let prev_child = children[insert_at - 1];
            if self.iv_get_start(prev_child) <= child_start {
                break;
            }
            insert_at -= 1;
        }
        self.intervals[parent].children.insert(insert_at, child_idx);

        child_idx
    }

    /// C# LiveInterval.GetSplitChild(position)  called on a PARENT;
    /// checks itself then its (sorted) children. Depends on cursor state.
    fn iv_get_split_child(&self, i: usize, position: i32) -> Option<usize> {
        if self.iv_overlaps_position(i, position) {
            return Some(i);
        }

        for &child in &self.intervals[self.intervals[i].parent].children {
            if self.iv_overlaps_position(child, position) {
                return Some(child);
            } else if self.iv_get_start(child) > position {
                break;
            }
        }

        None
    }

    /// C# LiveInterval.TrySpillWithSiblingOffset.
    fn iv_try_spill_with_sibling_offset(&mut self, i: usize) -> bool {
        let parent = self.intervals[i].parent;
        for ci in 0..self.intervals[parent].children.len() {
            let child = self.intervals[parent].children[ci];
            if self.intervals[child].is_spilled() {
                let offset = self.intervals[child].spill_offset;
                self.intervals[i].spill_offset = offset;
                return true;
            }
        }
        false
    }

    // Allocation (C# AllocateInterval etc.)
    fn allocate_interval(
        &mut self,
        context: &mut AllocationContext,
        current: usize,
        registers_count: i32,
    ) {
        let current_start = self.iv_get_start(current);

        // Check active intervals that already ended.
        let active_list: Vec<usize> = context.active.iter().collect();
        for interval in active_list {
            self.iv_forward(interval, current_start);

            if self.iv_get_end(interval) < current_start {
                context.active.clear(interval);
            } else if !self.iv_overlaps_position(interval, current_start) {
                context.move_active_to_inactive(interval);
            }
        }

        // Check inactive intervals that already ended or were reactivated.
        let inactive_list: Vec<usize> = context.inactive.iter().collect();
        for interval in inactive_list {
            self.iv_forward(interval, current_start);

            if self.iv_get_end(interval) < current_start {
                context.inactive.clear(interval);
            } else if self.iv_overlaps_position(interval, current_start) {
                context.move_inactive_to_active(interval);
            }
        }

        if !self.try_allocate_reg_without_spill(context, current, registers_count) {
            self.allocate_reg_with_spill(context, current, registers_count);
        }
    }

    /// C# TryAllocateRegWithoutSpill.
    fn try_allocate_reg_without_spill(
        &mut self,
        context: &mut AllocationContext,
        current: usize,
        registers_count: i32,
    ) -> bool {
        let reg_type = self.intervals[current].local_type.to_register_type();

        let (mut free_positions, mut free_positions_count) = context.get_free_positions(reg_type);

        for interval in context.active.iter() {
            let register = self.intervals[interval].register;
            if register.reg_type == reg_type {
                free_positions[register.index as usize] = 0;
                free_positions_count -= 1;
            }
        }

        // If all registers are already active, return early.
        if free_positions_count <= 0 {
            return false;
        }

        let inactive_list: Vec<usize> = context.inactive.iter().collect();
        for interval in inactive_list {
            let register = self.intervals[interval].register;
            if register.reg_type == reg_type && free_positions[register.index as usize] != 0 {
                let overlap_position = self.iv_get_overlap_position(interval, current);
                if overlap_position != NOT_FOUND
                    && free_positions[register.index as usize] > overlap_position
                {
                    free_positions[register.index as usize] = overlap_position;
                }
            }
        }

        // Prefer the copy source register when free for the whole lifetime.
        let current_end = self.iv_get_end(current);
        let selected_reg = match self.intervals[current].copy_source {
            Some(copy_source)
                if free_positions[self.intervals[copy_source].register.index as usize]
                    >= current_end =>
            {
                self.intervals[copy_source].register.index as usize
            }
            _ => get_highest_value_index(&free_positions),
        };

        let mut selected_next_use = free_positions[selected_reg];
        selected_next_use &= !INSTRUCTION_GAP_MASK;

        let current_start = self.iv_get_start(current);
        if selected_next_use <= current_start {
            return false;
        } else if selected_next_use < current_end {
            let split_child = self.iv_split(current, selected_next_use);

            if self.intervals[split_child].uses_count() != 0 {
                debug_assert!(
                    self.iv_get_start(split_child) > current_start,
                    "Split interval has an invalid start position."
                );
                self.insert_interval(split_child, registers_count);
            } else {
                self.spill(context, split_child);
            }
        }

        self.intervals[current].register = Register {
            index: selected_reg as i32,
            reg_type,
        };

        if reg_type == RegisterType::Integer {
            context.int_used_registers |= 1u32 << selected_reg;
        } else {
            context.vec_used_registers |= 1u32 << selected_reg;
        }

        context.active.set(current);

        true
    }

    /// C# AllocateRegWithSpill.
    fn allocate_reg_with_spill(
        &mut self,
        context: &mut AllocationContext,
        current: usize,
        registers_count: i32,
    ) {
        let reg_type = self.intervals[current].local_type.to_register_type();
        let current_start = self.iv_get_start(current);
        let current_end = self.iv_get_end(current);

        let (mut use_positions, _) = context.get_free_positions(reg_type);
        let (mut blocked_positions, _) = context.get_free_positions(reg_type);

        let active_list: Vec<usize> = context.active.iter().collect();
        for interval in active_list {
            let register = self.intervals[interval].register;
            if register.reg_type == reg_type {
                let reg_index = register.index as usize;
                if self.intervals[interval].is_fixed {
                    use_positions[reg_index] = 0;
                    blocked_positions[reg_index] = 0;
                } else {
                    let next_use = self.intervals[interval].next_use_after(current_start);
                    if next_use != NOT_FOUND && use_positions[reg_index] > next_use {
                        use_positions[reg_index] = next_use;
                    }
                }
            }
        }

        let inactive_list: Vec<usize> = context.inactive.iter().collect();
        for interval in inactive_list {
            let register = self.intervals[interval].register;
            if register.reg_type == reg_type {
                let reg_index = register.index as usize;
                if self.intervals[interval].is_fixed {
                    let overlap_position = self.iv_get_overlap_position(interval, current);
                    if overlap_position != NOT_FOUND {
                        blocked_positions[reg_index] =
                            blocked_positions[reg_index].min(overlap_position);
                        use_positions[reg_index] = use_positions[reg_index].min(overlap_position);
                    }
                } else if self.iv_overlaps(interval, current) {
                    let next_use = self.intervals[interval].next_use_after(current_start);
                    if next_use != NOT_FOUND && use_positions[reg_index] > next_use {
                        use_positions[reg_index] = next_use;
                    }
                }
            }
        }

        let selected_reg = get_highest_value_index(&use_positions);
        let current_first_use = self.intervals[current].first_use();

        debug_assert!(current_first_use >= 0, "Current interval has no uses.");

        if use_positions[selected_reg] < current_first_use {
            // All intervals on inactive and active are being used before
            // current, so spill the current interval.
            debug_assert!(
                current_first_use > current_start,
                "Trying to spill a interval currently being used."
            );

            let split_child = self.iv_split(current, current_first_use);

            debug_assert!(
                self.iv_get_start(split_child) > current_start,
                "Split interval has an invalid start position."
            );

            self.insert_interval(split_child, registers_count);

            self.spill(context, current);
        } else if blocked_positions[selected_reg] > current_end {
            // Spill made the register available for the entire current
            // lifetime, so we only need to split the intervals using the
            // selected register.
            self.intervals[current].register = Register {
                index: selected_reg as i32,
                reg_type,
            };

            self.split_and_spill_overlapping_intervals(context, current, registers_count);

            context.active.set(current);
        } else {
            // There are conflicts even after spill due to the use of fixed
            // registers that can't be spilled, so we need to also split
            // current at the point of the first fixed register use.
            self.intervals[current].register = Register {
                index: selected_reg as i32,
                reg_type,
            };

            let split_position = blocked_positions[selected_reg] & !INSTRUCTION_GAP_MASK;

            debug_assert!(
                split_position > current_start,
                "Trying to split a interval at a invalid position."
            );

            let split_child = self.iv_split(current, split_position);

            if self.intervals[split_child].uses_count() != 0 {
                debug_assert!(
                    self.iv_get_start(split_child) > current_start,
                    "Split interval has an invalid start position."
                );
                self.insert_interval(split_child, registers_count);
            } else {
                self.spill(context, split_child);
            }

            self.split_and_spill_overlapping_intervals(context, current, registers_count);

            context.active.set(current);
        }
    }

    /// C# SplitAndSpillOverlappingIntervals.
    fn split_and_spill_overlapping_intervals(
        &mut self,
        context: &mut AllocationContext,
        current: usize,
        registers_count: i32,
    ) {
        let current_register = self.intervals[current].register;

        let active_list: Vec<usize> = context.active.iter().collect();
        for interval in active_list {
            if !self.intervals[interval].is_fixed
                && self.intervals[interval].register == current_register
            {
                self.split_and_spill_overlapping_interval(
                    context,
                    current,
                    interval,
                    registers_count,
                );
                context.active.clear(interval);
            }
        }

        let inactive_list: Vec<usize> = context.inactive.iter().collect();
        for interval in inactive_list {
            if !self.intervals[interval].is_fixed
                && self.intervals[interval].register == current_register
                && self.iv_overlaps(interval, current)
            {
                self.split_and_spill_overlapping_interval(
                    context,
                    current,
                    interval,
                    registers_count,
                );
                context.inactive.clear(interval);
            }
        }
    }

    /// C# SplitAndSpillOverlappingInterval.
    fn split_and_spill_overlapping_interval(
        &mut self,
        context: &mut AllocationContext,
        current: usize,
        interval: usize,
        registers_count: i32,
    ) {
        // If there's a next use after the start of the current interval,
        // we need to split the spilled interval twice, and re-insert it
        // on the "pending" list to ensure that it will get a new register
        // on that use position.
        let current_start = self.iv_get_start(current);
        let next_use = self.intervals[interval].next_use_after(current_start);

        let split_child = if self.iv_get_start(interval) < current_start {
            self.iv_split(interval, current_start)
        } else {
            interval
        };

        if next_use != NOT_FOUND {
            debug_assert!(
                next_use > current_start,
                "Trying to spill a interval currently being used."
            );

            let mut split_child = split_child;
            if next_use > self.iv_get_start(split_child) {
                let right = self.iv_split(split_child, next_use);
                self.spill(context, split_child);
                split_child = right;
            }

            self.insert_interval(split_child, registers_count);
        } else {
            self.spill(context, split_child);
        }
    }

    /// C# InsertInterval  sorted insert into the scan list, after the
    /// fixed prefix.
    fn insert_interval(&mut self, interval: usize, registers_count: i32) {
        debug_assert!(
            self.intervals[interval].uses_count() != 0,
            "Trying to insert a interval without uses."
        );
        debug_assert!(
            !self.intervals[interval].is_empty(),
            "Trying to insert a empty interval."
        );
        debug_assert!(
            !self.intervals[interval].is_spilled(),
            "Trying to insert a spilled interval."
        );

        let start_index = registers_count as usize * 2;
        let start = self.iv_get_start(interval);

        let offset = self.sorted[start_index..].partition_point(|&id| {
            !self.intervals[id].is_empty() && self.iv_get_start(id) <= start
        });

        self.sorted.insert(start_index + offset, interval);
    }

    /// C# Spill.
    fn spill(&mut self, context: &mut AllocationContext, interval: usize) {
        debug_assert!(
            !self.intervals[interval].is_fixed,
            "Trying to spill a fixed interval."
        );
        debug_assert!(
            self.intervals[interval].uses_count() == 0,
            "Trying to spill a interval with uses."
        );

        // We first check if any of the siblings were spilled, if so we can
        // reuse the stack offset. This prevents stack-to-stack copies being
        // necessary for a split interval.
        if !self.iv_try_spill_with_sibling_offset(interval) {
            let offset = context
                .stack_alloc
                .allocate(self.intervals[interval].local_type);
            self.intervals[interval].spill_offset = offset;
        }
    }

    // Split copies (C# InsertSplitCopies / ...AtEdges)
    /// C# InsertSplitCopies.
    fn insert_split_copies(&mut self, ctx: &mut TranslationContext) {
        let mut copy_resolvers: std::collections::BTreeMap<i32, CopyResolver> =
            std::collections::BTreeMap::new();

        // Only parents have children (a child's split registers on the
        // parent), so iterating the arena where is_split == parents.
        for interval in 0..self.intervals.len() {
            if !self.intervals[interval].is_split() {
                continue;
            }

            let mut previous = interval;
            for ci in 0..self.intervals[interval].children.len() {
                let split_child = self.intervals[interval].children[ci];
                let split_position = self.iv_get_start(split_child);

                if !self.block_edges.contains(&split_position)
                    && self.iv_get_end(previous) == split_position
                {
                    copy_resolvers
                        .entry(split_position)
                        .or_insert_with(CopyResolver::new)
                        .add_split(&self.intervals[previous], &self.intervals[split_child]);
                }

                previous = split_child;
            }
        }

        // C# uses an intrusive LinkedList  AddBefore(node, op) is O(1). Our
        // block ops are a Vec, so a per-copy position()+insert was O(n) each →
        // O(n^2) over a big function. Instead batch every sequence by
        // (block, node), then rebuild each affected block's ops in ONE pass:
        // O(ops + copies). copy_resolvers is a BTreeMap, so iteration is in
        // ascending split_position order  preserved by extend() so multiple
        // sequences targeting the same node keep C#'s before-node ordering.
        let mut by_block: HashMap<BlockIndex, HashMap<OperationIdx, Vec<OperationIdx>>> =
            HashMap::default();
        for (&split_position, copy_resolver) in &copy_resolvers {
            if !copy_resolver.has_copy {
                continue;
            }

            let sequence = copy_resolver.sequence(ctx);

            let (block, node) = self.operation_nodes[(split_position / INSTRUCTION_GAP) as usize];
            debug_assert!(
                node.0 != u32::MAX,
                "Split copy lands on an empty block dummy."
            );

            by_block
                .entry(block)
                .or_default()
                .entry(node)
                .or_default()
                .extend(sequence);
        }

        for (block, node_seqs) in by_block {
            let old = std::mem::take(&mut ctx.blocks.get_mut(block).ops);
            let extra: usize = node_seqs.values().map(Vec::len).sum();
            let mut new_ops = Vec::with_capacity(old.len() + extra);
            for op in old {
                if let Some(seq) = node_seqs.get(&op) {
                    new_ops.extend_from_slice(seq);
                }
                new_ops.push(op);
            }
            ctx.blocks.get_mut(block).ops = new_ops;
        }
    }

    /// C# InsertSplitCopiesAtEdges.
    fn insert_split_copies_at_edges(&mut self, ctx: &mut TranslationContext) {
        let blocks_count = ctx.blocks.order().len();

        // Reset iterators because GetSplitChild depends on cursor state.
        for interval in 0..self.intervals.len() {
            self.iv_reset(interval);
        }

        let order: Vec<BlockIndex> = ctx.blocks.order().to_vec();
        for &block in &order {
            // C# IsSplitEdgeBlock(block)  blocks added by SplitEdge below.
            if ctx.blocks.get(block).index as usize >= blocks_count {
                continue;
            }

            let has_single_or_no_successor = ctx.blocks.get(block).successor_count <= 1;

            for s in 0..ctx.blocks.get(block).successor_count as usize {
                let successor =
                    ctx.blocks.get(block).successors[s].expect("successor slot within count");

                // If the successor is a split node, the actual successor is
                // right after it.
                let succ_index = if ctx.blocks.get(successor).index as usize >= blocks_count {
                    let real = ctx.blocks.get(successor).successors[0]
                        .expect("split block has a successor");
                    ctx.blocks.get(real).index as usize
                } else {
                    ctx.blocks.get(successor).index as usize
                };

                let mut copy_resolver: Option<CopyResolver> = None;

                let live_in: Vec<usize> = self.block_live_in[succ_index].iter().collect();
                for interval_id in live_in {
                    // liveIn bits are original (parent) interval ids.
                    if interval_id >= self.parent_count {
                        continue;
                    }
                    if !self.intervals[interval_id].is_split() {
                        continue;
                    }

                    let block_index = ctx.blocks.get(block).index as usize;
                    let l_end = self.block_ranges[block_index].1 - 1;
                    let r_start = self.block_ranges[succ_index].0;

                    let left = self.iv_get_split_child(interval_id, l_end);
                    let right = self.iv_get_split_child(interval_id, r_start);

                    if let (Some(left), Some(right)) = (left, right) {
                        if left != right {
                            copy_resolver
                                .get_or_insert_with(CopyResolver::new)
                                .add_split(&self.intervals[left], &self.intervals[right]);
                        }
                    }
                }

                let Some(copy_resolver) = copy_resolver else {
                    continue;
                };
                if !copy_resolver.has_copy {
                    continue;
                }

                let sequence = copy_resolver.sequence(ctx);

                if has_single_or_no_successor {
                    for &op in &sequence {
                        append_before_terminator(ctx, block, op);
                    }
                } else if ctx.blocks.get(successor).predecessors.len() == 1 {
                    for (k, &op) in sequence.iter().enumerate() {
                        ctx.blocks.get_mut(successor).ops.insert(k, op);
                    }
                } else {
                    // Split the critical edge.
                    let split_block =
                        ControlFlowGraph::split_edge(&mut ctx.blocks, block, successor);
                    ctx.blocks.compute_predecessors();
                    for &op in &sequence {
                        append_before_terminator(ctx, split_block, op);
                    }
                }
            }
        }
    }

    // Local replacement (C# ReplaceLocalWithRegister)
    fn replace_local_with_register(&self, ctx: &mut TranslationContext, current: usize) {
        let local = self.intervals[current].local;
        let register = Register {
            index: self.intervals[current].register.index,
            reg_type: self.intervals[current].register.reg_type,
        };
        let register_op = register_operand(ctx, register, self.intervals[current].local_type);

        for ui in 0..self.intervals[current].uses.len() {
            let use_position = self.intervals[current].uses[ui];
            let (_, node) = self.operation_nodes[(use_position / INSTRUCTION_GAP) as usize];
            if node.0 == u32::MAX {
                continue;
            }

            for index in 0..ctx.ops.sources(node).len() {
                let source = ctx.ops.sources(node)[index];
                if source == local {
                    ctx.ops.set_source_raw(node, index, register_op);
                } else if ctx.operands.get(source).kind == OperandKind::Memory {
                    let data = ctx.operands.get(source);
                    let (base, idx, scale, disp, typ) = (
                        data.mem_base(),
                        data.mem_index(),
                        data.mem_scale,
                        data.mem_disp,
                        data.typ,
                    );
                    let new_base = if base == Some(local) {
                        Some(register_op)
                    } else {
                        base
                    };
                    let new_index = if idx == Some(local) {
                        Some(register_op)
                    } else {
                        idx
                    };
                    if new_base != base || new_index != idx {
                        let new_mem = ctx.operands.memory(typ, new_base, new_index, scale, disp);
                        ctx.ops.set_source_raw(node, index, new_mem);
                    }
                }
            }

            for index in 0..ctx.ops.dests(node).len() {
                if ctx.ops.dests(node)[index] == local {
                    ctx.ops.set_destination_raw(node, index, register_op);
                }
            }
        }
    }

    // NumberLocals (C#)
    fn number_locals(
        &mut self,
        cfg: &ControlFlowGraph,
        ctx: &mut TranslationContext,
        registers_count: i32,
    ) {
        self.operation_nodes.clear();
        self.intervals.clear();
        self.sorted.clear();

        for index in 0..registers_count {
            let int_id = self.intervals.len();
            self.intervals.push(LiveInterval::new_fixed(
                Register {
                    index,
                    reg_type: RegisterType::Integer,
                },
                int_id,
            ));
            let vec_id = self.intervals.len();
            self.intervals.push(LiveInterval::new_fixed(
                Register {
                    index,
                    reg_type: RegisterType::Vector,
                },
                vec_id,
            ));
        }
        // NOTE: fixed interval ids match get_register_id: (index<<1)|vec.
        self.sorted.extend(0..self.intervals.len());

        self.operations_count = 0;

        // C#: for (index = PostOrderBlocks.Length - 1; index >= 0; index--)
        //  reverse post-order.
        for pi in (0..cfg.post_order_blocks.len()).rev() {
            let block = cfg.post_order_blocks[pi];
            // Iterate by index (re-fetch each step) instead of cloning the whole
            // ops Vec per block  the body mutates ctx.operands, but each
            // ctx.blocks borrow ends before that (NLL), so no clone is needed.
            let ops_len = ctx.blocks.get(block).ops.len();

            for i in 0..ops_len {
                let node = ctx.blocks.get(block).ops[i];
                self.operation_nodes.push((block, node));

                for di in 0..ctx.ops.dests(node).len() {
                    let dest = ctx.ops.dests(node)[di];
                    let data = ctx.operands.get(dest);

                    if data.kind == OperandKind::LocalVariable && (data.value & VISITED_MASK) == 0 {
                        let typ = data.typ;
                        let interval_id = self.intervals.len();

                        ctx.operands.get_mut(dest).number_local(interval_id as u32);
                        ctx.operands.get_mut(dest).value |= VISITED_MASK;

                        let mut interval = LiveInterval::new_local(dest, typ, interval_id);
                        interval.register = Register {
                            index: 0,
                            reg_type: typ.to_register_type(),
                        };

                        // C#: copy-source preferencing for Copy /
                        // ZeroExtend32 destinations.
                        let inst = ctx.ops.get(node).instruction;
                        if inst == Instruction::Copy || inst == Instruction::ZeroExtend32 {
                            let source = ctx.ops.sources(node)[0];
                            let sdata = ctx.operands.get(source);
                            if sdata.kind == OperandKind::LocalVariable
                                // Divergence: also require the visited bit so
                                // the number is OURS (an interval index), not a
                                // leftover EmitterContext number. Affects
                                // preferencing only.
                                && (sdata.value & VISITED_MASK) != 0
                                && sdata.local_number() > 0
                                && (inst == Instruction::Copy || sdata.typ == OperandType::I32)
                            {
                                interval.copy_source = Some(sdata.local_number() as usize);
                            }
                        }

                        self.intervals.push(interval);
                        self.sorted.push(interval_id);
                    }
                }
            }

            self.operations_count += ops_len as i32 * INSTRUCTION_GAP;

            if ops_len == 0 {
                // Pretend we have a dummy instruction on the empty block.
                self.operation_nodes.push((block, OperationIdx(u32::MAX)));
                self.operations_count += INSTRUCTION_GAP;
            }
        }

        // C#: _parentIntervals = _intervals.ToArray()  snapshot boundary.
        self.parent_count = self.intervals.len();
    }

    // BuildIntervals (C#)
    fn build_intervals(
        &mut self,
        cfg: &ControlFlowGraph,
        ctx: &mut TranslationContext,
        context: &AllocationContext,
    ) {
        let blocks_count = ctx.blocks.order().len();
        self.block_ranges = vec![(0, 0); blocks_count];

        let map_size = self.intervals.len();

        // Compute local live sets  indexed by block.index (C#).
        let mut blk_live_gen: Vec<BitMap> =
            (0..blocks_count).map(|_| BitMap::new(map_size)).collect();
        let mut blk_live_kill: Vec<BitMap> =
            (0..blocks_count).map(|_| BitMap::new(map_size)).collect();

        let order: Vec<BlockIndex> = ctx.blocks.order().to_vec();
        for &block in &order {
            let block_index = ctx.blocks.get(block).index as usize;
            let mut live_gen = BitMap::new(map_size);
            let mut live_kill = BitMap::new(map_size);

            let ops_len = ctx.blocks.get(block).ops.len();
            for i in 0..ops_len {
                let node = ctx.blocks.get(block).ops[i];
                for si in 0..ctx.ops.sources(node).len() {
                    visit_source_liveness(
                        ctx,
                        ctx.ops.sources(node)[si],
                        &mut live_gen,
                        &live_kill,
                    );
                }
                for di in 0..ctx.ops.dests(node).len() {
                    let dest = ctx.ops.dests(node)[di];
                    if is_local_or_register(ctx.operands.get(dest).kind) {
                        live_kill.set(get_operand_id(ctx, dest));
                    }
                }
            }

            blk_live_gen[block_index] = live_gen;
            blk_live_kill[block_index] = live_kill;
        }

        // Compute global live sets.
        let mut blk_live_in: Vec<BitMap> =
            (0..blocks_count).map(|_| BitMap::new(map_size)).collect();
        let mut blk_live_out: Vec<BitMap> =
            (0..blocks_count).map(|_| BitMap::new(map_size)).collect();

        let mut modified = true;
        while modified {
            modified = false;

            for pi in 0..cfg.post_order_blocks.len() {
                let block = cfg.post_order_blocks[pi];
                let block_index = ctx.blocks.get(block).index as usize;

                let successors = {
                    let data = ctx.blocks.get(block);
                    let mut list = [None, None];
                    list[..data.successor_count as usize]
                        .copy_from_slice(&data.successors[..data.successor_count as usize]);
                    list
                };
                for succ in successors.into_iter().flatten() {
                    let succ_index = ctx.blocks.get(succ).index as usize;
                    modified |= blk_live_out[block_index].union(&blk_live_in[succ_index]);
                }

                // liveIn = (liveOut & ~liveKill) | liveGen
                let mut new_live_in = BitMap::new(map_size);
                new_live_in.union(&blk_live_out[block_index]);
                for bit in blk_live_kill[block_index].iter() {
                    new_live_in.clear(bit);
                }
                new_live_in.union(&blk_live_gen[block_index]);

                // C# tracks modification through liveOut only; tracking
                // liveIn changes too just adds iterations, same fixpoint.
                blk_live_in[block_index] = new_live_in;
            }
        }

        self.block_live_in = blk_live_in;
        self.block_edges.clear();

        // Compute lifetime intervals.
        let mut operation_pos = self.operations_count;

        for pi in 0..cfg.post_order_blocks.len() {
            let block = cfg.post_order_blocks[pi];
            let block_index = ctx.blocks.get(block).index as usize;
            let ops_len = ctx.blocks.get(block).ops.len();

            // Empty blocks get a dummy instruction.
            let inst_count = ops_len.max(1) as i32;

            let block_start = operation_pos - inst_count * INSTRUCTION_GAP;
            let block_end = operation_pos;

            self.block_ranges[block_index] = (block_start, block_end);
            self.block_edges.insert(block_start);

            for id in blk_live_out[block_index].iter() {
                self.iv_add_range(id, block_start, block_end);
            }

            if ops_len == 0 {
                operation_pos -= INSTRUCTION_GAP;
                continue;
            }

            for i in (0..ops_len).rev() {
                let node = ctx.blocks.get(block).ops[i];
                operation_pos -= INSTRUCTION_GAP;

                for di in 0..ctx.ops.dests(node).len() {
                    let dest = ctx.ops.dests(node)[di];
                    if is_local_or_register(ctx.operands.get(dest).kind) {
                        let id = get_operand_id(ctx, dest);
                        if self.intervals[id].is_fixed {
                            self.intervals[id].is_fixed_and_used = true;
                        }
                        self.iv_set_start(id, operation_pos + 1);
                        self.intervals[id].add_use_position(operation_pos + 1);
                    }
                }

                for si in 0..ctx.ops.sources(node).len() {
                    let source = ctx.ops.sources(node)[si];
                    self.visit_source_build(ctx, source, block_start, operation_pos);
                }

                if ctx.ops.get(node).instruction == Instruction::Call {
                    self.add_interval_caller_saved_reg(
                        context.masks.int_caller_saved,
                        operation_pos,
                        RegisterType::Integer,
                    );
                    self.add_interval_caller_saved_reg(
                        context.masks.vec_caller_saved,
                        operation_pos,
                        RegisterType::Vector,
                    );
                }
            }
        }

        for interval in 0..self.intervals.len() {
            self.iv_reset(interval);
        }
    }

    fn visit_source_build(
        &mut self,
        ctx: &TranslationContext,
        source: OperandIndex,
        block_start: i32,
        operation_pos: i32,
    ) {
        let data = ctx.operands.get(source);
        if is_local_or_register(data.kind) {
            let id = get_operand_id(ctx, source);
            self.iv_add_range(id, block_start, operation_pos + 1);
            self.intervals[id].add_use_position(operation_pos);
        } else if data.kind == OperandKind::Memory {
            if let Some(base) = data.mem_base() {
                self.visit_source_build(ctx, base, block_start, operation_pos);
            }
            if let Some(index) = data.mem_index() {
                self.visit_source_build(ctx, index, block_start, operation_pos);
            }
        }
    }

    /// C# AddIntervalCallerSavedReg.
    fn add_interval_caller_saved_reg(
        &mut self,
        mask: u32,
        operation_pos: i32,
        reg_type: RegisterType,
    ) {
        let mut mask = mask;
        while mask != 0 {
            let reg_index = mask.trailing_zeros() as i32;
            let caller_saved_reg = Register {
                index: reg_index,
                reg_type,
            };
            let id = get_register_id(caller_saved_reg);
            self.iv_add_range(id, operation_pos + 1, operation_pos + INSTRUCTION_GAP);
            mask &= !(1 << reg_index);
        }
    }
}

// Free functions (C# statics)
/// C# GetHighestValueIndex.
fn get_highest_value_index(span: &[i32]) -> usize {
    let mut highest = i32::MIN;
    let mut selected = 0usize;
    for (index, &current) in span.iter().enumerate() {
        if highest < current {
            highest = current;
            selected = index;
            if current == i32::MAX {
                break;
            }
        }
    }
    selected
}

fn visit_source_liveness(
    ctx: &TranslationContext,
    source: OperandIndex,
    live_gen: &mut BitMap,
    live_kill: &BitMap,
) {
    let data = ctx.operands.get(source);
    if is_local_or_register(data.kind) {
        let id = get_operand_id(ctx, source);
        if !live_kill.is_set(id) {
            live_gen.set(id);
        }
    } else if data.kind == OperandKind::Memory {
        if let Some(base) = data.mem_base() {
            visit_source_liveness(ctx, base, live_gen, live_kill);
        }
        if let Some(index) = data.mem_index() {
            visit_source_liveness(ctx, index, live_gen, live_kill);
        }
    }
}

/// C# GetOperandId.
fn get_operand_id(ctx: &TranslationContext, operand: OperandIndex) -> usize {
    let data = ctx.operands.get(operand);
    match data.kind {
        OperandKind::LocalVariable => data.local_number() as usize,
        OperandKind::Register => get_register_id(Register {
            index: data.reg_index() as i32,
            reg_type: data.reg_type(),
        }),
        _ => panic!("Invalid operand kind \"{:?}\".", data.kind),
    }
}

fn is_local_or_register(kind: OperandKind) -> bool {
    matches!(kind, OperandKind::LocalVariable | OperandKind::Register)
}

/// C# BasicBlock.Append  insert before the terminator.
fn append_before_terminator(ctx: &mut TranslationContext, block: BlockIndex, node: OperationIdx) {
    let insert_at = {
        let op_list = &ctx.blocks.get(block).ops;
        match op_list.last() {
            Some(&last) => match ctx.ops.get(last).instruction {
                Instruction::Return | Instruction::Tailcall | Instruction::BranchIf => {
                    op_list.len() - 1
                }
                _ => op_list.len(),
            },
            None => 0,
        }
    };
    ctx.blocks.get_mut(block).ops.insert(insert_at, node);
}
