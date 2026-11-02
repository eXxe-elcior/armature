use crate::util::hash::FxHashMap as HashMap;

use crate::decoder::instruction_name::InstructionName;
use crate::decoder::opcode_kind::OpCodeKind;
use crate::decoder::utilities::Condition;
use crate::decoder::{DecodedInstruction, RegisterSize};
use crate::ir::block::{BlockFrequency, BlockIndex};
use crate::ir::operand::OperandIndex;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Comparison, Instruction, MemoryBarrierKind, OperandType};
use crate::state::PState;

/// ARMATURE_REG_SYNC=1 → keep the synchronization rate-limiter counter in a
/// register across loop back-edges (memory-free hot spin loops) instead of
/// round-tripping ctx.counter through memory every iteration. Default off
/// (legacy memory sync) so it can be A/B'd / reverted live without a rebuild.
fn reg_sync_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ARMATURE_REG_SYNC").ok().as_deref() == Some("1"))
}

// ArmEmitterCtx
//
// Flat Rust replacement for ARMeilleure's EmitterContext + ArmEmitterContext
// class hierarchy.
//
// ## Design (derived from C# sources)
//
// The C# `EmitterContext` owns three interleaved responsibilities:
//
//   1. Operation builder    Add(inst, dest, sources) → create IR op, append
//   2. Block boundaries     auto-create blocks after terminators, wire edges
//   3. Label resolution     guest-address → IR-label-op → target-block
//
// The block-management state machine (C# private fields) is:
//
//   normal → BranchIf        → sets _ifBlock + needs_new_block
//   normal → Branch / Return → sets needs_new_block
//   needs_new → Add / MarkLabel → creates block, calls NextBlock
//   NextBlock → wires fallthrough + _ifBlock → clears needs_new_block
//
// We port this state identically. The ONLY Rust-specific change is that
// predecessor edges are computed in a post-pass rather than eagerly in
// add_successor(). This is safe because:
//
//   - Verified: zero reads of block.predecessors during emission
//     (all 21 C# accesses are in SsaConstruction, Dominance,
//      RegisterUsage, ControlFlowGraph, TailMerge, LinearScanAllocator)
//   - compute_predecessors() runs once after emission, before SSA
//
// ## Borrow-checker strategy
//
// The only operation requiring simultaneous &mut to two different blocks
// is C#'s BasicBlock.AddSuccessor (writes from.successors + to.predecessors).
// We split this:
//
//   - Emission:    BlockArena::add_successor writes only the `from` slot.
//   - Post-pass:   BlockArena::compute_predecessors backfills all predecessors.
//
// Every other method in EmitterContext touches exactly one block at a time
// + the operation/operand arenas (different fields → field-level borrow
// splitting works).
//
// ## Edge-wiring order (matching C#)
//
// When a block has both a fallthrough and a conditional-branch target,
// the successors array is [fallthrough, condition_target]  exactly the
// order C# produces. This matters for codegen (the fallthrough is at
// index 0, which is what BranchIf expects).
// ────────────────────────────────────────────────────────────────────────────

pub struct ArmEmitterCtx<'a> {
    /// The translation context holding arenas for blocks, ops, operands.
    pub ctx: &'a mut TranslationContext,

    /// The guest memory manager (C# `ArmEmitterContext.Memory`). Emitters
    /// read manager_type / address_space_bits / page_table_pointer at
    /// IR-build time; runtime access goes through NativeInterface.
    pub memory: &'a dyn crate::memory::MemoryManager,

    /// The currently decoded instruction (C# `CurrOp`).
    /// Set by the translation loop before each emit call. Stored by value
    /// (`DecodedInst` is `Copy`, ~64 bytes)  no lifetime coupling to the
    /// decoded-instruction buffer.
    curr_op: Option<DecodedInstruction>,

    /// C# EmitterContext.IsSingleStep  when true, calls/jumps return
    /// to the managed dispatch loop instead of using the function table.
    pub is_single_step: bool,

    /// Host address of TranslatorStubs.DispatchStub (C# context.Stubs).
    /// 0 when no stubs are wired (scratch contexts); EmitTableBranch
    /// asserts on it.
    pub dispatch_stub: u64,

    /// C# ArmEmitterContext.EntryAddress  the guest address of the
    /// function currently being translated.
    pub entry_address: u64,

    /// Rejit hotness counter (LCQ only; None for HighCq). The entry block and
    /// every loop back-edge bump it; at the threshold the function enqueues for
    /// HighCq promotion. Back-edge counting lets hot inner loops (entered few
    /// times) promote via iteration count, not just call count.
    pub rejit_counter: Option<*mut u32>,

    /// ARMATURE_REG_SYNC: function-local register holding the synchronization
    /// counter (the CheckSynchronization rate-limiter), so spin-loop back-edges
    /// decrement it in a register instead of round-tripping ctx.counter through
    /// memory. Lazily created at the first sync point (function entry), flushed
    /// back to ctx.counter at every store_to_context (function exit / SVC).
    sync_counter: Option<OperandIndex>,

    // Block state (matches C# private fields of EmitterContext)
    /// Current block being appended to (_irBlock).
    ir_block: BlockIndex,

    /// Deferred successor for conditional-branch true-path (_ifBlock).
    /// Set by branch_to_label(uncond:false), consumed by next_block().
    if_block: Option<BlockIndex>,

    /// If true, the next Add() call must create a new block first.
    /// Set by BranchToLabel, Return, Tailcall.
    needs_new_block: bool,

    /// Frequency hint for the next block created by new_block().
    next_block_freq: BlockFrequency,

    /// Numbered-local counter for CFG / debug output (_localsCount).
    locals_count: u32,

    /// IR label operand → target block index (_irLabels).
    ir_labels: HashMap<OperandIndex, BlockIndex>,

    // Address label mapping (matches C# ArmEmitterContext._labels)
    /// Guest virtual address → IR label operand.
    labels: HashMap<u64, OperandIndex>,

    // Flag-optimization state (matches C# ArmEmitterContext)
    /// Full decoded instruction of the last compare (C# `_optOpLastCompare`).
    /// The condition optimization in flow_helpers reads its InstName and
    /// RegSize, so the whole DecodedInst is stored, not just the kind.
    opt_op_last_compare: Option<DecodedInstruction>,

    /// Full decoded instruction of the last NZCV-setting op (C# `_optOpLastFlagSet`).
    opt_op_last_flag_set: Option<DecodedInstruction>,

    /// Cached operands for compare-and-branch optimisation.
    opt_cmp_temp_n: Option<OperandIndex>,
    opt_cmp_temp_m: Option<OperandIndex>,

    /// C# ArmEmitterContext._pendingQcFlagSync. Set by saturating SIMD
    /// emitters, consumed by sync_qc_flag / clear_qc_flag_if_modified.
    pending_qc_flag_sync: bool,
}

impl<'a> ArmEmitterCtx<'a> {
    // Construction
    pub fn new(
        ctx: &'a mut TranslationContext,
        memory: &'a dyn crate::memory::MemoryManager,
    ) -> Self {
        Self {
            ctx,
            memory,
            rejit_counter: None,
            sync_counter: None,
            curr_op: None,
            is_single_step: false,
            dispatch_stub: 0,
            entry_address: 0,
            ir_block: BlockIndex(u32::MAX), // sentinel: no block yet
            if_block: None,
            needs_new_block: true, // first Add() creates block 0
            next_block_freq: BlockFrequency::Default,
            locals_count: 0,
            ir_labels: HashMap::default(),
            labels: HashMap::default(),
            opt_op_last_compare: None,
            opt_op_last_flag_set: None,
            opt_cmp_temp_n: None,
            opt_cmp_temp_m: None,
            pending_qc_flag_sync: false,
        }
    }

    /// Set the currently decoded instruction (C# `CurrOp = opCode`).
    /// Called by the translation loop before each emitter invocation.
    pub fn set_curr_op(&mut self, decoded: DecodedInstruction) {
        self.curr_op = Some(decoded);
    }

    /// The current IR block being emitted into.
    pub fn current_block(&self) -> BlockIndex {
        self.ir_block
    }

    /// Read the currently decoded instruction.
    /// Panics if called before `set_curr_op`  a JIT invariant violation.
    pub fn curr_op(&self) -> &DecodedInstruction {
        self.curr_op
            .as_ref()
            .expect("curr_op read before set_curr_op")
    }

    /// Clear per-block optimization state. Called by the translation loop
    /// when it advances to a new decoded block (C# `CurrBlock` setter →
    /// `ResetBlockState`).
    pub fn reset_block_state(&mut self) {
        self.opt_op_last_compare = None;
        self.opt_op_last_flag_set = None;
    }

    // Core operation emission (the private Add() in C#)
    /// Internal: create an operation and append it to the current block.
    ///
    /// Mirrors C# `EmitterContext.Add` exactly:
    ///   1. `NewNextBlockIfNeeded()`           → ensure_block()
    ///   2. `Operation.Factory.Operation(...)` → ctx.make_op() (wires def-use)
    ///   3. `_irBlock.Operations.AddLast(op)`  → blocks.append_op()
    ///
    /// Returns dest (or None if no dest).
    fn emit_raw(
        &mut self,
        inst: Instruction,
        dest: Option<OperandIndex>,
        sources: &[OperandIndex],
    ) -> Option<OperandIndex> {
        self.ensure_block();
        let dsts = match dest {
            Some(d) => &[d][..],
            None => &[],
        };
        let op_idx = self.ctx.make_op(inst, dsts, sources);
        self.ctx.blocks.append_op(self.ir_block, op_idx);
        dest
    }

    // Block management (1:1 with C# EmitterContext private methods)
    /// If needs_new_block, create one. (C# NewNextBlockIfNeeded)
    fn ensure_block(&mut self) {
        if self.needs_new_block {
            self.new_block();
        }
    }

    /// Allocate a new block, add it to the emission order, and switch to it.
    /// (C# NewNextBlock: `new BasicBlock(_irBlocks.Count)` + `AddLast`)
    fn new_block(&mut self) {
        let block = self.ctx.blocks.alloc();
        self.ctx.blocks.push_order(block);
        self.next_block(block);
    }

    /// Switch to `next` as the current block.
    ///
    /// Before switching, wires edges from the previous block:
    ///
    ///   1. If the previous block has no successors and does not end
    ///      with an unconditional branch (Return / Tailcall), add a
    ///      fallthrough edge to `next`.
    ///   2. If `_ifBlock` was set by a prior BranchIf, add that as
    ///      the second successor (the conditional-branch-true target).
    ///
    /// This matches C#'s NextBlock() exactly. The ordering of
    /// add_successor calls determines the successor order:
    ///   [0] = fallthrough
    ///   [1] = condition-true
    ///
    /// IMPORTANT: we only write SUCCESSORS during emission. Predecessors
    /// are computed in a post-pass (compute_predecessors). See block.rs
    /// for the rationale.
    fn next_block(&mut self, next: BlockIndex) {
        let curr = self.ir_block;

        // Skip edge-wiring for the very first block (sentinel).
        // C#: `if (_irBlock?.SuccessorsCount == 0 && ...)`  null check.
        if curr.0 != u32::MAX {
            let needs_fallthrough = {
                let b = self.ctx.blocks.get(curr);
                b.successor_count == 0 && !self.block_ends_unconditional(curr)
            };

            if needs_fallthrough {
                self.ctx.blocks.add_successor(curr, next);

                // C# clears _ifBlock only here, when consumed. take() matches.
                if let Some(if_block) = self.if_block.take() {
                    self.ctx.blocks.add_successor(curr, if_block);
                }
            }
        }

        self.ir_block = next;
        // C#: `_irBlock.Frequency = _nextBlockFreq;`
        self.ctx.blocks.get_mut(next).frequency = self.next_block_freq;

        self.needs_new_block = false;
        self.next_block_freq = BlockFrequency::Default;
    }

    /// Branch-to-label logic shared by Branch and BranchIf.
    ///
    /// If the label already has an entry in ir_labels, reuse it.
    /// Otherwise, eagerly allocate a new block (matching C#
    /// BranchToLabel's `new BasicBlock()`).
    ///
    /// unconditional = true  → wire edge immediately
    /// unconditional = false → defer to _ifBlock for next_block()
    fn branch_to_label(
        &mut self,
        label: OperandIndex,
        unconditional: bool,
        next_freq: BlockFrequency,
    ) {
        let target = *self
            .ir_labels
            .entry(label)
            .or_insert_with(|| self.ctx.blocks.alloc());

        if unconditional {
            self.ctx.blocks.add_successor(self.ir_block, target);
        } else {
            self.if_block = Some(target);
        }

        self.needs_new_block = true;
        self.next_block_freq = next_freq;
    }

    /// Check if a block ends with Return or Tailcall.
    ///
    /// Matches C# private static EndsWithUnconditional(BasicBlock).
    /// Only Return/Tailcall are checked  BranchIf has an implicit
    /// fallthrough and does NOT block the fallthrough edge.
    fn block_ends_unconditional(&self, block: BlockIndex) -> bool {
        let b = self.ctx.blocks.get(block);
        if let Some(&last) = b.ops.last() {
            let inst = self.ctx.ops.get(last).instruction;
            matches!(inst, Instruction::Return | Instruction::Tailcall)
        } else {
            false
        }
    }

    // Public API for emitters
    // --- Branches ---

    /// Unconditional branch to label. (Branch)
    pub fn branch(&mut self, label: OperandIndex) {
        self.ensure_block();
        self.branch_to_label(label, true, BlockFrequency::Default);
    }

    /// Conditional branch to label. (BranchIf)
    pub fn branch_if(
        &mut self,
        label: OperandIndex,
        op1: OperandIndex,
        op2: OperandIndex,
        comp: Comparison,
        false_freq: BlockFrequency,
    ) {
        self.ensure_block();
        let comp_op = self.ctx.operands.constant_i32(comp as i32);
        self.emit_raw(Instruction::BranchIf, None, &[op1, op2, comp_op]);
        self.branch_to_label(label, false, false_freq);
    }

    /// Branch if operand == 0. (BranchIfFalse)
    pub fn branch_if_false(&mut self, label: OperandIndex, op: OperandIndex) {
        let zero = self.ctx.operands.const_zero(self.ctx.operands.get(op).typ);
        self.branch_if(label, op, zero, Comparison::Equal, BlockFrequency::Default);
    }

    /// Branch if operand != 0. (BranchIfTrue)
    pub fn branch_if_true(&mut self, label: OperandIndex, op: OperandIndex) {
        let zero = self.ctx.operands.const_zero(self.ctx.operands.get(op).typ);
        self.branch_if(
            label,
            op,
            zero,
            Comparison::NotEqual,
            BlockFrequency::Default,
        );
    }

    // --- Returns ---

    /// Return with no value.
    pub fn return_op(&mut self) {
        self.ensure_block();
        self.emit_raw(Instruction::Return, None, &[]);
        self.needs_new_block = true;
    }

    /// Return with a value.
    pub fn return_with(&mut self, op: OperandIndex) {
        self.ensure_block();
        self.emit_raw(Instruction::Return, None, &[op]);
        self.needs_new_block = true;
    }

    /// Void return (C# EmitterContext.Return() with no value). Used by the
    /// unmanaged dispatch loop, whose return type is None.
    pub fn return_void(&mut self) {
        self.ensure_block();
        self.emit_raw(Instruction::Return, None, &[]);
        self.needs_new_block = true;
    }

    // --- Calls ---

    /// Call a registered native helper. (C# ArmEmitterContext.Call(MethodInfo):
    /// Const(funcPtr, Symbol(DelegateTable, index)) + Call emission.)
    /// Returns the result operand, or None for void helpers.
    pub fn call(
        &mut self,
        func: crate::instructions::native_function::NativeFn,
        args: &[OperandIndex],
    ) -> Option<OperandIndex> {
        // C# ArmEmitterContext.Call(MethodInfo) calls SyncQcFlag() first.
        self.sync_qc_flag();
        let symbol = crate::ir::types::Symbol {
            typ: crate::ir::types::SymbolType::DelegateTable,
            value: func.index(),
        };
        let address = self
            .ctx
            .operands
            .constant_i64_with_symbol(func.ptr() as i64, symbol);
        self.call_address(address, func.ret_type(), args)
    }

    /// Call an arbitrary address. (C# EmitterContext.Call(Operand, OperandType,
    /// params): sources are [address, args...]; dest = Local(retType) unless
    /// retType is None.)
    pub fn call_address(
        &mut self,
        address: OperandIndex,
        return_type: OperandType,
        args: &[OperandIndex],
    ) -> Option<OperandIndex> {
        let mut sources = Vec::with_capacity(args.len() + 1);
        sources.push(address);
        sources.extend_from_slice(args);

        if return_type != OperandType::None {
            let dest = self.allocate_local(return_type);
            self.emit_raw(Instruction::Call, Some(dest), &sources);
            Some(dest)
        } else {
            self.emit_raw(Instruction::Call, None, &sources);
            None
        }
    }

    /// Tail call. (C# EmitterContext.Tailcall: [address, args...], no dest,
    /// then _needsNewBlock = true.)
    pub fn tailcall(&mut self, address: OperandIndex, args: &[OperandIndex]) {
        let mut sources = Vec::with_capacity(args.len() + 1);
        sources.push(address);
        sources.extend_from_slice(args);

        self.emit_raw(Instruction::Tailcall, None, &sources);
        self.needs_new_block = true;
    }

    /// C# EmitterContext.MemoryBarrier.
    pub fn memory_barrier(&mut self, kind: MemoryBarrierKind) {
        let inst = match kind {
            MemoryBarrierKind::Full => Instruction::MemoryBarrier,
            MemoryBarrierKind::StoreStore => Instruction::MemoryBarrierStore,
            MemoryBarrierKind::LoadLoad => Instruction::MemoryBarrierLoad,
        };
        self.emit_raw(inst, None, &[]);
    }

    /// C# EmitterContext.GetControlFlowGraph():
    /// `new ControlFlowGraph(_irBlocks.First, _irBlocks, _localsCount)`.
    pub fn get_control_flow_graph(&mut self) -> crate::translation::cfg::ControlFlowGraph {
        let entry = *self
            .ctx
            .blocks
            .order()
            .first()
            .expect("GetControlFlowGraph on empty emission order");
        crate::translation::cfg::ControlFlowGraph::new(
            &mut self.ctx.blocks,
            entry,
            self.locals_count,
        )
    }

    // --- Context transfer ---

    /// C# EmitterContext.LoadFromContext: forces a new block FIRST, then
    /// emits (order is the reverse of StoreToContext  keep it).
    pub fn load_from_context(&mut self) {
        self.needs_new_block = true;
        self.emit_raw(Instruction::LoadFromContext, None, &[]);
    }

    /// C# EmitterContext.StoreToContext: emits, THEN forces a new block.
    pub fn store_to_context(&mut self) {
        self.flush_sync_counter();
        self.emit_raw(Instruction::StoreToContext, None, &[]);
        self.needs_new_block = true;
    }

    // --- QC flag sync (C# ArmEmitterContext) ---
    //
    // Divergence: the C# bodies emit FPSR intrinsics only under
    // Optimizations.UseAdvSimd (ARM64 host). Host is x86-64 only, so those
    // emission paths are excluded; the _pendingQcFlagSync bookkeeping is
    // kept so SIMD emitters and Call ordering match C# exactly.

    /// C# SetPendingQcFlagSync.
    pub fn set_pending_qc_flag_sync(&mut self) {
        self.pending_qc_flag_sync = true;
    }

    /// C# SyncQcFlag (AdvSimd emission excluded  flag bookkeeping only).
    pub fn sync_qc_flag(&mut self) {
        self.pending_qc_flag_sync = false;
    }

    /// C# ClearQcFlag (AdvSimd-only emission  no-op on x86-64).
    pub fn clear_qc_flag(&mut self) {}

    /// Bump the rejit hotness counter and, on first crossing the threshold,
    /// enqueue this function for HighCq promotion. No-op for HighCq (counter
    /// is None). Emitted at the entry block and at loop back-edges.
    pub fn emit_rejit_count(&mut self) {
        use crate::ir::block::BlockFrequency;
        use crate::ir::types::{Comparison, OperandType};

        let Some(counter) = self.rejit_counter else {
            return;
        };

        let lbl_end = self.ctx.operands.label();
        let address = self.ctx.operands.constant_i64(counter as i64);
        let cur_count = self.load(OperandType::I32, address);
        let one = self.ctx.operands.constant_i32(1);
        let count = self.add(cur_count, one);
        self.store(address, count);

        let threshold = self.ctx.operands.constant_i32(rejit_threshold());
        self.branch_if(
            lbl_end,
            cur_count,
            threshold,
            Comparison::NotEqual,
            BlockFrequency::Cold,
        );

        let entry = self.ctx.operands.constant_i64(self.entry_address as i64);
        self.call(
            crate::instructions::native_function::NativeFn::EnqueueForRejit,
            &[entry],
        );

        self.mark_label(lbl_end, BlockFrequency::Default);
    }

    /// On-stack replacement (OSR) exit for never-returning hot loops.
    ///
    /// Tiering promotes a function to HighCq when its rejit counter crosses
    /// MINS_CALL_FOR_REJIT, but a thread trapped in an LCQ loop that never
    /// returns to the dispatcher never picks up the published HighCq version.
    /// Distinguishing feature: a normal hot function is replaced by HighCq at
    /// dispatch and its LCQ counter freezes near the promotion threshold
    /// only a *stuck* LCQ loop keeps incrementing the counter unbounded. So
    /// once the counter climbs far past promotion (OSR threshold ≫ 30), bounce
    /// through the dispatch stub at the loop header: emit_virtual_jump issues a
    /// StoreToContext (flushing all guest state to the CpuState) then tailcalls
    /// the dispatcher, which re-enters via the FunctionTable, running the
    /// header-entry function (itself LCQ→HighCq tiering). State lives in the
    /// CpuState across the bounce, so no register-state mapping is needed.
    ///
    /// No C# counterpart  ARMeilleure has no OSR; this is a deliberate
    /// divergence. No-op for HighCq (counter is None), below threshold, or
    /// when disabled via ARMATURE_NO_OSR.
    pub fn emit_osr_exit(&mut self, target: u64) {
        use crate::ir::block::BlockFrequency;
        use crate::ir::types::{Comparison, OperandType};

        let Some(counter) = self.rejit_counter else {
            return;
        };
        let threshold_val = osr_threshold();
        if threshold_val == 0 {
            return; // disabled
        }

        let lbl_continue = self.ctx.operands.label();
        let addr = self.ctx.operands.constant_i64(counter as i64);
        let cur = self.load(OperandType::I32, addr);
        let threshold = self.ctx.operands.constant_i32(threshold_val);
        // Below threshold: stay in the inline LCQ loop (fast path).
        self.branch_if(
            lbl_continue,
            cur,
            threshold,
            Comparison::Less,
            BlockFrequency::Cold,
        );

        // Stuck/hot: bounce to the dispatcher at the loop header.
        let header = self.ctx.operands.constant_i64(target as i64);
        crate::instructions::flow_helpers::emit_virtual_jump(self, header, false);

        self.mark_label(lbl_continue, BlockFrequency::Default);
    }

    /// Condition-aware OSR for a *conditional* loop back-edge. Same idea as
    /// emit_osr_exit, but the bounce fires only when the branch would actually
    /// be taken  otherwise loop exit (the not-taken fall-through to post-loop
    /// code) would be hijacked into an infinite re-entry. `predicate` is the
    /// branch's taken value; `taken_when_nonzero` selects whether nonzero (CBNZ,
    /// B.cond-true) or zero (CBZ) means taken.
    pub fn emit_osr_taken(
        &mut self,
        target: u64,
        predicate: OperandIndex,
        taken_when_nonzero: bool,
    ) {
        use crate::ir::block::BlockFrequency;
        use crate::ir::types::{Comparison, OperandType};

        let Some(counter) = self.rejit_counter else {
            return;
        };
        let threshold_val = osr_threshold();
        if threshold_val == 0 {
            return;
        }

        let lbl_continue = self.ctx.operands.label();
        let addr = self.ctx.operands.constant_i64(counter as i64);
        let cur = self.load(OperandType::I32, addr);
        let threshold = self.ctx.operands.constant_i32(threshold_val);
        // Not hot enough → stay in the inline loop.
        self.branch_if(
            lbl_continue,
            cur,
            threshold,
            Comparison::Less,
            BlockFrequency::Cold,
        );

        // Hot, but skip the bounce when the branch would NOT be taken, so loop
        // exit still falls through to the post-loop code.
        let pred_ty = self.ctx.operands.get(predicate).typ;
        let zero = self.ctx.operands.const_zero(pred_ty);
        let skip_cmp = if taken_when_nonzero {
            Comparison::Equal
        } else {
            Comparison::NotEqual
        };
        self.branch_if(
            lbl_continue,
            predicate,
            zero,
            skip_cmp,
            BlockFrequency::Cold,
        );

        // Hot and the back-edge is taken: bounce through the dispatcher.
        let header = self.ctx.operands.constant_i64(target as i64);
        crate::instructions::flow_helpers::emit_virtual_jump(self, header, false);

        self.mark_label(lbl_continue, BlockFrequency::Default);
    }

    /// C# ClearQcFlagIfModified (AdvSimd-only emission  no-op on x86-64).
    pub fn clear_qc_flag_if_modified(&mut self) {}

    /// C# ArmEmitterContext.EnterArmFpMode / UpdateArmFpMode (same body).
    pub fn enter_arm_fp_mode(&mut self) {
        crate::instructions::simd_helper::enter_arm_fp_mode(self);
    }

    pub fn update_arm_fp_mode(&mut self) {
        self.enter_arm_fp_mode();
    }

    /// C# ArmEmitterContext.ExitArmFpMode.
    pub fn exit_arm_fp_mode(&mut self) {
        crate::instructions::simd_helper::exit_arm_fp_mode(self);
    }

    // --- Synchronization ---

    /// Port of Translator.EmitSynchronization (Translation/Translator.cs:499).
    pub fn emit_synchronization(&mut self) {
        if reg_sync_enabled() {
            self.emit_synchronization_reg();
        } else {
            self.emit_synchronization_mem();
        }
    }

    /// ARMATURE_REG_SYNC path: the CheckSynchronization rate-limiter lives in a
    /// function-local register. Each sync point decrements the register; only
    /// when it reaches zero (every MIN_COUNT_FOR_CHECK back-edges) do we touch
    /// memory + call CheckSynchronization. CheckSynchronization sees running==0
    /// on StopRunning, so stop/interrupts are observed within MIN iterations
    /// (the cadence already used for interrupt checks). The register is flushed
    /// to ctx.counter at every store_to_context so the cross-function cadence
    /// (and StopRunning's counter=0 seen at the next entry) is preserved.
    fn emit_synchronization_reg(&mut self) {
        let count_offs = crate::registers::CpuState::counter_offset() as i64;

        // Lazily create + prime the local from ctx.counter at the first sync
        // point (function entry, always emitted before any back-edge sync).
        let local = match self.sync_counter {
            Some(l) => l,
            None => {
                let l = self.allocate_local(OperandType::I32);
                let nc = self.load_argument(OperandType::I64, 0);
                let offs = self.ctx.operands.constant_i64(count_offs);
                let addr = self.add(nc, offs);
                let c = self.load(OperandType::I32, addr);
                self.copy_to(l, c);
                self.sync_counter = Some(l);
                l
            }
        };

        let lbl_running = self.ctx.operands.label();
        let lbl_exit = self.ctx.operands.label();

        // local -= 1  (pure register decrement  the hot path)
        let one = self.ctx.operands.constant_i32(1);
        let dec = self.subtract(local, one);
        self.copy_to(local, dec);
        let zero = self.ctx.operands.const_zero(OperandType::I32);
        // Common case (local > 0): fall straight through to exit. Must be a
        // SIGNED `> 0`, not `!= 0`: an external StopRunning sets ctx.counter=0,
        // so a function entered after the stop primes `local` to 0, then this
        // decrement makes it -1. With `!= 0` that runs away into the negatives
        // (CheckSynchronization only fires at exactly 0, ~2^32 iters later), so
        // running==0 is never observed → the dispatch loop never exits →
        // teardown frees state under a live JIT thread → crash. `> 0` makes any
        // value <= 0 take the check path immediately.
        self.branch_if(
            lbl_exit,
            local,
            zero,
            Comparison::Greater,
            BlockFrequency::Default,
        );

        // Rate-limiter expired: do the real check (cold, ~1/MIN iterations).
        let running = self
            .call(
                crate::instructions::native_function::NativeFn::CheckSynchronization,
                &[],
            )
            .expect("CheckSynchronization returns I32");
        self.branch_if(
            lbl_running,
            running,
            zero,
            Comparison::NotEqual,
            BlockFrequency::Cold,
        );
        let zero64 = self.ctx.operands.constant_i64(0);
        self.return_with(zero64);

        // Still running: reload the budget (CheckSynchronization reset
        // ctx.counter to MIN_COUNT_FOR_CHECK).
        self.mark_label(lbl_running, BlockFrequency::Default);
        let nc = self.load_argument(OperandType::I64, 0);
        let offs = self.ctx.operands.constant_i64(count_offs);
        let addr = self.add(nc, offs);
        let reloaded = self.load(OperandType::I32, addr);
        self.copy_to(local, reloaded);

        self.mark_label(lbl_exit, BlockFrequency::Default);
    }

    /// Flush the register-resident sync counter back to ctx.counter. Called
    /// before every StoreToContext (function exit / SVC / exception) so memory
    /// reflects the consumed budget and the next function continues the cadence.
    fn flush_sync_counter(&mut self) {
        if !reg_sync_enabled() {
            return;
        }
        let Some(local) = self.sync_counter else {
            return;
        };
        let count_offs = crate::registers::CpuState::counter_offset() as i64;
        let nc = self.load_argument(OperandType::I64, 0);
        let offs = self.ctx.operands.constant_i64(count_offs);
        let addr = self.add(nc, offs);
        self.store(addr, local);
    }

    fn emit_synchronization_mem(&mut self) {
        let count_offs = crate::registers::CpuState::counter_offset() as i64;

        let lbl_non_zero = self.ctx.operands.label();
        let lbl_exit = self.ctx.operands.label();

        let native_ctx = self.load_argument(OperandType::I64, 0);
        let offs = self.ctx.operands.constant_i64(count_offs);
        let count_addr = self.add(native_ctx, offs);
        let count = self.load(OperandType::I32, count_addr);
        // C#: BranchIfTrue(lblNonZero, count, BasicBlockFrequency.Cold)
        let zero = self.ctx.operands.const_zero(OperandType::I32);
        self.branch_if(
            lbl_non_zero,
            count,
            zero,
            Comparison::NotEqual,
            BlockFrequency::Cold,
        );

        let running = self
            .call(
                crate::instructions::native_function::NativeFn::CheckSynchronization,
                &[],
            )
            .expect("CheckSynchronization returns I32");
        let zero = self.ctx.operands.const_zero(OperandType::I32);
        self.branch_if(
            lbl_exit,
            running,
            zero,
            Comparison::NotEqual,
            BlockFrequency::Cold,
        );

        let zero64 = self.ctx.operands.constant_i64(0);
        self.return_with(zero64);

        self.mark_label(lbl_non_zero, BlockFrequency::Default);
        let one = self.ctx.operands.constant_i32(1);
        let count = self.subtract(count, one);
        self.store(count_addr, count);

        self.mark_label(lbl_exit, BlockFrequency::Default);
    }

    // --- Labels ---

    /// Get or create a label operand for a guest address. (GetLabel)
    pub fn get_label(&mut self, address: u64) -> OperandIndex {
        *self
            .labels
            .entry(address)
            .or_insert_with(|| self.ctx.operands.label())
    }

    /// Mark the current position as the target of a label. (MarkLabel)
    pub fn mark_label(&mut self, label: OperandIndex, freq: BlockFrequency) {
        self.next_block_freq = freq;

        if let Some(&target) = self.ir_labels.get(&label) {
            // Label already has a block (previously created by BranchToLabel).
            // It enters the emission order NOW, not at allocation time.
            // C#: `nextBlock.Index = _irBlocks.Count; _irBlocks.AddLast(nextBlock);`
            self.ctx.blocks.push_order(target);
            self.next_block(target);
        } else {
            // First occurrence  create the block now.
            self.new_block();
            self.ir_labels.insert(label, self.ir_block);
        }
    }

    // --- Allocate local variable ---

    /// Allocate a numbered local variable. (AllocateLocal)
    pub fn allocate_local(&mut self, typ: OperandType) -> OperandIndex {
        self.locals_count += 1;
        let local = self.ctx.temp(typ);
        self.ctx
            .operands
            .get_mut(local)
            .number_local(self.locals_count);
        local
    }

    // Convenience: unary/binary arithmetic (returning new local)
    //
    // These match C# EmitterContext's convenience methods exactly.
    // Each creates a local, emits the operation, returns the local.

    fn unary(&mut self, inst: Instruction, op: OperandIndex) -> OperandIndex {
        let typ = self.ctx.operands.get(op).typ;
        let dest = self.allocate_local(typ);
        self.emit_raw(inst, Some(dest), &[op]);
        dest
    }

    fn binary(&mut self, inst: Instruction, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        let typ = self.ctx.operands.get(op1).typ;
        let dest = self.allocate_local(typ);
        self.emit_raw(inst, Some(dest), &[op1, op2]);
        dest
    }

    pub fn add(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::Add, op1, op2)
    }
    pub fn subtract(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::Subtract, op1, op2)
    }
    pub fn multiply(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::Multiply, op1, op2)
    }
    pub fn divide(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::Divide, op1, op2)
    }
    pub fn divide_ui(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::DivideUI, op1, op2)
    }
    pub fn bitwise_and(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::BitwiseAnd, op1, op2)
    }
    /// (~op1) & op2  lowered to a single x86 ANDN. Only emit when BMI1 is
    /// available (`crate::optimizations::use_bmi1()`).
    pub fn bitwise_and_not(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::BitwiseAndNot, op1, op2)
    }
    pub fn bitwise_or(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::BitwiseOr, op1, op2)
    }
    pub fn bitwise_xor(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::BitwiseExclusiveOr, op1, op2)
    }
    pub fn bitwise_not(&mut self, op: OperandIndex) -> OperandIndex {
        self.unary(Instruction::BitwiseNot, op)
    }
    pub fn negate(&mut self, op: OperandIndex) -> OperandIndex {
        self.unary(Instruction::Negate, op)
    }
    pub fn copy(&mut self, op: OperandIndex) -> OperandIndex {
        self.unary(Instruction::Copy, op)
    }

    /// Copy to a specific destination operand. (C# Copy(dest, op1) overload)
    pub fn copy_to(&mut self, dest: OperandIndex, src: OperandIndex) {
        self.emit_raw(Instruction::Copy, Some(dest), &[src]);
    }

    // --- Shifts ---

    pub fn shift_left(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::ShiftLeft, op1, op2)
    }
    pub fn shift_right_si(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::ShiftRightSI, op1, op2)
    }
    pub fn shift_right_ui(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::ShiftRightUI, op1, op2)
    }
    pub fn rotate_right(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.binary(Instruction::RotateRight, op1, op2)
    }

    // --- Extensions ---

    pub fn sign_extend8(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::SignExtend8, Some(dest), &[op]);
        dest
    }
    pub fn sign_extend16(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::SignExtend16, Some(dest), &[op]);
        dest
    }
    pub fn sign_extend32(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::SignExtend32, Some(dest), &[op]);
        dest
    }
    pub fn zero_extend8(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::ZeroExtend8, Some(dest), &[op]);
        dest
    }
    pub fn zero_extend16(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::ZeroExtend16, Some(dest), &[op]);
        dest
    }
    pub fn zero_extend32(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::ZeroExtend32, Some(dest), &[op]);
        dest
    }

    // --- Conversions ---

    pub fn convert_i64_to_i32(&mut self, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        self.emit_raw(Instruction::ConvertI64ToI32, Some(dest), &[op]);
        dest
    }
    pub fn convert_to_fp(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::ConvertToFP, Some(dest), &[op]);
        dest
    }
    pub fn convert_to_fpui(&mut self, typ: OperandType, op: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::ConvertToFPUI, Some(dest), &[op]);
        dest
    }

    // --- Comparison helpers (C# ICompare family) ---

    pub fn icompare(
        &mut self,
        op1: OperandIndex,
        op2: OperandIndex,
        comp: Comparison,
    ) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        let comp_op = self.ctx.operands.constant_i32(comp as i32);
        self.emit_raw(Instruction::Compare, Some(dest), &[op1, op2, comp_op]);
        dest
    }
    pub fn icompare_equal(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.icompare(op1, op2, Comparison::Equal)
    }
    pub fn icompare_not_equal(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.icompare(op1, op2, Comparison::NotEqual)
    }
    pub fn icompare_greater(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.icompare(op1, op2, Comparison::Greater)
    }
    pub fn icompare_less(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.icompare(op1, op2, Comparison::Less)
    }
    pub fn icompare_greater_or_equal(
        &mut self,
        op1: OperandIndex,
        op2: OperandIndex,
    ) -> OperandIndex {
        self.icompare(op1, op2, Comparison::GreaterOrEqual)
    }
    pub fn icompare_less_or_equal(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.icompare(op1, op2, Comparison::LessOrEqual)
    }
    pub fn icompare_greater_ui(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.icompare(op1, op2, Comparison::GreaterUI)
    }
    pub fn icompare_less_ui(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        self.icompare(op1, op2, Comparison::LessUI)
    }
    pub fn icompare_greater_or_equal_ui(
        &mut self,
        op1: OperandIndex,
        op2: OperandIndex,
    ) -> OperandIndex {
        self.icompare(op1, op2, Comparison::GreaterOrEqualUI)
    }
    pub fn icompare_less_or_equal_ui(
        &mut self,
        op1: OperandIndex,
        op2: OperandIndex,
    ) -> OperandIndex {
        self.icompare(op1, op2, Comparison::LessOrEqualUI)
    }

    // --- Memory ---

    pub fn load(&mut self, typ: OperandType, address: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::Load, Some(dest), &[address]);
        dest
    }
    pub fn load8(&mut self, address: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        self.emit_raw(Instruction::Load8, Some(dest), &[address]);
        dest
    }
    pub fn load16(&mut self, address: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        self.emit_raw(Instruction::Load16, Some(dest), &[address]);
        dest
    }

    pub fn store(&mut self, address: OperandIndex, value: OperandIndex) {
        self.emit_raw(Instruction::Store, None, &[address, value]);
    }
    pub fn store8(&mut self, address: OperandIndex, value: OperandIndex) {
        self.emit_raw(Instruction::Store8, None, &[address, value]);
    }
    pub fn store16(&mut self, address: OperandIndex, value: OperandIndex) {
        self.emit_raw(Instruction::Store16, None, &[address, value]);
    }

    // --- Misc ---

    pub fn byte_swap(&mut self, op: OperandIndex) -> OperandIndex {
        self.unary(Instruction::ByteSwap, op)
    }
    pub fn count_leading_zeros(&mut self, op: OperandIndex) -> OperandIndex {
        self.unary(Instruction::CountLeadingZeros, op)
    }

    // --- Intrinsic helpers ---

    /// Emit an intrinsic operation. Matches C# context.AddIntrinsic(intrinsic, args...).
    pub fn emit_intrinsic(
        &mut self,
        inst: crate::ir::types::Intrinsic,
        args: &[OperandIndex],
    ) -> OperandIndex {
        self.ensure_block();
        let dest = self.allocate_local(OperandType::V128);
        let idx = self.ctx.make_intrinsic_op(inst, &[dest], args);
        self.ctx.blocks.append_op(self.ir_block, idx);
        dest
    }

    /// Emit an intrinsic with no dest (void). Matches C# context.AddIntrinsicNoRet.
    pub fn emit_intrinsic_no_ret(
        &mut self,
        inst: crate::ir::types::Intrinsic,
        args: &[OperandIndex],
    ) {
        self.ensure_block();
        let idx = self.ctx.make_intrinsic_op(inst, &[], args);
        self.ctx.blocks.append_op(self.ir_block, idx);
    }

    /// C# AddIntrinsicInt  I32 destination.
    pub fn emit_intrinsic_int(
        &mut self,
        inst: crate::ir::types::Intrinsic,
        args: &[OperandIndex],
    ) -> OperandIndex {
        self.ensure_block();
        let dest = self.allocate_local(OperandType::I32);
        let idx = self.ctx.make_intrinsic_op(inst, &[dest], args);
        self.ctx.blocks.append_op(self.ir_block, idx);
        dest
    }

    /// C# AddIntrinsicLong  I64 destination.
    pub fn emit_intrinsic_long(
        &mut self,
        inst: crate::ir::types::Intrinsic,
        args: &[OperandIndex],
    ) -> OperandIndex {
        self.ensure_block();
        let dest = self.allocate_local(OperandType::I64);
        let idx = self.ctx.make_intrinsic_op(inst, &[dest], args);
        self.ctx.blocks.append_op(self.ir_block, idx);
        dest
    }

    // --- Vector helpers ---

    pub fn vector_create_scalar(&mut self, value: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        self.emit_raw(Instruction::VectorCreateScalar, Some(dest), &[value]);
        dest
    }
    pub fn vector_extract(
        &mut self,
        typ: OperandType,
        vector: OperandIndex,
        index: i32,
    ) -> OperandIndex {
        let dest = self.allocate_local(typ);
        let idx = self.ctx.operands.constant_i32(index);
        self.emit_raw(Instruction::VectorExtract, Some(dest), &[vector, idx]);
        dest
    }
    pub fn vector_extract8(&mut self, vector: OperandIndex, index: i32) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        let idx = self.ctx.operands.constant_i32(index);
        self.emit_raw(Instruction::VectorExtract8, Some(dest), &[vector, idx]);
        dest
    }
    pub fn vector_extract16(&mut self, vector: OperandIndex, index: i32) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        let idx = self.ctx.operands.constant_i32(index);
        self.emit_raw(Instruction::VectorExtract16, Some(dest), &[vector, idx]);
        dest
    }
    pub fn vector_insert(
        &mut self,
        vector: OperandIndex,
        value: OperandIndex,
        index: i32,
    ) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        let idx = self.ctx.operands.constant_i32(index);
        self.emit_raw(Instruction::VectorInsert, Some(dest), &[vector, value, idx]);
        dest
    }
    pub fn vector_insert8(
        &mut self,
        vector: OperandIndex,
        value: OperandIndex,
        index: i32,
    ) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        let idx = self.ctx.operands.constant_i32(index);
        self.emit_raw(
            Instruction::VectorInsert8,
            Some(dest),
            &[vector, value, idx],
        );
        dest
    }
    pub fn vector_insert16(
        &mut self,
        vector: OperandIndex,
        value: OperandIndex,
        index: i32,
    ) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        let idx = self.ctx.operands.constant_i32(index);
        self.emit_raw(
            Instruction::VectorInsert16,
            Some(dest),
            &[vector, value, idx],
        );
        dest
    }
    pub fn vector_zero(&mut self) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        self.emit_raw(Instruction::VectorZero, Some(dest), &[]);
        dest
    }
    pub fn vector_one(&mut self) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        self.emit_raw(Instruction::VectorOne, Some(dest), &[]);
        dest
    }
    pub fn vector_zero_upper64(&mut self, vector: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        self.emit_raw(Instruction::VectorZeroUpper64, Some(dest), &[vector]);
        dest
    }
    pub fn vector_zero_upper96(&mut self, vector: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::V128);
        self.emit_raw(Instruction::VectorZeroUpper96, Some(dest), &[vector]);
        dest
    }

    // --- Compare-and-swap ---

    pub fn compare_and_swap(
        &mut self,
        address: OperandIndex,
        expected: OperandIndex,
        desired: OperandIndex,
    ) -> OperandIndex {
        let dest = self.allocate_local(self.ctx.operands.get(desired).typ);
        self.emit_raw(
            Instruction::CompareAndSwap,
            Some(dest),
            &[address, expected, desired],
        );
        dest
    }
    pub fn compare_and_swap16(
        &mut self,
        address: OperandIndex,
        expected: OperandIndex,
        desired: OperandIndex,
    ) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        self.emit_raw(
            Instruction::CompareAndSwap16,
            Some(dest),
            &[address, expected, desired],
        );
        dest
    }
    pub fn compare_and_swap8(
        &mut self,
        address: OperandIndex,
        expected: OperandIndex,
        desired: OperandIndex,
    ) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I32);
        self.emit_raw(
            Instruction::CompareAndSwap8,
            Some(dest),
            &[address, expected, desired],
        );
        dest
    }

    // --- Conditional select ---

    pub fn conditional_select(
        &mut self,
        op1: OperandIndex,
        op2: OperandIndex,
        op3: OperandIndex,
    ) -> OperandIndex {
        let typ = self.ctx.operands.get(op2).typ;
        let dest = self.allocate_local(typ);
        self.emit_raw(Instruction::ConditionalSelect, Some(dest), &[op1, op2, op3]);
        dest
    }

    // --- Multiply-high ---

    pub fn multiply64_high_si(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I64);
        self.emit_raw(Instruction::Multiply64HighSI, Some(dest), &[op1, op2]);
        dest
    }
    pub fn multiply64_high_ui(&mut self, op1: OperandIndex, op2: OperandIndex) -> OperandIndex {
        let dest = self.allocate_local(OperandType::I64);
        self.emit_raw(Instruction::Multiply64HighUI, Some(dest), &[op1, op2]);
        dest
    }

    // --- Load argument ---

    pub fn load_argument(&mut self, typ: OperandType, index: i32) -> OperandIndex {
        let dest = self.allocate_local(typ);
        let idx = self.ctx.operands.constant_i32(index);
        self.emit_raw(Instruction::LoadArgument, Some(dest), &[idx]);
        dest
    }

    // Flag optimization (C# ArmEmitterContext)
    /// Record the last comparison for subsequent flag optimisation.
    /// (C# MarkComparison)
    pub fn mark_comparison(&mut self, n: OperandIndex, m: OperandIndex) {
        let op = *self.curr_op();
        self.opt_op_last_compare = Some(op);
        self.opt_cmp_temp_n = Some(self.copy(n));
        self.opt_cmp_temp_m = Some(self.copy(m));
    }

    /// Reuse the operands of the last compare instead of re-deriving the
    /// condition from NZCV flags, when the last compare is also the last
    /// NZCV writer. (C# ArmEmitterContext.TryGetComparisonResult, :208)
    ///
    /// Identity check: C# compares object references
    /// (`_optOpLastCompare != _optOpLastFlagSet`). We compare guest
    /// addresses of the stored DecodedInst copies  an address uniquely
    /// identifies one decoded instruction within a translation.
    pub fn try_get_comparison_result(&mut self, condition: Condition) -> Option<OperandIndex> {
        let last_compare = self.opt_op_last_compare?;
        let last_flag_set = self.opt_op_last_flag_set?;
        if last_compare.address != last_flag_set.address {
            return None;
        }

        let n = self.opt_cmp_temp_n?;
        let m = self.opt_cmp_temp_m?;

        let cmp_name = last_compare.info.name;

        if cmp_name == InstructionName::Subs {
            return match condition {
                Condition::Eq => Some(self.icompare_equal(n, m)),
                Condition::Ne => Some(self.icompare_not_equal(n, m)),
                Condition::GeUn => Some(self.icompare_greater_or_equal_ui(n, m)),
                Condition::LtUn => Some(self.icompare_less_ui(n, m)),
                Condition::GtUn => Some(self.icompare_greater_ui(n, m)),
                Condition::LeUn => Some(self.icompare_less_or_equal_ui(n, m)),
                Condition::Ge => Some(self.icompare_greater_or_equal(n, m)),
                Condition::Lt => Some(self.icompare_less(n, m)),
                Condition::Gt => Some(self.icompare_greater(n, m)),
                Condition::Le => Some(self.icompare_less_or_equal(n, m)),
                _ => None,
            };
        } else if cmp_name == InstructionName::Adds {
            // CMN limitations (C# comment, ArmEmitterContext.cs:240):
            // - unsigned comparisons invalid (carry vs borrow differ
            //   between addition and subtraction);
            // - immediate variants only  a register operand could be
            //   exactly INT_MIN, which has no positive counterpart.
            //   Invalid values can't be encoded as immediates.
            // C# checks `is IOpCodeAluImm`; MarkComparison only fires from
            // Adds/Subs emitters, so the immediate case is AluImm here.
            if let OpCodeKind::AluImm { immediate, .. } = last_compare.kind {
                let m = if last_compare.reg_size == RegisterSize::Int32 {
                    self.ctx.operands.constant_i32((-immediate) as i32)
                } else {
                    self.ctx.operands.constant_i64(-immediate)
                };
                return match condition {
                    Condition::Eq => Some(self.icompare_equal(n, m)),
                    Condition::Ne => Some(self.icompare_not_equal(n, m)),
                    Condition::Ge => Some(self.icompare_greater_or_equal(n, m)),
                    Condition::Lt => Some(self.icompare_less(n, m)),
                    Condition::Gt => Some(self.icompare_greater(n, m)),
                    Condition::Le => Some(self.icompare_less_or_equal(n, m)),
                    _ => None,
                };
            }
        }

        None
    }

    /// Record that a flag-setting instruction was emitted.
    ///
    /// C# MarkFlagSet(PState): only NZCV writes (flag >= VFlag) record the
    /// op  the compare+branch optimization must not trust stale comparison
    /// values once any NZCV bit is overwritten, but non-NZCV flags (Q, GE,
    /// T, E) don't invalidate it.
    pub fn mark_flag_set(&mut self, flag: PState) {
        if (flag as u32) >= (PState::VFlag as u32) {
            let op = *self.curr_op();
            self.opt_op_last_flag_set = Some(op);
        }
    }
}

/// OSR back-edge threshold (rejit-counter value past which a stuck LCQ loop
/// bounces through the dispatcher  see emit_osr_exit). Must be ≫ the rejit
/// promotion threshold (30) so normal hot loops, whose LCQ counter freezes
/// near 30 once HighCq takes over, never bounce. Env-tunable for in-game A/B;
/// ARMATURE_NO_OSR=1 (or threshold 0) disables OSR entirely.
fn osr_threshold() -> i32 {
    static OSR_THRESHOLD: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *OSR_THRESHOLD.get_or_init(|| {
        if std::env::var("ARMATURE_NO_OSR").ok().as_deref() == Some("1") {
            return 0;
        }
        std::env::var("ARMATURE_OSR_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|&v| v >= 0)
            .unwrap_or(200)
    })
}

/// LCQ→HighCq promotion threshold: bumps at the entry block + every loop
/// back-edge, promotes once the count reaches it. Default 30 (C# uses 100;
/// we promote more aggressively). Lower → more functions reach HighCq sooner
/// (lower steady-state CPU) at the cost of more background compile during
/// warmup. Env-tunable for in-game A/B via ARMATURE_REJIT_THRESHOLD. Read once.
pub(crate) fn rejit_threshold() -> i32 {
    static REJIT_THRESHOLD: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *REJIT_THRESHOLD.get_or_init(|| {
        std::env::var("ARMATURE_REJIT_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok())
            .filter(|&v| v > 0)
            // Match ARMeilleure's MinsCallForRejit (100): promote only genuinely
            // hot functions. A low threshold compiles many barely-hot functions
            // to HighCq, burning background-compile CPU for little gain.
            .unwrap_or(100)
    })
}
