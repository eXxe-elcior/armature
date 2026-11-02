// Port of Translation/TranslatorStubs.cs.
// Dispatch stubs check AddressTable before falling back to GetFunctionAddress.
// SlowDispatchStub handles addresses without a registered translation.

use crate::instructions::context::ArmEmitterCtx;
use crate::instructions::native_function::NativeFn;
use crate::instructions::simd_helper;
use crate::ir::translation::TranslationContext;
use crate::ir::types::OperandType;
use crate::memory::MemoryManager;
use crate::registers::CpuState;
use crate::translation::compiler::{self, CompilerOptions};
use crate::translation::translated_function::TranslatedFunction;
use crate::util::address_table::AddressTable;

/// C# WrapperFunction delegate: (nativeContext, guestMethod) → next PC.
pub type WrapperFn = unsafe extern "C" fn(*mut u8, u64) -> u64;

/// C# DispatcherFunction delegate: (nativeContext, startAddress) → void.
/// Runs the whole guest dispatch loop in native code (no managed bounce).
pub type DispatcherFn = unsafe extern "C" fn(*mut u8, u64);

/// C# TranslatorStubs. Mappings are owned here; the raw pointers stay
/// valid for the lifetime of the struct.
pub struct TranslatorStubs {
    dispatch_stub: TranslatedFunction,
    slow_dispatch_stub: TranslatedFunction,
    context_wrapper: TranslatedFunction,
    dispatch_loop: TranslatedFunction,
}

impl TranslatorStubs {
    /// Create stubs. `table` provides the AddressTable info for the
    /// dispatch stub's inline walk. The table's fill value is set to
    /// this stub's slow_dispatch_stub address after construction  the
    /// caller (Translator) is responsible for that.
    pub fn new(memory: &dyn MemoryManager, table: &AddressTable) -> Self {
        let slow = generate_slow_dispatch_stub(memory);
        let fast = generate_dispatch_stub(memory, table);
        let wrapper = generate_context_wrapper(memory);
        // Dispatch loop bakes the dispatch-stub address as a constant, so the
        // stub must be generated (and mapped) first.
        let fast_addr = fast.func_ptr.expect("dispatch stub not mapped").as_ptr() as u64;
        let dispatch_loop = generate_dispatch_loop(memory, fast_addr);

        Self {
            dispatch_stub: fast,
            slow_dispatch_stub: slow,
            context_wrapper: wrapper,
            dispatch_loop,
        }
    }

    /// Host address of the dispatch stub (C# DispatchStub property).
    /// Fast path: inline table walk + fallback to GetFunctionAddress.
    pub fn dispatch_stub(&self) -> u64 {
        self.dispatch_stub
            .func_ptr
            .expect("dispatch stub not mapped")
            .as_ptr() as u64
    }

    /// Host address of the slow dispatch stub (C# SlowDispatchStub).
    /// Used as the table's fill value  pure fallback via GetFunctionAddress.
    pub fn slow_dispatch_stub(&self) -> u64 {
        self.slow_dispatch_stub
            .func_ptr
            .expect("slow dispatch stub not mapped")
            .as_ptr() as u64
    }

    /// C# ContextWrapper property.
    pub fn context_wrapper(&self) -> WrapperFn {
        let ptr = self
            .context_wrapper
            .func_ptr
            .expect("context wrapper not mapped")
            .as_ptr();
        unsafe { std::mem::transmute::<*mut u8, WrapperFn>(ptr) }
    }

    /// C# DispatchLoop property  the unmanaged dispatch loop entry point.
    pub fn dispatch_loop(&self) -> DispatcherFn {
        let ptr = self
            .dispatch_loop
            .func_ptr
            .expect("dispatch loop not mapped")
            .as_ptr();
        unsafe { std::mem::transmute::<*mut u8, DispatcherFn>(ptr) }
    }
}

/// C# GenerateDispatchStub  inline AddressTable walk with fallback.
fn generate_dispatch_stub(memory: &dyn MemoryManager, table: &AddressTable) -> TranslatedFunction {
    let mut ctx = TranslationContext::new();
    let mut e = ArmEmitterCtx::new(&mut ctx, memory);

    let lbl_fallback = e.ctx.operands.label();

    // Load the target guest address from the native context.
    let native_context = e.load_argument(OperandType::I64, 0);
    let disp_offset = e
        .ctx
        .operands
        .constant_i64(CpuState::dispatch_address_offset() as i64);
    let disp_addr = e.add(native_context, disp_offset);
    let guest_address = e.load(OperandType::I64, disp_addr);

    // C#: check IsValid  if (guestAddress & ~Mask) != 0 → fallback.
    let not_mask = e.ctx.operands.constant_i64(!table.mask() as i64);
    let masked = e.bitwise_and(guest_address, not_mask);
    // C#: BranchIfTrue(lblFallback, masked)  if (address & ~Mask) != 0 → fallback
    e.branch_if_true(lbl_fallback, masked);

    // C#: inline table walk
    let entry_base = e.ctx.operands.constant_i64(table.base() as i64);
    let mut page = entry_base;

    let levels = table.levels();
    let mut index = guest_address;

    let const3 = e.ctx.operands.constant_i32(3);

    for i in 0..levels.len() {
        let level = levels[i];

        // C#: mask = level.Mask >> level.Index
        let level_mask_val = (level.mask() >> level.index) as i64;
        let level_mask = e.ctx.operands.constant_i64(level_mask_val);
        let shift_amount = e.ctx.operands.constant_i32(level.index);

        // C#: index = (guestAddress >> level.Index) & mask
        let shifted = e.shift_right_ui(guest_address, shift_amount);
        index = e.bitwise_and(shifted, level_mask);

        if i < levels.len() - 1 {
            // C#: page = Load(I64, page + (index << 3))
            let index_times_8 = e.shift_left(index, const3);
            let addr = e.add(page, index_times_8);
            page = e.load(OperandType::I64, addr);

            // C#: if page == null → fallback
            e.branch_if_false(lbl_fallback, page);
        }
    }

    // C#: hostAddress = Load(I64, page + (index << 3))
    let index_times_8 = e.shift_left(index, const3);
    let host_addr_op = e.add(page, index_times_8);
    let host_address = e.load(OperandType::I64, host_addr_op);

    // C#: Tailcall(hostAddress, nativeContext)
    e.tailcall(host_address, &[native_context]);

    // C# lblFallback: GetFunctionAddress(guestAddress) → tailcall
    e.mark_label(lbl_fallback, crate::ir::block::BlockFrequency::Cold);
    let fallback_host = e
        .call(NativeFn::GetFunctionAddress, &[guest_address])
        .expect("GetFunctionAddress returns I64");
    e.tailcall(fallback_host, &[native_context]);

    let cfg = e.get_control_flow_graph();
    drop(e);

    let compiled = compiler::compile(
        cfg,
        vec![OperandType::I64],
        OperandType::I64,
        CompilerOptions::new(compiler::compiler_options::NONE),
        &mut ctx,
    );
    TranslatedFunction::new(compiled.code, 0)
}

/// C# GenerateSlowDispatchStub  pure fallback: GetFunctionAddress + tailcall.
fn generate_slow_dispatch_stub(memory: &dyn MemoryManager) -> TranslatedFunction {
    let mut ctx = TranslationContext::new();
    let mut e = ArmEmitterCtx::new(&mut ctx, memory);

    let native_context = e.load_argument(OperandType::I64, 0);
    let disp_offset = e
        .ctx
        .operands
        .constant_i64(CpuState::dispatch_address_offset() as i64);
    let disp_addr = e.add(native_context, disp_offset);
    let guest_address = e.load(OperandType::I64, disp_addr);

    let host_address = e
        .call(NativeFn::GetFunctionAddress, &[guest_address])
        .expect("GetFunctionAddress returns I64");
    e.tailcall(host_address, &[native_context]);

    let cfg = e.get_control_flow_graph();
    drop(e);

    let compiled = compiler::compile(
        cfg,
        vec![OperandType::I64],
        OperandType::I64,
        CompilerOptions::new(compiler::compiler_options::NONE),
        &mut ctx,
    );
    TranslatedFunction::new(compiled.code, 0)
}

/// C# GenerateContextWrapper: FP-mode sync around one guest-function call.
fn generate_context_wrapper(memory: &dyn MemoryManager) -> TranslatedFunction {
    let mut ctx = TranslationContext::new();
    let mut e = ArmEmitterCtx::new(&mut ctx, memory);

    let native_context = e.load_argument(OperandType::I64, 0);
    let guest_method = e.load_argument(OperandType::I64, 1);

    // C# EmitSyncFpContext(enter: true)
    simd_helper::enter_arm_fp_mode_with(&mut e, |e, flag| {
        let offset = e
            .ctx
            .operands
            .constant_i64(CpuState::fp_flag_offset(flag as u8) as i64);
        let flag_address = e.add(native_context, offset);
        e.load(OperandType::I32, flag_address)
    });

    let return_value = e
        .call_address(guest_method, OperandType::I64, &[native_context])
        .expect("guest function returns I64");

    // C# EmitSyncFpContext(enter: false)
    simd_helper::exit_arm_fp_mode(&mut e);

    e.return_with(return_value);

    let cfg = e.get_control_flow_graph();
    drop(e);

    let compiled = compiler::compile(
        cfg,
        vec![OperandType::I64, OperandType::I64],
        OperandType::I64,
        CompilerOptions::new(compiler::compiler_options::NONE),
        &mut ctx,
    );
    TranslatedFunction::new(compiled.code, 0)
}

/// C# GenerateDispatchLoop  the unmanaged dispatch loop. Runs the entire
/// guest dispatch in native code: store target → call dispatch stub (inline
/// table walk, no lock) → loop while next != 0 && running. Replaces the
/// managed Rust loop (RwLock + hashmap per bounce). `dispatch_stub_addr` is
/// baked as a constant, so the dispatch stub must already be mapped.
fn generate_dispatch_loop(
    memory: &dyn MemoryManager,
    dispatch_stub_addr: u64,
) -> TranslatedFunction {
    use crate::ir::block::BlockFrequency;

    let mut ctx = TranslationContext::new();
    let mut e = ArmEmitterCtx::new(&mut ctx, memory);

    let begin_lbl = e.ctx.operands.label();
    let end_lbl = e.ctx.operands.label();

    let native_context = e.load_argument(OperandType::I64, 0);
    // C#: guestAddress = Copy(AllocateLocal(I64), LoadArgument(I64, 1))  a
    // mutable local reassigned each iteration.
    let arg_address = e.load_argument(OperandType::I64, 1);
    let guest_address = e.ctx.operands.local(OperandType::I64);
    e.copy_to(guest_address, arg_address);

    let running_off = e
        .ctx
        .operands
        .constant_i64(CpuState::running_offset() as i64);
    let running_address = e.add(native_context, running_off);
    let disp_off = e
        .ctx
        .operands
        .constant_i64(CpuState::dispatch_address_offset() as i64);
    let dispatch_address = e.add(native_context, disp_off);

    // C# EmitSyncFpContext(enter: true)  once for the whole loop.
    simd_helper::enter_arm_fp_mode_with(&mut e, |e, flag| {
        let offset = e
            .ctx
            .operands
            .constant_i64(CpuState::fp_flag_offset(flag as u8) as i64);
        let flag_address = e.add(native_context, offset);
        e.load(OperandType::I32, flag_address)
    });

    e.mark_label(begin_lbl, BlockFrequency::Default);
    e.store(dispatch_address, guest_address);
    let stub_addr = e.ctx.operands.constant_i64(dispatch_stub_addr as i64);
    let next = e
        .call_address(stub_addr, OperandType::I64, &[native_context])
        .expect("dispatch stub returns I64");
    e.copy_to(guest_address, next);
    e.branch_if_false(end_lbl, guest_address);
    let running = e.load(OperandType::I32, running_address);
    e.branch_if_false(end_lbl, running);
    e.branch(begin_lbl);

    e.mark_label(end_lbl, BlockFrequency::Default);
    // C# EmitSyncFpContext(enter: false)
    simd_helper::exit_arm_fp_mode(&mut e);
    e.return_void();

    let cfg = e.get_control_flow_graph();
    drop(e);

    let compiled = compiler::compile(
        cfg,
        vec![OperandType::I64, OperandType::I64],
        OperandType::None,
        CompilerOptions::new(compiler::compiler_options::NONE),
        &mut ctx,
    );
    TranslatedFunction::new(compiled.code, 0)
}
