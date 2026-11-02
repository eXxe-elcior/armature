// C ABI for memory callbacks, exception callbacks, and translator/context lifecycle.

use crate::memory::{MemoryManager, MemoryManagerType};
use crate::registers::V128;
use crate::state::execution_context::ExecutionContext;
use crate::translation::translator::Translator;

/// Host-provided memory callbacks. `ctx` is an opaque host pointer passed
/// back on every call. V128 crosses as a *mut u8 to 16 bytes (LE).
#[repr(C)]
pub struct ArmatureMemoryCallbacks {
    pub ctx: *mut std::ffi::c_void,
    pub address_space_bits: i32,
    /// 0 = SoftwareMmu, 1 = SoftwarePageTable, 2 = HostMapped,
    /// 3 = HostMappedUnsafe, 4 = HostTracked, 5 = HostTrackedUnsafe.
    pub manager_type: i32,
    pub page_table_pointer: *mut u8,

    pub read_u8: unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u8,
    pub read_u16: unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u16,
    pub read_u32: unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u32,
    pub read_u64: unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u64,
    pub read_v128: unsafe extern "C" fn(*mut std::ffi::c_void, u64, *mut u8),
    pub write_u8: unsafe extern "C" fn(*mut std::ffi::c_void, u64, u8),
    pub write_u16: unsafe extern "C" fn(*mut std::ffi::c_void, u64, u16),
    pub write_u32: unsafe extern "C" fn(*mut std::ffi::c_void, u64, u32),
    pub write_u64: unsafe extern "C" fn(*mut std::ffi::c_void, u64, u64),
    pub write_v128: unsafe extern "C" fn(*mut std::ffi::c_void, u64, *const u8),
    pub is_mapped: unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u8,
    pub signal_memory_tracking: unsafe extern "C" fn(*mut std::ffi::c_void, u64, u64, u8),
}

struct FfiMemory(ArmatureMemoryCallbacks);
unsafe impl Send for FfiMemory {}
unsafe impl Sync for FfiMemory {}

impl MemoryManager for FfiMemory {
    fn address_space_bits(&self) -> i32 {
        self.0.address_space_bits
    }
    fn page_table_pointer(&self) -> *mut u8 {
        self.0.page_table_pointer
    }
    fn manager_type(&self) -> MemoryManagerType {
        match self.0.manager_type {
            0 => MemoryManagerType::SoftwareMmu,
            1 => MemoryManagerType::SoftwarePageTable,
            2 => MemoryManagerType::HostMapped,
            3 => MemoryManagerType::HostMappedUnsafe,
            4 => MemoryManagerType::HostTracked,
            5 => MemoryManagerType::HostTrackedUnsafe,
            _ => panic!("Unknown MemoryManagerType: {}", self.0.manager_type),
        }
    }
    fn read_u8(&self, va: u64) -> u8 {
        unsafe { (self.0.read_u8)(self.0.ctx, va) }
    }
    fn read_u16(&self, va: u64) -> u16 {
        unsafe { (self.0.read_u16)(self.0.ctx, va) }
    }
    fn read_u32(&self, va: u64) -> u32 {
        unsafe { (self.0.read_u32)(self.0.ctx, va) }
    }
    fn read_u64(&self, va: u64) -> u64 {
        unsafe { (self.0.read_u64)(self.0.ctx, va) }
    }
    fn read_v128(&self, va: u64) -> V128 {
        let mut buf = [0u8; 16];
        unsafe { (self.0.read_v128)(self.0.ctx, va, buf.as_mut_ptr()) };
        u128::from_le_bytes(buf)
    }
    fn read_tracked_u8(&self, va: u64) -> u8 {
        self.read_u8(va)
    }
    fn read_tracked_u16(&self, va: u64) -> u16 {
        self.read_u16(va)
    }
    fn read_tracked_u32(&self, va: u64) -> u32 {
        self.read_u32(va)
    }
    fn read_tracked_u64(&self, va: u64) -> u64 {
        self.read_u64(va)
    }
    fn read_tracked_v128(&self, va: u64) -> V128 {
        self.read_v128(va)
    }
    fn write_u8(&self, va: u64, value: u8) {
        unsafe { (self.0.write_u8)(self.0.ctx, va, value) }
    }
    fn write_u16(&self, va: u64, value: u16) {
        unsafe { (self.0.write_u16)(self.0.ctx, va, value) }
    }
    fn write_u32(&self, va: u64, value: u32) {
        unsafe { (self.0.write_u32)(self.0.ctx, va, value) }
    }
    fn write_u64(&self, va: u64, value: u64) {
        unsafe { (self.0.write_u64)(self.0.ctx, va, value) }
    }
    fn write_v128(&self, va: u64, value: V128) {
        let buf = value.to_le_bytes();
        unsafe { (self.0.write_v128)(self.0.ctx, va, buf.as_ptr()) }
    }
    fn is_mapped(&self, va: u64) -> bool {
        unsafe { (self.0.is_mapped)(self.0.ctx, va) != 0 }
    }
    fn signal_memory_tracking(&self, va: u64, size: u64, write: bool, _p: bool, _e: Option<i32>) {
        unsafe { (self.0.signal_memory_tracking)(self.0.ctx, va, size, write as u8) }
    }
}

/// Create a translator over host memory callbacks. Returns an owned
/// handle; destroy with armature_translator_destroy.
#[unsafe(no_mangle)]
pub extern "C" fn armature_translator_create(
    callbacks: *const ArmatureMemoryCallbacks,
) -> *mut Translator {
    // One-shot proof that the armature JIT (not ARMeilleure) is driving this
    // run. Fires the moment the C# shim builds the translator.
    static BANNER: std::sync::Once = std::sync::Once::new();
    BANNER.call_once(|| {
        eprintln!(
            "[armature] JIT engaged  v{} (AArch64 → x86-64)",
            env!("CARGO_PKG_VERSION")
        );
    });

    let cb = unsafe { std::ptr::read(callbacks) };
    Box::into_raw(Box::new(Translator::new(Box::new(FfiMemory(cb)))))
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_translator_destroy(translator: *mut Translator) {
    if !translator.is_null() {
        drop(unsafe { Box::from_raw(translator) });
    }
}

/// Block until every guest thread has returned from `armature_execute`. The
/// host MUST call thi after StopRunning on all contexts before unmapping
/// the guest address space, so no in-flight JIT memory access faults on a
/// freed/untracked page (crash on Ryujinx STOP).
#[unsafe(no_mangle)]
pub extern "C" fn armature_translator_drain_execute(translator: *const Translator) {
    if !translator.is_null() {
        unsafe { &*translator }.drain_execute();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_translator_invalidate(
    translator: *mut Translator,
    address: u64,
    size: u64,
) {
    unsafe { &mut *translator }.invalidate_jit_cache_region(address, size);
}

/// Create an execution context (v1: no exception callbacks).
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_create() -> *mut ExecutionContext {
    Box::into_raw(Box::new(ExecutionContext::new(
        None, None, None, None, None,
    )))
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_destroy(context: *mut ExecutionContext) {
    if !context.is_null() {
        drop(unsafe { Box::from_raw(context) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_get_x(context: *const ExecutionContext, index: i32) -> u64 {
    unsafe { &*context }.get_x(index as usize)
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_set_x(context: *mut ExecutionContext, index: i32, value: u64) {
    unsafe { &*context }.set_x(index as usize, value);
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_get_v(
    context: *const ExecutionContext,
    index: i32,
    out: *mut u8,
) {
    let v = unsafe { &*context }.get_v(index as usize);
    unsafe { std::ptr::copy_nonoverlapping(v.to_le_bytes().as_ptr(), out, 16) };
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_set_v(
    context: *mut ExecutionContext,
    index: i32,
    value: *const u8,
) {
    let mut buf = [0u8; 16];
    unsafe { std::ptr::copy_nonoverlapping(value, buf.as_mut_ptr(), 16) };
    unsafe { &*context }.set_v(index as usize, u128::from_le_bytes(buf));
}

/// Next-execution PC (the dispatch address). Eden's GetContext/SetContext need
/// a live PC; v1's exception-only Pc was insufficient for a dynarmic-style host.
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_get_pc(context: *const ExecutionContext) -> u64 {
    unsafe { (*(&*context).cpu_ptr()).dispatch_address }
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_set_pc(context: *mut ExecutionContext, value: u64) {
    unsafe { (*(&*context).cpu_ptr()).dispatch_address = value };
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_request_interrupt(context: *mut ExecutionContext) {
    // Cross-thread: shared ref only  pokes an atomic flag, no exclusive
    // borrow while the owner thread runs guest code on the same context.
    unsafe { &*context }.request_interrupt();
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_stop_running(context: *mut ExecutionContext) {
    // Cross-thread: shared ref only (atomic stores). See request_interrupt.
    unsafe { &*context }.stop_running();
}

/// Re-arm Running=true (+ counter) for the next run. Halt-return hosts (Eden)
/// bounce out of the dispatch loop via stop_running at each SVC/preemption,
/// then call this before re-entering armature_execute.
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_resume_running(context: *mut ExecutionContext) {
    unsafe { &*context }.resume_running();
}

/// Run guest code at `address` on this context until it exits (returns 0
/// or stops running). Registers the context for native-interface
/// callbacks for the duration of the run.
#[unsafe(no_mangle)]
pub extern "C" fn armature_execute(
    translator: *mut Translator,
    context: *mut ExecutionContext,
    address: u64,
) {
    crate::instructions::native_interface::register_exec_context(context);
    let native = unsafe { &*context }.native_context_ptr();
    unsafe { &mut *translator }.execute(native, address);
    crate::instructions::native_interface::unregister_exec_context();
}

// FFI v2: exception callbacks, counter, system register access
/// Host exception/interrupt callbacks. `ctx` opaque; the ExecutionContext
/// handle is passed back so the host can mutate state re-entrantly.
#[repr(C)]
pub struct ArmatureExceptionCallbacks {
    pub ctx: *mut std::ffi::c_void,
    pub on_break:
        Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut ExecutionContext, u64, i32)>,
    pub on_supervisor_call:
        Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut ExecutionContext, u64, i32)>,
    pub on_undefined:
        Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut ExecutionContext, u64, i32)>,
    pub on_interrupt: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut ExecutionContext)>,
    /// CNTFRQ_EL0 / CNTPCT_EL0 sources (0 if absent).
    pub counter_frequency: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> u64>,
    pub counter_value: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> u64>,
}

struct FfiCtxPtr(*mut std::ffi::c_void);
unsafe impl Send for FfiCtxPtr {}
unsafe impl Sync for FfiCtxPtr {}

struct FfiCounter {
    ctx: FfiCtxPtr,
    frequency: unsafe extern "C" fn(*mut std::ffi::c_void) -> u64,
    value: unsafe extern "C" fn(*mut std::ffi::c_void) -> u64,
}
impl crate::state::execution_context::Counter for FfiCounter {
    fn frequency(&self) -> u64 {
        unsafe { (self.frequency)(self.ctx.0) }
    }
    fn counter(&self) -> u64 {
        unsafe { (self.value)(self.ctx.0) }
    }
}

/// Create an execution context with host callbacks (C# ExecutionContext
/// ctor with ExceptionCallback delegates).
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_create_with_callbacks(
    callbacks: *const ArmatureExceptionCallbacks,
) -> *mut ExecutionContext {
    let cb = unsafe { std::ptr::read(callbacks) };
    let host: usize = cb.ctx as usize;

    let counter: Option<Box<dyn crate::state::execution_context::Counter>> =
        match (cb.counter_frequency, cb.counter_value) {
            (Some(f), Some(v)) => Some(Box::new(FfiCounter {
                ctx: FfiCtxPtr(host as *mut std::ffi::c_void),
                frequency: f,
                value: v,
            })),
            _ => None,
        };

    macro_rules! exc {
        ($slot:expr) => {
            $slot.map(|f| -> crate::state::execution_context::ExceptionCallback {
                Box::new(
                    move |ec: &ExecutionContext, address: u64, imm: i32| unsafe {
                        f(
                            host as *mut std::ffi::c_void,
                            ec as *const ExecutionContext as *mut ExecutionContext,
                            address,
                            imm,
                        )
                    },
                )
            })
        };
    }

    let interrupt = cb.on_interrupt.map(
        |f| -> crate::state::execution_context::ExceptionCallbackNoArgs {
            Box::new(move |ec: &ExecutionContext| unsafe {
                f(
                    host as *mut std::ffi::c_void,
                    ec as *const ExecutionContext as *mut ExecutionContext,
                )
            })
        },
    );

    Box::into_raw(Box::new(ExecutionContext::new(
        counter,
        interrupt,
        exc!(cb.on_break),
        exc!(cb.on_supervisor_call),
        exc!(cb.on_undefined),
    )))
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_get_tpidr(context: *const ExecutionContext) -> i64 {
    unsafe { (*(&*context).cpu_ptr()).tpidr_el0 }
}
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_set_tpidr(context: *mut ExecutionContext, value: i64) {
    unsafe { (*(&*context).cpu_ptr()).tpidr_el0 = value };
}
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_get_tpidrro(context: *const ExecutionContext) -> i64 {
    unsafe { (*(&*context).cpu_ptr()).tpidrro_el0 }
}
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_set_tpidrro(context: *mut ExecutionContext, value: i64) {
    unsafe { (*(&*context).cpu_ptr()).tpidrro_el0 = value };
}

/// NZCV packed as PState bits 31..28.
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_get_nzcv(context: *const ExecutionContext) -> u32 {
    let f = unsafe { &(*(&*context).cpu_ptr()).flags };
    (f[31] & 1) << 31 | (f[30] & 1) << 30 | (f[29] & 1) << 29 | (f[28] & 1) << 28
}
#[unsafe(no_mangle)]
pub extern "C" fn armature_context_set_nzcv(context: *mut ExecutionContext, value: u32) {
    let f = unsafe { &mut (*(&*context).cpu_ptr()).flags };
    f[31] = (value >> 31) & 1;
    f[30] = (value >> 30) & 1;
    f[29] = (value >> 29) & 1;
    f[28] = (value >> 28) & 1;
}

#[unsafe(no_mangle)]
pub extern "C" fn armature_context_is_running(context: *const ExecutionContext) -> u8 {
    u8::from(unsafe { &*context }.running())
}
