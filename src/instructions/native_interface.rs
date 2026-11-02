// Port of Instructions/NativeInterface.cs and Translation/Delegates.cs.
// Emitters reference helper addresses while building IR; helpers run at guest runtime.
// Registry indices are NativeFn discriminants. V128 arguments follow the host ABI.

use crate::memory::MemoryManager;
use crate::registers::V128;

use std::cell::Cell;
// Thread context (C# NativeInterface.ThreadContext)
//
// Translator::execute registers memory and translator pointers for the guest run.
// The FFI entry point also registers ExecutionContext; Rust callers using exception
// helpers must call register_exec_context and unregister_exec_context themselves.
use std::ptr::NonNull;

thread_local! {
    static MEMORY: Cell<Option<*const dyn MemoryManager>> = const { Cell::new(None) };
    static TRANSLATOR: Cell<Option<*const crate::translation::translator::Translator>> =
        const { Cell::new(None) };
    static EXEC_CONTEXT: Cell<Option<NonNull<crate::state::execution_context::ExecutionContext>>> =
        const { Cell::new(None) };
}

/// C# ThreadContext.Context slot. Optional: the differential harness
/// drives a raw CpuState with no ExecutionContext.
pub fn register_exec_context(context: *mut crate::state::execution_context::ExecutionContext) {
    let context = NonNull::new(context).expect("register_exec_context called with null");
    EXEC_CONTEXT.set(Some(context));
}

pub fn unregister_exec_context() {
    EXEC_CONTEXT.set(None);
}

// C# GetContext(). Shared ref: the context is concurrently reachable from
// another host thread (stop/interrupt), so no exclusive borrow is formed.
// State mutation goes through the context's `&self` API (atomics for the
// cross-thread fields, raw cpu pointer for guest registers).
fn get_context<'a>() -> &'a crate::state::execution_context::ExecutionContext {
    let ptr = EXEC_CONTEXT
        .get()
        .expect("NativeInterface called with no registered ExecutionContext");

    unsafe { ptr.as_ref() }
}

/// Register the memory manager and translator for the current host thread.
///
/// The translator pointer is raw on purpose: guest code running inside
/// `Translator::execute` calls back into `get_function_address`, which
/// must mutate the function cache while the outer `&mut self` borrow is
/// live. Single-threaded per guest thread; no Rust reference is held
/// across the guest call (see Translator::execute_single).
pub fn register_thread(
    memory: *const dyn MemoryManager,
    translator: *const crate::translation::translator::Translator,
) {
    MEMORY.set(Some(memory));
    TRANSLATOR.set(Some(translator));
}

/// C# NativeInterface.UnregisterThread.
pub fn unregister_thread() {
    MEMORY.set(None);
    TRANSLATOR.set(None);
}

// C# GetContext().Translator access.
fn get_translator<'a>() -> &'a crate::translation::translator::Translator {
    let ptr = TRANSLATOR
        .get()
        .expect("NativeInterface called with no registered translator");
    unsafe { &*ptr }
}

// C# GetMemoryManager().
fn get_memory_manager<'a>() -> &'a dyn MemoryManager {
    let ptr = MEMORY
        .get()
        .expect("NativeInterface called with no registered thread context");
    unsafe { &*ptr }
}

// extern "C" helpers (C# [UnmanagedCallersOnly] methods)
pub extern "C" fn break_(address: u64, imm: i32) {
    get_context().on_break(address, imm);
}

pub extern "C" fn supervisor_call(address: u64, imm: i32) {
    let x0 = get_context().get_x(0) as i64;
    svc_stats::record(imm, x0, address, svc_stats::thread_id());
    get_context().on_supervisor_call(address, imm);
}

/// Env-gated SVC histogram (ARMATURE_SVC_STATS=1). Counts each Horizon SVC by
/// its immediate (the SVC id on AArch64 Horizon). Fires under BOTH dispatch
/// loops (the SVC callback is always native→host). Dumps a histogram to stderr
/// every 1<<20 SVCs so a thread that busy-retries a blocking wait
/// (0x18 WaitSynchronization / 0x1C WaitProcessWideKeyAtomic / 0x1A ArbitrateLock)
/// shows up as a runaway count, distinguishing "wait returns instantly" from
/// "never even tries to sleep".
mod svc_stats {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static ON: AtomicBool = AtomicBool::new(false);
    static INIT: std::sync::Once = std::sync::Once::new();
    static NEXT_TID: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static TID: u64 = NEXT_TID.fetch_add(1, Ordering::Relaxed);
    }
    /// Stable, non-zero, per-OS-thread id (sequential, assigned on first use).
    #[inline]
    pub fn thread_id() -> u64 {
        TID.with(|t| *t)
    }
    static HIST: [AtomicU64; 256] = {
        const Z: AtomicU64 = AtomicU64::new(0);
        [Z; 256]
    };
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    // svcSleepThread (0x0b) timeout breakdown: yield0 = X0==0, yieldNeg = X0<0
    // (-1/-2 yield variants), sleep = X0>0 (real timed sleep).
    static SLEEP_YIELD0: AtomicU64 = AtomicU64::new(0);
    static SLEEP_YIELDNEG: AtomicU64 = AtomicU64::new(0);
    static SLEEP_REAL: AtomicU64 = AtomicU64::new(0);

    // Guest-PC histogram for svcSleepThread (0x0b): fixed open-addressed table.
    // Identifies which guest loop is yield-spinning.
    const PC_SLOTS: usize = 128;
    static PC_ADDR: [AtomicU64; PC_SLOTS] = {
        const Z: AtomicU64 = AtomicU64::new(0);
        [Z; PC_SLOTS]
    };
    static PC_CNT: [AtomicU64; PC_SLOTS] = {
        const Z: AtomicU64 = AtomicU64::new(0);
        [Z; PC_SLOTS]
    };

    fn record_pc(addr: u64) {
        let mut i = (addr.wrapping_mul(0x9E3779B97F4A7C15) >> 57) as usize & (PC_SLOTS - 1);
        for _ in 0..PC_SLOTS {
            let cur = PC_ADDR[i].load(Ordering::Relaxed);
            if cur == addr {
                PC_CNT[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
            if cur == 0
                && PC_ADDR[i]
                    .compare_exchange(0, addr, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                PC_CNT[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
            i = (i + 1) & (PC_SLOTS - 1);
        }
    }

    // Per-thread profile: open-addressed by ExecutionContext ptr (per guest
    // thread). Classifies each thread as spinning vs blocking.
    const T_SLOTS: usize = 64;
    struct Thread {
        tid: AtomicU64,
        total: AtomicU64,
        yield_neg: AtomicU64, // svcSleepThread(-1/-2): yield-spin
        wait_sync: AtomicU64, // 0x18 WaitSynchronization
        wait_addr: AtomicU64, // 0x34 WaitForAddress
        wait_key: AtomicU64,  // 0x1c WaitProcessWideKeyAtomic
    }
    static THREADS: [Thread; T_SLOTS] = {
        const T: Thread = Thread {
            tid: AtomicU64::new(0),
            total: AtomicU64::new(0),
            yield_neg: AtomicU64::new(0),
            wait_sync: AtomicU64::new(0),
            wait_addr: AtomicU64::new(0),
            wait_key: AtomicU64::new(0),
        };
        [T; T_SLOTS]
    };

    fn thread_slot(tid: u64) -> Option<&'static Thread> {
        let mut i = (tid.wrapping_mul(0x9E3779B97F4A7C15) >> 58) as usize & (T_SLOTS - 1);
        for _ in 0..T_SLOTS {
            let cur = THREADS[i].tid.load(Ordering::Relaxed);
            if cur == tid {
                return Some(&THREADS[i]);
            }
            if cur == 0
                && THREADS[i]
                    .tid
                    .compare_exchange(0, tid, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                return Some(&THREADS[i]);
            }
            i = (i + 1) & (T_SLOTS - 1);
        }
        None
    }

    #[inline]
    pub fn record(imm: i32, x0: i64, address: u64, tid: u64) {
        INIT.call_once(|| {
            ON.store(
                std::env::var("ARMATURE_SVC_STATS").as_deref() == Ok("1"),
                Ordering::Relaxed,
            );
        });
        if !ON.load(Ordering::Relaxed) {
            return;
        }
        let i = (imm as u32 & 0xff) as usize;
        HIST[i].fetch_add(1, Ordering::Relaxed);
        if i == 0x0b {
            match x0 {
                0 => &SLEEP_YIELD0,
                n if n < 0 => &SLEEP_YIELDNEG,
                _ => &SLEEP_REAL,
            }
            .fetch_add(1, Ordering::Relaxed);
            record_pc(address);
        }
        if let Some(th) = thread_slot(tid) {
            th.total.fetch_add(1, Ordering::Relaxed);
            match i {
                0x0b if x0 < 0 => th.yield_neg.fetch_add(1, Ordering::Relaxed),
                0x18 => th.wait_sync.fetch_add(1, Ordering::Relaxed),
                0x34 => th.wait_addr.fetch_add(1, Ordering::Relaxed),
                0x1c => th.wait_key.fetch_add(1, Ordering::Relaxed),
                _ => 0,
            };
        }
        let t = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        if t % (1 << 20) == 0 {
            dump(t);
        }
    }

    fn dump(total: u64) {
        let mut line = format!("[svc-stats] total={total} | ");
        for i in 0..256 {
            let c = HIST[i].load(Ordering::Relaxed);
            if c > 0 {
                line.push_str(&format!("{i:#04x}={c} "));
            }
        }
        line.push_str(&format!(
            "| sleep0={} sleepNeg={} sleepReal={}",
            SLEEP_YIELD0.load(Ordering::Relaxed),
            SLEEP_YIELDNEG.load(Ordering::Relaxed),
            SLEEP_REAL.load(Ordering::Relaxed),
        ));
        eprintln!("{line}");

        // Top guest PCs issuing svcSleepThread.
        let mut pcs: Vec<(u64, u64)> = (0..PC_SLOTS)
            .filter_map(|i| {
                let a = PC_ADDR[i].load(Ordering::Relaxed);
                let c = PC_CNT[i].load(Ordering::Relaxed);
                (a != 0 && c != 0).then_some((c, a))
            })
            .collect();
        pcs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut pl = String::from("[svc-stats] top sleep PCs:");
        for (c, a) in pcs.iter().take(8) {
            pl.push_str(&format!(" {a:#x}={c}"));
        }
        eprintln!("{pl}");

        // Per-thread profile: spin (yield_neg) vs block (wait_*).
        let mut rows: Vec<(u64, u64, u64, u64, u64, u64)> = (0..T_SLOTS)
            .filter_map(|i| {
                let tid = THREADS[i].tid.load(Ordering::Relaxed);
                let tot = THREADS[i].total.load(Ordering::Relaxed);
                (tid != 0 && tot != 0).then(|| {
                    (
                        tot,
                        tid,
                        THREADS[i].yield_neg.load(Ordering::Relaxed),
                        THREADS[i].wait_sync.load(Ordering::Relaxed),
                        THREADS[i].wait_addr.load(Ordering::Relaxed),
                        THREADS[i].wait_key.load(Ordering::Relaxed),
                    )
                })
            })
            .collect();
        rows.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for (tot, tid, yn, ws, wa, wk) in rows.iter().take(12) {
            eprintln!(
                "[svc-stats]   thr {tid:#x} total={tot} yieldNeg={yn} waitSync={ws} waitAddr={wa} waitKey={wk}"
            );
        }
    }
}

pub extern "C" fn undefined(address: u64, op_code: i32) {
    get_context().on_undefined(address, op_code);
}

// System registers.
pub extern "C" fn get_ctr_el0() -> u64 {
    crate::state::execution_context::ExecutionContext::CTR_EL0 as u64
}
pub extern "C" fn get_dczid_el0() -> u64 {
    crate::state::execution_context::ExecutionContext::DCZID_EL0 as u64
}
pub extern "C" fn get_cntfrq_el0() -> u64 {
    get_context().cntfrq_el0()
}
pub extern "C" fn get_cntpct_el0() -> u64 {
    get_context().cntpct_el0()
}
pub extern "C" fn get_cntvct_el0() -> u64 {
    get_context().cntvct_el0()
}

// Reads (C# ReadGuest<T>).
pub extern "C" fn read_byte(address: u64) -> u8 {
    get_memory_manager().read_guest_u8(address)
}
pub extern "C" fn read_uint16(address: u64) -> u16 {
    get_memory_manager().read_guest_u16(address)
}
pub extern "C" fn read_uint32(address: u64) -> u32 {
    get_memory_manager().read_guest_u32(address)
}
pub extern "C" fn read_uint64(address: u64) -> u64 {
    get_memory_manager().read_guest_u64(address)
}

/// Debug-only: recompute the host-mapped/host-tracked address translation in
/// Rust, log it, and return the host address the inline JIT path would have
/// produced. Gated by ARMATURE_MEM_DEBUG (see memory_helpers). Logs the first
/// 64 accesses plus every access whose page-table entry is 0 (= unmapped page,
/// the prime suspect for wild stores). Faithful to emit_pt_pointer_load.
pub extern "C" fn dbg_translate(va: u64) -> u64 {
    use crate::memory::MemoryManagerType as T;
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);

    let mm = get_memory_manager();
    let mt = mm.manager_type();
    let asb = mm.address_space_bits();
    let pt_base = mm.page_table_pointer() as u64;

    let mut masked = va;
    let mut entry = 0u64;
    let mut pte_addr = 0u64;
    let host = match mt {
        T::HostMapped => {
            masked = va & (u64::MAX >> (64 - asb));
            pt_base.wrapping_add(masked)
        }
        T::HostMappedUnsafe => pt_base.wrapping_add(va),
        _ => {
            // HostTracked / HostTrackedUnsafe.
            if mt == T::HostTracked {
                masked = va & (u64::MAX >> (64 - asb));
            }
            pte_addr = pt_base.wrapping_add((masked >> 12) << 3);
            entry = unsafe { *(pte_addr as *const u64) };
            masked.wrapping_add(entry)
        }
    };

    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 64 || entry == 0 {
        eprintln!(
            "[memdbg #{n}] va={va:#x} masked={masked:#x} type={mt:?} asb={asb} \
             pt_base={pt_base:#x} pte_addr={pte_addr:#x} entry={entry:#x} host={host:#x}"
        );
    }
    host
}
pub extern "C" fn read_vector128(address: u64) -> V128 {
    get_memory_manager().read_guest_v128(address)
}

// Writes (C# WriteGuest<T>).
pub extern "C" fn write_byte(address: u64, value: u8) {
    get_memory_manager().write_guest_u8(address, value)
}
pub extern "C" fn write_uint16(address: u64, value: u16) {
    get_memory_manager().write_guest_u16(address, value)
}
pub extern "C" fn write_uint32(address: u64, value: u32) {
    get_memory_manager().write_guest_u32(address, value)
}
pub extern "C" fn write_uint64(address: u64, value: u64) {
    get_memory_manager().write_guest_u64(address, value)
}
pub extern "C" fn write_vector128(address: u64, value: V128) {
    get_memory_manager().write_guest_v128(address, value)
}

/// C# NativeInterface.EnqueueForRejit → Context.Translator.EnqueueForRejit.
pub extern "C" fn enqueue_for_rejit(address: u64) {
    get_translator().enqueue_for_rejit(address);
}

pub extern "C" fn signal_memory_tracking(address: u64, size: u64, write: u8) {
    get_memory_manager().signal_memory_tracking(address, size, write == 1, false, None);
}

/// C#: throw new InvalidAccessException(address). No exception type to
/// surface through JIT frames  a guest invalid access is fatal here.
pub extern "C" fn throw_invalid_memory_access(address: u64) {
    panic!("invalid memory access at address {address:#x}");
}

/// C# NativeInterface.GetFunctionAddress: translate (or fetch cached) the
/// function at `address`, return its host entry point. Called from the
/// dispatch stub and table-branch fallback paths.
pub extern "C" fn get_function_address(address: u64) -> u64 {
    get_translator().get_or_translate_ptr(address)
}

/// C# InstEmit.DczSizeInBytes = 4 << DczSizeLog2 (DczidEl0 = 4 → 64).
pub extern "C" fn invalidate_cache_line(address: u64) {
    get_translator().invalidate_jit_cache_region(address, 64);
}

/// C# NativeInterface.CheckSynchronization: GetContext().CheckInterrupt(),
/// return context.Running.
#[unsafe(no_mangle)]
pub extern "C" fn check_synchronization() -> u8 {
    let Some(ptr) = EXEC_CONTEXT.get() else {
        return 1;
    };
    let context = unsafe { ptr.as_ref() };
    context.check_interrupt();
    context.running() as u8
}
