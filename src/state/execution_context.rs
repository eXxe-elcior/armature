// Port of State/ExecutionContext.cs.
//
// Wraps the CpuState (C# NativeContext  heap-allocated so its address is
// stable while baked into JIT code) plus the exception/interrupt callback
// surface. Guest registers are owned by the executing thread; stop/interrupt use atomics.

use crate::registers::{CpuState, V128};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

/// C# ICounter.
pub trait Counter {
    fn frequency(&self) -> u64;
    fn counter(&self) -> u64;
}

// Callbacks take `&ExecutionContext`: the context is shared across host
// threads (the owner runs guest code; another may stop/interrupt it), so no
// part of the API forms an exclusive `&mut ExecutionContext`. Guest-register
// mutation from inside a callback goes through the `&self` setters, which
// write the heap-stable CpuState via a raw pointer (see `cpu_ptr`).
pub type ExceptionCallback = Box<dyn Fn(&ExecutionContext, u64, i32) + Send>;
pub type ExceptionCallbackNoArgs = Box<dyn Fn(&ExecutionContext) + Send>;

/// C# ExecutionContext.MinCountForCheck.
const MIN_COUNT_FOR_CHECK: i32 = 4000;

pub struct ExecutionContext {
    /// C# _nativeContext. Heap-stable (its address is baked into JIT code)
    /// and concurrently mutated by JIT'd machine code through the native
    /// pointer, by the owner thread through `&self` accessors, and (atomic
    /// fields only) by other host threads. Wrapped in `UnsafeCell` so that
    /// all of those happen behind a shared `&self` legally: every access
    /// goes through `cpu_ptr()` (`UnsafeCell::get`) and projects to a single
    /// field  a `&CpuState`/`&mut CpuState` spanning the whole struct is
    /// never formed, so concurrent narrow accesses to disjoint fields do not
    /// alias. `UnsafeCell<T>` is `repr(transparent)`, so CpuState's byte
    /// layout and the JIT offsets are unchanged.
    cpu: Box<UnsafeCell<CpuState>>,

    /// C# Interrupted. Set cross-thread (`request_interrupt`, from another
    /// host thread) and cleared by the owner thread in `check_interrupt`;
    /// atomic so that hand-off is not a data race.
    interrupted: AtomicBool,

    counter_source: Option<Box<dyn Counter>>,

    interrupt_callback: Option<ExceptionCallbackNoArgs>,
    break_callback: Option<ExceptionCallback>,
    supervisor_callback: Option<ExceptionCallback>,
    undefined_callback: Option<ExceptionCallback>,
}

impl ExecutionContext {
    pub fn new(
        counter_source: Option<Box<dyn Counter>>,
        interrupt_callback: Option<ExceptionCallbackNoArgs>,
        break_callback: Option<ExceptionCallback>,
        supervisor_callback: Option<ExceptionCallback>,
        undefined_callback: Option<ExceptionCallback>,
    ) -> Self {
        let cpu: Box<UnsafeCell<CpuState>> =
            Box::new(UnsafeCell::new(unsafe { std::mem::zeroed() }));
        // C# ctor: Running = true; SetCounter(MinCountForCheck). Not yet
        // shared with any other thread, so Relaxed is sufficient here.
        // SAFETY: sole owner during construction; project through the cell.
        unsafe {
            let p = cpu.get();
            (*p).running.store(true, Ordering::Relaxed);
            (*p).counter.store(MIN_COUNT_FOR_CHECK, Ordering::Relaxed);
        }

        Self {
            cpu,
            interrupted: AtomicBool::new(false),
            counter_source,
            interrupt_callback,
            break_callback,
            supervisor_callback,
            undefined_callback,
        }
    }

    /// C# NativeContextPtr.
    pub fn native_context_ptr(&self) -> *mut u8 {
        self.cpu_ptr() as *mut u8
    }

    /// Raw pointer to the heap-stable CpuState for interior mutation through
    /// `&self`. The CpuState is already mutated concurrently by JIT'd code
    /// (raw `mov` via the baked native-context pointer) and, for its `running`
    /// / `counter` atomics, by other host threads. We therefore never form a
    /// `&mut CpuState`; the owner thread writes non-atomic fields (registers,
    /// dispatch address, sysregs) through this raw pointer, and a shared
    /// `&self` is the correct access mode.
    #[inline]
    pub(crate) fn cpu_ptr(&self) -> *mut CpuState {
        self.cpu.get()
    }

    /// Narrow shared borrow of the `running` atomic (only those bytes are
    /// retagged, never the whole CpuState).
    #[inline]
    fn running_atomic(&self) -> &AtomicBool {
        // SAFETY: field projection off the cell pointer; see `cpu`.
        unsafe { &(*self.cpu_ptr()).running }
    }

    /// Narrow shared borrow of the `counter` atomic.
    #[inline]
    fn counter_atomic(&self) -> &AtomicI32 {
        // SAFETY: as `running_atomic`.
        unsafe { &(*self.cpu_ptr()).counter }
    }

    // C# static system register values.
    pub const CTR_EL0: u32 = 0x8444c004;
    pub const DCZID_EL0: u32 = 0x00000004;

    pub fn cntfrq_el0(&self) -> u64 {
        self.counter_source.as_ref().map_or(0, |c| c.frequency())
    }
    pub fn cntpct_el0(&self) -> u64 {
        self.counter_source.as_ref().map_or(0, |c| c.counter())
    }
    /// CNTVCT_EL0 = CNTPCT_EL0 (CNTVOFF_EL2 = 0, no EL2).
    pub fn cntvct_el0(&self) -> u64 {
        self.cntpct_el0()
    }

    pub fn get_x(&self, index: usize) -> u64 {
        // SAFETY: narrow read of a single field through the cell pointer.
        unsafe { (*self.cpu_ptr()).x[index] }
    }
    pub fn set_x(&self, index: usize, value: u64) {
        // SAFETY: owner-thread write to a non-atomic CpuState field via the
        // cell pointer; see `cpu`. No `&mut CpuState` is formed.
        unsafe { (*self.cpu_ptr()).x[index] = value };
    }
    pub fn get_v(&self, index: usize) -> V128 {
        // SAFETY: narrow read; see `get_x`.
        unsafe { (*self.cpu_ptr()).v[index] }
    }
    pub fn set_v(&self, index: usize, value: V128) {
        unsafe { (*self.cpu_ptr()).v[index] = value };
    }

    /// C# Running (NativeContext.GetRunning). Acquire so a thread observing
    /// `running == false` also observes the writes the stopping thread made
    /// before `stop_running`.
    pub fn running(&self) -> bool {
        self.running_atomic().load(Ordering::Acquire)
    }

    /// C# CheckInterrupt: fire the pending interrupt callback, then re-arm
    /// the counter. Owner-thread only. `swap` clears the cross-thread flag
    /// and reads the previous value in one step; `Release` on the counter
    /// re-arm publishes the callback's effects.
    pub fn check_interrupt(&self) {
        if self.interrupted.swap(false, Ordering::AcqRel) {
            if let Some(cb) = &self.interrupt_callback {
                cb(self);
            }
        }
        self.counter_atomic()
            .store(MIN_COUNT_FOR_CHECK, Ordering::Release);
    }

    /// C# RequestInterrupt. Called cross-thread, hence `&self`: it only pokes
    /// the atomic flag, observed by the owner at its next synchronization
    /// point.
    pub fn request_interrupt(&self) {
        self.interrupted.store(true, Ordering::Release);
    }

    /// C# OnBreak.
    pub fn on_break(&self, address: u64, imm: i32) {
        if let Some(cb) = &self.break_callback {
            cb(self, address, imm);
        }
    }

    /// C# OnSupervisorCall.
    pub fn on_supervisor_call(&self, address: u64, imm: i32) {
        if let Some(cb) = &self.supervisor_callback {
            cb(self, address, imm);
        }
    }

    /// C# OnUndefined.
    pub fn on_undefined(&self, address: u64, op_code: i32) {
        if let Some(cb) = &self.undefined_callback {
            cb(self, address, op_code);
        }
    }

    /// C# StopRunning: Running = false; counter = 0 (the next
    /// synchronization point exits the dispatch loop). Called cross-thread,
    /// hence `&self`.
    ///
    /// Order matters: publish `running = false` first, then zero `counter`.
    /// A synchronization point that loads `counter == 0` falls into the slow
    /// path and reads `running` with Acquire; pairing the counter's Release
    /// store after the running store guarantees that load sees `false`.
    pub fn stop_running(&self) {
        self.running_atomic().store(false, Ordering::Release);
        self.counter_atomic().store(0, Ordering::Release);
    }

    /// Re-arm for another run after a halt-return exit (Eden/dynarmic-style
    /// host): Running = true, counter re-primed so emit_synchronization fires
    /// again. Needed because stop_running (used to bounce out at an SVC /
    /// preemption) leaves Running = false. Owner thread, between runs.
    pub fn resume_running(&self) {
        self.counter_atomic()
            .store(MIN_COUNT_FOR_CHECK, Ordering::Release);
        self.running_atomic().store(true, Ordering::Release);
    }
}
