//! Miri-targeted soundness checks for the cross-thread ExecutionContext
//! access model (atomics + UnsafeCell interior mutation). These exercise the
//! pure-Rust concurrency surface only  no JIT'd code is executed, so they run
//! under `cargo +nightly miri test`.
//!
//! The real system shares the context across host threads via a raw
//! `*mut ExecutionContext` over the FFI boundary; we reproduce that with a
//! Send wrapper rather than `&` (the context is intentionally `!Sync`).

use armature::{ExceptionCallbackNoArgs, ExecutionContext};

/// Send wrapper carrying real pointer provenance across the thread boundary
/// (so Miri's Stacked-Borrows tracking is not weakened by an int-to-ptr cast),
/// mirroring the FFI `*mut ExecutionContext` handle.
#[derive(Clone, Copy)]
struct CtxPtr(*const ExecutionContext);
// SAFETY: the pointee outlives both threads; the only cross-thread mutation is
// through atomics / UnsafeCell field projection.
unsafe impl Send for CtxPtr {}

/// Owner thread mutates guest registers + polls `running` while another thread
/// concurrently interrupts and stops it  the dispatch-loop vs scheduler race.
#[test]
fn cross_thread_stop_and_mutate() {
    let ctx = Box::new(ExecutionContext::new(None, None, None, None, None));
    let raw = CtxPtr(Box::into_raw(ctx) as *const ExecutionContext);

    let owner = std::thread::spawn(move || {
        let raw = raw; // capture the whole wrapper, not the inner field
        let ec = unsafe { &*raw.0 };
        for i in 0..50_000u64 {
            ec.set_x((i % 31) as usize, i); // UnsafeCell write, owner-only field
            let _ = ec.get_x((i % 31) as usize); // narrow read
            ec.check_interrupt(); // atomic swap + counter store
            if !ec.running() {
                // atomic Acquire load
                break;
            }
        }
    });

    let other = std::thread::spawn(move || {
        let raw = raw;
        let ec = unsafe { &*raw.0 };
        ec.request_interrupt(); // atomic store
        ec.stop_running(); // atomic stores: running then counter
    });

    owner.join().unwrap();
    other.join().unwrap();

    // SAFETY: both threads joined; sole owner again.
    unsafe { drop(Box::from_raw(raw.0 as *mut ExecutionContext)) };
}

/// A callback mutates a guest register through `&ExecutionContext`  the exact
/// residual the UnsafeCell wrap makes sound (write behind a shared ref).
#[test]
fn callback_mutates_via_shared_ref() {
    let cb: ExceptionCallbackNoArgs = Box::new(|ec| ec.set_x(0, 0xdead_beef));
    let ctx = ExecutionContext::new(None, Some(cb), None, None, None);

    ctx.request_interrupt();
    ctx.check_interrupt(); // fires the callback through &self

    assert_eq!(ctx.get_x(0), 0xdead_beef);
}
