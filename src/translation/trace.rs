// Execution tracer / lightweight debugger for the JIT.
//
// Not a port of any C# file  a Rust-side observability layer over the
// managed dispatch loop and the JIT cache. Entirely env-gated: when
// disabled the hot path is a single relaxed atomic load and returns.
//
// Enable with the ARMATURE_TRACE env var (read once, on first touch):
//   ARMATURE_TRACE=1        block-level PC trace
//   ARMATURE_TRACE=blocks   same as 1
//   ARMATURE_TRACE=regs     block PC + changed general registers each step
//   ARMATURE_TRACE=jit      JIT compile/evict events only
//   ARMATURE_TRACE=all      blocks + regs + jit
// Comma-combine freely, e.g. ARMATURE_TRACE=blocks,jit
//
// Output goes to stderr unless ARMATURE_TRACE_FILE=<path> is set.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::registers::CpuState;

/// Fast-path gate. Stays false (and the whole module is a no-op) unless
/// ARMATURE_TRACE selected at least one channel. Set once by `config()`.
static ANY: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Default)]
struct Channels {
    blocks: bool,
    regs: bool,
    jit: bool,
}

struct Config {
    ch: Channels,
    sink: Mutex<Box<dyn Write + Send>>,
    /// ARMATURE_TRACE_ADDR=<hex>: when set, block_enter only logs blocks whose
    /// pc is in [addr-0x80, addr+0x800] (the target fn + neighbours), dumping
    /// the FULL register file each time so a stuck decision value is visible.
    target: Option<(u64, u64)>,
    /// ARMATURE_TRACE_CAP (default 4000): stop emitting after this many
    /// windowed blocks, so the file stays small for a tight spin.
    cap: usize,
}

static WINDOW_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn config() -> &'static Config {
    static CFG: OnceLock<Config> = OnceLock::new();
    CFG.get_or_init(|| {
        let spec = std::env::var("ARMATURE_TRACE").unwrap_or_default();
        let mut ch = Channels::default();
        for tok in spec.split(',').map(str::trim) {
            match tok {
                "1" | "blocks" => ch.blocks = true,
                "regs" => {
                    ch.blocks = true;
                    ch.regs = true;
                }
                "jit" => ch.jit = true,
                "all" => {
                    ch.blocks = true;
                    ch.regs = true;
                    ch.jit = true;
                }
                _ => {}
            }
        }
        let on = ch.blocks || ch.regs || ch.jit;
        ANY.store(on, Ordering::Relaxed);

        let sink: Box<dyn Write + Send> = match std::env::var("ARMATURE_TRACE_FILE") {
            Ok(path) if !path.is_empty() => match std::fs::File::create(&path) {
                Ok(f) => Box::new(std::io::BufWriter::new(f)),
                Err(e) => {
                    eprintln!("[trace] cannot open {path}: {e}; falling back to stderr");
                    Box::new(std::io::stderr())
                }
            },
            _ => Box::new(std::io::stderr()),
        };

        let target = std::env::var("ARMATURE_TRACE_ADDR").ok().and_then(|s| {
            u64::from_str_radix(s.trim().trim_start_matches("0x"), 16)
                .ok()
                .map(|a| (a.saturating_sub(0x80), a + 0x800))
        });
        let cap = std::env::var("ARMATURE_TRACE_CAP")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4000);

        Config {
            ch,
            sink: Mutex::new(sink),
            target,
            cap,
        }
    })
}

/// True if any trace channel is active. Cheap; safe to call in hot paths.
#[inline]
pub fn enabled() -> bool {
    // First call forces config() so the gate reflects the env. After that
    // the relaxed load is all the hot path pays.
    if ANY.load(Ordering::Relaxed) {
        return true;
    }
    // Lazy one-time init: only reached while still false.
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        config();
    });
    ANY.load(Ordering::Relaxed)
}

fn emit(args: std::fmt::Arguments) {
    let cfg = config();
    if let Ok(mut sink) = cfg.sink.lock() {
        let _ = sink.write_fmt(args);
        let _ = sink.flush();
    }
}

// Per-thread previous register snapshot, for the regs channel diff.
thread_local! {
    static PREV: std::cell::RefCell<Option<[u64; 31]>> = const { std::cell::RefCell::new(None) };
}

/// SAFETY: `native_context` must point to a valid CpuState (same contract
/// as the guest call it precedes).
unsafe fn read_x(native_context: *const u8) -> [u64; 31] {
    // CpuState.x is at offset 0; read X0..=X30 (31 GPRs; index 31 is XZR/SP,
    // not a stored slot we trace here).
    let base = native_context as *const u64;
    let mut out = [0u64; 31];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = unsafe { base.add(i).read() };
    }
    out
}

/// Called at the top of each managed-dispatch iteration, before the guest
/// block at `pc` runs. Logs the PC and, on the regs channel, registers that
/// changed since the previous block.
///
/// SAFETY: `native_context` must point to a valid CpuState.
#[inline]
pub unsafe fn block_enter(pc: u64, native_context: *const u8) {
    if !enabled() {
        return;
    }
    let cfg = config();
    let ch = cfg.ch;
    if !ch.blocks {
        return;
    }

    // Address-windowed mode: log ONLY the target fn's blocks, full reg file,
    // capped  for pinpointing a stuck spin/decision value.
    if let Some((lo, hi)) = cfg.target {
        if pc < lo || pc > hi {
            return;
        }
        let n = WINDOW_HITS.fetch_add(1, Ordering::Relaxed);
        if n >= cfg.cap {
            return;
        }
        let cur = unsafe { read_x(native_context) };
        emit(format_args!("[trace] pc={pc:#018x}\n"));
        for (i, v) in cur.iter().enumerate() {
            emit(format_args!("[trace]   x{i}={v:#018x}\n"));
        }
        return;
    }

    emit(format_args!("[trace] pc={pc:#018x}\n"));

    if ch.regs {
        let cur = unsafe { read_x(native_context) };
        PREV.with(|p| {
            let mut p = p.borrow_mut();
            match *p {
                None => {
                    // First block: dump the full register file.
                    for (i, v) in cur.iter().enumerate() {
                        emit(format_args!("[trace]   x{i}={v:#018x}\n"));
                    }
                }
                Some(prev) => {
                    let mut any = false;
                    for (i, (&new, &old)) in cur.iter().zip(prev.iter()).enumerate() {
                        if new != old {
                            emit(format_args!("[trace]   x{i}: {old:#018x} -> {new:#018x}\n"));
                            any = true;
                        }
                    }
                    if !any {
                        emit(format_args!("[trace]   (no register changes)\n"));
                    }
                }
            }
            *p = Some(cur);
        });
    }
}

/// Called when a guest function is freshly compiled.
pub fn jit_compiled(address: u64, high_cq: bool, code_len: usize) {
    if !enabled() {
        return;
    }
    if config().ch.jit {
        let tier = if high_cq { "HighCq" } else { "Lcq" };
        emit(format_args!(
            "[trace] jit compiled addr={address:#018x} tier={tier} bytes={code_len}\n"
        ));
    }
}

/// Called when a translated function is evicted from the cache (rejit
/// replacement or region invalidation) and retired to the graveyard.
pub fn jit_evicted(address: u64, reason: &str) {
    if !enabled() {
        return;
    }
    if config().ch.jit {
        emit(format_args!(
            "[trace] jit evicted  addr={address:#018x} reason={reason}\n"
        ));
    }
}

// Compile-time assurance the offset assumption holds: CpuState.x is at
// offset 0, so reading native_context as *const u64 yields X0 first.
const _: () = assert!(std::mem::offset_of!(CpuState, x) == 0);
