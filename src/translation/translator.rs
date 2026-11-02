// Port of Translation/Translator.cs.
// Compiles and dispatches guest functions, schedules background recompilation,
// and manages the function table and persistent translation cache.

use crate::instructions::context::ArmEmitterCtx;
use crate::ir::block::BlockFrequency;
use crate::ir::translation::TranslationContext;
use crate::ir::types::OperandType;
use crate::memory::MemoryManager;
use crate::translation::block_decoder::{self, DecodedBlocks, DecoderMode};
use crate::translation::cache::TranslatorCache;
use crate::translation::cfg::ControlFlowGraph;
use crate::translation::compiler::{self, CompilerOptions};
use crate::translation::register_usage;
use crate::translation::stubs::TranslatorStubs;
use crate::translation::translated_function::TranslatedFunction;
use crate::util::address_table::AddressTable;

// SAFETY: mutable state is protected by locks or atomics; MemoryManager is
// Send + Sync. Stubs are immutable after construction. Function-table pages
// use release publication for JIT readers. Code and table pointers stay valid
// until guest execution and background workers have stopped. Each address
// space has its own Translator.
unsafe impl Send for Translator {}
unsafe impl Sync for Translator {}
pub struct Translator {
    memory: Box<dyn MemoryManager>,
    stubs: TranslatorStubs,
    /// Guest address → TranslatedFunction (C# TranslatorCache, a
    /// concurrent structure there  RwLock'd here).
    functions: std::sync::RwLock<TranslatorCache>,
    /// Rejit counters: append-only graveyard. Counter addresses are baked
    /// into LCQ code that another guest thread may still be executing
    /// after a rejit replaces the cache entry  never free them early.
    rejit_counters: std::sync::Mutex<Vec<Box<u32>>>,
    /// Evicted-function graveyard. A function's executable mapping may still
    /// be live: its host pointer was handed out as a raw u64 (FunctionTable
    /// direct-link, dispatch return) and may be baked into another function's
    /// code or sitting on the guest call stack. Dropping it would munmap the
    /// mapping out from under those references (use-after-free). Retired
    /// functions are kept alive here until the Translator itself is dropped,
    /// when no guest is running and every mapping is safe to release.
    /// Same rationale as `rejit_counters` above.
    retired_functions: std::sync::Mutex<Vec<TranslatedFunction>>,
    /// Recompilation queue consumed by background workers and managed dispatch.
    rejit_queue: std::sync::Mutex<std::collections::VecDeque<u64>>,
    /// Wakes background rejit workers when the queue gains work / on stop.
    rejit_cv: std::sync::Condvar,
    /// Background HighCq-rejit worker pool (lazily spawned on first enqueue).
    rejit_workers: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// Set on drop to make workers exit; checked under the queue lock.
    rejit_stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Addresses already enqueued/done for rejit  prevents re-enqueue thrash.
    rejitted: std::sync::Mutex<std::collections::HashSet<u64>>,
    /// C# FunctionTable. Mutex guards structural mutation (page alloc);
    /// the JIT-side inline walk reads racily like C# Volatile semantics.
    function_table: std::sync::Mutex<AddressTable>,
    /// Function-table base, cached so resolve_reloc needs no lock.
    /// The dispatch stub reads table geometry directly from AddressTable.
    table_base: u64,
    /// PTC: collected relocatable HighCq translations to persist (None disables
    /// PTC). Path from ARMATURE_PTC. See translation/ptc.rs.
    ptc_path: Option<String>,
    ptc_infos: std::sync::Mutex<Vec<crate::translation::ptc::PtcInfo>>,
    ptc_load_once: std::sync::Once,
    /// Number of guest threads currently inside `execute()`. Teardown drains
    /// this to zero before the host (Ryujinx) unmaps the guest address space,
    /// so no in-flight JIT memory access faults on a vanished/untracked page.
    execute_active: std::sync::atomic::AtomicU64,
}

impl Translator {
    pub fn new(memory: Box<dyn MemoryManager>) -> Self {
        // C#: AddressTable<ulong>.CreateForArm(true, memory.Type)
        // We use a fixed 5-level non-sparse preset (divergence, see
        // address_table.rs). Fill is set to 0 temporarily and updated
        // after stubs are generated.
        let mut table = AddressTable::create_for_arm(
            true,
            memory.manager_type(),
            0, // temporary fill
        );

        let stubs = TranslatorStubs::new(memory.as_ref(), &table);

        // C#: FunctionTable.Fill = (ulong)Stubs.SlowDispatchStub
        let slow_stub = stubs.slow_dispatch_stub();
        table.set_fill(slow_stub);

        let table_base = table.base();
        Self {
            memory,
            stubs,
            functions: std::sync::RwLock::new(TranslatorCache::new()),
            rejit_counters: std::sync::Mutex::new(Vec::new()),
            retired_functions: std::sync::Mutex::new(Vec::new()),
            rejit_queue: std::sync::Mutex::new(std::collections::VecDeque::new()),
            rejit_cv: std::sync::Condvar::new(),
            rejit_workers: std::sync::Mutex::new(Vec::new()),
            rejit_stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            rejitted: std::sync::Mutex::new(std::collections::HashSet::new()),
            function_table: std::sync::Mutex::new(table),
            table_base,
            ptc_path: std::env::var("ARMATURE_PTC").ok().filter(|s| !s.is_empty()),
            ptc_infos: std::sync::Mutex::new(Vec::new()),
            ptc_load_once: std::sync::Once::new(),
            execute_active: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Block until every guest thread has left `execute()`. Called at teardown
    /// (before the host unmaps guest memory). `stop_running` must already have
    /// been issued on every context so the dispatch loops exit promptly; this
    /// just waits for them to actually return. Bounded so a wedged guest thread
    /// can't hang shutdown forever.
    pub fn drain_execute(&self) {
        use std::sync::atomic::Ordering;
        // Stop the background rejit workers FIRST. They decode guest memory in
        // translate(); if they're still running when the caller frees the guest
        // address space (ArmProcessContext.Dispose: DecrementReferenceCount runs
        // right after this), a worker mid-decode reads a freed page → SIGSEGV on
        // STOP. Drop also does this, but Drop runs only at translator_destroy
        // AFTER the memory is unmapped. Idempotent: Drop then joins an empty vec.
        self.stop_rejit_workers();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while self.execute_active.load(Ordering::Acquire) != 0 {
            if std::time::Instant::now() >= deadline {
                break;
            }
            std::thread::yield_now();
        }
    }

    /// Signal the background rejit pool to exit and join it. Safe to call more
    /// than once (the workers vec is drained on the first call).
    fn stop_rejit_workers(&self) {
        self.rejit_stop
            .store(true, std::sync::atomic::Ordering::Release);
        self.rejit_cv.notify_all();
        let handles: Vec<_> = self.rejit_workers.lock().unwrap().drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
    }

    /// C# Translator.Translate (internal)  decode → emit IR → compile.
    /// Returns the function; the caller caches it (C# GetOrTranslate).
    /// PTC reloc resolver: map a baked Symbol to this run's host address.
    /// Shared by PTC load-patch (and a single source of truth for what each
    /// symboled constant means). Every host-absolute address baked into HighCq
    /// code must be covered here, or a reloaded cache jumps into garbage.
    pub fn resolve_reloc(&self, sym: crate::ir::types::Symbol) -> u64 {
        use crate::ir::types::SymbolType;
        match sym.typ {
            // NativeFn pointer, value = NativeFn discriminant (index()).
            SymbolType::DelegateTable => {
                crate::instructions::native_function::NativeFn::from_index(sym.value).ptr() as u64
            }
            // Special runtime pointers (C# Ptc): 1 = page table / memory base,
            // 3 = dispatch stub.
            SymbolType::Special => match sym.value {
                1 => self.memory.page_table_pointer() as u64,
                3 => self.stubs.dispatch_stub(),
                v => panic!("unknown Special reloc symbol value {v}"),
            },
            // Function-table base (only baked inside the stub today, never in
            // cached code  handled for completeness).
            SymbolType::FunctionTable => self.table_base,
            SymbolType::None => panic!("reloc with SymbolType::None"),
        }
    }

    pub fn translate(&self, guest_address: u64, high_cq: bool) -> TranslatedFunction {
        let (compiled, guest_size, rejit_counter) =
            self.compile_function(guest_address, high_cq, false);
        crate::translation::trace::jit_compiled(guest_address, high_cq, compiled.code.len());
        let func = TranslatedFunction::new(compiled.code, guest_size);
        if let Some(counter) = rejit_counter {
            self.rejit_counters.lock().unwrap().push(counter);
        }
        func
    }

    /// Compile a HighCq function with RELOCATABLE codegen for PTC: returns the
    /// raw code bytes + reloc table (NOT mapped). Each RelocEntry's symbol is
    /// resolvable via resolve_reloc on any future run.
    pub fn translate_relocatable(
        &self,
        guest_address: u64,
    ) -> (Vec<u8>, crate::codegen::linking::RelocationInfo, u64) {
        let (compiled, guest_size, _counter) = self.compile_function(guest_address, true, true);
        (compiled.code, compiled.reloc_info, guest_size)
    }

    /// Shared compile path. `relocatable` adds the RELOCATABLE flag so the
    /// assembler records relocs for symboled constants (used by PTC).
    fn compile_function(
        &self,
        guest_address: u64,
        high_cq: bool,
        relocatable: bool,
    ) -> (
        crate::codegen::compiled_function::CompiledFunction,
        u64,
        Option<Box<u32>>,
    ) {
        // C#: if (!context.HighCq) EmitRejitCheck allocates Counter<uint>.
        let mut rejit_counter: Option<Box<u32>> = if high_cq { None } else { Some(Box::new(0)) };
        let dispatch_stub = self.stubs.dispatch_stub();

        // 1. Decode guest code
        let blocks = block_decoder::decode(
            self.memory.as_ref(),
            guest_address,
            high_cq,
            DecoderMode::MultipleBlocks,
        );

        // 2. Emit IR
        // Reuse a per-thread arena set across translations (capacity retained)
        // instead of allocating fresh Vecs every compile  Rust-side win the GC
        // can't match. Given back to the pool after compile.
        let mut ctx = take_pooled_ctx();
        let mut arm_ctx = ArmEmitterCtx::new(&mut ctx, self.memory.as_ref());
        arm_ctx.entry_address = guest_address;
        // LCQ only: raw pointer to the hotness counter, baked into the entry
        // and back-edge rejit checks. Stable Box address, kept alive in the
        // TranslatedFunction. None for HighCq.
        arm_ctx.rejit_counter = rejit_counter.as_deref_mut().map(|c| c as *mut u32);
        arm_ctx.dispatch_stub = dispatch_stub;

        // C#: EmitSynchronization(context)
        arm_ctx.emit_synchronization();

        // C#: if (blocks[0].Address != address) → Branch to entry label
        let first_block = &blocks.arena[blocks.order[0]];
        if first_block.address != guest_address {
            let entry_label = arm_ctx.get_label(guest_address);
            arm_ctx.branch(entry_label);
        }

        // C#: EmitAndGetCFG → walks blocks, emits IR, returns CFG + range
        let (mut cfg, range_start, range_end) =
            emit_and_get_cfg(&mut arm_ctx, &blocks, guest_address);

        let guest_size = range_end - range_start;

        // 3. RegisterUsage pass
        register_usage::run_pass(&mut cfg, &mut ctx);

        // 4. Compile
        let mut opt_bits = if high_cq {
            compiler::compiler_options::HIGH_CQ
        } else {
            compiler::compiler_options::NONE
        };
        if relocatable {
            opt_bits |= compiler::compiler_options::RELOCATABLE;
        }
        let options = CompilerOptions::new(opt_bits);

        let arg_types = vec![OperandType::I64];
        let ret_type = OperandType::I64;

        crate::codegen::x86_64::code_generator::set_dump_fn_entry(guest_address);
        let compiled = compiler::compile(cfg, arg_types, ret_type, options, &mut ctx);

        // ctx is no longer needed; return it to the per-thread pool for reuse.
        give_pooled_ctx(ctx);

        (compiled, guest_size, rejit_counter)
    }

    /// C# GetOrTranslate, returning the host entry point. Also the
    /// NativeInterface.GetFunctionAddress backend  guest code calls back
    /// into this through the dispatch stub mid-execution.
    ///
    /// The `ARMATURE_HIGHCQ` env var overrides per-function high_cq selection:
    /// when set to "1", all translations use the HighCQ pipeline (SSA + LSRA).
    pub fn get_or_translate_ptr(&self, address: u64) -> u64 {
        if let Some(f) = self.functions.read().unwrap().get(address) {
            return f.func_ptr.expect("cached function unmapped").as_ptr() as u64;
        }

        let force_high_cq = std::env::var("ARMATURE_HIGHCQ").ok().as_deref() == Some("1");
        let translated = self.translate(address, force_high_cq);

        // The pointer MUST come from the entry that actually stays in the
        // cache. get_or_add is insert-if-absent: if another thread already
        // published this address, our `translated` is dropped here  and its
        // Drop munmaps its code page. Returning that (now-freed) page's
        // pointer hands a caller a dangling entry point → it jumps into an
        // unmapped page and SIGSEGVs. So take the pointer from the cached
        // (winning) function, never from the local translation.
        let (ptr, code_size, dump_bytes) = {
            let mut functions = self.functions.write().unwrap();
            let cached = functions.get_or_add(address, translated);
            let ptr = cached.func_ptr.expect("cached function unmapped").as_ptr() as u64;
            // Clone the host bytes only when dumping (a few KB, gated) so I can
            // objdump the hot functions offline.
            let dump = if jit_dump_enabled() {
                Some(cached.code.clone())
            } else {
                None
            };
            (ptr, cached.code.len(), dump)
        };
        perf_map_record(ptr, code_size, address, force_high_cq);
        if let Some(bytes) = dump_bytes {
            dump_jit_bytes(ptr, address, force_high_cq, &bytes);
        }

        // C# RegisterFunction: publish into the FunctionTable so the inline
        // dispatch walk finds this address directly. Publish LCQ too (not just
        // HighCq): otherwise every branch to a not-yet-rejitted function misses
        // the table and falls to the slow dispatch stub's native
        // GetFunctionAddress (lock + hashmap) on EVERY call  and most guest
        // functions never reach the rejit threshold, so that slow path
        // dominates. The background rejit worker overwrites the slot with the
        // HighCq pointer later; the old LCQ mapping is retired, not freed.
        {
            let mut table = self.function_table.lock().unwrap();
            if table.is_valid(address) {
                table.set(address, ptr);
            }
        }

        ptr
    }

    /// C# InvalidateJitCacheRegion also clears AddressTable entries.

    /// C# InvalidateJitCacheRegion  evict every translated function whose
    /// guest range overlaps [address, address + size). Also clears the
    /// AddressTable entries. Rejit-queue clearing is not ported (no rejit).
    /// The executable mappings are dropped with the evicted entries.
    pub fn invalidate_jit_cache_region(&self, address: u64, size: u64) {
        let mut functions = self.functions.write().unwrap();
        let mut table = self.function_table.lock().unwrap();
        let fill = table.fill();
        for start in functions.get_overlaps(address, size) {
            // C#: Volatile.Write(ref FunctionTable.GetValue(overlapAddress), FunctionTable.Fill);
            if table.is_valid(start) {
                table.set(start, fill);
            }
            // Retire rather than drop: the mapping may still be live (see
            // `retired_functions`). The table entry above is reset to the
            // slow-dispatch fill so no new direct-link reaches it.
            if let Some(old) = functions.remove(start) {
                crate::translation::trace::jit_evicted(start, "invalidate");
                self.retired_functions.lock().unwrap().push(old);
            }
        }
    }

    /// C# Translator.EnqueueForRejit. A hot LCQ function asks to be promoted to
    /// HighCq. We hand the address to a background worker thread (C# uses a
    /// thread pool) so the guest NEVER blocks on the (often large) HighCq
    /// compile  it keeps running the LCQ code until the worker atomically
    /// swaps in the faster HighCq version via the FunctionTable.
    pub fn enqueue_for_rejit(&self, address: u64) {
        if no_rejit() {
            return;
        }
        // Dedup: a still-live LCQ copy keeps hitting its rejit check and would
        // re-enqueue the same address forever. Rejit each address at most once.
        if !self.rejitted.lock().unwrap().insert(address) {
            return;
        }
        self.ensure_rejit_worker();
        self.rejit_queue.lock().unwrap().push_back(address);
        self.rejit_cv.notify_one();
    }

    /// Spawn the background rejit worker on first use. The worker dereferences a
    /// raw pointer to `self`; this is sound because the Translator lives in a
    /// Box owned by the FFI layer (stable address) and `Drop` stops+joins the
    /// worker before any field is freed. Same raw-self-pointer pattern already
    /// used by `register_thread`.
    fn ensure_rejit_worker(&self) {
        let mut workers = self.rejit_workers.lock().unwrap();
        if !workers.is_empty() {
            return;
        }
        // Size the pool to the host: one worker can't keep up with a game's
        // One worker can't keep up with a game's burst of new functions, so
        // most code would stay stuck on slow LCQ. Default to a pool sized to
        // the host (leave cores for guest + GPU). The multi-worker path is
        // race-free (the AddressTable publishes with release ordering; the
        // earlier "pool crash" was the unrelated cmd-90 applet gap).
        // ARMATURE_REJIT_THREADS overrides.
        // Match ARMeilleure's restraint: Min(4, Max(1, (cores-6)/3)). A big pool
        // of full-priority compile threads contends with the guest + GPU and
        // burns CPU during bursts; ARMeilleure caps at 4 and runs 1-3 typically.
        // ARMATURE_REJIT_THREADS overrides (hard cap 8).
        let default_n = std::thread::available_parallelism()
            .map(|p| (p.get().saturating_sub(6) / 3).clamp(1, 4))
            .unwrap_or(1);
        let n = std::env::var("ARMATURE_REJIT_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(default_n)
            .min(8);
        let this = self as *const Translator as usize;
        for _ in 0..n {
            let stop = self.rejit_stop.clone();
            workers.push(std::thread::spawn(move || {
                // Background compile must yield to guest + GPU threads  run at
                // lowest OS priority so it consumes only otherwise-idle cycles.
                lower_thread_to_idle();
                let translator = unsafe { &*(this as *const Translator) };
                loop {
                    let address = {
                        let mut queue = translator.rejit_queue.lock().unwrap();
                        loop {
                            if stop.load(std::sync::atomic::Ordering::Acquire) {
                                return;
                            }
                            // LIFO: compile the most-recently-hot function first
                            // (matches ARMeilleure's rejit stack) so freshly hot
                            // code reaches HighCq fastest.
                            if let Some(addr) = queue.pop_back() {
                                break addr;
                            }
                            queue = translator.rejit_cv.wait(queue).unwrap();
                        }
                    };
                    translator.rejit_one(address);
                }
            }));
        }

        // PTC periodic flush: persist the growing cache every ~60s instead of
        // only at teardown, so a crash/kill keeps most of the warmup work.
        // Spawned once alongside the workers, joined via the same vec on stop.
        // Only meaningful when capture is enabled (ptc_path set).
        if self.ptc_path.is_some() {
            let stop = self.rejit_stop.clone();
            workers.push(std::thread::spawn(move || {
                lower_thread_to_idle();
                let translator = unsafe { &*(this as *const Translator) };
                // Sleep in 1s steps so stop is observed within a second at
                // teardown rather than blocking the join for a full minute.
                let mut elapsed = 0u32;
                while !stop.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    elapsed += 1;
                    if elapsed >= 60 {
                        elapsed = 0;
                        translator.ptc_save();
                    }
                }
            }));
        }
    }

    /// C# BackgroundTranslate body: compile HighCq, atomically replace the
    /// cached function, publish into the FunctionTable. Runs on the worker.
    fn rejit_one(&self, address: u64) {
        let translated = self.translate(address, true);
        let ptr = translated
            .func_ptr
            .expect("rejitted function unmapped")
            .as_ptr() as u64;
        perf_map_record(ptr, translated.code.len(), address, true);
        if jit_dump_enabled() {
            dump_jit_bytes(ptr, address, true, &translated.code);
        }
        self.publish_translation(address, translated, ptr);

        // PTC capture (option b): when a cache path is set, compile this HighCq
        // function a SECOND time with RELOCATABLE codegen  this copy is NEVER
        // executed, only serialized. The live run keeps the non-relocatable code
        // published above (the relocatable form mis-executes on memory/dispatch
        // functions); the relocatable bytes are verified byte-correct cross-run
        // (examples/ptc_smoke.rs), so they are safe to persist and reload. Runs
        // on the background worker, off the guest hot path.
        if self.ptc_path.is_some() {
            let (code, reloc, guest_size) = self.translate_relocatable(address);
            let info = crate::translation::ptc::PtcInfo::from_translation(
                address, guest_size, code, reloc,
            );
            let mut infos = self.ptc_infos.lock().unwrap();
            // Rejit can re-capture the same address; keep the latest.
            if let Some(slot) = infos.iter_mut().find(|i| i.guest_address == address) {
                *slot = info;
            } else {
                infos.push(info);
            }
        }
    }

    /// Swap a freshly compiled function into the cache + function table (the
    /// LCQ→HighCq promotion point, also used by PTC load).
    fn publish_translation(&self, address: u64, translated: TranslatedFunction, ptr: u64) {
        {
            let mut functions = self.functions.write().unwrap();
            if let Some(old) = functions.remove(address) {
                crate::translation::trace::jit_evicted(address, "rejit");
                self.retired_functions.lock().unwrap().push(old);
            }
            functions.get_or_add(address, translated);
        }
        let mut table = self.function_table.lock().unwrap();
        if table.is_valid(address) {
            table.set(address, ptr);
        }
    }

    /// Load a PTC cache from disk (if ARMATURE_PTC set and the file matches this
    /// build + memory mode): patch each cached function's relocs to this run's
    /// addresses, map it, publish it  skipping recompilation. Call once after
    /// construction (resolve_reloc needs the fully-built Translator).
    pub fn ptc_load(&self) -> usize {
        let Some(path) = &self.ptc_path else { return 0 };
        let Ok(bytes) = std::fs::read(path) else {
            return 0;
        };
        let mem_mode = self.memory.manager_type() as u8;
        let Some(infos) = crate::translation::ptc::deserialize(&bytes, mem_mode) else {
            return 0;
        };
        let mut loaded = 0;
        for info in infos {
            // Patch the baked addresses to this run's via resolve_reloc.
            let mut code = info.code.clone();
            for r in &info.relocs {
                let addr = self.resolve_reloc(r.symbol);
                let pos = r.position as usize;
                code[pos..pos + 8].copy_from_slice(&addr.to_le_bytes());
            }
            let func = TranslatedFunction::new(code, info.guest_size);
            let ptr = func.func_ptr.expect("ptc function unmapped").as_ptr() as u64;
            self.publish_translation(info.guest_address, func, ptr);
            self.ptc_infos.lock().unwrap().push(info);
            loaded += 1;
        }
        loaded
    }

    /// Serialize the collected HighCq translations to the PTC file.
    pub fn ptc_save(&self) {
        let Some(path) = &self.ptc_path else { return };
        // Snapshot + serialize under the lock, then release it before touching
        // the filesystem  don't hold the capture lock across IO.
        let bytes = {
            let infos = self.ptc_infos.lock().unwrap();
            if infos.is_empty() {
                return;
            }
            let mem_mode = self.memory.manager_type() as u8;
            crate::translation::ptc::serialize(&infos, mem_mode)
        };
        // Atomic replace: write a temp then rename, so a crash mid-write can't
        // corrupt or truncate the existing good cache (a partial file would fail
        // deserialize's magic/length checks anyway, but rename keeps the old one).
        let tmp = format!("{path}.tmp");
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    /// C# ExecuteSingle: get or translate, then run through the context
    /// wrapper (FP-mode sync bracket).
    fn execute_single(&self, native_context: *mut u8, address: u64) -> u64 {
        // No Rust borrow may be held across the guest call: the guest
        // calls back into get_or_translate_ptr through the TLS translator
        // pointer and mutates the cache.
        let func_ptr = self.get_or_translate_ptr(address);
        let wrapper = self.stubs.context_wrapper();

        unsafe { wrapper(native_context, func_ptr) }
    }

    /// C# managed dispatch loop:
    ///   do { address = ExecuteSingle(context, address); }
    ///   while (context.Running && address != 0);
    pub fn execute(&self, native_context: *mut u8, mut address: u64) {
        use std::sync::atomic::Ordering;
        // Mark this thread in-flight so teardown waits for it (drain_execute)
        // before the host unmaps guest memory under us.
        self.execute_active.fetch_add(1, Ordering::AcqRel);
        let _guard = ExecuteGuard(&self.execute_active);

        crate::instructions::native_interface::register_thread(
            self.memory.as_ref() as *const dyn MemoryManager,
            self as *const Translator,
        );

        // Load the PTC cache once, on the first guest run (resolve_reloc needs
        // the fully-built Translator + a registered thread). No-op if disabled.
        self.ptc_load_once.call_once(|| {
            let n = self.ptc_load();
            if n > 0 {
                crate::translation::trace::jit_compiled(0, true, n);
            }
        });

        if use_managed_dispatch() {
            // Managed dispatch loop (C# UseUnmanagedDispatchLoop = false).
            // Kept as a fallback / debug path: bounces to Rust each block
            // (RwLock + hashmap per dispatch via get_or_translate_ptr).
            loop {
                // SAFETY: native_context is the CpuState the caller handed us,
                // valid for the duration of this run.
                unsafe { crate::translation::trace::block_enter(address, native_context) };
                address = self.execute_single(native_context, address);
                let running = unsafe {
                    *(native_context.add(crate::registers::CpuState::running_offset())
                        as *const i32)
                };
                if address == 0 || running == 0 {
                    break;
                }
            }
        } else {
            // C#: Stubs.DispatchLoop(context.NativeContextPtr, address).
            // The entire dispatch runs in native code: each iteration calls the
            // dispatch stub (lockless inline AddressTable walk), tailcalls the
            // target, and loops while next != 0 && running  no managed bounce,
            // no per-block RwLock/hashmap. First-seen addresses still compile
            // lazily via the stub's GetFunctionAddress fallback (TLS callback
            // into get_or_translate_ptr).
            let dispatch_loop = self.stubs.dispatch_loop();
            unsafe { dispatch_loop(native_context, address) };
        }

        crate::instructions::native_interface::unregister_thread();
    }
}

/// Decrements `execute_active` when an `execute()` call returns or unwinds.
struct ExecuteGuard<'a>(&'a std::sync::atomic::AtomicU64);
impl Drop for ExecuteGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl Drop for Translator {
    fn drop(&mut self) {
        // Persist the PTC cache before teardown.
        self.ptc_save();
        // Stop the background rejit worker before any field is freed: it holds
        // a raw pointer to `self` and touches these fields. Usually already done
        // by drain_execute (called before guest memory is freed); idempotent.
        self.stop_rejit_workers();

        // All fields drop naturally here. TranslatedFunction::Drop calls
        // jit_unmap → munmap on each JIT code page. The old code used
        // std::mem::forget to avoid use-after-free from stale FunctionTable
        // entries / return-address references  but drain_execute() (called
        // from the C# side before armature_translator_destroy) guarantees
        // every guest thread has left the execution loop, so no live
        // reference to these mappings remains when Drop runs.
    }
}

/// ARMATURE_PERF_MAP=1 → append a perf JIT symbol-map line per translated
/// function to /tmp/perf-<pid>.map, so `perf report` names JIT frames as
/// guest_<addr>_<tier> instead of raw hex. Format: "START SIZE name" (hex).
/// ARMATURE_DUMP_JIT=1 → dump each compiled function's host bytes to
/// /tmp/armature-jit/<host_start_hex>.bin plus an index line, so a hot host
/// address from perf can be matched to a function and objdump'd offline.
fn jit_dump_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ARMATURE_DUMP_JIT").ok().as_deref() == Some("1"))
}

fn dump_jit_bytes(start: u64, guest: u64, hcq: bool, code: &[u8]) {
    use std::io::Write;
    static INDEX: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> = std::sync::OnceLock::new();
    let dir = "/tmp/armature-jit";
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(format!("{dir}/{start:x}.bin"), code);
    let index = INDEX.get_or_init(|| {
        std::sync::Mutex::new(
            std::fs::File::create(format!("{dir}/index.txt")).expect("create jit index"),
        )
    });
    if let Ok(mut f) = index.lock() {
        let tier = if hcq { "hcq" } else { "lcq" };
        let _ = writeln!(f, "{start:x} {:x} guest_{guest:x}_{tier}", code.len());
    }
}

fn perf_map_record(start: u64, size: usize, guest: u64, hcq: bool) {
    use std::io::Write;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("ARMATURE_PERF_MAP").ok().as_deref() == Some("1")) {
        return;
    }
    static FILE: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> = std::sync::OnceLock::new();
    let file = FILE.get_or_init(|| {
        let path = format!("/tmp/perf-{}.map", std::process::id());
        std::sync::Mutex::new(std::fs::File::create(path).expect("create perf map"))
    });
    if let Ok(mut f) = file.lock() {
        let tier = if hcq { "hcq" } else { "lcq" };
        let _ = writeln!(f, "{start:x} {size:x} guest_{guest:x}_{tier}");
    }
}

// Per-thread pool of one TranslationContext, reused across translations so the
// IR arenas (operations/operands/blocks + pools) keep their capacity instead of
// reallocating every compile. Each translate() runs to completion on its own
// thread (worker or guest) with no re-entrancy, so a single slot per thread is
// enough. mem::take leaves an empty Default in the slot during the (brief, by
// construction non-reentrant) window; give_pooled_ctx stores the grown one back.
thread_local! {
    static CTX_POOL: std::cell::RefCell<TranslationContext> =
        std::cell::RefCell::new(TranslationContext::new());
}

fn take_pooled_ctx() -> TranslationContext {
    CTX_POOL.with(|cell| {
        let mut ctx = std::mem::take(&mut *cell.borrow_mut());
        ctx.reset();
        ctx
    })
}

fn give_pooled_ctx(ctx: TranslationContext) {
    CTX_POOL.with(|cell| *cell.borrow_mut() = ctx);
}

/// Drop the calling (background-compile) thread to the lowest OS scheduling
/// priority so HighCq compilation never preempts guest or GPU threads  it
/// consumes only otherwise-idle CPU. Best-effort: failure (restricted env) is
/// ignored. FFI to the always-linked C runtime / Win32  no crate added.
#[cfg(target_os = "linux")]
fn lower_thread_to_idle() {
    // setpriority(PRIO_PROCESS, 0, 19): on Linux `nice` is per-thread and who=0
    // targets the calling thread. nice 19 = lowest CFS weight (yields under
    // contention, still runs full speed on idle cores).
    unsafe extern "C" {
        fn setpriority(which: i32, who: u32, prio: i32) -> i32;
    }
    const PRIO_PROCESS: i32 = 0;
    unsafe {
        let _ = setpriority(PRIO_PROCESS, 0, 19);
    }
}

#[cfg(target_os = "windows")]
fn lower_thread_to_idle() {
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(thread: isize, priority: i32) -> i32;
    }
    const THREAD_PRIORITY_LOWEST: i32 = -2;
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn lower_thread_to_idle() {}

/// ARMATURE_NO_REJIT=1 keeps every function at its initial LCQ tier (the HighCq
/// pipeline is never exercised). Read once.
fn no_rejit() -> bool {
    static NO_REJIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *NO_REJIT.get_or_init(|| std::env::var("ARMATURE_NO_REJIT").ok().as_deref() == Some("1"))
}

/// ARMATURE_MANAGED_DISPATCH=1 forces the old managed (Rust) dispatch loop.
/// Default is the unmanaged native dispatch loop (C# UseUnmanagedDispatchLoop =
/// true)  the steady-state win. Read once.
fn use_managed_dispatch() -> bool {
    static MANAGED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MANAGED.get_or_init(|| std::env::var("ARMATURE_MANAGED_DISPATCH").ok().as_deref() == Some("1"))
}

/// Port of C# EmitAndGetCFG (Translator.cs local function). Walks decoded
/// blocks, emits IR, returns the CFG plus the guest address range covered
/// by non-exit blocks. AArch32 / IfThenBlock / EnableDebugging paths
/// excluded; EmitRejitCheck skipped (no rejit  see header note).
fn emit_and_get_cfg(
    ctx: &mut ArmEmitterCtx,
    blocks: &DecodedBlocks,
    entry_address: u64,
) -> (ControlFlowGraph, u64, u64) {
    let mut range_start = u64::MAX;
    let mut range_end = 0u64;

    for &blk_idx in &blocks.order {
        let block = &blocks.arena[blk_idx];

        if !block.exit {
            if range_start > block.address {
                range_start = block.address;
            }
            if range_end < block.end_address {
                range_end = block.end_address;
            }
        }

        // C#: if (block.Address == context.EntryAddress) →
        // if (!HighCq) EmitRejitCheck + ClearQcFlag.
        if block.address == entry_address {
            ctx.emit_rejit_count();
            ctx.clear_qc_flag();
        }

        // C#: context.CurrBlock = block
        ctx.reset_block_state();

        // C#: MarkLabel(GetLabel(block.Address))
        let label = ctx.get_label(block.address);
        ctx.mark_label(label, BlockFrequency::Default);

        if block.exit {
            // C#: EmitVirtualJump(context, Const(block.Address),
            // isReturn: useReturns) with useReturns = EnableDebugging = false.
            let addr = ctx.ctx.operands.constant_i64(block.address as i64);
            crate::instructions::flow_helpers::emit_virtual_jump(ctx, addr, false);
        } else {
            for op_idx in 0..block.op_codes.len() {
                let decoded = &block.op_codes[op_idx];
                let is_last_op = op_idx == block.op_codes.len() - 1;

                // C#: context.CurrOp = opCode
                ctx.set_curr_op(*decoded);

                if is_last_op {
                    ctx.sync_qc_flag();

                    // C#: backward branches re-check synchronization so
                    // guest spin loops stay interruptible.
                    if let Some(branch_idx) = block.branch {
                        let branch = &blocks.arena[branch_idx];
                        if !branch.exit && branch.address <= block.address {
                            ctx.emit_synchronization();
                        }
                    }
                }

                // C#: opCode.Instruction.Emitter(context)
                let emit_fn = decoded.info.emit;
                emit_fn(ctx);
            }
        }
    }

    // C#: return context.GetControlFlowGraph()
    (ctx.get_control_flow_graph(), range_start, range_end)
}
