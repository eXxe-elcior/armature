// PTC relocation smoke test (offline, no game).
//
// Translate the SAME guest function on two independent Translators (T1, T2).
// Their stubs / page table / etc. live at different host addresses, so the two
// compiled blobs differ ONLY at baked host-absolute addresses (the relocs).
// Patch T1's blob using T2's resolve_reloc → it must become byte-identical to
// T2's own blob. If any baked host address lacks a reloc, T1's (un-patched)
// address stays where T2's differs → mismatch → relocation is incomplete.
use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::ptc;
use armature::translation::translator::Translator;

const CODE_BASE: u64 = 0x1000;

struct Flat {
    buf: Vec<u128>,
    pt: Vec<u64>,
}
impl Flat {
    fn new() -> Self {
        let buf = vec![0u128; 16 * 0x1000 / 16];
        let base = buf.as_ptr() as u64;
        let pt = (0..16u64).map(|i| base + i * 0x1000).collect();
        Self { buf, pt }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut u8, self.buf.len() * 16)
        }
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.buf.len() * 16) }
    }
}
impl MemoryManager for Flat {
    fn address_space_bits(&self) -> i32 {
        16
    }
    fn page_table_pointer(&self) -> *mut u8 {
        self.pt.as_ptr() as *mut u8
    }
    fn manager_type(&self) -> MemoryManagerType {
        MemoryManagerType::SoftwarePageTable
    }
    fn read_u8(&self, va: u64) -> u8 {
        self.bytes()[va as usize]
    }
    fn read_u16(&self, va: u64) -> u16 {
        u16::from_le_bytes(
            self.bytes()[va as usize..va as usize + 2]
                .try_into()
                .unwrap(),
        )
    }
    fn read_u32(&self, va: u64) -> u32 {
        u32::from_le_bytes(
            self.bytes()[va as usize..va as usize + 4]
                .try_into()
                .unwrap(),
        )
    }
    fn read_u64(&self, va: u64) -> u64 {
        u64::from_le_bytes(
            self.bytes()[va as usize..va as usize + 8]
                .try_into()
                .unwrap(),
        )
    }
    fn read_v128(&self, va: u64) -> V128 {
        u128::from_le_bytes(
            self.bytes()[va as usize..va as usize + 16]
                .try_into()
                .unwrap(),
        )
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
    fn write_u8(&self, va: u64, v: u8) {
        unsafe { *(self.buf.as_ptr() as *mut u8).add(va as usize) = v }
    }
    fn write_u16(&self, va: u64, v: u16) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(va as usize),
                2,
            )
        }
    }
    fn write_u32(&self, va: u64, v: u32) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(va as usize),
                4,
            )
        }
    }
    fn write_u64(&self, va: u64, v: u64) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(va as usize),
                8,
            )
        }
    }
    fn write_v128(&self, va: u64, v: V128) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(va as usize),
                16,
            )
        }
    }
    fn is_mapped(&self, va: u64) -> bool {
        (va as usize) < 16 * 0x1000
    }
    fn signal_memory_tracking(&self, _: u64, _: u64, _: bool, _: bool, _: Option<i32>) {}
}

fn load_prog(mem: &mut Flat, prog: &[u32]) {
    let b = mem.bytes_mut();
    for (i, op) in prog.iter().enumerate() {
        b[CODE_BASE as usize + i * 4..CODE_BASE as usize + i * 4 + 4]
            .copy_from_slice(&op.to_le_bytes());
    }
}

fn main() {
    // movz x0,#0x42 ; ldr x1,[x0] ; b #0x2000 (external -> table branch -> dispatch stub reloc)
    // ldr x1,[x0] -> software page-table path; movz/ldr give register + memory relocs as applicable.
    let prog: Vec<u32> = vec![
        0xD2800840,         // movz x0, #0x42
        0xF9400001,         // ldr  x1, [x0]
        0x14000000 | 0x3FE, // b    #(0x2000) from 0x1008
    ];

    let mut m1 = Flat::new();
    load_prog(&mut m1, &prog);
    let mut m2 = Flat::new();
    load_prog(&mut m2, &prog);
    let t1 = Translator::new(Box::new(m1));
    let t2 = Translator::new(Box::new(m2));

    let (code1, reloc1, _sz1) = t1.translate_relocatable(CODE_BASE);
    let (code2, _reloc2, _sz2) = t2.translate_relocatable(CODE_BASE);

    println!(
        "code1 {} bytes, code2 {} bytes, {} relocs",
        code1.len(),
        code2.len(),
        reloc1.entries().len()
    );
    assert_eq!(code1.len(), code2.len(), "blobs differ in length");
    assert!(
        !reloc1.entries().is_empty(),
        "no relocs collected  RELOCATABLE not working"
    );

    // Patch a copy of code1 using T2's symbol addresses.
    let mut patched = code1.clone();
    for e in reloc1.entries() {
        let addr = t2.resolve_reloc(e.symbol);
        let pos = e.position as usize;
        patched[pos..pos + 8].copy_from_slice(&addr.to_le_bytes());
    }

    // Where code1 and code2 differ must be exactly the reloc positions; after
    // patching with T2's addresses, patched must equal code2.
    let mismatches: Vec<usize> = (0..code2.len())
        .filter(|&i| patched[i] != code2[i])
        .collect();
    if mismatches.is_empty() {
        println!(
            "PASS test1: relocation reconstructs T2's code from T1's bytes  all baked host addresses are relocatable"
        );
    } else {
        println!(
            "FAIL test1: {} byte(s) differ after patch (a baked host address with no reloc):",
            mismatches.len()
        );
        for &i in mismatches.iter().take(16) {
            println!(
                "  off {i}: patched={:#04x} code2={:#04x}",
                patched[i], code2[i]
            );
        }
        std::process::exit(1);
    }

    // ── test2: full serialize → deserialize → patch → map → EXECUTE round-trip
    // on a fresh translator, for a self-contained function (movz x0,#0x42; ret).
    let prog2: Vec<u32> = vec![0xD2800840 /*movz x0,#0x42*/, 0xD65F03C0 /*ret*/];
    let exec = |t: &Translator| -> u64 {
        let ptr = t.get_or_translate_ptr(CODE_BASE);
        let mut st: CpuState = unsafe { std::mem::zeroed() };
        st.counter = std::sync::atomic::AtomicI32::new(1_000_000);
        st.running = std::sync::atomic::AtomicBool::new(true);
        let f: extern "C" fn(*mut u8) -> u64 = unsafe { std::mem::transmute(ptr as *const ()) };
        f(&mut st as *mut CpuState as *mut u8);
        st.x[0]
    };

    let mut ma = Flat::new();
    load_prog(&mut ma, &prog2);
    let ta = Translator::new(Box::new(ma));
    let result_a = exec(&ta); // baseline live execution

    let (codea, reloca, sza) = ta.translate_relocatable(CODE_BASE);
    let info = ptc::PtcInfo::from_translation(CODE_BASE, sza, codea, reloca);
    let bytes = ptc::serialize(
        std::slice::from_ref(&info),
        MemoryManagerType::SoftwarePageTable as u8,
    );

    // "Next run": fresh translator (different stub/mem addresses), load the cache.
    let infos = ptc::deserialize(&bytes, MemoryManagerType::SoftwarePageTable as u8)
        .expect("deserialize failed");
    let mut mb = Flat::new();
    load_prog(&mut mb, &prog2);
    let tb = Translator::new(Box::new(mb));
    let mut loaded_code = infos[0].code.clone();
    for r in &infos[0].relocs {
        let addr = tb.resolve_reloc(r.symbol);
        let p = r.position as usize;
        loaded_code[p..p + 8].copy_from_slice(&addr.to_le_bytes());
    }
    // Map + execute the reconstructed code in tb's context.
    use armature::translation::translated_function::TranslatedFunction;
    let func = TranslatedFunction::new(loaded_code, infos[0].guest_size);
    let fptr = func.func_ptr.expect("mapped").as_ptr();
    let mut st: CpuState = unsafe { std::mem::zeroed() };
    st.counter = std::sync::atomic::AtomicI32::new(1_000_000);
    st.running = std::sync::atomic::AtomicBool::new(true);
    let f: extern "C" fn(*mut u8) -> u64 = unsafe { std::mem::transmute(fptr as *const ()) };
    f(&mut st as *mut CpuState as *mut u8);
    let result_b = st.x[0];

    println!(
        "test2: live X0={:#x}, reloaded+patched X0={:#x}",
        result_a, result_b
    );
    if result_a == result_b && result_a == 0x42 {
        println!(
            "PASS test2: serialized → reloaded → patched code executes correctly on a fresh run"
        );
    } else {
        println!("FAIL test2: reloaded execution diverged");
        std::process::exit(1);
    }

    // ── test3: EXECUTE a relocatable function that touches memory (str+ldr) and
    // compare to the non-relocatable build. Reproduces the game crash offline if
    // RELOCATABLE codegen mis-executes memory accesses.
    use armature::translation::translated_function::TranslatedFunction as TF;
    // movz x0,#0x200; movz x1,#0x77; str x1,[x0]; ldr x0,[x0]; ret
    let prog3: Vec<u32> = vec![
        0xD2804000, // movz x0,#0x200
        0xD2800EE1, // movz x1,#0x77
        0xF9000001, // str  x1,[x0]
        0xF9400000, // ldr  x0,[x0]
        0xD65F03C0, // ret
    ];
    let run = |code: Vec<u8>, sz: u64| -> u64 {
        let f = TF::new(code, sz);
        let p = f.func_ptr.expect("mapped").as_ptr();
        let mut st: CpuState = unsafe { std::mem::zeroed() };
        st.counter = std::sync::atomic::AtomicI32::new(1_000_000);
        st.running = std::sync::atomic::AtomicBool::new(true);
        let g: extern "C" fn(*mut u8) -> u64 = unsafe { std::mem::transmute(p as *const ()) };
        g(&mut st as *mut CpuState as *mut u8);
        st.x[0]
    };
    let mut mc = Flat::new();
    load_prog(&mut mc, &prog3);
    let tc = Translator::new(Box::new(mc));
    // non-relocatable (translate true) result:
    let nonrel_ptr = tc.get_or_translate_ptr(CODE_BASE);
    let mut st: CpuState = unsafe { std::mem::zeroed() };
    st.counter = std::sync::atomic::AtomicI32::new(1_000_000);
    st.running = std::sync::atomic::AtomicBool::new(true);
    let gf: extern "C" fn(*mut u8) -> u64 = unsafe { std::mem::transmute(nonrel_ptr as *const ()) };
    gf(&mut st as *mut CpuState as *mut u8);
    let nonrel = st.x[0];
    // relocatable result (map the relocatable bytes, execute live  real addrs baked):
    let (rcode, _rreloc, rsz) = tc.translate_relocatable(CODE_BASE);
    let rel = run(rcode, rsz);
    println!(
        "test3: non-relocatable X0={:#x}, relocatable X0={:#x} (expect 0x77)",
        nonrel, rel
    );
    if nonrel == 0x77 && rel == 0x77 {
        println!("PASS test3: relocatable code with memory access executes correctly");
    } else {
        println!("FAIL test3: relocatable memory codegen diverges (this is the game crash)");
        std::process::exit(1);
    }
}
