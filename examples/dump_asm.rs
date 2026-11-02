// Compile a guest loop through LCQ and HighCq, then disassemble with objdump.
// Compare with tools/asmdump; pass "hcq soft" to use the same memory mode.
//
// Usage: cargo run --example dump_asm -- [both|lcq|hcq] [soft|host]
// Requires objdump (binutils) on PATH.

use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::translator::Translator;
use std::io::Write;
use std::process::Command;

const CODE_BASE: u64 = 0x1000;
const GUEST_PAGES: usize = 256;

struct FlatMemory {
    buf: Vec<u128>,
    page_table: Vec<u64>,
    mgr_type: MemoryManagerType,
}

impl FlatMemory {
    fn new(mgr_type: MemoryManagerType) -> Self {
        let buf = vec![0u128; GUEST_PAGES * 0x1000 / 16];
        let host_base = buf.as_ptr() as u64;
        let page_table = (0..GUEST_PAGES as u64)
            .map(|i| host_base + i * 0x1000)
            .collect();
        Self {
            buf,
            page_table,
            mgr_type,
        }
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

impl MemoryManager for FlatMemory {
    fn address_space_bits(&self) -> i32 {
        // HostMapped masks the va to AS bits before adding the flat base; the
        // game's mirror is a 39-bit AS. SoftwarePageTable only uses the 16-bit
        // harness table, but the value is harmless there.
        if self.mgr_type == MemoryManagerType::SoftwarePageTable {
            16
        } else {
            39
        }
    }
    fn page_table_pointer(&self) -> *mut u8 {
        self.page_table.as_ptr() as *mut u8
    }
    fn manager_type(&self) -> MemoryManagerType {
        self.mgr_type
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
        (va as usize) < GUEST_PAGES * 0x1000
    }
    fn signal_memory_tracking(&self, _: u64, _: u64, _: bool, _: bool, _: Option<i32>) {}
}

// Representative integer hot loop: load-accumulate with a flag-setting
// decrement and a conditional back-edge (exercises memory load, ALU, NZCV
// flags, the backward-branch synchronization check, and the return).
//
//   0x1000 loop: ldr  x3, [x1]      F9400023
//   0x1004       add  x1, x1, #8    91002021
//   0x1008       add  x0, x0, x3    8B030000
//   0x100c       subs x2, x2, #1    F1000442
//   0x1010       b.ne loop          54FFFF81   (offset -4 insns)
//   0x1014       ret                D65F03C0
const ACCUM_LOOP: &[u32] = &[
    0xF9400023, 0x91002021, 0x8B030000, 0xF1000442, 0x54FFFF81, 0xD65F03C0,
];

// Pure-register vec4 FP accumulate (no memory): fmul v0.4s,v1.4s,v2.4s;
// fadd v3.4s,v3.4s,v0.4s; subs x2,x2,#1; b.ne loop; ret. Representative of a
// 3D game's hot FP work; memory-mode-independent. Select with DUMP_FP=1.
const FP_LOOP: &[u32] = &[0x4E22DC20, 0x4E20D463, 0xF1000442, 0x54FFFFA1, 0xD65F03C0];

fn objdump(code: &[u8], tier: &str) {
    let path = format!("/tmp/armature-dump-{tier}.bin");
    {
        let mut f = std::fs::File::create(&path).expect("create dump bin");
        f.write_all(code).expect("write dump bin");
    }
    println!(
        "\n==================== {tier} ({} bytes) ====================",
        code.len()
    );
    let out = Command::new("objdump")
        .args([
            "-D",
            "-b",
            "binary",
            "-m",
            "i386:x86-64",
            "-M",
            "intel",
            "--no-show-raw-insn",
            &path,
        ])
        .output();
    match out {
        Ok(o) => {
            print!("{}", String::from_utf8_lossy(&o.stdout));
            if !o.stderr.is_empty() {
                eprint!("{}", String::from_utf8_lossy(&o.stderr));
            }
        }
        Err(e) => eprintln!("objdump failed ({e}); raw bytes at {path}"),
    }
}

// High register-pressure straight-line function: define x1..x20 (all live),
// then K rounds of `add x0,x0,xr` re-reading each  21 simultaneously-live
// values > allocatable regs → forces spills/splits, exercising LSRA.
fn gen_pressure(k: usize) -> Vec<u32> {
    let mut v = Vec::new();
    for r in 1..=20u32 {
        v.push(0x91000000 | (r << 10) | (0 << 5) | r); // add xr, x0, #r
    }
    for _ in 0..k {
        for r in 1..=20u32 {
            v.push(0x8B000000 | (r << 16) | (0 << 5) | 0); // add x0, x0, xr
        }
    }
    v.push(0xD65F03C0); // ret
    v
}

fn run_bench() {
    for k in [40usize, 160, 640, 2560] {
        let code = gen_pressure(k);
        let mut memory = FlatMemory::new(MemoryManagerType::HostMapped);
        {
            let bytes = memory.bytes_mut();
            for (i, op) in code.iter().enumerate() {
                let off = CODE_BASE as usize + i * 4;
                bytes[off..off + 4].copy_from_slice(&op.to_le_bytes());
            }
        }
        let translator = Translator::new(Box::new(memory));
        let t = std::time::Instant::now();
        let f = translator.translate(CODE_BASE, true);
        let el = t.elapsed();
        eprintln!(
            "K={k:>5} guest_ops={:>6} -> hcq {:>6} bytes  total_compile={:?}",
            code.len(),
            f.code.len(),
            el
        );
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "both".into());
    if which == "bench" {
        run_bench();
        return;
    }
    let do_lcq = which == "both" || which == "lcq";
    let do_hcq = which == "both" || which == "hcq";

    // Default to HostMapped  the memory mode real games run under (flat 2-instr
    // load). Pass "soft" as the 2nd arg to dump the SoftwarePageTable walk.
    let mgr = match std::env::args().nth(2).as_deref() {
        Some("soft") => MemoryManagerType::SoftwarePageTable,
        _ => MemoryManagerType::HostMapped,
    };
    let fp = std::env::var("DUMP_FP").as_deref() == Ok("1");
    let block: &[u32] = if fp { FP_LOOP } else { ACCUM_LOOP };
    println!(
        "guest: {} | memory: {mgr:?}",
        if fp {
            "vec4 FP accumulate loop"
        } else {
            "integer load-accumulate loop"
        }
    );

    let mut memory = FlatMemory::new(mgr);
    {
        let bytes = memory.bytes_mut();
        for (i, op) in block.iter().enumerate() {
            bytes[CODE_BASE as usize + i * 4..CODE_BASE as usize + i * 4 + 4]
                .copy_from_slice(&op.to_le_bytes());
        }
    }

    let translator = Translator::new(Box::new(memory));

    // translate() decodes + compiles without executing; no exec context needed.
    if do_lcq {
        let f = translator.translate(CODE_BASE, false);
        objdump(&f.code, "lcq");
    }
    if do_hcq {
        let f = translator.translate(CODE_BASE, true);
        objdump(&f.code, "hcq");
    }

    // Keep a CpuState type reference so the layout this code targets is the one
    // we link against (and to silence unused-import lints in trimmed builds).
    let _ = std::mem::size_of::<CpuState>();
}
