// Execute an AArch64 program that computes fib(20) and copies indexed memory.
// Exercises recursive calls, guest stack spills, branches, and recompilation.
//
// Usage: cargo run --example guest_blob

use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::translator::Translator;

const CODE_BASE: u64 = 0x1000;
const GUEST_PAGES: usize = 16;

struct FlatMemory {
    buf: Vec<u128>,
    page_table: Vec<u64>,
}

impl FlatMemory {
    fn new() -> Self {
        let buf = vec![0u128; GUEST_PAGES * 0x1000 / 16];
        let host_base = buf.as_ptr() as u64;
        let page_table = (0..GUEST_PAGES as u64)
            .map(|i| host_base + i * 0x1000)
            .collect();
        Self { buf, page_table }
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
        16
    }
    fn page_table_pointer(&self) -> *mut u8 {
        self.page_table.as_ptr() as *mut u8
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
        (va as usize) < GUEST_PAGES * 0x1000
    }
    fn signal_memory_tracking(&self, _: u64, _: u64, _: bool, _: bool, _: Option<i32>) {}
}

fn run(name: &str, program: &[u32], checks: &[(usize, u64)]) -> bool {
    let mut memory = FlatMemory::new();
    {
        let bytes = memory.bytes_mut();
        for (i, opcode) in program.iter().enumerate() {
            bytes[CODE_BASE as usize + i * 4..CODE_BASE as usize + i * 4 + 4]
                .copy_from_slice(&opcode.to_le_bytes());
        }
    }

    let translator = Translator::new(Box::new(memory));

    let mut state: CpuState = unsafe { std::mem::zeroed() };
    state.counter = std::sync::atomic::AtomicI32::new(1_000_000);
    state.running = std::sync::atomic::AtomicBool::new(true);

    let state_ptr = &mut state as *mut CpuState as *mut u8;
    translator.execute(state_ptr, CODE_BASE);

    let mut ok = true;
    for &(reg, expected) in checks {
        if state.x[reg] != expected {
            eprintln!(
                "{name}: X{reg} = {:#x}, expected {expected:#x}",
                state.x[reg]
            );
            ok = false;
        }
    }
    println!("{name}: {}\n", if ok { "OK" } else { "FAIL" });
    ok
}

fn main() {
    // Real compiler-assembled guest blob (clang --target=aarch64):
    // entry sets up a guest stack, computes recursive fib(20)=6765 via
    // nested BL/RET with stack spills (STP/LDP), then a 100-iteration
    // STR/LDR indexed loop over the data page. Exercises: calls, deep
    // recursion through the dispatch path + rejit (fib entered ~10k
    // times), conditional branches, guest stack memory, scaled
    // addressing.
    let program: &[u32] = &[
        0xD29E0009, 0x9100013F, 0xD2800280, 0x9400000D, 0xAA0003F5, 0xD2840001, 0xD2800002,
        0xD2800C83, 0xF8227823, 0xF8627824, 0x91000442, 0xEB03005F, 0x54FFFF81, 0xAA1503E0,
        0xAA1F03FE, 0xD65F03C0, 0xF100081F, 0x54000042, 0xD65F03C0, 0xD10083FF, 0xA9007BF3,
        0xF9000BF4, 0xAA0003F3, 0xD1000660, 0x97FFFFF8, 0xAA0003F4, 0xD1000A60, 0x97FFFFF5,
        0x8B140000, 0xA9407BF3, 0xF9400BF4, 0x910083FF, 0xD65F03C0,
    ];

    let ok = run(
        "guest-blob-fib20",
        program,
        &[(0, 6765), (2, 100), (4, 100)],
    );

    if ok {
        println!("== guest_blob: OK ==");
    } else {
        println!("== guest_blob: FAILED ==");
        std::process::exit(1);
    }
}
