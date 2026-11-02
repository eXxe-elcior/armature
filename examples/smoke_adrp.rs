// ADRP / ADR / ALU-immediate smoke test.
// Exercises ADRP page-offset computation, ADR pc-relative addressing,
// and ALU immediate operations  the shared decoder + emitter path used
// by both LCQ and HIGHCQ pipelines.
//
// Format: each test is a small guest program at CODE_BASE (0x1000). The
// program runs to completion (LR=0 → RET exits). Final register state is
// checked against hand-computed expected values.
//
// Usage: cargo run --example smoke_adrp

#![allow(unused)]
use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::translator::Translator;

const CODE_BASE: u64 = 0x1000;
const GUEST_PAGES: usize = 16;

// Encoding helpers
// ADRP X|Rd|, #|imm_pages|
//
// Encoding: 1 immlo[1:0] 10000 immhi[18:0] Rd[4:0]
// The 21-bit signed page offset: value = immhi:immlo (units: 4 KB pages).
fn adrp(rd: u8, imm_pages: i64) -> u32 {
    let v = imm_pages;
    let immlo = (v & 3) as u32;
    let immhi = ((v >> 2) & 0x7_FFFF) as u32;
    0x9000_0000 | (immlo << 29) | (immhi << 5) | (rd as u32)
}

// ADR X|Rd|, #|imm_bytes|
//
// Encoding: 0 immlo[1:0] 10000 immhi[18:0] Rd[4:0]
// 21-bit signed byte offset: value = immhi:immlo.
fn adr(rd: u8, imm_bytes: i64) -> u32 {
    let v = imm_bytes;
    let immlo = (v & 3) as u32;
    let immhi = ((v >> 2) & 0x7_FFFF) as u32;
    0x1000_0000 | (immlo << 29) | (immhi << 5) | (rd as u32)
}

// ADD X|Rd|, X|Rn|, #|imm12|
//
// Encoding: sf=1 0010001 0 sh imm12 Rn Rd = 0x91000000 base
fn add_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x9100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

// SUB X|Rd|, X|Rn|, #|imm12|
//
// Encoding: sf=1 1010001 0 sh imm12 Rn Rd = 0xD1000000 base
fn sub_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0xD100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

// ORR X|Rd|, X|Rn|, X|Rm| (shifted register, shift=0, N=0)
// Encoding: 1 0 1 0 1 0 1 0 0 0 Rm imm6 Rn Rd
fn orr_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xAA00_0000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

// AND X|Rd|, X|Rn|, X|Rm| (shifted register, shift=0, N=0)
fn and_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x8A00_0000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

const MOVZ_X30_0: u32 = 0xD280_001E; // MOVZ X30, #0
const RET: u32 = 0xD65F_03C0; // RET

// Memory manager
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

// Test runner
fn run(name: &str, program: &[u32], initial: &[(usize, u64)], checks: &[(usize, u64)]) -> bool {
    let mut memory = FlatMemory::new();
    {
        let bytes = memory.bytes_mut();
        for (i, opcode) in program.iter().enumerate() {
            bytes[CODE_BASE as usize + i * 4..CODE_BASE as usize + i * 4 + 4]
                .copy_from_slice(&opcode.to_le_bytes());
        }
    }

    let mut translator = Translator::new(Box::new(memory));

    let mut state: CpuState = unsafe { std::mem::zeroed() };
    state.counter = std::sync::atomic::AtomicI32::new(1_000_000);
    state.running = std::sync::atomic::AtomicBool::new(true);
    for &(reg, val) in initial {
        state.x[reg] = val;
    }

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
    if ok {
        println!("  {name}: OK");
    }
    ok
}

// Programs
// HostTracked memory manager (flat offset page table)
// Used by the actual Ryujinx integration. The PTE stores the translation
// offset (host_addr - guest_addr), NOT the base address.
struct HostTrackedMemory {
    buf: Vec<u128>,
    page_table_offsets: Vec<u64>,
}

impl HostTrackedMemory {
    fn new() -> Self {
        let buf = vec![0u128; GUEST_PAGES * 0x1000 / 16];
        let host_base = buf.as_ptr() as u64;
        // For 1:1 mapping, translation offset = host_base for all pages.
        // Guest VA → host VA: host_base + (va & 0xFFF) = va + host_base
        let page_table_offsets = (0..GUEST_PAGES as u64).map(|_| host_base).collect();
        Self {
            buf,
            page_table_offsets,
        }
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.buf.len() * 16) }
    }
}

impl MemoryManager for HostTrackedMemory {
    fn address_space_bits(&self) -> i32 {
        16
    }
    fn page_table_pointer(&self) -> *mut u8 {
        self.page_table_offsets.as_ptr() as *mut u8
    }
    fn manager_type(&self) -> MemoryManagerType {
        MemoryManagerType::HostTracked
    }
    fn read_u8(&self, va: u64) -> u8 {
        self.bytes()[(va & 0xFFFF) as usize]
    }
    fn read_u16(&self, va: u64) -> u16 {
        u16::from_le_bytes(
            self.bytes()[(va & 0xFFFF) as usize..(va & 0xFFFF) as usize + 2]
                .try_into()
                .unwrap(),
        )
    }
    fn read_u32(&self, va: u64) -> u32 {
        u32::from_le_bytes(
            self.bytes()[(va & 0xFFFF) as usize..(va & 0xFFFF) as usize + 4]
                .try_into()
                .unwrap(),
        )
    }
    fn read_u64(&self, va: u64) -> u64 {
        u64::from_le_bytes(
            self.bytes()[(va & 0xFFFF) as usize..(va & 0xFFFF) as usize + 8]
                .try_into()
                .unwrap(),
        )
    }
    fn read_v128(&self, va: u64) -> V128 {
        u128::from_le_bytes(
            self.bytes()[(va & 0xFFFF) as usize..(va & 0xFFFF) as usize + 16]
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
        let idx = (va & 0xFFFF) as usize;
        unsafe { *(self.buf.as_ptr() as *mut u8).add(idx) = v }
    }
    fn write_u16(&self, va: u64, v: u16) {
        let idx = (va & 0xFFFF) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(idx),
                2,
            )
        }
    }
    fn write_u32(&self, va: u64, v: u32) {
        let idx = (va & 0xFFFF) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(idx),
                4,
            )
        }
    }
    fn write_u64(&self, va: u64, v: u64) {
        let idx = (va & 0xFFFF) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(idx),
                8,
            )
        }
    }
    fn write_v128(&self, va: u64, v: V128) {
        let idx = (va & 0xFFFF) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                v.to_le_bytes().as_ptr(),
                (self.buf.as_ptr() as *mut u8).add(idx),
                16,
            )
        }
    }
    fn is_mapped(&self, va: u64) -> bool {
        (va as usize) < GUEST_PAGES * 0x1000
    }
    fn signal_memory_tracking(&self, _: u64, _: u64, _: bool, _: bool, _: Option<i32>) {}
}

fn run_host_tracked(name: &str, program: &[u32], checks: &[(usize, u64)]) -> bool {
    let mut memory = HostTrackedMemory::new();
    {
        let buf_ptr = memory.buf.as_ptr() as *mut u8;
        for (i, opcode) in program.iter().enumerate() {
            let va = CODE_BASE as usize + i * 4;
            let idx = va & 0xFFFF;
            unsafe {
                std::ptr::copy_nonoverlapping(opcode.to_le_bytes().as_ptr(), buf_ptr.add(idx), 4)
            };
        }
    }

    let mut translator = Translator::new(Box::new(memory));

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
    if ok {
        println!("  {name}: OK");
    }
    ok
}

fn main() {
    let mut all = true;

    // ADRP tests (SoftwarePageTable)
    // ADRP X0, #0: X0 = (0x1000 & ~0xfff) + 0 = 0x1000
    all &= run(
        "adrp-imm0",
        &[adrp(0, 0), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x1000)],
    );

    // ADRP X0, #1: X0 = 0x1000 + 0x1000 = 0x2000
    all &= run(
        "adrp-imm+1",
        &[adrp(0, 1), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x2000)],
    );

    // ADRP X0, #2: X0 = 0x1000 + 0x2000 = 0x3000
    all &= run(
        "adrp-imm+2",
        &[adrp(0, 2), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x3000)],
    );

    // ADRP X5, #0: different destination register
    all &= run(
        "adrp-rd5",
        &[adrp(5, 0), MOVZ_X30_0, RET],
        &[],
        &[(5, 0x1000)],
    );

    // ADRP X0, #7: 0x1000 + 0x7000 = 0x8000
    all &= run(
        "adrp-imm+7",
        &[adrp(0, 7), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x8000)],
    );

    // ADRP X0, #-1 (one page backward): X0 = 0x1000 - 0x1000 = 0x0000
    all &= run(
        "adrp-imm-1",
        &[adrp(0, -1), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x0000)],
    );

    // ADRP X1, #-2 (two pages backward): X1 = 0x1000 - 0x2000 = 0xFFFFFFFFFFFFF000
    // (32-bit sign-extended for 16-bit address space: wraps to 0xFFF0)
    all &= run(
        "adrp-imm-2-wrapped",
        &[adrp(1, -2), MOVZ_X30_0, RET],
        &[],
        &[(1, 0xFFFFFFFFFFFFF000u64)],
    );

    // ADR tests
    // ADR X0, #0: X0 = 0x1000 + 0 = 0x1000
    all &= run(
        "adr-imm0",
        &[adr(0, 0), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x1000)],
    );

    // ADR X0, #8: X0 = 0x1000 + 8 = 0x1008
    all &= run(
        "adr-imm+8",
        &[adr(0, 8), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x1008)],
    );

    // ADR X0, #-4: X0 = 0x1000 - 4 = 0xFFC
    all &= run(
        "adr-imm-4",
        &[adr(0, -4), MOVZ_X30_0, RET],
        &[],
        &[(0, 0xFFC)],
    );

    // ADR X2, #0x100: X2 = 0x1000 + 0x100 = 0x1100
    all &= run(
        "adr-imm+256",
        &[adr(2, 0x100), MOVZ_X30_0, RET],
        &[],
        &[(2, 0x1100)],
    );

    // ADRP + ADD combo (the common "load symbol" pattern)
    // ADRP X0, #1; ADD X0, X0, #0x100
    //   X0 = (0x1000 & ~0xfff) + 0x1000 + 0x100 = 0x2100
    all &= run(
        "adrp+add-lo12",
        &[adrp(0, 1), add_imm(0, 0, 0x100), MOVZ_X30_0, RET],
        &[],
        &[(0, 0x2100)],
    );

    // ADRP X5, #3; ADD X5, X5, #0xABC
    //   X5 = (0x1000 & ~0xfff) + 0x3000 + 0xABC = 0x4ABC
    all &= run(
        "adrp+add-lo12-2",
        &[adrp(5, 3), add_imm(5, 5, 0xABC), MOVZ_X30_0, RET],
        &[],
        &[(5, 0x4ABC)],
    );

    // ADD immediate
    // ADD X0, X1, #42 (0x2A): X0 = X1 + 42
    all &= run(
        "add-imm-x1+42",
        &[add_imm(0, 1, 42), MOVZ_X30_0, RET],
        &[(1, 0x100)], // X1 starts at 0x100
        &[(0, 0x12A)],
    );

    // ADD X0, XZR, #42 → X0 = 0 + 42 = 42
    all &= run(
        "add-imm-xzr+42",
        &[add_imm(0, 31, 42), MOVZ_X30_0, RET],
        &[],
        &[(0, 42)],
    );

    // ADD X0, X0, #0 (no-op): X0 = 0 + 0 = 0
    all &= run(
        "add-imm-x0+0",
        &[add_imm(0, 0, 0), MOVZ_X30_0, RET],
        &[],
        &[(0, 0)],
    );

    // SUB immediate
    // SUB X0, X1, #42: X0 = 0x200 - 42 = 0x1D6
    all &= run(
        "sub-imm",
        &[sub_imm(0, 1, 42), MOVZ_X30_0, RET],
        &[(1, 0x200)],
        &[(0, 0x1D6)],
    );

    // SUB X0, X0, #1 (decrement): 0 - 1 = -1 (wraps)
    all &= run(
        "sub-imm-decr",
        &[sub_imm(0, 0, 1), MOVZ_X30_0, RET],
        &[],
        &[(0, 0xFFFFFFFFFFFFFFFFu64)],
    );

    // Register ALU
    // ORR X0, X1, X2: X0 = X1 | X2 = 0x0F0F | 0xF0F0 = 0xFFFF
    all &= run(
        "orr-reg",
        &[orr_reg(0, 1, 2), MOVZ_X30_0, RET],
        &[(1, 0x0F0F), (2, 0xF0F0)],
        &[(0, 0xFFFF)],
    );

    // AND X0, X1, X2: X0 = X1 & X2 = 0xFF00 & 0x0FF0 = 0x0F00
    all &= run(
        "and-reg",
        &[and_reg(0, 1, 2), MOVZ_X30_0, RET],
        &[(1, 0xFF00), (2, 0x0FF0)],
        &[(0, 0x0F00)],
    );

    // HostTracked tests (same programs, different memory manager)
    // ADRP X0, #0 via HostTracked
    all &= run_host_tracked(
        "ht-adrp-imm0",
        &[adrp(0, 0), MOVZ_X30_0, RET],
        &[(0, 0x1000)],
    );

    // ADRP X0, #1 via HostTracked
    all &= run_host_tracked(
        "ht-adrp-imm+1",
        &[adrp(0, 1), MOVZ_X30_0, RET],
        &[(0, 0x2000)],
    );

    // ADRP X0, #-1 via HostTracked
    all &= run_host_tracked(
        "ht-adrp-imm-1",
        &[adrp(0, -1), MOVZ_X30_0, RET],
        &[(0, 0x0000)],
    );

    // ADRP + ADD via HostTracked
    all &= run_host_tracked(
        "ht-adrp+add",
        &[adrp(0, 1), add_imm(0, 0, 0x100), MOVZ_X30_0, RET],
        &[(0, 0x2100)],
    );

    // ADD X0, XZR, #42 via HostTracked
    all &= run_host_tracked(
        "ht-add-imm",
        &[add_imm(0, 31, 42), MOVZ_X30_0, RET],
        &[(0, 42)],
    );

    // Results
    println!();
    if all {
        println!("== smoke_adrp: ALL programs OK ==");
    } else {
        println!("== smoke_adrp: FAILURES ==");
        std::process::exit(1);
    }
}
