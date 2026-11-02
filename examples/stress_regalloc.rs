// Keep 20 values live across a loop back-edge to force register spills and edge copies.
// Compare LCQ and HighCq results with the same arithmetic computed in Rust.
//
// Usage: cargo run --release --example stress_regalloc

use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::translator::Translator;

const CODE_BASE: u64 = 0x1000;
const GUEST_PAGES: usize = 16;

// AArch64 encoders (each verified against the passing vector set)
fn movz(rd: u32, imm16: u32) -> u32 {
    0xD280_0000 | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)
}
fn add_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0x8B00_0000 | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F)
}
fn sub_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0xD100_0000 | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
}
fn mov_reg(rd: u32, rm: u32) -> u32 {
    // ORR Xd, XZR, Xm
    0xAA00_0000 | ((rm & 0x1F) << 16) | (31 << 5) | (rd & 0x1F)
}
fn cbnz(rt: u32, off_instrs: i32) -> u32 {
    let imm19 = (off_instrs as u32) & 0x7FFFF;
    0xB500_0000 | (imm19 << 5) | (rt & 0x1F)
}
const RET: u32 = 0xD65F_03C0;

const NVALS: u32 = 20;
const CTR_REG: u32 = NVALS + 1; // loop counter register (above the value range)
const COUNTER: u32 = 25;
const ITERS: u64 = 25;
const MAXV: usize = 31; // array sizing upper bound (X1..X30)

/// Build the guest program; return (program, loop_body_arithmetic closure
/// inputs)  arithmetic is replicated in `ground_truth`.
fn build_program() -> Vec<u32> {
    let mut p: Vec<u32> = Vec::new();

    // init X1..X20 = 1..20
    for i in 1..=NVALS {
        p.push(movz(i, i));
    }
    // counter
    p.push(movz(CTR_REG, COUNTER));

    let loop_start = p.len() as i32;

    // chain: x[i] += x[i+1] for i in 1..20, then x20 += x1
    for i in 1..NVALS {
        p.push(add_reg(i, i, i + 1));
    }
    p.push(add_reg(NVALS, NVALS, 1));

    // counter--
    p.push(sub_imm(CTR_REG, CTR_REG, 1));
    let cbnz_pos = p.len() as i32;
    p.push(cbnz(CTR_REG, loop_start - cbnz_pos)); // back-edge

    // fold all into X0
    p.push(mov_reg(0, 1));
    for i in 2..=NVALS {
        p.push(add_reg(0, 0, i));
    }
    p.push(RET);
    p
}

/// Same arithmetic as the guest program, in Rust, with wrapping adds.
fn ground_truth() -> [u64; MAXV] {
    let mut x = [0u64; MAXV]; // x[1..=20] used
    for i in 1..=NVALS as usize {
        x[i] = i as u64;
    }
    for _ in 0..ITERS {
        for i in 1..NVALS as usize {
            x[i] = x[i].wrapping_add(x[i + 1]);
        }
        x[NVALS as usize] = x[NVALS as usize].wrapping_add(x[1]);
    }
    let mut x0 = x[1];
    for i in 2..=NVALS as usize {
        x0 = x0.wrapping_add(x[i]);
    }
    let mut out = [0u64; MAXV];
    out[0] = x0;
    out[1..=NVALS as usize].copy_from_slice(&x[1..=NVALS as usize]);
    out
}

// FlatMemory (SoftwarePageTable)
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

fn run_tier(program: &[u32], high_cq: bool) -> [u64; 32] {
    let mut memory = FlatMemory::new();
    {
        let bytes = memory.bytes_mut();
        for (i, op) in program.iter().enumerate() {
            bytes[CODE_BASE as usize + i * 4..CODE_BASE as usize + i * 4 + 4]
                .copy_from_slice(&op.to_le_bytes());
        }
    }
    let translator = Translator::new(Box::new(memory));
    let func = translator.translate(CODE_BASE, high_cq);

    let mut state: CpuState = unsafe { std::mem::zeroed() };
    state.counter = std::sync::atomic::AtomicI32::new(1_000_000);
    state.running = std::sync::atomic::AtomicBool::new(true);
    let state_ptr = &mut state as *mut CpuState as *mut u8;
    unsafe { func.execute(state_ptr) };

    let mut regs = [0u64; 32];
    regs.copy_from_slice(&state.x);
    regs
}

fn main() {
    let program = build_program();
    let truth = ground_truth();
    println!(
        "program: {} instrs, {NVALS} live values across loop, {ITERS} iters",
        program.len()
    );

    let lcq = run_tier(&program, false);
    let hcq = run_tier(&program, true);

    let mut lcq_bad = Vec::new();
    let mut hcq_bad = Vec::new();
    for i in 0..=NVALS as usize {
        if lcq[i] != truth[i] {
            lcq_bad.push(format!("X{i}: got {:#x} want {:#x}", lcq[i], truth[i]));
        }
        if hcq[i] != truth[i] {
            hcq_bad.push(format!("X{i}: got {:#x} want {:#x}", hcq[i], truth[i]));
        }
    }

    println!(
        "\n-- LCQ --  {}",
        if lcq_bad.is_empty() {
            "OK".into()
        } else {
            format!("{} WRONG", lcq_bad.len())
        }
    );
    for m in &lcq_bad {
        println!("   {m}");
    }
    println!(
        "-- HighCq -- {}",
        if hcq_bad.is_empty() {
            "OK".into()
        } else {
            format!("{} WRONG", hcq_bad.len())
        }
    );
    for m in &hcq_bad {
        println!("   {m}");
    }

    if lcq_bad.is_empty() && hcq_bad.is_empty() {
        println!("\n== both tiers correct  this shape did not reproduce the bug ==");
    } else {
        println!("\n== REPRODUCED: tier(s) diverge from ground truth ==");
    }
}
