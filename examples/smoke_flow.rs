// Check loops, BL/RET, indirect branches, BLR X30, and recompilation.
// Expected register values are derived from the guest instructions below.
//
// Usage: cargo run --example smoke_flow

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
    let mut all = true;

    // 1. Backward branch: sum 10..1 via CBNZ loop. X1 = 55, X0 = 0.
    all &= run(
        "cbnz-loop",
        &[
            0xD2800140, // 0x1000 MOVZ X0, #10
            0xD2800001, // 0x1004 MOVZ X1, #0
            0x8B000021, // 0x1008 ADD  X1, X1, X0
            0xD1000400, // 0x100c SUB  X0, X0, #1
            0xB5FFFFC0, // 0x1010 CBNZ X0, 0x1008
            0xD65F03C0, // 0x1014 RET            (LR = 0 → exit)
        ],
        &[(1, 55), (0, 0)],
    );

    // 2. Minimal BL: call callee, callee sets X0=42 and RETs back to
    //    0x1004; caller clears LR and exits. (A bare RET at 0x1004 would
    //    jump to X30 = 0x1004  itself  forever; BL leaves LR set.)
    //    X0 = 42.
    all &= run(
        "bl-minimal",
        &[
            0x94000003, // 0x1000 BL   0x100c     (X30 = 0x1004)
            0xD280001E, // 0x1004 MOVZ X30, #0
            0xD65F03C0, // 0x1008 RET             (X30 = 0 → exit)
            0xD2800540, // 0x100c MOVZ X0, #42
            0xD65F03C0, // 0x1010 RET             (X30 = 0x1004 → continue)
        ],
        &[(0, 42)],
    );

    // 3. BL / RET call-and-continue. Callee adds 1, caller continues at
    //    the instruction after BL, adds 100, clears LR, exits.
    //    X0 = 5 + 1 + 100 = 106.
    all &= run(
        "bl-ret-continue",
        &[
            0xD28000A0, // 0x1000 MOVZ X0, #5
            0x94000004, // 0x1004 BL   0x1014
            0x91019000, // 0x1008 ADD  X0, X0, #100
            0xD280001E, // 0x100c MOVZ X30, #0
            0xD65F03C0, // 0x1010 RET            (LR = 0 → exit)
            0x91000400, // 0x1014 ADD  X0, X0, #1
            0xD65F03C0, // 0x1018 RET            (→ 0x1008, caller continues)
        ],
        &[(0, 106)],
    );

    // 4. Indirect BR skips a MOVZ. X0 = 7 (not 99), then exit via LR = 0.
    all &= run(
        "br-indirect",
        &[
            0xD2820202, // 0x1000 MOVZ X2, #0x1010
            0xD61F0040, // 0x1004 BR   X2
            0xD2800C60, // 0x1008 MOVZ X0, #99   (skipped)
            0xD65F03C0, // 0x100c RET
            0xD28000E0, // 0x1010 MOVZ X0, #7
            0xD65F03C0, // 0x1014 RET            (LR = 0 → exit)
        ],
        &[(0, 7), (2, 0x1010)],
    );

    // 5. BLR X30: target must be snapshotted before the LR write
    //    (C# context.Copy). X6 = 9 proves the callee ran; X5 = 1 proves
    //    the caller continued.
    all &= run(
        "blr-x30-snapshot",
        &[
            0xD282029E, // 0x1000 MOVZ X30, #0x1014
            0xD63F03C0, // 0x1004 BLR  X30       (LR ← 0x1008, call 0x1014)
            0xD2800025, // 0x1008 MOVZ X5, #1
            0xD280001E, // 0x100c MOVZ X30, #0
            0xD65F03C0, // 0x1010 RET            (LR = 0 → exit)
            0xD2800126, // 0x1014 MOVZ X6, #9
            0xD65F03C0, // 0x1018 RET            (→ 0x1008)
        ],
        &[(5, 1), (6, 9), (30, 0)],
    );

    // 6. Rejit: callee invoked 150 times via BL  crosses the 100-call
    //    threshold, gets enqueued, retranslated HighCq mid-loop and
    //    published into the function table. X0 = 150 either way.
    all &= run(
        "rejit-150-calls",
        &[
            0xD28012C1, // 0x1000 MOVZ X1, #150
            0x94000005, // 0x1004 BL   0x1018
            0xD1000421, // 0x1008 SUB  X1, X1, #1
            0xB5FFFFC1, // 0x100c CBNZ X1, 0x1004
            0xD280001E, // 0x1010 MOVZ X30, #0
            0xD65F03C0, // 0x1014 RET             (X30 = 0 → exit)
            0x91000400, // 0x1018 ADD  X0, X0, #1
            0xD65F03C0, // 0x101c RET
        ],
        &[(0, 150), (1, 0)],
    );

    if all {
        println!("== smoke_flow: all programs OK ==\n");
    } else {
        println!("== smoke_flow: FAILURES ==\n");
        std::process::exit(1);
    }
}
