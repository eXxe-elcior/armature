// Check barrier lowering and execution.
// ST/LD barriers need no fence under x86 TSO; full barriers close the store-load hole.
// LDAR/STLR retain their fences. Full barriers must emit mfence, without lock or.
//
// Usage: cargo run --example smoke_barrier

use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::translator::Translator;

const ENTRY: u64 = 0x1000;

// Guest instruction encodings
//
// DMB:  11010101000000110011 CRm 10111111
// DSB:  11010101000000110011 CRm 10011111
// CRm encodes scope in [3:2] and type in [1:0]:
//   type 0b11 = SY (full ordering), 0b01 = ST, 0b10 = LD
const DMB_SY: u32 = 0xD5033FBF; // CRm = 0b1111 → type=SY
const DMB_ST: u32 = 0xD5033EBF; // CRm = 0b1110 → type=ST (stores only)
const DMB_LD: u32 = 0xD5033DBF; // CRm = 0b1101 → type=LD (loads only)
const DSB_SY: u32 = 0xD5033F9F; // CRm = 0b1111 → type=SY
const DSB_ST: u32 = 0xD5033E9F; // CRm = 0b1110 → type=ST (stores only)

// LDAR X0, [X1] (64-bit):  xx001000110xxxxx1xxxxxxxxxxxxxxx
//   size=10, rs=00000, rt2=00000, rn=00001, rt=00000
const LDAR_X0_X1: u32 = 0x88C08000;

// STLR X0, [X1] (64-bit):  xx001000100xxxxx1xxxxxxxxxxxxxxx
//   size=10, rs=00000, rt2=00000, rn=00001, rt=00000
const STLR_X0_X1: u32 = 0x88808000;

const RET: u32 = 0xD65F03C0;

// Two blocks
// Block A: every barrier type exercised, ending with RET.
// Block B: only DMB ST/LD + DSB ST (should emit zero mfence).

const BLOCK_A: &[u32] = &[
    DMB_SY,     // 0x1000  should emit mfence
    DMB_ST,     // 0x1004  should emit NOTHING
    DMB_LD,     // 0x1008  should emit NOTHING
    DSB_SY,     // 0x100c  should emit mfence
    DSB_ST,     // 0x1010  should emit NOTHING
    LDAR_X0_X1, // 0x1014  should emit mfence + mov
    STLR_X0_X1, // 0x1018  should emit mov + mfence
    RET,        // 0x101c
];

const BLOCK_B: &[u32] = &[
    DMB_ST, // 0x1000  should emit NOTHING
    DMB_LD, // 0x1004  should emit NOTHING
    DSB_ST, // 0x1008  should emit NOTHING
    RET,    // 0x100c
];

// Flat memory (HostMapped  supports guest loads/stores)
const GUEST_PAGES: usize = 256;

struct FlatMemory {
    buf: Vec<u128>,
}

impl FlatMemory {
    fn new(code: &[u32]) -> Self {
        let buf = vec![0u128; GUEST_PAGES * 0x1000 / 16];
        let mut fm = Self { buf };
        // Write guest code into the buffer.
        for (i, op) in code.iter().enumerate() {
            let off = ENTRY as usize + i * 4;
            fm.bytes_mut()[off..off + 4].copy_from_slice(&op.to_le_bytes());
        }
        fm
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
        39
    }
    fn page_table_pointer(&self) -> *mut u8 {
        self.buf.as_ptr() as *mut u8
    }
    fn manager_type(&self) -> MemoryManagerType {
        // HostMapped: flat load-mov without page-table walk
        // necessary for LDAR/STLR to execute without a page table.
        MemoryManagerType::HostMapped
    }
    fn read_u32(&self, va: u64) -> u32 {
        u32::from_le_bytes(
            self.bytes()[va as usize..va as usize + 4]
                .try_into()
                .unwrap(),
        )
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
    fn signal_memory_tracking(
        &self,
        _va: u64,
        _size: u64,
        _write: bool,
        _precise: bool,
        _exempt_id: Option<i32>,
    ) {
    }
}

fn count_mfence(code: &[u8]) -> usize {
    code.windows(3)
        .filter(|w| w[0] == 0x0f && w[1] == 0xae && w[2] == 0xf0)
        .count()
}

fn count_lock_or(code: &[u8]) -> usize {
    // lock or   0x83 /0 imm8   or   0x81 /0 imm32
    // Patterns: f0 83 ...  or  f0 81 ...
    code.windows(2)
        .filter(|w| w[0] == 0xf0 && (w[1] == 0x83 || w[1] == 0x81))
        .count()
}

fn test_block(block: &[u32], label: &str, expected_mfence: usize) {
    let memory = FlatMemory::new(block);
    let translator = Translator::new(Box::new(memory));

    // Test both tiers.
    for &hcq in &[false, true] {
        let tier = if hcq { "hcq" } else { "lcq" };
        let func = translator.translate(ENTRY, hcq);

        let n_mfence = count_mfence(&func.code);
        let n_lock_or = count_lock_or(&func.code);

        assert_eq!(
            n_mfence,
            expected_mfence,
            "{label}/{tier}: expected {expected_mfence} mfence, got {n_mfence} (code len {})",
            func.code.len()
        );

        assert_eq!(
            n_lock_or, 0,
            "{label}/{tier}: expected 0 lock or, got {n_lock_or}"
        );

        println!(
            "  [{tier:>4}] {label}: {n_mfence} mfence (expected {expected_mfence}), \
             {n_lock_or} lock or → ✓"
        );
    }
}

fn main() {
    // Block A: 4 mfences (DMB SY + DSB SY + LDAR + STLR)
    test_block(BLOCK_A, "all barrier types", 4);

    // Block B: 0 mfences (all ST/LD  no-ops on TSO)
    test_block(BLOCK_B, "ST/LD only (elided)", 0);

    // Block A execution: verify the compiled code actually runs.
    {
        let memory = FlatMemory::new(BLOCK_A);
        let translator = Translator::new(Box::new(memory));
        let mut state: CpuState = unsafe { std::mem::zeroed() };
        state.counter = std::sync::atomic::AtomicI32::new(100);
        state.x[1] = ENTRY; // X1 → valid code memory for LDAR/STLR
        state.x[30] = 0; // LR = 0 → RET exits dispatch
        let state_ptr = &mut state as *mut CpuState as *mut u8;
        translator.execute(state_ptr, ENTRY);
        println!("  [exec] all barrier types: executed without crash → ✓");
    }

    println!("\nall barrier smoke tests PASSED");
}
