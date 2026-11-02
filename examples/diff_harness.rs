// Replay golden vectors and compare register and memory state with ARMeilleure.
//
// Usage: cargo run --example diff_harness -- vectors.golden

use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::translator::Translator;

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x2000;
const GUEST_PAGES: usize = 16; // 64 KiB guest space → 16-entry flat PT

// Guest memory backing: 16 guest pages mirrored in one 16-aligned host
// buffer (Vec<u128> for cmpxchg16b alignment), plus the flat page table
// the JIT-generated code walks (entry = host address of the page).
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
        16 // 64 KiB  matches the flat PT above
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
        let b = self.bytes();
        u16::from_le_bytes(b[va as usize..va as usize + 2].try_into().unwrap())
    }
    fn read_u32(&self, va: u64) -> u32 {
        let b = self.bytes();
        u32::from_le_bytes(b[va as usize..va as usize + 4].try_into().unwrap())
    }
    fn read_u64(&self, va: u64) -> u64 {
        let b = self.bytes();
        u64::from_le_bytes(b[va as usize..va as usize + 8].try_into().unwrap())
    }
    fn read_v128(&self, va: u64) -> V128 {
        let b = self.bytes();
        u128::from_le_bytes(b[va as usize..va as usize + 16].try_into().unwrap())
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
    // The harness keeps a raw alias of the buffer for hashing; writes go
    // through the same backing storage (scratch driver, single thread).
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

// Same xorshift64* stream as the oracle's data-page fill.
fn fill_data_page(bytes: &mut [u8], dseed: u64) {
    let mut s = if dseed == 0 {
        0x9E3779B97F4A7C15
    } else {
        dseed
    };
    let base = DATA_BASE as usize;
    for off in (0..0x1000).step_by(8) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let v = s.wrapping_mul(0x2545F4914F6CDD1D);
        bytes[base + off..base + off + 8].copy_from_slice(&v.to_le_bytes());
    }
}

// FNV-1a 64 over the data page  must match the oracle.
fn data_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xCBF29CE484222325;
    for &b in &bytes[DATA_BASE as usize..DATA_BASE as usize + 0x1000] {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001B3);
    }
    hash
}

fn parse_hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap_or_else(|_| panic!("bad hex field: {s}"))
}

fn nzcv_of(state: &CpuState) -> u32 {
    // PState bit positions: N=31, Z=30, C=29, V=28; flags array is one
    // u32 per flag index.
    (state.flags[31] & 1) << 31
        | (state.flags[30] & 1) << 30
        | (state.flags[29] & 1) << 29
        | (state.flags[28] & 1) << 28
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: diff_harness <vectors.golden>");
    let data = std::fs::read_to_string(&path).expect("read golden file");

    let mut total = 0usize;
    let mut failed = 0usize;

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (spec, golden) = line.split_once('|').expect("golden line missing '|'");
        let sf: Vec<&str> = spec.split_whitespace().collect();
        let gf: Vec<&str> = golden.split_whitespace().collect();
        assert_eq!(sf.len(), 12, "spec fields");
        assert_eq!(
            gf.len(),
            66,
            "golden fields (x0..x30, sp, nzcv, dhash, v0..v31)"
        );

        let opcode = parse_hex(sf[0]) as u32;
        let in_nzcv = parse_hex(sf[6]) as u32;
        let dseed = parse_hex(sf[7]);

        let mut memory = FlatMemory::new();
        {
            let bytes = memory.bytes_mut();
            bytes[CODE_BASE as usize..CODE_BASE as usize + 4]
                .copy_from_slice(&opcode.to_le_bytes());
            bytes[CODE_BASE as usize + 4..CODE_BASE as usize + 8]
                .copy_from_slice(&0xd65f03c0u32.to_le_bytes());
            fill_data_page(bytes, dseed);
        }

        // Buffer address is stable across the move into the Translator;
        // keep a raw view for post-run hashing (scratch driver).
        let buf_ptr = memory.buf.as_ptr() as *const u8;
        let buf_len = memory.buf.len() * 16;

        let translator = Translator::new(Box::new(memory));

        if std::env::var("ARMATURE_DUMP").is_ok() {
            let high_cq = std::env::var("ARMATURE_HIGHCQ").ok().as_deref() == Some("1");
            let f = translator.translate(CODE_BASE, high_cq);
            eprintln!("opcode {opcode:08x}: {} bytes", f.code.len());
            for chunk in f.code.chunks(16) {
                eprintln!(
                    "  {}",
                    chunk
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }

        let mut state: CpuState = unsafe { std::mem::zeroed() };
        state.x[0] = parse_hex(sf[1]);
        state.x[1] = parse_hex(sf[2]);
        state.x[2] = parse_hex(sf[3]);
        state.x[3] = parse_hex(sf[4]);
        state.x[31] = parse_hex(sf[5]);
        state.flags[31] = (in_nzcv >> 31) & 1;
        state.flags[30] = (in_nzcv >> 30) & 1;
        state.flags[29] = (in_nzcv >> 29) & 1;
        state.flags[28] = (in_nzcv >> 28) & 1;
        state.counter = std::sync::atomic::AtomicI32::new(1_000_000);
        state.running = std::sync::atomic::AtomicBool::new(true);
        for i in 0..4 {
            state.v[i] = u128::from_str_radix(sf[8 + i], 16).expect("v field");
        }

        let state_ptr = &mut state as *mut CpuState as *mut u8;
        translator.execute(state_ptr, CODE_BASE);

        total += 1;

        let mut mismatches: Vec<String> = Vec::new();

        for i in 0..31 {
            let expected = parse_hex(gf[i]);
            if state.x[i] != expected {
                mismatches.push(format!("X{i}: got {:x} want {expected:x}", state.x[i]));
            }
        }
        let expected_sp = parse_hex(gf[31]);
        if state.x[31] != expected_sp {
            mismatches.push(format!("SP: got {:x} want {expected_sp:x}", state.x[31]));
        }
        let expected_nzcv = parse_hex(gf[32]) as u32;
        let got_nzcv = nzcv_of(&state);
        if got_nzcv != expected_nzcv {
            mismatches.push(format!("NZCV: got {got_nzcv:08x} want {expected_nzcv:08x}"));
        }

        for i in 0..32 {
            let expected_v = u128::from_str_radix(gf[34 + i], 16).expect("v golden");
            if state.v[i] != expected_v {
                mismatches.push(format!(
                    "V{i}: got {:032x} want {expected_v:032x}",
                    state.v[i]
                ));
            }
        }

        let expected_dhash = parse_hex(gf[33]);
        let got_dhash = data_hash(unsafe { std::slice::from_raw_parts(buf_ptr, buf_len) });
        if got_dhash != expected_dhash {
            mismatches.push(format!(
                "DATA: hash got {got_dhash:x} want {expected_dhash:x}"
            ));
        }

        if !mismatches.is_empty() {
            failed += 1;
            println!(
                "FAIL {opcode:08x}  in: x0={} x1={} x2={} x3={} sp={} nzcv={}",
                sf[1], sf[2], sf[3], sf[4], sf[5], sf[6]
            );
            for m in &mismatches {
                println!("    {m}");
            }
        }
    }

    println!(
        "== diff: {}/{total} passed, {failed} failed ==",
        total - failed
    );
    if failed > 0 {
        std::process::exit(1);
    }
}
