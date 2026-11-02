// Persistent Translation Cache (PTC): serialize compiled HighCq functions to
// disk and reload them next run, skipping recompilation (warmup).
//
// Relocation is what makes this sound: JIT code bakes absolute host addresses
// (dispatch stub, page table, NativeFn pointers) that change every run. Each
// cached function carries a reloc table; on load we patch those positions with
// the current run's addresses via Translator::resolve_reloc. Validated offline
// by examples/ptc_smoke.rs (translate on two translators, patch one to the
// other, bytes match).
//
// Only HighCq functions are cached  they have no per-function rejit counter
// (LCQ-only), so there is no per-run Box pointer to relocate.

use crate::codegen::linking::{RelocationEntry, RelocationInfo};
use crate::ir::types::{Symbol, SymbolType};

/// Bump on ANY change that alters generated code or this format, so a stale
/// cache from an older build is rejected instead of miscompiling.
const PTC_VERSION: u64 = 2;
const PTC_MAGIC: &[u8; 8] = b"ARMPTC\x00\x01";

/// One cached function: code baked with the producing run's addresses, plus the
/// reloc table to re-point it to any future run.
pub struct PtcInfo {
    pub guest_address: u64,
    pub guest_size: u64,
    pub code: Vec<u8>,
    pub relocs: Vec<RelocationEntry>,
}

impl PtcInfo {
    pub fn from_translation(
        guest_address: u64,
        guest_size: u64,
        code: Vec<u8>,
        reloc: RelocationInfo,
    ) -> Self {
        Self {
            guest_address,
            guest_size,
            code,
            relocs: reloc.entries().to_vec(),
        }
    }
}

// Binary serialization (hand-rolled, no external crate)
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.p..self.p + n)?;
        self.p += n;
        Some(s)
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
}

fn sym_type_to_u8(t: SymbolType) -> u8 {
    match t {
        SymbolType::None => 0,
        SymbolType::DelegateTable => 1,
        SymbolType::FunctionTable => 2,
        SymbolType::Special => 3,
    }
}
fn u8_to_sym_type(v: u8) -> Option<SymbolType> {
    Some(match v {
        0 => SymbolType::None,
        1 => SymbolType::DelegateTable,
        2 => SymbolType::FunctionTable,
        3 => SymbolType::Special,
        _ => return None,
    })
}

/// Serialize the whole cache. `mem_mode` is the MemoryManagerType discriminant
///  a cache produced under one memory model must not load under another (the
/// memory-access codegen differs).
pub fn serialize(infos: &[PtcInfo], mem_mode: u8) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(PTC_MAGIC);
    put_u64(&mut out, PTC_VERSION);
    out.push(mem_mode);
    put_u32(&mut out, infos.len() as u32);
    for i in infos {
        put_u64(&mut out, i.guest_address);
        put_u64(&mut out, i.guest_size);
        put_u32(&mut out, i.code.len() as u32);
        out.extend_from_slice(&i.code);
        put_u32(&mut out, i.relocs.len() as u32);
        for r in &i.relocs {
            put_i32(&mut out, r.position);
            out.push(sym_type_to_u8(r.symbol.typ));
            put_u64(&mut out, r.symbol.value);
        }
    }
    out
}

/// Parse a cache. Returns None on magic/version/mem-mode mismatch or truncation
/// (caller then just recompiles  never trusts a stale/garbage cache).
pub fn deserialize(bytes: &[u8], expected_mem_mode: u8) -> Option<Vec<PtcInfo>> {
    let mut r = Reader::new(bytes);
    if r.take(8)? != PTC_MAGIC {
        return None;
    }
    if r.u64()? != PTC_VERSION {
        return None;
    }
    if r.u8()? != expected_mem_mode {
        return None;
    }
    let count = r.u32()? as usize;
    let mut infos = Vec::with_capacity(count);
    for _ in 0..count {
        let guest_address = r.u64()?;
        let guest_size = r.u64()?;
        let code_len = r.u32()? as usize;
        let code = r.take(code_len)?.to_vec();
        let reloc_count = r.u32()? as usize;
        let mut relocs = Vec::with_capacity(reloc_count);
        for _ in 0..reloc_count {
            let position = r.i32()?;
            let typ = u8_to_sym_type(r.u8()?)?;
            let value = r.u64()?;
            relocs.push(RelocationEntry {
                position,
                symbol: Symbol { typ, value },
            });
        }
        infos.push(PtcInfo {
            guest_address,
            guest_size,
            code,
            relocs,
        });
    }
    Some(infos)
}
