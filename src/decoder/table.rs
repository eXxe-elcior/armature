use crate::decoder::instruction_name::InstructionName;
use crate::decoder::opcode_kind::OpCodeKind;
use crate::decoder::table_entries::A64_RAW;
use crate::decoder::utilities::*;
use std::sync::LazyLock;

// TableEntry
// Carries the instruction name, emitter, and decoder.
#[derive(Clone, Copy)]
pub struct TableEntry {
    pub mask: u32,
    pub value: u32,
    pub info: InstructionInfo,
    pub decode: fn(u32, u64) -> Option<(OpCodeKind, RegisterSize)>,
}

// Raw entry (source of truth, one per SetA64 call)
// Mirrors C# SetA64(encoding, InstName, InstEmit delegate, OpCode ctor).
pub struct RawEntry {
    pub pattern: &'static str,
    pub name: InstructionName,
    pub emit: EmitFn,
    pub decode: fn(u32, u64) -> Option<(OpCodeKind, RegisterSize)>,
}

// Fast-lookup table (built once, immutable)
const FAST_LOOKUP_SIZE: usize = 0x1000;

struct A64FastTable {
    entries: Vec<TableEntry>,
    buckets: Vec<(u32, u32)>, // length = FAST_LOOKUP_SIZE, (start, end) into entries
}

/// ToFastLookupIndexA  exact port from C#.
#[inline]
fn fast_lookup_index(value: u32) -> u32 {
    ((value >> 10) & 0x00F) | ((value >> 18) & 0xFF0)
}

/// Port of C# Set()  parses a 32-char pattern string, handles << / >> expansion,
/// returns all resulting (mask, value) pairs.
fn expand_pattern(
    pattern: &str,
    name: InstructionName,
    emit: EmitFn,
    decode: fn(u32, u64) -> Option<(OpCodeKind, RegisterSize)>,
) -> Vec<TableEntry> {
    let mut bit: i32 = 31;
    let mut value: u32 = 0;
    let mut x_mask: u32 = 0;
    let mut x_bits: usize = 0;
    let mut x_pos = [0i32; 32];
    let mut blacklisted: u32 = 0;

    for chr in pattern.chars() {
        match chr {
            '1' => value |= 1 << bit,
            '0' => {}
            'x' => x_mask |= 1 << bit,
            '>' => {
                x_pos[x_bits] = bit;
                x_bits += 1;
            }
            '<' => {
                x_pos[x_bits] = bit;
                x_bits += 1;
                blacklisted |= 1 << bit;
            }
            _ => unreachable!(),
        }
        bit -= 1;
    }

    let mask = !x_mask;

    if x_bits == 0 {
        return vec![TableEntry {
            mask,
            value,
            info: InstructionInfo { name, emit },
            decode,
        }];
    }

    let mut entries = Vec::with_capacity(1 << x_bits);
    for index in 0..(1 << x_bits) {
        let mut mask_val = 0u32;
        for x in 0..x_bits {
            mask_val |= ((index >> x) & 1) << x_pos[x];
        }
        if mask_val != blacklisted {
            entries.push(TableEntry {
                mask,
                value: value | mask_val,
                info: InstructionInfo { name, emit },
                decode,
            });
        }
    }
    entries
}

// Fast table
static A64_FAST_TABLE: LazyLock<A64FastTable> = LazyLock::new(|| {
    // Step 1: expand all raw entries
    let all_entries: Vec<TableEntry> = A64_RAW
        .iter()
        .flat_map(|raw| expand_pattern(raw.pattern, raw.name, raw.emit, raw.decode))
        .collect();

    // Step 2: distribute into fast-lookup buckets (ports FillFastLookupTable)
    let mut temp: Vec<Vec<TableEntry>> = (0..FAST_LOOKUP_SIZE).map(|_| Vec::new()).collect();

    for entry in &all_entries {
        let mask_idx = fast_lookup_index(entry.mask);
        let value_idx = fast_lookup_index(entry.value);

        for idx in 0..FAST_LOOKUP_SIZE {
            if (idx as u32 & mask_idx) == value_idx {
                temp[idx].push(*entry);
            }
        }
    }

    // Step 3: flatten into compact representation
    let mut entries: Vec<TableEntry> = Vec::with_capacity(all_entries.len());
    let mut buckets = vec![(0u32, 0u32); FAST_LOOKUP_SIZE];

    for (idx, bucket) in temp.iter().enumerate() {
        let start = entries.len() as u32;
        entries.extend_from_slice(bucket);
        buckets[idx] = (start, entries.len() as u32);
    }

    A64FastTable { entries, buckets }
});

// Public API
// Called from decode_one.rs in the hot path.
#[inline]
pub fn lookup_entry(raw: u32) -> Option<&'static TableEntry> {
    let table = &*A64_FAST_TABLE;
    let idx = fast_lookup_index(raw) as usize;
    let (start, end) = table.buckets[idx];
    let slice = &table.entries[start as usize..end as usize];
    for entry in slice {
        if (raw & entry.mask) == entry.value {
            return Some(entry);
        }
    }
    None
}
