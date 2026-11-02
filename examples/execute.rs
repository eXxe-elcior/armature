// Compile MOVZ X0, #42; RET and execute the generated x86-64 code.
//
// Usage: cargo run --example execute

use armature::memory::{MemoryManager, MemoryManagerType};
use armature::registers::{CpuState, V128};
use armature::translation::translator::Translator;

struct FlatMemory {
    base: u64,
    code: Vec<u32>,
}

impl MemoryManager for FlatMemory {
    fn address_space_bits(&self) -> i32 {
        39
    }

    fn page_table_pointer(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    fn manager_type(&self) -> MemoryManagerType {
        // Decode-time reads only for this smoke  no guest loads/stores.
        MemoryManagerType::SoftwarePageTable
    }

    fn read_u8(&self, _va: u64) -> u8 {
        unimplemented!()
    }
    fn read_u16(&self, _va: u64) -> u16 {
        unimplemented!()
    }
    fn read_u32(&self, va: u64) -> u32 {
        let idx = ((va - self.base) / 4) as usize;
        self.code[idx]
    }
    fn read_u64(&self, _va: u64) -> u64 {
        unimplemented!()
    }
    fn read_v128(&self, _va: u64) -> V128 {
        unimplemented!()
    }

    fn read_tracked_u8(&self, _va: u64) -> u8 {
        unimplemented!()
    }
    fn read_tracked_u16(&self, _va: u64) -> u16 {
        unimplemented!()
    }
    fn read_tracked_u32(&self, _va: u64) -> u32 {
        unimplemented!()
    }
    fn read_tracked_u64(&self, _va: u64) -> u64 {
        unimplemented!()
    }
    fn read_tracked_v128(&self, _va: u64) -> V128 {
        unimplemented!()
    }

    fn write_u8(&self, _va: u64, _v: u8) {
        unimplemented!()
    }
    fn write_u16(&self, _va: u64, _v: u16) {
        unimplemented!()
    }
    fn write_u32(&self, _va: u64, _v: u32) {
        unimplemented!()
    }
    fn write_u64(&self, _va: u64, _v: u64) {
        unimplemented!()
    }
    fn write_v128(&self, _va: u64, _v: V128) {
        unimplemented!()
    }

    fn is_mapped(&self, va: u64) -> bool {
        va >= self.base && va < self.base + (self.code.len() as u64) * 4
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

fn main() {
    const ENTRY: u64 = 0x1000;

    let memory = FlatMemory {
        base: ENTRY,
        code: vec![
            0xd2800540, // movz x0, #42
            0xd65f03c0, // ret
        ],
    };

    let translator = Translator::new(Box::new(memory));

    let func = translator.translate(ENTRY, false);

    println!("== compiled {} bytes ==", func.code.len());
    for chunk in func.code.chunks(16) {
        for b in chunk {
            print!("{b:02x} ");
        }
        println!();
    }

    let mut state: CpuState = unsafe { std::mem::zeroed() };
    state.counter = std::sync::atomic::AtomicI32::new(100); // Avoid synchronization callbacks in this example.
    state.x[30] = 0; // LR = 0 → RET returns next-PC 0 → dispatch loop exits

    let state_ptr = &mut state as *mut CpuState as *mut u8;
    translator.execute(state_ptr, ENTRY);

    println!("X0 = {}", state.x[0]);
    assert_eq!(state.x[0], 42, "MOVZ X0,#42 did not land in CpuState");
    println!("execute: passed");
}
