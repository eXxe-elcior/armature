// Port of Memory/MemoryManagerType.cs + Memory/IMemoryManager.cs.
//
// Divergences:
//   - C# generic Read<T>/ReadTracked<T>/Write<T> over unmanaged T are
//     monomorphized to the widths the JIT actually calls through
//     NativeInterface (u8/u16/u32/u64/V128)  Rust trait objects can't
//     carry generic methods.
//   - ReadGuest<T>/WriteGuest<T> C# default implementations become Rust
//     default methods, same bodies.
//
// The trait is implemented by the embedder (Ryujinx side); the JIT emits
// against `manager_type()` + `address_space_bits()` + `page_table_pointer()`
// at translation time and calls the accessors at runtime via NativeInterface.

use crate::registers::V128;

/// C# Memory/MemoryManagerType.cs  controls the code generated for
/// memory accesses on the JIT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryManagerType {
    /// Complete software MMU, read/write methods always called.
    SoftwareMmu,
    /// Software flat page table for address translation.
    SoftwarePageTable,
    /// Mappings managed by the host OS; memory accessed directly.
    HostMapped,
    /// HostMapped without masking the address within the address space.
    HostMappedUnsafe,
    /// Software flat page table, no support for invalid/non-contiguous access.
    HostTracked,
    /// HostTracked without address masking.
    HostTrackedUnsafe,
}

impl MemoryManagerType {
    pub fn is_host_mapped(self) -> bool {
        matches!(self, Self::HostMapped | Self::HostMappedUnsafe)
    }

    pub fn is_host_tracked(self) -> bool {
        matches!(self, Self::HostTracked | Self::HostTrackedUnsafe)
    }

    pub fn is_host_mapped_or_tracked(self) -> bool {
        self.is_host_mapped() || self.is_host_tracked()
    }
}

/// C# Memory/IMemoryManager.cs.
pub trait MemoryManager {
    fn address_space_bits(&self) -> i32;

    /// C# `nint PageTablePointer`.
    fn page_table_pointer(&self) -> *mut u8;

    /// C# `MemoryManagerType Type` (`type` is a Rust keyword).
    fn manager_type(&self) -> MemoryManagerType;

    // Reads data from CPU mapped memory (C# Read<T>).
    fn read_u8(&self, va: u64) -> u8;
    fn read_u16(&self, va: u64) -> u16;
    fn read_u32(&self, va: u64) -> u32;
    fn read_u64(&self, va: u64) -> u64;
    fn read_v128(&self, va: u64) -> V128;

    // Reads with read tracking (C# ReadTracked<T>).
    fn read_tracked_u8(&self, va: u64) -> u8;
    fn read_tracked_u16(&self, va: u64) -> u16;
    fn read_tracked_u32(&self, va: u64) -> u32;
    fn read_tracked_u64(&self, va: u64) -> u64;
    fn read_tracked_v128(&self, va: u64) -> V128;

    // Reads from guest code (C# ReadGuest<T> default: ReadTracked<T>).
    fn read_guest_u8(&self, va: u64) -> u8 {
        self.read_tracked_u8(va)
    }
    fn read_guest_u16(&self, va: u64) -> u16 {
        self.read_tracked_u16(va)
    }
    fn read_guest_u32(&self, va: u64) -> u32 {
        self.read_tracked_u32(va)
    }
    fn read_guest_u64(&self, va: u64) -> u64 {
        self.read_tracked_u64(va)
    }
    fn read_guest_v128(&self, va: u64) -> V128 {
        self.read_tracked_v128(va)
    }

    // Writes data to CPU mapped memory (C# Write<T>).
    fn write_u8(&self, va: u64, value: u8);
    fn write_u16(&self, va: u64, value: u16);
    fn write_u32(&self, va: u64, value: u32);
    fn write_u64(&self, va: u64, value: u64);
    fn write_v128(&self, va: u64, value: V128);

    // Writes from guest code (C# WriteGuest<T> default: Write<T>).
    fn write_guest_u8(&self, va: u64, value: u8) {
        self.write_u8(va, value)
    }
    fn write_guest_u16(&self, va: u64, value: u16) {
        self.write_u16(va, value)
    }
    fn write_guest_u32(&self, va: u64, value: u32) {
        self.write_u32(va, value)
    }
    fn write_guest_u64(&self, va: u64, value: u64) {
        self.write_u64(va, value)
    }
    fn write_guest_v128(&self, va: u64, value: V128) {
        self.write_v128(va, value)
    }

    /// Checks if the page at a given CPU virtual address is mapped.
    fn is_mapped(&self, va: u64) -> bool;

    /// Alerts the memory tracking that a region has been read/written.
    /// Called before the read/write is performed.
    fn signal_memory_tracking(
        &self,
        va: u64,
        size: u64,
        write: bool,
        precise: bool,
        exempt_id: Option<i32>,
    );
}
