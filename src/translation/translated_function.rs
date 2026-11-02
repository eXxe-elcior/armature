// Port of Translation/TranslatedFunction.cs.
// Owns compiled code, executable memory, and per-function metadata.

use std::ptr::NonNull;

/// Guest function signature: takes native context pointer, returns next PC.
pub type GuestFunc = unsafe extern "C" fn(*mut u8) -> u64;

/// Port of C# TranslatedFunction  the output of translation + JIT mapping.
pub struct TranslatedFunction {
    pub code: Vec<u8>,                 // compiled machine code bytes
    pub func_ptr: Option<NonNull<u8>>, // pointer to executable mapping
    pub guest_size: u64,               // size of guest code this covers
    /// C# Counter<uint> in CountTable  rejit call counter, address baked
    /// into LCQ code by EmitRejitCheck. Owned here so it lives exactly as
    /// long as the code that increments it. None for HighCq functions.
    pub rejit_counter: Option<Box<u32>>,
}

impl TranslatedFunction {
    pub fn new(code: Vec<u8>, guest_size: u64) -> Self {
        let func_ptr = if !code.is_empty() {
            let ptr = unsafe { jit_map(&code) };
            Some(NonNull::new(ptr).expect("JitCache returned null"))
        } else {
            None
        };

        Self {
            code,
            func_ptr,
            guest_size,
            rejit_counter: None,
        }
    }

    /// Call the compiled function with a native context pointer.
    /// Returns the next guest PC.
    ///
    /// SAFETY: `native_context` must point to a valid, properly aligned
    /// CpuState whose layout matches what the compiled code expects.
    pub unsafe fn execute(&self, native_context: *mut u8) -> u64 {
        match self.func_ptr {
            Some(ptr) => {
                let func: GuestFunc = unsafe { std::mem::transmute(ptr.as_ptr()) };
                unsafe { func(native_context) }
            }
            None => 0,
        }
    }
}

impl Drop for TranslatedFunction {
    fn drop(&mut self) {
        if let Some(ptr) = self.func_ptr {
            let len = self.code.len().next_multiple_of(4096);
            unsafe { jit_unmap(ptr.as_ptr(), len) };
        }
    }
}

/// Map a slice of machine code into executable memory.
///
/// Allocate with mmap and copy the code, then remove write access with mprotect.
#[cfg(target_os = "linux")]
unsafe fn jit_map(code: &[u8]) -> *mut u8 {
    let page_size = 4096usize;
    let len = code.len().next_multiple_of(page_size);

    let ptr = unsafe { linux_mmap(len) };
    // mmap returns -errno in [-4095, -1] on failure, not NULL.
    if ptr.is_null() || (ptr as isize) < 0 {
        panic!("JitCache: mmap failed (errno {})", -(ptr as isize));
    }

    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
        // mprotect to RX (remove W). On failure, unmap and panic.
        linux_mprotect(ptr, len, true);
    }

    ptr
}

/// Unmap previously JIT-mapped memory.
#[cfg(target_os = "linux")]
unsafe fn jit_unmap(ptr: *mut u8, len: usize) {
    unsafe { linux_munmap(ptr, len) };
}

// Linux mmap/mprotect wrappers (raw syscalls, no libc dependency)
#[cfg(target_os = "linux")]
unsafe fn linux_mmap(size: usize) -> *mut u8 {
    use std::arch::asm;

    let result: *mut u8;
    unsafe {
        asm!(
            "syscall",
            in("rax") 9u64,       // SYS_mmap
            in("rdi") 0u64,       // addr = NULL
            in("rsi") size,
            in("rdx") 7u64,       // PROT_READ | PROT_WRITE | PROT_EXEC
            in("r10") 0x22u64,    // MAP_PRIVATE | MAP_ANONYMOUS
            in("r8") 0xffffffffffffffffu64, // fd = -1
            in("r9") 0u64,        // offset = 0
            lateout("rax") result,
            // syscall clobbers rcx (return rip) and r11 (rflags).
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    result
}

#[cfg(target_os = "linux")]
unsafe fn linux_mprotect(ptr: *mut u8, len: usize, exec: bool) {
    use std::arch::asm;

    let prot = if exec { 5u64 } else { 0u64 }; // PROT_READ | PROT_EXEC = 5

    let result: i64;
    unsafe {
        asm!(
            "syscall",
            in("rax") 10u64,       // SYS_mprotect
            in("rdi") ptr,
            in("rsi") len,
            in("rdx") prot,
            lateout("rax") result,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    assert_eq!(result, 0, "mprotect failed");
}

#[cfg(target_os = "linux")]
unsafe fn linux_munmap(ptr: *mut u8, len: usize) {
    use std::arch::asm;

    unsafe {
        asm!(
            "syscall",
            in("rax") 11u64,       // SYS_munmap
            in("rdi") ptr,
            in("rsi") len,
            lateout("rax") _,      // discard result
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
}

// Non-Linux stub.
#[cfg(not(target_os = "linux"))]
fn jit_map(code: &[u8]) -> *mut u8 {
    // Allocate from heap (not executable)  will fault on execution.
    let layout =
        std::alloc::Layout::from_size_align(code.len(), 16).expect("invalid layout for JIT code");
    let ptr = unsafe { std::alloc::alloc(layout) };
    if !ptr.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()) };
    }
    ptr
}

#[cfg(not(target_os = "linux"))]
unsafe fn jit_unmap(ptr: *mut u8, len: usize) {
    let layout =
        std::alloc::Layout::from_size_align(len, 16).expect("invalid layout for JIT unmap");
    unsafe { std::alloc::dealloc(ptr, layout) };
}
