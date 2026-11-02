// Port of Common/AddressTableLevel.cs + Ryujinx.Cpu/AddressTable.cs.
//
// Multi-level page table mapping guest address → host function pointer.
// Specialized for u64 entries (function addresses).
//
// Divergence: plain Vec<Box<[u64]>> pages instead of NativeAllocator +
// sparse signal-handler lazy mapping. Non-sparse (5-level arm64 preset)
// only  levels are small enough to allocate on demand.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::memory::MemoryManagerType;

/// C# AddressTableLevel struct.
#[derive(Debug, Clone, Copy)]
pub struct AddressTableLevel {
    /// Bit index of this level in the guest address.
    pub index: i32,
    /// Number of bits used by this level.
    pub length: i32,
}

impl AddressTableLevel {
    /// Mask that extracts this level's bits from a guest address.
    pub fn mask(self) -> u64 {
        ((1u64 << self.length) - 1) << self.index
    }

    /// Extract this level's index value from a guest address.
    pub fn get_value(self, address: u64) -> u64 {
        (address >> self.index) & ((1u64 << self.length) - 1)
    }
}

/// C# AddressTablePresets.GetArmPreset.
pub fn get_arm_preset(
    for_64_bits: bool,
    _memory_type: MemoryManagerType,
) -> Vec<AddressTableLevel> {
    // For SoftwareMmu/SoftwarePageTable, C# uses non-sparse (5-level).
    // For HostMapped/HostTracked, C# uses sparse (2-level giant).
    //
    // Divergence: we always use the 5-level non-sparse preset, regardless
    // of memory manager type. This avoids the sparse signal-handler
    // dependency (Linux userfaultfd / Windows MM) while preserving the
    // same observable behavior. All levels are small enough to allocate
    // eagerly on demand.
    //
    // 3-level 64-bit: bits covered = 16+15+15 = 46, tiling bits 2..47 exactly
    // (same coverage/mask as the old 5-level, so IsValid is unchanged). Fewer
    // levels = fewer dependent loads in the inline dispatch walk emitted at
    // every guest branch/call  the dispatch hot path. Trades a 512 KB root +
    // 256 KB on-demand pages (vs the old tiny pages) for 2 fewer loads/dispatch.
    let _levels_64_bit: &[AddressTableLevel] = &[
        AddressTableLevel {
            index: 32,
            length: 16,
        },
        AddressTableLevel {
            index: 17,
            length: 15,
        },
        AddressTableLevel {
            index: 2,
            length: 15,
        },
    ];

    let _levels_32_bit: &[AddressTableLevel] = &[
        AddressTableLevel {
            index: 31,
            length: 17,
        },
        AddressTableLevel {
            index: 23,
            length: 8,
        },
        AddressTableLevel {
            index: 15,
            length: 8,
        },
        AddressTableLevel {
            index: 7,
            length: 8,
        },
        AddressTableLevel {
            index: 1,
            length: 6,
        },
    ];

    if for_64_bits {
        _levels_64_bit.to_vec()
    } else {
        _levels_32_bit.to_vec()
    }
}

/// C# AddressTable<ulong> specialized for u64 entries.
///
/// Owns a multi-level page table where:
/// - Root + non-leaf pages hold pointers to child pages (as raw u64 addresses)
/// - Leaf pages hold function pointers (u64 host addresses)
/// - Unallocated non-leaf entries = 0 (null → dispatch stub fallback)
/// - Unallocated leaf entries = fill value (slow dispatch stub address)
pub struct AddressTable {
    levels: Vec<AddressTableLevel>,
    mask: u64,
    fill: u64,
    /// Root page (level 0): pre-allocated on construction.
    /// All entries are 0 (no child pages allocated yet).
    root: Box<[u64]>,
    /// All non-root pages. The pages vector owns them; raw pointers into
    /// these boxes are stored in parent levels. Pages are never removed
    /// or reallocated.
    pages: Vec<Box<[u64]>>,
}

impl AddressTable {
    /// Create a new AddressTable with the given levels and fill value.
    /// The root page is pre-allocated; child pages are allocated on demand
    /// by `register` or `get`.
    ///
    /// C# AddressTable(levels, sparse) constructor (non-sparse path only).
    pub fn new(levels: Vec<AddressTableLevel>, fill: u64) -> Self {
        let mut mask = 0u64;
        for &level in &levels {
            mask |= level.mask();
        }

        let root_len = 1usize << levels[0].length;

        Self {
            root: vec![0u64; root_len].into_boxed_slice(),
            levels,
            mask,
            fill,
            pages: Vec::new(),
        }
    }

    /// C# AddressTable.CreateForArm: create with the arm preset levels.
    pub fn create_for_arm(for_64_bits: bool, memory_type: MemoryManagerType, fill: u64) -> Self {
        Self::new(get_arm_preset(for_64_bits, memory_type), fill)
    }

    /// C# AddressTable.IsValid: true if the address fits within the table's mask.
    pub fn is_valid(&self, address: u64) -> bool {
        (address & !self.mask) == 0
    }

    /// C# AddressTable.Base  the base address of the root page.
    /// Used by the generated dispatch stub as the starting point for
    /// the inline table walk.
    pub fn base(&self) -> u64 {
        self.root.as_ptr() as u64
    }

    /// C# AddressTable.Fill (getter).
    pub fn fill(&self) -> u64 {
        self.fill
    }

    /// Set the fill value for newly allocated leaf pages.
    /// C# Fill setter → UpdateFill(value).
    pub fn set_fill(&mut self, fill: u64) {
        self.fill = fill;
    }

    /// C# AddressTable.Mask.
    pub fn mask(&self) -> u64 {
        self.mask
    }

    /// C# AddressTable.Levels.
    pub fn levels(&self) -> &[AddressTableLevel] {
        &self.levels
    }

    /// Get a reference to the value at the given guest address.
    /// Allocates child pages on demand. Returns None if the address is
    /// out of range (not IsValid).
    ///
    /// C# AddressTable.GetValue(address).
    pub fn get(&self, address: u64) -> Option<&u64> {
        if !self.is_valid(address) {
            return None;
        }
        // Safety: we only hand out references into pages that are owned
        // by self.pages or self.root. The borrow checker can't see through
        // the pointer arithmetic in the walk, so we use unsafe internally
        // but expose a safe reference.
        unsafe { Some(self.get_unchecked(address)) }
    }

    /// Get a mutable reference to the value at the given guest address.
    /// Allocates child pages on demand. Returns None if the address is
    /// out of range.
    pub fn get_mut(&mut self, address: u64) -> Option<&mut u64> {
        if !self.is_valid(address) {
            return None;
        }
        let ptr = self.resolve_mut(address);
        unsafe { Some(&mut *ptr) }
    }

    /// Compute the leaf entry address for a guest address, ensuring all
    /// intermediate pages are allocated. Returns the address of the leaf
    /// entry in the table. Reading from this address at runtime gives the
    /// current fill or registered function pointer.
    ///
    /// C#: Const(ref FunctionTable.GetValue(guestAddress.Value))  used by
    /// InstEmitFlowHelper's EmitTableBranch constant-address fast path.
    pub fn ensure_entry(&mut self, address: u64) -> u64 {
        self.resolve_mut(address) as u64
    }

    /// Set the entry for a guest address. Allocates child pages as needed.
    /// C# Volatile.Write(ref FunctionTable.GetValue(guestAddress), value).
    pub fn set(&mut self, address: u64, value: u64) {
        let ptr = self.resolve_mut(address);
        // Release store: the guest dispatch walk reads this slot locklessly.
        // A release store publishes any prior writes (e.g. a freshly allocated
        // child page) before this value becomes observable, and stops the
        // compiler from hoisting the publish ahead of that initialization.
        unsafe { (*(ptr as *const AtomicU64)).store(value, Ordering::Release) };
    }

    /// Walk the levels and return a pointer to the leaf entry.
    /// Allocates intermediate/leaf pages on demand.
    fn resolve_mut(&mut self, address: u64) -> *mut u64 {
        debug_assert!(self.is_valid(address));

        let mut page: *mut u64 = self.root.as_mut_ptr();

        // Descend through all non-leaf levels. The child allocated at the
        // last iteration (i+1 == leaf level) IS the leaf page and must be
        // initialized with `fill`, not zero  after the loop `page` points
        // at the leaf page and the final index selects a VALUE slot.
        for i in 0..self.levels.len() - 1 {
            let level = self.levels[i];
            let idx = level.get_value(address) as usize;

            unsafe {
                let entry = page.add(idx);
                let entry_atomic = &*(entry as *const AtomicU64);
                if entry_atomic.load(Ordering::Acquire) == 0 {
                    let child_len = 1usize << self.levels[i + 1].length;
                    let is_leaf = i + 1 == self.levels.len() - 1;
                    let init = if is_leaf { self.fill } else { 0u64 };
                    let child = vec![init; child_len].into_boxed_slice();
                    let child_ptr = child.as_ptr() as u64;
                    self.pages.push(child);
                    // Release: the child page is fully initialized above; this
                    // store publishes it to lockless guest readers without the
                    // compiler reordering the publish before the fill.
                    entry_atomic.store(child_ptr, Ordering::Release);
                }
                page = entry_atomic.load(Ordering::Acquire) as *mut u64;
            }
        }

        let leaf_level = self.levels[self.levels.len() - 1];
        let idx = leaf_level.get_value(address) as usize;
        unsafe { &mut *page.add(idx) }
    }

    /// Get a pointer to the entry without allocation (for the dispatch
    /// stub's inline walk). Returns null if the entry doesn't exist yet
    /// (no child page allocated) or address is out of range.
    ///
    /// SAFETY: the returned pointer is valid for reading if non-null.
    /// It remains valid for the table's lifetime.
    pub unsafe fn get_unchecked(&self, address: u64) -> &u64 {
        debug_assert!(self.is_valid(address));
        let mut page: *const u64 = self.root.as_ptr();
        let levels = &self.levels;

        for i in 0..levels.len() {
            let level = levels[i];
            let idx = level.get_value(address) as usize;
            unsafe {
                let entry = page.add(idx);
                if i < levels.len() - 1 {
                    let child_ptr = *entry as *const u64;
                    page = child_ptr;
                } else {
                    return &*entry;
                }
            }
        }

        // Unreachable: the loop always returns for the last level.
        unreachable!()
    }
}
