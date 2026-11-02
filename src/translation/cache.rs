// Port of Translation/TranslatorCache.cs  translated-function cache with
// range-overlap queries (C# backs this with an AVL IntervalTree; we use a
// HashMap for the hot exact-address path plus a BTreeMap start-index for
// the cold overlap path. Same observable behavior: insert-if-absent,
// exact lookup, remove, and overlap collection for invalidation).

use std::collections::{BTreeMap, HashMap};

use crate::translation::translated_function::TranslatedFunction;
use crate::util::hash::BuildAddrHasher;

pub struct TranslatorCache {
    map: HashMap<u64, TranslatedFunction, BuildAddrHasher>,
    /// start address → guest size, ordered for range scans.
    starts: BTreeMap<u64, u64>,
    /// Largest guest_size ever inserted  bounds the overlap back-scan.
    max_size: u64,
}

impl TranslatorCache {
    pub fn new() -> Self {
        Self {
            map: HashMap::with_hasher(BuildAddrHasher::default()),
            starts: BTreeMap::new(),
            max_size: 0,
        }
    }

    /// C# GetOrAdd: insert only if absent; always returns the cached entry.
    pub fn get_or_add(&mut self, address: u64, func: TranslatedFunction) -> &TranslatedFunction {
        use std::collections::hash_map::Entry;
        match self.map.entry(address) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let guest_size = func.guest_size;
                self.starts.insert(address, guest_size);
                self.max_size = self.max_size.max(guest_size);
                entry.insert(func)
            }
        }
    }

    /// C# TryGetValue.
    pub fn get(&self, address: u64) -> Option<&TranslatedFunction> {
        self.map.get(&address)
    }

    /// C# Remove.
    pub fn remove(&mut self, address: u64) -> Option<TranslatedFunction> {
        self.starts.remove(&address);
        self.map.remove(&address)
    }

    /// C# GetOverlaps: collect start addresses of cached functions whose
    /// [start, start + guest_size) range intersects [address, address + size).
    pub fn get_overlaps(&self, address: u64, size: u64) -> Vec<u64> {
        let lo = address.saturating_sub(self.max_size);
        let hi = address.saturating_add(size);
        self.starts
            .range(lo..hi)
            .filter(|pair| {
                let (start, guest_size) = pair;
                **guest_size != 0 && (*start).wrapping_add(**guest_size) > address
            })
            .map(|(&start, _)| start)
            .collect()
    }

    /// C# Count.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
