// Hashers for JIT-internal keys.
// FxHasher handles small keys; AddrHasher hashes aligned guest addresses as addr >> 2.

use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(SEED);
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.0 = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(SEED);
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(SEED);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

#[derive(Default)]
pub struct AddrHasher(u64);

impl Hasher for AddrHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("AddrHasher is for u64 keys only");
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n >> 2;
    }
}

pub type BuildFxHasher = BuildHasherDefault<FxHasher>;
pub type BuildAddrHasher = BuildHasherDefault<AddrHasher>;

/// Drop-in faster replacements for std HashMap/HashSet for the JIT's internal,
/// trusted, integer/small-key maps. Construct with `::default()`.
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildFxHasher>;
pub type FxHashSet<K> = std::collections::HashSet<K, BuildFxHasher>;
