//! AArch64 dynamic recompiler for x86-64 hosts.
//!
//! Build: cargo build --release
//! Test: cargo test --all-targets
//! Example: cargo run --example execute

/// Implement conversions between an arena index and its u32 slot.
#[macro_export]
macro_rules! define_index {
    ($name:ident) => {
        impl From<u32> for $name {
            #[inline]
            fn from(value: u32) -> Self {
                Self(value)
            }
        }
        impl From<$name> for u32 {
            #[inline]
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

/// Define a masked enum lookup. The variant count must equal mask + 1.
#[macro_export]
macro_rules! enum_from_bits {
    ($name:ident, $mask:expr, [$($variant:ident),+ $(,)?]) => {
        impl $name {
            pub fn from_raw(val: u32) -> Self {
                const VARIANTS: &[$name] = &[$($name::$variant),+];
                VARIANTS[(val & $mask) as usize]
            }
        }
    };
}

pub mod codegen;
pub mod decoder;
pub mod ffi;
pub mod instructions;
pub mod ir;
pub mod memory;
pub mod optimizations;
pub mod registers;
pub mod state;
pub mod translation;
pub mod util;

// Host applications supply memory, create a context per guest thread,
// and call Translator::execute with a context pointer and guest entry address.
pub use memory::{MemoryManager, MemoryManagerType};
pub use registers::{CpuState, V128};
pub use state::execution_context::{
    Counter, ExceptionCallback, ExceptionCallbackNoArgs, ExecutionContext,
};
pub use translation::translator::Translator;
