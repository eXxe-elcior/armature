// Port of CodeGen/X86/Mxcsr.cs ([Flags] enum).

pub const FTZ: i32 = 1 << 15; // Flush To Zero.
pub const RHI: i32 = 1 << 14; // Round Mode high bit.
pub const RLO: i32 = 1 << 13; // Round Mode low bit.
pub const UM: i32 = 1 << 11; // Underflow Mask.
pub const DM: i32 = 1 << 8; // Denormal Mask.
pub const DAZ: i32 = 1 << 6; // Denormals Are Zero.
