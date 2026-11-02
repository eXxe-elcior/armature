use crate::decoder::utilities::{Condition, DataOperation, RegisterSize};
use crate::registers::{GpRegisterIndex, VecRegisterIndex};

// Port of ARMeilleure.Common.BitUtils

pub fn fill_with_ones(bits: i32) -> i64 {
    if bits >= 64 {
        -1i64
    } else {
        // C# unchecked: bits == 63 → (1L << 63) - 1 wraps MIN-1 → MAX,
        // which is exactly the 63-bit all-ones mask. Rust debug builds
        // panic on the plain subtraction.
        (1i64 << bits).wrapping_sub(1)
    }
}

pub fn highest_bit_set(value: i32) -> i32 {
    31 - (value as u32).leading_zeros() as i32
}

pub const fn highest_bit_set_nibble(value: i32) -> i32 {
    let v = value as u8;
    // NOTE (future me): Need to check if this is correct
    if v == 0 { -1 } else { v.ilog2() as i32 }
}

pub fn replicate(bits: i64, size: i32) -> i64 {
    let mut val = bits as u64;
    let mut i = size;
    while i < 64 {
        val |= val << i;
        i <<= 1;
    }
    val as i64
}

pub fn rotate_within_size(value: i64, r: i32, size: i32) -> i64 {
    let val = value as u64;
    let r = (r as u32) % (size as u32);
    ((val >> r) | (val << (size as u32 - r))) as i64
}

// Size-from-bit helpers
// AArch64 encoding: bit 31 = SF (64-bit if set, else 32-bit for integer ops),
// bit 30 = Q (128-bit if set, else 64-bit for SIMD ops). These appear
// inline at ~16 decode-fn sites; naming the pattern once documents intent.

/// Bit 31: register size (Int64 if set, Int32 otherwise).
#[inline]
pub fn sf_size(raw: u32) -> RegisterSize {
    if (raw >> 31) != 0 {
        RegisterSize::Int64
    } else {
        RegisterSize::Int32
    }
}

/// Bit 30: SIMD size (Simd128 if set, Simd64 otherwise).
#[inline]
pub fn q_size(raw: u32) -> RegisterSize {
    if (raw >> 30) & 1 != 0 {
        RegisterSize::Simd128
    } else {
        RegisterSize::Simd64
    }
}
// Common field extraction (collapses C# base class constructors)
#[inline]
pub fn alu_base(
    raw: u32,
) -> (
    GpRegisterIndex,
    GpRegisterIndex,
    DataOperation,
    RegisterSize,
) {
    let rd = GpRegisterIndex::from((raw & 0x1f) as u8);
    let rn = GpRegisterIndex::from(((raw >> 5) & 0x1f) as u8);
    let data_op = DataOperation::from_raw((raw >> 24) & 0x3);
    let reg_size = sf_size(raw);
    (rd, rn, data_op, reg_size)
}

#[inline]
pub fn mem_base(raw: u32) -> (GpRegisterIndex, GpRegisterIndex, u8) {
    let rt = GpRegisterIndex::from((raw & 0x1f) as u8);
    let rn = GpRegisterIndex::from(((raw >> 5) & 0x1f) as u8);
    let size = ((raw >> 30) & 0x3) as u8;
    (rt, rn, size)
}

#[inline]
pub fn simd_base(raw: u32) -> (VecRegisterIndex, VecRegisterIndex, u8, u8) {
    let rd = VecRegisterIndex::from((raw & 0x1f) as u8);
    let rn = VecRegisterIndex::from(((raw >> 5) & 0x1f) as u8);
    let opc = ((raw >> 15) & 0x3) as u8;
    let size = ((raw >> 22) & 0x3) as u8;
    (rd, rn, opc, size)
}

#[inline]
pub fn simd_reg_base(
    raw: u32,
) -> (
    VecRegisterIndex,
    VecRegisterIndex,
    VecRegisterIndex,
    VecRegisterIndex,
    u8,
    u8,
    bool,
) {
    let (rd, rn, opc, size) = simd_base(raw);
    let bit3 = ((raw >> 3) & 1) != 0;
    let ra = VecRegisterIndex::from(((raw >> 10) & 0x1f) as u8);
    let rm = VecRegisterIndex::from(((raw >> 16) & 0x1f) as u8);
    (rd, rn, rm, ra, opc, size, bit3)
}

// CCMN vs CCMP is carried by InstName (C# dispatches via the Ccmn/Ccmp
// wrapper methods)  OpCodeCcmp has no field for it, so neither do we.
#[inline]
pub fn ccmp_base(raw: u32) -> (GpRegisterIndex, Condition, u8, u8, RegisterSize) {
    let rn = GpRegisterIndex::from(((raw >> 5) & 0x1f) as u8);
    let cond = Condition::from_raw(raw >> 12);
    let nzcv = (raw & 0xf) as u8;
    let rm_imm = ((raw >> 16) & 0x1f) as u8;
    let reg_size = sf_size(raw);
    (rn, cond, nzcv, rm_imm, reg_size)
}

// Immediate decoders (from DecoderHelper.cs)
/// 26-bit signed offset, shifted left 2 (for B/BL)
#[inline]
pub fn decode_imm26_2(raw: u32, addr: u64) -> i64 {
    addr as i64 + (((raw as i64) << 38) >> 36)
}

/// 19-bit signed offset, shifted left 2 (no address add  caller adds if needed)
#[inline]
pub fn decode_imm_s19_2(raw: u32) -> i64 {
    (((raw as i64) << 40) >> 43) & !3
}

/// 14-bit signed offset, shifted left 2 (for TBZ/TBNZ)
#[inline]
pub fn decode_imm_s14_2(raw: u32) -> i64 {
    (((raw as i64) << 45) >> 48) & !3
}

// Bitmask decoder
pub fn decode_bit_mask(raw: u32, immediate: bool) -> Option<(i64, i64, u8, u8)> {
    let imm_s = ((raw >> 10) & 0x3f) as i32;
    let imm_r = ((raw >> 16) & 0x3f) as i32;
    let n = ((raw >> 22) & 1) as i32;
    let sf = ((raw >> 31) & 1) as i32;

    let length = highest_bit_set((((!imm_s) & 0x3f) | (n << 6)) as i32);

    if length < 1 || (sf == 0 && n != 0) {
        return None;
    }

    let size = 1 << length;
    let levels = size - 1;
    let s = imm_s & levels;
    let r = imm_r & levels;

    if immediate && s == levels {
        return None;
    }

    let mut w_mask = fill_with_ones(s + 1);
    let t_mask = fill_with_ones(((s - r) & levels) + 1);

    if r > 0 {
        w_mask = rotate_within_size(w_mask, r, size);
        w_mask &= fill_with_ones(size);
    }

    let w_mask = replicate(w_mask, size);
    let t_mask = replicate(t_mask, size);

    Some((w_mask, t_mask, imm_s as u8, imm_r as u8))
}

// FP immediate expansion (from DecoderHelper.cs)
/// abcdefgh -> aBbbbbbc defgh000 00000000 00000000 (B = ~b)
pub const fn imm8_to_fp32_bits(imm: u8) -> u32 {
    let u = imm as u32;
    let b = (u >> 6) & 1;
    ((u & 0x80) << 24) | ((b ^ 1) << 30) | ((b * 0x1f) << 25) | ((u & 0x3f) << 19)
}

/// abcdefgh -> aBbbbbbb bbcdefgh 00000000 00000000 00000000 00000000 00000000 00000000 (B = ~b)
pub const fn imm8_to_fp64_bits(imm: u8) -> u64 {
    let u = imm as u64;
    let b = (u >> 6) & 1;
    ((u & 0x80) << 56) | ((b ^ 1) << 62) | ((b * 0xff) << 54) | ((u & 0x3f) << 48)
}

pub fn shl_ones(value: i64, shift: i32) -> i64 {
    if shift <= 0 {
        value
    } else if shift >= 64 {
        -1
    } else {
        (value << shift) | (u64::MAX >> (64 - shift)) as i64
    }
}
