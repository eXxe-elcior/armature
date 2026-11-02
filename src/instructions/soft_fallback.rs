// Port of Instructions/SoftFallback.cs.
// Private CRC helpers use a _poly suffix to distinguish them from public entry points.

const CRC32_REV_POLY: u32 = 0xedb88320;
const CRC32C_REV_POLY: u32 = 0x82f63b78;

pub extern "C" fn crc32b(crc: u32, value: u8) -> u32 {
    crc32(crc, CRC32_REV_POLY, value)
}
pub extern "C" fn crc32h(crc: u32, value: u16) -> u32 {
    crc32h_poly(crc, CRC32_REV_POLY, value)
}
pub extern "C" fn crc32w(crc: u32, value: u32) -> u32 {
    crc32w_poly(crc, CRC32_REV_POLY, value)
}
pub extern "C" fn crc32x(crc: u32, value: u64) -> u32 {
    crc32x_poly(crc, CRC32_REV_POLY, value)
}

pub extern "C" fn crc32cb(crc: u32, value: u8) -> u32 {
    crc32(crc, CRC32C_REV_POLY, value)
}
pub extern "C" fn crc32ch(crc: u32, value: u16) -> u32 {
    crc32h_poly(crc, CRC32C_REV_POLY, value)
}
pub extern "C" fn crc32cw(crc: u32, value: u32) -> u32 {
    crc32w_poly(crc, CRC32C_REV_POLY, value)
}
pub extern "C" fn crc32cx(crc: u32, value: u64) -> u32 {
    crc32x_poly(crc, CRC32C_REV_POLY, value)
}

fn crc32h_poly(mut crc: u32, poly: u32, val: u16) -> u32 {
    crc = crc32(crc, poly, (val >> 0) as u8);
    crc = crc32(crc, poly, (val >> 8) as u8);

    crc
}

fn crc32w_poly(mut crc: u32, poly: u32, val: u32) -> u32 {
    crc = crc32(crc, poly, (val >> 0) as u8);
    crc = crc32(crc, poly, (val >> 8) as u8);
    crc = crc32(crc, poly, (val >> 16) as u8);
    crc = crc32(crc, poly, (val >> 24) as u8);

    crc
}

fn crc32x_poly(mut crc: u32, poly: u32, val: u64) -> u32 {
    crc = crc32(crc, poly, (val >> 0) as u8);
    crc = crc32(crc, poly, (val >> 8) as u8);
    crc = crc32(crc, poly, (val >> 16) as u8);
    crc = crc32(crc, poly, (val >> 24) as u8);
    crc = crc32(crc, poly, (val >> 32) as u8);
    crc = crc32(crc, poly, (val >> 40) as u8);
    crc = crc32(crc, poly, (val >> 48) as u8);
    crc = crc32(crc, poly, (val >> 56) as u8);

    crc
}

// "ShrImm64" region (InstEmitSimdShift EmitShrImm64 callees)
pub extern "C" fn signed_shr_imm64(value: i64, round_const: i64, shift: i32) -> i64 {
    if round_const == 0 {
        if shift <= 63 {
            value >> shift
        } else {
            // shift == 64
            if value < 0 { -1 } else { 0 }
        }
    } else {
        // round_const == 1 << (shift - 1)
        if shift <= 63 {
            let add = value.wrapping_add(round_const);

            if (!value & (value ^ add)) < 0 {
                ((add as u64) >> shift) as i64
            } else {
                add >> shift
            }
        } else {
            // shift == 64
            0
        }
    }
}

pub extern "C" fn unsigned_shr_imm64(value: u64, round_const: i64, shift: i32) -> u64 {
    if round_const == 0 {
        if shift <= 63 {
            value >> shift
        } else {
            // shift == 64
            0
        }
    } else {
        // round_const == 1 << (shift - 1)
        let add = value.wrapping_add(round_const as u64);

        if (add < value) && (add < round_const as u64) {
            if shift <= 63 {
                (add >> shift) | (0x8000000000000000u64 >> (shift - 1))
            } else {
                // shift == 64
                1
            }
        } else if shift <= 63 {
            add >> shift
        } else {
            // shift == 64
            0
        }
    }
}

// "Count" region
pub extern "C" fn count_leading_signs(mut value: u64, size: i32) -> u64 {
    // size is 8, 16, 32 or 64 (SIMD&FP or Base Inst.).
    value ^= value >> 1;

    let high_bit = size - 2;

    let mut bit = high_bit;
    while bit >= 0 {
        if ((value >> bit) & 0b1) != 0 {
            return (high_bit - bit) as u64;
        }
        bit -= 1;
    }

    (size - 1) as u64
}

const CLZ_NIBBLE_TBL: [u8; 16] = [4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

pub extern "C" fn count_leading_zeros(value: u64, size: i32) -> u64 {
    // size is 8, 16, 32 or 64 (SIMD&FP or Base Inst.).
    if value == 0 {
        return size as u64;
    }

    let mut nibble_idx = size;
    let mut count = 0u64;

    loop {
        nibble_idx -= 4;
        let pre_count = CLZ_NIBBLE_TBL[((value >> nibble_idx) & 0b1111) as usize];
        count += pre_count as u64;

        if pre_count != 4 {
            break;
        }
    }

    count
}

// C# loops `for (int bit = 7; bit >= 0; bit--)`  8 iterations, index unused.
fn crc32(mut crc: u32, poly: u32, val: u8) -> u32 {
    crc ^= val as u32;

    for _ in 0..8 {
        let mask = (-((crc & 1) as i32)) as u32;

        crc = (crc >> 1) ^ (poly & mask);
    }

    crc
}

// SHA1 (C# SoftFallback "Sha1" region)
use crate::registers::V128;

fn v128_extract_u32(v: V128, i: u32) -> u32 {
    (v >> (32 * i)) as u32
}

fn v128_insert_u32(v: V128, i: u32, value: u32) -> V128 {
    (v & !(0xffff_ffffu128 << (32 * i))) | ((value as V128) << (32 * i))
}

fn v128_from_u64s(lo: u64, hi: u64) -> V128 {
    (hi as V128) << 64 | lo as V128
}

fn v128_from_u32s(e0: u32, e1: u32, e2: u32, e3: u32) -> V128 {
    (e3 as V128) << 96 | (e2 as V128) << 64 | (e1 as V128) << 32 | e0 as V128
}

// C# private Rol32_160(ref uint y, ref V128 x).
fn rol32_160(y: &mut u32, x: &mut V128) {
    let x_e3 = v128_extract_u32(*x, 3);

    *x <<= 32;
    *x = v128_insert_u32(*x, 0, *y);

    *y = x_e3;
}

fn sha_choose(x: u32, y: u32, z: u32) -> u32 {
    ((y ^ z) & x) ^ z
}

fn sha_majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}

fn sha_parity(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

pub extern "C" fn hash_choose(hash_abcd: V128, hash_e: u32, wk: V128) -> V128 {
    let mut hash_abcd = hash_abcd;
    let mut hash_e = hash_e;

    for e in 0..=3 {
        let t = sha_choose(
            v128_extract_u32(hash_abcd, 1),
            v128_extract_u32(hash_abcd, 2),
            v128_extract_u32(hash_abcd, 3),
        );

        hash_e = hash_e
            .wrapping_add(v128_extract_u32(hash_abcd, 0).rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(v128_extract_u32(wk, e));

        let t = v128_extract_u32(hash_abcd, 1).rotate_left(30);

        hash_abcd = v128_insert_u32(hash_abcd, 1, t);

        rol32_160(&mut hash_e, &mut hash_abcd);
    }

    hash_abcd
}

pub extern "C" fn fixed_rotate(hash_e: u32) -> u32 {
    hash_e.rotate_left(30)
}

pub extern "C" fn hash_majority(hash_abcd: V128, hash_e: u32, wk: V128) -> V128 {
    let mut hash_abcd = hash_abcd;
    let mut hash_e = hash_e;

    for e in 0..=3 {
        let t = sha_majority(
            v128_extract_u32(hash_abcd, 1),
            v128_extract_u32(hash_abcd, 2),
            v128_extract_u32(hash_abcd, 3),
        );

        hash_e = hash_e
            .wrapping_add(v128_extract_u32(hash_abcd, 0).rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(v128_extract_u32(wk, e));

        let t = v128_extract_u32(hash_abcd, 1).rotate_left(30);

        hash_abcd = v128_insert_u32(hash_abcd, 1, t);

        rol32_160(&mut hash_e, &mut hash_abcd);
    }

    hash_abcd
}

pub extern "C" fn hash_parity(hash_abcd: V128, hash_e: u32, wk: V128) -> V128 {
    let mut hash_abcd = hash_abcd;
    let mut hash_e = hash_e;

    for e in 0..=3 {
        let t = sha_parity(
            v128_extract_u32(hash_abcd, 1),
            v128_extract_u32(hash_abcd, 2),
            v128_extract_u32(hash_abcd, 3),
        );

        hash_e = hash_e
            .wrapping_add(v128_extract_u32(hash_abcd, 0).rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(v128_extract_u32(wk, e));

        let t = v128_extract_u32(hash_abcd, 1).rotate_left(30);

        hash_abcd = v128_insert_u32(hash_abcd, 1, t);

        rol32_160(&mut hash_e, &mut hash_abcd);
    }

    hash_abcd
}

pub extern "C" fn sha1_schedule_part1(w0_3: V128, w4_7: V128, w8_11: V128) -> V128 {
    let t2 = w4_7 as u64; // Extract<ulong>(0)
    let t1 = (w0_3 >> 64) as u64; // Extract<ulong>(1)

    let result = v128_from_u64s(t1, t2);

    result ^ (w0_3 ^ w8_11)
}

pub extern "C" fn sha1_schedule_part2(tw0_3: V128, w12_15: V128) -> V128 {
    let t = tw0_3 ^ (w12_15 >> 32);

    let t_e0 = t as u32;
    let t_e1 = (t >> 32) as u32;
    let t_e2 = (t >> 64) as u32;
    let t_e3 = (t >> 96) as u32;

    v128_from_u32s(
        t_e0.rotate_left(1),
        t_e1.rotate_left(1),
        t_e2.rotate_left(1),
        t_e3.rotate_left(1) ^ t_e0.rotate_left(2),
    )
}
