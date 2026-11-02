use crate::decoder::helpers::*;
use crate::decoder::opcode_kind::OpCodeKind;
use crate::decoder::utilities::*;
use crate::registers::*;

pub fn decode_alu(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, data_op, reg_size) = alu_base(raw);
    Some((OpCodeKind::Alu { rd, rn, data_op }, reg_size))
}

pub fn decode_alu_imm(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, data_op, reg_size) = alu_base(raw);

    let immediate = match data_op {
        DataOperation::Arithmetic => {
            let imm = (raw >> 10) & 0xfff;
            let shift = (raw >> 22) & 3;
            (imm << (shift * 12)) as i64
        }
        DataOperation::Logical => match decode_bit_mask(raw, true) {
            Some((wmask, _, _, _)) => wmask,
            None => return None,
        },
        _ => return None,
    };

    Some((
        OpCodeKind::AluImm {
            rd,
            rn,
            data_op,
            immediate,
        },
        reg_size,
    ))
}

pub fn decode_alu_rs(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, data_op, reg_size) = alu_base(raw);

    let shift = ((raw >> 10) & 0x3f) as u8;
    if shift >= reg_size.bits() as u8 {
        return None;
    }

    let rm = GpRegisterIndex::from((raw >> 16) as u8);
    let shift_type = ShiftType::from_raw((raw >> 22) & 0x3);

    Some((
        OpCodeKind::AluRs {
            rd,
            rn,
            rm,
            data_op,
            shift,
            shift_type,
        },
        reg_size,
    ))
}

pub fn decode_alu_rx(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, data_op, reg_size) = alu_base(raw);

    let shift = ((raw >> 10) & 0x7) as u8;
    let int_type = IntegerType::from_raw((raw >> 13) & 0x7);
    let rm = GpRegisterIndex::from((raw >> 16) as u8);

    Some((
        OpCodeKind::AluRx {
            rd,
            rn,
            rm,
            data_op,
            shift,
            int_type,
        },
        reg_size,
    ))
}

pub fn decode_bfm(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, _data_op, reg_size) = alu_base(raw);

    match decode_bit_mask(raw, false) {
        Some((wmask, tmask, pos, shift)) => Some((
            OpCodeKind::Bfm {
                rd,
                rn,
                wmask,
                tmask,
                pos,
                shift,
            },
            reg_size,
        )),
        None => None,
    }
}

pub fn decode_mov(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let p1 = (raw >> 22) & 1;
    let sf = (raw >> 31) & 1;
    let reg_size = if sf != 0 {
        RegisterSize::Int64
    } else {
        RegisterSize::Int32
    };

    if sf == 0 && p1 != 0 {
        return None;
    }

    let rd = GpRegisterIndex::from(raw as u8);
    let bit = (((raw >> 21) & 0x3) << 4) as u8;
    let immediate = (((raw >> 5) & 0xffff) as i64) << bit;

    Some((OpCodeKind::Mov { rd, immediate, bit }, reg_size))
}

pub fn decode_mul(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, data_op, reg_size) = alu_base(raw);

    let ra = GpRegisterIndex::from((raw >> 10) as u8);
    let rm = GpRegisterIndex::from((raw >> 16) as u8);

    Some((
        OpCodeKind::Mul {
            rd,
            rn,
            rm,
            ra,
            data_op,
        },
        reg_size,
    ))
}

pub fn decode_csel(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, _data_op, reg_size) = alu_base(raw);

    let rm = GpRegisterIndex::from((raw >> 16) as u8);
    let cond = Condition::from_raw(raw >> 12);

    Some((OpCodeKind::Csel { rd, rn, rm, cond }, reg_size))
}

pub fn decode_ccmp_imm(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let o3 = (raw >> 4) & 1;
    if o3 != 0 {
        return None;
    }
    let (rn, cond, nzcv, rm_imm, reg_size) = ccmp_base(raw);
    Some((
        OpCodeKind::Ccmp {
            rn,
            rm_or_imm: rm_imm,
            nzcv,
            cond,
            is_imm: true,
        },
        reg_size,
    ))
}

pub fn decode_ccmp_reg(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let o3 = (raw >> 4) & 1;
    if o3 != 0 {
        return None;
    }
    let (rn, cond, nzcv, rm_imm, reg_size) = ccmp_base(raw);
    Some((
        OpCodeKind::Ccmp {
            rn,
            rm_or_imm: rm_imm,
            nzcv,
            cond,
            is_imm: false,
        },
        reg_size,
    ))
}

pub fn decode_adr(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rd = GpRegisterIndex::from(raw as u8);
    let immediate = decode_imm_s19_2(raw) | (((raw >> 29) & 3) as i64);
    Some((OpCodeKind::Adr { rd, immediate }, RegisterSize::Int64))
}

pub fn decode_b_imm_al(raw: u32, addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let immediate = decode_imm26_2(raw, addr);
    let link = ((raw >> 31) & 1) != 0;
    Some((OpCodeKind::BImm { immediate, link }, RegisterSize::Int64))
}

pub fn decode_b_imm_cond(raw: u32, addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let o0 = (raw >> 4) & 1;
    if o0 != 0 {
        return None;
    }
    let cond = Condition::from_raw(raw);
    let immediate = addr as i64 + decode_imm_s19_2(raw);
    Some((
        OpCodeKind::BImmCond { cond, immediate },
        RegisterSize::Int64,
    ))
}

pub fn decode_b_imm_cmp(raw: u32, addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let reg_size = sf_size(raw);
    let rt = GpRegisterIndex::from(raw as u8);
    let immediate = addr as i64 + decode_imm_s19_2(raw);
    Some((OpCodeKind::BImmCmp { rt, immediate }, reg_size))
}

pub fn decode_b_imm_test(raw: u32, addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rt = GpRegisterIndex::from(raw as u8);
    let bit = (((raw >> 19) & 0x1f) | ((raw >> 26) & 0x20)) as u8;
    let immediate = addr as i64 + decode_imm_s14_2(raw);
    Some((
        OpCodeKind::BImmTest { rt, bit, immediate },
        RegisterSize::Int64,
    ))
}

pub fn decode_b_reg(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let op2 = (raw >> 16) & 0x1f;
    let op4 = raw & 0x1f;
    if op2 != 0x1f || op4 != 0x00 {
        return None;
    }
    let rn = GpRegisterIndex::from((raw >> 5) as u8);
    Some((OpCodeKind::BReg { rn }, RegisterSize::Int64))
}

pub fn decode_mem_imm(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rt, rn, size) = mem_base(raw);
    let extend64 = ((raw >> 22) & 3) == 2;
    let mut wback = ((raw >> 24) & 1) == 0;
    let mem_op = if wback { ((raw >> 10) & 3) as u8 } else { 4 };
    let post_idx = mem_op == 1;
    let unscaled = mem_op == 0 || mem_op == 2;
    if unscaled {
        wback = false;
    }
    let immediate = if wback || unscaled {
        ((raw << 11) as i32 >> 23) as i64
    } else {
        (((raw >> 10) & 0xfff) << size) as i64
    };
    Some((
        OpCodeKind::MemImm {
            rt,
            rn,
            size,
            extend64,
            immediate,
            wback,
            post_idx,
            unscaled,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_mem_pair(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rt, rn, _) = mem_base(raw);
    let rt2 = GpRegisterIndex::from((raw >> 10) as u8);
    let size = ((raw >> 31) & 1 | 2) as u8;
    let extend64 = ((raw >> 30) & 3) == 1;
    let wback = ((raw >> 23) & 1) != 0;
    let post_idx = ((raw >> 23) & 3) == 1;
    let imm7 = ((raw >> 15) & 0x7f) as i64;
    let immediate = ((imm7 << 57) >> 57) << (size as i64);
    Some((
        OpCodeKind::MemPair {
            rt,
            rt2,
            rn,
            size,
            extend64,
            immediate,
            wback,
            post_idx,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_mem_reg(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rt, rn, size) = mem_base(raw);
    let rm = GpRegisterIndex::from((raw >> 16) as u8);
    let extend64 = ((raw >> 22) & 3) == 2;
    let shift = ((raw >> 12) & 1) != 0;
    let int_type = IntegerType::from_raw((raw >> 13) & 0x7);
    Some((
        OpCodeKind::MemReg {
            rt,
            rn,
            rm,
            size,
            extend64,
            shift,
            int_type,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_mem_lit(raw: u32, addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rt = GpRegisterIndex::from(raw as u8);
    let immediate = addr as i64 + decode_imm_s19_2(raw);
    let (size, signed, prefetch) = match (raw >> 30) & 3 {
        0 => (2, false, false),
        1 => (3, false, false),
        2 => (2, true, false),
        _ => (0, false, true),
    };
    Some((
        OpCodeKind::MemLit {
            rt,
            size,
            immediate,
            signed,
            prefetch,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_mem_ex(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rt, rn, size) = mem_base(raw);
    let rt2 = GpRegisterIndex::from((raw >> 10) as u8);
    let rs = GpRegisterIndex::from((raw >> 16) as u8);
    Some((
        OpCodeKind::MemEx {
            rt,
            rt2,
            rn,
            rs,
            size,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_system(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rt = GpRegisterIndex::from(raw as u8);
    let op0 = (((raw >> 19) & 0x1) | 2) as u8;
    let op1 = ((raw >> 16) & 0x7) as u8;
    let crn = ((raw >> 12) & 0xf) as u8;
    let crm = ((raw >> 8) & 0xf) as u8;
    let op2 = ((raw >> 5) & 0x7) as u8;
    Some((
        OpCodeKind::System {
            rt,
            op0,
            op1,
            op2,
            crn,
            crm,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_exception(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let id = ((raw >> 5) & 0xffff) as u16;
    Some((OpCodeKind::Exception { id }, RegisterSize::Int64))
}

pub fn decode_simd(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, opc, size) = simd_base(raw);
    let reg_size = q_size(raw);
    Some((OpCodeKind::Simd { rd, rn, opc, size }, reg_size))
}

pub fn decode_simd_reg(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, rm, ra, opc, size, bit3) = simd_reg_base(raw);
    let reg_size = q_size(raw);
    Some((
        OpCodeKind::SimdReg {
            rd,
            rn,
            rm,
            ra,
            opc,
            size,
            bit3,
        },
        reg_size,
    ))
}

pub fn decode_simd_imm(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rd = VecRegisterIndex::from(raw as u8);
    let c_mode = ((raw >> 12) & 0xf) as u8;
    let op = ((raw >> 29) & 1) as u8;
    let mode_low = c_mode & 1;
    let mode_high = c_mode >> 1;

    let mut imm = (((raw >> 5) & 0x1f) | ((raw >> 11) & 0xe0)) as i64;

    let size = if mode_high == 0b111 {
        match op | (mode_low << 1) {
            0 => {
                // 64-bits Immediate.
                // Transform abcd efgh into abcd efgh abcd efgh ...
                imm = (imm as u64).wrapping_mul(0x0101010101010101) as i64;
                3
            }
            1 => {
                // 64-bits Immediate.
                // Transform abcd efgh into aaaa aaaa bbbb bbbb ...
                imm = (imm & 0xf0) >> 4 | (imm & 0x0f) << 4;
                imm = (imm & 0xcc) >> 2 | (imm & 0x33) << 2;
                imm = (imm & 0xaa) >> 1 | (imm & 0x55) << 1;
                imm = (imm as u64).wrapping_mul(0x8040201008040201) as i64;
                imm &= 0x8080808080808080u64 as i64;
                imm |= imm >> 4;
                imm |= imm >> 2;
                imm |= imm >> 1;
                3
            }
            2 => {
                // 2 x 32-bits floating point Immediate.
                imm = imm8_to_fp32_bits(imm as u8) as i64;
                imm |= imm << 32;
                0
            }
            _ => {
                // 64-bits floating point Immediate.
                imm = imm8_to_fp64_bits(imm as u8) as i64;
                1
            }
        }
    } else if (mode_high & 0b110) == 0b100 {
        // 16-bits shifted Immediate.
        imm <<= (mode_high & 1) << 3;
        1
    } else if (mode_high & 0b100) == 0b000 {
        // 32-bits shifted Immediate.
        imm <<= mode_high << 3;
        2
    } else if (mode_high & 0b111) == 0b110 {
        // 32-bits shifted Immediate (fill with ones).
        imm = shl_ones(imm, 8 << mode_low);
        2
    } else {
        // 8-bits without shift.
        0
    };

    let reg_size = q_size(raw);
    Some((
        OpCodeKind::SimdImm {
            rd,
            size,
            immediate: imm,
        },
        reg_size,
    ))
}

pub fn decode_simd_sh_imm(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, opc, _) = simd_base(raw);
    let imm = ((raw >> 16) & 0x7f) as u8;
    let size = highest_bit_set_nibble((imm >> 3) as i32) as i8;
    let reg_size = q_size(raw);
    Some((
        OpCodeKind::SimdShImm {
            rd,
            rn,
            opc,
            size,
            imm,
        },
        reg_size,
    ))
}

pub fn decode_simd_fmov(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rd = VecRegisterIndex::from(raw as u8);
    let type_ = ((raw >> 22) & 0x3) as u8;
    let imm = ((raw >> 13) & 0xff) as u8;
    let immediate = if type_ == 0 {
        imm8_to_fp32_bits(imm) as i64
    } else {
        imm8_to_fp64_bits(imm) as i64
    };
    Some((
        OpCodeKind::SimdFmov {
            rd,
            immediate,
            size: type_,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_simd_cvt(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, opc, size) = simd_base(raw);
    let scale = ((raw >> 10) & 0x3f) as u8;
    let fbits = 64u8 - scale;
    let reg_size = sf_size(raw);
    Some((
        OpCodeKind::SimdCvt {
            rd,
            rn,
            opc,
            size,
            fbits,
        },
        reg_size,
    ))
}

pub fn decode_simd_fcond(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, rm, ra, opc, size, bit3) = simd_reg_base(raw);
    let nzcv = (raw & 0xf) as u8;
    let cond = Condition::from_raw((raw >> 12) & 0xf);
    let reg_size = q_size(raw);
    Some((
        OpCodeKind::SimdFcond {
            rd,
            rn,
            rm,
            ra,
            opc,
            size,
            bit3,
            nzcv,
            cond,
        },
        reg_size,
    ))
}

pub fn decode_simd_ext(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, rm, ra, opc, size, bit3) = simd_reg_base(raw);
    let imm4 = ((raw >> 11) & 0xf) as u8;
    let reg_size = q_size(raw);
    Some((
        OpCodeKind::SimdExt {
            rd,
            rn,
            rm,
            ra,
            opc,
            size,
            bit3,
            imm4,
        },
        reg_size,
    ))
}

pub fn decode_simd_mem_pair(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rt = VecRegisterIndex::from((raw & 0x1f) as u8);
    let rt2 = VecRegisterIndex::from(((raw >> 10) & 0x1f) as u8);
    let rn = GpRegisterIndex::from(((raw >> 5) & 0x1f) as u8);
    let wback = (raw >> 23) & 1 != 0;
    let post_idx = (raw >> 23) & 3 == 1;
    let size = (((raw >> 30) & 3) + 2) as u8;
    let immediate = ((((raw >> 15) as i64) << 57) >> (57 - size as i64)) as i64;
    Some((
        OpCodeKind::SimdMemPair {
            rt,
            rt2,
            rn,
            size,
            immediate,
            wback,
            post_idx,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_simd_mem_imm(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (gpr_kind, _) = decode_mem_imm(raw, _addr)?;
    let OpCodeKind::MemImm {
        rn,
        size: base_size,
        immediate,
        wback,
        post_idx,
        unscaled,
        ..
    } = gpr_kind
    else {
        unreachable!()
    };
    let rt = VecRegisterIndex::from(raw as u8);
    let size = base_size | ((raw >> 21) & 4) as u8;

    if size > 4 {
        return None;
    }
    let immediate = if !wback && !unscaled && size == 4 {
        immediate << 4
    } else {
        immediate
    };
    Some((
        OpCodeKind::SimdMemImm {
            rt,
            rn,
            size,
            immediate,
            wback,
            post_idx,
            unscaled,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_simd_mem_reg(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (gpr_kind, _) = decode_mem_reg(raw, _addr)?;
    let OpCodeKind::MemReg {
        rn,
        rm,
        size: base_size,
        shift,
        int_type,
        ..
    } = gpr_kind
    else {
        unreachable!()
    };
    let rt = VecRegisterIndex::from(raw as u8);
    let size = base_size | ((raw >> 21) & 4) as u8;
    if size > 4 {
        return None;
    }
    Some((
        OpCodeKind::SimdMemReg {
            rt,
            rn,
            rm,
            size,
            shift,
            int_type,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_simd_mem_ms(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rt = VecRegisterIndex::from((raw & 0x1f) as u8);
    let rn = GpRegisterIndex::from(((raw >> 5) & 0x1f) as u8);
    let rm = GpRegisterIndex::from(((raw >> 16) & 0x1f) as u8);
    let wback = ((raw >> 23) & 1) != 0;
    let q = ((raw >> 30) & 1) != 0;

    let (reps, s_elems): (u8, u8) = match (raw >> 12) & 0xf {
        0b0000 => (1, 4),
        0b0010 => (4, 1),
        0b0100 => (1, 3),
        0b0110 => (3, 1),
        0b0111 => (1, 1),
        0b1000 => (1, 2),
        0b1010 => (2, 1),
        _ => {
            return None;
        }
    };

    let size = ((raw >> 10) & 3) as u8;

    if !q && size == 3 && s_elems != 1 {
        return None;
    }

    let reg_size = q_size(raw);
    let bits_count = reg_size.bits();
    let elems = ((bits_count >> 3) >> size) as u8;

    Some((
        OpCodeKind::SimdMemMs {
            rt,
            rn,
            rm,
            size,
            wback,
            reps,
            s_elems,
            elems,
        },
        reg_size,
    ))
}

pub fn decode_simd_mem_ss(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let rt = VecRegisterIndex::from((raw & 0x1f) as u8);
    let rn = GpRegisterIndex::from(((raw >> 5) & 0x1f) as u8);
    let rm = GpRegisterIndex::from(((raw >> 16) & 0x1f) as u8);
    let l = ((raw >> 22) & 1) != 0;
    let q = ((raw >> 30) & 1) != 0;

    let size = ((raw >> 10) & 3) as u8;
    let s = ((raw >> 12) & 1) as u8;
    let s_elems = (((raw >> 12) & 2) | ((raw >> 21) & 1)) as u8 + 1;
    let scale = ((raw >> 14) & 3) as u8;

    let mut index = ((q as u8) << 3) | (s << 2) | size;
    let mut size_out = scale;
    let mut replicate = false;

    match scale {
        1 => {
            if (size & 1) != 0 {
                return None;
            }
            index >>= 1;
        }
        2 => {
            if (size & 2) != 0 || ((size & 1) != 0 && s != 0) {
                return None;
            }
            if (size & 1) != 0 {
                index >>= 3;
                size_out = 3;
            } else {
                index >>= 2;
            }
        }
        3 => {
            if !l || s != 0 {
                return None;
            }
            size_out = size;
            replicate = true;
        }
        _ => {}
    }

    let reg_size = q_size(raw);

    Some((
        OpCodeKind::SimdMemSs {
            rt,
            rn,
            rm,
            size: size_out,
            wback: ((raw >> 23) & 1) != 0,
            s_elems,
            index,
            replicate,
        },
        reg_size,
    ))
}

pub fn decode_alu_binary(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, data_op, reg_size) = alu_base(raw);
    let rm = GpRegisterIndex::from((raw >> 16) as u8);
    Some((
        OpCodeKind::AluBinary {
            rd,
            rn,
            rm,
            data_op,
        },
        reg_size,
    ))
}

pub fn decode_simd_ins(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, _opc, _) = simd_base(raw);
    let imm4 = ((raw >> 11) & 0xf) as u8;
    let imm5 = ((raw >> 16) & 0x1f) as u8;
    let reg_size = q_size(raw);

    if imm5 == 0b10000 {
        return None;
    }

    let size = match imm5 & imm5.wrapping_neg() {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => return None,
    };

    let src_index = (imm4 >> size) as u8;
    let dst_index = (imm5 >> (size + 1)) as u8;

    Some((
        OpCodeKind::SimdIns {
            rd,
            rn,
            size,
            src_index,
            dst_index,
        },
        reg_size,
    ))
}

pub fn decode_simd_mem_lit(raw: u32, addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let opc = (raw >> 30) & 3;

    if opc == 3 {
        return None;
    }

    let rt = VecRegisterIndex::from(raw as u8);
    let immediate = addr as i64 + decode_imm_s19_2(raw);
    let size = (opc + 2) as u8;

    Some((
        OpCodeKind::SimdMemLit {
            rt,
            size,
            immediate,
        },
        RegisterSize::Int64,
    ))
}

pub fn decode_simd_reg_elem(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, rm, ra, opc, size, bit3) = simd_reg_base(raw);
    let reg_size = q_size(raw);

    let (rm, index) = match size {
        1 => {
            let index = (((raw >> 20) & 3) | ((raw >> 9) & 4)) as u8;
            let rm = VecRegisterIndex::from(rm.idx() & 0xf);
            (rm, index)
        }
        2 => {
            let index = (((raw >> 21) & 1) | ((raw >> 10) & 2)) as u8;
            (rm, index)
        }
        _ => return None,
    };

    Some((
        OpCodeKind::SimdRegElem {
            rd,
            rn,
            rm,
            ra,
            opc,
            size,
            bit3,
            index,
        },
        reg_size,
    ))
}

pub fn decode_simd_reg_elem_f(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, rm, ra, opc, size, bit3) = simd_reg_base(raw);
    let reg_size = q_size(raw);

    let index = match (raw >> 21) & 3 {
        0 => ((raw >> 10) & 2) as u8,
        1 => (((raw >> 10) & 2) as u8) + 1,
        2 => ((raw >> 11) & 1) as u8,
        _ => return None,
    };

    Some((
        OpCodeKind::SimdRegElemF {
            rd,
            rn,
            rm,
            ra,
            opc,
            size,
            bit3,
            index,
        },
        reg_size,
    ))
}

pub fn decode_simd_tbl(raw: u32, _addr: u64) -> Option<(OpCodeKind, RegisterSize)> {
    let (rd, rn, rm, ra, opc, _size, bit3) = simd_reg_base(raw);
    let reg_size = q_size(raw);
    let size = (((raw >> 13) & 3) + 1) as u8;

    Some((
        OpCodeKind::SimdTbl {
            rd,
            rn,
            rm,
            ra,
            opc,
            size,
            bit3,
        },
        reg_size,
    ))
}
