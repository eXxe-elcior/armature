use crate::decoder::decode_fns::*;
pub use crate::decoder::instruction_name::InstructionName;
use crate::decoder::table::RawEntry;
// Instruction-to-emitter mappings follow the C# OpCodeTable SetA64 entries.
use crate::instructions::alu::{
    emit_adc, emit_adcs, emit_add, emit_adds, emit_and, emit_ands, emit_asrv, emit_bic, emit_bics,
    emit_cls, emit_clz, emit_eon, emit_eor, emit_extr, emit_lslv, emit_lsrv, emit_orn, emit_orr,
    emit_rbit, emit_rev16, emit_rev32, emit_rev64, emit_rorv, emit_sbc, emit_sbcs, emit_sub,
    emit_subs,
};
// InstEmitBfm.cs
use crate::instructions::bfm::{emit_bfm, emit_sbfm, emit_ubfm};
// InstEmitCcmp.cs
use crate::instructions::ccmp::{emit_ccmn, emit_ccmp};
// InstEmitCsel.cs
use crate::instructions::csel::{emit_csel, emit_csinc, emit_csinv, emit_csneg};
// InstEmitDiv.cs
use crate::instructions::div::{emit_sdiv, emit_udiv};
// InstEmitException.cs (emit_und has no table entry  it backs the
// undefined-encoding fallback)
use crate::instructions::exceptions::{emit_brk, emit_svc};
use crate::instructions::flow::{
    emit_b, emit_b_cond, emit_bl, emit_blr, emit_br, emit_cbnz, emit_cbz, emit_ret, emit_tbnz,
    emit_tbz,
};
// InstEmitHash.cs
use crate::instructions::hash::{
    emit_crc32b, emit_crc32cb, emit_crc32ch, emit_crc32cw, emit_crc32cx, emit_crc32h, emit_crc32w,
    emit_crc32x,
};
// InstEmitMemory.cs
use crate::instructions::memory::{
    emit_adr, emit_adrp, emit_ldp, emit_ldr, emit_ldr_literal, emit_ldrs, emit_stp, emit_str,
};
// InstEmitMemoryEx.cs (Sev/Sevl/Wfe/Wfi/Yield map to emit_nop per C# table)
use crate::instructions::memory_ex::{
    emit_clrex, emit_csdb, emit_dmb, emit_dsb, emit_ldar, emit_ldaxp, emit_ldaxr, emit_ldxp,
    emit_ldxr, emit_prfm, emit_stlr, emit_stlxp, emit_stlxr, emit_stxp, emit_stxr,
};
// InstEmitMove.cs
use crate::instructions::mov::{emit_movk, emit_movn, emit_movz};
// InstEmitMul.cs
use crate::instructions::mul::{
    emit_madd, emit_msub, emit_smaddl, emit_smsubl, emit_smulh, emit_umaddl, emit_umsubl,
    emit_umulh,
};
// InstEmitSimdArithmetic.cs
use crate::instructions::simd::simd_arithmetic::*;
// InstEmitSimdCmp.cs
use crate::instructions::simd::simd_cmp::*;
// InstEmitSimdCrypto.cs
use crate::instructions::simd::simd_crypto::{
    emit_aesd_v, emit_aese_v, emit_aesimc_v, emit_aesmc_v,
};
// InstEmitSimdCvt.cs
use crate::instructions::simd::simd_cvt::{
    emit_fcvt_s, emit_fcvtas_gp, emit_fcvtas_s, emit_fcvtas_v, emit_fcvtau_gp, emit_fcvtau_s,
    emit_fcvtau_v, emit_fcvtl_v, emit_fcvtms_gp, emit_fcvtms_v, emit_fcvtmu_gp, emit_fcvtn_v,
    emit_fcvtns_gp, emit_fcvtns_s, emit_fcvtns_v, emit_fcvtnu_s, emit_fcvtnu_v, emit_fcvtps_gp,
    emit_fcvtpu_gp, emit_fcvtzs_gp, emit_fcvtzs_gp_fixed, emit_fcvtzs_s, emit_fcvtzs_v,
    emit_fcvtzs_v_fixed, emit_fcvtzu_gp, emit_fcvtzu_gp_fixed, emit_fcvtzu_s, emit_fcvtzu_v,
    emit_fcvtzu_v_fixed, emit_scvtf_gp, emit_scvtf_gp_fixed, emit_scvtf_s, emit_scvtf_s_fixed,
    emit_scvtf_v, emit_scvtf_v_fixed, emit_ucvtf_gp, emit_ucvtf_gp_fixed, emit_ucvtf_s,
    emit_ucvtf_s_fixed, emit_ucvtf_v, emit_ucvtf_v_fixed,
};
// InstEmitSimdHash.cs
use crate::instructions::simd::simd_hash::{
    emit_sha1c_v, emit_sha1h_v, emit_sha1m_v, emit_sha1p_v, emit_sha1su0_v, emit_sha1su1_v,
    emit_sha256h_v, emit_sha256h2_v, emit_sha256su0_v, emit_sha256su1_v,
};
// InstEmitSimdLogical.cs
use crate::instructions::simd::simd_logical::{
    emit_and_v, emit_bic_v, emit_bic_vi, emit_bif_v, emit_bit_v, emit_bsl_v, emit_eor_v,
    emit_not_v, emit_orn_v, emit_orr_v, emit_orr_vi, emit_rbit_v, emit_rev16_v, emit_rev32_v,
    emit_rev64_v,
};
// InstEmitSimdMemory.cs
use crate::instructions::simd::simd_memory::{
    emit_ld__vms, emit_ld__vss, emit_st__vms, emit_st__vss,
};
// InstEmitSimdMove.cs
use crate::instructions::simd::simd_mov::*;
// InstEmitSimdShift.cs
use crate::instructions::simd::simd_shift::*;
// InstEmitSystem.cs (Hint entries map to emit_nop, matching C# InstEmit.Nop)
use crate::instructions::system::{emit_isb, emit_mrs, emit_msr, emit_nop, emit_sys};

// Static raw table
pub static A64_RAW: &[RawEntry] = &[
    // x0011010000xxxxx000000xxxxxxxxxx  Adc  alu_rs
    RawEntry {
        pattern: "x0011010000xxxxx000000xxxxxxxxxx",
        name: InstructionName::Adc,
        emit: emit_adc,
        decode: decode_alu_rs,
    },
    // x0111010000xxxxx000000xxxxxxxxxx  Adcs  alu_rs
    RawEntry {
        pattern: "x0111010000xxxxx000000xxxxxxxxxx",
        name: InstructionName::Adcs,
        emit: emit_adcs,
        decode: decode_alu_rs,
    },
    // x00100010xxxxxxxxxxxxxxxxxxxxxxx  Add  alu_imm
    RawEntry {
        pattern: "x00100010xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Add,
        emit: emit_add,
        decode: decode_alu_imm,
    },
    // 00001011<<0xxxxx0xxxxxxxxxxxxxxx  Add  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "00001011<<0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Add,
        emit: emit_add,
        decode: decode_alu_rs,
    },
    // 10001011<<0xxxxxxxxxxxxxxxxxxxxx  Add  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "10001011<<0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Add,
        emit: emit_add,
        decode: decode_alu_rs,
    },
    // x0001011001xxxxxxxx0xxxxxxxxxxxx  Add  alu_rx
    RawEntry {
        pattern: "x0001011001xxxxxxxx0xxxxxxxxxxxx",
        name: InstructionName::Add,
        emit: emit_add,
        decode: decode_alu_rx,
    },
    // x0001011001xxxxxxxx100xxxxxxxxxx  Add  alu_rx
    RawEntry {
        pattern: "x0001011001xxxxxxxx100xxxxxxxxxx",
        name: InstructionName::Add,
        emit: emit_add,
        decode: decode_alu_rx,
    },
    // x01100010xxxxxxxxxxxxxxxxxxxxxxx  Adds  alu_imm
    RawEntry {
        pattern: "x01100010xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Adds,
        emit: emit_adds,
        decode: decode_alu_imm,
    },
    // 00101011<<0xxxxx0xxxxxxxxxxxxxxx  Adds  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "00101011<<0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Adds,
        emit: emit_adds,
        decode: decode_alu_rs,
    },
    // 10101011<<0xxxxxxxxxxxxxxxxxxxxx  Adds  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "10101011<<0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Adds,
        emit: emit_adds,
        decode: decode_alu_rs,
    },
    // x0101011001xxxxxxxx0xxxxxxxxxxxx  Adds  alu_rx
    RawEntry {
        pattern: "x0101011001xxxxxxxx0xxxxxxxxxxxx",
        name: InstructionName::Adds,
        emit: emit_adds,
        decode: decode_alu_rx,
    },
    // x0101011001xxxxxxxx100xxxxxxxxxx  Adds  alu_rx
    RawEntry {
        pattern: "x0101011001xxxxxxxx100xxxxxxxxxx",
        name: InstructionName::Adds,
        emit: emit_adds,
        decode: decode_alu_rx,
    },
    // 0xx10000xxxxxxxxxxxxxxxxxxxxxxxx  Adr  adr
    RawEntry {
        pattern: "0xx10000xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Adr,
        emit: emit_adr,
        decode: decode_adr,
    },
    // 1xx10000xxxxxxxxxxxxxxxxxxxxxxxx  Adrp  adr
    RawEntry {
        pattern: "1xx10000xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Adrp,
        emit: emit_adrp,
        decode: decode_adr,
    },
    // 0001001000xxxxxxxxxxxxxxxxxxxxxx  And  alu_imm
    RawEntry {
        pattern: "0001001000xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::And,
        emit: emit_and,
        decode: decode_alu_imm,
    },
    // 100100100xxxxxxxxxxxxxxxxxxxxxxx  And  alu_imm
    RawEntry {
        pattern: "100100100xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::And,
        emit: emit_and,
        decode: decode_alu_imm,
    },
    // 00001010xx0xxxxx0xxxxxxxxxxxxxxx  And  alu_rs
    RawEntry {
        pattern: "00001010xx0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::And,
        emit: emit_and,
        decode: decode_alu_rs,
    },
    // 10001010xx0xxxxxxxxxxxxxxxxxxxxx  And  alu_rs
    RawEntry {
        pattern: "10001010xx0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::And,
        emit: emit_and,
        decode: decode_alu_rs,
    },
    // 0111001000xxxxxxxxxxxxxxxxxxxxxx  Ands  alu_imm
    RawEntry {
        pattern: "0111001000xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ands,
        emit: emit_ands,
        decode: decode_alu_imm,
    },
    // 111100100xxxxxxxxxxxxxxxxxxxxxxx  Ands  alu_imm
    RawEntry {
        pattern: "111100100xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ands,
        emit: emit_ands,
        decode: decode_alu_imm,
    },
    // 01101010xx0xxxxx0xxxxxxxxxxxxxxx  Ands  alu_rs
    RawEntry {
        pattern: "01101010xx0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Ands,
        emit: emit_ands,
        decode: decode_alu_rs,
    },
    // 11101010xx0xxxxxxxxxxxxxxxxxxxxx  Ands  alu_rs
    RawEntry {
        pattern: "11101010xx0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ands,
        emit: emit_ands,
        decode: decode_alu_rs,
    },
    // x0011010110xxxxx001010xxxxxxxxxx  Asrv  alu_rs
    RawEntry {
        pattern: "x0011010110xxxxx001010xxxxxxxxxx",
        name: InstructionName::Asrv,
        emit: emit_asrv,
        decode: decode_alu_rs,
    },
    // 000101xxxxxxxxxxxxxxxxxxxxxxxxxx  B  b_imm_al
    RawEntry {
        pattern: "000101xxxxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::B,
        emit: emit_b,
        decode: decode_b_imm_al,
    },
    // 01010100xxxxxxxxxxxxxxxxxxx0xxxx  B_Cond  b_imm_cond
    RawEntry {
        pattern: "01010100xxxxxxxxxxxxxxxxxxx0xxxx",
        name: InstructionName::B_Cond,
        emit: emit_b_cond,
        decode: decode_b_imm_cond,
    },
    // 00110011000xxxxx0xxxxxxxxxxxxxxx  Bfm  bfm
    RawEntry {
        pattern: "00110011000xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Bfm,
        emit: emit_bfm,
        decode: decode_bfm,
    },
    // 1011001101xxxxxxxxxxxxxxxxxxxxxx  Bfm  bfm
    RawEntry {
        pattern: "1011001101xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Bfm,
        emit: emit_bfm,
        decode: decode_bfm,
    },
    // 00001010xx1xxxxx0xxxxxxxxxxxxxxx  Bic  alu_rs
    RawEntry {
        pattern: "00001010xx1xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Bic,
        emit: emit_bic,
        decode: decode_alu_rs,
    },
    // 10001010xx1xxxxxxxxxxxxxxxxxxxxx  Bic  alu_rs
    RawEntry {
        pattern: "10001010xx1xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Bic,
        emit: emit_bic,
        decode: decode_alu_rs,
    },
    // 01101010xx1xxxxx0xxxxxxxxxxxxxxx  Bics  alu_rs
    RawEntry {
        pattern: "01101010xx1xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Bics,
        emit: emit_bics,
        decode: decode_alu_rs,
    },
    // 11101010xx1xxxxxxxxxxxxxxxxxxxxx  Bics  alu_rs
    RawEntry {
        pattern: "11101010xx1xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Bics,
        emit: emit_bics,
        decode: decode_alu_rs,
    },
    // 100101xxxxxxxxxxxxxxxxxxxxxxxxxx  Bl  b_imm_al
    RawEntry {
        pattern: "100101xxxxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Bl,
        emit: emit_bl,
        decode: decode_b_imm_al,
    },
    // 1101011000111111000000xxxxx00000  Blr  b_reg
    RawEntry {
        pattern: "1101011000111111000000xxxxx00000",
        name: InstructionName::Blr,
        emit: emit_blr,
        decode: decode_b_reg,
    },
    // 1101011000011111000000xxxxx00000  Br  b_reg
    RawEntry {
        pattern: "1101011000011111000000xxxxx00000",
        name: InstructionName::Br,
        emit: emit_br,
        decode: decode_b_reg,
    },
    // 11010100001xxxxxxxxxxxxxxxx00000  Brk  exception
    RawEntry {
        pattern: "11010100001xxxxxxxxxxxxxxxx00000",
        name: InstructionName::Brk,
        emit: emit_brk,
        decode: decode_exception,
    },
    // x0110101xxxxxxxxxxxxxxxxxxxxxxxx  Cbnz  b_imm_cmp
    RawEntry {
        pattern: "x0110101xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Cbnz,
        emit: emit_cbnz,
        decode: decode_b_imm_cmp,
    },
    // x0110100xxxxxxxxxxxxxxxxxxxxxxxx  Cbz  b_imm_cmp
    RawEntry {
        pattern: "x0110100xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Cbz,
        emit: emit_cbz,
        decode: decode_b_imm_cmp,
    },
    // x0111010010xxxxxxxxx10xxxxx0xxxx  Ccmn  ccmp_imm
    RawEntry {
        pattern: "x0111010010xxxxxxxxx10xxxxx0xxxx",
        name: InstructionName::Ccmn,
        emit: emit_ccmn,
        decode: decode_ccmp_imm,
    },
    // x0111010010xxxxxxxxx00xxxxx0xxxx  Ccmn  ccmp_reg
    RawEntry {
        pattern: "x0111010010xxxxxxxxx00xxxxx0xxxx",
        name: InstructionName::Ccmn,
        emit: emit_ccmn,
        decode: decode_ccmp_reg,
    },
    // x1111010010xxxxxxxxx10xxxxx0xxxx  Ccmp  ccmp_imm
    RawEntry {
        pattern: "x1111010010xxxxxxxxx10xxxxx0xxxx",
        name: InstructionName::Ccmp,
        emit: emit_ccmp,
        decode: decode_ccmp_imm,
    },
    // x1111010010xxxxxxxxx00xxxxx0xxxx  Ccmp  ccmp_reg
    RawEntry {
        pattern: "x1111010010xxxxxxxxx00xxxxx0xxxx",
        name: InstructionName::Ccmp,
        emit: emit_ccmp,
        decode: decode_ccmp_reg,
    },
    // 11010101000000110011xxxx01011111  Clrex  system
    RawEntry {
        pattern: "11010101000000110011xxxx01011111",
        name: InstructionName::Clrex,
        emit: emit_clrex,
        decode: decode_system,
    },
    // x101101011000000000101xxxxxxxxxx  Cls  alu
    RawEntry {
        pattern: "x101101011000000000101xxxxxxxxxx",
        name: InstructionName::Cls,
        emit: emit_cls,
        decode: decode_alu,
    },
    // x101101011000000000100xxxxxxxxxx  Clz  alu
    RawEntry {
        pattern: "x101101011000000000100xxxxxxxxxx",
        name: InstructionName::Clz,
        emit: emit_clz,
        decode: decode_alu,
    },
    // 00011010110xxxxx010000xxxxxxxxxx  Crc32b  alu_binary
    RawEntry {
        pattern: "00011010110xxxxx010000xxxxxxxxxx",
        name: InstructionName::Crc32B,
        emit: emit_crc32b,
        decode: decode_alu_binary,
    },
    // 00011010110xxxxx010001xxxxxxxxxx  Crc32h  alu_binary
    RawEntry {
        pattern: "00011010110xxxxx010001xxxxxxxxxx",
        name: InstructionName::Crc32H,
        emit: emit_crc32h,
        decode: decode_alu_binary,
    },
    // 00011010110xxxxx010010xxxxxxxxxx  Crc32w  alu_binary
    RawEntry {
        pattern: "00011010110xxxxx010010xxxxxxxxxx",
        name: InstructionName::Crc32W,
        emit: emit_crc32w,
        decode: decode_alu_binary,
    },
    // 10011010110xxxxx010011xxxxxxxxxx  Crc32x  alu_binary
    RawEntry {
        pattern: "10011010110xxxxx010011xxxxxxxxxx",
        name: InstructionName::Crc32X,
        emit: emit_crc32x,
        decode: decode_alu_binary,
    },
    // 00011010110xxxxx010100xxxxxxxxxx  Crc32cb  alu_binary
    RawEntry {
        pattern: "00011010110xxxxx010100xxxxxxxxxx",
        name: InstructionName::Crc32Cb,
        emit: emit_crc32cb,
        decode: decode_alu_binary,
    },
    // 00011010110xxxxx010101xxxxxxxxxx  Crc32ch  alu_binary
    RawEntry {
        pattern: "00011010110xxxxx010101xxxxxxxxxx",
        name: InstructionName::Crc32Ch,
        emit: emit_crc32ch,
        decode: decode_alu_binary,
    },
    // 00011010110xxxxx010110xxxxxxxxxx  Crc32cw  alu_binary
    RawEntry {
        pattern: "00011010110xxxxx010110xxxxxxxxxx",
        name: InstructionName::Crc32Cw,
        emit: emit_crc32cw,
        decode: decode_alu_binary,
    },
    // 10011010110xxxxx010111xxxxxxxxxx  Crc32cx  alu_binary
    RawEntry {
        pattern: "10011010110xxxxx010111xxxxxxxxxx",
        name: InstructionName::Crc32Cx,
        emit: emit_crc32cx,
        decode: decode_alu_binary,
    },
    // 11010101000000110010001010011111  Csdb  system
    RawEntry {
        pattern: "11010101000000110010001010011111",
        name: InstructionName::Csdb,
        emit: emit_csdb,
        decode: decode_system,
    },
    // x0011010100xxxxxxxxx00xxxxxxxxxx  Csel  csel
    RawEntry {
        pattern: "x0011010100xxxxxxxxx00xxxxxxxxxx",
        name: InstructionName::Csel,
        emit: emit_csel,
        decode: decode_csel,
    },
    // x0011010100xxxxxxxxx01xxxxxxxxxx  Csinc  csel
    RawEntry {
        pattern: "x0011010100xxxxxxxxx01xxxxxxxxxx",
        name: InstructionName::Csinc,
        emit: emit_csinc,
        decode: decode_csel,
    },
    // x1011010100xxxxxxxxx00xxxxxxxxxx  Csinv  csel
    RawEntry {
        pattern: "x1011010100xxxxxxxxx00xxxxxxxxxx",
        name: InstructionName::Csinv,
        emit: emit_csinv,
        decode: decode_csel,
    },
    // x1011010100xxxxxxxxx01xxxxxxxxxx  Csneg  csel
    RawEntry {
        pattern: "x1011010100xxxxxxxxx01xxxxxxxxxx",
        name: InstructionName::Csneg,
        emit: emit_csneg,
        decode: decode_csel,
    },
    // 11010101000000110011xxxx10111111  Dmb  system
    RawEntry {
        pattern: "11010101000000110011xxxx10111111",
        name: InstructionName::Dmb,
        emit: emit_dmb,
        decode: decode_system,
    },
    // 11010101000000110011xxxx10011111  Dsb  system
    RawEntry {
        pattern: "11010101000000110011xxxx10011111",
        name: InstructionName::Dsb,
        emit: emit_dsb,
        decode: decode_system,
    },
    // 01001010xx1xxxxx0xxxxxxxxxxxxxxx  Eon  alu_rs
    RawEntry {
        pattern: "01001010xx1xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Eon,
        emit: emit_eon,
        decode: decode_alu_rs,
    },
    // 11001010xx1xxxxxxxxxxxxxxxxxxxxx  Eon  alu_rs
    RawEntry {
        pattern: "11001010xx1xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Eon,
        emit: emit_eon,
        decode: decode_alu_rs,
    },
    // 0101001000xxxxxxxxxxxxxxxxxxxxxx  Eor  alu_imm
    RawEntry {
        pattern: "0101001000xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Eor,
        emit: emit_eor,
        decode: decode_alu_imm,
    },
    // 110100100xxxxxxxxxxxxxxxxxxxxxxx  Eor  alu_imm
    RawEntry {
        pattern: "110100100xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Eor,
        emit: emit_eor,
        decode: decode_alu_imm,
    },
    // 01001010xx0xxxxx0xxxxxxxxxxxxxxx  Eor  alu_rs
    RawEntry {
        pattern: "01001010xx0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Eor,
        emit: emit_eor,
        decode: decode_alu_rs,
    },
    // 11001010xx0xxxxxxxxxxxxxxxxxxxxx  Eor  alu_rs
    RawEntry {
        pattern: "11001010xx0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Eor,
        emit: emit_eor,
        decode: decode_alu_rs,
    },
    // 00010011100xxxxx0xxxxxxxxxxxxxxx  Extr  alu_rs
    RawEntry {
        pattern: "00010011100xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Extr,
        emit: emit_extr,
        decode: decode_alu_rs,
    },
    // 10010011110xxxxxxxxxxxxxxxxxxxxx  Extr  alu_rs
    RawEntry {
        pattern: "10010011110xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Extr,
        emit: emit_extr,
        decode: decode_alu_rs,
    },
    // 11010101000000110010000011011111  Hint  system
    RawEntry {
        pattern: "11010101000000110010000011011111",
        name: InstructionName::Hint,
        emit: emit_nop,
        decode: decode_system,
    },
    // 11010101000000110010000011111111  Hint  system
    RawEntry {
        pattern: "11010101000000110010000011111111",
        name: InstructionName::Hint,
        emit: emit_nop,
        decode: decode_system,
    },
    // 110101010000001100100001xxx11111  Hint  system
    RawEntry {
        pattern: "110101010000001100100001xxx11111",
        name: InstructionName::Hint,
        emit: emit_nop,
        decode: decode_system,
    },
    // 1101010100000011001000100xx11111  Hint  system
    RawEntry {
        pattern: "1101010100000011001000100xx11111",
        name: InstructionName::Hint,
        emit: emit_nop,
        decode: decode_system,
    },
    // 1101010100000011001000101>>11111  Hint  system  (>> at ...)
    RawEntry {
        pattern: "1101010100000011001000101>>11111",
        name: InstructionName::Hint,
        emit: emit_nop,
        decode: decode_system,
    },
    // 110101010000001100100011xxx11111  Hint  system
    RawEntry {
        pattern: "110101010000001100100011xxx11111",
        name: InstructionName::Hint,
        emit: emit_nop,
        decode: decode_system,
    },
    // 11010101000000110010>>xxxxx11111  Hint  system  (>> at ...)
    RawEntry {
        pattern: "11010101000000110010>>xxxxx11111",
        name: InstructionName::Hint,
        emit: emit_nop,
        decode: decode_system,
    },
    // 11010101000000110011xxxx11011111  Isb  system
    RawEntry {
        pattern: "11010101000000110011xxxx11011111",
        name: InstructionName::Isb,
        emit: emit_isb,
        decode: decode_system,
    },
    // xx001000110xxxxx1xxxxxxxxxxxxxxx  Ldar  mem_ex
    RawEntry {
        pattern: "xx001000110xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Ldar,
        emit: emit_ldar,
        decode: decode_mem_ex,
    },
    // 1x001000011xxxxx1xxxxxxxxxxxxxxx  Ldaxp  mem_ex
    RawEntry {
        pattern: "1x001000011xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Ldaxp,
        emit: emit_ldaxp,
        decode: decode_mem_ex,
    },
    // xx001000010xxxxx1xxxxxxxxxxxxxxx  Ldaxr  mem_ex
    RawEntry {
        pattern: "xx001000010xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Ldaxr,
        emit: emit_ldaxr,
        decode: decode_mem_ex,
    },
    // <<10100xx1xxxxxxxxxxxxxxxxxxxxxx  Ldp  mem_pair  (<< at 31,30)
    RawEntry {
        pattern: "<<10100xx1xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldp,
        emit: emit_ldp,
        decode: decode_mem_pair,
    },
    // xx111000010xxxxxxxxxxxxxxxxxxxxx  Ldr  mem_imm
    RawEntry {
        pattern: "xx111000010xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_mem_imm,
    },
    // xx11100101xxxxxxxxxxxxxxxxxxxxxx  Ldr  mem_imm
    RawEntry {
        pattern: "xx11100101xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_mem_imm,
    },
    // xx111000011xxxxxxxxx10xxxxxxxxxx  Ldr  mem_reg
    RawEntry {
        pattern: "xx111000011xxxxxxxxx10xxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_mem_reg,
    },
    // xx011000xxxxxxxxxxxxxxxxxxxxxxxx  Ldr_Literal  mem_lit
    RawEntry {
        pattern: "xx011000xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldr_Literal,
        emit: emit_ldr_literal,
        decode: decode_mem_lit,
    },
    // 0x1110001x0xxxxxxxxxxxxxxxxxxxxx  Ldrs  mem_imm
    RawEntry {
        pattern: "0x1110001x0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldrs,
        emit: emit_ldrs,
        decode: decode_mem_imm,
    },
    // 0x1110011xxxxxxxxxxxxxxxxxxxxxxx  Ldrs  mem_imm
    RawEntry {
        pattern: "0x1110011xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldrs,
        emit: emit_ldrs,
        decode: decode_mem_imm,
    },
    // 10111000100xxxxxxxxxxxxxxxxxxxxx  Ldrs  mem_imm
    RawEntry {
        pattern: "10111000100xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldrs,
        emit: emit_ldrs,
        decode: decode_mem_imm,
    },
    // 1011100110xxxxxxxxxxxxxxxxxxxxxx  Ldrs  mem_imm
    RawEntry {
        pattern: "1011100110xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldrs,
        emit: emit_ldrs,
        decode: decode_mem_imm,
    },
    // 0x1110001x1xxxxxxxxx10xxxxxxxxxx  Ldrs  mem_reg
    RawEntry {
        pattern: "0x1110001x1xxxxxxxxx10xxxxxxxxxx",
        name: InstructionName::Ldrs,
        emit: emit_ldrs,
        decode: decode_mem_reg,
    },
    // 10111000101xxxxxxxxx10xxxxxxxxxx  Ldrs  mem_reg
    RawEntry {
        pattern: "10111000101xxxxxxxxx10xxxxxxxxxx",
        name: InstructionName::Ldrs,
        emit: emit_ldrs,
        decode: decode_mem_reg,
    },
    // xx001000010xxxxx0xxxxxxxxxxxxxxx  Ldxr  mem_ex
    RawEntry {
        pattern: "xx001000010xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Ldxr,
        emit: emit_ldxr,
        decode: decode_mem_ex,
    },
    // 1x001000011xxxxx0xxxxxxxxxxxxxxx  Ldxp  mem_ex
    RawEntry {
        pattern: "1x001000011xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Ldxp,
        emit: emit_ldxp,
        decode: decode_mem_ex,
    },
    // x0011010110xxxxx001000xxxxxxxxxx  Lslv  alu_rs
    RawEntry {
        pattern: "x0011010110xxxxx001000xxxxxxxxxx",
        name: InstructionName::Lslv,
        emit: emit_lslv,
        decode: decode_alu_rs,
    },
    // x0011010110xxxxx001001xxxxxxxxxx  Lsrv  alu_rs
    RawEntry {
        pattern: "x0011010110xxxxx001001xxxxxxxxxx",
        name: InstructionName::Lsrv,
        emit: emit_lsrv,
        decode: decode_alu_rs,
    },
    // x0011011000xxxxx0xxxxxxxxxxxxxxx  Madd  mul
    RawEntry {
        pattern: "x0011011000xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Madd,
        emit: emit_madd,
        decode: decode_mul,
    },
    // 0111001010xxxxxxxxxxxxxxxxxxxxxx  Movk  mov
    RawEntry {
        pattern: "0111001010xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Movk,
        emit: emit_movk,
        decode: decode_mov,
    },
    // 111100101xxxxxxxxxxxxxxxxxxxxxxx  Movk  mov
    RawEntry {
        pattern: "111100101xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Movk,
        emit: emit_movk,
        decode: decode_mov,
    },
    // 0001001010xxxxxxxxxxxxxxxxxxxxxx  Movn  mov
    RawEntry {
        pattern: "0001001010xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Movn,
        emit: emit_movn,
        decode: decode_mov,
    },
    // 100100101xxxxxxxxxxxxxxxxxxxxxxx  Movn  mov
    RawEntry {
        pattern: "100100101xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Movn,
        emit: emit_movn,
        decode: decode_mov,
    },
    // 0101001010xxxxxxxxxxxxxxxxxxxxxx  Movz  mov
    RawEntry {
        pattern: "0101001010xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Movz,
        emit: emit_movz,
        decode: decode_mov,
    },
    // 110100101xxxxxxxxxxxxxxxxxxxxxxx  Movz  mov
    RawEntry {
        pattern: "110100101xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Movz,
        emit: emit_movz,
        decode: decode_mov,
    },
    // 110101010011xxxxxxxxxxxxxxxxxxxx  Mrs  system
    RawEntry {
        pattern: "110101010011xxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Mrs,
        emit: emit_mrs,
        decode: decode_system,
    },
    // 110101010001xxxxxxxxxxxxxxxxxxxx  Msr  system
    RawEntry {
        pattern: "110101010001xxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Msr,
        emit: emit_msr,
        decode: decode_system,
    },
    // x0011011000xxxxx1xxxxxxxxxxxxxxx  Msub  mul
    RawEntry {
        pattern: "x0011011000xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Msub,
        emit: emit_msub,
        decode: decode_mul,
    },
    // 11010101000000110010000000011111  Nop  system
    RawEntry {
        pattern: "11010101000000110010000000011111",
        name: InstructionName::Nop,
        emit: emit_nop,
        decode: decode_system,
    },
    // 00101010xx1xxxxx0xxxxxxxxxxxxxxx  Orn  alu_rs
    RawEntry {
        pattern: "00101010xx1xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Orn,
        emit: emit_orn,
        decode: decode_alu_rs,
    },
    // 10101010xx1xxxxxxxxxxxxxxxxxxxxx  Orn  alu_rs
    RawEntry {
        pattern: "10101010xx1xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Orn,
        emit: emit_orn,
        decode: decode_alu_rs,
    },
    // 0011001000xxxxxxxxxxxxxxxxxxxxxx  Orr  alu_imm
    RawEntry {
        pattern: "0011001000xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Orr,
        emit: emit_orr,
        decode: decode_alu_imm,
    },
    // 101100100xxxxxxxxxxxxxxxxxxxxxxx  Orr  alu_imm
    RawEntry {
        pattern: "101100100xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Orr,
        emit: emit_orr,
        decode: decode_alu_imm,
    },
    // 00101010xx0xxxxx0xxxxxxxxxxxxxxx  Orr  alu_rs
    RawEntry {
        pattern: "00101010xx0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Orr,
        emit: emit_orr,
        decode: decode_alu_rs,
    },
    // 10101010xx0xxxxxxxxxxxxxxxxxxxxx  Orr  alu_rs
    RawEntry {
        pattern: "10101010xx0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Orr,
        emit: emit_orr,
        decode: decode_alu_rs,
    },
    // 1111100110xxxxxxxxxxxxxxxxxxxxxx  Prfm  mem_imm
    RawEntry {
        pattern: "1111100110xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Prfm,
        emit: emit_prfm,
        decode: decode_mem_imm,
    },
    // 11111000100xxxxxxxxx00xxxxxxxxxx  Prfm  mem_imm
    RawEntry {
        pattern: "11111000100xxxxxxxxx00xxxxxxxxxx",
        name: InstructionName::Prfm,
        emit: emit_prfm,
        decode: decode_mem_imm,
    },
    // 11011000xxxxxxxxxxxxxxxxxxxxxxxx  Prfm  mem_lit
    RawEntry {
        pattern: "11011000xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Prfm,
        emit: emit_prfm,
        decode: decode_mem_lit,
    },
    // 11111000101xxxxxxxxx10xxxxxxxxxx  Prfm  mem_reg
    RawEntry {
        pattern: "11111000101xxxxxxxxx10xxxxxxxxxx",
        name: InstructionName::Prfm,
        emit: emit_prfm,
        decode: decode_mem_reg,
    },
    // x101101011000000000000xxxxxxxxxx  Rbit  alu
    RawEntry {
        pattern: "x101101011000000000000xxxxxxxxxx",
        name: InstructionName::Rbit,
        emit: emit_rbit,
        decode: decode_alu,
    },
    // 1101011001011111000000xxxxx00000  Ret  b_reg
    RawEntry {
        pattern: "1101011001011111000000xxxxx00000",
        name: InstructionName::Ret,
        emit: emit_ret,
        decode: decode_b_reg,
    },
    // x101101011000000000001xxxxxxxxxx  Rev16  alu
    RawEntry {
        pattern: "x101101011000000000001xxxxxxxxxx",
        name: InstructionName::Rev16,
        emit: emit_rev16,
        decode: decode_alu,
    },
    // x101101011000000000010xxxxxxxxxx  Rev32  alu
    RawEntry {
        pattern: "x101101011000000000010xxxxxxxxxx",
        name: InstructionName::Rev32,
        emit: emit_rev32,
        decode: decode_alu,
    },
    // 1101101011000000000011xxxxxxxxxx  Rev64  alu
    RawEntry {
        pattern: "1101101011000000000011xxxxxxxxxx",
        name: InstructionName::Rev64,
        emit: emit_rev64,
        decode: decode_alu,
    },
    // x0011010110xxxxx001011xxxxxxxxxx  Rorv  alu_rs
    RawEntry {
        pattern: "x0011010110xxxxx001011xxxxxxxxxx",
        name: InstructionName::Rorv,
        emit: emit_rorv,
        decode: decode_alu_rs,
    },
    // x1011010000xxxxx000000xxxxxxxxxx  Sbc  alu_rs
    RawEntry {
        pattern: "x1011010000xxxxx000000xxxxxxxxxx",
        name: InstructionName::Sbc,
        emit: emit_sbc,
        decode: decode_alu_rs,
    },
    // x1111010000xxxxx000000xxxxxxxxxx  Sbcs  alu_rs
    RawEntry {
        pattern: "x1111010000xxxxx000000xxxxxxxxxx",
        name: InstructionName::Sbcs,
        emit: emit_sbcs,
        decode: decode_alu_rs,
    },
    // 00010011000xxxxx0xxxxxxxxxxxxxxx  Sbfm  bfm
    RawEntry {
        pattern: "00010011000xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Sbfm,
        emit: emit_sbfm,
        decode: decode_bfm,
    },
    // 1001001101xxxxxxxxxxxxxxxxxxxxxx  Sbfm  bfm
    RawEntry {
        pattern: "1001001101xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Sbfm,
        emit: emit_sbfm,
        decode: decode_bfm,
    },
    // x0011010110xxxxx000011xxxxxxxxxx  Sdiv  alu_binary
    RawEntry {
        pattern: "x0011010110xxxxx000011xxxxxxxxxx",
        name: InstructionName::Sdiv,
        emit: emit_sdiv,
        decode: decode_alu_binary,
    },
    // 11010101000000110010000010011111  Sev  system
    RawEntry {
        pattern: "11010101000000110010000010011111",
        name: InstructionName::Sev,
        emit: emit_nop,
        decode: decode_system,
    },
    // 11010101000000110010000010111111  Sevl  system
    RawEntry {
        pattern: "11010101000000110010000010111111",
        name: InstructionName::Sevl,
        emit: emit_nop,
        decode: decode_system,
    },
    // 10011011001xxxxx0xxxxxxxxxxxxxxx  Smaddl  mul
    RawEntry {
        pattern: "10011011001xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Smaddl,
        emit: emit_smaddl,
        decode: decode_mul,
    },
    // 10011011001xxxxx1xxxxxxxxxxxxxxx  Smsubl  mul
    RawEntry {
        pattern: "10011011001xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Smsubl,
        emit: emit_smsubl,
        decode: decode_mul,
    },
    // 10011011010xxxxx0xxxxxxxxxxxxxxx  Smulh  mul
    RawEntry {
        pattern: "10011011010xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Smulh,
        emit: emit_smulh,
        decode: decode_mul,
    },
    // xx001000100xxxxx1xxxxxxxxxxxxxxx  Stlr  mem_ex
    RawEntry {
        pattern: "xx001000100xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Stlr,
        emit: emit_stlr,
        decode: decode_mem_ex,
    },
    // 1x001000001xxxxx1xxxxxxxxxxxxxxx  Stlxp  mem_ex
    RawEntry {
        pattern: "1x001000001xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Stlxp,
        emit: emit_stlxp,
        decode: decode_mem_ex,
    },
    // xx001000000xxxxx1xxxxxxxxxxxxxxx  Stlxr  mem_ex
    RawEntry {
        pattern: "xx001000000xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Stlxr,
        emit: emit_stlxr,
        decode: decode_mem_ex,
    },
    // x010100xx0xxxxxxxxxxxxxxxxxxxxxx  Stp  mem_pair
    RawEntry {
        pattern: "x010100xx0xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Stp,
        emit: emit_stp,
        decode: decode_mem_pair,
    },
    // xx111000000xxxxxxxxxxxxxxxxxxxxx  Str  mem_imm
    RawEntry {
        pattern: "xx111000000xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_mem_imm,
    },
    // xx11100100xxxxxxxxxxxxxxxxxxxxxx  Str  mem_imm
    RawEntry {
        pattern: "xx11100100xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_mem_imm,
    },
    // xx111000001xxxxxxxxx10xxxxxxxxxx  Str  mem_reg
    RawEntry {
        pattern: "xx111000001xxxxxxxxx10xxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_mem_reg,
    },
    // 1x001000001xxxxx0xxxxxxxxxxxxxxx  Stxp  mem_ex
    RawEntry {
        pattern: "1x001000001xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Stxp,
        emit: emit_stxp,
        decode: decode_mem_ex,
    },
    // xx001000000xxxxx0xxxxxxxxxxxxxxx  Stxr  mem_ex
    RawEntry {
        pattern: "xx001000000xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Stxr,
        emit: emit_stxr,
        decode: decode_mem_ex,
    },
    // x10100010xxxxxxxxxxxxxxxxxxxxxxx  Sub  alu_imm
    RawEntry {
        pattern: "x10100010xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Sub,
        emit: emit_sub,
        decode: decode_alu_imm,
    },
    // 01001011<<0xxxxx0xxxxxxxxxxxxxxx  Sub  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "01001011<<0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Sub,
        emit: emit_sub,
        decode: decode_alu_rs,
    },
    // 11001011<<0xxxxxxxxxxxxxxxxxxxxx  Sub  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "11001011<<0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Sub,
        emit: emit_sub,
        decode: decode_alu_rs,
    },
    // x1001011001xxxxxxxx0xxxxxxxxxxxx  Sub  alu_rx
    RawEntry {
        pattern: "x1001011001xxxxxxxx0xxxxxxxxxxxx",
        name: InstructionName::Sub,
        emit: emit_sub,
        decode: decode_alu_rx,
    },
    // x1001011001xxxxxxxx100xxxxxxxxxx  Sub  alu_rx
    RawEntry {
        pattern: "x1001011001xxxxxxxx100xxxxxxxxxx",
        name: InstructionName::Sub,
        emit: emit_sub,
        decode: decode_alu_rx,
    },
    // x11100010xxxxxxxxxxxxxxxxxxxxxxx  Subs  alu_imm
    RawEntry {
        pattern: "x11100010xxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Subs,
        emit: emit_subs,
        decode: decode_alu_imm,
    },
    // 01101011<<0xxxxx0xxxxxxxxxxxxxxx  Subs  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "01101011<<0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Subs,
        emit: emit_subs,
        decode: decode_alu_rs,
    },
    // 11101011<<0xxxxxxxxxxxxxxxxxxxxx  Subs  alu_rs  (<< at 23,22)
    RawEntry {
        pattern: "11101011<<0xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Subs,
        emit: emit_subs,
        decode: decode_alu_rs,
    },
    // x1101011001xxxxxxxx0xxxxxxxxxxxx  Subs  alu_rx
    RawEntry {
        pattern: "x1101011001xxxxxxxx0xxxxxxxxxxxx",
        name: InstructionName::Subs,
        emit: emit_subs,
        decode: decode_alu_rx,
    },
    // x1101011001xxxxxxxx100xxxxxxxxxx  Subs  alu_rx
    RawEntry {
        pattern: "x1101011001xxxxxxxx100xxxxxxxxxx",
        name: InstructionName::Subs,
        emit: emit_subs,
        decode: decode_alu_rx,
    },
    // 11010100000xxxxxxxxxxxxxxxx00001  Svc  exception
    RawEntry {
        pattern: "11010100000xxxxxxxxxxxxxxxx00001",
        name: InstructionName::Svc,
        emit: emit_svc,
        decode: decode_exception,
    },
    // 1101010100001xxxxxxxxxxxxxxxxxxx  Sys  system
    RawEntry {
        pattern: "1101010100001xxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Sys,
        emit: emit_sys,
        decode: decode_system,
    },
    // x0110111xxxxxxxxxxxxxxxxxxxxxxxx  Tbnz  b_imm_test
    RawEntry {
        pattern: "x0110111xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Tbnz,
        emit: emit_tbnz,
        decode: decode_b_imm_test,
    },
    // x0110110xxxxxxxxxxxxxxxxxxxxxxxx  Tbz  b_imm_test
    RawEntry {
        pattern: "x0110110xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Tbz,
        emit: emit_tbz,
        decode: decode_b_imm_test,
    },
    // 01010011000xxxxx0xxxxxxxxxxxxxxx  Ubfm  bfm
    RawEntry {
        pattern: "01010011000xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Ubfm,
        emit: emit_ubfm,
        decode: decode_bfm,
    },
    // 1101001101xxxxxxxxxxxxxxxxxxxxxx  Ubfm  bfm
    RawEntry {
        pattern: "1101001101xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ubfm,
        emit: emit_ubfm,
        decode: decode_bfm,
    },
    // x0011010110xxxxx000010xxxxxxxxxx  Udiv  alu_binary
    RawEntry {
        pattern: "x0011010110xxxxx000010xxxxxxxxxx",
        name: InstructionName::Udiv,
        emit: emit_udiv,
        decode: decode_alu_binary,
    },
    // 10011011101xxxxx0xxxxxxxxxxxxxxx  Umaddl  mul
    RawEntry {
        pattern: "10011011101xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Umaddl,
        emit: emit_umaddl,
        decode: decode_mul,
    },
    // 10011011101xxxxx1xxxxxxxxxxxxxxx  Umsubl  mul
    RawEntry {
        pattern: "10011011101xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Umsubl,
        emit: emit_umsubl,
        decode: decode_mul,
    },
    // 10011011110xxxxx0xxxxxxxxxxxxxxx  Umulh  mul
    RawEntry {
        pattern: "10011011110xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Umulh,
        emit: emit_umulh,
        decode: decode_mul,
    },
    // 11010101000000110010000001011111  Wfe  system
    RawEntry {
        pattern: "11010101000000110010000001011111",
        name: InstructionName::Wfe,
        emit: emit_nop,
        decode: decode_system,
    },
    // 11010101000000110010000001111111  Wfi  system
    RawEntry {
        pattern: "11010101000000110010000001111111",
        name: InstructionName::Wfi,
        emit: emit_nop,
        decode: decode_system,
    },
    // 11010101000000110010000000111111  Yield  system
    RawEntry {
        pattern: "11010101000000110010000000111111",
        name: InstructionName::Yield,
        emit: emit_nop,
        decode: decode_system,
    },
    // FP & SIMD
    // 0101111011100000101110xxxxxxxxxx  Abs_S  simd
    RawEntry {
        pattern: "0101111011100000101110xxxxxxxxxx",
        name: InstructionName::Abs_S,
        emit: emit_abs_s,
        decode: decode_simd,
    },
    // 0>001110<<100000101110xxxxxxxxxx  Abs_V  simd
    RawEntry {
        pattern: "0>001110<<100000101110xxxxxxxxxx",
        name: InstructionName::Abs_V,
        emit: emit_abs_v,
        decode: decode_simd,
    },
    // 01011110111xxxxx100001xxxxxxxxxx  Add_S  simd_reg
    RawEntry {
        pattern: "01011110111xxxxx100001xxxxxxxxxx",
        name: InstructionName::Add_S,
        emit: emit_add_s,
        decode: decode_simd_reg,
    },
    // 0>001110<<1xxxxx100001xxxxxxxxxx  Add_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx100001xxxxxxxxxx",
        name: InstructionName::Add_V,
        emit: emit_add_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx010000xxxxxxxxxx  Addhn_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx010000xxxxxxxxxx",
        name: InstructionName::Addhn_V,
        emit: emit_addhn_v,
        decode: decode_simd_reg,
    },
    // 0101111011110001101110xxxxxxxxxx  Addp_S  simd
    RawEntry {
        pattern: "0101111011110001101110xxxxxxxxxx",
        name: InstructionName::Addp_S,
        emit: emit_addp_s,
        decode: decode_simd,
    },
    // 0>001110<<1xxxxx101111xxxxxxxxxx  Addp_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx101111xxxxxxxxxx",
        name: InstructionName::Addp_V,
        emit: emit_addp_v,
        decode: decode_simd_reg,
    },
    // 000011100x110001101110xxxxxxxxxx  Addv_V  simd
    RawEntry {
        pattern: "000011100x110001101110xxxxxxxxxx",
        name: InstructionName::Addv_V,
        emit: emit_addv_v,
        decode: decode_simd,
    },
    // 01001110<<110001101110xxxxxxxxxx  Addv_V  simd
    RawEntry {
        pattern: "01001110<<110001101110xxxxxxxxxx",
        name: InstructionName::Addv_V,
        emit: emit_addv_v,
        decode: decode_simd,
    },
    // 0100111000101000010110xxxxxxxxxx  Aesd_V  simd
    RawEntry {
        pattern: "0100111000101000010110xxxxxxxxxx",
        name: InstructionName::Aesd_V,
        emit: emit_aesd_v,
        decode: decode_simd,
    },
    // 0100111000101000010010xxxxxxxxxx  Aese_V  simd
    RawEntry {
        pattern: "0100111000101000010010xxxxxxxxxx",
        name: InstructionName::Aese_V,
        emit: emit_aese_v,
        decode: decode_simd,
    },
    // 0100111000101000011110xxxxxxxxxx  Aesimc_V  simd
    RawEntry {
        pattern: "0100111000101000011110xxxxxxxxxx",
        name: InstructionName::Aesimc_V,
        emit: emit_aesimc_v,
        decode: decode_simd,
    },
    // 0100111000101000011010xxxxxxxxxx  Aesmc_V  simd
    RawEntry {
        pattern: "0100111000101000011010xxxxxxxxxx",
        name: InstructionName::Aesmc_V,
        emit: emit_aesmc_v,
        decode: decode_simd,
    },
    // 0x001110001xxxxx000111xxxxxxxxxx  And_V  simd_reg
    RawEntry {
        pattern: "0x001110001xxxxx000111xxxxxxxxxx",
        name: InstructionName::And_V,
        emit: emit_and_v,
        decode: decode_simd_reg,
    },
    // 0x001110011xxxxx000111xxxxxxxxxx  Bic_V  simd_reg
    RawEntry {
        pattern: "0x001110011xxxxx000111xxxxxxxxxx",
        name: InstructionName::Bic_V,
        emit: emit_bic_v,
        decode: decode_simd_reg,
    },
    // 0x10111100000xxx0xx101xxxxxxxxxx  Bic_Vi  simd_imm
    RawEntry {
        pattern: "0x10111100000xxx0xx101xxxxxxxxxx",
        name: InstructionName::Bic_Vi,
        emit: emit_bic_vi,
        decode: decode_simd_imm,
    },
    // 0x10111100000xxx10x101xxxxxxxxxx  Bic_Vi  simd_imm
    RawEntry {
        pattern: "0x10111100000xxx10x101xxxxxxxxxx",
        name: InstructionName::Bic_Vi,
        emit: emit_bic_vi,
        decode: decode_simd_imm,
    },
    // 0x101110111xxxxx000111xxxxxxxxxx  Bif_V  simd_reg
    RawEntry {
        pattern: "0x101110111xxxxx000111xxxxxxxxxx",
        name: InstructionName::Bif_V,
        emit: emit_bif_v,
        decode: decode_simd_reg,
    },
    // 0x101110101xxxxx000111xxxxxxxxxx  Bit_V  simd_reg
    RawEntry {
        pattern: "0x101110101xxxxx000111xxxxxxxxxx",
        name: InstructionName::Bit_V,
        emit: emit_bit_v,
        decode: decode_simd_reg,
    },
    // 0x101110011xxxxx000111xxxxxxxxxx  Bsl_V  simd_reg
    RawEntry {
        pattern: "0x101110011xxxxx000111xxxxxxxxxx",
        name: InstructionName::Bsl_V,
        emit: emit_bsl_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<100000010010xxxxxxxxxx  Cls_V  simd
    RawEntry {
        pattern: "0x001110<<100000010010xxxxxxxxxx",
        name: InstructionName::Cls_V,
        emit: emit_cls_v,
        decode: decode_simd,
    },
    // 0x101110<<100000010010xxxxxxxxxx  Clz_V  simd
    RawEntry {
        pattern: "0x101110<<100000010010xxxxxxxxxx",
        name: InstructionName::Clz_V,
        emit: emit_clz_v,
        decode: decode_simd,
    },
    // 01111110111xxxxx100011xxxxxxxxxx  Cmeq_S  simd_reg
    RawEntry {
        pattern: "01111110111xxxxx100011xxxxxxxxxx",
        name: InstructionName::Cmeq_S,
        emit: emit_cmeq_s,
        decode: decode_simd_reg,
    },
    // 0101111011100000100110xxxxxxxxxx  Cmeq_S  simd
    RawEntry {
        pattern: "0101111011100000100110xxxxxxxxxx",
        name: InstructionName::Cmeq_S,
        emit: emit_cmeq_s,
        decode: decode_simd,
    },
    // 0>101110<<1xxxxx100011xxxxxxxxxx  Cmeq_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx100011xxxxxxxxxx",
        name: InstructionName::Cmeq_V,
        emit: emit_cmeq_v,
        decode: decode_simd_reg,
    },
    // 0>001110<<100000100110xxxxxxxxxx  Cmeq_V  simd
    RawEntry {
        pattern: "0>001110<<100000100110xxxxxxxxxx",
        name: InstructionName::Cmeq_V,
        emit: emit_cmeq_v,
        decode: decode_simd,
    },
    // 01011110111xxxxx001111xxxxxxxxxx  Cmge_S  simd_reg
    RawEntry {
        pattern: "01011110111xxxxx001111xxxxxxxxxx",
        name: InstructionName::Cmge_S,
        emit: emit_cmge_s,
        decode: decode_simd_reg,
    },
    // 0111111011100000100010xxxxxxxxxx  Cmge_S  simd
    RawEntry {
        pattern: "0111111011100000100010xxxxxxxxxx",
        name: InstructionName::Cmge_S,
        emit: emit_cmge_s,
        decode: decode_simd,
    },
    // 0>001110<<1xxxxx001111xxxxxxxxxx  Cmge_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx001111xxxxxxxxxx",
        name: InstructionName::Cmge_V,
        emit: emit_cmge_v,
        decode: decode_simd_reg,
    },
    // 0>101110<<100000100010xxxxxxxxxx  Cmge_V  simd
    RawEntry {
        pattern: "0>101110<<100000100010xxxxxxxxxx",
        name: InstructionName::Cmge_V,
        emit: emit_cmge_v,
        decode: decode_simd,
    },
    // 01011110111xxxxx001101xxxxxxxxxx  Cmgt_S  simd_reg
    RawEntry {
        pattern: "01011110111xxxxx001101xxxxxxxxxx",
        name: InstructionName::Cmgt_S,
        emit: emit_cmgt_s,
        decode: decode_simd_reg,
    },
    // 0101111011100000100010xxxxxxxxxx  Cmgt_S  simd
    RawEntry {
        pattern: "0101111011100000100010xxxxxxxxxx",
        name: InstructionName::Cmgt_S,
        emit: emit_cmgt_s,
        decode: decode_simd,
    },
    // 0>001110<<1xxxxx001101xxxxxxxxxx  Cmgt_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx001101xxxxxxxxxx",
        name: InstructionName::Cmgt_V,
        emit: emit_cmgt_v,
        decode: decode_simd_reg,
    },
    // 0>001110<<100000100010xxxxxxxxxx  Cmgt_V  simd
    RawEntry {
        pattern: "0>001110<<100000100010xxxxxxxxxx",
        name: InstructionName::Cmgt_V,
        emit: emit_cmgt_v,
        decode: decode_simd,
    },
    // 01111110111xxxxx001101xxxxxxxxxx  Cmhi_S  simd_reg
    RawEntry {
        pattern: "01111110111xxxxx001101xxxxxxxxxx",
        name: InstructionName::Cmhi_S,
        emit: emit_cmhi_s,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx001101xxxxxxxxxx  Cmhi_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx001101xxxxxxxxxx",
        name: InstructionName::Cmhi_V,
        emit: emit_cmhi_v,
        decode: decode_simd_reg,
    },
    // 01111110111xxxxx001111xxxxxxxxxx  Cmhs_S  simd_reg
    RawEntry {
        pattern: "01111110111xxxxx001111xxxxxxxxxx",
        name: InstructionName::Cmhs_S,
        emit: emit_cmhs_s,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx001111xxxxxxxxxx  Cmhs_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx001111xxxxxxxxxx",
        name: InstructionName::Cmhs_V,
        emit: emit_cmhs_v,
        decode: decode_simd_reg,
    },
    // 0111111011100000100110xxxxxxxxxx  Cmle_S  simd
    RawEntry {
        pattern: "0111111011100000100110xxxxxxxxxx",
        name: InstructionName::Cmle_S,
        emit: emit_cmle_s,
        decode: decode_simd,
    },
    // 0>101110<<100000100110xxxxxxxxxx  Cmle_V  simd
    RawEntry {
        pattern: "0>101110<<100000100110xxxxxxxxxx",
        name: InstructionName::Cmle_V,
        emit: emit_cmle_v,
        decode: decode_simd,
    },
    // 0101111011100000101010xxxxxxxxxx  Cmlt_S  simd
    RawEntry {
        pattern: "0101111011100000101010xxxxxxxxxx",
        name: InstructionName::Cmlt_S,
        emit: emit_cmlt_s,
        decode: decode_simd,
    },
    // 0>001110<<100000101010xxxxxxxxxx  Cmlt_V  simd
    RawEntry {
        pattern: "0>001110<<100000101010xxxxxxxxxx",
        name: InstructionName::Cmlt_V,
        emit: emit_cmlt_v,
        decode: decode_simd,
    },
    // 01011110111xxxxx100011xxxxxxxxxx  Cmtst_S  simd_reg
    RawEntry {
        pattern: "01011110111xxxxx100011xxxxxxxxxx",
        name: InstructionName::Cmtst_S,
        emit: emit_cmtst_s,
        decode: decode_simd_reg,
    },
    // 0>001110<<1xxxxx100011xxxxxxxxxx  Cmtst_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx100011xxxxxxxxxx",
        name: InstructionName::Cmtst_V,
        emit: emit_cmtst_v,
        decode: decode_simd_reg,
    },
    // 0x00111000100000010110xxxxxxxxxx  Cnt_V  simd
    RawEntry {
        pattern: "0x00111000100000010110xxxxxxxxxx",
        name: InstructionName::Cnt_V,
        emit: emit_cnt_v,
        decode: decode_simd,
    },
    // 0>001110000x<>>>000011xxxxxxxxxx  Dup_Gp  simd_ins
    RawEntry {
        pattern: "0>001110000x<>>>000011xxxxxxxxxx",
        name: InstructionName::Dup_Gp,
        emit: emit_dup_gp,
        decode: decode_simd_ins,
    },
    // 01011110000xxxxx000001xxxxxxxxxx  Dup_S  simd_ins
    RawEntry {
        pattern: "01011110000xxxxx000001xxxxxxxxxx",
        name: InstructionName::Dup_S,
        emit: emit_dup_s,
        decode: decode_simd_ins,
    },
    // 0>001110000x<>>>000001xxxxxxxxxx  Dup_V  simd_ins
    RawEntry {
        pattern: "0>001110000x<>>>000001xxxxxxxxxx",
        name: InstructionName::Dup_V,
        emit: emit_dup_v,
        decode: decode_simd_ins,
    },
    // 0x101110001xxxxx000111xxxxxxxxxx  Eor_V  simd_reg
    RawEntry {
        pattern: "0x101110001xxxxx000111xxxxxxxxxx",
        name: InstructionName::Eor_V,
        emit: emit_eor_v,
        decode: decode_simd_reg,
    },
    // 0>101110000xxxxx0<xxx0xxxxxxxxxx  Ext_V  simd_ext
    RawEntry {
        pattern: "0>101110000xxxxx0<xxx0xxxxxxxxxx",
        name: InstructionName::Ext_V,
        emit: emit_ext_v,
        decode: decode_simd_ext,
    },
    // 011111101x1xxxxx110101xxxxxxxxxx  Fabd_S  simd_reg
    RawEntry {
        pattern: "011111101x1xxxxx110101xxxxxxxxxx",
        name: InstructionName::Fabd_S,
        emit: emit_fabd_s,
        decode: decode_simd_reg,
    },
    // 0>1011101<1xxxxx110101xxxxxxxxxx  Fabd_V  simd_reg
    RawEntry {
        pattern: "0>1011101<1xxxxx110101xxxxxxxxxx",
        name: InstructionName::Fabd_V,
        emit: emit_fabd_v,
        decode: decode_simd_reg,
    },
    // 000111100x100000110000xxxxxxxxxx  Fabs_S  simd
    RawEntry {
        pattern: "000111100x100000110000xxxxxxxxxx",
        name: InstructionName::Fabs_S,
        emit: emit_fabs_s,
        decode: decode_simd,
    },
    // 0>0011101<100000111110xxxxxxxxxx  Fabs_V  simd
    RawEntry {
        pattern: "0>0011101<100000111110xxxxxxxxxx",
        name: InstructionName::Fabs_V,
        emit: emit_fabs_v,
        decode: decode_simd,
    },
    // 011111100x1xxxxx111011xxxxxxxxxx  Facge_S  simd_reg
    RawEntry {
        pattern: "011111100x1xxxxx111011xxxxxxxxxx",
        name: InstructionName::Facge_S,
        emit: emit_facge_s,
        decode: decode_simd_reg,
    },
    // 0>1011100<1xxxxx111011xxxxxxxxxx  Facge_V  simd_reg
    RawEntry {
        pattern: "0>1011100<1xxxxx111011xxxxxxxxxx",
        name: InstructionName::Facge_V,
        emit: emit_facge_v,
        decode: decode_simd_reg,
    },
    // 011111101x1xxxxx111011xxxxxxxxxx  Facgt_S  simd_reg
    RawEntry {
        pattern: "011111101x1xxxxx111011xxxxxxxxxx",
        name: InstructionName::Facgt_S,
        emit: emit_facgt_s,
        decode: decode_simd_reg,
    },
    // 0>1011101<1xxxxx111011xxxxxxxxxx  Facgt_V  simd_reg
    RawEntry {
        pattern: "0>1011101<1xxxxx111011xxxxxxxxxx",
        name: InstructionName::Facgt_V,
        emit: emit_facgt_v,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxx001010xxxxxxxxxx  Fadd_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx001010xxxxxxxxxx",
        name: InstructionName::Fadd_S,
        emit: emit_fadd_s,
        decode: decode_simd_reg,
    },
    // 0>0011100<1xxxxx110101xxxxxxxxxx  Fadd_V  simd_reg
    RawEntry {
        pattern: "0>0011100<1xxxxx110101xxxxxxxxxx",
        name: InstructionName::Fadd_V,
        emit: emit_fadd_v,
        decode: decode_simd_reg,
    },
    // 011111100x110000110110xxxxxxxxxx  Faddp_S  simd
    RawEntry {
        pattern: "011111100x110000110110xxxxxxxxxx",
        name: InstructionName::Faddp_S,
        emit: emit_faddp_s,
        decode: decode_simd,
    },
    // 0>1011100<1xxxxx110101xxxxxxxxxx  Faddp_V  simd_reg
    RawEntry {
        pattern: "0>1011100<1xxxxx110101xxxxxxxxxx",
        name: InstructionName::Faddp_V,
        emit: emit_faddp_v,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxxxxxx01xxxxx0xxxx  Fccmp_S  simd_fcond
    RawEntry {
        pattern: "000111100x1xxxxxxxxx01xxxxx0xxxx",
        name: InstructionName::Fccmp_S,
        emit: emit_fccmp_s,
        decode: decode_simd_fcond,
    },
    // 000111100x1xxxxxxxxx01xxxxx1xxxx  Fccmpe_S  simd_fcond
    RawEntry {
        pattern: "000111100x1xxxxxxxxx01xxxxx1xxxx",
        name: InstructionName::Fccmpe_S,
        emit: emit_fccmpe_s,
        decode: decode_simd_fcond,
    },
    // 010111100x1xxxxx111001xxxxxxxxxx  Fcmeq_S  simd_reg
    RawEntry {
        pattern: "010111100x1xxxxx111001xxxxxxxxxx",
        name: InstructionName::Fcmeq_S,
        emit: emit_fcmeq_s,
        decode: decode_simd_reg,
    },
    // 010111101x100000110110xxxxxxxxxx  Fcmeq_S  simd
    RawEntry {
        pattern: "010111101x100000110110xxxxxxxxxx",
        name: InstructionName::Fcmeq_S,
        emit: emit_fcmeq_s,
        decode: decode_simd,
    },
    // 0>0011100<1xxxxx111001xxxxxxxxxx  Fcmeq_V  simd_reg
    RawEntry {
        pattern: "0>0011100<1xxxxx111001xxxxxxxxxx",
        name: InstructionName::Fcmeq_V,
        emit: emit_fcmeq_v,
        decode: decode_simd_reg,
    },
    // 0>0011101<100000110110xxxxxxxxxx  Fcmeq_V  simd
    RawEntry {
        pattern: "0>0011101<100000110110xxxxxxxxxx",
        name: InstructionName::Fcmeq_V,
        emit: emit_fcmeq_v,
        decode: decode_simd,
    },
    // 011111100x1xxxxx111001xxxxxxxxxx  Fcmge_S  simd_reg
    RawEntry {
        pattern: "011111100x1xxxxx111001xxxxxxxxxx",
        name: InstructionName::Fcmge_S,
        emit: emit_fcmge_s,
        decode: decode_simd_reg,
    },
    // 011111101x100000110010xxxxxxxxxx  Fcmge_S  simd
    RawEntry {
        pattern: "011111101x100000110010xxxxxxxxxx",
        name: InstructionName::Fcmge_S,
        emit: emit_fcmge_s,
        decode: decode_simd,
    },
    // 0>1011100<1xxxxx111001xxxxxxxxxx  Fcmge_V  simd_reg
    RawEntry {
        pattern: "0>1011100<1xxxxx111001xxxxxxxxxx",
        name: InstructionName::Fcmge_V,
        emit: emit_fcmge_v,
        decode: decode_simd_reg,
    },
    // 0>1011101<100000110010xxxxxxxxxx  Fcmge_V  simd
    RawEntry {
        pattern: "0>1011101<100000110010xxxxxxxxxx",
        name: InstructionName::Fcmge_V,
        emit: emit_fcmge_v,
        decode: decode_simd,
    },
    // 011111101x1xxxxx111001xxxxxxxxxx  Fcmgt_S  simd_reg
    RawEntry {
        pattern: "011111101x1xxxxx111001xxxxxxxxxx",
        name: InstructionName::Fcmgt_S,
        emit: emit_fcmgt_s,
        decode: decode_simd_reg,
    },
    // 010111101x100000110010xxxxxxxxxx  Fcmgt_S  simd
    RawEntry {
        pattern: "010111101x100000110010xxxxxxxxxx",
        name: InstructionName::Fcmgt_S,
        emit: emit_fcmgt_s,
        decode: decode_simd,
    },
    // 0>1011101<1xxxxx111001xxxxxxxxxx  Fcmgt_V  simd_reg
    RawEntry {
        pattern: "0>1011101<1xxxxx111001xxxxxxxxxx",
        name: InstructionName::Fcmgt_V,
        emit: emit_fcmgt_v,
        decode: decode_simd_reg,
    },
    // 0>0011101<100000110010xxxxxxxxxx  Fcmgt_V  simd
    RawEntry {
        pattern: "0>0011101<100000110010xxxxxxxxxx",
        name: InstructionName::Fcmgt_V,
        emit: emit_fcmgt_v,
        decode: decode_simd,
    },
    // 011111101x100000110110xxxxxxxxxx  Fcmle_S  simd
    RawEntry {
        pattern: "011111101x100000110110xxxxxxxxxx",
        name: InstructionName::Fcmle_S,
        emit: emit_fcmle_s,
        decode: decode_simd,
    },
    // 0>1011101<100000110110xxxxxxxxxx  Fcmle_V  simd
    RawEntry {
        pattern: "0>1011101<100000110110xxxxxxxxxx",
        name: InstructionName::Fcmle_V,
        emit: emit_fcmle_v,
        decode: decode_simd,
    },
    // 010111101x100000111010xxxxxxxxxx  Fcmlt_S  simd
    RawEntry {
        pattern: "010111101x100000111010xxxxxxxxxx",
        name: InstructionName::Fcmlt_S,
        emit: emit_fcmlt_s,
        decode: decode_simd,
    },
    // 0>0011101<100000111010xxxxxxxxxx  Fcmlt_V  simd
    RawEntry {
        pattern: "0>0011101<100000111010xxxxxxxxxx",
        name: InstructionName::Fcmlt_V,
        emit: emit_fcmlt_v,
        decode: decode_simd,
    },
    // 000111100x1xxxxx001000xxxxx0x000  Fcmp_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx001000xxxxx0x000",
        name: InstructionName::Fcmp_S,
        emit: emit_fcmp_s,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxx001000xxxxx1x000  Fcmpe_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx001000xxxxx1x000",
        name: InstructionName::Fcmpe_S,
        emit: emit_fcmpe_s,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxxxxxx11xxxxxxxxxx  Fcsel_S  simd_fcond
    RawEntry {
        pattern: "000111100x1xxxxxxxxx11xxxxxxxxxx",
        name: InstructionName::Fcsel_S,
        emit: emit_fcsel_s,
        decode: decode_simd_fcond,
    },
    // 00011110xx10001xx10000xxxxxxxxxx  Fcvt_S  simd
    RawEntry {
        pattern: "00011110xx10001xx10000xxxxxxxxxx",
        name: InstructionName::Fcvt_S,
        emit: emit_fcvt_s,
        decode: decode_simd,
    },
    // x00111100x100100000000xxxxxxxxxx  Fcvtas_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x100100000000xxxxxxxxxx",
        name: InstructionName::Fcvtas_Gp,
        emit: emit_fcvtas_gp,
        decode: decode_simd_cvt,
    },
    // 010111100x100001110010xxxxxxxxxx  Fcvtas_S  simd
    RawEntry {
        pattern: "010111100x100001110010xxxxxxxxxx",
        name: InstructionName::Fcvtas_S,
        emit: emit_fcvtas_s,
        decode: decode_simd,
    },
    // 0>0011100<100001110010xxxxxxxxxx  Fcvtas_V  simd
    RawEntry {
        pattern: "0>0011100<100001110010xxxxxxxxxx",
        name: InstructionName::Fcvtas_V,
        emit: emit_fcvtas_v,
        decode: decode_simd,
    },
    // x00111100x100101000000xxxxxxxxxx  Fcvtau_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x100101000000xxxxxxxxxx",
        name: InstructionName::Fcvtau_Gp,
        emit: emit_fcvtau_gp,
        decode: decode_simd_cvt,
    },
    // 011111100x100001110010xxxxxxxxxx  Fcvtau_S  simd
    RawEntry {
        pattern: "011111100x100001110010xxxxxxxxxx",
        name: InstructionName::Fcvtau_S,
        emit: emit_fcvtau_s,
        decode: decode_simd,
    },
    // 0>1011100<100001110010xxxxxxxxxx  Fcvtau_V  simd
    RawEntry {
        pattern: "0>1011100<100001110010xxxxxxxxxx",
        name: InstructionName::Fcvtau_V,
        emit: emit_fcvtau_v,
        decode: decode_simd,
    },
    // 0x0011100x100001011110xxxxxxxxxx  Fcvtl_V  simd
    RawEntry {
        pattern: "0x0011100x100001011110xxxxxxxxxx",
        name: InstructionName::Fcvtl_V,
        emit: emit_fcvtl_v,
        decode: decode_simd,
    },
    // x00111100x110000000000xxxxxxxxxx  Fcvtms_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x110000000000xxxxxxxxxx",
        name: InstructionName::Fcvtms_Gp,
        emit: emit_fcvtms_gp,
        decode: decode_simd_cvt,
    },
    // 0>0011100<100001101110xxxxxxxxxx  Fcvtms_V  simd
    RawEntry {
        pattern: "0>0011100<100001101110xxxxxxxxxx",
        name: InstructionName::Fcvtms_V,
        emit: emit_fcvtms_v,
        decode: decode_simd,
    },
    // x00111100x110001000000xxxxxxxxxx  Fcvtmu_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x110001000000xxxxxxxxxx",
        name: InstructionName::Fcvtmu_Gp,
        emit: emit_fcvtmu_gp,
        decode: decode_simd_cvt,
    },
    // 0x0011100x100001011010xxxxxxxxxx  Fcvtn_V  simd
    RawEntry {
        pattern: "0x0011100x100001011010xxxxxxxxxx",
        name: InstructionName::Fcvtn_V,
        emit: emit_fcvtn_v,
        decode: decode_simd,
    },
    // x00111100x100000000000xxxxxxxxxx  Fcvtns_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x100000000000xxxxxxxxxx",
        name: InstructionName::Fcvtns_Gp,
        emit: emit_fcvtns_gp,
        decode: decode_simd_cvt,
    },
    // 010111100x100001101010xxxxxxxxxx  Fcvtns_S  simd
    RawEntry {
        pattern: "010111100x100001101010xxxxxxxxxx",
        name: InstructionName::Fcvtns_S,
        emit: emit_fcvtns_s,
        decode: decode_simd,
    },
    // 0>0011100<100001101010xxxxxxxxxx  Fcvtns_V  simd
    RawEntry {
        pattern: "0>0011100<100001101010xxxxxxxxxx",
        name: InstructionName::Fcvtns_V,
        emit: emit_fcvtns_v,
        decode: decode_simd,
    },
    // 011111100x100001101010xxxxxxxxxx  Fcvtnu_S  simd
    RawEntry {
        pattern: "011111100x100001101010xxxxxxxxxx",
        name: InstructionName::Fcvtnu_S,
        emit: emit_fcvtnu_s,
        decode: decode_simd,
    },
    // 0>1011100<100001101010xxxxxxxxxx  Fcvtnu_V  simd
    RawEntry {
        pattern: "0>1011100<100001101010xxxxxxxxxx",
        name: InstructionName::Fcvtnu_V,
        emit: emit_fcvtnu_v,
        decode: decode_simd,
    },
    // x00111100x101000000000xxxxxxxxxx  Fcvtps_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x101000000000xxxxxxxxxx",
        name: InstructionName::Fcvtps_Gp,
        emit: emit_fcvtps_gp,
        decode: decode_simd_cvt,
    },
    // x00111100x101001000000xxxxxxxxxx  Fcvtpu_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x101001000000xxxxxxxxxx",
        name: InstructionName::Fcvtpu_Gp,
        emit: emit_fcvtpu_gp,
        decode: decode_simd_cvt,
    },
    // x00111100x111000000000xxxxxxxxxx  Fcvtzs_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x111000000000xxxxxxxxxx",
        name: InstructionName::Fcvtzs_Gp,
        emit: emit_fcvtzs_gp,
        decode: decode_simd_cvt,
    },
    // >00111100x011000>xxxxxxxxxxxxxxx  Fcvtzs_Gp_Fixed  simd_cvt
    RawEntry {
        pattern: ">00111100x011000>xxxxxxxxxxxxxxx",
        name: InstructionName::Fcvtzs_Gp_Fixed,
        emit: emit_fcvtzs_gp_fixed,
        decode: decode_simd_cvt,
    },
    // 010111101x100001101110xxxxxxxxxx  Fcvtzs_S  simd
    RawEntry {
        pattern: "010111101x100001101110xxxxxxxxxx",
        name: InstructionName::Fcvtzs_S,
        emit: emit_fcvtzs_s,
        decode: decode_simd,
    },
    // 0>0011101<100001101110xxxxxxxxxx  Fcvtzs_V  simd
    RawEntry {
        pattern: "0>0011101<100001101110xxxxxxxxxx",
        name: InstructionName::Fcvtzs_V,
        emit: emit_fcvtzs_v,
        decode: decode_simd,
    },
    // 0x001111001xxxxx111111xxxxxxxxxx  Fcvtzs_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0x001111001xxxxx111111xxxxxxxxxx",
        name: InstructionName::Fcvtzs_V_Fixed,
        emit: emit_fcvtzs_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // 0100111101xxxxxx111111xxxxxxxxxx  Fcvtzs_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0100111101xxxxxx111111xxxxxxxxxx",
        name: InstructionName::Fcvtzs_V_Fixed,
        emit: emit_fcvtzs_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // x00111100x111001000000xxxxxxxxxx  Fcvtzu_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x111001000000xxxxxxxxxx",
        name: InstructionName::Fcvtzu_Gp,
        emit: emit_fcvtzu_gp,
        decode: decode_simd_cvt,
    },
    // >00111100x011001>xxxxxxxxxxxxxxx  Fcvtzu_Gp_Fixed  simd_cvt
    RawEntry {
        pattern: ">00111100x011001>xxxxxxxxxxxxxxx",
        name: InstructionName::Fcvtzu_Gp_Fixed,
        emit: emit_fcvtzu_gp_fixed,
        decode: decode_simd_cvt,
    },
    // 011111101x100001101110xxxxxxxxxx  Fcvtzu_S  simd
    RawEntry {
        pattern: "011111101x100001101110xxxxxxxxxx",
        name: InstructionName::Fcvtzu_S,
        emit: emit_fcvtzu_s,
        decode: decode_simd,
    },
    // 0>1011101<100001101110xxxxxxxxxx  Fcvtzu_V  simd
    RawEntry {
        pattern: "0>1011101<100001101110xxxxxxxxxx",
        name: InstructionName::Fcvtzu_V,
        emit: emit_fcvtzu_v,
        decode: decode_simd,
    },
    // 0x101111001xxxxx111111xxxxxxxxxx  Fcvtzu_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0x101111001xxxxx111111xxxxxxxxxx",
        name: InstructionName::Fcvtzu_V_Fixed,
        emit: emit_fcvtzu_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx111111xxxxxxxxxx  Fcvtzu_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx111111xxxxxxxxxx",
        name: InstructionName::Fcvtzu_V_Fixed,
        emit: emit_fcvtzu_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // 000111100x1xxxxx000110xxxxxxxxxx  Fdiv_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx000110xxxxxxxxxx",
        name: InstructionName::Fdiv_S,
        emit: emit_fdiv_s,
        decode: decode_simd_reg,
    },
    // 0>1011100<1xxxxx111111xxxxxxxxxx  Fdiv_V  simd_reg
    RawEntry {
        pattern: "0>1011100<1xxxxx111111xxxxxxxxxx",
        name: InstructionName::Fdiv_V,
        emit: emit_fdiv_v,
        decode: decode_simd_reg,
    },
    // 000111110x0xxxxx0xxxxxxxxxxxxxxx  Fmadd_S  simd_reg
    RawEntry {
        pattern: "000111110x0xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Fmadd_S,
        emit: emit_fmadd_s,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxx010010xxxxxxxxxx  Fmax_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx010010xxxxxxxxxx",
        name: InstructionName::Fmax_S,
        emit: emit_fmax_s,
        decode: decode_simd_reg,
    },
    // 0>0011100<1xxxxx111101xxxxxxxxxx  Fmax_V  simd_reg
    RawEntry {
        pattern: "0>0011100<1xxxxx111101xxxxxxxxxx",
        name: InstructionName::Fmax_V,
        emit: emit_fmax_v,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxx011010xxxxxxxxxx  Fmaxnm_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx011010xxxxxxxxxx",
        name: InstructionName::Fmaxnm_S,
        emit: emit_fmaxnm_s,
        decode: decode_simd_reg,
    },
    // 0>0011100<1xxxxx110001xxxxxxxxxx  Fmaxnm_V  simd_reg
    RawEntry {
        pattern: "0>0011100<1xxxxx110001xxxxxxxxxx",
        name: InstructionName::Fmaxnm_V,
        emit: emit_fmaxnm_v,
        decode: decode_simd_reg,
    },
    // 011111100x110000110010xxxxxxxxxx  Fmaxnmp_S  simd
    RawEntry {
        pattern: "011111100x110000110010xxxxxxxxxx",
        name: InstructionName::Fmaxnmp_S,
        emit: emit_fmaxnmp_s,
        decode: decode_simd,
    },
    // 0>1011100<1xxxxx110001xxxxxxxxxx  Fmaxnmp_V  simd_reg
    RawEntry {
        pattern: "0>1011100<1xxxxx110001xxxxxxxxxx",
        name: InstructionName::Fmaxnmp_V,
        emit: emit_fmaxnmp_v,
        decode: decode_simd_reg,
    },
    // 0110111000110000110010xxxxxxxxxx  Fmaxnmv_V  simd
    RawEntry {
        pattern: "0110111000110000110010xxxxxxxxxx",
        name: InstructionName::Fmaxnmv_V,
        emit: emit_fmaxnmv_v,
        decode: decode_simd,
    },
    // 011111100x110000111110xxxxxxxxxx  Fmaxp_S  simd
    RawEntry {
        pattern: "011111100x110000111110xxxxxxxxxx",
        name: InstructionName::Fmaxp_S,
        emit: emit_fmaxp_s,
        decode: decode_simd,
    },
    // 0>1011100<1xxxxx111101xxxxxxxxxx  Fmaxp_V  simd_reg
    RawEntry {
        pattern: "0>1011100<1xxxxx111101xxxxxxxxxx",
        name: InstructionName::Fmaxp_V,
        emit: emit_fmaxp_v,
        decode: decode_simd_reg,
    },
    // 0110111000110000111110xxxxxxxxxx  Fmaxv_V  simd
    RawEntry {
        pattern: "0110111000110000111110xxxxxxxxxx",
        name: InstructionName::Fmaxv_V,
        emit: emit_fmaxv_v,
        decode: decode_simd,
    },
    // 000111100x1xxxxx010110xxxxxxxxxx  Fmin_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx010110xxxxxxxxxx",
        name: InstructionName::Fmin_S,
        emit: emit_fmin_s,
        decode: decode_simd_reg,
    },
    // 0>0011101<1xxxxx111101xxxxxxxxxx  Fmin_V  simd_reg
    RawEntry {
        pattern: "0>0011101<1xxxxx111101xxxxxxxxxx",
        name: InstructionName::Fmin_V,
        emit: emit_fmin_v,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxx011110xxxxxxxxxx  Fminnm_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx011110xxxxxxxxxx",
        name: InstructionName::Fminnm_S,
        emit: emit_fminnm_s,
        decode: decode_simd_reg,
    },
    // 0>0011101<1xxxxx110001xxxxxxxxxx  Fminnm_V  simd_reg
    RawEntry {
        pattern: "0>0011101<1xxxxx110001xxxxxxxxxx",
        name: InstructionName::Fminnm_V,
        emit: emit_fminnm_v,
        decode: decode_simd_reg,
    },
    // 011111101x110000110010xxxxxxxxxx  Fminnmp_S  simd
    RawEntry {
        pattern: "011111101x110000110010xxxxxxxxxx",
        name: InstructionName::Fminnmp_S,
        emit: emit_fminnmp_s,
        decode: decode_simd,
    },
    // 0>1011101<1xxxxx110001xxxxxxxxxx  Fminnmp_V  simd_reg
    RawEntry {
        pattern: "0>1011101<1xxxxx110001xxxxxxxxxx",
        name: InstructionName::Fminnmp_V,
        emit: emit_fminnmp_v,
        decode: decode_simd_reg,
    },
    // 0110111010110000110010xxxxxxxxxx  Fminnmv_V  simd
    RawEntry {
        pattern: "0110111010110000110010xxxxxxxxxx",
        name: InstructionName::Fminnmv_V,
        emit: emit_fminnmv_v,
        decode: decode_simd,
    },
    // 011111101x110000111110xxxxxxxxxx  Fminp_S  simd
    RawEntry {
        pattern: "011111101x110000111110xxxxxxxxxx",
        name: InstructionName::Fminp_S,
        emit: emit_fminp_s,
        decode: decode_simd,
    },
    // 0>1011101<1xxxxx111101xxxxxxxxxx  Fminp_V  simd_reg
    RawEntry {
        pattern: "0>1011101<1xxxxx111101xxxxxxxxxx",
        name: InstructionName::Fminp_V,
        emit: emit_fminp_v,
        decode: decode_simd_reg,
    },
    // 0110111010110000111110xxxxxxxxxx  Fminv_V  simd
    RawEntry {
        pattern: "0110111010110000111110xxxxxxxxxx",
        name: InstructionName::Fminv_V,
        emit: emit_fminv_v,
        decode: decode_simd,
    },
    // 010111111xxxxxxx0001x0xxxxxxxxxx  Fmla_Se  simd_reg_elemf
    RawEntry {
        pattern: "010111111xxxxxxx0001x0xxxxxxxxxx",
        name: InstructionName::Fmla_Se,
        emit: emit_fmla_se,
        decode: decode_simd_reg_elem_f,
    },
    // 0>0011100<1xxxxx110011xxxxxxxxxx  Fmla_V  simd_reg
    RawEntry {
        pattern: "0>0011100<1xxxxx110011xxxxxxxxxx",
        name: InstructionName::Fmla_V,
        emit: emit_fmla_v,
        decode: decode_simd_reg,
    },
    // 0>0011111<xxxxxx0001x0xxxxxxxxxx  Fmla_Ve  simd_reg_elemf
    RawEntry {
        pattern: "0>0011111<xxxxxx0001x0xxxxxxxxxx",
        name: InstructionName::Fmla_Ve,
        emit: emit_fmla_ve,
        decode: decode_simd_reg_elem_f,
    },
    // 010111111xxxxxxx0101x0xxxxxxxxxx  Fmls_Se  simd_reg_elemf
    RawEntry {
        pattern: "010111111xxxxxxx0101x0xxxxxxxxxx",
        name: InstructionName::Fmls_Se,
        emit: emit_fmls_se,
        decode: decode_simd_reg_elem_f,
    },
    // 0>0011101<1xxxxx110011xxxxxxxxxx  Fmls_V  simd_reg
    RawEntry {
        pattern: "0>0011101<1xxxxx110011xxxxxxxxxx",
        name: InstructionName::Fmls_V,
        emit: emit_fmls_v,
        decode: decode_simd_reg,
    },
    // 0>0011111<xxxxxx0101x0xxxxxxxxxx  Fmls_Ve  simd_reg_elemf
    RawEntry {
        pattern: "0>0011111<xxxxxx0101x0xxxxxxxxxx",
        name: InstructionName::Fmls_Ve,
        emit: emit_fmls_ve,
        decode: decode_simd_reg_elem_f,
    },
    // 000111100x100000010000xxxxxxxxxx  Fmov_S  simd
    RawEntry {
        pattern: "000111100x100000010000xxxxxxxxxx",
        name: InstructionName::Fmov_S,
        emit: emit_fmov_s,
        decode: decode_simd,
    },
    // 000111100x1xxxxxxxx10000000xxxxx  Fmov_Si  simd_fmov
    RawEntry {
        pattern: "000111100x1xxxxxxxx10000000xxxxx",
        name: InstructionName::Fmov_Si,
        emit: emit_fmov_si,
        decode: decode_simd_fmov,
    },
    // 0x00111100000xxx111101xxxxxxxxxx  Fmov_Vi  simd_imm
    RawEntry {
        pattern: "0x00111100000xxx111101xxxxxxxxxx",
        name: InstructionName::Fmov_Vi,
        emit: emit_fmov_vi,
        decode: decode_simd_imm,
    },
    // 0110111100000xxx111101xxxxxxxxxx  Fmov_Vi  simd_imm
    RawEntry {
        pattern: "0110111100000xxx111101xxxxxxxxxx",
        name: InstructionName::Fmov_Vi,
        emit: emit_fmov_vi,
        decode: decode_simd_imm,
    },
    // 0001111000100110000000xxxxxxxxxx  Fmov_Ftoi  simd
    RawEntry {
        pattern: "0001111000100110000000xxxxxxxxxx",
        name: InstructionName::Fmov_Ftoi,
        emit: emit_fmov_ftoi,
        decode: decode_simd,
    },
    // 1001111001100110000000xxxxxxxxxx  Fmov_Ftoi  simd
    RawEntry {
        pattern: "1001111001100110000000xxxxxxxxxx",
        name: InstructionName::Fmov_Ftoi,
        emit: emit_fmov_ftoi,
        decode: decode_simd,
    },
    // 0001111000100111000000xxxxxxxxxx  Fmov_Itof  simd
    RawEntry {
        pattern: "0001111000100111000000xxxxxxxxxx",
        name: InstructionName::Fmov_Itof,
        emit: emit_fmov_itof,
        decode: decode_simd,
    },
    // 1001111001100111000000xxxxxxxxxx  Fmov_Itof  simd
    RawEntry {
        pattern: "1001111001100111000000xxxxxxxxxx",
        name: InstructionName::Fmov_Itof,
        emit: emit_fmov_itof,
        decode: decode_simd,
    },
    // 1001111010101110000000xxxxxxxxxx  Fmov_Ftoi1  simd
    RawEntry {
        pattern: "1001111010101110000000xxxxxxxxxx",
        name: InstructionName::Fmov_Ftoi1,
        emit: emit_fmov_ftoi1,
        decode: decode_simd,
    },
    // 1001111010101111000000xxxxxxxxxx  Fmov_Itof1  simd
    RawEntry {
        pattern: "1001111010101111000000xxxxxxxxxx",
        name: InstructionName::Fmov_Itof1,
        emit: emit_fmov_itof1,
        decode: decode_simd,
    },
    // 000111110x0xxxxx1xxxxxxxxxxxxxxx  Fmsub_S  simd_reg
    RawEntry {
        pattern: "000111110x0xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Fmsub_S,
        emit: emit_fmsub_s,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxx000010xxxxxxxxxx  Fmul_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx000010xxxxxxxxxx",
        name: InstructionName::Fmul_S,
        emit: emit_fmul_s,
        decode: decode_simd_reg,
    },
    // 010111111xxxxxxx1001x0xxxxxxxxxx  Fmul_Se  simd_reg_elemf
    RawEntry {
        pattern: "010111111xxxxxxx1001x0xxxxxxxxxx",
        name: InstructionName::Fmul_Se,
        emit: emit_fmul_se,
        decode: decode_simd_reg_elem_f,
    },
    // 0>1011100<1xxxxx110111xxxxxxxxxx  Fmul_V  simd_reg
    RawEntry {
        pattern: "0>1011100<1xxxxx110111xxxxxxxxxx",
        name: InstructionName::Fmul_V,
        emit: emit_fmul_v,
        decode: decode_simd_reg,
    },
    // 0>0011111<xxxxxx1001x0xxxxxxxxxx  Fmul_Ve  simd_reg_elemf
    RawEntry {
        pattern: "0>0011111<xxxxxx1001x0xxxxxxxxxx",
        name: InstructionName::Fmul_Ve,
        emit: emit_fmul_ve,
        decode: decode_simd_reg_elem_f,
    },
    // 010111100x1xxxxx110111xxxxxxxxxx  Fmulx_S  simd_reg
    RawEntry {
        pattern: "010111100x1xxxxx110111xxxxxxxxxx",
        name: InstructionName::Fmulx_S,
        emit: emit_fmulx_s,
        decode: decode_simd_reg,
    },
    // 011111111xxxxxxx1001x0xxxxxxxxxx  Fmulx_Se  simd_reg_elemf
    RawEntry {
        pattern: "011111111xxxxxxx1001x0xxxxxxxxxx",
        name: InstructionName::Fmulx_Se,
        emit: emit_fmulx_se,
        decode: decode_simd_reg_elem_f,
    },
    // 0>0011100<1xxxxx110111xxxxxxxxxx  Fmulx_V  simd_reg
    RawEntry {
        pattern: "0>0011100<1xxxxx110111xxxxxxxxxx",
        name: InstructionName::Fmulx_V,
        emit: emit_fmulx_v,
        decode: decode_simd_reg,
    },
    // 0>1011111<xxxxxx1001x0xxxxxxxxxx  Fmulx_Ve  simd_reg_elemf
    RawEntry {
        pattern: "0>1011111<xxxxxx1001x0xxxxxxxxxx",
        name: InstructionName::Fmulx_Ve,
        emit: emit_fmulx_ve,
        decode: decode_simd_reg_elem_f,
    },
    // 000111100x100001010000xxxxxxxxxx  Fneg_S  simd
    RawEntry {
        pattern: "000111100x100001010000xxxxxxxxxx",
        name: InstructionName::Fneg_S,
        emit: emit_fneg_s,
        decode: decode_simd,
    },
    // 0>1011101<100000111110xxxxxxxxxx  Fneg_V  simd
    RawEntry {
        pattern: "0>1011101<100000111110xxxxxxxxxx",
        name: InstructionName::Fneg_V,
        emit: emit_fneg_v,
        decode: decode_simd,
    },
    // 000111110x1xxxxx0xxxxxxxxxxxxxxx  Fnmadd_S  simd_reg
    RawEntry {
        pattern: "000111110x1xxxxx0xxxxxxxxxxxxxxx",
        name: InstructionName::Fnmadd_S,
        emit: emit_fnmadd_s,
        decode: decode_simd_reg,
    },
    // 000111110x1xxxxx1xxxxxxxxxxxxxxx  Fnmsub_S  simd_reg
    RawEntry {
        pattern: "000111110x1xxxxx1xxxxxxxxxxxxxxx",
        name: InstructionName::Fnmsub_S,
        emit: emit_fnmsub_s,
        decode: decode_simd_reg,
    },
    // 000111100x1xxxxx100010xxxxxxxxxx  Fnmul_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx100010xxxxxxxxxx",
        name: InstructionName::Fnmul_S,
        emit: emit_fnmul_s,
        decode: decode_simd_reg,
    },
    // 010111101x100001110110xxxxxxxxxx  Frecpe_S  simd
    RawEntry {
        pattern: "010111101x100001110110xxxxxxxxxx",
        name: InstructionName::Frecpe_S,
        emit: emit_frecpe_s,
        decode: decode_simd,
    },
    // 0>0011101<100001110110xxxxxxxxxx  Frecpe_V  simd
    RawEntry {
        pattern: "0>0011101<100001110110xxxxxxxxxx",
        name: InstructionName::Frecpe_V,
        emit: emit_frecpe_v,
        decode: decode_simd,
    },
    // 010111100x1xxxxx111111xxxxxxxxxx  Frecps_S  simd_reg
    RawEntry {
        pattern: "010111100x1xxxxx111111xxxxxxxxxx",
        name: InstructionName::Frecps_S,
        emit: emit_frecps_s,
        decode: decode_simd_reg,
    },
    // 0>0011100<1xxxxx111111xxxxxxxxxx  Frecps_V  simd_reg
    RawEntry {
        pattern: "0>0011100<1xxxxx111111xxxxxxxxxx",
        name: InstructionName::Frecps_V,
        emit: emit_frecps_v,
        decode: decode_simd_reg,
    },
    // 010111101x100001111110xxxxxxxxxx  Frecpx_S  simd
    RawEntry {
        pattern: "010111101x100001111110xxxxxxxxxx",
        name: InstructionName::Frecpx_S,
        emit: emit_frecpx_s,
        decode: decode_simd,
    },
    // 000111100x100110010000xxxxxxxxxx  Frinta_S  simd
    RawEntry {
        pattern: "000111100x100110010000xxxxxxxxxx",
        name: InstructionName::Frinta_S,
        emit: emit_frinta_s,
        decode: decode_simd,
    },
    // 0>1011100<100001100010xxxxxxxxxx  Frinta_V  simd
    RawEntry {
        pattern: "0>1011100<100001100010xxxxxxxxxx",
        name: InstructionName::Frinta_V,
        emit: emit_frinta_v,
        decode: decode_simd,
    },
    // 000111100x100111110000xxxxxxxxxx  Frinti_S  simd
    RawEntry {
        pattern: "000111100x100111110000xxxxxxxxxx",
        name: InstructionName::Frinti_S,
        emit: emit_frinti_s,
        decode: decode_simd,
    },
    // 0>1011101<100001100110xxxxxxxxxx  Frinti_V  simd
    RawEntry {
        pattern: "0>1011101<100001100110xxxxxxxxxx",
        name: InstructionName::Frinti_V,
        emit: emit_frinti_v,
        decode: decode_simd,
    },
    // 000111100x100101010000xxxxxxxxxx  Frintm_S  simd
    RawEntry {
        pattern: "000111100x100101010000xxxxxxxxxx",
        name: InstructionName::Frintm_S,
        emit: emit_frintm_s,
        decode: decode_simd,
    },
    // 0>0011100<100001100110xxxxxxxxxx  Frintm_V  simd
    RawEntry {
        pattern: "0>0011100<100001100110xxxxxxxxxx",
        name: InstructionName::Frintm_V,
        emit: emit_frintm_v,
        decode: decode_simd,
    },
    // 000111100x100100010000xxxxxxxxxx  Frintn_S  simd
    RawEntry {
        pattern: "000111100x100100010000xxxxxxxxxx",
        name: InstructionName::Frintn_S,
        emit: emit_frintn_s,
        decode: decode_simd,
    },
    // 0>0011100<100001100010xxxxxxxxxx  Frintn_V  simd
    RawEntry {
        pattern: "0>0011100<100001100010xxxxxxxxxx",
        name: InstructionName::Frintn_V,
        emit: emit_frintn_v,
        decode: decode_simd,
    },
    // 000111100x100100110000xxxxxxxxxx  Frintp_S  simd
    RawEntry {
        pattern: "000111100x100100110000xxxxxxxxxx",
        name: InstructionName::Frintp_S,
        emit: emit_frintp_s,
        decode: decode_simd,
    },
    // 0>0011101<100001100010xxxxxxxxxx  Frintp_V  simd
    RawEntry {
        pattern: "0>0011101<100001100010xxxxxxxxxx",
        name: InstructionName::Frintp_V,
        emit: emit_frintp_v,
        decode: decode_simd,
    },
    // 000111100x100111010000xxxxxxxxxx  Frintx_S  simd
    RawEntry {
        pattern: "000111100x100111010000xxxxxxxxxx",
        name: InstructionName::Frintx_S,
        emit: emit_frintx_s,
        decode: decode_simd,
    },
    // 0>1011100<100001100110xxxxxxxxxx  Frintx_V  simd
    RawEntry {
        pattern: "0>1011100<100001100110xxxxxxxxxx",
        name: InstructionName::Frintx_V,
        emit: emit_frintx_v,
        decode: decode_simd,
    },
    // 000111100x100101110000xxxxxxxxxx  Frintz_S  simd
    RawEntry {
        pattern: "000111100x100101110000xxxxxxxxxx",
        name: InstructionName::Frintz_S,
        emit: emit_frintz_s,
        decode: decode_simd,
    },
    // 0>0011101<100001100110xxxxxxxxxx  Frintz_V  simd
    RawEntry {
        pattern: "0>0011101<100001100110xxxxxxxxxx",
        name: InstructionName::Frintz_V,
        emit: emit_frintz_v,
        decode: decode_simd,
    },
    // 011111101x100001110110xxxxxxxxxx  Frsqrte_S  simd
    RawEntry {
        pattern: "011111101x100001110110xxxxxxxxxx",
        name: InstructionName::Frsqrte_S,
        emit: emit_frsqrte_s,
        decode: decode_simd,
    },
    // 0>1011101<100001110110xxxxxxxxxx  Frsqrte_V  simd
    RawEntry {
        pattern: "0>1011101<100001110110xxxxxxxxxx",
        name: InstructionName::Frsqrte_V,
        emit: emit_frsqrte_v,
        decode: decode_simd,
    },
    // 010111101x1xxxxx111111xxxxxxxxxx  Frsqrts_S  simd_reg
    RawEntry {
        pattern: "010111101x1xxxxx111111xxxxxxxxxx",
        name: InstructionName::Frsqrts_S,
        emit: emit_frsqrts_s,
        decode: decode_simd_reg,
    },
    // 0>0011101<1xxxxx111111xxxxxxxxxx  Frsqrts_V  simd_reg
    RawEntry {
        pattern: "0>0011101<1xxxxx111111xxxxxxxxxx",
        name: InstructionName::Frsqrts_V,
        emit: emit_frsqrts_v,
        decode: decode_simd_reg,
    },
    // 000111100x100001110000xxxxxxxxxx  Fsqrt_S  simd
    RawEntry {
        pattern: "000111100x100001110000xxxxxxxxxx",
        name: InstructionName::Fsqrt_S,
        emit: emit_fsqrt_s,
        decode: decode_simd,
    },
    // 0>1011101<100001111110xxxxxxxxxx  Fsqrt_V  simd
    RawEntry {
        pattern: "0>1011101<100001111110xxxxxxxxxx",
        name: InstructionName::Fsqrt_V,
        emit: emit_fsqrt_v,
        decode: decode_simd,
    },
    // 000111100x1xxxxx001110xxxxxxxxxx  Fsub_S  simd_reg
    RawEntry {
        pattern: "000111100x1xxxxx001110xxxxxxxxxx",
        name: InstructionName::Fsub_S,
        emit: emit_fsub_s,
        decode: decode_simd_reg,
    },
    // 0>0011101<1xxxxx110101xxxxxxxxxx  Fsub_V  simd_reg
    RawEntry {
        pattern: "0>0011101<1xxxxx110101xxxxxxxxxx",
        name: InstructionName::Fsub_V,
        emit: emit_fsub_v,
        decode: decode_simd_reg,
    },
    // 01001110000xxxxx000111xxxxxxxxxx  Ins_Gp  simd_ins
    RawEntry {
        pattern: "01001110000xxxxx000111xxxxxxxxxx",
        name: InstructionName::Ins_Gp,
        emit: emit_ins_gp,
        decode: decode_simd_ins,
    },
    // 01101110000xxxxx0xxxx1xxxxxxxxxx  Ins_V  simd_ins
    RawEntry {
        pattern: "01101110000xxxxx0xxxx1xxxxxxxxxx",
        name: InstructionName::Ins_V,
        emit: emit_ins_v,
        decode: decode_simd_ins,
    },
    // 0x00110001000000xxxxxxxxxxxxxxxx  Ld__Vms  simd_mem_ms
    RawEntry {
        pattern: "0x00110001000000xxxxxxxxxxxxxxxx",
        name: InstructionName::Ld__Vms,
        emit: emit_ld__vms,
        decode: decode_simd_mem_ms,
    },
    // 0x001100110xxxxxxxxxxxxxxxxxxxxx  Ld__Vms  simd_mem_ms
    RawEntry {
        pattern: "0x001100110xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ld__Vms,
        emit: emit_ld__vms,
        decode: decode_simd_mem_ms,
    },
    // 0x00110101x00000xxxxxxxxxxxxxxxx  Ld__Vss  simd_mem_ss
    RawEntry {
        pattern: "0x00110101x00000xxxxxxxxxxxxxxxx",
        name: InstructionName::Ld__Vss,
        emit: emit_ld__vss,
        decode: decode_simd_mem_ss,
    },
    // 0x00110111xxxxxxxxxxxxxxxxxxxxxx  Ld__Vss  simd_mem_ss
    RawEntry {
        pattern: "0x00110111xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ld__Vss,
        emit: emit_ld__vss,
        decode: decode_simd_mem_ss,
    },
    // <<10110xx1xxxxxxxxxxxxxxxxxxxxxx  Ldp  simd_mem_pair
    RawEntry {
        pattern: "<<10110xx1xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldp,
        emit: emit_ldp,
        decode: decode_simd_mem_pair,
    },
    // xx111100x10xxxxxxxxx00xxxxxxxxxx  Ldr  simd_mem_imm
    RawEntry {
        pattern: "xx111100x10xxxxxxxxx00xxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_simd_mem_imm,
    },
    // xx111100x10xxxxxxxxx01xxxxxxxxxx  Ldr  simd_mem_imm
    RawEntry {
        pattern: "xx111100x10xxxxxxxxx01xxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_simd_mem_imm,
    },
    // xx111100x10xxxxxxxxx11xxxxxxxxxx  Ldr  simd_mem_imm
    RawEntry {
        pattern: "xx111100x10xxxxxxxxx11xxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_simd_mem_imm,
    },
    // xx111101x1xxxxxxxxxxxxxxxxxxxxxx  Ldr  simd_mem_imm
    RawEntry {
        pattern: "xx111101x1xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_simd_mem_imm,
    },
    // xx111100x11xxxxxx1xx10xxxxxxxxxx  Ldr  simd_mem_reg
    RawEntry {
        pattern: "xx111100x11xxxxxx1xx10xxxxxxxxxx",
        name: InstructionName::Ldr,
        emit: emit_ldr,
        decode: decode_simd_mem_reg,
    },
    // xx011100xxxxxxxxxxxxxxxxxxxxxxxx  Ldr_Literal  simd_mem_lit
    RawEntry {
        pattern: "xx011100xxxxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Ldr_Literal,
        emit: emit_ldr_literal,
        decode: decode_simd_mem_lit,
    },
    // 0x001110<<1xxxxx100101xxxxxxxxxx  Mla_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx100101xxxxxxxxxx",
        name: InstructionName::Mla_V,
        emit: emit_mla_v,
        decode: decode_simd_reg,
    },
    // 0x101111xxxxxxxx0000x0xxxxxxxxxx  Mla_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x101111xxxxxxxx0000x0xxxxxxxxxx",
        name: InstructionName::Mla_Ve,
        emit: emit_mla_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x101110<<1xxxxx100101xxxxxxxxxx  Mls_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx100101xxxxxxxxxx",
        name: InstructionName::Mls_V,
        emit: emit_mls_v,
        decode: decode_simd_reg,
    },
    // 0x101111xxxxxxxx0100x0xxxxxxxxxx  Mls_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x101111xxxxxxxx0100x0xxxxxxxxxx",
        name: InstructionName::Mls_Ve,
        emit: emit_mls_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x00111100000xxx0xx001xxxxxxxxxx  Movi_V  simd_imm
    RawEntry {
        pattern: "0x00111100000xxx0xx001xxxxxxxxxx",
        name: InstructionName::Movi_V,
        emit: emit_movi_v,
        decode: decode_simd_imm,
    },
    // 0x00111100000xxx10x001xxxxxxxxxx  Movi_V  simd_imm
    RawEntry {
        pattern: "0x00111100000xxx10x001xxxxxxxxxx",
        name: InstructionName::Movi_V,
        emit: emit_movi_v,
        decode: decode_simd_imm,
    },
    // 0x00111100000xxx110x01xxxxxxxxxx  Movi_V  simd_imm
    RawEntry {
        pattern: "0x00111100000xxx110x01xxxxxxxxxx",
        name: InstructionName::Movi_V,
        emit: emit_movi_v,
        decode: decode_simd_imm,
    },
    // 0xx0111100000xxx111001xxxxxxxxxx  Movi_V  simd_imm
    RawEntry {
        pattern: "0xx0111100000xxx111001xxxxxxxxxx",
        name: InstructionName::Movi_V,
        emit: emit_movi_v,
        decode: decode_simd_imm,
    },
    // 0x001110<<1xxxxx100111xxxxxxxxxx  Mul_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx100111xxxxxxxxxx",
        name: InstructionName::Mul_V,
        emit: emit_mul_v,
        decode: decode_simd_reg,
    },
    // 0x001111xxxxxxxx1000x0xxxxxxxxxx  Mul_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x001111xxxxxxxx1000x0xxxxxxxxxx",
        name: InstructionName::Mul_Ve,
        emit: emit_mul_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x10111100000xxx0xx001xxxxxxxxxx  Mvni_V  simd_imm
    RawEntry {
        pattern: "0x10111100000xxx0xx001xxxxxxxxxx",
        name: InstructionName::Mvni_V,
        emit: emit_mvni_v,
        decode: decode_simd_imm,
    },
    // 0x10111100000xxx10x001xxxxxxxxxx  Mvni_V  simd_imm
    RawEntry {
        pattern: "0x10111100000xxx10x001xxxxxxxxxx",
        name: InstructionName::Mvni_V,
        emit: emit_mvni_v,
        decode: decode_simd_imm,
    },
    // 0x10111100000xxx110x01xxxxxxxxxx  Mvni_V  simd_imm
    RawEntry {
        pattern: "0x10111100000xxx110x01xxxxxxxxxx",
        name: InstructionName::Mvni_V,
        emit: emit_mvni_v,
        decode: decode_simd_imm,
    },
    // 0111111011100000101110xxxxxxxxxx  Neg_S  simd
    RawEntry {
        pattern: "0111111011100000101110xxxxxxxxxx",
        name: InstructionName::Neg_S,
        emit: emit_neg_s,
        decode: decode_simd,
    },
    // 0>101110<<100000101110xxxxxxxxxx  Neg_V  simd
    RawEntry {
        pattern: "0>101110<<100000101110xxxxxxxxxx",
        name: InstructionName::Neg_V,
        emit: emit_neg_v,
        decode: decode_simd,
    },
    // 0x10111000100000010110xxxxxxxxxx  Not_V  simd
    RawEntry {
        pattern: "0x10111000100000010110xxxxxxxxxx",
        name: InstructionName::Not_V,
        emit: emit_not_v,
        decode: decode_simd,
    },
    // 0x001110111xxxxx000111xxxxxxxxxx  Orn_V  simd_reg
    RawEntry {
        pattern: "0x001110111xxxxx000111xxxxxxxxxx",
        name: InstructionName::Orn_V,
        emit: emit_orn_v,
        decode: decode_simd_reg,
    },
    // 0x001110101xxxxx000111xxxxxxxxxx  Orr_V  simd_reg
    RawEntry {
        pattern: "0x001110101xxxxx000111xxxxxxxxxx",
        name: InstructionName::Orr_V,
        emit: emit_orr_v,
        decode: decode_simd_reg,
    },
    // 0x00111100000xxx0xx101xxxxxxxxxx  Orr_Vi  simd_imm
    RawEntry {
        pattern: "0x00111100000xxx0xx101xxxxxxxxxx",
        name: InstructionName::Orr_Vi,
        emit: emit_orr_vi,
        decode: decode_simd_imm,
    },
    // 0x00111100000xxx10x101xxxxxxxxxx  Orr_Vi  simd_imm
    RawEntry {
        pattern: "0x00111100000xxx10x101xxxxxxxxxx",
        name: InstructionName::Orr_Vi,
        emit: emit_orr_vi,
        decode: decode_simd_imm,
    },
    // 0x001110001xxxxx111000xxxxxxxxxx  Pmull_V  simd_reg
    RawEntry {
        pattern: "0x001110001xxxxx111000xxxxxxxxxx",
        name: InstructionName::Pmull_V,
        emit: emit_pmull_v,
        decode: decode_simd_reg,
    },
    // 0x001110111xxxxx111000xxxxxxxxxx  Pmull_V  simd_reg
    RawEntry {
        pattern: "0x001110111xxxxx111000xxxxxxxxxx",
        name: InstructionName::Pmull_V,
        emit: emit_pmull_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx010000xxxxxxxxxx  Raddhn_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx010000xxxxxxxxxx",
        name: InstructionName::Raddhn_V,
        emit: emit_raddhn_v,
        decode: decode_simd_reg,
    },
    // 0x10111001100000010110xxxxxxxxxx  Rbit_V  simd
    RawEntry {
        pattern: "0x10111001100000010110xxxxxxxxxx",
        name: InstructionName::Rbit_V,
        emit: emit_rbit_v,
        decode: decode_simd,
    },
    // 0x00111000100000000110xxxxxxxxxx  Rev16_V  simd
    RawEntry {
        pattern: "0x00111000100000000110xxxxxxxxxx",
        name: InstructionName::Rev16_V,
        emit: emit_rev16_v,
        decode: decode_simd,
    },
    // 0x1011100x100000000010xxxxxxxxxx  Rev32_V  simd
    RawEntry {
        pattern: "0x1011100x100000000010xxxxxxxxxx",
        name: InstructionName::Rev32_V,
        emit: emit_rev32_v,
        decode: decode_simd,
    },
    // 0x001110<<100000000010xxxxxxxxxx  Rev64_V  simd
    RawEntry {
        pattern: "0x001110<<100000000010xxxxxxxxxx",
        name: InstructionName::Rev64_V,
        emit: emit_rev64_v,
        decode: decode_simd,
    },
    // 0x00111100>>>xxx100011xxxxxxxxxx  Rshrn_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx100011xxxxxxxxxx",
        name: InstructionName::Rshrn_V,
        emit: emit_rshrn_v,
        decode: decode_simd_sh_imm,
    },
    // 0x101110<<1xxxxx011000xxxxxxxxxx  Rsubhn_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx011000xxxxxxxxxx",
        name: InstructionName::Rsubhn_V,
        emit: emit_rsubhn_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx011111xxxxxxxxxx  Saba_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx011111xxxxxxxxxx",
        name: InstructionName::Saba_V,
        emit: emit_saba_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx010100xxxxxxxxxx  Sabal_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx010100xxxxxxxxxx",
        name: InstructionName::Sabal_V,
        emit: emit_sabal_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx011101xxxxxxxxxx  Sabd_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx011101xxxxxxxxxx",
        name: InstructionName::Sabd_V,
        emit: emit_sabd_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx011100xxxxxxxxxx  Sabdl_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx011100xxxxxxxxxx",
        name: InstructionName::Sabdl_V,
        emit: emit_sabdl_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<100000011010xxxxxxxxxx  Sadalp_V  simd
    RawEntry {
        pattern: "0x001110<<100000011010xxxxxxxxxx",
        name: InstructionName::Sadalp_V,
        emit: emit_sadalp_v,
        decode: decode_simd,
    },
    // 0x001110<<1xxxxx000000xxxxxxxxxx  Saddl_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx000000xxxxxxxxxx",
        name: InstructionName::Saddl_V,
        emit: emit_saddl_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<100000001010xxxxxxxxxx  Saddlp_V  simd
    RawEntry {
        pattern: "0x001110<<100000001010xxxxxxxxxx",
        name: InstructionName::Saddlp_V,
        emit: emit_saddlp_v,
        decode: decode_simd,
    },
    // 000011100x110000001110xxxxxxxxxx  Saddlv_V  simd
    RawEntry {
        pattern: "000011100x110000001110xxxxxxxxxx",
        name: InstructionName::Saddlv_V,
        emit: emit_saddlv_v,
        decode: decode_simd,
    },
    // 01001110<<110000001110xxxxxxxxxx  Saddlv_V  simd
    RawEntry {
        pattern: "01001110<<110000001110xxxxxxxxxx",
        name: InstructionName::Saddlv_V,
        emit: emit_saddlv_v,
        decode: decode_simd,
    },
    // 0x001110<<1xxxxx000100xxxxxxxxxx  Saddw_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx000100xxxxxxxxxx",
        name: InstructionName::Saddw_V,
        emit: emit_saddw_v,
        decode: decode_simd_reg,
    },
    // x00111100x100010000000xxxxxxxxxx  Scvtf_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x100010000000xxxxxxxxxx",
        name: InstructionName::Scvtf_Gp,
        emit: emit_scvtf_gp,
        decode: decode_simd_cvt,
    },
    // >00111100x000010>xxxxxxxxxxxxxxx  Scvtf_Gp_Fixed  simd_cvt
    RawEntry {
        pattern: ">00111100x000010>xxxxxxxxxxxxxxx",
        name: InstructionName::Scvtf_Gp_Fixed,
        emit: emit_scvtf_gp_fixed,
        decode: decode_simd_cvt,
    },
    // 010111100x100001110110xxxxxxxxxx  Scvtf_S  simd
    RawEntry {
        pattern: "010111100x100001110110xxxxxxxxxx",
        name: InstructionName::Scvtf_S,
        emit: emit_scvtf_s,
        decode: decode_simd,
    },
    // 010111110>>xxxxx111001xxxxxxxxxx  Scvtf_S_Fixed  simd_sh_imm
    RawEntry {
        pattern: "010111110>>xxxxx111001xxxxxxxxxx",
        name: InstructionName::Scvtf_S_Fixed,
        emit: emit_scvtf_s_fixed,
        decode: decode_simd_sh_imm,
    },
    // 0>0011100<100001110110xxxxxxxxxx  Scvtf_V  simd
    RawEntry {
        pattern: "0>0011100<100001110110xxxxxxxxxx",
        name: InstructionName::Scvtf_V,
        emit: emit_scvtf_v,
        decode: decode_simd,
    },
    // 0x001111001xxxxx111001xxxxxxxxxx  Scvtf_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0x001111001xxxxx111001xxxxxxxxxx",
        name: InstructionName::Scvtf_V_Fixed,
        emit: emit_scvtf_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // 0100111101xxxxxx111001xxxxxxxxxx  Scvtf_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0100111101xxxxxx111001xxxxxxxxxx",
        name: InstructionName::Scvtf_V_Fixed,
        emit: emit_scvtf_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // 01011110000xxxxx000000xxxxxxxxxx  Sha1c_V  simd_reg
    RawEntry {
        pattern: "01011110000xxxxx000000xxxxxxxxxx",
        name: InstructionName::Sha1c_V,
        emit: emit_sha1c_v,
        decode: decode_simd_reg,
    },
    // 0101111000101000000010xxxxxxxxxx  Sha1h_V  simd
    RawEntry {
        pattern: "0101111000101000000010xxxxxxxxxx",
        name: InstructionName::Sha1h_V,
        emit: emit_sha1h_v,
        decode: decode_simd,
    },
    // 01011110000xxxxx001000xxxxxxxxxx  Sha1m_V  simd_reg
    RawEntry {
        pattern: "01011110000xxxxx001000xxxxxxxxxx",
        name: InstructionName::Sha1m_V,
        emit: emit_sha1m_v,
        decode: decode_simd_reg,
    },
    // 01011110000xxxxx000100xxxxxxxxxx  Sha1p_V  simd_reg
    RawEntry {
        pattern: "01011110000xxxxx000100xxxxxxxxxx",
        name: InstructionName::Sha1p_V,
        emit: emit_sha1p_v,
        decode: decode_simd_reg,
    },
    // 01011110000xxxxx001100xxxxxxxxxx  Sha1su0_V  simd_reg
    RawEntry {
        pattern: "01011110000xxxxx001100xxxxxxxxxx",
        name: InstructionName::Sha1su0_V,
        emit: emit_sha1su0_v,
        decode: decode_simd_reg,
    },
    // 0101111000101000000110xxxxxxxxxx  Sha1su1_V  simd
    RawEntry {
        pattern: "0101111000101000000110xxxxxxxxxx",
        name: InstructionName::Sha1su1_V,
        emit: emit_sha1su1_v,
        decode: decode_simd,
    },
    // 01011110000xxxxx010000xxxxxxxxxx  Sha256h_V  simd_reg
    RawEntry {
        pattern: "01011110000xxxxx010000xxxxxxxxxx",
        name: InstructionName::Sha256h_V,
        emit: emit_sha256h_v,
        decode: decode_simd_reg,
    },
    // 01011110000xxxxx010100xxxxxxxxxx  Sha256h2_V  simd_reg
    RawEntry {
        pattern: "01011110000xxxxx010100xxxxxxxxxx",
        name: InstructionName::Sha256h2_V,
        emit: emit_sha256h2_v,
        decode: decode_simd_reg,
    },
    // 0101111000101000001010xxxxxxxxxx  Sha256su0_V  simd
    RawEntry {
        pattern: "0101111000101000001010xxxxxxxxxx",
        name: InstructionName::Sha256su0_V,
        emit: emit_sha256su0_v,
        decode: decode_simd,
    },
    // 01011110000xxxxx011000xxxxxxxxxx  Sha256su1_V  simd_reg
    RawEntry {
        pattern: "01011110000xxxxx011000xxxxxxxxxx",
        name: InstructionName::Sha256su1_V,
        emit: emit_sha256su1_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx000001xxxxxxxxxx  Shadd_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx000001xxxxxxxxxx",
        name: InstructionName::Shadd_V,
        emit: emit_shadd_v,
        decode: decode_simd_reg,
    },
    // 0101111101xxxxxx010101xxxxxxxxxx  Shl_S  simd_sh_imm
    RawEntry {
        pattern: "0101111101xxxxxx010101xxxxxxxxxx",
        name: InstructionName::Shl_S,
        emit: emit_shl_s,
        decode: decode_simd_sh_imm,
    },
    // 0x00111100>>>xxx010101xxxxxxxxxx  Shl_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx010101xxxxxxxxxx",
        name: InstructionName::Shl_V,
        emit: emit_shl_v,
        decode: decode_simd_sh_imm,
    },
    // 0100111101xxxxxx010101xxxxxxxxxx  Shl_V  simd_sh_imm
    RawEntry {
        pattern: "0100111101xxxxxx010101xxxxxxxxxx",
        name: InstructionName::Shl_V,
        emit: emit_shl_v,
        decode: decode_simd_sh_imm,
    },
    // 0x101110<<100001001110xxxxxxxxxx  Shll_V  simd
    RawEntry {
        pattern: "0x101110<<100001001110xxxxxxxxxx",
        name: InstructionName::Shll_V,
        emit: emit_shll_v,
        decode: decode_simd,
    },
    // 0x00111100>>>xxx100001xxxxxxxxxx  Shrn_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx100001xxxxxxxxxx",
        name: InstructionName::Shrn_V,
        emit: emit_shrn_v,
        decode: decode_simd_sh_imm,
    },
    // 0x001110<<1xxxxx001001xxxxxxxxxx  Shsub_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx001001xxxxxxxxxx",
        name: InstructionName::Shsub_V,
        emit: emit_shsub_v,
        decode: decode_simd_reg,
    },
    // 0111111101xxxxxx010101xxxxxxxxxx  Sli_S  simd_sh_imm
    RawEntry {
        pattern: "0111111101xxxxxx010101xxxxxxxxxx",
        name: InstructionName::Sli_S,
        emit: emit_sli_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx010101xxxxxxxxxx  Sli_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx010101xxxxxxxxxx",
        name: InstructionName::Sli_V,
        emit: emit_sli_v,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx010101xxxxxxxxxx  Sli_V  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx010101xxxxxxxxxx",
        name: InstructionName::Sli_V,
        emit: emit_sli_v,
        decode: decode_simd_sh_imm,
    },
    // 0x001110<<1xxxxx011001xxxxxxxxxx  Smax_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx011001xxxxxxxxxx",
        name: InstructionName::Smax_V,
        emit: emit_smax_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx101001xxxxxxxxxx  Smaxp_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx101001xxxxxxxxxx",
        name: InstructionName::Smaxp_V,
        emit: emit_smaxp_v,
        decode: decode_simd_reg,
    },
    // 000011100x110000101010xxxxxxxxxx  Smaxv_V  simd
    RawEntry {
        pattern: "000011100x110000101010xxxxxxxxxx",
        name: InstructionName::Smaxv_V,
        emit: emit_smaxv_v,
        decode: decode_simd,
    },
    // 01001110<<110000101010xxxxxxxxxx  Smaxv_V  simd
    RawEntry {
        pattern: "01001110<<110000101010xxxxxxxxxx",
        name: InstructionName::Smaxv_V,
        emit: emit_smaxv_v,
        decode: decode_simd,
    },
    // 0x001110<<1xxxxx011011xxxxxxxxxx  Smin_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx011011xxxxxxxxxx",
        name: InstructionName::Smin_V,
        emit: emit_smin_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx101011xxxxxxxxxx  Sminp_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx101011xxxxxxxxxx",
        name: InstructionName::Sminp_V,
        emit: emit_sminp_v,
        decode: decode_simd_reg,
    },
    // 000011100x110001101010xxxxxxxxxx  Sminv_V  simd
    RawEntry {
        pattern: "000011100x110001101010xxxxxxxxxx",
        name: InstructionName::Sminv_V,
        emit: emit_sminv_v,
        decode: decode_simd,
    },
    // 01001110<<110001101010xxxxxxxxxx  Sminv_V  simd
    RawEntry {
        pattern: "01001110<<110001101010xxxxxxxxxx",
        name: InstructionName::Sminv_V,
        emit: emit_sminv_v,
        decode: decode_simd,
    },
    // 0x001110<<1xxxxx100000xxxxxxxxxx  Smlal_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx100000xxxxxxxxxx",
        name: InstructionName::Smlal_V,
        emit: emit_smlal_v,
        decode: decode_simd_reg,
    },
    // 0x001111xxxxxxxx0010x0xxxxxxxxxx  Smlal_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x001111xxxxxxxx0010x0xxxxxxxxxx",
        name: InstructionName::Smlal_Ve,
        emit: emit_smlal_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x001110<<1xxxxx101000xxxxxxxxxx  Smlsl_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx101000xxxxxxxxxx",
        name: InstructionName::Smlsl_V,
        emit: emit_smlsl_v,
        decode: decode_simd_reg,
    },
    // 0x001111xxxxxxxx0110x0xxxxxxxxxx  Smlsl_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x001111xxxxxxxx0110x0xxxxxxxxxx",
        name: InstructionName::Smlsl_Ve,
        emit: emit_smlsl_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x001110000xxxxx001011xxxxxxxxxx  Smov_S  simd_ins
    RawEntry {
        pattern: "0x001110000xxxxx001011xxxxxxxxxx",
        name: InstructionName::Smov_S,
        emit: emit_smov_s,
        decode: decode_simd_ins,
    },
    // 0x001110<<1xxxxx110000xxxxxxxxxx  Smull_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx110000xxxxxxxxxx",
        name: InstructionName::Smull_V,
        emit: emit_smull_v,
        decode: decode_simd_reg,
    },
    // 0x001111xxxxxxxx1010x0xxxxxxxxxx  Smull_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x001111xxxxxxxx1010x0xxxxxxxxxx",
        name: InstructionName::Smull_Ve,
        emit: emit_smull_ve,
        decode: decode_simd_reg_elem,
    },
    // 01011110xx100000011110xxxxxxxxxx  Sqabs_S  simd
    RawEntry {
        pattern: "01011110xx100000011110xxxxxxxxxx",
        name: InstructionName::Sqabs_S,
        emit: emit_sqabs_s,
        decode: decode_simd,
    },
    // 0>001110<<100000011110xxxxxxxxxx  Sqabs_V  simd
    RawEntry {
        pattern: "0>001110<<100000011110xxxxxxxxxx",
        name: InstructionName::Sqabs_V,
        emit: emit_sqabs_v,
        decode: decode_simd,
    },
    // 01011110xx1xxxxx000011xxxxxxxxxx  Sqadd_S  simd_reg
    RawEntry {
        pattern: "01011110xx1xxxxx000011xxxxxxxxxx",
        name: InstructionName::Sqadd_S,
        emit: emit_sqadd_s,
        decode: decode_simd_reg,
    },
    // 0>001110<<1xxxxx000011xxxxxxxxxx  Sqadd_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx000011xxxxxxxxxx",
        name: InstructionName::Sqadd_V,
        emit: emit_sqadd_v,
        decode: decode_simd_reg,
    },
    // 01011110011xxxxx101101xxxxxxxxxx  Sqdmulh_S  simd_reg
    RawEntry {
        pattern: "01011110011xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqdmulh_S,
        emit: emit_sqdmulh_s,
        decode: decode_simd_reg,
    },
    // 01011110101xxxxx101101xxxxxxxxxx  Sqdmulh_S  simd_reg
    RawEntry {
        pattern: "01011110101xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqdmulh_S,
        emit: emit_sqdmulh_s,
        decode: decode_simd_reg,
    },
    // 0x001110011xxxxx101101xxxxxxxxxx  Sqdmulh_V  simd_reg
    RawEntry {
        pattern: "0x001110011xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqdmulh_V,
        emit: emit_sqdmulh_v,
        decode: decode_simd_reg,
    },
    // 0x001110101xxxxx101101xxxxxxxxxx  Sqdmulh_V  simd_reg
    RawEntry {
        pattern: "0x001110101xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqdmulh_V,
        emit: emit_sqdmulh_v,
        decode: decode_simd_reg,
    },
    // 0x00111101xxxxxx1100x0xxxxxxxxxx  Sqdmulh_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x00111101xxxxxx1100x0xxxxxxxxxx",
        name: InstructionName::Sqdmulh_Ve,
        emit: emit_sqdmulh_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x00111110xxxxxx1100x0xxxxxxxxxx  Sqdmulh_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x00111110xxxxxx1100x0xxxxxxxxxx",
        name: InstructionName::Sqdmulh_Ve,
        emit: emit_sqdmulh_ve,
        decode: decode_simd_reg_elem,
    },
    // 01111110xx100000011110xxxxxxxxxx  Sqneg_S  simd
    RawEntry {
        pattern: "01111110xx100000011110xxxxxxxxxx",
        name: InstructionName::Sqneg_S,
        emit: emit_sqneg_s,
        decode: decode_simd,
    },
    // 0>101110<<100000011110xxxxxxxxxx  Sqneg_V  simd
    RawEntry {
        pattern: "0>101110<<100000011110xxxxxxxxxx",
        name: InstructionName::Sqneg_V,
        emit: emit_sqneg_v,
        decode: decode_simd,
    },
    // 01111110011xxxxx101101xxxxxxxxxx  Sqrdmulh_S  simd_reg
    RawEntry {
        pattern: "01111110011xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqrdmulh_S,
        emit: emit_sqrdmulh_s,
        decode: decode_simd_reg,
    },
    // 01111110101xxxxx101101xxxxxxxxxx  Sqrdmulh_S  simd_reg
    RawEntry {
        pattern: "01111110101xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqrdmulh_S,
        emit: emit_sqrdmulh_s,
        decode: decode_simd_reg,
    },
    // 0x101110011xxxxx101101xxxxxxxxxx  Sqrdmulh_V  simd_reg
    RawEntry {
        pattern: "0x101110011xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqrdmulh_V,
        emit: emit_sqrdmulh_v,
        decode: decode_simd_reg,
    },
    // 0x101110101xxxxx101101xxxxxxxxxx  Sqrdmulh_V  simd_reg
    RawEntry {
        pattern: "0x101110101xxxxx101101xxxxxxxxxx",
        name: InstructionName::Sqrdmulh_V,
        emit: emit_sqrdmulh_v,
        decode: decode_simd_reg,
    },
    // 0x00111101xxxxxx1101x0xxxxxxxxxx  Sqrdmulh_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x00111101xxxxxx1101x0xxxxxxxxxx",
        name: InstructionName::Sqrdmulh_Ve,
        emit: emit_sqrdmulh_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x00111110xxxxxx1101x0xxxxxxxxxx  Sqrdmulh_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x00111110xxxxxx1101x0xxxxxxxxxx",
        name: InstructionName::Sqrdmulh_Ve,
        emit: emit_sqrdmulh_ve,
        decode: decode_simd_reg_elem,
    },
    // 0>001110<<1xxxxx010111xxxxxxxxxx  Sqrshl_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx010111xxxxxxxxxx",
        name: InstructionName::Sqrshl_V,
        emit: emit_sqrshl_v,
        decode: decode_simd_reg,
    },
    // 0101111100>>>xxx100111xxxxxxxxxx  Sqrshrn_S  simd_sh_imm
    RawEntry {
        pattern: "0101111100>>>xxx100111xxxxxxxxxx",
        name: InstructionName::Sqrshrn_S,
        emit: emit_sqrshrn_s,
        decode: decode_simd_sh_imm,
    },
    // 0x00111100>>>xxx100111xxxxxxxxxx  Sqrshrn_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx100111xxxxxxxxxx",
        name: InstructionName::Sqrshrn_V,
        emit: emit_sqrshrn_v,
        decode: decode_simd_sh_imm,
    },
    // 0111111100>>>xxx100011xxxxxxxxxx  Sqrshrun_S  simd_sh_imm
    RawEntry {
        pattern: "0111111100>>>xxx100011xxxxxxxxxx",
        name: InstructionName::Sqrshrun_S,
        emit: emit_sqrshrun_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx100011xxxxxxxxxx  Sqrshrun_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx100011xxxxxxxxxx",
        name: InstructionName::Sqrshrun_V,
        emit: emit_sqrshrun_v,
        decode: decode_simd_sh_imm,
    },
    // 010111110>>>>xxx011101xxxxxxxxxx  Sqshl_Si  simd_sh_imm
    RawEntry {
        pattern: "010111110>>>>xxx011101xxxxxxxxxx",
        name: InstructionName::Sqshl_Si,
        emit: emit_sqshl_si,
        decode: decode_simd_sh_imm,
    },
    // 0>001110<<1xxxxx010011xxxxxxxxxx  Sqshl_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx010011xxxxxxxxxx",
        name: InstructionName::Sqshl_V,
        emit: emit_sqshl_v,
        decode: decode_simd_reg,
    },
    // 0000111100>>>xxx011101xxxxxxxxxx  Sqshl_Vi  simd_sh_imm
    RawEntry {
        pattern: "0000111100>>>xxx011101xxxxxxxxxx",
        name: InstructionName::Sqshl_Vi,
        emit: emit_sqshl_vi,
        decode: decode_simd_sh_imm,
    },
    // 010011110>>>>xxx011101xxxxxxxxxx  Sqshl_Vi  simd_sh_imm
    RawEntry {
        pattern: "010011110>>>>xxx011101xxxxxxxxxx",
        name: InstructionName::Sqshl_Vi,
        emit: emit_sqshl_vi,
        decode: decode_simd_sh_imm,
    },
    // 0101111100>>>xxx100101xxxxxxxxxx  Sqshrn_S  simd_sh_imm
    RawEntry {
        pattern: "0101111100>>>xxx100101xxxxxxxxxx",
        name: InstructionName::Sqshrn_S,
        emit: emit_sqshrn_s,
        decode: decode_simd_sh_imm,
    },
    // 0x00111100>>>xxx100101xxxxxxxxxx  Sqshrn_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx100101xxxxxxxxxx",
        name: InstructionName::Sqshrn_V,
        emit: emit_sqshrn_v,
        decode: decode_simd_sh_imm,
    },
    // 0111111100>>>xxx100001xxxxxxxxxx  Sqshrun_S  simd_sh_imm
    RawEntry {
        pattern: "0111111100>>>xxx100001xxxxxxxxxx",
        name: InstructionName::Sqshrun_S,
        emit: emit_sqshrun_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx100001xxxxxxxxxx  Sqshrun_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx100001xxxxxxxxxx",
        name: InstructionName::Sqshrun_V,
        emit: emit_sqshrun_v,
        decode: decode_simd_sh_imm,
    },
    // 01011110xx1xxxxx001011xxxxxxxxxx  Sqsub_S  simd_reg
    RawEntry {
        pattern: "01011110xx1xxxxx001011xxxxxxxxxx",
        name: InstructionName::Sqsub_S,
        emit: emit_sqsub_s,
        decode: decode_simd_reg,
    },
    // 0>001110<<1xxxxx001011xxxxxxxxxx  Sqsub_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx001011xxxxxxxxxx",
        name: InstructionName::Sqsub_V,
        emit: emit_sqsub_v,
        decode: decode_simd_reg,
    },
    // 01011110<<100001010010xxxxxxxxxx  Sqxtn_S  simd
    RawEntry {
        pattern: "01011110<<100001010010xxxxxxxxxx",
        name: InstructionName::Sqxtn_S,
        emit: emit_sqxtn_s,
        decode: decode_simd,
    },
    // 0x001110<<100001010010xxxxxxxxxx  Sqxtn_V  simd
    RawEntry {
        pattern: "0x001110<<100001010010xxxxxxxxxx",
        name: InstructionName::Sqxtn_V,
        emit: emit_sqxtn_v,
        decode: decode_simd,
    },
    // 01111110<<100001001010xxxxxxxxxx  Sqxtun_S  simd
    RawEntry {
        pattern: "01111110<<100001001010xxxxxxxxxx",
        name: InstructionName::Sqxtun_S,
        emit: emit_sqxtun_s,
        decode: decode_simd,
    },
    // 0x101110<<100001001010xxxxxxxxxx  Sqxtun_V  simd
    RawEntry {
        pattern: "0x101110<<100001001010xxxxxxxxxx",
        name: InstructionName::Sqxtun_V,
        emit: emit_sqxtun_v,
        decode: decode_simd,
    },
    // 0x001110<<1xxxxx000101xxxxxxxxxx  Srhadd_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx000101xxxxxxxxxx",
        name: InstructionName::Srhadd_V,
        emit: emit_srhadd_v,
        decode: decode_simd_reg,
    },
    // 0111111101xxxxxx010001xxxxxxxxxx  Sri_S  simd_sh_imm
    RawEntry {
        pattern: "0111111101xxxxxx010001xxxxxxxxxx",
        name: InstructionName::Sri_S,
        emit: emit_sri_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx010001xxxxxxxxxx  Sri_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx010001xxxxxxxxxx",
        name: InstructionName::Sri_V,
        emit: emit_sri_v,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx010001xxxxxxxxxx  Sri_V  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx010001xxxxxxxxxx",
        name: InstructionName::Sri_V,
        emit: emit_sri_v,
        decode: decode_simd_sh_imm,
    },
    // 0>001110<<1xxxxx010101xxxxxxxxxx  Srshl_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx010101xxxxxxxxxx",
        name: InstructionName::Srshl_V,
        emit: emit_srshl_v,
        decode: decode_simd_reg,
    },
    // 0101111101xxxxxx001001xxxxxxxxxx  Srshr_S  simd_sh_imm
    RawEntry {
        pattern: "0101111101xxxxxx001001xxxxxxxxxx",
        name: InstructionName::Srshr_S,
        emit: emit_srshr_s,
        decode: decode_simd_sh_imm,
    },
    // 0x00111100>>>xxx001001xxxxxxxxxx  Srshr_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx001001xxxxxxxxxx",
        name: InstructionName::Srshr_V,
        emit: emit_srshr_v,
        decode: decode_simd_sh_imm,
    },
    // 0100111101xxxxxx001001xxxxxxxxxx  Srshr_V  simd_sh_imm
    RawEntry {
        pattern: "0100111101xxxxxx001001xxxxxxxxxx",
        name: InstructionName::Srshr_V,
        emit: emit_srshr_v,
        decode: decode_simd_sh_imm,
    },
    // 0101111101xxxxxx001101xxxxxxxxxx  Srsra_S  simd_sh_imm
    RawEntry {
        pattern: "0101111101xxxxxx001101xxxxxxxxxx",
        name: InstructionName::Srsra_S,
        emit: emit_srsra_s,
        decode: decode_simd_sh_imm,
    },
    // 0x00111100>>>xxx001101xxxxxxxxxx  Srsra_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx001101xxxxxxxxxx",
        name: InstructionName::Srsra_V,
        emit: emit_srsra_v,
        decode: decode_simd_sh_imm,
    },
    // 0100111101xxxxxx001101xxxxxxxxxx  Srsra_V  simd_sh_imm
    RawEntry {
        pattern: "0100111101xxxxxx001101xxxxxxxxxx",
        name: InstructionName::Srsra_V,
        emit: emit_srsra_v,
        decode: decode_simd_sh_imm,
    },
    // 01011110111xxxxx010001xxxxxxxxxx  Sshl_S  simd_reg
    RawEntry {
        pattern: "01011110111xxxxx010001xxxxxxxxxx",
        name: InstructionName::Sshl_S,
        emit: emit_sshl_s,
        decode: decode_simd_reg,
    },
    // 0>001110<<1xxxxx010001xxxxxxxxxx  Sshl_V  simd_reg
    RawEntry {
        pattern: "0>001110<<1xxxxx010001xxxxxxxxxx",
        name: InstructionName::Sshl_V,
        emit: emit_sshl_v,
        decode: decode_simd_reg,
    },
    // 0x00111100>>>xxx101001xxxxxxxxxx  Sshll_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx101001xxxxxxxxxx",
        name: InstructionName::Sshll_V,
        emit: emit_sshll_v,
        decode: decode_simd_sh_imm,
    },
    // 0101111101xxxxxx000001xxxxxxxxxx  Sshr_S  simd_sh_imm
    RawEntry {
        pattern: "0101111101xxxxxx000001xxxxxxxxxx",
        name: InstructionName::Sshr_S,
        emit: emit_sshr_s,
        decode: decode_simd_sh_imm,
    },
    // 0x00111100>>>xxx000001xxxxxxxxxx  Sshr_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx000001xxxxxxxxxx",
        name: InstructionName::Sshr_V,
        emit: emit_sshr_v,
        decode: decode_simd_sh_imm,
    },
    // 0100111101xxxxxx000001xxxxxxxxxx  Sshr_V  simd_sh_imm
    RawEntry {
        pattern: "0100111101xxxxxx000001xxxxxxxxxx",
        name: InstructionName::Sshr_V,
        emit: emit_sshr_v,
        decode: decode_simd_sh_imm,
    },
    // 0101111101xxxxxx000101xxxxxxxxxx  Ssra_S  simd_sh_imm
    RawEntry {
        pattern: "0101111101xxxxxx000101xxxxxxxxxx",
        name: InstructionName::Ssra_S,
        emit: emit_ssra_s,
        decode: decode_simd_sh_imm,
    },
    // 0x00111100>>>xxx000101xxxxxxxxxx  Ssra_V  simd_sh_imm
    RawEntry {
        pattern: "0x00111100>>>xxx000101xxxxxxxxxx",
        name: InstructionName::Ssra_V,
        emit: emit_ssra_v,
        decode: decode_simd_sh_imm,
    },
    // 0100111101xxxxxx000101xxxxxxxxxx  Ssra_V  simd_sh_imm
    RawEntry {
        pattern: "0100111101xxxxxx000101xxxxxxxxxx",
        name: InstructionName::Ssra_V,
        emit: emit_ssra_v,
        decode: decode_simd_sh_imm,
    },
    // 0x001110<<1xxxxx001000xxxxxxxxxx  Ssubl_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx001000xxxxxxxxxx",
        name: InstructionName::Ssubl_V,
        emit: emit_ssubl_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx001100xxxxxxxxxx  Ssubw_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx001100xxxxxxxxxx",
        name: InstructionName::Ssubw_V,
        emit: emit_ssubw_v,
        decode: decode_simd_reg,
    },
    // 0x00110000000000xxxxxxxxxxxxxxxx  St__Vms  simd_mem_ms
    RawEntry {
        pattern: "0x00110000000000xxxxxxxxxxxxxxxx",
        name: InstructionName::St__Vms,
        emit: emit_st__vms,
        decode: decode_simd_mem_ms,
    },
    // 0x001100100xxxxxxxxxxxxxxxxxxxxx  St__Vms  simd_mem_ms
    RawEntry {
        pattern: "0x001100100xxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::St__Vms,
        emit: emit_st__vms,
        decode: decode_simd_mem_ms,
    },
    // 0x00110100x00000xxxxxxxxxxxxxxxx  St__Vss  simd_mem_ss
    RawEntry {
        pattern: "0x00110100x00000xxxxxxxxxxxxxxxx",
        name: InstructionName::St__Vss,
        emit: emit_st__vss,
        decode: decode_simd_mem_ss,
    },
    // 0x00110110xxxxxxxxxxxxxxxxxxxxxx  St__Vss  simd_mem_ss
    RawEntry {
        pattern: "0x00110110xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::St__Vss,
        emit: emit_st__vss,
        decode: decode_simd_mem_ss,
    },
    // <<10110xx0xxxxxxxxxxxxxxxxxxxxxx  Stp  simd_mem_pair
    RawEntry {
        pattern: "<<10110xx0xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Stp,
        emit: emit_stp,
        decode: decode_simd_mem_pair,
    },
    // xx111100x00xxxxxxxxx00xxxxxxxxxx  Str  simd_mem_imm
    RawEntry {
        pattern: "xx111100x00xxxxxxxxx00xxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_simd_mem_imm,
    },
    // xx111100x00xxxxxxxxx01xxxxxxxxxx  Str  simd_mem_imm
    RawEntry {
        pattern: "xx111100x00xxxxxxxxx01xxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_simd_mem_imm,
    },
    // xx111100x00xxxxxxxxx11xxxxxxxxxx  Str  simd_mem_imm
    RawEntry {
        pattern: "xx111100x00xxxxxxxxx11xxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_simd_mem_imm,
    },
    // xx111101x0xxxxxxxxxxxxxxxxxxxxxx  Str  simd_mem_imm
    RawEntry {
        pattern: "xx111101x0xxxxxxxxxxxxxxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_simd_mem_imm,
    },
    // xx111100x01xxxxxx1xx10xxxxxxxxxx  Str  simd_mem_reg
    RawEntry {
        pattern: "xx111100x01xxxxxx1xx10xxxxxxxxxx",
        name: InstructionName::Str,
        emit: emit_str,
        decode: decode_simd_mem_reg,
    },
    // 01111110111xxxxx100001xxxxxxxxxx  Sub_S  simd_reg
    RawEntry {
        pattern: "01111110111xxxxx100001xxxxxxxxxx",
        name: InstructionName::Sub_S,
        emit: emit_sub_s,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx100001xxxxxxxxxx  Sub_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx100001xxxxxxxxxx",
        name: InstructionName::Sub_V,
        emit: emit_sub_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<1xxxxx011000xxxxxxxxxx  Subhn_V  simd_reg
    RawEntry {
        pattern: "0x001110<<1xxxxx011000xxxxxxxxxx",
        name: InstructionName::Subhn_V,
        emit: emit_subhn_v,
        decode: decode_simd_reg,
    },
    // 01011110xx100000001110xxxxxxxxxx  Suqadd_S  simd
    RawEntry {
        pattern: "01011110xx100000001110xxxxxxxxxx",
        name: InstructionName::Suqadd_S,
        emit: emit_suqadd_s,
        decode: decode_simd,
    },
    // 0>001110<<100000001110xxxxxxxxxx  Suqadd_V  simd
    RawEntry {
        pattern: "0>001110<<100000001110xxxxxxxxxx",
        name: InstructionName::Suqadd_V,
        emit: emit_suqadd_v,
        decode: decode_simd,
    },
    // 0x001110000xxxxx0xx000xxxxxxxxxx  Tbl_V  simd_tbl
    RawEntry {
        pattern: "0x001110000xxxxx0xx000xxxxxxxxxx",
        name: InstructionName::Tbl_V,
        emit: emit_tbl_v,
        decode: decode_simd_tbl,
    },
    // 0x001110000xxxxx0xx100xxxxxxxxxx  Tbx_V  simd_tbl
    RawEntry {
        pattern: "0x001110000xxxxx0xx100xxxxxxxxxx",
        name: InstructionName::Tbx_V,
        emit: emit_tbx_v,
        decode: decode_simd_tbl,
    },
    // 0>001110<<0xxxxx001010xxxxxxxxxx  Trn1_V  simd_reg
    RawEntry {
        pattern: "0>001110<<0xxxxx001010xxxxxxxxxx",
        name: InstructionName::Trn1_V,
        emit: emit_trn1_v,
        decode: decode_simd_reg,
    },
    // 0>001110<<0xxxxx011010xxxxxxxxxx  Trn2_V  simd_reg
    RawEntry {
        pattern: "0>001110<<0xxxxx011010xxxxxxxxxx",
        name: InstructionName::Trn2_V,
        emit: emit_trn2_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx011111xxxxxxxxxx  Uaba_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx011111xxxxxxxxxx",
        name: InstructionName::Uaba_V,
        emit: emit_uaba_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx010100xxxxxxxxxx  Uabal_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx010100xxxxxxxxxx",
        name: InstructionName::Uabal_V,
        emit: emit_uabal_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx011101xxxxxxxxxx  Uabd_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx011101xxxxxxxxxx",
        name: InstructionName::Uabd_V,
        emit: emit_uabd_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx011100xxxxxxxxxx  Uabdl_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx011100xxxxxxxxxx",
        name: InstructionName::Uabdl_V,
        emit: emit_uabdl_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<100000011010xxxxxxxxxx  Uadalp_V  simd
    RawEntry {
        pattern: "0x101110<<100000011010xxxxxxxxxx",
        name: InstructionName::Uadalp_V,
        emit: emit_uadalp_v,
        decode: decode_simd,
    },
    // 0x101110<<1xxxxx000000xxxxxxxxxx  Uaddl_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx000000xxxxxxxxxx",
        name: InstructionName::Uaddl_V,
        emit: emit_uaddl_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<100000001010xxxxxxxxxx  Uaddlp_V  simd
    RawEntry {
        pattern: "0x101110<<100000001010xxxxxxxxxx",
        name: InstructionName::Uaddlp_V,
        emit: emit_uaddlp_v,
        decode: decode_simd,
    },
    // 001011100x110000001110xxxxxxxxxx  Uaddlv_V  simd
    RawEntry {
        pattern: "001011100x110000001110xxxxxxxxxx",
        name: InstructionName::Uaddlv_V,
        emit: emit_uaddlv_v,
        decode: decode_simd,
    },
    // 01101110<<110000001110xxxxxxxxxx  Uaddlv_V  simd
    RawEntry {
        pattern: "01101110<<110000001110xxxxxxxxxx",
        name: InstructionName::Uaddlv_V,
        emit: emit_uaddlv_v,
        decode: decode_simd,
    },
    // 0x101110<<1xxxxx000100xxxxxxxxxx  Uaddw_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx000100xxxxxxxxxx",
        name: InstructionName::Uaddw_V,
        emit: emit_uaddw_v,
        decode: decode_simd_reg,
    },
    // x00111100x100011000000xxxxxxxxxx  Ucvtf_Gp  simd_cvt
    RawEntry {
        pattern: "x00111100x100011000000xxxxxxxxxx",
        name: InstructionName::Ucvtf_Gp,
        emit: emit_ucvtf_gp,
        decode: decode_simd_cvt,
    },
    // >00111100x000011>xxxxxxxxxxxxxxx  Ucvtf_Gp_Fixed  simd_cvt
    RawEntry {
        pattern: ">00111100x000011>xxxxxxxxxxxxxxx",
        name: InstructionName::Ucvtf_Gp_Fixed,
        emit: emit_ucvtf_gp_fixed,
        decode: decode_simd_cvt,
    },
    // 011111100x100001110110xxxxxxxxxx  Ucvtf_S  simd
    RawEntry {
        pattern: "011111100x100001110110xxxxxxxxxx",
        name: InstructionName::Ucvtf_S,
        emit: emit_ucvtf_s,
        decode: decode_simd,
    },
    // 011111110>>xxxxx111001xxxxxxxxxx  Ucvtf_S_Fixed  simd_sh_imm
    RawEntry {
        pattern: "011111110>>xxxxx111001xxxxxxxxxx",
        name: InstructionName::Ucvtf_S_Fixed,
        emit: emit_ucvtf_s_fixed,
        decode: decode_simd_sh_imm,
    },
    // 0>1011100<100001110110xxxxxxxxxx  Ucvtf_V  simd
    RawEntry {
        pattern: "0>1011100<100001110110xxxxxxxxxx",
        name: InstructionName::Ucvtf_V,
        emit: emit_ucvtf_v,
        decode: decode_simd,
    },
    // 0x101111001xxxxx111001xxxxxxxxxx  Ucvtf_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0x101111001xxxxx111001xxxxxxxxxx",
        name: InstructionName::Ucvtf_V_Fixed,
        emit: emit_ucvtf_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx111001xxxxxxxxxx  Ucvtf_V_Fixed  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx111001xxxxxxxxxx",
        name: InstructionName::Ucvtf_V_Fixed,
        emit: emit_ucvtf_v_fixed,
        decode: decode_simd_sh_imm,
    },
    // 0x101110<<1xxxxx000001xxxxxxxxxx  Uhadd_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx000001xxxxxxxxxx",
        name: InstructionName::Uhadd_V,
        emit: emit_uhadd_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx001001xxxxxxxxxx  Uhsub_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx001001xxxxxxxxxx",
        name: InstructionName::Uhsub_V,
        emit: emit_uhsub_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx011001xxxxxxxxxx  Umax_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx011001xxxxxxxxxx",
        name: InstructionName::Umax_V,
        emit: emit_umax_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx101001xxxxxxxxxx  Umaxp_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx101001xxxxxxxxxx",
        name: InstructionName::Umaxp_V,
        emit: emit_umaxp_v,
        decode: decode_simd_reg,
    },
    // 001011100x110000101010xxxxxxxxxx  Umaxv_V  simd
    RawEntry {
        pattern: "001011100x110000101010xxxxxxxxxx",
        name: InstructionName::Umaxv_V,
        emit: emit_umaxv_v,
        decode: decode_simd,
    },
    // 01101110<<110000101010xxxxxxxxxx  Umaxv_V  simd
    RawEntry {
        pattern: "01101110<<110000101010xxxxxxxxxx",
        name: InstructionName::Umaxv_V,
        emit: emit_umaxv_v,
        decode: decode_simd,
    },
    // 0x101110<<1xxxxx011011xxxxxxxxxx  Umin_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx011011xxxxxxxxxx",
        name: InstructionName::Umin_V,
        emit: emit_umin_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx101011xxxxxxxxxx  Uminp_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx101011xxxxxxxxxx",
        name: InstructionName::Uminp_V,
        emit: emit_uminp_v,
        decode: decode_simd_reg,
    },
    // 001011100x110001101010xxxxxxxxxx  Uminv_V  simd
    RawEntry {
        pattern: "001011100x110001101010xxxxxxxxxx",
        name: InstructionName::Uminv_V,
        emit: emit_uminv_v,
        decode: decode_simd,
    },
    // 01101110<<110001101010xxxxxxxxxx  Uminv_V  simd
    RawEntry {
        pattern: "01101110<<110001101010xxxxxxxxxx",
        name: InstructionName::Uminv_V,
        emit: emit_uminv_v,
        decode: decode_simd,
    },
    // 0x101110<<1xxxxx100000xxxxxxxxxx  Umlal_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx100000xxxxxxxxxx",
        name: InstructionName::Umlal_V,
        emit: emit_umlal_v,
        decode: decode_simd_reg,
    },
    // 0x101111xxxxxxxx0010x0xxxxxxxxxx  Umlal_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x101111xxxxxxxx0010x0xxxxxxxxxx",
        name: InstructionName::Umlal_Ve,
        emit: emit_umlal_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x101110<<1xxxxx101000xxxxxxxxxx  Umlsl_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx101000xxxxxxxxxx",
        name: InstructionName::Umlsl_V,
        emit: emit_umlsl_v,
        decode: decode_simd_reg,
    },
    // 0x101111xxxxxxxx0110x0xxxxxxxxxx  Umlsl_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x101111xxxxxxxx0110x0xxxxxxxxxx",
        name: InstructionName::Umlsl_Ve,
        emit: emit_umlsl_ve,
        decode: decode_simd_reg_elem,
    },
    // 0x001110000xxxxx001111xxxxxxxxxx  Umov_S  simd_ins
    RawEntry {
        pattern: "0x001110000xxxxx001111xxxxxxxxxx",
        name: InstructionName::Umov_S,
        emit: emit_umov_s,
        decode: decode_simd_ins,
    },
    // 0x101110<<1xxxxx110000xxxxxxxxxx  Umull_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx110000xxxxxxxxxx",
        name: InstructionName::Umull_V,
        emit: emit_umull_v,
        decode: decode_simd_reg,
    },
    // 0x101111xxxxxxxx1010x0xxxxxxxxxx  Umull_Ve  simd_reg_elem
    RawEntry {
        pattern: "0x101111xxxxxxxx1010x0xxxxxxxxxx",
        name: InstructionName::Umull_Ve,
        emit: emit_umull_ve,
        decode: decode_simd_reg_elem,
    },
    // 01111110xx1xxxxx000011xxxxxxxxxx  Uqadd_S  simd_reg
    RawEntry {
        pattern: "01111110xx1xxxxx000011xxxxxxxxxx",
        name: InstructionName::Uqadd_S,
        emit: emit_uqadd_s,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx000011xxxxxxxxxx  Uqadd_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx000011xxxxxxxxxx",
        name: InstructionName::Uqadd_V,
        emit: emit_uqadd_v,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx010111xxxxxxxxxx  Uqrshl_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx010111xxxxxxxxxx",
        name: InstructionName::Uqrshl_V,
        emit: emit_uqrshl_v,
        decode: decode_simd_reg,
    },
    // 0111111100>>>xxx100111xxxxxxxxxx  Uqrshrn_S  simd_sh_imm
    RawEntry {
        pattern: "0111111100>>>xxx100111xxxxxxxxxx",
        name: InstructionName::Uqrshrn_S,
        emit: emit_uqrshrn_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx100111xxxxxxxxxx  Uqrshrn_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx100111xxxxxxxxxx",
        name: InstructionName::Uqrshrn_V,
        emit: emit_uqrshrn_v,
        decode: decode_simd_sh_imm,
    },
    // 0>101110<<1xxxxx010011xxxxxxxxxx  Uqshl_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx010011xxxxxxxxxx",
        name: InstructionName::Uqshl_V,
        emit: emit_uqshl_v,
        decode: decode_simd_reg,
    },
    // 0111111100>>>xxx100101xxxxxxxxxx  Uqshrn_S  simd_sh_imm
    RawEntry {
        pattern: "0111111100>>>xxx100101xxxxxxxxxx",
        name: InstructionName::Uqshrn_S,
        emit: emit_uqshrn_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx100101xxxxxxxxxx  Uqshrn_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx100101xxxxxxxxxx",
        name: InstructionName::Uqshrn_V,
        emit: emit_uqshrn_v,
        decode: decode_simd_sh_imm,
    },
    // 01111110xx1xxxxx001011xxxxxxxxxx  Uqsub_S  simd_reg
    RawEntry {
        pattern: "01111110xx1xxxxx001011xxxxxxxxxx",
        name: InstructionName::Uqsub_S,
        emit: emit_uqsub_s,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx001011xxxxxxxxxx  Uqsub_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx001011xxxxxxxxxx",
        name: InstructionName::Uqsub_V,
        emit: emit_uqsub_v,
        decode: decode_simd_reg,
    },
    // 01111110<<100001010010xxxxxxxxxx  Uqxtn_S  simd
    RawEntry {
        pattern: "01111110<<100001010010xxxxxxxxxx",
        name: InstructionName::Uqxtn_S,
        emit: emit_uqxtn_s,
        decode: decode_simd,
    },
    // 0x101110<<100001010010xxxxxxxxxx  Uqxtn_V  simd
    RawEntry {
        pattern: "0x101110<<100001010010xxxxxxxxxx",
        name: InstructionName::Uqxtn_V,
        emit: emit_uqxtn_v,
        decode: decode_simd,
    },
    // 0x101110<<1xxxxx000101xxxxxxxxxx  Urhadd_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx000101xxxxxxxxxx",
        name: InstructionName::Urhadd_V,
        emit: emit_urhadd_v,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx010101xxxxxxxxxx  Urshl_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx010101xxxxxxxxxx",
        name: InstructionName::Urshl_V,
        emit: emit_urshl_v,
        decode: decode_simd_reg,
    },
    // 0111111101xxxxxx001001xxxxxxxxxx  Urshr_S  simd_sh_imm
    RawEntry {
        pattern: "0111111101xxxxxx001001xxxxxxxxxx",
        name: InstructionName::Urshr_S,
        emit: emit_urshr_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx001001xxxxxxxxxx  Urshr_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx001001xxxxxxxxxx",
        name: InstructionName::Urshr_V,
        emit: emit_urshr_v,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx001001xxxxxxxxxx  Urshr_V  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx001001xxxxxxxxxx",
        name: InstructionName::Urshr_V,
        emit: emit_urshr_v,
        decode: decode_simd_sh_imm,
    },
    // 0111111101xxxxxx001101xxxxxxxxxx  Ursra_S  simd_sh_imm
    RawEntry {
        pattern: "0111111101xxxxxx001101xxxxxxxxxx",
        name: InstructionName::Ursra_S,
        emit: emit_ursra_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx001101xxxxxxxxxx  Ursra_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx001101xxxxxxxxxx",
        name: InstructionName::Ursra_V,
        emit: emit_ursra_v,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx001101xxxxxxxxxx  Ursra_V  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx001101xxxxxxxxxx",
        name: InstructionName::Ursra_V,
        emit: emit_ursra_v,
        decode: decode_simd_sh_imm,
    },
    // 01111110111xxxxx010001xxxxxxxxxx  Ushl_S  simd_reg
    RawEntry {
        pattern: "01111110111xxxxx010001xxxxxxxxxx",
        name: InstructionName::Ushl_S,
        emit: emit_ushl_s,
        decode: decode_simd_reg,
    },
    // 0>101110<<1xxxxx010001xxxxxxxxxx  Ushl_V  simd_reg
    RawEntry {
        pattern: "0>101110<<1xxxxx010001xxxxxxxxxx",
        name: InstructionName::Ushl_V,
        emit: emit_ushl_v,
        decode: decode_simd_reg,
    },
    // 0x10111100>>>xxx101001xxxxxxxxxx  Ushll_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx101001xxxxxxxxxx",
        name: InstructionName::Ushll_V,
        emit: emit_ushll_v,
        decode: decode_simd_sh_imm,
    },
    // 0111111101xxxxxx000001xxxxxxxxxx  Ushr_S  simd_sh_imm
    RawEntry {
        pattern: "0111111101xxxxxx000001xxxxxxxxxx",
        name: InstructionName::Ushr_S,
        emit: emit_ushr_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx000001xxxxxxxxxx  Ushr_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx000001xxxxxxxxxx",
        name: InstructionName::Ushr_V,
        emit: emit_ushr_v,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx000001xxxxxxxxxx  Ushr_V  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx000001xxxxxxxxxx",
        name: InstructionName::Ushr_V,
        emit: emit_ushr_v,
        decode: decode_simd_sh_imm,
    },
    // 01111110xx100000001110xxxxxxxxxx  Usqadd_S  simd
    RawEntry {
        pattern: "01111110xx100000001110xxxxxxxxxx",
        name: InstructionName::Usqadd_S,
        emit: emit_usqadd_s,
        decode: decode_simd,
    },
    // 0>101110<<100000001110xxxxxxxxxx  Usqadd_V  simd
    RawEntry {
        pattern: "0>101110<<100000001110xxxxxxxxxx",
        name: InstructionName::Usqadd_V,
        emit: emit_usqadd_v,
        decode: decode_simd,
    },
    // 0111111101xxxxxx000101xxxxxxxxxx  Usra_S  simd_sh_imm
    RawEntry {
        pattern: "0111111101xxxxxx000101xxxxxxxxxx",
        name: InstructionName::Usra_S,
        emit: emit_usra_s,
        decode: decode_simd_sh_imm,
    },
    // 0x10111100>>>xxx000101xxxxxxxxxx  Usra_V  simd_sh_imm
    RawEntry {
        pattern: "0x10111100>>>xxx000101xxxxxxxxxx",
        name: InstructionName::Usra_V,
        emit: emit_usra_v,
        decode: decode_simd_sh_imm,
    },
    // 0110111101xxxxxx000101xxxxxxxxxx  Usra_V  simd_sh_imm
    RawEntry {
        pattern: "0110111101xxxxxx000101xxxxxxxxxx",
        name: InstructionName::Usra_V,
        emit: emit_usra_v,
        decode: decode_simd_sh_imm,
    },
    // 0x101110<<1xxxxx001000xxxxxxxxxx  Usubl_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx001000xxxxxxxxxx",
        name: InstructionName::Usubl_V,
        emit: emit_usubl_v,
        decode: decode_simd_reg,
    },
    // 0x101110<<1xxxxx001100xxxxxxxxxx  Usubw_V  simd_reg
    RawEntry {
        pattern: "0x101110<<1xxxxx001100xxxxxxxxxx",
        name: InstructionName::Usubw_V,
        emit: emit_usubw_v,
        decode: decode_simd_reg,
    },
    // 0>001110<<0xxxxx000110xxxxxxxxxx  Uzp1_V  simd_reg
    RawEntry {
        pattern: "0>001110<<0xxxxx000110xxxxxxxxxx",
        name: InstructionName::Uzp1_V,
        emit: emit_uzp1_v,
        decode: decode_simd_reg,
    },
    // 0>001110<<0xxxxx010110xxxxxxxxxx  Uzp2_V  simd_reg
    RawEntry {
        pattern: "0>001110<<0xxxxx010110xxxxxxxxxx",
        name: InstructionName::Uzp2_V,
        emit: emit_uzp2_v,
        decode: decode_simd_reg,
    },
    // 0x001110<<100001001010xxxxxxxxxx  Xtn_V  simd
    RawEntry {
        pattern: "0x001110<<100001001010xxxxxxxxxx",
        name: InstructionName::Xtn_V,
        emit: emit_xtn_v,
        decode: decode_simd,
    },
    // 0>001110<<0xxxxx001110xxxxxxxxxx  Zip1_V  simd_reg
    RawEntry {
        pattern: "0>001110<<0xxxxx001110xxxxxxxxxx",
        name: InstructionName::Zip1_V,
        emit: emit_zip1_v,
        decode: decode_simd_reg,
    },
    // 0>001110<<0xxxxx011110xxxxxxxxxx  Zip2_V  simd_reg
    RawEntry {
        pattern: "0>001110<<0xxxxx011110xxxxxxxxxx",
        name: InstructionName::Zip2_V,
        emit: emit_zip2_v,
        decode: decode_simd_reg,
    },
];
