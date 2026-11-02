// Generate deterministic AArch64 instruction vectors for tools/oracle.
// Randomizes free encoding bits and input state; skips encodings rejected by the decoder.
//
// Usage: cargo run --example gen_vectors > vectors.spec

use armature::decoder::decode_one::decode_one;
use armature::decoder::instruction_name::InstructionName;
use armature::decoder::opcode_kind::OpCodeKind;
use armature::decoder::table_entries::A64_RAW;

// xorshift64* with a fixed seed for reproducible vectors.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
}

const SEED: u64 = 0x4152_4D45_494C_4C45; // "ARMEILLE"
const VECTORS_PER_OPCODE: usize = 24;

// Whitelist: pure-register integer instructions. No branches
// (escape the opcode+RET stub), no memory (no data mapping yet), no
// system/exception.
fn in_scope(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Adc | Adcs
            | Add
            | Adds
            | And
            | Ands
            | Asrv
            | Bic
            | Bics
            | Cls
            | Clz
            | Eon
            | Eor
            | Extr
            | Lslv
            | Lsrv
            | Orn
            | Orr
            | Rbit
            | Rev16
            | Rev32
            | Rev64
            | Rorv
            | Sbc
            | Sbcs
            | Sub
            | Subs
            | Csel
            | Csinc
            | Csinv
            | Csneg
            | Ccmn
            | Ccmp
            | Madd
            | Msub
            | Smaddl
            | Smsubl
            | Smulh
            | Umaddl
            | Umsubl
            | Umulh
            | Sdiv
            | Udiv
            | Movk
            | Movn
            | Movz
            | Sbfm
            | Ubfm
            | Bfm
            | Crc32B
            | Crc32H
            | Crc32W
            | Crc32X
            | Crc32Cb
            | Crc32Ch
            | Crc32Cw
            | Crc32Cx
    ) || is_memory(name)
        || is_simd_logical(name)
        || is_simd_arithmetic(name)
        || is_simd_shift(name)
        || is_simd_move(name)
        || is_simd_cmp(name)
        || is_simd_cvt(name)
        || is_simd_crypto(name)
        || is_simd_hash(name)
        || is_simd_memory(name)
}

// InstEmitSimdLogical.cs coverage.
fn is_simd_logical(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        And_V
            | Bic_V
            | Bic_Vi
            | Bif_V
            | Bit_V
            | Bsl_V
            | Eor_V
            | Not_V
            | Orn_V
            | Orr_V
            | Orr_Vi
            | Rbit_V
            | Rev16_V
            | Rev32_V
            | Rev64_V
    )
}

// InstEmitSimdShift.cs coverage.
fn is_simd_shift(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Rshrn_V
            | Shl_S
            | Shl_V
            | Shll_V
            | Shrn_V
            | Sli_S
            | Sli_V
            | Sqrshl_V
            | Sqrshrn_S
            | Sqrshrn_V
            | Sqrshrun_S
            | Sqrshrun_V
            | Sqshl_Si
            | Sqshl_V
            | Sqshl_Vi
            | Sqshrn_S
            | Sqshrn_V
            | Sqshrun_S
            | Sqshrun_V
            | Sri_S
            | Sri_V
            | Srshl_V
            | Srshr_S
            | Srshr_V
            | Srsra_S
            | Srsra_V
            | Sshl_S
            | Sshl_V
            | Sshll_V
            | Sshr_S
            | Sshr_V
            | Ssra_S
            | Ssra_V
            | Uqrshl_V
            | Uqrshrn_S
            | Uqrshrn_V
            | Uqshl_V
            | Uqshrn_S
            | Uqshrn_V
            | Urshl_V
            | Urshr_S
            | Urshr_V
            | Ursra_S
            | Ursra_V
            | Ushl_S
            | Ushl_V
            | Ushll_V
            | Ushr_S
            | Ushr_V
            | Usra_S
            | Usra_V
    )
}

// InstEmitSimdMove.cs coverage.
fn is_simd_move(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Dup_Gp
            | Dup_S
            | Dup_V
            | Ext_V
            | Fcsel_S
            | Fmov_S
            | Fmov_Si
            | Fmov_Vi
            | Fmov_Ftoi
            | Fmov_Ftoi1
            | Fmov_Itof
            | Fmov_Itof1
            | Ins_Gp
            | Ins_V
            | Movi_V
            | Mvni_V
            | Smov_S
            | Tbl_V
            | Tbx_V
            | Trn1_V
            | Trn2_V
            | Umov_S
            | Uzp1_V
            | Uzp2_V
            | Xtn_V
            | Zip1_V
            | Zip2_V
    )
}

// InstEmitSimdArithmetic.cs coverage.
fn is_simd_arithmetic(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Abs_S
            | Abs_V
            | Add_S
            | Add_V
            | Addhn_V
            | Addp_S
            | Addp_V
            | Addv_V
            | Cls_V
            | Clz_V
            | Cnt_V
            | Fabd_S
            | Fabd_V
            | Fabs_S
            | Fabs_V
            | Fadd_S
            | Fadd_V
            | Faddp_S
            | Faddp_V
            | Fdiv_S
            | Fdiv_V
            | Fmadd_S
            | Fmax_S
            | Fmax_V
            | Fmaxnm_S
            | Fmaxnm_V
            | Fmaxnmp_S
            | Fmaxnmp_V
            | Fmaxnmv_V
            | Fmaxp_S
            | Fmaxp_V
            | Fmaxv_V
            | Fmin_S
            | Fmin_V
            | Fminnm_S
            | Fminnm_V
            | Fminnmp_S
            | Fminnmp_V
            | Fminnmv_V
            | Fminp_S
            | Fminp_V
            | Fminv_V
            | Fmla_Se
            | Fmla_V
            | Fmla_Ve
            | Fmls_Se
            | Fmls_V
            | Fmls_Ve
            | Fmulx_S
            | Fmulx_Se
            | Fmulx_V
            | Fmulx_Ve
            | Frecpx_S
            | Frinti_S
            | Frinti_V
            | Frintx_S
            | Frintx_V
            | Fmsub_S
            | Fmul_S
            | Fmul_Se
            | Fmul_V
            | Fmul_Ve
            | Fneg_S
            | Fneg_V
            | Fnmadd_S
            | Fnmsub_S
            | Fnmul_S
            | Frecpe_S
            | Frecpe_V
            | Frecps_S
            | Frecps_V
            | Frinta_S
            | Frinta_V
            | Frintm_S
            | Frintm_V
            | Frintn_S
            | Frintn_V
            | Frintp_S
            | Frintp_V
            | Frintz_S
            | Frintz_V
            | Frsqrte_S
            | Frsqrte_V
            | Frsqrts_S
            | Frsqrts_V
            | Fsqrt_S
            | Fsqrt_V
            | Fsub_S
            | Fsub_V
            | Mla_V
            | Mla_Ve
            | Mls_V
            | Mls_Ve
            | Mul_V
            | Mul_Ve
            | Neg_S
            | Neg_V
            | Pmull_V
            | Raddhn_V
            | Rsubhn_V
            | Saba_V
            | Sabal_V
            | Sabd_V
            | Sabdl_V
            | Sadalp_V
            | Saddl_V
            | Saddlp_V
            | Saddlv_V
            | Saddw_V
            | Shadd_V
            | Shsub_V
            | Smax_V
            | Smaxp_V
            | Smaxv_V
            | Smin_V
            | Sminp_V
            | Sminv_V
            | Smlal_V
            | Smlal_Ve
            | Smlsl_V
            | Smlsl_Ve
            | Smull_V
            | Smull_Ve
            | Sqabs_S
            | Sqabs_V
            | Sqadd_S
            | Sqadd_V
            | Sqdmulh_S
            | Sqdmulh_V
            | Sqdmulh_Ve
            | Sqneg_S
            | Sqneg_V
            | Sqrdmulh_S
            | Sqrdmulh_V
            | Sqrdmulh_Ve
            | Sqsub_S
            | Sqsub_V
            | Sqxtn_S
            | Sqxtn_V
            | Sqxtun_S
            | Sqxtun_V
            | Srhadd_V
            | Ssubl_V
            | Ssubw_V
            | Sub_S
            | Sub_V
            | Subhn_V
            | Suqadd_S
            | Suqadd_V
            | Uaba_V
            | Uabal_V
            | Uabd_V
            | Uabdl_V
            | Uadalp_V
            | Uaddl_V
            | Uaddlp_V
            | Uaddlv_V
            | Uaddw_V
            | Uhadd_V
            | Uhsub_V
            | Umax_V
            | Umaxp_V
            | Umaxv_V
            | Umin_V
            | Uminp_V
            | Uminv_V
            | Umlal_V
            | Umlal_Ve
            | Umlsl_V
            | Umlsl_Ve
            | Umull_V
            | Umull_Ve
            | Uqadd_S
            | Uqadd_V
            | Uqsub_S
            | Uqsub_V
            | Uqxtn_S
            | Uqxtn_V
            | Urhadd_V
            | Usqadd_S
            | Usqadd_V
            | Usubl_V
            | Usubw_V
    )
}

// InstEmitSimdCmp.cs coverage.
fn is_simd_cmp(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Cmeq_S
            | Cmeq_V
            | Cmge_S
            | Cmge_V
            | Cmgt_S
            | Cmgt_V
            | Cmhi_S
            | Cmhi_V
            | Cmhs_S
            | Cmhs_V
            | Cmle_S
            | Cmle_V
            | Cmlt_S
            | Cmlt_V
            | Cmtst_S
            | Cmtst_V
            | Facge_S
            | Facge_V
            | Facgt_S
            | Facgt_V
            | Fccmp_S
            | Fccmpe_S
            | Fcmeq_S
            | Fcmeq_V
            | Fcmge_S
            | Fcmge_V
            | Fcmgt_S
            | Fcmgt_V
            | Fcmle_S
            | Fcmle_V
            | Fcmlt_S
            | Fcmlt_V
            | Fcmp_S
            | Fcmpe_S
    )
}

// InstEmitSimdCvt.cs coverage.
fn is_simd_cvt(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Fcvt_S
            | Fcvtas_Gp
            | Fcvtas_S
            | Fcvtas_V
            | Fcvtau_Gp
            | Fcvtau_S
            | Fcvtau_V
            | Fcvtl_V
            | Fcvtn_V
            | Fcvtms_Gp
            | Fcvtms_V
            | Fcvtmu_Gp
            | Fcvtns_Gp
            | Fcvtns_S
            | Fcvtns_V
            | Fcvtnu_S
            | Fcvtnu_V
            | Fcvtps_Gp
            | Fcvtpu_Gp
            | Fcvtzs_Gp
            | Fcvtzs_Gp_Fixed
            | Fcvtzs_S
            | Fcvtzs_V
            | Fcvtzs_V_Fixed
            | Fcvtzu_Gp
            | Fcvtzu_Gp_Fixed
            | Fcvtzu_S
            | Fcvtzu_V
            | Fcvtzu_V_Fixed
            | Scvtf_Gp
            | Scvtf_Gp_Fixed
            | Scvtf_S
            | Scvtf_S_Fixed
            | Scvtf_V
            | Scvtf_V_Fixed
            | Ucvtf_Gp
            | Ucvtf_Gp_Fixed
            | Ucvtf_S
            | Ucvtf_S_Fixed
            | Ucvtf_V
            | Ucvtf_V_Fixed
    )
}

// InstEmitSimdCrypto.cs coverage (AESNI paths  host has AES).
fn is_simd_crypto(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(name, Aesd_V | Aese_V | Aesimc_V | Aesmc_V)
}

// InstEmitSimdHash.cs coverage. SHA1 runs the SoftFallback
// path  upstream has no x86 SHA1 fast path either.
fn is_simd_hash(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Sha1c_V
            | Sha1h_V
            | Sha1m_V
            | Sha1p_V
            | Sha1su0_V
            | Sha1su1_V
            | Sha256h_V
            | Sha256h2_V
            | Sha256su0_V
            | Sha256su1_V
    )
}

// InstEmitSimdMemory.cs coverage (LD/ST structure forms).
fn is_simd_memory(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(name, Ld__Vms | Ld__Vss | St__Vms | St__Vss)
}

// Memory vectors: memory instructions. Addressing registers are
// forced into {0..3, 31} (the spec-settable set) and bases are pointed
// into the data page.
fn is_memory(name: InstructionName) -> bool {
    use armature::decoder::instruction_name::InstructionName::*;
    matches!(
        name,
        Ldr | Ldrs
            | Str
            | Ldp
            | Stp
            | Ldar
            | Ldaxr
            | Ldxr
            | Ldxp
            | Ldaxp
            | Stlr
            | Stlxr
            | Stxr
            | Stxp
            | Stlxp
    )
}

const DATA_BASE: u64 = 0x2000;

// Overwrite a 5-bit register field with a value from the settable set.
fn force_reg_field(raw: u32, lsb: u32, rng: &mut Rng, allow_sp: bool) -> u32 {
    let pool: &[u32] = if allow_sp {
        &[0, 1, 2, 3, 31]
    } else {
        &[0, 1, 2, 3]
    };
    let pick = pool[(rng.next() % pool.len() as u64) as usize];
    (raw & !(0x1f << lsb)) | (pick << lsb)
}

fn main() {
    let mut rng = Rng(SEED);

    println!("# armature differential vectors  seed {SEED:#x}, {VECTORS_PER_OPCODE}/opcode");

    let mut emitted = 0usize;

    for entry in A64_RAW {
        if !in_scope(entry.name) {
            continue;
        }

        // Derive mask/value from the pattern string (same derivation the
        // table builder uses: fixed bits 0/1, free bits x/>).
        let mut mask: u32 = 0;
        let mut value: u32 = 0;
        for (i, c) in entry.pattern.chars().enumerate() {
            let bit = 31 - i as u32;
            match c {
                '0' => mask |= 1 << bit,
                '1' => {
                    mask |= 1 << bit;
                    value |= 1 << bit;
                }
                _ => {}
            }
        }

        let mut produced = 0usize;
        let mut attempts = 0usize;

        let memory_family = is_memory(entry.name);
        let simd_mem_family = is_simd_memory(entry.name);

        while produced < VECTORS_PER_OPCODE && attempts < VECTORS_PER_OPCODE * 8 {
            attempts += 1;

            let mut raw = (rng.next_u32() & !mask) | value;

            if memory_family {
                // Force every register field into the settable set so the
                // base can be pointed at the data page. rt/rt2/rs allow
                // 31 (= ZR for those roles), rn allows 31 (= SP).
                raw = force_reg_field(raw, 0, &mut rng, true); // rt
                raw = force_reg_field(raw, 5, &mut rng, true); // rn
                // rm / rs / rt2 share bits 16-20 / 10-14 across the
                // memory kinds; force both  they are free pattern bits
                // wherever they exist.
                if mask & (0x1f << 16) == 0 {
                    raw = force_reg_field(raw, 16, &mut rng, true);
                }
                if mask & (0x1f << 10) == 0 {
                    raw = force_reg_field(raw, 10, &mut rng, true);
                }
            }

            if simd_mem_family {
                // rn = base (SP allowed). rm (bits 16-20, free only in the
                // post-index forms): 31 = immediate writeback, 0-3 =
                // register writeback from the settable set. rt stays
                // random  Vt is a vector register, all 32 are compared.
                raw = force_reg_field(raw, 5, &mut rng, true);
                if mask & (0x1f << 16) == 0 {
                    raw = force_reg_field(raw, 16, &mut rng, true);
                }
            }

            // Keep register fields in the tested range: remap Rd/Rn/Rm/Ra
            // nibbles occasionally to 31 to also exercise ZR/SP paths
            // the random bits already do this naturally; just filter
            // through the decoder.
            if decode_one(raw, 0x1000).is_err() {
                continue;
            }

            // Rd = X30 would clobber LR and break the opcode+RET stub
            // (same constraint the C# CpuTest cases apply to rd).
            if raw & 0x1f == 30 {
                continue;
            }

            // Fcvt_S: only 6 (type,opc) pairs are allocated conversions.
            // The C# table accepts the rest and Debug.Assert-FailFasts in
            // the emitter; ours panics. Filter like CpuTest does.
            if entry.name == InstructionName::Fcvt_S {
                if let Ok(d) = decode_one(raw, 0x1000) {
                    if let OpCodeKind::Simd { size, opc, .. } = d.kind {
                        if size == 2 || opc == 2 || size == opc {
                            continue;
                        }
                    }
                }
            }

            let mut x0 = rng.next();
            let mut x1 = rng.next();
            let mut x2 = rng.next();
            let mut x3 = rng.next();

            // Known upstream divergence: C# ARMeilleure's
            // PCLMULQDQ CRC32X path leaks the upper 32 bits of the crc
            // register (Wn) into the computation; spec/soft/SSE4.2 all
            // truncate. Constrain CRC32X inputs to W-width so vectors
            // compare cleanly  our (correct) behavior is unaffected.
            if entry.name == InstructionName::Crc32X {
                x0 &= 0xffff_ffff;
                x1 &= 0xffff_ffff;
                x2 &= 0xffff_ffff;
                x3 &= 0xffff_ffff;
            }
            let mut sp = rng.next() & !0xf; // keep SP 16-aligned

            if memory_family || simd_mem_family {
                // Point the base register at the data page so the
                // effective address (incl. immediate/offset) stays in
                // [DATA_BASE, DATA_BASE + 0x1000).
                let decoded = decode_one(raw, 0x1000).unwrap();
                let (rn, base) = match decoded.kind {
                    OpCodeKind::MemImm {
                        rn,
                        immediate,
                        post_idx,
                        ..
                    } => {
                        let slack = DATA_BASE as i64 + 0x780 + (rng.next() & 0x3f0) as i64;
                        let base = if post_idx { slack } else { slack - immediate };
                        (rn.idx(), base as u64)
                    }
                    OpCodeKind::MemPair {
                        rn,
                        immediate,
                        post_idx,
                        ..
                    } => {
                        let slack = DATA_BASE as i64 + 0x780 + ((rng.next() & 0x3f) as i64) * 16;
                        let base = if post_idx { slack } else { slack - immediate };
                        (rn.idx(), base as u64)
                    }
                    OpCodeKind::MemReg { rn, rm, .. } => {
                        // rn == rm: the base assignment below would clobber
                        // the offset  skip the combination.
                        if rn.idx() == rm.idx() {
                            continue;
                        }
                        // Offset register: small positive value so any
                        // extend/shift stays within the page.
                        let off = rng.next() & 0x3f;
                        let base = DATA_BASE + 0x400 + (rng.next() & 0x1f0);
                        // rm in settable set or ZR.
                        match rm.idx() {
                            0 => x0 = off,
                            1 => x1 = off,
                            2 => x2 = off,
                            3 => x3 = off,
                            _ => {} // ZR  offset 0
                        }
                        (rn.idx(), base)
                    }
                    OpCodeKind::MemEx { rn, .. } => {
                        // Exclusives require natural alignment; 256-align
                        // covers every access size.
                        (rn.idx(), DATA_BASE + 0x400 + ((rng.next() & 0xb) << 8))
                    }
                    // Structure forms touch at most 4 regs × 16 bytes = 64
                    // bytes from base; keep the base mid-page.
                    OpCodeKind::SimdMemMs { rn, .. } | OpCodeKind::SimdMemSs { rn, .. } => {
                        (rn.idx(), DATA_BASE + 0x400 + (rng.next() & 0x3f0))
                    }
                    _ => continue,
                };

                match rn {
                    0 => x0 = base,
                    1 => x1 = base,
                    2 => x2 = base,
                    3 => x3 = base,
                    31 => sp = base,
                    r => panic!("unforced rn {r}"),
                }
            }

            let nzcv = (rng.next_u32() & 0xf) << 28;
            let dseed = rng.next() | 1;

            // v0-v3 inputs (32 hex chars each); only SIMD families get
            // nonzero values  encodings touching v4+ read zeros on both
            // sides, which stays comparable.
            let (v0, v1, v2, v3) = if is_simd_logical(entry.name)
                || is_simd_arithmetic(entry.name)
                || is_simd_shift(entry.name)
                || is_simd_move(entry.name)
                || is_simd_cvt(entry.name)
                || is_simd_crypto(entry.name)
                || is_simd_hash(entry.name)
                || is_simd_memory(entry.name)
            {
                (
                    (rng.next() as u128) << 64 | rng.next() as u128,
                    (rng.next() as u128) << 64 | rng.next() as u128,
                    (rng.next() as u128) << 64 | rng.next() as u128,
                    (rng.next() as u128) << 64 | rng.next() as u128,
                )
            } else {
                (0, 0, 0, 0)
            };

            println!(
                "{raw:08x} {x0:x} {x1:x} {x2:x} {x3:x} {sp:x} {nzcv:x} {dseed:x} {v0:032x} {v1:032x} {v2:032x} {v3:032x}"
            );
            produced += 1;
        }

        emitted += produced;
    }

    eprintln!("gen_vectors: {emitted} vectors");
}
