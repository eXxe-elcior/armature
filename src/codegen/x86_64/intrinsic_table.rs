// Port of CodeGen/X86/IntrinsicType.cs + IntrinsicInfo.cs +
// IntrinsicTable.cs. Body generated mechanically from the C# source in
// our Intrinsic enum order (X86None gets a placeholder  C# leaves its
// slot default and nothing emits it).

use super::instruction::X86Instruction;
use crate::ir::types::Intrinsic;

/// C# enum IntrinsicType.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicType {
    Comis_,
    Mxcsr,
    PopCount,
    Unary,
    UnaryToGpr,
    Binary,
    BinaryGpr,
    BinaryImm,
    Crc32,
    Ternary,
    TernaryImm,
    Fma,
}

/// C# readonly struct IntrinsicInfo.
#[derive(Debug, Clone, Copy)]
pub struct IntrinsicInfo {
    pub inst: X86Instruction,
    pub typ: IntrinsicType,
}

const fn instrinsic_info(inst: X86Instruction, typ: IntrinsicType) -> IntrinsicInfo {
    IntrinsicInfo { inst, typ }
}

/// C# IntrinsicTable.GetInfo.
pub fn info(intrin: Intrinsic) -> IntrinsicInfo {
    INTRIN_TABLE[intrin as usize]
}

#[rustfmt::skip]
static INTRIN_TABLE: [IntrinsicInfo; 193] = [
    instrinsic_info(X86Instruction::None, IntrinsicType::Unary), // X86None (no table entry)
    instrinsic_info(X86Instruction::Addpd, IntrinsicType::Binary), // X86Addpd
    instrinsic_info(X86Instruction::Addps, IntrinsicType::Binary), // X86Addps
    instrinsic_info(X86Instruction::Addsd, IntrinsicType::Binary), // X86Addsd
    instrinsic_info(X86Instruction::Addss, IntrinsicType::Binary), // X86Addss
    instrinsic_info(X86Instruction::Aesdec, IntrinsicType::Binary), // X86Aesdec
    instrinsic_info(X86Instruction::Aesdeclast, IntrinsicType::Binary), // X86Aesdeclast
    instrinsic_info(X86Instruction::Aesenc, IntrinsicType::Binary), // X86Aesenc
    instrinsic_info(X86Instruction::Aesenclast, IntrinsicType::Binary), // X86Aesenclast
    instrinsic_info(X86Instruction::Aesimc, IntrinsicType::Unary), // X86Aesimc
    instrinsic_info(X86Instruction::Andnpd, IntrinsicType::Binary), // X86Andnpd
    instrinsic_info(X86Instruction::Andnps, IntrinsicType::Binary), // X86Andnps
    instrinsic_info(X86Instruction::Andpd, IntrinsicType::Binary), // X86Andpd
    instrinsic_info(X86Instruction::Andps, IntrinsicType::Binary), // X86Andps
    instrinsic_info(X86Instruction::Blendvpd, IntrinsicType::Ternary), // X86Blendvpd
    instrinsic_info(X86Instruction::Blendvps, IntrinsicType::Ternary), // X86Blendvps
    instrinsic_info(X86Instruction::Cmppd, IntrinsicType::TernaryImm), // X86Cmppd
    instrinsic_info(X86Instruction::Cmpps, IntrinsicType::TernaryImm), // X86Cmpps
    instrinsic_info(X86Instruction::Cmpsd, IntrinsicType::TernaryImm), // X86Cmpsd
    instrinsic_info(X86Instruction::Cmpss, IntrinsicType::TernaryImm), // X86Cmpss
    instrinsic_info(X86Instruction::Comisd, IntrinsicType::Comis_), // X86Comisdeq
    instrinsic_info(X86Instruction::Comisd, IntrinsicType::Comis_), // X86Comisdge
    instrinsic_info(X86Instruction::Comisd, IntrinsicType::Comis_), // X86Comisdlt
    instrinsic_info(X86Instruction::Comiss, IntrinsicType::Comis_), // X86Comisseq
    instrinsic_info(X86Instruction::Comiss, IntrinsicType::Comis_), // X86Comissge
    instrinsic_info(X86Instruction::Comiss, IntrinsicType::Comis_), // X86Comisslt
    instrinsic_info(X86Instruction::Crc32, IntrinsicType::Crc32), // X86Crc32
    instrinsic_info(X86Instruction::Crc32_16, IntrinsicType::Crc32), // X86Crc32_16
    instrinsic_info(X86Instruction::Crc32_8, IntrinsicType::Crc32), // X86Crc32_8
    instrinsic_info(X86Instruction::Cvtdq2pd, IntrinsicType::Unary), // X86Cvtdq2pd
    instrinsic_info(X86Instruction::Cvtdq2ps, IntrinsicType::Unary), // X86Cvtdq2ps
    instrinsic_info(X86Instruction::Cvtpd2dq, IntrinsicType::Unary), // X86Cvtpd2dq
    instrinsic_info(X86Instruction::Cvtpd2ps, IntrinsicType::Unary), // X86Cvtpd2ps
    instrinsic_info(X86Instruction::Cvtps2dq, IntrinsicType::Unary), // X86Cvtps2dq
    instrinsic_info(X86Instruction::Cvtps2pd, IntrinsicType::Unary), // X86Cvtps2pd
    instrinsic_info(X86Instruction::Cvtsd2si, IntrinsicType::UnaryToGpr), // X86Cvtsd2si
    instrinsic_info(X86Instruction::Cvtsd2ss, IntrinsicType::Binary), // X86Cvtsd2ss
    instrinsic_info(X86Instruction::Cvtsi2sd, IntrinsicType::BinaryGpr), // X86Cvtsi2sd
    instrinsic_info(X86Instruction::Movd, IntrinsicType::UnaryToGpr), // X86Cvtsi2si
    instrinsic_info(X86Instruction::Cvtsi2ss, IntrinsicType::BinaryGpr), // X86Cvtsi2ss
    instrinsic_info(X86Instruction::Cvtss2sd, IntrinsicType::Binary), // X86Cvtss2sd
    instrinsic_info(X86Instruction::Cvtss2si, IntrinsicType::UnaryToGpr), // X86Cvtss2si
    instrinsic_info(X86Instruction::Divpd, IntrinsicType::Binary), // X86Divpd
    instrinsic_info(X86Instruction::Divps, IntrinsicType::Binary), // X86Divps
    instrinsic_info(X86Instruction::Divsd, IntrinsicType::Binary), // X86Divsd
    instrinsic_info(X86Instruction::Divss, IntrinsicType::Binary), // X86Divss
    instrinsic_info(X86Instruction::Gf2p8affineqb, IntrinsicType::TernaryImm), // X86Gf2p8affineqb
    instrinsic_info(X86Instruction::Haddpd, IntrinsicType::Binary), // X86Haddpd
    instrinsic_info(X86Instruction::Haddps, IntrinsicType::Binary), // X86Haddps
    instrinsic_info(X86Instruction::Insertps, IntrinsicType::TernaryImm), // X86Insertps
    instrinsic_info(X86Instruction::None, IntrinsicType::Mxcsr), // X86Ldmxcsr
    instrinsic_info(X86Instruction::Maxpd, IntrinsicType::Binary), // X86Maxpd
    instrinsic_info(X86Instruction::Maxps, IntrinsicType::Binary), // X86Maxps
    instrinsic_info(X86Instruction::Maxsd, IntrinsicType::Binary), // X86Maxsd
    instrinsic_info(X86Instruction::Maxss, IntrinsicType::Binary), // X86Maxss
    instrinsic_info(X86Instruction::Minpd, IntrinsicType::Binary), // X86Minpd
    instrinsic_info(X86Instruction::Minps, IntrinsicType::Binary), // X86Minps
    instrinsic_info(X86Instruction::Minsd, IntrinsicType::Binary), // X86Minsd
    instrinsic_info(X86Instruction::Minss, IntrinsicType::Binary), // X86Minss
    instrinsic_info(X86Instruction::Movhlps, IntrinsicType::Binary), // X86Movhlps
    instrinsic_info(X86Instruction::Movlhps, IntrinsicType::Binary), // X86Movlhps
    instrinsic_info(X86Instruction::Movss, IntrinsicType::Binary), // X86Movss
    instrinsic_info(X86Instruction::Mulpd, IntrinsicType::Binary), // X86Mulpd
    instrinsic_info(X86Instruction::Mulps, IntrinsicType::Binary), // X86Mulps
    instrinsic_info(X86Instruction::Mulsd, IntrinsicType::Binary), // X86Mulsd
    instrinsic_info(X86Instruction::Mulss, IntrinsicType::Binary), // X86Mulss
    instrinsic_info(X86Instruction::Paddb, IntrinsicType::Binary), // X86Paddb
    instrinsic_info(X86Instruction::Paddd, IntrinsicType::Binary), // X86Paddd
    instrinsic_info(X86Instruction::Paddq, IntrinsicType::Binary), // X86Paddq
    instrinsic_info(X86Instruction::Paddw, IntrinsicType::Binary), // X86Paddw
    instrinsic_info(X86Instruction::Palignr, IntrinsicType::TernaryImm), // X86Palignr
    instrinsic_info(X86Instruction::Pand, IntrinsicType::Binary), // X86Pand
    instrinsic_info(X86Instruction::Pandn, IntrinsicType::Binary), // X86Pandn
    instrinsic_info(X86Instruction::Pavgb, IntrinsicType::Binary), // X86Pavgb
    instrinsic_info(X86Instruction::Pavgw, IntrinsicType::Binary), // X86Pavgw
    instrinsic_info(X86Instruction::Pblendvb, IntrinsicType::Ternary), // X86Pblendvb
    instrinsic_info(X86Instruction::Pclmulqdq, IntrinsicType::TernaryImm), // X86Pclmulqdq
    instrinsic_info(X86Instruction::Pcmpeqb, IntrinsicType::Binary), // X86Pcmpeqb
    instrinsic_info(X86Instruction::Pcmpeqd, IntrinsicType::Binary), // X86Pcmpeqd
    instrinsic_info(X86Instruction::Pcmpeqq, IntrinsicType::Binary), // X86Pcmpeqq
    instrinsic_info(X86Instruction::Pcmpeqw, IntrinsicType::Binary), // X86Pcmpeqw
    instrinsic_info(X86Instruction::Pcmpgtb, IntrinsicType::Binary), // X86Pcmpgtb
    instrinsic_info(X86Instruction::Pcmpgtd, IntrinsicType::Binary), // X86Pcmpgtd
    instrinsic_info(X86Instruction::Pcmpgtq, IntrinsicType::Binary), // X86Pcmpgtq
    instrinsic_info(X86Instruction::Pcmpgtw, IntrinsicType::Binary), // X86Pcmpgtw
    instrinsic_info(X86Instruction::Pmaxsb, IntrinsicType::Binary), // X86Pmaxsb
    instrinsic_info(X86Instruction::Pmaxsd, IntrinsicType::Binary), // X86Pmaxsd
    instrinsic_info(X86Instruction::Pmaxsw, IntrinsicType::Binary), // X86Pmaxsw
    instrinsic_info(X86Instruction::Pmaxub, IntrinsicType::Binary), // X86Pmaxub
    instrinsic_info(X86Instruction::Pmaxud, IntrinsicType::Binary), // X86Pmaxud
    instrinsic_info(X86Instruction::Pmaxuw, IntrinsicType::Binary), // X86Pmaxuw
    instrinsic_info(X86Instruction::Pminsb, IntrinsicType::Binary), // X86Pminsb
    instrinsic_info(X86Instruction::Pminsd, IntrinsicType::Binary), // X86Pminsd
    instrinsic_info(X86Instruction::Pminsw, IntrinsicType::Binary), // X86Pminsw
    instrinsic_info(X86Instruction::Pminub, IntrinsicType::Binary), // X86Pminub
    instrinsic_info(X86Instruction::Pminud, IntrinsicType::Binary), // X86Pminud
    instrinsic_info(X86Instruction::Pminuw, IntrinsicType::Binary), // X86Pminuw
    instrinsic_info(X86Instruction::Pmovsxbw, IntrinsicType::Unary), // X86Pmovsxbw
    instrinsic_info(X86Instruction::Pmovsxdq, IntrinsicType::Unary), // X86Pmovsxdq
    instrinsic_info(X86Instruction::Pmovsxwd, IntrinsicType::Unary), // X86Pmovsxwd
    instrinsic_info(X86Instruction::Pmovzxbw, IntrinsicType::Unary), // X86Pmovzxbw
    instrinsic_info(X86Instruction::Pmovzxdq, IntrinsicType::Unary), // X86Pmovzxdq
    instrinsic_info(X86Instruction::Pmovzxwd, IntrinsicType::Unary), // X86Pmovzxwd
    instrinsic_info(X86Instruction::Pmulld, IntrinsicType::Binary), // X86Pmulld
    instrinsic_info(X86Instruction::Pmullw, IntrinsicType::Binary), // X86Pmullw
    instrinsic_info(X86Instruction::Popcnt, IntrinsicType::PopCount), // X86Popcnt
    instrinsic_info(X86Instruction::Por, IntrinsicType::Binary), // X86Por
    instrinsic_info(X86Instruction::Pshufb, IntrinsicType::Binary), // X86Pshufb
    instrinsic_info(X86Instruction::Pshufd, IntrinsicType::BinaryImm), // X86Pshufd
    instrinsic_info(X86Instruction::Pslld, IntrinsicType::Binary), // X86Pslld
    instrinsic_info(X86Instruction::Pslldq, IntrinsicType::Binary), // X86Pslldq
    instrinsic_info(X86Instruction::Psllq, IntrinsicType::Binary), // X86Psllq
    instrinsic_info(X86Instruction::Psllw, IntrinsicType::Binary), // X86Psllw
    instrinsic_info(X86Instruction::Psrad, IntrinsicType::Binary), // X86Psrad
    instrinsic_info(X86Instruction::Psraw, IntrinsicType::Binary), // X86Psraw
    instrinsic_info(X86Instruction::Psrld, IntrinsicType::Binary), // X86Psrld
    instrinsic_info(X86Instruction::Psrlq, IntrinsicType::Binary), // X86Psrlq
    instrinsic_info(X86Instruction::Psrldq, IntrinsicType::Binary), // X86Psrldq
    instrinsic_info(X86Instruction::Psrlw, IntrinsicType::Binary), // X86Psrlw
    instrinsic_info(X86Instruction::Psubb, IntrinsicType::Binary), // X86Psubb
    instrinsic_info(X86Instruction::Psubd, IntrinsicType::Binary), // X86Psubd
    instrinsic_info(X86Instruction::Psubq, IntrinsicType::Binary), // X86Psubq
    instrinsic_info(X86Instruction::Psubw, IntrinsicType::Binary), // X86Psubw
    instrinsic_info(X86Instruction::Punpckhbw, IntrinsicType::Binary), // X86Punpckhbw
    instrinsic_info(X86Instruction::Punpckhdq, IntrinsicType::Binary), // X86Punpckhdq
    instrinsic_info(X86Instruction::Punpckhqdq, IntrinsicType::Binary), // X86Punpckhqdq
    instrinsic_info(X86Instruction::Punpckhwd, IntrinsicType::Binary), // X86Punpckhwd
    instrinsic_info(X86Instruction::Punpcklbw, IntrinsicType::Binary), // X86Punpcklbw
    instrinsic_info(X86Instruction::Punpckldq, IntrinsicType::Binary), // X86Punpckldq
    instrinsic_info(X86Instruction::Punpcklqdq, IntrinsicType::Binary), // X86Punpcklqdq
    instrinsic_info(X86Instruction::Punpcklwd, IntrinsicType::Binary), // X86Punpcklwd
    instrinsic_info(X86Instruction::Pxor, IntrinsicType::Binary), // X86Pxor
    instrinsic_info(X86Instruction::Rcpps, IntrinsicType::Unary), // X86Rcpps
    instrinsic_info(X86Instruction::Rcpss, IntrinsicType::Unary), // X86Rcpss
    instrinsic_info(X86Instruction::Roundpd, IntrinsicType::BinaryImm), // X86Roundpd
    instrinsic_info(X86Instruction::Roundps, IntrinsicType::BinaryImm), // X86Roundps
    instrinsic_info(X86Instruction::Roundsd, IntrinsicType::BinaryImm), // X86Roundsd
    instrinsic_info(X86Instruction::Roundss, IntrinsicType::BinaryImm), // X86Roundss
    instrinsic_info(X86Instruction::Rsqrtps, IntrinsicType::Unary), // X86Rsqrtps
    instrinsic_info(X86Instruction::Rsqrtss, IntrinsicType::Unary), // X86Rsqrtss
    instrinsic_info(X86Instruction::Sha256Msg1, IntrinsicType::Binary), // X86Sha256Msg1
    instrinsic_info(X86Instruction::Sha256Msg2, IntrinsicType::Binary), // X86Sha256Msg2
    instrinsic_info(X86Instruction::Sha256Rnds2, IntrinsicType::Ternary), // X86Sha256Rnds2
    instrinsic_info(X86Instruction::Shufpd, IntrinsicType::TernaryImm), // X86Shufpd
    instrinsic_info(X86Instruction::Shufps, IntrinsicType::TernaryImm), // X86Shufps
    instrinsic_info(X86Instruction::Sqrtpd, IntrinsicType::Unary), // X86Sqrtpd
    instrinsic_info(X86Instruction::Sqrtps, IntrinsicType::Unary), // X86Sqrtps
    instrinsic_info(X86Instruction::Sqrtsd, IntrinsicType::Unary), // X86Sqrtsd
    instrinsic_info(X86Instruction::Sqrtss, IntrinsicType::Unary), // X86Sqrtss
    instrinsic_info(X86Instruction::None, IntrinsicType::Mxcsr), // X86Stmxcsr
    instrinsic_info(X86Instruction::Subpd, IntrinsicType::Binary), // X86Subpd
    instrinsic_info(X86Instruction::Subps, IntrinsicType::Binary), // X86Subps
    instrinsic_info(X86Instruction::Subsd, IntrinsicType::Binary), // X86Subsd
    instrinsic_info(X86Instruction::Subss, IntrinsicType::Binary), // X86Subss
    instrinsic_info(X86Instruction::Unpckhpd, IntrinsicType::Binary), // X86Unpckhpd
    instrinsic_info(X86Instruction::Unpckhps, IntrinsicType::Binary), // X86Unpckhps
    instrinsic_info(X86Instruction::Unpcklpd, IntrinsicType::Binary), // X86Unpcklpd
    instrinsic_info(X86Instruction::Unpcklps, IntrinsicType::Binary), // X86Unpcklps
    instrinsic_info(X86Instruction::Vcvtph2ps, IntrinsicType::Unary), // X86Vcvtph2ps
    instrinsic_info(X86Instruction::Vcvtps2ph, IntrinsicType::BinaryImm), // X86Vcvtps2ph
    instrinsic_info(X86Instruction::Vfmadd231pd, IntrinsicType::Fma), // X86Vfmadd231pd
    instrinsic_info(X86Instruction::Vfmadd231ps, IntrinsicType::Fma), // X86Vfmadd231ps
    instrinsic_info(X86Instruction::Vfmadd231sd, IntrinsicType::Fma), // X86Vfmadd231sd
    instrinsic_info(X86Instruction::Vfmadd231ss, IntrinsicType::Fma), // X86Vfmadd231ss
    instrinsic_info(X86Instruction::Vfmsub231sd, IntrinsicType::Fma), // X86Vfmsub231sd
    instrinsic_info(X86Instruction::Vfmsub231ss, IntrinsicType::Fma), // X86Vfmsub231ss
    instrinsic_info(X86Instruction::Vfnmadd231pd, IntrinsicType::Fma), // X86Vfnmadd231pd
    instrinsic_info(X86Instruction::Vfnmadd231ps, IntrinsicType::Fma), // X86Vfnmadd231ps
    instrinsic_info(X86Instruction::Vfnmadd231sd, IntrinsicType::Fma), // X86Vfnmadd231sd
    instrinsic_info(X86Instruction::Vfnmadd231ss, IntrinsicType::Fma), // X86Vfnmadd231ss
    instrinsic_info(X86Instruction::Vfnmsub231sd, IntrinsicType::Fma), // X86Vfnmsub231sd
    instrinsic_info(X86Instruction::Vfnmsub231ss, IntrinsicType::Fma), // X86Vfnmsub231ss
    instrinsic_info(X86Instruction::Vpternlogd, IntrinsicType::TernaryImm), // X86Vpternlogd
    instrinsic_info(X86Instruction::Xorpd, IntrinsicType::Binary), // X86Xorpd
    instrinsic_info(X86Instruction::Xorps, IntrinsicType::Binary), // X86Xorps
    instrinsic_info(X86Instruction::Cvttps2dq, IntrinsicType::Unary), // X86Cvttps2dq
    instrinsic_info(X86Instruction::Cvttsd2si, IntrinsicType::UnaryToGpr), // X86Cvttsd2si
    instrinsic_info(X86Instruction::Cvttss2si, IntrinsicType::UnaryToGpr), // X86Cvttss2si
    instrinsic_info(X86Instruction::Pabsb, IntrinsicType::Unary), // X86Pabsb
    instrinsic_info(X86Instruction::Pabsw, IntrinsicType::Unary), // X86Pabsw
    instrinsic_info(X86Instruction::Pabsd, IntrinsicType::Unary), // X86Pabsd
    instrinsic_info(X86Instruction::Paddsb, IntrinsicType::Binary), // X86Paddsb
    instrinsic_info(X86Instruction::Paddsw, IntrinsicType::Binary), // X86Paddsw
    instrinsic_info(X86Instruction::Paddusb, IntrinsicType::Binary), // X86Paddusb
    instrinsic_info(X86Instruction::Paddusw, IntrinsicType::Binary), // X86Paddusw
    instrinsic_info(X86Instruction::Psubsb, IntrinsicType::Binary), // X86Psubsb
    instrinsic_info(X86Instruction::Psubsw, IntrinsicType::Binary), // X86Psubsw
    instrinsic_info(X86Instruction::Psubusb, IntrinsicType::Binary), // X86Psubusb
    instrinsic_info(X86Instruction::Psubusw, IntrinsicType::Binary), // X86Psubusw
    instrinsic_info(X86Instruction::Pmovmskb, IntrinsicType::UnaryToGpr), // X86Pmovmskb
    instrinsic_info(X86Instruction::Psllvd, IntrinsicType::Binary), // X86Psllvd
    instrinsic_info(X86Instruction::Psrlvd, IntrinsicType::Binary), // X86Psrlvd
    instrinsic_info(X86Instruction::Psravd, IntrinsicType::Binary), // X86Psravd
];
