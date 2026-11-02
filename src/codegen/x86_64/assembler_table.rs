// Port of CodeGen/X86/AssemblerTable.cs.
// Entries follow the X86Instruction discriminant order; BadOp maps to BAD_OP.

use super::instruction::X86Instruction;

pub const BAD_OP: u32 = 0;

/// C# [Flags] enum InstructionFlags.
pub mod flags {
    pub const NONE: u32 = 0;
    pub const REG_ONLY: u32 = 1 << 0;
    pub const REG8_SRC: u32 = 1 << 1;
    pub const REG8_DEST: u32 = 1 << 2;
    pub const REX_W: u32 = 1 << 3;
    pub const VEX: u32 = 1 << 4;
    pub const EVEX: u32 = 1 << 5;

    pub const PREFIX_BIT: u32 = 16;
    pub const PREFIX_MASK: u32 = 7 << PREFIX_BIT;
    pub const PREFIX_66: u32 = 1 << PREFIX_BIT;
    pub const PREFIX_F3: u32 = 2 << PREFIX_BIT;
    pub const PREFIX_F2: u32 = 4 << PREFIX_BIT;
}

/// C# readonly struct InstructionInfo.
#[derive(Debug, Clone, Copy)]
pub struct InstructionInfo {
    pub op_rmr: u32,
    pub op_rm_imm8: u32,
    pub op_rm_imm32: u32,
    pub op_r_imm64: u32,
    pub op_rrm: u32,
    pub flags: u32,
}

const fn ii(
    op_rmr: u32,
    op_rm_imm8: u32,
    op_rm_imm32: u32,
    op_r_imm64: u32,
    op_rrm: u32,
    flags: u32,
) -> InstructionInfo {
    InstructionInfo {
        op_rmr,
        op_rm_imm8,
        op_rm_imm32,
        op_r_imm64,
        op_rrm,
        flags,
    }
}

/// C# Assembler._instTable. Indexed by X86Instruction discriminant.
#[rustfmt::skip]
pub static INST_TABLE: [InstructionInfo; X86Instruction::Count as usize] = [
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // None
    ii(0x00000001, 0x00000083, 0x00000081, BAD_OP, 0x00000003, flags::NONE), // Add
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f58, flags::VEX | flags::PREFIX_66), // Addpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f58, flags::VEX), // Addps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f58, flags::VEX | flags::PREFIX_F2), // Addsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f58, flags::VEX | flags::PREFIX_F3), // Addss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38de, flags::VEX | flags::PREFIX_66), // Aesdec
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38df, flags::VEX | flags::PREFIX_66), // Aesdeclast
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38dc, flags::VEX | flags::PREFIX_66), // Aesenc
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38dd, flags::VEX | flags::PREFIX_66), // Aesenclast
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38db, flags::VEX | flags::PREFIX_66), // Aesimc
    ii(0x00000021, 0x04000083, 0x04000081, BAD_OP, 0x00000023, flags::NONE), // And
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f55, flags::VEX | flags::PREFIX_66), // Andnpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f55, flags::VEX), // Andnps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f54, flags::VEX | flags::PREFIX_66), // Andpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f54, flags::VEX), // Andps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3815, flags::PREFIX_66), // Blendvpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3814, flags::PREFIX_66), // Blendvps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fbd, flags::NONE), // Bsr
    ii(0x00000fc8, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::REG_ONLY), // Bswap
    ii(0x020000ff, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Call
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f40, flags::NONE), // Cmovcc
    ii(0x00000039, 0x07000083, 0x07000081, BAD_OP, 0x0000003b, flags::NONE), // Cmp
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc2, flags::VEX | flags::PREFIX_66), // Cmppd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc2, flags::VEX), // Cmpps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc2, flags::VEX | flags::PREFIX_F2), // Cmpsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc2, flags::VEX | flags::PREFIX_F3), // Cmpss
    ii(0x00000fb1, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Cmpxchg
    ii(0x01000fc7, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::REX_W), // Cmpxchg16b
    ii(0x00000fb0, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::REG8_SRC), // Cmpxchg8
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2f, flags::VEX | flags::PREFIX_66), // Comisd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2f, flags::VEX), // Comiss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38f1, flags::PREFIX_F2), // Crc32
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38f1, flags::PREFIX_F2 | flags::PREFIX_66), // Crc32_16
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38f0, flags::PREFIX_F2 | flags::REG8_SRC), // Crc32_8
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fe6, flags::VEX | flags::PREFIX_F3), // Cvtdq2pd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5b, flags::VEX), // Cvtdq2ps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fe6, flags::VEX | flags::PREFIX_F2), // Cvtpd2dq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5a, flags::VEX | flags::PREFIX_66), // Cvtpd2ps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5b, flags::VEX | flags::PREFIX_66), // Cvtps2dq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5a, flags::VEX), // Cvtps2pd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2d, flags::VEX | flags::PREFIX_F2), // Cvtsd2si
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5a, flags::VEX | flags::PREFIX_F2), // Cvtsd2ss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2a, flags::VEX | flags::PREFIX_F2), // Cvtsi2sd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2a, flags::VEX | flags::PREFIX_F3), // Cvtsi2ss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5a, flags::VEX | flags::PREFIX_F3), // Cvtss2sd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2d, flags::VEX | flags::PREFIX_F3), // Cvtss2si
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x060000f7, flags::NONE), // Div
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5e, flags::VEX | flags::PREFIX_66), // Divpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5e, flags::VEX), // Divps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5e, flags::VEX | flags::PREFIX_F2), // Divsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5e, flags::VEX | flags::PREFIX_F3), // Divss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3ace, flags::PREFIX_66), // Gf2p8affineqb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f7c, flags::VEX | flags::PREFIX_66), // Haddpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f7c, flags::VEX | flags::PREFIX_F2), // Haddps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x070000f7, flags::NONE), // Idiv
    ii(BAD_OP, 0x0000006b, 0x00000069, BAD_OP, 0x00000faf, flags::NONE), // Imul
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x050000f7, flags::NONE), // Imul128
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a21, flags::VEX | flags::PREFIX_66), // Insertps
    ii(0x040000ff, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Jmp
    ii(0x02000fae, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::VEX), // Ldmxcsr
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x0000008d, flags::NONE), // Lea
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5f, flags::VEX | flags::PREFIX_66), // Maxpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5f, flags::VEX), // Maxps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5f, flags::VEX | flags::PREFIX_F2), // Maxsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5f, flags::VEX | flags::PREFIX_F3), // Maxss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5d, flags::VEX | flags::PREFIX_66), // Minpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5d, flags::VEX), // Minps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5d, flags::VEX | flags::PREFIX_F2), // Minsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5d, flags::VEX | flags::PREFIX_F3), // Minss
    ii(0x00000089, BAD_OP, 0x000000c7, 0x000000b8, 0x0000008b, flags::NONE), // Mov
    ii(0x00000089, BAD_OP, 0x000000c7, BAD_OP, 0x0000008b, flags::PREFIX_66), // Mov16
    ii(0x00000088, 0x000000c6, BAD_OP, BAD_OP, 0x0000008a, flags::REG8_SRC | flags::REG8_DEST), // Mov8
    ii(0x00000f7e, BAD_OP, BAD_OP, BAD_OP, 0x00000f6e, flags::VEX | flags::PREFIX_66), // Movd
    ii(0x00000f7f, BAD_OP, BAD_OP, BAD_OP, 0x00000f6f, flags::VEX | flags::PREFIX_F3), // Movdqu
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f12, flags::VEX), // Movhlps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f16, flags::VEX), // Movlhps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f7e, flags::VEX | flags::PREFIX_F3), // Movq
    ii(0x00000f11, BAD_OP, BAD_OP, BAD_OP, 0x00000f10, flags::VEX | flags::PREFIX_F2), // Movsd
    ii(0x00000f11, BAD_OP, BAD_OP, BAD_OP, 0x00000f10, flags::VEX | flags::PREFIX_F3), // Movss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fbf, flags::NONE), // Movsx16
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000063, flags::NONE), // Movsx32
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fbe, flags::REG8_SRC), // Movsx8
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fb7, flags::NONE), // Movzx16
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fb6, flags::REG8_SRC), // Movzx8
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x040000f7, flags::NONE), // Mul128
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f59, flags::VEX | flags::PREFIX_66), // Mulpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f59, flags::VEX), // Mulps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f59, flags::VEX | flags::PREFIX_F2), // Mulsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f59, flags::VEX | flags::PREFIX_F3), // Mulss
    ii(0x030000f7, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Neg
    ii(0x020000f7, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Not
    ii(0x00000009, 0x01000083, 0x01000081, BAD_OP, 0x0000000b, flags::NONE), // Or
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000ffc, flags::VEX | flags::PREFIX_66), // Paddb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000ffe, flags::VEX | flags::PREFIX_66), // Paddd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fd4, flags::VEX | flags::PREFIX_66), // Paddq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000ffd, flags::VEX | flags::PREFIX_66), // Paddw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a0f, flags::VEX | flags::PREFIX_66), // Palignr
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fdb, flags::VEX | flags::PREFIX_66), // Pand
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fdf, flags::VEX | flags::PREFIX_66), // Pandn
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fe0, flags::VEX | flags::PREFIX_66), // Pavgb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fe3, flags::VEX | flags::PREFIX_66), // Pavgw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3810, flags::PREFIX_66), // Pblendvb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a44, flags::VEX | flags::PREFIX_66), // Pclmulqdq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f74, flags::VEX | flags::PREFIX_66), // Pcmpeqb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f76, flags::VEX | flags::PREFIX_66), // Pcmpeqd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3829, flags::VEX | flags::PREFIX_66), // Pcmpeqq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f75, flags::VEX | flags::PREFIX_66), // Pcmpeqw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f64, flags::VEX | flags::PREFIX_66), // Pcmpgtb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f66, flags::VEX | flags::PREFIX_66), // Pcmpgtd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3837, flags::VEX | flags::PREFIX_66), // Pcmpgtq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f65, flags::VEX | flags::PREFIX_66), // Pcmpgtw
    ii(0x000f3a14, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::VEX | flags::PREFIX_66), // Pextrb
    ii(0x000f3a16, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::VEX | flags::PREFIX_66), // Pextrd
    ii(0x000f3a16, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::VEX | flags::REX_W | flags::PREFIX_66), // Pextrq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc5, flags::VEX | flags::PREFIX_66), // Pextrw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a20, flags::VEX | flags::PREFIX_66), // Pinsrb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a22, flags::VEX | flags::PREFIX_66), // Pinsrd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a22, flags::VEX | flags::REX_W | flags::PREFIX_66), // Pinsrq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc4, flags::VEX | flags::PREFIX_66), // Pinsrw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f383c, flags::VEX | flags::PREFIX_66), // Pmaxsb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f383d, flags::VEX | flags::PREFIX_66), // Pmaxsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fee, flags::VEX | flags::PREFIX_66), // Pmaxsw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fde, flags::VEX | flags::PREFIX_66), // Pmaxub
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f383f, flags::VEX | flags::PREFIX_66), // Pmaxud
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f383e, flags::VEX | flags::PREFIX_66), // Pmaxuw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3838, flags::VEX | flags::PREFIX_66), // Pminsb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3839, flags::VEX | flags::PREFIX_66), // Pminsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fea, flags::VEX | flags::PREFIX_66), // Pminsw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fda, flags::VEX | flags::PREFIX_66), // Pminub
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f383b, flags::VEX | flags::PREFIX_66), // Pminud
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f383a, flags::VEX | flags::PREFIX_66), // Pminuw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3820, flags::VEX | flags::PREFIX_66), // Pmovsxbw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3825, flags::VEX | flags::PREFIX_66), // Pmovsxdq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3823, flags::VEX | flags::PREFIX_66), // Pmovsxwd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3830, flags::VEX | flags::PREFIX_66), // Pmovzxbw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3835, flags::VEX | flags::PREFIX_66), // Pmovzxdq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3833, flags::VEX | flags::PREFIX_66), // Pmovzxwd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3840, flags::VEX | flags::PREFIX_66), // Pmulld
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fd5, flags::VEX | flags::PREFIX_66), // Pmullw
    ii(0x0000008f, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Pop
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fb8, flags::PREFIX_F3), // Popcnt
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000feb, flags::VEX | flags::PREFIX_66), // Por
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3800, flags::VEX | flags::PREFIX_66), // Pshufb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f70, flags::VEX | flags::PREFIX_66), // Pshufd
    ii(BAD_OP, 0x06000f72, BAD_OP, BAD_OP, 0x00000ff2, flags::VEX | flags::PREFIX_66), // Pslld
    ii(BAD_OP, 0x07000f73, BAD_OP, BAD_OP, BAD_OP, flags::VEX | flags::PREFIX_66), // Pslldq
    ii(BAD_OP, 0x06000f73, BAD_OP, BAD_OP, 0x00000ff3, flags::VEX | flags::PREFIX_66), // Psllq
    ii(BAD_OP, 0x06000f71, BAD_OP, BAD_OP, 0x00000ff1, flags::VEX | flags::PREFIX_66), // Psllw
    ii(BAD_OP, 0x04000f72, BAD_OP, BAD_OP, 0x00000fe2, flags::VEX | flags::PREFIX_66), // Psrad
    ii(BAD_OP, 0x04000f71, BAD_OP, BAD_OP, 0x00000fe1, flags::VEX | flags::PREFIX_66), // Psraw
    ii(BAD_OP, 0x02000f72, BAD_OP, BAD_OP, 0x00000fd2, flags::VEX | flags::PREFIX_66), // Psrld
    ii(BAD_OP, 0x02000f73, BAD_OP, BAD_OP, 0x00000fd3, flags::VEX | flags::PREFIX_66), // Psrlq
    ii(BAD_OP, 0x03000f73, BAD_OP, BAD_OP, BAD_OP, flags::VEX | flags::PREFIX_66), // Psrldq
    ii(BAD_OP, 0x02000f71, BAD_OP, BAD_OP, 0x00000fd1, flags::VEX | flags::PREFIX_66), // Psrlw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000ff8, flags::VEX | flags::PREFIX_66), // Psubb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000ffa, flags::VEX | flags::PREFIX_66), // Psubd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000ffb, flags::VEX | flags::PREFIX_66), // Psubq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000ff9, flags::VEX | flags::PREFIX_66), // Psubw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f68, flags::VEX | flags::PREFIX_66), // Punpckhbw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f6a, flags::VEX | flags::PREFIX_66), // Punpckhdq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f6d, flags::VEX | flags::PREFIX_66), // Punpckhqdq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f69, flags::VEX | flags::PREFIX_66), // Punpckhwd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f60, flags::VEX | flags::PREFIX_66), // Punpcklbw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f62, flags::VEX | flags::PREFIX_66), // Punpckldq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f6c, flags::VEX | flags::PREFIX_66), // Punpcklqdq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f61, flags::VEX | flags::PREFIX_66), // Punpcklwd
    ii(BAD_OP, 0x0000006a, 0x00000068, BAD_OP, 0x060000ff, flags::NONE), // Push
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fef, flags::VEX | flags::PREFIX_66), // Pxor
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f53, flags::VEX), // Rcpps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f53, flags::VEX | flags::PREFIX_F3), // Rcpss
    ii(0x010000d3, 0x010000c1, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Ror
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a09, flags::VEX | flags::PREFIX_66), // Roundpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a08, flags::VEX | flags::PREFIX_66), // Roundps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a0b, flags::VEX | flags::PREFIX_66), // Roundsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a0a, flags::VEX | flags::PREFIX_66), // Roundss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f52, flags::VEX), // Rsqrtps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f52, flags::VEX | flags::PREFIX_F3), // Rsqrtss
    ii(0x070000d3, 0x070000c1, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Sar
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f90, flags::REG8_DEST), // Setcc
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38cc, flags::NONE), // Sha256Msg1
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38cd, flags::NONE), // Sha256Msg2
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38cb, flags::NONE), // Sha256Rnds2
    ii(0x040000d3, 0x040000c1, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Shl
    ii(0x050000d3, 0x050000c1, BAD_OP, BAD_OP, BAD_OP, flags::NONE), // Shr
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc6, flags::VEX | flags::PREFIX_66), // Shufpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fc6, flags::VEX), // Shufps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f51, flags::VEX | flags::PREFIX_66), // Sqrtpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f51, flags::VEX), // Sqrtps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f51, flags::VEX | flags::PREFIX_F2), // Sqrtsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f51, flags::VEX | flags::PREFIX_F3), // Sqrtss
    ii(0x03000fae, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::VEX), // Stmxcsr
    ii(0x00000029, 0x05000083, 0x05000081, BAD_OP, 0x0000002b, flags::NONE), // Sub
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5c, flags::VEX | flags::PREFIX_66), // Subpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5c, flags::VEX), // Subps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5c, flags::VEX | flags::PREFIX_F2), // Subsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5c, flags::VEX | flags::PREFIX_F3), // Subss
    ii(0x00000085, BAD_OP, 0x000000f7, BAD_OP, BAD_OP, flags::NONE), // Test
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f15, flags::VEX | flags::PREFIX_66), // Unpckhpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f15, flags::VEX), // Unpckhps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f14, flags::VEX | flags::PREFIX_66), // Unpcklpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f14, flags::VEX), // Unpcklps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a4b, flags::VEX | flags::PREFIX_66), // Vblendvpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a4a, flags::VEX | flags::PREFIX_66), // Vblendvps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3813, flags::VEX | flags::PREFIX_66), // Vcvtph2ps
    ii(0x000f3a1d, BAD_OP, BAD_OP, BAD_OP, BAD_OP, flags::VEX | flags::PREFIX_66), // Vcvtps2ph
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38b8, flags::VEX | flags::PREFIX_66 | flags::REX_W), // Vfmadd231pd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38b8, flags::VEX | flags::PREFIX_66), // Vfmadd231ps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38b9, flags::VEX | flags::PREFIX_66 | flags::REX_W), // Vfmadd231sd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38b9, flags::VEX | flags::PREFIX_66), // Vfmadd231ss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bb, flags::VEX | flags::PREFIX_66 | flags::REX_W), // Vfmsub231sd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bb, flags::VEX | flags::PREFIX_66), // Vfmsub231ss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bc, flags::VEX | flags::PREFIX_66 | flags::REX_W), // Vfnmadd231pd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bc, flags::VEX | flags::PREFIX_66), // Vfnmadd231ps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bd, flags::VEX | flags::PREFIX_66 | flags::REX_W), // Vfnmadd231sd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bd, flags::VEX | flags::PREFIX_66), // Vfnmadd231ss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bf, flags::VEX | flags::PREFIX_66 | flags::REX_W), // Vfnmsub231sd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38bf, flags::VEX | flags::PREFIX_66), // Vfnmsub231ss
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a4c, flags::VEX | flags::PREFIX_66), // Vpblendvb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3a25, flags::EVEX | flags::PREFIX_66), // Vpternlogd
    ii(0x00000031, 0x06000083, 0x06000081, BAD_OP, 0x00000033, flags::NONE), // Xor
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f57, flags::VEX | flags::PREFIX_66), // Xorpd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f57, flags::VEX), // Xorps
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f5b, flags::VEX | flags::PREFIX_F3), // Cvttps2dq
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2c, flags::VEX | flags::PREFIX_F2), // Cvttsd2si
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000f2c, flags::VEX | flags::PREFIX_F3), // Cvttss2si
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f381c, flags::VEX | flags::PREFIX_66), // Pabsb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f381d, flags::VEX | flags::PREFIX_66), // Pabsw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f381e, flags::VEX | flags::PREFIX_66), // Pabsd
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fec, flags::VEX | flags::PREFIX_66), // Paddsb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fed, flags::VEX | flags::PREFIX_66), // Paddsw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fdc, flags::VEX | flags::PREFIX_66), // Paddusb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fdd, flags::VEX | flags::PREFIX_66), // Paddusw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fe8, flags::VEX | flags::PREFIX_66), // Psubsb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fe9, flags::VEX | flags::PREFIX_66), // Psubsw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fd8, flags::VEX | flags::PREFIX_66), // Psubusb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fd9, flags::VEX | flags::PREFIX_66), // Psubusw
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fd7, flags::VEX | flags::PREFIX_66), // Pmovmskb
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3847, flags::VEX | flags::PREFIX_66), // Psllvd (AVX2)
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3845, flags::VEX | flags::PREFIX_66), // Psrlvd (AVX2)
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f3846, flags::VEX | flags::PREFIX_66), // Psravd (AVX2)
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x00000fbd, flags::PREFIX_F3), // Lzcnt (F3 0F BD, = BSR + F3 prefix)
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38f2, flags::VEX), // Andn (VEX.0F38.W F2 /r, 3-op GPR)
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38f7, flags::VEX | flags::PREFIX_66), // Shlx (66.0F38.W F7 /r)
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38f7, flags::VEX | flags::PREFIX_F2), // Shrx (F2.0F38.W F7 /r)
    ii(BAD_OP, BAD_OP, BAD_OP, BAD_OP, 0x000f38f7, flags::VEX | flags::PREFIX_F3), // Sarx (F3.0F38.W F7 /r)
];

/// C# Assembler.SupportsVexPrefix.
pub fn supports_vex_prefix(inst: X86Instruction) -> bool {
    INST_TABLE[inst as usize].flags & flags::VEX != 0
}
