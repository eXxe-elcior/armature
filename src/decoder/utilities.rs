use crate::decoder::instruction_name::InstructionName;
use crate::decoder::opcode_kind::OpCodeKind;

// Primitive enums
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RegisterSize {
    Int32,
    Int64,
    Simd64,
    Simd128,
}

impl RegisterSize {
    pub fn bits(self) -> u32 {
        match self {
            RegisterSize::Int32 => 32,
            RegisterSize::Int64 => 64,
            RegisterSize::Simd64 => 64,
            RegisterSize::Simd128 => 128,
        }
    }
    pub fn is_64bit(self) -> bool {
        matches!(
            self,
            RegisterSize::Int64 | RegisterSize::Simd64 | RegisterSize::Simd128
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Condition {
    Eq = 0,
    Ne,
    GeUn,
    LtUn,
    Mi,
    Pl,
    Vs,
    Vc,
    GtUn,
    LeUn,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}

enum_from_bits!(
    Condition,
    0xf,
    [
        Eq, Ne, GeUn, LtUn, Mi, Pl, Vs, Vc, GtUn, LeUn, Ge, Lt, Gt, Le, Al, Nv,
    ]
);

impl Condition {
    pub fn invert(self) -> Self {
        Self::from_raw((self as u32) ^ 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShiftType {
    Lsl = 0,
    Lsr,
    Asr,
    Ror,
}

enum_from_bits!(ShiftType, 0x3, [Lsl, Lsr, Asr, Ror]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntegerType {
    Uxtb = 0,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

enum_from_bits!(
    IntegerType,
    0x7,
    [Uxtb, Uxth, Uxtw, Uxtx, Sxtb, Sxth, Sxtw, Sxtx]
);

/// Matches C# DataOp: Adr=0, Arithmetic=1, Logical=2, BitField=3.
/// C# field extraction: (raw >> 24) & 0x3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataOperation {
    Adr = 0,
    Arithmetic = 1,
    Logical = 2,
    BitField = 3,
}

enum_from_bits!(DataOperation, 0x3, [Adr, Arithmetic, Logical, BitField]);

// Decoder types
pub type EmitFn = fn(&mut crate::instructions::ArmEmitterCtx);

#[derive(Clone, Copy)]
pub struct InstructionInfo {
    pub name: InstructionName,
    pub emit: EmitFn,
}

impl InstructionInfo {
    pub const UNDEFINED: InstructionInfo = InstructionInfo {
        name: InstructionName::Und,
        emit: emit_undefined,
    };
}

pub(crate) fn emit_undefined(_ctx: &mut crate::instructions::ArmEmitterCtx) {}

#[derive(Clone, Copy)]
pub struct DecodedInstruction {
    pub address: u64,
    pub raw: u32,
    pub info: InstructionInfo,
    pub reg_size: RegisterSize,
    pub kind: OpCodeKind,
}

impl DecodedInstruction {
    pub fn bits(&self) -> u32 {
        self.reg_size.bits()
    }
    pub fn bytes(&self) -> u32 {
        self.bits() / 8
    }
    pub fn pairs(&self) -> u32 {
        self.bits() / 16
    }
}

// Decode error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Undefined(u32),
    Unimplemented(u32),
}

impl core::fmt::Debug for InstructionInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InstInfo")
            .field("name", &self.name)
            .field("emit", &(self.emit as usize))
            .finish()
    }
}

impl PartialEq for InstructionInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for InstructionInfo {}
