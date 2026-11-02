// Port of CodeGen/X86/X86Condition.cs.

use crate::ir::types::Comparison;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum X86Condition {
    Overflow = 0x0,
    NotOverflow = 0x1,
    Below = 0x2,
    AboveOrEqual = 0x3,
    Equal = 0x4,
    NotEqual = 0x5,
    BelowOrEqual = 0x6,
    Above = 0x7,
    Sign = 0x8,
    NotSign = 0x9,
    ParityEven = 0xa,
    ParityOdd = 0xb,
    Less = 0xc,
    GreaterOrEqual = 0xd,
    LessOrEqual = 0xe,
    Greater = 0xf,
}

// C# ComparisonX86Extensions.ToX86Condition.
pub fn to_x86_condition(comp: Comparison) -> X86Condition {
    match comp {
        Comparison::Equal => X86Condition::Equal,
        Comparison::NotEqual => X86Condition::NotEqual,
        Comparison::Greater => X86Condition::Greater,
        Comparison::LessOrEqual => X86Condition::LessOrEqual,
        Comparison::GreaterUI => X86Condition::Above,
        Comparison::LessOrEqualUI => X86Condition::BelowOrEqual,
        Comparison::GreaterOrEqual => X86Condition::GreaterOrEqual,
        Comparison::Less => X86Condition::Less,
        Comparison::GreaterOrEqualUI => X86Condition::AboveOrEqual,
        Comparison::LessUI => X86Condition::Below,
    }
}
