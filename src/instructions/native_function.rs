use crate::instructions::native_interface;
use crate::ir::types::OperandType;

// Registry (C# Translation/Delegates.cs)
//
// One variant per registered native helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum NativeFn {
    Break,
    SupervisorCall,
    Undefined,
    GetCtrEl0,
    GetDczidEl0,
    GetCntfrqEl0,
    GetCntpctEl0,
    GetCntvctEl0,
    ReadByte,
    ReadUInt16,
    ReadUInt32,
    ReadUInt64,
    ReadVector128,
    WriteByte,
    WriteUInt16,
    WriteUInt32,
    WriteUInt64,
    WriteVector128,
    EnqueueForRejit,
    DbgTranslate,
    SignalMemoryTracking,
    ThrowInvalidMemoryAccess,
    GetFunctionAddress,
    InvalidateCacheLine,
    CheckSynchronization,
    // SoftFallback.cs "Crc32" region.
    Crc32b,
    Crc32h,
    Crc32w,
    Crc32x,
    Crc32cb,
    Crc32ch,
    Crc32cw,
    Crc32cx,
    // SoftFallback.cs "ShrImm64" region.
    SignedShrImm64,
    UnsignedShrImm64,
    // SoftFallback.cs "Count" region.
    CountLeadingSigns,
    CountLeadingZeros,
    // SoftFloat32.
    SoftF32FPAdd,
    SoftF32FPSub,
    SoftF32FPMul,
    SoftF32FPDiv,
    SoftF32FPSqrt,
    SoftF32FPMax,
    SoftF32FPMin,
    SoftF32FPMulX,
    SoftF32FPRecipEstimate,
    SoftF32FPRecpX,
    SoftF32FPRSqrtEstimate,
    SoftF32FPCompare,
    SoftF32FPCompareEQ,
    SoftF32FPCompareGE,
    SoftF32FPCompareGT,
    SoftF32FPCompareLE,
    SoftF32FPCompareLT,
    SoftF32FPRecipStep,
    SoftF32FPRSqrtStep,
    SoftF32FPRecipStepFused,
    SoftF32FPRSqrtStepFused,
    // SoftFloat64.
    SoftF64FPAdd,
    SoftF64FPSub,
    SoftF64FPMul,
    SoftF64FPDiv,
    SoftF64FPSqrt,
    SoftF64FPMax,
    SoftF64FPMin,
    SoftF64FPMulX,
    SoftF64FPRecipEstimate,
    SoftF64FPRecpX,
    SoftF64FPRSqrtEstimate,
    SoftF64FPCompare,
    SoftF64FPCompareEQ,
    SoftF64FPCompareGE,
    SoftF64FPCompareGT,
    SoftF64FPCompareLE,
    SoftF64FPCompareLT,
    // SoftFloat64 step functions.
    SoftF64FPRecipStep,
    SoftF64FPRSqrtStep,
    SoftF64FPRecipStepFused,
    SoftF64FPRSqrtStepFused,
    // SoftFloat32 fused multiply-add family.
    SoftF32FPMulAdd,
    SoftF32FPMulSub,
    SoftF32FPNegMulAdd,
    SoftF32FPNegMulSub,
    // SoftFloat64 fused multiply-add family.
    SoftF64FPMulAdd,
    SoftF64FPMulSub,
    SoftF64FPNegMulAdd,
    SoftF64FPNegMulSub,
    // Abs helpers (used by Facge/Facgt soft fallback).
    SoftF32Abs,
    SoftF64Abs,
    // MathHelper / MathHelperF (C# Instructions/MathHelper.cs), used by
    // EmitRoundByRMode (Frinti/Frintx).
    MathF32Ceiling,
    MathF32Floor,
    MathF32Round,
    MathF32Truncate,
    MathF64Ceiling,
    MathF64Floor,
    MathF64Round,
    MathF64Truncate,
    // SoftFallback.cs "Sha1" region (software SHA1 implementation).
    HashChoose,
    FixedRotate,
    HashMajority,
    HashParity,
    Sha1SchedulePart1,
    Sha1SchedulePart2,
}

impl NativeFn {
    /// Host address of the helper (C#: MethodHandle.GetFunctionPointer()).
    pub fn ptr(self) -> *const u8 {
        match self {
            Self::Break => native_interface::break_ as *const u8,
            Self::SupervisorCall => native_interface::supervisor_call as *const u8,
            Self::Undefined => native_interface::undefined as *const u8,
            Self::GetCtrEl0 => native_interface::get_ctr_el0 as *const u8,
            Self::GetDczidEl0 => native_interface::get_dczid_el0 as *const u8,
            Self::GetCntfrqEl0 => native_interface::get_cntfrq_el0 as *const u8,
            Self::GetCntpctEl0 => native_interface::get_cntpct_el0 as *const u8,
            Self::GetCntvctEl0 => native_interface::get_cntvct_el0 as *const u8,
            Self::ReadByte => native_interface::read_byte as *const u8,
            Self::ReadUInt16 => native_interface::read_uint16 as *const u8,
            Self::ReadUInt32 => native_interface::read_uint32 as *const u8,
            Self::ReadUInt64 => native_interface::read_uint64 as *const u8,
            Self::ReadVector128 => native_interface::read_vector128 as *const u8,
            Self::WriteByte => native_interface::write_byte as *const u8,
            Self::WriteUInt16 => native_interface::write_uint16 as *const u8,
            Self::WriteUInt32 => native_interface::write_uint32 as *const u8,
            Self::WriteUInt64 => native_interface::write_uint64 as *const u8,
            Self::WriteVector128 => native_interface::write_vector128 as *const u8,
            Self::EnqueueForRejit => native_interface::enqueue_for_rejit as *const u8,
            Self::DbgTranslate => native_interface::dbg_translate as *const u8,
            Self::SignalMemoryTracking => native_interface::signal_memory_tracking as *const u8,
            Self::ThrowInvalidMemoryAccess => {
                native_interface::throw_invalid_memory_access as *const u8
            }
            Self::GetFunctionAddress => native_interface::get_function_address as *const u8,
            Self::InvalidateCacheLine => native_interface::invalidate_cache_line as *const u8,
            Self::CheckSynchronization => native_interface::check_synchronization as *const u8,
            Self::Crc32b => crate::instructions::soft_fallback::crc32b as *const u8,
            Self::Crc32h => crate::instructions::soft_fallback::crc32h as *const u8,
            Self::Crc32w => crate::instructions::soft_fallback::crc32w as *const u8,
            Self::Crc32x => crate::instructions::soft_fallback::crc32x as *const u8,
            Self::Crc32cb => crate::instructions::soft_fallback::crc32cb as *const u8,
            Self::Crc32ch => crate::instructions::soft_fallback::crc32ch as *const u8,
            Self::Crc32cw => crate::instructions::soft_fallback::crc32cw as *const u8,
            Self::Crc32cx => crate::instructions::soft_fallback::crc32cx as *const u8,
            Self::SignedShrImm64 => {
                crate::instructions::soft_fallback::signed_shr_imm64 as *const u8
            }
            Self::UnsignedShrImm64 => {
                crate::instructions::soft_fallback::unsigned_shr_imm64 as *const u8
            }
            Self::CountLeadingSigns => {
                crate::instructions::soft_fallback::count_leading_signs as *const u8
            }
            Self::CountLeadingZeros => {
                crate::instructions::soft_fallback::count_leading_zeros as *const u8
            }
            // SoftFloat32.
            Self::SoftF32FPAdd => crate::instructions::soft_float::sf32_fpadd as *const u8,
            Self::SoftF32FPSub => crate::instructions::soft_float::sf32_fpsub as *const u8,
            Self::SoftF32FPMul => crate::instructions::soft_float::sf32_fpmul as *const u8,
            Self::SoftF32FPDiv => crate::instructions::soft_float::sf32_fpdiv as *const u8,
            Self::SoftF32FPSqrt => crate::instructions::soft_float::sf32_fpsqrt as *const u8,
            Self::SoftF32FPMax => crate::instructions::soft_float::sf32_fpmax as *const u8,
            Self::SoftF32FPMin => crate::instructions::soft_float::sf32_fpmin as *const u8,
            Self::SoftF32FPMulX => crate::instructions::soft_float::sf32_fpmulx as *const u8,
            Self::SoftF32FPRecipEstimate => {
                crate::instructions::soft_float::sf32_fprecipestimate as *const u8
            }
            Self::SoftF32FPRecpX => crate::instructions::soft_float::sf32_fprecpx as *const u8,
            Self::SoftF32FPRSqrtEstimate => {
                crate::instructions::soft_float::sf32_fprsqtestimate as *const u8
            }
            Self::SoftF32FPCompare => crate::instructions::soft_float::sf32_fpcompare as *const u8,
            Self::SoftF32FPCompareEQ => {
                crate::instructions::soft_float::sf32_fpcompareeq as *const u8
            }
            Self::SoftF32FPCompareGE => {
                crate::instructions::soft_float::sf32_fpcomparege as *const u8
            }
            Self::SoftF32FPCompareGT => {
                crate::instructions::soft_float::sf32_fpcomparegt as *const u8
            }
            Self::SoftF32FPCompareLE => {
                crate::instructions::soft_float::sf32_fpcomparele as *const u8
            }
            Self::SoftF32FPCompareLT => {
                crate::instructions::soft_float::sf32_fpcomparelt as *const u8
            }
            Self::SoftF32FPRecipStep => {
                crate::instructions::soft_float::sf32_fprecipstep as *const u8
            }
            Self::SoftF32FPRSqrtStep => {
                crate::instructions::soft_float::sf32_fprsqrtstep as *const u8
            }
            Self::SoftF32FPRecipStepFused => {
                crate::instructions::soft_float::sf32_fprecipstepfused as *const u8
            }
            Self::SoftF32FPRSqrtStepFused => {
                crate::instructions::soft_float::sf32_fprsqrtstepfused as *const u8
            }
            // SoftFloat64.
            Self::SoftF64FPAdd => crate::instructions::soft_float::sf64_fpadd as *const u8,
            Self::SoftF64FPSub => crate::instructions::soft_float::sf64_fpsub as *const u8,
            Self::SoftF64FPMul => crate::instructions::soft_float::sf64_fpmul as *const u8,
            Self::SoftF64FPDiv => crate::instructions::soft_float::sf64_fpdiv as *const u8,
            Self::SoftF64FPSqrt => crate::instructions::soft_float::sf64_fpsqrt as *const u8,
            Self::SoftF64FPMax => crate::instructions::soft_float::sf64_fpmax as *const u8,
            Self::SoftF64FPMin => crate::instructions::soft_float::sf64_fpmin as *const u8,
            Self::SoftF64FPMulX => crate::instructions::soft_float::sf64_fpmulx as *const u8,
            Self::SoftF64FPRecipEstimate => {
                crate::instructions::soft_float::sf64_fprecipestimate as *const u8
            }
            Self::SoftF64FPRecpX => crate::instructions::soft_float::sf64_fprecpx as *const u8,
            Self::SoftF64FPRSqrtEstimate => {
                crate::instructions::soft_float::sf64_fprsqtestimate as *const u8
            }
            Self::SoftF64FPCompare => crate::instructions::soft_float::sf64_fpcompare as *const u8,
            Self::SoftF64FPCompareEQ => {
                crate::instructions::soft_float::sf64_fpcompareeq as *const u8
            }
            Self::SoftF64FPCompareGE => {
                crate::instructions::soft_float::sf64_fpcomparege as *const u8
            }
            Self::SoftF64FPCompareGT => {
                crate::instructions::soft_float::sf64_fpcomparegt as *const u8
            }
            Self::SoftF64FPCompareLE => {
                crate::instructions::soft_float::sf64_fpcomparele as *const u8
            }
            Self::SoftF64FPCompareLT => {
                crate::instructions::soft_float::sf64_fpcomparelt as *const u8
            }
            // SoftFloat64 step functions.
            Self::SoftF64FPRecipStep => {
                crate::instructions::soft_float::sf64_fprecipstep as *const u8
            }
            Self::SoftF64FPRSqrtStep => {
                crate::instructions::soft_float::sf64_fprsqrtstep as *const u8
            }
            Self::SoftF64FPRecipStepFused => {
                crate::instructions::soft_float::sf64_fprecipstepfused as *const u8
            }
            Self::SoftF64FPRSqrtStepFused => {
                crate::instructions::soft_float::sf64_fprsqrtstepfused as *const u8
            }
            // SoftFloat32 fused multiply-add.
            Self::SoftF32FPMulAdd => crate::instructions::soft_float::sf32_fpmuladd as *const u8,
            Self::SoftF32FPMulSub => crate::instructions::soft_float::sf32_fpmulsub as *const u8,
            Self::SoftF32FPNegMulAdd => {
                crate::instructions::soft_float::sf32_fpnegmuladd as *const u8
            }
            Self::SoftF32FPNegMulSub => {
                crate::instructions::soft_float::sf32_fpnegmulsub as *const u8
            }
            // SoftFloat64 fused multiply-add.
            Self::SoftF64FPMulAdd => crate::instructions::soft_float::sf64_fpmuladd as *const u8,
            Self::SoftF64FPMulSub => crate::instructions::soft_float::sf64_fpmulsub as *const u8,
            Self::SoftF64FPNegMulAdd => {
                crate::instructions::soft_float::sf64_fpnegmuladd as *const u8
            }
            Self::SoftF64FPNegMulSub => {
                crate::instructions::soft_float::sf64_fpnegmulsub as *const u8
            }
            // Abs helpers.
            Self::SoftF32Abs => crate::instructions::soft_float::sf32_abs as *const u8,
            Self::SoftF64Abs => crate::instructions::soft_float::sf64_abs as *const u8,
            // MathHelper / MathHelperF.
            Self::MathF32Ceiling => crate::instructions::soft_float::mh32_ceiling as *const u8,
            Self::MathF32Floor => crate::instructions::soft_float::mh32_floor as *const u8,
            Self::MathF32Round => crate::instructions::soft_float::mh32_round as *const u8,
            Self::MathF32Truncate => crate::instructions::soft_float::mh32_truncate as *const u8,
            Self::MathF64Ceiling => crate::instructions::soft_float::mh64_ceiling as *const u8,
            Self::MathF64Floor => crate::instructions::soft_float::mh64_floor as *const u8,
            Self::MathF64Round => crate::instructions::soft_float::mh64_round as *const u8,
            Self::MathF64Truncate => crate::instructions::soft_float::mh64_truncate as *const u8,
            // SoftFallback "Sha1".
            Self::HashChoose => crate::instructions::soft_fallback::hash_choose as *const u8,
            Self::FixedRotate => crate::instructions::soft_fallback::fixed_rotate as *const u8,
            Self::HashMajority => crate::instructions::soft_fallback::hash_majority as *const u8,
            Self::HashParity => crate::instructions::soft_fallback::hash_parity as *const u8,
            Self::Sha1SchedulePart1 => {
                crate::instructions::soft_fallback::sha1_schedule_part1 as *const u8
            }
            Self::Sha1SchedulePart2 => {
                crate::instructions::soft_fallback::sha1_schedule_part2 as *const u8
            }
        }
    }

    /// IR return type (C# EmitterContext.GetOperandType over the C#
    /// return type: byte/ushort/uint/int → I32, ulong → I64, V128 → V128,
    /// void → None).
    pub fn ret_type(self) -> OperandType {
        match self {
            Self::Break
            | Self::SupervisorCall
            | Self::Undefined
            | Self::WriteByte
            | Self::WriteUInt16
            | Self::WriteUInt32
            | Self::WriteUInt64
            | Self::WriteVector128
            | Self::EnqueueForRejit
            | Self::SignalMemoryTracking
            | Self::ThrowInvalidMemoryAccess
            | Self::InvalidateCacheLine => OperandType::None,

            Self::ReadByte | Self::ReadUInt16 | Self::ReadUInt32 | Self::CheckSynchronization => {
                OperandType::I32
            }

            // SoftFallback Crc32* all return uint → I32.
            Self::Crc32b
            | Self::Crc32h
            | Self::Crc32w
            | Self::Crc32x
            | Self::Crc32cb
            | Self::Crc32ch
            | Self::Crc32cw
            | Self::Crc32cx => OperandType::I32,

            Self::GetCtrEl0
            | Self::GetDczidEl0
            | Self::GetCntfrqEl0
            | Self::GetCntpctEl0
            | Self::GetCntvctEl0
            | Self::ReadUInt64
            | Self::DbgTranslate
            | Self::GetFunctionAddress => OperandType::I64,

            // SoftFallback long/ulong returns → I64.
            Self::SignedShrImm64
            | Self::UnsignedShrImm64
            | Self::CountLeadingSigns
            | Self::CountLeadingZeros => OperandType::I64,

            Self::ReadVector128 => OperandType::V128,
            // SoftFloat32 → FP32.
            Self::SoftF32FPAdd
            | Self::SoftF32FPSub
            | Self::SoftF32FPMul
            | Self::SoftF32FPDiv
            | Self::SoftF32FPSqrt
            | Self::SoftF32FPMax
            | Self::SoftF32FPMin
            | Self::SoftF32FPMulX
            | Self::SoftF32FPRecipEstimate
            | Self::SoftF32FPRecpX
            | Self::SoftF32FPRSqrtEstimate
            | Self::SoftF32FPRecipStep
            | Self::SoftF32FPRSqrtStep
            | Self::SoftF32FPRecipStepFused
            | Self::SoftF32FPRSqrtStepFused
            | Self::SoftF32FPCompareEQ
            | Self::SoftF32FPCompareGE
            | Self::SoftF32FPCompareGT
            | Self::SoftF32FPCompareLE
            | Self::SoftF32FPCompareLT => OperandType::FP32,
            // SoftFloat32 compare → i32.
            Self::SoftF32FPCompare => OperandType::I32,
            // SoftFloat64 → FP64.
            Self::SoftF64FPAdd
            | Self::SoftF64FPSub
            | Self::SoftF64FPMul
            | Self::SoftF64FPDiv
            | Self::SoftF64FPSqrt
            | Self::SoftF64FPMax
            | Self::SoftF64FPMin
            | Self::SoftF64FPMulX
            | Self::SoftF64FPRecipEstimate
            | Self::SoftF64FPRecpX
            | Self::SoftF64FPRSqrtEstimate
            | Self::SoftF64FPCompareEQ
            | Self::SoftF64FPCompareGE
            | Self::SoftF64FPCompareGT
            | Self::SoftF64FPCompareLE
            | Self::SoftF64FPCompareLT => OperandType::FP64,
            // SoftFloat64 compare → i32.
            Self::SoftF64FPCompare => OperandType::I32,
            // SoftFloat64 step functions → FP64.
            Self::SoftF64FPRecipStep
            | Self::SoftF64FPRSqrtStep
            | Self::SoftF64FPRecipStepFused
            | Self::SoftF64FPRSqrtStepFused => OperandType::FP64,
            // SoftFloat32 fused multiply-add → FP32.
            Self::SoftF32FPMulAdd
            | Self::SoftF32FPMulSub
            | Self::SoftF32FPNegMulAdd
            | Self::SoftF32FPNegMulSub => OperandType::FP32,
            // SoftFloat64 fused multiply-add → FP64.
            Self::SoftF64FPMulAdd
            | Self::SoftF64FPMulSub
            | Self::SoftF64FPNegMulAdd
            | Self::SoftF64FPNegMulSub => OperandType::FP64,
            // Abs helpers.
            Self::SoftF32Abs => OperandType::FP32,
            Self::SoftF64Abs => OperandType::FP64,
            // MathHelper / MathHelperF.
            Self::MathF32Ceiling
            | Self::MathF32Floor
            | Self::MathF32Round
            | Self::MathF32Truncate => OperandType::FP32,
            Self::MathF64Ceiling
            | Self::MathF64Floor
            | Self::MathF64Round
            | Self::MathF64Truncate => OperandType::FP64,
            // SoftFallback "Sha1".
            Self::FixedRotate => OperandType::I32,
            Self::HashChoose
            | Self::HashMajority
            | Self::HashParity
            | Self::Sha1SchedulePart1
            | Self::Sha1SchedulePart2 => OperandType::V128,
        }
    }

    /// DelegateTable symbol value (C#: Delegates.GetDelegateIndex).
    pub fn index(self) -> u64 {
        self as u64
    }

    /// Number of NativeFn variants (keep in sync with the enum; guarded by the
    /// const-assert below so a drift is a compile error).
    pub const COUNT: u64 = 103;

    /// Inverse of `index()`  reconstruct the NativeFn from its DelegateTable
    /// symbol value. Used by the PTC reloc resolver to map a saved delegate
    /// index back to the current run's native-helper pointer. Safe: the enum is
    /// `#[repr(u64)]` with sequential discriminants 0..COUNT, and the index is
    /// bounds-checked.
    pub fn from_index(idx: u64) -> NativeFn {
        assert!(idx < Self::COUNT, "NativeFn index {idx} out of range");
        // SAFETY: repr(u64), fieldless, discriminants 0..COUNT, idx in range.
        unsafe { std::mem::transmute::<u64, NativeFn>(idx) }
    }
}

// Compile-time guard: COUNT == last variant's discriminant + 1. If a variant is
// added/removed this fails to compile until COUNT is updated.
const _: () = assert!(
    NativeFn::Sha1SchedulePart2 as u64 + 1 == NativeFn::COUNT,
    "NativeFn variant count changed  update NativeFn::COUNT"
);
