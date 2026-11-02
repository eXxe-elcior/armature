// Physical registers
use std::sync::atomic::{AtomicBool, AtomicI32};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpRegisterIndex(u8); // X0-X29, X30 = LR, XZR/SP = 31

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VecRegisterIndex(u8); // V0-V31

impl GpRegisterIndex {
    pub const ZR_SP: Self = Self(31);
    #[inline]
    pub fn idx(self) -> u8 {
        self.0
    }
    #[inline]
    pub fn is_zr_or_sp(self) -> bool {
        self.0 == 31
    }
}

impl VecRegisterIndex {
    #[inline]
    pub fn new(v: u32) -> Self {
        Self(v as u8 & 0x1f)
    }
    #[inline]
    pub fn idx(self) -> u8 {
        self.0
    }
}

impl From<u8> for GpRegisterIndex {
    #[inline]
    fn from(v: u8) -> Self {
        Self(v & 0x1f)
    }
}
impl From<u8> for VecRegisterIndex {
    #[inline]
    fn from(v: u8) -> Self {
        Self(v & 0x1f)
    }
}

// System registers
// These are NOT physical registers. They are named slots in CpuState,
// accessed only via MRS/MSR (OpCodeSystem). The u16 is the raw AArch64
// system register encoding: op0[1]|op1[3]|CRn[4]|CRm[4]|op2[3] = 15 bits.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemRegister(pub u16);

impl SystemRegister {
    // Encoding: (op0, op1, CRn, CRm, op2)
    const fn enc(op0: u16, op1: u16, crn: u16, crm: u16, op2: u16) -> Self {
        Self((op0 << 13) | (op1 << 10) | (crn << 6) | (crm << 2) | op2)
    }

    // FP / status
    pub const FPSR: Self = Self::enc(3, 3, 4, 4, 1);
    pub const FPCR: Self = Self::enc(3, 3, 4, 4, 0);

    // Thread pointer (what ARMeilleure calls TPREL0 / TPREL1)
    pub const TPIDR_EL0: Self = Self::enc(3, 3, 13, 0, 2); // user r/w
    // TPIDRRO_EL0 (ARMeilleure's TPIDRREL0  read-only for EL0)
    pub const TPIDRRO_EL0: Self = Self::enc(3, 3, 13, 0, 3);

    // Debug / ID regs you'll hit eventually
    pub const CTR_EL0: Self = Self::enc(3, 3, 0, 0, 1);
    pub const DCZID_EL0: Self = Self::enc(3, 3, 0, 0, 7);

    /// Decode directly from an MRS/MSR encoding (raw bits [19:5] of the instruction)
    pub fn from_raw(raw: u32) -> Self {
        Self(((raw >> 5) & 0x7fff) as u16)
    }

    pub fn is_known(self) -> bool {
        matches!(
            self,
            Self::FPSR
                | Self::FPCR
                | Self::TPIDR_EL0
                | Self::TPIDRRO_EL0
                | Self::CTR_EL0
                | Self::DCZID_EL0
        )
    }
}

// 128-bit vector register type
pub type V128 = u128;

// CPU state layout
// This is what the JIT-compiled code actually reads/writes via pointer.
// Offsets here are what the emitter uses to generate memory operands.
// Mirrors ARMeilleure's NativeCtxStorage exactly  offset-for-offset,
// byte-for-byte. No padding mismatches.
//
//     field                 offset     size
//     ─────────────────────────────────────
//     x: [u64; 32]              0      256
//     v: [V128; 32]           256      512
//     flags: [u32; 32]        768      128
//     fp_flags: [u32; 32]     896      128
//     tpidr_el0              1024        8
//     tpidr2_el0             1032        8
//     tpidrro_el0            1040        8
//     dispatch_address       1048        8
//     exclusive_address      1056        8
//     exclusive_value_low    1064        8
//     exclusive_value_high   1072        8
//     debug_precise_pc       1080        8
//     counter                1088        4
//     running                1092        1
//     _tail_pad              1093        11
//     ─────────────────────────────────────
//     total                          1104

#[repr(C, align(16))]
pub struct CpuState {
    /// Integer Registers X0-X31 (index 31 is the SP slot / physical storage)
    pub x: [u64; 32], // Offset: 0, Size: 256

    /// Vector Registers V0-V31
    pub v: [V128; 32], // Offset: 256, Size: 512

    /// NZCV / PState flags (one bit per slot, for fast JIT access)
    pub flags: [u32; 32], // Offset: 768, Size: 128

    /// FPCR/FPSR status flags (one bit per slot, for fast JIT access)
    pub fp_flags: [u32; 32], // Offset: 896, Size: 128

    /// Thread Pointer Register (EL0)
    pub tpidr_el0: i64, // Offset: 1024, Size: 8

    /// Thread Pointer Register 2 (EL0)
    pub tpidr2_el0: i64, // Offset: 1032, Size: 8

    /// Read-Only Thread Pointer Register (EL0)
    pub tpidrro_el0: i64, // Offset: 1040, Size: 8

    /// JIT dispatcher branching target address
    pub dispatch_address: u64, // Offset: 1048, Size: 8

    /// Exclusive Monitor Address
    pub exclusive_address: u64, // Offset: 1056, Size: 8

    /// Exclusive Monitor Value Low
    pub exclusive_value_low: u64, // Offset: 1064, Size: 8

    /// Exclusive Monitor Value High
    pub exclusive_value_high: u64, // Offset: 1072, Size: 8

    /// Precise PC value used for debugger integration
    pub debug_precise_pc: u64, // Offset: 1080, Size: 8

    /// Interrupt/Execution counter.
    ///
    /// Written cross-thread (another host thread calls `stop_running` to
    /// zero it while this state's owner thread runs guest code), and read
    /// both by JIT'd code (raw `mov` at `counter_offset`) and by Rust at
    /// the synchronization points. `AtomicI32` is layout-identical to `i32`
    /// (same size/align, `repr(transparent)` over an `UnsafeCell<i32>`), so
    /// the JIT offset and the baked machine loads are unchanged; the atomic
    /// only constrains the Rust-side accesses so the cross-thread write is
    /// not a data race.
    pub counter: AtomicI32, // Offset: 1088, Size: 4

    /// JIT execution loop state (0 = stopped, 1 = running). Cross-thread
    /// written by `stop_running`; see `counter` for the layout rationale.
    pub running: AtomicBool, // Offset: 1092, Size: 1
                             // _tail_pad Bytes : 11
}

impl CpuState {
    pub const fn gpr_offset(idx: GpRegisterIndex) -> usize {
        std::mem::offset_of!(CpuState, x) + idx.0 as usize * 8
    }
    pub const fn flag_offset(idx: u8) -> usize {
        std::mem::offset_of!(CpuState, flags) + idx as usize * 4
    }
    pub const fn fp_flag_offset(idx: u8) -> usize {
        std::mem::offset_of!(CpuState, fp_flags) + idx as usize * 4
    }
    pub const fn vec_offset(idx: VecRegisterIndex) -> usize {
        std::mem::offset_of!(CpuState, v) + idx.0 as usize * 16
    }
    // C# NativeContext offset getters used by emitters (GetCounterOffset,
    // GetTpidrEl0Offset, GetTpidrroEl0Offset, GetTpidr2El0Offset,
    // GetExclusiveAddressOffset, GetExclusiveValueOffset,
    // GetDispatchAddressOffset, GetRunningOffset).
    pub const fn dispatch_address_offset() -> usize {
        std::mem::offset_of!(CpuState, dispatch_address)
    }
    pub const fn running_offset() -> usize {
        std::mem::offset_of!(CpuState, running)
    }
    pub const fn exclusive_address_offset() -> usize {
        std::mem::offset_of!(CpuState, exclusive_address)
    }
    /// The exclusive value is a 16-byte slot; this is the low half  the
    /// high half lives at +8 (C# ExclusiveValueLow/High pair).
    pub const fn exclusive_value_offset() -> usize {
        std::mem::offset_of!(CpuState, exclusive_value_low)
    }
    pub const fn counter_offset() -> usize {
        std::mem::offset_of!(CpuState, counter)
    }
    pub const fn tpidr_el0_offset() -> usize {
        std::mem::offset_of!(CpuState, tpidr_el0)
    }
    pub const fn tpidrro_el0_offset() -> usize {
        std::mem::offset_of!(CpuState, tpidrro_el0)
    }
    pub const fn tpidr2_el0_offset() -> usize {
        std::mem::offset_of!(CpuState, tpidr2_el0)
    }
    pub const fn sys_offset(reg: SystemRegister) -> Option<usize> {
        match reg {
            SystemRegister::TPIDR_EL0 => Some(std::mem::offset_of!(CpuState, tpidr_el0)),
            SystemRegister::TPIDRRO_EL0 => Some(std::mem::offset_of!(CpuState, tpidrro_el0)),
            _ => None,
        }
    }
}
