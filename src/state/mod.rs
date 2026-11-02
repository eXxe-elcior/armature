use crate::ir::types::RegisterType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PState {
    TFlag = 5,
    EFlag = 9,
    GE0Flag = 16,
    GE1Flag = 17,
    GE2Flag = 18,
    GE3Flag = 19,
    QFlag = 27,
    VFlag = 28,
    CFlag = 29,
    ZFlag = 30,
    NFlag = 31,
}

impl PState {
    pub fn register_type(self) -> RegisterType {
        match self {
            Self::VFlag
            | Self::CFlag
            | Self::ZFlag
            | Self::NFlag
            | Self::TFlag
            | Self::EFlag
            | Self::GE0Flag
            | Self::GE1Flag
            | Self::GE2Flag
            | Self::GE3Flag
            | Self::QFlag => RegisterType::Flag,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FPState {
    IocFlag = 0,
    DzcFlag = 1,
    OfcFlag = 2,
    UfcFlag = 3,
    IxcFlag = 4,
    IdcFlag = 7,
    QcFlag = 27,
    VFlag = 28,
    CFlag = 29,
    ZFlag = 30,
    NFlag = 31,
    IoeFlag = 8,
    DzeFlag = 9,
    OfeFlag = 10,
    UfeFlag = 11,
    IxeFlag = 12,
    IdeFlag = 15,
    RMode0Flag = 22,
    RMode1Flag = 23,
    FzFlag = 24,
    DnFlag = 25,
    AhpFlag = 26,
}

impl FPState {
    pub fn register_type(self) -> RegisterType {
        RegisterType::FpFlag
    }
}

// Port of State/FPCR.cs ([Flags] enum FPCR : uint). Bit positions match
// the FPState flag indices above (IoeFlag=8 … AhpFlag=26).
pub mod fpcr {
    pub const IOE: u32 = 1 << 8;
    pub const DZE: u32 = 1 << 9;
    pub const OFE: u32 = 1 << 10;
    pub const UFE: u32 = 1 << 11;
    pub const IXE: u32 = 1 << 12;
    pub const IDE: u32 = 1 << 15;
    pub const RMODE0: u32 = 1 << 22;
    pub const RMODE1: u32 = 1 << 23;
    pub const FZ: u32 = 1 << 24;
    pub const DN: u32 = 1 << 25;
    pub const AHP: u32 = 1 << 26;

    pub const MASK: u32 = AHP | DN | FZ | RMODE1 | RMODE0 | IDE | IXE | UFE | OFE | DZE | IOE; // 0x07C09F00
}

// Port of State/FPSR.cs ([Flags] enum FPSR : uint). Bit positions match
// the FPState flag indices above (IocFlag=0 … QcFlag=27).
pub mod fpsr {
    pub const IOC: u32 = 1 << 0;
    pub const DZC: u32 = 1 << 1;
    pub const OFC: u32 = 1 << 2;
    pub const UFC: u32 = 1 << 3;
    pub const IXC: u32 = 1 << 4;
    pub const IDC: u32 = 1 << 7;
    pub const QC: u32 = 1 << 27;

    pub const MASK: u32 = QC | IDC | IXC | UFC | OFC | DZC | IOC; // 0x0800009F
}
// Execution context and host callbacks.

pub mod execution_context;
