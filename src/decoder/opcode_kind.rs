use crate::decoder::utilities::{Condition, DataOperation, IntegerType, ShiftType};
use crate::registers::{GpRegisterIndex, VecRegisterIndex};

#[derive(Debug, Clone, Copy)]
pub enum OpCodeKind {
    // -- Integer ALU ----------------------------------------------------------
    Alu {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        data_op: DataOperation,
    },

    AluImm {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        data_op: DataOperation,
        immediate: i64,
    },

    AluRs {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        data_op: DataOperation,
        shift: u8,
        shift_type: ShiftType,
    },

    AluRx {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        data_op: DataOperation,
        shift: u8,
        int_type: IntegerType,
    },

    AluBinary {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        data_op: DataOperation,
    },

    Bfm {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        wmask: i64,
        tmask: i64,
        pos: u8,
        shift: u8,
    },

    Mov {
        rd: GpRegisterIndex,
        immediate: i64,
        bit: u8,
    },

    Mul {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        ra: GpRegisterIndex,
        data_op: DataOperation,
    },

    Csel {
        rd: GpRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        cond: Condition,
    },

    Ccmp {
        rn: GpRegisterIndex,
        rm_or_imm: u8,
        nzcv: u8,
        cond: Condition,
        is_imm: bool,
    },

    // -- PC-relative addressing -----------------------------------------------
    Adr {
        rd: GpRegisterIndex,
        immediate: i64,
    },

    // -- Branches -------------------------------------------------------------
    BImm {
        immediate: i64,
        link: bool,
    },

    BImmCond {
        cond: Condition,
        immediate: i64,
    },

    // CBZ vs CBNZ (and TBZ vs TBNZ) is carried by InstName  the C#
    // OpCodeBImmCmp/OpCodeBImmTest classes have no field for it; the
    // emitters dispatch via the Cbz/Cbnz (Tbz/Tbnz) wrappers.
    BImmCmp {
        rt: GpRegisterIndex,
        immediate: i64,
    },

    BImmTest {
        rt: GpRegisterIndex,
        bit: u8,
        immediate: i64,
    },

    BReg {
        rn: GpRegisterIndex,
    },

    // -- Memory ---------------------------------------------------------------
    MemImm {
        rt: GpRegisterIndex,
        rn: GpRegisterIndex,
        size: u8,
        extend64: bool,
        immediate: i64,
        wback: bool,
        post_idx: bool,
        unscaled: bool,
    },

    MemPair {
        rt: GpRegisterIndex,
        rt2: GpRegisterIndex,
        rn: GpRegisterIndex,
        size: u8,
        extend64: bool,
        immediate: i64,
        wback: bool,
        post_idx: bool,
    },

    MemReg {
        rt: GpRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        size: u8,
        extend64: bool,
        shift: bool,
        int_type: IntegerType,
    },

    MemLit {
        rt: GpRegisterIndex,
        size: u8,
        immediate: i64,
        signed: bool,
        prefetch: bool,
    },

    MemEx {
        rt: GpRegisterIndex,
        rt2: GpRegisterIndex,
        rn: GpRegisterIndex,
        rs: GpRegisterIndex,
        size: u8,
    },

    // -- System ---------------------------------------------------------------
    System {
        rt: GpRegisterIndex,
        op0: u8,
        op1: u8,
        op2: u8,
        crn: u8,
        crm: u8,
    },

    Exception {
        id: u16,
    },

    // -- SIMD / FP ------------------------------------------------------------
    Simd {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        opc: u8,
        size: u8,
    },

    SimdReg {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        rm: VecRegisterIndex,
        ra: VecRegisterIndex,
        opc: u8,
        size: u8,
        bit3: bool,
    },

    SimdImm {
        rd: VecRegisterIndex,
        immediate: i64,
        size: u8,
    },

    SimdShImm {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        opc: u8,
        size: i8,
        imm: u8,
    },

    SimdFmov {
        rd: VecRegisterIndex,
        immediate: i64,
        size: u8,
    },

    SimdCvt {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        opc: u8,
        size: u8,
        fbits: u8,
    },

    SimdExt {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        rm: VecRegisterIndex,
        ra: VecRegisterIndex,
        opc: u8,
        size: u8,
        bit3: bool,
        imm4: u8,
    },

    SimdFcond {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        rm: VecRegisterIndex,
        ra: VecRegisterIndex,
        opc: u8,
        size: u8,
        bit3: bool,
        nzcv: u8,
        cond: Condition,
    },

    SimdMemReg {
        rt: VecRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        size: u8,
        shift: bool,
        int_type: IntegerType,
    },

    SimdMemMs {
        rt: VecRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        size: u8,
        wback: bool,
        reps: u8,
        s_elems: u8,
        elems: u8,
    },

    SimdMemImm {
        rt: VecRegisterIndex,
        rn: GpRegisterIndex,
        size: u8,
        immediate: i64,
        wback: bool,
        post_idx: bool,
        unscaled: bool,
    },

    SimdMemPair {
        rt: VecRegisterIndex,
        rt2: VecRegisterIndex,
        rn: GpRegisterIndex,
        size: u8,
        immediate: i64,
        wback: bool,
        post_idx: bool,
    },

    SimdMemSs {
        rt: VecRegisterIndex,
        rn: GpRegisterIndex,
        rm: GpRegisterIndex,
        size: u8,
        wback: bool,
        s_elems: u8,
        index: u8,
        replicate: bool,
    },

    SimdIns {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        size: u8,
        src_index: u8,
        dst_index: u8,
    },

    SimdMemLit {
        rt: VecRegisterIndex,
        size: u8,
        immediate: i64,
    },

    SimdRegElem {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        rm: VecRegisterIndex,
        ra: VecRegisterIndex,
        opc: u8,
        size: u8,
        bit3: bool,
        index: u8,
    },

    SimdRegElemF {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        rm: VecRegisterIndex,
        ra: VecRegisterIndex,
        opc: u8,
        size: u8,
        bit3: bool,
        index: u8,
    },

    SimdTbl {
        rd: VecRegisterIndex,
        rn: VecRegisterIndex,
        rm: VecRegisterIndex,
        ra: VecRegisterIndex,
        opc: u8,
        size: u8,
        bit3: bool,
    },

    Undefined,
}
