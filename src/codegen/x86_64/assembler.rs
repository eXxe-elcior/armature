// Port of CodeGen/X86/Assembler.cs (the combined encoder, jump management,
// and public mnemonic layer). Exception: GetCode fixpoint re-emission
// is split into its own method (caller must use jmp_offset/jcc_offset).
//
// Design divergence: C# passes default Operand structs for absent operands;
// we use Option<OperandIdx> consistently. The callee resolves operands
// through &OperandArena.

use crate::util::hash::FxHashMap as HashMap;

use crate::codegen::linking::{RelocationEntry, RelocationInfo};
use crate::codegen::x86_64::assembler_table::{self, BAD_OP, flags};
use crate::ir::operand::{OperandArena, OperandData, OperandIndex};
use crate::ir::types::{OperandKind, OperandType, Symbol};

use super::condition::X86Condition;
use super::instruction::X86Instruction;

// Constants
const RESERVED_BYTES_FOR_JUMP: i64 = 1;
const OP_MODRM_BITS: u32 = 24;
const REX_PREFIX: u8 = 0x40;
const REX_W_PREFIX: u8 = 0x48;
const LOCK_PREFIX: u8 = 0xf0;
const _MAX_REG_NUMBER: u8 = 15;

// Hardware capabilities (stub  will detect at runtime)
pub mod capabilities {
    // C# HardwareCapabilities.SupportsVexEncoding  real CPUID detection
    // via optimizations.rs (cached). Emitting VEX on a non-AVX host is
    // SIGILL, so no hardcoding.
    pub fn supports_vex_encoding() -> bool {
        crate::optimizations::supports_vex_encoding()
    }
    pub fn supports_evex_encoding() -> bool {
        false // EVEX encoding is not implemented.
    }
}

// Internal state
#[derive(Debug, Clone)]
struct Jump {
    is_conditional: bool,
    condition: X86Condition,
    jump_label: OperandIndex,
    jump_target: Option<u64>,
    jump_position: u64,
    offset: i64,
    inst_size: u16,
}

impl Jump {
    fn new(jump_label: OperandIndex, jump_position: u64) -> Self {
        Self {
            is_conditional: false,
            condition: X86Condition::Overflow,
            jump_label,
            jump_target: None,
            jump_position,
            offset: 0,
            inst_size: 0,
        }
    }

    fn new_conditional(
        condition: X86Condition,
        jump_label: OperandIndex,
        jump_position: u64,
    ) -> Self {
        Self {
            is_conditional: true,
            condition,
            jump_label,
            jump_target: None,
            jump_position,
            offset: 0,
            inst_size: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct Reloc {
    jump_index: i32,
    position: i32,
    symbol: Symbol,
}

pub struct Assembler {
    stream: Vec<u8>,
    labels: HashMap<OperandIndex, u64>,
    jumps: Vec<Jump>,
    relocs: Option<Vec<Reloc>>,
}

impl Assembler {
    pub fn new(relocatable: bool) -> Self {
        Self {
            stream: Vec::new(),
            labels: HashMap::default(),
            jumps: Vec::new(),
            relocs: if relocatable { Some(Vec::new()) } else { None },
        }
    }

    pub fn has_relocs(&self) -> bool {
        self.relocs.is_some()
    }

    pub fn mark_label(&mut self, label: OperandIndex) {
        self.labels.insert(label, self.stream.len() as u64);
    }

    pub fn position(&self) -> u64 {
        self.stream.len() as u64
    }

    // Byte core
    fn write_byte(&mut self, value: u8) {
        self.stream.push(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.stream.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.stream.extend_from_slice(&value.to_le_bytes());
    }

    fn _write_u64(&mut self, value: u64) {
        self.stream.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i16(&mut self, value: i16) {
        self.write_u16(value as u16);
    }

    fn write_i32(&mut self, value: i32) {
        self.write_u32(value as u32);
    }

    // Fit helpers
    pub fn const_fits_on_s8(value: i64) -> bool {
        i8::try_from(value).is_ok()
    }

    pub fn const_fits_on_s32(value: i64) -> bool {
        i32::try_from(value).is_ok()
    }

    pub fn is_imm8(immediate: u64, typ: OperandType) -> bool {
        let value = match typ {
            OperandType::I32 => immediate as i32 as i64,
            _ => immediate as i64,
        };
        Self::const_fits_on_s8(value)
    }

    pub fn is_imm32(immediate: u64, typ: OperandType) -> bool {
        let value = match typ {
            OperandType::I32 => immediate as i32 as i64,
            _ => immediate as i64,
        };
        Self::const_fits_on_s32(value)
    }

    fn is_64_bits(typ: OperandType) -> bool {
        matches!(typ, OperandType::I64 | OperandType::FP64)
    }

    // Jump length helpers
    pub fn get_jcc_length(offset: i64) -> u16 {
        if Self::const_fits_on_s8(if offset < 0 { offset - 2 } else { offset }) {
            2
        } else if Self::const_fits_on_s32(if offset < 0 { offset - 6 } else { offset }) {
            6
        } else {
            panic!("Jcc offset {} out of range (need rel8 or rel32)", offset);
        }
    }

    pub fn get_jmp_length(offset: i64) -> u16 {
        if Self::const_fits_on_s8(if offset < 0 { offset - 2 } else { offset }) {
            2
        } else if Self::const_fits_on_s32(if offset < 0 { offset - 5 } else { offset }) {
            5
        } else {
            panic!("Jmp offset {} out of range (need rel8 or rel32)", offset);
        }
    }

    // REX prefix
    fn rex_prefix(
        dest: Option<&OperandData>,
        src: Option<&OperandData>,
        typ: OperandType,
        rrm: bool,
    ) -> u8 {
        let mut rex = 0u8;
        if Self::is_64_bits(typ) {
            rex |= REX_W_PREFIX;
        }
        // C# GetRexPrefix guards Kind == Register  Memory/Constant
        // operands must not contribute register high bits.
        if let Some(d) = dest {
            if d.kind == OperandKind::Register {
                let idx = d.reg_index() as u8;
                if idx >= 8 {
                    rex |= REX_PREFIX | (idx >> 3) << (if rrm { 2 } else { 0 });
                }
            }
        }
        if let Some(s) = src {
            if s.kind == OperandKind::Register {
                let idx = s.reg_index() as u8;
                if idx >= 8 {
                    rex |= REX_PREFIX | (idx >> 3) << (if rrm { 0 } else { 2 });
                }
            }
        }
        rex
    }

    // Operand helpers (internal)
    fn kind(arena: &OperandArena, idx: Option<OperandIndex>) -> OperandKind {
        idx.map_or(OperandKind::None, |i| arena.get(i).kind)
    }

    fn data<'a>(arena: &'a OperandArena, idx: Option<OperandIndex>) -> Option<&'a OperandData> {
        idx.map(|i| arena.get(i))
    }

    // WriteOpCode (private, central encoder)
    #[allow(clippy::too_many_arguments)]
    fn write_op_code(
        &mut self,
        operands: &OperandArena,
        dest: Option<OperandIndex>,
        src1: Option<OperandIndex>,
        src2: Option<OperandIndex>,
        typ: OperandType,
        flags: u32,
        mut op_code: u32,
        rrm: bool,
    ) {
        let dest_data = Self::data(operands, dest);
        let src1_data = Self::data(operands, src1);
        let src2_data = Self::data(operands, src2);

        // C#: GetRexPrefix(dest, src2, type, rrm)
        let mut rex_prefix = Self::rex_prefix(dest_data, src2_data, typ, rrm);

        if flags & flags::REX_W != 0 {
            rex_prefix |= REX_W_PREFIX;
        }

        // C#: modRM = (opCode >> OpModRMBits) << 3
        let mut mod_rm = ((op_code >> OP_MODRM_BITS) << 3) as u8;

        // Track memory operand from either dest or src2.
        let mut mem_base: Option<u8> = None;
        let mut mem_index: Option<u8> = None;
        let mut mem_scale: u8 = 0;
        let mut mem_disp: i32 = 0;
        let mut has_mem_op = false;

        // dest
        if let Some(dd) = dest_data {
            match dd.kind {
                OperandKind::Register => {
                    let reg_idx = dd.reg_index() as u8;
                    // C#: modRM |= regIndex & 0b111 << (rrm ? 3 : 0)
                    mod_rm |= (reg_idx & 0b111) << (if rrm { 3 } else { 0 });

                    if flags & flags::REG8_DEST != 0 && reg_idx >= 4 {
                        rex_prefix |= REX_PREFIX;
                    }
                }
                OperandKind::Memory => {
                    mem_base = dd.mem_base().map(|bi| operands.get(bi).reg_index() as u8);
                    mem_index = dd.mem_index().map(|ii| operands.get(ii).reg_index() as u8);
                    mem_scale = dd.mem_scale as u8;
                    mem_disp = dd.mem_disp;
                    has_mem_op = true;
                }
                _ => panic!("Invalid destination operand kind {:?}", dd.kind),
            }
        }

        // src2
        if let Some(sd) = src2_data {
            match sd.kind {
                OperandKind::Register => {
                    let reg_idx = sd.reg_index() as u8;
                    // C#: modRM |= regIndex & 0b111 << (rrm ? 0 : 3)
                    mod_rm |= (reg_idx & 0b111) << (if rrm { 0 } else { 3 });

                    if flags & flags::REG8_SRC != 0 && reg_idx >= 4 {
                        rex_prefix |= REX_PREFIX;
                    }
                }
                OperandKind::Memory if !has_mem_op => {
                    mem_base = sd.mem_base().map(|bi| operands.get(bi).reg_index() as u8);
                    mem_index = sd.mem_index().map(|ii| operands.get(ii).reg_index() as u8);
                    mem_scale = sd.mem_scale as u8;
                    mem_disp = sd.mem_disp;
                    has_mem_op = true;
                }
                _ => panic!("Invalid source operand kind {:?}", sd.kind),
            }
        }

        // SIB / displacement calculation
        let needs_sib_byte: bool;
        let needs_displacement: bool;
        let sib: u8;

        if has_mem_op {
            let base_reg_low = mem_base.unwrap_or(0) & 0b111;

            needs_sib_byte = mem_index.is_some() || base_reg_low == 4;
            needs_displacement = mem_disp != 0 || base_reg_low == 5;

            if needs_displacement {
                if Self::const_fits_on_s8(mem_disp as i64) {
                    mod_rm |= 0x40;
                } else {
                    mod_rm |= 0x80;
                }
            }

            // C#: baseReg index >= 8 → REX.B
            if let Some(base) = mem_base {
                if base >= 8 {
                    rex_prefix |= REX_PREFIX | (base >> 3);
                }
            }

            let mut sib_val: u8 = 0;
            if needs_sib_byte {
                sib_val = base_reg_low;

                if let Some(idx_reg) = mem_index {
                    assert_ne!(idx_reg, 4, "Using RSP as index register is not allowed");
                    if idx_reg >= 8 {
                        rex_prefix |= REX_PREFIX | ((idx_reg >> 3) << 1);
                    }
                    sib_val |= (idx_reg & 0b111) << 3;
                } else {
                    // No index → encode RSP as index (= "no index")
                    sib_val |= 0b100 << 3;
                }

                sib_val |= (mem_scale & 0b11) << 6;
                mod_rm |= 0b100;
            } else {
                mod_rm |= base_reg_low;
            }
            sib = sib_val;
        } else {
            needs_sib_byte = false;
            needs_displacement = false;
            sib = 0;
            mod_rm |= 0xc0; // mod = 11 (register direct)
        }

        debug_assert_ne!(op_code, BAD_OP, "Bad opcode value");

        // Prefix emission
        if flags & flags::EVEX != 0 && capabilities::supports_evex_encoding() {
            todo!("WriteEvexInst");
        } else if flags & flags::VEX != 0 && capabilities::supports_vex_encoding() {
            // C# switch on prefix mask
            let prefix_value = flags & flags::PREFIX_MASK;
            let mut vex_byte2: u8 = match prefix_value {
                _ if prefix_value == flags::PREFIX_66 => 1,
                _ if prefix_value == flags::PREFIX_F3 => 2,
                _ if prefix_value == flags::PREFIX_F2 => 3,
                _ => 0,
            };

            // C#: src1 register → vex vvvv field
            if let Some(s1) = src1_data {
                vex_byte2 |= (s1.reg_index() as u8 ^ 0xf) << 3;
            } else {
                vex_byte2 |= 0b1111 << 3;
            }

            let op_code_high = (op_code >> 8) as u16;

            if rex_prefix & 0b1011 == 0 && op_code_high == 0x0f {
                // Two-byte VEX form (C5)
                self.write_byte(0xc5);
                vex_byte2 |= ((!rex_prefix & 4) << 5) as u8;
                self.write_byte(vex_byte2);
            } else {
                // Three-byte VEX form (C4)
                self.write_byte(0xc4);

                let mut vex_byte1: u8 = ((!rex_prefix & 7) << 5) as u8;
                match op_code_high {
                    0x0f => vex_byte1 |= 1,
                    0xf38 => vex_byte1 |= 2,
                    0xf3a => vex_byte1 |= 3,
                    _ => panic!("Failed to VEX encode opcode 0x{op_code:X}"),
                }

                vex_byte2 |= ((rex_prefix & 8) << 4) as u8;
                self.write_byte(vex_byte1);
                self.write_byte(vex_byte2);
            }

            // C#: opCode &= 0xff  the 0x0f/0x0f38/0x0f3a escape bytes are
            // encoded in the VEX map field; only the low byte is emitted.
            op_code &= 0xff;
        } else {
            // Legacy prefix emission (66 / F2 / F3)
            if flags & flags::PREFIX_66 != 0 {
                self.write_byte(0x66);
            }
            if flags & flags::PREFIX_F2 != 0 {
                self.write_byte(0xf2);
            }
            if flags & flags::PREFIX_F3 != 0 {
                self.write_byte(0xf3);
            }
            if rex_prefix != 0 {
                self.write_byte(rex_prefix);
            }
        }

        // Opcode bytes
        let mut final_op = op_code;

        // C#: if (dest != default && (flags & RegOnly) != 0) → opCode += dest reg index & 7
        if dest.is_some() && flags & flags::REG_ONLY != 0 {
            if let Some(dd) = dest_data {
                final_op += (dd.reg_index() & 7) as u32;
            }
        }

        if final_op & 0xff0000 != 0 {
            self.write_byte((final_op >> 16) as u8);
        }
        if final_op & 0xff00 != 0 {
            self.write_byte((final_op >> 8) as u8);
        }
        self.write_byte(final_op as u8);

        // ModRM / SIB / disp
        if flags & flags::REG_ONLY == 0 {
            self.write_byte(mod_rm);
            if needs_sib_byte {
                self.write_byte(sib);
            }
            if needs_displacement {
                if Self::const_fits_on_s8(mem_disp as i64) {
                    self.write_byte(mem_disp as u8);
                } else {
                    self.write_i32(mem_disp);
                }
            }
        }
    }

    // WriteInstruction (2-operand: dest, source, type, inst)
    fn write_instruction_2op(
        &mut self,
        operands: &OperandArena,
        dest: Option<OperandIndex>,
        source: Option<OperandIndex>,
        typ: OperandType,
        inst: X86Instruction,
    ) {
        let info = &assembler_table::INST_TABLE[inst as usize];
        let source_kind = Self::kind(operands, source);
        let dest_kind = Self::kind(operands, dest);

        if source_kind != OperandKind::None {
            match source_kind {
                OperandKind::Constant => {
                    let imm = source.map(|i| operands.get(i).value).unwrap_or(0);
                    // C# Operand.Relocatable: Symbol.Type != None. Relocatable
                    // constants must take the imm64 form so the 8-byte slot
                    // can be patched  even if the value fits in 32 bits.
                    let relocatable = source.map_or(false, |i| {
                        operands.get(i).symbol.typ != crate::ir::types::SymbolType::None
                    });

                    // C# special cases: Mov8 → always OpRMImm8, Mov16 → OpRMImm32
                    if inst == X86Instruction::Mov8 {
                        self.write_op_code(
                            operands,
                            dest,
                            None,
                            None,
                            typ,
                            info.flags,
                            info.op_rm_imm8,
                            false,
                        );
                        self.write_byte(imm as u8);
                    } else if inst == X86Instruction::Mov16 {
                        self.write_op_code(
                            operands,
                            dest,
                            None,
                            None,
                            typ,
                            info.flags,
                            info.op_rm_imm32,
                            false,
                        );
                        self.write_i16(imm as i16);
                    } else if Self::is_imm8(imm, typ) && info.op_rm_imm8 != BAD_OP {
                        self.write_op_code(
                            operands,
                            dest,
                            None,
                            None,
                            typ,
                            info.flags,
                            info.op_rm_imm8,
                            false,
                        );
                        self.write_byte(imm as u8);
                    } else if !relocatable && Self::is_imm32(imm, typ) && info.op_rm_imm32 != BAD_OP
                    {
                        self.write_op_code(
                            operands,
                            dest,
                            None,
                            None,
                            typ,
                            info.flags,
                            info.op_rm_imm32,
                            false,
                        );
                        self.write_i32(imm as i32);
                    } else if dest_kind == OperandKind::Register && info.op_r_imm64 != BAD_OP {
                        // C# r/m=register → use OpRImm64 form (e.g., MOV r, imm64)
                        let dest_data = Self::data(operands, dest);
                        let rex = Self::rex_prefix(dest_data, None, typ, false);
                        if rex != 0 {
                            self.write_byte(rex);
                        }
                        self.write_byte(
                            (info.op_r_imm64 + (dest_data.unwrap().reg_index() & 0b111)) as u8,
                        );

                        // Relocatable support
                        if let Some(src) = source {
                            let src_data = operands.get(src);
                            if self.relocs.is_some()
                                && src_data.symbol.typ != crate::ir::types::SymbolType::None
                            {
                                let reloc = Reloc {
                                    jump_index: (self.jumps.len() as i32).wrapping_sub(1),
                                    position: self.stream.len() as i32,
                                    symbol: src_data.symbol,
                                };
                                self.relocs.as_mut().unwrap().push(reloc);
                            }
                        }

                        self._write_u64(imm);
                    } else {
                        panic!("Failed to encode constant 0x{imm:X} for inst {inst:?}");
                    }
                }
                OperandKind::Register if info.op_rmr != BAD_OP => {
                    self.write_op_code(
                        operands,
                        dest,
                        None,
                        source,
                        typ,
                        info.flags,
                        info.op_rmr,
                        false,
                    );
                }
                _ if info.op_rrm != BAD_OP => {
                    self.write_op_code(
                        operands,
                        dest,
                        None,
                        source,
                        typ,
                        info.flags,
                        info.op_rrm,
                        true,
                    );
                }
                _ => panic!("Invalid source operand kind {source_kind:?}"),
            }
        } else if info.op_rrm != BAD_OP {
            self.write_op_code(
                operands,
                dest,
                None,
                source,
                typ,
                info.flags,
                info.op_rrm,
                true,
            );
        } else if info.op_rmr != BAD_OP {
            self.write_op_code(
                operands,
                dest,
                None,
                source,
                typ,
                info.flags,
                info.op_rmr,
                false,
            );
        } else {
            panic!("Source operand is None and no suitable opcode form");
        }
    }

    // WriteInstruction (3-operand: dest, src1, src2, inst [, type])
    #[allow(clippy::too_many_arguments)]
    fn write_instruction_3op(
        &mut self,
        operands: &OperandArena,
        dest: Option<OperandIndex>,
        src1: Option<OperandIndex>,
        src2: Option<OperandIndex>,
        inst: X86Instruction,
        typ: OperandType,
    ) {
        let info = &assembler_table::INST_TABLE[inst as usize];
        let src2_kind = Self::kind(operands, src2);

        if src2_kind != OperandKind::None {
            match src2_kind {
                OperandKind::Constant => {
                    let imm = src2.map(|i| operands.get(i).value).unwrap_or(0);
                    if imm as u8 as u64 == imm && info.op_rm_imm8 != BAD_OP {
                        self.write_op_code(
                            operands,
                            dest,
                            src1,
                            None,
                            typ,
                            info.flags,
                            info.op_rm_imm8,
                            false,
                        );
                        self.write_byte(imm as u8);
                    } else {
                        panic!("Failed to encode constant 0x{imm:X} for inst {inst:?}");
                    }
                }
                OperandKind::Register if info.op_rmr != BAD_OP => {
                    self.write_op_code(
                        operands,
                        dest,
                        src1,
                        src2,
                        typ,
                        info.flags,
                        info.op_rmr,
                        false,
                    );
                }
                _ if info.op_rrm != BAD_OP => {
                    self.write_op_code(
                        operands,
                        dest,
                        src1,
                        src2,
                        typ,
                        info.flags,
                        info.op_rrm,
                        true,
                    );
                }
                _ => panic!("Invalid source operand kind {src2_kind:?}"),
            }
        } else if info.op_rrm != BAD_OP {
            self.write_op_code(
                operands,
                dest,
                src1,
                src2,
                typ,
                info.flags,
                info.op_rrm,
                true,
            );
        } else if info.op_rmr != BAD_OP {
            self.write_op_code(
                operands,
                dest,
                src1,
                src2,
                typ,
                info.flags,
                info.op_rmr,
                false,
            );
        } else {
            panic!("Source operand is None and no suitable opcode form");
        }
    }

    // WriteShiftInst (shift/rotate)
    fn write_shift_inst(
        &mut self,
        operands: &OperandArena,
        dest: Option<OperandIndex>,
        source: Option<OperandIndex>,
        typ: OperandType,
        inst: X86Instruction,
    ) {
        let source_kind = Self::kind(operands, source);
        match source_kind {
            OperandKind::Register => {
                // C#: assert shiftReg == RCX (reg_index 1), then shift by CL  no imm byte.
                debug_assert_eq!(
                    operands.get(source.unwrap()).reg_index(),
                    1,
                    "Invalid shift register  only RCX/cl allowed"
                );
                // Emit with source=None → OpRMR (D3 /r) form.
                self.write_instruction_2op(operands, dest, None, typ, inst);
            }
            OperandKind::Constant => {
                let info = &assembler_table::INST_TABLE[inst as usize];
                let raw = operands.get(source.unwrap()).value;
                // C# masks by dest.Type, not the type parameter.
                let dest_typ = operands.get(dest.expect("shift needs a dest")).typ;
                let mask = if dest_typ == OperandType::I32 {
                    0x1f
                } else {
                    0x3f
                };
                let masked_imm = (raw & mask) as u8;
                self.write_op_code(
                    operands,
                    dest,
                    None,
                    None,
                    typ,
                    info.flags,
                    info.op_rm_imm8,
                    false,
                );
                self.write_byte(masked_imm);
            }
            _ => {
                // 1-operand shift (e.g., SAR r/m, 1)  not supported yet.
                panic!("Unsupported shift source kind {source_kind:?}");
            }
        }
    }

    // WriteCompactInst (for Pop/Push register form)
    fn write_compact_inst(
        &mut self,
        operands: &OperandArena,
        reg: OperandIndex,
        opcode: u8,
        _typ: OperandType,
    ) {
        let data = operands.get(reg);
        debug_assert_eq!(data.kind, OperandKind::Register);
        // C#: regIndex >= 8 → WriteByte(0x41) (REX.B only  push/pop default
        // to 64-bit operand size, no REX.W; the old rex_prefix call here
        // emitted REX.R, encoding push r12 as push rsp).
        let reg_index = data.reg_index() as u8;
        if reg_index >= 8 {
            self.write_byte(0x41);
        }
        self.write_byte(opcode + (reg_index & 0b111));
    }

    // Mnemonic layer
    pub fn add(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Add);
    }

    pub fn addsd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Addsd,
            OperandType::None,
        );
    }

    pub fn addss(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Addss,
            OperandType::None,
        );
    }

    pub fn and(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::And);
    }

    pub fn bsr(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Bsr);
    }

    pub fn bswap(&mut self, operands: &OperandArena, dest: OperandIndex, typ: OperandType) {
        self.write_instruction_2op(operands, Some(dest), None, typ, X86Instruction::Bswap);
    }

    pub fn lzcnt(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Lzcnt,
        );
    }

    /// andn dest, src1, src2 = (~src1) & src2 (BMI1, VEX 3-operand GPR).
    /// src1 → VEX.vvvv, src2 → ModRM.r/m, dest → ModRM.reg.
    pub fn andn(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Andn,
            typ,
        );
    }

    // BMI2 variable shifts: dest = value shifted by count. value → ModRM.r/m,
    // count → VEX.vvvv, dest → ModRM.reg. Non-destructive, count from any GPR.
    pub fn shlx(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        value: OperandIndex,
        count: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(count),
            Some(value),
            X86Instruction::Shlx,
            typ,
        );
    }
    pub fn shrx(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        value: OperandIndex,
        count: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(count),
            Some(value),
            X86Instruction::Shrx,
            typ,
        );
    }
    pub fn sarx(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        value: OperandIndex,
        count: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(count),
            Some(value),
            X86Instruction::Sarx,
            typ,
        );
    }

    pub fn call(&mut self, operands: &OperandArena, dest: OperandIndex) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            None,
            OperandType::None,
            X86Instruction::Call,
        );
    }

    pub fn cdq(&mut self) {
        self.write_byte(0x99);
    }

    pub fn cmovcc(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
        condition: X86Condition,
    ) {
        let info = &assembler_table::INST_TABLE[X86Instruction::Cmovcc as usize];
        let op_code = info.op_rrm | condition as u32;
        self.write_op_code(
            operands,
            Some(dest),
            None,
            Some(source),
            typ,
            info.flags,
            op_code,
            true,
        );
    }

    pub fn cmp(
        &mut self,
        operands: &OperandArena,
        src1: OperandIndex,
        src2: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(src1), Some(src2), typ, X86Instruction::Cmp);
    }

    pub fn cqo(&mut self) {
        self.write_byte(0x48);
        self.write_byte(0x99);
    }

    pub fn cmpxchg(&mut self, operands: &OperandArena, mem_op: OperandIndex, src: OperandIndex) {
        self.write_byte(LOCK_PREFIX);
        let typ = operands.get(src).typ;
        self.write_instruction_2op(
            operands,
            Some(mem_op),
            Some(src),
            typ,
            X86Instruction::Cmpxchg,
        );
    }

    pub fn cmpxchg16(&mut self, operands: &OperandArena, mem_op: OperandIndex, src: OperandIndex) {
        self.write_byte(LOCK_PREFIX);
        self.write_byte(0x66);
        let typ = operands.get(src).typ;
        self.write_instruction_2op(
            operands,
            Some(mem_op),
            Some(src),
            typ,
            X86Instruction::Cmpxchg,
        );
    }

    pub fn cmpxchg16b(&mut self, operands: &OperandArena, mem_op: OperandIndex) {
        self.write_byte(LOCK_PREFIX);
        self.write_instruction_2op(
            operands,
            Some(mem_op),
            None,
            OperandType::None,
            X86Instruction::Cmpxchg16b,
        );
    }

    pub fn cmpxchg8(&mut self, operands: &OperandArena, mem_op: OperandIndex, src: OperandIndex) {
        self.write_byte(LOCK_PREFIX);
        let typ = operands.get(src).typ;
        self.write_instruction_2op(
            operands,
            Some(mem_op),
            Some(src),
            typ,
            X86Instruction::Cmpxchg8,
        );
    }

    pub fn comisd(&mut self, operands: &OperandArena, src1: OperandIndex, src2: OperandIndex) {
        self.write_instruction_3op(
            operands,
            Some(src1),
            None,
            Some(src2),
            X86Instruction::Comisd,
            OperandType::None,
        );
    }

    pub fn comiss(&mut self, operands: &OperandArena, src1: OperandIndex, src2: OperandIndex) {
        self.write_instruction_3op(
            operands,
            Some(src1),
            None,
            Some(src2),
            X86Instruction::Comiss,
            OperandType::None,
        );
    }

    pub fn cvtsd2ss(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Cvtsd2ss,
            OperandType::None,
        );
    }

    pub fn cvtsi2sd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Cvtsi2sd,
            typ,
        );
    }

    pub fn cvtsi2ss(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Cvtsi2ss,
            typ,
        );
    }

    pub fn cvtss2sd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Cvtss2sd,
            OperandType::None,
        );
    }

    pub fn div(&mut self, operands: &OperandArena, source: OperandIndex, typ: OperandType) {
        self.write_instruction_2op(operands, None, Some(source), typ, X86Instruction::Div);
    }

    pub fn divsd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Divsd,
            OperandType::None,
        );
    }

    pub fn divss(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Divss,
            OperandType::None,
        );
    }

    pub fn idiv(&mut self, operands: &OperandArena, source: OperandIndex, typ: OperandType) {
        self.write_instruction_2op(operands, None, Some(source), typ, X86Instruction::Idiv);
    }

    pub fn imul1(&mut self, operands: &OperandArena, source: OperandIndex, typ: OperandType) {
        self.write_instruction_2op(operands, None, Some(source), typ, X86Instruction::Imul128);
    }

    pub fn imul2(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        // C# asserts source is register
        let src_data = operands.get(source);
        debug_assert_eq!(src_data.kind, OperandKind::Register);
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Imul,
        );
    }

    pub fn imul3(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        typ: OperandType,
    ) {
        let info = &assembler_table::INST_TABLE[X86Instruction::Imul as usize];
        let src2_data = operands.get(src2);
        debug_assert_eq!(src2_data.kind, OperandKind::Constant);
        let imm = src2_data.value;

        // C# judges immediate fit on the `type` parameter, not src2.Type
        // an I32-typed const in an I64 imul must not take the imm8 form.
        if Self::is_imm8(imm, typ) && info.op_rm_imm8 != BAD_OP {
            self.write_op_code(
                operands,
                Some(dest),
                None,
                Some(src1),
                typ,
                info.flags,
                info.op_rm_imm8,
                true,
            );
            self.write_byte(imm as u8);
        } else if Self::is_imm32(imm, typ) && info.op_rm_imm32 != BAD_OP {
            self.write_op_code(
                operands,
                Some(dest),
                None,
                Some(src1),
                typ,
                info.flags,
                info.op_rm_imm32,
                true,
            );
            self.write_i32(imm as i32);
        } else {
            panic!("Failed to encode constant 0x{imm:X} for imul3");
        }
    }

    pub fn insertps(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Insertps,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn ldmxcsr(&mut self, operands: &OperandArena, dest: OperandIndex) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            None,
            OperandType::I32,
            X86Instruction::Ldmxcsr,
        );
    }

    pub fn lea(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Lea);
    }

    pub fn mfence(&mut self) {
        // MFENCE: 0F AE F0
        self.write_byte(0x0F);
        self.write_byte(0xAE);
        self.write_byte(0xF0);
    }

    pub fn lock_or(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_byte(LOCK_PREFIX);
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Or);
    }

    pub fn mov(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Mov);
    }

    pub fn mov16(&mut self, operands: &OperandArena, dest: OperandIndex, source: OperandIndex) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            OperandType::None,
            X86Instruction::Mov16,
        );
    }

    pub fn mov8(&mut self, operands: &OperandArena, dest: OperandIndex, source: OperandIndex) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            OperandType::None,
            X86Instruction::Mov8,
        );
    }

    pub fn movd(&mut self, operands: &OperandArena, dest: OperandIndex, source: OperandIndex) {
        let info = &assembler_table::INST_TABLE[X86Instruction::Movd as usize];
        let src_data = operands.get(source);

        if src_data.typ.is_integer() || src_data.kind == OperandKind::Memory {
            self.write_op_code(
                operands,
                Some(dest),
                None,
                Some(source),
                OperandType::None,
                info.flags,
                info.op_rrm,
                true,
            );
        } else {
            self.write_op_code(
                operands,
                Some(dest),
                None,
                Some(source),
                OperandType::None,
                info.flags,
                info.op_rmr,
                false,
            );
        }
    }

    pub fn movdqu(&mut self, operands: &OperandArena, dest: OperandIndex, source: OperandIndex) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            None,
            Some(source),
            X86Instruction::Movdqu,
            OperandType::None,
        );
    }

    pub fn movhlps(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Movhlps,
            OperandType::None,
        );
    }

    pub fn movlhps(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Movlhps,
            OperandType::None,
        );
    }

    pub fn movq(&mut self, operands: &OperandArena, dest: OperandIndex, source: OperandIndex) {
        let info = &assembler_table::INST_TABLE[X86Instruction::Movd as usize];
        let flags = info.flags | flags::REX_W;
        let src_data = operands.get(source);
        let dest_data = operands.get(dest);

        if src_data.typ.is_integer() || src_data.kind == OperandKind::Memory {
            self.write_op_code(
                operands,
                Some(dest),
                None,
                Some(source),
                OperandType::None,
                flags,
                info.op_rrm,
                true,
            );
        } else if dest_data.typ.is_integer() || dest_data.kind == OperandKind::Memory {
            self.write_op_code(
                operands,
                Some(dest),
                None,
                Some(source),
                OperandType::None,
                flags,
                info.op_rmr,
                false,
            );
        } else {
            self.write_instruction_3op(
                operands,
                Some(dest),
                None,
                Some(source),
                X86Instruction::Movq,
                OperandType::None,
            );
        }
    }

    pub fn movsd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Movsd,
            OperandType::None,
        );
    }

    pub fn movss(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Movss,
            OperandType::None,
        );
    }

    pub fn movsx16(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Movsx16,
        );
    }

    pub fn movsx32(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Movsx32,
        );
    }

    pub fn movsx8(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Movsx8,
        );
    }

    pub fn movzx16(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Movzx16,
        );
    }

    pub fn movzx8(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Movzx8,
        );
    }

    pub fn mul(&mut self, operands: &OperandArena, source: OperandIndex, typ: OperandType) {
        self.write_instruction_2op(operands, None, Some(source), typ, X86Instruction::Mul128);
    }

    pub fn mulsd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Mulsd,
            OperandType::None,
        );
    }

    pub fn mulss(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Mulss,
            OperandType::None,
        );
    }

    pub fn neg(&mut self, operands: &OperandArena, dest: OperandIndex, typ: OperandType) {
        self.write_instruction_2op(operands, Some(dest), None, typ, X86Instruction::Neg);
    }

    pub fn not(&mut self, operands: &OperandArena, dest: OperandIndex, typ: OperandType) {
        self.write_instruction_2op(operands, Some(dest), None, typ, X86Instruction::Not);
    }

    pub fn or(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Or);
    }

    pub fn pclmulqdq(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            None,
            Some(source),
            X86Instruction::Pclmulqdq,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pcmpeqw(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Pcmpeqw,
            OperandType::None,
        );
    }

    pub fn pextrb(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            None,
            Some(source),
            X86Instruction::Pextrb,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pextrd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            None,
            Some(source),
            X86Instruction::Pextrd,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pextrq(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            None,
            Some(source),
            X86Instruction::Pextrq,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pextrw(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            None,
            Some(source),
            X86Instruction::Pextrw,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pinsrb(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Pinsrb,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pinsrd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Pinsrd,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pinsrq(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Pinsrq,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pinsrw(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Pinsrw,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn pop(&mut self, operands: &OperandArena, dest: OperandIndex, typ: OperandType) {
        let data = operands.get(dest);
        if data.kind == OperandKind::Register {
            self.write_compact_inst(operands, dest, 0x58, typ);
        } else {
            self.write_instruction_2op(operands, Some(dest), None, typ, X86Instruction::Pop);
        }
    }

    pub fn popcnt(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            Some(source),
            typ,
            X86Instruction::Popcnt,
        );
    }

    pub fn pshufd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        imm: u8,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            None,
            Some(source),
            X86Instruction::Pshufd,
            OperandType::None,
        );
        self.write_byte(imm);
    }

    pub fn push(&mut self, operands: &OperandArena, source: OperandIndex, typ: OperandType) {
        let data = operands.get(source);
        if data.kind == OperandKind::Register {
            self.write_compact_inst(operands, source, 0x50, typ);
        } else {
            self.write_instruction_2op(operands, None, Some(source), typ, X86Instruction::Push);
        }
    }

    pub fn r#return(&mut self) {
        self.write_byte(0xc3);
    }

    pub fn ror(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_shift_inst(operands, Some(dest), Some(source), typ, X86Instruction::Ror);
    }

    pub fn sar(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_shift_inst(operands, Some(dest), Some(source), typ, X86Instruction::Sar);
    }

    pub fn setcc(&mut self, operands: &OperandArena, dest: OperandIndex, condition: X86Condition) {
        let info = &assembler_table::INST_TABLE[X86Instruction::Setcc as usize];
        let op_code = info.op_rrm | condition as u32;
        self.write_op_code(
            operands,
            Some(dest),
            None,
            None,
            OperandType::None,
            info.flags,
            op_code,
            false,
        );
    }

    pub fn shl(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_shift_inst(operands, Some(dest), Some(source), typ, X86Instruction::Shl);
    }

    pub fn shr(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_shift_inst(operands, Some(dest), Some(source), typ, X86Instruction::Shr);
    }

    pub fn stmxcsr(&mut self, operands: &OperandArena, dest: OperandIndex) {
        self.write_instruction_2op(
            operands,
            Some(dest),
            None,
            OperandType::I32,
            X86Instruction::Stmxcsr,
        );
    }

    pub fn sub(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Sub);
    }

    pub fn subsd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Subsd,
            OperandType::None,
        );
    }

    pub fn subss(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Subss,
            OperandType::None,
        );
    }

    pub fn test(
        &mut self,
        operands: &OperandArena,
        src1: OperandIndex,
        src2: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(src1), Some(src2), typ, X86Instruction::Test);
    }

    pub fn xor(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        source: OperandIndex,
        typ: OperandType,
    ) {
        self.write_instruction_2op(operands, Some(dest), Some(source), typ, X86Instruction::Xor);
    }

    pub fn xorpd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Xorpd,
            OperandType::None,
        );
    }

    pub fn xorps(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Xorps,
            OperandType::None,
        );
    }

    pub fn addpd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Addpd,
            OperandType::None,
        );
    }

    pub fn addps(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Addps,
            OperandType::None,
        );
    }

    pub fn subpd(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Subpd,
            OperandType::None,
        );
    }

    pub fn subps(
        &mut self,
        operands: &OperandArena,
        dest: OperandIndex,
        src1: OperandIndex,
        src2: OperandIndex,
    ) {
        self.write_instruction_3op(
            operands,
            Some(dest),
            Some(src1),
            Some(src2),
            X86Instruction::Subps,
            OperandType::None,
        );
    }

    // ── Public generic forms (C# WriteInstruction overloads)  used by
    //    the intrinsic (Extended) codegen arm ─────────────────────────────

    /// C# WriteInstruction(inst, dest, source, type = None).
    pub fn write_instruction_2(
        &mut self,
        operands: &OperandArena,
        inst: X86Instruction,
        dest: Option<OperandIndex>,
        source: Option<OperandIndex>,
        typ: OperandType,
    ) {
        self.write_instruction_3op(operands, dest, None, source, inst, typ);
    }

    /// C# WriteInstruction(inst, dest, src1, src2). Constant src2 swaps
    /// dest/src1 into the private form  that is the VEX shift-by-imm
    /// encoding (vvvv = dest, rm = src1).
    pub fn write_instruction_3(
        &mut self,
        operands: &OperandArena,
        inst: X86Instruction,
        dest: Option<OperandIndex>,
        src1: Option<OperandIndex>,
        src2: Option<OperandIndex>,
    ) {
        let src2_is_const = src2.is_some_and(|s| operands.get(s).kind == OperandKind::Constant);
        if src2_is_const {
            self.write_instruction_3op(operands, src1, dest, src2, inst, OperandType::None);
        } else {
            self.write_instruction_3op(operands, dest, src1, src2, inst, OperandType::None);
        }
    }

    /// C# WriteInstruction(inst, dest, src1, src2, type).
    pub fn write_instruction_3t(
        &mut self,
        operands: &OperandArena,
        inst: X86Instruction,
        dest: Option<OperandIndex>,
        src1: Option<OperandIndex>,
        src2: Option<OperandIndex>,
        typ: OperandType,
    ) {
        self.write_instruction_3op(operands, dest, src1, src2, inst, typ);
    }

    /// C# WriteInstruction(inst, dest, source, byte imm).
    pub fn write_instruction_2imm(
        &mut self,
        operands: &OperandArena,
        inst: X86Instruction,
        dest: Option<OperandIndex>,
        source: Option<OperandIndex>,
        imm: u8,
    ) {
        self.write_instruction_3op(operands, dest, None, source, inst, OperandType::None);
        self.write_byte(imm);
    }

    /// C# WriteInstruction(inst, dest, src1, src2, src3)  VEX-encoded
    /// 4-operand form (blend register in imm[7:4]).
    pub fn write_instruction_4(
        &mut self,
        operands: &OperandArena,
        inst: X86Instruction,
        dest: Option<OperandIndex>,
        src1: Option<OperandIndex>,
        src2: Option<OperandIndex>,
        src3: OperandIndex,
    ) {
        debug_assert!(capabilities::supports_vex_encoding());
        self.write_instruction_3op(operands, dest, src1, src2, inst, OperandType::None);
        let reg = operands.get(src3).reg_index() as u8;
        self.write_byte(reg << 4);
    }

    /// C# WriteInstruction(inst, dest, src1, src2, byte imm).
    pub fn write_instruction_3imm(
        &mut self,
        operands: &OperandArena,
        inst: X86Instruction,
        dest: Option<OperandIndex>,
        src1: Option<OperandIndex>,
        src2: Option<OperandIndex>,
        imm: u8,
    ) {
        self.write_instruction_3op(operands, dest, src1, src2, inst, OperandType::None);
        self.write_byte(imm);
    }

    // Jumps (label form  records jump for later resolution)
    pub fn jmp(&mut self, operands: &OperandArena, target: OperandIndex) {
        let data = operands.get(target);
        match data.kind {
            OperandKind::Label => {
                let pos = self.stream.len() as u64;
                self.write_byte(0); // placeholder (ReservedBytesForJump)
                self.jumps.push(Jump::new(target, pos));
            }
            _ => {
                // register or memory target  emit through write_op_code
                let info = &assembler_table::INST_TABLE[X86Instruction::Jmp as usize];
                self.write_op_code(
                    operands,
                    Some(target),
                    None,
                    None,
                    OperandType::None,
                    info.flags,
                    info.op_rmr,
                    false,
                );
            }
        }
    }

    pub fn jcc(&mut self, condition: X86Condition, operands: &OperandArena, target: OperandIndex) {
        let data = operands.get(target);
        match data.kind {
            OperandKind::Label => {
                let pos = self.stream.len() as u64;
                self.write_byte(0); // placeholder
                self.jumps
                    .push(Jump::new_conditional(condition, target, pos));
            }
            _ => {
                panic!("Jcc destination must be a label operand");
            }
        }
    }

    // Direct offset jumps (used by GetCode during re-emission)
    pub fn jmp_offset(&mut self, offset: i64) {
        if Self::const_fits_on_s8(offset) {
            self.write_byte(0xeb);
            self.write_byte(offset as u8);
        } else {
            self.write_byte(0xe9);
            self.write_i32(offset as i32);
        }
    }

    pub fn jcc_offset(&mut self, condition: X86Condition, offset: i64) {
        let opcode_base: u8 = 0x80;
        if Self::const_fits_on_s8(offset) {
            self.write_byte(0x70 | (condition as u8));
            self.write_byte(offset as u8);
        } else {
            self.write_byte(0x0f);
            self.write_byte(opcode_base | (condition as u8));
            self.write_i32(offset as i32);
        }
    }

    // GetCode: resolve jumps, re-emit final binary
    pub fn get_code(&mut self, _operands: &OperandArena) -> (Vec<u8>, RelocationInfo) {
        // Fixpoint offset resolution
        // No early return for jumps_len == 0: C# still flushes reloc
        // entries (offset 0) and copies the stream through the same path.
        let jumps_len = self.jumps.len();

        let mut modified = true;
        while modified {
            modified = false;

            for i in 0..jumps_len {
                // C#: jump.JumpTarget ??= _labels[jump.JumpLabel]
                if self.jumps[i].jump_target.is_none() {
                    self.jumps[i].jump_target = self.labels.get(&self.jumps[i].jump_label).copied();
                }

                let jump_target = self.jumps[i].jump_target.expect("Jump label not marked");
                let jump_position = self.jumps[i].jump_position;
                let is_conditional = self.jumps[i].is_conditional;

                let mut offset = jump_target as i64 - jump_position as i64;

                if offset < 0 {
                    // Backward: subtract sizes of jumps past the target.
                    for j in (0..i).rev() {
                        let jump2 = &self.jumps[j];
                        if jump2.jump_position < jump_target {
                            break;
                        }
                        offset -= jump2.inst_size as i64 - RESERVED_BYTES_FOR_JUMP;
                    }
                } else {
                    // Forward: add sizes of jumps up to the target.
                    for j in i + 1..jumps_len {
                        let jump2 = &self.jumps[j];
                        if jump2.jump_position >= jump_target {
                            break;
                        }
                        offset += jump2.inst_size as i64 - RESERVED_BYTES_FOR_JUMP;
                    }
                    offset -= RESERVED_BYTES_FOR_JUMP;
                }

                // C# sets InstSize unconditionally each pass; only Offset
                // drives the fixpoint. (A zero-offset jump would otherwise
                // never get its size assigned.)
                let new_inst_size = if is_conditional {
                    Self::get_jcc_length(offset)
                } else {
                    Self::get_jmp_length(offset)
                };
                self.jumps[i].inst_size = new_inst_size;

                // C#: for backward jumps, offset -= inst_size
                // (relative to next instruction, not current)
                if offset < 0 {
                    offset -= new_inst_size as i64;
                }

                if self.jumps[i].offset != offset {
                    self.jumps[i].offset = offset;
                    modified = true;
                }
            }
        }

        // Re-emission
        // Swap out the stream, read from the old one, write resolved
        // code into self.stream.
        let old_stream = std::mem::take(&mut self.stream);

        let has_relocs = self.relocs.is_some();
        let mut reloc_entries: Vec<RelocationEntry> = Vec::new();
        let mut reloc_idx = 0usize;
        let mut reloc_offset = 0i32;
        let mut read_pos = 0usize;

        for i in 0..jumps_len {
            // Relocation adjustment
            if has_relocs {
                reloc_offset += self.jumps[i].inst_size as i32 - RESERVED_BYTES_FOR_JUMP as i32;
                if let Some(ref relocs) = self.relocs {
                    while reloc_idx < relocs.len() {
                        if relocs[reloc_idx].jump_index > i as i32 {
                            break;
                        }
                        reloc_entries.push(RelocationEntry::new(
                            relocs[reloc_idx].position + reloc_offset,
                            relocs[reloc_idx].symbol,
                        ));
                        reloc_idx += 1;
                    }
                }
            }

            // Copy bytes up to jump position
            let jump_pos = self.jumps[i].jump_position as usize;
            let copy_len = jump_pos - read_pos;
            self.stream
                .extend_from_slice(&old_stream[read_pos..read_pos + copy_len]);
            read_pos = jump_pos + RESERVED_BYTES_FOR_JUMP as usize;

            // Emit resolved jump
            if self.jumps[i].is_conditional {
                self.jcc_offset(self.jumps[i].condition, self.jumps[i].offset);
            } else {
                self.jmp_offset(self.jumps[i].offset);
            }
        }

        // Trailing bytes after last jump
        self.stream.extend_from_slice(&old_stream[read_pos..]);

        // Remaining relocs (no-jump case)
        if has_relocs {
            if let Some(ref relocs) = self.relocs {
                while reloc_idx < relocs.len() {
                    reloc_entries.push(RelocationEntry::new(
                        relocs[reloc_idx].position + reloc_offset,
                        relocs[reloc_idx].symbol,
                    ));
                    reloc_idx += 1;
                }
            }
        }

        let result = std::mem::take(&mut self.stream);
        let reloc_info = if has_relocs {
            RelocationInfo::new(reloc_entries)
        } else {
            RelocationInfo::empty()
        };

        (result, reloc_info)
    }
}
