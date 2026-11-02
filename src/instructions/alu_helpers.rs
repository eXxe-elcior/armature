use crate::decoder::opcode_kind::OpCodeKind;
use crate::decoder::utilities::{DataOperation, IntegerType, ShiftType};
use crate::ir::operand::OperandIndex;
use crate::ir::types::{OperandType, RegisterType};
use crate::registers::GpRegisterIndex;
use crate::state::PState;

use super::context::ArmEmitterCtx;
use super::helpers::{get_int_or_sp, get_int_or_zr, set_flag};

// Flag checks matching C# InstEmitAluHelper
pub fn emit_nz_flags_check(ctx: &mut ArmEmitterCtx, d: OperandIndex) {
    let typ = ctx.ctx.operands.get(d).typ;
    let zero = ctx.ctx.operands.const_zero(typ);
    let n_cmp = ctx.icompare_less(d, zero);
    set_flag(ctx, PState::NFlag, n_cmp);
    let z_cmp = ctx.icompare_equal(d, zero);
    set_flag(ctx, PState::ZFlag, z_cmp);
}

pub fn emit_adcs_c_check(ctx: &mut ArmEmitterCtx, n: OperandIndex, d: OperandIndex) {
    let c_in =
        ctx.ctx
            .operands
            .register(PState::CFlag as u32, RegisterType::Flag, OperandType::I32);
    let eq = ctx.icompare_equal(d, n);
    let lt = ctx.icompare_less_ui(d, n);
    let and = ctx.bitwise_and(eq, c_in);
    let c_out = ctx.bitwise_or(and, lt);
    set_flag(ctx, PState::CFlag, c_out);
}

pub fn emit_adds_c_check(ctx: &mut ArmEmitterCtx, n: OperandIndex, d: OperandIndex) {
    let cmp = ctx.icompare_less_ui(d, n);
    set_flag(ctx, PState::CFlag, cmp);
}

pub fn emit_adds_v_check(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    m: OperandIndex,
    d: OperandIndex,
) {
    let xor1 = ctx.bitwise_xor(d, n);
    let xor2 = ctx.bitwise_xor(n, m);
    let not_xor2 = ctx.bitwise_not(xor2);
    let v_out = ctx.bitwise_and(xor1, not_xor2);
    let typ = ctx.ctx.operands.get(v_out).typ;
    let zero = ctx.ctx.operands.const_zero(typ);
    let cmp = ctx.icompare_less(v_out, zero);
    set_flag(ctx, PState::VFlag, cmp);
}

pub fn emit_sbcs_c_check(ctx: &mut ArmEmitterCtx, n: OperandIndex, m: OperandIndex) {
    let c_in =
        ctx.ctx
            .operands
            .register(PState::CFlag as u32, RegisterType::Flag, OperandType::I32);
    let eq = ctx.icompare_equal(n, m);
    let gt = ctx.icompare_greater_ui(n, m);
    let and = ctx.bitwise_and(eq, c_in);
    let c_out = ctx.bitwise_or(and, gt);
    set_flag(ctx, PState::CFlag, c_out);
}

pub fn emit_subs_c_check(ctx: &mut ArmEmitterCtx, n: OperandIndex, m: OperandIndex) {
    let cmp = ctx.icompare_greater_or_equal_ui(n, m);
    set_flag(ctx, PState::CFlag, cmp);
}

pub fn emit_subs_v_check(
    ctx: &mut ArmEmitterCtx,
    n: OperandIndex,
    m: OperandIndex,
    d: OperandIndex,
) {
    let xor1 = ctx.bitwise_xor(d, n);
    let xor2 = ctx.bitwise_xor(n, m);
    let v_out = ctx.bitwise_and(xor1, xor2);
    let typ = ctx.ctx.operands.get(v_out).typ;
    let zero = ctx.ctx.operands.const_zero(typ);
    let cmp = ctx.icompare_less(v_out, zero);
    set_flag(ctx, PState::VFlag, cmp);
}

// Operand extraction helpers
pub fn get_alu_n(ctx: &mut ArmEmitterCtx) -> OperandIndex {
    let (rn, data_op) = get_alu_rn_and_data_op(ctx);
    if data_op == DataOperation::Logical || is_alu_rs(ctx) {
        get_int_or_zr(ctx, rn)
    } else {
        get_int_or_sp(ctx, rn)
    }
}

pub fn get_alu_m(ctx: &mut ArmEmitterCtx) -> OperandIndex {
    match ctx.curr_op().kind {
        OpCodeKind::AluImm { immediate, .. } => {
            let typ = type_from_reg_size(ctx);
            match typ {
                OperandType::I32 => ctx.ctx.operands.constant_i32(immediate as i32),
                OperandType::I64 => ctx.ctx.operands.constant_i64(immediate),
                _ => unreachable!(),
            }
        }
        OpCodeKind::AluRs {
            rm,
            shift,
            shift_type,
            ..
        } => {
            let mut value = get_int_or_zr(ctx, rm);
            if shift > 0 {
                let shift_op = ctx.ctx.operands.constant_i32(shift as i32);
                value = match shift_type {
                    ShiftType::Lsl => ctx.shift_left(value, shift_op),
                    ShiftType::Lsr => ctx.shift_right_ui(value, shift_op),
                    ShiftType::Asr => ctx.shift_right_si(value, shift_op),
                    ShiftType::Ror => ctx.rotate_right(value, shift_op),
                };
            }
            value
        }
        OpCodeKind::AluRx {
            rm,
            shift,
            int_type,
            ..
        } => {
            let mut value = get_extended_m_helper(ctx, rm, int_type);
            if shift > 0 {
                let shift_op = ctx.ctx.operands.constant_i32(shift as i32);
                value = ctx.shift_left(value, shift_op);
            }
            value
        }
        OpCodeKind::AluBinary { rm, .. } => get_int_or_zr(ctx, rm),
        // C# dispatch: OpCodeCcmpImm is IOpCodeAluImm (Immediate => RmImm),
        // OpCodeCcmpReg is IOpCodeAluRs (Rm => RmImm, Shift = 0, Lsl)
        // i.e. typed constant for the imm form, plain register (shift 0,
        // a no-op we skip) for the reg form.
        OpCodeKind::Ccmp {
            rm_or_imm, is_imm, ..
        } => {
            if is_imm {
                let typ = type_from_reg_size(ctx);
                ctx.ctx.operands.constant_int(typ, rm_or_imm as i64)
            } else {
                get_int_or_zr(ctx, GpRegisterIndex::from(rm_or_imm))
            }
        }
        _ => panic!("GetAluM called on non-ALU opcode: {:?}", ctx.curr_op().kind),
    }
}

pub fn get_alu_m_shift(ctx: &mut ArmEmitterCtx) -> OperandIndex {
    let (rm, _) = get_alu_rs_rm_and_data_op(ctx);
    let mut m = get_int_or_zr(ctx, rm);
    if ctx.curr_op().reg_size == crate::decoder::RegisterSize::Int64 {
        m = ctx.convert_i64_to_i32(m);
    }
    let mask = ctx
        .ctx
        .operands
        .constant_i32(ctx.curr_op().bits() as i32 - 1);
    ctx.bitwise_and(m, mask)
}

pub fn set_alu_d(ctx: &mut ArmEmitterCtx, d: OperandIndex) {
    set_alu_d_inner(ctx, d, false);
}

pub fn set_alu_d_or_zr(ctx: &mut ArmEmitterCtx, d: OperandIndex) {
    set_alu_d_inner(ctx, d, true);
}

// Reverse bits/bytes helpers
pub fn emit_reverse_bits_32_op(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    let c_aa = ctx.ctx.operands.constant_i32(0xaaaaaaaau32 as i32);
    let c_55 = ctx.ctx.operands.constant_i32(0x55555555);
    let c_1 = ctx.ctx.operands.constant_i32(1);
    let c_cc = ctx.ctx.operands.constant_i32(0xccccccccu32 as i32);
    let c_33 = ctx.ctx.operands.constant_i32(0x33333333);
    let c_2 = ctx.ctx.operands.constant_i32(2);
    let c_f0 = ctx.ctx.operands.constant_i32(0xf0f0f0f0u32 as i32);
    let c_0f = ctx.ctx.operands.constant_i32(0x0f0f0f0f);
    let c_4 = ctx.ctx.operands.constant_i32(4);
    let c_ff00 = ctx.ctx.operands.constant_i32(0xff00ff00u32 as i32);
    let c_00ff = ctx.ctx.operands.constant_i32(0x00ff00ff);
    let c_8 = ctx.ctx.operands.constant_i32(8);
    let c_16 = ctx.ctx.operands.constant_i32(16);

    let and1 = ctx.bitwise_and(op, c_aa);
    let shr1 = ctx.shift_right_ui(and1, c_1);
    let and2 = ctx.bitwise_and(op, c_55);
    let shl1 = ctx.shift_left(and2, c_1);
    let v = ctx.bitwise_or(shr1, shl1);

    let and3 = ctx.bitwise_and(v, c_cc);
    let shr2 = ctx.shift_right_ui(and3, c_2);
    let and4 = ctx.bitwise_and(v, c_33);
    let shl2 = ctx.shift_left(and4, c_2);
    let v = ctx.bitwise_or(shr2, shl2);

    let and5 = ctx.bitwise_and(v, c_f0);
    let shr3 = ctx.shift_right_ui(and5, c_4);
    let and6 = ctx.bitwise_and(v, c_0f);
    let shl3 = ctx.shift_left(and6, c_4);
    let v = ctx.bitwise_or(shr3, shl3);

    let and7 = ctx.bitwise_and(v, c_ff00);
    let shr4 = ctx.shift_right_ui(and7, c_8);
    let and8 = ctx.bitwise_and(v, c_00ff);
    let shl4 = ctx.shift_left(and8, c_8);
    let v = ctx.bitwise_or(shr4, shl4);

    let shr5 = ctx.shift_right_ui(v, c_16);
    let shl5 = ctx.shift_left(v, c_16);
    ctx.bitwise_or(shr5, shl5)
}

pub fn emit_reverse_bytes_16_64_op(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    let c_ff00ff00 = ctx.ctx.operands.constant_i64(0xff00ff00ff00ff00u64 as i64);
    let c_00ff00ff = ctx.ctx.operands.constant_i64(0x00ff00ff00ff00ffu64 as i64);
    let c_8 = ctx.ctx.operands.constant_i32(8);
    let and1 = ctx.bitwise_and(op, c_ff00ff00);
    let a = ctx.shift_right_ui(and1, c_8);
    let and2 = ctx.bitwise_and(op, c_00ff00ff);
    let b = ctx.shift_left(and2, c_8);
    ctx.bitwise_or(a, b)
}

pub fn emit_reverse_bytes_16_32_op(ctx: &mut ArmEmitterCtx, op: OperandIndex) -> OperandIndex {
    let ext = ctx.zero_extend32(OperandType::I64, op);
    let val = emit_reverse_bytes_16_64_op(ctx, ext);
    ctx.convert_i64_to_i32(val)
}

// Internal helpers
fn type_from_reg_size(ctx: &ArmEmitterCtx) -> OperandType {
    match ctx.curr_op().reg_size {
        crate::decoder::RegisterSize::Int32 => OperandType::I32,
        crate::decoder::RegisterSize::Int64 => OperandType::I64,
        _ => panic!(
            "type_from_reg_size: unexpected {:?}",
            ctx.curr_op().reg_size
        ),
    }
}

fn get_alu_rn_and_data_op(ctx: &ArmEmitterCtx) -> (GpRegisterIndex, DataOperation) {
    match ctx.curr_op().kind {
        OpCodeKind::Alu { rn, data_op, .. } => (rn, data_op),
        OpCodeKind::AluImm { rn, data_op, .. } => (rn, data_op),
        OpCodeKind::AluRs { rn, data_op, .. } => (rn, data_op),
        OpCodeKind::AluRx { rn, data_op, .. } => (rn, data_op),
        OpCodeKind::AluBinary { rn, data_op, .. } => (rn, data_op),
        // OpCodeCcmp : OpCodeAlu  DataOp comes from base extraction
        // (raw >> 24) & 3. Both Ccmp patterns fix bits 25:24 = 10 →
        // DataOp::Logical, so GetAluN takes the ZR path. Constant here
        // because OpCodeKind::Ccmp doesn't carry data_op.
        OpCodeKind::Ccmp { rn, .. } => (rn, DataOperation::Logical),
        _ => panic!("get_alu_rn: non-ALU opcode {:?}", ctx.curr_op().kind),
    }
}

fn is_alu_rs(ctx: &ArmEmitterCtx) -> bool {
    matches!(ctx.curr_op().kind, OpCodeKind::AluRs { .. })
}

fn get_alu_rs_rm_and_data_op(ctx: &ArmEmitterCtx) -> (GpRegisterIndex, DataOperation) {
    match ctx.curr_op().kind {
        OpCodeKind::AluRs { rm, data_op, .. } => (rm, data_op),
        _ => panic!("get_alu_rs_rm: expected AluRs opcode"),
    }
}

fn get_extended_m_helper(
    ctx: &mut ArmEmitterCtx,
    rm: GpRegisterIndex,
    int_type: IntegerType,
) -> OperandIndex {
    let value = get_int_or_zr(ctx, rm);
    let typ = ctx.ctx.operands.get(value).typ;
    match int_type {
        IntegerType::Uxtb => ctx.zero_extend8(typ, value),
        IntegerType::Uxth => ctx.zero_extend16(typ, value),
        IntegerType::Uxtw => ctx.zero_extend32(typ, value),
        IntegerType::Uxtx => value,
        IntegerType::Sxtb => ctx.sign_extend8(typ, value),
        IntegerType::Sxth => ctx.sign_extend16(typ, value),
        IntegerType::Sxtw => ctx.sign_extend32(typ, value),
        IntegerType::Sxtx => value,
    }
}

fn set_alu_d_inner(ctx: &mut ArmEmitterCtx, d: OperandIndex, x31_is_zr: bool) {
    let kind = ctx.curr_op().kind;
    let rd = match kind {
        OpCodeKind::Alu { rd, .. }
        | OpCodeKind::AluImm { rd, .. }
        | OpCodeKind::AluRs { rd, .. }
        | OpCodeKind::AluRx { rd, .. }
        | OpCodeKind::AluBinary { rd, .. } => rd,
        _ => panic!("set_alu_d: non-ALU opcode {:?}", kind),
    };
    if (x31_is_zr || matches!(kind, OpCodeKind::AluRs { .. })) && rd.is_zr_or_sp() {
        return;
    }
    super::helpers::set_int_or_sp(ctx, rd, d);
}
