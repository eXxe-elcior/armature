use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Instruction, OperandKind, OperandType};

pub fn run_pass(ctx: &mut TranslationContext, node: OperationIdx) {
    let srcs = ctx.ops.sources(node);
    if srcs.is_empty() {
        return;
    }

    match ctx.ops.get(node).instruction {
        Instruction::Add => {
            if is_relocatable(ctx, srcs[0]) || is_relocatable(ctx, srcs[1]) {
                return;
            }
            try_eliminate_binary_op_commutative(ctx, node, 0);
        }

        Instruction::BitwiseAnd => {
            try_eliminate_bitwise_and(ctx, node);
        }

        Instruction::BitwiseOr => {
            try_eliminate_bitwise_or(ctx, node);
        }

        Instruction::BitwiseExclusiveOr => {
            try_eliminate_bitwise_exclusive_or(ctx, node);
        }

        Instruction::ConditionalSelect => {
            try_eliminate_conditional_select(ctx, node);
        }

        Instruction::Divide => {
            try_eliminate_binary_op_y(ctx, node, 1);
        }

        Instruction::DivideUI => {
            // x / 2^k == x >>u k (unsigned only; signed Divide rounds toward
            // zero, not floor, so it is NOT a shift).
            if !try_strength_reduce_div_ui(ctx, node) {
                try_eliminate_binary_op_y(ctx, node, 1);
            }
        }

        Instruction::Multiply => {
            let x = srcs[0];
            let y = srcs[1];
            // x * 0 == 0  integer only. FP would mishandle NaN/inf (NaN*0,
            // inf*0 = NaN) and signed zero, so it is NOT a valid fold there.
            if ctx.operands.get(x).typ.is_integer()
                && (is_const_equal(ctx, x, 0) || is_const_equal(ctx, y, 0))
            {
                let zero = constant_zero(ctx, ctx.operands.get(x).typ);
                ctx.ops.turn_into_copy(node, zero, &mut ctx.operands);
            } else if !try_strength_reduce_mul(ctx, node) {
                // x * 2^k == x << k (low bits identical signed/unsigned).
                try_eliminate_binary_op_commutative(ctx, node, 1);
            }
        }

        Instruction::ShiftLeft
        | Instruction::ShiftRightSI
        | Instruction::ShiftRightUI
        | Instruction::Subtract => {
            try_eliminate_binary_op_y(ctx, node, 0);
        }

        _ => {}
    }
}

fn try_eliminate_bitwise_and(ctx: &mut TranslationContext, node: OperationIdx) {
    let srcs = ctx.ops.sources(node);
    let x = srcs[0];
    let y = srcs[1];

    if is_const_equal(ctx, x, all_ones(ctx.operands.get(x).typ)) {
        ctx.ops.turn_into_copy(node, y, &mut ctx.operands);
    } else if is_const_equal(ctx, y, all_ones(ctx.operands.get(y).typ)) {
        ctx.ops.turn_into_copy(node, x, &mut ctx.operands);
    } else if is_const_equal(ctx, x, 0) || is_const_equal(ctx, y, 0) {
        let zero = constant_zero(ctx, ctx.operands.get(x).typ);
        ctx.ops.turn_into_copy(node, zero, &mut ctx.operands);
    }
}

fn try_eliminate_bitwise_or(ctx: &mut TranslationContext, node: OperationIdx) {
    let srcs = ctx.ops.sources(node);
    let x = srcs[0];
    let y = srcs[1];

    if is_const_equal(ctx, x, 0) {
        ctx.ops.turn_into_copy(node, y, &mut ctx.operands);
    } else if is_const_equal(ctx, y, 0) {
        ctx.ops.turn_into_copy(node, x, &mut ctx.operands);
    } else if is_const_equal(ctx, x, all_ones(ctx.operands.get(x).typ))
        || is_const_equal(ctx, y, all_ones(ctx.operands.get(y).typ))
    {
        let ones = constant_all_ones(ctx, ctx.operands.get(x).typ);
        ctx.ops.turn_into_copy(node, ones, &mut ctx.operands);
    }
}

fn try_eliminate_bitwise_exclusive_or(ctx: &mut TranslationContext, node: OperationIdx) {
    let srcs = ctx.ops.sources(node);
    let x = srcs[0];
    let y = srcs[1];

    if x == y && ctx.operands.get(x).typ.is_integer() {
        let zero = constant_zero(ctx, ctx.operands.get(x).typ);
        ctx.ops.turn_into_copy(node, zero, &mut ctx.operands);
    } else {
        try_eliminate_binary_op_commutative(ctx, node, 0);
    }
}

fn try_eliminate_binary_op_y(ctx: &mut TranslationContext, node: OperationIdx, comparand: u64) {
    let srcs = ctx.ops.sources(node);
    let y = srcs[1];

    if is_const_equal(ctx, y, comparand) {
        ctx.ops.turn_into_copy(node, srcs[0], &mut ctx.operands);
    }
}

fn try_eliminate_binary_op_commutative(
    ctx: &mut TranslationContext,
    node: OperationIdx,
    comparand: u64,
) {
    let srcs = ctx.ops.sources(node);
    let x = srcs[0];
    let y = srcs[1];

    if is_const_equal(ctx, x, comparand) {
        ctx.ops.turn_into_copy(node, y, &mut ctx.operands);
    } else if is_const_equal(ctx, y, comparand) {
        ctx.ops.turn_into_copy(node, x, &mut ctx.operands);
    }
}

fn try_eliminate_conditional_select(ctx: &mut TranslationContext, node: OperationIdx) {
    let srcs = ctx.ops.sources(node);
    let cond = srcs[0];

    if ctx.operands.get(cond).kind != OperandKind::Constant {
        return;
    }

    let src_index = if ctx.operands.get(cond).value != 0 {
        1
    } else {
        2
    };
    let source = srcs[src_index];

    ctx.ops.turn_into_copy(node, source, &mut ctx.operands);
}

/// `x * 2^k` -> `x << k`. Commutative: either operand may be the constant.
fn try_strength_reduce_mul(ctx: &mut TranslationContext, node: OperationIdx) -> bool {
    let srcs = ctx.ops.sources(node);
    let (x, y) = (srcs[0], srcs[1]);
    let (value, k) = if let Some(k) = const_pow2_log2(ctx, y) {
        (x, k)
    } else if let Some(k) = const_pow2_log2(ctx, x) {
        (y, k)
    } else {
        return false;
    };
    let kc = ctx.operands.constant_i32(k as i32);
    ctx.ops
        .turn_into_binary_op(node, Instruction::ShiftLeft, value, kc, &mut ctx.operands);
    true
}

/// `x /u 2^k` -> `x >>u k` (unsigned division only).
fn try_strength_reduce_div_ui(ctx: &mut TranslationContext, node: OperationIdx) -> bool {
    let srcs = ctx.ops.sources(node);
    let (x, y) = (srcs[0], srcs[1]);
    let Some(k) = const_pow2_log2(ctx, y) else {
        return false;
    };
    let kc = ctx.operands.constant_i32(k as i32);
    ctx.ops
        .turn_into_binary_op(node, Instruction::ShiftRightUI, x, kc, &mut ctx.operands);
    true
}

/// If `operand` is a non-relocatable integer constant equal to 2^k with k>=1,
/// return k. (k==0 i.e. value 1 is left to the identity simplification.)
fn const_pow2_log2(ctx: &TranslationContext, operand: OperandIndex) -> Option<u32> {
    let d = ctx.operands.get(operand);
    if d.kind != OperandKind::Constant
        || !d.typ.is_integer()
        || d.symbol != crate::ir::types::Symbol::default()
    {
        return None;
    }
    let v = match d.typ {
        OperandType::I32 => d.value & 0xFFFF_FFFF,
        _ => d.value,
    };
    if v > 1 && (v & (v - 1)) == 0 {
        Some(v.trailing_zeros())
    } else {
        None
    }
}

fn is_const_equal(ctx: &TranslationContext, operand: OperandIndex, comparand: u64) -> bool {
    let data = ctx.operands.get(operand);
    data.kind == OperandKind::Constant && data.typ.is_integer() && data.value == comparand
}

fn all_ones(typ: OperandType) -> u64 {
    match typ {
        OperandType::I32 => 0xFFFF_FFFF,
        OperandType::I64 => 0xFFFF_FFFF_FFFF_FFFF,
        _ => panic!("Invalid operand type for AllOnes"),
    }
}

fn constant_zero(ctx: &mut TranslationContext, typ: OperandType) -> OperandIndex {
    ctx.const_zero(typ)
}

fn constant_all_ones(ctx: &mut TranslationContext, typ: OperandType) -> OperandIndex {
    match typ {
        OperandType::I32 => ctx.operands.constant_i32(-1),
        OperandType::I64 => ctx.operands.constant_i64(-1),
        _ => panic!("Invalid operand type for AllOnes"),
    }
}

fn is_relocatable(ctx: &TranslationContext, operand: OperandIndex) -> bool {
    ctx.operands.get(operand).symbol != crate::ir::types::Symbol::default()
}
