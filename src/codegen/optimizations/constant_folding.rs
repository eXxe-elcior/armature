use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{Comparison, OperandType};

pub fn run_pass(ctx: &mut TranslationContext, node: OperationIdx) {
    let dests = ctx.ops.dests(node);
    let srcs = ctx.ops.sources(node);

    if dests.is_empty() || srcs.is_empty() {
        return;
    }

    if !are_all_sources_constant(ctx, node) {
        return;
    }

    let typ = ctx.operands.get(dests[0]).typ;

    match ctx.ops.get(node).instruction {
        crate::ir::types::Instruction::Add => {
            if is_relocatable(ctx, srcs[0]) || is_relocatable(ctx, srcs[1]) {
                return;
            }
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x.wrapping_add(y));
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x.wrapping_add(y));
            }
        }

        crate::ir::types::Instruction::BitwiseAnd => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x & y);
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x & y);
            }
        }

        crate::ir::types::Instruction::BitwiseExclusiveOr => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x ^ y);
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x ^ y);
            }
        }

        crate::ir::types::Instruction::BitwiseNot => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| !x);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| !x);
            }
        }

        crate::ir::types::Instruction::BitwiseOr => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x | y);
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x | y);
            }
        }

        crate::ir::types::Instruction::ConvertI64ToI32 => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x);
            }
        }

        crate::ir::types::Instruction::Compare => {
            if typ == OperandType::I32
                && ctx.operands.get(srcs[0]).typ == typ
                && ctx.operands.get(srcs[1]).typ == typ
            {
                let comp_val = ctx.operands.get(srcs[2]).value;
                match comp_val as u8 {
                    v if v == Comparison::Equal as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| if x == y { 1 } else { 0 });
                    }
                    v if v == Comparison::NotEqual as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| if x != y { 1 } else { 0 });
                    }
                    v if v == Comparison::Greater as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| if x > y { 1 } else { 0 });
                    }
                    v if v == Comparison::LessOrEqual as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| if x <= y { 1 } else { 0 });
                    }
                    v if v == Comparison::GreaterUI as u8 => {
                        evaluate_binary_i32(
                            ctx,
                            node,
                            |x, y| {
                                if (x as u32) > (y as u32) { 1 } else { 0 }
                            },
                        );
                    }
                    v if v == Comparison::LessOrEqualUI as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| {
                            if (x as u32) <= (y as u32) { 1 } else { 0 }
                        });
                    }
                    v if v == Comparison::GreaterOrEqual as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| if x >= y { 1 } else { 0 });
                    }
                    v if v == Comparison::Less as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| if x < y { 1 } else { 0 });
                    }
                    v if v == Comparison::GreaterOrEqualUI as u8 => {
                        evaluate_binary_i32(ctx, node, |x, y| {
                            if (x as u32) >= (y as u32) { 1 } else { 0 }
                        });
                    }
                    v if v == Comparison::LessUI as u8 => {
                        evaluate_binary_i32(
                            ctx,
                            node,
                            |x, y| {
                                if (x as u32) < (y as u32) { 1 } else { 0 }
                            },
                        );
                    }
                    _ => {}
                }
            }
        }

        crate::ir::types::Instruction::Copy => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x);
            }
        }

        crate::ir::types::Instruction::Divide => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| if y != 0 { x / y } else { 0 });
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| if y != 0 { x / y } else { 0 });
            }
        }

        crate::ir::types::Instruction::DivideUI => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| {
                    if y != 0 {
                        (x as u32 / y as u32) as i32
                    } else {
                        0
                    }
                });
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| {
                    if y != 0 {
                        (x as u64 / y as u64) as i64
                    } else {
                        0
                    }
                });
            }
        }

        crate::ir::types::Instruction::Multiply => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x.wrapping_mul(y));
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x.wrapping_mul(y));
            }
        }

        crate::ir::types::Instruction::Negate => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x.wrapping_neg());
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x.wrapping_neg());
            }
        }

        crate::ir::types::Instruction::ShiftLeft => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x << y);
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x << (y as i32));
            }
        }

        crate::ir::types::Instruction::ShiftRightSI => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x >> y);
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x >> (y as i32));
            }
        }

        crate::ir::types::Instruction::ShiftRightUI => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| (x as u32 >> y) as i32);
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| (x as u64 >> (y as i32)) as i64);
            }
        }

        crate::ir::types::Instruction::SignExtend16 => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x as i16 as i32);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x as i16 as i64);
            }
        }

        crate::ir::types::Instruction::SignExtend32 => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x as i32 as i64);
            }
        }

        crate::ir::types::Instruction::SignExtend8 => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x as i8 as i32);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x as i8 as i64);
            }
        }

        crate::ir::types::Instruction::ZeroExtend16 => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x as u16 as i32);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x as u16 as i64);
            }
        }

        crate::ir::types::Instruction::ZeroExtend32 => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x as u32 as i64);
            }
        }

        crate::ir::types::Instruction::ZeroExtend8 => {
            if typ == OperandType::I32 {
                evaluate_unary_i32(ctx, node, |x| x as u8 as i32);
            } else if typ == OperandType::I64 {
                evaluate_unary_i64(ctx, node, |x| x as u8 as i64);
            }
        }

        crate::ir::types::Instruction::Subtract => {
            if typ == OperandType::I32 {
                evaluate_binary_i32(ctx, node, |x, y| x.wrapping_sub(y));
            } else if typ == OperandType::I64 {
                evaluate_binary_i64(ctx, node, |x, y| x.wrapping_sub(y));
            }
        }

        _ => {}
    }
}

fn are_all_sources_constant(ctx: &TranslationContext, node: OperationIdx) -> bool {
    for &src in ctx.ops.sources(node) {
        if ctx.operands.get(src).kind != crate::ir::types::OperandKind::Constant {
            return false;
        }
    }
    true
}

fn evaluate_unary_i32(ctx: &mut TranslationContext, node: OperationIdx, op: fn(i32) -> i32) {
    let x = ctx.operands.get(ctx.ops.sources(node)[0]).as_i32();
    let result = ctx.operands.constant_i32(op(x));
    ctx.ops.turn_into_copy(node, result, &mut ctx.operands);
}

fn evaluate_unary_i64(ctx: &mut TranslationContext, node: OperationIdx, op: fn(i64) -> i64) {
    let x = ctx.operands.get(ctx.ops.sources(node)[0]).as_i64();
    let result = ctx.operands.constant_i64(op(x));
    ctx.ops.turn_into_copy(node, result, &mut ctx.operands);
}

fn evaluate_binary_i32(ctx: &mut TranslationContext, node: OperationIdx, op: fn(i32, i32) -> i32) {
    let srcs = ctx.ops.sources(node);
    let x = ctx.operands.get(srcs[0]).as_i32();
    let y = ctx.operands.get(srcs[1]).as_i32();
    let result = ctx.operands.constant_i32(op(x, y));
    ctx.ops.turn_into_copy(node, result, &mut ctx.operands);
}

fn evaluate_binary_i64(ctx: &mut TranslationContext, node: OperationIdx, op: fn(i64, i64) -> i64) {
    let srcs = ctx.ops.sources(node);
    let x = ctx.operands.get(srcs[0]).as_i64();
    let y = ctx.operands.get(srcs[1]).as_i64();
    let result = ctx.operands.constant_i64(op(x, y));
    ctx.ops.turn_into_copy(node, result, &mut ctx.operands);
}

fn is_relocatable(ctx: &TranslationContext, operand: OperandIndex) -> bool {
    ctx.operands.get(operand).symbol != crate::ir::types::Symbol::default()
}
