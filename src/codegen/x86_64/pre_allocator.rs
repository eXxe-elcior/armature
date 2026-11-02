// Port of CodeGen/X86/PreAllocator.cs  runs before register allocation:
// lowers constants x86 can't encode, enforces destructive two-operand
// form, inserts fixed-register copies (div→RAX/RDX, shift→RCX, CAS→RAX,
// cmpxchg16b→RDX:RAX/RCX:RBX), and lowers Call/Return/Tailcall/
// LoadArgument to the ABI (PreAllocatorSystemV / PreAllocatorWindows).
//
// List-surgery mapping: C# walks an IntrusiveList doing AddBefore/
// AddAfter/Remove around the current node. The Rust port rebuilds each
// block's op Vec: "AddBefore(node, X)" → push X to `out` before node,
// chained "node = AddAfter(node, X)" → push to `afters` (extended after
// node), Delete → clear def-use and don't push. C#'s nextNode
// re-processing of inserted Copy ops is skipped  the three Insert*
// passes are no-ops on a plain Copy of a local (verified per case).
//

use crate::codegen::x86_64::allocation_result::StackAllocator;
use crate::codegen::x86_64::calling_convention::{self, CallConvName};
use crate::codegen::x86_64::register::X86Register;
use crate::ir::operand::OperandIndex;
use crate::ir::operation::OperationIdx;
use crate::ir::translation::TranslationContext;
use crate::ir::types::{
    Comparison, Instruction, Intrinsic, OperandKind, OperandType, RegisterType,
};
use crate::optimizations;
use crate::translation::compiler::CompilerContext;

use super::{pre_allocator_system_v, pre_allocator_windows};

pub(crate) fn gpr(
    ctx: &mut TranslationContext,
    register: X86Register,
    typ: OperandType,
) -> OperandIndex {
    ctx.operands
        .register(register.index() as u32, RegisterType::Integer, typ)
}

pub(crate) fn xmm(
    ctx: &mut TranslationContext,
    register: X86Register,
    typ: OperandType,
) -> OperandIndex {
    ctx.operands
        .register(register.index() as u32, RegisterType::Vector, typ)
}

// C# Delete(nodes, node): clear def-use, drop from the list (the caller
// simply does not push the node into the rebuilt Vec).
pub(crate) fn delete(ctx: &mut TranslationContext, node: OperationIdx) {
    ctx.ops.set_destinations(node, &[], &mut ctx.operands);
    ctx.ops.set_sources(node, &[], &mut ctx.operands);
}

/// C# PreAllocator.RunPass. Returns maxCallArgs (the C# out param).
pub fn run_pass(
    cctx: &CompilerContext,
    ctx: &mut TranslationContext,
    stack_alloc: &mut StackAllocator,
) -> i32 {
    let mut max_call_args: i32 = -1;

    let call_conv = calling_convention::get_current_call_conv();

    let mut preserved_args: Vec<Option<OperandIndex>> =
        vec![None; calling_convention::get_arguments_on_regs_count() as usize];

    let entry = cctx.cfg.entry;
    let order: Vec<_> = ctx.blocks.order().to_vec();

    for block in order {
        let ops = std::mem::take(&mut ctx.blocks.get_mut(block).ops);
        let mut out: Vec<OperationIdx> = Vec::with_capacity(ops.len());

        for node in ops {
            if ctx.ops.get(node).instruction == Instruction::Phi {
                out.push(node);
                continue;
            }

            let mut afters: Vec<OperationIdx> = Vec::new();
            let mut deleted = false;

            insert_constant_reg_copies(ctx, &mut out, node);
            insert_destructive_reg_copies(ctx, &mut out, &mut afters, node);
            insert_constrained_reg_copies(ctx, &mut out, &mut afters, node);

            match ctx.ops.get(node).instruction {
                Instruction::Call => {
                    // Get the maximum number of arguments used on a call.
                    // On windows, when a struct is returned from the call,
                    // we also need to pass the pointer where the struct
                    // should be written on the first argument.
                    let mut args_count = ctx.ops.sources(node).len() as i32 - 1;

                    let dest = ctx.ops.dests(node).first().copied();
                    if dest.is_some_and(|d| ctx.operands.get(d).typ == OperandType::V128) {
                        args_count += 1;
                    }

                    if max_call_args < args_count {
                        max_call_args = args_count;
                    }

                    // Copy values to registers expected by the function
                    // being called, as mandated by the ABI.
                    if call_conv == CallConvName::Windows {
                        pre_allocator_windows::insert_call_copies(
                            ctx,
                            stack_alloc,
                            &mut out,
                            &mut afters,
                            node,
                        );
                    } else {
                        pre_allocator_system_v::insert_call_copies(
                            ctx,
                            &mut out,
                            &mut afters,
                            node,
                        );
                    }
                }

                Instruction::ConvertToFPUI => {
                    generate_convert_to_fpui(ctx, &mut afters, node);
                    delete(ctx, node);
                    deleted = true;
                }

                Instruction::LoadArgument => {
                    let handled = if call_conv == CallConvName::Windows {
                        pre_allocator_windows::insert_load_argument_copy(
                            cctx,
                            ctx,
                            &mut out,
                            &mut preserved_args,
                            entry,
                            block,
                            node,
                        )
                    } else {
                        pre_allocator_system_v::insert_load_argument_copy(
                            cctx,
                            ctx,
                            &mut out,
                            &mut preserved_args,
                            entry,
                            block,
                            node,
                        )
                    };
                    deleted = handled;
                }

                Instruction::Negate => {
                    let src_typ = ctx.operands.get(ctx.ops.sources(node)[0]).typ;
                    if !src_typ.is_integer() {
                        generate_negate(ctx, &mut afters, node);
                        delete(ctx, node);
                        deleted = true;
                    }
                }

                Instruction::Return => {
                    if call_conv == CallConvName::Windows {
                        pre_allocator_windows::insert_return_copy(
                            cctx,
                            ctx,
                            &mut out,
                            &mut preserved_args,
                            entry,
                            block,
                            node,
                        );
                    } else {
                        pre_allocator_system_v::insert_return_copy(ctx, &mut out, node);
                    }
                }

                Instruction::Tailcall => {
                    if call_conv == CallConvName::Windows {
                        pre_allocator_windows::insert_tailcall_copies(ctx, &mut out, node);
                    } else {
                        pre_allocator_system_v::insert_tailcall_copies(ctx, &mut out, node);
                    }
                }

                Instruction::VectorInsert8 => {
                    if !optimizations::supports_sse41() {
                        generate_vector_insert8(ctx, &mut afters, node);
                        delete(ctx, node);
                        deleted = true;
                    }
                }

                Instruction::Extended => {
                    let intrinsic = ctx.ops.get(node).intrinsic;
                    if intrinsic == Intrinsic::X86Ldmxcsr {
                        let stack_offset = stack_alloc.allocate(OperandType::I32);
                        let offset = ctx.operands.constant_i32(stack_offset);
                        let src0 = ctx.ops.sources(node)[0];
                        ctx.ops
                            .set_sources(node, &[offset, src0], &mut ctx.operands);
                    } else if intrinsic == Intrinsic::X86Stmxcsr {
                        let stack_offset = stack_alloc.allocate(OperandType::I32);
                        let offset = ctx.operands.constant_i32(stack_offset);
                        ctx.ops.set_sources(node, &[offset], &mut ctx.operands);
                    }
                }

                _ => {}
            }

            if !deleted {
                out.push(node);
            }
            out.extend(afters);
        }

        ctx.blocks.get_mut(block).ops = out;
    }

    max_call_args
}

// C# InsertConstantRegCopies. Also called by the ABI files on freshly
// created Copy/Store/SpillArg ops (their sources may be constants).
pub(crate) fn insert_constant_reg_copies(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    if ctx.ops.sources(node).is_empty() || is_xmm_intrinsic(ctx, node) {
        return;
    }

    let inst = ctx.ops.get(node).instruction;

    let mut src1 = ctx.ops.sources(node)[0];

    if ctx.operands.get(src1).kind == OperandKind::Constant {
        if !ctx.operands.get(src1).typ.is_integer() {
            // Handle non-integer types (FP32, FP64 and V128).
            // For instructions without an immediate operand, we do the following:
            // - Insert a copy with the constant value (as integer) to a GPR.
            // - Insert a copy from the GPR to a XMM register.
            // - Replace the constant use with the XMM register.
            src1 = add_xmm_copy(ctx, out, src1);

            ctx.ops.set_source(node, 0, src1, &mut ctx.operands);
        } else if !has_const_src1(inst) {
            // Handle integer types.
            // Most ALU instructions accepts a 32-bits immediate on the second operand.
            // We need to ensure the following:
            // - If the constant is on operand 1, we need to move it.
            // -- But first, we try to swap operand 1 and 2 if the instruction is commutative.
            // -- Doing so may allow us to encode the constant as operand 2 and avoid a copy.
            // - If the constant is on operand 2, we check if the instruction supports it,
            // if not, we also add a copy. 64-bits constants are usually not supported.
            if is_commutative(ctx, node) {
                let src2 = ctx.ops.sources(node)[1];

                ctx.ops.set_source(node, 0, src2, &mut ctx.operands);
                ctx.ops.set_source(node, 1, src1, &mut ctx.operands);

                src1 = src2;
            }

            if ctx.operands.get(src1).kind == OperandKind::Constant {
                src1 = add_copy(ctx, out, src1);

                ctx.ops.set_source(node, 0, src1, &mut ctx.operands);
            }
        }
    }

    if ctx.ops.sources(node).len() < 2 {
        return;
    }

    let src2 = ctx.ops.sources(node)[1];

    if ctx.operands.get(src2).kind == OperandKind::Constant {
        if !ctx.operands.get(src2).typ.is_integer() {
            let new = add_xmm_copy(ctx, out, src2);

            ctx.ops.set_source(node, 1, new, &mut ctx.operands);
        } else if !has_const_src2(inst) || is_long_const(ctx, src2) {
            let new = add_copy(ctx, out, src2);

            ctx.ops.set_source(node, 1, new, &mut ctx.operands);
        }
    }
}

// C# CodeGenCommon.IsLongConst.
fn is_long_const(ctx: &TranslationContext, op: OperandIndex) -> bool {
    let d = ctx.operands.get(op);
    let value = if d.typ == OperandType::I32 {
        d.value as i32 as i64
    } else {
        d.value as i64
    };
    value != value as i32 as i64
}

// C# InsertConstrainedRegCopies.
fn insert_constrained_reg_copies(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    afters: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let dest = ctx.ops.dests(node).first().copied();

    match ctx.ops.get(node).instruction {
        Instruction::CompareAndSwap
        | Instruction::CompareAndSwap16
        | Instruction::CompareAndSwap8 => {
            let typ = ctx.operands.get(ctx.ops.sources(node)[1]).typ;
            let dest = dest.expect("CompareAndSwap has a destination");

            if typ == OperandType::V128 {
                // Handle the many restrictions of the compare and exchange (16 bytes) instruction:
                // - The expected value should be in RDX:RAX.
                // - The new value to be written should be in RCX:RBX.
                // - The value at the memory location is loaded to RDX:RAX.
                let rax = gpr(ctx, X86Register::RAX, OperandType::I64);
                let rbx = gpr(ctx, X86Register::RBX, OperandType::I64);
                let rcx = gpr(ctx, X86Register::RCX, OperandType::I64);
                let rdx = gpr(ctx, X86Register::RDX, OperandType::I64);

                let split_operand = |ctx: &mut TranslationContext,
                                     out: &mut Vec<OperationIdx>,
                                     source: OperandIndex,
                                     lr: OperandIndex,
                                     hr: OperandIndex| {
                    let zero = ctx.operands.constant_i32(0);
                    let one = ctx.operands.constant_i32(1);
                    let lo = ctx.make_op(Instruction::VectorExtract, &[lr], &[source, zero]);
                    out.push(lo);
                    let hi = ctx.make_op(Instruction::VectorExtract, &[hr], &[source, one]);
                    out.push(hi);
                };

                let src1 = ctx.ops.sources(node)[1];
                let src2 = ctx.ops.sources(node)[2];
                split_operand(ctx, out, src1, rax, rdx);
                split_operand(ctx, out, src2, rbx, rcx);

                let vcreate = ctx.make_op(Instruction::VectorCreateScalar, &[dest], &[rax]);
                afters.push(vcreate);
                let one = ctx.operands.constant_i32(1);
                let vinsert = ctx.make_op(Instruction::VectorInsert, &[dest], &[dest, rdx, one]);
                afters.push(vinsert);

                let src0 = ctx.ops.sources(node)[0];
                ctx.ops
                    .set_destinations(node, &[rdx, rax], &mut ctx.operands);
                ctx.ops
                    .set_sources(node, &[src0, rdx, rax, rcx, rbx], &mut ctx.operands);
            } else {
                // Handle the many restrictions of the compare and exchange (32/64) instruction:
                // - The expected value should be in (E/R)AX.
                // - The value at the memory location is loaded to (E/R)AX.
                let expected = ctx.ops.sources(node)[1];
                let new_value = ctx.ops.sources(node)[2];
                let expected_typ = ctx.operands.get(expected).typ;

                let rax = gpr(ctx, X86Register::RAX, expected_typ);

                let copy = ctx.make_op(Instruction::Copy, &[rax], &[expected]);
                out.push(copy);

                // We need to store the new value into a temp, since it may
                // be a constant, and this instruction does not support immediate operands.
                let new_typ = ctx.operands.get(new_value).typ;
                let temp = ctx.temp(new_typ);

                let copy_temp = ctx.make_op(Instruction::Copy, &[temp], &[new_value]);
                out.push(copy_temp);

                let src0 = ctx.ops.sources(node)[0];
                ctx.ops
                    .set_sources(node, &[src0, rax, temp], &mut ctx.operands);

                let copy_out = ctx.make_op(Instruction::Copy, &[dest], &[rax]);
                afters.push(copy_out);

                ctx.ops.set_destinations(node, &[rax], &mut ctx.operands);
            }
        }

        Instruction::Divide | Instruction::DivideUI => {
            // Handle the many restrictions of the division instructions:
            // - The dividend is always in RDX:RAX.
            // - The result is always in RAX.
            // - Additionally it also writes the remainder in RDX.
            let dest = dest.expect("Divide has a destination");
            if ctx.operands.get(dest).typ.is_integer() {
                let src1 = ctx.ops.sources(node)[0];
                let src1_typ = ctx.operands.get(src1).typ;

                let rax = gpr(ctx, X86Register::RAX, src1_typ);
                let rdx = gpr(ctx, X86Register::RDX, src1_typ);

                let copy = ctx.make_op(Instruction::Copy, &[rax], &[src1]);
                out.push(copy);
                let clobber = ctx.make_op(Instruction::Clobber, &[rdx], &[]);
                out.push(clobber);

                let copy_out = ctx.make_op(Instruction::Copy, &[dest], &[rax]);
                afters.push(copy_out);

                let src2 = ctx.ops.sources(node)[1];
                ctx.ops
                    .set_sources(node, &[rdx, rax, src2], &mut ctx.operands);
                ctx.ops.set_destinations(node, &[rax], &mut ctx.operands);
            }
        }

        Instruction::Extended => {
            let intrinsic = ctx.ops.get(node).intrinsic;
            let is_blend = matches!(
                intrinsic,
                Intrinsic::X86Blendvpd | Intrinsic::X86Blendvps | Intrinsic::X86Pblendvb
            );

            // BLENDVPD, BLENDVPS, PBLENDVB last operand is always implied
            // to be XMM0 when VEX is not supported.
            // SHA256RNDS2 always has an implied XMM0 as a last operand.
            if (is_blend && !optimizations::supports_vex_encoding())
                || intrinsic == Intrinsic::X86Sha256Rnds2
            {
                let xmm0 = xmm(ctx, X86Register::XMM0, OperandType::V128);

                let src2 = ctx.ops.sources(node)[2];
                let copy = ctx.make_op(Instruction::Copy, &[xmm0], &[src2]);
                out.push(copy);

                ctx.ops.set_source(node, 2, xmm0, &mut ctx.operands);
            }
        }

        Instruction::Multiply64HighSI | Instruction::Multiply64HighUI => {
            // Handle the many restrictions of the i64 * i64 = i128 multiply instructions:
            // - The multiplicand is always in RAX.
            // - The lower 64-bits of the result is always in RAX.
            // - The higher 64-bits of the result is always in RDX.
            let dest = dest.expect("Multiply64High has a destination");
            let src1 = ctx.ops.sources(node)[0];
            let src1_typ = ctx.operands.get(src1).typ;

            let rax = gpr(ctx, X86Register::RAX, src1_typ);
            let rdx = gpr(ctx, X86Register::RDX, src1_typ);

            let copy = ctx.make_op(Instruction::Copy, &[rax], &[src1]);
            out.push(copy);

            ctx.ops.set_source(node, 0, rax, &mut ctx.operands);

            let copy_out = ctx.make_op(Instruction::Copy, &[dest], &[rdx]);
            afters.push(copy_out);

            ctx.ops
                .set_destinations(node, &[rdx, rax], &mut ctx.operands);
        }

        Instruction::RotateRight
        | Instruction::ShiftLeft
        | Instruction::ShiftRightSI
        | Instruction::ShiftRightUI => {
            // The shift register is always implied to be CL (low 8-bits of RCX
            // or ECX)  except a BMI2 variable shift, which takes the count
            // from any GPR via SHLX/SHRX/SARX (handled in codegen).
            let src1 = ctx.ops.sources(node)[1];
            if ctx.operands.get(src1).kind == OperandKind::LocalVariable
                && !is_bmi2_variable_shift(ctx, node)
            {
                let rcx = gpr(ctx, X86Register::RCX, OperandType::I32);

                let copy = ctx.make_op(Instruction::Copy, &[rcx], &[src1]);
                out.push(copy);

                ctx.ops.set_source(node, 1, rcx, &mut ctx.operands);
            }
        }

        _ => {}
    }
}

// C# InsertDestructiveRegCopies  enforce the x86 two-operand form
// (dest = src1; dest op= src2).
fn insert_destructive_reg_copies(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    afters: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    if ctx.ops.dests(node).is_empty() || ctx.ops.sources(node).is_empty() {
        return;
    }

    let inst = ctx.ops.get(node).instruction;

    // BMI2 SHLX/SHRX/SARX are non-destructive (3-operand)  no dest=src1 copy.
    // The matching count→CL copy is skipped in insert_constrained_reg_copies,
    // and codegen emits the *X form. Runs before that pass (sources unchanged).
    if is_bmi2_variable_shift(ctx, node) {
        return;
    }

    let dest = ctx.ops.dests(node)[0];
    let src1 = ctx.ops.sources(node)[0];

    // The multiply instruction (that maps to IMUL) is somewhat special, it has
    // a three operand form where the second source is a immediate value.
    let three_operand_form = inst == Instruction::Multiply
        && ctx.operands.get(ctx.ops.sources(node)[1]).kind == OperandKind::Constant;

    if is_same_operand_dest_src1(ctx, node)
        && ctx.operands.get(src1).kind == OperandKind::LocalVariable
        && !three_operand_form
    {
        let mut use_new_local = false;

        for src_index in 1..ctx.ops.sources(node).len() {
            if ctx.ops.sources(node)[src_index] == dest {
                use_new_local = true;
                break;
            }
        }

        if use_new_local {
            // Dest is being used as some source already, we need to use a new
            // local to store the temporary value, otherwise the value on dest
            // local would be overwritten.
            let dest_typ = ctx.operands.get(dest).typ;
            let temp = ctx.temp(dest_typ);

            let copy_in = ctx.make_op(Instruction::Copy, &[temp], &[src1]);
            out.push(copy_in);

            ctx.ops.set_source(node, 0, temp, &mut ctx.operands);

            let copy_out = ctx.make_op(Instruction::Copy, &[dest], &[temp]);
            afters.push(copy_out);

            ctx.ops.set_destinations(node, &[temp], &mut ctx.operands);
        } else {
            let copy_in = ctx.make_op(Instruction::Copy, &[dest], &[src1]);
            out.push(copy_in);

            ctx.ops.set_source(node, 0, dest, &mut ctx.operands);
        }
    } else if inst == Instruction::ConditionalSelect {
        let src2 = ctx.ops.sources(node)[1];
        let src3 = ctx.ops.sources(node)[2];

        if src1 == dest || src2 == dest {
            let dest_typ = ctx.operands.get(dest).typ;
            let temp = ctx.temp(dest_typ);

            let copy_in = ctx.make_op(Instruction::Copy, &[temp], &[src3]);
            out.push(copy_in);

            ctx.ops.set_source(node, 2, temp, &mut ctx.operands);

            let copy_out = ctx.make_op(Instruction::Copy, &[dest], &[temp]);
            afters.push(copy_out);

            ctx.ops.set_destinations(node, &[temp], &mut ctx.operands);
        } else {
            let copy_in = ctx.make_op(Instruction::Copy, &[dest], &[src3]);
            out.push(copy_in);

            ctx.ops.set_source(node, 2, dest, &mut ctx.operands);
        }
    }
}

// C# GenerateConvertToFPUI  unsigned int → FP has no x86 instruction.
fn generate_convert_to_fpui(
    ctx: &mut TranslationContext,
    afters: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let dest = ctx.ops.dests(node)[0];
    let source = ctx.ops.sources(node)[0];
    let source_typ = ctx.operands.get(source).typ;
    let dest_typ = ctx.operands.get(dest).typ;

    debug_assert!(
        source_typ.is_integer(),
        "Invalid source type {source_typ:?}"
    );

    if source_typ == OperandType::I32 {
        // For 32-bits integers, we can just zero-extend to 64-bits,
        // and then use the 64-bits signed conversion instructions.
        let zex = ctx.temp(OperandType::I64);

        let zext = ctx.make_op(Instruction::ZeroExtend32, &[zex], &[source]);
        afters.push(zext);
        let conv = ctx.make_op(Instruction::ConvertToFP, &[dest], &[zex]);
        afters.push(conv);
    } else {
        // For 64-bits integers, we need to do the following:
        // - Ensure that the integer has the most significant bit clear.
        // -- This can be done by shifting the value right by 1, that is, dividing by 2.
        // -- The least significant bit is lost in this case though.
        // - We can then convert the shifted value with a signed integer instruction.
        // - The result still needs to be corrected after that.
        // -- First, we need to multiply the result by 2, as we divided it by 2 before.
        // --- This can be done efficiently by adding the result to itself.
        // -- Then, we need to add the least significant bit that was shifted out.
        // --- We can convert the least significant bit to float, and add it to the result.
        let lsb = ctx.temp(OperandType::I64);
        let half = ctx.temp(OperandType::I64);
        let lsb_f = ctx.temp(dest_typ);

        let c1 = ctx.make_op(Instruction::Copy, &[lsb], &[source]);
        afters.push(c1);
        let c2 = ctx.make_op(Instruction::Copy, &[half], &[source]);
        afters.push(c2);

        let one64 = ctx.operands.constant_i64(1);
        let and = ctx.make_op(Instruction::BitwiseAnd, &[lsb], &[lsb, one64]);
        afters.push(and);
        let one32 = ctx.operands.constant_i32(1);
        let shr = ctx.make_op(Instruction::ShiftRightUI, &[half], &[half, one32]);
        afters.push(shr);

        let conv_lsb = ctx.make_op(Instruction::ConvertToFP, &[lsb_f], &[lsb]);
        afters.push(conv_lsb);
        let conv = ctx.make_op(Instruction::ConvertToFP, &[dest], &[half]);
        afters.push(conv);

        let add1 = ctx.make_op(Instruction::Add, &[dest], &[dest, dest]);
        afters.push(add1);
        let add2 = ctx.make_op(Instruction::Add, &[dest], &[dest, lsb_f]);
        afters.push(add2);
    }
}

// C# GenerateNegate  no SSE FP negate; XOR with a sign-bit mask.
fn generate_negate(
    ctx: &mut TranslationContext,
    afters: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let dest = ctx.ops.dests(node)[0];
    let source = ctx.ops.sources(node)[0];
    let dest_typ = ctx.operands.get(dest).typ;

    debug_assert!(
        matches!(dest_typ, OperandType::FP32 | OperandType::FP64),
        "Invalid destination type {dest_typ:?}"
    );

    let res = ctx.temp(dest_typ);

    let ones = ctx.make_op(Instruction::VectorOne, &[res], &[]);
    afters.push(ones);

    if dest_typ == OperandType::FP32 {
        let c31 = ctx.operands.constant_i32(31);
        let shl = ctx.make_intrinsic_op(Intrinsic::X86Pslld, &[res], &[res, c31]);
        afters.push(shl);
    } else {
        let c63 = ctx.operands.constant_i32(63);
        let shl = ctx.make_intrinsic_op(Intrinsic::X86Psllq, &[res], &[res, c63]);
        afters.push(shl);
    }

    let xor = ctx.make_intrinsic_op(Intrinsic::X86Xorps, &[res], &[res, source]);
    afters.push(xor);

    let copy = ctx.make_op(Instruction::Copy, &[dest], &[res]);
    afters.push(copy);
}

// C# GenerateVectorInsert8  vector byte insertion without SSE 4.1.
fn generate_vector_insert8(
    ctx: &mut TranslationContext,
    afters: &mut Vec<OperationIdx>,
    node: OperationIdx,
) {
    let dest = ctx.ops.dests(node)[0];
    let src1 = ctx.ops.sources(node)[0]; // Vector
    let src2 = ctx.ops.sources(node)[1]; // Value
    let src3 = ctx.ops.sources(node)[2]; // Index

    debug_assert_eq!(ctx.operands.get(src3).kind, OperandKind::Constant);

    let index = ctx.operands.get(src3).value as u8;

    debug_assert!(index < 16);

    let temp1 = ctx.temp(OperandType::I32);
    let temp2 = ctx.temp(OperandType::I32);

    let copy = ctx.make_op(Instruction::Copy, &[temp2], &[src2]);
    afters.push(copy);

    let idx_half = ctx.operands.constant_i32((index >> 1) as i32);
    let vext = ctx.make_op(Instruction::VectorExtract16, &[temp1], &[src1, idx_half]);
    afters.push(vext);

    if index & 1 != 0 {
        let zx = ctx.make_op(Instruction::ZeroExtend8, &[temp1], &[temp1]);
        afters.push(zx);
        let eight = ctx.operands.constant_i32(8);
        let shl = ctx.make_op(Instruction::ShiftLeft, &[temp2], &[temp2, eight]);
        afters.push(shl);
        let or = ctx.make_op(Instruction::BitwiseOr, &[temp1], &[temp1, temp2]);
        afters.push(or);
    } else {
        let zx = ctx.make_op(Instruction::ZeroExtend8, &[temp2], &[temp2]);
        afters.push(zx);
        let mask = ctx.operands.constant_i32(0xff00);
        let and = ctx.make_op(Instruction::BitwiseAnd, &[temp1], &[temp1, mask]);
        afters.push(and);
        let or = ctx.make_op(Instruction::BitwiseOr, &[temp1], &[temp1, temp2]);
        afters.push(or);
    }

    let idx_half = ctx.operands.constant_i32((index >> 1) as i32);
    let vins = ctx.make_op(
        Instruction::VectorInsert16,
        &[dest],
        &[src1, temp1, idx_half],
    );
    afters.push(vins);
}

// C# AddXmmCopy.
pub(crate) fn add_xmm_copy(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    source: OperandIndex,
) -> OperandIndex {
    let source_typ = ctx.operands.get(source).typ;
    let temp = ctx.temp(source_typ);
    let int_const = get_int_const(ctx, source);
    let int_copy = add_copy(ctx, out, int_const);

    let copy_op = ctx.make_op(Instruction::VectorCreateScalar, &[temp], &[int_copy]);
    out.push(copy_op);

    temp
}

// C# AddCopy.
pub(crate) fn add_copy(
    ctx: &mut TranslationContext,
    out: &mut Vec<OperationIdx>,
    source: OperandIndex,
) -> OperandIndex {
    let source_typ = ctx.operands.get(source).typ;
    let temp = ctx.temp(source_typ);

    let copy_op = ctx.make_op(Instruction::Copy, &[temp], &[source]);
    out.push(copy_op);

    temp
}

// C# GetIntConst  reinterpret an FP constant's bits as an integer const.
fn get_int_const(ctx: &mut TranslationContext, value: OperandIndex) -> OperandIndex {
    let d = ctx.operands.get(value);
    match d.typ {
        OperandType::FP32 => {
            let v = d.value as i32;
            ctx.operands.constant_i32(v)
        }
        OperandType::FP64 => {
            let v = d.value as i64;
            ctx.operands.constant_i64(v)
        }
        _ => value,
    }
}

// C# IsSameOperandDestSrc1.
// A variable-count shift (count in a register) that BMI2 lowers to a single
// non-destructive SHLX/SHRX/SARX: no count→CL copy, no dest=value copy.
// RotateRight is excluded  BMI2 has no variable-rotate form (RORX is
// immediate-only), so rotates keep the CL path.
fn is_bmi2_variable_shift(ctx: &TranslationContext, node: OperationIdx) -> bool {
    if !optimizations::use_bmi2() {
        return false;
    }
    let srcs = ctx.ops.sources(node);
    matches!(
        ctx.ops.get(node).instruction,
        Instruction::ShiftLeft | Instruction::ShiftRightSI | Instruction::ShiftRightUI
    ) && srcs.len() > 1
        && ctx.operands.get(srcs[1]).kind == OperandKind::LocalVariable
}

fn is_same_operand_dest_src1(ctx: &TranslationContext, node: OperationIdx) -> bool {
    let dest_is_integer = ctx
        .ops
        .dests(node)
        .first()
        .map(|&d| ctx.operands.get(d).typ.is_integer())
        .unwrap_or(false);
    let supports_vex = optimizations::supports_vex_encoding();

    match ctx.ops.get(node).instruction {
        Instruction::Add => !supports_vex && !dest_is_integer,
        Instruction::Multiply | Instruction::Subtract => !supports_vex || dest_is_integer,

        Instruction::BitwiseAnd
        | Instruction::BitwiseExclusiveOr
        | Instruction::BitwiseNot
        | Instruction::BitwiseOr
        | Instruction::ByteSwap
        | Instruction::Negate
        | Instruction::RotateRight
        | Instruction::ShiftLeft
        | Instruction::ShiftRightSI
        | Instruction::ShiftRightUI => true,

        Instruction::Divide => !supports_vex && !dest_is_integer,

        Instruction::VectorInsert | Instruction::VectorInsert16 | Instruction::VectorInsert8 => {
            !supports_vex
        }

        Instruction::Extended => is_intrinsic_same_operand_dest_src1(ctx, node),

        // C# falls through to IsVexSameOperandDestSrc1, which is false for
        // non-Extended instructions.
        _ => false,
    }
}

// C# IsIntrinsicSameOperandDestSrc1.
fn is_intrinsic_same_operand_dest_src1(ctx: &TranslationContext, node: OperationIdx) -> bool {
    use crate::codegen::x86_64::intrinsic_table::{self, IntrinsicType};

    let info = intrinsic_table::info(ctx.ops.get(node).intrinsic);

    info.typ == IntrinsicType::Crc32
        || info.typ == IntrinsicType::Fma
        || is_vex_same_operand_dest_src1(ctx, node)
}

// C# IsVexSameOperandDestSrc1 (the IsIntrinsic guard is satisfied  only
// called from the Extended arm).
fn is_vex_same_operand_dest_src1(ctx: &TranslationContext, node: OperationIdx) -> bool {
    use crate::codegen::x86_64::assembler_table;
    use crate::codegen::x86_64::intrinsic_table;

    let info = intrinsic_table::info(ctx.ops.get(node).intrinsic);

    let has_vex =
        optimizations::supports_vex_encoding() && assembler_table::supports_vex_prefix(info.inst);

    let is_unary = ctx.ops.sources(node).len() < 2;

    let has_vec_dest = ctx
        .ops
        .dests(node)
        .first()
        .is_some_and(|&d| ctx.operands.get(d).typ == OperandType::V128);

    !has_vex && !is_unary && has_vec_dest
}

// C# HasConstSrc1.
fn has_const_src1(inst: Instruction) -> bool {
    matches!(
        inst,
        Instruction::Copy | Instruction::LoadArgument | Instruction::Spill | Instruction::SpillArg
    )
}

// C# HasConstSrc2.
fn has_const_src2(inst: Instruction) -> bool {
    matches!(
        inst,
        Instruction::Add
            | Instruction::BitwiseAnd
            | Instruction::BitwiseExclusiveOr
            | Instruction::BitwiseOr
            | Instruction::BranchIf
            | Instruction::Compare
            | Instruction::Multiply
            | Instruction::RotateRight
            | Instruction::ShiftLeft
            | Instruction::ShiftRightSI
            | Instruction::ShiftRightUI
            | Instruction::Store
            | Instruction::Store16
            | Instruction::Store8
            | Instruction::Subtract
            | Instruction::VectorExtract
            | Instruction::VectorExtract16
            | Instruction::VectorExtract8
    )
}

// C# IsCommutative.
fn is_commutative(ctx: &TranslationContext, node: OperationIdx) -> bool {
    match ctx.ops.get(node).instruction {
        Instruction::Add
        | Instruction::BitwiseAnd
        | Instruction::BitwiseExclusiveOr
        | Instruction::BitwiseOr
        | Instruction::Multiply => true,

        Instruction::BranchIf | Instruction::Compare => {
            let comp = ctx.ops.sources(node)[2];
            let d = ctx.operands.get(comp);

            debug_assert_eq!(d.kind, OperandKind::Constant);

            let comp_value = d.value as i32;
            comp_value == Comparison::Equal as i32 || comp_value == Comparison::NotEqual as i32
        }

        _ => false,
    }
}

// C# IsXmmIntrinsic.
fn is_xmm_intrinsic(ctx: &TranslationContext, node: OperationIdx) -> bool {
    use crate::codegen::x86_64::intrinsic_table::{self, IntrinsicType};

    if ctx.ops.get(node).instruction != Instruction::Extended {
        return false;
    }

    let info = intrinsic_table::info(ctx.ops.get(node).intrinsic);

    info.typ != IntrinsicType::Crc32
}
