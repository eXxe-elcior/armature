// Port of CodeGen/X86/CodeGenerator.cs: lowers IR operations to x86-64.

use crate::codegen::compiled_function::CompiledFunction;
use crate::codegen::x86_64::codegen_context::CodeGenContext;
use crate::codegen::x86_64::condition::{X86Condition, to_x86_condition};
use crate::codegen::x86_64::register::X86Register;
use crate::ir::operand::{OperandArena, OperandIndex};
use crate::ir::translation::TranslationContext;
use crate::ir::types::{
    Comparison, Instruction, Multiplier, OperandKind, OperandType, RegisterType,
};
use crate::translation::compiler::CompilerContext;

thread_local! {
    /// Guest entry address of the function currently being compiled on this
    /// thread; set by the translator so ARMATURE_DUMP_FN can target one function.
    pub static DUMP_FN_ENTRY: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Set the guest entry address for the function about to be compiled (used only
/// for ARMATURE_DUMP_FN-gated diagnostics).
pub fn set_dump_fn_entry(addr: u64) {
    DUMP_FN_ENTRY.with(|c| c.set(addr));
}

fn dump_fn_target() -> Option<u64> {
    static T: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("ARMATURE_DUMP_FN")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
    })
}

pub fn generate(cctx: &mut CompilerContext, ir: &mut TranslationContext) -> CompiledFunction {
    use crate::translation::compiler::compiler_options;

    if std::env::var("ARMATURE_DUMP_IR").is_ok() {
        dump_ir(ir, "pre-optimize");
    }

    if cctx.options.contains(compiler_options::OPTIMIZE) {
        if cctx.options.contains(compiler_options::SSA_FORM) {
            crate::codegen::optimizations::optimizer::run_pass(ir);
        }
        crate::codegen::optimizations::block_placement::run_pass(
            &mut cctx.cfg,
            &mut ir.blocks,
            &mut ir.ops,
            &mut ir.operands,
        );
    }

    crate::codegen::x86_64::x86_optimizer::run_pass(&cctx.cfg, ir);

    let mut stack_alloc = crate::codegen::x86_64::allocation_result::StackAllocator::new();
    let max_call_args = crate::codegen::x86_64::pre_allocator::run_pass(cctx, ir, &mut stack_alloc);

    if std::env::var("ARMATURE_DUMP_IR").is_ok() {
        dump_ir(ir, "pre-deconstruct");
    }

    if cctx.options.contains(compiler_options::SSA_FORM) {
        // C#: Ssa.Deconstruct(cfg)  does not change the block set (phi
        // copies land on existing blocks; no edge splitting).
        crate::translation::ssa::deconstruct(&mut cctx.cfg, ir);
    }

    let reg_masks = crate::codegen::x86_64::allocation_result::RegisterMasks::new(16);
    let alloc_result = if cctx.options.contains(compiler_options::LSRA) {
        crate::codegen::regalloc::linear_scan::run_pass(&cctx.cfg, ir, &mut stack_alloc, &reg_masks)
    } else {
        crate::codegen::regalloc::hybrid::run_pass(&cctx.cfg, ir, &mut stack_alloc, &reg_masks)
    };

    if std::env::var("ARMATURE_DUMP_IR").is_ok() {
        dump_ir(ir, "post-regalloc");
    }
    // ARMATURE_DUMP_FN=<hex guest addr>: dump only that function's final IR.
    if dump_fn_target().is_some_and(|t| t == DUMP_FN_ENTRY.with(|c| c.get())) {
        dump_ir(
            ir,
            &format!("post-regalloc fn={:#x}", DUMP_FN_ENTRY.with(|c| c.get())),
        );
    }

    let relocatable = cctx
        .options
        .contains(crate::translation::compiler::compiler_options::RELOCATABLE);

    let order: Vec<_> = ir.blocks.order().iter().copied().collect();
    let mut cg = CodeGenContext::new(alloc_result, max_call_args, order.len(), relocatable);

    write_prologue(&mut cg, ir);

    for i in 0..order.len() {
        let block_idx = order[i];
        cg.enter_block(block_idx, &mut ir.operands);

        let ops_snapshot: Vec<crate::ir::operation::OperationIdx> =
            ir.blocks.get(block_idx).ops.iter().copied().collect();

        for &op_idx in &ops_snapshot {
            generate_operation(&mut cg, op_idx, ir);
        }

        let block = ir.blocks.get_mut(block_idx);
        let succ_count = block.successor_count;
        if succ_count > 0 {
            let fallthrough = block.successors[0];
            let next = order.get(i + 1).copied();
            if fallthrough != next {
                cg.jump_to(fallthrough.unwrap(), &mut ir.operands);
            }
        }
    }

    let (code, reloc_info) = cg.get_code(&ir.operands);
    CompiledFunction { code, reloc_info }
}

fn fmt_operand(ir: &TranslationContext, o: crate::ir::operand::OperandIndex) -> String {
    if o.0 == u32::MAX {
        return "∅".into();
    }
    let d = ir.operands.get(o);
    match d.kind {
        crate::ir::types::OperandKind::LocalVariable => format!("%{}", o.0),
        crate::ir::types::OperandKind::Constant => format!("#{:x}", d.value),
        crate::ir::types::OperandKind::Register => {
            format!("{:?}{}", d.reg_type(), d.reg_index())
        }
        k => format!("{k:?}"),
    }
}

fn dump_ir(ir: &TranslationContext, tag: &str) {
    eprintln!("== IR {tag} ==");
    for &b in ir.blocks.order() {
        let data = ir.blocks.get(b);
        eprintln!(
            "B{} (idx={}) succ={:?}:",
            b.0,
            data.index,
            &data.successors[..data.successor_count as usize]
        );
        for &op in &data.ops {
            let inst = ir.ops.get(op).instruction;
            let dests: Vec<String> = ir
                .ops
                .dests(op)
                .iter()
                .map(|&d| fmt_operand(ir, d))
                .collect();
            let srcs: Vec<String> = ir
                .ops
                .sources(op)
                .iter()
                .map(|&s| fmt_operand(ir, s))
                .collect();
            if ir.ops.get(op).is_phi() {
                let inputs: Vec<String> = ir
                    .ops
                    .phi_inputs(op)
                    .iter()
                    .map(|i| format!("B{}:{}", i.block.0, fmt_operand(ir, i.value)))
                    .collect();
                eprintln!(
                    "  {:?} {} = phi {}",
                    inst,
                    dests.join(","),
                    inputs.join(", ")
                );
            } else {
                eprintln!("  {:?} {} <- {}", inst, dests.join(","), srcs.join(", "));
            }
        }
    }
}

fn generate_operation(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let inst = ir.ops.get(op_idx).instruction;

    if inst == Instruction::Extended {
        generate_extended(cg, op_idx, ir);
        return;
    }

    match inst {
        Instruction::Add => generate_add(cg, op_idx, ir),
        Instruction::Subtract => generate_subtract(cg, op_idx, ir),
        Instruction::Multiply => generate_multiply(cg, op_idx, ir),
        Instruction::BitwiseAnd => generate_bitwise_and(cg, op_idx, ir),
        Instruction::BitwiseAndNot => generate_bitwise_and_not(cg, op_idx, ir),
        Instruction::BitwiseOr => generate_bitwise_or(cg, op_idx, ir),
        Instruction::BitwiseExclusiveOr => generate_bitwise_xor(cg, op_idx, ir),
        Instruction::BitwiseNot => generate_bitwise_not(cg, op_idx, ir),
        Instruction::Negate => generate_negate(cg, op_idx, ir),
        Instruction::Copy => generate_copy(cg, op_idx, ir),
        Instruction::Load => generate_load(cg, op_idx, ir),
        Instruction::Load8 => generate_load8(cg, op_idx, ir),
        Instruction::Load16 => generate_load16(cg, op_idx, ir),
        Instruction::Store => generate_store(cg, op_idx, ir),
        Instruction::Store8 => generate_store8(cg, op_idx, ir),
        Instruction::Store16 => generate_store16(cg, op_idx, ir),
        Instruction::Return => generate_return(cg, op_idx, ir),
        Instruction::BranchIf => generate_branch_if(cg, op_idx, ir),
        Instruction::Compare => generate_compare(cg, op_idx, ir),
        Instruction::ShiftLeft => generate_shift_left(cg, op_idx, ir),
        Instruction::ShiftRightSI => generate_shift_right_si(cg, op_idx, ir),
        Instruction::ShiftRightUI => generate_shift_right_ui(cg, op_idx, ir),
        Instruction::RotateRight => generate_rotate_right(cg, op_idx, ir),
        Instruction::ConvertI64ToI32 => generate_convert_i64_to_i32(cg, op_idx, ir),
        Instruction::ConvertToFP => generate_convert_to_fp(cg, op_idx, ir),
        Instruction::ZeroExtend8 => generate_zero_extend8(cg, op_idx, ir),
        Instruction::ZeroExtend16 => generate_zero_extend16(cg, op_idx, ir),
        Instruction::ZeroExtend32 => generate_zero_extend32(cg, op_idx, ir),
        Instruction::SignExtend8 => generate_sign_extend8(cg, op_idx, ir),
        Instruction::SignExtend16 => generate_sign_extend16(cg, op_idx, ir),
        Instruction::SignExtend32 => generate_sign_extend32(cg, op_idx, ir),
        Instruction::Call => generate_call(cg, op_idx, ir),
        Instruction::StackAlloc => generate_stack_alloc(cg, op_idx, ir),
        Instruction::Divide => generate_divide(cg, op_idx, ir),
        Instruction::DivideUI => generate_divide_ui(cg, op_idx, ir),
        Instruction::Clobber => {}
        Instruction::Tailcall => generate_tailcall(cg, op_idx, ir),
        Instruction::Fill => generate_fill(cg, op_idx, ir),
        Instruction::Spill => generate_spill(cg, op_idx, ir, true),
        Instruction::SpillArg => generate_spill(cg, op_idx, ir, false),
        Instruction::MemoryBarrier => generate_memory_barrier(cg, ir),
        Instruction::MemoryBarrierStore => {} // no-op on x86-TSO (store-store ordered)
        Instruction::MemoryBarrierLoad => {}  // no-op on x86-TSO (load-load ordered)
        Instruction::ConditionalSelect => generate_conditional_select(cg, op_idx, ir),
        Instruction::ByteSwap => generate_byte_swap(cg, op_idx, ir),
        Instruction::CountLeadingZeros => generate_count_leading_zeros(cg, op_idx, ir),
        Instruction::Multiply64HighSI => generate_multiply64_high_si(cg, op_idx, ir),
        Instruction::Multiply64HighUI => generate_multiply64_high_ui(cg, op_idx, ir),
        Instruction::CompareAndSwap => generate_compare_and_swap(cg, op_idx, ir),
        Instruction::CompareAndSwap16 => generate_compare_and_swap16(cg, op_idx, ir),
        Instruction::CompareAndSwap8 => generate_compare_and_swap8(cg, op_idx, ir),
        Instruction::VectorZero => generate_vector_zero(cg, op_idx, ir),
        Instruction::VectorOne => generate_vector_one(cg, op_idx, ir),
        Instruction::VectorCreateScalar => generate_vector_create_scalar(cg, op_idx, ir),
        Instruction::VectorExtract => generate_vector_extract(cg, op_idx, ir),
        Instruction::VectorExtract16 => generate_vector_extract16(cg, op_idx, ir),
        Instruction::VectorExtract8 => generate_vector_extract8(cg, op_idx, ir),
        Instruction::VectorInsert => generate_vector_insert(cg, op_idx, ir),
        Instruction::VectorInsert16 => generate_vector_insert16(cg, op_idx, ir),
        Instruction::VectorInsert8 => generate_vector_insert8(cg, op_idx, ir),
        Instruction::VectorZeroUpper64 => {
            let d = dest(&ir.ops, op_idx);
            let s = src(&ir.ops, op_idx, 0);
            cg.asm.movq(&ir.operands, d, s);
        }
        Instruction::VectorZeroUpper96 => {
            let d = dest(&ir.ops, op_idx);
            let s = src(&ir.ops, op_idx, 0);
            cg.asm.movq(&ir.operands, d, s);
            cg.asm.pshufd(&ir.operands, d, d, 0xfc);
        }
        // C# has no LoadArgument handler  the PreAllocator consumes all
        // of them (stack-passed args are its TODO too).
        _ => panic!("codegen: unimplemented instruction {:?}", inst),
    }
}

// Operand helpers
fn dest(
    ops: &crate::ir::operation::OperationArena,
    idx: crate::ir::operation::OperationIdx,
) -> OperandIndex {
    ops.dests(idx)[0]
}

fn src(
    ops: &crate::ir::operation::OperationArena,
    idx: crate::ir::operation::OperationIdx,
    n: usize,
) -> OperandIndex {
    ops.sources(idx)[n]
}

fn is_const(op: &OperandArena, idx: OperandIndex) -> bool {
    op.get(idx).kind == OperandKind::Constant
}

fn is_zero(op: &OperandArena, idx: OperandIndex) -> bool {
    op.get(idx).kind == OperandKind::Constant && op.get(idx).value == 0
}

fn same_operand(op: &OperandArena, a: OperandIndex, b: OperandIndex) -> bool {
    let da = op.get(a);
    let db = op.get(b);
    da.kind == db.kind && da.value == db.value
}

fn make_memory(ops: &mut OperandArena, base: OperandIndex) -> OperandIndex {
    ops.memory(OperandType::I64, Some(base), None, Multiplier::x1, 0)
}

fn gen_reg(ops: &mut OperandArena, reg: X86Register, typ: OperandType) -> OperandIndex {
    ops.register(reg.index() as u32, RegisterType::Integer, typ)
}

// Arithmetic
fn generate_add(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(d).typ;
    if typ.is_integer() {
        // C#: dest == src1 → plain add; otherwise 3-operand via LEA
        // [src1 + src2] / [src1 + offset] (integer Add is NOT forced
        // destructive by the PreAllocator).
        if same_operand(&ir.operands, d, s1) {
            cg.asm.add(&ir.operands, d, s2, typ);
        } else {
            let (offset, index) = if is_const(&ir.operands, s2) {
                (ir.operands.get(s2).value as i32, None)
            } else {
                (0, Some(s2))
            };
            let mem = ir
                .operands
                .memory(typ, Some(s1), index, Multiplier::x1, offset);
            cg.asm.lea(&ir.operands, d, mem, typ);
        }
    } else if typ == OperandType::FP32 {
        cg.asm.addss(&ir.operands, d, s1, s2);
    } else {
        cg.asm.addsd(&ir.operands, d, s1, s2);
    }
}

fn generate_subtract(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(d).typ;
    if typ.is_integer() {
        cg.asm.sub(&ir.operands, d, s2, typ);
    } else if typ == OperandType::FP32 {
        cg.asm.subss(&ir.operands, d, s1, s2);
    } else {
        cg.asm.subsd(&ir.operands, d, s1, s2);
    }
}

fn generate_multiply(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(d).typ;
    if typ.is_integer() {
        if is_const(&ir.operands, s2) {
            cg.asm.imul3(&ir.operands, d, s1, s2, typ);
        } else {
            cg.asm.imul2(&ir.operands, d, s2, typ);
        }
    } else if typ == OperandType::FP32 {
        cg.asm.mulss(&ir.operands, d, s1, s2);
    } else {
        cg.asm.mulsd(&ir.operands, d, s1, s2);
    }
}

fn generate_divide(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let typ = ir.operands.get(d).typ;
    if typ.is_integer() {
        let divisor = src(&ir.ops, op_idx, 2);
        if typ == OperandType::I32 {
            cg.asm.cdq();
        } else {
            cg.asm.cqo();
        }
        cg.asm.idiv(&ir.operands, divisor, typ);
    } else if typ == OperandType::FP32 {
        cg.asm.divss(
            &ir.operands,
            d,
            src(&ir.ops, op_idx, 0),
            src(&ir.ops, op_idx, 1),
        );
    } else {
        cg.asm.divsd(
            &ir.operands,
            d,
            src(&ir.ops, op_idx, 0),
            src(&ir.ops, op_idx, 1),
        );
    }
}

fn generate_divide_ui(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let divisor = src(&ir.ops, op_idx, 2);
    let rdx = gen_reg(&mut ir.operands, X86Register::RDX, OperandType::I64);
    cg.asm.xor(&ir.operands, rdx, rdx, OperandType::I32);
    cg.asm
        .div(&ir.operands, divisor, ir.operands.get(divisor).typ);
}

// Bitwise
fn generate_bitwise_and(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s2 = src(&ir.ops, op_idx, 1);
    cg.asm.and(&ir.operands, d, s2, ir.operands.get(d).typ);
}

// (~src0) & src1 via BMI1 ANDN (non-destructive 3-operand). Only reached when
// the emitter chose BitwiseAndNot, i.e. host has BMI1.
fn generate_bitwise_and_not(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s0 = src(&ir.ops, op_idx, 0); // negated operand → VEX.vvvv
    let s1 = src(&ir.ops, op_idx, 1); // → ModRM.r/m
    cg.asm.andn(&ir.operands, d, s0, s1, ir.operands.get(d).typ);
}

fn generate_bitwise_or(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s2 = src(&ir.ops, op_idx, 1);
    cg.asm.or(&ir.operands, d, s2, ir.operands.get(d).typ);
}

fn generate_bitwise_xor(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(d).typ;
    if typ.is_integer() {
        cg.asm.xor(&ir.operands, d, s2, typ);
    } else {
        cg.asm.xorps(&ir.operands, d, s1, s2);
    }
}

fn generate_bitwise_not(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    cg.asm.not(&ir.operands, d, ir.operands.get(d).typ);
}

fn generate_negate(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    cg.asm.neg(&ir.operands, d, ir.operands.get(d).typ);
}

// Copy / Mov
fn generate_copy(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    let typ = ir.operands.get(d).typ;
    if same_operand(&ir.operands, d, s) {
        return;
    }
    // C#: xor-zero only valid when dest is a register (xor mem, mem isn't
    // a thing).
    if ir.operands.get(d).kind == OperandKind::Register && is_zero(&ir.operands, s) {
        cg.asm.xor(&ir.operands, d, d, OperandType::I32);
        return;
    }
    if typ.is_integer() {
        cg.asm.mov(&ir.operands, d, s, typ);
    } else {
        cg.asm.movdqu(&ir.operands, d, s);
    }
}

// Load / Store
fn generate_load(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let addr = src(&ir.ops, op_idx, 0);
    let typ = ir.operands.get(d).typ;
    let mem = if ir.operands.get(addr).kind == OperandKind::Memory {
        addr
    } else {
        make_memory(&mut ir.operands, addr)
    };
    match typ {
        OperandType::I32 | OperandType::I64 => cg.asm.mov(&ir.operands, d, mem, typ),
        OperandType::FP32 => cg.asm.movd(&ir.operands, d, mem),
        OperandType::FP64 => cg.asm.movq(&ir.operands, d, mem),
        OperandType::V128 => cg.asm.movdqu(&ir.operands, d, mem),
        _ => {}
    }
}

fn generate_load8(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let addr = src(&ir.ops, op_idx, 0);
    let mem = if ir.operands.get(addr).kind == OperandKind::Memory {
        addr
    } else {
        make_memory(&mut ir.operands, addr)
    };
    cg.asm.movzx8(&ir.operands, d, mem, ir.operands.get(d).typ);
}

fn generate_load16(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let addr = src(&ir.ops, op_idx, 0);
    let mem = if ir.operands.get(addr).kind == OperandKind::Memory {
        addr
    } else {
        make_memory(&mut ir.operands, addr)
    };
    cg.asm.movzx16(&ir.operands, d, mem, ir.operands.get(d).typ);
}

fn generate_store(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let value = src(&ir.ops, op_idx, 1);
    let addr = src(&ir.ops, op_idx, 0);
    let typ = ir.operands.get(value).typ;
    let mem = if ir.operands.get(addr).kind == OperandKind::Memory {
        addr
    } else {
        make_memory(&mut ir.operands, addr)
    };
    match typ {
        OperandType::I32 | OperandType::I64 => cg.asm.mov(&ir.operands, mem, value, typ),
        OperandType::FP32 => cg.asm.movd(&ir.operands, mem, value),
        OperandType::FP64 => cg.asm.movq(&ir.operands, mem, value),
        OperandType::V128 => cg.asm.movdqu(&ir.operands, mem, value),
        _ => {}
    }
}

fn generate_store8(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let value = src(&ir.ops, op_idx, 1);
    let addr = src(&ir.ops, op_idx, 0);
    let mem = if ir.operands.get(addr).kind == OperandKind::Memory {
        addr
    } else {
        make_memory(&mut ir.operands, addr)
    };
    cg.asm.mov8(&ir.operands, mem, value);
}

fn generate_store16(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let value = src(&ir.ops, op_idx, 1);
    let addr = src(&ir.ops, op_idx, 0);
    let mem = if ir.operands.get(addr).kind == OperandKind::Memory {
        addr
    } else {
        make_memory(&mut ir.operands, addr)
    };
    cg.asm.mov16(&ir.operands, mem, value);
}

// Control flow
fn generate_return(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    // C# relies on register allocation to place the return value in RAX.
    // Our LSRA may not do this, so explicitly move it.
    if ir.ops.sources(op_idx).len() > 0 {
        let src = ir.ops.sources(op_idx)[0];
        let rax = ir
            .operands
            .register(0, RegisterType::Integer, OperandType::I64);
        cg.asm.mov(&mut ir.operands, rax, src, OperandType::I64);
    }
    write_epilogue(cg, &mut ir.operands);
    cg.asm.r#return();
}

fn generate_branch_if(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let comp_op = src(&ir.ops, op_idx, 2);
    let comp_val = ir.operands.get(comp_op).value as i32;
    let cond = x86_cond_from_comparison(comp_val);
    generate_compare_common(cg, op_idx, ir);
    let block = ir.blocks.get_mut(cg.curr_block.unwrap());
    let taken = block.successors[1].unwrap();
    cg.jump_to_cond(cond, taken, &mut ir.operands);
}

fn generate_compare(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let comp_op = src(&ir.ops, op_idx, 2);
    let comp_val = ir.operands.get(comp_op).value as i32;
    let cond = x86_cond_from_comparison(comp_val);
    generate_compare_common(cg, op_idx, ir);
    cg.asm.setcc(&ir.operands, d, cond);
    cg.asm.movzx8(&ir.operands, d, d, OperandType::I32);
}

fn generate_compare_common(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(s1).typ;
    if is_const(&ir.operands, s2) && ir.operands.get(s2).value == 0 {
        cg.asm.test(&ir.operands, s1, s1, typ);
    } else {
        cg.asm.cmp(&ir.operands, s1, s2, typ);
    }
}

// C#: (Comparison)comp.AsInt32()  the constant IS the discriminant.
// Match against the enum's own values; never re-invent the numbering.
fn x86_cond_from_comparison(val: i32) -> X86Condition {
    to_x86_condition(match val {
        v if v == Comparison::Equal as i32 => Comparison::Equal,
        v if v == Comparison::NotEqual as i32 => Comparison::NotEqual,
        v if v == Comparison::Greater as i32 => Comparison::Greater,
        v if v == Comparison::LessOrEqual as i32 => Comparison::LessOrEqual,
        v if v == Comparison::GreaterUI as i32 => Comparison::GreaterUI,
        v if v == Comparison::LessOrEqualUI as i32 => Comparison::LessOrEqualUI,
        v if v == Comparison::GreaterOrEqual as i32 => Comparison::GreaterOrEqual,
        v if v == Comparison::Less as i32 => Comparison::Less,
        v if v == Comparison::GreaterOrEqualUI as i32 => Comparison::GreaterOrEqualUI,
        v if v == Comparison::LessUI as i32 => Comparison::LessUI,
        _ => panic!("invalid Comparison value: {val}"),
    })
}

// Shifts
// A register-count shift uses BMI2 SHLX/SHRX/SARX (non-destructive, count from
// any GPR) when available; otherwise the legacy CL-based SHL/SAR/SHR, for which
// the pre-allocator has already copied value→dest and count→CL. Immediate-count
// shifts keep SHL/SAR/SHR imm (count stays a constant). `value` is src0, `count`
// src1; the BMI2/non-BMI2 split here mirrors is_bmi2_variable_shift exactly.
fn generate_shift_left(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s2 = src(&ir.ops, op_idx, 1);
    if crate::optimizations::use_bmi2() && !is_const(&ir.operands, s2) {
        let s1 = src(&ir.ops, op_idx, 0);
        cg.asm.shlx(&ir.operands, d, s1, s2, ir.operands.get(d).typ);
    } else {
        cg.asm.shl(&ir.operands, d, s2, ir.operands.get(d).typ);
    }
}

fn generate_shift_right_si(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s2 = src(&ir.ops, op_idx, 1);
    if crate::optimizations::use_bmi2() && !is_const(&ir.operands, s2) {
        let s1 = src(&ir.ops, op_idx, 0);
        cg.asm.sarx(&ir.operands, d, s1, s2, ir.operands.get(d).typ);
    } else {
        cg.asm.sar(&ir.operands, d, s2, ir.operands.get(d).typ);
    }
}

fn generate_shift_right_ui(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s2 = src(&ir.ops, op_idx, 1);
    if crate::optimizations::use_bmi2() && !is_const(&ir.operands, s2) {
        let s1 = src(&ir.ops, op_idx, 0);
        cg.asm.shrx(&ir.operands, d, s1, s2, ir.operands.get(d).typ);
    } else {
        cg.asm.shr(&ir.operands, d, s2, ir.operands.get(d).typ);
    }
}

fn generate_rotate_right(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s2 = src(&ir.ops, op_idx, 1);
    cg.asm.ror(&ir.operands, d, s2, ir.operands.get(d).typ);
}

// Conversions
fn generate_convert_to_fp(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);

    let dest_type = ir.operands.get(d).typ;
    let src_type = ir.operands.get(s).typ;

    debug_assert!(matches!(dest_type, OperandType::FP32 | OperandType::FP64));

    if dest_type == OperandType::FP32 {
        debug_assert!(src_type.is_integer() || src_type == OperandType::FP64);

        if src_type.is_integer() {
            cg.asm.xorps(&ir.operands, d, d, d);
            cg.asm.cvtsi2ss(&ir.operands, d, d, s, src_type);
        } else {
            cg.asm.cvtsd2ss(&ir.operands, d, d, s);
            // C# GenerateZeroUpper96
            cg.asm.movq(&ir.operands, d, d);
            cg.asm.pshufd(&ir.operands, d, d, 0xfc);
        }
    } else {
        debug_assert!(src_type.is_integer() || src_type == OperandType::FP32);

        if src_type.is_integer() {
            cg.asm.xorps(&ir.operands, d, d, d);
            cg.asm.cvtsi2sd(&ir.operands, d, d, s, src_type);
        } else {
            cg.asm.cvtss2sd(&ir.operands, d, d, s);
            // C# GenerateZeroUpper64
            cg.asm.movq(&ir.operands, d, d);
        }
    }
}

fn generate_convert_i64_to_i32(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    cg.asm.mov(&ir.operands, d, s, OperandType::I32);
}

fn generate_zero_extend8(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    cg.asm.movzx8(&ir.operands, d, s, OperandType::I32);
}

fn generate_zero_extend16(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    cg.asm.movzx16(&ir.operands, d, s, OperandType::I32);
}

fn generate_zero_extend32(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    if same_operand(&ir.operands, d, s) && ir.operands.get(s).typ == OperandType::I32 {
        return;
    }
    cg.asm.mov(&ir.operands, d, s, OperandType::I32);
}

fn generate_sign_extend8(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    cg.asm.movsx8(&ir.operands, d, s, ir.operands.get(d).typ);
}

fn generate_sign_extend16(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    cg.asm.movsx16(&ir.operands, d, s, ir.operands.get(d).typ);
}

fn generate_sign_extend32(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    cg.asm.movsx32(&ir.operands, d, s, ir.operands.get(d).typ);
}

// Call / Tailcall / LoadArgument / StackAlloc
fn generate_call(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let target = src(&ir.ops, op_idx, 0);
    cg.asm.call(&ir.operands, target);
}

fn generate_tailcall(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    write_epilogue(cg, &mut ir.operands);
    let target = src(&ir.ops, op_idx, 0);
    cg.asm.jmp(&ir.operands, target);
}

fn generate_stack_alloc(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let offs_op = src(&ir.ops, op_idx, 0);
    let offs = ir.operands.get(offs_op).value as i32 + cg.call_args_region_size;
    let rsp = ir.operands.register(
        X86Register::RSP.index() as u32,
        RegisterType::Integer,
        OperandType::I64,
    );
    let mem = ir
        .operands
        .memory(OperandType::I64, Some(rsp), None, Multiplier::x1, offs);
    cg.asm.lea(&ir.operands, d, mem, OperandType::I64);
}

// Fill / Spill (rsp-relative stack slots)
// C# GenerateFill: load dest from [rsp + offset + CallArgsRegionSize].
fn generate_fill(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let offset_op = src(&ir.ops, op_idx, 0);
    debug_assert!(is_const(&ir.operands, offset_op));

    let offs = ir.operands.get(offset_op).value as i32 + cg.call_args_region_size;
    let typ = ir.operands.get(d).typ;

    let rsp = gen_reg(&mut ir.operands, X86Register::RSP, OperandType::I64);
    let mem = ir
        .operands
        .memory(typ, Some(rsp), None, Multiplier::x1, offs);

    generate_load_to(cg, ir, mem, d);
}

// C# GenerateSpill (baseOffset = CallArgsRegionSize) / GenerateSpillArg
// (baseOffset = 0).
fn generate_spill(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
    with_region_base: bool,
) {
    let offset_op = src(&ir.ops, op_idx, 0);
    let source = src(&ir.ops, op_idx, 1);
    debug_assert!(is_const(&ir.operands, offset_op));

    let base = if with_region_base {
        cg.call_args_region_size
    } else {
        0
    };
    let offs = ir.operands.get(offset_op).value as i32 + base;
    let typ = ir.operands.get(source).typ;

    let rsp = gen_reg(&mut ir.operands, X86Register::RSP, OperandType::I64);
    let mem = ir
        .operands
        .memory(typ, Some(rsp), None, Multiplier::x1, offs);

    generate_store_to(cg, ir, mem, source);
}

// C# private GenerateLoad(context, address, value).
fn generate_load_to(
    cg: &mut CodeGenContext,
    ir: &TranslationContext,
    address: OperandIndex,
    value: OperandIndex,
) {
    match ir.operands.get(value).typ {
        OperandType::I32 => cg.asm.mov(&ir.operands, value, address, OperandType::I32),
        OperandType::I64 => cg.asm.mov(&ir.operands, value, address, OperandType::I64),
        OperandType::FP32 => cg.asm.movd(&ir.operands, value, address),
        OperandType::FP64 => cg.asm.movq(&ir.operands, value, address),
        OperandType::V128 => cg.asm.movdqu(&ir.operands, value, address),
        t => panic!("generate_load_to: invalid type {t:?}"),
    }
}

// C# private GenerateStore(context, address, value).
fn generate_store_to(
    cg: &mut CodeGenContext,
    ir: &TranslationContext,
    address: OperandIndex,
    value: OperandIndex,
) {
    match ir.operands.get(value).typ {
        OperandType::I32 => cg.asm.mov(&ir.operands, address, value, OperandType::I32),
        OperandType::I64 => cg.asm.mov(&ir.operands, address, value, OperandType::I64),
        OperandType::FP32 => cg.asm.movd(&ir.operands, address, value),
        OperandType::FP64 => cg.asm.movq(&ir.operands, address, value),
        OperandType::V128 => cg.asm.movdqu(&ir.operands, address, value),
        t => panic!("generate_store_to: invalid type {t:?}"),
    }
}

// Misc integer arms (C# parity)
// C# GenerateMemoryBarrier: lock or dword [rsp], 0.
fn generate_memory_barrier(cg: &mut CodeGenContext, _ir: &mut TranslationContext) {
    // MFENCE (3 bytes) is faster than lock or dword [rsp], 0 (5 bytes):
    // no cacheline write, ~10 fewer cycles per barrier.
    // Both provide the same full memory-ordering semantics on x86-TSO.
    cg.asm.mfence();
}

// C# GenerateConditionalSelect: test src1; cmovne dest, src2
// (PreAllocator guarantees dest == src3).
fn generate_conditional_select(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(d).typ;
    let s1_typ = ir.operands.get(s1).typ;

    cg.asm.test(&ir.operands, s1, s1, s1_typ);
    cg.asm
        .cmovcc(&ir.operands, d, s2, typ, X86Condition::NotEqual);
}

// C# GenerateByteSwap.
fn generate_byte_swap(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let typ = ir.operands.get(d).typ;
    cg.asm.bswap(&ir.operands, d, typ);
}

// C# GenerateCountLeadingZeros: BSR + zero-input fixup + XOR invert.
//
// ABM fast path (Armature divergence): LZCNT computes CLZ directly in one
// instruction, including the zero-input case (LZCNT(0) = operand width =
// ARM CLZ(0)), so it is bit-identical to the BSR sequence and needs no
// branch/fixup. Gated by host ABM support; BSR fallback otherwise.
fn generate_count_leading_zeros(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    let typ = ir.operands.get(d).typ;

    if crate::optimizations::use_lzcnt() {
        cg.asm.lzcnt(&ir.operands, d, s, typ);
        return;
    }

    cg.asm.bsr(&ir.operands, d, s, typ);

    let operand_size: i32 = if typ == OperandType::I32 { 32 } else { 64 };
    let operand_mask = operand_size - 1;

    // When the input operand is 0, the result is undefined, however the
    // ZF flag is set. We are supposed to return the operand size on that
    // case. So, add an additional jump to handle that case, by moving the
    // operand size constant to the destination register.
    let ne_label = ir.operands.label();

    cg.asm.jcc(X86Condition::NotEqual, &ir.operands, ne_label);

    let size_const = ir.operands.constant_i32(operand_size | operand_mask);
    cg.asm.mov(&ir.operands, d, size_const, OperandType::I32);

    cg.asm.mark_label(ne_label);

    // BSR returns the zero based index of the last bit set on the operand,
    // starting from the least significant bit. However we are supposed to
    // return the number of 0 bits on the high end. So, we invert the result
    // of the BSR using XOR to get the correct value.
    let mask_const = ir.operands.constant_i32(operand_mask);
    cg.asm.xor(&ir.operands, d, mask_const, OperandType::I32);
}

// C# GenerateMultiply64HighSI: one-operand IMUL (RDX:RAX = RAX * src).
fn generate_multiply64_high_si(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let s = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(s).typ;
    cg.asm.imul1(&ir.operands, s, typ);
}

// C# GenerateMultiply64HighUI: one-operand MUL.
fn generate_multiply64_high_ui(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let s = src(&ir.ops, op_idx, 1);
    let typ = ir.operands.get(s).typ;
    cg.asm.mul(&ir.operands, s, typ);
}

// C# GenerateCompareAndSwap (5-source form = 128-bit cmpxchg16b).
fn generate_compare_and_swap(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let s1 = src(&ir.ops, op_idx, 0);

    if ir.ops.sources(op_idx).len() == 5 {
        let mem = ir
            .operands
            .memory(OperandType::I64, Some(s1), None, Multiplier::x1, 0);
        cg.asm.cmpxchg16b(&ir.operands, mem);
    } else {
        let s3 = src(&ir.ops, op_idx, 2);
        let typ = ir.operands.get(s3).typ;
        let mem = ir.operands.memory(typ, Some(s1), None, Multiplier::x1, 0);
        cg.asm.cmpxchg(&ir.operands, mem, s3);
    }
}

fn generate_compare_and_swap16(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let s1 = src(&ir.ops, op_idx, 0);
    let s3 = src(&ir.ops, op_idx, 2);
    let typ = ir.operands.get(s3).typ;
    let mem = ir.operands.memory(typ, Some(s1), None, Multiplier::x1, 0);
    cg.asm.cmpxchg16(&ir.operands, mem, s3);
}

fn generate_compare_and_swap8(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let s1 = src(&ir.ops, op_idx, 0);
    let s3 = src(&ir.ops, op_idx, 2);
    let typ = ir.operands.get(s3).typ;
    let mem = ir.operands.memory(typ, Some(s1), None, Multiplier::x1, 0);
    cg.asm.cmpxchg8(&ir.operands, mem, s3);
}

// ── Intrinsic (Extended) dispatch  C# GenerateOperation's Extended
//    branch, switch on IntrinsicTable info type ──────────────────────────

fn generate_extended(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    use crate::codegen::x86_64::intrinsic_table::{self, IntrinsicType};
    use crate::ir::types::Intrinsic;

    let intrinsic = ir.ops.get(op_idx).intrinsic;
    let info = intrinsic_table::info(intrinsic);
    let supports_vex = crate::optimizations::supports_vex_encoding();

    match info.typ {
        IntrinsicType::Comis_ => {
            let d = dest(&ir.ops, op_idx);
            let s1 = src(&ir.ops, op_idx, 0);
            let s2 = src(&ir.ops, op_idx, 1);

            match intrinsic {
                Intrinsic::X86Comisdeq => {
                    cg.asm.comisd(&ir.operands, s1, s2);
                    cg.asm.setcc(&ir.operands, d, X86Condition::Equal);
                }
                Intrinsic::X86Comisdge => {
                    cg.asm.comisd(&ir.operands, s1, s2);
                    cg.asm.setcc(&ir.operands, d, X86Condition::AboveOrEqual);
                }
                Intrinsic::X86Comisdlt => {
                    cg.asm.comisd(&ir.operands, s1, s2);
                    cg.asm.setcc(&ir.operands, d, X86Condition::Below);
                }
                Intrinsic::X86Comisseq => {
                    cg.asm.comiss(&ir.operands, s1, s2);
                    cg.asm.setcc(&ir.operands, d, X86Condition::Equal);
                }
                Intrinsic::X86Comissge => {
                    cg.asm.comiss(&ir.operands, s1, s2);
                    cg.asm.setcc(&ir.operands, d, X86Condition::AboveOrEqual);
                }
                Intrinsic::X86Comisslt => {
                    cg.asm.comiss(&ir.operands, s1, s2);
                    cg.asm.setcc(&ir.operands, d, X86Condition::Below);
                }
                _ => unreachable!("non-Comis intrinsic with Comis_ type"),
            }

            cg.asm.movzx8(&ir.operands, d, d, OperandType::I32);
        }

        IntrinsicType::Mxcsr => {
            let offset = src(&ir.ops, op_idx, 0);
            debug_assert!(is_const(&ir.operands, offset));

            let offs = ir.operands.get(offset).value as i32 + cg.call_args_region_size;

            let rsp = gen_reg(&mut ir.operands, X86Register::RSP, OperandType::I64);
            let mem = ir
                .operands
                .memory(OperandType::I32, Some(rsp), None, Multiplier::x1, offs);

            if intrinsic == Intrinsic::X86Ldmxcsr {
                let bits = src(&ir.ops, op_idx, 1);
                cg.asm.mov(&ir.operands, mem, bits, OperandType::I32);
                cg.asm.ldmxcsr(&ir.operands, mem);
            } else if intrinsic == Intrinsic::X86Stmxcsr {
                let d = dest(&ir.ops, op_idx);
                cg.asm.stmxcsr(&ir.operands, mem);
                cg.asm.mov(&ir.operands, d, mem, OperandType::I32);
            }
        }

        IntrinsicType::PopCount => {
            let d = dest(&ir.ops, op_idx);
            let s = src(&ir.ops, op_idx, 0);
            let typ = ir.operands.get(d).typ;
            cg.asm.popcnt(&ir.operands, d, s, typ);
        }

        IntrinsicType::Unary => {
            let d = dest(&ir.ops, op_idx);
            let s = src(&ir.ops, op_idx, 0);
            cg.asm.write_instruction_2(
                &ir.operands,
                info.inst,
                Some(d),
                Some(s),
                OperandType::None,
            );
        }

        IntrinsicType::UnaryToGpr => {
            let d = dest(&ir.ops, op_idx);
            let s = src(&ir.ops, op_idx, 0);
            let dest_typ = ir.operands.get(d).typ;

            if intrinsic == Intrinsic::X86Cvtsi2si {
                if dest_typ == OperandType::I32 {
                    cg.asm.movd(&ir.operands, d, s); // int _mm_cvtsi128_si32
                } else {
                    cg.asm.movq(&ir.operands, d, s); // __int64 _mm_cvtsi128_si64
                }
            } else {
                cg.asm
                    .write_instruction_2(&ir.operands, info.inst, Some(d), Some(s), dest_typ);
            }
        }

        IntrinsicType::Binary => {
            let d = dest(&ir.ops, op_idx);
            let s1 = src(&ir.ops, op_idx, 0);
            let s2 = src(&ir.ops, op_idx, 1);
            cg.asm
                .write_instruction_3(&ir.operands, info.inst, Some(d), Some(s1), Some(s2));
        }

        IntrinsicType::BinaryGpr => {
            let d = dest(&ir.ops, op_idx);
            let s1 = src(&ir.ops, op_idx, 0);
            let s2 = src(&ir.ops, op_idx, 1);
            let s2_typ = ir.operands.get(s2).typ;
            cg.asm.write_instruction_3t(
                &ir.operands,
                info.inst,
                Some(d),
                Some(s1),
                Some(s2),
                s2_typ,
            );
        }

        IntrinsicType::Crc32 => {
            let d = dest(&ir.ops, op_idx);
            let s2 = src(&ir.ops, op_idx, 1);
            let typ = ir.operands.get(d).typ;
            cg.asm
                .write_instruction_2(&ir.operands, info.inst, Some(d), Some(s2), typ);
        }

        IntrinsicType::BinaryImm => {
            let d = dest(&ir.ops, op_idx);
            let s1 = src(&ir.ops, op_idx, 0);
            let s2 = src(&ir.ops, op_idx, 1);
            debug_assert!(is_const(&ir.operands, s2));
            let imm = ir.operands.get(s2).value as u8;
            cg.asm
                .write_instruction_2imm(&ir.operands, info.inst, Some(d), Some(s1), imm);
        }

        IntrinsicType::Ternary => {
            let d = dest(&ir.ops, op_idx);
            let s1 = src(&ir.ops, op_idx, 0);
            let s2 = src(&ir.ops, op_idx, 1);
            let s3 = src(&ir.ops, op_idx, 2);

            use crate::codegen::x86_64::instruction::X86Instruction as XI;
            if info.inst == XI::Blendvpd && supports_vex {
                cg.asm.write_instruction_4(
                    &ir.operands,
                    XI::Vblendvpd,
                    Some(d),
                    Some(s1),
                    Some(s2),
                    s3,
                );
            } else if info.inst == XI::Blendvps && supports_vex {
                cg.asm.write_instruction_4(
                    &ir.operands,
                    XI::Vblendvps,
                    Some(d),
                    Some(s1),
                    Some(s2),
                    s3,
                );
            } else if info.inst == XI::Pblendvb && supports_vex {
                cg.asm.write_instruction_4(
                    &ir.operands,
                    XI::Vpblendvb,
                    Some(d),
                    Some(s1),
                    Some(s2),
                    s3,
                );
            } else {
                // Legacy form: implied XMM0 operand (PreAllocator pinned it).
                debug_assert_eq!(ir.operands.get(s3).reg_index(), 0);
                cg.asm
                    .write_instruction_3(&ir.operands, info.inst, Some(d), Some(s1), Some(s2));
            }
        }

        IntrinsicType::TernaryImm => {
            let d = dest(&ir.ops, op_idx);
            let s1 = src(&ir.ops, op_idx, 0);
            let s2 = src(&ir.ops, op_idx, 1);
            let s3 = src(&ir.ops, op_idx, 2);
            debug_assert!(is_const(&ir.operands, s3));
            let imm = ir.operands.get(s3).value as u8;
            cg.asm.write_instruction_3imm(
                &ir.operands,
                info.inst,
                Some(d),
                Some(s1),
                Some(s2),
                imm,
            );
        }

        IntrinsicType::Fma => {
            let d = dest(&ir.ops, op_idx);
            let s2 = src(&ir.ops, op_idx, 1);
            let s3 = src(&ir.ops, op_idx, 2);
            debug_assert!(supports_vex);
            cg.asm
                .write_instruction_3(&ir.operands, info.inst, Some(d), Some(s2), Some(s3));
        }
    }
}

// Vector operations.

fn generate_vector_zero(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    cg.asm.xorps(&ir.operands, d, d, d);
}

fn generate_vector_one(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    cg.asm.pcmpeqw(&ir.operands, d, d, d);
}

fn generate_vector_create_scalar(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s = src(&ir.ops, op_idx, 0);
    if ir.operands.get(s).typ == OperandType::I32 {
        cg.asm.movd(&ir.operands, d, s);
    } else {
        cg.asm.movq(&ir.operands, d, s);
    }
}

// C# BitUtils.RotateRight(value, shift, size) for the pshufd masks.
fn rotate_right_bits(value: u32, shift: u32, size: u32) -> u32 {
    if shift == 0 || shift == size {
        return value & ((1u32 << size) - 1);
    }
    let m = (1u32 << size) - 1;
    let v = value & m;
    ((v >> shift) | (v << (size - shift))) & m
}

fn generate_vector_extract(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx); // Value
    let s1 = src(&ir.ops, op_idx, 0); // Vector
    let s2 = src(&ir.ops, op_idx, 1); // Index
    debug_assert!(is_const(&ir.operands, s2));
    let index = ir.operands.get(s2).value as u8;
    let dest_typ = ir.operands.get(d).typ;

    if dest_typ == OperandType::I32 {
        if index == 0 {
            cg.asm.movd(&ir.operands, d, s1);
        } else if crate::optimizations::supports_sse41() {
            cg.asm.pextrd(&ir.operands, d, s1, index);
        } else {
            let mask0 = rotate_right_bits(0b11_10_01_00, index as u32 * 2, 8);
            let mask1 = rotate_right_bits(0b11_10_01_00, 8 - index as u32 * 2, 8);
            cg.asm.pshufd(&ir.operands, s1, s1, mask0 as u8);
            cg.asm.movd(&ir.operands, d, s1);
            cg.asm.pshufd(&ir.operands, s1, s1, mask1 as u8);
        }
    } else if dest_typ == OperandType::I64 {
        if index == 0 {
            cg.asm.movq(&ir.operands, d, s1);
        } else if crate::optimizations::supports_sse41() {
            cg.asm.pextrq(&ir.operands, d, s1, index);
        } else {
            const MASK: u8 = 0b01_00_11_10;
            cg.asm.pshufd(&ir.operands, s1, s1, MASK);
            cg.asm.movq(&ir.operands, d, s1);
            cg.asm.pshufd(&ir.operands, s1, s1, MASK);
        }
    } else {
        // Floating-point types.
        if (index >= 2 && dest_typ == OperandType::FP32)
            || (index == 1 && dest_typ == OperandType::FP64)
        {
            cg.asm.movhlps(&ir.operands, d, d, s1);
            cg.asm.movq(&ir.operands, d, d);
        } else {
            cg.asm.movq(&ir.operands, d, s1);
        }
        if dest_typ == OperandType::FP32 {
            cg.asm.pshufd(&ir.operands, d, d, 0xfc | (index & 1));
        }
    }
}

fn generate_vector_extract16(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let index = ir.operands.get(s2).value as u8;
    cg.asm.pextrw(&ir.operands, d, s1, index);
}

fn generate_vector_extract8(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let index = ir.operands.get(s2).value as u8;

    if crate::optimizations::supports_sse41() {
        cg.asm.pextrb(&ir.operands, d, s1, index);
    } else {
        cg.asm.pextrw(&ir.operands, d, s1, index >> 1);
        if index & 1 != 0 {
            let eight = ir.operands.constant_i32(8);
            cg.asm.shr(&ir.operands, d, eight, OperandType::I32);
        } else {
            cg.asm.movzx8(&ir.operands, d, d, OperandType::I32);
        }
    }
}

fn generate_vector_insert(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0); // Vector
    let s2 = src(&ir.ops, op_idx, 1); // Value
    let s3 = src(&ir.ops, op_idx, 2); // Index
    debug_assert!(is_const(&ir.operands, s3));
    let index = ir.operands.get(s3).value as u8;
    let val_typ = ir.operands.get(s2).typ;
    let sse41 = crate::optimizations::supports_sse41();

    // C# local function InsertIntSse2(words).
    let insert_int_sse2 = |cg: &mut CodeGenContext, ir: &mut TranslationContext, words: u8| {
        if !same_operand(&ir.operands, d, s1) {
            cg.asm.movdqu(&ir.operands, d, s1);
        }
        for word in 0..words {
            // Insert lower 16-bits, then move the next word down.
            cg.asm.pinsrw(&ir.operands, d, d, s2, index * words + word);
            let sixteen = ir.operands.constant_i32(16);
            let s2_typ = ir.operands.get(s2).typ;
            cg.asm.ror(&ir.operands, s2, sixteen, s2_typ);
        }
    };

    if val_typ == OperandType::I32 {
        if sse41 {
            cg.asm.pinsrd(&ir.operands, d, s1, s2, index);
        } else {
            insert_int_sse2(cg, ir, 2);
        }
    } else if val_typ == OperandType::I64 {
        if sse41 {
            cg.asm.pinsrq(&ir.operands, d, s1, s2, index);
        } else {
            insert_int_sse2(cg, ir, 4);
        }
    } else if val_typ == OperandType::FP32 {
        if index != 0 {
            if sse41 {
                cg.asm.insertps(&ir.operands, d, s1, s2, index << 4);
            } else if same_operand(&ir.operands, s1, s2) {
                let mask = (0b11_10_01_00u32 & !(0b11 << (index * 2))) as u8;
                cg.asm.pshufd(&ir.operands, d, s1, mask);
            } else {
                let mask0 = rotate_right_bits(0b11_10_01_00, index as u32 * 2, 8) as u8;
                let mask1 = rotate_right_bits(0b11_10_01_00, 8 - index as u32 * 2, 8) as u8;
                cg.asm.pshufd(&ir.operands, s1, s1, mask0);
                cg.asm.movss(&ir.operands, d, s1, s2);
                cg.asm.pshufd(&ir.operands, d, d, mask1);
                if !same_operand(&ir.operands, d, s1) {
                    cg.asm.pshufd(&ir.operands, s1, s1, mask1);
                }
            }
        } else {
            cg.asm.movss(&ir.operands, d, s1, s2);
        }
    } else {
        // FP64
        if index != 0 {
            cg.asm.movlhps(&ir.operands, d, s1, s2);
        } else {
            cg.asm.movsd(&ir.operands, d, s1, s2);
        }
    }
}

fn generate_vector_insert16(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let s3 = src(&ir.ops, op_idx, 2);
    let index = ir.operands.get(s3).value as u8;
    cg.asm.pinsrw(&ir.operands, d, s1, s2, index);
}

fn generate_vector_insert8(
    cg: &mut CodeGenContext,
    op_idx: crate::ir::operation::OperationIdx,
    ir: &mut TranslationContext,
) {
    // Without SSE 4.1 the PreAllocator lowers this away.
    debug_assert!(crate::optimizations::supports_sse41());
    let d = dest(&ir.ops, op_idx);
    let s1 = src(&ir.ops, op_idx, 0);
    let s2 = src(&ir.ops, op_idx, 1);
    let s3 = src(&ir.ops, op_idx, 2);
    let index = ir.operands.get(s3).value as u8;
    cg.asm.pinsrb(&ir.operands, d, s1, s2, index);
}

// Prologue / Epilogue
const PAGE_SIZE: i32 = 0x1000;
const STACK_GUARD_SIZE: i32 = 0x2000;

// C# WritePrologue (UnwindInfo entries deferred to Windows bring-up
// the byte stream is identical).
fn write_prologue(cg: &mut CodeGenContext, ir: &mut TranslationContext) {
    use crate::codegen::x86_64::calling_convention as cc;

    let rsp = gen_reg(&mut ir.operands, X86Register::RSP, OperandType::I64);

    let mut mask =
        (cc::get_int_callee_saved_registers() & cg.alloc_result.int_used_registers as i32) as u32;

    while mask != 0 {
        let bit = mask.trailing_zeros();
        let reg = gen_reg(
            &mut ir.operands,
            X86Register::from_index(bit),
            OperandType::I64,
        );
        cg.asm.push(&ir.operands, reg, OperandType::I64);
        mask &= !(1 << bit);
    }

    let reserved_stack_size =
        cg.call_args_region_size + cg.alloc_result.spill_region_size + cg.xmm_save_region_size;

    if reserved_stack_size >= STACK_GUARD_SIZE {
        generate_inline_stack_probe(cg, ir, reserved_stack_size);
    }

    if reserved_stack_size != 0 {
        let sz = ir.operands.constant_i64(reserved_stack_size as i64);
        cg.asm.sub(&ir.operands, rsp, sz, OperandType::I64);
    }

    // XMM callee-saved registers are saved into the reserved region,
    // from the top down.
    let mut offset = reserved_stack_size;

    let mut mask =
        (cc::get_vec_callee_saved_registers() & cg.alloc_result.vec_used_registers as i32) as u32;

    while mask != 0 {
        let bit = mask.trailing_zeros();

        offset -= 16;

        let mem = ir
            .operands
            .memory(OperandType::V128, Some(rsp), None, Multiplier::x1, offset);
        let xmm_reg = ir
            .operands
            .register(bit, RegisterType::Vector, OperandType::V128);
        cg.asm.movdqu(&ir.operands, mem, xmm_reg);

        mask &= !(1 << bit);
    }
}

// C# WriteEpilogue.
fn write_epilogue(cg: &mut CodeGenContext, operands: &mut OperandArena) {
    use crate::codegen::x86_64::calling_convention as cc;

    let rsp = gen_reg(operands, X86Register::RSP, OperandType::I64);

    let reserved_stack_size =
        cg.call_args_region_size + cg.alloc_result.spill_region_size + cg.xmm_save_region_size;

    let mut offset = reserved_stack_size;

    let mut mask =
        (cc::get_vec_callee_saved_registers() & cg.alloc_result.vec_used_registers as i32) as u32;

    while mask != 0 {
        let bit = mask.trailing_zeros();

        offset -= 16;

        let mem = operands.memory(OperandType::V128, Some(rsp), None, Multiplier::x1, offset);
        let xmm_reg = operands.register(bit, RegisterType::Vector, OperandType::V128);
        cg.asm.movdqu(operands, xmm_reg, mem);

        mask &= !(1 << bit);
    }

    if reserved_stack_size != 0 {
        let sz = operands.constant_i64(reserved_stack_size as i64);
        cg.asm.add(operands, rsp, sz, OperandType::I64);
    }

    // C# pops highest bit first  the exact reverse of the prologue's
    // lowest-first pushes.
    let mut mask =
        (cc::get_int_callee_saved_registers() & cg.alloc_result.int_used_registers as i32) as u32;

    while mask != 0 {
        let bit = 31 - mask.leading_zeros();
        let reg = gen_reg(operands, X86Register::from_index(bit), OperandType::I64);
        cg.asm.pop(operands, reg, OperandType::I64);
        mask &= !(1 << bit);
    }
}

// C# GenerateInlineStackProbe: touch each page of a large frame so the
// OS maps it (Windows lazy stack commit; harmless on Linux).
fn generate_inline_stack_probe(cg: &mut CodeGenContext, ir: &mut TranslationContext, size: i32) {
    let page_mask = PAGE_SIZE - 1;
    let size = (size + page_mask) & !page_mask;

    let rsp = gen_reg(&mut ir.operands, X86Register::RSP, OperandType::I64);
    let temp = gen_reg(
        &mut ir.operands,
        crate::codegen::x86_64::calling_convention::get_int_return_register(),
        OperandType::I32,
    );

    let mut offset = PAGE_SIZE;
    while offset < size {
        let mem = ir
            .operands
            .memory(OperandType::I32, Some(rsp), None, Multiplier::x1, -offset);
        cg.asm.mov(&ir.operands, temp, mem, OperandType::I32);
        offset += PAGE_SIZE;
    }
}
