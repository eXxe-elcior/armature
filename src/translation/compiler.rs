// Port of Translation/Compiler.cs, CompilerContext.cs, and CompilerOptions.cs.
// LCQ renames guest registers to locals; HCQ adds SSA, optimization, and linear scan.

use crate::codegen::compiled_function::CompiledFunction;
use crate::codegen::x86_64::code_generator;
use crate::translation::cfg::ControlFlowGraph;
use crate::translation::register_to_local;

/// C# [Flags] enum CompilerOptions  same bit positions. Plain u32 to
/// avoid adding a crate dependency.
pub mod compiler_options {
    pub const NONE: u32 = 0;
    pub const SSA_FORM: u32 = 1 << 0;
    pub const OPTIMIZE: u32 = 1 << 1;
    pub const LSRA: u32 = 1 << 2;
    pub const RELOCATABLE: u32 = 1 << 3;

    pub const MEDIUM_CQ: u32 = SSA_FORM | OPTIMIZE;
    pub const HIGH_CQ: u32 = SSA_FORM | OPTIMIZE | LSRA;
}

/// Wrapper for CompilerOptions with convenience methods matching C# HasFlag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompilerOptions(u32);

impl CompilerOptions {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub fn contains(self, flag: u32) -> bool {
        self.0 & flag == flag
    }
}

/// C# readonly struct CompilerContext  carries cfg + options through codegen.
pub struct CompilerContext {
    pub cfg: ControlFlowGraph,
    pub func_arg_types: Vec<crate::ir::types::OperandType>,
    pub func_return_type: crate::ir::types::OperandType,
    pub options: CompilerOptions,
}

impl CompilerContext {
    pub fn new(
        cfg: ControlFlowGraph,
        func_arg_types: Vec<crate::ir::types::OperandType>,
        func_return_type: crate::ir::types::OperandType,
        options: CompilerOptions,
    ) -> Self {
        Self {
            cfg,
            func_arg_types,
            func_return_type,
            options,
        }
    }
}

/// C# static class Compiler.
pub fn compile(
    cfg: ControlFlowGraph,
    arg_types: Vec<crate::ir::types::OperandType>,
    ret_type: crate::ir::types::OperandType,
    options: CompilerOptions,
    ctx: &mut crate::ir::translation::TranslationContext,
) -> CompiledFunction {
    let cctx = CompilerContext::new(cfg, arg_types, ret_type, options);
    compile_with_context(cctx, ctx)
}

pub fn compile_with_context(
    mut cctx: CompilerContext,
    ctx: &mut crate::ir::translation::TranslationContext,
) -> CompiledFunction {
    // C#: if (options.HasFlag(CompilerOptions.Optimize)) → TailMerge.RunPass(cctx)
    if cctx.options.contains(compiler_options::OPTIMIZE) {
        crate::translation::tail_merge::run_pass(&mut cctx, ctx);
    }

    if cctx.options.contains(compiler_options::SSA_FORM) {
        crate::translation::dominance::find_dominators(&cctx.cfg, &mut ctx.blocks);
        crate::translation::dominance::find_dominance_frontiers(&mut ctx.blocks);
        crate::translation::ssa::construct(&cctx.cfg, ctx);
    } else {
        // LCQ path: RegisterToLocal.Rename
        register_to_local::rename(&cctx.cfg, ctx);
    }

    code_generator::generate(&mut cctx, ctx)
}
