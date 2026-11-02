// Block decoding, control-flow analysis, compilation, and dispatch.

pub mod block_decoder;
pub mod cache;
pub mod cfg;
pub mod compiler;
pub mod dominance;
pub mod ptc;
pub mod register_to_local;
pub mod register_usage;
pub mod ssa;
pub mod stubs;
pub mod tail_merge;
pub mod trace;
pub mod translated_function;
pub mod translator;
