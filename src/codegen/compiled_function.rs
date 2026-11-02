// Port of CodeGen/CompiledFunction.cs + CodeGen/Unwinding/UnwindInfo.cs.
// UnwindInfo is a placeholder until unwinding tables are needed.

use crate::codegen::linking::RelocationInfo;

#[derive(Debug, Clone)]
pub struct CompiledFunction {
    pub code: Vec<u8>,
    pub reloc_info: RelocationInfo,
}

impl CompiledFunction {
    pub fn new(code: Vec<u8>, reloc_info: RelocationInfo) -> Self {
        Self { code, reloc_info }
    }
}
