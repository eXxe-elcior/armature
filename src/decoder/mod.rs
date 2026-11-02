pub mod decode_fns;
pub mod decode_one;
pub mod helpers;
pub mod instruction_name;
pub mod opcode_kind;
pub mod table;
pub mod table_entries; // pub for the differential-test vector generator
pub mod utilities;

// Re-export commonly-used decoder types for convenience
pub use instruction_name::InstructionName;
// Re-export commonly-used decoder types for convenience
pub use opcode_kind::OpCodeKind;
// Re-export commonly-used decoder types for convenience
pub use utilities::{DecodedInstruction, EmitFn, InstructionInfo, RegisterSize};
