use crate::decoder::table::lookup_entry;
use crate::decoder::utilities::{DecodeError, DecodedInstruction};

pub fn decode_one(raw: u32, addr: u64) -> Result<DecodedInstruction, DecodeError> {
    let entry = lookup_entry(raw).ok_or(DecodeError::Undefined(raw))?;
    let (kind, reg_size) = (entry.decode)(raw, addr).ok_or(DecodeError::Undefined(raw))?;
    Ok(DecodedInstruction {
        address: addr,
        raw,
        info: entry.info,
        reg_size,
        kind,
    })
}
