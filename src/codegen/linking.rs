use crate::ir::types::Symbol;

#[derive(Debug, Clone, Copy)]
pub struct RelocationEntry {
    pub position: i32,
    pub symbol: Symbol,
}

impl RelocationEntry {
    pub const STRIDE: i32 = 13;

    pub fn new(position: i32, symbol: Symbol) -> Self {
        Self { position, symbol }
    }
}

#[derive(Debug, Clone)]
pub struct RelocationInfo {
    entries: Vec<RelocationEntry>,
}

impl RelocationInfo {
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn new(entries: Vec<RelocationEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[RelocationEntry] {
        &self.entries
    }
}
