use crate::ir::operation::OperationIdx;
use crate::ir::types::{Multiplier, OperandKind, OperandType, RegisterType, Symbol};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperandIndex(pub u32);

define_index!(OperandIndex);

impl OperandIndex {
    /// Sentinel for "no operand"  matches the `u32::MAX` NONE convention
    /// already used by the src/dst/phi pools (see operation.rs).
    pub const NONE: OperandIndex = OperandIndex(u32::MAX);

    #[inline]
    pub fn to_option(self) -> Option<OperandIndex> {
        if self.0 == u32::MAX { None } else { Some(self) }
    }
}

impl From<Option<OperandIndex>> for OperandIndex {
    #[inline]
    fn from(value: Option<OperandIndex>) -> Self {
        value.unwrap_or(OperandIndex::NONE)
    }
}

#[derive(Debug, Clone)]
pub struct OperandData {
    pub kind: OperandKind,
    pub typ: OperandType,
    pub value: u64,
    pub symbol: Symbol,
    // Sentinel-encoded (NONE = OperandIdx::NONE) rather than Option<OperandIdx>
    // to save 8 bytes/node  OperandIdx has no niche so the Option would pad to
    // 8 bytes each. Read through mem_base()/mem_index() for Option semantics.
    pub mem_base: OperandIndex,
    pub mem_index: OperandIndex,
    pub mem_scale: Multiplier,
    pub mem_disp: i32,
}

impl Default for OperandData {
    fn default() -> Self {
        Self {
            kind: OperandKind::None,
            typ: OperandType::None,
            value: 0,
            symbol: Symbol::default(),
            mem_base: OperandIndex::NONE,
            mem_index: OperandIndex::NONE,
            mem_scale: Multiplier::x1,
            mem_disp: 0,
        }
    }
}

/// Def-use bookkeeping, split out of `OperandData` into a parallel arena array
/// (indexed by the same `OperandIdx`). Only LocalVariable/Memory operands ever
/// populate these chains, but interned Constants/Registers  the majority of
/// the arena  used to carry ~32 dead bytes here. Splitting keeps the hot
/// `OperandData` at 40 bytes so codegen/regalloc/assembler walks (which never
/// read def-use) get ~2 nodes per cache line; the opt passes that need the
/// chains index the cold array.
#[derive(Debug, Clone)]
pub struct DefUse {
    pub uses_inline: [OperationIdx; 2],
    pub uses_count: u16,
    pub uses_start: u32,
    pub uses_cap: u16,
    pub assigns_inline: [OperationIdx; 1],
    pub assigns_count: u16,
    pub assigns_start: u32,
    pub assigns_cap: u16,
}

impl Default for DefUse {
    fn default() -> Self {
        Self {
            uses_inline: [OperationIdx(0); 2],
            uses_count: 0,
            uses_start: u32::MAX,
            uses_cap: 0,
            assigns_inline: [OperationIdx(0); 1],
            assigns_count: 0,
            assigns_start: u32::MAX,
            assigns_cap: 0,
        }
    }
}

impl OperandData {
    #[inline]
    pub fn mem_base(&self) -> Option<OperandIndex> {
        self.mem_base.to_option()
    }

    #[inline]
    pub fn mem_index(&self) -> Option<OperandIndex> {
        self.mem_index.to_option()
    }

    #[inline]
    pub fn reg_index(&self) -> u32 {
        (self.value as u32) & 0x00ff_ffff
    }

    #[inline]
    pub fn reg_type(&self) -> RegisterType {
        RegisterType::from_raw((self.value >> 24) as u32)
    }

    #[inline]
    pub fn local_number(&self) -> u32 {
        debug_assert!(self.kind == OperandKind::LocalVariable);
        self.value as u32
    }

    #[inline]
    pub fn as_u8(&self) -> u8 {
        self.value as u8
    }

    #[inline]
    pub fn as_i16(&self) -> i16 {
        self.value as i16
    }

    #[inline]
    pub fn as_i32(&self) -> i32 {
        self.value as i32
    }

    #[inline]
    pub fn as_i64(&self) -> i64 {
        self.value as i64
    }

    #[inline]
    pub fn as_f32(&self) -> f32 {
        f32::from_bits(self.value as u32)
    }

    #[inline]
    pub fn as_f64(&self) -> f64 {
        f64::from_bits(self.value)
    }

    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.value
    }

    #[inline]
    pub fn number_local(&mut self, number: u32) {
        debug_assert!(self.kind == OperandKind::LocalVariable);
        self.value = number as u64;
    }
}

#[derive(Clone)]
pub struct InternalTable {
    slots: [InternalSlot; 256],
}

#[derive(Clone, Copy, Default)]
struct InternalSlot {
    kind: OperandKind,
    typ: OperandType,
    value: u64,
    symbol: Symbol,
    idx: Option<OperandIndex>,
}

impl InternalTable {
    pub fn new() -> Self {
        Self {
            slots: [InternalSlot::default(); 256],
        }
    }

    pub fn reset(&mut self) {
        self.slots = [InternalSlot::default(); 256];
    }

    pub fn get_or_insert(
        &mut self,
        kind: OperandKind,
        typ: OperandType,
        value: u64,
        symbol: Symbol,
        operands: &mut Vec<OperandData>,
        def_use: &mut Vec<DefUse>,
    ) -> OperandIndex {
        // Push to operands + the parallel def_use array in lockstep so the two
        // arrays stay index-aligned (the whole-arena invariant).
        let push = |operands: &mut Vec<OperandData>, def_use: &mut Vec<DefUse>| -> OperandIndex {
            let idx = OperandIndex(operands.len() as u32);
            operands.push(OperandData {
                kind,
                typ,
                value,
                symbol,
                ..Default::default()
            });
            def_use.push(DefUse::default());
            idx
        };

        if !matches!(kind, OperandKind::Constant | OperandKind::Register) {
            return push(operands, def_use);
        }

        let hash = hash_operand(kind, typ, value) as usize;

        for i in 0..8 {
            let slot = &self.slots[(hash + i) & 0xff];
            if slot.kind == kind && slot.typ == typ && slot.value == value && slot.symbol == symbol
            {
                return slot.idx.unwrap();
            }
        }

        for i in 0..8 {
            if self.slots[(hash + i) & 0xff].idx.is_none() {
                let idx = push(operands, def_use);
                self.slots[(hash + i) & 0xff] = InternalSlot {
                    kind,
                    typ,
                    value,
                    symbol,
                    idx: Some(idx),
                };
                return idx;
            }
        }

        push(operands, def_use)
    }
}

pub struct OperandArena {
    operands: Vec<OperandData>,
    /// Parallel to `operands` (same index). See `DefUse`.
    def_use: Vec<DefUse>,
    intern: InternalTable,
    uses_pool: Vec<OperationIdx>,
    assigns_pool: Vec<OperationIdx>,
}

impl OperandArena {
    pub fn new() -> Self {
        Self {
            operands: Vec::new(),
            def_use: Vec::new(),
            intern: InternalTable::new(),
            uses_pool: Vec::new(),
            assigns_pool: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            operands: Vec::with_capacity(capacity),
            def_use: Vec::with_capacity(capacity),
            intern: InternalTable::new(),
            uses_pool: Vec::with_capacity(capacity * 2),
            assigns_pool: Vec::with_capacity(capacity),
        }
    }

    /// Clear all state for arena reuse across translations, retaining capacity.
    /// MUST reset every field (incl. the intern table) or stale operands from
    /// the previous function leak in. Verified by sequential vector translation.
    pub fn reset(&mut self) {
        self.operands.clear();
        self.def_use.clear();
        self.uses_pool.clear();
        self.assigns_pool.clear();
        self.intern.reset();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.operands.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.operands.is_empty()
    }

    #[inline]
    pub fn get(&self, idx: OperandIndex) -> &OperandData {
        &self.operands[idx.0 as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, idx: OperandIndex) -> &mut OperandData {
        &mut self.operands[idx.0 as usize]
    }

    fn allocate(&mut self, data: OperandData) -> OperandIndex {
        let idx = OperandIndex(self.operands.len() as u32);
        self.operands.push(data);
        self.def_use.push(DefUse::default());
        idx
    }

    pub fn constant_i32(&mut self, value: i32) -> OperandIndex {
        self.constant_i32_with_symbol(value, Symbol::default())
    }

    pub fn constant_i32_with_symbol(&mut self, value: i32, symbol: Symbol) -> OperandIndex {
        self.intern.get_or_insert(
            OperandKind::Constant,
            OperandType::I32,
            value as u32 as u64,
            symbol,
            &mut self.operands,
            &mut self.def_use,
        )
    }

    pub fn constant_i64(&mut self, value: i64) -> OperandIndex {
        self.constant_i64_with_symbol(value, Symbol::default())
    }

    pub fn constant_i64_with_symbol(&mut self, value: i64, symbol: Symbol) -> OperandIndex {
        self.intern.get_or_insert(
            OperandKind::Constant,
            OperandType::I64,
            value as u64,
            symbol,
            &mut self.operands,
            &mut self.def_use,
        )
    }

    pub fn constant_u64(&mut self, value: u64) -> OperandIndex {
        self.intern.get_or_insert(
            OperandKind::Constant,
            OperandType::I64,
            value,
            Symbol::default(),
            &mut self.operands,
            &mut self.def_use,
        )
    }

    pub fn constant_f32(&mut self, value: f32) -> OperandIndex {
        self.intern.get_or_insert(
            OperandKind::Constant,
            OperandType::FP32,
            value.to_bits() as u64,
            Symbol::default(),
            &mut self.operands,
            &mut self.def_use,
        )
    }

    pub fn constant_f64(&mut self, value: f64) -> OperandIndex {
        self.intern.get_or_insert(
            OperandKind::Constant,
            OperandType::FP64,
            value.to_bits(),
            Symbol::default(),
            &mut self.operands,
            &mut self.def_use,
        )
    }

    pub fn constant_bool(&mut self, value: bool) -> OperandIndex {
        self.constant_i32(if value { 1 } else { 0 })
    }

    pub fn constant_int(&mut self, typ: OperandType, value: i64) -> OperandIndex {
        match typ {
            OperandType::I32 => self.constant_i32(value as i32),
            OperandType::I64 => self.constant_i64(value),
            _ => panic!("constant_int: expected I32/I64, got {:?}", typ),
        }
    }

    pub fn constant_ptr<T>(&mut self, reference: &T, symbol: Symbol) -> OperandIndex {
        let value = reference as *const T as usize as u64;
        self.constant_i64_with_symbol(value as i64, symbol)
    }

    pub fn register(
        &mut self,
        index: u32,
        reg_type: RegisterType,
        typ: OperandType,
    ) -> OperandIndex {
        let value = (index & 0x00ff_ffff) as u64 | ((reg_type as u64) << 24);
        self.intern.get_or_insert(
            OperandKind::Register,
            typ,
            value,
            Symbol::default(),
            &mut self.operands,
            &mut self.def_use,
        )
    }

    pub fn local(&mut self, typ: OperandType) -> OperandIndex {
        // C# pre-allocates 4 slots for uses + 1 slot for assignments via
        // bump-arena (Allocators.References) per local at creation  cheap
        // because arena bump is 2-3 instructions. Guarantees zero growth for
        // the first 4 uses / 1 assign.
        //
        // Rust: Vec::with_capacity(4) calls the global allocator per local,
        // which is not cheap. We inline up to 2 uses / 1 assign directly in
        // OperandData fields (zero additional allocation). Only at edge 3+
        // (uses) or edge 2+ (assigns) do we push into shared pool Vecs.
        // Saves allocation for the majority of locals (≤2 uses, 1 assign)
        // at the cost of a rare Vec::push when migrating to pool storage.
        self.allocate(OperandData {
            kind: OperandKind::LocalVariable,
            typ,
            ..Default::default()
        })
    }

    pub fn label(&mut self) -> OperandIndex {
        self.allocate(OperandData {
            kind: OperandKind::Label,
            ..Default::default()
        })
    }

    pub fn undef(&mut self) -> OperandIndex {
        self.allocate(OperandData {
            kind: OperandKind::Undefined,
            ..Default::default()
        })
    }

    pub fn memory(
        &mut self,
        typ: OperandType,
        base: Option<OperandIndex>,
        index: Option<OperandIndex>,
        scale: Multiplier,
        displacement: i32,
    ) -> OperandIndex {
        self.allocate(OperandData {
            kind: OperandKind::Memory,
            typ,
            mem_base: base.into(),
            mem_index: index.into(),
            mem_scale: scale,
            mem_disp: displacement,
            ..Default::default()
        })
    }

    pub fn const_zero(&mut self, typ: OperandType) -> OperandIndex {
        match typ {
            OperandType::I32 => self.constant_i32(0),
            OperandType::I64 => self.constant_i64(0),
            OperandType::FP32 => self.constant_f32(0.0),
            OperandType::FP64 => self.constant_f64(0.0),
            OperandType::V128 => self.intern.get_or_insert(
                OperandKind::Constant,
                OperandType::V128,
                0,
                Symbol::default(),
                &mut self.operands,
                &mut self.def_use,
            ),
            OperandType::None => panic!("const_zero called with OperandType::None"),
        }
    }

    pub fn into_inner(self) -> Vec<OperandData> {
        self.operands
    }

    // Def-use chain pool methods
    #[inline]
    pub fn uses_count(&self, operand: OperandIndex) -> u16 {
        self.def_use[operand.0 as usize].uses_count
    }

    #[inline]
    pub fn assigns_count(&self, operand: OperandIndex) -> u16 {
        self.def_use[operand.0 as usize].assigns_count
    }

    #[inline]
    pub fn uses(&self, operand: OperandIndex) -> &[OperationIdx] {
        let d = &self.def_use[operand.0 as usize];
        if d.uses_start != u32::MAX {
            &self.uses_pool[d.uses_start as usize..][..d.uses_count as usize]
        } else {
            &d.uses_inline[..d.uses_count as usize]
        }
    }

    #[inline]
    pub fn assignments(&self, operand: OperandIndex) -> &[OperationIdx] {
        let d = &self.def_use[operand.0 as usize];
        if d.assigns_start != u32::MAX {
            &self.assigns_pool[d.assigns_start as usize..][..d.assigns_count as usize]
        } else {
            &d.assigns_inline[..d.assigns_count as usize]
        }
    }

    fn pool_use_push(&mut self, operand: OperandIndex, op: OperationIdx) {
        let (uses_count, uses_start) = {
            let d = &self.def_use[operand.0 as usize];
            (d.uses_count, d.uses_start)
        };

        match uses_count {
            // Inline arm only while still inline: after migration to the
            // pool (uses_start set), the count can drop back to 0/1 via
            // removals  pushes must then go to the POOL slot, or uses()
            // reads a stale sentinel from the pool.
            0 | 1 if uses_start == u32::MAX => {
                let d = &mut self.def_use[operand.0 as usize];
                d.uses_inline[uses_count as usize] = op;
                d.uses_count = uses_count + 1;
            }
            2 if uses_start == u32::MAX => {
                let inline = self.def_use[operand.0 as usize].uses_inline;
                let new_start = self.uses_pool.len() as u32;
                self.uses_pool.push(inline[0]);
                self.uses_pool.push(inline[1]);
                self.uses_pool.push(op);
                self.uses_pool.push(OperationIdx(u32::MAX));
                let d = &mut self.def_use[operand.0 as usize];
                d.uses_start = new_start;
                d.uses_cap = 4;
                d.uses_count = 3;
            }
            _ => {
                let (count, start, cap) = {
                    let d = &self.def_use[operand.0 as usize];
                    (
                        d.uses_count as usize,
                        d.uses_start as usize,
                        d.uses_cap as usize,
                    )
                };
                if count < cap {
                    self.uses_pool[start + count] = op;
                } else {
                    let old_slab: Vec<_> = (0..count).map(|i| self.uses_pool[start + i]).collect();
                    let new_cap = cap.saturating_mul(2).max(4);
                    let new_start = self.uses_pool.len() as u32;
                    for &item in &old_slab {
                        self.uses_pool.push(item);
                    }
                    self.uses_pool.push(op);
                    for _ in (count + 1)..new_cap {
                        self.uses_pool.push(OperationIdx(u32::MAX));
                    }
                    let d = &mut self.def_use[operand.0 as usize];
                    d.uses_start = new_start;
                    d.uses_cap = new_cap as u16;
                }
                self.def_use[operand.0 as usize].uses_count += 1;
            }
        }
    }

    fn pool_use_remove(&mut self, operand: OperandIndex, op: OperationIdx) {
        let (uses_count, uses_start) = {
            let d = &self.def_use[operand.0 as usize];
            (d.uses_count, d.uses_start)
        };

        if uses_count == 0 {
            return;
        }

        if uses_start == u32::MAX && uses_count <= 2 {
            let d = &mut self.def_use[operand.0 as usize];
            if uses_count == 2 {
                if d.uses_inline[0] == op {
                    d.uses_inline[0] = d.uses_inline[1];
                    d.uses_count = 1;
                } else if d.uses_inline[1] == op {
                    d.uses_count = 1;
                }
            } else {
                if d.uses_inline[0] == op {
                    d.uses_count = 0;
                }
            }
            return;
        }

        let (start, count) = (uses_start as usize, uses_count as usize);
        let mut found = false;
        for i in 0..count {
            if self.uses_pool[start + i] == op {
                for j in i..(count - 1) {
                    self.uses_pool[start + j] = self.uses_pool[start + j + 1];
                }
                self.uses_pool[start + count - 1] = OperationIdx(u32::MAX);
                found = true;
                break;
            }
        }
        if found {
            self.def_use[operand.0 as usize].uses_count -= 1;
        }
    }

    fn pool_assign_push(&mut self, operand: OperandIndex, op: OperationIdx) {
        let (assigns_count, assigns_start) = {
            let d = &self.def_use[operand.0 as usize];
            (d.assigns_count, d.assigns_start)
        };

        match assigns_count {
            // Same pooled-slot rule as pool_use_push: once migrated, count
            // 0 pushes must write the pool, not the inline slot.
            0 if assigns_start == u32::MAX => {
                let d = &mut self.def_use[operand.0 as usize];
                d.assigns_inline[0] = op;
                d.assigns_count = 1;
            }
            1 if assigns_start == u32::MAX => {
                let inline = self.def_use[operand.0 as usize].assigns_inline[0];
                let new_start = self.assigns_pool.len() as u32;
                self.assigns_pool.push(inline);
                self.assigns_pool.push(op);
                let d = &mut self.def_use[operand.0 as usize];
                d.assigns_start = new_start;
                d.assigns_cap = 2;
                d.assigns_count = 2;
            }
            _ => {
                let (count, start, cap) = {
                    let d = &self.def_use[operand.0 as usize];
                    (
                        d.assigns_count as usize,
                        d.assigns_start as usize,
                        d.assigns_cap as usize,
                    )
                };
                if count < cap {
                    self.assigns_pool[start + count] = op;
                } else {
                    let old_slab: Vec<_> =
                        (0..count).map(|i| self.assigns_pool[start + i]).collect();
                    let new_cap = cap.saturating_mul(2).max(2);
                    let new_start = self.assigns_pool.len() as u32;
                    for &item in &old_slab {
                        self.assigns_pool.push(item);
                    }
                    self.assigns_pool.push(op);
                    for _ in (count + 1)..new_cap {
                        self.assigns_pool.push(OperationIdx(u32::MAX));
                    }
                    let d = &mut self.def_use[operand.0 as usize];
                    d.assigns_start = new_start;
                    d.assigns_cap = new_cap as u16;
                }
                self.def_use[operand.0 as usize].assigns_count += 1;
            }
        }
    }

    fn pool_assign_remove(&mut self, operand: OperandIndex, op: OperationIdx) {
        let (assigns_count, assigns_start) = {
            let d = &self.def_use[operand.0 as usize];
            (d.assigns_count, d.assigns_start)
        };

        if assigns_count == 0 {
            return;
        }

        if assigns_start == u32::MAX && assigns_count == 1 {
            let d = &mut self.def_use[operand.0 as usize];
            if d.assigns_inline[0] == op {
                d.assigns_count = 0;
            }
            return;
        }

        let (start, count) = (assigns_start as usize, assigns_count as usize);
        let mut found = false;
        for i in 0..count {
            if self.assigns_pool[start + i] == op {
                for j in i..(count - 1) {
                    self.assigns_pool[start + j] = self.assigns_pool[start + j + 1];
                }
                self.assigns_pool[start + count - 1] = OperationIdx(u32::MAX);
                found = true;
                break;
            }
        }
        if found {
            self.def_use[operand.0 as usize].assigns_count -= 1;
        }
    }
}

impl Default for OperandArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<OperandIndex> for OperandArena {
    type Output = OperandData;
    #[inline]
    fn index(&self, idx: OperandIndex) -> &Self::Output {
        &self.operands[idx.0 as usize]
    }
}

impl IndexMut<OperandIndex> for OperandArena {
    #[inline]
    fn index_mut(&mut self, idx: OperandIndex) -> &mut Self::Output {
        &mut self.operands[idx.0 as usize]
    }
}

pub fn operand_add_use(arena: &mut OperandArena, operand: OperandIndex, op: OperationIdx) {
    match arena.get(operand).kind {
        OperandKind::LocalVariable => {
            arena.pool_use_push(operand, op);
        }
        OperandKind::Memory => {
            let (base, index) = {
                let data = arena.get(operand);
                (data.mem_base(), data.mem_index())
            };

            if let Some(base) = base {
                arena.pool_use_push(base, op);
            }

            if let Some(index) = index {
                arena.pool_use_push(index, op);
            }
        }
        _ => {}
    }
}

pub fn operand_remove_use(arena: &mut OperandArena, operand: OperandIndex, op: OperationIdx) {
    match arena.get(operand).kind {
        OperandKind::LocalVariable => {
            arena.pool_use_remove(operand, op);
        }
        OperandKind::Memory => {
            let (base, index) = {
                let data = arena.get(operand);
                (data.mem_base(), data.mem_index())
            };

            if let Some(base) = base {
                arena.pool_use_remove(base, op);
            }

            if let Some(index) = index {
                arena.pool_use_remove(index, op);
            }
        }
        _ => {}
    }
}

pub fn operand_add_assignment(arena: &mut OperandArena, operand: OperandIndex, op: OperationIdx) {
    match arena.get(operand).kind {
        OperandKind::LocalVariable => {
            arena.pool_assign_push(operand, op);
        }
        OperandKind::Memory => {
            let (base, index) = {
                let data = arena.get(operand);
                (data.mem_base(), data.mem_index())
            };

            if let Some(base) = base {
                arena.pool_assign_push(base, op);
            }

            if let Some(index) = index {
                arena.pool_assign_push(index, op);
            }
        }
        _ => {}
    }
}

pub fn operand_remove_assignment(
    arena: &mut OperandArena,
    operand: OperandIndex,
    op: OperationIdx,
) {
    match arena.get(operand).kind {
        OperandKind::LocalVariable => {
            arena.pool_assign_remove(operand, op);
        }
        OperandKind::Memory => {
            let (base, index) = {
                let data = arena.get(operand);
                (data.mem_base(), data.mem_index())
            };

            if let Some(base) = base {
                arena.pool_assign_remove(base, op);
            }

            if let Some(index) = index {
                arena.pool_assign_remove(index, op);
            }
        }
        _ => {}
    }
}

#[inline]
fn hash_operand(kind: OperandKind, typ: OperandType, value: u64) -> u32 {
    let golden = 0x9e37_79b9u32;
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    let mut h = golden;
    h = h.wrapping_add(kind as u32).wrapping_mul(golden);
    h = h.wrapping_add(typ as u32).wrapping_mul(golden);
    h = h.wrapping_add(lo).wrapping_mul(golden);
    h = h.wrapping_add(hi).wrapping_mul(golden);
    h ^= h >> 16;
    h
}
