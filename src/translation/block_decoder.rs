// Port of Decoders/Decoder.cs + Decoders/Block.cs +
// Decoders/Optimizations/TailCallRemover.cs (A64 only).
//
// Divergences:
//   - A32/Thumb excluded: ExecutionMode parameter dropped, the
//     IsAarch32* family and the Thumb IT-block tracking in FillBlock
//     vanish, OpCodeSizeInBytes is always 4, IsCall loses Blx and
//     IsTrap loses InstName.Trap (A32-only name).
//   - C# Block object references become arena indices: `decode` returns
//     DecodedBlocks { arena, order } where `order` is the C# returned
//     Block[] (address-sorted, post-TailCallRemover) and next/branch are
//     Option<usize> into `arena`. Blocks dropped by TailCallRemover stay
//     in the arena so no index dangles.
//   - C# OpCodeTable.GetInstA64 falls back to InstDescriptor(Und,
//     InstEmit.Und) for unmatched encodings; decode_one returns
//     Err(DecodeError) instead, mapped here to a DecodedInst carrying
//     InstName::Und + emit_und + OpCodeKind::Undefined.

use std::collections::{HashMap, VecDeque};

use crate::decoder::decode_one::decode_one;
use crate::decoder::instruction_name::InstructionName;
use crate::decoder::opcode_kind::OpCodeKind;
use crate::decoder::utilities::{DecodedInstruction, InstructionInfo, RegisterSize};
use crate::instructions::exceptions::emit_und;
use crate::memory::MemoryManager;

// We define a limit on the number of instructions that a function may
// have, this prevents functions being potentially too large, which would
// take too long to compile and use too much memory.
const MAX_INSTS_PER_FUNCTION: i32 = 2500;

// For lower code quality translation, we set a lower limit since we're
// blocking execution.
const MAX_INSTS_PER_FUNCTION_LOW_CQ: i32 = 500;

/// C# Decoders/DecoderMode.cs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderMode {
    MultipleBlocks,
    SingleBlock,
    SingleInstruction,
}

/// C# Decoders/Block.cs.
pub struct Block {
    pub address: u64,
    pub end_address: u64,

    pub next: Option<usize>,
    pub branch: Option<usize>,

    pub exit: bool,

    pub op_codes: Vec<DecodedInstruction>,
}

impl Block {
    fn new(address: u64) -> Self {
        Self {
            address,
            end_address: 0,
            next: None,
            branch: None,
            exit: false,
            op_codes: Vec::new(),
        }
    }

    pub fn get_last_op(&self) -> Option<&DecodedInstruction> {
        self.op_codes.last()
    }

    // C# private Block.BinarySearch(opCodes, address).
    fn binary_search_ops(&self, address: u64) -> usize {
        let mut left: isize = 0;
        let mut middle: isize = 0;
        let mut right: isize = self.op_codes.len() as isize - 1;

        while left <= right {
            let size = right - left;

            middle = left + (size >> 1);

            let op = &self.op_codes[middle as usize];

            match address.cmp(&op.address) {
                std::cmp::Ordering::Equal => break,
                std::cmp::Ordering::Less => right = middle - 1,
                std::cmp::Ordering::Greater => left = middle + 1,
            }
        }

        middle as usize
    }
}

/// Result of `decode`  see the divergence note at the top.
pub struct DecodedBlocks {
    pub arena: Vec<Block>,
    /// The C# returned `Block[]`: address-sorted, post-TailCallRemover.
    pub order: Vec<usize>,
}

// C# Block.Split(rightBlock)  `left`/`right` are arena indices.
fn split(arena: &mut [Block], left: usize, right: usize) {
    let mut split_index = arena[left].binary_search_ops(arena[right].address);

    if arena[left].op_codes[split_index].address < arena[right].address {
        split_index += 1;
    }

    let split_count = arena[left].op_codes.len() - split_index;

    if split_count == 0 {
        panic!("Can't split at right block address.");
    }

    let tail: Vec<DecodedInstruction> = arena[left].op_codes.drain(split_index..).collect();
    let (l_end, l_next, l_branch) = (
        arena[left].end_address,
        arena[left].next,
        arena[left].branch,
    );

    {
        let r = &mut arena[right];
        r.end_address = l_end;
        r.next = l_next;
        r.branch = l_branch;
        r.op_codes.extend(tail);
    }

    let r_address = arena[right].address;
    let l = &mut arena[left];
    l.end_address = r_address;
    l.next = Some(right);
    l.branch = None;
}

struct DecodeState<'m> {
    memory: &'m dyn MemoryManager,
    d_mode: DecoderMode,
    instruction_limit: i32,
    ops_count: i32,
    arena: Vec<Block>,
    visited: HashMap<u64, usize>,
    work_queue: VecDeque<usize>,
}

impl DecodeState<'_> {
    // C# local function GetBlock(blkAddress).
    fn get_block(&mut self, blk_address: u64) -> usize {
        if let Some(&idx) = self.visited.get(&blk_address) {
            return idx;
        }

        let mut block = Block::new(blk_address);

        if (self.d_mode != DecoderMode::MultipleBlocks && !self.visited.is_empty())
            || self.ops_count > self.instruction_limit
            || (!self.visited.is_empty() && !self.memory.is_mapped(blk_address))
        {
            block.exit = true;
            block.end_address = blk_address;
        }

        let idx = self.arena.len();
        self.arena.push(block);
        self.work_queue.push_back(idx);
        self.visited.insert(blk_address, idx);

        idx
    }
}

/// C# Decoder.Decode (ExecutionMode dropped  A64 only).
pub fn decode(
    memory: &dyn MemoryManager,
    address: u64,
    high_cq: bool,
    d_mode: DecoderMode,
) -> DecodedBlocks {
    let mut blocks: Vec<usize> = Vec::new();

    let instruction_limit = if high_cq {
        MAX_INSTS_PER_FUNCTION
    } else {
        MAX_INSTS_PER_FUNCTION_LOW_CQ
    };

    let mut st = DecodeState {
        memory,
        d_mode,
        instruction_limit,
        ops_count: 0,
        arena: Vec::new(),
        visited: HashMap::new(),
        work_queue: VecDeque::new(),
    };

    st.get_block(address);

    while let Some(curr) = st.work_queue.pop_front() {
        // Check if the current block is inside another block.
        let (found, n_blk_index) = binary_search(&st.arena, &blocks, st.arena[curr].address);
        if found {
            let n_block = blocks[n_blk_index];

            if st.arena[n_block].address == st.arena[curr].address {
                panic!("Found duplicate block address on the list.");
            }

            st.arena[curr].exit = false;

            split(&mut st.arena, n_block, curr);

            blocks.insert(n_blk_index + 1, curr);

            continue;
        }

        if !st.arena[curr].exit {
            // If we have a block after the current one, set the limit address.
            let mut limit_address = u64::MAX;

            if n_blk_index != blocks.len() {
                let n_block = &st.arena[blocks[n_blk_index]];

                let next_index = n_blk_index + 1;

                if n_block.address < st.arena[curr].address && next_index < blocks.len() {
                    limit_address = st.arena[blocks[next_index]].address;
                } else if n_block.address > st.arena[curr].address {
                    limit_address = st.arena[blocks[n_blk_index]].address;
                }
            }

            if d_mode == DecoderMode::SingleInstruction {
                // Only read at most one instruction.
                limit_address = st.arena[curr].address + 1;
            }

            fill_block(st.memory, &mut st.arena[curr], limit_address);

            st.ops_count += st.arena[curr].op_codes.len() as i32;

            if !st.arena[curr].op_codes.is_empty() {
                // Set child blocks. "branch" is the block the branch
                // instruction points to (when taken), "next" is the block
                // at the next address, executed when the branch is not
                // taken. For unconditional branches (except BL/BLR that
                // are sub calls) or end of executable, next is None.
                let last_op = *st.arena[curr].op_codes.last().unwrap();

                let is_call = is_call(&last_op);

                if let Some(immediate) = bimm_immediate(&last_op) {
                    if !is_call {
                        let target = st.get_block(immediate as u64);
                        st.arena[curr].branch = Some(target);
                    }
                }

                if is_call || !(is_unconditional_branch(&last_op) || is_trap(&last_op)) {
                    let end_address = st.arena[curr].end_address;
                    let target = st.get_block(end_address);
                    st.arena[curr].next = Some(target);
                }
            }
        }

        // Insert the new block on the list (sorted by address).
        if !blocks.is_empty() {
            let n_block = &st.arena[blocks[n_blk_index]];

            let offset = if n_block.address < st.arena[curr].address {
                1
            } else {
                0
            };
            blocks.insert(n_blk_index + offset, curr);
        } else {
            blocks.push(curr);
        }
    }

    if blocks.len() == 1 && st.arena[blocks[0]].op_codes.is_empty() {
        debug_assert!(st.arena[blocks[0]].exit);
        debug_assert!(st.arena[blocks[0]].address == st.arena[blocks[0]].end_address);

        panic!("Decoded a single empty exit block. Entry point = 0x{address:X}.");
    }

    let order = if d_mode == DecoderMode::MultipleBlocks {
        tail_call_remover_run_pass(address, &mut st.arena, blocks)
    } else {
        blocks
    };

    DecodedBlocks {
        arena: st.arena,
        order,
    }
}

/// C# Decoder.BinarySearch(blocks, address, out index)  searches the
/// address-sorted `blocks` index list over `arena`.
pub fn binary_search(arena: &[Block], blocks: &[usize], address: u64) -> (bool, usize) {
    let mut index: usize = 0;

    let mut left: isize = 0;
    let mut right: isize = blocks.len() as isize - 1;

    while left <= right {
        let size = right - left;

        let middle = left + (size >> 1);

        let block = &arena[blocks[middle as usize]];

        index = middle as usize;

        if address >= block.address && address < block.end_address {
            return (true, index);
        }

        if address < block.address {
            right = middle - 1;
        } else {
            left = middle + 1;
        }
    }

    (false, index)
}

// C# Decoder.FillBlock (IT-block tracking dropped with Thumb).
/// ARMATURE_DUMP_GUEST=<hex addr> → disassemble guest ops within ±0x80 of it.
fn dump_guest_match(address: u64) -> bool {
    static TARGET: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let t = *TARGET.get_or_init(|| {
        std::env::var("ARMATURE_DUMP_GUEST")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
    });
    match t {
        Some(t) => address >= t.saturating_sub(0x80) && address <= t + 0x800,
        None => false,
    }
}

fn fill_block(memory: &dyn MemoryManager, block: &mut Block, limit_address: u64) {
    let mut address = block.address;

    loop {
        if address >= limit_address {
            break;
        }

        let op_code = decode_op_code(memory, address);

        if dump_guest_match(address) {
            eprintln!(
                "[guest] {address:#x}: {:08x}  {:?}  {:?}",
                op_code.raw, op_code.info.name, op_code.kind
            );
        }

        block.op_codes.push(op_code);

        address += 4; // A64: OpCodeSizeInBytes is always 4.

        if is_branch(&op_code) || is_exception(&op_code) {
            break;
        }
    }

    block.end_address = address;
}

// C# IsBranch: OpCodeBImm (Al/Cond/Cmp/Test) or OpCodeBReg.
fn is_branch(op: &DecodedInstruction) -> bool {
    matches!(
        op.kind,
        OpCodeKind::BImm { .. }
            | OpCodeKind::BImmCond { .. }
            | OpCodeKind::BImmCmp { .. }
            | OpCodeKind::BImmTest { .. }
            | OpCodeKind::BReg { .. }
    )
}

// C# IsUnconditionalBranch: OpCodeBImmAl or OpCodeBReg.
fn is_unconditional_branch(op: &DecodedInstruction) -> bool {
    matches!(op.kind, OpCodeKind::BImm { .. } | OpCodeKind::BReg { .. })
}

// C# `lastOp is IOpCodeBImm op` → op.Immediate.
fn bimm_immediate(op: &DecodedInstruction) -> Option<i64> {
    match op.kind {
        OpCodeKind::BImm { immediate, .. }
        | OpCodeKind::BImmCond { immediate, .. }
        | OpCodeKind::BImmCmp { immediate, .. }
        | OpCodeKind::BImmTest { immediate, .. } => Some(immediate),
        _ => None,
    }
}

// C# IsCall: Bl, Blr (Blx is A32-only).
fn is_call(op: &DecodedInstruction) -> bool {
    matches!(op.info.name, InstructionName::Bl | InstructionName::Blr)
}

fn is_exception(op: &DecodedInstruction) -> bool {
    is_trap(op) || op.info.name == InstructionName::Svc
}

// C# IsTrap: Brk, Trap, Und (InstName.Trap is A32-only).
fn is_trap(op: &DecodedInstruction) -> bool {
    matches!(op.info.name, InstructionName::Brk | InstructionName::Und)
}

/// C# Decoder.DecodeOpCode (A64 path only).
pub fn decode_op_code(memory: &dyn MemoryManager, address: u64) -> DecodedInstruction {
    let raw = memory.read_u32(address);

    match decode_one(raw, address) {
        Ok(inst) => inst,
        // C# OpCodeTable fallback: InstDescriptor(InstName.Und, InstEmit.Und)
        // wrapped in a base OpCode (no decoded fields).
        // This DecodedInstruction is consumed by emit_und, which only reads
        // address and raw  kind is arbitrary (handlers dispatch on name).
        Err(_) => DecodedInstruction {
            address,
            raw,
            info: InstructionInfo {
                name: InstructionName::Und,
                emit: emit_und,
            },
            reg_size: RegisterSize::Int64,
            kind: OpCodeKind::Undefined,
        },
    }
}

// C# Decoders/Optimizations/TailCallRemover.RunPass. Takes and returns
// the address-sorted index list; excluded blocks stay in the arena.
fn tail_call_remover_run_pass(
    entry_address: u64,
    arena: &mut [Block],
    blocks: Vec<usize>,
) -> Vec<usize> {
    // Detect tail calls:
    // - Assume this function spans the space covered by contiguous code
    //   blocks surrounding the entry address.
    // - A jump to an area outside this contiguous region will be treated
    //   as an exit block.
    // - Include a small allowance for jumps outside the contiguous range.

    let (found, entry_block_id) = binary_search(arena, &blocks, entry_address);
    if !found {
        panic!("Function entry point is not contained in a block.");
    }

    const ALLOWANCE: u64 = 4;

    let mut start_block_index = entry_block_id;
    let mut end_block_index = entry_block_id;

    // Search forwards.
    for i in entry_block_id + 1..blocks.len() {
        let end_block = &arena[blocks[end_block_index]];
        let block = &arena[blocks[i]];

        // C# ulong arithmetic is unchecked  wrapping matches it.
        if end_block.end_address < block.address.wrapping_sub(ALLOWANCE) {
            break; // End of contiguous function.
        }

        end_block_index = i;
    }

    // Search backwards.
    for i in (0..entry_block_id).rev() {
        let start_block = &arena[blocks[start_block_index]];
        let block = &arena[blocks[i]];

        // C# ulong arithmetic is unchecked  wrapping matches it.
        if start_block.address > block.end_address.wrapping_add(ALLOWANCE) {
            break; // End of contiguous function.
        }

        start_block_index = i;
    }

    if start_block_index == 0 && end_block_index == blocks.len() - 1 {
        return blocks; // Nothing to do here.
    }

    let start_address = arena[blocks[start_block_index]].address;
    let end_end_address = arena[blocks[end_block_index]].end_address;

    // Mark branches whose target is outside of the contiguous region as
    // an exit block.
    for i in start_block_index..=end_block_index {
        if let Some(branch) = arena[blocks[i]].branch {
            if arena[branch].address > end_end_address || arena[branch].end_address < start_address
            {
                arena[branch].exit = true;
            }
        }
    }

    // Finally, rebuild decoded block list, ignoring blocks outside the
    // contiguous range.
    let mut new_blocks = Vec::with_capacity(blocks.len());

    for (i, &blk) in blocks.iter().enumerate() {
        if arena[blk].exit || (i >= start_block_index && i <= end_block_index) {
            new_blocks.push(blk);
        }
    }

    new_blocks
}
