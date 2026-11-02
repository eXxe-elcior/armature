use crate::ir::block::{BlockArena, BlockIndex};
use crate::util::hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone)]
pub struct LoopInfo {
    pub header: BlockIndex,
    pub body: FxHashSet<BlockIndex>,
}

/// Does `a` dominate `b`? Walk b's idom chain to the root.
pub fn dominates(blocks: &BlockArena, a: BlockIndex, b: BlockIndex) -> bool {
    let mut cur = b;
    loop {
        if cur == a {
            return true;
        }
        match blocks.get(cur).idom {
            Some(i) if i != cur => cur = i,
            _ => return false,
        }
    }
}

/// Natural loop body for back-edge `back -> header`: header plus every block
/// that reaches `back` without passing through `header`.
pub fn natural_loop(
    header: BlockIndex,
    back: BlockIndex,
    preds: &FxHashMap<BlockIndex, Vec<BlockIndex>>,
) -> FxHashSet<BlockIndex> {
    let mut body = FxHashSet::default();
    body.insert(header);
    let mut work = vec![back];
    while let Some(n) = work.pop() {
        if body.insert(n) {
            if let Some(ps) = preds.get(&n) {
                for &p in ps {
                    work.push(p);
                }
            }
        }
    }
    body
}

/// Find all natural loops in the CFG. Returns loops in arbitrary order;
/// call `sort_innermost_first` to order by nesting depth.
pub fn find_all_loops(blocks: &BlockArena) -> Vec<LoopInfo> {
    let order = blocks.order();
    if order.is_empty() {
        return Vec::new();
    }

    let mut preds: FxHashMap<BlockIndex, Vec<BlockIndex>> = FxHashMap::default();
    for &b in order {
        let data = blocks.get(b);
        for s in 0..data.successor_count as usize {
            if let Some(succ) = data.successors[s] {
                preds.entry(succ).or_default().push(b);
            }
        }
    }

    let mut loops = Vec::new();
    for &b in order {
        let data = blocks.get(b);
        for s in 0..data.successor_count as usize {
            if let Some(header) = data.successors[s] {
                if dominates(blocks, header, b) {
                    let body = natural_loop(header, b, &preds);
                    loops.push(LoopInfo { header, body });
                }
            }
        }
    }

    loops
}

/// Sort loops by body size ascending (innermost loops have the smallest
/// body and appear first). Stable sort keeps outer loops in insertion order.
pub fn sort_innermost_first(loops: &mut Vec<LoopInfo>) {
    loops.sort_by_key(|l| l.body.len());
}
