// Create and drop translated functions, then check for leaked executable mappings.
// Reads /proc/self/maps; requires Linux.
//
// Usage: cargo run --example smoke_mem_leak

use std::fs;

use armature::translation::translated_function::TranslatedFunction;

/// Count anonymous executable (r-xp) mmaps  these are our JIT code pages.
fn count_jit_pages() -> usize {
    let maps = fs::read_to_string("/proc/self/maps").expect("/proc/self/maps");
    maps.lines()
        .filter(|line| {
            // Anonymous RX mapping: perm=r-xp, device=00:00, inode=0, no pathname.
            line.contains(" r-xp ")
                && line.contains(" 00:00 0")
                && !line.contains("  /")
                && !line.contains(" [")
        })
        .count()
}

fn main() {
    const N: usize = 200;

    let before = count_jit_pages();
    println!("JIT pages before: {before}");

    // Create and drop many TranslatedFunctions  each mmap's 4 KB.
    // Without the Drop fix, every one leaks. With it, all are freed.
    for _ in 0..N {
        let code = vec![0xCCu8; 16]; // 16 bytes → triggers mmap(4096)
        let func = TranslatedFunction::new(code, 0);
        drop(func);
    }

    let after = count_jit_pages();
    println!("JIT pages after:  {after}");

    if after == before {
        println!("== smoke_mem_leak: OK (no leak, {N} create+drop cycles) ==");
    } else {
        eprintln!(
            "LEAK: {before} pages before, {after} after (delta = {})",
            after.saturating_sub(before)
        );
        std::process::exit(1);
    }
}
