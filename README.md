# Armature

Armature is an AArch64 to x86-64 dynamic recompiler written in Rust. It translates
guest instructions into native code and runs them in the host process. Much of
the implementation is ported from Ryujinx's ARMeilleure, including the decoder,
instruction emitters, and compiler passes.

It's a library for programs that supply their own guest memory and execution
environment. It has no external Rust crate dependencies.

The current execution target is **x86-64 Linux**. The backend includes System V
and Windows calling conventions, but executable memory allocation outside Linux
is still a stub.

## Build and try it

Use a recent Rust toolchain with support for the 2024 edition.

```bash
cargo build --release
cargo run --example execute
```

The example runs `MOVZ X0, #42; RET`, prints the generated host bytes, and checks
that X0 contains 42. Its last two lines should be:

```text
X0 = 42
execute: passed
```

The release build produces a Rust library and `target/release/libarmature.so`
for C ABI callers.

## How it fits together

Guest instructions are decoded into blocks, emitted into an intermediate
representation, and compiled to x86-64. The initial LCQ tier uses a hybrid
register allocator. Hot functions are recompiled in the background using SSA,
optimization passes, and linear-scan register allocation in the HighCq tier.
A persistent translation cache can save compiled HighCq functions between runs.

For embedding, start with the [MemoryManager trait](src/memory/manager.rs) and
[Translator](src/translation/translator.rs). The host supplies memory access,
creates an execution context for each guest thread, and handles exceptions and
interrupts. [execute.rs](examples/execute.rs) is the smallest example;
[smoke_flow.rs](examples/smoke_flow.rs) also sets up guest memory and a page table.

The [C ABI](src/ffi.rs) exposes translator and context lifecycle, memory
callbacks, and execution. [ryubing-shim](ryubing-shim) contains the C# integration
code; it depends on the surrounding Ryujinx interfaces and is not a standalone
C# project.

## Tests and examples

```bash
cargo test --all-targets
cargo test --release
```

These run the execution-context tests and build the example test targets. Run
the examples themselves to exercise generated code:

```bash
cargo run --release --example smoke_flow
cargo run --release --example stress_regalloc
cargo run --release --example ptc_smoke
```

| Example | What it checks or shows |
| --- | --- |
| `execute` | Compile and execute a small AArch64 function. |
| `smoke_flow` | Loops, calls, returns, indirect branches, and recompilation. |
| `smoke_adrp` | ADR, ADRP, and immediate arithmetic. |
| `guest_blob` | Recursive Fibonacci and indexed memory access in a compiled guest program. |
| `stress_regalloc` | Register pressure, spills, and block-edge copies in both tiers. |
| `ptc_smoke` | Relocation, serialization, and execution of cached code at new host addresses. |
| `smoke_barrier` | Barrier lowering and execution. |
| `smoke_mem_leak` | Executable mappings are released when translated functions are dropped. |
| `dump_asm` | Disassemble generated x86-64 with `objdump`. |

### Compare against ARMeilleure

The differential harness generates deterministic instruction inputs, runs them
through the C# ARMeilleure implementation, then compares Armature's register
state and memory hash with those results. This needs .NET 10 and a compatible
Ryujinx source checkout in addition to Rust.

```bash
cargo run --release --example gen_vectors > vectors.spec

RyujinxRoot=/path/to/ryujinx \
  dotnet run --project tools/oracle -- vectors.spec vectors.golden

cargo run --release --example diff_harness -- vectors.golden
ARMATURE_HIGHCQ=1 cargo run --release --example diff_harness -- vectors.golden
```

`tools/asmdump` can also compile and disassemble an ARMeilleure guest block for
comparison with `dump_asm`. Both disassembly tools need `objdump` on `PATH`.

## Configuration

[ENV Vars.md](ENV%20Vars.md) lists the runtime switches, compiler diagnostics,
trace settings, and tool options, including their defaults. Set them before
starting the host; many are cached on first use.

## What's unfinished

Some instruction paths still panic with `todo!()`. These include software
fallbacks for AES and SHA256, several floating-point conversions, and EVEX
encoding. Supported instructions can therefore depend on the host CPU's
extensions. A passing run of the examples or differential vectors does not
cover every instruction or guest program.

Reports with a small guest program, the failing opcode, and the host CPU details
are useful when tracking down a bad translation. If it only fails in one tier,
include that too.

## License

[MIT](LICENSE).
