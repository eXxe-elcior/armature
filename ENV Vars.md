# Environment variables

Set these before starting the host application or example. Many are read once
and cached for the life of the process.

Flags listed as `1` require that exact value. Presence switches turn on whenever
the variable is set, including an empty value or `0`. Unset flags are off.
Addresses are hexadecimal, with an optional `0x` prefix.

## Compilation and execution

| Variable | Value | Default | Effect |
| --- | --- | --- | --- |
| `ARMATURE_HIGHCQ` | `1` | Off | Use HighCq (SSA and linear scan) for new translations requested through dispatch. Existing cached functions are unchanged. |
| `ARMATURE_NO_REJIT` | `1` | Off | Disable background promotion from LCQ to HighCq. Does not override `ARMATURE_HIGHCQ` or prevent loading cached HighCq code. |
| `ARMATURE_REJIT_THRESHOLD` | Positive decimal integer | `100` | Promote a function once its counter reaches this value. Entries and loop back-edges increment the counter. Invalid or nonpositive values use the default. |
| `ARMATURE_REJIT_THREADS` | Positive decimal integer | Host-dependent, 1–4 | Number of background compilation workers, capped at 8. Default: `clamp((available_parallelism - 6) / 3, 1, 4)`, or 1 if host parallelism is unavailable. Invalid values and zero use the default. |
| `ARMATURE_NO_OSR` | `1` | Off | Disable on-stack replacement: a long-running LCQ loop will not exit through dispatch to pick up its compiled HighCq replacement. Overrides `ARMATURE_OSR_THRESHOLD`. |
| `ARMATURE_OSR_THRESHOLD` | Nonnegative decimal integer | `200` | Counter threshold for an LCQ loop to return through dispatch. `0` disables OSR; invalid or negative values use the default. |
| `ARMATURE_MANAGED_DISPATCH` | `1` | Off | Use the Rust dispatch loop instead of the native dispatch loop. Required for block and register tracing. |
| `ARMATURE_REG_SYNC` | `1` | Off | Keep the synchronization counter in a function-local register across loop back-edges instead of accessing context memory at each check. |
| `ARMATURE_PTC` | Cache file path | Disabled | Load and save the persistent translation cache. Captures relocatable HighCq functions during background recompilation; saves periodically and at teardown. Empty values disable it. |

Sources: [translator.rs](src/translation/translator.rs),
[context.rs](src/instructions/context.rs).

## Tracing

| Variable | Value | Default | Effect |
| --- | --- | --- | --- |
| `ARMATURE_TRACE` | `1`, `blocks`, `regs`, `jit`, `all`, or a comma-separated combination | Disabled | `1` and `blocks` log guest block PCs. `regs` also logs changed general registers. `jit` logs compilation and eviction. `all` enables every channel. Unknown tokens are ignored. |
| `ARMATURE_TRACE_FILE` | Output file path | Standard error | Write trace output to this file, replacing its contents. Falls back to standard error if the file cannot be opened. |
| `ARMATURE_TRACE_ADDR` | Guest address in hex | No filter | Restrict block traces to the inclusive range `[address - 0x80, address + 0x800]` and print all X0–X30 registers for each matching block. Does not filter JIT events. |
| `ARMATURE_TRACE_CAP` | Nonnegative decimal integer | `4000` | Maximum number of address-filtered block visits to log across the process. Only applies with `ARMATURE_TRACE_ADDR`; `0` suppresses those block traces. Invalid values use the default. |

Block and register traces need `ARMATURE_MANAGED_DISPATCH=1`. JIT event tracing
works with either dispatch loop. Setting an output file or address filter does
not enable tracing by itself; select a channel with `ARMATURE_TRACE`.

Source: [trace.rs](src/translation/trace.rs).

## Compiler and memory diagnostics

| Variable | Value | Default | Effect |
| --- | --- | --- | --- |
| `ARMATURE_DUMP_IR` | Presence switch | Off | Print IR before optimization, before SSA deconstruction, and after register allocation to standard error. |
| `ARMATURE_DUMP_FN` | Guest function entry address in hex | Disabled | Print final IR after register allocation for this function only. |
| `ARMATURE_DUMP_GUEST` | Guest address in hex | Disabled | Print decoded guest instructions in the inclusive range `[address - 0x80, address + 0x800]`. |
| `ARMATURE_REGALLOC_DUMP` | Presence switch | Off | Print linear-scan live intervals, splits, register assignments, and spills to standard error. Applies to compilations using linear scan. |
| `ARMATURE_LSRA_TIME` | Presence switch | Off | Print linear-scan phase timings and interval counts to standard error. |
| `ARMATURE_DUMP_JIT` | `1` | Off | Write compiled host code to `/tmp/armature-jit/<host-address>.bin`, with an `index.txt` mapping host addresses to guest functions. |
| `ARMATURE_PERF_MAP` | `1` | Off | Write JIT symbols to `/tmp/perf-<pid>.map` for Linux `perf`, using names such as `guest_<address>_<tier>`. |
| `ARMATURE_MEM_DEBUG` | `1` | Off | Route host-mapped and host-tracked address translation through a diagnostic helper. Logs the first 64 accesses and accesses with a zero page-table entry to standard error. |
| `ARMATURE_SVC_STATS` | `1` | Off | Collect Horizon SVC counts, including sleep/yield and per-thread statistics. Print a histogram every 1,048,576 SVC calls. |

Sources: [code_generator.rs](src/codegen/x86_64/code_generator.rs),
[linear_scan.rs](src/codegen/regalloc/linear_scan.rs),
[block_decoder.rs](src/translation/block_decoder.rs),
[translator.rs](src/translation/translator.rs),
[memory_helpers.rs](src/instructions/memory_helpers.rs),
[native_interface.rs](src/instructions/native_interface.rs).

## Examples and C# tools

| Variable | Value | Default | Effect |
| --- | --- | --- | --- |
| `ARMATURE_DUMP` | Presence switch | Off | In `diff_harness`, print each vector's compiled host bytes to standard error. `ARMATURE_HIGHCQ=1` also selects HighCq for this dump. |
| `DUMP_FP` | `1` | Off | Select the floating-point loop instead of the integer loop in `examples/dump_asm.rs` and `tools/asmdump`. |
| `ASMDUMP_PRESSURE_K` | Decimal integer | Unset | In the C# `asmdump` tool, generate a register-pressure block with this many rounds of additions. Takes precedence over `DUMP_FP`. Invalid integers fail parsing. |
| `RyujinxRoot` | Ryujinx checkout directory | `../ryujinx` relative to this repository | Build-time MSBuild property used by both C# tools. Can be supplied as an environment variable or with `-p:RyujinxRoot=/path/to/ryujinx`. The checkout must contain `src/ARMeilleure`, `src/Ryujinx.Memory`, and `src/Ryujinx.Cpu`. |

Sources: [diff_harness.rs](examples/diff_harness.rs),
[dump_asm.rs](examples/dump_asm.rs),
[asmdump](tools/asmdump/Program.cs),
[Oracle.csproj](tools/oracle/Oracle.csproj),
[asmdump.csproj](tools/asmdump/asmdump.csproj).

## Examples

```bash
# Trace guest blocks and register changes.
ARMATURE_MANAGED_DISPATCH=1 ARMATURE_TRACE=regs \
  ARMATURE_TRACE_FILE=trace.log cargo run --example smoke_flow

# Run differential vectors through HighCq and print allocation timings.
ARMATURE_HIGHCQ=1 ARMATURE_LSRA_TIME=1 \
  cargo run --example diff_harness -- vectors.golden

# Disassemble a floating-point loop.
DUMP_FP=1 cargo run --example dump_asm -- hcq soft

# Build the C# oracle against an existing Ryujinx checkout.
dotnet build tools/oracle/Oracle.csproj -p:RyujinxRoot=/path/to/ryujinx
```
