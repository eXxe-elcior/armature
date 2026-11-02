// Compile an ARMeilleure HighCq guest block and disassemble with objdump.
// Compare with: cargo run --example dump_asm -- hcq soft
//
// Usage: dotnet run --project tools/asmdump
// Set RyujinxRoot to an upstream checkout if it is not at ../ryujinx.
// Requires .NET 10 and objdump (binutils).

using ARMeilleure;
using ARMeilleure.Common;
using ARMeilleure.Memory;
using ARMeilleure.Translation;
using Ryujinx.Cpu.Jit;
using Ryujinx.Memory;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

class Program
{
    static ulong Size = MemoryBlock.GetPageSize();
    static ulong CodeBaseAddress;
    static ulong DataBaseAddress;

    static MemoryBlock _ram;
    static MemoryManager _memory;
    static Translator _translator;

    // Same representative loop as examples/dump_asm.rs ACCUM_LOOP:
    //   loop: ldr x3,[x1]; add x1,x1,#8; add x0,x0,x3; subs x2,x2,#1; b.ne loop; ret
    static readonly uint[] AccumLoop =
    {
        0xF9400023, 0x91002021, 0x8B030000, 0xF1000442, 0x54FFFF81, 0xD65F03C0,
    };

    // Matches examples/dump_asm.rs gen_pressure(k): define x1..x20 (all live),
    // then k rounds of `add x0,x0,xr` re-reading each → 21 simultaneously-live
    // values, forces spills/splits. Stresses LSRA spill QUALITY (not just speed).
    static uint[] GenPressure(int k)
    {
        var v = new System.Collections.Generic.List<uint>();
        for (uint r = 1; r <= 20; r++) v.Add(0x91000000u | (r << 10) | r); // add xr,x0,#r
        for (int i = 0; i < k; i++)
            for (uint r = 1; r <= 20; r++) v.Add(0x8B000000u | (r << 16)); // add x0,x0,xr
        v.Add(0xD65F03C0u); // ret
        return v.ToArray();
    }

    // Pure-register vec4 FP accumulate (matches examples/dump_asm.rs FP_LOOP).
    static readonly uint[] FpLoop =
    {
        0x4E22DC20, 0x4E20D463, 0xF1000442, 0x54FFFFA1, 0xD65F03C0,
    };

    static uint[] Block => Environment.GetEnvironmentVariable("ASMDUMP_PRESSURE_K") is string s
        ? GenPressure(int.Parse(s))
        : Environment.GetEnvironmentVariable("DUMP_FP") == "1"
            ? FpLoop
            : AccumLoop;

    static void Setup()
    {
        int pageBits = (int)ulong.Log2(Size);

        CodeBaseAddress = Size;
        DataBaseAddress = CodeBaseAddress + Size;
        if (Size > 0x1000)
        {
            DataBaseAddress = 0;
            CodeBaseAddress = Size;
        }

        _ram = new MemoryBlock(Size * 2);
        _memory = new MemoryManager(_ram, 1ul << (pageBits + 4));
        _memory.IncrementReferenceCount();
        _memory.Map(CodeBaseAddress, 0, Size, MemoryMapFlags.Private);
        _memory.Map(DataBaseAddress, Size, Size, MemoryMapFlags.Private);

        _translator = new Translator(
            new JitMemoryAllocator(),
            _memory,
            AddressTable<ulong>.CreateForArm(true, _memory.Type));

        Optimizations.AllowLcqInFunctionTable = false;
        Optimizations.UseUnmanagedDispatchLoop = false;
    }

    static void Main()
    {
        Setup();

        uint[] block = Block;
        for (uint i = 0; i < block.Length; i++)
        {
            _memory.Write(CodeBaseAddress + i * 4, block[i]);
        }

        // internal TranslatedFunction Translate(ulong address, ExecutionMode mode,
        //     bool highCq, bool singleStep = false, bool pptcTranslation = false)
        MethodInfo translate = typeof(Translator).GetMethod(
            "Translate", BindingFlags.NonPublic | BindingFlags.Instance);
        if (translate == null)
        {
            Console.Error.WriteLine("could not reflect Translator.Translate");
            Environment.Exit(1);
        }

        // ExecutionMode is internal  build the Aarch64 value by reflection.
        Type execMode = typeof(Translator).Assembly.GetType("ARMeilleure.State.ExecutionMode");
        object aarch64 = Enum.Parse(execMode, "Aarch64");

        // Dump both tiers so LowCq (HybridAllocator) can be diffed too.
        Dump(translate, aarch64, /*highCq*/ false, "lcq", "/tmp/armeil-lcq.bin");
        Dump(translate, aarch64, /*highCq*/ true, "hcq", "/tmp/armeil-hcq.bin");
    }

    static void Dump(MethodInfo translate, object aarch64, bool highCq, string tag, string path)
    {
        object func = translate.Invoke(
            _translator,
            new object[] { CodeBaseAddress, aarch64, highCq, false, false });

        nint funcPtr = (nint)func.GetType()
            .GetProperty("FuncPointer", BindingFlags.Public | BindingFlags.Instance)
            .GetValue(func);

        // No host code length is exposed; read a generous window. objdump will
        // show the function then trailing cache bytes  the `ret` + zero padding
        // marks the real end.
        const int Window = 16384;
        byte[] code = new byte[Window];
        Marshal.Copy(funcPtr, code, 0, Window);

        File.WriteAllBytes(path, code);
        Console.WriteLine($"ARMeilleure {tag} @ guest {CodeBaseAddress:x} -> {path} (FuncPointer {funcPtr:x})");
        Console.WriteLine("memory: SoftwarePageTable | guest: integer load-accumulate loop");
        Console.WriteLine($"==================== armeilleure {tag} ====================");

        var psi = new ProcessStartInfo("objdump")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (string a in new[]
        {
            "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
            "--no-show-raw-insn", path,
        })
        {
            psi.ArgumentList.Add(a);
        }

        using Process p = Process.Start(psi);
        Console.Write(p.StandardOutput.ReadToEnd());
        Console.Error.Write(p.StandardError.ReadToEnd());
        p.WaitForExit();
    }
}
