// Execute instruction vectors through ARMeilleure and record the resulting state.
// Uses the setup from Ryujinx.Tests.Cpu, without NUnit or Unicorn.
//
// Usage: dotnet run --project tools/oracle -- vectors.spec vectors.golden
// Set RyujinxRoot to an upstream checkout if it is not at ../ryujinx.
// Requires .NET 10.
//
// spec: opcode x0 x1 x2 x3 sp nzcv dseed v0 v1 v2 v3 (hex)
// golden: spec fields | x0..x30 sp nzcv dhash v0..v31 (hex)
// dseed seeds the 4 KiB data page using xorshift64*.
// dhash is the FNV-1a hash of that page after execution.

using ARMeilleure;
using ARMeilleure.Common;
using ARMeilleure.Memory;
using ARMeilleure.State;
using ARMeilleure.Translation;
using Ryujinx.Cpu;
using Ryujinx.Cpu.Jit;
using Ryujinx.Memory;
using System;
using System.Globalization;
using System.IO;
using System.Text;

class Program
{
    static ulong Size = MemoryBlock.GetPageSize();
    static ulong CodeBaseAddress;
    static ulong DataBaseAddress;

    static MemoryBlock _ram;
    static MemoryManager _memory;
    static ExecutionContext _context;
    static Translator _translator;

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

        _context = new ExecutionContext(new JitMemoryAllocator(), new TickSource(19200000));

        _translator = new Translator(
            new JitMemoryAllocator(),
            _memory,
            AddressTable<ulong>.CreateForArm(true, _memory.Type));

        Optimizations.AllowLcqInFunctionTable = false;
        Optimizations.UseUnmanagedDispatchLoop = false;
    }

    static void Teardown()
    {
        _memory.DecrementReferenceCount();
        _context.Dispose();
        _ram.Dispose();
    }

    static void Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("usage: oracle <vectors.spec> <vectors.golden>");
            Environment.Exit(1);
        }

        string[] lines = File.ReadAllLines(args[0]);
        using StreamWriter outp = new(args[1]);

        int done = 0;

        foreach (string raw in lines)
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                outp.WriteLine(line);
                continue;
            }

            string[] f = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            uint opcode = uint.Parse(f[0], NumberStyles.HexNumber);
            ulong x0 = ulong.Parse(f[1], NumberStyles.HexNumber);
            ulong x1 = ulong.Parse(f[2], NumberStyles.HexNumber);
            ulong x2 = ulong.Parse(f[3], NumberStyles.HexNumber);
            ulong x3 = ulong.Parse(f[4], NumberStyles.HexNumber);
            ulong sp = ulong.Parse(f[5], NumberStyles.HexNumber);
            uint nzcv = uint.Parse(f[6], NumberStyles.HexNumber);
            ulong dseed = f.Length > 7 ? ulong.Parse(f[7], NumberStyles.HexNumber) : 0;

            // v0-v3 inputs: 32 hex chars each (hi64 then lo64 of the u128).
            V128[] vin = new V128[4];
            for (int i = 0; i < 4 && f.Length > 8 + i; i++)
            {
                string h = f[8 + i].PadLeft(32, '0');
                ulong hi = ulong.Parse(h.Substring(0, 16), NumberStyles.HexNumber);
                ulong lo = ulong.Parse(h.Substring(16), NumberStyles.HexNumber);
                vin[i] = new V128(lo, hi);
            }

            Setup();

            _memory.Write(CodeBaseAddress, opcode);
            _memory.Write(CodeBaseAddress + 4, 0xD65F03C0u); // RET

            // Deterministic data-page fill (xorshift64*, same as Rust side).
            ulong s = dseed == 0 ? 0x9E3779B97F4A7C15UL : dseed;
            for (ulong off = 0; off < 0x1000; off += 8)
            {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                _memory.Write(DataBaseAddress + off, s * 0x2545F4914F6CDD1DUL);
            }

            _context.SetX(0, x0);
            _context.SetX(1, x1);
            _context.SetX(2, x2);
            _context.SetX(3, x3);
            _context.SetX(31, sp);

            for (int i = 0; i < 4; i++)
            {
                _context.SetV(i, vin[i]);
            }

            _context.SetPstateFlag(PState.NFlag, (nzcv & (1u << 31)) != 0);
            _context.SetPstateFlag(PState.ZFlag, (nzcv & (1u << 30)) != 0);
            _context.SetPstateFlag(PState.CFlag, (nzcv & (1u << 29)) != 0);
            _context.SetPstateFlag(PState.VFlag, (nzcv & (1u << 28)) != 0);

            _translator.Execute(_context, CodeBaseAddress);

            StringBuilder sb = new(line);
            sb.Append(" |");
            for (int i = 0; i < 31; i++)
            {
                sb.Append(' ').Append(_context.GetX(i).ToString("x"));
            }
            sb.Append(' ').Append(_context.GetX(31).ToString("x"));

            uint outNzcv = 0;
            if (_context.GetPstateFlag(PState.NFlag)) outNzcv |= 1u << 31;
            if (_context.GetPstateFlag(PState.ZFlag)) outNzcv |= 1u << 30;
            if (_context.GetPstateFlag(PState.CFlag)) outNzcv |= 1u << 29;
            if (_context.GetPstateFlag(PState.VFlag)) outNzcv |= 1u << 28;
            sb.Append(' ').Append(outNzcv.ToString("x"));

            // FNV-1a 64 over the data page (post-execution).
            ulong hash = 0xCBF29CE484222325UL;
            for (ulong off = 0; off < 0x1000; off++)
            {
                hash ^= _memory.Read<byte>(DataBaseAddress + off);
                hash *= 0x100000001B3UL;
            }
            sb.Append(' ').Append(hash.ToString("x"));

            for (int i = 0; i < 32; i++)
            {
                V128 v = _context.GetV(i);
                sb.Append(' ')
                  .Append(v.Extract<ulong>(1).ToString("x16"))
                  .Append(v.Extract<ulong>(0).ToString("x16"));
            }

            outp.WriteLine(sb.ToString());

            Teardown();

            done++;
            if (done % 500 == 0)
            {
                Console.Error.WriteLine($"{done} vectors...");
            }
        }

        Console.Error.WriteLine($"oracle: {done} vectors written");
    }
}
