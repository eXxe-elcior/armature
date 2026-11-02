using ARMeilleure.Memory;
using ARMeilleure.State;
using Ryujinx.Cpu.Signal;
using Ryujinx.Memory;
using System;
using System.Runtime.InteropServices;

namespace Ryujinx.Cpu.Armature
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct ArmatureMemoryCallbacks
    {
        public IntPtr Ctx;
        public int AddressSpaceBits;
        public int ManagerType;
        public IntPtr PageTablePointer;
        public IntPtr ReadU8, ReadU16, ReadU32, ReadU64, ReadV128;
        public IntPtr WriteU8, WriteU16, WriteU32, WriteU64, WriteV128;
        public IntPtr IsMapped, SignalMemoryTracking;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct ArmatureExceptionCallbacks
    {
        public IntPtr Ctx;
        public IntPtr OnBreak, OnSupervisorCall, OnUndefined, OnInterrupt;
        public IntPtr CounterFrequency, CounterValue;
    }

    internal static class Native
    {
        private const string Lib = "armature";

        [DllImport(Lib)] public static extern IntPtr armature_translator_create(ref ArmatureMemoryCallbacks cb);
        [DllImport(Lib)] public static extern void armature_translator_destroy(IntPtr t);
        [DllImport(Lib)] public static extern void armature_translator_drain_execute(IntPtr t);
        [DllImport(Lib)] public static extern void armature_translator_invalidate(IntPtr t, ulong address, ulong size);
        [DllImport(Lib)] public static extern IntPtr armature_context_create_with_callbacks(ref ArmatureExceptionCallbacks cb);
        [DllImport(Lib)] public static extern void armature_context_destroy(IntPtr c);
        [DllImport(Lib)] public static extern ulong armature_context_get_x(IntPtr c, int i);
        [DllImport(Lib)] public static extern void armature_context_set_x(IntPtr c, int i, ulong v);
        [DllImport(Lib)] public static extern unsafe void armature_context_get_v(IntPtr c, int i, byte* outp);
        [DllImport(Lib)] public static extern unsafe void armature_context_set_v(IntPtr c, int i, byte* v);
        [DllImport(Lib)] public static extern long armature_context_get_tpidr(IntPtr c);
        [DllImport(Lib)] public static extern void armature_context_set_tpidr(IntPtr c, long v);
        [DllImport(Lib)] public static extern long armature_context_get_tpidrro(IntPtr c);
        [DllImport(Lib)] public static extern void armature_context_set_tpidrro(IntPtr c, long v);
        [DllImport(Lib)] public static extern uint armature_context_get_nzcv(IntPtr c);
        [DllImport(Lib)] public static extern void armature_context_set_nzcv(IntPtr c, uint v);
        [DllImport(Lib)] public static extern byte armature_context_is_running(IntPtr c);
        [DllImport(Lib)] public static extern void armature_context_request_interrupt(IntPtr c);
        [DllImport(Lib)] public static extern void armature_context_stop_running(IntPtr c);
        [DllImport(Lib)] public static extern void armature_execute(IntPtr t, IntPtr c, ulong address);
    }

    public sealed class ArmatureExecutionContext : IExecutionContext
    {
        internal readonly IntPtr Handle;

        private readonly GCHandle _self;

        private readonly Delegate[] _pins;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void ExcDel(IntPtr host, IntPtr ec, ulong address, int imm);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void IntDel(IntPtr host, IntPtr ec);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate ulong CntDel(IntPtr host);

        public ArmatureExecutionContext(ICounter counter, ExceptionCallbacks callbacks)
        {
            _self = GCHandle.Alloc(this);

            ExcDel brk = (_, ec, a, i) => callbacks.BreakCallback?.Invoke(this, a, i);
            ExcDel svc = (_, ec, a, i) => callbacks.SupervisorCallback?.Invoke(this, a, i);
            ExcDel und = (_, ec, a, i) => callbacks.UndefinedCallback?.Invoke(this, a, i);
            IntDel irq = (_, ec) => callbacks.InterruptCallback?.Invoke(this);
            CntDel frq = _ => counter.Frequency;
            CntDel cnt = _ => counter.Counter;
            _pins = [brk, svc, und, irq, frq, cnt];

            var cb = new ArmatureExceptionCallbacks
            {
                Ctx = IntPtr.Zero,
                OnBreak = Marshal.GetFunctionPointerForDelegate(brk),
                OnSupervisorCall = Marshal.GetFunctionPointerForDelegate(svc),
                OnUndefined = Marshal.GetFunctionPointerForDelegate(und),
                OnInterrupt = Marshal.GetFunctionPointerForDelegate(irq),
                CounterFrequency = Marshal.GetFunctionPointerForDelegate(frq),
                CounterValue = Marshal.GetFunctionPointerForDelegate(cnt),
            };

            Handle = Native.armature_context_create_with_callbacks(ref cb);
        }

        public ulong Pc => 0;
        public long TpidrEl0 { get => Native.armature_context_get_tpidr(Handle); set => Native.armature_context_set_tpidr(Handle, value); }
        public long TpidrroEl0 { get => Native.armature_context_get_tpidrro(Handle); set => Native.armature_context_set_tpidrro(Handle, value); }
        public uint Pstate { get => Native.armature_context_get_nzcv(Handle); set => Native.armature_context_set_nzcv(Handle, value); }
        public uint Fpcr { get; set; }
        public uint Fpsr { get; set; }
        public bool IsAarch32 { get => false; set { if (value) throw new NotSupportedException("AArch32 excluded"); } }
        public ulong ThreadUid { get; set; }
        public bool Running => Native.armature_context_is_running(Handle) != 0;
        public ulong DebugPc { get; set; }

        public ulong GetX(int index) => Native.armature_context_get_x(Handle, index);
        public void SetX(int index, ulong value) => Native.armature_context_set_x(Handle, index, value);

        public unsafe V128 GetV(int index)
        {
            Span<byte> buf = stackalloc byte[16];
            fixed (byte* p = buf) Native.armature_context_get_v(Handle, index, p);
            return new V128(BitConverter.ToUInt64(buf), BitConverter.ToUInt64(buf.Slice(8)));
        }
        public unsafe void SetV(int index, V128 value)
        {
            Span<byte> buf = stackalloc byte[16];
            BitConverter.TryWriteBytes(buf, value.Extract<ulong>(0));
            BitConverter.TryWriteBytes(buf.Slice(8), value.Extract<ulong>(1));
            fixed (byte* p = buf) Native.armature_context_set_v(Handle, index, p);
        }

        public void RequestInterrupt() => Native.armature_context_request_interrupt(Handle);
        public void StopRunning() => Native.armature_context_stop_running(Handle);
        public void RequestDebugStep() => throw new NotSupportedException();

        public void Dispose()
        {
            Native.armature_context_destroy(Handle);
            _self.Free();
        }
    }

    public sealed class ArmatureCpuContext : ICpuContext
    {
        private readonly IntPtr _translator;
        private readonly ICounter _counter;
        private readonly Delegate[] _pins;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate byte R8(IntPtr c, ulong va);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate ushort R16(IntPtr c, ulong va);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate uint R32(IntPtr c, ulong va);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate ulong R64(IntPtr c, ulong va);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private unsafe delegate void RV(IntPtr c, ulong va, byte* o);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void W8(IntPtr c, ulong va, byte v);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void W16(IntPtr c, ulong va, ushort v);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void W32(IntPtr c, ulong va, uint v);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void W64(IntPtr c, ulong va, ulong v);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private unsafe delegate void WV(IntPtr c, ulong va, byte* v);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate byte IM(IntPtr c, ulong va);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void SMT(IntPtr c, ulong va, ulong size, byte write);

        public unsafe ArmatureCpuContext(ITickSource tickSource, IMemoryManager memory, bool for64Bit)
        {
            if (!for64Bit) throw new NotSupportedException("AArch32 excluded");
            _counter = tickSource;

            R8 r8 = (_, va) => memory.Read<byte>(va);
            R16 r16 = (_, va) => memory.Read<ushort>(va);
            R32 r32 = (_, va) => memory.Read<uint>(va);
            R64 r64 = (_, va) => memory.Read<ulong>(va);
            RV rv = (_, va, o) => { var v = memory.Read<V128>(va); *(ulong*)o = v.Extract<ulong>(0); *(ulong*)(o + 8) = v.Extract<ulong>(1); };
            W8 w8 = (_, va, v) => memory.Write(va, v);
            W16 w16 = (_, va, v) => memory.Write(va, v);
            W32 w32 = (_, va, v) => memory.Write(va, v);
            W64 w64 = (_, va, v) => memory.Write(va, v);
            WV wv = (_, va, v) => memory.Write(va, new V128(*(ulong*)v, *(ulong*)(v + 8)));
            IM im = (_, va) => (byte)(memory.IsMapped(va) ? 1 : 0);
            SMT smt = (_, va, size, write) => memory.SignalMemoryTracking(va, size, write != 0);
            _pins = [r8, r16, r32, r64, rv, w8, w16, w32, w64, wv, im, smt];

            var cb = new ArmatureMemoryCallbacks
            {
                Ctx = IntPtr.Zero,
                AddressSpaceBits = memory.AddressSpaceBits,
                ManagerType = (int)memory.Type,
                PageTablePointer = memory.PageTablePointer,
                ReadU8 = Marshal.GetFunctionPointerForDelegate(r8),
                ReadU16 = Marshal.GetFunctionPointerForDelegate(r16),
                ReadU32 = Marshal.GetFunctionPointerForDelegate(r32),
                ReadU64 = Marshal.GetFunctionPointerForDelegate(r64),
                ReadV128 = Marshal.GetFunctionPointerForDelegate(rv),
                WriteU8 = Marshal.GetFunctionPointerForDelegate(w8),
                WriteU16 = Marshal.GetFunctionPointerForDelegate(w16),
                WriteU32 = Marshal.GetFunctionPointerForDelegate(w32),
                WriteU64 = Marshal.GetFunctionPointerForDelegate(w64),
                WriteV128 = Marshal.GetFunctionPointerForDelegate(wv),
                IsMapped = Marshal.GetFunctionPointerForDelegate(im),
                SignalMemoryTracking = Marshal.GetFunctionPointerForDelegate(smt),
            };

            _translator = Native.armature_translator_create(ref cb);

            // Host-mapped/host-tracked commit mirror pages lazily via a SIGSEGV
            // handler. Like JitCpuContext, install it; without this our JIT's
            // faults on not-yet-committed pages are never resumed → hard crash.
            if (memory.Type.IsHostMappedOrTracked)
            {
                NativeSignalHandler.InitializeSignalHandler();
            }

            memory.UnmapEvent += UnmapHandler;
        }

        private void UnmapHandler(ulong address, ulong size)
        {
            Native.armature_translator_invalidate(_translator, address, size);
        }

        public IExecutionContext CreateExecutionContext(ExceptionCallbacks exceptionCallbacks)
            => new ArmatureExecutionContext(_counter, exceptionCallbacks);

        public void Execute(IExecutionContext context, ulong address)
            => Native.armature_execute(_translator, ((ArmatureExecutionContext)context).Handle, address);

        public void InvalidateCacheRegion(ulong address, ulong size)
            => Native.armature_translator_invalidate(_translator, address, size);

        public IDiskCacheLoadState LoadDiskCache(string titleIdText, string displayVersion, bool enabled, string cacheSelector)
            => new DummyDiskCacheLoadState();

        public void PrepareCodeRange(ulong address, ulong size) { }

        /// Block until every guest thread has left the JIT execute loop. Must be
        /// called (after StopRunning on all contexts) before the guest address
        /// space is unmapped, else in-flight JIT memory accesses fault on freed
        /// pages → InvalidMemoryRegionException on STOP.
        public void DrainExecute()
        {
            Native.armature_translator_drain_execute(_translator);
        }

        public void Dispose()
        {
            Native.armature_translator_destroy(_translator);
        }
    }
}
