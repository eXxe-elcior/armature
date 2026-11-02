using ARMeilleure.Memory;

namespace Ryujinx.Cpu.Armature
{
    public class ArmatureEngine : ICpuEngine
    {
        private readonly ITickSource _tickSource;

        public ArmatureEngine(ITickSource tickSource)
        {
            _tickSource = tickSource;
        }

        public ICpuContext CreateCpuContext(IMemoryManager memoryManager, bool for64Bit)
        {
            return new ArmatureCpuContext(_tickSource, memoryManager, for64Bit);
        }
    }
}
