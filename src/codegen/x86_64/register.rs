// Port of CodeGen/X86/X86Register.cs.
//
// Divergence (forced): the C# enum gives GPRs and XMMs overlapping
// discriminants (Rax = 0, Xmm0 = 0)  Rust enums can't. A newtype with
// associated consts keeps the C# semantics: the numeric value is the
// hardware encoding, and int-vs-vec is contextual, exactly as in C#.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct X86Register(pub i32);

impl X86Register {
    pub const INVALID: X86Register = X86Register(-1);

    pub const RAX: X86Register = X86Register(0);
    pub const RCX: X86Register = X86Register(1);
    pub const RDX: X86Register = X86Register(2);
    pub const RBX: X86Register = X86Register(3);
    pub const RSP: X86Register = X86Register(4);
    pub const RBP: X86Register = X86Register(5);
    pub const RSI: X86Register = X86Register(6);
    pub const RDI: X86Register = X86Register(7);
    pub const R8: X86Register = X86Register(8);
    pub const R9: X86Register = X86Register(9);
    pub const R10: X86Register = X86Register(10);
    pub const R11: X86Register = X86Register(11);
    pub const R12: X86Register = X86Register(12);
    pub const R13: X86Register = X86Register(13);
    pub const R14: X86Register = X86Register(14);
    pub const R15: X86Register = X86Register(15);

    pub const XMM0: X86Register = X86Register(0);
    pub const XMM1: X86Register = X86Register(1);
    pub const XMM2: X86Register = X86Register(2);
    pub const XMM3: X86Register = X86Register(3);
    pub const XMM4: X86Register = X86Register(4);
    pub const XMM5: X86Register = X86Register(5);
    pub const XMM6: X86Register = X86Register(6);
    pub const XMM7: X86Register = X86Register(7);
    pub const XMM8: X86Register = X86Register(8);
    pub const XMM9: X86Register = X86Register(9);
    pub const XMM10: X86Register = X86Register(10);
    pub const XMM11: X86Register = X86Register(11);
    pub const XMM12: X86Register = X86Register(12);
    pub const XMM13: X86Register = X86Register(13);
    pub const XMM14: X86Register = X86Register(14);
    pub const XMM15: X86Register = X86Register(15);

    #[inline]
    pub fn index(self) -> i32 {
        self.0
    }
}
