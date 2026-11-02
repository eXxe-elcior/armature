// Port of CodeGen/X86/CallConvName.cs and CallingConvention.cs.
// Selects the System V or Windows ABI for the build target.

use super::register::X86Register;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallConvName {
    SystemV,
    Windows,
}

const REGISTERS_MASK: i32 = 0xffff;

pub fn get_int_available_registers() -> i32 {
    REGISTERS_MASK & !(1 << X86Register::RSP.index())
}

pub fn get_vec_available_registers() -> i32 {
    REGISTERS_MASK
}

pub fn get_int_caller_saved_registers() -> i32 {
    if get_current_call_conv() == CallConvName::Windows {
        (1 << X86Register::RAX.index())
            | (1 << X86Register::RCX.index())
            | (1 << X86Register::RDX.index())
            | (1 << X86Register::R8.index())
            | (1 << X86Register::R9.index())
            | (1 << X86Register::R10.index())
            | (1 << X86Register::R11.index())
    } else {
        (1 << X86Register::RAX.index())
            | (1 << X86Register::RCX.index())
            | (1 << X86Register::RDX.index())
            | (1 << X86Register::RSI.index())
            | (1 << X86Register::RDI.index())
            | (1 << X86Register::R8.index())
            | (1 << X86Register::R9.index())
            | (1 << X86Register::R10.index())
            | (1 << X86Register::R11.index())
    }
}

pub fn get_vec_caller_saved_registers() -> i32 {
    if get_current_call_conv() == CallConvName::Windows {
        (1 << X86Register::XMM0.index())
            | (1 << X86Register::XMM1.index())
            | (1 << X86Register::XMM2.index())
            | (1 << X86Register::XMM3.index())
            | (1 << X86Register::XMM4.index())
            | (1 << X86Register::XMM5.index())
    } else {
        REGISTERS_MASK
    }
}

pub fn get_int_callee_saved_registers() -> i32 {
    get_int_caller_saved_registers() ^ REGISTERS_MASK
}

pub fn get_vec_callee_saved_registers() -> i32 {
    get_vec_caller_saved_registers() ^ REGISTERS_MASK
}

pub fn get_arguments_on_regs_count() -> i32 {
    4
}

pub fn get_int_arguments_on_regs_count() -> i32 {
    6
}

pub fn get_vec_arguments_on_regs_count() -> i32 {
    8
}

pub fn get_int_argument_register(index: i32) -> X86Register {
    if get_current_call_conv() == CallConvName::Windows {
        match index {
            0 => return X86Register::RCX,
            1 => return X86Register::RDX,
            2 => return X86Register::R8,
            3 => return X86Register::R9,
            _ => {}
        }
    } else {
        match index {
            0 => return X86Register::RDI,
            1 => return X86Register::RSI,
            2 => return X86Register::RDX,
            3 => return X86Register::RCX,
            4 => return X86Register::R8,
            5 => return X86Register::R9,
            _ => {}
        }
    }

    panic!("get_int_argument_register: index {index} out of range");
}

pub fn get_vec_argument_register(index: i32) -> X86Register {
    let count = if get_current_call_conv() == CallConvName::Windows {
        4
    } else {
        8
    };

    if (index as u32) < count {
        return X86Register(X86Register::XMM0.index() + index);
    }

    panic!("get_vec_argument_register: index {index} out of range");
}

pub fn get_int_return_register() -> X86Register {
    X86Register::RAX
}

pub fn get_int_return_register_high() -> X86Register {
    X86Register::RDX
}

pub fn get_vec_return_register() -> X86Register {
    X86Register::XMM0
}

pub fn get_current_call_conv() -> CallConvName {
    if cfg!(target_os = "windows") {
        CallConvName::Windows
    } else {
        CallConvName::SystemV
    }
}
