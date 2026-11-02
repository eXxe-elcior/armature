// Port of Instructions/SoftFloat.cs: floating-point helpers called by JIT code.
// FPCR/FPSR state is thread-local.
// standard_fpcr selects default FPCR settings instead of thread-local state.

use std::cell::Cell;
use std::sync::LazyLock;

// Thread-local FPCR / FPSR state (C# NativeInterface.GetContext)
thread_local! {
    static SF_FPCR: Cell<u32> = const { Cell::new(0) };
    static SF_FPSR: Cell<u32> = const { Cell::new(0) };
}

pub fn set_fpcr(value: u32) {
    SF_FPCR.with(|f| f.set(value));
}
pub fn get_fpcr() -> u32 {
    SF_FPCR.with(|f| f.get())
}
pub fn set_fpsr(value: u32) {
    SF_FPSR.with(|f| f.set(value));
}
pub fn get_fpsr() -> u32 {
    SF_FPSR.with(|f| f.get())
}

/// Standard FPCR value: DN=0, FZ=0, RMode=ToNearest, IDE/IXE/...=0.
const STANDARD_FPCR: u32 = 0;

// Enums
#[derive(Clone, Copy, PartialEq, Eq)]
enum FPType {
    Zero,
    Infinity,
    QNaN,
    SNaN,
    Nonzero,
}

#[derive(Clone, Copy)]
enum FPException {
    InvalidOp,
    DivideByZero,
    Overflow,
    Underflow,
    Inexact,
    InputDenorm,
}

#[derive(Clone, Copy)]
enum FPRoundingMode {
    ToNearest,
    TowardsPlusInfinity,
    TowardsMinusInfinity,
    TowardsZero,
}

// RecipEstimateTable (C# SoftFloat.RecipEstimateTable)
static RECIP_ESTIMATE_TABLE: LazyLock<[u8; 256]> = LazyLock::new(|| {
    let mut tbl = [0u8; 256];
    for idx in 0..256 {
        let mut src = (idx as u32) + 256;
        src = (src << 1) + 1;
        let aux = (1u32 << 19) / src;
        let dst = (aux + 1) >> 1;
        tbl[idx] = (dst - 256) as u8;
    }
    tbl
});

// RecipSqrtEstimateTable (C# SoftFloat.RecipSqrtEstimateTable)
static RECIP_SQRT_ESTIMATE_TABLE: LazyLock<[u8; 384]> = LazyLock::new(|| {
    let mut tbl = [0u8; 384];
    for idx in 0..384 {
        let mut src = (idx as u32) + 128;
        if src < 256 {
            src = (src << 1) + 1;
        } else {
            src = (src >> 1) << 1;
            src = (src + 1) << 1;
        }
        let mut aux = 512;
        while src * (aux + 1) * (aux + 1) < (1u32 << 28) {
            aux += 1;
        }
        let dst = (aux + 1) >> 1;
        tbl[idx] = (dst - 256) as u8;
    }
    tbl
});

// Shared helpers (C# SoftFloat static methods)
fn get_rounding_mode(fpcr: u32) -> FPRoundingMode {
    match (fpcr >> 22) & 3 {
        0 => FPRoundingMode::ToNearest,
        1 => FPRoundingMode::TowardsPlusInfinity,
        2 => FPRoundingMode::TowardsMinusInfinity,
        3 => FPRoundingMode::TowardsZero,
        _ => unreachable!(),
    }
}

/// C# SoftFloat.FPProcessException(exc, context, fpcr).
/// When traps are enabled (FPCR.{IDE,IXE,...}) → panic. Otherwise set FPSR bit.
fn fp_process_exception(exc: FPException, fpcr: u32, fpsr: &mut u32) {
    let enable = 1 << ((exc as u32) + 8);
    if (fpcr & enable) != 0 {
        unimplemented!("Floating-point trap handling");
    }
    *fpsr |= 1 << (exc as u32);
}

fn fp_process_exception_ctx(exc: FPException, fpcr: u32) {
    SF_FPSR.with(|f| {
        let mut fpsr = f.get();
        fp_process_exception(exc, fpcr, &mut fpsr);
        f.set(fpsr);
    });
}

// SoftFloat32 private helpers
fn sf32_apply_fz(value: f32) -> f32 {
    if value.is_subnormal() {
        SF_FPSR.with(|f| f.set(f.get() | (1 << FPException::Underflow as u32)));
        if value < 0.0 { -0.0f32 } else { 0.0f32 }
    } else {
        value
    }
}

fn sf32_fpzero(sign: bool) -> f32 {
    if sign { -0.0f32 } else { 0.0f32 }
}

fn sf32_fpinfinity(sign: bool) -> f32 {
    if sign {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    }
}

fn sf32_fpdefaultnan() -> f32 {
    f32::from_bits(0x7fc00000)
}

fn sf32_fpmaxnormal(sign: bool) -> f32 {
    if sign { f32::MIN } else { f32::MAX }
}

fn sf32_fptwo(sign: bool) -> f32 {
    if sign { -2.0 } else { 2.0 }
}

fn sf32_fpthree(sign: bool) -> f32 {
    if sign { -3.0 } else { 3.0 }
}

fn sf32_fponepointfive(sign: bool) -> f32 {
    if sign { -1.5 } else { 1.5 }
}

fn sf32_zerosorones(ones: bool) -> f32 {
    f32::from_bits(if ones { 0xFFFFFFFFu32 } else { 0 })
}

fn sf32_fpneg(value: f32) -> f32 {
    -value
}

// C# float.FPUnpack extension method.
fn sf32_fpunpack(value: f32, fpcr: u32) -> (f32, FPType, bool, u32) {
    let bits = value.to_bits();
    let sign = (bits & 0x80000000u32) != 0;
    let exp = bits & 0x7F800000u32;
    let frac = bits & 0x007FFFFFu32;

    if exp == 0 {
        if frac == 0 || (fpcr & (1 << 24)) != 0 {
            // FPCR.Fz = bit 24
            if frac != 0 {
                fp_process_exception_ctx(FPException::InputDenorm, fpcr);
            }
            (sf32_fpzero(sign), FPType::Zero, sign, bits)
        } else {
            (value, FPType::Nonzero, sign, bits)
        }
    } else if exp == 0x7F800000u32 {
        if frac == 0 {
            (value, FPType::Infinity, sign, bits)
        } else {
            let typ = if (frac & 0x00400000u32) != 0 {
                FPType::QNaN
            } else {
                FPType::SNaN
            };
            (sf32_fpzero(sign), typ, sign, bits)
        }
    } else {
        (value, FPType::Nonzero, sign, bits)
    }
}

// C# SoftFloat32.FPProcessNaN.
fn sf32_fpprocessnan(typ: FPType, op: u32, fpcr: u32) -> f32 {
    let bits = if typ == FPType::SNaN {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        op | (1u32 << 22)
    } else {
        op
    };
    if (fpcr & (1 << 25)) != 0 {
        // FPCR.Dn = bit 25
        sf32_fpdefaultnan()
    } else {
        f32::from_bits(bits)
    }
}

// C# SoftFloat32.FPProcessNaNs (2-operand).
fn sf32_fpprocessnans(type1: FPType, type2: FPType, op1: u32, op2: u32, fpcr: u32) -> (f32, bool) {
    if type1 == FPType::SNaN {
        (sf32_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::SNaN {
        (sf32_fpprocessnan(type2, op2, fpcr), true)
    } else if type1 == FPType::QNaN {
        (sf32_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::QNaN {
        (sf32_fpprocessnan(type2, op2, fpcr), true)
    } else {
        (0.0f32, false)
    }
}

// C# SoftFloat32.FPMulFpscrImpl (used by FPMul, FPMulX, FPRecipStep, FPRSqrtStep).
fn sf32_fpmulfpscr_impl(value1: f32, value2: f32, standard_fpcr: bool) -> f32 {
    let fpcr = if standard_fpcr {
        STANDARD_FPCR
    } else {
        get_fpcr()
    };
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    if inf1 || inf2 {
        return sf32_fpinfinity(s1 ^ s2);
    }
    if zero1 || zero2 {
        return sf32_fpzero(s1 ^ s2);
    }
    let mut result = v1 * v2;
    if (fpcr & (1 << 24)) != 0 {
        // Fz
        result = sf32_apply_fz(result);
    }
    result
}

// SoftFloat32 extern "C" functions (C# SoftFloat32.[UnmanagedCallersOnly])
pub extern "C" fn sf32_fpmulx(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        return sf32_fptwo(s1 ^ s2);
    }
    if inf1 || inf2 {
        return sf32_fpinfinity(s1 ^ s2);
    }
    if zero1 || zero2 {
        return sf32_fpzero(s1 ^ s2);
    }
    let mut result = v1 * v2;
    if (fpcr & (1 << 24)) != 0 {
        // Fz
        result = sf32_apply_fz(result);
    }
    result
}

pub extern "C" fn sf32_fprecipestimate(value: f32) -> f32 {
    let fpcr = get_fpcr();
    let (_v, typ, sign, op) = sf32_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf32_fpprocessnan(typ, op, fpcr);
    }
    if typ == FPType::Infinity {
        return sf32_fpzero(sign);
    }
    if typ == FPType::Zero {
        fp_process_exception_ctx(FPException::DivideByZero, fpcr);
        return sf32_fpinfinity(sign);
    }
    if value.abs() < 2.0f32.powi(-128) {
        let overflow_to_inf = match get_rounding_mode(fpcr) {
            FPRoundingMode::ToNearest => true,
            FPRoundingMode::TowardsPlusInfinity => !sign,
            FPRoundingMode::TowardsMinusInfinity => sign,
            FPRoundingMode::TowardsZero => false,
        };
        fp_process_exception_ctx(FPException::Overflow, fpcr);
        fp_process_exception_ctx(FPException::Inexact, fpcr);
        return if overflow_to_inf {
            sf32_fpinfinity(sign)
        } else {
            sf32_fpmaxnormal(sign)
        };
    }
    if (fpcr & (1 << 24)) != 0 && value.abs() >= 2.0f32.powi(126) {
        // Fz
        SF_FPSR.with(|f| f.set(f.get() | (1 << 3))); // UFC bit
        return sf32_fpzero(sign);
    }
    let mut fraction = (op as u64 & 0x007FFFFFu64) << 29;
    let mut exp = (op & 0x7F800000u32) >> 23;
    if exp == 0 {
        if (fraction & 0x0008000000000000u64) == 0 {
            fraction = (fraction & 0x0003FFFFFFFFFFFFu64) << 2;
            exp = exp.wrapping_sub(1);
        } else {
            fraction = (fraction & 0x0007FFFFFFFFFFFFu64) << 1;
        }
    }
    let scaled = ((fraction & 0x000FF00000000000u64) | 0x0010000000000000u64) >> 44;
    let result_exp = 253u32.wrapping_sub(exp);
    let estimate = RECIP_ESTIMATE_TABLE[(scaled - 256) as usize] as u64 + 256;
    fraction = estimate << 44;
    if result_exp == 0 {
        fraction = ((fraction & 0x000FFFFFFFFFFFFEu64) | 0x0010000000000000u64) >> 1;
    } else if result_exp == 0xFFFFFFFFu32 {
        fraction = ((fraction & 0x000FFFFFFFFFFFFCu64) | 0x0010000000000000u64) >> 2;
    }
    let result_bits = ((if sign { 1u32 } else { 0 }) << 31)
        | ((result_exp & 0xFF) << 23)
        | ((fraction >> 29) as u32 & 0x007FFFFFu32);
    f32::from_bits(result_bits)
}

pub extern "C" fn sf32_fprecpx(value: f32) -> f32 {
    let fpcr = get_fpcr();
    let (_v, typ, sign, op) = sf32_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf32_fpprocessnan(typ, op, fpcr);
    }
    let not_exp = (!op >> 23) & 0xFF;
    let max_exp = 0xFE;
    let result_exp = if not_exp == 0xFF { max_exp } else { not_exp };
    let bits = ((if sign { 1u32 } else { 0 }) << 31) | (result_exp << 23);
    f32::from_bits(bits)
}

pub extern "C" fn sf32_fprsqtestimate(value: f32) -> f32 {
    let fpcr = get_fpcr();
    let (_v, typ, sign, op) = sf32_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf32_fpprocessnan(typ, op, fpcr);
    }
    if typ == FPType::Zero {
        fp_process_exception_ctx(FPException::DivideByZero, fpcr);
        return sf32_fpinfinity(sign);
    }
    if sign {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    if typ == FPType::Infinity {
        return sf32_fpzero(false);
    }
    let mut fraction = (op as u64 & 0x007FFFFFu64) << 29;
    let mut exp = (op & 0x7F800000u32) >> 23;
    if exp == 0 {
        while (fraction & 0x0008000000000000u64) == 0 {
            fraction = (fraction & 0x0007FFFFFFFFFFFFu64) << 1;
            exp = exp.wrapping_sub(1);
        }
        fraction = (fraction & 0x0007FFFFFFFFFFFFu64) << 1;
    }
    let scaled = if (exp & 1) == 0 {
        ((fraction & 0x000FF00000000000u64) | 0x0010000000000000u64) >> 44
    } else {
        ((fraction & 0x000FE00000000000u64) | 0x0010000000000000u64) >> 45
    };
    let result_exp = (380u32.wrapping_sub(exp)) >> 1;
    let estimate = RECIP_SQRT_ESTIMATE_TABLE[(scaled - 128) as usize] as u64 + 256;
    let bits = ((result_exp & 0xFF) << 23) | ((estimate as u32 & 0xFF) << 15);
    f32::from_bits(bits)
}

fn sf32_fpadd_impl(value1: f32, value2: f32, fpcr: u32) -> f32 {
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if inf1 && inf2 && s1 != s2 {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    if (inf1 && !s1) || (inf2 && !s2) {
        return sf32_fpinfinity(false);
    }
    if (inf1 && s1) || (inf2 && s2) {
        return sf32_fpinfinity(true);
    }
    if zero1 && zero2 && s1 == s2 {
        return sf32_fpzero(s1);
    }
    let mut result = v1 + v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf32_apply_fz(result);
    }
    result
}

pub extern "C" fn sf32_fpadd(value1: f32, value2: f32) -> f32 {
    sf32_fpadd_impl(value1, value2, get_fpcr())
}

fn sf32_fpsub_impl(value1: f32, value2: f32, fpcr: u32) -> f32 {
    sf32_fpadd_impl(value1, -value2, fpcr)
}

pub extern "C" fn sf32_fpsub(value1: f32, value2: f32) -> f32 {
    sf32_fpsub_impl(value1, value2, get_fpcr())
}

// C# SoftFloat32.FPHalvedSub (used by FPRSqrtStep with standard FPCR).
fn sf32_fphalvedsub_impl(value1: f32, value2: f32, fpcr: u32) -> f32 {
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if inf1 && inf2 && s1 == s2 {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    if (inf1 && !s1) || (inf2 && s2) {
        return sf32_fpinfinity(false);
    }
    if (inf1 && s1) || (inf2 && !s2) {
        return sf32_fpinfinity(true);
    }
    if zero1 && zero2 && s1 == !s2 {
        return sf32_fpzero(s1);
    }
    let mut result = (v1 - v2) / 2.0;
    if (fpcr & (1 << 24)) != 0 {
        result = sf32_apply_fz(result);
    }
    result
}

pub extern "C" fn sf32_fpmul(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    if inf1 || inf2 {
        return sf32_fpinfinity(s1 ^ s2);
    }
    if zero1 || zero2 {
        return sf32_fpzero(s1 ^ s2);
    }
    let mut result = v1 * v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf32_apply_fz(result);
    }
    result
}

pub extern "C" fn sf32_fpdiv(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (mut result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && inf2) || (zero1 && zero2) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    if inf1 || zero2 {
        if !inf1 {
            fp_process_exception_ctx(FPException::DivideByZero, fpcr);
        }
        return sf32_fpinfinity(s1 ^ s2);
    }
    if zero1 || inf2 {
        return sf32_fpzero(s1 ^ s2);
    }
    result = v1 / v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf32_apply_fz(result);
    }
    result
}

pub extern "C" fn sf32_fpsqrt(value: f32) -> f32 {
    let fpcr = get_fpcr();
    let (v, typ, sign, op) = sf32_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf32_fpprocessnan(typ, op, fpcr);
    }
    if typ == FPType::Zero {
        return sf32_fpzero(sign);
    }
    if typ == FPType::Infinity && !sign {
        return sf32_fpinfinity(sign);
    }
    if sign {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    let mut result = v.sqrt();
    if (fpcr & (1 << 24)) != 0 {
        result = sf32_apply_fz(result);
    }
    result
}

pub extern "C" fn sf32_fpmax(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    if v1 > v2 {
        if t1 == FPType::Infinity {
            return sf32_fpinfinity(s1);
        }
        if t1 == FPType::Zero {
            return sf32_fpzero(s1 && s2);
        }
        let mut r = v1;
        if (fpcr & (1 << 24)) != 0 {
            r = sf32_apply_fz(r);
        }
        r
    } else {
        if t2 == FPType::Infinity {
            return sf32_fpinfinity(s2);
        }
        if t2 == FPType::Zero {
            return sf32_fpzero(s1 && s2);
        }
        let mut r = v2;
        if (fpcr & (1 << 24)) != 0 {
            r = sf32_apply_fz(r);
        }
        r
    }
}

pub extern "C" fn sf32_fpmin(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    if v1 < v2 {
        if t1 == FPType::Infinity {
            return sf32_fpinfinity(s1);
        }
        if t1 == FPType::Zero {
            return sf32_fpzero(s1 || s2);
        }
        let mut r = v1;
        if (fpcr & (1 << 24)) != 0 {
            r = sf32_apply_fz(r);
        }
        r
    } else {
        if t2 == FPType::Infinity {
            return sf32_fpinfinity(s2);
        }
        if t2 == FPType::Zero {
            return sf32_fpzero(s1 || s2);
        }
        let mut r = v2;
        if (fpcr & (1 << 24)) != 0 {
            r = sf32_apply_fz(r);
        }
        r
    }
}

pub extern "C" fn sf32_fpcompare(value1: f32, value2: f32, signal_nans: u8) -> i32 {
    let fpcr = get_fpcr();
    let (v1, t1, _, _) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, _, _) = sf32_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        if t1 == FPType::SNaN || t2 == FPType::SNaN || signal_nans != 0 {
            fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        }
        return 0b0011;
    }
    if v1 == v2 {
        return 0b0110;
    }
    if v1 < v2 {
        return 0b1000;
    }
    0b0010
}

pub extern "C" fn sf32_fpcompareeq(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (_, t1, _, _) = sf32_fpunpack(value1, fpcr);
    let (_, t2, _, _) = sf32_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        if t1 == FPType::SNaN || t2 == FPType::SNaN {
            fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        }
        return sf32_zerosorones(false);
    }
    sf32_zerosorones(value1 == value2)
}

pub extern "C" fn sf32_fpcomparege(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (_, t1, _, _) = sf32_fpunpack(value1, fpcr);
    let (_, t2, _, _) = sf32_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_zerosorones(false);
    }
    sf32_zerosorones(value1 >= value2)
}

pub extern "C" fn sf32_fpcomparegt(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let (_, t1, _, _) = sf32_fpunpack(value1, fpcr);
    let (_, t2, _, _) = sf32_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_zerosorones(false);
    }
    sf32_zerosorones(value1 > value2)
}

pub extern "C" fn sf32_fpcomparele(value1: f32, value2: f32) -> f32 {
    sf32_fpcomparege(value2, value1)
}

pub extern "C" fn sf32_fpcomparelt(value1: f32, value2: f32) -> f32 {
    sf32_fpcomparegt(value2, value1)
}

pub extern "C" fn sf32_fprecipstep(value1: f32, value2: f32) -> f32 {
    let fpcr = STANDARD_FPCR;
    let (v1, t1, _, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, _, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    let product = if (inf1 && zero2) || (zero1 && inf2) {
        sf32_fpzero(false)
    } else {
        sf32_fpmulfpscr_impl(v1, v2, true)
    };
    sf32_fpsub_impl(sf32_fptwo(false), product, fpcr)
}

pub extern "C" fn sf32_fprsqrtstep(value1: f32, value2: f32) -> f32 {
    let fpcr = STANDARD_FPCR;
    let (v1, t1, _, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, _, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    let product = if (inf1 && zero2) || (zero1 && inf2) {
        sf32_fpzero(false)
    } else {
        sf32_fpmulfpscr_impl(v1, v2, true)
    };
    sf32_fphalvedsub_impl(sf32_fpthree(false), product, fpcr)
}

pub extern "C" fn sf32_fprecipstepfused(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let v1 = sf32_fpneg(value1);
    let (v1, t1, s1, o1) = sf32_fpunpack(v1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        return sf32_fptwo(false);
    }
    if inf1 || inf2 {
        return sf32_fpinfinity(s1 ^ s2);
    }
    let mut result = v1.mul_add(v2, 2.0);
    if (fpcr & (1 << 24)) != 0 {
        result = sf32_apply_fz(result);
    }
    result
}

pub extern "C" fn sf32_fprsqrtstepfused(value1: f32, value2: f32) -> f32 {
    let fpcr = get_fpcr();
    let v1 = sf32_fpneg(value1);
    let (v1, t1, s1, o1) = sf32_fpunpack(v1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let (result, done) = sf32_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        return sf32_fponepointfive(false);
    }
    if inf1 || inf2 {
        return sf32_fpinfinity(s1 ^ s2);
    }
    let mut result = v1.mul_add(v2, 3.0) / 2.0;
    if (fpcr & (1 << 24)) != 0 {
        result = sf32_apply_fz(result);
    }
    result
}

fn sf32_fpprocessnans3(
    type1: FPType,
    type2: FPType,
    type3: FPType,
    op1: u32,
    op2: u32,
    op3: u32,
    fpcr: u32,
) -> (f32, bool) {
    if type1 == FPType::SNaN {
        (sf32_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::SNaN {
        (sf32_fpprocessnan(type2, op2, fpcr), true)
    } else if type3 == FPType::SNaN {
        (sf32_fpprocessnan(type3, op3, fpcr), true)
    } else if type1 == FPType::QNaN {
        (sf32_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::QNaN {
        (sf32_fpprocessnan(type2, op2, fpcr), true)
    } else if type3 == FPType::QNaN {
        (sf32_fpprocessnan(type3, op3, fpcr), true)
    } else {
        (0.0f32, false)
    }
}

fn sf32_fpmuladd_impl(value_a: f32, value1: f32, value2: f32, standard_fpcr: bool) -> f32 {
    let fpcr = if standard_fpcr {
        STANDARD_FPCR
    } else {
        get_fpcr()
    };
    let (va, ta, sa, oa) = sf32_fpunpack(value_a, fpcr);
    let (v1, t1, s1, o1) = sf32_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf32_fpunpack(value2, fpcr);
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    let (result, done) = sf32_fpprocessnans3(ta, t1, t2, oa, o1, o2, fpcr);
    if ta == FPType::QNaN && ((inf1 && zero2) || (zero1 && inf2)) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf32_fpdefaultnan();
    }
    if !done {
        let inf_a = ta == FPType::Infinity;
        let zero_a = ta == FPType::Zero;
        let sign_p = s1 ^ s2;
        let inf_p = inf1 || inf2;
        let zero_p = zero1 || zero2;
        if (inf1 && zero2) || (zero1 && inf2) || (inf_a && inf_p && sa != sign_p) {
            fp_process_exception_ctx(FPException::InvalidOp, fpcr);
            return sf32_fpdefaultnan();
        }
        if (inf_a && !sa) || (inf_p && !sign_p) {
            return sf32_fpinfinity(false);
        }
        if (inf_a && sa) || (inf_p && sign_p) {
            return sf32_fpinfinity(true);
        }
        if zero_a && zero_p && sa == sign_p {
            return sf32_fpzero(sa);
        }
        let mut rv = v1.mul_add(v2, va);
        if (fpcr & (1 << 24)) != 0 {
            rv = sf32_apply_fz(rv);
        }
        rv
    } else {
        result
    }
}

pub extern "C" fn sf32_fpmuladd(value_a: f32, value1: f32, value2: f32) -> f32 {
    sf32_fpmuladd_impl(value_a, value1, value2, false)
}

pub extern "C" fn sf32_fpmulsub(value_a: f32, value1: f32, value2: f32) -> f32 {
    sf32_fpmuladd_impl(value_a, -value1, value2, false)
}

pub extern "C" fn sf32_fpnegmuladd(value_a: f32, value1: f32, value2: f32) -> f32 {
    sf32_fpmuladd_impl(-value_a, -value1, value2, false)
}

pub extern "C" fn sf32_fpnegmulsub(value_a: f32, value1: f32, value2: f32) -> f32 {
    sf32_fpmuladd_impl(-value_a, value1, value2, false)
}

// SoftFloat64 private helpers
fn sf64_apply_fz(value: f64) -> f64 {
    if value.is_subnormal() {
        SF_FPSR.with(|f| f.set(f.get() | (1 << FPException::Underflow as u32)));
        if value < 0.0 { -0.0f64 } else { 0.0f64 }
    } else {
        value
    }
}

fn sf64_fpzero(sign: bool) -> f64 {
    if sign { -0.0f64 } else { 0.0f64 }
}

fn sf64_fpinfinity(sign: bool) -> f64 {
    if sign {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

fn sf64_fpdefaultnan() -> f64 {
    f64::from_bits(0x7ff8000000000000u64)
}

fn sf64_fpmaxnormal(sign: bool) -> f64 {
    if sign { f64::MIN } else { f64::MAX }
}

fn sf64_fptwo(sign: bool) -> f64 {
    if sign { -2.0 } else { 2.0 }
}

fn sf64_fpthree(sign: bool) -> f64 {
    if sign { -3.0 } else { 3.0 }
}

fn sf64_fponepointfive(sign: bool) -> f64 {
    if sign { -1.5 } else { 1.5 }
}

fn sf64_zerosorones(ones: bool) -> f64 {
    f64::from_bits(if ones { 0xFFFFFFFFFFFFFFFFu64 } else { 0 })
}

fn sf64_fpneg(value: f64) -> f64 {
    -value
}

fn sf64_fpunpack(value: f64, fpcr: u32) -> (f64, FPType, bool, u64) {
    let bits = value.to_bits();
    let sign = (bits & 0x8000000000000000u64) != 0;
    let exp = bits & 0x7FF0000000000000u64;
    let frac = bits & 0x000FFFFFFFFFFFFFu64;
    if exp == 0 {
        if frac == 0 || (fpcr & (1 << 24)) != 0 {
            // Fz
            if frac != 0 {
                fp_process_exception_ctx(FPException::InputDenorm, fpcr);
            }
            (sf64_fpzero(sign), FPType::Zero, sign, bits)
        } else {
            (value, FPType::Nonzero, sign, bits)
        }
    } else if exp == 0x7FF0000000000000u64 {
        if frac == 0 {
            (value, FPType::Infinity, sign, bits)
        } else {
            let typ = if (frac & 0x0008000000000000u64) != 0 {
                FPType::QNaN
            } else {
                FPType::SNaN
            };
            (sf64_fpzero(sign), typ, sign, bits)
        }
    } else {
        (value, FPType::Nonzero, sign, bits)
    }
}

fn sf64_fpprocessnan(typ: FPType, op: u64, fpcr: u32) -> f64 {
    let bits = if typ == FPType::SNaN {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        op | (1u64 << 51)
    } else {
        op
    };
    if (fpcr & (1 << 25)) != 0 {
        // Dn
        sf64_fpdefaultnan()
    } else {
        f64::from_bits(bits)
    }
}

fn sf64_fpprocessnans(type1: FPType, type2: FPType, op1: u64, op2: u64, fpcr: u32) -> (f64, bool) {
    if type1 == FPType::SNaN {
        (sf64_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::SNaN {
        (sf64_fpprocessnan(type2, op2, fpcr), true)
    } else if type1 == FPType::QNaN {
        (sf64_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::QNaN {
        (sf64_fpprocessnan(type2, op2, fpcr), true)
    } else {
        (0.0f64, false)
    }
}

fn sf64_fpmulfpscr_impl(value1: f64, value2: f64, standard_fpcr: bool) -> f64 {
    let fpcr = if standard_fpcr {
        STANDARD_FPCR
    } else {
        get_fpcr()
    };
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    if inf1 || inf2 {
        return sf64_fpinfinity(s1 ^ s2);
    }
    if zero1 || zero2 {
        return sf64_fpzero(s1 ^ s2);
    }
    let mut result = v1 * v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

// SoftFloat64 extern "C" functions
fn sf64_fpadd_impl(value1: f64, value2: f64, fpcr: u32) -> f64 {
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if inf1 && inf2 && s1 != s2 {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    if (inf1 && !s1) || (inf2 && !s2) {
        return sf64_fpinfinity(false);
    }
    if (inf1 && s1) || (inf2 && s2) {
        return sf64_fpinfinity(true);
    }
    if zero1 && zero2 && s1 == s2 {
        return sf64_fpzero(s1);
    }
    let mut result = v1 + v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fpadd(value1: f64, value2: f64) -> f64 {
    sf64_fpadd_impl(value1, value2, get_fpcr())
}

fn sf64_fpsub_impl(value1: f64, value2: f64, fpcr: u32) -> f64 {
    sf64_fpadd_impl(value1, -value2, fpcr)
}

pub extern "C" fn sf64_fpsub(value1: f64, value2: f64) -> f64 {
    sf64_fpsub_impl(value1, value2, get_fpcr())
}

fn sf64_fphalvedsub_impl(value1: f64, value2: f64, fpcr: u32) -> f64 {
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if inf1 && inf2 && s1 == s2 {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    if (inf1 && !s1) || (inf2 && s2) {
        return sf64_fpinfinity(false);
    }
    if (inf1 && s1) || (inf2 && !s2) {
        return sf64_fpinfinity(true);
    }
    if zero1 && zero2 && s1 == !s2 {
        return sf64_fpzero(s1);
    }
    let mut result = (v1 - v2) / 2.0;
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fpmul(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    if inf1 || inf2 {
        return sf64_fpinfinity(s1 ^ s2);
    }
    if zero1 || zero2 {
        return sf64_fpzero(s1 ^ s2);
    }
    let mut result = v1 * v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fpdiv(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (mut result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && inf2) || (zero1 && zero2) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    if inf1 || zero2 {
        if !inf1 {
            fp_process_exception_ctx(FPException::DivideByZero, fpcr);
        }
        return sf64_fpinfinity(s1 ^ s2);
    }
    if zero1 || inf2 {
        return sf64_fpzero(s1 ^ s2);
    }
    result = v1 / v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fpsqrt(value: f64) -> f64 {
    let fpcr = get_fpcr();
    let (v, typ, sign, op) = sf64_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf64_fpprocessnan(typ, op, fpcr);
    }
    if typ == FPType::Zero {
        return sf64_fpzero(sign);
    }
    if typ == FPType::Infinity && !sign {
        return sf64_fpinfinity(sign);
    }
    if sign {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    let mut result = v.sqrt();
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fpmulx(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        return sf64_fptwo(s1 ^ s2);
    }
    if inf1 || inf2 {
        return sf64_fpinfinity(s1 ^ s2);
    }
    if zero1 || zero2 {
        return sf64_fpzero(s1 ^ s2);
    }
    let mut result = v1 * v2;
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fprecipestimate(value: f64) -> f64 {
    let fpcr = get_fpcr();
    let (_v, typ, sign, op) = sf64_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf64_fpprocessnan(typ, op, fpcr);
    }
    if typ == FPType::Infinity {
        return sf64_fpzero(sign);
    }
    if typ == FPType::Zero {
        fp_process_exception_ctx(FPException::DivideByZero, fpcr);
        return sf64_fpinfinity(sign);
    }
    if value.abs() < 2.0f64.powi(-1024) {
        let overflow_to_inf = match get_rounding_mode(fpcr) {
            FPRoundingMode::ToNearest => true,
            FPRoundingMode::TowardsPlusInfinity => !sign,
            FPRoundingMode::TowardsMinusInfinity => sign,
            FPRoundingMode::TowardsZero => false,
        };
        fp_process_exception_ctx(FPException::Overflow, fpcr);
        fp_process_exception_ctx(FPException::Inexact, fpcr);
        return if overflow_to_inf {
            sf64_fpinfinity(sign)
        } else {
            sf64_fpmaxnormal(sign)
        };
    }
    if (fpcr & (1 << 24)) != 0 && value.abs() >= 2.0f64.powi(1022) {
        // Fz
        SF_FPSR.with(|f| f.set(f.get() | (1 << 3))); // UFC
        return sf64_fpzero(sign);
    }
    let mut fraction = op & 0x000FFFFFFFFFFFFFu64;
    let mut exp = ((op & 0x7FF0000000000000u64) >> 52) as u32;
    if exp == 0 {
        if (fraction & 0x0008000000000000u64) == 0 {
            fraction = (fraction & 0x0003FFFFFFFFFFFFu64) << 2;
            exp = exp.wrapping_sub(1);
        } else {
            fraction = (fraction & 0x0007FFFFFFFFFFFFu64) << 1;
        }
    }
    let scaled = ((fraction & 0x000FF00000000000u64) | 0x0010000000000000u64) >> 44;
    let result_exp = 2045u32.wrapping_sub(exp);
    let estimate = RECIP_ESTIMATE_TABLE[(scaled - 256) as usize] as u64 + 256;
    fraction = estimate << 44;
    if result_exp == 0 {
        fraction = ((fraction & 0x000FFFFFFFFFFFFEu64) | 0x0010000000000000u64) >> 1;
    } else if result_exp == 0xFFFFFFFFu32 {
        fraction = ((fraction & 0x000FFFFFFFFFFFFCu64) | 0x0010000000000000u64) >> 2;
    }
    let result_bits = ((if sign { 1u64 } else { 0 }) << 63)
        | ((result_exp as u64 & 0x7FF) << 52)
        | (fraction & 0x000FFFFFFFFFFFFFu64);
    f64::from_bits(result_bits)
}

pub extern "C" fn sf64_fprecpx(value: f64) -> f64 {
    let fpcr = get_fpcr();
    let (_v, typ, sign, op) = sf64_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf64_fpprocessnan(typ, op, fpcr);
    }
    let not_exp = (!op >> 52) & 0x7FF;
    let max_exp = 0x7FE;
    let result_exp = if not_exp == 0x7FF { max_exp } else { not_exp };
    let bits = ((if sign { 1u64 } else { 0 }) << 63) | (result_exp << 52);
    f64::from_bits(bits)
}

pub extern "C" fn sf64_fprsqtestimate(value: f64) -> f64 {
    let fpcr = get_fpcr();
    let (_v, typ, sign, op) = sf64_fpunpack(value, fpcr);
    if typ == FPType::SNaN || typ == FPType::QNaN {
        return sf64_fpprocessnan(typ, op, fpcr);
    }
    if typ == FPType::Zero {
        fp_process_exception_ctx(FPException::DivideByZero, fpcr);
        return sf64_fpinfinity(sign);
    }
    if sign {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    if typ == FPType::Infinity {
        return sf64_fpzero(false);
    }
    let mut fraction = op & 0x000FFFFFFFFFFFFFu64;
    let mut exp = ((op & 0x7FF0000000000000u64) >> 52) as u32;
    if exp == 0 {
        while (fraction & 0x0008000000000000u64) == 0 {
            fraction = (fraction & 0x0007FFFFFFFFFFFFu64) << 1;
            exp = exp.wrapping_sub(1);
        }
        fraction = (fraction & 0x0007FFFFFFFFFFFFu64) << 1;
    }
    let scaled = if (exp & 1) == 0 {
        ((fraction & 0x000FF00000000000u64) | 0x0010000000000000u64) >> 44
    } else {
        ((fraction & 0x000FE00000000000u64) | 0x0010000000000000u64) >> 45
    };
    let result_exp = (3068u32.wrapping_sub(exp)) >> 1;
    let estimate = RECIP_SQRT_ESTIMATE_TABLE[(scaled - 128) as usize] as u64 + 256;
    let bits = ((result_exp as u64 & 0x7FF) << 52) | ((estimate & 0xFF) << 44);
    f64::from_bits(bits)
}

pub extern "C" fn sf64_fprecipstep(value1: f64, value2: f64) -> f64 {
    let fpcr = STANDARD_FPCR;
    let (v1, t1, _, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, _, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    let product = if (inf1 && zero2) || (zero1 && inf2) {
        sf64_fpzero(false)
    } else {
        sf64_fpmulfpscr_impl(v1, v2, true)
    };
    sf64_fpsub_impl(sf64_fptwo(false), product, fpcr)
}

pub extern "C" fn sf64_fprsqrtstep(value1: f64, value2: f64) -> f64 {
    let fpcr = STANDARD_FPCR;
    let (v1, t1, _, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, _, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    let product = if (inf1 && zero2) || (zero1 && inf2) {
        sf64_fpzero(false)
    } else {
        sf64_fpmulfpscr_impl(v1, v2, true)
    };
    sf64_fphalvedsub_impl(sf64_fpthree(false), product, fpcr)
}

pub extern "C" fn sf64_fprecipstepfused(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let v1 = sf64_fpneg(value1);
    let (v1, t1, s1, o1) = sf64_fpunpack(v1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        return sf64_fptwo(false);
    }
    if inf1 || inf2 {
        return sf64_fpinfinity(s1 ^ s2);
    }
    let mut result = v1.mul_add(v2, 2.0);
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fprsqrtstepfused(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let v1 = sf64_fpneg(value1);
    let (v1, t1, s1, o1) = sf64_fpunpack(v1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    if (inf1 && zero2) || (zero1 && inf2) {
        return sf64_fponepointfive(false);
    }
    if inf1 || inf2 {
        return sf64_fpinfinity(s1 ^ s2);
    }
    let mut result = v1.mul_add(v2, 3.0) / 2.0;
    if (fpcr & (1 << 24)) != 0 {
        result = sf64_apply_fz(result);
    }
    result
}

pub extern "C" fn sf64_fpmax(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    if v1 > v2 {
        if t1 == FPType::Infinity {
            return sf64_fpinfinity(s1);
        }
        if t1 == FPType::Zero {
            return sf64_fpzero(s1 && s2);
        }
        let mut r = v1;
        if (fpcr & (1 << 24)) != 0 {
            r = sf64_apply_fz(r);
        }
        r
    } else {
        if t2 == FPType::Infinity {
            return sf64_fpinfinity(s2);
        }
        if t2 == FPType::Zero {
            return sf64_fpzero(s1 && s2);
        }
        let mut r = v2;
        if (fpcr & (1 << 24)) != 0 {
            r = sf64_apply_fz(r);
        }
        r
    }
}

pub extern "C" fn sf64_fpmin(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let (result, done) = sf64_fpprocessnans(t1, t2, o1, o2, fpcr);
    if done {
        return result;
    }
    if v1 < v2 {
        if t1 == FPType::Infinity {
            return sf64_fpinfinity(s1);
        }
        if t1 == FPType::Zero {
            return sf64_fpzero(s1 || s2);
        }
        let mut r = v1;
        if (fpcr & (1 << 24)) != 0 {
            r = sf64_apply_fz(r);
        }
        r
    } else {
        if t2 == FPType::Infinity {
            return sf64_fpinfinity(s2);
        }
        if t2 == FPType::Zero {
            return sf64_fpzero(s1 || s2);
        }
        let mut r = v2;
        if (fpcr & (1 << 24)) != 0 {
            r = sf64_apply_fz(r);
        }
        r
    }
}

pub extern "C" fn sf64_fpcompare(value1: f64, value2: f64, signal_nans: u8) -> i32 {
    let fpcr = get_fpcr();
    let (v1, t1, _, _) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, _, _) = sf64_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        if t1 == FPType::SNaN || t2 == FPType::SNaN || signal_nans != 0 {
            fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        }
        return 0b0011;
    }
    if v1 == v2 {
        return 0b0110;
    }
    if v1 < v2 {
        return 0b1000;
    }
    0b0010
}

pub extern "C" fn sf64_fpcompareeq(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (_, t1, _, _) = sf64_fpunpack(value1, fpcr);
    let (_, t2, _, _) = sf64_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        if t1 == FPType::SNaN || t2 == FPType::SNaN {
            fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        }
        return sf64_zerosorones(false);
    }
    sf64_zerosorones(value1 == value2)
}

pub extern "C" fn sf64_fpcomparege(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (_, t1, _, _) = sf64_fpunpack(value1, fpcr);
    let (_, t2, _, _) = sf64_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_zerosorones(false);
    }
    sf64_zerosorones(value1 >= value2)
}

pub extern "C" fn sf64_fpcomparegt(value1: f64, value2: f64) -> f64 {
    let fpcr = get_fpcr();
    let (_, t1, _, _) = sf64_fpunpack(value1, fpcr);
    let (_, t2, _, _) = sf64_fpunpack(value2, fpcr);
    if t1 == FPType::SNaN || t1 == FPType::QNaN || t2 == FPType::SNaN || t2 == FPType::QNaN {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_zerosorones(false);
    }
    sf64_zerosorones(value1 > value2)
}

pub extern "C" fn sf64_fpcomparele(value1: f64, value2: f64) -> f64 {
    sf64_fpcomparege(value2, value1)
}

pub extern "C" fn sf64_fpcomparelt(value1: f64, value2: f64) -> f64 {
    sf64_fpcomparegt(value2, value1)
}

fn sf64_fpprocessnans3(
    type1: FPType,
    type2: FPType,
    type3: FPType,
    op1: u64,
    op2: u64,
    op3: u64,
    fpcr: u32,
) -> (f64, bool) {
    if type1 == FPType::SNaN {
        (sf64_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::SNaN {
        (sf64_fpprocessnan(type2, op2, fpcr), true)
    } else if type3 == FPType::SNaN {
        (sf64_fpprocessnan(type3, op3, fpcr), true)
    } else if type1 == FPType::QNaN {
        (sf64_fpprocessnan(type1, op1, fpcr), true)
    } else if type2 == FPType::QNaN {
        (sf64_fpprocessnan(type2, op2, fpcr), true)
    } else if type3 == FPType::QNaN {
        (sf64_fpprocessnan(type3, op3, fpcr), true)
    } else {
        (0.0f64, false)
    }
}

fn sf64_fpmuladd_impl(value_a: f64, value1: f64, value2: f64, standard_fpcr: bool) -> f64 {
    let fpcr = if standard_fpcr {
        STANDARD_FPCR
    } else {
        get_fpcr()
    };
    let (va, ta, sa, oa) = sf64_fpunpack(value_a, fpcr);
    let (v1, t1, s1, o1) = sf64_fpunpack(value1, fpcr);
    let (v2, t2, s2, o2) = sf64_fpunpack(value2, fpcr);
    let inf1 = t1 == FPType::Infinity;
    let zero1 = t1 == FPType::Zero;
    let inf2 = t2 == FPType::Infinity;
    let zero2 = t2 == FPType::Zero;
    let (result, done) = sf64_fpprocessnans3(ta, t1, t2, oa, o1, o2, fpcr);
    if ta == FPType::QNaN && ((inf1 && zero2) || (zero1 && inf2)) {
        fp_process_exception_ctx(FPException::InvalidOp, fpcr);
        return sf64_fpdefaultnan();
    }
    if !done {
        let inf_a = ta == FPType::Infinity;
        let zero_a = ta == FPType::Zero;
        let sign_p = s1 ^ s2;
        let inf_p = inf1 || inf2;
        let zero_p = zero1 || zero2;
        if (inf1 && zero2) || (zero1 && inf2) || (inf_a && inf_p && sa != sign_p) {
            fp_process_exception_ctx(FPException::InvalidOp, fpcr);
            return sf64_fpdefaultnan();
        }
        if (inf_a && !sa) || (inf_p && !sign_p) {
            return sf64_fpinfinity(false);
        }
        if (inf_a && sa) || (inf_p && sign_p) {
            return sf64_fpinfinity(true);
        }
        if zero_a && zero_p && sa == sign_p {
            return sf64_fpzero(sa);
        }
        let mut rv = v1.mul_add(v2, va);
        if (fpcr & (1 << 24)) != 0 {
            rv = sf64_apply_fz(rv);
        }
        rv
    } else {
        result
    }
}

pub extern "C" fn sf64_fpmuladd(value_a: f64, value1: f64, value2: f64) -> f64 {
    sf64_fpmuladd_impl(value_a, value1, value2, false)
}

pub extern "C" fn sf64_fpmulsub(value_a: f64, value1: f64, value2: f64) -> f64 {
    sf64_fpmuladd_impl(value_a, sf64_fpneg(value1), value2, false)
}

pub extern "C" fn sf64_fpnegmuladd(value_a: f64, value1: f64, value2: f64) -> f64 {
    sf64_fpmuladd_impl(sf64_fpneg(value_a), sf64_fpneg(value1), value2, false)
}

pub extern "C" fn sf64_fpnegmulsub(value_a: f64, value1: f64, value2: f64) -> f64 {
    sf64_fpmuladd_impl(sf64_fpneg(value_a), value1, value2, false)
}

// Abs helpers used by Facge/Facgt fallback
pub extern "C" fn sf32_abs(value: f32) -> f32 {
    value.abs()
}

pub extern "C" fn sf64_abs(value: f64) -> f64 {
    value.abs()
}

// Port of Instructions/MathHelper.cs (MathHelper / MathHelperF)
// C# MidpointRounding: ToEven = 0, AwayFromZero = 1.

pub extern "C" fn mh32_ceiling(value: f32) -> f32 {
    value.ceil()
}

pub extern "C" fn mh32_floor(value: f32) -> f32 {
    value.floor()
}

pub extern "C" fn mh32_round(value: f32, mode: i32) -> f32 {
    match mode {
        0 => value.round_ties_even(),
        1 => value.round(),
        _ => panic!("mh32_round: unsupported MidpointRounding {mode}"),
    }
}

pub extern "C" fn mh32_truncate(value: f32) -> f32 {
    value.trunc()
}

pub extern "C" fn mh64_ceiling(value: f64) -> f64 {
    value.ceil()
}

pub extern "C" fn mh64_floor(value: f64) -> f64 {
    value.floor()
}

pub extern "C" fn mh64_round(value: f64, mode: i32) -> f64 {
    match mode {
        0 => value.round_ties_even(),
        1 => value.round(),
        _ => panic!("mh64_round: unsupported MidpointRounding {mode}"),
    }
}

pub extern "C" fn mh64_truncate(value: f64) -> f64 {
    value.trunc()
}
