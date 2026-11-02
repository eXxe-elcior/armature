// Port of Optimizations.cs for x86-64 hosts.
//
// C# exposes user-settable `*IfAvailable` flags ANDed with
// HardwareCapabilities. Hardware detection here uses
// std::arch::is_x86_feature_detected!, cached on first use.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

// User-settable flags (C# defaults)
pub static FAST_FP: AtomicBool = AtomicBool::new(true);
pub static ALLOW_LCQ_IN_FUNCTION_TABLE: AtomicBool = AtomicBool::new(true);
pub static USE_UNMANAGED_DISPATCH_LOOP: AtomicBool = AtomicBool::new(true);
pub static ENABLE_DEBUGGING: AtomicBool = AtomicBool::new(false);

pub static USE_SSE_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_SSE2_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_SSE3_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_SSSE3_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_SSE41_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_SSE42_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_POPCNT_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
// ABM/LZCNT  not a C# flag (ARMeilleure always uses BSR); Armature uses
// LZCNT for CLZ when present, BSR fallback otherwise.
pub static USE_LZCNT_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
// BMI1 (ANDN) / BMI2 (SHLX/SHRX/SARX)  Armature divergences, gated like LZCNT.
pub static USE_BMI1_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_BMI2_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_AVX_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_AVX512F_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_AVX512VL_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_AVX512BW_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_AVX512DQ_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_F16C_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_FMA_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_AESNI_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_PCLMULQDQ_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_SHA_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);
pub static USE_GFNI_IF_AVAILABLE: AtomicBool = AtomicBool::new(true);

pub static FORCE_LEGACY_SSE: AtomicBool = AtomicBool::new(false);

// Hardware capability cache
struct HardwareCapabilities {
    sse: bool,
    sse2: bool,
    sse3: bool,
    ssse3: bool,
    sse41: bool,
    sse42: bool,
    popcnt: bool,
    lzcnt: bool,
    bmi1: bool,
    bmi2: bool,
    avx: bool,
    avx2: bool,
    avx512f: bool,
    avx512vl: bool,
    avx512bw: bool,
    avx512dq: bool,
    f16c: bool,
    fma: bool,
    aesni: bool,
    pclmulqdq: bool,
    sha: bool,
    gfni: bool,
}

fn capabilities() -> &'static HardwareCapabilities {
    static CAPS: OnceLock<HardwareCapabilities> = OnceLock::new();
    CAPS.get_or_init(|| HardwareCapabilities {
        sse: is_x86_feature_detected!("sse"),
        sse2: is_x86_feature_detected!("sse2"),
        sse3: is_x86_feature_detected!("sse3"),
        ssse3: is_x86_feature_detected!("ssse3"),
        sse41: is_x86_feature_detected!("sse4.1"),
        sse42: is_x86_feature_detected!("sse4.2"),
        popcnt: is_x86_feature_detected!("popcnt"),
        lzcnt: is_x86_feature_detected!("lzcnt"),
        bmi1: is_x86_feature_detected!("bmi1"),
        bmi2: is_x86_feature_detected!("bmi2"),
        avx: is_x86_feature_detected!("avx"),
        avx2: is_x86_feature_detected!("avx2"),
        avx512f: is_x86_feature_detected!("avx512f"),
        avx512vl: is_x86_feature_detected!("avx512vl"),
        avx512bw: is_x86_feature_detected!("avx512bw"),
        avx512dq: is_x86_feature_detected!("avx512dq"),
        f16c: is_x86_feature_detected!("f16c"),
        fma: is_x86_feature_detected!("fma"),
        aesni: is_x86_feature_detected!("aes"),
        pclmulqdq: is_x86_feature_detected!("pclmulqdq"),
        sha: is_x86_feature_detected!("sha"),
        gfni: is_x86_feature_detected!("gfni"),
    })
}

// Effective flags (C# internal Use* properties)
#[inline]
fn flag(b: &AtomicBool) -> bool {
    b.load(Ordering::Relaxed)
}

pub fn fast_fp() -> bool {
    flag(&FAST_FP)
}
pub fn enable_debugging() -> bool {
    flag(&ENABLE_DEBUGGING)
}
pub fn force_legacy_sse() -> bool {
    flag(&FORCE_LEGACY_SSE)
}

pub fn use_sse() -> bool {
    flag(&USE_SSE_IF_AVAILABLE) && capabilities().sse
}
pub fn use_sse2() -> bool {
    flag(&USE_SSE2_IF_AVAILABLE) && capabilities().sse2
}
pub fn use_sse3() -> bool {
    flag(&USE_SSE3_IF_AVAILABLE) && capabilities().sse3
}
pub fn use_ssse3() -> bool {
    flag(&USE_SSSE3_IF_AVAILABLE) && capabilities().ssse3
}
pub fn use_sse41() -> bool {
    flag(&USE_SSE41_IF_AVAILABLE) && capabilities().sse41
}
pub fn use_sse42() -> bool {
    flag(&USE_SSE42_IF_AVAILABLE) && capabilities().sse42
}
pub fn use_popcnt() -> bool {
    flag(&USE_POPCNT_IF_AVAILABLE) && capabilities().popcnt
}
pub fn use_lzcnt() -> bool {
    flag(&USE_LZCNT_IF_AVAILABLE) && capabilities().lzcnt
}
pub fn use_bmi1() -> bool {
    flag(&USE_BMI1_IF_AVAILABLE) && capabilities().bmi1
}
pub fn use_bmi2() -> bool {
    flag(&USE_BMI2_IF_AVAILABLE) && capabilities().bmi2
}
pub fn use_avx() -> bool {
    flag(&USE_AVX_IF_AVAILABLE) && capabilities().avx && !force_legacy_sse()
}
// C# HardwareCapabilities.SupportsVexEncoding = SupportsAvx && !ForceLegacySse
// (raw CPUID, not gated by the UseAvxIfAvailable knob).
pub fn supports_vex_encoding() -> bool {
    capabilities().avx && !force_legacy_sse()
}
// AVX2 (per-lane variable shifts etc). Implies AVX; gated by the same AVX knob
// + legacy-SSE override. No C# counterpart  ARMeilleure does not use AVX2.
pub fn use_avx2() -> bool {
    use_avx() && capabilities().avx2
}
// C# HardwareCapabilities.SupportsSse41 (raw CPUID).
pub fn supports_sse41() -> bool {
    capabilities().sse41
}
pub fn use_avx512f() -> bool {
    flag(&USE_AVX512F_IF_AVAILABLE) && capabilities().avx512f && !force_legacy_sse()
}
pub fn use_avx512vl() -> bool {
    flag(&USE_AVX512VL_IF_AVAILABLE) && capabilities().avx512vl && !force_legacy_sse()
}
pub fn use_avx512bw() -> bool {
    flag(&USE_AVX512BW_IF_AVAILABLE) && capabilities().avx512bw && !force_legacy_sse()
}
pub fn use_avx512dq() -> bool {
    flag(&USE_AVX512DQ_IF_AVAILABLE) && capabilities().avx512dq && !force_legacy_sse()
}
pub fn use_f16c() -> bool {
    flag(&USE_F16C_IF_AVAILABLE) && capabilities().f16c
}
pub fn use_fma() -> bool {
    flag(&USE_FMA_IF_AVAILABLE) && capabilities().fma
}
pub fn use_aesni() -> bool {
    flag(&USE_AESNI_IF_AVAILABLE) && capabilities().aesni
}
pub fn use_pclmulqdq() -> bool {
    flag(&USE_PCLMULQDQ_IF_AVAILABLE) && capabilities().pclmulqdq
}
pub fn use_sha() -> bool {
    flag(&USE_SHA_IF_AVAILABLE) && capabilities().sha
}
pub fn use_gfni() -> bool {
    flag(&USE_GFNI_IF_AVAILABLE) && capabilities().gfni
}

pub fn use_avx512_ortho() -> bool {
    use_avx512f() && use_avx512vl()
}
pub fn use_avx512_ortho_float() -> bool {
    use_avx512_ortho() && use_avx512dq()
}
