#pragma once
#include <stdint.h>

enum ggml_msvc_simd {
    GGML_MSVC_AVX = 1u << 0,
    GGML_MSVC_AVX2 = 1u << 1,
    GGML_MSVC_FMA = 1u << 2,
    GGML_MSVC_AVX512 = 1u << 3,
    GGML_MSVC_VBMI = 1u << 4,
    GGML_MSVC_VNNI = 1u << 5,
    GGML_MSVC_BF16 = 1u << 6,
};

// CPU feature bits alone are insufficient: Windows must save the vector state.
static inline unsigned ggml_msvc_simd_features(uint32_t leaf1_ecx, uint32_t leaf7_ebx,
        uint32_t leaf7_ecx, uint32_t leaf7_1_eax, uint64_t xcr0) {
    const uint32_t avx = (1u << 26) | (1u << 27) | (1u << 28); // XSAVE, OSXSAVE, AVX
    if ((leaf1_ecx & avx) != avx || (xcr0 & 6u) != 6u) return 0;
    unsigned features = GGML_MSVC_AVX;
    if (leaf7_ebx & (1u << 5)) features |= GGML_MSVC_AVX2;
    if (leaf1_ecx & (1u << 12)) features |= GGML_MSVC_FMA;
    const uint32_t avx512 = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31); // F, DQ, BW, VL
    if ((features & (GGML_MSVC_AVX2 | GGML_MSVC_FMA)) != (GGML_MSVC_AVX2 | GGML_MSVC_FMA) ||
            (leaf7_ebx & avx512) != avx512 || (xcr0 & 0xe6u) != 0xe6u) return features;
    features |= GGML_MSVC_AVX512;
    if (leaf7_ecx & (1u << 1)) features |= GGML_MSVC_VBMI;
    if (leaf7_ecx & (1u << 11)) features |= GGML_MSVC_VNNI;
    if (leaf7_1_eax & (1u << 5)) features |= GGML_MSVC_BF16;
    return features;
}

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
static inline unsigned ggml_msvc_observed_simd(void) {
    int cpu[4];
    __cpuidex(cpu, 0, 0);
    const unsigned max_leaf = (unsigned) cpu[0];
    if (max_leaf < 1) return 0;
    __cpuidex(cpu, 1, 0);
    const uint32_t leaf1_ecx = (uint32_t) cpu[2];
    // Never execute XGETBV unless the OS has enabled XSAVE.
    const uint32_t xsave = (1u << 26) | (1u << 27);
    if ((leaf1_ecx & xsave) != xsave) return 0;
    const uint64_t xcr0 = _xgetbv(0);
    uint32_t leaf7_ebx = 0, leaf7_ecx = 0, leaf7_1_eax = 0;
    if (max_leaf >= 7) {
        __cpuidex(cpu, 7, 0);
        const unsigned max_subleaf = (unsigned) cpu[0];
        leaf7_ebx = (uint32_t) cpu[1];
        leaf7_ecx = (uint32_t) cpu[2];
        if (max_subleaf >= 1) {
            __cpuidex(cpu, 7, 1);
            leaf7_1_eax = (uint32_t) cpu[0];
        }
    }
    return ggml_msvc_simd_features(leaf1_ecx, leaf7_ebx, leaf7_ecx, leaf7_1_eax, xcr0);
}
#endif
