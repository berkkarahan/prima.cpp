include(CheckCSourceRuns)

if (CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "MSVC GGML_NATIVE requires a runnable target; use GGML_NATIVE=OFF and explicit target ISA flags when cross-compiling")
endif()

file(READ "${CMAKE_CURRENT_LIST_DIR}/msvc-native.h" GGML_MSVC_SIMD_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/msvc-native.h")

# CPU/OS support and compiler intrinsics must both pass. The old unused vectors
# could be optimized away, so compile success alone cannot select a native ISA.
function(ggml_msvc_check_native option feature flags probe)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${flags}")
    set(source "${GGML_MSVC_SIMD_SOURCE}\n#include <immintrin.h>\n${probe}\n
        int main(void) {
            if (!(ggml_msvc_observed_simd() & ${feature})) return 1;
            volatile int seed = 3;
            return ggml_probe(seed);
        }")
    # Native capability observations must be fresh when reconfiguring on a
    # different host or after its OS vector-state policy changes.
    unset(GGML_MSVC_NATIVE_${option} CACHE)
    check_c_source_runs("${source}" GGML_MSVC_NATIVE_${option})
    set(${option} ${GGML_MSVC_NATIVE_${option}} PARENT_SCOPE)
endfunction()

ggml_msvc_check_native(GGML_AVX GGML_MSVC_AVX /arch:AVX "
    __declspec(noinline) int ggml_probe(int n) {
        float result[8];
        _mm256_storeu_ps(result, _mm256_add_ps(_mm256_set1_ps((float)n), _mm256_set1_ps((float)n)));
        return result[0] != 6.0f;
    }")
ggml_msvc_check_native(GGML_AVX2 GGML_MSVC_AVX2 /arch:AVX2 "
    __declspec(noinline) int ggml_probe(int n) {
        const __m256i result = _mm256_abs_epi16(_mm256_set1_epi16((short)-n));
        return _mm256_extract_epi16(result, 0) != 3;
    }")
ggml_msvc_check_native(GGML_FMA GGML_MSVC_FMA /arch:AVX2 "
    __declspec(noinline) int ggml_probe(int n) {
        float result[8];
        const __m256 a = _mm256_set1_ps((float)n);
        _mm256_storeu_ps(result, _mm256_fmadd_ps(a, a, a));
        return result[0] != 12.0f;
    }")
if (NOT GGML_FMA)
    # /arch:AVX2 lets MSVC emit FMA even outside explicit intrinsics.
    set(GGML_AVX2 OFF)
endif()
ggml_msvc_check_native(GGML_AVX512 GGML_MSVC_AVX512 /arch:AVX512 "
    __declspec(noinline) int ggml_probe(int n) {
        const __m512i a = _mm512_set1_epi8((char)n);
        return _mm512_cmp_epi8_mask(a, _mm512_add_epi8(a, a), _MM_CMPINT_EQ) != 0;
    }")

if (GGML_AVX512)
    ggml_msvc_check_native(GGML_AVX512_VBMI GGML_MSVC_VBMI /arch:AVX512 "
        __declspec(noinline) int ggml_probe(int n) {
            const __m512i a = _mm512_set1_epi8((char)n);
            const __m512i result = _mm512_permutexvar_epi8(a, a);
            return _mm_cvtsi128_si32(_mm512_castsi512_si128(result)) != 0x03030303;
        }")
    ggml_msvc_check_native(GGML_AVX512_VNNI GGML_MSVC_VNNI /arch:AVX512 "
        __declspec(noinline) int ggml_probe(int n) {
            const __m256i a = _mm256_set1_epi8((char)n);
            const __m256i result = _mm256_dpbusd_epi32(_mm256_setzero_si256(), a, a);
            return _mm256_extract_epi32(result, 0) != 36;
        }")
    ggml_msvc_check_native(GGML_AVX512_BF16 GGML_MSVC_BF16 /arch:AVX512 "
        __declspec(noinline) int ggml_probe(int n) {
            float result[16];
            const __m512 f = _mm512_set1_ps((float)n);
            const __m512bh a = _mm512_cvtne2ps_pbh(f, f);
            _mm512_storeu_ps(result, _mm512_dpbf16_ps(_mm512_setzero_ps(), a, a));
            return result[0] != 18.0f;
        }")
else()
    set(GGML_AVX512_VBMI OFF)
    set(GGML_AVX512_VNNI OFF)
    set(GGML_AVX512_BF16 OFF)
endif()
