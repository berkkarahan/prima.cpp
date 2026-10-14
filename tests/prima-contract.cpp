#include "prima-sha256.h"
#include "prima-env.h"
#include "../src/prima-device-placement.h"
#include "../ggml/src/ggml-vulkan-dispatch.h"
#include "../ggml/cmake/msvc-native.h"
#include "../ggml/src/ggml-vulkan-timing.h"
#include <limits>
#include <iostream>
int main() {
    const auto split = prima_device_splits({3, 1});
    for (size_t i = 0; i < 8; ++i) if (prima_layer_device(split, i, 8) != (i < 6 ? 0 : 1)) return 19;
    const auto zero = prima_device_splits({0, 1, 0});
    for (size_t i = 0; i < 7; ++i) if (prima_layer_device(zero, i, 7) != 1) return 20;
    for (const auto & bad : std::vector<std::vector<double>>{{}, {0, 0}, {-1, 2}, {std::numeric_limits<double>::infinity()}, {std::numeric_limits<double>::quiet_NaN()}}) {
        try { prima_device_splits(bad); return 21; } catch (const std::invalid_argument &) {}
    }
    if (ggml_vk_timestamp_ms(100, 200, 64, 0.5) != 0.00005 ||
            ggml_vk_timestamp_ms(0xfffffff0u, 0x10u, 32, 0.25) != 0.000008 ||
            ggml_vk_timestamp_ms(UINT64_MAX - 15, 16, 64, 1.0) != 0.000032 ||
            ggml_vk_timestamp_ms(15, 0, 4, 1000000.0) != 1.0) return 16;
    for (unsigned bits : {0u, 65u}) {
        try { ggml_vk_timestamp_ms(0, 1, bits, 1.0); return 17; } catch (const std::invalid_argument &) {}
    }
    for (double period : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
        try { ggml_vk_timestamp_ms(0, 1, 64, period); return 18; } catch (const std::invalid_argument &) {}
    }
    const uint32_t avx = (1u << 26) | (1u << 27) | (1u << 28) | (1u << 12);
    const uint32_t avx512 = (1u << 5) | (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    const uint32_t extensions = (1u << 1) | (1u << 11);
    const unsigned base = GGML_MSVC_AVX | GGML_MSVC_AVX2 | GGML_MSVC_FMA;
    const unsigned all = base | GGML_MSVC_AVX512 | GGML_MSVC_VBMI | GGML_MSVC_VNNI | GGML_MSVC_BF16;
    if (ggml_msvc_simd_features(avx & ~(1u << 12), avx512, extensions, 1u << 5, 0xe6) !=
            (GGML_MSVC_AVX | GGML_MSVC_AVX2) ||
            ggml_msvc_simd_features(avx, avx512 & ~(1u << 5), extensions, 1u << 5, 0xe6) !=
            (GGML_MSVC_AVX | GGML_MSVC_FMA)) return 14;
    if (ggml_msvc_simd_features(avx, avx512, extensions, 1u << 5, 0xe6) != all ||
            ggml_msvc_simd_features(avx, avx512, extensions, 1u << 5, 6) != base ||
            ggml_msvc_simd_features(avx & ~(1u << 27), avx512, extensions, 1u << 5, 0xe6) != 0 ||
            ggml_msvc_simd_features(avx & ~(1u << 26), avx512, extensions, 1u << 5, 0xe6) != 0 ||
            ggml_msvc_simd_features(avx & ~(1u << 28), avx512, extensions, 1u << 5, 0xe6) != 0 ||
            ggml_msvc_simd_features(avx, avx512, extensions, 1u << 5, 2) != 0 ||
            ggml_msvc_simd_features(avx, avx512 & ~(1u << 31), extensions, 1u << 5, 0xe6) != base ||
            ggml_msvc_simd_features(avx, avx512, 0, 0, 0xe6) != (base | GGML_MSVC_AVX512)) return 10;
    for (const unsigned bit : {GGML_MSVC_VBMI, GGML_MSVC_VNNI, GGML_MSVC_BF16}) {
        const auto ecx = bit == GGML_MSVC_VBMI ? 1u << 1 : bit == GGML_MSVC_VNNI ? 1u << 11 : 0;
        const auto eax = bit == GGML_MSVC_BF16 ? 1u << 5 : 0;
        if (ggml_msvc_simd_features(avx, avx512, ecx, eax, 0xe6) != (base | GGML_MSVC_AVX512 | bit)) return 11;
    }
    for (const unsigned bit : {16u, 17u, 30u, 31u}) {
        if (ggml_msvc_simd_features(avx, avx512 & ~(1u << bit), extensions, 1u << 5, 0xe6) != base) return 12;
    }
    for (const unsigned bit : {1u, 2u, 5u, 6u, 7u}) {
        const auto expected = bit < 3 ? 0u : base;
        if (ggml_msvc_simd_features(avx, avx512, extensions, 1u << 5, 0xe6 & ~(1u << bit)) != expected) return 13;
    }
    std::cout << "MSVC SIMD CPU/OS fixture vectors passed\n";
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    std::cout << "MSVC observed SIMD features=" << ggml_msvc_observed_simd() << "\n";
#endif
    for (bool quant : {false, true}) {
        for (uint32_t limit : {16384u, 32768u, 65536u}) {
            for (uint32_t desired : {0u, 1u, 2u}) {
                const uint32_t tile = ggml_vk_f32_tile(limit, quant, desired);
                const uint32_t width = 32u << tile;
                const uint32_t depth = quant ? 32 : 16;
                if (tile > desired || 2*width*(depth+1)*sizeof(float) > limit) return 7;
                if (tile < desired && 4*width*(depth+1)*sizeof(float) <= limit) return 8;
            }
        }
    }
    for (const auto invalid : {std::array<uint32_t, 2>{0, 0}, {4096, 2}, {65536, 3}}) {
        try { ggml_vk_f32_tile(invalid[0], false, invalid[1]); return 9; }
        catch (const std::invalid_argument &) {}
    }
    for (uint32_t rows_per_group : {1u, 4u}) {
        for (uint32_t max_x : {64u, 65535u}) {
            for (uint64_t rows : {1ull, 3ull, 4ull, 7ull, 8ull, 65535ull, 65536ull, 65537ull, 262145ull}) {
                const auto grid = ggml_vk_vec_grid(rows, rows_per_group, max_x, 65535);
                const uint64_t coverage = (uint64_t)grid[0]*grid[1]*rows_per_group;
                if (!grid[0] || grid[0] > max_x || grid[1] > 65535 || coverage < rows) return 5;
            }
        }
    }
    for (const auto invalid : {std::array<uint64_t, 4>{0, 1, 65535, 65535},
                              std::array<uint64_t, 4>{1, 0, 65535, 65535},
                              std::array<uint64_t, 4>{1, 1, 0, 65535},
                              std::array<uint64_t, 4>{1, 1, 65535, 0},
                              std::array<uint64_t, 4>{65537, 1, 65535, 1},
                              std::array<uint64_t, 4>{std::numeric_limits<uint64_t>::max(), 1, 65535, 65535}}) {
        try {
            ggml_vk_vec_grid(invalid[0], (uint32_t)invalid[1], (uint32_t)invalid[2], (uint32_t)invalid[3]);
            return 6;
        } catch (const std::invalid_argument &) {
        }
    }
    if (prima_flag_enabled(nullptr) || prima_flag_enabled("") || prima_flag_enabled("0") ||
        prima_flag_enabled("false") || !prima_flag_enabled("1")) return 4;
    if (prima_sha256("") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") return 1;
    if (prima_sha256("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return 2;
    if (prima_sha256(std::string(1000000, 'a')) != "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") return 3;
    std::cout << "SHA-256 known vectors passed\n";
    return 0;
}
