#include "profiler.h"
#include <cmath>
#include <cstring>
#include <iostream>

int main(int argc, char ** argv) {
    const auto backend = argc == 2 && std::strcmp(argv[1], "--vulkan") == 0
        ? PROFILER_BACKEND_TYPE_VULKAN : PROFILER_BACKEND_TYPE_CPU;
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--vulkan") != 0)) return 2;
    int failures = 0;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K,
                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q5_0,
                     GGML_TYPE_Q8_0, GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ1_M}) {
        const float gflops = device_matrix_flops(1024, type, GGML_TYPE_F32, backend, 2);
        // This backend has no matrix kernels for these IQ formats. Returning zero
        // is refusal, not CPU substitution or a claim of Vulkan format support.
        const bool unavailable = backend == PROFILER_BACKEND_TYPE_VULKAN
            && (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M);
        if (unavailable) {
            std::cout << ggml_type_name(type) << " unavailable_vulkan=1\n";
            if (gflops != 0) ++failures;
            continue;
        }
        std::cout << ggml_type_name(type) << " verified_gflops=" << gflops << '\n';
        if (!std::isfinite(gflops) || gflops <= 0) ++failures;
    }
    if (device_matrix_flops(0, GGML_TYPE_F32, GGML_TYPE_F32, backend, 2) != 0
        || device_matrix_flops(1024, GGML_TYPE_F32, GGML_TYPE_F32, backend, 0) != 0) ++failures;
    return failures ? 1 : 0;
}
