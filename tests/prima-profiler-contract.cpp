#include "profiler.h"
#include "prima-profile-io.h"
#include <cmath>
#include <cstring>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--hold-io-file") == 0) {
        prima_profile_file file;
        std::cout << "ready\n" << std::flush;
        std::cin.get();
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--io") == 0) {
        float read = 0, write = 0;
        for (bool random : {false, true}) {
            prima_profile_disk(&read, &write, random);
            if (!std::isfinite(read) || !std::isfinite(write) || read <= 0 || write <= 0) return 3;
            std::cout << (random ? "random" : "sequential") << " read=" << read << " write=" << write << '\n';
        }
        prima_profile_file file;
        try { file.transfer(true, 1, 4096); return 4; }
        catch (const std::invalid_argument &) {}
        return 0;
    }
    const auto backend = argc == 2 && std::strcmp(argv[1], "--vulkan") == 0
        ? PROFILER_BACKEND_TYPE_VULKAN : PROFILER_BACKEND_TYPE_CPU;
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--vulkan") != 0)) return 2;
    int failures = 0;
    // Qwen3's Q projection width is heads*head_dim, not embedding width.
    // GQA reads one KV set; expert arithmetic uses active experts, storage uses all.
    auto * context = ggml_init({10 * ggml_tensor_overhead(), nullptr, true});
    prima_layer_shape shape = {16, 8, 128, 128, 512, 128, 8, 1024};
    prima_tensor_cost query, key, norm, expert;
    prima_cost_layer_tensor(query, "attn_q.weight", ggml_new_tensor_2d(context, GGML_TYPE_F32, 1024, 2048), shape);
    if (query.flops[GGML_TYPE_F32] != 4199424) ++failures;
    prima_cost_layer_tensor(key, "attn_k.weight", ggml_new_tensor_2d(context, GGML_TYPE_F32, 1024, 1024), shape);
    if (key.flops[GGML_TYPE_F16] != 2097152 || key.kv_bytes != 1048576) ++failures;
    prima_cost_layer_tensor(norm, "attn_q_norm.weight", ggml_new_tensor_1d(context, GGML_TYPE_F32, 128), shape);
    if (norm.flops[GGML_TYPE_F32] != 10256) ++failures;
    const auto * experts = ggml_new_tensor_3d(context, GGML_TYPE_Q4_K, 1024, 3072, 128);
    prima_cost_layer_tensor(expert, "ffn_gate_exps.weight", experts, shape);
    if (expert.flops[GGML_TYPE_Q4_K] != 50331648 || expert.active_bytes * 16 != expert.weights_bytes
        || expert.weights_bytes != ggml_nbytes(experts)) ++failures;
    for (const auto & role : {"unknown.weight", "attn_q_norm.weight"}) {
        try {
            prima_tensor_cost invalid;
            prima_cost_layer_tensor(invalid, role, ggml_new_tensor_1d(context, GGML_TYPE_F32, 256), shape);
            ++failures;
        } catch (const std::invalid_argument &) {}
    }
    ggml_free(context);
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
    for (int width : {0, -1, 1048577}) {
        try { device_host_transfer(width, backend); ++failures; }
        catch (const std::invalid_argument &) {}
    }
    if (backend == PROFILER_BACKEND_TYPE_CPU) {
        try { device_host_transfer(1024, backend); ++failures; }
        catch (const std::invalid_argument &) {}
    } else {
        for (int width : {1, 1024, 2048, 4097}) {
            const auto handoff = device_host_transfer(width, backend);
            if (handoff.bytes != uint64_t(width) * sizeof(float) || handoff.upload_ms.size() != 5
                || handoff.download_ms.size() != 5) ++failures;
            for (auto value : handoff.upload_ms) if (!std::isfinite(value) || value <= 0) ++failures;
            for (auto value : handoff.download_ms) if (!std::isfinite(value) || value <= 0) ++failures;
            std::cout << "verified_host_transfer_bytes=" << handoff.bytes << '\n';
        }
    }
    if (backend == PROFILER_BACKEND_TYPE_VULKAN && device_vulkan_read_vram_bw() <= 0) ++failures;
    return failures ? 1 : 0;
}
