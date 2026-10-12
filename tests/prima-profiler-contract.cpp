#include "profiler.h"
#include "prima-profile-io.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../src/llama-moe.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static bool expert_reduction(ggml_backend_t backend, int embedding, int used, int tokens) {
    auto * context = ggml_init({64 * ggml_tensor_overhead() + 3 * ggml_graph_overhead(), nullptr, true});
    auto * storage = ggml_new_tensor_1d(context, GGML_TYPE_F32, embedding * used * tokens + 1);
    auto * experts = ggml_view_3d(context, storage, embedding, used, tokens,
        embedding * sizeof(float), embedding * used * sizeof(float), sizeof(float));
    auto * output = llama_sum_experts(context, experts, !ggml_backend_is_cpu(backend), !ggml_backend_is_cpu(backend));
    auto * packed_output = embedding == 2048 && used == 8 && tokens == 2
        ? llama_sum_experts(context, experts, true) : nullptr;
    // Allocate a following tensor in the same buffer. A rounded row dispatch
    // must not overwrite it when embedding*tokens is not a multiple of512.
    auto * guard = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1024);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_cgraph * serial_graph = nullptr;
    ggml_cgraph * packed_graph = graph;
    if (embedding == 2048 && used == 8 && tokens == 2) {
        auto * serial = ggml_view_2d(context, experts, embedding, tokens, experts->nb[2], 0);
        for (int i = 1; i < used; ++i) {
            auto * next = ggml_view_2d(context, experts, embedding, tokens, experts->nb[2], i * experts->nb[1]);
            serial = ggml_add(context, serial, next);
        }
        serial_graph = ggml_new_graph(context);
        ggml_build_forward_expand(serial_graph, serial);
        packed_graph = ggml_new_graph(context);
        ggml_build_forward_expand(packed_graph, packed_output);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    const auto release = [&]() { ggml_backend_buffer_free(buffer); ggml_free(context); };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) { release(); return false; }
    }
    if (!buffer) { release(); return false; }
    std::vector<float> input(embedding * used * tokens), actual(embedding * tokens);
    // Dyadic signed inputs make both serial and tree sums exactly representable;
    // nonuniform token/embedding/expert values expose axis or offset confusion.
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 13 + i / embedding * 7) % 37) - 18) / 32;
    if (!ggml_backend_is_cpu(backend) && used >= 4) {
        // Serial addition is observable for cancellation-sensitive values.
        // A parallel/tree reduction would return a different result here.
        for (int token = 0; token < tokens; ++token) {
            for (int expert = 0; expert < used; ++expert) {
                input[(token * used + expert) * embedding] = expert % 4 == 0 ? 1e8f :
                    expert % 4 == 2 ? -1e8f : 1.0f;
            }
        }
    }
    ggml_backend_tensor_set(experts, input.data(), 0, input.size() * sizeof(float));
    std::vector<float> sentinel(1024, 12345), after(1024);
    ggml_backend_tensor_set(guard, sentinel.data(), 0, sentinel.size() * sizeof(float));
    bool valid = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_tensor_get(guard, after.data(), 0, after.size() * sizeof(float));
    valid = valid && after == sentinel;
    valid = valid && ggml_is_contiguous(output) && output->ne[0] == embedding && output->ne[1] == tokens;
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < embedding; ++row) {
            float expected = 0;
            for (int expert = 0; expert < used; ++expert) expected += input[(token * used + expert) * embedding + row];
            if (actual[token * embedding + row] != expected) valid = false;
        }
    }
    std::cout << "expert_reduction embedding=" << embedding << " experts=" << used << " tokens=" << tokens
              << " verified=" << valid << '\n';
    if (serial_graph && valid) {
        std::vector<double> serial_ms, packed_ms, selected_ms;
        for (int sample = 0; sample < 7; ++sample) {
            for (int engine = 0; engine < 3; ++engine) {
                const int method = (sample + engine) % 3;
                const auto begin = std::chrono::steady_clock::now();
                valid = valid && ggml_backend_graph_compute(backend,
                    method == 0 ? serial_graph : method == 1 ? packed_graph : graph) == GGML_STATUS_SUCCESS;
                ggml_backend_synchronize(backend);
                const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                if (!std::isfinite(elapsed) || elapsed <= 0) valid = false;
                if (sample > 1) (method == 0 ? serial_ms : method == 1 ? packed_ms : selected_ms).push_back(elapsed);
            }
        }
        std::sort(serial_ms.begin(), serial_ms.end());
        std::sort(packed_ms.begin(), packed_ms.end());
        std::sort(selected_ms.begin(), selected_ms.end());
        std::cout << "expert_reduction_microbench serial_ms=" << serial_ms[2]
                  << " packed_ms=" << packed_ms[2] << " selected_ms=" << selected_ms[2] << " rounds=5\n";
        std::vector<float> packed_actual(actual.size());
        if (ggml_backend_graph_compute(backend, packed_graph) != GGML_STATUS_SUCCESS) valid = false;
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(packed_output, packed_actual.data(), 0, packed_actual.size() * sizeof(float));
        std::cout << "expert_reduction_rounding selected=" << actual[0] << " packed=" << packed_actual[0] << '\n';
    }
    release();
    return valid;
}

static bool indexed_vectors(ggml_backend_t backend, ggml_type type, int tokens, int used, bool down) {
    // Both expert input projections (one column) and down projections (one
    // column per selected expert), with distinct tokens and padded ID rows.
    constexpr int width = 256, rows = 35, experts = 16;
    const int columns = down ? used : 1;
    auto * context = ggml_init({12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto * weights = ggml_new_tensor_3d(context, type, width, rows, experts);
    auto * inputs = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, columns, tokens);
    auto * id_storage = ggml_new_tensor_2d(context, GGML_TYPE_I32, used + 1, tokens);
    auto * ids = ggml_view_2d(context, id_storage, used, tokens, id_storage->nb[1], 0);
    auto * output = ggml_mul_mat_id(context, weights, inputs, ids);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto * allocation = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const auto release = [&]() { ggml_gallocr_free(allocation); ggml_free(context); };
    if (!ggml_backend_supports_op(backend, output) || !ggml_gallocr_alloc_graph(allocation, graph)) {
        release();
        return false;
    }
    std::vector<float> source(width * rows * experts), decoded(source.size()), input(width * columns * tokens);
    for (size_t i = 0; i < source.size(); ++i) source[i] = float(int((i * 13 + i / width * 7) % 37) - 18) / 32;
    // Integer inputs are exactly representable by the CPU's internal Q8 dot
    // operand too, keeping the reference independent of input re-quantization.
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 17 + i / width * 3) % 3) - 1);
    std::vector<uint8_t> encoded(ggml_nbytes(weights));
    std::vector<float> importance(width, 1);
    ggml_quantize_chunk(type, source.data(), encoded.data(), 0, rows * experts, width, importance.data());
    const auto row_bytes = ggml_row_size(type, width);
    for (int row = 0; row < rows * experts; ++row) {
        if (type == GGML_TYPE_F32) std::memcpy(decoded.data() + row * width, encoded.data() + row * row_bytes, width * sizeof(float));
        else ggml_internal_get_type_traits(type).to_float(encoded.data() + row * row_bytes, decoded.data() + row * width, width);
    }
    std::vector<int32_t> indices((used + 1) * tokens, -1);
    for (int token = 0; token < tokens; ++token)
        for (int expert = 0; expert < used; ++expert) indices[token * (used + 1) + expert] = (token * 3 + expert * 5) % experts;
    ggml_backend_tensor_set(weights, encoded.data(), 0, encoded.size());
    ggml_backend_tensor_set(inputs, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(id_storage, indices.data(), 0, indices.size() * sizeof(int32_t));
    const auto begin = std::chrono::steady_clock::now();
    bool valid = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    std::vector<float> actual(rows * used * tokens);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    double worst_error = 0;
    for (int token = 0; token < tokens; ++token) {
        for (int expert = 0; expert < used; ++expert) {
            const int selected = indices[token * (used + 1) + expert];
            for (int row = 0; row < rows; ++row) {
                double expected = 0, magnitude = 0;
                for (int col = 0; col < width; ++col) {
                    const double term = double(decoded[(selected * rows + row) * width + col]) * input[(token * columns + expert % columns) * width + col];
                    expected += term;
                    magnitude += std::abs(term);
                }
                const float value = actual[(token * used + expert) * rows + row];
                worst_error = std::max(worst_error, std::abs(value - expected));
                // The existing general Vulkan quantized matrix shader decodes
                // weights through F16. Bound that rounding by the dot's L1
                // magnitude; the new small-batch vectors retain the stricter
                // F32 bound. Cancellation must not hide incorrect indexing.
                const double half_bound = !ggml_backend_is_cpu(backend) && tokens > 8 && ggml_is_quantized(type)
                    ? std::ldexp(magnitude, -10) : 0;
                if (!std::isfinite(value) || std::abs(value - expected) >
                    0.002 * std::max(1.0, std::abs(expected)) + half_bound) valid = false;
            }
        }
    }
    std::cout << "indexed_vectors type=" << ggml_type_name(type) << " tokens=" << tokens
              << " experts=" << used << " down=" << down << " elapsed_ms=" << elapsed
              << " worst_error=" << worst_error << " verified=" << valid << '\n';
    release();
    return valid;
}

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
    ggml_backend_t indexed_backend = nullptr;
    if (backend == PROFILER_BACKEND_TYPE_CPU) indexed_backend = ggml_backend_cpu_init();
#ifdef GGML_USE_VULKAN
    else indexed_backend = ggml_backend_vk_init(0);
#endif
    if (!indexed_backend) return 5;
    if (backend == PROFILER_BACKEND_TYPE_CPU) ggml_backend_cpu_set_n_threads(indexed_backend, 2);
    for (int embedding : {35, 513, 2048})
        for (int used : {1, 2, 3, 4, 8, 16})
            for (int tokens : {1, 2, 9})
                if (!expert_reduction(indexed_backend, embedding, used, tokens)) ++failures;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                     GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K,
                     GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL})
        for (int tokens : {1, 2, 8, 9})
            for (int used : {1, 8})
                for (bool down : {false, true})
                    if (!indexed_vectors(indexed_backend, type, tokens, used, down)) ++failures;
    ggml_backend_free(indexed_backend);
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
        for (int width : {1, 1024, 2048, 4097, 131073, 151936}) {
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
