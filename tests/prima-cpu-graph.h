#pragma once
#include "ggml.h"
#include <chrono>
#include <iostream>
#include <vector>

static bool prima_cpu_graph_contract(int threads) {
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, false});
    auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 35, 3);
    auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 35, 3);
    auto * first = ggml_add(ctx, a, b);
    auto * value = first;
    for (int stage = 0; stage < 7; ++stage) {
        for (int i = 0; i < 8; ++i) {
            value = ggml_reshape_2d(ctx, value, 35, 3);
            value = ggml_view_2d(ctx, value, 35, 3, value->nb[1], 0);
            value = ggml_permute(ctx, value, 1, 0, 2, 3);
            value = ggml_transpose(ctx, value);
        }
        value = ggml_add(ctx, value, b);
    }
    auto * output = ggml_mul(ctx, value, b);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto params = ggml_threadpool_params_default(threads);
    auto * pool = ggml_threadpool_new(&params);
    auto plan = ggml_graph_plan(graph, threads, pool);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    auto * input = static_cast<float *>(a->data);
    auto * other = static_cast<float *>(b->data);
    auto * result = static_cast<float *>(output->data);
    bool passed = true;
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 64; ++repeat) {
        for (int i = 0; i < 105; ++i) {
            input[i] = float(repeat + i % 13 - 6);
            other[i] = 2.0f;
        }
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) passed = false;
        for (int i = 0; i < 105; ++i) if (result[i] != (input[i] + 16.0f) * 2.0f) passed = false;
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    auto * first_result = static_cast<float *>(first->data);
    bool clean = true;
    for (int cancellation = 0; cancellation < 32; ++cancellation) {
        for (int i = 0; i < 105; ++i) first_result[i] = result[i] = -999.0f;
        plan.abort_callback = [](void *) { return true; };
        const bool cancelled = ggml_graph_compute(graph, &plan) == GGML_STATUS_ABORTED;
        for (int i = 0; i < 105; ++i) if (first_result[i] != -999.0f || result[i] != -999.0f) clean = false;
        if (!cancelled || !clean) passed = false;

        // Cancel after observing real work; dependent output must remain untouched.
        plan.abort_callback = [](void * data) { return *static_cast<float *>(data) != -999.0f; };
        plan.abort_callback_data = first_result;
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_ABORTED) passed = false;
        for (int i = 0; i < 105; ++i) {
            if (first_result[i] != input[i] + 2.0f || result[i] != -999.0f) passed = false;
        }
        plan.abort_callback = nullptr;
        plan.abort_callback_data = nullptr;
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) passed = false;
        for (int i = 0; i < 105; ++i) if (result[i] != (input[i] + 16.0f) * 2.0f) passed = false;
    }

    // Metadata-only graphs still settle all workers and respect prior cancellation.
    auto * history = ggml_view_2d(ctx, a, 35, 3, a->nb[1], 0);
    for (int i = 0; i < 32; ++i) history = ggml_reshape_2d(ctx, history, 35, 3);
    auto * metadata = ggml_new_graph(ctx);
    ggml_build_forward_expand(metadata, history);
    auto metadata_plan = ggml_graph_plan(metadata, threads, pool);
    metadata_plan.n_threads = threads;
    if (ggml_graph_compute(metadata, &metadata_plan) != GGML_STATUS_SUCCESS) passed = false;
    metadata_plan.abort_callback = [](void *) { return true; };
    if (ggml_graph_compute(metadata, &metadata_plan) != GGML_STATUS_ABORTED) passed = false;
    metadata_plan.abort_callback = nullptr;
    if (ggml_graph_compute(metadata, &metadata_plan) != GGML_STATUS_SUCCESS) passed = false;
    for (int i = 0; i < 105; ++i) if (static_cast<float *>(history->data)[i] != input[i]) passed = false;
    auto * empty = ggml_sqr(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0));
    auto * empty_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(empty_graph, empty);
    auto empty_plan = ggml_graph_plan(empty_graph, threads, pool);
    empty_plan.n_threads = threads;
    if (ggml_graph_compute(empty_graph, &empty_plan) != GGML_STATUS_SUCCESS) passed = false;

    ggml_threadpool_free(pool);
    ggml_free(ctx);
    std::cout << "cpu_graph threads=" << threads << " repetitions=64 cancellation_cycles=32 elapsed_ms=" << elapsed
              << " pre_cancel_clean=" << clean << " verified=" << passed << '\n';
    return passed;
}

static bool prima_cpu_graph_contracts() {
    bool passed = true;
    for (int threads : {1, 2, 8}) if (!prima_cpu_graph_contract(threads)) passed = false;
    return passed;
}
