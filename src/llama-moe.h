#pragma once

#include "ggml.h"

// Weighted expert values are [embedding, selected expert, token]. Packing the
// expert axis lets one reduction replace a chain of device-wide additions.
inline ggml_tensor * llama_sum_experts(ggml_context * ctx, ggml_tensor * experts, bool pack_experts) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && experts->ne[3] == 1);
    const int64_t embedding = experts->ne[0];
    const int64_t used = experts->ne[1];
    const int64_t tokens = experts->ne[2];
    GGML_ASSERT(used > 0);
    if (pack_experts && used >= 4) {
        auto * packed = ggml_cont(ctx, ggml_permute(ctx, experts, 1, 0, 2, 3));
        return ggml_reshape_2d(ctx, ggml_sum_rows(ctx, packed), embedding, tokens);
    }
    auto * result = ggml_view_2d(ctx, experts, embedding, tokens, experts->nb[2], 0);
    for (int64_t i = 1; i < used; ++i) {
        auto * next = ggml_view_2d(ctx, experts, embedding, tokens, experts->nb[2], i * experts->nb[1]);
        result = ggml_add(ctx, result, next);
    }
    return used == 1 ? ggml_cont(ctx, result) : result;
}
