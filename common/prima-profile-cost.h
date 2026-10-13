#ifndef PRIMA_PROFILE_COST_H
#define PRIMA_PROFILE_COST_H

#include "ggml.h"
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct prima_tensor_cost {
    std::map<ggml_type, int64_t> flops;
    uint64_t weights_bytes = 0;
    uint64_t parameters = 0;
    uint64_t active_bytes = 0;
    uint64_t kv_bytes = 0;
};

struct prima_model_costs {
    std::vector<prima_tensor_cost> layers;
    prima_tensor_cost input, output;
    bool tied_output = false;
    int64_t embedding_row_bytes = 0;
    int64_t history = 0;
    int64_t context = 0;
};

struct prima_layer_shape {
    int64_t heads, kv_heads, head_k, head_v, kv_tokens, experts, experts_used, embedding;
    int64_t expert_width = 0;
    bool swiglu_oai = false;
    bool selected_softmax = false;
};

inline int64_t prima_cost_product(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
        throw std::invalid_argument("Profiler tensor cost overflow");
    return a * b;
}

inline int64_t prima_cost_sum(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b)
        throw std::invalid_argument("Profiler tensor cost overflow");
    return a + b;
}

inline void prima_cost_add(prima_tensor_cost & cost, ggml_type type, int64_t flops) {
    if (flops < 0 || cost.flops[type] > std::numeric_limits<int64_t>::max() - flops)
        throw std::invalid_argument("Profiler tensor cost overflow");
    cost.flops[type] += flops;
}

// Costs use actual tensor dimensions. All experts contribute storage, while only
// the selected experts contribute per-token arithmetic. These remain estimates.
inline void prima_cost_layer_tensor(prima_tensor_cost & cost, const std::string & role,
                                   const ggml_tensor * tensor, const prima_layer_shape & shape) {
    const int64_t elements = ggml_nelements(tensor);
    if (elements <= 0 || tensor->ne[3] != 1)
        throw std::invalid_argument("Unsupported profiler tensor shape: " + role);
    if (role == "attn_norm.weight" || role == "ffn_norm.weight" || role == "post_attention_norm.weight") {
        if (elements != shape.embedding || tensor->ne[1] != 1 || tensor->ne[2] != 1)
            throw std::invalid_argument("Invalid profiler norm shape");
        prima_cost_add(cost, GGML_TYPE_F32, 4 * shape.embedding + 1);
        prima_cost_add(cost, tensor->type, elements);
    } else if (role == "attn_q_norm.weight" || role == "attn_k_norm.weight") {
        const int64_t heads = role == "attn_q_norm.weight" ? shape.heads : shape.kv_heads;
        if (elements != shape.head_k || tensor->ne[1] != 1 || tensor->ne[2] != 1)
            throw std::invalid_argument("Invalid profiler head norm shape");
        prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(4 * elements + 1, heads));
        prima_cost_add(cost, tensor->type, prima_cost_product(elements, heads));
    } else if (role == "attn_sinks.weight") {
        if (tensor->type != GGML_TYPE_F32 || elements != shape.heads || tensor->ne[1] != 1 || tensor->ne[2] != 1)
            throw std::invalid_argument("Invalid profiler attention sink shape/type");
        prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(3, shape.heads));
    } else if (role == "ffn_gate_exps.bias" || role == "ffn_up_exps.bias" || role == "ffn_down_exps.bias") {
        const int64_t width = role == "ffn_down_exps.bias" ? shape.embedding : shape.expert_width;
        if (width <= 0 || tensor->ne[0] != width || tensor->ne[1] != shape.experts || tensor->ne[2] != 1 ||
            shape.experts_used <= 0 || shape.experts_used > shape.experts)
            throw std::invalid_argument("Invalid profiler expert bias shape");
        prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(width, shape.experts_used));
    } else if (role == "ffn_gate_inp.bias") {
        if (elements != shape.experts || tensor->ne[1] != 1 || tensor->ne[2] != 1)
            throw std::invalid_argument("Invalid profiler router bias shape");
        prima_cost_add(cost, GGML_TYPE_F32, elements);
    } else if (role.size() >= 5 && role.substr(role.size() - 5) == ".bias") {
        int64_t width = tensor->ne[0];
        if (role == "attn_q.bias") width = prima_cost_product(shape.heads, shape.head_k);
        else if (role == "attn_k.bias") width = prima_cost_product(shape.kv_heads, shape.head_k);
        else if (role == "attn_v.bias") width = prima_cost_product(shape.kv_heads, shape.head_v);
        else if (role == "attn_qkv.bias") width = prima_cost_sum(prima_cost_product(shape.heads, shape.head_k),
            prima_cost_product(shape.kv_heads, prima_cost_sum(shape.head_k, shape.head_v)));
        else if (role == "attn_output.bias" || role == "attn_norm.bias" || role == "ffn_norm.bias" ||
                 role == "post_attention_norm.bias" || role == "ffn_down.bias") width = shape.embedding;
        else if (role != "ffn_gate.bias" && role != "ffn_up.bias")
            throw std::invalid_argument("Unsupported profiler bias role: " + role);
        if (elements != width || tensor->ne[1] != 1 || tensor->ne[2] != 1)
            throw std::invalid_argument("Invalid profiler bias shape: " + role);
        prima_cost_add(cost, tensor->type, elements);
    } else if (role == "attn_qkv.weight") {
        const int64_t q = prima_cost_product(shape.heads, shape.head_k);
        const int64_t k = prima_cost_product(shape.kv_heads, shape.head_k);
        const int64_t v = prima_cost_product(shape.kv_heads, shape.head_v);
        if (tensor->ne[0] != shape.embedding || tensor->ne[1] != prima_cost_sum(prima_cost_sum(q, k), v) || tensor->ne[2] != 1)
            throw std::invalid_argument("Invalid profiler fused attention shape");
        prima_cost_add(cost, tensor->type, prima_cost_product(2, elements));
        prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(5, prima_cost_sum(q, k)) / 2);
        prima_cost_add(cost, GGML_TYPE_F16, prima_cost_product(prima_cost_product(2, prima_cost_sum(shape.head_k, shape.head_v)),
            prima_cost_product(shape.heads, shape.kv_tokens)));
        prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(7 * shape.heads, shape.kv_tokens));
    } else if (role == "attn_q.weight" || role == "attn_k.weight" || role == "attn_v.weight"
               || role == "attn_output.weight" || role == "ffn_gate.weight" || role == "ffn_up.weight"
               || role == "ffn_down.weight" || role == "ffn_gate_inp.weight"
               || role == "ffn_gate_exps.weight" || role == "ffn_up_exps.weight" || role == "ffn_down_exps.weight") {
        const bool expert = role.find("_exps.") != std::string::npos;
        if ((role == "attn_q.weight" && (tensor->ne[0] != shape.embedding || tensor->ne[1] != shape.heads * shape.head_k))
            || (role == "attn_k.weight" && (tensor->ne[0] != shape.embedding || tensor->ne[1] != shape.kv_heads * shape.head_k))
            || (role == "attn_v.weight" && (tensor->ne[0] != shape.embedding || tensor->ne[1] != shape.kv_heads * shape.head_v))
            || (role == "attn_output.weight" && (tensor->ne[0] != shape.heads * shape.head_v || tensor->ne[1] != shape.embedding)))
            throw std::invalid_argument("Invalid profiler attention shape: " + role);
        if ((expert && (shape.experts_used <= 0 || shape.experts_used > shape.experts || tensor->ne[2] != shape.experts))
            || (!expert && tensor->ne[2] != 1))
            throw std::invalid_argument("Unsupported profiler expert shape: " + role);
        if (expert && shape.expert_width > 0) {
            const bool down = role == "ffn_down_exps.weight";
            if (tensor->ne[0] != (down ? shape.expert_width : shape.embedding) ||
                tensor->ne[1] != (down ? shape.embedding : shape.expert_width))
                throw std::invalid_argument("Invalid profiler expert matrix dimensions");
        }
        int64_t arithmetic = prima_cost_product(2, prima_cost_product(tensor->ne[0], tensor->ne[1]));
        if (expert) arithmetic = prima_cost_product(arithmetic, shape.experts_used);
        prima_cost_add(cost, tensor->type, arithmetic);
        const int64_t hidden = role.find("down") != std::string::npos ? tensor->ne[0] : tensor->ne[1];
        // Non-matrix arithmetic remains a logical-work proxy, as in the
        // existing roofline model. OAI adds clipping, alpha and shifted up work.
        if (role == "ffn_gate.weight" || role == "ffn_gate_exps.weight")
            prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product((shape.swiglu_oai ? 10 : 8) * hidden, expert ? shape.experts_used : 1));
        if (role == "ffn_up.weight" || role == "ffn_up_exps.weight")
            prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product((shape.swiglu_oai ? 3 : 1) * hidden, expert ? shape.experts_used : 1));
        if (role == "ffn_down.weight" || role == "ffn_down_exps.weight" || role == "attn_output.weight")
            prima_cost_add(cost, GGML_TYPE_F32, shape.embedding);
        if (shape.swiglu_oai && role == "ffn_down_exps.weight")
            prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(2 * shape.experts_used - 1, shape.embedding));
        if (shape.selected_softmax && role == "ffn_gate_inp.weight") {
            prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(5, shape.experts_used));
            int64_t passes = 0;
            for (int64_t n = shape.experts - 1; n > 0; n >>= 1) ++passes;
            // Sorting proxy uses all router candidates, not just selected ones.
            prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(shape.experts, passes));
        }
        if (role == "attn_q.weight" || role == "attn_k.weight")
            prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(5, tensor->ne[1]) / 2);
        if (role == "attn_k.weight") {
            prima_cost_add(cost, GGML_TYPE_F16, prima_cost_product(2 * shape.head_k,
                prima_cost_product(shape.heads, shape.kv_tokens)));
            prima_cost_add(cost, GGML_TYPE_F32, prima_cost_product(7 * shape.heads, shape.kv_tokens));
        }
        if (role == "attn_v.weight")
            prima_cost_add(cost, GGML_TYPE_F16, prima_cost_product(2 * shape.head_v,
                prima_cost_product(shape.heads, shape.kv_tokens)));
    } else {
        throw std::invalid_argument("Unsupported profiler tensor: " + role);
    }
    cost.weights_bytes += ggml_nbytes(tensor);
    const bool expert_storage = role.find("_exps.") != std::string::npos;
    cost.active_bytes += expert_storage ? ggml_nbytes(tensor) / shape.experts * shape.experts_used : ggml_nbytes(tensor);
    if (role == "attn_qkv.weight")
        cost.kv_bytes += prima_cost_product(prima_cost_product(2, prima_cost_product(shape.kv_heads, shape.kv_tokens)),
            prima_cost_sum(shape.head_k, shape.head_v));
    if (role == "attn_k.weight" || role == "attn_v.weight")
        cost.kv_bytes += prima_cost_product(2 * shape.kv_heads * shape.kv_tokens,
            role == "attn_k.weight" ? shape.head_k : shape.head_v);
    cost.parameters += elements;
}
#endif
