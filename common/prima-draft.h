#pragma once
#include "common.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

inline void prima_validate_draft_limits(int minimum, int maximum, float probability) {
    if (minimum < 0 || minimum > 1024 || maximum < 0 || maximum > 1024 ||
        !std::isfinite(probability) || probability < 0 || probability > 1)
        throw std::invalid_argument("Invalid draft limits: token counts require 0..1024 and probability 0..1");
}

inline void prima_trim_draft_to_vocab(llama_tokens & tokens, int vocabulary) {
    const auto invalid = std::find_if(tokens.begin(), tokens.end(), [vocabulary](llama_token id) {
        return id < 0 || id >= vocabulary;
    });
    tokens.erase(invalid, tokens.end());
}

inline gpt_params prima_draft_params(const gpt_params & target) {
    if (target.speculative.type != "auto" && target.speculative.type != "draft" && target.speculative.type != "draft-simple")
        throw std::invalid_argument("This runtime requires draft-simple speculative decoding");
    if (target.speculative.model.empty()) throw std::invalid_argument("Draft decoding requires a draft model");
    prima_validate_draft_limits(target.speculative.n_min, target.speculative.n_max, target.speculative.p_min);
    gpt_params draft = target;
    draft.local_draft = true;
    draft.layer_owners.clear();
    draft.lora_adapters.clear();
    draft.segmented = false;
    draft.cache_ram_mib = 0;
    draft.n_ctx_checkpoints = 0;
    draft.checkpoint_cache_bytes = 0;
    draft.execution_manifest.clear();
    draft.check_execution_manifest = false;
    draft.model = target.speculative.model;
    draft.model_url.clear();
    draft.hf_repo.clear();
    draft.hf_file.clear();
    draft.n_world = 1;
    draft.rank = 0;
    draft.n_parallel = 1;
    std::fill_n(draft.n_layer_window, 32, 0);
    draft.n_ctx = target.speculative.n_ctx > 0 ? target.speculative.n_ctx : target.n_ctx;
    draft.n_gpu_layers = target.speculative.n_gpu_layers < 0 ? 0 : target.speculative.n_gpu_layers;
    draft.cache_type_k = target.speculative.cache_type_k;
    draft.cache_type_v = target.speculative.cache_type_v;
    if (!draft.n_gpu_layers) {
        draft.devices = {nullptr};
        draft.keep_out_in_cuda = false;
    } else if (!draft.devices.empty() && draft.devices.front() == nullptr) {
        throw std::invalid_argument("Draft GPU layers conflict with explicit CPU-only device selection");
    }
    return draft;
}
