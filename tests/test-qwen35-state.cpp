// Model-backed cache contract. Pass the complete local Saluki GGUF explicitly;
// no model download or hardware requirement is introduced into default CTest.
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct owned_context {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    ~owned_context() {
        if (context) llama_free(context);
        if (model) llama_free_model(model);
    }
};

static void same_logits(const std::vector<float> & expected, const float * actual) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(expected[i] - actual[i]) > 1e-4f)
            throw std::runtime_error("Hybrid sequence replay changed continuation logits");
    }
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) return 77;
    owned_context owned;
    try {
        llama_backend_init();
        gpt_params params;
        params.model = argv[1];
        params.devices = {nullptr};
        params.n_gpu_layers = 0;
        if (argc == 3) {
            auto * device = ggml_backend_dev_by_name(argv[2]);
            check(device != nullptr, "Requested GPU is unavailable");
            params.devices = {device, nullptr};
            params.n_gpu_layers = 64;
            params.keep_out_in_cuda = true; // legacy name also selects Vulkan output
        }
        params.n_ctx = 512;
        params.n_batch = params.n_ubatch = 64;
        params.n_parallel = 2;
        params.cpuparams.n_threads = 8;
        params.warmup = false;
        auto init = llama_init_from_gpt_params(params);
        owned.model = init.model;
        owned.context = init.context;
        check(owned.model && owned.context, "Could not initialize the local model");
        auto * ctx = owned.context;
        // Explicitly developmental labels: these logs prove completed device
        // work, not a prepared runtime identity or a signed cluster manifest.
        llama_set_execution_identity(ctx, "development-unpublished", "development-qwen35-state-contract");
        auto tokens = ::llama_tokenize(ctx, "The capital of France is Paris.\n", false);
        check(llama_decode(ctx, llama_batch_get_one(tokens.data(), int32_t(tokens.size()), 0, 0)) == 0, "Prefill failed");
        const int vocab = llama_n_vocab(owned.model);
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token next = llama_token(std::max_element(logits, logits + vocab) - logits);
        const size_t size = llama_state_seq_get_size(ctx, 0);
        check(size > 0, "Empty hybrid snapshot");
        std::vector<uint8_t> state(size);
        check(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == state.size(), "Snapshot write failed");
        check(llama_decode(ctx, llama_batch_get_one(&next, 1, llama_pos(tokens.size()), 0)) == 0, "Decode failed");
        logits = llama_get_logits_ith(ctx, -1);
        const std::vector<float> expected(logits, logits + vocab);

        check(llama_kv_cache_seq_rm(ctx, 0, -1, -1), "Full sequence clear failed");
        check(llama_state_seq_set_data(ctx, state.data(), state.size(), 0) == state.size(), "Snapshot restore failed");
        check(llama_decode(ctx, llama_batch_get_one(&next, 1, llama_pos(tokens.size()), 0)) == 0, "Restored decode failed");
        same_logits(expected, llama_get_logits_ith(ctx, -1));

        check(llama_kv_cache_seq_rm(ctx, 0, -1, -1), "Sequence reset failed");
        check(llama_state_seq_set_data(ctx, state.data(), state.size(), 0) == state.size(), "Second restore failed");
        llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);
        check(llama_decode(ctx, llama_batch_get_one(&next, 1, llama_pos(tokens.size()), 1)) == 0, "Copied sequence decode failed");
        same_logits(expected, llama_get_logits_ith(ctx, -1));
        std::printf("Qwen3.5 sequence save/restore/copy passed (%zu snapshot bytes)\n", size);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
