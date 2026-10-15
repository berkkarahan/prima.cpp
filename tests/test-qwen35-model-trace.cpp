// Model-backed diagnostic: dump first-layer F32 intermediates and final logits.
// Compile without PRIMA_MODEL_TRACE_NATIVE against pinned upstream libraries to
// compare the same GGUF and prompt; this is not an unconditional CTest test.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#ifdef PRIMA_MODEL_TRACE_NATIVE
#include "common.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool trace_tensor(ggml_tensor * tensor, bool ask, void * user) {
    const std::string name = ggml_get_name(tensor);
    const bool selected = tensor->type == GGML_TYPE_F32 &&
        ((name.size() >= 2 && (name.substr(name.size() - 2) == "-0" || name.substr(name.size() - 2) == "-3")) ||
         name.compare(0, 6, "l_out-") == 0 || name == "result_output");
    if (ask) return selected;
    if (!selected) return true;
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    const std::string path = *static_cast<std::string *>(user) + "." + name + ".bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char *>(tensor->ne), sizeof(tensor->ne));
    uint64_t strides[4];
    for (int i = 0; i < 4; ++i) strides[i] = tensor->nb[i];
    out.write(reinterpret_cast<const char *>(strides), sizeof(strides));
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return bool(out);
}

int main(int argc, char ** argv) {
    if (argc < 3) return 77;
    std::string prefix = argv[2];
    const std::string prompt = argc > 3 ? argv[3] : "The capital of France is Paris.\n";
    llama_backend_init();
    llama_model * model;
    llama_context * ctx;
    std::vector<llama_token> tokens;
#ifdef PRIMA_MODEL_TRACE_NATIVE
    gpt_params params;
    params.model = argv[1];
    params.n_ctx = 512;
    params.n_batch = params.n_ubatch = 64;
    params.n_gpu_layers = 0;
    params.devices = {nullptr};
    params.cpuparams.n_threads = 8;
    params.warmup = false;
    params.cb_eval = trace_tensor;
    params.cb_eval_user_data = &prefix;
    auto init = llama_init_from_gpt_params(params);
    model = init.model;
    ctx = init.context;
    if (!model || !ctx) return 1;
    tokens = ::llama_tokenize(ctx, prompt, false);
    auto batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()), 0, 0);
    const int vocab_size = llama_n_vocab(model);
#else
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = {nullptr};
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.cb_eval = trace_tensor;
    cp.cb_eval_user_data = &prefix;
    ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const auto * vocab = llama_model_get_vocab(model);
    tokens.resize(prompt.size()*2 + 8);
    const int count = llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), tokens.data(), int32_t(tokens.size()), false, true);
    if (count <= 0) return 1;
    tokens.resize(count);
    auto batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()));
    const int vocab_size = llama_vocab_n_tokens(vocab);
#endif
    const int status = llama_decode(ctx, batch);
    if (!status) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        const auto token = std::max_element(logits, logits + vocab_size) - logits;
        std::printf("greedy token: %lld logit: %f\n", static_cast<long long>(token), logits[token]);
    }
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return status;
}
