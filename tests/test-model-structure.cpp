#include "llama.h"
#include "ggml.h"
#include "profiler.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void fixture(const char * path, const std::string & arch, uint32_t blocks, uint32_t mtp = 0, bool weights = false) {
    gguf_context * ctx = gguf_init_empty();
    const auto key = [&](const char * suffix) { return arch + "." + suffix; };
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    for (const auto & pair : {std::pair<const char *, uint32_t>{"block_count", blocks},
            {"context_length", 512}, {"embedding_length", 32}, {"vocab_size", 64},
            {"feed_forward_length", 64}, {"attention.head_count", 4}})
        gguf_set_val_u32(ctx, key(pair.first).c_str(), pair.second);
    gguf_set_val_f32(ctx, key("attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    if (arch == "qwen2moe") {
        gguf_set_val_u32(ctx, key("expert_count").c_str(), 4);
        gguf_set_val_u32(ctx, key("expert_used_count").c_str(), 2);
    }
    if (arch == "qwen35") {
        gguf_set_val_u32(ctx, key("nextn_predict_layers").c_str(), mtp);
        const int32_t sections[] = {2, 1, 1, 0};
        gguf_set_arr_data(ctx, key("rope.dimension_sections").c_str(), GGUF_TYPE_INT32, sections, 4);
        for (const auto & pair : {std::pair<const char *, uint32_t>{"ssm.conv_kernel", 4},
                {"ssm.inner_size", 32}, {"ssm.state_size", 8},
                {"ssm.time_step_rank", 4}, {"ssm.group_count", 2}})
            gguf_set_val_u32(ctx, key(pair.first).c_str(), pair.second);
    }
    ggml_context * tensors = nullptr;
    if (weights) {
        check(arch == "llama", "weight fixture is Llama");
        gguf_set_val_str(ctx, "tokenizer.ggml.model", "no_vocab");
        tensors = ggml_init({1024 * 1024, nullptr, false});
        check(tensors != nullptr, "fixture allocation failed");
        const auto tensor = [&](const std::string & name, int64_t x, int64_t y = 0) {
            auto * t = y ? ggml_new_tensor_2d(tensors, GGML_TYPE_F32, x, y)
                        : ggml_new_tensor_1d(tensors, GGML_TYPE_F32, x);
            ggml_set_name(t, name.c_str());
            ggml_set_zero(t);
            gguf_add_tensor(ctx, t);
        };
        tensor("token_embd.weight", 32, 64);
        tensor("output_norm.weight", 32);
        tensor("output.weight", 32, 64);
        for (uint32_t i = 0; i < blocks; ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            tensor(prefix + "attn_norm.weight", 32);
            for (const auto * role : {"attn_q", "attn_k", "attn_v", "attn_output"})
                tensor(prefix + role + ".weight", 32, 32);
            tensor(prefix + "ffn_norm.weight", 32);
            tensor(prefix + "ffn_gate.weight", 32, 64);
            tensor(prefix + "ffn_up.weight", 32, 64);
            tensor(prefix + "ffn_down.weight", 64, 32);
        }
    }
    gguf_write_to_file(ctx, path, !weights);
    gguf_free(ctx);
    if (tensors) ggml_free(tensors);
}

int main() {
    const char * path = "model-structure-fixture.gguf";
    try {
        llama_model_structure result{};
        char error[2048] = {};
        for (const auto & arch : {"llama", "qwen2moe", "qwen35"}) {
            fixture(path, arch, 4);
            check(llama_model_inspect(path, &result, error, sizeof(error)), error);
            check(result.stored_blocks == 4 && result.execution_layers == 4, "ordinary model layer mismatch");
            check(std::string(result.architecture) == arch, "architecture mismatch");
        }
        fixture(path, "qwen35", 65, 1);
        check(llama_model_inspect(path, &result, error, sizeof(error)), error);
        check(result.stored_blocks == 65 && result.execution_layers == 64, "MTP counted as execution layer");
        // Metadata-only success must also agree with the real tensor loader.
        // This catches accidentally routing Llama into MiniCPM3's loader.
        fixture(path, "llama", 2, 0, true);
        check(llama_model_inspect(path, &result, error, sizeof(error)), error);
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.n_layer_window[0] = result.execution_layers;
        std::unique_ptr<llama_model, decltype(&llama_free_model)> model(
            llama_load_model_from_file(path, params), llama_free_model);
        check(model != nullptr, "Llama initialization failed");
        std::unique_ptr<llama_model_loader, decltype(&device_free_model_metadata)> metadata(
            llama_model_load(path, model.get(), &params), device_free_model_metadata);
        check(llm_load_tensors(metadata.get(), model.get(), params) == 0, "Llama tensor loading failed");
        check(llama_n_layer(model.get()) == int32_t(result.execution_layers), "inspection/loading layer mismatch");
        metadata.reset();
        model.reset();
        llama_backend_free();
        for (const auto & arch : {"unsupported_architecture", "qwen35"}) {
            fixture(path, arch, 1, 1);
            check(!llama_model_inspect(path, &result, error, sizeof(error)), "invalid structure accepted");
            check(error[0] && !result.execution_layers, "failure has no diagnostic or leaks result");
        }
        FILE * file = std::fopen(path, "wb");
        std::fputs("GGUF", file);
        std::fclose(file);
        check(!llama_model_inspect(path, &result, error, sizeof(error)), "truncated model accepted");
        std::remove(path);
        return 0;
    } catch (const std::exception & error) {
        std::remove(path);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
