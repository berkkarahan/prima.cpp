#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <stdexcept>
#include <string>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void fixture(const char * path, const std::string & arch, uint32_t blocks, uint32_t mtp = 0) {
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
    gguf_write_to_file(ctx, path, true);
    gguf_free(ctx);
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
