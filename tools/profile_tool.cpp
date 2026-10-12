#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "llama.h"
#include "json.hpp"
#include "prima-profile-io.h"
#include <cstdio>
#include <cstdlib>
#include <set>
#include <cmath>

static nlohmann::ordered_json typed_profile(llama_model * model, const llama_model_loader * metadata, const gpt_params & params) {
    using nlohmann::ordered_json;
    const char * requested = std::getenv("KLLM_PRIMA_PROFILE_BACKEND");
    const std::string backend = requested ? requested : device_has_vulkan() ? "vulkan"
        : device_has_metal() ? "metal" : device_has_cuda() ? "cuda" : "cpu";
    if (backend != "cpu" && backend != "vulkan" && backend != "metal" && backend != "cuda")
        throw std::invalid_argument("Unsupported kernel profile backend");
    const auto costs = device_model_costs(model, metadata, params.n_ctx, params.n_ctx, params.flash_attn);
    std::set<ggml_type> types;
    for (const auto & layer : costs.layers) for (const auto & entry : layer.flops) types.insert(entry.first);
    for (const auto & entry : costs.output.flops) types.insert(entry.first);
    ordered_json cpu = ordered_json::object(), gpu = ordered_json::object();
    std::map<ggml_type, double> cpu_rates, gpu_rates;
    for (auto type : types) {
        const double rate = device_cpu_flops(model, type, GGML_TYPE_F32, params.cpuparams.n_threads);
        if (!std::isfinite(rate) || rate <= 0)
            throw std::runtime_error(std::string("Unavailable CPU kernel for model type ") + ggml_type_name(type));
        cpu_rates[type] = rate;
        cpu[ggml_type_name(type)] = rate;
        double accelerated = 0;
        if (backend == "vulkan") accelerated = device_vulkan_flops(model, type, GGML_TYPE_F32);
        if (backend == "metal") accelerated = device_metal_flops(model, type, GGML_TYPE_F32);
        if (backend == "cuda") accelerated = device_cuda_flops(model, type, GGML_TYPE_F32);
        if (backend != "cpu") {
            if (!std::isfinite(accelerated) || accelerated <= 0)
                throw std::runtime_error(backend + " kernel unavailable for model type " + ggml_type_name(type));
            gpu_rates[type] = accelerated;
            gpu[ggml_type_name(type)] = accelerated;
        }
    }
    const double ram = device_memory_bw(params.cpuparams.n_threads);
    double vram = 0, copy = 0;
    if (backend == "vulkan") { vram = device_vulkan_read_vram_bw(); copy = device_vulkan_mem_copy(model); }
    if (backend == "metal") { vram = device_metal_read_vram_bw(); copy = device_metal_mem_copy(model); }
    if (backend == "cuda") { vram = device_cuda_read_vram_bw(); copy = device_cuda_mem_copy(model); }
    if (!std::isfinite(ram) || ram <= 0 || (backend != "cpu" && (!std::isfinite(vram) || vram <= 0 || !std::isfinite(copy) || copy <= 0)))
        throw std::runtime_error("Unavailable kernel profile memory/copy measurement");
    float disk_seq = 0, disk_random = 0, write_seq = 0, write_random = 0;
    const auto slash = params.model.find_last_of("/\\");
    const std::string disk_directory = slash == std::string::npos ? "." : params.model.substr(0, slash + 1);
    prima_profile_disk(&disk_seq, &write_seq, false, disk_directory);
    const auto random_block = prima_profile_disk(&disk_random, &write_random, true, disk_directory);
    const auto encode_cost = [&](const prima_tensor_cost & cost) {
        ordered_json flops = ordered_json::object();
        for (const auto & entry : cost.flops) flops[ggml_type_name(entry.first)] = entry.second;
        return ordered_json{{"flops", flops}, {"weights_bytes", cost.weights_bytes},
            {"active_bytes", cost.active_bytes}, {"kv_bytes", cost.kv_bytes}};
    };
    const auto estimate = [&](const prima_tensor_cost & cost, const std::map<ggml_type, double> & rates, double bandwidth) {
        double compute = 0;
        for (const auto & entry : cost.flops) compute += entry.second / (rates.at(entry.first) * 1e6);
        return std::max(compute, (cost.active_bytes + cost.kv_bytes) / (bandwidth * 1e6));
    };
    ordered_json layers = ordered_json::array();
    for (size_t i = 0; i < costs.layers.size(); ++i) {
        auto entry = encode_cost(costs.layers[i]);
        entry["layer"] = i;
        entry["cpu_ms"] = estimate(costs.layers[i], cpu_rates, ram);
        entry["gpu_ms"] = backend == "cpu" ? ordered_json(nullptr) : ordered_json(estimate(costs.layers[i], gpu_rates, vram));
        layers.push_back(entry);
    }
    auto output = encode_cost(costs.output);
    output["cpu_ms"] = estimate(costs.output, cpu_rates, ram);
    output["gpu_ms"] = backend == "cpu" ? ordered_json(nullptr) : ordered_json(estimate(costs.output, gpu_rates, vram));
    ordered_json handoff = nullptr;
    if (backend != "cpu") {
        const auto kind = backend == "vulkan" ? PROFILER_BACKEND_TYPE_VULKAN
            : backend == "metal" ? PROFILER_BACKEND_TYPE_METAL : PROFILER_BACKEND_TYPE_CUDA;
        const auto transfer = device_host_transfer(llama_n_embd(model), kind);
        handoff = {{"method", "completed_f32_embedding_v1"}, {"bytes", transfer.bytes},
            {"upload_ms", transfer.upload_ms}, {"download_ms", transfer.download_ms}};
    }
    return ordered_json{{"schema_version", 2}, {"measurement_kind", "typed_roofline_estimate"},
        {"backend", backend}, {"context", costs.context}, {"history", costs.history},
        {"threads", params.cpuparams.n_threads}, {"total_layers", costs.layers.size()},
        {"cpu_gflops", cpu}, {"gpu_gflops", gpu}, {"layers", layers}, {"output", output},
        {"input_ms", costs.embedding_row_bytes / (ram * 1e6)}, {"tied_output", costs.tied_output},
        {"ram_gbytes_s", ram}, {"gpu_read_gbytes_s", backend == "cpu" ? ordered_json(nullptr) : ordered_json(vram)},
        {"gpu_kv_copy_ms", backend == "cpu" ? ordered_json(nullptr) : ordered_json(copy)},
        {"gpu_host_transfer", handoff},
        {"disk_read_seq_gbytes_s", disk_seq}, {"disk_read_random_gbytes_s", disk_random},
        {"disk_write_seq_gbytes_s", write_seq}, {"disk_write_random_gbytes_s", write_random},
        {"disk_method", "serial_unbuffered_v1"}, {"disk_queue_depth", 1},
        {"disk_file_bytes", 64 * 1024 * 1024}, {"disk_seq_block_bytes", 1024 * 1024},
        {"disk_random_block_bytes", random_block},
        {"memory_available_gib", device_physical_memory(true) / (double) (1ULL << 30)}};
}

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    // load the model and apply lora adapter, if any
    auto mparams = llama_model_params_from_gpt_params(params);
    struct llama_context_params cparams = llama_context_params_from_gpt_params(params);

    struct llama_model * model = nullptr;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else {
        model = llama_load_model_from_file(params.model.c_str(), mparams);
    }

    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.c_str());
        return -1;
    }

    llama_model_loader * ml = llama_model_load(params.model.c_str(), model, &mparams);
    if (std::getenv("KLLM_PRIMA_PROFILE_JSON")) {
        try {
            const auto value = typed_profile(model, ml, params);
            std::printf("KLLM_PRIMA_PROFILE=%s\n", value.dump().c_str());
        } catch (const std::exception & error) {
            LOG_ERR("Kernel profile failed: %s\n", error.what());
            device_free_model_metadata(ml);
            llama_free_model(model);
            return 1;
        }
    } else {
        device_info dev_info;
        llama_profile_device(&dev_info, model, ml, params.gpu_mem, params.n_predict, params.n_ctx, params.cpuparams.n_threads, params.flash_attn);
        device_print_props(&dev_info, 1, model, cparams);
    }

    device_free_model_metadata(ml);
    llama_free_model(model);
    return 0;
}
