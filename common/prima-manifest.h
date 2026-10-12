#pragma once
#include "common.h"
#include "json.hpp"
#include "prima-sha256.h"
#include <fstream>
#include <cmath>
#include <map>
#include <set>
#include <memory>
#include <cstdio>
#include <regex>
#include <stdexcept>

inline nlohmann::json prima_read_manifest(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() > 4 * 1024 * 1024) throw std::runtime_error("Missing or oversized execution manifest");
    input.seekg(0);
    nlohmann::json value;
    input >> value;
    if (value.at("schema_version") != 1 || !value.at("ranks").is_array() ||
        value.at("ranks").empty() || value.at("ranks").size() > 32 ||
        !std::regex_match(value.at("runtime_id").get<std::string>(), std::regex("[0-9a-f]{64}"))) {
        throw std::runtime_error("Invalid execution manifest identity/schema");
    }
    return value;
}

inline std::string prima_adapter_basename(const std::string & path) {
    const auto separator = path.find_last_of("/\\");
    return separator == std::string::npos ? path : path.substr(separator + 1);
}

inline void prima_verify_adapter_bytes(const std::string & path, const nlohmann::json & asset) {
    std::unique_ptr<FILE, decltype(&std::fclose)> input(ggml_fopen(path.c_str(), "rb"), std::fclose);
    if (!input || !asset.at("size").is_number_integer() || asset.at("size").get<int64_t>() <= 0)
        throw std::runtime_error("Missing or invalid approved LoRA asset");
    const auto expected = asset.at("size").get<uint64_t>();
    const auto hash = asset.at("sha256").get<std::string>();
    if (!std::regex_match(hash, std::regex("[0-9a-f]{64}"))) throw std::runtime_error("Invalid LoRA asset digest");
    prima_sha256_state sha;
    std::array<char, 65536> buffer;
    uint64_t total = 0;
    for (;;) {
        const auto count = std::fread(buffer.data(), 1, buffer.size(), input.get());
        if (count > expected - total) throw std::runtime_error("LoRA asset exceeds its approved size");
        sha.update(buffer.data(), count);
        total += count;
        if (count < buffer.size()) break;
    }
    if (std::ferror(input.get()) || !std::feof(input.get()) || total != expected || sha.digest() != hash)
        throw std::runtime_error("LoRA bytes differ from the approved asset");
}

inline void prima_apply_manifest(gpt_params & params) {
    const auto value = prima_read_manifest(params.execution_manifest);
    const auto & ranks = value.at("ranks");
    const int total = value.at("total_layers").get<int>();
    if (total <= 0 || params.rank < 0 || static_cast<size_t>(params.rank) >= ranks.size()) {
        throw std::runtime_error("Invalid execution rank/layer count");
    }
    if (value.at("split_mode") != "layer" || !value.at("tensor_placements").empty()) {
        throw std::runtime_error("This runtime cannot execute row/tensor or remote expert placement exactly");
    }
    int covered = 0;
    std::fill_n(params.n_layer_window, 32, 0);
    params.layer_owners.assign(total, UINT32_MAX);
    for (size_t index = 0; index < ranks.size(); ++index) {
        const auto & rank = ranks.at(index);
        if (rank.at("rank") != index || rank.at("ngl").get<int>() < 0)
            throw std::runtime_error("Invalid rank order/offload");
        int previous = -1;
        for (const auto & range : rank.at("layers")) {
            const int start = range.at("start").get<int>();
            const int end = range.at("end").get<int>();
            if (start < 0 || start < previous || end <= start || end > total)
                throw std::runtime_error("Invalid or unordered layer ownership range");
            for (int layer = start; layer < end; ++layer) {
                if (params.layer_owners[layer] != UINT32_MAX) throw std::runtime_error("Overlapping layer ownership");
                params.layer_owners[layer] = uint32_t(index);
            }
            previous = end;
            params.n_layer_window[index] += end - start;
            covered += end - start;
        }
        if (rank.at("ngl").get<int>() > int(params.n_layer_window[index])) throw std::runtime_error("Offload exceeds rank allocation");
        if (rank.value("output_gpu", false) && (index != 0 || rank.at("devices").size() != 1))
            throw std::runtime_error("Output GPU placement requires rank zero with one selected device");
    }
    if (covered != total) throw std::runtime_error("Incomplete model coverage");
    if (value.contains("segments")) {
        const auto & segments = value.at("segments");
        if (!segments.is_array()) throw std::runtime_error("Manifest segments require an ordered array");
        int cursor = 0;
        for (const auto & segment : segments) {
            const int start = segment.at("start").get<int>();
            const int end = segment.at("end").get<int>();
            const auto node = segment.at("node_id").get<std::string>();
            if (start != cursor || end <= start || end > total)
                throw std::runtime_error("Manifest segments do not cover the model in order");
            const auto owner = params.layer_owners[start];
            if (ranks.at(owner).at("node_id") != node)
                throw std::runtime_error("Manifest segment node differs from absolute layer ownership");
            for (int layer = start; layer < end; ++layer)
                if (params.layer_owners[layer] != owner)
                    throw std::runtime_error("Manifest segment crosses rank ownership");
            cursor = end;
        }
        if (cursor != total) throw std::runtime_error("Incomplete manifest segment coverage");
    }
    params.segmented = false;
    int next_layer = 0;
    for (size_t step = 0; step < ranks.size(); ++step) {
        const size_t index = ranks.size() == 1 ? 0 : (step + 1) % ranks.size();
        const auto & ranges = ranks.at(index).at("layers");
        if (ranges.size() > 1 || (!ranges.empty() && ranges.at(0).at("start") != next_layer)) params.segmented = true;
        next_layer += params.n_layer_window[index];
    }
    if (params.segmented) {
        const auto & settings = value.at("settings");
        if (!settings.at("prima_graphs").is_boolean() || !settings.at("prima_graphs").get<bool>())
            throw std::runtime_error("Segmented execution requires planned graph reservations");
        const auto & declared = settings.at("prima_graph_counts");
        if (!declared.is_object() || declared.size() != ranks.size())
            throw std::runtime_error("Graph reservations must cover every selected rank");
        std::vector<uint32_t> graphs(ranks.size(), 0);
        graphs[0] = 2;
        for (int layer = 0; layer < total; ++layer)
            if (layer == 0 || params.layer_owners[layer] != params.layer_owners[layer - 1])
                ++graphs[params.layer_owners[layer]];
        for (size_t index = 0; index < ranks.size(); ++index) {
            const auto node = ranks.at(index).at("node_id").get<std::string>();
            const auto & count = declared.at(node);
            if (!count.is_number_integer() || count.get<int64_t>() != graphs[index])
                throw std::runtime_error("Graph reservation count differs from absolute execution visits");
        }
    }
    const auto & mine = ranks.at(params.rank);
    params.n_world = ranks.size();
    params.n_gpu_layers = mine.at("ngl").get<int>();
    // The legacy name controls the generic accelerator output buffer, including Vulkan.
    params.keep_out_in_cuda = mine.value("output_gpu", false);
    if (!params.devices.empty()) {
        const size_t selected = params.devices.size() - 1;
        const size_t declared = mine.contains("devices") ? mine.at("devices").size() : selected;
        const bool gpu_work = params.n_gpu_layers || params.keep_out_in_cuda;
        if ((gpu_work && (selected == 0 || selected != declared)) ||
            (!gpu_work && selected != 0))
            throw std::runtime_error("Explicit device selection disagrees with the execution manifest");
    }
    params.master_ip = ranks.at(0).at("address").get<std::string>();
    params.next_node_ip = ranks.at((params.rank + 1) % ranks.size()).at("address").get<std::string>();
    params.data_port = value.at("data_port").get<uint32_t>();
    params.signal_port = value.at("signal_port").get<uint32_t>();
    if (params.rank == 0 && value.contains("settings")) {
        const auto & settings = value.at("settings");
        const auto present = [&](const char * key) { return settings.contains(key) && !settings.at(key).is_null(); };
        const auto format = present("reasoning_format") ? settings.at("reasoning_format").get<std::string>() : "auto";
        const auto effort = present("reasoning_effort") ? settings.at("reasoning_effort").get<std::string>() : "";
        const auto budget = present("reasoning_budget") ? settings.at("reasoning_budget").get<int64_t>() : -1;
        if (budget < -1 || budget > INT32_MAX || (present("reasoning_budget") && !settings.at("reasoning_budget").is_number_integer()))
            throw std::runtime_error("Invalid approved reasoning budget");
        if (present("enable_thinking") && !settings.at("enable_thinking").is_boolean())
            throw std::runtime_error("Invalid approved thinking mode");
        const int thinking = present("enable_thinking") ? (settings.at("enable_thinking").get<bool>() ? 1 : 0) : -1;
        if (format != params.reasoning_format || effort != params.reasoning_effort || budget != params.sparams.reasoning_budget_tokens || thinking != params.enable_reasoning)
            throw std::runtime_error("Reasoning defaults differ from the approved execution manifest");
    }
    if (params.rank == 0 && value.contains("settings") && value.at("settings").contains("extra_args")) {
        const auto & extra = value.at("settings").at("extra_args");
        if (!extra.is_array()) throw std::runtime_error("Manifest launch extras must be an array");
        auto approved_kwargs = nlohmann::ordered_json::object();
        bool found_kwargs = false;
        for (size_t i = 0; i < extra.size(); ++i) {
            if (extra.at(i) != "--chat-template-kwargs") continue;
            if (found_kwargs || i + 1 >= extra.size() || !extra.at(i + 1).is_string())
                throw std::runtime_error("Invalid manifest template keyword defaults");
            const auto source = extra.at(++i).get<std::string>();
            if (source.size() > 64 * 1024) throw std::runtime_error("Manifest template keywords exceed 64 KiB");
            approved_kwargs = nlohmann::ordered_json::parse(source);
            if (!approved_kwargs.is_object()) throw std::runtime_error("Manifest template keywords require an object");
            found_kwargs = true;
        }
        if (approved_kwargs != nlohmann::ordered_json::parse(params.chat_template_kwargs))
            throw std::runtime_error("Template keyword defaults differ from the approved execution manifest");
    }
    if (value.contains("settings") && value.at("settings").contains("cache_ram")) {
        const auto & cache = value.at("settings").at("cache_ram");
        if (!cache.is_number_integer() || cache.get<int64_t>() < 0 || cache.get<int64_t>() > INT32_MAX)
            throw std::runtime_error("Manifest prompt cache requires a bounded nonnegative budget");
        params.cache_ram_mib = cache.get<int32_t>();
    }
    if (value.contains("settings") && value.at("settings").contains("checkpoints")) {
        const auto & settings = value.at("settings");
        const auto & count = settings.at("checkpoints");
        if (!count.is_number_integer() || count.get<int64_t>() < 0 || count.get<int64_t>() > 16384)
            throw std::runtime_error("Invalid manifest checkpoint count");
        params.n_ctx_checkpoints = count.get<int32_t>();
        if (params.n_ctx_checkpoints) {
            const auto & budget = settings.at("prima_checkpoint_bytes");
            const auto & spacing = settings.at("checkpoint_min_step");
            if (!budget.is_number_integer() || budget.get<int64_t>() <= 0 ||
                !spacing.is_number_integer() || spacing.get<int64_t>() < 0 || spacing.get<int64_t>() > INT32_MAX)
                throw std::runtime_error("Checkpoint execution requires its approved budget and spacing");
            params.checkpoint_cache_bytes = budget.get<uint64_t>();
            params.checkpoint_min_step = spacing.get<int32_t>();
        }
    }
    const auto lora_settings = value.value("settings", nlohmann::json::object());
    const auto declared = lora_settings.value("prima_lora", nlohmann::json::array());
    std::map<std::string, nlohmann::json> adapters;
    for (const auto & asset : value.value("assets", nlohmann::json::array())) if (asset.at("kind") == "adapter") {
        const auto name = asset.at("name").get<std::string>();
        if (name.empty() || prima_adapter_basename(name) != name || !adapters.emplace(name, asset).second)
            throw std::runtime_error("Invalid or duplicate LoRA asset basename");
    }
    if (!declared.is_array() || declared.size() > 4096 || declared.size() != params.lora_adapters.size())
        throw std::runtime_error("LoRA launch count differs from the execution manifest");
    if (!declared.empty() && params.lora_init_without_apply)
        throw std::runtime_error("Initial LoRA selection must match the approved scales");
    std::set<std::string> used, verified;
    for (size_t index = 0; index < declared.size(); ++index) {
        const auto & binding = declared.at(index);
        const auto name = binding.at("name").get<std::string>();
        const float scale = binding.at("scale").get<float>();
        const auto & launched = params.lora_adapters.at(index);
        if (!binding.at("scale").is_number() || !std::isfinite(scale) || !std::isfinite(launched.scale) ||
            name != prima_adapter_basename(launched.path) || scale != launched.scale || !adapters.count(name))
            throw std::runtime_error("LoRA ordered file/scale binding differs from the manifest");
        used.insert(name);
        if (verified.insert(launched.path).second) prima_verify_adapter_bytes(launched.path, adapters.at(name));
    }
    if (used.size() != adapters.size()) throw std::runtime_error("Unreferenced LoRA assets are not executable");
    if (!declared.empty()) for (const char * key : {"prima_adapter_bytes", "prima_adapter_scratch_per_token"}) {
        const auto & bound = value.at("settings").at(key);
        if (!bound.is_number_integer() || bound.get<int64_t>() <= 0)
            throw std::runtime_error("LoRA execution requires its planned storage and scratch bounds");
    }
    const auto & proportions = mine.at("tensor_split");
    if (proportions.size() > 16) throw std::runtime_error("Too many tensor split devices");
    std::fill_n(params.tensor_split, llama_max_devices(), 0.0f);
    for (size_t i = 0; i < proportions.size(); ++i) params.tensor_split[i] = proportions.at(i).get<float>();
}

inline std::string prima_manifest_digest(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return prima_sha256(std::string(std::istreambuf_iterator<char>(file), {}));
}

inline nlohmann::json prima_ready_report(const gpt_params & params, const std::string & stage) {
    const auto manifest = prima_read_manifest(params.execution_manifest);
    if (params.rank < 0 || static_cast<size_t>(params.rank) >= manifest.at("ranks").size()) {
        throw std::runtime_error("Readiness rank is absent from the execution manifest");
    }
    const auto & rank = manifest.at("ranks").at(params.rank);
    int layers = 0;
    for (const auto & window : rank.at("layers")) {
        layers += window.at("end").get<int>() - window.at("start").get<int>();
    }
    const std::string expected_stage = params.rank == 0 ? "server_request_loop" : "worker_decode_loop";
    if (stage != expected_stage || params.n_world != manifest.at("ranks").size() ||
        params.n_gpu_layers != rank.at("ngl").get<int>() || params.n_layer_window[params.rank] != layers) {
        throw std::runtime_error("Initialized rank disagrees with the readiness contract");
    }
    return {{"schema_version", 1}, {"runtime_id", manifest.at("runtime_id")},
        {"rank", params.rank}, {"ngl", params.n_gpu_layers}, {"layers", layers},
        {"manifest_mode", true}, {"manifest_digest", prima_manifest_digest(params.execution_manifest)},
        {"phase", "ready"}, {"readiness_stage", stage}};
}
