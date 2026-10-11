#pragma once
#include "common.h"
#include "json.hpp"
#include "prima-sha256.h"
#include <fstream>
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
    // The current piped-ring kernel visits workers first, then rank zero.
    // Reject unrepresentable schedules rather than altering layer ownership.
    for (size_t step = 0; step < ranks.size(); ++step) {
        size_t index = ranks.size() == 1 ? 0 : (step + 1) % ranks.size();
        const auto & rank = ranks.at(index);
        if (rank.at("rank") != index || rank.at("ngl").get<int>() < 0) throw std::runtime_error("Invalid rank order/offload");
        const auto & layers = rank.at("layers");
        if (layers.size() > 1) throw std::runtime_error("Repeated rank visits require a newer execution kernel");
        if (!layers.empty()) {
            const int start = layers.at(0).at("start").get<int>();
            const int end = layers.at(0).at("end").get<int>();
            if (start != covered || end <= start || end > total) throw std::runtime_error("Layer coverage/order mismatch");
            params.n_layer_window[index] = end - start;
            covered = end;
        }
        if (rank.at("ngl").get<int>() > static_cast<int>(params.n_layer_window[index])) throw std::runtime_error("Offload exceeds rank allocation");
        if (rank.value("output_gpu", false) && (index != 0 || rank.at("ngl").get<int>() == 0)) {
            throw std::runtime_error("Output GPU placement requires rank zero with selected GPU layers");
        }
    }
    if (covered != total) throw std::runtime_error("Incomplete model coverage");
    const auto & mine = ranks.at(params.rank);
    params.n_world = ranks.size();
    params.n_gpu_layers = mine.at("ngl").get<int>();
    // The legacy name controls the generic accelerator output buffer, including Vulkan.
    params.keep_out_in_cuda = mine.value("output_gpu", false);
    params.master_ip = ranks.at(0).at("address").get<std::string>();
    params.next_node_ip = ranks.at((params.rank + 1) % ranks.size()).at("address").get<std::string>();
    params.data_port = value.at("data_port").get<uint32_t>();
    params.signal_port = value.at("signal_port").get<uint32_t>();
    const auto & proportions = mine.at("tensor_split");
    if (proportions.size() > 16) throw std::runtime_error("Too many tensor split devices");
    std::fill_n(params.tensor_split, llama_max_devices(), 0.0f);
    for (size_t i = 0; i < proportions.size(); ++i) params.tensor_split[i] = proportions.at(i).get<float>();
}

inline std::string prima_manifest_digest(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return prima_sha256(std::string(std::istreambuf_iterator<char>(file), {}));
}
