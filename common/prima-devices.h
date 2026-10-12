#pragma once
#include "llama.h"
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<ggml_backend_dev_t> prima_parse_devices(const std::string & value) {
    if (value == "none") return {nullptr};
    std::vector<ggml_backend_dev_t> devices;
    std::set<ggml_backend_dev_t> seen;
    size_t begin = 0;
    do {
        const auto comma = value.find(',', begin);
        const auto name = value.substr(begin, comma == std::string::npos ? comma : comma - begin);
        auto * dev = name.empty() ? nullptr : ggml_backend_dev_by_name(name.c_str());
        if (!dev || (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU_FULL) ||
            !seen.insert(dev).second || devices.size() == llama_max_devices())
            throw std::invalid_argument("Invalid, duplicate or excessive selected device: " + name);
        devices.push_back(dev);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    } while (true);
    devices.push_back(nullptr);
    return devices;
}
