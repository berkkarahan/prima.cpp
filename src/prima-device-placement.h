#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Baseline layer-split rule: select the first cumulative split strictly greater
// than the position of this offloaded layer (output is the final position).
inline std::vector<double> prima_device_splits(const std::vector<double> & weights) {
    if (weights.empty()) throw std::invalid_argument("No offload devices selected");
    double sum = 0;
    std::vector<double> result;
    for (const auto weight : weights) {
        if (!std::isfinite(weight) || weight < 0) throw std::invalid_argument("Invalid device split weight");
        sum += weight;
        result.push_back(sum);
    }
    if (!(sum > 0) || !std::isfinite(sum)) throw std::invalid_argument("Device split weights have no finite capacity");
    for (auto & value : result) value /= sum;
    result.back() = 1.0;
    return result;
}

inline size_t prima_layer_device(const std::vector<double> & splits, size_t position, size_t count) {
    if (!count || position >= count || splits.empty()) throw std::invalid_argument("Invalid offload position");
    const auto next = std::upper_bound(splits.begin(), splits.end(), double(position) / double(count));
    if (next == splits.end()) throw std::invalid_argument("Offload position has no selected device");
    return size_t(next - splits.begin());
}
