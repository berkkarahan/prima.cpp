#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

inline double ggml_vk_timestamp_ms(uint64_t begin, uint64_t end, uint32_t valid_bits, double period_ns) {
    if (valid_bits == 0 || valid_bits > 64 || !std::isfinite(period_ns) || period_ns <= 0)
        throw std::invalid_argument("Invalid Vulkan timestamp width/period");
    const uint64_t mask = valid_bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t(1) << valid_bits) - 1;
    const double ms = double((end - begin) & mask) * period_ns / 1000000.0;
    if (!std::isfinite(ms)) throw std::invalid_argument("Vulkan timestamp duration overflow");
    return ms;
}
