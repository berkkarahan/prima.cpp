#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

// Cover every output row within the device's X/Z dispatch limits. Division with
// remainder avoids overflow when a caller supplies a large tensor dimension.
inline std::array<uint32_t, 2> ggml_vk_vec_grid(uint64_t rows, uint32_t rows_per_group,
                                            uint32_t max_x, uint32_t max_z) {
    if (!rows || !rows_per_group || !max_x || !max_z) {
        throw std::invalid_argument("Invalid Vulkan vector dispatch dimensions");
    }
    const uint64_t count = rows / rows_per_group + (rows % rows_per_group != 0);
    const uint64_t z = count / max_x + (count % max_x != 0);
    if (z > max_z) {
        throw std::invalid_argument("Vulkan vector dispatch exceeds device limits");
    }
    const uint64_t x = count / z + (count % z != 0);
    return { (uint32_t)x, (uint32_t)z };
}
