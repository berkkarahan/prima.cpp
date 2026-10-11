#include "prima-sha256.h"
#include "prima-env.h"
#include "../ggml/src/ggml-vulkan-dispatch.h"
#include <limits>
#include <iostream>
int main() {
    for (uint32_t rows_per_group : {1u, 4u}) {
        for (uint32_t max_x : {64u, 65535u}) {
            for (uint64_t rows : {1ull, 3ull, 4ull, 7ull, 8ull, 65535ull, 65536ull, 65537ull, 262145ull}) {
                const auto grid = ggml_vk_vec_grid(rows, rows_per_group, max_x, 65535);
                const uint64_t coverage = (uint64_t)grid[0]*grid[1]*rows_per_group;
                if (!grid[0] || grid[0] > max_x || grid[1] > 65535 || coverage < rows) return 5;
            }
        }
    }
    for (const auto invalid : {std::array<uint64_t, 4>{0, 1, 65535, 65535},
                              std::array<uint64_t, 4>{1, 0, 65535, 65535},
                              std::array<uint64_t, 4>{1, 1, 0, 65535},
                              std::array<uint64_t, 4>{1, 1, 65535, 0},
                              std::array<uint64_t, 4>{65537, 1, 65535, 1},
                              std::array<uint64_t, 4>{std::numeric_limits<uint64_t>::max(), 1, 65535, 65535}}) {
        try {
            ggml_vk_vec_grid(invalid[0], (uint32_t)invalid[1], (uint32_t)invalid[2], (uint32_t)invalid[3]);
            return 6;
        } catch (const std::invalid_argument &) {
        }
    }
    if (prima_flag_enabled(nullptr) || prima_flag_enabled("") || prima_flag_enabled("0") ||
        prima_flag_enabled("false") || !prima_flag_enabled("1")) return 4;
    if (prima_sha256("") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") return 1;
    if (prima_sha256("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return 2;
    if (prima_sha256(std::string(1000000, 'a')) != "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") return 3;
    std::cout << "SHA-256 known vectors passed\n";
    return 0;
}
