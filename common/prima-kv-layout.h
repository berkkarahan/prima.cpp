#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Storage planning is independent of the two pools' mutable sequence/cell state.
// Full-size mode preserves existing allocation until compact-pool consumers are
// wired together. This class alone does not enable compact cache execution.
class prima_kv_layout {
public:
    struct layer {
        uint32_t absolute;
        uint32_t cells;
        bool sliding;
    };

private:
    std::vector<layer> layers_;
    uint32_t base_cells_ = 0;
    uint32_t compact_cells_ = 0;
    uint32_t base_layers_ = 0;
    uint32_t sliding_layers_ = 0;
    bool full_sliding_ = true;

public:
    prima_kv_layout() = default;

    prima_kv_layout(const uint32_t * sliding, uint32_t layer_count,
                    const std::vector<uint32_t> & owned,
                    uint32_t base_cells, uint32_t window, uint32_t sequences,
                    uint32_t microbatch, bool full_sliding = true) :
        base_cells_(base_cells), compact_cells_(base_cells), full_sliding_(full_sliding) {
        if (!sliding || !layer_count || !base_cells || base_cells > INT32_MAX ||
            !sequences || !microbatch || owned.size() > layer_count) {
            throw std::invalid_argument("Invalid KV storage layout dimensions");
        }
        uint32_t previous = 0;
        bool first = true;
        for (uint32_t absolute : owned) {
            if (absolute >= layer_count || (!first && absolute <= previous) || sliding[absolute] > 1) {
                throw std::invalid_argument("Invalid KV storage layer ownership/pattern");
            }
            first = false;
            previous = absolute;
            if (sliding[absolute]) ++sliding_layers_;
            else ++base_layers_;
        }
        if (sliding_layers_) {
            if (!window || window > INT32_MAX) throw std::invalid_argument("Invalid KV sliding window");
            // Unified sequence pool, as in the pinned baseline: retain each
            // sequence's window plus the complete incoming microbatch before
            // attention consumes its earliest token. Always pad SWA to 256.
            const uint64_t needed = uint64_t(window) * sequences + microbatch;
            const uint64_t bounded = needed < base_cells ? needed : base_cells;
            const uint64_t padded = (bounded + 255u) & ~uint64_t(255u);
            if (padded > INT32_MAX) throw std::overflow_error("KV sliding pool size overflow");
            compact_cells_ = static_cast<uint32_t>(padded);
        }
        layers_.reserve(owned.size());
        for (uint32_t absolute : owned) {
            const bool is_sliding = sliding[absolute] != 0;
            layers_.push_back({absolute, is_sliding && !full_sliding_ ? compact_cells_ : base_cells_, is_sliding});
        }
    }

    const layer & local(uint32_t index) const { return layers_.at(index); }
    uint32_t count() const { return static_cast<uint32_t>(layers_.size()); }
    uint32_t base_cells() const { return base_cells_; }
    uint32_t compact_cells() const { return compact_cells_; }
    uint32_t base_layers() const { return base_layers_; }
    uint32_t sliding_layers() const { return sliding_layers_; }
    bool full_sliding() const { return full_sliding_; }
};
