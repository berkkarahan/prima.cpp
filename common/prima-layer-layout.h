#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

// Immutable absolute ownership and O(1) rank-local tensor/graph indexing. This
// metadata does not by itself enable segmented transport or row/tensor splitting.
class prima_layer_layout {
    std::vector<uint32_t> owners_;
    std::vector<int32_t> local_;
    std::vector<int32_t> graph_;
    uint32_t world_ = 0, rank_ = 0, count_ = 0, graphs_ = 0;
public:
    prima_layer_layout() = default;
    prima_layer_layout(const uint32_t * owners, uint32_t size, uint32_t world, uint32_t rank) {
        if (!owners || !size || !world || world > 32 || rank >= world || size > INT32_MAX)
            throw std::invalid_argument("Invalid absolute layer ownership");
        world_ = world;
        rank_ = rank;
        owners_.assign(owners, owners + size);
        local_.assign(size, -1);
        graph_.assign(size, -1);
        bool previous = false;
        for (uint32_t i = 0; i < size; ++i) {
            if (owners_[i] >= world) throw std::invalid_argument("Layer owner is outside the approved world");
            const bool mine = owners_[i] == rank;
            if (mine) {
                if (!previous) ++graphs_;
                local_[i] = count_++;
                graph_[i] = graphs_ - 1;
            }
            previous = mine;
        }
    }
    bool active() const { return !owners_.empty(); }
    uint32_t size() const { return owners_.size(); }
    uint32_t world() const { return world_; }
    uint32_t rank() const { return rank_; }
    uint32_t count() const { return count_; }
    uint32_t graphs() const { return graphs_; }
    bool owns(uint32_t layer) const { return local_.at(layer) >= 0; }
    int32_t local(uint32_t layer) const { return local_.at(layer); }
    int32_t graph(uint32_t layer) const { return graph_.at(layer); }
    bool matches(const uint32_t * owners, uint32_t size) const {
        if (!owners || size != owners_.size()) return false;
        for (uint32_t i = 0; i < size; ++i) if (owners_[i] != owners[i]) return false;
        return true;
    }
    bool window_compatible(const uint32_t * windows) const {
        uint64_t cycle = 0;
        for (uint32_t i = 0; i < world_; ++i) cycle += windows[i];
        if (!cycle) return false;
        for (uint32_t layer = 0; layer < size(); ++layer) {
            const uint64_t offset = layer % cycle;
            uint64_t end = 0;
            uint32_t expected = world_;
            for (uint32_t step = 0; step < world_; ++step) {
                const uint32_t owner = world_ == 1 ? 0 : (step + 1) % world_;
                end += windows[owner];
                if (offset < end) { expected = owner; break; }
            }
            if (owners_[layer] != expected) return false;
        }
        return true;
    }
    bool execution_compatible(const uint32_t * windows) const {
        uint64_t layers = 0;
        for (uint32_t i = 0; i < world_; ++i) layers += windows[i];
        return layers == size() && window_compatible(windows);
    }
};
