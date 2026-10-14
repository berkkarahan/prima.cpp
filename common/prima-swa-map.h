#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

// Compact tensor rows refer to the full cache's authoritative sequence cells.
// Preparing a slot does not reclaim rows or change either cache. Commit only
// after the full cache has successfully reserved the corresponding global IDs.
class prima_swa_map {
public:
    class reservation {
        friend class prima_swa_map;
        std::shared_ptr<const uint8_t> owner_;
        uint64_t generation_;
        uint32_t head_, count_;
        std::vector<int32_t> rows_, globals_;
        std::vector<bool> seen_;

        reservation(std::shared_ptr<const uint8_t> owner, uint64_t generation,
                    uint32_t head, uint32_t count, std::vector<int32_t> rows, std::vector<int32_t> globals) :
            owner_(std::move(owner)), generation_(generation), head_(head), count_(count),
            rows_(std::move(rows)), globals_(std::move(globals)), seen_(globals_.size(), false) {}
    public:
        uint32_t head() const { return head_; }
        uint32_t count() const { return count_; }
    };

private:
    std::vector<int32_t> global_to_physical_;
    std::vector<int32_t> physical_to_global_;
    const std::shared_ptr<const uint8_t> identity_ = std::make_shared<const uint8_t>(0);
    uint64_t generation_ = 0;
    uint32_t head_ = 0;
    uint32_t used_ = 0;

    void require_generation() const {
        if (generation_ == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("SWA row map generation exhausted");
        }
    }

    // Validate before publication. Allocation or invalid input leaves the map
    // untouched, including its outstanding reservation generation.
    std::vector<int32_t> reverse(const std::vector<int32_t> & rows) const {
        if (rows.size() != physical_to_global_.size()) {
            throw std::invalid_argument("SWA row map capacity mismatch");
        }
        std::vector<int32_t> result(global_to_physical_.size(), -1);
        for (uint32_t physical = 0; physical < rows.size(); ++physical) {
            const int32_t global = rows[physical];
            if (global == -1) continue;
            if (global < 0 || static_cast<size_t>(global) >= result.size() || result[global] != -1) {
                throw std::invalid_argument("SWA row map has invalid or duplicate global cell");
            }
            result[global] = static_cast<int32_t>(physical);
        }
        return result;
    }

    void publish(std::vector<int32_t> rows, uint32_t head) {
        require_generation();
        if (head >= capacity()) throw std::invalid_argument("SWA row map head out of range");
        auto globals = reverse(rows);
        const auto used = static_cast<uint32_t>(std::count_if(rows.begin(), rows.end(),
            [](int32_t global) { return global >= 0; }));
        physical_to_global_.swap(rows);
        global_to_physical_.swap(globals);
        head_ = head;
        used_ = used;
        ++generation_;
    }

public:
    prima_swa_map(uint32_t global_cells, uint32_t physical_cells) {
        if (!global_cells || !physical_cells || global_cells > INT32_MAX || physical_cells > INT32_MAX) {
            throw std::invalid_argument("Invalid SWA row map dimensions");
        }
        global_to_physical_.assign(global_cells, -1);
        physical_to_global_.assign(physical_cells, -1);
    }
    prima_swa_map(const prima_swa_map &) = delete;
    prima_swa_map & operator=(const prima_swa_map &) = delete;
    prima_swa_map(prima_swa_map &&) = delete;
    prima_swa_map & operator=(prima_swa_map &&) = delete;

    uint32_t capacity() const { return static_cast<uint32_t>(physical_to_global_.size()); }
    uint32_t head() const { return head_; }
    uint32_t used() const { return used_; }
    int32_t physical(uint32_t global) const { return global_to_physical_.at(global); }
    int32_t global(uint32_t physical) const { return physical_to_global_.at(physical); }
    const std::vector<int32_t> & rows() const { return physical_to_global_; }
    uint32_t extent() const {
        for (uint32_t end = capacity(); end > 0; --end) {
            if (physical_to_global_[end - 1] >= 0) return end;
        }
        return 0;
    }

    // Predicate must retain a cell while ANY of its shared sequences needs it.
    // For prefill, evaluate retention at the incoming batch's earliest position.
    template<class Reclaimable>
    std::unique_ptr<reservation> prepare(uint32_t count, Reclaimable reclaimable) const {
        if (!count || count > capacity()) return nullptr;
        require_generation();
        auto rows = physical_to_global_;
        auto globals = global_to_physical_;
        std::vector<bool> available(capacity(), false);
        for (uint32_t physical = 0; physical < capacity(); ++physical) {
            const int32_t global = physical_to_global_[physical];
            available[physical] = global < 0 || reclaimable(static_cast<uint32_t>(global));
            if (global >= 0 && available[physical]) {
                rows[physical] = -1;
                globals[global] = -1;
            }
        }
        // Start at the next row, wrap the SEARCH, never wrap a batch's storage.
        for (uint64_t step = 0; step < capacity(); ++step) {
            const uint32_t start = static_cast<uint32_t>((uint64_t(head_) + step) % capacity());
            if (count > capacity() - start) {
                step += capacity() - start - 1;
                continue;
            }
            bool fits = true;
            for (uint32_t offset = 0; offset < count; ++offset) {
                if (!available[start + offset]) {
                    // Every intervening start overlaps this occupied row.
                    // Skip them rather than rescanning the same long span.
                    step += offset;
                    fits = false;
                    break;
                }
            }
            if (fits) return std::unique_ptr<reservation>(new reservation(identity_, generation_, start, count,
                std::move(rows), std::move(globals)));
        }
        return nullptr;
    }

    // A rejected/stale ticket does not mutate the map. Input IDs must already
    // name the full cache's successful reservation, in incoming token order.
    bool commit(reservation & ticket, const std::vector<uint32_t> & globals) {
        if (ticket.owner_ != identity_ || ticket.generation_ != generation_ || globals.size() != ticket.count_) {
            return false;
        }
        std::fill(ticket.seen_.begin(), ticket.seen_.end(), false);
        for (uint32_t global : globals) {
            if (global >= ticket.seen_.size() || ticket.seen_[global]) return false;
            ticket.seen_[global] = true;
        }
        // All storage was allocated during preparation, before full-cache
        // admission. The sealed ticket guarantees the destination rows are free.
        for (uint32_t global : globals) {
            const int32_t old = ticket.globals_[global];
            if (old >= 0) ticket.rows_[old] = -1;
            ticket.globals_[global] = -1;
        }
        for (uint32_t i = 0; i < ticket.count_; ++i) {
            ticket.rows_[ticket.head_ + i] = static_cast<int32_t>(globals[i]);
            ticket.globals_[globals[i]] = static_cast<int32_t>(ticket.head_ + i);
        }
        used_ = static_cast<uint32_t>(std::count_if(ticket.rows_.begin(), ticket.rows_.end(),
            [](int32_t global) { return global >= 0; }));
        physical_to_global_.swap(ticket.rows_);
        global_to_physical_.swap(ticket.globals_);
        head_ = (ticket.head_ + ticket.count_) % capacity();
        ++generation_;
        return true;
    }

    void clear() { publish(std::vector<int32_t>(capacity(), -1), 0); }
    void restore(const std::vector<int32_t> & rows, uint32_t head) { publish(rows, head); }

    // Full-cache defragmentation supplies a complete old-global -> new-global
    // permutation. Compact tensor bytes remain at their physical row offsets.
    void remap_global(const std::vector<uint32_t> & old_to_new) {
        if (old_to_new.size() != global_to_physical_.size()) {
            throw std::invalid_argument("SWA global permutation size mismatch");
        }
        std::vector<bool> seen(old_to_new.size(), false);
        for (uint32_t global : old_to_new) {
            if (global >= seen.size() || seen[global]) {
                throw std::invalid_argument("Invalid SWA global permutation");
            }
            seen[global] = true;
        }
        auto rows = physical_to_global_;
        for (auto & global : rows) if (global >= 0) global = static_cast<int32_t>(old_to_new[global]);
        publish(std::move(rows), head_);
    }
};
