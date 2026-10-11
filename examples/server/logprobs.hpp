#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

struct prima_logprob_scores {
    double selected;
    std::vector<std::pair<int, double>> top;
};

// Probabilities describe the full pre-sampling vocabulary, including a selected
// grammar-forced token outside the top alternatives. Reporting never clips sampling.
inline prima_logprob_scores prima_logprobs(const float * logits, size_t count, int selected, size_t top) {
    if (!logits || !count || selected < 0 || (size_t)selected >= count) {
        throw std::invalid_argument("Invalid logprob vocabulary or selected token");
    }
    double maximum = -INFINITY;
    for (size_t i = 0; i < count; ++i) {
        if (std::isnan(logits[i]) || logits[i] == INFINITY) throw std::runtime_error("Invalid model logits");
        maximum = std::max(maximum, (double)logits[i]);
    }
    if (!std::isfinite(maximum)) throw std::runtime_error("No finite model logits");
    double total = 0;
    for (size_t i = 0; i < count; ++i) total += std::exp((double)logits[i] - maximum);
    const double normalizer = maximum + std::log(total);
    prima_logprob_scores result{std::max(-9999.0, (double)logits[selected] - normalizer), {}};
    top = std::min(top, count);
    if (top) {
        std::vector<int> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
            return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
        });
        for (size_t i = 0; i < top; ++i) result.top.push_back({order[i], std::max(-9999.0, (double)logits[order[i]] - normalizer)});
    }
    return result;
}
