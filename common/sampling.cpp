#include "sampling.h"
#include "reasoning-budget.h"

#include "common.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct gpt_sampler {
    gpt_sampler_params params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void set_logits(struct llama_context * ctx, int idx) {
        const auto * logits = llama_get_logits_ith(ctx, idx);

        const int n_vocab = llama_n_vocab(llama_get_model(ctx));

        cur.resize(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

std::string gpt_sampler_params::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            top_k, tfs_z, top_p, min_p, typ_p, temp,
            mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}

static void sampling_checked(bool success) {
    if (!success) throw std::runtime_error(llama_sampler_last_error());
}

struct gpt_sampler * gpt_sampler_init(const struct llama_model * model, const struct gpt_sampler_params & params) {
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    std::vector<const char *> patterns;
    for (const auto & pattern : params.grammar_trigger_patterns) patterns.push_back(pattern.c_str());
    auto * grammar = params.grammar_lazy ? llama_sampler_init_grammar_lazy_patterns(
        model, params.grammar.c_str(), "root", patterns.data(), patterns.size(),
        params.grammar_trigger_tokens.data(), params.grammar_trigger_tokens.size()) :
        llama_sampler_init_grammar(model, params.grammar.c_str(), "root");
    if (!grammar) throw std::invalid_argument("Failed to initialize the requested grammar");
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> grammar_owner(grammar, llama_sampler_free);
    auto * chain = llama_sampler_chain_init(lparams);
    if (!chain) throw std::runtime_error(llama_sampler_last_error());
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> chain_owner(chain, llama_sampler_free);

    std::unique_ptr<gpt_sampler, decltype(&gpt_sampler_free)> result_owner(new gpt_sampler {
        /* .params = */ params,
        /* .grmr   = */ grammar,
        /* .rbudget = */ nullptr,
        /* .chain  = */ chain,
        /* .prev   = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    }, gpt_sampler_free);
    grammar_owner.release();
    chain_owner.release();
    auto * result = result_owner.get();

    llama_sampler_chain_add(result->chain,
            llama_sampler_init_logit_bias(
                llama_n_vocab(model),
                params.logit_bias.size(),
                params.logit_bias.data()));

    llama_sampler_chain_add(result->chain,
            llama_sampler_init_penalties(
                llama_n_vocab  (model),
                llama_token_eos(model),
                llama_token_nl (model),
                params.penalty_last_n,
                params.penalty_repeat,
                params.penalty_freq,
                params.penalty_present,
                params.penalize_nl,
                params.ignore_eos));

    if (params.temp > 0.0f) {
        if (params.mirostat == 0) {
            for (const auto & cnstr : params.samplers) {
                switch (cnstr) {
                    case GPT_SAMPLER_TYPE_TOP_K:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_top_k    (params.top_k));
                        break;
                    case GPT_SAMPLER_TYPE_TOP_P:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_top_p    (params.top_p, params.min_keep));
                        break;
                    case GPT_SAMPLER_TYPE_MIN_P:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_min_p    (params.min_p, params.min_keep));
                        break;
                    case GPT_SAMPLER_TYPE_TFS_Z:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_tail_free(params.tfs_z, params.min_keep));
                        break;
                    case GPT_SAMPLER_TYPE_TYPICAL_P:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_typical  (params.typ_p, params.min_keep));
                        break;
                    case GPT_SAMPLER_TYPE_TEMPERATURE:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_temp_ext (params.temp, params.dynatemp_range, params.dynatemp_exponent));
                        break;
                    default:
                        GGML_ASSERT(false && "unknown sampler type");
                }
            }
            llama_sampler_chain_add(result->chain, llama_sampler_init_softmax());
            llama_sampler_chain_add(result->chain, llama_sampler_init_dist(params.seed));
        } else if (params.mirostat == 1) {
            llama_sampler_chain_add(result->chain, llama_sampler_init_temp(params.temp));
            llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat(llama_n_vocab(model), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
        } else if (params.mirostat == 2) {
            llama_sampler_chain_add(result->chain, llama_sampler_init_temp(params.temp));
            llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
        } else {
            GGML_ASSERT(false && "unknown mirostat version");
        }
    } else {
        if (params.n_probs > 0) {
            // some use cases require to sample greedily, but still obtain the probabilities of the top tokens
            // ref: https://github.com/ggerganov/llama.cpp/pull/9605
            //
            // the following will not produce exactly the same probs as applyging softmax to the full vocabulary, but
            // it is much faster, since we avoid sorting all tokens and should give a good approximation
            llama_sampler_chain_add(result->chain, llama_sampler_init_top_k(params.n_probs));
            llama_sampler_chain_add(result->chain, llama_sampler_init_softmax());
        }
        llama_sampler_chain_add(result->chain, llama_sampler_init_greedy());
    }

    try {
        if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() &&
                (params.grammar_lazy || params.reasoning_budget_tokens >= 0)) {
            result->rbudget = common_reasoning_budget_init(model, {params.reasoning_budget_start},
                params.reasoning_budget_end, params.reasoning_budget_forced,
                params.reasoning_budget_tokens < 0 ? INT32_MAX : params.reasoning_budget_tokens);
            if (!result->rbudget) throw std::runtime_error("Reasoning budget sampler initialization failed");
            for (const auto token : params.reasoning_prefill_tokens)
                sampling_checked(llama_sampler_accept_checked(result->rbudget, token));
        }
        if (!params.grammar_lazy) for (const auto token : params.grammar_prefill_tokens)
            sampling_checked(llama_sampler_accept_checked(result->grmr, token));
    } catch (...) {
        throw;
    }
    return result_owner.release();
}

void gpt_sampler_free(struct gpt_sampler * gsmpl) {
    if (gsmpl) {
        llama_sampler_free(gsmpl->grmr);
        llama_sampler_free(gsmpl->rbudget);

        llama_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

static bool grammar_should_apply(const gpt_sampler * sampler) {
    if (!sampler->params.grammar_lazy || !sampler->rbudget) return true;
    const auto state = common_reasoning_budget_get_state(sampler->rbudget);
    return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
}

void gpt_sampler_accept(struct gpt_sampler * gsmpl, llama_token token, bool accept_grammar) {
    const bool apply_grammar = accept_grammar && grammar_should_apply(gsmpl);
    if (accept_grammar && gsmpl->rbudget) {
        sampling_checked(llama_sampler_accept_checked(gsmpl->rbudget, token));
        if (!apply_grammar && common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE) {
            const auto * end = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end) for (const auto token : *end) sampling_checked(llama_sampler_accept_checked(gsmpl->grmr, token));
        }
    }
    if (apply_grammar) {
        sampling_checked(llama_sampler_accept_checked(gsmpl->grmr, token));
    }

    sampling_checked(llama_sampler_accept_checked(gsmpl->chain, token));

    gsmpl->prev.push_back(token);
}

void gpt_sampler_reset(struct gpt_sampler * gsmpl) {
    if (gsmpl->rbudget) {
        sampling_checked(llama_sampler_reset_checked(gsmpl->rbudget));
        for (const auto token : gsmpl->params.reasoning_prefill_tokens)
            sampling_checked(llama_sampler_accept_checked(gsmpl->rbudget, token));
    }
    sampling_checked(llama_sampler_reset_checked(gsmpl->grmr));
    if (!gsmpl->params.grammar_lazy) for (const auto token : gsmpl->params.grammar_prefill_tokens)
        sampling_checked(llama_sampler_accept_checked(gsmpl->grmr, token));

    sampling_checked(llama_sampler_reset_checked(gsmpl->chain));
}

struct gpt_sampler * gpt_sampler_clone(gpt_sampler * gsmpl) {
    auto * grammar = llama_sampler_clone(gsmpl->grmr);
    if (!grammar) throw std::runtime_error(llama_sampler_last_error());
    auto * chain = llama_sampler_clone(gsmpl->chain);
    if (!chain) {
        const std::string error = llama_sampler_last_error();
        llama_sampler_free(grammar);
        throw std::runtime_error(error);
    }
    try {
        auto * budget = gsmpl->rbudget ? llama_sampler_clone(gsmpl->rbudget) : nullptr;
        if (gsmpl->rbudget && !budget) throw std::runtime_error(llama_sampler_last_error());
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> owner(budget, llama_sampler_free);
        auto * result = new gpt_sampler{gsmpl->params, grammar, budget, chain, gsmpl->prev, gsmpl->cur, gsmpl->cur_p};
        owner.release();
        result->cur_p.data = result->cur.data();
        return result;
    } catch (...) {
        llama_sampler_free(grammar);
        llama_sampler_free(chain);
        throw;
    }
}

void gpt_perf_print(const struct llama_context * ctx, const struct gpt_sampler * gsmpl) {
    // TODO: measure grammar performance

    if (gsmpl) {
        llama_perf_sampler_print(gsmpl->chain);
    }
    if (ctx) {
        llama_perf_context_print(ctx);
    }
}

llama_token gpt_sampler_sample(struct gpt_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    const bool apply_grammar = grammar_should_apply(gsmpl);
    if (gsmpl->rbudget) sampling_checked(llama_sampler_apply_checked(gsmpl->rbudget, &cur_p));
    if (grammar_first && apply_grammar) {
        sampling_checked(llama_sampler_apply_checked(grmr, &cur_p));
    }

    sampling_checked(llama_sampler_apply_checked(chain, &cur_p));

    if (cur_p.selected < 0 || size_t(cur_p.selected) >= cur_p.size || !std::isfinite(cur_p.data[cur_p.selected].logit))
        throw std::runtime_error("Sampling produced no permitted finite token");

    const llama_token id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !apply_grammar) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        sampling_checked(llama_sampler_apply_checked(grmr, &single_token_data_array));

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);
    if (gsmpl->rbudget) sampling_checked(llama_sampler_apply_checked(gsmpl->rbudget, &cur_p));

    sampling_checked(llama_sampler_apply_checked(grmr,  &cur_p));
    sampling_checked(llama_sampler_apply_checked(chain, &cur_p));

    if (cur_p.selected < 0 || size_t(cur_p.selected) >= cur_p.size || !std::isfinite(cur_p.data[cur_p.selected].logit))
        throw std::runtime_error("Grammar re-sampling produced no permitted finite token");

    return cur_p.data[cur_p.selected].id;
}

std::vector<llama_token> gpt_sampler_sample_and_accept_n(struct gpt_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = gpt_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        gpt_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = gpt_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        gpt_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> gpt_sampler_sample_and_accept_n(struct gpt_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return gpt_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t gpt_sampler_get_seed(const struct gpt_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

// helpers

llama_token_data_array * gpt_sampler_get_candidates(struct gpt_sampler * gsmpl) {
    return &gsmpl->cur_p;
}

llama_token gpt_sampler_last(const struct gpt_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string gpt_sampler_print(const struct gpt_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ") + llama_sampler_name(smpl) + " ";
    }

    return result;
}

std::string gpt_sampler_prev_str(gpt_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += llama_token_to_piece(ctx_main, id);
    }

    return result;
}

char gpt_sampler_type_to_chr(enum gpt_sampler_type cnstr) {
    switch (cnstr) {
        case GPT_SAMPLER_TYPE_TOP_K:       return 'k';
        case GPT_SAMPLER_TYPE_TFS_Z:       return 'f';
        case GPT_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case GPT_SAMPLER_TYPE_TOP_P:       return 'p';
        case GPT_SAMPLER_TYPE_MIN_P:       return 'm';
        case GPT_SAMPLER_TYPE_TEMPERATURE: return 't';
        default : return '?';
    }
}

std::string gpt_sampler_type_to_str(enum gpt_sampler_type cnstr) {
    switch (cnstr) {
        case GPT_SAMPLER_TYPE_TOP_K:       return "top_k";
        case GPT_SAMPLER_TYPE_TFS_Z:       return "tfs_z";
        case GPT_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case GPT_SAMPLER_TYPE_TOP_P:       return "top_p";
        case GPT_SAMPLER_TYPE_MIN_P:       return "min_p";
        case GPT_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        default : return "";
    }
}

std::vector<gpt_sampler_type> gpt_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, gpt_sampler_type> sampler_canonical_name_map {
        { "top_k",       GPT_SAMPLER_TYPE_TOP_K },
        { "top_p",       GPT_SAMPLER_TYPE_TOP_P },
        { "typ_p",       GPT_SAMPLER_TYPE_TYPICAL_P },
        { "min_p",       GPT_SAMPLER_TYPE_MIN_P },
        { "tfs_z",       GPT_SAMPLER_TYPE_TFS_Z },
        { "temperature", GPT_SAMPLER_TYPE_TEMPERATURE },
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, gpt_sampler_type> sampler_alt_name_map {
        { "top-k",       GPT_SAMPLER_TYPE_TOP_K },
        { "top-p",       GPT_SAMPLER_TYPE_TOP_P },
        { "nucleus",     GPT_SAMPLER_TYPE_TOP_P },
        { "typical-p",   GPT_SAMPLER_TYPE_TYPICAL_P },
        { "typical",     GPT_SAMPLER_TYPE_TYPICAL_P },
        { "typ-p",       GPT_SAMPLER_TYPE_TYPICAL_P },
        { "typ",         GPT_SAMPLER_TYPE_TYPICAL_P },
        { "min-p",       GPT_SAMPLER_TYPE_MIN_P },
        { "tfs-z",       GPT_SAMPLER_TYPE_TFS_Z },
        { "tfs",         GPT_SAMPLER_TYPE_TFS_Z },
        { "temp",        GPT_SAMPLER_TYPE_TEMPERATURE },
    };

    std::vector<gpt_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        auto sampler = sampler_canonical_name_map.find(name);
        if (sampler != sampler_canonical_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            if (allow_alt_names) {
                sampler = sampler_alt_name_map.find(name);
                if (sampler != sampler_alt_name_map.end()) {
                    samplers.push_back(sampler->second);
                }
            }
        }
    }

    return samplers;
}

std::vector<gpt_sampler_type> gpt_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, gpt_sampler_type> sampler_name_map = {
        { gpt_sampler_type_to_chr(GPT_SAMPLER_TYPE_TOP_K),       GPT_SAMPLER_TYPE_TOP_K },
        { gpt_sampler_type_to_chr(GPT_SAMPLER_TYPE_TFS_Z),       GPT_SAMPLER_TYPE_TFS_Z },
        { gpt_sampler_type_to_chr(GPT_SAMPLER_TYPE_TYPICAL_P),   GPT_SAMPLER_TYPE_TYPICAL_P },
        { gpt_sampler_type_to_chr(GPT_SAMPLER_TYPE_TOP_P),       GPT_SAMPLER_TYPE_TOP_P },
        { gpt_sampler_type_to_chr(GPT_SAMPLER_TYPE_MIN_P),       GPT_SAMPLER_TYPE_MIN_P },
        { gpt_sampler_type_to_chr(GPT_SAMPLER_TYPE_TEMPERATURE), GPT_SAMPLER_TYPE_TEMPERATURE }
    };

    std::vector<gpt_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        }
    }

    return samplers;
}
