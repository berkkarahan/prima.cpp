#pragma once

#include "llama-grammar.h"
#include "llama-sampling.h"
#include "llama-vocab.h"
#include <cmath>
#include <memory>
#include <stdexcept>

static void prima_check_lazy_grammar() {
    llama_vocab vocab;
    vocab.cache_token_to_piece = {"plain", "<tool_", "call>x", "prefix<tool_call>x", "x", "y", "", "<tool_call>", "\xc3", "\xa9"};
    vocab.n_vocab = uint32_t(vocab.cache_token_to_piece.size());
    vocab.special_eog_ids = {6};
    const char * rule = "root ::= \"<tool_call>\" \"x\"";
    using grammar_ptr = std::unique_ptr<llama_grammar, decltype(&llama_grammar_free_impl)>;
    using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    const auto check = [](bool condition) { if (!condition) throw std::runtime_error("Lazy grammar contract failed"); };
    const auto masked = [&](const llama_grammar & grammar, llama_token token) {
        llama_token_data data = {token, 7.0f, 0.0f};
        llama_token_data_array array = {&data, 1, -1, false};
        llama_grammar_apply_impl(grammar, &array);
        return std::isinf(data.logit) && data.logit < 0;
    };
    const auto make = [&]() {
        grammar_ptr grammar(llama_grammar_init_impl(&vocab, rule, "root"), llama_grammar_free_impl);
        check(bool(grammar));
        llama_grammar_set_lazy(*grammar, {"<tool_call>"}, {});
        return grammar;
    };
    auto split = make();
    check(!masked(*split, 0) && !masked(*split, 6));
    llama_grammar_accept_impl(*split, 0);
    llama_grammar_accept_impl(*split, 1);
    grammar_ptr copy(llama_grammar_clone_impl(*split), llama_grammar_free_impl);
    llama_grammar_accept_impl(*split, 2);
    check(!split->awaiting_trigger && masked(*split, 5) && !masked(*split, 6));
    check(copy->awaiting_trigger && !masked(*copy, 5));
    llama_grammar_accept_impl(*copy, 2);
    check(!copy->awaiting_trigger && !masked(*copy, 6));
    auto middle = make();
    llama_grammar_accept_impl(*middle, 3);
    check(!middle->awaiting_trigger && !masked(*middle, 6));
    auto captured = make();
    llama_grammar_set_lazy(*captured, {"^prefix(<tool_call>.*)$"}, {});
    llama_grammar_accept_impl(*captured, 3);
    check(!captured->awaiting_trigger && !masked(*captured, 6));
    auto token = make();
    llama_grammar_set_lazy(*token, {}, {7});
    llama_grammar_accept_impl(*token, 7);
    check(masked(*token, 5) && masked(*token, 6) && !masked(*token, 4));
    llama_grammar_accept_impl(*token, 4);
    llama_grammar_accept_impl(*token, 6);
    grammar_ptr unicode(llama_grammar_init_impl(&vocab, "root ::= \"<tool_call>\" \"\xc3\xa9\"", "root"), llama_grammar_free_impl);
    llama_grammar_set_lazy(*unicode, {"<tool_call>"}, {});
    llama_grammar_accept_impl(*unicode, 7);
    llama_grammar_accept_impl(*unicode, 8);
    check(masked(*unicode, 6) && !masked(*unicode, 9));
    llama_grammar_accept_impl(*unicode, 9);
    check(!masked(*unicode, 6));
    sampler_ptr sampler(llama_sampler_init_grammar_impl(vocab, rule, "root", true, {"<tool_call>"}, {}), llama_sampler_free);
    llama_sampler_accept(sampler.get(), 1);
    sampler_ptr cloned(llama_sampler_clone(sampler.get()), llama_sampler_free);
    llama_sampler_accept(cloned.get(), 2);
    llama_sampler_reset(cloned.get());
    llama_sampler_accept(cloned.get(), 0); // Reset must return to unconstrained text.
    llama_sampler_accept(cloned.get(), 3);
    llama_sampler_accept(sampler.get(), 2); // Clone/reset must not change the original.
    for (const auto & patterns : std::vector<std::vector<std::string>>{{}, {"["}}) {
        bool refused = false;
        try {
            sampler_ptr invalid(llama_sampler_init_grammar_impl(vocab, rule, "root", true, patterns, {}), llama_sampler_free);
        } catch (const std::exception &) { refused = true; }
        check(refused);
    }
    bool refused = false;
    try {
        sampler_ptr invalid(llama_sampler_init_grammar_impl(vocab, "", "root", true, {"trigger"}, {}), llama_sampler_free);
    } catch (const std::exception &) { refused = true; }
    check(refused);
    grammar_ptr nullable(llama_grammar_init_impl(&vocab, "root ::= (maybe space)+\nmaybe ::= | \"x\"\nspace ::= | \" \"", "root"), llama_grammar_free_impl);
    check(!nullable); // An indirect nullable loop must refuse instead of overflowing.
}
