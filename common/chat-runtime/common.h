#pragma once
#include "../common.h"
#include <map>
#include <sstream>

// Compatibility for baseline format handling, preserving Prima's model/runtime API.
using llama_vocab = llama_model;
enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE, COMMON_REASONING_FORMAT_AUTO,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY, COMMON_REASONING_FORMAT_DEEPSEEK,
};
enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};
struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};
inline const llama_vocab * llama_model_get_vocab(const llama_model * model) { return model; }
inline llama_token llama_vocab_bos(const llama_vocab * model) { return llama_token_bos(model); }
inline llama_token llama_vocab_eos(const llama_vocab * model) { return llama_token_eos(model); }
inline bool llama_vocab_get_add_bos(const llama_vocab * model) { return llama_add_bos_token(model); }
inline bool llama_vocab_get_add_eos(const llama_vocab * model) { return llama_add_eos_token(model); }
inline llama_tokens common_tokenize(const llama_vocab * model, const std::string & text, bool add_special, bool parse_special) {
    return llama_tokenize(model, text, add_special, parse_special);
}
inline std::string common_token_to_piece(const llama_vocab * model, llama_token token, bool special) {
    std::string result(256, '\0');
    int size = llama_token_to_piece(model, token, &result[0], int32_t(result.size()), 0, special);
    if (size < 0) {
        const int64_t required = -int64_t(size);
        if (required > INT32_MAX) throw std::runtime_error("Template token exceeds supported size");
        result.resize(size_t(required));
        size = llama_token_to_piece(model, token, &result[0], int32_t(result.size()), 0, special);
    }
    if (size < 0) throw std::runtime_error("Cannot decode a template token");
    result.resize(size_t(size));
    return result;
}
inline const char * llama_model_chat_template(const llama_model * model, const char * variant) {
    const std::string key = std::string("tokenizer.chat_template") + (variant ? std::string(".") + variant : "");
    const int size = llama_model_meta_val_str(model, key.c_str(), nullptr, 0);
    if (size <= 0) return nullptr;
    if (size > 65536) throw std::runtime_error("Model chat template exceeds 64 KiB");
    thread_local std::map<std::string, std::string> templates;
    auto & value = templates[key];
    value.resize(size_t(size) + 1);
    if (llama_model_meta_val_str(model, key.c_str(), &value[0], size_t(value.size())) != size)
        throw std::runtime_error("Model template read failed");
    value.resize(size_t(size));
    return value.c_str();
}
inline bool string_starts_with(const std::string & text, const std::string & prefix) { return text.compare(0, prefix.size(), prefix) == 0; }
inline bool string_ends_with(const std::string & text, const std::string & suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}
inline std::vector<std::string> string_split(const std::string & text, const std::string & separator) {
    if (separator.empty()) return {text};
    std::vector<std::string> parts;
    size_t start = 0, found;
    while ((found = text.find(separator, start)) != std::string::npos) {
        parts.push_back(text.substr(start, found - start));
        start = found + separator.size();
    }
    parts.push_back(text.substr(start));
    return parts;
}
inline std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) result << separator;
        result << values[i];
    }
    return result.str();
}
inline std::string string_repeat(const std::string & value, size_t count) {
    std::string result;
    for (size_t i = 0; i < count; ++i) result += value;
    return result;
}
