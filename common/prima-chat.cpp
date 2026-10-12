#include "prima-chat.h"
#include "jinja/parser.h"
#include "jinja/caps.h"
#include "json.h"
#include <fstream>
#include <memory>
#include <stdexcept>

static constexpr size_t template_limit = 64 * 1024;

std::string prima_template_source(const llama_model * model, const std::string & override_source) {
    if (!override_source.empty()) return override_source;
    if (!model) throw std::runtime_error("Jinja rendering requires a template");
    int size = llama_model_meta_val_str(model, "tokenizer.chat_template", nullptr, 0);
    if (size <= 0) return "chatml";
    if (size > int(template_limit)) throw std::runtime_error("Model chat template exceeds 64 KiB");
    std::string source(size_t(size) + 1, '\0');
    if (llama_model_meta_val_str(model, "tokenizer.chat_template", &source[0], source.size()) != size)
        throw std::runtime_error("Cannot read model chat template");
    source.resize(size_t(size));
    return source;
}

void prima_validate_jinja(const std::string & source) {
    if (source.empty() || source.size() > template_limit)
        throw std::runtime_error("Jinja template must contain 1 to 65536 bytes");
    jinja::lexer lexer;
    jinja::parse_from_tokens(lexer.tokenize(source));
}

std::string prima_read_template(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open chat template file");
    std::string source(template_limit + 1, '\0');
    file.read(&source[0], source.size());
    source.resize(size_t(file.gcount()));
    if (file.bad()) throw std::runtime_error("Cannot read chat template file");
    prima_validate_jinja(source);
    return source;
}

static std::string special_token(const llama_model * model, llama_token token) {
    if (!model || token < 0) return "";
    std::string result(256, '\0');
    int size = llama_token_to_piece(model, token, &result[0], result.size(), 0, true);
    if (size < 0) {
        result.resize(size_t(-size));
        size = llama_token_to_piece(model, token, &result[0], result.size(), 0, true);
    }
    if (size < 0) throw std::runtime_error("Cannot render template special token");
    result.resize(size_t(size));
    return result;
}

std::string prima_render_jinja(const llama_model * model, const std::string & source,
    const nlohmann::ordered_json & messages, bool add_generation_prompt,
    const nlohmann::ordered_json & kwargs, const nlohmann::ordered_json & tools) {
    if (!kwargs.is_null() && !kwargs.is_object())
        throw std::runtime_error("chat_template_kwargs must be an object");
    // Parsed programs are immutable; execution always owns a fresh context.
    thread_local std::string cached_source;
    thread_local std::unique_ptr<jinja::program> cached_program;
    thread_local jinja::caps cached_caps;
    if (!cached_program || cached_source != source) {
        prima_validate_jinja(source);
        jinja::lexer lexer;
        auto parsed = std::make_unique<jinja::program>(jinja::parse_from_tokens(lexer.tokenize(source)));
        cached_caps = jinja::caps_get(*parsed);
        cached_source = source;
        cached_program = std::move(parsed);
    }
    nlohmann::ordered_json input = kwargs.is_object() ? kwargs : nlohmann::ordered_json::object();
    input["messages"] = messages;
    for (auto & message : input["messages"]) {
        if (message.contains("tool_calls") && !message["tool_calls"].empty() && !cached_caps.supports_tool_calls)
            throw std::runtime_error("Selected Jinja template cannot render tool-call history");
        if (message.contains("tool_calls")) for (auto & call : message["tool_calls"]) {
            auto & arguments = call["function"]["arguments"];
            if (cached_caps.supports_object_arguments && arguments.is_string())
                arguments = nlohmann::ordered_json::parse(arguments.get<std::string>());
            else if (!cached_caps.supports_object_arguments && !arguments.is_string()) arguments = arguments.dump();
        }
        if (!message.contains("content")) continue;
        auto & content = message["content"];
        if (cached_caps.supports_string_content && !cached_caps.supports_typed_content && content.is_array()) {
            std::string joined;
            for (const auto & part : content) {
                if (part.value("type", "") != "text") continue;
                if (!joined.empty()) joined += '\n';
                joined += part.value("text", "");
            }
            content = joined;
        } else if (!cached_caps.supports_string_content && cached_caps.supports_typed_content && content.is_string()) {
            content = nlohmann::ordered_json::array({{{"type", "text"}, {"text", content}}});
        }
    }
    input["bos_token"] = special_token(model, model ? llama_token_bos(model) : -1);
    input["eos_token"] = special_token(model, model ? llama_token_eos(model) : -1);
    input["add_generation_prompt"] = add_generation_prompt;
    if (!input.contains("enable_thinking")) input["enable_thinking"] = true;
    if (!tools.is_null()) input["tools"] = tools;
    jinja::context context(source);
    if (input.contains("preserve_reasoning") && input["preserve_reasoning"].is_boolean())
        jinja::caps_apply_preserve_reasoning(context, input["preserve_reasoning"].get<bool>());
    if (input.contains("reasoning_effort") && input["reasoning_effort"].is_string())
        jinja::caps_apply_reasoning_effort(context, input["reasoning_effort"].get<std::string>());
    jinja::global_from_json(context, common_json::parse(input.dump()), false);
    jinja::runtime runtime(context);
    return runtime.gather_string_parts(runtime.execute(*cached_program))->as_string().str();
}
