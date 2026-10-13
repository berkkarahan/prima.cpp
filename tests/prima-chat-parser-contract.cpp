#include "prima-auto-chat.h"
#include "prima-chat.h"
#include "prima-lazy-grammar-contract.h"
#include "sampling.h"
#include "profiler.h"
#include <algorithm>
#include <iostream>

using json = nlohmann::ordered_json;

int main(int argc, char ** argv) {
    try {
        std::cerr << "Checking lazy grammar primitives\n";
        prima_check_lazy_grammar();
        // Synthetic contract template: fixture-owned source, with a tool format shared
        // by Qwen-style templates. An explicit file exercises the actual model preset.
        const std::string synthetic =
            "{% if tools %}{{ tools|tojson }}{% endif %}"
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            "{% if m.tool_calls %}{% for c in m.tool_calls %}<tool_call>\n"
            "{\"name\":\"{{ c.function.name }}\",\"arguments\":"
            "{% if c.function.arguments is string %}{{ c.function.arguments }}{% else %}{{ c.function.arguments|tojson }}{% endif %}"
            "}\n</tool_call>{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        const auto source = argc >= 2 ? prima_read_template(argv[1]) : synthetic;
        using model_ptr = std::unique_ptr<llama_model, decltype(&llama_free_model)>;
        model_ptr model(nullptr, llama_free_model);
        if (argc == 3) {
            llama_backend_init();
            std::cerr << "Loading contract vocabulary\n";
            auto params = llama_model_default_params();
            params.vocab_only = true;
            model.reset(llama_load_model_from_file(argv[2], params));
            if (!model) throw std::runtime_error("Contract vocabulary model load failed");
            std::unique_ptr<llama_model_loader, decltype(&device_free_model_metadata)> metadata(
                llama_model_load(argv[2], model.get(), &params), device_free_model_metadata);
            if (!metadata) throw std::runtime_error("Contract vocabulary metadata load failed");
            std::cerr << "Vocabulary " << llama_n_vocab(model.get()) << " BOS " << llama_token_bos(model.get()) << " EOS " << llama_token_eos(model.get()) << '\n';
        }
        const json body = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
            {"chat_template_kwargs", {{"enable_thinking", false}}},
            {"tools", json::array({{{"type", "function"}, {"function", {{"name", "echo"},
                {"parameters", {{"type", "object"}, {"properties", {{"text", {{"type", "string"}}}}}, {"required", json::array({"text"})}}}}}}})}};
        std::cerr << "Preparing baseline parser\n";
        auto prepared = prima_auto_chat_prepare(model.get(), source, body);
        if (!prepared.at("grammar_lazy").get<bool>() || prepared.at("grammar").get<std::string>().empty() ||
            prepared.at("grammar_triggers").empty() || prepared.at("parser").get<std::string>().empty()) return 1;
        const std::string output = "<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"text\":\"hello\"}}\n</tool_call>";
        const auto call = prima_auto_chat_parse(prepared, output, false);
        if (call.at("tool_calls").size() != 1 || call.at("tool_calls").at(0).at("function").at("name") != "echo" ||
            json::parse(call.at("tool_calls").at(0).at("function").at("arguments").get<std::string>()) != json{{"text", "hello"}}) return 2;
        if (prima_auto_chat_parse(prepared, "hello", false).at("content") != "hello") return 3;
        if (prima_auto_chat_parse(prepared, output + "\n" + output, false).at("tool_calls").size() != 2) return 5;
        for (size_t n = 0; n < output.size(); ++n) (void) prima_auto_chat_parse(prepared, output.substr(0, n), true);
        auto required = body;
        required["tool_choice"] = "required";
        const auto prepared_required = prima_auto_chat_prepare(model.get(), source, required);
        if (prepared_required.at("grammar_lazy").get<bool>()) return 4;
        for (const auto & params : {prepared, prepared_required}) {
            std::cerr << "Checking generated grammar\n";
            auto * grammar = llama_grammar_init_impl(nullptr, params.at("grammar").get<std::string>().c_str(), "root");
            if (!grammar) throw std::runtime_error("Generated baseline grammar cannot be parsed by Prima");
            llama_grammar_free_impl(grammar);
        }
        if (model) {
            gpt_sampler_params params;
            params.grammar = prepared.at("grammar").get<std::string>();
            params.grammar_lazy = true;
            for (const auto & trigger : prepared.at("grammar_triggers")) {
                // The checked Qwen fixture emits a literal WORD trigger.
                if (trigger.at("type") != 1) throw std::runtime_error("Unexpected Qwen trigger type");
                std::string escaped;
                for (const auto c : trigger.at("value").get<std::string>()) {
                    if (std::string("\\^$.|?*+()[]{}").find(c) != std::string::npos) escaped += '\\';
                    escaped += c;
                }
                params.grammar_trigger_patterns.push_back(escaped);
            }
            using sampler_ptr = std::unique_ptr<gpt_sampler, decltype(&gpt_sampler_free)>;
            sampler_ptr sampler(gpt_sampler_init(model.get(), params), gpt_sampler_free);
            for (const auto token : llama_tokenize(model.get(), "ordinary answer\n" + output, false, true))
                gpt_sampler_accept(sampler.get(), token, true);
            gpt_sampler_accept(sampler.get(), llama_token_eos(model.get()), true);
            required["chat_template_kwargs"]["enable_thinking"] = true;
            const auto thinking = prima_auto_chat_prepare(model.get(), source, required);
            gpt_sampler_params regular;
            regular.grammar = thinking.at("grammar").get<std::string>();
            regular.grammar_prefill_tokens = llama_tokenize(model.get(), thinking.at("generation_prompt").get<std::string>(), false, true);
            sampler_ptr reasoned(gpt_sampler_init(model.get(), regular), gpt_sampler_free);
            const std::string reasoned_output = "<think>\nwhy\n</think>\n\n" + output;
            const auto reasoning = prima_auto_chat_parse(thinking, reasoned_output, false);
            if (reasoning.at("reasoning_content").get<std::string>().find("why") == std::string::npos ||
                reasoning.at("tool_calls").size() != 1) return 6;
            for (int repeat = 0; repeat < 2; ++repeat) {
                for (const auto token : llama_tokenize(model.get(), reasoned_output, false, true))
                    gpt_sampler_accept(reasoned.get(), token, true);
                gpt_sampler_accept(reasoned.get(), llama_token_eos(model.get()), true);
                gpt_sampler_reset(reasoned.get());
            }
            gpt_sampler_reset(sampler.get());
            for (const auto token : llama_tokenize(model.get(), "ordinary answer", false, true))
                gpt_sampler_accept(sampler.get(), token, true);
            gpt_sampler_accept(sampler.get(), llama_token_eos(model.get()), true);
        }
        // Regression: JSON-native tool formats must fall back to content when the
        // complete output mentions a tool marker but is not a valid tool call
        // (ordinary text may legally contain the marker substring).
        const std::string marker_text = "foo <tool_call> bar";
        if (prima_auto_chat_parse(prepared, marker_text, false).at("content") != marker_text) return 7;
        // A tool marker followed by invalid tool JSON previously failed the whole
        // parse; with tool_choice auto it must degrade to plain content.
        const std::string broken_tool = "foo <tool_call>\n{oops}";
        if (prima_auto_chat_parse(prepared, broken_tool, false).at("content") != broken_tool) return 8;
        // tool_choice required keeps rejecting output that is not a valid tool call.
        bool required_threw = false;
        try {
            (void) prima_auto_chat_parse(prepared_required, broken_tool, false);
        } catch (const std::exception &) {
            required_threw = true;
        }
        if (!required_threw) return 9;
        // Regression: template workaround patches must be reflected in preserved
        // tokens so the tokenizer preserves the patched markers the parser uses.
        const std::string deepseek =
            "{% if tools %}{% for tool in tools %}"
            "{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'}}"
            "{% endfor %}{% endif %}"
            "{% for m in messages %}<im_start>{{ m.role }}\n{{ m.content }}"
            "{% if m.tool_calls %}{% for c in m.tool_calls %}<tool_call>\n"
            "{\"name\":\"{{ c.function.name }}\",\"arguments\":{{ c.function.arguments|tojson }}}\n</tool_call>"
            "{% endfor %}{% endif %}"
            "<im_end>\n{% endfor %}"
            "{% if add_generation_prompt %}<｜Assistant｜>{% endif %}";
        std::cerr << "Preparing patched-marker parser\n";
        const auto patched = prima_auto_chat_prepare(model.get(), deepseek, body);
        std::cerr << "Patched-marker parser prepared\n";
        const auto & kept = patched.at("preserved_tokens");
        if (std::find(kept.begin(), kept.end(), "<｜tool▁calls▁begin｜>") == kept.end()) return 10;
        if (std::find(kept.begin(), kept.end(), "<｜tool▁call▁end｜>") == kept.end()) return 11;
        std::cout << "Baseline automatic tool/text parsing, partial prefixes and grammar policy verified\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
