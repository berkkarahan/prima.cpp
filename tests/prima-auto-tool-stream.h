#pragma once

static bool prima_auto_tool_stream_contract() {
    const std::string source = "{% if tools %}{{ tools|tojson }}{% endif %}"
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
        "{% for c in m.tool_calls %}<tool_call>\n{\"name\":\"{{ c.function.name }}\",\"arguments\":{{ c.function.arguments|tojson }}}\n</tool_call>{% endfor %}"
        "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
        {"tools", json::array({{{"type", "function"}, {"function", {{"name", "echo"}, {"parameters", {
            {"type", "object"}, {"properties", {{"text", {{"type", "string"}}}}}, {"required", json::array({"text"})}}}}}}})},
        {"chat_template_kwargs", {{"enable_thinking", false}}}};
    const auto request = oaicompat_completion_params_parse(nullptr, body, source, true);
    const std::string arguments = "{\"text\":\"hello\"}";
    const std::string call = "<tool_call>\n{\"name\":\"echo\",\"arguments\":" + arguments + "}\n</tool_call>";
    for (auto api : {prima_api::chat, prima_api::responses, prima_api::messages}) {
        for (const auto & output : {std::string("hello"), call + "\n" + call, std::string("I will call.\n") + call}) {
            prima_protocol_stream stream(api, body, request, "fixed", 7);
            std::vector<json> events;
            for (const char c : output) {
                const auto part = stream.consume({{"content", std::string(1, c)}, {"stop", false}});
                events.insert(events.end(), part.begin(), part.end());
            }
            const auto terminal = stream.consume({{"content", output}, {"stop", true}, {"stopped_eos", true},
                {"tokens_evaluated", 7}, {"tokens_predicted", 19}});
            events.insert(events.end(), terminal.begin(), terminal.end());
            if (!stream.consume({{"stop", true}}).empty()) return false;
            std::map<size_t, std::string> args;
            std::string text;
            for (const auto & event : events) {
                if (event.contains("error") || json_value(event, "type", std::string()) == "error") {
                    std::cerr << event.dump() << '\n'; return false;
                }
                if (api == prima_api::chat) {
                    const auto & delta = event["choices"][0]["delta"];
                    text += json_value(delta, "content", std::string());
                    if (delta.contains("tool_calls")) for (const auto & tc : delta["tool_calls"])
                        args[tc.at("index").get<size_t>()] += tc["function"]["arguments"].get<std::string>();
                } else if (api == prima_api::responses) {
                    if (event["type"] == "response.function_call_arguments.delta") args[event["output_index"].get<size_t>()] += event["delta"].get<std::string>();
                    if (event["type"] == "response.output_text.delta") text += event["delta"].get<std::string>();
                } else if (event["type"] == "content_block_delta") {
                    if (event["delta"]["type"] == "input_json_delta") args[event["index"].get<size_t>()] += event["delta"]["partial_json"].get<std::string>();
                    if (event["delta"]["type"] == "text_delta") text += event["delta"]["text"].get<std::string>();
                }
            }
            if (output == "hello") { if (text != "hello" || !args.empty()) return false; }
            else if (output.find("I will call.") == 0) {
                if (text != "I will call.\n" || args.size() != 1 || args.begin()->second != arguments) return false;
            } else if (!text.empty() || args.size() != 2 || args[0] != arguments || args[1] != arguments) return false;
        }
        prima_protocol_stream partial(api, body, request, "partial", 7);
        partial.consume({{"content", call.substr(0, call.size() - 15)}, {"stop", false}});
        const auto limited = partial.consume({{"stop", true}, {"stopped_limit", true}, {"tokens_evaluated", 7}, {"tokens_predicted", 2}});
        if (limited.empty() || limited.back().contains("error") || json_value(limited.back(), "type", std::string()) == "error") return false;
        if (api == prima_api::chat && limited.back()["choices"][0]["finish_reason"] != "length") return false;
        if (api == prima_api::responses && limited.back()["type"] != "response.incomplete") return false;
        if (api == prima_api::messages && limited[limited.size()-2]["delta"]["stop_reason"] != "max_tokens") return false;
        prima_protocol_stream failed(api, body, request, "failed", 7);
        failed.consume({{"content", call.substr(0, 20)}, {"stop", false}});
        failed.error({{"message", "native failed"}});
        if (!failed.consume({{"stop", true}}).empty()) return false;
    }
    return true;
}
