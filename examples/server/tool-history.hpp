#pragma once

static std::string prima_history_string(const json & value, const char * key) {
    if (!value.contains(key) || !value.at(key).is_string() || value.at(key).get<std::string>().empty())
        throw std::runtime_error(std::string("Tool history requires non-empty ") + key);
    return value.at(key).get<std::string>();
}

static json prima_history_call(const std::string & id, const std::string & name, const json & arguments) {
    if (!arguments.is_string() && !arguments.is_object()) throw std::runtime_error("Tool arguments must be JSON text or an object");
    if (arguments.is_string()) (void) json::parse(arguments.get<std::string>());
    return {{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", arguments}}}};
}

static std::string prima_history_text(const json & value) {
    if (value.is_null()) return "";
    if (value.is_string()) return value.get<std::string>();
    if (!value.is_array()) throw std::runtime_error("Tool result must be text or text blocks");
    std::string result;
    for (const auto & part : value) {
        const auto type = part.at("type").get<std::string>();
        if (type != "text" && type != "input_text" && type != "output_text")
            throw std::runtime_error("Non-text tool results require multimodal support");
        if (!result.empty()) result += '\n';
        result += part.at("text").get<std::string>();
    }
    return result;
}

static void prima_append_history_call(json & messages, const json & call) {
    if (!messages.empty() && messages.back().at("role") == "assistant") {
        if (!messages.back().contains("tool_calls")) messages.back()["tool_calls"] = json::array();
        messages.back()["tool_calls"].push_back(call);
    } else messages.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({call})}});
}

static json prima_normalize_history(const json & raw) {
    if (!raw.is_array()) throw std::runtime_error("Messages must be an array");
    json messages = raw;
    for (auto & message : messages) {
        const auto role = prima_history_string(message, "role");
        if (role != "system" && role != "developer" && role != "user" && role != "assistant" && role != "tool")
            throw std::runtime_error("Unsupported message role");
        if (role == "developer") message["role"] = "system";
        if (message.contains("tool_calls")) {
            if (role != "assistant" || !message.at("tool_calls").is_array())
                throw std::runtime_error("Only assistant messages may contain function tool calls");
            for (auto & call : message["tool_calls"]) {
                if (call.at("type") != "function") throw std::runtime_error("Only function tool history is supported");
                const auto & function = call.at("function");
                call = prima_history_call(prima_history_string(call, "id"), prima_history_string(function, "name"), function.at("arguments"));
            }
            if (!message.contains("content") || message["content"].is_null()) message["content"] = "";
        }
        if (role == "tool") prima_history_string(message, "tool_call_id");
        if (!message.contains("content")) throw std::runtime_error("Message requires content or assistant tool calls");
        if (message["content"].is_null()) message["content"] = "";
        if (!message["content"].is_string() && !message["content"].is_array())
            throw std::runtime_error("Message content must be text or text blocks");
        if (message["content"].is_array()) for (auto & part : message["content"]) {
            const auto type = prima_history_string(part, "type");
            if (type != "text" && type != "input_text" && type != "output_text")
                throw std::runtime_error("Non-text content requires multimodal support");
            if (!part.at("text").is_string()) throw std::runtime_error("Text content must be a string");
            part["type"] = "text";
        }
    }
    return messages;
}
