#pragma once

static bool prima_tool_history_contract() {
    const json call = {{"type", "function_call"}, {"call_id", "call_a"}, {"name", "echo"}, {"arguments", "{\"text\":\"hello\"}"}};
    const json output = {{"type", "function_call_output"}, {"call_id", "call_a"},
        {"output", json::array({{{"type", "input_text"}, {"text", "hello"}}, {{"type", "input_text"}, {"text", "world"}}})}};
    json response_input = {{"input", json::array({call, output})}};
    auto chat = prima_protocol_request(prima_api::responses, response_input);
    if (chat["messages"].size() != 2 || chat["messages"][0]["tool_calls"][0]["id"] != "call_a" ||
        chat["messages"][1]["tool_call_id"] != "call_a" || chat["messages"][1]["content"] != "hello\nworld") return false;
    response_input["input"].insert(response_input["input"].begin() + 1, call);
    if (prima_protocol_request(prima_api::responses, response_input)["messages"][0]["tool_calls"].size() != 2) return false;
    const json anthropic = {{"max_tokens", 20}, {"messages", json::array({
        {{"role", "assistant"}, {"content", json::array({{{"type", "text"}, {"text", "Calling"}},
            {{"type", "tool_use"}, {"id", "call_a"}, {"name", "echo"}, {"input", {{"text", "hello"}}}}})}},
        {{"role", "user"}, {"content", json::array({{{"type", "tool_result"}, {"tool_use_id", "call_a"},
            {"is_error", true}, {"content", json::array({{{"type", "text"}, {"text", "hello"}}, {{"type", "text"}, {"text", "world"}}})}}})}}
    })}};
    chat = prima_protocol_request(prima_api::messages, anthropic);
    if (chat["messages"].size() != 2 || chat["messages"][0]["tool_calls"][0]["id"] != "call_a" ||
        chat["messages"][1]["content"] != "hello\nworld") return false;
    auto normalized = prima_normalize_history(chat["messages"]);
    normalized[0]["content"] = nullptr;
    if (prima_normalize_history(normalized)[0]["content"] != "") return false;
    const std::string source = "{% for m in messages %}{% if m.tool_calls %}{{ m.tool_calls[0].function.arguments | tojson }}"
        "{% else %}{{ m.content }}{% endif %}{% endfor %}";
    const auto rendered = prima_render_jinja(nullptr, source, normalized, false, json::object(), nullptr);
    if (rendered.find("hello") == std::string::npos || rendered.find("world") == std::string::npos) return false;
    try {
        prima_render_jinja(nullptr, "{% for m in messages %}{{ m.content }}{% endfor %}", normalized, false, json::object(), nullptr);
        return false;
    } catch (const std::runtime_error &) {}
    for (const auto bad : {json{{"role", "tool"}, {"content", "result"}},
            json{{"role", "user"}, {"content", json::array({{{"type", "image_url"}, {"image_url", "file"}}})}},
            json{{"role", "assistant"}, {"tool_calls", json::array({{{"id", "call"}, {"type", "function"},
                {"function", {{"name", "echo"}, {"arguments", "invalid JSON"}}}}})}}}) {
        try { prima_normalize_history(json::array({bad})); return false; } catch (const std::exception &) {}
    }
    return true;
}
