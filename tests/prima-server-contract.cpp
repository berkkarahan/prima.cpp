#include "../examples/server/utils.hpp"
#include <iostream>

int main() {
    const json request = {{"model", "assessment"}};
    json result = {{"content", "hello"}, {"stopped_word", false}, {"stopped_eos", false},
        {"stopped_limit", true}, {"tokens_predicted", 1}, {"tokens_evaluated", 2}};
    if (format_final_response_oaicompat(request, result, "test")["choices"][0]["finish_reason"] != "length") return 1;
    result["stopped_word"] = true;
    result["stopped_limit"] = false;
    if (format_final_response_oaicompat(request, result, "test")["choices"][0]["finish_reason"] != "stop") return 2;
    result["stopped_word"] = false;
    result["stopped_eos"] = true;
    if (format_final_response_oaicompat(request, result, "test")["choices"][0]["finish_reason"] != "stop") return 3;
    const json forced = {{"tools", json::array({{{"type", "function"}, {"function", {{"name", "echo"}, {"parameters", {{"type", "object"}}}}}}})},
        {"tool_choice", {{"type", "function"}, {"function", {{"name", "echo"}}}}}};
    if (prima_forced_tool(forced).at("name") != "echo") return 4;
    json invalid = forced;
    invalid["stream"] = true;
    try { prima_forced_tool(invalid); return 5; } catch (const std::runtime_error &) {}
    invalid = forced;
    invalid["tool_choice"]["function"]["name"] = "missing";
    try { prima_forced_tool(invalid); return 6; } catch (const std::runtime_error &) {}
    json tool_request = request;
    tool_request["__prima_tool_name"] = "echo";
    result["content"] = "{\"text\":\"hello\"}";
    const auto response = format_final_response_oaicompat(tool_request, result, "test");
    if (response["choices"][0]["finish_reason"] != "tool_calls" || response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] != "echo") return 7;
    std::cout << "Completion termination contracts passed\n";
    return 0;
}
