#include "../examples/server/utils.hpp"
#include "../examples/server/protocols.hpp"
#include "prima-manifest.h"
#include "../examples/server/structured.hpp"
#include "../src/llama-grammar.h"
#include <chrono>
#include <filesystem>
#include <iostream>

static bool output_placement_contract() {
    const auto path = std::filesystem::temp_directory_path() / ("prima manifest " +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    struct cleanup { std::filesystem::path path; ~cleanup() { std::filesystem::remove(path); } } guard{path};
    json manifest = {{"schema_version", 1}, {"runtime_id", std::string(64, 'a')}, {"total_layers", 4},
        {"split_mode", "layer"}, {"tensor_placements", json::array()}, {"data_port", 9000}, {"signal_port", 10000},
        {"ranks", json::array({{{"rank", 0}, {"address", "127.0.0.1"}, {"ngl", 4}, {"output_gpu", true},
            {"layers", json::array({{{"start", 0}, {"end", 4}}})}, {"tensor_split", json::array({1.0})}}})}};
    const auto apply = [&]() {
        std::ofstream(path, std::ios::binary) << manifest.dump();
        gpt_params params;
        params.rank = 0;
        params.execution_manifest = path.string();
        prima_apply_manifest(params);
        return params.keep_out_in_cuda;
    };
    if (!apply()) return false;
    manifest["ranks"][0].erase("output_gpu");
    if (apply()) return false;
    manifest["ranks"][0]["output_gpu"] = true;
    manifest["ranks"][0]["ngl"] = 0;
    try { apply(); return false; } catch (const std::runtime_error &) {}
    manifest["ranks"][0]["output_gpu"] = false;
    manifest["ranks"][0]["layers"] = json::array();
    manifest["ranks"].push_back({{"rank", 1}, {"address", "127.0.0.2"}, {"ngl", 4}, {"output_gpu", true},
        {"layers", json::array({{{"start", 0}, {"end", 4}}})}, {"tensor_split", json::array({1.0})}});
    try { apply(); return false; } catch (const std::runtime_error &) {}
    return true;
}

int main() {
    if (!output_placement_contract()) return 25;
    const auto framed = prima_json_grammar("root ::= \"root\" root-item?\nroot-item ::= root\nprima-json-value ::= \"keep\"\n");
    if (framed.find("\"root\"") == std::string::npos || framed.find("root-item ::= prima-json-value-1") == std::string::npos) return 26;
    llama_grammar_parser grammar;
    if (!grammar.parse(framed.c_str()) || grammar.rules.empty()) return 27;
    for (const std::string value : {"\"hello\"", "{\"text\":\"```json\\n\",\"nested\":[1,true,null]}", "42", "null"}) {
        const std::string encoded = "```json\n" + value + "\n```";
        if (prima_json_content(encoded) != value || prima_json_content(value) != value) return 28;
        prima_json_stream streamed;
        std::string content;
        for (const char c : encoded) content += streamed.append(std::string(1, c));
        if (content != value) return 29;
    }
    if (prima_json_content("```j") != "" || prima_json_content("```json\n\"hello  ") != "\"hello  ") return 30;
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
    if (!prima_forced_tool({{"tools", json::array()}, {"tool_choice", "auto"}}).is_null()) return 18;
    try { prima_forced_tool({{"tools", "invalid"}}); return 19; } catch (const std::runtime_error &) {}
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
    const json response_request = {{"model", "assessment"}, {"input", json::array({{
        {"role", "developer"}, {"content", json::array({{{"type", "input_text"}, {"text", "system"}}})}},
        {{"role", "user"}, {"content", "hello"}}})}, {"max_output_tokens", 10}};
    const auto parsed_response = prima_protocol_request(prima_api::responses, response_request);
    if (parsed_response.at("messages").at(0).at("role") != "system" || parsed_response.at("max_tokens") != 10) return 8;
    json invalid_response = response_request;
    invalid_response["previous_response_id"] = "old";
    try { prima_protocol_request(prima_api::responses, invalid_response); return 9; } catch (const std::runtime_error &) {}
    invalid_response = response_request;
    invalid_response["input"][1]["content"] = json::array({{{"type", "input_image"}, {"image_url", "x"}}});
    try { prima_protocol_request(prima_api::responses, invalid_response); return 10; } catch (const std::runtime_error &) {}
    const auto response_tool = prima_protocol_response(prima_api::responses, response, response_request, "test");
    if (response_tool.at("output").at(0).at("type") != "function_call" || response_tool.at("usage").at("input_tokens") != 2) return 11;
    const auto message_tool = prima_protocol_response(prima_api::messages, response, request, "test");
    if (message_tool.at("stop_reason") != "tool_use" || message_tool.at("content").at(0).at("input").at("text") != "hello") return 12;
    json truncated_tool = response;
    truncated_tool["choices"][0]["finish_reason"] = "length";
    truncated_tool["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{\"text\":";
    const auto truncated = prima_protocol_response(prima_api::messages, truncated_tool, request, "test");
    if (truncated.at("stop_reason") != "max_tokens" || truncated.at("content").at(0).at("type") != "text") return 20;
    result["content"] = "hello";
    result["stop"] = true;
    const auto stopped = format_final_response_oaicompat(request, result, "test");
    const auto stopped_message = prima_protocol_response(prima_api::messages, stopped, request, "test", "END");
    if (stopped_message.at("stop_reason") != "stop_sequence" || stopped_message.at("stop_sequence") != "END") return 13;
    for (const auto api : {prima_api::responses, prima_api::messages}) {
        prima_protocol_stream stream(api, request, request, "stream", 2);
        const auto first_events = stream.consume({{"content", "hel"}, {"stop", false}});
        if (first_events.empty()) return 14;
        stream.consume({{"content", "lo"}, {"stop", false}});
        const auto terminal = stream.consume(result);
        if (terminal.back().at("type") != (api == prima_api::responses ? "response.completed" : "message_stop")) return 15;
        if (api == prima_api::responses && terminal.back().at("response").at("output").at(0).at("content").at(0).at("text") != "hello") return 16;
        if (!stream.consume(result).empty()) return 17;
        prima_protocol_stream errored(api, request, request, "error", 2);
        const auto error_event = errored.error({{"error", {{"message", "native failed"}}}});
        if (error_event.at("type") != "error") return 21;
        if (api == prima_api::responses && error_event.at("message") != "native failed") return 22;
        if (api == prima_api::messages && error_event.at("error").at("message") != "native failed") return 23;
        if (!errored.consume(result).empty()) return 24;
    }
    std::cout << "Completion termination contracts passed\n";
    return 0;
}
