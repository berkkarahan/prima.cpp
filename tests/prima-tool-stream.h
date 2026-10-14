#pragma once

static bool prima_tool_stream_contract() {
    const json request = {{"model", "test"}, {"__prima_tool_name", "echo"}, {"__prima_tool_call_id", "call_fixed"}};
    const std::string arguments = "{\"text\":\"hello\\nworld\"}";
    for (const auto api : {prima_api::chat, prima_api::responses, prima_api::messages}) {
        for (bool limited : {false, true}) {
            prima_protocol_stream stream(api, request, request, "stream", 7);
            std::vector<json> all;
            const std::string expected = limited ? arguments.substr(0, 9) : arguments;
            const std::vector<std::string> fragments = limited ? std::vector<std::string>{expected} :
                std::vector<std::string>{arguments.substr(0, 9), arguments.substr(9, 4), arguments.substr(13)};
            for (const auto & delta : fragments) {
                const auto events = stream.consume({{"content", delta}, {"stop", false}});
                all.insert(all.end(), events.begin(), events.end());
            }
            const json result = {{"content", "aggregate must not be duplicated"}, {"stop", true},
                {"stopped_limit", limited}, {"stopped_eos", !limited}, {"tokens_predicted", 12}, {"tokens_evaluated", 7}};
            const auto terminal = stream.consume(result);
            all.insert(all.end(), terminal.begin(), terminal.end());
            if (!stream.consume(result).empty()) return false;
            std::string collected;
            if (api == prima_api::chat) {
                const auto & first = all.front().at("choices").at(0).at("delta");
                if (first.at("role") != "assistant" || first.at("tool_calls").at(0).at("id") != "call_fixed") return false;
                for (const auto & event : all) {
                    const auto & delta = event.at("choices").at(0).at("delta");
                    if (delta.contains("content")) return false;
                    if (delta.contains("tool_calls")) collected += delta.at("tool_calls").at(0).at("function").at("arguments").get<std::string>();
                }
                if (all.back().at("choices").at(0).at("finish_reason") != (limited ? "length" : "tool_calls") ||
                    all.back().at("usage").at("total_tokens") != 19) return false;
            } else if (api == prima_api::responses) {
                int sequence = 0;
                for (const auto & event : all) {
                    if (event.at("sequence_number") != sequence++) return false;
                    if (event.at("type") == "response.function_call_arguments.delta") {
                        if (event.at("item_id") != "fc_call_fixed") return false;
                        collected += event.at("delta").get<std::string>();
                    }
                    if (event.at("type") == "response.output_text.delta") return false;
                }
                const auto & response = all.back().at("response");
                const auto & item = response.at("output").at(0);
                if (response.at("status") != (limited ? "incomplete" : "completed") || item.at("call_id") != "call_fixed" ||
                    item.at("id") != "fc_call_fixed" || item.at("arguments") != expected) return false;
            } else {
                if (all.at(1).at("content_block").at("type") != "tool_use" || all.at(1).at("content_block").at("id") != "call_fixed") return false;
                for (const auto & event : all) if (event.at("type") == "content_block_delta") {
                    if (event.at("delta").at("type") != "input_json_delta") return false;
                    collected += event.at("delta").at("partial_json").get<std::string>();
                }
                if (terminal.at(1).at("delta").at("stop_reason") != (limited ? "max_tokens" : "tool_use")) return false;
            }
            if (collected != expected) return false;
            prima_protocol_stream errored(api, request, request, "error", 7);
            errored.consume({{"content", "{"}, {"stop", false}});
            errored.error({{"error", {{"message", "native failed"}}}});
            if (!errored.consume(result).empty()) return false;
        }
    }
    // Empty terminal completion still emits one stable call and one terminal reason.
    prima_protocol_stream empty(prima_api::chat, request, request, "empty", 7);
    auto events = empty.consume({{"stop", true}, {"stopped_limit", true}, {"tokens_predicted", 0}, {"tokens_evaluated", 7}});
    if (events.size() != 2 || events.back().at("choices").at(0).at("finish_reason") != "length") return false;
    auto scored_request = request;
    scored_request["__prima_logprobs"] = true;
    prima_protocol_stream scored(prima_api::chat, scored_request, scored_request, "scored", 7);
    auto score_events = scored.consume({{"stop", false}, {"content", ""}, {"completion_probabilities", json::array({{
        {"logprob_entry", {{"token", "x"}, {"bytes", json::array({120})}, {"logprob", -1}, {"top_logprobs", json::array()}}}}})}});
    return score_events.size() == 2 && score_events.back().at("choices").at(0).at("delta").empty() &&
        score_events.back().at("choices").at(0).at("logprobs").at("content").size() == 1;
}
