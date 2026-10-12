#pragma once

// Native protocol adapters share the server's completion scheduler and cancellation.
// Stored state, multimodal and automatic tool behavior remain separate contracts.
enum class prima_api { chat, responses, messages };

static bool prima_protocol_sse(httplib::DataSink & sink, const std::string & event, const json & data) {
    const auto wire = "event: " + event + "\ndata: " + data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    return sink.write(wire.data(), wire.size());
}

static std::string prima_text(const json & value) {
    if (value.is_string()) return value.get<std::string>();
    if (!value.is_array()) throw std::runtime_error("Message content must be text or text blocks");
    std::string result;
    for (const auto & part : value) {
        const auto type = json_value(part, "type", std::string());
        if (type != "text" && type != "input_text" && type != "output_text") {
            throw std::runtime_error("This Prima protocol requires text content; multimodal/tool-turn blocks are not implemented");
        }
        result += part.at("text").get<std::string>();
    }
    return result;
}

static json prima_protocol_request(prima_api api, const json & body) {
    if (!body.is_object()) throw std::runtime_error("Request must be a JSON object");
    if (api == prima_api::chat) return body;
    for (const auto & item : body.items()) {
        if (item.key().compare(0, 2, "__") == 0) throw std::runtime_error("Internal completion parameters are reserved");
    }
    if (body.contains("previous_response_id") && !body.at("previous_response_id").is_null()) {
        throw std::runtime_error("Prima Responses is stateless; supply the complete input instead of previous_response_id");
    }
    if (json_value(body, "store", false) || json_value(body, "background", false)) {
        throw std::runtime_error("Prima does not implement stored or background responses");
    }
    json chat = json::object();
    for (const auto * key : {"model", "temperature", "top_p", "top_k", "min_p", "seed", "stream",
            "stop", "frequency_penalty", "presence_penalty", "repeat_penalty", "cache_prompt", "chat_template_kwargs", "response_format", "parallel_tool_calls", "lora",
            "reasoning_format", "reasoning_effort", "reasoning_budget_tokens", "thinking_budget_tokens", "reasoning_budget_message", "enable_thinking"}) {
        if (body.contains(key)) chat[key] = body.at(key);
    }
    json messages = json::array();
    const auto system_key = api == prima_api::responses ? "instructions" : "system";
    if (body.contains(system_key) && !body.at(system_key).is_null()) {
        messages.push_back({{"role", "system"}, {"content", prima_text(body.at(system_key))}});
    }
    if (api == prima_api::responses) {
        if (body.contains("reasoning") && !body.at("reasoning").is_null()) {
            const auto & reasoning = body.at("reasoning");
            if (!reasoning.is_object()) throw std::runtime_error("Responses reasoning must be an object");
            if (reasoning.contains("effort")) chat["reasoning_effort"] = reasoning.at("effort").get<std::string>();
        }
        const auto & input = body.at("input");
        if (input.is_string()) messages.push_back({{"role", "user"}, {"content", input}});
        else if (input.is_array()) {
            for (const auto & item : input) {
                const auto type = json_value(item, "type", std::string("message"));
                if (type == "reasoning") {
                    const auto & summary = item.at("summary");
                    if (!summary.is_array()) throw std::runtime_error("Reasoning summary must be an array");
                    std::string text;
                    for (const auto & part : summary) {
                        if (part.at("type") != "summary_text") throw std::runtime_error("Unsupported reasoning summary type");
                        text += part.at("text").get<std::string>();
                    }
                    messages.push_back({{"role", "assistant"}, {"content", ""}, {"reasoning_content", text}});
                    continue;
                }
                if (type == "function_call") {
                    prima_append_history_call(messages, prima_history_call(prima_history_string(item, "call_id"),
                        prima_history_string(item, "name"), item.at("arguments")));
                    continue;
                }
                if (type == "function_call_output") {
                    messages.push_back({{"role", "tool"}, {"tool_call_id", prima_history_string(item, "call_id")},
                        {"content", prima_history_text(item.at("output"))}});
                    continue;
                }
                if (type != "message") throw std::runtime_error("Unsupported Responses input item");
                std::string role = item.at("role").get<std::string>();
                if (role == "developer") role = "system";
                if (role != "system" && role != "user" && role != "assistant") throw std::runtime_error("Unsupported Responses message role");
                const auto text = prima_text(item.at("content"));
                if (role == "assistant" && !messages.empty() && messages.back().contains("reasoning_content") &&
                    messages.back().at("content") == "" && !messages.back().contains("tool_calls")) messages.back()["content"] = text;
                else messages.push_back({{"role", role}, {"content", text}});
            }
        } else throw std::runtime_error("Responses input must be text or a message array");
        if (body.contains("max_output_tokens")) chat["max_tokens"] = body.at("max_output_tokens");
        if (body.contains("text") && body.at("text").contains("format")) {
            auto format = body.at("text").at("format");
            if (json_value(format, "type", std::string("text")) == "json_schema") {
                chat["response_format"] = {{"type", "json_schema"}, {"json_schema", format}};
            } else chat["response_format"] = format;
        }
    } else {
        if (!body.at("messages").is_array()) throw std::runtime_error("Messages must be an array");
        for (const auto & item : body.at("messages")) {
            const auto role = item.at("role").get<std::string>();
            if (role != "user" && role != "assistant") throw std::runtime_error("Unsupported Messages role");
            const auto & content = item.at("content");
            if (!content.is_array()) {
                messages.push_back({{"role", role}, {"content", prima_text(content)}});
                continue;
            }
            json text = json::array(), calls = json::array(), results = json::array();
            std::string reasoning;
            for (const auto & block : content) {
                const auto type = json_value(block, "type", std::string());
                if (type == "text") text.push_back(block);
                else if (type == "thinking" && role == "assistant") reasoning += block.at("thinking").get<std::string>();
                else if (type == "tool_use" && role == "assistant") {
                    calls.push_back(prima_history_call(prima_history_string(block, "id"),
                        prima_history_string(block, "name"), block.at("input")));
                } else if (type == "tool_result" && role == "user") {
                    results.push_back({{"role", "tool"}, {"tool_call_id", prima_history_string(block, "tool_use_id")},
                        {"content", block.contains("content") ? prima_history_text(block.at("content")) : ""}});
                } else throw std::runtime_error("Unsupported Messages content block or tool role");
            }
            if (!text.empty() || !calls.empty() || !reasoning.empty()) {
                json message = {{"role", role}, {"content", text.empty() ? json("") : text}};
                if (!calls.empty()) message["tool_calls"] = calls;
                if (!reasoning.empty()) message["reasoning_content"] = reasoning;
                messages.push_back(message);
            }
            for (const auto & result : results) messages.push_back(result);
        }
        chat["max_tokens"] = body.at("max_tokens");
        if (body.contains("stop_sequences")) chat["stop"] = body.at("stop_sequences");
        if (body.contains("thinking")) {
            const auto & thinking = body.at("thinking");
            if (!thinking.is_object()) throw std::runtime_error("Messages thinking must be an object");
            const auto type = json_value(thinking, "type", std::string());
            if (type != "enabled" && type != "disabled" && type != "adaptive") throw std::runtime_error("Unsupported Messages thinking type");
            chat["chat_template_kwargs"]["enable_thinking"] = type != "disabled";
            if (type == "disabled") chat["reasoning_effort"] = "none";
            if (type == "enabled" && thinking.contains("budget_tokens")) chat["reasoning_budget_tokens"] = thinking.at("budget_tokens");
        }
    }
    if (messages.empty()) throw std::runtime_error("At least one input message is required");
    chat["messages"] = messages;
    if (body.contains("tools")) {
        if (!body.at("tools").is_array()) throw std::runtime_error("Tools must be an array");
        chat["tools"] = json::array();
        for (const auto & tool : body.at("tools")) {
            if (api == prima_api::responses && json_value(tool, "type", std::string()) != "function") {
                throw std::runtime_error("Only function tools are supported");
            }
            const auto parameters = api == prima_api::responses ? "parameters" : "input_schema";
            chat["tools"].push_back({{"type", "function"}, {"function", {
                {"name", tool.at("name")}, {"description", json_value(tool, "description", std::string())},
                {"parameters", tool.at(parameters)}}}});
        }
    }
    if (body.contains("tool_choice")) {
        const auto & choice = body.at("tool_choice");
        const auto type = choice.is_object() ? json_value(choice, "type", std::string()) : std::string();
        if (type == "function" || type == "tool") {
            chat["tool_choice"] = {{"type", "function"}, {"function", {{"name", choice.at("name")}}}};
        } else if (type == "none") chat["tool_choice"] = "none";
        else if (type == "auto") chat["tool_choice"] = "auto";
        else if (type == "any") chat["tool_choice"] = "required";
        else chat["tool_choice"] = choice;
        if (choice.is_object() && choice.contains("disable_parallel_tool_use"))
            chat["parallel_tool_calls"] = !choice.at("disable_parallel_tool_use").get<bool>();
    }
    return chat;
}

static json prima_protocol_response(prima_api api, const json & chat, const json & request,
                                   const std::string & id, const std::string & stopping_word = "") {
    if (api == prima_api::chat) return chat;
    const auto & choice = chat.at("choices").at(0);
    const auto & message = choice.at("message");
    const auto & usage = chat.at("usage");
    const auto finish = choice.at("finish_reason").get<std::string>();
    const std::string text = message.at("content").is_string() ? message.at("content").get<std::string>() : "";
    const bool tool = message.contains("tool_calls") && !message.at("tool_calls").empty();
    const bool limited = finish == "length";
    if (api == prima_api::messages) {
        json blocks = json::array();
        if (message.contains("reasoning_content") && message.at("reasoning_content").is_string() && message.at("reasoning_content") != "")
            blocks.push_back({{"type", "thinking"}, {"thinking", message.at("reasoning_content")}});
        if (tool && !limited) {
            if (!text.empty()) blocks.push_back({{"type", "text"}, {"text", text}});
            for (const auto & call : message.at("tool_calls")) {
                const auto & fn = call.at("function");
                blocks.push_back({{"type", "tool_use"}, {"id", call.at("id")},
                    {"name", fn.at("name")}, {"input", json::parse(fn.at("arguments").get<std::string>())}});
            }
        } else {
            // A truncated JSON argument string is not a completed tool action.
            const auto partial = tool ? message.at("tool_calls").at(0).at("function").at("arguments") : json(text);
            blocks.push_back({{"type", "text"}, {"text", partial}});
        }
        return {{"id", "msg_" + id}, {"type", "message"}, {"role", "assistant"}, {"model", chat.at("model")},
            {"content", blocks}, {"stop_reason", limited ? "max_tokens" : tool ? "tool_use" : stopping_word.empty() ? "end_turn" : "stop_sequence"},
            {"stop_sequence", stopping_word.empty() ? json(nullptr) : json(stopping_word)},
            {"usage", {{"input_tokens", usage.at("prompt_tokens")}, {"output_tokens", usage.at("completion_tokens")}}}};
    }
    json output = json::array();
    if (message.contains("reasoning_content") && message.at("reasoning_content").is_string() && message.at("reasoning_content") != "")
        output.push_back({{"id", "rs_" + id}, {"type", "reasoning"}, {"summary", json::array({{{"type", "summary_text"}, {"text", message.at("reasoning_content")}}})}});
    if (tool && !text.empty()) output.push_back({{"id", "msg_" + id}, {"type", "message"}, {"role", "assistant"},
        {"status", limited ? "incomplete" : "completed"},
        {"content", json::array({{{"type", "output_text"}, {"text", text}, {"annotations", json::array()}}})}});
    if (tool) {
        for (const auto & call : message.at("tool_calls")) {
            const auto & fn = call.at("function");
            output.push_back({{"id", "fc_" + call.at("id").get<std::string>()}, {"type", "function_call"},
                {"status", limited ? "incomplete" : "completed"}, {"call_id", call.at("id")},
                {"name", fn.at("name")}, {"arguments", fn.at("arguments")}});
        }
    } else output.push_back({{"id", "msg_" + id}, {"type", "message"}, {"role", "assistant"},
        {"status", limited ? "incomplete" : "completed"},
        {"content", json::array({{{"type", "output_text"}, {"text", text}, {"annotations", json::array()}}})}});
    return {{"id", "resp_" + id}, {"object", "response"}, {"created_at", chat.at("created")}, {"completed_at", std::time(nullptr)},
        {"status", limited ? "incomplete" : "completed"}, {"error", nullptr},
        {"incomplete_details", limited ? json{{"reason", "max_output_tokens"}} : json(nullptr)},
        {"model", chat.at("model")}, {"output", output}, {"store", false}, {"background", false},
        {"parallel_tool_calls", json_value(request, "parallel_tool_calls", true)}, {"tools", json_value(request, "tools", json::array())},
        {"tool_choice", json_value(request, "tool_choice", json("auto"))},
        {"instructions", json_value(request, "instructions", json(nullptr))},
        {"metadata", json_value(request, "metadata", json::object())},
        {"text", json_value(request, "text", json{{"format", {{"type", "text"}}}})},
        {"usage", {{"input_tokens", usage.at("prompt_tokens")}, {"output_tokens", usage.at("completion_tokens")},
            {"total_tokens", usage.at("total_tokens")}, {"input_tokens_details", {{"cached_tokens", 0}}},
            {"output_tokens_details", {{"reasoning_tokens", 0}}}}}};
}

#include "auto-tools.hpp"

struct prima_protocol_stream {
    prima_api api;
    json request, chat_request;
    std::string id, text;
    int input_tokens;
    int sequence = 0;
    bool started = false, finished = false;
    std::unique_ptr<prima_auto_tool_stream> automatic;

    bool tool() const { return chat_request.contains("__prima_tool_name"); }
    std::string call_id() const { return json_value(chat_request, "__prima_tool_call_id", "call_" + id); }
    std::string item_id() const { return "fc_" + call_id(); }

    json chat_event(json delta, json finish = nullptr, const json * result = nullptr) const {
        json event = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", std::time(nullptr)},
            {"model", json_value(chat_request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
            {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish},
                {"logprobs", result && json_value(chat_request, "__prima_logprobs", false) ? oaicompat_logprobs(*result) : json(nullptr)}}})}};
        if (result && !finish.is_null()) {
            const int input = json_value(*result, "tokens_evaluated", 0);
            const int output = json_value(*result, "tokens_predicted", 0);
            event["usage"] = {{"prompt_tokens", input}, {"completion_tokens", output}, {"total_tokens", input + output}};
        }
        return event;
    }

    prima_protocol_stream(prima_api api, json request, json chat_request, std::string id, int input_tokens)
        : api(api), request(std::move(request)), chat_request(std::move(chat_request)), id(std::move(id)), input_tokens(input_tokens) {
        if (this->chat_request.contains("__prima_auto_chat")) automatic.reset(new prima_auto_tool_stream(
            api, this->request, this->chat_request, this->id, input_tokens));
        if (tool() && !this->chat_request.contains("__prima_tool_call_id"))
            this->chat_request["__prima_tool_call_id"] = "call_" + this->id;
    }

    json tagged(const std::string & type, json values = json::object()) {
        values["type"] = type;
        if (api == prima_api::responses) values["sequence_number"] = sequence++;
        return values;
    }

    json error(const json & native) {
        finished = true;
        const auto detail = native.contains("error") ? native.at("error") : native;
        if (api == prima_api::chat) return native.contains("error") ? native : json{{"error", detail}};
        const auto message = json_value(detail, "message", std::string("Native inference failed"));
        if (api == prima_api::messages) return tagged("error", {{"error", {{"type", "api_error"}, {"message", message}}}});
        return tagged("error", {{"code", "server_error"}, {"message", message}, {"param", nullptr}});
    }

    std::vector<json> consume(const json & result) {
        std::vector<json> events;
        if (finished) return events;
        if (automatic) {
            try {
                events = automatic->consume(result);
                finished = automatic->finished;
                return events;
            } catch (const std::exception & failure) {
                return {error({{"message", failure.what()}})};
            }
        }
        if (api == prima_api::chat && !tool()) {
            events = format_partial_response_oaicompat(result, id);
            finished = json_value(result, "stop", false);
            return events;
        }
        const auto model = json_value(chat_request, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
        if (!started) {
            started = true;
            if (api == prima_api::chat) {
                events.push_back(chat_event({{"role", "assistant"}, {"tool_calls", json::array({{
                    {"index", 0}, {"id", call_id()}, {"type", "function"},
                    {"function", {{"name", chat_request.at("__prima_tool_name")}, {"arguments", ""}}}}})}}));
            } else if (api == prima_api::messages) {
                events.push_back(tagged("message_start", {{"message", {{"id", "msg_" + id}, {"type", "message"},
                    {"role", "assistant"}, {"model", model}, {"content", json::array()}, {"stop_reason", nullptr},
                    {"stop_sequence", nullptr}, {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}}}}}}));
                const json block = tool() ? json{{"type", "tool_use"}, {"id", call_id()},
                    {"name", chat_request.at("__prima_tool_name")}, {"input", json::object()}} : json{{"type", "text"}, {"text", ""}};
                events.push_back(tagged("content_block_start", {{"index", 0}, {"content_block", block}}));
            } else {
                json pending = {{"id", "resp_" + id}, {"object", "response"}, {"created_at", std::time(nullptr)},
                    {"status", "in_progress"}, {"model", model}, {"output", json::array()}, {"usage", nullptr},
                    {"error", nullptr}, {"incomplete_details", nullptr}, {"store", false}};
                events.push_back(tagged("response.created", {{"response", pending}}));
                events.push_back(tagged("response.in_progress", {{"response", pending}}));
                const json item = tool() ? json{{"id", item_id()}, {"type", "function_call"}, {"status", "in_progress"},
                    {"call_id", call_id()}, {"name", chat_request.at("__prima_tool_name")}, {"arguments", ""}} : json{
                    {"id", "msg_" + id}, {"type", "message"}, {"role", "assistant"}, {"status", "in_progress"}, {"content", json::array()}};
                events.push_back(tagged("response.output_item.added", {{"output_index", 0}, {"item", item}}));
                if (!tool()) events.push_back(tagged("response.content_part.added", {{"item_id", "msg_" + id}, {"output_index", 0},
                    {"content_index", 0}, {"part", {{"type", "output_text"}, {"text", ""}, {"annotations", json::array()}}}}));
            }
        }
        // Terminal completion records contain aggregate content, not another delta.
        const bool stop = json_value(result, "stop", false);
        if (!stop) {
            const auto delta = json_value(result, "content", std::string());
            if (!delta.empty()) {
                text += delta;
                if (api == prima_api::chat) events.push_back(chat_event({{"tool_calls", json::array({{
                    {"index", 0}, {"function", {{"arguments", delta}}}}})}}, nullptr, &result));
                else if (api == prima_api::messages) events.push_back(tagged("content_block_delta", {{"index", 0},
                    {"delta", tool() ? json{{"type", "input_json_delta"}, {"partial_json", delta}} : json{{"type", "text_delta"}, {"text", delta}}}}));
                else if (tool()) events.push_back(tagged("response.function_call_arguments.delta", {{"item_id", item_id()},
                    {"output_index", 0}, {"delta", delta}}));
                else events.push_back(tagged("response.output_text.delta", {{"item_id", "msg_" + id},
                    {"output_index", 0}, {"content_index", 0}, {"delta", delta}}));
            } else if (api == prima_api::chat && json_value(chat_request, "__prima_logprobs", false)) {
                // Framing/UTF-8 buffering can consume a scored token without argument bytes.
                // Keep its logprob receipt even while the content delta is empty.
                events.push_back(chat_event(json::object(), nullptr, &result));
            }
            return events;
        }
        finished = true;
        json aggregate = result;
        aggregate["content"] = text;
        auto chat = format_final_response_oaicompat(chat_request, aggregate, id);
        if (api == prima_api::chat) {
            events.push_back(chat_event(json::object(), chat.at("choices").at(0).at("finish_reason"), &result));
            return events;
        }
        auto completed = prima_protocol_response(api, chat, request, id, json_value(result, "stopping_word", std::string()));
        if (api == prima_api::messages) {
            events.push_back(tagged("content_block_stop", {{"index", 0}}));
            events.push_back(tagged("message_delta", {{"delta", {{"stop_reason", completed.at("stop_reason")},
                {"stop_sequence", completed.at("stop_sequence")}}}, {"usage", completed.at("usage")}}));
            events.push_back(tagged("message_stop"));
        } else {
            const auto & item = completed.at("output").at(0);
            if (tool()) events.push_back(tagged("response.function_call_arguments.done", {{"item_id", item_id()},
                {"output_index", 0}, {"arguments", text}}));
            else {
            events.push_back(tagged("response.output_text.done", {{"item_id", "msg_" + id}, {"output_index", 0},
                {"content_index", 0}, {"text", text}}));
            events.push_back(tagged("response.content_part.done", {{"item_id", "msg_" + id}, {"output_index", 0},
                {"content_index", 0}, {"part", item.at("content").at(0)}}));
            }
            events.push_back(tagged("response.output_item.done", {{"output_index", 0}, {"item", item}}));
            events.push_back(tagged(completed.at("status") == "incomplete" ? "response.incomplete" : "response.completed", {{"response", completed}}));
        }
        return events;
    }
};
