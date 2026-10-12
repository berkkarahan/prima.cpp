#pragma once

// Incremental baseline parsing is shared across the three wire protocols. Raw
// markers never become text deltas; generated IDs are cached by parser call index.
struct prima_auto_tool_stream {
    struct item {
        std::string key, kind, id, call_id, name, value;
        bool closed = false;
    };
    prima_api api;
    json request, prepared_request;
    std::string id, raw;
    int input_tokens, sequence = 0;
    bool started = false, finished = false;
    json previous = {{"role", "assistant"}, {"content", ""}};
    std::vector<std::string> call_ids;
    std::vector<item> items;

    prima_auto_tool_stream(prima_api api, json request, json prepared, std::string id, int input)
        : api(api), request(std::move(request)), prepared_request(std::move(prepared)), id(std::move(id)), input_tokens(input) {}

    json tagged(const std::string & type, json values = json::object()) {
        values["type"] = type;
        if (api == prima_api::responses) values["sequence_number"] = sequence++;
        return values;
    }
    json chat_event(json delta, json finish = nullptr, const json * result = nullptr) const {
        json event = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", std::time(nullptr)},
            {"model", json_value(prepared_request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
            {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish},
                {"logprobs", result && json_value(prepared_request, "__prima_logprobs", false) ? oaicompat_logprobs(*result) : json(nullptr)}}})}};
        if (result && !finish.is_null()) {
            const int input = json_value(*result, "tokens_evaluated", 0), output = json_value(*result, "tokens_predicted", 0);
            event["usage"] = {{"prompt_tokens", input}, {"completion_tokens", output}, {"total_tokens", input + output}};
        }
        return event;
    }
    json wire_item(const item & value, const std::string & status, bool empty = false) const {
        if (value.kind == "tool") return {{"id", value.id}, {"type", "function_call"}, {"status", status},
            {"call_id", value.call_id}, {"name", value.name}, {"arguments", empty ? "" : value.value}};
        if (value.kind == "reasoning") return {{"id", value.id}, {"type", "reasoning"},
            {"summary", json::array()}, {"content", empty ? json::array() : json::array({{{"type", "reasoning_text"}, {"text", value.value}}})},
            {"encrypted_content", ""}, {"status", status}};
        return {{"id", value.id}, {"type", "message"}, {"role", "assistant"}, {"status", status},
            {"content", empty ? json::array() : json::array({{{"type", "output_text"}, {"text", value.value}, {"annotations", json::array()}}})}};
    }
    size_t open(std::vector<json> & events, const std::string & key, const std::string & kind,
                const std::string & call_id = "", const std::string & name = "") {
        for (size_t i = 0; i < items.size(); ++i) if (items[i].key == key) return i;
        const size_t index = items.size();
        items.push_back({key, kind, kind == "tool" ? "fc_" + call_id : kind == "reasoning" ? "rs_" + id : "msg_" + id,
                         call_id, name, "", false});
        if (api == prima_api::messages) {
            const json block = kind == "tool" ? json{{"type", "tool_use"}, {"id", call_id}, {"name", name}, {"input", json::object()}} :
                kind == "reasoning" ? json{{"type", "thinking"}, {"thinking", ""}} : json{{"type", "text"}, {"text", ""}};
            events.push_back(tagged("content_block_start", {{"index", index}, {"content_block", block}}));
        } else if (api == prima_api::responses) {
            events.push_back(tagged("response.output_item.added", {{"output_index", index}, {"item", wire_item(items.back(), "in_progress", true)}}));
            if (kind == "text") events.push_back(tagged("response.content_part.added",
                {{"item_id", items.back().id}, {"output_index", index}, {"content_index", 0},
                 {"part", {{"type", "output_text"}, {"text", ""}, {"annotations", json::array()}}}}));
        }
        return index;
    }
    void delta(std::vector<json> & events, size_t index, const std::string & value) {
        auto & target = items.at(index);
        target.value += value;
        if (value.empty()) return;
        if (api == prima_api::messages) {
            const json change = target.kind == "tool" ? json{{"type", "input_json_delta"}, {"partial_json", value}} :
                target.kind == "reasoning" ? json{{"type", "thinking_delta"}, {"thinking", value}} : json{{"type", "text_delta"}, {"text", value}};
            events.push_back(tagged("content_block_delta", {{"index", index}, {"delta", change}}));
        } else if (api == prima_api::responses) {
            json values = {{"item_id", target.id}, {"output_index", index}, {"delta", value}};
            if (target.kind == "text") values["content_index"] = 0;
            events.push_back(tagged(target.kind == "tool" ? "response.function_call_arguments.delta" :
                target.kind == "reasoning" ? "response.reasoning_text.delta" : "response.output_text.delta", values));
        }
    }
    bool complete_name(const std::string & name) const {
        for (const auto & tool : prepared_request.value("tools", json::array())) if (tool.at("function").at("name") == name) return true;
        return false;
    }
    std::vector<json> consume(const json & result) {
        std::vector<json> events;
        if (finished) return events;
        if (!started) {
            started = true;
            const auto model = json_value(prepared_request, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
            if (api == prima_api::chat) events.push_back(chat_event({{"role", "assistant"}}));
            else if (api == prima_api::messages) events.push_back(tagged("message_start", {{"message", {{"id", "msg_" + id},
                {"type", "message"}, {"role", "assistant"}, {"model", model}, {"content", json::array()},
                {"stop_reason", nullptr}, {"stop_sequence", nullptr}, {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}}}}}}));
            else {
                const json pending = {{"id", "resp_" + id}, {"object", "response"}, {"created_at", std::time(nullptr)},
                    {"status", "in_progress"}, {"model", model}, {"output", json::array()}, {"usage", nullptr},
                    {"error", nullptr}, {"incomplete_details", nullptr}, {"store", false}};
                events.push_back(tagged("response.created", {{"response", pending}}));
                events.push_back(tagged("response.in_progress", {{"response", pending}}));
            }
        }
        const bool stop = json_value(result, "stop", false);
        if (!stop) raw += json_value(result, "content", std::string());
        const bool limited = stop && !json_value(result, "stopped_word", false) && !json_value(result, "stopped_eos", false);
        auto current = prima_auto_message(prepared_request.at("__prima_auto_chat"), raw, !stop || limited, id);
        if (current.contains("tool_calls")) for (size_t i = 0; i < current["tool_calls"].size(); ++i) {
            if (call_ids.size() <= i) call_ids.push_back(current["tool_calls"][i].at("id"));
            current["tool_calls"][i]["id"] = call_ids[i];
        }
        bool scored = false;
        for (const auto & diff : prima_auto_chat_diffs(previous, current)) {
            for (const auto & key : {"reasoning_content", "content"}) {
                const auto value = diff.at(key).get<std::string>();
                if (value.empty()) continue;
                if (api == prima_api::chat) { events.push_back(chat_event({{key, value}}, nullptr, scored ? nullptr : &result)); scored = true; }
                else delta(events, open(events, key, std::string(key) == "content" ? "text" : "reasoning"), value);
            }
        }
        // Tool names may be incomplete during parsing. Start a block only after a
        // complete declared name, then emit any arguments already buffered.
        if (current.contains("tool_calls")) for (size_t i = 0; i < current["tool_calls"].size(); ++i) {
            const auto & call = current["tool_calls"][i];
            const auto & fn = call.at("function");
            const auto name = fn.at("name").get<std::string>();
            if (!complete_name(name)) continue;
            const auto key = "tool:" + std::to_string(i);
            size_t position = 0;
            while (position < items.size() && items[position].key != key) ++position;
            const bool fresh = position == items.size();
            const auto args = fn.at("arguments").get<std::string>();
            if (!fresh && args.compare(0, items[position].value.size(), items[position].value) != 0)
                throw std::runtime_error("Tool parser changed emitted argument bytes");
            const auto change = fresh ? args : args.substr(items[position].value.size());
            if (fresh) position = open(events, key, "tool", call.at("id"), name);
            if (api == prima_api::chat) {
                json wire_call = {{"index", i}, {"function", {{"arguments", change}}}};
                if (fresh) { wire_call["id"] = call.at("id"); wire_call["type"] = "function"; wire_call["function"]["name"] = name; }
                if (fresh || !change.empty()) { events.push_back(chat_event({{"tool_calls", json::array({wire_call})}}, nullptr, scored ? nullptr : &result)); scored = true; }
                items[position].value += change;
            } else delta(events, position, change);
        }
        previous = current;
        if (!stop) {
            if (api == prima_api::chat && !scored && json_value(prepared_request, "__prima_logprobs", false))
                events.push_back(chat_event(json::object(), nullptr, &result));
            return events;
        }
        finished = true;
        auto params = prepared_request;
        params.erase("__prima_auto_chat");
        auto chat = format_final_response_oaicompat(params, result, id);
        chat["choices"][0]["message"] = current;
        if (!limited && current.contains("tool_calls") && !current["tool_calls"].empty()) chat["choices"][0]["finish_reason"] = "tool_calls";
        if (api == prima_api::chat) events.push_back(chat_event(json::object(), chat["choices"][0]["finish_reason"], &result));
        else if (api == prima_api::messages) {
            for (size_t i = 0; i < items.size(); ++i) {
                auto & item = items[i];
                if (item.closed) continue;
                if (item.kind == "reasoning") events.push_back(tagged("content_block_delta", {{"index", i},
                    {"delta", {{"type", "signature_delta"}, {"signature", ""}}}}));
                events.push_back(tagged("content_block_stop", {{"index", i}}));
                item.closed = true;
            }
            const auto completed = prima_protocol_response(api, chat, request, id, json_value(result, "stopping_word", std::string()));
            events.push_back(tagged("message_delta", {{"delta", {{"stop_reason", completed.at("stop_reason")}, {"stop_sequence", completed.at("stop_sequence")}}}, {"usage", completed.at("usage")}}));
            events.push_back(tagged("message_stop"));
        } else {
            auto completed = prima_protocol_response(api, chat, request, id);
            completed["output"] = json::array();
            for (size_t i = 0; i < items.size(); ++i) {
                const auto & target = items[i];
                json values = {{"item_id", target.id}, {"output_index", i}};
                if (target.kind == "tool") values["arguments"] = target.value;
                else { values["text"] = target.value; if (target.kind == "text") values["content_index"] = 0; }
                if (target.kind != "reasoning") events.push_back(tagged(target.kind == "tool" ? "response.function_call_arguments.done" : "response.output_text.done", values));
                const auto value = wire_item(target, limited ? "incomplete" : "completed");
                if (target.kind == "text") events.push_back(tagged("response.content_part.done",
                    {{"item_id", target.id}, {"output_index", i}, {"content_index", 0},
                     {"part", value.at("content").at(0)}}));
                events.push_back(tagged("response.output_item.done", {{"output_index", i}, {"item", value}}));
                completed["output"].push_back(value);
            }
            events.push_back(tagged(limited ? "response.incomplete" : "response.completed", {{"response", completed}}));
        }
        return events;
    }
};
