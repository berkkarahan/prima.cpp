#include "../prima-auto-chat.h"
#include "chat.h"

using wire_json = nlohmann::ordered_json;

wire_json prima_auto_chat_prepare(const llama_model * model, const std::string & source, const wire_json & body) {
    if (source.empty() || source.size() > 65536) throw std::runtime_error("Chat parser requires bounded Jinja source");
    const auto request = common_json::parse(body.dump());
    auto templates = common_chat_templates_init(model, source);
    common_chat_templates_inputs input;
    input.use_jinja = true;
    input.messages = common_chat_msgs_parse_oaicompat(request.at("messages"));
    input.add_generation_prompt = request.value("add_generation_prompt", true);
    input.parallel_tool_calls = request.value("parallel_tool_calls", true);
    if (request.contains("grammar")) input.grammar = request.at("grammar").get<std::string>();
    if (request.contains("json_schema")) input.json_schema = request.at("json_schema").dump();
    if (request.contains("response_format")) {
        const auto & format = request.at("response_format");
        if (format.value("type", "") == "json_schema") {
            const auto & schema = format.at("json_schema");
            input.json_schema = (schema.contains("schema") ? schema.at("schema") : schema).dump();
        } else if (format.value("type", "") == "json_object") input.json_schema = "{}";
    }
    input.reasoning_format = common_reasoning_format_from_name(request.value("reasoning_format", std::string("auto")));
    const auto kwargs = request.value("chat_template_kwargs", common_json::object());
    if (!kwargs.is_object()) throw std::runtime_error("Template kwargs must be an object");
    input.enable_thinking = kwargs.value("enable_thinking", request.value("enable_thinking", true));
    for (auto item = kwargs.begin(); item != kwargs.end(); ++item) input.chat_template_kwargs[item.key()] = item.value().dump();
    if (request.contains("reasoning_effort")) {
        const auto effort = request.at("reasoning_effort").get<std::string>();
        if (effort == "none") {
            input.enable_thinking = false;
            input.chat_template_kwargs.erase("reasoning_effort");
        } else if (!effort.empty()) input.chat_template_kwargs["reasoning_effort"] = common_json(effort).dump();
    }
    if (request.contains("tools")) {
        if (!request.at("tools").is_array()) throw std::runtime_error("Tools must be an array");
        for (const auto & tool : request.at("tools")) {
            if (tool.at("type") != "function") throw std::runtime_error("Only function tools are supported");
            const auto & fn = tool.at("function");
            input.tools.push_back({fn.at("name").get<std::string>(), fn.value("description", ""),
                common_chat_tool_parameters(fn).dump()});
        }
    }
    const auto choice = request.value("tool_choice", common_json("auto"));
    if (choice == "none") input.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    else if (choice == "required") input.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    else if (choice.is_object() && choice.value("type", "") == "function") {
        const auto name = choice.at("function").at("name").get<std::string>();
        input.tools.erase(std::remove_if(input.tools.begin(), input.tools.end(),
            [&](const common_chat_tool & tool) { return tool.name != name; }), input.tools.end());
        if (input.tools.size() != 1) throw std::runtime_error("Named tool choice requires one matching function");
        input.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    } else if (choice != "auto") throw std::runtime_error("Unsupported tool choice");
    auto result = common_chat_templates_apply(templates.get(), input);
    wire_json triggers = wire_json::array();
    wire_json patterns = wire_json::array(), tokens = wire_json::array();
    for (const auto & trigger : result.grammar_triggers)
    {
        triggers.push_back({{"type", int(trigger.type)}, {"value", trigger.value}, {"token", trigger.token}});
        auto value = trigger.value;
        switch (trigger.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN: tokens.push_back(trigger.token); break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD: {
                std::string escaped;
                for (const char c : value) {
                    if (std::string("\\^$.|?*+()[]{}").find(c) != std::string::npos) escaped += '\\';
                    escaped += c;
                }
                patterns.push_back(escaped); break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN: patterns.push_back(value); break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                patterns.push_back(value.empty() ? "^$" :
                    (value.front() == '^' ? "" : "^") + value + (value.back() == '$' ? "" : "$")); break;
            default: throw std::runtime_error("Unknown grammar trigger type");
        }
    }
    if (!input.tools.empty() && input.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && result.grammar.empty())
        throw std::runtime_error("Selected template does not provide constrained automatic tools");
    return {{"format", int(result.format)}, {"prompt", result.prompt}, {"grammar", result.grammar},
        {"grammar_lazy", result.grammar_lazy}, {"grammar_triggers", triggers},
        {"grammar_trigger_patterns", patterns}, {"grammar_trigger_tokens", tokens},
        {"preserved_tokens", result.preserved_tokens}, {"additional_stops", result.additional_stops},
        {"generation_prompt", result.generation_prompt}, {"parser", result.parser},
        {"reasoning_format", int(input.reasoning_format)}, {"enable_thinking", input.enable_thinking},
        {"thinking_start_tag", result.thinking_start_tag}, {"thinking_end_tags", result.thinking_end_tags},
        {"stream", request.value("stream", false)}};
}

wire_json prima_auto_chat_diffs(const wire_json & previous, const wire_json & current) {
    const auto messages = common_chat_msgs_parse_oaicompat(common_json::array({
        common_json::parse(previous.dump()), common_json::parse(current.dump())}));
    wire_json result = wire_json::array();
    for (const auto & diff : common_chat_msg_diff::compute_diffs(messages.at(0), messages.at(1))) {
        wire_json value = {{"content", diff.content_delta}, {"reasoning_content", diff.reasoning_content_delta}};
        if (diff.tool_call_index != std::string::npos) value["tool_call"] = {
            {"index", diff.tool_call_index}, {"id", diff.tool_call_delta.id},
            {"name", diff.tool_call_delta.name}, {"arguments", diff.tool_call_delta.arguments}};
        result.push_back(value);
    }
    return result;
}

wire_json prima_auto_chat_parse(const wire_json & prepared, const std::string & output, bool partial) {
    const int format = prepared.at("format").get<int>();
    if (format < 0 || format >= COMMON_CHAT_FORMAT_COUNT) throw std::runtime_error("Invalid internal chat parser format");
    common_chat_parser_params params;
    params.format = common_chat_format(format);
    params.reasoning_format = common_reasoning_format(prepared.at("reasoning_format").get<int>());
    params.reasoning_in_content = prepared.value("stream", false) && params.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    params.generation_prompt = prepared.at("generation_prompt").get<std::string>();
    const auto serialized = prepared.at("parser").get<std::string>();
    if (!serialized.empty()) params.parser.load(serialized);
    const auto parsed = common_chat_parse(output, partial, params);
    return wire_json::parse(parsed.to_json_oaicompat(true).dump());
}
