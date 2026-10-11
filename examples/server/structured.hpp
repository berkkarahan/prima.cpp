#pragma once
#include <cctype>
#include <string>

// Rename grammar identifiers without touching JSON string literals, classes or comments.
inline std::string prima_json_grammar(const std::string & grammar) {
    std::string value = "prima-json-value";
    unsigned suffix = 0;
    const std::string lines = "\n" + grammar;
    while (lines.find("\n" + value + " ::=") != std::string::npos) value = "prima-json-value-" + std::to_string(++suffix);
    std::string output;
    for (size_t i = 0; i < grammar.size();) {
        const char c = grammar[i];
        if (c == '"' || c == '[') {
            const char close = c == '"' ? '"' : ']';
            output += grammar[i++];
            while (i < grammar.size()) {
                const char next = grammar[i++];
                output += next;
                if (next == '\\' && i < grammar.size()) output += grammar[i++];
                else if (next == close) break;
            }
        } else if (c == '#') {
            while (i < grammar.size() && grammar[i] != '\n') output += grammar[i++];
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const size_t begin = i++;
            while (i < grammar.size() && (std::isalnum(static_cast<unsigned char>(grammar[i])) || grammar[i] == '-' || grammar[i] == '_')) ++i;
            const auto name = grammar.substr(begin, i - begin);
            output += name == "root" ? value : name;
        } else {
            output += grammar[i++];
        }
    }
    return output + "\nroot ::= " + value + " | \"```json\\n\" " + value + " \"```\"\n";
}

inline std::string prima_json_content(const std::string & source) {
    static const std::string opening = "```json\n";
    size_t begin = source.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    if (source[begin] == '`') {
        const auto prefix = source.substr(begin);
        if (prefix.size() < opening.size() && opening.compare(0, prefix.size(), prefix) == 0) return "";
        if (source.compare(begin, opening.size(), opening) != 0) return "";
        begin += opening.size();
        begin = source.find_first_not_of(" \t\r\n", begin);
        if (begin == std::string::npos) return "";
    }
    bool quoted = false, escaped = false;
    int depth = 0;
    const bool container = source[begin] == '{' || source[begin] == '[';
    const bool string = source[begin] == '"';
    for (size_t i = begin; i < source.size(); ++i) {
        const char c = source[i];
        if (quoted) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') {
                quoted = false;
                if (string && depth == 0) return source.substr(begin, i - begin + 1);
            }
        } else if (c == '"') quoted = true;
        else if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') {
            if (--depth == 0) return source.substr(begin, i - begin + 1);
        } else if (!container && !string && (std::isspace(static_cast<unsigned char>(c)) || c == '`')) {
            return source.substr(begin, i - begin);
        }
    }
    // Hold outer trailing whitespace, but preserve whitespace inside an unfinished string.
    const size_t end = quoted ? source.size() : source.find_last_not_of(" \t\r\n") + 1;
    return source.substr(begin, end - begin);
}

struct prima_json_stream {
    std::string source;
    size_t sent = 0;
    std::string append(const std::string & delta) {
        source += delta;
        const auto content = prima_json_content(source);
        const auto result = content.substr(sent);
        sent = content.size();
        return result;
    }
};
