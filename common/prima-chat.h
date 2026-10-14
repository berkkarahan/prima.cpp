#pragma once
#include "llama.h"
#include "json.hpp"
#include <string>

// Server Jinja rendering uses the exact parser/runtime from the Kluster baseline.
void prima_validate_jinja(const std::string & source);
std::string prima_template_source(const llama_model * model, const std::string & override_source);
std::string prima_read_template(const std::string & path);
nlohmann::ordered_json prima_parse_template_kwargs(const std::string & value);
std::string prima_read_template_kwargs(const std::string & path);
std::string prima_render_jinja(const llama_model * model, const std::string & source,
    const nlohmann::ordered_json & messages, bool add_generation_prompt,
    const nlohmann::ordered_json & kwargs, const nlohmann::ordered_json & tools);
