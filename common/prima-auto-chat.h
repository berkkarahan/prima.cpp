#pragma once
#include "llama.h"
#include "json.hpp"
#include <string>

// Internal baseline parser bridge. Serving admission is enabled only after sampler integration.
nlohmann::ordered_json prima_auto_chat_prepare(const llama_model * model, const std::string & source,
                                               const nlohmann::ordered_json & body);
nlohmann::ordered_json prima_auto_chat_parse(const nlohmann::ordered_json & prepared,
                                             const std::string & output, bool partial);
