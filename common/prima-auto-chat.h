#pragma once
#include "llama.h"
#include "json.hpp"
#include <string>

// Serialized baseline parser bridge shared by native HTTP generation and streams.
nlohmann::ordered_json prima_auto_chat_prepare(const llama_model * model, const std::string & source,
                                               const nlohmann::ordered_json & body);
nlohmann::ordered_json prima_auto_chat_parse(const nlohmann::ordered_json & prepared,
                                             const std::string & output, bool partial);
nlohmann::ordered_json prima_auto_chat_diffs(const nlohmann::ordered_json & previous,
                                             const nlohmann::ordered_json & current);
