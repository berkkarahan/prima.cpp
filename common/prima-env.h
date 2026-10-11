#pragma once
#include <cstring>

// Runtime telemetry is opt-in. A set variable containing "0" is disabled.
inline bool prima_flag_enabled(const char * value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}
