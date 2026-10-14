#pragma once

#include <string>

#include <cstdint>

bool is_port_open(const std::string& ip, uint32_t port, int timeout_sec = 2);