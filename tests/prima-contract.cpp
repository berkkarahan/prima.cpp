#include "prima-sha256.h"
#include "prima-env.h"
#include <iostream>
int main() {
    if (prima_flag_enabled(nullptr) || prima_flag_enabled("") || prima_flag_enabled("0") ||
        prima_flag_enabled("false") || !prima_flag_enabled("1")) return 4;
    if (prima_sha256("") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") return 1;
    if (prima_sha256("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return 2;
    if (prima_sha256(std::string(1000000, 'a')) != "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") return 3;
    std::cout << "SHA-256 known vectors passed\n";
    return 0;
}
