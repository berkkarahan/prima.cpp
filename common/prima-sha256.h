#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

class prima_sha256_state {
    std::array<uint32_t,8> h_ = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<uint8_t,64> block_{};
    size_t used_ = 0;
    uint64_t total_ = 0;
    void transform(const uint8_t * bytes) {
        const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        auto rotr=[](uint32_t v,int n){return (v>>n)|(v<<(32-n));};
        uint32_t w[64];
        for(int i=0;i<16;++i) w[i]=(uint32_t(bytes[4*i])<<24)|(uint32_t(bytes[4*i+1])<<16)|(uint32_t(bytes[4*i+2])<<8)|bytes[4*i+3];
        for(int i=16;i<64;++i) w[i]=w[i-16]+(rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10));
        auto s=h_;
        for(int i=0;i<64;++i) {
            uint32_t t1=s[7]+(rotr(s[4],6)^rotr(s[4],11)^rotr(s[4],25))+((s[4]&s[5])^(~s[4]&s[6]))+k[i]+w[i];
            uint32_t t2=(rotr(s[0],2)^rotr(s[0],13)^rotr(s[0],22))+((s[0]&s[1])^(s[0]&s[2])^(s[1]&s[2]));
            s={t1+t2,s[0],s[1],s[2],s[3]+t1,s[4],s[5],s[6]};
        }
        for(int i=0;i<8;++i) h_[i]+=s[i];

    }
public:
    void update(const void * data, size_t size) {
        if (size > UINT64_MAX / 8 - total_) throw std::overflow_error("SHA-256 input exceeds its length bound");
        total_ += size;
        const auto * bytes = static_cast<const uint8_t *>(data);
        while (size) {
            const size_t count = size < 64 - used_ ? size : 64 - used_;
            std::memcpy(block_.data() + used_, bytes, count);
            used_ += count;
            bytes += count;
            size -= count;
            if (used_ == 64) { transform(block_.data()); used_ = 0; }
        }
    }
    std::string digest() const {
        auto final = *this;
        const uint64_t bits = total_ * uint64_t(8);
        const uint8_t marker = 0x80, zero = 0;
        final.update(&marker, 1);
        while (final.used_ != 56) final.update(&zero, 1);
        uint8_t length[8];
        for (int i = 0; i < 8; ++i) length[i] = uint8_t(bits >> ((7 - i) * 8));
        final.update(length, sizeof(length));
        std::ostringstream out;
        for (auto value : final.h_) out << std::hex << std::setfill('0') << std::setw(8) << value;
        return out.str();
    }
};

inline std::string prima_sha256(const std::string & input) {
    prima_sha256_state state;
    state.update(input.data(), input.size());
    return state.digest();
}
