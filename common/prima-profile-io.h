#ifndef PRIMA_PROFILE_IO_H
#define PRIMA_PROFILE_IO_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// A private exclusive file is removed by handle lifetime, including child death.
// Unbuffered I/O refuses unsupported storage rather than timing the page cache.
class prima_profile_file {
public:
    size_t alignment = 4096;
    void * data = nullptr;
    const size_t capacity = 1024 * 1024;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int handle = -1;
#endif
    prima_profile_file(const std::string & directory = ".") {
        static std::atomic<unsigned> serial(0);
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
        const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, directory.c_str(), -1, nullptr, 0);
        if (length <= 0) throw std::invalid_argument("Invalid profiler I/O directory");
        std::wstring prefix(length, L'\0');
        if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, directory.c_str(), -1, &prefix[0], length))
            throw std::invalid_argument("Invalid profiler I/O directory");
        prefix.resize(length - 1);
        const std::wstring name = prefix + L"/prima-profile-" + std::to_wstring(_getpid()) + L"-"
            + std::to_wstring(tick) + L"-" + std::to_wstring(serial++) + L".bin";
        handle = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Unable to create exclusive profiler I/O file");
        FILE_STORAGE_INFO info = {};
        if (!GetFileInformationByHandleEx(handle, FileStorageInfo, &info, sizeof(info))) {
            CloseHandle(handle);
            throw std::runtime_error("Unable to inspect profiler storage alignment");
        }
        alignment = std::max<size_t>({4096, info.LogicalBytesPerSector, info.PhysicalBytesPerSectorForPerformance,
            info.PhysicalBytesPerSectorForAtomicity, info.FileSystemEffectivePhysicalBytesPerSectorForAtomicity});
        if (alignment > 65536 || (alignment & (alignment - 1))) {
            CloseHandle(handle);
            throw std::runtime_error("Unsupported profiler storage alignment");
        }
        data = VirtualAlloc(nullptr, capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!data) {
            CloseHandle(handle);
            throw std::runtime_error("Unable to allocate aligned profiler I/O buffer");
        }
#else
        const std::string name = directory + "/prima-profile-" + std::to_string(getpid()) + "-"
            + std::to_string(tick) + "-" + std::to_string(serial++) + ".bin";
        int flags = O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW;
#if defined(__linux__)
        flags |= O_DIRECT | O_SYNC;
#endif
        handle = open(name.c_str(), flags, 0600);
        if (handle < 0) throw std::runtime_error("Unable to create unbuffered profiler I/O file");
        if (unlink(name.c_str()) != 0) {
            close(handle);
            throw std::runtime_error("Unable to retire profiler I/O file name");
        }
#if defined(__APPLE__)
        if (fcntl(handle, F_NOCACHE, 1) != 0) {
            close(handle);
            throw std::runtime_error("Unable to disable profiler file cache");
        }
#endif
        if (posix_memalign(&data, alignment, capacity) != 0) {
            close(handle);
            throw std::runtime_error("Unable to allocate aligned profiler I/O buffer");
        }
#endif
        if (reinterpret_cast<uintptr_t>(data) % alignment) {
#ifdef _WIN32
            VirtualFree(data, 0, MEM_RELEASE);
            CloseHandle(handle);
#else
            free(data);
            close(handle);
#endif
            throw std::runtime_error("Profiler I/O buffer is not storage-aligned");
        }
        std::memset(data, 0x5a, capacity);
    }
    ~prima_profile_file() {
#ifdef _WIN32
        VirtualFree(data, 0, MEM_RELEASE);
        CloseHandle(handle);
#else
        free(data);
        close(handle);
#endif
    }
    prima_profile_file(const prima_profile_file &) = delete;
    prima_profile_file & operator=(const prima_profile_file &) = delete;
    void transfer(bool writing, uint64_t offset, size_t size) {
        if (!size || size > capacity || size % alignment || offset % alignment)
            throw std::invalid_argument("Invalid aligned profiler I/O request");
#ifdef _WIN32
        LARGE_INTEGER position;
        position.QuadPart = offset;
        DWORD done = 0;
        if (!SetFilePointerEx(handle, position, nullptr, FILE_BEGIN)
            || !(writing ? WriteFile(handle, data, (DWORD) size, &done, nullptr)
                         : ReadFile(handle, data, (DWORD) size, &done, nullptr)) || done != size)
            throw std::runtime_error("Unbuffered profiler I/O failed");
#else
        const auto done = writing ? pwrite(handle, data, size, offset) : pread(handle, data, size, offset);
        if (done != (ssize_t) size) throw std::runtime_error("Unbuffered profiler I/O failed");
#endif
    }
    void flush() {
#ifdef _WIN32
        if (!FlushFileBuffers(handle)) throw std::runtime_error("Unable to flush profiler data");
#else
        if (fsync(handle) != 0) throw std::runtime_error("Unable to flush profiler data");
#endif
    }
};

inline size_t prima_profile_disk(float * read_bw, float * write_bw, bool random, const std::string & directory = ".") {
    prima_profile_file file(directory);
    const uint64_t total = 64 * 1024 * 1024;
    const size_t block = random ? file.alignment : file.capacity;
    const size_t operations = random ? 512 : total / block;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    // Initialize every possible read range. Never read unwritten or sparse bytes.
    for (uint64_t offset = 0; offset < total; offset += file.capacity) {
        if (std::chrono::steady_clock::now() >= until) throw std::runtime_error("Profiler I/O exceeded its deadline");
        file.transfer(true, offset, file.capacity);
    }
    file.flush();
    uint64_t random_state = 0x9e3779b97f4a7c15ULL;
    for (bool writing : {false, true}) {
        double seconds = 0;
        for (size_t i = 0; i < operations; ++i) {
            if (std::chrono::steady_clock::now() >= until) throw std::runtime_error("Profiler I/O exceeded its deadline");
            random_state ^= random_state << 13;
            random_state ^= random_state >> 7;
            random_state ^= random_state << 17;
            const uint64_t offset = (random ? random_state % (total / block) : i) * block;
            if (!writing) std::memset(file.data, 0, block);
            const auto start = std::chrono::steady_clock::now();
            file.transfer(writing, offset, block);
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (!writing) {
                const auto * bytes = static_cast<const unsigned char *>(file.data);
                for (size_t j = 0; j < block; ++j)
                    if (bytes[j] != 0x5a) throw std::runtime_error("Profiler I/O returned incorrect data");
            }
        }
        const auto flush_start = std::chrono::steady_clock::now();
        if (writing) file.flush();
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - flush_start).count();
        if (!std::isfinite(seconds) || seconds <= 0) throw std::runtime_error("Invalid profiler I/O time");
        const float bandwidth = (float) ((double) operations * block / seconds / 1e9);
        *(writing ? write_bw : read_bw) = bandwidth;
    }
    return file.alignment;
}
#endif
