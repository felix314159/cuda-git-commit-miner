// Example Ubuntu/Debian requirements:
//   sudo apt update && sudo apt install -y build-essential git nvidia-cuda-toolkit
//
// Example build command (RTX 4070 Ti / Ada, sm_89):
//   nvcc -O3 -std=c++17 -arch=sm_89 main.cu -o gitminer-head
//
// Run:
//   ./gitminer-head [prefix=0000000] [device=0]

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(err__));                 \
        }                                                                        \
    } while (0)

namespace {

constexpr const char* kDefaultPrefix = "0000000";
constexpr int kMaxNonceDigits = 18;
constexpr int kThreadsPerBlock = 256;
constexpr uint64_t kCandidatesPerThread = 2048;
constexpr int kMaxTailBytes = 256;

struct PrefixTarget {
    std::array<uint8_t, 20> bytes{};
    std::array<uint8_t, 20> mask{};
    int hex_chars = 0;
};

struct MiningResult {
    bool found = false;
    uint64_t nonce = 0;
    std::array<uint32_t, 5> hash{};
    std::string backend;
};

struct CommitterHeaderParts {
    std::string base_name;
    std::string email;
    std::string payload_prefix;
    std::string payload_suffix;
};

__constant__ uint8_t c_prefix_bytes[20];
__constant__ uint8_t c_prefix_mask[20];

__host__ __device__ inline uint32_t rotl32(uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

__host__ __device__ void sha1_transform(const uint8_t block[64], uint32_t state[5]) {
    uint32_t w[80];

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int base = i * 4;
        w[i] = (static_cast<uint32_t>(block[base]) << 24) |
               (static_cast<uint32_t>(block[base + 1]) << 16) |
               (static_cast<uint32_t>(block[base + 2]) << 8) |
               static_cast<uint32_t>(block[base + 3]);
    }

#pragma unroll
    for (int i = 16; i < 80; ++i) {
        w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];

#pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint32_t f = 0;
        uint32_t k = 0;

        if (i < 20) {
            f = (b & c) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ c ^ d;
            k = 0xca62c1d6u;
        }

        const uint32_t temp = rotl32(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

void sha1_init(uint32_t state[5]) {
    state[0] = 0x67452301u;
    state[1] = 0xefcdab89u;
    state[2] = 0x98badcfeu;
    state[3] = 0x10325476u;
    state[4] = 0xc3d2e1f0u;
}

int pad_tail_host(uint8_t* buffer, int raw_tail_len, int prefix_len, int total_len) {
    int len = raw_tail_len;
    buffer[len++] = 0x80;
    while (((prefix_len + len) % 64) != 56) {
        buffer[len++] = 0x00;
    }

    const uint64_t total_bits = static_cast<uint64_t>(total_len) * 8u;
    for (int i = 7; i >= 0; --i) {
        buffer[len++] = static_cast<uint8_t>((total_bits >> (i * 8)) & 0xffu);
    }
    return len;
}

void sha1_process_full_blocks(const uint8_t* data, size_t len, uint32_t state[5]) {
    for (size_t offset = 0; offset < len; offset += 64) {
        sha1_transform(data + offset, state);
    }
}

std::string run_command(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to run command: " + command);
    }

    std::string output;
    char buffer[4096];
    while (true) {
        const size_t read = fread(buffer, 1, sizeof(buffer), pipe);
        if (read > 0) {
            output.append(buffer, read);
        }
        if (read < sizeof(buffer)) {
            if (feof(pipe)) {
                break;
            }
            if (ferror(pipe)) {
                pclose(pipe);
                throw std::runtime_error("Error reading command output: " + command);
            }
        }
    }

    const int rc = pclose(pipe);
    if (rc == -1 || !WIFEXITED(rc) || WEXITSTATUS(rc) != 0) {
        throw std::runtime_error("Command failed: " + command);
    }

    return output;
}

std::string trim_trailing_newlines(std::string value) {
    while (!value.empty() && value.back() == '\n') {
        value.pop_back();
    }
    return value;
}

bool is_digit_at(const std::string& value, size_t index) {
    return index < value.size() &&
           std::isdigit(static_cast<unsigned char>(value[index])) != 0;
}

bool has_existing_mined_date_suffix(const std::string& message, size_t start) {
    if (message.compare(start, 8, ", Date: ") != 0) {
        return false;
    }

    size_t pos = start + 8;
    const int digit_groups[] = {4, 2, 2, 2, 2, 2};
    const char separators[] = {'-', '-', ',', ':', ':', '.'};

    for (int group = 0; group < 6; ++group) {
        for (int i = 0; i < digit_groups[group]; ++i, ++pos) {
            if (!is_digit_at(message, pos)) {
                return false;
            }
        }

        if (group == 2) {
            if (pos >= message.size() || message[pos] != separators[group]) {
                return false;
            }
            ++pos;
            if (pos >= message.size() || message[pos] != ' ') {
                return false;
            }
            ++pos;
        } else {
            if (pos >= message.size() || message[pos] != separators[group]) {
                return false;
            }
            ++pos;
        }
    }

    if (pos >= message.size() || !is_digit_at(message, pos)) {
        return false;
    }
    while (pos < message.size()) {
        if (!is_digit_at(message, pos)) {
            return false;
        }
        ++pos;
    }

    return true;
}

std::string strip_existing_mined_date_suffix(const std::string& message) {
    const size_t start = message.rfind(", Date: ");
    if (start == std::string::npos) {
        return message;
    }
    if (!has_existing_mined_date_suffix(message, start)) {
        return message;
    }
    return message.substr(0, start);
}

std::string strip_gpgsig_header(const std::string& headers) {
    std::string out;
    out.reserve(headers.size());

    bool skipping_gpgsig = false;
    size_t pos = 0;
    while (pos < headers.size()) {
        const size_t line_end = headers.find('\n', pos);
        if (line_end == std::string::npos) {
            throw std::runtime_error("Malformed commit headers in HEAD.");
        }

        const std::string line = headers.substr(pos, line_end - pos);
        if (!skipping_gpgsig) {
            if (line.rfind("gpgsig ", 0) == 0) {
                skipping_gpgsig = true;
            } else {
                out.append(line);
                out.push_back('\n');
            }
        } else if (line.empty() || line[0] != ' ') {
            skipping_gpgsig = false;
            out.append(line);
            out.push_back('\n');
        }

        pos = line_end + 1;
    }

    return out;
}

CommitterHeaderParts parse_committer_header(const std::string& headers,
                                            const std::string& message_with_final_newline) {
    constexpr const char* kCommitterPrefix = "committer ";
    constexpr size_t kCommitterPrefixLen = 10;

    size_t line_start = std::string::npos;
    if (headers.rfind(kCommitterPrefix, 0) == 0) {
        line_start = 0;
    } else {
        const size_t marker = headers.find("\ncommitter ");
        if (marker != std::string::npos) {
            line_start = marker + 1;
        }
    }
    if (line_start == std::string::npos) {
        throw std::runtime_error("Could not find the committer header in HEAD.");
    }

    const size_t line_end = headers.find('\n', line_start);
    if (line_end == std::string::npos) {
        throw std::runtime_error("Could not parse the committer header in HEAD.");
    }

    const std::string line = headers.substr(line_start, line_end - line_start);
    if (line.rfind(kCommitterPrefix, 0) != 0) {
        throw std::runtime_error("Malformed committer header in HEAD.");
    }

    const std::string body = line.substr(kCommitterPrefixLen);
    const size_t tz_start = body.rfind(' ');
    if (tz_start == std::string::npos || tz_start == 0) {
        throw std::runtime_error("Could not parse the committer timezone in HEAD.");
    }
    const size_t timestamp_start = body.rfind(' ', tz_start - 1);
    if (timestamp_start == std::string::npos || timestamp_start == 0) {
        throw std::runtime_error("Could not parse the committer timestamp in HEAD.");
    }

    const std::string name_and_email = body.substr(0, timestamp_start);
    const size_t email_start = name_and_email.rfind(" <");
    if (email_start == std::string::npos || email_start == 0) {
        throw std::runtime_error("Could not parse the committer name/email in HEAD.");
    }

    CommitterHeaderParts parts;
    parts.base_name = name_and_email.substr(0, email_start);
    parts.email = name_and_email.substr(email_start + 2,
                                        name_and_email.size() - email_start - 3);
    parts.payload_prefix =
        headers.substr(0, line_start) + kCommitterPrefix + parts.base_name + " ";
    parts.payload_suffix =
        body.substr(email_start) + headers.substr(line_end) + message_with_final_newline;
    return parts;
}

std::string lower_hex(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string shell_single_quote(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('\'');
    for (char ch : value) {
        if (ch == '\'') {
            out += "'\"'\"'";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

bool is_hex_string(const std::string& value) {
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

PrefixTarget parse_prefix(const std::string& value) {
    if (value.empty() || value.size() > 40 || !is_hex_string(value)) {
        throw std::runtime_error("Prefix must be 1-40 hex characters.");
    }

    PrefixTarget target;
    target.hex_chars = static_cast<int>(value.size());

    for (size_t i = 0; i < value.size(); ++i) {
        uint8_t nibble = 0;
        const char ch = value[i];
        if (ch >= '0' && ch <= '9') {
            nibble = static_cast<uint8_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            nibble = static_cast<uint8_t>(ch - 'a' + 10);
        } else if (ch >= 'A' && ch <= 'F') {
            nibble = static_cast<uint8_t>(ch - 'A' + 10);
        }

        const size_t byte_index = i / 2;
        if ((i % 2) == 0) {
            target.bytes[byte_index] |= static_cast<uint8_t>(nibble << 4);
            target.mask[byte_index] |= 0xf0u;
        } else {
            target.bytes[byte_index] |= nibble;
            target.mask[byte_index] |= 0x0fu;
        }
    }

    return target;
}

uint64_t pow10_u64(int digits) {
    if (digits <= 0 || digits > kMaxNonceDigits) {
        throw std::runtime_error("Nonce digits must be between 1 and " +
                                 std::to_string(kMaxNonceDigits) + ".");
    }

    uint64_t result = 1;
    for (int i = 0; i < digits; ++i) {
        result *= 10;
    }
    return result;
}

int auto_nonce_digits_for_prefix(int prefix_hex_chars) {
    if (prefix_hex_chars <= 0) {
        throw std::runtime_error("Prefix must be at least 1 hex character.");
    }

    const double decimal_digits = std::ceil(static_cast<double>(prefix_hex_chars) *
                                            std::log10(16.0));
    return std::min(kMaxNonceDigits, std::max(1, static_cast<int>(decimal_digits)));
}

std::string format_nonce(uint64_t nonce, int digits) {
    std::string out(static_cast<size_t>(digits), '0');
    for (int i = digits - 1; i >= 0; --i) {
        out[static_cast<size_t>(i)] = static_cast<char>('0' + (nonce % 10));
        nonce /= 10;
    }
    return out;
}

std::string hex_digest(const uint32_t state[5]) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < 5; ++i) {
        out << std::setw(8) << state[i];
    }
    return out.str();
}

void sha1_bytes_from_state(const uint32_t state[5], uint8_t digest[20]) {
    for (int i = 0; i < 5; ++i) {
        digest[i * 4] = static_cast<uint8_t>(state[i] >> 24);
        digest[i * 4 + 1] = static_cast<uint8_t>(state[i] >> 16);
        digest[i * 4 + 2] = static_cast<uint8_t>(state[i] >> 8);
        digest[i * 4 + 3] = static_cast<uint8_t>(state[i]);
    }
}

bool matches_prefix_host(const uint32_t state[5], const PrefixTarget& target) {
    uint8_t digest[20];
    sha1_bytes_from_state(state, digest);
    for (int i = 0; i < 20; ++i) {
        if ((digest[i] & target.mask[static_cast<size_t>(i)]) != target.bytes[static_cast<size_t>(i)]) {
            return false;
        }
    }
    return true;
}

std::array<uint32_t, 5> to_array(const uint32_t state[5]) {
    std::array<uint32_t, 5> out{};
    for (int i = 0; i < 5; ++i) {
        out[static_cast<size_t>(i)] = state[i];
    }
    return out;
}

std::vector<uint8_t> make_candidate_object(const std::string& header_and_body_prefix,
                                           const std::string& nonce,
                                           const std::string& suffix) {
    const std::string payload = header_and_body_prefix + nonce + suffix;
    const std::string object = "commit " + std::to_string(payload.size()) + '\0' + payload;
    return std::vector<uint8_t>(object.begin(), object.end());
}

std::string usage(const char* argv0) {
    std::ostringstream out;
    out << "Usage: " << argv0 << " [prefix=" << kDefaultPrefix << "] [device=0]\n";
    return out.str();
}

__host__ __device__ void copy_bytes(uint8_t* dst, const uint8_t* src, int len) {
    for (int i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
}

__host__ __device__ void copy_words(uint32_t* dst, const uint32_t* src, int len) {
    for (int i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
}

__host__ __device__ void write_nonce_decimal(uint8_t* dst, int digits, uint64_t nonce) {
    for (int i = digits - 1; i >= 0; --i) {
        dst[i] = static_cast<uint8_t>('0' + (nonce % 10));
        nonce /= 10;
    }
}

__device__ int pad_tail(uint8_t* buffer, int raw_tail_len, int prefix_len, int total_len) {
    int len = raw_tail_len;
    buffer[len++] = 0x80;
    while (((prefix_len + len) % 64) != 56) {
        buffer[len++] = 0x00;
    }

    const uint64_t total_bits = static_cast<uint64_t>(total_len) * 8u;
    for (int i = 7; i >= 0; --i) {
        buffer[len++] = static_cast<uint8_t>((total_bits >> (i * 8)) & 0xffu);
    }
    return len;
}

__device__ bool digest_matches_prefix(const uint32_t state[5]) {
    uint8_t digest[20];
    for (int i = 0; i < 5; ++i) {
        digest[i * 4] = static_cast<uint8_t>(state[i] >> 24);
        digest[i * 4 + 1] = static_cast<uint8_t>(state[i] >> 16);
        digest[i * 4 + 2] = static_cast<uint8_t>(state[i] >> 8);
        digest[i * 4 + 3] = static_cast<uint8_t>(state[i]);
    }

    for (int i = 0; i < 20; ++i) {
        if ((digest[i] & c_prefix_mask[i]) != c_prefix_bytes[i]) {
            return false;
        }
    }
    return true;
}

__global__ void mine_nonce_kernel(const uint8_t* tail_template,
                                  int tail_len,
                                  int prefix_len,
                                  int total_len,
                                  int nonce_offset_in_tail,
                                  int nonce_digits,
                                  uint64_t batch_start,
                                  uint64_t max_nonce,
                                  uint64_t candidates_per_thread,
                                  const uint32_t* prefix_state,
                                  int* found_flag,
                                  uint64_t* found_nonce,
                                  uint32_t* found_hash) {
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint8_t local_tail[kMaxTailBytes];
    uint32_t state[5];

    for (uint64_t iter = 0; iter < candidates_per_thread; ++iter) {
        if (*found_flag) {
            return;
        }

        const uint64_t nonce = batch_start + tid + iter * stride;
        if (nonce >= max_nonce) {
            return;
        }

        copy_bytes(local_tail, tail_template, tail_len);
        write_nonce_decimal(local_tail + nonce_offset_in_tail, nonce_digits, nonce);

        copy_words(state, prefix_state, 5);
        const int padded_len = pad_tail(local_tail, tail_len, prefix_len, total_len);

        for (int offset = 0; offset < padded_len; offset += 64) {
            sha1_transform(local_tail + offset, state);
        }

        if (digest_matches_prefix(state)) {
            if (atomicCAS(found_flag, 0, 1) == 0) {
                *found_nonce = nonce;
                for (int i = 0; i < 5; ++i) {
                    found_hash[i] = state[i];
                }
            }
            return;
        }
    }
}

MiningResult mine_on_cpu(const std::vector<uint8_t>& tail_template,
                         int tail_len,
                         int prefix_len,
                         int total_len,
                         int nonce_offset_in_tail,
                         int nonce_digits,
                         uint64_t max_nonce,
                         const uint32_t prefix_state[5],
                         const PrefixTarget& prefix) {
    MiningResult result;
    result.backend = "CPU";

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    const unsigned int thread_count = std::max(1u, hw_threads);
    std::atomic<bool> found{false};
    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> found_nonce{0};
    std::array<uint32_t, 5> found_hash{};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    const auto started_at = std::chrono::steady_clock::now();

    for (unsigned int thread_index = 0; thread_index < thread_count; ++thread_index) {
        workers.emplace_back([&, thread_index]() {
            uint8_t local_tail[kMaxTailBytes];
            uint32_t state[5];

            for (uint64_t nonce = thread_index; nonce < max_nonce && !found.load(std::memory_order_relaxed);
                 nonce += thread_count) {
                copy_words(state, prefix_state, 5);
                std::copy(tail_template.begin(), tail_template.end(), local_tail);
                write_nonce_decimal(local_tail + nonce_offset_in_tail, nonce_digits, nonce);
                const int padded_len = pad_tail_host(local_tail, tail_len, prefix_len, total_len);

                for (int offset = 0; offset < padded_len; offset += 64) {
                    sha1_transform(local_tail + offset, state);
                }

                processed.fetch_add(1, std::memory_order_relaxed);

                if (matches_prefix_host(state, prefix)) {
                    bool expected = false;
                    if (found.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                        found_nonce.store(nonce, std::memory_order_relaxed);
                        found_hash = to_array(state);
                    }
                    break;
                }
            }
        });
    }

    while (!found.load(std::memory_order_relaxed) &&
           processed.load(std::memory_order_relaxed) < max_nonce) {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - started_at).count();
        if (seconds > 0.0) {
            const uint64_t seen = processed.load(std::memory_order_relaxed);
            const double mh_s = static_cast<double>(seen) / seconds / 1e6;
            std::cout << "\rProcessed " << seen << " / " << max_nonce
                      << " candidates (" << std::fixed << std::setprecision(2)
                      << mh_s << " MH/s)" << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - started_at).count();
    if (seconds > 0.0) {
        const uint64_t seen = processed.load(std::memory_order_relaxed);
        const double mh_s = static_cast<double>(seen) / seconds / 1e6;
        std::cout << "\rProcessed " << seen << " / " << max_nonce
                  << " candidates (" << std::fixed << std::setprecision(2)
                  << mh_s << " MH/s)" << std::flush;
    }
    std::cout << "\n";

    if (found.load(std::memory_order_relaxed)) {
        result.found = true;
        result.nonce = found_nonce.load(std::memory_order_relaxed);
        result.hash = found_hash;
    }

    return result;
}

MiningResult try_mine_on_cuda(const std::vector<uint8_t>& tail_template,
                              int tail_len,
                              int prefix_len,
                              int total_len,
                              int nonce_offset_in_tail,
                              int nonce_digits,
                              uint64_t max_nonce,
                              const uint32_t prefix_state[5],
                              const PrefixTarget& prefix,
                              const std::string& prefix_arg,
                              int device,
                              std::string& warning) {
    MiningResult result;
    result.backend = "CUDA";

    uint8_t* d_tail_template = nullptr;
    uint32_t* d_prefix_state = nullptr;
    int* d_found_flag = nullptr;
    uint64_t* d_found_nonce = nullptr;
    uint32_t* d_found_hash = nullptr;

    try {
        int device_count = 0;
        cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count <= 0) {
            warning = "Warning: CUDA is not available; falling back to CPU mining. "
                      "This will be slower than on a CUDA GPU.";
            return result;
        }

        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));

        int blocks = props.multiProcessorCount * 8;
        if (blocks < 256) {
            blocks = 256;
        }

        CUDA_CHECK(cudaMemcpyToSymbol(c_prefix_bytes, prefix.bytes.data(), prefix.bytes.size()));
        CUDA_CHECK(cudaMemcpyToSymbol(c_prefix_mask, prefix.mask.data(), prefix.mask.size()));

        CUDA_CHECK(cudaMalloc(&d_tail_template, tail_template.size()));
        CUDA_CHECK(cudaMalloc(&d_prefix_state, sizeof(uint32_t) * 5));
        CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_found_nonce, sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_found_hash, sizeof(uint32_t) * 5));

        CUDA_CHECK(cudaMemcpy(d_tail_template,
                              tail_template.data(),
                              tail_template.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_prefix_state,
                              prefix_state,
                              sizeof(uint32_t) * 5,
                              cudaMemcpyHostToDevice));

        std::cout << "Mining HEAD for prefix " << prefix_arg << " on " << props.name
                  << " with " << blocks << " blocks x "
                  << kThreadsPerBlock << " threads\n";

        uint64_t batch_start = 0;
        const uint64_t stride = static_cast<uint64_t>(blocks) * kThreadsPerBlock;
        const uint64_t per_launch = stride * kCandidatesPerThread;
        const auto started_at = std::chrono::steady_clock::now();

        while (batch_start < max_nonce) {
            const int zero = 0;
            CUDA_CHECK(cudaMemcpy(d_found_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice));

            mine_nonce_kernel<<<blocks, kThreadsPerBlock>>>(
                d_tail_template,
                tail_len,
                prefix_len,
                total_len,
                nonce_offset_in_tail,
                nonce_digits,
                batch_start,
                max_nonce,
                kCandidatesPerThread,
                d_prefix_state,
                d_found_flag,
                d_found_nonce,
                d_found_hash);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int host_found = 0;
            CUDA_CHECK(cudaMemcpy(&host_found, d_found_flag, sizeof(host_found), cudaMemcpyDeviceToHost));
            if (host_found) {
                CUDA_CHECK(cudaMemcpy(&result.nonce,
                                      d_found_nonce,
                                      sizeof(result.nonce),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(result.hash.data(),
                                      d_found_hash,
                                      sizeof(uint32_t) * 5,
                                      cudaMemcpyDeviceToHost));
                result.found = true;
                break;
            }

            batch_start += per_launch;

            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - started_at).count();
            if (seconds > 0.0) {
                const uint64_t processed = std::min(batch_start, max_nonce);
                const double mh_s = static_cast<double>(processed) / seconds / 1e6;
                std::cout << "\rProcessed " << processed << " / " << max_nonce
                          << " candidates (" << std::fixed << std::setprecision(2)
                          << mh_s << " MH/s)" << std::flush;
            }
        }

        std::cout << "\n";
    } catch (const std::exception& ex) {
        warning = std::string("Warning: CUDA mining failed (") + ex.what() +
                  "); falling back to CPU mining. This will be slower than on a CUDA GPU.";
        result.found = false;
    }

    if (d_tail_template) {
        cudaFree(d_tail_template);
    }
    if (d_prefix_state) {
        cudaFree(d_prefix_state);
    }
    if (d_found_flag) {
        cudaFree(d_found_flag);
    }
    if (d_found_nonce) {
        cudaFree(d_found_nonce);
    }
    if (d_found_hash) {
        cudaFree(d_found_hash);
    }

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string prefix_arg = kDefaultPrefix;
        int device = 0;
        int positional_index = 0;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                std::cout << usage(argv[0]);
                return 0;
            }

            if (positional_index == 0) {
                prefix_arg = lower_hex(arg);
            } else if (positional_index == 1) {
                device = std::stoi(arg);
            } else {
                std::cerr << usage(argv[0]);
                return 1;
            }
            ++positional_index;
        }

        prefix_arg = lower_hex(prefix_arg);
        const PrefixTarget prefix = parse_prefix(prefix_arg);
        const int nonce_digits = auto_nonce_digits_for_prefix(prefix.hex_chars);
        const uint64_t max_nonce = pow10_u64(nonce_digits);

        const std::string inside_work_tree = trim_trailing_newlines(
            run_command("git rev-parse --is-inside-work-tree 2>/dev/null"));
        if (inside_work_tree != "true") {
            throw std::runtime_error("Run this program inside a git repository.");
        }

        const std::string object_format = trim_trailing_newlines(
            run_command("git rev-parse --show-object-format 2>/dev/null"));
        if (object_format != "sha1") {
            throw std::runtime_error(
                "This miner only supports SHA-1 Git repositories. "
                "The current repository uses object format '" +
                object_format +
                "', so the program's SHA-1 commit mining would produce the wrong commit ID.");
        }

        const std::string raw_commit = run_command("git cat-file commit HEAD");
        const size_t separator = raw_commit.find("\n\n");
        if (separator == std::string::npos) {
            throw std::runtime_error("Could not split HEAD commit into headers and message.");
        }

        const std::string raw_headers = raw_commit.substr(0, separator + 2);
        const std::string headers = strip_gpgsig_header(raw_headers);
        const bool head_was_signed = headers.size() != raw_headers.size();
        if (head_was_signed) {
            std::cout << "HEAD is signed; its signature will be stripped during mining, "
                         "so the mined replacement commit will be unsigned.\n";
        }

        const std::string original_message = raw_commit.substr(separator + 2);
        const std::string base_message =
            strip_existing_mined_date_suffix(trim_trailing_newlines(original_message));
        const std::string message_with_final_newline = base_message + "\n";
        const CommitterHeaderParts committer =
            parse_committer_header(headers, message_with_final_newline);
        const std::string payload_prefix = committer.payload_prefix;
        const std::string payload_suffix = committer.payload_suffix;
        const std::string zero_nonce(static_cast<size_t>(nonce_digits), '0');

        std::vector<uint8_t> base_object =
            make_candidate_object(payload_prefix, zero_nonce, payload_suffix);
        const size_t nonce_offset = payload_prefix.size() + std::string("commit ").size() +
                                    std::to_string(payload_prefix.size() + zero_nonce.size() +
                                                   payload_suffix.size())
                                        .size() +
                                    1;

        const size_t tail_start = (nonce_offset / 64) * 64;
        const int nonce_offset_in_tail = static_cast<int>(nonce_offset - tail_start);
        const int tail_len = static_cast<int>(base_object.size() - tail_start);
        const int max_padded_tail = tail_len + 1 + 63 + 8;
        if (max_padded_tail > kMaxTailBytes) {
            throw std::runtime_error("Tail section too large for the fixed CUDA buffer.");
        }

        uint32_t prefix_state[5];
        sha1_init(prefix_state);
        sha1_process_full_blocks(base_object.data(), tail_start, prefix_state);

        const std::vector<uint8_t> tail_template(base_object.begin() + static_cast<long>(tail_start),
                                                 base_object.end());
        std::cout << "Committer name seed: " << committer.base_name << "\n";
        std::cout << "Nonce digits: " << nonce_digits << " (auto)\n";

        std::string warning;
        MiningResult mining = try_mine_on_cuda(tail_template,
                                               tail_len,
                                               static_cast<int>(tail_start),
                                               static_cast<int>(base_object.size()),
                                               nonce_offset_in_tail,
                                               nonce_digits,
                                               max_nonce,
                                               prefix_state,
                                               prefix,
                                               prefix_arg,
                                               device,
                                               warning);
        if (!warning.empty()) {
            std::cerr << warning << "\n";
        }
        if (!mining.found && !warning.empty()) {
            std::cout << "Mining HEAD for prefix " << prefix_arg << " on CPU with "
                      << std::max(1u, std::thread::hardware_concurrency()) << " threads\n";
            mining = mine_on_cpu(tail_template,
                                 tail_len,
                                 static_cast<int>(tail_start),
                                 static_cast<int>(base_object.size()),
                                 nonce_offset_in_tail,
                                 nonce_digits,
                                 max_nonce,
                                 prefix_state,
                                 prefix);
        }

        if (!mining.found) {
            std::cerr << "No matching nonce found in the " << nonce_digits
                      << "-digit search space.\n";
            return 2;
        }

        const std::string nonce = format_nonce(mining.nonce, nonce_digits);
        const std::string final_message = message_with_final_newline;
        const std::string final_committer_name = committer.base_name + " " + nonce;
        std::vector<uint8_t> final_object = make_candidate_object(payload_prefix, nonce, payload_suffix);
        uint32_t verify_state[5];
        sha1_init(verify_state);

        std::vector<uint8_t> padded = final_object;
        padded.push_back(0x80);
        while ((padded.size() % 64) != 56) {
            padded.push_back(0x00);
        }
        const uint64_t bits = static_cast<uint64_t>(final_object.size()) * 8u;
        for (int i = 7; i >= 0; --i) {
            padded.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xffu));
        }
        sha1_process_full_blocks(padded.data(), padded.size(), verify_state);

        if (!matches_prefix_host(verify_state, prefix)) {
            throw std::runtime_error("Verification failed: mined hash does not match the prefix.");
        }

        const std::string author_date =
            trim_trailing_newlines(run_command("git log -1 --format=%aI HEAD"));
        const std::string committer_date =
            trim_trailing_newlines(run_command("git log -1 --format=%cI HEAD"));

        std::cout << "Matched hash:  " << hex_digest(verify_state) << "\n";
        std::cout << "Matched nonce: " << nonce << "\n";
        std::cout << "Matched committer name: " << final_committer_name << "\n";
        std::cout << "\nUse this exact commit message:\n";
        std::cout << "-----BEGIN COMMIT MESSAGE-----\n";
        std::cout << final_message;
        std::cout << "-----END COMMIT MESSAGE-----\n";
        std::cout << "\nUse this exact GIT_COMMITTER_NAME:\n";
        std::cout << final_committer_name << "\n";
        std::cout << "\nSuggested amend command:\n";
        std::cout << "GIT_AUTHOR_DATE=" << shell_single_quote(author_date)
                  << " GIT_COMMITTER_DATE=" << shell_single_quote(committer_date)
                  << " GIT_COMMITTER_EMAIL=" << shell_single_quote(committer.email)
                  << " GIT_COMMITTER_NAME=" << shell_single_quote(final_committer_name)
                  << " git commit --amend --allow-empty --no-gpg-sign "
                     "--cleanup=verbatim -F - <<'__GITMINER_MESSAGE__'\n";
        std::cout << final_message;
        std::cout << "__GITMINER_MESSAGE__\n";

        const std::string mined_hash = hex_digest(mining.hash.data());
        if (mined_hash != hex_digest(verify_state)) {
            std::cout << "Host verification hash:  " << hex_digest(verify_state) << "\n";
            std::cout << "Backend match hash:      " << mined_hash << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
