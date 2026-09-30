#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
namespace strata::program {
using PrefixBlob = std::vector<uint8_t>;
struct PrefixFile {
    std::string identity;
    std::vector<int32_t> tokens;
    std::vector<uint64_t> positions;
    std::vector<PrefixBlob> blobs;
};
inline constexpr uint64_t kPrefixMaxBytes = 2ull << 30;
bool prefix_is_system(const std::vector<int64_t>& tokens);
uint64_t prefix_hash(const void* p, size_t n);
bool prefix_write(const std::string& path, const PrefixFile& file, std::string& error,
                  uint64_t max_bytes = kPrefixMaxBytes, const std::function<bool()>& cancelled = {});
bool prefix_read(const std::string& path, PrefixFile& file, bool metadata_only, std::string& error,
                 uint64_t max_bytes = kPrefixMaxBytes, const std::function<bool()>& cancelled = {});
int64_t prefix_match(const PrefixFile& file, const std::vector<int64_t>& tokens);
}
