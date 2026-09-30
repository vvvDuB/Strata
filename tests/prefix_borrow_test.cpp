#include "strata/program/conversation_store.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
using namespace strata::program;
namespace fs = std::filesystem;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto dir = fs::path(argv[1]) / ("borrow-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    std::string error;
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); failures += !ok;
    };
    PrefixBlob first(1024, 13), second(511, 42), empty;
    const BorrowedPrefixBlobs loans{first, empty, second};
    PrefixFile f; f.identity = "borrow-model"; f.tokens = {10,20,30,40}; f.positions = {2,4};
    f.blobs = {{1,2,3,4}}; // captured KV follows the checkpoint records on disk
    auto owned = f; owned.blobs = {first, empty, second, f.blobs[0]};
    const auto expected = dir / "owned.bin", actual = dir / "borrowed.bin";
    check(prefix_write(expected.string(), owned, error), "owned reference write");
    check(prefix_write(actual.string(), f, error, kPrefixMaxBytes, {}, loans), "borrowed write");
    auto bytes = [](const fs::path& path) {
        std::ifstream in(path, std::ios::binary); return std::string(std::istreambuf_iterator<char>(in), {});
    };
    check(bytes(expected) == bytes(actual), "borrowed and owned files are byte-identical including hashes/order");
    PrefixFile loaded;
    check(prefix_read(actual.string(), loaded, false, error) && loaded.blobs == owned.blobs,
          "ordinary reader reconstructs complete owned payload from borrowed write");
    check(f.blobs.size() == 1 && first == PrefixBlob(1024,13) && second == PrefixBlob(511,42),
          "writer does not consume or modify borrowed checkpoints");
    const auto limited = dir / "limited.bin";
    check(!prefix_write(limited.string(), f, error, fs::file_size(expected)-1, {}, loans) && !fs::exists(limited),
          "byte limit includes borrowed records and leaves no published file");
    BorrowedPrefixBlobs too_many(256, first);
    const auto excess = dir / "excess.bin";
    check(!prefix_write(excess.string(), f, error, kPrefixMaxBytes, {}, too_many) && !fs::exists(excess),
          "record cap counts borrowed plus owned payloads");
    int polls = 0;
    const auto cancelled = dir / "cancelled.bin";
    check(!prefix_write(cancelled.string(), f, error, kPrefixMaxBytes, [&]{return ++polls > 12;}, loans) &&
          error == "cancelled" && !fs::exists(cancelled), "cancellation while encoding loans cannot publish partial snapshot");
    {
        ConversationStore store;
        const auto cache = dir / "cache";
        const auto need = fs::file_size(expected);
        check(store.open(cache.string(), f.identity, 2, need*2, error), "open borrowed store");
        check(store.put(f, "", error, {}, loans) && store.bytes() == need,
              "store admission and accounting include borrowed bytes");
        const auto hit = store.find({10,20,30,40,99});
        check(hit.position == 4 && store.load(hit, loaded, error) && loaded.blobs == owned.blobs,
              "borrowed store produces ordinary restorable snapshot");
        auto next = f; next.tokens[2] = 31;
        BorrowedPrefixBlobs huge{first, first, first, first};
        check(4*first.size() > need*2, "oversized fixture payload alone exceeds the store budget");
        const auto before = store.bytes();
        check(!store.put(next, "", error, {}, huge) && store.bytes() == before && store.size() == 1,
              "oversized loans rejected before evicting existing valid branch");
        check(!store.put(next, "", error, []{return true;}, loans) && store.bytes() == before,
              "cancelled put leaves borrowed sources and stored branch unchanged");
    }
    for (const auto& file : fs::directory_iterator(dir))
        check(file.path().filename().string().find(".tmp.") == std::string::npos, "failed writer removes private temporary files");
    return failures ? 1 : 0;
}
