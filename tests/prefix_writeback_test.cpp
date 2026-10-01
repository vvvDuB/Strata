#include "strata/program/prefix_file.hpp"
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

// A derived cache must not force durable storage on the inference thread.
// Model a slow/unavailable disk barrier without requiring actual disk trouble.
static int barriers = 0;
static bool fail_barrier = true;
static bool close_error = false;
static bool closed_seen = false;
extern "C" int __real_fclose(FILE*);
extern "C" int __wrap_fclose(FILE* file) {
    const int result = __real_fclose(file);
    closed_seen = true;
    if (!close_error) return result;
    close_error = false;
    errno = EIO;
    return EOF;
}
extern "C" int __wrap_fsync(int) {
    ++barriers;
    if (!fail_barrier) return 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    errno = EIO;
    return -1;
}
extern "C" int __wrap_fdatasync(int fd) { return __wrap_fsync(fd); }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    namespace fs = std::filesystem;
    const auto dir = fs::path(argv[1]) / ("writeback-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
        failures += !ok;
    };
    strata::program::PrefixFile source;
    source.identity = "derived-cache";
    source.tokens = {10, 20, 30}; source.positions = {3};
    source.blobs = {{1, 2, 3}, {4, 5}};
    std::string error;
    const auto path = (dir / "snapshot.bin").string();
    check(strata::program::prefix_write(path, source, error), "publish without a durability barrier");
    check(barriers == 0, "no fsync/fdatasync on the request path");
    strata::program::PrefixFile restored;
    check(strata::program::prefix_read(path, restored, false, error) &&
          restored.blobs == source.blobs && restored.tokens == source.tokens,
          "immediately readable complete payload");
    check(!strata::program::prefix_write(path, source, error), "no overwrite");
    const auto cancel_path = (dir / "cancel.bin").string();
    check(!strata::program::prefix_write(cancel_path, source, error,
          strata::program::kPrefixMaxBytes, [] { return true; }) &&
          !fs::exists(cancel_path), "cancellation does not publish");
    // A lost or truncated cache after a power failure must remain a safe miss.
    if (fs::exists(path)) {
        check((fs::status(path).permissions() & (fs::perms::group_all | fs::perms::others_all)) ==
              fs::perms::none, "snapshot is private");
        fs::resize_file(path, 13);
        check(!strata::program::prefix_read(path, restored, false, error), "truncated cache is rejected");
    }
    // The absence of fsync must not weaken ordinary write/close error handling.
    const auto close_path = (dir / "close-error.bin").string();
    fail_barrier = false;
    close_error = true;
    check(!strata::program::prefix_write(close_path, source, error) &&
          !fs::exists(close_path), "close failure does not publish");
    const auto late_path = (dir / "late-cancel.bin").string();
    closed_seen = false;
    check(!strata::program::prefix_write(late_path, source, error,
          strata::program::kPrefixMaxBytes, [] { return closed_seen; }) &&
          !fs::exists(late_path), "cancellation after close does not publish");
    size_t pending = 0;
    for (const auto& entry : fs::directory_iterator(dir))
        pending += entry.path().filename().string().find(".tmp.") != std::string::npos;
    check(pending == 0, "no leftover temporary file");
    return failures ? 1 : 0;
}
