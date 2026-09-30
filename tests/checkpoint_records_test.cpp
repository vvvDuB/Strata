#include "strata/program/checkpoint_records.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
struct C { std::vector<uint8_t> gdn, ple, tails, dead, block_pos; };
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    using namespace strata::program::checkpoint_records;
    C a{{1,2}, {}, {3,4,5}, {6,7}, {8,9,10,11}};
    Blobs b; append(b,a); append(b,a);
    CHECK(valid(b,2,{2,0,3,2,4}));
    CHECK(!valid(b,std::numeric_limits<size_t>::max(),{2,0,3,2,4}));
    CHECK(!valid(b,2,{2,0,3,1,4}));
    auto truncated=b; truncated.pop_back(); CHECK(!valid(truncated,2,{2,0,3,2,4}));
    Blobs old{a.gdn,a.ple,a.tails}; CHECK(!valid(old,1,{2,0,3,2,4}));
    C out; CHECK(!take(b,2,out)); CHECK(out.gdn.empty()); CHECK(take(b,1,out));
    CHECK(out.gdn==a.gdn && out.ple==a.ple && out.tails==a.tails && out.dead==a.dead && out.block_pos==a.block_pos);
    CHECK(b[0]==a.gdn && b[4]==a.block_pos); CHECK(b[5].empty());
    std::puts("checkpoint_records: checks passed");
}
