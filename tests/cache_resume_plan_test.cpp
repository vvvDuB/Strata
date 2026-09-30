#include "strata/program/cache_resume_plan.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
using namespace strata::program::cache_resume;
void check(bool ok,const char* what){if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
int main(){
 std::vector<Candidate> choices={{Source::active,8000},{Source::ram,48000},{Source::legacy_disk,56000}};
 check(select(choices).source==Source::legacy_disk,"longest prefix wins before materialization");
 choices.push_back({Source::shared_disk,60000});check(select(choices).source==Source::shared_disk,"all tiers considered");
 choices.push_back({Source::system_disk,61000});check(select(choices).source==Source::system_disk,"system candidate competes globally");
 choices={{Source::ram,8000},{Source::active,8000},{Source::legacy_disk,8000}};
 check(select(choices).source==Source::active,"active state has no transfer on a tie");
 choices.erase(choices.begin()+1);check(select(choices).source==Source::ram,"RAM beats disk on a tie");
 choices={{Source::legacy_disk,9000,0,9000},{Source::shared_disk,9000,0,8000}};
 check(select(choices).source==Source::shared_disk,"equal disk prefixes prefer smaller payload");
 choices={{Source::active,0},{Source::ram,-1}};check(select(choices).tokens==0,"negative/empty candidates are misses");
 choices={{Source::active,8},{Source::ram,20}};choices.pop_back();check(select(choices).tokens==8,"invalid winner falls back without restoring intermediate state");
 check(!should_preserve(40027,40022,true),"exact repeat does not capture generated tail");
 check(!should_preserve(40027,39981,true),"small formatting/assistant-tail rewind is not a whole branch switch");
 check(!should_preserve(40027,40027,true),"append does not park active branch");
 check(should_preserve(40027,39445,true),"real history edit still preserves old positional branch");
 check(should_preserve(40027,3,true),"A/B switch captures outgoing state");
 check(should_preserve(40027,40022,false),"image/control mismatch never uses token-only fast path");
 check(should_preserve(8000,1000,true),"deep truncation preserves returnable branch");
 check(!should_preserve(0,0,false),"no live branch to preserve");
 check(final_checkpoint(4199,4096,12),"raw tail gap is checkpointed before generation");
 check(!final_checkpoint(4199,4199,12),"existing final point is not captured twice");
 check(!final_checkpoint(40021,40015,12),"short chat header is cheaper than another full checkpoint");
 check(!final_checkpoint(4199,4096,0),"disabled prompt cache remains disabled");
 check(!final_checkpoint(4199,4096,1),"one slot must not evict its root for an opportunistic leaf");
 std::puts("cache_resume_plan_test: PASS");
}
