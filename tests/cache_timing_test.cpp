#include "strata/program/cache_timing.hpp"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <thread>
using strata::program::CacheTiming;
void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
int main(){
 CacheTiming timing;
 auto outer=timing.measure(CacheTiming::Phase::selection);
 std::this_thread::sleep_for(std::chrono::milliseconds(5));
 {
  auto nested=timing.measure(CacheTiming::Phase::selection);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
 }
 check(timing.milliseconds(CacheTiming::Phase::selection)==0,"nested scope does not double count outer wall time");
 outer.stop();
 check(timing.milliseconds(CacheTiming::Phase::selection)>=9,"selection wall time is measured");
 const auto saved=timing.milliseconds(CacheTiming::Phase::selection);
 outer.stop();check(timing.milliseconds(CacheTiming::Phase::selection)==saved,"stopping a scope twice is harmless");
 check(timing.fields().find("cache_select_ms=")!=std::string::npos,"stable tagged wire extension");
 check(timing.fields().find("first_generated_ms=")==std::string::npos,"cancelled/empty request has no invented first token");
 timing.first_generated(12.5);timing.first_generated(99);
 check(timing.fields().find("first_generated_ms=12.500")!=std::string::npos,"first generated token is only recorded once");
 timing={};check(timing.milliseconds(CacheTiming::Phase::selection)==0,"request resets do not leak measurements");
 check(timing.fields().find("first_generated_ms=")==std::string::npos,"request resets clear first token");
 std::puts("cache_timing_test: PASS");
}
