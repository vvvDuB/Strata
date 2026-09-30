#pragma once
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
namespace strata::program {
// Serving-thread wall clocks, not CUDA event durations. Nested scopes of the
// same phase count only once; asynchronous work is counted when it blocks here.
class CacheTiming {
 using Clock=std::chrono::steady_clock;
public:
 enum class Phase {selection,parking,read,restore,checkpoint,replay_prefill,new_prefill,mixed_prefill,count};
 class Scope {
 public:
  Scope(CacheTiming& timing,Phase phase):owner_(&timing),phase_(phase){owner_->begin(phase_);}
  Scope(const Scope&)=delete;
  Scope& operator=(const Scope&)=delete;
  ~Scope(){stop();}
  void stop(){if(owner_){owner_->end(phase_);owner_=nullptr;}}
 private:
  CacheTiming* owner_;
  Phase phase_;
 };
 Scope measure(Phase phase){return Scope(*this,phase);}
 double milliseconds(Phase phase)const{return elapsed_[size_t(phase)];}
 void first_generated(double ms){if(first_<0 && std::isfinite(ms) && ms>=0)first_=ms;}
 std::string fields()const{
  static constexpr std::array<const char*,size_t(Phase::count)> names={
   "cache_select_ms","cache_park_ms","cache_read_ms","cache_restore_ms","checkpoint_ms",
   "prefill_replay_ms","prefill_new_ms","prefill_mixed_ms"};
  std::ostringstream out;out<<std::fixed<<std::setprecision(3);
  for(size_t i=0;i<names.size();++i)out<<' '<<names[i]<<'='<<elapsed_[i];
  if(first_>=0)out<<" first_generated_ms="<<first_;
  return out.str();
 }
private:
 std::array<double,size_t(Phase::count)> elapsed_{};
 std::array<unsigned,size_t(Phase::count)> depth_{};
 std::array<Clock::time_point,size_t(Phase::count)> started_{};
 double first_=-1;
 void begin(Phase phase){const auto i=size_t(phase);if(depth_[i]++==0)started_[i]=Clock::now();}
 void end(Phase phase){const auto i=size_t(phase);if(--depth_[i]==0)
  elapsed_[i]+=std::chrono::duration<double,std::milli>(Clock::now()-started_[i]).count();}
};
}
