#include "strata/program/conversation_store.hpp"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::program {
namespace fs = std::filesystem;
namespace {
bool owned_name(const std::string& name) {
    return name.size()==27 && name.substr(0,7)=="branch-" && name.substr(23)==".bin" &&
        name.find_first_not_of("0123456789abcdef",7)>=23;
}
uint64_t wire_bytes(const PrefixFile& f,const BorrowedPrefixBlobs& borrowed) {
    if (f.blobs.size()>256 || borrowed.size()>256-f.blobs.size())
        throw std::runtime_error("too many state records");
    uint64_t n=8+16+f.identity.size()+16+f.tokens.size()*4+16+f.positions.size()*8+8+8;
    for (const auto& b:borrowed) n+=16+b.get().size();
    for (const auto& b:f.blobs) n+=16+b.size();
    return n;
}
bool extends(const PrefixFile& older,const PrefixFile& newer) {
    return older.tokens.size()<=newer.tokens.size() &&
        std::equal(older.tokens.begin(),older.tokens.end(),newer.tokens.begin());
}
}
ConversationStore::~ConversationStore() {
#if !defined(_WIN32)
    if (lock_>=0) { flock(lock_,LOCK_UN); close(lock_); }
#endif
}
uint64_t ConversationStore::bytes() const {
    uint64_t n=0; for (const auto& e:entries_) n+=e.bytes; return n;
}
bool ConversationStore::open(const std::string& dir,const std::string& identity,size_t slots,
                             uint64_t budget,std::string& error) {
#if defined(_WIN32)
    error="conversation disk cache currently requires Linux";return false;
#else
    if (lock_>=0) { error="cache already open";return false; }
    try {
        if (dir.empty() || identity.empty() || slots<1 || slots>64 || budget<1)
            throw std::runtime_error("invalid conversation cache configuration");
        dir_=fs::absolute(dir).lexically_normal().string();identity_=identity;slots_=slots;budget_=budget;
        if (!fs::exists(fs::symlink_status(dir_))) {
            fs::create_directories(dir_);fs::permissions(dir_,fs::perms::owner_all);
        }
        struct stat st{};
        if (lstat(dir_.c_str(),&st) || !S_ISDIR(st.st_mode) || st.st_uid!=geteuid() || (st.st_mode&077))
            throw std::runtime_error("cache directory must be private, owned, and not a symlink");
        lock_=::open((dir_+"/.lock").c_str(),O_CREAT|O_RDWR|O_NOFOLLOW|O_CLOEXEC,0600);
        if (lock_<0 || fstat(lock_,&st) || !S_ISREG(st.st_mode) || st.st_uid!=geteuid() || (st.st_mode&077) ||
            flock(lock_,LOCK_EX|LOCK_NB)) throw std::runtime_error("cache directory locked or invalid lock file");
        for (const auto& item:fs::directory_iterator(dir_)) {
            const auto name=item.path().filename().string();
            if (!fs::is_regular_file(item.symlink_status())) continue;
            // Only our atomic temporary files, in an exclusively locked private directory.
            if (name.size()==38 && owned_name(name.substr(0,27)) && name.substr(27,5)==".tmp.") {
                fs::remove(item.path());continue;
            }
            if (!owned_name(name)) continue;
            PrefixFile meta;std::string why;
            if (!prefix_read(item.path().string(),meta,true,why,kConversationMaxBytes)) {
                fs::remove(item.path());continue;
            }
            // A different executable or inference setting is a safe miss, not corruption.
            // Keep valid snapshots for rollback; never index, load or evict another identity.
            if (meta.identity!=identity_) continue;
            entries_.push_back({item.path().string(),std::move(meta),item.file_size(),0});
        }
        std::sort(entries_.begin(),entries_.end(),[](const Entry& a,const Entry& b) {
            return fs::last_write_time(a.path)<fs::last_write_time(b.path);
        });
        for (auto& e:entries_) e.used=++clock_;
        while (entries_.size()>slots_ || bytes()>budget_) {
            const auto before=entries_.size();discard(entries_.front().path);
            if (entries_.size()==before) throw std::runtime_error("cannot enforce cache budget");
            ++evicted_;
        }
        return true;
    } catch (const std::exception& e) {
        if (lock_>=0) {close(lock_);lock_=-1;}
        entries_.clear();error=e.what();return false;
    }
#endif
}
ConversationHit ConversationStore::find(const std::vector<int64_t>& tokens,int64_t better_than,
                                        const std::vector<std::string>& excluded) const {
    ConversationHit hit;
    for (const auto& e:entries_) {
        if (std::find(excluded.begin(),excluded.end(),e.path)!=excluded.end()) continue;
        const auto pos=prefix_match(e.meta,tokens);
        if (pos>better_than) {hit={e.path,pos};better_than=pos;}
    }
    return hit;
}
bool ConversationStore::contains(const std::vector<int32_t>& tokens) const {
    for (const auto& e : entries_) if (e.meta.tokens == tokens) {
        std::error_code ec;
        const auto st = fs::symlink_status(e.path, ec);
        if (!ec && fs::is_regular_file(st) && fs::file_size(e.path, ec) == e.bytes && !ec) return true;
    }
    return false;
}
void ConversationStore::discard(const std::string& path) {
    const auto it=std::find_if(entries_.begin(),entries_.end(),[&](const Entry& e){return e.path==path;});
    if (it==entries_.end()) return;
    std::error_code error;fs::remove(it->path,error);
    if (!error) entries_.erase(it);
}
bool ConversationStore::load(const ConversationHit& hit,PrefixFile& f,std::string& error,
                             const std::function<bool()>& cancelled) {
    const auto it=std::find_if(entries_.begin(),entries_.end(),[&](const Entry& e){return e.path==hit.path;});
    if (it==entries_.end()) {error="branch no longer cached";return false;}
    if (!prefix_read(hit.path,f,false,error,kConversationMaxBytes,cancelled)) {
        if (error!="cancelled") discard(hit.path);
        return false;
    }
    if (f.identity!=identity_ || f.tokens!=it->meta.tokens || f.positions!=it->meta.positions) {
        error="branch metadata changed";discard(hit.path);return false;
    }
    it->used=++clock_;
    std::error_code ec;fs::last_write_time(it->path,fs::file_time_type::clock::now(),ec);
    return true;
}
bool ConversationStore::put(const PrefixFile& f,const std::string& protected_path,std::string& error,
                            const std::function<bool()>& cancelled,const BorrowedPrefixBlobs& borrowed) {
    try {
        if (lock_<0 || f.identity!=identity_ || f.tokens.empty() || f.positions.empty() ||
            f.positions.back()!=f.tokens.size()) throw std::runtime_error("invalid branch metadata");
        if (cancelled && cancelled()) throw std::runtime_error("cancelled");
        const auto need=wire_bytes(f,borrowed);
        if (need>budget_ || need>kConversationMaxBytes) throw std::runtime_error("branch exceeds cache byte budget");
        if (contains(f.tokens)) return true;
        // The index can outlive a file removed/truncated outside the engine. Do not
        // report a successful duplicate save when no usable snapshot remains.
        std::vector<std::string> stale;
        for (const auto& e:entries_) if (e.meta.tokens==f.tokens) stale.push_back(e.path);
        for (const auto& path:stale) {
            if (path==protected_path) throw std::runtime_error("incoming branch file changed");
            const auto before=entries_.size();discard(path);
            if (entries_.size()==before) throw std::runtime_error("cannot remove stale cache entry");
        }
        // Reserve space before writing: completed snapshots + temporary never exceed the byte budget.
        if (fs::space(dir_).available<need+(256ull<<20)) throw std::runtime_error("insufficient free disk space");
        while (entries_.size()>=slots_ || bytes()>budget_-need) {
            auto victim=entries_.end();
            for (auto it=entries_.begin();it!=entries_.end();++it) {
                if (it->path==protected_path) continue;
                if (victim==entries_.end() || (extends(it->meta,f) && !extends(victim->meta,f)) ||
                    (extends(it->meta,f)==extends(victim->meta,f) && it->used<victim->used)) victim=it;
            }
            if (victim==entries_.end()) throw std::runtime_error("budget reserved for incoming branch");
            const auto before=entries_.size();discard(victim->path);
            if (entries_.size()==before) throw std::runtime_error("cannot evict cache file");
            ++evicted_;
        }
        std::ostringstream name;name<<dir_<<"/branch-"<<std::hex<<std::setfill('0')<<std::setw(16)
            <<(prefix_hash(f.tokens.data(),f.tokens.size()*4)^prefix_hash(identity_.data(),identity_.size()))<<".bin";
        if (!prefix_write(name.str(),f,error,kConversationMaxBytes,cancelled,borrowed)) return false;
        std::vector<std::string> superseded;
        for (const auto& e:entries_) if (e.path!=protected_path && extends(e.meta,f)) superseded.push_back(e.path);
        for (const auto& path:superseded) discard(path);
        PrefixFile meta;meta.identity=f.identity;meta.tokens=f.tokens;meta.positions=f.positions;
        entries_.push_back({name.str(),std::move(meta),need,++clock_});
        return true;
    } catch (const std::exception& e) {error=e.what();return false;}
}
}
