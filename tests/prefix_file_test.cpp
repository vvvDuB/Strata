#include "strata/program/prefix_file.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <chrono>
using namespace strata::program;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string dir=std::string(argv[1])+"/case-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(dir);
    std::string path=dir+"/roundtrip.bin", err;
    PrefixFile x; x.identity="model+build+kv"; x.tokens={10,20,30,40};
    x.positions={2,4}; x.blobs={{1,2,3},{4,5,6,7}};
    int fail=0;
    auto check=[&](bool ok,const char* what){std::printf("%s %s\n",ok?"PASS":"FAIL",what);fail+=!ok;};
    check(prefix_is_system({248045,8678,198,123}),"system header accepted");
    check(!prefix_is_system({248045,846,198,123}),"user-first request never persisted");
    check(!prefix_is_system({248045,74455,198,123}),"assistant-first request never persisted");
    check(!prefix_is_system({248045,8678}),"incomplete header rejected");
    check(!prefix_is_system({}),"empty header rejected");
    check(prefix_match(x,{10,20,99,100})==2,"partial prefix checkpoint");
    check(prefix_match(x,{10,20,30,40,50})==4,"complete root");
    check(prefix_match(x,{10,20,30,40})==2,"leave final prompt token for decode");
    check(prefix_match(x,{99,20,30})==0,"changed first token is a miss");
    check(prefix_write(path,x,err),"atomic write");
    PrefixFile y;
    check(prefix_read(path,y,false,err),"read back");
    check(y.identity==x.identity && y.tokens==x.tokens && y.positions==x.positions && y.blobs==x.blobs,"exact roundtrip");
    PrefixFile meta;
    check(prefix_read(path,meta,true,err) && meta.blobs.empty() && meta.positions==x.positions,"metadata-only index");
    check(!prefix_write(path,x,err),"never overwrite existing cache");
    if (std::filesystem::exists(path)) {
        auto perms=std::filesystem::status(path).permissions();
        check((perms & (std::filesystem::perms::group_all|std::filesystem::perms::others_all))==std::filesystem::perms::none,"private file");
        std::fstream f(path,std::ios::binary|std::ios::in|std::ios::out);
        f.seekp(-9,std::ios::end);f.put('X');f.close();
        check(!prefix_read(path,y,false,err),"corrupt payload rejected");
        std::filesystem::resize_file(path,12);
        check(!prefix_read(path,y,false,err),"truncated file rejected");
    }
    return fail?1:0;
}
