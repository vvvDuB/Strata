#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#endif
namespace fs = std::filesystem;
namespace cpu = strata::kernels::cpu;
void require(bool condition, const std::string& why) { if (!condition) throw std::runtime_error(why); }
void write_experts(const fs::path& dir, const std::vector<uint64_t>& sizes) {
    std::ofstream out(dir / "experts.bin", std::ios::binary);
    for (size_t l=0; l<sizes.size(); ++l) for (int e=0; e<2; ++e) {
        std::vector<char> bytes(sizes[l], char(1+l*2+e));
        out.write(bytes.data(), bytes.size());
    }
    require(bool(out), "write fixture");
}
void check_mapping(strata::core::FileExpertSource& src, const std::vector<uint64_t>& sizes, bool check_overflow=true) {
    uint64_t off=0;
    const auto base=reinterpret_cast<uintptr_t>(src.blob(0,0));
    for (size_t l=0; l<sizes.size(); ++l) {
        for (int e=0; e<2; ++e) {
            const auto* p=src.blob(l,e);
            require(p && reinterpret_cast<uintptr_t>(p)==base+off+e*sizes[l], "wrong expert offset");
            require(p[0]==1+l*2+e && p[sizes[l]-1]==1+l*2+e, "wrong expert bytes");
        }
        off+=2*sizes[l];
    }
    for (auto pair : std::vector<std::pair<int64_t,int64_t>>{{-1,0},{0,-1},{2,0},{0,2}})
        require(src.blob(pair.first,pair.second)==nullptr, "invalid coordinate accepted");
    if (check_overflow) require(!src.blob(INT64_MAX,0) && !src.blob(0,INT64_MAX), "overflow coordinate accepted");
    src.close();
    require(!src.mapped() && src.blobs()==0 && src.reads()==0 && !src.blob(0,0), "close did not reset source");
#ifndef _WIN32
    const uintptr_t page=uintptr_t(sysconf(_SC_PAGESIZE));
    unsigned char resident=0;
    errno=0;
    require(mincore(reinterpret_cast<void*>((base+off-1)/page*page),page,&resident)==-1 && errno==ENOMEM,
            "last page still mapped after close");
#endif
    src.close();
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"supply a NEW empty test directory");
        const fs::path dir(argv[1]);
        require(fs::create_directory(dir),"test directory must not already exist");
        std::string err;
        strata::core::FileExpertSource src;
        // Canonical behavior must remain intact.
        require(cpu::expert_layout_load(dir.string(),2,2,err),err);
        write_experts(dir,{cpu::BLOB,cpu::BLOB});
        require(src.open(dir.string(),2,2,err),"canonical open: "+err);
        check_mapping(src,{cpu::BLOB,cpu::BLOB});
        // Native layers deliberately have different blob sizes (actual model formats).
        cpu::NativeFmt a,b;
        require(cpu::native_fmt(17,20,cpu::H,cpu::FF,a,err),err);
        require(cpu::native_fmt(16,42,cpu::H,cpu::FF,b,err),err);
        { std::ofstream layout(dir/"native_experts.txt");
          layout << "0 17 20 0 " << a.bytes << "\n1 16 42 " << 2*a.bytes << ' ' << b.bytes << '\n'; }
        require(cpu::expert_layout_load(dir.string(),2,2,err),err);
        write_experts(dir,{a.bytes,b.bytes});
        require(src.open(dir.string(),2,2,err),"native open: "+err);
        // A later global layout change must not change this source's offsets or unmap size.
        const fs::path canonical=dir/"canonical"; fs::create_directory(canonical);
        require(cpu::expert_layout_load(canonical.string(),1,1,err),err);
        check_mapping(src,{a.bytes,b.bytes});
        require(cpu::expert_layout_load(dir.string(),2,2,err),err);
        require(!src.open(dir.string(),1,2,err) && !src.mapped(),"geometry mismatch accepted");
        require(!src.open(dir.string(),0,2,err),"empty geometry accepted");
        require(cpu::expert_layout_load(canonical.string(),1,1,err),err);
        require(!src.open(canonical.string(),INT64_MAX,2,err),"geometry overflow accepted");
        require(!src.open(canonical.string(),INT64_MAX,1,err),"byte-size overflow accepted");
        require(cpu::expert_layout_load(dir.string(),2,2,err),err);
        const auto total=2*(a.bytes+b.bytes);
        for (auto bytes : {total-1,total+1}) {
            fs::resize_file(dir/"experts.bin",bytes);
            require(!src.open(dir.string(),2,2,err) && !src.mapped(),"incorrect file size accepted");
        }
        write_experts(dir,{a.bytes,b.bytes});
        require(src.open(dir.string(),2,2,err),"reopen failed: "+err);
        check_mapping(src,{a.bytes,b.bytes});
        std::cout << "PASS: canonical/native sizes, offsets, boundaries, full unmap, layout isolation, reopen\n";
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
