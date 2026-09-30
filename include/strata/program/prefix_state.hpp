#pragma once
#include "strata/program/prefix_file.hpp"
#include "strata/core/session.hpp"
#include "strata/core/mtp.hpp"
namespace strata::program {
// v1 deliberately supports the tested fully-resident INT8 KV layout only.
bool prefix_state_capture(const core::SessionState&, const core::ModelGeometry&, const core::MtpDrafter&,
                          int64_t root, std::vector<PrefixBlob>&, std::string&,
                          const std::function<bool()>& cancelled = {}, uint64_t max_bytes = kPrefixMaxBytes);
bool prefix_state_validate(const core::SessionState&, const core::ModelGeometry&, const core::MtpDrafter&,
                           int64_t root, const std::vector<PrefixBlob>&, size_t offset, std::string&);
bool prefix_state_restore(const core::SessionState&, const core::ModelGeometry&, const core::MtpDrafter&,
                          int64_t root, const std::vector<PrefixBlob>&, size_t offset, std::string&);
}
