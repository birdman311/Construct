// SPDX-FileCopyrightText: 2026 birdman311
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "common/logging.h"
#include "common/settings.h"

namespace VideoCommon {

// Construct: shaders built at one resolution scale were seen to come back corrupted after the
// scale changed. Each game's shader folder remembers the scale its cache was built at; when the
// game starts at a different scale (global or per-game setting), the folder's cache files are
// deleted first so everything is rebuilt cleanly. A cache without a record was built at an
// unknown scale, so it is cleared once too.
inline void ClearShaderCacheIfResolutionChanged(const std::filesystem::path& base_dir) {
    const auto marker = base_dir / "construct_resolution.txt";
    const std::string current =
        std::to_string(static_cast<unsigned>(Settings::values.resolution_setup.GetValue()));

    std::string recorded;
    if (std::ifstream in{marker}; in) {
        std::getline(in, recorded);
    }
    if (recorded != current) {
        std::error_code ec;
        size_t removed = 0;
        for (const auto& entry : std::filesystem::directory_iterator(base_dir, ec)) {
            if (entry.path() != marker && entry.is_regular_file(ec) &&
                std::filesystem::remove(entry.path(), ec)) {
                ++removed;
            }
        }
        if (removed > 0) {
            LOG_INFO(Render, "Resolution scale changed ({} -> {}): cleared {} shader cache file(s)",
                     recorded.empty() ? "unknown" : recorded, current, removed);
        }
        std::ofstream{marker, std::ios::trunc} << current << '\n';
    }
}

} // namespace VideoCommon
