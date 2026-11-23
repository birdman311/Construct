// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <span>

#include "common/common_types.h"
#include "video_core/surface.h"

namespace VideoCommon {

// TEST BUILD ONLY (frame dump tool): turns a downloaded color image into a viewable PNG.

/// Whether WriteFrameDumpImage can convert images of this format.
bool FrameDumpSupportsFormat(VideoCore::Surface::PixelFormat format);

/// Writes level 0 of an image (rows `row_length` pixels apart) as an 8-bit RGB PNG. HDR colors
/// are tone mapped; two-channel float images (motion vectors) are shown as 0.5 + 16 * value;
/// one-channel float images are normalized to their own range.
void WriteFrameDumpImage(const std::filesystem::path& path, VideoCore::Surface::PixelFormat format,
                         u32 width, u32 height, u32 row_length, std::span<const u8> data);

} // namespace VideoCommon
