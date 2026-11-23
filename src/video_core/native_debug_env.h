// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdlib>
#include <optional>

namespace VideoCommon {

// The native (Switch 2) GPU debug switches, read from the environment once. std::getenv takes the
// CRT environment lock and scans every variable, and these are checked per draw, macro and query:
// calling it each time cost two thirds of the GPU thread in Pokopia.
inline bool NativeDrawDiagnostics() {
    static const bool enabled = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
    return enabled;
}

inline bool NativeFrameCapture() {
    static const bool enabled = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    return enabled;
}

inline bool NativeDrawCapture() {
    static const bool enabled = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
    return enabled;
}

// Dump the stages of every pipeline whose shader build throws. Off by default: Ocarina of Time has
// ~1,600 lowered programs whose bindless handles cannot be tracked, and dumping each failed
// pipeline (control-flow re-read plus a file write) stutters the play build.
inline bool DumpFailedShaders() {
    static const bool enabled = std::getenv("EDEN_DUMP_FAILED_SHADERS") != nullptr;
    return enabled;
}

// Bloom scale override for the titles in Tegra::Nx2Bloom (EDEN_NX2_BLOOM_SCALE=0..1, e.g. 0 = no
// bloom, 1 = the game's own strength); unset = the title's default factor.
inline std::optional<float> BloomScaleOverride() {
    static const std::optional<float> value = []() -> std::optional<float> {
        const char* text = std::getenv("EDEN_NX2_BLOOM_SCALE");
        if (!text || !*text) {
            return std::nullopt;
        }
        char* end = nullptr;
        const float parsed = std::strtof(text, &end);
        if (end == text || parsed < 0.0f || parsed > 4.0f) {
            return std::nullopt;
        }
        return parsed;
    }();
    return value;
}

} // namespace VideoCommon
