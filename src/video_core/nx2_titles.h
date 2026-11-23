// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <optional>

#include "common/common_types.h"

namespace Tegra {

// The running application's program id, set when its GPU channels are created, for the few
// Switch 2 GPU behaviours that differ per game.
inline std::atomic<u64> g_application_program_id{0};

// Switch 2 titles whose guest CPU reads buffer data the GPU computed, so GPU-written buffers are
// copied back to guest memory at every fence (BufferCache::AsyncBufferDownloads). Pokemon Pokopia
// builds its sky, fog and exposure from such values: without the copy-back its sky is black
// (found by the PokeDinner fork; ~58 FPS with it on an RTX 4060).
inline bool Nx2NeedsBufferDownloads() {
    switch (g_application_program_id.load(std::memory_order_relaxed)) {
    case 0x0400A3A01EE4C000ULL: // Pokemon Pokopia
        return true;
    default:
        return false;
    }
}

// Switch 2 titles whose own FidelityFX FSR 1 RCAS sharpen pass is turned off, as the compute shader
// hash of that pass (0 for none). Hyrule Warriors: Age of Imprisonment sharpens its 900p-to-1080p
// upscale at near-maximum strength (0.1 stops), which turns the faint noise its anti-aliasing
// leaves into coarse grain once Eden scales the frame to a PC monitor; Eden's own scaling filter
// sharpens at the screen's resolution instead.
inline u64 Nx2DisabledRcasShader() {
    switch (g_application_program_id.load(std::memory_order_relaxed)) {
    case 0x04003F40232DE000ULL: // Hyrule Warriors: Age of Imprisonment
        return 0x8a2be50605df0a02ULL;
    default:
        return 0;
    }
}

// Switch 2 titles whose bloom is scaled down (EDEN_NX2_BLOOM_SCALE overrides the factor; 1 = the
// game's own look): the fragment shader hash of the pass that adds the blurred bloom pyramid onto
// the scene, and the fragment constant-buffer bank and word holding the bloom intensity it
// multiplies by. Hyrule Warriors: Age of Imprisonment composites a wide (down to 16x7) bloom at full
// strength, which reads as a milky haze over characters once Eden shows the frame on a monitor.
struct Nx2BloomScale {
    u64 composite_fragment;
    u32 bank;
    u32 word;
    float default_scale;
};

inline std::optional<Nx2BloomScale> Nx2Bloom() {
    switch (g_application_program_id.load(std::memory_order_relaxed)) {
    case 0x04003F40232DE000ULL: // Hyrule Warriors: Age of Imprisonment
        return Nx2BloomScale{0x445bdaac309fc7beULL, 5, 19, 1.0f}; // 1 = the game's own look
    default:
        return std::nullopt;
    }
}

// Switch 2 titles whose own FidelityFX FSR 1 EASU upscale is replaced by an averaging downscale
// when Eden renders at a higher resolution scale: the compute shader hash of the EASU pass. EASU
// samples its input with constants the game computed for the unscaled size, so it skips the extra
// samples a 2x render produced; a linear blit from the scaled image into the pass's output averages
// them instead (true supersampling). Hyrule Warriors: 1600x900 (3200x1800 at 2x) -> 1920x1080.
inline u64 Nx2SupersampleEasuShader() {
    switch (g_application_program_id.load(std::memory_order_relaxed)) {
    case 0x04003F40232DE000ULL: // Hyrule Warriors: Age of Imprisonment
        return 0xd1835c1e213e0941ULL;
    default:
        return 0;
    }
}

// Where NVIDIA DLSS (EDEN_DLSS) plugs into a Switch 2 title: the fragment shader hash of the game's
// own final anti-aliasing pass, which DLSS replaces, and the vertex shader hash and constant buffer
// bank holding the scene camera (world to clip, column-major, without jitter) for motion vectors.
struct Nx2DlssHooks {
    u64 final_pass_fragment;
    u64 camera_vertex;
    u32 camera_bank;
};

inline std::optional<Nx2DlssHooks> Nx2Dlss() {
    switch (g_application_program_id.load(std::memory_order_relaxed)) {
    case 0x0400A3A01EE4C000ULL: // Pokemon Pokopia: FXAA 3.11 final pass, terrain vertex shader
        return Nx2DlssHooks{0xaf149421bb361385ULL, 0x10f0ac2edb3a4457ULL, 3};
    case 0x04003F40232DE000ULL: // Hyrule Warriors: Age of Imprisonment: the same FXAA 3.11 pass
                                // (1600x900, before its FSR 1 upscale), G-buffer prop vertex shader
                                // whose gl_Position = c3[0..15] (column-major, c3[28..39] = object)
        return Nx2DlssHooks{0xaf149421bb361385ULL, 0x308db3ea96deb9e8ULL, 3};
    default:
        return std::nullopt;
    }
}

} // namespace Tegra
