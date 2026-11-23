// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <span>
#include "common/common_types.h"
#include "common/swap.h"

namespace Service::Nvidia::Devices {

inline constexpr size_t NativeGpuCharacteristicsSize = 0xd0;

// Built-in NVN program sets are chosen by SM version. Pokopia/HWAI SDKs accept 0x806-0x809, each
// selecting a different set (poko sdk+0x5f8420); Ocarina of Time's newer SDK accepts only 0x808
// (nnSdk+0x674560) and fails nvnDeviceInitialize otherwise. Older SDKs keep 0x807.
inline constexpr u32 NativeSmVersion = 0x807;
inline constexpr u32 NativeSmVersionNewSdk = 0x808;
// Ocarina of Time's NVN sets channel preemption modes after creating each queue channel
// (nnSdk+0x670190); NvRm sends 'H' 0x1f only when the channel level is at least 0x1f
// (nnSdk+0x634078) and fails the queue otherwise. Higher levels open further, unanalysed gates.
inline constexpr s16 NativeChannelLevel = 0x1d;
inline constexpr s16 NativeChannelLevelNewSdk = 0x1f;
// Subcontexts per TSG are K[0xb4] only when the TSG level is above 0xa and flag bits 8 and 22 are
// set (nnSdk+0x632c48), else 1. Ocarina of Time puts two or three subcontext queues on its main
// queue's TSG (Owl.nss+0xc31190), so the second failed with 0x11. 0x40 is the chip limit
// (nnSdk+0x632dd0). Level 0xb also makes NvRm bind channels with 'T' 0xb; 0xc and up open
// further, unanalysed ioctls.
inline constexpr s16 NativeTsgLevel = 2;
inline constexpr s16 NativeTsgLevelNewSdk = 0xb;
inline constexpr u32 NativeTsgSubcontextsNewSdk = 0x40;

inline std::array<u8, NativeGpuCharacteristicsSize> MakeNativeGpuCharacteristics(
    std::span<const u8, 0xa0> legacy, u32 sm_version = NativeSmVersion,
    s16 channel_level = NativeChannelLevel, s16 tsg_level = NativeTsgLevel,
    u32 max_subcontexts = 0) {
    std::array<u8, NativeGpuCharacteristicsSize> result{};
    // The GPU description and engine classes share the legacy prefix. The native
    // SDK consumes signed ioctl interface levels where the legacy ABI has topology.
    std::memcpy(result.data(), legacy.data(), 0x70);
    // Select the SDK's native control ABI, whose normal mappings support
    // 4 KiB CPU/GPU aliases. Engine execution remains a separate implementation.
    const u32_le native_arch{0x170};
    const u32_le native_impl{0xe};
    std::memcpy(result.data(), &native_arch, sizeof(native_arch));
    std::memcpy(result.data() + 4, &native_impl, sizeof(native_impl));
    // Native NVN's built-in shader table accepts Ampere SM 8.x. Carrying the
    // inherited Maxwell SM 5.3 value here rejects its initialization shaders.
    const u32_le native_sm_version{sm_version};
    std::memcpy(result.data() + 0x3c, &native_sm_version, sizeof(native_sm_version));
    std::memcpy(result.data() + 0x40, &native_sm_version, sizeof(native_sm_version));
    // Native NVN maps CPU buffers into its shared VA reservation. Those buffers
    // are 4 KiB aligned, so large pages must remain optional for allocation.
    const u32_le preferred_page_size{0x1000};
    std::memcpy(result.data() + 0x28, &preferred_page_size, sizeof(preferred_page_size));
    u32_le available_page_sizes;
    std::memcpy(&available_page_sizes, legacy.data() + 0x34, sizeof(available_page_sizes));
    available_page_sizes = (static_cast<u32>(available_page_sizes) & 0x10000) | 0x1000;
    std::memcpy(result.data() + 0x34, &available_page_sizes, sizeof(available_page_sizes));
    u64_le flags;
    std::memcpy(&flags, legacy.data() + 0x50, sizeof(flags));
    // Native SDK class enumeration filters 2D/3D/compute/inline families
    // with bits 40..43 before selecting context identities from its profile.
    // The native SDK gates ModifyKind (AS ioctl 12) on capability bit 34.
    flags = static_cast<u64>(flags) | (1ULL << 8) | (1ULL << 26) | (1ULL << 34) | (0xfULL << 40);
    if (max_subcontexts != 0) {
        flags = static_cast<u64>(flags) | (1ULL << 22);
        const u32_le subcontexts{max_subcontexts};
        std::memcpy(result.data() + 0xb4, &subcontexts, sizeof(subcontexts));
    }
    std::memcpy(result.data() + 0x50, &flags, sizeof(flags));
    const std::array<s16, 9> levels{
        0x1c,          // GPU control: GetGpuTime
        tsg_level,     // TSG: bind/unbind channel (2), bind with subcontext 'T' 0xb (0xb)
        -1,            // GPU debugger device is unavailable
        channel_level, // Channel: SetTimeslice (0x1d), SetPreemptionMode (0x1f)
        0x14,          // Address space: Remap
        0, 0, 0, 0, // Additional interfaces have not been implemented.
    };
    for (size_t i = 0; i < levels.size(); ++i) {
        const u16_le value{static_cast<u16>(levels[i])};
        std::memcpy(result.data() + 0x70 + i * sizeof(value), &value, sizeof(value));
    }
    return result;
}

} // namespace Service::Nvidia::Devices
