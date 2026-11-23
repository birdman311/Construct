// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include "common/container/unordered_map.h"

#include "video_core/dma_pusher.h"

namespace Tegra {

class GPU;

namespace Control {

struct ChannelState;

class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    void Push(GPU& gpu, s32 channel, CommandList&& entries);

    void ResumePending(GPU& gpu);
    [[nodiscard]] bool HasPendingNativeChannels() const { return !pending_native_channels.empty(); }

    void DeclareChannel(std::shared_ptr<ChannelState> new_channel);

    /// Stall watchdog (EDEN_STALL_WATCHDOG): writes every channel's state and the last
    /// submissions to the launch log.
    void DumpState();

private:
    struct PushRecord {
        std::chrono::steady_clock::time_point time{};
        s32 channel{};
        u32 lists{};
        bool suspended_before{};
        u64 before_address{};
        u32 before_value{};
        bool suspended_after{};
        u64 after_address{};
        u32 after_value{};
        size_t queued_after{};
    };
    std::mutex record_mutex;
    std::array<PushRecord, 128> records{};
    size_t record_count{};
    bool record_pushes{};

    ::Common::unordered_map<s32, std::shared_ptr<ChannelState>> channels;
    std::mutex scheduling_guard;
    std::vector<std::shared_ptr<ChannelState>> pending_native_channels;
};

} // namespace Control

} // namespace Tegra
