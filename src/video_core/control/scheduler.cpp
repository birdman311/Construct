// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "common/assert.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/gpu.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/control/channel_state.h"
#include "core/launch_log.h"

namespace Tegra::Control {

Scheduler::Scheduler() : record_pushes{std::getenv("EDEN_STALL_WATCHDOG") != nullptr} {
    if (record_pushes) {
        Core::LaunchLog::SetGpuDumpHook([this] { DumpState(); });
    }
}

Scheduler::~Scheduler() {
    if (record_pushes) {
        Core::LaunchLog::SetGpuDumpHook({});
    }
}

void Scheduler::Push(GPU& gpu, s32 channel, CommandList&& entries) {
    PushRecord record{};
    if (record_pushes) {
        record.time = std::chrono::steady_clock::now();
        record.channel = channel;
        record.lists = static_cast<u32>(entries.command_lists.size());
    }
    std::shared_ptr<ChannelState> channel_state;
    {
        std::unique_lock lk(scheduling_guard);
        auto it = channels.find(channel);
        ASSERT(it != channels.end());
        channel_state = it->second;
        gpu.BindChannel(channel_state->bind_id);
    }
    // Process commands outside the lock to reduce contention.
    // Multiple channels can prepare their commands in parallel.
    if (record_pushes) {
        if (const auto& acquire = channel_state->payload->dma_pusher.PendingNativeAcquire()) {
            record.suspended_before = true;
            record.before_address = acquire->address;
            record.before_value = acquire->value;
        }
    }
    channel_state->payload->dma_pusher.Push(std::move(entries));
    channel_state->payload->dma_pusher.DispatchCalls();
    if (record_pushes) {
        if (const auto& acquire = channel_state->payload->dma_pusher.PendingNativeAcquire()) {
            record.suspended_after = true;
            record.after_address = acquire->address;
            record.after_value = acquire->value;
        }
        record.queued_after = channel_state->payload->dma_pusher.QueuedCommandLists();
        std::scoped_lock lk{record_mutex};
        records[record_count++ % records.size()] = record;
    }
    if (channel_state->payload->dma_pusher.HasPendingNativeAcquire() &&
        std::find(pending_native_channels.begin(), pending_native_channels.end(), channel_state) == pending_native_channels.end()) {
        pending_native_channels.push_back(channel_state);
    }
    if (!pending_native_channels.empty()) ResumePending(gpu);
}

void Scheduler::ResumePending(GPU& gpu) {
    ResumeReadyNativeChannels(pending_native_channels,
        [](const auto& state) { return state->payload->dma_pusher.HasPendingNativeAcquire() &&
                                     state->payload->dma_pusher.CanResumeNativeAcquire(); },
        [&](const auto& state) {
            {
                std::scoped_lock lock{scheduling_guard};
                gpu.BindChannel(state->bind_id);
            }
            state->payload->dma_pusher.DispatchCalls();
        });
    std::erase_if(pending_native_channels,
        [](const auto& state) { return !state->payload->dma_pusher.HasPendingNativeAcquire(); });
}

void Scheduler::DumpState() {
    const auto now = std::chrono::steady_clock::now();
    Core::LaunchLog::Write("GPU channels (read while the GPU thread runs):");
    {
        std::scoped_lock lk{scheduling_guard};
        for (const auto& [id, state] : channels) {
            if (!state || !state->payload) {
                Core::LaunchLog::Write("  channel {}: no payload", id);
                continue;
            }
            state->payload->kepler_compute.NativeQmds().Describe(
                [id](u64 address, bool loaded, bool scheduled, bool completed, s32 dependencies,
                     s32 deferred, const std::string& releases) {
                    Core::LaunchLog::Write("  channel {} QMD {:#x}: loaded {} scheduled {} completed {} "
                                           "dependencies {} deferred {}{}",
                                           id, address, loaded, scheduled, completed, dependencies,
                                           deferred, releases);
                });
            auto& pusher = state->payload->dma_pusher;
            const auto& acquire = pusher.PendingNativeAcquire();
            const bool pending = std::find(pending_native_channels.begin(),
                                           pending_native_channels.end(),
                                           state) != pending_native_channels.end();
            if (acquire) {
                Core::LaunchLog::Write("  channel {}: SUSPENDED on semaphore {:#x} (op {}) for {:#x}, "
                                       "{} command list(s) queued, in pending list: {}",
                                       id, acquire->address, acquire->operation, acquire->value,
                                       pusher.QueuedCommandLists(), pending);
                pusher.DescribeQueued([](const std::string& line) { Core::LaunchLog::Write("    {}", line); }, 300);
            } else {
                Core::LaunchLog::Write("  channel {}: not suspended, {} command list(s) queued, "
                                       "in pending list: {}",
                                       id, pusher.QueuedCommandLists(), pending);
            }
        }
    }
    std::scoped_lock lk{record_mutex};
    const size_t count = std::min(record_count, records.size());
    Core::LaunchLog::Write("last {} GPU submissions (oldest first, {} in total):", count, record_count);
    for (size_t i = record_count - count; i < record_count; ++i) {
        const auto& r = records[i % records.size()];
        const double ago = std::chrono::duration<double>(now - r.time).count();
        const std::string before = r.suspended_before
            ? fmt::format("suspended on {:#x} for {:#x}", r.before_address, r.before_value)
            : std::string("free");
        const std::string after = r.suspended_after
            ? fmt::format("suspended on {:#x} for {:#x}", r.after_address, r.after_value)
            : std::string("free");
        Core::LaunchLog::Write("  {:8.3f} s ago: channel {} +{} list(s); before: {}; after: {}; "
                               "queued after: {}",
                               ago, r.channel, r.lists, before, after, r.queued_after);
    }
}

void Scheduler::DeclareChannel(std::shared_ptr<ChannelState> new_channel) {
    s32 channel = new_channel->bind_id;
    std::unique_lock lk(scheduling_guard);
    channels.emplace(channel, new_channel);
}

} // namespace Tegra::Control
