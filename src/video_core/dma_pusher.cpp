// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fmt/format.h>
#include "common/settings.h"
#include "core/core.h"
#include "video_core/dma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/macro.h"
#include "video_core/native_qmd.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 ComputeInline = 0x6D;

DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

void DmaPusher::DispatchCalls() {
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    system.GPU().FlushCommands();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::HasFailedNativeWork() const {
    const auto& context = channel_state.native_macro_context;
    if (!channel_state.native_abi || !context ||
        !(context->engine.native_execution_failed || context->compute_engine.native_execution_failed)) {
        return false;
    }
    // Construct: a failed native macro/QMD used to halt the channel for good - no later command
    // or fence increment ever ran, so the game waited forever (a permanent freeze with every
    // thread idle). The failed item has already been dropped; clear the flag and keep going.
    // A wrong or missing draw for one frame is far better than a hung game.
    static u32 recovered{};
    if (recovered++ < 16) {
        LOG_ERROR(HW_GPU, "Native GPU work failed; dropped it and continuing the channel ({} so far)",
                  recovered);
    }
    context->engine.native_execution_failed = false;
    context->compute_engine.native_execution_failed = false;
    return false;
}

bool DmaPusher::IsNvn2() const {
    const auto& context = channel_state.native_macro_context;
    return channel_state.native_abi && context && context->nvn2;
}

namespace {
template <typename Q>
typename Q::container_type& MutableQueueContainer(Q& queue) {
    struct Access : Q {
        static typename Q::container_type& Get(Q& q) { return q.*(&Access::c); }
    };
    return Access::Get(queue);
}
} // namespace

void DmaPusher::RefreshSnapshot(CommandList& list) {
    for (size_t i = 0; i < list.command_lists.size() && i < list.submitted_offsets.size(); ++i) {
        const CommandListHeader& entry = list.command_lists[i];
        const u32 offset = list.submitted_offsets[i];
        if (entry.size == 0 || offset + entry.size > list.submitted_words.size() ||
            !memory_manager.IsFullyMappedRange(entry.addr, entry.size * sizeof(u32))) {
            continue;
        }
        memory_manager.ReadBlockUnsafe(entry.addr, list.submitted_words.data() + offset,
                                       entry.size * sizeof(u32));
    }
}

bool DmaPusher::Step() {
    if (HasFailedNativeWork()) return false;
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }

    const bool was_suspended = puller.HasPendingNativeAcquire();
    if (!puller.ResumeNativeAcquire(*this)) return false;
    // NVN2: lists queued while the channel waited hold commands the game writes before it
    // releases the acquire, so the words at submit time could be older; the acquire is
    // released now, so the commands are written and the memory not yet reused (from Lichii).
    if (was_suspended && IsNvn2()) {
        for (CommandList& list : MutableQueueContainer(dma_pushbuffer)) {
            if (!list.submitted_offsets.empty()) RefreshSnapshot(list);
        }
    }

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        dma_pushbuffer.pop();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        if (puller.HasPendingNativeAcquire() || HasFailedNativeWork()) return false;
        command_resume_offset = 0;
        dma_pushbuffer.pop();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (signal_sync && !synced) {
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced; });
        signal_sync = false;
        synced = false;
    }

    if (header.size > 0) {
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            if (kepler_payload || macro_payload) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const engine_state = subchannels[dma_state.subchannel];
                const bool gpu_dirty =
                    memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                // Switch 2 (NVN2) passes indirect draw arguments the GPU computes as a macro's
                // parameters in a GP entry of their own. Those words are executed from the submit
                // snapshot, taken before the GPU wrote them. IsMemoryDirty only says whether the
                // host GPU holds newer bytes *now*: it is false again once anything downloaded
                // the page (a download clears the whole 4 KiB page, e.g. the previous macro's
                // own refresh or a reactive CPU read) or when a GPU-thread engine wrote guest
                // memory directly (inline-to-memory, DMA copy/clear, semaphore/QMD releases).
                // Memory is then newer than the snapshot but reads as clean, and the macro ran
                // with the snapshot's junk (Hyrule Warriors: Quads count 0x3F800000, macro 0x1d2
                // instance loop over the step limit). Re-read such words whatever the cache says.
                // Only for an entry that holds nothing but this method's data (the NVN2 4/5-word
                // argument entry): a macro call that merely straddles two command-memory chunks
                // must keep the snapshot, as that memory may already hold a later frame.
                const bool argument_entry = macro_payload && channel_state.native_abi &&
                                            header.size <= dma_state.method_count &&
                                            header.size <= 64;
                engine_state->current_dirty = gpu_dirty || argument_entry;
                engine_state->current_gpu_dirty = gpu_dirty;
            }
        }
        const bool use_safe = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe();
        // The words as submitted (Switch 2), unless this entry is being resumed or the GPU has
        // newer words in memory that the safe read would flush.
        const bool submitted = command_resume_offset == 0 &&
                               dma_pushbuffer_subindex < command_list.submitted_offsets.size() &&
                               !(use_safe && memory_manager.IsMemoryDirty(dma_state.dma_get,
                                                                          header.size * sizeof(u32)));
        if (submitted) {
            const std::span<const CommandHeader> commands{
                command_list.submitted_words.data() +
                    command_list.submitted_offsets[dma_pushbuffer_subindex],
                header.size};
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (use_safe) {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        }
    }

    if (puller.HasPendingNativeAcquire() || HasFailedNativeWork()) return false;
    command_resume_offset = 0;
    if (++dma_pushbuffer_subindex >= command_list_size) {
        dma_pushbuffer.pop();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        rasterizer->SignalFence([this]() {
            std::scoped_lock lk(sync_mutex);
            synced = true;
            sync_cv.notify_all();
        });
    }

    return true;
}

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset) {
    for (size_t index = 0; index < commands.size();) {
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (channel_state.native_abi && dma_state.method < non_puller_methods) max_write = 1;
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            CallMethod(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            // No command active - this is the first word of a new one
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingNativeAcquire() || HasFailedNativeWork()) return index;
    }
    return commands.size();
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}

void DmaPusher::CallMethod(u32 argument) {
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

namespace {
template <typename Q>
const typename Q::container_type& QueueContainer(const Q& queue) {
    struct Access : Q {
        static const typename Q::container_type& Get(const Q& q) { return q.*(&Access::c); }
    };
    return Access::Get(queue);
}
} // namespace

void DmaPusher::DescribeQueued(const std::function<void(const std::string&)>& out, size_t max_lines) {
    size_t lines = 0;
    const auto emit = [&](const std::string& text) {
        if (lines++ < max_lines) out(text);
    };
    u32 method = dma_state.method, subchannel = dma_state.subchannel, count = dma_state.method_count;
    bool non_incrementing = dma_state.non_incrementing, increment_once = dma_increment_once;
    std::array<u64, max_subchannels> qmd_address{};
    emit(fmt::format("queued: {} command list(s); front entry {} from word {}; decoder method {:#x} sc {} count {}",
                     dma_pushbuffer.size(), dma_pushbuffer_subindex, command_resume_offset, method, subchannel, count));
    const auto call = [&](u32 sc, u32 m, u32 value) {
        const bool compute = sc < max_subchannels && subchannel_type[sc] == Engines::EngineTypes::KeplerCompute;
        const bool maxwell = sc < max_subchannels && subchannel_type[sc] == Engines::EngineTypes::Maxwell3D;
        if (m < non_puller_methods) {
            emit(fmt::format("  host m={:#x} v={:#x}", m, value));
        } else if (compute && m == 0xad) {
            qmd_address[sc] = u64{value} << 8;
        } else if (compute && (m == 0xae || m == 0xaf || m == 0xb0)) {
            const u64 address = qmd_address[sc];
            std::array<u32, 64> qmd{};
            const bool mapped = address && memory_manager.IsFullyMappedRange(address, sizeof(qmd));
            if (mapped) memory_manager.ReadBlockUnsafe(address, qmd.data(), sizeof(qmd));
            std::string releases;
            for (u32 slot = 0; mapped && slot < 2; ++slot) {
                const u32 flags = qmd[25 + slot * 4];
                if (!(flags & (1U << 23))) continue;
                releases += fmt::format(" release{} {:#x} <- {:#x} (flags {:#x})", slot,
                                        (u64{flags & 0xff} << 32) | qmd[24 + slot * 4], qmd[26 + slot * 4], flags);
            }
            emit(fmt::format("  compute PCAS m={:#x} action {} QMD {:#x}: deps {} dependent {}{}", m,
                             value & 15, address, mapped ? (qmd[4] >> 14) & 0xffff : 0,
                             mapped && (qmd[16] & 1) ? fmt::format("{:#x} action {}", u64{qmd[15]} << 8, (qmd[16] >> 1) & 7) : std::string("none"),
                             releases));
        } else if (maxwell && m >= 0x6c0 && m <= 0x6c3) {
            emit(fmt::format("  3D report semaphore m={:#x} v={:#x}", m, value));
        }
    };
    const auto& lists = QueueContainer(dma_pushbuffer);
    for (size_t list = 0; list < lists.size(); ++list) {
        const auto& command_list = lists[list];
        if (!command_list.prefetch_command_list.empty()) {
            emit(fmt::format(" list {}: {} prefetched words", list, command_list.prefetch_command_list.size()));
        }
        for (size_t entry = list == 0 ? dma_pushbuffer_subindex : 0; entry < command_list.command_lists.size(); ++entry) {
            const auto& header = command_list.command_lists[entry];
            std::vector<u32> words(static_cast<size_t>(header.size));
            const bool mapped = !words.empty() && memory_manager.IsFullyMappedRange(header.addr, words.size() * sizeof(u32));
            if (mapped) memory_manager.ReadBlockUnsafe(header.addr, words.data(), words.size() * sizeof(u32));
            size_t changed = 0;
            const bool has_snapshot = entry < command_list.submitted_offsets.size();
            if (has_snapshot) {
                for (size_t i = 0; i < words.size(); ++i) {
                    if (words[i] != command_list.submitted_words[command_list.submitted_offsets[entry] + i].argument) ++changed;
                }
            }
            const bool front = list == 0 && entry == dma_pushbuffer_subindex;
            emit(fmt::format(" list {} entry {}: {:#x} {} words{}{}", list, entry, static_cast<u64>(header.addr), words.size(),
                             has_snapshot ? fmt::format(", {} word(s) differ from the submitted copy", changed) : std::string(", no submitted copy"),
                             front ? " (resumes here, read from memory)" : ""));
            // Decode what will execute: memory for the resumed entry, the snapshot otherwise.
            if (!front && has_snapshot) {
                for (size_t i = 0; i < words.size(); ++i) words[i] = command_list.submitted_words[command_list.submitted_offsets[entry] + i].argument;
            }
            for (size_t i = front ? command_resume_offset : 0; i < words.size(); ++i) {
                const CommandHeader command{words[i]};
                if (count) {
                    call(subchannel, method, words[i]);
                    if (!non_incrementing) ++method;
                    non_incrementing |= increment_once;
                    --count;
                    continue;
                }
                switch (command.mode) {
                case SubmissionMode::Increasing:
                case SubmissionMode::NonIncreasing:
                case SubmissionMode::IncreaseOnce:
                    method = command.method;
                    subchannel = command.subchannel;
                    count = command.method_count;
                    non_incrementing = command.mode == SubmissionMode::NonIncreasing;
                    increment_once = command.mode == SubmissionMode::IncreaseOnce;
                    break;
                case SubmissionMode::Inline:
                    call(command.subchannel, command.method, command.arg_count);
                    break;
                default:
                    break;
                }
            }
        }
    }
    if (lines > max_lines) out(fmt::format("  ... {} more line(s)", lines - max_lines));
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

} // namespace Tegra
