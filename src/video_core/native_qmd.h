// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace Tegra {

// Native QMD 3.0 dependency scheduling; nonempty grids require a dispatch backend.
class NativeQmdScheduler {
public:
    NativeQmdScheduler() = default;
    /// nvn2: newer SDKs (Ocarina of Time). Their channels share one table, NVN2 patches a third
    /// release slot at word 52 for queue syncs, and a QMD whose program can't run still
    /// completes (other queues and the CPU wait on its releases). From Lichii's scheduler.
    explicit NativeQmdScheduler(bool nvn2_) : nvn2{nvn2_} {}
    [[nodiscard]] bool Nvn2() const { return nvn2; }
    void SetNvn2(bool value) { nvn2 = value; }
    using Word = std::uint32_t;
    using Address = std::uint64_t;
    using Descriptor = std::array<Word, 64>;
    struct RejectShaderDispatch { bool operator()(const Descriptor&) const { return false; } };
    template <typename Read, typename Release>
    std::string_view Signal(Address address, Word argument, Read read, Release release) {
        return Signal(address, argument, read, release, RejectShaderDispatch{}, nullptr);
    }
    template <typename Read, typename Release, typename Dispatch>
    std::string_view Signal(Address address, Word argument, Read read, Release release, Dispatch dispatch) {
        return Signal(address, argument, read, release, dispatch, nullptr);
    }
    // `self` is the engine issuing this PCAS action. A scheduler shared by the channels of a
    // TSG (newer SDKs: one QMD dependency table per TSG, as on the GPU) runs a QMD that another
    // channel made ready on the engine that scheduled it; Dispatch then takes (owner, QMD).
    template <typename Read, typename Release, typename Dispatch>
    std::string_view Signal(Address address, Word argument, Read read, Release release, Dispatch dispatch, void* self) {
        std::scoped_lock lock{mutex};
        std::string_view error;
        struct Pending { Address address; Word action; Word count; };
        std::array<Pending, 256> pending{};
        std::size_t count = 1;
        pending[0] = {address, argument & 15, ((argument >> 14) & 0x3ff) + 1};
        for (std::size_t i = 0; i < count; ++i) {
            const auto request = pending[i];
            if (states.size() >= 4096 && !states.contains(request.address)) {
                if (nvn2) {
                    // Finished QMDs with nothing deferred are only history.
                    std::erase_if(states, [](const auto& entry) {
                        return entry.second.completed && entry.second.deferred == 0;
                    });
                }
                if (states.size() >= 4096) return "QMD cache limit";
            }
            auto& state = states[request.address];
            if (request.action == 3 || request.action == 2) {
                if (request.action == 3 || !state.loaded) {
                    if (!read(request.address, state.descriptor)) return "unmapped QMD";
                    const auto& qmd = state.descriptor;
                    if ((qmd[18] & 0xff) != 0x30 || (qmd[4] & 0x100)) return "unsupported QMD version or queue";
                    if constexpr (std::is_same_v<Dispatch, RejectShaderDispatch>) {
                        if (qmd[12] && (qmd[13] & 0xffff) && (qmd[14] & 0xffff))
                            return "native shader dispatch required";
                    }
                    state.dependencies = std::int32_t((qmd[4] >> 14) & 0xffff) + state.deferred;
                    state.deferred = 0;
                    state.loaded = true;
                    state.completed = false;
                }
                state.scheduled = true;
                state.owner = self;
            } else if (request.action == 7 || request.action == 12) {
                const auto delta = request.action == 7 ? -std::int32_t(request.count) : std::int32_t(request.count);
                if (!state.loaded || state.completed) { state.deferred += delta; continue; }
                state.dependencies += delta;
            } else if (request.action == 0 || request.action == 1) {
                continue;
            } else {
                return "unsupported PCAS action";
            }
            if (!state.loaded || !state.scheduled || state.completed || state.dependencies > 0) continue;
            if (state.dependencies < 0) return "QMD dependency underflow";
            // The SDK patches release payloads while a scheduled QMD is
            // waiting for its remaining dependencies. Fetch them at execution.
            if (!read(request.address, state.descriptor)) return "unmapped QMD";
            const auto qmd = state.descriptor;
            if ((qmd[18] & 0xff) != 0x30 || (qmd[4] & 0x100)) return "unsupported QMD version or queue";
            // Validate the release slots before making any visible.
            for (Word slot = 0; slot < (nvn2 ? 3U : 2U); ++slot) {
                const Word flags = qmd[ReleaseSlot(slot) + 1];
                if (!(flags & (1U << 23))) continue;
                const bool add = (flags & (1U << 26)) != 0;
                if ((flags & (0x3bU << 24)) || (flags >> 30) != 1 ||
                    (add && (flags & (7U << 20)))) return "unsupported QMD release format";
                if (add) {
                    if constexpr (!std::is_invocable_r_v<bool, Release, Address, Word, Word>)
                        return "unsupported QMD reduction callback";
                }
            }
            if (qmd[12] && (qmd[13] & 0xffff) && (qmd[14] & 0xffff)) {
                bool dispatched{};
                if constexpr (std::is_invocable_r_v<bool, Dispatch, void*, const Descriptor&>) {
                    dispatched = dispatch(state.owner ? state.owner : self, qmd);
                } else {
                    dispatched = dispatch(qmd);
                }
                if (!dispatched) {
                    if (!nvn2) return "native shader dispatch required";
                    error = "native shader dispatch required";
                }
            }
            state.completed = true;
            for (Word slot = 0; slot < (nvn2 ? 3U : 2U); ++slot) {
                const Word base = ReleaseSlot(slot);
                const Word flags = qmd[base + 1];
                if (!(flags & (1U << 23))) continue;
                const Address destination = (Address{flags & 0xff} << 32) | qmd[base];
                bool released{};
                if constexpr (std::is_invocable_r_v<bool, Release, Address, Word, Word>) {
                    released = release(destination, qmd[base + 2], flags);
                } else {
                    released = release(destination, qmd[base + 2]);
                }
                if (!released) return "unmapped QMD release";
            }
            if (qmd[16] & 1) {
                const Word action = (qmd[16] >> 1) & 7;
                if (action != 4 && action != 3 && action != 1) return "unsupported dependent QMD action";
                if (count == pending.size()) return "QMD chain limit";
                pending[count++] = {Address{qmd[15]} << 8, action == 4 ? 7U : action == 3 ? 3U : 2U, 1};
            }
            if (qmd[16] & (1U << 5)) return "second dependent QMD unsupported";
        }
        return error;
    }
    /// The engine is going away: QMDs it scheduled run on whichever engine readies them.
    void ForgetOwner(void* owner) {
        std::scoped_lock lock{mutex};
        for (auto& [address, state] : states) {
            if (state.owner == owner) state.owner = nullptr;
        }
    }

    /// Stall watchdog: QMDs that are loaded but not completed, and dependency changes recorded
    /// for QMDs this scheduler never loaded (deferred).
    template <typename Out>
    void Describe(Out&& out) const {
        for (const auto& [address, state] : states) {
            if ((state.loaded && !state.completed) || state.deferred != 0) {
                std::string releases;
                for (Word slot = 0; slot < (nvn2 ? 3U : 2U); ++slot) {
                    const Word flags = state.descriptor[ReleaseSlot(slot) + 1];
                    releases += " release" + std::to_string(slot) + (flags & (1U << 23) ? "=on" : "=off") +
                                " dst " + std::to_string((Address{flags & 0xff} << 32) | state.descriptor[ReleaseSlot(slot)]) +
                                " payload " + std::to_string(state.descriptor[ReleaseSlot(slot) + 2]);
                }
                out(address, state.loaded, state.scheduled, state.completed, state.dependencies,
                    state.deferred, releases);
            }
        }
    }

private:
    // Release slots: (address low, flags | address high, payload) at words 24 and 28, and for
    // NVN2 a third at word 52.
    static constexpr Word ReleaseSlot(Word slot) { return slot == 2 ? 52U : 24U + slot * 4; }
    bool nvn2{};

    struct State {
        Descriptor descriptor{};
        std::int32_t dependencies{};
        std::int32_t deferred{};
        bool loaded{};
        bool scheduled{};
        bool completed{};
        void* owner{};
    };
    std::unordered_map<Address, State> states;
    std::mutex mutex;
};
} // namespace Tegra
