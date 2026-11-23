// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/fs/path_util.h"

// Construct launch log: one plain-text file per launch (log/launch_<program id>.log) holding the
// details the main log hides: every service call with its result, failing and memory syscalls,
// missing services, the game's own log text, and named backtraces at exit/break.
namespace Core::LaunchLog {

namespace detail {

struct IpcStats {
    u64 calls{};
    u64 failures{};
    u32 last_failure{};
};

struct State {
    std::mutex mutex;
    std::FILE* file{};
    std::atomic<bool> active{false};
    std::chrono::steady_clock::time_point start{};
    std::map<std::string, IpcStats> ipc;
    std::map<std::string, u64> unimplemented;
    std::map<std::pair<u32, u32>, u64> svc_results;
    std::map<std::pair<u64, u64>, u64> info_queries;
    std::map<u32, u64> svc_detail_counts;
    std::vector<std::string> missing_services;
    std::filesystem::path snapshot_path;
    std::chrono::steady_clock::time_point last_snapshot{};
    u64 last_snapshot_frames{};
    // Stall watchdog (EDEN_STALL_WATCHDOG=1): when frames stop, the next syscall of any guest
    // thread writes a thread dump (launch_log_backtrace.h DumpThreads).
    std::atomic<bool> dump_requested{false};
    // First 0x2000-byte GPU-accessible block (svc 0x80): NVN's per-queue semaphores.
    std::atomic<u64> queue_semaphore_block{0};
    u64 dump_at_frame{}; // EDEN_DUMP_AT_FRAME: one dump when this frame is presented
    std::mutex hook_mutex;
    std::function<void()> gpu_dump_hook; // the GPU scheduler's state dump
    std::atomic<bool> watchdog_stop{false};
    std::thread watchdog;

    ~State() {
        watchdog_stop.store(true);
        if (watchdog.joinable()) {
            watchdog.join();
        }
    }
};

inline State& Get() {
    static State state;
    return state;
}

inline void WriteLocked(State& s, std::string_view text) {
    if (!s.file) {
        return;
    }
    const double t =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - s.start).count();
    fmt::print(s.file, "[{:10.4f}] {}\n", t, text);
    std::fflush(s.file);
}

inline std::string ResultText(u32 r) {
    if (r == 0) {
        return "ok";
    }
    return fmt::format("FAILED {:#x} ({:04}-{:04})", r, 2000 + (r & 0x1ff), (r >> 9) & 0x1fff);
}

// Timeouts, cancellations, termination requests and "value already changed" (InvalidState from
// WaitForAddress/ResetSignal) are normal traffic, so they are only counted.
constexpr bool IsQuietResult(u32 r) {
    return r == 0xEA01 || r == 0xEC01 || r == 0x7601 || r == 0xFA01;
}

constexpr bool ReturnsResult(u32 id) {
    switch (id) {
    case 0x07: case 0x0A: case 0x0B: case 0x10: case 0x1D: case 0x1E: case 0x28:
    case 0x2A: case 0x36: case 0x3C: case 0x3D: case 0x7F:
        return false;
    default:
        return true;
    }
}

constexpr std::string_view SvcName(u32 id) {
    switch (id) {
    case 0x01: return "SetHeapSize";
    case 0x02: return "SetMemoryPermission";
    case 0x03: return "SetMemoryAttribute";
    case 0x04: return "MapMemory";
    case 0x05: return "UnmapMemory";
    case 0x06: return "QueryMemory";
    case 0x07: return "ExitProcess";
    case 0x08: return "CreateThread";
    case 0x09: return "StartThread";
    case 0x0A: return "ExitThread";
    case 0x0B: return "SleepThread";
    case 0x0C: return "GetThreadPriority";
    case 0x0D: return "SetThreadPriority";
    case 0x0E: return "GetThreadCoreMask";
    case 0x0F: return "SetThreadCoreMask";
    case 0x10: return "GetCurrentProcessorNumber";
    case 0x11: return "SignalEvent";
    case 0x12: return "ClearEvent";
    case 0x13: return "MapSharedMemory";
    case 0x14: return "UnmapSharedMemory";
    case 0x15: return "CreateTransferMemory";
    case 0x16: return "CloseHandle";
    case 0x17: return "ResetSignal";
    case 0x18: return "WaitSynchronization";
    case 0x19: return "CancelSynchronization";
    case 0x1A: return "ArbitrateLock";
    case 0x1B: return "ArbitrateUnlock";
    case 0x1C: return "WaitProcessWideKeyAtomic";
    case 0x1D: return "SignalProcessWideKey";
    case 0x1E: return "GetSystemTick";
    case 0x1F: return "ConnectToNamedPort";
    case 0x20: return "SendSyncRequestLight";
    case 0x21: return "SendSyncRequest";
    case 0x22: return "SendSyncRequestWithUserBuffer";
    case 0x23: return "SendAsyncRequestWithUserBuffer";
    case 0x24: return "GetProcessId";
    case 0x25: return "GetThreadId";
    case 0x26: return "Break";
    case 0x27: return "OutputDebugString";
    case 0x28: return "ReturnFromException";
    case 0x29: return "GetInfo";
    case 0x2C: return "MapPhysicalMemory";
    case 0x2D: return "UnmapPhysicalMemory";
    case 0x32: return "SetThreadActivity";
    case 0x33: return "GetThreadContext3";
    case 0x34: return "WaitForAddress";
    case 0x35: return "SignalToAddress";
    case 0x36: return "SynchronizePreemptionState";
    case 0x80: return "SetGpuMemoryAttribute";
    default: return {};
    }
}

constexpr std::string_view InfoName(u64 type) {
    switch (type) {
    case 0: return "CoreMask";
    case 1: return "PriorityMask";
    case 2: return "AliasRegionAddress";
    case 3: return "AliasRegionSize";
    case 4: return "HeapRegionAddress";
    case 5: return "HeapRegionSize";
    case 6: return "TotalMemorySize";
    case 7: return "UsedMemorySize";
    case 8: return "DebuggerAttached";
    case 9: return "ResourceLimit";
    case 10: return "IdleTickCount";
    case 11: return "RandomEntropy";
    case 12: return "AslrRegionAddress";
    case 13: return "AslrRegionSize";
    case 14: return "StackRegionAddress";
    case 15: return "StackRegionSize";
    case 16: return "SystemResourceSizeTotal";
    case 17: return "SystemResourceSizeUsed";
    case 18: return "ProgramId";
    case 20: return "UserExceptionContextAddress";
    case 21: return "TotalNonSystemMemorySize";
    case 22: return "UsedNonSystemMemorySize";
    case 23: return "IsApplication";
    case 24: return "FreeThreadCount";
    case 25: return "ThreadTickCount";
    case 26: return "IsSvcPermitted";
    case 27: return "IoRegionHint";
    case 28: return "AliasRegionExtraSize";
    case 0x1f: case 0x20: case 0x21: return "Switch2Extended";
    default: return "?";
    }
}

inline u64 FramesPresented(const State& s) {
    u64 frames = 0;
    for (const auto& [key, st] : s.ipc) {
        if (key.find("PresentClientBuffer") != std::string::npos) {
            frames += st.calls;
        }
    }
    return frames;
}

inline void WriteSummaryTables(const State& s, std::FILE* f) {
    std::vector<std::pair<std::string, IpcStats>> ipc(s.ipc.begin(), s.ipc.end());
    std::sort(ipc.begin(), ipc.end(),
              [](const auto& a, const auto& b) { return a.second.calls > b.second.calls; });
    fmt::print(f, "Service commands used: {}\n", ipc.size());
    for (const auto& [key, st] : ipc) {
        fmt::print(f, "  {:>8} calls  {:>6} failed{}  {}\n", st.calls, st.failures,
                   st.failures ? fmt::format(" (last {:#x})", st.last_failure) : "", key);
    }
    fmt::print(f, "Unimplemented commands: {}\n", s.unimplemented.size());
    for (const auto& [key, n] : s.unimplemented) {
        fmt::print(f, "  {:>8}x  {}\n", n, key);
    }
    fmt::print(f, "Missing services requested: {}\n", s.missing_services.size());
    for (const auto& name : s.missing_services) {
        fmt::print(f, "  {}\n", name);
    }
    fmt::print(f, "Failing syscall results: {}\n", s.svc_results.size());
    for (const auto& [key, n] : s.svc_results) {
        const auto name = SvcName(key.first);
        fmt::print(f, "  {:>8}x  {} -> {}\n", n,
                   name.empty() ? fmt::format("svc {:#x}", key.first) : std::string(name),
                   ResultText(key.second));
    }
    std::fflush(f);
}

// Rewrites launch_<id>_summary.txt every 15 s so the summary survives a kill or crash.
inline void MaybeWriteSnapshot(State& s) {
    const auto now = std::chrono::steady_clock::now();
    if (s.snapshot_path.empty() || now - s.last_snapshot < std::chrono::seconds(15)) {
        return;
    }
    const double elapsed = std::chrono::duration<double>(now - s.start).count();
    const double window = std::chrono::duration<double>(now - s.last_snapshot).count();
    const u64 frames = FramesPresented(s);
    const double fps =
        window > 0 ? static_cast<double>(frames - s.last_snapshot_frames) / window : 0.0;
    s.last_snapshot = now;
    s.last_snapshot_frames = frames;
#ifdef _WIN32
    std::FILE* f = _wfopen(s.snapshot_path.c_str(), L"w");
#else
    std::FILE* f = std::fopen(s.snapshot_path.c_str(), "w");
#endif
    if (!f) {
        return;
    }
    fmt::print(f, "Construct launch summary at {:.1f} s\n", elapsed);
    fmt::print(f, "Frames presented: {} total, {:.1f} FPS over the last {:.0f} s\n\n", frames,
               fps, window);
    WriteSummaryTables(s, f);
    std::fclose(f);
}

// Polls the presented-frame count; once frames stopped for 4 s, asks for a thread dump (at most
// three, 15 s apart, so a second dump shows whether the threads moved).
inline void WatchdogLoop() {
    auto& s = Get();
    u64 last_frames = 0;
    auto last_change = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_dump{};
    int dumps = 0;
    while (!s.watchdog_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds{500});
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lk{s.mutex};
        if (!s.file) {
            continue;
        }
        const u64 frames = FramesPresented(s);
        if (frames != last_frames) {
            last_frames = frames;
            last_change = now;
            dumps = 0; // frames came back: a later stall gets its own dumps
            continue;
        }
        // Ignore the boot (shader prebuild / first loads pause presentation for seconds).
        if (frames < 600 || now - last_change < std::chrono::seconds{4} || dumps >= 3 ||
            (dumps > 0 && now - last_dump < std::chrono::seconds{15})) {
            continue;
        }
        ++dumps;
        last_dump = now;
        WriteLocked(s, fmt::format("!! STALL WATCHDOG: no frame presented for {:.1f} s (stuck after "
                                   "frame {}); thread dump #{} requested",
                                   std::chrono::duration<double>(now - last_change).count(),
                                   frames, dumps));
        s.dump_requested.store(true);
    }
}

inline void StopWatchdog(State& s) {
    s.watchdog_stop.store(true);
    if (s.watchdog.joinable()) {
        s.watchdog.join();
    }
    s.watchdog_stop.store(false);
    s.dump_requested.store(false);
}

} // namespace detail

inline bool Active() {
    return detail::Get().active.load(std::memory_order_relaxed);
}

// Movie diagnostics (EDEN_MOVIE_TRACE=1, set by play_movie_test.bat): off in normal play.
inline bool MovieTraceEnabled() {
    static const bool enabled = std::getenv("EDEN_MOVIE_TRACE") != nullptr;
    return enabled;
}

// True while a movie decoder session is started and movie tracing is on.
inline std::atomic<bool> g_movie_timeline{false};

template <typename... Args>
inline void Write(fmt::format_string<Args...> format, Args&&... args);

// Test build: during a movie, log every service call except the per-frame ones.
inline void MovieIpc(std::string_view service, std::string_view function) {
    if (!g_movie_timeline.load(std::memory_order_relaxed)) {
        return;
    }
    constexpr std::string_view skip[] = {"Ioctl", "AudioOut", "hid", "PresentClientBuffer",
                                         "SetGraphicsFrameLengthMode", "IVideoDecoderSession",
                                         "ISyncpointSupervisor", "GetDecoderOutputImageDescriptor",
                                         "IAudioRenderer", "RequestUpdate", "hwopus", "IHardwareOpusDecoder"};
    for (const auto word : skip) {
        if (service.find(word) != std::string_view::npos || function.find(word) != std::string_view::npos) {
            return;
        }
    }
    // Storage reads come in thousands: count them and log the count every 250 ms.
    if (function == "Read" && (service.find("IStorage") != std::string_view::npos ||
                               service.find("IFile") != std::string_view::npos)) {
        static std::atomic<u32> reads{0};
        static std::atomic<s64> window_start{0};
        const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
        reads.fetch_add(1);
        s64 start = window_start.load();
        if (start == 0) {
            window_start.compare_exchange_strong(start, now);
        } else if (now - start >= 250 && window_start.compare_exchange_strong(start, now)) {
            Write("TL storage reads {} in {} ms", reads.exchange(0), now - start);
        }
        return;
    }
    static std::atomic<u32> written{0};
    if (written.fetch_add(1) < 20000) {
        Write("TL ipc {}::{}", service, function);
    }
}

template <typename... Args>
inline void Write(fmt::format_string<Args...> format, Args&&... args) {
    auto& s = detail::Get();
    if (!s.active.load(std::memory_order_relaxed)) {
        return;
    }
    const std::string text = fmt::format(format, std::forward<Args>(args)...);
    std::scoped_lock lk{s.mutex};
    detail::WriteLocked(s, text);
}

inline void Open(u64 program_id) {
    auto& s = detail::Get();
    detail::StopWatchdog(s);
    std::scoped_lock lk{s.mutex};
    if (s.file) {
        std::fclose(s.file);
        s.file = nullptr;
    }
    s.ipc.clear();
    s.unimplemented.clear();
    s.svc_results.clear();
    s.info_queries.clear();
    s.svc_detail_counts.clear();
    s.missing_services.clear();
    const auto& dir = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / fmt::format("launch_{:016X}.log", program_id);
    s.snapshot_path = dir / fmt::format("launch_{:016X}_summary.txt", program_id);
#ifdef _WIN32
    s.file = _wfopen(path.c_str(), L"w");
#else
    s.file = std::fopen(path.c_str(), "w");
#endif
    s.start = std::chrono::steady_clock::now();
    s.last_snapshot = s.start;
    s.last_snapshot_frames = 0;
    s.active.store(s.file != nullptr);
    s.queue_semaphore_block.store(0);
    s.dump_at_frame = 0;
    if (const char* frame = std::getenv("EDEN_DUMP_AT_FRAME")) {
        s.dump_at_frame = std::strtoull(frame, nullptr, 10);
    }
    if (s.file && std::getenv("EDEN_STALL_WATCHDOG") != nullptr) {
        detail::WriteLocked(s, "stall watchdog on (EDEN_STALL_WATCHDOG)");
        s.watchdog = std::thread{detail::WatchdogLoop};
    }
}

inline void SetGpuDumpHook(std::function<void()> hook) {
    auto& s = detail::Get();
    std::scoped_lock lk{s.hook_mutex};
    s.gpu_dump_hook = std::move(hook);
}

inline u64 QueueSemaphoreBlock() {
    return detail::Get().queue_semaphore_block.load();
}

inline void RunGpuDumpHook() {
    auto& s = detail::Get();
    std::scoped_lock lk{s.hook_mutex};
    if (s.gpu_dump_hook) {
        s.gpu_dump_hook();
    }
}

// True once per watchdog request; the caller (a guest syscall) then writes the thread dump.
inline bool TakeDumpRequest() {
    auto& s = detail::Get();
    return s.dump_requested.load(std::memory_order_relaxed) && s.dump_requested.exchange(false);
}

inline void Close() {
    auto& s = detail::Get();
    detail::StopWatchdog(s);
    std::scoped_lock lk{s.mutex};
    if (!s.file) {
        return;
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - s.start).count();
    detail::WriteLocked(s, "==================== Summary ====================");
    fmt::print(s.file, "Frames presented: {} in {:.1f} s ({:.1f} FPS average)\n",
               detail::FramesPresented(s), elapsed,
               elapsed > 0 ? static_cast<double>(detail::FramesPresented(s)) / elapsed : 0.0);
    detail::WriteSummaryTables(s, s.file);
    std::fclose(s.file);
    s.file = nullptr;
    s.active.store(false);
}

inline void RecordIpc(std::string_view port, std::string_view name, u32 command,
                      std::optional<u32> result) {
    auto& s = detail::Get();
    if (!s.active.load(std::memory_order_relaxed)) {
        return;
    }
    std::string key = fmt::format("{}::{} ({})", port, name, command);
    std::scoped_lock lk{s.mutex};
    auto& st = s.ipc[key];
    ++st.calls;
    if (s.dump_at_frame != 0 && st.calls == s.dump_at_frame &&
        key.find("PresentClientBuffer") != std::string::npos) {
        detail::WriteLocked(s, fmt::format("!! frame {} presented: thread dump requested "
                                           "(EDEN_DUMP_AT_FRAME)", st.calls));
        s.dump_requested.store(true);
    }
    const bool failed = result && *result != 0;
    if (failed) {
        ++st.failures;
        st.last_failure = *result;
    }
    if (st.calls <= 2 || (failed && st.failures <= 8)) {
        detail::WriteLocked(s, fmt::format("ipc  {} -> {}", key,
                                           result ? detail::ResultText(*result)
                                                  : std::string("no reply yet (deferred)")));
    }
    detail::MaybeWriteSnapshot(s);
}

inline void RecordUnimplemented(std::string_view port, std::string_view name, u32 command,
                                bool auto_stub) {
    auto& s = detail::Get();
    if (!s.active.load(std::memory_order_relaxed)) {
        return;
    }
    std::string key = fmt::format("{}::{} ({})", port, name, command);
    std::scoped_lock lk{s.mutex};
    if (++s.unimplemented[key] <= 3) {
        detail::WriteLocked(s, fmt::format("!! UNIMPLEMENTED {} -> {}", key,
                                           auto_stub ? "auto-stubbed as success"
                                                     : "no reply sent (game reads garbage)"));
    }
}

inline void RecordMissingService(std::string_view name) {
    auto& s = detail::Get();
    if (!s.active.load(std::memory_order_relaxed)) {
        return;
    }
    std::scoped_lock lk{s.mutex};
    if (std::find(s.missing_services.begin(), s.missing_services.end(), name) !=
        s.missing_services.end()) {
        return;
    }
    s.missing_services.emplace_back(name);
    detail::WriteLocked(s, fmt::format("!! MISSING SERVICE '{}': Construct does not provide it, so "
                                       "the requesting game thread now waits forever",
                                       name));
}

inline void RecordSvc(u32 id, const std::array<u64, 8>& in, const std::array<u64, 8>& out) {
    auto& s = detail::Get();
    const u32 result = static_cast<u32>(out[0]);
    const bool detailed =
        id == 0x01 || id == 0x08 || id == 0x29 || id == 0x2C || id == 0x2D || id == 0x80;
    if (!detailed && (result == 0 || !detail::ReturnsResult(id))) {
        return;
    }
    if (id == 0x80 && in[1] == 0x2000 && result == 0) {
        u64 none = 0;
        s.queue_semaphore_block.compare_exchange_strong(none, in[0]);
    }
    const auto svc_name = detail::SvcName(id);
    const std::string label =
        svc_name.empty() ? fmt::format("svc {:#x}", id) : std::string(svc_name);
    std::scoped_lock lk{s.mutex};
    if (!s.file) {
        return;
    }
    const u64 n = ++s.svc_detail_counts[id];
    switch (id) {
    case 0x01:
        detail::WriteLocked(
            s, fmt::format("svc  SetHeapSize(size={:#x} = {} MiB) -> {}, heap at {:#x}", in[1],
                           in[1] >> 20, detail::ResultText(result), out[1]));
        break;
    case 0x2C:
    case 0x2D:
        if (n <= 64 || result != 0) {
            detail::WriteLocked(s, fmt::format("svc  {}(address={:#x}, size={:#x} = {} MiB) -> {}",
                                               label, in[0], in[1], in[1] >> 20,
                                               detail::ResultText(result)));
        }
        break;
    case 0x29: {
        const u64 type = in[1] & 0xffffffff;
        if (++s.info_queries[{type, in[3]}] == 1 || result != 0) {
            detail::WriteLocked(
                s, fmt::format(
                       "svc  GetInfo({} [{:#x}], handle={:#x}, sub={:#x}) -> {}, value={:#x} ({} MiB)",
                       detail::InfoName(type), type, in[2], in[3], detail::ResultText(result),
                       out[1], out[1] >> 20));
        }
        break;
    }
    case 0x08:
        if (n <= 64 || result != 0) {
            detail::WriteLocked(
                s, fmt::format("svc  CreateThread(entry={:#x}, priority={}, core={}) -> {}", in[1],
                               static_cast<s32>(in[4]), static_cast<s32>(in[5]),
                               detail::ResultText(result)));
        }
        break;
    case 0x80:
        if (n <= 16 || result != 0) {
            detail::WriteLocked(
                s, fmt::format("svc  SetGpuMemoryAttribute({:#x}, {:#x}, {:#x}, {:#x}) -> {}",
                               in[0], in[1], in[2], in[3], detail::ResultText(result)));
        }
        break;
    default:
        break;
    }
    if (detail::ReturnsResult(id) && result != 0) {
        const u64 count = ++s.svc_results[{id, result}];
        if (!detailed && !detail::IsQuietResult(result) && count <= 5) {
            detail::WriteLocked(
                s, fmt::format("svc  {} -> {} (args {:#x} {:#x} {:#x} {:#x})", label,
                               detail::ResultText(result), in[0], in[1], in[2], in[3]));
        }
    }
}

} // namespace Core::LaunchLog
