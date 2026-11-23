// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>

#include "common/common_types.h"
#include "core/hle/service/cmif_types.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Core::Timing {
struct EventType;
}

namespace Kernel {
class KEvent;
class KProcess;
class KReadableEvent;
} // namespace Kernel

namespace Service::Audio {

// nn::audio::StreamFormat as devmgr command 300 and sysmix OpenStream send it
// (nnSdk+0x2ae044, nnSdk+0x2b357c): sample format, channel count, sample rate.
struct StreamFormat {
    u32 sample_format;
    s32 channel_count;
    s32 sample_rate;
};
static_assert(sizeof(StreamFormat) == 0xc, "StreamFormat has wrong size");

// Switch 2 audio streams (nn::audio::OutputStream, built on nn::audio::sysmix and
// nn::audio::devmgr). The client keeps the stream ring in its own memory and passes the address
// to sysmix CreateStream; the server consumes frames straight from it. "audsmx:u" creates and
// opens streams, "auddmg:u" AttachStream/DetachStream starts and stops them (OutputStreamImpl
// Start/Stop, nnSdk+0x225368/0x2255a0), so both services share this registry.
class SystemMixerStreams {
public:
    // Eden's one output device: the name the SDK itself asks for (nnSdk+0x23ba0c, "MainAudioOut"
    // with functionality 1), one stereo 48 kHz sink.
    static constexpr std::string_view OutputDeviceName = "MainAudioOut";
    static constexpr u32 OutputFunctionality = 1;
    static constexpr u32 OutputDeviceId = 1;
    static constexpr u32 OutputChannelCount = 2;
    // Streams are served every 5 ms (240 frames at 48 kHz); the device profile reports it as the
    // device latency, which OutputStreamImpl::Open turns into the minimum stream size
    // (nnSdk+0x225a94-0x225ad0).
    static constexpr u64 PumpPeriodNs = 5'000'000;

    explicit SystemMixerStreams(Core::System& system_);
    ~SystemMixerStreams();

    static bool IsFormatSupported(const StreamFormat& format);

    Result CreateStream(u32* out_id, Kernel::KProcess* process, u32 creation_flags, u64 buffer,
                        u64 size, Kernel::KEvent* data_event);
    void DestroyStream(u32 id);
    Result OpenStream(u32 id, const StreamFormat& format, s32 frame_count, s32 filled_frame_count,
                      u32 flags);
    Result CloseStream(u32 id);
    Result SetDataNotification(u32 id, u32 mode, s32 threshold);
    Result PurgeStream(s32* out_purged, u32 id, s32 frame_count);
    Result AttachStream(u32 id, u64 process_id);
    Result DetachStream(u32 id, u64 process_id);

private:
    struct Stream;

    void Pump();
    void PumpStream(Stream& stream);
    void Notify(Stream& stream, u32 write_index);
    void CloseSink(Stream& stream);

    Core::System& system;
    std::mutex mutex;
    std::map<u32, std::unique_ptr<Stream>> streams;
    u32 next_id{1};
    std::shared_ptr<Core::Timing::EventType> pump_event;
    std::atomic<bool> pump_scheduled{false};
};

class IAudioSystemMixerSession;

// "audsmx:u" (nn::audio::sysmix::IUserSessionManager). The SDK converts the session to a domain
// before its first request (nnSdk+0x2b2b18).
class IAudioSystemMixer final : public ServiceFramework<IAudioSystemMixer> {
public:
    explicit IAudioSystemMixer(Core::System& system_, std::shared_ptr<SystemMixerStreams> streams_);
    ~IAudioSystemMixer() override;

private:
    Result OpenUserSession(Out<SharedPointer<IAudioSystemMixerSession>> out_session, u32 version,
                           ClientProcessId process_id, InCopyHandle<Kernel::KProcess> process);

    std::shared_ptr<SystemMixerStreams> streams;
};

} // namespace Service::Audio
