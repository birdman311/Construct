// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <span>
#include <string>

#include <fmt/format.h>

#include "audio_core/audio_core.h"
#include "audio_core/common/common.h"
#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_stream.h"
#include "common/logging.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/audio/audio_system_mixer.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/memory.h"

namespace Service::Audio {

namespace {

// Stream ring the SDK lays out in the client buffer (nnSdk+0x2b38d0): consumer header at 0x40,
// producer header at 0x1000, each with its frame index at +4, then the frames from 0x1040 when
// OpenStream has no timestamp area. Indexes run modulo frame count + 1 (nnSdk+0x2b3a04). The
// client writes frames and the producer index (nnSdk+0x2b3c4c-0x2b3d28) and reads the consumer
// index for its free frame count (nnSdk+0x2b2164).
constexpr u64 ConsumerIndexOffset = 0x44;
constexpr u64 ProducerIndexOffset = 0x1004;
constexpr u64 PayloadOffset = 0x1040;

// Creation flags with (flags & 7) == 4 make the client the producer (nnSdk+0x2b1da8);
// OutputStreamImpl passes 0x24 or 4 (nnSdk+0x225ae0-0x225b18), InputStreamImpl 2 (nnSdk+0x1f6984).
constexpr u32 CreationRoleMask = 7;
constexpr u32 CreationRoleClientProducer = 4;

// StreamDataNotificationMode that output and input streams ask for (nnSdk+0x225b68,
// nnSdk+0x1f69c0), with a frame threshold. For a stream the client writes, the event means room
// for at least that many frames; the movie player's audio thread waits on it (10 ms timeout) when
// it cannot write more (Owl.nss+0x3b608bc-0x3b608c8).
constexpr u32 DataNotificationFrames = 2;

// Bytes per sample of sample formats 1-5 (nnSdk ro 0xc18360, read by OpenStreamImpl at
// nnSdk+0x2b1c14). Format 2 is the only 2-byte one; the engine opens its stream with it
// (Owl.nss+0x1023ce4, Owl.nss+0x1023d80) and writes interleaved 16-bit samples
// (Owl.nss+0x36d5ed0-0x36d5f04). Other formats are consumed in time but not played.
constexpr std::array<u32, 5> SampleByteSizes{1, 2, 4, 4, 4};
constexpr u32 SampleFormatPcm16 = 2;

constexpr u32 MaxBlockFrames = AudioCore::TargetSampleCount;
constexpr u32 MaxQueuedSinkBlocks = 4;
constexpr u64 NsPerSecond = 1'000'000'000;

u32 GetSampleByteSize(u32 sample_format) {
    return sample_format >= 1 && sample_format <= SampleByteSizes.size()
               ? SampleByteSizes[sample_format - 1]
               : 0;
}

} // namespace

struct SystemMixerStreams::Stream {
    Kernel::KProcess* process{};
    Kernel::KEvent* data_event{};
    u64 buffer{};
    u64 size{};
    u32 creation_flags{};
    bool open{};
    bool engaged{};
    StreamFormat format{};
    u32 frame_bytes{};
    u32 slots{};
    u32 read_index{};
    u64 frame_remainder{};
    u32 data_mode{};
    s32 data_threshold{};
    AudioCore::Sink::SinkStream* sink{};
    u32 sink_channels{};
    u64 sink_tag{};
    std::array<s16, MaxBlockFrames * AudioCore::MaxChannels> samples{};
};

SystemMixerStreams::SystemMixerStreams(Core::System& system_) : system{system_} {
    pump_event = Core::Timing::CreateEvent(
        "audsmx:u StreamPump",
        [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
            Pump();
            return std::nullopt;
        });
}

SystemMixerStreams::~SystemMixerStreams() {
    if (pump_scheduled) {
        system.CoreTiming().UnscheduleEvent(pump_event);
    }
    for (auto& [id, stream] : streams) {
        CloseSink(*stream);
        stream->process->Close(system.Kernel());
    }
}

bool SystemMixerStreams::IsFormatSupported(const StreamFormat& format) {
    return GetSampleByteSize(format.sample_format) != 0 && format.channel_count >= 1 &&
           format.channel_count <= static_cast<s32>(AudioCore::MaxChannels) &&
           format.sample_rate > 0;
}

Result SystemMixerStreams::CreateStream(u32* out_id, Kernel::KProcess* process,
                                        u32 creation_flags, u64 buffer, u64 size,
                                        Kernel::KEvent* data_event) {
    R_UNLESS(process != nullptr, ResultInvalidHandle);
    // Only streams the client writes are served: the device list offers no input device.
    R_UNLESS((creation_flags & CreationRoleMask) == CreationRoleClientProducer, ResultNotSupported);
    R_UNLESS(buffer != 0 && size > PayloadOffset &&
                 process->GetMemory().IsValidVirtualAddressRange(buffer, size),
             ResultInvalidAddressInfo);

    auto stream = std::make_unique<Stream>();
    stream->process = process;
    stream->data_event = data_event;
    stream->buffer = buffer;
    stream->size = size;
    stream->creation_flags = creation_flags;
    process->Open(system.Kernel());

    std::scoped_lock lk{mutex};
    const u32 id = next_id++;
    streams.emplace(id, std::move(stream));
    *out_id = id;
    R_SUCCEED();
}

void SystemMixerStreams::DestroyStream(u32 id) {
    std::unique_ptr<Stream> stream;
    {
        std::scoped_lock lk{mutex};
        const auto it = streams.find(id);
        if (it == streams.end()) {
            return;
        }
        stream = std::move(it->second);
        streams.erase(it);
        CloseSink(*stream);
    }
    stream->process->Close(system.Kernel());
}

Result SystemMixerStreams::OpenStream(u32 id, const StreamFormat& format, s32 frame_count,
                                      s32 filled_frame_count, u32 flags) {
    std::scoped_lock lk{mutex};
    const auto it = streams.find(id);
    R_UNLESS(it != streams.end(), ResultNotFound);
    auto& stream = *it->second;
    R_UNLESS(!stream.open, ResultOperationFailed);
    R_UNLESS(IsFormatSupported(format), ResultNotSupported);
    R_UNLESS(frame_count > 0 && filled_frame_count >= 0 && filled_frame_count <= frame_count,
             ResultInvalidArgument);
    // Flag bit 0 adds a timestamp area before the frames (nnSdk+0x2b1cdc); OutputStreamImpl
    // never sets it (nnSdk+0x225b50) and only the layout without it is served.
    R_UNLESS(flags == 0, ResultNotSupported);

    const u64 slots = static_cast<u64>(frame_count) + 1;
    const u64 frame_bytes = static_cast<u64>(GetSampleByteSize(format.sample_format)) *
                            static_cast<u64>(format.channel_count);
    R_UNLESS(PayloadOffset + slots * frame_bytes <= stream.size, ResultInsufficientBuffer);

    // The client lays out the ring before it sends OpenStream (nnSdk+0x2b1dbc).
    auto& memory = stream.process->GetMemory();
    const u32 read_index = memory.Read32(stream.buffer + ConsumerIndexOffset);
    const u32 write_index = memory.Read32(stream.buffer + ProducerIndexOffset);
    R_UNLESS(read_index < slots && write_index < slots, ResultInvalidAddressInfo);

    stream.format = format;
    stream.frame_bytes = static_cast<u32>(frame_bytes);
    stream.slots = static_cast<u32>(slots);
    stream.read_index = read_index;
    stream.frame_remainder = 0;
    stream.data_mode = 0;
    stream.open = true;
    stream.engaged = false;

    if (format.sample_format == SampleFormatPcm16 &&
        format.sample_rate == static_cast<s32>(AudioCore::TargetSampleRate) &&
        (format.channel_count == 1 || format.channel_count == 2 || format.channel_count == 6)) {
        stream.sink_channels =
            format.channel_count == 1 ? 2 : static_cast<u32>(format.channel_count);
        stream.sink = system.AudioCore().GetOutputSink().AcquireSinkStream(
            system, stream.sink_channels, fmt::format("audsmx:u stream {}", id),
            AudioCore::Sink::StreamType::Out);
    } else {
        LOG_WARNING(Service_Audio,
                    "stream {}: sample format {}, {} channels, {} Hz is consumed without output",
                    id, format.sample_format, format.channel_count, format.sample_rate);
    }
    LOG_INFO(Service_Audio,
             "stream {}: format {}, {} channels, {} Hz, {} frames, buffer {:#x}+{:#x}", id,
             format.sample_format, format.channel_count, format.sample_rate, frame_count,
             stream.buffer, stream.size);
    R_SUCCEED();
}

Result SystemMixerStreams::CloseStream(u32 id) {
    std::scoped_lock lk{mutex};
    const auto it = streams.find(id);
    R_UNLESS(it != streams.end(), ResultNotFound);
    auto& stream = *it->second;
    CloseSink(stream);
    stream.engaged = false;
    stream.open = false;
    stream.data_mode = 0;
    R_SUCCEED();
}

Result SystemMixerStreams::SetDataNotification(u32 id, u32 mode, s32 threshold) {
    std::scoped_lock lk{mutex};
    const auto it = streams.find(id);
    R_UNLESS(it != streams.end(), ResultNotFound);
    auto& stream = *it->second;
    if (mode == 0) {
        stream.data_mode = 0;
        R_SUCCEED();
    }
    R_UNLESS(stream.open, ResultOperationFailed);
    R_UNLESS(mode == DataNotificationFrames, ResultNotSupported);
    R_UNLESS(threshold >= 0, ResultInvalidArgument);
    stream.data_mode = mode;
    stream.data_threshold = threshold;
    const u32 write_index =
        stream.process->GetMemory().Read32(stream.buffer + ProducerIndexOffset);
    if (write_index < stream.slots) {
        Notify(stream, write_index);
    }
    R_SUCCEED();
}

Result SystemMixerStreams::PurgeStream(s32* out_purged, u32 id, s32 frame_count) {
    std::scoped_lock lk{mutex};
    const auto it = streams.find(id);
    R_UNLESS(it != streams.end(), ResultNotFound);
    auto& stream = *it->second;
    R_UNLESS(stream.open, ResultOperationFailed);
    R_UNLESS(frame_count >= 0, ResultInvalidArgument);

    // The client accepts at most the frames it asked for, or its capacity for 0
    // (nnSdk+0x2b26e4-0x2b26f8).
    auto& memory = stream.process->GetMemory();
    const u32 write_index = memory.Read32(stream.buffer + ProducerIndexOffset);
    R_UNLESS(write_index < stream.slots, ResultInvalidAddressInfo);
    const u32 filled = (write_index + stream.slots - stream.read_index) % stream.slots;
    const u32 purged =
        frame_count == 0 ? filled : std::min(filled, static_cast<u32>(frame_count));
    stream.read_index = (stream.read_index + purged) % stream.slots;
    memory.Write32(stream.buffer + ConsumerIndexOffset, stream.read_index);
    *out_purged = static_cast<s32>(purged);
    Notify(stream, write_index);
    R_SUCCEED();
}

Result SystemMixerStreams::AttachStream(u32 id, u64 process_id) {
    {
        std::scoped_lock lk{mutex};
        const auto it = streams.find(id);
        R_UNLESS(it != streams.end(), ResultNotFound);
        auto& stream = *it->second;
        R_UNLESS(stream.process->GetProcessId() == process_id, ResultInvalidHandle);
        R_UNLESS(stream.open, ResultOperationFailed);
        if (!stream.engaged && stream.sink != nullptr) {
            stream.sink->Start();
        }
        stream.engaged = true;
    }
    if (!pump_scheduled.exchange(true)) {
        system.CoreTiming().ScheduleLoopingEvent(std::chrono::nanoseconds(PumpPeriodNs),
                                                 std::chrono::nanoseconds(PumpPeriodNs),
                                                 pump_event);
    }
    R_SUCCEED();
}

Result SystemMixerStreams::DetachStream(u32 id, u64 process_id) {
    std::scoped_lock lk{mutex};
    const auto it = streams.find(id);
    R_UNLESS(it != streams.end(), ResultNotFound);
    auto& stream = *it->second;
    R_UNLESS(stream.process->GetProcessId() == process_id, ResultInvalidHandle);
    if (stream.engaged && stream.sink != nullptr) {
        stream.sink->Stop();
    }
    stream.engaged = false;
    R_SUCCEED();
}

void SystemMixerStreams::Pump() {
    std::scoped_lock lk{mutex};
    if (system.IsShuttingDown()) {
        return;
    }
    for (auto& [id, stream] : streams) {
        PumpStream(*stream);
    }
}

void SystemMixerStreams::PumpStream(Stream& stream) {
    if (!stream.open || !stream.engaged) {
        return;
    }
    auto& memory = stream.process->GetMemory();
    const u32 write_index = memory.Read32(stream.buffer + ProducerIndexOffset);
    if (write_index >= stream.slots) {
        LOG_ERROR(Service_Audio, "stream ring producer index {} is outside {} slots, stopping it",
                  write_index, stream.slots);
        stream.engaged = false;
        return;
    }
    // While the host output still holds enough, leave the frames to the next period.
    if (stream.sink != nullptr && stream.sink->GetQueueSize() >= MaxQueuedSinkBlocks) {
        return;
    }

    const u64 scaled = static_cast<u64>(stream.format.sample_rate) * PumpPeriodNs +
                       stream.frame_remainder;
    stream.frame_remainder = scaled % NsPerSecond;
    const u32 due = static_cast<u32>(scaled / NsPerSecond);
    const u32 filled = (write_index + stream.slots - stream.read_index) % stream.slots;
    u32 count = std::min(filled, due);

    if (count > 0 && stream.sink != nullptr) {
        count = std::min(count, MaxBlockFrames);
        const u64 payload = stream.buffer + PayloadOffset;
        const u32 channels = static_cast<u32>(stream.format.channel_count);
        const u32 first = std::min(count, stream.slots - stream.read_index);
        std::array<s16, MaxBlockFrames * AudioCore::MaxChannels> input{};
        memory.ReadBlock(payload + static_cast<u64>(stream.read_index) * stream.frame_bytes,
                         input.data(), static_cast<size_t>(first) * stream.frame_bytes);
        if (count > first) {
            memory.ReadBlock(payload, input.data() + static_cast<size_t>(first) * channels,
                             static_cast<size_t>(count - first) * stream.frame_bytes);
        }
        if (channels == 1) {
            for (u32 i = 0; i < count; ++i) {
                stream.samples[i * 2] = input[i];
                stream.samples[i * 2 + 1] = input[i];
            }
        } else {
            std::copy_n(input.begin(), static_cast<size_t>(count) * channels,
                        stream.samples.begin());
        }
        AudioCore::Sink::SinkBuffer block{
            .frames = count,
            .frames_played = 0,
            .tag = stream.sink_tag++,
            .consumed = false,
        };
        stream.sink->AppendBuffer(
            block, std::span<s16>(stream.samples.data(),
                                  static_cast<size_t>(count) * stream.sink_channels));
    }
    if (count > 0) {
        stream.read_index = (stream.read_index + count) % stream.slots;
        memory.Write32(stream.buffer + ConsumerIndexOffset, stream.read_index);
    }
    Notify(stream, write_index);
}

void SystemMixerStreams::Notify(Stream& stream, u32 write_index) {
    if (stream.data_mode != DataNotificationFrames || stream.data_event == nullptr) {
        return;
    }
    const u32 filled = (write_index + stream.slots - stream.read_index) % stream.slots;
    const u32 free_frames = stream.slots - 1 - filled;
    if (free_frames >= static_cast<u32>(stream.data_threshold)) {
        stream.data_event->Signal(system.Kernel());
    }
}

void SystemMixerStreams::CloseSink(Stream& stream) {
    if (stream.sink == nullptr) {
        return;
    }
    stream.sink->Stop();
    system.AudioCore().GetOutputSink().CloseStream(stream.sink);
    stream.sink = nullptr;
}

// nn::audio::sysmix::IExternalStreamSession, the domain object CreateStream returns. Command ids
// from the SDK proxy vtable (nnSdk data 0xe632b8) and the sysmix functions that use each slot.
class IAudioSystemMixerStream final : public ServiceFramework<IAudioSystemMixerStream> {
public:
    explicit IAudioSystemMixerStream(Core::System& system_,
                                     std::shared_ptr<SystemMixerStreams> streams_)
        : ServiceFramework{system_, "IAudioSystemMixerStream"},
          service_context{system_, "IAudioSystemMixerStream"},
          data_event{service_context.CreateEvent("IAudioSystemMixerStream:DataEvent")},
          streams{std::move(streams_)} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, D<&IAudioSystemMixerStream::OpenStream>, "OpenStream"},
            {1, D<&IAudioSystemMixerStream::CloseStream>, "CloseStream"},
            {2, nullptr, "GetStreamState"},
            {3, nullptr, "GetStreamGain"},
            {4, nullptr, "SetStreamGain"},
            {5, nullptr, "AcquireStreamStateChangeEvent"},
            {6, nullptr, "ReleaseStreamStateChangeEvent"},
            {7, D<&IAudioSystemMixerStream::AcquireStreamDataEvent>, "AcquireStreamDataEvent"},
            {8, D<&IAudioSystemMixerStream::ReleaseStreamDataEvent>, "ReleaseStreamDataEvent"},
            {9, nullptr, "ClockStream"},
            {10, nullptr, "PrimeStream"},
            {11, D<&IAudioSystemMixerStream::PurgeStream>, "PurgeStream"},
            {12, nullptr, "RenameStream"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

    ~IAudioSystemMixerStream() override {
        streams->DestroyStream(id);
        service_context.CloseEvent(data_event);
    }

    Kernel::KEvent* GetDataEvent() const {
        return data_event;
    }

    void SetId(u32 id_) {
        id = id_;
    }

private:
    // Format, frame count, frames the client marks filled at start, open flags
    // (nnSdk+0x2b357c-0x2b3594).
    Result OpenStream(StreamFormat format, s32 frame_count, s32 filled_frame_count, u32 flags) {
        R_RETURN(streams->OpenStream(id, format, frame_count, filled_frame_count, flags));
    }

    Result CloseStream() {
        LOG_DEBUG(Service_Audio, "called, stream {}", id);
        R_RETURN(streams->CloseStream(id));
    }

    // Mode and frame threshold (nnSdk+0x2b36e4); the SDK attaches the copy handle as an
    // auto-clear event (nnSdk+0x2b1b60).
    Result AcquireStreamDataEvent(OutCopyHandle<Kernel::KReadableEvent> out_event, u32 mode,
                                  s32 threshold) {
        LOG_DEBUG(Service_Audio, "called, stream {}, mode {}, threshold {}", id, mode, threshold);
        R_TRY(streams->SetDataNotification(id, mode, threshold));
        *out_event = &data_event->GetReadableEvent();
        R_SUCCEED();
    }

    Result ReleaseStreamDataEvent() {
        R_RETURN(streams->SetDataNotification(id, 0, 0));
    }

    // Server-side for streams the client writes (nnSdk+0x2b26a0-0x2b26b8); the frame count goes
    // in and the purged frame count comes back (nnSdk+0xf291c, nnSdk+0xf29ac).
    Result PurgeStream(Out<s32> out_purged, s32 frame_count) {
        R_RETURN(streams->PurgeStream(out_purged, id, frame_count));
    }

    KernelHelpers::ServiceContext service_context;
    Kernel::KEvent* data_event;
    std::shared_ptr<SystemMixerStreams> streams;
    u32 id{};
};

// nn::audio::sysmix::IUserSession (SDK proxy vtable nnSdk data 0xe631a0).
class IAudioSystemMixerSession final : public ServiceFramework<IAudioSystemMixerSession> {
public:
    explicit IAudioSystemMixerSession(Core::System& system_,
                                      std::shared_ptr<SystemMixerStreams> streams_,
                                      Kernel::KProcess* process_)
        : ServiceFramework{system_, "IAudioSystemMixerSession"}, streams{std::move(streams_)},
          process{process_} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {1, D<&IAudioSystemMixerSession::CreateStream>, "CreateStream"},
            {2, nullptr, "DisengageStreams"},
            {3, nullptr, "EngageStreams"},
        };
        // clang-format on
        RegisterHandlers(functions);
        process->Open(system_.Kernel());
    }

    ~IAudioSystemMixerSession() override {
        process->Close(system.Kernel());
    }

private:
    // Device id from the device profile, creation flags, aruid, address and size of the stream
    // memory in the client, and its name in an A buffer (nnSdk+0x2b31ec-0x2b31fc); the stream id
    // and the stream object come back (nnSdk+0x2b32a0-0x2b32ac).
    Result CreateStream(Out<u32> out_stream_id,
                        Out<SharedPointer<IAudioSystemMixerStream>> out_stream, u32 device_id,
                        u32 creation_flags, ClientAppletResourceUserId aruid, u64 buffer, u64 size,
                        InBuffer<BufferAttr_HipcMapAlias> name) {
        LOG_INFO(Service_Audio,
                 "called, device {}, flags {:#x}, buffer {:#x}+{:#x}, name {}, aruid {:#x}",
                 device_id, creation_flags, buffer, size, Common::StringFromBuffer(name),
                 aruid.pid);
        R_UNLESS(device_id == SystemMixerStreams::OutputDeviceId, ResultNotFound);
        auto stream = std::make_shared<IAudioSystemMixerStream>(system, streams);
        u32 stream_id{};
        R_TRY(streams->CreateStream(&stream_id, process, creation_flags, buffer, size,
                                    stream->GetDataEvent()));
        stream->SetId(stream_id);
        *out_stream_id = stream_id;
        *out_stream = std::move(stream);
        R_SUCCEED();
    }

    std::shared_ptr<SystemMixerStreams> streams;
    Kernel::KProcess* process;
};

IAudioSystemMixer::IAudioSystemMixer(Core::System& system_,
                                     std::shared_ptr<SystemMixerStreams> streams_)
    : ServiceFramework{system_, "audsmx:u"}, streams{std::move(streams_)} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IAudioSystemMixer::OpenUserSession>, "OpenUserSession"},
    };
    // clang-format on
    RegisterHandlers(functions);
}

IAudioSystemMixer::~IAudioSystemMixer() = default;

// u32 1, the process id and the client's own process handle (nnSdk+0x2b1448, nnSdk+0x2b2d30-
// 0x2b2d94); the object id of the user session comes back (nnSdk+0x2b2e3c).
Result IAudioSystemMixer::OpenUserSession(Out<SharedPointer<IAudioSystemMixerSession>> out_session,
                                          u32 version, ClientProcessId process_id,
                                          InCopyHandle<Kernel::KProcess> process) {
    LOG_INFO(Service_Audio, "called, version {}, process id {:#x}", version, process_id.pid);
    R_UNLESS(process, ResultInvalidHandle);
    *out_session = std::make_shared<IAudioSystemMixerSession>(system, streams, process.Get());
    R_SUCCEED();
}

} // namespace Service::Audio
