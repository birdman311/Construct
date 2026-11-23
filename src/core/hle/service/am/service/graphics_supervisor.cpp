// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <map>
#include "common/alignment.h"
#include "core/hle/service/nvdrv/core/nvmap.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/launch_log.h"
#include "core/hle/service/nvnflinger/buffer_queue_producer.h"
#include "core/hle/service/nvnflinger/ui/graphic_buffer.h"
#include "core/hle/kernel/k_client_session.h"
#include "core/hle/kernel/k_scoped_resource_reservation.h"
#include "core/hle/kernel/k_session.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/k_transfer_memory.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/am/service/graphics_supervisor.h"
#include "core/hle/service/am/service/graphics_session.h"
#include "core/hle/service/am/service/gpu_session.h"
#include "core/hle/service/am/service/graphics_supervisor_protocol.h"
#include "core/hle/service/am/service/graphics_syncpoint_cache.h"
#include "core/hle/service/am/service/native_image_layout.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/nvdrv/nvdrv_interface.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "core/hle/service/nvdrv/core/syncpoint_manager.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/sm/sm.h"
#include "core/hle/service/vi/application_display_service.h"
#include "core/hle/service/vi/container.h"
#include "core/hle/service/vi/manager_root_service.h"
#include "core/hle/service/vi/vi_types.h"
#include "core/memory.h"
#include "video_core/gpu.h"
#include "video_core/host1x/ffmpeg.h"
#include "video_core/host1x/host1x.h"

namespace Service::AM {
struct NativeImageRegistry {
    struct Image {
        u32 id;
        u32 memory_id;
        std::array<u32, 16> preference;
        u64 size;
        bool window_pool{};
    };
    NativeImageRegistry(Kernel::KernelCore& kernel_, Kernel::KProcess* process_)
        : kernel{kernel_}, process{process_} {
        process->Open(kernel);
    }
    ~NativeImageRegistry() {
        if (module) {
            for (const auto& image : images) {
                ReleaseMemory(image.memory_id);
            }
        }
        if (memory_session) {
            module->GetContainer().CloseSession(*memory_session);
        }
        process->Close(kernel);
    }
    void ReleaseMemory(u32 id) {
        const auto info = module->GetContainer().GetNvMapFile().FreeHandle(id, true);
        if (info && info->can_unlock && info->address != 0) {
            process->GetPageTable().UnlockForDeviceAddressSpace(info->address, info->size);
        }
    }
    Kernel::KernelCore& kernel;
    Kernel::KProcess* process;
    std::mutex mutex;
    std::shared_ptr<Nvidia::Module> module;
    std::optional<Nvidia::NvCore::SessionId> memory_session;
    std::vector<Image> images;
};
using NativeImage = NativeImageRegistry::Image;

// Video decoder image pools (supervisor 991/911/912/961): NV12 images in guest memory that the
// video decoder sessions write decoded frames into and the guest reads back on the CPU.
struct DecoderImageRegistry {
    struct Image {
        u32 pool;         ///< Handle of the pool's first image.
        u64 address;      ///< Guest address of the image.
        u64 pool_offset;  ///< Offset of the image from the pool's memory.
        u32 width;
        u32 height;
        u32 pitch;
        u32 aligned_height;
    };
    std::mutex mutex;
    std::map<u32, Image> images;
};


namespace {

// Test build: while a movie decoder session is started, log every frame handed to the game,
// every image it hands back and every present ("TL" lines in the launch log).
using Core::LaunchLog::g_movie_timeline;

// TIPC payloads follow a 32-bit result, so these handlers push result.raw.

class ISyncpointSupervisor final : public ServiceFramework<ISyncpointSupervisor> {
public:
    ISyncpointSupervisor(Core::System& system, std::shared_ptr<Nvidia::Module> module)
        : ServiceFramework{system, "ISyncpointSupervisor"}, m_module{std::move(module)},
          m_context{system, "SyncpointSupervisor"} {
        static const FunctionInfo functions[] = {
            {1, &ISyncpointSupervisor::GetMaximumEventCount, "GetMaximumEventCount"},
            {21, &ISyncpointSupervisor::SetEventCount, "SetEventCount"},
            {22, &ISyncpointSupervisor::GetEvent, "GetEvent"},
            {103, &ISyncpointSupervisor::QueryCounter, "QueryCounter"},
            {201, &ISyncpointSupervisor::RequestEvent, "RequestEvent"},
            {202, &ISyncpointSupervisor::CancelEvent, "CancelEvent"},
        };
        RegisterHandlersTipc(functions);
    }

    ~ISyncpointSupervisor() override {
        if (m_poll) {
            system.CoreTiming().UnscheduleEvent(m_poll);
        }
        for (auto* event : m_events) {
            m_context.CloseEvent(event);
        }
    }

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = LockService();
        LOG_DEBUG(Service_AM, "syncpoint unsupported command={}, words={:08x}", ctx.GetCommand(),
                  fmt::join(std::span<const u32>(ctx.CommandBuffer() + 2, std::min<u32>(ctx.CommandBuffer()[1], 32)), ","));
        InvokeRequestTipc(ctx);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Kernel::ResultNotImplemented.raw);
        return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }

private:
    static constexpr u32 MaximumEventCount = 63;

    void GetMaximumEventCount(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const Result result = packet[0] == 0x11 && packet[1] == 0
            ? ResultSuccess : Kernel::ResultInvalidCombination;
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result.raw);
        rb.Push(MaximumEventCount);
        LOG_DEBUG(Service_AM, "syncpoint maximum events={}", MaximumEventCount);
    }

    void SetEventCount(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x25 && packet[1] == 1 && packet[2] <= MaximumEventCount) {
            m_event_count = packet[2];
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
        LOG_DEBUG(Service_AM, "syncpoint event count={}, result={:#x}", m_event_count, result.raw);
    }

    void QueryCounter(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        u64 value{};
        if (packet[0] == 0x77 && packet[1] == 1) {
            const u32 id = ~packet[2];
            auto current = m_module->GetContainer().GetSyncpointManager().ReadAllocatedSyncpoint(id);
            result = current ? ResultSuccess : Kernel::ResultNotFound;
            value = current.value_or(0);
            LOG_DEBUG(Service_AM, "query syncpoint id={}, value={}, result={:#x}", id, value, result.raw);
        }
        IPC::ResponseBuilder rb{ctx, 5};
        rb.Push(result.raw);
        rb.Push(value);
        rb.Push<u32>(0); // Existing NVDRV counters use 32-bit modular comparisons.
    }

    // Event-backed syncpoint waits (nn::syncpt WaitSyncPoint, TimedWaitSyncPoint and
    // CreateSyncPointEvent, which the movie library aborts without). Command 22 hands out the
    // manual-clear event of a slot, 201 arms it for a syncpoint threshold and 202 cancels it; the
    // client waits on the event after a cancel too, so a cancel signals it.
    void GetEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Kernel::KReadableEvent* readable{};
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x26 && packet[1] == 1 && packet[2] < m_events.size()) {
            auto*& event = m_events[packet[2]];
            if (!event) {
                event = m_context.CreateEvent("SyncpointEvent");
            }
            result = event ? ResultSuccess : Kernel::ResultOutOfResource;
            if (event) {
                readable = &event->GetReadableEvent();
                readable->Open(kernel);
            }
        }
        LOG_DEBUG(Service_AM, "syncpoint event index={}, result={:#x}", packet[2], result.raw);
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(readable);
    }

    void RequestEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xd9 && packet[1] == 5 && packet[2] < m_events.size() && m_events[packet[2]]) {
            const u32 index = packet[2];
            Request request{~packet[3], u64{packet[4]} | (u64{packet[5]} << 32), (packet[6] & 0xff) != 0};
            result = m_module->GetContainer().GetSyncpointManager().IsSyncpointAllocated(request.id)
                ? ResultSuccess : Kernel::ResultNotFound;
            if (result.IsSuccess()) {
                m_events[index]->Clear(kernel);
                if (IsReached(request)) {
                    m_events[index]->Signal(kernel);
                } else {
                    m_pending[index] = request;
                    SchedulePoll();
                }
            }
            LOG_DEBUG(Service_AM, "syncpoint request index={}, id={}, threshold={}, wide={}, result={:#x}",
                      index, request.id, request.threshold, request.wide, result.raw);
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }

    void CancelEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xda && packet[1] == 1 && packet[2] < m_events.size() && m_events[packet[2]]) {
            m_pending[packet[2]].reset();
            m_events[packet[2]]->Signal(kernel);
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }

    struct Request {
        u32 id;
        u64 threshold;
        bool wide; // SyncValueMode 1: 64-bit comparison; 0: 32-bit wrapping comparison.
    };

    bool IsReached(const Request& request) {
        const auto value = m_module->GetContainer().GetSyncpointManager().ReadAllocatedSyncpoint(request.id);
        if (!value) {
            return true; // A freed syncpoint can no longer advance; do not leave the waiter hanging.
        }
        return request.wide ? u64{*value} >= request.threshold
                            : static_cast<s32>(*value - static_cast<u32>(request.threshold)) >= 0;
    }

    void SchedulePoll() {
        if (m_poll) {
            return;
        }
        m_poll = Core::Timing::CreateEvent("SyncpointEventPoll", [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
            const auto guard = LockService();
            for (size_t i = 0; i < m_pending.size(); ++i) {
                if (m_pending[i] && IsReached(*m_pending[i])) {
                    m_pending[i].reset();
                    m_events[i]->Signal(kernel);
                }
            }
            return std::nullopt;
        });
        system.CoreTiming().ScheduleLoopingEvent(std::chrono::milliseconds{1}, std::chrono::milliseconds{1}, m_poll);
    }

    std::shared_ptr<Nvidia::Module> m_module;
    u32 m_event_count{};
    KernelHelpers::ServiceContext m_context;
    std::array<Kernel::KEvent*, 64> m_events{};
    std::array<std::optional<Request>, 64> m_pending{};
    std::shared_ptr<Core::Timing::EventType> m_poll;
};

// The session supervisor command 1121 hands out for nn::oe::QueryApplicationGpuTimeSpan. Command 0
// is a handshake that must echo the u64 it is sent (the SDK aborts otherwise); command 11 reports
// the application's GPU time between two ticks, which Eden does not measure (0 ns).
class IApplicationGpuTimeSession final : public ServiceFramework<IApplicationGpuTimeSession> {
public:
    explicit IApplicationGpuTimeSession(Core::System& system)
        : ServiceFramework{system, "IApplicationGpuTimeSession"} {
        static const FunctionInfo functions[] = {
            {0, &IApplicationGpuTimeSession::Handshake, "Handshake"},
            {11, &IApplicationGpuTimeSession::QueryGpuTimeSpan, "QueryGpuTimeSpan"},
        };
        RegisterHandlersTipc(functions);
    }

private:
    void Handshake(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess.raw);
        rb.Push(u64{packet[2]} | (u64{packet[3]} << 32));
    }

    void QueryGpuTimeSpan(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess.raw);
        rb.Push<u64>(0);
    }
};

struct DecoderImageLayout {
    u32 width;
    u32 height;
    u32 pitch;
    u32 aligned_height;
    u64 size;
};

// nn::codec's VideoDecoderImagePreference: word 1 is the format code, word 2 (width - 1) |
// (height - 1) << 16. Codes 1 and 14 are the CPU read-back NV12 formats NVIDIA's movie library
// uses; Eden lays them out pitch-linear: luma rows, then interleaved chroma rows, same pitch.
std::optional<DecoderImageLayout> GetDecoderImageLayout(std::span<const u32> preference) {
    if (preference[1] != 1 && preference[1] != 14) {
        return std::nullopt;
    }
    DecoderImageLayout layout{};
    layout.width = (preference[2] & 0xffff) + 1;
    layout.height = (preference[2] >> 16) + 1;
    layout.pitch = Common::AlignUp(layout.width, 256U);
    layout.aligned_height = Common::AlignUp(layout.height, 16U);
    layout.size = Common::AlignUp(u64{layout.pitch} * layout.aligned_height * 3 / 2, Kernel::PageSize);
    return layout;
}

// A hardware video decoder session (nn::codec::*Decoder*, used by NVIDIA's movie library for the
// movies of Hyrule Warriors: Age of Imprisonment, WebM VP9). Each submitted bitstream block is
// decoded with FFmpeg and the frame written as NV12 into the oldest output image the client has
// handed over (311 register, 314 release). Every handed-over image gets a fence on the session's
// syncpoint that is reached once a frame is in it; 213 then describes that frame. The client
// aborts on any error, so a codec FFmpeg cannot decode gets black frames: the movie still runs
// for its length and ends.
class IVideoDecoderSession final : public ServiceFramework<IVideoDecoderSession> {
public:
    IVideoDecoderSession(Core::System& system, std::shared_ptr<Nvidia::Module> module,
                         Kernel::KProcess* process, std::shared_ptr<DecoderImageRegistry> images)
        : ServiceFramework{system, "IVideoDecoderSession"}, m_module{std::move(module)},
          m_process{process}, m_images{std::move(images)}, m_context{system, "VideoDecoderSession"} {
        m_process->Open(kernel);
        static const FunctionInfo functions[] = {
            {3, &IVideoDecoderSession::GetSubmissionReadyEvent, "GetSubmissionReadyEvent"},
            {6, &IVideoDecoderSession::Start, "Start"},
            {8, &IVideoDecoderSession::Flush, "Flush"},
            {101, &IVideoDecoderSession::Configure, "Configure"},
            {109, &IVideoDecoderSession::SubmitInputBuffer, "SubmitInputBuffer"},
            {115, &IVideoDecoderSession::Finalize, "Finalize"},
            {200, &IVideoDecoderSession::Initialize, "InitializeWithTransferMemory"},
            {201, &IVideoDecoderSession::Initialize, "InitializeH264WithTransferMemory"},
            {207, &IVideoDecoderSession::Stop, "Stop"},
            {210, &IVideoDecoderSession::Initialize, "InitializeWithProcessMemory"},
            {212, &IVideoDecoderSession::UnregisterBuffer, "UnregisterBuffer"},
            {213, &IVideoDecoderSession::GetOutputBufferProperties, "GetOutputBufferProperties"},
            {311, &IVideoDecoderSession::RegisterBuffer, "RegisterBuffer"},
            {314, &IVideoDecoderSession::RegisterBuffer, "ReleaseOutputBuffer"},
        };
        RegisterHandlersTipc(functions);
    }

    ~IVideoDecoderSession() override {
        if (m_syncpoint) {
            m_module->GetContainer().GetSyncpointManager().FreeSyncpoint(*m_syncpoint);
        }
        m_context.CloseEvent(m_ready_event);
        m_process->Close(kernel);
    }

private:
    struct DecodedFrame {
        std::vector<u8> nv12; ///< Tightly packed: width x height luma, then interleaved chroma.
        u32 width;
        u32 height;
        s64 timestamp_ns;
        bool end_of_stream;
    };
    struct OutputRecord {
        u32 image;
        u32 width;
        u32 height;
        s64 timestamp_ns;
        bool end_of_stream;
    };

    void Reply(HLERequestContext& ctx, Result result) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }

    // 200 (VP9/VP8/AV1) and 201 (H264): u64 work size, codec, input and output formats after a
    // copied transfer memory (the work buffer, not needed here); 210: codec and formats after the
    // process handle. Codec ids: 0 H264, 1 VP9, 3 AV1, 4 VP8.
    void Initialize(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u32 codec = packet[ctx.GetCommand() == 210 ? 4 : 6];
        Result result = ResultSuccess;
        if (!m_syncpoint) {
            auto& syncpoints = m_module->GetContainer().GetSyncpointManager();
            const u32 id = syncpoints.AllocateSyncpoint(false);
            if (id == 0) {
                result = Kernel::ResultOutOfResource;
            } else {
                m_syncpoint = id;
                m_fence_value = syncpoints.ReadAllocatedSyncpoint(id).value_or(0);
            }
        }
        using Codec = Tegra::Host1x::NvdecCommon::VideoCodec;
        m_codec = codec == 0 ? Codec::H264 : codec == 1 ? Codec::VP9 : codec == 4 ? Codec::VP8 : Codec::None;
        OpenDecoder();
        LOG_INFO(Service_AM, "Video decoder codec={}, FFmpeg={}, syncpoint={}, result={:#x}", codec,
                 m_decoder.has_value(), m_syncpoint.value_or(0), result.raw);
        Reply(ctx, result);
    }

    void OpenDecoder() {
        m_decoder.reset();
        m_packet_pts.clear();
        m_next_packet = 0;
        m_drained = false;
        if (m_codec == Tegra::Host1x::NvdecCommon::VideoCodec::None) {
            return;
        }
        m_decoder.emplace();
        if (!m_decoder->Initialize(m_codec)) {
            m_decoder.reset();
            LOG_WARNING(Service_AM, "Video decoder: FFmpeg cannot open codec {}, frames are black",
                        static_cast<u64>(m_codec));
        }
    }

    // 101: one A buffer (the codec's configuration: "video/vp9" etc., or H264 parameter sets),
    // then u32 width, u32 height and six more words.
    void Configure(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        m_width = packet[5];
        m_height = packet[6];
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess.raw);
        rb.Push<u64>(0);
    }

    // 3: the event the client waits on before each submission (a copied handle).
    void GetSubmissionReadyEvent(HLERequestContext& ctx) {
        if (!m_ready_event) {
            m_ready_event = m_context.CreateEvent("VideoDecoderSubmissionReady");
        }
        IPC::ResponseBuilder rb{ctx, 2, 1, 0};
        rb.Push((m_ready_event ? ResultSuccess : Kernel::ResultOutOfResource).raw);
        ctx.AddCopyObject(m_ready_event ? &m_ready_event->GetReadableEvent() : nullptr);
    }

    void Start(HLERequestContext& ctx) {
        g_movie_timeline.store(Core::LaunchLog::MovieTraceEnabled());
        m_started = true;
        UpdateReadyEvent();
        Reply(ctx, ResultSuccess);
    }

    void Stop(HLERequestContext& ctx) {
        g_movie_timeline.store(false);
        m_started = false;
        Reply(ctx, ResultSuccess);
    }

    // 8: drop decoded frames not yet written and frames not yet described by 213; images the
    // client handed over stay with the decoder, and their fences are met by later frames.
    void Flush(HLERequestContext& ctx) {
        OpenDecoder();
        m_backlog.clear();
        m_records.clear();
        UpdateReadyEvent();
        Reply(ctx, ResultSuccess);
    }

    void Finalize(HLERequestContext& ctx) {
        m_started = false;
        m_decoder.reset();
        m_backlog.clear();
        m_records.clear();
        m_free.clear();
        // Also drop the last decoded frame and the NV12 staging copy (a full frame each).
        m_last_frame.reset();
        m_packet_pts.clear();
        m_image_buffer.clear();
        m_image_buffer.shrink_to_fit();
        Reply(ctx, ResultSuccess);
    }

    // 109: one A buffer with one container block (VP9: a frame or superframe), s64 pts in
    // microseconds, and end of stream in the low byte of word 7 (also implied by an empty block).
    void SubmitInputBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const s64 pts_us = static_cast<s64>(u64{packet[5]} | (u64{packet[6]} << 32));
        const bool end_of_stream = (packet[7] & 0xff) != 0;
        // End of stream sends an empty A buffer: reading it asserted (dozens of false
        // Critical lines per movie).
        const auto data = ctx.CanReadBuffer(0) ? ctx.ReadBuffer(0) : std::span<const u8>{};
        if (!data.empty()) {
            Decode(data, pts_us);
        }
        if (end_of_stream || data.empty()) {
            Drain();
            // The movie ends on the frame flagged end of stream: repeat the last picture for it.
            DecodedFrame last = m_last_frame ? *m_last_frame : BlackFrame(m_last_pts);
            last.end_of_stream = true;
            m_backlog.push_back(std::move(last));
        }
        PlaceFrames();
        UpdateReadyEvent();
        MovieSummary();
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess.raw);
        rb.Push<u64>(0);
    }

    // 311 registers an output image, 314 hands it back after the client has read its frame (u32
    // image handle, then a consumer fence for 314). Both reply the image's new 16-byte fence
    // {~syncpoint, 0, u64 threshold} and 16 unused bytes.
    void RegisterBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u32 image = packet[2];
        bool known{};
        {
            const std::scoped_lock lock{m_images->mutex};
            known = m_images->images.contains(image);
        }
        Result result = known && m_syncpoint ? ResultSuccess : Kernel::ResultNotFound;
        u32 threshold{};
        if (result.IsSuccess()) {
            std::erase_if(m_records, [image](const OutputRecord& record) { return record.image == image; });
            std::erase(m_free, image);
            m_free.push_back(image);
            threshold = ++m_fence_value;
            if (g_movie_timeline.load()) {
                Core::LaunchLog::Write("TL {} image {} (backlog {} free {} unread {})",
                                       ctx.GetCommand() == 314 ? "release" : "register", image,
                                       m_backlog.size(), m_free.size(), m_records.size());
            }
        }
        IPC::ResponseBuilder rb{ctx, 10};
        rb.Push(result.raw);
        rb.Push<u32>(result.IsSuccess() ? ~*m_syncpoint : 0);
        rb.Push<u32>(0);
        rb.Push<u64>(threshold);
        for (u32 i = 0; i < 4; ++i) rb.Push<u32>(0);
        if (result.IsSuccess()) {
            PlaceFrames();
            UpdateReadyEvent();
        }
    }

    void UnregisterBuffer(HLERequestContext& ctx) {
        std::erase(m_free, ctx.CommandBuffer()[2]);
        Reply(ctx, ResultSuccess);
    }

    // 213: describes the oldest written frame: image handle, width, height (twice), an unused word,
    // s64 timestamp in nanoseconds and end of stream, padded to 16 words.
    void GetOutputBufferProperties(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 18};
        if (m_records.empty()) {
            ++m_stat.not_found;
            if (Core::LaunchLog::MovieTraceEnabled() && m_stat.logged_not_found++ < 300) {
                Core::LaunchLog::Write("movie: output poll found NO decoded frame (backlog {} free images {})",
                                       m_backlog.size(), m_free.size());
            }
            rb.Push(Kernel::ResultNotFound.raw);
            for (u32 i = 0; i < 16; ++i) rb.Push<u32>(0);
            return;
        }
        const OutputRecord record = m_records.front();
        m_records.pop_front();
        if (g_movie_timeline.load()) {
            Core::LaunchLog::Write("TL frame-out image {} pts {} ms (unread left {} backlog {})", record.image,
                                   record.timestamp_ns / 1000000, m_records.size(), m_backlog.size());
        }
        rb.Push(ResultSuccess.raw);
        rb.Push(record.image);
        rb.Push(record.width);
        rb.Push(record.height);
        rb.Push(record.width);
        rb.Push(record.height);
        rb.Push<u32>(0);
        rb.Push<u64>(static_cast<u64>(record.timestamp_ns));
        rb.Push<u32>(record.end_of_stream ? 1 : 0);
        for (u32 i = 0; i < 7; ++i) rb.Push<u32>(0);
    }

    // Movie timing diagnostics (test build): one line per keyframe or slow decode, every output
    // 213 that finds no frame, every black frame from a failed packet, and a summary every 5 s.
    struct MovieStats {
        u64 submits{};
        u64 slow{};
        u64 keys{};
        u64 not_found{};
        u64 starved{};
        u64 send_fail{};
        double decode_ms_sum{};
        double decode_ms_max{};
        size_t backlog_max{};
        u64 logged_not_found{};
        u64 logged_starved{};
        std::chrono::steady_clock::time_point last_summary{std::chrono::steady_clock::now()};
    };
    MovieStats m_stat;

    void Decode(std::span<const u8> data, s64 pts_us) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t backlog_before = m_backlog.size();
        const u64 fail_before = m_stat.send_fail;
        DecodeImpl(data, pts_us);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const bool key = !data.empty() && (data[0] >> 6) == 2 && (data[0] & 0x0c) == 0;
        ++m_stat.submits;
        m_stat.decode_ms_sum += ms;
        m_stat.decode_ms_max = std::max(m_stat.decode_ms_max, ms);
        m_stat.backlog_max = std::max(m_stat.backlog_max, m_backlog.size());
        if (key) {
            ++m_stat.keys;
        }
        const bool slow = ms > 18.0;
        if (slow) {
            ++m_stat.slow;
        }
        if ((key || slow || m_stat.send_fail != fail_before) && Core::LaunchLog::MovieTraceEnabled()) {
            Core::LaunchLog::Write("movie: submit #{} {} bytes{} pts {} ms decode {:.1f} ms frames out {} "
                                   "backlog {} free images {} unread records {}{}",
                                   m_stat.submits, data.size(), key ? " KEYFRAME" : "", pts_us / 1000, ms,
                                   m_backlog.size() + 0 - backlog_before, m_backlog.size(), m_free.size(),
                                   m_records.size(), m_stat.send_fail != fail_before ? " SEND FAILED -> black frame" : "");
        }
    }

    void MovieSummary() {
        if (!Core::LaunchLog::MovieTraceEnabled()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - m_stat.last_summary < std::chrono::seconds(5)) {
            return;
        }
        m_stat.last_summary = now;
        Core::LaunchLog::Write("movie stats: submits {} keyframes {} slow(>18ms) {} decode avg {:.1f} max {:.1f} ms, "
                               "output polls with no frame {} , frames starved of images {}, send failures {}, "
                               "backlog max {}",
                               m_stat.submits, m_stat.keys, m_stat.slow,
                               m_stat.submits ? m_stat.decode_ms_sum / static_cast<double>(m_stat.submits) : 0.0,
                               m_stat.decode_ms_max, m_stat.not_found, m_stat.starved, m_stat.send_fail,
                               m_stat.backlog_max);
    }

    void DecodeImpl(std::span<const u8> data, s64 pts_us) {
        if (m_drained) {
            OpenDecoder();
        }
        if (!m_decoder || !m_decoder->SendPacket(data, {})) {
            ++m_stat.send_fail;
            m_backlog.push_back(BlackFrame(pts_us));
            m_last_pts = pts_us;
            return;
        }
        // DecodeApi numbers packets from 0 as their FFmpeg pts.
        m_packet_pts.emplace(m_next_packet++, pts_us);
        ReceiveFrames();
    }

    void Drain() {
        if (!m_decoder || m_drained) {
            return;
        }
        m_decoder->SendPacket({}, {});
        ReceiveFrames();
        m_drained = true;
    }

    void ReceiveFrames() {
        while (auto decoded = m_decoder->ReceiveFrame()) {
            const auto& frame = *decoded->frame;
            s64 pts_us = m_last_pts;
            if (const auto it = m_packet_pts.find(frame.GetFrame()->pts); it != m_packet_pts.end()) {
                pts_us = it->second;
                m_packet_pts.erase(m_packet_pts.begin(), std::next(it));
            }
            m_last_pts = pts_us;
            m_backlog.push_back(ConvertFrame(frame, pts_us));
            m_last_frame = m_backlog.back();
        }
    }

    DecodedFrame BlackFrame(s64 pts_us) const {
        const u32 width = m_width ? m_width : 16;
        const u32 height = m_height ? m_height : 16;
        DecodedFrame frame{std::vector<u8>(size_t{width} * height * 3 / 2, 0x80), width, height,
                           pts_us * 1000, false};
        std::fill_n(frame.nv12.begin(), size_t{width} * height, u8{0x10});
        return frame;
    }

    DecodedFrame ConvertFrame(const FFmpeg::Frame& frame, s64 pts_us) {
        const u32 width = static_cast<u32>(frame.GetWidth()) & ~1U;
        const u32 height = static_cast<u32>(frame.GetHeight()) & ~1U;
        DecodedFrame out{std::vector<u8>(size_t{width} * height * 3 / 2), width, height,
                         pts_us * 1000, false};
        u8* const luma = out.nv12.data();
        u8* const chroma = luma + size_t{width} * height;
        for (u32 row = 0; row < height; ++row) {
            std::memcpy(luma + size_t{row} * width, frame.GetPlane(0) + size_t{row} * frame.GetStride(0), width);
        }
        switch (frame.GetPixelFormat()) {
        case AV_PIX_FMT_NV12:
            for (u32 row = 0; row < height / 2; ++row) {
                std::memcpy(chroma + size_t{row} * width,
                            frame.GetPlane(1) + size_t{row} * frame.GetStride(1), width);
            }
            break;
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            for (u32 row = 0; row < height / 2; ++row) {
                const u8* u = frame.GetPlane(1) + size_t{row} * frame.GetStride(1);
                const u8* v = frame.GetPlane(2) + size_t{row} * frame.GetStride(2);
                u8* line = chroma + size_t{row} * width;
                for (u32 column = 0; column < width / 2; ++column) {
                    line[column * 2] = u[column];
                    line[column * 2 + 1] = v[column];
                }
            }
            break;
        default:
            LOG_WARNING(Service_AM, "Video decoder: unsupported frame format {}",
                        static_cast<int>(frame.GetPixelFormat()));
            std::fill_n(luma, size_t{width} * height, u8{0x10});
            std::fill_n(chroma, size_t{width} * height / 2, u8{0x80});
            break;
        }
        return out;
    }

    // Writes backlogged frames into handed-over images, oldest first, meeting one fence each.
    void PlaceFrames() {
        auto& host_syncpoints = system.Host1x().GetSyncpointManager();
        while (!m_backlog.empty() && !m_free.empty() && m_syncpoint) {
            const u32 image = m_free.front();
            m_free.pop_front();
            const DecodedFrame& frame = m_backlog.front();
            WriteImage(image, frame);
            m_records.push_back({image, frame.width, frame.height, frame.timestamp_ns, frame.end_of_stream});
            m_backlog.pop_front();
            host_syncpoints.IncrementGuest(*m_syncpoint);
            host_syncpoints.IncrementHost(*m_syncpoint);
        }
        if (!m_backlog.empty() && m_free.empty()) {
            ++m_stat.starved;
            if (Core::LaunchLog::MovieTraceEnabled() && m_stat.logged_starved++ < 300) {
                Core::LaunchLog::Write("movie: {} decoded frame(s) waiting, the game has no free output image",
                                       m_backlog.size());
            }
        }
    }

    void WriteImage(u32 handle, const DecodedFrame& frame) {
        DecoderImageRegistry::Image image{};
        {
            const std::scoped_lock lock{m_images->mutex};
            const auto it = m_images->images.find(handle);
            if (it == m_images->images.end()) {
                LOG_ERROR(Service_AM, "Video decoder: output image {} is gone", handle);
                return;
            }
            image = it->second;
        }
        const size_t luma_size = size_t{image.pitch} * image.aligned_height;
        m_image_buffer.assign(luma_size * 3 / 2, u8{0x80});
        std::fill_n(m_image_buffer.begin(), luma_size, u8{0x10});
        const u32 width = std::min(frame.width, image.width);
        const u32 height = std::min(frame.height, image.height);
        const u8* luma = frame.nv12.data();
        const u8* chroma = luma + size_t{frame.width} * frame.height;
        for (u32 row = 0; row < height; ++row) {
            std::memcpy(m_image_buffer.data() + size_t{row} * image.pitch,
                        luma + size_t{row} * frame.width, width);
        }
        for (u32 row = 0; row < height / 2; ++row) {
            std::memcpy(m_image_buffer.data() + luma_size + size_t{row} * image.pitch,
                        chroma + size_t{row} * frame.width, width);
        }
        m_process->GetMemory().WriteBlock(image.address, m_image_buffer.data(), m_image_buffer.size());
    }

    // The client waits for this before every submission; hold it while frames wait for images.
    void UpdateReadyEvent() {
        if (!m_ready_event) {
            return;
        }
        if (m_started && m_backlog.size() < 4) {
            m_ready_event->Signal(kernel);
        } else {
            m_ready_event->Clear(kernel);
        }
    }

    std::shared_ptr<Nvidia::Module> m_module;
    Kernel::KProcess* const m_process;
    std::shared_ptr<DecoderImageRegistry> m_images;
    KernelHelpers::ServiceContext m_context;
    Kernel::KEvent* m_ready_event{};
    std::optional<u32> m_syncpoint;
    u32 m_fence_value{};
    Tegra::Host1x::NvdecCommon::VideoCodec m_codec{};
    std::optional<FFmpeg::DecodeApi> m_decoder;
    std::map<s64, s64> m_packet_pts;
    s64 m_next_packet{};
    bool m_drained{};
    bool m_started{};
    u32 m_width{};
    u32 m_height{};
    s64 m_last_pts{};
    std::optional<DecodedFrame> m_last_frame;
    std::deque<DecodedFrame> m_backlog;
    std::deque<OutputRecord> m_records;
    std::deque<u32> m_free;
    std::vector<u8> m_image_buffer;
};

// The decoder manager session (supervisor 902/904/905): command 1 creates a decoder session.
class IVideoDecoderManager final : public ServiceFramework<IVideoDecoderManager> {
public:
    IVideoDecoderManager(Core::System& system, std::shared_ptr<Nvidia::Module> module,
                         Kernel::KProcess* process, std::shared_ptr<DecoderImageRegistry> images)
        : ServiceFramework{system, "IVideoDecoderManager"}, m_module{std::move(module)},
          m_process{process}, m_images{std::move(images)} {
        m_process->Open(kernel);
        static const FunctionInfo functions[] = {
            {1, &IVideoDecoderManager::CreateDecoderSession, "CreateVideoDecoderService"},
        };
        RegisterHandlersTipc(functions);
    }

    ~IVideoDecoderManager() override {
        m_process->Close(kernel);
    }

private:
    void CreateDecoderSession(HLERequestContext& ctx) {
        Kernel::KClientSession* client{};
        const Result result = m_module
            ? CreateGraphicsSession(system, ctx,
                  std::make_shared<IVideoDecoderSession>(system, m_module, m_process, m_images), &client)
            : Kernel::ResultInvalidState;
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(client);
    }

    std::shared_ptr<Nvidia::Module> m_module;
    Kernel::KProcess* const m_process;
    std::shared_ptr<DecoderImageRegistry> m_images;
};

// The encoder manager session (supervisor 1002/1004). nn::codec::video::Initialize opens it, but
// only the H264 encoder (GameChat, movie recording) creates encoders, which Eden does not have.
class IVideoEncoderManager final : public ServiceFramework<IVideoEncoderManager> {
public:
    explicit IVideoEncoderManager(Core::System& system) : ServiceFramework{system, "IVideoEncoderManager"} {
        static const FunctionInfo functions[] = {
            {1, &IVideoEncoderManager::CreateEncoderSession, "CreateVideoEncoderService"},
        };
        RegisterHandlersTipc(functions);
    }

private:
    void CreateEncoderSession(HLERequestContext& ctx) {
        LOG_WARNING(Service_AM, "(STUBBED) video encoder requested");
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(Kernel::ResultNotImplemented.raw);
        ctx.AddMoveObject(nullptr);
    }
};

// The session supervisor command 1202 hands out for GameChat (nn::chat initializes it at boot,
// e.g. Hyrule Warriors: Age of Imprisonment). Eden 2 has no chat: command 1 reports a count of 0,
// command 11 gives an event that never fires, and other commands succeed with no data.
class IChatGraphicsSession final : public ServiceFramework<IChatGraphicsSession> {
public:
    explicit IChatGraphicsSession(Core::System& system)
        : ServiceFramework{system, "IChatGraphicsSession"}, m_context{system, "ChatGraphicsSession"} {
        static const FunctionInfo functions[] = {
            {1, &IChatGraphicsSession::GetCount, "GetCount"},
            {11, &IChatGraphicsSession::GetEvent, "GetEvent"},
            {115, &IChatGraphicsSession::RequestReceiveImage, "RequestReceiveImage"},
            {231, &IChatGraphicsSession::GetMetadata, "GetMetadata"},
        };
        RegisterHandlersTipc(functions);
    }

    ~IChatGraphicsSession() override {
        m_context.CloseEvent(m_event);
    }

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = LockService();
        LOG_WARNING(Service_AM, "(STUBBED) chat graphics command={}", ctx.GetCommand());
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess.raw);
        return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }

private:
    void GetCount(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess.raw);
        rb.Push<u32>(0);
    }

    // Nothing is ever received: an all-zero 16-byte NativeSyncHandle means "no fence", which the
    // SDK waits on as already signalled.
    void RequestReceiveImage(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(ResultSuccess.raw);
        for (u32 i = 0; i < 4; ++i) rb.Push<u32>(0);
    }

    // No metadata: both flags clear.
    void GetMetadata(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess.raw);
        rb.Push<u64>(0);
    }

    void GetEvent(HLERequestContext& ctx) {
        if (!m_event) {
            m_event = m_context.CreateEvent("ChatGraphics");
        }
        Kernel::KReadableEvent* readable{};
        if (m_event) {
            readable = &m_event->GetReadableEvent();
            readable->Open(kernel);
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push((m_event ? ResultSuccess : Kernel::ResultOutOfResource).raw);
        ctx.AddMoveObject(readable);
    }

    KernelHelpers::ServiceContext m_context;
    Kernel::KEvent* m_event{};
};

class IVi2Session final : public ServiceFramework<IVi2Session> {
    struct ClientState {
        u64 token{};
        u32 slots{};
        std::optional<u64> display;
        std::optional<u64> layer;
        u64 available{0x7fff};
        u32 retire_notification{};
        u64 output_encoding{0x14};
        u32 available_notification{};
        std::shared_ptr<android::BufferQueueProducer> producer;
        std::array<std::optional<s32>, 15> queued{};
        std::optional<size_t> player_display;
    };

public:
    explicit IVi2Session(Core::System& system, u64 aruid, std::shared_ptr<NativeImageRegistry> images) : ServiceFramework{system, "IVi2Session"}, m_images{std::move(images)}, m_aruid{aruid}, m_context{system, "Vi2Session"} {
        static const FunctionInfo functions[] = {
            {1, &IVi2Session::GetSessionLimit, "GetNotificationLimit"},
            {11, &IVi2Session::GetNotificationEvent, "GetNotificationEvent"},
            {13, &IVi2Session::ReleaseNotificationEvent, "ReleaseNotificationEvent"},
            {21, &IVi2Session::GetNotifications, "GetNotifications"},
            {3, &IVi2Session::GetSessionLimit, "GetSessionLimit"},
            {5, &IVi2Session::GetSessionLimit, "GetRequestGroupLimit"},
            {51, &IVi2Session::ConfigureRequestGroups, "ConfigureRequestGroups"},
            {101, &IVi2Session::OpenDisplay, "OpenDisplay"},
            {102, &IVi2Session::CloseDisplay, "CloseDisplay"},
            {111, &IVi2Session::OpenEndpoint, "OpenEndpoint"},
            {112, &IVi2Session::CloseEndpoint, "CloseEndpoint"},
            {201, &IVi2Session::RegisterClientState, "RegisterClientState"},
            {202, &IVi2Session::UnregisterClientState, "UnregisterClientState"},
            {212, &IVi2Session::GetClientAvailability, "GetClientAvailability"},
            {229, &IVi2Session::PresentClientBuffer, "PresentClientBuffer"},
            {228, &IVi2Session::CancelClientBuffer, "CancelClientBuffer"},
            {351, &IVi2Session::SetClientOutputEncoding, "SetClientOutputEncoding"},
            {251, &IVi2Session::BindClientDisplay, "BindClientDisplay"},
            {252, &IVi2Session::UnbindClientDisplay, "UnbindClientDisplay"},
            {301, &IVi2Session::Unknown301, "SetDisplayRefreshRateCapability"},
            {31, &IVi2Session::GetStatusEvent, "GetStatusEvent"},
            {32, &IVi2Session::DuplicateEvent, "DuplicateEvent"},
            {53, &IVi2Session::ReadTimestamps, "ReadTimestamps"},
            {103, &IVi2Session::QueryDisplayProperty, "QueryDisplayRefreshRate"},
            {104, &IVi2Session::QueryDisplayProperty, "QueryDisplayHdrStatus"},
            {105, &IVi2Session::QueryDisplayProperty, "QueryDisplayHdrPaperWhite"},
            {106, &IVi2Session::QueryDisplayProperty, "QueryDisplayBufferMaxResolution"},
            {107, &IVi2Session::QueryDisplayProperty, "QueryDisplayModeCapability"},
            {113, &IVi2Session::AcknowledgeDisplaySetting, "Unknown113"},
            {261, &IVi2Session::SetDisplayNotification, "RegisterVsyncNotification"},
            {262, &IVi2Session::SetDisplayNotification, "UnregisterVsyncNotification"},
            {263, &IVi2Session::SetDisplayNotification, "RegisterRefreshRateNotification"},
            {264, &IVi2Session::SetDisplayNotification, "UnregisterRefreshRateNotification"},
            {265, &IVi2Session::SetDisplayNotification, "RegisterHdrStatusNotification"},
            {266, &IVi2Session::SetDisplayNotification, "UnregisterHdrStatusNotification"},
            {267, &IVi2Session::SetDisplayNotification, "RegisterHdrPaperWhiteNotification"},
            {268, &IVi2Session::SetDisplayNotification, "UnregisterHdrPaperWhiteNotification"},
            {269, &IVi2Session::SetDisplayNotification, "RegisterBufferMaxResolutionNotification"},
            {270, &IVi2Session::SetDisplayNotification, "UnregisterBufferMaxResolutionNotification"},
            {271, &IVi2Session::SetDisplayNotification, "RegisterModeCapabilityNotification"},
            {272, &IVi2Session::SetDisplayNotification, "UnregisterModeCapabilityNotification"},
            {302, &IVi2Session::AcknowledgeDisplaySetting, "SetDisplayHdrOutputFormatCapability"},
            {352, &IVi2Session::AcknowledgeDisplaySetting, "SetClientHdrToneMappingMetadata"},
            {353, &IVi2Session::AcknowledgeDisplaySetting, "ClearClientHdrToneMappingMetadata"},
        };
        RegisterHandlersTipc(functions);
    }

    ~IVi2Session() override {
        if (m_release_poll) system.CoreTiming().UnscheduleEvent(m_release_poll);
        for (u32 slot = 0; slot < m_vsync_displays.size(); ++slot) {
            if (m_vsync_displays[slot]) {
                m_display_service->GetContainer()->UnlinkVsyncEvent(*m_vsync_displays[slot], &*m_slot_events[slot]);
            }
        }
        if (m_display_service) {
            for (auto& state : m_client_states) {
                if (state.layer) m_display_service->GetContainer()->DestroyManagedLayer(*state.layer);
            }
        }
        if (m_default_display) { m_display_service->CloseDisplay(*m_default_display); }
        m_context.CloseEvent(m_event);
        m_context.CloseEvent(m_status_event);
    }

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = LockService();
        LOG_DEBUG(Service_AM, "VI2 unsupported command={}, words={:08x}", ctx.GetCommand(),
                  fmt::join(std::span<const u32>(ctx.CommandBuffer() + 2, std::min<u32>(ctx.CommandBuffer()[1], 32)), ","));
        InvokeRequestTipc(ctx);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Kernel::ResultNotImplemented.raw);
        return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }
private:
    static std::optional<size_t> GetPlayerDisplayIndex(u64 name) {
        if ((name & 0xffffffffffffULL) != 0x726579616c70ULL || (name >> 56) != 0) {
            return std::nullopt;
        }
        const u32 player = (name >> 48) & 0xff;
        if (player < '1' || player > '3') return std::nullopt;
        return player - '1';
    }
    Result OpenDefaultDisplay(u64 name, u32 flags) {
        if (const auto player = GetPlayerDisplayIndex(name)) {
            R_UNLESS(flags < m_notifications.size() * 32, Kernel::ResultInvalidCombination);
            R_UNLESS(m_player_display_references[*player] < 1024, Kernel::ResultOutOfResource);
            // GameShare's detached player displays retain their own lifetime;
            // they do not target the host's default display compositor.
            m_player_display_notifications[*player] = flags;
            ++m_player_display_references[*player];
            R_SUCCEED();
        }
        // The native display token is an eight-byte name; command 101 returns
        // only a result. Retain the real VI display ID for later operations.
        R_UNLESS(name == 0x00746c7561666564ULL, Kernel::ResultNotFound);
        // The third word is the notification slot signaled when the display retires (Pokopia
        // uses 1, Hyrule Warriors: Age of Imprisonment 4), as for player displays above.
        R_UNLESS(flags < m_notifications.size() * 32, Kernel::ResultInvalidCombination);
        R_UNLESS(m_display_references < 1024, Kernel::ResultOutOfResource);
        if (!m_display_service) {
            const auto root = system.ServiceManager().GetService<VI::IManagerRootService>("vi:m", true);
            R_UNLESS(root, Kernel::ResultInvalidState);
            R_TRY(root->GetDisplayService(&m_display_service, VI::Policy::Compositor));
        }
        if (!m_default_display) {
            u64 id{};
            R_TRY(m_display_service->OpenDefaultDisplay(&id));
            m_default_display = id;
        }
        m_display_notification = flags;
        ++m_display_references;
        R_SUCCEED();
    }
    void OpenDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const auto result = packet[0] == 0x75 && packet[1] == 3
            ? OpenDefaultDisplay(name, packet[4]) : Kernel::ResultInvalidCombination;
        LOG_DEBUG(Service_AM, "VI2 open display name={:#x}, flags={:#x}, references={}, result={:#x}",
                  name, packet[4], m_display_references, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void CloseDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        if (packet[0] == 0x76 && packet[1] == 2 && GetPlayerDisplayIndex(name)) {
            const auto player = *GetPlayerDisplayIndex(name);
            result = Kernel::ResultNotFound;
            if (m_player_display_references[player]) {
                if (--m_player_display_references[player] == 0) {
                    const auto notification = m_player_display_notifications[player];
                    m_notifications[notification / 32] |= 1U << (notification % 32);
                    if (m_event) m_event->Signal(kernel);
                }
                result = ResultSuccess;
            }
        } else if (packet[0] == 0x76 && packet[1] == 2 && name == 0x00746c7561666564ULL) {
            result = Kernel::ResultNotFound;
            if (m_default_display && m_display_references) {
                result = ResultSuccess;
                if (m_display_references == 1) {
                    result = m_display_service->CloseDisplay(*m_default_display);
                    if (result.IsSuccess()) { m_default_display.reset(); }
                }
                if (result.IsSuccess()) {
                    --m_display_references;
                    if (!m_display_references) {
                        m_notifications[m_display_notification / 32] |= 1U << (m_display_notification % 32);
                        if (m_event) m_event->Signal(kernel);
                        LOG_DEBUG(Service_AM, "VI2 display retired notification={}", m_display_notification);
                    }
                }
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void RegisterClientState(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u32 first = packet[4];
        const u32 second = packet[5];
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xd9 && packet[1] == 4 && token != 0 &&
            first < 32 && second < 32 && first != second) {
            const u32 slots = (1U << first) | (1U << second);
            const auto existing = std::ranges::find(m_client_states, token, &ClientState::token);
            const auto free = std::ranges::find(m_client_states, u64{0}, &ClientState::token);
            result = Kernel::ResultInvalidState;
            if (existing == m_client_states.end() && !(m_client_slots & slots)) {
                result = Kernel::ResultOutOfResource;
                if (free != m_client_states.end()) {
                    // The SDK sends its object address as an opaque identity, not a
                    // service-accessible buffer. Keep the associated slots session-local.
                    *free = {token, slots};
                    free->retire_notification = first;
                    free->available_notification = second;
                    m_client_slots |= slots;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 register client state token={:#x}, slots={}/{}, result={:#x}",
                  token, first, second, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void UnregisterClientState(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xda && packet[1] == 2 && token != 0) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (state != m_client_states.end()) {
                if (m_default_endpoint == token) { m_default_endpoint.reset(); }
                if (state->layer) m_display_service->GetContainer()->DestroyManagedLayer(*state->layer);
                m_notifications[state->retire_notification / 32] |= 1U << (state->retire_notification % 32);
                if (m_event) m_event->Signal(kernel);
                LOG_DEBUG(Service_AM, "VI2 client retired token={:#x}, notification={}", token, state->retire_notification);
                m_client_slots &= ~state->slots;
                *state = {};
                result = ResultSuccess;
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void GetClientAvailability(HLERequestContext& ctx) {
        RefreshReleasedBuffers();
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        u64 id{};
        if (packet[0] == 0xe4 && packet[1] == 2 && token) {
            auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (state != m_client_states.end() && m_default_display) {
                result = ResultSuccess;
                if (state->player_display) {
                    id = 0;
                } else if (!state->layer) {
                    result = m_display_service->GetContainer()->CreateManagedLayer(&id, *m_default_display, m_aruid);
                    if (result.IsSuccess()) state->layer = id;
                } else {
                    id = *state->layer;
                }
            }
        }
        const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
        const u64 available = result.IsSuccess() ? state->available : 0;
        LOG_DEBUG(Service_AM, "VI2 client availability token={:#x}, layer={}, mask={:#x}, result={:#x}", token, id, available, result.raw);
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(result.raw);
        rb.Push(available);
    }
    void CancelClientBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u32 slot = GraphicsSupervisorProtocol::DecodeBufferSlot(packet[4]);
        Result result = Kernel::ResultInvalidCombination;
        u64 available{};
        if (packet[0] == 0xf4 && packet[1] == 19 && slot < 15) {
            auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end()) {
                // Unlike present (229), cancel has no image or input GPU fence.
                state->available |= u64{1} << slot;
                available = state->available;
                result = ResultSuccess;
            }
        }
        LOG_DEBUG(Service_AM, "VI2 cancel buffer token={:#x}, slot={}, mask={:#x}, result={:#x}", token, slot, available, result.raw);
        IPC::ResponseBuilder rb{ctx, 8};
        rb.Push(result.raw);
        rb.Push(available);
        for (u32 i = 0; i < 4; ++i) rb.Push<u32>(0);
    }
    void RefreshReleasedBuffers() {
        for (auto& state : m_client_states) {
            if (!state.producer || !m_images->module) continue;
            bool changed{};
            for (size_t slot = 0; slot < state.queued.size(); ++slot) {
                if (!state.queued[slot]) continue;
                const auto fence = state.producer->GetReleasedBufferFence(*state.queued[slot]);
                if (!fence) continue;
                bool complete = true;
                for (u32 i = 0; i < fence->num_fences; ++i) {
                    const auto& point = fence->fences[i];
                    if (point.id != ~u32{0}) {
                        const auto counter = m_images->module->GetContainer().GetSyncpointManager().ReadAllocatedSyncpoint(point.id);
                        if (!counter || static_cast<s32>(*counter - point.value) < 0) {
                            complete = false;
                        }
                    }
                }
                if (!complete) continue;
                state.queued[slot].reset();
                state.available |= u64{1} << slot;
                LOG_DEBUG(Service_AM, "Native display buffer retired: slot={}, mask={:#x}", slot, state.available);
                changed = true;
            }
            if (changed) {
                m_notifications[state.available_notification / 32] |= 1U << (state.available_notification % 32);
                if (m_event) m_event->Signal(kernel);
            }
        }
    }
    Result QueueNativeImage(ClientState& state, const u32* packet) {
        const u32 slot = GraphicsSupervisorProtocol::DecodeBufferSlot(packet[4]);
        R_UNLESS(!state.player_display, Kernel::ResultNotImplemented);
        R_UNLESS(state.layer && state.display && slot < 15 && !state.queued[slot], Kernel::ResultInvalidState);
        const std::scoped_lock images_lock{m_images->mutex};
        const auto image = std::ranges::find(m_images->images, packet[5], &NativeImage::id);
        R_UNLESS(image != m_images->images.end() && image->window_pool && m_images->module, Kernel::ResultNotFound);
        // The observed presentation is a four-byte color image with the
        // backend's 16-GOB block height and no byte offset within its pool. The presented size
        // (word 7, like the image's preference word 2: width - 1 low, height - 1 high) may be a
        // top-left region of the image: Hyrule Warriors: Age of Imprisonment shows 1920x1080
        // of a 3840x2160 image, so the frame is cropped to it.
        const s32 present_width = static_cast<s32>((packet[7] & 0xffffU) + 1);
        const s32 present_height = static_cast<s32>((packet[7] >> 16) + 1);
        const s32 image_width = static_cast<s32>((image->preference[2] & 0xffffU) + 1);
        const s32 image_height = static_cast<s32>((image->preference[2] >> 16) + 1);
        if (!(packet[6] == 0 && present_width <= image_width && present_height <= image_height &&
              packet[8] == 3 && image->preference[4] == 7)) {
            LOG_WARNING(Service_AM,
                        "VI2 unsupported presentation: offset={:#x}, size={:#x}, layout={}, "
                        "image preference={:#x}, encoding={:#x}",
                        packet[6], packet[7], packet[8], fmt::join(image->preference, ","),
                        state.output_encoding);
            R_THROW(Kernel::ResultNotImplemented);
        }
        const u32 fence_id = ~packet[22];
        R_UNLESS(packet[23] == 0 && packet[25] == 0 &&
                 (fence_id == ~u32{0} || m_images->module->GetContainer().GetSyncpointManager().IsSyncpointAllocated(fence_id)),
                 Kernel::ResultInvalidCombination);
        if (!state.producer) {
            R_TRY(m_display_service->GetContainer()->GetLayerProducerHandle(&state.producer, *state.layer));
            android::QueueBufferOutput output{};
            R_UNLESS(state.producer->Connect({}, android::NativeWindowApi::Egl, false, &output) == android::Status::NoError,
                     Kernel::ResultInvalidState);
            R_UNLESS(state.producer->SetBufferCount(16) == android::Status::NoError, Kernel::ResultInvalidState);
            s32 binder{};
            R_TRY(m_display_service->GetContainer()->OpenLayer(&binder, *state.layer, m_aruid));
            R_TRY(m_display_service->GetContainer()->SetLayerVisibility(*state.layer, true));
        }
        auto native = std::make_shared<android::NvGraphicBuffer>();
        native->width = image_width;
        native->height = image_height;
        native->stride = static_cast<s32>(image->preference[5] * 16);
        native->format = android::PixelFormat::Rgba8888;
        native->external_format = native->format;
        native->buffer_id = image->memory_id;
        auto buffer = std::make_shared<android::GraphicBuffer>(m_images->module->GetContainer().GetNvMapFile(), native);
        s32 queue_slot{};
        R_UNLESS(state.producer->AttachBuffer(&queue_slot, buffer) == android::Status::NoError, Kernel::ResultOutOfResource);
        android::QueueBufferInput input{};
        android::QueueBufferOutput output{};
        input.crop = {0, 0, present_width, present_height};
        input.swap_interval = 1;
        input.fence = android::Fence::NoFence();
        if (fence_id != ~u32{0}) {
            input.fence.num_fences = 1;
            input.fence.fences[0] = {static_cast<s32>(fence_id), packet[24]};
        }
        if (state.producer->QueueBuffer(queue_slot, input, &output) != android::Status::NoError) {
            state.producer->CancelBuffer(queue_slot, input.fence);
            return Kernel::ResultInvalidState;
        }
        state.queued[slot] = queue_slot;
        state.available &= ~(u64{1} << slot);
        if (!m_release_poll) {
            m_release_poll = Core::Timing::CreateEvent("NativeViBufferRelease", [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
                const auto guard = LockService();
                RefreshReleasedBuffers();
                return std::nullopt;
            });
            system.CoreTiming().ScheduleLoopingEvent(std::chrono::milliseconds{1}, std::chrono::milliseconds{1}, m_release_poll);
        }
        LOG_DEBUG(Service_AM, "Native image queued for display: image={}, slot={}, fence={}/{}, dimensions={}x{}",
                 image->id, slot, fence_id, packet[24], native->width, native->height);
        R_SUCCEED();
    }
    void PresentClientBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        RefreshReleasedBuffers();
        Result result = Kernel::ResultInvalidCombination;
        u64 available{};
        if (packet[0] == 0xf5 && packet[1] == 24) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end()) {
                result = QueueNativeImage(*state, packet);
                if (g_movie_timeline.load()) {
                    Core::LaunchLog::Write("TL present image {}", packet[5]);
                }
                available = state->available;
            }
        }
        LOG_DEBUG(Service_AM, "VI2 present token={:#x}, image={}, slot={}, mask={:#x}, result={:#x}", token, packet[5], packet[4], available, result.raw);
        IPC::ResponseBuilder rb{ctx, 8};
        rb.Push(result.raw);
        rb.Push(available);
        for (u32 i = 0; i < 4; ++i) rb.Push<u32>(0);
    }
    void SetClientOutputEncoding(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 encoding = u64{packet[4]} | (u64{packet[5]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x16f && packet[1] == 4) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end()) {
                result = Kernel::ResultNotImplemented;
                // These are the client's timestamp checkpoint bits (nn::vi's frame history always
                // sends 0x14); Eden records no timestamps, so any of the five bits is accepted.
                if (encoding <= 0x1f) {
                    state->output_encoding = encoding;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 output encoding token={:#x}, encoding={:#x}, result={:#x}", token, encoding, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void BindClientDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 name = u64{packet[4]} | (u64{packet[5]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x10b && packet[1] == 4) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end() && GetPlayerDisplayIndex(name)) {
                const auto player = *GetPlayerDisplayIndex(name);
                result = Kernel::ResultInvalidState;
                if (m_player_display_references[player] && !state->display) {
                    result = ResultSuccess;
                    if (state->layer) {
                        result = m_display_service->GetContainer()->DestroyManagedLayer(*state->layer);
                    }
                    if (result.IsSuccess()) {
                        state->layer.reset();
                        state->display = name;
                        state->player_display = player;
                    }
                }
            } else if (token && state != m_client_states.end() && name == 0x00746c7561666564ULL &&
                m_default_display && m_display_references) {
                result = Kernel::ResultInvalidState;
                if (!state->display) {
                    state->display = m_default_display;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 bind client token={:#x}, display={:#x}, result={:#x}",
                  token, name, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void Unknown301(HLERequestContext& ctx) {
        // nn::vi::SetDisplayVrrCapability, sent every frame by some games; Eden's display is fixed-rate.
        LOG_DEBUG(Service_AM, "VI2 refresh rate capability header={:#x}", ctx.CommandBuffer()[0]);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }
    void UnbindClientDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x10c && packet[1] == 2) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end() && state->display) {
                state->display.reset();
                state->player_display.reset();
                result = ResultSuccess;
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void OpenEndpoint(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 display_name = u64{packet[4]} | (u64{packet[5]} << 32);
        const u64 token = u64{packet[6]} | (u64{packet[7]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x7f && packet[1] == 6) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end() && state->display &&
                m_default_display == state->display && name == 0x00746c7561666564ULL &&
                display_name == name) {
                result = Kernel::ResultInvalidState;
                if (!m_default_endpoint) {
                    m_default_endpoint = token;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 open endpoint name={:#x}, display={:#x}, client={:#x}, result={:#x}",
                  name, display_name, token, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void CloseEndpoint(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 display_name = u64{packet[4]} | (u64{packet[5]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x80 && packet[1] == 4 && name == 0x00746c7561666564ULL &&
            display_name == name) {
            result = Kernel::ResultNotFound;
            if (m_default_endpoint) {
                m_default_endpoint.reset();
                result = ResultSuccess;
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void ConfigureRequestGroups(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x43 && packet[1] == 1 && packet[2] < 8) {
            m_request_group_limit = packet[2];
            m_request_groups_initialized = true;
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void GetSessionLimit(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const Result result = packet[0] == 16 + ctx.GetCommand() && packet[1] == 0
                                  ? ResultSuccess : Kernel::ResultInvalidCombination;
        // The SDK converts this inclusive upper group index into 32 client slots.
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result.raw);
        rb.Push<u32>(ctx.GetCommand() == 5 ? 7 : 0);
    }

    void GetNotifications(HLERequestContext& ctx) {
        Result result = Kernel::ResultInvalidSize;
        if (ctx.GetWriteBufferSize() >= sizeof(m_notifications)) {
            ctx.WriteBuffer(m_notifications);
            LOG_DEBUG(Service_AM, "VI2 notifications={:08x}", fmt::join(m_notifications, ","));
            m_notifications.fill(0);
            if (m_event) m_event->Clear(kernel);
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void ReleaseNotificationEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const auto result = packet[0] == 0x1d && packet[1] == 1 && packet[2] == 0
            ? ResultSuccess : Kernel::ResultInvalidCombination;
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void GetNotificationEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Kernel::KReadableEvent* readable{};
        if (packet[0] == 0x1b && packet[1] == 1 && packet[2] == 0) {
            if (!m_event) {
                m_event = m_context.CreateEvent("Vi2Notification");
            }
            result = m_event ? ResultSuccess : Kernel::ResultOutOfResource;
            if (m_event) {
                if (std::ranges::any_of(m_notifications, [](u32 value) { return value != 0; })) m_event->Signal(kernel);
                readable = &m_event->GetReadableEvent();
                readable->Open(kernel);
            }
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(readable);
    }
    // Command 31 (slot) hands out the event of one nn::vi2 status-event slot (stub nnSdk+0x5e39d0,
    // OoT: header 0x2f, slot in word 2). The SDK's allocator (nnSdk+0x5eac80) takes slots from one
    // 64-bit bitmap, caches each slot's handle and reuses it for whichever display property
    // 261-272 binds to the slot next, so every slot keeps its own event. Used by Hyrule Warriors:
    // Age of Imprisonment (properties only) and OoT (vsync, see SetDisplayNotification).
    void GetStatusEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Kernel::KReadableEvent* readable{};
        if (packet[0] == 0x2f && packet[1] == 1 && packet[2] < m_slot_events.size()) {
            readable = GetSlotEvent(packet[2]).GetHandle();
        } else {
            if (!m_status_event) {
                m_status_event = m_context.CreateEvent("Vi2StatusEvent");
            }
            if (m_status_event) {
                readable = &m_status_event->GetReadableEvent();
            }
        }
        if (readable) {
            readable->Open(kernel);
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push((readable ? ResultSuccess : Kernel::ResultOutOfResource).raw);
        ctx.AddMoveObject(readable);
    }
    Event& GetSlotEvent(u32 slot) {
        if (!m_slot_events[slot]) {
            m_slot_events[slot].emplace(m_context);
        }
        return *m_slot_events[slot];
    }
    // Commands 261-272 register (odd) and unregister (even) a display property notification:
    // vsync, refresh rate, HDR status, HDR paper white, maximum buffer resolution and mode
    // capability, each as (display name, event slot). Eden's display properties never change, so
    // those are never signalled; the SDK seeds its own defaults (60 Hz, SDR) when a display opens.
    // Vsync does fire: OoT's frame loop waits untimed on the event it got for its vsync slot
    // (nn::vi::GetDisplayVsyncEvent at Owl.nss+0x346bf54 -> 31, 261, 32; WaitSystemEvent at
    // Owl.nss+0x346c93c), so 261 links the slot to the VI display's vsync, as vi:u does. An
    // unregister failure aborts the SDK (nn::vi::CloseDisplay sends 264, 262 aborts at
    // nnSdk+0x5e486c), so these always succeed.
    void SetDisplayNotification(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const u32 slot = packet[4];
        LOG_DEBUG(Service_AM, "VI2 display notification command={}, display={:#x}, slot={}",
                  ctx.GetCommand(), name, slot);
        // 261/262 (stubs nnSdk+0x5e4740/0x5e47e0): header 0x115/0x116, 3 words, slot in word 4.
        if (packet[1] == 3 && slot < m_slot_events.size()) {
            if (ctx.GetCommand() == 261 && packet[0] == 0x115) {
                UnlinkVsyncSlot(slot);
                if (name == 0x00746c7561666564ULL && m_default_display) {
                    m_display_service->GetContainer()->LinkVsyncEvent(*m_default_display, &GetSlotEvent(slot));
                    m_vsync_displays[slot] = *m_default_display;
                }
            } else if (ctx.GetCommand() == 262 && packet[0] == 0x116) {
                UnlinkVsyncSlot(slot);
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess.raw);
    }
    void UnlinkVsyncSlot(u32 slot) {
        if (!m_vsync_displays[slot]) {
            return;
        }
        m_display_service->GetContainer()->UnlinkVsyncEvent(*m_vsync_displays[slot], &*m_slot_events[slot]);
        m_vsync_displays[slot].reset();
        // The slot and its event are reused for a property that has not changed.
        m_slot_events[slot]->Clear(kernel);
    }
    // Commands 103-107 query a display property (display name): refresh rate code 1 = 60 Hz, HDR
    // status 1 = SDR, paper white 0, maximum buffer resolution code 1 (the SDK's default) and the
    // mode capability bitsets {60 Hz, SDR, 60 Hz}. Zero codes are errors in the SDK.
    void QueryDisplayProperty(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const bool valid = packet[0] == 16 + ctx.GetCommand() && packet[1] == 2 &&
                           (name == 0x00746c7561666564ULL || GetPlayerDisplayIndex(name));
        const Result result = valid ? ResultSuccess : Kernel::ResultNotFound;
        switch (ctx.GetCommand()) {
        case 103:
        case 106: {
            IPC::ResponseBuilder rb{ctx, 4};
            rb.Push(result.raw);
            rb.Push<u64>(1);
            break;
        }
        case 104: {
            IPC::ResponseBuilder rb{ctx, 5};
            rb.Push(result.raw);
            rb.Push<u64>(1);
            rb.Push<u32>(0);
            break;
        }
        case 105: {
            IPC::ResponseBuilder rb{ctx, 3};
            rb.Push(result.raw);
            rb.Push<u32>(0);
            break;
        }
        default: {
            IPC::ResponseBuilder rb{ctx, 8};
            rb.Push(result.raw);
            rb.Push<u64>(1);
            rb.Push<u64>(1);
            rb.Push<u64>(1);
            break;
        }
        }
    }
    // Command 32 (slot, copy handle of an event from command 31) returns the same event again as a
    // moved handle.
    void DuplicateEvent(HLERequestContext& ctx) {
        Kernel::KReadableEvent* readable{};
        auto event = ctx.GetObjectFromHandle<Kernel::KReadableEvent>(ctx.GetCopyHandle(0));
        if (event.IsNotNull()) {
            readable = event.GetPointerUnsafe();
            readable->Open(kernel);
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push((readable ? ResultSuccess : Kernel::ResultInvalidHandle).raw);
        ctx.AddMoveObject(readable);
    }
    // Command 53 reads a timestamp recorder (nn::vi destroys its eight on every layer close and
    // aborts if this fails): no timestamps were recorded (u64 bytes written 0), none pending.
    void ReadTimestamps(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 5};
        rb.Push(ResultSuccess.raw);
        rb.Push<u64>(0);
        rb.Push<u32>(0);
    }
    // Display settings Eden's SDR, fixed-rate output has nothing to apply for (HDR output format
    // capability 302, HDR tone-mapping metadata 352/353, endpoint command 113).
    void AcknowledgeDisplaySetting(HLERequestContext& ctx) {
        LOG_DEBUG(Service_AM, "VI2 display setting command={}", ctx.GetCommand());
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess.raw);
    }

    std::shared_ptr<NativeImageRegistry> m_images;
    std::shared_ptr<Core::Timing::EventType> m_release_poll;
    std::array<ClientState, 16> m_client_states{};
    u32 m_client_slots{};
    std::shared_ptr<VI::IApplicationDisplayService> m_display_service;
    std::optional<u64> m_default_display;
    u32 m_display_references{};
    u32 m_display_notification{};
    std::array<u32, 3> m_player_display_references{};
    std::array<u32, 3> m_player_display_notifications{};
    bool m_request_groups_initialized{};
    u32 m_request_group_limit{};
    std::optional<u64> m_default_endpoint;
    std::array<u32, 32> m_notifications{};
    const u64 m_aruid;
    KernelHelpers::ServiceContext m_context;
    Kernel::KEvent* m_event{};
    Kernel::KEvent* m_status_event{};
    // Per status-event slot (after m_context, which must outlive them); m_vsync_displays holds the
    // VI display a 261-registered slot is linked to.
    std::array<std::optional<Event>, 64> m_slot_events{};
    std::array<std::optional<u64>, 64> m_vsync_displays{};
};

} // namespace


IGraphicsSupervisor::IGraphicsSupervisor(Core::System& system, Kernel::KProcess* process, u64 aruid)
    : ServiceFramework{system, "IGraphicsSupervisor"}, m_context{system, "GraphicsSupervisor"},
      m_syncpoint_cache_event{m_context}, m_process{process}, m_aruid{aruid} {
    m_process->Open(kernel);
    m_image_registry = std::make_shared<NativeImageRegistry>(kernel, m_process);
    m_decoder_images = std::make_shared<DecoderImageRegistry>();
    static const FunctionInfo functions[] = {
        {4, &IGraphicsSupervisor::Initialize, "Initialize"},
        {11, &IGraphicsSupervisor::BindProcess, "BindProcess"},
        {101, &IGraphicsSupervisor::CompleteBootstrap, "CompleteBootstrap"},
        {120, &IGraphicsSupervisor::QueryBootContinuation, "ContinueBoot"},
        {126, &IGraphicsSupervisor::QueryBootContinuation, "QueryBootContinuation"},
        {115, &IGraphicsSupervisor::ConfigureNativeMemory, "ConfigureNativeMemory"},
        {111, &IGraphicsSupervisor::CreateNativeMemory, "CreateNativeMemory"},
        {112, &IGraphicsSupervisor::DestroyNativeMemory, "DestroyNativeMemory"},
        {114, &IGraphicsSupervisor::BindNativeMemory, "BindNativeMemory"},
        {201, &IGraphicsSupervisor::InitializeSyncpointContext, "InitializeSyncpointContext"},
        {202, &IGraphicsSupervisor::GetSyncpointSession, "GetSyncpointSession"},
        {204, &IGraphicsSupervisor::BindSyncpointMemory, "BindSyncpointMemory"},
        {301, &IGraphicsSupervisor::InitializeDisplayContext, "InitializeDisplayContext"},
        {305, &IGraphicsSupervisor::OpenDisplaySession, "OpenDisplaySession"},
        {314, &IGraphicsSupervisor::ConfigureNativeWindowImagePool, "DestroyNativeWindowAliasImagePool"},
        {315, &IGraphicsSupervisor::ConfigureNativeWindowImagePool, "CreateNativeWindowAliasImagePool"},
        {401, &IGraphicsSupervisor::InitializeGpuContext, "InitializeGpuContext"},
        {402, &IGraphicsSupervisor::InitializeGpuContext, "ActivateGpuContext"},
        {423, &IGraphicsSupervisor::BindNativeGpuMemory, "BindNativeGpuMemory"},
        {429, &IGraphicsSupervisor::CreateNativeImage, "CreateNativeImage"},
        {430, &IGraphicsSupervisor::DestroyNativeImage, "DestroyNativeImage"},
        {434, &IGraphicsSupervisor::RegisterGpuAddressSpace, "RegisterGpuAddressSpace"},
        {497, &IGraphicsSupervisor::OpenGpuSession, "OpenGpuSession"},
        {1101, &IGraphicsSupervisor::InitializeFrameContext, "InitializeFrameContext"},
        {1102, &IGraphicsSupervisor::InitializeFrameContext, "ActivateFrameContext"},
        {1201, &IGraphicsSupervisor::ChatGraphicsCommand, "InitializeChatGraphics"},
        {1202, &IGraphicsSupervisor::OpenChatGraphicsSession, "OpenChatGraphicsSession"},
        {1203, &IGraphicsSupervisor::ChatGraphicsCommand, "FinalizeChatGraphics"},
        {901, &IGraphicsSupervisor::VideoCodecContextCommand, "InitializeVideoDecoderContext"},
        {902, &IGraphicsSupervisor::OpenVideoDecoderManager, "OpenVideoDecoderManager"},
        {903, &IGraphicsSupervisor::VideoCodecContextCommand, "FinalizeVideoDecoderContext"},
        {904, &IGraphicsSupervisor::OpenVideoDecoderManager, "OpenVideoDecoderManagerForSystem"},
        {905, &IGraphicsSupervisor::OpenVideoDecoderManager, "OpenVideoDecoderManagerUnlimited"},
        // Newer SDKs: 902 plus a flag byte and u64 library version; same reply.
        {906, &IGraphicsSupervisor::OpenVideoDecoderManager, "OpenVideoDecoderManagerWithVersion"},
        {911, &IGraphicsSupervisor::CreateDecoderImagePool, "InitializeDecoderImagePool"},
        {912, &IGraphicsSupervisor::DestroyDecoderImagePool, "FinalizeDecoderImagePool"},
        {921, &IGraphicsSupervisor::VideoCodecContextCommand, "InitializeDecoderAliasImagePool"},
        {922, &IGraphicsSupervisor::VideoCodecContextCommand, "FinalizeDecoderAliasImagePool"},
        {961, &IGraphicsSupervisor::GetDecoderImageDescriptor, "GetDecoderOutputImageDescriptor"},
        {991, &IGraphicsSupervisor::CalculateDecoderImagePool, "CalculateDecoderImagePoolBuffer"},
        {1001, &IGraphicsSupervisor::VideoCodecContextCommand, "InitializeVideoEncoderContext"},
        {1002, &IGraphicsSupervisor::OpenVideoEncoderManager, "OpenVideoEncoderManager"},
        {1003, &IGraphicsSupervisor::VideoCodecContextCommand, "FinalizeVideoEncoderContext"},
        {1004, &IGraphicsSupervisor::OpenVideoEncoderManager, "OpenVideoEncoderManagerForSystem"},
        {1111, &IGraphicsSupervisor::SetGraphicsFrameMode, "SetGraphicsFrameLengthMode"},
        {1112, &IGraphicsSupervisor::SetGraphicsFrameMode, "SetGraphicsFrameRestartMode"},
        {1113, &IGraphicsSupervisor::SetGraphicsFrameMode, "SetGraphicsFrameGpuTimesliceMode"},
        {1121, &IGraphicsSupervisor::OpenApplicationGpuTimeSession, "OpenApplicationGpuTimeSession"},
        {303, &IGraphicsSupervisor::FinalizeDisplayContext, "FinalizeDisplayContext"},
        {498, &IGraphicsSupervisor::QueryDonatedTransferMemoryUsage, "QueryDonatedTransferMemoryUsage"},
        {1402, &IGraphicsSupervisor::GetNvidiaDriver, "GetNvidiaDriver"},
        {1501, &IGraphicsSupervisor::InitializeImageContext, "InitializeImageContext1501"},
        {435, &IGraphicsSupervisor::CalculateImageRequirements, "CalculateNativeImageSize"},
        {1510, &IGraphicsSupervisor::CalculateImageRequirements, "CalculateImageRequirements"},
        {1502, &IGraphicsSupervisor::InitializeImageContext, "InitializeImageContext1502"},
    };
    RegisterHandlersTipc(functions);
}

IGraphicsSupervisor::~IGraphicsSupervisor() {
    if (m_cache_poll) {
        system.CoreTiming().UnscheduleEvent(m_cache_poll);
    }
    if (m_syncpoint_rw) {
        m_syncpoint_rw->Close(kernel);
        m_syncpoint_ro->Close(kernel);
    }
    if (m_nv_module) {
        auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
        for (const auto id : m_native_memories) {
            if (const auto info = nvmap.FreeHandle(id, false); info && info->can_unlock && info->address != 0) {
                m_process->GetPageTable().UnlockForDeviceAddressSpace(info->address, info->size);
            }
        }
        if (m_native_memory_session) {
            m_nv_module->GetContainer().CloseSession(Nvidia::NvCore::SessionId{*m_native_memory_session});
        }
    }
    m_context.CloseEvent(m_syncpoint_request_event);
    m_process->Close(kernel);
}

Result IGraphicsSupervisor::HandleSyncRequest(Kernel::KServerSession& session,
                                               HLERequestContext& ctx) {
    if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
        return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
    }
    const auto guard = LockService();
    InvokeRequestTipc(ctx);
    // Unsupported TIPC must return an error instead of leaving request data as a reply.
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(Kernel::ResultNotImplemented.raw);
    return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
}

void IGraphicsSupervisor::Initialize(HLERequestContext& ctx) {
    const auto request = GraphicsSupervisorProtocol::DecodeBootstrap(
        std::span<const u32>{ctx.CommandBuffer(), IPC::COMMAND_BUFFER_LENGTH});
    Result result = Kernel::ResultInvalidCombination;
    if (request) {
        auto process = ctx.GetObjectFromHandle<Kernel::KProcess>(request->process_handle);
        const bool empty_remote_memory = std::ranges::all_of(
            request->parameters, [](u64 value) { return value == 0; });
        if (process.IsNull() || process.GetPointerUnsafe() != m_process) {
            result = Kernel::ResultInvalidHandle;
        } else if (m_phase != Phase::Created) {
            result = Kernel::ResultInvalidState;
        } else if (request->applet_resource_user_id != m_aruid) {
            result = Kernel::ResultInvalidArgument;
        } else if ((request->flags & ~0x2ULL) != 0x8000000000000005ULL || !empty_remote_memory) {
            // Newer SDKs (Zelda: Ocarina of Time, 0400486024DEC000) send 0x8000000000000007 and
            // then ask 126 whether the boot needs 120.
            result = Kernel::ResultNotImplemented;
        } else {
            m_phase = Phase::Initialized;
            result = ResultSuccess;
        }
        LOG_DEBUG(Service_AM, "graphics bootstrap aruid={:#x}, flags={:#x}, result={:#x}",
                  request->applet_resource_user_id, request->flags, result.raw);
    }
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::BindProcess(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 0x1b && packet[1] == 0x80000000 && packet[2] == 2) {
        auto process = ctx.GetObjectFromHandle<Kernel::KProcess>(ctx.GetCopyHandle(0));
        if (process.IsNull() || process.GetPointerUnsafe() != m_process) {
            result = Kernel::ResultInvalidHandle;
        } else if (m_phase != Phase::Initialized) {
            result = Kernel::ResultInvalidState;
        } else {
            m_phase = Phase::ProcessBound;
            result = ResultSuccess;
        }
    }
    LOG_DEBUG(Service_AM, "graphics process binding result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::CompleteBootstrap(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 0x75 && packet[1] == 0) {
        if (m_phase != Phase::ProcessBound) {
            result = Kernel::ResultInvalidState;
        } else {
            m_phase = Phase::Ready;
            result = ResultSuccess;
        }
    }
    LOG_DEBUG(Service_AM, "graphics bootstrap completion result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

// 126: sent by newer nn::gsv::BootWithRemoteMemory after 101; replies a u32, and the SDK sends
// 120 (no data) only when it is 1. Eden has no further boot step, so it answers 0.
void IGraphicsSupervisor::QueryBootContinuation(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 16 + ctx.GetCommand() && packet[1] == 0) {
        result = m_phase == Phase::Ready ? ResultSuccess : Kernel::ResultInvalidState;
    }
    LOG_DEBUG(Service_AM, "graphics boot continuation command={}, result={:#x}",
              ctx.GetCommand(), result.raw);
    if (ctx.GetCommand() == 120) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
        return;
    }
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result.raw);
    rb.Push<u32>(0);
}

void IGraphicsSupervisor::InitializeImageContext(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 16 + ctx.GetCommand() && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else if (ctx.GetCommand() == 1501) {
            m_image_context_initialized = true;
            result = ResultSuccess;
        } else if (m_image_context_initialized) {
            m_image_formats_initialized = true;
            result = ResultSuccess;
        } else {
            result = Kernel::ResultInvalidState;
        }
    }
    LOG_DEBUG(Service_AM, "empty image context command={}, result={:#x}",
              ctx.GetCommand(), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::OpenGpuSession(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    Kernel::KClientSession* client{};
    const u64 flags = u64{packet[2]} | (u64{packet[3]} << 32);
    if (packet[0] == 0x201 && packet[1] == 2 && flags == 4) {
        if (m_phase != Phase::Ready || !m_nv_module) {
            result = Kernel::ResultInvalidState;
        } else {
            auto handler = std::make_shared<IGpuSession>(system, m_nv_module, m_process);
            result = CreateGraphicsSession(system, ctx, handler, &client);
            if (result.IsSuccess()) {
                std::erase_if(m_gpu_sessions, [](const auto& item) { return item.expired(); });
                m_gpu_sessions.emplace_back(handler);
            }
        }
    }
    LOG_DEBUG(Service_AM, "GPU control session flags={:#x}, result={:#x}", flags, result.raw);
    // This SDK reads the move-handle slot before the result, including on failure.
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

void IGraphicsSupervisor::RegisterGpuAddressSpace(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x1c2 && packet[1] == 1) {
        result = m_gpu_context_ready ? Kernel::ResultNotFound : Kernel::ResultInvalidState;
        if (m_gpu_context_ready) {
            for (const auto& item : m_gpu_sessions) {
                if (auto handler = item.lock()) {
                    result = handler->RegisterAddressSpace(packet[2]);
                    if (result.IsSuccess()) {
                        break;
                    }
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "register GPU address space id={}, result={:#x}", packet[2], result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::InitializeDisplayContext(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x13d && packet[1] == 0) {
        result = m_phase == Phase::Ready ? ResultSuccess : Kernel::ResultInvalidState;
        if (result.IsSuccess()) {
            m_display_context_initialized = true;
        }
    }
    LOG_DEBUG(Service_AM, "display context initialization result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::OpenDisplaySession(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Kernel::KClientSession* client{};
    const u64 flags = u64{packet[2]} | (u64{packet[3]} << 32);
    Result result = Kernel::ResultInvalidCombination;
    // 0x300020000 from Pokopia/HWAI; newer SDKs (Ocarina of Time) send 0x300040000.
    if (packet[0] == 0x141 && packet[1] == 2 &&
        (flags == 0x300020000ULL || flags == 0x300040000ULL)) {
        result = m_display_context_initialized ? ResultSuccess : Kernel::ResultInvalidState;
        if (result.IsSuccess()) {
            result = CreateGraphicsSession(system, ctx, std::make_shared<IVi2Session>(system, m_aruid, m_image_registry),
                                           &client);
        }
    }
    LOG_DEBUG(Service_AM, "VI2 session flags={:#x}, result={:#x}", flags, result.raw);
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

void IGraphicsSupervisor::InitializeGpuContext(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 16 + ctx.GetCommand() && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else if (ctx.GetCommand() == 401) {
            m_gpu_context_initialized = true;
            result = ResultSuccess;
        } else if (m_gpu_context_initialized) {
            m_gpu_context_ready = true;
            result = ResultSuccess;
        } else {
            result = Kernel::ResultInvalidState;
        }
    }
    LOG_DEBUG(Service_AM, "GPU context command={}, result={:#x}", ctx.GetCommand(), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::InitializeFrameContext(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 16 + ctx.GetCommand() && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else if (ctx.GetCommand() == 1101) {
            m_frame_context_initialized = true;
            result = ResultSuccess;
        } else if (m_frame_context_initialized) {
            m_frame_context_ready = true;
            result = ResultSuccess;
        } else {
            result = Kernel::ResultInvalidState;
        }
    }
    LOG_DEBUG(Service_AM, "empty frame context command={}, result={:#x}", ctx.GetCommand(), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::ConfigureNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x83 && packet[1] == 2 && packet[3] == 0 &&
        (packet[2] == 0 || packet[2] == 2)) {
        result = m_phase == Phase::Ready ? ResultSuccess : Kernel::ResultInvalidState;
    }
    // Clear the optional Gpuk-memory mode: Eden currently provides the NVDRV path.
    IPC::ResponseBuilder rb{ctx, 4};
    rb.Push(result.raw);
    rb.Push<u64>(0);
    LOG_DEBUG(Service_AM, "native memory uses NVDRV-compatible mode, result={:#x}", result.raw);
}

void IGraphicsSupervisor::InitializeSyncpointContext(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 0xd9 && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else {
            m_syncpoint_context_initialized = true;
            result = ResultSuccess;
        }
    }
    LOG_DEBUG(Service_AM, "syncpoint context initialization result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::GetSyncpointSession(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidState;
    Kernel::KClientSession* client{};
    if (packet[0] == 0xda && packet[1] == 0 && m_syncpoint_context_initialized && m_nv_module) {
        result = CreateGraphicsSession(system, ctx,
            std::make_shared<ISyncpointSupervisor>(system, m_nv_module), &client);
    }
    if (result.IsError()) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
        return;
    }
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(ResultSuccess.raw);
    rb.PushMoveObjects(ctx, client);
    LOG_DEBUG(Service_AM, "opened syncpoint-supervisor session");
}

// nn::gs::sdk's frame modes, one word each, abort unless they succeed: 1111
// SetGraphicsFrameLengthMode (0 = auto or 0x100 | target fps, sent every frame by
// nn::oe::SetGraphicsTargetFrameRate), 1112 SetGraphicsFrameRestartMode and 1113
// SetGraphicsFrameGpuTimesliceMode. Eden's frame pacing has no such modes to switch.
void IGraphicsSupervisor::SetGraphicsFrameMode(HLERequestContext& ctx) {
    LOG_DEBUG(Service_AM, "graphics frame mode command={}, value={:#x}", ctx.GetCommand(),
              ctx.CommandBuffer()[2]);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(ResultSuccess.raw);
}

void IGraphicsSupervisor::FinalizeDisplayContext(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x13f && packet[1] == 0) {
        result = m_display_context_initialized ? ResultSuccess : Kernel::ResultInvalidState;
        m_display_context_initialized = false;
    }
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

// Command 498 (nn::gpu::sdk::TryQueryDonatedTransferMemoryUsage): a remote-memory boot donates no
// transfer memory, so all three usage figures are zero.
void IGraphicsSupervisor::QueryDonatedTransferMemoryUsage(HLERequestContext& ctx) {
    IPC::ResponseBuilder rb{ctx, 8};
    rb.Push(ResultSuccess.raw);
    rb.Push<u64>(0);
    rb.Push<u64>(0);
    rb.Push<u64>(0);
}

void IGraphicsSupervisor::OpenApplicationGpuTimeSession(HLERequestContext& ctx) {
    Kernel::KClientSession* client{};
    const Result result = CreateGraphicsSession(system, ctx,
        std::make_shared<IApplicationGpuTimeSession>(system), &client);
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

// nn::codec::video: 901/903 and 1001/1003 begin and end the decoder and encoder contexts (once per
// movie), 921/922 the alias image pools, which Eden keeps no state for.
void IGraphicsSupervisor::VideoCodecContextCommand(HLERequestContext& ctx) {
    LOG_DEBUG(Service_AM, "video codec command={}", ctx.GetCommand());
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(ResultSuccess.raw);
}

void IGraphicsSupervisor::OpenVideoDecoderManager(HLERequestContext& ctx) {
    Kernel::KClientSession* client{};
    const Result result = CreateGraphicsSession(system, ctx,
        std::make_shared<IVideoDecoderManager>(system, m_nv_module, m_process, m_decoder_images), &client);
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

void IGraphicsSupervisor::OpenVideoEncoderManager(HLERequestContext& ctx) {
    Kernel::KClientSession* client{};
    const Result result = CreateGraphicsSession(system, ctx,
        std::make_shared<IVideoEncoderManager>(system), &client);
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

// 991: image preference (16 words) and image count; replies the pool's u64 size and alignment.
void IGraphicsSupervisor::CalculateDecoderImagePool(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    std::optional<DecoderImageLayout> layout;
    if (packet[0] == 0x3ef && packet[1] == 17) {
        layout = GetDecoderImageLayout(std::span{packet + 2, 16});
    }
    const u64 count = packet[18];
    LOG_DEBUG(Service_AM, "decoder image pool format={}, dimensions={:#x}, count={}, image size={:#x}",
              packet[3], packet[4], count, layout ? layout->size : 0);
    IPC::ResponseBuilder rb{ctx, 6};
    rb.Push((layout && count ? ResultSuccess : Kernel::ResultNotImplemented).raw);
    rb.Push<u64>(layout ? layout->size * count : 0);
    rb.Push<u64>(Kernel::PageSize);
}

// 911: image preference, image count and the native memory (already bound to guest memory by
// 111/114) holding the pool; replies the first image handle. Image i is that handle + i.
void IGraphicsSupervisor::CreateDecoderImagePool(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    u32 base{};
    if (packet[0] == 0x39f && packet[1] == 18) {
        const auto layout = GetDecoderImageLayout(std::span{packet + 2, 16});
        const u32 count = packet[18];
        result = layout && count && count <= 64 ? Kernel::ResultNotFound : Kernel::ResultNotImplemented;
        if (layout && count && count <= 64 && m_nv_module &&
            std::ranges::find(m_native_memories, packet[19]) != m_native_memories.end()) {
            const auto memory = m_nv_module->GetContainer().GetNvMapFile().GetHandle(packet[19]);
            result = Kernel::ResultInvalidSize;
            if (memory && memory->allocated && memory->size >= layout->size * count &&
                m_next_native_image_id != 0 && m_next_native_image_id + count > m_next_native_image_id) {
                base = m_next_native_image_id;
                m_next_native_image_id += count;
                const std::scoped_lock lock{m_decoder_images->mutex};
                for (u32 i = 0; i < count; ++i) {
                    const u64 offset = layout->size * i;
                    m_decoder_images->images[base + i] = {base, memory->address + offset, offset,
                                                          layout->width, layout->height,
                                                          layout->pitch, layout->aligned_height};
                }
                result = ResultSuccess;
            }
        }
    }
    LOG_DEBUG(Service_AM, "decoder image pool memory={}, count={}, base={}, result={:#x}",
              packet[19], packet[18], base, result.raw);
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result.raw);
    rb.Push(base);
}

void IGraphicsSupervisor::DestroyDecoderImagePool(HLERequestContext& ctx) {
    const u32 base = ctx.CommandBuffer()[2];
    const std::scoped_lock lock{m_decoder_images->mutex};
    const auto erased = std::erase_if(m_decoder_images->images,
                                      [base](const auto& entry) { return entry.second.pool == base; });
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push((erased ? ResultSuccess : Kernel::ResultNotFound).raw);
}

// 961: the 128-byte CpuUserImageDescriptor nn::gs::sdk::ReadUserImageNv12 copies a frame out by:
// byte 0 kind (10 = pitch-linear NV12), +8 luma offset, +16 u64 luma pitch, +24 chroma offset and
// +32 u64 chroma pitch, offsets relative to the pool's memory.
void IGraphicsSupervisor::GetDecoderImageDescriptor(HLERequestContext& ctx) {
    std::array<u32, 32> descriptor{};
    Result result = Kernel::ResultNotFound;
    {
        const std::scoped_lock lock{m_decoder_images->mutex};
        if (const auto it = m_decoder_images->images.find(ctx.CommandBuffer()[2]);
            it != m_decoder_images->images.end()) {
            const auto& image = it->second;
            descriptor[0] = 10;
            descriptor[2] = static_cast<u32>(image.pool_offset);
            descriptor[4] = image.pitch;
            descriptor[6] = static_cast<u32>(image.pool_offset + u64{image.pitch} * image.aligned_height);
            descriptor[8] = image.pitch;
            result = ResultSuccess;
        }
    }
    IPC::ResponseBuilder rb{ctx, 34};
    rb.Push(result.raw);
    for (const u32 word : descriptor) rb.Push(word);
}

void IGraphicsSupervisor::ChatGraphicsCommand(HLERequestContext& ctx) {
    LOG_WARNING(Service_AM, "(STUBBED) chat graphics supervisor command={}", ctx.GetCommand());
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(ResultSuccess.raw);
}

void IGraphicsSupervisor::OpenChatGraphicsSession(HLERequestContext& ctx) {
    Kernel::KClientSession* client{};
    const Result result = CreateGraphicsSession(system, ctx,
        std::make_shared<IChatGraphicsSession>(system), &client);
    // The SDK reads the handle slot (word 3) and result (word 4) even on failure.
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
    LOG_WARNING(Service_AM, "(STUBBED) opened chat graphics session, result={:#x}", result.raw);
}

void IGraphicsSupervisor::GetNvidiaDriver(HLERequestContext& ctx) {
    const auto Fail = [&](Result result) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    };
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] != 0x58a || packet[1] != 0) {
        Fail(Kernel::ResultInvalidCombination);
        return;
    }
    if (m_phase != Phase::Ready || !m_image_formats_initialized) {
        Fail(Kernel::ResultInvalidState);
        return;
    }
    auto source = system.ServiceManager().GetService<Nvidia::NVDRV>("nvdrv:s", true);
    if (!source) {
        Fail(Kernel::ResultNotFound);
        return;
    }
    m_nv_module = source->GetModule();
    Kernel::KClientSession* client{};
    const auto result = CreateGraphicsSession(system, ctx,
        std::make_shared<Nvidia::NVDRV>(system, m_nv_module, "nvdrv:gsv", m_process, m_aruid),
        &client);
    if (result.IsError()) {
        Fail(result);
        return;
    }
    LOG_DEBUG(Service_AM, "opened process-bound NVDRV session for aruid={:#x}", m_aruid);
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(ResultSuccess.raw);
    rb.PushMoveObjects(ctx, client);
}

void IGraphicsSupervisor::BindSyncpointMemory(HLERequestContext& ctx) {
    using namespace GraphicsSyncpointCache;
    const auto Fail = [&](Result result) {
        IPC::ResponseBuilder rb{ctx, 2, 0, 2};
        rb.Push(result.raw);
        ctx.AddMoveObject(nullptr);
        ctx.AddMoveObject(nullptr);
    };
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] != 0xdc || packet[1] != 0x80000006 || packet[2] != 4 ||
        packet[5] != 1 || packet[6] != 0x1ff || packet[7] != 0x1000 || packet[8] != 0 ||
        packet[9] != 0x2000 || packet[10] != 0) {
        Fail(Kernel::ResultInvalidCombination);
        return;
    }
    if (!m_syncpoint_context_initialized || !m_nv_module || m_syncpoint_rw) {
        Fail(Kernel::ResultInvalidState);
        return;
    }
    auto rw = ctx.GetObjectFromHandle<Kernel::KTransferMemory>(ctx.GetCopyHandle(0));
    auto ro = ctx.GetObjectFromHandle<Kernel::KTransferMemory>(ctx.GetCopyHandle(1));
    if (rw.IsNull() || ro.IsNull() || rw->GetOwner() != m_process || ro->GetOwner() != m_process ||
        rw.GetPointerUnsafe() == ro.GetPointerUnsafe()) {
        Fail(Kernel::ResultInvalidHandle);
        return;
    }
    if (rw->GetSize() != 0x1000 || ro->GetSize() != 0x2000) {
        Fail(Kernel::ResultInvalidSize);
        return;
    }
    auto* request_event = m_context.CreateEvent("SyncpointCacheRequest");
    if (request_event == nullptr) {
        Fail(Kernel::ResultOutOfResource);
        return;
    }
    m_syncpoint_request_event = request_event;
    m_syncpoint_rw = rw.GetPointerUnsafe();
    m_syncpoint_ro = ro.GetPointerUnsafe();
    m_syncpoint_rw->Open(kernel);
    m_syncpoint_ro->Open(kernel);
    auto& memory = m_process->GetMemory();
    const auto rw_base = rw->GetSourceAddress();
    const auto ro_base = ro->GetSourceAddress();
    const std::array<u8, 0x2000> zeros{};
    memory.WriteBlock(rw_base, zeros.data(), 0x1000);
    memory.WriteBlock(ro_base, zeros.data(), 0x2000);
    memory.WriteBlock(ro_base, Header.data(), sizeof(Header));
    for (u32 id = 0; id < CounterCount; ++id) {
        memory.Write64(ro_base + ValuesOffset + id * sizeof(u64), Unallocated);
    }
    m_cache_poll = Core::Timing::CreateEvent("GraphicsSyncpointCache",
        [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
            const auto guard = LockService();
            this->RefreshSyncpointCache();
            return std::nullopt;
        });
    system.CoreTiming().ScheduleLoopingEvent(std::chrono::milliseconds{1},
                                             std::chrono::milliseconds{1}, m_cache_poll);
    request_event->Open(kernel);
    auto* completed = m_syncpoint_cache_event.GetHandle();
    completed->Open(kernel);
    IPC::ResponseBuilder rb{ctx, 2, 0, 2};
    rb.Push(ResultSuccess.raw);
    ctx.AddMoveObject(request_event);
    ctx.AddMoveObject(completed);
    LOG_DEBUG(Service_AM, "bound syncpoint cache: {} real counters, rw={:#x}, ro={:#x}",
              CounterCount, GetInteger(rw_base), GetInteger(ro_base));
}

void IGraphicsSupervisor::RefreshSyncpointCache() {
    using namespace GraphicsSyncpointCache;
    if (!m_syncpoint_rw || !system.IsPoweredOn()) {
        return;
    }
    auto& memory = m_process->GetMemory();
    const auto rw = m_syncpoint_rw->GetSourceAddress();
    const auto ro = m_syncpoint_ro->GetSourceAddress();
    const u64 requests = memory.Read64(rw + GroupRequestOffset);
    const u64 acknowledgements = memory.Read64(ro + GroupAckOffset);
    u64 groups = (requests ^ acknowledgements) & ((1ULL << GroupCount) - 1);
    if (groups == 0) {
        return;
    }
    auto& syncpoints = m_nv_module->GetContainer().GetSyncpointManager();
    while (groups) {
        const u32 group = static_cast<u32>(std::countr_zero(groups));
        groups &= groups - 1;
        const u64 requested_slots = memory.Read64(rw + SlotRequestOffset + group * sizeof(u64));
        u64 slots = requested_slots ^ memory.Read64(ro + SlotAckOffset + group * sizeof(u64));
        while (slots) {
            const u32 bit = static_cast<u32>(std::countr_zero(slots));
            slots &= slots - 1;
            const u32 id = group * 64 + bit;
            const auto address = ro + ValuesOffset + id * sizeof(u64);
            const auto value = syncpoints.ReadAllocatedSyncpoint(id);
            const u64 updated = m_counter_values[id].Observe(value);
            memory.Write64(address, updated);
        }
        std::atomic_thread_fence(std::memory_order_release);
        memory.Write64(ro + SlotAckOffset + group * sizeof(u64), requested_slots);
    }
    std::atomic_thread_fence(std::memory_order_release);
    memory.Write64(ro + GroupAckOffset, requests);
    m_syncpoint_request_event->Clear(kernel);
    m_syncpoint_cache_event.Signal(kernel);
}

void IGraphicsSupervisor::CalculateImageRequirements(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    std::optional<NativeImageRequirements> requirements;
    const bool gpu_image = ctx.GetCommand() == 435;
    if (packet[0] == (gpu_image ? 0x1c3U : 0x5f6U) && packet[1] == 17 &&
        packet[3] == (gpu_image ? 1U : 2U)) {
        result = Kernel::ResultInvalidState;
        if (m_phase == Phase::Ready && m_image_formats_initialized) {
            requirements = CalculateNativeImageRequirements(std::span{packet + 2, 16}, packet[18]);
            result = requirements ? ResultSuccess : Kernel::ResultNotImplemented;
        }
    }
    if (result.IsError()) {
        LOG_WARNING(Service_AM,
                    "native image requirements rejected: command={} count={} preference={:08x} {:08x} {:08x} "
                    "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} "
                    "{:08x} {:08x}",
                    ctx.GetCommand(), packet[18], packet[2], packet[3], packet[4], packet[5], packet[6],
                    packet[7], packet[8], packet[9], packet[10], packet[11], packet[12], packet[13],
                    packet[14], packet[15], packet[16], packet[17]);
    }
    LOG_DEBUG(Service_AM, "native image requirements dimensions={:#x}, block={}, pitch={}, size={:#x}, alignment={:#x}, result={:#x}",
              packet[4], packet[6], packet[7], requirements ? requirements->size : 0,
              requirements ? requirements->alignment : 0, result.raw);
    IPC::ResponseBuilder rb{ctx, gpu_image ? 4U : 6U};
    rb.Push(result.raw);
    rb.Push<u64>(requirements ? requirements->size : 0);
    if (!gpu_image) { rb.Push<u64>(requirements ? requirements->alignment : 0); }
}
void IGraphicsSupervisor::CreateNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const u64 size = u64{packet[3]} | (u64{packet[4]} << 32);
    Result result = Kernel::ResultInvalidCombination;
    u32 id{};
    if (packet[0] == 0x7f && packet[1] == 3 && (packet[2] & 0xff) == 3) {
        result = Kernel::ResultInvalidState;
        if (m_phase == Phase::Ready && m_image_formats_initialized && m_nv_module) {
            result = Kernel::ResultInvalidSize;
            if (size && size <= (1ULL << 32) && Common::IsAligned(size, Kernel::PageSize)) {
                result = Kernel::ResultOutOfResource;
                if (m_native_memories.size() < 1024) {
                    auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
                    std::shared_ptr<Nvidia::NvCore::NvMap::Handle> handle;
                    if (nvmap.CreateHandle(size, handle) == Nvidia::NvResult::Success) {
                        id = handle->id;
                        m_native_memories.push_back(id);
                        result = ResultSuccess;
                    }
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native memory create size={:#x}, id={}, result={:#x}", size, id, result.raw);
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result.raw);
    rb.Push(id);
}

Result IGraphicsSupervisor::BindNativeMemoryBacking(u32 id, u64 address, u64 size, u64 flags, u64 offset) {
    R_UNLESS(m_nv_module && std::ranges::find(m_native_memories, id) != m_native_memories.end(),
             Kernel::ResultNotFound);
    R_UNLESS((flags == 0 || flags == 2) && offset == 0, Kernel::ResultNotImplemented);
    auto handle = m_nv_module->GetContainer().GetNvMapFile().GetHandle(id);
    R_UNLESS(handle && !handle->allocated, Kernel::ResultInvalidState);
    R_UNLESS(size && size == handle->size && address != 0 && address + size > address,
             Kernel::ResultInvalidSize);
    R_UNLESS(Common::IsAligned(address, Kernel::PageSize), Kernel::ResultInvalidAddress);
    auto& page_table = m_process->GetPageTable();
    R_UNLESS(page_table.Contains(address, size), Kernel::ResultInvalidCurrentMemory);
    if (flags == 2) {
        for (u64 current = address; current < address + size;) {
            Kernel::KMemoryInfo info{};
            Kernel::Svc::PageInfo page{};
            R_TRY(page_table.QueryInfo(&info, &page, current));
            R_UNLESS(True(info.m_attribute & Kernel::KMemoryAttribute::GpuAccessible),
                     Kernel::ResultInvalidCurrentMemory);
            const u64 next = info.m_address + info.m_size;
            R_UNLESS(next > current, Kernel::ResultInvalidCurrentMemory);
            current = next;
        }
    }
    bool is_io{};
    R_TRY(page_table.LockForMapDeviceAddressSpace(&is_io, address, size,
                                                 Kernel::KMemoryPermission::None, true, true));
    if (is_io) {
        page_table.UnlockForDeviceAddressSpace(address, size);
        R_RETURN(Kernel::ResultNotImplemented);
    }
    if (!m_native_memory_session) {
        m_native_memory_session = m_nv_module->GetContainer().OpenSession(m_process).id;
    }
    const auto result = handle->Alloc({}, Kernel::PageSize, 0, address,
                                      Nvidia::NvCore::SessionId{*m_native_memory_session});
    if (result != Nvidia::NvResult::Success) {
        page_table.UnlockForDeviceAddressSpace(address, size);
        R_RETURN(Kernel::ResultOutOfResource);
    }
    R_SUCCEED();
}

void IGraphicsSupervisor::BindNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const auto pair = [&](size_t i) { return u64{packet[i]} | (u64{packet[i + 1]} << 32); };
    const auto result = packet[0] == 0x82 && packet[1] == 9
        ? BindNativeMemoryBacking(packet[2], pair(3), pair(5), pair(7), pair(9))
        : Kernel::ResultInvalidCombination;
    LOG_DEBUG(Service_AM, "native memory bind id={}, address={:#x}, size={:#x}, flags={:#x}, result={:#x}",
              packet[2], pair(3), pair(5), pair(7), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

Result IGraphicsSupervisor::ReleaseNativeMemory(u32 id) {
    const auto it = std::ranges::find(m_native_memories, id);
    R_UNLESS(m_nv_module && it != m_native_memories.end(), Kernel::ResultNotFound);
    auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
    auto handle = nvmap.GetHandle(id);
    R_UNLESS(handle, Kernel::ResultNotFound);
    {
        const std::scoped_lock lock{handle->mutex};
        R_UNLESS(handle->pins == 0 && handle->internal_dupes == 0 && handle->dupes == 1,
                 Kernel::ResultBusy);
    }
    const bool allocated = handle->allocated;
    handle.reset();
    const auto info = nvmap.FreeHandle(id, false);
    m_native_memories.erase(it);
    if (allocated) {
        R_UNLESS(info && info->can_unlock, Kernel::ResultInvalidState);
        R_TRY(m_process->GetPageTable().UnlockForDeviceAddressSpace(info->address, info->size));
    }
    R_SUCCEED();
}

void IGraphicsSupervisor::DestroyNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const auto result = packet[0] == 0x80 && packet[1] == 1
        ? ReleaseNativeMemory(packet[2]) : Kernel::ResultInvalidCombination;
    LOG_DEBUG(Service_AM, "native memory destroy id={}, result={:#x}", packet[2], result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::BindNativeGpuMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const auto pair = [&](size_t i) { return u64{packet[i]} | (u64{packet[i + 1]} << 32); };
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x1b7 && packet[1] == 12 && pair(10) == 0 && pair(12) == 0) {
        result = Kernel::ResultNotFound;
        if (m_nv_module) {
            auto& container = m_nv_module->GetContainer();
            const auto parent = container.GetNvMapFile().GetHandle(packet[7]);
            if (parent && parent->allocated) {
                const auto* session = container.GetSession(parent->session_id);
                const u64 offset = pair(8);
                const u64 address = pair(3);
                const u64 size = pair(5);
                if (session && session->process == m_process && offset <= parent->size &&
                    size <= parent->size - offset && parent->address + offset >= parent->address &&
                    address == parent->address + offset) {
                    result = BindNativeMemoryBacking(packet[2], address, size, 2, 0);
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native GPU memory bind id={}, parent={}, address={:#x}, size={:#x}, offset={:#x}, result={:#x}",
              packet[2], packet[7], pair(3), pair(5), pair(8), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::CreateNativeImage(HLERequestContext& ctx) {
    const std::scoped_lock images_lock{m_image_registry->mutex};
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    u32 id{};
    if (packet[0] == 0x1bd && packet[1] == 18 && packet[3] == 1) {
        result = Kernel::ResultInvalidState;
        if (m_phase == Phase::Ready && m_gpu_context_ready && m_image_formats_initialized && m_nv_module) {
            const auto requirements = CalculateNativeImageRequirements(std::span{packet + 2, 16}, packet[18]);
            result = Kernel::ResultNotImplemented;
            if (requirements) {
                result = Kernel::ResultNotFound;
                auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
                const auto memory = nvmap.GetHandle(packet[19]);
                if (std::ranges::find(m_native_memories, packet[19]) != m_native_memories.end() &&
                    memory && memory->allocated) {
                    result = Kernel::ResultInvalidSize;
                    if (memory->size >= requirements->size) {
                        result = Kernel::ResultOutOfResource;
                        if (m_image_registry->images.size() < 1024 && m_next_native_image_id != 0 &&
                            memory->Duplicate(true) == Nvidia::NvResult::Success) {
                            NativeImage image{m_next_native_image_id++, packet[19], {}, requirements->size};
                            std::copy_n(packet + 2, image.preference.size(), image.preference.begin());
                            id = image.id;
                            if (!m_image_registry->module) {
                                m_image_registry->module = m_nv_module;
                                m_image_registry->memory_session = m_nv_module->GetContainer().OpenSession(m_process);
                            }
                            m_image_registry->images.push_back(image);
                            result = ResultSuccess;
                        }
                    }
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native image create memory={}, count={}, id={}, result={:#x}",
              packet[19], packet[18], id, result.raw);
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result.raw);
    rb.Push(id);
}

void IGraphicsSupervisor::DestroyNativeImage(HLERequestContext& ctx) {
    const std::scoped_lock images_lock{m_image_registry->mutex};
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x1be && packet[1] == 1) {
        const auto image = std::ranges::find(m_image_registry->images, packet[2], &NativeImage::id);
        result = Kernel::ResultNotFound;
        if (m_nv_module && image != m_image_registry->images.end()) {
            result = Kernel::ResultBusy;
            if (!image->window_pool) {
                m_image_registry->ReleaseMemory(image->memory_id);
                m_image_registry->images.erase(image);
                result = ResultSuccess;
            }
        }
    }
    LOG_DEBUG(Service_AM, "native image destroy id={}, result={:#x}", packet[2], result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::ConfigureNativeWindowImagePool(HLERequestContext& ctx) {
    const std::scoped_lock images_lock{m_image_registry->mutex};
    const auto* packet = ctx.CommandBuffer();
    const bool create = ctx.GetCommand() == 315;
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == (create ? 0x14bU : 0x14aU) && packet[1] == (create ? 2U : 1U)) {
        const auto image = std::ranges::find(m_image_registry->images, packet[2], &NativeImage::id);
        result = Kernel::ResultNotFound;
        if (image != m_image_registry->images.end()) {
            result = Kernel::ResultNotImplemented;
            if (!create || packet[3] == 1) {
                result = Kernel::ResultInvalidState;
                if (image->window_pool != create) {
                    image->window_pool = create;
                    result = ResultSuccess;
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native window image pool id={}, create={}, result={:#x}",
              packet[2], create, result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

} // namespace Service::AM
