// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <string_view>

#include "common/logging.h"
#include "common/string_util.h"
#include "core/hle/service/audio/audio_device_manager.h"
#include "core/hle/service/audio/audio_system_mixer.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/cmif_serialization.h"

namespace Service::Audio {

namespace {

// nn::audio::detail::NameStorage<255>: names travel in 0x100-byte entries.
using DeviceName = std::array<char, 0x100>;

// VirtualDeviceProfile, the 0x228-byte B buffer of command 401 (nnSdk+0x224bf4). The SDK reads
// three fields of it: +0x108 goes to VirtualDeviceProperties+0x108 (nnSdk+0x224cec), +0x218 is a
// time in ns that OutputStreamImpl::Open turns into the minimum stream frame count
// (nnSdk+0x225a94-0x225ad0) and GetVirtualOutputDeviceProperties copies to
// VirtualDeviceProperties+0x100 (nnSdk+0x224ce4), +0x220 is the device id passed to sysmix
// CreateStream (nnSdk+0x225b00, nnSdk+0x225b2c).
constexpr size_t VirtualDeviceProfileSize = 0x228;
constexpr size_t ProfileChannelCountOffset = 0x108;
constexpr size_t ProfileLatencyOffset = 0x218;
constexpr size_t ProfileDeviceIdOffset = 0x220;

// The SDK sends a name as its characters without the terminator (NameStorage length,
// nnSdk+0x2ad0e4).
bool IsOutputDevice(std::span<const u8> name, u32 functionality) {
    const std::string_view view{reinterpret_cast<const char*>(name.data()), name.size()};
    return functionality == SystemMixerStreams::OutputFunctionality &&
           view.substr(0, view.find('\0')) == SystemMixerStreams::OutputDeviceName;
}

} // namespace

// nn::audio::devmgr::IUserSession, the session command 0 of auddmg:u moves to the client. Command
// ids from the SDK proxy vtable (nnSdk data 0xe62f08) and the devmgr functions that use each slot.
class IAudioDeviceManagerSession final : public ServiceFramework<IAudioDeviceManagerSession> {
public:
    explicit IAudioDeviceManagerSession(Core::System& system_,
                                        std::shared_ptr<SystemMixerStreams> streams_,
                                        u64 process_id_)
        : ServiceFramework{system_, "IAudioDeviceManagerSession"}, streams{std::move(streams_)},
          process_id{process_id_} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, D<&IAudioDeviceManagerSession::ListVirtualDeviceNames>, "ListVirtualDeviceNames"},
            {100, nullptr, "GetDefaultVirtualDeviceName"},
            {200, D<&IAudioDeviceManagerSession::IsVirtualDeviceAvailable>, "IsVirtualDeviceAvailable"},
            {300, D<&IAudioDeviceManagerSession::IsVirtualDeviceStreamFormatSupported>, "IsVirtualDeviceStreamFormatSupported"},
            {401, D<&IAudioDeviceManagerSession::GetVirtualDeviceProfile>, "GetVirtualDeviceProfile"},
            {500, nullptr, "AcquireVirtualDeviceStateChangeEvent"},
            {600, nullptr, "ReleaseVirtualDeviceStateChangeEvent"},
            {700, D<&IAudioDeviceManagerSession::AttachStream>, "AttachStream"},
            {800, D<&IAudioDeviceManagerSession::DetachStream>, "DetachStream"},
            {900, nullptr, "GetPhysicalDeviceName"},
            {1000, nullptr, "GetDeviceStreamFormat"},
            {1500, nullptr, "GetMixCount"},
            {1550, nullptr, "GetMixTopology"},
            {1600, nullptr, "GetMixNode"},
            {2000, nullptr, "IsCaptureDeviceStreamFormatSupported"},
            {2100, nullptr, "AttachCaptureStream"},
            {2301, nullptr, "GetCaptureDeviceProfile"},
            {3000, nullptr, "AttachInjectionStream"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

private:
    // u8 flag at 0 and the functionality at 4 (nnSdk+0x2adeb4-0x2adebc); the names go to a B
    // buffer and the count comes back (nnSdk+0x2adf3c). Functionality 1 is output
    // (GetDefaultVirtualOutputDevice takes the first name, nnSdk+0x224b90); 2 is what
    // ListVirtualInputDevices asks for (nnSdk+0x1f60d0). Eden has no input device to offer, and
    // the game's microphone module simply returns on an empty list (Owl.nss+0xf48128).
    Result ListVirtualDeviceNames(Out<s32> out_count,
                                  OutArray<DeviceName, BufferAttr_HipcMapAlias> out_names,
                                  bool flag, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, functionality {}, flag {}", functionality, flag);
        std::fill(out_names.begin(), out_names.end(), DeviceName{});
        s32 count = 0;
        if (functionality == SystemMixerStreams::OutputFunctionality && !out_names.empty()) {
            std::memcpy(out_names[0].data(), SystemMixerStreams::OutputDeviceName.data(),
                        SystemMixerStreams::OutputDeviceName.size());
            count = 1;
        }
        *out_count = count;
        R_SUCCEED();
    }

    // Name in an A buffer, functionality (nnSdk+0x1a3158); the result is the answer.
    Result IsVirtualDeviceAvailable(InBuffer<BufferAttr_HipcMapAlias> name, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, {} functionality {}", Common::StringFromBuffer(name),
                  functionality);
        R_UNLESS(IsOutputDevice(name, functionality), ResultNotFound);
        R_SUCCEED();
    }

    // Name in an A buffer, the stream format and functionality (nnSdk+0x2ae044-0x2ae050). A
    // failure makes OutputStreamImpl::Open fail (nnSdk+0x225a68) and the game abort
    // (Owl.nss+0x3b617f0), so every format the stream service can consume is accepted.
    Result IsVirtualDeviceStreamFormatSupported(InBuffer<BufferAttr_HipcMapAlias> name,
                                                StreamFormat format, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, {} functionality {}, format {}, {} channels, {} Hz",
                  Common::StringFromBuffer(name), functionality, format.sample_format,
                  format.channel_count, format.sample_rate);
        R_UNLESS(IsOutputDevice(name, functionality), ResultNotFound);
        R_UNLESS(SystemMixerStreams::IsFormatSupported(format), ResultNotSupported);
        R_SUCCEED();
    }

    // Profile to a B buffer, name in an A buffer, functionality (nnSdk+0x2ae1dc).
    Result GetVirtualDeviceProfile(OutBuffer<BufferAttr_HipcMapAlias> out_profile,
                                   InBuffer<BufferAttr_HipcMapAlias> name, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, {} functionality {}", Common::StringFromBuffer(name),
                  functionality);
        R_UNLESS(IsOutputDevice(name, functionality), ResultNotFound);
        std::array<u8, VirtualDeviceProfileSize> profile{};
        const u32 channel_count = SystemMixerStreams::OutputChannelCount;
        const u64 latency = SystemMixerStreams::PumpPeriodNs;
        const u32 device_id = SystemMixerStreams::OutputDeviceId;
        std::memcpy(profile.data() + ProfileChannelCountOffset, &channel_count,
                    sizeof(channel_count));
        std::memcpy(profile.data() + ProfileLatencyOffset, &latency, sizeof(latency));
        std::memcpy(profile.data() + ProfileDeviceIdOffset, &device_id, sizeof(device_id));
        std::fill(out_profile.begin(), out_profile.end(), u8{0});
        std::memcpy(out_profile.data(), profile.data(),
                    std::min(out_profile.size(), profile.size()));
        R_SUCCEED();
    }

    // Name in an A buffer, the sysmix stream id and functionality (nnSdk+0x2ae680); starts the
    // stream (OutputStreamImpl::Start, nnSdk+0x225368).
    Result AttachStream(InBuffer<BufferAttr_HipcMapAlias> name, u32 stream_id, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, stream {} to {} functionality {}", stream_id,
                  Common::StringFromBuffer(name), functionality);
        R_UNLESS(IsOutputDevice(name, functionality), ResultNotFound);
        R_RETURN(streams->AttachStream(stream_id, process_id));
    }

    // Stream id (nnSdk+0x1a4a50); stops the stream (OutputStreamImpl::Stop, nnSdk+0x2255a0).
    Result DetachStream(u32 stream_id) {
        LOG_DEBUG(Service_Audio, "called, stream {}", stream_id);
        R_RETURN(streams->DetachStream(stream_id, process_id));
    }

    std::shared_ptr<SystemMixerStreams> streams;
    u64 process_id;
};

IAudioDeviceManager::IAudioDeviceManager(Core::System& system_,
                                         std::shared_ptr<SystemMixerStreams> streams_)
    : ServiceFramework{system_, "auddmg:u"}, streams{std::move(streams_)} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IAudioDeviceManager::OpenUserSession>, "OpenUserSession"},
    };
    // clang-format on
    RegisterHandlers(functions);
}

IAudioDeviceManager::~IAudioDeviceManager() = default;

// u32 2, the aruid with the process id and a u64 0 (nnSdk+0x2ad954-0x2ad964, nnSdk+0x2ac978).
// The session must come back as a moved handle: without one the SDK keeps a NULL session and the
// next request jumps to address 0 (nnSdk+0x2ada00-0x2ada64, nnSdk+0x2ace08).
Result IAudioDeviceManager::OpenUserSession(
    Out<SharedPointer<IAudioDeviceManagerSession>> out_session, u32 version,
    ClientAppletResourceUserId aruid, u64 reserved) {
    LOG_INFO(Service_Audio, "called, version {}, aruid {:#x}, reserved {:#x}", version, aruid.pid,
             reserved);
    *out_session = std::make_shared<IAudioDeviceManagerSession>(system, streams, aruid.pid);
    R_SUCCEED();
}

} // namespace Service::Audio
