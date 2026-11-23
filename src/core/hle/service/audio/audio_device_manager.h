// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

#include "core/hle/service/cmif_types.h"
#include "core/hle/service/service.h"

namespace Service::Audio {

class IAudioDeviceManagerSession;
class SystemMixerStreams;

// "auddmg:u" (nn::audio::devmgr::IUserSessionManager). Behind nn::audio's virtual device list
// (GetDefaultVirtualOutputDevice, ListVirtualInputDevices, ...) and OutputStream Start/Stop.
class IAudioDeviceManager final : public ServiceFramework<IAudioDeviceManager> {
public:
    explicit IAudioDeviceManager(Core::System& system_,
                                 std::shared_ptr<SystemMixerStreams> streams_);
    ~IAudioDeviceManager() override;

private:
    Result OpenUserSession(Out<SharedPointer<IAudioDeviceManagerSession>> out_session,
                           u32 version, ClientAppletResourceUserId aruid, u64 reserved);

    std::shared_ptr<SystemMixerStreams> streams;
};

} // namespace Service::Audio
