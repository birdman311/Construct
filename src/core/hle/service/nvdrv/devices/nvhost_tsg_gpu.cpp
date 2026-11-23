// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <array>
#include <cstring>
#include "common/logging.h"
#include "core/core.h"
#include "video_core/macro.h"
#include "video_core/native_qmd.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/nvdrv/nvdrv.h"
#include "core/hle/service/nvdrv/devices/nvhost_gpu.h"
#include "core/hle/service/nvdrv/devices/nvhost_tsg_gpu.h"

namespace Service::Nvidia::Devices {

nvhost_tsg_gpu::nvhost_tsg_gpu(Core::System& system, Module& module_, NvCore::Container& core_)
    : nvdevice{system, Type::GpuTsg}, native_macros{std::make_shared<Tegra::NativeMacroContext>()}, module{module_}, core{core_} { native_macros->engine.native_turing = true; native_macros->compute_engine.native_turing = true; }

nvhost_tsg_gpu::~nvhost_tsg_gpu() { OnClose(0); }

void nvhost_tsg_gpu::OnOpen(NvCore::SessionId id, DeviceFD fd) {
    group_fd = fd;
    if (const auto* session = core.GetSession(id)) {
        process = session->process;
        process->Open(system.Kernel());
    }
}

void nvhost_tsg_gpu::OnClose(DeviceFD fd) {
    for (const auto& [channel_fd, channel] : channels) { channel->UnbindFromTsg(group_fd); }
    channels.clear();
    if (process) {
        process->Close(system.Kernel());
        process = nullptr;
    }
}


NvResult nvhost_tsg_gpu::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<u8> output) {
    std::array<u32, 8> words{};
    if (!input.empty()) { std::memcpy(words.data(), input.data(), std::min(input.size(), sizeof(words))); }
    LOG_DEBUG(Service_NVDRV, "native TSG ioctl={:#x}, input={}, output={}, words={:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
              command.raw, input.size(), output.size(), words[0], words[1], words[2], words[3],
              words[4], words[5], words[6], words[7]);
    // 'T' 0xb (bind channel with subcontext) replaces 'T' 1 once the TSG level is 0xb or more:
    // {u32 channel fd, u32 subcontext id, 16 reserved bytes} (Ocarina of Time, nnSdk+0x63e5e0).
    // Eden does not model subcontext ids, so it binds the channel like 'T' 1.
    const bool bind_with_subcontext = command.raw == 0xc018540b;
    if (command.raw != 0xc0045401 && command.raw != 0xc0045402 && !bind_with_subcontext) {
        return NvResult::NotImplemented;
    }
    const size_t args_size = bind_with_subcontext ? 0x18 : 4;
    if (!process || input.size() != args_size || output.size() != args_size) {
        return NvResult::BadParameter;
    }
    const DeviceFD channel_fd = static_cast<s32>(words[0]);
    if (command.cmd == 1 || bind_with_subcontext) {
        if (channels.contains(channel_fd)) { return NvResult::Busy; }
        if (channels.size() >= 64) { return NvResult::InsufficientMemory; }
        const auto device = module.GetDevice<nvdevice>(channel_fd);
        if (!device || device->GetType() != Type::GpuChannel) { return NvResult::BadParameter; }
        const auto channel = std::static_pointer_cast<nvhost_gpu>(device);
        // Channels bound with a subcontext ('T' 0xb, Ocarina of Time's SDK) schedule a QMD on one
        // channel and decrement its dependencies from another, across TSGs (its fence ring QMDs):
        // a QMD's dependency counter is state of the QMD, so they all share one NVN2 table, as
        // the GPU does (and Lichii). Older SDKs ('T' 1) keep one table per channel.
        if (bind_with_subcontext && !native_macros->nvn2) {
            native_macros->nvn2 = true;
            LOG_INFO(Service_NVDRV, "TSG {}: newer-SDK (NVN2) GPU sync rules", fd);
        }
        static const bool shared_qmds = std::getenv("EDEN_NVN2_SHARED_QMDS") != nullptr;
        if (bind_with_subcontext && shared_qmds && !native_macros->shared_qmds) {
            static std::mutex shared_mutex;
            static std::weak_ptr<Tegra::NativeQmdScheduler> shared_table;
            std::scoped_lock lk{shared_mutex};
            auto table = shared_table.lock();
            if (!table) {
                table = std::make_shared<Tegra::NativeQmdScheduler>(true);
                shared_table = table;
            }
            native_macros->shared_qmds = std::move(table);
            LOG_INFO(Service_NVDRV, "TSG {}: compute QMDs use the shared NVN2 dependency table", fd);
        }
        const auto result = channel->BindToTsg(fd, process, native_macros);
        if (result != NvResult::Success) { return result; }
        channels.emplace(channel_fd, channel);
    } else {
        const auto it = channels.find(channel_fd);
        if (it == channels.end()) { return NvResult::BadParameter; }
        it->second->UnbindFromTsg(fd);
        channels.erase(it);
    }
    std::copy(input.begin(), input.end(), output.begin());
    return NvResult::Success;
}

NvResult nvhost_tsg_gpu::Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<const u8> inline_input, std::span<u8> output) {
    return NvResult::NotImplemented;
}

NvResult nvhost_tsg_gpu::Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<u8> output, std::span<u8> inline_output) {
    return NvResult::NotImplemented;
}

} // namespace Service::Nvidia::Devices
