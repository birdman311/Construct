// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <string>
#include <string_view>

#include "core/arm/arm_interface.h"
#include "core/arm/debug.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_light_lock.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_synchronization_object.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/launch_log.h"
#include "core/memory.h"

namespace Core::LaunchLog {

inline void WriteBacktrace(Kernel::KernelCore& kernel, std::string_view what) {
    if (!Active()) {
        return;
    }
    auto* process = Kernel::GetCurrentProcessPointer(kernel);
    if (!process) {
        return;
    }
    auto* arm = process->GetArmInterface(kernel.CurrentPhysicalCoreIndex());
    if (!arm) {
        return;
    }
    Kernel::Svc::ThreadContext ctx{};
    arm->GetContext(ctx);
    Write("{}: pc={:#x} lr={:#x} sp={:#x} fp={:#x}", what, ctx.pc, ctx.lr, ctx.sp, ctx.fp);
    for (const auto& entry : Core::GetBacktraceFromContext(process, ctx)) {
        Write("    {}+{:#x}  {}", entry.module, entry.offset,
              entry.name.empty() ? std::string_view{"?"} : std::string_view{entry.name});
    }
}

namespace detail {

// What a blocked thread waits on, from the syscall arguments still in its saved registers
// (64-bit ABI): WaitSynchronization(X1 handles, X2 count, X3 timeout), WaitForAddress(X0 address,
// X1 type, X2 value, X3 timeout), WaitProcessWideKeyAtomic(X0 mutex, X1 key, X2 tag, X3 timeout),
// ArbitrateLock(X0 owner handle, X1 mutex, X2 tag), SendSyncRequest(X0 session).
inline void WriteWaitDetail(Kernel::KernelCore& kernel, Kernel::KProcess& process,
                            const Kernel::KThread& thread, const Kernel::Svc::ThreadContext& ctx) {
    auto& memory = process.GetMemory();
    const auto describe = [&](Kernel::Handle handle) -> std::string {
        auto object = process.GetHandleTable().GetObject(kernel, handle);
        if (object.IsNull()) {
            return fmt::format("{:#x}=invalid", handle);
        }
        std::string text = fmt::format("{:#x}={}@{}", handle, object->GetTypeName(),
                                       fmt::ptr(object.GetPointerUnsafe()));
        if (auto* sync = object->DynamicCast<Kernel::KSynchronizationObject*>()) {
            Kernel::KScopedSchedulerLock sl{kernel};
            text += sync->IsSignaled(kernel) ? "(signaled)" : "(not signaled)";
        }
        return text;
    };
    switch (thread.GetWaitReasonForDebugging()) {
    case Kernel::ThreadWaitReasonForDebugging::Synchronization: {
        const s32 count = static_cast<s32>(ctx.r[2]);
        std::string list;
        for (s32 i = 0; i < std::clamp(count, 0, 16); ++i) {
            list += ' ';
            list += describe(memory.Read32(ctx.r[1] + 4 * static_cast<u64>(i)));
        }
        Write("    waits on {} handle(s), timeout {:#x}:{}", count, ctx.r[3], list);
        break;
    }
    case Kernel::ThreadWaitReasonForDebugging::Arbitration:
        Write("    address wait: address {:#x} (now {:#x}), type {}, value {:#x}, timeout {:#x}",
              ctx.r[0], memory.Read32(ctx.r[0]), ctx.r[1], ctx.r[2], ctx.r[3]);
        break;
    case Kernel::ThreadWaitReasonForDebugging::ConditionVar:
        Write("    condvar/lock wait: x0 {:#x} (now {:#x}), x1 {:#x} (now {:#x}), x2 {:#x}, x3 {:#x}",
              ctx.r[0], memory.Read32(ctx.r[0]), ctx.r[1], memory.Read32(ctx.r[1]), ctx.r[2],
              ctx.r[3]);
        break;
    case Kernel::ThreadWaitReasonForDebugging::IPC:
        Write("    IPC wait: session {}", describe(static_cast<Kernel::Handle>(ctx.r[0])));
        break;
    case Kernel::ThreadWaitReasonForDebugging::Sleep:
        Write("    sleeping: x0 {:#x}", ctx.r[0]);
        break;
    default:
        break;
    }
}

} // namespace detail

// Writes every thread of the calling process: name, state, wait target and backtrace. Called from
// a guest syscall (Kernel::Svc::Call) when the stall watchdog asks for it.
inline void DumpThreads(Kernel::KernelCore& kernel, std::string_view why) {
    if (!Active()) {
        return;
    }
    auto* process = Kernel::GetCurrentProcessPointer(kernel);
    if (!process || !process->Is64Bit()) {
        return;
    }
    auto* current = Kernel::GetCurrentThreadPointer(kernel);
    Write("==================== thread dump: {} ====================", why);
    Kernel::KScopedLightLock ll{process->GetListLock()};
    for (auto& thread : process->GetThreadList()) {
        Kernel::Svc::ThreadContext ctx = thread.GetContext();
        std::string_view where = "";
        if (&thread == current) {
            if (auto* arm = process->GetArmInterface(kernel.CurrentPhysicalCoreIndex())) {
                arm->GetContext(ctx);
            }
            where = " [this thread, in a syscall]";
        } else {
            for (s32 core = 0; core < static_cast<s32>(Core::Hardware::NUM_CPU_CORES); ++core) {
                if (kernel.Scheduler(static_cast<std::size_t>(core)).GetSchedulerCurrentThread() ==
                    &thread) {
                    where = " [running now, registers stale]";
                }
            }
        }
        const auto name = Core::GetThreadName(&thread);
        Write("thread {} '{}' prio {} core {} {}{}: pc={:#x} lr={:#x} sp={:#x} fp={:#x} "
              "x0-x5 {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
              thread.GetThreadId(), name.value_or("?"), thread.GetPriority(),
              thread.GetActiveCore(), Core::GetThreadState(&thread), where, ctx.pc, ctx.lr, ctx.sp,
              ctx.fp, ctx.r[0], ctx.r[1], ctx.r[2], ctx.r[3], ctx.r[4], ctx.r[5]);
        Write("    x6-x18 {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
              ctx.r[6], ctx.r[7], ctx.r[8], ctx.r[9], ctx.r[10], ctx.r[11], ctx.r[12], ctx.r[13],
              ctx.r[14], ctx.r[15], ctx.r[16], ctx.r[17], ctx.r[18]);
        // Callee-saved registers keep a polling loop's address and target across its calls.
        std::string saved;
        auto& memory = process->GetMemory();
        for (size_t reg = 19; reg <= 28; ++reg) {
            saved += fmt::format(" x{}={:#x}", reg, ctx.r[reg]);
            if (ctx.r[reg] % 4 == 0 && memory.IsValidVirtualAddress(ctx.r[reg])) {
                saved += fmt::format("[{:#x}]", memory.Read32(ctx.r[reg]));
            }
        }
        Write("   {}", saved);
        if (where.empty() && thread.GetState() == Kernel::ThreadState::Waiting) {
            detail::WriteWaitDetail(kernel, *process, thread, ctx);
        }
        for (const auto& entry : Core::GetBacktraceFromContext(process, ctx)) {
            Write("    {}+{:#x}  {}", entry.module, entry.offset,
                  entry.name.empty() ? std::string_view{"?"} : std::string_view{entry.name});
        }
    }
    if (const u64 block = QueueSemaphoreBlock(); block != 0) {
        auto& memory = process->GetMemory();
        for (u64 page = 0; page < 2; ++page) {
            std::string words;
            for (u64 slot = 0; slot < 8; ++slot) {
                words += fmt::format(" {:#x}", memory.Read32(block + page * 0x1000 + slot * 0x10));
            }
            Write("queue semaphores at {:#x} page {} (slots 0-7, +0x10 apart):{}",
                  block + page * 0x1000, page, words);
        }
    }
    RunGpuDumpHook();
    Write("==================== end of thread dump ====================");
}

} // namespace Core::LaunchLog
