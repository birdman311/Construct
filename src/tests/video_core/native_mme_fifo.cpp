// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <span>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/common_types.h"
#include "video_core/macro_turing.h"

namespace {

// NVN's render-enable-conditional macro 0x16 (pc 0x142 of the RAM Ocarina of Time's SDK uploads,
// nnSdk+0x68b2c0; the same words in Hyrule Warriors: Age of Imprisonment's SDK). Parameters
// {mode == 0, address high, address low} (nnSdk+0x698100); it reads 6 words of two query reports
// through MME_DMA_READ_FIFOED and writes SET_RENDER_ENABLE_C (0x556).
constexpr std::array<u32, 27> RenderEnableConditional{
    0x00940003U, 0x18c00455U, 0x31c10300U,
    0x8a600003U, 0x18c01856U, 0x318c0300U,
    0x00000007U, 0x18c00400U, 0x359c1f00U,
    0x00000003U, 0x1d000000U, 0x31c08300U,
    0x00000003U, 0x18c00000U, 0x3bcc0300U,
    0x00000003U, 0xa003c000U, 0x3810bf00U,
    0x00000003U, 0x18c00000U, 0x0010bb00U,
    0x0018155bU, 0x18c00000U, 0x72213c29U,
    0x00200003U, 0x18c00000U, 0x302c0300U,
};

struct FifoMachine {
    std::array<u32, 0x1000> registers{};
    std::vector<std::pair<u32, u32>> writes;
    std::vector<std::pair<u64, size_t>> reads;
    std::array<u32, 6> memory{};

    template <typename Code>
    Tegra::TuringMacroInterpreter::Result Run(const Code& code, std::span<const u32> parameters) {
        Tegra::TuringMacroInterpreter interpreter;
        return interpreter.Execute(code, 0, parameters,
            [&](u32 method) { return registers[method]; },
            [&](u32 method, u32 value) {
                registers[method] = value;
                writes.emplace_back(method, value);
                return true;
            },
            [&](u64 address, std::span<u32> words) {
                reads.emplace_back(address, words.size());
                for (size_t i = 0; i < words.size(); ++i) words[i] = i < memory.size() ? memory[i] : 0;
                return true;
            });
    }
};

} // Anonymous namespace

TEST_CASE("Native MME load barrier feeds MME_DMA_READ_FIFOED words to later loads", "[native-gpu]") {
    struct Case {
        u32 first;
        u64 counter0, counter1;
        u32 render_enable;
    };
    // The macro writes its first parameter when the report counters at +0 and +16 match, else
    // the inverse (Owl.nss+0x23e36c passes mode 1, first = 0: render when they differ).
    for (const auto& test : {Case{0, 5, 9, 1}, Case{0, 7, 7, 0}, Case{1, 7, 7, 1}, Case{1, 5, 9, 0},
                             Case{0, 0x100000000ULL, 0, 1}}) {
        FifoMachine machine;
        machine.memory = {u32(test.counter0), u32(test.counter0 >> 32), 0x1234, 0x5678,
                          u32(test.counter1), u32(test.counter1 >> 32)};
        const std::array<u32, 3> parameters{test.first, 0x21, 0x30000020};
        const auto result = machine.Run(RenderEnableConditional, parameters);
        REQUIRE(result.success);
        REQUIRE(result.consumed == 9);
        REQUIRE(machine.reads == std::vector<std::pair<u64, size_t>>{{0x2130000020ULL, 6}});
        REQUIRE(machine.writes == std::vector<std::pair<u32, u32>>{
            {0x154, 0x21}, {0x155, 0x30000020}, {0x158, 6}, {0x556, test.render_enable}});
    }
}

TEST_CASE("Native MME FIFO address uses SET_MME_MEM_ADDRESS_A bits 16:0", "[native-gpu]") {
    FifoMachine machine;
    const std::array<u32, 3> parameters{1, 0xfffe0021U, 0x40};
    const auto result = machine.Run(RenderEnableConditional, parameters);
    REQUIRE(result.success);
    REQUIRE(machine.reads == std::vector<std::pair<u64, size_t>>{{0x2100000040ULL, 6}});
}

TEST_CASE("Native MME load barrier failures stay failures", "[native-gpu]") {
    Tegra::TuringMacroInterpreter interpreter;
    const auto read = [](u32) { return 0U; };
    const auto emit = [](u32, u32) { return true; };
    const std::array<u32, 3> parameters{1, 0x21, 0x40};
    // Without a memory reader the pending FIFO read cannot be served.
    auto result = interpreter.Execute(RenderEnableConditional, 0, parameters, read, emit);
    REQUIRE_FALSE(result.success);
    REQUIRE(result.pc == 2);
    REQUIRE(result.consumed == 3);
    REQUIRE(result.error == "MME DMA FIFO read failed");
    // A failed memory read fails the macro before any FIFO word is loaded.
    result = interpreter.Execute(RenderEnableConditional, 0, parameters, read, emit,
                                 [](u64, std::span<u32>) { return false; });
    REQUIRE_FALSE(result.success);
    REQUIRE(result.error == "MME DMA FIFO read failed");
    // EXTENDED operands other than the load barrier's (0x1000, 1) are not known.
    auto other = RenderEnableConditional;
    other[6] = 0x0000000bU; // second immediate 2
    result = interpreter.Execute(other, 0, parameters, read, emit,
                                 [](u64, std::span<u32>) { return true; });
    REQUIRE_FALSE(result.success);
    REQUIRE(result.error == "unsupported extended operation");
}

TEST_CASE("Native MME load barrier without a pending FIFO read keeps the parameters", "[native-gpu]") {
    // EXTENDED 0x1000, 1 with END, then a no-op delay slot.
    constexpr std::array<u32, 6> code{
        0x00000007U, 0x18c00400U, 0x359c1f01U,
        0x00000003U, 0x18c00000U, 0x318c0300U,
    };
    Tegra::TuringMacroInterpreter interpreter;
    u32 fifo_reads{};
    const auto result = interpreter.Execute(code, 0, std::span<const u32>{}, [](u32) { return 0U; },
        [](u32, u32) { return true; },
        [&](u64, std::span<u32>) { ++fifo_reads; return true; });
    REQUIRE(result.success);
    REQUIRE(result.consumed == 0);
    REQUIRE(fifo_reads == 0);
}
