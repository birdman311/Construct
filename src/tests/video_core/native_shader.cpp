// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "video_core/texture_cache/format_lookup_table.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/frontend/maxwell/instruction.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace {
namespace IR = Shader::IR;
using Native = Shader::Ampere::Instruction;

class EmptyEnvironment : public Shader::Environment {
public:
    u64 ReadInstruction(u32) override { throw std::logic_error("unexpected instruction read"); }
    u32 ReadCbufValue(u32, u32) override { throw std::logic_error("unexpected constant folding"); }
    Shader::TextureType ReadTextureType(u32) override { return {}; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override { return {}; }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return {}; }
    void Dump(u64, u64) override {}
};

constexpr std::array<Native, 7> VertexTransform{{
    {{0xff007321, 0x000080ff, 0x00000400, 0x000e2200}}, // load two input components
    {{0x00037a02, 0x00c00000, 0x00000f00, 0x000fe400}}, // scale X
    {{0x00027a02, 0x00c00100, 0x00000f00, 0x000fe200}}, // scale Y
    {{0x00007623, 0x00c00200, 0x00010003, 0x001fe200}}, // X * scale X + offset X
    {{0x01017623, 0x00c00300, 0x00010002, 0x000fc400}}, // Y * scale Y + offset Y
    {{0xff007322, 0x00007000, 0x000004ff, 0x000fe200}}, // store position X/Y
    {{0x0000794d, 0x00000000, 0x03800000, 0x000fea00}},
}};

// Evaluate the existing frontend's emitted IR, independently of native bit
// decoding, to check actual attribute/constant/register data flow.
std::array<float, 2> EvaluateVertex(IR::Block& block, std::array<float, 2> input,
                                  std::array<float, 4> constants, s32 vertex_id = 0) {
    std::unordered_map<const IR::Inst*, u32> values;
    std::unordered_map<const IR::Inst*, std::array<u32, 2>> pairs;
    std::array<u32, 256> regs{};
    std::array<float, 2> output{};
    const auto get = [&](const IR::Value& value) {
        if (!value.IsImmediate()) return values.at(value.Inst());
        if (value.Type() == IR::Type::U1) return static_cast<u32>(value.U1());
        return value.Type() == IR::Type::F32 ? std::bit_cast<u32>(value.F32()) : value.U32();
    };
    for (const auto& inst : block.Instructions()) {
        const auto arg = [&](size_t i) { return get(inst.Arg(i)); };
        switch (inst.GetOpcode()) {
        case IR::Opcode::GetRegister:
            values[&inst] = regs[static_cast<u32>(inst.Arg(0).Reg())];
            break;
        case IR::Opcode::SetRegister:
            if (inst.Arg(0).Reg() != IR::Reg::RZ) regs[static_cast<u32>(inst.Arg(0).Reg())] = arg(1);
            break;
        case IR::Opcode::GetAttributeU32:
            REQUIRE(static_cast<u32>(inst.Arg(0).Attribute()) == 0x2fc / 4);
            values[&inst] = static_cast<u32>(vertex_id);
            break;
        case IR::Opcode::ShiftLeftLogical32:
            values[&inst] = arg(0) << arg(1);
            break;
        case IR::Opcode::ShiftRightArithmetic32:
            values[&inst] = static_cast<u32>(static_cast<s32>(arg(0)) >> arg(1));
            break;
        case IR::Opcode::SLessThan:
            values[&inst] = static_cast<s32>(arg(0)) < static_cast<s32>(arg(1));
            break;
        case IR::Opcode::ULessThan:
            values[&inst] = arg(0) < arg(1);
            break;
        case IR::Opcode::SelectU32:
            values[&inst] = arg(0) ? arg(1) : arg(2);
            break;
        case IR::Opcode::SMin32:
            values[&inst] = static_cast<u32>(std::min(static_cast<s32>(arg(0)), static_cast<s32>(arg(1))));
            break;
        case IR::Opcode::SMax32:
            values[&inst] = static_cast<u32>(std::max(static_cast<s32>(arg(0)), static_cast<s32>(arg(1))));
            break;
        case IR::Opcode::INeg32:
            values[&inst] = 0U - arg(0);
            break;
        case IR::Opcode::IAdd32:
            values[&inst] = arg(0) + arg(1);
            break;
        case IR::Opcode::ConvertF32S32:
            values[&inst] = std::bit_cast<u32>(static_cast<float>(static_cast<s32>(arg(0))));
            break;
        case IR::Opcode::GetAttribute: {
            const auto attr = static_cast<u32>(inst.Arg(0).Attribute());
            if (attr == 0x2fc / 4) {
                values[&inst] = static_cast<u32>(vertex_id);
                break;
            }
            REQUIRE(attr >= 0x80 / 4);
            REQUIRE(attr < 0x88 / 4);
            values[&inst] = std::bit_cast<u32>(input[attr - 0x80 / 4]);
            break;
        }
        case IR::Opcode::GetCbufU32x2:
            REQUIRE(arg(0) == 3);
            REQUIRE(arg(1) % 4 == 0);
            REQUIRE(arg(1) + 8 <= sizeof(constants));
            pairs[&inst] = {std::bit_cast<u32>(constants[arg(1) / 4]), std::bit_cast<u32>(constants[arg(1) / 4 + 1])};
            break;
        case IR::Opcode::CompositeExtractU32x2:
            values[&inst] = pairs.at(inst.Arg(0).Inst()).at(arg(1));
            break;
        case IR::Opcode::GetCbufF32:
        case IR::Opcode::GetCbufU32:
            REQUIRE(arg(0) == 3);
            REQUIRE(arg(1) < 16);
            values[&inst] = std::bit_cast<u32>(constants[arg(1) / 4]);
            break;
        case IR::Opcode::BitCastF32U32:
        case IR::Opcode::BitCastU32F32:
            values[&inst] = arg(0);
            break;
        case IR::Opcode::LogicalAnd:
            values[&inst] = arg(0) && arg(1);
            break;
        case IR::Opcode::LogicalOr:
            values[&inst] = arg(0) || arg(1);
            break;
        case IR::Opcode::LogicalNot:
            values[&inst] = !arg(0);
            break;
        case IR::Opcode::FPIsNan32:
            values[&inst] = std::isnan(std::bit_cast<float>(arg(0)));
            break;
        case IR::Opcode::FPOrdGreaterThanEqual32:
            values[&inst] = std::bit_cast<float>(arg(0)) >= std::bit_cast<float>(arg(1));
            break;
        case IR::Opcode::FPRoundEven32:
            values[&inst] = std::bit_cast<u32>(std::nearbyint(std::bit_cast<float>(arg(0))));
            break;
        case IR::Opcode::FPFloor32:
            values[&inst] = std::bit_cast<u32>(std::floor(std::bit_cast<float>(arg(0))));
            break;
        case IR::Opcode::FPTrunc32:
            values[&inst] = std::bit_cast<u32>(std::trunc(std::bit_cast<float>(arg(0))));
            break;
        case IR::Opcode::FPClamp32:
            values[&inst] = std::bit_cast<u32>(std::clamp(std::bit_cast<float>(arg(0)),
                std::bit_cast<float>(arg(1)), std::bit_cast<float>(arg(2))));
            break;
        case IR::Opcode::ConvertU32F32:
            values[&inst] = static_cast<u32>(std::bit_cast<float>(arg(0)));
            break;
        case IR::Opcode::ConvertF32U32:
            values[&inst] = std::bit_cast<u32>(static_cast<float>(arg(0)));
            break;
        case IR::Opcode::BitFieldUExtract:
            values[&inst] = (arg(0) >> arg(1)) & ((1U << arg(2)) - 1);
            break;
        case IR::Opcode::BitFieldSExtract:
            values[&inst] = static_cast<u32>(static_cast<s32>(arg(0) << (32 - arg(1) - arg(2))) >> (32 - arg(2)));
            break;
        case IR::Opcode::BitwiseOr32:
            values[&inst] = arg(0) | arg(1);
            break;
        case IR::Opcode::FPOrdLessThan32:
            values[&inst] = std::bit_cast<float>(arg(0)) < std::bit_cast<float>(arg(1));
            break;
        case IR::Opcode::BitwiseXor32:
            values[&inst] = arg(0) ^ arg(1);
            break;
        case IR::Opcode::BitwiseNot32:
            values[&inst] = ~arg(0);
            break;
        case IR::Opcode::BitwiseAnd32:
            values[&inst] = arg(0) & arg(1);
            break;
        case IR::Opcode::FPAdd32:
            values[&inst] = std::bit_cast<u32>(std::bit_cast<float>(arg(0)) + std::bit_cast<float>(arg(1)));
            break;
        case IR::Opcode::FPMul32:
            values[&inst] = std::bit_cast<u32>(std::bit_cast<float>(arg(0)) * std::bit_cast<float>(arg(1)));
            break;
        case IR::Opcode::FPMin32:
            values[&inst] = std::bit_cast<u32>(std::fmin(std::bit_cast<float>(arg(0)), std::bit_cast<float>(arg(1))));
            break;
        case IR::Opcode::FPMax32:
            values[&inst] = std::bit_cast<u32>(std::fmax(std::bit_cast<float>(arg(0)), std::bit_cast<float>(arg(1))));
            break;
        case IR::Opcode::SelectF32:
            values[&inst] = arg(0) ? arg(1) : arg(2);
            break;
        case IR::Opcode::FPFma32: {
            const auto control = inst.Flags<IR::FpControl>();
            REQUIRE(control.fmz_mode == IR::FmzMode::FTZ);
            REQUIRE(control.rounding == IR::FpRounding::RN);
            values[&inst] = std::bit_cast<u32>(std::fma(std::bit_cast<float>(arg(0)),
                std::bit_cast<float>(arg(1)), std::bit_cast<float>(arg(2))));
            break;
        }
        case IR::Opcode::SetAttribute: {
            const auto attr = static_cast<u32>(inst.Arg(0).Attribute());
            REQUIRE(attr >= 0x70 / 4);
            REQUIRE(attr < 0x78 / 4);
            output[attr - 0x70 / 4] = std::bit_cast<float>(arg(1));
            break;
        }
        default:
            FAIL("Unexpected IR operation " << IR::NameOf(inst.GetOpcode()));
        }
    }
    return output;
}
} // namespace

TEST_CASE("Native vertex lowering preserves attribute transforms through existing shader IR", "[native-shader]") {
    const auto lowered = Shader::Ampere::LowerToMaxwell(VertexTransform);
    REQUIRE(lowered);
    REQUIRE(lowered.source_size == VertexTransform.size() * 16);
    REQUIRE(lowered.instructions.size() >= VertexTransform.size());
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const u64 word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::FFMA_rc: visitor.FFMA_rc(word); break;
        case Opcode::FFMA_reg: visitor.FFMA_reg(word); break;
        case Opcode::MOV_reg: visitor.MOV_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected lowered vertex instruction");
        }
    }
    REQUIRE(EvaluateVertex(block, {2, 4}, {3, 7, 5, 11}) == std::array<float, 2>{11, 39});
    REQUIRE(EvaluateVertex(block, {-4, 0.5f}, {-2, 4, 1, -3}) == std::array<float, 2>{9, -1});
}

TEST_CASE("Native lowering rejects unsupported forms without a partial executable", "[native-shader]") {
    auto code = VertexTransform;
    code[3].words[2] |= 1U << 12; // DNZ has no accepted lowering yet.
    auto result = Shader::Ampere::LowerToMaxwell(code);
    REQUIRE_FALSE(result);
    REQUIRE(result.error_offset == 3 * 16);
    REQUIRE(result.instructions.empty());
    code = VertexTransform;
    code[1].words[2] &= ~(15U << 8); // Partial quad-lane mask.
    result = Shader::Ampere::LowerToMaxwell(code);
    REQUIRE_FALSE(result);
    REQUIRE(result.instructions.empty());
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell({}));
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(std::span{VertexTransform}.first(6)));
}

TEST_CASE("Native instruction predicates survive lowering", "[native-shader]") {
    auto code = VertexTransform;
    code[3].words[0] = (code[3].words[0] & ~0xf000U) | 0x9000U;
    const auto result = Shader::Ampere::LowerToMaxwell(code);
    REQUIRE(result);
    const auto pred = Shader::Maxwell::Instruction{result.instructions[result.source_instruction_offsets[3]]}.Pred();
    REQUIRE(pred.index == 1);
    REQUIRE(pred.negated);
}

// These private input fixtures are read from the investigation directory, not
// embedded or distributed with the emulator's source tree.
TEST_CASE("Captured native shader pair compiles through existing control flow and shader IR", "[.native-shader-captured]") {
    const char* directory = std::getenv("EDEN_NATIVE_SHADER_FIXTURES");
    REQUIRE(directory != nullptr);
    class FixtureEnvironment final : public EmptyEnvironment {
    public:
        FixtureEnvironment(const std::vector<u64>& code_, std::span<const u8> header, Shader::Stage stage_)
            : code{code_} {
            std::memcpy(&sph, header.data(), sizeof(sph));
            stage = stage_;
            start_address = 0;
            is_proprietary_driver = true;
        }
        u64 ReadInstruction(u32 address) override { return code.at(address / 8); }
        u32 TranslateProgramAddress(u32 address) const override {
            const auto translated = Shader::Ampere::NativeProgramAddress(code, address);
            if (!translated) throw std::logic_error("invalid private native branch target " + std::to_string(address));
            return *translated;
        }
        u32 LocalMemorySize() const override { return static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size; }
        u32 TextureBoundBuffer() const override { return 2; }
        u32 ReadCbufValue(u32 bank, u32 offset) override {
            if (bank == 1) {
                REQUIRE(offset % 4 == 0);
                return cbuf1.at(offset / 4);
            }
            REQUIRE(bank == 2);
            return 0; // Synthetic texture handle; only compilation is tested here.
        }
        Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
        const std::vector<u64>& code;
        std::vector<u32> cbuf1;
    };
    for (const int stage : {1, 5}) {
        const auto path = std::filesystem::path{directory} / ("native-graphics-shader-" + std::to_string(stage) + ".bin");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        REQUIRE(file.good());
        const auto size = static_cast<size_t>(file.tellg());
        REQUIRE(size >= 0x90);
        std::vector<u8> bytes(size);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        std::vector<Native> input((size - 0x80) / 16);
        std::memcpy(input.data(), bytes.data() + 0x80, input.size() * 16);
        const auto lowered = Shader::Ampere::LowerToMaxwell(input);
        INFO("stage=" << stage << " offset=" << lowered.error_offset << " error=" << lowered.error);
        REQUIRE(lowered);
        Shader::Ampere::ProgramCaptureState capture;
        u32 captured_size{};
        for (u32 i = 0; i < input.size(); ++i) {
            const auto status = capture.Observe(input[i], i);
            REQUIRE(status != Shader::Ampere::ProgramCaptureState::Result::InvalidTarget);
            if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) {
                captured_size = (i + 1) * sizeof(Native);
                break;
            }
        }
        REQUIRE(captured_size == lowered.source_size);
        std::vector<u64> code(0x50 / 8);
        for (const u64 word : lowered.instructions) {
            if ((code.size() * 8) % 32 == 0) code.push_back(0);
            code.push_back(word);
        }
        code.push_back(0);
        Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
        FixtureEnvironment env{code, std::span{bytes}.first(0x50), stage == 1 ? Shader::Stage::VertexB : Shader::Stage::Fragment};
        auto constants_path = std::filesystem::path{directory} / ("cbuf1-" + std::to_string(stage) + ".bin");
        if (!std::filesystem::exists(constants_path)) constants_path = std::filesystem::path{directory} / "cbuf1.bin";
        std::ifstream constants(constants_path, std::ios::binary | std::ios::ate);
        if (constants) {
            env.cbuf1.resize(static_cast<size_t>(constants.tellg()) / 4);
            constants.seekg(0);
            constants.read(reinterpret_cast<char*>(env.cbuf1.data()), env.cbuf1.size() * 4);
        }
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::ObjectPool<IR::Inst> inst_pool;
        Shader::ObjectPool<IR::Block> block_pool;
        Shader::Maxwell::Flow::CFG cfg{env, flow_pool, 0x50};
        Shader::HostTranslateInfo host;
        host.support_int64 = true;
        host.support_float64 = true;
        host.support_float16 = true;
        const auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
        REQUIRE(program.stage == env.ShaderStage());
        REQUIRE_FALSE(program.blocks.empty());
        REQUIRE_FALSE(program.post_order_blocks.empty());
    }
}

TEST_CASE("Native scalar texture sampling preserves coordinates and split outputs", "[native-shader]") {
    std::array<Native, 2> input{{
        {{0x04007b60, 0x30800805, 0x041e0f02, 0x100fc200}},
        VertexTransform.back(),
    }};
    const bool predicated = GENERATE(false, true);
    if (predicated) input[0].words[0] = (input[0].words[0] & ~0xf000U) | 0x3000;
    const bool explicit_lod = GENERATE(false, true);
    if (explicit_lod) input[0].words[2] |= 1U << 23;
    const auto result = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(result);
    REQUIRE(result.required_texture_buffer == 2);
    REQUIRE(Shader::Maxwell::Instruction{result.instructions.front()}.Pred().index == (predicated ? 3U : 7U));
    REQUIRE_FALSE(Shader::Maxwell::Instruction{result.instructions.front()}.Pred().negated);
    REQUIRE(Shader::Maxwell::Decode(result.instructions.front()) == Shader::Maxwell::Opcode::TEXS);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TEXS(result.instructions.front());
    std::vector<u32> reads;
    std::vector<u32> writes;
    u32 samples = 0;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::GetRegister) reads.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        if (inst.GetOpcode() == IR::Opcode::SetRegister) writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        if (inst.GetOpcode() == (explicit_lod ? IR::Opcode::BoundImageSampleExplicitLod : IR::Opcode::BoundImageSampleImplicitLod)) {
            ++samples;
            if (explicit_lod) REQUIRE(inst.Arg(2).F32() == 0.0f);
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructF32x2);
            for (size_t component = 0; component < 2; ++component) {
                const auto* cast = coords->Arg(component).Inst();
                REQUIRE(cast->GetOpcode() == IR::Opcode::BitCastF32U32);
                const auto* reg = cast->Arg(0).Inst();
                REQUIRE(reg->GetOpcode() == IR::Opcode::GetRegister);
                REQUIRE(static_cast<u32>(reg->Arg(0).Reg()) == 4 + component);
            }
            REQUIRE(inst.Arg(0).U32() == 0x20);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type == Shader::TextureType::Color2D);
        }
    }
    REQUIRE(samples == 1);
    std::ranges::sort(reads);
    REQUIRE(reads == std::vector<u32>{4, 5});
    REQUIRE(writes == std::vector<u32>{0, 1, 2, 3});
    // Reject unsupported LOD modes, a different cbuf bank, and sparse results.
    for (u32 bit : {89U, 54U, 81U}) {
        auto unsupported = input;
        unsupported[0].words[bit / 32] ^= 1U << (bit % 32);
        const auto rejected = Shader::Ampere::LowerToMaxwell(unsupported);
        REQUIRE_FALSE(rejected);
        REQUIRE(rejected.instructions.empty());
    }
}

TEST_CASE("Native texel fetch keeps integer coordinates and RGB outputs", "[native-shader]") {
    std::array<Native, 3> input{{
        {{0x00027305, 0x00000004, 0x0021f100, 0x001ff000}},
        {{0x02007b66, 0x30800a05, 0x049e0702, 0x101fc200}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.required_texture_buffer == 2);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[0]) == Shader::Maxwell::Opcode::F2I_reg);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[1]) == Shader::Maxwell::Opcode::TLDS);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.F2I_reg(lowered.instructions[0]);
    visitor.TLDS(lowered.instructions[1]);
    std::vector<u32> writes;
    u32 fetches = 0;
    bool truncates = false;
    bool handles_nan = false;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::SetRegister) writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        if (inst.GetOpcode() == IR::Opcode::FPTrunc32) {
            truncates = true;
            REQUIRE(inst.Flags<IR::FpControl>().fmz_mode == IR::FmzMode::FTZ);
        }
        if (inst.GetOpcode() == IR::Opcode::FPIsNan32) handles_nan = true;
        if (inst.GetOpcode() == IR::Opcode::BoundImageFetch) {
            ++fetches;
            REQUIRE(inst.Arg(0).U32() == 0x28);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type == Shader::TextureType::Color2D);
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructU32x2);
            REQUIRE(coords->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{2});
            REQUIRE(coords->Arg(1).Inst()->Arg(0).Reg() == IR::Reg{5});
            REQUIRE(inst.Arg(3).U32() == 0);
        }
    }
    REQUIRE(fetches == 1);
    REQUIRE(truncates);
    REQUIRE(handles_nan);
    REQUIRE(writes == std::vector<u32>{2, 0, 1, 2});
    auto unsupported = input;
    unsupported[0].words[2] ^= 1U << 13; // NTZ is required by this conversion lowering.
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
    unsupported = input;
    unsupported[1].words[2] ^= (1U << 14) | (1U << 12); // multisample fetch with offsets is not implemented here.
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
}

TEST_CASE("Native texture component masks preserve sampled components", "[native-shader]") {
    for (const u32 mask : {1U, 2U, 4U, 8U, 3U, 9U, 10U, 12U, 7U, 11U, 13U, 14U, 15U}) {
        const bool split = std::popcount(mask) > 2;
        std::array<Native, 2> input{{
            {{0x04007b60, 0x30800805, 0x041e0f02, 0x100fc200}}, VertexTransform.back(),
        }};
        input[0].words[2] = (input[0].words[2] & ~0xfffU) | (split ? 6U : 255U) | (mask << 8);
        const auto lowered = Shader::Ampere::LowerToMaxwell(input);
        REQUIRE(lowered);
        Shader::ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        EmptyEnvironment env;
        Shader::Maxwell::TranslatorVisitor visitor{env, block};
        visitor.TEXS(lowered.instructions.front());
        std::vector<u32> extracted;
        std::vector<u32> writes;
        for (const auto& inst : block.Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::CompositeExtractF32x4) extracted.push_back(inst.Arg(1).U32());
            if (inst.GetOpcode() == IR::Opcode::SetRegister) writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        }
        std::vector<u32> expected;
        for (u32 component = 0; component < 4; ++component) if (mask & (1U << component)) expected.push_back(component);
        REQUIRE(extracted == expected);
        REQUIRE(writes.size() == expected.size());
        for (u32 i = 0; i < writes.size(); ++i) REQUIRE(writes[i] == (i < 2 ? i : i + 4));
    }
}

TEST_CASE("Native half packing and arithmetic preserve lane order numerically", "[native-shader]") {
    const std::array<Native, 15> input{{
        {{0x0102723e, 0x00000000, 0x000000ff, 0}}, // PACK_AB R2, R1, R0
        {{0x02027232, 0x00000002, 0x00000000, 0}}, // HMUL2 R2, R2, R2
        {{0xff037230, 0x20000002, 0x00004100, 0}}, // extract low half to F32 R3
        {{0xff047230, 0x30000002, 0x00004100, 0}}, // extract high half to F32 R4
        {{0xff05723e, 0x00000000, 0x00004002, 0}}, // MERGE_C R5, R0, R2
        {{0x02067230, 0x20000002, 0x00000800, 0}}, // add two low halves: (2,2)
        {{0x02077231, 0x20000002, 0x00040802, 0}}, // fused 1*1+1: (2,2)
        {{0x02087240, 0x30000002, 0x03800800, 0}}, // min of low/high: (1,1)
        {{0x02097840, 0x3c003c00, 0x07800c00, 0}}, // max of high/one: (4,4)
        {{0x000a7308, 0x00000008, 0x00001200, 0}}, // reciprocal of half one
        {{0x020b7816, 0x00005410, 0x00000007, 0}}, // concatenate low halves
        {{0xff0c7230, 0xa0000008, 0x00004100, 0}}, // extract negated low half
        {{0x020d7230, 0x30000002, 0x00000f00, 0}}, // -abs(high) + high: (0,0)
        {{0x020e7431, 0x3c003c00, 0x00400802, 0}}, // (1,1) * (1,-4) + (1,1)
        VertexTransform.back(),
    }};
    auto code = input;
    const u32 selector = GENERATE(0x5410U, 0x5432U, 0x7610U, 0x7632U);
    code[10].words[1] = selector;
    if (selector == 0x7610 || selector == 0x7632) code[10].words[2] = 2;
    const auto lowered = Shader::Ampere::LowerToMaxwell(code);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const u64 word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::F2F_reg: visitor.F2F_reg(word); break;
        case Opcode::IADD_reg: visitor.IADD_reg(word); break;
        case Opcode::ISCADD_reg: visitor.ISCADD_reg(word); break;
        case Opcode::HMUL2_reg: visitor.HMUL2_reg(word); break;
        case Opcode::HADD2_reg: visitor.HADD2_reg(word); break;
        case Opcode::HFMA2_reg: visitor.HFMA2_reg(word); break;
        case Opcode::FMNMX_reg: visitor.FMNMX_reg(word); break;
        case Opcode::MUFU: visitor.MUFU(word); break;
        case Opcode::MOV32I: visitor.MOV32I(word); break;
        case Opcode::SHR_imm: visitor.SHR_imm(word); break;
        case Opcode::LOP32I: visitor.LOP32I(word); break;
        case Opcode::MOV_reg: visitor.MOV_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected canonical half instruction");
        }
    }
    // Independent values with exact binary16 representations exercise packing,
    // multiplication and extraction through the actual emitted IR data flow.
    const std::array<std::pair<u16, float>, 5> halves{{{u16{0x4200}, 3.0f}, {u16{0}, 0.0f}, {u16{0x3c00}, 1.0f}, {u16{0x4000}, 2.0f}, {u16{0x4400}, 4.0f}}};
    const auto from_half = [&](u16 bits) {
        for (const auto& [encoded, value] : halves) if (encoded == (bits & 0x7fff)) return (bits & 0x8000) ? -value : value;
        throw std::logic_error("unexpected half value");
    };
    const auto to_half = [&](float value) {
        for (const auto& [encoded, reference] : halves) if (std::abs(value) == reference) return static_cast<u16>(encoded | (std::signbit(value) ? 0x8000 : 0));
        throw std::logic_error("unexpected float value");
    };
    std::array<u32, 256> regs{};
    regs[0] = std::bit_cast<u32>(1.0f);
    regs[1] = std::bit_cast<u32>(2.0f);
    std::unordered_map<const IR::Inst*, u64> values;
    const auto get = [&](const IR::Value& value) -> u64 {
        if (!value.IsImmediate()) return values.at(value.Inst());
        if (value.Type() == IR::Type::U1) return value.U1();
        if (value.Type() == IR::Type::F32) return std::bit_cast<u32>(value.F32());
        if (value.Type() == IR::Type::U16 || value.Type() == IR::Type::F16) return value.U16();
        return value.U32();
    };
    for (const auto& inst : block.Instructions()) {
        const auto arg = [&](size_t i) { return get(inst.Arg(i)); };
        switch (inst.GetOpcode()) {
        case IR::Opcode::GetRegister: values[&inst] = regs[static_cast<u32>(inst.Arg(0).Reg())]; break;
        case IR::Opcode::SetRegister: regs[static_cast<u32>(inst.Arg(0).Reg())] = static_cast<u32>(arg(1)); break;
        case IR::Opcode::BitCastF32U32:
        case IR::Opcode::BitCastU32F32: values[&inst] = arg(0); break;
        case IR::Opcode::ConvertF16F32: values[&inst] = to_half(std::bit_cast<float>(static_cast<u32>(arg(0)))); break;
        case IR::Opcode::ConvertF32F16: values[&inst] = std::bit_cast<u32>(from_half(static_cast<u16>(arg(0)))); break;
        case IR::Opcode::CompositeConstructF16x2: values[&inst] = arg(0) | (arg(1) << 32); break;
        case IR::Opcode::CompositeExtractF16x2: values[&inst] = (arg(0) >> (arg(1) * 32)) & 0xffff; break;
        case IR::Opcode::PackFloat2x16: values[&inst] = (arg(0) & 0xffff) | ((arg(0) >> 32) << 16); break;
        case IR::Opcode::UnpackFloat2x16: values[&inst] = (arg(0) & 0xffff) | ((arg(0) >> 16) << 32); break;
        case IR::Opcode::FPMul16: values[&inst] = to_half(from_half(static_cast<u16>(arg(0))) * from_half(static_cast<u16>(arg(1)))); break;
        case IR::Opcode::FPAdd16: values[&inst] = to_half(from_half(static_cast<u16>(arg(0))) + from_half(static_cast<u16>(arg(1)))); break;
        case IR::Opcode::FPFma16: values[&inst] = to_half(std::fma(from_half(static_cast<u16>(arg(0))), from_half(static_cast<u16>(arg(1))), from_half(static_cast<u16>(arg(2))))); break;
        case IR::Opcode::FPNeg16: values[&inst] = arg(0) ^ 0x8000; break;
        case IR::Opcode::FPAbs16: values[&inst] = arg(0) & 0x7fff; break;
        case IR::Opcode::FPAdd32: values[&inst] = std::bit_cast<u32>(std::bit_cast<float>(static_cast<u32>(arg(0))) + std::bit_cast<float>(static_cast<u32>(arg(1)))); break;
        case IR::Opcode::FPMin32: values[&inst] = std::bit_cast<u32>(std::fmin(std::bit_cast<float>(static_cast<u32>(arg(0))), std::bit_cast<float>(static_cast<u32>(arg(1))))); break;
        case IR::Opcode::FPMax32: values[&inst] = std::bit_cast<u32>(std::fmax(std::bit_cast<float>(static_cast<u32>(arg(0))), std::bit_cast<float>(static_cast<u32>(arg(1))))); break;
        case IR::Opcode::FPRecip32: values[&inst] = std::bit_cast<u32>(1.0f / std::bit_cast<float>(static_cast<u32>(arg(0)))); break;
        case IR::Opcode::SelectF32: values[&inst] = arg(0) ? arg(1) : arg(2); break;
        case IR::Opcode::BitwiseAnd32: values[&inst] = arg(0) & arg(1); break;
        case IR::Opcode::BitwiseXor32: values[&inst] = arg(0) ^ arg(1); break;

        case IR::Opcode::ShiftRightLogical32: values[&inst] = static_cast<u32>(arg(0)) >> arg(1); break;
        case IR::Opcode::SLessThan: values[&inst] = static_cast<s32>(arg(0)) < static_cast<s32>(arg(1)); break;
        case IR::Opcode::ULessThan: values[&inst] = static_cast<u32>(arg(0)) < static_cast<u32>(arg(1)); break;
        case IR::Opcode::SelectU32: values[&inst] = arg(0) ? arg(1) : arg(2); break;
        case IR::Opcode::IAdd32: values[&inst] = static_cast<u32>(arg(0) + arg(1)); break;
        case IR::Opcode::ShiftLeftLogical32: values[&inst] = static_cast<u32>(arg(0) << arg(1)); break;
        default: FAIL("Unexpected IR operation in half arithmetic");
        }
    }
    REQUIRE(regs[2] == 0x44003c00);
    REQUIRE(regs[5] == 0x3c003c00);
    REQUIRE(regs[6] == 0x40004000U);
    REQUIRE(regs[7] == 0x40004000U);
    REQUIRE(regs[8] == 0x3c003c00U);
    REQUIRE(regs[9] == 0x44004400U);
    REQUIRE(regs[10] == 0x3c00U);
    REQUIRE(regs[11] == (selector == 0x7632 ? 0x44004400U : selector == 0x7610 ? 0x44003c00U : selector == 0x5432 ? 0x40004400U : 0x40003c00U));
    REQUIRE(std::bit_cast<float>(regs[12]) == -1.0f);
    REQUIRE(regs[13] == 0U);
    REQUIRE(regs[14] == 0xc2004000U);

    REQUIRE(std::bit_cast<float>(regs[3]) == 1.0f);
    REQUIRE(std::bit_cast<float>(regs[4]) == 4.0f);
}

TEST_CASE("Live shader overrides require the exact original instructions", "[native-shader]") {
    const auto program = Shader::Ampere::LowerToMaxwell(VertexTransform);
    REQUIRE(program);
    const auto bytes = Shader::Ampere::EncodeProgramOverride(VertexTransform, program);
    REQUIRE_FALSE(bytes.empty());
    REQUIRE(Shader::Ampere::ProgramOverrideSourceSize(bytes) == VertexTransform.size() * 16);
    REQUIRE(Shader::Ampere::ProgramOverrideSourceSize(std::span{bytes}.first(12)) == 0);
    auto invalid_source = bytes;
    invalid_source[8] ^= 1;
    REQUIRE(Shader::Ampere::ProgramOverrideSourceSize(invalid_source) == 0);
    const auto decoded = Shader::Ampere::DecodeProgramOverride(bytes, VertexTransform);
    REQUIRE(decoded);
    REQUIRE(decoded.instructions == program.instructions);
    REQUIRE(decoded.source_instruction_offsets == program.source_instruction_offsets);
    auto invalid_map = bytes;
    const u32 invalid_target = static_cast<u32>(program.instructions.size());
    std::memcpy(invalid_map.data() + invalid_map.size() - program.source_instruction_offsets.size() * 4,
                &invalid_target, sizeof(invalid_target));
    const auto invalid_mapping = Shader::Ampere::DecodeProgramOverride(invalid_map, VertexTransform);
    REQUIRE_FALSE(invalid_mapping);
    REQUIRE(invalid_mapping.instructions.empty());

    REQUIRE(decoded.source_size == program.source_size);
    REQUIRE(decoded.required_texture_buffer == program.required_texture_buffer);
    auto different = VertexTransform;
    different[0].words[1] ^= 4;
    REQUIRE(Shader::Ampere::NativeProgramKey(different) != Shader::Ampere::NativeProgramKey(VertexTransform));
    const auto mismatch = Shader::Ampere::DecodeProgramOverride(bytes, different);
    REQUIRE_FALSE(mismatch);
    REQUIRE(mismatch.instructions.empty());
    REQUIRE_FALSE(Shader::Ampere::DecodeProgramOverride(std::span{bytes}.first(12), VertexTransform));
    REQUIRE_FALSE(Shader::Ampere::DecodeProgramOverride(std::span{bytes}.first(bytes.size() - 1), VertexTransform));
    auto wrong_version = bytes;
    wrong_version[4] ^= 1;
    REQUIRE_FALSE(Shader::Ampere::DecodeProgramOverride(wrong_version, VertexTransform));
    auto wrong_binding = bytes;
    const u32 bank = 18;
    std::memcpy(wrong_binding.data() + 16, &bank, sizeof(bank));
    REQUIRE_FALSE(Shader::Ampere::DecodeProgramOverride(wrong_binding, VertexTransform));
}

TEST_CASE("Native integer shader forms preserve signed arithmetic and immediate values", "[native-shader]") {
    // Synthetic vertex program: output (id + arithmetic_shift(id, 1), 4.0).
    const std::array<Native, 7> input{{
        {{0xff007321, 0x0002fcff, 0x00000000, 0x000e2200}},
        {{0xff017819, 0x00000001, 0x00011400, 0x000fe400}},
        {{0x00007824, 0x00000001, 0x078e0201, 0x000fe200}},
        {{0x00007245, 0x00000000, 0x00201400, 0x000fc600}},
        {{0xff017424, 0x40800000, 0x078e00ff, 0x000fe200}},
        {{0xff007322, 0x00007000, 0x000004ff, 0x000fe200}},
        VertexTransform.back(),
    }};
    auto program = input;
    const u32 mode = GENERATE(0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U);
    if (mode == 7) program[2] = Native{{0x00007217, 1, 0x03800200, 0}};
    if (mode == 8) program[2] = Native{{0x00007217, 1, 0x07800200, 0}};
    if (mode == 5) program[2] = Native{{0x00007210, 0x00000001, 0x07ffe1ff, 0}};
    if (mode == 6) program[2] = Native{{0x00007817, 0x0000003f, 0x07800200, 0}};
    if (mode == 4) program[2] = Native{{0x00007817, 0x0000003f, 0x03800200, 0}};
    if (mode == 1) program[2] = Native{{0x00007210, 0x00000001, 0x07ffe0ff, 0x000fe200}};
    if (mode == 3) program[2] = Native{{0x00007210, 0x80000001, 0x07ffe0ff, 0x000fe200}};
    if (mode == 2) program[1] = Native{{0x00017819, 0x00000001, 0x000006ff, 0x000fe200}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(program);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const u64 word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV32I: visitor.MOV32I(word); break;
        case Opcode::SHR_imm: visitor.SHR_imm(word); break;
        case Opcode::SHL_imm: visitor.SHL_imm(word); break;
        case Opcode::IMNMX_reg: visitor.IMNMX_reg(word); break;
        case Opcode::IMNMX_imm: visitor.IMNMX_imm(word); break;
        case Opcode::IADD_reg: visitor.IADD_reg(word); break;
        case Opcode::I2F_reg: visitor.I2F_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected lowered integer shader instruction");
        }
    }
    for (const s32 id : {0, 1, 2, -1, -3, 12345}) {
        REQUIRE(EvaluateVertex(block, {}, {}, id) ==
                std::array<float, 2>{static_cast<float>(mode == 2 ? id * 3 : mode == 3 ? id - (id >> 1) : mode == 4 ? std::min<s32>(id, 63) : mode == 5 ? -id + (id >> 1) : mode == 6 ? std::max<s32>(id, 63) : mode == 7 ? std::min<s32>(id, id >> 1) : mode == 8 ? std::max<s32>(id, id >> 1) : id + (id >> 1)), 4.0f});
    }
}

TEST_CASE("Native single-channel texel fetch writes only its selected result", "[native-shader]") {
    std::array<Native, 2> input{{
        {{0x04017b66, 0x30800805, 0x049e01ff, 0x001fc200}},
        VertexTransform.back(),
    }};
    const u32 component = GENERATE(0U, 1U, 2U, 3U);
    input[0].words[2] = (input[0].words[2] & ~0xf00U) | (1U << (8 + component));
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.required_texture_buffer == 2);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TLDS(lowered.instructions.front());
    std::vector<u32> writes;
    u32 fetches = 0;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::SetRegister)
            writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        if (inst.GetOpcode() == IR::Opcode::BoundImageFetch) {
            ++fetches;
            REQUIRE(inst.Arg(0).U32() == 0x20);
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructU32x2);
            REQUIRE(coords->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{4});
            REQUIRE(coords->Arg(1).Inst()->Arg(0).Reg() == IR::Reg{5});
        }
        if (inst.GetOpcode() == IR::Opcode::CompositeExtractF32x4)
            REQUIRE(inst.Arg(1).U32() == component);
    }
    REQUIRE(fetches == 1);
    REQUIRE(writes == std::vector<u32>{1});
}

TEST_CASE("Native IMAD move loads the actual constant-buffer value", "[native-shader]") {
    const std::array<Native, 3> input{{
        {{0xff007624, 0x00c00100, 0x078e00ff, 0x000fe200}},
        {{0xff007322, 0x00007000, 0x000000ff, 0x000fe200}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions.front()) == Shader::Maxwell::Opcode::MOV_cbuf);
    visitor.MOV_cbuf(lowered.instructions[0]);
    visitor.AST(lowered.instructions[1]);
    REQUIRE(EvaluateVertex(block, {}, {13, -2.5f, 99, 7}) == std::array<float, 2>{-2.5f, 0.0f});
    auto unsupported = input;
    unsupported[0].words[1] |= 1U << 30;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
}

TEST_CASE("Depth-stencil descriptors ignore the unused blue and alpha component types", "[native-gpu]") {
    using Tegra::Texture::ComponentType;
    using Tegra::Texture::TextureFormat;
    using VideoCore::Surface::PixelFormat;
    for (const auto unused : {ComponentType::UNORM, ComponentType::UINT}) {
        REQUIRE(VideoCommon::PixelFormatFromTextureInfo(TextureFormat::S8Z24,
            ComponentType::UNORM, ComponentType::UINT, unused, unused, false) ==
            PixelFormat::D24_UNORM_S8_UINT);
    }
}

TEST_CASE("Native floating min/max clamps actual shader input", "[native-shader]") {
    const std::array<Native, 5> input{{
        {{0xff007321, 0x000080ff, 0x00000000, 0x000e2200}},
        {{0x00007a09, 0x00c00000, 0x07810000, 0x000fe400}},
        {{0x00007a09, 0x00c00100, 0x03810000, 0x000fe400}},
        {{0xff007322, 0x00007000, 0x000000ff, 0x000fe200}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::FMNMX_reg: visitor.FMNMX_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected native min/max lowering");
        }
    }
    for (const float value : {-20.0f, -1.0f, 0.25f, 2.0f, 100.0f})
        REQUIRE(EvaluateVertex(block, {value, 0}, {-1, 2, 0, 0}) ==
                std::array<float, 2>{std::clamp(value, -1.0f, 2.0f), 0});
}

TEST_CASE("Native boolean float comparison preserves register moves and absolute inputs", "[native-shader]") {
    const std::array<Native, 5> input{{
        {{0xff007321, 0x000080ff, 0x00000000, 0}},
        {{0xff017224, 0x000000ff, 0x078e0000, 0}},
        {{0x01007a0a, 0x00c00000, 0x03811200, 0}},
        {{0xff007322, 0x00007000, 0x000000ff, 0}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV_reg: visitor.MOV_reg(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::LOP32I: visitor.LOP32I(word); break;
        case Opcode::FSET_reg: visitor.FSET_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected native boolean float comparison lowering");
        }
    }
    for (const float value : {-3.0f, -1.0f, 0.0f, 2.0f, 4.0f})
        REQUIRE(EvaluateVertex(block, {value, 0}, {2, 0, 0, 0}) ==
                std::array<float, 2>{std::abs(value) < 2.0f ? 1.0f : 0.0f, 0});
}

TEST_CASE("Native unsigned conversion saturates, clears NaNs, and extracts the low byte", "[native-shader]") {
    std::array<Native, 5> input{{
        {{0xff007321, 0x000080ff, 0x00000000, 0}},
        {{0x00017305, 0x00000000, 0x0021f000, 0}},
        {{0x00007306, 0x00000001, 0x00001000, 0}},
        {{0xff007322, 0x00007000, 0x000000ff, 0}},
        VertexTransform.back(),
    }};
    const bool constant_source = GENERATE(false, true);
    if (constant_source) input[1] = Native{{0x00017b05, 0x00c00000, 0x0021f000, 0}};
    const u32 conversion_mode = GENERATE(0U, 1U, 2U);
    const bool short_round = conversion_mode == 1;
    if (conversion_mode == 2) input[1].words[2] = 0x00207000;
    if (short_round) input[1].words[2] = 0x00212800;
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::MOV32I: visitor.MOV32I(word); break;
        case Opcode::FSET_reg: visitor.FSET_reg(word); break;
        case Opcode::FMNMX_reg: visitor.FMNMX_reg(word); break;
        case Opcode::LOP_reg: visitor.LOP_reg(word); break;
        case Opcode::F2I_reg: visitor.F2I_reg(word); break;
        case Opcode::I2F_reg: visitor.I2F_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected unsigned conversion lowering");
        }
    }
    const std::array<std::pair<float, float>, 9> cases{{
        {std::numeric_limits<float>::quiet_NaN(), 0.0f},
        {-std::numeric_limits<float>::infinity(), 0.0f}, {-3.0f, 0.0f}, {1.9f, 1.0f},
        {300.75f, 44.0f}, {65535.0f, 255.0f}, {4294967040.0f, 0.0f},
        {4294967296.0f, 255.0f}, {std::numeric_limits<float>::infinity(), 255.0f},
    }};
    for (const auto& [value, expected] : cases) {
        const float rounded = std::isnan(value) ? 0.0f : static_cast<float>(static_cast<u32>(
            std::clamp(std::nearbyint(static_cast<double>(value)), 0.0, 65535.0)) & 255);
        REQUIRE(EvaluateVertex(block, {value, 0}, {value, 0, 0, 0}) == std::array<float, 2>{short_round ? rounded : expected, 0});
    }
}

TEST_CASE("Native 16-bit and uniform byte integer-to-float conversions keep sign and source part", "[native-shader]") {
    std::array<Native, 5> input{{
        {{0xff007321, 0x000080ff, 0x00000000, 0}}, // R0 = input X bits
        {{0x00047ab9, 0x00c00000, 0x00000800, 0}}, // ULDC UR4, c[3][0]
        {{0x00007306, 0x00000000, 0x00101400, 0}}, // I2F.F32.S16 R0, R0
        {{0xff007322, 0x00007000, 0x000000ff, 0}},
        VertexTransform.back(),
    }};
    // 0-1: I2F.F32.S16 R0, R0.H0/.H1; 2-3: I2F.F32.U16 R0, UR4.H0/.H1; 4-5: the same with S16;
    // 6-9: I2F.F32.U8 R0, UR4.B0-.B3. Bits 60-61 pick the halfword or byte, bit 74 the sign.
    const u32 mode = GENERATE(0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U);
    const u32 part = mode < 2 ? mode : mode < 6 ? (mode - 2) & 1 : mode - 6;
    if (mode >= 2)
        input[2] = Native{{0x00007d06, 0x00000004,
                           mode < 4 ? 0x08101000U : mode < 6 ? 0x08101400U : 0x08001000U, 0}};
    input[2].words[1] |= part << 28;
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::I2F_reg: visitor.I2F_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected 16-bit or uniform byte conversion lowering");
        }
    }
    const bool is_signed = mode < 2 || mode == 4 || mode == 5;
    const u32 width = mode < 6 ? 16 : 8;
    for (const u32 bits : {0x3fff8000U, 0xc001fffeU, 0x4123fedcU, 0x3f807fffU}) {
        const u32 field = (bits >> (part * width)) & ((1U << width) - 1);
        const float expected = is_signed ? static_cast<float>(static_cast<s32>(field << (32 - width)) >> (32 - width))
                                         : static_cast<float>(field);
        const float value = std::bit_cast<float>(bits);
        REQUIRE(EvaluateVertex(block, {value, 0}, {value, 0, 0, 0}) == std::array<float, 2>{expected, 0});
    }
    if (mode < 6) {
        // Bit 61 (no 16-bit meaning known), a source in 24-31, |Rb| and RM rounding stay rejected.
        for (const auto& [word, bit] : {std::pair{1U, 2U << 28}, std::pair{0U, 5U << 24},
                                        std::pair{1U, 1U << 30}, std::pair{2U, 1U << 14}}) {
            auto unsupported = input;
            unsupported[2].words[word] |= bit;
            REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
        }
    }
}

TEST_CASE("Native unsigned 16-bit register conversion keeps its halfword select and rejects unknown bits", "[native-shader]") {
    // I2F.F32.U16 R5, R5.H1 as Hyrule Warriors' programs encode it (00057306 10000005 00101000):
    // Maxwell I2F (0x5cb8) with F32 destination (2 at 8), U16 source (1 at 10), the source at 20
    // and byte selector 2 for .H1 (bit 60). Bit 61 and a register in 24-31, which have no known
    // 16-bit meaning, are rejected (before, bit 61 reached Maxwell bit 43).
    for (const u32 high : {0U, 1U}) {
        std::array<Native, 2> input{{{{0x00057306, 0x00000005U | (high << 28), 0x00101000, 0x000e2200}},
                                     VertexTransform.back()}};
        const auto lowered = Shader::Ampere::LowerToMaxwell(input);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(lowered.instructions[0] == (0x5cb8000000000000ULL | (u64{7} << 16) | 5 | (u64{2} << 8) |
                                            (u64{1} << 10) | (u64{5} << 20) | (u64{high * 2} << 41)));
        for (const auto& [word, bit] : {std::pair{1U, 1U << 29}, std::pair{0U, 3U << 24}}) {
            auto unsupported = input;
            unsupported[0].words[word] |= bit;
            const auto rejected = Shader::Ampere::LowerToMaxwell(unsupported);
            REQUIRE_FALSE(rejected);
            REQUIRE(rejected.error == "native byte-to-float conversion mode");
        }
    }
}

TEST_CASE("Native depth-comparison sample preserves XY coordinates and reference depth", "[native-shader]") {
    const std::array<Native, 2> input{{
        {{0x04017b60, 0x30800806, 0x041e41ff, 0}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TEXS(lowered.instructions.front());
    u32 samples = 0;
    std::vector<u32> writes;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::SetRegister)
            writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        if (inst.GetOpcode() == IR::Opcode::BoundImageSampleDrefImplicitLod) {
            ++samples;
            REQUIRE(inst.Arg(0).U32() == 0x20);
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructF32x2);
            for (u32 i = 0; i < 2; ++i)
                REQUIRE(coords->Arg(i).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{4 + i});
            REQUIRE(inst.Arg(2).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{6});
            REQUIRE(inst.Flags<IR::TextureInstInfo>().is_depth == 1);
        }
    }
    REQUIRE(samples == 1);
    REQUIRE(writes == std::vector<u32>{1});
}

TEST_CASE("Native array texel fetch preserves layer and separates XY registers", "[native-shader]") {
    std::array<Native, 2> input{{
        {{0x06007b66, 0xb0800809, 0x009e0f02, 0}},
        VertexTransform.back(),
    }};
    const bool sparse = GENERATE(false, true);
    if (sparse) input[0].words[2] = 0x009e0b23;
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.MOV_reg(lowered.instructions[0]);
    visitor.MOV_reg(lowered.instructions[1]);
    visitor.TLDS(lowered.instructions[2]);
    u32 fetches = 0;
    std::vector<u32> writes, components;
    std::vector<std::pair<u32, u32>> copies;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::SetRegister && static_cast<u32>(inst.Arg(0).Reg()) < 250)
            writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        if (inst.GetOpcode() == IR::Opcode::CompositeExtractF32x4)
            components.push_back(inst.Arg(1).U32());
        if (inst.GetOpcode() == IR::Opcode::SetRegister &&
            static_cast<u32>(inst.Arg(0).Reg()) >= 250) {
            REQUIRE(inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::GetRegister);
            copies.emplace_back(static_cast<u32>(inst.Arg(0).Reg()),
                                static_cast<u32>(inst.Arg(1).Inst()->Arg(0).Reg()));
        }
        if (inst.GetOpcode() == IR::Opcode::BoundImageFetch) {
            ++fetches;
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type == Shader::TextureType::ColorArray2D);
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructU32x3);
            REQUIRE(coords->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{250});
            REQUIRE(coords->Arg(1).Inst()->Arg(0).Reg() == IR::Reg{251});
            const auto* layer = coords->Arg(2).Inst();
            REQUIRE(layer->GetOpcode() == IR::Opcode::BitFieldUExtract);
            REQUIRE(layer->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{6});
            REQUIRE(layer->Arg(1).U32() == 0);
            REQUIRE(layer->Arg(2).U32() == 16);
        }
    }
    REQUIRE(fetches == 1);
    REQUIRE(writes == (sparse ? std::vector<u32>{0, 1, 35} : std::vector<u32>{0, 1, 2, 3}));
    REQUIRE(components == (sparse ? std::vector<u32>{0, 1, 3} : std::vector<u32>{0, 1, 2, 3}));
    REQUIRE(copies == std::vector<std::pair<u32, u32>>{{250, 7}, {251, 9}});
}

TEST_CASE("Native bitwise LUT forms preserve source bits without changing predicates", "[native-shader]") {
    std::array<Native, 5> input{{
        {{0x00007a02, 0x00c00000, 0x00000f00, 0}},
        {{0x00027a02, 0x00c00200, 0x00000f00, 0}},
        {{0x00007812, 0x80000000, 0x078ef802, 0}},
        {{0xff007322, 0x00007000, 0x000000ff, 0}},
        VertexTransform.back(),
    }};
    const u32 mode = GENERATE(0U, 1U, 2U);
    if (mode == 2) input[2] = Native{{0x00007812, 0x80000000, 0x078ee202, 0}};
    if (mode == 1) input[2] = Native{{0x00007212, 0x00000002, 0x078efcff, 0}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::MOV32I: visitor.MOV32I(word); break;
        case Opcode::LOP_reg: visitor.LOP_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected native bitwise LUT lowering");
        }
    }
    for (const auto& inst : block.Instructions()) REQUIRE(inst.GetOpcode() != IR::Opcode::SetPred);
    for (const float a : {-1.0f, 1.0f}) for (const float c : {-2.0f, 2.0f}) {
        const u32 c_bits = std::bit_cast<u32>(c);
        const u32 a_bits = std::bit_cast<u32>(a);
        const u32 expected = mode == 2 ? (a_bits & 0x80000000U) | (c_bits & 0x7fffffffU) :
            a_bits | (mode == 1 ? c_bits : c_bits & 0x80000000U);
        REQUIRE(EvaluateVertex(block, {}, {a, 0, c, 0}) ==
                std::array<float, 2>{std::bit_cast<float>(expected), 0});
    }
}

TEST_CASE("Native one-dimensional texel fetch preserves its integer coordinate", "[native-shader]") {
    const std::array<Native, 2> input{{
        {{0x04017b66, 0x108008ff, 0x009e01ff, 0}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TLDS(lowered.instructions.front());
    u32 fetches = 0;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageFetch) {
            ++fetches;
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type == Shader::TextureType::Color1D);
            REQUIRE(inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::GetRegister);
            REQUIRE(inst.Arg(1).Inst()->Arg(0).Reg() == IR::Reg{4});
        }
    }
    REQUIRE(fetches == 1);
}

TEST_CASE("Native call and return preserve values through the actual shader frontend", "[native-shader]") {
    std::array<Native, 9> input{{
        {{0x00007802, 0x40000000, 0x00000f00, 0}}, // argument 2.0
        {{0x00377802, 0x00000030, 0x00000f00, 0}}, // native return PC
        {{0x00007944, 0x00000020, 0x03c00000, 0}}, // call offset0x50
        {{0xff007322, 0x00007003, 0x000000ff, 0}},
        VertexTransform.back(),
        {{0x00037421, 0x40400000, 0x00010000, 0}}, // callee: add 3.0
        {{0x00087202, 0x00000037, 0x00000f00, 0}},
        {{0x00097802, 0x00000000, 0x00000f00, 0}},
        {{0x08007950, 0xffffff70, 0x03c3ffff, 0}},
    }};
    const bool alternate_link = GENERATE(false, true);
    if (alternate_link) {
        input[1].words[0] = 0x00397802;
        input[6] = Native{{0x00007202, 0x00000039, 0x00000f00, 0}};
        input[7].words[0] = 0x00017802;
        input[8].words[0] = 0x00007950;
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.source_size == input.size() * 16);
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(std::span{input}.first(5)));
    auto broken = input;
    broken[1].words[1] += 16;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(broken));
    broken = input;
    broken.back().words[1] += 16;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(broken));
    std::vector<u64> code(0x50 / 8);
    for (const auto word : lowered.instructions) {
        if ((code.size() * 8) % 32 == 0) code.push_back(0);
        code.push_back(word);
    }
    class CallEnvironment final : public EmptyEnvironment {
    public:
        explicit CallEnvironment(const std::vector<u64>& code_) : code{code_} { stage = Shader::Stage::VertexB; }
        u64 ReadInstruction(u32 address) override { return code.at(address / 8); }
        const std::vector<u64>& code;
    } env{code};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::ObjectPool<IR::Inst> inst_pool;
    Shader::ObjectPool<IR::Block> block_pool;
    Shader::Maxwell::Flow::CFG cfg{env, flow_pool, 0x50};
    REQUIRE(cfg.Functions().size() == 2);
    Shader::HostTranslateInfo host;
    const auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    u32 stores = 0;
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::SetAttribute) continue;
            ++stores;
            REQUIRE(inst.Arg(1).IsImmediate());
            REQUIRE(inst.Arg(1).F32() == 5.0f);
        }
    }
    REQUIRE(stores == 1);
}

TEST_CASE("Native scaled multiplication preserves captured power-of-two scales", "[native-shader]") {
    std::array<Native, 4> input{{
        VertexTransform.front(),
        {{0x00007220, 0x00000001, 0x00310000, 0}},
        {{0xff007322, 0x00007000, 0x000000ff, 0}},
        VertexTransform.back(),
    }};
    const u32 scale = GENERATE(3U, 4U, 5U, 6U);
    input[1].words[2] = (scale << 20) | 0x10000;
    const float factor = scale == 3 ? 0.5f : scale == 5 ? 2.0f : scale == 6 ? 4.0f : 1.0f;
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.ALD(lowered.instructions[0]);
    visitor.FMUL_reg(lowered.instructions[1]);
    visitor.AST(lowered.instructions[2]);
    REQUIRE(EvaluateVertex(block, {8, 4}, {}) == std::array<float, 2>{32 * factor, 0});
    REQUIRE(EvaluateVertex(block, {-2, 3}, {}) == std::array<float, 2>{-6 * factor, 0});
}

TEST_CASE("Native integer multiply preserves high words and modular multiply-add", "[native-shader]") {
    const u32 mode = GENERATE(0U, 1U, 2U, 3U, 4U);
    Native instruction{{0x00037827, 0xcccccccd, 0x078e00ff, 0}};
    if (mode == 1) instruction = Native{{0x00037824, 0xfffffff6, 0x078e0202, 0}};
    if (mode == 2) instruction = Native{{0x00037a24, 0x00c00000, 0x078e0202, 0}};
    if (mode == 4) instruction = Native{{0x00037811, 0x00000070, 0x078e20ff, 0}};
    if (mode == 3) instruction = Native{{0x00037224, 0x00000001, 0x078e0202, 0}};
    const std::array<Native, 2> input{{instruction, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::XMAD_imm: visitor.XMAD_imm(word); break;
        case Opcode::XMAD_reg: visitor.XMAD_reg(word); break;
        case Opcode::MOV32I: visitor.MOV32I(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::ISCADD_reg: visitor.ISCADD_reg(word); break;
        case Opcode::SHR_imm: visitor.SHR_imm(word); break;
        case Opcode::IADD_reg: visitor.IADD_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected canonical integer multiplication instruction");
        }
    }
    for (u32 a : {0U, 1U, 9U, 10U, 65535U, 65536U, 0x80000000U, 0xfffffffeU, 0xffffffffU, 0x12345678U}) {
        std::array<u32, 256> regs{};
        regs[0] = a;
        regs[1] = 0x76543210;
        regs[2] = 0xfedcba98;
        std::unordered_map<const IR::Inst*, u32> values;
        const auto get = [&](const IR::Value& value) -> u32 {
            if (!value.IsImmediate()) return values.at(value.Inst());
            return value.Type() == IR::Type::U1 ? static_cast<u32>(value.U1()) : value.U32();
        };
        for (const auto& inst : block.Instructions()) {
            const auto arg = [&](size_t i) { return get(inst.Arg(i)); };
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetRegister: values[&inst] = regs[static_cast<u32>(inst.Arg(0).Reg())]; break;
            case IR::Opcode::SetRegister: regs[static_cast<u32>(inst.Arg(0).Reg())] = arg(1); break;
            case IR::Opcode::GetCbufU32:
                REQUIRE(arg(0) == 3);
                REQUIRE(arg(1) == 0);
                values[&inst] = 0x76543210;
                break;
            case IR::Opcode::BitFieldUExtract: values[&inst] = (arg(0) >> arg(1)) & ((1U << arg(2)) - 1); break;
            case IR::Opcode::IMul32: values[&inst] = arg(0) * arg(1); break;
            case IR::Opcode::IAdd32: values[&inst] = arg(0) + arg(1); break;
            case IR::Opcode::ShiftLeftLogical32: values[&inst] = arg(0) << arg(1); break;
            case IR::Opcode::ShiftRightLogical32: values[&inst] = arg(0) >> arg(1); break;
            case IR::Opcode::ULessThan: values[&inst] = arg(0) < arg(1); break;
            case IR::Opcode::SLessThan: values[&inst] = static_cast<s32>(arg(0)) < static_cast<s32>(arg(1)); break;
            case IR::Opcode::SelectU32: values[&inst] = arg(0) ? arg(1) : arg(2); break;
            default: FAIL("Unexpected multiplication IR operation " << IR::NameOf(inst.GetOpcode()));
            }
        }
        const u32 b = mode == 0 ? 0xcccccccd : mode == 1 ? 0xfffffff6 : 0x76543210;
        const u64 product = u64{a} * b;
        const u32 expected = mode == 4 ? (a << 4) + 0x70U : mode == 0 ? static_cast<u32>(product >> 32)
                                       : static_cast<u32>(product + 0xfedcba98U);
        REQUIRE(regs[3] == expected);
    }
}

TEST_CASE("Native uniform constants keep their indexed load and arithmetic values", "[native-shader]") {
    const std::array<Native, 7> input{{
        {{0xff007321, 0x000080ff, 0x00000000, 0}},
        {{0x00047ab9, 0x00c00000, 0x00000800, 0}},
        {{0x04047899, 0x00000002, 0x0800063f, 0}},
        {{0x04057abb, 0x00c00000, 0x08000800, 0}},
        {{0x00007c20, 0x00000005, 0x08410000, 0}},
        {{0x00007e21, 0x80000005, 0x08010000, 0}},
        VertexTransform.back(),
    }};
    auto code = input;
    // Store the computed scalar instead of ending immediately.
    std::vector<Native> program(code.begin(), code.end() - 1);
    program.push_back(Native{{0xff007322, 0x00007000, 0x000000ff, 0}});
    program.push_back(VertexTransform.back());
    const auto lowered = Shader::Ampere::LowerToMaxwell(program);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::MOV_reg: visitor.MOV_reg(word); break;
        case Opcode::LOP32I: visitor.LOP32I(word); break;
        case Opcode::SHL_imm: visitor.SHL_imm(word); break;
        case Opcode::LDC: visitor.LDC(word); break;
        case Opcode::FMUL_reg: visitor.FMUL_reg(word); break;
        case Opcode::FADD_reg: visitor.FADD_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected uniform instruction lowering");
        }
    }
    const std::array<float, 4> constants{std::bit_cast<float>(1U), 3.0f, 7.0f, 9.0f};
    REQUIRE(EvaluateVertex(block, {2, 0}, constants) == std::array<float, 2>{3, 0});
    REQUIRE(EvaluateVertex(block, {4, 0}, constants) == std::array<float, 2>{9, 0});
    REQUIRE(lowered.local_memory_bias == 0);
    auto conflict = program;
    conflict[0].words[0] = 0xfff87321; // Native R248 conflicts with the reserved uniform bank,
    const auto homed = Shader::Ampere::LowerToMaxwell(conflict); // so it is kept in local memory.
    REQUIRE(homed);
    REQUIRE(homed.local_memory_bias == 0x100);
}

TEST_CASE("Native predicate-output LUTs preserve boolean bit tests", "[native-shader]") {
    const std::array<Native, 3> input{{
        {{0x00ff7812, 0x00000001, 0x0780c0ff, 0}},
        {{0x01ff7812, 0x00000001, 0x07824802, 0}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::MOV32I: visitor.MOV32I(word); break;
        case Opcode::LOP_reg: visitor.LOP_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected predicate LUT lowering");
        }
    }
    for (u32 a = 0; a < 4; ++a) for (u32 b = 0; b < 4; ++b) for (u32 c = 0; c < 4; ++c) {
        std::array<u32, 256> regs{};
        regs[0] = a; regs[1] = b; regs[2] = c;
        std::array<bool, 7> predicates{false, false, true, false, true, false, true};
        std::unordered_map<const IR::Inst*, u32> values;
        const auto get = [&](const IR::Value& value) -> u32 {
            if (!value.IsImmediate()) return values.at(value.Inst());
            return value.Type() == IR::Type::U1 ? static_cast<u32>(value.U1()) : value.U32();
        };
        for (const auto& inst : block.Instructions()) {
            const auto arg = [&](size_t i) { return get(inst.Arg(i)); };
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetRegister: values[&inst] = regs[static_cast<u32>(inst.Arg(0).Reg())]; break;
            case IR::Opcode::SetRegister: regs[static_cast<u32>(inst.Arg(0).Reg())] = arg(1); break;
            case IR::Opcode::BitwiseAnd32: values[&inst] = arg(0) & arg(1); break;
            case IR::Opcode::BitwiseOr32: values[&inst] = arg(0) | arg(1); break;
            case IR::Opcode::BitwiseXor32: values[&inst] = arg(0) ^ arg(1); break;
            case IR::Opcode::INotEqual: values[&inst] = arg(0) != arg(1); break;
            case IR::Opcode::SetPred: predicates[static_cast<u32>(inst.Arg(0).Pred())] = arg(1) != 0; break;
            default: FAIL("Unexpected predicate LUT IR " << IR::NameOf(inst.GetOpcode()));
            }
        }
        REQUIRE(predicates[0] == ((a & 1) != 0));
        REQUIRE(predicates[1] == (((b ^ c) & 1) != 0));
        REQUIRE(std::array<bool, 5>{predicates[2], predicates[3], predicates[4], predicates[5], predicates[6]} ==
                std::array<bool, 5>{true, false, true, false, true});
    }
}

TEST_CASE("Native cube sampling preserves XYZ, explicit LOD, and split RGB destinations", "[native-shader]") {
    std::array<Native, 2> input{{
        {{0x04007b60, 0x70800808, 0x059e0703, 0}},
        VertexTransform.back(),
    }};
    const bool volume = GENERATE(false, true);
    if (volume) {
        input[0].words[1] = (input[0].words[1] & ~(7U << 29)) | (2U << 29);
        input[0].words[2] = (input[0].words[2] & ~(7U << 23)) | (1U << 23);
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::MOV_reg: visitor.MOV_reg(word); break;
        case Opcode::TEX: visitor.TEX(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected cube sampling lowering");
        }
    }
    u32 samples = 0;
    std::vector<std::pair<u32, u32>> copies;
    std::vector<u32> writes;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::SetRegister) {
            const u32 target = static_cast<u32>(inst.Arg(0).Reg());
            if (target < 248) writes.push_back(target);
            if (inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::GetRegister)
                copies.emplace_back(target, static_cast<u32>(inst.Arg(1).Inst()->Arg(0).Reg()));
        }
        if (inst.GetOpcode() == IR::Opcode::BoundImageSampleExplicitLod) {
            ++samples;
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type == (volume ? Shader::TextureType::Color3D : Shader::TextureType::ColorCube));
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructF32x3);
            for (u32 i = 0; i < 3; ++i)
                REQUIRE(coords->Arg(i).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{250 + i});
            if (volume) REQUIRE(inst.Arg(2).F32() == 0.0f);
            else REQUIRE(inst.Arg(2).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{9});
        }
    }
    REQUIRE(samples == 1);
    REQUIRE(writes == std::vector<u32>{0, 1, 3});
    REQUIRE(copies == std::vector<std::pair<u32, u32>>{{250, 4}, {251, 5}, {252, 8}, {0, 250}, {1, 251}, {3, 252}});
}

TEST_CASE("Native branch maps survive code caching and reject invalid addresses", "[native-shader]") {
    std::vector<u64> code(64);
    const std::array<u32, 4> offsets{0, 2, 4, 8};
    Shader::Ampere::AppendNativeAddressMap(code, offsets);
    REQUIRE(Shader::Ampere::HasNativeAddressMap(code));
    REQUIRE(Shader::Ampere::NativeProgramAddress(code, 0) == 0x50U);
    REQUIRE(Shader::Ampere::NativeProgramAddress(code, 16) == 0x68U);
    const auto restored = code;
    REQUIRE(Shader::Ampere::NativeProgramAddress(restored, 48) == 0xa8U);
    REQUIRE_FALSE(Shader::Ampere::NativeProgramAddress(code, 1));
    REQUIRE_FALSE(Shader::Ampere::NativeProgramAddress(code, 64));
    code[code.size() - 2] = 100000;
    REQUIRE_FALSE(Shader::Ampere::NativeProgramAddress(code, 0));
}

TEST_CASE("Native computed jumps retain table values while mapping code destinations", "[native-shader]") {
    const std::array<Native, 12> input{{
        {{0xff007321, 0x0002fcff, 0x00000000, 0}},
        {{0x00007817, 0x00000001, 0x03800000, 0}},
        {{0x00027819, 0x00000002, 0x000006ff, 0}},
        {{0x02007b82, 0x00400000, 0x00000800, 0}},
        {{0xff037421, 0x3f000000, 0x00010000, 0}},
        {{0x00007949, 0xffffffa0, 0x0383ffff, 0}},
        {{0x00007802, 0x3f800000, 0x00000f00, 0}},
        {{0x00007947, 0x00000020, 0x03800000, 0}},
        {{0x00007802, 0x40000000, 0x00000f00, 0}},
        {{0x00007947, 0x00000000, 0x03800000, 0}},
        {{0xff007322, 0x00007000, 0x000000ff, 0}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    std::vector<u64> code(0x50 / 8);
    for (const auto word : lowered.instructions) {
        if ((code.size() * 8) % 32 == 0) code.push_back(0);
        code.push_back(word);
    }
    code.push_back(0);
    Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
    class BranchEnvironment final : public EmptyEnvironment {
    public:
        explicit BranchEnvironment(const std::vector<u64>& code_) : code{code_} { stage = Shader::Stage::VertexB; }
        u64 ReadInstruction(u32 address) override { return code.at(address / 8); }
        u32 ReadCbufValue(u32 bank, u32 offset) override {
            REQUIRE(bank == 1);
            REQUIRE((offset == 0 || offset == 4));
            return offset == 0 ? 0x60 : 0x80;
        }
        u32 TranslateProgramAddress(u32 address) const override {
            const auto target = Shader::Ampere::NativeProgramAddress(code, address);
            if (!target) throw std::logic_error("invalid synthetic branch target");
            return *target;
        }
        const std::vector<u64>& code;
    } env{code};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::ObjectPool<IR::Inst> inst_pool;
    Shader::ObjectPool<IR::Block> block_pool;
    Shader::Maxwell::Flow::CFG cfg{env, flow_pool, 0x50};
    u32 jumps = 0;
    for (const auto& block : cfg.Functions().front().blocks) {
        if (block.end_class != Shader::Maxwell::Flow::EndClass::IndirectBranch) continue;
        ++jumps;
        REQUIRE(block.branch_offset == 0);
        REQUIRE(block.indirect_branches.size() == 2);
        for (const auto& branch : block.indirect_branches) {
            REQUIRE((branch.address == 0x60 || branch.address == 0x80));
            REQUIRE(branch.block->begin.Offset() == env.TranslateProgramAddress(branch.address));
            REQUIRE(branch.block->begin.Offset() != branch.address);
        }
    }
    REQUIRE(jumps == 1);
    Shader::HostTranslateInfo host;
    const auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    REQUIRE_FALSE(program.blocks.empty());
}

TEST_CASE("Native forward branches retain paths beyond an early unconditional exit", "[native-shader]") {
    const Native exit{{0x0000794d, 0, 0x03800000, 0x000fea00}};
    const std::array<Native, 4> source{{
        {{0x00000947, 0x20, 0x03800000, 0x000fea00}},
        exit,
        {{0x00017802, 42, 0x00000f00, 0x000fe400}},
        exit,
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(source);
    REQUIRE(lowered);
    REQUIRE(lowered.source_size == sizeof(source));
    REQUIRE(lowered.source_instruction_offsets.size() == source.size());
    REQUIRE(lowered.instructions.size() == 4);
    const auto truncated = Shader::Ampere::LowerToMaxwell(std::span{source}.first(2));
    REQUIRE_FALSE(truncated);
    REQUIRE(truncated.instructions.empty());
}

TEST_CASE("Native register-to-predicate writes only masked predicate bits", "[native-shader]") {
    const std::array<Native, 2> source{{{{0x05007804, 7, 0, 0x000fe200}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(source);
    REQUIRE(lowered);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[0]) == Shader::Maxwell::Opcode::R2P_imm);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.R2P_imm(lowered.instructions[0]);
    for (u32 value = 0; value < 128; ++value) {
        std::array<bool, 7> predicates{true, false, true, false, true, false, true};
        std::unordered_map<const IR::Inst*, u32> values;
        const auto get = [&](const IR::Value& v) -> u32 {
            if (!v.IsImmediate()) return values.at(v.Inst());
            return v.Type() == IR::Type::U1 ? static_cast<u32>(v.U1()) : v.U32();
        };
        for (const auto& inst : block.Instructions()) {
            const auto arg = [&](size_t i) { return get(inst.Arg(i)); };
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetRegister:
                REQUIRE(static_cast<u32>(inst.Arg(0).Reg()) == 5);
                values[&inst] = value; break;
            case IR::Opcode::GetPred: values[&inst] = predicates[static_cast<u32>(inst.Arg(0).Pred())]; break;
            case IR::Opcode::SetPred: predicates[static_cast<u32>(inst.Arg(0).Pred())] = arg(1) != 0; break;
            case IR::Opcode::BitFieldUExtract: values[&inst] = (arg(0) >> arg(1)) & ((1U << arg(2)) - 1); break;
            case IR::Opcode::GetZeroFromOp: values[&inst] = arg(0) == 0; break;
            case IR::Opcode::IEqual: values[&inst] = arg(0) == arg(1); break;
            case IR::Opcode::LogicalNot: values[&inst] = !arg(0); break;
            case IR::Opcode::SelectU1: values[&inst] = arg(0) ? arg(1) : arg(2); break;
            default: FAIL("Unexpected R2P IR " << IR::NameOf(inst.GetOpcode()));
            }
        }
        for (u32 bit = 0; bit < 7; ++bit)
            REQUIRE(predicates[bit] == (bit < 3 ? bool(value & (1U << bit)) : bool(bit % 2 == 0)));
    }
    auto invalid = source;
    invalid[0].words[2] = 1;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(invalid));
}

TEST_CASE("Native 2D explicit LOD sampling preserves its paired coordinates", "[native-shader]") {
    const std::array<Native, 2> input{{{{0xc007b60, 0x30800812, 0x59e070b, 0x100f4200}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TEXS(lowered.instructions[0]);
    u32 samples = 0;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() != IR::Opcode::BoundImageSampleExplicitLod) continue;
        ++samples;
        REQUIRE(inst.Flags<IR::TextureInstInfo>().type == Shader::TextureType::Color2D);
        const auto* coords = inst.Arg(1).Inst();
        REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructF32x2);
        for (u32 i = 0; i < 2; ++i)
            REQUIRE(coords->Arg(i).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{12 + i});
        REQUIRE(inst.Arg(2).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{18});
    }
    REQUIRE(samples == 1);
}

namespace {
using Warp = std::array<u32, 32>;
void ExecuteNativeWarp(IR::Block& block, u32 active,
                       std::array<std::array<u32, 256>, 32>& registers,
                       std::array<std::array<bool, 7>, 32>& predicates) {
    std::unordered_map<const IR::Inst*, Warp> values;
    const auto get = [&](const IR::Value& value) -> Warp {
        if (!value.IsImmediate()) return values.at(value.Inst());
        Warp result;
        result.fill(value.Type() == IR::Type::U1 ? static_cast<u32>(value.U1()) : value.U32());
        return result;
    };
    for (const auto& inst : block.Instructions()) {
        std::array<Warp, 4> args{};
        for (size_t i = 0; i < inst.NumArgs(); ++i) {
            if (inst.Arg(i).Type() != IR::Type::Reg && inst.Arg(i).Type() != IR::Type::Pred)
                args[i] = get(inst.Arg(i));
        }
        Warp result{};
        const auto op = inst.GetOpcode();
        if (op == IR::Opcode::SubgroupBallot || op == IR::Opcode::VoteAny || op == IR::Opcode::VoteAll) {
            u32 mask = 0;
            for (u32 lane = 0; lane < 32; ++lane)
                if ((active & (1U << lane)) && args[0][lane]) mask |= 1U << lane;
            result.fill(op == IR::Opcode::SubgroupBallot ? mask : op == IR::Opcode::VoteAny ? mask != 0 : mask == active);
        } else {
            for (u32 lane = 0; lane < 32; ++lane) {
                const auto a = args[0][lane], b = args[1][lane], c = args[2][lane];
                switch (op) {
                case IR::Opcode::GetRegister: result[lane] = static_cast<u32>(inst.Arg(0).Reg()) == 255 ? 0 : registers[lane][static_cast<u32>(inst.Arg(0).Reg())]; break;
                case IR::Opcode::SetRegister:
                    if ((active & (1U << lane)) && static_cast<u32>(inst.Arg(0).Reg()) != 255) registers[lane][static_cast<u32>(inst.Arg(0).Reg())] = b;
                    break;
                case IR::Opcode::GetPred: result[lane] = predicates[lane][static_cast<u32>(inst.Arg(0).Pred())]; break;
                case IR::Opcode::SetPred:
                    if (active & (1U << lane)) predicates[lane][static_cast<u32>(inst.Arg(0).Pred())] = b != 0;
                    break;
                case IR::Opcode::BitCastF32U32: case IR::Opcode::BitCastU32F32: result[lane] = a; break;
                case IR::Opcode::SelectU32: case IR::Opcode::SelectU1: result[lane] = a ? b : c; break;
                case IR::Opcode::BitwiseAnd32: result[lane] = a & b; break;
                case IR::Opcode::BitwiseOr32: result[lane] = a | b; break;
                case IR::Opcode::ShiftRightLogical32: result[lane] = b < 32 ? a >> b : 0; break;
                case IR::Opcode::ShiftLeftLogical32: result[lane] = b < 32 ? a << b : 0; break;
                case IR::Opcode::INeg32: result[lane] = 0U - a; break;
                case IR::Opcode::IAdd32: result[lane] = a + b; break;
                case IR::Opcode::FindUMsb32: result[lane] = a ? 31U - std::countl_zero(a) : ~0U; break;
                case IR::Opcode::BitFieldUExtract: result[lane] = (a >> b) & ((1U << c) - 1); break;
                case IR::Opcode::ULessThan: result[lane] = a < b; break;
                case IR::Opcode::SLessThan: result[lane] = static_cast<s32>(a) < static_cast<s32>(b); break;
                case IR::Opcode::IEqual: result[lane] = a == b; break;
                case IR::Opcode::GetZeroFromOp: result[lane] = a == 0; break;
                case IR::Opcode::LogicalAnd: result[lane] = a && b; break;
                case IR::Opcode::LogicalNot: result[lane] = !a; break;
                case IR::Opcode::GetInBoundsFromOp: result[lane] = get(inst.Arg(0).Inst()->Arg(1))[lane] <= 31; break;
                case IR::Opcode::FSwizzleAdd: {
                    const u32 mode = (c >> ((lane & 3) * 2)) & 3;
                    const float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b);
                    const float ma = mode == 0 || mode == 2 ? -1.0f : mode == 1 ? 1.0f : 0.0f;
                    const float mb = mode == 0 || mode == 1 ? -1.0f : mode == 2 ? 1.0f : 0.0f;
                    result[lane] = std::bit_cast<u32>(x * ma + y * mb);
                    break;
                }
                case IR::Opcode::ShuffleButterfly: result[lane] = args[0][lane ^ b]; break;
                case IR::Opcode::ShuffleIndex: result[lane] = (active & (1U << (b & 31))) ? args[0][b & 31] : 0xdeadbeefU; break;
                default: FAIL("Unexpected warp-uniform IR " << IR::NameOf(op));
                }
            }
        }
        values.emplace(&inst, result);
    }
}
}

TEST_CASE("Native warp uniform transfers use the first selected active lane and preserve predicates", "[native-shader]") {
    const u32 operation = GENERATE(0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U);
    const Native instruction = operation < 2
        ? Native{{operation == 0 ? 0x020513c2U : 0x020593c2U, 0, 0x000e0000, 0}}
        : operation < 4 ? Native{{0x003f7886, 0, operation == 2 ? 0x00800100U : 0x00800000U, 0}}
        : Native{{0x0000781c, 0, operation == 4 ? 0x03f2f008U : operation == 5 ? 0x03f0f070U : operation == 6 ? 0x03f0e170U : 0x00f41070U, 0}};
    const std::array<Native, 2> source{{instruction, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(source);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::PSETP: visitor.PSETP(word); break;
        case Opcode::SHR_imm: visitor.SHR_imm(word); break;
        case Opcode::SHL_imm: visitor.SHL_imm(word); break;
        case Opcode::MOV32I: visitor.MOV32I(word); break;
        case Opcode::P2R_imm: visitor.P2R_imm(word); break;
        case Opcode::R2P_imm: visitor.R2P_imm(word); break;
        case Opcode::VOTE: visitor.VOTE(word); break;
        case Opcode::IADD_reg: visitor.IADD_reg(word); break;
        case Opcode::LOP_reg: visitor.LOP_reg(word); break;
        case Opcode::FLO_reg: visitor.FLO_reg(word); break;
        case Opcode::SHFL: visitor.SHFL(word); break;
        case Opcode::SEL_reg: visitor.SEL_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected warp-uniform lowering");
        }
    }
    using Warp = std::array<u32, 32>;
    for (const u32 active : {1U, 0x80000000U, 0x80000001U, 0x55555555U, 0xffffffffU}) {
        for (const u32 selected : {0U, 1U, 0x80000000U, 0xaaaaaaaaU, 0xffffffffU}) {
            std::array<std::array<u32, 256>, 32> registers{};
            std::array<std::array<bool, 7>, 32> predicates{};
            for (u32 lane = 0; lane < 32; ++lane) {
                registers[lane][2] = 100 + lane * 17;
                registers[lane][221] = 0xcafebeef;
                registers[lane][239] = (selected & (1U << lane)) ? 64 : 0;
                for (u32 p = 0; p < 7; ++p) predicates[lane][p] = ((lane + p) & 1) != 0;
                predicates[lane][1] = ((selected & (1U << lane)) != 0) != (operation == 4);
            }
            const auto original_predicates = predicates;
            ExecuteNativeWarp(block, active, registers, predicates);
            const u32 ballot = active & (operation == 1 ? ~selected : selected);
            const u32 first = ballot ? std::countr_zero(ballot) : 0;
            for (u32 lane = 0; lane < 32; ++lane) {
                if (!(active & (1U << lane))) continue;
                auto expected_predicates = original_predicates[lane];
                if (operation == 7) expected_predicates[2] = original_predicates[lane][1] && original_predicates[lane][0];
                if (operation == 4) expected_predicates[1] = (selected & (1U << lane)) != 0;
                if (operation == 5 || operation == 6) expected_predicates[0] = operation == 5;
                REQUIRE(predicates[lane] == expected_predicates);
                if (operation < 2) REQUIRE(registers[lane][221] == (ballot ? 100 + first * 17 : 0xcafebeef));
                else if (operation < 4) REQUIRE(registers[lane][239] == (operation == 2 ? (ballot ? 64U : 0U) : (ballot == active ? 64U : 0U)));
            }
        }
    }
}

TEST_CASE("Native uniform 64-bit loads populate consecutive registers", "[native-shader]") {
    const std::array<Native, 6> input{{
        {{0x00057ab9, 0x00c00000, 0x00000800, 0}},
        {{0x05067abb, 0x00c00100, 0x08000a00, 0}},
        {{0x00007c02, 6, 0x08000f00, 0}},
        {{0x00017c02, 7, 0x08000f00, 0}},
        {{0xff007322, 0x00007000, 0x000004ff, 0}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const auto word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::MOV_reg: visitor.MOV_reg(word); break;
        case Opcode::LDC: visitor.LDC(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected uniform pair lowering");
        }
    }
    REQUIRE(EvaluateVertex(block, {}, {std::bit_cast<float>(0U), 3, 7, 9}) == std::array<float, 2>{3, 7});
}

TEST_CASE("Native texture dimension queries retain mip input and width-height outputs", "[native-shader]") {
    const std::array<Native, 2> source{{{{0x03027b6f, 0x00800800, 0x000003ff, 0}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(source);
    REQUIRE(lowered);
    REQUIRE(lowered.required_texture_buffer == 2);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[0]) == Shader::Maxwell::Opcode::TXQ);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TXQ(lowered.instructions[0]);
    std::vector<std::pair<u32, u32>> outputs;
    u32 queries = 0;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageQueryDimensions) {
            ++queries;
            REQUIRE(inst.Arg(0).U32() == 32);
            REQUIRE(inst.Arg(1).Inst()->Arg(0).Reg() == IR::Reg{3});
            REQUIRE(inst.Arg(2).U1());
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister) {
            const auto* extract = inst.Arg(1).Inst();
            REQUIRE(extract->GetOpcode() == IR::Opcode::CompositeExtractU32x4);
            outputs.emplace_back(static_cast<u32>(inst.Arg(0).Reg()), extract->Arg(1).U32());
        }
    }
    REQUIRE(queries == 1);
    REQUIRE(outputs == std::vector<std::pair<u32, u32>>{{2, 0}, {3, 1}});
    auto invalid = source;
    invalid[0].words[1] |= 1U << 30;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(invalid));
}

TEST_CASE("Native texture dimension queries write a lone depth or layer count to Rd", "[native-shader]") {
    // TXQ.DIMENSION with only channel 2 (mask 4) and no Rd2, as Ocarina of Time's effect shaders
    // encode it: R9 = c[2][0x40] at the level in R15 (odd Rd), and R2 = c[2][0x48] at the level
    // in R2 (Rd == Ra).
    struct Case {
        std::array<u32, 4> words;
        u32 dest, lod, handle;
    };
    static constexpr std::array<Case, 2> cases{{
        {{0x0f097b6f, 0x00801000, 0x000004ff, 0x00014600}, 9, 15, 0x40},
        {{0x02027b6f, 0x00801200, 0x000004ff, 0x100f4200}, 2, 2, 0x48},
    }};
    const Case& test = cases[GENERATE(0U, 1U)];
    const std::array<Native, 2> source{{{test.words}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(source);
    REQUIRE(lowered);
    REQUIRE(lowered.required_texture_buffer == 2);
    REQUIRE(lowered.instructions.size() == 2);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[0]) == Shader::Maxwell::Opcode::TXQ);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TXQ(lowered.instructions[0]);
    std::vector<std::pair<u32, u32>> outputs;
    u32 queries = 0;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageQueryDimensions) {
            ++queries;
            REQUIRE(inst.Arg(0).U32() == test.handle);
            REQUIRE(inst.Arg(1).Inst()->Arg(0).Reg() == IR::Reg{test.lod});
            REQUIRE(inst.Arg(2).U1()); // no level count requested
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister) {
            const auto* extract = inst.Arg(1).Inst();
            REQUIRE(extract->GetOpcode() == IR::Opcode::CompositeExtractU32x4);
            outputs.emplace_back(static_cast<u32>(inst.Arg(0).Reg()), extract->Arg(1).U32());
        }
    }
    REQUIRE(queries == 1);
    REQUIRE(outputs == std::vector<std::pair<u32, u32>>{{test.dest, 2}});
    // Still rejected: other sparse masks, a second destination for one channel, Rd >= 254.
    for (const u32 mask : {5U, 6U, 8U, 9U, 10U, 12U}) {
        auto other = source;
        other[0].words[2] = (other[0].words[2] & ~0xf00U) | (mask << 8);
        REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(other));
    }
    auto second = source;
    second[0].words[2] = (second[0].words[2] & ~0xffU) | 0x20U;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(second));
    auto high = source;
    high[0].words[0] = (high[0].words[0] & ~0xff0000U) | (254U << 16);
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(high));
}

TEST_CASE("Native scalar global stores retain both uniform address words and source data", "[native-shader]") {
    std::array<Native, 2> input{{
        {{0xff007986, 7, 0x0c10a904, 0}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions.front()) == Shader::Maxwell::Opcode::STG);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.STG(lowered.instructions.front());
    std::vector<u32> registers;
    u32 stores{};
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::GetRegister)
            registers.push_back(static_cast<u32>(inst.Arg(0).Reg()));
        if (inst.GetOpcode() == IR::Opcode::CompositeConstructU32x2) {
            REQUIRE(inst.Arg(0).Inst()->Arg(0).Reg() == IR::Reg::R220);
            REQUIRE(inst.Arg(1).Inst()->Arg(0).Reg() == IR::Reg::R221);
        }
        if (inst.GetOpcode() == IR::Opcode::WriteGlobal32) {
            REQUIRE(inst.Arg(1).Inst()->Arg(0).Reg() == IR::Reg::R7);
            ++stores;
        }
    }
    std::sort(registers.begin(), registers.end());
    REQUIRE(registers == std::vector<u32>{7, 220, 221});
    REQUIRE(stores == 1);
    input[0].words[2] |= 1; // An unaligned uniform address pair is unsupported.
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(input));
    input[0].words[2] = 0x0c10ab04; // A different store width must not become scalar.
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(input));
}

TEST_CASE("Captured native compute compiles with shared memory and atomics", "[.native-compute-captured]") {
    const char* directory = std::getenv("EDEN_NATIVE_SHADER_FIXTURES");
    REQUIRE(directory != nullptr);
    std::ifstream file(std::filesystem::path{directory} / "compute.source.bin", std::ios::binary | std::ios::ate);
    REQUIRE(file.good());
    const auto size = static_cast<size_t>(file.tellg());
    REQUIRE(size % 16 == 0);
    std::vector<Native> input(size / 16);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(input.data()), size);
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    INFO("offset=" << lowered.error_offset << " error=" << lowered.error);
    REQUIRE(lowered);
    std::vector<u64> code(0x50 / 8);
    for (u64 word : lowered.instructions) {
        if ((code.size() * 8) % 32 == 0) code.push_back(0);
        code.push_back(word);
    }
    code.push_back(0);
    Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
    std::array<u32, 64> qmd{};
    std::ifstream descriptor(std::filesystem::path{directory} / "compute.qmd.bin", std::ios::binary);
    REQUIRE(descriptor.good());
    descriptor.read(reinterpret_cast<char*>(qmd.data()), sizeof(qmd));
    const std::array<u32, 3> group{qmd[18] >> 16, qmd[19] & 0xffff, qmd[19] >> 16};
    const u32 shared_size = qmd[17] & 0x3ffff;
    class ComputeFixtureEnvironment final : public EmptyEnvironment {
    public:
        ComputeFixtureEnvironment(const std::vector<u64>& code_, u32 shared_, std::array<u32, 3> group_, Shader::TextureType texture_)
            : code{code_}, shared{shared_}, group{group_}, texture{texture_} {
            stage = Shader::Stage::Compute;
            start_address = 0x50;
            is_proprietary_driver = true;
        }
        u64 ReadInstruction(u32 address) override { return code.at(address / 8); }
        u32 TranslateProgramAddress(u32 address) const override {
            const auto translated = Shader::Ampere::NativeProgramAddress(code, address);
            if (!translated) throw std::logic_error("invalid compute branch target");
            return *translated;
        }
        u32 ReadCbufValue(u32, u32) override { return 0; }
        u32 TextureBoundBuffer() const override { return 0; }
        u32 SharedMemorySize() const override { return shared; }
        std::array<u32, 3> WorkgroupSize() const override { return group; }
        Shader::TextureType ReadTextureType(u32) override { return texture; }
        const std::vector<u64>& code;
        u32 shared;
        std::array<u32, 3> group;
        Shader::TextureType texture;
    } env{code, shared_size, group,
        std::any_of(input.begin(), input.end(), [](const auto& instruction) {
            return instruction.Bits(0, 12) == 0xb66 && instruction.Bits(61, 3) == 2;
        }) ? Shader::TextureType::Color3D : std::any_of(input.begin(), input.end(), [](const auto& instruction) {
            return (instruction.Bits(0, 12) == 0xb66 && instruction.Bits(61, 3) == 1) ||
                   instruction.Bits(0, 12) == 0xb97 || instruction.Bits(0, 12) == 0xb9b;
        }) ? Shader::TextureType::Color2D : Shader::TextureType::Color1D};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::ObjectPool<IR::Inst> inst_pool;
    Shader::ObjectPool<IR::Block> block_pool;
    Shader::Maxwell::Flow::CFG cfg{env, flow_pool, env.StartAddress()};
    Shader::HostTranslateInfo host;
    host.support_int64 = true;
    host.support_float64 = true;
    host.support_float16 = true;
    host.support_conditional_barrier = true;
    const auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    REQUIRE(program.stage == Shader::Stage::Compute);
    REQUIRE_FALSE(program.blocks.empty());
    u32 barriers{}, shared_atomics{}, global_atomics{}, stores{}, image_stores{}, image_loads{};
    for (const auto* block : program.blocks) for (const auto& inst : block->Instructions()) {
        image_loads += inst.GetOpcode() == IR::Opcode::ImageRead || inst.GetOpcode() == IR::Opcode::BoundImageRead;
        barriers += inst.GetOpcode() == IR::Opcode::Barrier;
        shared_atomics += inst.GetOpcode() == IR::Opcode::SharedAtomicIAdd32;
        global_atomics += inst.GetOpcode() == IR::Opcode::GlobalAtomicIAdd32 || inst.GetOpcode() == IR::Opcode::StorageAtomicIAdd32;
        stores += inst.GetOpcode() == IR::Opcode::WriteGlobal32 || inst.GetOpcode() == IR::Opcode::WriteStorage32;
        image_stores += inst.GetOpcode() == IR::Opcode::ImageWrite || inst.GetOpcode() == IR::Opcode::BoundImageWrite;
    }
    const bool surface = std::any_of(input.begin(), input.end(), [](const auto& instruction) { return instruction.Bits(0, 12) == 0xb9b; });
    if (surface) {
        const auto expected_barriers = std::count_if(input.begin(), input.end(), [](const auto& instruction) {
            return instruction.Bits(0, 12) == 0xb1d;
        });
        REQUIRE(barriers >= expected_barriers);
        REQUIRE(image_stores >= 1);
        const auto expected_loads = std::count_if(input.begin(), input.end(), [](const auto& instruction) {
            return instruction.Bits(0, 12) == 0xb97;
        });
        REQUIRE(image_loads >= expected_loads);
    } else if (std::any_of(input.begin(), input.end(), [](const auto& instruction) { return instruction.Bits(0, 12) == 0xf8c; })) {
        REQUIRE(barriers >= 2);
        REQUIRE(shared_atomics >= 4);
        REQUIRE(global_atomics >= 2);
    } else if (shared_size) {
        REQUIRE(barriers >= 5);
        REQUIRE(shared_atomics >= 2);
        REQUIRE(global_atomics >= 1);
        REQUIRE(stores >= 8);
    } else {
        REQUIRE(barriers == 0);
        REQUIRE(stores >= 4);
    }
}

namespace {
double DecodeTestHalf(u16 bits) {
    const u32 exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    const double value = exponent == 31 ? (mantissa ? std::numeric_limits<double>::quiet_NaN()
                                                   : std::numeric_limits<double>::infinity())
                        : exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -24)
                                        : std::ldexp(static_cast<double>(1024 + mantissa), static_cast<int>(exponent) - 25);
    return (bits & 0x8000) ? -value : value;
}
u16 EncodeTestHalf(double value) {
    const u16 sign = std::signbit(value) ? 0x8000 : 0;
    value = std::abs(value);
    if (std::isnan(value)) return 0xfe01; // Deliberately noncanonical: ReLU must canonicalize it.
    if (value >= 65520) return sign | 0x7c00;
    if (value < std::ldexp(1.0, -14))
        return sign | static_cast<u16>(std::nearbyint(std::ldexp(value, 24)));
    int exponent{};
    std::frexp(value, &exponent);
    --exponent;
    u32 mantissa = static_cast<u32>(std::nearbyint(std::ldexp(value, 10 - exponent)));
    if (mantissa == 2048) { ++exponent; mantissa = 1024; }
    return sign | static_cast<u16>(((exponent + 15) << 10) | (mantissa - 1024));
}
struct ScalarNativeMachine {
    std::array<u32, 256> regs{};
    std::array<bool, 7> predicates{};
    std::unordered_map<u64, u32> constants, global, shared;
    u32 workgroup_barriers{}, device_barriers{};
    void Execute(IR::Block& block) {
        std::unordered_map<const IR::Inst*, u64> values;
        bool carry{}, zero{}, sign{}, overflow{};
        const auto get = [&](const IR::Value& value) -> u64 {
            if (!value.IsImmediate()) return values.at(value.Inst());
            if (value.Type() == IR::Type::U1) return value.U1();
            if (value.Type() == IR::Type::F16 || value.Type() == IR::Type::U16) return value.U16();
            if (value.Type() == IR::Type::U64) return value.U64();
            if (value.Type() == IR::Type::F32) return std::bit_cast<u32>(value.F32());
            return value.U32();
        };
        for (const auto& inst : block.Instructions()) {
            const auto arg = [&](size_t i) { return get(inst.Arg(i)); };
            u64 result{};
            switch (inst.GetOpcode()) {
            case IR::Opcode::BitCastF32U32: case IR::Opcode::BitCastU32F32: result = arg(0); break;
            case IR::Opcode::UnpackFloat2x16: result = (arg(0) & 0xffff) | ((arg(0) >> 16) << 32); break;
            case IR::Opcode::PackFloat2x16: result = (arg(0) & 0xffff) | ((arg(0) >> 32) << 16); break;
            case IR::Opcode::CompositeConstructF16x2: result = arg(0) | (arg(1) << 32); break;
            case IR::Opcode::CompositeExtractF16x2: result = (arg(0) >> (arg(1) * 32)) & 0xffff; break;
            case IR::Opcode::FPNeg16: result = arg(0) ^ 0x8000; break;
            case IR::Opcode::FPAbs16: result = arg(0) & 0x7fff; break;
            case IR::Opcode::FPFma16: result = EncodeTestHalf(std::fma(DecodeTestHalf(static_cast<u16>(arg(0))),
                DecodeTestHalf(static_cast<u16>(arg(1))), DecodeTestHalf(static_cast<u16>(arg(2))))); break;
            case IR::Opcode::FPFloor32: result = std::bit_cast<u32>(std::floor(std::bit_cast<float>(static_cast<u32>(arg(0))))); break;
            case IR::Opcode::ConvertF32F16: result = std::bit_cast<u32>(static_cast<float>(DecodeTestHalf(static_cast<u16>(arg(0))))); break;
            case IR::Opcode::FPOrdEqual32: result = std::bit_cast<float>(static_cast<u32>(arg(0))) == std::bit_cast<float>(static_cast<u32>(arg(1))); break;
            case IR::Opcode::FPOrdGreaterThanEqual32: result = std::bit_cast<float>(static_cast<u32>(arg(0))) >= std::bit_cast<float>(static_cast<u32>(arg(1))); break;
            case IR::Opcode::FPMax32: result = std::bit_cast<u32>(std::fmax(std::bit_cast<float>(static_cast<u32>(arg(0))), std::bit_cast<float>(static_cast<u32>(arg(1))))); break;
            case IR::Opcode::FPMin32: result = std::bit_cast<u32>(std::fmin(std::bit_cast<float>(static_cast<u32>(arg(0))), std::bit_cast<float>(static_cast<u32>(arg(1))))); break;
            case IR::Opcode::ConvertU32F32: result = static_cast<u32>(std::bit_cast<float>(static_cast<u32>(arg(0)))); break;
            case IR::Opcode::FPTrunc32: result = std::bit_cast<u32>(std::trunc(std::bit_cast<float>(static_cast<u32>(arg(0))))); break;
            case IR::Opcode::FPNeg32: result = arg(0) ^ 0x80000000U; break;
            case IR::Opcode::FPAbs32: result = arg(0) & 0x7fffffffU; break;
            case IR::Opcode::ConvertF16F32: result = EncodeTestHalf(std::bit_cast<float>(static_cast<u32>(arg(0)))); break;
            case IR::Opcode::FPRoundEven32: result = std::bit_cast<u32>(std::nearbyint(std::bit_cast<float>(static_cast<u32>(arg(0))))); break;
            case IR::Opcode::FPIsNan32: result = std::isnan(std::bit_cast<float>(static_cast<u32>(arg(0)))); break;
            case IR::Opcode::FPClamp32: result = std::bit_cast<u32>(std::clamp(std::bit_cast<float>(static_cast<u32>(arg(0))), std::bit_cast<float>(static_cast<u32>(arg(1))), std::bit_cast<float>(static_cast<u32>(arg(2))))); break;
            case IR::Opcode::ConvertS32F32: {
                const float value = std::bit_cast<float>(static_cast<u32>(arg(0)));
                result = std::isnan(value) ? 0x87654321U : static_cast<u32>(static_cast<s32>(value));
                break;
            }
            case IR::Opcode::GetRegister: result = inst.Arg(0).Reg() == IR::Reg::RZ ? 0 : regs[static_cast<u32>(inst.Arg(0).Reg())]; break;
            case IR::Opcode::SetRegister: if (inst.Arg(0).Reg() != IR::Reg::RZ) regs[static_cast<u32>(inst.Arg(0).Reg())] = static_cast<u32>(arg(1)); break;
            case IR::Opcode::GetPred: result = predicates[static_cast<u32>(inst.Arg(0).Pred())]; break;
            case IR::Opcode::SetPred: predicates[static_cast<u32>(inst.Arg(0).Pred())] = arg(1) != 0; break;
            case IR::Opcode::IAdd32: result = static_cast<u32>(arg(0) + arg(1)); break;
            case IR::Opcode::ISub32: result = static_cast<u32>(arg(0) - arg(1)); break;
            case IR::Opcode::IMul32: result = static_cast<u32>(arg(0) * arg(1)); break;
            case IR::Opcode::IAdd64: result = arg(0) + arg(1); break;
            case IR::Opcode::UMax32: result = std::max(arg(0), arg(1)); break;
            case IR::Opcode::UMin32: result = std::min(arg(0), arg(1)); break;
            case IR::Opcode::FindUMsb32: result = arg(0) ? 31U - std::countl_zero(static_cast<u32>(arg(0))) : ~0U; break;
            case IR::Opcode::BitReverse32: {
                const u32 source = static_cast<u32>(arg(0));
                for (u32 bit = 0; bit < 32; ++bit) result |= u64{(source >> bit) & 1U} << (31 - bit);
                break;
            }
            case IR::Opcode::ShiftLeftLogical64: result = arg(1) < 64 ? arg(0) << arg(1) : 0; break;
            case IR::Opcode::ShiftRightLogical64: result = arg(1) < 64 ? arg(0) >> arg(1) : 0; break;
            case IR::Opcode::INeg32: result = 0U - static_cast<u32>(arg(0)); break;
            case IR::Opcode::ShiftLeftLogical32: result = arg(1) < 32 ? static_cast<u32>(arg(0) << arg(1)) : 0; break;
            case IR::Opcode::ShiftRightArithmetic32: result = static_cast<u32>(static_cast<s32>(arg(0)) >> std::min<u64>(arg(1), 31)); break;
            case IR::Opcode::ShiftRightLogical32: result = arg(1) < 32 ? static_cast<u32>(arg(0)) >> arg(1) : 0; break;
            case IR::Opcode::BitwiseAnd32: result = arg(0) & arg(1); break;
            case IR::Opcode::BitwiseOr32: result = arg(0) | arg(1); break;
            case IR::Opcode::BitwiseXor32: result = arg(0) ^ arg(1); break;
            case IR::Opcode::BitwiseNot32: result = arg(0) ^ 0xffffffffULL; break;
            case IR::Opcode::BitFieldUExtract: result = (arg(0) >> arg(1)) & ((u64{1} << arg(2)) - 1); break;
            case IR::Opcode::BitFieldInsert: { const u64 mask = ((u64{1} << arg(3)) - 1) << arg(2); result = (arg(0) & ~mask) | ((arg(1) << arg(2)) & mask); break; }
            case IR::Opcode::SelectF32: case IR::Opcode::SelectU32: case IR::Opcode::SelectU1: result = arg(0) ? arg(1) : arg(2); break;
            case IR::Opcode::LogicalAnd: result = arg(0) && arg(1); break;
            case IR::Opcode::LogicalOr: result = arg(0) || arg(1); break;
            case IR::Opcode::LogicalNot: result = !arg(0); break;
            case IR::Opcode::SMin32: result = static_cast<u32>(std::min(static_cast<s32>(arg(0)), static_cast<s32>(arg(1)))); break;
            case IR::Opcode::SMax32: result = static_cast<u32>(std::max(static_cast<s32>(arg(0)), static_cast<s32>(arg(1)))); break;
            case IR::Opcode::SLessThan: result = static_cast<s32>(arg(0)) < static_cast<s32>(arg(1)); break;
            case IR::Opcode::ULessThan: result = arg(0) < arg(1); break;
            case IR::Opcode::IEqual: result = arg(0) == arg(1); break;
            case IR::Opcode::GetCFlag: result = carry; break;
            case IR::Opcode::SetCFlag: carry = arg(0) != 0; break;
            case IR::Opcode::GetZFlag: result = zero; break;
            case IR::Opcode::SetZFlag: zero = arg(0) != 0; break;
            case IR::Opcode::GetSFlag: result = sign; break;
            case IR::Opcode::SetSFlag: sign = arg(0) != 0; break;
            case IR::Opcode::GetOFlag: result = overflow; break;
            case IR::Opcode::SetOFlag: overflow = arg(0) != 0; break;
            case IR::Opcode::GetCarryFromOp: { const auto* sum = inst.Arg(0).Inst(); result = get(sum->Arg(0)) + get(sum->Arg(1)) > 0xffffffffULL; break; }
            case IR::Opcode::GetZeroFromOp: result = arg(0) == 0; break;
            case IR::Opcode::GetSignFromOp: result = (arg(0) & 0x80000000U) != 0; break;
            case IR::Opcode::GetOverflowFromOp: { const auto* sum = inst.Arg(0).Inst(); result = ((get(sum->Arg(0)) ^ arg(0)) & (get(sum->Arg(1)) ^ arg(0)) & 0x80000000U) != 0; break; }
            case IR::Opcode::GetCbufU32: result = constants.at((arg(0) << 32) | arg(1)); break;
            case IR::Opcode::CompositeConstructU32x2: result = arg(0) | (arg(1) << 32); break;
            case IR::Opcode::PackUint2x32: case IR::Opcode::UnpackUint2x32: result = arg(0); break;
            case IR::Opcode::CompositeExtractU32x2: result = static_cast<u32>(arg(0) >> (arg(1) * 32)); break;
            case IR::Opcode::WriteSharedU64: shared[arg(0)] = static_cast<u32>(arg(1)); shared[arg(0) + 4] = static_cast<u32>(arg(1) >> 32); break;
            case IR::Opcode::LoadSharedU64: result = shared.at(arg(0)) | (u64{shared.at(arg(0) + 4)} << 32); break;
            case IR::Opcode::WriteGlobal32: global[arg(0)] = static_cast<u32>(arg(1)); break;
            case IR::Opcode::WriteSharedU32: shared[arg(0)] = static_cast<u32>(arg(1)); break;
            case IR::Opcode::LoadSharedU32: result = shared.at(arg(0)); break;
            case IR::Opcode::GlobalAtomicIAdd32: result = global.at(arg(0)); global[arg(0)] += static_cast<u32>(arg(1)); break;
            case IR::Opcode::SharedAtomicIAdd32: result = shared.at(arg(0)); shared[arg(0)] += static_cast<u32>(arg(1)); break;
            case IR::Opcode::Barrier: ++workgroup_barriers; break;
            case IR::Opcode::DeviceMemoryBarrier: ++device_barriers; break;
            default: FAIL("Unexpected scalar compute IR " << IR::NameOf(inst.GetOpcode()));
            }
            values[&inst] = result;
        }
    }
};
void TranslateScalarNative(std::span<const Native> input, IR::Block& block) {
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    INFO("offset=" << lowered.error_offset << " error=" << lowered.error);
    REQUIRE(lowered);
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (u64 word : lowered.instructions) {
        const auto opcode = Shader::Maxwell::Decode(word);
        if (opcode == Shader::Maxwell::Opcode::EXIT) continue;
        switch (opcode) {
#define INST(name, cute, encode) case Shader::Maxwell::Opcode::name: visitor.name(word); break;
#include "shader_recompiler/frontend/maxwell/maxwell.inc"
#undef INST
        }
    }
}
}

TEST_CASE("Native uniform comparisons and selects preserve independent predicate banks", "[native-shader]") {
    const std::array<Native, 7> input{{
        {{0x09057891, 0x40, 0x0f8e303f, 0}},
        {{0x0500728c, 7, 0x0bf21070, 0}},
        {{0x05057287, 7, 0x08800000, 0}},
        {{0x05057290, 7, 0x0fffe13f, 0}},
        {{0x0400728c, 6, 0x0bf01070, 0}},
        {{0x04047287, 6, 0x08000000, 0}},
        VertexTransform.back(),
    }};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 width : {100U, 256U}) {
        ScalarNativeMachine machine;
        machine.regs[225] = 2;
        machine.regs[223] = width;
        machine.regs[220] = 40;
        machine.regs[222] = 32;
        for (u32 p = 0; p < 7; ++p) machine.predicates[p] = (p & 1) != 0;
        const auto original = machine.predicates;
        machine.Execute(block);
        REQUIRE(machine.regs[221] == width - std::min(192U, width));
        REQUIRE(machine.regs[220] == 32);
        REQUIRE(((machine.regs[239] & 0x20) != 0) == (192U < width));
        REQUIRE((machine.regs[239] & 0x40) == 0);
        REQUIRE(machine.predicates == original);
    }
}

TEST_CASE("Native global stores carry a register offset into the high address word", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x02007986, 9, 0x0810a906, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    ScalarNativeMachine machine;
    machine.regs[222] = 0xfffffff0;
    machine.regs[223] = 0x21;
    machine.regs[2] = 0x30;
    machine.regs[9] = 0xdeadbeef;
    machine.Execute(block);
    REQUIRE(machine.global == std::unordered_map<u64, u32>{{0x2200000020ULL, 0xdeadbeef}});
}

TEST_CASE("Native LEA predicate carry reaches the high pointer word", "[native-shader]") {
    const std::array<Native, 4> input{{
        {{0x00007a11, 0x800, 0x078010ff, 0}},
        {{0xff017a10, 0x900, 0x007fe4ff, 0}},
        {{0x00007386, 7, 0x0010a900, 0}},
        VertexTransform.back(),
    }};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 base : {0x100U, 0xfffffff0U}) {
        ScalarNativeMachine machine;
        machine.regs[0] = 8;
        machine.regs[7] = 0x12345678;
        machine.constants[0x20] = base;
        machine.constants[0x24] = 0x21;
        machine.Execute(block);
        REQUIRE(machine.global == std::unordered_map<u64, u32>{{(u64{0x21} << 32) + base + 32, 0x12345678}});
    }
}

TEST_CASE("Native shared atomics and barriers retain real shared-memory effects", "[native-shader]") {
    const std::array<Native, 6> input{{
        {{0x03007388, 0x407, 0x4800, 0}},
        {{0x03097984, 0x400, 0x4800, 0}},
        {{0xff0a738c, 0x7fc09, 0, 0}},
        {{0x00007992, 0, 0x5000, 0}},
        {{0x00007b1d, 0, 0, 0}},
        VertexTransform.back(),
    }};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    ScalarNativeMachine machine;
    machine.regs[3] = 3;
    machine.regs[7] = 5;
    machine.shared[0x7fc] = 100;
    machine.Execute(block);
    REQUIRE(machine.shared.at(16) == 5);
    REQUIRE(machine.shared.at(0x7fc) == 105);
    REQUIRE(machine.regs[10] == 100);
    REQUIRE(machine.workgroup_barriers == 1);
    REQUIRE(machine.device_barriers == 1);
}

TEST_CASE("Native uniform high shifts and multiply operands retain numeric values", "[native-shader]") {
    std::array<Native, 7> input{{
        {{0x00057882, 123, 0, 0}},
        {{0x3f047899, 5, 0x08010605, 0}},
        {{0x05057899, 5, 0x0800063f, 0}},
        {{0x01027c24, 5, 0x0f8e0202, 0}},
        {{0x00007c17, 4, 0x0b800200, 0}},
        {{0x000a7424, 4, 0x078e0205, 0}},
        VertexTransform.back(),
    }};
    for (u32 value : {0U, 1U, 123U, 0x07ffffffU, 0xffffffffU}) {
        input[0].words[1] = value;
        Shader::ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        TranslateScalarNative(input, block);
        ScalarNativeMachine machine;
        machine.regs[0] = 5000;
        machine.regs[1] = 11;
        machine.regs[2] = 7;
        machine.regs[5] = 3;
        machine.Execute(block);
        const u32 shifted = value << 5;
        const u32 minimum = static_cast<u32>(std::min(5000, static_cast<s32>(shifted)));
        REQUIRE(machine.regs[220] == shifted);
        REQUIRE(machine.regs[221] == shifted);
        REQUIRE(machine.regs[2] == 11U * shifted + 7);
        REQUIRE(machine.regs[0] == minimum);
        REQUIRE(machine.regs[10] == minimum * 3U + 4);
    }
}

TEST_CASE("Native uniform 64-bit right shifts keep the low word of the register pair", "[native-shader]") {
    // USHF.R.U64 URd, URa, n, URc (bits 73-74 = 1, 76 right, 80 low word): low word of
    // (URc:URa) >> n, the OoT mesh shaders' form (n = 16, 21, 24, 26).
    std::array<Native, 9> input{{
        {{0x00067882, 0, 0, 0}},           // UMOV UR6, low
        {{0x00077882, 0, 0, 0}},           // UMOV UR7, high
        {{0x060a7899, 24, 0x08001207, 0}}, // USHF.R.U64 UR10, UR6, 24, UR7
        {{0x06097899, 16, 0x08001207, 0}}, // USHF.R.U64 UR9, UR6, 16, UR7
        {{0x3f0b7899, 21, 0x08001207, 0}}, // USHF.R.U64 UR11, URZ, 21, UR7
        {{0x060c7899, 32, 0x08001207, 0}}, // USHF.R.U64 UR12, UR6, 32, UR7
        {{0x060d7899, 0, 0x08001207, 0}},  // USHF.R.U64 UR13, UR6, 0, UR7
        {{0x06287899, 26, 0x08001207, 0}}, // USHF.R.U64 UR40 (R188), UR6, 26, UR7
        VertexTransform.back(),
    }};
    constexpr std::array<std::array<u32, 2>, 4> values{{
        {0, 0}, {0xffffffff, 0xffffffff}, {0x12345678, 0x9abcdef0}, {0x80000000, 1}}};
    for (const auto& [low, high] : values) {
        input[0].words[1] = low;
        input[1].words[1] = high;
        Shader::ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        TranslateScalarNative(input, block);
        ScalarNativeMachine machine;
        machine.Execute(block);
        const u64 pair = (u64{high} << 32) | low;
        REQUIRE(machine.regs[226] == static_cast<u32>(pair >> 24));
        REQUIRE(machine.regs[225] == static_cast<u32>(pair >> 16));
        REQUIRE(machine.regs[227] == high << 11);
        REQUIRE(machine.regs[228] == high);
        REQUIRE(machine.regs[229] == low);
        REQUIRE(machine.regs[188] == static_cast<u32>(pair >> 26));
        REQUIRE(machine.predicates == std::array<bool, 7>{});
    }
    // @UP2: the guard is swapped into P4 (R239 bit 4) around the shift, as for the other USHF forms.
    const std::array<Native, 2> guarded{{{{0x060e2899, 24, 0x08001207, 0}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(guarded);
    REQUIRE(lowered);
    REQUIRE(lowered.instructions.size() == 5);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[0]) == Shader::Maxwell::Opcode::P2R_imm);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[1]) == Shader::Maxwell::Opcode::R2P_imm);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[2]) == Shader::Maxwell::Opcode::SHF_r_imm);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[3]) == Shader::Maxwell::Opcode::R2P_imm);
    const auto predicate = Shader::Maxwell::Instruction{lowered.instructions[2]}.Pred();
    REQUIRE(predicate.index == 4);
    REQUIRE_FALSE(predicate.negated);
    // Shifts above 32, the high result word and the wrapping form stay rejected.
    for (const auto& [word, value] : {std::array<u32, 2>{1, 33}, std::array<u32, 2>{2, 0x08011207},
                                      std::array<u32, 2>{2, 0x08001a07}}) {
        auto rejected = input;
        rejected[2].words[word] = value;
        const auto result = Shader::Ampere::LowerToMaxwell(rejected);
        REQUIRE_FALSE(result);
        REQUIRE(result.error == "native uniform shift mode");
        REQUIRE(result.error_offset == 2 * 16);
    }
}

TEST_CASE("Native bit reversal and register shifts preserve FFT indexing", "[native-shader]") {
    const std::array<Native, 6> input{{
        {{0x00037301, 19, 0, 0}},
        {{0x03017211, 255, 0x078f50ff, 0}},
        {{0xff087219, 4, 0x00011613, 0}},
        {{0x0f0f7219, 12, 0x00000eff, 0}},
        {{0xff047224, 255, 0x078e0a0c, 0}},
        VertexTransform.back(),
    }};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 shift : {0U, 1U, 10U, 31U, 32U, 33U, 63U}) {
        ScalarNativeMachine machine;
        machine.regs[19] = 0x12345678;
        machine.regs[4] = shift;
        machine.regs[12] = shift;
        machine.regs[15] = 0x9abcdef0;
        machine.Execute(block);
        REQUIRE(machine.regs[3] == 0x1e6a2c48);
        REQUIRE(machine.regs[1] == (0x1e6a2c48U >> 22));
        REQUIRE(machine.regs[8] == (shift < 32 ? 0x12345678U >> shift : 0));
        REQUIRE(machine.regs[15] == (0x9abcdef0U << (shift & 31)));
        REQUIRE(machine.regs[4] == 0U - shift);
    }
}

TEST_CASE("Native 64-bit right funnel shifts keep the low word for every shift amount", "[native-shader]") {
    // SHF.R.U64 Rd, Ra, Rb, Rc (type 73-74 = 1, no .W, low word), as in the task/mesh programs:
    // Rd = low word of (Rc:Ra) >> min(Rb, 64), so 0 once Rb >= 64 (Mesa NAK OpShf fold).
    struct Form {
        Native insn;
        u32 dest;
    };
    const std::array<Form, 5> forms{{
        {{{0x23187219, 0x00000043, 0x0000122c, 0x148fe400}}, 24}, // R24 = R35, R67, R44
        {{{0x23237219, 0x00000043, 0x0000122c, 0x000fe400}}, 35}, // Rd == Ra
        {{{0x23437219, 0x00000043, 0x0000122c, 0x000fe400}}, 67}, // Rd == Rb
        {{{0x232c7219, 0x00000043, 0x0000122c, 0x000fe400}}, 44}, // Rd == Rc
        {{{0x23187219, 0x00000043, 0x000012ff, 0x000fe400}}, 24}, // Rc = RZ
    }};
    for (const auto& form : forms) {
        const std::array<Native, 2> input{{form.insn, VertexTransform.back()}};
        Shader::ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        TranslateScalarNative(input, block);
        const u32 high = form.insn.Bits(64, 8) == 255 ? 0U : 0xfedcba98U;
        for (u32 shift : {0U, 1U, 7U, 31U, 32U, 33U, 45U, 63U, 64U, 65U, 127U, 0x80000000U, 0xffffffffU}) {
            ScalarNativeMachine machine;
            machine.regs[35] = 0x89abcdef;
            machine.regs[44] = 0xfedcba98;
            machine.regs[67] = shift;
            machine.Execute(block);
            const u64 value = (u64{high} << 32) | 0x89abcdefU;
            INFO("dest=" << form.dest << " shift=" << shift);
            REQUIRE(machine.regs[form.dest] == (shift >= 64 ? 0U : static_cast<u32>(value >> shift)));
            if (form.dest != 35) REQUIRE(machine.regs[35] == 0x89abcdefU);
            if (form.dest != 44) REQUIRE(machine.regs[44] == 0xfedcba98U);
            if (form.dest != 67) REQUIRE(machine.regs[67] == shift);
        }
    }
    // The native guard reaches all four lowered instructions.
    Native guarded = forms[0].insn;
    guarded.words[0] = (guarded.words[0] & ~0xf000U) | 0xb000U; // @!P3
    const std::array<Native, 2> guarded_input{{guarded, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(guarded_input);
    REQUIRE(lowered);
    REQUIRE(lowered.source_instruction_offsets[1] == 4);
    for (u32 i = 0; i < 4; ++i) {
        const auto pred = Shader::Maxwell::Instruction{lowered.instructions[i]}.Pred();
        REQUIRE(pred.index == 3);
        REQUIRE(pred.negated);
    }
    // .W (bit 75), .HI (bit 80) and S64 (type 0) are other forms and stay rejected.
    for (const u32 w2 : {0x00001a2cU, 0x0001122cU, 0x0000102cU}) {
        Native other = forms[0].insn;
        other.words[2] = w2;
        const std::array<Native, 2> other_input{{other, VertexTransform.back()}};
        const auto rejected = Shader::Ampere::LowerToMaxwell(other_input);
        REQUIRE_FALSE(rejected);
        REQUIRE(rejected.error == "native register shift mode");
    }
}

TEST_CASE("Native paired shared values keep both words under scaled addressing", "[native-shader]") {
    const std::array<Native, 3> input{{
        {{0x13007388, 4, 0x8a00, 0}},
        {{0x13027984, 0, 0x8a00, 0}},
        VertexTransform.back(),
    }};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    ScalarNativeMachine machine;
    machine.regs[19] = 17;
    machine.regs[4] = 0x12345678;
    machine.regs[5] = 0x9abcdef0;
    machine.Execute(block);
    REQUIRE(machine.shared.at(17 * 8) == 0x12345678);
    REQUIRE(machine.shared.at(17 * 8 + 4) == 0x9abcdef0);
    REQUIRE(machine.regs[2] == 0x12345678);
    REQUIRE(machine.regs[3] == 0x9abcdef0);
}

TEST_CASE("Native formatted surface stores retain coordinates channels and descriptor", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x00007b9b, 0x60000a04, 0x00100f00, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 writes{};
    for (const auto& inst : block.Instructions()) if (inst.GetOpcode() == IR::Opcode::BoundImageWrite) {
        REQUIRE(inst.Arg(0).U32() == 40);
        const auto info = inst.Flags<IR::TextureInstInfo>();
        REQUIRE(info.type.Value() == Shader::TextureType::Color2D);
        const auto* coordinates = inst.Arg(1).Inst();
        REQUIRE(coordinates->Arg(0).Inst()->Arg(0).Reg() == IR::Reg::R0);
        REQUIRE(coordinates->Arg(1).Inst()->Arg(0).Reg() == IR::Reg::R1);
        const auto* color = inst.Arg(2).Inst();
        for (u32 component = 0; component < 4; ++component)
            REQUIRE(static_cast<u32>(color->Arg(component).Inst()->Arg(0).Reg()) == 4 + component);
        ++writes;
    }
    REQUIRE(writes == 1);
}

TEST_CASE("Native predicated predicate copies retain their guard and source", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x0000281c, 0, 0x01f0f070, 0}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    const auto predicate = Shader::Maxwell::Instruction{lowered.instructions.front()}.Pred();
    REQUIRE(predicate.index == 2);
    REQUIRE_FALSE(predicate.negated);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (bool source : {false, true}) {
        ScalarNativeMachine machine;
        machine.predicates[3] = source;
        machine.Execute(block);
        REQUIRE(machine.predicates[0] == source);
    }
}

TEST_CASE("Native NTZ nearest conversion rounds ties to even and clears NaNs", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x00117305, 2, 0x00203100, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 rounds{};
    for (const auto& inst : block.Instructions()) if (inst.GetOpcode() == IR::Opcode::FPRoundEven32) {
        REQUIRE(inst.Flags<IR::FpControl>().fmz_mode == IR::FmzMode::None);
        ++rounds;
    }
    REQUIRE(rounds == 1);
    for (const auto [value, expected] : std::array<std::pair<float, s32>, 9>{{
        {0.5f, 0}, {1.5f, 2}, {2.5f, 2}, {-0.5f, 0}, {-1.5f, -2}, {-2.5f, -2},
        {0.9f, 1}, {-0.9f, -1}, {std::numeric_limits<float>::quiet_NaN(), 0},
    }}) {
        ScalarNativeMachine machine;
        machine.regs[2] = std::bit_cast<u32>(value);
        machine.Execute(block);
        REQUIRE(machine.regs[17] == static_cast<u32>(expected));
    }
    auto unsupported = input;
    unsupported[0].words[2] ^= 1U << 13;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
}

TEST_CASE("Native predicate packing merges only selected predicate bits", "[native-shader]") {
    for (u32 mask : {0U, 8U, 16U, 0x55U, 0x7fU}) {
        const std::array<Native, 2> input{{{{0x02137803, mask, 0, 0}}, VertexTransform.back()}};
        Shader::ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        TranslateScalarNative(input, block);
        for (u32 bits = 0; bits < 128; ++bits) {
            ScalarNativeMachine machine;
            machine.regs[2] = 0xa5a5a5a5U;
            for (u32 i = 0; i < 7; ++i) machine.predicates[i] = (bits >> i) & 1;
            const auto saved = machine.predicates;
            machine.Execute(block);
            REQUIRE(machine.regs[19] == ((0xa5a5a5a5U & ~mask) | (bits & mask)));
            REQUIRE(machine.predicates == saved);
        }
    }
}

TEST_CASE("Native signed NTZ floor conversion rounds negative fractions down", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x00237305, 0x24, 0x00207100, 0x000e2400}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (const auto [value, expected] : std::array<std::pair<float, s32>, 8>{{
        {1.9f, 1}, {-1.1f, -2}, {-0.1f, -1}, {0.9f, 0},
        {0.0f, 0}, {-2.0f, -2}, {12345.5f, 12345},
        {std::numeric_limits<float>::quiet_NaN(), 0},
    }}) {
        ScalarNativeMachine machine;
        machine.regs[36] = std::bit_cast<u32>(value);
        machine.Execute(block);
        REQUIRE(machine.regs[35] == static_cast<u32>(expected));
    }
}

TEST_CASE("Native dependency waits retain the full six-bit pending-operation count", "[native-shader]") {
    for (u32 count : {0U, 1U, 15U, 22U, 63U}) {
        const std::array<Native, 2> input{{{{0x0000791a, 0xd000U | (count << 6), 0, 0}}, VertexTransform.back()}};
        const auto lowered = Shader::Ampere::LowerToMaxwell(input);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 1);
    }
    // Only .LE waits (bit 47) are lowered; any scoreboard and count is accepted.
    const std::array<Native, 2> unsupported{{{{0x0000791a, 0x5000, 0, 0}}, VertexTransform.back()}};
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
}

TEST_CASE("Native histogram warp match compares both address words and preserves predicates", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x00ff73a1, 0, 0x00000200, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 active : {1U, 0x80000000U, 0x80000001U, 0x55555555U, 0xffffffffU}) {
        for (u32 mismatch : {0U, 1U, 2U}) {
            std::array<std::array<u32, 256>, 32> registers{};
            std::array<std::array<bool, 7>, 32> predicates{};
            const u32 first = std::countr_zero(active);
            for (u32 lane = 0; lane < 32; ++lane) {
                registers[lane][0] = 0xfffffff0U + (mismatch == 1 && lane != first ? 4 : 0);
                registers[lane][1] = 0x21U + (mismatch == 2 && lane != first ? 1 : 0);
                for (u32 p = 0; p < 7; ++p) predicates[lane][p] = ((lane + p) & 1) != 0;
            }
            const auto original = predicates;
            ExecuteNativeWarp(block, active, registers, predicates);
            for (u32 lane = 0; lane < 32; ++lane) {
                auto expected = original[lane];
                if (active & (1U << lane)) expected[0] = mismatch == 0 || std::popcount(active) == 1;
                REQUIRE(predicates[lane] == expected);
            }
        }
    }
}

TEST_CASE("Native histogram reduction sums only active lanes with unsigned wrap", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x030573c4, 0, 0x0000c000, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 active : {1U, 0x80000000U, 0x80000001U, 0x55555555U, 0xffffffffU}) {
        std::array<std::array<u32, 256>, 32> registers{};
        std::array<std::array<bool, 7>, 32> predicates{};
        u32 sum = 0;
        for (u32 lane = 0; lane < 32; ++lane) {
            registers[lane][3] = 0xfffffff0U + lane * 17;
            registers[lane][221] = 0xcafebeef;
            if (active & (1U << lane)) sum += registers[lane][3];
            for (u32 p = 0; p < 7; ++p) predicates[lane][p] = ((lane + p) & 1) != 0;
        }
        const auto original = predicates;
        ExecuteNativeWarp(block, active, registers, predicates);
        REQUIRE(predicates == original);
        for (u32 lane = 0; lane < 32; ++lane)
            REQUIRE(registers[lane][221] == ((active & (1U << lane)) ? sum : 0xcafebeef));
    }
}

TEST_CASE("Native histogram pointer carry and global reduction retain the address and addend", "[native-shader]") {
    const std::array<Native, 4> input{{
        {{0x00007a10, 0x00001000, 0x07f1e0ff, 0}},
        {{0xff017624, 0x00001100, 0x000e06ff, 0}},
        {{0x0000798e, 3, 0x0010e100, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 low : {0U, 0xfffffff0U, 0xffffffffU}) {
        ScalarNativeMachine machine;
        machine.regs[0] = 32;
        machine.regs[3] = 17;
        machine.constants[0x40] = low;
        machine.constants[0x44] = 0x21;
        const u64 address = 0x2100000000ULL + low + 32;
        machine.global[address] = 0xfffffff8U;
        machine.Execute(block);
        REQUIRE(machine.regs[0] == static_cast<u32>(address));
        REQUIRE(machine.regs[1] == address >> 32);
        REQUIRE(machine.predicates[0] == (low > 0xffffffdfU));
        REQUIRE(machine.global == std::unordered_map<u64, u32>{{address, 9}});
    }
}

TEST_CASE("Native histogram shared increment scales its address and increments once", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x03ff7f8c, 0, 0x0d00403f, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    ScalarNativeMachine machine;
    machine.regs[3] = 63;
    machine.shared[252] = 0xffffffffU;
    machine.Execute(block);
    REQUIRE(machine.shared == std::unordered_map<u64, u32>{{252, 0}});
}

TEST_CASE("Native histogram uniform leading bit and comparison select the highest active lane", "[native-shader]") {
    const std::array<Native, 3> input{{{{0x000472bd, 4, 0x080e0000, 0}},
        {{0x02007c0c, 4, 0x0bf02070, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 mask : {1U, 0x80000000U, 0x55555555U, 0xffffffffU}) {
        for (u32 lane = 0; lane < 32; ++lane) {
            ScalarNativeMachine machine;
            machine.regs[220] = mask;
            machine.regs[2] = lane;
            machine.Execute(block);
            REQUIRE(machine.regs[220] == 31U - std::countl_zero(mask));
            REQUIRE(machine.predicates[0] == (lane == 31U - std::countl_zero(mask)));
        }
    }
}

TEST_CASE("Native formatted surface loads retain red data coordinates descriptor and guard", "[native-shader]") {
    const Native instruction = GENERATE(Native{{0x400a0b97, 0x60001200, 0x001e0100, 0}},
                                        Native{{0x12250b97, 0x60001200, 0x001e0100, 0}});
    const std::array<Native, 2> input{{instruction, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.required_texture_buffer == 0);
    const auto guard = Shader::Maxwell::Instruction{lowered.instructions.front()}.Pred();
    REQUIRE(guard.index == 0);
    REQUIRE_FALSE(guard.negated);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 loads{}, writes{};
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageRead) {
            REQUIRE(inst.Arg(0).U32() == 0x12 * 4);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::Color2D);
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(static_cast<u32>(coords->Arg(0).Inst()->Arg(0).Reg()) == instruction.Bits(24, 8));
            REQUIRE(static_cast<u32>(coords->Arg(1).Inst()->Arg(0).Reg()) == instruction.Bits(24, 8) + 1);
            ++loads;
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister) {
            REQUIRE(static_cast<u32>(inst.Arg(0).Reg()) == instruction.Bits(16, 8));
            const auto* red = inst.Arg(1).Inst();
            REQUIRE(red->GetOpcode() == IR::Opcode::CompositeExtractU32x4);
            REQUIRE(red->Arg(1).U32() == 0);
            REQUIRE(red->Arg(0).Inst()->GetOpcode() == IR::Opcode::BoundImageRead);
            ++writes;
        }
    }
    REQUIRE(loads == 1);
    REQUIRE(writes == 1);
    auto invalid = input;
    invalid[0].words[2] ^= 0x200; // Reject other channel masks until implemented.
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(invalid));
}

TEST_CASE("Native volume texel fetch preserves XYZ LOD descriptor and split RGBA outputs", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x00087b66, 0x50000a02, 0x049e0f0a, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 fetches{};
    std::vector<u32> writes;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageFetch) {
            REQUIRE(inst.Arg(0).U32() == 40);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::Color3D);
            const auto* coords = inst.Arg(1).Inst();
            for (u32 i = 0; i < 3; ++i)
                REQUIRE(static_cast<u32>(coords->Arg(i).Inst()->Arg(0).Reg()) == i);
            REQUIRE(inst.Arg(3).U32() == 0);
            ++fetches;
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister) {
            writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
            const auto* extract = inst.Arg(1).Inst()->Arg(0).Inst();
            REQUIRE(extract->GetOpcode() == IR::Opcode::CompositeExtractF32x4);
            REQUIRE(extract->Arg(1).U32() == writes.size() - 1);
        }
    }
    REQUIRE(fetches == 1);
    REQUIRE(writes == std::vector<u32>{8, 9, 10, 11});
    auto invalid = input;
    invalid[0].words[0] |= 1U << 24;
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(invalid));
}

TEST_CASE("Native volume surface store retains XYZ and all four data channels", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x00007b9b, 0xa0000c04, 0x00100f00, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 stores{};
    for (const auto& inst : block.Instructions()) if (inst.GetOpcode() == IR::Opcode::BoundImageWrite) {
        REQUIRE(inst.Arg(0).U32() == 48);
        REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::Color3D);
        for (u32 i = 0; i < 3; ++i)
            REQUIRE(static_cast<u32>(inst.Arg(1).Inst()->Arg(i).Inst()->Arg(0).Reg()) == i);
        for (u32 i = 0; i < 4; ++i)
            REQUIRE(static_cast<u32>(inst.Arg(2).Inst()->Arg(i).Inst()->Arg(0).Reg()) == 4 + i);
        ++stores;
    }
    REQUIRE(stores == 1);
}

TEST_CASE("Native half FMA ReLU preserves fusion values above one and canonical NaNs", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x0d127231, 0x3000000d, 0x03848d10, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    const std::array<std::array<u16, 3>, 9> cases{{
        {{0x3c00, 0x4400, 0x4200}}, // -1*1+4 = 3, not saturated to one
        {{0x4000, 0x3c00, 0x0000}}, // -2*2+1 < 0
        {{0x3c00, 0x3c00, 0x0000}},
        {{0x0000, 0x0001, 0x0001}}, // smallest positive subnormal survives
        {{0x7c00, 0x0000, 0x0000}},
        {{0x0000, 0x7c00, 0x7c00}},
        {{0x7e01, 0x3c00, 0x7fff}},
        {{0xfe01, 0x3c00, 0x7fff}},
        {{0x8000, 0x8000, 0x0000}},
    }};
    for (const auto& values : cases) {
        ScalarNativeMachine machine;
        machine.regs[13] = (u32{values[0]} << 16) | 0x4400;
        machine.regs[16] = 0xc0000000 | values[1];
        machine.predicates = {true, false, true, false, true, false, true};
        const auto predicates = machine.predicates;
        machine.Execute(block);
        REQUIRE(machine.regs[18] == (u32{values[2]} | (u32{values[2]} << 16)));
        REQUIRE(machine.predicates == predicates);
    }
    // A separate register operand permits a fused product residual that would
    // vanish if the multiplication rounded before adding its third operand.
    auto fused = input;
    fused[0].words[1] = 0x3000000e;
    IR::Block fused_block{pool};
    TranslateScalarNative(fused, fused_block);
    ScalarNativeMachine machine;
    machine.regs[13] = 0x3c013c01;
    machine.regs[14] = 0x3bfe3bfe;
    machine.regs[16] = 0x3c003c00;
    machine.Execute(fused_block);
    REQUIRE(machine.regs[18] == 0x00100010); // 2^-20 in each half
}

TEST_CASE("Native half FMA ReLU accepts packed immediates without losing input aliases", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x02027831, 0x3c003c00, 0x03968805, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (const auto [source, addend, expected] : std::array<std::array<u16, 3>, 4>{{
        {{0x4000, 0x3c00, 0x3c00}}, {{0x3c00, 0x4400, 0x0000}},
        {{0x4400, 0x3c00, 0x4200}}, {{0x7e01, 0x3c00, 0x7fff}},
    }}) {
        ScalarNativeMachine machine;
        machine.regs[2] = source | 0xc4000000U;
        machine.regs[5] = (u32{addend} << 16) | 0x4800;
        machine.Execute(block);
        REQUIRE(machine.regs[2] == (u32{expected} | (u32{expected} << 16)));
    }
}

TEST_CASE("Native quad butterfly and swizzled add retain horizontal and vertical differences", "[native-shader]") {
    const bool vertical = GENERATE(false, true);
    const std::array<Native, 3> input{{
        {{0x05107f89, vertical ? 0x0c5c0300U : 0x0c3c0300U, 0x000e0000, 0}},
        {{0x100c7822, vertical ? 0x5aU : 0x66U, 5, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    std::array<std::array<u32, 256>, 32> registers{};
    std::array<std::array<bool, 7>, 32> predicates{};
    for (u32 lane = 0; lane < 32; ++lane)
        registers[lane][5] = std::bit_cast<u32>(static_cast<float>(lane / 4 * 100 + (lane & 1) * 3 + ((lane >> 1) & 1) * 11));
    ExecuteNativeWarp(block, 0xffffffffU, registers, predicates);
    for (u32 lane = 0; lane < 32; ++lane) {
        REQUIRE(registers[lane][16] == registers[lane ^ (vertical ? 2 : 1)][5]);
        REQUIRE(std::bit_cast<float>(registers[lane][12]) == (vertical ? 11.0f : 3.0f));
    }
}

TEST_CASE("Native array gradient fetch retains array index XY derivatives descriptor and RGBA", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x04047b6c, 0xa080080c, 0x001e0f06, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 samples{};
    std::vector<u32> writes;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageGradient) {
            REQUIRE(inst.Arg(0).U32() == 32);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::ColorArray2D);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().num_derivatives.Value() == 2);
            const auto* coords = inst.Arg(1).Inst();
            REQUIRE(coords->Arg(0).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{4});
            REQUIRE(coords->Arg(1).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{5});
            const auto* layer = coords->Arg(2).Inst()->Arg(0).Inst();
            REQUIRE(layer->GetOpcode() == IR::Opcode::BitFieldUExtract);
            REQUIRE(layer->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{6});
            REQUIRE(layer->Arg(2).U32() == 16);
            for (u32 component = 0; component < 4; ++component)
                REQUIRE(inst.Arg(2).Inst()->Arg(component).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{12 + component});
            ++samples;
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister) writes.push_back(static_cast<u32>(inst.Arg(0).Reg()));
    }
    REQUIRE(samples == 1);
    REQUIRE(writes == std::vector<u32>{4, 5, 6, 7});
}

TEST_CASE("Native capture distinguishes forward branches from call targets beyond main EXIT", "[native-shader]") {
    using Capture = Shader::Ampere::ProgramCaptureState;
    // CALL.REL.NOINC right after MOV Rlink, return PC is a linked call (as case 0x944 verifies).
    Capture calls;
    REQUIRE(calls.Observe(Native{{0x00087802, 32, 0xf00, 0}}, 0) == Capture::Result::Continue);
    REQUIRE(calls.Observe(Native{{0x7944, 48, 0x03c00000, 0}}, 1) == Capture::Result::Continue);
    REQUIRE(calls.Observe(VertexTransform.back(), 2) == Capture::Result::Continue);
    REQUIRE(calls.Observe(Native{{0x08007950, 0, 0, 0}}, 6) == Capture::Result::Complete);
    // A CALL that pushes its return address (bit 86 clear) needs no link setup.
    Capture pushes;
    REQUIRE(pushes.Observe(Native{{0x7944, 48, 0x03800000, 0}}, 1) == Capture::Result::Continue);
    REQUIRE(pushes.Observe(VertexTransform.back(), 2) == Capture::Result::Continue);
    REQUIRE(pushes.Observe(Native{{0x08007950, 0, 0, 0}}, 6) == Capture::Result::Complete);
    // Without the link setup a NOINC call is a forward jump: an EXIT past its target ends the program.
    Capture jumps;
    REQUIRE(jumps.Observe(Native{{0x7944, 48, 0x03c00000, 0}}, 1) == Capture::Result::Continue);
    REQUIRE(jumps.Observe(VertexTransform.back(), 2) == Capture::Result::Continue);
    REQUIRE(jumps.Observe(Native{{0x08007950, 0, 0, 0}}, 6) == Capture::Result::Continue);
    REQUIRE(jumps.Observe(VertexTransform.back(), 7) == Capture::Result::Complete);
    Capture branches;
    REQUIRE(branches.Observe(Native{{0x7947, 48, 0x03800000, 0}}, 0) == Capture::Result::Continue);
    REQUIRE(branches.Observe(VertexTransform.back(), 1) == Capture::Result::Continue);
    REQUIRE(branches.Observe(VertexTransform.back(), 4) == Capture::Result::Complete);
    Capture multiple;
    REQUIRE(multiple.Observe(Native{{0x00087802, 32, 0xf00, 0}}, 0) == Capture::Result::Continue);
    REQUIRE(multiple.Observe(Native{{0x7944, 96, 0x03c00000, 0}}, 1) == Capture::Result::Continue);
    REQUIRE(multiple.Observe(Native{{0x00087802, 64, 0xf00, 0}}, 2) == Capture::Result::Continue);
    REQUIRE(multiple.Observe(Native{{0x7944, 128, 0x03c00000, 0}}, 3) == Capture::Result::Continue);
    REQUIRE(multiple.Observe(VertexTransform.back(), 4) == Capture::Result::Continue);
    REQUIRE(multiple.Observe(Native{{0x08007950, 0, 0, 0}}, 9) == Capture::Result::Continue);
    REQUIRE(multiple.Observe(Native{{0x08007950, 0, 0, 0}}, 13) == Capture::Result::Complete);
    // BRA bit 32 is .U, not an offset bit (see the BRA.U test): 8 bytes is the smallest misalignment.
    Capture invalid;
    REQUIRE(invalid.Observe(Native{{0x7947, 8, 0x03800000, 0}}, 0) == Capture::Result::InvalidTarget);
}

TEST_CASE("Native capture ends at the return of a linked NOINC subroutine", "[native-shader]") {
    using Capture = Shader::Ampere::ProgramCaptureState;
    // Game shaders can be followed by data that decodes as a CALL with an invalid target (this is
    // ASCII text), so the capture must stop at the subroutine's RET.REL.NODEC instead of reading on.
    const Native data{{0x74736944, 0x0074726f, 0x73614273, 0x4d730065}};
    const auto capture = [](std::span<const Native> input) {
        Capture state;
        for (u32 i = 0; i < input.size(); ++i) {
            const auto status = state.Observe(input[i], i);
            REQUIRE(status != Capture::Result::InvalidTarget);
            if (status == Capture::Result::Complete) return i + 1;
        }
        return 0U;
    };
    std::array<Native, 7> input{{
        {{0x00107802, 0x00000020, 0x00000f00, 0}}, // MOV R16, return PC
        {{0x00007944, 0x00000010, 0x03c00000, 0}}, // CALL.REL.NOINC -> 3
        VertexTransform.back(),
        {{0x00037421, 0x40400000, 0x00010000, 0}}, // callee: add 3.0
        {{0x10007950, 0xffffffb0, 0x03c3ffff, 0}}, // RET.REL.NODEC R16
        data,
        data,
    }};
    REQUIRE(capture(input) == 5);
    const auto lowered = Shader::Ampere::LowerToMaxwell(std::span{input}.first(5));
    REQUIRE(lowered);
    REQUIRE(lowered.source_size == 5 * 16);
    // A subroutine that ends the shader completes the capture at its EXIT.
    input[4] = VertexTransform.back();
    REQUIRE(capture(input) == 5);
    REQUIRE(Shader::Ampere::LowerToMaxwell(std::span{input}.first(5)));
}

TEST_CASE("Native uniform branch flag does not move the branch target", "[native-shader]") {
    using Capture = Shader::Ampere::ProgramCaptureState;
    // NAK OpBra: bit 32 is .U and the word offset starts at bit 34, so @!P0 BRA.U (words[1] bit 0
    // set) has the target of @!P0 BRA and lowers to the same code.
    const auto forward = [](u32 offset) {
        return std::array<Native, 3>{{
            {{0x00008947, offset, 0x03800000, 0}}, // @!P0 BRA -> 2
            {{0x00027802, 0x00000001, 0x00000f00, 0}},
            VertexTransform.back(),
        }};
    };
    const auto backward = [](u32 offset) {
        return std::array<Native, 3>{{
            {{0x00027802, 0x00000001, 0x00000f00, 0}},
            {{0x00008947, offset, 0x0383ffff, 0}}, // @!P0 BRA -> 0
            VertexTransform.back(),
        }};
    };
    for (const auto& [plain, uniform] : {std::pair{forward(0x10), forward(0x11)},
                                         std::pair{backward(0xffffffe0), backward(0xffffffe1)}}) {
        Capture capture;
        REQUIRE(capture.Observe(uniform[0], 0) == Capture::Result::Continue);
        REQUIRE(capture.Observe(uniform[1], 1) == Capture::Result::Continue);
        REQUIRE(capture.Observe(uniform[2], 2) == Capture::Result::Complete);
        const auto expected = Shader::Ampere::LowerToMaxwell(plain);
        const auto lowered = Shader::Ampere::LowerToMaxwell(uniform);
        REQUIRE(expected);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions == expected.instructions);
        REQUIRE(lowered.source_instruction_offsets == expected.source_instruction_offsets);
        REQUIRE(lowered.source_size == expected.source_size);
    }
    // Bit 33 has no known meaning: it still misaligns the target and stays rejected.
    Capture capture;
    REQUIRE(capture.Observe(forward(0x12)[0], 0) == Capture::Result::InvalidTarget);
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(forward(0x12)));
}

TEST_CASE("Native callee link check ignores instructions without a register destination", "[native-shader]") {
    // With the link in R0, bits 16-23 are 0 in FSETP/ISETP (predicate destinations), NOP, BRA and
    // RET, and none of them writes a register. A real write of R0 is still rejected.
    const std::array<Native, 8> input{{
        {{0x00007802, 0x00000020, 0x00000f00, 0}}, // MOV R0, return PC
        {{0x00007944, 0x00000010, 0x03c00000, 0}}, // CALL.REL.NOINC -> 3
        VertexTransform.back(),
        {{0x3e00720b, 0x00000039, 0x03f3d000, 0}}, // FSETP.NEU.FTZ.AND P1, PT, R62, R57, PT
        {{0x0300720c, 0x00000004, 0x03f06070, 0}}, // ISETP.GE.AND P0, PT, R3, R4, PT
        {{0x00007918, 0x00000000, 0x00000000, 0}}, // NOP
        {{0x00008947, 0x00000000, 0x03800000, 0}}, // @!P0 BRA -> 7
        {{0x00007950, 0xffffff80, 0x03c3ffff, 0}}, // RET.REL.NODEC R0
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.source_size == input.size() * 16);
    auto broken = input;
    broken[4] = Native{{0xff007221, 0x0000004a, 0x00014000, 0}}; // FADD R0, -RZ, R74
    const auto rejected = Shader::Ampere::LowerToMaxwell(broken);
    REQUIRE_FALSE(rejected);
    REQUIRE(rejected.error_offset == 4 * 16);
    REQUIRE(rejected.error == "native callee overwrites link register");
}

TEST_CASE("Native callee may reuse its link register after copying it for the return", "[native-shader]") {
    // MOV R0, R1 copies link R1, MOV R1, 0 reuses the link and RET.REL.NODEC R0 returns through the
    // copy. The run from the reuse to the RET is plain MOVs that keep R0, so no return target changes.
    const std::array<Native, 7> input{{
        {{0x00017802, 0x00000020, 0x00000f00, 0}}, // MOV R1, return PC
        {{0x00007944, 0x00000010, 0x03c00000, 0}}, // CALL.REL.NOINC -> 3
        VertexTransform.back(),
        {{0x00007202, 0x00000001, 0x00000f00, 0}}, // MOV R0, R1
        {{0x00017802, 0x00000000, 0x00000f00, 0}}, // MOV R1, 0
        {{0x004a7202, 0x0000004b, 0x00000f00, 0}}, // MOV R74, R75
        {{0x00007950, 0xffffff90, 0x03c3ffff, 0}}, // RET.REL.NODEC R0
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.source_size == input.size() * 16);
    // Rejected at the reuse: R0 copied again from the reused link, a RET through the reused link,
    // a non-MOV before the RET, and R0 not being a copy of the link.
    for (const auto& [index, replacement] : std::array<std::pair<u32, Native>, 4>{{
             {5, Native{{0x00007202, 0x00000001, 0x00000f00, 0}}},
             {6, Native{{0x01007950, 0xffffff90, 0x03c3ffff, 0}}},
             {5, Native{{0x3e00720b, 0x00000039, 0x03f3d000, 0}}},
             {3, Native{{0x00007202, 0x00000002, 0x00000f00, 0}}},
         }}) {
        auto broken = input;
        broken[index] = replacement;
        const auto rejected = Shader::Ampere::LowerToMaxwell(broken);
        REQUIRE_FALSE(rejected);
        REQUIRE(rejected.error_offset == 4 * 16);
        REQUIRE(rejected.error == "native callee overwrites link register");
    }
}

TEST_CASE("Native implicit volume sample preserves XYZ and all four split channels", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x0a0a7b60, 0x5080400e, 0x041e0f0c, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 samples{};
    std::vector<std::pair<u32, u32>> copies;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageSampleImplicitLod) {
            REQUIRE(inst.Arg(0).U32() == 256);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::Color3D);
            for (u32 i = 0; i < 3; ++i)
                REQUIRE(inst.Arg(1).Inst()->Arg(i).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{250 + i});
            ++samples;
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister && inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::GetRegister)
            copies.emplace_back(static_cast<u32>(inst.Arg(0).Reg()), static_cast<u32>(inst.Arg(1).Inst()->Arg(0).Reg()));
    }
    REQUIRE(samples == 1);
    REQUIRE(copies == std::vector<std::pair<u32, u32>>{{250, 10}, {251, 11}, {252, 14}, {10, 250}, {11, 251}, {12, 252}, {13, 253}});
}

TEST_CASE("Native array shadow sampling retains layer XY depth reference and scalar destination", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x082e7b60, 0xb080140a, 0x001e41ff, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 samples{};
    std::vector<std::pair<u32, u32>> copies;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BoundImageSampleDrefImplicitLod) {
            REQUIRE(inst.Arg(0).U32() == 80);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::ColorArray2D);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().is_depth.Value() == 1);
            REQUIRE(inst.Arg(1).Inst()->Arg(0).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{251});
            REQUIRE(inst.Arg(1).Inst()->Arg(1).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{252});
            REQUIRE(inst.Arg(2).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{11});
            ++samples;
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister && inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::GetRegister)
            copies.emplace_back(static_cast<u32>(inst.Arg(0).Reg()), static_cast<u32>(inst.Arg(1).Inst()->Arg(0).Reg()));
    }
    REQUIRE(samples == 1);
    REQUIRE(copies == std::vector<std::pair<u32, u32>>{{250, 8}, {251, 9}, {252, 10}});
    REQUIRE(block.Instructions().back().Arg(0).Reg() == IR::Reg{46});
}

TEST_CASE("Native immediate uniform comparison and select preserve predicates at unsigned boundaries", "[native-shader]") {
    const std::array<Native, 3> input{{{{0x0400788c, 6, 0x0bf01070, 0}}, {{0x04057887, 6, 0x08000000, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 value : {0U, 5U, 6U, 7U, 0x80000000U, 0xffffffffU}) {
        ScalarNativeMachine machine;
        machine.regs[220] = value;
        for (u32 i = 0; i < 7; ++i) machine.predicates[i] = (i & 1) != 0;
        const auto saved = machine.predicates;
        machine.Execute(block);
        REQUIRE(((machine.regs[239] >> 6) & 1) == (value < 6));
        REQUIRE(machine.regs[221] == std::min(value, 6U));
        REQUIRE(machine.predicates == saved);
    }
}

TEST_CASE("Native uniform signed right shift sign extends the high word", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x3f057899, 31, 0x08011404, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 value : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU}) {
        ScalarNativeMachine machine;
        machine.regs[220] = value;
        machine.Execute(block);
        REQUIRE(machine.regs[221] == ((value & 0x80000000U) ? 0xffffffffU : 0U));
    }
}

TEST_CASE("Native indexed constant load adds a signed byte displacement", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x30077b82, 0x00fe0500, 0x00000800, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    ScalarNativeMachine machine;
    machine.regs[48] = 0x800;
    machine.constants[(u64{3} << 32) | 0x14] = 0x12345678;
    machine.Execute(block);
    REQUIRE(machine.regs[7] == 0x12345678);
}

TEST_CASE("Native local spill retains byte address data and execution predicate", "[native-shader]") {
    const std::array<Native, 3> input{{{{0xff001387, 85, 0x00100800, 0}},
        {{0xff2a7983, 0, 0x00300800, 0}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[0]) == Shader::Maxwell::Opcode::STL);
    REQUIRE(Shader::Maxwell::Instruction{lowered.instructions[0]}.Pred().index == 1);
    REQUIRE((lowered.instructions[0] & 255) == 85);
    REQUIRE((lowered.instructions[1] & 255) == 42);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    class LocalEnvironment : public EmptyEnvironment {
    public:
        u32 LocalMemorySize() const override { return 4; }
    } env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.STL(lowered.instructions[0]);
    visitor.LDL(lowered.instructions[1]);
    u32 loads{}, stores{};
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::LoadLocal) {
            REQUIRE(inst.Arg(0).U32() == 0);
            ++loads;
        }
        if (inst.GetOpcode() == IR::Opcode::WriteLocal) {
            REQUIRE(inst.Arg(0).U32() == 0);
            REQUIRE(inst.Arg(1).Inst()->Arg(0).Reg() == IR::Reg{85});
            ++stores;
        }
    }
    REQUIRE(loads == 1);
    REQUIRE(stores == 1);
}

TEST_CASE("Native scalar half conversion applies absolute value before rounding", "[native-shader]") {
    const bool negate = GENERATE(false, true);
    const std::array<Native, 2> input{{{{0x00047304, negate ? 0x80000000U : 0x40000000U, 0x00200800, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (const auto [value, expected] : std::array<std::pair<float, u32>, 7>{{
        {-1.0f, 0x3c00}, {-0.0f, 0}, {-2.5f, 0x4100},
        {1.00048828125f, 0x3c00}, {1.00146484375f, 0x3c02},
        {-0x1p-24f, 1}, {-std::numeric_limits<float>::infinity(), 0x7c00},
    }}) {
        ScalarNativeMachine machine;
        machine.regs[0] = std::bit_cast<u32>(value);
        machine.Execute(block);
        REQUIRE(machine.regs[4] == (expected | (negate && !std::signbit(value) ? 0x8000U : 0U)));
    }
}

TEST_CASE("Native predicated half ReLU guards its complete expansion including predicate scratch", "[native-shader]") {
    for (u32 pred : {1U, 6U, 9U, 14U}) {
        const std::array<Native, 2> input{{{{0x12140231U | (pred << 12), 0x20000012, 0x03848914, 0}}, VertexTransform.back()}};
        const auto lowered = Shader::Ampere::LowerToMaxwell(input);
        REQUIRE(lowered);
        REQUIRE(Shader::Maxwell::Decode(lowered.instructions.front()) == Shader::Maxwell::Opcode::BRA);
        REQUIRE(((lowered.instructions.front() >> 16) & 15) == (pred ^ 8));
        Shader::ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        EmptyEnvironment env;
        Shader::Maxwell::TranslatorVisitor visitor{env, block};
        for (size_t i = 1; i + 1 < lowered.instructions.size(); ++i) {
            const u64 word = lowered.instructions[i];
            REQUIRE(((word >> 16) & 15) == 7);
            switch (Shader::Maxwell::Decode(word)) {
#define INST(name, cute, encode) case Shader::Maxwell::Opcode::name: visitor.name(word); break;
#include "shader_recompiler/frontend/maxwell/maxwell.inc"
#undef INST
            }
        }
        for (bool value : {false, true}) {
            ScalarNativeMachine machine;
            machine.regs[18] = 0x3c003c00;
            machine.regs[20] = 0x44004400;
            machine.predicates.fill(true);
            machine.predicates[pred & 7] = value;
            const auto saved = machine.predicates;
            if (value != ((pred & 8) != 0)) machine.Execute(block);
            REQUIRE(machine.regs[20] == (value != ((pred & 8) != 0) ? 0x42004200U : 0x44004400U));
            REQUIRE(machine.predicates == saved);
        }
    }
}

TEST_CASE("Native unsigned half conversion truncates and saturates NTZ edge cases", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x000e7305, 13, 0x0010f000, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (const auto [value, expected] : std::array<std::pair<u32, u32>, 8>{{
        {0x3e00, 1}, {0x4100, 2}, {0xbe00, 0}, {0x0001, 0},
        {0x7bff, 65504}, {0x7c00, 0xffffffffU}, {0xfc00, 0}, {0x7e00, 0},
    }}) {
        ScalarNativeMachine machine;
        machine.regs[13] = 0xdead0000 | value;
        machine.Execute(block);
        REQUIRE(machine.regs[14] == expected);
    }
}

TEST_CASE("Native predicate LUT OR combines three inputs while preserving other predicates", "[native-shader]") {
    const std::array<Native, 3> input{{{{0x0000781c, 0, 0x01f25f16, 0}}, {{0x0000781c, 0, 0x00703570, 0}}, VertexTransform.back()}};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 bits = 0; bits < 128; ++bits) {
        ScalarNativeMachine machine;
        for (u32 i = 0; i < 7; ++i) machine.predicates[i] = (bits >> i) & 1;
        auto expected = machine.predicates;
        expected[1] = expected[1] || expected[2] || expected[3];
        expected[0] = expected[0] || expected[1];
        machine.Execute(block);
        REQUIRE(machine.predicates == expected);
    }
}

TEST_CASE("Native half FMA absolute source clears each half sign before broadcast", "[native-shader]") {
    for (bool broadcast : {false, true}) {
        const std::array<Native, 2> input{{{{0x1a087231, 0x20000008,
            broadcast ? 0x00040a0dU : 0x0004020dU, 0}}, VertexTransform.back()}};
        Shader::ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        TranslateScalarNative(input, block);
        ScalarNativeMachine machine;
        machine.regs[26] = 0xc400c000;
        machine.regs[8] = 0x40003c00;
        machine.regs[13] = 0x44004200;
        machine.Execute(block);
        REQUIRE(machine.regs[8] == (broadcast ? 0x45004500U : 0x47004500U));
    }
}

TEST_CASE("Native constant interpolation retains attribute and flat input behavior", "[native-shader]") {
    const std::array<Native, 2> input{{{{0x00047326, 255, 0x000e403c, 0}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(((lowered.instructions[0] >> 54) & 3) == 2);
    class FlatEnvironment : public EmptyEnvironment {
    public:
        FlatEnvironment() { sph.ps.imap_generic_vector[7].raw = 0x55; }
    } env;
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.IPA(lowered.instructions[0]);
    u32 reads{};
    for (const auto& inst : block.Instructions()) {
        REQUIRE(inst.GetOpcode() != IR::Opcode::FPMul32);
        if (inst.GetOpcode() == IR::Opcode::GetAttribute) {
            REQUIRE(inst.Arg(0).Attribute() == IR::Attribute{60});
            ++reads;
        }
    }
    REQUIRE(reads == 1);
}

// Native programs whose registers reach the lowering's reserved range (R216-R254 with a uniform
// operation, else R246-R254; Hyrule Warriors / Ocarina model vertex shaders use up to R252) keep
// those native registers in local memory. These checks run the lowered words through the actual
// Maxwell translator and evaluate the IR, including the local memory the environment adds.
namespace {
using HomedAttributes = std::unordered_map<u32, u32>;
HomedAttributes EvaluateHomed(IR::Block& block, const HomedAttributes& input,
                              const std::unordered_map<u64, u32>& constants, u32 local_size) {
    std::unordered_map<const IR::Inst*, u32> values;
    std::array<u32, 256> regs{};
    regs.fill(0x7fc0dead); // garbage: nothing may read a register it did not write
    std::vector<u32> local(local_size / 4, 0x7fc0beef);
    HomedAttributes output;
    const auto get = [&](const IR::Value& value) -> u32 {
        if (!value.IsImmediate()) return values.at(value.Inst());
        if (value.Type() == IR::Type::U1) return value.U1();
        return value.Type() == IR::Type::F32 ? std::bit_cast<u32>(value.F32()) : value.U32();
    };
    for (const auto& inst : block.Instructions()) {
        const auto arg = [&](size_t i) { return get(inst.Arg(i)); };
        const auto f = [&](size_t i) { return std::bit_cast<float>(arg(i)); };
        u32 result{};
        switch (inst.GetOpcode()) {
        case IR::Opcode::GetRegister: result = inst.Arg(0).Reg() == IR::Reg::RZ ? 0 : regs[static_cast<u32>(inst.Arg(0).Reg())]; break;
        case IR::Opcode::SetRegister: if (inst.Arg(0).Reg() != IR::Reg::RZ) regs[static_cast<u32>(inst.Arg(0).Reg())] = arg(1); break;
        case IR::Opcode::GetAttribute: result = input.at(static_cast<u32>(inst.Arg(0).Attribute()) * 4); break;
        case IR::Opcode::SetAttribute: output[static_cast<u32>(inst.Arg(0).Attribute()) * 4] = arg(1); break;
        case IR::Opcode::GetCbufU32: case IR::Opcode::GetCbufF32: result = constants.at((u64{arg(0)} << 32) | arg(1)); break;
        case IR::Opcode::BitCastF32U32: case IR::Opcode::BitCastU32F32: result = arg(0); break;
        case IR::Opcode::FPAdd32: result = std::bit_cast<u32>(f(0) + f(1)); break;
        case IR::Opcode::FPMul32: result = std::bit_cast<u32>(f(0) * f(1)); break;
        case IR::Opcode::FPFma32: result = std::bit_cast<u32>(std::fma(f(0), f(1), f(2))); break;
        case IR::Opcode::IAdd32: result = arg(0) + arg(1); break;
        case IR::Opcode::ULessThan: result = arg(0) < arg(1); break;
        case IR::Opcode::SelectU32: result = arg(0) ? arg(1) : arg(2); break;
        case IR::Opcode::LoadLocal: REQUIRE(arg(0) < local.size()); result = local[arg(0)]; break;
        case IR::Opcode::WriteLocal: REQUIRE(arg(0) < local.size()); local[arg(0)] = arg(1); break;
        default: FAIL("Unexpected homed-register IR " << IR::NameOf(inst.GetOpcode()));
        }
        values[&inst] = result;
    }
    return output;
}

HomedAttributes RunHomed(std::span<const Native> input, const HomedAttributes& attributes,
                         const std::unordered_map<u64, u32>& constants, u32 native_local_size) {
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    INFO("offset=" << lowered.error_offset << " error=" << lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.local_memory_bias == 0x100);
    // What the patched GraphicsEnvironment reports: the SPH's local memory plus the bias.
    class HomedEnvironment final : public EmptyEnvironment {
    public:
        explicit HomedEnvironment(u32 size_) : size{size_} {}
        u32 LocalMemorySize() const override { return size; }
        u32 size;
    } env{native_local_size + lowered.local_memory_bias};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    for (const u64 word : lowered.instructions) {
        using Shader::Maxwell::Opcode;
        switch (Shader::Maxwell::Decode(word)) {
        case Opcode::ALD: visitor.ALD(word); break;
        case Opcode::AST: visitor.AST(word); break;
        case Opcode::LDL: visitor.LDL(word); break;
        case Opcode::STL: visitor.STL(word); break;
        case Opcode::MOV_cbuf: visitor.MOV_cbuf(word); break;
        case Opcode::MOV_reg: visitor.MOV_reg(word); break;
        case Opcode::FADD_reg: visitor.FADD_reg(word); break;
        case Opcode::FMUL_reg: visitor.FMUL_reg(word); break;
        case Opcode::FFMA_reg: visitor.FFMA_reg(word); break;
        case Opcode::EXIT: break;
        default: FAIL("Unexpected lowered homed-register instruction " << std::hex << word);
        }
    }
    return EvaluateHomed(block, attributes, constants, env.LocalMemorySize());
}
} // namespace

TEST_CASE("Native registers in the reserved range keep live values across scratch use", "[native-shader]") {
    const float x = 1.5f, y = -2.25f, z = 3.0f, w = 0.5f, m = 4.0f, n = -1.0f;
    const HomedAttributes attributes{{0x80, std::bit_cast<u32>(x)}, {0x84, std::bit_cast<u32>(y)},
                                     {0x88, std::bit_cast<u32>(z)}, {0x8c, std::bit_cast<u32>(w)}};
    const std::unordered_map<u64, u32> constants{{(u64{3} << 32) | 0x8, std::bit_cast<u32>(m)},
                                                 {(u64{3} << 32) | 0xc, std::bit_cast<u32>(n)}};
    // No uniform operation: R246-R254 are reserved. FFMA's constant-buffer B is materialized in
    // scratch R251 while native R251 holds w; without homing the lowering used to be refused.
    std::vector<Native> program{
        {{0xfff67321, 0x000080ff, 0x00000400, 0x000fe200}}, // ALD.64 R246, a[0x80]
        {{0xfffc7321, 0x000088ff, 0x00000000, 0x000fe200}}, // ALD R252, a[0x88]
        {{0xfffe7321, 0x00008cff, 0x00000000, 0x000fe200}}, // ALD R254, a[0x8c]
        {{0xf6fa7a23, 0x00c00200, 0x000000f7, 0x000fe200}}, // FFMA R250, R246, c[3][0x8], R247
        {{0x00fb7202, 0x000000fe, 0x00000f00, 0x000fe200}}, // MOV R251, R254
        {{0xfbfd7a23, 0x00c00300, 0x000000fc, 0x000fe200}}, // FFMA R253, R251, c[3][0xc], R252
        {{0xff007322, 0x0000a0fa, 0x000000ff, 0x000fe200}}, // AST a[0xa0], R250
        {{0xff007322, 0x0000a4fb, 0x000000ff, 0x000fe200}}, // AST a[0xa4], R251
        {{0xff007322, 0x0000a8fd, 0x000000ff, 0x000fe200}}, // AST a[0xa8], R253
        {{0xff007322, 0x0000b0f6, 0x000004ff, 0x000fe200}}, // AST.64 a[0xb0], R246
        VertexTransform.back(),
    };
    const auto output = RunHomed(program, attributes, constants, 0);
    REQUIRE(output.at(0xa0) == std::bit_cast<u32>(std::fma(x, m, y)));
    REQUIRE(output.at(0xa4) == std::bit_cast<u32>(w));
    REQUIRE(output.at(0xa8) == std::bit_cast<u32>(std::fma(w, n, z)));
    REQUIRE(output.at(0xb0) == std::bit_cast<u32>(x));
    REQUIRE(output.at(0xb4) == std::bit_cast<u32>(y));
}

TEST_CASE("Native registers in the reserved range coexist with uniform register homes", "[native-shader]") {
    const float x = 1.5f, y = -2.25f, z = 3.0f, w = 0.5f, u = 7.0f, k = 0.125f;
    const HomedAttributes attributes{{0x80, std::bit_cast<u32>(x)}, {0x84, std::bit_cast<u32>(y)},
                                     {0x88, std::bit_cast<u32>(z)}, {0x8c, std::bit_cast<u32>(w)},
                                     {0x90, std::bit_cast<u32>(u)}};
    const std::unordered_map<u64, u32> constants{{0x10, std::bit_cast<u32>(k)}};
    // ULDC makes R216-R254 reserved and writes UR4's home R220 while native R220 holds u; native
    // R248-R251 are one 128-bit attribute vector; the native local store moves above the homes.
    std::vector<Native> program{
        {{0xfff87321, 0x000080ff, 0x00000c00, 0x000fe200}}, // ALD.128 R248, a[0x80]
        {{0xffdc7321, 0x000090ff, 0x00000000, 0x000fe200}}, // ALD R220, a[0x90]
        {{0x00047ab9, 0x00000400, 0x00000800, 0x000fe200}}, // ULDC UR4, c[0x0][0x10]
        {{0xdcdd7e21, 0x00000004, 0x08000000, 0x000fe200}}, // FADD R221, R220, UR4
        {{0xfaf87220, 0x000000fb, 0x00400000, 0x000fe200}}, // FMUL R248, R250, R251 (z * w)
        {{0xff007387, 0x000010f9, 0x00100800, 0x000fe200}}, // STL [0x10], R249
        {{0xfffc7983, 0x00001000, 0x00100800, 0x000fe200}}, // LDL R252, [0x10]
        {{0xff007322, 0x0000a0f8, 0x00000cff, 0x000fe200}}, // AST.128 a[0xa0], R248
        {{0xff007322, 0x0000b0dd, 0x000000ff, 0x000fe200}}, // AST a[0xb0], R221
        {{0xff007322, 0x0000b4fc, 0x000000ff, 0x000fe200}}, // AST a[0xb4], R252
        {{0xff007322, 0x0000b8dc, 0x000000ff, 0x000fe200}}, // AST a[0xb8], R220
        VertexTransform.back(),
    };
    const auto output = RunHomed(program, attributes, constants, 0x20); // the native STL/LDL at 0x10
    REQUIRE(output.at(0xa0) == std::bit_cast<u32>(z * w));
    REQUIRE(output.at(0xa4) == std::bit_cast<u32>(y));
    REQUIRE(output.at(0xa8) == std::bit_cast<u32>(z));
    REQUIRE(output.at(0xac) == std::bit_cast<u32>(w));
    REQUIRE(output.at(0xb0) == std::bit_cast<u32>(u + k));
    REQUIRE(output.at(0xb4) == std::bit_cast<u32>(y));
    REQUIRE(output.at(0xb8) == std::bit_cast<u32>(u));
}

TEST_CASE("Native reserved-range registers are homed only where every operand is known", "[native-shader]") {
    constexpr std::string_view scratch = "native program uses lowering scratch registers";
    const Native ald250{{0xfffa7321, 0x000080ff, 0x00000000, 0x000fe200}}; // ALD R250, a[0x80]
    const auto rejects = [&](std::vector<Native> program, u32 index) {
        const auto lowered = Shader::Ampere::LowerToMaxwell(program);
        REQUIRE_FALSE(lowered);
        REQUIRE(lowered.error == scratch);
        REQUIRE(lowered.error_offset == index * 16);
    };
    rejects({ald250, {{0xfa007949, 0, 0, 0x000fea00}}, VertexTransform.back()}, 1);          // BRX R250
    rejects({ald250, {{0x04fa7231, 5, 6, 0x000fe200}}, VertexTransform.back()}, 1);          // HFMA2 R250
    rejects({ald250, {{0x00047ab9, 0x00000400, 0x00000800, 0}},                              // ALD.64 R215:R216
             {{0xffd77321, 0x000088ff, 0x00000400, 0x000fe200}}, VertexTransform.back()}, 2);
    // A program below the reserved range lowers exactly as before.
    const auto plain = Shader::Ampere::LowerToMaxwell(VertexTransform);
    REQUIRE(plain);
    REQUIRE(plain.local_memory_bias == 0);
    // ISETP keeps its .EX predicate in bits 68-71 (0x70 = PT), which is no register.
    const auto isetp = Shader::Ampere::DescribeNativeGprs({{0x7b00720c, 0x000000ff, 0x03fa5070, 0x000fe200}});
    REQUIRE(isetp);
    REQUIRE(isetp->count == 2);
    REQUIRE(isetp->fields[0].bit == 24);
    REQUIRE(isetp->fields[1].bit == 32);
}

TEST_CASE("Native reserved-range operands past R254 or over a callee link stay rejected", "[native-shader]") {
    // Cases R3 and R4 of the scratch-regs review (review/adversarial_test.cpp).
    // CS2R.64 R254, SRZ names R254 and RZ: renamed, the lowering would also write the register after
    // its temporary (here a live R3 of the ALD.128).
    const std::array<Native, 4> cs2r{{
        {{0xff007321, 0x000080ff, 0x00000c00, 0x000fe200}}, // ALD.128 R0, a[0x80]
        {{0x00fe7805, 0x00000000, 0x0001ff00, 0x000fe200}}, // CS2R.64 R254, SRZ
        {{0xff007322, 0x0000a000, 0x00000cff, 0x000fe200}}, // AST.128 a[0xa0], R0
        VertexTransform.back(),
    }};
    const auto cs2r_lowered = Shader::Ampere::LowerToMaxwell(cs2r);
    REQUIRE_FALSE(cs2r_lowered);
    REQUIRE(cs2r_lowered.error == "native program uses lowering scratch registers");
    REQUIRE(cs2r_lowered.error_offset == 16);
    // A callee's LDL.64 over its link register: rejected for R146:R147 (link R147) by the LDL case,
    // and so for R246:R247 (link R247), whose destination is renamed for the home slots.
    for (const u32 link : {147U, 247U}) {
        const std::array<Native, 6> callee{{
            {{0x00007802U | link << 16, 0x20, 0xf00, 0x000fe200}},                 // MOV Rlink, 0x20
            {{0x00007944, 0x10, 0x03c00000, 0x000fea00}},                          // CALL.REL.NOINC #3
            {{0x0000794d, 0, 0x03800000, 0x000fea00}},                             // EXIT
            {{0xff007983U | (link - 1) << 16, 0, 0x00100a00, 0x000fe200}},         // LDL.64 Rlink-1, [RZ]
            {{0x000a7202, link, 0xf00, 0x000fe200}},                               // MOV R10, Rlink
            {{0x0a007950, 0xffffffa0, 0x03c3ffff, 0x000fea00}},                    // RET.REL.NODEC R10
        }};
        const auto lowered = Shader::Ampere::LowerToMaxwell(callee);
        REQUIRE_FALSE(lowered);
        REQUIRE(lowered.error == "native callee overwrites link register");
        REQUIRE(lowered.error_offset == 3 * 16);
    }
}

TEST_CASE("Native bindless shadow sample passes the register handle, XY and reference to TEX.B", "[native-shader]") {
    // Ocarina of Time Model.bfsha "model": TEX.B R86, R52, R50 2D.DC (x, y | handle, reference).
    std::array<Native, 2> input{{{{0x34567361, 0x38000032, 0x041e41ff, 0x00014200}}, VertexTransform.back()}};
    const bool predicated = GENERATE(false, true);
    if (predicated) input[0].words[0] = (input[0].words[0] & ~0xf000U) | 0x3000;
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.required_texture_buffer == ~0U);
    u32 tex_b{};
    for (const u64 word : lowered.instructions) {
        if (Shader::Maxwell::Decode(word) == Shader::Maxwell::Opcode::EXIT) continue;
        tex_b += Shader::Maxwell::Decode(word) == Shader::Maxwell::Opcode::TEX_b ? 1U : 0U;
        REQUIRE(Shader::Maxwell::Instruction{word}.Pred().index == (predicated ? 3U : 7U));
    }
    REQUIRE(tex_b == 1);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 samples{};
    std::vector<std::pair<u32, u32>> copies;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::BindlessImageSampleDrefImplicitLod) {
            REQUIRE(inst.Arg(0).Inst()->GetOpcode() == IR::Opcode::GetRegister);
            REQUIRE(inst.Arg(0).Inst()->Arg(0).Reg() == IR::Reg{246});
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::Color2D);
            REQUIRE(inst.Flags<IR::TextureInstInfo>().is_depth.Value() == 1);
            for (u32 i = 0; i < 2; ++i)
                REQUIRE(inst.Arg(1).Inst()->Arg(i).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{250 + i});
            REQUIRE(inst.Arg(2).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{247});
            ++samples;
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister && inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::GetRegister)
            copies.emplace_back(static_cast<u32>(inst.Arg(0).Reg()), static_cast<u32>(inst.Arg(1).Inst()->Arg(0).Reg()));
    }
    REQUIRE(samples == 1);
    REQUIRE(copies == std::vector<std::pair<u32, u32>>{{250, 52}, {251, 53}, {246, 50}, {247, 51}, {86, 250}});
    REQUIRE(block.Instructions().back().Arg(0).Reg() == IR::Reg{86});
}

TEST_CASE("Native bindless volume sample reads Z and the handle from Rb and splits RGB", "[native-shader]") {
    // TEX.B R30, R0, R30, R8 3D RGB (x, y | z, handle); bit 87 selects LZ.
    std::array<Native, 2> input{{{{0x1e1e7361, 0x58000008, 0x001e0700, 0x00414200}}, VertexTransform.back()}};
    const bool lz = GENERATE(false, true);
    if (lz) input[0].words[2] |= 1U << 23;
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    u32 samples{};
    std::vector<std::pair<u32, u32>> copies;
    for (const auto& inst : block.Instructions()) {
        if (inst.GetOpcode() == (lz ? IR::Opcode::BindlessImageSampleExplicitLod
                                    : IR::Opcode::BindlessImageSampleImplicitLod)) {
            REQUIRE(inst.Arg(0).Inst()->Arg(0).Reg() == IR::Reg{246});
            REQUIRE(inst.Flags<IR::TextureInstInfo>().type.Value() == Shader::TextureType::Color3D);
            for (u32 i = 0; i < 3; ++i)
                REQUIRE(inst.Arg(1).Inst()->Arg(i).Inst()->Arg(0).Inst()->Arg(0).Reg() == IR::Reg{250 + i});
            if (lz) REQUIRE(inst.Arg(2).F32() == 0.0f);
            ++samples;
        }
        if (inst.GetOpcode() == IR::Opcode::SetRegister && inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::GetRegister)
            copies.emplace_back(static_cast<u32>(inst.Arg(0).Reg()), static_cast<u32>(inst.Arg(1).Inst()->Arg(0).Reg()));
    }
    REQUIRE(samples == 1);
    REQUIRE(copies == std::vector<std::pair<u32, u32>>{{250, 30}, {251, 31}, {252, 8}, {246, 9}, {30, 250}, {31, 251}, {0, 252}});
}

TEST_CASE("Native bindless sample rejects forms with an unverified operand layout", "[native-shader]") {
    const std::array<Native, 2> shadow{{{{0x34567361, 0x38000032, 0x041e41ff, 0}}, VertexTransform.back()}};
    const std::array<Native, 2> volume{{{{0x1e1e7361, 0x58000008, 0x001e0700, 0}}, VertexTransform.back()}};
    // .B, packed operands, AOFFI, NDV, FP16, sparse result, cbuf reference bits, LOD bias/clamp,
    // bit 91, another dimension.
    for (const u32 bit : {59U, 60U, 76U, 77U, 79U, 81U, 40U, 58U, 88U, 89U, 91U, 62U}) {
        for (const auto& base : {shadow, volume}) {
            auto unsupported = base;
            unsupported[0].words[bit / 32] ^= 1U << (bit % 32);
            const auto rejected = Shader::Ampere::LowerToMaxwell(unsupported);
            REQUIRE_FALSE(rejected);
            REQUIRE(rejected.instructions.empty());
        }
    }
    auto unsupported = shadow;
    unsupported[0].words[2] &= ~(1U << 14); // 2D without a depth reference (three values)
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
    unsupported = volume;
    unsupported[0].words[2] |= 1U << 14; // 3D with a depth reference
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(unsupported));
    // With a uniform register in use (UMOV UR7, 1), R216 holds UR0: Ra = R215 would read it.
    std::array<Native, 3> uniform{{{{0x00077882, 1, 0, 0}}, shadow[0], VertexTransform.back()}};
    REQUIRE(Shader::Ampere::LowerToMaxwell(uniform));
    uniform[1].words[0] = (uniform[1].words[0] & 0x00ffffffU) | (215U << 24);
    REQUIRE_FALSE(Shader::Ampere::LowerToMaxwell(uniform));
}

TEST_CASE("Native wide immediate multiply-add carries the product and addend into the high word", "[native-shader]") {
    // IMAD.WIDE.U32 R12:R13, R13, imm, R20:R21 (Rd+1 = Ra, as in OoT's model shaders), then
    // IMAD.WIDE.U32 R20:R21, R5, imm, R20:R21 (Rd = Rc) and IMAD.WIDE.U32 R6:R7, R9, imm, RZ.
    // Powers of two (0x10 and 0x4 are the corpus' immediates) take the shift path, the others XMAD.
    const u32 multiplier = GENERATE(0x10U, 0x4U, 2U, 0x80000000U, 1U, 0U, 3U, 0x10001U, 0xcccccccdU, 0xffffffffU);
    std::array<Native, 4> input{{
        {{0x0d0c7825, multiplier, 0x078e0014, 0x000fe400}},
        {{0x05147825, multiplier, 0x078e0014, 0x000fe400}},
        {{0x09067825, multiplier, 0x078e00ff, 0x000fe400}},
        VertexTransform.back(),
    }};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 a : {0U, 1U, 0xffffU, 0x10000U, 0x0fffffffU, 0x10000000U, 0x12345678U, 0x80000000U, 0xffffffffU}) {
        for (u64 c : {0ULL, 0xfffffff0ULL, 0xffffffffULL, 0x00000001fffffff0ULL, 0x8000000000000000ULL,
                      0xffffffffffffffffULL}) {
            ScalarNativeMachine machine;
            machine.regs[13] = a;
            machine.regs[5] = a ^ 0x5a5a5a5aU;
            const u32 not_a = ~a;
            machine.regs[9] = not_a;
            machine.regs[20] = static_cast<u32>(c);
            machine.regs[21] = static_cast<u32>(c >> 32);
            machine.regs[7] = 0xdeadbeef;
            machine.Execute(block);
            const u64 first = u64{a} * multiplier + c;
            const u64 second = u64{a ^ 0x5a5a5a5aU} * multiplier + c;
            const u64 third = u64{not_a} * multiplier;
            REQUIRE(machine.regs[12] == static_cast<u32>(first));
            REQUIRE(machine.regs[13] == static_cast<u32>(first >> 32));
            REQUIRE(machine.regs[20] == static_cast<u32>(second));
            REQUIRE(machine.regs[21] == static_cast<u32>(second >> 32));
            REQUIRE(machine.regs[6] == static_cast<u32>(third));
            REQUIRE(machine.regs[7] == static_cast<u32>(third >> 32));
            REQUIRE(machine.regs[9] == not_a);
        }
    }
}

TEST_CASE("Native wide immediate multiply-add guards its expansion and rejects other modes", "[native-shader]") {
    for (const u32 multiplier : {0x10U, 3U}) {
        // @!P0 IMAD.WIDE.U32 R12:R13, R13, imm, R20:R21: every emitted instruction carries the guard.
        const std::array<Native, 2> input{{{{0x0d0c8825, multiplier, 0x078e0014, 0}}, VertexTransform.back()}};
        const auto lowered = Shader::Ampere::LowerToMaxwell(input);
        REQUIRE(lowered);
        const u32 begin = lowered.source_instruction_offsets[0], end = lowered.source_instruction_offsets[1];
        REQUIRE(end - begin == (multiplier == 0x10 ? 5U : 13U));
        for (u32 i = begin; i < end; ++i) {
            const auto pred = Shader::Maxwell::Instruction{lowered.instructions[i]}.Pred();
            REQUIRE(pred.index == 0);
            REQUIRE(pred.negated);
        }
    }
    // Signed (bit 73), -Ra (72), .X (74), -Rc (75), a carry-out in P0 (81-83), a uniform B (91),
    // an odd Rd pair and an odd Rc pair are not this form.
    for (const auto& words : std::array<std::array<u32, 3>, 8>{{
             {0x0d0c7825, 0x10, 0x078e0214}, {0x0d0c7825, 0x10, 0x078e0114}, {0x0d0c7825, 0x10, 0x078e0414},
             {0x0d0c7825, 0x10, 0x078e0814}, {0x0d0c7825, 0x10, 0x07800014}, {0x0d0c7825, 0x10, 0x0f8e0014},
             {0x0d0d7825, 0x10, 0x078e0014}, {0x0d0c7825, 0x10, 0x078e0015},
         }}) {
        const std::array<Native, 2> input{{{{words[0], words[1], words[2], 0}}, VertexTransform.back()}};
        const auto rejected = Shader::Ampere::LowerToMaxwell(input);
        REQUIRE_FALSE(rejected);
        REQUIRE(rejected.error_offset == 0);
        REQUIRE(rejected.error == "native wide immediate multiply mode");
        REQUIRE(rejected.instructions.empty());
    }
}

TEST_CASE("Native wide immediate multiply-add in a callee may not write the link as its high word", "[native-shader]") {
    const std::array<Native, 5> input{{
        {{0x00017802, 0x00000020, 0x00000f00, 0}}, // MOV R1, return PC
        {{0x00007944, 0x00000010, 0x03c00000, 0}}, // CALL.REL.NOINC -> 3
        VertexTransform.back(),
        {{0x05027825, 0x00000010, 0x078e0004, 0}}, // IMAD.WIDE.U32 R2:R3, R5, 0x10, R4:R5
        {{0x01007950, 0xffffffb0, 0x03c3ffff, 0}}, // RET.REL.NODEC R1
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.source_size == input.size() * 16);
    auto broken = input;
    broken[3].words[0] = 0x05007825; // IMAD.WIDE.U32 R0:R1, ...: R1 is the link
    const auto rejected = Shader::Ampere::LowerToMaxwell(broken);
    REQUIRE_FALSE(rejected);
    REQUIRE(rejected.error_offset == 3 * 16);
    REQUIRE(rejected.error == "native callee overwrites link register");
}

TEST_CASE("Native uniform immediate high-word shifts clamp or wrap the uniform shift amount", "[native-shader]") {
    // USHF.L.U32.HI UR16, URZ, UR10, 0xffffffff and USHF.L.W.U32.HI UR5, URZ, UR14, 0x1 (opcode
    // 0x499: form 2, the immediate is the high source), as in OoT's model_meshlet programs.
    const std::array<Native, 3> input{{
        {{0x3f107499, 0xffffffff, 0x0801060a, 0}},
        {{0x3f057499, 0x00000001, 0x08010e0e, 0}},
        VertexTransform.back(),
    }};
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    TranslateScalarNative(input, block);
    for (u32 shift : {0U, 1U, 5U, 31U, 32U, 33U, 63U, 0x80000000U, 0xffffffffU}) {
        ScalarNativeMachine machine;
        machine.regs[226] = shift; // UR10
        machine.regs[230] = shift; // UR14
        machine.Execute(block);
        REQUIRE(machine.regs[232] == (shift < 32 ? 0xffffffffU << shift : 0U));
        REQUIRE(machine.regs[221] == (1U << (shift & 31)));
    }
}

TEST_CASE("Native guarded uniform immediate shift reads its uniform predicate and keeps guest predicates", "[native-shader]") {
    // @UP4 USHF.L.U32.HI UR9, URZ, UR13, 0xffffffff: UP4 (R239 bit 2) is swapped into P2.
    const std::array<Native, 2> input{{{{0x3f094499, 0xffffffff, 0x0801060d, 0}}, VertexTransform.back()}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.source_instruction_offsets[1] == 5);
    EmptyEnvironment env;
    for (bool up4 : {false, true}) {
        ScalarNativeMachine machine;
        machine.regs[229] = 7;          // UR13
        machine.regs[225] = 0x12345678; // UR9
        machine.regs[239] = up4 ? 1U << 2 : 0U;
        for (u32 p = 0; p < 7; ++p) machine.predicates[p] = (p & 1) == 0; // guest P2 is true
        const auto saved = machine.predicates;
        for (size_t i = 0; i < 5; ++i) {
            const u64 word = lowered.instructions[i];
            const u32 guard = (word >> 16) & 15;
            REQUIRE((guard == 7 || guard == 2));
            if (guard == 2 && !machine.predicates[2]) continue;
            Shader::ObjectPool<IR::Inst> pool;
            IR::Block block{pool};
            Shader::Maxwell::TranslatorVisitor visitor{env, block};
            switch (Shader::Maxwell::Decode(word)) {
#define INST(name, cute, encode) case Shader::Maxwell::Opcode::name: visitor.name(word); break;
#include "shader_recompiler/frontend/maxwell/maxwell.inc"
#undef INST
            }
            machine.Execute(block);
        }
        REQUIRE(machine.regs[225] == (up4 ? 0xffffff80U : 0x12345678U));
        REQUIRE(machine.predicates == saved);
    }
}

TEST_CASE("Native uniform immediate shift rejects other forms and reserves the uniform bank", "[native-shader]") {
    const Native ushf{{0x3f057499, 0x00000001, 0x0801060d, 0}};
    for (const auto& [word, value] : std::initializer_list<std::pair<u32, u32>>{
             {0, 0x04057499}, {2, 0x0801160d}, {2, 0x0800060d}, {2, 0x0801040d}, {2, 0x0001060d}}) {
        auto bad = ushf;
        bad.words[word] = value; // low source UR4, right shift, low result, I32 type, no bit 91
        const std::array<Native, 2> input{{bad, VertexTransform.back()}};
        const auto result = Shader::Ampere::LowerToMaxwell(input);
        REQUIRE_FALSE(result);
        REQUIRE(result.error_offset == 0);
        REQUIRE(result.instructions.empty());
    }
    // The USHF alone makes R216-R245 the uniform bank, so a native write of R216 cannot use the
    // register itself: it is kept in local memory (without a uniform operation R216 is no
    // reserved register and the program needs no local memory).
    const std::array<Native, 3> input{{ushf, {{0x00d87802, 1, 0x00000f00, 0}}, VertexTransform.back()}};
    const auto result = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(result);
    REQUIRE(result.local_memory_bias == 0x100);
    // The override format carries no local memory bias, so only the unhomed program is encoded.
    REQUIRE(Shader::Ampere::EncodeProgramOverride(input, result).empty());
    const std::array<Native, 2> plain{{{{0x00d87802, 1, 0x00000f00, 0}}, VertexTransform.back()}};
    const auto unreserved = Shader::Ampere::LowerToMaxwell(plain);
    REQUIRE(unreserved);
    REQUIRE(unreserved.local_memory_bias == 0);
    REQUIRE_FALSE(Shader::Ampere::EncodeProgramOverride(plain, unreserved).empty());
}

TEST_CASE("Native thread kill and variable-rate special registers reach the Maxwell S2R and a 1x1 rate", "[native-shader]") {
    // OoT fragment programs: 070d4c5044bae124 #3278 S2R R4, SR_THREAD_KILL (0x13) and
    // 013a349ddd5e7ebd #1631 S2R R0, SR_VARIABLE_RATE (0x84).
    const std::array<Native, 3> input{{
        {{0x00047919, 0x00000000, 0x00001300, 0x000e2800}},
        {{0x00007919, 0x00000000, 0x00008400, 0x000e2400}},
        VertexTransform.back(),
    }};
    const auto lowered = Shader::Ampere::LowerToMaxwell(input);
    REQUIRE(lowered);
    REQUIRE(lowered.source_instruction_offsets[1] == 1);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[0]) == Shader::Maxwell::Opcode::S2R);
    REQUIRE(Shader::Maxwell::Decode(lowered.instructions[1]) == Shader::Maxwell::Opcode::MOV32I);
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    EmptyEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    visitor.S2R(lowered.instructions[0]);
    visitor.MOV32I(lowered.instructions[1]);
    bool helper{}, select{}, rate{};
    for (const auto& inst : block.Instructions()) {
        switch (inst.GetOpcode()) {
        case IR::Opcode::IsHelperInvocation:
            helper = true;
            break;
        case IR::Opcode::SelectU32:
            REQUIRE(inst.Arg(0).Inst()->GetOpcode() == IR::Opcode::IsHelperInvocation);
            REQUIRE(inst.Arg(1).U32() == ~0U);
            REQUIRE(inst.Arg(2).U32() == 0U);
            select = true;
            break;
        case IR::Opcode::SetRegister:
            if (inst.Arg(0).Reg() == IR::Reg::R4) {
                REQUIRE(inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::SelectU32);
            } else {
                REQUIRE(inst.Arg(0).Reg() == IR::Reg::R0);
                REQUIRE(inst.Arg(1).U32() == 0x00010100U); // width 1 (bits 8-15), height 1 (16-23)
                rate = true;
            }
            break;
        default:
            FAIL("Unexpected IR operation " << IR::NameOf(inst.GetOpcode()));
        }
    }
    REQUIRE((helper && select && rate));
    auto guarded = input;
    guarded[1].words[0] = (guarded[1].words[0] & ~0xf000U) | 0xa000U; // @!P2 keeps its guard
    const auto predicated = Shader::Ampere::LowerToMaxwell(guarded);
    REQUIRE(predicated);
    const auto pred = Shader::Maxwell::Instruction{predicated.instructions[1]}.Pred();
    REQUIRE(pred.index == 2);
    REQUIRE(pred.negated);
    for (const u32 uniform_form : {0U, 1U}) { // S2UR of either register stays rejected
        auto rejected = input;
        rejected[uniform_form].words[0] = (rejected[uniform_form].words[0] & ~0xfffU) | 0x9c3U;
        const auto result = Shader::Ampere::LowerToMaxwell(rejected);
        REQUIRE_FALSE(result);
        REQUIRE(result.error_offset == uniform_form * 16);
    }
}
