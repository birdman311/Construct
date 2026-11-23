#pragma GCC diagnostic ignored "-Wshadow"
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Encoding reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.
#pragma once

#include <cstdlib>
#include <algorithm>
#include <array>
#include <bit>
#include <span>
#include <optional>
#include <string_view>
#include <vector>
#include "common/common_types.h"

namespace Shader::Ampere {

struct Instruction {
    std::array<u32, 4> words;

    u32 Bits(u32 offset, u32 count) const {
        u64 bits = words[offset / 32];
        if (offset / 32 + 1 < words.size()) {
            bits |= u64{words[offset / 32 + 1]} << 32;
        }
        return static_cast<u32>((bits >> (offset % 32)) & ((u64{1} << count) - 1));
    }
};
static_assert(sizeof(Instruction) == 16);

struct LoweredProgram {
    std::vector<u64> instructions;
    std::vector<u32> source_instruction_offsets;
    u32 source_size{};
    u32 required_texture_buffer = ~0U;
    // Bytes of local memory the lowering keeps below the native program's own local memory: the
    // native accesses are moved up by this much and the environment adds it to the local memory
    // size. Nonzero only when native registers live in local memory (HomeRegisterArea below).
    u32 local_memory_bias{};
    u32 error_offset{};
    std::string_view error;

    explicit operator bool() const { return error.empty(); }
};

// Native GPR operands of one instruction: 8-bit register fields and an upper bound of how many
// consecutive registers each one names. Mesa NAK sm70_encode.rs encodes every GPR operand as an
// 8-bit field at bit 16 (set_dst), 24, 32 or 64 (set_reg / set_reg_src / set_reg_addr /
// set_alu_reg); the uniform datapath's fields name uniform registers (set_udst / set_ureg).
struct NativeGprOperands {
    struct Field { u32 bit, width; };
    std::array<Field, 4> fields{};
    u32 count = 0;
    bool control_flow = false; // the lowering may leave this instruction's code by a branch
    void Add(u32 bit, u32 width) { fields[count++] = {bit, width}; }
};

// The GPR operands of the forms listed here; nullopt for any other instruction.
inline std::optional<NativeGprOperands> DescribeNativeGprs(const Instruction& insn) {
    const u32 op = insn.Bits(0, 12);
    const u32 form = insn.Bits(9, 3);
    // Global/local/shared/constant sizes (bits 73-75): U8, S8, U16, S16, 32, 64, 128.
    const auto size_registers = [&](u32 size) { return size < 5 ? 1U : size == 5 ? 2U : 4U; };
    NativeGprOperands result;
    switch (op) {
    // No register operand: NOP, DEPBAR, MEMBAR, ERRBAR, BAR.SYNC; BRA, BSSY, BSYNC, BREAK,
    // WARPSYNC, EXIT, CALL.REL.NOINC (targets are immediates), BRXU (uniform register target).
    case 0x918: case 0x91a: case 0x992: case 0x9ab: case 0xb1d:
        return result;
    case 0x947: case 0x945: case 0x941: case 0x942: case 0x948: case 0x94d: case 0x944: case 0x958:
        result.control_flow = true;
        return result;
    case 0x949: case 0x950: // BRX Ra, RET.REL.NODEC Ra: register targets (bits 24-31)
        result.control_flow = true;
        result.Add(24, 2);
        return result;
    // Uniform datapath (NAK encode_ualu / set_udst / set_ureg): no GPR operand.
    case 0xab9: case 0xabb: case 0x882: case 0xc82: case 0x28c: case 0x88c: case 0x290: case 0x890:
    case 0x291: case 0x891: case 0x292: case 0x892: case 0x299: case 0x899: case 0x499: case 0x2a4:
    case 0x8a4: case 0x4a4: case 0x2a5: case 0x8a5: case 0x287: case 0x887: case 0x896: case 0x2bd: case 0x2be:
    case 0x2bf: case 0x89c: case 0x883: case 0x886: case 0x81c:
        return result;
    case 0x3c2: // R2UR URd, Ra
        result.Add(24, 1);
        return result;
    case 0x381: // LDG Rd, [Ra(.64 with bit 72)]
        result.Add(16, size_registers(insn.Bits(73, 3)));
        result.Add(24, insn.Bits(72, 1) ? 2 : 1);
        return result;
    case 0x981: // LDG Rd, [Ra(.64 with bit 90) + URb (bits 32-37)]
        result.Add(16, size_registers(insn.Bits(73, 3)));
        result.Add(24, insn.Bits(90, 1) ? 2 : 1);
        return result;
    case 0x386: // STG [Ra(.64 with bit 72)], Rb
        result.Add(24, insn.Bits(72, 1) ? 2 : 1);
        result.Add(32, size_registers(insn.Bits(73, 3)));
        return result;
    case 0x986: // STG [Ra(.64 with bit 90) + URc (bits 64-69)], Rb
        result.Add(24, insn.Bits(90, 1) ? 2 : 1);
        result.Add(32, size_registers(insn.Bits(73, 3)));
        return result;
    case 0x983: case 0x984: // LDL / LDS Rd, [Ra + URb (bits 32-37)]
        result.Add(16, size_registers(insn.Bits(73, 3)));
        result.Add(24, 1);
        return result;
    case 0x387: case 0x388: case 0x987: case 0x988: // STL / STS [Ra (+ URc, bits 64-69)], Rb
        result.Add(24, 1);
        result.Add(32, size_registers(insn.Bits(73, 3)));
        return result;
    case 0xb82: // LDC Rd, c[bank][Ra + offset]
        result.Add(16, size_registers(insn.Bits(73, 3)));
        result.Add(24, 1);
        return result;
    case 0x321: // ALD Rd (bits 74-75 + 1 components), a[Ra + offset], Rvertex (bits 32-39)
        // ALD.O (bit 79) is lowered only after tracing its vertex register back through the
        // native code by number, which a renamed operand would no longer match.
        if (insn.Bits(79, 1)) return std::nullopt;
        result.Add(16, insn.Bits(74, 2) + 1);
        result.Add(24, 1);
        result.Add(32, 1);
        return result;
    case 0x322: // AST a[Ra + offset], Rb (bits 74-75 + 1 components), Rvertex (bits 64-71)
        result.Add(24, 1);
        result.Add(32, insn.Bits(74, 2) + 1);
        result.Add(64, 1);
        return result;
    case 0x923: // ISBERD Rd, [Ra + offset]
        result.Add(16, 1);
        result.Add(24, 1);
        return result;
    case 0x927: // ISBEWR [Ra + offset], Rb
        result.Add(24, 1);
        result.Add(32, 1);
        return result;
    case 0x805: // CS2R Rd (.64 with bit 80), SR
        result.Add(16, insn.Bits(80, 1) ? 2 : 1);
        return result;
    case 0x919: case 0x806: // S2R Rd, SR; VOTE Rd
        result.Add(16, 1);
        return result;
    case 0x803: // P2R Rd, PR, Ra, mask
        result.Add(16, 1);
        result.Add(24, 1);
        return result;
    case 0x804: // R2P PR, Ra, mask
        result.Add(24, 1);
        return result;
    case 0xb66: { // TLD Rd, Ra, Rb, Rd2 (bits 64-71): the registers its lowering reads and writes
        if (std::getenv("EDEN_TLD_HOMING_OFF")) return std::nullopt; // test toggle
        // Rd takes the components (mask, bits 72-75) that Rd2 does not: up to four, or the first
        // two when Rd2 is used; FP16 results (bit 79) pack two per register. Ra is a pair for the
        // paired 2D forms (.AOFFI, .MS, .LL with Ra even), arrays and 3D; Rb is one register.
        const u32 components = std::max(1U, static_cast<u32>(std::popcount(insn.Bits(72, 4))));
        const bool half = insn.Bits(79, 1) != 0, split = insn.Bits(64, 8) != 255;
        const u32 dimension = insn.Bits(61, 3);
        const bool pair = dimension == 2 || dimension == 5 ||
                          (dimension == 1 && (insn.Bits(76, 3) || insn.Bits(87, 3) == 3));
        result.Add(16, half ? (split ? 1U : (components + 1) / 2) : (split ? 2U : components));
        result.Add(24, pair ? 2U : 1U);
        result.Add(32, 1);
        result.Add(64, half ? 1U : 2U);
        return result;
    }
    default:
        break;
    }
    // Non-uniform ALU forms (NAK encode_alu_base): Rd in 16-23, source 0 in 24-31, a register
    // source 1 in 32-39 only in form 1 (forms 4-6 hold an immediate, constant or uniform register
    // there), and bits 64-71 hold source 2 (forms 1, 4, 5, 6) or source 1 (forms 2, 3, 7, whose
    // source 2 takes bits 32-63). An operation without that source leaves 64-71 to other fields
    // (ISETP keeps its .EX predicate in 68-71, NAK OpISetP), so they count only when used.
    if (form == 0) return std::nullopt;
    const u32 low = op & 0x1ff;
    u32 sources = 0, dst = 1, wide = 1;
    switch (low) {
    case 0x002: case 0x013: case 0x045: // MOV, IABS, I2FP.F32 (Eden reads it as operand 1)
    case 0x100: case 0x101: case 0x105: case 0x106: case 0x107: case 0x108: case 0x109:
        // FLO, BREV, F2I, I2F, FRND, MUFU, POPC: the 32-bit forms (NAK: 0x111-0x113 are 64-bit),
        // the one source is operand 1.
        sources = 1;
        break;
    case 0x007: case 0x008: case 0x009: case 0x00a: // SEL, FSEL, FMNMX, FSET
    case 0x017: case 0x01a: case 0x020: case 0x021: // IMNMX, SGXT, FMUL, FADD
        sources = 2;
        break;
    case 0x00b: case 0x00c: // FSETP, ISETP: predicate results only
        sources = 2;
        dst = 0;
        break;
    case 0x010: case 0x011: case 0x012: case 0x016: // IADD3, LEA, LOP3, PRMT
    case 0x019: case 0x023: case 0x024: case 0x027: // SHF, FFMA, IMAD, IMAD.HI
        sources = 3;
        break;
    case 0x025: // IMAD.WIDE: Rd:Rd+1 and a 64-bit addend
        sources = 3;
        dst = 2;
        wide = 2;
        break;
    default:
        return std::nullopt;
    }
    const bool source1_high = form == 2 || form == 3 || form == 7;
    // FADD (NAK OpFAdd) is the only two-source operation encoded with source 1 (RZ) in 64-71.
    if (source1_high && sources < 3 && low != 0x021) return std::nullopt;
    if (dst) result.Add(16, dst);
    if (sources >= 2) result.Add(24, 1);
    if (form == 1) result.Add(32, 1);
    if (sources == 3 || source1_high) result.Add(64, wide);
    return result;
}

namespace detail {
// Programs whose registers reach the lowering's reserved range (R216-R254 with uniform
// operations, else R246-R254) keep those native registers in local memory instead: register r
// lives at byte 4 * (r - 216), and an instruction naming one runs on low temporaries that are
// saved at HomeSaveOffset and loaded from / stored back to the home slots around it.
inline constexpr u32 HomeSaveOffset = 0xa0;
inline constexpr u32 HomeRegisterArea = 0x100;

// Lower only operations whose semantics are representable by the existing
// Maxwell frontend. These are new instructions, not reinterpretations of the
// native bytes. Scheduling/control words are added by the environment adapter.
inline LoweredProgram LowerToMaxwellPass(std::span<const Instruction> code, bool home_registers,
                                        bool* reserved_conflict) {
    LoweredProgram result;
    if (home_registers) result.local_memory_bias = HomeRegisterArea;
    struct Relocation { u32 output_index; s64 target; };
    std::vector<Relocation> relocations;
    std::vector<u32> barrier_tokens;
    std::vector<u32> call_targets;
    std::vector<u32> call_links;
    bool main_exited = false;
    bool computed_branch = false; // a jump table: the program ends at its self-branch, not an EXIT
    bool uses_uniform_bank = false;
    for (const auto& instruction : code) {
        const u32 op = instruction.Bits(0, 12);
        uses_uniform_bank |= op == 0x3c4 || op == 0x2bd || op == 0xc0c || op == 0x986 || op == 0x9a8 || op == 0xc12 || op == 0xc17 || op == 0xe24 || op == 0xc24 || op == 0xd00 || op == 0xd09 || op == 0x9c3 || op == 0xc45 || op == 0x2a4 || op == 0x891 || op == 0x890 || op == 0x290 || op == 0x28c || op == 0x88c || op == 0x287 || op == 0x887 || op == 0x958 || op == 0xab9 || op == 0xabb || op == 0x899 || op == 0xc20 || op == 0xe21 || op == 0xe23 || op == 0xc02 ||
                             op == 0x886 || op == 0x3c2 || op == 0xc82 || op == 0x882 || op == 0x292 || op == 0x892 || op == 0x291 || op == 0x981 || op == 0xc13 || op == 0xc10 || op == 0xc23 || op == 0x526 || op == 0x2be || op == 0x8a4 || op == 0xd05 || (op == 0x984 && instruction.Bits(91, 1)) || op == 0x89c || op == 0x883 || op == 0x2a5 || op == 0xd06 || op == 0x299 || op == 0xc09 || (op == 0x81c && instruction.Bits(67, 1)) ||
                             op == 0xd04 || op == 0xd08 || op == 0x896 || op == 0x4a4 || op == 0x8a5 || op == 0xc11 || op == 0xc0b || op == 0x988 ||
                             op == 0x499;
    }
    // UR28-UR61 live in R176-R209 (see uniform_register), which only programs that use them
    // give up: finish() then rejects native accesses to that range.
    bool high_uniform = false;
    const auto native_gpr_at_32 = [](const Instruction& instruction) {
        const u32 op = instruction.Bits(0, 12);
        return instruction.Bits(9, 3) == 1 || op == 0x589 || op == 0x986 || op == 0x987 || op == 0x988 ||
               op == 0x98e || op == 0x9a8 || (op >= 0xb60 && op <= 0xb6f) || op == 0xb97 || op == 0xb9b;
    };
    const auto address_of = [](u32 index) { return 0x50U + index * 8 + ((index + 1) / 3) * 8; };
    const auto fail = [&](u32 index, std::string_view reason) {
        result.instructions.clear();
        result.source_instruction_offsets.clear();
        result.error_offset = index * sizeof(Instruction);
        result.error = reason;
        return result;
    };
    const auto finish = [&](u32 index) {
        if (high_uniform) {
            for (u32 i = 0; i <= index; ++i) {
                const Instruction& check = code[i];
                if (check.Bits(12, 4) == 15) continue;
                for (u32 bit : {16U, 24U, 32U, 64U}) {
                    if (bit == 32 && !native_gpr_at_32(check)) continue;
                    const u32 reg = check.Bits(bit, 8);
                    if (reg >= 176 && reg < 210) return fail(i, "native program uses lowering scratch registers");
                }
            }
        }
        for (const auto& relocation : relocations) {
            if (relocation.target > index) return fail(index, "native branch or call leaves captured program");
            const u32 target = result.source_instruction_offsets[static_cast<size_t>(relocation.target)];
            const s64 relative = s64{address_of(target)} - address_of(relocation.output_index) - 8;
            if (relative < -0x800000 || relative > 0x7fffff) return fail(index, "lowered branch distance");
            result.instructions[relocation.output_index] |= (static_cast<u64>(relative) & 0xffffff) << 20;
        }
        result.source_size = (index + 1) * sizeof(Instruction);
        return result;
    };
    if (code.empty() || code.size() > 65536) return fail(0, "invalid native program size");
    // MOV Rd, Rb / MOV Rd, imm (PT, all lanes, no modifiers): writes Rd and nothing else.
    const auto plain_mov = [](const Instruction& mov) {
        const u32 low = mov.words[0] & 0xff00ffffU;
        return mov.words[2] == 0xf00 && (low == 0x7802U || (low == 0x7202U && mov.words[1] < 256));
    };
    // A callee may reuse its link register in a straight run of plain MOVs
    //   MOV Rc, Rlink; ... MOV Rlink, x; ...; RET.REL.NODEC Rc
    // when no MOV after the copy writes Rc. The run cannot branch and keeps Rc, so on every path
    // the write changes no return target: the RET returns exactly as it would without the write
    // (case 0x950 still requires the last write to Rc before it to be that copy of the link).
    const auto releases_link = [&](u32 index, u32 entry, u32 link) {
        if (!plain_mov(code[index])) return false;
        u32 ret = index + 1;
        while (ret < code.size() && ret <= index + 8 && plain_mov(code[ret])) ++ret;
        if (ret >= code.size() || (code[ret].words[0] & 0x00ffffffU) != 0x7950U ||
            (code[ret].words[2] & ~0x3ffffU) != 0x03c00000)
            return false;
        const u32 copy = code[ret].Bits(24, 8);
        if (copy == link || copy >= 249) return false;
        for (u32 next = index + 1; next < ret; ++next) {
            if (code[next].Bits(16, 8) == copy) return false;
        }
        std::optional<u32> source;
        for (u32 prior = index; prior > entry && prior + 8 > index;) {
            const Instruction& mov = code[--prior];
            if (!plain_mov(mov)) break;
            if (mov.Bits(16, 8) == copy) {
                if ((mov.words[0] & 0xff00ffffU) == 0x7202U && mov.words[1] == link) source = prior;
                break;
            }
        }
        return source.has_value();
    };
    for (u32 index = 0; index < code.size(); ++index) {
        auto insn = code[index];
        result.source_instruction_offsets.push_back(static_cast<u32>(result.instructions.size()));
        u32 opcode = insn.Bits(0, 12);
        if (insn.Bits(12, 4) == 15) continue; // @!PT: never executes (compiler padding)
        std::optional<u32> active_link;
        u32 active_entry = 0;
        if (main_exited) {
            for (size_t call = 0; call < call_targets.size(); ++call) {
                if (call_targets[call] <= index && call_targets[call] >= active_entry) {
                    active_entry = call_targets[call];
                    active_link = call_links[call];
                }
            }
            // Bits 16-23 are no GPR destination of FSETP/ISETP (they write predicates 81-86), BRA,
            // RET, EXIT, BSSY/BSYNC/BREAK (16-19: a convergence barrier), WARPSYNC or NOP (NAK
            // sm70_encode.rs), so these cannot overwrite the link.
            const bool no_register_destination =
                opcode == 0x20b || opcode == 0x80b || opcode == 0xa0b || opcode == 0x20c ||
                opcode == 0x80c || opcode == 0xa0c || opcode == 0xc0c || opcode == 0x947 ||
                opcode == 0x950 || opcode == 0x94d || opcode == 0x945 || opcode == 0x941 ||
                opcode == 0x942 || opcode == 0x948 || opcode == 0x918;
            if (active_link && insn.Bits(16, 8) == *active_link && !no_register_destination &&
                !releases_link(index, active_entry, *active_link))
                return fail(index, "native callee overwrites link register");
        }
        // Scratch values and emulated uniform state occupy reserved high GPRs.
        const u32 reserved_base = uses_uniform_bank ? 216U : 246U;
        struct HomeRun { u32 native, temporary, count; };
        std::array<HomeRun, 4> home_runs{};
        u32 home_run_count = 0;
        u32 home_window = 0;
        // LDL/STL of `count` registers from `reg` at absolute local byte `offset`, unpredicated,
        // 128/64-bit where the register (and so the slot, which has the same alignment) allows.
        const auto home_transfer = [&](bool store, u32 reg, u32 count, u32 offset) {
            for (u32 k = 0; k < count;) {
                const u32 n = ((reg + k) % 4 == 0 && count - k >= 4) ? 4U
                            : ((reg + k) % 2 == 0 && count - k >= 2) ? 2U : 1U;
                result.instructions.push_back(
                    (store ? 0xef50000000000000ULL : 0xef40000000000000ULL) | (u64{7} << 16) |
                    (reg + k) | (u64{255} << 8) | (u64{offset + 4 * k} << 20) |
                    (u64{n == 4 ? 6U : n == 2 ? 5U : 4U} << 48));
                k += n;
            }
        };
        if (!home_registers) {
            // Reject native accesses that could alias that state; LowerToMaxwell then lowers the
            // program again with the native registers of that range kept in local memory.
            for (u32 bit : {16U, 24U, 32U, 64U}) {
                if (bit == 32 && insn.Bits(9, 3) != 1 && opcode != 0xb60 && opcode != 0xb66) continue;
                const u32 reg = insn.Bits(bit, 8);
                if (reg >= reserved_base && reg < 255) {
                    if (reserved_conflict) *reserved_conflict = true;
                    return fail(index, "native program uses lowering scratch registers");
                }
            }
        } else if (const auto operands = DescribeNativeGprs(insn); !operands) {
            // Operand layout not described: no field that can hold a GPR (NAK: bits 16, 24, 32
            // and 64, at most four registers each) may reach the registers kept in local memory.
            for (u32 bit : {16U, 24U, 32U, 64U}) {
                const u32 reg = insn.Bits(bit, 8);
                if (reg < 255 && reg + 3 >= reserved_base)
                    return fail(index, "native program uses lowering scratch registers");
            }
        } else {
            // Native registers from reserved_base up live in local memory. The instruction runs on
            // temporaries in one 16-register window below R176 that it does not name (each keeps
            // its native register's alignment mod 4): save them, load the home slots, lower the
            // renamed instruction, store the slots back and restore the temporaries (end of loop).
            // Every named register is loaded and stored, so the result is right whichever of them
            // the instruction reads or writes, and a false guard stores back what was loaded.
            std::array<bool, 256> named{};
            std::array<HomeRun, 4> spans{};
            u32 span_count = 0;
            for (u32 i = 0; i < operands->count; ++i) {
                const auto field = operands->fields[i];
                const u32 reg = insn.Bits(field.bit, 8);
                if (reg == 255) continue;
                const u32 last = std::min(reg + field.width - 1, 254U);
                for (u32 r = reg; r <= last; ++r) named[r] = true;
                if (reg >= reserved_base) {
                    // An operand running past R254 also names RZ, which its renamed run of
                    // temporaries cannot keep (CS2R.64 R254 would write the next temporary).
                    if (reg + field.width - 1 > 254) return fail(index, "native program uses lowering scratch registers");
                    // The cases' own wide-destination link checks see the renamed register, so a
                    // destination run over a callee's link is checked here (bit 16: the destination).
                    if (field.bit == 16 && active_link && *active_link > reg && *active_link < reg + field.width)
                        return fail(index, "native callee overwrites link register");
                    spans[span_count++] = {reg, 0, last - reg + 1};
                } else if (last >= reserved_base) {
                    return fail(index, "native program uses lowering scratch registers");
                }
            }
            if (span_count) {
                if (operands->control_flow) return fail(index, "native program uses lowering scratch registers");
                std::sort(spans.begin(), spans.begin() + span_count,
                          [](const HomeRun& a, const HomeRun& b) { return a.native < b.native; });
                for (u32 i = 0; i < span_count; ++i) {
                    HomeRun* previous = home_run_count ? &home_runs[home_run_count - 1] : nullptr;
                    // Overlapping or adjacent spans share one run (one displacement keeps every
                    // register of a multi-register operand consecutive).
                    if (previous && spans[i].native <= previous->native + previous->count) {
                        previous->count = std::max(previous->count, spans[i].native + spans[i].count - previous->native);
                    } else {
                        home_runs[home_run_count++] = spans[i];
                    }
                }
                bool found = false;
                for (u32 window = 0; window < 11 && !found; ++window) {
                    found = std::none_of(named.begin() + window * 16, named.begin() + window * 16 + 16,
                                         [](bool used) { return used; });
                    if (found) home_window = window * 16;
                }
                if (!found) return fail(index, "native program uses lowering scratch registers");
                u32 cursor = home_window;
                for (u32 i = 0; i < home_run_count; ++i) {
                    HomeRun& run = home_runs[i];
                    run.temporary = cursor + ((run.native - cursor) & 3);
                    cursor = run.temporary + run.count;
                    if (cursor > home_window + 16) return fail(index, "native program uses lowering scratch registers");
                }
                for (u32 i = 0; i < home_run_count; ++i) {
                    const HomeRun& run = home_runs[i];
                    home_transfer(true, run.temporary, run.count, HomeSaveOffset + 4 * (run.temporary - home_window));
                    home_transfer(false, run.temporary, run.count, 4 * (run.native - 216));
                }
                for (u32 i = 0; i < operands->count; ++i) {
                    const auto field = operands->fields[i];
                    const u32 reg = insn.Bits(field.bit, 8);
                    if (reg == 255 || reg < reserved_base) continue;
                    for (u32 k = 0; k < home_run_count; ++k) {
                        const HomeRun& run = home_runs[k];
                        if (reg < run.native || reg >= run.native + run.count) continue;
                        u32& word = insn.words[field.bit / 32];
                        word = (word & ~(0xffU << (field.bit % 32))) |
                               ((reg - run.native + run.temporary) << (field.bit % 32));
                        break;
                    }
                }
            }
        }
        // UR0-UR21 live in R216-R237 and UR22-UR27 in R240-R245 (R239 holds the uniform
        // predicates), so even/odd pairs stay adjacent. Sites that still check reg < 16 rely on
        // UR0-UR15 being R216-R231. UR28-UR61 live in R176-R209 (even/odd pairs adjacent too);
        // using one makes finish() reject native accesses to R176-R209. UR62 stays unmapped, so a
        // 64-bit UR62:URZ pair never reads R210 as its high word.
        const auto uniform_register = [&](u32 reg) -> u32 {
            if (reg < 22) return 216 + reg;
            if (reg < 28) return 218 + reg;
            if (reg == 63) return 255;
            if (reg < 62) {
                high_uniform = true;
                return 148 + reg;
            }
            return ~0U;
        };
        bool rewritten_uniform = false;
        bool uniform_guard = false;
        u32 uniform_output = 7; // ULOP3's UPn output: computed in P(6-n), then merged into R239
        if (opcode == 0x2a4 || opcode == 0x8a4 || opcode == 0x891 || opcode == 0x890 || opcode == 0x290 ||
            opcode == 0x292 || opcode == 0x892 || opcode == 0x291 || opcode == 0x896 || opcode == 0x4a4) {
            // A UPn guard is swapped into P(6-n) around the rewritten op (guarded_uniform below);
            // forms whose lowering itself saves predicates in R253 reject it (uniform_guard).
            uniform_guard = insn.Bits(12, 4) != 7;
            rewritten_uniform = true;
            const u32 dst = uniform_register(insn.Bits(16, 8));
            const u32 a = uniform_register(insn.Bits(24, 8));
            const u32 c = uniform_register(insn.Bits(64, 8));
            if (dst == ~0U || a == ~0U || c == ~0U) return fail(index, "native uniform arithmetic register");
            insn.words[0] = (insn.words[0] & 0xffff) | (dst << 16) | (a << 24);
            insn.words[2] = (insn.words[2] & ~0x080000ffU) | c;
            if (opcode == 0x2a4 || opcode == 0x290 || opcode == 0x292 || opcode == 0x291) {
                const u32 b = uniform_register(insn.Bits(32, 8));
                if (b == ~0U) return fail(index, "native uniform arithmetic second register");
                insn.words[1] = (insn.words[1] & ~0xffU) | b;
            }
            if ((opcode == 0x292 || opcode == 0x892) && insn.Bits(80, 4) != 0xe) {
                if (insn.Bits(80, 1)) return fail(index, "native uniform logic predicate output");
                uniform_output = insn.Bits(81, 3);
                insn.words[2] = (insn.words[2] & ~(7U << 17)) | ((6 - uniform_output) << 17);
            }
            opcode = opcode == 0x2a4 ? 0x224 : opcode == 0x8a4 ? 0x824 : opcode == 0x891 ? 0x811 : opcode == 0x890 ? 0x810
                   : opcode == 0x292 ? 0x212 : opcode == 0x892 ? 0x812 : opcode == 0x291 ? 0x211
                   : opcode == 0x896 ? 0x816 : opcode == 0x4a4 ? 0x424 : 0x210;
        }
        const bool uniform_predication = opcode == 0xabb || opcode == 0xab9 || opcode == 0xc82 || opcode == 0x882 ||
                                         opcode == 0x899 || opcode == 0x499;
        const u32 native_predicate = insn.Bits(12, 4);
        const bool guarded_uniform = (uniform_predication || uniform_guard) && native_predicate != 7;
        // A uniform guard UPn lives in R239 bit 6-n and is swapped into P(6-n) for the op.
        const u32 guard_slot = 6 - (native_predicate & 7);
        u64 predicate = u64{guarded_uniform ? ((native_predicate & 8) | guard_slot) : native_predicate} << 16;
        const u64 dest = insn.Bits(16, 8);
        const auto base = [&](u64 opcode_bits) { return opcode_bits | predicate | dest; };
        // NVN2 fullscreen vertex shaders read bank 24, which the game never binds (the console reads
        // zeros from an unbound bank); Maxwell has 18, so read it from bank 17, also never bound.
        const auto bank = [&]() -> u32 {
            const u32 value = insn.Bits(54, 5);
            return value == 24 ? 17U : value;
        };
        const auto cbuf = [&]() -> u64 {
            return (u64{insn.Bits(38, 16) / 4} << 20) | (u64{bank()} << 34);
        };
        const auto valid_cbuf = [&] {
            return !insn.Bits(91, 1) && bank() < 18 &&
                   (insn.Bits(38, 16) % 4) == 0;
        };
        std::string_view operand_error;
        const auto emit = [&](u64 word) { result.instructions.push_back(word); };
        // P(guard slot) while a uniform guard is swapped in; P6 for lowerings that use it as
        // scratch (R2UR, MATCH, HFMA2.RELU), which save and restore it with the defaults below.
        const u32 guard_mask = guard_slot < 7 ? 1U << guard_slot : 0U;
        const auto save_predicate = [&](u32 mask = 1U << 6) {
            emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
        };
        const auto restore_predicate = [&](u32 mask = 1U << 6) {
            emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
        };
        if (guarded_uniform) {
            save_predicate(guard_mask);
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{guard_mask} << 20));
        }
        const u32 output_mask = uniform_output == 7 ? 0U : 1U << (6 - uniform_output);
        if (output_mask) {
            // Keep the guest P(6-n) in R254 and start from the current UPn (a false guard keeps it).
            emit(0x38e8000000070000ULL | 254 | (u64{255} << 8) | (u64{output_mask} << 20));
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{output_mask} << 20));
        }
        // Pd = LUT(a, b, c) over predicate operands (index | negate << 3): each input becomes
        // 0 or ~0 (SEL), and LOP3 sets Pd when the result is nonzero.
        const auto predicate_lut = [&](u32 lut, u32 a, u32 b, u32 c, u32 target, bool uniform_c = false) {
            const u32 inputs[3]{a, b, c};
            for (u32 i = 0; i < 3; ++i) {
                if (i == 2 && uniform_c) {
                    // UPn from R239 bit 6-n as 0 or ~0 (UPT is ~0); a negation flips the table.
                    if ((c & 7) == 7) {
                        emit(0x0100000000000000ULL | predicate | 252 | (u64{15} << 12) | (u64{~0U} << 20));
                    } else {
                        emit(0x3800000000000000ULL | predicate | 252 | (u64{239} << 8) |
                             (u64{(6 - (c & 7)) | (1U << 8)} << 20));
                        emit(0x5c10000000000000ULL | predicate | 252 | (u64{255} << 8) |
                             (u64{252} << 20) | (u64{1} << 48));
                    }
                    if (c & 8) lut = ((lut & 0x55) << 1) | ((lut & 0xaa) >> 1);
                    continue;
                }
                emit(0x38a0000000000000ULL | predicate | (250 + i) | (u64{255} << 8) |
                     (u64{0x7ffff} << 20) | (u64{1} << 56) | (u64{inputs[i] ^ 8U} << 39));
            }
            emit(0x5be0000000000000ULL | predicate | 255 | (u64{250} << 8) | (u64{251} << 20) |
                 (u64{lut} << 28) | (u64{3} << 36) | (u64{252} << 39) | (u64{target} << 48));
        };
        const auto mov_imm = [&](u32 reg, u32 value) {
            emit(0x0100000000000000ULL | predicate | reg | (u64{15} << 12) | (u64{value} << 20));
        };
        const auto materialize = [&](u32 kind, u32 reg, u32 scratch, bool absolute, bool negate) -> u32 {
            if (kind == 3) {
                const u32 mapped = uniform_register(reg);
                if (mapped == ~0U) { operand_error = "native uniform source register"; return 255; }
                reg = mapped;
            } else if (kind == 1) {
                mov_imm(scratch, insn.words[1]);
                reg = scratch;
            } else if (kind == 2) {
                if (!valid_cbuf()) { operand_error = "native constant buffer address"; return 255; }
                emit(0x4c98000000000000ULL | predicate | scratch | (u64{15} << 39) | cbuf());
                reg = scratch;
            }
            if (absolute || negate) {
                if (reg != scratch) emit(0x5c98000000000000ULL | predicate | scratch | (u64{reg} << 20) | (u64{15} << 39));
                if (absolute) emit(0x0400000000000000ULL | predicate | scratch | (u64{scratch} << 8) | (u64{0x7fffffff} << 20));
                if (negate) emit(0x0400000000000000ULL | predicate | scratch | (u64{scratch} << 8) | (u64{0x80000000} << 20) | (u64{2} << 53));
                reg = scratch;
            }
            return reg;
        };
        const auto operand = [&](u32 which, bool floating = true) -> u32 {
            const u32 form = insn.Bits(9, 3);
            u32 reg{}, kind{}, abs{}, neg{};
            if (which == 0) {
                reg = insn.Bits(24, 8); abs = insn.Bits(73, 1); neg = insn.Bits(72, 1);
            } else {
                if (form != 1 && form != 2 && form != 3 && form != 4 && form != 5 && form != 6 && form != 7) {
                    operand_error = "native ALU source form"; return 255;
                }
                const bool wide = which == 1 ? (form == 4 || form == 5 || form == 6) : (form == 2 || form == 3 || form == 7);
                if (wide) {
                    kind = form == 6 || form == 7 ? 3 : (form == 4 || form == 2) ? 1 : 2;
                    if (kind == 3) reg = insn.Bits(32, 8);
                    if (kind == 2 || kind == 3) { abs = insn.Bits(62, 1); neg = insn.Bits(63, 1); }
                } else {
                    const bool upper = which == 1 ? (form == 2 || form == 3 || form == 7) : true;
                    reg = insn.Bits(upper ? 64 : 32, 8);
                    abs = insn.Bits(upper ? 74 : 62, 1); neg = insn.Bits(upper ? 75 : 63, 1);
                }
            }
            return materialize(kind, reg, 250 + which, floating && abs, floating && neg);
        };
        struct HalfSource { u32 reg, swizzle; bool absolute, negate; };
        const auto half_source = [&](u32 which) -> HalfSource {
            if (which == 0) return {insn.Bits(24, 8), insn.Bits(74, 2),
                                    insn.Bits(73, 1) != 0, insn.Bits(72, 1) != 0};
            const u32 form = insn.Bits(9, 3);
            const bool wide = which == 1 ? (form == 4 || form == 5) : (form == 2 || form == 3);
            if (wide) {
                const bool immediate = form == 2 || form == 4;
                const u32 reg = materialize(immediate ? 1 : 2, 255, 250 + which, false, false);
                return {reg, immediate ? 0U : insn.Bits(60, 2),
                        !immediate && insn.Bits(62, 1), !immediate && insn.Bits(63, 1)};
            }
            const bool upper = which == 2 || form == 2 || form == 3;
            return {insn.Bits(upper ? 64 : 32, 8), insn.Bits(upper ? 81 : 60, 2),
                    insn.Bits(upper ? 83 : 62, 1) != 0, insn.Bits(upper ? 84 : 63, 1) != 0};
        };
        const auto extract_half = [&](HalfSource source, u32 lane, u32 target) {
            const bool high = source.swizzle == 3 || (source.swizzle == 0 && lane == 1);
            emit(0x5ca8000000000000ULL | predicate | target | (u64{2} << 8) |
                 (u64{source.swizzle == 1 ? 2U : 1U} << 10) | (u64{source.reg} << 20) |
                 (u64{high} << 41) | (u64{source.negate} << 45) | (u64{source.absolute} << 49));
        };
        // Texture instructions with FP16 results (bit 79) pack two components per register,
        // the first in the low half (lane 0, as HADD2/HMUL2 and F2FP read and write it): (c0, c1)
        // of the 32-bit results in R250-R253 into `first`, (c2, c3) into `second`. F2F writes
        // the half in the low 16 bits; ISCADD computes (Ra << 16) + Rb.
        const auto pack_half_results = [&](u32 first, u32 second, u32 components) {
            const auto pack = [&](u32 target, u32 low, u32 high) {
                if (target == 255) return;
                emit(0x5ca8000000000000ULL | predicate | 246 | (u64{1} << 8) | (u64{2} << 10) | (u64{low} << 20));
                emit(0x5ca8000000000000ULL | predicate | 247 | (u64{1} << 8) | (u64{2} << 10) | (u64{high} << 20));
                emit(0x5c18000000000000ULL | predicate | target | (u64{247} << 8) | (u64{246} << 20) | (u64{16} << 39));
            };
            pack(first, 250, components > 1 ? 251U : 255U);
            if (components > 2) pack(second, 252, components > 3 ? 253U : 255U);
        };
        const auto xmad = [&](u32 target, u32 a, u32 b, u32 c, bool a_high,
                              bool b_high, u32 mode, bool immediate) {
            emit((immediate ? 0x3600000000000000ULL : 0x5b00000000000000ULL) |
                 predicate | target | (u64{a} << 8) |
                 (u64{immediate ? ((b >> (b_high ? 16 : 0)) & 0xffff) : b} << 20) |
                 (u64{!immediate && b_high} << 35) | (u64{c} << 39) |
                 (u64{mode} << 50) | (u64{a_high} << 53));
        };
        const auto multiply_low = [&](u32 a, u32 b, u32 c, bool immediate) {
            xmad(250, a, b, c, false, false, 0, immediate);
            xmad(251, a, b, 255, true, false, 0, immediate);
            xmad(251, a, b, 251, false, true, 0, immediate);
            emit(base(0x5c18000000000000ULL) | (u64{251} << 8) |
                 (u64{250} << 20) | (u64{16} << 39));
        };
        switch (opcode) {
        case 0x304: case 0xb04: case 0xd04: { // F2F.F16.F32, scalar round-to-nearest conversion.
            // 0xb04: constant-buffer source (offset 38-53, bank 54-58; bits 32-37 and 59-61 zero);
            // 0xd04: uniform source (bits 32-39, bit 91). |B| is bit 62 and -B bit 63 in all three.
            if (insn.words[2] != (opcode == 0xd04 ? 0x08200800U : 0x00200800U) || insn.Bits(24, 8) ||
                (insn.words[1] & (opcode == 0xb04 ? 0x3800003fU : ~0xc00000ffU)))
                return fail(index, "native scalar float conversion mode");
            const u32 source = opcode == 0x304 ? insn.Bits(32, 8)
                             : materialize(opcode == 0xb04 ? 2 : 3, insn.Bits(32, 8), 250, false, false);
            emit(base(0x5ca8000000000000ULL) | (u64{1} << 8) | (u64{2} << 10) |
                 (u64{source} << 20) | (u64{insn.Bits(62, 1)} << 49) |
                 (u64{insn.Bits(63, 1)} << 45));
            break;
        }
        case 0x23e: case 0xa3e: { // F2FP.PACK_AB / MERGE_C, round-to-nearest
            if (insn.Bits(62, 2) || insn.Bits(73, 5) || insn.Bits(79, 2) ||
                insn.Bits(81, 10) || insn.Bits(91, 1)) {
                return fail(index, "native float packing modifiers");
            }
            const bool merge = insn.Bits(78, 1);
            if (insn.Bits(72, 1)) return fail(index, "native float packing selector");
            const u32 a = merge ? operand(2, false) : operand(0, false);
            const u32 b = operand(1, false);
            // Canonical F2F writes the converted half in its low 16 bits.
            if (merge) {
                emit(0x5c98000000000000ULL | predicate | 250 | (u64{a} << 20) | (u64{15} << 39));
            } else {
                emit(0x5ca8000000000000ULL | predicate | 250 | (u64{1} << 8) |
                     (u64{2} << 10) | (u64{a} << 20));
            }
            emit(0x5ca8000000000000ULL | predicate | 251 | (u64{1} << 8) |
                 (u64{2} << 10) | (u64{b} << 20));
            emit(base(0x5c18000000000000ULL) | (u64{250} << 8) | (u64{251} << 20) | (u64{16} << 39));
            break;
        }
        case 0x232: case 0x832: case 0xa32: { // HMUL2
            if (insn.Bits(78, 2) || insn.Bits(81, 10) || insn.Bits(91, 1))
                return fail(index, "native half multiply mode");
            const auto a = half_source(0), b = half_source(1);
            const u32 precision = insn.Bits(76, 1) ? 2 : insn.Bits(80, 1);
            emit(base(0x5d08000000000000ULL) | (u64{a.reg} << 8) | (u64{b.reg} << 20) |
                 (u64{b.swizzle} << 28) | (u64{b.absolute} << 30) | (u64{a.negate != b.negate} << 31) |
                 (u64{insn.Bits(77, 1)} << 32) | (u64{precision} << 39) |
                 (u64{a.absolute} << 44) | (u64{a.swizzle} << 47));
            break;
        }
        case 0x230: case 0x430: case 0x630: { // HADD2
            if (insn.Bits(76, 1) || insn.Bits(79, 1) || insn.Bits(81, 10) || insn.Bits(91, 1))
                return fail(index, "native half add mode");
            const auto a = half_source(0);
            const auto b = half_source(opcode == 0x230 ? 1 : 2);
            if (insn.Bits(78, 1)) {
                if (a.reg != 255 || !a.negate || a.absolute || a.swizzle != 0 ||
                    b.swizzle < 2 || insn.Bits(77, 1) || insn.Bits(80, 1))
                    return fail(index, "native float-output half add mode");
                extract_half(b, 0, static_cast<u32>(dest));
            } else {
                emit(base(0x5d10000000000000ULL) | (u64{a.reg} << 8) | (u64{b.reg} << 20) |
                     (u64{b.swizzle} << 28) | (u64{b.absolute} << 30) | (u64{b.negate} << 31) |
                     (u64{insn.Bits(77, 1)} << 32) | (u64{insn.Bits(80, 1)} << 39) |
                     (u64{a.negate} << 43) | (u64{a.absolute} << 44) | (u64{a.swizzle} << 47));
            }
            break;
        }
        case 0x231: case 0x431: case 0x631: case 0x831: case 0xa31: { // HFMA2
            const bool relu = insn.Bits(79, 1) != 0;
            if (insn.Bits(78, 1) || insn.Bits(85, 1) || insn.Bits(91, 1) ||
                (relu ? (insn.Bits(87, 4) != 7 ||
                         insn.Bits(77, 1)) : insn.Bits(87, 4) != 0))
                return fail(index, "native half fused multiply-add mode");
            if (relu && native_predicate != 7) {
                // Guard the whole expansion before its temporary P6 updates.
                relocations.push_back({static_cast<u32>(result.instructions.size()), index + 1});
                emit(0xe24000000000000fULL | (u64{native_predicate ^ 8U} << 16));
                predicate = u64{7} << 16;
            }
            auto a = half_source(0);
            auto b = half_source(1);
            auto c = half_source(2);
            if (insn.Bits(86, 1)) { // H0_NH1 (written high lane first): high = H0, low = -H1.
                if (b.swizzle != 0 || b.absolute || b.reg >= 250)
                    return fail(index, "native half mixed-sign source mode");
                // Swap the FP16 halves (rotate by 16), then negate the new low half. It used to
                // negate the high half in place, which broke HWAI's shadow PCF filter
                // (3d36b1e7e0ecc856) and FSR1 EASU tap rotation (d1835c1e213e0941) every frame.
                // Same form as Lichii (11-04).
                emit(0x36f8000000000000ULL | predicate | 251 | (u64{b.reg} << 8) |
                     (u64{16} << 20) | (u64{b.reg} << 39));
                emit(0x0400000000000000ULL | predicate | 251 | (u64{251} << 8) |
                     (u64{0x8000} << 20) | (u64{2} << 53));
                b.reg = 251;
            }
            if (b.absolute || (a.absolute && a.reg >= 248))
                return fail(index, "native half fused multiply-add absolute source");
            if (a.absolute) {
                emit(0x0400000000000000ULL | predicate | 250 | (u64{a.reg} << 8) |
                     (u64{a.swizzle == 1 ? 0x7fffffffU : 0x7fff7fffU} << 20));
                a.reg = 250;
                a.absolute = false;
            }
            if (c.absolute) {
                // |C| (bit 83, or bit 62 of a constant-buffer C): Maxwell HFMA2 has only neg_c, so
                // clear the sign bits in R252 (the wide-C scratch register, otherwise unused here).
                emit(0x0400000000000000ULL | predicate | 252 | (u64{c.reg} << 8) |
                     (u64{c.swizzle == 1 ? 0x7fffffffU : 0x7fff7fffU} << 20));
                c.reg = 252;
                c.absolute = false;
            }
            const u32 precision = insn.Bits(76, 1) ? 2 : insn.Bits(80, 1);
            emit(0x5d00000000000000ULL | predicate | (relu ? 254U : dest) | (u64{a.reg} << 8) | (u64{b.reg} << 20) |
                 (u64{b.swizzle} << 28) | (u64{c.negate} << 30) | (u64{a.negate != b.negate} << 31) |
                 (u64{insn.Bits(77, 1)} << 32) | (u64{c.swizzle} << 35) |
                 (u64{precision} << 37) | (u64{c.reg} << 39) | (u64{a.swizzle} << 47));
            if (relu) {
                // Clamp the rounded binary16 lanes, preserving positive infinity
                // and canonicalizing either sign of NaN without saturating to one.
                save_predicate();
                for (u32 lane = 0; lane < 2; ++lane) {
                    emit(0x3828000000070000ULL | 251 | (u64{254} << 8) | (u64{lane * 16} << 20));
                    emit(0x0400000000070000ULL | 251 | (u64{251} << 8) | (u64{0xffff} << 20));
                    emit(0x0400000000070000ULL | 252 | (u64{251} << 8) | (u64{0x7fff} << 20));
                    mov_imm(250, 0x7c00);
                    emit(0x5b60000000070000ULL | 7 | (u64{6} << 3) | (u64{250} << 8) |
                         (u64{252} << 20) | (u64{7} << 39) | (u64{1} << 49));
                    emit(0x3828000000070000ULL | 252 | (u64{251} << 8) | (u64{15} << 20));
                    mov_imm(250, 1);
                    emit(0x5c10000000070000ULL | 252 | (u64{252} << 8) | (u64{250} << 20) | (u64{1} << 48));
                    emit(0x5c40000000070000ULL | 251 | (u64{251} << 8) | (u64{252} << 20) | (u64{7} << 48));
                    mov_imm(250, 0x7fff);
                    emit(0x5ca0000000070000ULL | 251 | (u64{250} << 8) | (u64{251} << 20) | (u64{6} << 39));
                    if (lane == 0) {
                        emit(base(0x5c98000000000000ULL) | (u64{251} << 20) | (u64{15} << 39));
                    } else {
                        emit(0x3848000000070000ULL | 251 | (u64{251} << 8) | (u64{16} << 20));
                        emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{251} << 20) |
                             (u64{1} << 41) | (u64{7} << 48));
                    }
                }
                restore_predicate();
            }
            break;
        }
        case 0x240: case 0x840: { // HMNMX2, exact half promotion for min/max
            if (insn.Bits(76, 11) || insn.Bits(91, 1))
                return fail(index, "native half min/max mode");
            const auto a = half_source(0);
            const auto b = opcode == 0x840 ? HalfSource{255, 0, false, false} : half_source(1);
            if (a.swizzle == 1 || b.swizzle == 1 || (b.reg >= 250 && b.reg != 255))
                return fail(index, "native half min/max source mode");
            for (u32 lane = 0; lane < 2; ++lane) {
                extract_half(a, lane, 250);
                if (opcode == 0x840) {
                    const u32 half = (insn.words[1] >> (lane * 16)) & 0xffff;
                    mov_imm(251, half);
                    extract_half(HalfSource{251, 2, false, false}, 0, 251);
                } else {
                    extract_half(b, lane, 251);
                }
                const u32 target = lane == 0 ? 252 : 250;
                emit(0x5c60000000000000ULL | predicate | target | (u64{250} << 8) | (u64{251} << 20) |
                     (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                emit(0x5ca8000000000000ULL | predicate | target | (u64{1} << 8) |
                     (u64{2} << 10) | (u64{target} << 20));
            }
            emit(base(0x5c18000000000000ULL) | (u64{250} << 8) | (u64{252} << 20) | (u64{16} << 39));
            break;
        }
        case 0xc24: case 0xe24: { // IMAD low word with a uniform multiplier or addend.
            // 0xe24 keeps the uniform addend in bits 32-37; bit 63 negates it ("RZ, RZ, -UR4").
            const bool negate_c = opcode == 0xe24 && insn.Bits(63, 1) != 0;
            if ((insn.words[2] & ~0x080002ffU) != 0x078e0000 ||
                (insn.words[1] & ~(0xffU | (negate_c ? 0x80000000U : 0U))))
                return fail(index, "native uniform-addend multiply mode");
            const u32 a = operand(0, false), b = operand(1, false);
            u32 c = operand(2, false);
            if (negate_c) {
                emit(0x5c10000000000000ULL | predicate | 252 | (u64{255} << 8) | (u64{c} << 20) |
                     (u64{1} << 48));
                c = 252;
            }
            multiply_low(a, b, c, false);
            break;
        }
        case 0x224: { // IMAD register / zero-product move
            if (insn.Bits(24, 8) == 255 && insn.words[1] == 255 && insn.Bits(75, 1) &&
                (insn.words[2] & ~0xaffU) == 0x078e0000) {
                emit(base(0x5c10000000000000ULL) | (u64{255} << 8) |
                    (u64{insn.Bits(64, 8)} << 20) | (u64{1} << 48));
                break;
            }
            if (insn.Bits(24, 8) == 255 && insn.words[1] == 255 &&
                (insn.words[2] & ~0xffU) == 0x078e0000) {
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{insn.Bits(64, 8)} << 20));
                break;
            }
            if (!rewritten_uniform && insn.Bits(24, 8) == 255 && insn.words[1] == 255 &&
                (insn.words[2] & ~0x078000ffU) == 0x000e0400) {
                // IMAD.X Rd, RZ, RZ, Rc, Pc (bit 74, carry-in 87-90): Rc + Pc, a 64-bit high word.
                // Not for UIMAD (rewritten to 0x224), whose carry-in UPn lives in R239.
                emit(0x38a0000000000000ULL | predicate | 251 | (u64{255} << 8) | (u64{1} << 20) |
                     (u64{insn.Bits(87, 4) ^ 8U} << 39));
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(64, 8)} << 8) | (u64{251} << 20));
                break;
            }
            if ((insn.words[2] & ~0x2ffU) != 0x078e0000 || (insn.words[1] & ~0xffU))
                return fail(index, "native integer register multiply mode");
            multiply_low(insn.Bits(24, 8), insn.Bits(32, 8), insn.Bits(64, 8), false);
            break;
        }
        case 0x2a5: { // UIMAD.WIDE.U32 UR(d):UR(d+1) = URa * URb + URc:UR(c+1)
            const u32 a = uniform_register(insn.Bits(24, 8)), b = uniform_register(insn.Bits(32, 8));
            const u32 c = insn.Bits(64, 8);
            const u32 c_low = uniform_register(c), c_high = c == 63 ? 255U : uniform_register(c + 1);
            const u32 d_low = uniform_register(insn.Bits(16, 8)), d_high = uniform_register(insn.Bits(16, 8) + 1);
            if (native_predicate != 7 || (insn.words[1] & ~0xffU) || (insn.words[2] & ~0xffU) != 0x0f8e0000 ||
                (insn.Bits(16, 8) & 1) || (c != 63 && (c & 1)) || a == ~0U || b == ~0U || c_low == ~0U ||
                c_high == ~0U || d_low == ~0U || d_high == ~0U || d_low == 255)
                return fail(index, "native uniform wide multiply mode");
            // High word in R253 (base-2^16 long multiplication, as IMAD.HI), low word in R250.
            xmad(250, a, b, 255, false, false, 0, false);
            xmad(251, a, b, 250, true, false, 2, false);
            xmad(250, a, b, 251, false, true, 1, false);
            xmad(252, a, b, 251, true, true, 2, false);
            emit(0x3828000000070000ULL | 250 | (u64{250} << 8) | (u64{16} << 20));
            emit(0x5c10000000070000ULL | 253 | (u64{250} << 8) | (u64{252} << 20));
            xmad(250, a, b, 255, false, false, 0, false);
            xmad(251, a, b, 255, true, false, 0, false);
            xmad(251, a, b, 251, false, true, 0, false);
            emit(0x5c18000000070000ULL | 250 | (u64{251} << 8) | (u64{250} << 20) | (u64{16} << 39));
            emit(0x5c10000000070000ULL | 250 | (u64{250} << 8) | (u64{c_low} << 20) | (u64{1} << 47));
            emit(0x5c10000000070000ULL | d_high | (u64{253} << 8) | (u64{c_high} << 20) | (u64{1} << 43));
            emit(0x5c10000000070000ULL | d_low | (u64{250} << 8) | (u64{255} << 20));
            break;
        }
        case 0x8a5: { // UIMAD.WIDE.U32 UR(d):UR(d+1) = URa * imm + URc:UR(c+1), as 0x2a5 with an immediate B.
            const u32 a = uniform_register(insn.Bits(24, 8)), b = insn.words[1];
            const u32 c = insn.Bits(64, 8);
            const u32 c_low = uniform_register(c), c_high = c == 63 ? 255U : uniform_register(c + 1);
            const u32 d_low = uniform_register(insn.Bits(16, 8)), d_high = uniform_register(insn.Bits(16, 8) + 1);
            if (native_predicate != 7 || (insn.words[2] & ~0xffU) != 0x0f8e0000 ||
                (insn.Bits(16, 8) & 1) || (c != 63 && (c & 1)) || a == ~0U || c_low == ~0U ||
                c_high == ~0U || d_low == ~0U || d_high == ~0U || d_low == 255)
                return fail(index, "native uniform wide multiply mode");
            // High word in R253 (base-2^16 long multiplication, as IMAD.HI), low word in R250;
            // XMAD's 16-bit immediate takes the selected half of imm.
            xmad(250, a, b, 255, false, false, 0, true);
            xmad(251, a, b, 250, true, false, 2, true);
            xmad(250, a, b, 251, false, true, 1, true);
            xmad(252, a, b, 251, true, true, 2, true);
            emit(0x3828000000070000ULL | 250 | (u64{250} << 8) | (u64{16} << 20));
            emit(0x5c10000000070000ULL | 253 | (u64{250} << 8) | (u64{252} << 20));
            xmad(250, a, b, 255, false, false, 0, true);
            xmad(251, a, b, 255, true, false, 0, true);
            xmad(251, a, b, 251, false, true, 0, true);
            emit(0x5c18000000070000ULL | 250 | (u64{251} << 8) | (u64{250} << 20) | (u64{16} << 39));
            emit(0x5c10000000070000ULL | 250 | (u64{250} << 8) | (u64{c_low} << 20) | (u64{1} << 47));
            emit(0x5c10000000070000ULL | d_high | (u64{253} << 8) | (u64{c_high} << 20) | (u64{1} << 43));
            emit(0x5c10000000070000ULL | d_low | (u64{250} << 8) | (u64{255} << 20));
            break;
        }
        case 0x827: { // IMAD.HI.U32, captured zero addend
            if (insn.words[2] != 0x078e00ff) return fail(index, "native high multiply mode");
            const u32 a = insn.Bits(24, 8), b = insn.words[1];
            // Base-2^16 long multiplication. Each intermediate sum fits U32.
            xmad(250, a, b, 255, false, false, 0, true);
            xmad(251, a, b, 250, true, false, 2, true);
            xmad(250, a, b, 251, false, true, 1, true);
            xmad(252, a, b, 251, true, true, 2, true);
            emit(0x3828000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{16} << 20));
            emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{252} << 20));
            break;
        }
        case 0xa24: { // IMAD, constant-buffer multiplier
            if ((insn.words[2] & ~0x2ffU) == 0x078e0800 && valid_cbuf() && !insn.Bits(62, 2)) {
                // Bit 75 negates the addend (as 0x824): a * c[][] - Rc, with -Rc in R250.
                emit(0x4c98000000000000ULL | predicate | 252 | (u64{15} << 39) | cbuf());
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{255} << 8) |
                     (u64{insn.Bits(64, 8)} << 20) | (u64{1} << 48));
                multiply_low(insn.Bits(24, 8), 252, 250, false);
                break;
            }
            if ((insn.words[2] & ~0x2ffU) != 0x078e0000 || !valid_cbuf() || insn.Bits(62, 2))
                return fail(index, "native integer constant-buffer multiply mode");
            emit(0x4c98000000000000ULL | predicate | 252 | (u64{15} << 39) | cbuf());
            multiply_low(insn.Bits(24, 8), 252, insn.Bits(64, 8), false);
            break;
        }
        case 0x625: { // IMAD.WIDE.U32 Rd:Rd+1, Ra, Rb, c[][]:c[][+4] (form 3: Rb in bits 64-71)
            const u32 offset = insn.Bits(38, 16);
            if ((insn.words[2] & ~0xffU) != 0x078e0000 || !valid_cbuf() || insn.Bits(62, 2) ||
                (offset % 8) || offset + 4 > 0xfffc || (dest & 1) ||
                dest + 1 >= (uses_uniform_bank ? 216U : 246U))
                return fail(index, "native wide constant multiply mode");
            const u32 a = insn.Bits(24, 8), b = insn.Bits(64, 8);
            emit(0x4c98000000000000ULL | predicate | 248 | (u64{15} << 39) | cbuf());
            emit(0x4c98000000000000ULL | predicate | 249 | (u64{15} << 39) |
                 (u64{(offset + 4) / 4} << 20) | (u64{bank()} << 34));
            // High word of a * b in R253 (base-2^16 long multiplication, as IMAD.HI), low in R250.
            xmad(250, a, b, 255, false, false, 0, false);
            xmad(251, a, b, 250, true, false, 2, false);
            xmad(250, a, b, 251, false, true, 1, false);
            xmad(252, a, b, 251, true, true, 2, false);
            emit(0x3828000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{16} << 20));
            emit(0x5c10000000000000ULL | predicate | 253 | (u64{250} << 8) | (u64{252} << 20));
            xmad(250, a, b, 255, false, false, 0, false);
            xmad(251, a, b, 255, true, false, 0, false);
            xmad(251, a, b, 251, false, true, 0, false);
            emit(0x5c18000000000000ULL | predicate | 250 | (u64{251} << 8) | (u64{250} << 20) | (u64{16} << 39));
            // 64-bit add: low = R250 + c_low, carry mask (-1) in R251, high = R253 + c_high - mask.
            emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{248} << 20));
            emit(0x5b50000000000000ULL | predicate | 251 | (u64{250} << 8) | (u64{248} << 20) |
                 (u64{7} << 39) | (u64{1} << 49));
            emit(0x5c10000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{249} << 20));
            emit(0x5c10000000000000ULL | predicate | (dest + 1) | (u64{253} << 8) | (u64{251} << 20) |
                 (u64{1} << 48));
            emit(base(0x5c98000000000000ULL) | (u64{250} << 20) | (u64{15} << 39));
            break;
        }
        case 0x825: { // IMAD.WIDE.U32 Rd:Rd+1, Ra, imm32, Rc:Rc+1 (form 4: imm32 in bits 32-63)
            // NAK OpIMad64 (sm70_encode.rs): bit 73 signed, bits 81-83 carry-out. Only the unsigned
            // form without a carry-out and without modifiers (72-75), with carry-in !PT (87-90), no
            // uniform register (91) and even register pairs is accepted; Rc = RZ adds a zero pair.
            const u32 a = insn.Bits(24, 8), b = insn.words[1], c = insn.Bits(64, 8);
            if ((insn.words[2] & ~0xffU) != 0x078e0000 || (dest & 1) || (c != 255 && (c & 1)))
                return fail(index, "native wide immediate multiply mode");
            // The link check above reads only Rd; this also writes Rd+1.
            if (active_link && dest + 1 == *active_link) return fail(index, "native callee overwrites link register");
            if (b > 1 && !(b & (b - 1))) {
                // imm = 2^s (s = 1..31): low word a << s in R250, high word a >> (32 - s) in R253.
                const u32 shift = static_cast<u32>(std::countr_zero(b));
                emit(0x3848000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{shift} << 20));
                emit(0x3828000000000000ULL | predicate | 253 | (u64{a} << 8) | (u64{32 - shift} << 20));
            } else {
                // High word in R253 (base-2^16 long multiplication, as IMAD.HI), low word in R250;
                // XMAD's 16-bit immediate takes the selected half of imm (as 0x8a5).
                xmad(250, a, b, 255, false, false, 0, true);
                xmad(251, a, b, 250, true, false, 2, true);
                xmad(250, a, b, 251, false, true, 1, true);
                xmad(252, a, b, 251, true, true, 2, true);
                emit(0x3828000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{16} << 20));
                emit(0x5c10000000000000ULL | predicate | 253 | (u64{250} << 8) | (u64{252} << 20));
                xmad(250, a, b, 255, false, false, 0, true);
                xmad(251, a, b, 255, true, false, 0, true);
                xmad(251, a, b, 251, false, true, 0, true);
                emit(0x5c18000000000000ULL | predicate | 250 | (u64{251} << 8) | (u64{250} << 20) | (u64{16} << 39));
            }
            // 64-bit add (IADD.CC, IADD.X as 0x8a5). Ra and Rc:Rc+1 are read before Rd+1 and then Rd
            // are written, so Rd:Rd+1 may alias Ra or Rc:Rc+1 (OoT's model shaders have Rd+1 = Ra).
            const u32 c_high = c == 255 ? 255U : c + 1;
            emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{c} << 20) | (u64{1} << 47));
            emit(0x5c10000000000000ULL | predicate | (dest + 1) | (u64{253} << 8) | (u64{c_high} << 20) | (u64{1} << 43));
            emit(base(0x5c98000000000000ULL) | (u64{250} << 20) | (u64{15} << 39));
            break;
        }
        case 0x424: case 0x624: { // IMAD.MOV.U32: zero product plus immediate/cbuf
            if (opcode == 0x624 && insn.Bits(24, 8) == 255 &&
                (insn.words[2] & ~0x07800000U) == 0x000e06ffU) {
                const u32 source = operand(2, false);
                mov_imm(250, 1);
                emit(0x5ca0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                    (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                emit(base(0x5c10000000000000ULL) | (u64{source} << 8) | (u64{250} << 20));
                break;
            }
            if (insn.Bits(24, 8) != 255 || insn.Bits(64, 8) != 255 || insn.words[2] != 0x078e00ff) {
                if (opcode == 0x624 && insn.Bits(24, 8) == 255 && (insn.words[2] & ~0x200U) == 0x078e00ffU &&
                    insn.Bits(62, 2) == 2 && valid_cbuf()) {
                    // IMAD.MOV Rd, RZ, RZ, -c[][] (bit 63 negates the constant addend).
                    emit(0x4c98000000000000ULL | predicate | 250 | (u64{15} << 39) | cbuf());
                    emit(base(0x5c10000000000000ULL) | (u64{255} << 8) | (u64{250} << 20) | (u64{1} << 48));
                    break;
                }
                if ((insn.words[2] & ~0x2ffU) != 0x078e0000 || (opcode == 0x624 && insn.Bits(62, 2)))
                    return fail(index, "native multiply wide addend mode");
                const u32 a = operand(0, false), b = operand(1, false), c = operand(2, false);
                multiply_low(a, b, c, false);
                break;
            }
            if (insn.Bits(24, 8) != 255 || insn.Bits(64, 8) != 255 ||
                insn.words[2] != 0x078e00ff)
                return fail(index, "native integer move mode");
            if (opcode == 0x624) {
                if (!valid_cbuf() || insn.Bits(62, 2))
                    return fail(index, "native integer move constant buffer");
                emit(base(0x4c98000000000000ULL) | (u64{15} << 39) | cbuf());
            } else {
                emit(base(0x0100000000000000ULL) | (u64{15} << 12) |
                     (u64{insn.words[1]} << 20));
            }
            break;
        }
        case 0x824: { // Captured low-word IMAD forms; bit 75 negates the addend.
            const bool negate_c = insn.Bits(75, 1) != 0;
            if (insn.words[1] == 1 && (insn.words[2] & 0xfffff700U) == 0x078e0200) {
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(64, 8)} << 20) | (u64{negate_c} << 48));
                break;
            }
            const u32 multiplier = insn.words[1];
            if (insn.Bits(64, 8) == 255 && !negate_c && multiplier > 1 && !(multiplier & (multiplier - 1)) &&
                (insn.words[2] & ~0x2ffU) == 0x078e0000) {
                // IMAD.SHL by a power of two: one shift (also the form Eden's jump-table tracking
                // expects between the clamp and the table load).
                u32 shift = 0;
                while ((1U << shift) != multiplier) ++shift;
                emit(base(0x3848000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{shift} << 20));
                break;
            }
            if (insn.words[1] != 48 || insn.words[2] != 0x078e02ff) {
                if ((insn.words[2] & ~0xaffU) != 0x078e0000)
                    return fail(index, "native integer multiply mode");
                u32 c = insn.Bits(64, 8);
                if (negate_c && c != 255) {
                    emit(0x5c10000000000000ULL | predicate | 252 | (u64{255} << 8) | (u64{c} << 20) |
                         (u64{1} << 48));
                    c = 252;
                }
                multiply_low(insn.Bits(24, 8), insn.words[1], c, true);
                break;
            }
            const u64 source = insn.Bits(24, 8);
            emit(0x5c18000000000000ULL | predicate | 250 | (source << 8) | (source << 20) | (u64{1} << 39));
            emit(base(0x3848000000000000ULL) | (u64{250} << 8) | (u64{4} << 20));
            break;
        }
        case 0x806: { // VOTE: register ballot, ordinary predicate input and output.
            if (insn.words[1] || insn.Bits(72, 2) > 2 || (insn.words[2] & ~0x078e0300U))
                return fail(index, "native vote mode");
            emit(base(0x50d8000000000000ULL) | (u64{insn.Bits(87, 4)} << 39) |
                 (u64{insn.Bits(81, 3)} << 45) | (u64{insn.Bits(72, 2)} << 48));
            break;
        }
        case 0x886: { // VOTEU ballot and independent uniform predicate output.
            const u32 output = insn.Bits(81, 3), target = insn.Bits(16, 8);
            if (uniform_register(target) == ~0U || insn.words[1] ||
                insn.Bits(72, 2) > 2 || native_predicate != 7)
                return fail(index, "native uniform vote mode");
            const u32 temporary = output == 7 ? 7 : 6 - output;
            const u32 mask = 1U << temporary;
            if (output != 7) emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
            emit(0x50d8000000070000ULL | uniform_register(target) |
                (u64{insn.Bits(87, 4)} << 39) | (u64{temporary} << 45) | (u64{insn.Bits(72, 2)} << 48));
            if (output != 7) {
                emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
                emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
            }
            break;
        }
        case 0x3c2: { // R2UR: broadcast the first lane satisfying the execution predicate.
            const u32 target = uniform_register(static_cast<u32>(dest));
            if (target == ~0U || target == 255 || insn.words[1] || insn.words[2] != 0x000e0000)
                return fail(index, "native register-to-uniform mode");
            save_predicate();
            emit(0x50d8000000070000ULL | 250 | (u64{native_predicate} << 39) |
                 (u64{6} << 45) | (u64{1} << 48));
            emit(0x5c10000000070000ULL | 251 | (u64{250} << 8) | (u64{255} << 20) | (u64{1} << 49));
            emit(0x5c40000000070000ULL | 250 | (u64{250} << 8) | (u64{251} << 20) | (u64{7} << 48));
            emit(0x5c30000000070000ULL | 250 | (u64{250} << 20));
            emit(0xef10000000070000ULL | 251 | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{250} << 20) | (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            emit(0x5ca0000000070000ULL | target | (u64{251} << 8) |
                 (u64{target} << 20) | (u64{6} << 39));
            restore_predicate();
            break;
        }
        case 0x81c: { // PLOP3.LUT; bit 67 makes the third source a uniform predicate.
            const u32 target = insn.Bits(81, 3);
            const u32 lut = insn.Bits(64, 3) | (insn.Bits(72, 5) << 3);
            if (target == 7 || insn.Bits(84, 3) != 7 ||
                insn.Bits(16, 8) || insn.words[1])
                return fail(index, "native predicate LUT mode");
            if (!insn.Bits(67, 1) && (lut == 0xfe || lut == 0xa8)) {
                emit(0x5090000000000000ULL | predicate | (u64{target} << 3) | 7 |
                     (u64{insn.Bits(87, 4)} << 12) | (u64{1} << 24) |
                     (u64{insn.Bits(77, 4)} << 29) | (u64{insn.Bits(68, 4)} << 39) |
                     (u64{lut == 0xfe} << 45));
                break;
            }
            if (!insn.Bits(67, 1) && (lut == 0x80 ||
                (lut == 8 && insn.Bits(68, 4) == 7 && insn.Bits(77, 4) == 7))) {
                emit(0x5090000000000000ULL | predicate | (u64{target} << 3) | 7 |
                     (u64{insn.Bits(87, 4) ^ (lut == 8 ? 8U : 0U)} << 12) | (u64{insn.Bits(77, 4)} << 29) |
                     (u64{insn.Bits(68, 4)} << 39));
                break;
            }
            predicate_lut(lut, insn.Bits(87, 4), insn.Bits(77, 4), insn.Bits(68, 4), target,
                          insn.Bits(67, 1) != 0);
            break;
        }
        case 0x89c: { // UPLOP3.LUT: load the uniform predicates into P0-P6, combine, merge back.
            const u32 target = insn.Bits(81, 3);
            const u32 lut = insn.Bits(64, 3) | (insn.Bits(72, 5) << 3);
            if (insn.Bits(84, 3) != 7 || insn.Bits(67, 1) || insn.Bits(16, 8) || insn.words[1])
                return fail(index, "native uniform predicate LUT mode");
            if (target == 7) break;
            const auto slot = [](u32 operand) { return (operand & 7) == 7 ? operand : (operand & 8) | (6 - (operand & 7)); };
            predicate = u64{slot(native_predicate)} << 16; // a UPn guard is loaded as well
            emit(0x38e8000000070000ULL | 254 | (u64{255} << 8) | (u64{0x7f} << 20));
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{0x7f} << 20));
            predicate_lut(lut, slot(insn.Bits(87, 4)), slot(insn.Bits(77, 4)), slot(insn.Bits(68, 4)), 6 - target);
            emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{1U << (6 - target)} << 20));
            emit(0x38f0000000070000ULL | (u64{254} << 8) | (u64{0x7f} << 20));
            break;
        }
        case 0x883: { // UP2UR UR, UPR, URZ, mask: bit k of the result is UPk (R239 bit 6-k).
            const u32 target = uniform_register(insn.Bits(16, 8)), mask = insn.words[1];
            if (target == ~0U || insn.Bits(24, 8) != 63 || native_predicate != 7 ||
                insn.words[2] != 0x08000000 || mask > 0x7f)
                return fail(index, "native uniform predicate move mode");
            if (target == 255) break;
            mov_imm(target, 0);
            for (u32 k = 0; k < 7; ++k) {
                if (!((mask >> k) & 1)) continue;
                emit(0x3800000000070000ULL | 250 | (u64{239} << 8) | (u64{(6 - k) | (1U << 8)} << 20));
                emit(0x36f0000000070000ULL | target | (u64{250} << 8) | (u64{k | (1U << 8)} << 20) |
                     (u64{target} << 39));
            }
            break;
        }
        case 0x882: { // UMOV immediate.
            const u32 target = uniform_register(static_cast<u32>(dest));
            if (target == ~0U || insn.words[2]) return fail(index, "native uniform immediate move mode");
            if (target != 255) mov_imm(target, insn.words[1]); // URZ: discarded
            break;
        }
        case 0xc82: { // UMOV UR, UR (or URZ), including uniform execution predicates.
            const u32 source = insn.words[1] < 64 ? uniform_register(insn.words[1]) : ~0U;
            const u32 target = uniform_register(static_cast<u32>(dest));
            if (target == ~0U || source == ~0U || insn.words[2] != 0x08000000)
                return fail(index, "native uniform move mode");
            if (target == 255) break; // URZ: discarded
            emit(0x5c98000000000000ULL | predicate | target |
                 (u64{source} << 20) | (u64{15} << 39));
            break;
        }
        case 0xab9: case 0xabb: { // ULDC U32/U64, direct or indexed.
            const u32 uniform_dest = insn.Bits(16, 8), index_reg = insn.Bits(24, 8);
            const u32 width = insn.Bits(73, 3);
            const u32 target = uniform_register(uniform_dest);
            const u32 index_target = opcode == 0xabb ? uniform_register(index_reg) : 255U;
            // Direct ULDC.U8/S8/U16/S16 (widths 0-3, numbered as Maxwell LDC's sizes) zero- or
            // sign-extends a naturally aligned sub-word, as LDC does; Maxwell's offset is signed.
            const bool narrow = opcode == 0xab9 && width < 4 && insn.Bits(38, 16) < 0x8000 &&
                                insn.Bits(38, 16) % (width < 2 ? 1U : 2U) == 0;
            if (target == ~0U || target == 255 || ((width != 4 && width != 5) && !narrow) ||
                (width == 5 && ((uniform_dest & 1) || uniform_register(uniform_dest + 1) != target + 1)) ||
                bank() >= 18 || (!narrow && insn.Bits(38, 16) % 4) || insn.Bits(62, 2) ||
                (insn.words[2] & ~0xe00U) != (opcode == 0xab9 ? 0U : 0x08000000U) ||
                index_target == ~0U)
                return fail(index, "native uniform constant load mode");
            if (opcode == 0xab9 && width == 4) {
                emit(0x4c98000000000000ULL | predicate | target | (u64{15} << 39) | cbuf());
            } else {
                emit(0xef90000000000000ULL | predicate | target |
                     (u64{index_target} << 8) | (u64{insn.Bits(38, 16)} << 20) |
                     (u64{bank()} << 36) | (u64{width} << 48));
            }
            break;
        }
        case 0x899: { // USHF.L.U32; accept a zero word outside the selected half.
            // A UPn guard is swapped into P(6-n) like UMOV's (uniform_predication above).
            // URd, URa (low word) and URb (high word) in R216+ (uniform_register; URZ is RZ).
            const u32 dst = uniform_register(insn.Bits(16, 8)), low = uniform_register(insn.Bits(24, 8)),
                      high = uniform_register(insn.Bits(64, 8));
            const bool mapped = dst != ~0U && low != ~0U && high != ~0U;
            if (mapped && insn.words[2] == (0x08011400U | insn.Bits(64, 8)) && low == 255 &&
                high != 255 && insn.words[1] <= 32) {
                emit(0x3828000000000000ULL | predicate | dst |
                     (u64{high} << 8) | (u64{insn.words[1]} << 20) | (u64{1} << 48));
                break;
            }
            // Logical right shifts of one register (bit 76 right, bit 80 high word, bits 73-74
            // U32): USHF.R.U32 URd, URa, n, URZ and USHF.R.U32.HI URd, URZ, n, URb.
            const bool right_low = insn.words[2] == 0x0800163f && low != 255;
            const bool right_high = insn.words[2] == (0x08011600U | insn.Bits(64, 8)) && low == 255 && high != 255;
            if (mapped && (right_low || right_high) && insn.words[1] <= 32) {
                emit(0x3828000000000000ULL | predicate | dst |
                     (u64{right_high ? high : low} << 8) | (u64{insn.words[1]} << 20));
                break;
            }
            const bool high_result = insn.words[2] == (0x08010600U | insn.Bits(64, 8)) && low == 255 && high != 255;
            const bool low_result = insn.words[2] == 0x0800063f && low != 255;
            if (mapped && !high_result && !low_result && insn.words[1] <= 32 &&
                (insn.words[2] == (0x08010600U | insn.Bits(64, 8)) ||
                 insn.words[2] == (0x08010200U | insn.Bits(64, 8)))) {
                // USHF.L.U32.HI / USHF.L.U64.HI URd, URa, n, URb (n <= 32, no wrap): the high word
                // of (URb:URa) << n, as Maxwell SHF.L (0x819's left funnel shift).
                emit(0x36f8000000000000ULL | predicate | dst | (u64{low} << 8) |
                     (u64{high} << 39) | (u64{insn.words[1]} << 20));
                break;
            }
            if (mapped && (insn.words[2] & ~0xffU) == 0x08001200U && insn.words[1] <= 32) {
                // USHF.R.U64 URd, URa, n, URc (bits 73-74 = 1 U64, bit 75 no wrap, bit 76 right,
                // bit 80 low word): the low word of (URc:URa) >> n, as Maxwell SHF.R (0x38f8).
                // For n <= 32 the U64 clamp (64) and the U32 clamp (32) give the same word, so
                // this is the U32 form: Eden's SHF reads its immediate from bits 20-38, which
                // overlap the clamp field (37-38), so a U64 clamp would change the shift.
                emit(0x38f8000000000000ULL | predicate | dst | (u64{low} << 8) |
                     (u64{high} << 39) | (u64{insn.words[1]} << 20));
                break;
            }
            if (!mapped || insn.words[1] > 32 || (!high_result && !low_result))
                return fail(index, "native uniform shift mode");
            emit(0x3848000000000000ULL | predicate | dst |
                 (u64{high_result ? high : low} << 8) | (u64{insn.words[1]} << 20));
            break;
        }
        case 0xb82: { // LDC.64 indexed byte offset
            // Bits 78-79 are the address mode, numbered as Maxwell LDC's bits 44-45; mode 2 (.IS)
            // takes the bank from the high half of Ra + offset (Hyrule Warriors: Age of
            // Imprisonment). IL and ISL have no Maxwell translation.
            const u32 width = insn.Bits(73, 3), mode = insn.Bits(78, 2);
            if ((width != 4 && width != 5) || (mode != 0 && mode != 2) || insn.Bits(91, 1) ||
                bank() >= 18 ||
                (width == 5 && ((dest & 1) || dest >= 254))) return fail(index, "native indexed constant load mode");
            emit(base(0xef90000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(38, 16)} << 20) | (u64{bank()} << 36) | (u64{mode} << 44) |
                 (u64{width} << 48));
            break;
        }
        case 0xf89: { // SHFL with immediate lane and segmentation/clamp mask, any mode.
            const u32 lane = insn.Bits(53, 5), mask = insn.Bits(40, 13), mode = insn.Bits(58, 2);
            if (insn.words[2] != 0x000e0000 || insn.Bits(32, 8) || insn.Bits(60, 4))
                return fail(index, "native immediate shuffle mode");
            emit(base(0xef10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{lane} << 20) | (u64{1} << 28) | (u64{1} << 29) |
                 (u64{mode} << 30) | (u64{mask} << 34) | (u64{7} << 48));
            break;
        }
        case 0x822: { // FSWZADD: per lane 0 = a+b, 1 = a-b, 2 = b-a, 3 = b (nvdisasm PP/PN/NP/ZP).
            // Bit 77 is .NDV, the compiler's promise that the quad is converged. Eden's FSWZADD is
            // lane-local (laneid & 3 picks the operation) and ignores Maxwell's NDV (bit 38), so
            // it is accepted without changing the result.
            const u32 swizzle = insn.words[1];
            if (swizzle > 0xff || (insn.words[2] & ~0x20ffU))
                return fail(index, "native swizzled add mode");
            // Maxwell's lane codes 1 and 2 are the other way round (1 = b-a, 2 = a-b).
            const u32 maxwell = ((swizzle & 0x55) << 1) | ((swizzle & 0xaa) >> 1);
            emit(base(0x50f8000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(64, 8)} << 20) | (u64{maxwell} << 28));
            break;
        }
        case 0xb6c: { // TXD 2D or ARRAY_2D, four channels; the two destination pairs anywhere.
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 coordinates = insn.Bits(24, 8), derivatives = insn.Bits(32, 8);
            // Ampere dimensions: 1 = 2D, 5 = ARRAY_2D; Maxwell TXD's are 2 and 3. The coordinates
            // (x, y[, layer]) and the four gradients sit in the same registers for both.
            const u32 dimension = insn.Bits(61, 3);
            const u32 high = insn.Bits(64, 8); // destination of the third and fourth channels
            // Bits 72-75: channel mask (written packed, the first two to the destination pair);
            // bit 90: .NODEP.
            const u32 mask = insn.Bits(72, 4);
            const u32 components = static_cast<u32>(std::popcount(mask));
            if ((insn.words[2] & ~0x04000fffU) != 0x001e0000 || mask == 0 || insn.Bits(59, 2) ||
                (dimension != 1 && dimension != 5) || binding >= 18 || offset > 0x1fff ||
                (components > 1 && (dest & 1)) || dest >= 248 ||
                (components > 3 && (high & 1)) || (components > 2 && high >= 248 && high != 255) ||
                coordinates >= 248 || derivatives >= 247)
                return fail(index, "native gradient texture mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native texture descriptor buffer conflict");
            result.required_texture_buffer = binding;
            // Maxwell writes the channels to consecutive registers: when a second pair is
            // elsewhere (Splatoon Raiders: R12/R18, R18/R44), write them to scratch and move each
            // pair into place.
            const bool consecutive = components <= 2 || high == dest + 2;
            const u64 target = consecutive ? dest : 248;
            emit(0xde00000000000000ULL | predicate | target | (u64{coordinates} << 8) |
                 (u64{derivatives} << 20) | (u64{dimension == 5 ? 3U : 2U} << 28) | (u64{mask} << 31) |
                 (u64{offset} << 36) | (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51));
            if (!consecutive) {
                for (u32 i = 0; i < 2; ++i) {
                    emit(0x5c98000000000000ULL | predicate | (dest + i) | (u64{15} << 39) | (u64{248 + i} << 20));
                    if (high != 255 && i < components - 2)
                        emit(0x5c98000000000000ULL | predicate | (high + i) | (u64{15} << 39) | (u64{250 + i} << 20));
                }
            }
            break;
        }
        case 0x989: { // SHFL, immediate lane and register clamp/segmentation mask, any mode.
            if ((insn.words[1] & 0xf01fffffU) || (insn.words[2] & ~0x000e00ffU))
                return fail(index, "native register-mask shuffle mode");
            emit(base(0xef10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(53, 5)} << 20) | (u64{1} << 28) | (u64{insn.Bits(58, 2)} << 30) |
                 (u64{insn.Bits(64, 8)} << 39) | (u64{insn.Bits(81, 3)} << 48));
            break;
        }
        case 0x589: { // SHFL.IDX, register lane and immediate segmentation mask.
            if ((insn.words[2] & ~0x000e0000U) || insn.Bits(53, 11))
                return fail(index, "native shuffle mode");
            emit(0xef10000000000000ULL | predicate | dest |
                (u64{insn.Bits(24, 8)} << 8) | (u64{insn.Bits(32, 8)} << 20) |
                (u64{1} << 29) | (u64{insn.Bits(40, 13)} << 34) | (u64{insn.Bits(81, 3)} << 48));
            break;
        }
        case 0x98e: { // RED.E.{ADD..XOR}.STRONG.GPU U32 (op in bits 87-89), register address.
            if ((insn.words[2] & ~0x0380003fU) == 0x0c10e100 && insn.Bits(24, 8) == 255) {
                // [URa + offset]: a uniform 64-bit address with a signed byte offset.
                const u32 address = uniform_register(insn.Bits(64, 6));
                const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
                if (address == ~0U || (address & 1) || offset < -0x80000 || offset > 0x7ffff)
                    return fail(index, "native global reduction uniform address");
                emit(0xebf8000000000000ULL | predicate | insn.Bits(32, 8) | (u64{address} << 8) |
                     (u64{insn.Bits(87, 3)} << 23) | (u64{static_cast<u32>(offset) & 0xfffff} << 28) |
                     (u64{1} << 48));
                break;
            }
            if ((insn.words[2] & ~0x03800000U) != 0x0010e100 || insn.Bits(40, 24) || (insn.Bits(24, 8) & 1))
                return fail(index, "native global reduction mode");
            emit(0xebf8000000000000ULL | predicate | insn.Bits(32, 8) |
                (u64{insn.Bits(24, 8)} << 8) | (u64{insn.Bits(87, 3)} << 23) | (u64{1} << 48));
            break;
        }
        case 0x2bd: // UFLO.U32, the highest set-bit index (.SH, bit 74: its shift amount).
            if (dest >= 16 || insn.words[1] >= 16 || (insn.words[2] & ~0x400U) != 0x080e0000 || native_predicate != 7)
                return fail(index, "native uniform leading bit mode");
            emit(0x5c30000000000000ULL | predicate | (dest + 216) | (u64{insn.words[1] + 216} << 20) |
                 (u64{insn.Bits(74, 1)} << 41));
            break;
        case 0x2bf: // UPOPC, the set-bit count of a uniform register (Hyrule Warriors: AoI compute).
            if (dest >= 16 || insn.words[1] >= 16 || insn.words[2] != 0x08000000 || native_predicate != 7)
                return fail(index, "native uniform population count mode");
            emit(0x5c08000000000000ULL | predicate | (dest + 216) | (u64{insn.words[1] + 216} << 20));
            break;
        case 0x2be: // UBREV, as a bit-reversing zero shift like BREV.
            if (dest >= 16 || insn.words[1] >= 16 || insn.words[2] != 0x08000000 || native_predicate != 7)
                return fail(index, "native uniform bit reverse mode");
            emit(0x3828000000000000ULL | predicate | (dest + 216) | (u64{insn.words[1] + 216} << 8) |
                 (u64{1} << 39) | (u64{1} << 40));
            break;
        case 0x3a1: { // MATCH.ALL.U64 with only the all-equal predicate retained.
            const u32 source = insn.Bits(24, 8), target = insn.Bits(81, 3);
            if (dest == 255 && target < 6 && native_predicate == 7 && !insn.words[1] &&
                !(insn.words[2] & ~0xe0000U)) {
                // MATCH.ALL.U32 PT/Pn, RZ, Rs (bit 73 clear): Pn = every active lane holds the
                // value of the first active one, compared as one word like the U64 form below.
                save_predicate();
                emit(0x50d8000000070000ULL | 250 | (u64{7} << 39) | (u64{7} << 45) | (u64{1} << 48));
                emit(0x5c10000000070000ULL | 251 | (u64{250} << 8) | (u64{255} << 20) | (u64{1} << 49));
                emit(0x5c40000000070000ULL | 250 | (u64{250} << 8) | (u64{251} << 20) | (u64{7} << 48));
                emit(0x5c30000000070000ULL | 250 | (u64{250} << 20));
                emit(0xef10000000070000ULL | 251 | (u64{source} << 8) |
                    (u64{250} << 20) | (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
                emit(0x5b60000000070000ULL | 7 | (u64{6} << 3) | (u64{source} << 8) |
                    (u64{251} << 20) | (u64{7} << 39) | (u64{2} << 49));
                emit(0x50d8000000070000ULL | 255 | (u64{6} << 39) | (u64{target} << 45));
                restore_predicate();
                break;
            }
            if (dest != 255 || source >= 254 || (source & 1) || target >= 6 ||
                native_predicate != 7 || insn.words[1] || (insn.words[2] & ~0xe0000U) != 0x200)
                return fail(index, "native warp match mode");
            save_predicate();
            emit(0x50d8000000070000ULL | 250 | (u64{7} << 39) | (u64{7} << 45) | (u64{1} << 48));
            emit(0x5c10000000070000ULL | 251 | (u64{250} << 8) | (u64{255} << 20) | (u64{1} << 49));
            emit(0x5c40000000070000ULL | 250 | (u64{250} << 8) | (u64{251} << 20) | (u64{7} << 48));
            emit(0x5c30000000070000ULL | 250 | (u64{250} << 20));
            for (u32 word = 0; word < 2; ++word)
                emit(0xef10000000070000ULL | (251 + word) | (u64{source + word} << 8) |
                    (u64{250} << 20) | (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            emit(0x5b60000000070000ULL | 7 | (u64{6} << 3) | (u64{source} << 8) |
                (u64{251} << 20) | (u64{7} << 39) | (u64{2} << 49));
            emit(0x5b60000000070000ULL | 7 | (u64{6} << 3) | (u64{source + 1} << 8) |
                (u64{252} << 20) | (u64{6} << 39) | (u64{2} << 49));
            emit(0x50d8000000070000ULL | 255 | (u64{6} << 39) | (u64{target} << 45));
            restore_predicate();
            break;
        }
        case 0x3c4: { // REDUX.SUM (modulo 2^32), .MAX or .MIN (unsigned) over the actual active lanes.
            const u32 source = insn.Bits(24, 8), target = uniform_register(static_cast<u32>(dest));
            const bool maximum = insn.words[2] == 0x14000;
            const bool minimum = insn.words[2] == 0x10000; // REDUX.MIN.U32: op 4 in bits 78-80
            if (target == ~0U || target == 255 || insn.words[1] ||
                (insn.words[2] != 0xc000 && !maximum && !minimum) || native_predicate != 7)
                return fail(index, "native warp sum mode");
            emit(0x50d8000000070000ULL | 250 | (u64{7} << 39) | (u64{7} << 45) | (u64{1} << 48));
            mov_imm(target, minimum ? ~0U : 0U);
            for (u32 lane = 0; lane < 32; ++lane) {
                emit(0xef10000000070000ULL | 251 | (u64{source} << 8) | (u64{lane} << 20) |
                    (u64{1} << 28) | (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
                emit(0x3828000000070000ULL | 252 | (u64{250} << 8) | (u64{lane} << 20));
                emit(0x0400000000070000ULL | 252 | (u64{252} << 8) | (u64{1} << 20));
                emit(0x5c10000000070000ULL | 252 | (u64{255} << 8) | (u64{252} << 20) | (u64{1} << 48));
                if (minimum) {
                    // An inactive lane contributes ~0 (R251 | ~mask), the identity of unsigned min;
                    // IMNMX with PT picks the minimum.
                    emit(0x5c40000000070000ULL | 251 | (u64{251} << 8) | (u64{252} << 20) |
                         (u64{1} << 40) | (u64{1} << 41) | (u64{7} << 48));
                    emit(0x5c20000000070000ULL | target | (u64{target} << 8) | (u64{251} << 20) | (u64{7} << 39));
                    continue;
                }
                emit(0x5c40000000070000ULL | 251 | (u64{251} << 8) | (u64{252} << 20) | (u64{7} << 48));
                if (maximum) {
                    emit(0x5c20000000070000ULL | target | (u64{target} << 8) | (u64{251} << 20) | (u64{15} << 39));
                } else {
                    emit(0x5c10000000070000ULL | target | (u64{target} << 8) | (u64{251} << 20));
                }
            }
            break;
        }
        case 0xf8c: { // ATOMS.POPC.INC.32 with no returned value: one increment per active lane.
            const u32 stride = insn.Bits(78, 2);
            if (dest != 255 || insn.words[1] || (insn.words[2] & ~0xc000U) != 0x0d00003f)
                return fail(index, "native shared increment mode");
            u32 address = insn.Bits(24, 8);
            if (stride && address != 255) {
                emit(0x3848000000000000ULL | predicate | 251 | (u64{address} << 8) |
                    (u64{stride + 1} << 20));
                address = 251;
            }
            mov_imm(250, 1);
            emit(0xec00000000000000ULL | predicate | 255 | (u64{address} << 8) | (u64{250} << 20));
            break;
        }
        case 0x38c: { // ATOMS.ADD.U32, naturally aligned shared byte offset.
            const u32 offset = insn.Bits(40, 24);
            if (insn.words[2] && !(offset & 3) && (insn.words[2] & ~0x07800200U) == 0 &&
                (insn.Bits(87, 4) == 1 || insn.Bits(87, 4) == 2)) {
                // ATOMS.MIN / .MAX (bits 87-90 = 1 / 2), U32 or S32 (bit 73): Maxwell ATOMS numbers
                // the operation (bits 52-55) and the U32/S32 size (bits 28-29) the same way.
                emit(0xec00000000000000ULL | predicate | dest |
                    (u64{insn.Bits(24, 8)} << 8) | (u64{insn.Bits(32, 8)} << 20) |
                    (u64{insn.Bits(73, 1)} << 28) | (u64{offset >> 2} << 30) | (u64{insn.Bits(87, 4)} << 52));
                break;
            }
            if (insn.words[2] || (offset & 3)) return fail(index, "native shared atomic mode");
            emit(0xec00000000000000ULL | predicate | dest |
                (u64{insn.Bits(24, 8)} << 8) | (u64{insn.Bits(32, 8)} << 20) |
                (u64{offset >> 2} << 30));
            break;
        }
        case 0x3a8: { // ATOMG.E.ADD.STRONG.GPU U32 at a 64-bit register address plus an offset.
            const u32 address = insn.Bits(24, 8);
            const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
            if (insn.words[2] != 0x001ee100 || (address & 1) || address >= 246 ||
                offset < -0x80000 || offset > 0x7ffff)
                return fail(index, "native global atomic register mode");
            emit(0xed00000000000000ULL | predicate | dest | (u64{address} << 8) |
                (u64{insn.Bits(32, 8)} << 20) | (u64{static_cast<u32>(offset) & 0xfffff} << 28) |
                (u64{1} << 48));
            break;
        }
        case 0x9a8: { // ATOMG.E.ADD.STRONG.GPU U32 through a uniform pointer plus an offset.
            const u32 address = uniform_register(insn.Bits(64, 6));
            const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
            if (insn.Bits(24, 8) != 255 || offset < -0x80000 || offset > 0x7ffff ||
                (insn.words[2] & ~0x3fU) != 0x081ee140 || address == ~0U || (address & 1))
                return fail(index, "native global atomic mode");
            emit(0xed00000000000000ULL | predicate | dest | (u64{address} << 8) |
                (u64{insn.Bits(32, 8)} << 20) | (u64{static_cast<u32>(offset) & 0xfffff} << 28) |
                (u64{1} << 48));
            break;
        }
        case 0xb97: { // SULD.P.2D/2D_ARRAY/3D.IGN, direct descriptor, channel mask at 72-75.
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 coordinate = insn.Bits(24, 8);
            const u32 channels = insn.Bits(72, 4);
            const u32 dimension = insn.Bits(61, 3); // Maxwell's codes: 3 = 2D, 4 = 2D_ARRAY, 5 = 3D
            // 3D reads x, y, z from Ra..Ra+2 on both sides (as SUST.P.3D below).
            if ((insn.words[2] & ~0xf00U) != 0x001e0000 || channels == 0 || (dimension < 3 || dimension > 5) ||
                binding >= 18 || offset > 0x1fff || insn.Bits(59, 2) || dest == 255 ||
                coordinate > (dimension == 4 ? 252U : 253U) ||
                (dimension == 5 && coordinate + 2 >= (uses_uniform_bank ? 216U : 246U)))
                return fail(index, "native formatted surface load mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native surface descriptor buffer conflict");
            result.required_texture_buffer = binding;
            emit(base(0xeb00000000000000ULL) | (u64{coordinate} << 8) |
                (u64{channels} << 20) | (u64{dimension} << 33) | (u64{offset} << 36) | (u64{1} << 51));
            break;
        }
        case 0xb9b: { // SUST.P.2D/3D.IGN, direct descriptor and all four channels.
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 data = insn.Bits(32, 8), coordinate = insn.Bits(24, 8);
            const u32 dimension = insn.Bits(61, 3);
            if (insn.words[2] != 0x00100f00 || dimension < 3 || dimension > 5 ||
                binding >= 18 || offset > 0x1fff || insn.Bits(59, 2) || data > 252 || coordinate > (dimension >= 4 ? 252U : 253U))
                return fail(index, "native formatted surface store mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native surface descriptor buffer conflict");
            result.required_texture_buffer = binding;
            emit(0xeb20000000000000ULL | predicate | data | (u64{coordinate} << 8) |
                (u64{15} << 20) | (u64{dimension} << 33) | (u64{offset} << 36) | (u64{1} << 51));
            break;
        }
        case 0x301: // BREV register.
            if (insn.words[2] || insn.words[1] > 255) return fail(index, "native bit reverse mode");
            emit(base(0x3828000000000000ULL) | (u64{insn.words[1]} << 8) |
                (u64{1} << 39) | (u64{1} << 40));
            break;
        case 0x219: case 0xc19: { // U32 register (or uniform, 0xc19) shifts with a zero word outside the selected result.
            const u32 low = insn.Bits(24, 8), high = insn.Bits(64, 8);
            const u32 shift = opcode == 0xc19 ? uniform_register(insn.Bits(32, 8)) : insn.Bits(32, 8);
            const bool right = insn.Bits(76, 1) != 0, high_result = insn.Bits(80, 1) != 0;
            if (opcode == 0x219 && (insn.words[2] & ~0xffU) == 0x00001200U && insn.words[1] <= 255) {
                // SHF.R.U64 Rd, Ra, Rb, Rc (type 73-74 = 1, no .W, low word): the low word of
                // (Rc:Ra) >> min(Rb, 64), so 0 for Rb >= 64 (NAK OpShf fold). Maxwell SHF.R.U32
                // shifts (Rc:Ra) by min(Rb, 32); the rest, Rb - min(Rb, 32), is applied to that word
                // with a zero high word and clamps at 32 too. Every shift stays within 0..32; Eden's
                // SHF.U64 would clamp at 63 instead. Rd is written last (it may be Ra, Rb or Rc).
                emit(0x3820000000000000ULL | predicate | 250 | (u64{shift} << 8) | (u64{32} << 20) |
                     (u64{7} << 39)); // IMNMX.U32 R250 = min(Rb, 32)
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{shift} << 8) | (u64{250} << 20) |
                     (u64{1} << 48)); // IADD R250 = Rb - R250
                emit(0x5cf8000000000000ULL | predicate | 251 | (u64{low} << 8) | (u64{shift} << 20) |
                     (u64{high} << 39)); // SHF.R.U32 R251 = low word of (Rc:Ra) >> min(Rb, 32)
                emit(base(0x5cf8000000000000ULL) | (u64{251} << 8) | (u64{250} << 20) |
                     (u64{255} << 39)); // SHF.R.U32 Rd = R251 >> min(R250, 32)
                break;
            }
            // SHF.R.U32 of the low word with a zero high word: a plain logical right shift.
            if (right && !high_result && high == 255 && insn.Bits(73, 2) == 3 && shift != ~0U &&
                insn.words[1] <= 255 &&
                (insn.words[2] & ~(0x000118ffU | (opcode == 0xc19 ? 0x08000000U : 0U))) == 0x00000600U) {
                emit(base(0x5c28000000000000ULL) | (u64{low} << 8) | (u64{shift} << 20));
                break;
            }
            if (insn.words[1] > 255 || insn.Bits(73, 2) != 3 || shift == ~0U ||
                (insn.words[2] & ~(0x000118ffU | (opcode == 0xc19 ? 0x08000000U : 0U))) != 0x00000600U ||
                (right ? (!high_result || low != 255) : (high_result || high != 255)))
                return fail(index, "native register shift mode");
            emit(base(right ? 0x5cf8000000000000ULL : 0x5bf8000000000000ULL) |
                (u64{right ? high : 255U} << 8) | (u64{shift} << 20) |
                (u64{right ? 255U : low} << 39) | (u64{insn.Bits(75, 1)} << 50));
            break;
        }
        case 0x300: case 0xd00: { // FLO, including a uniform source.
            if ((insn.words[2] & ~(0x600U | (opcode == 0xd00 ? 0x08000000U : 0U))) != 0x000e0000U)
                return fail(index, "native leading bit mode");
            const u32 source = operand(1, false);
            emit(base(0x5c30000000000000ULL) | (u64{source} << 20) |
                (u64{insn.Bits(63, 1)} << 40) | (u64{insn.Bits(74, 1)} << 41) |
                (u64{insn.Bits(73, 1)} << 48));
            break;
        }
        case 0x309: case 0xd09: { // POPC, including a uniform source.
            if (insn.words[2] != (opcode == 0xd09 ? 0x08000000U : 0U))
                return fail(index, "native population count mode");
            const u32 source = operand(1, false);
            emit(base(0x5c08000000000000ULL) | (u64{source} << 20) | (u64{insn.Bits(63, 1)} << 40));
            break;
        }
        case 0x355: // BMOV.32(.CLEAR, bit 84) RZ, Bn: the lowering does not model convergence barriers
            // (BSSY / BSYNC / BREAK are no-ops below), so reading one into RZ or clearing it does nothing.
            if (dest != 255 || insn.Bits(24, 8) >= 16 || insn.words[1] || (insn.words[2] & ~0x00100000U))
                return fail(index, "native barrier register move mode");
            break;
        case 0x918:
            if (insn.words[1] || insn.words[2]) return fail(index, "native NOP mode");
            emit(0x50b0000000000f00ULL | predicate);
            break;
        case 0x992: { // MEMBAR.SC: preserve workgroup/device visibility.
            const u32 scope = insn.Bits(76, 3);
            if (insn.words[1] || insn.words[2] != (scope << 12) ||
                (scope != 0 && scope != 2 && scope != 3 && scope != 5))
                return fail(index, "native memory barrier scope");
            const u32 legacy_scope = scope == 0 ? 0 : scope == 2 ? 1 : scope == 3 ? 2 : 3;
            emit(0xef98000000000000ULL | predicate | (u64{legacy_scope} << 8));
            break;
        }
        case 0x9ab: // ERRBAR: wait for prior memory operations on the host backend.
            if (insn.words[1] || insn.words[2]) return fail(index, "native error barrier mode");
            emit(0xef98000000000000ULL | predicate | (u64{1} << 8));
            break;
        case 0xb1d: // BAR.SYNC 0, full workgroup.
            if (insn.words[1] || insn.words[2] || native_predicate != 7)
                return fail(index, "native workgroup barrier mode");
            emit(0xf0a8000000000000ULL | predicate | (u64{1} << 43) |
                (u64{1} << 44) | (u64{1} << 39) | (u64{7} << 39));
            break;
        case 0x388: case 0x984: { // Shared memory load/store with scaled register offsets.
            const u32 width = insn.Bits(73, 3), stride = insn.Bits(78, 2);
            // LDS [Ra.Xn + URb + offset]: bit 91 adds a uniform register (bits 32-39).
            const bool uniform_base = opcode == 0x984 && insn.Bits(91, 1);
            const u32 base_register = uniform_base ? uniform_register(insn.Bits(32, 8)) : 255U;
            if (width > 6 || (insn.words[2] & ~(uniform_base ? 0x08000000U : 0U)) != ((width << 9) | (stride << 14)) ||
                (opcode == 0x984 && !uniform_base && insn.Bits(32, 8)) || base_register == ~0U)
                return fail(index, "native shared memory addressing mode");
            u32 address = insn.Bits(24, 8);
            if (stride && address != 255) {
                emit(0x3848000000000000ULL | predicate | 250 |
                    (u64{address} << 8) | (u64{stride + 1} << 20));
                address = 250;
            }
            if (base_register != 255) {
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{address} << 8) | (u64{base_register} << 20));
                address = 250;
            }
            const u32 data = opcode == 0x388 ? insn.Bits(32, 8) : static_cast<u32>(dest);
            emit((opcode == 0x388 ? 0xef58000000000000ULL : 0xef48000000000000ULL) |
                predicate | data | (u64{address} << 8) | (u64{insn.Bits(40, 24)} << 20) |
                (u64{width} << 48));
            break;
        }
        case 0x988: { // STS [Ra.Xn + URb + offset], Rd: the uniform register sits in bits 64-69.
            const u32 width = insn.Bits(73, 3), stride = insn.Bits(78, 2);
            const u32 base_register = uniform_register(insn.Bits(64, 6));
            const u32 data = insn.Bits(32, 8), count = width < 5 ? 1U : 1U << (width - 4);
            // The data register (bits 32-39) is not covered by the scratch check above: keep it
            // and its pair/quad below R216 and aligned (Maxwell STS.64/.128 need that too).
            if (width > 6 || !insn.Bits(91, 1) ||
                (insn.words[2] & ~(0x08000000U | 0x3fU)) != ((width << 9) | (stride << 14)) ||
                base_register == ~0U || (data == 255 && count > 1) ||
                (data != 255 && (data % count || data + count > 216)))
                return fail(index, "native shared memory addressing mode");
            u32 address = insn.Bits(24, 8);
            if (stride && address != 255) {
                emit(0x3848000000000000ULL | predicate | 250 |
                    (u64{address} << 8) | (u64{stride + 1} << 20));
                address = 250;
            }
            if (base_register != 255) {
                if (address == 255) {
                    address = base_register;
                } else {
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{address} << 8) | (u64{base_register} << 20));
                    address = 250;
                }
            }
            emit(0xef58000000000000ULL | predicate | data | (u64{address} << 8) |
                 (u64{insn.Bits(40, 24)} << 20) | (u64{width} << 48));
            break;
        }
        case 0x23a: { // MOVM.16.MT88: transpose an 8x8 16-bit matrix held one row pair per lane.
            // Lane t holds row t/4, columns 2(t%4)..+1; the result is the transposed matrix in the
            // same layout, so each half comes from lane 8(t%4) + t/8 (+4) at half (t/4)%2.
            const u32 source = insn.Bits(24, 8);
            if (insn.words[1] || insn.words[2] || source >= 246 || dest >= 246)
                return fail(index, "native matrix move mode");
            emit(0xf0c8000000000000ULL | predicate | 246);                                        // laneid
            emit(0x0400000000000000ULL | predicate | 249 | (u64{246} << 8) | (u64{3} << 20));
            emit(0x3848000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{3} << 20));     // 8(t%4)
            emit(0x3828000000000000ULL | predicate | 247 | (u64{246} << 8) | (u64{3} << 20));     // t/8
            emit(0x5c10000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{247} << 20));
            emit(0xef10000000000000ULL | predicate | 247 | (u64{source} << 8) | (u64{249} << 20) |
                 (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            emit(0x1c00000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{4} << 20));
            emit(0xef10000000000000ULL | predicate | 248 | (u64{source} << 8) | (u64{249} << 20) |
                 (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            emit(0x3828000000000000ULL | predicate | 249 | (u64{246} << 8) | (u64{2} << 20));
            emit(0x0400000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{1} << 20));
            emit(0x3848000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{4} << 20));     // 0 or 16
            emit(0x5c28000000000000ULL | predicate | 247 | (u64{247} << 8) | (u64{249} << 20));
            emit(0x0400000000000000ULL | predicate | 247 | (u64{247} << 8) | (u64{0xffff} << 20));
            emit(0x5c28000000000000ULL | predicate | 248 | (u64{248} << 8) | (u64{249} << 20));
            emit(0x5c18000000000000ULL | predicate | dest | (u64{248} << 8) | (u64{247} << 20) | (u64{16} << 39));
            break;
        }
        case 0x23c: { // HMMA.16816.F16: D(16x8) = A(16x16) * B(16x8) + C, f16 fragments.
            // Lane t (group g = t/4, i = t%4): A regs hold rows g / g+8 at k 2i..2i+1 (a0, a1) and
            // 8+2i.. (a2, a3); B regs hold k 2i.. (b0) / 8+2i.. (b1) of column g; D holds rows
            // g / g+8 at columns 2i, 2i+1. Rebuilt from shuffles with HFMA2, accumulating in R252/R253.
            // The expansion is correct but large: title 4's 372-HMMA network takes the host driver
            // minutes to compile, so it stays opt-in (the QMD still completes when rejected).
            // EDEN_NX2_TENSOR=stub keeps the rest of the program but makes every HMMA D = C.
            static const char* const mode = std::getenv("EDEN_NX2_TENSOR");
            if (!mode) return fail(index, "native matrix multiply (set EDEN_NX2_TENSOR to emulate)");
            const u32 a = insn.Bits(24, 8), b = insn.Bits(32, 8), c = insn.Bits(64, 8);
            if ((insn.words[1] & ~0xffU) || (insn.words[2] & ~0xffU) != 0x800 || dest + 2 > 246 ||
                a + 4 > 246 || b + 2 > 246 || (c != 255 && c + 2 > 246))
                return fail(index, "native matrix multiply mode");
            if (std::string_view{mode} == "stub") {
                for (u32 row = 0; row < 2; ++row) {
                    if (c == 255) mov_imm(static_cast<u32>(dest) + row, 0);
                    else if (c != dest) emit(0x5c98000000000000ULL | predicate | (dest + row) | (u64{15} << 39) | (u64{c + row} << 20));
                }
                break;
            }
            const auto shuffle = [&](u32 target, u32 source_reg) {
                emit(0xef10000000000000ULL | predicate | target | (u64{source_reg} << 8) | (u64{246} << 20) |
                     (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            };
            const auto lane_a = [&](u32 i) { // 4g + i
                emit(0xf0c8000000000000ULL | predicate | 246);
                emit(0x0400000000000000ULL | predicate | 246 | (u64{246} << 8) | (u64{0x1c} << 20));
                if (i) emit(0x1c00000000000000ULL | predicate | 246 | (u64{246} << 8) | (u64{i} << 20));
            };
            const auto lane_b = [&](u32 add) { // 8(t%4) + add: column 2(t%4) (+1 when add >= 4)
                emit(0xf0c8000000000000ULL | predicate | 246);
                emit(0x0400000000000000ULL | predicate | 246 | (u64{246} << 8) | (u64{3} << 20));
                emit(0x3848000000000000ULL | predicate | 246 | (u64{246} << 8) | (u64{3} << 20));
                if (add) emit(0x1c00000000000000ULL | predicate | 246 | (u64{246} << 8) | (u64{add} << 20));
            };
            const auto hfma2 = [&](u32 accumulator, u32 swizzle_a, u32 pair) {
                emit(0x5d00000000000000ULL | predicate | accumulator | (u64{251} << 8) | (u64{pair} << 20) |
                     (u64{accumulator} << 39) | (u64{swizzle_a} << 47));
            };
            for (u32 row = 0; row < 2; ++row) {
                if (c == 255) mov_imm(252 + row, 0);
                else emit(0x5c98000000000000ULL | predicate | (252 + row) | (u64{15} << 39) | (u64{c + row} << 20));
            }
            for (u32 i = 0; i < 4; ++i) {
                for (u32 k_block = 0; k_block < 2; ++k_block) {
                    lane_b(i);
                    shuffle(247, b + k_block);
                    lane_b(4 + i);
                    shuffle(248, b + k_block);
                    // P0 = (B[2i][2t], B[2i][2t+1]), P1 = (B[2i+1][2t], B[2i+1][2t+1])
                    emit(0x0400000000000000ULL | predicate | 249 | (u64{247} << 8) | (u64{0xffff} << 20));
                    emit(0x5c18000000000000ULL | predicate | 249 | (u64{248} << 8) | (u64{249} << 20) | (u64{16} << 39));
                    emit(0x3828000000000000ULL | predicate | 250 | (u64{247} << 8) | (u64{16} << 20));
                    emit(0x0400000000000000ULL | predicate | 248 | (u64{248} << 8) | (u64{0xffff0000} << 20));
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{248} << 20));
                    for (u32 row = 0; row < 2; ++row) {
                        lane_a(i);
                        shuffle(251, a + k_block * 2 + row);
                        hfma2(252 + row, 2, 249); // A.H0 broadcast * P0
                        hfma2(252 + row, 3, 250); // A.H1 broadcast * P1
                    }
                }
            }
            for (u32 row = 0; row < 2; ++row)
                emit(0x5c98000000000000ULL | predicate | (dest + row) | (u64{15} << 39) | (u64{252 + row} << 20));
            break;
        }
        case 0x83b: { // LDSM.16.M88.{1,2,4} (ldmatrix): 8x8 16-bit matrices from shared memory.
            // Lanes 8m..8m+7 give the row addresses of matrix m; lane t receives the 32 bits at
            // row t/4, byte (t%4)*4 of each matrix in Rd+m. Emulated with SHFL.IDX + LDS.32.
            const u32 count = 1U << insn.Bits(72, 2);
            const bool transpose = insn.Bits(78, 1) != 0;
            if (insn.Bits(72, 2) == 3 || (insn.words[2] & ~0x4300U) || insn.Bits(32, 8) ||
                dest + count > 246)
                return fail(index, "native matrix load mode");
            u32 address = insn.Bits(24, 8);
            const u64 offset = insn.Bits(40, 24);
            if (address != 255 && address >= dest && address < dest + count) {
                emit(0x5c10000000000000ULL | predicate | 253 | (u64{address} << 8) | (u64{255} << 20));
                address = 253;
            }
            if (transpose) {
                // .MT88: lane t receives stored rows 2(t%4) and 2(t%4)+1 at column t/4 (16 bits each).
                emit(0xf0c8000000000000ULL | predicate | 246);                                    // laneid
                emit(0x0400000000000000ULL | predicate | 247 | (u64{246} << 8) | (u64{3} << 20));
                emit(0x3848000000000000ULL | predicate | 247 | (u64{247} << 8) | (u64{1} << 20)); // row 2(t%4)
                emit(0x3828000000000000ULL | predicate | 248 | (u64{246} << 8) | (u64{2} << 20));
                emit(0x3848000000000000ULL | predicate | 248 | (u64{248} << 8) | (u64{1} << 20)); // byte 2(t/4)
                for (u32 matrix = 0; matrix < count; ++matrix) {
                    for (u32 half = 0; half < 2; ++half) {
                        emit(0x1c00000000000000ULL | predicate | 249 | (u64{247} << 8) |
                             (u64{matrix * 8 + half} << 20));
                        emit(0xef10000000000000ULL | predicate | 250 | (u64{address} << 8) | (u64{249} << 20) |
                             (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
                        emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{248} << 20));
                        emit(0xef48000000000000ULL | predicate | (251 + half) | (u64{250} << 8) |
                             (offset << 20) | (u64{2} << 48));                                  // LDS.U16
                    }
                    emit(0x5c18000000000000ULL | predicate | (dest + matrix) | (u64{252} << 8) |
                         (u64{251} << 20) | (u64{16} << 39));
                }
                break;
            }
            emit(0xf0c8000000000000ULL | predicate | 250);                                    // S2R laneid
            emit(0x3828000000000000ULL | predicate | 251 | (u64{250} << 8) | (u64{2} << 20)); // row
            emit(0x0400000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{3} << 20));
            emit(0x3848000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{2} << 20)); // column bytes
            for (u32 matrix = 0; matrix < count; ++matrix) {
                u32 lane = 251;
                if (matrix) {
                    emit(0x1c00000000000000ULL | predicate | 252 | (u64{251} << 8) | (u64{matrix * 8} << 20));
                    lane = 252;
                }
                emit(0xef10000000000000ULL | predicate | 253 | (u64{address} << 8) | (u64{lane} << 20) |
                     (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
                emit(0x5c10000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{250} << 20));
                emit(0xef48000000000000ULL | predicate | (dest + matrix) | (u64{253} << 8) |
                     (offset << 20) | (u64{4} << 48));
            }
            break;
        }
        case 0x919: case 0x9c3: { // S2R / S2UR thread and workgroup identities.
            const u32 special = insn.Bits(72, 8);
            // S2R (not S2UR) of SR_INVOCATION_ID (17) and SR_INVOCATION_INFO (29) in tessellation
            // and geometry shaders: same SR numbers as Maxwell, whose S2R translator reads them as
            // InvocationId and InvocationInfo (vertex count << 16).
            const bool stage_special = opcode == 0x919 && (special == 17 || special == 29);
            // S2R (not S2UR) of SR_THREAD_KILL (19, Mesa NAK NAK_SV_THREAD_KILL = 0x13): nonzero in a
            // helper invocation (NAK lowers is_helper_invocation to it and tests it != 0). Maxwell's
            // S2R has the same number; its translator reads it as IsHelperInvocation ? -1 : 0.
            const bool thread_kill = opcode == 0x919 && special == 19;
            // S2R (not S2UR) of SR_VARIABLE_RATE (132, NAK_SV_VARIABLE_RATE = 0x84): the fragment's
            // shading-rate width (1, 2 or 4) in bits 8-15 and height in bits 16-23. Maxwell has no
            // such register; Eden never sets a fragment shading rate, so every fragment is 1x1.
            const bool variable_rate = opcode == 0x919 && special == 132;
            if (insn.words[1] || insn.words[2] != (special << 8) ||
                (special != 0 && special != 33 && special != 34 && special != 35 &&
                 special != 37 && special != 38 && special != 39 &&
                 special != 56 && special != 57 && special != 58 && special != 59 && special != 60 &&
                 !stage_special && !thread_kill && !variable_rate))
                return fail(index, "native special register");
            const u32 target = opcode == 0x9c3 ? uniform_register(static_cast<u32>(dest)) : static_cast<u32>(dest);
            if (target == ~0U) return fail(index, "native uniform special destination");
            if (variable_rate) {
                if (target != 255) mov_imm(target, 0x00010100); // 1x1; RZ: discarded
                break;
            }
            emit(0xf0c8000000000000ULL | predicate | target | (u64{special} << 20));
            break;
        }
        case 0x28c: case 0x88c: { // UISETP, signed (bit 73) or unsigned, AND/OR/XOR (bits 74-75)
            // with an accumulator (bits 87-89, negate 90). A UPn accumulator is swapped from R239
            // bit 6-n into P(6-n) as Maxwell ISETP's Pc (bits 39-41, negate 42, bop 45-46).
            const u32 output = insn.Bits(81, 3);
            const u32 a = uniform_register(insn.Bits(24, 8)), b = opcode == 0x88c ? 252U : uniform_register(insn.Bits(32, 8));
            const u32 combine = insn.Bits(74, 2), accumulator = insn.Bits(87, 3);
            if (output > 6 || insn.Bits(84, 3) != 7 || combine > 2 ||
                insn.Bits(72, 1) || insn.Bits(12, 4) != 7 || a == ~0U || b == ~0U)
                return fail(index, "native uniform comparison mode");
            const u32 temporary = 6 - output, mask = 1U << temporary;
            const u32 source = accumulator == 7 ? 7U : 6 - accumulator;
            const u32 loaded = accumulator == 7 ? 0U : 1U << source, saved = mask | loaded;
            if (opcode == 0x88c) mov_imm(252, insn.words[1]);
            emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{saved} << 20));
            if (loaded) emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{loaded} << 20));
            emit(0x5b60000000070000ULL | 7 | (u64{temporary} << 3) |
                (u64{a} << 8) | (u64{b} << 20) | (u64{source} << 39) | (u64{insn.Bits(90, 1)} << 42) |
                (u64{combine} << 45) | (u64{insn.Bits(73, 1)} << 48) | (u64{insn.Bits(76, 3)} << 49));
            emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
            emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{saved} << 20));
            break;
        }
        case 0x287: case 0x887: { // USEL, UP0/UP1; preserve ordinary predicates.
            const u32 condition = insn.Bits(87, 3);
            const u32 target = uniform_register(static_cast<u32>(dest));
            const u32 a = uniform_register(insn.Bits(24, 8)), b = opcode == 0x887 ? 252U : uniform_register(insn.Bits(32, 8));
            if (condition > 6 || target == ~0U || a == ~0U || b == ~0U ||
                insn.Bits(12, 4) != 7 || (insn.words[2] & ~0x07800000U) != 0x08000000U)
                return fail(index, "native uniform select mode");
            if (opcode == 0x887 && index > 0 && condition == 0 && insn.words[1] <= 0x7ffff &&
                code[index - 1].Bits(0, 12) == 0x88c &&
                code[index - 1].Bits(24, 8) == insn.Bits(24, 8) &&
                code[index - 1].words[1] == insn.words[1] && code[index - 1].words[2] == 0x0bf01070 &&
                insn.words[2] == 0x08000000) {
                // Preserve the unsigned clamp idiom for indirect-table analysis.
                emit(0x3820000000070000ULL | target | (u64{a} << 8) |
                     (u64{insn.words[1]} << 20) | (u64{7} << 39));
                break;
            }
            if (opcode == 0x887) mov_imm(252, insn.words[1]);
            const u32 temporary = 6 - condition, mask = 1U << temporary;
            emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{mask} << 20));
            emit(0x5ca0000000070000ULL | target | (u64{a} << 8) | (u64{b} << 20) |
                (u64{temporary} << 39) | (u64{insn.Bits(90, 1)} << 42));
            emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
            break;
        }
        case 0x381: case 0x981: { // LDG.E of 8 to 128 bits, register or uniform 64-bit address.
            // Sizes as Maxwell's: U8, S8, U16, S16, 32, 64, 128.
            const u32 size = insn.Bits(73, 3);
            u32 address = insn.Bits(24, 8);
            if (size > 6 || (opcode == 0x381 && insn.Bits(32, 8)) ||
                (size == 5 && (dest & 1)) || (size == 6 && (dest & 3)))
                return fail(index, "native global load mode");
            if (opcode == 0x381) {
                // Bit 77 (strong/scoped ordering) needs nothing extra from a Maxwell LDG.
                if ((insn.words[2] & ~0x2e00U) != 0x001e8100 || (address & 1) || address >= 246)
                    return fail(index, "native global load mode");
            } else {
                // The uniform base sits where a register source would (bits 32-37).
                const u32 uniform_address =
                    insn.Bits(38, 2) ? ~0U : uniform_register(insn.Bits(32, 6));
                if ((insn.words[2] & ~(0x04000000U | 0x3fU | 0xe00U | 0x2000U)) != 0x081e8100 ||
                    uniform_address == ~0U || (uniform_address & 1))
                    return fail(index, "native global load uniform addressing mode");
                if (address == 255) {
                    address = uniform_address;
                } else {
                    if (insn.Bits(90, 1) && ((address & 1) || address >= 246))
                        return fail(index, "native global load register address pair");
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{uniform_address} << 8) |
                         (u64{address} << 20) | (u64{1} << 47));
                    emit(0x5c10000000000000ULL | predicate | 251 | (u64{uniform_address + 1} << 8) |
                         (u64{insn.Bits(90, 1) ? address + 1 : 255} << 20) | (u64{1} << 43));
                    address = 250;
                }
            }
            emit(base(0xeed0000000000000ULL) | (u64{address} << 8) |
                 (u64{insn.Bits(40, 24)} << 20) | (u64{1} << 45) | (u64{size} << 48));
            break;
        }
        case 0x386: case 0x986: { // Global store of 32/64/128 bits, 64-bit address.
            u32 data = insn.Bits(32, 8);
            const u32 size = insn.Bits(73, 3);
            if (size > 6 ||
                (data != 255 && ((size == 5 && (data & 1)) || (size == 6 && (data & 3)) ||
                                 data + (size < 4 ? 1U : 1U << (size - 4)) > 246)))
                return fail(index, "native global store size");
            if (data == 255 && size > 4) {
                // Zeros wider than one register: store from zeroed scratch registers (aligned to the
                // access: R248-R251 for 128 bits, R250-R251 for 64).
                data = size == 6 ? 248 : 250;
                for (u32 i = 0; i < (1U << (size - 4)); ++i) mov_imm(data + i, 0);
            }
            u32 address = insn.Bits(24, 8);
            bool extended = insn.Bits(72, 1) != 0;
            if (opcode == 0x986) {
                const u32 uniform_address = uniform_register(insn.Bits(64, 6));
                if ((insn.words[2] & ~(0x04000000U | 0x3fU | 0xe00U)) != 0x0810a100 ||
                    uniform_address == ~0U || (uniform_address & 1) || !extended)
                    return fail(index, "native global store uniform addressing mode");
                if (address == 255) address = uniform_address;
                else {
                    if (insn.Bits(90, 1) && ((address & 1) || address >= 246))
                        return fail(index, "native global store register address pair");
                    // Address into R246:R247: R250:R251 (and R248:R249) may already hold the
                    // zeros of an RZ store of 64/128 bits; summing the address there stored the
                    // address instead of zeros (HWAI's RefProbe spherical-harmonics clear).
                    emit(0x5c10000000000000ULL | predicate | 246 | (u64{uniform_address} << 8) |
                        (u64{address} << 20) | (u64{1} << 47));
                    emit(0x5c10000000000000ULL | predicate | 247 | (u64{uniform_address + 1} << 8) |
                        (u64{insn.Bits(90, 1) ? address + 1 : 255} << 20) | (u64{1} << 43));
                    address = 246;
                }
            } else if ((insn.words[2] & ~0xe00U) != 0x0010a100) {
                return fail(index, "native global store register addressing mode");
            }
            emit(0xeed8000000000000ULL | predicate | data |
                (u64{address} << 8) | (u64{insn.Bits(40, 24)} << 20) |
                (u64{extended} << 45) | (u64{size} << 48));
            break;
        }
        case 0x95b: // Fragment discard; execution predicate is preserved.
            if (insn.words[1] || insn.words[2] != 0x03800000)
                return fail(index, "native fragment discard mode");
            emit(0xe33000000000000fULL | predicate);
            break;
        case 0x305: case 0xb05: case 0xd05: { // F2I NTZ, register/constant-buffer/uniform source
            const u32 mode = insn.words[2] & ~(opcode == 0xd05 ? 0x08000000U : 0U);
            // F16 source (signed or unsigned); bit 60 selects the high half of Rb.
            const bool half = opcode == 0x305 && (mode & ~0x100U) == 0x0010f000;
            // U16.F32.NTZ.FTZ, rounding RN (bits 78-79 = 0) or RZ (3).
            const bool small_rz = mode == 0x0021e800;
            const bool small = mode == 0x00212800 || small_rz;
            const bool floor = (mode & ~0x10100U) == 0x00207000;
            const bool nearest = (mode & ~0x10100U) == 0x00203000;
            // 32-bit result, F32 source, NTZ, rounding RP (bits 78-79 = 2); signed and FTZ optional.
            const bool ceil = (mode & ~0x10100U) == 0x0020b000;
            if ((!half && !small && !floor && !nearest && !ceil && (mode & ~0x100U) != 0x0021f000) ||
                (opcode != 0xb05 && (insn.words[1] & (half ? 0xefffff00U : 0x3fffff00U))) ||
                // These modes take a constant only without |c|/-c (bits 62-63), the F16 swizzle
                // (60-61), bit 59 or a bindless uniform register (32-37).
                (opcode == 0xb05 && (ceil || small_rz) && (insn.words[1] & 0xf800003fU)) ||
                (opcode == 0xd05 && uniform_register(insn.Bits(32, 8)) == ~0U)) {
                return fail(index, "native float-to-integer conversion mode");
            }
            const bool is_signed = insn.Bits(72, 1);
            u32 source = opcode == 0xb05 ? materialize(2, 0, 252, false, false)
                       : opcode == 0xd05 ? uniform_register(insn.Bits(32, 8)) : insn.Bits(32, 8);
            if (half) {
                extract_half({source, insn.Bits(60, 1) ? 3U : 2U, false, false}, 0, 252);
                source = 252;
            } else if (opcode != 0xb05 && insn.Bits(62, 2)) {
                // |Rb| (bit 62) and -Rb (bit 63) of a 32-bit float register source.
                source = materialize(0, source, 252, insn.Bits(62, 1), insn.Bits(63, 1));
            }
            if (!is_signed) {
                // Clear NaNs without consuming a guest predicate. Clamp the
                // conversion below 2^32, retaining an overflow mask for saturation.
                emit(0x5800000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{source} << 20) | (u64{7} << 39) | (u64{7} << 48));
                emit(0x5c40000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{250} << 20) | (u64{7} << 48));
                if (!small) {
                mov_imm(252, 0x4f800000);
                emit(0x5800000000000000ULL | predicate | 251 | (u64{250} << 8) |
                     (u64{252} << 20) | (u64{7} << 39) | (u64{6} << 48));
                mov_imm(252, 0x4f7fffff);
                emit(0x5c60000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{252} << 20) | (u64{7} << 39));
                }
                source = 250;
            }
            // Native rounding (78-79: RN 0, RM 1, RP 2, RZ 3) is Maxwell's (39-40: Round, Floor,
            // Ceil, Trunc). Every accepted mode fixes it: RN, RM, RP, RZ (F16 and 32.F32.RZ.FTZ), U16 RN/RZ.
            emit(base(0x5cb0000000000000ULL) | (u64{small ? 1U : 2U} << 8) | (u64{2} << 10) |
                 (u64{is_signed} << 12) | (u64{source} << 20) |
                 (u64{insn.Bits(78, 2)} << 39) | (u64{insn.Bits(80, 1)} << 44));
            if (!is_signed && !small) {
                emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{251} << 20) |
                     (u64{1} << 41) | (u64{7} << 48));
            }
            break;
        }
        case 0xb6f: { // TXQ texture dimensions, two packed scalar outputs.
            // The handle comes from c[bank][offset * 4] (bank at 54-58, as for TLD/TEX/TLD4).
            // With three or four channels, Rd/Rd+1 get the first two and Rd2 (64-71) the rest,
            // as for the other split-destination texture instructions. Without Rd2: width (mask
            // 1), height (2), both (3), or the depth / layer count alone (4, which Ocarina of
            // Time's effect shaders query), packed into Rd as Maxwell's TXQ writes it.
            const u32 mask = insn.Bits(72, 4), second_dest = insn.Bits(64, 8);
            const u32 binding = insn.Bits(54, 5);
            const u32 components = static_cast<u32>(std::popcount(mask));
            const bool split = components > 2;
            if (insn.Bits(59, 5) || mask == 0 || insn.Bits(76, 14) || insn.Bits(91, 1) || binding >= 18 ||
                insn.Bits(40, 14) > 0x1fff || dest >= 254 ||
                (split ? (second_dest == 255 || (dest & 1) || (components == 4 && (second_dest & 1)) ||
                          (second_dest >> 1) == (dest >> 1))
                       : (second_dest != 255 || (mask > 3 && mask != 4) || (mask == 3 && (dest & 1)))))
                return fail(index, "native texture dimension query mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native texture descriptor buffer conflict");
            result.required_texture_buffer = binding;
            emit(0xdf48000000000000ULL | predicate | (split ? 250U : dest) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{1} << 22) | (u64{mask} << 31) | (u64{insn.Bits(40, 14)} << 36) |
                 (u64{insn.Bits(90, 1)} << 49));
            for (u32 i = 0; split && i < components; ++i) {
                const u32 target = i < 2 ? static_cast<u32>(dest) + i : second_dest + (i - 2);
                emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{250 + i} << 20));
            }
            break;
        }
        case 0xb66: { // TLD.SCR.LZ, direct texel fetch
            const u32 mask = insn.Bits(72, 4);
            const u32 dimension = insn.Bits(61, 3);
            const bool array = dimension == 5;
            const bool volume = dimension == 2;
            u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 second_dest = insn.Bits(64, 8);
            const bool split = second_dest != 255;
            constexpr std::array<u32, 8> rg_masks{1, 2, 4, 8, 3, 9, 10, 12};
            constexpr std::array<u32, 5> rgba_masks{7, 11, 13, 14, 15};
            u32 swizzle = ~0U;
            if (split) {
                for (u32 i = 0; i < rgba_masks.size(); ++i) if (rgba_masks[i] == mask) swizzle = i;
            } else {
                for (u32 i = 0; i < rg_masks.size(); ++i) if (rg_masks[i] == mask) swizzle = i;
            }
            // Bit 79: FP16 results, two components per destination register.
            const bool half_results = insn.Bits(79, 1) != 0;
            // Channel sets Maxwell's TLDS swizzles cannot express (GB in Hyrule Warriors: Age of
            // Imprisonment): fetch all four channels into R250-R253 and copy the wanted ones out.
            const bool any_mask = swizzle == ~0U && mask != 0 && !half_results;
            // 2D forms with the coordinates in the pair (Ra, Ra+1) and Rb holding the packed
            // 4-bit offsets (.AOFFI, bit 76, with .LZ), the explicit LOD (.LL, LOD mode 3) or the
            // sample index (.MS, bit 78, with .LZ): Maxwell's TLDS encodings 4, 5 and 6.
            const bool offsets = insn.Bits(76, 1) != 0, multisample = insn.Bits(78, 1) != 0;
            const u32 lod = insn.Bits(87, 3);
            const bool paired = dimension == 1 && !insn.Bits(77, 1) && !(offsets && multisample) &&
                                (lod == ((offsets || multisample) ? 1U : 3U));
            if (!insn.Bits(60, 1) || (dimension != 0 && dimension != 1 && !array && !volume) ||
                (swizzle == ~0U && !any_mask) || (insn.Bits(76, 3) && !paired) || insn.Bits(80, 1) ||
                insn.Bits(81, 3) != 7 || (lod != 1 && !paired) ||
                (paired && ((source_a & 1) || source_a + 1 >= (uses_uniform_bank ? 216U : 246U))) ||
                insn.Bits(59, 1) ||
                insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) >= 18 ||
                dest == 255 || (!half_results && mask & (mask - 1) && ((dest & 1) || dest >= 254)) ||
                (!half_results && split && mask == 15 && ((second_dest & 1) || second_dest >= 254)) ||
                (array && ((source_a & 1) || source_a >= 249)) ||
                (volume && ((source_a & 1) || source_a >= 254))) {
                return fail(index, "native texel fetch mode or descriptor binding");
            }
            const u32 binding = insn.Bits(54, 5);
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native texture descriptor buffer conflict");
            result.required_texture_buffer = binding;
            if (array) {
                emit(0x5c98000000000000ULL | predicate | 250 | (u64{15} << 39) | (u64{source_a + 1} << 20));
                emit(0x5c98000000000000ULL | predicate | 251 | (u64{15} << 39) | (u64{source_b} << 20));
                source_b = 250;
            }
            const u32 maxwell_type = paired ? (offsets ? 4U : multisample ? 6U : 5U)
                                   : array ? 8U : volume ? 7U : dimension == 0 ? 0U : 2U;
            if (any_mask) {
                emit(0xda00000000000000ULL | predicate | 250 | (u64{source_a} << 8) |
                     (u64{source_b} << 20) | (u64{252} << 28) |
                     (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                     (u64{4} << 50) | (u64{maxwell_type} << 53));
                u32 written = 0;
                for (u32 channel = 0; channel < 4; ++channel) {
                    if (!(mask & (1U << channel))) continue;
                    const u32 target = !split || written < 2 ? static_cast<u32>(dest) + written
                                                             : second_dest + (written - 2);
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) |
                         (u64{250 + channel} << 20));
                    ++written;
                }
                break;
            }
            if (half_results) {
                // Fetch 32-bit components into R250/R251 (and R252/R253 for the second
                // destination), then pack them into the native destinations.
                emit(0xda00000000000000ULL | predicate | 250 | (u64{source_a} << 8) |
                     (u64{source_b} << 20) | (u64{split ? 252U : 255U} << 28) |
                     (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                     (u64{swizzle} << 50) | (u64{maxwell_type} << 53));
                pack_half_results(static_cast<u32>(dest), split ? second_dest : 255U,
                                  static_cast<u32>(std::popcount(mask)));
                break;
            }
            emit(0xda00000000000000ULL | predicate | dest | (u64{source_a} << 8) |
                 (u64{source_b} << 20) | (u64{insn.Bits(64, 8)} << 28) |
                 (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                 (u64{swizzle} << 50) | (u64{maxwell_type} << 53));
            break;
        }
        case 0x91a: // DEPBAR.LE SBn, count{, mask}: scoreboard wait, represented by IR data dependencies.
            // Bit 47 .LE, bits 44-46 the scoreboard, 38-43 the count; bits 32-37 also name
            // scoreboards (main.ro compute uses 0x8002, 0x800e, 0x8020). Any wait has no data effect.
            if (insn.words[0] != 0x791a || (insn.words[1] & ~0x7fffU) != 0x8000 || insn.words[2]) {
                return fail(index, "native dependency barrier mode");
            }
            break;
        case 0xb69: { // TMML (LOD query), 2D. Results are 8.8 fixed point, as on Maxwell.
            const u32 mask = insn.Bits(72, 4), second_dest = insn.Bits(64, 8);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 binding = insn.Bits(54, 5);
            if (insn.Bits(61, 3) != 1 || insn.Bits(76, 4) || insn.Bits(80, 10) || insn.Bits(59, 1) ||
                insn.Bits(40, 14) > 0x1fff || binding >= 18 || mask == 0 || source_a >= 246 ||
                (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding))
                return fail(index, "native texture LOD query mode");
            result.required_texture_buffer = binding;
            // Coordinates: (Ra, Ra+1) when Rb is RZ, else (Ra, Rb).
            emit(0x5c98000000000000ULL | predicate | 250 | (u64{15} << 39) | (u64{source_a} << 20));
            emit(0x5c98000000000000ULL | predicate | 251 | (u64{15} << 39) |
                 (u64{source_b == 255 ? source_a + 1 : source_b} << 20));
            emit(0xdf58000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                 (u64{2} << 28) | (u64{mask} << 31) | (u64{insn.Bits(40, 14)} << 36) |
                 (u64{insn.Bits(90, 1)} << 49));
            const u32 components = static_cast<u32>(std::popcount(mask));
            const bool split = second_dest != 255;
            for (u32 i = 0; i < components; ++i) {
                u32 target = 255;
                if (!split || i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                else target = second_dest + (i - 2);
                if (target != 255)
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{250 + i} << 20));
            }
            break;
        }
        case 0xb63: { // TLD4 (gather), 2D without offsets or depth compare; component at bits 87-88.
            const u32 mask = insn.Bits(72, 4), second_dest = insn.Bits(64, 8);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 binding = insn.Bits(54, 5);
            // Also 2D with .AOFFI (bit 76) and ARRAY_2D (dimension 5) with or without it. The
            // operand list {array index, x, y, packed 6-bit offsets} splits two per register:
            // 2D.AOFFI is {x, y} in Ra/Ra+1 and the offsets in Rb; ARRAY_2D is {index, x} in
            // Ra/Ra+1, then y (and the offsets) in Rb (Rb+1). Maxwell TLD4 reads the array index,
            // x, y from consecutive registers and the offsets from its meta register.
            const u32 dimension = insn.Bits(61, 3);
            const bool offsets = insn.Bits(76, 1) != 0, array = dimension == 5;
            const u32 scratch_base = uses_uniform_bank ? 216U : 246U;
            const bool extended = offsets || array;
            if ((dimension != 1 && !array) || insn.Bits(77, 2) || insn.Bits(81, 3) != 7 || insn.Bits(59, 1) ||
                insn.Bits(40, 14) > 0x1fff || binding >= 18 || insn.Bits(89, 1) || mask == 0 ||
                (extended && ((source_a & 1) || source_a + 1 >= scratch_base)) ||
                (array && offsets && ((source_b & 1) || source_b + 1 >= scratch_base)) ||
                (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding))
                return fail(index, "native texture gather mode");
            result.required_texture_buffer = binding;
            if (extended) {
                const auto copy = [&](u32 target, u32 source) {
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{source} << 20));
                };
                copy(250, source_a);
                copy(251, source_a + 1);
                if (array) copy(252, source_b);
                if (offsets) copy(array ? 253U : 252U, array ? source_b + 1 : source_b);
                emit(0xc800000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{offsets ? (array ? 253U : 252U) : 255U} << 20) | (u64{array ? 3U : 2U} << 28) |
                     (u64{mask} << 31) | (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                     (u64{7} << 51) | (u64{offsets} << 54) | (u64{insn.Bits(87, 2)} << 56));
            } else {
                // Two coordinates: x in Ra, y in Rb (the coordinate list splits evenly across them).
                emit(0x5c98000000000000ULL | predicate | 250 | (u64{15} << 39) | (u64{source_a} << 20));
                emit(0x5c98000000000000ULL | predicate | 251 | (u64{15} << 39) |
                     (u64{source_b == 255 ? 255U : source_b} << 20));
                emit(0xc800000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                     (u64{2} << 28) | (u64{mask} << 31) | (u64{insn.Bits(40, 14)} << 36) |
                     (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51) | (u64{insn.Bits(87, 2)} << 56));
            }
            const u32 components = static_cast<u32>(std::popcount(mask));
            const bool split = second_dest != 255;
            if (insn.Bits(79, 1)) {
                // FP16 results, packed as TEX does: (c0, c1) into the first destination,
                // (c2, c3) into the second (Hyrule Warriors: Age of Imprisonment lighting).
                pack_half_results(static_cast<u32>(dest),
                                  split ? second_dest : (dest == 255 ? 255U : static_cast<u32>(dest) + 1),
                                  components);
                break;
            }
            for (u32 i = 0; i < components; ++i) {
                u32 target = 255;
                if (!split || i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                else target = second_dest + (i - 2);
                if (target != 255)
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{250 + i} << 20));
            }
            break;
        }
        case 0xb60: { // TEX, scalar 2D implicit-LOD sample
            const u32 mask = insn.Bits(72, 4);
            const u32 second_dest = insn.Bits(64, 8);
            constexpr std::array<u32, 8> rg_masks{1, 2, 4, 8, 3, 9, 10, 12};
            constexpr std::array<u32, 5> rgba_masks{7, 11, 13, 14, 15};
            u32 swizzle = ~0U;
            const bool split = second_dest != 255;
            const bool depth_compare = insn.Bits(78, 1);
            const u32 dimension = insn.Bits(61, 3), lod = insn.Bits(87, 3);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            if (dimension == 5 && depth_compare && lod == 0 && !split && mask == 1) {
                if (!insn.Bits(60, 1) || insn.Bits(76, 2) || insn.Bits(81, 3) != 7 ||
                    insn.Bits(59, 1) || insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) != 2 ||
                    (source_a & 1) || (source_b & 1) || source_a >= 248 || source_b >= 248 || dest >= 248)
                    return fail(index, "native array depth comparison sample mode");
                if (result.required_texture_buffer != ~0U && result.required_texture_buffer != 2)
                    return fail(index, "native texture descriptor buffer conflict");
                result.required_texture_buffer = 2;
                for (u32 i = 0; i < 3; ++i) {
                    const u32 source = i < 2 ? source_a + i : source_b;
                    emit(0x5c98000000000000ULL | predicate | (250 + i) |
                         (u64{15} << 39) | (u64{source} << 20));
                }
                emit(0xc000000000000000ULL | predicate | dest | (u64{250} << 8) |
                     (u64{source_b + 1} << 20) | (u64{3} << 28) | (u64{1} << 31) |
                     (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                     (u64{1} << 50) | (u64{7} << 51));
                break;
            }
            // General form: gather the native operand list (array index first, coordinates,
            // then LOD/bias and the depth reference) into Maxwell TEX's coordinate and meta
            // registers, sample into scratch and copy the packed components out. Native packs
            // up to four values as halves of Ra and Rb (1+1, 2+1, 2+2), five or more as 4+n.
            const auto general_tex = [&]() -> bool {
                constexpr std::array<u32, 8> coordinate_count{1, 2, 3, 3, 2, 3, 0, 4};
                constexpr std::array<u32, 8> maxwell_type{0, 2, 4, 6, 1, 3, 0, 7};
                const bool offsets = insn.Bits(76, 1) != 0;
                // Bit 77 (.NDV, derivatives from the whole quad) is Maxwell TEX bit 35.
                if (dimension == 6 || insn.Bits(81, 3) != 7 || insn.Bits(59, 1) ||
                    insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) >= 18 || (lod > 3 && lod != 6 && lod != 7) ||
                    (depth_compare && mask != 1) || mask == 0)
                    return false;
                const u32 coordinates = coordinate_count[dimension];
                const bool has_lod = lod == 2 || lod == 3 || lod == 6 || lod == 7;
                const u32 count = coordinates + (has_lod ? 1 : 0) + (offsets ? 1 : 0) + (depth_compare ? 1 : 0);
                const u32 in_a = count <= 4 ? (count + 1) / 2 : 4;
                const auto value = [&](u32 i) -> u32 {
                    if (i < in_a) return source_a == 255 ? 255U : source_a + i;
                    return source_b == 255 ? 255U : source_b + (i - in_a);
                };
                if ((source_a != 255 && source_a + in_a > 246) ||
                    (count > in_a && source_b != 255 && source_b + (count - in_a) > 246))
                    return false;
                const u32 binding = insn.Bits(54, 5);
                if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                    return false;
                result.required_texture_buffer = binding;
                const auto copy = [&](u32 target, u32 source) {
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{source} << 20));
                };
                // Coordinates to R250.., then LOD/bias, offsets and reference to R246...
                for (u32 i = 0; i < coordinates; ++i) copy(250 + i, value(i));
                u32 meta = 246;
                u32 next = coordinates;
                if (has_lod) copy(meta++, value(next++));
                if (offsets) copy(meta++, value(next++));
                if (depth_compare) copy(meta++, value(next++));
                emit(0xc000000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{meta == 246 ? 255U : 246U} << 20) | (u64{maxwell_type[dimension]} << 28) |
                     (u64{mask} << 31) | (u64{insn.Bits(77, 1)} << 35) | (u64{insn.Bits(40, 14)} << 36) |
                     (u64{insn.Bits(90, 1)} << 49) | (u64{depth_compare} << 50) | (u64{7} << 51) |
                     (u64{offsets} << 54) | (u64{lod} << 55));
                const u32 components = static_cast<u32>(std::popcount(mask));
                if (insn.Bits(79, 1)) {
                    pack_half_results(static_cast<u32>(dest),
                                      split ? second_dest : (dest == 255 ? 255U : static_cast<u32>(dest) + 1),
                                      components);
                    return true;
                }
                for (u32 i = 0; i < components; ++i) {
                    u32 target = 255;
                    if (!split) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                    else if (i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                    else target = second_dest + (i - 2);
                    if (target != 255) copy(target, 250 + i);
                }
                return true;
            };
            if (split) {
                for (u32 i = 0; i < rgba_masks.size(); ++i) if (rgba_masks[i] == mask) swizzle = i;
            } else {
                for (u32 i = 0; i < rg_masks.size(); ++i) if (rg_masks[i] == mask) swizzle = i;
            }
            // SM70 scalar coordinates and split RG/BA destinations match TEXS.
            // Descriptor offsets are words in both encodings, but only native
            // explicitly encodes the constant-buffer binding.
            if (!insn.Bits(60, 1) || (dimension != 1 && dimension != 5) ||
                swizzle == ~0U || insn.Bits(76, 2) ||
                insn.Bits(81, 3) != 7 || (lod > 1 && (lod != 3 || dimension != 1 || depth_compare)) ||
                (lod == 3 && ((source_a & 1) || source_a >= 249)) ||
                insn.Bits(59, 1) ||
                insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) != 2 ||
                dest == 255 || (mask & (mask - 1) && ((dest & 1) || dest >= 254)) ||
                (split && mask == 15 && ((second_dest & 1) || second_dest >= 254)) ||
                (depth_compare && (dimension != 1 || split || mask != 1 || (source_a & 1) || source_a >= 249)) ||
                (dimension == 5 && ((source_a & 1) || source_a >= 249))) {
                if (!general_tex()) return fail(index, "native texture mode or descriptor binding");
                break;
            }
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != 2)
                return fail(index, "native texture descriptor buffer conflict");
            result.required_texture_buffer = 2;
            emit((0xd800000000000000ULL & ~(u64{insn.Bits(79, 1)} << 59)) | predicate | dest | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(64, 8)} << 28) |
                 (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                 (u64{swizzle} << 50) | (u64{dimension == 5 ? (lod ? 8U : 7U) : depth_compare ? (lod ? 6U : 4U) : (lod == 3 ? 3U : lod ? 2U : 1U)} << 53));
            break;
        }
        case 0x361: { // TEX.B: the texture handle is a register value (NAK OpTex, TexRef::Bindless).
            // The operand list is 0xb60's with the handle right after the coordinates, packed as
            // halves of Ra and Rb (bit 60): 2D.DC is (x, y | handle, reference) and 3D is
            // (x, y | z, handle) (Ocarina of Time: Model shadows, agl HDR LUTs). Only these
            // four-value forms are accepted: implicit LOD or LZ, no offsets/NDV/FP16/sparse
            // result, no constant-buffer reference bits (40-58) and no bit 91.
            const u32 mask = insn.Bits(72, 4);
            const u32 second_dest = insn.Bits(64, 8);
            const u32 dimension = insn.Bits(61, 3), lod = insn.Bits(87, 3);
            const bool depth_compare = insn.Bits(78, 1) != 0;
            const bool split = second_dest != 255;
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 coordinates = dimension == 1 ? 2U : 3U;
            const u32 components = static_cast<u32>(std::popcount(mask));
            const u32 first_count = split ? std::min(components, 2U) : components;
            // Register pairs and results must stay below the lowering's reserved registers.
            const u32 limit = uses_uniform_bank ? 216U : 246U;
            if (!insn.Bits(59, 1) || !insn.Bits(60, 1) || (dimension != 1 && dimension != 2) ||
                lod > 1 || insn.Bits(76, 2) || insn.Bits(79, 2) || insn.Bits(81, 3) != 7 ||
                insn.Bits(40, 19) || insn.Bits(91, 1) || mask == 0 ||
                depth_compare != (dimension == 1) || (depth_compare && (mask != 1 || split)) ||
                (source_a != 255 && source_a + 2 > limit) || (source_b != 255 && source_b + 2 > limit) ||
                (dest != 255 && dest + first_count > limit) ||
                (split && components > 2 && second_dest + (components - 2) > limit))
                return fail(index, "native bindless texture mode");
            const auto value = [&](u32 i) -> u32 {
                if (i < 2) return source_a == 255 ? 255U : source_a + i;
                return source_b == 255 ? 255U : source_b + (i - 2);
            };
            const auto copy = [&](u32 target, u32 source) {
                emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{source} << 20));
            };
            for (u32 i = 0; i < coordinates; ++i) copy(250 + i, value(i));
            copy(246, value(coordinates)); // Maxwell TEX.B reads the handle from its first meta register
            if (depth_compare) copy(247, value(coordinates + 1));
            // Maxwell TEX.B (texture_fetch.cpp TEX_b): type 2D = 2, 3D = 4 at 28, mask at 31,
            // blod at 37 (0 = none, 1 = LZ, as native 87-89), .NODEP 49, DC 50, sparse PT at 51.
            emit(0xde80000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{246} << 20) |
                 (u64{dimension == 1 ? 2U : 4U} << 28) | (u64{mask} << 31) | (u64{lod} << 37) |
                 (u64{insn.Bits(90, 1)} << 49) | (u64{depth_compare} << 50) | (u64{7} << 51));
            for (u32 i = 0; i < components; ++i) {
                u32 target = 255;
                if (!split || i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                else target = second_dest + (i - 2);
                if (target != 255) copy(target, 250 + i);
            }
            break;
        }
        case 0x816: { // Captured PRMT half-word concatenation
            if (insn.words[1] > 0xffff || (insn.words[2] & ~0xffU)) return fail(index, "native permutation mode");
            if (insn.words[1] != 0x5410 && insn.words[1] != 0x5432 && insn.words[1] != 0x7610 && insn.words[1] != 0x7632) {
                // Byte i of the result is byte (s & 7) of c:a, or its sign bit repeated when s & 8.
                const u32 a = insn.Bits(24, 8), c = insn.Bits(64, 8);
                if (a >= 250 && a != 255) return fail(index, "native permutation source");
                mov_imm(250, 0);
                for (u32 i = 0; i < 4; ++i) {
                    const u32 select = (insn.words[1] >> (i * 4)) & 0xf;
                    const u32 source = select & 4 ? c : a, position = (select & 3) * 8;
                    emit(0x3800000000000000ULL | predicate | 251 | (u64{source} << 8) |
                         (u64{select & 8 ? (position + 7) | (1U << 8) : position | (8U << 8)} << 20) |
                         (u64{(select >> 3) & 1} << 48));
                    emit(0x36f0000000000000ULL | predicate | 250 | (u64{251} << 8) |
                         (u64{(i * 8) | (8U << 8)} << 20) | (u64{250} << 39));
                }
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{255} << 20));
                break;
            }
            const u32 a = insn.Bits(24, 8), b = insn.Bits(64, 8);
            if (insn.words[1] != 0x5432 && insn.words[1] != 0x7632) {
                emit(0x0400000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{0xffff} << 20));
            } else {
                emit(0x3828000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{16} << 20));
            }
            if (insn.words[1] == 0x7610 || insn.words[1] == 0x7632) {
                emit(0x0400000000000000ULL | predicate | 251 | (u64{b} << 8) | (u64{0xffff0000} << 20));
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{251} << 20));
            } else {
                emit(base(0x5c18000000000000ULL) | (u64{b} << 8) | (u64{250} << 20) | (u64{16} << 39));
            }
            break;
        }
        case 0x307: case 0xb07: { // FRND.F32, round to an integral floating-point value
            // 0xb07: constant-buffer source; only rounding (78-79) and FTZ (80) may vary in w2.
            if (insn.Bits(75, 2) != 2 || insn.Bits(84, 2) != 2 || insn.Bits(81, 3) ||
                insn.Bits(86, 6) || (opcode == 0xb07 && ((insn.words[2] & ~0x1c000U) != 0x00201000 ||
                insn.Bits(24, 8) || (insn.words[1] & 0x3800003fU))))
                return fail(index, "native floating round format");
            const u32 source = opcode == 0xb07 ? materialize(2, 0, 250, false, false) : insn.Bits(32, 8);
            emit(base(0x5ca8000000000000ULL) | (u64{2} << 8) | (u64{2} << 10) |
                 (u64{source} << 20) | (u64{8 + insn.Bits(78, 2)} << 39) |
                 (u64{insn.Bits(80, 1)} << 44) | (u64{insn.Bits(63, 1)} << 45) |
                 (u64{insn.Bits(62, 1)} << 49));
            break;
        }
        case 0x321: // ALD
        case 0x322: { // AST
            // NAK: .P (bit 76) addresses per-patch attributes, .PHYS is bit 77, ALD.O (bit 79)
            // reads the stage's outputs. Maxwell ALD/AST keep .P in bit 31 and ALD .O in bit 32.
            const bool patch = insn.Bits(76, 1) != 0;
            const bool output = opcode == 0x321 && insn.Bits(79, 1) != 0;
            if (insn.Bits(77, 1) || (opcode == 0x322 && insn.Bits(79, 1)) || (patch && output))
                return fail(index, "native indexed/patch attribute access");
            const u32 address = insn.Bits(40, 10);
            const u32 components = insn.Bits(74, 2) + 1;
            if (address % 4 || address + components * 4 > 0x400 || insn.Bits(24, 8) != 255) {
                return fail(index, "native attribute address or indexing");
            }
            const u32 vertex = opcode == 0x321 ? insn.Bits(32, 8) : insn.Bits(64, 8);
            // Eden maps a[n].P to IR::Patch n / 4 (tessellation levels below 0x18, then generic
            // components up to Component119 = 125); its backends load only generic patch values.
            if (patch && (vertex != 255 || address + components * 4 > 0x1f8 ||
                          (opcode == 0x321 && address < 0x18)))
                return fail(index, "native indexed/patch attribute access");
            if (output) {
                // Eden's Maxwell ALD ignores .O, so only the tessellation coordinate is accepted:
                // ALD.O a[0x2f0]/a[0x2f4] indexed by this lane (NAK reads gl_TessCoord this way),
                // which Eden reads as TessellationEvaluationPointU/V of the invocation itself. The
                // vertex must be this lane's S2R SR_LANEID, reached through single-destination
                // writes of other registers only, with no branch into that range.
                const auto own_lane = [&] {
                    std::optional<u32> lane;
                    for (u32 prior = index; prior-- > 0;) {
                        const Instruction& p = code[prior];
                        if (p.Bits(12, 4) == 15) continue;
                        const u32 op = p.Bits(0, 12), d = p.Bits(16, 8);
                        if ((p.words[0] & 0xff00ffffU) == 0x00007919U && !p.words[1] && !p.words[2] &&
                            d == vertex) {
                            lane = prior;
                            break;
                        }
                        if ((op & 0x1ff) == 0x00c) continue; // ISETP writes predicates only
                        const bool single = op == 0x216 || op == 0x816 || op == 0x224 || op == 0x424 ||
                                            op == 0x824 || op == 0xa24 || op == 0x202 || op == 0x802 ||
                                            op == 0xa02 || op == 0x919 || op == 0x923 || op == 0x321;
                        const u32 width = op == 0x321 ? p.Bits(74, 2) + 1 : 1;
                        if (!single || (d != 255 && vertex >= d && vertex < d + width)) return false;
                    }
                    if (!lane) return false;
                    for (u32 at = 0; at < code.size(); ++at) {
                        const u32 op = code[at].Bits(0, 12);
                        if (op == 0x949 || op == 0x958) return false;
                        if (op != 0x944 && op != 0x945 && op != 0x947) continue;
                        const u64 raw = code[at].words[1] | (u64{code[at].words[2] & 0x3ffff} << 32);
                        const s64 relative = op == 0x945 ? static_cast<s32>(code[at].words[1])
                                                         : static_cast<s64>(raw << 14) >> 14;
                        const s64 target = s64{at} * 16 + 16 + relative;
                        if (target > s64{*lane} * 16 && target <= s64{index} * 16) return false;
                    }
                    return true;
                };
                if (address < 0x2f0 || address + components * 4 > 0x2f8 || !own_lane())
                    return fail(index, "native indexed/patch attribute access");
            }
            // a[0x5c] has no Maxwell meaning (Ampere uses it for a per-primitive hint such as the
            // shading rate). Nothing reads it back; dropping the store keeps full-rate shading.
            if (opcode == 0x322 && !patch && address == 0x5c && components == 1) break;
            u32 value_reg = opcode == 0x321 ? static_cast<u32>(dest) : insn.Bits(32, 8);
            if (value_reg == 255) {
                if (opcode == 0x321) break; // load into RZ: nothing to do
                // AST of RZ writes zeros: store them from zeroed scratch registers.
                for (u32 i = 0; i < components; ++i) mov_imm(250 + i, 0);
                value_reg = 250;
            }
            if (value_reg + components > 255) return fail(index, "native attribute register range");
            const u64 translated = (opcode == 0x321 ? 0xefd8000000000000ULL : 0xeff0000000000000ULL) |
                predicate | value_reg | (u64{255} << 8) | (u64{address} << 20) |
                (u64{patch} << 31) | (u64{output} << 32) |
                (u64{vertex} << 39) | (u64{components - 1} << 47);
            result.instructions.push_back(translated);
            break;
        }
        case 0x923: { // ISBERD Rd, [Ra]: attribute handle of the vertex index in Ra (map access)
            // NAK OpIsberd: offset register 24-31, immediate 40-55, type 74-75, access 76-77, skew
            // 78, output 79; only the all-zero form. Maxwell ISBERD in its default form (no
            // immediate, skew, .O, mode or shift): Rd 0-7, Ra 8-15. Eden copies Ra into Rd, so the
            // following ALD reads the vertex index, as for Maxwell programs.
            if (insn.words[1] || insn.words[2]) return fail(index, "native stage buffer read mode");
            emit(base(0xefd0000000000000ULL) | (u64{insn.Bits(24, 8)} << 8));
            break;
        }
        case 0x324: { // OUT.EMIT / .CUT / .EMIT_THEN_CUT Rd, Rhandle, Rstream (geometry)
            // NAK OpOut, register stream: handle 24-31, stream 32-39, type 78-79 (1 EMIT, 2 CUT,
            // 3 both; 0 is OUT.FINAL). Maxwell OUT (reg): handle 8-15, stream 20-27, EMIT bit 39,
            // CUT bit 40; Eden emits and/or ends the primitive on stream & 3 and writes 0 to Rd.
            const u32 type = insn.Bits(78, 2);
            if ((insn.words[1] & ~0xffU) || (insn.words[2] & ~0xc000U) || type == 0)
                return fail(index, "native geometry output mode");
            emit(base(0xfbe0000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(32, 8)} << 20) | (u64{type} << 39));
            break;
        }
        case 0x34e:   // LEPC Rd: Rd:Rd+1 = address of this instruction
        case 0x357: { // BMOV.64 ATEXIT_PC, Ra: EXIT continues at the address in Ra:Ra+1
            // (Names from nvdisasm's SM90 encoding listing; NAK's BMOV is 0x355/0x356.) Geometry
            // shaders open with LEPC Ra; MOV Rb, size; IADD3 Ra, P0, Ra, Rb, RZ; IADD3.X or
            // IMAD.X Ra+1 (+P0); BMOV.64 ATEXIT_PC, Ra, so that EXIT runs the code just past the
            // captured program: OUT.FINAL R0; BMOV.64 ATEXIT_PC, 0; EXIT (in every HWAI geometry
            // shader). That closes the output primitive, which Eden's geometry stage does by
            // itself, so the handler is dropped: BMOV emits nothing and LEPC writes the
            // program-relative address (as call links do), consumed only by that BMOV.
            const u32 at = opcode == 0x34e ? index : index - 4;
            const auto prologue = [&] {
                if ((opcode == 0x357 && index < 4) || at + 4 >= code.size()) return false;
                const Instruction* p = &code[at];
                const u32 reg = p[0].Bits(16, 8), high = reg + 1, size_reg = p[1].Bits(16, 8);
                if ((p[0].words[0] & 0xff00ffffU) != 0x0000734eU || p[0].words[1] || p[0].words[2] ||
                    reg % 2 || high >= (uses_uniform_bank ? 216U : 246U))
                    return false;
                if ((p[1].words[0] & 0xff00ffffU) != 0x00007802U || p[1].words[2] != 0xf00 ||
                    p[1].words[1] != static_cast<u32>(code.size() - at) * 16 ||
                    size_reg == reg || size_reg == high || size_reg == 255)
                    return false;
                if (p[2].words[0] != (0x00007210U | (reg << 16) | (reg << 24)) ||
                    p[2].words[1] != size_reg || p[2].words[2] != 0x07f1e0ffU)
                    return false;
                const bool add_high = p[3].words[0] == (0x00007210U | (high << 16) | (high << 24)) &&
                                      p[3].words[1] == 0xff && p[3].words[2] == 0x007fe4ffU;
                const bool mad_high = p[3].words[0] == (0xff007224U | (high << 16)) &&
                                      p[3].words[1] == 0xff && p[3].words[2] == (0x000e0400U | high);
                if ((!add_high && !mad_high) || p[4].words[0] != (0x00007357U | (reg << 24)) ||
                    p[4].words[1] || p[4].words[2])
                    return false;
                return std::any_of(code.begin(), code.end(),
                                   [](const Instruction& other) { return other.Bits(0, 12) == 0x324; });
            };
            if (!prologue()) return fail(index, "native exit handler mode");
            if (opcode == 0x34e) {
                mov_imm(static_cast<u32>(dest), index * 16);
                mov_imm(static_cast<u32>(dest) + 1, 0);
            }
            break;
        }
        case 0x387: case 0x983: { // STL / LDL of 8 to 128 bits at [Ra + immediate] (Ra = RZ: absolute).
            // Sizes (bits 73-75) as Maxwell's (bits 48-50): U8, S8, U16, S16, 32, 64, 128. Bits 84-86
            // are the cache eviction hint: 1 (normal), or 3 (.LU, last use) on loads; a hint only.
            // The offset is kept below 0x800000, where Maxwell's absolute (RZ) and sign-extended
            // relative offsets read the same value as the native one.
            // Native local memory starts above the home slots of registers kept in local memory
            // (local_memory_bias, which the environment adds to the local memory size).
            const bool store = opcode == 0x387;
            u32 data = insn.Bits(32, 8);
            const u32 address = insn.Bits(24, 8), size = insn.Bits(73, 3);
            const u32 offset = insn.Bits(40, 24) + result.local_memory_bias;
            const u32 eviction = insn.Bits(84, 3);
            const u32 registers = size < 4 ? 1U : 1U << (size - 4);
            const u32 bytes = size < 4 ? 1U << (size >> 1) : 4U * registers;
            const u32 value = store ? data : static_cast<u32>(dest);
            if (size > 6 || (insn.words[2] & ~0x00700e00U) || offset % bytes || offset > 0x7fffff ||
                (store ? (insn.Bits(16, 8) || eviction != 1)
                       : (data || (eviction != 1 && eviction != 3) || value == 255)) ||
                (value != 255 && ((value & (registers - 1)) ||
                                  value + registers > (uses_uniform_bank ? 216U : 246U))))
                return fail(index, "native local scalar spill mode");
            // A wide load must not write a callee's link register either (only Rd is checked above).
            if (!store && active_link && *active_link > value && *active_link < value + registers)
                return fail(index, "native callee overwrites link register");
            if (store && data == 255 && size > 4) {
                // Zeros wider than one register: store from zeroed scratch registers (aligned to the
                // access: R248-R251 for 128 bits, R250-R251 for 64), as the global store does.
                data = size == 6 ? 248 : 250;
                for (u32 i = 0; i < registers; ++i) mov_imm(data + i, 0);
            }
            emit((store ? 0xef50000000000000ULL : 0xef40000000000000ULL) | predicate |
                 (store ? data : dest) | (u64{address} << 8) | (u64{offset} << 20) | (u64{size} << 48));
            break;
        }
        case 0x803: { // P2R PR, register merge source, immediate mask, byte select (76-77)
            if (insn.words[1] > 0x7f || (insn.words[2] & ~0x3000U))
                return fail(index, "native predicate-to-register mode");
            emit(base(0x38e8000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.words[1]} << 20) | (u64{insn.Bits(76, 2)} << 41));
            break;
        }
        case 0x804: { // R2P PR, register, immediate mask, byte select (76-77)
            if (insn.Bits(16, 8) || insn.words[1] > 0x7f || (insn.words[2] & ~0x3000U))
                return fail(index, "native register-to-predicate mode");
            emit(0x38f0000000000000ULL | predicate | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.words[1]} << 20) | (u64{insn.Bits(76, 2)} << 41));
            break;
        }
        case 0xc02: // MOV uniform register
        case 0x202: // MOV register
        case 0x802: // MOV immediate
        case 0xa02: { // MOV constant buffer
            if (opcode == 0x202 && insn.Bits(72, 4) != 15 && insn.Bits(72, 4) && !insn.Bits(76, 16) &&
                !insn.Bits(40, 24)) {
                // Bits 72-75: quad lane mask; only lanes whose (laneid & 3) bit is set are written
                // (Eden's Maxwell MOV ignores its mask). dest = (src & m) | (dest & ~m) with
                // m = -((mask >> (laneid & 3)) & 1), the quad lane FSWZADD also uses.
                emit(0xf0c8000000000000ULL | predicate | 250);                                    // S2R laneid
                emit(0x0400000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{3} << 20));
                mov_imm(251, insn.Bits(72, 4));
                emit(0x5c28000000000000ULL | predicate | 251 | (u64{251} << 8) | (u64{250} << 20));
                emit(0x0400000000000000ULL | predicate | 251 | (u64{251} << 8) | (u64{1} << 20));
                emit(0x5c10000000000000ULL | predicate | 251 | (u64{255} << 8) | (u64{251} << 20) | (u64{1} << 48));
                emit(base(0x5be0000000000000ULL) | (u64{insn.Bits(32, 8)} << 8) | (u64{251} << 20) |
                     (u64{0xe2} << 28) | (dest << 39) | (u64{7} << 48));
                break;
            }
            if (insn.Bits(72, 4) != 15 || (insn.Bits(76, 16) & ~(opcode == 0xc02 ? 0x8000U : 0U))) return fail(index, "native MOV modifiers");
            if (opcode == 0x802) {
                result.instructions.push_back(base(0x0100000000000000ULL) | (u64{15} << 12) |
                    (u64{insn.words[1]} << 20));
            } else if (opcode == 0xc02) {
                const u32 source = insn.words[1] < 64 ? uniform_register(insn.words[1]) : ~0U;
                if (source == ~0U || insn.words[2] != 0x08000f00) return fail(index, "native uniform move source");
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{source} << 20));
            } else if (opcode == 0x202) {
                if (insn.Bits(40, 24)) return fail(index, "native MOV register modifiers");
                result.instructions.push_back(base(0x5c98000000000000ULL) | (u64{15} << 39) |
                    (u64{insn.Bits(32, 8)} << 20));
            } else {
                if (!valid_cbuf() || insn.Bits(62, 2)) return fail(index, "native MOV constant buffer");
                result.instructions.push_back(base(0x4c98000000000000ULL) | (u64{15} << 39) | cbuf());
            }
            break;
        }
        case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23: case 0xc23: case 0xe23: { // FFMA
            if (insn.Bits(76, 1) || (insn.Bits(81, 11) & ~(opcode == 0xe23 || opcode == 0xc23 ? 0x400U : 0U))) return fail(index, "native FFMA denormal/source mode");
            const u32 a = operand(0), b = operand(1), c = operand(2);
            emit(base(0x5980000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) | (u64{c} << 39) |
                (u64{insn.Bits(77, 1)} << 50) | (u64{insn.Bits(78, 2)} << 51) | (u64{insn.Bits(80, 1)} << 53));
            break;
        }
        case 0x220: case 0x820: case 0xa20: case 0xc20: { // FMUL
            const u32 scale = insn.Bits(84, 3);
            if (insn.Bits(76, 1) || scale == 0 || insn.Bits(81, 3) ||
                insn.Bits(87, 4) || insn.Bits(91, 1) != static_cast<u32>(opcode == 0xc20) ||
                (scale != 4 && (!insn.Bits(80, 1) || insn.Bits(78, 2))))
                return fail(index, "native FMUL scale/mode");
            const u32 a = operand(0), b = operand(1);
            emit(base(0x5c68000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(78, 2)} << 39) | (u64{std::array<u32, 8>{0, 3, 2, 1, 0, 6, 5, 4}[scale]} << 41) | (u64{insn.Bits(80, 1)} << 44) | (u64{insn.Bits(77, 1)} << 50));
            break;
        }
        case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21: case 0xe21: { // FADD
            if (insn.Bits(81, 10) || insn.Bits(91, 1) != static_cast<u32>(opcode == 0xe21))
                return fail(index, "native FADD mode");
            const u32 a = operand(0), b = operand(insn.Bits(9, 3) == 2 || insn.Bits(9, 3) == 3 || insn.Bits(9, 3) == 7 ? 2 : 1);
            emit(base(0x5c58000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(78, 2)} << 39) | (u64{insn.Bits(80, 1)} << 44) | (u64{insn.Bits(77, 1)} << 50));
            break;
        }
        case 0x209: case 0x809: case 0xa09: case 0xc09: { // FMNMX, including a uniform source
            if (insn.Bits(74, 6) || insn.Bits(81, 6) || (opcode != 0xc09 && insn.Bits(91, 1)))
                return fail(index, "native floating min/max mode");
            const u32 a = operand(0), b = operand(1);
            emit(base(0x5c60000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                 (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42) |
                 (u64{insn.Bits(80, 1)} << 44));
            break;
        }
        case 0x308: case 0xb08: case 0xd08: { // MUFU (0xd08: uniform source, which sets bit 91)
            const u32 operation = insn.Bits(74, 6);
            if (operation > 8 || operation == 6 || operation == 7 || insn.Bits(80, 11) ||
                insn.Bits(91, 1) != static_cast<u32>(opcode == 0xd08) ||
                (opcode == 0xd08 && ((insn.words[2] & ~0xfc00U) != 0x08000000 || insn.Bits(24, 8) ||
                                     (insn.words[1] & ~0xc00000ffU))))
                return fail(index, "native MUFU mode");
            // SM70+ MUFU.COS/SIN take the angle in turns (compilers multiply by 1/2pi first);
            // Eden's Maxwell MUFU takes radians (its RRO is a plain move), so scale by 2pi.
            const bool turns = operation == 0 || operation == 1;
            const auto to_radians = [&](u32 source) {
                emit(0x1e00000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{0x40c90fdbU} << 20));
                return 250U;
            };
            if (insn.Bits(73, 1)) {
                if (opcode != 0x308 || insn.Bits(60, 2) > 1)
                    return fail(index, "native half special-function source");
                emit(0x5ca8000000000000ULL | predicate | 250 | (u64{2} << 8) | (u64{1} << 10) |
                     (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(60, 1)} << 41) |
                     (u64{insn.Bits(63, 1)} << 45) | (u64{insn.Bits(62, 1)} << 49));
                if (turns) to_radians(250);
                emit(0x5080000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{operation} << 20));
                emit(base(0x5ca8000000000000ULL) | (u64{1} << 8) | (u64{2} << 10) | (u64{250} << 20));
            } else {
                u32 src = operand(1);
                if (turns) src = to_radians(src);
                emit(base(0x5080000000000000ULL) | (u64{src} << 8) | (u64{operation} << 20));
            }
            break;
        }
        case 0x526: { // IPA.PASS / CONSTANT with the attribute indexed by a uniform register
            const u32 index_register = uniform_register(insn.Bits(24, 8));
            if (index_register == ~0U || insn.Bits(32, 8) != 255 ||
                (insn.Bits(76, 4) != 0 && insn.Bits(76, 4) != 4) || insn.Bits(81, 3) != 7)
                return fail(index, "native indexed IPA mode");
            if (index_register == 255) {
                emit(base(0xe000000000000000ULL) | (u64{255} << 8) | (u64{255} << 20) |
                     (u64{insn.Bits(64, 8)} << 30) | (u64{insn.Bits(78, 1)} << 55));
                break;
            }
            // Maxwell IPA.IDX reads the whole byte address from the index register.
            emit(0x1c00000000000000ULL | predicate | 250 | (u64{index_register} << 8) |
                 (u64{insn.Bits(64, 8) * 4} << 20));
            emit(base(0xe000000000000000ULL) | (u64{250} << 8) | (u64{255} << 20) |
                 (u64{insn.Bits(64, 8)} << 30) | (u64{1} << 38) | (u64{insn.Bits(78, 1)} << 55));
            break;
        }
        case 0x326: { // IPA.PASS / CONSTANT, default or centroid sample location
            // Bits 76-77 are the sample mode (title 1 interpolates most varyings at the centroid),
            // bit 78 the constant mode; Maxwell keeps them at bits 52-53 and 55.
            if (insn.Bits(32, 8) != 255 || insn.Bits(76, 2) > 1 || insn.Bits(79, 1) ||
                insn.Bits(81, 3) != 7)
                return fail(index, "native IPA interpolation mode");
            emit(base(0xe000000000000000ULL) | (u64{255} << 8) | (u64{255} << 20) | (u64{insn.Bits(64, 8)} << 30) |
                 (u64{insn.Bits(76, 2)} << 52) | (u64{insn.Bits(78, 1)} << 55));
            break;
        }
        case 0x81a: { // SGXT.U32 Rd, Ra, n: keep the low n bits (BFE.U32 at position 0)
            if (insn.words[2] || insn.words[1] > 32 || dest >= 255)
                return fail(index, "native sign extend mode");
            emit(base(0x3800000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.words[1] << 8} << 20));
            break;
        }
        case 0x299: { // USHF by a uniform amount, one register shifted (SHL/SHR, bit 75 wraps).
            const u32 dst = uniform_register(insn.Bits(16, 8)), low = uniform_register(insn.Bits(24, 8));
            const u32 shift = uniform_register(insn.Bits(32, 8)), high = uniform_register(insn.Bits(64, 8));
            const bool right = insn.Bits(76, 1) != 0, high_result = insn.Bits(80, 1) != 0;
            if (dst == ~0U || dst == 255 || low == ~0U || shift == ~0U || high == ~0U ||
                native_predicate != 7 || insn.Bits(73, 2) != 3 ||
                (insn.words[2] & ~0x000118ffU) != 0x08000600U)
                return fail(index, "native uniform register shift mode");
            u32 source;
            if (high_result && low == 255) source = high;        // (high:0) >> or << s, high word
            else if (!high_result && high == 255) source = low;  // (0:low) >> or << s, low word
            else return fail(index, "native uniform funnel shift");
            emit((right ? 0x5c28000000000000ULL : 0x5c48000000000000ULL) | predicate | dst |
                 (u64{source} << 8) | (u64{shift} << 20) | (u64{insn.Bits(75, 1)} << 39));
            break;
        }
        case 0x499: { // USHF.L(.W).U32.HI URd, URZ, URb, imm: the uniform form of 0x419 below.
            // Form 2 (NAK encode_ualu_base): the immediate (bits 32-63) is the high source and the
            // shift amount URb is in bits 64-71. High word of (imm:0) << URb = imm << (URb & 31)
            // with .W (bit 75); without .W the U32 shift clamps (URb >= 32 gives 0), as Maxwell
            // SHL without .W. A UPn guard is swapped in (uniform_predication above).
            const u32 dst = uniform_register(insn.Bits(16, 8)), shift = uniform_register(insn.Bits(64, 8));
            if (dst == ~0U || dst == 255 || shift == ~0U || insn.Bits(24, 8) != 63 ||
                (insn.words[2] & ~0x8ffU) != 0x08010600U || (insn.words[3] & 0xc00001ffU))
                return fail(index, "native uniform funnel shift immediate mode");
            mov_imm(250, insn.words[1]);
            emit(0x5c48000000000000ULL | predicate | dst | (u64{250} << 8) | (u64{shift} << 20) |
                 (u64{insn.Bits(75, 1)} << 39));
            break;
        }
        case 0x419: { // SHF.L.W.U32.HI Rd, RZ, Rb, imm (high word of imm:0 << Rb) = imm << (Rb & 31)
            // Without .W (bit 75) the shift clamps (U32 Rb >= 32 gives 0), as Maxwell SHL without .W.
            if (insn.Bits(24, 8) != 255 || (insn.words[2] & ~0x8ffU) != 0x00010600)
                return fail(index, "native funnel shift immediate mode");
            mov_imm(250, insn.words[1]);
            emit(base(0x5c48000000000000ULL) | (u64{250} << 8) | (u64{insn.Bits(64, 8)} << 20) |
                 (u64{insn.Bits(75, 1)} << 39));
            break;
        }
        case 0x619: { // SHF.L(.W).U32.HI Rd, RZ, Rb, c[][]: the constant-buffer form of 0x419.
            if (insn.Bits(24, 8) != 255 || (insn.words[2] & ~0x8ffU) != 0x00010600 ||
                !valid_cbuf() || insn.Bits(62, 2))
                return fail(index, "native funnel shift constant mode");
            emit(0x4c98000000000000ULL | predicate | 250 | (u64{15} << 39) | cbuf());
            emit(base(0x5c48000000000000ULL) | (u64{250} << 8) | (u64{insn.Bits(64, 8)} << 20) |
                 (u64{insn.Bits(75, 1)} << 39));
            break;
        }
        case 0x805: { // CS2R SRZ
            if (insn.Bits(72, 8) != 255 || !insn.Bits(80, 1) || dest >= 249) return fail(index, "native special-register read");
            mov_imm(static_cast<u32>(dest), 0);
            mov_imm(static_cast<u32>(dest) + 1, 0);
            break;
        }
        case 0x306: case 0x906: case 0xd06: { // I2F.F32.U8, round-to-nearest (0xd06: uniform source)
            const u32 source = opcode == 0xd06 ? uniform_register(insn.Bits(32, 8)) : insn.Bits(32, 8);
            if (opcode != 0x906 && (insn.words[2] & ~0x400U) == (opcode == 0xd06 ? 0x08101000U : 0x00101000U) &&
                !insn.Bits(24, 8) && !(insn.words[1] & 0xefffff00U) && source != ~0U) {
                // I2F.F32.U16/S16 Rd, Rs(.H1) and Rd, URs(.H1) (84-85 = 1: 16-bit source,
                // 74: signed as Maxwell's bit 13, RN). Bit 60 picks the high halfword: Maxwell byte
                // selector 2. Bit 61 (no known 16-bit meaning) stays rejected.
                emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{1} << 10) | (u64{insn.Bits(74, 1)} << 13) |
                     (u64{source} << 20) | (u64{insn.Bits(60, 1) * 2} << 41));
                break;
            }
            if (opcode != 0xd06 && (insn.words[2] & ~0xc400U) == 0x00201000 && !insn.Bits(24, 8) &&
                (opcode == 0x906 || !(insn.words[1] & 0xffffff00U))) {
                // I2F.F32.U32/S32 (74: signed, 84-85 = 2: 32-bit source), rounding 78-79 as Maxwell
                // I2F's 39-40, register or (0x906) 32-bit immediate source.
                u32 value = source;
                if (opcode == 0x906) {
                    mov_imm(250, insn.words[1]);
                    value = 250;
                }
                emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{2} << 10) | (u64{insn.Bits(74, 1)} << 13) |
                     (u64{value} << 20) | (u64{insn.Bits(78, 2)} << 39));
                break;
            }
            // Bits 60-61 select the source byte (.B1-.B3) of the register or uniform register form,
            // as Maxwell I2F's byte selector at 41-42.
            if (opcode == 0x906 || insn.words[2] != (opcode == 0xd06 ? 0x08001000U : 0x1000U) ||
                (insn.words[1] & 0xcfffff00U) || source == ~0U)
                return fail(index, "native byte-to-float conversion mode");
            emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{source} << 20) |
                 (u64{insn.Bits(60, 2)} << 41));
            break;
        }
        case 0x245: case 0x845: case 0xa45: case 0xc45: { // I2FP.F32.U32 (0x845: immediate source)
            // Bits 78-79: rounding (RN/RM/RP/RZ), as Maxwell I2F's bits 39-40.
            if ((insn.words[2] & ~(0x400U | 0xc000U | (opcode == 0xc45 ? 0x08000000U : 0U))) != 0x00201000 ||
                (opcode == 0x845 && insn.Bits(24, 8))) return fail(index, "native packed integer-to-float format");
            const u32 src = operand(1, false);
            emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{2} << 10) | (u64{insn.Bits(74, 1)} << 13) | (u64{src} << 20) |
                 (u64{insn.Bits(78, 2)} << 39));
            break;
        }
        case 0x207: case 0x807: case 0xa07:
        case 0x208: case 0x808: case 0xa08: { // Bitwise SEL / FSEL
            // FSEL applies the floating operand modifiers (negate/absolute) before selecting
            // (as in Eglantine/Lichii). Dropping them turned "p ? x : -x" into x: HWAI's
            // octahedral normal decode in deferred lighting (5c65163e4105372b) lost its signs,
            // and two-sided G-buffer shaders wrote un-flipped back-face normals.
            const bool floating = (opcode & 0x1ff) == 0x008;
            const u32 a = operand(0, floating), b = operand(1, floating);
            emit(base(0x5ca0000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
            break;
        }
        case 0x20a: case 0x80a: case 0xa0a: { // FSET.BF, combined with Pc (bits 87-90) by bits 74-75
            if (insn.Bits(64, 8) || insn.Bits(74, 2) == 3 || insn.Bits(81, 6) || insn.Bits(91, 1))
                return fail(index, "native floating comparison output mode");
            const u32 a = operand(0), b = operand(1);
            emit(base(0x5800000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                 (u64{insn.Bits(87, 4)} << 39) | (u64{insn.Bits(74, 2)} << 45) |
                 (u64{insn.Bits(76, 4)} << 48) | (u64{1} << 52) | (u64{insn.Bits(80, 1)} << 55));
            break;
        }
        case 0x434: { // HSETP2 with an immediate half pair: A swizzle 74-75 (neg 72, abs 73),
                      // combine 69-70, H_AND 71, compare 76-79, FTZ 80, P/Q 81-86, input 87-90.
            mov_imm(250, insn.words[1]);
            emit(0x5d20000000000000ULL | predicate | insn.Bits(84, 3) | (u64{insn.Bits(81, 3)} << 3) |
                 (u64{insn.Bits(80, 1)} << 6) | (u64{insn.Bits(24, 8)} << 8) | (u64{250} << 20) |
                 (u64{insn.Bits(76, 4)} << 35) | (u64{insn.Bits(87, 3)} << 39) |
                 (u64{insn.Bits(90, 1)} << 42) | (u64{insn.Bits(72, 1)} << 43) |
                 (u64{insn.Bits(73, 1)} << 44) | (u64{insn.Bits(69, 2)} << 45) |
                 (u64{insn.Bits(74, 2)} << 47) | (u64{insn.Bits(71, 1)} << 49));
            break;
        }
        case 0x234: { // HSETP2 with a register B: B 32-39, swizzle 60-61, abs 62, neg 63 (NAK
                      // encode_alu_reg); A, combine, compare, FTZ, P/Q and input as 0x434.
            // Swizzle 1 is invalid for HSETP2 sources (F32 on Maxwell); H_AND, BF16_V2 (65),
            // combine 3, bit 91 and bits outside the decoded fields are not handled.
            if (insn.Bits(16, 8) || insn.Bits(40, 20) || insn.Bits(60, 2) == 1 || insn.Bits(64, 5) ||
                insn.Bits(69, 2) == 3 || insn.Bits(71, 1) || insn.Bits(74, 2) == 1 || insn.Bits(91, 14))
                return fail(index, "native half comparison mode");
            const auto b = half_source(1);
            emit(0x5d20000000000000ULL | predicate | insn.Bits(84, 3) | (u64{insn.Bits(81, 3)} << 3) |
                 (u64{insn.Bits(80, 1)} << 6) | (u64{insn.Bits(24, 8)} << 8) | (u64{b.reg} << 20) |
                 (u64{b.swizzle} << 28) | (u64{b.absolute} << 30) | (u64{b.negate} << 31) |
                 (u64{insn.Bits(76, 4)} << 35) | (u64{insn.Bits(87, 3)} << 39) |
                 (u64{insn.Bits(90, 1)} << 42) | (u64{insn.Bits(72, 1)} << 43) |
                 (u64{insn.Bits(73, 1)} << 44) | (u64{insn.Bits(69, 2)} << 45) |
                 (u64{insn.Bits(74, 2)} << 47));
            break;
        }
        case 0x435: { // HFMA2.MMA Rd, -RZ, RZ, imm: HFMA2's layout (C immediate 32-63, B 64-71),
                      // NVIDIA's 32-bit constant move. -0 * 0 + c is c in each lane (no FTZ);
                      // NaN lanes are rejected, as the hardware may not keep their payload.
            const auto nan_half = [](u32 half) { return (half & 0x7c00) == 0x7c00 && (half & 0x3ff) != 0; };
            if (insn.Bits(24, 8) != 255 || insn.words[2] != 0x000001ff || insn.Bits(96, 9) ||
                nan_half(insn.words[1] & 0xffff) || nan_half(insn.words[1] >> 16))
                return fail(index, "native tensor-pipe half multiply-add mode");
            mov_imm(static_cast<u32>(dest), insn.words[1]);
            break;
        }
        case 0x20b: case 0x80b: case 0xa0b: case 0xc0b: // FSETP, including a uniform B (|UR| 62, -UR 63)
        case 0x20c: case 0x80c: case 0xa0c: case 0xc0c: { // ISETP
            const bool floating = (opcode & 0x1ff) == 0xb;
            if (!floating && insn.Bits(72, 1)) {
                // ISETP.EX: the high-word step of a 64-bit compare. Equal high words take the
                // low-word result (predicate in bits 68-70, negated by 71), others compare the
                // high words (bit 73 signed); then combine with Pc (87-90) by bits 74-75.
                // Masks in registers: R250 = (a == b), R252 = (a cmp b), R251 = low;
                // R250 = R250 ? R251 : R252 (LUT 0xac); Pd = (R250 != 0) op Pc.
                const u32 compare = insn.Bits(76, 3);
                if (compare == 0 || compare == 7 || insn.Bits(74, 2) > 2 || insn.Bits(84, 3) != 7 ||
                    insn.Bits(64, 4) || insn.Bits(79, 2))
                    return fail(index, "native extended integer comparison");
                const u32 b = operand(1, false), a = operand(0, false);
                const u64 is_signed = insn.Bits(73, 1);
                emit(0x5b50000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{b} << 20) |
                     (u64{7} << 39) | (u64{2} << 49));
                emit(0x5b50000000000000ULL | predicate | 252 | (u64{a} << 8) | (u64{b} << 20) |
                     (u64{7} << 39) | (is_signed << 48) | (u64{compare} << 49));
                emit(0x38a0000000000000ULL | predicate | 251 | (u64{255} << 8) |
                     (u64{0x7ffff} << 20) | (u64{1} << 56) | (u64{insn.Bits(68, 4) ^ 8U} << 39));
                emit(0x5be0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{252} << 20) |
                     (u64{0xac} << 28) | (u64{251} << 39) | (u64{7} << 48));
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{insn.Bits(81, 3)} << 3) |
                     (u64{250} << 8) | (u64{255} << 20) | (u64{insn.Bits(87, 3)} << 39) |
                     (u64{insn.Bits(90, 1)} << 42) | (u64{insn.Bits(74, 2)} << 45) | (u64{5} << 49));
                break;
            }
            const u32 a = operand(0, floating), b = operand(1, floating);
            const u32 combine = insn.Bits(74, 2);
            if (combine > 2) return fail(index, "native predicate combination");
            u64 out = (floating ? 0x5bb0000000000000ULL : 0x5b60000000000000ULL) | predicate |
                insn.Bits(84, 3) | (u64{insn.Bits(81, 3)} << 3) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42) | (u64{combine} << 45);
            if (floating) out |= (u64{insn.Bits(80, 1)} << 47) | (u64{insn.Bits(76, 4)} << 48);
            else out |= (u64{insn.Bits(73, 1)} << 48) | (u64{insn.Bits(76, 3)} << 49);
            emit(out);
            break;
        }
        case 0x212: case 0x812: case 0xa12: case 0xc12: { // LOP3.LUT, any table, optional P = (result != 0)
            const u32 lut = insn.Bits(72, 8);
            const u32 pred_target = insn.Bits(81, 3);
            if (opcode == 0x212 && !rewritten_uniform && insn.Bits(87, 4) != 15 && pred_target != 7 &&
                !insn.Bits(80, 1) && !insn.Bits(40, 24)) {
                // LOP3.LUT Pd, Rd, a, b, c, lut, Pq without .PAND: Pd = (result != 0) OR Pq, the
                // second step of a 64-bit nonzero test (Pq = !PT is the plain form below).
                const u32 a = insn.Bits(24, 8), b = insn.Bits(32, 8), c = insn.Bits(64, 8);
                emit(0x5be0000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{b} << 20) |
                     (u64{lut} << 28) | (u64{c} << 39) | (u64{7} << 48));
                if (dest != 255) emit(base(0x5c98000000000000ULL) | (u64{250} << 20) | (u64{15} << 39));
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{pred_target} << 3) | (u64{250} << 8) |
                     (u64{255} << 20) | (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42) |
                     (u64{1} << 45) | (u64{5} << 49));
                break;
            }
            if (insn.Bits(80, 1) || insn.Bits(87, 4) != 15 ||
                (opcode == 0x212 && insn.Bits(40, 24)) || (opcode == 0xa12 && insn.Bits(62, 2)))
                return fail(index, "native LUT/predicate mode");
            u32 b = 255;
            if (opcode == 0x212) {
                b = insn.Bits(32, 8);
            } else if (opcode == 0x812) {
                mov_imm(251, insn.words[1]);
                b = 251;
            } else if (opcode == 0xc12) {
                b = uniform_register(insn.Bits(32, 8));
                if (b == ~0U) return fail(index, "native uniform logic source");
            } else {
                b = operand(1, false);
            }
            const u32 a = insn.Bits(24, 8), c = insn.Bits(64, 8);
            emit(0x5be0000000000000ULL | predicate | dest | (u64{a} << 8) | (u64{b} << 20) |
                 (u64{lut} << 28) | (u64{pred_target == 7 ? 0U : 3U} << 36) | (u64{c} << 39) |
                 (u64{pred_target} << 48));
            break;
        }
        case 0x213: case 0xa13: case 0xc13: { // IABS: max(x, 0 - x), signed.
            if (insn.words[2] & ~(opcode == 0xc13 ? 0x08000000U : 0U))
                return fail(index, "native integer absolute mode");
            const u32 source = operand(1, false);
            emit(0x5c10000000000000ULL | predicate | 250 | (u64{255} << 8) | (u64{source} << 20) |
                 (u64{1} << 48));
            emit(base(0x5c20000000000000ULL) | (u64{source} << 8) | (u64{250} << 20) |
                 (u64{15} << 39) | (u64{1} << 48));
            break;
        }
        case 0x217: case 0x817: case 0xa17: case 0xc17: { // IMNMX register, immediate, cbuf or uniform operand.
            if ((insn.words[2] & ~(0x04000200U | (opcode == 0xc17 ? 0x08000000U : 0U))) != 0x03800000 ||
                (opcode != 0xa17 && insn.words[1] > (opcode == 0x817 ? 0x7ffffU : 255U)) ||
                (opcode == 0xa17 && (!valid_cbuf() || insn.Bits(62, 2))))
                return fail(index, "native integer min/max mode");
            const u32 source = opcode == 0xc17   ? uniform_register(insn.words[1])
                               : opcode == 0xa17 ? operand(1, false)
                                                 : insn.words[1];
            if (source == ~0U) return fail(index, "native uniform min/max operand");
            emit(base(opcode == 0x817 ? 0x3820000000000000ULL : 0x5c20000000000000ULL) |
                 (u64{insn.Bits(24, 8)} << 8) | (u64{source} << 20) |
                 (u64{insn.Bits(87, 4)} << 39) | (u64{insn.Bits(73, 1)} << 48));
            break;
        }
        case 0x819: { // SHF.HI, immediate shift
            const bool is_signed = insn.Bits(73, 2) == 2;
            if (!insn.Bits(80, 1) && !insn.Bits(76, 1) && insn.Bits(64, 8) == 255 &&
                insn.Bits(73, 2) == 3 && !insn.Bits(75, 1) && insn.words[1] <= 32) {
                emit(base(0x3848000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{insn.words[1]} << 20));
                break;
            }
            if (!insn.Bits(80, 1) && insn.Bits(76, 1) && insn.Bits(64, 8) == 255 &&
                (insn.Bits(73, 2) == 3 || is_signed) && !insn.Bits(75, 1) && insn.words[1] <= 32) {
                // SHF.R low word with a zero high word: a logical right shift.
                emit(base(0x3828000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{insn.words[1]} << 20));
                break;
            }
            if ((insn.words[2] & ~0xffU) == 0x00010200 && insn.words[1] <= 32) {
                // SHF.L.U64.HI Rd, Ra, s, Rc: high word of (Rc:Ra) << s. For s <= 32 the U64
                // clamp (64) and the U32 clamp (32) give the same word, so this is the U32 form.
                emit(base(0x36f8000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(64, 8)} << 39) | (u64{insn.words[1]} << 20));
                break;
            }
            if ((insn.Bits(73, 2) != 3 && !is_signed) || insn.Bits(75, 1) || !insn.Bits(80, 1) || insn.words[1] > 32) return fail(index, "native funnel-shift mode");
            const u32 low = insn.Bits(24, 8), high = insn.Bits(64, 8), shift = insn.words[1];
            if (insn.Bits(76, 1)) {
                if (low != 255) return fail(index, "native nonzero low right-shift word");
                emit(base(0x3828000000000000ULL) | (u64{high} << 8) | (u64{shift} << 20) |
                     (u64{is_signed} << 48));
            } else {
                if (is_signed) return fail(index, "native signed left funnel shift");
                emit(base(0x36f8000000000000ULL) | (u64{low} << 8) | (u64{high} << 39) | (u64{shift} << 20));
            }
            break;
        }
        case 0xa11: { // LEA with constant base and carry predicate.
            if (insn.Bits(80, 1)) {
                // LEA.HI Rd, Ra, c[][], RZ, s: c + high word of (Ra << s), as 0x211's high form;
                // bit 73 sign-extends Ra.
                if (insn.Bits(72, 1) || insn.Bits(74, 1) || insn.Bits(81, 3) != 7 ||
                    insn.Bits(64, 8) != 255 || insn.Bits(62, 2))
                    return fail(index, "native constant high effective address mode");
                const u32 b = operand(1, false);
                const u64 shift = (u64{32 - insn.Bits(75, 5)} << 20) | (u64{insn.Bits(73, 1)} << 48);
                emit(0x3828000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) | shift);
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{b} << 20));
                break;
            }
            if (insn.Bits(81, 3) == 7 && !insn.Bits(72, 3) && insn.Bits(64, 8) == 255 && !insn.Bits(62, 2)) {
                // LEA Rd, Ra, c[][], s without a carry-out: ISCADD.
                const u32 b = operand(1, false);
                emit(base(0x5c18000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                    (u64{b} << 20) | (u64{insn.Bits(75, 5)} << 39));
                break;
            }
            if (insn.Bits(72, 1) || insn.Bits(74, 1) || insn.Bits(81, 3) == 7 || insn.Bits(62, 2))
                return fail(index, "native constant LEA carry mode");
            const u32 b = operand(1, false);
            emit(base(0x5c18000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                (u64{b} << 20) | (u64{insn.Bits(75, 5)} << 39));
            emit(0x5b60000000000000ULL | predicate | 7 | (u64{insn.Bits(81, 3)} << 3) |
                (dest << 8) | (u64{b} << 20) | (u64{7} << 39) | (u64{1} << 49));
            break;
        }
        case 0xc11: { // LEA with a uniform-register B (bits 32-37): low word, optional carry-out.
            // LEA Rd, Pc, Ra, URb, s = (Ra << s) + URb, the carry-out of that 32-bit add in Pc
            // (as 0xa11); LEA.HI Rd, Ra, URb, RZ, s = URb + high word of (Ra << s) (as 0x211's
            // high form, bit 73 sign-extends Ra). Without .X (bit 74), bits 84-86 are 0 and the
            // carry-in field 87-90 is !PT.
            const u32 b = insn.words[1] < 64 ? uniform_register(insn.words[1]) : ~0U;
            const bool high = insn.Bits(80, 1) != 0;
            const u32 carry_out = insn.Bits(81, 3);
            if (insn.Bits(74, 1)) {
                // LEA.HI.X Rd, Ra, URb, Rc, s, Pc = URb + high word of ((Rc:Ra) << s) + Pc (carry-in
                // in bits 87-89, negate 90): Maxwell SHF.L, then the carry as 0/1 (as 0x210's .X).
                if (b == ~0U || !high || carry_out != 7 || insn.Bits(72, 2) ||
                    (insn.words[2] & 0xf8700000U) != 0x08000000U)
                    return fail(index, "native uniform effective address mode");
                emit(0x36f8000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(75, 5)} << 20) | (u64{insn.Bits(64, 8)} << 39));
                mov_imm(252, 1);
                emit(0x5ca0000000000000ULL | predicate | 251 | (u64{252} << 8) | (u64{255} << 20) |
                     (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{251} << 20));
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{b} << 20));
                break;
            }
            if (b == ~0U || insn.Bits(64, 8) != 255 || insn.Bits(72, 1) || insn.Bits(74, 1) ||
                (insn.words[2] & 0xfff00000U) != 0x0f800000U || (!high && insn.Bits(73, 1)) ||
                (high && carry_out != 7) || (carry_out != 7 && dest == 255))
                return fail(index, "native uniform effective address mode");
            if (high) {
                const u64 shift = (u64{32 - insn.Bits(75, 5)} << 20) | (u64{insn.Bits(73, 1)} << 48);
                emit(0x3828000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) | shift);
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{b} << 20));
                break;
            }
            emit(base(0x5c18000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{b} << 20) | (u64{insn.Bits(75, 5)} << 39));
            if (carry_out != 7) {
                // carry = ((Ra << s) + URb) < URb, unsigned
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{carry_out} << 3) |
                     (dest << 8) | (u64{b} << 20) | (u64{7} << 39) | (u64{1} << 49));
            }
            break;
        }
        case 0x211: case 0x811: { // LEA low word, no carry result
            if (opcode == 0x211 && insn.Bits(80, 1) && insn.Bits(74, 1)) {
                // LEA.HI.X Rd, Ra, Rb, Rc, s, Pc: Rb + Pc + high word of (Rc:Ra) << s, the
                // high-word step after a LEA with a carry-out. With Rc = RZ, bit 73 makes the
                // high word the sign of Ra (SX32), as the 0x211 high form below.
                const u32 a = insn.Bits(24, 8), b = insn.Bits(32, 8), c = insn.Bits(64, 8);
                const u32 shift = insn.Bits(75, 5), carry = insn.Bits(87, 4);
                if (rewritten_uniform || insn.Bits(72, 1) || insn.Bits(81, 3) != 7 || insn.Bits(62, 2) ||
                    (c != 255 && insn.Bits(73, 1)) || (c == 255 && shift == 0))
                    return fail(index, "native high effective address mode");
                if (c == 255) {
                    emit(0x3828000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{32 - shift} << 20) |
                         (u64{insn.Bits(73, 1)} << 48));
                } else {
                    emit(0x36f8000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{shift} << 20) |
                         (u64{c} << 39));
                }
                if (carry == 15) {
                    emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{b} << 20));
                } else {
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{b} << 20));
                    emit(0x38a0000000000000ULL | predicate | 251 | (u64{255} << 8) | (u64{1} << 20) |
                         (u64{carry ^ 8U} << 39));
                    emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{251} << 20));
                }
                break;
            }
            if (!insn.Bits(80, 1) && insn.Bits(81, 3) != 7) {
                // LEA Rd, Pc, Ra, Rb, s: Rd = (Ra << s) + Rb with the carry-out in Pc (UPn in R239
                // bit 6-n for ULEA), the low-word step of a 64-bit address.
                const u32 carry = insn.Bits(81, 3);
                if (insn.Bits(72, 3) || insn.Bits(64, 8) != 255 || (opcode == 0x211 && insn.Bits(62, 2)) ||
                    (rewritten_uniform && carry > 1))
                    return fail(index, "native LEA carry/high mode");
                u32 b = insn.Bits(32, 8);
                if (opcode == 0x811) { mov_imm(251, insn.words[1]); b = 251; }
                emit(0x3848000000000000ULL | predicate | 252 | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(75, 5)} << 20));
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{252} << 8) | (u64{b} << 20));
                if (dest != 255) emit(base(0x5c98000000000000ULL) | (u64{250} << 20) | (u64{15} << 39));
                const u32 target = rewritten_uniform ? 6 - carry : carry;
                const u32 mask = 1U << target;
                if (rewritten_uniform) save_predicate(mask);
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{target} << 3) | (u64{250} << 8) |
                     (u64{252} << 20) | (u64{7} << 39) | (u64{1} << 49));
                if (rewritten_uniform) {
                    emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
                    restore_predicate(mask);
                }
                break;
            }
            if (insn.Bits(80, 1)) {
                // LEA.HI Rd, Ra, Rb (or an immediate, 0x811), RZ, s: Rb + high word of (Ra << s);
                // bit 73 sign-extends Ra.
                if (insn.Bits(72, 1) || insn.Bits(74, 1) || insn.Bits(81, 3) != 7 ||
                    insn.Bits(64, 8) != 255 || (opcode == 0x211 && insn.Bits(62, 2)))
                    return fail(index, "native high effective address mode");
                u32 b = insn.Bits(32, 8);
                if (opcode == 0x811) {
                    b = insn.words[1] ? 251U : 255U;
                    if (b == 251) mov_imm(251, insn.words[1]);
                }
                const u64 shift = (u64{32 - insn.Bits(75, 5)} << 20) | (u64{insn.Bits(73, 1)} << 48);
                if (b == 255) {
                    emit(base(0x3828000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | shift);
                } else {
                    emit(0x3828000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) | shift);
                    emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{b} << 20));
                }
                break;
            }
            if (insn.Bits(72, 1) || insn.Bits(74, 1) || insn.Bits(80, 1) || insn.Bits(81, 3) != 7 ||
                (opcode == 0x211 && insn.Bits(62, 1))) return fail(index, "native LEA carry/high mode");
            // 0x211 bit 63 negates Rb (LEA Rd, Ra, -Rb, s), as Maxwell ISCADD's bit 48.
            u32 b = insn.Bits(32, 8);
            if (opcode == 0x811) { mov_imm(251, insn.words[1]); b = 251; }
            emit(base(0x5c18000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{b} << 20) | (u64{insn.Bits(75, 5)} << 39) |
                 (u64{opcode == 0x211 ? insn.Bits(63, 1) : 0U} << 48));
            break;
        }
        case 0x210: case 0x810: case 0xa10: case 0xc10: { // IADD3 with zero third source and no carry outputs
            // 64-bit adds: IADD3 with a carry-out predicate (bits 81-83), then IADD3.X (bit 74) with
            // a carry-in predicate (bits 87-89, negate 90). Uniform forms keep UPn in R239 bit 6-n.
            // general_add (only reached where the forms below reject): non-uniform a + b + c without
            // source modifiers. .X adds the carry-ins Pc0 (87-90) and Pc1 (77-80) and has no
            // carry-outs; otherwise, as Mesa NAK's OpIAdd3 (whose fold its hw_tests check against
            // the GPU), P(81-83) = sum >= 2^32 and P(84-86) = sum >= 2^33 of the full a + b + c.
            // With c1 = carry of a + b and c2 = carry of (a + b) + c, that is c1 | c2 and c1 & c2.
            const auto general_add = [&]() -> bool {
                const bool extended = insn.Bits(74, 1) != 0;
                const u32 out0 = insn.Bits(81, 3), out1 = insn.Bits(84, 3);
                const u32 guard = native_predicate & 7;
                if (rewritten_uniform || opcode == 0xc10 || insn.Bits(72, 2) || insn.Bits(75, 2) ||
                    insn.Bits(91, 1) || (opcode == 0x210 && (insn.words[1] & ~0xffU)) ||
                    (opcode == 0xa10 && (!valid_cbuf() || insn.Bits(62, 2))))
                    return false;
                if (extended ? (out0 != 7 || out1 != 7)
                             : (insn.Bits(77, 4) != 15 || insn.Bits(87, 4) != 15 || (out0 == 7 && out1 == 7) ||
                                out0 == out1 || (guard != 7 && (guard == out0 || guard == out1))))
                    return false;
                const u32 a = insn.Bits(24, 8), b = operand(1, false), c = insn.Bits(64, 8);
                const auto add = [&](u32 target, u32 x, u32 y) {
                    emit(0x5c10000000000000ULL | predicate | target | (u64{x} << 8) | (u64{y} << 20));
                };
                // ISETP.LT.U32 P(target) = R250 < source, combined (op 0 AND, 1 OR) with P(accumulate).
                const auto carry_flag = [&](u32 target, u32 source, u32 accumulate, u32 op) {
                    emit(0x5b60000000000000ULL | predicate | 7 | (u64{target} << 3) | (u64{250} << 8) |
                         (u64{source} << 20) | (u64{accumulate} << 39) | (u64{op} << 45) | (u64{1} << 49));
                };
                if (!extended) {
                    // c1 is held in an output predicate (out0, else out1) until c2 is known; out1
                    // is written before out0 so it still reads c1. The guard is neither output.
                    const u32 hold = out0 != 7 ? out0 : out1;
                    add(250, a, b);
                    carry_flag(hold, b, 7, 0);
                    if (c != 255) add(250, 250, c);
                    if (out1 != 7) carry_flag(out1, c, hold, 0);
                    if (out0 != 7 && c != 255) carry_flag(out0, c, hold, 1);
                    if (dest != 255) emit(base(0x5c98000000000000ULL) | (u64{250} << 20) | (u64{15} << 39));
                    return true;
                }
                const u32 carries[2]{insn.Bits(87, 4), insn.Bits(77, 4)};
                const u32 steps = 1 + (c != 255) + (carries[0] != 15) + (carries[1] != 15);
                u32 step = 0;
                const auto next = [&] { return ++step == steps ? static_cast<u32>(dest) : 250U; };
                add(next(), a, b);
                if (c != 255) add(next(), 250, c);
                for (const u32 carry : carries) {
                    if (carry == 15) continue;
                    emit(0x38a0000000000000ULL | predicate | 251 | (u64{255} << 8) | (u64{1} << 20) |
                         (u64{carry ^ 8U} << 39));
                    add(next(), 250, 251);
                }
                return true;
            };
            if ((opcode == 0x210 || opcode == 0x810) && insn.Bits(64, 8) == 255 && !insn.Bits(73, 1) &&
                !insn.Bits(75, 2) && insn.Bits(84, 3) == 7 && insn.Bits(77, 4) == 15 && !insn.Bits(72, 1) &&
                (insn.Bits(81, 3) != 7 || insn.Bits(74, 1))) {
                const bool carry_in = insn.Bits(74, 1) != 0;
                const u32 carry_out = insn.Bits(81, 3);
                const u32 a = insn.Bits(24, 8);
                // The carry forms save P(6-n) in R253, which a swapped-in UPn guard already uses.
                if (uniform_guard) return fail(index, "native uniform arithmetic predication");
                if ((carry_in && carry_out != 7) || (opcode == 0x210 && (insn.words[1] & ~0xffU)) ||
                    (rewritten_uniform && carry_out != 7 && carry_out > 1) ||
                    (rewritten_uniform && carry_in && insn.Bits(87, 3) > 1 && insn.Bits(87, 3) != 7))
                    return fail(index, "native integer carry mode");
                if (opcode == 0x810) {
                    emit(0x1c00000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{insn.words[1]} << 20));
                } else {
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{insn.Bits(32, 8)} << 20));
                }
                const auto save = [&](u32 mask) { emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20)); };
                const auto restore = [&](u32 mask) { emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20)); };
                if (!carry_in) {
                    // carry = (a + b) < a, unsigned
                    const u32 target = rewritten_uniform ? 6 - carry_out : carry_out;
                    const u32 mask = 1U << target;
                    if (rewritten_uniform) save(mask);
                    emit(0x5b60000000000000ULL | predicate | 7 | (u64{target} << 3) | (u64{250} << 8) |
                         (u64{a} << 20) | (u64{7} << 39) | (u64{1} << 49));
                    if (rewritten_uniform) {
                        emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
                        restore(mask);
                    }
                    emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{250} << 20));
                    break;
                }
                const u32 condition = insn.Bits(87, 3);
                const u32 source = rewritten_uniform && condition != 7 ? 6 - condition : condition;
                const u32 mask = 1U << source;
                mov_imm(252, 1);
                if (rewritten_uniform && condition != 7) {
                    save(mask);
                    emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{mask} << 20));
                }
                emit(0x5ca0000000000000ULL | predicate | 251 | (u64{252} << 8) | (u64{255} << 20) |
                     (u64{source} << 39) | (u64{insn.Bits(90, 1)} << 42));
                if (rewritten_uniform && condition != 7) restore(mask);
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{251} << 20));
                break;
            }
            if (opcode == 0xa10 && insn.Bits(81, 3) != 7) {
                if (insn.Bits(64, 8) != 255 || insn.Bits(73, 4) || insn.Bits(84, 7) != 0x7f ||
                    insn.Bits(77, 4) != 15 || insn.Bits(63, 1) || insn.Bits(72, 1)) {
                    if (general_add()) break;
                    return fail(index, "native constant add carry mode");
                }
                const u32 b = operand(1, false);
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{b} << 20));
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{insn.Bits(81, 3)} << 3) |
                    (dest << 8) | (u64{b} << 20) | (u64{7} << 39) | (u64{1} << 49));
                break;
            }
            if (opcode == 0xa10 && insn.Bits(74, 1)) {
                if (insn.Bits(24, 8) != 255 || (insn.words[2] & ~0x07800000U) != 0x007fe4ffU) {
                    if (general_add()) break;
                    return fail(index, "native extended constant add mode");
                }
                const u32 b = operand(1, false);
                mov_imm(250, 1);
                emit(0x5ca0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                    (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                emit(base(0x5c10000000000000ULL) | (u64{b} << 8) | (u64{250} << 20));
                break;
            }
            // Third source, optionally negated (bit 75; 63 and 72 negate b and a).
            if (insn.Bits(64, 8) != 255 && (insn.Bits(73, 4) & ~4U) == 0 && insn.Bits(77, 7) == 0x7f && insn.Bits(84, 7) == 0x7f) {
                const u32 b = operand(1, false);
                // 0x810: bits 32-63 are the whole immediate (bit 63 is its sign, already loaded),
                // not a negate flag. Treating it as neg_b turned Ra - 126 - Rc into Ra + 126 - Rc
                // in HWAI's particle-emitter compute shaders (sphere-radius cbrt).
                const u64 neg_b = opcode == 0x810 ? 0 : u64{insn.Bits(63, 1)};
                if (neg_b && insn.Bits(72, 1)) {
                    // Maxwell IADD with both negates is .PO (-a + b + 1), not -a - b.
                    return fail(index, "native integer add with both sources negated");
                }
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) |
                    (u64{b} << 20) | (neg_b << 48) | (u64{insn.Bits(72, 1)} << 49));
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{insn.Bits(64, 8)} << 20) |
                     (u64{insn.Bits(75, 1)} << 48));
                break;
            }
            if (insn.Bits(64, 8) != 255 || insn.Bits(73, 4) || insn.Bits(77, 7) != 0x7f || insn.Bits(84, 7) != 0x7f) {
                if (general_add()) break;
                return fail(index, "native integer carry/third source");
            }
            if (opcode == 0xa10 || opcode == 0xc10) {
                if (opcode == 0xc10 && (insn.words[1] & ~0x8000003fU)) return fail(index, "native uniform add operand");
                const u32 b = operand(1, false);
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{b} << 20) |
                    (u64{insn.Bits(63, 1)} << 48) | (u64{insn.Bits(72, 1)} << 49));
            } else if (opcode == 0x210) {
                if (insn.words[1] & ~0x800000ffU) return fail(index, "native integer add register modifiers");
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(63, 1)} << 48) | (u64{insn.Bits(72, 1)} << 49));
            } else {
                emit(base(0x1c00000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{insn.words[1]} << 20) | (u64{insn.Bits(72, 1)} << 56));
            }
            break;
        }
        case 0x949: case 0x958: { // Captured BRX / BRXU with absolute native-PC table values
            if (insn.Bits(12, 4) != 7 || (insn.words[2] & ~0x3ffffU) != (opcode == 0x958 ? 0x0b800000U : 0x03800000U))
                return fail(index, "native computed branch mode");
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            if (s64{index} * 16 + 16 + relative != 0)
                return fail(index, "native computed branch relative base");
            const s64 offset = -s64{address_of(static_cast<u32>(result.instructions.size()))} - 8;
            if (offset < -0x800000) return fail(index, "native computed branch range");
            const u32 source = opcode == 0x958 ? uniform_register(insn.Bits(24, 8)) : insn.Bits(24, 8);
            if (source == ~0U) return fail(index, "native uniform computed branch source");
            emit(0xe25000000000000fULL | predicate | (u64{source} << 8) |
                 ((static_cast<u64>(offset) & 0xffffff) << 20));
            computed_branch = true;
            break;
        }
        case 0x944: { // Verified CALL.REL.NOINC with R55 holding the return PC
            // A predicated NOINC forward transfer can terminate a loop at its
            // reconvergence point without a return-link setup or matching RET.
            if (native_predicate != 7 && insn.words[1] == 16 && insn.words[2] == 0x03c00000 &&
                index + 2 < code.size() && code[index + 1].Bits(0, 12) == 0x947) {
                relocations.push_back({static_cast<u32>(result.instructions.size()), index + 2});
                emit(0xe24000000000000fULL | predicate);
                break;
            }
            if (main_exited || index == 0 || insn.words[0] != 0x7944 ||
                (insn.words[2] & ~0x3ffffU) != 0x03c00000 ||
                (code[index - 1].words[0] & ~0x00ff0000U) != 0x00007802 ||
                code[index - 1].Bits(16, 8) >= 250 ||
                code[index - 1].words[1] != (index + 1) * 16 ||
                code[index - 1].words[2] != 0xf00)
                return fail(index, "native call linkage mode");
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16) return fail(index, "native call destination alignment");
            const u32 link = code[index - 1].Bits(16, 8);
            for (size_t call = 0; call < call_targets.size(); ++call) {
                if (call_targets[call] == target / 16 && call_links[call] != link)
                    return fail(index, "native inconsistent call linkage");
            }
            call_targets.push_back(static_cast<u32>(target / 16));
            call_links.push_back(link);
            relocations.push_back({static_cast<u32>(result.instructions.size()), target / 16});
            emit(0xe26000000007000fULL);
            break;
        }
        case 0x950: { // Verified RET.REL.NODEC epilogue restoring the call link
            const u32 return_reg = insn.Bits(24, 8);
            // The return register is the link itself, or the last write to it in this function
            // is a copy of the link (MOV or IMAD.MOV.U32 Rd, RZ, RZ, Rlink).
            const auto copies_link = [&](const Instruction& prior) {
                if (prior.Bits(16, 8) != return_reg || prior.Bits(12, 4) != 7) return false;
                if ((prior.words[0] & 0xff00ffffU) == 0x00007202U)
                    return prior.words[1] == *active_link && prior.words[2] == 0xf00;
                if ((prior.words[0] & 0xff00ffffU) == 0xff007224U)
                    return prior.words[1] == 0xff && prior.words[2] == (0x078e0000U | *active_link);
                return false;
            };
            bool linked = active_link && return_reg == *active_link;
            for (u32 prior = index; active_link && !linked && prior-- > active_entry;) {
                if (copies_link(code[prior])) linked = true;
                else if (code[prior].Bits(16, 8) == return_reg) break;
            }
            if (!main_exited || !linked ||
                (insn.words[0] & 0x00ffffffU) != 0x00007950 || return_reg >= 249 ||
                (insn.words[2] & ~0x3ffffU) != 0x03c00000 || !barrier_tokens.empty())
                return fail(index, "native return linkage mode");
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            if (s64{index} * 16 + 16 + relative != 0)
                return fail(index, "native return relative base");
            emit(0xe32000000007000fULL);
            bool complete = !computed_branch;
            for (u32 target : call_targets) complete &= target <= index;
            if (complete) return finish(index);
            break;
        }
        case 0x945: case 0x941: case 0x942: case 0x948: // (+ WARPSYNC)
            // BSSY / BSYNC / BREAK only manage warp reconvergence: BSYNC waits for the other
            // threads and falls through, BREAK drops the thread from that wait. Eden runs threads
            // independently, so all three are no-ops (a SYNC would wrongly jump, and a branch
            // out of a lowered SSY region would leave its token on Eden's flow stack).
            break;
        case 0x947: { // BRA
            u64 branch_predicate = predicate;
            if (opcode == 0x947 && insn.Bits(87, 4) != 7) {
                if (insn.Bits(12, 4) != 7) {
                    relocations.push_back({static_cast<u32>(result.instructions.size()), index + 1});
                    emit(0xe24000000000000fULL | (u64{insn.Bits(12, 4) ^ 8U} << 16));
                }
                branch_predicate = u64{insn.Bits(87, 4)} << 16;
            } else if (insn.Bits(87, 4) != 7) {
                return fail(index, "native extra branch condition");
            }
            // Bit 32 is .U (warp-uniform hint, NAK OpBra); the word offset is bits 34-81.
            const u64 raw = (insn.words[1] & ~1U) | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16) return fail(index, "native branch destination alignment");
            relocations.push_back({static_cast<u32>(result.instructions.size()), target / 16});
            emit(0xe24000000000000fULL | branch_predicate);
            if (computed_branch && target == s64{index} * 16 && branch_predicate == (u64{7} << 16))
                return finish(index); // the closing self-branch
            break;
        }
        case 0x94d: { // EXIT
            if (insn.Bits(84, 2)) {
                return fail(index, "native conditional/special EXIT");
            }
            if (insn.Bits(87, 4) != 7) {
                // @Pg EXIT Pc: exits when both hold; skip over a Pc-predicated EXIT unless Pg.
                if (native_predicate != 7) {
                    relocations.push_back({static_cast<u32>(result.instructions.size()), index + 1});
                    emit(0xe24000000000000fULL | (u64{native_predicate ^ 8U} << 16));
                }
                emit(0xe30000000000000fULL | (u64{insn.Bits(87, 4)} << 16));
                break;
            }
            result.instructions.push_back(0xe30000000000000fULL | predicate);
            if (insn.Bits(12, 4) != 7) break;
            if (!barrier_tokens.empty()) return fail(index, "native unclosed reconvergence scope");
            if (call_targets.empty()) {
                const bool pending_branch = std::any_of(relocations.begin(), relocations.end(),
                    [&](const Relocation& relocation) { return relocation.target > index; });
                if (!pending_branch && !computed_branch) return finish(index);
                break;
            }
            if (main_exited) {
                // A library function may end the shader instead of returning (Eden inlines
                // calls, so this is an ordinary exit at that point).
                bool complete = std::none_of(relocations.begin(), relocations.end(),
                    [&](const Relocation& relocation) { return relocation.target > index; });
                for (u32 target : call_targets) complete &= target <= index;
                if (complete && !computed_branch) return finish(index);
                break;
            }
            for (u32 target : call_targets) {
                if (target <= index) return fail(index, "native call target overlaps main program");
            }
            main_exited = true;
            break;
        }
        default:
            return fail(index, "unsupported native opcode");
        }
        if (!operand_error.empty()) return fail(index, operand_error);
        if (output_mask) {
            emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{output_mask} << 20));
            emit(0x38f0000000070000ULL | (u64{254} << 8) | (u64{output_mask} << 20));
        }
        if (guarded_uniform) restore_predicate(guard_mask);
        for (u32 i = 0; i < home_run_count; ++i) {
            const HomeRun& run = home_runs[i];
            home_transfer(true, run.temporary, run.count, 4 * (run.native - 216));
            home_transfer(false, run.temporary, run.count, HomeSaveOffset + 4 * (run.temporary - home_window));
        }
    }
    return fail(static_cast<u32>(code.size()), main_exited ? "native program lacks captured library return" : "native program has no unconditional EXIT");
}
} // namespace detail

// Programs that only fail because their registers reach the lowering's reserved range are lowered
// again with those native registers kept in local memory; every other program lowers exactly as
// before (same instructions, no local memory bias).
inline LoweredProgram LowerToMaxwell(std::span<const Instruction> code) {
    bool reserved_conflict = false;
    LoweredProgram result = detail::LowerToMaxwellPass(code, false, &reserved_conflict);
    if (!result && reserved_conflict) result = detail::LowerToMaxwellPass(code, true, nullptr);
    return result;
}

} // namespace Shader::Ampere
