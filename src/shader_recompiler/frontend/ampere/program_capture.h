// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace Shader::Ampere {
class ProgramCaptureState {
public:
    enum class Result { Continue, Complete, InvalidTarget };
    static constexpr u32 MaximumInstructions = 65536;

    Result Observe(const Instruction& instruction, u32 index) {
        const Instruction prior = previous;
        previous = instruction;
        const u32 opcode = instruction.Bits(0, 12);
        if (opcode == 0x944 || opcode == 0x947 || opcode == 0x945) {
            u64 raw = instruction.words[1] | (u64{instruction.words[2] & 0x3ffff} << 32);
            // BRA bit 32 is .U (warp-uniform branch, NAK OpBra); its offset is bits 34-81.
            if (opcode == 0x947) raw &= ~u64{1};
            const s64 relative = opcode == 0x945 ? static_cast<s32>(instruction.words[1])
                                                : static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16 || target >= MaximumInstructions * 16) {
                previous = prior; // the caller replaces or rejects this instruction
                return Result::InvalidTarget;
            }
            // CALL.REL.NOINC (bit 86) pushes no return address. Right after MOV Rlink, return_pc
            // it is a linked call (as the lowering's case 0x944 verifies): the subroutine returns
            // with RET.REL.NODEC through Rlink or ends the shader. Otherwise it is a jump.
            const bool noinc_call = opcode == 0x944 && (instruction.words[2] & 0x400000) != 0;
            const bool linked_call = noinc_call && index > 0 && instruction.words[0] == 0x7944 &&
                                     (instruction.words[2] & ~0x3ffffU) == 0x03c00000 &&
                                     (prior.words[0] & ~0x00ff0000U) == 0x00007802 &&
                                     prior.Bits(16, 8) < 250 && prior.words[1] == (index + 1) * 16 &&
                                     prior.words[2] == 0xf00;
            if (opcode == 0x944 && !noinc_call) {
                last_call_target = std::max(last_call_target, static_cast<u32>(target / 16));
            } else if (linked_call) {
                last_linked_target = std::max(last_linked_target, static_cast<u32>(target / 16));
            } else {
                last_forward_target = std::max(last_forward_target, static_cast<u32>(target / 16));
            }
            // The compiler ends every program with an unconditional branch to itself; with a jump
            // table (whose targets live in a constant buffer) that is the only reliable end.
            if (opcode == 0x947 && computed_branch && instruction.Bits(12, 4) == 7 &&
                target == s64{index} * 16)
                return Result::Complete;
        }
        if (opcode == 0x949 || opcode == 0x958) computed_branch = true;
        if (computed_branch) return Result::Continue;
        if (opcode == 0x94d && instruction.Bits(12, 4) == 7 && index >= last_forward_target) {
            // Past every linked subroutine entry an EXIT ends the shader (none: the main EXIT).
            if (!last_call_target && index >= last_linked_target) return Result::Complete;
            main_exited = true;
        }
        if (opcode == 0x950 && instruction.Bits(12, 4) == 7 && main_exited &&
            index >= last_call_target && index >= last_linked_target && index >= last_forward_target)
            return Result::Complete;
        return Result::Continue;
    }

private:
    Instruction previous{};
    u32 last_call_target{};
    u32 last_linked_target{};
    u32 last_forward_target{};
    bool main_exited{};
    bool computed_branch{};
};
} // namespace Shader::Ampere
