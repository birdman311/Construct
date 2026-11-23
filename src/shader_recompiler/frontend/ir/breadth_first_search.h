// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <optional>
#include <type_traits>
#include <queue>
#include <unordered_set>

#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::IR {

template <typename Pred, typename Resolve = std::identity>
auto BreadthFirstSearch(Value value, Pred&& pred, Resolve&& resolve = {})
    -> std::invoke_result_t<Pred, const Inst*> {
    value = resolve(value);
    if (value.IsImmediate()) {
        // Nothing to do with immediates
        return std::nullopt;
    }
    // Breadth-first search visiting the right most arguments first.
    // Visited set is a hash set: a failing search walks the whole SSA graph, and a linear lookup
    // made that quadratic (Switch 2 programs of 20k+ instructions took ~0.2 s per storage buffer
    // lookup). Membership answers, and so the visit order and result, are unchanged.
    std::unordered_set<const Inst*> visited;
    std::queue<const Inst*> queue;
    queue.push(value.InstRecursive());

    while (!queue.empty()) {
        // Pop one instruction from the queue
        const Inst* const inst{queue.front()};
        queue.pop();
        if (const std::optional result = pred(inst)) {
            // This is the instruction we were looking for
            return result;
        }
        // Visit the right most arguments first
        for (size_t arg = inst->NumArgs(); arg--;) {
            const Value arg_value{resolve(inst->Arg(arg))};
            if (arg_value.IsImmediate()) {
                continue;
            }
            // Queue instruction if it hasn't been visited
            const Inst* const arg_inst{arg_value.InstRecursive()};
            if (visited.insert(arg_inst).second) {
                queue.push(arg_inst);
            }
        }
    }
    // SSA tree has been traversed and the result hasn't been found
    return std::nullopt;
}

} // namespace Shader::IR
