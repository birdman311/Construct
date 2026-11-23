// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Shader::Ampere {

// Version of LowerToMaxwell's output. Disk shader caches store the lowered Maxwell code of native
// (Switch 2) shaders, so a cache written by an older lowering keeps running its old translation;
// Switch 2 titles mix this into their cache version, and bumping it rebuilds their caches.
// Bump it whenever the lowering of an already-supported instruction changes.
// 2: Ocarina of Time forms (TXQ mask 4, TEX.B, IMAD.WIDE imm, SHF.R.U64, USHF imm and R.U64, I2F S16,
// S2R SR 19/132), reserved-range registers kept in local memory, I2F.U16 bits 24-31/61 rejected.
// 3: HFMA2 H0_NH1 swaps halves before negating (shadow PCF, FSR EASU); IADD3 immediate with a
// third source no longer treats the immediate sign bit as negate-B (particle emitters).
// 4: FSEL applies its negate/absolute source modifiers.
// 5: STG.64/.128 of RZ with a uniform base + register stores zeros, not the address.
inline constexpr u32 LoweringVersion = 5;

} // namespace Shader::Ampere
