// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// @brief Shared F16 result policy for AMDGPU transcendental (TRANS) instructions.

#include "util/data_types.h"

#include <bit>
#include <concepts>
#include <cstdint>

namespace util {

/// @brief Evaluate an F16 transcendental through its F32 core and round the result to half.
/// @details Measured on gfx1201 for RCP, RSQ, SIN, COS, EXP and LOG. Denormal mode bit 0
/// preserves half input subnormals and bit 1 output subnormals; flushing retains the sign.
/// @p core maps the exactly promoted source to an F32 value, which rounds to nearest-even half
/// regardless of guest rounding. FP16_OVFL saturates an infinite result from a non-infinite
/// source, including a division by zero. The returned F32 value is exactly the half result;
/// callers apply output modifiers to it.
inline float amdgpu_trans_f16(float value, uint32_t denorm_mode, bool fp16_ovfl,
                              std::invocable<float> auto core) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((denorm_mode & 1u) == 0 && (bits & 0x7fffffffu) < 0x38800000u)
    bits &= 0x80000000u;
  uint16_t result = f32_to_f16(core(std::bit_cast<float>(bits)));
  if ((denorm_mode & 2u) == 0 && (result & 0x7c00u) == 0)
    result &= 0x8000u;
  if (fp16_ovfl && (result & 0x7fffu) == 0x7c00u && (bits & 0x7fffffffu) != 0x7f800000u)
    result = (result & 0x8000u) | 0x7bffu;
  return f16_to_f32(result);
}

} // namespace util
