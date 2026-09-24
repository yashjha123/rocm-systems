// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file Shared pseudo-scalar transcendental implementations.
///
/// @details Physical GFX12 V_S_* results match their vector equivalents bit for bit, in every
/// MODE setting and with every VOP3 modifier; transcendental_valu.h describes that pipeline. The
/// F16 forms clear destination bits [31:16]. The remaining helpers in this header round exact F64
/// results for fused operations and apply MODE to ordinary floating-point arithmetic.

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental_valu.h"
#include "util/data_types.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu::pseudo_scalar {

/// @brief Pseudo-scalar transcendental operation implemented by the shared helper.
enum class Operation : uint8_t { EXP2, LOG2, RCP, RSQ, SQRT };

namespace detail {

enum class ResultProvenance : uint8_t { VALUE, FINITE_OVERFLOW, FINITE_UNDERFLOW, INVALID_DOMAIN };

struct EvaluationResult {
  double value;
  ResultProvenance provenance;
};

inline float flush_input_f32(float value, uint32_t denorm_mode) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((denorm_mode & 1u) == 0 && (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0)
    return std::copysign(0.0f, value);
  return value;
}

inline float flush_input_f16(float value, uint32_t denorm_mode) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if ((denorm_mode & 1u) == 0 && magnitude != 0 && magnitude < 0x38800000u)
    return std::bit_cast<float>(bits & 0x80000000u);
  return value;
}

inline float quiet_nan(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  // Test signaling NaNs specifically: a general NaN check can become a host
  // floating-point comparison that raises FE_INVALID before the quieting step.
  if ((bits & 0x7fc00000u) == 0x7f800000u && (bits & 0x003fffffu) != 0)
    bits |= 0x00400000u;
  return std::bit_cast<float>(bits);
}

inline EvaluationResult apply_output_modifiers(EvaluationResult result, uint32_t omod, bool clamp) {
  const double unmodified_value = result.value;
  if (omod == 1)
    result.value *= 2.0;
  else if (omod == 2)
    result.value *= 4.0;
  else if (omod == 3)
    result.value *= 0.5;

  if (result.provenance == ResultProvenance::VALUE && std::isfinite(unmodified_value) &&
      unmodified_value != 0.0) {
    if (std::isinf(result.value))
      result.provenance = ResultProvenance::FINITE_OVERFLOW;
    else if (result.value == 0.0)
      result.provenance = ResultProvenance::FINITE_UNDERFLOW;
  }

  if (omod != 0 && result.value == 0.0 && result.provenance == ResultProvenance::VALUE)
    result.value = 0.0;

  if (clamp) {
    const bool negative_underflow =
        result.provenance == ResultProvenance::FINITE_UNDERFLOW && std::signbit(result.value);
    if (std::isnan(result.value) || result.value < 0.0 || negative_underflow) {
      result = {0.0, ResultProvenance::VALUE};
    } else if (result.value > 1.0 || result.provenance == ResultProvenance::FINITE_OVERFLOW) {
      result = {1.0, ResultProvenance::VALUE};
    } else if (result.value == 0.0 && result.provenance == ResultProvenance::VALUE) {
      result.value = 0.0;
    }
  }
  return result;
}

inline uint16_t next_up_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0x7c00u)
    return value;
  if (value == 0xfc00u)
    return 0xfbffu;
  if ((value & 0x7fffu) == 0)
    return 0x0001u;
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value - 1u : value + 1u);
}

inline uint16_t next_down_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0xfc00u)
    return value;
  if (value == 0x7c00u)
    return 0x7bffu;
  if ((value & 0x7fffu) == 0)
    return 0x8001u;
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value + 1u : value - 1u);
}

inline uint16_t saturated_f16(EvaluationResult result, uint32_t round_mode, bool fp16_ovfl) {
  const bool negative = std::signbit(result.value);
  const uint16_t sign = negative ? 0x8000u : 0;
  if (result.provenance == ResultProvenance::FINITE_UNDERFLOW) {
    const bool rounds_to_minimum =
        (!negative && (round_mode & 3u) == 1) || (negative && (round_mode & 3u) == 2);
    return static_cast<uint16_t>(sign | (rounds_to_minimum ? 1u : 0u));
  }

  const bool rounds_to_infinity = (round_mode & 3u) == 0 || (!negative && (round_mode & 3u) == 1) ||
                                  (negative && (round_mode & 3u) == 2);
  if (fp16_ovfl || !rounds_to_infinity)
    return static_cast<uint16_t>(sign | 0x7bffu);
  return static_cast<uint16_t>(sign | 0x7c00u);
}

inline uint16_t round_f64_to_f16(EvaluationResult result, uint32_t round_mode, bool fp16_ovfl) {
  if (result.provenance == ResultProvenance::INVALID_DOMAIN)
    return 0xfe00u;
  if (result.provenance != ResultProvenance::VALUE)
    return saturated_f16(result, round_mode, fp16_ovfl);

  const double value = result.value;
  if (std::isnan(value) || std::isinf(value))
    return util::f32_to_f16(static_cast<float>(value));

  constexpr double MAX_F16 = 65504.0;
  constexpr double RNE_OVERFLOW_THRESHOLD = 65520.0;
  if (std::fabs(value) > MAX_F16) {
    const bool negative = std::signbit(value);
    bool to_infinity = false;
    switch (round_mode & 3u) {
    case 0:
      to_infinity = std::fabs(value) >= RNE_OVERFLOW_THRESHOLD;
      break;
    case 1:
      to_infinity = !negative;
      break;
    case 2:
      to_infinity = negative;
      break;
    default:
      break;
    }
    if (to_infinity && !fp16_ovfl)
      return negative ? 0xfc00u : 0x7c00u;
    return negative ? 0xfbffu : 0x7bffu;
  }

  const uint16_t candidate = util::f32_to_f16(static_cast<float>(value));
  const double candidate_value = util::f16_to_f32(candidate);
  if (candidate_value == value)
    return candidate;

  const uint16_t lower = candidate_value < value ? candidate : next_down_f16(candidate);
  const uint16_t upper = candidate_value > value ? candidate : next_up_f16(candidate);

  switch (round_mode & 3u) {
  case 0: {
    const double lower_distance = value - util::f16_to_f32(lower);
    const double upper_distance = util::f16_to_f32(upper) - value;
    if (lower_distance < upper_distance)
      return lower;
    if (upper_distance < lower_distance)
      return upper;
    return (lower & 1u) == 0 ? lower : upper;
  }
  case 1:
    return upper;
  case 2:
    return lower;
  case 3:
    return value < 0.0 ? upper : lower;
  default:
    return candidate;
  }
}

inline float apply_source_modifiers(float value, bool absolute, bool negate) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if (absolute)
    bits &= 0x7fffffffu;
  if (negate)
    bits ^= 0x80000000u;
  return std::bit_cast<float>(bits);
}

} // namespace detail

/// @brief Apply F16 result modifiers and perform one direct F64-to-F16 rounding.
/// @details This is the supported policy surface for fused operations whose exact result is
/// representable in F64. It avoids exposing pseudo-scalar implementation details to other
/// execution helpers. CLAMP's NaN conversion is selected separately because older profiles
/// require MODE.DX10_CLAMP while GFX12 and gfx1250 always convert NaN to positive zero.
inline uint16_t round_f16_result(double value, uint32_t round_mode, uint32_t omod, bool clamp,
                                 bool fp16_ovfl, bool clamp_nan_to_zero) {
  const bool effective_clamp = clamp && (clamp_nan_to_zero || !std::isnan(value));
  const detail::EvaluationResult modified = detail::apply_output_modifiers(
      {value, detail::ResultProvenance::VALUE}, omod, effective_clamp);
  return detail::round_f64_to_f16(modified, round_mode, fp16_ovfl);
}

namespace detail {

inline transcendental::Operation valu_operation(Operation operation) {
  switch (operation) {
  case Operation::EXP2:
    return transcendental::Operation::EXP;
  case Operation::LOG2:
    return transcendental::Operation::LOG;
  case Operation::RCP:
    return transcendental::Operation::RCP;
  case Operation::RSQ:
    return transcendental::Operation::RSQ;
  case Operation::SQRT:
    return transcendental::Operation::SQRT;
  }
  return transcendental::Operation::RCP;
}

} // namespace detail

/// @brief Execute a pseudo-scalar F32 transcendental operation.
/// @details Matches the vector instruction: MODE.FP_ROUND and MODE.FP_DENORM are ignored,
/// subnormal sources and results flush to signed zero, and CLAMP converts NaN to zero.
/// @param operation Transcendental operation to execute.
/// @param source Raw F32 source value.
/// @param absolute Whether to clear the source sign bit before evaluation.
/// @param negate Whether to toggle the source sign bit after applying absolute value.
/// @param round_mode Numeric MODE.FP_ROUND encoding for F32; ignored by the hardware.
/// @param denorm_mode Numeric MODE.FP_DENORM encoding for F32; ignored by the hardware.
/// @param omod Numeric VOP3 OMOD encoding: 0 unchanged, 1 multiply by 2, 2 multiply by 4, and 3
/// multiply by 0.5.
/// @param clamp Whether to clamp NaN and negative results to zero and results above one to one.
/// @returns Raw 32-bit F32 result encoding.
inline uint32_t execute_f32(Operation operation, float source, bool absolute, bool negate,
                            [[maybe_unused]] uint32_t round_mode,
                            [[maybe_unused]] uint32_t denorm_mode, uint32_t omod, bool clamp) {
  return transcendental::execute_f32(detail::valu_operation(operation),
                                     std::bit_cast<uint32_t>(source), absolute, negate, omod,
                                     clamp);
}

/// @brief Execute a pseudo-scalar F16 transcendental operation.
/// @details Matches the vector instruction: MODE.FP_ROUND is ignored, MODE.FP_DENORM selects
/// whether F16 source and result subnormals are kept, FP16_OVFL saturates infinite results of
/// finite sources, and CLAMP converts NaN to zero.
/// @param operation Transcendental operation to execute.
/// @param source F16 source value represented exactly as an F32 value.
/// @param absolute Whether to clear the source sign bit before evaluation.
/// @param negate Whether to toggle the source sign bit after applying absolute value.
/// @param round_mode Numeric MODE.FP_ROUND encoding for F16; ignored by the hardware.
/// @param denorm_mode Numeric MODE.FP_DENORM encoding for F16.
/// @param omod Numeric VOP3 OMOD encoding: 0 unchanged, 1 multiply by 2, 2 multiply by 4, and 3
/// multiply by 0.5.
/// @param clamp Whether to clamp NaN and negative results to zero and results above one to one.
/// @param fp16_ovfl Whether MODE.FP16_OVFL finite-overflow saturation is enabled.
/// @returns Raw F16 encoding in bits 15:0 with bits 31:16 cleared.
inline uint32_t execute_f16(Operation operation, float source, bool absolute, bool negate,
                            [[maybe_unused]] uint32_t round_mode, uint32_t denorm_mode,
                            uint32_t omod, bool clamp, bool fp16_ovfl) {
  // Promotion from F16 is exact, including NaN payloads, so narrowing recovers the source bits.
  return transcendental::execute_f16(detail::valu_operation(operation), util::f32_to_f16(source),
                                     absolute, negate, omod, clamp, denorm_mode, fp16_ovfl);
}

} // namespace rocjitsu::amdgpu::pseudo_scalar
