// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file Bit-exact RDNA3/RDNA4 unary transcendental result pipeline.
///
/// @details Physical gfx1201 V_EXP, V_LOG, V_RCP, V_RSQ and V_SQRT results, in F32 and F16,
/// vector and pseudo-scalar forms, follow one pipeline:
///  - Source ABS/NEG edit the sign bit only.
///  - F32 ignores MODE entirely: subnormal sources act as signed zero and subnormal results flush
///    to signed zero, and results round to nearest-even regardless of FP_ROUND.
///  - F16 flushes subnormal sources unless the F16 input-denormal bit is set, evaluates the F32
///    approximation of the promoted source and rounds that once to nearest-even. FP_ROUND is
///    ignored. Subnormal results flush to signed zero when OMOD is nonzero or the F16
///    output-denormal bit is clear. FP16_OVFL turns an infinite result of a finite source into the
///    signed maximum finite value, including divide-by-zero results.
///  - OMOD scales that rounded result exactly. A zero result becomes positive zero; a result that
///    OMOD scales below the normal range flushes to signed zero; overflow produces infinity, which
///    FP16_OVFL again saturates. NaN and infinity are unchanged.
///  - CLAMP maps NaN and negative values, including negative zero, to positive zero and values
///    above one to one.
/// These rules were measured on every F16 source and over 1.3M F32 sources, for each form,
/// modifier combination and sixteen FP_ROUND/FP_DENORM settings with and without FP16_OVFL.

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"
#include "util/data_types.h"

#include <cstdint>

namespace rocjitsu::amdgpu::transcendental {

/// @brief Unary transcendental operation.
enum class Operation : uint8_t { EXP, LOG, RCP, RSQ, SQRT };

/// @brief Evaluate the hardware F32 approximation on raw bits, including special values.
inline uint32_t evaluate_f32_bits(Operation operation, uint32_t source) {
  switch (operation) {
  case Operation::EXP:
    return util::detail::amdgpu_exp_bits(source);
  case Operation::LOG:
    return util::detail::amdgpu_log_bits(source);
  case Operation::RCP:
    return std::bit_cast<uint32_t>(util::amdgpu_rcp_f32(std::bit_cast<float>(source)));
  case Operation::RSQ:
    return util::detail::amdgpu_rsq_bits(source);
  case Operation::SQRT:
    return util::detail::amdgpu_sqrt_bits(source);
  }
  return source;
}

/// @brief Evaluate the hardware F16 result for a raw F16 source, before modes and modifiers.
/// @details Subnormal sources are evaluated as given; subnormal results are retained.
inline uint16_t evaluate_f16_bits(Operation operation, uint16_t source) {
  if (operation == Operation::EXP)
    return util::amdgpu_exp_f16_bits(source);
  const uint32_t promoted = std::bit_cast<uint32_t>(util::f16_to_f32(source));
  return util::f32_to_f16(std::bit_cast<float>(evaluate_f32_bits(operation, promoted)));
}

namespace detail {

inline uint32_t source_modifiers_f32(uint32_t bits, bool absolute, bool negate) {
  if (absolute)
    bits &= 0x7fffffffu;
  if (negate)
    bits ^= 0x80000000u;
  return bits;
}

inline uint16_t source_modifiers_f16(uint16_t bits, bool absolute, bool negate) {
  if (absolute)
    bits &= 0x7fffu;
  if (negate)
    bits ^= 0x8000u;
  return bits;
}

/// OMOD exponent adjustment: +1 for x2, +2 for x4 and -1 for /2.
inline int omod_exponent_delta(uint32_t omod) {
  return omod == 1 ? 1 : omod == 2 ? 2 : omod == 3 ? -1 : 0;
}

inline uint32_t apply_omod_f32(uint32_t bits, uint32_t omod) {
  if (omod == 0)
    return bits;
  const uint32_t sign = bits & 0x80000000u;
  const int exponent = static_cast<int>((bits >> 23) & 0xffu);
  if ((bits & 0x7fffffffu) == 0)
    return 0;
  if (exponent == 0xff)
    return bits;
  // Pipeline results are never subnormal, so the exponent field alone carries the scale.
  const int scaled = exponent + omod_exponent_delta(omod);
  if (scaled >= 0xff)
    return sign | 0x7f800000u;
  if (scaled <= 0)
    return sign;
  return sign | (static_cast<uint32_t>(scaled) << 23) | (bits & 0x007fffffu);
}

inline uint16_t apply_omod_f16(uint16_t bits, uint32_t omod) {
  if (omod == 0)
    return bits;
  const uint16_t sign = bits & 0x8000u;
  const int exponent = (bits >> 10) & 0x1f;
  if ((bits & 0x7fffu) == 0)
    return 0;
  if (exponent == 0x1f)
    return bits;
  // Subnormal results were already flushed because OMOD is nonzero.
  const int scaled = exponent + omod_exponent_delta(omod);
  if (scaled >= 0x1f)
    return static_cast<uint16_t>(sign | 0x7c00u);
  if (scaled <= 0)
    return sign;
  return static_cast<uint16_t>(sign | (scaled << 10) | (bits & 0x03ffu));
}

inline uint32_t clamp_f32(uint32_t bits, bool nan_to_zero) {
  if ((bits & 0x7fffffffu) > 0x7f800000u)
    return nan_to_zero ? 0 : bits;
  if ((bits & 0x80000000u) != 0)
    return 0;
  return bits > 0x3f800000u ? 0x3f800000u : bits;
}

inline uint16_t clamp_f16(uint16_t bits, bool nan_to_zero) {
  if ((bits & 0x7fffu) > 0x7c00u)
    return nan_to_zero ? 0 : bits;
  if ((bits & 0x8000u) != 0)
    return 0;
  return bits > 0x3c00u ? 0x3c00u : bits;
}

inline uint16_t saturate_f16_overflow(uint16_t bits, uint16_t source, bool fp16_ovfl) {
  if (fp16_ovfl && (bits & 0x7fffu) == 0x7c00u && (source & 0x7fffu) != 0x7c00u)
    return static_cast<uint16_t>((bits & 0x8000u) | 0x7bffu);
  return bits;
}

} // namespace detail

/// @brief Execute an F32 transcendental with VOP3 source and result modifiers.
/// @param omod Effective OMOD encoding: 0 none, 1 x2, 2 x4, 3 /2.
/// @param clamp_nan_to_zero Whether CLAMP converts NaN to positive zero, as on GFX12.
/// @returns Raw F32 result bits. MODE does not affect F32 transcendentals.
inline uint32_t execute_f32(Operation operation, uint32_t source, bool absolute, bool negate,
                            uint32_t omod, bool clamp, bool clamp_nan_to_zero = true) {
  uint32_t result =
      evaluate_f32_bits(operation, detail::source_modifiers_f32(source, absolute, negate));
  result = detail::apply_omod_f32(result, omod);
  return clamp ? detail::clamp_f32(result, clamp_nan_to_zero) : result;
}

/// @brief Execute an F16 transcendental with VOP3 source and result modifiers.
/// @param omod Effective OMOD encoding: 0 none, 1 x2, 2 x4, 3 /2.
/// @param denorm_mode MODE.FP_DENORM for F16/F64: bit 0 keeps source subnormals and bit 1 keeps
/// result subnormals.
/// @param fp16_ovfl MODE.FP16_OVFL.
/// @param clamp_nan_to_zero Whether CLAMP converts NaN to positive zero, as on GFX12.
/// @returns Raw F16 result bits.
inline uint16_t execute_f16(Operation operation, uint16_t source, bool absolute, bool negate,
                            uint32_t omod, bool clamp, uint32_t denorm_mode, bool fp16_ovfl,
                            bool clamp_nan_to_zero = true) {
  source = detail::source_modifiers_f16(source, absolute, negate);
  if ((denorm_mode & 1u) == 0 && (source & 0x7c00u) == 0)
    source &= 0x8000u;
  uint16_t result = evaluate_f16_bits(operation, source);
  if ((omod != 0 || (denorm_mode & 2u) == 0) && (result & 0x7c00u) == 0)
    result &= 0x8000u;
  result = detail::saturate_f16_overflow(result, source, fp16_ovfl);
  if (omod != 0)
    result = detail::saturate_f16_overflow(detail::apply_omod_f16(result, omod), source, fp16_ovfl);
  return clamp ? detail::clamp_f16(result, clamp_nan_to_zero) : result;
}

} // namespace rocjitsu::amdgpu::transcendental
