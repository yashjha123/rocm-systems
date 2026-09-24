// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
// Numerical mapping adapted from yashjha123/rocm-systems commit 43507cdc9b51.
// Coefficients were derived from gfx1201 observations, not Intel reference code.
/// @brief Shared base-2 exponential mapping for AMDGPU instruction execution.

#include <bit>
#include <cstdint>

namespace util::detail {

// Empirical fixed-point mapping that reproduces every V_EXP_F32 output captured
// on physical gfx1201 (all 2^32 inputs, default MODE). It is derived from
// hardware outputs and mirrors their observed
// structure; it is not a claim about the actual circuit.
//
// The source is converted to sign-magnitude fixed point with 29 fraction bits
// (magnitude truncated). A positive source splits into integer n and fraction
// F; a negative source uses n = -N - 1 and F = ~g (the one's complement of its
// 29-bit fraction g, not the two's complement). 2^F is evaluated per 1/32
// segment as a cubic in the segment offset t (24 bits) with 36 fraction bits:
//   A = c0 + L + Q, where
//   L = K * t rounded to 24 significant bits, then truncated to 2^-28;
//   Q = sq * D rounded to 24 significant bits (2^-36 grid, or 2^-35 once the
//       product needs 48 bits), sq = ((t >> 6)^2) >> 12, and
//   D = c2 + c3 * (t >> 6) / 128 rounded to nearest even.
// The FP32 result rounds A to 24 bits with round-to-nearest-even.
struct AmdgpuExpSegment {
  // Units relative to the segment offset x = t / 2^24 in [0, 1); 1.0 == 2^36 internally.
  uint32_t constant;  // c0: 2^-28
  uint32_t linear;    // K:  2^-28 per x
  uint32_t quadratic; // c2: 2^-35 per x^2
  uint32_t cubic;     // c3: 2^-24 per x^3
};

inline constexpr AmdgpuExpSegment kAmdgpuExpSegments[32] = {
    {268435456u, 5814541u, 8060268u, 29u},   {274313427u, 5941862u, 8236765u, 29u},
    {280320109u, 6071972u, 8417127u, 30u},   {286458319u, 6204931u, 8601438u, 31u},
    {292730941u, 6340801u, 8789785u, 31u},   {299140913u, 6479647u, 8982256u, 32u},
    {305691246u, 6621533u, 9178942u, 33u},   {312385013u, 6766525u, 9379934u, 33u},
    {319225354u, 6914693u, 9585328u, 34u},   {326215479u, 7066105u, 9795219u, 35u},
    {333358667u, 7220833u, 10009707u, 36u},  {340658273u, 7378948u, 10228891u, 36u},
    {348117717u, 7540526u, 10452874u, 37u},  {355740503u, 7705642u, 10681762u, 38u},
    {363530205u, 7874374u, 10915663u, 39u},  {371490480u, 8046800u, 11154685u, 40u},
    {379625062u, 8223002u, 11398940u, 41u},  {387937769u, 8403062u, 11648545u, 42u},
    {396432501u, 8587066u, 11903614u, 42u},  {405113242u, 8775098u, 12164270u, 43u},
    {413984066u, 8967247u, 12430633u, 44u},  {423049137u, 9163605u, 12702828u, 45u},
    {432312707u, 9364261u, 12980984u, 46u},  {441779122u, 9569312u, 13265230u, 47u},
    {451452826u, 9778853u, 13555701u, 48u},  {461338356u, 9992982u, 13852532u, 49u},
    {471440350u, 10211799u, 14155863u, 50u}, {481763549u, 10435409u, 14465836u, 52u},
    {492312797u, 10663914u, 14782597u, 53u}, {503093043u, 10897424u, 15106293u, 54u},
    {514109347u, 11136046u, 15437078u, 55u}, {525366876u, 11379894u, 15775106u, 56u},
};

/// @brief Unrounded 2^(fraction / 2^29) with 36 fraction bits, in [2^36, 2^37).
inline uint64_t amdgpu_exp2_fraction(uint32_t fraction) {
  const AmdgpuExpSegment &segment = kAmdgpuExpSegments[(fraction >> 24) & 31u];
  const uint64_t offset = fraction & 0xffffffu;
  const uint64_t linear_product = uint64_t{segment.linear} * offset;
  const uint64_t linear =
      (linear_product + (linear_product >> 47 ? uint64_t{1} << 23 : uint64_t{1} << 22)) >> 24;
  const uint64_t high_offset = offset >> 6;
  const uint64_t square = (high_offset * high_offset) >> 12;
  const uint64_t coefficient_sum = (uint64_t{segment.quadratic} << 7) + segment.cubic * high_offset;
  const uint64_t coefficient = (coefficient_sum >> 7) + (((coefficient_sum & 127u) > 64u) ||
                                                         ((coefficient_sum & 255u) == 192u));
  const uint64_t quadratic_product = square * coefficient;
  const uint64_t quadratic = quadratic_product >> 47
                                 ? ((quadratic_product + (uint64_t{1} << 23)) >> 24) << 1
                                 : (quadratic_product + (uint64_t{1} << 22)) >> 23;
  return ((uint64_t{segment.constant} + linear) << 8) + quadratic;
}

/// @brief Exact pre-rounding V_EXP result: significand * 2^(exponent - 36) for finite kinds.
struct AmdgpuExpWide {
  enum class Kind : uint8_t { FINITE, ZERO, INFINITE, QUIET_NAN };
  Kind kind;
  uint32_t nan_bits;    // quieted FP32 NaN for Kind::QUIET_NAN
  int32_t exponent;     // for Kind::FINITE
  uint64_t significand; // for Kind::FINITE: in [2^36, 2^37)
};

/// @brief Evaluate the unrounded exponential of an FP32 source (input subnormals flushed).
/// @details Sources with |x| < 2^-24 produce exactly 1.0. Sources >= 128 are infinite and
/// sources <= -128 are zero before rounding; everything else is finite and exact here.
inline AmdgpuExpWide amdgpu_exp_wide(uint32_t input) {
  const uint32_t magnitude_bits = input & 0x7fffffffu;
  const bool negative = (input >> 31) != 0;
  if (magnitude_bits > 0x7f800000u)
    return {AmdgpuExpWide::Kind::QUIET_NAN, input | 0x00400000u, 0, 0};
  const uint32_t biased_exponent = magnitude_bits >> 23;
  if (biased_exponent < 103) // |x| < 2^-24, including zero and subnormals
    return {AmdgpuExpWide::Kind::FINITE, 0, 0, uint64_t{1} << 36};
  if (biased_exponent >= 134) // |x| >= 128, including infinity
    return {negative ? AmdgpuExpWide::Kind::ZERO : AmdgpuExpWide::Kind::INFINITE, 0, 0, 0};
  // Fixed point with 29 fraction bits; low source bits beyond 2^-29 are truncated.
  const uint64_t mantissa = (magnitude_bits & 0x7fffffu) | 0x800000u;
  const uint64_t fixed = biased_exponent >= 121 ? mantissa << (biased_exponent - 121)
                                                : mantissa >> (121 - biased_exponent);
  const int32_t integer = static_cast<int32_t>(fixed >> 29);
  const uint32_t fraction = static_cast<uint32_t>(fixed) & 0x1fffffffu;
  if (negative)
    return {AmdgpuExpWide::Kind::FINITE, 0, -integer - 1,
            amdgpu_exp2_fraction(~fraction & 0x1fffffffu)};
  return {AmdgpuExpWide::Kind::FINITE, 0, integer, amdgpu_exp2_fraction(fraction)};
}

/// @brief V_EXP_F32 bit mapping. MODE rounding and denormal fields are ignored.
inline uint32_t amdgpu_exp_bits(uint32_t input) {
  const AmdgpuExpWide wide = amdgpu_exp_wide(input);
  switch (wide.kind) {
  case AmdgpuExpWide::Kind::QUIET_NAN:
    return wide.nan_bits;
  case AmdgpuExpWide::Kind::ZERO:
    return 0;
  case AmdgpuExpWide::Kind::INFINITE:
    return 0x7f800000u;
  case AmdgpuExpWide::Kind::FINITE:
    break;
  }
  const uint64_t quotient = wide.significand >> 13;
  const uint64_t remainder = wide.significand & 0x1fffu;
  uint64_t rounded = quotient + (remainder > 0x1000u || (remainder == 0x1000u && (quotient & 1)));
  int32_t exponent = wide.exponent + 127;
  if (rounded >> 24) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent <= 0)
    return 0; // results below 2^-126 flush to +0
  if (exponent >= 255)
    return 0x7f800000u;
  return (static_cast<uint32_t>(exponent) << 23) | (static_cast<uint32_t>(rounded) & 0x7fffffu);
}

} // namespace util::detail

namespace util {

/// @brief AMD single-precision 2^x matching physical gfx1201 V_EXP_F32 bit for bit.
/// @details Flushes input subnormals and never produces subnormal results (they become +0),
/// independently of MODE. NaNs are quieted with sign and payload kept. exp(+inf) = +inf,
/// exp(-inf) = +0, exp(+/-0) = 1. Sources >= 128 overflow to +inf. Results are independent
/// of host rounding.
inline float amdgpu_exp_f32(float value) {
  return std::bit_cast<float>(detail::amdgpu_exp_bits(std::bit_cast<uint32_t>(value)));
}

} // namespace util
