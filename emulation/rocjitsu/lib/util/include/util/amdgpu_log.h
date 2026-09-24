// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// @brief Shared FP32 base-2 logarithm mapping for AMDGPU instruction execution.

#include <bit>
#include <cstdint>

namespace util::detail {

// An empirical integer mapping that reproduces every FP32 V_LOG_F32 result
// captured on physical gfx1201 (all 2^32 inputs). Its truncation points were
// recovered from the outputs, but it is not a claim about the hardware
// implementation.
//
// Away from 1, log2(2^e * 1.m) is the fixed-point sum e + L(m) rounded toward
// -inf to FP32. L(m) has 36 fraction bits and comes from a cubic segment
// selected by m[22:18]; m[22:17] == 0b000011 selects its own half-segment.
// With d = m & 0x3ffff (all divisions truncate toward -inf):
//   L = 64 * C0 + 64 * (C1 * d / 2^16) - keep25(inner * (d^2 / 2^12) / 2^22),
//   inner = C2 - (C3 * d + bias) / 2^23,
// where keep25 clears bit 0 of values with 26 significant bits.
struct AmdgpuLogSegment {
  uint32_t constant;   // C0, units of 2^-30.
  uint32_t linear;     // C1, units of 2^-46 per mantissa step.
  uint32_t quadratic;  // C2, units of 2^-24.
  uint32_t cubic;      // C3, units of 2^-47 per mantissa step.
  uint32_t cubic_bias; // Fraction below the truncated C3 * d term.
};

inline constexpr AmdgpuLogSegment kAmdgpuLogSegments[] = {
    {15u, 12102179u, 12095173u, 7703430u, 0u},
    {47667825u, 11735468u, 11378345u, 7196210u, 7786496u},
    {93912517u, 11390291u, 10714741u, 6440064u, 7413760u},
    {138816594u, 11064855u, 10111507u, 5910132u, 7176192u},
    {182455591u, 10757500u, 9557803u, 5437300u, 6864896u},
    {224898847u, 10466758u, 9048378u, 5014168u, 6701056u},
    {266210154u, 10191316u, 8578611u, 4633666u, 6881280u},
    {306448312u, 9930003u, 8144494u, 4290660u, 6971392u},
    {345667679u, 9681753u, 7742500u, 3980348u, 6138880u},
    {383918558u, 9445614u, 7369567u, 3699876u, 6987776u},
    {421247645u, 9220718u, 7022931u, 3444856u, 5824512u},
    {457698312u, 9006284u, 6700175u, 3212382u, 6217728u},
    {493310964u, 8801596u, 6399165u, 3000264u, 5619712u},
    {528123258u, 8606005u, 6118007u, 2807020u, 5586944u},
    {562170399u, 8418918u, 5854955u, 2629240u, 5513216u},
    {595485263u, 8239793u, 5608532u, 2466858u, 5709824u},
    {628098719u, 8068131u, 5377341u, 2317504u, 5332992u},
    {660039693u, 7903475u, 5160150u, 2179820u, 5226496u},
    {691335339u, 7745406u, 4955875u, 2053444u, 5193728u},
    {722011229u, 7593536u, 4763477u, 1936151u, 5345280u},
    {752091435u, 7447506u, 4582059u, 1827318u, 5513216u},
    {781598650u, 7306987u, 4410807u, 1726376u, 5046272u},
    {810554297u, 7171673u, 4248984u, 1632887u, 5227520u},
    {838978616u, 7041279u, 4095937u, 1547042u, 5300224u},
    {866890752u, 6915542u, 3950988u, 1466372u, 4939776u},
    {894308853u, 6794217u, 3813578u, 1390544u, 4890624u},
    {921250083u, 6677075u, 3683220u, 1321584u, 4857856u},
    {947730761u, 6563905u, 3559471u, 1255436u, 4808704u},
    {973766367u, 6454506u, 3441820u, 1193876u, 4784128u},
    {999371610u, 6348694u, 3329908u, 1137588u, 4767744u},
    {1024560488u, 6246298u, 3223482u, 1083560u, 4734976u},
    {1049346329u, 6147160u, 3121894u, 1032952u, 4718592u},
    // m in [0x60000, 0x80000): second half of segment 1.
    {47667912u, 11735350u, 11363743u, 6879954u, 8077312u},
};

inline uint64_t amdgpu_log_keep25(uint64_t value) {
  return value >= (uint64_t{1} << 25) ? (value & ~uint64_t{1}) : value;
}

// L(m) ~= log2(1.m) in units of 2^-36.
inline int64_t amdgpu_log_mantissa_fixed(uint32_t mantissa) {
  const uint32_t index = (mantissa >> 17) == 3 ? 32 : mantissa >> 18;
  const AmdgpuLogSegment &segment = kAmdgpuLogSegments[index];
  const uint64_t d = mantissa & 0x3ffffu;
  const uint64_t square = (d * d) >> 12;
  const uint64_t inner =
      segment.quadratic - ((uint64_t{segment.cubic} * d + segment.cubic_bias) >> 23);
  const uint64_t correction = amdgpu_log_keep25((inner * square) >> 22);
  const uint64_t linear = ((uint64_t{segment.linear} * d) >> 16) << 6;
  return static_cast<int64_t>((uint64_t{segment.constant} << 6) + linear - correction);
}

// Converts a nonzero fixed-point value with `fraction_bits` fraction bits to FP32,
// rounding toward -inf (round_up_magnitude == false) or away from zero (true).
inline uint32_t amdgpu_log_fixed_to_f32(bool negative, uint64_t magnitude, int fraction_bits,
                                        bool round_up_magnitude) {
  const int msb = 63 - std::countl_zero(magnitude);
  int exponent = msb - fraction_bits;
  uint64_t significand;
  if (msb > 23) {
    const unsigned shift = static_cast<unsigned>(msb - 23);
    significand = magnitude >> shift;
    if (round_up_magnitude && (significand << shift) != magnitude) {
      ++significand;
      if (significand >> 24) {
        significand >>= 1;
        ++exponent;
      }
    }
  } else {
    significand = magnitude << (23 - msb);
  }
  return (negative ? 0x80000000u : 0u) | (static_cast<uint32_t>(exponent + 127) << 23) |
         (static_cast<uint32_t>(significand) & 0x7fffffu);
}

// Inputs in (1, 1 + 2^-5) and (1 - 2^-6, 1) use separate cubic tables for
// |log2(x)| in terms of n = |x - 1| / q, where q is 2^-23 above one and 2^-24
// below one. With s = 4k the smallest shift making n << s >= 2^14:
//   |L| = C0 + 8 * (C1 * (n << s) / 2^15) -+ keep25(inner * sq / 2^p),
//   sq = (n << s)^2 / 2^(12 + s),
// with inner = C2 - (C3 * n + bias) / 2^23, p = 22 above one (subtracted) and
// inner = C2 + (C3 * n + bias) / 2^22, p = 23 below one (added). |L| has
// 36 + s (above) or 37 + s (below) fraction bits and rounds away from zero.
struct AmdgpuLogNearOneSegment {
  int32_t constant;
  uint32_t linear;
  uint32_t quadratic;
  uint32_t cubic;
  uint32_t cubic_bias;
};

// n in [1, 2^16), [2^16, 2^17), and [2^17, 2^18); the correction is subtracted.
inline constexpr AmdgpuLogNearOneSegment kAmdgpuLogAboveOne[] = {
    {1, 48408812u, 12102094u, 7996204u, 8257536u},
    {385, 48408744u, 12097614u, 7795547u, 8196096u},
    {5713, 48408288u, 12084310u, 7533868u, 7961344u},
};

// n in [1, 2^17) and [2^17, 2^18); the correction is added.
inline constexpr AmdgpuLogNearOneSegment kAmdgpuLogBelowOne[] = {
    {1, 48408812u, 12101979u, 2035248u, 61440u},
    {-799, 48408884u, 12097323u, 2088428u, 0u},
};

inline uint32_t amdgpu_log_near_one_bits(bool below_one, uint32_t n) {
  unsigned scale = 0;
  while (scale < 16 && (n << scale) < (1u << 14))
    scale += 4;
  const uint64_t scaled = uint64_t{n} << scale;
  const AmdgpuLogNearOneSegment &segment =
      below_one ? kAmdgpuLogBelowOne[n >> 17]
                : kAmdgpuLogAboveOne[n < (1u << 16) ? 0 : 1 + ((n >> 17) & 1)];
  const uint64_t linear = ((uint64_t{segment.linear} * scaled) >> 15) << 3;
  const uint64_t square = (scaled * scaled) >> (12 + scale);
  const uint64_t cubic = uint64_t{segment.cubic} * n + segment.cubic_bias;
  int64_t value = segment.constant + static_cast<int64_t>(linear);
  if (below_one) {
    const uint64_t inner = segment.quadratic + (cubic >> 22);
    value += static_cast<int64_t>(amdgpu_log_keep25((inner * square) >> 23));
  } else {
    const uint64_t inner = segment.quadratic - (cubic >> 23);
    value -= static_cast<int64_t>(amdgpu_log_keep25((inner * square) >> 22));
  }
  return amdgpu_log_fixed_to_f32(below_one, static_cast<uint64_t>(value),
                                 (below_one ? 37 : 36) + static_cast<int>(scale), true);
}

inline uint32_t amdgpu_log_bits(uint32_t input) {
  const uint32_t magnitude = input & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return input | 0x00400000u;
  if (magnitude < 0x00800000u)
    return 0xff800000u;
  if ((input & 0x80000000u) != 0)
    return 0xffc00000u;
  if (magnitude == 0x7f800000u)
    return 0x7f800000u;

  const int exponent = static_cast<int>(magnitude >> 23) - 127;
  const uint32_t mantissa = magnitude & 0x7fffffu;
  if (exponent == 0 && mantissa < (1u << 18))
    return mantissa == 0 ? 0u : amdgpu_log_near_one_bits(false, mantissa);
  if (exponent == -1 && mantissa > 0x7c0000u)
    return amdgpu_log_near_one_bits(true, 0x800000u - mantissa);

  const int64_t sum = exponent * (int64_t{1} << 36) + amdgpu_log_mantissa_fixed(mantissa);
  const bool negative = sum < 0;
  const uint64_t sum_bits = static_cast<uint64_t>(sum);
  return amdgpu_log_fixed_to_f32(negative, negative ? 0 - sum_bits : sum_bits, 36, negative);
}

} // namespace util::detail

namespace util {

/// @brief AMD single-precision base-2 logarithm matching captured gfx1201 outputs.
/// @details Empirical integer mapping, bit-exact against physical gfx1201 V_LOG_F32 on
/// all 2^32 inputs and independent of MODE and host rounding. Input subnormals are
/// flushed: +-0 and subnormals give -inf (0xff800000). Negative normals and -inf give
/// 0xffc00000. +inf gives +inf, 1.0 gives +0. NaNs keep sign and payload and are
/// quieted. Results stay within one ULP of log2 but often differ from round-to-nearest.
inline float amdgpu_log_f32(float value) {
  return std::bit_cast<float>(detail::amdgpu_log_bits(std::bit_cast<uint32_t>(value)));
}

} // namespace util
