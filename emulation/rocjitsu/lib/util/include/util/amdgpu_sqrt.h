// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// @brief Shared FP32 square-root mapping for AMDGPU instruction execution.

#include <bit>
#include <cstdint>

namespace util::detail {

// Piecewise cubic in fixed point that reproduces every FP32 square root
// captured on physical gfx1201 with no residual table. It is an empirical
// model of the datapath, not documented hardware behaviour. The index combines
// exponent parity and the 23 mantissa bits; its top five bits select one of 32
// segments, and the low 19 bits are the in-segment offset t. With u = t / 2^19,
// the unrounded significand is c0 + c1*u - (c2 - c3*(u + 2^-20))*u^2 in ULPs.
// Units: c0 1/4 ULP, c1 1/8 ULP, c2 1/8 ULP (magnitude of a negative
// coefficient), c3 1/4 ULP.
struct AmdgpuSqrtCoefficient {
  uint32_t constant;
  uint32_t linear;
  uint16_t quadratic;
  uint16_t cubic;
};

inline constexpr AmdgpuSqrtCoefficient kAmdgpuSqrtCoefficients[] = {
    {33554432u, 2097143u, 32722u, 474u}, {34587116u, 2034529u, 29882u, 409u},
    {35589849u, 1977208u, 27431u, 356u}, {36565094u, 1924474u, 25297u, 312u},
    {37514995u, 1875745u, 23425u, 275u}, {38441431u, 1830541u, 21774u, 244u},
    {39346059u, 1788454u, 20308u, 218u}, {40230350u, 1749143u, 18999u, 195u},
    {41095618u, 1712315u, 17825u, 176u}, {41943040u, 1677719u, 16767u, 159u},
    {42773675u, 1645139u, 15810u, 145u}, {43588485u, 1614387u, 14940u, 132u},
    {44388341u, 1585296u, 14147u, 120u}, {45174036u, 1557724u, 13422u, 110u},
    {45946298u, 1531542u, 12757u, 102u}, {46705792u, 1506637u, 12145u, 94u},
    {47453132u, 2965809u, 46276u, 670u}, {48913569u, 2877259u, 42260u, 578u},
    {50331648u, 2796194u, 38793u, 503u}, {51710852u, 2721617u, 35775u, 441u},
    {53054215u, 2652705u, 33129u, 389u}, {54364393u, 2588776u, 30793u, 346u},
    {55643730u, 2529256u, 28720u, 308u}, {56894307u, 2473662u, 26869u, 277u},
    {58117981u, 2421579u, 25208u, 249u}, {59316416u, 2372654u, 23712u, 225u},
    {60491112u, 2326579u, 22358u, 205u}, {61643427u, 2283088u, 21129u, 186u},
    {62774594u, 2241948u, 20008u, 171u}, {63885735u, 2202954u, 18982u, 156u},
    {64977878u, 2165927u, 18042u, 144u}, {66051965u, 2130707u, 17176u, 133u},
};

// Returns the 24-bit significand (1.23 fixed point) of sqrt for a normalized
// input with the given exponent parity (bit 23) and mantissa (bits 22..0).
// Products are rounded to 24 significant bits: when a product reaches the top
// of its range the rounding position moves up one bit.
inline uint32_t amdgpu_sqrt_normalized_bits(uint32_t index) {
  const AmdgpuSqrtCoefficient coefficient = kAmdgpuSqrtCoefficients[index >> 19];
  const uint64_t t = index & 0x7ffffu;

  // Linear term in 1/32 ULP.
  const uint64_t linear_product = uint64_t{coefficient.linear} * t;
  const uint64_t linear =
      (linear_product + (linear_product >= (uint64_t{1} << 40) ? 0x10000u : 0x8000u)) >> 17;

  // Quadratic and cubic terms share a 24-bit factor rounded half up; the square
  // uses the top 17 offset bits and keeps the top 24 bits of the 34-bit square.
  const uint64_t inner = ((uint64_t{coefficient.quadratic} << 19) -
                          uint64_t{coefficient.cubic} * (2 * t + 1) + 0x400u) >>
                         11;
  const uint64_t square = ((t >> 2) * (t >> 2)) >> 10;
  const uint64_t product = inner * square;
  uint64_t quadratic; // 2^-12 ULP.
  if (product >= (uint64_t{1} << 47)) {
    quadratic = ((product + (uint64_t{1} << 23)) >> 24) << 1;
  } else {
    const uint64_t quotient = product >> 23;
    const uint64_t remainder = product & ((uint64_t{1} << 23) - 1);
    const uint64_t midpoint = uint64_t{1} << 22;
    quadratic = quotient + (remainder > midpoint || (remainder == midpoint && (quotient & 1)));
  }

  // Sum in 2^-12 ULP and round to nearest even.
  const uint64_t sum = (uint64_t{coefficient.constant} << 10) + (linear << 7) - quadratic;
  const uint64_t result = sum >> 12;
  const uint64_t remainder = sum & 0xfffu;
  return static_cast<uint32_t>(result +
                               (remainder > 0x800u || (remainder == 0x800u && (result & 1))));
}

inline uint32_t amdgpu_sqrt_bits(uint32_t input) {
  const uint32_t magnitude = input & 0x7fffffffu;
  const uint32_t sign = input & 0x80000000u;
  if (magnitude > 0x7f800000u)
    return input | 0x00400000u;
  if (magnitude < 0x00800000u)
    return sign;
  if (sign != 0)
    return 0xffc00000u;
  if (magnitude == 0x7f800000u)
    return magnitude;

  const int exponent = static_cast<int>(magnitude >> 23) - 127;
  const uint32_t parity = static_cast<uint32_t>(exponent) & 1u;
  const uint32_t significand =
      amdgpu_sqrt_normalized_bits((parity << 23) | (magnitude & 0x007fffffu));
  const int result_exponent = (exponent - static_cast<int>(parity)) / 2 + 127;
  return (static_cast<uint32_t>(result_exponent) << 23) | (significand & 0x007fffffu);
}

} // namespace util::detail

namespace util {

/// @brief AMD single-precision square root matching physical gfx1201 bit for bit.
/// @details Empirical integer mapping, independent of MODE and host rounding.
/// Input subnormals flush to signed zero and yield that zero; results are never
/// subnormal. NaN inputs are quieted with sign and payload kept. Negative
/// normal inputs and negative infinity produce 0xffc00000; +inf yields +inf.
/// F16 V_SQRT matches nearest-even narrowing of this function on the exactly
/// promoted input.
inline float amdgpu_sqrt_f32(float value) {
  return std::bit_cast<float>(detail::amdgpu_sqrt_bits(std::bit_cast<uint32_t>(value)));
}

} // namespace util
