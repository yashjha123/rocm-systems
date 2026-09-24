// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file transcendental_test.cpp
/// @brief Phase C unit tests for shared transcendental functions.

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace {

using namespace rocjitsu::amdgpu::transcendental;

// ---------------------------------------------------------------------------
// Special-case tests (±0, ±Inf, NaN, denormals)
// ---------------------------------------------------------------------------

TEST(TranscendentalTest, RcpF32SpecialCases) {
  EXPECT_EQ(rcp_f32(0.0f), std::numeric_limits<float>::infinity());
  EXPECT_EQ(rcp_f32(-0.0f), -std::numeric_limits<float>::infinity());
  EXPECT_EQ(rcp_f32(std::numeric_limits<float>::infinity()), 0.0f);
  EXPECT_EQ(rcp_f32(-std::numeric_limits<float>::infinity()), -0.0f);
  EXPECT_TRUE(std::isnan(rcp_f32(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_FLOAT_EQ(rcp_f32(2.0f), 0.5f);
}

TEST(TranscendentalTest, RcpF32MatchesPhysicalRdna3AndRdna4) {
  const uint32_t cases[][2] = {
      {0x3f800000u, 0x3f800000u}, {0x3f800001u, 0x3f7ffffeu}, {0x3f81ffffu, 0x3f7c0fc3u},
      {0x3f820000u, 0x3f7c0fc1u}, {0x3f83ffffu, 0x3f783e12u}, {0x3f840000u, 0x3f783e10u},
      {0x3fc00000u, 0x3f2aaaaau}, {0x3fffffffu, 0x3f000001u}, {0x3f802922u, 0x3f7fadd6u},
      {0x3f932eddu, 0x3f5ea262u}, {0x3fb0333cu, 0x3f39f868u}, {0x3ffff486u, 0x3f0005beu},
      {0x00800000u, 0x7e800000u}, {0x7f000000u, 0x00000000u}, {0x00000001u, 0x7f800000u},
      {0x7fa12345u, 0x7fe12345u},
  };
  for (const auto &test : cases)
    for (uint32_t sign : {0u, 0x80000000u})
      EXPECT_EQ(std::bit_cast<uint32_t>(rcp_f32(std::bit_cast<float>(test[0] | sign))),
                test[1] | sign)
          << std::hex << test[0] << " sign=" << sign;
}

TEST(TranscendentalTest, RcpF32CompleteNormalizedHardwareDigest) {
  // FNV-style hash of raw result words captured independently on gfx1100 and gfx1201.
  // Check the complete mantissa domain without storing the 32 MiB capture.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t mantissa = 0; mantissa < (1u << 23); ++mantissa) {
    const float input = std::bit_cast<float>(0x3f800000u | mantissa);
    digest = (digest ^ std::bit_cast<uint32_t>(rcp_f32(input))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0xd54ec24992572df9ull);
}

TEST(TranscendentalTest, RsqF32SpecialCases) {
  EXPECT_EQ(rsq_f32(0.0f), std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(rsq_f32(-1.0f)));
  EXPECT_EQ(rsq_f32(std::numeric_limits<float>::infinity()), 0.0f);
  EXPECT_TRUE(std::isnan(rsq_f32(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_FLOAT_EQ(rsq_f32(4.0f), 0.5f);
}

TEST(TranscendentalTest, RsqF32MatchesPhysicalRdna3AndRdna4) {
  const uint32_t cases[][2] = {
      {0x3f800000u, 0x3f800000u}, {0x3f800001u, 0x3f7fffffu}, {0x3f83ffffu, 0x3f7c1765u},
      {0x3f840000u, 0x3f7c1764u}, {0x3f8040c4u, 0x3f7fbf55u}, {0x3f80cab3u, 0x3f7f363du},
      {0x3fbfffffu, 0x3f5105ecu}, {0x3fc00000u, 0x3f5105ecu}, {0x3fffffffu, 0x3f3504f3u},
      {0x40000000u, 0x3f3504f3u}, {0x40000001u, 0x3f3504f2u}, {0x4003ffffu, 0x3f32416au},
      {0x40040000u, 0x3f32416au}, {0x403fffffu, 0x3f13cd3bu}, {0x40400000u, 0x3f13cd3au},
      {0x407fffffu, 0x3f000000u}, {0x00000000u, 0x7f800000u}, {0x80000000u, 0xff800000u},
      {0x00000001u, 0x7f800000u}, {0x80000001u, 0xff800000u}, {0xbf800000u, 0xffc00000u},
      {0xff800000u, 0xffc00000u}, {0x7f800000u, 0x00000000u}, {0x7fa12345u, 0x7fe12345u},
      {0xffa12345u, 0xffe12345u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(rsq_f32(std::bit_cast<float>(test[0]))), test[1])
        << std::hex << test[0];
}

TEST(TranscendentalTest, RsqF32CompleteNormalizedHardwareDigest) {
  // FNV-style hash of raw result words captured independently on gfx1100 and gfx1201.
  // Cover every mantissa in both exponent parities without storing the 64 MiB capture.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t index = 0; index < (1u << 24); ++index) {
    const float input = std::bit_cast<float>(0x3f800000u + index);
    digest = (digest ^ std::bit_cast<uint32_t>(rsq_f32(input))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0x010bc79eb6e48cafull);
}

TEST(TranscendentalTest, SqrtF32SpecialCases) {
  EXPECT_TRUE(std::isnan(sqrt_f32(-1.0f)));
  EXPECT_EQ(sqrt_f32(0.0f), 0.0f);
  EXPECT_EQ(sqrt_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(sqrt_f32(4.0f), 2.0f);
}

TEST(TranscendentalTest, LogF32SpecialCases) {
  EXPECT_EQ(log_f32(0.0f), -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(log_f32(-1.0f)));
  EXPECT_EQ(log_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(log_f32(1.0f), 0.0f);
  EXPECT_FLOAT_EQ(log_f32(4.0f), 2.0f);
}

TEST(TranscendentalTest, ExpF32SpecialCases) {
  EXPECT_EQ(exp_f32(-std::numeric_limits<float>::infinity()), 0.0f);
  EXPECT_EQ(exp_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(exp_f32(0.0f), 1.0f);
  EXPECT_FLOAT_EQ(exp_f32(1.0f), 2.0f);
}

TEST(TranscendentalTest, SqrtF32MatchesPhysicalRdna4) {
  // A piecewise cubic with 24-bit rounded products; results are never subnormal.
  const uint32_t cases[][2] = {
      {0x3f800003u, 0x3f800002u}, {0x3f800007u, 0x3f800004u}, {0x3f80005fu, 0x3f800030u},
      {0x3f805d16u, 0x3f802e82u}, {0x3f80c013u, 0x3f805fe5u}, {0x3f832057u, 0x3f818dc2u},
      {0x3f8395a3u, 0x3f81c7a7u}, {0x402ebc53u, 0x3fd3800cu}, {0x40074a23u, 0x3fba1a3cu},
      {0x4007e34fu, 0x3fba8378u}, {0x405ff71cu, 0x3fef728fu}, {0x402fa7cau, 0x3fd40e5du},
      {0x7f5ff71cu, 0x5f6f728fu}, {0x015ff71cu, 0x206f728fu}, {0x0080c013u, 0x20005fe5u},
      {0x7e832057u, 0x5f018dc2u}, {0x40744000u, 0x3ffa0e56u}, {0x007fffffu, 0x00000000u},
      {0x807fffffu, 0x80000000u}, {0xbf800000u, 0xffc00000u}, {0xff800000u, 0xffc00000u},
      {0x7f800000u, 0x7f800000u}, {0x7fa00000u, 0x7fe00000u}, {0xffc12345u, 0xffc12345u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(sqrt_f32(std::bit_cast<float>(test[0]))), test[1])
        << std::hex << test[0];
}

TEST(TranscendentalTest, SqrtF32CompleteNormalizedHardwareDigest) {
  // FNV-1a over little-endian result words captured on gfx1201 for every mantissa in both
  // exponent parities.
  uint64_t digest = 0xcbf29ce484222325ull;
  for (uint32_t index = 0; index < (1u << 24); ++index) {
    const uint32_t word =
        std::bit_cast<uint32_t>(sqrt_f32(std::bit_cast<float>(0x3f800000u + index)));
    for (int shift = 0; shift < 32; shift += 8)
      digest = (digest ^ ((word >> shift) & 0xffu)) * 0x100000001b3ull;
  }
  EXPECT_EQ(digest, 0xc696d1c206aff8a9ull);
}

TEST(TranscendentalTest, LogF32MatchesPhysicalRdna4) {
  // Away from one the fixed-point sum rounds toward -inf; near one a separate path keeps
  // relative precision and rounds the magnitude up.
  const uint32_t cases[][2] = {
      {0x7f7fffffu, 0x42ffffffu}, {0x00ffffffu, 0xc2fa0001u}, {0x0d80001au, 0xc2c80000u},
      {0x4200000du, 0x40a00004u}, {0x4000000du, 0x3f800012u}, {0x3f86005bu, 0x3d87619cu},
      {0x3f400027u, 0xbed47f35u}, {0x3f800001u, 0x3438aa3bu}, {0x3f800004u, 0x3538aa39u},
      {0x3f800043u, 0x37415204u}, {0x3f800400u, 0x3938a759u}, {0x3f804007u, 0x3b389049u},
      {0x3f810007u, 0x3c37f789u}, {0x3f820007u, 0x3cb73f30u}, {0x3f7fffffu, 0xb3b8aa3cu},
      {0x3f7ffc00u, 0xb8b8abadu}, {0x3f7fc000u, 0xbab8c155u}, {0x3f7e001cu, 0xbc3959b0u},
      {0x3f7c0001u, 0xbcba1f46u}, {0x00000001u, 0xff800000u}, {0xbf800000u, 0xffc00000u},
      {0x80000000u, 0xff800000u}, {0xff800000u, 0xffc00000u}, {0x7f800001u, 0x7fc00001u},
      {0xff800001u, 0xffc00001u}, {0x3f800000u, 0x00000000u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(log_f32(std::bit_cast<float>(test[0]))), test[1])
        << std::hex << test[0];
}

TEST(TranscendentalTest, LogF32HardwareDigest) {
  // FNV-1a over little-endian result words captured on gfx1201 for every mantissa with
  // exponents 0 and -1, which cover both near-one paths.
  uint64_t digest = 0xcbf29ce484222325ull;
  for (uint32_t base : {0x3f800000u, 0x3f000000u})
    for (uint32_t mantissa = 0; mantissa < (1u << 23); ++mantissa) {
      const uint32_t word = std::bit_cast<uint32_t>(log_f32(std::bit_cast<float>(base + mantissa)));
      for (int shift = 0; shift < 32; shift += 8)
        digest = (digest ^ ((word >> shift) & 0xffu)) * 0x100000001b3ull;
    }
  EXPECT_EQ(digest, 0x511828d8bc60ccbfull);
}

TEST(TranscendentalTest, ExpF32MatchesPhysicalRdna4) {
  // Hardware converts |x| to 29 fraction bits by truncation; negative sources use the one's
  // complement of that fraction. Results below 2^-126 flush to +0.
  const uint32_t cases[][2] = {
      {0x3f3504f7u, 0x3fd0f6a6u}, {0x3e9a2b52u, 0x3f9db53du}, {0x3d4cccf4u, 0x3f848390u},
      {0x3b12369fu, 0x3f8032b6u}, {0x40490fdbu, 0x410d331cu}, {0x41a3d72fu, 0x49b2892eu},
      {0x42c7ae1du, 0x71652301u}, {0xbf350523u, 0x3f1ccfd2u}, {0xbb123478u, 0x3f7f9abcu},
      {0xc0491070u, 0x3de81016u}, {0xc2c7ae4au, 0x0d8ef8e7u}, {0xb3700000u, 0x3f800000u},
      {0xb4000000u, 0x3f7ffffeu}, {0xc2fd0000u, 0x00000000u}, {0xc3150000u, 0x00000000u},
      {0x43000000u, 0x7f800000u}, {0x80000001u, 0x3f800000u}, {0xffa12345u, 0xffe12345u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(exp_f32(std::bit_cast<float>(test[0]))), test[1])
        << std::hex << test[0];
}

TEST(TranscendentalTest, ExpF32HardwareDigests) {
  // FNV-1a over little-endian result words captured on gfx1201.
  const auto digest = [](uint32_t count, uint32_t stride) {
    uint64_t value = 0xcbf29ce484222325ull;
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t word = std::bit_cast<uint32_t>(exp_f32(std::bit_cast<float>(index * stride)));
      for (int shift = 0; shift < 32; shift += 8)
        value = (value ^ ((word >> shift) & 0xffu)) * 0x100000001b3ull;
    }
    return value;
  };
  EXPECT_EQ(digest(1u << 16, 0x00010001u), 0xd71549c3b95204bbull);
  EXPECT_EQ(digest(1u << 24, 257u), 0x95acfb817f514b79ull);
}

TEST(TranscendentalTest, SinCosF32SpecialCases) {
  // sin(2*pi*0) = 0, cos(2*pi*0) = 1
  EXPECT_NEAR(sin_f32(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(cos_f32(0.0f), 1.0f, 1e-6f);
  // sin(2*pi*0.25) = 1, cos(2*pi*0.25) = 0
  EXPECT_NEAR(sin_f32(0.25f), 1.0f, 1e-6f);
  EXPECT_NEAR(cos_f32(0.25f), 0.0f, 1e-6f);
  // NaN/Inf inputs
  EXPECT_TRUE(std::isnan(sin_f32(std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(cos_f32(std::numeric_limits<float>::infinity())));
}

TEST(TranscendentalTest, RcpF64SpecialCases) {
  EXPECT_EQ(rcp_f64(0.0), std::numeric_limits<double>::infinity());
  EXPECT_EQ(rcp_f64(-0.0), -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(rcp_f64(2.0), 0.5);
}

TEST(TranscendentalTest, SqrtF64SpecialCases) {
  EXPECT_TRUE(std::isnan(sqrt_f64(-1.0)));
  EXPECT_DOUBLE_EQ(sqrt_f64(4.0), 2.0);
}

// ---------------------------------------------------------------------------
// ULP accuracy tests (pseudorandom inputs)
// ---------------------------------------------------------------------------

TEST(TranscendentalTest, RcpF32Ulp) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.001f, 1000.0f);
  for (int i = 0; i < 10000; ++i) {
    float x = dist(rng);
    float result = rcp_f32(x);
    float expected = 1.0f / x;
    // Allow 1 ULP difference
    ASSERT_NEAR(result, expected, std::abs(expected) * 1.2e-7f)
        << "rcp_f32(" << x << ") = " << result << " expected " << expected;
  }
}

TEST(TranscendentalTest, SqrtF32Ulp) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1e6f);
  for (int i = 0; i < 10000; ++i) {
    float x = dist(rng);
    float result = sqrt_f32(x);
    float expected = std::sqrt(x);
    ASSERT_NEAR(result, expected, std::abs(expected) * 1.2e-7f)
        << "sqrt_f32(" << x << ") = " << result << " expected " << expected;
  }
}

} // namespace
