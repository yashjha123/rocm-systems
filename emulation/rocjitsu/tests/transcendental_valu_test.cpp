// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file transcendental_valu_test.cpp
/// @brief Hardware digests for the RDNA transcendental result pipeline.

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental_valu.h"

#include <gtest/gtest.h>

#include <array>
#include <cfenv>
#include <cstdint>
#include <string_view>

namespace {

using rocjitsu::amdgpu::transcendental::execute_f16;
using rocjitsu::amdgpu::transcendental::execute_f32;
using rocjitsu::amdgpu::transcendental::Operation;

constexpr std::array<Operation, 5> kOperations{Operation::EXP, Operation::LOG, Operation::RCP,
                                               Operation::RSQ, Operation::SQRT};

// FNV-1a over the little-endian bytes of each 32-bit result.
struct Digest {
  uint64_t value = 0xcbf29ce484222325ull;
  void add(uint32_t word) {
    for (int shift = 0; shift < 32; shift += 8)
      value = (value ^ ((word >> shift) & 0xffu)) * 0x100000001b3ull;
  }
};

struct Modifiers {
  bool absolute = false;
  bool negate = false;
  bool clamp = false;
  uint32_t omod = 0;
};

Modifiers modifiers(std::string_view name) {
  if (name == "neg")
    return {.negate = true};
  if (name == "abs")
    return {.absolute = true};
  if (name == "negabs")
    return {.absolute = true, .negate = true};
  if (name == "clamp")
    return {.clamp = true};
  if (name == "mul2")
    return {.omod = 1};
  if (name == "mul4")
    return {.omod = 2};
  if (name == "div2")
    return {.omod = 3};
  if (name == "clampmul2")
    return {.clamp = true, .omod = 1};
  return {};
}

struct HalfConfig {
  std::string_view modifiers;
  uint32_t mode;
  uint64_t digest;
};

// V_{EXP,LOG,RCP,RSQ,SQRT}_F16 results for every F16 source, in that operation order, captured
// on gfx1201 with the given VOP3 modifiers and MODE (FP_ROUND [3:0], FP_DENORM [7:4],
// FP16_OVFL bit 23). V_S_*_F16 produced identical halves.
constexpr std::array<HalfConfig, 8> kHalfConfigs{{
    {"none", 0x000000f0u, 0x17c9ef1078f9b1a5ull},
    {"none", 0x00000000u, 0x0527382759f0b7a6ull},
    {"none", 0x008000f0u, 0x97d6d7cc080de2e1ull},
    {"div2", 0x008000f0u, 0x9ecefd3a65b12162ull},
    {"mul4", 0x00000070u, 0x1bcb83eb3a4a5be1ull},
    {"clampmul2", 0x000000f0u, 0xd857ff04d20b5141ull},
    {"negabs", 0x000000b0u, 0xa07f6ef2768ff661ull},
    {"neg", 0x0000000fu, 0x7b270ccf1af75986ull},
}};

uint64_t half_digest(const HalfConfig &config) {
  const Modifiers mods = modifiers(config.modifiers);
  const uint32_t denorm_mode = (config.mode >> 6) & 3u;
  const bool fp16_ovfl = (config.mode & (1u << 23)) != 0;
  Digest digest;
  for (Operation operation : kOperations)
    for (uint32_t source = 0; source < 0x10000u; ++source)
      digest.add(execute_f16(operation, static_cast<uint16_t>(source), mods.absolute, mods.negate,
                             mods.omod, mods.clamp, denorm_mode, fp16_ovfl));
  return digest.value;
}

TEST(TranscendentalValuTest, HalfResultsMatchPhysicalGfx1201) {
  for (const HalfConfig &config : kHalfConfigs)
    EXPECT_EQ(half_digest(config), config.digest)
        << config.modifiers << " mode=0x" << std::hex << config.mode;
}

struct SingleConfig {
  std::string_view modifiers;
  uint64_t digest;
};

// V_{EXP,LOG,RCP,RSQ,SQRT}_F32 results for sources k * 0x9e3779b9, k < 2^20, captured on gfx1201.
// MODE does not affect these results; V_S_*_F32 produced identical words.
constexpr std::array<SingleConfig, 8> kSingleConfigs{{
    {"none", 0x83684362e754ab7full},
    {"neg", 0x087ed5f537bb5cc8ull},
    {"abs", 0x9bdfb24e80003805ull},
    {"clamp", 0x455b8a1dcac00b91ull},
    {"mul2", 0xf38fa6b1d5067ffbull},
    {"mul4", 0xb2e29e102f6bad1cull},
    {"div2", 0x50ae9eae84ac014dull},
    {"clampmul2", 0x86cbd9eb2331e386ull},
}};

uint64_t single_digest(const SingleConfig &config) {
  const Modifiers mods = modifiers(config.modifiers);
  Digest digest;
  for (Operation operation : kOperations)
    for (uint32_t index = 0; index < (1u << 20); ++index)
      digest.add(execute_f32(operation, index * 0x9e3779b9u, mods.absolute, mods.negate, mods.omod,
                             mods.clamp));
  return digest.value;
}

TEST(TranscendentalValuTest, SingleResultsMatchPhysicalGfx1201) {
  for (const SingleConfig &config : kSingleConfigs)
    EXPECT_EQ(single_digest(config), config.digest) << config.modifiers;
}

TEST(TranscendentalValuTest, ResultsIgnoreHostRounding) {
  std::fenv_t saved_environment{};
  ASSERT_EQ(std::fegetenv(&saved_environment), 0);
  for (int host_mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(host_mode), 0);
    EXPECT_EQ(half_digest(kHalfConfigs[3]), kHalfConfigs[3].digest);
    EXPECT_EQ(single_digest(kSingleConfigs[7]), kSingleConfigs[7].digest);
  }
  EXPECT_EQ(std::fesetenv(&saved_environment), 0);
}

TEST(TranscendentalValuTest, OutputModifierZeroSignFollowsPhysicalGfx1201) {
  // A zero result becomes +0 under OMOD, but OMOD underflow keeps the sign.
  EXPECT_EQ(execute_f32(Operation::RCP, 0xff800000u, false, false, 1, false), 0u);
  EXPECT_EQ(execute_f32(Operation::RCP, 0xfe5a6e5eu, false, false, 3, false), 0x80000000u);
  EXPECT_EQ(execute_f16(Operation::RCP, 0xf001u, false, false, 3, false, 3, false), 0x8000u);
  // F16 results are rounded, range-checked and saturated before OMOD scales them.
  EXPECT_EQ(execute_f16(Operation::RCP, 0x7401u, false, false, 1, false, 3, false), 0u);
  EXPECT_EQ(execute_f16(Operation::EXP, 0x4c00u, false, false, 3, false, 3, false), 0x7c00u);
  EXPECT_EQ(execute_f16(Operation::EXP, 0x4c00u, false, false, 3, false, 3, true), 0x77ffu);
  EXPECT_EQ(execute_f16(Operation::LOG, 0x0000u, false, false, 3, false, 3, true), 0xf7ffu);
  // FP16_OVFL saturates divide-by-zero results but not infinite sources.
  EXPECT_EQ(execute_f16(Operation::RSQ, 0x8000u, false, false, 0, false, 3, true), 0xfbffu);
  EXPECT_EQ(execute_f16(Operation::EXP, 0x7c00u, false, false, 0, false, 3, true), 0x7c00u);
  // V_EXP_F16 rounds the wide result once: F32 gives exactly 1 + 2^-11 for this source.
  EXPECT_EQ(execute_f16(Operation::EXP, 0x11c5u, false, false, 0, false, 3, false), 0x3c01u);
}

} // namespace
