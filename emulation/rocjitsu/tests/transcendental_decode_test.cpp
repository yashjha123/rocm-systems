// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file transcendental_decode_test.cpp
/// @brief Decoded RDNA3/RDNA4 vector transcendental instructions against physical gfx1201.

#include "decode_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace {

using namespace rocjitsu;

struct DecodedCase {
  std::string_view assembly;
  std::array<uint32_t, 2> words;
  uint32_t mode;
  uint32_t source;
  uint32_t expected;
  // RDNA3 shares the mapping and modifier pipeline. Cases using OMOD or CLAMP stay RDNA4-only
  // because RDNA3 applies its own OMOD and NaN-clamp MODE policy.
  bool rdna3;
};

// Destination v10 starts as 0xa5a5a5a5 and v11 holds the source in every lane. Expected values
// were captured on gfx1201; MODE uses the rocjitsu raw layout.
constexpr std::array<DecodedCase, 23> kCases{{
    {"v_exp_f32_e32 v10, v11",
     {0x7E144B0Bu, 0x00000000u},
     0x000000F0u,
     0x3F3504F7u,
     0x3FD0F6A6u,
     true},
    {"v_exp_f32_e32 v10, v11",
     {0x7E144B0Bu, 0x00000000u},
     0x000000F0u,
     0xB4000000u,
     0x3F7FFFFEu,
     true},
    {"v_log_f32_e32 v10, v11",
     {0x7E144F0Bu, 0x00000000u},
     0x000000F0u,
     0x3F800001u,
     0x3438AA3Bu,
     true},
    {"v_log_f32_e32 v10, v11",
     {0x7E144F0Bu, 0x00000000u},
     0x000000F0u,
     0xBF800000u,
     0xFFC00000u,
     true},
    {"v_sqrt_f32_e32 v10, v11",
     {0x7E14670Bu, 0x00000000u},
     0x000000F0u,
     0x405FF71Cu,
     0x3FEF728Fu,
     true},
    {"v_sqrt_f32_e32 v10, v11",
     {0x7E14670Bu, 0x00000000u},
     0x000000F0u,
     0x80000001u,
     0x80000000u,
     true},
    {"v_rcp_f32_e64 v10, -v11",
     {0xD5AA000Au, 0x2000010Bu},
     0x000000F0u,
     0x3F802922u,
     0xBF7FADD6u,
     true},
    {"v_rsq_f32_e64 v10, |v11|",
     {0xD5AE010Au, 0x0000010Bu},
     0x000000F0u,
     0xBF80CAB3u,
     0x3F7F363Du,
     true},
    {"v_exp_f32_e64 v10, v11 mul:2",
     {0xD5A5000Au, 0x0800010Bu},
     0x000000F0u,
     0xC2F80000u,
     0x02000000u,
     false},
    {"v_rcp_f32_e64 v10, v11 div:2",
     {0xD5AA000Au, 0x1800010Bu},
     0x000000F0u,
     0xFE5A6E5Eu,
     0x80000000u,
     false},
    {"v_sqrt_f32_e64 v10, v11 clamp",
     {0xD5B3800Au, 0x0000010Bu},
     0x000000F0u,
     0xBF800000u,
     0x00000000u,
     false},
    {"v_log_f32_e64 v10, v11",
     {0xD5A7000Au, 0x0000010Bu},
     0x000000FFu,
     0x00000001u,
     0xFF800000u,
     true},
    {"v_exp_f16_e32 v10.l, v11.l",
     {0x7E14B10Bu, 0x00000000u},
     0x000000F0u,
     0x5A5A11C5u,
     0xA5A53C01u,
     true},
    {"v_exp_f16_e32 v10.h, v11.h",
     {0x7F14B18Bu, 0x00000000u},
     0x000000F0u,
     0x11C55A5Au,
     0x3C01A5A5u,
     true},
    {"v_log_f16_e32 v10.l, v11.h",
     {0x7E14AF8Bu, 0x00000000u},
     0x00000000u,
     0x00015A5Au,
     0xA5A5FC00u,
     true},
    {"v_log_f16_e32 v10.l, v11.h",
     {0x7E14AF8Bu, 0x00000000u},
     0x000000F0u,
     0x00015A5Au,
     0xA5A5CE00u,
     true},
    {"v_rcp_f16_e64 v10.h, v11.l",
     {0xD5D4400Au, 0x0000010Bu},
     0x000000F0u,
     0x5A5A7401u,
     0x03FFA5A5u,
     true},
    {"v_rcp_f16_e64 v10.h, v11.l mul:2",
     {0xD5D4400Au, 0x0800010Bu},
     0x000000F0u,
     0x5A5A7401u,
     0x0000A5A5u,
     false},
    {"v_rsq_f16_e64 v10.l, v11.l",
     {0xD5D6000Au, 0x0000010Bu},
     0x008000F0u,
     0x5A5A8000u,
     0xA5A5FBFFu,
     true},
    {"v_sqrt_f16_e64 v10.l, -|v11.h| div:2",
     {0xD5D5090Au, 0x3800010Bu},
     0x000000F0u,
     0xC4005A5Au,
     0xA5A5FE00u,
     false},
    {"v_sqrt_f16_e64 v10.h, v11.l clamp",
     {0xD5D5C00Au, 0x0000010Bu},
     0x000000F0u,
     0x5A5A7C01u,
     0x0000A5A5u,
     false},
    {"v_exp_f16_e64 v10.l, v11.l",
     {0xD5D8000Au, 0x0000010Bu},
     0x008000F0u,
     0x5A5A4C00u,
     0xA5A57BFFu,
     true},
    {"v_exp_f16_e64 v10.l, v11.l div:2",
     {0xD5D8000Au, 0x1800010Bu},
     0x008000F0u,
     0x5A5A4C00u,
     0xA5A577FFu,
     false},
}};

class TranscendentalDecodeTest : public ::testing::TestWithParam<rj_code_arch_t> {};

TEST_P(TranscendentalDecodeTest, MatchesPhysicalGfx1201) {
  constexpr uint32_t kDestination = 10;
  constexpr uint32_t kSource = 11;
  const rj_code_arch_t arch = GetParam();
  amdgpu::GpuMemory memory("transcendental_decode_memory");
  amdgpu::L2Cache l2("transcendental_decode_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  auto compute_unit =
      amdgpu::ComputeUnitCore::create("transcendental_decode", config, &memory, &l2);
  auto decoder = Decoder::create(arch);
  ASSERT_NE(compute_unit, nullptr);
  ASSERT_NE(decoder, nullptr);
  amdgpu::Wavefront *wavefront = compute_unit->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wavefront, nullptr);
  const bool original_force_scalar = util::force_scalar();

  for (const DecodedCase &test_case : kCases) {
    if (arch != ROCJITSU_CODE_ARCH_RDNA4 && !test_case.rdna3)
      continue;
    SCOPED_TRACE(test_case.assembly);
    std::unique_ptr<Instruction> instruction(decode_valid(
        *decoder, reinterpret_cast<const rj_code_binary_inst_t *>(test_case.words.data())));
    ASSERT_NE(instruction, nullptr);
    for (bool force_scalar : {false, true}) {
      SCOPED_TRACE(force_scalar ? "scalar" : "simd");
      util::set_force_scalar_for_testing(force_scalar);
      const uint32_t base = wavefront->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < wavefront->wf_size(); ++lane) {
        compute_unit->write_vgpr(base + kSource, lane, test_case.source);
        compute_unit->write_vgpr(base + kDestination, lane, 0xa5a5a5a5u);
      }
      wavefront->set_mode_raw(test_case.mode);
      wavefront->set_exec(~uint64_t{0});
      EXPECT_TRUE(compute_unit->execute_instruction(instruction.get(), *wavefront).succeeded());
      for (uint32_t lane = 0; lane < wavefront->wf_size(); ++lane)
        EXPECT_EQ(compute_unit->read_vgpr(base + kDestination, lane), test_case.expected)
            << "lane=" << lane;
    }
  }
  util::set_force_scalar_for_testing(original_force_scalar);
  wavefront->halt();
}

INSTANTIATE_TEST_SUITE_P(Rdna, TranscendentalDecodeTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4));

struct DppCase {
  std::string_view assembly;
  std::array<uint32_t, 3> words;
  uint32_t mode;
  uint32_t source;
  uint32_t source_step;
  std::array<uint32_t, 32> expected;
};

// Lane L of v11 holds source + L * source_step; v10 starts as 0xa5a5a5a5. Captured on gfx1201
// in wave32, including lanes whose DPP source is invalid.
constexpr std::array<DppCase, 8> kDppCases{{
    {"v_exp_f32_dpp v10, v11 row_shr:1 row_mask:0xf bank_mask:0xf",
     {0x7E144AFAu, 0xFF01110Bu, 0x00000000u},
     0x000000F0u,
     0x3F3504F7u,
     0x00012345u,
     {{0xA5A5A5A5u, 0x3FD0F6A6u, 0x3FD19BB4u, 0x3FD24143u, 0x3FD2E756u, 0x3FD38DEBu, 0x3FD43504u,
       0x3FD4DCA2u, 0x3FD584C3u, 0x3FD62D69u, 0x3FD6D695u, 0x3FD78046u, 0x3FD82A7Du, 0x3FD8D53Bu,
       0x3FD9807Fu, 0x3FDA2C4Bu, 0xA5A5A5A5u, 0x3FDB857Au, 0x3FDC32DEu, 0x3FDCE0CBu, 0x3FDD8F42u,
       0x3FDE3E42u, 0x3FDEEDCCu, 0x3FDF9DE1u, 0x3FE04E81u, 0x3FE0FFADu, 0x3FE1B165u, 0x3FE263A9u,
       0x3FE3167Au, 0x3FE3C9D8u, 0x3FE47DC3u, 0x3FE5323Du}}},
    {"v_exp_f32_dpp v10, v11 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
     {0x7E144AFAu, 0xFF09110Bu, 0x00000000u},
     0x000000F0u,
     0x3F3504F7u,
     0x00012345u,
     {{0x3F800000u, 0x3FD0F6A6u, 0x3FD19BB4u, 0x3FD24143u, 0x3FD2E756u, 0x3FD38DEBu, 0x3FD43504u,
       0x3FD4DCA2u, 0x3FD584C3u, 0x3FD62D69u, 0x3FD6D695u, 0x3FD78046u, 0x3FD82A7Du, 0x3FD8D53Bu,
       0x3FD9807Fu, 0x3FDA2C4Bu, 0x3F800000u, 0x3FDB857Au, 0x3FDC32DEu, 0x3FDCE0CBu, 0x3FDD8F42u,
       0x3FDE3E42u, 0x3FDEEDCCu, 0x3FDF9DE1u, 0x3FE04E81u, 0x3FE0FFADu, 0x3FE1B165u, 0x3FE263A9u,
       0x3FE3167Au, 0x3FE3C9D8u, 0x3FE47DC3u, 0x3FE5323Du}}},
    {"v_log_f32_dpp v10, v11 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
     {0x7E144EFAu, 0xFF00B10Bu, 0x00000000u},
     0x000000F0u,
     0x3F3504F7u,
     0x00012345u,
     {{0xBEFB5F1Eu, 0xBEFFFFF1u, 0xBEF23383u, 0xBEF6C5ACu, 0xBEE924B3u, 0xBEEDA88Du, 0xBEE031FAu,
       0xBEE4A7DEu, 0xBED75AABu, 0xBEDBC2EFu, 0xBECE9E22u, 0xBED2F918u, 0xBEC5FBBDu, 0xBECA49B5u,
       0xBEBD72E2u, 0xBEC1B428u, 0xBEB502FDu, 0xBEB937DAu, 0xBEACAB7Bu, 0xBEB0D438u, 0xBEA46BD3u,
       0xBEA888B4u, 0xBE9C437Du, 0xBEA054C6u, 0xBE9431F6u, 0xBE9837E7u, 0xBE8C36C1u, 0xBE903199u,
       0xBE845163u, 0xBE88415Fu, 0xBE7902CDu, 0xBE8066C0u}}},
    {"v_sqrt_f32 v10, v11 dpp8:[7,6,5,4,3,2,1,0]",
     {0x7E1466E9u, 0x0539770Bu, 0x00000000u},
     0x000000F0u,
     0x3F3504F7u,
     0x00012345u,
     {{0x3F5BF447u, 0x3F5B4A85u, 0x3F5AA03Fu, 0x3F59F574u, 0x3F594A24u, 0x3F589E4Bu, 0x3F57F1EAu,
       0x3F5744FFu, 0x3F613057u, 0x3F608A8Au, 0x3F5FE443u, 0x3F5F3D80u, 0x3F5E963Fu, 0x3F5DEE81u,
       0x3F5D4643u, 0x3F5C9D86u, 0x3F664DF4u, 0x3F65ABD9u, 0x3F65094Bu, 0x3F646649u, 0x3F63C2D3u,
       0x3F631EE7u, 0x3F627A84u, 0x3F61D5AAu, 0x3F6B4F1Au, 0x3F6AB074u, 0x3F6A1162u, 0x3F6971E4u,
       0x3F68D1F8u, 0x3F68319Eu, 0x3F6790D6u, 0x3F66EF9Du}}},
    {"v_rsq_f32_e64_dpp v10, -v11 mul:2 row_ror:3 row_mask:0xf bank_mask:0xf",
     {0xD5AE000Au, 0x280000FAu, 0xFF01230Bu},
     0x000000F0u,
     0x3F3504F7u,
     0x00012345u,
     {{0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u,
       0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u,
       0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u,
       0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u,
       0xFFC00000u, 0xFFC00000u, 0xFFC00000u, 0xFFC00000u}}},
    {"v_exp_f16_dpp v10.h, v11.l row_shl:2 row_mask:0xf bank_mask:0xf",
     {0x7F14B0FAu, 0xFF01020Bu, 0x00000000u},
     0x000000F0u,
     0x11C53C01u,
     0x00012345u,
     {{0x3C00A5A5u, 0x3BE0A5A5u, 0x1324A5A5u, 0x0000A5A5u, 0x3C00A5A5u, 0x3CA5A5A5u, 0x7C00A5A5u,
       0x7C00A5A5u, 0x3BF9A5A5u, 0x3406A5A5u, 0x0000A5A5u, 0x3C00A5A5u, 0x3C21A5A5u, 0x7C00A5A5u,
       0xA5A5A5A5u, 0xA5A5A5A5u, 0x39F2A5A5u, 0x0000A5A5u, 0xFF65A5A5u, 0x3C06A5A5u, 0x4BD1A5A5u,
       0x7C00A5A5u, 0x3C00A5A5u, 0x3B85A5A5u, 0x0000A5A5u, 0x0000A5A5u, 0x3C01A5A5u, 0x3F39A5A5u,
       0x7C00A5A5u, 0x3C00A5A5u, 0xA5A5A5A5u, 0xA5A5A5A5u}}},
    {"v_rcp_f16_e64_dpp v10.l, |v11.h| clamp quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
     {0xD5D4890Au, 0x000000FAu, 0xFF001B0Bu},
     0x000000F0u,
     0x11C53C01u,
     0x00012345u,
     {{0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u,
       0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u,
       0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u,
       0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u,
       0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u, 0xA5A53C00u}}},
    {"v_log_f16_e64_dpp v10.l, v11.l row_share:1 row_mask:0xf bank_mask:0xf",
     {0xD5D7000Au, 0x000000FAu, 0xFF01510Bu},
     0x000000F0u,
     0x11C53C01u,
     0x00012345u,
     {{0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu,
       0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5486Eu,
       0xA5A5486Eu, 0xA5A5486Eu, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u,
       0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u,
       0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u, 0xA5A5FE00u}}},
}};

TEST(TranscendentalDecodeTest, DppMatchesPhysicalGfx1201) {
  constexpr uint32_t kDestination = 10;
  constexpr uint32_t kSource = 11;
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  amdgpu::GpuMemory memory("transcendental_dpp_memory");
  amdgpu::L2Cache l2("transcendental_dpp_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  auto compute_unit = amdgpu::ComputeUnitCore::create("transcendental_dpp", config, &memory, &l2);
  auto decoder = Decoder::create(arch);
  ASSERT_NE(compute_unit, nullptr);
  ASSERT_NE(decoder, nullptr);
  amdgpu::Wavefront *wavefront = compute_unit->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wavefront, nullptr);
  ASSERT_GE(wavefront->wf_size(), 32u);
  const bool original_force_scalar = util::force_scalar();

  for (const DppCase &test_case : kDppCases) {
    SCOPED_TRACE(test_case.assembly);
    std::unique_ptr<Instruction> instruction(decode_valid(
        *decoder, reinterpret_cast<const rj_code_binary_inst_t *>(test_case.words.data())));
    ASSERT_NE(instruction, nullptr);
    for (bool force_scalar : {false, true}) {
      SCOPED_TRACE(force_scalar ? "scalar" : "simd");
      util::set_force_scalar_for_testing(force_scalar);
      const uint32_t base = wavefront->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < wavefront->wf_size(); ++lane) {
        compute_unit->write_vgpr(base + kSource, lane,
                                 test_case.source + lane * test_case.source_step);
        compute_unit->write_vgpr(base + kDestination, lane, 0xa5a5a5a5u);
      }
      wavefront->set_mode_raw(test_case.mode);
      wavefront->set_exec(0xffffffffu);
      EXPECT_TRUE(compute_unit->execute_instruction(instruction.get(), *wavefront).succeeded());
      for (uint32_t lane = 0; lane < 32; ++lane)
        EXPECT_EQ(compute_unit->read_vgpr(base + kDestination, lane), test_case.expected[lane])
            << "lane=" << lane;
    }
  }
  util::set_force_scalar_for_testing(original_force_scalar);
  wavefront->halt();
}

} // namespace
