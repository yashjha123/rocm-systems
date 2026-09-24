// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cfenv>
#include <cstdint>
#include <memory>

using namespace rocjitsu;

// Physical gfx1201 witnesses. These test execution policy, not an exact
// numerical mapping for arbitrary finite EXP/LOG inputs.
TEST(ExpLogF32Mode, DecodedFormsMatchHardwarePolicy) {
  amdgpu::GpuMemory memory("exp_log_memory");
  amdgpu::L2Cache cache("exp_log_cache");
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("exp_log", cfg, &memory, &cache);
  auto decoder = Decoder::create(cfg.arch);
  auto *wf = cu->dispatch_wf(0, 0, 106, 256, 32);
  const unsigned vb = wf->vgpr_alloc().base, sb = wf->sgpr_alloc().base;
  struct Case {
    bool logarithm;
    uint32_t input;
    std::array<uint32_t, 4> expected;
    bool absolute = false;
    bool negate = false;
    bool clamp = false;
  };
  const Case cases[] = {
      {false, 0x00000001u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
      {false, 0x80000001u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
      {false, 0x3f000000u, {0x3fb504f3u, 0x403504f3u, 0x40b504f3u, 0x3f3504f3u}},
      {false, 0xc3150000u, {0, 0, 0, 0}}, // -149: output flush is unconditional.
      {false, 0xc2fe0000u, {0, 0, 0, 0}}, // -127: OMOD cannot rescue core underflow.
      {false, 0xc2fc0000u, {0x00800000u, 0x01000000u, 0x01800000u, 0}},
      {false, 0x43000000u, {0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u}},
      {false, 0xff800000u, {0, 0, 0, 0}},
      {true, 0x00000001u, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
      {true, 0x807fffffu, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
      {true, 0x80800000u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {true, 0xff800000u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {true, 0x3f800000u, {0, 0, 0, 0}},
      {true, 0x3f000000u, {0xbf800000u, 0xc0000000u, 0xc0800000u, 0xbf000000u}},
      {true, 0x40000000u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
      {true, 0x7f800001u, {0x7fc00001u, 0x7fc00001u, 0x7fc00001u, 0x7fc00001u}},
      {false, 0xffa00001u, {0xffe00001u, 0xffe00001u, 0xffe00001u, 0xffe00001u}},
      {true, 0xc0000000u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}, true, false, false},
      {true, 0xc0000000u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}, true, true, false},
      {true, 0xc0000000u, {0, 0, 0, 0}, true, true, true},
      {false,
       0x7f800001u,
       {0xffc00001u, 0xffc00001u, 0xffc00001u, 0xffc00001u},
       false,
       true,
       false},
      {false,
       0x43000000u,
       {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u},
       false,
       false,
       true},
      {false,
       0xc2fe0000u,
       {0x7f000000u, 0x7f800000u, 0x7f800000u, 0x7e800000u},
       true,
       false,
       false},
      {false, 0xc2fe0000u, {0, 0, 0, 0}, true, true, false},
  };
  for (unsigned host_round = 0; host_round < 4; ++host_round) {
    amdgpu::fp_mode::ScopedEnvironment environment(host_round);
    for (const auto &test : cases) {
      for (unsigned form = 0; form < 3; ++form) {
        if (form == 0 && (test.absolute || test.negate || test.clamp))
          continue;
        for (unsigned omod = 0; omod < (form == 0 ? 1u : 4u); ++omod) {
          std::array<uint32_t, 4> words{};
          if (form == 0)
            words[0] = 0x7e000000u | (6u << 17) | ((test.logarithm ? 39u : 37u) << 9) | 256u;
          else {
            words[0] = form == 2 ? (test.logarithm ? 0xd6820005u : 0xd6800005u)
                                 : (test.logarithm ? 0xd5a70006u : 0xd5a50006u);
            words[0] |= (uint32_t{test.absolute} << 8) | (uint32_t{test.clamp} << 15);
            words[1] = (form == 2 ? 4u : 256u) | (omod << 27) | (uint32_t{test.negate} << 29);
          }
          auto decoded = decoder->decode(words.data());
          ASSERT_FALSE(decoded.failed());
          std::unique_ptr<Instruction> instruction(std::move(decoded).value());
          for (unsigned mode = 0; mode < 16; ++mode) {
            wf->set_mode_raw((mode & 3u) | ((mode >> 2) << 4));
            wf->set_exec(form == 2 ? 0 : 5);
            cu->write_sgpr(sb + 4, test.input);
            for (unsigned lane = 0; lane < 32; ++lane) {
              cu->write_vgpr(vb, lane, test.input);
              cu->write_vgpr(vb + 6, lane, 0xdeadbeefu);
            }
            ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
            const uint32_t got = form == 2 ? cu->read_sgpr(sb + 5) : cu->read_vgpr(vb + 6, 0);
            EXPECT_EQ(got, test.expected[omod])
                << "form=" << form << " mode=" << mode << " log=" << test.logarithm
                << " omod=" << omod << " input=" << std::hex << test.input;
            if (form != 2) {
              EXPECT_EQ(cu->read_vgpr(vb + 6, 2), test.expected[omod]);
              EXPECT_EQ(cu->read_vgpr(vb + 6, 1), 0xdeadbeefu);
            }
          }
        }
      }
    }
  }
  wf->halt();
}

TEST(ExpLogF32Mode, SourceModifiersClampAndHostEnvironment) {
  for (unsigned round = 0; round < 4; ++round) {
    amdgpu::fp_mode::ScopedEnvironment host(round);
    const int saved_round = std::fegetround();
    std::feraiseexcept(FE_INEXACT);
    const int saved_flags = std::fetestexcept(FE_ALL_EXCEPT);
    auto execute = amdgpu::fp_mode::rdna4_exp_log_f32;
    EXPECT_EQ(execute(false, 0x3f000000u, false, false, 0, false), 0x3fb504f3u);
    EXPECT_EQ(execute(true, 0xc0000000u, true, false, 0, false), 0x3f800000u);
    EXPECT_EQ(execute(true, 0xc0000000u, true, true, 0, false), 0xffc00000u);
    EXPECT_EQ(execute(true, 0xc0000000u, true, true, 0, true), 0u);
    EXPECT_EQ(execute(false, 0x7f800001u, false, true, 0, false), 0xffc00001u);
    EXPECT_EQ(execute(true, 0xffa00001u, true, false, 0, true), 0u);
    EXPECT_EQ(execute(false, 0x43000000u, false, false, 3, true), 0x3f800000u);
    EXPECT_EQ(std::fegetround(), saved_round);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), saved_flags);
  }
}
