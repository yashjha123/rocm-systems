// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include <array>
#include <bit>
#include <cfenv>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace rocjitsu;
enum class Operation { EXP, LOG };

struct Witness {
  Operation operation;
  uint32_t input;
  std::array<uint32_t, 4> outputs;
  uint32_t modifiers = 0; // bit 0: ABS, bit 1: NEG
  bool clamp = false;
};

// Captured on RX 9070 XT (gfx1201), 2026-09-24. Output columns are OMOD
// none, x2, x4, /2. Both VOP3 forms matched under all 16 F32 MODE settings.
constexpr Witness kWitnesses[] = {
    {Operation::EXP, 0x00000000u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x00000001u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x00000002u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x007fffffu, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x00800000u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x00800001u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x3f7fffffu, {0x3fffffffu, 0x407fffffu, 0x40ffffffu, 0x3f7fffffu}},
    {Operation::EXP, 0x3f800000u, {0x40000000u, 0x40800000u, 0x41000000u, 0x3f800000u}},
    {Operation::EXP, 0x3f800001u, {0x40000001u, 0x40800001u, 0x41000001u, 0x3f800001u}},
    {Operation::EXP, 0x7f7fffffu, {0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u}},
    {Operation::EXP, 0x7f800000u, {0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u}},
    {Operation::EXP, 0x7f800001u, {0x7fc00001u, 0x7fc00001u, 0x7fc00001u, 0x7fc00001u}},
    {Operation::EXP, 0x7fc00000u, {0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u}},
    {Operation::EXP, 0x7fa00001u, {0x7fe00001u, 0x7fe00001u, 0x7fe00001u, 0x7fe00001u}},
    {Operation::EXP, 0x80000000u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x80000001u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x807fffffu, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0x80800000u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0xff7fffffu, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {Operation::EXP, 0xff800000u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {Operation::EXP, 0xffc00000u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::EXP, 0xc2fe0001u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {Operation::EXP, 0xbf800002u, {0x3efffffdu, 0x3f7ffffdu, 0x3ffffffdu, 0x3e7ffffdu}},
    {Operation::EXP, 0x41000000u, {0x43800000u, 0x44000000u, 0x44800000u, 0x43000000u}},
    {Operation::EXP, 0x4300ffffu, {0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u}},
    {Operation::EXP, 0xc269c628u, {0x223c3f66u, 0x22bc3f66u, 0x233c3f66u, 0x21bc3f66u}},
    {Operation::EXP, 0x42115895u, {0x51a19ffbu, 0x52219ffbu, 0x52a19ffbu, 0x51219ffbu}},
    {Operation::EXP, 0xc2e1450fu, {0x0724dcd0u, 0x07a4dcd0u, 0x0824dcd0u, 0x06a4dcd0u}},
    {Operation::EXP, 0xc12c9ebeu, {0x3a142f12u, 0x3a942f12u, 0x3b142f12u, 0x39942f12u}},
    {Operation::EXP, 0x67551d72u, {0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u}},
    {Operation::EXP, 0x04c86267u, {0x3f800000u, 0x40000000u, 0x40800000u, 0x3f000000u}},
    {Operation::EXP, 0xfc8aec98u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {Operation::LOG, 0x00000000u, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
    {Operation::LOG, 0x00000001u, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
    {Operation::LOG, 0x00000002u, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
    {Operation::LOG, 0x007fffffu, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
    {Operation::LOG, 0x00800000u, {0xc2fc0000u, 0xc37c0000u, 0xc3fc0000u, 0xc27c0000u}},
    {Operation::LOG, 0x00800001u, {0xc2fc0000u, 0xc37c0000u, 0xc3fc0000u, 0xc27c0000u}},
    {Operation::LOG, 0x3f7fffffu, {0xb3b8aa3cu, 0xb438aa3cu, 0xb4b8aa3cu, 0xb338aa3cu}},
    {Operation::LOG, 0x3f800000u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {Operation::LOG, 0x3f800001u, {0x3438aa3bu, 0x34b8aa3bu, 0x3538aa3bu, 0x33b8aa3bu}},
    {Operation::LOG, 0x7f7fffffu, {0x42ffffffu, 0x437fffffu, 0x43ffffffu, 0x427fffffu}},
    {Operation::LOG, 0x7f800000u, {0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u}},
    {Operation::LOG, 0x7f800001u, {0x7fc00001u, 0x7fc00001u, 0x7fc00001u, 0x7fc00001u}},
    {Operation::LOG, 0x7fc00000u, {0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u}},
    {Operation::LOG, 0x7fa00001u, {0x7fe00001u, 0x7fe00001u, 0x7fe00001u, 0x7fe00001u}},
    {Operation::LOG, 0x80000000u, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
    {Operation::LOG, 0x80000001u, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
    {Operation::LOG, 0x807fffffu, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}},
    {Operation::LOG, 0x80800000u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0xff7fffffu, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0xff800000u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0xffc00000u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0xc2fe0001u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0xbf800002u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0x41000000u, {0x40400000u, 0x40c00000u, 0x41400000u, 0x3fc00000u}},
    {Operation::LOG, 0x4300ffffu, {0x40e05bf8u, 0x41605bf8u, 0x41e05bf8u, 0x40605bf8u}},
    {Operation::LOG, 0xc269c628u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0x42115895u, {0x40a5ddfcu, 0x4125ddfcu, 0x41a5ddfcu, 0x4025ddfcu}},
    {Operation::LOG, 0xc2e1450fu, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0xc12c9ebeu, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    {Operation::LOG, 0x67551d72u, {0x429f7891u, 0x431f7891u, 0x439f7891u, 0x421f7891u}},
    {Operation::LOG, 0x04c86267u, {0xc2eab4eeu, 0xc36ab4eeu, 0xc3eab4eeu, 0xc26ab4eeu}},
    {Operation::LOG, 0xfc8aec98u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
    // Captured source-modifier and CLAMP combinations (both VOP3 forms).
    {Operation::EXP, 0xc31ffffeu, {0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u}, 1u, false},
    {Operation::EXP, 0xc31ffffeu, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, 3u, false},
    {Operation::EXP, 0xbf800002u, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}, 2u, true},
    {Operation::EXP, 0x7f800001u, {0xffc00001u, 0xffc00001u, 0xffc00001u, 0xffc00001u}, 2u, false},
    {Operation::EXP, 0x7fa00001u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, 1u, true},
    {Operation::EXP, 0x4300ffffu, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, 2u, false},
    {Operation::LOG, 0x807fffffu, {0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u}, 1u, false},
    {Operation::LOG, 0x80800000u, {0xc2fc0000u, 0xc37c0000u, 0xc3fc0000u, 0xc27c0000u}, 1u, false},
    {Operation::LOG, 0x3f800001u, {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}, 2u, false},
    {Operation::LOG, 0x3f800001u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, 3u, true},
    {Operation::LOG, 0xbf800002u, {0x34b8aa3au, 0x3538aa3au, 0x35b8aa3au, 0x3438aa3au}, 1u, false},
    {Operation::LOG, 0x7fa00001u, {0xffe00001u, 0xffe00001u, 0xffe00001u, 0xffe00001u}, 3u, false},
    {Operation::LOG, 0x7fa00001u, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, 1u, true},
    {Operation::LOG, 0x3f7fffffu, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, 0u, true},
    {Operation::LOG, 0x3f800001u, {0x3438aa3bu, 0x34b8aa3bu, 0x3538aa3bu, 0x33b8aa3bu}, 0u, true},
    {Operation::LOG, 0x7f7fffffu, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}, 0u, true},
};

// Digests of physical gfx1201 result words, not Python/model-generated expectations.
// EXP covers i / 2^23 in [0, 1), not every representable F32 in that interval.
// LOG covers every F32 mantissa in [1, 2). Other exponents and signs are covered
// by decoded witnesses and the separately recorded all-2^32-input hardware sweep.
TEST(Rdna4ExpLog, CapturedDomainHardwareDigests) {
  uint64_t exp_digest = 14695981039346656037ull;
  uint64_t log_digest = 14695981039346656037ull;
  for (uint32_t index = 0; index < (1u << 23); ++index) {
    uint32_t exp_input = 0;
    if (index != 0) {
      const unsigned top = std::bit_width(index) - 1;
      exp_input = ((top + 104u) << 23) | ((index << (23 - top)) & 0x7fffffu);
    }
    exp_digest = (exp_digest ^ util::detail::amdgpu_exp_bits(exp_input)) * 1099511628211ull;
    log_digest =
        (log_digest ^ util::detail::amdgpu_log_bits(0x3f800000u | index)) * 1099511628211ull;
  }
  EXPECT_EQ(exp_digest, 0xef14981b949b3960ull);
  EXPECT_EQ(log_digest, 0xdcabe314d2964004ull);
}

TEST(Rdna4ExpLog, CoreAndModifiersIgnoreHostRounding) {
  const int saved = std::fegetround();
  for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(mode), 0);
    for (const auto &test : kWitnesses)
      for (uint32_t omod = 0; omod < 4; ++omod)
        EXPECT_EQ(amdgpu::fp_mode::rdna4_exp_log_f32(test.operation == Operation::LOG, test.input,
                                                     (test.modifiers & 1u) != 0,
                                                     (test.modifiers & 2u) != 0, omod, test.clamp),
                  test.outputs[omod])
            << std::hex << test.input << " omod=" << omod;
  }
  EXPECT_EQ(std::fesetround(saved), 0);
}

TEST(Rdna4ExpLog, DecodedFormsModesExecAndSources) {
  for (uint32_t width : {32u, 64u}) {
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
    auto *wf = cu->dispatch_wf(0, 0, 106, 256, width);
    ASSERT_NE(wf, nullptr);
    const unsigned vb = wf->vgpr_alloc().base, sb = wf->sgpr_alloc().base;
    for (const auto &test : kWitnesses) {
      const bool logarithm = test.operation == Operation::LOG;
      for (unsigned form = 0; form < 4; ++form) {
        const bool pseudo = form == 2;
        const bool e32 = form == 0;
        if (e32 && (test.modifiers || test.clamp))
          continue;
        const bool scalar_source = pseudo || form == 3;
        for (uint32_t omod = 0; omod < (e32 ? 1u : 4u); ++omod) {
          const uint32_t source = scalar_source ? 4u : 256u;
          const uint32_t op = logarithm ? rdna4::kVLogF32Vop1 : rdna4::kVExpF32Vop1;
          const uint32_t first = e32      ? 0x7e000000u | (6u << 17) | (op << 9) | source
                                 : pseudo ? (logarithm ? 0xd6820005u : 0xd6800005u)
                                          : (logarithm ? 0xd5a70006u : 0xd5a50006u);
          std::array<uint32_t, 4> words{
              first | ((test.modifiers & 1u) << 8) | (uint32_t{test.clamp} << 15),
              source | (omod << 27) | ((test.modifiers >> 1) << 29), 0, 0};
          auto decoded = decoder->decode(words.data());
          ASSERT_FALSE(decoded.failed());
          std::unique_ptr<Instruction> instruction(std::move(decoded).value());
          for (uint32_t mode_index = 0; mode_index < 16; ++mode_index) {
            const uint32_t mode = (mode_index & 3u) | ((mode_index >> 2) << 4);
            for (uint64_t mask : {uint64_t{0}, ~uint64_t{0}, uint64_t{0xaaaaaaaaaaaaaaaa}}) {
              wf->set_mode_raw(mode);
              wf->set_exec(mask);
              cu->write_sgpr(sb + 4, test.input);
              cu->write_sgpr(sb + 5, 0xa5a55a5au);
              for (uint32_t lane = 0; lane < width; ++lane) {
                cu->write_vgpr(vb, lane, test.input);
                cu->write_vgpr(vb + 6, lane, 0xa5a55a5au);
              }
              ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
              if (pseudo) {
                EXPECT_EQ(cu->read_sgpr(sb + 5), test.outputs[omod]);
              } else {
                for (uint32_t lane = 0; lane < width; ++lane)
                  EXPECT_EQ(cu->read_vgpr(vb + 6, lane),
                            mask & (uint64_t{1} << lane) ? test.outputs[omod] : 0xa5a55a5au)
                      << "form=" << form << " lane=" << lane << " mode=" << mode;
              }
              EXPECT_EQ(wf->mode_raw(), mode);
            }
          }
        }
      }
    }
    wf->halt();
  }
}
// FNV-style word hashes computed from independent gfx1201 captures for all
// 65,536 F16 sources. Order: operation, FP16_OVFL, denormal mode, OMOD.
constexpr uint64_t kF16HardwareDigests[2][2][4][4] = {
    {
        {
            {0xa9b68cfe7b8372d8ull, 0x0b72f56e5a6d7814ull, 0xe4ea4ced067c3750ull,
             0xcfcd3f09573176b0ull},
            {0xa9b68cfe7b8372d8ull, 0x0b72f56e5a6d7814ull, 0xe4ea4ced067c3750ull,
             0xcfcd3f09573176b0ull},
            {0x7a49d8175530c589ull, 0x0b72f56e5a6d7814ull, 0xe4ea4ced067c3750ull,
             0xcfcd3f09573176b0ull},
            {0x7a49d8175530c589ull, 0x0b72f56e5a6d7814ull, 0xe4ea4ced067c3750ull,
             0xcfcd3f09573176b0ull},
        },
        {
            {0xbdc5546b38d182d8ull, 0xe3911d2073676194ull, 0xeb9c191e40f92250ull,
             0xb3152a70f19f46b0ull},
            {0xbdc5546b38d182d8ull, 0xe3911d2073676194ull, 0xeb9c191e40f92250ull,
             0xb3152a70f19f46b0ull},
            {0x36d64dcb7d64d589ull, 0xe3911d2073676194ull, 0xeb9c191e40f92250ull,
             0xb3152a70f19f46b0ull},
            {0x36d64dcb7d64d589ull, 0xe3911d2073676194ull, 0xeb9c191e40f92250ull,
             0xb3152a70f19f46b0ull},
        },
    },
    {
        {
            {0x7b139780c1becc29ull, 0x4a1c0bff99c93029ull, 0x92aee0e380774429ull,
             0xa50f91f3a56dd829ull},
            {0xab561022e31ba143ull, 0xaaf41f2388071943ull, 0x6eb7cb2eb6676143ull,
             0x04a7a4407b6b6943ull},
            {0x7b139780c1becc29ull, 0x4a1c0bff99c93029ull, 0x92aee0e380774429ull,
             0xa50f91f3a56dd829ull},
            {0xab561022e31ba143ull, 0xaaf41f2388071943ull, 0x6eb7cb2eb6676143ull,
             0x04a7a4407b6b6943ull},
        },
        {
            {0x3999dfce791d6c29ull, 0x9b025637254cd829ull, 0xa55c770db2b8c429ull,
             0x22a34e921f103029ull},
            {0xab21f15c2191eb2bull, 0x5f44fd3d7afccb2bull, 0xa83561af3a084b2bull,
             0xf19cf0fb96af3b2bull},
            {0x3999dfce791d6c29ull, 0x9b025637254cd829ull, 0xa55c770db2b8c429ull,
             0x22a34e921f103029ull},
            {0xab21f15c2191eb2bull, 0x5f44fd3d7afccb2bull, 0xa83561af3a084b2bull,
             0xf19cf0fb96af3b2bull},
        },
    },
};

TEST(Rdna4ExpLogF16, ExhaustiveHardwareDigestsIgnoreHostRounding) {
  const int saved = std::fegetround();
  for (int host_mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(host_mode), 0);
    for (unsigned op = 0; op < 2; ++op)
      for (unsigned overflow = 0; overflow < 2; ++overflow)
        for (unsigned denorm = 0; denorm < 4; ++denorm)
          for (unsigned omod = 0; omod < 4; ++omod) {
            uint64_t digest = 14695981039346656037ull;
            for (uint32_t input = 0; input < 65536; ++input)
              digest = (digest ^ amdgpu::fp_mode::rdna4_exp_log_f16(op != 0, input, denorm,
                                                                    overflow, false, false, omod)) *
                       1099511628211ull;
            EXPECT_EQ(digest, kF16HardwareDigests[op][overflow][denorm][omod])
                << "op=" << op << " overflow=" << overflow << " denorm=" << denorm
                << " omod=" << omod << " host_mode=" << host_mode;
          }
  }
  EXPECT_EQ(std::fesetround(saved), 0);
}

struct HalfWitness {
  Operation operation;
  uint16_t input;
  uint32_t denorm;
  bool overflow;
  uint32_t omod;
  uint32_t modifiers;
  bool clamp;
  uint16_t expected;
};

constexpr HalfWitness kHalfWitnesses[] = {
    {Operation::EXP, 0x11c5, 0, false, 0, 0, false, 0x3c01},
    {Operation::EXP, 0x3800, 0, false, 0, 0, false, 0x3da8},
    {Operation::EXP, 0x3800, 3, false, 1, 0, false, 0x41a8},
    {Operation::EXP, 0xce00, 0, false, 0, 0, false, 0x0000},
    {Operation::EXP, 0xce00, 2, false, 0, 0, false, 0x0001},
    {Operation::EXP, 0xce00, 3, false, 1, 0, false, 0x0000},
    {Operation::EXP, 0xcb01, 3, false, 0, 0, false, 0x03fa},
    {Operation::EXP, 0xcb01, 3, false, 2, 0, false, 0x0000},
    {Operation::EXP, 0xce40, 3, false, 0, 0, false, 0x0000},
    {Operation::EXP, 0x4c00, 0, false, 3, 0, false, 0x7c00},
    {Operation::EXP, 0x4c00, 0, true, 3, 0, false, 0x77ff},
    {Operation::EXP, 0x4bff, 0, true, 2, 0, false, 0x7bff},
    {Operation::EXP, 0x7bff, 0, true, 0, 0, false, 0x7bff},
    {Operation::EXP, 0x7c00, 0, true, 3, 0, false, 0x7c00},
    {Operation::EXP, 0xfc00, 0, true, 0, 0, false, 0x0000},
    {Operation::EXP, 0x7d01, 0, false, 0, 2, false, 0xff01},
    {Operation::EXP, 0xfd01, 3, true, 2, 1, true, 0x0000},
    {Operation::EXP, 0xbc00, 0, false, 2, 3, true, 0x3c00},
    {Operation::LOG, 0x0000, 0, false, 0, 0, false, 0xfc00},
    {Operation::LOG, 0x0000, 0, true, 3, 0, false, 0xf7ff},
    {Operation::LOG, 0x8000, 3, true, 0, 0, false, 0xfbff},
    {Operation::LOG, 0x0001, 0, false, 0, 0, false, 0xfc00},
    {Operation::LOG, 0x0001, 1, false, 0, 0, false, 0xce00},
    {Operation::LOG, 0x8001, 1, false, 0, 0, false, 0xfe00},
    {Operation::LOG, 0x7c00, 0, true, 0, 0, false, 0x7c00},
    {Operation::LOG, 0xfc00, 0, true, 0, 0, false, 0xfe00},
    {Operation::LOG, 0xbc00, 0, false, 0, 1, false, 0x0000},
    {Operation::LOG, 0x3c01, 3, false, 3, 0, false, 0x11c5},
    {Operation::LOG, 0x3bff, 3, false, 3, 0, false, 0x8dc6},
    {Operation::LOG, 0x3bff, 3, false, 3, 0, true, 0x0000},
    {Operation::LOG, 0x7d01, 0, false, 0, 2, false, 0xff01},
    {Operation::LOG, 0xfd01, 3, true, 2, 1, true, 0x0000},
    {Operation::LOG, 0x8001, 3, false, 0, 1, false, 0xce00},
};

TEST(Rdna4ExpLogF16, DecodedFormsHalfSelectionExecAndModifiers) {
  for (uint32_t width : {32u, 64u}) {
    amdgpu::GpuMemory memory("f16_exp_log_memory");
    amdgpu::L2Cache cache("f16_exp_log_cache");
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("f16_exp_log", cfg, &memory, &cache);
    auto decoder = Decoder::create(cfg.arch);
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(decoder, nullptr);
    auto *wf = cu->dispatch_wf(0, 0, 106, 256, width);
    ASSERT_NE(wf, nullptr);
    const unsigned vb = wf->vgpr_alloc().base, sb = wf->sgpr_alloc().base;
    for (const auto &test : kHalfWitnesses) {
      const bool logarithm = test.operation == Operation::LOG;
      // Vector VOP3, pseudo scalar, VOP1, and vector VOP3 with an SGPR source.
      for (unsigned form = 0; form < 4; ++form) {
        const bool pseudo = form == 1, e32 = form == 2;
        if (e32 && (test.omod || test.modifiers || test.clamp))
          continue;
        for (bool source_high : {false, true}) {
          for (bool destination_high : {false, true}) {
            if (pseudo && destination_high)
              continue;
            const uint32_t src = (pseudo || form == 3) ? 4u : 256u;
            const uint32_t first =
                e32 ? 0x7e000000u | ((6u + (destination_high ? 128u : 0u)) << 17) |
                          ((logarithm ? rdna4::kVLogF16Vop1 : rdna4::kVExpF16Vop1) << 9) |
                          (src + (source_high ? 128u : 0u))
                    : (pseudo ? (logarithm ? 0xd6830005u : 0xd6810005u)
                              : (logarithm ? 0xd5d70006u : 0xd5d80006u)) |
                          ((test.modifiers & 1u) << 8) | (uint32_t{test.clamp} << 15) |
                          (uint32_t{source_high} << 11) | (uint32_t{destination_high} << 14);
            std::array<uint32_t, 4> words{
                first, src | (test.omod << 27) | ((test.modifiers >> 1) << 29), 0, 0};
            auto decoded = decoder->decode(words.data());
            ASSERT_FALSE(decoded.failed());
            std::unique_ptr<Instruction> inst(std::move(decoded).value());
            for (unsigned rounding = 0; rounding < 4; ++rounding) {
              // Deliberately vary the unrelated F32 controls as well.
              const uint32_t mode = (rounding | (rounding << 4)) | (rounding << 2) |
                                    (test.denorm << 6) | (uint32_t{test.overflow} << 23);
              for (uint64_t mask : {uint64_t{0}, ~uint64_t{0}, uint64_t{0xaaaaaaaaaaaaaaaa}}) {
                SCOPED_TRACE(::testing::Message()
                             << "input=" << std::hex << test.input << " form=" << form
                             << " mode=" << mode << " src_hi=" << source_high
                             << " dst_hi=" << destination_high << " mask=" << mask);
                wf->set_mode_raw(mode);
                wf->set_exec(mask);
                const uint32_t input = source_high && !pseudo
                                           ? (uint32_t{test.input} << 16) | 0xcafeu
                                           : 0xcafe0000u | test.input;
                cu->write_sgpr(sb + 4, input);
                cu->write_sgpr(sb + 5, 0xa5a55a5au);
                for (uint32_t lane = 0; lane < width; ++lane) {
                  cu->write_vgpr(vb, lane, input);
                  cu->write_vgpr(vb + 6, lane, 0xa5a55a5au);
                }
                ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
                if (pseudo) {
                  EXPECT_EQ(cu->read_sgpr(sb + 5), test.expected);
                } else {
                  const uint32_t expected = destination_high
                                                ? (uint32_t{test.expected} << 16) | 0x5a5au
                                                : 0xa5a50000u | test.expected;
                  for (uint32_t lane = 0; lane < width; ++lane)
                    EXPECT_EQ(cu->read_vgpr(vb + 6, lane),
                              mask & (uint64_t{1} << lane) ? expected : 0xa5a55a5au);
                }
                EXPECT_EQ(wf->mode_raw(), mode);
                EXPECT_EQ(wf->exec(), mask & (width == 32 ? 0xffffffffull : ~uint64_t{0}));
              }
            }
          }
        }
      }
    }
    wf->halt();
  }
}

TEST(Rdna4ExpLogF16, DppPermutesSelectedHalfAndPreservesMaskedDestinations) {
  constexpr uint16_t inputs[] = {0x11c5, 0x3800, 0x3c01, 0x7d01, 0xbc00, 0xcb01, 0x4c00, 0x1};
  // Captured with preserved F16 denormals, FP16_OVFL and OMOD=/2.
  constexpr uint16_t outputs[2][8] = {
      {0x3801, 0x39a8, 0x3c01, 0x7f01, 0x3400, 0x0, 0x77ff, 0x3800},
      {0xc53c, 0xb800, 0x11c5, 0x7f01, 0xfe00, 0xfe00, 0x4000, 0xca00},
  };
  amdgpu::GpuMemory memory("f16_dpp_memory");
  amdgpu::L2Cache cache("f16_dpp_cache");
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("f16_dpp", cfg, &memory, &cache);
  auto decoder = Decoder::create(cfg.arch);
  auto *wf = cu->dispatch_wf(0, 0, 106, 256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_mode_raw(0xf0u | amdgpu::Wavefront::FP16_OVFL_BIT);
  wf->set_exec(0xffffffffu);
  const uint32_t vb = wf->vgpr_alloc().base;
  for (unsigned op = 0; op < 2; ++op) {
    for (bool dpp8 : {false, true}) {
      std::array<uint32_t, 3> words{};
      if (dpp8) {
        rdna4::Vop3VopDpp8MachineInst raw{};
        raw.vdst = 6;
        raw.op = op ? rdna4::kVLogF16Vop3 : rdna4::kVExpF16Vop3;
        raw.encoding = 0x35;
        raw.src0 = amdgpu::SRC_DPP8_FI_1;
        raw.vsrc0 = 0;
        raw.opsel = 9;
        raw.omod = 3;
        raw.lane_sel_0 = 7;
        raw.lane_sel_1 = 6;
        raw.lane_sel_2 = 5;
        raw.lane_sel_3 = 4;
        raw.lane_sel_4 = 3;
        raw.lane_sel_5 = 2;
        raw.lane_sel_6 = 1;
        raw.lane_sel_7 = 0;
        words = std::bit_cast<decltype(words)>(raw);
      } else {
        rdna4::Vop3VopDpp16MachineInst raw{};
        raw.vdst = 6;
        raw.op = op ? rdna4::kVLogF16Vop3 : rdna4::kVExpF16Vop3;
        raw.encoding = 0x35;
        raw.src0 = amdgpu::SRC_DPP;
        raw.vsrc0 = 0;
        raw.opsel = 9;
        raw.omod = 3;
        raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
        raw.fi = 1;
        raw.row_mask = 5;
        raw.bank_mask = 10;
        words = std::bit_cast<decltype(words)>(raw);
      }
      auto decoded = decoder->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      std::unique_ptr<Instruction> inst(std::move(decoded).value());
      for (unsigned lane = 0; lane < 32; ++lane) {
        cu->write_vgpr(vb, lane, (uint32_t{inputs[lane % 8]} << 16) | 0xcafeu);
        cu->write_vgpr(vb + 6, lane, 0xa5a55a5au);
      }
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      for (unsigned lane = 0; lane < 32; ++lane) {
        const bool writes = dpp8 || (lane < 16 && (10u & (1u << ((lane >> 2) & 3))) && (lane & 15));
        const unsigned source = dpp8 ? 7 - (lane & 7) : (lane - 1) & 7;
        const uint32_t expected =
            writes ? (uint32_t{outputs[op][source]} << 16) | 0x5a5au : 0xa5a55a5au;
        EXPECT_EQ(cu->read_vgpr(vb + 6, lane), expected)
            << "op=" << op << " dpp8=" << dpp8 << " lane=" << lane;
      }
      EXPECT_EQ(wf->exec(), 0xffffffffu);
    }
  }
  wf->halt();
}

} // namespace
