// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
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
} // namespace
