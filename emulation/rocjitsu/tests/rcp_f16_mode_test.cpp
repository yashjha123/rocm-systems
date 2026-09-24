#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

using namespace rocjitsu;

// Low-half result bits captured on physical gfx1201 for both VOP3 forms.
TEST(RcpF16Mode, VectorAndPseudoScalarMatchHardware) {
  amdgpu::GpuMemory memory("rcp_f16_mode_memory");
  amdgpu::L2Cache cache("rcp_f16_mode_cache");
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rcp_f16_mode", cfg, &memory, &cache);
  auto decoder = Decoder::create(cfg.arch);
  auto *wf = cu->dispatch_wf(0, 0, 106, 256, 32);
  const unsigned vb = wf->vgpr_alloc().base, sb = wf->sgpr_alloc().base;

  struct Case {
    uint32_t mode_index;
    uint32_t omod;
    uint16_t input;
    uint16_t expected;
  };
  const std::array cases{
      Case{0, 0, 0x0101, 0x7c00}, // Flush input subnormal before reciprocal.
      Case{0, 0, 0x7401, 0x0000}, // Flush output subnormal.
      Case{0, 1, 0x7401, 0x0000}, // Flush intermediate before scaling.
      Case{0, 0, 0x4200, 0x3555}, // Nearest-even reciprocal of 3.
      Case{1, 0, 0x4200, 0x3555}, // RCP ignores directed result rounding.
      Case{2, 0, 0xc200, 0xb555},  Case{3, 0, 0xc200, 0xb555},
      Case{12, 0, 0x0101, 0x7bf8}, // Preserve input/output subnormals.
      Case{0, 3, 0xf001, 0x8000},  // Scaling underflow retains its sign.
      Case{0, 1, 0x7c00, 0x0000},  // An already-zero reciprocal becomes +0.
      Case{15, 3, 0x7d00, 0x7f00}, // Quiet signaling NaN, retaining payload.
  };

  for (bool pseudo_scalar : {false, true}) {
    for (const auto &test : cases) {
      const uint32_t mode = 0x30u | ((test.mode_index & 3u) << 2) | ((test.mode_index >> 2) << 6);
      std::array<uint32_t, 4> words{
          pseudo_scalar ? 0xd6850005u : 0xd5d40006u,
          (pseudo_scalar ? 4u : 256u) | (test.omod << 27),
          0,
          0,
      };
      auto decoded = decoder->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      std::unique_ptr<Instruction> instruction(std::move(decoded).value());
      wf->set_mode_raw(mode);
      wf->set_exec(~0ull);
      cu->write_sgpr(sb + 4, 0xcafe0000u | test.input);
      for (unsigned lane = 0; lane < 32; ++lane)
        cu->write_vgpr(vb, lane, 0xcafe0000u | test.input);
      ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
      const uint16_t got =
          static_cast<uint16_t>(pseudo_scalar ? cu->read_sgpr(sb + 5) : cu->read_vgpr(vb + 6, 0));
      EXPECT_EQ(got, test.expected) << "pseudo_scalar=" << pseudo_scalar << " mode=" << mode
                                    << " omod=" << test.omod << " input=" << std::hex << test.input;
    }
  }
  wf->halt();
}
