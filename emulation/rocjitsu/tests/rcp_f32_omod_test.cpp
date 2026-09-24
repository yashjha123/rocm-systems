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
#include <utility>

using namespace rocjitsu;

// Expected bits captured on gfx1201 with V_RCP_F32 and V_S_RCP_F32.
TEST(RcpF32Omod, Div2PreservesUnderflowSign) {
  amdgpu::GpuMemory memory("rcp_omod_memory");
  amdgpu::L2Cache cache("rcp_omod_cache");
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rcp_omod", cfg, &memory, &cache);
  auto decoder = Decoder::create(cfg.arch);
  auto *wf = cu->dispatch_wf(0, 0, 106, 256, 32);
  const unsigned vgpr_base = wf->vgpr_alloc().base;
  const unsigned sgpr_base = wf->sgpr_alloc().base;

  // Negative source modifier followed by /2: normal, boundary, underflow,
  // already-zero reciprocal, and infinity input.
  const std::array<std::pair<uint32_t, uint32_t>, 5> cases{{
      {0x7e000000u, 0x80800000u},
      {0x7e000001u, 0x80000000u},
      {0x7e800000u, 0x80000000u},
      {0x7e800001u, 0x00000000u},
      {0x7f800000u, 0x00000000u},
  }};
  for (bool pseudo_scalar : {false, true}) {
    std::array<uint32_t, 4> words{
        pseudo_scalar ? 0xd6840005u : 0xd5aa0006u,
        pseudo_scalar ? 0x38000004u : 0x38000100u,
        0,
        0,
    };
    auto decoded = decoder->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    std::unique_ptr<Instruction> instruction(std::move(decoded).value());
    for (uint32_t mode : {0u, 0xffu, 0x300u, 0x3ffu}) {
      for (auto [input, expected] : cases) {
        wf->set_mode_raw(mode);
        wf->set_exec(~0ull);
        cu->write_sgpr(sgpr_base + 4, input);
        for (unsigned lane = 0; lane < 32; ++lane)
          cu->write_vgpr(vgpr_base, lane, input);
        ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
        const uint32_t got =
            pseudo_scalar ? cu->read_sgpr(sgpr_base + 5) : cu->read_vgpr(vgpr_base + 6, 0);
        EXPECT_EQ(got, expected) << "pseudo_scalar=" << pseudo_scalar << " input=" << std::hex
                                 << input << " mode=" << mode;
      }
    }
  }
  wf->halt();
}
