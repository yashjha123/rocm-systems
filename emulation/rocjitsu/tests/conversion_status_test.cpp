// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file conversion_status_test.cpp
/// @brief Exception causes of the measured RDNA4 conversions against gfx1201 captures.
///
/// @details The distilled fixture replays through the conversion.h model and through decoded
/// execution on gfx1201, comparing every destination bit, the sticky status and the causes the
/// instruction reports for trap delivery. Directed tests cover wave64, trap enables, repeated
/// causes, aliasing and the targets and forms that stay unqualified.

#include "decode_test_util.h"
#include "fixtures/conversion_status/gfx1201_cases.h"
#include "legacy_gpu_memory_fixture.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/conversion.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace {

using namespace rocjitsu;
namespace cvt = rocjitsu::amdgpu::conversion;
namespace cause = cvt::conversion_cause;

constexpr uint32_t kSgprs = 106;
constexpr uint32_t kVgprs = 64;
constexpr uint32_t kKernelAddr = 0x1000;
constexpr uint32_t kSEndpgm = 0xBFB00000u;
constexpr uint32_t kSrc0 = 1; // v1, or v[1:2] for a 64-bit source
constexpr uint32_t kSrc1 = 3;
constexpr uint32_t kDst = 5; // v5, or v[5:6] for a 64-bit destination

constexpr uint32_t vgpr(uint32_t index) { return 256 + index; }

// GFX11/GFX12 VOP1: 0x3f in [31:25], vdst[24:17], op[16:9], src0[8:0].
constexpr uint32_t vop1(uint32_t opcode, uint32_t vdst, uint32_t src0) {
  return (0x3fu << 25) | (vdst << 17) | (opcode << 9) | src0;
}

// GFX11/GFX12 VOP2: op[30:25], vdst[24:17], vsrc1[16:9], src0[8:0].
constexpr uint32_t vop2(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return (opcode << 25) | (vdst << 17) | (vsrc1 << 9) | src0;
}

struct Vop3Modifiers {
  uint32_t abs = 0;
  uint32_t neg = 0;
  uint32_t op_sel = 0;
  uint32_t clamp = 0;
  uint32_t omod = 0;
};

// GFX11/GFX12 VOP3: vdst[7:0], abs[10:8], op_sel[14:11], clamp[15], op[25:16],
// encoding 0x35 in [31:26]; src0[8:0], src1[17:9], src2[26:18], omod[28:27],
// neg[31:29].
constexpr std::array<uint32_t, 2> vop3(uint32_t opcode, uint32_t vdst, uint32_t src0, uint32_t src1,
                                       Vop3Modifiers m = {}) {
  return {vdst | (m.abs << 8) | (m.op_sel << 11) | (m.clamp << 15) | (opcode << 16) | (0x35u << 26),
          src0 | (src1 << 9) | (m.omod << 27) | (m.neg << 29)};
}

enum class Width : uint8_t { B16, B32, B64 };

/// How one fixture opcode is encoded and evaluated.
struct OpInfo {
  const char *name;
  uint32_t vop1_or_vop2; ///< 0 when the opcode has no 32-bit encoding.
  bool vop2;
  uint32_t vop3;
  Width src0;
  bool has_src1;
  Width dst;
};

OpInfo op_info(CvtOp op) {
  using W = Width;
  switch (op) {
  case CvtOp::kCvtF16F32:
    return {"V_CVT_F16_F32", rdna4::kVCvtF16F32Vop1, false, rdna4::kVCvtF16F32Vop3, W::B32, false,
            W::B16};
  case CvtOp::kCvtF16I16:
    return {"V_CVT_F16_I16", rdna4::kVCvtF16I16Vop1, false, rdna4::kVCvtF16I16Vop3, W::B16, false,
            W::B16};
  case CvtOp::kCvtF16U16:
    return {"V_CVT_F16_U16", rdna4::kVCvtF16U16Vop1, false, rdna4::kVCvtF16U16Vop3, W::B16, false,
            W::B16};
  case CvtOp::kCvtF32Bf8:
    return {"V_CVT_F32_BF8", rdna4::kVCvtF32Bf8Vop1, false, rdna4::kVCvtF32Bf8Vop3, W::B32, false,
            W::B32};
  case CvtOp::kCvtF32F16:
    return {"V_CVT_F32_F16", rdna4::kVCvtF32F16Vop1, false, rdna4::kVCvtF32F16Vop3, W::B16, false,
            W::B32};
  case CvtOp::kCvtF32F64:
    return {"V_CVT_F32_F64", rdna4::kVCvtF32F64Vop1, false, rdna4::kVCvtF32F64Vop3, W::B64, false,
            W::B32};
  case CvtOp::kCvtF32Fp8:
    return {"V_CVT_F32_FP8", rdna4::kVCvtF32Fp8Vop1, false, rdna4::kVCvtF32Fp8Vop3, W::B32, false,
            W::B32};
  case CvtOp::kCvtF32I32:
    return {"V_CVT_F32_I32", rdna4::kVCvtF32I32Vop1, false, rdna4::kVCvtF32I32Vop3, W::B32, false,
            W::B32};
  case CvtOp::kCvtF32U32:
    return {"V_CVT_F32_U32", rdna4::kVCvtF32U32Vop1, false, rdna4::kVCvtF32U32Vop3, W::B32, false,
            W::B32};
  case CvtOp::kCvtF64F32:
    return {"V_CVT_F64_F32", rdna4::kVCvtF64F32Vop1, false, rdna4::kVCvtF64F32Vop3, W::B32, false,
            W::B64};
  case CvtOp::kCvtF64I32:
    return {"V_CVT_F64_I32", rdna4::kVCvtF64I32Vop1, false, rdna4::kVCvtF64I32Vop3, W::B32, false,
            W::B64};
  case CvtOp::kCvtFloorI32F32:
    return {"V_CVT_FLOOR_I32_F32",
            rdna4::kVCvtFloorI32F32Vop1,
            false,
            rdna4::kVCvtFloorI32F32Vop3,
            W::B32,
            false,
            W::B32};
  case CvtOp::kCvtI16F16:
    return {"V_CVT_I16_F16", rdna4::kVCvtI16F16Vop1, false, rdna4::kVCvtI16F16Vop3, W::B16, false,
            W::B16};
  case CvtOp::kCvtI32F32:
    return {"V_CVT_I32_F32", rdna4::kVCvtI32F32Vop1, false, rdna4::kVCvtI32F32Vop3, W::B32, false,
            W::B32};
  case CvtOp::kCvtI32F64:
    return {"V_CVT_I32_F64", rdna4::kVCvtI32F64Vop1, false, rdna4::kVCvtI32F64Vop3, W::B64, false,
            W::B32};
  case CvtOp::kCvtNearestI32F32:
    return {"V_CVT_NEAREST_I32_F32",
            rdna4::kVCvtNearestI32F32Vop1,
            false,
            rdna4::kVCvtNearestI32F32Vop3,
            W::B32,
            false,
            W::B32};
  case CvtOp::kCvtPkBf8F32:
    return {"V_CVT_PK_BF8_F32", 0, false, rdna4::kVCvtPkBf8F32Vop3, W::B32, true, W::B32};
  case CvtOp::kCvtPkFp8F32:
    return {"V_CVT_PK_FP8_F32", 0, false, rdna4::kVCvtPkFp8F32Vop3, W::B32, true, W::B32};
  case CvtOp::kCvtPkNormI16F32:
    return {"V_CVT_PK_NORM_I16_F32", 0, false, rdna4::kVCvtPkNormI16F32Vop3, W::B32, true, W::B32};
  case CvtOp::kCvtPkNormU16F32:
    return {"V_CVT_PK_NORM_U16_F32", 0, false, rdna4::kVCvtPkNormU16F32Vop3, W::B32, true, W::B32};
  case CvtOp::kCvtPkRtzF16F32:
    return {"V_CVT_PK_RTZ_F16_F32",
            rdna4::kVCvtPkRtzF16F32Vop2,
            true,
            rdna4::kVCvtPkRtzF16F32Vop3,
            W::B32,
            true,
            W::B32};
  case CvtOp::kCvtSrBf8F32:
    return {"V_CVT_SR_BF8_F32", 0, false, rdna4::kVCvtSrBf8F32Vop3, W::B32, true, W::B32};
  case CvtOp::kCvtSrFp8F32:
    return {"V_CVT_SR_FP8_F32", 0, false, rdna4::kVCvtSrFp8F32Vop3, W::B32, true, W::B32};
  case CvtOp::kCvtU16F16:
    return {"V_CVT_U16_F16", rdna4::kVCvtU16F16Vop1, false, rdna4::kVCvtU16F16Vop3, W::B16, false,
            W::B16};
  case CvtOp::kCvtU32F32:
    return {"V_CVT_U32_F32", rdna4::kVCvtU32F32Vop1, false, rdna4::kVCvtU32F32Vop3, W::B32, false,
            W::B32};
  case CvtOp::kCvtU32F64:
    return {"V_CVT_U32_F64", rdna4::kVCvtU32F64Vop1, false, rdna4::kVCvtU32F64Vop3, W::B64, false,
            W::B32};
  }
  return {};
}

struct LaneModel {
  uint64_t bits;       ///< Converted value, before it is merged into the destination.
  uint32_t causes = 0; ///< Causes under the measured rule.
  bool bits_modeled = true;
};

/// The conversion.h evaluation of one lane, as the generated RDNA4 bodies call it.
LaneModel model_lane(const CvtStatusRow &row, uint64_t s0, uint32_t s1) {
  const cvt::Mode m{row.mode & 3u, (row.mode >> 2) & 3u, (row.mode >> 4) & 3u, (row.mode >> 6) & 3u,
                    ((row.mode >> 23) & 1u) != 0};
  const cvt::Modifiers mods{row.abs, row.neg, row.clamp != 0, row.omod};
  const auto rule = [&](cvt::CauseRule r, uint64_t bits, uint32_t facts) {
    return LaneModel{bits, cvt::conversion_causes(r, facts, mods)};
  };
  const auto flt = [&](cvt::Format from, cvt::Format to) {
    const auto e = cvt::evaluate_float(s0, from, to, mods, m);
    return rule(cvt::CauseRule::FLOAT, e.bits, e.facts);
  };
  const auto itf = [&](int64_t v, cvt::Format to) {
    const auto e = cvt::evaluate_integer(v, to, mods, m);
    return rule(cvt::CauseRule::INTEGER_TO_FLOAT, e.bits, e.facts);
  };
  const auto fti = [&](cvt::Format from, int64_t lo, int64_t hi, uint64_t mask) {
    const auto e = cvt::evaluate_to_integer(s0, from, 0, mods, m, cvt::IntegerRounding::TRUNCATE,
                                            lo, hi, false);
    return rule(cvt::CauseRule::FLOAT_TO_INTEGER, static_cast<uint64_t>(e.bits) & mask, e.facts);
  };
  const auto nearest = [&](cvt::IntegerRounding how) {
    return LaneModel{static_cast<uint32_t>(
        cvt::convert_to_integer(s0, cvt::F32, 0, mods, m, how, INT32_MIN, INT32_MAX, true))};
  };
  switch (row.op) {
  case CvtOp::kCvtF16F32:
    return flt(cvt::F32, cvt::F16);
  case CvtOp::kCvtF32F16:
    return flt(cvt::F16, cvt::F32);
  case CvtOp::kCvtF32F64:
    return flt(cvt::F64, cvt::F32);
  case CvtOp::kCvtF64F32:
    return flt(cvt::F32, cvt::F64);
  case CvtOp::kCvtF16I16:
    return itf(static_cast<int16_t>(s0), cvt::F16);
  case CvtOp::kCvtF16U16:
    return itf(static_cast<uint16_t>(s0), cvt::F16);
  case CvtOp::kCvtF32I32:
    return itf(static_cast<int32_t>(s0), cvt::F32);
  case CvtOp::kCvtF32U32:
    return itf(static_cast<uint32_t>(s0), cvt::F32);
  case CvtOp::kCvtF64I32:
    return {cvt::convert_integer(static_cast<int32_t>(s0), cvt::F64, mods, m)};
  case CvtOp::kCvtI32F32:
    return fti(cvt::F32, INT32_MIN, INT32_MAX, 0xffffffffu);
  case CvtOp::kCvtU32F32:
    return fti(cvt::F32, 0, UINT32_MAX, 0xffffffffu);
  case CvtOp::kCvtI32F64:
    return fti(cvt::F64, INT32_MIN, INT32_MAX, 0xffffffffu);
  case CvtOp::kCvtU32F64:
    return fti(cvt::F64, 0, UINT32_MAX, 0xffffffffu);
  case CvtOp::kCvtI16F16:
    return fti(cvt::F16, -32768, 32767, 0xffffu);
  case CvtOp::kCvtU16F16:
    return fti(cvt::F16, 0, 65535, 0xffffu);
  case CvtOp::kCvtFloorI32F32:
    return nearest(cvt::IntegerRounding::FLOOR);
  case CvtOp::kCvtNearestI32F32:
    return nearest(cvt::IntegerRounding::NEAREST_UP);
  case CvtOp::kCvtPkRtzF16F32:
    return {cvt::pack_rtz_f16(static_cast<uint32_t>(s0), s1, mods, m)};
  case CvtOp::kCvtF32Fp8:
    return {cvt::decode_fp8(static_cast<uint8_t>(s0), cvt::FP8)};
  case CvtOp::kCvtF32Bf8:
    return {cvt::decode_fp8(static_cast<uint8_t>(s0), cvt::BF8)};
  case CvtOp::kCvtPkFp8F32:
  case CvtOp::kCvtPkBf8F32: {
    const auto e =
        cvt::evaluate_pack_fp8(static_cast<uint32_t>(s0), s1,
                               row.op == CvtOp::kCvtPkFp8F32 ? cvt::FP8 : cvt::BF8, mods, m);
    return rule(cvt::CauseRule::FP8, e.bits, e.facts);
  }
  case CvtOp::kCvtSrFp8F32:
  case CvtOp::kCvtSrBf8F32: {
    const auto e = cvt::evaluate_fp8(static_cast<uint32_t>(s0),
                                     row.op == CvtOp::kCvtSrFp8F32 ? cvt::FP8 : cvt::BF8, 0, mods,
                                     m, true, s1);
    return rule(cvt::CauseRule::FP8, e.bits, e.facts);
  }
  case CvtOp::kCvtPkNormI16F32:
  case CvtOp::kCvtPkNormU16F32:
    // Measured silent; the portable body computes these bits.
    return {0, 0, false};
  }
  return {0, 0, false};
}

/// The captured destination bits the model's value occupies.
uint64_t captured_field(const CvtStatusRow &row, const OpInfo &info, uint64_t destination) {
  switch (row.op) {
  case CvtOp::kCvtPkFp8F32:
  case CvtOp::kCvtPkBf8F32:
    return (row.op_sel & 0x8u) ? (destination >> 16) & 0xffffu : destination & 0xffffu;
  case CvtOp::kCvtSrFp8F32:
  case CvtOp::kCvtSrBf8F32:
    return (destination >> (((row.op_sel >> 2) & 3u) * 8u)) & 0xffu;
  default:
    return info.dst == Width::B16 ? destination & 0xffffu : destination;
  }
}

struct LaneInput {
  uint64_t src0;
  uint32_t src1;
  std::optional<uint64_t> result; ///< Captured destination of a listed lane.
};

LaneInput lane_input(const CvtStatusRow &row, uint32_t lane) {
  for (uint32_t i = 0; i < row.lane_count; ++i) {
    const CvtStatusLane &l = kGfx1201CvtStatusLanes[row.first_lane + i];
    if (l.lane == lane)
      return {l.src0, l.src1, l.result};
  }
  return {row.fill0, row.fill1, std::nullopt};
}

uint64_t expected_destination(const CvtStatusRow &row, uint32_t lane) {
  const LaneInput in = lane_input(row, lane);
  if (in.result)
    return *in.result;
  return ((row.exec >> lane) & 1u) && row.has_fill_result ? row.fill_result : row.dst_init;
}

TEST(ConversionStatus, ModelMatchesGfx1201Captures) {
  for (const CvtStatusRow &row : kGfx1201CvtStatusRows) {
    const OpInfo info = op_info(row.op);
    SCOPED_TRACE(testing::Message() << info.name << " vop3 " << int(row.vop3) << " mode 0x"
                                    << std::hex << row.mode << " exec 0x" << row.exec);
    uint32_t causes = 0;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      if (!((row.exec >> lane) & 1u))
        continue;
      const LaneInput in = lane_input(row, lane);
      const LaneModel model = model_lane(row, in.src0, in.src1);
      causes |= model.causes;
      if (model.bits_modeled) {
        EXPECT_EQ(model.bits, captured_field(row, info, expected_destination(row, lane)))
            << "lane " << std::dec << lane << " src 0x" << std::hex << in.src0;
      }
    }
    EXPECT_EQ(row.status_before | causes, row.status_after);
    if (HasFailure())
      return;
  }
}

struct CvtFixture {
  test::LegacyGpuMemoryFixture gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;

  explicit CvtFixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                      rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201)
      : gpu_mem("conversion_status_mem"), l2("conversion_status_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.target = target;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = kSgprs;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("conversion_status_cu", cfg, &gpu_mem, &l2);
    l2.set_gpu_vm(&gpu_mem.gpu_vm());
    cu->set_gpu_vm(&gpu_mem.gpu_vm());
    decoder = Decoder::create(arch);
  }

  amdgpu::Wavefront *dispatch(uint32_t wave_size, uint64_t pc = 0) {
    return cu->dispatch_wf(0, pc, kSgprs, kVgprs, wave_size);
  }

  void write(const amdgpu::Wavefront &wf, uint32_t reg, uint32_t lane, uint64_t value, bool wide) {
    cu->write_vgpr(wf.vgpr_alloc().base + reg, lane, static_cast<uint32_t>(value));
    if (wide)
      cu->write_vgpr(wf.vgpr_alloc().base + reg + 1, lane, static_cast<uint32_t>(value >> 32));
  }

  uint64_t read(const amdgpu::Wavefront &wf, uint32_t reg, uint32_t lane, bool wide) const {
    uint64_t value = cu->read_vgpr_storage(wf.vgpr_alloc().base + reg, lane);
    if (wide)
      value |= uint64_t{cu->read_vgpr_storage(wf.vgpr_alloc().base + reg + 1, lane)} << 32;
    return value;
  }

  /// Decode and execute one instruction; returns the causes it reported.
  uint32_t execute(amdgpu::Wavefront &wf, std::span<const uint32_t> words) {
    std::array<uint32_t, 4> buffer{};
    std::copy(words.begin(), words.end(), buffer.begin());
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, buffer.data()));
    EXPECT_NE(inst, nullptr);
    if (!inst)
      return 0;
    wf.clear_pending_alu_causes();
    EXPECT_TRUE(cu->execute_instruction(inst.get(), wf).succeeded());
    return wf.pending_alu_causes();
  }
};

std::array<uint32_t, 3> encode(const CvtStatusRow &row, const OpInfo &info) {
  if (!row.vop3) {
    if (info.vop2)
      return {vop2(info.vop1_or_vop2, kDst, kSrc1, vgpr(kSrc0)), 0, 0};
    return {vop1(info.vop1_or_vop2, kDst, vgpr(kSrc0)), 0, 0};
  }
  const auto w = vop3(info.vop3, kDst, vgpr(kSrc0), info.has_src1 ? vgpr(kSrc1) : 0,
                      {row.abs, row.neg, row.op_sel, row.clamp, row.omod});
  return {w[0], w[1], 0};
}

TEST(ConversionStatus, DecodedExecutionMatchesGfx1201Captures) {
  CvtFixture f;
  amdgpu::Wavefront *wf = f.dispatch(32);
  ASSERT_NE(wf, nullptr);
  for (const CvtStatusRow &row : kGfx1201CvtStatusRows) {
    const OpInfo info = op_info(row.op);
    SCOPED_TRACE(testing::Message()
                 << info.name << " vop3 " << int(row.vop3) << " abs " << int(row.abs) << " neg "
                 << int(row.neg) << " clamp " << int(row.clamp) << " omod " << int(row.omod)
                 << " op_sel " << int(row.op_sel) << " mode 0x" << std::hex << row.mode
                 << " exec 0x" << row.exec);
    const bool wide_src = info.src0 == Width::B64;
    const bool wide_dst = info.dst == Width::B64;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      const LaneInput in = lane_input(row, lane);
      f.write(*wf, kSrc0, lane, in.src0, wide_src);
      f.write(*wf, kSrc1, lane, in.src1, false);
      f.write(*wf, kDst, lane, row.dst_init, wide_dst);
    }
    wf->set_mode_raw(row.mode);
    wf->set_exec(row.exec);
    wf->set_trapsts(row.status_before);
    const uint32_t reported = f.execute(*wf, encode(row, info));
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(f.read(*wf, kDst, lane, wide_dst), expected_destination(row, lane))
          << "lane " << std::dec << lane;
    EXPECT_EQ(wf->trapsts() & 0x7fu, row.status_after);
    EXPECT_EQ(row.status_before | reported, row.status_after);
    if (HasFailure())
      return;
  }
}

constexpr uint32_t kModeKeepAll = 0xf0u; // RNE, every denormal kept

struct Scenario {
  CvtFixture f;
  amdgpu::Wavefront *wf;

  explicit Scenario(uint32_t wave_size) : wf(f.dispatch(wave_size)) {
    wf->set_mode_raw(kModeKeepAll);
  }

  void fill(uint32_t reg, uint32_t value) {
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      f.write(*wf, reg, lane, value, false);
  }
};

// Causes are the union over participating lanes of a wave64, including the upper half.
TEST(ConversionStatus, Wave64ReducesCausesOverParticipatingLanes) {
  Scenario s(64);
  ASSERT_NE(s.wf, nullptr);
  s.fill(kSrc0, 0x3f800000u); // 1.0: exact
  s.fill(kDst, 0xa5a5a5a5u);
  s.f.write(*s.wf, kSrc0, 40, 0x7f800001u, false); // signaling NaN
  s.f.write(*s.wf, kSrc0, 63, 0x477ff000u, false); // overflows F16
  s.f.write(*s.wf, kSrc0, 17, 0x33000001u, false); // tiny, inexact
  const uint32_t words[] = {vop1(rdna4::kVCvtF16F32Vop1, kDst, vgpr(kSrc0))};
  struct Case {
    uint64_t exec;
    uint32_t causes;
  };
  const Case cases[] = {
      {0, 0},
      {~uint64_t{0} & ~(uint64_t{1} << 40) & ~(uint64_t{1} << 63) & ~(uint64_t{1} << 17), 0},
      {uint64_t{1} << 40, cause::kInvalid},
      {uint64_t{1} << 63, cause::kOverflow | cause::kInexact},
      {uint64_t{1} << 17, cause::kUnderflow | cause::kInexact},
      {~uint64_t{0}, cause::kInvalid | cause::kOverflow | cause::kUnderflow | cause::kInexact},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(testing::Message() << "exec 0x" << std::hex << c.exec);
    s.fill(kDst, 0xa5a5a5a5u);
    s.wf->set_exec(c.exec);
    s.wf->set_trapsts(0x40u);
    EXPECT_EQ(s.f.execute(*s.wf, words), c.causes);
    EXPECT_EQ(s.wf->trapsts(), 0x40u | c.causes);
    for (uint32_t lane = 0; lane < 64; ++lane)
      if (!((c.exec >> lane) & 1u))
        EXPECT_EQ(s.f.read(*s.wf, kDst, lane, false), 0xa5a5a5a5u) << "lane " << lane;
  }
}

// Modifiers change which causes a lane raises; each source of a packed form contributes.
TEST(ConversionStatus, ModifiersAndPackedSourcesSelectCauses) {
  struct Case {
    const char *name;
    uint32_t opcode;
    Vop3Modifiers mods;
    uint32_t src0;
    uint32_t src1;
    uint32_t causes;
    bool two_sources = false;
  };
  const Case cases[] = {
      {"CLAMP hides a signaling NaN", rdna4::kVCvtF16F32Vop3, {.clamp = 1}, 0x7f800001u, 0, 0},
      {"OMOD hides inexact", rdna4::kVCvtF16F32Vop3, {.omod = 1}, 0x3f800001u, 0, 0},
      {"OMOD overflow alone",
       rdna4::kVCvtF16F32Vop3,
       {.omod = 1},
       0x477fe000u,
       0,
       cause::kOverflow},
      {"NEG makes an unsigned source invalid",
       rdna4::kVCvtU32F32Vop3,
       {.neg = 1},
       0x3f800000u,
       0,
       cause::kInvalid},
      {"ABS keeps it valid", rdna4::kVCvtU32F32Vop3, {.abs = 1}, 0xbf800000u, 0, 0},
      {"CLAMP reports truncation",
       rdna4::kVCvtI32F32Vop3,
       {.clamp = 1},
       0x40200000u,
       0,
       cause::kInexact},
      {"truncation alone is silent", rdna4::kVCvtI32F32Vop3, {}, 0x40200000u, 0, 0},
      {"packed high source",
       rdna4::kVCvtPkFp8F32Vop3,
       {},
       0x3f800000u,
       0x7f800001u,
       cause::kInvalid,
       true},
      {"packed low subnormal",
       rdna4::kVCvtPkBf8F32Vop3,
       {},
       0x00000001u,
       0x3f800000u,
       cause::kInputDenormal,
       true},
      {"packed overflow is silent",
       rdna4::kVCvtPkFp8F32Vop3,
       {},
       0x447a0000u,
       0x3f800000u,
       0,
       true},
      {"RTZ pack is silent", rdna4::kVCvtPkRtzF16F32Vop3, {}, 0x7f800001u, 0x477ff000u, 0, true},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.name);
    Scenario s(32);
    ASSERT_NE(s.wf, nullptr);
    s.fill(kSrc0, c.src0);
    s.fill(kSrc1, c.src1);
    s.wf->set_exec(0x0000'8001u);
    s.wf->set_trapsts(0);
    const auto w = vop3(c.opcode, kDst, vgpr(kSrc0), c.two_sources ? vgpr(kSrc1) : 0, c.mods);
    EXPECT_EQ(s.f.execute(*s.wf, w), c.causes);
    EXPECT_EQ(s.wf->trapsts(), c.causes);
  }
}

// Sources are read before the destination is written, so aliasing keeps the lane values.
TEST(ConversionStatus, AliasedDestinationKeepsResultsAndCauses) {
  Scenario s(32);
  ASSERT_NE(s.wf, nullptr);
  for (uint32_t lane = 0; lane < 32; ++lane)
    s.f.write(*s.wf, 0, lane, lane == 3 ? 0x47f0000000000000ull : 0x3ff8000000000000ull, true);
  s.wf->set_exec(0xffff'ffffu);
  s.wf->set_trapsts(0);
  // v1 = cvt_f32_f64(v[0:1]): the destination overlaps the high half of the source.
  const uint32_t words[] = {vop1(rdna4::kVCvtF32F64Vop1, 1, vgpr(0))};
  EXPECT_EQ(s.f.execute(*s.wf, words), cause::kOverflow | cause::kInexact);
  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(s.f.read(*s.wf, 1, lane, false), lane == 3 ? 0x7f800000u : 0x3fc00000u)
        << "lane " << lane;
}

// Trap delivery follows the causes of the current instruction; sticky status accumulates
// whether or not the cause is enabled in TRAP_CTRL.
TEST(ConversionStatus, TrapDeliveryUsesCurrentCausesWithStickyStatus) {
  CvtFixture f;
  const uint32_t overflow = vop1(rdna4::kVCvtF16F32Vop1, 2, vgpr(0));
  const uint32_t exact = vop1(rdna4::kVCvtF16F32Vop1, 3, vgpr(1));
  const uint32_t program[] = {overflow, overflow, overflow, exact, kSEndpgm};
  for (uint32_t i = 0; i < std::size(program); ++i)
    f.gpu_mem.write32(kKernelAddr + 4 * i, program[i]);
  amdgpu::Wavefront *wf = f.dispatch(32, kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_mode_raw(kModeKeepAll);
  wf->set_exec(1);
  const uint32_t base = wf->vgpr_alloc().base;
  f.cu->write_vgpr(base, 0, 0x477ff000u);
  f.cu->write_vgpr(base + 1, 0, 0x3f800000u);
  uint32_t handler_calls = 0;
  f.cu->set_alu_exception_handler([&](amdgpu::Wavefront &) {
    ++handler_calls;
    return false;
  });
  constexpr uint32_t kOverflowInexact = cause::kOverflow | cause::kInexact;

  // Enabled overflow with the sticky bit already latched.
  wf->set_trapsts(cause::kOverflow);
  wf->set_gfx12_trap_ctrl_raw(cause::kOverflow);
  f.cu->step();
  EXPECT_EQ(handler_calls, 1u);
  EXPECT_EQ(wf->pending_alu_causes(), kOverflowInexact);
  EXPECT_EQ(wf->trapsts() & 0x7fu, kOverflowInexact);
  EXPECT_EQ(f.cu->read_vgpr_storage(base + 2, 0) & 0xffffu, 0x7c00u);

  // A disabled repeat is sticky only.
  wf->set_gfx12_trap_ctrl_raw(0);
  f.cu->step();
  EXPECT_EQ(handler_calls, 1u);
  EXPECT_EQ(wf->pending_alu_causes(), kOverflowInexact);

  // Re-enabling delivers the next occurrence.
  wf->set_gfx12_trap_ctrl_raw(cause::kOverflow);
  f.cu->step();
  EXPECT_EQ(handler_calls, 2u);

  // A latched, enabled cause that this exact conversion does not raise is not an event.
  wf->set_gfx12_trap_ctrl_raw(cause::kInexact);
  f.cu->step();
  EXPECT_EQ(handler_calls, 2u);
  EXPECT_EQ(wf->pending_alu_causes(), 0u);
  EXPECT_EQ(wf->trapsts() & 0x7fu, kOverflowInexact);
  EXPECT_EQ(f.cu->read_vgpr_storage(base + 3, 0) & 0xffffu, 0x3c00u);
}

// Only gfx1201 status is measured. Other architectures keep their conversions, which report no
// causes, and measured RDNA4 forms without a status capture report none either.
TEST(ConversionStatus, UnqualifiedTargetsAndFormsReportNoCauses) {
  {
    CvtFixture f(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_TARGET_GFX1100);
    amdgpu::Wavefront *wf = f.dispatch(32);
    ASSERT_NE(wf, nullptr);
    wf->set_mode_raw(kModeKeepAll);
    for (uint32_t lane = 0; lane < 32; ++lane)
      f.write(*wf, kSrc0, lane, 0x7f800001u, false);
    wf->set_exec(0xffff'ffffu);
    wf->set_trapsts(0);
    const uint32_t words[] = {vop1(rdna3::kVCvtF16F32Vop1, kDst, vgpr(kSrc0))};
    EXPECT_EQ(f.execute(*wf, words), 0u);
    EXPECT_EQ(wf->trapsts() & 0x7fu, 0u);
  }
  Scenario s(32);
  ASSERT_NE(s.wf, nullptr);
  s.fill(kSrc0, 0x7f800001u);
  s.fill(kSrc1, 0x4f800000u);
  s.wf->set_exec(0xffff'ffffu);
  s.wf->set_trapsts(0);
  const auto w = vop3(rdna4::kVCvtPkU16F32Vop3, kDst, vgpr(kSrc0), vgpr(kSrc1));
  EXPECT_EQ(s.f.execute(*s.wf, w), 0u);
  EXPECT_EQ(s.wf->trapsts() & 0x7fu, 0u);
}

} // namespace
