// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file Vector execution of the bit-exact RDNA3/RDNA4 unary transcendental pipeline.

#include "rocjitsu/isa/arch/amdgpu/shared/alu_exceptions.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/transcendental_valu.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu::transcendental {

/// @brief Execute a VOP1 or VOP3 V_EXP/V_LOG/V_RCP/V_RSQ/V_SQRT F16 or F32 instruction.
/// @details F16 forms read and write the true16 half selected by the e32 register encoding or
/// VOP3 OP_SEL and preserve the other destination half. SIMD lanes evaluate the same scalar
/// pipeline, so both paths produce identical bits.
template <Operation Op, bool F16, bool E32, typename Inst>
inline void execute_valu(Inst &inst, Wavefront &wf) {
  bool absolute = false, negate = false, clamp = false;
  uint32_t omod = 0;
  if constexpr (!E32) {
    absolute = (inst.inst_.abs & 1u) != 0;
    negate = (inst.inst_.neg & 1u) != 0;
    clamp = inst.inst_.clamp != 0;
    omod = F16 ? fp_mode::effective_f16_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(),
                                             wf.ieee_mode(), false, inst.inst_.omod)
               : fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f32(), wf.ieee_mode(),
                                         inst.inst_.omod);
  }
  const bool clamp_nan_to_zero = floating_clamp_nan_to_zero(wf);
  const uint32_t denorm_mode = wf.fp_denorm_mode_f16_f64();
  const bool fp16_ovfl = wf.fp16_ovfl();
  const auto evaluate = [=](uint32_t source) -> uint32_t {
    if constexpr (F16)
      return execute_f16(Op, static_cast<uint16_t>(source), absolute, negate, omod, clamp,
                         denorm_mode, fp16_ovfl, clamp_nan_to_zero);
    else
      return execute_f32(Op, source, absolute, negate, omod, clamp, clamp_nan_to_zero);
  };
  uint32_t trap_enables = 0;
  if constexpr (Op == Operation::SQRT && !F16) {
    // A negative source raises the invalid cause; latch it before any fast-path return.
    uint32_t alu_causes = 0;
    if constexpr (E32)
      alu_causes = classify_sqrt_f32_vop1(inst, wf);
    else
      alu_causes = classify_sqrt_f32_vop3(inst, wf);
    wf.set_trapsts(wf.trapsts() | alu_causes);
    trap_enables = alu_exception_trap_enables(wf);
  }
  if (!trap_enables && try_execute_words_simd < 1, E32, F16,
      F16 ? 1u : 0u > (inst, wf, [&](auto words) {
                   return util::map_native_convert_scalar<uint32_t, uint32_t>(words, evaluate);
                 }))
    return;

  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    if constexpr (F16 && !E32) {
      const uint32_t opsel = vop3_opsel(inst.inst_);
      write_vop3_true16_dst(inst.vdst, wf, lane, opsel,
                            evaluate(read_vop3_true16_src(inst.src0, wf, lane, opsel, 0)), true);
    } else {
      const uint32_t source = RegisterAccess(wf).read_lane(inst.src0, lane);
      if constexpr (F16)
        sdwa::write_lane<sdwa::ResultFormat::F16>(inst, wf, inst.vdst, lane,
                                                  evaluate(source & 0xffffu));
      else
        sdwa::write_lane<sdwa::ResultFormat::F32>(inst, wf, inst.vdst, lane, evaluate(source));
    }
  }
}

} // namespace rocjitsu::amdgpu::transcendental
