// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/vm/amdgpu/register_access.h"

namespace rocjitsu::amdgpu::rdna4_exp_log {

/// @brief Execute vector EXP/LOG with the existing DPP, true16 and EXEC routing.
template <bool Logarithm, bool E32, typename Inst>
inline void execute_vector_f16(Inst &inst, Wavefront &wf) {
  bool absolute = false, negate = false, clamp = false;
  uint32_t omod = 0;
  if constexpr (!E32) {
    absolute = (inst.inst_.abs & 1u) != 0;
    negate = (inst.inst_.neg & 1u) != 0;
    omod = inst.inst_.omod;
    clamp = inst.inst_.clamp;
  }
  const uint32_t denorm_mode = wf.fp_denorm_mode_f16_f64();
  const bool fp16_ovfl = wf.fp16_ovfl();
  const auto compute = [=](uint32_t source) -> uint32_t {
    return fp_mode::rdna4_exp_log_f16(Logarithm, static_cast<uint16_t>(source), denorm_mode,
                                      fp16_ovfl, absolute, negate, omod, clamp);
  };
  if (try_execute_words_simd<1, E32, true, 1>(inst, wf, [&](auto words) {
        return util::map_native_convert_scalar<uint32_t, uint32_t>(words, compute);
      }))
    return;
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    if constexpr (!E32) {
      const uint32_t opsel = vop3_opsel(inst.inst_);
      write_vop3_true16_dst(inst.vdst, wf, lane, opsel,
                            compute(read_vop3_true16_src(inst.src0, wf, lane, opsel, 0)));
    } else {
      sdwa::write_lane<sdwa::ResultFormat::F16>(
          inst, wf, inst.vdst, lane, compute(RegisterAccess(wf).read_lane(inst.src0, lane)));
    }
  }
}

} // namespace rocjitsu::amdgpu::rdna4_exp_log
