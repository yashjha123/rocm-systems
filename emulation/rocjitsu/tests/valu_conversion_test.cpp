// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Physical gfx1201 witnesses for the measured VALU conversions in conversion.h.
// Each row is a lane the emulator previously got wrong, captured on an RX 9070 XT
// with the MODE value, VOP3 modifier variant and raw operands shown.

#include "rocjitsu/isa/arch/amdgpu/shared/conversion.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

using namespace rocjitsu::amdgpu::conversion;

static Mode mode_of(uint32_t md) {
  return Mode{md & 3u, (md >> 2) & 3u, (md >> 4) & 3u, (md >> 6) & 3u, ((md >> 23) & 1u) != 0};
}
static Modifiers mods_of(const std::string &v) {
  Modifiers m;
  if (v.rfind("abs", 0) == 0)
    m.abs = 1u << (v[3] - '0');
  if (v.rfind("neg", 0) == 0)
    m.neg = 1u << (v[3] - '0');
  if (v == "clamp")
    m.clamp = true;
  if (v == "mul2")
    m.omod = 1;
  if (v == "mul4")
    m.omod = 2;
  if (v == "div2")
    m.omod = 3;
  return m;
}
// returns {value, mask}
std::pair<uint64_t, uint64_t> evaluate(const std::string &ins, const std::string &v,
                                       const uint64_t *in, uint32_t md) {
  Mode m = mode_of(md);
  Modifiers mo = mods_of(v);
  bool hi = v.rfind("hi", 0) == 0;
  uint64_t H16 = hi ? 0xffff0000ull : 0xffffull;
  auto h = [&](uint64_t x) { return hi ? (x >> 16) & 0xffff : x & 0xffff; };
  auto put = [&](uint64_t r) { return hi ? r << 16 : r; };
  if (ins == "v_cvt_f16_f32")
    return {put(convert_float(in[0] & 0xffffffff, F32, F16, mo, m)), H16};
  if (ins == "v_cvt_f32_f64")
    return {convert_float(in[0], F64, F32, mo, m), 0xffffffffull};
  if (ins == "v_cvt_f64_f32")
    return {convert_float(in[0] & 0xffffffff, F32, F64, mo, m), ~0ull};
  if (ins == "v_cvt_f16_i16")
    return {put(convert_integer(static_cast<int16_t>(h(in[0])), F16, mo, m)), H16};
  if (ins == "v_cvt_f16_u16")
    return {put(convert_integer(static_cast<uint16_t>(h(in[0])), F16, mo, m)), H16};
  if (ins == "v_cvt_f32_i32")
    return {convert_integer(static_cast<int32_t>(in[0]), F32, mo, m), 0xffffffffull};
  if (ins == "v_cvt_f32_u32")
    return {convert_integer(static_cast<uint32_t>(in[0]), F32, mo, m), 0xffffffffull};
  if (ins == "v_cvt_pk_rtz_f16_f32")
    return {pack_rtz_f16(in[0], in[1], mo, m), 0xffffffffull};
  if (ins == "v_pack_b32_f16") {
    uint16_t a = v == "hi" ? in[0] >> 16 : in[0], b = v == "hi" ? in[1] >> 16 : in[1];
    return {pack_f16(a, b, mo, m), 0xffffffffull};
  }
  auto i32 = [&](Format f, IntegerRounding r, bool s, bool nan_sign) {
    int64_t lo = s ? INT32_MIN : 0, hiv = s ? INT32_MAX : UINT32_MAX;
    return std::pair<uint64_t, uint64_t>{
        static_cast<uint64_t>(convert_to_integer(f.fraction_bits == 52 ? in[0] : in[0] & 0xffffffff,
                                                 f, 0, mo, m, r, lo, hiv, nan_sign)) &
            0xffffffff,
        0xffffffffull};
  };
  if (ins == "v_cvt_i32_f64")
    return i32(F64, IntegerRounding::TRUNCATE, true, false);
  if (ins == "v_cvt_u32_f64")
    return i32(F64, IntegerRounding::TRUNCATE, false, false);
  if (ins == "v_cvt_floor_i32_f32")
    return i32(F32, IntegerRounding::FLOOR, true, true);
  if (ins == "v_cvt_nearest_i32_f32")
    return i32(F32, IntegerRounding::NEAREST_UP, true, true);
  if (ins == "v_cvt_pk_i16_f32" || ins == "v_cvt_pk_u16_f32") {
    bool s = ins == "v_cvt_pk_i16_f32";
    int64_t lo = s ? -32768 : 0, hv = s ? 32767 : 65535;
    uint64_t r = 0;
    for (int i = 0; i < 2; ++i)
      r |= (static_cast<uint64_t>(convert_to_integer(in[i] & 0xffffffff, F32, i, mo, m,
                                                     IntegerRounding::TRUNCATE, lo, hv, false)) &
            0xffff)
           << (16 * i);
    return {r, 0xffffffffull};
  }
  if (ins == "v_cvt_pk_u8_f32") {
    uint64_t r =
        convert_to_integer(in[0] & 0xffffffff, F32, 0, mo, m, IntegerRounding::MODE, 0, 255, false);
    int sh = (in[1] & 3) * 8;
    return {((in[2] & 0xffffffff) & ~(0xffull << sh)) | (r << sh), 0xffffffffull};
  }
  if (ins == "v_cvt_norm_i16_f16" || ins == "v_cvt_norm_u16_f16")
    return {put(normalize_f16(h(in[0]), ins == "v_cvt_norm_i16_f16", 0, mo, m)), H16};
  if (ins == "v_cvt_pk_norm_i16_f16" || ins == "v_cvt_pk_norm_u16_f16") {
    bool s = ins == "v_cvt_pk_norm_i16_f16";
    bool hh = v == "hi";
    uint64_t r = 0;
    for (int i = 0; i < 2; ++i)
      r |= uint64_t(normalize_f16(hh ? in[i] >> 16 : in[i], s, i, mo, m)) << (16 * i);
    return {r, 0xffffffffull};
  }
  if (ins == "v_sat_pk_u8_i16") {
    uint32_t x = in[0];
    uint64_t r = 0;
    for (int i = 0; i < 2; ++i) {
      int32_t s = static_cast<int16_t>(x >> (16 * i));
      r |= uint64_t(s < 0 ? 0 : s > 255 ? 255 : s) << (8 * i);
    }
    return {put(r), H16};
  }
  if (ins == "v_cvt_f32_fp8" || ins == "v_cvt_f32_bf8") {
    int sel = v.rfind("byte", 0) == 0 ? v[4] - '0' : 0;
    return {decode_fp8(in[0] >> (8 * sel), ins == "v_cvt_f32_fp8" ? FP8 : BF8), 0xffffffffull};
  }
  if (ins == "v_cvt_pk_f32_fp8" || ins == "v_cvt_pk_f32_bf8") {
    Format f = ins == "v_cvt_pk_f32_fp8" ? FP8 : BF8;
    uint32_t w = v == "opsel_hi" ? in[0] >> 16 : in[0];
    return {uint64_t(decode_fp8(w & 0xff, f)) | (uint64_t(decode_fp8((w >> 8) & 0xff, f)) << 32),
            ~0ull};
  }
  if (ins == "v_cvt_pk_fp8_f32" || ins == "v_cvt_pk_bf8_f32") {
    Format f = ins == "v_cvt_pk_fp8_f32" ? FP8 : BF8;
    uint64_t r = encode_fp8(in[0], f, 0, mo, m, false, 0) |
                 (uint64_t(encode_fp8(in[1], f, 1, mo, m, false, 0)) << 8);
    int sh = v == "opsel_hi" ? 16 : 0;
    return {((in[2] & 0xffffffff) & ~(0xffffull << sh)) | (r << sh), 0xffffffffull};
  }
  if (ins == "v_cvt_sr_fp8_f32" || ins == "v_cvt_sr_bf8_f32") {
    Format f = ins == "v_cvt_sr_fp8_f32" ? FP8 : BF8;
    int sel = v.rfind("byte", 0) == 0 ? v[4] - '0' : 0;
    uint64_t r = encode_fp8(in[0], f, 0, mo, m, true, static_cast<uint32_t>(in[1]));
    return {((in[2] & 0xffffffff) & ~(0xffull << (8 * sel))) | (r << (8 * sel)), 0xffffffffull};
  }
  return {0, 0};
}

struct Witness {
  const char *instruction;
  const char *variant;
  uint32_t mode;
  uint64_t in[3];
  uint64_t expected;
};

// Variants: e32/e64 plain, negN/absN source modifiers, clamp, mul2/mul4/div2
// OMOD, hi* true16 .h halves, byteN/opsel_hi byte and word selects.
constexpr Witness kWitnesses[] = {
    {"v_cvt_f16_f32", "div2", 0x8000f0u, {0xb3dbd142ull, 0x0ull, 0x0ull}, 0xa5a50000ull},
    {"v_cvt_f16_f32", "mul2", 0xffu, {0x74b682f5ull, 0x0ull, 0x0ull}, 0xa5a57bffull},
    {"v_cvt_f16_f32", "neg0", 0x800000u, {0x3a9ull, 0x0ull, 0x0ull}, 0xa5a58000ull},
    {"v_cvt_f32_f64", "neg0", 0x8000f0u, {0x40d20ea000000000ull, 0x0ull, 0x0ull}, 0xc6907500ull},
    {"v_cvt_f32_f64", "abs0", 0x800000u, {0xc0f886fc00000000ull, 0x0ull, 0x0ull}, 0x47c437e0ull},
    {"v_cvt_f32_f64", "e64", 0xffu, {0xbce2e681dd7b236bull, 0x0ull, 0x0ull}, 0xa717340eull},
    {"v_cvt_f64_f32", "neg0", 0xa0u, {0xd0137084ull, 0x0ull, 0x0ull}, 0x42026e1080000000ull},
    {"v_cvt_f64_f32", "e32", 0xa5u, {0x490c7dull, 0x0ull, 0x0ull}, 0x0ull},
    {"v_cvt_f64_f32", "clamp", 0xa0u, {0x76ee58ull, 0x0ull, 0x0ull}, 0x0ull},
    {"v_cvt_pk_rtz_f16_f32",
     "e64",
     0x800000u,
     {0xb7b442dfull, 0xc4ecc000ull, 0x0ull},
     0xe7668000ull},
    {"v_cvt_pk_rtz_f16_f32", "e32", 0x1au, {0xc0400000ull, 0x387fc000ull, 0x0ull}, 0xc200ull},
    {"v_cvt_pk_rtz_f16_f32", "neg1", 0x50u, {0xc4184c6dull, 0x35ed65a5ull, 0x0ull}, 0x8000e0c2ull},
    {"v_pack_b32_f16", "abs0", 0xa0u, {0xd06c924bull, 0xb3cfb3bdull, 0x0ull}, 0xb3bd124bull},
    {"v_pack_b32_f16", "hi", 0xa0u, {0x80cc0abaull, 0xdde4ea25ull, 0x0ull}, 0xdde48000ull},
    {"v_pack_b32_f16", "neg0", 0xffu, {0x9d5aa814ull, 0xffdda7aeull, 0x0ull}, 0xa7ae2814ull},
    {"v_cvt_f16_i16", "clamp", 0x50u, {0xd7ee0025ull, 0x0ull, 0x0ull}, 0xa5a53c00ull},
    {"v_cvt_f16_i16", "e64", 0xffu, {0xab2d93e7ull, 0x0ull, 0x0ull}, 0xa5a5f6c1ull},
    {"v_cvt_f16_i16", "hi_e32", 0xffu, {0xa6443dd6ull, 0x0ull, 0x0ull}, 0xf59ba5a5ull},
    {"v_cvt_f16_u16", "mul2", 0xf0u, {0x3fdc252full, 0x0ull, 0x0ull}, 0xa5a574a6ull},
    {"v_cvt_f16_u16", "clamp", 0x8000f0u, {0x8a295e80ull, 0x0ull, 0x0ull}, 0xa5a53c00ull},
    {"v_cvt_f16_u16", "e32", 0x4fu, {0x7a11ffdeull, 0x0ull, 0x0ull}, 0xa5a57bfeull},
    {"v_cvt_f32_i32", "e64", 0xffu, {0xcee1c49aull, 0x0ull, 0x0ull}, 0xce4478edull},
    {"v_cvt_f32_i32", "e32", 0x3au, {0x802b0e45ull, 0x0ull, 0x0ull}, 0xceffa9e4ull},
    {"v_cvt_f32_i32", "div2", 0xffu, {0x923461eull, 0x0ull, 0x0ull}, 0x4c923461ull},
    {"v_cvt_f32_u32", "mul4", 0xffu, {0xffffffecull, 0x0ull, 0x0ull}, 0x507fffffull},
    {"v_cvt_f32_u32", "e32", 0xfu, {0xfffffff8ull, 0x0ull, 0x0ull}, 0x4f7fffffull},
    {"v_cvt_f32_u32", "mul2", 0xffu, {0xe03a1cd4ull, 0x0ull, 0x0ull}, 0x4fe03a1cull},
    {"v_cvt_i32_f64", "neg0", 0x50u, {0x41508ba0f8f50c1cull, 0x0ull, 0x0ull}, 0xffbdd17dull},
    {"v_cvt_i32_f64", "abs0", 0xf0u, {0xd31ada94b51e40a0ull, 0x0ull, 0x0ull}, 0x7fffffffull},
    {"v_cvt_u32_f64", "abs0", 0xffu, {0xc1210e8300000000ull, 0x0ull, 0x0ull}, 0x88741ull},
    {"v_cvt_u32_f64", "neg0", 0x800000u, {0xc02b9840ebf676c0ull, 0x0ull, 0x0ull}, 0xdull},
    {"v_cvt_floor_i32_f32", "neg0", 0xa0u, {0x7faf03c2ull, 0x0ull, 0x0ull}, 0x80000000ull},
    {"v_cvt_floor_i32_f32", "e32", 0x800030u, {0x7fe954e9ull, 0x0ull, 0x0ull}, 0x7fffffffull},
    {"v_cvt_floor_i32_f32", "e64", 0xa0u, {0x80562693ull, 0x0ull, 0x0ull}, 0x0ull},
    {"v_cvt_nearest_i32_f32", "abs0", 0x8000f0u, {0xffd52613ull, 0x0ull, 0x0ull}, 0x7fffffffull},
    {"v_cvt_nearest_i32_f32", "e32", 0x800030u, {0x7f80f177ull, 0x0ull, 0x0ull}, 0x7fffffffull},
    {"v_cvt_nearest_i32_f32", "e64", 0xa0u, {0x7ffb4c2bull, 0x0ull, 0x0ull}, 0x7fffffffull},
    {"v_cvt_pk_i16_f32", "neg1", 0x8000f0u, {0x807fffffull, 0xff7fffffull, 0x0ull}, 0x7fff0000ull},
    {"v_cvt_pk_i16_f32", "abs1", 0x0u, {0x90f11773ull, 0xc9b4b4e2ull, 0x0ull}, 0x7fff0000ull},
    {"v_cvt_pk_i16_f32", "neg0", 0x800000u, {0xc4368000ull, 0xe3e5e2ull, 0x0ull}, 0x2daull},
    {"v_cvt_pk_u16_f32", "neg1", 0xf0u, {0x483521e8ull, 0xbf94b1d1ull, 0x0ull}, 0x1ffffull},
    {"v_cvt_pk_u16_f32", "abs1", 0xf0u, {0x35a058e3ull, 0xc4d5a000ull, 0x0ull}, 0x6ad0000ull},
    {"v_cvt_pk_u16_f32", "abs0", 0x800000u, {0xc7800000ull, 0x3e800000ull, 0x0ull}, 0xffffull},
    {"v_cvt_pk_u8_f32", "abs0", 0xffu, {0xff800000ull, 0x2ull, 0xffff8000ull}, 0xffff8000ull},
    {"v_cvt_pk_u8_f32", "neg0", 0x800000u, {0xcaa30471ull, 0x1f311dbbull, 0x40ull}, 0xff000040ull},
    {"v_cvt_pk_u8_f32", "e64", 0x800075u, {0x14297993ull, 0x29ull, 0x157b87ull}, 0x150187ull},
    {"v_cvt_norm_i16_f16", "e32", 0x20u, {0x6f090262ull, 0x0ull, 0x0ull}, 0xa5a50000ull},
    {"v_cvt_norm_i16_f16", "abs0", 0xa0u, {0x1c838269ull, 0x0ull, 0x0ull}, 0xa5a50000ull},
    {"v_cvt_norm_i16_f16", "hi_e64", 0x0u, {0x82fd8590ull, 0x0ull, 0x0ull}, 0xa5a5ull},
    {"v_cvt_norm_u16_f16", "hi_e32", 0x0u, {0x3e8812full, 0x0ull, 0x0ull}, 0xa5a5ull},
    {"v_cvt_norm_u16_f16", "hi_e64", 0xa0u, {0x9f17d5ull, 0x0ull, 0x0ull}, 0xa5a5ull},
    {"v_cvt_norm_u16_f16", "e64", 0xa0u, {0x3e2f0383ull, 0x0ull, 0x0ull}, 0xa5a50000ull},
    {"v_cvt_pk_norm_i16_f16", "neg0", 0xa0u, {0xd52d03ffull, 0xc85d3400ull, 0x0ull}, 0x20000000ull},
    {"v_cvt_pk_norm_i16_f16", "abs0", 0x0u, {0xe7a7364dull, 0xeff403c6ull, 0x0ull}, 0x3268ull},
    {"v_cvt_pk_norm_i16_f16", "hi", 0xa0u, {0x368c500ull, 0x95c1be00ull, 0x0ull}, 0xffd20000ull},
    {"v_cvt_pk_norm_u16_f16", "neg0", 0x0u, {0x755e8242ull, 0xa6a34940ull, 0x0ull}, 0xffff0000ull},
    {"v_cvt_pk_norm_u16_f16", "e64", 0x3au, {0x7b4ac994ull, 0x7fb002f9ull, 0x0ull}, 0x0ull},
    {"v_cvt_pk_norm_u16_f16", "abs0", 0x0u, {0x1bb057ffull, 0x79610200ull, 0x0ull}, 0xffffull},
    {"v_cvt_f32_fp8", "byte2", 0xffu, {0x217ff7d6ull, 0x0ull, 0x0ull}, 0xffc00000ull},
    {"v_cvt_f32_fp8", "byte1", 0x8000bau, {0x91c17f6full, 0x0ull, 0x0ull}, 0xffc00000ull},
    {"v_cvt_f32_fp8", "e64", 0x800000u, {0x7f7f7f7full, 0x0ull, 0x0ull}, 0xffc00000ull},
    {"v_cvt_f32_bf8", "byte1", 0xeau, {0xa2617f64ull, 0x0ull, 0x0ull}, 0xffc00000ull},
    {"v_cvt_f32_bf8", "byte2", 0x60u, {0x307d89efull, 0x0ull, 0x0ull}, 0xffc00000ull},
    {"v_cvt_f32_bf8", "e32", 0x8000fau, {0x1dc9327eull, 0x0ull, 0x0ull}, 0xffc00000ull},
    {"v_cvt_pk_f32_fp8", "opsel_hi", 0xa0u, {0x7fce26e3ull, 0x0ull, 0x0ull}, 0xffc00000c0e00000ull},
    {"v_cvt_pk_f32_fp8", "e32", 0xa0u, {0xdee7f20ull, 0x0ull, 0x0ull}, 0xffc000003e000000ull},
    {"v_cvt_pk_f32_fp8", "e64", 0xf0u, {0x826c5c7full, 0x0ull, 0x0ull}, 0x41c00000ffc00000ull},
    {"v_cvt_pk_f32_bf8", "e64", 0xa0u, {0xfc86937full, 0x0ull, 0x0ull}, 0xba600000ffc00000ull},
    {"v_cvt_pk_f32_bf8", "opsel_hi", 0xffu, {0x7ec1518dull, 0x0ull, 0x0ull}, 0xffc00000c0200000ull},
    {"v_cvt_pk_f32_bf8", "e32", 0x1fu, {0x7full, 0x0ull, 0x0ull}, 0xffc00000ull},
    {"v_cvt_pk_fp8_f32", "opsel_hi", 0x8000f0u, {0xff800000ull, 0x0ull, 0xffffffull}, 0xffffffull},
    {"v_cvt_pk_fp8_f32",
     "neg0",
     0x800000u,
     {0x815bed9full, 0xb606d275ull, 0xff8c4c3eull},
     0xff8c8000ull},
    {"v_cvt_pk_fp8_f32", "e64", 0x90u, {0x7f800001ull, 0x1ull, 0x20ull}, 0xffull},
    {"v_cvt_pk_bf8_f32", "e64", 0xf9u, {0x7f800001ull, 0x3f800000ull, 0xffffffull}, 0xff3cfeull},
    {"v_cvt_pk_bf8_f32",
     "abs1",
     0xf0u,
     {0xb54665b7ull, 0xc1c80000ull, 0xfffffffeull},
     0xffff4e80ull},
    {"v_cvt_pk_bf8_f32",
     "neg0",
     0xffu,
     {0xc2d42495ull, 0xb275e1adull, 0xea765df9ull},
     0xea768057ull},
    {"v_cvt_sr_fp8_f32",
     "e64",
     0x800030u,
     {0xb3bd91ffull, 0xffffff23ull, 0x9c8f50bcull},
     0x9c8f5081ull},
    {"v_cvt_sr_fp8_f32", "byte1", 0xffu, {0x38aa3f64ull, 0xffffffc2ull, 0x19000cull}, 0x19010cull},
    {"v_cvt_sr_fp8_f32", "neg0", 0x8000f0u, {0x7fc00000ull, 0x1full, 0x800000ull}, 0x8000ffull},
    {"v_cvt_sr_bf8_f32", "byte2", 0x8000f0u, {0x7f800000ull, 0x1full, 0x1full}, 0x7c001full},
    {"v_cvt_sr_bf8_f32", "e64", 0x80u, {0x7f800001ull, 0x1full, 0xffffffffull}, 0xfffffffeull},
    {"v_cvt_sr_bf8_f32", "byte3", 0xa0u, {0x7fc00000ull, 0x1full, 0xffffull}, 0xfe00ffffull},
    {"v_sat_pk_u8_i16", "hi_e64", 0xf0u, {0x1abeeaull, 0x0ull, 0x0ull}, 0x1a00a5a5ull},
    {"v_sat_pk_u8_i16", "e64", 0xf0u, {0x1d5aull, 0x0ull, 0x0ull}, 0xa5a500ffull},
};

// Directed gfx1201 captures at the smallest-normal boundary. Output flushing and the OMOD
// zeroing of tiny results both judge tininess after rounding to the destination precision
// with an unbounded exponent: 0x387ff000 (F32) rounds to the smallest F16 normal and is kept,
// while the exact 11-bit 0x387fe000 is tiny even though its subnormal encoding rounds up.
constexpr Witness kTininessWitnesses[] = {
    {"v_cvt_f16_f32", "e64", 0x00u, {0x387ff000ull, 0x0ull, 0x0ull}, 0xa5a50400ull},
    {"v_cvt_f16_f32", "e64", 0x00u, {0x387fe000ull, 0x0ull, 0x0ull}, 0xa5a50000ull},
    {"v_cvt_f16_f32", "neg0", 0x50u, {0x387ff000ull, 0x0ull, 0x0ull}, 0xa5a58400ull},
    {"v_cvt_f16_f32", "e64", 0xf0u, {0x387fe000ull, 0x0ull, 0x0ull}, 0xa5a50400ull},
    {"v_cvt_f16_f32", "mul2", 0xf0u, {0x387ff000ull, 0x0ull, 0x0ull}, 0xa5a50800ull},
    {"v_cvt_f16_f32", "mul4", 0x00u, {0x387ff000ull, 0x0ull, 0x0ull}, 0xa5a50c00ull},
    {"v_cvt_f16_f32", "mul2", 0xf0u, {0x387fe000ull, 0x0ull, 0x0ull}, 0xa5a50000ull},
    {"v_cvt_f32_f64", "e64", 0x00u, {0x380ffffff0000000ull, 0x0ull, 0x0ull}, 0x800000ull},
    {"v_cvt_f32_f64", "e64", 0x50u, {0x380fffffe0000000ull, 0x0ull, 0x0ull}, 0x0ull},
    {"v_cvt_f32_f64", "mul2", 0x50u, {0x380ffffff0000000ull, 0x0ull, 0x0ull}, 0x1000000ull},
    {"v_cvt_f32_f64", "mul2", 0xf0u, {0x380fffffe0000000ull, 0x0ull, 0x0ull}, 0x0ull},
};

TEST(ValuConversion, JudgesTininessAfterRounding) {
  for (const Witness &w : kTininessWitnesses) {
    const auto [value, mask] = evaluate(w.instruction, w.variant, w.in, w.mode);
    EXPECT_EQ(value & mask, w.expected & mask) << w.instruction << " " << w.variant << " mode=0x"
                                               << std::hex << w.mode << " in=0x" << w.in[0];
  }
}

TEST(ValuConversion, MatchesGfx1201Witnesses) {
  for (const Witness &w : kWitnesses) {
    const auto [value, mask] = evaluate(w.instruction, w.variant, w.in, w.mode);
    ASSERT_NE(mask, 0u) << w.instruction;
    EXPECT_EQ(value & mask, w.expected & mask)
        << w.instruction << " " << w.variant << " mode=0x" << std::hex << w.mode << " in=0x"
        << w.in[0] << ",0x" << w.in[1] << ",0x" << w.in[2];
  }
}

namespace fact = conversion_fact;

struct FactCase {
  const char *name;
  uint64_t bits;
  uint32_t facts;
};

void expect_evaluation(const FactCase &c, uint64_t bits, uint32_t facts) {
  EXPECT_EQ(bits, c.bits) << c.name;
  EXPECT_EQ(facts, c.facts) << c.name << " facts=0x" << std::hex << facts;
}

TEST(ValuConversion, ReportsFloatRoundingFacts) {
  const auto f16 = [](uint32_t in, uint32_t md, const char *variant = "e64") {
    return evaluate_float(in, F32, F16, mods_of(variant), mode_of(md));
  };
  const auto check = [](const FactCase &c, Evaluation<uint64_t> e) {
    expect_evaluation(c, e.bits, e.facts);
  };
  // Rounds up to the smallest normal: tiny only before rounding, so nothing flushes.
  check({"round up to normal", 0x0400, fact::kTinyBeforeRounding | fact::kInexact},
        f16(0x387ff000u, 0x00u));
  check({"tiny, flushed", 0x0000,
         fact::kTinyBeforeRounding | fact::kTinyAfterRounding | fact::kOutputFlushed |
             fact::kInexact},
        f16(0x387fe000u, 0x00u));
  check({"tiny, subnormal encoding rounds up", 0x0400,
         fact::kTinyBeforeRounding | fact::kTinyAfterRounding | fact::kInexact},
        f16(0x387fe000u, 0xf0u));
  check({"tiny under OMOD", 0x0000,
         fact::kTinyBeforeRounding | fact::kTinyAfterRounding | fact::kInexact |
             fact::kOmodUnderflow},
        f16(0x387fe000u, 0xf0u, "mul2"));
  check({"overflow", 0x7c00, fact::kOverflow | fact::kInexact}, f16(0x477ff000u, 0xf0u));
  check({"RTZ lands on max", 0x7bff, fact::kInexact}, f16(0x477ff000u, 0xfcu));
  check({"FP16_OVFL saturates", 0x7bff, fact::kOverflow | fact::kInexact | fact::kSaturated},
        f16(0x477ff000u, 0x8000f0u));
  check({"OMOD overflow", 0x7c00, fact::kOmodOverflow}, f16(0x477fe000u, 0xf0u, "mul2"));
  check({"CLAMP", 0x3c00, fact::kClamped}, f16(0x40000000u, 0xf0u, "clamp"));
  check({"signaling NaN", 0x7e00, fact::kNan | fact::kSignalingNan}, f16(0x7f800001u, 0xf0u));
  check({"quiet NaN", 0x7e00, fact::kNan}, f16(0x7fc00000u, 0xf0u));
  check({"infinity", 0xfc00, fact::kInfinite}, f16(0x7f800000u, 0xf0u, "neg0"));
  const auto f32 = [](uint32_t in, uint32_t md) {
    return evaluate_float(in, F16, F32, Modifiers{}, mode_of(md));
  };
  check({"kept subnormal", 0x33800000, fact::kInputDenormal}, f32(0x0001u, 0xf0u));
  check({"flushed subnormal", 0x0, fact::kInputDenormal | fact::kInputFlushed},
        f32(0x0001u, 0x30u));
  check({"integer inexact", 0x4b800000, fact::kInexact},
        evaluate_integer(16777217, F32, Modifiers{}, mode_of(0xf0u)));
  check({"integer overflow", 0x7c00, fact::kOverflow | fact::kInexact},
        evaluate_integer(65535, F16, Modifiers{}, mode_of(0xf0u)));
}

TEST(ValuConversion, ReportsIntegerRangeFacts) {
  const auto i32 = [](uint64_t in, Format f, bool is_signed, const char *variant = "e64") {
    return evaluate_to_integer(in, f, 0, mods_of(variant), mode_of(0xf0u),
                               IntegerRounding::TRUNCATE, is_signed ? INT32_MIN : 0,
                               is_signed ? INT32_MAX : UINT32_MAX, false);
  };
  const auto check = [](const FactCase &c, Evaluation<int64_t> e) {
    expect_evaluation(c, static_cast<uint64_t>(e.bits), e.facts);
  };
  check({"truncation", 2, fact::kInexact}, i32(0x40200000u, F32, true));
  check({"below INT32_MIN before truncation", static_cast<uint64_t>(int64_t{INT32_MIN}),
         fact::kOutOfRange | fact::kInexact},
        i32(0xc1e0000000100000ull, F64, true));
  check({"exactly INT32_MIN", static_cast<uint64_t>(int64_t{INT32_MIN}), 0},
        i32(0xcf000000u, F32, true));
  check({"2^31", INT32_MAX, fact::kOutOfRange}, i32(0x4f000000u, F32, true));
  check({"negative unsigned", 0, fact::kOutOfRange | fact::kInexact}, i32(0xbf000000u, F32, false));
  check({"negated unsigned", 0, fact::kOutOfRange}, i32(0x3f800000u, F32, false, "neg0"));
  check({"NaN", 0, fact::kNan}, i32(0x7fc00000u, F32, true));
  check({"flushed subnormal", 0, fact::kInputDenormal | fact::kInputFlushed},
        evaluate_to_integer(0x1u, F32, 0, Modifiers{}, mode_of(0x00u), IntegerRounding::TRUNCATE,
                            INT32_MIN, INT32_MAX, false));
}

TEST(ValuConversion, ReportsFp8Facts) {
  const auto fp8 = [](uint32_t in, uint32_t md) {
    return evaluate_fp8(in, FP8, 0, Modifiers{}, mode_of(md), false, 0);
  };
  const auto check = [](const FactCase &c, Evaluation<uint8_t> e) {
    expect_evaluation(c, e.bits, e.facts);
  };
  check({"exact", 0x38, 0}, fp8(0x3f800000u, 0xf0u));
  check({"inexact", 0x38, fact::kInexact}, fp8(0x3f800001u, 0xf0u));
  check({"overflow", 0x7f, fact::kOverflow | fact::kInexact}, fp8(0x447a0000u, 0xf0u));
  check({"FP16_OVFL", 0x7e, fact::kOverflow | fact::kInexact | fact::kSaturated},
        fp8(0x447a0000u, 0x8000f0u));
  check({"signaling NaN", 0xff, fact::kNan | fact::kSignalingNan}, fp8(0x7f800001u, 0xf0u));
  check({"kept subnormal", 0x00, fact::kInputDenormal | fact::kInexact}, fp8(0x1u, 0xf0u));
  check({"flushed subnormal", 0x00, fact::kInputDenormal | fact::kInputFlushed}, fp8(0x1u, 0xc0u));
}

} // namespace
