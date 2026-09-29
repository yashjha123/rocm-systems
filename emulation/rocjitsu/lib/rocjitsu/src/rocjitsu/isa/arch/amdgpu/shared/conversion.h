// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file conversion.h
/// @brief Bit-exact VALU conversions as measured on gfx1201.
/// @details Every helper takes raw register bits and returns raw destination bits. Source
/// modifiers apply first, then MODE input flushing, then one rounding to the destination
/// format, then OMOD on the rounded result, then CLAMP. The rules come from comparing a
/// physical RX 9070 XT (gfx1201) with the emulator across all MODE settings; other profiles
/// keep their existing generated conversions.
///
/// The evaluate_*() helpers also return the numerical facts learned on the way, before later
/// stages lose them; the value-only helpers return the same bits without the facts.

#include "rocjitsu/code/rj_code.h"

#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu::conversion {

/// @brief Whether an architecture uses the measured gfx1201 conversion rules.
inline bool measured_rules(rj_code_arch_t arch) { return arch == ROCJITSU_CODE_ARCH_RDNA4; }

/// @brief VOP3 modifier fields; encodings without them pass a default value.
struct Modifiers {
  uint32_t abs = 0;
  uint32_t neg = 0;
  bool clamp = false;
  uint32_t omod = 0;
};

/// @brief MODE fields that govern conversions.
struct Mode {
  uint32_t round_f32 = 0;
  uint32_t round_f16_f64 = 0;
  uint32_t denorm_f32 = 3;
  uint32_t denorm_f16_f64 = 3;
  bool fp16_overflow = false;
};

/// @brief Binary floating-point interchange layout.
struct Format {
  int exponent_bits;
  int fraction_bits;
  constexpr int width() const { return 1 + exponent_bits + fraction_bits; }
  constexpr int bias() const { return (1 << (exponent_bits - 1)) - 1; }
  constexpr uint64_t sign() const { return uint64_t{1} << (exponent_bits + fraction_bits); }
  constexpr uint64_t exponent_mask() const {
    return ((uint64_t{1} << exponent_bits) - 1) << fraction_bits;
  }
  constexpr uint64_t fraction_mask() const { return (uint64_t{1} << fraction_bits) - 1; }
  constexpr uint64_t magnitude_mask() const { return exponent_mask() | fraction_mask(); }
};

inline constexpr Format F16{5, 10};
inline constexpr Format F32{8, 23};
inline constexpr Format F64{11, 52};
inline constexpr Format FP8{4, 3};
inline constexpr Format BF8{5, 2};

/// Numerical facts of one conversion. These are not TRAPSTS bits: a target's cause mapping
/// decides which of them raise, combine or suppress architectural causes. The conversion core
/// and OMOD report separately, so an event is not lost when a later stage changes the value.
namespace conversion_fact {
/// The source, after ABS and NEG, is a signaling NaN.
inline constexpr uint32_t kSignalingNan = 1u << 0;
/// The source is a NaN, quiet or signaling.
inline constexpr uint32_t kNan = 1u << 1;
inline constexpr uint32_t kInfinite = 1u << 2;
/// A source was subnormal before input flushing.
inline constexpr uint32_t kInputDenormal = 1u << 3;
inline constexpr uint32_t kInputFlushed = 1u << 4;
/// Rounding, truncation or output flushing discarded nonzero bits, or the value overflowed.
inline constexpr uint32_t kInexact = 1u << 5;
/// The value rounded to the destination precision with an unbounded exponent exceeds the
/// largest finite destination value.
inline constexpr uint32_t kOverflow = 1u << 6;
/// The exact value is nonzero and smaller in magnitude than the smallest normal.
inline constexpr uint32_t kTinyBeforeRounding = 1u << 7;
/// The value rounded to the destination precision with an unbounded exponent is nonzero and
/// smaller in magnitude than the smallest normal.
inline constexpr uint32_t kTinyAfterRounding = 1u << 8;
inline constexpr uint32_t kOutputFlushed = 1u << 9;
/// An overflowing value produced the largest finite value (rounding mode or FP16_OVFL).
inline constexpr uint32_t kSaturated = 1u << 10;
/// An integer destination cannot hold the value before truncation: it lies outside
/// [lowest, highest + 1).
inline constexpr uint32_t kOutOfRange = 1u << 11;
/// OMOD scaled a finite result past the largest finite value.
inline constexpr uint32_t kOmodOverflow = 1u << 16;
/// OMOD produced zero from a nonzero result: a tiny or subnormal result, or a scale-down
/// below the normal range.
inline constexpr uint32_t kOmodUnderflow = 1u << 17;
/// CLAMP replaced the value.
inline constexpr uint32_t kClamped = 1u << 18;
} // namespace conversion_fact

/// @brief Destination bits together with the facts learned computing them.
template <typename Bits> struct Evaluation {
  Bits bits = 0;
  uint32_t facts = 0;
};

namespace detail {

inline bool is_nan(uint64_t bits, Format f) {
  return (bits & f.exponent_mask()) == f.exponent_mask() && (bits & f.fraction_mask()) != 0;
}

inline bool is_inf(uint64_t bits, Format f) {
  return (bits & f.magnitude_mask()) == f.exponent_mask();
}

inline bool to_infinity(uint32_t rounding, bool negative) {
  return rounding == 0 || (rounding == 1 && !negative) || (rounding == 2 && negative);
}

/// Apply ABS then NEG for source `index`.
inline uint64_t modify(uint64_t bits, Format f, const Modifiers &mods, int index) {
  if (mods.abs & (1u << index))
    bits &= ~f.sign();
  if (mods.neg & (1u << index))
    bits ^= f.sign();
  return bits;
}

/// With input denormals disabled a subnormal becomes zero of the same sign.
inline uint64_t flush_input(uint64_t bits, Format f, uint32_t denorm_mode) {
  if (!(denorm_mode & 1u) && (bits & f.exponent_mask()) == 0)
    return bits & f.sign();
  return bits;
}

/// Facts about a modified source that input flushing is about to erase.
inline uint32_t source_facts(uint64_t bits, Format f, uint32_t denorm_mode) {
  namespace fact = conversion_fact;
  if (is_nan(bits, f))
    return fact::kNan | ((bits >> (f.fraction_bits - 1)) & 1u ? 0u : fact::kSignalingNan);
  if (is_inf(bits, f))
    return fact::kInfinite;
  if ((bits & f.exponent_mask()) == 0 && (bits & f.fraction_mask()) != 0)
    return fact::kInputDenormal | (denorm_mode & 1u ? 0u : fact::kInputFlushed);
  return 0;
}

/// A finite nonzero magnitude as significand * 2^exponent.
struct Exact {
  uint64_t significand;
  int exponent;
};

inline Exact exact(uint64_t bits, Format f) {
  const uint64_t field = (bits & f.exponent_mask()) >> f.fraction_bits;
  const uint64_t fraction = bits & f.fraction_mask();
  if (field == 0)
    return {fraction, 1 - f.bias() - f.fraction_bits};
  return {fraction | (uint64_t{1} << f.fraction_bits),
          static_cast<int>(field) - f.bias() - f.fraction_bits};
}

inline int top_bit(uint64_t value) { return 63 - std::countl_zero(value); }

/// Round `significand >> shift` to an integer under a guest rounding mode.
inline uint64_t shift_round(uint64_t significand, int shift, uint32_t rounding, bool negative) {
  if (shift <= 0)
    return significand << -shift;
  if (shift >= 64)
    return significand != 0 && ((rounding == 1 && !negative) || (rounding == 2 && negative));
  const uint64_t kept = significand >> shift;
  const uint64_t rest = significand & ((uint64_t{1} << shift) - 1);
  const uint64_t half = uint64_t{1} << (shift - 1);
  if (rest == 0)
    return kept;
  switch (rounding & 3u) {
  case 0:
    return kept + (rest > half || (rest == half && (kept & 1)));
  case 1:
    return kept + !negative;
  case 2:
    return kept + negative;
  default:
    return kept;
  }
}

/// Whether `significand >> shift` drops nonzero bits.
inline bool discards(uint64_t significand, int shift) {
  if (shift <= 0)
    return false;
  if (shift >= 64)
    return significand != 0;
  return (significand & ((uint64_t{1} << shift) - 1)) != 0;
}

/// Options for the final rounding to a destination format.
struct Rounding {
  uint32_t mode = 0;
  bool flush_output = false;    ///< Tiny results (after rounding) become signed zero.
  bool overflow_to_max = false; ///< Finite overflow saturates regardless of mode.
  int max_exponent = 0;         ///< Largest unbiased exponent; 0 selects the IEEE limit.
};

/// Whether significand * 2^exponent stays below the smallest normal of `f` after rounding to
/// the full significand precision of `f` with an unbounded exponent.
inline bool tiny_after_rounding(uint64_t significand, int exponent, Format f, uint32_t rounding,
                                bool negative) {
  const int min_exponent = 1 - f.bias();
  const int top = top_bit(significand) + exponent;
  if (top != min_exponent - 1)
    return top < min_exponent;
  const uint64_t units =
      shift_round(significand, top - f.fraction_bits - exponent, rounding, negative);
  return !(units >> (f.fraction_bits + 1));
}

/// Round significand * 2^exponent to `f` exactly once.
inline Evaluation<uint64_t> round_to(bool negative, uint64_t significand, int exponent, Format f,
                                     const Rounding &r) {
  namespace fact = conversion_fact;
  const uint64_t sign = negative ? f.sign() : 0;
  if (significand == 0)
    return {sign, 0};
  const int min_exponent = 1 - f.bias();
  const int max_exponent = r.max_exponent ? r.max_exponent : f.bias();
  const int top = top_bit(significand) + exponent;
  uint32_t facts = 0;
  if (top < min_exponent)
    facts |= fact::kTinyBeforeRounding;
  if (tiny_after_rounding(significand, exponent, f, r.mode, negative))
    facts |= fact::kTinyAfterRounding;
  if (r.flush_output && (facts & fact::kTinyAfterRounding))
    return {sign, facts | fact::kOutputFlushed | fact::kInexact};
  const int result_exponent = top > min_exponent ? top : min_exponent;
  const int shift = result_exponent - f.fraction_bits - exponent;
  uint64_t units = shift_round(significand, shift, r.mode, negative);
  if (discards(significand, shift))
    facts |= fact::kInexact;
  int unit_exponent = result_exponent;
  if (units >> (f.fraction_bits + 1)) {
    units >>= 1;
    ++unit_exponent;
  }
  if (unit_exponent > max_exponent) {
    const bool infinite = to_infinity(r.mode, negative) && !r.overflow_to_max;
    facts |= fact::kOverflow | fact::kInexact | (infinite ? 0u : fact::kSaturated);
    return {sign | (infinite ? f.exponent_mask()
                             : (static_cast<uint64_t>(max_exponent + f.bias()) << f.fraction_bits) |
                                   f.fraction_mask()),
            facts};
  }
  if (!(units >> f.fraction_bits))
    return {sign | units, facts};
  return {sign | (static_cast<uint64_t>(unit_exponent + f.bias()) << f.fraction_bits) |
              (units & f.fraction_mask()),
          facts};
}

inline uint64_t encode(bool negative, uint64_t significand, int exponent, Format f,
                       const Rounding &r) {
  return round_to(negative, significand, exponent, f, r).bits;
}

/// OMOD on a rounded result: zero and subnormal become +0, underflow keeps the
/// sign, and overflow rounds in the guest mode.
inline Evaluation<uint64_t> apply_omod(uint64_t bits, Format f, uint32_t omod, uint32_t rounding,
                                       bool overflow_to_max) {
  namespace fact = conversion_fact;
  if (omod == 0 || (bits & f.exponent_mask()) == f.exponent_mask())
    return {bits, 0};
  const int field = static_cast<int>((bits & f.exponent_mask()) >> f.fraction_bits);
  if (field == 0)
    return {0, (bits & f.fraction_mask()) ? fact::kOmodUnderflow : 0u};
  const uint64_t sign = bits & f.sign();
  const int adjusted = field + (omod == 3 ? -1 : static_cast<int>(omod));
  if (adjusted <= 0)
    return {sign, fact::kOmodUnderflow};
  const int limit = (1 << f.exponent_bits) - 1;
  if (adjusted >= limit) {
    const bool infinite = to_infinity(rounding, sign != 0) && !overflow_to_max;
    return {sign | (infinite
                        ? f.exponent_mask()
                        : f.exponent_mask() - (uint64_t{1} << f.fraction_bits) + f.fraction_mask()),
            fact::kOmodOverflow};
  }
  return {(bits & ~f.exponent_mask()) | (static_cast<uint64_t>(adjusted) << f.fraction_bits), 0};
}

/// CLAMP: NaN and negative values become +0, values above one become one.
inline uint64_t apply_clamp(uint64_t bits, Format f) {
  const uint64_t one = static_cast<uint64_t>(f.bias()) << f.fraction_bits;
  if (is_nan(bits, f) || (bits & f.sign()))
    return 0;
  return (bits & f.magnitude_mask()) > one ? one : bits;
}

/// Quiet a NaN and move its payload to the destination width.
inline uint64_t convert_nan(uint64_t bits, Format from, Format to) {
  uint64_t fraction = bits & from.fraction_mask();
  fraction = from.fraction_bits > to.fraction_bits
                 ? fraction >> (from.fraction_bits - to.fraction_bits)
                 : fraction << (to.fraction_bits - from.fraction_bits);
  return ((bits & from.sign()) ? to.sign() : 0) | to.exponent_mask() |
         (uint64_t{1} << (to.fraction_bits - 1)) | fraction;
}

inline uint32_t round_mode(const Mode &m, Format f) {
  return f.fraction_bits == F32.fraction_bits ? m.round_f32 : m.round_f16_f64;
}

inline uint32_t denorm_mode(const Mode &m, Format f) {
  return f.fraction_bits == F32.fraction_bits ? m.denorm_f32 : m.denorm_f16_f64;
}

/// OMOD then CLAMP on a rounded floating result.
inline Evaluation<uint64_t> finish(Evaluation<uint64_t> rounded, Format f, const Modifiers &mods,
                                   const Mode &m) {
  const bool saturate = f.fraction_bits == F16.fraction_bits && m.fp16_overflow;
  const Evaluation<uint64_t> scaled =
      apply_omod(rounded.bits, f, mods.omod, round_mode(m, f), saturate);
  Evaluation<uint64_t> result{scaled.bits, rounded.facts | scaled.facts};
  if (mods.clamp) {
    result.bits = apply_clamp(scaled.bits, f);
    if (result.bits != scaled.bits)
      result.facts |= conversion_fact::kClamped;
  }
  return result;
}

/// Rounding options for a MODE-governed floating destination.
inline Rounding destination(const Mode &m, Format f) {
  return {round_mode(m, f), !(denorm_mode(m, f) & 2u),
          f.fraction_bits == F16.fraction_bits && m.fp16_overflow, 0};
}

} // namespace detail

/// @brief Convert between F16, F32 and F64 (V_CVT_F16_F32, V_CVT_F32_F16, V_CVT_F32_F64,
/// V_CVT_F64_F32).
/// @details Tininess is detected after rounding to the destination precision with an unbounded
/// exponent: a value that rounds up to the smallest normal is kept even with output denormals
/// disabled, while a tiny value flushes, and under OMOD becomes +0 even when its subnormal
/// encoding would round up to the smallest normal. NaNs are quieted and keep the leading payload
/// bits.
inline Evaluation<uint64_t> evaluate_float(uint64_t bits, Format from, Format to,
                                           const Modifiers &mods, const Mode &m) {
  bits = detail::modify(bits & (from.sign() | from.magnitude_mask()), from, mods, 0);
  const uint32_t source = detail::source_facts(bits, from, detail::denorm_mode(m, from));
  bits = detail::flush_input(bits, from, detail::denorm_mode(m, from));
  const bool negative = (bits & from.sign()) != 0;
  Evaluation<uint64_t> rounded;
  if (detail::is_nan(bits, from))
    rounded.bits = detail::convert_nan(bits, from, to);
  else if (detail::is_inf(bits, from))
    rounded.bits = (negative ? to.sign() : 0) | to.exponent_mask();
  else if ((bits & from.magnitude_mask()) == 0)
    rounded.bits = negative ? to.sign() : 0;
  else {
    const detail::Exact e = detail::exact(bits, from);
    rounded = detail::round_to(negative, e.significand, e.exponent, to, detail::destination(m, to));
    // OMOD treats a result that is tiny after rounding to the destination precision as zero,
    // even when the subnormal encoding rounds up to the smallest normal.
    if (mods.omod && (rounded.facts & conversion_fact::kTinyAfterRounding)) {
      rounded.facts |= conversion_fact::kOmodUnderflow;
      rounded.bits = 0;
    }
  }
  rounded.facts |= source;
  return detail::finish(rounded, to, mods, m);
}

inline uint64_t convert_float(uint64_t bits, Format from, Format to, const Modifiers &mods,
                              const Mode &m) {
  return evaluate_float(bits, from, to, mods, m).bits;
}

/// @brief Convert a signed or unsigned integer (V_CVT_F16_I16/U16, V_CVT_F32_I32/U32).
/// @details The destination format's MODE.FP_ROUND field selects the rounding.
inline Evaluation<uint64_t> evaluate_integer(int64_t value, Format to, const Modifiers &mods,
                                             const Mode &m) {
  const bool negative = value < 0;
  const uint64_t magnitude =
      negative ? uint64_t{0} - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
  return detail::finish(detail::round_to(negative, magnitude, 0, to, detail::destination(m, to)),
                        to, mods, m);
}

inline uint64_t convert_integer(int64_t value, Format to, const Modifiers &mods, const Mode &m) {
  return evaluate_integer(value, to, mods, m).bits;
}

/// @brief V_CVT_PK_RTZ_F16_F32: two F32 sources rounded toward zero into packed halves.
/// @details The facts of both halves are combined.
inline Evaluation<uint32_t> evaluate_pack_rtz_f16(uint32_t lo, uint32_t hi, const Modifiers &mods,
                                                  const Mode &m) {
  Mode rtz = m;
  rtz.round_f16_f64 = 3;
  rtz.fp16_overflow = false;
  const Modifiers source_only{mods.abs, mods.neg, false, 0};
  const auto half = [&](uint32_t bits, int index) {
    Modifiers shifted = source_only;
    shifted.abs >>= index;
    shifted.neg >>= index;
    return evaluate_float(bits, F32, F16, shifted, rtz);
  };
  const Evaluation<uint64_t> low = half(lo, 0);
  const Evaluation<uint64_t> high = half(hi, 1);
  return {static_cast<uint32_t>(low.bits) | (static_cast<uint32_t>(high.bits) << 16),
          low.facts | high.facts};
}

inline uint32_t pack_rtz_f16(uint32_t lo, uint32_t hi, const Modifiers &mods, const Mode &m) {
  return evaluate_pack_rtz_f16(lo, hi, mods, m).bits;
}

/// @brief V_PACK_B32_F16: source modifiers, input flushing and sNaN quieting per half.
inline uint32_t pack_f16(uint16_t lo, uint16_t hi, const Modifiers &mods, const Mode &m) {
  const auto half = [&](uint16_t bits, int index) {
    uint64_t v = detail::modify(bits, F16, mods, index);
    v = detail::flush_input(v, F16, m.denorm_f16_f64);
    if (detail::is_nan(v, F16))
      v |= 0x200u;
    return static_cast<uint32_t>(v);
  };
  return half(lo, 0) | (half(hi, 1) << 16);
}

/// @brief Float-to-integer rounding for the measured conversions.
enum class IntegerRounding : uint8_t { TRUNCATE, FLOOR, NEAREST_UP, MODE };

/// @brief Convert a float to a saturated integer.
/// @details NaN becomes zero, or the saturation bound matching its sign when `nan_by_sign` is
/// set (V_CVT_FLOOR_I32_F32, V_CVT_NEAREST_I32_F32). Infinities saturate. The range fact
/// compares the value before rounding with [lowest, highest + 1); the inexact fact reports
/// bits the rounding discarded.
inline Evaluation<int64_t> evaluate_to_integer(uint64_t bits, Format from, int source_index,
                                               const Modifiers &mods, const Mode &m,
                                               IntegerRounding how, int64_t lowest, int64_t highest,
                                               bool nan_by_sign) {
  namespace fact = conversion_fact;
  bits = detail::modify(bits & (from.sign() | from.magnitude_mask()), from, mods, source_index);
  const uint32_t source = detail::source_facts(bits, from, detail::denorm_mode(m, from));
  bits = detail::flush_input(bits, from, detail::denorm_mode(m, from));
  const bool negative = (bits & from.sign()) != 0;
  if (detail::is_nan(bits, from))
    return {nan_by_sign ? (negative ? lowest : highest) : 0, source};
  if (detail::is_inf(bits, from))
    return {negative ? lowest : highest, source};
  if ((bits & from.magnitude_mask()) == 0)
    return {0, source};
  const detail::Exact e = detail::exact(bits, from);
  const int top = detail::top_bit(e.significand) + e.exponent;
  if (top >= 63)
    return {negative ? lowest : highest, source | fact::kOutOfRange | fact::kInexact};
  uint32_t facts = source;
  // |value| truncated, and whether that dropped a fraction, decide the range check exactly.
  const uint64_t whole = detail::shift_round(e.significand, -e.exponent, 3, negative);
  const bool fraction = detail::discards(e.significand, -e.exponent);
  if (fraction)
    facts |= fact::kInexact;
  const bool out_of_range =
      negative ? (whole > uint64_t{0} - static_cast<uint64_t>(lowest) ||
                  (whole == uint64_t{0} - static_cast<uint64_t>(lowest) && fraction))
               : whole > static_cast<uint64_t>(highest);
  if (out_of_range)
    facts |= fact::kOutOfRange;
  uint64_t units;
  switch (how) {
  case IntegerRounding::TRUNCATE:
    units = whole;
    break;
  case IntegerRounding::FLOOR:
    units = detail::shift_round(e.significand, -e.exponent, 2, negative);
    break;
  case IntegerRounding::NEAREST_UP: {
    // floor(x + 0.5) on the signed value.
    const int shift = -e.exponent;
    if (shift <= 0) {
      units = e.significand << -shift;
    } else if (shift >= 64) {
      units = 0;
    } else {
      const uint64_t kept = e.significand >> shift;
      const uint64_t rest = e.significand & ((uint64_t{1} << shift) - 1);
      const uint64_t half = uint64_t{1} << (shift - 1);
      units = kept + (negative ? rest > half : rest >= half);
    }
    break;
  }
  default:
    units = detail::shift_round(e.significand, -e.exponent, detail::round_mode(m, from), negative);
    break;
  }
  const int64_t value = negative ? -static_cast<int64_t>(units) : static_cast<int64_t>(units);
  return {value < lowest ? lowest : (value > highest ? highest : value), facts};
}

inline int64_t convert_to_integer(uint64_t bits, Format from, int source_index,
                                  const Modifiers &mods, const Mode &m, IntegerRounding how,
                                  int64_t lowest, int64_t highest, bool nan_by_sign) {
  return evaluate_to_integer(bits, from, source_index, mods, m, how, lowest, highest, nan_by_sign)
      .bits;
}

/// @brief V_CVT_NORM_I16_F16 / V_CVT_NORM_U16_F16 on one half, round to nearest even.
inline uint16_t normalize_f16(uint16_t bits, bool is_signed, int source_index,
                              const Modifiers &mods, const Mode &m) {
  uint64_t v = detail::modify(bits, F16, mods, source_index);
  v = detail::flush_input(v, F16, m.denorm_f16_f64);
  if (detail::is_nan(v, F16))
    return 0;
  const bool negative = (v & F16.sign()) != 0;
  const int64_t scale = is_signed ? 32767 : 65535;
  if (negative && !is_signed)
    return 0;
  const uint64_t one = static_cast<uint64_t>(F16.bias()) << F16.fraction_bits;
  int64_t magnitude;
  if ((v & F16.magnitude_mask()) >= one) {
    magnitude = scale;
  } else if ((v & F16.magnitude_mask()) == 0) {
    magnitude = 0;
  } else {
    const detail::Exact e = detail::exact(v, F16);
    magnitude = static_cast<int64_t>(
        detail::shift_round(e.significand * static_cast<uint64_t>(scale), -e.exponent, 0, false));
  }
  return static_cast<uint16_t>(negative ? -magnitude : magnitude);
}

/// @brief Decode an OCP FP8 (E4M3) or BF8 (E5M2) byte to F32.
/// @details Every NaN decodes to the negative quiet NaN 0xffc00000.
inline uint32_t decode_fp8(uint8_t byte, Format f) {
  const bool negative = (byte & 0x80u) != 0;
  if (f.exponent_bits == FP8.exponent_bits && (byte & 0x7fu) == 0x7fu)
    return 0xffc00000u;
  if (f.exponent_bits == BF8.exponent_bits && (byte & 0x7cu) == 0x7cu)
    return (byte & 3u) ? 0xffc00000u : (negative ? 0xff800000u : 0x7f800000u);
  if ((byte & 0x7fu) == 0)
    return negative ? 0x80000000u : 0u;
  const detail::Exact e = detail::exact(byte, f);
  return static_cast<uint32_t>(detail::encode(negative, e.significand, e.exponent, F32, {}));
}

/// @brief Encode F32 as OCP FP8 or BF8, round to nearest even or stochastic.
/// @details NaN encodes as 0xff (FP8) or 0xfe (BF8). Overflow and FP8 infinity produce the
/// signed NaN (FP8) or infinity (BF8) pattern, or the signed maximum under FP16_OVFL for a
/// finite value. MODE.FP_DENORM for F32 flushes the input. Stochastic rounding truncates a
/// value headed for the subnormal range to the normal-range precision first, then adds the
/// top bits of `random` below the kept precision. The inexact fact compares the encoded
/// value with the exact source, so stochastic rounding reports it whenever the two differ.
inline Evaluation<uint8_t> evaluate_fp8(uint32_t bits, Format f, int source_index,
                                        const Modifiers &mods, const Mode &m, bool stochastic,
                                        uint32_t random) {
  namespace fact = conversion_fact;
  const bool fp8 = f.exponent_bits == FP8.exponent_bits;
  uint64_t v = detail::modify(bits, F32, mods, source_index);
  const uint32_t source = detail::source_facts(v, F32, m.denorm_f32);
  v = detail::flush_input(v, F32, m.denorm_f32);
  const bool negative = (v & F32.sign()) != 0;
  const uint8_t sign = negative ? 0x80u : 0u;
  if (detail::is_nan(v, F32))
    return {static_cast<uint8_t>(fp8 ? 0xffu : 0xfeu), source};
  const uint8_t maximum = fp8 ? 0x7eu : 0x7bu;
  const uint8_t overflow = sign | (fp8 ? 0x7fu : 0x7cu);
  if (detail::is_inf(v, F32))
    return {overflow, source};
  const Evaluation<uint8_t> saturated{
      m.fp16_overflow ? static_cast<uint8_t>(sign | maximum) : overflow,
      source | fact::kOverflow | fact::kInexact | (m.fp16_overflow ? fact::kSaturated : 0u)};
  if ((v & F32.magnitude_mask()) == 0)
    return {sign, source};
  const detail::Exact e = detail::exact(v, F32);
  const int min_exponent = 1 - f.bias();
  const int top = detail::top_bit(e.significand) + e.exponent;
  const int result_exponent = top > min_exponent ? top : min_exponent;
  const int shift = result_exponent - f.fraction_bits - e.exponent;
  uint64_t units;
  if (!stochastic) {
    units = detail::shift_round(e.significand, shift, 0, negative);
  } else {
    const int normal_shift = F32.fraction_bits - f.fraction_bits;
    const int extra = shift - normal_shift;
    const uint64_t kept = extra >= 64 ? 0 : e.significand >> extra;
    units = (kept + (random >> (32 - normal_shift))) >> normal_shift;
  }
  // A rounded magnitude above the format maximum overflows; otherwise it is exact.
  const int unit_exponent = result_exponent - f.fraction_bits;
  if (units != 0) {
    const detail::Exact max = detail::exact(maximum, f);
    const int units_top = detail::top_bit(units) + unit_exponent;
    const int max_top = detail::top_bit(max.significand) + max.exponent;
    if (units_top > max_top)
      return saturated;
    if (units_top == max_top) {
      const int distance = unit_exponent - max.exponent;
      if (distance >= 0 ? (units << distance) > max.significand
                        : units > (max.significand << -distance))
        return saturated;
    }
  }
  // `shift` is positive here: an F32 significand always has more fraction bits than FP8/BF8.
  const bool inexact =
      shift >= 64 || (units << shift) >> shift != units || (units << shift) != e.significand;
  const int max_exponent = fp8 ? 8 : 15;
  return {static_cast<uint8_t>(
              detail::encode(negative, units, unit_exponent, f, {0, false, false, max_exponent})),
          source | (inexact ? fact::kInexact : 0u)};
}

inline uint8_t encode_fp8(uint32_t bits, Format f, int source_index, const Modifiers &mods,
                          const Mode &m, bool stochastic, uint32_t random) {
  return evaluate_fp8(bits, f, source_index, mods, m, stochastic, random).bits;
}

} // namespace rocjitsu::amdgpu::conversion
