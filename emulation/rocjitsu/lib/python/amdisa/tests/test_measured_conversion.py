"""Tests for the RDNA4 measured-conversion execute bodies."""

from amdisa.codegen.execute.measured_conversion import (
    MEASURED_CONVERSIONS,
    measured_conversion_body,
)


def test_vop3_body_passes_modifiers_and_true16_halves():
    body = measured_conversion_body('V_CVT_F16_F32', True, ['src0'], ['vdst'])
    assert body is not None
    assert 'static_cast<uint32_t>(inst_.abs)' in body
    assert 'static_cast<uint32_t>(inst_.omod)' in body
    assert 'amdgpu::conversion::convert_float(s0, amdgpu::conversion::F32,' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel, result, true)' in body


def test_vop1_body_has_no_modifier_fields():
    body = measured_conversion_body('V_CVT_F32_F64', False, ['src0'], ['vdst'])
    assert body is not None
    assert 'inst_.' not in body
    assert 'read_lane64(src0, lane)' in body
    assert 'amdgpu::conversion::Modifiers mods{};' in body


def test_fp8_selects_follow_opsel():
    decode = measured_conversion_body('V_CVT_F32_FP8', True, ['src0'], ['vdst'])
    assert '((opsel & 0x1u) << 1) | ((opsel & 0x2u) >> 1)' in decode
    stochastic = measured_conversion_body(
        'V_CVT_SR_BF8_F32', True, ['src0', 'src1'], ['vdst']
    )
    assert '((opsel >> 2) & 0x3u) * 8u' in stochastic
    packed = measured_conversion_body(
        'V_CVT_PK_FP8_F32', True, ['src0', 'src1'], ['vdst']
    )
    assert 'opsel & 0x8u' in packed


def test_every_measured_conversion_has_a_body():
    for name in MEASURED_CONVERSIONS:
        assert measured_conversion_body(name, True, ['src0', 'src1', 'src2'], ['vdst'])
