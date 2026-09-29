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
    assert 'amdgpu::conversion::evaluate_float(s0, amdgpu::conversion::F32,' in body
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


def test_float_to_integer_and_f16_widening_use_measured_helpers():
    widen = measured_conversion_body('V_CVT_F32_F16', True, ['src0'], ['vdst'])
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in widen
    assert 'amdgpu::conversion::F16, amdgpu::conversion::F32' in widen
    narrow = measured_conversion_body('V_CVT_I16_F16', True, ['src0'], ['vdst'])
    assert 'IntegerRounding::TRUNCATE, -32768, 32767, false' in narrow
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel, result, true)' in narrow
    for name in ('V_CVT_I32_F32', 'V_CVT_U32_F32', 'V_CVT_U16_F16'):
        assert name in MEASURED_CONVERSIONS


def test_reporting_bodies_publish_causes_before_writing():
    from amdisa.codegen.execute.measured_conversion import (
        STATUS_QUALIFIED_CONVERSIONS,
        STATUS_REPORTING_CONVERSIONS,
    )

    assert STATUS_REPORTING_CONVERSIONS <= STATUS_QUALIFIED_CONVERSIONS
    for name in STATUS_REPORTING_CONVERSIONS:
        body = measured_conversion_body(name, True, ['src0', 'src1'], ['vdst'])
        assert body.count('wf.raise_alu_causes(causes);') == 1
        assert 'wf.set_trapsts(wf.trapsts() | causes);' in body
        assert 'conversion_causes(amdgpu::conversion::CauseRule::' in body
        raise_at = body.index('wf.raise_alu_causes')
        assert body.index('evaluated.facts') < raise_at
        assert body.index('write_') > raise_at


def test_silent_and_unmeasured_bodies_report_nothing():
    silent = measured_conversion_body(
        'V_CVT_PK_RTZ_F16_F32', True, ['src0', 'src1'], ['vdst']
    )
    assert 'raises no exception causes' in silent
    assert 'raise_alu_causes' not in silent
    unmeasured = measured_conversion_body(
        'V_PACK_B32_F16', True, ['src0', 'src1'], ['vdst']
    )
    assert 'unmeasured' in unmeasured
    assert 'raise_alu_causes' not in unmeasured
