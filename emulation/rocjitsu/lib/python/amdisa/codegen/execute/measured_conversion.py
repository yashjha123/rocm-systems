"""Execute bodies for the RDNA4 conversions measured on gfx1201.

Each body reads raw register bits, calls a helper from
``shared/conversion.h`` and writes raw destination bits. The helpers apply
source modifiers, MODE input flushing, one rounding, OMOD and CLAMP in the
order measured on a physical RX 9070 XT; see that header for the rules.
"""

from __future__ import annotations

from dataclasses import dataclass

MEASURED_CONVERSION_ARCHES = frozenset({'rdna4'})

_NS = 'amdgpu::conversion'


@dataclass(frozen=True)
class _Spec:
    sources: tuple[str, ...]  # 'b16', 'b32' or 'b64' per source
    result: str  # C++ expression over s0..s2, mods, mode and opsel
    write: str  # 'b16', 'b32', 'b64', 'b16_word' or 'byte'


def _float(src: str, dst: str) -> str:
    return f'{_NS}::convert_float(s0, {_NS}::{src}, {_NS}::{dst}, mods, mode)'


def _integer(cast: str, dst: str) -> str:
    return f'{_NS}::convert_integer(static_cast<{cast}>(s0), {_NS}::{dst}, mods, mode)'


def _to_integer(
    src: str, index: int, how: str, low: str, high: str, nan_by_sign: bool
) -> str:
    return (
        f'{_NS}::convert_to_integer(s{index}, {_NS}::{src}, {index}, mods, mode, '
        f'{_NS}::IntegerRounding::{how}, {low}, {high}, {"true" if nan_by_sign else "false"})'
    )


def _pk_integer(low: str, high: str) -> str:
    halves = [
        f'(static_cast<uint32_t>({_to_integer("F32", i, "TRUNCATE", low, high, False)}) & 0xffffu)'
        for i in range(2)
    ]
    return f'{halves[0]} | ({halves[1]} << 16)'


def _pk_norm(signed: str) -> str:
    return (
        f'static_cast<uint32_t>({_NS}::normalize_f16(static_cast<uint16_t>(s0), {signed}, 0, mods, mode)) | '
        f'(static_cast<uint32_t>({_NS}::normalize_f16(static_cast<uint16_t>(s1), {signed}, 1, mods, mode)) << 16)'
    )


def _fp8_decode(fmt: str, byte: str) -> str:
    return (
        f'{_NS}::decode_fp8(static_cast<uint8_t>(s0 >> (8u * ({byte}))), {_NS}::{fmt})'
    )


def _fp8_pk_decode(fmt: str) -> str:
    word = '(s0 >> ((opsel & 0x1u) ? 16u : 0u))'
    return (
        f'static_cast<uint64_t>({_NS}::decode_fp8(static_cast<uint8_t>({word}), {_NS}::{fmt})) | '
        f'(static_cast<uint64_t>({_NS}::decode_fp8(static_cast<uint8_t>({word} >> 8), {_NS}::{fmt})) << 32)'
    )


def _fp8_pk_encode(fmt: str) -> str:
    return (
        f'static_cast<uint32_t>({_NS}::encode_fp8(s0, {_NS}::{fmt}, 0, mods, mode, false, 0)) | '
        f'(static_cast<uint32_t>({_NS}::encode_fp8(s1, {_NS}::{fmt}, 1, mods, mode, false, 0)) << 8)'
    )


def _fp8_sr_encode(fmt: str) -> str:
    return f'{_NS}::encode_fp8(s0, {_NS}::{fmt}, 0, mods, mode, true, s1)'


_INT32 = ('INT32_MIN', 'INT32_MAX')
_UINT32 = ('0', 'UINT32_MAX')

_SPECS: dict[str, _Spec] = {
    'V_CVT_F16_F32': _Spec(('b32',), _float('F32', 'F16'), 'b16'),
    'V_CVT_F32_F64': _Spec(('b64',), _float('F64', 'F32'), 'b32'),
    'V_CVT_F64_F32': _Spec(('b32',), _float('F32', 'F64'), 'b64'),
    'V_CVT_F32_F16': _Spec(('b16',), _float('F16', 'F32'), 'b32'),
    'V_CVT_F16_I16': _Spec(('b16',), _integer('int16_t', 'F16'), 'b16'),
    'V_CVT_F16_U16': _Spec(('b16',), _integer('uint16_t', 'F16'), 'b16'),
    'V_CVT_F32_I32': _Spec(('b32',), _integer('int32_t', 'F32'), 'b32'),
    'V_CVT_F32_U32': _Spec(('b32',), _integer('uint32_t', 'F32'), 'b32'),
    'V_CVT_PK_RTZ_F16_F32': _Spec(
        ('b32', 'b32'), f'{_NS}::pack_rtz_f16(s0, s1, mods, mode)', 'b32'
    ),
    'V_PACK_B32_F16': _Spec(
        ('b16', 'b16'),
        f'{_NS}::pack_f16(static_cast<uint16_t>(s0), static_cast<uint16_t>(s1), mods, mode)',
        'b32',
    ),
    'V_CVT_I32_F32': _Spec(
        ('b32',), _to_integer('F32', 0, 'TRUNCATE', *_INT32, False), 'b32'
    ),
    'V_CVT_U32_F32': _Spec(
        ('b32',), _to_integer('F32', 0, 'TRUNCATE', *_UINT32, False), 'b32'
    ),
    'V_CVT_I16_F16': _Spec(
        ('b16',), _to_integer('F16', 0, 'TRUNCATE', '-32768', '32767', False), 'b16'
    ),
    'V_CVT_U16_F16': _Spec(
        ('b16',), _to_integer('F16', 0, 'TRUNCATE', '0', '65535', False), 'b16'
    ),
    'V_CVT_I32_F64': _Spec(
        ('b64',), _to_integer('F64', 0, 'TRUNCATE', *_INT32, False), 'b32'
    ),
    'V_CVT_U32_F64': _Spec(
        ('b64',), _to_integer('F64', 0, 'TRUNCATE', *_UINT32, False), 'b32'
    ),
    'V_CVT_FLOOR_I32_F32': _Spec(
        ('b32',), _to_integer('F32', 0, 'FLOOR', *_INT32, True), 'b32'
    ),
    'V_CVT_NEAREST_I32_F32': _Spec(
        ('b32',), _to_integer('F32', 0, 'NEAREST_UP', *_INT32, True), 'b32'
    ),
    'V_CVT_PK_I16_F32': _Spec(('b32', 'b32'), _pk_integer('-32768', '32767'), 'b32'),
    'V_CVT_PK_U16_F32': _Spec(('b32', 'b32'), _pk_integer('0', '65535'), 'b32'),
    'V_CVT_PK_U8_F32': _Spec(
        ('b32', 'b32', 'b32'),
        '[&]() { const uint32_t shift = (s1 & 3u) * 8u;'
        f' const uint32_t byte = static_cast<uint32_t>({_to_integer("F32", 0, "MODE", "0", "255", False)});'
        ' return (s2 & ~(0xffu << shift)) | (byte << shift); }()',
        'b32',
    ),
    'V_CVT_NORM_I16_F16': _Spec(
        ('b16',),
        f'{_NS}::normalize_f16(static_cast<uint16_t>(s0), true, 0, mods, mode)',
        'b16',
    ),
    'V_CVT_NORM_U16_F16': _Spec(
        ('b16',),
        f'{_NS}::normalize_f16(static_cast<uint16_t>(s0), false, 0, mods, mode)',
        'b16',
    ),
    'V_CVT_PK_NORM_I16_F16': _Spec(('b16', 'b16'), _pk_norm('true'), 'b32'),
    'V_CVT_PK_NORM_U16_F16': _Spec(('b16', 'b16'), _pk_norm('false'), 'b32'),
    'V_CVT_F32_FP8': _Spec(('b32',), _fp8_decode('FP8', '{byte}'), 'b32'),
    'V_CVT_F32_BF8': _Spec(('b32',), _fp8_decode('BF8', '{byte}'), 'b32'),
    'V_CVT_PK_F32_FP8': _Spec(('b32',), _fp8_pk_decode('FP8'), 'b64'),
    'V_CVT_PK_F32_BF8': _Spec(('b32',), _fp8_pk_decode('BF8'), 'b64'),
    'V_CVT_PK_FP8_F32': _Spec(('b32', 'b32'), _fp8_pk_encode('FP8'), 'b16_word'),
    'V_CVT_PK_BF8_F32': _Spec(('b32', 'b32'), _fp8_pk_encode('BF8'), 'b16_word'),
    'V_CVT_SR_FP8_F32': _Spec(('b32', 'b32'), _fp8_sr_encode('FP8'), 'byte'),
    'V_CVT_SR_BF8_F32': _Spec(('b32', 'b32'), _fp8_sr_encode('BF8'), 'byte'),
    'V_SAT_PK_U8_I16': _Spec(
        ('b32',),
        '[&]() { uint32_t r = 0; for (uint32_t i = 0; i < 2; ++i) {'
        ' const int32_t v = static_cast<int16_t>(s0 >> (16u * i));'
        ' r |= static_cast<uint32_t>(v < 0 ? 0 : (v > 255 ? 255 : v)) << (8u * i); }'
        ' return r; }()',
        'b16',
    ),
}

MEASURED_CONVERSIONS = frozenset(_SPECS)


def measured_conversion_body(
    name: str, is_vop3: bool, src_ops: list[str], dst_ops: list[str]
) -> str | None:
    """Return the complete execute body for a measured conversion, or None."""
    spec = _SPECS.get(name)
    if spec is None or not dst_ops or len(src_ops) < len(spec.sources):
        return None
    opsel = 'amdgpu::vop3_opsel(inst_)' if is_vop3 else '0u'
    lines = [
        f'  [[maybe_unused]] const {_NS}::Mode mode{{wf.fp_round_mode_f32(), wf.fp_round_mode_f16_f64(),',
        '      wf.fp_denorm_mode_f32(), wf.fp_denorm_mode_f16_f64(), wf.fp16_ovfl()};',
    ]
    if is_vop3:
        lines.append(
            f'  [[maybe_unused]] const {_NS}::Modifiers mods{{static_cast<uint32_t>(inst_.abs),'
            ' static_cast<uint32_t>(inst_.neg), inst_.clamp != 0,'
            ' static_cast<uint32_t>(inst_.omod)};'
        )
    else:
        lines.append(f'  [[maybe_unused]] const {_NS}::Modifiers mods{{}};')
    lines.append(f'  [[maybe_unused]] const uint32_t opsel = {opsel};')
    lines += [
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane)))',
        '      continue;',
    ]
    for i, kind in enumerate(spec.sources):
        src = src_ops[i]
        if kind == 'b64':
            read = f'amdgpu::RegisterAccess(wf).read_lane64({src}, lane)'
            lines.append(f'    const uint64_t s{i} = {read};')
            continue
        if kind == 'b16' and is_vop3:
            read = f'::rocjitsu::amdgpu::read_vop3_true16_src({src}, wf, lane, opsel, {i}) & 0xffffu'
        elif kind == 'b16':
            read = f'amdgpu::RegisterAccess(wf).read_lane({src}, lane) & 0xffffu'
        else:
            read = f'amdgpu::RegisterAccess(wf).read_lane({src}, lane)'
        lines.append(f'    const uint32_t s{i} = {read};')
    byte = '((opsel & 0x1u) << 1) | ((opsel & 0x2u) >> 1)' if is_vop3 else '0u'
    result = spec.result.replace('{byte}', byte)
    dst = dst_ops[0]
    if spec.write == 'b64':
        lines.append(f'    const uint64_t result = {result};')
        lines.append(
            f'    amdgpu::RegisterAccess(wf).write_lane64({dst}, lane, result);'
        )
    elif spec.write == 'b16':
        lines.append(
            f'    const uint32_t result = static_cast<uint32_t>({result}) & 0xffffu;'
        )
        if is_vop3:
            lines.append(
                f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst}, wf, lane, opsel, result, true);'
            )
        else:
            lines.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({dst}, lane, result);'
            )
    elif spec.write == 'b16_word':
        lines.append(f'    const uint32_t result = {result};')
        lines.append(
            f'    ::rocjitsu::amdgpu::write_vop3_true16_dst({dst}, wf, lane, opsel & 0x8u, result, true);'
        )
    elif spec.write == 'byte':
        lines.append(f'    const uint32_t result = {result};')
        lines += [
            '    const uint32_t shift = ((opsel >> 2) & 0x3u) * 8u;',
            f'    const uint32_t old = amdgpu::RegisterAccess(wf).read_lane({dst}, lane);',
            f'    amdgpu::RegisterAccess(wf).write_lane({dst}, lane,'
            ' (old & ~(0xffu << shift)) | (result << shift));',
        ]
    else:
        lines.append(f'    const uint32_t result = static_cast<uint32_t>({result});')
        lines.append(f'    amdgpu::RegisterAccess(wf).write_lane({dst}, lane, result);')
    lines.append('  }')
    return '\n'.join(lines)
