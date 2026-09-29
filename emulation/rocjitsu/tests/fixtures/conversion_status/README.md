# VALU conversion exception-status fixture

`gfx1201_cases.h` holds exception status and result bits of VALU conversions
captured on a physical Radeon RX 9070 XT (`gfx1201`) on 2026-09-29 with
ROCm 7.1.0 (HIP 7.1.25424). `conversion_status_test.cpp` consumes the captured
bits directly and requires no GPU.

Each capture ran one wave32 workgroup that set MODE, seeded
`HW_REG_WAVE_EXCP_FLAG_USER` (hwreg 18), installed an EXEC mask, executed
exactly one instruction under test and read the status back. Traps were
disabled (`TRAP_CTRL` read 0). Status bits [6:0] use the TRAPSTS layout:
invalid, input denormal, float divide-by-zero, overflow, underflow, inexact
and integer divide-by-zero. Destinations were preset to `0xa5a5a5a5` so
preserved halves and bytes and inactive lanes are visible.

The full capture has 95,073 rows over 26 conversion opcodes, 127
encoding/modifier variants and 11 MODE values (every rounding mode, input and
output denormal combinations, and FP16_OVFL), plus V_MUL_F32 rows that
reproduce the plain-multiply witnesses in `hwfloat_mul_f32_test.cpp`. Its
SHA-256 is recorded in the header. The distilled fixture keeps the first
capture of every behaviour class, keyed by variant, the conversion facts of
the participating lanes, the raised causes, and whether the row exercised a
single lane or wave attribution (zero EXEC, exceptional inactive lanes, one
exceptional lane among many, and seeded status). Lanes whose sources equal a
row's fill value share one recorded result.

The captures qualify the causes in `conversion.h` `CauseRule` for the listed
opcodes, encodings and modifiers on gfx1201. Forms outside the capture remain
unqualified and report no causes: V_PACK_B32_F16, V_CVT_PK_{I16,U16,U8}_F32,
V_CVT_{PK_,}NORM_{I16,U16}_F16, V_CVT_PK_F32_{FP8,BF8}, V_SAT_PK_U8_I16 and
the scalar S_CVT_* conversions.
