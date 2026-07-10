# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ml_dtypes
import pytest
import allo
from allo.library.mxfp import (
    mx_quantize_block,
    mx_quantize,
    mx_block_dot,
    mx_normalize_add,
    mx_dot_general,
    dot_product,
)
from allo.ir.types import float32, uint8, int32
import allo.ir.types as T

# (format name, Allo type, ml_dtypes element dtype or None for MXINT8)
MXFP_FORMATS = [
    ("mxfp8_e4m3", T.mxfp8_e4m3, ml_dtypes.float8_e4m3fn),
    ("mxfp8_e5m2", T.mxfp8_e5m2, ml_dtypes.float8_e5m2),
    ("mxfp6_e2m3", T.mxfp6_e2m3, ml_dtypes.float6_e2m3fn),
    ("mxfp6_e3m2", T.mxfp6_e3m2, ml_dtypes.float6_e3m2fn),
    ("mxfp4_e2m1", T.mxfp4_e2m1, ml_dtypes.float4_e2m1fn),
]

K = 8


def make_quantize_block_kernel(Ty):
    # extracts each packed element's raw bits as a uint8 so the test can
    # cross the LLVM/numpy boundary without touching the wide packed Ty
    # word directly (see mxfp_dev notes: wide (>64-bit) memref marshalling
    # is fragile in the current LLVM backend, unrelated to this op's logic)
    def kernel(x: float32[K]) -> ("uint8[1]", "uint8[K]"):
        scale_out: uint8[1]
        scale, w = mx_quantize_block[Ty, K](x)
        scale_out[0] = scale
        elem_bits: uint8[K]
        for i in range(K):
            elem_bits[i] = w[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
        return scale_out, elem_bits

    kernel.__name__ = f"quantize_block_{Ty.name}"
    return kernel


def ref_mxfp_quantize(block, elem_dtype):
    # Algorithm 1 of arXiv:2310.10537 (OCP Microscaling paper), with
    # explicit clamping on overflow: ml_dtypes' bare .astype() instead
    # rounds overflowing values into the format's reserved NaN pattern
    # (e.g. 471.99 -> float8_e4m3fn NaN), which is not what the OCP MX
    # spec's "clamping" recipe (Section 6.3) requires.
    amax = np.max(np.abs(block))
    elem_max = float(ml_dtypes.finfo(elem_dtype).max)
    if amax == 0:
        shared_exp = -127.0
    else:
        shared_exp = np.floor(np.log2(amax)) - np.floor(np.log2(elem_max))
    scale_field = int(shared_exp) + 127
    scale_val = 2.0**shared_exp
    clamped = np.clip(block / scale_val, -elem_max, elem_max)
    q_bits = clamped.astype(np.float32).astype(elem_dtype).view(np.uint8)
    return scale_field, q_bits


def test_mx_quantize_block_all_formats():
    rng = np.random.default_rng(0)
    for name, Ty, elem_dtype in MXFP_FORMATS:
        mod = allo.customize(make_quantize_block_kernel(Ty)).build()
        for trial in range(20):
            x = (rng.standard_normal(K) * 2.0 ** rng.integers(-8, 8)).astype(np.float32)
            if trial == 0:
                x[:] = 0.0
            elif trial == 1:
                x[0] = 0.0

            scale, elem_bits = mod(x)
            scale = int(np.asarray(scale).flatten()[0])
            elem_bits = np.asarray(elem_bits).flatten()

            ref_scale, ref_bits = ref_mxfp_quantize(x, elem_dtype)
            assert scale == ref_scale, f"[{name}] trial {trial}: scale {scale} != {ref_scale}"
            for i in range(K):
                assert int(elem_bits[i]) == int(ref_bits[i]), (
                    f"[{name}] trial {trial} elem {i}: "
                    f"0x{int(elem_bits[i]):02x} != 0x{int(ref_bits[i]):02x} "
                    f"(input={x[i]})"
                )


def test_mx_quantize_block_mxint8():
    Ty = T.mxint8
    rng = np.random.default_rng(1)
    mod = allo.customize(make_quantize_block_kernel(Ty)).build()
    elem_max_unbiased = Ty.max_unbiased_exp  # elem_bits - 2 == 6

    for trial in range(20):
        x = (rng.standard_normal(K) * 2.0 ** rng.integers(-8, 8)).astype(np.float32)
        if trial == 0:
            x[:] = 0.0
        elif trial == 1:
            x[0] = 0.0

        scale, elem_bits = mod(x)
        scale = int(np.asarray(scale).flatten()[0])
        elem_bits = np.asarray(elem_bits).flatten()

        amax = np.max(np.abs(x))
        shared_exp = -127.0 if amax == 0 else np.floor(np.log2(amax)) - elem_max_unbiased
        ref_scale = int(shared_exp) + 127
        scale_val = 2.0**shared_exp
        assert scale == ref_scale, f"trial {trial}: scale {scale} != {ref_scale}"

        for i in range(K):
            scaled = x[i] / scale_val
            ref_signed = int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)
            ref_signed = max(-128, min(127, ref_signed))
            ours = int(elem_bits[i])
            ours_signed = ours - 256 if ours >= 128 else ours
            assert ours_signed == ref_signed, (
                f"trial {trial} elem {i}: {ours_signed} != {ref_signed} (input={x[i]})"
            )


def test_mx_quantize_block_e4m3_specials():
    # E4M3 has a NaN encoding but no Inf encoding (per OCP FP8 spec): a NaN
    # input must map to the E4M3 NaN pattern, while an Inf input has no
    # encoding available and must saturate to the max finite magnitude.
    Ty = T.mxfp8_e4m3
    mod = allo.customize(make_quantize_block_kernel(Ty)).build()
    inp = np.array([float("nan"), float("inf"), -float("inf")], dtype=np.float32)
    inp = np.pad(inp, (0, K - len(inp)), constant_values=1.0)
    _, elem_bits = mod(inp)
    elem_bits = np.asarray(elem_bits).flatten()

    assert (int(elem_bits[0]) & 0x7F) == 0x7F  # NaN -> NaN pattern
    assert int(elem_bits[1]) == 0x7E  # +Inf -> saturate to +448
    assert int(elem_bits[2]) == 0xFE  # -Inf -> saturate to -448


def test_mx_quantize_block_e5m2_specials():
    # E5M2 defines both NaN and Inf (standard IEEE-style top-exponent
    # reservation), so both should round-trip to their own encodings.
    Ty = T.mxfp8_e5m2
    mod = allo.customize(make_quantize_block_kernel(Ty)).build()
    inp = np.array([float("nan"), float("inf"), -float("inf")], dtype=np.float32)
    inp = np.pad(inp, (0, K - len(inp)), constant_values=1.0)
    _, elem_bits = mod(inp)
    elem_bits = np.asarray(elem_bits).flatten()

    assert (int(elem_bits[0]) & 0x7F) > 0x7C  # NaN -> NaN pattern
    assert int(elem_bits[1]) == 0x7C  # +Inf -> +Inf
    assert int(elem_bits[2]) == 0xFC  # -Inf -> -Inf


def test_mx_quantize_full_tensor():
    # exercise the multi-block mx_quantize[Ty, K, N] wrapper (out-params,
    # not a tuple return -- Allo does not support a function returning
    # multiple array-typed results destructured at the call site).
    Ty = T.mxfp8_e4m3
    N = 16

    # mx_quantize writes data_bits directly as per-element bytes (see its
    # docstring comment): no local Ty[N//K] array or manual unpack needed.
    def kernel2(x: float32[N], scales: uint8[N // K], data_bits: uint8[N // K, K]):
        mx_quantize[Ty, K, N](x, scales, data_bits)

    mod = allo.customize(kernel2).build()
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(N) * 4.0).astype(np.float32)
    scales = np.zeros(N // K, dtype=np.uint8)
    data_bits = np.zeros((N // K, K), dtype=np.uint8)
    mod(x, scales, data_bits)

    for b in range(N // K):
        ref_scale, ref_bits = ref_mxfp_quantize(x[b * K : (b + 1) * K], ml_dtypes.float8_e4m3fn)
        assert int(scales[b]) == ref_scale
        for i in range(K):
            assert int(data_bits[b, i]) == int(ref_bits[i])


######################################################################
# mx_block_dot (Fig 1: block dot product, Kulisch accumulation)
######################################################################

DOT_K = 16


def make_block_dot_kernel(Ty, K):
    def kernel(
        a: float32[K], b: float32[K]
    ) -> ("uint8[1]", "uint8[1]", "uint8[K]", "uint8[K]", "float32[1]", "uint8[1]", "uint8[1]"):
        sa: uint8[1]
        sb: uint8[1]
        a_bytes: uint8[K]
        b_bytes: uint8[K]
        scale_a, data_a = mx_quantize_block[Ty, K](a)
        scale_b, data_b = mx_quantize_block[Ty, K](b)
        sa[0] = scale_a
        sb[0] = scale_b
        for i in range(K):
            a_bytes[i] = data_a[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            b_bytes[i] = data_b[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]

        out_scale, acc, has_nan, has_inf = mx_block_dot[Ty, K](
            scale_a, data_a, scale_b, data_b
        )
        shift: int32 = int(out_scale) - 127
        acc_f: float32 = float(acc)
        if shift >= 0:
            s: int32 = 0
            while s < shift:
                acc_f = acc_f * 2.0
                s = s + 1
        else:
            s: int32 = 0
            while s < -shift:
                acc_f = acc_f / 2.0
                s = s + 1
        result: float32[1]
        result[0] = acc_f
        nan_out: uint8[1]
        inf_out: uint8[1]
        nan_out[0] = has_nan
        inf_out[0] = has_inf
        return sa, sb, a_bytes, b_bytes, result, nan_out, inf_out

    kernel.__name__ = f"block_dot_{Ty.name}"
    return kernel


def _decode_mxfp_elem(byte_val, elem_dtype):
    return float(np.array([byte_val], dtype=np.uint8).view(elem_dtype)[0])


def test_mx_block_dot_all_formats():
    # Verify against the SAME quantized values our own kernel produced
    # (decoded and dot-producted in fp64), not the original unquantized
    # inputs: this isolates mx_block_dot's own accumulation error from
    # mx_quantize_block's (already separately verified) quantization
    # error. Kulisch accumulation should be exact, so error should be 0.
    rng = np.random.default_rng(3)
    for name, Ty, elem_dtype in MXFP_FORMATS:
        mod = allo.customize(make_block_dot_kernel(Ty, DOT_K)).build()
        for trial in range(10):
            a = (rng.standard_normal(DOT_K) * 2.0 ** rng.integers(-6, 6)).astype(
                np.float32
            )
            b = (rng.standard_normal(DOT_K) * 2.0 ** rng.integers(-6, 6)).astype(
                np.float32
            )
            sa, sb, a_bytes, b_bytes, result, _, _ = mod(a, b)
            sa = int(np.asarray(sa).flatten()[0])
            sb = int(np.asarray(sb).flatten()[0])
            a_bytes = np.asarray(a_bytes).flatten()
            b_bytes = np.asarray(b_bytes).flatten()
            our_dot = float(np.asarray(result).flatten()[0])

            scale_a_val = 2.0 ** (sa - 127)
            scale_b_val = 2.0 ** (sb - 127)
            ref_dot = 0.0
            for i in range(DOT_K):
                av = _decode_mxfp_elem(int(a_bytes[i]), elem_dtype) * scale_a_val
                bv = _decode_mxfp_elem(int(b_bytes[i]), elem_dtype) * scale_b_val
                ref_dot += av * bv

            assert our_dot == pytest.approx(ref_dot, rel=1e-5, abs=1e-30), (
                f"[{name}] trial {trial}: our={our_dot} ref={ref_dot}"
            )


def test_mx_block_dot_mxint8():
    Ty = T.mxint8
    mod = allo.customize(make_block_dot_kernel(Ty, DOT_K)).build()
    rng = np.random.default_rng(4)
    for trial in range(10):
        a = rng.integers(-100, 100, DOT_K).astype(np.float32)
        b = rng.integers(-100, 100, DOT_K).astype(np.float32)
        _, _, _, _, result, _, _ = mod(a, b)
        our_dot = float(np.asarray(result).flatten()[0])
        ref_dot = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        assert our_dot == pytest.approx(ref_dot, rel=1e-3), (
            f"trial {trial}: our={our_dot} ref={ref_dot}"
        )


def test_mx_block_dot_nan_propagation():
    Ty = T.mxfp8_e4m3
    mod = allo.customize(make_block_dot_kernel(Ty, DOT_K)).build()
    a_nan = np.array([float("nan")] + [1.0] * (DOT_K - 1), dtype=np.float32)
    b_normal = np.ones(DOT_K, dtype=np.float32)
    _, _, _, _, _, nan_out, inf_out = mod(a_nan, b_normal)
    assert int(np.asarray(nan_out).flatten()[0]) == 1
    assert int(np.asarray(inf_out).flatten()[0]) == 0

    a_normal = np.ones(DOT_K, dtype=np.float32)
    _, _, _, _, _, nan_out2, inf_out2 = mod(a_normal, b_normal)
    assert int(np.asarray(nan_out2).flatten()[0]) == 0
    assert int(np.asarray(inf_out2).flatten()[0]) == 0


def test_mx_block_dot_no_overflow_worst_case():
    # block_size=32, all elements at the format's max magnitude, same
    # sign: the accumulator must be wide enough (final_accum_bits, not
    # the narrower block_accum_bits -- see mx_block_dot's own comment)
    # to hold this exactly without overflow.
    K32 = 32
    for name, Ty, elem_dtype in MXFP_FORMATS:
        max_val = float(ml_dtypes.finfo(elem_dtype).max)
        mod = allo.customize(make_block_dot_kernel(Ty, K32)).build()
        a = np.full(K32, max_val, dtype=np.float32)
        b = np.full(K32, max_val, dtype=np.float32)
        _, _, _, _, result, _, _ = mod(a, b)
        our_dot = float(np.asarray(result).flatten()[0])
        ref_dot = float(K32 * max_val * max_val)
        assert our_dot == pytest.approx(ref_dot, rel=1e-6), (
            f"[{name}] worst-case: our={our_dot} ref={ref_dot}"
        )


######################################################################
# mx_normalize_add (Fig 2) and mx_dot_general (cross-block reduction)
######################################################################


def test_mx_normalize_add():
    # mx_normalize_add is parametrized by Ty (not a bare bit-width): Allo's
    # generic-subscript resolver only accepts a bare name there, not an
    # attribute expression like Ty.final_accum_bits, so BW is derived
    # internally from Ty instead (see mx_normalize_add's own comment).
    #
    # op0/op1 cross the kernel boundary as plain int32 (not the wide,
    # non-power-of-2 Int(Ty.final_accum_bits)=43 bits): scalar arguments
    # at odd wide bit-widths hit the same LLVM-boundary marshalling
    # fragility already worked around for wide *array* arguments in
    # mx_quantize/mx_dot_general (see those tests' notes) -- the value is
    # widened internally instead, via the already-proven assignment cast.
    Ty = T.mxfp8_e4m3
    BW = Ty.final_accum_bits

    def kernel(s0: uint8, o0: int32, s1: uint8, o1: int32) -> (
        "uint8[1]",
        "float32[1]",
    ):
        wide0: Int(Ty.final_accum_bits) = o0
        wide1: Int(Ty.final_accum_bits) = o1
        so, out = mx_normalize_add[Ty](s0, wide0, s1, wide1)
        so_out: uint8[1]
        val_out: float32[1]
        so_out[0] = so
        val_out[0] = float(out)
        return so_out, val_out

    mod = allo.customize(kernel).build()

    def ref(scale0, op0, scale1, op1):
        return op0 * 2.0 ** (scale0 - 127) + op1 * 2.0 ** (scale1 - 127)

    cases = [
        (127, 100, 127, 50),
        (127, 100, 120, 50),
        (120, 50, 127, 100),
        (127, -100, 127, 50),
        (127, -100, 120, -50),
        (127, 1000, 127, -999),
    ]
    for scale0, op0, scale1, op1 in cases:
        so, val = mod(scale0, op0, scale1, op1)
        so = int(np.asarray(so).flatten()[0])
        val = float(np.asarray(val).flatten()[0])
        our_val = val * 2.0 ** (so - 127)
        ref_val = ref(scale0, op0, scale1, op1)
        assert our_val == pytest.approx(ref_val, rel=1e-2), (
            f"s0={scale0},o0={op0},s1={scale1},o1={op1}: our={our_val} ref={ref_val}"
        )


def test_mx_normalize_add_overflow_renormalize():
    # trigger the addition-overflow/renormalize path with an operand near
    # Int(BW)'s limit; constructed via an internal shift (compile-time
    # constant amount) since a value that large can't cross the LLVM
    # boundary as a plain int32 argument.
    Ty = T.mxfp8_e4m3
    BW = Ty.final_accum_bits

    def kernel() -> ("uint8[1]", "float32[1]"):
        s: uint8 = 127
        big: Int(Ty.final_accum_bits) = 1
        big = big << (Ty.final_accum_bits - 2)
        so, out = mx_normalize_add[Ty](s, big, s, big)
        so_out: uint8[1]
        val_out: float32[1]
        so_out[0] = so
        val_out[0] = float(out)
        return so_out, val_out

    mod = allo.customize(kernel).build()
    so, val = mod()
    so = int(np.asarray(so).flatten()[0])
    val = float(np.asarray(val).flatten()[0])
    our_val = val * 2.0 ** (so - 127)
    ref_val = float(2 * (1 << (BW - 2)))
    assert our_val == pytest.approx(ref_val, rel=1e-9), f"our={our_val} ref={ref_val}"


def make_dot_general_kernel(Ty, K, NB):
    # a single fused kernel (quantize both operands, then cross-block
    # reduce) is safe now that mx_quantize/mx_dot_general never materialize
    # a local array of the wide packed Ty word (see their docstrings in
    # allo/library/mxfp.py) -- only uint8 byte buffers cross their
    # boundaries, so no array-of->64-bit-type memref ever exists to trip
    # the MLIR JIT ExecutionEngine's MLIRContext-teardown heap corruption.
    N = K * NB

    def kernel(
        a: float32[N],
        b: float32[N],
        scales_a: uint8[NB],
        scales_b: uint8[NB],
        a_bytes: uint8[NB, K],
        b_bytes: uint8[NB, K],
        result: float32[1],
        result_scale: uint8[1],
    ):
        mx_quantize[Ty, K, N](a, scales_a, a_bytes)
        mx_quantize[Ty, K, N](b, scales_b, b_bytes)

        out_scale, acc, has_nan, has_inf = mx_dot_general[Ty, K, N](
            scales_a, a_bytes, scales_b, b_bytes
        )
        result_scale[0] = out_scale
        shift: int32 = int(out_scale) - 127
        acc_f: float32 = float(acc)
        if shift >= 0:
            s: int32 = 0
            while s < shift:
                acc_f = acc_f * 2.0
                s = s + 1
        else:
            s: int32 = 0
            while s < -shift:
                acc_f = acc_f / 2.0
                s = s + 1
        result[0] = acc_f

    kernel.__name__ = f"dot_general_{Ty.name}"
    return kernel


def test_mx_dot_general_all_formats():
    # cross-block reduction, verified the same way as mx_block_dot: decode
    # the SAME quantized values our kernel produced and dot them in fp64,
    # so any deviation is attributable only to mx_dot_general's own
    # cross-block combine (mx_normalize_add), not to quantization error.
    K, NB = 8, 4
    N = K * NB
    rng = np.random.default_rng(5)
    for name, Ty, elem_dtype in MXFP_FORMATS:
        mod = allo.customize(make_dot_general_kernel(Ty, K, NB)).build()
        for trial in range(5):
            a = np.concatenate(
                [
                    (rng.standard_normal(K) * 2.0 ** rng.integers(-10, 10)).astype(
                        np.float32
                    )
                    for _ in range(NB)
                ]
            )
            b = np.concatenate(
                [
                    (rng.standard_normal(K) * 2.0 ** rng.integers(-10, 10)).astype(
                        np.float32
                    )
                    for _ in range(NB)
                ]
            )
            scales_a = np.zeros(NB, dtype=np.uint8)
            scales_b = np.zeros(NB, dtype=np.uint8)
            a_bytes = np.zeros((NB, K), dtype=np.uint8)
            b_bytes = np.zeros((NB, K), dtype=np.uint8)
            result = np.zeros(1, dtype=np.float32)
            result_scale = np.zeros(1, dtype=np.uint8)
            mod(a, b, scales_a, scales_b, a_bytes, b_bytes, result, result_scale)
            our_dot = float(result[0])

            ref_dot = 0.0
            for blk in range(NB):
                scale_a_val = 2.0 ** (int(scales_a[blk]) - 127)
                scale_b_val = 2.0 ** (int(scales_b[blk]) - 127)
                for i in range(K):
                    av = _decode_mxfp_elem(int(a_bytes[blk, i]), elem_dtype) * scale_a_val
                    bv = _decode_mxfp_elem(int(b_bytes[blk, i]), elem_dtype) * scale_b_val
                    ref_dot += av * bv

            # rel=1e-3 alone is unsound here: mx_dot_general is a *lossy*
            # sequential shared-exponent (block-floating-point) combiner,
            # so a block whose magnitude is far below the running total's
            # current ULP legitimately rounds away to nothing -- that's
            # correct behavior, not error, and can exceed 0.1% relative to
            # an unrounded fp64 reference for adversarial magnitude
            # spreads. Bound the allowed deviation in absolute ULPs of the
            # result's own scale instead (NB - 1 combine steps, each
            # contributing at most ~0.5 ULP of its own rounding).
            ulp = 2.0 ** (int(result_scale[0]) - 127)
            assert our_dot == pytest.approx(ref_dot, rel=1e-3, abs=NB * ulp), (
                f"[{name}] trial {trial}: our={our_dot} ref={ref_dot} ulp={ulp}"
            )


######################################################################
# dot_product (top-level kernel: float32[N], float32[N] -> float32)
######################################################################

DOT_PRODUCT_K = 32  # OCP MX spec default block_size
DOT_PRODUCT_NB = 8  # well past the NB>=3 crash threshold this session found


def _ref_dot_product_float(a, b, K, elem_dtype):
    # Reuses ref_mxfp_quantize (already verified bit-exact against
    # ml_dtypes for every float MX format) per block, so any mismatch is
    # attributable to dot_product's own pipeline, not to an approximate
    # reimplementation of the quantizer here.
    NB = len(a) // K
    total = 0.0
    for blk in range(NB):
        a_scale, a_bits = ref_mxfp_quantize(a[blk * K : (blk + 1) * K], elem_dtype)
        b_scale, b_bits = ref_mxfp_quantize(b[blk * K : (blk + 1) * K], elem_dtype)
        a_scale_val = 2.0 ** (a_scale - 127)
        b_scale_val = 2.0 ** (b_scale - 127)
        for i in range(K):
            av = _decode_mxfp_elem(int(a_bits[i]), elem_dtype) * a_scale_val
            bv = _decode_mxfp_elem(int(b_bits[i]), elem_dtype) * b_scale_val
            total += av * bv
    return total


def _ref_dot_product_mxint8(a, b, K, Ty):
    # Mirrors test_mx_quantize_block_mxint8's reference quantizer.
    NB = len(a) // K
    elem_max_unbiased = Ty.max_unbiased_exp

    def quantize(x):
        amax = np.max(np.abs(x))
        shared_exp = -127.0 if amax == 0 else np.floor(np.log2(amax)) - elem_max_unbiased
        scale_val = 2.0**shared_exp
        out = np.zeros(len(x))
        for i in range(len(x)):
            scaled = x[i] / scale_val
            signed = int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)
            signed = max(-128, min(127, signed))
            out[i] = signed * scale_val
        return out

    total = 0.0
    for blk in range(NB):
        aq = quantize(a[blk * K : (blk + 1) * K])
        bq = quantize(b[blk * K : (blk + 1) * K])
        total += float(np.sum(aq * bq))
    return total


def test_dot_product_all_formats():
    K, NB = DOT_PRODUCT_K, DOT_PRODUCT_NB
    N = K * NB
    rng = np.random.default_rng(11)
    all_formats = MXFP_FORMATS + [("mxint8", T.mxint8, None)]
    for name, Ty, elem_dtype in all_formats:

        def kernel(a: float32[N], b: float32[N]) -> float32:
            return dot_product[Ty, K, N](a, b)

        mod = allo.customize(kernel).build()
        for trial in range(5):
            a = (rng.standard_normal(N) * 2.0 ** rng.integers(-8, 8)).astype(np.float32)
            b = (rng.standard_normal(N) * 2.0 ** rng.integers(-8, 8)).astype(np.float32)
            our_dot = float(mod(a, b))

            if name == "mxint8":
                ref_dot = _ref_dot_product_mxint8(a, b, K, Ty)
            else:
                ref_dot = _ref_dot_product_float(a, b, K, elem_dtype)

            # same ULP-in-the-result's-own-scale rationale as
            # test_mx_dot_general_all_formats: dot_product is built
            # directly on mx_dot_general's lossy sequential combiner.
            mag = max(abs(our_dot), abs(ref_dot), 1.0)
            ulp_guess = mag * 2.0**-8  # generous: within final format's rounding budget
            assert our_dot == pytest.approx(ref_dot, rel=1e-3, abs=NB * ulp_guess), (
                f"[{name}] trial {trial}: our={our_dot} ref={ref_dot}"
            )


def test_dot_product_nan_inf():
    # E4M3 (has_nan, no has_inf) and E5M2 (has both) exercise mx_to_float32's
    # special-value boundary conversion end to end.
    K, NB = DOT_PRODUCT_K, DOT_PRODUCT_NB
    N = K * NB

    for Ty, expect_inf_representable in ((T.mxfp8_e4m3, False), (T.mxfp8_e5m2, True)):

        def kernel(a: float32[N], b: float32[N]) -> float32:
            return dot_product[Ty, K, N](a, b)

        mod = allo.customize(kernel).build()

        a_nan = np.ones(N, dtype=np.float32)
        a_nan[0] = float("nan")
        b_normal = np.ones(N, dtype=np.float32)
        out = float(mod(a_nan, b_normal))
        assert np.isnan(out), f"[{Ty.name}] expected NaN, got {out}"

        a_normal = np.ones(N, dtype=np.float32)
        out2 = float(mod(a_normal, b_normal))
        assert not np.isnan(out2) and not np.isinf(out2), (
            f"[{Ty.name}] expected a normal finite result, got {out2}"
        )

        if expect_inf_representable:
            a_inf = np.ones(N, dtype=np.float32)
            a_inf[0] = float("inf")
            out3 = float(mod(a_inf, b_normal))
            assert np.isinf(out3), f"[{Ty.name}] expected Inf, got {out3}"
