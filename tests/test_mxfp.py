# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.library.mxfp import (
    mx_quantize_block,
    mx_quantize,
    mx_block_dot,
    mx_dot_general,
    dot_product,
)
from allo.ir.types import float32, uint8, int32
import allo.ir.types as T

# (format name, Allo type)
MXFP_FORMATS = [
    ("mxfp8_e4m3", T.mxfp8_e4m3),
    ("mxfp8_e5m2", T.mxfp8_e5m2),
    ("mxfp6_e2m3", T.mxfp6_e2m3),
    ("mxfp6_e3m2", T.mxfp6_e3m2),
    ("mxfp4_e2m1", T.mxfp4_e2m1),
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


def _round_to_nearest_even(mant, shift):
    kept = mant >> shift
    remainder = mant & ((1 << shift) - 1)
    halfpoint = 1 << (shift - 1)
    if remainder > halfpoint or (remainder == halfpoint and (kept & 1)):
        kept += 1
    return kept


def _encode_narrow_fp_elem(v, shared_exp, Ty):
    # Mirrors allo.library.mxfp._mx_quantize_elem_fp's bit manipulation
    # directly (round-to-nearest-even, full range, no reserved NaN/Inf
    # codes) -- there is no third-party reference for this exact
    # (deliberately simplified) element format to cross-check against.
    exp_bits, mantissa_bits, bias = Ty.exp_bits, Ty.mantissa_bits, Ty.bias
    bits = int(np.float32(v).view(np.uint32))
    sign = (bits >> 31) & 1
    exp_field = (bits >> 23) & 0xFF
    mant32 = bits & 0x7FFFFF
    max_exp_code = (1 << exp_bits) - 1
    max_mant_code = (1 << mantissa_bits) - 1

    if exp_field == 0:
        return sign << (exp_bits + mantissa_bits)

    unbiased_exp = exp_field - 127
    target_exp = unbiased_exp - shared_exp + bias
    full_mant = (1 << 23) | mant32

    if target_exp <= 0:
        total_shift = (23 - mantissa_bits) + (1 - target_exp)
        if total_shift > 24:
            result_exp, result_mant = 0, 0
        else:
            kept = _round_to_nearest_even(full_mant, total_shift)
            if kept >= (1 << mantissa_bits):
                result_exp, result_mant = 1, 0
            else:
                result_exp, result_mant = 0, kept
    else:
        total_shift = 23 - mantissa_bits
        kept = _round_to_nearest_even(full_mant, total_shift)
        if kept >= (1 << (mantissa_bits + 1)):
            result_exp, result_mant = target_exp + 1, 0
        else:
            result_exp, result_mant = target_exp, kept & max_mant_code

    if result_exp > max_exp_code or (
        result_exp == max_exp_code and result_mant > max_mant_code
    ):
        result_exp, result_mant = max_exp_code, max_mant_code

    return (sign << (exp_bits + mantissa_bits)) | (result_exp << mantissa_bits) | result_mant


def ref_mxfp_quantize(block, Ty):
    # Mirrors mx_quantize_block's own shared-exponent search (raw IEEE-754
    # exponent field, not a log2) and per-element rounding exactly.
    bits = block.astype(np.float32).view(np.uint32)
    exp_fields = (bits >> 23) & 0xFF
    max_exp_field = int(np.max(exp_fields)) if len(exp_fields) else 0
    shared_exp = max_exp_field - 127 - Ty.max_unbiased_exp
    shared_exp = min(max(shared_exp, -127), 127)
    scale_field = shared_exp + 127
    q_bits = np.array(
        [_encode_narrow_fp_elem(v, shared_exp, Ty) for v in block], dtype=np.uint8
    )
    return scale_field, q_bits


def test_mx_quantize_block_all_formats():
    rng = np.random.default_rng(0)
    for name, Ty in MXFP_FORMATS:
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

            ref_scale, ref_bits = ref_mxfp_quantize(x, Ty)
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
        ref_scale, ref_bits = ref_mxfp_quantize(x[b * K : (b + 1) * K], Ty)
        assert int(scales[b]) == ref_scale
        for i in range(K):
            assert int(data_bits[b, i]) == int(ref_bits[i])


######################################################################
# mx_block_dot (dequantize each element to float32, accumulate in float32)
######################################################################

DOT_K = 16


def make_block_dot_kernel(Ty, K):
    def kernel(
        a: float32[K], b: float32[K]
    ) -> ("uint8[1]", "uint8[1]", "uint8[K]", "uint8[K]", "float32[1]"):
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

        result: float32[1]
        result[0] = mx_block_dot[Ty, K](scale_a, data_a, scale_b, data_b)
        return sa, sb, a_bytes, b_bytes, result

    kernel.__name__ = f"block_dot_{Ty.name}"
    return kernel


def _decode_mxfp_elem(byte_val, Ty):
    exp_bits, mantissa_bits, bias = Ty.exp_bits, Ty.mantissa_bits, Ty.bias
    sign = (byte_val >> (exp_bits + mantissa_bits)) & 1
    exp_field = (byte_val >> mantissa_bits) & ((1 << exp_bits) - 1)
    mant = byte_val & ((1 << mantissa_bits) - 1)
    if exp_field == 0:
        val = (mant / (1 << mantissa_bits)) * 2.0 ** (1 - bias)
    else:
        val = (1 + mant / (1 << mantissa_bits)) * 2.0 ** (exp_field - bias)
    return -val if sign else val


def test_mx_block_dot_all_formats():
    # Verify against the SAME quantized values our own kernel produced
    # (decoded and dot-producted in fp64), not the original unquantized
    # inputs: this isolates mx_block_dot's own accumulation error from
    # mx_quantize_block's (already separately verified) quantization
    # error. mx_block_dot dequantizes to float32 and accumulates in
    # float32 (no exact/Kulisch accumulator), so some rounding is expected.
    rng = np.random.default_rng(3)
    for name, Ty in MXFP_FORMATS:
        mod = allo.customize(make_block_dot_kernel(Ty, DOT_K)).build()
        for trial in range(10):
            a = (rng.standard_normal(DOT_K) * 2.0 ** rng.integers(-6, 6)).astype(
                np.float32
            )
            b = (rng.standard_normal(DOT_K) * 2.0 ** rng.integers(-6, 6)).astype(
                np.float32
            )
            sa, sb, a_bytes, b_bytes, result = mod(a, b)
            sa = int(np.asarray(sa).flatten()[0])
            sb = int(np.asarray(sb).flatten()[0])
            a_bytes = np.asarray(a_bytes).flatten()
            b_bytes = np.asarray(b_bytes).flatten()
            our_dot = float(np.asarray(result).flatten()[0])

            scale_a_val = 2.0 ** (sa - 127)
            scale_b_val = 2.0 ** (sb - 127)
            ref_dot = 0.0
            for i in range(DOT_K):
                av = _decode_mxfp_elem(int(a_bytes[i]), Ty) * scale_a_val
                bv = _decode_mxfp_elem(int(b_bytes[i]), Ty) * scale_b_val
                ref_dot += av * bv

            assert our_dot == pytest.approx(ref_dot, rel=1e-4, abs=1e-30), (
                f"[{name}] trial {trial}: our={our_dot} ref={ref_dot}"
            )


def test_mx_block_dot_mxint8():
    Ty = T.mxint8
    mod = allo.customize(make_block_dot_kernel(Ty, DOT_K)).build()
    rng = np.random.default_rng(4)
    for trial in range(10):
        a = rng.integers(-100, 100, DOT_K).astype(np.float32)
        b = rng.integers(-100, 100, DOT_K).astype(np.float32)
        _, _, _, _, result = mod(a, b)
        our_dot = float(np.asarray(result).flatten()[0])
        ref_dot = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        assert our_dot == pytest.approx(ref_dot, rel=1e-3), (
            f"trial {trial}: our={our_dot} ref={ref_dot}"
        )


######################################################################
# mx_dot_general (cross-block reduction)
######################################################################


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
    ):
        mx_quantize[Ty, K, N](a, scales_a, a_bytes)
        mx_quantize[Ty, K, N](b, scales_b, b_bytes)
        result[0] = mx_dot_general[Ty, K, N](scales_a, a_bytes, scales_b, b_bytes)

    kernel.__name__ = f"dot_general_{Ty.name}"
    return kernel


def test_mx_dot_general_all_formats():
    # cross-block reduction, verified the same way as mx_block_dot: decode
    # the SAME quantized values our kernel produced and dot them in fp64,
    # so any deviation is attributable only to mx_dot_general's own
    # cross-block float32 accumulation, not to quantization error.
    K, NB = 8, 4
    N = K * NB
    rng = np.random.default_rng(5)
    for name, Ty in MXFP_FORMATS:
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
            mod(a, b, scales_a, scales_b, a_bytes, b_bytes, result)
            our_dot = float(result[0])

            ref_dot = 0.0
            for blk in range(NB):
                scale_a_val = 2.0 ** (int(scales_a[blk]) - 127)
                scale_b_val = 2.0 ** (int(scales_b[blk]) - 127)
                for i in range(K):
                    av = _decode_mxfp_elem(int(a_bytes[blk, i]), Ty) * scale_a_val
                    bv = _decode_mxfp_elem(int(b_bytes[blk, i]), Ty) * scale_b_val
                    ref_dot += av * bv

            # plain float32 accumulation across NB*K terms -- bound the
            # allowed deviation relative to the result's own magnitude
            # rather than an exact match.
            mag = max(abs(our_dot), abs(ref_dot), 1.0)
            tol = mag * 2.0**-10
            assert our_dot == pytest.approx(ref_dot, rel=1e-3, abs=tol), (
                f"[{name}] trial {trial}: our={our_dot} ref={ref_dot}"
            )


######################################################################
# dot_product (top-level kernel: float32[N], float32[N] -> float32)
######################################################################

DOT_PRODUCT_K = 32  # OCP MX spec default block_size
DOT_PRODUCT_NB = 8  # well past the NB>=3 crash threshold this session found


def _ref_dot_product_float(a, b, K, Ty):
    # Reuses ref_mxfp_quantize (mirrors mx_quantize_block bit-for-bit) per
    # block, so any mismatch is attributable to dot_product's own pipeline,
    # not to an approximate reimplementation of the quantizer here.
    NB = len(a) // K
    total = 0.0
    for blk in range(NB):
        a_scale, a_bits = ref_mxfp_quantize(a[blk * K : (blk + 1) * K], Ty)
        b_scale, b_bits = ref_mxfp_quantize(b[blk * K : (blk + 1) * K], Ty)
        a_scale_val = 2.0 ** (a_scale - 127)
        b_scale_val = 2.0 ** (b_scale - 127)
        for i in range(K):
            av = _decode_mxfp_elem(int(a_bits[i]), Ty) * a_scale_val
            bv = _decode_mxfp_elem(int(b_bits[i]), Ty) * b_scale_val
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
    all_formats = MXFP_FORMATS + [("mxint8", T.mxint8)]
    for name, Ty in all_formats:

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
                ref_dot = _ref_dot_product_float(a, b, K, Ty)

            # same rationale as test_mx_dot_general_all_formats: dot_product
            # is built directly on mx_dot_general's float32 accumulation.
            mag = max(abs(our_dot), abs(ref_dot), 1.0)
            ulp_guess = mag * 2.0**-8  # generous: within final format's rounding budget
            assert our_dot == pytest.approx(ref_dot, rel=1e-3, abs=NB * ulp_guess), (
                f"[{name}] trial {trial}: our={our_dot} ref={ref_dot}"
            )
