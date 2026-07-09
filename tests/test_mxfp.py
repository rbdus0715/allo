# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ml_dtypes
import allo
from allo.library.mxfp import mx_quantize_block, mx_quantize
from allo.ir.types import float32, uint8
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

    # data as raw per-element bytes (uint8[N//K, K]) so the wide Ty[N//K]
    # word never has to cross the numpy boundary directly.
    def kernel2(x: float32[N], scales: uint8[N // K], data_bits: uint8[N // K, K]):
        mx_scales: uint8[N // K]
        mx_data: Ty[N // K]
        mx_quantize[Ty, K, N](x, mx_scales, mx_data)
        for b in range(N // K):
            scales[b] = mx_scales[b]
            for i in range(K):
                data_bits[b, i] = mx_data[b][i * Ty.elem_bits : (i + 1) * Ty.elem_bits]

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
