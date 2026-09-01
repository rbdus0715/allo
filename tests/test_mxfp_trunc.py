# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
from allo.backend.config import PART_NUMBER
from allo.library.mxfp import (
    mx_quantize_block_f32_trunc,
    mx_block_dot_trunc,
    block_dot_product_trunc,
    schedule_mx_quantize_block_f32_trunc,
    schedule_mx_block_dot_trunc,
    schedule_block_dot_product_trunc,
    make_mx_dot_general_dataflow_trunc,
    patch_extern_c_for_class_return_types,
)
from allo.ir.types import float32, uint8
import allo.ir.types as T

# mxint8's block_size; also what make_mx_dot_general_dataflow_trunc's 512b
# word-splitting assumes (32 float32 = two 512b words), so it's used
# throughout this file rather than a separate small constant.
K = 32

PART_NUMBER.setdefault("u55c", "xcu55c-fsvh2892-2L-e")
_HLS_CONFIGS = {"device": "u55c", "frequency": 300}

# HLS csynth builds write substantial scratch output; keep it inside the repo
# (gitignored) instead of the system /tmp.
_REPO_TMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_tmp")
os.makedirs(_REPO_TMP_DIR, exist_ok=True)


def _mxint8_trunc_quantize_ref(x_f32):
    """Bit-exact reference for the truncating quantizer: shared scale =
    max exponent field + 1, mantissa truncated to 8b (no rounding), sign-
    magnitude output. Mirrors _mx_quantize_elem_trunc_f32."""
    bits = x_f32.view(np.uint32).astype(np.int64)
    sign = (bits >> 31) & 1
    exp_field = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF

    max_exp_field = int(np.max(exp_field))
    shared_field = (max_exp_field + 1) & 0xFF

    mant_bits = 7
    narrow_mant = (1 << mant_bits) | (mant >> (23 - mant_bits))
    shift_amt = shared_field - exp_field
    safe_shift = np.clip(shift_amt, 0, 31)
    magnitude = np.where(shift_amt <= 8, narrow_mant >> safe_shift, 0)
    magnitude = np.where(exp_field == 0, 0, magnitude)

    data = (sign << mant_bits) | (magnitude & ((1 << mant_bits) - 1))
    return shared_field, data.astype(np.uint8)


def make_quantize_block_trunc_kernel(Ty):
    def kernel(x: "float32[K]") -> ("uint8[1]", "uint8[K]"):
        scale_out: uint8[1]
        data_out: uint8[K]
        w: Ty = mx_quantize_block_f32_trunc[Ty, K](x)
        scale_out[0] = w[Ty.bits - 8 : Ty.bits]
        for i in range(K):
            data_out[i] = w[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
        return scale_out, data_out

    kernel.__name__ = f"quantize_block_trunc_{Ty.name}"
    return kernel


def test_mx_quantize_block_trunc_mxint8():
    Ty = T.mxint8
    rng = np.random.default_rng(1)
    mod = allo.customize(make_quantize_block_trunc_kernel(Ty)).build()

    for trial in range(20):
        x = (rng.standard_normal(K) * 2.0 ** rng.integers(-8, 8, K)).astype(np.float32)
        if trial == 0:
            x[:] = 0.0
        elif trial == 1:
            x[0] = 0.0

        scale, data = mod(x)
        scale = int(np.asarray(scale).flatten()[0])
        data = np.asarray(data).flatten()

        ref_scale, ref_data = _mxint8_trunc_quantize_ref(x)
        assert scale == int(ref_scale), f"trial {trial}: scale {scale} != {ref_scale}"
        np.testing.assert_array_equal(data, ref_data, err_msg=f"trial {trial}: input={x}")


######################################################################
# mx_block_dot_trunc (multiply-accumulate over sign-magnitude elements)
######################################################################


def make_block_dot_trunc_kernel(Ty, K):
    def kernel(a: "float32[K]", b: "float32[K]") -> float32:
        word_a: Ty = mx_quantize_block_f32_trunc[Ty, K](a)
        word_b: Ty = mx_quantize_block_f32_trunc[Ty, K](b)
        return mx_block_dot_trunc[Ty, K](word_a, word_b)

    kernel.__name__ = f"block_dot_trunc_{Ty.name}"
    return kernel


def _assert_close_trunc(our_dot, ref_dot, trial):
    # truncating quantizer: no rounding, so tolerance is looser than the
    # round-to-nearest mx_block_dot in test_mxfp.py.
    mag = max(abs(our_dot), abs(ref_dot), 1.0)
    assert our_dot == pytest.approx(ref_dot, rel=0.2, abs=mag * 0.2), (
        f"trial {trial}: our={our_dot} ref={ref_dot}"
    )


def test_mx_block_dot_trunc_mxint8():
    Ty = T.mxint8
    mod = allo.customize(make_block_dot_trunc_kernel(Ty, K)).build()
    rng = np.random.default_rng(4)
    for trial in range(10):
        # one shared magnitude per block (mild per-element jitter) -- a
        # per-element random exponent range is adversarial for 7-bit
        # truncating quantization and blows the error past what's worth
        # bounding here.
        a = (rng.standard_normal(K) * 2.0 ** rng.integers(-2, 2)).astype(np.float32)
        b = (rng.standard_normal(K) * 2.0 ** rng.integers(-2, 2)).astype(np.float32)
        our_dot = float(mod(a, b))
        ref_dot = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        _assert_close_trunc(our_dot, ref_dot, trial)


def test_block_dot_product_trunc_mxint8():
    Ty = T.mxint8

    def kernel(a: "float32[K]", b: "float32[K]") -> float32:
        return block_dot_product_trunc[Ty, K](a, b)

    s = allo.customize(kernel)
    schedule_block_dot_product_trunc(s)
    mod = s.build()

    rng = np.random.default_rng(5)
    for trial in range(10):
        a = (rng.standard_normal(K) * 2.0 ** rng.integers(-2, 2)).astype(np.float32)
        b = (rng.standard_normal(K) * 2.0 ** rng.integers(-2, 2)).astype(np.float32)
        our_dot = float(mod(a, b))
        ref_dot = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        _assert_close_trunc(our_dot, ref_dot, trial)


######################################################################
# make_mx_dot_general_dataflow_trunc (4-stage streaming dataflow)
######################################################################


def test_mx_dot_general_dataflow_trunc_mxint8():
    Ty = T.mxint8
    NB = 128
    P = 4
    s = make_mx_dot_general_dataflow_trunc(Ty, K, NB, P)

    if not hls.is_available("vitis_hls"):
        return

    mode = os.environ.get("ALLO_HLSMODE")
    project = os.environ.get("ALLO_PROJECT")

    if mode != "csyn" and "XDEVICE" not in os.environ:
        print(f"Skipping {mode} run: set XDEVICE to a platform .xpfm to run this mode")
        return

    hls_mod = s.build(
        target="vitis_hls", mode=mode, project=project, wrap_io=False, configs=_HLS_CONFIGS
    )

    # issue #603 (https://github.com/alloy-lang/allo/issues/603)
    patched = patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
    assert patched, "expected mx_quantize_block_f32_trunc/_mx_pack_word to need the patch"

    if mode == "csyn":
        hls_mod()
        csynth_rpt = os.path.join(
            project, "out.prj", "solution1", "syn", "report", "top_csynth.rpt"
        )
        assert os.path.isfile(csynth_rpt)
        with open(csynth_rpt, encoding="utf-8") as f:
            report = f.read()
        assert "dataflow" in report
    else:
        N = K * NB
        HALF = K // 2  # float32 elements per 512b half
        rng = np.random.default_rng(0)
        A = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(np.float32)
        B = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(np.float32)

        def split(x):
            blocks = x.reshape(NB, K)
            to_words = lambda chunk: (
                np.frombuffer(chunk.tobytes(), dtype=np.uint8)
                .reshape(NB, HALF * 4)
                .copy()
                .view(f"V{HALF * 4}")
                .reshape(NB)
            )
            return to_words(blocks[:, :HALF].copy()), to_words(blocks[:, HALF:].copy())

        a_lo, a_hi = split(A)
        b_lo, b_hi = split(B)
        result = np.zeros((1,), dtype=np.float32)
        hls_mod(a_lo, a_hi, b_lo, b_hi, result)
        ref = float(np.dot(A.astype(np.float64), B.astype(np.float64)))
        print(f"[{mode}] result={result[0]} ref={ref}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csyn", "sw_emu", "hw_emu", "hw"], default="csyn")
    parser.add_argument(
        "--project",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp_trunc.prj"),
    )
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        shutil.rmtree(args.project, ignore_errors=True)

    pytest.main([__file__, *pytest_args])
