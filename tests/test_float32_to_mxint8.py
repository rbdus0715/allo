# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone float32[32] -> mxint8 block quantizer.

Written from scratch (does not reuse allo/library/mxfp.py): a single block of
32 float32 values comes in, the shared exponent is picked from the block's
max exponent field with a fully-unrolled comparison, and each element is
rounded to a signed 8-bit integer with round-nearest-even (ties to even),
pipelined at II=1.
"""

import glob
import os
import shutil

import numpy as np
import pytest

import allo
import allo.backend.hls as hls
from allo.backend.report import parse_xml
from allo.backend.config import PART_NUMBER
from allo.ir.types import float32, int32, uint8

PART_NUMBER.setdefault("u55c", "xcu55c-fsvh2892-2L-e")
_HLS_CONFIGS = {"device": "u55c", "frequency": 300}

K = 32  # OCP MX block size
ELEM_BITS = 8  # mxint8 element width
MAX_UNBIASED_EXP = ELEM_BITS - 2  # 6 -- largest |element| a block can hold is 2**6

_REPO_TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_REPO_TMP_DIR, exist_ok=True)


def float32_to_mxint8(x: "float32[K]", scale_out: "uint8[1]", data_out: "uint8[K]"):
    # 1) shared exponent: max exponent field over the block, fully unrolled
    max_exp_field: int32 = 0
    for i0 in range(K):
        bits_i: int32 = x[i0].bitcast()
        exp_field_i: int32 = bits_i[23:31]
        if exp_field_i > max_exp_field:
            max_exp_field = exp_field_i

    shared_exp: int32 = max_exp_field - 127 - MAX_UNBIASED_EXP
    shared_exp = min(shared_exp, 127)
    shared_exp = max(shared_exp, -127)
    scale_out[0] = shared_exp + 127

    # 2) per-element quantize to int8, RNE rounding, pipelined II=1
    for i1 in range(K):
        bits: int32 = x[i1].bitcast()
        sign: int32 = bits[31:32]
        exp_field: int32 = bits[23:31]
        mant: int32 = bits[0:23]

        rounded: int32 = 0
        if exp_field != 0:
            unbiased_exp: int32 = exp_field - 127
            target_exp: int32 = unbiased_exp - shared_exp
            full_mant: int32 = (1 << 23) | mant
            shift: int32 = 23 - target_exp

            mag: int32 = 0
            if shift <= 0:
                mag = 128
            elif shift > 24:
                mag = 0
            else:
                kept: int32 = full_mant >> shift
                remainder: int32 = full_mant & ((1 << shift) - 1)
                half: int32 = 1 << (shift - 1)
                round_up: int32 = 0
                if remainder > half:
                    round_up = 1
                elif remainder == half:
                    round_up = kept & 1  # tie -> round to even
                mag = kept + round_up

            if sign == 1:
                rounded = -mag
            else:
                rounded = mag
            rounded = min(rounded, 127)
            rounded = max(rounded, -128)

        data_out[i1] = rounded


def schedule_float32_to_mxint8(s):
    s.unroll("float32_to_mxint8:i0")
    s.pipeline("float32_to_mxint8:i1", initiation_interval=1)
    return s


def schedule_float32_to_mxint8_unrolled(s):
    # block-at-a-time variant: quantize loop fully unrolled instead of
    # pipelined, so all 32 elements are produced combinationally in ~1 cycle.
    s.unroll("float32_to_mxint8:i0")
    s.unroll("float32_to_mxint8:i1")
    return s


######################################################################
# "Paper-faithful" variant, reconstructed from QLlama (Wen et al., IEEE
# Embedded Systems Letters 2025) Section III-B / Fig. 4:
#   - shared scale = max exponent field + 1 (not max - headroom like above)
#   - mantissa is truncated down to 8 significant bits (implicit 1 + top 7
#     explicit mantissa bits) BEFORE the alignment shift, so the shifter
#     only ever operates on an 8-bit value instead of the full 24-bit
#     fp32 mantissa
#   - plain truncating right-shift, no RNE rounding ("low bit truncation")
#   - sign-magnitude output (sign bit + 7-bit magnitude), so there's no
#     two's-complement negation logic for negative elements either
# shift_amt = shared_field - elem_field is always >= 1 here (shared_field
# is defined as strictly greater than every element's field in the block),
# so the magnitude can never exceed 127 and no saturate/wrap branch is
# reachable -- unlike our RNE version, which can round the max element up
# past 127.
######################################################################


def float32_to_mxint8_paper(x: "float32[K]", scale_out: "uint8[1]", data_out: "uint8[K]"):
    max_exp_field: int32 = 0
    for i0 in range(K):
        bits_i: int32 = x[i0].bitcast()
        exp_field_i: int32 = bits_i[23:31]
        if exp_field_i > max_exp_field:
            max_exp_field = exp_field_i

    shared_field: int32 = max_exp_field + 1
    scale_out[0] = shared_field

    for i1 in range(K):
        bits: int32 = x[i1].bitcast()
        sign: int32 = bits[31:32]
        exp_field: int32 = bits[23:31]
        mant: int32 = bits[0:23]

        narrow_mant: int32 = 128 | (mant >> 16)  # 8b value in [128, 255]
        shift_amt: int32 = shared_field - exp_field

        magnitude: int32 = 0
        if exp_field != 0 and shift_amt <= 8:
            magnitude = narrow_mant >> shift_amt  # truncating shift, no rounding

        data_out[i1] = (sign << 7) | (magnitude & 127)


def schedule_float32_to_mxint8_paper(s):
    # Fig. 4's Shift Quant stage is described as "purely combinational".
    s.unroll("float32_to_mxint8_paper:i0")
    s.unroll("float32_to_mxint8_paper:i1")
    return s


def _mxint8_paper_ref(x_f32):
    bits = x_f32.view(np.uint32).astype(np.int64)
    sign = (bits >> 31) & 1
    exp_field = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF

    max_exp_field = int(np.max(exp_field))
    shared_field = (max_exp_field + 1) & 0xFF

    narrow_mant = 128 | (mant >> 16)
    shift_amt = shared_field - exp_field
    safe_shift = np.clip(shift_amt, 0, 31)
    magnitude = np.where(shift_amt <= 8, narrow_mant >> safe_shift, 0)
    magnitude = np.where(exp_field == 0, 0, magnitude)

    data = (sign << 7) | (magnitude & 0x7F)
    return shared_field, data.astype(np.uint8)


def test_float32_to_mxint8_paper_functional():
    mod = allo.customize(float32_to_mxint8_paper).build()
    rng = np.random.default_rng(3)

    for trial in range(50):
        x = (rng.standard_normal(K) * 2.0 ** rng.integers(-20, 20, K)).astype(np.float32)
        if trial == 0:
            x[:] = 0.0
        elif trial == 1:
            x[0] = 0.0

        scale_out = np.zeros(1, dtype=np.uint8)
        data_out = np.zeros(K, dtype=np.uint8)
        mod(x, scale_out, data_out)

        ref_scale, ref_data = _mxint8_paper_ref(x)
        assert int(scale_out[0]) == int(ref_scale), f"trial {trial}"
        np.testing.assert_array_equal(data_out, ref_data, err_msg=f"trial {trial}: input={x}")


def test_float32_to_mxint8_paper_csynth():
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")

    mode = os.environ.get("ALLO_HLSMODE", "csyn")
    if mode != "csyn":
        pytest.skip("this comparison test only runs in csyn mode")

    project = os.environ.get(
        "ALLO_PROJECT_PAPER", os.path.join(_REPO_TMP_DIR, "float32_to_mxint8_paper.prj")
    )

    s = allo.customize(float32_to_mxint8_paper)
    schedule_float32_to_mxint8_paper(s)
    mod = s.build(
        target="vitis_hls", mode=mode, project=project, wrap_io=False, configs=_HLS_CONFIGS
    )
    mod()

    report_dir = os.path.join(project, "out.prj", "solution1", "syn", "report")
    csynth_rpt = os.path.join(report_dir, "float32_to_mxint8_paper_csynth.rpt")
    assert os.path.isfile(csynth_rpt)
    parse_xml(project, "Vitis HLS", top="float32_to_mxint8_paper", print_flag=True)


def _mxint8_rne_ref(x_f32):
    """Bit-exact reference: same max-exponent-field scale rule, numpy's
    round() is round-half-to-even so it matches the kernel's RNE tie rule."""
    bits = x_f32.view(np.uint32)
    exp_field = (bits >> 23) & 0xFF
    max_exp_field = int(np.max(exp_field))
    shared_exp = max_exp_field - 127 - MAX_UNBIASED_EXP
    shared_exp = min(max(shared_exp, -127), 127)
    scale = 2.0**shared_exp

    q = x_f32.astype(np.float64) / scale
    rounded = np.round(q)
    rounded = np.clip(rounded, -128, 127).astype(np.int64)
    rounded = np.where(exp_field == 0, 0, rounded)
    return (shared_exp + 127) & 0xFF, rounded


def test_float32_to_mxint8_functional():
    mod = allo.customize(float32_to_mxint8).build()
    rng = np.random.default_rng(0)

    for trial in range(50):
        x = (rng.standard_normal(K) * 2.0 ** rng.integers(-20, 20, K)).astype(np.float32)
        if trial == 0:
            x[:] = 0.0
        elif trial == 1:
            x[0] = 0.0
        elif trial == 2:
            x[:] = 0.0
            x[0] = 1.0  # single nonzero element, scale must still be well-defined

        scale_out = np.zeros(1, dtype=np.uint8)
        data_out = np.zeros(K, dtype=np.uint8)
        mod(x, scale_out, data_out)
        scale = int(scale_out[0])
        data = data_out.astype(np.int64)
        data_signed = np.where(data >= 128, data - 256, data)

        ref_scale, ref_signed = _mxint8_rne_ref(x)
        assert scale == ref_scale, f"trial {trial}: scale {scale} != {ref_scale}"
        np.testing.assert_array_equal(
            data_signed, ref_signed, err_msg=f"trial {trial}: input={x}"
        )


def test_float32_to_mxint8_rne_ties():
    # scale = 1.0 exactly (max element 2**6 -> shared_exp = 0), so quantized
    # elements are just round-half-to-even of the raw float value.
    x = np.zeros(K, dtype=np.float32)
    x[0] = 64.0  # 2**MAX_UNBIASED_EXP, pins shared_exp to 0
    ties = [2.5, 3.5, -2.5, -3.5, 0.5, -0.5, 1.5, -1.5]
    for i, v in enumerate(ties):
        x[1 + i] = v
    expected = [2, 4, -2, -4, 0, 0, 2, -2]  # round-half-to-even of each tie

    mod = allo.customize(float32_to_mxint8).build()
    scale_out = np.zeros(1, dtype=np.uint8)
    data_out = np.zeros(K, dtype=np.uint8)
    mod(x, scale_out, data_out)
    assert int(scale_out[0]) == 127  # shared_exp == 0
    data = data_out.astype(np.int64)
    data_signed = np.where(data >= 128, data - 256, data)
    assert int(data_signed[0]) == 64
    assert list(data_signed[1 : 1 + len(expected)]) == expected


def test_float32_to_mxint8_csynth():
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")

    mode = os.environ.get("ALLO_HLSMODE", "csyn")
    project = os.environ.get(
        "ALLO_PROJECT", os.path.join(_REPO_TMP_DIR, "float32_to_mxint8.prj")
    )
    if mode != "csyn" and "XDEVICE" not in os.environ:
        pytest.skip(f"set XDEVICE to a platform .xpfm to run mode={mode}")

    s = allo.customize(float32_to_mxint8)
    schedule_float32_to_mxint8(s)
    mod = s.build(
        target="vitis_hls", mode=mode, project=project, wrap_io=False, configs=_HLS_CONFIGS
    )

    if mode == "csyn":
        mod()
        report_dir = os.path.join(project, "out.prj", "solution1", "syn", "report")
        csynth_rpt = os.path.join(report_dir, "float32_to_mxint8_csynth.rpt")
        assert os.path.isfile(csynth_rpt)

        # the K=32 quantize loop must be pipelined at II=1 (it's synthesized
        # into its own sub-module report, not the top-level one)
        pipelined_at_ii1 = False
        for xml_path in glob.glob(os.path.join(report_dir, "*_csynth.xml")):
            with open(xml_path, encoding="utf-8") as f:
                content = f.read()
            if "<PipelineII>1</PipelineII>" in content and "<TripCount>32</TripCount>" in content:
                pipelined_at_ii1 = True
                break
        assert pipelined_at_ii1, "expected the K=32 quantize loop to be pipelined at II=1"

        parse_xml(project, "Vitis HLS", top="float32_to_mxint8", print_flag=True)
    else:
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(K) * 2.0 ** rng.integers(-8, 8, K)).astype(np.float32)
        scale_out = np.zeros(1, dtype=np.uint8)
        data_out = np.zeros(K, dtype=np.uint8)
        mod(x, scale_out, data_out)
        ref_scale, ref_signed = _mxint8_rne_ref(x)
        data_signed = np.where(data_out.astype(np.int64) >= 128, data_out.astype(np.int64) - 256, data_out.astype(np.int64))
        print(f"[{mode}] scale={scale_out[0]} ref_scale={ref_scale}")
        np.testing.assert_array_equal(data_signed, ref_signed)
        assert int(scale_out[0]) == ref_scale


def test_float32_to_mxint8_csynth_unrolled():
    """Block-at-a-time variant (both loops unrolled) for LUT comparison
    against the pipelined II=1 quantize loop above."""
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")

    mode = os.environ.get("ALLO_HLSMODE", "csyn")
    if mode != "csyn":
        pytest.skip("this comparison test only runs in csyn mode")

    project = os.environ.get(
        "ALLO_PROJECT_UNROLLED", os.path.join(_REPO_TMP_DIR, "float32_to_mxint8_unrolled.prj")
    )

    s = allo.customize(float32_to_mxint8)
    schedule_float32_to_mxint8_unrolled(s)
    mod = s.build(
        target="vitis_hls", mode=mode, project=project, wrap_io=False, configs=_HLS_CONFIGS
    )
    mod()

    report_dir = os.path.join(project, "out.prj", "solution1", "syn", "report")
    csynth_rpt = os.path.join(report_dir, "float32_to_mxint8_csynth.rpt")
    assert os.path.isfile(csynth_rpt)
    parse_xml(project, "Vitis HLS", top="float32_to_mxint8", print_flag=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csyn", "sw_emu", "hw_emu", "hw"], default="csyn")
    parser.add_argument(
        "--project",
        default=os.path.join(_REPO_TMP_DIR, "float32_to_mxint8.prj"),
    )
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        shutil.rmtree(args.project, ignore_errors=True)

    pytest.main([__file__, *pytest_args])
