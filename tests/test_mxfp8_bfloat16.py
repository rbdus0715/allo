# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import numpy as np
import ml_dtypes
import pytest
import allo.backend.hls as hls
from allo.library.mxint8 import patch_extern_c_for_class_return_types
from allo.library.mxfp8_bf16 import make_mxfp8_dot_general_dataflow_bf16
import allo.ir.types as T


@pytest.mark.parametrize("ty_name", ["mxfp8_e4m3", "mxfp8_e5m2"])
def test_mxfp8_dot_general_dataflow_bf16(ty_name):
    Ty = getattr(T, ty_name)
    K = 32  # Ty's actual block_size
    NB = 128
    P = 4
    s = make_mxfp8_dot_general_dataflow_bf16(Ty, K, NB, P)

    if not hls.is_available("vitis_hls"):
        return

    mode = os.environ.get("ALLO_HLSMODE")
    project = os.environ.get("ALLO_PROJECT")
    if project:
        project = f"{project}_{ty_name}"

    # sw_emu/hw_emu/hw go through the v++ Makefile flow, which needs XDEVICE
    # pointing at a platform .xpfm (same env var Allo already reads).
    if mode != "csyn" and "XDEVICE" not in os.environ:
        print(f"Skipping {mode} run: set XDEVICE to a platform .xpfm to run this mode")
        return

    hls_mod = s.build(
        target="vitis_hls",
        mode=mode,
        project=project,
        wrap_io=False,
    )

    # issue #603 (https://github.com/alloy-lang/allo/issues/603)
    patched = patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
    assert patched, "expected _mx_pack_word/mx_quantize_block_fp_bf16 to need the patch"

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
        # sw_emu/hw_emu/hw can't be called with zero args like csyn -- pack
        # real bf16 input blocks into a single UInt(512)[NB] array each (a
        # K=32 bf16 block is exactly 512b, matching mxint8_bf16's layout
        # since e4m3/e5m2 elements are also 8 bits).
        N = K * NB
        rng = np.random.default_rng(0)
        A = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(
            ml_dtypes.bfloat16
        )
        B = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(
            ml_dtypes.bfloat16
        )

        def to_words(x):
            blocks = x.reshape(NB, K)
            return (
                np.frombuffer(blocks.tobytes(), dtype=np.uint8)
                .reshape(NB, K * 2)
                .copy()
                .view(f"V{K * 2}")
                .reshape(NB)
            )

        a0 = to_words(A)
        b0 = to_words(B)
        result = np.zeros((1,), dtype=np.uint16)
        hls_mod(a0, b0, result)
        hw_dot = float(result.view(ml_dtypes.bfloat16)[0])
        ref = float(np.dot(A.astype(np.float64), B.astype(np.float64)))
        abs_diff = abs(hw_dot - ref)
        rel_diff = abs_diff / abs(ref) if ref != 0 else float("nan")
        print(
            f"[{mode}][{ty_name}] hw_result={hw_dot} ref={ref} "
            f"abs_diff={abs_diff} rel_diff={rel_diff:.6%}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csyn", "sw_emu", "hw_emu", "hw"], default="csyn")
    parser.add_argument(
        "--project",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp8_bf16.prj"),
    )
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        shutil.rmtree(f"{args.project}_mxfp8_e4m3", ignore_errors=True)
        shutil.rmtree(f"{args.project}_mxfp8_e5m2", ignore_errors=True)

    pytest.main([__file__, *pytest_args])
