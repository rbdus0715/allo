# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import numpy as np
import pytest
import allo.backend.hls as hls
from allo.library.mxfp import (
    make_mx_dot_general_dataflow,
    patch_extern_c_for_class_return_types,
)
import allo.ir.types as T


def test_mx_dot_general_dataflow_mxint8():
    Ty = T.mxint8
    K = 32  # mxint8's actual block_size
    NB = 128
    P = 4
    s = make_mx_dot_general_dataflow(Ty, K, NB, P)

    if not hls.is_available("vitis_hls"):
        return

    mode = os.environ.get("ALLO_HLSMODE")
    project = os.environ.get("ALLO_PROJECT")

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
    assert patched, "expected _mx_pack_word/mx_quantize_block_f32 to need the patch"

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
        # real float32 input blocks into two UInt(512)[NB] halves each
        # (512b is the platform's m_axi limit; each half is a separate
        # top-level array so it gets its own m_axi bundle/port -- see
        # make_mx_dot_general_dataflow).
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
        hw_dot = float(result[0])
        ref = float(np.dot(A.astype(np.float64), B.astype(np.float64)))
        abs_diff = abs(hw_dot - ref)
        rel_diff = abs_diff / abs(ref) if ref != 0 else float("nan")
        print(
            f"[{mode}] hw_result={hw_dot} ref={ref} "
            f"abs_diff={abs_diff} rel_diff={rel_diff:.6%}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csyn", "sw_emu", "hw_emu", "hw"], default="csyn")
    parser.add_argument("--project", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp.prj"))
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        shutil.rmtree(args.project, ignore_errors=True)

    pytest.main([__file__, *pytest_args])
