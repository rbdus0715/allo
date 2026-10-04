# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# allo.linear on MX (mxint8) operands, in the style of test_mxfp.py:
#   pytest tests/test_mx_linear.py                       -> LLVM backend only
#   python tests/test_mx_linear.py --mode csyn           -> + Vitis HLS csynth
#   python tests/test_mx_linear.py --mode hw_emu --device u55c   (or u250)

import glob
import os
import shutil
import time

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
import allo.ir.types as T
from allo.ir.types import float32, int8, e8m0
from allo.library.nn import mx_linear2d_ref
from allo.library.mxint8 import mx_quantize, patch_extern_c_for_class_return_types

Ty = T.mxint8
M, N, K = 4, 16, 128
NB = K // Ty.block_size

PLATFORMS = "/backup/opt/xilinx/platforms"
DEVICES = {  # --device -> (platform .xpfm, memory kind)
    "u55c": (
        f"{PLATFORMS}/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm",
        "HBM",
    ),
    "u250": (
        f"{PLATFORMS}/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm",
        "DDR",
    ),
}


def linear_mx(
    Xq: "int8[M, K]",
    Xs: "e8m0[M, NB]",
    Wq: "int8[N, K]",
    Ws: "e8m0[N, NB]",
    bias: "float32[N]",
) -> "float32[M, N]":
    return allo.linear(Xq, Xs, Wq, Ws, bias)


def linear_mx_weight(
    X: "float32[M, K]", Wq: "int8[N, K]", Ws: "e8m0[N, NB]", bias: "float32[N]"
) -> "float32[M, N]":
    return allo.linear(X, Wq, Ws, bias)


def _inputs(seed):
    rng = np.random.default_rng(seed)
    rand = lambda shape: (
        rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)
    ).astype(np.float32)
    return rand((M, K)), rand((N, K)), rand((N,))


def _memory_mapping(arg_names):
    mem = os.environ.get("ALLO_MEMORY", "HBM")
    bulk = [a for a in arg_names if a in ("X", "Xq", "Wq")]
    shared = f"{mem}[{len(bulk)}]"
    mapping = {a: f"{mem}[{i}]" for i, a in enumerate(bulk)}
    mapping.update({a: shared for a in arg_names if a not in bulk})
    mapping["output_0"] = shared
    return mapping


def _run_hls(kernel, args, Z_llvm):
    """Builds/runs `kernel` with Vitis when ALLO_HLSMODE is set; hw must equal LLVM."""
    mode = os.environ.get("ALLO_HLSMODE")
    if mode is None or not hls.is_available("vitis_hls"):
        return
    if mode != "csyn" and "XDEVICE" not in os.environ:
        print(f"Skipping {mode} run: set XDEVICE (or --device) to run this mode")
        return
    name = kernel.__name__
    project = os.path.join(os.environ["ALLO_PROJECT"], name)
    s = allo.customize(kernel)
    arg_names = list(kernel.__code__.co_varnames[: kernel.__code__.co_argcount])
    hls_mod = s.build(
        target="vitis_hls",
        mode=mode,
        project=project,
        configs={"hbm_mapping": _memory_mapping(arg_names)},
    )
    # issue #603 (https://github.com/alloy-lang/allo/issues/603)
    patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")

    if mode == "csyn":
        hls_mod()
        assert os.path.isfile(
            os.path.join(
                project, "out.prj", "solution1", "syn", "report", f"{name}_csynth.rpt"
            )
        )
        return

    Z = np.zeros((M, N), dtype=np.float32)
    start = time.time()
    try:
        hls_mod(*args, Z)
    except RuntimeError:
        outs = [
            f
            for f in glob.glob(f"{project}/output*.data")
            if os.path.getmtime(f) >= start
        ]
        if not outs:
            raise
        Z = np.fromfile(sorted(outs)[-1], dtype=np.float32).reshape(M, N)
    print(f"[{mode}] {name}: hw == LLVM bit-exact: {np.array_equal(Z, Z_llvm)}")
    np.testing.assert_array_equal(Z, Z_llvm)


@pytest.mark.parametrize("kernel", [linear_mx, linear_mx_weight])
def test_allo_linear_mx(kernel):
    X, W, bias = _inputs(0)
    Wq, Ws = mx_quantize(Ty, W)
    if kernel is linear_mx:
        args = (*mx_quantize(Ty, X), Wq, Ws, bias)
    else:
        args = (X, Wq, Ws, bias)
    Z = allo.customize(kernel).build(target="llvm")(*args)

    ref = allo.customize(mx_linear2d_ref, instantiate=[Ty, M, N, K]).build(
        target="llvm"
    )(X, W, bias)
    np.testing.assert_array_equal(Z, ref)
    ref64 = X.astype(np.float64) @ W.astype(np.float64).T + bias
    print(
        f"{kernel.__name__}: rel err vs fp64 {np.linalg.norm(Z - ref64) / np.linalg.norm(ref64):.4f}"
    )

    _run_hls(kernel, args, Z)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["csyn", "sw_emu", "hw_emu", "hw"], default="csyn"
    )
    parser.add_argument("--device", choices=sorted(DEVICES), default="u55c")
    parser.add_argument(
        "--project",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mx_linear.prj"
        ),
    )
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    xpfm, memory = DEVICES[args.device]
    os.environ.setdefault("XDEVICE", xpfm)
    os.environ["ALLO_MEMORY"] = memory
    if args.clean:
        shutil.rmtree(args.project, ignore_errors=True)

    pytest.main([__file__, "-s", *pytest_args])
