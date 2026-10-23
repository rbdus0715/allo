# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object, no-value-for-parameter

# nn.mx_linear2d (mxint8) vs the same dataflow in plain float32: X rows stream
# one 32-element block per token and the dot stage does one block dot per
# cycle in both; float32 just has no quantize stages.

import os
import shutil

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
import allo.ir.types as T
import allo.library.nn as nn
from allo.ir.types import int32, float32, UInt, ConstExpr
from allo.ir.utils import MockBuffer
from allo.library.nn import _mx_lin_xpack, partition_block_lanes
from allo.library.mxint8 import (
    mx_pack_f32_blocks,
    patch_extern_c_for_class_return_types,
)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BS = 32  # block size, same as mxint8

# (M, N, K): Z[M, N] = X[M, K] @ W[N, K]^T + b
CONFIGS = {
    "small": (16, 64, 128),  # 4 blocks per row
    "medium": (32, 256, 256),  # 8 blocks per row
}


def _fp_lin_wpack[N, K, NB, BS](W: "float32[N, K]", WF: "UInt(32 * BS)[N, NB]"):
    # same as nn._mx_lin_wquant: one weight block per iteration (zero past K)
    for wn, b in allo.grid(N, NB):
        word: UInt(32 * BS) = 0
        with allo.meta_for(BS) as e:
            fw: float32 = 0.0
            if b * BS + e < K:
                fw = W[wn, b * BS + e]
            word[e * 32 : (e + 1) * 32] = fw.bitcast()
        WF[wn, b] = word


def _fp_lin_dot[
    M, N, NB, BS
](
    P: "UInt(32 * BS)[M, NB]",
    WF: "UInt(32 * BS)[N, NB]",
    bias: "float32[N]",
    Z: "float32[M, N]",
):
    # same loop order as nn._mx_lin_dot: one float32 block dot per iteration
    acc: float32[N] = 0.0
    xw: UInt(32 * BS) = 0
    for m in range(M):
        for b, n in allo.grid(NB, N):
            if n == 0:
                xw = P[m, b]
            ww: UInt(32 * BS) = WF[n, b]
            d: float32 = 0.0
            with allo.meta_for(BS) as e:
                xb: int32 = xw[e * 32 : (e + 1) * 32]
                wb: int32 = ww[e * 32 : (e + 1) * 32]
                xf: float32 = xb.bitcast()
                wf: float32 = wb.bitcast()
                d = d + xf * wf
            prev: float32 = 0.0
            if b > 0:
                prev = acc[n]
            total: float32 = prev + d
            acc[n] = total
            if b == NB - 1:
                Z[m, n] = total + bias[n]


def fp32_linear2d[
    M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    PW: ConstExpr[int32] = 32 * BS
    WF: UInt(PW)[N, NB]
    _fp_lin_wpack[N, K, NB, BS](W, WF)
    P: UInt(PW)[M, NB]
    _mx_lin_xpack[M, K, BS, NB](X, P)
    Z: float32[M, N]
    _fp_lin_dot[M, N, NB, BS](P, WF, bias, Z)
    return Z


def schedule_fp32_linear2d(s, depth=4):
    """Same schedule as nn.schedule_mx_linear2d, minus the quantize stage."""
    _, _, K = s.inst_list[:3]
    NB = (K + BS - 1) // BS
    top = s.top_func_name
    for arr in ("X", "W"):
        partition_block_lanes(s, MockBuffer(top, arr), K, BS)
    s.pipeline("_fp_lin_wpack:b")
    s.pipeline("_mx_lin_xpack:b")
    s.pipeline("_fp_lin_dot:n")
    s.to(MockBuffer(top, "P"), "_fp_lin_dot", depth=max(depth, NB))
    s.dataflow(top)
    return s


# DRAM boundary: X and W arrive as 512b words over two ports (one 32-float
# block per cycle), unpacked on chip by nn.mx_unpack_f32
Ty = T.mxint8


def mx_linear_top[
    M, N, K
](
    X0: "UInt(512)[M, K // 32]",
    X1: "UInt(512)[M, K // 32]",
    W0: "UInt(512)[N, K // 32]",
    W1: "UInt(512)[N, K // 32]",
    bias: "float32[N]",
) -> "float32[M, N]":
    X: float32[M, K]
    W: float32[N, K]
    nn.mx_unpack_f32[M, K](X0, X1, X)
    nn.mx_unpack_f32[N, K](W0, W1, W)
    return nn.mx_linear2d[Ty, M, N, K](X, W, bias)


def fp32_linear_top[
    M, N, K
](
    X0: "UInt(512)[M, K // 32]",
    X1: "UInt(512)[M, K // 32]",
    W0: "UInt(512)[N, K // 32]",
    W1: "UInt(512)[N, K // 32]",
    bias: "float32[N]",
) -> "float32[M, N]":
    X: float32[M, K]
    W: float32[N, K]
    nn.mx_unpack_f32[M, K](X0, X1, X)
    nn.mx_unpack_f32[N, K](W0, W1, W)
    return fp32_linear2d[M, N, K](X, W, bias)


def _top_schedule(top, layer, M, N, K):
    s = allo.customize(top, instantiate=[M, N, K])
    s.compose(nn.mx_unpack_f32, instantiate=[M, K])
    s.compose(nn.mx_unpack_f32, instantiate=[N, K], id="1")
    s.compose(layer)  # its X/W block-lane partitions reach the unpacks too
    s.dataflow(top.__name__)
    return s


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_linear_numerics(name):
    M, N, K = CONFIGS[name]
    rng = np.random.default_rng(0)
    X, W, bias = _rand((M, K), rng), _rand((N, K), rng), _rand((N,), rng)
    exact = X.astype(np.float64) @ W.astype(np.float64).T + bias
    rel = lambda Z: np.linalg.norm(Z - exact) / np.linalg.norm(exact)
    args = (*mx_pack_f32_blocks(X), *mx_pack_f32_blocks(W), bias)
    Z_fp = allo.customize(fp32_linear_top, instantiate=[M, N, K]).build(target="llvm")(
        *args
    )
    Z_mx = allo.customize(mx_linear_top, instantiate=[M, N, K]).build(target="llvm")(
        *args
    )
    # the packed interface changes nothing but how X and W reach the chip
    ref_mx = allo.customize(nn.mx_linear2d, instantiate=[Ty, M, N, K]).build(
        target="llvm"
    )(X, W, bias)
    np.testing.assert_array_equal(Z_mx, ref_mx)
    print(
        f"\n[{name}] rel norm err vs fp64: fp32 {rel(Z_fp):.2e}, mxint8 {rel(Z_mx):.2e}"
    )
    assert rel(Z_fp) < 1e-5
    assert rel(Z_mx) < 0.1


def _schedules(name):
    M, N, K = CONFIGS[name]
    s_mx = nn.schedule_mx_linear2d(
        allo.customize(nn.mx_linear2d, instantiate=[Ty, M, N, K])
    )
    s_fp = schedule_fp32_linear2d(allo.customize(fp32_linear2d, instantiate=[M, N, K]))
    return (
        _top_schedule(mx_linear_top, s_mx, M, N, K),
        _top_schedule(fp32_linear_top, s_fp, M, N, K),
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_linear_csyn(name):
    s_mx, s_fp = _schedules(name)
    for s in (s_mx, s_fp):
        assert "#pragma HLS dataflow" in str(s.build(target="vhls"))
    if os.environ.get("ALLO_HLSMODE", "csyn") != "csyn":
        return
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")
    root = os.environ.get("ALLO_PROJECT", os.path.join(_TESTS_DIR, "comp_fp_mx_linear"))
    for tag, s, top in (
        ("mxint8", s_mx, "mx_linear_top"),
        ("fp32", s_fp, "fp32_linear_top"),
    ):
        project = f"{root}_{name}_{tag}.prj"
        mod = s.build(target="vitis_hls", mode="csyn", project=project)
        # issue #603: extern "C" functions returning ap_int<N>
        patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
        mod()
        rpt = os.path.join(
            project, "out.prj", "solution1", "syn", "report", f"{top}_csynth.rpt"
        )
        assert os.path.isfile(rpt)
        print(f"[{name}] {tag}: {rpt}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=os.path.join(_TESTS_DIR, "comp_fp_mx_linear"),
        help="Project root; projects are <root>_<config>_{mxint8,fp32}.prj",
    )
    parser.add_argument("--clean", action="store_true", help="Clean the project dirs")
    args, pytest_args = parser.parse_known_args()
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        for name in CONFIGS:
            for tag in ("mxint8", "fp32"):
                shutil.rmtree(f"{args.project}_{name}_{tag}.prj", ignore_errors=True)
    pytest.main([__file__, "-s", *pytest_args])
