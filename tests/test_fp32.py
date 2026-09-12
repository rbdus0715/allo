# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import numpy as np
import pytest
import allo.dataflow as df
import allo.backend.hls as hls
from allo.ir.types import float32, int32, UInt, Stream


def make_fp32_dot_general_dataflow(K, NB, P, depth=4):
    # Same 512b channel-split read as make_mx_dot_general_dataflow: a block
    # is K*32b = 1024b for K=32, wider than Alveo shells auto-widen m_axi to
    # (Vitis HLS's default config_interface -m_axi_max_widen_bitwidth=512),
    # so each block is read as two 512b halves, each its own top-level array
    # (and thus its own m_axi bundle/port) so both halves read concurrently.
    WORD_BITS = 512
    HALF = WORD_BITS // 32  # float32 elements per 512b half

    @df.region()
    def top(
        A0: "UInt(WORD_BITS)[NB]",
        A1: "UInt(WORD_BITS)[NB]",
        B0: "UInt(WORD_BITS)[NB]",
        B1: "UInt(WORD_BITS)[NB]",
        out: "float32[1]",
    ):
        pipe_a: Stream[float32[K], depth]
        pipe_b: Stream[float32[K], depth]

        @df.kernel(mapping=[1], args=[A0, A1])
        def read_a(local_A0: "UInt(WORD_BITS)[NB]", local_A1: "UInt(WORD_BITS)[NB]"):
            for b in range(NB):
                blk: float32[K]
                word_lo: UInt(WORD_BITS) = local_A0[b]
                for j0 in range(HALF):
                    bits: int32 = word_lo[j0 * 32 : (j0 + 1) * 32]
                    blk[j0] = bits.bitcast()
                word_hi: UInt(WORD_BITS) = local_A1[b]
                for j1 in range(HALF):
                    bits: int32 = word_hi[j1 * 32 : (j1 + 1) * 32]
                    blk[HALF + j1] = bits.bitcast()
                pipe_a.put(blk)

        @df.kernel(mapping=[1], args=[B0, B1])
        def read_b(local_B0: "UInt(WORD_BITS)[NB]", local_B1: "UInt(WORD_BITS)[NB]"):
            for b in range(NB):
                blk: float32[K]
                word_lo: UInt(WORD_BITS) = local_B0[b]
                for j0 in range(HALF):
                    bits: int32 = word_lo[j0 * 32 : (j0 + 1) * 32]
                    blk[j0] = bits.bitcast()
                word_hi: UInt(WORD_BITS) = local_B1[b]
                for j1 in range(HALF):
                    bits: int32 = word_hi[j1 * 32 : (j1 + 1) * 32]
                    blk[HALF + j1] = bits.bitcast()
                pipe_b.put(blk)

        @df.kernel(mapping=[1], args=[out])
        def dot_product_stage(local_out: "float32[1]"):
            partials: float32[P]
            for p0 in range(P):
                partials[p0] = 0.0

            for i in range(NB // P):
                for j in range(P):
                    blk_a: float32[K] = pipe_a.get()
                    blk_b: float32[K] = pipe_b.get()
                    acc: float32 = 0.0
                    for k in range(K):
                        acc = acc + blk_a[k] * blk_b[k]
                    partials[j] = partials[j] + acc

            total: float32 = 0.0
            for r in range(P):
                total = total + partials[r]
            local_out[0] = total

    s = df.customize(top, opt_default=False)
    s.pipeline("read_a_0:j0")
    s.pipeline("read_b_0:j0")
    s.pipeline("read_a_0:j1")
    s.pipeline("read_b_0:j1")
    s.pipeline("read_a_0:b")
    s.pipeline("read_b_0:b")
    s.pipeline("dot_product_stage_0:i", initiation_interval=P)
    s.unroll("dot_product_stage_0:j")
    s.unroll("dot_product_stage_0:k")
    s.unroll("dot_product_stage_0:p0")
    s.unroll("dot_product_stage_0:r")
    s.partition("dot_product_stage_0:partials", dim=0)
    return s


def test_dot_general_dataflow_fp32():
    K = 32
    NB = 128
    P = 4
    s = make_fp32_dot_general_dataflow(K, NB, P)

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
        # make_fp32_dot_general_dataflow).
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
    parser.add_argument("--project", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp32.prj"))
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        shutil.rmtree(args.project, ignore_errors=True)

    pytest.main([__file__, *pytest_args])
