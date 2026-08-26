import os
import shutil

import numpy as np
import pytest

import allo.dataflow as df
import allo.backend.hls as hls
from allo.backend.config import PART_NUMBER
from allo.backend.report import parse_xml
from allo.ir.types import float32, int32, UInt, Stream
import allo.ir.types as T
from allo.library.mxfp import (
    make_mx_dot_general_dataflow,
    patch_extern_c_for_class_return_types,
)

PART_NUMBER.setdefault("u55c", "xcu55c-fsvh2892-2L-e")
_HLS_CONFIGS = {"device": "u55c", "frequency": 300}

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

K, NB, P = 32, 128, 4
N = K * NB


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


def _split_f32_words(x, K, NB):
    """Split a float32[K * NB] array into two UInt(512)[NB] halves (lo/hi),
    matching the channel-split A0/A1 (or B0/B1) layout make_mx_dot_general_dataflow
    / make_fp32_dot_general_dataflow expect for their inputs."""
    half = 512 // 32
    word_bytes = half * 4
    blocks = x.reshape(NB, K)
    to_words = lambda chunk: (
        np.frombuffer(chunk.tobytes(), dtype=np.uint8)
        .reshape(NB, word_bytes)
        .copy()
        .view(f"V{word_bytes}")
        .reshape(NB)
    )
    return to_words(blocks[:, :half].copy()), to_words(blocks[:, half:].copy())


def test_compare_dataflow_mxint8_vs_fp32():
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")

    mode = os.environ.get("ALLO_HLSMODE", "csyn")
    project_root = os.environ.get("ALLO_PROJECT", os.path.join(_TESTS_DIR, "comp_fp_mx"))
    project_mx = f"{project_root}_mxint8.prj"
    project_fp = f"{project_root}_fp32.prj"

    if mode != "csyn" and "XDEVICE" not in os.environ:
        pytest.skip(f"set XDEVICE to a platform .xpfm to run mode={mode}")

    s_mx = make_mx_dot_general_dataflow(T.mxint8, K, NB, P)
    s_fp = make_fp32_dot_general_dataflow(K, NB, P)

    mod_mx = s_mx.build(
        target="vitis_hls", mode=mode, project=project_mx, wrap_io=False, configs=_HLS_CONFIGS
    )
    # issue #603 (https://github.com/alloy-lang/allo/issues/603)
    patched = patch_extern_c_for_class_return_types(f"{project_mx}/kernel.cpp")
    assert patched, "expected _mx_pack_word/mx_quantize_block_f32 to need the patch"

    mod_fp = s_fp.build(
        target="vitis_hls", mode=mode, project=project_fp, wrap_io=False, configs=_HLS_CONFIGS
    )

    if mode == "csyn":
        mod_mx()
        mod_fp()

        for project in (project_mx, project_fp):
            csynth_rpt = os.path.join(
                project, "out.prj", "solution1", "syn", "report", "top_csynth.rpt"
            )
            assert os.path.isfile(csynth_rpt)
            with open(csynth_rpt, encoding="utf-8") as f:
                assert "dataflow" in f.read()

        print("\n=== mxint8 dataflow dot product (mx_dot_general) ===")
        parse_xml(project_mx, "Vitis HLS", top="top", print_flag=True)
        print("\n=== float32 dataflow dot product (plain fp) ===")
        parse_xml(project_fp, "Vitis HLS", top="top", print_flag=True)
    else:
        rng = np.random.default_rng(0)
        A = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(np.float32)
        B = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(np.float32)
        a_lo, a_hi = _split_f32_words(A, K, NB)
        b_lo, b_hi = _split_f32_words(B, K, NB)

        result_mx = np.zeros((1,), dtype=np.float32)
        mod_mx(a_lo, a_hi, b_lo, b_hi, result_mx)

        result_fp = np.zeros((1,), dtype=np.float32)
        mod_fp(a_lo, a_hi, b_lo, b_hi, result_fp)

        ref = float(np.dot(A.astype(np.float64), B.astype(np.float64)))
        mx_err = abs(result_mx[0] - ref) / abs(ref)
        fp_err = abs(result_fp[0] - ref) / abs(ref)
        print(f"\n[{mode}] reference (fp64)  = {ref}")
        print(f"[{mode}] mxint8 dataflow   = {result_mx[0]}  (rel err {mx_err:.3e})")
        print(f"[{mode}] float32 dataflow  = {result_fp[0]}  (rel err {fp_err:.3e})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csyn", "sw_emu", "hw_emu", "hw"], default="csyn")
    parser.add_argument(
        "--project",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "comp_fp_mx"),
        help="Project root; actual projects are <root>_mxint8.prj and <root>_fp32.prj",
    )
    parser.add_argument("--clean", action="store_true", help="Clean both project dirs")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        shutil.rmtree(f"{args.project}_mxint8.prj", ignore_errors=True)
        shutil.rmtree(f"{args.project}_fp32.prj", ignore_errors=True)

    pytest.main([__file__, *pytest_args])
