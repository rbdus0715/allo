# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Chaining library kernels whose insides are a dataflow: nn.mx_conv2d is a
# plain Allo function, and its registered schedule (im2col -> MX quantize ->
# block dot, streamed one MX block per token) is applied per call site with
# s.compose(nn.mx_conv2d, instantiate=..., id=...).

import os
import tempfile

import numpy as np
import allo
import allo.backend.hls as hls
import allo.ir.types as T
import allo.library.nn as nn
from allo.library.mxint8 import patch_extern_c_for_class_return_types

Ty = T.mxint8

# conv1: C0 -> C1 channels, conv2: C1 -> C2, both 3x3 / stride 1 / pad 1
C0, C1, C2, H0, K0 = 2, 3, 2, 6, 3
CONV1 = [Ty, 1, C0, C1, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1]
CONV2 = [Ty, 1, C1, C2, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1]


def model(
    inp: "float32[1, C0, H0, H0]",
    k1: "float32[C1, C0, K0, K0]",
    b1: "float32[C1]",
    k2: "float32[C2, C1, K0, K0]",
    b2: "float32[C2]",
) -> "float32[1, C2, H0, H0]":
    a = nn.mx_conv2d[Ty, 1, C0, C1, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1](inp, k1, b1)
    return nn.mx_conv2d[Ty, 1, C1, C2, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1](a, k2, b2)


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


def _compose_model():
    s = allo.customize(model)
    # the n-th call of a library function (and of its helpers) is suffixed
    # "_n" for n > 0, so the first compose takes no id
    s.compose(nn.mx_conv2d, instantiate=CONV1)
    s.compose(nn.mx_conv2d, instantiate=CONV2, id="1")
    s.dataflow("model")
    return s


def test_mx_conv2d_chain():
    rng = np.random.default_rng(1)
    inp = _rand((1, C0, H0, H0), rng)
    k1, b1 = _rand((C1, C0, K0, K0), rng), _rand((C1,), rng)
    k2, b2 = _rand((C2, C1, K0, K0), rng), _rand((C2,), rng)

    # numerics: the chain equals the two layers run one by one
    Y = allo.customize(model).build(target="llvm")(inp, k1, b1, k2, b2)
    conv = lambda inst: allo.customize(nn.mx_conv2d, instantiate=inst).build(
        target="llvm"
    )
    a = conv(CONV1)(inp, k1, b1)
    np.testing.assert_array_equal(Y, conv(CONV2)(a, k2, b2))

    code = str(_compose_model().build(target="vhls"))
    assert code.count("#pragma HLS dataflow") >= 3  # model + one per conv
    assert code.count("#pragma HLS stream") >= 6  # P, XS, XE per conv


def test_mx_conv2d_chain_csyn():
    if os.environ.get("ALLO_HLSMODE", "csyn") != "csyn":
        return
    if not hls.is_available("vitis_hls"):
        return
    project = os.environ.get("ALLO_PROJECT") or tempfile.mkdtemp(suffix=".prj")
    hls_mod = _compose_model().build(target="vitis_hls", mode="csyn", project=project)
    # issue #603: extern "C" functions returning ap_int<N> (the packed MX word)
    patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
    hls_mod()
    rpt = os.path.join(
        project, "out.prj", "solution1", "syn", "report", "model_csynth.rpt"
    )
    assert os.path.isfile(rpt)
    print(f"csynth report: {rpt}")


if __name__ == "__main__":
    test_mx_conv2d_chain()
    test_mx_conv2d_chain_csyn()
