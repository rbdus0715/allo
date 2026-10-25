# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Chaining library kernels: the activations are MX-quantized on chip
# (nn.mx_quantize_channels), the weights on the host (mx_pack_np). Each call's
# registered schedule is applied with s.compose(fn, instantiate=..., id=...).

import os
import tempfile

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
import allo.ir.types as T
import allo.library.nn as nn
from allo.library.mxint8 import mx_pack_np, patch_extern_c_for_class_return_types

# MX formats as MXInt(elem_bits, block_size) / MXFP(exp_bits, mantissa_bits, block_size)
FORMATS = [T.MXInt(8, 32), T.MXInt(4, 16), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32)]

# conv1: C0 -> C1 channels, conv2: C1 -> C2, both 3x3 / stride 1 / pad 1
C0, C1, C2, H0, K0 = 2, 3, 2, 6, 3
CONV1 = [1, C0, C1, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1]
CONV2 = [1, C1, C2, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1]


def model[
    Ty
](
    inp: "float32[1, C0, H0, H0]",
    k1: "Ty[C1, (C0 + Ty.block_size - 1) // Ty.block_size, K0, K0]",
    b1: "float32[C1]",
    k2: "Ty[C2, (C1 + Ty.block_size - 1) // Ty.block_size, K0, K0]",
    b2: "float32[C2]",
) -> "float32[1, C2, H0, H0]":
    x = nn.mx_quantize_channels[Ty, 1, C0, H0, H0](inp)
    a = nn.mx_conv2d[Ty, 1, C0, C1, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1](x, k1, b1)
    aq = nn.mx_quantize_channels[Ty, 1, C1, H0, H0](a)
    return nn.mx_conv2d[Ty, 1, C1, C2, H0, H0, K0, K0, H0, H0, 1, 1, 1, 1](aq, k2, b2)


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


def _compose_model(Ty):
    s = allo.customize(model, instantiate=[Ty])
    # the n-th call of a library function (and of its helpers) is suffixed
    # "_n" for n > 0, so the first compose takes no id
    s.compose(nn.mx_quantize_channels, instantiate=[Ty, 1, C0, H0, H0])
    s.compose(nn.mx_quantize_channels, instantiate=[Ty, 1, C1, H0, H0], id="1")
    s.compose(nn.mx_conv2d, instantiate=[Ty, *CONV1])
    s.compose(nn.mx_conv2d, instantiate=[Ty, *CONV2], id="1")
    s.dataflow("model")
    return s


@pytest.mark.parametrize("Ty", FORMATS, ids=lambda t: t.name)
def test_mx_conv2d_chain(Ty):
    rng = np.random.default_rng(1)
    inp = _rand((1, C0, H0, H0), rng)
    k1 = mx_pack_np(Ty, _rand((C1, C0, K0, K0), rng), axis=1)
    k2 = mx_pack_np(Ty, _rand((C2, C1, K0, K0), rng), axis=1)
    b1, b2 = _rand((C1,), rng), _rand((C2,), rng)

    # numerics: the chain equals the two layers run one by one
    Y = allo.customize(model, instantiate=[Ty]).build(target="llvm")(
        inp, k1, b1, k2, b2
    )
    conv = lambda inst: allo.customize(nn.mx_conv2d, instantiate=[Ty, *inst]).build(
        target="llvm"
    )
    a = conv(CONV1)(mx_pack_np(Ty, inp, axis=1), k1, b1)
    np.testing.assert_array_equal(Y, conv(CONV2)(mx_pack_np(Ty, a, axis=1), k2, b2))

    code = str(_compose_model(Ty).build(target="vhls"))
    assert "#pragma HLS dataflow" in code  # the two layers overlap
    assert code.count("#pragma HLS pipeline") >= 4  # quantizers + block-dot loops


def test_mx_conv2d_chain_csyn():
    if os.environ.get("ALLO_HLSMODE", "csyn") != "csyn":
        return
    if not hls.is_available("vitis_hls"):
        return
    project = os.environ.get("ALLO_PROJECT") or tempfile.mkdtemp(suffix=".prj")
    hls_mod = _compose_model(FORMATS[0]).build(
        target="vitis_hls", mode="csyn", project=project
    )
    # issue #603: extern "C" functions returning ap_int<N> (the packed MX word)
    patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
    hls_mod()
    rpt = os.path.join(
        project, "out.prj", "solution1", "syn", "report", "model_csynth.rpt"
    )
    assert os.path.isfile(rpt)
    print(f"csynth report: {rpt}")


if __name__ == "__main__":
    test_mx_conv2d_chain(FORMATS[0])
    test_mx_conv2d_chain_csyn()
