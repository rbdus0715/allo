# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Chaining nn.mx_linear2d calls: each call's registered dataflow schedule
# (row pack -> MX block quantize -> block dot, one MX block per stream token)
# is applied per call site with s.compose(nn.mx_linear2d, instantiate=..., id=...).

import numpy as np
import pytest
import allo
import allo.ir.types as T
import allo.library.nn as nn

# MX formats as MXInt(elem_bits, block_size) / MXFP(exp_bits, mantissa_bits, block_size)
FORMATS = [T.MXInt(8, 32), T.MXInt(4, 16), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32)]

# fc1: K0 -> N1, fc2: N1 -> N2 on M rows
M, K0, N1, N2 = 8, 96, 64, 32


def mlp[
    Ty
](
    X: "float32[M, K0]",
    W1: "float32[N1, K0]",
    b1: "float32[N1]",
    W2: "float32[N2, N1]",
    b2: "float32[N2]",
) -> "float32[M, N2]":
    a = nn.mx_linear2d[Ty, M, N1, K0](X, W1, b1)
    return nn.mx_linear2d[Ty, M, N2, N1](a, W2, b2)


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


@pytest.mark.parametrize("Ty", FORMATS, ids=lambda t: t.name)
def test_mx_linear2d_chain(Ty):
    rng = np.random.default_rng(2)
    X = _rand((M, K0), rng)
    W1, b1 = _rand((N1, K0), rng), _rand((N1,), rng)
    W2, b2 = _rand((N2, N1), rng), _rand((N2,), rng)

    # numerics: the chain equals the two layers run one by one
    FC1, FC2 = [Ty, M, N1, K0], [Ty, M, N2, N1]
    Y = allo.customize(mlp, instantiate=[Ty]).build(target="llvm")(X, W1, b1, W2, b2)
    fc = lambda inst: allo.customize(nn.mx_linear2d, instantiate=inst).build(
        target="llvm"
    )
    a = fc(FC1)(X, W1, b1)
    np.testing.assert_array_equal(Y, fc(FC2)(a, W2, b2))

    s = allo.customize(mlp, instantiate=[Ty])
    # the n-th call (and its helpers) is suffixed "_n" for n > 0
    s.compose(nn.mx_linear2d, instantiate=FC1)
    s.compose(nn.mx_linear2d, instantiate=FC2, id="1")
    s.dataflow("mlp")
    code = str(s.build(target="vhls"))
    assert code.count("#pragma HLS dataflow") >= 3  # mlp + one per layer
    assert code.count("#pragma HLS stream") >= 6  # P, XS, XE per layer


if __name__ == "__main__":
    test_mx_linear2d_chain(FORMATS[0])
