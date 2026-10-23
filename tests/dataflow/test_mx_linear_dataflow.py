# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Chaining nn.mx_linear2d calls: each call's registered dataflow schedule
# (row pack -> MX block quantize -> block dot, one MX block per stream token)
# is applied per call site with s.compose(nn.mx_linear2d, instantiate=..., id=...).

import numpy as np
import allo
import allo.ir.types as T
import allo.library.nn as nn

Ty = T.mxint8

# fc1: K0 -> N1, fc2: N1 -> N2 on M rows
M, K0, N1, N2 = 8, 96, 64, 32
FC1 = [Ty, M, N1, K0]
FC2 = [Ty, M, N2, N1]


def mlp(
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


def test_mx_linear2d_chain():
    rng = np.random.default_rng(2)
    X = _rand((M, K0), rng)
    W1, b1 = _rand((N1, K0), rng), _rand((N1,), rng)
    W2, b2 = _rand((N2, N1), rng), _rand((N2,), rng)

    # numerics: the chain equals the two layers run one by one
    Y = allo.customize(mlp).build(target="llvm")(X, W1, b1, W2, b2)
    fc = lambda inst: allo.customize(nn.mx_linear2d, instantiate=inst).build(
        target="llvm"
    )
    a = fc(FC1)(X, W1, b1)
    np.testing.assert_array_equal(Y, fc(FC2)(a, W2, b2))

    s = allo.customize(mlp)
    # the n-th call (and its helpers) is suffixed "_n" for n > 0
    s.compose(nn.mx_linear2d, instantiate=FC1)
    s.compose(nn.mx_linear2d, instantiate=FC2, id="1")
    s.dataflow("mlp")
    code = str(s.build(target="vhls"))
    assert code.count("#pragma HLS dataflow") >= 3  # mlp + one per layer
    assert code.count("#pragma HLS stream") >= 6  # P, XS, XE per layer


if __name__ == "__main__":
    test_mx_linear2d_chain()
