# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation, chained-comparison, unused-import

import re

import allo
from .. import dsl
from .systolic import systolic

# the stages the MX layers are built from (also reachable as nn.<name>)
from .mx import (
    mx_quantize_block_f32,
    mx_block_dot,
    mx_quantize_rows,
    mx_quantize_cols,
    _mx_dequantize_elem,
    mx_dequantize,
    schedule_mx,
    partition_block_lanes,
    mx_unpack_f32,
    schedule_mx_unpack_f32,
    schedule_mx_quantize_rows,
    mx_quantize_channels,
    schedule_mx_quantize_channels,
    mx_matmul,
    pipeline_mx_matmul,
    schedule_mx_matmul,
    bias_add,
    merge_rows,
    split_rows,
    mx_conv_windows,
    rows_to_nchw,
)

# names the Allo bodies and string annotations below resolve here
from ..ir.types import (
    float32,
    uint8,
    int32,
    UInt,
    ConstExpr,
    Stream,
)
from ..ir.utils import MockBuffer
from .._mlir.dialects import func as func_d
from .._mlir.ir import MemRefType  # pylint: disable=no-name-in-module


def linear2d[
    TyX, TyW, TyO, M, N, K
](X: "TyX[M, K]", W: "TyW[N, K]", b: "TyO[N]") -> "TyO[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Z: TyO[M, N]
    buf: TyO[N]
    for i in range(M):
        for j_init in range(N):
            buf[j_init] = 0
        for k in range(K):
            # reorder reduction loop outside, and pipeline
            x: TyX = X[i, k]
            for j in range(N):
                buf[j] += x * W[j, k]
        for j_back in range(N):
            Z[i, j_back] = buf[j_back] + b[j_back]
    return Z


def schedule_linear2d(s):
    s.pipeline("linear2d:j")
    s.pipeline("linear2d:j_init")
    s.pipeline("linear2d:j_back")
    return s


def linear3d[
    TyX, TyW, TyO, B, L, D, M
](X: "TyX[B, L, D]", W: "TyW[M, D]", bias: "TyO[M]") -> "TyO[B, L, M]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Z: TyO[B, L, M]
    buf: TyO[M]
    for b in range(B):
        for i in range(L):
            for j_init in range(M):
                buf[j_init] = 0
            for k in range(D):
                # reorder reduction loop outside, and pipeline
                x: TyX = X[b, i, k]
                for j in range(M):
                    buf[j] += x * W[j, k]
            for j_back in range(M):
                Z[b, i, j_back] = buf[j_back] + bias[j_back]
    return Z


def schedule_linear3d(s):
    s.pipeline("linear3d:j")
    s.pipeline("linear3d:j_init")
    s.pipeline("linear3d:j_back")
    return s


# --- MX layers: operands are Ty arrays (one scale | elems word per block) ------
# Quantize activations on chip with mx_quantize_rows / mx_quantize_channels, and
# weights once on the host with mx.mx_pack_np.


def mx_linear2d[
    Ty, TyX, M, N, K
](
    X: "TyX[M, K]",
    Wq: "Ty[N, (K + Ty.block_size - 1) // Ty.block_size]",
    bias: "TyX[N]",
) -> "TyX[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html with MX
    # arithmetic: X is quantized here, Wq on the host (mx.mx_pack_np)
    Xq = mx_quantize_rows[Ty, TyX, M, K](X)
    C = mx_matmul[Ty, M, N, K](Xq, Wq)
    Z = bias_add[TyX, M, N](C, bias)
    return Z


def schedule_mx_linear2d(s):
    """quantize -> mx_matmul -> bias_add as an HLS dataflow, streamed one
    block (one output) per token. Registered in KERNEL2SCHEDULE, so
    s.compose(mx_linear2d, instantiate=..., id=...) applies it."""
    Ty, _, _, N, K = s.inst_list[:5]
    top = s.top_func_name
    schedule_mx(s)
    partition_block_lanes(s, MockBuffer(top, "X"), K, Ty.block_size)
    s.pipeline("mx_quantize_rows:b")
    pipeline_mx_matmul(s, "mx_matmul", N)
    s.pipeline("bias_add:n")
    s.to(
        MockBuffer(top, "Xq"),
        "mx_matmul",
        depth=(K + Ty.block_size - 1) // Ty.block_size,
    )
    s.to(MockBuffer(top, "C"), "bias_add", depth=N)
    s.dataflow(top)
    return s


def mx_linear3d[
    Ty, TyX, B, L, D, M
](
    X: "TyX[B, L, D]",
    Wq: "Ty[M, (D + Ty.block_size - 1) // Ty.block_size]",
    bias: "TyX[M]",
) -> "TyX[B, L, M]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html: rows are
    # independent, so the B x L rows go through one mx_linear2d
    X2 = merge_rows[TyX, B, L, D](X)
    Z2 = mx_linear2d[Ty, TyX, B * L, M, D](X2, Wq, bias)
    Z = split_rows[TyX, B, L, M](Z2)
    return Z


def schedule_mx_linear3d(s):
    """merge_rows -> mx_linear2d -> split_rows as an HLS dataflow."""
    Ty, TyX, B, L, D, M = s.inst_list[:6]
    top = s.top_func_name
    s.compose(mx_linear2d, instantiate=[Ty, TyX, B * L, M, D])
    s.pipeline("merge_rows:d")
    s.pipeline("split_rows:d")
    s.to(MockBuffer(top, "X2"), "mx_linear2d", depth=D)
    s.to(MockBuffer(top, "Z2"), "split_rows", depth=M)
    s.dataflow(top)
    return s


def mx_conv2d[
    Ty, TyX, B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    X: "TyX[B, Cin, H, W]",
    Wq: "Ty[Cout, (Cin + Ty.block_size - 1) // Ty.block_size * Kh * Kw]",
    bias: "TyX[Cout]",
) -> "TyX[B, Cout, Oh, Ow]":
    # nn.conv2d with MX arithmetic, blocks along the input channels: X is
    # quantized here, Wq on the host (mx.mx_pack_channels_np). Each output
    # pixel's receptive field is one row of blocks, so the conv is a matmul of
    # those rows with the weights (flattened the same way), then the bias
    Xq = mx_quantize_channels[Ty, TyX, B, Cin, H, W](X)
    Xw = mx_conv_windows[Ty, B, Cin, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1](Xq)
    C = mx_matmul[
        Ty,
        B * Oh * Ow,
        Cout,
        (Cin + Ty.block_size - 1) // Ty.block_size * Ty.block_size * Kh * Kw,
    ](Xw, Wq)
    Zr = bias_add[TyX, B * Oh * Ow, Cout](C, bias)
    Z = rows_to_nchw[TyX, B, Cout, Oh, Ow](Zr)
    return Z


def schedule_mx_conv2d(s):
    """quantize -> windows -> mx_matmul -> bias_add -> NCHW as an HLS
    dataflow, streamed one block (one output) per token, one block dot per
    cycle. Registered in KERNEL2SCHEDULE, so s.compose(mx_conv2d, ...)
    applies it."""
    Ty, _, _, Cin, Cout, _, _, Kh, Kw = s.inst_list[:9]
    top = s.top_func_name
    NB = (Cin + Ty.block_size - 1) // Ty.block_size
    schedule_mx(s)
    partition_block_lanes(s, MockBuffer(top, "X"), Cin, Ty.block_size)
    s.pipeline("mx_quantize_channels:w")
    s.pipeline("mx_conv_windows:kw")
    pipeline_mx_matmul(s, "mx_matmul", Cout)
    s.pipeline("bias_add:n")
    s.pipeline("rows_to_nchw:c")
    s.to(MockBuffer(top, "Xw"), "mx_matmul", depth=NB * Kh * Kw)
    s.to(MockBuffer(top, "C"), "bias_add", depth=Cout)
    s.to(MockBuffer(top, "Zr"), "rows_to_nchw", depth=Cout)
    s.dataflow(top)
    return s


# --- dataflow across MX ops ---------------------------------------------------
# Each MX op's stream ports: the inputs it reads exactly once in row-major
# order, and whether it writes its result in that order. mx_dataflow turns a
# buffer into a FIFO only when its producer writes it in order and its single
# consumer reads it in order. A new MX op adds its entry here.
MX_STREAM_PORTS = {
    "mx_quantize_rows": ((0,), True),
    "mx_quantize_cols": ((), True),  # reads X down the columns
    "mx_quantize_channels": ((), True),  # reads X across the channels
    "mx_matmul": ((0,), True),  # Bq is re-read for every row
    "mx_linear2d": ((0,), True),  # Wq is re-read for every row
    "bias_add": ((0,), True),
    "mx_conv_windows": ((), True),  # windows overlap
    "mx_conv2d": ((), False),  # X feeds overlapping windows; Z is NCHW
}


def mx_dataflow(s, depth=None):
    """HLS dataflow over the calls in `s`'s top function: every MX result
    that the next MX op reads once, in the order it is written, becomes a
    FIFO (`depth`: default one row). Call after composing the ops' schedules.
    Returns the names of the streamed buffers."""
    top = s.top_func_name
    ports = {}
    for op in s._find_function(top).entry_block.operations:
        if isinstance(op, func_d.CallOp):
            callee = op.attributes["callee"].value
            ports[callee] = MX_STREAM_PORTS.get(re.sub(r"_\d+$", "", callee))
    streamed = []
    for op in s._find_function(top).entry_block.operations:
        if not isinstance(op, func_d.CallOp) or "name" not in op.attributes:
            continue
        src = ports[op.attributes["callee"].value]
        uses = list(op.result.uses)
        if src is None or not src[1] or len(uses) != 1:
            continue
        user = uses[0].owner
        if not isinstance(user, func_d.CallOp):
            continue
        dst_name = user.attributes["callee"].value
        dst = ports[dst_name]
        if dst is None or uses[0].operand_number not in dst[0]:
            continue
        name = op.attributes["name"].value
        row = MemRefType(op.result.type).shape[-1]
        s.to(MockBuffer(top, name), dst_name, depth=depth or row)
        streamed.append(name)
    s.dataflow(top)
    return streamed


def relu2d[Ty, H, W](X: "Ty[H, W]") -> "Ty[H, W]":
    Z: Ty[H, W]
    for h, w in dsl.grid(H, W):
        Z[h, w] = max(0.0, X[h, w])
    return Z


def schedule_relu2d(s):
    s.pipeline("relu2d:w")
    return s


def relu4d[Ty, N, C, H, W](X: "Ty[N, C, H, W]") -> "Ty[N, C, H, W]":
    Z: Ty[N, C, H, W]
    for n, c, h, w in dsl.grid(N, C, H, W):
        Z[n, c, h, w] = max(0.0, X[n, c, h, w])
    return Z


def schedule_relu4d(s):
    s.pipeline("relu4d:w")
    return s


def relu3d[Ty, N, L, C](X: "Ty[N, L, C]") -> "Ty[N, L, C]":
    Z: Ty[N, L, C]
    for n, l, c in dsl.grid(N, L, C):
        Z[n, l, c] = max(0.0, X[n, l, c])
    return Z


def schedule_relu3d(s):
    s.pipeline("relu3d:c")
    return s


def softmax[Ty, L](X: "Ty[L, L]") -> "Ty[L, L]":
    Z: Ty[L, L]
    E: Ty[L, L]
    M: Ty[L] = -1000000000000.0
    S: Ty[L] = 0.0

    for i, j in dsl.grid(L, L, name="row_max"):
        if X[i, j] > M[i]:
            M[i] = X[i, j]

    # compute exp and sum
    for i, j in dsl.grid(L, L, name="exp_sum"):
        E[i, j] = dsl.exp(X[i, j] - M[i])
        S[i] += E[i, j]

    for i, j in dsl.grid(L, L, name="update"):
        Z[i, j] = E[i, j] / S[i]

    return Z


def schedule_softmax(s):
    lj = s.get_loops(s.top_func_name)["exp_sum"]["j"]
    s.pipeline(lj)
    lj = s.get_loops(s.top_func_name)["update"]["j"]
    s.pipeline(lj)
    return s


def log_softmax[Ty, B, C](X: "Ty[B, C]") -> "Ty[B, C]":
    Z: Ty[B, C]
    E: Ty[B, C]
    M: Ty[B] = -1000000000000.0
    S: Ty[B] = 0.0

    # Row-wise max
    for i, j in dsl.grid(B, C, name="row_max"):
        if X[i, j] > M[i]:
            M[i] = X[i, j]

    # Compute exp and sum
    for i, j in dsl.grid(B, C, name="exp_sum"):
        E[i, j] = dsl.exp(X[i, j] - M[i])
        S[i] += E[i, j]

    # Log softmax update
    for i, j in dsl.grid(B, C, name="update"):
        Z[i, j] = X[i, j] - M[i] - dsl.log(S[i])

    return Z


def schedule_log_softmax(s):
    lj = s.get_loops(s.top_func_name)["exp_sum"]["j"]
    s.pipeline(lj)
    lj = s.get_loops(s.top_func_name)["update"]["j"]
    s.pipeline(lj)
    return s


def layer_norm[Ty, L, D](X: "Ty[L, D]", gamma: "Ty[D]", beta: "Ty[D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    mean: Ty[L] = 0.0
    mean2: Ty[L] = 0.0
    var: Ty[L]

    for i, j in dsl.grid(L, D, name="sum"):
        mean[i] += X[i, j]
        mean2[i] += X[i, j] * X[i, j]

    for i in dsl.grid(L, name="mean_var"):
        mean[i] = mean[i] / float(D)
        mean2[i] = mean2[i] / float(D)
        var[i] = mean2[i] - mean[i] * mean[i]

    for i, j in dsl.grid(L, D, name="norm"):
        Z[i, j] = gamma[j] * (X[i, j] - mean[i]) / dsl.sqrt(var[i] + 0.00001) + beta[j]

    return Z


def schedule_layernorm(s):
    lj = s.get_loops(s.top_func_name)["sum"]["j"]
    s.pipeline(lj)
    li = s.get_loops(s.top_func_name)["mean_var"]["i"]
    s.pipeline(li)
    lj = s.get_loops(s.top_func_name)["norm"]["j"]
    s.pipeline(lj)
    return s


def GeLU[Ty, L, D](X: "Ty[L, D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    for i, j in dsl.grid(L, D, name="gelu"):
        Z[i, j] = (
            0.5
            * X[i, j]
            * (
                1.0
                + dsl.tanh(0.797885 * (X[i, j] + 0.044715 * dsl.power(X[i, j], 3.0)))
            )
        )
    return Z


def schedule_gelu(s):
    lj = s.get_loops(s.top_func_name)["gelu"]["j"]
    s.pipeline(lj)
    return s


def residual_add[Ty, L, D](X1: "Ty[L, D]", X2: "Ty[L, D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    for i, j in dsl.grid(L, D):
        Z[i, j] = X1[i, j] + X2[i, j]
    return Z


def scaled_dot_product_attention[
    Ty, H, L, D, M0, M1
](Q: "Ty[L, D]", K: "Ty[L, D]", V: "Ty[L, D]") -> "Ty[L, D]":
    # softmax(QK^T/sqrt(D // H))
    Z: Ty[L, D]

    for h in range(H):
        Q_h: Ty[L, D // H]
        K_h: Ty[D // H, L]
        V_h: Ty[L, D // H]

        # split Q, K, V
        for i, j in dsl.grid(L, D // H, name="mha_split"):
            Q_h[i, j] = Q[i, h * (D // H) + j]
            # transposed
            K_h[j, i] = K[i, h * (D // H) + j]
            V_h[i, j] = V[i, h * (D // H) + j]

        # QK^T = (L, D//H) x (D//H, L) = (L, L)
        C_h: Ty[L, D // H] = 0
        Y: Ty[L, L] = 0
        systolic[Ty, Ty, Ty, L, D // H, L, M0, M1, "QKT"](Q_h, K_h, Y)
        # Need to return a new value
        S = softmax[Ty, L](Y)
        # YV = (L, L) x (L, D//H) = (L, D//H)
        systolic[Ty, Ty, Ty, L, L, D // H, M0, M1, "YV"](S, V_h, C_h)

        for i, j in dsl.grid(L, D // H, name="mha_merge"):
            Z[i, h * (D // H) + j] = C_h[i, j]

    return Z


def RoPE[
    Ty, H, L, D
](X: "Ty[L, D]", cos: "Ty[L, D // H // 2]", sin: "Ty[L, D // H // 2]") -> "Ty[L, D]":
    # Rotary Position Embedding
    # Reference: https://arxiv.org/abs/2104.09864
    X_rotary: Ty[L, D]
    for h in range(H):
        X_1_h: Ty[L, D // H // 2]
        X_2_h: Ty[L, D // H // 2]
        for i, j in dsl.grid(L, D // H // 2, name="rope_split_1"):
            X_1_h[i, j] = X[i, h * (D // H) + j]
        for i, j in dsl.grid(L, D // H // 2, name="rope_split_2"):
            X_2_h[i, j] = X[i, h * (D // H) + D // H // 2 + j]
        X_1_rotary: Ty[L, D // H // 2] = 0
        X_2_rotary: Ty[L, D // H // 2] = 0
        for i, j in dsl.grid(L, D // H // 2, name="rotary_1"):
            X_1_rotary[i, j] = cos[i, j] * X_1_h[i, j] - sin[i, j] * X_2_h[i, j]
        for i, j in dsl.grid(L, D // H // 2, name="rotary_2"):
            X_2_rotary[i, j] = sin[i, j] * X_1_h[i, j] + cos[i, j] * X_2_h[i, j]
        for i, j in dsl.grid(L, D // H // 2, name="rotary_merge_1"):
            X_rotary[i, h * (D // H) + j] = X_1_rotary[i, j]
        for i, j in dsl.grid(L, D // H // 2, name="rotary_merge_2"):
            X_rotary[i, h * (D // H) + D // H // 2 + j] = X_2_rotary[i, j]
    return X_rotary


def modulate_fused[
    Ty, L, D
](X: "Ty[L,D]", scale: "Ty[D]", shift: "Ty[D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    for i, j in dsl.grid(L, D, name="m_fused"):
        Z[i, j] = X[i, j] * (1 + scale[j]) + shift[j]
    return Z


def schedule_modulate_fused(s):
    lj = s.get_loops(s.top_func_name)["m_fused"]["j"]
    s.pipeline(lj)


def conv2d[
    Ty, B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    inp: "Ty[B, Cin, H, W]", kernel: "Ty[Cout, Cin, Kh, Kw]", bias: "Ty[Cout]"
) -> "Ty[B, Cout, Oh, Ow]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Z: Ty[B, Cout, Oh, Ow]

    # Current implementation is does not support dilation other than 1
    for batch, cout, oh, ow in dsl.grid(B, Cout, Oh, Ow):
        temp: Ty = bias[cout]

        for cin, kh, kw in dsl.grid(Cin, Kh, Kw):
            h_pos: Ty = oh * Sh + kh - Pd0
            w_pos: Ty = ow * Sw + kw - Pd1
            if h_pos >= 0 and h_pos < H and w_pos >= 0 and w_pos < W:
                temp += inp[batch, cin, h_pos, w_pos] * kernel[cout, cin, kh, kw]

        Z[batch, cout, oh, ow] = temp
    return Z


def schedule_conv2d(s):
    s.pipeline("conv2d:cout")
    s.pipeline("conv2d:ow")
    return s


def maxpool2d[
    Ty, B, C, H, W, K, Oh, Ow, S, Pd
](inp: "Ty[B, C, H, W]",) -> "Ty[B, C, Oh, Ow]":
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    Z: Ty[B, C, Oh, Ow]
    for batch, c, oh, ow in dsl.grid(B, C, Oh, Ow):
        max_val: Ty = -1000000000000.0
        for kh, kw in dsl.grid(K, K):
            h_pos: Ty = oh * S + kh - Pd
            w_pos: Ty = ow * S + kw - Pd
            if h_pos >= 0 and h_pos < H and w_pos >= 0 and w_pos < W:
                new_max: Ty = max(max_val, inp[batch, c, h_pos, w_pos])
                max_val = new_max
        Z[batch, c, oh, ow] = max_val
    return Z


def schedule_maxpool2d(s):
    s.pipeline("maxpool2d:c")
    s.pipeline("maxpool2d:ow")
    return s


def avgpool2d[
    Ty, B, C, H, W, K, Oh, Ow, S, Pd
](inp: "Ty[B, C, H, W]",) -> "Ty[B, C, Oh, Ow]":
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    Z: Ty[B, C, Oh, Ow]
    for batch, c, oh, ow in dsl.grid(B, C, Oh, Ow):
        temp: Ty = 0.0
        for kh, kw in dsl.grid(K, K):
            h_pos: Ty = oh * S + kh - Pd
            w_pos: Ty = ow * S + kw - Pd
            if h_pos >= 0 and h_pos < H and w_pos >= 0 and w_pos < W:
                temp += inp[batch, c, h_pos, w_pos]
        Z[batch, c, oh, ow] = temp / (K * K)
    return Z


def schedule_avgpool2d(s):
    s.pipeline("avgpool2d:c")
    s.pipeline("avgpool2d:ow")
    return s


def batchnorm2d[
    Ty, B, C, H, W
](
    X: "Ty[B, C, H, W]",
    gamma: "Ty[C]",
    beta: "Ty[C]",
    eps: "Ty",
    mean: "Ty[C]",
    var: "Ty[C]",
) -> "Ty[B, C, H, W]":
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    Z: Ty[B, C, H, W]
    for b, c, h, w in dsl.grid(B, C, H, W):
        Z[b, c, h, w] = (
            gamma[c] * (X[b, c, h, w] - mean[c]) / dsl.sqrt(var[c] + eps) + beta[c]
        )

    return Z


def schedule_batchnorm2d(s):
    s.pipeline("batchnorm2d:w")
    return s


def batchnorm1d_2d[
    Ty, B, C
](
    X: "Ty[B, C]", gamma: "Ty[C]", beta: "Ty[C]", eps: "Ty", mean: "Ty[C]", var: "Ty[C]"
) -> "Ty[B, C]":
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    Z: Ty[B, C]
    for b, c in dsl.grid(B, C):
        Z[b, c] = gamma[c] * (X[b, c] - mean[c]) / dsl.sqrt(var[c] + eps) + beta[c]
    return Z


def schedule_batchnorm1d_2d(s):
    s.pipeline("batchnorm1d_2d:c")
    return s


def batchnorm1d_3d[
    Ty, B, C, L
](
    X: "Ty[B, C, L]",
    gamma: "Ty[C]",
    beta: "Ty[C]",
    eps: "Ty",
    mean: "Ty[C]",
    var: "Ty[C]",
) -> "Ty[B, C, L]":
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    Z: Ty[B, C, L]
    for b, c, l in dsl.grid(B, C, L):
        Z[b, c, l] = (
            gamma[c] * (X[b, c, l] - mean[c]) / dsl.sqrt(var[c] + eps) + beta[c]
        )
    return Z


def schedule_batchnorm1d_3d(s):
    s.pipeline("batchnorm1d_3d:l")
    return s


def repeat_batch3d[Ty, B, L, C, N](X: "Ty[B, L, C]") -> "Ty[N*B, L, C]":
    """
    Repeat X along batch dimension N times for cls_token.
    """
    Y: Ty[N * B, L, C]
    for r, b, l, c in dsl.grid(N, B, L, C):
        Y[r * B + b, l, c] = X[b, l, c]
    return Y


def schedule_repeat_batch3d(s):
    s.pipeline("repeat_batch3d:c")
    return s


def concat[
    Ty, B, N1, N2, C
](X1: "Ty[B, N1, C]", X2: "Ty[B, N2, C]") -> "Ty[B, N1+N2, C]":
    Y: Ty[B, N1 + N2, C]
    for b, n, c in dsl.grid(B, N1, C):
        Y[b, n, c] = X1[b, n, c]
    for b2, n2, c2 in dsl.grid(B, N2, C):
        Y[b2, n2 + N1, c2] = X2[b2, n2, c2]
    return Y


def schedule_concat(s):
    s.pipeline("concat:c")
    return s
