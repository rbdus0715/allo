# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation, chained-comparison, no-value-for-parameter

import re

import allo
from .. import dsl
from .systolic import systolic
from .mxint8 import (
    mx_quantize_block_f32,
    mx_block_dot,
    _mx_pack_word,
    schedule_mx_quantize_block_f32,
    schedule_mx_block_dot,
)
from ..ir.types import float32, uint8, int32, UInt, ConstExpr, Stream
from ..ir.utils import MockBuffer
from .._mlir.dialects import func as func_d
from .._mlir.ir import MemRefType


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


def mx_matmul_ref[
    Ty, M, N, K
](A: "float32[M, K]", B: "float32[N, K]") -> "float32[M, N]":
    # reference: A @ B^T with A, B quantized block by block (float32 operands)
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = K // BS
    Z: float32[M, N]
    for i in range(M):
        for j in range(N):
            acc: float32 = 0.0
            for b in range(NB):
                blk_a: float32[BS]
                blk_b: float32[BS]
                for e in range(BS):
                    blk_a[e] = A[i, b * BS + e]
                    blk_b[e] = B[j, b * BS + e]
                qa: Ty = mx_quantize_block_f32[Ty, BS](blk_a)
                qb: Ty = mx_quantize_block_f32[Ty, BS](blk_b)
                acc += mx_block_dot[Ty, BS](qa, qb)
            Z[i, j] = acc
    return Z


def schedule_mx_matmul_ref(s):
    schedule_mx_quantize_block_f32(s)
    schedule_mx_block_dot(s)
    s.unroll("mx_matmul_ref:e")
    s.pipeline("mx_matmul_ref:b")
    s.pipeline("mx_matmul_ref:j")
    return s


def mx_linear2d_ref[
    Ty, M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    Z = mx_matmul_ref[Ty, M, N, K](X, W)
    for i in range(M):
        for j in range(N):
            Z[i, j] = Z[i, j] + bias[j]
    return Z


def _mx_quantize_rows[
    Ty, R, K, NB, TRANS
](X: "float32[K if TRANS else R, R if TRANS else K]", Q: "Ty[R, NB]"):
    # X is [R, K], or [K, R] when TRANS; blocks run along K either way, the
    # last one zero-padded when K is not a multiple of the block size
    BS: ConstExpr[int32] = Ty.block_size
    for r, b in allo.grid(R, NB):
        blk: float32[BS]
        for e in range(BS):
            fq: float32 = 0.0
            if b * BS + e < K:
                with allo.meta_if(TRANS):
                    fq = X[b * BS + e, r]
                with allo.meta_else():
                    fq = X[r, b * BS + e]
            blk[e] = fq
        Q[r, b] = mx_quantize_block_f32[Ty, BS](blk)


def _mx_pack_rows[
    Ty, R, K, NB, TRANS
](
    E: "Int(Ty.elem_bits)[K if TRANS else R, R if TRANS else K]",
    S: "e8m0[NB if TRANS else R, R if TRANS else NB]",
    Q: "Ty[R, NB]",
):
    # pre-quantized (elements, scales) -> packed scale | elems words
    BS: ConstExpr[int32] = Ty.block_size
    EB: ConstExpr[int32] = Ty.elem_bits
    for r, b in allo.grid(R, NB):
        w: Ty = 0
        for e in range(BS):
            with allo.meta_if(TRANS):
                w[e * EB : (e + 1) * EB] = E[b * BS + e, r]
            with allo.meta_else():
                w[e * EB : (e + 1) * EB] = E[r, b * BS + e]
        with allo.meta_if(TRANS):
            w[Ty.bits - 8 : Ty.bits] = S[b, r]
        with allo.meta_else():
            w[Ty.bits - 8 : Ty.bits] = S[r, b]
        Q[r, b] = w


def _mx_dot_rows[
    Ty, M, N, NB
](qx: "Ty[M, NB]", qw: "Ty[N, NB]", bias: "float32[N]", Z: "float32[M, N]"):
    # each X block is read once, in order, so qx can be a FIFO; W stays resident
    BS: ConstExpr[int32] = Ty.block_size
    acc: float32[N]
    for i in range(M):
        for j_init in range(N):
            acc[j_init] = 0.0
        for b in range(NB):
            xb: Ty = qx[i, b]
            for j in range(N):
                acc[j] += mx_block_dot[Ty, BS](xb, qw[j, b])
        for j_back in range(N):
            Z[i, j_back] = acc[j_back] + bias[j_back]


# ---- MX matrix API: Ty[R, NB] is an R x K matrix in MX blocks along K ----


def mx_quantize_rows[
    Ty, TyX, R, K
](X: "TyX[R, K]") -> "Ty[R, (K + Ty.block_size - 1) // Ty.block_size]":
    # float32 [R, K] -> MX matrix: one packed scale | elems word per block along
    # K (zero-padded to whole blocks); X is read once, row by row
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    Q: Ty[R, NB]
    for r, b in allo.grid(R, NB):
        blk: float32[BS]
        for e in range(BS):
            fq: float32 = 0.0
            if b * BS + e < K:
                fq = X[r, b * BS + e]
            blk[e] = fq
        Q[r, b] = mx_quantize_block_f32[Ty, BS](blk)
    return Q


def mx_quantize_cols[
    Ty, TyX, R, K
](X: "TyX[K, R]") -> "Ty[R, (K + Ty.block_size - 1) // Ty.block_size]":
    # float32 [K, R] -> MX matrix of X^T: blocks run down the columns, e.g.
    # V in attention's P @ V (mx_linear2d reduces along the block axis)
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    Q: Ty[R, NB]
    for r, b in allo.grid(R, NB):
        blk: float32[BS]
        for e in range(BS):
            fq: float32 = 0.0
            if b * BS + e < K:
                fq = X[b * BS + e, r]
            blk[e] = fq
        Q[r, b] = mx_quantize_block_f32[Ty, BS](blk)
    return Q


# earlier name; layers call mx_quantize_rows, which host code (mxint8.mx_quantize)
# does not shadow in the caller's globals
mx_quantize = mx_quantize_rows


def mx_pack[
    Ty, R, K
](
    E: "Int(Ty.elem_bits)[R, K]", S: "e8m0[R, K // Ty.block_size]"
) -> "Ty[R, K // Ty.block_size]":
    # host-quantized (elements, scales), e.g. from mxint8.mx_quantize -> MX matrix
    Q: "Ty[R, K // Ty.block_size]"
    _mx_pack_rows[Ty, R, K, K // Ty.block_size, 0](E, S, Q)
    return Q


def mx_dot[Ty, NB](x: "Ty[NB]", w: "Ty[NB]") -> float32:
    # dot product of two MX rows: exact within each block, float32 across blocks
    BS: ConstExpr[int32] = Ty.block_size
    acc: float32 = 0.0
    for b in range(NB):
        acc += mx_block_dot[Ty, BS](x[b], w[b])
    return acc


def _mx_dequantize_elem[Ty](e: "UInt(Ty.elem_bits)", scale: int32) -> float32:
    # value = mag * 2^sh, built by adjusting the exponent field of float(mag)
    v: int32 = int(e)
    neg: int32 = (v >> (Ty.elem_bits - 1)) & 1
    mag: int32 = 0
    sh: int32 = scale - 127
    with allo.meta_if(Ty.is_float):
        exp_field: int32 = (v >> Ty.mantissa_bits) & ((1 << Ty.exp_bits) - 1)
        mag = v & ((1 << Ty.mantissa_bits) - 1)
        sh = sh + 1 - Ty.bias - Ty.mantissa_bits  # subnormal: 0.m * 2^(1 - bias)
        if exp_field != 0:  # normal: 1.m * 2^(exp_field - bias)
            mag = mag | (1 << Ty.mantissa_bits)
            sh = sh + exp_field - 1
    with allo.meta_else():
        mag = v
        if neg == 1:  # two's complement
            mag = (1 << Ty.elem_bits) - v
    f: float32 = float(mag)
    bits: int32 = f.bitcast()
    new_exp: int32 = int(bits[23:31]) + sh
    result_bits: int32 = 0
    if mag != 0 and new_exp >= 255:
        result_bits = (neg << 31) | (255 << 23)
    elif mag != 0 and new_exp > 0:
        result_bits = (neg << 31) | (new_exp << 23) | (bits & 0x7FFFFF)
    return result_bits.bitcast()


def mx_dequantize[Ty, R, K](Q: "Ty[R, K // Ty.block_size]") -> "float32[R, K]":
    # MX matrix -> float32 [R, K] (debugging, mixing with other precisions)
    BS: ConstExpr[int32] = Ty.block_size
    EB: ConstExpr[int32] = Ty.elem_bits
    X: float32[R, K]
    for r, b in allo.grid(R, K // BS):
        w: Ty = Q[r, b]
        scale: int32 = int(w[Ty.bits - 8 : Ty.bits])
        for e in range(BS):
            el: UInt(Ty.elem_bits) = w[e * EB : (e + 1) * EB]
            X[r, b * BS + e] = _mx_dequantize_elem[Ty](el, scale)
    return X


def schedule_mx(s, id=None):
    """Unrolls the block-level loops inside the MX primitives a kernel uses.

    Like tests/dataflow/test_mlp.py's schedule_linear: `id` names the n-th
    instantiation (suffix "_<id>"), as in compose(..., id=...).
    """
    sfx = "" if id is None else f"_{id}"
    for fn, loops in (
        ("mx_block_dot", ("j0", "j1")),
        ("mx_quantize_block_f32", ("i1",)),
    ):
        if s._find_function(fn + sfx, error=False) is not None:
            for loop in loops:
                s.unroll(f"{fn}{sfx}:{loop}")
    return s


def mx_gemm_q[
    Ty, M, N, K, NB, TRANS
](
    Xq: "Int(Ty.elem_bits)[M, K]",
    Xs: "e8m0[M, NB]",
    Wq: "Int(Ty.elem_bits)[K if TRANS else N, N if TRANS else K]",
    Ws: "e8m0[NB if TRANS else N, N if TRANS else NB]",
    bias: "float32[N]",
) -> "float32[M, N]":
    # Z = X @ W.T + bias (W is [N, K]), or X @ W + bias when TRANS (W is [K, N])
    qx: Ty[M, NB]
    qw: Ty[N, NB]
    _mx_pack_rows[Ty, M, K, NB, 0](Xq, Xs, qx)
    _mx_pack_rows[Ty, N, K, NB, TRANS](Wq, Ws, qw)
    Z: float32[M, N]
    _mx_dot_rows[Ty, M, N, NB](qx, qw, bias, Z)
    return Z


def mx_gemm_wq[
    Ty, M, N, K, NB, TRANS
](
    X: "float32[M, K]",
    Wq: "Int(Ty.elem_bits)[K if TRANS else N, N if TRANS else K]",
    Ws: "e8m0[NB if TRANS else N, N if TRANS else NB]",
    bias: "float32[N]",
) -> "float32[M, N]":
    qx: Ty[M, NB]
    qw: Ty[N, NB]
    _mx_quantize_rows[Ty, M, K, NB, 0](X, qx)
    _mx_pack_rows[Ty, N, K, NB, TRANS](Wq, Ws, qw)
    Z: float32[M, N]
    _mx_dot_rows[Ty, M, N, NB](qx, qw, bias, Z)
    return Z


def mx_gemm_ff[
    Ty, M, N, K, NB, TRANS
](
    X: "float32[M, K]",
    W: "float32[K if TRANS else N, N if TRANS else K]",
    bias: "float32[N]",
) -> "float32[M, N]":
    qx: Ty[M, NB]
    qw: Ty[N, NB]
    _mx_quantize_rows[Ty, M, K, NB, 0](X, qx)
    _mx_quantize_rows[Ty, N, K, NB, TRANS](W, qw)
    Z: float32[M, N]
    _mx_dot_rows[Ty, M, N, NB](qx, qw, bias, Z)
    return Z


def mx_matmul_q[
    Ty, M, N, K, NB
](
    Aq: "Int(Ty.elem_bits)[M, K]",
    As: "e8m0[M, NB]",
    Bq: "Int(Ty.elem_bits)[K, N]",
    Bs: "e8m0[NB, N]",
) -> "float32[M, N]":
    zero: float32[N] = 0.0
    return mx_gemm_q[Ty, M, N, K, NB, 1](Aq, As, Bq, Bs, zero)


def mx_matmul_wq[
    Ty, M, N, K, NB
](
    A: "float32[M, K]", Bq: "Int(Ty.elem_bits)[K, N]", Bs: "e8m0[NB, N]"
) -> "float32[M, N]":
    zero: float32[N] = 0.0
    return mx_gemm_wq[Ty, M, N, K, NB, 1](A, Bq, Bs, zero)


def mx_matmul_ff[
    Ty, M, N, K, NB
](A: "float32[M, K]", B: "float32[K, N]") -> "float32[M, N]":
    zero: float32[N] = 0.0
    return mx_gemm_ff[Ty, M, N, K, NB, 1](A, B, zero)


def schedule_mx_gemm(s, depth=16):
    """Streams the quantized X blocks into the block dot (HLS dataflow)."""
    mode = s.top_func_name.rsplit("_", 1)[1]  # q / wq / ff
    gemm = f"mx_gemm_{mode}"  # also the callee of mx_matmul_{mode}
    schedule_mx_block_dot(s)
    if mode != "q":  # mx_gemm_q takes pre-quantized operands
        schedule_mx_quantize_block_f32(s)
    s.pipeline("_mx_dot_rows:j")
    s.to(MockBuffer(gemm, "qx"), "_mx_dot_rows", depth=depth)
    s.dataflow(gemm)
    return s


def partition_block_lanes(s, buf, length, BS, dim=2):
    """Partitions `buf` along `dim` (of `length` elements) into one bank per
    block lane, so the BS elements of a block are read or written in one cycle."""
    if length <= BS:
        s.partition(buf, dim=dim)
    else:
        s.partition(buf, partition_type=2, dim=dim, factor=BS)


def mx_unpack_f32[
    R, C
](A0: "UInt(512)[R, C // 32]", A1: "UInt(512)[R, C // 32]", X: "float32[R, C]"):
    # DRAM -> chip at one 32-float block per cycle: each block arrives as two
    # 512b words, one per port (A0: floats 0-15, A1: 16-31). C % 32 == 0;
    # mxint8.mx_pack_f32_blocks builds A0/A1 on the host. X is an output
    # argument, so the consumer's banking of it reaches this function too.
    for r, i in allo.grid(R, C // 32):
        lo: UInt(512) = A0[r, i]
        hi: UInt(512) = A1[r, i]
        with allo.meta_for(16) as e:
            lb: int32 = lo[e * 32 : (e + 1) * 32]
            hb: int32 = hi[e * 32 : (e + 1) * 32]
            X[r, i * 32 + e] = lb.bitcast()
            X[r, i * 32 + 16 + e] = hb.bitcast()


def schedule_mx_unpack_f32(s):
    # X's block-lane banks come from the consuming layer's schedule (its
    # partition of the same buffer propagates here); a second one would clash
    s.pipeline(f"{s.top_func_name}:i")
    return s


# --- MX layers: operands are Ty arrays (one scale | elems word per block) ------
# Quantize activations on chip with mx_quantize_rows / mx_quantize_channels, and
# weights once on the host with mxint8.mx_pack_np.


def schedule_mx_quantize_rows(s):
    """One block per cycle: X is banked by block lane."""
    Ty, _, _, K = s.inst_list[:4]
    schedule_mx(s)
    top = s.top_func_name
    partition_block_lanes(s, MockBuffer(top, "X"), K, Ty.block_size)
    s.pipeline(f"{top}:b")
    return s


def mx_quantize_channels[
    Ty, TyX, B, C, H, W
](X: "TyX[B, C, H, W]") -> "Ty[B, (C + Ty.block_size - 1) // Ty.block_size * H * W]":
    # float32 NCHW -> MX blocks along the channels (zero-padded to whole
    # blocks), one row per image: block (cb, h, w) at column (cb * H + h) * W + w
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (C + BS - 1) // BS
    Q: Ty[B, NB * H * W]
    for n, cb, h, w in allo.grid(B, NB, H, W):
        blk: float32[BS]
        for e in range(BS):
            fx: float32 = 0.0
            if cb * BS + e < C:
                fx = X[n, cb * BS + e, h, w]
            blk[e] = fx
        Q[n, (cb * H + h) * W + w] = mx_quantize_block_f32[Ty, BS](blk)
    return Q


def schedule_mx_quantize_channels(s):
    """One block per cycle: X is banked by block lane along the channels."""
    Ty, _, _, C = s.inst_list[:4]
    top = s.top_func_name
    schedule_mx(s)
    partition_block_lanes(s, MockBuffer(top, "X"), C, Ty.block_size)
    s.pipeline(f"{top}:w")
    return s


MX_ACC_LATENCY = 6


def mx_matmul[
    Ty, M, N, K
](
    Aq: "Ty[M, (K + Ty.block_size - 1) // Ty.block_size]",
    Bq: "Ty[N, (K + Ty.block_size - 1) // Ty.block_size]",
) -> "float32[M, N]":
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    P: ConstExpr[int32] = 4 - 3 * (N >= MX_ACC_LATENCY)
    C: float32[M, N]
    acc: float32[N] = 0.0
    aw: Ty[P]
    for m in range(M):
        for g, n in allo.grid((NB + P - 1) // P, N):
            d: float32 = 0.0
            with allo.meta_for(P) as j:
                if g * P + j < NB:
                    if n == 0:
                        aw[j] = Aq[m, g * P + j]
                    d += mx_block_dot[Ty, BS](aw[j], Bq[n, g * P + j])
            prev: float32 = 0.0
            if g > 0:
                prev = acc[n]
            total: float32 = prev + d
            acc[n] = total
            if g == (NB - 1) // P:
                C[m, n] = total
    return C


def pipeline_mx_matmul(s, func, N):
    s.pipeline(f"{func}:n", initiation_interval=1 if N >= MX_ACC_LATENCY else 4)


def schedule_mx_matmul(s):
    schedule_mx(s)
    pipeline_mx_matmul(s, s.top_func_name, s.inst_list[2])
    return s


def bias_add[Ty, M, N](X: "float32[M, N]", bias: "Ty[N]") -> "Ty[M, N]":
    # X + bias per row; X is read once, row by row
    Z: Ty[M, N]
    for m, n in allo.grid(M, N):
        Z[m, n] = X[m, n] + bias[n]
    return Z


def mx_linear2d[
    Ty, TyX, M, N, K
](
    X: "TyX[M, K]",
    Wq: "Ty[N, (K + Ty.block_size - 1) // Ty.block_size]",
    bias: "TyX[N]",
) -> "TyX[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html with MX
    # arithmetic: X is quantized here, Wq on the host (mxint8.mx_pack_np)
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
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Z: TyX[B, L, M]
    for b in range(B):
        X_b: TyX[L, D]
        for i in range(L):
            for k in range(D):
                X_b[i, k] = X[b, i, k]
        Z_b = mx_linear2d[Ty, TyX, L, M, D](X_b, Wq, bias)
        for i in range(L):
            for j in range(M):
                Z[b, i, j] = Z_b[i, j]
    return Z


def schedule_mx_linear3d(s):
    schedule_mx(s)
    s.pipeline("mx_linear3d:k")
    s.pipeline("mx_linear3d:j")
    return s


def mx_conv_windows[
    Ty, B, C, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    Xq: "Ty[B, (C + Ty.block_size - 1) // Ty.block_size * H * W]",
) -> "Ty[B * Oh * Ow, (C + Ty.block_size - 1) // Ty.block_size * Kh * Kw]":
    # MX image rows -> one row per output pixel holding its receptive field,
    # blocks in (channel block, kh, kw) order and zero blocks for the padding,
    # written in order, one block per iteration
    NB: ConstExpr[int32] = (C + Ty.block_size - 1) // Ty.block_size
    Xw: Ty[B * Oh * Ow, NB * Kh * Kw]
    for n, oh, ow, cb, kh, kw in allo.grid(B, Oh, Ow, NB, Kh, Kw):
        ih: int32 = oh * Sh + kh - Pd0
        iw: int32 = ow * Sw + kw - Pd1
        blk: Ty = 0
        if ih >= 0 and ih < H and iw >= 0 and iw < W:
            blk = Xq[n, (cb * H + ih) * W + iw]
        Xw[(n * Oh + oh) * Ow + ow, (cb * Kh + kh) * Kw + kw] = blk
    return Xw


def rows_to_nchw[Ty, B, C, H, W](Z: "Ty[B * H * W, C]") -> "Ty[B, C, H, W]":
    # one row per pixel (row-major, read in order) -> NCHW
    Y: Ty[B, C, H, W]
    for n, h, w, c in allo.grid(B, H, W, C):
        Y[n, c, h, w] = Z[(n * H + h) * W + w, c]
    return Y


def mx_conv2d[
    Ty, TyX, B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    X: "TyX[B, Cin, H, W]",
    Wq: "Ty[Cout, (Cin + Ty.block_size - 1) // Ty.block_size * Kh * Kw]",
    bias: "TyX[Cout]",
) -> "TyX[B, Cout, Oh, Ow]":
    # nn.conv2d with MX arithmetic, blocks along the input channels: X is
    # quantized here, Wq on the host (mxint8.mx_pack_channels_np). Each output
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
