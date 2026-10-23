# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation, chained-comparison, no-value-for-parameter

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


def mx_matmul[Ty, M, N, K](A: "float32[M, K]", B: "float32[N, K]") -> "float32[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
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


def schedule_mx_matmul(s):
    schedule_mx_quantize_block_f32(s)
    schedule_mx_block_dot(s)
    s.unroll("mx_matmul:e")
    s.pipeline("mx_matmul:b")
    s.pipeline("mx_matmul:j")
    return s


def mx_linear2d_ref[
    Ty, M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    Z = mx_matmul[Ty, M, N, K](X, W)
    for i in range(M):
        for j in range(N):
            Z[i, j] = Z[i, j] + bias[j]
    return Z


def _mx_quantize_rows[
    Ty, R, K, NB, TRANS
](X: "float32[K if TRANS else R, R if TRANS else K]", Q: "Ty[R, NB]"):
    # X is [R, K], or [K, R] when TRANS; blocks run along K either way
    BS: ConstExpr[int32] = Ty.block_size
    for r, b in allo.grid(R, NB):
        blk: float32[BS]
        for e in range(BS):
            with allo.meta_if(TRANS):
                blk[e] = X[b * BS + e, r]
            with allo.meta_else():
                blk[e] = X[r, b * BS + e]
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


def mx_quantize[Ty, R, K](X: "float32[R, K]") -> "Ty[R, K // Ty.block_size]":
    # float32 [R, K] -> MX matrix: one packed scale | elems word per block along K
    Q: "Ty[R, K // Ty.block_size]"
    _mx_quantize_rows[Ty, R, K, K // Ty.block_size, 0](X, Q)
    return Q


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


def mx_dequantize[
    Ty, R, K
](Q: "Ty[R, K // Ty.block_size]") -> "float32[R, K]":
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


def mx_unpack_nhwc[
    B, C, H, W
](
    A0: "UInt(512)[B * H, W * C // 32]",
    A1: "UInt(512)[B * H, W * C // 32]",
    X: "float32[B, C, H, W]",
):
    # DRAM -> chip feature map at one 32-float block per cycle: the host sends
    # it NHWC-ordered (mxint8.mx_pack_nhwc_blocks), so a block is the same BS
    # (pixel, channel) run mx_conv2d's row-block stage reads; its banking of
    # this buffer propagates here (X is an output argument). W * C % 32 == 0.
    for n, h, i in allo.grid(B, H, W * C // 32):
        lo: UInt(512) = A0[n * H + h, i]
        hi: UInt(512) = A1[n * H + h, i]
        with allo.meta_for(16) as e:
            lb: int32 = lo[e * 32 : (e + 1) * 32]
            hb: int32 = hi[e * 32 : (e + 1) * 32]
            X[n, (i * 32 + e) % C, h, (i * 32 + e) // C] = lb.bitcast()
            X[n, (i * 32 + 16 + e) % C, h, (i * 32 + 16 + e) // C] = hb.bitcast()


def schedule_mx_unpack_nhwc(s):
    # banked by the consuming conv (see schedule_mx_unpack_f32)
    s.pipeline(f"{s.top_func_name}:i")
    return s


def _mx_lin_wquant[
    Ty, N, K, NB
](W: "float32[N, K]", WS: "uint8[N, NB]", WE: "UInt(Ty.payload_bits)[N, NB]"):
    # one weight block per iteration (zero past K), quantized once: weights
    # stay put. W is partitioned by block lane, so a block is one read per bank.
    BS: ConstExpr[int32] = Ty.block_size
    for wn, b in allo.grid(N, NB):
        blk: float32[BS]
        with allo.meta_for(BS) as e:
            fw: float32 = 0.0
            if b * BS + e < K:
                fw = W[wn, b * BS + e]
            blk[e] = fw
        w: Ty = mx_quantize_block_f32[Ty, BS](blk)
        WS[wn, b] = w[Ty.bits - 8 : Ty.bits]
        WE[wn, b] = w[0 : Ty.payload_bits]


def _mx_lin_xpack[M, K, BS, NB](X: "float32[M, K]", P: "UInt(32 * BS)[M, NB]"):
    # one token per row block: BS float32 bit patterns in one word. X is
    # partitioned by block lane, so a block is one read per bank.
    for m, b in allo.grid(M, NB):
        word: UInt(32 * BS) = 0
        with allo.meta_for(BS) as e:
            # not `v`: v<N> names collide with HLS emitter value names
            fx: float32 = 0.0
            if b * BS + e < K:
                fx = X[m, b * BS + e]
            word[e * 32 : (e + 1) * 32] = fx.bitcast()
        P[m, b] = word


def _mx_lin_dot[
    Ty, M, N, NB
](
    XS: "uint8[M, NB]",
    XE: "UInt(Ty.payload_bits)[M, NB]",
    WS: "uint8[N, NB]",
    WE: "UInt(Ty.payload_bits)[N, NB]",
    bias: "float32[N]",
    Z: "float32[M, N]",
):
    # same loop order as _mx_conv_dot: one block dot per iteration, each X block
    # read once (n == 0) and reused for every output column
    BS: ConstExpr[int32] = Ty.block_size
    acc: float32[N] = 0.0
    xw: Ty = 0
    for m in range(M):
        for b, n in allo.grid(NB, N):
            if n == 0:
                xw[0 : Ty.payload_bits] = XE[m, b]
                xw[Ty.bits - 8 : Ty.bits] = XS[m, b]
            ww: Ty = 0
            ww[0 : Ty.payload_bits] = WE[n, b]
            ww[Ty.bits - 8 : Ty.bits] = WS[n, b]
            d: float32 = mx_block_dot[Ty, BS](xw, ww)
            prev: float32 = 0.0
            if b > 0:
                prev = acc[n]
            total: float32 = prev + d
            acc[n] = total
            if b == NB - 1:
                Z[m, n] = total + bias[n]


def mx_linear2d[
    Ty, M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html, with MX
    # arithmetic; K is zero-padded up to whole blocks. Stages pass one MX block
    # per element of P / XS / XE, which schedule_mx_linear2d streams.
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    PB: ConstExpr[int32] = Ty.payload_bits
    PW: ConstExpr[int32] = 32 * BS
    WS: uint8[N, NB]
    WE: UInt(PB)[N, NB]
    _mx_lin_wquant[Ty, N, K, NB](W, WS, WE)
    P: UInt(PW)[M, NB]
    _mx_lin_xpack[M, K, BS, NB](X, P)
    XS: uint8[M, NB]
    XE: UInt(PB)[M, NB]
    _mx_quant_blocks[Ty, M, NB](P, XS, XE)
    Z: float32[M, N]
    _mx_lin_dot[Ty, M, N, NB](XS, XE, WS, WE, bias, Z)
    return Z


def schedule_mx_linear2d(s, depth=4):
    """X row pack -> block quantize -> block dot as an HLS dataflow, one MX
    block per stream token and one block dot per cycle. Registered in
    KERNEL2SCHEDULE, so s.compose(mx_linear2d, instantiate=..., id=...) applies it.
    """
    Ty, _, _, K = s.inst_list[:4]
    NB = (K + Ty.block_size - 1) // Ty.block_size
    top = s.top_func_name
    schedule_mx(s)
    # one bank per block lane: X and W blocks are read whole, one per cycle
    for arr in ("X", "W"):
        partition_block_lanes(s, MockBuffer(top, arr), K, Ty.block_size)
    s.pipeline("_mx_lin_wquant:b")
    s.pipeline("_mx_lin_xpack:b")
    s.pipeline("_mx_quant_blocks:b")
    s.pipeline("_mx_lin_dot:n")
    for buf, consumer in (
        ("P", "_mx_quant_blocks"),
        ("XS", "_mx_lin_dot"),
        ("XE", "_mx_lin_dot"),
    ):
        s.to(MockBuffer(top, buf), consumer, depth=max(depth, NB))
    s.dataflow(top)
    return s


def mx_linear3d[
    Ty, B, L, D, M
](X: "float32[B, L, D]", W: "float32[M, D]", bias: "float32[M]") -> "float32[B, L, M]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Z: float32[B, L, M]
    for b in range(B):
        X_b: float32[L, D]
        for i in range(L):
            for k in range(D):
                X_b[i, k] = X[b, i, k]
        Z_b = mx_linear2d[Ty, L, M, D](X_b, W, bias)
        for i in range(L):
            for j in range(M):
                Z[b, i, j] = Z_b[i, j]
    return Z


def schedule_mx_linear3d(s):
    # mx_linear2d's dataflow schedule needs its own top; compose it for that
    schedule_mx(s)
    s.pipeline("mx_linear3d:k")
    s.pipeline("mx_linear3d:j")
    return s


def _mx_conv_wquant[
    Ty, Cout, Cin, Kh, Kw, CKK, NBc
](
    kernel: "float32[Cout, Cin, Kh, Kw]",
    WS: "uint8[Cout, NBc]",
    WE: "UInt(Ty.payload_bits)[Cout, NBc]",
):
    # flatten to [Cout, CKK] (zero-padded) and quantize once: weights stay put
    BS: ConstExpr[int32] = Ty.block_size
    # flatten and zero-pad in one pass (no separate zero-fill of kflat)
    kflat: float32[Cout, CKK]
    for fo, j in allo.grid(Cout, CKK):
        fk: float32 = 0.0
        if j < Cin * Kh * Kw:
            fk = kernel[fo, j // (Kh * Kw), (j // Kw) % Kh, j % Kw]
        kflat[fo, j] = fk
    for fo1, b in allo.grid(Cout, NBc):
        blk: float32[BS]
        with allo.meta_for(BS) as e:
            blk[e] = kflat[fo1, b * BS + e]
        w: Ty = mx_quantize_block_f32[Ty, BS](blk)
        WS[fo1, b] = w[Ty.bits - 8 : Ty.bits]
        WE[fo1, b] = w[0 : Ty.payload_bits]


def _mx_conv_rowblocks[
    B, Cin, H, W, BS, WB
](inp: "float32[B, Cin, H, W]", Q: "UInt(32 * BS)[B * H, WB]"):
    # input rows as tokens of BS consecutive NHWC values (pixel-major, channel-
    # minor), one block per iteration. inp is banked by channel and by pixel
    # mod BS // Cin, so a block's BS reads hit BS different banks.
    for n, h, wb in allo.grid(B, H, WB):
        word: UInt(32 * BS) = 0
        with allo.meta_for(BS) as e:
            fr: float32 = 0.0
            if wb * BS + e < W * Cin:
                fr = inp[n, (wb * BS + e) % Cin, h, (wb * BS + e) // Cin]
            word[e * 32 : (e + 1) * 32] = fr.bitcast()
        Q[n * H + h, wb] = word


def _mx_conv_lb_im2col[
    B, Cin, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1, BS, NBc, WB
](Q: "UInt(32 * BS)[B * H, WB]", P: "UInt(32 * BS)[B * Oh * Ow, NBc]"):
    # line buffer: input rows arrive one block per iteration into a Kh-row
    # ring; once the last row an output row needs is in, a Kh x Kw x Cin window
    # slides along it one input column at a time and emits each patch as NBc
    # block tokens, in (cin, kh, kw) order like the flattened kernel.
    LB: float32[Kh, W, Cin]
    win: float32[Kh, Kw, Cin]
    for n in range(B):
        for r in range(H + Pd0):
            if r < H:
                for wb in range(WB):
                    word: UInt(32 * BS) = Q[n * H + r, wb]
                    with allo.meta_for(BS) as e:
                        if wb * BS + e < W * Cin:
                            bits: int32 = word[e * 32 : (e + 1) * 32]
                            LB[r % Kh, (wb * BS + e) // Cin, (wb * BS + e) % Cin] = (
                                bits.bitcast()
                            )
            t: int32 = r - (Kh - 1 - Pd0)
            if t >= 0 and t % Sh == 0 and t // Sh < Oh:
                oh: int32 = t // Sh
                with allo.meta_for(Kh) as kh0:
                    with allo.meta_for(Kw) as kw0:
                        with allo.meta_for(Cin) as ci0:
                            win[kh0, kw0, ci0] = 0.0
                for c in range(W + Pd1):
                    # shift the window one column left, then load column c
                    with allo.meta_for(Kh) as kh:
                        with allo.meta_for(Kw - 1) as kw:
                            with allo.meta_for(Cin) as ci:
                                win[kh, kw, ci] = win[kh, kw + 1, ci]
                        with allo.meta_for(Cin) as ci1:
                            fc: float32 = 0.0
                            if (
                                c < W
                                and oh * Sh + kh - Pd0 >= 0
                                and oh * Sh + kh - Pd0 < H
                            ):
                                fc = LB[(oh * Sh + kh - Pd0) % Kh, c, ci1]
                            win[kh, Kw - 1, ci1] = fc
                    u: int32 = c - (Kw - 1 - Pd1)
                    if u >= 0 and u % Sw == 0 and u // Sw < Ow:
                        with allo.meta_for(NBc) as b:
                            word1: UInt(32 * BS) = 0
                            with allo.meta_for(BS) as e1:
                                fp: float32 = 0.0
                                with allo.meta_if(b * BS + e1 < Cin * Kh * Kw):
                                    fp = win[
                                        ((b * BS + e1) // Kw) % Kh,
                                        (b * BS + e1) % Kw,
                                        (b * BS + e1) // (Kh * Kw),
                                    ]
                                word1[e1 * 32 : (e1 + 1) * 32] = fp.bitcast()
                            P[(n * Oh + oh) * Ow + u // Sw, b] = word1


def _mx_quant_blocks[
    Ty, R, NBc
](
    P: "UInt(32 * Ty.block_size)[R, NBc]",
    XS: "uint8[R, NBc]",
    XE: "UInt(Ty.payload_bits)[R, NBc]",
):
    # one patch block in, its (scale, elements) bundle out
    BS: ConstExpr[int32] = Ty.block_size
    for r, b in allo.grid(R, NBc):
        word: UInt(32 * BS) = P[r, b]
        blk: float32[BS]
        with allo.meta_for(BS) as e:
            bits: int32 = word[e * 32 : (e + 1) * 32]
            blk[e] = bits.bitcast()
        w: Ty = mx_quantize_block_f32[Ty, BS](blk)
        XS[r, b] = w[Ty.bits - 8 : Ty.bits]
        XE[r, b] = w[0 : Ty.payload_bits]


def _mx_conv_dot[
    Ty, B, Cout, Oh, Ow, NBc
](
    XS: "uint8[B * Oh * Ow, NBc]",
    XE: "UInt(Ty.payload_bits)[B * Oh * Ow, NBc]",
    WS: "uint8[Cout, NBc]",
    WE: "UInt(Ty.payload_bits)[Cout, NBc]",
    bias: "float32[Cout]",
    Z: "float32[B, Cout, Oh, Ow]",
):
    # one block dot per iteration: each patch block is read once (cout == 0) and
    # reused for every output channel; acc[cout] is revisited every Cout iterations.
    # Slice bounds stay Ty attributes: a ConstExpr bound mis-slices here.
    BS: ConstExpr[int32] = Ty.block_size
    acc: float32[Cout] = 0.0
    xw: Ty = 0
    for n, oh, ow in allo.grid(B, Oh, Ow):
        for b, cout in allo.grid(NBc, Cout):
            if cout == 0:
                xw[0 : Ty.payload_bits] = XE[(n * Oh + oh) * Ow + ow, b]
                xw[Ty.bits - 8 : Ty.bits] = XS[(n * Oh + oh) * Ow + ow, b]
            ww: Ty = 0
            ww[0 : Ty.payload_bits] = WE[cout, b]
            ww[Ty.bits - 8 : Ty.bits] = WS[cout, b]
            d: float32 = mx_block_dot[Ty, BS](xw, ww)
            prev: float32 = 0.0
            if b > 0:
                prev = acc[cout]
            total: float32 = prev + d
            acc[cout] = total
            if b == NBc - 1:
                Z[n, cout, oh, ow] = total + bias[cout]


def mx_conv2d[
    Ty, B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    inp: "float32[B, Cin, H, W]",
    kernel: "float32[Cout, Cin, Kh, Kw]",
    bias: "float32[Cout]",
) -> "float32[B, Cout, Oh, Ow]":
    # nn.conv2d with MX arithmetic: each im2col patch dotted with the flat kernel;
    # Cin * Kh * Kw is zero-padded up to a whole number of blocks. The stages
    # pass one MX block per element of P / XS / XE, which schedule_mx_conv2d
    # turns into streams of an HLS dataflow.
    BS: ConstExpr[int32] = Ty.block_size
    NBc: ConstExpr[int32] = (Cin * Kh * Kw + BS - 1) // BS
    CKK: ConstExpr[int32] = NBc * BS
    R: ConstExpr[int32] = B * Oh * Ow
    PB: ConstExpr[int32] = Ty.payload_bits
    PW: ConstExpr[int32] = 32 * BS
    WS: uint8[Cout, NBc]
    WE: UInt(PB)[Cout, NBc]
    _mx_conv_wquant[Ty, Cout, Cin, Kh, Kw, CKK, NBc](kernel, WS, WE)
    WB: ConstExpr[int32] = (W * Cin + BS - 1) // BS
    Q: UInt(PW)[B * H, WB]
    _mx_conv_rowblocks[B, Cin, H, W, BS, WB](inp, Q)
    P: UInt(PW)[R, NBc]
    _mx_conv_lb_im2col[B, Cin, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1, BS, NBc, WB](
        Q, P
    )
    XS: uint8[R, NBc]
    XE: UInt(PB)[R, NBc]
    _mx_quant_blocks[Ty, R, NBc](P, XS, XE)
    Z: float32[B, Cout, Oh, Ow]
    _mx_conv_dot[Ty, B, Cout, Oh, Ow, NBc](XS, XE, WS, WE, bias, Z)
    return Z


def schedule_conv_lb_input(s, top, Cin, W, BS, depth):
    """Banks and pipelines the row-block + line-buffer im2col input path
    (_mx_conv_rowblocks -> Q stream -> _mx_conv_lb_im2col), shared by
    mx_conv2d and its float32 counterpart: one input block per cycle."""
    F = max(1, BS // Cin)  # pixels per block
    for buf, (cdim, wdim) in ((MockBuffer(top, "inp"), (2, 4)), (MockBuffer("_mx_conv_lb_im2col", "LB"), (3, 2))):
        partition_block_lanes(s, buf, Cin, BS, dim=cdim)
        if F > 1:
            s.partition(buf, partition_type=2, dim=wdim, factor=F)
    s.partition(MockBuffer("_mx_conv_lb_im2col", "LB"), dim=1)
    s.partition(MockBuffer("_mx_conv_lb_im2col", "win"))
    s.pipeline("_mx_conv_rowblocks:wb")
    s.pipeline("_mx_conv_lb_im2col:wb")
    s.pipeline("_mx_conv_lb_im2col:c")
    s.to(MockBuffer(top, "Q"), "_mx_conv_lb_im2col", depth=depth)


def schedule_mx_conv2d(s, depth=4):
    """row blocks -> line-buffer im2col -> patch quantize -> block dot as an
    HLS dataflow, one MX block per stream token and one block dot per cycle. Registered in
    KERNEL2SCHEDULE, so s.compose(mx_conv2d, instantiate=..., id=...) applies it.
    """
    Ty, _, Cin, _, _, W, Kh, Kw = s.inst_list[:8]
    NBc = (Cin * Kh * Kw + Ty.block_size - 1) // Ty.block_size
    top = s.top_func_name
    schedule_mx(s)
    # weights: flatten one per cycle, then quantize one block per cycle
    partition_block_lanes(
        s, MockBuffer("_mx_conv_wquant", "kflat"), NBc * Ty.block_size, Ty.block_size
    )
    s.pipeline("_mx_conv_wquant:j")
    s.pipeline("_mx_conv_wquant:b")
    schedule_conv_lb_input(s, top, Cin, W, Ty.block_size, max(depth, NBc))
    s.pipeline("_mx_quant_blocks:b")
    s.pipeline("_mx_conv_dot:cout")
    for buf, consumer in (
        ("P", "_mx_quant_blocks"),
        ("XS", "_mx_conv_dot"),
        ("XE", "_mx_conv_dot"),
    ):
        s.to(MockBuffer(top, buf), consumer, depth=max(depth, NBc))
    s.dataflow(top)
    return s


def _mx_df_widths(Ty):
    """Bundle lane bits, float32s per 512b word, and words per block for make_mx_*."""
    FPW = 512 // 32
    BS = Ty.block_size
    assert BS % FPW == 0, f"block_size={BS} must be a multiple of {FPW}"
    return max(Ty.elem_bits, Ty.scale_bits), FPW, BS // FPW


def make_mx_matmul_dataflow(Ty, M, N, K, P, depth=4):
    """Dataflow MX Z = A @ B.T with on-chip quantization."""
    import allo.dataflow as df

    BS = Ty.block_size
    NB = K // BS
    LB, FPW, WPB = _mx_df_widths(Ty)
    assert K % BS == 0, f"K={K} must be a multiple of block_size={BS}"
    assert N % P == 0, f"N={N} must be a multiple of P={P}"
    NJ = N // P
    FW = WPB * NB  # 512-bit words per float32 row

    @df.region()
    def top(A: "UInt(512)[M * FW]", B: "UInt(512)[N * FW]", Z: "float32[M * N]"):
        pipe_a_q: Stream["UInt(LB)[BS + 1]", depth]
        pipe_b_q: Stream["UInt(LB)[BS + 1]", depth]

        @df.kernel(mapping=[1], args=[A, B])
        def quantize_ab(local_A: "UInt(512)[M * FW]", local_B: "UInt(512)[N * FW]"):
            for tb in range(N * NB):
                blk: float32[BS]
                for hb in range(WPB):
                    fword: UInt(512) = local_B[tb * WPB + hb]
                    for eb in range(FPW):
                        bits: int32 = fword[eb * 32 : (eb + 1) * 32]
                        blk[hb * FPW + eb] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: UInt(LB)[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for kb in range(BS):
                    bundle[kb + 1] = word[kb * Ty.elem_bits : (kb + 1) * Ty.elem_bits]
                pipe_b_q.put(bundle)

            for ta in range(M * NB):
                blk: float32[BS]
                for ha in range(WPB):
                    fword: UInt(512) = local_A[ta * WPB + ha]
                    for ea in range(FPW):
                        bits: int32 = fword[ea * 32 : (ea + 1) * 32]
                        blk[ha * FPW + ea] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: UInt(LB)[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for ka in range(BS):
                    bundle[ka + 1] = word[ka * Ty.elem_bits : (ka + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle)

        @df.kernel(mapping=[1], args=[Z])
        def dot_product_stage(local_Z: "float32[M * N]"):
            b_buf: UInt(LB)[N, NB, BS + 1]
            for jd in range(N):
                for bd in range(NB):
                    tmp: UInt(LB)[BS + 1] = pipe_b_q.get()
                    for kd in range(BS + 1):
                        b_buf[jd, bd, kd] = tmp[kd]

            for i in range(M):
                acc: float32[N]
                for j0 in range(N):
                    acc[j0] = 0.0
                word_a: Ty = 0
                for b in range(NB):
                    for jj in range(NJ):
                        if jj == 0:
                            a_blk: UInt(LB)[BS + 1] = pipe_a_q.get()
                            block_a: UInt(Ty.elem_bits)[BS]
                            for ka in range(BS):
                                block_a[ka] = a_blk[ka + 1]
                            word_a = _mx_pack_word[Ty, BS](a_blk[0], block_a)
                        for p in range(P):
                            j: int32 = jj * P + p
                            scale_b: uint8 = b_buf[j, b, 0]
                            block_b: UInt(Ty.elem_bits)[BS]
                            for k in range(BS):
                                block_b[k] = b_buf[j, b, k + 1]
                            word_b: Ty = _mx_pack_word[Ty, BS](scale_b, block_b)
                            acc[j] += mx_block_dot[Ty, BS](word_a, word_b)
                for jz in range(N):
                    local_Z[i * N + jz] = acc[jz]

    s = df.customize(top, opt_default=False)
    schedule_mx_quantize_block_f32(s)
    schedule_mx_block_dot(s)
    s.pipeline("quantize_ab_0:tb")
    s.unroll("quantize_ab_0:hb")
    s.unroll("quantize_ab_0:eb")
    s.unroll("quantize_ab_0:kb")
    s.pipeline("quantize_ab_0:ta")
    s.unroll("quantize_ab_0:ha")
    s.unroll("quantize_ab_0:ea")
    s.unroll("quantize_ab_0:ka")
    s.unroll("dot_product_stage_0:k")
    s.unroll("dot_product_stage_0:ka")
    s.unroll("dot_product_stage_0:p")
    s.pipeline("dot_product_stage_0:jj")
    return s


def make_mx_linear2d_dataflow(Ty, M, N, K, P, depth=4):
    """Dataflow MX Z = X @ W.T + bias with on-chip quantization."""
    import allo.dataflow as df

    BS = Ty.block_size
    NB = K // BS
    LB, FPW, WPB = _mx_df_widths(Ty)
    assert K % BS == 0, f"K={K} must be a multiple of block_size={BS}"
    assert N % P == 0, f"N={N} must be a multiple of P={P}"
    NJ = N // P
    FW = WPB * NB  # 512-bit words per float32 row

    @df.region()
    def top(
        X: "UInt(512)[M * FW]",
        W: "UInt(512)[N * FW]",
        bias: "float32[N]",
        Z: "float32[M * N]",
    ):
        pipe_a_q: Stream["UInt(LB)[BS + 1]", depth]
        pipe_b_q: Stream["UInt(LB)[BS + 1]", depth]

        @df.kernel(mapping=[1], args=[X, W])
        def quantize_ab(local_X: "UInt(512)[M * FW]", local_W: "UInt(512)[N * FW]"):
            for tb in range(N * NB):
                blk: float32[BS]
                for hb in range(WPB):
                    fword: UInt(512) = local_W[tb * WPB + hb]
                    for eb in range(FPW):
                        bits: int32 = fword[eb * 32 : (eb + 1) * 32]
                        blk[hb * FPW + eb] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: UInt(LB)[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for kb in range(BS):
                    bundle[kb + 1] = word[kb * Ty.elem_bits : (kb + 1) * Ty.elem_bits]
                pipe_b_q.put(bundle)

            for ta in range(M * NB):
                blk: float32[BS]
                for ha in range(WPB):
                    fword: UInt(512) = local_X[ta * WPB + ha]
                    for ea in range(FPW):
                        bits: int32 = fword[ea * 32 : (ea + 1) * 32]
                        blk[ha * FPW + ea] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: UInt(LB)[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for ka in range(BS):
                    bundle[ka + 1] = word[ka * Ty.elem_bits : (ka + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle)

        @df.kernel(mapping=[1], args=[bias, Z])
        def dot_product_stage(local_bias: "float32[N]", local_Z: "float32[M * N]"):
            b_buf: UInt(LB)[N, NB, BS + 1]
            for jd in range(N):
                for bd in range(NB):
                    tmp: UInt(LB)[BS + 1] = pipe_b_q.get()
                    for kd in range(BS + 1):
                        b_buf[jd, bd, kd] = tmp[kd]

            for i in range(M):
                acc: float32[N]
                for j0 in range(N):
                    acc[j0] = 0.0
                word_a: Ty = 0
                for b in range(NB):
                    for jj in range(NJ):
                        if jj == 0:
                            a_blk: UInt(LB)[BS + 1] = pipe_a_q.get()
                            block_a: UInt(Ty.elem_bits)[BS]
                            for ka in range(BS):
                                block_a[ka] = a_blk[ka + 1]
                            word_a = _mx_pack_word[Ty, BS](a_blk[0], block_a)
                        for p in range(P):
                            j: int32 = jj * P + p
                            scale_b: uint8 = b_buf[j, b, 0]
                            block_b: UInt(Ty.elem_bits)[BS]
                            for k in range(BS):
                                block_b[k] = b_buf[j, b, k + 1]
                            word_b: Ty = _mx_pack_word[Ty, BS](scale_b, block_b)
                            acc[j] += mx_block_dot[Ty, BS](word_a, word_b)
                for jz in range(N):
                    local_Z[i * N + jz] = acc[jz] + local_bias[jz]

    s = df.customize(top, opt_default=False)
    schedule_mx_quantize_block_f32(s)
    schedule_mx_block_dot(s)
    s.pipeline("quantize_ab_0:tb")
    s.unroll("quantize_ab_0:hb")
    s.unroll("quantize_ab_0:eb")
    s.unroll("quantize_ab_0:kb")
    s.pipeline("quantize_ab_0:ta")
    s.unroll("quantize_ab_0:ha")
    s.unroll("quantize_ab_0:ea")
    s.unroll("quantize_ab_0:ka")
    s.unroll("dot_product_stage_0:k")
    s.unroll("dot_product_stage_0:ka")
    s.unroll("dot_product_stage_0:p")
    s.pipeline("dot_product_stage_0:jj")
    return s


def mx_hbm_mapping(s, base=0, memory="HBM"):
    """Memory channel assignment for a make_mx_*_dataflow schedule."""
    # pylint: disable=import-outside-toplevel
    from .._mlir.ir import IntegerType, MemRefType
    from ..ir.transform import find_func_in_module

    func = find_func_in_module(s.module, s.top_func_name)
    names = [getattr(a, "name", str(a)) for a in s.func_args[s.top_func_name]]
    wide, small = [], []
    for name, ty in zip(names, func.type.inputs):
        elem = MemRefType(ty).element_type
        is_wide = IntegerType.isinstance(elem) and IntegerType(elem).width > 64
        (wide if is_wide else small).append(name)
    mapping = {name: f"{memory}[{base + i}]" for i, name in enumerate(wide)}
    mapping.update({name: f"{memory}[{base + len(wide)}]" for name in small})
    return mapping


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
