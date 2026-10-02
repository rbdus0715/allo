# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation, chained-comparison

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




def _mx_wblocks_rows[
    Ty, N, K, NB, Tn
](Q: "int8[N, K]", S: "e8m0[N, NB]", e: "UInt(256)[N * NB]", s: "uint8[N * NB]"):
    BS: ConstExpr[int32] = Ty.block_size
    for tw in range(N // Tn):
        for bw in range(NB):
            for jw in range(Tn):
                w: UInt(256) = 0
                for kw in range(BS):
                    w[kw * 8 : (kw + 1) * 8] = Q[tw * Tn + jw, bw * BS + kw]
                e[(tw * NB + bw) * Tn + jw] = w
                s[(tw * NB + bw) * Tn + jw] = S[tw * Tn + jw, bw]


def _mx_wblocks_cols[
    Ty, K, N, NB, Tn
](Q: "int8[K, N]", S: "e8m0[NB, N]", e: "UInt(256)[N * NB]", s: "uint8[N * NB]"):
    BS: ConstExpr[int32] = Ty.block_size
    for tc in range(N // Tn):
        for bc in range(NB):
            for jc in range(Tn):
                w: UInt(256) = 0
                for kc in range(BS):
                    w[kc * 8 : (kc + 1) * 8] = Q[bc * BS + kc, tc * Tn + jc]
                e[(tc * NB + bc) * Tn + jc] = w
                s[(tc * NB + bc) * Tn + jc] = S[bc, tc * Tn + jc]


def _mx_xblocks_rows[
    Ty, M, K, NB, NT
](
    Q: "int8[M, K]",
    S: "e8m0[M, NB]",
    e: "UInt(256)[NT * M * NB]",
    s: "uint8[NT * M * NB]",
):
    BS: ConstExpr[int32] = Ty.block_size
    for tx in range(NT):
        for ix in range(M):
            for bx in range(NB):
                w: UInt(256) = 0
                for kx in range(BS):
                    w[kx * 8 : (kx + 1) * 8] = Q[ix, bx * BS + kx]
                e[(tx * M + ix) * NB + bx] = w
                s[(tx * M + ix) * NB + bx] = S[ix, bx]


def _mx_xquant_rows[
    Ty, M, K, NB, NT
](X: "float32[M, K]", e: "UInt(256)[NT * M * NB]", s: "uint8[NT * M * NB]"):
    BS: ConstExpr[int32] = Ty.block_size
    TB: ConstExpr[int32] = Ty.bits
    for tq in range(NT):
        for iq in range(M):
            for bq in range(NB):
                blk: float32[BS]
                for eq in range(BS):
                    blk[eq] = X[iq, bq * BS + eq]
                q: Ty = mx_quantize_block_f32[Ty, BS](blk)
                e[(tq * M + iq) * NB + bq] = q[0:256]
                s[(tq * M + iq) * NB + bq] = q[TB - 8 : TB]


def _mx_wquant_rows[
    Ty, N, K, NB, Tn
](W: "float32[N, K]", e: "UInt(256)[N * NB]", s: "uint8[N * NB]"):
    BS: ConstExpr[int32] = Ty.block_size
    TB: ConstExpr[int32] = Ty.bits
    for tv in range(N // Tn):
        for bv in range(NB):
            for jv in range(Tn):
                blk: float32[BS]
                for ev in range(BS):
                    blk[ev] = W[tv * Tn + jv, bv * BS + ev]
                q: Ty = mx_quantize_block_f32[Ty, BS](blk)
                e[(tv * NB + bv) * Tn + jv] = q[0:256]
                s[(tv * NB + bv) * Tn + jv] = q[TB - 8 : TB]


def _mx_wquant_cols[
    Ty, K, N, NB, Tn
](B: "float32[K, N]", e: "UInt(256)[N * NB]", s: "uint8[N * NB]"):
    BS: ConstExpr[int32] = Ty.block_size
    TB: ConstExpr[int32] = Ty.bits
    for tu in range(N // Tn):
        for bu in range(NB):
            for ju in range(Tn):
                blk: float32[BS]
                for eu in range(BS):
                    blk[eu] = B[bu * BS + eu, tu * Tn + ju]
                q: Ty = mx_quantize_block_f32[Ty, BS](blk)
                e[(tu * NB + bu) * Tn + ju] = q[0:256]
                s[(tu * NB + bu) * Tn + ju] = q[TB - 8 : TB]


def _mx_dot_df[
    Ty, M, N, NB, P, Tn
](
    xe: "UInt(256)[(N // Tn) * M * NB]",
    xs: "uint8[(N // Tn) * M * NB]",
    we: "UInt(256)[N * NB]",
    ws: "uint8[N * NB]",
    bias: "float32[N]",
    Z: "float32[M, N]",
):
    BS: ConstExpr[int32] = Ty.block_size
    TB: ConstExpr[int32] = Ty.bits
    wbe: UInt(256)[Tn, NB]
    wbs: uint8[Tn, NB]
    for t in range(N // Tn):
        acc0: float32[Tn]
        for j0 in range(Tn):
            acc0[j0] = 0.0
        a0: Ty = 0
        for b0 in range(NB):
            for jl0 in range(Tn):
                if jl0 == 0:
                    a0[0:256] = xe[t * M * NB + b0]
                    a0[TB - 8 : TB] = xs[t * M * NB + b0]
                ew: UInt(256) = we[(t * NB + b0) * Tn + jl0]
                sw: uint8 = ws[(t * NB + b0) * Tn + jl0]
                with allo.meta_if(M > 1):
                    wbe[jl0, b0] = ew
                    wbs[jl0, b0] = sw
                w0: Ty = 0
                w0[0:256] = ew
                w0[TB - 8 : TB] = sw
                acc0[jl0] += mx_block_dot[Ty, BS](a0, w0)
        for jz0 in range(Tn):
            Z[0, t * Tn + jz0] = acc0[jz0] + bias[t * Tn + jz0]

        with allo.meta_if(M > 1):
            for i in range(1, M):
                acc: float32[Tn]
                for ji in range(Tn):
                    acc[ji] = 0.0
                a: Ty = 0
                for b in range(NB):
                    for jj in range(Tn // P):
                        if jj == 0:
                            a[0:256] = xe[(t * M + i) * NB + b]
                            a[TB - 8 : TB] = xs[(t * M + i) * NB + b]
                        for p in range(P):
                            w: Ty = 0
                            w[0:256] = wbe[jj * P + p, b]
                            w[TB - 8 : TB] = wbs[jj * P + p, b]
                            acc[jj * P + p] += mx_block_dot[Ty, BS](a, w)
                for jz in range(Tn):
                    Z[i, t * Tn + jz] = acc[jz] + bias[t * Tn + jz]


def mx_linear2d_q[
    Ty, M, N, K, NB, P, Tn
](
    Xq: "int8[M, K]",
    Xs: "e8m0[M, NB]",
    Wq: "int8[N, K]",
    Ws: "e8m0[N, NB]",
    bias: "float32[N]",
) -> "float32[M, N]":
    we: UInt(256)[N * NB]
    ws: uint8[N * NB]
    xe: UInt(256)[(N // Tn) * M * NB]
    xs: uint8[(N // Tn) * M * NB]
    _mx_wblocks_rows[Ty, N, K, NB, Tn](Wq, Ws, we, ws)
    _mx_xblocks_rows[Ty, M, K, NB, N // Tn](Xq, Xs, xe, xs)
    Z: float32[M, N]
    _mx_dot_df[Ty, M, N, NB, P, Tn](xe, xs, we, ws, bias, Z)
    return Z


def mx_linear2d_wq[
    Ty, M, N, K, NB, P, Tn
](
    X: "float32[M, K]", Wq: "int8[N, K]", Ws: "e8m0[N, NB]", bias: "float32[N]"
) -> "float32[M, N]":
    we: UInt(256)[N * NB]
    ws: uint8[N * NB]
    xe: UInt(256)[(N // Tn) * M * NB]
    xs: uint8[(N // Tn) * M * NB]
    _mx_wblocks_rows[Ty, N, K, NB, Tn](Wq, Ws, we, ws)
    _mx_xquant_rows[Ty, M, K, NB, N // Tn](X, xe, xs)
    Z: float32[M, N]
    _mx_dot_df[Ty, M, N, NB, P, Tn](xe, xs, we, ws, bias, Z)
    return Z


def mx_matmul_q[
    Ty, M, N, K, NB, P, Tn
](
    Aq: "int8[M, K]", As: "e8m0[M, NB]", Bq: "int8[K, N]", Bs: "e8m0[NB, N]"
) -> "float32[M, N]":
    we: UInt(256)[N * NB]
    ws: uint8[N * NB]
    xe: UInt(256)[(N // Tn) * M * NB]
    xs: uint8[(N // Tn) * M * NB]
    _mx_wblocks_cols[Ty, K, N, NB, Tn](Bq, Bs, we, ws)
    _mx_xblocks_rows[Ty, M, K, NB, N // Tn](Aq, As, xe, xs)
    zero: float32[N] = 0.0
    Z: float32[M, N]
    _mx_dot_df[Ty, M, N, NB, P, Tn](xe, xs, we, ws, zero, Z)
    return Z


def mx_matmul_wq[
    Ty, M, N, K, NB, P, Tn
](A: "float32[M, K]", Bq: "int8[K, N]", Bs: "e8m0[NB, N]") -> "float32[M, N]":
    we: UInt(256)[N * NB]
    ws: uint8[N * NB]
    xe: UInt(256)[(N // Tn) * M * NB]
    xs: uint8[(N // Tn) * M * NB]
    _mx_wblocks_cols[Ty, K, N, NB, Tn](Bq, Bs, we, ws)
    _mx_xquant_rows[Ty, M, K, NB, N // Tn](A, xe, xs)
    zero: float32[N] = 0.0
    Z: float32[M, N]
    _mx_dot_df[Ty, M, N, NB, P, Tn](xe, xs, we, ws, zero, Z)
    return Z


def mx_linear2d_ff[
    Ty, M, N, K, NB, P, Tn
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    we: UInt(256)[N * NB]
    ws: uint8[N * NB]
    xe: UInt(256)[(N // Tn) * M * NB]
    xs: uint8[(N // Tn) * M * NB]
    _mx_wquant_rows[Ty, N, K, NB, Tn](W, we, ws)
    _mx_xquant_rows[Ty, M, K, NB, N // Tn](X, xe, xs)
    Z: float32[M, N]
    _mx_dot_df[Ty, M, N, NB, P, Tn](xe, xs, we, ws, bias, Z)
    return Z


def mx_matmul_ff[
    Ty, M, N, K, NB, P, Tn
](A: "float32[M, K]", B: "float32[K, N]") -> "float32[M, N]":
    we: UInt(256)[N * NB]
    ws: uint8[N * NB]
    xe: UInt(256)[(N // Tn) * M * NB]
    xs: uint8[(N // Tn) * M * NB]
    _mx_wquant_cols[Ty, K, N, NB, Tn](B, we, ws)
    _mx_xquant_rows[Ty, M, K, NB, N // Tn](A, xe, xs)
    zero: float32[N] = 0.0
    Z: float32[M, N]
    _mx_dot_df[Ty, M, N, NB, P, Tn](xe, xs, we, ws, zero, Z)
    return Z


def schedule_mx_dataflow(s, depth=16):
    """Streams + dataflow for the MX allo.linear/allo.matmul implementations (HLS only)."""
    # pylint: disable=import-outside-toplevel
    from .._mlir.dialects import func as func_d
    from ..ir.utils import MockBuffer

    def loop_names(fn_name):
        fn = next(
            op
            for op in s.module.body.operations
            if isinstance(op, func_d.FuncOp)
            and op.attributes["sym_name"].value == fn_name
        )
        names = set()

        def walk(block):
            for o in block.operations:
                if "loop_name" in o.attributes:
                    names.add(o.attributes["loop_name"].value)
                for region in o.regions:
                    for b in region.blocks:
                        walk(b)

        walk(fn.entry_block)
        return names

    loops = {  # producer prefix -> (loop to pipeline, loop to unroll)
        "_mx_wblocks_rows": ("jw", "kw"),
        "_mx_wblocks_cols": ("jc", "kc"),
        "_mx_xblocks_rows": ("bx", "kx"),
        "_mx_xquant_rows": ("bq", "eq"),
        "_mx_wquant_rows": ("jv", "ev"),
        "_mx_wquant_cols": ("ju", "eu"),
    }
    funcs = [op for op in s.module.body.operations if isinstance(op, func_d.FuncOp)]
    applied = False
    for fn in funcs:
        ops = list(fn.entry_block.operations)
        allocs = {
            o.attributes["name"].value
            for o in ops
            if o.operation.name == "memref.alloc" and "name" in o.attributes
        }
        callees = [
            o.attributes["callee"].value for o in ops if o.operation.name == "func.call"
        ]
        dots = [c for c in callees if c.startswith("_mx_dot_df")]
        if not dots or not {"we", "ws", "xe", "xs"} <= allocs:
            continue
        name, dot = fn.attributes["sym_name"].value, dots[0]
        for buf in ("we", "ws", "xe", "xs"):
            s.to(MockBuffer(name, buf), dot, depth=depth)
        s.dataflow(name)
        for callee in callees:
            for prefix, (pipe, unroll) in loops.items():
                if callee.startswith(prefix):
                    s.pipeline(f"{callee}:{pipe}")
                    s.unroll(f"{callee}:{unroll}")
        dot_loops = loop_names(dot)
        s.pipeline(f"{dot}:jl0")
        if "jj" in dot_loops:  # rows 1..M-1 exist only when M > 1
            s.pipeline(f"{dot}:jj")
            s.unroll(f"{dot}:p")
        applied = True
    return applied


def mx_pick_tile_units(N, NB, block_bytes=33, budget_bytes=4 * 2**20):
    """Weight tile Tn and block-dot unit count P for the MX dataflow."""
    tn = max(
        t
        for t in range(1, N + 1)
        if N % t == 0 and t * NB * block_bytes <= budget_bytes
    )
    units = next((p for p in (8, 4, 2) if tn % p == 0 and tn // p >= 8), 1)
    return tn, units


def mx_auto_tn(N, NB):
    return mx_pick_tile_units(N, NB)[0]


def mx_auto_p(N, NB):
    return mx_pick_tile_units(N, NB)[1]


def mx_linear2d[
    Ty, M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    _mx_nb: ConstExpr[int32] = K // Ty.block_size
    _mx_tn: ConstExpr[int32] = mx_auto_tn(N, K // Ty.block_size)
    _mx_p: ConstExpr[int32] = mx_auto_p(N, K // Ty.block_size)
    return mx_linear2d_ff[Ty, M, N, K, _mx_nb, _mx_p, _mx_tn](X, W, bias)


def schedule_mx_linear2d(s):
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
    schedule_mx_linear2d(s)
    s.pipeline("mx_linear3d:k")
    s.pipeline("mx_linear3d:j")
    return s




def make_mx_matmul_dataflow(Ty, M, N, K, P, depth=4):
    """Dataflow MX Z = A @ B.T with on-chip quantization."""
    import allo.dataflow as df

    BS = Ty.block_size
    NB = K // BS
    assert K % BS == 0, f"K={K} must be a multiple of block_size={BS}"
    assert N % P == 0, f"N={N} must be a multiple of P={P}"
    NJ = N // P
    FW = 2 * NB  # 512-bit words per float32 row

    @df.region()
    def top(A: "UInt(512)[M * FW]", B: "UInt(512)[N * FW]", Z: "float32[M * N]"):
        pipe_a_q: Stream["uint8[BS + 1]", depth]
        pipe_b_q: Stream["uint8[BS + 1]", depth]

        @df.kernel(mapping=[1], args=[A, B])
        def quantize_ab(local_A: "UInt(512)[M * FW]", local_B: "UInt(512)[N * FW]"):
            for tb in range(N * NB):
                blk: float32[BS]
                for hb in range(2):
                    fword: UInt(512) = local_B[tb * 2 + hb]
                    for eb in range(16):
                        bits: int32 = fword[eb * 32 : (eb + 1) * 32]
                        blk[hb * 16 + eb] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: uint8[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for kb in range(BS):
                    bundle[kb + 1] = word[kb * Ty.elem_bits : (kb + 1) * Ty.elem_bits]
                pipe_b_q.put(bundle)

            for ta in range(M * NB):
                blk: float32[BS]
                for ha in range(2):
                    fword: UInt(512) = local_A[ta * 2 + ha]
                    for ea in range(16):
                        bits: int32 = fword[ea * 32 : (ea + 1) * 32]
                        blk[ha * 16 + ea] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: uint8[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for ka in range(BS):
                    bundle[ka + 1] = word[ka * Ty.elem_bits : (ka + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle)

        @df.kernel(mapping=[1], args=[Z])
        def dot_product_stage(local_Z: "float32[M * N]"):
            b_buf: uint8[N, NB, BS + 1]
            for jd in range(N):
                for bd in range(NB):
                    tmp: uint8[BS + 1] = pipe_b_q.get()
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
                            a_blk: uint8[BS + 1] = pipe_a_q.get()
                            block_a: uint8[BS]
                            for ka in range(BS):
                                block_a[ka] = a_blk[ka + 1]
                            word_a = _mx_pack_word[Ty, BS](a_blk[0], block_a)
                        for p in range(P):
                            j: int32 = jj * P + p
                            scale_b: uint8 = b_buf[j, b, 0]
                            block_b: uint8[BS]
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
    assert K % BS == 0, f"K={K} must be a multiple of block_size={BS}"
    assert N % P == 0, f"N={N} must be a multiple of P={P}"
    NJ = N // P
    FW = 2 * NB  # 512-bit words per float32 row

    @df.region()
    def top(
        X: "UInt(512)[M * FW]",
        W: "UInt(512)[N * FW]",
        bias: "float32[N]",
        Z: "float32[M * N]",
    ):
        pipe_a_q: Stream["uint8[BS + 1]", depth]
        pipe_b_q: Stream["uint8[BS + 1]", depth]

        @df.kernel(mapping=[1], args=[X, W])
        def quantize_ab(local_X: "UInt(512)[M * FW]", local_W: "UInt(512)[N * FW]"):
            for tb in range(N * NB):
                blk: float32[BS]
                for hb in range(2):
                    fword: UInt(512) = local_W[tb * 2 + hb]
                    for eb in range(16):
                        bits: int32 = fword[eb * 32 : (eb + 1) * 32]
                        blk[hb * 16 + eb] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: uint8[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for kb in range(BS):
                    bundle[kb + 1] = word[kb * Ty.elem_bits : (kb + 1) * Ty.elem_bits]
                pipe_b_q.put(bundle)

            for ta in range(M * NB):
                blk: float32[BS]
                for ha in range(2):
                    fword: UInt(512) = local_X[ta * 2 + ha]
                    for ea in range(16):
                        bits: int32 = fword[ea * 32 : (ea + 1) * 32]
                        blk[ha * 16 + ea] = bits.bitcast()
                word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                bundle: uint8[BS + 1]
                bundle[0] = word[Ty.bits - 8 : Ty.bits]
                for ka in range(BS):
                    bundle[ka + 1] = word[ka * Ty.elem_bits : (ka + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle)

        @df.kernel(mapping=[1], args=[bias, Z])
        def dot_product_stage(local_bias: "float32[N]", local_Z: "float32[M * N]"):
            b_buf: uint8[N, NB, BS + 1]
            for jd in range(N):
                for bd in range(NB):
                    tmp: uint8[BS + 1] = pipe_b_q.get()
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
                            a_blk: uint8[BS + 1] = pipe_a_q.get()
                            block_a: uint8[BS]
                            for ka in range(BS):
                                block_a[ka] = a_blk[ka + 1]
                            word_a = _mx_pack_word[Ty, BS](a_blk[0], block_a)
                        for p in range(P):
                            j: int32 = jj * P + p
                            scale_b: uint8 = b_buf[j, b, 0]
                            block_b: uint8[BS]
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


def mx_pick_tile(Ty, N, K, P, budget_bytes=4 * 2**20):
    """Largest weight tile Tn that fits the on-chip budget."""
    blk_bytes = (K // Ty.block_size) * (Ty.block_size + 1)
    fits = [
        t for t in range(P, N + 1, P) if N % t == 0 and t * blk_bytes <= budget_bytes
    ]
    assert fits, f"no tile of N={N} (multiple of P={P}) fits {budget_bytes} bytes"
    return max(fits)


def make_mx_linear2d_dataflow_prequant(Ty, M, N, K, P, Tn=None, depth=4):
    """make_mx_linear2d_dataflow with W pre-quantized and tiled."""
    import allo.dataflow as df

    BS = Ty.block_size
    NB = K // BS
    Tn = N if Tn is None else Tn
    assert K % BS == 0, f"K={K} must be a multiple of block_size={BS}"
    assert N % Tn == 0, f"N={N} must be a multiple of Tn={Tn}"
    assert Tn % P == 0, f"Tn={Tn} must be a multiple of P={P}"
    NT = N // Tn  # weight tiles
    NJT = Tn // P  # P-wide column groups per tile
    FW = 2 * NB  # 512-bit words per float32 row

    @df.region()
    def top(
        X: "UInt(512)[M * FW]",
        W_q: "UInt(256)[N * NB]",
        W_s: "uint8[N * NB]",
        bias: "float32[N]",
        Z: "float32[M * N]",
    ):
        pipe_a_q: Stream["uint8[BS + 1]", depth]
        pipe_b_q: Stream["uint8[BS + 1]", depth]

        @df.kernel(mapping=[1], args=[X])
        def quantize_a(local_X: "UInt(512)[M * FW]"):
            for tq in range(NT):  # X is re-streamed once per weight tile
                for ta in range(M * NB):  # every block of every X row
                    blk: float32[BS]
                    for ha in range(2):
                        fword: UInt(512) = local_X[ta * 2 + ha]
                        for ea in range(16):
                            bits: int32 = fword[ea * 32 : (ea + 1) * 32]
                            blk[ha * 16 + ea] = bits.bitcast()
                    word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                    bundle: uint8[BS + 1]
                    bundle[0] = word[Ty.bits - 8 : Ty.bits]
                    for ka in range(BS):
                        bundle[ka + 1] = word[
                            ka * Ty.elem_bits : (ka + 1) * Ty.elem_bits
                        ]
                    pipe_a_q.put(bundle)

        @df.kernel(mapping=[1], args=[W_q, W_s])
        def load_w(local_Wq: "UInt(256)[N * NB]", local_Ws: "uint8[N * NB]"):
            for tw in range(N * NB):
                qword: UInt(256) = local_Wq[tw]
                bundle: uint8[BS + 1]
                bundle[0] = local_Ws[tw]
                for kw in range(BS):
                    bundle[kw + 1] = qword[kw * 8 : (kw + 1) * 8]
                pipe_b_q.put(bundle)

        @df.kernel(mapping=[1], args=[bias, Z])
        def dot_product_stage(local_bias: "float32[N]", local_Z: "float32[M * N]"):
            b_buf: uint8[Tn, NB, BS + 1]
            for t in range(NT):  # weight tile t: output columns t*Tn .. t*Tn+Tn-1
                acc0: float32[Tn]
                for j0 in range(Tn):
                    acc0[j0] = 0.0
                word_a0: Ty = 0
                for b0 in range(NB):  # K-direction block b0
                    for jl0 in range(Tn):  # tile column jl0 (II=1)
                        if jl0 == 0:
                            a_blk0: uint8[BS + 1] = pipe_a_q.get()
                            block_a0: uint8[BS]
                            for ka0 in range(BS):
                                block_a0[ka0] = a_blk0[ka0 + 1]
                            word_a0 = _mx_pack_word[Ty, BS](a_blk0[0], block_a0)
                        b_blk: uint8[BS + 1] = pipe_b_q.get()
                        with allo.meta_if(M > 1):
                            for kd in range(BS + 1):
                                b_buf[jl0, b0, kd] = b_blk[kd]
                        block_b0: uint8[BS]
                        for k0 in range(BS):
                            block_b0[k0] = b_blk[k0 + 1]
                        word_b0: Ty = _mx_pack_word[Ty, BS](b_blk[0], block_b0)
                        acc0[jl0] += mx_block_dot[Ty, BS](word_a0, word_b0)
                for jz0 in range(Tn):  # tile column jz0 of output row 0
                    local_Z[t * Tn + jz0] = acc0[jz0] + local_bias[t * Tn + jz0]

                with allo.meta_if(M > 1):
                    for i in range(1, M):  # X row i
                        acc: float32[Tn]
                        for ji in range(Tn):
                            acc[ji] = 0.0
                        word_a: Ty = 0
                        for b in range(NB):  # K-direction block b
                            for jj in range(NJT):  # P-wide column group jj
                                if jj == 0:
                                    a_blk: uint8[BS + 1] = pipe_a_q.get()
                                    block_a: uint8[BS]
                                    for ka in range(BS):
                                        block_a[ka] = a_blk[ka + 1]
                                    word_a = _mx_pack_word[Ty, BS](a_blk[0], block_a)
                                for p in range(P):  # unit p -> column jj*P+p
                                    j: int32 = jj * P + p
                                    scale_b: uint8 = b_buf[j, b, 0]
                                    block_b: uint8[BS]
                                    for k in range(BS):
                                        block_b[k] = b_buf[j, b, k + 1]
                                    word_b: Ty = _mx_pack_word[Ty, BS](scale_b, block_b)
                                    acc[j] += mx_block_dot[Ty, BS](word_a, word_b)
                        for jz in range(Tn):  # tile column jz of output row i
                            local_Z[i * N + t * Tn + jz] = (
                                acc[jz] + local_bias[t * Tn + jz]
                            )

    s = df.customize(top, opt_default=False)
    schedule_mx_quantize_block_f32(s)
    schedule_mx_block_dot(s)
    s.pipeline("quantize_a_0:ta")
    s.unroll("quantize_a_0:ha")
    s.unroll("quantize_a_0:ea")
    s.unroll("quantize_a_0:ka")
    s.pipeline("load_w_0:tw")
    s.unroll("load_w_0:kw")
    s.unroll("dot_product_stage_0:k0")
    s.unroll("dot_product_stage_0:ka0")
    s.pipeline("dot_product_stage_0:jl0")
    if M > 1:  # these loops only exist then (meta_if above)
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
        is_wide = IntegerType.isinstance(elem) and IntegerType(elem).width >= 256
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
