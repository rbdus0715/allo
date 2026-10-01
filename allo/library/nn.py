# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation, chained-comparison

from .. import dsl
from .systolic import systolic
from .mxint8 import (
    mx_quantize_block_f32,
    mx_block_dot,
    _mx_pack_word,
    schedule_mx_quantize_block_f32,
    schedule_mx_block_dot,
)
from ..ir.types import float32, uint8, int32, ConstExpr, Stream


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
    # B is stored row-major transposed, i.e. [N, K] like nn.Linear.weight,
    # so this computes A @ B^T. Each K-sized row is split into MX blocks
    # (mx_quantize_block_f32) and reduced with mx_block_dot.
    #
    # Quantized blocks are kept only as scalar Ty temporaries (qa/qb below),
    # never gathered into a Ty[] array: an array of Ty (mxint8's ~264-bit
    # packed word) corrupts the LLVM JIT's ExecutionEngine at interpreter
    # shutdown -- a backend bug in wide-custom-integer array lowering,
    # confirmed independent of the array's shape/length; plain Ty scalars
    # are unaffected. The cost is that A's row is re-quantized once per
    # output column instead of once per row -- acceptable here since this
    # `customize`-based version is for functional/software use, not the
    # actual HLS-accelerator path (see make_mx_matmul_dataflow for that).
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


def mx_linear2d[
    Ty, M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    # Same interface/semantics as linear2d above (W is [out_features,
    # in_features]), but the matmul reduction runs through mxint8 blocks.
    Z = mx_matmul[Ty, M, N, K](X, W)
    for i in range(M):
        for j in range(N):
            Z[i, j] = Z[i, j] + bias[j]
    return Z


def schedule_mx_linear2d(s):
    schedule_mx_matmul(s)
    s.pipeline("mx_linear2d:j")
    return s


def mx_linear3d[
    Ty, B, L, D, M
](X: "float32[B, L, D]", W: "float32[M, D]", bias: "float32[M]") -> "float32[B, L, M]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    # 3D input variant of mx_linear2d (batch of sequences), matching
    # linear3d's interface.
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


######################################################################
# Dataflow (streaming) variants, generalizing allo/library/mxint8.py's
# make_mx_dot_general_dataflow from a single dot product to a full
# [M, K] x [N, K]^T matmul:
#   - quantize_ab streams each of the M rows of A, then each of the N rows
#     of B, through mx_quantize_block_f32 exactly once.
#   - dot_product_stage first fully drains B's bundles into an on-chip
#     buffer (B is reused M times), then streams each A row once and dots
#     it against every cached B row with mx_block_dot (A is reused N times
#     but quantized only once, unlike mx_matmul above).
#
# A quantized block is shipped as a (scale, elements) uint8 bundle, exactly
# like make_mx_dot_general_dataflow's bundle_a/bundle_b, and repacked into
# the Ty word right before mx_block_dot with _mx_pack_word -- never as a Ty
# (or Ty[]) value on a stream or in a buffer. This isn't just the AXI-width
# workaround the comment in mxint8.py describes: it also sidesteps the same
# wide-custom-integer array bug mx_matmul's docstring above describes,
# which affects Ty arrays wherever they appear (stream element type or
# plain on-chip buffer), not only function returns.
######################################################################


def make_mx_matmul_dataflow(Ty, M, N, K, P, depth=4):
    # Imported lazily -- a module-level `import allo.dataflow` here would
    # break `import allo`: allo.dataflow's own import chain needs
    # allo.library.KERNEL2SCHEDULE, which doesn't exist yet while
    # allo/library/__init__.py is still executing its top-level imports
    # (this module is reachable from there, to register mx_matmul etc.).
    import allo.dataflow as df

    BS = Ty.block_size
    NB = K // BS

    @df.region()
    def top(A: "float32[M, K]", B: "float32[N, K]", Z: "float32[M, N]"):
        pipe_a_q: Stream["uint8[NB, BS + 1]", depth]
        pipe_b_q: Stream["uint8[NB, BS + 1]", depth]

        # A and B are quantized by a single kernel (rather than one kernel
        # each): mx_quantize_block_f32[Ty, BS] is a generic instantiation,
        # and two kernels each instantiating it independently emit two
        # copies of the same helper function name into the region's shared
        # module, which the dataflow lowering rejects as a symbol
        # redefinition. mxint8.py's quantize_ab kernel has the same shape
        # for the same reason.
        @df.kernel(mapping=[1], args=[A, B])
        def quantize_ab(local_A: "float32[M, K]", local_B: "float32[N, K]"):
            # loop names are suffixed (ba/ea/ka vs bb/eb/kb) because the
            # A-side and B-side loops would otherwise share the same bare
            # names within this one kernel, which makes s.unroll/s.pipeline
            # below unable to tell the two loop bands apart.
            for i in range(M):
                bundle: uint8[NB, BS + 1]
                for ba in range(NB):
                    blk: float32[BS]
                    for ea in range(BS):
                        blk[ea] = local_A[i, ba * BS + ea]
                    word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                    bundle[ba, 0] = word[Ty.bits - 8 : Ty.bits]
                    for ka in range(BS):
                        bundle[ba, ka + 1] = word[
                            ka * Ty.elem_bits : (ka + 1) * Ty.elem_bits
                        ]
                pipe_a_q.put(bundle)

            for j in range(N):
                bundle: uint8[NB, BS + 1]
                for bb in range(NB):
                    blk: float32[BS]
                    for eb in range(BS):
                        blk[eb] = local_B[j, bb * BS + eb]
                    word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                    bundle[bb, 0] = word[Ty.bits - 8 : Ty.bits]
                    for kb in range(BS):
                        bundle[bb, kb + 1] = word[
                            kb * Ty.elem_bits : (kb + 1) * Ty.elem_bits
                        ]
                pipe_b_q.put(bundle)

        @df.kernel(mapping=[1], args=[Z])
        def dot_product_stage(local_Z: "float32[M, N]"):
            # b_buf is filled element-by-element from a freshly-`get()`
            # bundle rather than `b_buf[jd] = pipe_b_q.get()` directly --
            # assigning a whole stream item straight into an indexed slot
            # of a bigger array silently drops the data (a separate, plain
            # bug from the Ty-array one above); copying through a fresh
            # local first works.
            b_buf: uint8[N, NB, BS + 1]
            for jd in range(N):
                tmp: uint8[NB, BS + 1] = pipe_b_q.get()
                for bd in range(NB):
                    for kd in range(BS + 1):
                        b_buf[jd, bd, kd] = tmp[bd, kd]

            for i in range(M):
                a_bundle: uint8[NB, BS + 1] = pipe_a_q.get()
                for j in range(N):
                    acc: float32 = 0.0
                    for b in range(NB):
                        scale_a: uint8 = a_bundle[b, 0]
                        scale_b: uint8 = b_buf[j, b, 0]
                        block_a: uint8[BS]
                        block_b: uint8[BS]
                        for k in range(BS):
                            block_a[k] = a_bundle[b, k + 1]
                            block_b[k] = b_buf[j, b, k + 1]
                        word_a: Ty = _mx_pack_word[Ty, BS](scale_a, block_a)
                        word_b: Ty = _mx_pack_word[Ty, BS](scale_b, block_b)
                        acc += mx_block_dot[Ty, BS](word_a, word_b)
                    local_Z[i, j] = acc

    s = df.customize(top, opt_default=False)
    schedule_mx_quantize_block_f32(s)
    schedule_mx_block_dot(s)
    s.unroll("quantize_ab_0:ea")
    s.unroll("quantize_ab_0:eb")
    s.unroll("quantize_ab_0:ka")
    s.unroll("quantize_ab_0:kb")
    s.pipeline("quantize_ab_0:ba")
    s.pipeline("quantize_ab_0:bb")
    s.unroll("dot_product_stage_0:k")
    s.pipeline("dot_product_stage_0:j", initiation_interval=P)
    s.unroll("dot_product_stage_0:b")
    return s


def make_mx_linear2d_dataflow(Ty, M, N, K, P, depth=4):
    import allo.dataflow as df

    BS = Ty.block_size
    NB = K // BS

    @df.region()
    def top(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]", Z: "float32[M, N]"
    ):
        pipe_a_q: Stream["uint8[NB, BS + 1]", depth]
        pipe_b_q: Stream["uint8[NB, BS + 1]", depth]

        # See make_mx_matmul_dataflow: X and W must be quantized by a single
        # kernel, not one each, or the shared mx_quantize_block_f32[Ty, BS]
        # instantiation gets emitted twice and the dataflow lowering rejects
        # the duplicate symbol.
        @df.kernel(mapping=[1], args=[X, W])
        def quantize_ab(local_X: "float32[M, K]", local_W: "float32[N, K]"):
            for i in range(M):
                bundle: uint8[NB, BS + 1]
                for ba in range(NB):
                    blk: float32[BS]
                    for ea in range(BS):
                        blk[ea] = local_X[i, ba * BS + ea]
                    word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                    bundle[ba, 0] = word[Ty.bits - 8 : Ty.bits]
                    for ka in range(BS):
                        bundle[ba, ka + 1] = word[
                            ka * Ty.elem_bits : (ka + 1) * Ty.elem_bits
                        ]
                pipe_a_q.put(bundle)

            for j in range(N):
                bundle: uint8[NB, BS + 1]
                for bb in range(NB):
                    blk: float32[BS]
                    for eb in range(BS):
                        blk[eb] = local_W[j, bb * BS + eb]
                    word: Ty = mx_quantize_block_f32[Ty, BS](blk)
                    bundle[bb, 0] = word[Ty.bits - 8 : Ty.bits]
                    for kb in range(BS):
                        bundle[bb, kb + 1] = word[
                            kb * Ty.elem_bits : (kb + 1) * Ty.elem_bits
                        ]
                pipe_b_q.put(bundle)

        @df.kernel(mapping=[1], args=[bias, Z])
        def dot_product_stage(local_bias: "float32[N]", local_Z: "float32[M, N]"):
            b_buf: uint8[N, NB, BS + 1]
            for jd in range(N):
                tmp: uint8[NB, BS + 1] = pipe_b_q.get()
                for bd in range(NB):
                    for kd in range(BS + 1):
                        b_buf[jd, bd, kd] = tmp[bd, kd]

            for i in range(M):
                a_bundle: uint8[NB, BS + 1] = pipe_a_q.get()
                for j in range(N):
                    acc: float32 = 0.0
                    for b in range(NB):
                        scale_a: uint8 = a_bundle[b, 0]
                        scale_b: uint8 = b_buf[j, b, 0]
                        block_a: uint8[BS]
                        block_b: uint8[BS]
                        for k in range(BS):
                            block_a[k] = a_bundle[b, k + 1]
                            block_b[k] = b_buf[j, b, k + 1]
                        word_a: Ty = _mx_pack_word[Ty, BS](scale_a, block_a)
                        word_b: Ty = _mx_pack_word[Ty, BS](scale_b, block_b)
                        acc += mx_block_dot[Ty, BS](word_a, word_b)
                    local_Z[i, j] = acc + local_bias[j]

    s = df.customize(top, opt_default=False)
    schedule_mx_quantize_block_f32(s)
    schedule_mx_block_dot(s)
    s.unroll("quantize_ab_0:ea")
    s.unroll("quantize_ab_0:eb")
    s.unroll("quantize_ab_0:ka")
    s.unroll("quantize_ab_0:kb")
    s.pipeline("quantize_ab_0:ba")
    s.pipeline("quantize_ab_0:bb")
    s.unroll("dot_product_stage_0:k")
    s.pipeline("dot_product_stage_0:j", initiation_interval=P)
    s.unroll("dot_product_stage_0:b")
    return s


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
