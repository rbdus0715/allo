# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object, no-value-for-parameter

# A small GPT-2 with MX arithmetic in its linear layers.
#
# The usual MX split: every Linear (QKV, attention out, MLP fc1/fc2) runs on
# MX operands, its activations quantized on chip (nn.mx_quantize_rows) and its
# weights packed once on the host (mx_pack_np). LayerNorm, GeLU, softmax and
# the residuals stay float32; embedding and the LM head run on the host.
# attn_mx=True also puts attention's Q @ K^T and P @ V on MX operands.
#
# One transformer block is the Allo kernel; the host runs it once per layer.
#   python examples/mx_gpt2.py

import numpy as np
import allo
import allo.ir.types as T
import allo.library.nn as nn
from allo.ir.types import float32
from allo.library.mxint8 import mx_pack_np

# sequence length, model width, heads, MLP width, layers, vocabulary
L, D, H, F, N_LAYER, VOCAB = 8, 64, 2, 256, 2, 128
DH = D // H
QKV = 3 * D
SCALE = 1.0 / float(np.sqrt(DH))


def make_block(Ty, attn_mx):
    """Builds one pre-LN GPT-2 block on MX format Ty."""
    # annotations name Ty (captured: the body uses it), not a closure count
    ATTN_MX = int(attn_mx)

    def causal_softmax(S: "float32[L, L]") -> "float32[L, L]":
        # row softmax of S * SCALE over j <= i
        P: float32[L, L] = 0.0
        mx: float32[L] = -1e30
        tot: float32[L] = 0.0
        for i, j in allo.grid(L, L):
            if j <= i and S[i, j] * SCALE > mx[i]:
                mx[i] = S[i, j] * SCALE
        for i1, j1 in allo.grid(L, L):
            if j1 <= i1:
                P[i1, j1] = allo.exp(S[i1, j1] * SCALE - mx[i1])
                tot[i1] += P[i1, j1]
        for i2, j2 in allo.grid(L, L):
            P[i2, j2] = P[i2, j2] / tot[i2]
        return P

    def attention(qkv: "float32[L, QKV]") -> "float32[L, D]":
        # causal multi-head attention on qkv = [Q | K | V]
        Z: float32[L, D]
        zl: float32[L] = 0.0
        zd: float32[DH] = 0.0
        for h in range(H):
            q: float32[L, DH]
            k: float32[L, DH]
            v: float32[L, DH]
            for r, c in allo.grid(L, DH):
                q[r, c] = qkv[r, h * DH + c]
                k[r, c] = qkv[r, D + h * DH + c]
                v[r, c] = qkv[r, 2 * D + h * DH + c]
            S: float32[L, L]
            O: float32[L, DH]
            with allo.meta_if(ATTN_MX):
                qq = nn.mx_quantize_rows[Ty, L, DH](q)
                kq = nn.mx_quantize_rows[Ty, L, DH](k)
                S = nn.mx_linear2d[Ty, L, L, DH](qq, kq, zl)  # Q @ K^T
                P = causal_softmax(S)
                pq = nn.mx_quantize_rows[Ty, L, L](P)
                vq = nn.mx_quantize_cols[Ty, DH, L](v)  # V^T blocked along L
                O = nn.mx_linear2d[Ty, L, DH, L](pq, vq, zd)  # P @ V
            with allo.meta_else():
                for i, j in allo.grid(L, L):
                    acc: float32 = 0.0
                    for d in range(DH):
                        acc += q[i, d] * k[j, d]
                    S[i, j] = acc
                P = causal_softmax(S)
                for i3, d3 in allo.grid(L, DH):
                    acc3: float32 = 0.0
                    for j3 in range(L):
                        acc3 += P[i3, j3] * v[j3, d3]
                    O[i3, d3] = acc3
            for r1, c1 in allo.grid(L, DH):
                Z[r1, h * DH + c1] = O[r1, c1]
        return Z

    def block(
        x: "float32[L, D]",
        ln1_g: "float32[D]",
        ln1_b: "float32[D]",
        w_qkv: "Ty[QKV, (D + Ty.block_size - 1) // Ty.block_size]",
        b_qkv: "float32[QKV]",
        w_o: "Ty[D, (D + Ty.block_size - 1) // Ty.block_size]",
        b_o: "float32[D]",
        ln2_g: "float32[D]",
        ln2_b: "float32[D]",
        w_fc1: "Ty[F, (D + Ty.block_size - 1) // Ty.block_size]",
        b_fc1: "float32[F]",
        w_fc2: "Ty[D, (F + Ty.block_size - 1) // Ty.block_size]",
        b_fc2: "float32[D]",
    ) -> "float32[L, D]":
        # x + attn(ln1(x)), then + mlp(ln2(.))
        h1 = nn.layer_norm[float32, L, D](x, ln1_g, ln1_b)
        h1q = nn.mx_quantize_rows[Ty, L, D](h1)
        qkv = nn.mx_linear2d[Ty, L, QKV, D](h1q, w_qkv, b_qkv)
        a = attention(qkv)
        aq = nn.mx_quantize_rows[Ty, L, D](a)
        o = nn.mx_linear2d[Ty, L, D, D](aq, w_o, b_o)
        x1 = nn.residual_add[float32, L, D](x, o)
        h2 = nn.layer_norm[float32, L, D](x1, ln2_g, ln2_b)
        h2q = nn.mx_quantize_rows[Ty, L, D](h2)
        f = nn.mx_linear2d[Ty, L, F, D](h2q, w_fc1, b_fc1)
        g = nn.GeLU[float32, L, F](f)
        gq = nn.mx_quantize_rows[Ty, L, F](g)
        y = nn.mx_linear2d[Ty, L, D, F](gq, w_fc2, b_fc2)
        return nn.residual_add[float32, L, D](x1, y)

    return block


# ---- weights and the float64 reference ---------------------------------------

LINEARS = ("w_qkv", "w_o", "w_fc1", "w_fc2")


def init_weights(seed=0):
    rng = np.random.default_rng(seed)
    w = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)
    b = lambda n: (rng.standard_normal(n) * 0.01).astype(np.float32)
    g = lambda n: (1 + rng.standard_normal(n) * 0.1).astype(np.float32)
    layers = [
        {
            "ln1_g": g(D),
            "ln1_b": b(D),
            "w_qkv": w(QKV, D),
            "b_qkv": b(QKV),
            "w_o": w(D, D),
            "b_o": b(D),
            "ln2_g": g(D),
            "ln2_b": b(D),
            "w_fc1": w(F, D),
            "b_fc1": b(F),
            "w_fc2": w(D, F),
            "b_fc2": b(D),
        }
        for _ in range(N_LAYER)
    ]
    # embeddings at GPT-2's scale relative to the block weights
    head = {"wte": w(VOCAB, D) * 5, "wpe": w(L, D) * 5, "lnf_g": g(D), "lnf_b": b(D)}
    return layers, head


def _ln(x, gamma, beta):
    mu = x.mean(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) + beta


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.797885 * (x + 0.044715 * x**3)))


def block_ref(x, p):
    """float64 numpy block, same math as make_block's."""
    p = {k: v.astype(np.float64) for k, v in p.items()}
    qkv = _ln(x, p["ln1_g"], p["ln1_b"]) @ p["w_qkv"].T + p["b_qkv"]
    a = np.zeros_like(x)
    causal = np.tril(np.ones((L, L), bool))
    for h in range(H):
        cols = slice(h * DH, (h + 1) * DH)
        q, k, v = (qkv[:, i * D :][:, cols] for i in range(3))
        s = np.where(causal, q @ k.T * SCALE, -np.inf)
        e = np.exp(s - s.max(-1, keepdims=True))
        a[:, cols] = e / e.sum(-1, keepdims=True) @ v
    x1 = x + a @ p["w_o"].T + p["b_o"]
    f = _gelu(_ln(x1, p["ln2_g"], p["ln2_b"]) @ p["w_fc1"].T + p["b_fc1"])
    return x1 + f @ p["w_fc2"].T + p["b_fc2"]


# ---- host side: embedding, layer loop, LM head ---------------------------------


def gpt2(tokens, layers, head, run_block):
    """Logits [L, VOCAB]; run_block(x, layer params) runs one block."""
    x = head["wte"][tokens] + head["wpe"]
    for p in layers:
        x = run_block(x, p)
    x = _ln(x.astype(np.float64), head["lnf_g"], head["lnf_b"])
    return x @ head["wte"].T.astype(np.float64)  # tied LM head, as in GPT-2


def mx_block_runner(Ty, attn_mx):
    mod = allo.customize(make_block(Ty, attn_mx)).build(target="llvm")
    # pack each layer's weights once
    cache = {}

    def run(x, p):
        key = id(p)
        if key not in cache:
            cache[key] = {
                k: (mx_pack_np(Ty, v) if k in LINEARS else v) for k, v in p.items()
            }
        q = cache[key]
        return mod(
            np.ascontiguousarray(x, np.float32),
            *(q[k] for k in p),
        )

    return run


FORMATS = [T.MXInt(8, 32), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32)]


def main():
    layers, head = init_weights()
    tokens = np.random.default_rng(1).integers(0, VOCAB, L)
    ref = gpt2(tokens, layers, head, lambda x, p: block_ref(x.astype(np.float64), p))
    top1 = ref.argmax(-1)
    print(f"GPT-2: L={L} D={D} heads={H} MLP={F} layers={N_LAYER} vocab={VOCAB}")
    print(f"{'format':<16}{'attention':<11}{'logits rel err':>16}{'top-1 match':>13}")
    results = {}
    for Ty in FORMATS:
        for attn_mx in (False, True):
            logits = gpt2(tokens, layers, head, mx_block_runner(Ty, attn_mx))
            rel = np.linalg.norm(logits - ref) / np.linalg.norm(ref)
            match = (logits.argmax(-1) == top1).mean()
            results[Ty.name, attn_mx] = rel
            attn = "MX" if attn_mx else "float32"
            print(f"{Ty.name:<16}{attn:<11}{rel:>16.2e}{match:>12.0%}")
    return results


if __name__ == "__main__":
    main()
