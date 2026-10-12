"""Сборка поправки Clef для одного ранга TP в раскладке слитых выходов vLLM (Qwen3.5/3.8, TP=2).

Только время загрузки: шаг зовёт расширение fa2sm70_clef, питона на пути шага нет.

Поправка для строк Clef на боевой ABL = обратная аблитерация (W_orig - W_abl, ранг 1, o/out/down
слоёв 2-63) + LoRA Clef (W_clef - W_orig, ранг 256, слои 40-63) + плотная разница in_proj_a/b.

Модули (номер mod в операциях расширения) и раскладка ЛОКАЛЬНОГО выхода ранга r:
  0 qkv      GDN  in_proj_qkvz: [q 1024 | k 1024 | v 3072 | z 3072]  -- каждая часть HF-строк режется
                                пополам по рангу (как weight_loader vLLM: narrow(rank*size, size))
             attn qkv_proj:     [q+gate 6144 | k 512 | v 512]
  1 ba       GDN  in_proj_ba:   [b 24 | a 24]                         (плотная разница)
  2 out      GDN out_proj / attn o_proj: row-parallel, вход -- свои столбцы, выход 5120 целиком
  3 gate_up  [gate 8704 | up 8704]
  4 down     row-parallel, вход 8704 своих столбцов
Row-parallel: A режется по входу, B целиком -- частичная сумма уходит в all-reduce самого слоя.
Ранги Clef и обратной аблитерации в одном модуле склеиваются в одно сжатие (257 -> 264, кратно 8).
"""
import re

import torch
from safetensors import safe_open

MOD = {"qkv": 0, "ba": 1, "out": 2, "gate_up": 3, "down": 4}
PFX = "model.language_model.layers."
ALIGN = 256
RPAD = 8


class Blob:
    def __init__(self):
        self.parts, self.off, self.ops = [], 0, []

    def put(self, t):
        t = t.to(torch.float16).contiguous()
        b = t.view(torch.uint8).reshape(-1)
        off = self.off
        self.parts.append((off, b))
        self.off += (b.numel() + ALIGN - 1) // ALIGN * ALIGN
        return off

    def shrink(self, mod, A):            # A [r, in]
        self.ops.append((mod, 0, self.put(A), A.shape[0], A.shape[1], 0))

    def expand(self, mod, B, col):       # B [n, r]
        self.ops.append((mod, 1, self.put(B), B.shape[0], B.shape[1], col))

    def dense(self, mod, D, col):        # D [n, in]
        self.ops.append((mod, 2, self.put(D), D.shape[0], D.shape[1], col))

    def pack(self, pinned):
        host = torch.empty(max(self.off, ALIGN), dtype=torch.uint8, pin_memory=pinned)
        host.zero_()
        for off, b in self.parts:
            host[off:off + b.numel()].copy_(b)
        return host, torch.tensor(self.ops, dtype=torch.int64).reshape(-1, 6)


def half_rows(t, r, tp):
    n = t.shape[0] // tp
    return t[r * n:(r + 1) * n]


def half_cols(t, r, tp):
    n = t.shape[1] // tp
    return t[:, r * n:(r + 1) * n]


def cat_pad(As, Bs):
    """склеить ранги: A [Σr, in], B [out, Σr], добить нулями до кратного RPAD"""
    A = torch.cat(As, 0)
    B = torch.cat(Bs, 1)
    r = A.shape[0]
    rp = (r + RPAD - 1) // RPAD * RPAD
    if rp != r:
        A = torch.cat([A, A.new_zeros(rp - r, A.shape[1])], 0)
        B = torch.cat([B, B.new_zeros(B.shape[0], rp - r)], 1)
    return A, B


class Src:
    def __init__(self, path):
        self.f = safe_open(path, "pt")
        self.keys = set(self.f.keys())

    def get(self, k):
        return self.f.get_tensor(k) if k in self.keys else None


def sobrat(clef_path, deabl_path, rank, tp=2, stream_from=40, sloi=None, layer_types=None,
           key_dim=2048, value_dim=6144):
    """-> [(model_layer, streamed, host uint8, ops int64 [P,6])] для ранга rank.

    key_dim = linear_num_key_heads * linear_key_head_dim (16*128), value_dim = 48*128 у Qwen3.8-27B."""
    clef, deabl = Src(clef_path), Src(deabl_path)
    vse = set()
    for k in clef.keys | deabl.keys:
        m = re.match(re.escape(PFX) + r"(\d+)\.", k)
        if m:
            vse.add(int(m.group(1)))
    out = []
    for L in sorted(vse):
        if sloi is not None and L not in sloi:
            continue
        p = f"{PFX}{L}."
        attn = any(k.startswith(p + "self_attn.") for k in clef.keys | deabl.keys)
        if layer_types is not None:
            attn = layer_types[L] == "full_attention"
        bl = Blob()

        def lora(name):
            A, B = clef.get(p + name + ".lora_A"), clef.get(p + name + ".lora_B")
            return (A, B) if A is not None else None

        # --- mod 0: входные проекции
        if attn:
            qkv = [lora(n) for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")]
            if any(x is not None for x in qkv):
                if any(x is None for x in qkv):
                    raise RuntimeError(f"слой {L}: LoRA только на части q/k/v -- раскладка столбцов не выводится")
                col = 0
                for A, B in qkv:
                    Bl = half_rows(B, rank, tp)
                    bl.shrink(MOD["qkv"], A)
                    bl.expand(MOD["qkv"], Bl, col)
                    col += Bl.shape[0]
        else:
            ab = lora("linear_attn.in_proj_qkv")
            if ab is not None:
                A, B = ab
                kd, vd = key_dim, value_dim
                assert B.shape[0] == 2 * kd + vd, (L, B.shape, kd, vd)
                q, k, v = B[:kd], B[kd:2 * kd], B[2 * kd:]
                bl.shrink(MOD["qkv"], A)
                ql, kl, vl = half_rows(q, rank, tp), half_rows(k, rank, tp), half_rows(v, rank, tp)
                bl.expand(MOD["qkv"], ql, 0)
                bl.expand(MOD["qkv"], kl, ql.shape[0])
                bl.expand(MOD["qkv"], vl, ql.shape[0] + kl.shape[0])
                zab = lora("linear_attn.in_proj_z")
                A, B = zab
                bl.shrink(MOD["qkv"], A)
                bl.expand(MOD["qkv"], half_rows(B, rank, tp), ql.shape[0] + kl.shape[0] + vl.shape[0])
            Db, Da = clef.get(p + "linear_attn.in_proj_b.delta"), clef.get(p + "linear_attn.in_proj_a.delta")
            if Db is not None:
                Dbl, Dal = half_rows(Db, rank, tp), half_rows(Da, rank, tp)
                bl.dense(MOD["ba"], Dbl, 0)
                bl.dense(MOD["ba"], Dal, Dbl.shape[0])

        # --- mod 2 / 4: row-parallel (Clef + обратная аблитерация, склеенные)
        for mod, name in ((MOD["out"], "self_attn.o_proj" if attn else "linear_attn.out_proj"),
                          (MOD["down"], "mlp.down_proj")):
            As, Bs = [], []
            ab = lora(name)
            if ab is not None:
                As.append(half_cols(ab[0], rank, tp)); Bs.append(ab[1])
            dA, dB = deabl.get(p + name + ".deabl_A"), deabl.get(p + name + ".deabl_B")
            if dA is not None:
                As.append(half_cols(dA, rank, tp)); Bs.append(dB)
            if As:
                A, B = cat_pad(As, Bs)
                bl.shrink(mod, A)
                bl.expand(mod, B, 0)

        # --- mod 3: gate | up
        g, u = lora("mlp.gate_proj"), lora("mlp.up_proj")
        if g is not None:
            gl = half_rows(g[1], rank, tp)
            bl.shrink(MOD["gate_up"], g[0]); bl.expand(MOD["gate_up"], gl, 0)
            bl.shrink(MOD["gate_up"], u[0]); bl.expand(MOD["gate_up"], half_rows(u[1], rank, tp), gl.shape[0])

        if not bl.ops:
            continue
        streamed = L >= stream_from
        host, ops = bl.pack(pinned=streamed)
        out.append((L, streamed, host, ops))
    return out


def ustanovit(ext, sobrannoe, nslot=3, max_rows=2048, rmax=264):
    ext.init_delta(nslot, max_rows, rmax)
    for L, streamed, host, ops in sobrannoe:
        ext.add_layer(L, streamed, host, ops)
    ext.finalize_delta()
