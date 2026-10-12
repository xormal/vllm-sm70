# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ТОЧНОСТЬ ТРЁХ ФОРМАТОВ ПРОТИВ НЕЗАВИСИМОГО ЭТАЛОНА fp32 (не против fp16-ядра!).
#
# ПОЧЕМУ ЭТО НЕ ТО ЖЕ САМОЕ, ЧТО НАШ СТАРЫЙ ТЕСТ. tests/test_volta_fwd.py проверяет
# torch.equal(e5m2-ядро, fp16-ядро) на ОДНИХ И ТЕХ ЖЕ хранимых значениях. Это утверждение ПРО ЯДРО:
# разворот байта точен. Про ошибку САМОГО КВАНТОВАНИЯ оно не говорит ничего. Здесь эталон -- fp32
# внимание на ИСХОДНЫХ (неквантованных) значениях, поэтому измеряется ровно цена формата.
#
# Раскладываем ошибку на два слагаемых:
#   формат  = |fp32-внимание на квантованных K/V  -  fp32-внимание на исходных|
#   ядро    = |наше ядро на квантованных K/V      -  fp32-внимание на квантованных|
# Первое -- цена представления, второе -- цена нашей арифметики. Складывать их в одно число значит
# скрыть, кто виноват.
#
# ДАННЫЕ. Снимков K/V с боевой модели в репозитории нет, и снимать их означало бы поднять 27B на
# картах, где идёт замер, -- поэтому распределения СИНТЕТИЧЕСКИЕ, и это сказано прямо. Зато они
# параметризованы РАЗБРОСОМ ПОРЯДКОВ внутри строки (rho = max/rms), который и есть та величина, где
# по теории проходит граница между двумя схемами. Семейство "spike" воспроизводит известное свойство
# KV настоящих трансформеров: несколько каналов с амплитудой в 10^2-10^3 раз выше типичной, ПРИЧЁМ
# НА ОДНИХ И ТЕХ ЖЕ индексах канала для почти всех позиций.
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256


def make_kv(kind, Hkv, T, gen, R=100.0, nspike=4):
    """Возвращает fp32 K/V нужного семейства."""
    k = torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.float32, generator=gen) * 0.4
    v = torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.float32, generator=gen) * 0.4
    if kind == "gauss":
        return k, v
    if kind == "spike":           # ФИКСИРОВАННЫЕ каналы-выбросы, общие для всех позиций
        idx = torch.arange(nspike, device="cuda") * (D // nspike)
        k[..., idx] *= R
        v[..., idx] *= R
        return k, v
    if kind == "spike_rand":      # выброс в СВОЁМ канале у каждой позиции (контроль к "spike")
        idx = torch.randint(0, D, (1, Hkv, T, nspike), device="cuda", generator=gen)
        for s in range(nspike):
            k.scatter_(3, idx[..., s:s + 1], k.gather(3, idx[..., s:s + 1]) * R)
            v.scatter_(3, idx[..., s:s + 1], v.gather(3, idx[..., s:s + 1]) * R)
        return k, v
    if kind == "lognorm":
        k = k * torch.exp(torch.randn_like(k) * 2.0)
        v = v * torch.exp(torch.randn_like(v) * 2.0)
        return k, v
    raise ValueError(kind)


def rho(x):
    """Разброс порядков внутри строки: max/rms, усреднённый по строкам."""
    m = x.abs().amax(-1)
    r = x.pow(2).mean(-1).sqrt()
    return (m / r.clamp_min(1e-30)).mean().item()


def quant_i8(x):
    a = x.abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    q = torch.clamp(torch.round(x / a), -127, 127)
    return q.to(torch.int8), a.squeeze(-1).contiguous(), (q * a)


def quant_e5m2(x):
    b = x.to(torch.float16).to(torch.float8_e5m2)
    return b.view(torch.uint8), b.to(torch.float32)


def ref_attn(q, k, v, sc, causal):
    """Эталон в fp32 (накопление в fp32, вход fp32) -- независим от наших ядер."""
    H, Hkv = q.size(2), k.size(1)
    Q = q.transpose(1, 2).float()
    K = k.repeat_interleave(H // Hkv, 1).float()
    V = v.repeat_interleave(H // Hkv, 1).float()
    s = (Q @ K.transpose(-1, -2)) * sc
    if causal:
        Sq, T = s.size(-2), s.size(-1)
        i = torch.arange(Sq, device=s.device).view(-1, 1)
        j = torch.arange(T, device=s.device).view(1, -1)
        s = s.masked_fill(j > i + (T - Sq), float("-inf"))
    return (torch.softmax(s, -1) @ V).transpose(1, 2)


def relL2(a, b):
    return ((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(1e-30)).item()


def main():
    p = fa2_sm70._ext.prefill_ext()
    gen = torch.Generator(device="cuda"); gen.manual_seed(7)
    H, Hkv, Sq, T = 8, 2, 128, 2048
    sc = 1.0 / math.sqrt(D)
    kinds = os.environ.get("KINDS", "gauss,spike,spike_rand,lognorm").split(",")
    Rs = [float(x) for x in os.environ.get("RS", "10,100,1000").split(",")]

    print("# Эталон: fp32-внимание на ИСХОДНЫХ fp32 K/V. Данные СИНТЕТИЧЕСКИЕ (снимков с модели нет).")
    print(f"# H={H} Hkv={Hkv} d={D} Sq={Sq} T={T} causal")
    print()
    hdr = (f"{'семейство':>12} {'rho K':>7} | {'формат e5m2':>12} {'формат int8':>12} | "
           f"{'ядро e5m2':>11} {'ядро int8':>11} | {'ИТОГО e5m2':>11} {'ИТОГО int8':>11} {'выигрыш':>8}")
    print(hdr); print("-" * len(hdr))
    for kind in kinds:
        for R in (Rs if kind.startswith("spike") else [0.0]):
            k32, v32 = make_kv(kind, Hkv, T, gen, R=R)
            q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16, generator=gen) * 0.3).contiguous()
            base = ref_attn(q, k32, v32, sc, True)                     # ЭТАЛОН

            kb, kd5 = quant_e5m2(k32); vb, vd5 = quant_e5m2(v32)
            kq, ks, kd8 = quant_i8(k32); vq, vs, vd8 = quant_i8(v32)

            f5 = ref_attn(q, kd5, vd5, sc, True)                       # цена формата e5m2
            f8 = ref_attn(q, kd8, vd8, sc, True)                       # цена формата int8
            o5, _ = p.attn_fwd_volta_e5m2(q, kb, vb, sc, True)
            o8, _ = p.attn_fwd_volta_i8(q, kq, ks.float(), vq, vs.float(), sc, True)

            e_f5, e_f8 = relL2(f5, base), relL2(f8, base)
            e_k5, e_k8 = relL2(o5.float(), f5), relL2(o8.float(), f8)
            e_t5, e_t8 = relL2(o5.float(), base), relL2(o8.float(), base)
            name = f"{kind}" + (f" R={R:g}" if R else "")
            print(f"{name:>12} {rho(k32):>7.1f} | {e_f5:>12.2e} {e_f8:>12.2e} | "
                  f"{e_k5:>11.2e} {e_k8:>11.2e} | {e_t5:>11.2e} {e_t8:>11.2e} {e_t8/max(e_t5,1e-30):>7.2f}x")
            del k32, v32, q, base, kb, kd5, vb, vd5, kq, ks, kd8, vq, vs, vd8, f5, f8, o5, o8
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
