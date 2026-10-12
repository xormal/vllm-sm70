# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# СЕКУНДОМЕР ПОД УЧЁТ ПО СЛОТАМ ВЫДАЧИ (docs/VOLTA_SM70.md §3b).
#
# Мерит варианты боевого форвардного ядра ролей ОДНИМ процессом, чередуя их ВНУТРИ раунда, и
# отдаёт МЕДИАНУ ПАРНЫХ ОТНОШЕНИЙ к базе (а не отношение медиан): дрейф частот тогда достаётся
# обоим вариантам одного раунда поровну и в отношении сокращается.
#
# Дисциплина, без которой число недействительно:
#   * чужие процессы на картах считаются ДО и ПОСЛЕ и печатаются рядом с результатом;
#   * частоты/мощность печатаются там же (карта под нагрузкой соседа даёт другой ответ);
#   * карты 0 и 1 систематически расходятся -- мерить обе (ключ CARD).
#
# Варианты задаются кодами nosc из attn_fwd_volta_i8 (см. карту форматов в volta_fwd_ws.cuh):
#   0 -- боевой int8 + таблица масштабов на позицию (БАЗА)
#   1 -- e5m2 (ЭТАЛОН СРАВНЕНИЯ, не трогается)
#   6 -- смещённый байт + таблица; 7 -- смещённый байт БЕЗ таблицы (масштаб столбцовый)
#   8 -- смещённый байт + чередующаяся таблица;  +100 -- то же с ШИРОКОЙ перекладкой у везущих
import math
import os
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256


def cards_state(idx):
    me = str(os.getpid())
    try:
        uu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", idx],
                            capture_output=True, text=True, timeout=10).stdout.split()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        n = 0
        for line in apps.strip().splitlines():
            if not line.strip():
                continue
            u, p = [s.strip() for s in line.split(",")]
            if u in uu and p != me:
                n += 1
        cl = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
                             "--format=csv,noheader,nounits", "-i", idx],
                            capture_output=True, text=True, timeout=10).stdout.strip().replace("\n", " | ")
        return n, cl
    except Exception as e:
        return -1, str(e)


def paired(fns, base, warmup=3, rounds=17):
    """Чередование ВНУТРИ раунда + медиана ПАРНЫХ отношений к базе."""
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    torch.cuda.synchronize()
    raw = {k: [] for k in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            raw[name].append(s.elapsed_time(e))
    out = {}
    for k, v in raw.items():
        rat = [raw[base][i] / v[i] for i in range(rounds)]     # >1 = вариант БЫСТРЕЕ базы
        out[k] = (statistics.median(v), statistics.median(rat), min(v))
    return out


def quant_pertoken(x):
    a = x.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8).contiguous()
    return q, a.squeeze(-1).contiguous().float()


def quant_perchan(x):
    a = x.float().abs().amax(2, keepdim=True).clamp_min(6.1e-5) / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8).contiguous()
    return q, a.squeeze(2).contiguous().float()


def to_biased(q):
    return (q.to(torch.int16) + 128).to(torch.uint8).view(torch.int8).contiguous()


def ref_attn(q, k, v, sc, causal):
    B, Sq, H, d = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    r = H // Hkv
    qq = q.float().permute(0, 2, 1, 3)
    kk = k.float().repeat_interleave(r, dim=1)
    vv = v.float().repeat_interleave(r, dim=1)
    s = torch.matmul(qq, kk.transpose(-1, -2)) * sc
    if causal:
        off = Sk - Sq
        i = torch.arange(Sq, device=q.device).view(-1, 1) + off
        j = torch.arange(Sk, device=q.device).view(1, -1)
        s = s.masked_fill(j > i, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vv).permute(0, 2, 1, 3)


def relL2(a, b):
    a, b = a.float(), b.float()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def main():
    torch.manual_seed(0)
    pext = fa2_sm70._ext.prefill_ext()
    card = os.environ.get("CARD", "0")
    H, Hkv = int(os.environ.get("H", 16)), 2
    Sq = int(os.environ.get("SQ", 512))
    rounds = int(os.environ.get("ROUNDS", 17))
    lens = [int(x) for x in os.environ.get("LENS", "8192,32768,131072").split(",")]
    codes = [int(x) for x in os.environ.get("CODES", "0,1,6,7,8").split(",")]
    sc = 1.0 / math.sqrt(D)

    n0, cl0 = cards_state(card)
    print(f"# КАРТА {card}   чужих процессов в начале: {n0}   {cl0}")
    print(f"# H={H} Hkv={Hkv} d={D} Sq={Sq} causal=True  раундов={rounds}  коды={codes}")
    hdr = f"{'Sk':>8} " + " ".join(f"{('ф%d' % c):>9}" for c in codes) + " | " + \
          " ".join(f"{('x%d/б' % c):>8}" for c in codes)
    print(hdr)
    print("-" * len(hdr))

    for Sk in lens:
        q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        kf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        vf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        kb = kf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        vb = vf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        # e4m3 -- ТОТ ЖЕ байтовый транспорт и ТА ЖЕ машинерия масштабов, что у e5m2 (холостая 1.f),
        # поэтому пара 1/11 меряет РОВНО цену разворота и ничего кроме неё.
        k4 = kf.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        v4 = vf.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        kq, ks = quant_pertoken(kf)
        vq, vs = quant_pertoken(vf)
        kqb, vqb = to_biased(kq), to_biased(vq)
        ksv = torch.stack([ks, vs], dim=-1).contiguous()
        kc, kcs = quant_perchan(kf)
        vc, vcs = quant_perchan(vf)
        kcb, vcb = to_biased(kc), to_biased(vc)
        rep = H // Hkv
        qw = (q.float() * kcs[0].repeat_interleave(rep, dim=0).view(1, 1, H, D)).half().contiguous()
        vwe = vcs[0].repeat_interleave(rep, dim=0).view(1, 1, H, D).float()
        empty = torch.empty(0, device="cuda")

        def mk(c):
            f = c % 100
            if f == 1:
                return lambda: pext.attn_fwd_volta_i8(q, kb.view(torch.int8), empty,
                                                      vb.view(torch.int8), empty, sc, True, 0)
            if f == 0:
                return lambda: pext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True, c)
            if f == 6:
                return lambda: pext.attn_fwd_volta_i8(q, kqb, ks, vqb, vs, sc, True, c)
            if f == 7:
                return lambda: pext.attn_fwd_volta_i8(qw, kcb, empty, vcb, empty, sc, True, c)
            if f in (8, 9, 10):
                return lambda: pext.attn_fwd_volta_i8(q, kqb, ksv, vqb, ksv, sc, True, c)
            if f in (11, 12, 13):
                # 11 -- боевой e4m3 (6 инстр./4 Б); 12 -- ФАЛЬСИФИКАТОР (разворот снят, ответ
                # неверен, меряется ДОЛЯ фазы); 13 -- тот же ответ прежней записью (8 инстр.).
                return lambda: pext.attn_fwd_volta_i8(q, k4.view(torch.int8), empty,
                                                      v4.view(torch.int8), empty, sc, True, c)
            raise SystemExit(f"неизвестный код {c}")

        fns = {str(c): mk(c) for c in codes}
        r = paired(fns, str(codes[0]), rounds=rounds)
        print(f"{Sk:>8} " + " ".join(f"{r[str(c)][0]:>9.3f}" for c in codes) + " | " +
              " ".join(f"{r[str(c)][1]:>8.4f}" for c in codes))

        if Sk == lens[0]:
            base = fns[str(codes[0])]()[0]
            ref = ref_attn(q, kf, vf, sc, True)
            print(f"    ТОЧНОСТЬ (Sk={Sk}): relL2 базы к fp32 = {relL2(base, ref):.4e}")
            for c in codes[1:]:
                o = fns[str(c)]()[0]
                if c % 100 == 7:
                    o = (o.float() * vwe).half()
                print(f"      ф{c}: torch.equal к базе = {torch.equal(o, base)}   "
                      f"relL2 к базе = {relL2(o, base):.3e}   relL2 к fp32 = {relL2(o, ref):.4e}")

        del q, kf, vf, kb, vb, kq, ks, vq, vs, kqb, vqb, ksv, kc, kcs, vc, vcs, kcb, vcb, qw, k4, v4
        torch.cuda.empty_cache()

    n1, cl1 = cards_state(card)
    print(f"# КАРТА {card}   чужих процессов в конце: {n1}   {cl1}")


if __name__ == "__main__":
    main()
