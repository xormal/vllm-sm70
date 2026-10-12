# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ЗАДАЧА: перекомпоновать int8 так, чтобы он ОБОГНАЛ e5m2, сохранив точность int8.
# e5m2 (KVFMT=1) -- НЕПОДВИЖНЫЙ эталон: не трогается и не улучшается.
#
# ДВЕ СТАТЬИ ОТСТАВАНИЯ, обе снимаются перекомпоновкой, а не ускорением операций:
#   (1) РАСПАКОВКА: у int8 три инструкции на четыре значения (XOR знака + две перестановки),
#       у e5m2 две (только перестановки). Лечится СМЕЩЁННЫМ БАЙТОМ (offset binary, u = v+128):
#       работу XOR делает квантователь ОДИН РАЗ, смещение остаётся той же константой 1152 и
#       по-прежнему сворачивается в ранг-1. Значения ПОБИТОВО те же -- это чистая перекладка
#       работы из ядра в подготовку.  (форматы 6/7)
#   (2) ТАБЛИЦА МАСШТАБОВ: масштаб НА ПОЗИЦИЮ требует ВТОРОГО потока подачи -- чтение у везущих
#       варпов, укладка в разделяемую, чтение оттуда на каждый элемент S и P. Масштаб НА КАНАЛ
#       (на столбец) -- ранг-1 по строке, поэтому сворачивается ВНЕ ядра и стоит НОЛЬ:
#           K:  sum_t q_t (w_t x_t) = sum_t (q_t w_t) x_t   -> домножить Q один раз до запуска
#           V:  sum_j p_j (w_d x_jd) = w_d sum_j p_j x_jd   -> домножить выход один раз после
#       Обе свёртки стоят O(Sq*d) против O(Sq*Sk*d) работы ядра.  (форматы 4/7)
#
# ДИСЦИПЛИНА ЗАМЕРА: варианты ЧЕРЕДУЮТСЯ внутри раунда (дрейф частот достаётся всем поровну),
# берётся медиана, чужие процессы на картах 0-1 считаются до и после и печатаются рядом.
import math
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256


def cards_state(idx="0,1"):
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
                            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return n, cl
    except Exception as e:
        return -1, [str(e)]


def interleaved(fns, warmup=3, rounds=11):
    for name, fn in fns.items():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    acc = {k: [] for k in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            acc[name].append(s.elapsed_time(e))
    out = {}
    for k, v in acc.items():
        v.sort()
        out[k] = (v[len(v) // 2], v[0], v[-1])
    return out


def make_kv(Hkv, Sk, gen, dev="cuda"):
    """Ключи/значения фп16. gen='iid' -- гауссовы; gen='chan' -- со СТОЛБЦОВЫМИ выбросами,
    то есть с той структурой, которая измерена на настоящем кэше (выброс сидит на ФИКСИРОВАННОМ
    индексе канала, факт F2 из разбора точности)."""
    k = torch.randn(1, Hkv, Sk, D, device=dev, dtype=torch.float16) * 0.4
    v = torch.randn(1, Hkv, Sk, D, device=dev, dtype=torch.float16) * 0.4
    if gen == "chan":
        # 4 канала из 256 несут в 100 раз больше -- та же структура, что у настоящих K
        idx = torch.tensor([7, 63, 128, 200], device=dev)
        k[:, :, :, idx] *= 100.0
        v[:, :, :, idx] *= 100.0
    return k.contiguous(), v.contiguous()


def quant_pertoken(x):
    """int8 + масштаб НА ПОЗИЦИЮ (строка = ключ): то, что отгружено."""
    a = x.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8).contiguous()
    return q, a.squeeze(-1).contiguous().float()


def quant_perchan(x):
    """int8 + масштаб НА КАНАЛ (столбец): свернётся в Q / в эпилог, таблицы в ядре нет."""
    a = x.float().abs().amax(2, keepdim=True).clamp_min(6.1e-5) / 127.0     # [1,Hkv,1,D]
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8).contiguous()
    return q, a.squeeze(2).contiguous().float()                            # [1,Hkv,D]


def to_biased(q):
    """Смещённый байт: u = v + 128. Байты те же, что даст квантователь в бою."""
    return (q.to(torch.int16) + 128).to(torch.uint8).view(torch.int8).contiguous()


def ref_attn(q, k, v, sc, causal):
    """Эталон в fp32 по ИСХОДНЫМ fp16 K/V."""
    B, Sq, H, d = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    r = H // Hkv
    qq = q.float().permute(0, 2, 1, 3)                                    # B,H,Sq,d
    kk = k.float().repeat_interleave(r, dim=1)
    vv = v.float().repeat_interleave(r, dim=1)
    s = torch.matmul(qq, kk.transpose(-1, -2)) * sc
    if causal:
        off = Sk - Sq
        i = torch.arange(Sq, device=q.device).view(-1, 1) + off
        j = torch.arange(Sk, device=q.device).view(1, -1)
        s = s.masked_fill(j > i, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vv).permute(0, 2, 1, 3)                        # B,Sq,H,d


def relL2(a, b):
    a, b = a.float(), b.float()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def main():
    torch.manual_seed(0)
    pext = fa2_sm70._ext.prefill_ext()
    H, Hkv = int(os.environ.get("H", 16)), 2
    Sq = int(os.environ.get("SQ", 512))
    lens = [int(x) for x in os.environ.get("LENS", "4096,16384,65536,131072").split(",")]
    gen = os.environ.get("GEN", "iid")
    sc = 1.0 / math.sqrt(D)

    n0, cl0 = cards_state()
    print(f"# чужих процессов на картах 0-1 в начале: {n0}   частоты/мощность: {cl0}")
    print(f"# H={H} Hkv={Hkv} d={D} Sq={Sq} causal=True  данные={gen}")
    print("# фмт: 0=int8+таблица(отгружено) 1=e5m2(ЭТАЛОН) 6=смещ.байт+таблица "
          "4=int8 без таблицы 7=смещ.байт без таблицы 5=e5m2 без таблицы(ДИАГН.)")
    print()
    hdr = (f"{'Sk':>8} {'i8 баз':>8} {'e5m2':>8} {'ф6 смещ':>8} {'ф4 бт':>8} {'ф7 полн':>8} "
           f"{'ф5 диаг':>8} {'ф8 чер':>8} | {'e5m2/баз':>8} {'e5m2/ф6':>8} {'e5m2/ф8':>8}")
    print(hdr)
    print("-" * len(hdr))

    for Sk in lens:
        q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        kf, vf = make_kv(Hkv, Sk, gen)
        kb = kf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        vb = vf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        kq, ks = quant_pertoken(kf)
        vq, vs = quant_pertoken(vf)
        kqb, vqb = to_biased(kq), to_biased(vq)
        kc, kcs = quant_perchan(kf)
        vc, vcs = quant_perchan(vf)
        kcb, vcb = to_biased(kc), to_biased(vc)
        # ЧЕРЕДУЮЩАЯСЯ таблица: (ks,vs) рядом по позиции -- одна широкая загрузка на обе
        ksv = torch.stack([ks, vs], dim=-1).contiguous()
        # свёртка столбцового масштаба: Q домножается на масштаб K по каналу (на голову kv)
        rep = H // Hkv
        qw = (q.float() * kcs[0].repeat_interleave(rep, dim=0).view(1, 1, H, D)).half().contiguous()
        vwe = vcs[0].repeat_interleave(rep, dim=0).view(1, 1, H, D).float()

        fns = {
            "i8":   lambda: pext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True),
            "e5m2": lambda: pext.attn_fwd_volta_e5m2(q, kb, vb, sc, True),
            "f6":   lambda: pext.attn_fwd_volta_i8(q, kqb, ks, vqb, vs, sc, True, 6),
            "f4":   lambda: pext.attn_fwd_volta_i8_nosc(qw, kc, vc, sc, True),
            "f7":   lambda: pext.attn_fwd_volta_i8(qw, kcb, torch.empty(0, device="cuda"),
                                                   vcb, torch.empty(0, device="cuda"), sc, True, 7),
            "f5":   lambda: pext.attn_fwd_volta_e5m2_nosc(q, kb, vb, sc, True),
            "f8":   lambda: pext.attn_fwd_volta_i8(q, kqb, ksv, vqb, ksv, sc, True, 8),
        }
        r = interleaved(fns)
        print(f"{Sk:>8} {r['i8'][0]:>8.3f} {r['e5m2'][0]:>8.3f} {r['f6'][0]:>8.3f} {r['f4'][0]:>8.3f} "
              f"{r['f7'][0]:>8.3f} {r['f5'][0]:>8.3f} {r['f8'][0]:>8.3f} | "
              f"{r['e5m2'][0]/r['i8'][0]:>8.4f} {r['e5m2'][0]/r['f6'][0]:>8.4f} "
              f"{r['e5m2'][0]/r['f8'][0]:>8.4f}")

        if Sk == lens[0]:
            # --- СВЕРКА: тождественность байтовых перекладок и точность против fp32
            o_i8 = fns["i8"]()[0]
            o_f6 = fns["f6"]()[0]
            o_f4 = fns["f4"]()[0].float() * vwe
            o_f7 = fns["f7"]()[0].float() * vwe
            o_e5 = fns["e5m2"]()[0]
            same06 = torch.equal(o_i8, o_f6)
            same08 = torch.equal(o_i8, fns["f8"]()[0])
            same47 = torch.equal(fns["f4"]()[0], fns["f7"]()[0])
            ref = ref_attn(q, kf, vf, sc, True)
            print()
            print(f"  ПОБИТОВО ф0==ф6 (смещ. байт ничего не меняет): {same06}")
            print(f"  ПОБИТОВО ф0==ф8 (чередование таблицы -- те же fp32):  {same08}")
            print(f"  ПОБИТОВО ф4==ф7 (то же на бестабличном пути):  {same47}")
            print(f"  relL2 к fp32:  int8-баз {relL2(o_i8, ref):.3e}   e5m2 {relL2(o_e5, ref):.3e}   "
                  f"столбцовый int8 {relL2(o_f4, ref):.3e}")
            print()

        del q, kf, vf, kb, vb, kq, ks, vq, vs, kqb, vqb, kc, kcs, vc, vcs, kcb, vcb, qw, ksv
        torch.cuda.empty_cache()

    n1, cl1 = cards_state()
    print()
    print(f"# чужих процессов на картах 0-1 в конце: {n1}   частоты/мощность: {cl1}")


if __name__ == "__main__":
    main()
