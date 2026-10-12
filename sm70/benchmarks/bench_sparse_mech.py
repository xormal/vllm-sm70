# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# МЕХАНИЗМ АДАПТИВНОЙ РАЗРЕЖЕННОСТИ: замер НАКЛАДНЫХ, а не рекламного ускорения.
#
# Порядок именно такой, потому что каждый следующий пункт имеет смысл только если предыдущий сошёлся:
#   0  ПОБИТОВОЕ равенство при плотности 100% (список и битовая карта против отгруженного ядра);
#   1  цена механизма при плотности 100% -- она платится ВСЕГДА, и её вычитают из любого выигрыша;
#   2  цена сводок и отбора в долях внимания;
#   3  выигрыш при плотности 75% и 60% -- ПРОПОРЦИОНАЛЬНОЙ (как считает оракул) и ПОГОЛОВНОЙ;
#   4  НЕРАВНОМЕРНАЯ НАГРУЗКА: та же суммарная работа, разное её распределение -> потеря;
#   5  лечение: очередь заданий по убыванию веса (LPT) вместо трёхмерной сетки;
#   6  свип плотности -> отделить ПОСТОЯННУЮ часть задания от цены механизма;
#   7  СОГЛАСОВАННОСТЬ МАСОК соседних блоков запросов.
#
# Пункты 4-5 задание ставило как главную опасность, а пункт 7 в нём не упоминался. ЗАМЕР ПОМЕНЯЛ ИХ
# МЕСТАМИ: неравномерность от поголовной плотности ничтожна (макс/среднее веса задания 1.11-1.15,
# любой порядок выдачи в пределах 3.5%), а несогласованность масок при 250K стоит от 9% до ВСЕГО
# выигрыша (плотность 0.835 при независимом отборе -- 0.978x, то есть медленнее плотного внимания).
import os, sys, math, time, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70

BQ = BK = 64
D = 256
NSM = 80                      # V100: 80 SM. Занятость этого ядра -- 1 блок/SM (разделяемая 83.5 КБ)

# Плотности из ОРАКУЛА (docs/ATTENTION_DENSITY.md), бюджет 1.8e-2 по худшей голове слоя:
#   слой  3 -> 0.90   слой 23 -> 0.85   слой 43 -> 0.78   слой 63 -> 0.52
# В модели 16 слоёв полного внимания, равномерно по глубине -> интерполируем по слою.
ORACLE = [(3, 0.90), (23, 0.85), (43, 0.78), (63, 0.52)]


def oracle_density(layer):
    xs = [l for l, _ in ORACLE]; ys = [d for _, d in ORACLE]
    return float(np.interp(layer, xs, ys))


def layer_densities():
    """16 слоёв полного внимания Qwen3.5-27B: каждый 4-й, то есть слои 3, 7, ..., 63."""
    return [(L, oracle_density(L)) for L in range(3, 64, 4)]


def gpu_state():
    """Кто ЕЩЁ на карте. Замер на общей карте -- это не замер, и знать об этом надо ДО выводов, а не
    после: в этой сессии сосед поднял на картах 0 и 1 свой сервер посреди прогона, и два «эффекта»
    (0.72x у одной раскладки, 183 мс у очереди заданий) оказались его всплесками, а не нашими."""
    import subprocess
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,clocks.sm",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        return " | ".join(x.strip() for x in q.strip().split("\n"))
    except Exception:
        return "nvidia-smi недоступен"


def measure(variants, rounds=4, reps=2, warm=2):
    """ЧЕРЕДУЮЩИЙСЯ замер набора вариантов. Против соседа на карте и дрейфа частот есть ровно два
    приёма, и здесь оба:
      * МИНИМУМ, а не медиана -- минимум это прогон, которому меньше всех мешали; помеха может
        только ЗАМЕДЛИТЬ, поэтому оценка снизу и есть оценка «чистого» времени;
      * ПАРНОЕ ОТНОШЕНИЕ по кругам -- в каждом круге меряются ВСЕ варианты, отношение берётся
        внутри круга, и медиана отношений сносит общий множитель помехи.
    Возвращает {имя: (мин, медиана отношений к первому варианту)}."""
    names = list(variants)
    for n in names:
        for _ in range(warm): variants[n]()
    torch.cuda.synchronize()
    per = {n: [] for n in names}
    for _ in range(rounds):
        for n in names:
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); variants[n](); torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            per[n].append(min(ts))
    base = names[0]
    out = {}
    for n in names:
        rat = sorted(per[n][r] / per[base][r] for r in range(rounds))
        out[n] = (min(per[n]), rat[len(rat) // 2])
    return out


def bench(fn, it=10, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)                          # МИНИМУМ: помеха умеет только замедлять


# ---------------------------------------------------------------------------------------------
# Построение маски на хосте. Отбор ЗДЕСЬ синтетический -- он служит замеру МЕХАНИЗМА, качество
# настоящего отбора мерится отдельно (tools/sparse_selector_vs_oracle.py, на дампе модели).
# ВАЖНО, И ЭТО ВЫЯСНИЛОСЬ ЗАМЕРОМ: РАСКЛАДКА оставленных плиток влияет на время СИЛЬНЕЕ, чем их
# число. При 250K и плотности 0.60 независимый случайный набор даёт 1.46x, а любой набор, ОДИНАКОВЫЙ
# у соседних блоков запросов, -- 1.63x. Поэтому «случайный» здесь -- не нейтральный выбор, а НИЖНЯЯ
# граница, и указывать раскладку в каждом замере обязательно.
# ---------------------------------------------------------------------------------------------
def nkbC_of(qb, Sq, Sk, causal):
    kend = min(qb * BQ + BQ + (Sk - Sq), Sk) if causal else Sk
    return (kend + BK - 1) // BK


def _pick(n, t, pattern, rng):
    """Какие t плиток из n оставить. ФОРМА НАБОРА -- отдельная ось замера, и не декоративная:
    внимание в префилле у нас на 97.7% обслуживается L2 (docs/VOLTA_SM70.md), а L2 общий на всю
    карту. Значит одинаковое ЧИСЛО оставленных плиток при разной их РАСКЛАДКЕ даёт разный трафик."""
    if t >= n:
        return np.arange(n, dtype=np.int64)
    if pattern == "recent":                       # непрерывный хвост: 0 (сток) + t-1 последних
        return np.sort(np.concatenate([[0], np.arange(n - (t - 1), n, dtype=np.int64)]))
    if pattern == "first":                        # непрерывное НАЧАЛО + диагональ. Отличается от
        # «recent» ровно тем, что плитки ДАЛЕКО от диагонали: разделяет «непрерывность» и «диагональ».
        return np.sort(np.concatenate([np.arange(0, t - 1, dtype=np.int64), [n - 1]]))
    if pattern == "stride":                       # равномерно РАЗРЕЖЕННЫЙ: тот же шаг между плитками
        idx = np.unique((np.linspace(1, n - 2, t - 2)).astype(np.int64)) if t > 2 else np.empty(0, np.int64)
        return np.sort(np.unique(np.concatenate([[0], idx, [n - 1]])))
    if pattern.startswith("cluster"):             # сгустки по g плиток -- так ведёт себя настоящий отбор
        g = int(pattern[len("cluster"):])
        ncl = max(1, (t - 2 + g - 1) // g)
        starts = rng.choice(max(1, (n - 2) // g), size=min(ncl, max(1, (n - 2) // g)), replace=False)
        keep = np.unique(np.concatenate([1 + starts * g + i for i in range(g)]))
        keep = keep[keep < n - 1][:t - 2]
        return np.sort(np.concatenate([[0], keep, [n - 1]]))
    pool = np.arange(1, n - 1, dtype=np.int64)    # равномерно случайный: НАИХУДШИЙ для L2
    pick = rng.choice(pool, size=t - 2, replace=False) if t > 2 else np.empty(0, np.int64)
    return np.sort(np.concatenate([[0], pick, [n - 1]]))


def build_mask_shared(B, H, Sq, Sk, causal, dens, G, fmt=2, seed=0):
    """Маска, ОБЩАЯ для группы из G подряд идущих блоков запросов одной головы.
    Зачем: замер на 250K показал, что при НЕЗАВИСИМОМ случайном отборе плотность 0.835 даёт 0.93x,
    то есть УБЫТОК, а при отборе, ОДИНАКОВОМ у всех блоков запросов, та же плотность 0.60 даёт
    1.64x -- почти идеал. Разница не в числе плиток (оно то же), а в том, читают ли одновременно
    идущие задания ОДНИ И ТЕ ЖЕ плитки ключей: K/V одной головы на 250K это 256 МБ против 6 МБ L2,
    и переиспользование между заданиями -- единственное, что спасает от похода в DRAM.
    G -- РАЗМЕР ГРУППЫ, ось замера: G=1 независимо, G=nqb общая на всю голову."""
    nqb = (Sq + BQ - 1) // BQ
    nkb = (Sk + BK - 1) // BK
    ld = nkb if fmt == 1 else (nkb + 31) // 32
    m = np.zeros((B * H * nqb, ld), dtype=np.int32)
    c = np.zeros(B * H * nqb, dtype=np.int32)
    for b in range(B):
        for h in range(H):
            for g0 in range(0, nqb, G):
                g1 = min(g0 + G, nqb)
                nmax = nkbC_of(g1 - 1, Sq, Sk, causal)
                rng = np.random.default_rng(seed + 1000 * h + g0)
                perm = rng.permutation(np.arange(1, max(2, nmax - 1), dtype=np.int64))
                for qb in range(g0, g1):
                    n = nkbC_of(qb, Sq, Sk, causal)
                    t = max(2 if n >= 2 else n, min(int(round(dens * n)), n))
                    if t >= n:
                        keep = np.arange(n, dtype=np.int64)
                    else:
                        cand = perm[perm < n - 1][:t - 2]
                        keep = np.sort(np.concatenate([[0], cand, [n - 1]]))
                    row = (b * H + h) * nqb + qb
                    if fmt == 1:
                        m[row, :len(keep)] = keep
                    else:
                        np.bitwise_or.at(m[row], keep // 32, (np.int32(1) << (keep % 32).astype(np.int32)))
                    c[row] = len(keep)
    return torch.from_numpy(m).cuda(), torch.from_numpy(c).cuda(), c


def build_mask(B, H, Sq, Sk, causal, dens_bh, fmt=1, const_count=None, seed=0, pattern="random"):
    """dens_bh -- матрица [B,H] плотностей. const_count -- если задан, вместо доли берётся
    ФИКСИРОВАННОЕ число плиток (min(nkbC, const_count)): нагрузка становится ровной, и это
    отдельный режим, нужный чтобы ОТДЕЛИТЬ неравномерность от самой разреженности."""
    rng = np.random.default_rng(seed)
    nqb = (Sq + BQ - 1) // BQ
    nkb = (Sk + BK - 1) // BK
    ld = nkb if fmt in (1, 3) else (nkb + 31) // 32
    m = np.zeros((B * H * nqb, ld), dtype=np.int32)
    c = np.zeros(B * H * nqb, dtype=np.int32)
    for b in range(B):
        for h in range(H):
            for qb in range(nqb):
                n = nkbC_of(qb, Sq, Sk, causal)
                t = const_count if const_count is not None else int(round(dens_bh[b][h] * n))
                t = max(2 if n >= 2 else n, min(t, n))
                keep = _pick(n, t, pattern, rng)
                row = (b * H + h) * nqb + qb
                if fmt in (1, 3):
                    m[row, :len(keep)] = keep
                else:
                    np.bitwise_or.at(m[row], keep // 32, (np.int32(1) << (keep % 32).astype(np.int32)))
                c[row] = len(keep)
    return torch.from_numpy(m).cuda(), torch.from_numpy(c).cuda(), c


def jobq_from_counts(cnt, B, H, nqb, order="lpt"):
    """Очередь заданий.
      grid       -- ТОЧНО тот порядок, что даёт трёхмерная сетка с REV=1 (голова снаружи, блоки
                    запросов от тяжёлого к легкому). КОНТРОЛЬ: если время совпало с сеткой, значит
                    сам механизм очереди бесплатен и всякая разница дальше -- это ПОРЯДОК.
      lpt        -- глобально по убыванию веса. Классическая жадность на makespan.
      lpt_inhead -- по убыванию веса ВНУТРИ головы, головы не перемешиваются. Существует потому,
                    что глобальный LPT собирает в одну волну по одному заданию от КАЖДОЙ головы,
                    и в L2 оказывается H разных K/V вместо одной."""
    rows = np.arange(len(cnt))
    qb = rows % nqb
    bh = rows // nqb
    if order == "grid":
        key = np.lexsort((-qb, bh))
    elif order == "lpt":
        key = np.argsort(-cnt, kind="stable")
    elif order == "lpt_inhead":
        key = np.lexsort((-cnt, bh))
    else:
        key = np.argsort(cnt, kind="stable")
    job = (qb[key].astype(np.int64) << 12) | bh[key].astype(np.int64)
    return torch.from_numpy(job.astype(np.int32)).cuda()


def makespan_eff(cnt, nsm=NSM, order=None):
    """Модель волн: nsm слотов, задание весит cnt плиток, жадный список-планировщик.
    Возвращает (эффективность машины, длина хвоста в долях)."""
    w = np.asarray(cnt, dtype=np.float64)
    if order is not None: w = w[order]
    slots = np.zeros(nsm)
    for x in w:
        i = int(np.argmin(slots)); slots[i] += x
    ms = slots.max()
    return float(w.sum() / (nsm * ms)), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,12,8192,65536")   # B,H,Sq,Sk
    ap.add_argument("--hkv", type=int, default=2)
    ap.add_argument("--it", type=int, default=10)
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-deploy", action="store_true")
    a = ap.parse_args()
    B, H, Sq, Sk = [int(x) for x in a.shape.split(",")]
    Hkv = a.hkv
    ext = fa2_sm70._ext.sparse_ext()
    dev = "cuda"
    sc = 1.0 / math.sqrt(D)
    nqb = (Sq + BQ - 1) // BQ
    nkb = (Sk + BK - 1) // BK
    out = {"shape": [B, H, Sq, Sk, Hkv]}

    print("СОСТОЯНИЕ КАРТ ДО ЗАМЕРА: " + gpu_state())
    print(f"=== forward d=256 causal, B={B} H={H} Hkv={Hkv} Sq={Sq} Sk={Sk}  "
          f"(блоков запросов {nqb}, плиток ключей {nkb}, заданий {B*H*nqb})")
    torch.manual_seed(0)
    q = (torch.randn(B, Sq, H, D, device=dev, dtype=torch.float16) * 0.3).contiguous()
    k = (torch.randn(B, Hkv, Sk, D, device=dev, dtype=torch.float16) * 0.3).contiguous()
    v = (torch.randn(B, Hkv, Sk, D, device=dev, dtype=torch.float16) * 0.3).contiguous()

    flop = 0
    for qb in range(nqb):
        flop += 4 * BQ * nkbC_of(qb, Sq, Sk, True) * BK * D * H * B
    print(f"    плотных FLOP по плиткам: {flop/1e12:.3f} T")

    # ---- 0/1. ПОБИТОВОЕ равенство и цена механизма при плотности 100%
    print("\n--- 0/1. Плотность 100%: побитовость и цена механизма (платится всегда) ---")
    ones = np.ones((B, H))
    ml1, mc1, cnt1 = build_mask(B, H, Sq, Sk, True, ones, fmt=1)
    ml2, mc2, _ = build_mask(B, H, Sq, Sk, True, ones, fmt=2)
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    o_ref, l_ref = ext.fwd_ref(q, k, v, sc, True)
    variants = {
        "отгруженное ядро (механизма нет)": lambda: ext.fwd_ref(q, k, v, sc, True),
        "то же ядро, sp=0 (рефакторинг цикла)": lambda: ext.fwd_sparse(q, k, v, sc, True, 0, empty, empty, empty),
        "sp=1 СПИСОК, плотность 100%": lambda: ext.fwd_sparse(q, k, v, sc, True, 1, ml1, mc1, empty),
        "sp=3 СПИСОК с чтением ВПЕРЁД, 100%": lambda: ext.fwd_sparse(q, k, v, sc, True, 3, ml1, mc1, empty),
        "sp=2 БИТОВАЯ КАРТА, плотность 100%": lambda: ext.fwd_sparse(q, k, v, sc, True, 2, ml2, mc2, empty),
    }
    for name, fn in variants.items():
        o, l = fn()
        bit = torch.equal(o, o_ref) and torch.equal(l, l_ref)
        rel = ((o.float() - o_ref.float()).norm() / o_ref.float().norm()).item()
        print(f'  {name:<38} побитово={bit}  relL2={rel:.1e}')
    r100 = measure(variants, rounds=5, reps=2)
    for name, (t, rat) in r100.items():
        print(f'  {name:<38} мин {t*1e3:8.3f} мс   отношение к плотному {rat:6.4f}  '
              f'(цена {(rat-1)*100:+6.2f}%)')
    t100 = {n: v[0] for n, v in r100.items()}
    out["t100"] = t100
    out["r100"] = {n: v[1] for n, v in r100.items()}
    tdense = t100["отгруженное ядро (механизма нет)"]
    print(f"    объём маски: список {ml1.numel()*4/2**20:.1f} МБ, карта {ml2.numel()*4/2**20:.1f} МБ, "
          f"то есть карта в {ml1.numel()/ml2.numel():.0f} раз меньше")

    # ---- 2. Цена сводок и отбора
    print("\n--- 2. Сводки плиток и отбор блоков (тензорные ядра, тот же оператор) ---")
    kbar, rmax = ext.block_summary(k, 0)
    t_sum = bench(lambda: ext.block_summary(k, 0), a.it)
    # в чанковом префилле сводки НОВЫХ ключей: k0 = nkb - Sq/BK
    k0 = max(0, nkb - Sq // BK)
    t_sum_inc = bench(lambda: ext.block_summary(k, k0), a.it)
    dens_t = torch.full((B, H), 0.60, dtype=torch.float32, device=dev)
    t_sel = bench(lambda: ext.select(q, kbar, rmax, dens_t, Sk, Hkv, sc, True, 0.0, 1, 1), a.it)
    km_s, kc_s = ext.select(q, kbar, rmax, dens_t, Sk, Hkv, sc, True, 0.0, 1, 1)
    cs = kc_s.cpu().numpy()
    dn = np.array([nkbC_of(qb, Sq, Sk, True) for qb in range(nqb)] * (B * H))
    print(f"  сводки (ВЕСЬ контекст)          {t_sum*1e3:8.3f} мс = {t_sum/tdense*100:5.2f}% внимания")
    print(f"  сводки (ТОЛЬКО новый чанк)      {t_sum_inc*1e3:8.3f} мс = {t_sum_inc/tdense*100:5.2f}% внимания  <- так в сервере")
    print(f"  отбор плиток                    {t_sel*1e3:8.3f} мс = {t_sel/tdense*100:5.2f}% внимания")
    print(f"  МЕХАНИЗМ ЦЕЛИКОМ (чанк+отбор)   {(t_sum_inc+t_sel)*1e3:8.3f} мс = "
          f"{(t_sum_inc+t_sel)/tdense*100:5.2f}% внимания")
    print(f"  проверка отбора: заданная плотность 0.600, полученная {cs.sum()/dn.sum():.4f} "
          f"(мин по строке {(cs/dn).min():.3f}, макс {(cs/dn).max():.3f})")
    out["t_summary_full"] = t_sum; out["t_summary_chunk"] = t_sum_inc; out["t_select"] = t_sel
    out["sel_density"] = float(cs.sum() / dn.sum())

    # отбор действительно работает на ядре? прогоняем внимание по ЕГО маске
    o_sel, _ = ext.fwd_sparse(q, k, v, sc, True, 1, km_s, kc_s, empty)
    print(f"  внимание по маске отбора: конечно={torch.isfinite(o_sel).all().item()}, "
          f"relL2 к плотному {(o_sel.float()-o_ref.float()).norm().item()/o_ref.float().norm().item():.2e}")

    # ---- 3. Выигрыш при плотности; РАСКЛАДКА оставленных плиток как отдельная ось
    print("\n--- 3. Выигрыш при плотности. Битовая карта (она дешевле, см. пункт 1) ---")
    print(f'{"режим":<44} {"мс":>8} {"x к плотному":>13} {"идеал x":>8} {"добор":>7} {"машина":>7}')
    rows = []
    lay = layer_densities()
    ph = np.array([[lay[i % len(lay)][1] for i in range(H)] for _ in range(B)])
    ncnt = int(round(0.60 * float(np.mean([nkbC_of(qb, Sq, Sk, True) for qb in range(nqb)]))))
    cases = [
        (f"ЕДИНАЯ 0.75, набор случайный", np.full((B, H), 0.75), None, "random"),
        (f"ЕДИНАЯ 0.60, набор случайный", np.full((B, H), 0.60), None, "random"),
        (f"ЕДИНАЯ 0.60, набор СГУСТКАМИ по 8", np.full((B, H), 0.60), None, "cluster8"),
        (f"ЕДИНАЯ 0.60, набор РОВНЫМ ШАГОМ", np.full((B, H), 0.60), None, "stride"),
        (f"ЕДИНАЯ 0.60, НЕПРЕРЫВНЫЙ ХВОСТ (окно)", np.full((B, H), 0.60), None, "recent"),
        (f"ЕДИНАЯ 0.60, НЕПРЕРЫВНОЕ НАЧАЛО", np.full((B, H), 0.60), None, "first"),
        (f"ПОГОЛОВНАЯ 0.52..0.90 (среднее {ph.mean():.3f})", ph, None, "random"),
        (f"ЕДИНАЯ = среднему поголовной {ph.mean():.3f}", np.full((B, H), ph.mean()), None, "random"),
        (f"ФИКС. число плиток {ncnt} (нагрузка РОВНАЯ)", None, ncnt, "random"),
    ]
    # ВСЕ режимы мерятся В ОДНОМ ЧЕРЕДУЮЩЕМСЯ НАБОРЕ вместе с плотным: только так отношения
    # «x к плотному» не зависят от того, что делал сосед на карте в эту минуту.
    keep = {}
    variants2 = {"плотное (механизма нет)": lambda: ext.fwd_ref(q, k, v, sc, True)}
    for name, dm, cc, pat in cases:
        ml, mc, cnt = build_mask(B, H, Sq, Sk, True, dm if dm is not None else np.ones((B, H)),
                                 fmt=2, const_count=cc, pattern=pat)
        keep[name] = (ml, mc, cnt)
        variants2[name] = (lambda m_=ml, c_=mc: ext.fwd_sparse(q, k, v, sc, True, 2, m_, c_, empty))
    r3 = measure(variants2, rounds=4, reps=2)
    for name, dm, cc, pat in cases:
        ml, mc, cnt = keep[name]
        frac = cnt.sum() / dn.sum()
        t, rat = r3[name]
        sp = 1.0 / rat
        eff_grid, _ = makespan_eff(cnt)
        print(f'  {name:<42} {t*1e3:8.3f} {sp:12.3f}x {1/frac:7.3f}x '
              f'{sp/(1/frac)*100:6.1f}% {eff_grid*100:6.1f}%')
        rows.append({"case": name, "pattern": pat, "dens": float(frac), "t": t,
                     "speedup": sp, "ideal": 1 / frac, "eff_grid": eff_grid})

    # ---- 4/5. ПОРЯДОК ВЫДАЧИ ЗАДАНИЙ против НЕРАВНОМЕРНОЙ НАГРУЗКИ. Берём режим, где вес заданий
    # РАЗНЫЙ по головам (боевой случай), и сравниваем четыре порядка. «grid» -- КОНТРОЛЬ: та же
    # перестановка, что даёт трёхмерная сетка, значит его время обязано совпасть с сеткой, иначе
    # дорога сама очередь, а не порядок.
    print("\n--- 4/5. НЕРАВНОМЕРНОСТЬ: порядок выдачи заданий (поголовная плотность) ---")
    nm = [c[0] for c in cases if c[0].startswith("ПОГОЛОВНАЯ")][0]
    ml, mc, cnt = keep[nm]
    _r = np.arange(len(cnt))
    _perm = {"grid": np.lexsort((-(_r % nqb), _r // nqb)),
             "lpt": np.argsort(-cnt, kind="stable"),
             "lpt_inhead": np.lexsort((-cnt, _r // nqb))}
    v4 = {"сетка 3D (как отгружено)": (lambda: ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, empty))}
    jqs = {}
    for od in ("grid", "lpt_inhead", "lpt"):
        jqs[od] = jobq_from_counts(cnt, B, H, nqb, od)
        v4["очередь: " + od] = (lambda j_=jqs[od]: ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, j_))
    r4 = measure(v4, rounds=4, reps=2)
    o_g, _ = ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, empty)
    o_q, _ = ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, jqs["grid"])
    print(f"  очередь даёт ПОБИТОВО то же, что сетка: {torch.equal(o_g, o_q)}")
    print(f"  разброс веса заданий: мин {cnt.min()} макс {cnt.max()} среднее {cnt.mean():.0f} "
          f"(макс/среднее = {cnt.max()/cnt.mean():.2f})")
    for name, (t, rat) in r4.items():
        eff = makespan_eff(cnt, order=_perm[name.split(": ")[-1]])[0] if ":" in name else makespan_eff(cnt)[0]
        print(f'  {name:<42} {t*1e3:8.3f} мс  к сетке {rat:6.4f}  модель машины {eff*100:5.1f}%')
    out["order"] = {n: v[1] for n, v in r4.items()}
    for k_ in list(keep): del keep[k_]
    del jqs; torch.cuda.empty_cache()
    out["cases"] = rows
    print("  «добор» = какую долю ИДЕАЛЬНОГО (1/плотность) выигрыша механизм донёс до конца.")
    print("  «машина» = модель волн: суммарная работа / (80 SM * makespan).")

    # ---- 6. РАЗЛОЖЕНИЕ НЕДОБОРА. Идеал 1/p молча считает, что время пропорционально числу плиток
    # ключей. Это НЕВЕРНО: у задания есть постоянная часть -- заливка Q в разделяемую, эпилог (запись
    # BQ x D выхода и LSE), запуск. Она НЕ сокращается разреженностью и ставит асимптоту любому
    # выигрышу. Сметаем её из замера, а не из головы: линейная подгонка t(n) = F + c*n по свипу.
    print("\n--- 6. Свип плотности: сколько недобора -- ПОСТОЯННАЯ часть задания, а не механизм ---")
    ps, ts_, ns_ = [], [], []
    for p in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        ml, mc, cnt = build_mask(B, H, Sq, Sk, True, np.full((B, H), p), fmt=2, pattern="random")
        t = bench(lambda: ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, empty), a.it)
        ps.append(cnt.sum() / dn.sum()); ts_.append(t); ns_.append(cnt.sum() / (B * H * nqb))
        del ml, mc; torch.cuda.empty_cache()
    A = np.vstack([np.ones(len(ns_)), np.array(ns_)]).T
    F, cpb = np.linalg.lstsq(A, np.array(ts_), rcond=None)[0]
    print(f'{"плотность":>10} {"мс":>9} {"x к плотному":>13} {"идеал x":>8} {"добор":>7} {"модель F+c*n":>13}')
    for p, t, n in zip(ps, ts_, ns_):
        print(f'{p:>10.3f} {t*1e3:9.3f} {tdense/t:12.3f}x {1/p:7.3f}x {tdense/t/(1/p)*100:6.1f}% '
              f'{(F+cpb*n)*1e3:12.3f}')
    print(f"  ПОДГОНКА: постоянная часть задания F = {F*1e3:.3f} мс = {F/tdense*100:.1f}% плотного времени,")
    print(f"            плитка ключей c = {cpb*1e6:.3f} мкс. Асимптота выигрыша при плотности p:")
    for p in (0.75, 0.60, 0.50):
        n0 = float(np.mean([nkbC_of(qb, Sq, Sk, True) for qb in range(nqb)]))
        print(f"            p={p:.2f} -> потолок {(F+cpb*n0)/(F+cpb*n0*p):.3f}x при идеале {1/p:.3f}x")
    out["fit"] = {"F": float(F), "c_per_block": float(cpb), "sweep_p": [float(x) for x in ps],
                  "sweep_t": ts_}

    # ---- 7. ГЛАВНАЯ ОСЬ, найденная замером: НАСКОЛЬКО СОГЛАСОВАНЫ маски соседних блоков запросов.
    # Число оставленных плиток во всех строках таблицы ОДНО И ТО ЖЕ. Меняется только то, читают ли
    # одновременно идущие задания одни и те же плитки ключей.
    print("\n--- 7. СОГЛАСОВАННОСТЬ МАСОК соседних блоков запросов (плотность 0.60, число плиток то же) ---")
    v7 = {"плотное (механизма нет)": lambda: ext.fwd_ref(q, k, v, sc, True)}
    hold, cnts = {}, {}
    Gs = [g for g in (1, 4, 16, 64, 256, 1024) if g <= nqb] + ([nqb] if nqb not in (1, 4, 16, 64, 256, 1024) else [])
    for G in Gs:
        ml, mc, cnt = build_mask_shared(B, H, Sq, Sk, True, 0.60, G, fmt=2)
        hold[G] = (ml, mc); cnts[G] = cnt
        v7[f"общая маска на группу G={G:<4}"] = (lambda m_=ml, c_=mc:
                                                 ext.fwd_sparse(q, k, v, sc, True, 2, m_, c_, empty))
    r7 = measure(v7, rounds=4, reps=2)
    print(f'{"группа":>28} {"мс":>9} {"x к плотному":>13} {"идеал x":>8} {"добор":>7}')
    sh = []
    for G in Gs:
        t, rat = r7[f"общая маска на группу G={G:<4}"]
        frac = cnts[G].sum() / dn.sum()
        print(f'{("G=" + str(G)):>28} {t*1e3:9.3f} {1/rat:12.3f}x {1/frac:7.3f}x {(1/rat)/(1/frac)*100:6.1f}%')
        sh.append({"G": G, "dens": float(frac), "speedup": 1 / rat})
    out["share"] = sh
    print("  G=1 -- каждый блок запросов отбирает НЕЗАВИСИМО (худший случай для L2);")
    print(f"  G={nqb} -- одна маска на всю голову (лучший). Разница -- ЦЕНА НЕСОГЛАСОВАННОСТИ.")
    for k_ in list(hold): del hold[k_]
    torch.cuda.empty_cache()

    print("\nСОСТОЯНИЕ КАРТ ПОСЛЕ ЗАМЕРА: " + gpu_state())
    out["gpu_after"] = gpu_state()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
        print(f"\nсохранено в {a.json}")


if __name__ == "__main__":
    main()
