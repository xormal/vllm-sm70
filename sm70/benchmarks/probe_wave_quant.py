# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ЗОНД: КВАНТОВАНИЕ ПО ВОЛНАМ И ПЕРЕКОС ПО SM В ПРЕФИЛЛЕ.
#
# РЕСУРС. Сетка префилла строится ОДИНАКОВО во всех трёх путях: dim3(ceil(Sq/BQ), H, B)
#   cutlass d<=128 : fa2_src/fmha_kernel/kernel_forward.h:451  (умножается ещё на num_splits)
#   fp16 d=256     : fa2_sm70/csrc/attn_fwd_cutlass.cu:655     (BQ=64,  smem 85504 Б -> 1 блок/SM)
#   байтовое/роли  : fa2_sm70/csrc/attn_fwd_cutlass.cu:853     (BQ=32,  smem 90688 Б -> 1 блок/SM)
# Число блоков N = ceil(Sq/BQ)*H*B*ns, число слотов S = 80*блоков_на_SM. Волн W = N/S.
# ДРОБНАЯ ЧАСТЬ W -- это последняя волна на НЕПОЛНОЙ машине. При однородных блоках (БЕЗ маски)
# время ровно ceil(W)/W от идеального: W=2.2 -> +36%, W=3.2 -> +25%, W=12.8 -> +1.6%.
#
# ПОЧЕМУ ЗОНД -- ФОРМЫ, А НЕ ПРАВКА. Квантование это свойство РАСПИСАНИЯ, а не фазы: снять его
# правкой нельзя, но можно ВЫБРАТЬ форму, где его нет (W целое), и форму, где оно максимально
# (W = k+0.05). Разность времени НА ЕДИНИЦУ РАБОТЫ между такими формами и есть цена квантования.
# Единственное, что при этом меняется, -- H (число голов): вес каждого блока при фиксированном
# Sq/Sk тождественен, суммарная работа строго пропорциональна H. Правка кода НЕ НУЖНА.
#
# ТРИ ЯРУСА (каждый отвечает на свой вопрос):
#   A. ЛЕСТНИЦА ВОЛН (causal=False). t/H как функция H. Предсказание: ступени с периодом
#      S/nq голов, амплитуда ceil(W)/W. Период ступени ИЗМЕРЯЕТ число слотов -- его не надо
#      брать на веру из cudaOccupancy.
#   B. ОТРЕЗОК (causal=False, Sq фиксировано, Sk растёт). N зависит ТОЛЬКО от Sq, работа мейнлупа
#      ~ Sk. Линейная подгонка t = a + b*Sk: a -- ПОСТОЯННАЯ ЦЕНА НА БЛОК (пролог Q, эпилог O,
#      запуск/слив волны), которую НЕ ВИДИТ НИ ОДИН фазовый фальсификатор (он снимает работу
#      внутри цикла по ключам). Именно a -- кандидат в ту самую ДЫРУ 22.3%, а не квантование.
#   C. ПОРЯДОК (causal=True, FA2SM70_FWD_REV=0/1 -- переключатель УЖЕ ЕСТЬ, читается один раз на
#      процесс, поэтому запускается дочерним процессом). Симулятор внизу предсказывает отношение
#      ДО замера; совпадение легализует третью оценку -- сколько ещё лежит в глобальном LPT
#      (очередь заданий JOBQ в volta_fwd_block.cuh:98 собрана, но НЕ инстанцирована).
#
# ЗАПУСК (на ЧИСТОЙ карте, частоты зафиксированы):
#   sudo nvidia-smi -i 0 -lgc 1380,1380
#   CUDA_VISIBLE_DEVICES=0 python benchmarks/probe_wave_quant.py --path all
#   sudo nvidia-smi -i 0 -rgc
import argparse, json, math, os, subprocess, sys, time, heapq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SLOTS_DEFAULT = {"vfp16": 80, "ws": 80, "cutlass128": 240}
BQ = {"vfp16": 64, "ws": 32, "cutlass128": 32}
BK = {"vfp16": 64, "ws": 64, "cutlass128": 128}


# ---------------------------------------------------------------- симулятор расписания
def weights(nq, bq, bk, Sq, Sk, causal):
    off = Sk - Sq
    return [max(0, -(-min(Sk, i * bq + off + bq) // bk)) if causal else -(-Sk // bk)
            for i in range(nq)]


def makespan(jobs, m):
    """Жадный список: блок уходит на первый освободившийся SM -- ровно то, что делает диспетчер."""
    h = [0.0] * m
    heapq.heapify(h)
    fin = []
    for w in jobs:
        t = heapq.heappop(h)
        heapq.heappush(h, t + w)
    while h:
        fin.append(heapq.heappop(h))
    return max(fin)


def predict(B, H, Sq, Sk, path, slots, causal, c=0.0):
    """Возвращает (W, ceil/W, ship/LB, asc/LB, LPT/LB). ship = отгруженный порядок (REV=1)."""
    bq, bk = BQ[path], BK[path]
    nq = -(-Sq // bq)
    w = [x + c for x in weights(nq, bq, bk, Sq, Sk, causal)]
    N = nq * H * B
    ship, asc = [], []
    for _ in range(B * H):
        ship += list(reversed(w)); asc += list(w)
    lpt = sorted(ship, reverse=True)
    LB = max(sum(ship) / slots, max(w))
    W = N / slots
    return (W, math.ceil(W) / W, makespan(ship, slots) / LB,
            makespan(asc, slots) / LB, makespan(lpt, slots) / LB)


# ---------------------------------------------------------------- обвязка замера
def gpu_state():
    q = "clocks.sm,clocks.mem,power.draw,temperature.gpu"
    try:
        s = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=10).stdout.strip().splitlines()
        return s, [x for x in p if x]
    except Exception:
        return "nvidia-smi unavailable", []


def bench(fn, it=30, wu=5):
    """ОДНА синхронизация на серию, а не на вызов: при 100-200 мкс на ядро посинхронный замер
    добавляет 10-20 мкс задержки запуска КАЖДОМУ вызову, и она бьёт по малым H сильнее, чем по
    большим -- то есть подделывает ровно ту лестницу, которую мы ищем."""
    import torch
    for _ in range(wu):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e6        # мкс на вызов


def qz(x):
    import torch
    a = x.abs().amax(-1, keepdim=True).float() / 127.
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8)
    return q.contiguous(), a.squeeze(-1).contiguous()


def make_call(path, B, H, Sq, Sk, causal, D):
    """Один вызов боевого ядра нужного пути. Входы аллоцируются ОДИН раз."""
    import torch, fa2_sm70
    ext = fa2_sm70._ext.prefill_ext()
    sc = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    if path == "cutlass128":                 # BSHD-вход, d=128 -> cutlass FMHA (kQ=32)
        q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16) * .3).contiguous()
        k = (torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float16) * .3).contiguous()
        v = (torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float16) * .3).contiguous()
        return lambda: ext.attn_fwd_qbshd(q, k, v, sc, causal)
    q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16) * .3).contiguous()
    k = (torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float16) * .3).contiguous()
    v = (torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float16) * .3).contiguous()
    if path == "vfp16":                      # рукописный мейнлуп d=256 (BQ=64)
        return lambda: ext.attn_fwd_volta(q, k, v, sc, causal)
    kq, ks = qz(k); vq, vs = qz(v)           # ядро ролей (BQ=32) -- то, на котором мерились фазы
    os.environ["FA2SM70_WS"] = "1"
    return lambda: ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, causal)


# ---------------------------------------------------------------- ярус A: лестница волн
def tier_A(path, slots, D, Sq, Hs, B=1, rounds=5):
    """causal=False: все блоки РАВНЫ, значит единственный источник нелинейности -- волны.
    Печатается t/H (работа строго ~H) и предсказание ceil(W)/W."""
    import torch
    nq = -(-Sq // BQ[path])
    fns = {H: make_call(path, B, H, Sq, Sq, False, D) for H in Hs}
    acc = {H: [] for H in Hs}
    for _ in range(rounds):                  # ЧЕРЕДОВАНИЕ ВНУТРИ РАУНДА
        for H in Hs:
            acc[H].append(bench(fns[H], it=20, wu=3))
    rows = []
    for H in Hs:
        t = sorted(acc[H])[len(acc[H]) // 2]
        W = nq * H * B / slots
        per = t / H
        rows.append(dict(H=H, N=nq * H * B, W=round(W, 3), t_us=round(t, 1),
                         t_per_head=round(per, 2), pred_ceilW=round(math.ceil(W) / W, 3)))
    b = min(r["t_per_head"] for r in rows)   # эталон = самая дешёвая голова (там волна целая)
    for r in rows:
        r["meas_overhead"] = round(r["t_per_head"] / b, 3)
    return rows


# ---------------------------------------------------------------- ярус B: постоянная цена блока
def tier_B(path, D, Sq, Sks, B=1, H=8, rounds=5):
    """causal=False, Sq фиксировано: N блоков ПОСТОЯННО, работа мейнлупа ~ Sk.
    t = a + b*Sk; a -- цена, которую фазовые фальсификаторы не видят."""
    fns = {Sk: make_call(path, B, H, Sq, Sk, False, D) for Sk in Sks}
    acc = {Sk: [] for Sk in Sks}
    for _ in range(rounds):
        for Sk in Sks:
            acc[Sk].append(bench(fns[Sk], it=20, wu=3))
    xs = [float(s) for s in Sks]
    ys = [sorted(acc[s])[len(acc[s]) // 2] for s in Sks]
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx); a = (sy - b * sx) / n
    return dict(a_us=round(a, 1), b_us_per_key=round(b, 5),
                points={s: round(t, 1) for s, t in zip(Sks, ys)},
                fixed_share_at={s: round(a / (a + b * s), 4) for s in Sks})


# ---------------------------------------------------------------- ярус C: порядок блоков
def tier_C(path, slots, D, shapes, rounds=5):
    """FA2SM70_FWD_REV читается ОДИН раз на процесс -> обе стороны в ДОЧЕРНИХ процессах."""
    out = []
    for (B, H, Sq) in shapes:
        ts = {}
        for rev in (1, 0):
            env = dict(os.environ, FA2SM70_FWD_REV=str(rev), FA2SM70_FWD_SPLITS="1")
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child",
                                "--path", path, "--shape", f"{B},{H},{Sq}", "--dim", str(D),
                                "--rounds", str(rounds)],
                               capture_output=True, text=True, env=env)
            ts[rev] = json.loads(r.stdout.strip().splitlines()[-1])["t_us"]
        W, cw, ship, asc, lpt = predict(B, H, Sq, Sq, path, slots, True)
        out.append(dict(shape=[B, H, Sq], t_rev1=round(ts[1], 1), t_rev0=round(ts[0], 1),
                        meas_asc_over_ship=round(ts[0] / ts[1], 3),
                        pred_asc_over_ship=round(asc / ship, 3), W=round(W, 2),
                        pred_ship_over_LB=round(ship, 3), pred_lpt_over_LB=round(lpt, 3),
                        headroom_LPT=round(ship / lpt - 1, 3)))
    return out


# ------------------------------------------- ярус D: усечение сетки (нужна сборка -DFA2_FWD_GRIDCUT)
def tier_D(path, slots, D, B, H, Sq, rounds=5, causal=False):
    """ФАЛЬСИФИКАТОР ХВОСТА. Оставляем только первые k блоков в порядке выдачи; остальные выходят
    сразу. Ответ НЕВЕРЕН -- читается ТОЛЬКО время. Предсказание: время СТОИТ НА МЕСТЕ, пока
    вычёркиваются блоки последней (неполной) волны, и падает ступенькой, когда волна опустела.
    Ширина полки ИЗМЕРЯЕТ число слотов, высота ступени -- цену одной волны."""
    nq = -(-Sq // BQ[path])
    N = nq * H * B
    fn = make_call(path, B, H, Sq, Sq, causal, D)
    keeps = [N] + [N - i for i in range(1, min(N, 2 * slots + 8))]
    rows = []
    for _ in range(rounds):
        for k in keeps:
            os.environ["FA2SM70_FWD_CTA_KEEP"] = str(k)
            rows.append((k, bench(fn, it=20, wu=2)))
    os.environ.pop("FA2SM70_FWD_CTA_KEEP", None)
    agg = {}
    for k, t in rows:
        agg.setdefault(k, []).append(t)
    return [dict(keep=k, dropped=N - k, t_us=round(sorted(v)[len(v) // 2], 1))
            for k, v in sorted(agg.items(), reverse=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="all", choices=["all", "vfp16", "ws", "cutlass128"])
    ap.add_argument("--gridcut", action="store_true",
                    help="ярус D: нужна сборка FA2SM70_EXTRA_NVCC=-DFA2_FWD_GRIDCUT")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--shape", default="1,4,2048")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--slots", type=int, default=0, help="переопределить число слотов")
    a = ap.parse_args()
    if a.child:                              # дочерний: одно число, причинная маска
        B, H, Sq = (int(x) for x in a.shape.split(","))
        fn = make_call(a.path, B, H, Sq, Sq, True, a.dim)
        ts = sorted(bench(fn, it=20, wu=3) for _ in range(a.rounds))
        print(json.dumps({"t_us": ts[len(ts) // 2]}))
        return
    st, procs = gpu_state()
    print("GPU до:", st, "| чужих процессов:", len(procs), procs)
    assert len(procs) <= 1, "ЗАМЕР НЕДЕЙСТВИТЕЛЕН: на карте есть чужой процесс"
    if a.gridcut:
        p = a.path if a.path != "all" else "ws"
        D = 128 if p == "cutlass128" else 256
        r = tier_D(p, a.slots or SLOTS_DEFAULT[p], D, 1, 4, 2048, rounds=a.rounds)
        print(p, "D_gridcut", json.dumps(r, ensure_ascii=False))
        return
    res = {}
    paths = ["vfp16", "ws", "cutlass128"] if a.path == "all" else [a.path]
    for p in paths:
        slots = a.slots or SLOTS_DEFAULT[p]
        D = 128 if p == "cutlass128" else 256
        # Sq подобран так, чтобы nq делило слоты нацело: ступень приходится на ЦЕЛОЕ H.
        #   vfp16 : BQ=64,  Sq=1024 -> nq=16, 80/16 = 5 голов на волну
        #   ws    : BQ=32,  Sq=512  -> nq=16, 5 голов на волну
        #   cut128: BQ=32,  Sq=1920 -> nq=60, 240/60 = 4 головы на волну
        Sq = {"vfp16": 1024, "ws": 512, "cutlass128": 1920}[p]
        Hs = list(range(1, 17)) if p != "cutlass128" else list(range(1, 13))
        res[p] = {"slots_assumed": slots, "Sq": Sq,
                  "A_wave_staircase": tier_A(p, slots, D, Sq, Hs, rounds=a.rounds)}
        res[p]["B_fixed_cost"] = tier_B(p, D, Sq=2048 if p != "cutlass128" else 2048,
                                        Sks=[2048, 4096, 8192, 16384, 32768],
                                        H=8, rounds=a.rounds)
        res[p]["C_order"] = tier_C(p, slots, D,
                                   [(1, 4, 2048), (1, 4, 4096), (1, 16, 4096)], rounds=a.rounds)
        print(p, json.dumps(res[p], ensure_ascii=False, indent=1))
    st2, procs2 = gpu_state()
    print("GPU после:", st2, "| чужих процессов:", len(procs2))
    out = os.environ.get("RESULTS_PATH")
    if out:
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("saved:", out)


if __name__ == "__main__":
    main()
