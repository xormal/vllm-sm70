"""Модульный тест blk_mean_update: таблица средних после шага ДЕКОДА 16 запросов x 4 слота.

  FA2SM70_BLKMEAN_ALL=0|1 CUDA_VISIBLE_DEVICES=0 python tests/test_blkmean_update.py [корень репозитория]
Эталон -- точное среднее (и размах) каждого тронутого подблока по пулу в torch. Печатает, сколько
тронутых подблоков совпали с эталоном. При BLKMEAN_ALL=0 (как было) ждём совпадения только у
первого запроса каждого среза из 64 слотов; при 1 -- у всех.
"""
import os
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fa2_sm70 import _ext  # noqa: E402

ext = _ext.prefill_ext()
dev = "cuda"
torch.manual_seed(0)
NB, BS, HKV, D, B = 24, 16384, 2, 256, 64
K = torch.randint(-127, 128, (NB, BS, HKV, D), dtype=torch.int8, device=dev)
Ks = (torch.rand(NB * BS * HKV, device=dev) * 0.02 + 0.001).float()
SUB = BS // B
kbar = torch.zeros(NB * SUB, HKV, D, dtype=torch.float16, device=dev)
devp = torch.zeros_like(kbar)


def ref(pg, sub):
    rows = torch.arange(sub * B, (sub + 1) * B, device=dev)
    k = K[pg, rows].float()                                     # [B, HKV, D]
    sc = Ks.view(NB, BS, HKV)[pg, rows].unsqueeze(-1)          # [B, HKV, 1]
    v = k * sc
    mu = v.mean(0)
    dv = torch.maximum(v.amax(0) - mu, mu - v.amin(0))
    return mu.half(), dv.half()


def run(slots, tag):
    kbar.zero_()
    devp.zero_()
    sm = torch.tensor(slots, dtype=torch.long, device=dev)
    ext.blk_mean_update(K, Ks, sm, kbar, B, devp)
    torch.cuda.synchronize()
    touched = sorted({(s // BS, (s % BS) // B) for s in slots if s >= 0})
    ok = 0
    for pg, sub in touched:
        mu, dv = ref(pg, sub)
        row = pg * SUB + sub
        if torch.allclose(kbar[row].float(), mu.float(), atol=2e-3) and \
           torch.allclose(devp[row].float(), dv.float(), atol=4e-3):
            ok += 1
    print(f"{tag}: тронуто подблоков {len(touched)}, совпало с эталоном {ok}")
    return ok, len(touched)


# 1) декод: 16 запросов x 4 слота подряд, каждый в своей странице и своём подблоке (как в бою)
dec = []
for r in range(16):
    pg, pos = r + 2, 1000 + 37 * r
    dec += [pg * BS + pos + j for j in range(4)]
ok1, n1 = run(dec, "декод 16x4 (один срез из 64 слотов)")
# 2) смешанный: 3 запроса декода, затем кусок префилла 300 позиций с позиции 1290 (не на границе)
mix = []
for r in range(3):
    mix += [(r + 2) * BS + 5000 + 64 * r + j for j in range(4)]
mix += [20 * BS + 1290 + t for t in range(300)]
ok2, n2 = run(mix, "смешанный: 3x4 декода + 300 префилла")
# 3) набивка -1 между запросами
pad = []
for r in range(8):
    pad += [(r + 2) * BS + 700 + j for j in range(4)] + [-1] * 4
ok3, n3 = run(pad, "декод 8x4 с набивкой -1")
print("ИТОГ:", "все тронутые подблоки верны" if (ok1, ok2, ok3) == (n1, n2, n3)
      else f"НЕВЕРНЫХ {n1 - ok1 + n2 - ok2 + n3 - ok3} из {n1 + n2 + n3}")
