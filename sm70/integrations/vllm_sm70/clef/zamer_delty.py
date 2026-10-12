"""Время и память родной поправки на полном наборе слоёв (2-63) для одного ранга, кусок M строк Clef.

Режимы: кольцо (--nslot 3) и всё на карте (--nslot 24) -- разность = цена подкачки без укрытия.
Порядок вызовов -- как в слое: вход (0, 1) -> выход (2) -> gate_up (3) -> down (4).
Без базового счёта слоя копии здесь НЕ прячутся: в бою их закрывает счёт самого слоя (~10+ мс на кусок 2048).
"""
import argparse, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from fa2_sm70._ext import clef_ext  # noqa: E402
import clef_sborka as S  # noqa: E402
import pynvml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adaptery", default=os.environ.get("FA2SM70_CLEF_DIR", "clef"))
ap.add_argument("--rang", type=int, default=0)
ap.add_argument("--nslot", type=int, default=3)
ap.add_argument("--M", type=int, default=2048)
ap.add_argument("--povtorov", type=int, default=5)
args = ap.parse_args()
dev = torch.device("cuda")
pynvml.nvmlInit()
_nv = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))


def used():
    torch.cuda.synchronize()
    return next(p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_nv) if p.pid == os.getpid())


LT = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)]
ext = clef_ext()
torch.zeros(1, device=dev)
sob = S.sobrat(f"{args.adaptery}/clef_lora_r256_fp16.safetensors", f"{args.adaptery}/deabl_abl_fp16.safetensors",
               args.rang, 2, layer_types=LT)
# входы/выходы на каждую форму -- выделяются ДО опоры (это активации самой модели)
shapes = {}
for L, st, host, ops in sob:
    for mod, typ, off, n, inn, col in ops.tolist():
        if typ in (0, 2):
            shapes.setdefault((L, mod), [inn, 0])
        if typ in (1, 2):
            shapes[(L, mod)][1] = max(shapes[(L, mod)][1], col + n)
bufs = {}
for (L, mod), (inn, cols) in shapes.items():
    key = (inn, cols)
    if key not in bufs:
        bufs[key] = ((torch.randn(args.M, inn, device=dev) * 0.5).half(), torch.zeros(args.M, cols, device=dev, dtype=torch.half))
torch.cuda.empty_cache()
u0 = used()
S.ustanovit(ext, sob, nslot=args.nslot, max_rows=2048)
u1 = used()
st = ext.stats()
print(f"ранг {args.rang}: слоёв {st['layers']} (подкачиваемых {st['streamed']}), ячейка {st['slot_bytes'] / 1e6:.1f} МБ x "
      f"{st['nslot']}; на карте после установки {(u1 - u0) / 1e6:.1f} МБ")
seg = torch.tensor([[0, args.M, 0, 0, 0]], dtype=torch.int32)
order = sorted(shapes)
tt = []
for i in range(args.povtorov + 1):
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.time()
    ev0.record()
    ext.set_step(seg)
    for (L, mod) in order:
        x, y = bufs[tuple(shapes[(L, mod)])]
        ext.delta(x, y, L, mod)
    ev1.record()
    torch.cuda.synchronize()
    if i:
        tt.append(ev0.elapsed_time(ev1))
u2 = used()
tt.sort()
print(f"шаг поправки на {args.M} строк: медиана {tt[len(tt) // 2]:.1f} мс (мин {tt[0]:.1f}); "
      f"на карте всего {(u2 - u0) / 1e6:.1f} МБ; stats {ext.stats()} {ext.last_error() or ''}")
