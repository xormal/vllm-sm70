"""Гейт родной поправки Clef (fa2sm70_clef.delta) против независимого эталона в fp32.

Эталон строится ДРУГИМ путём, чем сборщик: полные HF-матрицы разности D = B·A (Clef) + B·A (обратная
аблитерация) в fp32, затем нарезка по семантике weight_loader vLLM (часть слитого выхода -> narrow по
рангу; row-parallel -> свои столбцы входа). Сборщик же режет МНОЖИТЕЛИ. Совпадение двух путей и есть
проверка раскладки.

Проверяется: строки Clef = эталон до округления fp16; строки вне сегментов -- ПОБАЙТОВО нетронуты;
шаг без сегментов -- побайтово ничего; кольцо меньше числа подкачиваемых слоёв (замещение) на
нескольких шагах подряд.
"""
import argparse, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from fa2_sm70._ext import clef_ext  # noqa: E402
import clef_sborka as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adaptery", default=os.environ.get("FA2SM70_CLEF_DIR", "clef"))
ap.add_argument("--rang", type=int, default=0)
ap.add_argument("--sloi", default="2,3,40,43,63")
ap.add_argument("--nslot", type=int, default=2)
ap.add_argument("--shagov", type=int, default=3)
ap.add_argument("--M", type=int, default=600)
args = ap.parse_args()

dev = torch.device("cuda")
r, tp = args.rang, 2
sloi = [int(x) for x in args.sloi.split(",")]
LT = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)]
ext = clef_ext()
sob = S.sobrat(f"{args.adaptery}/clef_lora_r256_fp16.safetensors", f"{args.adaptery}/deabl_abl_fp16.safetensors",
               r, tp, sloi=set(sloi), layer_types=LT)
S.ustanovit(ext, sob, nslot=args.nslot, max_rows=2048)
print("слои:", [(L, st, h.numel() // 1000, tuple(o.shape)) for L, st, h, o in sob], ext.stats())

clef = S.Src(f"{args.adaptery}/clef_lora_r256_fp16.safetensors")
deabl = S.Src(f"{args.adaptery}/deabl_abl_fp16.safetensors")


def D(L, name):
    """полная HF-разность модуля в fp32 (или None)"""
    p = f"{S.PFX}{L}.{name}"
    acc = None
    for src, a, b in ((clef, ".lora_A", ".lora_B"), (deabl, ".deabl_A", ".deabl_B")):
        A, B = src.get(p + a), src.get(p + b)
        if A is not None:
            t = B.to(dev).float() @ A.to(dev).float()
            acc = t if acc is None else acc + t
    dd = clef.get(p + ".delta")
    if dd is not None:
        acc = dd.to(dev).float()
    return acc


def loc_rows(t):
    n = t.shape[0] // tp
    return t[r * n:(r + 1) * n]


def etalon(L):
    """{mod: W_loc [cols, in]} для ранга r"""
    out = {}
    if LT[L] == "full_attention":
        q, k, v = D(L, "self_attn.q_proj"), D(L, "self_attn.k_proj"), D(L, "self_attn.v_proj")
        if q is not None:
            out[S.MOD["qkv"]] = torch.cat([loc_rows(q), loc_rows(k), loc_rows(v)])
        o = D(L, "self_attn.o_proj")
    else:
        qkv, z = D(L, "linear_attn.in_proj_qkv"), D(L, "linear_attn.in_proj_z")
        if qkv is not None:
            qq, kk, vv = qkv[:2048], qkv[2048:4096], qkv[4096:]
            out[S.MOD["qkv"]] = torch.cat([loc_rows(qq), loc_rows(kk), loc_rows(vv), loc_rows(z)])
        b, a = D(L, "linear_attn.in_proj_b"), D(L, "linear_attn.in_proj_a")
        if b is not None:
            out[S.MOD["ba"]] = torch.cat([loc_rows(b), loc_rows(a)])
        o = D(L, "linear_attn.out_proj")
    if o is not None:
        n = o.shape[1] // tp
        out[S.MOD["out"]] = o[:, r * n:(r + 1) * n]
    g, u = D(L, "mlp.gate_proj"), D(L, "mlp.up_proj")
    if g is not None:
        out[S.MOD["gate_up"]] = torch.cat([loc_rows(g), loc_rows(u)])
    d = D(L, "mlp.down_proj")
    if d is not None:
        n = d.shape[1] // tp
        out[S.MOD["down"]] = d[:, r * n:(r + 1) * n]
    return out


ET = {L: etalon(L) for L in sloi}
M = args.M
SEGS = torch.tensor([[37, 300, 0, 0, 0], [400, 150, 1, 0, 0]], dtype=torch.int32)
mask = torch.zeros(M, dtype=torch.bool, device=dev)
for row0, rows, *_ in SEGS.tolist():
    mask[row0:row0 + rows] = True
g = torch.Generator(device="cuda").manual_seed(1)
plokho = 0
for shag in range(args.shagov + 1):
    pustoj = shag == args.shagov                       # последний шаг -- без сегментов
    ext.set_step(torch.empty(0, 5, dtype=torch.int32) if pustoj else SEGS)
    for L in sloi:
        for mod, Wl in ET[L].items():
            x = (torch.randn(M, Wl.shape[1], generator=g, device=dev) * 0.5).half()
            y0 = (torch.randn(M, Wl.shape[0], generator=g, device=dev) * 2).half()
            y = y0.clone()
            ext.delta(x, y, L, mod)
            if pustoj:
                ok = torch.equal(y, y0)
                print(f"шаг {shag} (пустой) L={L} mod={mod}: {'НЕТРОНУТО' if ok else 'ИЗМЕНЕНО!'}")
                plokho += not ok
                continue
            ref = y0.float() + x.float() @ Wl.t()
            # (а) точность самой поправки: тот же x при y0 = 0
            z = torch.zeros_like(y0)
            ext.delta(x, z, L, mod)
            dref = (x.float() @ Wl.t())[mask]
            err_d = (z.float()[mask] - dref).norm() / dref.norm()
            vne_z = bool((z[~mask] == 0).all())
            # (б) с ненулевым y0: ошибка против неустранимого пола -- округления точного ответа в fp16
            e_nat = (y.float()[mask] - ref[mask]).norm()
            e_pol = (ref[mask].half().float() - ref[mask]).norm()
            vne = torch.equal(y[~mask], y0[~mask])
            good = err_d < 1.5e-3 and e_nat < 1.6 * e_pol and vne and vne_z
            plokho += not good
            print(f"шаг {shag} L={L:2d} mod={mod} [{Wl.shape[0]:5d}x{Wl.shape[1]:5d}]: поправка отн. {err_d:.2e}; "
                  f"с y0: ошибка/пол fp16 {e_nat / e_pol:.2f}; вне сегментов {'побайтово' if vne and vne_z else 'ИЗМЕНЕНО!'}"
                  f"{'' if good else '  <-- ПЛОХО'}")
torch.cuda.synchronize()
print("stats:", ext.stats(), "ошибка:", ext.last_error() or "-")
print("ИТОГ:", "ПРОЙДЕН" if plokho == 0 and ext.stats()["err"] == 0 else f"НЕ ПРОЙДЕН ({plokho})")
