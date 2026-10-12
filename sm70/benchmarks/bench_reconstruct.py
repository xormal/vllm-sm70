"""ЛИШНИЙ ПРОХОД ПЕРЕД cuBLAS: сколько стоит разворот веса и есть ли путь мимо него.

ОТКУДА ПРЕДМЕТ. Опись пути по ядрам (tools/opis_puti.py) на боевом префилле Qwen3.8 в конвейерном
режиме: `vllm::gptq::reconstruct_exllama_8bit_kernel` -- 464 пуска по 684 мкс = 6.5 % времени карты.
Это разворот int8-веса в ПЛОТНЫЙ fp16 отдельным проходом по HBM; следом cuBLAS читает его обратно.
Классический лишний проход, и он наш по природе -- ровно то, что мы убирали в W12/W8A16.

ЧТО СРАВНИВАЕТСЯ НА БОЕВЫХ ФОРМАХ (TP=1, конвейер -> полная ширина слоя):
  1. разворот + cuBLAS   -- то, что делает движок сейчас;
  2. наш упакованный путь -- вес потребляется как есть (w8a16_gemm_hmma);
  3. плотный cuBLAS       -- ПОЛ: столько стоило бы, будь вес уже развёрнут (разворота нет вовсе).
Пункт 3 обязателен: без него не видно, сколько из разницы -- сам разворот, а сколько -- умножение.
"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70

ФОРМЫ = [("mlp.gate/up", 17408, 5120), ("mlp.down", 5120, 17408),
         ("gdn.in_proj_qkv", 10240, 5120), ("gdn.out_proj", 5120, 6144)]
M, G = 4096, 128


def замер(f, n=7):
    for _ in range(3): f()
    torch.cuda.synchronize(); т=[]
    for _ in range(n):
        t0=time.perf_counter(); f(); torch.cuda.synchronize(); т.append(time.perf_counter()-t0)
    т.sort(); return sum(т[1:-1])/len(т[1:-1])*1e3


def gptq(w):
    """w: [N, K] fp16 -> qweight/qzeros/scales в раскладке GPTQ bits=8 (как у нашего слепка)."""
    N, K = w.shape
    wt = w.t().contiguous().float()                       # [K, N]
    г = wt.view(K//G, G, N)
    мин, макс = г.amin(1), г.amax(1)
    s = ((макс-мин)/255).clamp_min(1e-12); z = torch.round(-мин/s).clamp(0,255)
    q = torch.round(г/s.unsqueeze(1)+z.unsqueeze(1)).clamp(0,255).to(torch.int64).view(K,N)
    qw = torch.zeros(K//4, N, dtype=torch.int64, device=w.device)
    for i in range(4): qw |= (q[i::4] & 0xFF) << (8*i)
    zf = (z-1).clamp(0,255).to(torch.int64)
    qz = torch.zeros(K//G, N//4, dtype=torch.int64, device=w.device)
    for i in range(4): qz |= (zf[:, i::4] & 0xFF) << (8*i)
    к32 = lambda t: torch.where(t >= (1<<31), t-(1<<32), t).to(torch.int32)
    return к32(qw), к32(qz), s.to(torch.float16)


def main():
    from vllm import _custom_ops as ops
    ext = fa2_sm70._ext.w8a16_ext()
    torch.manual_seed(0)
    print(f"M={M}, группа {G}; время в мс, меньше -- лучше\n")
    print(f"{'слой':<18s} {'форма':>15s} {'разворот+cuBLAS':>16s} {'наш упакованный':>16s} "
          f"{'плотный cuBLAS':>15s} {'наш/движок':>11s}")
    for имя, N, K in ФОРМЫ:
        w = (torch.randn(N, K, device="cuda")*0.02).half()
        qw, qz, sc = gptq(w)
        x = (torch.randn(M, K, device="cuda")*0.5).half()
        g = torch.empty((0,), dtype=torch.int32, device="cuda")
        qs = qw.clone(); ops.gptq_shuffle(qs, g, 8)
        t_dv = замер(lambda: ops.gptq_gemm(x, qs, qz, sc, g, True, False, 8))
        try:
            t_our = замер(lambda: ext.w8a16_gemm_hmma(x, qw, qz, sc, G))
        except Exception as e:
            t_our = float("nan"); print(f"  наш путь отказал: {str(e)[:60]}")
        wd = w.t().contiguous()
        t_fl = замер(lambda: torch.matmul(x, wd))
        print(f"{имя:<18s} {str((N,K)):>15s} {t_dv:>16.2f} {t_our:>16.2f} {t_fl:>15.2f} "
              f"{t_dv/t_our:>11.3f}")
        del w, qw, qz, sc, x, qs, wd; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
