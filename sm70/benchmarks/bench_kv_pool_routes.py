"""Стоимость СБОРКИ и ВНИМАНИЯ на боевых формах: e4m3-пул против нашего int8-пула.

Формы -- те, что реально приходят от сервера при TP=2: Hkv=2, d=256, чанк запросов 16384,
префикс T. Мерится min-of-N (шум машины), синхронно.
"""
import math, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fa2_sm70
D = 256
dext, pext = fa2_sm70._ext.decode_ext(), fa2_sm70._ext.prefill_ext()


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def run(T, Sq, Hkv=2, H=12, bs=16):
    nblk = (T + bs - 1) // bs
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    slots = torch.arange(T, device="cuda", dtype=torch.long)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32)
    q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    sc = 1.0 / math.sqrt(D)

    # --- e4m3 пул
    kc = torch.zeros(nblk, bs, Hkv, D, device="cuda", dtype=torch.uint8)
    vc = torch.zeros_like(kc)
    kb_ = torch.zeros(Hkv, device="cuda", dtype=torch.float32)
    ks_l = float(k.abs().max()) / 448.0
    vs_l = float(v.abs().max()) / 448.0
    fk = torch.empty(Hkv, T, D, device="cuda", dtype=torch.float16)
    fv = torch.empty_like(fk)
    t_store8 = bench(lambda: dext.reshape_and_cache_e4m3(k, v, kc, vc, slots, ks_l, vs_l, kb_, 1.15))
    t_gath8 = bench(lambda: dext.gather_paged_kv(kc, vc, bt, T, ks_l, vs_l, fk, fv))
    t_attn8 = bench(lambda: pext.attn_fwd_qbshd(q, fk.unsqueeze(0), fv.unsqueeze(0), sc, True), n=10, warm=3)

    # --- наш int8 пул
    kci = torch.zeros(nblk, bs, Hkv, D, device="cuda", dtype=torch.uint8)
    vci = torch.zeros_like(kci)
    kst = torch.zeros(nblk * bs * Hkv, device="cuda", dtype=torch.float32)
    vst = torch.zeros_like(kst)
    kq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    vq = torch.empty_like(kq)
    kqs = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    vqs = torch.empty_like(kqs)
    t_storei = bench(lambda: dext.reshape_and_cache_i8(k, v, kci, vci, slots, kst, vst))
    t_gathi = bench(lambda: dext.gather_paged_kv_i8_pool(kci, vci, kst, vst, bt, T, kq, kqs, vq, vqs))
    t_attni = bench(lambda: pext.attn_fwd_volta_i8(q, kq.unsqueeze(0), kqs.unsqueeze(0),
                                                   vq.unsqueeze(0), vqs.unsqueeze(0), sc, True), n=10, warm=3)
    # --- третий маршрут: НАШ int8-пул, но разворот в fp16 и обычное fp16-ядро (точность та же --
    #     разворот q*s точный; вопрос только в цене)
    t_gathf = bench(lambda: dext.gather_paged_kv_i8_pool_f16(kci, vci, kst, vst, bt, T, fk, fv))
    print(f"      int8-пул -> fp16-ядро: gather {t_gathf:7.3f} + attn {t_attn8:8.3f} = "
          f"{t_gathf+t_attn8:8.3f} ms/слой", flush=True)
    print(f"T={T:>6} Sq={Sq:>5} | store {t_store8:7.3f} -> {t_storei:7.3f} ms (x{t_store8/t_storei:.2f}) | "
          f"gather {t_gath8:7.3f} -> {t_gathi:7.3f} ms (x{t_gath8/t_gathi:.2f}) | "
          f"attn {t_attn8:8.3f} -> {t_attni:8.3f} ms (x{t_attn8/t_attni:.2f}) | "
          f"итого/слой {t_gath8+t_attn8:8.3f} -> {t_gathi+t_attni:8.3f} (x{(t_gath8+t_attn8)/(t_gathi+t_attni):.2f})",
          flush=True)


def run_decode(T, H=12, Hkv=2, bs=16, ns=64):
    """ДЕКОД из того же пула. Здесь у int8 есть СВОЯ цена: помимо тех же байтов ядро читает по
    4 байта масштаба на (позицию, kv-голову) -- +1.6% трафика по построению, но чтение скалярное
    (широковещательное на варп), поэтому реальная цена может быть выше расчётной. Мерим, а не
    прикидываем: декод на длинном контексте упирается в полосу памяти на ~90%."""
    nblk = (T + bs - 1) // bs
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    slots = torch.arange(T, device="cuda", dtype=torch.long)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32).view(1, nblk)
    q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.float16) * 0.3
    sc = 1.0 / math.sqrt(D)
    A = torch.empty(H * ns * D, device="cuda", dtype=torch.float32)
    Bw = torch.empty(2 * H * ns, device="cuda", dtype=torch.float32)
    out = torch.empty(1, H, D, device="cuda", dtype=torch.float16)
    km = torch.zeros(1, Hkv, device="cuda", dtype=torch.float32)
    sl = torch.full((1,), T, device="cuda", dtype=torch.int32)

    kc = torch.zeros(nblk, bs, Hkv, D, device="cuda", dtype=torch.uint8)
    vc = torch.zeros_like(kc)
    kb_ = torch.zeros(Hkv, device="cuda", dtype=torch.float32)
    ks_l, vs_l = float(k.abs().max()) / 448.0, float(v.abs().max()) / 448.0
    dext.reshape_and_cache_e4m3(k, v, kc, vc, slots, ks_l, vs_l, kb_, 1.15)
    t8 = bench(lambda: dext.flash_decode_defer_mqa_paged_into(
        q, kc, vc, km, bt, bs, sc, ns, nblk * bs, sl, True, ks_l, vs_l, A, Bw, out, H * D), n=50, warm=10)

    kci = torch.zeros(nblk, bs, Hkv, D, device="cuda", dtype=torch.uint8)
    vci = torch.zeros_like(kci)
    kst = torch.zeros(nblk * bs * Hkv, device="cuda", dtype=torch.float32)
    vst = torch.zeros_like(kst)
    dext.reshape_and_cache_i8(k, v, kci, vci, slots, kst, vst)
    ti = bench(lambda: dext.flash_decode_defer_mqa_paged_into(
        q, kci, vci, km, bt, bs, sc, ns, nblk * bs, sl, False, 1.0, 1.0, A, Bw, out, H * D,
        kst, vst), n=50, warm=10)
    print(f"decode ctx={T:>6} | e4m3 {t8*1e3:8.1f} us -> int8 {ti*1e3:8.1f} us (x{t8/ti:.3f})", flush=True)


if __name__ == "__main__":
    print("боевые формы: Hkv=2, H=12, d=256, чанк 16384 (сервер, TP=2)")
    for T, Sq in ((16384, 16384), (32768, 16384), (65536, 16384), (131072, 16384)):
        run(T, min(Sq, T))
    print()
    for T in (4096, 16384, 65536, 131072):
        run_decode(T)
