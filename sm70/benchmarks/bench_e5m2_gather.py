# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# СБОРКА СТРАНИЦ отдельно от внимания, три формата + доля крыши памяти.
#   fp16  : gather_paged_kv        -- 2 Б/значение на выходе
#   int8  : gather_paged_kv_i8     -- КВАНТУЕТ на лету (absmax по строке + деление) => это ВЫЧИСЛЕНИЕ
#   e5m2  : gather_paged_kv_bytes  -- перекладка uint4 как есть => это ПЕРЕВОЗКА
# Крыша: 841 ГБ/с достижимых на V100 (docs/VOLTA_SM70.md).
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D, ROOF = 256, 841.0


def foreign():
    me = str(os.getpid())
    uu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0,1"],
                        capture_output=True, text=True).stdout.split()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    return sum(1 for l in apps.strip().splitlines()
               if l.strip() and l.split(",")[0].strip() in uu and l.split(",")[1].strip() != me)


def med(fn, w=3, n=11):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        t.append(s.elapsed_time(e))
    return sorted(t)[n // 2]


def main():
    d = fa2_sm70._ext.decode_ext()
    Hkv, bs = 2, 16
    print(f"# чужих на картах 0-1: {foreign()}   крыша {ROOF} ГБ/с")
    hdr = (f"{'T':>8} {'fp16 мс':>9} {'int8 мс':>9} {'e5m2 мс':>9} | {'fp16 ГБ/с':>10} {'int8 ГБ/с':>10} "
           f"{'e5m2 ГБ/с':>10} | {'доля крыши e5m2':>16} {'Б/токен fp16':>13} {'int8':>6} {'e5m2':>6}")
    print(hdr); print("-" * len(hdr))
    for T in [4096, 16384, 65536, 131072]:
        nblk = (T + bs - 1) // bs
        bt = torch.arange(nblk, device="cuda", dtype=torch.int32)
        k16 = torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
        v16 = torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
        k1 = k16.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        v1 = v16.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        ok16 = torch.empty(Hkv, T, D, device="cuda", dtype=torch.float16)
        ov16 = torch.empty_like(ok16)
        okq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8); ovq = torch.empty_like(okq)
        oks = torch.empty(Hkv, T, device="cuda", dtype=torch.float32); ovs = torch.empty_like(oks)
        okb = torch.empty(Hkv, T, D, device="cuda", dtype=torch.uint8); ovb = torch.empty_like(okb)

        t16 = med(lambda: d.gather_paged_kv(k16, v16, bt, T, 1.0, 1.0, ok16, ov16))
        t8 = med(lambda: d.gather_paged_kv_i8(k1, v1, bt, T, 1.0, 1.0, okq, oks, ovq, ovs))
        t5 = med(lambda: d.gather_paged_kv_bytes(k1, v1, bt, T, okb, ovb))
        # трафик = прочитано + записано
        mv16 = 2 * Hkv * T * D * (2 + 2)                    # fp16 пул -> fp16 выход
        mv8 = 2 * Hkv * T * D * (1 + 1) + 2 * Hkv * T * 4   # 1-байт пул -> int8 + таблица масштабов
        mv5 = 2 * Hkv * T * D * (1 + 1)
        g16, g8, g5 = mv16 / t16 / 1e6, mv8 / t8 / 1e6, mv5 / t5 / 1e6
        print(f"{T:>8} {t16:>9.3f} {t8:>9.3f} {t5:>9.3f} | {g16:>10.1f} {g8:>10.1f} {g5:>10.1f} | "
              f"{100*g5/ROOF:>15.1f}% {2*Hkv*D*2:>13d} {2*Hkv*D+8:>6d} {2*Hkv*D:>6d}")
        del k16, v16, k1, v1, ok16, ov16, okq, ovq, oks, ovs, okb, ovb
        torch.cuda.empty_cache()
    print(f"# чужих на картах 0-1 в конце: {foreign()}")


if __name__ == "__main__":
    main()
