# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# РАЗЛОЖЕНИЕ ВЫИГРЫША e5m2 ПО ИСТОЧНИКАМ -- три замера на ОДНОМ ядре.
#
#   e5m2 (KVFMT=1)   -- перестановка байт, ни масштабов, ни свёртки смещения;
#   ФАЛЬС (KVFMT=2)  -- ТОТ ЖЕ e5m2 плюс ОДИН LOP3 в цепи распаковки (маска нулевая, ответ побитово
#                       тот же). В SASS отличается от e5m2 РОВНО на +128 LOP3 в блоках счёта и ни на
#                       что больше -- проверено разбором;
#   int8  (KVFMT=0)  -- те же +128 LOP3 в счёте ПЛЮС таблица масштабов у везущих (+14 LDG, +25
#                       адресных) и свёртка смещения в ранг-1.
#
# Отсюда: (ФАЛЬС - e5m2) = цена лишней команды распаковки; (int8 - ФАЛЬС) = цена таблицы масштабов.
# Трафик у всех трёх однобайтовый, поэтому полоса из объяснения исключена ПО ПОСТРОЕНИЮ.
import math
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256


def foreign():
    me = str(os.getpid())
    uu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0,1"],
                        capture_output=True, text=True).stdout.split()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    n = 0
    for line in apps.strip().splitlines():
        if line.strip():
            u, p = [s.strip() for s in line.split(",")]
            if u in uu and p != me:
                n += 1
    return n


def clocks():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
                           "--format=csv,noheader,nounits", "-i", "0"],
                          capture_output=True, text=True).stdout.strip()


def main():
    torch.manual_seed(0)
    p = fa2_sm70._ext.prefill_ext()
    H, Hkv, Sq = int(os.environ.get("H", 16)), 2, int(os.environ.get("SQ", 512))
    lens = [int(x) for x in os.environ.get("LENS", "4096,16384,65536,131072").split(",")]
    sc = 1.0 / math.sqrt(D)
    print(f"# чужих на картах 0-1 в начале: {foreign()}   карта0: {clocks()}")
    print(f"# H={H} Hkv={Hkv} d={D} Sq={Sq} causal   (все три варианта -- ОДНО ядро, разный KVFMT)")
    print()
    hdr = (f"{'Sk':>8} {'e5m2':>9} {'ФАЛЬС':>9} {'int8':>9} | {'команда':>9} {'масштабы':>9} "
           f"{'итого':>9} | доля команды")
    print(hdr); print("-" * len(hdr))
    for Sk in lens:
        q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        kf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        vf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        kb = kf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        vb = vf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        ka = kf.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
        va = vf.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
        kq = torch.clamp(torch.round(kf.float() / ka), -127, 127).to(torch.int8).contiguous()
        vq = torch.clamp(torch.round(vf.float() / va), -127, 127).to(torch.int8).contiguous()
        ks, vs = ka.squeeze(-1).contiguous().float(), va.squeeze(-1).contiguous().float()

        def run_e5(): return p.attn_fwd_volta_e5m2(q, kb, vb, sc, True)
        def run_i8(): return p.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True)

        # переключатель читается ЯДРОМ на каждом вызове (getenv в хостовой части), поэтому
        # чередование A/B/C законно: варианты не залипают.
        def with_env(v, fn):
            def g():
                os.environ["FA2SM70_FAKEXOR"] = v
                return fn()
            return g

        fns = {"e5m2": with_env("0", run_e5), "fake": with_env("1", run_e5),
               "int8": with_env("0", run_i8)}
        for f in fns.values():
            for _ in range(3):
                f()
        torch.cuda.synchronize()
        acc = {k: [] for k in fns}
        for _ in range(11):
            for k, f in fns.items():
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); f(); e.record(); torch.cuda.synchronize()
                acc[k].append(s.elapsed_time(e))
        m = {k: sorted(v)[len(v) // 2] for k, v in acc.items()}
        d_cmd = m["fake"] - m["e5m2"]
        d_scl = m["int8"] - m["fake"]
        tot = m["int8"] - m["e5m2"]
        print(f"{Sk:>8} {m['e5m2']:>9.3f} {m['fake']:>9.3f} {m['int8']:>9.3f} | "
              f"{d_cmd:>9.3f} {d_scl:>9.3f} {tot:>9.3f} | {100*d_cmd/tot if tot else 0:>6.1f}%")
        del q, kf, vf, kb, vb, kq, vq, ks, vs, ka, va
        torch.cuda.empty_cache()
    print()
    print(f"# чужих на картах 0-1 в конце: {foreign()}   карта0: {clocks()}")


if __name__ == "__main__":
    main()
