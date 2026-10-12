"""БЮДЖЕТ ФАЗ ЯДРА РОЛЕЙ НА ОТГРУЖАЕМОМ ФОРМАТЕ 0 (боевой путь), СНЯТИЕМ ФАЗЫ.

ЗАЧЕМ ЗАНОВО. Весь прежний бюджет («первое умножение 35.1 %, второе 19.7 %, софтмакс 15.2 %»)
снимался на формате 8. Боевой путь идёт `nosc=0` -- ДРУГАЯ ветвь чтения масштабов и другой счёт
обращений к разделяемой. А укладчик (`tools/kanal_ili_cep.py`) назвал связывающим каналом именно
MIO (99.1 % периода при тензорной трубе 73.5 %) -- то есть доли, снятые на другом формате, к этому
телу не переносятся по той самой причине, которая делает их важными.

КАК ЧИТАТЬ. Ответ при фальсификаторе НЕВЕРЕН -- читается ТОЛЬКО время. Разность с базой даёт долю
фазы. Доли НЕ ОБЯЗАНЫ давать в сумме 100 %: снятие фазы освобождает не только её такты, но и
слоты, которыми она мешала соседям, поэтому сумма бывает и больше, и меньше.

    python benchmarks/bench_byudzhet_faz0.py [--sk 16384]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa2_sm70  # noqa: E402
from bench_kspl import замер, кв, таблица_масштабов  # noqa: E402

# nosc = вариант*100 + формат. Формат 0 -- отгружаемый.
ФАЗЫ = [
    (0,    "база (отгружаемое ядро)",            True),
    (1500, "снято ПЕРВОЕ умножение (Q*K^T)",     False),
    (1300, "снято ВТОРОЕ умножение (P*V)",       False),
    (1600, "сняты ОБА умножения",                False),
    (1100, "снят СОФТМАКС",                      False),
    (1400, "снята ПОДАЧА (везущие не везут)",    False),
    (1200, "сняты внутренние рандеву",           False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sk", default="16384,46002")
    ap.add_argument("--sq", type=int, default=4096)
    a = ap.parse_args()
    ext = fa2_sm70._ext.prefill_ext()
    D, H, HKV = 256, 24, 4
    for Sk in [int(x) for x in a.sk.split(",")]:
        torch.manual_seed(0)
        Q = (torch.randn(1, a.sq, H, D, device="cuda") * 4.0).half()
        K = (torch.randn(1, Sk, HKV, D, device="cuda") * 0.5).half()
        V = (torch.randn(1, Sk, HKV, D, device="cuda") * 0.5).half()
        Kq, Ks, _a, _b = кв(K, Sk, D)
        Vq, Vs, _c, _d = кв(V, Sk, D)
        Kt, Vt = таблица_масштабов(0, Ks, Vs, Sk)
        del K, V
        м = 1.0 / (D ** 0.5)
        зов = lambda n: ext.attn_fwd_volta_i8(Q, Kq.unsqueeze(0), Kt.unsqueeze(0),   # noqa: E731
                                              Vq.unsqueeze(0), Vt.unsqueeze(0), м, True, n)
        видно = (Sk - a.sq) * a.sq + a.sq * (a.sq + 1) / 2
        флоп = 2 * 2 * видно * H * D
        print(f"\n=== Sq={a.sq} Sk={Sk} H={H} Hkv={HKV} d={D}, причинная, ФОРМАТ 0 ===")
        print(f"{'фаза':<34s} {'мс':>8s} {'ТФЛОП/с':>9s} {'доля фазы':>10s}  ответ")
        база = None
        for nosc, метка, верен in ФАЗЫ:
            try:
                t = замер(зов_ := (lambda n=nosc: зов(n)))
            except Exception as e:  # noqa: BLE001
                print(f"{метка:<34s} {'—':>8s}  не собран: {str(e)[:44]}"); continue
            if база is None:
                база = t
            тф = флоп / (t / 1e3) / 1e12
            доля = "" if верен else f"{1 - t / база:>9.1%}"
            print(f"{метка:<34s} {t:>8.2f} {тф:>9.1f} {доля:>10s}  "
                  f"{'верен' if верен else 'НЕВЕРЕН (прибор)'}")
        del Q, Kq, Ks, Vq, Vs, Kt, Vt, _a, _b, _c, _d
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
