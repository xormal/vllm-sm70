# -*- coding: utf-8 -*-
"""Лестница префилла по живому серверу: 32K/64K/128K/247K, такты пишутся ВО ВРЕМЯ проб.

Дисциплина, оплаченная граблями этого стенда (18.08.2026):
  * У КАЖДОЙ ПРОБЫ СВОЙ СТОГ -- одинаковый второй раз отдаётся из префикс-кэша, и число липовое
    (наблюдалось 12 617 ток/с против честных 640).
  * КАЛИБРОВКА САМОУТОЧНЯЕТСЯ ПО ФАКТУ: приставки проб удлиняются с номером (x1.059 на 128K),
    и фиксированная калибровка дважды переливала за предел контекста (400 Bad Request).
  * ТАКТЫ ПИШУТСЯ В КАЖДОЙ ТОЧКЕ: короб сбрасывает 1530 -> 945-1140 МГц на длинной дистанции,
    и без записи тактов сравнение конфигураций читает ТЕРМИКУ как разницу ядер.
  * Гейт точности лестница НЕ заменяет: варианты ядра обязаны проходить ГЛУБОКИЕ иглы отдельно
    (V408 был «быстрее на 8-12 %» и ломал ответы на глубинах 50/97 % при верной короткой
    арифметике -- пробы лестницы этого не видят).

    python benchmarks/bench_prefill_ladder.py <метка> [порт=8090]
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

МЕТКА = sys.argv[1] if len(sys.argv) > 1 else "?"
ПОРТ = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
ЗАП = ("Летний день клонился к вечеру, и в саду пахло нагретой землёй. "
       "Никто не спешил, разговоры текли сами собой, без цели и без конца. ")


def сбор(м, повт):
    return "".join(f"Проба {м}-{i}. " + ЗАП for i in range(повт))


def мера(текст, тм=2400):
    д = json.dumps({"model": "gpt-4-0125-preview",
                    "messages": [{"role": "user", "content": текст + "\n\nОтветь одним словом: да."}],
                    "max_tokens": 8, "temperature": 0}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{ПОРТ}/v1/chat/completions", д,
                               {"Content-Type": "application/json"})
    t = time.time()
    o = json.loads(urllib.request.urlopen(r, timeout=тм).read())
    return o["usage"]["prompt_tokens"], time.time() - t


class Такты:
    def __init__(self):
        self.мин = {0: 9999, 1: 9999}
        self.стоп = False

    def ход(self):
        while not self.стоп:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm",
                                  "--format=csv,noheader,nounits", "-i", "0,1"],
                                 capture_output=True, text=True).stdout
            for стр in out.strip().splitlines():
                i, c = стр.split(",")
                self.мин[int(i)] = min(self.мин[int(i)], int(c))
            time.sleep(12)


n, _ = мера(сбор("к", 300))
тнп = n / 300.0
print(f"[{МЕТКА}] калибровка {тнп:.2f} ток/повтор", flush=True)
м = 0
for цель, проб in ((32000, 2), (64000, 1), (128000, 1), (247000, 1)):
    for _ in range(проб):
        м += 1
        ч = Такты()
        threading.Thread(target=ч.ход, daemon=True).start()
        повт = int(цель / тнп)
        n, dt = мера(сбор(f"{МЕТКА}{м}", повт))
        ч.стоп = True
        тнп = n / повт          # самоуточнение: следующая точка бьёт точнее
        такт = f"такты мин 0/1: {ч.мин[0]}/{ч.мин[1]}" if dt > 30 and ч.мин[0] < 9999 else ""
        print(f"[{МЕТКА}] {n:>9,} ток {dt:>7.1f} с {n/dt:>7.0f} ток/с  {такт}", flush=True)
