# -*- coding: utf-8 -*-
"""Сводная кривая по ОБЕИМ фазам: префилл и декод НА ОДНОЙ И ТОЙ ЖЕ длине контекста.

Отличие от `bench_prefill_ladder.py`: проба идёт стримингом, поэтому одна проба даёт две
величины -- TTFT (время до первого токена = префилл) и темп остальных токенов (декод при
контексте ровно этой длины). Мерить декод отдельным коротким запросом НЕЛЬЗЯ: внимание
декода читает весь KV, и его цена -- функция длины.

Дисциплина унаследована от лестницы и оплачена граблями стенда:
  * У КАЖДОЙ ПРОБЫ СВОЙ СТОГ -- иначе второй раз отдаётся из префикс-кэша (12 617 ток/с
    против честных 640).
  * КАЛИБРОВКА САМОУТОЧНЯЕТСЯ по факту предыдущей точки (приставки удлиняются с номером).
  * ТАКТЫ ПИШУТСЯ В КАЖДОЙ ТОЧКЕ: короб сбрасывает 1530 -> 945 МГц на длинной дистанции, и
    без записи тактов сравнение конфигураций читает ТЕРМИКУ как разницу.

    python benchmarks/bench_dve_fazy.py <метка> [порт=8090] [токенов_декода=48]
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

МЕТКА = sys.argv[1] if len(sys.argv) > 1 else "?"
ПОРТ = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
ДЕК = int(sys.argv[3]) if len(sys.argv) > 3 else 48
ЗАП = ("Летний день клонился к вечеру, и в саду пахло нагретой землёй. "
       "Никто не спешил, разговоры текли сами собой, без цели и без конца. ")


def сбор(м, повт):
    return "".join(f"Проба {м}-{i}. " + ЗАП for i in range(повт))


def мера(текст, макс, тм=2400):
    """Возвращает (токенов промпта, TTFT, темп декода ток/с, число токенов ответа)."""
    д = json.dumps({"model": "gpt-4-0125-preview",
                    "messages": [{"role": "user", "content": текст +
                                  "\n\nПерескажи одним абзацем, о чём был текст."}],
                    "max_tokens": макс, "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True}}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{ПОРТ}/v1/chat/completions", д,
                               {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    t_посл = None
    штук = 0
    промпт = 0
    with urllib.request.urlopen(r, timeout=тм) as поток:
        for сырая in поток:
            стр = сырая.decode("utf-8", "replace").strip()
            if not стр.startswith("data: "):
                continue
            тело = стр[6:]
            if тело == "[DONE]":
                break
            o = json.loads(тело)
            if o.get("usage"):
                промпт = o["usage"]["prompt_tokens"]
            вар = o.get("choices") or []
            д_ = (вар[0].get("delta") or {}) if вар else {}
            # ДУМАЮЩАЯ СЕТЬ: содержимое приходит в `reasoning`, а `content` пуст до конца
            # блока размышления. Считать только `content` -- ослепнуть на ноль токенов.
            куски = (д_.get("content") or "") + (д_.get("reasoning") or "") + \
                    (д_.get("reasoning_content") or "")
            if куски:
                if ttft is None:
                    ttft = time.time() - t0
                штук += 1
                t_посл = time.time()
    # декод считается по токенам ПОСЛЕ первого: первый оплачен префиллом
    темп = (штук - 1) / (t_посл - t0 - ttft) if штук > 1 and t_посл else 0.0
    return промпт, ttft or 0.0, темп, штук


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
            time.sleep(10)


n, _, _, _ = мера(сбор("к", 300), 8)
тнп = n / 300.0
print(f"[{МЕТКА}] калибровка {тнп:.2f} ток/повтор", flush=True)
print(f"[{МЕТКА}] {'промпт':>9} | {'префилл':>8} {'ток/с':>7} | {'декод':>7} {'мс/ток':>7} | такты",
      flush=True)
м = 0
for цель in (32000, 64000, 128000, 247000):
    м += 1
    ч = Такты()
    threading.Thread(target=ч.ход, daemon=True).start()
    повт = int(цель / тнп)
    n, ttft, темп, штук = мера(сбор(f"{МЕТКА}{м}", повт), ДЕК)
    ч.стоп = True
    if n:
        тнп = n / повт
    такт = f"{ч.мин[0]}/{ч.мин[1]}" if ч.мин[0] < 9999 else "-"
    мснат = 1000.0 / темп if темп else 0.0
    print(f"[{МЕТКА}] {n:>9,} | {ttft:>7.1f}с {n/ttft if ttft else 0:>7.0f} | "
          f"{темп:>6.1f}/с {мснат:>6.1f} | {такт}  ({штук} ток ответа)", flush=True)
