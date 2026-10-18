"""Прибор «чанк префилла из 4 токенов = однородный декод» (находка 2 фаззинга 14.10).

  python probe_uniform.py <порт> <выход.json>
Одиночный B длиной 2048*m + 4: последний кусок ровно 4 токена -> шаг формы спекуляции (1 строка x 4).
Опорный ответ: тот же B, пока параллельно декодирует A (A берёт 4 токена бюджета -> куски B по 2044,
последний 8 токенов -> не однородный). Сравниваются жадные токены и логпробы первых 16 токенов B.
Контроль: длины 2051, 2053 (последний кусок 3 и 5 токенов) -- совпадать обязаны везде.
"""
import json
import random
import sys
import threading
import time
import urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
MODEL = "Qwen3.8-27B-ABLITERATED"


def call(prompt, max_tokens, logprobs=5):
    body = dict(model=MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0.0, logprobs=logprobs,
                seed=0)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as f:
        out = json.loads(f.read())
    lp = out["choices"][0].get("logprobs") or {}
    return dict(text=out["choices"][0]["text"], toks=lp.get("tokens"), lps=lp.get("token_logprobs"))


def ids(seed, n):
    r = random.Random(seed)
    return [r.randrange(1000, 100000) for _ in range(n)]


A_PROMPT = ("Напиши подробный рассказ о путешествии по северным рекам, с описанием погоды, людей и "
            "городов, не меньше двух тысяч слов.")
res = {}
for L in (2052, 4100, 2051, 2053):
    B = ids(L, L)
    solo = call(B, 16)
    # с параллельным A
    holder = {}
    th = threading.Thread(target=lambda: holder.update(a=call(A_PROMPT, 300, logprobs=None)))
    th.start()
    time.sleep(4.0)                       # A прошёл префилл и декодирует
    withA = call(B, 16)
    th.join()
    same = solo["toks"] == withA["toks"]
    dmax = max(abs(a - b) for a, b in zip(solo["lps"], withA["lps"]))
    res[L] = dict(solo=solo, withA=withA, same_tokens=same, max_dlp=dmax)
    print(f"L={L} (последний кусок {L - 2048 * (L // 2048)}): токены совпали {same}, "
          f"max|dlp| {dmax:.4f}; одиночный {solo['text'][:40]!r} / с A {withA['text'][:40]!r}", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
