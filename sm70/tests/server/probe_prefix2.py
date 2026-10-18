"""Точный прибор префикс-кэша: попавшие токены ПОВТОРА по счётчикам движка (/metrics).

  python probe_prefix2.py <порт> <выход.json>
1) Под нагрузкой (2 потока записей Clef + 2 чата) -- промпты A1 ~21K и A2 ~53K токенов; нагрузка ещё 40 с;
   нагрузка снимается, движок пустеет; повтор A1 и A2 -- попадание = прирост prefix_cache_hits_total.
2) Контроль без нагрузки: новый A3 ~21K, сразу повтор -- попадание обязано быть (16384) у любого варианта.
"""
import json
import random
import re
import sys
import threading
import time
import urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
import os  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "integrations", "vllm_sm70", "clef"))
import proverka_stenda as ps  # noqa: E402

DOCS = ps.текст_доков(10 ** 7)


def metrics():
    t = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    g = lambda n: float(re.search(rf"^{n}{{[^}}]*}} ([0-9.e+]+)", t, re.M).group(1))  # noqa: E731
    try:
        ext = g("vllm:prompt_tokens_by_source_total") if False else float(re.search(
            r'^vllm:prompt_tokens_by_source_total{[^}]*source="external_kv_transfer"[^}]*} ([0-9.e+]+)', t, re.M).group(1))
    except Exception:
        ext = 0.0
    return dict(q=g("vllm:prefix_cache_queries_total"), h=g("vllm:prefix_cache_hits_total"), ext=ext,
                run=g("vllm:num_requests_running"), wait=g("vllm:num_requests_waiting"))


def prompt(chars, tag):
    return {"model": ps.ИМЯ, "messages": [{"role": "user", "content": f"Документ {tag}.\n" + DOCS[:chars]
                                           + "\n\nОдним словом: о чём этот текст?"}],
            "max_tokens": 4, "temperature": 0.0}


def send(body):
    m0 = metrics()
    d, t = ps.post(PORT, "/v1/chat/completions", body)
    m1 = metrics()
    return dict(t=round(t, 2), ptok=d.get("usage", {}).get("prompt_tokens"), hit=int(m1["h"] - m0["h"]),
                ozu=int(m1["ext"] - m0["ext"]), query=int(m1["q"] - m0["q"]))


stop = threading.Event()
stat = {"clef": 0, "chat": 0}


def clef_loop(seed):
    r = random.Random(seed)
    while not stop.is_set():
        n = r.randrange(4000, 60000)
        off = r.randrange(0, max(1, len(DOCS) - n))
        rec = {"model": "clef", "state": f"[{seed}-{r.random()}] " + DOCS[off:off + n],
               "questions": {"urgency": {"type": "noul", "instructions": "Does this text express urgency?"}}}
        ps.post(PORT, "/v1/systemone", rec)
        stat["clef"] += 1


def chat_loop(seed):
    r = random.Random(seed)
    while not stop.is_set():
        ps.post(PORT, "/v1/chat/completions", {"model": ps.ИМЯ, "messages": [{"role": "user", "content":
                f"[{r.random()}] Напиши рассказ о горах, не меньше трёхсот слов."}], "max_tokens": 200,
                "temperature": 0.0})
        stat["chat"] += 1


res = {}
tag = random.random()
A1, A2 = prompt(60000, f"A1-{tag}"), prompt(150000, f"A2-{tag}")
ths = [threading.Thread(target=clef_loop, args=(i,), daemon=True) for i in range(2)] + \
      [threading.Thread(target=chat_loop, args=(10 + i,), daemon=True) for i in range(2)]
for t in ths:
    t.start()
time.sleep(15)
res["A1_под_нагрузкой"] = send(A1)
res["A2_под_нагрузкой"] = send(A2)
time.sleep(40)
stop.set()
t0 = time.time()
while time.time() - t0 < 600:
    m = metrics()
    if m["run"] == 0 and m["wait"] == 0:
        break
    time.sleep(2)
res["ожидание_пустоты_с"] = round(time.time() - t0, 1)
res["A1_повтор_после"] = send(A1)
res["A2_повтор_после"] = send(A2)
A3 = prompt(60000, f"A3-{tag}")
res["A3_без_нагрузки"] = send(A3)
res["A3_сразу_повтор"] = send(A3)
res["фон"] = dict(stat)
for k, v in res.items():
    print(f"{k}: {v}", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
