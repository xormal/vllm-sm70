"""Разбор расхождения ОЗУ против GPU (r3c: 2-3 блока расходятся с 19-й позиции декода, ~1e-2).

  python probe_ozu_ident2.py <порт> <выход.json>
Для X ~110K символов (2 блока):
  без нагрузки: cold1 -> gpu1 -> gpu2 (детерминизм повторов на тех же блоках);
  нагрузка Clef, снять: ram (из ОЗУ в НОВЫЕ физические блоки) -> gpu3 (сразу повтор на загруженных блоках);
  снова нагрузка, снять: (если в ОЗУ X уже затёрт -- cold2 в новых блоках; иначе ram2).
Вывод -- попарные сравнения: где расходятся и с какой позиции.
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
    m = re.search(r'^vllm:prompt_tokens_by_source_total{[^}]*source="external_kv_transfer"[^}]*} ([0-9.e+]+)', t, re.M)
    return dict(h=g("vllm:prefix_cache_hits_total"), ext=float(m.group(1)) if m else 0.0,
                run=g("vllm:num_requests_running"), wait=g("vllm:num_requests_waiting"))


def complete(text):
    m0 = metrics()
    d, t = ps.post(PORT, "/v1/completions", {"model": ps.ИМЯ, "prompt": text, "max_tokens": 40,
                                              "temperature": 0.0, "logprobs": 5})
    m1 = metrics()
    lp = d["choices"][0]["logprobs"]
    return dict(t=round(t, 2), gpu=int(m1["h"] - m0["h"]), ozu=int(m1["ext"] - m0["ext"]),
                toks=lp["tokens"], lps=lp["token_logprobs"])


def idle():
    t0 = time.time()
    while time.time() - t0 < 600:
        m = metrics()
        if m["run"] == 0 and m["wait"] == 0:
            return
        time.sleep(2)


def load(sec, nthr=3):
    stop = threading.Event()

    def loop(seed):
        r = random.Random(seed)
        while not stop.is_set():
            k = r.randrange(4000, 60000)
            off = r.randrange(0, max(1, len(DOCS) - k))
            ps.post(PORT, "/v1/systemone", {"model": "clef", "state": f"[{seed}-{r.random()}] " + DOCS[off:off + k],
                                            "questions": {"u": {"type": "noul", "instructions": "Urgent?"}}})
    ths = [threading.Thread(target=loop, args=(random.random(),), daemon=True) for _ in range(nthr)]
    for t in ths:
        t.start()
    time.sleep(sec)
    stop.set()
    idle()


def cmp(a, b):
    first = next((i for i, (x, y) in enumerate(zip(a["lps"], b["lps"])) if x != y), None)
    dm = max(abs(x - y) for x, y in zip(a["lps"], b["lps"]))
    return f"токены {'==' if a['toks'] == b['toks'] else '!='}, первое расхождение логпроба {first}, max|d| {dm:.2e}"


X = f"Документ {random.random()}.\n" + DOCS[:110000] + "\n\nКратко, о чём этот текст:"
R = {}
R["cold1"] = complete(X)
R["gpu1"] = complete(X)
R["gpu2"] = complete(X)
load(60)
R["ram"] = complete(X)
R["gpu3"] = complete(X)
load(90, 4)
R["after2"] = complete(X)
for k, v in R.items():
    print(f"{k}: GPU {v['gpu']} ОЗУ {v['ozu']} ({v['t']} с)", flush=True)
for a, b in (("cold1", "gpu1"), ("gpu1", "gpu2"), ("gpu1", "ram"), ("ram", "gpu3"), ("gpu1", "gpu3"),
             ("gpu1", "after2"), ("ram", "after2")):
    print(f"  {a} vs {b}: {cmp(R[a], R[b])}", flush=True)
json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1)
