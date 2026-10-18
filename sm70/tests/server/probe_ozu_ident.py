"""Гейт тождества выгрузки KV: ответ с блоками ИЗ ОЗУ == ответ с попаданием в кэш GPU (побайтово).

  python probe_ozu_ident.py <порт> <выход.json>
Оба пути начинают префилл с одной границы блока (16384*k), поэтому логпробы обязаны совпасть побайтово.
1) без нагрузки: X (холодный), сразу повтор X -> попадание GPU (счётчик prefix_cache_hits), L_gpu;
2) нагрузка Clef 60 с (вымывает блоки GDN из пула), снять, дождаться пустого движка;
3) повтор X -> попадание из ОЗУ (prompt_tokens_by_source external_kv_transfer), L_ram; сверка.
Для X берутся 3 длины (~21K, ~40K, ~57K) -- 1, 2 и 3 блока.
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
    body = {"model": ps.ИМЯ, "prompt": text, "max_tokens": 24, "temperature": 0.0, "logprobs": 5}
    m0 = metrics()
    d, t = ps.post(PORT, "/v1/completions", body)
    m1 = metrics()
    lp = d["choices"][0]["logprobs"]
    return dict(t=round(t, 2), gpu=int(m1["h"] - m0["h"]), ozu=int(m1["ext"] - m0["ext"]),
                toks=lp["tokens"], lps=lp["token_logprobs"], top=lp["top_logprobs"])


def idle():
    t0 = time.time()
    while time.time() - t0 < 600:
        m = metrics()
        if m["run"] == 0 and m["wait"] == 0:
            return
        time.sleep(2)


tag = random.random()
X = {n: f"Документ {n}-{tag}.\n" + DOCS[:n] + "\n\nКратко, о чём этот текст:" for n in (60000, 110000, 155000)}
res = {}
for n, x in X.items():
    res[n] = {"холодный": complete(x)}
    res[n]["gpu"] = complete(x)
stop = threading.Event()


def clef_loop(seed):
    r = random.Random(seed)
    while not stop.is_set():
        k = r.randrange(4000, 60000)
        off = r.randrange(0, max(1, len(DOCS) - k))
        ps.post(PORT, "/v1/systemone", {"model": "clef", "state": f"[{seed}-{r.random()}] " + DOCS[off:off + k],
                                        "questions": {"u": {"type": "noul", "instructions": "Urgent?"}}})


ths = [threading.Thread(target=clef_loop, args=(i,), daemon=True) for i in range(3)]
for t in ths:
    t.start()
time.sleep(60)
stop.set()
idle()
ok_all = True
for n, x in X.items():
    r = complete(x)
    res[n]["ozu"] = r
    g = res[n]["gpu"]
    same_t = g["toks"] == r["toks"]
    same_lp = g["lps"] == r["lps"] and g["top"] == r["top"]
    dmax = max(abs(a - b) for a, b in zip(g["lps"], r["lps"]))
    ok = same_t and same_lp and r["ozu"] > 0
    ok_all &= ok
    print(f"X~{n} символов: GPU-повтор попало {g['gpu']} ({g['t']} с); после нагрузки GPU {r['gpu']} + ОЗУ {r['ozu']} "
          f"({r['t']} с); токены {'==' if same_t else '!='}, логпробы+top5 {'ПОБАЙТОВО' if same_lp else f'max|d| {dmax:.2e}'}"
          f" -> {'ДА' if ok else 'НЕТ'}", flush=True)
print("ГЕЙТ ТОЖДЕСТВА ОЗУ:", "ПРОЙДЕН" if ok_all else "НЕ ПРОЙДЕН", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
