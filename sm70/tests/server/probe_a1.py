"""Проверка CLEF_A1: запись Clef, ВЫТЕСНЕННАЯ посреди префилла, обязана дать тот же ответ, что одиночная.

  python probe_a1.py <порт> <выход.json>
Стенд с маленьким пулом (FA2SM70_KV_BYTES_FORCE на 6 блоков, FA2SM70_MAXLEN=81920): 5 рабочих блоков
внимания. Чат A ~49 050 токенов (3 блока) декодирует; запись Clef C ~30K (2 блока) префиллится; когда
декод A переходит 49 152, ему нужен 4-й блок -> движок вытесняет самый новый (C) -> C считается заново
с позиции 0. Без A1 спаны C складываются дважды (тихая порча), с A1 -- ответ = одиночному.
"""
import json
import os
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

DOCS = ps.текст_доков(10 ** 8)


def ntok(text):
    d, _ = ps.post(PORT, "/tokenize", {"model": ps.ИМЯ, "prompt": text})
    return d["count"]


def fit(prefix, target, tail=""):
    lo, hi = 1000, min(len(DOCS), target * 6)
    while hi - lo > 200:
        mid = (lo + hi) // 2
        if ntok(prefix + DOCS[:mid] + tail) < target:
            lo = mid
        else:
            hi = mid
    return prefix + DOCS[:lo] + tail


def preempt():
    t = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    m = re.search(r"^vllm:num_preemptions_total{[^}]*} ([0-9.e+]+)", t, re.M)
    return float(m.group(1)) if m else -1.0


# запись C ~30K токенов (город / цвет / пароль -- известные ответы)
body = DOCS[200000:200000 + 76000]
fact = "\n\nIMPORTANT NOTE. The meeting is scheduled in the city of Lisbon. The vault password is FLAMINGO. The car is red.\n\n"
C = {"model": "clef", "state": body[:30000] + fact + body[30000:],
     "questions": {"city": {"type": "choice", "criteria": {"tbilisi": "Tbilisi", "kazan": "Kazan", "lisbon": "Lisbon",
                                                            "oslo": "Oslo"},
                            "instructions": "In which city is the meeting scheduled?"},
                   "password": {"type": "noul", "instructions": "Does the text say the vault password is FLAMINGO?"},
                   "car": {"type": "choice", "criteria": {c: f"The car is {c}" for c in ("red", "blue", "green", "black")},
                           "instructions": "What color is the car?"}}}


def probs(d):
    a = d.get("answers", {})
    out = {}
    for q, v in a.items():
        if "probabilities" in v:
            out.update({f"{q}:{k}": p for k, p in v["probabilities"].items()})
        if "noul" in v:
            out[f"{q}:noul"] = v["noul"]
    return out


res = {}

# [v2 18.10] r6a: A ~49K за C (4 из 5 блоков) не помещался вовсе и прошёл отдельно -- вытеснений 0.
# Теперь A короткий (2 блока) и долгий: держит пул, пока C префиллится; второй отрезок C (ещё 2 блока)
# не помещается -> движок вытесняет младший запрос (C). C при живом A идёт ПЕРВЫМ (холодным),
# одиночный C -- после (тождество попадания в кэш Clef проверено r4: 35/35, max|dp| 0).
A_TOK, A_MAX = int(os.environ.get("A1_A_TOK", "15000")), int(os.environ.get("A1_A_MAX", "1500"))
A_text = fit("Перескажи подробно документ ниже, по разделам.\n\n", A_TOK)
print(f"A: {ntok(A_text)} токенов промпта", flush=True)
first = threading.Event()
a_out = {}


def run_A():
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(
        {"model": ps.ИМЯ, "prompt": A_text, "max_tokens": A_MAX, "ignore_eos": True, "temperature": 0.0, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                n += 1
                if n == 3:
                    first.set()
                if b'"finish_reason":"' in line or b'"finish_reason": "' in line:
                    a_out["конец"] = line[-200:].decode(errors="replace")
    a_out["кусков"] = n


p0 = preempt()
ta = threading.Thread(target=run_A)
ta.start()
first.wait(600)
time.sleep(1.0)
d, t = ps.post(PORT, "/v1/systemone", C)
ta.join()
p1 = preempt()
res["C_с_вытеснением"] = {"t": round(t, 1), "tok": d.get("usage", {}).get("input_tokens"), "p": probs(d),
                          "вытеснений": p1 - p0}
print(f"A: {a_out}", flush=True)
d, t = ps.post(PORT, "/v1/systemone", C)
res["C_одиночная"] = {"t": round(t, 1), "tok": d.get("usage", {}).get("input_tokens"), "p": probs(d)}
print(f"C одиночная: {res['C_одиночная']['tok']} ток, {t:.1f} с, {res['C_одиночная']['p']}", flush=True)
s, v = res["C_одиночная"]["p"], res["C_с_вытеснением"]["p"]
dmax = max((abs(s[k] - v.get(k, 0.0)) for k in s), default=float("nan"))
print(f"C при живом A: {res['C_с_вытеснением']['t']} с, вытеснений за опыт {p1 - p0:.0f}, {v}", flush=True)
print(f"ИТОГ A1: max|dp| против одиночной {dmax:.4f} -> "
      f"{'СОВПАЛО' if dmax < 0.02 else 'РАСХОЖДЕНИЕ'} (вытеснение {'БЫЛО' if p1 - p0 > 0 else 'НЕ СЛУЧИЛОСЬ'})", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
