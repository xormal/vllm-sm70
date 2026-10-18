"""Функциональная проверка средних ключей разреженного декода: вспоминание СОБСТВЕННОГО вывода при
параллельных генерациях на длинном контексте.

  python probe_recall.py <порт> <выход.json> [параллельно=8] [раундов=2]
Каждый запрос: свой документ ~20K токенов (разреженный декод включается с 16384) + просьба придумать
шестизначный код, записать «КОД: XXXXXX», написать рассказ ~900 слов и в конце «ИТОГОВЫЙ КОД: XXXXXX».
Код существует только в подблоках, дописанных ДЕКОДОМ, и к концу ответа он далеко за окном 512 свежих
токенов. Успех -- коды совпали. Попутно: зацикливание (повтор 12-грамма в хвосте ответа).
"""
import concurrent.futures as cf
import json
import re
import sys

PORT, OUT = int(sys.argv[1]), sys.argv[2]
NPAR = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 2
import os  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "integrations", "vllm_sm70", "clef"))
import proverka_stenda as ps  # noqa: E402

DOCS = ps.текст_доков(10 ** 7)
TEMY = ["северной реке", "старом маяке", "горной деревне", "ночном поезде", "заброшенной обсерватории",
        "острове в тумане", "городе мостов", "караване в пустыне", "подводной станции", "лесной библиотеке",
        "зимнем порту", "часовой башне", "степном хуторе", "летающем острове", "шахтёрском посёлке", "оранжерее"]


def ask(i):
    off = (i * 50021) % (len(DOCS) - 62000)
    msg = (DOCS[off:off + 60000] + "\n\n---\nЗадание (документ выше прочитай, но он для задания не нужен). "
           "Придумай случайный шестизначный код и сразу запиши его первой строкой ответа в формате "
           "«КОД: XXXXXX». Затем напиши подробный художественный рассказ не меньше 900 слов о "
           f"{TEMY[i % len(TEMY)]}. В самом конце ответа отдельной строкой повтори тот же код в формате "
           "«ИТОГОВЫЙ КОД: XXXXXX».")
    d, t = ps.post(PORT, "/v1/chat/completions", {"model": ps.ИМЯ, "messages": [{"role": "user", "content": msg}],
                                                  "max_tokens": 2200, "temperature": 0.0})
    if "choices" not in d:
        return {"i": i, "err": str(d)[:300]}
    txt = d["choices"][0]["message"]["content"] or ""
    c1 = re.search(r"КОД:\s*\**\s*(\d{6})", txt, re.I)
    # [18.10] без учёта регистра: r5c дал ложный провал на «Итоговый код: 482915»
    c2 = re.findall(r"ИТОГОВЫЙ\s+КОД:\s*\**\s*(\d{6})", txt, re.I)
    words = txt.split()
    tail = words[-400:]
    grams = [" ".join(tail[k:k + 12]) for k in range(max(0, len(tail) - 12))]
    loop = len(grams) - len(set(grams)) > 20
    return {"i": i, "t": round(t, 1), "ptok": d.get("usage", {}).get("prompt_tokens"),
            "gtok": d.get("usage", {}).get("completion_tokens"), "код": c1.group(1) if c1 else None,
            "итог": c2[-1] if c2 else None, "цикл": loop, "хвост": txt[-160:]}


res = []
for r in range(NR):
    with cf.ThreadPoolExecutor(NPAR) as ex:
        rr = list(ex.map(ask, range(r * NPAR, (r + 1) * NPAR)))
    res += rr
    for x in rr:
        if "err" in x:
            print(f"  {x['i']}: ОШИБКА {x['err']}", flush=True)
            continue
        ok = x["код"] is not None and x["код"] == x["итог"]
        print(f"  {x['i']:2d}: {x['ptok']} ток промпт, {x['gtok']} ток ответ, {x['t']} с: код {x['код']} -> "
              f"итог {x['итог']} {'ВЕРНО' if ok else 'НЕВЕРНО'}{' ЦИКЛ' if x['цикл'] else ''}", flush=True)
good = [x for x in res if "err" not in x]
ok = sum(1 for x in good if x["код"] is not None and x["код"] == x["итог"])
nocode = sum(1 for x in good if x["код"] is None or x["итог"] is None)
loops = sum(1 for x in good if x["цикл"])
print(f"ВСПОМИНАНИЕ: верно {ok}/{len(good)}, без кода/итога {nocode}, зацикливаний {loops}", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
