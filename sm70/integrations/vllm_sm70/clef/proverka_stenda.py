"""Приборы стенда Clef (сервер с FA2SM70_CLEF=1 или без него).

    baza   <порт> <выход.json>              логпробы фиксированных текстов (эхо промпта) + жадные 48
                                            токенов чата -- для ПОБАЙТОВОГО сравнения базы с рычагом
                                            и без (одинаковая последовательность запросов по одному)
    sravn  <a.json> <b.json>                побайтовая сверка двух прогонов baza
    clef   <порт> <выход.json> [повторов]   набор записей через /v1/systemone (по одной)
    clefsr <a.json> <b.json>                расхождение вероятностей двух прогонов clef
    smes   <порт> <выход.json> [раундов]    смешанная нагрузка: записи Clef параллельно с чатами;
                                            ответы Clef сверяются с одиночными (clef-прогоном)
    smesb  <порт> [раундов]                 та же нагрузка БЕЗ Clef (тексты записей обычным чатом,
                                            max_tokens=1) -- контроль: держит ли её сама база
    kartinki <порт> <выход.json|-> [clef]   штурм картинками: 8 запросов по 2 картинки (1280x960 + 640x480)
                                            разом; с «clef» -- параллельно 2 записи Clef (ответы в выход)
    dlinnye <порт> <выход.json> [max_length] ДЕТЕРМИНИРОВАННЫЙ тест длины для Clef: факты с известным
                                            ответом (город -- выбор из 4, пароль -- да/нет, цвет -- выбор
                                            из 4) на 5/50/95 % записи длиной 4K/12K/24K/31K + запись без
                                            фактов на каждую длину (ложное «да»); max_length по умолчанию 32768
    clefkart <порт> <выход.json>           Clef С КАРТИНКАМИ (FA2SM70_CLEF_MEDIA=1): записи с известным ответом
                                            (цвет, цифра, фигура, слово, порядок двух картинок, картинка +
                                            длинное состояние), повтор на детерминизм, отказы (видео, плохой
                                            URL, без картинок -- как прежде)
    igla   <порт> <токенов> [выход.json]    кодовое слово в САМОМ НАЧАЛЕ длинного контекста (гейт
                                            поведения боевого: «первый токен верный, дальше мусор»)
                                            + короткий осмысленный ответ; худший случай по памяти
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ИМЯ = "Qwen3.8-27B-ABLITERATED"


def post(port, path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode(errors="replace")[:2000]}, time.time() - t


def текст_доков(n, off=0):
    s = ""
    for f in ("docs/VOLTA_SM70.md", "docs/EPILOG.md", "docs/SM70_KERNEL_PLAYBOOK.md", "docs/KERNELS.md"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            s += open(p, encoding="utf-8").read()
    return s[off:off + n]


def записи():
    R = []
    R.append({"model": "clef", "state": "My Stripe integration keeps failing. Please help ASAP.",
              "questions": {"urgency": {"type": "noul", "instructions": "Does this message express urgency?"}}})
    R.append({"model": "clef", "state": "Our checkout started returning errors and orders are blocked.",
              "questions": {
                  "department": {"type": "choice", "instructions": "Which team should handle the message?",
                                 "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
                  "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
                  "outage": {"type": "noul", "instructions": "Is a service down?"}}})
    R.append({"model": "clef", "state": {"review": "The hotel room was spotless and the staff were lovely, "
                                                   "but breakfast was cold and overpriced.", "stars": None},
              "questions": {
                  "sentiment": {"type": "choice", "criteria": {"positive": "Mostly positive", "negative": "Mostly negative",
                                                               "mixed": "Both positive and negative"}},
                  "rating": {"type": "score", "instructions": "Likely star rating",
                             "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]},
                  "mentions_food": {"type": "noul", "instructions": "Does the review mention food?"}}})
    R.append({"model": "clef", "state": "Привет! Я забыл пароль от личного кабинета и не могу войти уже второй день.",
              "questions": {
                  "language": {"type": "choice", "criteria": {"en": "English", "ru": "Russian", "de": "German"}},
                  "intent": {"type": "choice", "criteria": {"login": "Cannot log in / password", "refund": "Wants money back",
                                                            "other": "Anything else"}},
                  "angry": {"type": "noul", "instructions": "Is the author angry?"}}})
    R.append({"model": "clef", "state": "def add(a, b):\n    return a - b\n",
              "questions": {"bug": {"type": "noul", "instructions": "Does the function have a bug?"},
                            "lang": {"type": "choice", "criteria": {"python": "Python", "js": "JavaScript", "c": "C"}}}})
    R.append({"model": "clef", "state": {"messages": [{"from": "user", "text": "how do I cancel my subscription"},
                                                      {"from": "agent", "text": "Go to settings > billing"},
                                                      {"from": "user", "text": "done, thanks!"}]},
              "questions": {"resolved": {"type": "noul", "instructions": "Was the user's issue resolved?"},
                            "topic": {"type": "choice", "criteria": {"billing": "Billing/subscription", "bug": "Bug report",
                                                                     "feature": "Feature request"}}}})
    # длинные состояния: несколько кусков префилла
    for n, off in ((9000, 0), (30000, 5000), (60000, 20000)):
        R.append({"model": "clef", "state": текст_доков(n, off),
                  "questions": {
                      "topic": {"type": "choice", "instructions": "What is the document mainly about?",
                                "criteria": {"gpu": "GPU programming / performance", "cooking": "Cooking recipes",
                                             "law": "Legal text", "sports": "Sports news"}},
                      "russian": {"type": "noul", "instructions": "Is the document written mostly in Russian?"},
                      "quality": {"type": "score", "instructions": "How technical is the text?",
                                  "criteria": ["Not technical", "Somewhat technical", "Highly technical"]}}})
    # много вопросов и вариантов
    qs = {f"q{i}": {"type": "choice", "instructions": f"Pick the option that best describes aspect {i} of the text",
                    "criteria": {f"o{j}": f"Description of option {j} for aspect {i}" for j in range(6)}}
          for i in range(12)}
    R.append({"model": "clef", "state": текст_доков(4000, 1000), "questions": qs})
    return R


def baza(port, out):
    # ЭХО-ЛОГПРОБЫ ПРОМПТА НЕ БРАТЬ: логиты по всему словарю на каждую позицию куска 2048 -- это
    # 1.89 ГиБ на ранге, стенд (и боевой) падает по памяти (замерено 11.10). Генерация с логпробами
    # выбранных токенов считает логиты одной строки и сверяется побайтово так же.
    res = {}
    длинный = текст_доков(24000, 3000) + "\n\nSummarize the text above in three sentences."
    for i, q in enumerate(("Explain in two sentences what a GPU warp is.",
                           "Напиши короткое стихотворение о реке.", длинный)):
        d, t = post(port, "/v1/chat/completions", {"model": ИМЯ, "messages": [{"role": "user", "content": q}],
                                                    "max_tokens": 96, "temperature": 0.0, "logprobs": True})
        c = d["choices"][0]
        res[f"чат{i}"] = {"text": c["message"]["content"], "lp": [x["logprob"] for x in c["logprobs"]["content"]]}
        print(f"чат{i}: {c['message']['content'][:80]!r} ({t:.1f} с)", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False)


def sravn(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    vse = True
    for k in A:
        x, y = A[k], B.get(k)
        if isinstance(x, dict):
            same = x == y
            print(f"{k}: текст {'совпал' if x['text'] == y['text'] else 'РАЗНЫЙ'}, логпробы "
                  f"{'побайтово' if x['lp'] == y['lp'] else 'РАЗНЫЕ'}")
        else:
            same = x == y
            diff = [abs(p - q) for p, q in zip(x, y) if p is not None and q is not None]
            print(f"{k}: {'побайтово' if same else f'РАЗНЫЕ, max|Δ| {max(diff):.2e}'} ({len(x)} токенов)")
        vse &= same
    print("ИТОГ:", "БАЗА НЕ ТРОНУТА (побайтово)" if vse else "ЕСТЬ РАСХОЖДЕНИЯ")


def вероятности(ans):
    p = {}
    for qid, a in ans.items():
        if a["type"] == "noul":
            p[qid] = {"true": a["noul"]}
        else:
            p[qid] = a["probabilities"]
    return p


def clef(port, out, повторов=1):
    res = []
    for i, r in enumerate(записи()):
        for k in range(повторов):
            d, t = post(port, "/v1/systemone", r)
            ok = "answers" in d
            res.append({"i": i, "k": k, "t": t, "resp": d})
            кратко = {q: (a.get("choice") or a.get("score") or a.get("noul")) for q, a in d.get("answers", {}).items()} \
                if ok else d
            print(f"запись {i} повтор {k}: {d.get('usage', {}).get('input_tokens', '?')} ток, {t:.2f} с: "
                  f"{json.dumps(кратко, ensure_ascii=False)[:300]}", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)


def clefsr(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    mx_all, argmax_ok, n = 0.0, 0, 0
    for x, y in zip(A, B):
        if "answers" not in x["resp"] or "answers" not in y["resp"]:
            print(f"запись {x['i']}: нет ответа"); continue
        pa, pb = вероятности(x["resp"]["answers"]), вероятности(y["resp"]["answers"])
        mx = 0.0
        for q in pa:
            for o in pa[q]:
                mx = max(mx, abs(pa[q][o] - pb[q][o]))
            if len(pa[q]) > 1:
                n += 1
                argmax_ok += max(pa[q], key=pa[q].get) == max(pb[q], key=pb[q].get)
            else:
                n += 1
                argmax_ok += (pa[q]["true"] > 0.5) == (pb[q]["true"] > 0.5)
        mx_all = max(mx_all, mx)
        print(f"запись {x['i']} ({x['resp']['usage']['input_tokens']} ток): max|Δp| {mx:.4f}")
    print(f"ИТОГ: max|Δp| {mx_all:.4f}, решение совпало {argmax_ok}/{n}")


def smes(port, out, раундов=1):
    R = записи()
    чаты = [{"model": ИМЯ, "messages": [{"role": "user", "content": f"Count from 1 to {10 + i} in words."}],
             "max_tokens": 64, "temperature": 0.0} for i in range(6)]
    res, вседошли = [], True
    for k in range(раундов):
        with cf.ThreadPoolExecutor(8) as ex:
            fc = [ex.submit(post, port, "/v1/systemone", r) for r in R]
            fb = [ex.submit(post, port, "/v1/chat/completions", c) for c in чаты]
            rc = [f.result() for f in fc]
            rb = [f.result() for f in fb]
        res += [{"i": i, "k": k, "t": t, "resp": d} for i, (d, t) in enumerate(rc)]
        ок_ч = sum("choices" in d for d, _ in rb)
        ок_к = sum("answers" in d for d, _ in rc)
        вседошли &= ок_ч == len(rb) and ок_к == len(rc)
        print(f"раунд {k}: чаты {ок_ч}/{len(rb)}, clef {ок_к}/{len(rc)}", flush=True)
        if ок_ч < len(rb) or ок_к < len(rc):
            print("  отказы:", [d for d, _ in rb + rc if "choices" not in d and "answers" not in d][:2], flush=True)
            break
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("СМЕСЬ:", "ВСЕ ОТВЕТЫ" if вседошли else "ЕСТЬ ОТКАЗЫ")


def smesb(port, раундов=1):
    длинные = [{"model": ИМЯ, "messages": [{"role": "user", "content": json.dumps(r, ensure_ascii=False)}],
                "max_tokens": 1, "temperature": 0.0} for r in записи()]
    чаты = [{"model": ИМЯ, "messages": [{"role": "user", "content": f"Count from 1 to {10 + i} in words."}],
             "max_tokens": 64, "temperature": 0.0} for i in range(6)]
    вседошли = True
    for k in range(раундов):
        with cf.ThreadPoolExecutor(8) as ex:
            fl = [ex.submit(post, port, "/v1/chat/completions", c) for c in длинные]
            fb = [ex.submit(post, port, "/v1/chat/completions", c) for c in чаты]
            rl = [f.result() for f in fl]
            rb = [f.result() for f in fb]
        ок_ч = sum("choices" in d for d, _ in rb)
        ок_д = sum("choices" in d for d, _ in rl)
        ток = [d.get("usage", {}).get("prompt_tokens") for d, _ in rl]
        вседошли &= ок_ч == len(rb) and ок_д == len(rl)
        print(f"раунд {k}: чаты {ок_ч}/{len(rb)}, длинные {ок_д}/{len(rl)} (токены {ток})", flush=True)
        if ок_ч < len(rb) or ок_д < len(rl):
            print("  отказы:", [d for d, _ in rb + rl if "choices" not in d][:2], flush=True)
            break
    print("СМЕСЬ БАЗЫ:", "ВСЕ ОТВЕТЫ" if вседошли else "ЕСТЬ ОТКАЗЫ")


def png(w, h, c=(30, 160, 60)):
    import struct, zlib
    raw = b"".join(b"\x00" + bytes((c[0], (c[1] + y) % 256, c[2])) * w for y in range(h))
    def ch(t, d):
        x = struct.pack(">I", len(d)) + t + d
        return x + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))


def kartinki(port, out, s_clef=False):
    import base64
    def img(w, h, c):
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(w, h, c)).decode()}}
    big, small = img(1280, 960, (30, 160, 60)), img(640, 480, (200, 40, 40))
    запросы = [{"model": ИМЯ, "max_tokens": 32, "temperature": 0.0, "messages": [{"role": "user", "content": [
        big, small, {"type": "text", "text": f"Запрос {i}: какого цвета две картинки? Кратко."}]}]} for i in range(8)]
    R = записи()
    с = [R[1], R[8]] if s_clef else []          # короткая и самая длинная (16K) записи Clef
    t0 = time.time()
    with cf.ThreadPoolExecutor(10) as ex:
        fi = [ex.submit(post, port, "/v1/chat/completions", q) for q in запросы]
        fc = [ex.submit(post, port, "/v1/systemone", r) for r in с]
        ri = [f.result() for f in fi]
        rc = [f.result() for f in fc]
    ок_и = sum("choices" in d for d, _ in ri)
    ок_к = sum("answers" in d for d, _ in rc)
    ток = [d.get("usage", {}).get("prompt_tokens") for d, _ in ri[:2]]
    отв = [d["choices"][0]["message"]["content"][:40] for d, _ in ri[:2] if "choices" in d]
    print(f"КАРТИНКИ: {ок_и}/8 за {time.time() - t0:.1f} с (токены {ток}, ответы {отв})"
          + (f", clef {ок_к}/{len(с)}" if с else ""), flush=True)
    if ок_и < 8 or ок_к < len(с):
        print("  отказы:", [str(d)[:300] for d, _ in ri + rc if "choices" not in d and "answers" not in d][:2], flush=True)
    if out != "-" and с:
        json.dump([{"i": i, "k": 0, "t": t, "resp": d} for i, (d, t) in zip((1, 8), rc)], open(out, "w"),
                  ensure_ascii=False, indent=1)


def _картинка(вид, знач, w=448, h=448):
    """синтетическая картинка с однозначным ответом -> data-URL PNG"""
    import base64, io
    from PIL import Image, ImageDraw, ImageFont
    цв = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 220, 30),
          "black": (0, 0, 0), "white": (255, 255, 255)}
    im = Image.new("RGB", (w, h), цв["white"])
    d = ImageDraw.Draw(im)
    def шрифт(n):
        # замеры 13.10 -- на DejaVuSans-Bold; без него крупный встроенный шрифт Pillow (>= 10.1)
        for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(f, n)
            except OSError:
                pass
        return ImageFont.load_default(size=n)
    if вид == "цвет":
        im = Image.new("RGB", (w, h), цв[знач])
    elif вид == "цифра":
        d.text((w // 2, h // 2), знач, fill=(0, 0, 0), font=шрифт(int(h * 0.7)), anchor="mm")
    elif вид == "фигура":
        m = w // 6
        if знач == "circle":
            d.ellipse((m, m, w - m, h - m), fill=цв["blue"])
        elif знач == "square":
            d.rectangle((m, m, w - m, h - m), fill=цв["blue"])
        else:
            d.polygon([(w // 2, m), (w - m, h - m), (m, h - m)], fill=цв["blue"])
    elif вид == "слово":
        d.text((w // 2, h // 2), знач, fill=цв["red"], font=шрифт(int(h * 0.25)), anchor="mm")
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def записи_картинки():
    """(запись, ожидание): ожидание = {вопрос: верный вариант} (для noul -- True/False)"""
    R = []
    цвета = {c: f"The image is {c}" for c in ("red", "green", "blue", "yellow")}
    for c in цвета:
        R.append(({"model": "clef", "state": {"task": "Classify the attached image."}, "images": [_картинка("цвет", c)],
                   "questions": {"color": {"type": "choice", "instructions": "What is the color of the image?",
                                           "criteria": цвета}}}, {"color": c}))
    цифры = {x: f"The digit {x}" for x in ("2", "4", "7", "9")}
    for x in цифры:
        R.append(({"model": "clef", "state": "A scanned form field.", "images": [_картинка("цифра", x)],
                   "questions": {"digit": {"type": "choice", "instructions": "Which digit is written in the image?",
                                           "criteria": цифры}}}, {"digit": x}))
    фигуры = {f: f"A {f}" for f in ("circle", "square", "triangle")}
    for f in фигуры:
        R.append(({"model": "clef", "state": "Shape inspection.", "images": [_картинка("фигура", f)],
                   "questions": {"shape": {"type": "choice", "instructions": "What shape is drawn in the image?",
                                           "criteria": фигуры}}}, {"shape": f}))
    for слово, да in (("STOP", True), ("OPEN", False)):
        R.append(({"model": "clef", "state": "Road sign photo.", "images": [_картинка("слово", слово)],
                   "questions": {"stop": {"type": "noul", "instructions": "Does the image contain the word STOP?"}}},
                  {"stop": да}))
    # порядок двух картинок: вопрос о ВТОРОЙ
    R.append(({"model": "clef", "state": "Two photos are attached, in order.",
               "images": [_картинка("цвет", "red"), _картинка("цвет", "blue")],
               "questions": {"second": {"type": "choice", "instructions": "What is the color of the second image?",
                                        "criteria": цвета}}}, {"second": "blue"}))
    # картинка + длинное состояние: спаны схемы сдвинуты и картинкой, и текстом
    R.append(({"model": "clef", "state": текст_доков(30000, 5000) + "\n\nThe ticket is about the attached picture.",
               "images": [_картинка("цифра", "7")],
               "questions": {"digit": {"type": "choice", "instructions": "Which digit is written in the attached image?",
                                       "criteria": цифры},
                             "kernels": {"type": "noul", "instructions": "Is the text about GPU kernels or GPU programming?"}}},
              {"digit": "7", "kernels": True}))
    return R


def clefkart(port, out):
    рез, верно, всего = [], 0, 0
    for i, (r, ожид) in enumerate(записи_картинки()):
        d, t = post(port, "/v1/systemone", r)
        if "answers" not in d:
            print(f"запись {i}: ОТКАЗ {str(d)[:300]}", flush=True)
            рез.append({"i": i, "resp": d, "t": t})
            всего += len(ожид)
            continue
        отв = []
        for q, e in ожид.items():
            a = d["answers"][q]
            if a["type"] == "noul":
                ok = (a["noul"] > 0.5) == e
                отв.append(f"{q}: noul={a['noul']:.3f} {'ВЕРНО' if ok else 'НЕВЕРНО'}")
            else:
                ok = a["choice"] == e
                отв.append(f"{q}: {a['choice']} p={a['probabilities'][e]:.3f} {'ВЕРНО' if ok else 'НЕВЕРНО (ждали ' + e + ')'}")
            верно += ok
            всего += 1
        print(f"запись {i}: {d['usage']['input_tokens']} ток, {t:.2f} с: " + "; ".join(отв), flush=True)
        рез.append({"i": i, "resp": d, "t": t})
    # детерминизм: первая запись ещё раз
    r0 = записи_картинки()[0][0]
    d2, _ = post(port, "/v1/systemone", r0)
    повтор = "answers" in d2 and "answers" in рез[0]["resp"] and d2["answers"] == рез[0]["resp"]["answers"]
    # отказы
    плохие = {"видео": dict(r0, videos=["x"]), "плохой URL": dict(r0, images=["http://127.0.0.1:9/none.png"]),
              "не список": dict(r0, images="abc")}
    отказы = {k: post(port, "/v1/systemone", v)[0].get("http_error") for k, v in плохие.items()}
    print(f"ИТОГ картинки: {верно}/{всего} верно; повтор побайтово {повтор}; отказы {отказы}", flush=True)
    json.dump(рез, open(out, "w"), ensure_ascii=False, indent=1)


ГОРОДА = {"tbilisi": "Tbilisi", "kazan": "Kazan", "lisbon": "Lisbon", "oslo": "Oslo"}
ЦВЕТА = ["red", "blue", "green", "black"]


def длинные_записи(max_len=32768):
    """детерминированно: сдвиг окна текста, город и цвет -- функции номера записи"""
    доки = текст_доков(10 ** 8)
    R, k = [], 0
    for L in (4000, 12000, 24000, 31000, 33000):     # 33000 -- только при max_length >= 34816 (18.10)
        if L > max_len - 1500:                       # место под схему вопросов
            continue
        for pos in (0.05, 0.5, 0.95, None):
            город = list(ГОРОДА)[(k + k // 4) % 4]          # сдвиг по длинам: позиция != ответ
            цвет = ЦВЕТА[(3 * k + 1 + 2 * (k // 4)) % 4]
            nchar = int(L * 2.57)
            off = (k * 104729) % (len(доки) - nchar)
            тело = доки[off:off + nchar]
            if pos is not None:
                факт = (f"\n\nIMPORTANT NOTE. The participants' meeting is scheduled in the city of "
                        f"{ГОРОДА[город]}. The vault password is FLAMINGO. The director's car is {цвет}.\n\n")
                i = int(nchar * pos)
                тело = тело[:i] + факт + тело[i:]
            R.append({"meta": {"L": L, "pos": pos, "city": город if pos is not None else None,
                               "car": цвет if pos is not None else None, "password": pos is not None},
                      "record": {"model": "clef", "max_length": max_len, "state": тело, "questions": {
                          "city": {"type": "choice", "criteria": ГОРОДА,
                                   "instructions": "In which city is the participants' meeting scheduled?"},
                          "password": {"type": "noul",
                                       "instructions": "Does the text say that the vault password is FLAMINGO?"},
                          "car": {"type": "choice", "criteria": {c: f"The car is {c}" for c in ЦВЕТА},
                                  "instructions": "What color is the director's car?"}}}})
            k += 1
    return R


def dlinnye(port, out, max_len=32768):
    рез, по_длине = [], {}
    for z in длинные_записи(max_len):
        m = z["meta"]
        d, t = post(port, "/v1/systemone", z["record"])
        if "answers" not in d:
            print(f"L={m['L']} pos={m['pos']}: ОТКАЗ {str(d)[:300]}", flush=True)
            рез.append({"meta": m, "resp": d, "t": t})
            continue
        a = d["answers"]
        pc, pk = a["city"]["probabilities"], a["car"]["probabilities"]
        noul = a["password"]["noul"]
        ok = {"password": (noul > 0.5) == m["password"]}
        if m["pos"] is not None:
            ok["city"] = max(pc, key=pc.get) == m["city"]
            ok["car"] = max(pk, key=pk.get) == m["car"]
            подр = (f"город {m['city']}: p={pc[m['city']]:.3f} {'ВЕРНО' if ok['city'] else 'НЕВЕРНО ' + max(pc, key=pc.get)}; "
                    f"цвет {m['car']}: p={pk[m['car']]:.3f} {'ВЕРНО' if ok['car'] else 'НЕВЕРНО ' + max(pk, key=pk.get)}; ")
        else:
            подр = f"(без фактов) макс. p города {max(pc.values()):.3f}, цвета {max(pk.values()):.3f}; "
        подр += f"пароль noul={noul:.3f} {'ВЕРНО' if ok['password'] else 'НЕВЕРНО'}"
        print(f"L={m['L']:>5} ({d['usage']['input_tokens']} ток, {t:.1f} с) pos={str(m['pos']):>4}: {подр}", flush=True)
        s = по_длине.setdefault(m["L"], [0, 0])
        s[0] += sum(ok.values()); s[1] += len(ok)
        рез.append({"meta": m, "resp": d, "t": t, "ok": ok})
    json.dump(рез, open(out, "w"), ensure_ascii=False, indent=1)
    print("ИТОГ по длинам: " + "; ".join(f"{L}: {a}/{b}" for L, (a, b) in sorted(по_длине.items())), flush=True)


def igla(port, токенов, out=None):
    import re
    # ~2.6 символа на токен в смеси ru/en документов; длину уточняем по числу, которое назвал сервер
    доки = текст_доков(10 ** 7)
    слово = "ФЛАМИНГО-393"
    нужно = int(токенов * 2.6)
    for попытка in range(5):
        тело = (доки * (нужно // max(1, len(доки)) + 1))[:нужно]
        промпт = (f"Кодовое слово: {слово}. Запомни его, в конце спрошу.\n\n" + тело +
                  "\n\nКакое кодовое слово было названо в самом начале текста? Ответь только кодовым словом.")
        d, t = post(port, "/v1/chat/completions", {"model": ИМЯ, "messages": [{"role": "user", "content": промпт}],
                                                    "max_tokens": 24, "temperature": 0.0}, timeout=3600)
        if "choices" in d:
            break
        m = re.search(r"passed (\d+) input tokens", str(d.get("body", d)))
        if not m:
            break
        нужно = int(нужно * токенов / int(m.group(1)))      # сервер назвал длину -- масштабируем
    if "choices" not in d:
        print(f"ИГЛА: ОТКАЗ {str(d)[:400]}", flush=True)
        return
    ответ = d["choices"][0]["message"]["content"]
    n = d["usage"]["prompt_tokens"]
    ok = слово in ответ
    print(f"ИГЛА: {n} токенов за {t:.1f} с ({n / t:.0f} ток/с): {ответ!r} -> {'ПРОЙДЕНО' if ok else 'ПРОВАЛ'}", flush=True)
    d2, t2 = post(port, "/v1/chat/completions", {"model": ИМЯ, "messages": [{"role": "user", "content":
                  "Столица Франции? Ответь одним словом."}], "max_tokens": 8, "temperature": 0.0})
    print(f"КОРОТКО: {d2['choices'][0]['message']['content']!r} ({t2:.1f} с)" if "choices" in d2 else f"КОРОТКО: {d2}",
          flush=True)
    if out:
        json.dump({"tokens": n, "t": t, "answer": ответ, "ok": ok}, open(out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "baza":
        baza(int(sys.argv[2]), sys.argv[3])
    elif cmd == "sravn":
        sravn(sys.argv[2], sys.argv[3])
    elif cmd == "clef":
        clef(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif cmd == "clefsr":
        clefsr(sys.argv[2], sys.argv[3])
    elif cmd == "smes":
        smes(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif cmd == "kartinki":
        kartinki(int(sys.argv[2]), sys.argv[3], len(sys.argv) > 4 and sys.argv[4] == "clef")
    elif cmd == "clefkart":
        clefkart(int(sys.argv[2]), sys.argv[3])
    elif cmd == "dlinnye":
        dlinnye(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 32768)
    elif cmd == "igla":
        igla(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "smesb":
        smesb(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
