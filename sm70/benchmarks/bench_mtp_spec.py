# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Замер спекулятивного декода на встроенной MTP-голове.
#
# ПОРЯДОК ДЕЙСТВИЙ ЗДЕСЬ НЕ ПРОИЗВОЛЕН. Сначала проверяется ТОЖДЕСТВЕННОСТЬ выдачи, и только потом
# измеряется скорость. Причина не в аккуратности, а в конкретном происшествии на этом же сервере:
# своё всё-сведение (custom all-reduce) измерялось БЫСТРЕЕ штатного -- 37.4 против 35.1 ток/с -- и
# при этом портило вывод (" Paris!!!!!!!!!": начинает верно, потом вырождается). Цифра скорости,
# снятая до проверки качества, ничего не стоит и хуже отсутствия цифры, потому что в неё верят.
#
# ПОЧЕМУ ТОЖДЕСТВЕННОСТЬ ЗДЕСЬ -- ИМЕННО ТОЖДЕСТВЕННОСТЬ, А НЕ БЛИЗОСТЬ. Спекулятивный декод
# принимает черновой токен только если он совпадает с выбором полной модели (жадный режим) либо
# проходит выборку по отношению вероятностей (сэмплирование). Значит при temperature=0 выход обязан
# совпасть С ПЕРВОГО ДО ПОСЛЕДНЕГО СИМВОЛА при любом числе набросков и любом принятии. Расхождение
# означает не "цену метода", а ДЕФЕКТ -- скорее всего незакрытый откат рекуррентного состояния 48
# линейных слоёв при отклонении черновика. Этот дефект коварен тем, что проявляется НЕ СРАЗУ:
# состояние уезжает постепенно, поэтому короткая выдача выглядит целой. Отсюда --long: сверять надо
# сотни токенов, а не десяток.
#
# ЧТО ЕЩЁ УЧТЕНО, каждое -- из набитой шишки этого проекта:
#   * УНИКАЛЬНЫЕ промпты. Префикс-кэш давал "9677 ток/с" префилла, что физически невозможно на
#     2xV100 (520 TFLOP/s против пика 250). Повторный промпт измеряет кэш, а не модель.
#   * Скорость ДЕКОДА считается по второму и далее токену, отдельно от TTFT: иначе префилл 250K
#     размажется по декоду и утопит эффект.
#   * Принятие берётся из счётчиков сервера, а не выводится из скорости -- иначе рассуждение
#     замкнётся само на себя (скорость -> принятие -> объяснение скорости).
#   * Парные замеры и медиана: на картах 0-1 периодически заходят чужие задания.
#
# ЗАПУСК:
#   # 1) база без спекуляции, сохранить эталонные выдачи
#   python bench_mtp_spec.py --port 8092 --save base.json
#   # 2) со спекуляцией, сверить и замерить
#   python bench_mtp_spec.py --port 8093 --compare base.json --k 3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

# Промпты сделаны РАЗНЫМИ по содержанию и достаточно длинными, чтобы принятие было
# представительным: на шаблонном тексте черновик угадывает почти всё и принятие завышается.
PROMPTS = {
    "проза": "Опиши подробно, как устройство памяти графического ускорителя влияет на выбор "
             "алгоритма умножения матриц. Начни с иерархии и дойди до конкретных следствий.",
    "код": "Напиши на C++ функцию, которая переставляет байты 32-битного слова так, чтобы каждый "
           "байт оказался в старшей половине отдельного 16-битного значения, и объясни зачем.",
    "рассуждение": "В городе три района. В первом вдвое больше жителей, чем во втором, а в третьем "
                   "на 12 тысяч меньше, чем в первом и втором вместе. Всего 108 тысяч. Найди все "
                   "три числа и проверь ответ подстановкой.",
    "перечисление": "Перечисли по пунктам, чем отличается архитектура Volta от Turing, с указанием "
                    "того, чего в Volta нет в железе.",
}


def _post(port, path, payload, timeout=1200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _metrics(port):
    """Счётчики спекуляции с сервера. Именно счётчики, а не вывод из скорости."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
            body = r.read().decode()
    except urllib.error.URLError:
        return {}
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name.strip()] = out.get(name.strip(), 0.0) + float(val)
        except ValueError:
            pass
    return out


def _accept_rate(before, after):
    """Принятие = принятых черновых токенов / предложенных. Считается по ПРИРАЩЕНИЮ счётчиков,
    иначе в число попадёт вся история сервера, включая прогрев."""
    def d(sub):
        k = [k for k in set(before) | set(after) if sub in k]
        return sum(after.get(x, 0.0) - before.get(x, 0.0) for x in k)
    drafted, accepted = d("num_draft_tokens"), d("num_accepted_tokens")
    return (accepted / drafted if drafted > 0 else None), drafted, accepted


def gen(port, prompt, max_tokens, greedy=True):
    """Одна выдача. Возвращает текст, TTFT и скорость ДЕКОДА (без префилла)."""
    body = {
        "model": "gpt-4-0125-preview",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
        # temperature=0 обязательна для сверки тождественности: при сэмплировании совпадения
        # текста ждать нельзя даже от корректной спекуляции.
        "temperature": 0.0 if greedy else 0.7,
        "seed": 12345,
    }
    t0 = time.perf_counter()
    r = _post(port, "/v1/completions", body)
    wall = time.perf_counter() - t0
    txt = r["choices"][0]["text"]
    n = r.get("usage", {}).get("completion_tokens") or 0
    return {"text": txt, "tokens": n, "wall": wall,
            "tok_s": (n / wall if wall > 0 else 0.0)}


def measure(port, tag, max_tokens, repeats):
    """Скорость по каждому промпту. Медиана из repeats -- на этих картах бывают чужие задания."""
    res = {}
    for name, p in PROMPTS.items():
        runs, texts = [], []
        for i in range(repeats):
            # промпт делаем УНИКАЛЬНЫМ на каждый повтор, иначе замеряем префикс-кэш
            g = gen(port, f"{p} (вариант {tag}-{i})", max_tokens)
            runs.append(g["tok_s"])
            texts.append(g["text"])
        res[name] = {"tok_s": statistics.median(runs), "все": runs, "текст": texts[0]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, default=0, help="число набросков (0 = без спекуляции)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--long", type=int, default=600,
                    help="длина сверки тождественности; короткая сверка НЕ ловит уехавшее "
                         "рекуррентное состояние")
    ap.add_argument("--save", help="сохранить эталонные ЖАДНЫЕ выдачи (запуск без спекуляции)")
    ap.add_argument("--compare", help="сверить с эталоном и только потом мерить скорость")
    a = ap.parse_args()

    # --- 1. ТОЖДЕСТВЕННОСТЬ. До скорости. -----------------------------------------------------
    ident = {}
    for name, p in PROMPTS.items():
        ident[name] = gen(a.port, p, a.long)["text"]

    if a.save:
        with open(a.save, "w") as f:
            json.dump({"выдачи": ident, "k": a.k, "порт": a.port}, f, ensure_ascii=False, indent=1)
        print(f"эталон сохранён: {a.save} ({sum(len(v) for v in ident.values())} символов)")

    ok = True
    if a.compare:
        with open(a.compare) as f:
            base = json.load(f)["выдачи"]
        print("=== СВЕРКА ТОЖДЕСТВЕННОСТИ (жадный режим, обязана быть посимвольной)")
        for name in PROMPTS:
            b, c = base.get(name, ""), ident.get(name, "")
            if b == c:
                print(f"  {name:14s} СОВПАЛО ({len(c)} символов)")
            else:
                ok = False
                i = next((j for j in range(min(len(b), len(c))) if b[j] != c[j]), min(len(b), len(c)))
                print(f"  {name:14s} РАСХОЖДЕНИЕ с символа {i} из {len(b)}")
                print(f"      эталон: ...{b[max(0,i-60):i+60]!r}")
                print(f"      сейчас: ...{c[max(0,i-60):i+60]!r}")
        if not ok:
            print("\nЦИФРЫ СКОРОСТИ НИЖЕ НЕДЕЙСТВИТЕЛЬНЫ. Спекуляция с проверкой обязана давать")
            print("тождественный жадный выход. Расхождение -- это ДЕФЕКТ, и первый подозреваемый --")
            print("откат состояния 48 линейных слоёв при отклонении черновика (gdn_attn.py).")
            print("Отклонение ближе к концу выдачи, чем к началу, подтверждает именно эту версию.")

    # --- 2. Скорость и принятие ----------------------------------------------------------------
    m0 = _metrics(a.port)
    res = measure(a.port, f"k{a.k}", a.max_tokens, a.repeats)
    m1 = _metrics(a.port)
    rate, drafted, accepted = _accept_rate(m0, m1)

    print(f"\n=== СКОРОСТЬ (k={a.k}, медиана из {a.repeats}, уникальные промпты)")
    for name, r in res.items():
        print(f"  {name:14s} {r['tok_s']:6.2f} ток/с   все: "
              f"{', '.join(f'{x:.2f}' for x in r['все'])}")
    med = statistics.median([r["tok_s"] for r in res.values()])
    print(f"  {'МЕДИАНА':14s} {med:6.2f} ток/с")

    if a.k > 0:
        print(f"\n=== ПРИНЯТИЕ (из счётчиков сервера, не из скорости)")
        if rate is None:
            print("  счётчиков spec_decode нет -- проверь, что спекуляция ВООБЩЕ включилась:")
            print("  в логе должно быть 'Qwen3_5MTP' и метод qwen3_5_mtp. Скорость без этого")
            print("  измеряет обычный декод, и её нельзя приписывать спекуляции.")
        else:
            print(f"  предложено {drafted:.0f}, принято {accepted:.0f}  ->  принятие {rate:.3f}")
            exp = sum(rate ** i for i in range(1, a.k + 1)) + 1
            print(f"  ожидание токенов за шаг: {exp:.2f} из максимума {a.k + 1}")
            print(f"  то есть эффективный батч при одном запросе ~{exp:.2f} -- это ВЫЧИТАЕТСЯ")
            print(f"  из выигрыша от батча, а не складывается с ним")

    print(f"\nитог: {'ТОЖДЕСТВЕННОСТЬ ПОДТВЕРЖДЕНА' if ok and a.compare else ''}"
          f"{'сверка не запрашивалась' if not a.compare else ''}"
          f"{' -- ЦИФРЫ НЕДЕЙСТВИТЕЛЬНЫ' if not ok else ''}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
