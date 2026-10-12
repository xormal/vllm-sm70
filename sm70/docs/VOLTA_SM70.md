# Volta / `sm_70` (V100) — architecture reference for kernel work

Working reference for this project. Two kinds of entry, always distinguished:

- **[spec]** — from the hardware/ISA documentation. Fixed.
- **[measured]** — measured on *this* machine (V100-SXM2-32GB, CUDA 12.8, clocks locked at 1350 MHz
  unless stated). These are the numbers to plan against; vendor peaks are not achievable.

Every performance win in this repo so far came from an entry in this file, not from a better
algorithm. That is the reason it exists.

**Where the toolchain diverges from the machine — `COMPILER_VS_HARDWARE.md`** (Russian): measured
cases where nvcc / ptxas / cutlass do something other than what this file's hardware model implies —
the `__launch_bounds__` register ceiling and its silent spill, local memory that is not spill,
constant memory serialising on divergent lanes, `--use_fast_math` flushing legitimate subnormals, and
cutlass "walls" that are integer divisions reaching zero. Each entry carries its detection recipe.

**Writing a kernel by hand — start from `SM70_KERNEL_PLAYBOOK.md`.** This file describes the
machine; the playbook is the procedure: the instruction's operand map and all four of its variants,
what the compiler gets wrong, the tile-geometry law, the three-way trade-off (occupancy / traffic /
loads-per-MMA) and the measurement discipline. This file answers "what machine"; the playbook
answers "how to build on it".

> `SM70_KERNEL_PLAYBOOK.md` and `SM70_LATENCY_HIDING.md` are **written in Russian** — they are the
> working engineering notebooks behind this port, kept verbatim rather than translated so the
> measurements and their reasoning stay attached to each other. Everything they establish that a
> user needs is restated in English in this file and in `README.md` §2.

---

## 1. The SM, and the one equation that decides occupancy

| Quantity | Value | Note |
|---|---|---|
| SMs | **80** | [spec] |
| Register file | **65536 × 32-bit per SM** | [spec] |
| Max registers per thread | **255** | [spec] hard ISA cap |
| Warp slots per SM | **64** | [spec] |
| Shared memory per SM | **96 KB** (of a 128 KB unified L1+smem) | [spec] |
| Warp schedulers per SM | 4 (one per processing block) | [spec] |
| FP32 lanes per SM | 64 | [spec] |
| Tensor cores per SM | 8 (HMMA, `m8n8k4` native shape) | [spec] |

**The occupancy equation.** Warps resident per SM is
`floor(65536 / regs_per_thread / 32)`, and **the CTA shape cannot change it**:

```
255 regs -> 65536/255/32 =  8.03 ->  8 warps  (12.5% occupancy)
168 regs -> 65536/168/32 = 12.19 -> 12 warps  (18.8%)
128 regs -> 65536/128/32 = 16.00 -> 16 warps  (25.0%)
 96 regs -> 65536/ 96/32 = 21.33 -> 21 warps  (32.8%)
```

*[measured]* Our `d>128` backward sits at exactly 255 regs → 12.5% warps active, and **torch's
kernel sits at the same 8 warps**. When two implementations are both register-capped, the contest
is decided by what happens *inside* those 8 warps, never by fill. Chasing the gap from 255 down to
torch's 240 is pointless — same 8 warps.

**Consequence for planning:** below ~128 regs/thread, occupancy doubles and latency hiding becomes
possible; above it, the kernel *must* hide latency by instruction-level and memory-level
parallelism, because there are no other warps to switch to.

---

## 2. Memory hierarchy — what is actually reachable

| Path | Vendor peak | *[measured]* achievable | How measured |
|---|---|---|---|
| HBM2 read | 900 GB/s | **841 GB/s** | `sum()` over 256 MB fp32 |
| HBM2 read+write | 900 GB/s | **819 GB/s** | `copy_` 256 MB fp32 |
| fp16 read | — | **774 GB/s** | `sum()` fp16 |

Plan against **~820 GB/s**, not 900. A kernel at "47% of peak" in `ncu` is really at ~52% of what
the card gives.

*[measured] trap:* `torch.uint8.sum()` reports **47 GB/s** — it upcasts elementwise and is not a
bandwidth test. Do not use integer reductions to calibrate bandwidth; use fp32 `copy_`/`sum`.

Latency *[spec, order of magnitude]*: global ~400–800 cycles, L2 ~200, shared ~20–30, register ~0.
**One global miss costs more than a hundred arithmetic instructions** — this is why the stall
taxonomy in §4 matters more than instruction counts.

---

## 3. Instruction pipes — and what Volta does NOT have in hardware

Volta issues one instruction per warp scheduler per cycle. The pipes are separate, and `ncu`
counts them separately: `pipe_alu`, `pipe_fma`, `pipe_lsu` (memory), `pipe_tensor` (HMMA),
`pipe_xu` (transcendental: `ex2`, `rcp`, `sqrt`).

**No hardware support for — everything below is emulated in ALU instructions:**

| Missing | Emulated cost | Where it bit us |
|---|---|---|
| **fp8 (e4m3/e5m2)** | ~12 ALU ops per element with a denormal branch | **85% of the decode kernel's ALU.** Fixed → **1.50–1.81×** |
| **fp16 atomics** | CAS retry loop | measured **2× slower** than fp32 atomics; keeps the `dQ` fold in fp32 |
| bf16 arithmetic | — | not usable; we cast bf16→fp16 at the boundary |
| Async copy (`cp.async`) | — | sm_80+. Volta must do explicit `LDG` → register → `STS` |
| Independent thread scheduling helps, but there is **no** hardware prologue overlap | — | see §5 |

**The fp8 ↔ fp16 bit trick** (this is the single highest-value entry in this file). fp16 is
`E5M10` bias 15; e4m3 is `E4M3` bias 7. The fields are the same shape, only narrower:

```
read  (e4m3 -> float):  h = (b & 0x80) << 8 | (b & 0x7F) << 7;  value = __half2float(h) * 256.0f
write (float -> e4m3):  h = __float2half_rn(a * (1/256.0f));    byte = round_ne(h >> 7)
```

Three instructions instead of twelve, **no branch** — and denormals need no special case in the
read direction, because an fp16 denormal `0.MMM·2⁻¹⁴` scaled by `2⁸` *is* the e4m3 denormal
`0.MMM·2⁻⁶`. *[measured]* This also turned out to be **more accurate** than the hand-rolled
bit-picking it replaced: relL2 vs torch `2.6e-3 → 2.6e-7`.

*[measured] asymmetry worth remembering:* the same trick in the **write** direction *does* need a
denormal branch, because e4m3 denormals scaled by `2⁻⁸` land in fp16's own denormal range where
fp16 no longer has the bits — double rounding, 0.08% of elements off by one step. Compute those
directly (`M = round(a·2⁹)`). The read direction is exact; the write direction is not. Symmetry of
the formats does not imply symmetry of the rounding.

---

## 3b. Учёт по СЛОТАМ ВЫДАЧИ — единица счёта, которой мы считаем вспомогательную работу *[проверено 2026-07-31]*

Раздел появился, потому что мы дважды объяснили один и тот же эффект неверно: сначала «полосой», потом
«фазировкой». Обе гипотезы опровергнуты числом. Верная единица счёта — **слот выдачи**.

### Правило

Планировщик варпов выдаёт **одну команду за такт**. Тяжёлая тензорная команда занимает конвейер на
τ тактов, но **слот выдачи — ровно один**. Значит на каждую тяжёлую команду приходится **τ−1 свободных
слотов**, и вспомогательная работа помещается в тень тогда и только тогда, когда

> **ν ≤ τ − 1**,  где **ν = число выдач нетензорных команд на одну тензорную.**

Это точная граница, а не эвристика: задача сводится к modulo scheduling.

### Что в неё НЕ входит — и это главное

| величина | входит? | почему |
|---|---|---|
| **число выдач ν** | **ДА, это и есть предмет счёта** | слот занят или свободен, третьего нет |
| задержка команды λ | **нет** | не входит ни в одно условие вообще |
| глубина цепочки зависимостей | **нет** | при свободном опережении **не связывает никогда** |
| суммарная работа Σd | нет | это тень пары (ν, критическая длина) |
| сумма закодированных тактов стойла | **нет** | не является заменителем периода; верна лишь при строго внутрипорядковой выдаче |

Отсюда практический вывод, обратный интуиции: **вспомогательную работу надо делать не «дешевле» и не
«мельче по зависимостям», а РЕЖЕ ПО ВЫДАЧАМ.** Три команды, выданные подряд и зависимые друг от друга,
стоят ровно столько же, сколько три независимые: компилятор вынесет их вперёд (замерено: `LOP3`
поднимается на ~16 позиций от потребителя).

### Вторая половина правила: КТО выдаёт

Слот прячется, только если его есть кому занять. Прикрытие даёт **число варпов на планировщик**:

| роль | варпов на планировщик | что происходит со свободным слотом |
|---|---|---|
| считающие | 2 | пока один ждёт, второй выдаёт — слот занят |
| **везущие** | **1** | занять некем — слот простаивает |

Замерено на одинаковом искусственном приросте команд: **одна и та же лишняя выдача у везущих стоит в
1.18…1.94 раза дороже**, чем у считающих, и множитель растёт с длиной.

Следствие для проектирования: **перенос работы из везущей роли в считающую выгоден, даже если общее
число выдач при этом ВЫРАСТЕТ.**

### Пример: почему int8 отстаёт от e5m2 на 5–7% *[замерено]*

Два варианта ОДНОГО ядра, та же плитка, та же геометрия, оба однобайтовые (трафик одинаков — вклад
полосы **ровно ноль**). Отличается только доставка операнда:

| | e5m2 | int8 + масштаб на позицию |
|---|---|---|
| распаковка | одна перестановка (`PRMT`): старший разряд хранимого числа **и есть** старший разряд fp16 | `LOP3` (знак) + `PRMT` |
| масштаб | не нужен | отдельная доставка + адресная арифметика |
| **ν (выдач на тензорную)** | **1.41** | **1.67** |

Лишние выдачи и их цена:

| статья | сколько выдач | у кого |
|---|---|---|
| арифметика распаковки (XOR знака) | **+128** `LOP3` | считающие (прикрыты) |
| доставка масштабов | +14 загрузок, но **+36 `IMAD` и +11 `IADD3`** | везущие (не прикрыты) |

> **ВНИМАНИЕ: разложение «35 % распаковка / 65 % масштабы», стоявшее здесь до 31.07 вечера,
> ОПРОВЕРГНУТО** прямой переписью SASS по вариантам ядра. Верно обратное: XOR знака — **основная
> статья**, а не треть. Его снятие (хранить число со смещением: `u = v + 128`, тогда развернуть можно
> одной перестановкой, как у соперника) роняет общий прирост команд с **+7.56 % до +2.63 %**.
> Второй урок из того же разбора: остаток по масштабам — это **адресная арифметика, а не загрузки**
> (+36 `IMAD` при всего +14 `LDG`); лечится чередованием таблиц двух наборов (одна широкая загрузка
> на обе) и однократным вычислением адреса плитки. Итог: int8 стал **быстрее** соперника при
> **побитово той же** точности (`torch.equal` = True).

**Где закон «команды складываются 1:1» ЛОМАЕТСЯ.** Попытка убрать таблицу масштабов целиком дала
**−96 команд и БОЛЬШЕЕ время**: компилятор при этом уходит со 134 регистров на 119 и портит
расписание. То есть счёт выдач — верхний слой модели, под ним лежит давление на регистры, и оно
способно перевесить. Практический вывод: после каждой правки, сокращающей команды, смотреть
`ptxas -v` — падение числа регистров при росте времени и есть подпись этого случая.

### τ и лестница ν по нашим ядрам *[выведено и подтверждено 31.07]*

**τ = 2 такта на SASS-команду `HMMA.884.F32.F32`**, то есть бюджет `τ−1 = 1.00` выдачи на тензорную.
Выведено арифметикой (125.3 ТФЛОП/с ÷ 80 SM ÷ 4 планировщика = 128 МАС/такт; одна `HMMA.884` даёт
256 МАС на варп) и **подтверждено независимо**: у `volta_fwd_block`, замеренного на 95.2 % тензорного
пика, счёт даёт ν = 0.95 при бюджете 1.00. Совпадение секундомера со счётом — проверка правила, а не
подгонка под него.

| ядро | ν | потолок доли тензорного пика ТОЛЬКО по слотам |
|---|---|---|
| `mem_eff` (cutlass) | **12.05** | 15.3 % |
| `volta_fwd_i8` двухблочный | 2.60 | 55.6 % |
| ядро ролей int8 (боевое) | 1.70 | 63.5 % |
| e5m2 | 1.45 | 70.1 % |
| `volta_bwd_block` | 1.28 | 87.6 % |
| `volta_fwd_block` | **0.95** | **у стены** |

Оба боевых форвардных int8-ядра — **за стеной**: там ν > τ−1, и каждая лишняя выдача стоит полный такт.
Это и объясняет, почему выигрыш над cutlass был кратным (12.05 → 0.95, переход из-за стены под стену),
а разрыв int8/e5m2 — процентным (1.70 против 1.45, оба у стены).

### ВТОРОЙ РЕСУРС: задержка обращения, выставленная на рандеву *[открыто 31.07]*

Счёт слотов — верхний слой модели, и он **не объясняет всего**. Прямая проверка: замеренная доля
отставания на доставке масштабов ≈ 65 %, а по числу выдач, даже с максимальным множителем роли,
получается 16–24 %. Значит эта цена **не выводится из количества слотов вообще**.

Где она сидит, видно в SASS: масштабные `LDG.E.CONSTANT.SYS` стоят за ~10 команд до `STS` и **прямо
перед рандеву плитки** (`BAR.SYNC`). Их задержка ложится на защёлку: варп встаёт на барьере, и прикрыть
его нечем, потому что на барьере ждут все.

Практический вывод, которого из счёта выдач не получить: **обращение, поставленное близко к барьеру,
стоит не одну выдачу, а свою задержку целиком**. Лечится не сокращением команд, а разносом: поднять
обращение раньше по телу либо убрать сам поток (масштаб на блок вместо позиции).

Это ровно то, что на нашем внутреннем языке называлось «фазировка STS/LDS»: интуиция указывала на
верный класс операций и верное место (обращения к разделяемой памяти около рандеву), а неверным было
лишь приписанное объяснение (порядок и глубина). Формально это **второй ресурс** — конвейер обращений
со своей пропускной способностью и задержкой, — и в модели «слот занят или свободен» его нет.

### Как считать ν на практике

`cuobjdump --dump-sass` по собранной библиотеке, выделить мейнлуп, поделить число нетензорных команд
на число тензорных. `ncu` на этой машине нет; управляющие биты Volta (счётчик стойла, `yield`, барьеры)
читаются из того же дампа, но **суммировать их бесполезно** — см. таблицу выше.

Это автоматизировано: **`tools/issue_slots.py`** — выделяет мейнлуп по доминаторам, считает ν,
разводит выдачи **по ролям** (ветвление по `threadIdx` + достижимость блоков), даёт бюджет и вердикт
«под стеной / у стены / за стеной», сравнивает два ядра (`--diff`) и предлагает кандидатов на
сокращение. `--selftest` воспроизводит пример ниже из SASS независимо.

**τ = 2 такта на одну SASS-команду `HMMA.884.F32.F32`** (в разделе этого числа раньше не было):
125.3 TFLOP/s ÷ 80 SM ÷ 1530 МГц = 511 МАС/такт на SM = 128 на планировщик, а одна `HMMA.884` несёт
256 МАС на варп. Значит **на пике свободен ровно ОДИН слот**, `ν ≤ 1`. Число подтверждено независимо:
у `volta_fwd_block`, замеренного на **95.2 % тензорного пика**, инструмент даёт **ν = 0.95** — ядро
у самой границы. Для масштаба, тем же счётом: `mem_eff` cutlass ν = 12.0 (потолок 15 % пика),
двухблочный `volta_fwd_i8` ν = 2.60 (56 %), ядро ролей ν = 1.70 (63 % с учётом везущих),
`volta_bwd_block` ν = 1.28 (88 %).

### ГРАНИЦА ПРИМЕНИМОСТИ СЧЁТА ПО СЛОТАМ *[замерено 2026-07-31 вечером, боевое ядро ролей int8]*

Раздел выше говорит: «ядро за стеной (nu=1.71 > 1.00), каждая лишняя выдача стоит полный такт».
Прямая проверка это **опровергла для этого ядра**. Цену выдачи замерили ХОЛОСТОЙ командой
(фальсификаторы `KVFMT=2/3`: `LOP3` с нулевой маской из постоянной памяти, ответ побитово тот же):

| куда добавлена работа | выдач | доля слотов | прирост времени | **цена слота** |
|---|---|---|---|---|
| считающие, ХОЛОСТАЯ команда | +128 на варп | +8.7 % | **+1.93 %** | **0.22 такта** |
| везущие, ХОЛОСТАЯ команда | +64 на варп | +2.2 % | **+0.65 %** | **0.30 такта** |
| считающие, та же команда В ЦЕПИ | +128 на варп | +8.7 % | **+4.24 %** | **0.49 такта** |

**ЦЕНА СЛОТА -- НЕ ОДНО ЧИСЛО: ОНА ЗАВИСИТ ОТ МЕСТА В ЦЕПИ ЗАВИСИМОСТЕЙ.** Первые две строки --
холостая `LOP3` с нулевой маской: её результата никто не ждёт, и она падает в простаивающие слоты.
Третья -- РОВНО ТО ЖЕ ЧИСЛО выдач (+128 на варп), но это настоящий XOR знака из разворота байта,
стоящий на цепи, которая кормит тензорную операцию: пара форматов 0 и 6 отличается ровно этой
командой. Одинаковый счёт -- **разница по времени в 2.2 раза**.

Оба замера воспроизведены независимо 31.07 (чередование внутри раунда, 21 раунд, карта свободна,
частоты 1530): холостая даёт +2.10 / +1.92 / +1.89 % при Sk 8K/32K/128K, в цепи -- +4.09 / +4.33 /
+4.45 %. Отдельная проверка на РАЗНЫХ приращениях подтверждает линейность цепного случая: +513 выдач
-> +16.2 % (предсказано 17.0), +766 -> +23.8 % (предсказано 25.4).

**Практическое следствие: калибруйте той командой, которую собираетесь добавлять, и на том месте,
куда собираетесь её ставить.** Холостая команда даёт нижнюю оценку цены, а не цену.

То есть слот стоит **0.22-0.49 такта, а не 1.00**. Отношение ролей (везущие дороже в ~1.4 раза)
раздел подтверждает, АБСОЛЮТНУЮ величину — нет. Причина видна из сверки с секундомером: ядро идёт
на **39.7 % тензорного пика** (Sk=32768, поправка на квантование волн) при потолке по слотам
**65.2 %**, то есть на 61 % своего же счётного потолка. **Ядро НЕ У СЧЁТНОЙ СТЕНЫ**, и пока это так,
сокращение выдач возвращает лишь свою долю от 0.22-0.39.

**Четыре правки, уронившие счёт и НЕ уронившие время** (все побитово проверены, карты 0 и 1, медиана
парных отношений, 17 раундов, чужих процессов 0):

| правка | что сделала со счётом | потолок | **время** |
|---|---|---|---|
| перекладка 4 Б -> 16 Б у везущих (кандидат A8 советчика) | везущие 620 -> 403 выдач | 65.2 -> **70.0 %** | **0.952** (хуже) |
| то же, с разведением загрузок и укладок | то же | то же | 0.952 |
| перестановка кусков K против банкового конфликта | счёт не меняет | — | **0.983** (хуже) |
| расщепление цепочки накопителя первого умножения по k | nu 1.463 -> 1.480 (ХУЖЕ) | 64.8 % | 0.990 |
| погруппное рандеву считающих (фазы разведены) | счёт не меняет | — | 1.000 (ноль) |

Отдельно: **порядок по слотам и порядок по секундомеру у форматов 6/7/8 ПРОТИВОПОЛОЖНЫ**. У `KVFMT=7`
слотов меньше всех (2814, потолок 72.8 %), у `KVFMT=8` больше всех (3142, потолок 65.2 %) — а по
времени 8 выигрывает у 7 на 2 % и у боевого 0 на **6.1-6.7 %** (побитово, `torch.equal` = True).

### ЧТО СВЯЗЫВАЕТ НА САМОМ ДЕЛЕ: разложение фальсификаторами *[замерено]*

Собраны варианты ядра со снятой ФАЗОЙ (ответ неверен, меряется только время):

| снято | прирост скорости | доля времени фазы |
|---|---|---|
| ПЕРВОЕ умножение (Q·K^T) | x1.541 | **35.1 %** |
| ВТОРОЕ умножение (P·V) | x1.245 | **19.7 %** |
| софтмакс целиком (максимум, exp2, редукции) | x1.180 | **15.2 %** |
| подача (везущие не везут, рандеву осталось) | x1.061 | **5.7 %** |
| два внутренних рандеву считающих | x1.020 | 2.0 % |

> ### ⬜ ЭТА ТАБЛИЦА НЕ СХОДИТСЯ: 35.1 + 19.7 + 15.2 + 5.7 + 2.0 = **77.7 %**
>
> **Пятая часть времени префилла лежит в фазах, которые мы ни разу не назвали.** Таблица простояла
> здесь в таком виде, и никто (в том числе я) не сложил столбец — при том что по доктрине этого
> проекта единица анализа именно ФАЗА.
>
> Оговорка, без которой нельзя: снятие фазы меняет ПЕРЕКРЫТИЕ, поэтому доли не обязаны давать ровно
> 100 % — фаза, идеально спрятанная за другой, при снятии даст меньше своей занятости. Но 22 пункта
> это не погрешность округления, это отдельный предмет.
>
> Кандидаты, не покрытые ни одной строкой таблицы (каждый — гипотеза до замера):
>
> * **пересчёт накопителя** онлайн-софтмакса: `BQ·d` умножений НА ПЛИТКУ на `exp(m_old − m_new)`;
>   при d=256 сопоставимо с самим умножением, а в «софтмакс» (максимум/exp2/суммы) не входит;
> * **эпилог** — запись `O` (`BQ·d` fp16 на блок) и `LSE`; на бэкварде эпилог оказался решающим,
>   на префилле не мерен ни разу;
> * **пролог** — заполнение конвейера, загрузка Q, инициализация; виден как свободный член в
>   зависимости времени от `Sk`, то есть меряется БЕЗ правки кода;
> * **квантование по волнам и перекос по SM** — 80 SM, дробная последняя волна; у бэкварда на той же
>   машине `ncu` даёт 3.20 волны и перекос «максимум на 16.96 % выше среднего»;
> * **обвязка на ВХОДЕ** — выход мы починили (BSHD-direct), вход по фазам не разбирали.
>
> **ПОПРАВКА (тот же день, разбор шести ресурсов).** Формулировка выше — «фазы, которые мы не
> назвали» — скорее всего НЕВЕРНА. Шесть кандидатов дают в сумме 3.2–8.9 % (центр 5.5) против 22.3,
> и все шесть оценок СЧЁТНЫЕ, то есть верхние границы: настоящая сумма ниже, а дыра больше 17
> пунктов. Дыра — вероятно **артефакт метода**, а не адрес в коде. Три механизма, каждый даёт
> недостачу без единой пропущенной фазы:
>
> * **(A) сублинейность снятия** — три канала за одной очередью MIO: снятие любой фазы освобождает
>   канал остальным, поэтому КАЖДАЯ доля занижена примерно на общий множитель. `77.7/100 = 0.78` —
>   ровно тот масштаб. Косвенное подтверждение уже лежало здесь же: **`PHSPLIT = 1.000`**
>   (перестановка фаз не меняет ничего ⇒ связывает общий ресурс, а не порядок);
> * **(B) постоянная харнесса внутри окна замера** — события ставятся ВОКРУГ вызова с синхронизацией
>   каждой итерации, поэтому 20–60 мкс хостовой обвязки лежат внутри `t`; её не снимает ни один
>   фальсификатор, все доли делятся на `(G+K)`, и сумма физически не может дойти до 100 %;
> * **(C) неснятая поплиточная работа** — причинная маска, адресная арифметика, счётчики цикла,
>   запись `sP`, ожидание на `bar.sync`.
>
> **Решающий зонд стоит НОЛЬ правок: дыра как функция `Sk`.** `G` постоянна, `K` растёт квадратично.
> Сожмётся при `Sk=131072` с 22.3 до ~5 % — механизм (B). Останется ~22 % — (A) или (C), и тогда
> следующий ход не «искать фазу», а **тест сложимости**: снять ОБА умножения разом и сравнить с
> аддитивными 54.8 %; больше 60 % — перекрытие доказано, охота по фазам бессмысленна.
>
> Порядок работ отсюда: сперва зонды БЕЗ правки кода (дыра по `Sk`; пролог по свободному члену;
> волны по парам форм, отличающимся только дробной частью), затем тест сложимости, и только потом
> снятия отдельных фаз.
>
> **Урок метода.** Разложение по фазам — по-прежнему единственный способ увидеть время, которого нет
> в работе. Но у него есть своя слепая зона: **доли, полученные снятием, — НИЖНИЕ оценки**, потому
> что снятие меняет перекрытие и освобождает общий канал. Складывать их и объявлять остаток
> «ненайденной фазой» — ошибка, и я её здесь совершил.

Читать это надо так:

1. **Подача НЕ связывает** (5.7 %) — поэтому любая оптимизация везущих ограничена сверху этой цифрой,
   и широкая перекладка не могла выиграть, даже сработав идеально по счёту.
2. **Оба умножения несут РОВНО одинаковую тензорную работу (256 HMMA на варп на плитку) и одинаковое
   число вспомогательных выдач (208), но стоят 35.1 % против 19.7 %.** Разница 795 тактов на плитку.
   Ровно столько же даёт разница их трафика в РАЗДЕЛЯЕМОЙ памяти: первое умножение читает операнд Q
   64 раза по 8 Б на полосу (1024 такта MIO на плитку на SM), второе читает P 16 раз (256 тактов).
   **768 расчётных против 795 замеренных.** Связывает ПРОПУСКНАЯ СПОСОБНОСТЬ РАЗДЕЛЯЕМОЙ ПАМЯТИ
   (128 Б/такт на SM), причём внутри фаз умножения она загружена на 76-85 %.
3. Трафик Q неустраним при плитке варпа 16x16: карта фрагмента `mma.884` выдаёт КАЖДОЕ значение
   операнда ДВУМ полосам, поэтому 128 Б полезных стоят 2 такта (64 Б/такт полезной полосы). Обмен
   через `SHFL` вместо дублирующего чтения не помогает: у Volta `SHFL` идёт по тому же MIO с той же
   пропускной способностью (32 результата/такт/SM) — размен один в один. Единственный настоящий
   рычаг — БОЛЬШАЯ плитка варпа (MB=2, то есть BQ=64), а она не влезает в разделяемую память:
   sQ 33280 + sP 8704 + двойной буфер K/V 69632 = 111 616 Б при пределе 98 304.

> ### ✅ ЗАКРЫТО СЧЁТЧИКОМ: пункты 2 и 3 В СИЛЕ, цена мерится ВАЙВФРОНТАМИ (01.08.2026)
>
> Здесь одни сутки стояла пометка «пункты 2 и 3 под вопросом»: отдельный стенд намерил, что
> многоадресность в MIO не удорожает, а удешевляет вдвое (`цена = max(128 Б, ширина·32 / min(DUP,2))`),
> и я решил, что объяснение «связывает разделяемая память» лишилось опоры. **Под вопросом был не
> документ, а стенд.** Обе половины той ставки оказались подгонкой, и обе сняты замером:
>
> **1. Многоадресность бесплатна ПОЛНОСТЬЮ, а не вдвое.** При замороженном шаге 128 Б и `DUP` 1→32:
> 2048 → 1024 → 512 → 256 → 128 → **64.8** такта, то есть ровно `32/DUP`. Насыщения на двух нет.
> «×2» была ВТОРЫМ слагаемым и к рассылке отношения не имеет: `LDS.128` стоит ≥2 вайвфронта при любой
> рассылке, потому что обратный путь в регистровый файл отдаёт не больше 8 Б на полосу за такт.
>
> **2. Переменная — не «уникальные байты», а КОНФЛИКТНОСТЬ БАНКОВ.** При замороженных 128 уникальных
> байтах (`DUP=1`, шаг 4→128 Б) цена идёт 67.1 → 128 → 256 → 512 → 1024 → **2048**: разброс ×32 при
> неизменном объёме данных. Тело с шагом **132 Б** (дополнение на слово) стоит **67.7** — как при шаге
> 4, — значит дело не в секторах и не в размахе адресов, а в банках.
>
>     вайвфронтов на команду = max( КОНФЛИКТНОСТЬ , ширина_на_полосу / 8 Б , 1 )
>     конвейер отдаёт РОВНО ОДИН ВАЙВФРОНТ ЗА ТАКТ НА SM
>
> **3. Независимая проверка: `ncu`, 0 расхождений из 17.** Счётчик
> `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld` против предсказания: 1→1.000, 2→2.000, 8→8.000,
> 32→32.000, дополнение→1.000, полная рассылка→1.000, `LDS.128` при полной рассылке→**2.000**
> (прямое доказательство второго слагаемого). Предсказание ПЕРИОДА: медиана ошибки 0.6 %, максимум
> 17.5 % на 90 точках.
>
> **4. Пункт 3 подтверждён, но механизм в нём назван неточно.** В боевых ядрах совпадающих адресов
> между полосами НЕТ ВОВСЕ — дублирование в карте фрагмента `mma.884` идёт по позициям операнда
> внутри квадропары, а не по адресам одной команды, поэтому случай `DUP=2` к нашему трафику Q не
> относится. Замер `ncu` на боевых ядрах: `volta_fwd_block` **2.054** вайвфронта на команду (ставка
> «все полосы различны» даёт ровно 2.054), `volta_fwd_ws` ф0 — **2.221** при расчётных 1.874.
>
> **5. Поэтому 768 ≈ 795 не только устояло, но было КОНСЕРВАТИВНЫМ.** Счётчик даёт 2.221 вместо
> принятых в расчёте 2.0 — реальный трафик на 11 % ВЫШЕ, — то есть вывод «связывает пропускная
> способность разделяемой памяти» замером **усиливается**, а не ослабляется.
>
> **Чему это учит (дороже самого факта).** Я пометил верный документ как спорный, потому что новая
> модель была законна (0 нарушений границы на 589 точках) и точна на своём стенде. Законность
> достигалась подгонкой `min(DUP,2)`, а цена подгонки — 15 пунктов зазора ровно на тех телах, где
> старая ставка была права. **Модель, ставшая законной за счёт ослабления границы, не опровергает
> замер — она перестаёт что-либо утверждать.** Признак был на виду: зазор ВЫРОС там, где новая модель
> «починила» нарушение (LDS с совпадающими адресами: −9.9 % → +5.3 %, но LDS с различными: 20.5 % →
> 35.8 %).
>
> Регистровый излом, найденный тем же стендом, независим от этого и в силе:
> `Q(W) = min(255, 8·⌊256/W⌋)` (совпало с ptxas 4/4), разлив начинается с `MaxLive + 7`, первые два
> разлитых значения бесплатны, дальше шаг ×5.1.
>
> Данные сохраняются вне репозитория: таблица по шагу и `DUP` со столбцом `ncu`, а также
> распределение зазора по типам тел.

**Правило, которое из этого следует.** Счёт по слотам — верхняя граница, а не предсказание. Прежде
чем сокращать выдачи, проверьте фальсификатором, СКОЛЬКО СТОИТ ОДНА: если цена ниже ~0.5 такта, ядро
не у счётной стены, и сокращать надо тот ресурс, который назовёт разложение по фазам. Дешёвый способ
получить обе цифры: холостая команда из постоянной памяти (цена слота) и снятие фазы целиком
(её доля). Обе правки — на шаблонном параметре, обе не трогают боевой путь.

### Проверка, что вы объясняете эффект верно

Если время растёт **меньше**, чем работа и чем сумма стоек (у нас: +5.0 % против +8.8 % и +6.4 %) —
значит вы **под** счётной стеной, и лишняя работа падает в простаивающие слоты. Тогда объяснение через
«не хватает такта» неверно **арифметически**, независимо от того, насколько оно правдоподобно.
Фальсификатор, которым это ловится: собрать вариант ядра, побитово тождественный исходному, но с точно
известным приростом команд, и сравнить фактическую цену с предсказанной счётным пределом. У нас: 1.67 %
фактических против 10.4 % предсказанных.

---

## 4. Stall taxonomy — read this before optimising anything

`ncu --metrics smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`. The reason
names which lever applies; guessing gets it wrong, twice in this project.

| Stall | Means | The lever |
|---|---|---|
| `long_scoreboard` | waiting on **global** memory | prefetch / more MLP / more warps. **Not** fewer bytes (see below) |
| `short_scoreboard` | waiting on **shared** memory or **shuffles** | restructure the reduction, bank conflicts |
| `wait` | fixed-latency **arithmetic** dependency | shorten the dependency chain; remove emulated work |
| `math_pipe_throttle` | the pipe is saturated | genuinely compute-bound; only less work helps |
| `barrier` | `__syncthreads` imbalance | rebalance the tiles |
| `not_selected` | another warp had work | you have enough parallelism here |

**Two [measured] cases from this repo, deliberately contrasted:**

- *Paged decode*, ctx=131072: `wait` **1.31** dominant, `short_scoreboard` **0.07**, ALU 58.9M vs
  memory 9.9M instructions. The obvious suspect was the cross-lane shuffle reduction — and
  `short_scoreboard = 0.07` **refuted it outright**. The cost was emulated fp8. Fixing it: ALU
  58.9M → 37.1M, DRAM 15.1% → 23.6% of peak, wall **1.5–1.8×**.
- *Backward `d>128`*, B1H8 S2048 d256: `long_scoreboard` **2.93** dominant, `wait` 0.81, DRAM 47%,
  8 warps. Different stall → different lever entirely.

**The most expensive lesson so far, and it is counter-intuitive.** In that backward kernel the
`dK/dV` gmem accumulator round-trip is ~**90% of all DRAM traffic** (1245 MB moved against 58.7 MB
of actual tensors — 21×). Halving it (fp16 storage) bought **+5.7%**, where a bandwidth-limited
kernel would have given ~40%. **Volume is not latency.** With 8 warps there is nothing to overlap
the wait with, so fewer bytes shorten nothing. This one measurement closed an entire family of
ideas — fp16 storage, smem accumulators at `kBJ=32`, `d`-split accumulation — because they all
attack volume.

---

## 5. cutlass on Volta — structural facts that constrain us

- `AttentionBackwardKernel<Sm70, half_t, kIsAligned, kApplyDropout, kPreload, kBlockSizeI,
  kBlockSizeJ, kMaxK, kKeysQueriesAlignedToBlockSize, kEnableSplitKeys, kComputeVOnly, kSkipGradV,
  kSkipGradQ, kComputeQOnly>` — the last five are this project's additions.
- **`kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI`** is the hinge of the whole `d>128` problem.
  False ⇒ `kNeedsAccumGradK/V` true ⇒ the `dK/dV` accumulators are **forced into global memory** and
  re-read once per query block; and `kFuseQKGradK` (which needs `kOutputInRF`) is off, so its smem is
  already free — *the prefetch spec's plan to pay for staging by disabling that fusion does not work
  at `d>128`.*
- *[measured]* smem at `d>128`, tile 128×64: **74 KB of 96 KB** → only **22 KB** free.
- **Instantiation walls on sm_70 — diagnosed, and one of them removed.** All of them are the same
  metaprogramming gap in `PitchLinearWarpRakedThreadMap`, which derives **zero iterations per thread**
  by integer division and then dies far away as `NumericArrayConverter<half_t,half_t,N=0>`
  ("zero-sized variable"). The two cases are arithmetically different:
  - **`kBlockSizeJ = 32` (`<128,32,32>`)** — the tile's contiguous extent is 32 halves = **4**
    128-bit accesses, but the hard-wired `WarpThreadArrangement<8,4>` puts **8** threads on that
    axis. `4/8 = 0`. *[fixed]* `default_mma_core_sm70.h` now sizes the warp's contiguous span to what
    the tile has (`min(8, kN/kElementsPerAccess)`) and gives the freed threads to the strided axis.
    Shapes that already built keep the exact `<8,4>` they had — the branch only fires below 8.
  - **`<256,64,256>`** — different cause: the B tile holds `(64/8)·32 = 256` 128-bit accesses for
    **512** threads, i.e. fewer accesses than threads, so `kWarpsContiguous = 2` while
    `WarpAccessIterations::kContiguous = 1`. Fixable only by dropping that operand to 64-bit
    accesses. *Not worth doing:* at `kBI = 256` the `MatmulQK` accumulator alone is
    `kBJ·kBI·2 = 32 KB`, and the tile does not fit 96 KB regardless. **The wall here is smem, not
    the template** — the earlier note in this file had that backwards.
- **The hardware has no such limits.** A 32-wide tile is whole `m8n8k4` operations, and a
  hand-written mainloop on `mma.sync` reaches **120.96 TFLOP/s = 96.8% of peak** at these shapes
  (`fa2_src/fmha_kernel/gemm/volta_hmma.h`, gate in `tools/volta_gemm_bench.cu`). Use it when the
  template derivation, not the machine, is what refuses.
- All three accumulator tiles — `dQ`, `dK`, `dV` — alias **one** type, `MatmulGradQ::AccumTileGmem`.
  The declarations inside `MatmulGradV`/`MatmulGradK` are dead code. Change the *use sites*.
- `Mma::prologue` issues `LDG` and `STS` **adjacently**, so the store waits on the load and the
  latency stays exposed. *[measured]* This is why simply enabling the stock `kPreload` is a **wash
  even at zero register recoil** — the win requires separating issue from drain by a contraction
  span. `kPreload`'s own prologue is also sm_80-gated and buggy on our BHSD/causal path.

---

## 6. Wave quantisation — count the SMs, not round numbers

*[measured]* With one CTA per SM, a grid of 64 CTAs leaves 16 SMs idle for the whole kernel. The
`d>128` backward split count had been tuned over `ns ∈ {1,2,4,6,8,12}` and settled on "one wave =
64 CTAs" — but the card has **80** SMs, and the point landing exactly on it was never in that grid;
the next one tried (96 CTAs) is *two* waves and duly collapsed, which made 64 look optimal.
Targeting 80 gave **+2.4…7.7%** at `B*H=8` and the first win over torch there (1.014).

The general rule: **the useful grid sizes are multiples of 80** (or of `80 × CTAs-per-SM`), and a
sweep over powers of two will systematically miss them.

*[measured] counter-example, so this is not over-applied:* wider is not automatically better. At
`d≤128` a 128-query tile is ~30% **worse** at every length — it amortises per-tile bookkeeping over
more queries but costs CTA count and registers, and below the crossover that trade is negative. The
"query tile is the lever" result holds at `d=256/512`, where the limiter is the gmem output
accumulator instead. **Do not carry a conclusion across the `kOutputInRF` boundary** — it is a
different machine on each side.

---

## 6b. `d=256` backward — the rest of the profile *[measured, 2026-07-26]*

Re-profiled to complete §4's stall figures (which reproduce: `long_scoreboard` **2.86** here vs 2.93
there). Default build, `B1 H8 S2048 d256`, `<Sm70,half_t,…,128,64,256,…>`, grid (10,8,1)×256:

| | | |
|---|---|---|
| DRAM | 424.95 GB/s = **47.1%** | not the wall |
| tensor pipe | **17.8%** | not the wall either |
| warps active | **12.5%** = 8 warps/SM | one block, and only one |
| registers/thread | **255** | 255 × 256 = 65 280 of 65 536 |
| L2 hit rate | 55.9% | (the hand-written GEMM gets 90% for comparison) |
| `barrier` stall | **1.25** | second-largest after `long_scoreboard` |

The addition to §4 is the last two rows. **Occupancy is pinned by registers, not by shared memory**
— one block per SM is forced by the register file, so every smem-shaped idea for raising fill is
attacking the wrong resource. With the tensor pipe at 17.8% and DRAM at 47%, the kernel is
*latency-bound with nothing to hide latency behind*, exactly as §4 concluded — and the only two
levers on that are more warps or fewer dependent global accesses. See §7.1 and §7.2.

---

## 7. Open questions worth an experiment

1. **Re-run the 16-warp collective on the new mainloop — the reason it was closed is gone.**
   The verdict on record is *"collective 128×128, 16 warps: built, correct, fits 115 registers (the
   occupancy door opens), but measured 1.50× slower than torch at the irreducible floor; a
   hand-written 512-thread GEMM is >2× worse than the tuned cutlass Volta mainloop — stop chasing
   occupancy."* That conclusion rests entirely on the hand-written mainloop being far worse than
   cutlass's. **It no longer is:** `volta_hmma.h` measures **96.8% of peak** (120.96 TFLOP/s), verified
   in SASS and coverage-asserted. §6b says occupancy is exactly what the `d>128` kernel lacks (12.5%,
   pinned by registers **and** smem at once) and that its stall is global latency (2.86), not
   bandwidth (47%) and not math (17.8%). Same geometry, different engine — a new experiment, not a
   repeat.
   *[measured] the cheap version of this is already ruled out:* two blocks per SM is unreachable
   (it needs ≤128 regs **and** ≤48 KB, and the tile needs 74 KB), and the alternative — **one** block
   of 16 warps via a `<32,16>` warp tile, which would also halve every per-thread fragment — does not
   compile: `mma_tensor_op_sm70.h:136` asserts `Shape % GemmShape<32,32,4>`, since cutlass's Volta
   operand layouts and their `MmaVoltaTensorOpMultiplicandTileIterator` are built on 32×32
   interleaving. Unlike the narrow-N thread map there is nothing to relax — the layout itself would
   have to change. The hardware is fine with 32×16 (two `16×16×4` quad-pair ops). So the 16-warp route
   **requires** the hand-written warp op; it is not reachable by configuration.
   The parameter is in place and shipped at its old value: `FMHA_BWD_WARP_N` (default 32).

   **The warp op now exists and the trade is measured** — `fa2_src/fmha_kernel/gemm/volta_warp_mma.h`,
   gate `tools/volta_warp_test.cu`. Any tile `16·MB × 16·NB`, all four `row/col × row/col` instruction
   variants, one accumulator layout for all of them (*[measured]* — the C map does not depend on the
   operand layouts), coverage-asserted. On a clock-locked V100, 240 blocks:

   | warp tile | warps | TFLOP/s | % peak | registers |
   |---|---|---|---|---|
   | 32×32 | 8 (what cutlass gives) | 102.3 | 81.6% | 60 |
   | 64×64 | 8 | 119.2 | 95.2% | 190 |
   | **32×16** | **16** | **115.1** | **91.9%** | **68** |
   | 64×32 | 16 | 112.2 | 89.5% | 114 |

   The narrow tile at 16 warps **beats** the cutlass-shaped one by 12.5% and comes within 3.5% of the
   widest — doubling the warps fully pays for the ILP it gives up — at **68 registers instead of 190**.
   It also halves the residency cost: `(warp_M·warp_N/32)·(d/kBI)` = 16·d/kBI at 32×16, so resident
   `dK+dV` at d=256 costs 64 floats/thread, exactly what the working d=128 path costs (§7.3).
   *[measured] trap in the benchmark itself:* unpadded shared memory (`ld = K = 32`) puts rows two
   apart in the same bank — an 8-way conflict — and the first run read 32×32/8w as 41% of peak instead
   of 81.6%. It was comparing conflicts, not tiles.

   What is left is the integration: the operator replaces `MmaVoltaTensorOp` *and* the operand
   iterators/smem layouts that come with it, which is a new threadblock mainloop rather than a
   template swap.
2. **Head-slicing to restore `kOutputInRF` at `d>128`.** A CTA that owns only a *w*-wide slice of the
   head for OUTPUT pays `32·w/kBI` resident floats, so `w=128, kBI=128` costs exactly what the working
   d=128 path costs (32+32). `S` and `dS` need the whole head, so QK and DOIVJ are recomputed per
   slice: FLOP goes 5 units → 7 (**×1.4**) in exchange for deleting the `dK/dV` accumulator
   round-trips. Wants `p.head_dim` (reduction depth) separated from the output column range, which
   today are the same parameter.
   **Read §4 before believing this one.** §4 closed "`d`-split accumulation" together with fp16
   storage, on the grounds that they attack *volume* and volume is not latency (halving the bytes
   bought +5.7%). The distinction that keeps this alive is narrow but real: fp16 storage halves the
   **bytes per round-trip**, head-slicing removes the **round-trips themselves** — i.e. the number of
   *dependent global accesses*, which is what `long_scoreboard` counts. §4's measurement does not
   separate those two, so this is a hypothesis it neither proves nor refutes. It also costs ×1.4 FLOP
   against a tensor pipe that is only 17.8% busy, which is the cheap side of the trade.
3. ~~**Register-resident `dK/dV` at d=256.**~~ **CLOSED — measured, 2026-07-26.** Forcing the gate
   (`FMHA_BWD_RF256`) with the *correct* fragment count gives **128 regs + 696 B/thread spill**
   (178 KB of local memory per block) and **0.46×** against torch, where the shipped path is 0.98×.
   The earlier note claiming "224/254/255 regs, zero spill, only correctness blocks it" was taken on
   the one-fragment build — half the registers *and* half the head. Closed form: residency costs
   `(warp_M·warp_N/32)·(d/kBI)` floats/thread/tensor, and both levers are shut — `kBI` by smem
   (`kBI=256` ⇒ 32 KB for the `MatmulQK` accumulator alone; 128×128 already fails to launch) and the
   warp tile by `mma_tensor_op_sm70.h:136` (`Shape % GemmShape<32,32,4>`, because Volta's operand
   layouts are organised in 32×32 interleaved tiles — measured, see §7.1). **Within cutlass this is
   unreachable**; the route is the hand-written warp op.
   *Superseded detail, kept for the derivation:* `kOutputInRF` is a *structural* gate (`kMaxK ≤ kBI`),
   not a register-count computation. *[derived]* the resident cost per thread is exactly
   **`32·d/kBI` floats per tensor**, because `kNumThreads = kBI·kBJ/32` and each column tile is
   `kBJ·kN/kNumThreads = 32` floats with `kN == kBI`. So it depends on **`kBI` alone** — not on
   `kBJ`, not on the column-tile width, not on the warp count. d=128 → 32+32 = 64 (the shipped fast
   path, and it fits); d=256 → 128; d=512 → 256 (hopeless). This also refutes the premise written
   next to the `kBJ=32` probe: halving `kBJ` halves the tile **and** the warp count together, so
   floats-per-thread do not move.
4. **The 10 KB.** The prefetch schedule needs 32 KB of staging and `d>128` has 22 KB. Find 10 KB in
   the smem union, or accept the spec's marginal 8-slot variant.
5. **`gptq_w8_dequant`** *[measured]* runs at 545 GB/s against ~820 achievable, on every linear
   layer of the int8 model — the largest untouched target on the serving path, and §3 says why.
