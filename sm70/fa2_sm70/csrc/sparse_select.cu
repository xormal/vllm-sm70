// [FA2/SM70] ЭТАП 1 РАЗРЕЖЕННОГО ПРЕФИЛЛА: средние ключи -> счёт -> отбор -> CSR-индекс.
// Записка 20. Правило -- FlashPrefill V2 §3.1 (arXiv 2608.19758), но БЕЗ поправки на среднее:
// на нашей сети она замерена и в рабочем диапазоне 20-45 % оставленных блоков ВРЕДИТ
// (0.048 против 0.040 при alpha=0.02), помогая лишь при экстремальной разреженности.
// Значит из Этапа 2 уходит впрыск корректирующих кусков, а здесь не нужны средние ЗНАЧЕНИЯ.
//
// ПОЧЕМУ ЗОНДЫ, А НЕ ВСЕ ЗАПРОСЫ ПЛИТКИ. Q плитки (128 запросов x 6 голов x 256) -- это 393 КиБ,
// в разделяемую не влезает. Замерено оффлайн на настоящем срезе боевой сети при alpha=0.01:
//     все 128 -> 45.7 % блоков, ошибка 0.0169 | 32 зонда -> 45.2 %, 0.0173
//     16 зондов -> 47.3 %, 0.0166           |  8 зондов -> 51.4 %, 0.0156
// То есть 16 зондов дают ТУ ЖЕ выборку при 48 КиБ разделяемой. Меньше 16 -- отбор мягчеет
// (блоков остаётся больше), что безопасно по точности, но дороже по времени.
//
// АДРЕСАЦИЯ ПУЛА взята из attn_fwd_volta_i8_paged БУКВА В БУКВУ (иначе счёт пойдёт по другим
// байтам, чем внимание): для токена p страница pg = BlockTable[p / bs], intra = p % bs,
//     данные   K[pg * stride0 + intra * Hkv * D + hk * D + d]
//     масштаб  Ks[(pg * bs + intra) * Hkv + hk]
#include <cstdio>
#include <torch/extension.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Блочные свёртки для правила массы (все нити участвуют, результат у всех одинаков).
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < nw; ++i) r += red[i];
    return r;
}
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = -1e30f;
    for (int i = 0; i < nw; ++i) r = fmaxf(r, red[i]);
    return r;
}

// ---- K1: средние ключи блока. Сетка (NB, Hkv), 256 нитей = по нити на измерение d.
// Нити 0..255 читают ПОДРЯД ЛЕЖАЩИЕ 256 байт одной позиции -- слияние полное.
__global__ void blk_mean_i8_paged(
        const int8_t* __restrict__ K, const float* __restrict__ Ks,
        const int* __restrict__ bt, int T, int bs, int Hkv, int D, int B,
        long pstride, __half* __restrict__ out, int NB,
        float* __restrict__ rOut, __half* __restrict__ devOut) {
    const int J = blockIdx.x, hk = blockIdx.y, d = threadIdx.x;
    // РАННЕГО ВЫХОДА БОЛЬШЕ НЕТ: ниже стоит блочная свёртка, а `return` части нитей
    // подвесил бы `__syncthreads()` внутри неё. Лишние нити просто несут нулевой вклад.
    const int p0 = J * B, p1 = min(p0 + B, T);
    float acc = 0.f, acc2 = 0.f, vmin = 1e30f, vmax = -1e30f;
    if (d < D) {
        for (int p = p0; p < p1; ++p) {
            const int pg = bt[p / bs], intra = p - (p / bs) * bs;
            const float s = Ks[((long)pg * bs + intra) * Hkv + hk];
            const float v = s * (float)K[(long)pg * pstride + (long)intra * Hkv * D + (long)hk * D + d];
            acc += v; acc2 += v * v;
            vmin = fminf(vmin, v); vmax = fmaxf(vmax, v);   // размах -- тот же проход
        }
    }
    const int n = max(p1 - p0, 1);
    float var = 0.f;
    if (d < D) {
        const float mu = acc / (float)n;
        out[((long)hk * NB + J) * D + d] = __float2half(mu);
        // [РАЗБРОС БЛОКА -- ТОТ ЖЕ ПРОХОД, БЕЗ ВТОРОГО ЧТЕНИЯ]
        // Var_d = E[k_d^2] - (E k_d)^2. Сумма по d даёт КВАДРАТ СКО ключей блока вокруг
        // среднего: r_J = sqrt(sum_d Var_d). Второй проход по позициям не нужен, а он стоил
        // бы ровно столько же, сколько первый.
        var = fmaxf(acc2 / (float)n - mu * mu, 0.f);
        // [РАЗМАХ ПО ИЗМЕРЕНИЮ] Точная граница Гёльдера: max_i q*k_i <= q*k̄ + sum_d |q_d|*dev_d,
        // где dev_d -- наибольшее отклонение ключа блока от среднего ПО ЭТОМУ измерению. Она
        // СИЛЬНЕЕ изотропной ||q||*r_J: та не знает, какие измерения запросу важны, и потому
        // одинаково поднимает блок, разбросанный "не туда".
        devOut[((long)hk * NB + J) * D + d] =
            __float2half((p1 > p0) ? fmaxf(vmax - mu, mu - vmin) : 0.f);
    }
    // свёртка по 256 нитям (= по измерениям d)
    float t = warp_sum(var);
    __shared__ float rw[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) rw[warp] = t;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s2 = 0.f;
        for (int w = 0; w < (blockDim.x >> 5); ++w) s2 += rw[w];
        rOut[(long)hk * NB + J] = sqrtf(s2);
    }
}

// ---- K1b: ОБНОВЛЕНИЕ средних ключей ПРИ ЗАПИСИ KV (задел под разреженный декод).
// Адресация -- по ФИЗИЧЕСКОМУ подблоку пула, а НЕ по логическому номеру: логический протухает,
// как только движок переиспользует страницу под другой запрос (тихая порча, не отказ).
// Таблица [страниц * (bs/B), Hkv, D] -- свойство ПУЛА, и она верна, пока верно его содержимое.
//
// Зачем: в префилле средние стоят 0.41 мс на вызов -- мелочь. В ДЕКОДЕ те же 0.41 мс на слой
// дают 6.6 мс на шаг, то есть БОЛЬШЕ ПОЛОВИНЫ приза от разреженности (приз ~11 мс).
// Здесь пересчитываются только ТРОНУТЫЕ подблоки: за кусок префилла их 32 вместо ~2000,
// за шаг декода -- один.
//
// Группа нитей берёт срез слотов [i*B, min((i+1)*B, n)) и усредняет РОВНО их. Слоты внутри
// куска идут подряд по построению движка, поэтому срез не пересекает подблок; отрицательные
// слоты (набивка) пропускаются.
__global__ void blk_mean_update(
        const int8_t* __restrict__ K, const float* __restrict__ Ks,
        const long* __restrict__ slots, int nslots, int bs, int Hkv, int D, int B,
        long pstride, __half* __restrict__ out, int subPerPage,
        __half* __restrict__ devOut) {
    const int grp = blockIdx.x, hk = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const int s0 = grp * B, s1 = min(s0 + B, nslots);
    if (s0 >= s1) return;
    long first = -1;
    for (int i = s0; i < s1; ++i) { if (slots[i] >= 0) { first = slots[i]; break; } }
    if (first < 0) return;
    const long pg = first / bs, intra = first - pg * bs, sub = intra / B;
    // [СЧИТАЕМ ВЕСЬ ПОДБЛОК, А НЕ ТОЛЬКО СЛОТЫ ГРУППЫ]
    // Первая редакция усредняла лишь те слоты группы, что попали в этот подблок. Это верно
    // РОВНО когда чанк начинается на границе подблока -- а в приборе слоты шли `arange(T)`,
    // то есть выровненные, и ошибка не проявилась. В движке чанк может начинаться где угодно
    // (префикс-кэш!), и тогда подблок обновлялся ЧАСТИЧНО: половина средних оставалась старой.
    // Поймано формой деградации: при потере 2 % блоков текст расходился с 31-го символа --
    // размен качества так не выглядит, так выглядит ДЕФЕКТ.
    // Теперь берём ВСЕ B позиций подблока прямо из пула: покрытие полное независимо от того,
    // как легли слоты, а повторный счёт того же подблока идемпотентен.
    const long base = pg * bs + sub * B;
    float acc = 0.f, vmin = 1e30f, vmax = -1e30f;
    for (int t = 0; t < B; ++t) {
        const long in2 = sub * B + t;
        const float v = Ks[(base + t) * Hkv + hk]
                      * (float)K[pg * pstride + in2 * Hkv * D + (long)hk * D + d];
        acc += v; vmin = fminf(vmin, v); vmax = fmaxf(vmax, v);
    }
    const float mu = acc / (float)B;
    out[((pg * subPerPage + sub) * Hkv + hk) * D + d] = __float2half(mu);
    // Размах по измерению -- ТЕМ ЖЕ проходом. Он нужен декодному отбору ровно за тем же,
    // за чем префилльному: средний ключ не представляет блок с разбросанными ключами.
    if (devOut != nullptr)
        devOut[((pg * subPerPage + sub) * Hkv + hk) * D + d] =
            __float2half(fmaxf(vmax - mu, mu - vmin));
}

// ---- K1c: собрать средние логических блоков из ФИЗИЧЕСКОЙ таблицы по таблице блоков.
// Собирает СРЕДНИЕ и (если дана) таблицу РАЗМАХОВ одним проходом: обе имеют одну форму и
// один адрес подблока, второй проход по таблице блоков был бы чистой тратой.
__global__ void gather_means(const __half* __restrict__ pool, const int* __restrict__ bt,
                             int bs, int Hkv, int D, int B, int subPerPage,
                             __half* __restrict__ out, int NB,
                             const __half* __restrict__ devPool,
                             __half* __restrict__ devOut,
                             float* __restrict__ rOut) {
    const int J = blockIdx.x, hk = blockIdx.y, d = threadIdx.x;
    const long p0 = (long)J * B, pg = bt[p0 / bs], sub = (p0 - (p0 / bs) * bs) / B;
    const long src = ((pg * subPerPage + sub) * Hkv + hk) * D;
    float mu = 0.f, var = 0.f;
    if (d < D) {
        const __half m = pool[src + d];
        out[((long)hk * NB + J) * D + d] = m;
        if (devPool != nullptr && devOut != nullptr) devOut[((long)hk * NB + J) * D + d] = devPool[src + d];
        mu = __half2float(m); var = mu * mu;     // СКО среднего по измерениям -- для beta
    }
    // r_J для изотропной поправки собирается тут же свёрткой (beta по умолчанию 0, но
    // величина обязана быть определена, а не мусором).
    float t = warp_sum(var);
    __shared__ float rw[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) rw[warp] = t;
    __syncthreads();
    if (threadIdx.x == 0 && rOut != nullptr) {
        float s2 = 0.f;
        for (int w = 0; w < (blockDim.x >> 5); ++w) s2 += rw[w];
        rOut[(long)hk * NB + J] = sqrtf(s2);
    }
}


// ---- K2: счёт по средним ключам. РАСЩЕПЛЁН ПО БЛОКАМ КЛЮЧЕЙ (переукладка 27.08).
// Было: сетка (плитки, Hkv) = 64 блока на 80 SM, и цикл по NB блокам ключей шёл ВНУТРИ блока
// последовательно. Замер: 8.62 мс при загрузке SM 21.5 % -- связывал не счёт команд, а
// ОТСУТСТВИЕ ПАРАЛЛЕЛИЗМА. (Переукладка свода строк в регистры варпа перед этим дала РОВНО НОЛЬ
// -- 7.67 против 7.5 мс; это и указало, что искать надо не там.)
// Стало: третья ось сетки режет диапазон J на JS кусков -> блоков в JS раз больше.
// Каждый кусок пишет свои (m, S) в ГЛОБАЛЬНУЮ таблицу; порог и компактация -- отдельным
// маленьким ядром, потому что им нужен максимум по ВСЕМ J.
// Предел строк на варп для регистрового пути и предел D. При R=96 и 8 варпах строк 12.
#ifndef QREG_MAX
#define QREG_MAX 16
#endif
#ifndef QREG_D
#define QREG_D 256
#endif
__global__ void score_part(
        const __half* __restrict__ Q,      // [Sq, H, D]
        const __half* __restrict__ kbar,   // [Hkv, NB, D]
        const float* __restrict__ rbar,    // [Hkv, NB] -- СКО ключей блока вокруг среднего
        const float* __restrict__ rmu,     // [Hkv] -- среднее r_J по блокам (для безразмерности)
        const __half* __restrict__ dev,    // [Hkv, NB, D] -- размах ключей блока по измерению
        int Sq, int T, int H, int Hkv, int D, int B, int TI, int PROBE,
        float scale, float beta, float gamma, int NB, int JS, int Rc,
        float* __restrict__ mOut,          // [ntiles, Hkv, NB]
        float* __restrict__ sOut,
        int REGQ) {                        // 1 = строки Q в регистрах (см. разбор ниже)
    extern __shared__ char smem[];
    const int tile = blockIdx.x, hk = blockIdx.y, sl = blockIdx.z;
    const int g = H / Hkv, R = PROBE * g;
    __half* Qs = (__half*)smem;
    // [ЦЕЛЬНОЕ ЯДРО БЕЗ ПЛАТЫ ЗАНЯТОСТЬЮ -- 31.08]
    // Цельность (один проход по строкам, Rc = R) стоила 49.15 КиБ разделяемой под Q, и на SM
    // влезал РОВНО ОДИН блок: 8 варпов из 64, 12.5 %. Куски по 48 строк возвращали занятость,
    // но ценой ПОВТОРНОГО чтения средних блока на каждый кусок -- то есть ядро переставало
    // быть цельным.
    // Выход -- окно, а не дверь: разделяемая под Q вообще НЕ НУЖНА. Внутренний цикл идёт
    // `for (r = warp; r < Rn; r += NW)`, то есть варп читает ТОЛЬКО СВОИ строки; чужие ему
    // недоступны и не нужны. Значит строка живёт в РЕГИСТРАХ варпа: поток lane держит пары
    // c = lane, lane+32, ... своей строки. Разделяемая падает до статических ~2.8 КиБ,
    // занятость определяется регистрами, а цельность сохраняется по построению.
    // Ограничение -- строк на варп не больше QREG_MAX; иначе молча идём прежним путём.
    const int NWc = (int)(blockDim.x >> 5);
    const int rowsPerWarp = (R + NWc - 1) / NWc;
    const bool регистрами = (REGQ != 0) && (rowsPerWarp <= QREG_MAX) && (D <= QREG_D);
    __half2 qreg[QREG_MAX][QREG_D / 64];      // [строка варпа][пара на поток]
    float   qnr[QREG_MAX];
    int     posrr[QREG_MAX];
    // [ТЕМПОРАЛЬНАЯ УКЛАДКА] Буферы ДВОЙНЫЕ. Было три барьера на блок ключей: загрузка,
    // свод варпов, сведение -- и загрузка следующего блока не начиналась, пока не досчитан
    // текущий, то есть латентность глобальной памяти стояла голой. Двойной буфер позволяет
    // ВЫДАТЬ загрузку J+1 до счёта J и обойтись ОДНИМ барьером на итерацию: конфликтующая
    // запись в тот же буфер отстоит на ДВЕ итерации, между ними барьер.
    __shared__ __half kb[2][256];
    __shared__ __half db[2][256];      // размах блока по измерениям
    __shared__ float  wm[2][32], ws[2][32];
    __shared__ int    posr[256];
    __shared__ float  qn[256];         // ||q|| на зондовую строку -- для поправки на разброс

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int i0 = tile * TI, i1 = min(i0 + TI, Sq), TIe = i1 - i0;
    const int NBvis = min(NB, ((T - Sq) + i1 - 1) / B + 1);
    const int per = (NBvis + JS - 1) / JS;
    const int j0 = sl * per, j1 = min(j0 + per, NBvis);
    float* mJ = mOut + ((size_t)tile * Hkv + hk) * NB;
    float* SJ = sOut + ((size_t)tile * Hkv + hk) * NB;
    // [РАЗБИЕНИЕ ЗОНДОВЫХ СТРОК НА КУСКИ]
    // Связывала ЗАНЯТОСТЬ: стоянка всех R строк Q брала 49.15 КиБ разделяемой, и на SM
    // влезал РОВНО ОДИН блок -- 8 варпов из 64, 12.49 % (замер ncu 28.08). Уменьшать R
    // нельзя: PROBE -- ручка КАЧЕСТВА, гейт мульти-иглы отверг и 12, и 8. Поэтому строки
    // идут КУСКАМИ по Rc: разделяемая падает кратно, а счёт по кускам сводится онлайновым
    // софтмаксом в mJ/SJ. Цена -- повторное чтение средних блока на каждый кусок.
    for (int rc0 = 0; rc0 < R; rc0 += Rc) {
    const int Rn = min(Rc, R - rc0);
    for (int rl = warp; rl < Rn; rl += NW) {
        const int r = rc0 + rl;
        const int pr = r / g, hh = r % g;
        const int qi = (PROBE > 1) ? (pr * (TIe - 1)) / (PROBE - 1) : 0;
        const int row = i0 + qi, h = hk * g + hh;
        float qq = 0.f;
        if (регистрами) {
            // В РЕГИСТРЫ: поток держит пары своей строки, ни одного байта разделяемой.
            const __half2* Qp = reinterpret_cast<const __half2*>(
                Q + ((size_t)row * H + h) * D);
            const int slot = rl / NW;               // номер строки ВНУТРИ варпа
            #pragma unroll
            for (int t = 0; t < QREG_D / 64; ++t) {
                const int c = lane + 32 * t;
                if (c < D / 2) {
                    const __half2 v2 = Qp[c];
                    qreg[slot][t] = v2;
                    const float2 f2 = __half22float2(v2);
                    qq += f2.x * f2.x + f2.y * f2.y;
                } else {
                    qreg[slot][t] = __float2half2_rn(0.f);
                }
            }
            qq = warp_sum(qq);
            posrr[slot] = (T - Sq) + row;
            qnr[slot] = sqrtf(qq);
        } else {
            for (int c = lane; c < D; c += 32) {
                const __half v = Q[((size_t)row * H + h) * D + c];
                Qs[(size_t)rl * D + c] = v;
                const float f = __half2float(v); qq += f * f;
            }
            qq = warp_sum(qq);
            if (lane == 0) { posr[rl] = (T - Sq) + row; qn[rl] = sqrtf(qq); }
        }
    }
    __syncthreads();
    int cur = 0;
    if (j0 < j1 && tid < D) {                      // предзагрузка первого блока
        kb[0][tid] = kbar[((size_t)hk * NB + j0) * D + tid];
        db[0][tid] = (gamma != 0.f) ? dev[((size_t)hk * NB + j0) * D + tid]
                                    : __float2half(0.f);
    }
    __syncthreads();
    for (int J = j0; J < j1; ++J) {
        const int nx = cur ^ 1;
        if (J + 1 < j1 && tid < D) {               // загрузка СЛЕДУЮЩЕГО -- до счёта текущего
            kb[nx][tid] = kbar[((size_t)hk * NB + (J + 1)) * D + tid];
            db[nx][tid] = (gamma != 0.f) ? dev[((size_t)hk * NB + (J + 1)) * D + tid]
                                         : __float2half(0.f);
        }
        // [ПОПРАВКА НА РАЗБРОС] Средний ключ -- ПЛОХОЙ представитель разнородного блока:
        // у блока с одним сильно совпадающим ключом среднее указывает "в никуда", и блок
        // получает низкий счёт. Замерено 28.08: игла терялась даже при 89 % сохранённых
        // блоков, а при alpha->0 разреженный путь совпадал с плотным ПОБАЙТОВО -- значит
        // виновато правило, а не путь. Поправка beta*||q||*r_J поднимает счёт ровно тем
        // блокам, внутри которых ключи разбросаны. beta=0 -- прежнее правило бит в бит,
        // beta=1 -- гарантирующая граница (по замеру слишком груба, не режет ничего).
        // ПРАВИТ ОТКЛОНЕНИЕ, А НЕ САМА ВЕЛИЧИНА. Замерено 28.08: beta*||q||*r_J при beta=1
        // равно 28.4, тогда как ВЕСЬ разброс счётов q*kbar по блокам -- 0.162, то есть
        // поправка перебивала сигнал в 175 раз и правило вырождалось в "ранжируй по
        // разбросу". Ранжирование зависит только от РАЗНОСТИ между блоками, поэтому
        // вычитаем среднее: общий сдвиг всех счётов ранга не меняет, а масштаб становится
        // сопоставимым с сигналом.
        const float rj = (beta != 0.f)
                       ? beta * (rbar[(size_t)hk * NB + J] - rmu[hk]) : 0.f;
        float mw = -1e30f, sw = 0.f;                 // свод варпа -- в регистрах
        for (int r = warp; r < Rn; r += NW) {
            // [ПАРАМИ, А НЕ ПОШТУЧНО]  Поштучно на КАЖДОЕ умножение приходилось ДВА
            // преобразования half->float отдельными командами плюс две 16-битные загрузки
            // разделяемой -- около семи команд на одно полезное FFMA. Замер это и показывал:
            // 3.4 % пика fp32 при загрузке SM 22 %. Пара берётся одной 32-битной загрузкой,
            // разворачивается ОДНОЙ `__half22float2` и даёт два FFMA: ~2 команды на умножение.
            float a = 0.f, ad = 0.f;
            const __half2* kp = reinterpret_cast<const __half2*>(kb[cur]);
            const __half2* dp = reinterpret_cast<const __half2*>(db[cur]);
            float qn_r; int posr_r;
            if (регистрами) {
                const int slot = (r - warp) / NW;
                #pragma unroll
                for (int t = 0; t < QREG_D / 64; ++t) {
                    const int c = lane + 32 * t;
                    if (c < D / 2) {
                        const float2 qv = __half22float2(qreg[slot][t]);
                        const float2 kv = __half22float2(kp[c]);
                        a += qv.x * kv.x + qv.y * kv.y;
                        if (gamma != 0.f) {
                            const float2 dv = __half22float2(dp[c]);
                            ad += fabsf(qv.x) * dv.x + fabsf(qv.y) * dv.y;
                        }
                    }
                }
                qn_r = qnr[slot]; posr_r = posrr[slot];
            } else {
                const __half2* Qp = reinterpret_cast<const __half2*>(Qs + (size_t)r * D);
                for (int c = lane; c < D / 2; c += 32) {
                    const float2 qv = __half22float2(Qp[c]);
                    const float2 kv = __half22float2(kp[c]);
                    a += qv.x * kv.x + qv.y * kv.y;
                    if (gamma != 0.f) {
                        const float2 dv = __half22float2(dp[c]);
                        ad += fabsf(qv.x) * dv.x + fabsf(qv.y) * dv.y;
                    }
                }
                qn_r = qn[r]; posr_r = posr[r];
            }
            a = (warp_sum(a) + rj * qn_r
                 + ((gamma != 0.f) ? gamma * warp_sum(ad) : 0.f)) * scale;
            const float v = (J * B <= posr_r) ? a : -1e30f;
            if (v > -1e29f) { const float mn = fmaxf(mw, v);
                              sw = sw * __expf(mw - mn) + __expf(v - mn); mw = mn; }
        }
        if (lane == 0) { wm[cur][warp] = mw; ws[cur][warp] = sw; }
        // ЕДИНСТВЕННЫЙ барьер на итерацию: он же публикует своды варпов, он же -- подкачку.
        __syncthreads();
        if (tid == 0) {
            float m = -1e30f, sacc = 0.f;
            for (int w = 0; w < NW; ++w) {
                if (ws[cur][w] <= 0.f) continue;
                const float mn = fmaxf(m, wm[cur][w]);
                sacc = sacc * __expf(m - mn) + ws[cur][w] * __expf(wm[cur][w] - mn); m = mn;
            }
            if (rc0 == 0) { mJ[J] = (sacc > 0.f) ? m : -1e30f; SJ[J] = sacc; }
            else if (sacc > 0.f) {               // сведение с предыдущими кусками строк
                const float m0 = mJ[J], s0 = SJ[J];
                if (s0 <= 0.f) { mJ[J] = m; SJ[J] = sacc; }
                else { const float mn = fmaxf(m0, m);
                       SJ[J] = s0 * __expf(m0 - mn) + sacc * __expf(m - mn); mJ[J] = mn; }
            }
        }
        // Хвостового барьера НЕТ: чтение нити 0 из wm[cur] конфликтует с записью в тот же
        // буфер лишь ЧЕРЕЗ ИТЕРАЦИЮ, а между ними стоит барьер следующей итерации.
        cur = nx;
    }
    __syncthreads();     // куски строк идут последовательно: Qs следующего не должен
    }                    // затирать Qs текущего до конца счёта
}

// ---- K3: порог и компактация. Сетка (плитки, Hkv) -- работы здесь O(NB), а не O(NB*R).
__global__ void select_csr(
        int Sq, int T, int Hkv, int B, int TI, float alpha, int SINK, int WIN, int REC,
        const float* __restrict__ mOut, const float* __restrict__ sOut,
        int* __restrict__ cnt, int* __restrict__ idx, int NB, int STRAT, int PAIR,
        float* __restrict__ stat, float MASS) {   // [МАССА, 12.09] на (плитку, kv-голову): {масса удержанных, масса всех}
    const int tile = blockIdx.x, hk = blockIdx.y;
    const int i1 = min(tile * TI + TI, Sq);
    const int NBvis = min(NB, ((T - Sq) + i1 - 1) / B + 1);
    const float* mJ = mOut + ((size_t)tile * Hkv + hk) * NB;
    const float* SJ = sOut + ((size_t)tile * Hkv + hk) * NB;
    __shared__ float E[4096];
    for (int J = threadIdx.x; J < NBvis && J < 4096; J += blockDim.x) E[J] = 0.f;
    __syncthreads();
    __shared__ float Mall, emaxAll;
    if (threadIdx.x == 0) {
        float M = -1e30f;
        for (int J = 0; J < NBvis; ++J) M = fmaxf(M, mJ[J]);
        Mall = M;
    }
    __syncthreads();
    for (int J = threadIdx.x; J < NBvis; J += blockDim.x)
        E[J] = (mJ[J] > -1e29f) ? SJ[J] * __expf(mJ[J] - Mall) : 0.f;
    __syncthreads();
    // [ПОРОГ ПО МАССЕ, 12.09] Замер на Википедии 251K (alpha=0.02, B=64, STRAT=8): порог alpha·max
    // удерживает в медиане лишь 79 % массы софтмакса зондов (10-й перцентиль 62 %, минимум 39 %) --
    // на размытых головах выбрасывается 20-40 % внимания на каждом слое, и почти одинаковые иглы
    // становятся неразличимы (8-11/12 при плотном 12/12). Правило: порог t выбирается так, чтобы
    // Σ_{E>=t} E >= MASS·Σ E (бинарный поиск по t, 16 шагов, блочные свёртки); итоговый порог
    // min(alpha·max, t). MASS=0 -- прежнее правило бит в бит.
    __shared__ float red[32];
    float t_mass = 0.f;
    {
        float part = 0.f, pmax = 0.f;
        for (int J = threadIdx.x; J < NBvis; J += blockDim.x) { part += E[J]; pmax = fmaxf(pmax, E[J]); }
        const float massA = block_sum(part, red);
        const float emaxB = block_max(pmax, red);
        if (MASS > 0.f && massA > 0.f) {
            float lo = 0.f, hi = emaxB;
            for (int it = 0; it < 16; ++it) {
                const float mid = 0.5f * (lo + hi);
                float p = 0.f;
                for (int J = threadIdx.x; J < NBvis; J += blockDim.x) if (E[J] >= mid) p += E[J];
                const float m = block_sum(p, red);
                if (m >= MASS * massA) lo = mid; else hi = mid;
            }
            t_mass = lo;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float emax = 0.f;
        for (int J = 0; J < NBvis; ++J) emax = fmaxf(emax, E[J]);
        emaxAll = emax;
        float thr = alpha * emax;
        if (MASS > 0.f) thr = fminf(thr, t_mass);
        // [ЛУЧШИЙ В ОКРУГЕ] Порог глобален: блок, чей счёт ниже основной массы, не пройдёт
        // НИ ПРИ КАКОМ alpha (замер 28.08: 0.01 ХУЖЕ 0.02, блок 64 не помог). Но игла почти
        // всегда ЛУЧШАЯ В СВОЕЙ ОКРУГЕ, даже когда проигрывает далёким блокам. Поэтому
        // дополнительно берём argmax по каждому окну из STRAT блоков: покрытие становится
        // равномерным по длине, цена -- NBvis/STRAT блоков сверху.
        // STRAT = 0 отключает (прежнее правило бит в бит).
        const int jlast = (T - Sq + i1 - 1) / B;
        int n = 0;
        int* dst = idx + ((size_t)tile * Hkv + hk) * NB;
        // [ПАРА ЗА ЛУЧШИМ, 12.09] Мульти-игла на Википедии при B=64: потерянные иглы отвечают кодом
        // СОСЕДНЕЙ иглы -- предложение «warehouse NAME is CODE» (~15 токенов) разрезано границей
        // блока (~23 % при B=64): имя в одном блоке, код в следующем, а отобран один. Блок, взятый
        // как лучший в окне (кандидат-игла), тянет за собой СЛЕДУЮЩИЙ. Цена <= NBvis/STRAT блоков.
        bool pair_next = false;
        float massK = 0.f, massA = 0.f;
        for (int J = 0; J < NBvis; ++J) {
            bool keep = (E[J] >= thr);
            if (pair_next) { keep = true; pair_next = false; }
            if (!keep && STRAT > 0) {          // лучший в своём окне -- взять всегда
                const int w0 = (J / STRAT) * STRAT, w1 = min(w0 + STRAT, NBvis);
                float bm = -1.f; int bj = -1;
                for (int u = max(w0, 0); u < w1; ++u) if (E[u] > bm) { bm = E[u]; bj = u; }
                if (bj == J && bm > 0.f) { keep = true; pair_next = (PAIR != 0); }
            }
            if (J < SINK) keep = true;
            if (J > jlast - WIN && J <= jlast) keep = true;
            if (J > jlast - REC && J <= jlast) keep = true;
            if (keep) dst[n++] = J;
            massA += E[J]; if (keep) massK += E[J];
        }
        cnt[(size_t)tile * Hkv + hk] = n;
        if (stat) { stat[((size_t)tile * Hkv + hk) * 2] = massK; stat[((size_t)tile * Hkv + hk) * 2 + 1] = massA; }
    }
}


// ================= ОТБОР ДЛЯ ДЕКОДА: ДЛИНА НА СТРОКУ =================
// Отличие от префилльного отбора -- НЕ косметическое. В декоде строки это ВИРТУАЛЬНЫЙ БАТЧ:
// при спекуляции k+1 позиций разложены по строкам, у КАЖДОЙ своя `seq_len`, свой `kv_start`
// и своя причинность. Взять максимум длины на все строки нельзя молча: причинность отбора
// уедет на k позиций, ядро внимания обрежет по `k1 = min(..., b_len)` -- отказа не будет,
// будет ТИХАЯ ПОРЧА приёмки черновика.
//
// Средние ключи читаются ПРЯМО из таблицы пула по таблице блоков СТРОКИ -- промежуточный
// тензор не заводится вовсе (в декоде он стоил бы 1 МиБ выделения на вызов).
__global__ void score_part_rows(
        const __half* __restrict__ Q,        // [N, H, D]
        const __half* __restrict__ kbarPool, // [страниц * subPerPage, Hkv, D]
        const __half* __restrict__ devPool,  // [страниц * subPerPage, Hkv, D] или nullptr
        const int* __restrict__ bt,          // [запросов, maxBlocks]
        const int* __restrict__ seqLens,     // [запросов]
        const int* __restrict__ kvStart,     // [запросов] или nullptr
        int N, int H, int Hkv, int D, int B, int bs, int maxBlocks, int subPerPage,
        float scale, float gamma, int NB, int JS,
        float* __restrict__ mOut, float* __restrict__ sOut, int gstr, int qGrp, int qCausal) {
    // [ВАРП НА БЛОК, 14.09 18:20] Прежде CTA шла по своим блокам J ПОСЛЕДОВАТЕЛЬНО с двумя барьерами на блок
    // (5.4 мс/шаг на 250K после подъёма JS). Теперь каждый ВАРП берёт свой J: средний ключ и размах читаются
    // прямо в регистры (8 полуслов на лейн), счёт g голов -- варповые суммы, барьеров в цикле нет.
    // Q один раз в разделяемой [g][D]. Результат тот же: по головам (gstr>1) или свёртка LSE по головам.
    const int r = blockIdx.x, hk = blockIdx.y, sl = blockIdx.z;
    const int g = H / Hkv;
    extern __shared__ char smem[];
    __half* Qs = (__half*)smem;              // [g][D]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    const int req = (qGrp > 1) ? (r / qGrp) : r;
    const int qpos = (qGrp > 1) ? (r - req * qGrp) : 0;
    for (int hh = warp; hh < g; hh += NW)
        for (int c = lane; c < D; c += 32)
            Qs[(size_t)hh * D + c] = Q[((size_t)r * H + hk * g + hh) * D + c];
    __syncthreads();
    const int T = seqLens[req] - ((qGrp > 1 && qCausal) ? (qGrp - 1 - qpos) : 0);
    const int j0lim = kvStart ? (kvStart[req] / B) : 0;
    const int NBvis = min(NB, (T + B - 1) / B);
    const int per = (NBvis + JS - 1) / JS;
    const int j0 = max(j0lim, sl * per), j1 = min(sl * per + per, NBvis);
    float* mJ = mOut + ((size_t)r * Hkv + hk) * NB * (size_t)gstr;
    float* SJ = sOut + ((size_t)r * Hkv + hk) * NB * (size_t)gstr;
    const int EPL = D / 32;                  // полуслов на лейн (256/32 = 8)
    for (int J = j0 + warp; J < j1; J += NW) {
        const long p0 = (long)J * B, pg = bt[(long)req * maxBlocks + p0 / bs];
        const long sub = (p0 - (p0 / bs) * bs) / B;
        const __half* kb = kbarPool + ((pg * subPerPage + sub) * Hkv + hk) * D + lane * EPL;
        float kv[8], dv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { kv[e] = (e < EPL) ? __half2float(kb[e]) : 0.f; dv[e] = 0.f; }
        if (gamma != 0.f && devPool != nullptr) {
            const __half* db = devPool + ((pg * subPerPage + sub) * Hkv + hk) * D + lane * EPL;
#pragma unroll
            for (int e = 0; e < 8; ++e) dv[e] = (e < EPL) ? __half2float(db[e]) : 0.f;
        }
        const bool vis = (J * B <= T - 1);
        float mw = -1e30f, sw = 0.f;
        for (int hh = 0; hh < g; ++hh) {
            const __half* qp = Qs + (size_t)hh * D + lane * EPL;
            float a = 0.f, ad = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (e < EPL) { const float q = __half2float(qp[e]); a += q * kv[e]; ad += fabsf(q) * dv[e]; }
            }
            a = (warp_sum(a) + ((gamma != 0.f) ? gamma * warp_sum(ad) : 0.f)) * scale;
            const float v = vis ? a : -1e30f;
            if (gstr > 1) {
                if (lane == 0) { mJ[(size_t)hh * NB + J] = v; SJ[(size_t)hh * NB + J] = vis ? 1.f : 0.f; }
            } else if (v > -1e29f) {
                const float mn = fmaxf(mw, v);
                sw = sw * __expf(mw - mn) + __expf(v - mn); mw = mn;
            }
        }
        if (gstr == 1 && lane == 0) { mJ[J] = (sw > 0.f) ? mw : -1e30f; SJ[J] = sw; }
    }
}

// [НОРМИРОВКА ЕДИНИЦ ПО CTA, 14.09 19:10] Прежде одна CTA на (строку, голову) нормировала все 24 единицы (строки группы x
// q-головы) тремя проходами по NB -- 7.7 мс/шаг на 250K. Теперь CTA на единицу: max и сумма по своим J, доля массы
// atomicAdd в общий буфер Eg[r][hk][J] (одинаковый для всех строк группы). Порог и упаковка -- в select_csr_rows.
__global__ void norm_units_rows(
        const int* __restrict__ seqLens, const int* __restrict__ kvStart, int Hkv, int B,
        const float* __restrict__ mOut, const float* __restrict__ sOut, float* __restrict__ Eg,
        int NB, int grp, int gstr, int qGrp, int qCausal) {
    const int r = blockIdx.x, hk = blockIdx.y, u = blockIdx.z, N = gridDim.x;
    const int gb = (grp > 1) ? (r / grp) * grp : r;
    const int ge = (grp > 1) ? min(gb + grp, N) : (r + 1);
    const int gs = (gstr > 0) ? gstr : 1;
    if (u >= (ge - gb) * gs) return;
    int T = 0, j0lim = 1 << 30;
    for (int q = gb; q < ge; ++q) {
        const int req = (qGrp > 1) ? (q / qGrp) : q, qpos = (qGrp > 1) ? (q - req * qGrp) : 0;
        T = max(T, seqLens[req] - ((qGrp > 1 && qCausal) ? (qGrp - 1 - qpos) : 0));
        j0lim = min(j0lim, kvStart ? (kvStart[req] / B) : 0);
    }
    const int NBvis = min(NB, (T + B - 1) / B);
    const int q = gb + u / gs, h = u - (u / gs) * gs;
    const float* mq = mOut + (((size_t)q * Hkv + hk) * gs + h) * NB;
    const float* sq = sOut + (((size_t)q * Hkv + hk) * gs + h) * NB;
    __shared__ float red[32]; __shared__ float Ms, Zs;
    float m = -1e30f;
    for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x) m = fmaxf(m, mq[J]);
    m = block_max(m, red);
    __syncthreads();
    if (threadIdx.x == 0) Ms = m;
    __syncthreads();
    float z = 0.f;
    for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x)
        if (mq[J] > -1e29f) z += sq[J] * __expf(mq[J] - Ms);
    z = block_sum(z, red);
    __syncthreads();
    if (threadIdx.x == 0) Zs = fmaxf(z, 1e-30f);
    __syncthreads();
    float* E = Eg + ((size_t)r * Hkv + hk) * NB;
    for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x)
        if (mq[J] > -1e29f) atomicAdd(&E[J], sq[J] * __expf(mq[J] - Ms) / Zs);
}

// [СЧЁТ СВОДИТСЯ ПО ГРУППЕ СТРОК, А НЕ ПО ОДНОЙ]
// ЗАЧЕМ. Префилльное правило работает потому, что счёт блока сводится по 16-32 ЗОНДОВЫМ
// строкам плитки: шум одной строки гасится соседями. В декоде строка ОДНА, гасить нечем --
// и замер 28.08 это показал: качество ОДИНАКОВО при alpha 0.02, 0.005 и 0.001 (три порядка!)
// и чисто только при нулевом отбрасывании, то есть нужные блоки лежат НИЖЕ всей массы, а не
// у порога. Это не порог, это оценка.
// ЧТО ДЕЛАЕМ. У спекуляции на КАЖДЫЙ запрос уже приходится grp = k+1 строк (позиции
// виртуального батча), лежащих подряд. Сводим массу блока по ним всем с ОБЩИМ максимумом --
// получаем ту же аггрегацию, что в префилле, бесплатно: строки уже посчитаны.
// grp = 1 -- прежнее поведение бит в бит.
__global__ void select_csr_rows(
        const int* __restrict__ seqLens, const int* __restrict__ kvStart,
        int Hkv, int B, float alpha, int SINK, int WIN, int REC,
        const float* __restrict__ mOut, const float* __restrict__ sOut,
        int* __restrict__ cnt, int* __restrict__ idx, int NB, int grp, float topf, int gstr,
        int qGrp, int qCausal, const float* __restrict__ Eg, int minblk) {
    const int r = blockIdx.x, hk = blockIdx.y, N = gridDim.x;
    const int gb = (grp > 1) ? (r / grp) * grp : r;
    const int ge = (grp > 1) ? min(gb + grp, N) : (r + 1);
    // ГРАНИЦЫ ПО ГРУППЕ: длиннейшая строка задаёт правый край, левейшее окно -- левый.
    // Надмножество законно (лишний блок стоит времени, потерянный -- качества).
    int T = 0, j0lim = 1 << 30;
    for (int q = gb; q < ge; ++q) {
        const int req = (qGrp > 1) ? (q / qGrp) : q, qpos = (qGrp > 1) ? (q - req * qGrp) : 0;
        T = max(T, seqLens[req] - ((qGrp > 1 && qCausal) ? (qGrp - 1 - qpos) : 0));
        j0lim = min(j0lim, kvStart ? (kvStart[req] / B) : 0);
    }
    const int NBvis = min(NB, (T + B - 1) / B);
    __shared__ float E[4096]; __shared__ float Mall; __shared__ float red[32]; __shared__ float thrS;
    __shared__ float Mq[64]; __shared__ float Zq[64];
    // [ОТБОР ПО СТРОКЕ, 14.09 -- FA2SM70_SPARSE_DEC_TOPF] Оракул строки (tools/orakul_dekod_stroki.py, записка 26 §7.7):
    // у строк-искателей блок иглы в верхних 10 % блоков ПО ОЦЕНЩИКУ у 98 % строк, в верхних 24 % у всех, на всех слоях --
    // по одной строке оценщик точен. Терял иглы не он, а сведение k+1 строк с ОБЩИМ максимумом ниже (та же болезнь, что
    // в префилле, §7.5): строка с самыми большими счётами забирает всю «массу». Здесь каждая строка нормируется своей
    // суммой (softmax по блокам), строки группы складываются равноправно, и берётся ВЕРХНЯЯ ДОЛЯ topf блоков (порог --
    // k-й наибольший счёт, бисекция по значению с подсчётом), а не alpha·max. Сток/окно/REC добавляются как прежде.
    const int gs = (gstr > 0) ? gstr : 1;
    const bool po_stroke = (topf > 0.f) && (ge - gb) * gs <= 64;
    if (po_stroke) {
        // единица нормировки -- (строка q, голова h): каждая суммируется в 1 по блокам, потом все складываются равноправно
        // [19:10] массы единиц уже сложены ядром norm_units_rows в Eg -- здесь только копия в разделяемую
        const float* Er = Eg + ((size_t)r * Hkv + hk) * NB;
        for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x) E[J] = Er[J];
        __syncthreads();
        // порог = k-й наибольший E: бисекция по значению, 24 шага достаточно для fp32 на [0, emax]
        float em = 0.f;
        for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x) em = fmaxf(em, E[J]);
        em = block_max(em, red);
        const int nvis = max(1, NBvis - j0lim);
        // [ПОЛ ПО ЧИСЛУ БЛОКОВ 18.09] `topf` -- ДОЛЯ, и на коротком контексте доля превращается
        // в единицы блоков: при 8K это 128 блоков, 0.15 от них -- 19, то есть ~1.2K токенов из 8K.
        // Владелец поймал это прибором границы: 7 потерь из 50 на контекстах <= 11K, причём с
        // НОВОЙ подписью -- код найден, но с ошибкой в последней цифре (взят соседний токен).
        // На 100K и 250K потерь нет: там 0.15 даёт сотни блоков. Порог длины сюда поставить
        // нельзя -- декод идёт под ЗАХВАЧЕННЫМ графом, и хозяйская ветка запеклась бы навсегда.
        // Поэтому пол ставится ЗДЕСЬ, по числу блоков: при nvis <= minblk отбор становится
        // ПЛОТНЫМ сам собой, а на длинном контексте minblk меньше доли и ничего не меняет.
        // [ПОПРАВКА 18.09, ПОЙМАНА ЗАМЕРОМ] Сперва я написал ПОЛ: max(minblk, доля). Он лечит
        // короткий контекст, но срабатывает и в СЕРЕДИНЕ: при 52K блоков 822, доля даёт 123, а
        // пол поднимает до 256 -- вдвое больше работы внимания. Замер на боевом это и показал
        // (52 615 ткн: 48.4 -> 42-44 ток/с). Нужен не пол, а ПЕРЕКЛЮЧАТЕЛЬ: ниже порога отбор
        // плотный, выше -- ровно прежняя доля, без единого лишнего блока.
        const int kfl = (minblk > 1) ? minblk : 0;
        const int k = (nvis <= kfl) ? nvis
                                    : max(1, min(nvis, (int)ceilf(topf * (float)nvis)));
        float lo = 0.f, hi = em;
        for (int it = 0; it < 24; ++it) {
            const float mid = 0.5f * (lo + hi);
            float c = 0.f;
            for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x) c += (E[J] >= mid) ? 1.f : 0.f;
            c = block_sum(c, red);
            __syncthreads();
            if (c >= (float)k) lo = mid; else hi = mid;
        }
        if (threadIdx.x == 0) thrS = lo;
        __syncthreads();
    } else {
    if (threadIdx.x == 0) {
        float M = -1e30f;
        for (int q = gb; q < ge; ++q) {
            const float* mq = mOut + ((size_t)q * Hkv + hk) * NB;
            for (int J = j0lim; J < NBvis; ++J) M = fmaxf(M, mq[J]);
        }
        Mall = M;
    }
    __syncthreads();
    for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x) {
        float acc = 0.f;
        for (int q = gb; q < ge; ++q) {
            const float* mq = mOut + ((size_t)q * Hkv + hk) * NB;
            const float* sq = sOut + ((size_t)q * Hkv + hk) * NB;
            if (mq[J] > -1e29f) acc += sq[J] * __expf(mq[J] - Mall);
        }
        E[J] = acc;
    }
    __syncthreads();
    }
    // [ПАРАЛЛЕЛЬНАЯ УПАКОВКА, 14.09 17:50] Прежде список собирал ОДИН поток циклом по ~3700 блокам -- 14.7 мс на шаг
    // на 250K (профиль серии 26). Теперь: порог -- блочный максимум, упаковка -- префикс-сумма по чанкам blockDim
    // (баллот варпа + скан 8 варпов), порядок J возрастающий сохраняется бит в бит.
    float em2 = 0.f;
    for (int J = threadIdx.x + j0lim; J < NBvis; J += blockDim.x) em2 = fmaxf(em2, E[J]);
    em2 = block_max(em2, red);
    __syncthreads();
    const float thr = po_stroke ? thrS : alpha * em2;
    const int jlast = (T - 1) / B;
    int* dst = idx + ((size_t)r * Hkv + hk) * NB;
    __shared__ int wcnt[32]; __shared__ int running;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (int base = j0lim; base < NBvis; base += blockDim.x) {
        const int J = base + threadIdx.x;
        bool keep = false;
        if (J < NBvis) {
            keep = (E[J] >= thr);
            if (J < j0lim + SINK) keep = true;
            if (J > jlast - WIN && J <= jlast) keep = true;
            if (J > jlast - REC && J <= jlast) keep = true;
        }
        const unsigned bal = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) wcnt[wid] = __popc(bal);
        __syncthreads();
        int off = running;
        for (int w = 0; w < wid; ++w) off += wcnt[w];
        if (keep) dst[off + __popc(bal & ((1u << lane) - 1u))] = J;
        __syncthreads();
        if (threadIdx.x == 0) { int s = 0; for (int w = 0; w < nw; ++w) s += wcnt[w]; running += s; }
        __syncthreads();
    }
    if (threadIdx.x == 0) cnt[(size_t)r * Hkv + hk] = running;
}

}  // namespace

// Обновить таблицу средних ключей ПО ТРОНУТЫМ подблокам. Зовётся ПОСЛЕ записи KV.
// Форма таблицы: [страниц * (bs/B), Hkv, D] fp16. Создаётся вызывающим один раз на слой.
void fa2sm70_blk_mean_update(torch::Tensor Kc, torch::Tensor Kst, torch::Tensor slot_mapping,
                             torch::Tensor kbarPool, int64_t B,
                             c10::optional<torch::Tensor> devPool = c10::nullopt) {
    const c10::cuda::CUDAGuard guard(Kc.device());
    TORCH_CHECK(Kc.dim() == 4 && Kc.scalar_type() == torch::kChar, "пул K -- [nb, bs, Hkv, D] int8");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong && slot_mapping.is_contiguous(),
                "слоты -- int64 плотные");
    TORCH_CHECK(kbarPool.scalar_type() == torch::kFloat16 && kbarPool.is_contiguous(),
                "таблица средних -- fp16 плотная");
    const int bs = Kc.size(1), Hkv = Kc.size(2), D = Kc.size(3);
    TORCH_CHECK(bs % B == 0, "страница пула кратна блоку отбора: ", bs, " % ", B);
    const int subPerPage = bs / (int)B, n = (int)slot_mapping.numel();
    if (n == 0) return;
    const int grp = (n + (int)B - 1) / (int)B;
    blk_mean_update<<<dim3(grp, Hkv), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        Kc.data_ptr<int8_t>(), Kst.data_ptr<float>(), slot_mapping.data_ptr<long>(),
        n, bs, Hkv, D, (int)B, (long)Kc.stride(0),
        reinterpret_cast<__half*>(kbarPool.data_ptr<at::Half>()), subPerPage,
        (devPool.has_value() && devPool->defined())
            ? reinterpret_cast<__half*>(devPool->data_ptr<at::Half>()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Отбор для ДЕКОДА: длина на строку, буферы ПРЕДВЫДЕЛЕНЫ вызывающим.
// Предвыделение здесь не про скорость (хотя и про неё: замерено 3.1 мс/шаг накладных при 250K),
// а про CUDA-ГРАФ, под которым идёт декод: адреса обязаны быть постоянными между повторами.
void fa2sm70_sparse_select_rows(
        torch::Tensor Q, torch::Tensor kbarPool, torch::Tensor BlockTable, torch::Tensor seqLens,
        c10::optional<torch::Tensor> kvStart, double scale, double alpha, int64_t B,
        torch::Tensor cnt, torch::Tensor idx, torch::Tensor mBuf, torch::Tensor sBuf,
        int64_t bs, int64_t SINK, int64_t WIN, int64_t REC,
        c10::optional<torch::Tensor> devPool = c10::nullopt, double gamma = 0.0,
        int64_t grp = 1, double topf = 0.0, int64_t qGrp = 1, int64_t qCausal = 1,
        int64_t minblk = 1) {
    const c10::cuda::CUDAGuard guard(Q.device());
    TORCH_CHECK(Q.dim() == 3, "Q декода -- [строк, H, D]");
    TORCH_CHECK(topf >= 0.0 && topf <= 1.0, "topf -- доля блоков в [0, 1]");
    TORCH_CHECK(BlockTable.dim() == 2 && BlockTable.scalar_type() == torch::kInt32,
                "таблица блоков -- [строк, maxBlocks] int32");
    TORCH_CHECK(seqLens.scalar_type() == torch::kInt32, "длины -- int32");
    TORCH_CHECK(kbarPool.scalar_type() == torch::kFloat16, "таблица средних -- fp16");
    TORCH_CHECK(cnt.scalar_type() == torch::kInt32 && idx.scalar_type() == torch::kInt32,
                "выходы отбора -- int32");
    const int N = Q.size(0), H = Q.size(1), D = Q.size(2);
    const int Hkv = kbarPool.size(1), maxBlocks = BlockTable.size(1);
    const int NB = idx.size(2), subPerPage = (int)(bs / B);
    TORCH_CHECK(bs % B == 0, "страница пула кратна блоку отбора");
    TORCH_CHECK(NB <= 4096, "блоков ключей больше 4096 -- буфер порога мал");
    TORCH_CHECK(cnt.size(0) == N && idx.size(0) == N, "формы выходов не по строкам");
    TORCH_CHECK(qGrp >= 1 && (qGrp == 1 || N % qGrp == 0), "строк ", N, " не кратно qGrp ", qGrp);
    TORCH_CHECK(seqLens.numel() * qGrp >= N && BlockTable.size(0) * qGrp >= N,
                "длины/таблица блоков по запросу: ", seqLens.numel(), "x", qGrp, " < строк ", N);
    const int* kvp = kvStart.has_value() ? kvStart.value().data_ptr<int>() : nullptr;
    const int g = H / Hkv;
    const size_t sh = (size_t)g * D * sizeof(__half);
    auto st = at::cuda::getCurrentCUDAStream();
    // Кусков по блокам ключей столько, чтобы накрыть SM: строк обычно 1-4, значит вся
    // параллельность здесь и живёт.
    // [14.09] цель по числу CTA поднята 160 -> 1024: при 4 строках x 2 головы было 20 CTA-срезов по ~185 блоков
    // последовательно (8.5 мс/шаг на 250K); теперь 64 среза по ~58 блоков -- все CTA идут одновременно.
    const int JS = max(1, min(64, min((1024 + N * Hkv - 1) / max(1, N * Hkv), NB)));
    mBuf.fill_(-1e30f); sBuf.zero_();
    cudaFuncSetAttribute(score_part_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    TORCH_CHECK(grp >= 1, "grp >= 1");
    // Кратность обязана быть точной: иначе группа съедет и строка возьмёт индекс ЧУЖОГО
    // запроса -- это тихая порча, а не отказ.
    TORCH_CHECK(grp == 1 || (Q.size(0) % grp) == 0,
                "строк ", Q.size(0), " не кратно группе ", grp);
    const __half* _dvp = nullptr;
    if (devPool.has_value() && devPool->defined()) {
        TORCH_CHECK(devPool->scalar_type() == torch::kFloat16 && devPool->is_contiguous(),
                    "таблица размахов -- fp16, плотная");
        TORCH_CHECK(devPool->numel() == kbarPool.numel(),
                    "таблица размахов той же формы, что таблица средних");
        _dvp = reinterpret_cast<const __half*>(devPool->data_ptr<at::Half>());
    }
    TORCH_CHECK(!(gamma != 0.0 && _dvp == nullptr),
                "gamma>0 требует таблицу размахов devPool");
    // [ПО ГОЛОВАМ] при topf>0 и буферах m/s шириной >= g*NB счёт пишется по головам (gstr = g), иначе -- свёртка как прежде
    const int gstr = (topf > 0.0 && mBuf.dim() == 3 && mBuf.size(2) >= (int64_t)NB * g && sBuf.size(2) >= (int64_t)NB * g) ? g : 1;
    TORCH_CHECK(mBuf.is_contiguous() && sBuf.is_contiguous(), "буферы m/s плотные");
    // [СЧЁТ ЗОНДОВ ДЕКОДА МАТРИЦАМИ, 14.09, FA2SM70_SEL_GEMM_DEC=1] score_part_rows стоит 8.5 мс/шаг на 250K (цикл по ~3700
    // блокам с барьером на блок). Здесь: физические подблоки строки по таблице блоков -> gather таблицы средних/размахов ->
    // batched matmul [N,Hkv,g,D] x [N,Hkv,D,NB]; пишем ПО ГОЛОВАМ в m/s (gstr = g), select_csr_rows не меняется.
    const char* gd_e = getenv("FA2SM70_SEL_GEMM_DEC");
    if (gd_e && atoi(gd_e) == 1 && gstr > 1) {
        static bool _gd_skazano = false;
        if (!_gd_skazano) { _gd_skazano = true; fprintf(stderr, "[fa2_sm70] отбор декода: зонды счётом МАТРИЦАМИ (FA2SM70_SEL_GEMM_DEC=1)\n"); }
        // [ПАМЯТЬ] под графом декода каждое выделение ложится в частный пул графа (на каждый размер батча свой) --
        // 100 МиБ переходных на вызов съели запас карты (OOM на префилле 12:25). Буферы сборки СТАТИЧЕСКИЕ (растут только),
        // матрицы -- в fp16 без копий: Q переставлен в [Hkv, N, g, D] (48 КиБ), K/размах собраны в [Hkv, N*NB, D].
        static torch::Tensor _Kb, _Db; static int64_t _cap = 0;
        static std::vector<torch::Tensor> _KbDb_keep;   // [14.10] see _Eg below: old address may be baked into a graph
        const int64_t need = (int64_t)N * NB;
        if (!_Kb.defined() || _Kb.device() != Q.device() || _cap < need) {
            if (_Kb.defined()) { _KbDb_keep.push_back(_Kb); _KbDb_keep.push_back(_Db); }
            _cap = std::max<int64_t>(need, 64LL * NB);
            _Kb = torch::empty({(long)Hkv, (long)_cap, (long)D}, torch::dtype(torch::kFloat16).device(Q.device()));
            _Db = torch::empty({(long)Hkv, (long)_cap, (long)D}, torch::dtype(torch::kFloat16).device(Q.device()));
        }
        const auto optl = torch::dtype(torch::kLong).device(Q.device());
        auto Jb = torch::arange((long)NB, optl) * (int64_t)B;                       // [NB] начало блока (позиция)
        auto page_of = torch::div(Jb, (int64_t)bs, "floor"), sub_of = torch::div(Jb % (int64_t)bs, (int64_t)B, "floor");
        auto rows = torch::arange((long)N, optl); auto req = torch::div(rows, (int64_t)qGrp, "floor"); auto qpos = rows - req * (int64_t)qGrp;
        auto btl = BlockTable.to(torch::kLong).index_select(0, req);                // [N, maxBlocks]
        TORCH_CHECK((int64_t)(NB - 1) * B / bs < maxBlocks, "таблица блоков короче NB");
        auto phys = (btl.index_select(1, page_of) * (int64_t)subPerPage + sub_of.unsqueeze(0)).reshape(-1)
                        .clamp_(0, kbarPool.size(0) - 1);                                  // [N*NB]
        auto Kg = _Kb.narrow(1, 0, need);                                                  // [Hkv, N*NB, D]
        for (int hk = 0; hk < Hkv; ++hk) { auto _o = Kg[hk]; at::index_select_out(_o, kbarPool.select(1, hk), 0, phys); }
        auto Qh = Q.view({(long)N, (long)Hkv, (long)g, (long)D}).transpose(0, 1).contiguous();   // [Hkv, N, g, D] fp16
        auto Kv = Kg.view({(long)Hkv, (long)N, (long)NB, (long)D});
        auto S = torch::matmul(Qh, Kv.transpose(-1, -2)).to(torch::kFloat32);             // [Hkv, N, g, NB]
        if (gamma != 0.0 && _dvp != nullptr) {
            auto Dg = _Db.narrow(1, 0, need);
            for (int hk = 0; hk < Hkv; ++hk) { auto _o = Dg[hk]; at::index_select_out(_o, devPool->select(1, hk), 0, phys); }
            S.add_(torch::matmul(Qh.abs(), Dg.view({(long)Hkv, (long)N, (long)NB, (long)D}).transpose(-1, -2)).to(torch::kFloat32), gamma);
        }
        S.mul_(scale);
        auto Tr = seqLens.to(torch::kLong).index_select(0, req);
        if (qGrp > 1 && qCausal) Tr = Tr - ((int64_t)qGrp - 1 - qpos);
        auto valid = Jb.unsqueeze(0) <= (Tr - 1).unsqueeze(1);                       // [N, NB]: блок виден строке
        if (kvp != nullptr) {
            auto j0 = torch::div(kvStart.value().to(torch::kLong).index_select(0, req), (int64_t)B, "floor").unsqueeze(1);
            valid = valid & (torch::arange((long)NB, optl).unsqueeze(0) >= j0);
        }
        S = S.transpose(0, 1);                                                             // [N, Hkv, g, NB] (вид)
        auto v4 = valid.view({(long)N, 1, 1, (long)NB});
        S.masked_fill_(~v4, -1e30f);
        mBuf.view({(long)N, (long)Hkv, (long)g, (long)NB}).copy_(S);
        sBuf.view({(long)N, (long)Hkv, (long)g, (long)NB}).copy_(v4.to(torch::kFloat32).expand({(long)N, (long)Hkv, (long)g, (long)NB}));
    } else
    score_part_rows<<<dim3(N, Hkv, JS), 256, sh, st>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kbarPool.data_ptr<at::Half>()),
        _dvp,
        BlockTable.data_ptr<int>(), seqLens.data_ptr<int>(), kvp,
        N, H, Hkv, D, (int)B, (int)bs, maxBlocks, subPerPage, (float)scale, (float)gamma, NB, JS,
        mBuf.data_ptr<float>(), sBuf.data_ptr<float>(), gstr, (int)qGrp, (int)qCausal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const float* _Egp = nullptr;
    if (topf > 0.0) {
        // [ВИСЯЧИЙ АДРЕС -- 14.10] Буфер растёт при росте строк виртуального батча (запросы x (k+1)), а его
        // адрес запечён в CUDA-графах (полный граф цели, графы тела MTP): повтор графа делает memset и пишет
        // сюда. Прежде старый тензор отпускался присваиванием -- запись шла в память нового владельца
        // (Xid 31 / порча, разбор: docs/sm70/VALIDATION.md, инцидент 2026-10-14). Теперь: ёмкость сразу под
        // 64 строки (16 запросов x 4) и отставленные тензоры не освобождаются никогда.
        static torch::Tensor _Eg; static int64_t _Egcap = 0;
        static std::vector<torch::Tensor> _Eg_keep;
        const int64_t need = (int64_t)N * Hkv * NB;
        if (!_Eg.defined() || _Eg.device() != Q.device() || _Egcap < need) {
            if (_Eg.defined()) _Eg_keep.push_back(_Eg);
            _Egcap = std::max<int64_t>(need, 64LL * Hkv * NB);
            _Eg = torch::zeros({_Egcap}, torch::dtype(torch::kFloat32).device(Q.device()));
        }
        cudaMemsetAsync(_Eg.data_ptr<float>(), 0, need * sizeof(float), st);
        const int NU = (int)grp * gstr;
        norm_units_rows<<<dim3(N, Hkv, NU), 256, 0, st>>>(
            seqLens.data_ptr<int>(), kvp, Hkv, (int)B, mBuf.data_ptr<float>(), sBuf.data_ptr<float>(),
            _Eg.data_ptr<float>(), NB, (int)grp, gstr, (int)qGrp, (int)qCausal);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        _Egp = _Eg.data_ptr<float>();
    }
    select_csr_rows<<<dim3(N, Hkv), 256, 0, st>>>(
        seqLens.data_ptr<int>(), kvp, Hkv, (int)B, (float)alpha, (int)SINK, (int)WIN, (int)REC,
        mBuf.data_ptr<float>(), sBuf.data_ptr<float>(), cnt.data_ptr<int>(), idx.data_ptr<int>(), NB, (int)grp, (float)topf, gstr,
        (int)qGrp, (int)qCausal, _Egp, (int)minblk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> fa2sm70_sparse_select(
        torch::Tensor Q, torch::Tensor Kc, torch::Tensor Kst, torch::Tensor BlockTable,
        int64_t T, double scale, double alpha,
        int64_t B, int64_t TI, int64_t PROBE, int64_t SINK, int64_t WIN, int64_t REC,
        c10::optional<torch::Tensor> kbarPool = c10::nullopt, double beta = 0.0,
        double gamma = 0.0, c10::optional<torch::Tensor> devPool = c10::nullopt) {
    const c10::cuda::CUDAGuard guard(Q.device());
    TORCH_CHECK(Q.dim() == 3, "Q ожидается [Sq, H, D]");
    TORCH_CHECK(Kc.dim() == 4 && Kc.scalar_type() == torch::kChar, "пул K -- [nb, bs, Hkv, D] int8");
    TORCH_CHECK(Kst.scalar_type() == torch::kFloat32 && BlockTable.scalar_type() == torch::kInt32,
                "масштабы fp32, таблица int32");
    TORCH_CHECK(Kc.stride(3) == 1, "строка токена в пуле должна быть плотной");
    const int Sq = Q.size(0), H = Q.size(1), D = Q.size(2);
    const int bs = Kc.size(1), Hkv = Kc.size(2);
    TORCH_CHECK(D == Kc.size(3) && D <= 256, "d совпадает с пулом и не больше 256");
    TORCH_CHECK(H % Hkv == 0, "GQA: H кратно Hkv");
    TORCH_CHECK(PROBE * (H / Hkv) <= 256, "зондовых строк не больше 256");
    const int NB = (int)((T + B - 1) / B);
    const int ntiles = (Sq + (int)TI - 1) / (int)TI;
    auto opt16 = torch::dtype(torch::kFloat16).device(Q.device());
    auto opt32 = torch::dtype(torch::kInt32).device(Q.device());
    auto kbar = torch::empty({Hkv, NB, D}, opt16);
    // НУЛИ, А НЕ МУСОР: по ветке готовой таблицы средних разброс не считается, и beta*0
    // обязана дать РОВНО прежнее правило, а не случайную поправку.
    auto rbar = torch::zeros({Hkv, NB}, torch::dtype(torch::kFloat32).device(Q.device()));
    // Размах по измерениям -- та же форма, что и средние. Нули по ветке готовой таблицы.
    auto dev = torch::zeros({Hkv, NB, D}, opt16);
    auto st = at::cuda::getCurrentCUDAStream();

    // Если подана ГОТОВАЯ таблица (обновлённая при записи KV) -- собираем средние из неё
    // по таблице блоков, вместо чтения всего K. Иначе -- прежний полный пересчёт.
    if (kbarPool.has_value()) {
        auto KP = kbarPool.value();
        TORCH_CHECK(KP.scalar_type() == torch::kFloat16 && KP.is_contiguous(),
                    "таблица средних: fp16, плотная");
        const int subPerPage = bs / (int)B;
        TORCH_CHECK(bs % B == 0, "страница пула кратна блоку отбора: ", bs, " % ", B);
        TORCH_CHECK(KP.numel() == (long)Kc.size(0) * subPerPage * Hkv * D,
                    "таблица средних не той формы");
        const __half* _dpin = nullptr;
        if (devPool.has_value() && devPool->defined()) {
            TORCH_CHECK(devPool->scalar_type() == torch::kFloat16 && devPool->is_contiguous()
                        && devPool->numel() == KP.numel(),
                        "таблица размахов: fp16, плотная, той же формы что средние");
            _dpin = reinterpret_cast<const __half*>(devPool->data_ptr<at::Half>());
        }
        TORCH_CHECK(!(gamma != 0.0 && _dpin == nullptr),
                    "gamma>0 с готовой таблицей средних требует и таблицу размахов");
        gather_means<<<dim3(NB, Hkv), 256, 0, st>>>(
            reinterpret_cast<const __half*>(KP.data_ptr<at::Half>()),
            BlockTable.data_ptr<int>(), bs, Hkv, D, (int)B, subPerPage,
            reinterpret_cast<__half*>(kbar.data_ptr<at::Half>()), NB,
            _dpin, reinterpret_cast<__half*>(dev.data_ptr<at::Half>()),
            rbar.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else
    blk_mean_i8_paged<<<dim3(NB, Hkv), 256, 0, st>>>(
        Kc.data_ptr<int8_t>(), Kst.data_ptr<float>(), BlockTable.data_ptr<int>(),
        (int)T, bs, Hkv, D, (int)B, (long)Kc.stride(0),
        reinterpret_cast<__half*>(kbar.data_ptr<at::Half>()), NB,
        rbar.data_ptr<float>(), reinterpret_cast<__half*>(dev.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Среднее по блокам -- одним свёртыванием ВНУТРИ расширения: выносить его в питон
    // значило бы добавить хозяйскую работу в путь префилла ради одного числа на голову.
    auto rmu = (beta != 0.0) ? rbar.mean(1).contiguous()
                             : torch::zeros({Hkv}, torch::dtype(torch::kFloat32).device(Q.device()));
    TORCH_CHECK(NB <= 4096, "отбор: блоков ключей больше 4096 (", NB, ") -- буфер порога мал");
    auto cnt = torch::zeros({ntiles, Hkv}, opt32);
    auto idx = torch::zeros({ntiles, Hkv, NB}, opt32);
    auto optf = torch::dtype(torch::kFloat32).device(Q.device());
    // ЗАПОЛНЯТЬ -inf ОБЯЗАТЕЛЬНО: куски делят [0, NBvis), а хвост [NBvis, NB) не пишет НИКТО.
    // С `empty` туда попал бы мусор, и максимум по J взял бы его -- порог поехал бы молча.
    auto mBuf = torch::full({ntiles, Hkv, NB}, -1e30f, optf);
    auto sBuf = torch::zeros({ntiles, Hkv, NB}, optf);
    const int R = (int)PROBE * (H / Hkv);
    // [КУСОК ЗОНДОВЫХ СТРОК] Разделяемая падает кратно числу кусков; занятость растёт.
    // Умолчание 48 строк = 24 КиБ -> три блока на SM вместо одного. Перебирается
    // FA2SM70_SEL_RC; 0 или >=R -- прежнее поведение (один кусок).
    const char* rc_e = getenv("FA2SM70_SEL_RC");
    int Rc = (rc_e && *rc_e) ? atoi(rc_e) : 48;
    if (Rc <= 0 || Rc > R) Rc = R;
    // [РЕГИСТРОВЫЙ ПУТЬ] При нём разделяемая под Q не нужна вовсе: строку держит варп.
    // Условие применимости то же, что в ядре: строк на варп <= QREG_MAX и D <= QREG_D.
    // [ЗАМЕР 31.08 -- РЕГИСТРОВЫЙ ПУТЬ ОТВЕРГНУТ, УМОЛЧАНИЕ 0]
    // Замысел был верный по форме: разделяемая под Q (49.15 КиБ) держала занятость на одном
    // блоке из SM, а варп читает ТОЛЬКО СВОИ строки -- значит они могут жить в регистрах, и
    // ядро останется цельным без платы занятостью. Гейт пройден ПОБИТОВО (cnt и idx совпали
    // при gamma 0.0 и 0.5). Но замер отверг:
    //     Sq=512  T=8K    130.2 -> 173.3 мкс   x0.75
    //     Sq=1024 T=32K   564.0 -> 664.5       x0.85
    //     Sq=2048 T=131K 2686.7 -> 4478.5      x0.60
    // Причина -- регистровое давление: 12 строк на варп по 4 пары = 48 регистров только под Q,
    // и компилятор разливает их в локальную память, а она медленнее разделяемой. Занятость
    // выросла, а время -- тоже. Рычаг оставлен для повторной проверки на другой форме.
    const char* rg_e = getenv("FA2SM70_SEL_REGQ");
    int REGQ = (rg_e && *rg_e) ? atoi(rg_e) : 0;
    {
        const int _NW = 256 >> 5, _R = PROBE * (H / Hkv);
        if (((_R + _NW - 1) / _NW) > 16 || D > 256) REGQ = 0;
    }
    const size_t sh = REGQ ? (size_t)0
                           : (size_t)Rc * D * sizeof(__half);   // (m, S) уехали в глобальную
    // ГРАНИЦА СЧИТАЕТ И СТАТИЧЕСКУЮ РАЗДЕЛЯЕМУЮ, А НЕ ТОЛЬКО ДИНАМИЧЕСКУЮ.
    // ОТКАЗ, КОТОРЫЙ ЭТО ЛЕЧИТ (28.08): при PROBE=32 динамическая часть выходила ровно
    // 96 КиБ, проверка её пропускала -- а драйвер отвергал, потому что в те же 96 КиБ на
    // блок Volta входит и статика ядра (kb/wm/ws/posr/qn ~ 2.8 КиБ). Падало это не здесь,
    // а глубоко в воркере невнятным `CUDA error: invalid argument`, и подъём умирал.
    // Запас берём с полем: предел sm_70 -- 96 КиБ на блок ВСЕГО.
    const size_t _stat = 2 * 256 * sizeof(__half) + 2 * 64 * sizeof(float)
                       + 256 * sizeof(int) + 256 * sizeof(float)
                       + 2 * 256 * sizeof(__half);        // kb[2], wm+ws[2], posr, qn, db[2]
    TORCH_CHECK(sh + _stat <= 96 * 1024,
                "отбор: разделяемая не влезает: динамическая ", sh, " + статическая ", _stat,
                " > 96 КиБ на блок (sm_70). Уменьшите FA2SM70_SPARSE_PROBE: сейчас PROBE=",
                PROBE, ", строк ", R, ".");
    // [СКОЛЬКО КУСКОВ] Цель -- накрыть все SM: блоков ntiles*Hkv*JS, а их у нас 80.
    // Больше 32 кусков смысла нет: на кусок остаётся меньше блока ключей.
    const int JSbase = (ntiles * Hkv > 0) ? ((160 + ntiles * Hkv - 1) / (ntiles * Hkv)) : 1;
    const char* js_e = getenv("FA2SM70_SEL_JS");
    // Умолчание 16 -- ЗАМЕРЕНО свипом: 1 -> 8.98 мс, 3 -> 7.92, 8 -> 6.66, 16 -> 6.35, 32 -> 6.47.
    const int JS = max(1, min(64, min((js_e && *js_e) ? atoi(js_e) : max(JSbase, 16), NB)));
    // [ПЕРЕБОР УСКОРЕНИЙ 13.09, FA2SM70_SEL_GEMM=1] Счёт зондов -- ТЕНЗОРНЫМ УМНОЖЕНИЕМ вместо score_part.
    // Профиль 250K: score_part 15.6 мс/пуск = 6 % префилла при работе ~80 MFLOP на плитку -- ядро латентное
    // (последовательный цикл по ~200 блокам на CTA, барьеры, варповые своды). Та же математика матрицами:
    // S[t,hk,r,J] = (Qp·kbar^T + gamma·|Qp|·dev^T + beta·(rbar−rmu)·‖q‖)·scale, маска J*B <= pos_r, затем по строкам
    // mJ = max, SJ = Σ exp(s − mJ) -- ровно то, что score_part пишет в mBuf/sBuf; select_csr не меняется.
    // fp32 всюду (у ядра накопление fp32). Строки-зонды -- та же формула qi = pr*(TIe−1)/(PROBE−1).
    const char* gm_e = getenv("FA2SM70_SEL_GEMM");
    if (gm_e && atoi(gm_e) == 1) {
        static bool _gm_skazano = false;   // сообщение маршрута: путь опознаётся ЗАПУСКОМ, а не чтением исходника
        if (!_gm_skazano) { _gm_skazano = true; fprintf(stderr, "[fa2_sm70] отбор: зонды счётом МАТРИЦАМИ (FA2SM70_SEL_GEMM=1), score_part не зовётся\n"); }
        const int g = H / Hkv;
        auto rows = torch::empty({(long)ntiles, (long)PROBE}, torch::dtype(torch::kLong));
        auto ra = rows.accessor<int64_t, 2>();
        for (int t = 0; t < ntiles; ++t) {
            const int i0 = t * (int)TI, TIe = std::min((int)TI, Sq - i0);
            for (int pr = 0; pr < (int)PROBE; ++pr) {
                const int qi = (PROBE > 1) ? (pr * (TIe - 1)) / ((int)PROBE - 1) : 0;
                ra[t][pr] = i0 + qi;
            }
        }
        auto rows_d = rows.to(Q.device());
        auto Qp = Q.index_select(0, rows_d.view(-1))                       // [ntiles*PROBE, H, D]
                    .view({(long)ntiles, (long)PROBE, (long)Hkv, (long)g, (long)D})
                    .permute({0, 2, 1, 3, 4}).contiguous()
                    .view({(long)ntiles, (long)Hkv, (long)PROBE * g, (long)D})
                    .to(torch::kFloat32);                                     // r = pr*g + hh, как в ядре
        auto Kf = kbar.to(torch::kFloat32).transpose(1, 2).unsqueeze(0);      // [1, Hkv, D, NB]
        auto Sdot = torch::matmul(Qp, Kf);                                   // [ntiles, Hkv, R, NB]
        if (gamma != 0.0)
            Sdot.add_(torch::matmul(Qp.abs(), dev.to(torch::kFloat32).transpose(1, 2).unsqueeze(0)), gamma);
        if (beta != 0.0) {
            auto qn = Qp.norm(2, -1, true);                                  // [ntiles, Hkv, R, 1]
            auto rj = (rbar - rmu.unsqueeze(1)) * beta;                      // [Hkv, NB]
            Sdot.add_(qn * rj.unsqueeze(0).unsqueeze(2));
        }
        Sdot.mul_(scale);
        auto posr = (rows_d + (int64_t)(T - Sq)).view({(long)ntiles, 1, (long)PROBE, 1})
                        .expand({(long)ntiles, (long)Hkv, (long)PROBE, (long)g})
                        .reshape({(long)ntiles, (long)Hkv, (long)PROBE * g, 1});
        auto Jb = torch::arange((long)NB, torch::dtype(torch::kLong).device(Q.device())) * (int64_t)B;
        Sdot.masked_fill_(Jb.view({1, 1, 1, (long)NB}) > posr, -1e30f);
        // [МЕХАНИЗМ ПОТЕРИ ИГЛЫ, 14.09 05:05, FA2SM70_SEL_ROWNORM=1] Свёртка зондов E_J = Σ_r exp(s_rJ − M) с ОБЩИМ
        // максимумом M в поздних слоях (острое внимание, разный масштаб строк) отдаёт всю «массу» нескольким строкам:
        // 98 % набирается парой блоков-стоков, строки вопроса в свёртку не попадают, блок иглы теряется (оракул
        // tools/orakul_igly.py по снимку 16 слоёв: слои 31..51 -- 2.8-29 % блоков, игла ПОТЕРЯНА, массы вопроса 36-73 %).
        // Нормировка ПО СТРОКЕ (softmax по блокам в каждой строке, потом сумма строк) держит иглу на всех слоях при
        // 34-65 % блоков в поздних (в ранних 63-80 %, почти как было). Кодируем W_J в (m, S): m = ln W_J, S = 1 --
        // select_csr считает E_J = S·exp(m − M) = W_J / max W, то есть ту же массу с точностью до множителя.
        const char* rn_e = getenv("FA2SM70_SEL_ROWNORM");
        if (rn_e && atoi(rn_e) == 1) {
            static bool _rn_skazano = false;
            if (!_rn_skazano) { _rn_skazano = true; fprintf(stderr, "[fa2_sm70] отбор: свёртка зондов С НОРМИРОВКОЙ ПО СТРОКЕ (FA2SM70_SEL_ROWNORM=1)\n"); }
            auto W = torch::softmax(Sdot, 3).sum(2);                        // [ntiles, Hkv, NB], строка-маска даёт 0
            auto valid = W > 0.f;
            mBuf.copy_(torch::where(valid, torch::log(W.clamp_min(1e-30f)), torch::full_like(W, -1e30f)));
            sBuf.copy_(torch::where(valid, torch::ones_like(W), torch::zeros_like(W)));
        } else {
        auto mJ = std::get<0>(Sdot.max(2));                                  // [ntiles, Hkv, NB]
        auto valid = mJ > -1e29f;
        auto SJ = torch::exp(Sdot - mJ.unsqueeze(2)).sum(2);
        mBuf.copy_(torch::where(valid, mJ, torch::full_like(mJ, -1e30f)));
        sBuf.copy_(torch::where(valid, SJ, torch::zeros_like(SJ)));
        }
    } else {
    cudaFuncSetAttribute(score_part, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    score_part<<<dim3(ntiles, Hkv, JS), 256, sh, st>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kbar.data_ptr<at::Half>()),
        rbar.data_ptr<float>(), rmu.data_ptr<float>(),
        reinterpret_cast<const __half*>(dev.data_ptr<at::Half>()),
        Sq, (int)T, H, Hkv, D, (int)B, (int)TI, (int)PROBE, (float)scale, (float)beta, (float)gamma, NB, JS, Rc,
        mBuf.data_ptr<float>(), sBuf.data_ptr<float>(), REGQ);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const char* st_e = getenv("FA2SM70_SEL_STRAT");
    const int STRAT = (st_e && *st_e) ? atoi(st_e) : 0;
    const char* pr_e = getenv("FA2SM70_SEL_STRAT_PAIR");
    const int PAIR = (pr_e && *pr_e) ? atoi(pr_e) : 0;
    const char* ms_e = getenv("FA2SM70_SPARSE_MASS");
    const float MASS = (ms_e && *ms_e) ? (float)atof(ms_e) : 0.f;
    auto stat = torch::zeros({ntiles, Hkv, 2}, torch::dtype(torch::kFloat32).device(Q.device()));
    select_csr<<<dim3(ntiles, Hkv), 256, 0, st>>>(
        Sq, (int)T, Hkv, (int)B, (int)TI, (float)alpha, (int)SINK, (int)WIN, (int)REC,
        mBuf.data_ptr<float>(), sBuf.data_ptr<float>(),
        cnt.data_ptr<int>(), idx.data_ptr<int>(), NB, STRAT, PAIR, stat.data_ptr<float>(), MASS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Разбросы отдаются НАРУЖУ: величину, по которой правит поправка, обязано быть можно
    // ИЗМЕРИТЬ, а не выводить из поведения. Читатели берут первые три поля по индексу,
    // поэтому четвёртое совместимо со старым расширением (см. бэкенд).
    return {cnt, idx, kbar, rbar, dev, stat, mBuf, sBuf};   // 6: масса {удержано, всего}; 7-8: сырые (m, S) по блокам -- для офлайн-разбора правил
}

// Отбор живёт в ТОМ ЖЕ расширении, что и внимание: бэкенд зовёт его через уже имеющийся
// `self._pre()`, отдельного модуля и отдельной сборки заводить незачем. Собственный
// PYBIND11_MODULE остаётся только для автономных проб (FA2_SPARSE_STANDALONE).
#ifdef FA2_SPARSE_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Умолчания ОБЯЗАТЕЛЬНЫ и здесь: без них автономная проба зовёт функцию позиционно,
    // а всякий новый довод молча превращается в "incompatible function arguments".
    m.def("sparse_select", &fa2sm70_sparse_select,
          "FA2/SM70: отбор блоков ключей по средним ключам (FlashPrefill V2 §3.1), -> (cnt, idx, kbar)",
          pybind11::arg("Q"), pybind11::arg("Kc"), pybind11::arg("Kst"), pybind11::arg("BlockTable"),
          pybind11::arg("T"), pybind11::arg("scale"), pybind11::arg("alpha"),
          pybind11::arg("B") = 128, pybind11::arg("TI") = 128, pybind11::arg("PROBE") = 16,
          pybind11::arg("SINK") = 2, pybind11::arg("WIN") = 8, pybind11::arg("REC") = 2,
          pybind11::arg("kbarPool") = c10::nullopt, pybind11::arg("beta") = 0.0,
          pybind11::arg("gamma") = 0.0);
}
#endif
