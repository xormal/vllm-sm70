// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Я1: ВСЁ ТЕЛО MLP ОДНИМ ЗАПУСКОМ -- норма + gate_up + silu*mul + down.
//
// ЗАЧЕМ. Замер снятием фазы (reports/EV_decode_phase_falsifier.md): шаг декода 39.54 мс при поле
// чтения весов 20.0 мс, то есть 55 % шага -- НЕ поток весов. Поток весов ОСТАНАВЛИВАЕТСЯ ~877 раз
// за шаг: каждое ядро осушает конвейер памяти, следующее разгоняет его заново. Байтов в промежутке
// почти нет (u -- десятки КБ), но каждая граница -- точка, где чтение веса ПРЕКРАЩАЕТСЯ.
//
// ЧТО ЗДЕСЬ НОВОГО ПРОТИВ fused_norm_gemv.cu. Там слито «норма + gate_up + silu» и явно записано:
// «свернуть сюда же down нельзя: ему нужен ВЕСЬ промежуточный вектор, это редукция по блокам».
// Именно эта граница здесь и снимается -- кооперативным запуском и барьером по сетке. На слой
// становится ОДИН пуск вместо трёх, а промежуточный вектор I не выходит в HBM вовсе.
//
// ПОЧЕМУ ПОСТОЯННАЯ СЕТКА, А НЕ «сколько надо». Кооперативный запуск требует, чтобы ВСЯ сетка была
// одновременно резидентна, иначе барьер -- взаимная блокировка. Поэтому число блоков берётся из
// cudaOccupancyMaxActiveBlocksPerMultiprocessor, а работа раздаётся шагом сетки. Это же и есть
// скелет Я3: постоянные блоки, у которых граница слоя станет итерацией цикла.
//
// ЗАПОЛНЕНИЕ. Столбцов фиксировано геометрией выхода, поэтому число потоков в блоке параллелизма НЕ
// добавляет (замерено: лестница 0.42->3.40 волны дала 2.4 %). Единственная ось, которая добавляет
// байт в полёте, -- РАСЩЕПЛЕНИЕ ОСИ РЕДУКЦИИ: k-срезы разными блоками. Поэтому обе фазы расщеплены
// по k, а сведение частичных сумм делается ЗДЕСЬ ЖЕ между барьерами, без отдельного ядра.
//
// АРИФМЕТИКА СКОПИРОВАНА БЕЗ ИЗМЕНЕНИЙ из отгруженного тела (та же группировка по G, тот же порядок
// накопления, та же свёртка (q-z)*s). Проверяется СХЕМА, а не формат: иначе при расхождении не
// отличить ошибку укладки от ошибки квантования.

#include <torch/extension.h>
#include <cstdlib>
#include <unordered_map>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ПОТОКОВОЕ ЧТЕНИЕ ВЕСА. Вес не переиспользуется ни разу за токен, поэтому держать его в кэшах
// незачем: `.cs` не вытесняет ими плитку активации и таблицы. Гейт -- чтобы ход был ЗАМЕРЕН, а
// не принят на веру.
#ifdef FA2SM70_ПОТОК_ВЕСА
#define ЧТЕНИЕ_ВЕСА4(p) __ldcs(reinterpret_cast<const int4*>(p))
#define ЧТЕНИЕ_ВЕСА2(p) __ldcs(reinterpret_cast<const int2*>(p))
#else
// ОБЕЩАНИЕ ВЫРАВНИВАНИЯ ОПРОВЕРГНУТО: 128-разрядные чтения веса компилятор эмитит и БЕЗ него
// (146 LDG.E.128 в теле), время не изменилось (20.16 против 20.05), а кадр стека вырос 32->56.
// Найденные в SASS LDG.E.64 относятся не к весам. Оставлено обычное чтение.
#define ЧТЕНИЕ_ВЕСА4(p) (*reinterpret_cast<const int4*>(p))
#define ЧТЕНИЕ_ВЕСА2(p) (*reinterpret_cast<const int2*>(p))
#endif
#ifndef FA2SM70_K_UNROLL
#define FA2SM70_K_UNROLL 4
#endif
// Прагма НЕ проходит макроподстановку -- нужен _Pragma со стрингификацией через два уровня.
#define _СТР1(x) #x
#define _СТР(x) _СТР1(x)
#define РАЗВЕРНУТЬ_K _Pragma(_СТР(unroll FA2SM70_K_UNROLL))
#define XT2_ 512    // плитка активации в разделяемой; кратна группе 128

// ЧИТАТЕЛЬ ПОСТРАНИЧНОГО СКЛАДА. Раскладка и разворот взяты ИЗ ОТГРУЖЕННОГО `gather_paged_kv_kernel`
// (`split_decode.cu`), а не по памяти: пул [блок, block_size, Hkv, d], адрес
// `block_table[t/bs]*blk_stride + ((t%bs)*Hkv + h)*d`, байтовый формат e4m3 со скаляром.
// Смысл хода: сейчас плита KV СНАЧАЛА собирается в плотный fp16 отдельным ядром, и только потом её
// читает внимание. Здесь она читается ПРЯМО из склада -- сборки нет вовсе, и промежуточная плита
// (44 МБ на слой при контексте 43К) в HBM не появляется.
__device__ __forceinline__ float e4m3_fast(uint8_t b) {
    const unsigned s = (unsigned)(b & 0x80u) << 24;
    const unsigned e = (b >> 3) & 0x0Fu, m = b & 0x07u;
    if (e == 0u) {                       // денормаль e4m3: 2^-9 * m
        return (b & 0x80u ? -1.f : 1.f) * (float)m * 0.001953125f;
    }
    const unsigned bits = s | ((e + 120u) << 23) | (m << 20);
    return __int_as_float((int)bits);
}
// СВЁРТКА В e4m3 (E4M3FN): знак 1, порядок 4 со смещением 7, мантисса 3. S.1111.111 -- NaN,
// поэтому наибольшее конечное -- 0x7E = 448. Денормали: 2^-6 * (m/8), наименьшая 2^-9.
// Округление к ближайшему. Сверено против torch.float8_e4m3fn -- не "по смыслу", а побайтно.
__device__ __forceinline__ uint8_t f2e4m3(float x) {
    const uint8_t sgn = (x < 0.f || (x == 0.f && signbit(x))) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (!(a > 0.f)) return sgn;                       // ноль и NaN входа -> ноль
    if (a >= 448.f) return (uint8_t)(sgn | 0x7Eu);    // насыщение, не NaN
    if (a < 0.0009765625f) return sgn;                // меньше половины наименьшей денормали
    int e; const float m = frexpf(a, &e);             // a = m * 2^e, m в [0.5, 1)
    const int E = e - 1;
    if (E >= -6) {
        int ef = E + 7;
        int mant = __float2int_rn((m * 2.f - 1.f) * 8.f);
        if (mant == 8) { mant = 0; ++ef; if (ef > 15) return (uint8_t)(sgn | 0x7Eu); }
        if (ef == 15 && mant == 7) mant = 6;          // 0x7F -- NaN, ближайшее конечное 0x7E
        return (uint8_t)(sgn | (ef << 3) | mant);
    }
    int mant = __float2int_rn(a * 512.f);             // денормаль: a / 2^-9
    if (mant > 7) return (uint8_t)(sgn | (1 << 3));   // выросла до наименьшей нормальной
    return (uint8_t)(sgn | mant);
}

__device__ __forceinline__ float paged_at(const uint8_t* __restrict__ pool,
                                          const int* __restrict__ bt, int t, int h, int i,
                                          int Hkv, int d, long blk_stride, int bs, float sc) {
    const int lb = t / bs;
    const long src = (long)bt[lb] * blk_stride + ((long)(t - lb * bs) * Hkv + h) * d + i;
    return e4m3_fast(pool[src]) * sc;
}

// Распаковка пары __half из 32-разрядного слова БЕЗ обращения к памяти: адрес локальной переменной
// (&float4) утаскивает её в кадр стека, а кадр на этих телах мы держим нулевым.
__device__ __forceinline__ float h_lo(unsigned u) {
    return __half2float(__ushort_as_half((unsigned short)(u & 0xffffu)));
}
__device__ __forceinline__ float h_hi(unsigned u) {
    return __half2float(__ushort_as_half((unsigned short)(u >> 16)));
}


// ЦЕНА СЕТОЧНОГО БАРЬЕРА. Их 820 на токен (47 слоёв GDN x10 + 16 внимания x14 + 63x2 между
// слоями) -- это ФАЗА, и её нельзя оценить рассуждением. Под гейтом барьер вырождается в
// блочный: ответ заведомо неверен, читается ТОЛЬКО время. По умолчанию гейт выключен.
#ifdef FA2SM70_БЕЗ_БАРЬЕРОВ
#define ГБАР(g) __syncthreads()
#else
#define ГБАР(g) (g).sync()
#endif


// РАЗВОРОТ e4m3 ЧЕТВЁРКАМИ, 6 инструкций на 4 байта (перенесено из fa2_src/.../volta_prims.h,
// §1e). Поэлементный `e4m3_fast` стоит ветку на денормаль плюс сборку разрядов -- в фазе обхода
// ключей это 16 разворотов на ключ на лэйн, то есть самая частая операция ядра.
// Значение выходит ДЕЛЁННЫМ НА 256: экспонента кладётся в поле fp16 БЕЗ пересмещения, а смещения
// разные (15 против 7), то есть множитель 2^-8. Поэтому 256 складывается в масштаб склада ОДИН
// раз на задание, а не на элемент. (Ошибиться знаком степени здесь легко, и ошибка тихая:
// вывод уходит в 65536 раз вниз, а relL2 против эталона показывает ровно 1.0.)
// ОГОВОРКА, которую нельзя прятать: денормали e4m3 (|v|<=0.0136) становятся денормалями fp16 и
// при -ftz могут обнулиться. Проверяется A/B против поэлементного пути, а не рассуждением.
__device__ __forceinline__ void expand_e4m3x4(uint32_t w, uint32_t& f01, uint32_t& f23) {
    uint32_t p0, p1;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p0) : "r"(w), "r"(0u), "n"(0x9180));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p1) : "r"(w), "r"(0u), "n"(0xB3A2));
    f01 = (p0 << 7) & 0xBF80BF80u;
    f23 = (p1 << 7) & 0xBF80BF80u;
}


// ДЕКВАНТ GPTQ-БАЙТА ПЕРЕСТАНОВКОЙ (volta_prims §1a/§1b). Наивная запись стоит на КАЖДЫЙ байт
// сдвиг + маску + I2F; перестановка даёт четыре fp16 ДВУМЯ __byte_perm. Значение выходит
// СМЕЩЁННЫМ на 1024 (старший байт fp16 = 0x64 -> 2^10*(1+b/1024) = 1024+b), а смещение
// сворачивается в ТУ ЖЕ ранг-1 поправку, что уже стоит в эпилоге (q - z*xa): она становится
// (z+1024)*xa и стоит НОЛЬ. В доке замерено: 745 -> 787 ГБ/с у gptq-GEMV, 470 -> 700 у exllama.
// ПО УМОЛЧАНИЮ ВКЛЮЧЕНО (замер: 20.45 -> 20.05 мс, 594 -> 606 ГБ/с). Прежний поэлементный путь
// остаётся под -DFA2SM70_ДЕКВАНТ_НАИВНЫЙ как измерительная ступень.
#if !defined(FA2SM70_ДЕКВАНТ_НАИВНЫЙ) && !defined(FA2SM70_ДЕКВАНТ_ПЕРЕСТАНОВКОЙ)
#define FA2SM70_ДЕКВАНТ_ПЕРЕСТАНОВКОЙ 1
#endif
// НОЛЬ GPTQ ХРАНИТСЯ СМЕЩЁННЫМ НА ЕДИНИЦУ (exllama: dequant = (b - (z+1))*s).
// Замерено сверкой плиты с проекцией движка: без единицы z даёт relL2 1.50e-02, с ней 1.04e-03,
// а косинус на 63 слоях идёт 0.99041 -> 0.99997. Ошибка на единицу тут не падает и не шумит --
// она даёт СИСТЕМАТИЧЕСКИЙ сдвиг s*xa на группу, то есть ровно ту «почти правильность»,
// которую видно только сверкой с чужим ядром.
#ifndef FA2SM70_ZERO_PLUS
#define FA2SM70_ZERO_PLUS 1.f
#endif
#define FA2SM70_МАГИЯ4 0x64646464u
__device__ __forceinline__ void expand_u8x4_biased(uint32_t w, uint32_t& f01, uint32_t& f23) {
    f01 = __byte_perm(w, FA2SM70_МАГИЯ4, 0x4140);   // [b0, 0x64, b1, 0x64]
    f23 = __byte_perm(w, FA2SM70_МАГИЯ4, 0x4342);
}

__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }

// ---------------------------------------------------------------- норма своими силами, как в Ш1
template <int THREADS>
__device__ __forceinline__ float rms_scale2(const __half* __restrict__ x, int K, float eps) {
    __shared__ float red[THREADS / 32];
    float s = 0.f;
    for (int k = threadIdx.x; k < K; k += THREADS) {
        const float v = __half2float(x[k]);
        s = fmaf(v, v, s);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < THREADS / 32; ++i) t += red[i];
        red[0] = rsqrtf(t / (float)K + eps);
    }
    __syncthreads();
    return red[0];
}

// ---------------------------------------------------------------- один срез GEMV по k, с накоплением
// Общая для обеих фаз внутренность: столбцы [n0, n0+CPT) веса (qw, qz, sc), k из [kbeg, kend).
// `sxsrc` -- уже подготовленная активация в разделяемой (нормированная или промежуточная).
// ================================ M ТОКЕНОВ НА ОДНУ ЗАГРУЗКУ ВЕСА =================================
// ЗАЧЕМ. При M=1 вес читается ради ОДНОГО токена, и трафик на токен равен всему весу -- это 63%
// шага и оно стоит на потолке полосы, то есть иначе не двигается ВООБЩЕ. При M токенов те же байты
// обслуживают M выходов: трафик на токен падает в M раз. Тот же ход -- единственный способ пустить
// ПРЕФИЛЛ через слитую цепь: куски по M токенов идут сквозь 63 слоя без промежуточных плит.
//
// Цена -- регистры: накопитель становится [CPT][M]. Активация лежит в разделяемой как [M][len].
template <int THREADS, int CPT, int M>
__device__ __forceinline__ void gemm_slice(const float* __restrict__ sx, int sx_шаг,
                                           int gk_base, int len,
                                           const int* __restrict__ qw, const int* __restrict__ qz,
                                           const __half* __restrict__ sc,
                                           int n0, int I, int G, float* __restrict__ acc) {
    for (int gk = 0; gk < len; gk += G) {
        const int g = (gk_base + gk) / G;
        float s[CPT], z[CPT], q[CPT * M];
        const unsigned wz = (unsigned)qz[(long)g * (I >> 2) + (n0 >> 2)];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) {
            s[c] = __half2float(sc[(long)g * I + n0 + c]);
            z[c] = (float)((wz >> (((n0 + c) & 3) * 8)) & 0xFF) + FA2SM70_ZERO_PLUS;
            #pragma unroll
            for (int m = 0; m < M; ++m) q[c * M + m] = 0.f;
        }
        const int glen = min(G, len - gk);
        float xa[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) xa[m] = 0.f;
        for (int k = 0; k < glen; k += 4) {
            const long wi = (long)((gk_base + gk + k) >> 2) * I + n0;
            // ВЕС ЧИТАЕТСЯ ОДИН РАЗ НА ВСЕ M ТОКЕНОВ -- в этом весь ход.
            unsigned wv[CPT];
            if (CPT == 4) {
                const int4 w4 = ЧТЕНИЕ_ВЕСА4(&qw[wi]);
                wv[0] = w4.x; wv[1] = w4.y; wv[2] = w4.z; wv[3] = w4.w;
            } else if (CPT == 2) {
                const int2 w2 = ЧТЕНИЕ_ВЕСА2(&qw[wi]);
                wv[0] = w2.x; wv[1] = w2.y;
            } else {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) wv[c] = (unsigned)qw[wi + c];
            }
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                uint32_t f01, f23; expand_u8x4_biased(wv[c], f01, f23);
                const float w0 = h_lo(f01), w1 = h_hi(f01), w2v = h_lo(f23), w3 = h_hi(f23);
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const float4 a = *reinterpret_cast<const float4*>(&sx[(long)m * sx_шаг + gk + k]);
                    if (c == 0) xa[m] += a.x + a.y + a.z + a.w;
                    float* qq = &q[c * M + m];
                    *qq = fmaf(a.x, w0, *qq);
                    *qq = fmaf(a.y, w1, *qq);
                    *qq = fmaf(a.z, w2v, *qq);
                    *qq = fmaf(a.w, w3, *qq);
                }
            }
        }
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            #pragma unroll
            for (int m = 0; m < M; ++m)
                acc[c * M + m] += s[c] * (q[c * M + m] - (z[c] + 1024.f) * xa[m]);
    }
}

template <int THREADS, int CPT>
__device__ __forceinline__ void gemv_slice(const float* __restrict__ sx, int gk_base, int len,
                                           const int* __restrict__ qw, const int* __restrict__ qz,
                                           const __half* __restrict__ sc,
                                           int n0, int I, int G, float* __restrict__ acc) {
    for (int gk = 0; gk < len; gk += G) {
        const int g = (gk_base + gk) / G;
        float s[CPT], z[CPT], q[CPT];
        // ОДНО слово нулей на всю пачку: при CPT=2 столбцы n0 и n0+1 ВСЕГДА в одном слове (n0 чётно),
        // при CPT=4 -- тем более. Это наш же приём, раньше он стоял только под CPT=4.
        const unsigned wz = (unsigned)qz[(long)g * (I >> 2) + (n0 >> 2)];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) {
            s[c] = __half2float(sc[(long)g * I + n0 + c]);
            // СОГЛАШЕНИЕ О НУЛЕ. У exllama/GPTQ ноль хранится СМЕЩЁННЫМ на единицу
            // (dequant = (b - (z+1))*s). Расхождение на единицу даёт систематический сдвиг
            // s*xa на группу -- ровно тот порядок (1.5e-02), что показала сверка плиты.
            z[c] = (float)((wz >> (((n0 + c) & 3) * 8)) & 0xFF) + FA2SM70_ZERO_PLUS;
            q[c] = 0.f;
        }
        const int glen = min(G, len - gk);
        float xa = 0.f;
        РАЗВЕРНУТЬ_K
        for (int k = 0; k < glen; k += 4) {
            const long wi = (long)((gk_base + gk + k) >> 2) * I + n0;
            const float4 a = *reinterpret_cast<const float4*>(&sx[gk + k]);
            xa += a.x + a.y + a.z + a.w;
            unsigned wv[CPT];
            if (CPT == 4) {
                const int4 w4 = ЧТЕНИЕ_ВЕСА4(&qw[wi]);
                wv[0] = w4.x; wv[1] = w4.y; wv[2] = w4.z; wv[3] = w4.w;
            } else if (CPT == 2) {
                const int2 w2 = ЧТЕНИЕ_ВЕСА2(&qw[wi]);
                wv[0] = w2.x; wv[1] = w2.y;
            } else {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) wv[c] = (unsigned)qw[wi + c];
            }
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
#ifdef FA2SM70_ДЕКВАНТ_ПЕРЕСТАНОВКОЙ
                uint32_t f01, f23; expand_u8x4_biased(wv[c], f01, f23);
                q[c] = fmaf(a.x, h_lo(f01), q[c]);
                q[c] = fmaf(a.y, h_hi(f01), q[c]);
                q[c] = fmaf(a.z, h_lo(f23), q[c]);
                q[c] = fmaf(a.w, h_hi(f23), q[c]);
#else
                const unsigned w = wv[c];
                q[c] = fmaf(a.x, (float)( w        & 0xFF), q[c]);
                q[c] = fmaf(a.y, (float)((w >>  8) & 0xFF), q[c]);
                q[c] = fmaf(a.z, (float)((w >> 16) & 0xFF), q[c]);
                q[c] = fmaf(a.w, (float)((w >> 24) & 0xFF), q[c]);
#endif
            }
        }
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
#ifdef FA2SM70_ДЕКВАНТ_ПЕРЕСТАНОВКОЙ
            acc[c] += s[c] * (q[c] - (z[c] + 1024.f) * xa);   // 1024 -- смещение перестановки
#else
            acc[c] += s[c] * (q[c] - z[c] * xa);
#endif
    }
}

// ПАРА gate/up ОДНИМ ПРОХОДОМ. Разнеся их по двум вызовам помощника, я удвоил и `xa` (она одна на
// группу, а не на срез веса), и ЧТЕНИЕ `sx` из разделяемой. В отгруженном теле это было сделано
// правильно с самого начала -- регрессия появилась от моего же «красивого» рефакторинга.
template <int THREADS, int CPT>
__device__ __forceinline__ void gemv_slice_pair(const float* __restrict__ sx, int gk_base, int len,
                                                const int* __restrict__ qwg, const int* __restrict__ qzg,
                                                const __half* __restrict__ scg,
                                                const int* __restrict__ qwu, const int* __restrict__ qzu,
                                                const __half* __restrict__ scu,
                                                int n0g, int n0u, int I, int G,
                                                float* __restrict__ ag, float* __restrict__ au) {
    for (int gk = 0; gk < len; gk += G) {
        const int g = (gk_base + gk) / G;
        float sg[CPT], su[CPT], zg[CPT], zu[CPT], qg[CPT], qu[CPT];
        const unsigned wzg = (unsigned)qzg[(long)g * (I >> 2) + (n0g >> 2)];
        const unsigned wzu = (unsigned)qzu[(long)g * (I >> 2) + (n0u >> 2)];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) {
            sg[c] = __half2float(scg[(long)g * I + n0g + c]);
            su[c] = __half2float(scu[(long)g * I + n0u + c]);
            // СВОЙ сдвиг на каждую половину: при слитой плите база up отличается на смещение,
            // и общий сдвиг верен лишь пока смещение кратно 4. Не полагаться на это молча.
            zg[c] = (float)((wzg >> (((n0g + c) & 3) * 8)) & 0xFF) + FA2SM70_ZERO_PLUS;
            zu[c] = (float)((wzu >> (((n0u + c) & 3) * 8)) & 0xFF) + FA2SM70_ZERO_PLUS;
            qg[c] = 0.f; qu[c] = 0.f;
        }
        const int glen = min(G, len - gk);
        float xa = 0.f;                      // ОДНА на группу -- считается один раз, а не дважды
        РАЗВЕРНУТЬ_K
        for (int k = 0; k < glen; k += 4) {
            const long wrow = (long)((gk_base + gk + k) >> 2) * I;
            const long wi = wrow + n0g, wiu = wrow + n0u;   // СВОЙ адрес у up
            const float4 a = *reinterpret_cast<const float4*>(&sx[gk + k]);   // ОДНО чтение на пару
            xa += a.x + a.y + a.z + a.w;
            // ШИРИНА ОБРАЩЕНИЯ (наш приём Ш3): CPT слов веса лежат ПОДРЯД, значит берутся ОДНОЙ
            // загрузкой, а не CPT отдельными. n0 кратно CPT, I кратно 4 -- выравнивание есть.
            unsigned wgv[CPT], wuv[CPT];
            if (CPT == 4) {
                const int4 g4 = ЧТЕНИЕ_ВЕСА4(&qwg[wi]);
                const int4 u4 = ЧТЕНИЕ_ВЕСА4(&qwu[wiu]);
                wgv[0] = g4.x; wgv[1] = g4.y; wgv[2] = g4.z; wgv[3] = g4.w;
                wuv[0] = u4.x; wuv[1] = u4.y; wuv[2] = u4.z; wuv[3] = u4.w;
            } else if (CPT == 2) {
                const int2 g2 = *reinterpret_cast<const int2*>(&qwg[wi]);
                const int2 u2 = *reinterpret_cast<const int2*>(&qwu[wiu]);
                wgv[0] = g2.x; wgv[1] = g2.y;
                wuv[0] = u2.x; wuv[1] = u2.y;
            } else {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) { wgv[c] = (unsigned)qwg[wi + c]; wuv[c] = (unsigned)qwu[wiu + c]; }
            }
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
#ifdef FA2SM70_ДЕКВАНТ_ПЕРЕСТАНОВКОЙ
                uint32_t g01, g23, u01, u23;
                expand_u8x4_biased(wgv[c], g01, g23);
                expand_u8x4_biased(wuv[c], u01, u23);
                qg[c] = fmaf(a.x, h_lo(g01), qg[c]); qg[c] = fmaf(a.y, h_hi(g01), qg[c]);
                qg[c] = fmaf(a.z, h_lo(g23), qg[c]); qg[c] = fmaf(a.w, h_hi(g23), qg[c]);
                qu[c] = fmaf(a.x, h_lo(u01), qu[c]); qu[c] = fmaf(a.y, h_hi(u01), qu[c]);
                qu[c] = fmaf(a.z, h_lo(u23), qu[c]); qu[c] = fmaf(a.w, h_hi(u23), qu[c]);
#else
                const unsigned wg = wgv[c], wu = wuv[c];
                qg[c] = fmaf(a.x, (float)( wg        & 0xFF), qg[c]);
                qg[c] = fmaf(a.y, (float)((wg >>  8) & 0xFF), qg[c]);
                qg[c] = fmaf(a.z, (float)((wg >> 16) & 0xFF), qg[c]);
                qg[c] = fmaf(a.w, (float)((wg >> 24) & 0xFF), qg[c]);
                qu[c] = fmaf(a.x, (float)( wu        & 0xFF), qu[c]);
                qu[c] = fmaf(a.y, (float)((wu >>  8) & 0xFF), qu[c]);
                qu[c] = fmaf(a.z, (float)((wu >> 16) & 0xFF), qu[c]);
                qu[c] = fmaf(a.w, (float)((wu >> 24) & 0xFF), qu[c]);
#endif
            }
        }
        #pragma unroll
        for (int c = 0; c < CPT; ++c) {
#ifdef FA2SM70_ДЕКВАНТ_ПЕРЕСТАНОВКОЙ
            ag[c] += sg[c] * (qg[c] - (zg[c] + 1024.f) * xa);
            au[c] += su[c] * (qu[c] - (zu[c] + 1024.f) * xa);
#else
            ag[c] += sg[c] * (qg[c] - zg[c] * xa);
            au[c] += su[c] * (qu[c] - zu[c] * xa);
#endif
        }
    }
}

// ---------------------------------------------------------------- ВСЁ ТЕЛО ОДНИМ ЯДРОМ
template <int THREADS, int CPT>
__global__ void mlp_body_kernel(const __half* __restrict__ x, const __half* __restrict__ rw,
                                const int* __restrict__ qw_g, const int* __restrict__ qz_g,
                                const __half* __restrict__ sc_g,
                                const int* __restrict__ qw_u, const int* __restrict__ qz_u,
                                const __half* __restrict__ sc_u,
                                const int* __restrict__ qw_d, const int* __restrict__ qz_d,
                                const __half* __restrict__ sc_d,
                                float* __restrict__ apart,   // [ksA, I]   частичные фазы A
                                __half* __restrict__ acted,  // [I]        silu(u1)*u2
                                float* __restrict__ dpart,   // [ksD, N]   частичные фазы C
                                __half* __restrict__ out,    // [N]
                                float* __restrict__ nsq,     // [2] сумма квадратов входа, пинг-понг
                                int I, int K, int N, int G, float eps,
                                int colsA, int ksA, int gpsA,
                                int colsD, int ksD, int gpsD,
                                int nlayers, int wstride,
                                // СЛИТЫЙ gate_up_proj: одна плита, две базы столбца. upoff=0 и
                                // Ntot=0 означают "веса раздельные" -- прежнее поведение.
                                int upoff, int Ntot,
                                // ОБМЕН ВНУТРИ ЯДРА. Выход `down` -- частичная сумма своей половины
                                // столбцов; сосед считает вторую. Сейчас это отдельный NCCL-пуск
                                // ценой 70.5 мкс на 10 КБ; своими руками рандеву 2.17, обмен 4.97.
                                // Здесь он делается ТЕМИ ЖЕ резидентными блоками: ни запуска, ни
                                // осушения конвейера, ни перепланирования сетки.
                                __half* __restrict__ peer_inbox,   // почтовый ящик СОСЕДА
                                const __half* __restrict__ my_inbox,
                                volatile int* __restrict__ my_flag,
                                volatile int* __restrict__ peer_flag,
                                int seq0) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float sx[XT2_];

    const int nb = gridDim.x;                    // постоянная сетка
    const int b  = blockIdx.x;

    // ЦИКЛ ПО СЛОЯМ ВНУТРИ ОДНОГО ПУСКА. Граница слоя здесь -- итерация, а не запуск: конвейер
    // памяти не осушается, предвыборка следующего веса идёт СКВОЗЬ неё. Ради этого всё и делалось;
    // в однослойном стенде барьеры платятся, а прятать их нечем.
    for (int L = 0; L < nlayers; ++L) {
        const __half* xl = (L == 0) ? x : (const __half*)out;
        const int*    Wg = qw_g + (long)L * wstride * I;
        const int*    Zg = qz_g + (long)L * (K / G) * (I >> 2);
        const __half* Sg = sc_g + (long)L * (K / G) * I;
        const int*    Wu = qw_u + (long)L * wstride * I;
        const int*    Zu = qz_u + (long)L * (K / G) * (I >> 2);
        const __half* Su = sc_u + (long)L * (K / G) * I;
        const int*    Wd = qw_d + (long)L * (I / 4) * N;
        const int*    Zd = qz_d + (long)L * (I / G) * (N >> 2);
        const __half* Sd = sc_d + (long)L * (I / G) * N;
        // НОРМА НЕ ПЕРЕСЧИТЫВАЕТСЯ 640 БЛОКАМИ НА КАЖДОМ СЛОЕ. Сумма квадратов входа слоя L+1
        // уже прошла через регистры в конце слоя L (там писался out) -- значит она копится ТАМ,
        // а здесь только читается. Новый барьер не нужен: между записью и чтением уже стоит
        // барьер конца слоя. Слой 0 -- исключение, его вход пришёл снаружи, и он платит один раз.
        float rs;
        if (L == 0) {
            rs = rms_scale2<THREADS>(xl, K, eps);
        } else {
            rs = rsqrtf(nsq[L & 1] / (float)K + eps);
        }
        if (b == 0 && threadIdx.x == 0) nsq[(L + 1) & 1] = 0.f;   // слот следующего слоя

    // ---- ФАЗА A: норма + gate_up + silu*mul -> частичные суммы
    // Разбор блока: (плитка столбцов, срез k). Оба индекса идут ШАГОМ СЕТКИ, поэтому сетка может
    // быть любой -- лишь бы резидентной.
    for (int job = b; job < colsA * ksA; job += nb) {
        const int ci = job % colsA, ki = job / colsA;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gpsA * G, kend = min(K, kbeg + gpsA * G);
        float ag[CPT], au[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) { ag[c] = 0.f; au[c] = 0.f; }
        const bool live = (n0 + CPT <= I);
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS)
                sx[t] = __half2float(xl[k0 + t]) * rs * (1.f + __half2float(rw[k0 + t]));
            __syncthreads();
            if (!live) continue;
            gemv_slice_pair<THREADS, CPT>(sx, k0, len, Wg, Zg, Sg, Wu, Zu, Su,
                                          n0, n0 + upoff, Ntot ? Ntot : I, G, ag, au);
        }
        if (live) {
            // ПРОСТАЯ ЗАПИСЬ, А НЕ АТОМАРНАЯ. Работа (плитка столбцов, срез k) владеет СВОИМИ
            // адресами -- сложения между блоками здесь нет по построению. Атомарная тут была моей
            // перестраховкой и стоила и команд, и обнуления буфера на каждый вызов.
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                apart[(long)ki * 2 * I + n0 + c]     = ag[c];
                apart[(long)ki * 2 * I + I + n0 + c] = au[c];
            }
        }
    }
    ГБАР(grid);

    // ---- сведение частичных и silu*mul: ЗДЕСЬ ЖЕ, отдельного ядра нет
    for (long i = (long)b * THREADS + threadIdx.x; i < I; i += (long)nb * THREADS) {
        float g = 0.f, u = 0.f;
        for (int ki = 0; ki < ksA; ++ki) { g += apart[(long)ki * 2 * I + i]; u += apart[(long)ki * 2 * I + I + i]; }
        acted[i] = __float2half(silu_f(g) * u);
    }
    ГБАР(grid);

    // ---- ФАЗА C: down. Активация -- уже готовый вектор I, нормировать НЕ надо.
    for (int job = b; job < colsD * ksD; job += nb) {
        const int ci = job % colsD, ki = job / colsD;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gpsD * G, kend = min(I, kbeg + gpsD * G);
        float ad[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) ad[c] = 0.f;
        const bool live = (n0 + CPT <= N);
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS) sx[t] = __half2float(acted[k0 + t]);
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPT>(sx, k0, len, Wd, Zd, Sd, n0, N, G, ad);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) dpart[(long)ki * N + n0 + c] = ad[c];
        }
    }
    ГБАР(grid);

    float ps = 0.f;
    for (long i = (long)b * THREADS + threadIdx.x; i < N; i += (long)nb * THREADS) {
        float s = 0.f;
        for (int ki = 0; ki < ksD; ++ki) s += dpart[(long)ki * N + i];
        if (peer_inbox != nullptr) peer_inbox[i] = __float2half(s);   // сразу соседу, без буфера
        const __half hs = __float2half(s);
        out[i] = hs;
        // Квадрат берём от ОКРУГЛЁННОГО значения: следующий слой читает именно его. Иначе норма
        // считается от чуть другого вектора, чем тот, который нормируется.
        const float v = __half2float(hs);
        ps = fmaf(v, v, ps);
    }
    if (peer_inbox != nullptr) ps = 0.f;   // при обмене норма берётся от СВЕДЁННОГО, ниже
    if (peer_inbox != nullptr) {
        // РАНДЕВУ ТЕМИ ЖЕ БЛОКАМИ. Барьер по сетке гарантирует, что вся моя половина уже уехала;
        // после этого один поток ставит флаг, остальные ждут флага соседа. Спин ОГРАНИЧЕН: зависшее
        // ядро на карте недопустимо, лучше заведомо неверный ответ, чем повисшая карта.
        __threadfence_system();
        ГБАР(grid);
        if (b == 0 && threadIdx.x == 0) { *peer_flag = seq0 + L + 1; __threadfence_system(); }
        if (threadIdx.x == 0) {
            unsigned long long spin = 0;
            while (*my_flag < seq0 + L + 1) { if (++spin > 400000000ULL) break; }
        }
        __syncthreads();
        ГБАР(grid);
        for (long i = (long)b * THREADS + threadIdx.x; i < N; i += (long)nb * THREADS) {
            const float v = __half2float(out[i]) + __half2float(my_inbox[i]);
            const __half hv = __float2half(v);
            out[i] = hv;
            ps = fmaf(__half2float(hv), __half2float(hv), ps);   // норма считается от СВЕДЁННОГО
        }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) ps += __shfl_down_sync(0xffffffff, ps, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&nsq[(L + 1) & 1], ps);
    ГБАР(grid);     // выход слоя становится входом следующего -- барьер обязателен
    }
}

// ---------------------------------------------------------------- точка входа
torch::Tensor mlp_body(torch::Tensor x, torch::Tensor rms_weight, double eps,
                       torch::Tensor qw_g, torch::Tensor qz_g, torch::Tensor sc_g,
                       torch::Tensor qw_u, torch::Tensor qz_u, torch::Tensor sc_u,
                       torch::Tensor qw_d, torch::Tensor qz_d, torch::Tensor sc_d,
                       int64_t group_size, int64_t nlayers,
                       c10::optional<torch::Tensor> peer_inbox,
                       c10::optional<torch::Tensor> my_inbox,
                       c10::optional<torch::Tensor> my_flag,
                       c10::optional<torch::Tensor> peer_flag,
                       int64_t seq0, int64_t up_col_offset, int64_t total_width) {
    // ЗОНД БЫЛ КРИВОЙ. Здесь на КАЖДЫЙ вызов стояли: `.contiguous()` девяти тензоров, три
    // выделения через распределитель и запрос занятости у драйвера. Отсчёт (gptq_gemm) не делает
    // ничего из этого, и сравнивались не тела, а обвязки. Веса в инференсе постоянны и уже плотные,
    // рабочая область постоянна -- значит и то и другое считается ОДИН раз, а не на вызов.
    TORCH_CHECK(x.is_cuda(), "mlp_body: нужны тензоры на карте");
    TORCH_CHECK(x.is_contiguous() && qw_g.is_contiguous() && qw_u.is_contiguous() &&
                qw_d.is_contiguous(), "mlp_body: ждём плотные тензоры (копию тут не делаем)");
    const auto& xc = x; const auto& rwc = rms_weight;
    const auto& Wg = qw_g; const auto& Zg = qz_g; const auto& Sg = sc_g;
    const auto& Wu = qw_u; const auto& Zu = qz_u; const auto& Su = sc_u;
    const auto& Wd = qw_d; const auto& Zd = qz_d; const auto& Sd = sc_d;
    const int L = (int)nlayers;
    const int I = total_width ? (int)(total_width / 2) : (int)Wg.size(1);
    const int K = (int)(Wg.size(0) / L) * 4, G = (int)group_size;
    const int N = (int)Wd.size(1);
    TORCH_CHECK((total_width ? (int)Wu.size(1) == (int)total_width : (int)Wu.size(1) == I)
                && (int)(Wu.size(0) / L) * 4 == K, "mlp_body: gate/up разной формы");
    TORCH_CHECK((int)(Wd.size(0) / L) * 4 == I, "mlp_body: down не сходится с промежуточной осью");
    // Цепочка слоёв возможна только если выход подходит на вход: у боевой сети N == K == 5120.
    TORCH_CHECK(L == 1 || N == K, "mlp_body: цепочка слоёв требует N == K");
    TORCH_CHECK(xc.numel() == K, "mlp_body: собрано под M=1");
    TORCH_CHECK((I & 3) == 0 && (N & 3) == 0 && XT2_ % G == 0, "mlp_body: кратности");

    const at::cuda::CUDAGuard guard(xc.device());
    auto st = at::cuda::getCurrentCUDAStream();
    auto fo = torch::dtype(torch::kFloat32).device(xc.device());
    auto ho = torch::dtype(torch::kFloat16).device(xc.device());

    // Оси настройки, записанные у нас как значимые: столбцов на поток (ширина обращения) и
    // потоков в блоке (переупаковка тех же столбцов). Прогоняются ручками, а не угадываются.
    const char* ec = getenv("FA2SM70_MB_CPT");
    const char* et = getenv("FA2SM70_MB_T");
    const int CPT = ec ? atoi(ec) : 2;
    const int T   = et ? atoi(et) : 128;
    const int colsA = (I + T * CPT - 1) / (T * CPT);
    const int colsD = (N + T * CPT - 1) / (T * CPT);
    const int ngA = (K + G - 1) / G, ngD = (I + G - 1) / G;

    // Сетка ПОСТОЯННАЯ и обязана быть резидентной целиком -- иначе барьер повесит карту.
    static int perSM = 0, nsm = 0, cT = -1, cC = -1;   // одинаково на однотипных картах
    const void* kfn = (T == 256) ? ((CPT == 4) ? (const void*)mlp_body_kernel<256,4>
                                               : (const void*)mlp_body_kernel<256,2>)
                                 : ((CPT == 4) ? (const void*)mlp_body_kernel<128,4>
                                               : (const void*)mlp_body_kernel<128,2>);
    if (perSM == 0 || cT != T || cC != CPT) {   // свойство ТЕЛА и КАРТЫ, а не вызова
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &perSM, kfn, T, XT2_ * sizeof(float)));
        nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        cT = T; cC = CPT;
    }
    TORCH_CHECK(perSM > 0, "mlp_body: тело не резидентно, кооперативный запуск невозможен");
    const char* eg = getenv("FA2SM70_MLPBODY_BLOCKS");
    int nblk = eg ? atoi(eg) : perSM * nsm;
    nblk = max(1, min(nblk, perSM * nsm));

    // Срезы k подбираются так, чтобы работы хватило на всю постоянную сетку.
    int ksA = max(1, min(ngA, (nblk + colsA - 1) / colsA));
    int ksD = max(1, min(ngD, (nblk + colsD - 1) / colsD));
    const int gpsA = (ngA + ksA - 1) / ksA; ksA = (ngA + gpsA - 1) / gpsA;
    const int gpsD = (ngD + ksD - 1) / ksD; ksD = (ngD + gpsD - 1) / gpsD;

    // РАБОЧАЯ ОБЛАСТЬ ПОСТОЯННА: геометрия слоя в инференсе не меняется, значит выделение на вызов
    // -- чистая обвязка. Держим по одному набору на форму; при смене формы пересоздаём.
    // РАБОЧАЯ ОБЛАСТЬ -- ПО КАРТЕ, а не одна на процесс. При TP=2 обе карты живут в одном процессе,
    // и общий `static` отдал бы второй карте буфер первой: обращение через чужой указатель, которое
    // не падает, а молча считает не то. Ключ -- владение (номер карты), а не наличие.
    const int dev = (int)xc.device().index();
    static torch::Tensor apart_[8], dpart_[8], acted_[8], nsq_[8];
    static int cI_[8] = {-1,-1,-1,-1,-1,-1,-1,-1}, cN_[8] = {0}, cKA_[8] = {0}, cKD_[8] = {0};
    TORCH_CHECK(dev >= 0 && dev < 8, "mlp_body: неожиданный номер карты");
    torch::Tensor &apart = apart_[dev], &dpart = dpart_[dev], &acted = acted_[dev], &nsq = nsq_[dev];
    int &cI = cI_[dev], &cN = cN_[dev], &cKA = cKA_[dev], &cKD = cKD_[dev];
    if (cI != I || cN != N || cKA != ksA || cKD != ksD) {
        apart = torch::empty({ksA, 2 * I}, fo);   // обнулять НЕ нужно: пишем простой записью
        dpart = torch::empty({ksD, N}, fo);
        acted = torch::empty({I}, ho);
        nsq   = torch::zeros({2}, fo);
        cI = I; cN = N; cKA = ksA; cKD = ksD;
    }
    auto out = torch::empty({1, N}, ho);

    const __half* px = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
    const __half* pr = reinterpret_cast<const __half*>(rwc.data_ptr<at::Half>());
    const __half* psg = reinterpret_cast<const __half*>(Sg.data_ptr<at::Half>());
    const __half* psu = reinterpret_cast<const __half*>(Su.data_ptr<at::Half>());
    const __half* psd = reinterpret_cast<const __half*>(Sd.data_ptr<at::Half>());
    int* pwg = Wg.data_ptr<int>(); int* pzg = Zg.data_ptr<int>();
    int* pwu = Wu.data_ptr<int>(); int* pzu = Zu.data_ptr<int>();
    int* pwd = Wd.data_ptr<int>(); int* pzd = Zd.data_ptr<int>();
    float* pap = apart.data_ptr<float>(); float* pdp = dpart.data_ptr<float>();
    __half* pac = reinterpret_cast<__half*>(acted.data_ptr<at::Half>());
    __half* po  = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    float* pnsq = nsq.data_ptr<float>();
    float feps = (float)eps;
    int wstride = K / 4;      // шаг слоя в qweight у gate/up, в словах int32
    __half* ppi = peer_inbox.has_value()
        ? reinterpret_cast<__half*>(peer_inbox->data_ptr<at::Half>()) : nullptr;
    const __half* pmi = my_inbox.has_value()
        ? reinterpret_cast<const __half*>(my_inbox->data_ptr<at::Half>()) : nullptr;
    int* pmf = my_flag.has_value() ? my_flag->data_ptr<int>() : nullptr;
    int* ppf = peer_flag.has_value() ? peer_flag->data_ptr<int>() : nullptr;
    int sq0 = (int)seq0;
    int upoff = (int)up_col_offset, Ntot = (int)total_width;
    TORCH_CHECK(Ntot == 0 || (upoff + I <= Ntot && (upoff & 3) == 0),
                "mlp_body: смещение up вне плиты или не кратно 4");
    // ПРИ СЛИТОЙ ПЛИТЕ gate и up -- ОДИН тензор; требуем этого явно, иначе адреса разъедутся молча.
    TORCH_CHECK(Ntot == 0 || qw_g.data_ptr() == qw_u.data_ptr(),
                "mlp_body: total_width задан, но gate и up -- разные тензоры");

    void* kargs[] = {(void*)&px, (void*)&pr,
                     (void*)&pwg, (void*)&pzg, (void*)&psg,
                     (void*)&pwu, (void*)&pzu, (void*)&psu,
                     (void*)&pwd, (void*)&pzd, (void*)&psd,
                     (void*)&pap, (void*)&pac, (void*)&pdp, (void*)&po, (void*)&pnsq,
                     (void*)&I, (void*)&K, (void*)&N, (void*)&G, (void*)&feps,
                     (void*)&colsA, (void*)&ksA, (void*)&gpsA,
                     (void*)&colsD, (void*)&ksD, (void*)&gpsD,
                     (void*)&L, (void*)&wstride, (void*)&upoff, (void*)&Ntot,
                     (void*)&ppi, (void*)&pmi, (void*)&pmf, (void*)&ppf, (void*)&sq0};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        kfn, dim3(nblk), dim3(T), kargs, XT2_ * sizeof(float), st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ================================================================= ТЕЛО ПРОЕКЦИЙ GDN
// ПЕРЕСЕЧЕНИЕ, НАЙДЕННОЕ ОПИСЬЮ: `in_proj_qkv` (5120->10240) и `in_proj_z` (5120->6144) читают ОДИН
// И ТОТ ЖЕ нормированный вектор -- как gate/up в MLP. Сейчас это два прохода: две перекладки sx,
// две суммы `xa`, две нормы. Здесь -- ОДИН проход по СКЛЕЕННОЙ ширине 10240+6144: столбец сам
// выбирает, из какого среза веса читать. Обе ширины кратны 512, поэтому плитка столбцов НИКОГДА
// не пересекает границу, и ветка снаружи внутреннего витка.
template <int THREADS, int CPT>
__global__ void gdn_proj_kernel(const __half* __restrict__ x, const __half* __restrict__ rw,
                                const int* __restrict__ qw_q, const int* __restrict__ qz_q,
                                const __half* __restrict__ sc_q,
                                const int* __restrict__ qw_z, const int* __restrict__ qz_z,
                                const __half* __restrict__ sc_z,
                                const __half* __restrict__ w_ab,   // [K, Nab] плотный, БЕЗ квантования
                                float* __restrict__ part, __half* __restrict__ out,
                                int Nq, int Nz, int Nab, int K, int G, float eps, int Nqz,
                                int cols, int ks, int gps) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float sx[XT2_];
    const int nb = gridDim.x, b = blockIdx.x;
    const int NQZ = Nq + Nz, NT = NQZ + Nab;
    const float rs = rms_scale2<THREADS>(x, K, eps);

    // ЗАДАНИЯ ДВУХ КЛАССОВ В ОДНОМ ПРОСТРАНСТВЕ. Плотные a/b (по 24 столбца) в плитку 512 не
    // укладываются, но им и не нужно: они идут ОТДЕЛЬНЫМ классом заданий ТОГО ЖЕ пуска. Смысл не в
    // том, чтобы сделать их быстрее -- 0.49 МБ читаются за 0.56 мкс по крыше, -- а в том, что
    // сейчас это ДВА отдельных пуска, стоящих в очереди ПОСЛЕДОВАТЕЛЬНО с большими проекциями.
    // Внутри одного ядра они идут ОДНОВРЕМЕННО с ними и уходят с критического пути.
    for (int job = b; job < cols * ks + ks; job += nb) {
      if (job >= cols * ks) {                       // --- плотный класс: a и b
        const int ki = job - cols * ks;
        const int kbeg = ki * gps * G, kend = min(K, kbeg + gps * G);
        // 24 столбца на 256 потоков -- это 232 бездельника и ПОСЛЕДОВАТЕЛЬНАЯ цепь по k, из-за
        // которой плотный класс стал критическим путём (+30.7 мкс вместо тени). Даём ему ТУ ЖЕ ось,
        // что и всем: поток берёт (столбец, срез k), потом сведение по срезам через разделяемую.
        // Ширина плотного класса -- 64, а не варп: на боевой форме Nab = 2*Hv = 48 столбцов
        // (a и b по 24), и ограничение в 32 не покрывало её вовсе.
        __shared__ float red[1024];
        constexpr int WID = 64;
        const int lanes = THREADS / WID;            // сколько срезов k внутри блока
        const int j  = threadIdx.x % WID;           // столбец (Nab <= 64)
        const int sl = threadIdx.x / WID;           // номер среза
        float acc = 0.f;
        if (j < Nab) {
            for (int k = kbeg + sl; k < kend; k += lanes)
                acc = fmaf(__half2float(x[k]) * rs * (1.f + __half2float(rw[k])),
                           __half2float(w_ab[(long)k * Nab + j]), acc);
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        if (threadIdx.x < WID) {
            float t = 0.f;
            for (int q = 0; q < lanes; ++q) t += red[q * WID + threadIdx.x];
            if (threadIdx.x < Nab) part[(long)ki * NT + NQZ + threadIdx.x] = t;
        }
        __syncthreads();
        continue;
      }
        const int ci = job % cols, ki = job / cols;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const bool live = (n0 + CPT <= NQZ);
        const bool inq  = (n0 < Nq);
        const int  nl   = Nqz ? n0 : (inq ? n0 : (n0 - Nq));
        // СЛИТАЯ ПЛИТА in_proj_qkvz: z начинается со столбца Nq в ТОМ ЖЕ тензоре, значит локальный
        // индекс = глобальный, а шаг строки = полная ширина. Nqz=0 -> раздельные тензоры, как было.
        const int  W    = Nqz ? Nqz : (inq ? Nq : Nz);
        const int* qw = inq ? qw_q : qw_z;
        const int* qz = inq ? qz_q : qz_z;
        const __half* sc = inq ? sc_q : sc_z;
        const int kbeg = ki * gps * G, kend = min(K, kbeg + gps * G);
        float acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS)
                sx[t] = __half2float(x[k0 + t]) * rs * (1.f + __half2float(rw[k0 + t]));
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPT>(sx, k0, len, qw, qz, sc, nl, W, G, acc);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) part[(long)ki * NT + n0 + c] = acc[c];
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < NT; i += (long)nb * THREADS) {
        float s = 0.f;
        for (int kk = 0; kk < ks; ++kk) s += part[(long)kk * NT + i];
        out[i] = __float2half(s);
    }
}

torch::Tensor gdn_proj(torch::Tensor x, torch::Tensor rms_weight, double eps,
                       torch::Tensor qw_q, torch::Tensor qz_q, torch::Tensor sc_q,
                       torch::Tensor qw_z, torch::Tensor qz_z, torch::Tensor sc_z,
                       torch::Tensor w_ab, int64_t group_size, int64_t total_width, int64_t q_width) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "gdn_proj: плотный тензор на карте");
    const int Nqz = (int)total_width;
    // ПОДМЕНА ЛОВИТСЯ В ОБЕ СТОРОНЫ: слитая плита без ширины посчитала бы q ДВАЖДЫ и молча.
    TORCH_CHECK(Nqz == 0 || qw_q.data_ptr() == qw_z.data_ptr(),
                "gdn_proj: total_width задан, но q и z -- разные тензоры");
    TORCH_CHECK(Nqz != 0 || qw_q.data_ptr() != qw_z.data_ptr(),
                "gdn_proj: q и z -- ОДИН тензор, но total_width не задан");
    // У СЛИТОЙ ПЛИТЫ ОБЕ ширины равны полной -- вывести их из формы НЕЛЬЗЯ, нужна точка раздела.
    const int Nq = Nqz ? (int)q_width : (int)qw_q.size(1);
    const int Nz = Nqz ? (Nqz - Nq)   : (int)qw_z.size(1);
    TORCH_CHECK(!Nqz || (q_width > 0 && Nq < Nqz), "gdn_proj: при слитой плите нужна ширина q");
    const int K = (int)qw_q.size(0) * 4, G = (int)group_size;
    TORCH_CHECK((int)qw_z.size(0) * 4 == K, "gdn_proj: срезы разной глубины");
    const at::cuda::CUDAGuard guard(x.device());
    auto st = at::cuda::getCurrentCUDAStream();
    auto fo = torch::dtype(torch::kFloat32).device(x.device());
    auto ho = torch::dtype(torch::kFloat16).device(x.device());
    const char* ec = getenv("FA2SM70_MB_CPT"); const char* et = getenv("FA2SM70_MB_T");
    const int CPT = ec ? atoi(ec) : 4, T = et ? atoi(et) : 128;
    const int Nab = (int)w_ab.size(1);
    TORCH_CHECK(w_ab.size(0) == K && Nab <= 64, "gdn_proj: плотный кусок шире 64 столбцов");
    const int NT = Nq + Nz + Nab;
    TORCH_CHECK(Nq % (T * CPT) == 0 && Nz % (T * CPT) == 0,
                "gdn_proj: плитка столбцов обязана не пересекать границу срезов");
    const int cols = (Nq + Nz) / (T * CPT), ng = (K + G - 1) / G;
    const void* kfn = (T == 256) ? ((CPT == 4) ? (const void*)gdn_proj_kernel<256,4> : (const void*)gdn_proj_kernel<256,2>)
                                 : ((CPT == 4) ? (const void*)gdn_proj_kernel<128,4> : (const void*)gdn_proj_kernel<128,2>);
    int perSM = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kfn, T, XT2_ * sizeof(float)));
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int nblk = max(1, perSM * nsm);
    // ПРОВЕРКА ГИПОТЕЗЫ: если разница цепи и обёртки -- лишь ДРУГОЕ ДРОБЛЕНИЕ суммы, то при
    // одинаковом числе блоков она обязана исчезнуть. Гейт только для замера, умолчание прежнее.
    if (const char* e_ = getenv("FA2SM70_NBLK")) nblk = atoi(e_);
    int ks = max(1, min(ng, (nblk + cols - 1) / cols));
    const int gps = (ng + ks - 1) / ks; ks = (ng + gps - 1) / gps;
    const int dev = (int)x.device().index();
    static torch::Tensor part_[8]; static int cNT_[8] = {-1,-1,-1,-1,-1,-1,-1,-1}, cKS_[8] = {0};
    TORCH_CHECK(dev >= 0 && dev < 8, "gdn_proj: неожиданный номер карты");
    torch::Tensor &part = part_[dev]; int &cNT = cNT_[dev], &cKS = cKS_[dev];
    if (cNT != NT || cKS != ks) { part = torch::empty({ks, NT}, fo); cNT = NT; cKS = ks; }
    auto out = torch::empty({1, NT}, ho);
    const __half* px = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const __half* pr = reinterpret_cast<const __half*>(rms_weight.data_ptr<at::Half>());
    const __half* psq = reinterpret_cast<const __half*>(sc_q.data_ptr<at::Half>());
    const __half* psz = reinterpret_cast<const __half*>(sc_z.data_ptr<at::Half>());
    int* pwq = qw_q.data_ptr<int>(); int* pzq = qz_q.data_ptr<int>();
    int* pwz = qw_z.data_ptr<int>(); int* pzz = qz_z.data_ptr<int>();
    float* pp = part.data_ptr<float>();
    __half* po = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    float feps = (float)eps;
    const __half* pab = reinterpret_cast<const __half*>(w_ab.data_ptr<at::Half>());
    void* ka[] = {(void*)&px,(void*)&pr,(void*)&pwq,(void*)&pzq,(void*)&psq,
                  (void*)&pwz,(void*)&pzz,(void*)&psz,(void*)&pab,(void*)&pp,(void*)&po,
                  (void*)&Nq,(void*)&Nz,(void*)&Nab,(void*)&K,(void*)&G,(void*)&feps,(void*)&Nqz,
                  (void*)&cols,(void*)&ks,(void*)&gps};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kfn, dim3(nblk), dim3(T), ka, XT2_ * sizeof(float), st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ================================================================= Я2б: МЕШАЛКА GDN (декод)
// Семантика взята ДОСЛОВНО из их ядра `fused_recurrent_gated_delta_rule`, а не из головы:
//   q /= ||q||;  k /= ||k||;  q *= scale
//   h *= exp(g);  v -= h·k;  v *= beta;  h += v (x) k;  o = h·q
// РАСКЛАДКА: блок = одна голова значений, ПОТОК = одна СТРОКА состояния. Строки независимы
// (всё, что их связывает, -- скаляры g и beta), поэтому редукции между потоками нет вовсе.
// Состояние читается ДВАЖДЫ: полная сумма h·k нужна до того, как строку можно переписать, а
// держать 128 значений в регистрах дороже, чем прочитать ещё раз (3.15 -> 4.7 МБ на слой при
// нынешних 7.2 мкс у Triton).
template <int DK, int DV>
__global__ void gdn_mixer_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                                 const __half* __restrict__ v, const float* __restrict__ gg,
                                 const __half* __restrict__ beta,
                                 float* __restrict__ h, __half* __restrict__ o,
                                 int Hk, int Hv, float scale) {
    const int hv = blockIdx.x;
    if (hv >= Hv) return;
    const int hk = hv / (Hv / Hk);
    __shared__ float sq[DK], sk[DK], red[DK];

    const int t = threadIdx.x;
    float qv = __half2float(q[(long)hk * DK + t]);
    float kv = __half2float(k[(long)hk * DK + t]);
    red[t] = qv * qv; __syncthreads();
    for (int s2 = DK >> 1; s2; s2 >>= 1) { if (t < s2) red[t] += red[t + s2]; __syncthreads(); }
    const float nq = rsqrtf(red[0] + 1e-6f); __syncthreads();
    red[t] = kv * kv; __syncthreads();
    for (int s2 = DK >> 1; s2; s2 >>= 1) { if (t < s2) red[t] += red[t + s2]; __syncthreads(); }
    const float nk = rsqrtf(red[0] + 1e-6f); __syncthreads();
    sq[t] = qv * nq * scale;
    sk[t] = kv * nk;
    __syncthreads();

    const float gd = __expf(gg[hv]);
    const float bt = __half2float(beta[hv]);
    float* hrow = h + (long)hv * DV * DK + (long)t * DK;

    float dk = 0.f;
    #pragma unroll 4
    for (int j = 0; j < DK; ++j) dk = fmaf(hrow[j], sk[j], dk);
    const float vv = bt * (__half2float(v[(long)hv * DV + t]) - gd * dk);

    float acc = 0.f;
    #pragma unroll 4
    for (int j = 0; j < DK; ++j) {
        const float hn = fmaf(gd, hrow[j], vv * sk[j]);
        hrow[j] = hn;
        acc = fmaf(hn, sq[j], acc);
    }
    o[(long)hv * DV + t] = __float2half(acc);
}

torch::Tensor gdn_mixer(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                        torch::Tensor g, torch::Tensor beta, torch::Tensor h, double scale) {
    TORCH_CHECK(q.is_cuda() && h.is_cuda(), "gdn_mixer: тензоры на карте");
    const int Hk = (int)q.size(0), DK = (int)q.size(1);
    const int Hv = (int)v.size(0), DV = (int)v.size(1);
    TORCH_CHECK(DK == 128 && DV == 128, "gdn_mixer: собрано под 128x128");
    TORCH_CHECK(Hv % Hk == 0, "gdn_mixer: головы значений не кратны головам ключей");
    const at::cuda::CUDAGuard guard(q.device());
    auto o = torch::empty({Hv, DV}, torch::dtype(torch::kFloat16).device(q.device()));
    gdn_mixer_kernel<128,128><<<Hv, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        g.data_ptr<float>(),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        h.data_ptr<float>(),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()), Hk, Hv, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}


// ================================================================= Я2в: ПРОЕКЦИЯ ИЗ ГОТОВОГО ВЕКТОРА
// `gdn.out_proj` и `attn.o_proj` -- те же GEMV, но БЕЗ нормы: вход уже посчитан мешалкой. Это фаза C
// тела MLP, вынутая отдельной точкой входа; в резидентном шаге она станет ещё одним классом заданий.
template <int THREADS, int CPT>
__global__ void proj_from_kernel(const __half* __restrict__ act,
                                 const int* __restrict__ qw, const int* __restrict__ qz,
                                 const __half* __restrict__ sc,
                                 float* __restrict__ part, __half* __restrict__ out,
                                 int K, int N, int G, int cols, int ks, int gps) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float sx[XT2_];
    const int nb = gridDim.x, b = blockIdx.x;
    for (int job = b; job < cols * ks; job += nb) {
        const int ci = job % cols, ki = job / cols;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gps * G, kend = min(K, kbeg + gps * G);
        const bool live = (n0 + CPT <= N);
        float acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS) sx[t] = __half2float(act[k0 + t]);
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPT>(sx, k0, len, qw, qz, sc, n0, N, G, acc);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) part[(long)ki * N + n0 + c] = acc[c];
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < N; i += (long)nb * THREADS) {
        float v = 0.f;
        for (int kk = 0; kk < ks; ++kk) v += part[(long)kk * N + i];
        out[i] = __float2half(v);
    }
}

torch::Tensor proj_from(torch::Tensor act, torch::Tensor qw, torch::Tensor qz,
                        torch::Tensor sc, int64_t group_size) {
    TORCH_CHECK(act.is_cuda() && act.is_contiguous(), "proj_from: плотный вектор на карте");
    const int K = (int)qw.size(0) * 4, N = (int)qw.size(1), G = (int)group_size;
    TORCH_CHECK(act.numel() == K, "proj_from: собрано под M=1");
    const at::cuda::CUDAGuard guard(act.device());
    auto st = at::cuda::getCurrentCUDAStream();
    const char* ec = getenv("FA2SM70_MB_CPT"); const char* et = getenv("FA2SM70_MB_T");
    const int CPT = ec ? atoi(ec) : 4, T = et ? atoi(et) : 128;
    TORCH_CHECK(N % (T * CPT) == 0, "proj_from: N кратно плитке столбцов");
    const int cols = N / (T * CPT), ng = (K + G - 1) / G;
    const void* kfn = (T == 256) ? ((CPT == 4) ? (const void*)proj_from_kernel<256,4> : (const void*)proj_from_kernel<256,2>)
                                 : ((CPT == 4) ? (const void*)proj_from_kernel<128,4> : (const void*)proj_from_kernel<128,2>);
    int perSM = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kfn, T, XT2_ * sizeof(float)));
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int nblk = max(1, perSM * nsm);
    // ПРОВЕРКА ГИПОТЕЗЫ: если разница цепи и обёртки -- лишь ДРУГОЕ ДРОБЛЕНИЕ суммы, то при
    // одинаковом числе блоков она обязана исчезнуть. Гейт только для замера, умолчание прежнее.
    if (const char* e_ = getenv("FA2SM70_NBLK")) nblk = atoi(e_);
    int ks = max(1, min(ng, (nblk + cols - 1) / cols));
    const int gps = (ng + ks - 1) / ks; ks = (ng + gps - 1) / gps;
    auto part = torch::empty({ks, N}, torch::dtype(torch::kFloat32).device(act.device()));
    auto out  = torch::empty({1, N}, torch::dtype(torch::kFloat16).device(act.device()));
    const __half* pa = reinterpret_cast<const __half*>(act.data_ptr<at::Half>());
    const __half* ps = reinterpret_cast<const __half*>(sc.data_ptr<at::Half>());
    int* pw = qw.data_ptr<int>(); int* pz = qz.data_ptr<int>();
    float* pp = part.data_ptr<float>();
    __half* po = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    void* ka[] = {(void*)&pa,(void*)&pw,(void*)&pz,(void*)&ps,(void*)&pp,(void*)&po,
                  (void*)&K,(void*)&N,(void*)&G,(void*)&cols,(void*)&ks,(void*)&gps};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kfn, dim3(nblk), dim3(T), ka, XT2_ * sizeof(float), st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ================================================================= Я3: ЯДРО ВНИМАНИЯ НА ДЕКОДЕ
// Один запрос против Sk ключей. Онлайн-softmax (максимум и сумма ведутся на лету), срезы по ключам
// РАЗНЫМИ блоками -- та же ось расщепления редукции, что и у проекций; сведение срезов после
// барьера по сетке. Ключи/значения приходят ПЛОТНЫМИ [Sk, Hkv, D]: связывание с постраничным
// складом -- отдельный шаг, здесь проверяется арифметика.
template <int D>
__global__ void attn_decode_kernel(const __half* __restrict__ q,      // [H, D]
                                   const __half* __restrict__ kc,     // [Sk, Hkv, D]
                                   const __half* __restrict__ vc,
                                   float* __restrict__ pm,            // [nsp, H] максимум среза
                                   float* __restrict__ pl,            // [nsp, H] сумма среза
                                   float* __restrict__ pa,            // [nsp, H, D] накопитель
                                   __half* __restrict__ out,          // [H, D]
                                   int H, int Hkv, int Sk, int nsp, float scale) {
    cg::grid_group grid = cg::this_grid();
    const int nb = gridDim.x, b = blockIdx.x;
    const int njob = H * nsp;
    __shared__ float sq[D], red[D];
    for (int job = b; job < njob; job += nb) {
        const int hh = job % H, sp = job / H;
        const int hkv = hh / (H / Hkv);
        const int t = threadIdx.x;
        sq[t] = __half2float(q[(long)hh * D + t]) * scale;
        __syncthreads();
        const int beg = (long)sp * Sk / nsp, end = (long)(sp + 1) * Sk / nsp;
        float m = -INFINITY, l = 0.f, acc = 0.f;
        for (int s = beg; s < end; ++s) {
            const __half* kp = kc + ((long)s * Hkv + hkv) * D;
            red[t] = sq[t] * __half2float(kp[t]);
            __syncthreads();
            for (int r = D >> 1; r; r >>= 1) { if (t < r) red[t] += red[t + r]; __syncthreads(); }
            const float sc_ = red[0];
            __syncthreads();
            const float mn = fmaxf(m, sc_);
            const float cor = __expf(m - mn), p = __expf(sc_ - mn);
            l = l * cor + p;
            acc = acc * cor + p * __half2float(vc[((long)s * Hkv + hkv) * D + t]);
            m = mn;
        }
        pm[(long)sp * H + hh] = m;
        pl[(long)sp * H + hh] = l;
        pa[((long)sp * H + hh) * D + t] = acc;
    }
    ГБАР(grid);
    // сведение срезов: тот же онлайн-softmax, но по срезам
    for (int hh = b; hh < H; hh += nb) {
        const int t = threadIdx.x;
        float m = -INFINITY;
        for (int sp = 0; sp < nsp; ++sp) m = fmaxf(m, pm[(long)sp * H + hh]);
        float l = 0.f, acc = 0.f;
        for (int sp = 0; sp < nsp; ++sp) {
            const float w = __expf(pm[(long)sp * H + hh] - m);
            l += pl[(long)sp * H + hh] * w;
            acc += pa[((long)sp * H + hh) * D + t] * w;
        }
        out[(long)hh * D + t] = __float2half(acc / l);
    }
}

torch::Tensor attn_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, double scale) {
    TORCH_CHECK(q.is_cuda() && kc.is_cuda(), "attn_decode: тензоры на карте");
    const int H = (int)q.size(0), D = (int)q.size(1);
    const int Sk = (int)kc.size(0), Hkv = (int)kc.size(1);
    TORCH_CHECK(D == 256, "attn_decode: собрано под d=256");
    TORCH_CHECK(H % Hkv == 0, "attn_decode: головы запроса не кратны головам ключей");
    const at::cuda::CUDAGuard guard(q.device());
    auto st = at::cuda::getCurrentCUDAStream();
    auto fo = torch::dtype(torch::kFloat32).device(q.device());
    const void* kfn = (const void*)attn_decode_kernel<256>;
    int perSM = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kfn, D, 0));
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int nblk = max(1, perSM * nsm);
    // ПРОВЕРКА ГИПОТЕЗЫ: если разница цепи и обёртки -- лишь ДРУГОЕ ДРОБЛЕНИЕ суммы, то при
    // одинаковом числе блоков она обязана исчезнуть. Гейт только для замера, умолчание прежнее.
    if (const char* e_ = getenv("FA2SM70_NBLK")) nblk = atoi(e_);
    int nsp = max(1, min(Sk, nblk / max(1, H)));
    auto pm = torch::zeros({nsp, H}, fo), pl = torch::zeros({nsp, H}, fo);
    auto pa = torch::zeros({nsp, H, D}, fo);
    auto out = torch::empty({H, D}, torch::dtype(torch::kFloat16).device(q.device()));
    const __half* pq = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const __half* pk = reinterpret_cast<const __half*>(kc.data_ptr<at::Half>());
    const __half* pv = reinterpret_cast<const __half*>(vc.data_ptr<at::Half>());
    float* ppm = pm.data_ptr<float>(); float* ppl = pl.data_ptr<float>(); float* ppa = pa.data_ptr<float>();
    __half* po = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    float fs = (float)scale;
    void* ka[] = {(void*)&pq,(void*)&pk,(void*)&pv,(void*)&ppm,(void*)&ppl,(void*)&ppa,(void*)&po,
                  (void*)&H,(void*)&Hkv,(void*)&Sk,(void*)&nsp,(void*)&fs};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kfn, dim3(nblk), dim3(D), ka, 0, st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}




// ================================================================== ОБМЕН МЕЖДУ КАРТАМИ ПО ПЛАНУ
// Порядок в сети: проекция -> СВЕДЕНИЕ МЕЖДУ КАРТАМИ -> остаточная связь. Вставлять в конец нельзя:
// остаточная связь не делится между картами, её нельзя складывать дважды.
//
// ЧТО ЗАМЕРЕНО (карты 0-1, мост PHB, посылка 10 КБ, 300 кругов; scratchpad/exch_bench.cu):
//   пол (пинг-понг одним флагом)            6.06 мкс   <- цена ОДНОГО рандеву
//   ТОЛКАЮ (запись соседу), поблочные флаги  8.35       <- +2.3 к полу за 10 КБ записи
//   ТЯНУ, 2 барьера (конструкция vLLM)      19.09       <- чтение вдвое дороже записи + 2-е рандеву
//   ТЯНУ, 1 барьер (двойной буфер)          11.23
// Отсюда: на мосту PCIe ТОЛКАТЬ дешевле, чем ТЯНУТЬ (x1.34-1.54). vLLM тянет -- это верно для
// NVLink, где чтение дёшево и тянущая схема вдобавок даёт побитово одинаковую сумму на всех
// рангах. Направление поэтому -- ПОЛЕ ПЛАНА, а не константа: выбирается по классу канала.
//
// ШИРИНА РАНДЕВУ. Замерено: 36 блоков лучше 80 в 2.1 раза (8.35 против 17.82) -- одновременные
// рандеву через мост толкаются друг с другом. 36 -- это ровно `defaultBlockLimit` их ядра, сошлось
// независимо. Поэтому в обмене участвуют ПЕРВЫЕ `ширина` блоков сетки, остальные ждут на барьере.
//
// БАРЬЕРОВ ДВА НА ВЕСЬ ОБМЕН, А НЕ ТРИ НА ЯРУС. Блок b берёт свою долю данных и ждёт флаг b от
// соседа: доля, которую он прочтёт, и есть та, о приходе которой ему сказали. Между ярусами
// барьер не нужен -- вход яруса 2 у блока b это его же выход яруса 1.
#define EXCH_MAX_PEERS 3
#define EXCH_MAX_TIERS  2
#define EXCH_SPIN    200000000ULL

struct PeerPlan {
    int tiers;                                   // 0 -> обмена нет
    int peers[EXCH_MAX_TIERS];
    int pull[EXCH_MAX_TIERS];                       // 1 = читать у соседа, 0 = писать соседу
    int width;                                   // блоков, участвующих в рандеву
    __half*       far[EXCH_MAX_TIERS][EXCH_MAX_PEERS]; // толкаю: ящик соседа; тяну: его буфер-источник
    const __half* inbox[EXCH_MAX_TIERS][EXCH_MAX_PEERS];// мой ящик от этого соседа
    int*          far_flag[EXCH_MAX_TIERS][EXCH_MAX_PEERS];
    int*          mine_flag[EXCH_MAX_TIERS][EXCH_MAX_PEERS];
    // СЧЁТ КРУГОВ ДЕРЖИТ УСТРОЙСТВО, а не хост. Номер, заданный с хоста, начинается заново на
    // следующем шаге -- и ожидание проскочило бы по СТАРЫМ флагам; в CUDA-графе он вдобавок
    // запечён и никогда не меняется. Свой счётчик на блок (не расшарен) снимает обе беды:
    // обе карты делают одинаковое число рандеву за пуск, поэтому счётчики идут в ногу.
    int*          count;                                  // длина = width, приватный, обнулён один раз
};

static __device__ __forceinline__ void exch_st_flag(int* p, int v) {
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(v), "l"(p) : "memory");
}
static __device__ __forceinline__ int exch_ld_flag(int* p) {
    int v;
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.sys;" : "=r"(v) : "l"(p) : "memory");
    return v;
}
// ОПРОС БЕЗ ЗАБОРА -- ОПРОВЕРГНУТО ЗАМЕРОМ (11.08.2026), оставлено под гейтом как память.
// Ход был: `membar.sys` в цикле -- это остановка конвейера системного масштаба на КАЖДЫЙ опрос,
// а захват нужен один раз. Замер в сервере: 0.2 ток/с в ОБЕИХ попытках (`ld.volatile.global`
// и `ld.global.cv`) -- флаг соседа без забора не становится виден вовсе, спин уходит до упора
// EXCH_SPIN. Значит `membar.sys` у читателя тут НЕСУЩИЙ: он разрешает системную видимость, а не
// просто упорядочивает. Рычаг остаётся в ЧАСТОТЕ опроса (FA2SM70_СПИН_ПАУЗА), не в семантике.
static __device__ __forceinline__ int exch_ld_flag_opros(int* p) {
    int v;
    // `.cv` -- "считать закэшированное УСТАРЕВШИМ и перечитать". Именно это, а не забор, нужно
    // ждущему: `ld.volatile` без забора флаг НЕ обновляет (замерено -- сервер уходил в упор
    // спина, 0.2 ток/с), то есть прежний `membar.sys` в цикле работал ПЕРЕЧИТЫВАНИЕМ, а его
    // настоящая роль -- захват -- нужна РОВНО ОДИН раз, после ожидания.
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
static __device__ __forceinline__ void exch_zahvat() {
    asm volatile("membar.sys;" ::: "memory");
}

template <int THREADS>
__device__ __forceinline__ void peer_reduce(cg::grid_group& grid, __half* __restrict__ mine,
                                            const PeerPlan& plan, int n, int seq) {
#ifndef FA2SM70_CPT_OUT
// ШИРИНА ОБРАЩЕНИЯ ОТДЕЛЬНО ДЛЯ out_proj. У него столбцовых плиток всего K/(T*CPT)=10, поэтому
// срезов по k выходит 24 и на задание достаётся 128 значений k -- ВТРОЕ меньше, чем у MLP (384).
// Постоянная цена задания (стадирование активации, два барьера блока, установка адресов) при этом
// та же, то есть размазана на втрое меньшую работу. Ширина -- свободная переменная этой фазы.
#define FA2SM70_CPT_OUT CPT
#define FA2SM70_CPT_OUT_HOST CPT
#endif
#ifndef FA2SM70_CPT_OUT_HOST
#define FA2SM70_CPT_OUT_HOST FA2SM70_CPT_OUT
#endif
#ifndef FA2SM70_EXCH_W
#define FA2SM70_EXCH_W 36
#endif
#ifdef FA2SM70_БЕЗ_ОБМЕНА
    return;                           // фальсификатор: обмен между картами снят целиком
#endif
    if (plan.tiers <= 0) return;
    const int b = blockIdx.x;
    const int W = plan.width;
    ГБАР(grid);                       // весь `mine` дописан сеткой -- только теперь его можно отдавать
    if (b < W) {
        // Доля блока: непрерывный кусок, ОДИНАКОВО нарезанный на всех картах.
        const long шаг = (n + W - 1) / W;
        const long от = (long)b * шаг, до = min((long)n, от + шаг);
        for (int t = 0; t < plan.tiers; ++t) {
            const int S = plan.peers[t];
            __shared__ int seq_now;
            if (threadIdx.x == 0) seq_now = ++plan.count[b];   // круг считает УСТРОЙСТВО
            __syncthreads();
            if (!plan.pull[t]) {
                for (long i = от + threadIdx.x; i < до; i += THREADS) {
                    const __half v = mine[i];
                    for (int k = 0; k < S; ++k) plan.far[t][k][i] = v;
                }
                __threadfence_system();
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                for (int k = 0; k < S; ++k) exch_st_flag(plan.far_flag[t][k] + b, seq_now);
                unsigned long long spin = 0;
#ifdef FA2SM70_ЗАБОР_ОДИН
                for (int k = 0; k < S; ++k)
                    while (exch_ld_flag_opros(plan.mine_flag[t][k] + b) < seq_now)
                        if (++spin > EXCH_SPIN) break;
                exch_zahvat();                 // забор ОДИН раз, после ожидания
#else
                for (int k = 0; k < S; ++k)
                    while (exch_ld_flag(plan.mine_flag[t][k] + b) < seq_now) {
                        if (++spin > EXCH_SPIN) break;
#if defined(FA2SM70_СПИН_ПАУЗА) && FA2SM70_СПИН_ПАУЗА > 0
                        // Забор несущий, но его ЧАСТОТА -- свободная переменная: 36 блоков
                        // одновременно бьют системным забором по одной и той же шине.
                        __nanosleep(FA2SM70_СПИН_ПАУЗА);
#endif
                    }
#endif
            }
            __syncthreads();
            // Порядок слагаемых по НОМЕРУ СОСЕДА -- одинаков на всех картах, сумма побитово та же.
            for (long i = от + threadIdx.x; i < до; i += THREADS) {
                float s = __half2float(mine[i]);
#ifdef FA2SM70_ЯЩИК_МИМО_КЭША
                // Ящик пишет ЧУЖАЯ карта, а L1 у Volta между картами не согласован: строка,
                // прочитанная на прошлом слое, может вернуться как есть. Читаем `ld.global.cv`.
                for (int k = 0; k < S; ++k) {
                    const __half* ap = plan.pull[t] ? (plan.far[t][k] + i) : (plan.inbox[t][k] + i);
                    unsigned short raw;
                    asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(raw) : "l"(ap) : "memory");
                    __half hv2; memcpy(&hv2, &raw, sizeof(raw));
                    s += __half2float(hv2);
                }
#else
                for (int k = 0; k < S; ++k)
                    s += __half2float(plan.pull[t] ? plan.far[t][k][i] : plan.inbox[t][k][i]);
#endif
                mine[i] = __float2half(s);
            }
            __syncthreads();
        }
    }
    ГБАР(grid);                       // сведённое видно всей сетке
}

// Однорангового соседа (наш прежний случай TP=2) выражаем тем же планом -- один ярус, один сосед.
__device__ int exch_pair_count[64];   // только для одиночных ядер слоёв

__device__ __forceinline__ PeerPlan plan_pair(__half* peer_inbox, const __half* my_inbox,
                                              int* my_flag, int* peer_flag, int width) {
    PeerPlan p{};
    if (peer_inbox == nullptr) { p.tiers = 0; return p; }
    p.tiers = 1; p.peers[0] = 1; p.pull[0] = 0; p.width = width;
    p.far[0][0] = peer_inbox; p.inbox[0][0] = my_inbox;
    p.far_flag[0][0] = peer_flag; p.mine_flag[0][0] = my_flag; p.count = exch_pair_count;
    return p;
}


// ============================== ПРОВЕРКА ОБМЕНА ОТДЕЛЬНО ОТ ЦЕПИ =================================
// Дефект искался подъёмами сервера по три минуты -- это негодная единица отладки. Здесь ТОЛЬКО
// сведение, тем же кодом и тем же планом, кооперативным пуском. Ответ читается по значению
// (сумма единиц обязана дать число карт), а не по отсутствию падения.
template <int THREADS>
__global__ void exch_selftest_kernel(PeerPlan plan, __half* __restrict__ mine, int n) {
    cg::grid_group grid = cg::this_grid();
    peer_reduce<THREADS>(grid, mine, plan, n, 0);
}

torch::Tensor exch_selftest(torch::Tensor mine,
                            int64_t tiers, std::vector<int64_t> peers, std::vector<int64_t> pull,
                            std::vector<int64_t> far, std::vector<torch::Tensor> inbox,
                            std::vector<int64_t> far_flag, std::vector<torch::Tensor> my_flag,
                            int64_t width, torch::Tensor count, int64_t blocks) {
    TORCH_CHECK(mine.is_cuda() && mine.scalar_type() == torch::kHalf && mine.is_contiguous(),
                "exch_selftest: mine обязан быть непрерывным fp16 на карте");
    TORCH_CHECK(tiers >= 1 && tiers <= EXCH_MAX_TIERS, "exch_selftest: ярусов вне 1..", EXCH_MAX_TIERS);
    const int T = 128;
    PeerPlan plan{};
    plan.tiers = (int)tiers;
    plan.width = (int)width;
    plan.count = count.data_ptr<int>();
    TORCH_CHECK(count.numel() >= plan.width, "exch_selftest: счётчик короче ширины");
    int c = 0;
    for (int t = 0; t < plan.tiers; ++t) {
        plan.peers[t] = (int)peers[t];
        plan.pull[t] = pull[t] ? 1 : 0;
        TORCH_CHECK(plan.peers[t] >= 1 && plan.peers[t] <= EXCH_MAX_PEERS, "exch_selftest: соседей вне 1..3");
        for (int k = 0; k < plan.peers[t]; ++k, ++c) {
            TORCH_CHECK(far[c] && far_flag[c], "exch_selftest: нулевой чужой указатель");
            TORCH_CHECK(inbox[c].numel() >= mine.numel() && my_flag[c].numel() >= plan.width,
                        "exch_selftest: свои буферы короче нужного");
            plan.far[t][k] = (__half*)(uintptr_t)far[c];
            plan.inbox[t][k] = (const __half*)inbox[c].data_ptr();
            plan.far_flag[t][k] = (int*)(uintptr_t)far_flag[c];
            plan.mine_flag[t][k] = my_flag[c].data_ptr<int>();
        }
    }
    int n = (int)mine.numel();
    __half* pm = (__half*)mine.data_ptr();
    void* ka[] = {(void*)&plan, (void*)&pm, (void*)&n};
    auto st = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)exch_selftest_kernel<128>,
                                               dim3((unsigned)blocks), dim3(T), ka, 0, st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mine;
}

// ================================================================= СБОРКА: ВЕСЬ СЛОЙ GDN ОДНИМ ПУСКОМ
// Ярусы сведены в одно кооперативное ядро классами заданий в общем пространстве. Между фазами --
// барьер по сетке, буферы передаются напрямую, наружу не выходит НИЧЕГО, кроме выхода слоя.
//
// Порядок фаз ровно как в SEQ_DECODE.md:
//   1) норма -> (qkv | z | a | b) одним проходом по общему входу   [классы: квантованный + плотный]
//   2) свёртка по 4 токенам на qkv (состояние) + silu
//   3) мешалка: l2norm, гейт, дельта-правило, выход                [блок = голова, поток = строка]
//   4) norm(o) * silu(z)
//   5) out_proj
// Обмен и MLP уже есть в `mlp_body`; здесь собран НЕЛИНЕЙНЫЙ кусок, которого не было.
// ТЕЛО КАК УСТРОЙСТВЕННАЯ ФУНКЦИЯ. Барьер и плитка приходят СНАРУЖИ: только так цикл по слоям
// сможет вызывать оба типа тела в ОДНОМ пуске. Ветвление идёт по номеру слоя -- он одинаков во всех
// блоках, поэтому расхождения на барьере не возникает по построению.
// ==================================================================================================
// РАЗБОР qkv НА МЕСТЕ, СЕТКОЙ. Тот же код, что и в проверенном одиночном узле -- не вторая копия:
// две копии расходятся молча, и это ровно тот класс, который сегодня ловился четыре раза.
// На входе mid = [q_gate(Hq*2D) k(Hkv*D) v(Hkv*D)], на выходе mid = [q k v ГЕЙТ], длина та же.
// ==================================================================================================
// ПЛОТНЫЙ ОПЕРАТОР С НОРМОЙ НА ЛЕТУ. В теле слоя разделяемая занята плиткой, а K=5120 float это
// 20 КБ -- вход туда не кладём. Варп читает свою строку веса подряд, а x/rw берёт из глобальной:
// они 20 КБ на блок и живут в L1, то есть повторное чтение идёт не из HBM.
template <int THREADS>
__device__ __forceinline__ void gemv_dense_norm(const __half* __restrict__ x,
                                                const __half* __restrict__ rw, float rs, int K,
                                                const __half* __restrict__ w, int n0, int nN,
                                                __half* __restrict__ out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = THREADS >> 5;
    for (int n = n0 + warp; n < nN; n += nwarp) {
        const __half* row = w + (long)n * K;
        float sum = 0.f;
        for (int k = lane * 8; k < K; k += 32 * 8) {
            const int4 vw = *reinterpret_cast<const int4*>(row + k);
            const int4 vx = *reinterpret_cast<const int4*>(x + k);
            const int4 vr = *reinterpret_cast<const int4*>(rw + k);
            const __half2 *hw = reinterpret_cast<const __half2*>(&vw),
                          *hx = reinterpret_cast<const __half2*>(&vx),
                          *hr = reinterpret_cast<const __half2*>(&vr);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 fw = __half22float2(hw[j]), fx = __half22float2(hx[j]),
                             fr = __half22float2(hr[j]);
                sum += fw.x * (fx.x * rs * (1.f + fr.x)) + fw.y * (fx.y * rs * (1.f + fr.y));
            }
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
        if (lane == 0) out[n] = __float2half(sum);
    }
}

template <int THREADS>
__device__ void qkv_post_grid(cg::grid_group& grid, __half* __restrict__ mid,
                              float* __restrict__ tmp,
                              const __half* __restrict__ q_norm, const __half* __restrict__ k_norm,
                              const float* __restrict__ cs,
                              int Hq, int Hkv, int D, int rot, float eps, int nb, int b) {
    __shared__ float red[THREADS / 32];
    __shared__ float sh[512];
    const int oq = 0, ok = Hq * D, ov = ok + Hkv * D, og = ov + Hkv * D;
    for (int job = b; job < Hq + 2 * Hkv; job += nb) {
        const bool is_q = job < Hq;
        const bool is_k = !is_q && job < Hq + Hkv;
        const int h = is_q ? job : (is_k ? job - Hq : job - Hq - Hkv);
        if (!is_q && !is_k) {                                   // V: без нормы и вращения
            for (int t = threadIdx.x; t < D; t += THREADS)
                tmp[ov + h * D + t] = __half2float(mid[(long)Hq * 2 * D + Hkv * D + h * D + t]);
            continue;
        }
        const long base = is_q ? (long)h * 2 * D : (long)Hq * 2 * D + (long)h * D;
        float ss = 0.f;
        for (int t = threadIdx.x; t < D; t += THREADS) {
            const float v = __half2float(mid[base + t]);
            sh[t] = v; ss = fmaf(v, v, ss);
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t2 = 0.f;
            for (int i = 0; i < THREADS / 32; ++i) t2 += red[i];
            red[0] = rsqrtf(t2 / (float)D + eps);
        }
        __syncthreads();
        const float rs = red[0];
        const __half* nw = is_q ? q_norm : k_norm;
        for (int t = threadIdx.x; t < D; t += THREADS)          // ЕДИНИЦА: GemmaRMSNorm
            sh[t] = sh[t] * rs * (1.f + __half2float(nw[t]));
        __syncthreads();
        const int half = rot >> 1;
        for (int t = threadIdx.x; t < D; t += THREADS) {
            float y = sh[t];
            if (t < rot) {                                      // вращение ТОЛЬКО первых rot
                const int j = t % half;
                const float c = cs[j], sn = cs[half + j];
                y = (t < half) ? (sh[j] * c - sh[half + j] * sn)
                               : (sh[half + j] * c + sh[j] * sn);
            }
            tmp[(is_q ? oq : ok) + h * D + t] = y;
        }
        if (is_q) {
            for (int t = threadIdx.x; t < D; t += THREADS) {
                const float g = __half2float(mid[(long)h * 2 * D + D + t]);
                tmp[og + h * D + t] = 1.f / (1.f + __expf(-g));  // sigmoid СРАЗУ
            }
        }
        __syncthreads();
    }
    ГБАР(grid);
    const int total = og + Hq * D;
    for (long i = (long)b * THREADS + threadIdx.x; i < total; i += (long)nb * THREADS)
        mid[i] = __float2half(tmp[i]);
    ГБАР(grid);
}

template <int THREADS, int CPT>
__device__ void gdn_layer_body(cg::grid_group& grid, float* __restrict__ sx,
                               const __half* __restrict__ x, const __half* __restrict__ rw,
                                 const int* __restrict__ qw_q, const int* __restrict__ qz_q,
                                 const __half* __restrict__ sc_q,
                                 const int* __restrict__ qw_z, const int* __restrict__ qz_z,
                                 const __half* __restrict__ sc_z,
                                 const __half* __restrict__ w_ab,
                                 const int* __restrict__ qw_o, const int* __restrict__ qz_o,
                                 const __half* __restrict__ sc_o,
                                 const __half* __restrict__ conv_w,   // [Cqkv, 4]
                                 __half* __restrict__ conv_state,     // хвост из 3 токенов
                                 // ШАГИ вместо предположения о плотности: у движка слот лежит
                                 // как [3, Cqkv], у нас исторически как [Cqkv, 3]. Читать по
                                 // шагам -- значит НЕ делать транспонирующую копию на каждый шаг
                                 // и не зависеть от чужой раскладки.
                                 long кш_i, long кш_j,
                                 const float* __restrict__ a_log, const __half* __restrict__ dt_bias,
                                 float* __restrict__ hstate,          // [Hv, DV, DK]
                                 const __half* __restrict__ gnorm,    // вес RMSNormGated выхода
                                 __half* __restrict__ dbg_плита,      // слепок mid[0..NT) после фазы 2
                                 float* __restrict__ part1, __half* __restrict__ mid,
                                 float* __restrict__ part2, __half* __restrict__ out,
                                 const __half* __restrict__ rw2,
                                 const int* __restrict__ qw_g2, const int* __restrict__ qz_g2,
                                 const __half* __restrict__ sc_g2,
                                 const int* __restrict__ qw_u2, const int* __restrict__ qz_u2,
                                 const __half* __restrict__ sc_u2,
                                 const int* __restrict__ qw_d2, const int* __restrict__ qz_d2,
                                 const __half* __restrict__ sc_d2,
                                 float* __restrict__ part3, __half* __restrict__ mid2,
                                 float* __restrict__ part4,
                                 int Nq, int Nz, int Nab, int K, int G, float eps, float scale,
                                 int cols1, int ks1, int gps1,
                                 int cols2, int ks2, int gps2,
                                 int Hk, int Hv, int DK, int DV, int Kout,
                                 int I2, int cols3, int ks3, int gps3,
                               int cols4, int ks4, int gps4,
                               const PeerPlan& plan,
                               int seq0, __half* __restrict__ buf_mlp,
                                  int upoff2, int Ntot2, int Nqz2) {
    const int nb = gridDim.x, b = blockIdx.x;
    const int NQZ = Nq + Nz, NT = NQZ + Nab;
    const float rs = rms_scale2<THREADS>(x, K, eps);

#ifdef FA2SM70_ТОЛЬКО_MLP
    // ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ: вся GDN-часть слоя снята, остаётся ровно MLP. Ответ заведомо
    // неверен, читается ТОЛЬКО время -- разность с полной цепью даёт ДОЛЮ, а не догадку.
    (void)rs;
    {
        float ps2 = 0.f;
        for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
            const float v = __half2float(x[i]);
            out[i] = __float2half(v);
            ps2 = fmaf(v, v, ps2);
        }
        #pragma unroll
        for (int o2 = 16; o2; o2 >>= 1) ps2 += __shfl_down_sync(0xffffffff, ps2, o2);
        if ((threadIdx.x & 31) == 0) atomicAdd(&part3[ks3 * I2 * 2], ps2);
    }
    ГБАР(grid);
#else
    // ---- фаза 1: проекции (квантованный класс + плотный класс a/b)
#ifndef FA2SM70_БЕЗ_ФАЗЫ1
    for (int job = b; job < cols1 * ks1 + ks1; job += nb) {
        if (job >= cols1 * ks1) {
            const int ki = job - cols1 * ks1;
            const int kbeg = ki * gps1 * G, kend = min(K, kbeg + gps1 * G);
            __shared__ float red2[1024];
            constexpr int WID = 64;                 // Nab = 2*Hv = 48 на боевой форме
            const int lanes = THREADS / WID, j = threadIdx.x % WID, sl = threadIdx.x / WID;
            float acc = 0.f;
            if (j < Nab)
                for (int k = kbeg + sl; k < kend; k += lanes)
                    acc = fmaf(__half2float(x[k]) * rs * (1.f + __half2float(rw[k])),
                               __half2float(w_ab[(long)k * Nab + j]), acc);
            red2[threadIdx.x] = acc; __syncthreads();
            if (threadIdx.x < WID) {
                float t = 0.f;
                for (int q2 = 0; q2 < lanes; ++q2) t += red2[q2 * WID + threadIdx.x];
                if (threadIdx.x < Nab) part1[(long)ki * NT + NQZ + threadIdx.x] = t;
            }
            __syncthreads();
            continue;
        }
        const int ci = job % cols1, ki = job / cols1;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const bool live = (n0 + CPT <= NQZ);
        const bool inq = (n0 < Nq);
        const int nl = Nqz2 ? n0 : (inq ? n0 : (n0 - Nq));
        const int W  = Nqz2 ? Nqz2 : (inq ? Nq : Nz);
        const int* qw = inq ? qw_q : qw_z; const int* qz = inq ? qz_q : qz_z;
        const __half* sc = inq ? sc_q : sc_z;
        const int kbeg = ki * gps1 * G, kend = min(K, kbeg + gps1 * G);
        float acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS)
                sx[t] = __half2float(x[k0 + t]) * rs * (1.f + __half2float(rw[k0 + t]));
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPT>(sx, k0, len, qw, qz, sc, nl, W, G, acc);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) part1[(long)ki * NT + n0 + c] = acc[c];
        }
    }
#endif
    ГБАР(grid);

    // ---- фаза 2: сведение проекций + свёртка по 4 токенам + silu (только для qkv)
    // ОТКАТ 10.08.2026: попытка читать плиту ПО K-ГОЛОВЕ (как делит модель в
    // fix_query_key_value_ordering) СЛОМАЛА вывод в мусор -- значит сплошное чтение здесь уже
    // чем-то компенсировано, и разрез, показавший расхождение, сам требует перепроверки.
    // Гипотеза не подтверждена ПОВЕДЕНИЕМ и потому не принята.
    for (long i = (long)b * THREADS + threadIdx.x; i < NT; i += (long)nb * THREADS) {
        float v = 0.f;
        for (int kk = 0; kk < ks1; ++kk) v += part1[(long)kk * NT + i];
        if (i < Nq) {
            // причинная свёртка: три прошлых значения канала + текущее
            const float w0 = __half2float(conv_w[(long)i * 4 + 0]);
            const float w1 = __half2float(conv_w[(long)i * 4 + 1]);
            const float w2 = __half2float(conv_w[(long)i * 4 + 2]);
            const float w3 = __half2float(conv_w[(long)i * 4 + 3]);
            const float s0 = __half2float(conv_state[(long)i * кш_i + 0 * кш_j]);
            const float s1 = __half2float(conv_state[(long)i * кш_i + 1 * кш_j]);
            const float s2 = __half2float(conv_state[(long)i * кш_i + 2 * кш_j]);
            const float c = w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
            conv_state[(long)i * кш_i + 0 * кш_j] = __float2half(s1);
            conv_state[(long)i * кш_i + 1 * кш_j] = __float2half(s2);
            conv_state[(long)i * кш_i + 2 * кш_j] = __float2half(v);
            mid[i] = __float2half(c / (1.f + __expf(-c)));       // silu
        } else {
            mid[i] = __float2half(v);
        }
    }
    ГБАР(grid);
    // СЛЕПОК ПЛИТЫ: что ядро РЕАЛЬНО держит после проекций и свёртки. Спор о раскладке
    // (сплошная против блоков по k-голове) решается этим, а не чтением чужого кода.
    if (dbg_плита) {
        for (long i = (long)b * THREADS + threadIdx.x; i < NT; i += (long)nb * THREADS)
            dbg_плита[i] = mid[i];
        ГБАР(grid);
    }

    // ---- фаза 3: мешалка. Блок = голова значений, поток = строка состояния.
    // q,k лежат в mid[0..2*Hk*DK), v -- следом; z в mid[Nq..Nq+Nz); a,b в хвосте.
    const int qoff = 0, koff = Hk * DK, voff = 2 * Hk * DK;
    // ТРЕБОВАНИЕ, А НЕ ВЕТКА: мешалке нужно РОВНО DV потоков (поток = строка состояния). Ветка
    // «лишние потоки пропускают» ставила бы __syncthreads в расходящемся пути -- это ЗАВИСАНИЕ
    // блока, а не неверный ответ. Поэтому число потоков проверяется на хозяине.
#ifndef FA2SM70_БЕЗ_МЕШАЛКИ
#ifndef FA2SM70_МЕШАЛКА_ПРЕЖНЯЯ
    // ЗАДАНИЕ = (голова, ГРУППА СТРОК), ВАРП = строка состояния.
    //
    // Прежде задание = голова: 24 задания на 240 блоков, то есть 90% машины стояло. Снятие фазы
    // намерило цену: 20.16 -> 17.65 мс, ДВЕНАДЦАТЬ ПРОЦЕНТОВ цепи за 1.2% её байт. Это не счёт
    // команд и не «мешалка дорогая» -- это НЕЗАНЯТАЯ машина внутри барьера, на котором ждут все.
    //
    // Строки состояния независимы, поэтому дробятся свободно. Единственная связь между ними --
    // норма выхода головы (по всем DV строкам), и она уносится во ВТОРУЮ фазу через рабочую
    // область: частичные суммы квадратов пишутся в СВОЙ слот на группу, без атомарных и без
    // обнуления (каждый слот переписывается каждым слоем).
#ifndef FA2SM70_MIX_RG
#define FA2SM70_MIX_RG 8
#endif
    constexpr int RG = FA2SM70_MIX_RG;          // групп строк на голову: 24*8 = 192 задания
    constexpr int WARPS = THREADS / 32;
    float* const pacc = part1;                  // [Hv*DV] выход + [Hv*RG] частичных сумм квадратов
    for (int job = b; job < Hv * RG; job += nb) {
        const int hv = job / RG, rg = job % RG;
        const int hk = hv / (Hv / Hk);
        __shared__ __align__(16) float sq[256], sk[256], red[256], red1[256], red2[256];
        const int t = threadIdx.x;
        float qv = __half2float(mid[qoff + (long)hk * DK + t]);
        float kv = __half2float(mid[koff + (long)hk * DK + t]);
        // ТРИ СУММЫ ОДНИМ ПРОХОДОМ. Барьер блока -- это не пометка, а ОСТАНОВКА всех 128 нитей;
        // три отдельные свёртки давали 21 барьер на задание, а заданий 192 на слой. Замерено:
        // добавление ОДНОЙ свёртки в преамбулу стоило 0.07 мс на цепь -- значит цена барьеров тут
        // измерима, и три прохода складываются в один без единого лишнего такта арифметики.
#ifndef FA2SM70_МЕШАЛКА_АЛГЕБРОЙ
        red[t] = qv * qv; __syncthreads();
        for (int r = DK >> 1; r; r >>= 1) { if (t < r) red[t] += red[t + r]; __syncthreads(); }
        const float nq = rsqrtf(red[0] + 1e-6f); __syncthreads();
        red[t] = kv * kv; __syncthreads();
        for (int r = DK >> 1; r; r >>= 1) { if (t < r) red[t] += red[t + r]; __syncthreads(); }
        const float nk = rsqrtf(red[0] + 1e-6f); __syncthreads();
#else
        red[t] = qv * qv; red1[t] = kv * kv; red2[t] = qv * kv;
        __syncthreads();
        for (int r = DK >> 1; r; r >>= 1) {
            if (t < r) { red[t] += red[t + r]; red1[t] += red1[t + r]; red2[t] += red2[t + r]; }
            __syncthreads();
        }
        const float nq = rsqrtf(red[0] + 1e-6f);
        const float nk = rsqrtf(red1[0] + 1e-6f);
        // sk*sq = (qv*nq*scale)*(kv*nk), просуммированное по j -- то есть та же третья сумма
        const float sksq = red2[0] * nq * nk * scale;
        __syncthreads();
#endif
        sq[t] = qv * nq * scale; sk[t] = kv * nk;
        __syncthreads();
        // ХОД АЛГЕБРОЙ -- ОПРОВЕРГНУТ ПАРНЫМ ЗАМЕРОМ (11.08.2026), оставлен под
        // FA2SM70_МЕШАЛКА_АЛГЕБРОЙ вместе с этим опровержением.
        //   acc = SUM_j (gd*h[j] + vv*sk[j]) * sq[j] = gd*(SUM_j h[j]sq[j]) + vv*(SUM_j sk[j]sq[j]),
        // и второе слагаемое ОБЩЕЕ для всех строк головы. Значит в цикле по строкам берутся ДВЕ
        // суммы одним проходом (перемежённые shfl -- та же задержка, что у одной), а вторая
        // варповая свёртка, стоявшая ПОСЛЕ записи состояния, исчезает вместе со своей задержкой.
#ifdef FA2SM70_BA_ПРЕЖНИЙ
        const float av = __half2float(mid[NQZ + hv]);
        const float bv = __half2float(mid[NQZ + Nab / 2 + hv]);
#else
        const float bv = __half2float(mid[NQZ + hv]);
        const float av = __half2float(mid[NQZ + Nab / 2 + hv]);
#endif
        const float dt = __half2float(dt_bias[hv]);
        const float xg = av + dt;
        const float sp = (xg <= 20.f) ? log1pf(__expf(xg)) : xg;
        const float gd = __expf(-__expf(a_log[hv]) * sp);
        const float bt = 1.f / (1.f + __expf(-bv));

        const int R = (DV + RG - 1) / RG, wid = t >> 5, lane = t & 31;
        float ss = 0.f;                          // сумма квадратов выхода по СВОИМ строкам
#ifndef FA2SM70_МЕШАЛКА_ОДНА_СТРОКА
        // ДВЕ СТРОКИ В ПОЛЁТЕ НА ВАРП. Мешалка возит 151 МБ за ~0.83 мс = ~180 ГБ/с при
        // достижимых 815 на смешанном потоке 1:1 (замерено: чтение 858, копия 815) -- значит её
        // связывает НЕ полоса, а цепочка «загрузка -> варповая свёртка -> запись» на одну строку:
        // варп стоит на ней ~800 тактов ради 512 байт. Вторая строка идёт независимо и прячет
        // первую. Регистры на это ЕСТЬ: карта показала, что снятие мешалки оставляет REG:168,
        // то есть пик держит не она.
#ifndef FA2SM70_МЕШАЛКА_ПАР
#define FA2SM70_МЕШАЛКА_ПАР 2
#endif
        // N СТРОК В ПОЛЁТЕ. Варп стоит на цепочке «загрузка -> варповая свёртка -> запись» ~800
        // тактов ради 512 байт; независимые строки прячут друг друга. Тело общее по массивам, с
        // маской на хвост -- иначе при R, не кратном ПАР, часть строк молча не посчиталась бы.
#ifndef FA2SM70_МЕШАЛКА_ПАР
#define FA2SM70_МЕШАЛКА_ПАР 2
#endif
        constexpr int ПАР = FA2SM70_МЕШАЛКА_ПАР;
        const long баз = (long)hv * DV * DK;
        const float4 s4 = *reinterpret_cast<const float4*>(&sk[4 * lane]);
        const float4 q4 = *reinterpret_cast<const float4*>(&sq[4 * lane]);
        for (int rr = wid * ПАР; rr < R; rr += WARPS * ПАР) {
            int стр[ПАР]; bool есть[ПАР]; float4 h[ПАР]; float dk[ПАР], ac[ПАР];
            #pragma unroll
            for (int u = 0; u < ПАР; ++u) {
                const int r0 = rr + u;
                стр[u]  = rg * R + r0;
                есть[u] = (r0 < R) && (стр[u] < DV);
                const float* hp = hstate + баз + (long)(есть[u] ? стр[u] : 0) * DK;
                h[u] = *reinterpret_cast<const float4*>(hp + 4 * lane);
                dk[u] = fmaf(h[u].x, s4.x, fmaf(h[u].y, s4.y, fmaf(h[u].z, s4.z, h[u].w * s4.w)));
            }
            #pragma unroll
            for (int r2 = 16; r2; r2 >>= 1) {
                #pragma unroll
                for (int u = 0; u < ПАР; ++u) dk[u] += __shfl_xor_sync(0xffffffffu, dk[u], r2);
            }
            #pragma unroll
            for (int u = 0; u < ПАР; ++u) {
                const int ст = есть[u] ? стр[u] : 0;
                const float vv = bt * (__half2float(mid[voff + (long)hv * DV + ст]) - gd * dk[u]);
                float4 n;
                n.x = fmaf(gd, h[u].x, vv * s4.x); n.y = fmaf(gd, h[u].y, vv * s4.y);
                n.z = fmaf(gd, h[u].z, vv * s4.z); n.w = fmaf(gd, h[u].w, vv * s4.w);
                if (есть[u]) *reinterpret_cast<float4*>(hstate + баз + (long)стр[u] * DK + 4 * lane) = n;
                ac[u] = fmaf(n.x, q4.x, fmaf(n.y, q4.y, fmaf(n.z, q4.z, n.w * q4.w)));
            }
            #pragma unroll
            for (int r2 = 16; r2; r2 >>= 1) {
                #pragma unroll
                for (int u = 0; u < ПАР; ++u) ac[u] += __shfl_xor_sync(0xffffffffu, ac[u], r2);
            }
            if (lane == 0) {
                #pragma unroll
                for (int u = 0; u < ПАР; ++u)
                    if (есть[u]) { pacc[(long)hv * DV + стр[u]] = ac[u]; ss = fmaf(ac[u], ac[u], ss); }
            }
        }
#else
        for (int rr = wid; rr < R; rr += WARPS) {
            const int row = rg * R + rr;
            if (row >= DV) continue;             // ровных групп может не быть -- хвост пропускаем
            float* hrow = hstate + (long)hv * DV * DK + (long)row * DK;
            // Лента строки берётся 32 соседними полосами: обращение варпа сплошное, а свёртка по
            // строке достаётся перестановкой в регистрах, а не разделяемой памятью.
            // 128 бит за обращение: полоса варпа кроет 512 сплошных байт одной командой вместо
            // четырёх. Строка выровнена (DK*4 = 512 Б), доля лица -- 16 Б, значит выравнивание есть.
            float4 hb = *reinterpret_cast<const float4*>(hrow + 4 * lane);
            const float4 s4 = *reinterpret_cast<const float4*>(&sk[4 * lane]);
            const float4 q4 = *reinterpret_cast<const float4*>(&sq[4 * lane]);
            float dk = fmaf(hb.x, s4.x, fmaf(hb.y, s4.y, fmaf(hb.z, s4.z, hb.w * s4.w)));
#ifndef FA2SM70_МЕШАЛКА_АЛГЕБРОЙ
            #pragma unroll
            for (int r2 = 16; r2; r2 >>= 1) dk += __shfl_xor_sync(0xffffffffu, dk, r2);
            const float vv = bt * (__half2float(mid[voff + (long)hv * DV + row]) - gd * dk);
            float4 hn;
            hn.x = fmaf(gd, hb.x, vv * s4.x); hn.y = fmaf(gd, hb.y, vv * s4.y);
            hn.z = fmaf(gd, hb.z, vv * s4.z); hn.w = fmaf(gd, hb.w, vv * s4.w);
            *reinterpret_cast<float4*>(hrow + 4 * lane) = hn;
            float acc = fmaf(hn.x, q4.x, fmaf(hn.y, q4.y, fmaf(hn.z, q4.z, hn.w * q4.w)));
            #pragma unroll
            for (int r2 = 16; r2; r2 >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, r2);
#else
            float hq = fmaf(hb.x, q4.x, fmaf(hb.y, q4.y, fmaf(hb.z, q4.z, hb.w * q4.w)));
            #pragma unroll
            for (int r2 = 16; r2; r2 >>= 1) {           // две суммы перемежены: задержка одной
                dk += __shfl_xor_sync(0xffffffffu, dk, r2);
                hq += __shfl_xor_sync(0xffffffffu, hq, r2);
            }
            const float vv = bt * (__half2float(mid[voff + (long)hv * DV + row]) - gd * dk);
            float4 hn;
            hn.x = fmaf(gd, hb.x, vv * s4.x); hn.y = fmaf(gd, hb.y, vv * s4.y);
            hn.z = fmaf(gd, hb.z, vv * s4.z); hn.w = fmaf(gd, hb.w, vv * s4.w);
            *reinterpret_cast<float4*>(hrow + 4 * lane) = hn;   // запись НЕ ждёт второй свёртки
            const float acc = fmaf(gd, hq, vv * sksq);
#endif
            if (lane == 0) { pacc[(long)hv * DV + row] = acc; ss = fmaf(acc, acc, ss); }
        }
#endif
        __syncthreads();
        red[t] = (lane == 0) ? ss : 0.f;
        __syncthreads();
        if (t == 0) {
            float s = 0.f;
            for (int w = 0; w < THREADS; w += 32) s += red[w];
            pacc[(long)Hv * DV + (long)hv * RG + rg] = s;
        }
        __syncthreads();
    }
    ГБАР(grid);
    // ВТОРАЯ ФАЗА: RMSNormGated выхода мешалки. norm(o)*gnorm*silu(z), norm_before_gate=True,
    // нормируется по DV значениям ОДНОЙ головы -- сумма собирается из RG частичных.
    //
    // ОПРОВЕРГНУТО (11.08.2026): перенос этого прохода в ЗАГРУЗЧИК ПЛИТКИ фазы 4 («отложенная
    // редукция» -- сведение у потребителя) снимает сетевой барьер и целый проход, но даёт РОВНО
    // НОЛЬ: 18.49 против 18.49. Причина названа замером -- у фазы 4 cols2=10 столбцовых плиток, и
    // каждая пере-стадирует ту же активацию, то есть норма считается ДЕСЯТЬ раз вместо одного.
    // Снятый барьер стоит ровно столько же, сколько девять лишних пересчётов.
    for (long i = (long)b * THREADS + threadIdx.x; i < (long)Hv * DV; i += (long)nb * THREADS) {
        const int hvv = (int)(i / DV), t = (int)(i % DV);
        float s = 0.f;
        for (int g2 = 0; g2 < RG; ++g2) s += pacc[(long)Hv * DV + (long)hvv * RG + g2];
        const float no = rsqrtf(s / (float)DV + eps);
        const float on = pacc[i] * no * (gnorm ? __half2float(gnorm[t]) : 1.f);
        const float zv = __half2float(mid[Nq + i]);
        mid[voff + i] = __float2half(on * (zv / (1.f + __expf(-zv))));
    }
#else
    for (int hv = b; hv < Hv; hv += nb) {
        const int hk = hv / (Hv / Hk);
        __shared__ __align__(16) float sq[256], sk[256], red[256];
        const int t = threadIdx.x;
        float qv = __half2float(mid[qoff + (long)hk * DK + t]);
        float kv = __half2float(mid[koff + (long)hk * DK + t]);
        red[t] = qv * qv; __syncthreads();
        for (int r = DK >> 1; r; r >>= 1) { if (t < r) red[t] += red[t + r]; __syncthreads(); }
        const float nq = rsqrtf(red[0] + 1e-6f); __syncthreads();
        red[t] = kv * kv; __syncthreads();
        for (int r = DK >> 1; r; r >>= 1) { if (t < r) red[t] += red[t + r]; __syncthreads(); }
        const float nk = rsqrtf(red[0] + 1e-6f); __syncthreads();
        sq[t] = qv * nq * scale; sk[t] = kv * nk;
        __syncthreads();
        // гейт: g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
        // ПОРЯДОК В ПЛИТЕ ba. Плита сплошная (замерено: слепок против проекции relL2 1.15e-04),
        // но модель делит её как (b, a): `(b, a) = torch.split(mixed_ba, [r, r], dim=2)`.
        // Гейт FA2SM70_BA_КАК_ЕСТЬ=0 берёт порядок модели, =1 -- прежний (a, b).
#ifdef FA2SM70_BA_ПРЕЖНИЙ
        const float av = __half2float(mid[NQZ + hv]);
        const float bv = __half2float(mid[NQZ + Nab / 2 + hv]);
#else
        const float bv = __half2float(mid[NQZ + hv]);
        const float av = __half2float(mid[NQZ + Nab / 2 + hv]);
#endif
        const float dt = __half2float(dt_bias[hv]);
        // softplus С ПОРОГОМ, как у них: при большом аргументе exp переполняется, и правильный
        // предел -- сам аргумент. Без этого гейт даёт inf на редких входах, и ошибка уедет в
        // состояние, которое переносится в следующий токен.
        const float xg = av + dt;
        const float sp = (xg <= 20.f) ? log1pf(__expf(xg)) : xg;
        const float gd = __expf(-__expf(a_log[hv]) * sp);
        const float bt = 1.f / (1.f + __expf(-bv));
        float* hrow = hstate + (long)hv * DV * DK + (long)t * DK;
        float dk = 0.f;
        #pragma unroll 4
        for (int j = 0; j < 128; ++j) dk = fmaf(hrow[j], sk[j], dk);
        const float vv = bt * (__half2float(mid[voff + (long)hv * DV + t]) - gd * dk);
        float acc = 0.f;
        #pragma unroll 4
        for (int j = 0; j < 128; ++j) {
            const float hn = fmaf(gd, hrow[j], vv * sk[j]);
            hrow[j] = hn;
            acc = fmaf(hn, sq[j], acc);
        }
        // RMSNormGated ВЫХОДА МЕШАЛКИ: out = norm(o) * w * silu(z), norm_before_gate=True.
        // Раньше здесь стояло acc*silu(z) -- НОРМЫ НЕ БЫЛО ВОВСЕ, хотя комментарий её обещал.
        // Нормируем по 128 значениям ОДНОЙ ГОЛОВЫ (блок = голова, t = элемент головы).
        __syncthreads();
        red[t] = acc * acc;
        __syncthreads();
        for (int r = DV >> 1; r; r >>= 1) { if (t < r) red[t] += red[t + r]; __syncthreads(); }
        const float no = rsqrtf(red[0] / (float)DV + eps);
        __syncthreads();
        const float on = acc * no * (gnorm ? __half2float(gnorm[t]) : 1.f);
        const float zv = __half2float(mid[Nq + (long)hv * DV + t]);
        mid[voff + (long)hv * DV + t] = __float2half(on * (zv / (1.f + __expf(-zv))));
    }
#endif
#endif
    ГБАР(grid);

    // ---- фаза 4: out_proj из полученного вектора
    const __half* actv = mid + voff;
    constexpr int CPTO = FA2SM70_CPT_OUT;
#ifndef FA2SM70_БЕЗ_OUTPROJ
    for (int job = b; job < cols2 * ks2; job += nb) {
        const int ci = job % cols2, ki = job / cols2;
        const int n0 = (ci * THREADS + threadIdx.x) * CPTO;
        const int kbeg = ki * gps2 * G, kend = min(Kout, kbeg + gps2 * G);
        const bool live = (n0 + CPTO <= K);
        float acc[CPTO];
        #pragma unroll
        for (int c = 0; c < CPTO; ++c) acc[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS) sx[t] = __half2float(actv[k0 + t]);
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPTO>(sx, k0, len, qw_o, qz_o, sc_o, n0, K, G, acc);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPTO; ++c) part2[(long)ki * K + n0 + c] = acc[c];
        }
    }
#endif
    ГБАР(grid);
    // ---- остаточная связь: вход слоя + выход мешалки. Значение уже в регистре, ходить в HBM
    // за ним второй раз незачем -- сумма квадратов для ВТОРОЙ нормы копится тут же.
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
        float v = 0.f;
        for (int kk = 0; kk < ks2; ++kk) v += part2[(long)kk * K + i];
        out[i] = __float2half(v);                      // ЕЩЁ БЕЗ остаточной: сперва обмен
    }
    // барьер здесь НЕ нужен: peer_reduce первым же действием ставит свой.
    peer_reduce<THREADS>(grid, out, plan, K, seq0);
    float ps2 = 0.f;
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
        const float v = __half2float(out[i]) + __half2float(x[i]);
        out[i] = __float2half(v);
        ps2 = fmaf(v, v, ps2);
    }
    // ДЕТЕРМИНИРОВАННОЕ СВЕДЕНИЕ ВМЕСТО АТОМАРНОГО. `atomicAdd` по float складывает в ПОРЯДКЕ
    // ПРИХОДА -- на одном и том же входе при T=0 два прогона расходятся (замерено: два снимка с
    // ОДНОГО сервера совпали 6/8). Слот на блок + сумма в фиксированном порядке даёт ту же
    // работу и повторяемый ответ.
    #pragma unroll
    for (int o2 = 16; o2; o2 >>= 1) ps2 += __shfl_down_sync(0xffffffff, ps2, o2);
    {
        __shared__ float крс[THREADS / 32];
        if ((threadIdx.x & 31) == 0) крс[threadIdx.x >> 5] = ps2;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < THREADS / 32; ++w) s += крс[w];
            part3[(long)ks3 * I2 * 2 + b] = s;
        }
    }
    ГБАР(grid);
#endif
    float _сум = 0.f;
    for (int q = 0; q < nb; ++q) _сум += part3[(long)ks3 * I2 * 2 + q];
    const float rs2 = rsqrtf(_сум / (float)K + eps);

    // ---- тело MLP: gate_up одним проходом по паре срезов + silu*mul
    for (int job = b; job < cols3 * ks3; job += nb) {
        const int ci = job % cols3, ki = job / cols3;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gps3 * G, kend = min(K, kbeg + gps3 * G);
        const bool live = (n0 + CPT <= I2);
        float ag[CPT], au[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) { ag[c] = 0.f; au[c] = 0.f; }
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS)
                sx[t] = __half2float(out[k0 + t]) * rs2 * (1.f + __half2float(rw2[k0 + t]));
            __syncthreads();
            if (!live) continue;
            // ПАРНЫЙ гемв держит ВДВОЕ больше накопителей (sg,su,zg,zu,qg,qu = 6*CPT) ради
            // экономии чтений из разделяемой. Но связывает нас РЕГИСТР, а разделяемая в избытке
            // (12368 из 96K): при 128 регистрах кадр стека 136 байт, и это запирает 4-й блок на SM.
            // Под гейтом пара разворачивается в два одиночных прохода -- тот же ответ, вдвое меньше
            // живых накопителей, вдвое больше (дешёвых) чтений плитки.
#ifdef FA2SM70_БЕЗ_ПАРЫ
            gemv_slice<THREADS, CPT>(sx, k0, len, qw_g2, qz_g2, sc_g2,
                                     n0, Ntot2 ? Ntot2 : I2, G, ag);
            gemv_slice<THREADS, CPT>(sx, k0, len, qw_u2, qz_u2, sc_u2,
                                     n0 + upoff2, Ntot2 ? Ntot2 : I2, G, au);
#else
            gemv_slice_pair<THREADS, CPT>(sx, k0, len, qw_g2, qz_g2, sc_g2,
                                          qw_u2, qz_u2, sc_u2,
                                          n0, n0 + upoff2, Ntot2 ? Ntot2 : I2, G, ag, au);
#endif
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                part3[(long)ki * 2 * I2 + n0 + c]      = ag[c];
                part3[(long)ki * 2 * I2 + I2 + n0 + c] = au[c];
            }
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < I2; i += (long)nb * THREADS) {
        float g2 = 0.f, u2 = 0.f;
        for (int ki = 0; ki < ks3; ++ki) {
            g2 += part3[(long)ki * 2 * I2 + i];
            u2 += part3[(long)ki * 2 * I2 + I2 + i];
        }
        mid2[i] = __float2half((g2 / (1.f + __expf(-g2))) * u2);
    }
    ГБАР(grid);
    for (int job = b; job < cols4 * ks4; job += nb) {          // ---- down
        const int ci = job % cols4, ki = job / cols4;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gps4 * G, kend = min(I2, kbeg + gps4 * G);
        const bool live = (n0 + CPT <= K);
        float ad[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) ad[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS) sx[t] = __half2float(mid2[k0 + t]);
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPT>(sx, k0, len, qw_d2, qz_d2, sc_d2, n0, K, G, ad);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) part4[(long)ki * K + n0 + c] = ad[c];
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
        float v = 0.f;
        for (int kk = 0; kk < ks4; ++kk) v += part4[(long)kk * K + i];
        buf_mlp[i] = __float2half(v);                  // своя половина -- сразу в буфер обмена
    }
    // барьер здесь НЕ нужен: peer_reduce первым же действием ставит свой.
    peer_reduce<THREADS>(grid, buf_mlp, plan, K, seq0 + 1);
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS)
        out[i] = __float2half(__half2float(out[i]) + __half2float(buf_mlp[i]));   // остаточная
}







template <int THREADS, int CPT>
__global__ void gdn_layer_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ rw,
        const int* __restrict__ qw_q,
        const int* __restrict__ qz_q,
        const __half* __restrict__ sc_q,
        const int* __restrict__ qw_z,
        const int* __restrict__ qz_z,
        const __half* __restrict__ sc_z,
        const __half* __restrict__ w_ab,
        const int* __restrict__ qw_o,
        const int* __restrict__ qz_o,
        const __half* __restrict__ sc_o,
        const __half* __restrict__ conv_w,
        __half* __restrict__ conv_state,
        const float* __restrict__ a_log,
        const __half* __restrict__ dt_bias,
        float* __restrict__ hstate,
        const __half* __restrict__ gnorm,
        float* __restrict__ part1,
        __half* __restrict__ mid,
        float* __restrict__ part2,
        __half* __restrict__ out,
        const __half* __restrict__ rw2,
        const int* __restrict__ qw_g2,
        const int* __restrict__ qz_g2,
        const __half* __restrict__ sc_g2,
        const int* __restrict__ qw_u2,
        const int* __restrict__ qz_u2,
        const __half* __restrict__ sc_u2,
        const int* __restrict__ qw_d2,
        const int* __restrict__ qz_d2,
        const __half* __restrict__ sc_d2,
        float* __restrict__ part3,
        __half* __restrict__ mid2,
        float* __restrict__ part4,
        int Nq,
        int Nz,
        int Nab,
        int K,
        int G,
        float eps,
        float scale,
        int cols1,
        int ks1,
        int gps1,
        int cols2,
        int ks2,
        int gps2,
        int Hk,
        int Hv,
        int DK,
        int DV,
        int Kout,
        int I2,
        int cols3,
        int ks3,
        int gps3,
        int cols4,
        int ks4,
        int gps4,
        __half* __restrict__ peer_inbox,
        const __half* __restrict__ my_inbox,
        int* __restrict__ my_flag,
        int* __restrict__ peer_flag,
        int seq0,
        __half* __restrict__ buf_mlp,
        int Nqz_in,
        int upoff_in, int Ntot_in) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float sx[XT2_];
    gdn_layer_body<THREADS, CPT>(grid, sx, x, rw, qw_q, qz_q, sc_q, qw_z, qz_z, sc_z, w_ab, qw_o, qz_o, sc_o, conv_w, conv_state, 3, 1, a_log, dt_bias, hstate, gnorm, nullptr, part1, mid, part2, out, rw2, qw_g2, qz_g2, sc_g2, qw_u2, qz_u2, sc_u2, qw_d2, qz_d2, sc_d2, part3, mid2, part4, Nq, Nz, Nab, K, G, eps, scale, cols1, ks1, gps1, cols2, ks2, gps2, Hk, Hv, DK, DV, Kout, I2, cols3, ks3, gps3, cols4, ks4, gps4, plan_pair(peer_inbox, my_inbox, my_flag, peer_flag, min((int)gridDim.x, FA2SM70_EXCH_W)), seq0, buf_mlp, upoff_in, Ntot_in, Nqz_in);
}

torch::Tensor gdn_layer(torch::Tensor x, torch::Tensor rw, double eps,
                        torch::Tensor qw_q, torch::Tensor qz_q, torch::Tensor sc_q,
                        torch::Tensor qw_z, torch::Tensor qz_z, torch::Tensor sc_z,
                        torch::Tensor w_ab,
                        torch::Tensor qw_o, torch::Tensor qz_o, torch::Tensor sc_o,
                        torch::Tensor conv_w, torch::Tensor conv_state,
                        torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor hstate,
                        torch::Tensor rw2,
                        torch::Tensor qw_g2, torch::Tensor qz_g2, torch::Tensor sc_g2,
                        torch::Tensor qw_u2, torch::Tensor qz_u2, torch::Tensor sc_u2,
                        torch::Tensor qw_d2, torch::Tensor qz_d2, torch::Tensor sc_d2,
                        int64_t group_size, double scale, int64_t Hk, int64_t Hv,
                        int64_t total_width, int64_t q_width,
                        int64_t up_offset, int64_t mlp_width) {
    // СЛИТЫЙ gate_up: подмена ловится в ОБЕ стороны. Без этого ядро считало ширину ВДВОЕ больше
    // и выходило за границу памяти вместо отказа -- это и есть тихая неверность с падением.
    TORCH_CHECK(mlp_width == 0 || (qw_g2.data_ptr() == qw_u2.data_ptr() && up_offset > 0),
                "gdn_layer: mlp_width задан, но gate и up -- разные тензоры или нет смещения");
    TORCH_CHECK(mlp_width != 0 || qw_g2.data_ptr() != qw_u2.data_ptr(),
                "gdn_layer: gate и up -- ОДИН тензор, но mlp_width не задан (посчиталось бы вдвое шире)");
    // СТЫК СОСТОЯНИЙ. vLLM отдаёт conv_state ТРАНСПОНИРОВАННЫМ ВИДОМ (`[0].transpose(-1,-2)`),
    // то есть неплотным. Ядро читает по СВОЕМУ шагу и посчитает не то МОЛЧА -- поэтому отвергаем.
    TORCH_CHECK(conv_state.is_contiguous() && hstate.is_contiguous(),
                "gdn_layer: состояния обязаны быть ПЛОТНЫМИ (conv_state у vLLM -- транспонированный вид)");
    TORCH_CHECK(conv_state.scalar_type() == torch::kHalf && hstate.scalar_type() == torch::kFloat,
                "gdn_layer: conv_state -- fp16, hstate -- fp32");
    TORCH_CHECK(hstate.numel() == (long)Hv * 128 * 128,
                "gdn_layer: hstate не Hv*DV*DK");
    const int upoff_ = (int)up_offset, Ntot_ = (int)mlp_width;
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "gdn_layer: плотный вектор на карте");
    const int K = (int)qw_q.size(0) * 4, G = (int)group_size;
    // СЛИТАЯ ПЛИТА in_proj_qkvz: обе ширины равны полной, поэтому форма НЕ источник геометрии.
    const int Nqz = (int)total_width;
    const int Nq = Nqz ? (int)q_width : (int)qw_q.size(1);
    const int Nz = Nqz ? (Nqz - Nq)   : (int)qw_z.size(1);
    const int Nab = (int)w_ab.size(1);
    TORCH_CHECK(Nqz == 0 || (qw_q.data_ptr() == qw_z.data_ptr() && q_width > 0 && Nq < Nqz),
                "gdn_layer: при слитой плите нужны ОДИН тензор и ширина q");
    TORCH_CHECK(Nqz != 0 || qw_q.data_ptr() != qw_z.data_ptr(),
                "gdn_layer: q и z -- ОДИН тензор, но total_width не задан");
    const int Kout = (int)qw_o.size(0) * 4;
    const int DK = 128, DV = 128, T = 128, CPT = 4;      // T == DV -- требование мешалки
    const int NQZ = Nq + Nz, NT = NQZ + Nab;
    TORCH_CHECK(NQZ % (T * CPT) == 0 && K % (T * CPT) == 0, "gdn_layer: плитки не сходятся");
    TORCH_CHECK(Nab <= 64 && (int)Hv * DV == Nz && (int)Hk * DK * 2 + (int)Hv * DV == Nq,
                "gdn_layer: раскладка qkv/z не сходится с числом голов");
    TORCH_CHECK(Kout == (int)Hv * DV, "gdn_layer: вход out_proj не равен выходу мешалки");
    const at::cuda::CUDAGuard guard(x.device());
    auto st = at::cuda::getCurrentCUDAStream();
    auto fo = torch::dtype(torch::kFloat32).device(x.device());
    auto ho = torch::dtype(torch::kFloat16).device(x.device());
    const int cols1 = NQZ / (T * CPT), cols2 = K / (T * CPT);
    const int ng1 = (K + G - 1) / G, ng2 = (Kout + G - 1) / G;
    const void* kfn = (const void*)gdn_layer_kernel<128, 4>;
    int perSM = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kfn, T, XT2_ * sizeof(float)));
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int nblk = max(1, perSM * nsm);
    // ПРОВЕРКА ГИПОТЕЗЫ: если разница цепи и обёртки -- лишь ДРУГОЕ ДРОБЛЕНИЕ суммы, то при
    // одинаковом числе блоков она обязана исчезнуть. Гейт только для замера, умолчание прежнее.
    if (const char* e_ = getenv("FA2SM70_NBLK")) nblk = atoi(e_);
    int ks1 = max(1, min(ng1, (nblk + cols1 - 1) / cols1));
    const int gps1 = (ng1 + ks1 - 1) / ks1; ks1 = (ng1 + gps1 - 1) / gps1;
    int ks2 = max(1, min(ng2, (nblk + cols2 - 1) / cols2));
    const int gps2 = (ng2 + ks2 - 1) / ks2; ks2 = (ng2 + gps2 - 1) / gps2;
    const int I2 = (mlp_width ? (int)(mlp_width/2) : (int)qw_g2.size(1));
    const int cols3 = I2 / (T * CPT), cols4 = K / (T * CPT);
    const int ng3 = (K + G - 1) / G, ng4 = (I2 + G - 1) / G;
    TORCH_CHECK(I2 % (T * CPT) == 0, "gdn_layer: промежуточная ось MLP не кратна плитке");
    int ks3 = max(1, min(ng3, (nblk + cols3 - 1) / cols3));
    const int gps3 = (ng3 + ks3 - 1) / ks3; ks3 = (ng3 + gps3 - 1) / gps3;
    int ks4 = max(1, min(ng4, (nblk + cols4 - 1) / cols4));
    const int gps4 = (ng4 + ks4 - 1) / ks4; ks4 = (ng4 + gps4 - 1) / gps4;
    // ФАЛЬСИФИКАТОР ПОКАЗАЛ: засорение памяти перед вызовом МЕНЯЕТ ответ, значит область
    // читается раньше записи. Обнуляем и меряем -- если расхождение исчезнет, причина найдена.
    auto part1 = torch::zeros({ks1, NT}, fo), part2 = torch::zeros({ks2, K}, fo);
    // ХВОСТОВОЙ СЛОТ под сумму квадратов второй нормы копится атомарно -- значит обнуляется.
    auto part3 = torch::zeros({ks3 * 2 * I2 + 1024}, fo), part4 = torch::zeros({ks4, K}, fo);
    auto mid2 = torch::zeros({I2}, ho);
    auto bmlp = torch::zeros({K}, ho);
    auto mid = torch::zeros({NT}, ho), out = torch::zeros({1, K}, ho);
    auto H2 = [](torch::Tensor t){ return reinterpret_cast<const __half*>(t.data_ptr<at::Half>()); };
    const __half *px=H2(x), *prw=H2(rw), *psq=H2(sc_q), *psz=H2(sc_z), *pab=H2(w_ab),
                 *pso=H2(sc_o), *pcw=H2(conv_w), *pdt=H2(dt_bias);
    __half *pcs = reinterpret_cast<__half*>(conv_state.data_ptr<at::Half>());
    __half *pmid = reinterpret_cast<__half*>(mid.data_ptr<at::Half>());
    __half *po = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    int *pwq=qw_q.data_ptr<int>(), *pzq=qz_q.data_ptr<int>(), *pwz=qw_z.data_ptr<int>(),
        *pzz=qz_z.data_ptr<int>(), *pwo=qw_o.data_ptr<int>(), *pzo=qz_o.data_ptr<int>();
    float *pal=a_log.data_ptr<float>(), *phs=hstate.data_ptr<float>();
    const __half *prw2=H2(rw2), *psg2=H2(sc_g2), *psu2=H2(sc_u2), *psd2=H2(sc_d2);
    int *pwg2=qw_g2.data_ptr<int>(), *pzg2=qz_g2.data_ptr<int>();
    int *pwu2=qw_u2.data_ptr<int>(), *pzu2=qz_u2.data_ptr<int>();
    int *pwd2=qw_d2.data_ptr<int>(), *pzd2=qz_d2.data_ptr<int>();
    float *pp3=part3.data_ptr<float>(), *pp4=part4.data_ptr<float>();
    __half *pmid2=reinterpret_cast<__half*>(mid2.data_ptr<at::Half>());
    float *pp1=part1.data_ptr<float>(), *pp2=part2.data_ptr<float>();
    float feps=(float)eps, fsc=(float)scale;
    int hk=(int)Hk, hv=(int)Hv, dk=DK, dv=DV;
    __half* nullpx=nullptr; const __half* nullcpx=nullptr;
    int *nullip=nullptr, *nullip2=nullptr; int zseq=0;
    __half* pbmlp=reinterpret_cast<__half*>(bmlp.data_ptr<at::Half>());
    void* ka[] = {(void*)&px,(void*)&prw,(void*)&pwq,(void*)&pzq,(void*)&psq,
                  (void*)&pwz,(void*)&pzz,(void*)&psz,(void*)&pab,
                  (void*)&pwo,(void*)&pzo,(void*)&pso,(void*)&pcw,(void*)&pcs,
                  (void*)&pal,(void*)&pdt,(void*)&phs,
                  (void*)&pp1,(void*)&pmid,(void*)&pp2,(void*)&po,
                  (void*)&prw2,(void*)&pwg2,(void*)&pzg2,(void*)&psg2,
                  (void*)&pwu2,(void*)&pzu2,(void*)&psu2,
                  (void*)&pwd2,(void*)&pzd2,(void*)&psd2,
                  (void*)&pp3,(void*)&pmid2,(void*)&pp4,
                  (void*)&Nq,(void*)&Nz,(void*)&Nab,(void*)&K,(void*)&G,(void*)&feps,(void*)&fsc,
                  (void*)&cols1,(void*)&ks1,(void*)&gps1,
                  (void*)&cols2,(void*)&ks2,(void*)&gps2,
                  (void*)&hk,(void*)&hv,(void*)&dk,(void*)&dv,(void*)&Kout,
                  (void*)&I2,(void*)&cols3,(void*)&ks3,(void*)&gps3,
                  (void*)&cols4,(void*)&ks4,(void*)&gps4,
                  (void*)&nullpx,(void*)&nullcpx,(void*)&nullip,(void*)&nullip2,
                  (void*)&zseq,(void*)&pbmlp,(void*)&Nqz,(void*)&upoff_,(void*)&Ntot_};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kfn, dim3(nblk), dim3(T), ka, XT2_ * sizeof(float), st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ================================================================= СБОРКА: СЛОЙ ВНИМАНИЯ ОДНИМ ПУСКОМ
// Норма -> qkv_proj -> ядро внимания (онлайн-softmax, срезы по ключам) -> сведение срезов -> o_proj.
// Число потоков ЗАДАНО ядром внимания: ровно d=256, поток на элемент головы. Поэтому плитка
// столбцов проекций здесь 256*CPT, а не 128*CPT, как в слое GDN -- геометрия блока диктуется
// САМОЙ СВЯЗНОЙ фазой, остальные под неё подстраиваются.
template <int THREADS, int CPT>
__device__ void attn_layer_body(cg::grid_group& grid, float* __restrict__ sx,
                                const __half* __restrict__ x, const __half* __restrict__ rw,
                                  const int* __restrict__ qw_p, const int* __restrict__ qz_p,
                                  const __half* __restrict__ sc_p,
                                  const int* __restrict__ qw_o, const int* __restrict__ qz_o,
                                  const __half* __restrict__ sc_o,
                                  const __half* __restrict__ kc, const __half* __restrict__ vc,
                                  uint8_t* __restrict__ kp8, uint8_t* __restrict__ vp8,
                                  const int* __restrict__ btab, long blk_stride, int bsz,
                                  float kscale, float vscale,
                                  float* __restrict__ part1, __half* __restrict__ mid,
                                  float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                                  float* __restrict__ part2, __half* __restrict__ out,
                                  const __half* __restrict__ rw2,
                                  const int* __restrict__ qw_g2, const int* __restrict__ qz_g2,
                                  const __half* __restrict__ sc_g2,
                                  const int* __restrict__ qw_u2, const int* __restrict__ qz_u2,
                                  const __half* __restrict__ sc_u2,
                                  const int* __restrict__ qw_d2, const int* __restrict__ qz_d2,
                                  const __half* __restrict__ sc_d2,
                                  float* __restrict__ part3, __half* __restrict__ mid2,
                                  float* __restrict__ part4,
                                  int Nqkv, int K, int G, float eps, float scale,
                                  int cols1, int ks1, int gps1,
                                  int cols2, int ks2, int gps2,
                                  int H, int Hkv, int D, int Sk, int nsp, int Kout,
                                  int I2, int cols3, int ks3, int gps3,
                                  int cols4, int ks4, int gps4,
                                  const PeerPlan& plan,
                                  int seq0, __half* __restrict__ buf_mlp,
                                  int kv_pos,
                                  const __half* __restrict__ dense_qkv,
                                  const __half* __restrict__ q_norm, const __half* __restrict__ k_norm,
                                  const float* __restrict__ cs, int rot,
                                  int upoff2, int Ntot2, __half* __restrict__ dbg) {
    const int nb = gridDim.x, b = blockIdx.x;
    const float rs = rms_scale2<THREADS>(x, K, eps);

#ifdef FA2SM70_ТОЛЬКО_MLP
    (void)rs;                                   // фальсификатор: снята вся часть внимания
    {
        float ps2 = 0.f;
        for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
            const float v = __half2float(x[i]);
            out[i] = __float2half(v);
            ps2 = fmaf(v, v, ps2);
        }
        #pragma unroll
        for (int o2 = 16; o2; o2 >>= 1) ps2 += __shfl_down_sync(0xffffffff, ps2, o2);
        if ((threadIdx.x & 31) == 0) atomicAdd(&part3[(long)ks3 * I2 * 2], ps2);
    }
    ГБАР(grid);
#else
    // ПЛОТНЫЙ qkv: у боевой сети он НЕ квантован (разведка по живой модели). Тогда частичные суммы
    // не нужны вовсе -- каждый столбец считается одним варпом за один проход по K.
    if (dense_qkv) {
        const int per = (Nqkv + nb - 1) / nb;
        gemv_dense_norm<THREADS>(x, rw, rs, K, dense_qkv, b * per, min(Nqkv, (b + 1) * per), mid);
        ГБАР(grid);
    } else {
    for (int job = b; job < cols1 * ks1; job += nb) {          // ---- норма + qkv
        const int ci = job % cols1, ki = job / cols1;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gps1 * G, kend = min(K, kbeg + gps1 * G);
        const bool live = (n0 + CPT <= Nqkv);
        float acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS)
                sx[t] = __half2float(x[k0 + t]) * rs * (1.f + __half2float(rw[k0 + t]));
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPT>(sx, k0, len, qw_p, qz_p, sc_p, n0, Nqkv, G, acc);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) part1[(long)ki * Nqkv + n0 + c] = acc[c];
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < Nqkv; i += (long)nb * THREADS) {
        float v = 0.f;
        for (int kk = 0; kk < ks1; ++kk) v += part1[(long)kk * Nqkv + i];
        mid[i] = __float2half(v);
    }
    }                                        // конец ветви "квантованный qkv"
    ГБАР(grid);

    // РАЗБОР qkv ТЕМ ЖЕ КОДОМ, что и проверенный одиночный узел: q/гейт чередуются ПО ГОЛОВЕ,
    // нормы Gemma-стиля (1+w), вращение частичное. part1 больше не нужен -- он и есть рабочая область.
    if (q_norm) qkv_post_grid<THREADS>(grid, mid, part1, q_norm, k_norm, cs, H, Hkv, D, rot, eps, nb, b);

    // ЗАПИСЬ K/V ТЕКУЩЕГО ТОКЕНА В СКЛАД. Позиция передаётся ЧИСЛОМ, а не выводится из Sk:
    // вывод позиции из длины -- ровно та догадка, которая сегодня стоила пяти неверных предпосылок.
    // kv_pos < 0 означает "склад наполняет вызывающий", как было раньше.
    if (kv_pos >= 0 && kp8) {
        const int lb = kv_pos / bsz, off = kv_pos - lb * bsz;
        const long kbase = (long)btab[lb] * blk_stride + (long)off * Hkv * D;
        const float ik = 1.f / kscale, iv = 1.f / vscale;
        for (int i = b * THREADS + threadIdx.x; i < Hkv * D; i += nb * THREADS) {
            const int hh2 = i / D, ii = i - hh2 * D;
            const long dst = kbase + (long)hh2 * D + ii;
            kp8[dst] = f2e4m3(__half2float(mid[(long)H * D + i]) * ik);
            vp8[dst] = f2e4m3(__half2float(mid[(long)H * D + Hkv * D + i]) * iv);
        }
        ГБАР(grid);
    }

    // ---- ядро внимания. q лежит в mid[0..H*D); K/V приходят снаружи (склад -- следующий шаг).
    // ГЕОМЕТРИЯ СВЕДЕНА К ОДНОЙ: ЭПТ элементов головы на поток вместо «поток на элемент». Иначе
    // слой внимания требует 256 потоков, а мешалка GDN -- 128, и один пуск не может обслужить оба
    // типа слоя, то есть цепочка из 64 слоёв в одном запуске невозможна В ПРИНЦИПЕ.
    // КЛЮЧ ПРИНАДЛЕЖИТ ВАРПУ, а не блоку. Было: полная блочная редукция с __syncthreads НА КАЖДЫЙ
    // ключ -- 7 барьеров x (Sk/nsp) на блок, при Sk=50K это ~117 тысяч ПОСЛЕДОВАТЕЛЬНЫХ барьеров.
    // Снятие фазы (FA2SM70_KEYCAP) замерило её долю: 136 мс из 166 мс ядра, 82%.
    // Стало: редукция ВНУТРИ варпа (__shfl_xor, 5 шагов, НОЛЬ барьеров), у каждого варпа свой
    // онлайн-softmax, сведение варпов ОДИН раз на задание. Заодно сняты две фазы, которых в
    // алгоритме нет: номер страницы берётся ПРИРАЩЕНИЕМ (шаг варпа WARPS < bsz => граница
    // пересекается не чаще раза за итерацию, целочисленного деления нет вовсе), и восемь
    // однобайтовых чтений склада заменены ОДНИМ 64-разрядным.
    constexpr int WARPS = THREADS / 32;         // 4 при THREADS=128
    constexpr int EPW   = 8;                    // D / 32 при D=256: лэйн держит 8 подряд идущих
    __shared__ float red[256], wm_[WARPS], wl_[WARPS];
#ifdef FA2SM70_Q_В_ОБЩЕЙ
    // q В РАЗДЕЛЯЕМОЙ, А НЕ В РЕГИСТРАХ: восемь регистров на нить -- цена входа в четвёртый блок
    // на SM. Раскладка ТРАНСПОНИРОВАННАЯ (sq[e*32+lane]): при постоянном e лэйны читают подряд,
    // то есть по разным банкам. Прямая раскладка sq[lane*8+e] дала бы восьмикратный конфликт.
    __shared__ float sq_[256];
#endif
    const int lane = (int)(threadIdx.x & 31), wid = (int)(threadIdx.x >> 5);
    for (int job = b; job < H * nsp; job += nb) {
        const int hh = job % H, sp = job / H;
        const int hkv = hh / (H / Hkv);
        float acc[EPW];
#ifndef FA2SM70_Q_В_ОБЩЕЙ
        float qv_[EPW];
#endif
        #pragma unroll
        // МАСШТАБ СКЛАДА -- В q, ОДИН РАЗ НА ЗАДАНИЕ. Тогда во внутреннем цикле на элемент
        // остаётся один fmaf, а не «умножить на масштаб и потом fmaf».
        const float _kq = kp8 ? (scale * kscale * 256.f) : scale;
#ifdef FA2SM70_Q_В_ОБЩЕЙ
        for (int i = threadIdx.x; i < D; i += THREADS)
            sq_[(i % EPW) * 32 + (i / EPW)] = __half2float(mid[(long)hh * D + i]) * _kq;
        __syncthreads();
        #define qv(e) sq_[(e) * 32 + lane]
        #pragma unroll
        for (int e = 0; e < EPW; ++e) acc[e] = 0.f;
#else
        #define qv(e) qv_[e]
        for (int e = 0; e < EPW; ++e) {
            qv_[e]  = __half2float(mid[(long)hh * D + lane * EPW + e]) * _kq;
            acc[e] = 0.f;
        }
#endif
        float m = -INFINITY, l = 0.f;
        const int beg = (int)((long)sp * Sk / nsp), end = (int)((long)(sp + 1) * Sk / nsp);
        int s2 = beg + wid;
        int lb = 0, off = 0; long pbase = 0;
        if (kp8 && s2 < end) { lb = s2 / bsz; off = s2 - lb * bsz; pbase = (long)btab[lb] * blk_stride; }
        for (; s2 < end; s2 += WARPS) {
            float dot = 0.f;
            unsigned v0 = 0u, v1 = 0u, v2 = 0u, v3 = 0u;   // V держим УПАКОВАННЫМ через редукцию
            if (kp8) {
                const long o = pbase + ((long)off * Hkv + hkv) * D + lane * EPW;
                const uint2 kw = *reinterpret_cast<const uint2*>(kp8 + o);
                const uint2 vw = *reinterpret_cast<const uint2*>(vp8 + o);
                v0 = vw.x; v1 = vw.y;
                off += WARPS;                     // страницу берём ТОЛЬКО если итерация ещё будет:
                if (off >= bsz) {                 // иначе btab читается на элемент за концом
                    off -= bsz; ++lb;
                    if (s2 + WARPS < end) pbase = (long)btab[lb] * blk_stride;
                }
                uint32_t k01, k23, k45, k67;
                expand_e4m3x4(kw.x, k01, k23);      // элементы 0..3 (пара в слове: младшая, старшая)
                expand_e4m3x4(kw.y, k45, k67);      // элементы 4..7
                dot = fmaf(qv(0), h_lo(k01), dot); dot = fmaf(qv(1), h_hi(k01), dot);
                dot = fmaf(qv(2), h_lo(k23), dot); dot = fmaf(qv(3), h_hi(k23), dot);
                dot = fmaf(qv(4), h_lo(k45), dot); dot = fmaf(qv(5), h_hi(k45), dot);
                dot = fmaf(qv(6), h_lo(k67), dot); dot = fmaf(qv(7), h_hi(k67), dot);
            } else {
                const long o = ((long)s2 * Hkv + hkv) * D + lane * EPW;
                const uint4 kw = *reinterpret_cast<const uint4*>(kc + o);
                const uint4 vw = *reinterpret_cast<const uint4*>(vc + o);
                v0 = vw.x; v1 = vw.y; v2 = vw.z; v3 = vw.w;
                dot = fmaf(qv(0), h_lo(kw.x), dot); dot = fmaf(qv(1), h_hi(kw.x), dot);
                dot = fmaf(qv(2), h_lo(kw.y), dot); dot = fmaf(qv(3), h_hi(kw.y), dot);
                dot = fmaf(qv(4), h_lo(kw.z), dot); dot = fmaf(qv(5), h_hi(kw.z), dot);
                dot = fmaf(qv(6), h_lo(kw.w), dot); dot = fmaf(qv(7), h_hi(kw.w), dot);
            }
            #pragma unroll
            for (int r = 16; r; r >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, r);
            const float mn = fmaxf(m, dot), cor = __expf(m - mn), p2 = __expf(dot - mn);
            l = l * cor + p2;
            if (kp8) {
                const float p2v = p2 * vscale * 256.f;           // масштаб -- в вес, не в элемент
                uint32_t w01, w23, w45, w67;
                expand_e4m3x4(v0, w01, w23);
                expand_e4m3x4(v1, w45, w67);
                acc[0] = fmaf(p2v, h_lo(w01), acc[0] * cor); acc[1] = fmaf(p2v, h_hi(w01), acc[1] * cor);
                acc[2] = fmaf(p2v, h_lo(w23), acc[2] * cor); acc[3] = fmaf(p2v, h_hi(w23), acc[3] * cor);
                acc[4] = fmaf(p2v, h_lo(w45), acc[4] * cor); acc[5] = fmaf(p2v, h_hi(w45), acc[5] * cor);
                acc[6] = fmaf(p2v, h_lo(w67), acc[6] * cor); acc[7] = fmaf(p2v, h_hi(w67), acc[7] * cor);
            } else {
                acc[0] = fmaf(p2, h_lo(v0), acc[0] * cor); acc[1] = fmaf(p2, h_hi(v0), acc[1] * cor);
                acc[2] = fmaf(p2, h_lo(v1), acc[2] * cor); acc[3] = fmaf(p2, h_hi(v1), acc[3] * cor);
                acc[4] = fmaf(p2, h_lo(v2), acc[4] * cor); acc[5] = fmaf(p2, h_hi(v2), acc[5] * cor);
                acc[6] = fmaf(p2, h_lo(v3), acc[6] * cor); acc[7] = fmaf(p2, h_hi(v3), acc[7] * cor);
            }
            m = mn;
        }
        // ---- сведение варпов: ОДИН раз на задание, а не на ключ.
        if (lane == 0) { wm_[wid] = m; wl_[wid] = l; }
        for (int i = threadIdx.x; i < D; i += THREADS) red[i] = 0.f;
        __syncthreads();
        float gm = -INFINITY;
        #pragma unroll
        for (int w2 = 0; w2 < WARPS; ++w2) gm = fmaxf(gm, wm_[w2]);
        float gl = 0.f;
        #pragma unroll
        for (int w2 = 0; w2 < WARPS; ++w2)                    // пустой варп даёт inf-inf=NaN: гасим явно
            gl = fmaf(wl_[w2], (wm_[w2] == -INFINITY) ? 0.f : __expf(wm_[w2] - gm), gl);
        const float cw = (m == -INFINITY) ? 0.f : __expf(m - gm);
        #pragma unroll
        for (int w2 = 0; w2 < WARPS; ++w2) {
            if (wid == w2) {
                #pragma unroll
                for (int e = 0; e < EPW; ++e) red[lane * EPW + e] = fmaf(acc[e], cw, red[lane * EPW + e]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) { pm[(long)sp * H + hh] = gm; pl[(long)sp * H + hh] = gl; }
        for (int i = threadIdx.x; i < D; i += THREADS) pacc[((long)sp * H + hh) * D + i] = red[i];
        __syncthreads();
    }
    #undef qv
    ГБАР(grid);
    for (int hh = b; hh < H; hh += nb) {                       // ---- сведение срезов
        const int t = threadIdx.x;
        float m = -INFINITY;
        for (int sp = 0; sp < nsp; ++sp) m = fmaxf(m, pm[(long)sp * H + hh]);
        float l = 0.f, ac[2] = {0.f, 0.f};
        for (int sp = 0; sp < nsp; ++sp) {
            const float w = __expf(pm[(long)sp * H + hh] - m);
            l += pl[(long)sp * H + hh] * w;
            #pragma unroll
            for (int e = 0; e < 2; ++e) ac[e] += pacc[((long)sp * H + hh) * D + t * 2 + e] * w;
        }
        #pragma unroll
        // ВЫХОД УМНОЖАЕТСЯ НА sigmoid(ГЕЙТ) -- он уже посчитан разбором и лежит за k/v.
        const long og = (long)H * D + 2L * Hkv * D;
        for (int e = 0; e < 2; ++e) {
            float y = ac[e] / l;
            if (q_norm) y *= __half2float(mid[og + (long)hh * D + t * 2 + e]);
            mid[(long)hh * D + t * 2 + e] = __float2half(y);
        }
    }
    ГБАР(grid);

    // РАЗРЕЗ: слепок входа o_proj. Делит слой пополам ОДНИМ замером.
    if (dbg) {
        for (long i2 = (long)b * THREADS + threadIdx.x; i2 < (long)Nqkv; i2 += (long)nb * THREADS)
            dbg[i2] = mid[i2];
        if (b == 0 && threadIdx.x == 0) {          // ЗАМЕРИТЬ делитель, а не гадать о нём
            dbg[Nqkv + 0] = __float2half(pm[0]);
            dbg[Nqkv + 1] = __float2half(pl[0]);
            dbg[Nqkv + 2] = __float2half((float)nsp);
            dbg[Nqkv + 3] = __float2half(pacc[0]);
        }
        ГБАР(grid);
    }
    constexpr int CPTO = FA2SM70_CPT_OUT;
    for (int job = b; job < cols2 * ks2; job += nb) {          // ---- o_proj
        const int ci = job % cols2, ki = job / cols2;
        const int n0 = (ci * THREADS + threadIdx.x) * CPTO;
        const int kbeg = ki * gps2 * G, kend = min(Kout, kbeg + gps2 * G);
        const bool live = (n0 + CPTO <= K);
        float acc[CPTO];
        #pragma unroll
        for (int c = 0; c < CPTO; ++c) acc[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS) sx[t] = __half2float(mid[k0 + t]);
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPTO>(sx, k0, len, qw_o, qz_o, sc_o, n0, K, G, acc);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPTO; ++c) part2[(long)ki * K + n0 + c] = acc[c];
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
        float v = 0.f;
        for (int kk = 0; kk < ks2; ++kk) v += part2[(long)kk * K + i];
        out[i] = __float2half(v);                      // ЕЩЁ БЕЗ остаточной: сперва обмен
    }
    // барьер здесь НЕ нужен: peer_reduce первым же действием ставит свой.
    peer_reduce<THREADS>(grid, out, plan, K, seq0);
    float ps2 = 0.f;
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
        const float v = __half2float(out[i]) + __half2float(x[i]);
        out[i] = __float2half(v);
        ps2 = fmaf(v, v, ps2);
    }
    #pragma unroll
    for (int o2 = 16; o2; o2 >>= 1) ps2 += __shfl_down_sync(0xffffffff, ps2, o2);
    {                                        // см. пояснение в теле GDN: порядок слагаемых ФИКСИРОВАН
        __shared__ float крс[THREADS / 32];
        if ((threadIdx.x & 31) == 0) крс[threadIdx.x >> 5] = ps2;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < THREADS / 32; ++w) s += крс[w];
            part3[(long)ks3 * I2 * 2 + b] = s;
        }
    }
    ГБАР(grid);
#endif
    float _сум = 0.f;
    for (int q = 0; q < nb; ++q) _сум += part3[(long)ks3 * I2 * 2 + q];
    const float rs2 = rsqrtf(_сум / (float)K + eps);
    for (int job = b; job < cols3 * ks3; job += nb) {           // ---- gate_up + silu*mul
        const int ci = job % cols3, ki = job / cols3;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gps3 * G, kend = min(K, kbeg + gps3 * G);
        const bool live = (n0 + CPT <= I2);
        float ag[CPT], au[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) { ag[c] = 0.f; au[c] = 0.f; }
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS)
                sx[t] = __half2float(out[k0 + t]) * rs2 * (1.f + __half2float(rw2[k0 + t]));
            __syncthreads();
            if (!live) continue;
            // ПАРНЫЙ гемв держит ВДВОЕ больше накопителей (sg,su,zg,zu,qg,qu = 6*CPT) ради
            // экономии чтений из разделяемой. Но связывает нас РЕГИСТР, а разделяемая в избытке
            // (12368 из 96K): при 128 регистрах кадр стека 136 байт, и это запирает 4-й блок на SM.
            // Под гейтом пара разворачивается в два одиночных прохода -- тот же ответ, вдвое меньше
            // живых накопителей, вдвое больше (дешёвых) чтений плитки.
#ifdef FA2SM70_БЕЗ_ПАРЫ
            gemv_slice<THREADS, CPT>(sx, k0, len, qw_g2, qz_g2, sc_g2,
                                     n0, Ntot2 ? Ntot2 : I2, G, ag);
            gemv_slice<THREADS, CPT>(sx, k0, len, qw_u2, qz_u2, sc_u2,
                                     n0 + upoff2, Ntot2 ? Ntot2 : I2, G, au);
#else
            gemv_slice_pair<THREADS, CPT>(sx, k0, len, qw_g2, qz_g2, sc_g2,
                                          qw_u2, qz_u2, sc_u2,
                                          n0, n0 + upoff2, Ntot2 ? Ntot2 : I2, G, ag, au);
#endif
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                part3[(long)ki * 2 * I2 + n0 + c]      = ag[c];
                part3[(long)ki * 2 * I2 + I2 + n0 + c] = au[c];
            }
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < I2; i += (long)nb * THREADS) {
        float g2 = 0.f, u2 = 0.f;
        for (int ki = 0; ki < ks3; ++ki) {
            g2 += part3[(long)ki * 2 * I2 + i];
            u2 += part3[(long)ki * 2 * I2 + I2 + i];
        }
        mid2[i] = __float2half((g2 / (1.f + __expf(-g2))) * u2);
    }
    ГБАР(grid);
    for (int job = b; job < cols4 * ks4; job += nb) {           // ---- down
        const int ci = job % cols4, ki = job / cols4;
        const int n0 = (ci * THREADS + threadIdx.x) * CPT;
        const int kbeg = ki * gps4 * G, kend = min(I2, kbeg + gps4 * G);
        const bool live = (n0 + CPT <= K);
        float ad[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) ad[c] = 0.f;
        for (int k0 = kbeg; k0 < kend; k0 += XT2_) {
            const int len = min(XT2_, kend - k0);
            __syncthreads();
            for (int t = threadIdx.x; t < len; t += THREADS) sx[t] = __half2float(mid2[k0 + t]);
            __syncthreads();
            if (!live) continue;
            gemv_slice<THREADS, CPT>(sx, k0, len, qw_d2, qz_d2, sc_d2, n0, K, G, ad);
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < CPT; ++c) part4[(long)ki * K + n0 + c] = ad[c];
        }
    }
    ГБАР(grid);
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS) {
        float v = 0.f;
        for (int kk = 0; kk < ks4; ++kk) v += part4[(long)kk * K + i];
        buf_mlp[i] = __float2half(v);
    }
    // барьер здесь НЕ нужен: peer_reduce первым же действием ставит свой.
    peer_reduce<THREADS>(grid, buf_mlp, plan, K, seq0 + 1);
    for (long i = (long)b * THREADS + threadIdx.x; i < K; i += (long)nb * THREADS)
        out[i] = __float2half(__half2float(out[i]) + __half2float(buf_mlp[i]));
}






template <int THREADS, int CPT>
__global__ void attn_layer_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ rw,
        const int* __restrict__ qw_p,
        const int* __restrict__ qz_p,
        const __half* __restrict__ sc_p,
        const int* __restrict__ qw_o,
        const int* __restrict__ qz_o,
        const __half* __restrict__ sc_o,
        const __half* __restrict__ kc,
        const __half* __restrict__ vc,
        uint8_t* __restrict__ kp8,
        uint8_t* __restrict__ vp8,
        const int* __restrict__ btab,
        long blk_stride,
        int bsz,
        float kscale,
        float vscale,
        float* __restrict__ part1,
        __half* __restrict__ mid,
        float* __restrict__ pm,
        float* __restrict__ pl,
        float* __restrict__ pacc,
        float* __restrict__ part2,
        __half* __restrict__ out,
        const __half* __restrict__ rw2,
        const int* __restrict__ qw_g2,
        const int* __restrict__ qz_g2,
        const __half* __restrict__ sc_g2,
        const int* __restrict__ qw_u2,
        const int* __restrict__ qz_u2,
        const __half* __restrict__ sc_u2,
        const int* __restrict__ qw_d2,
        const int* __restrict__ qz_d2,
        const __half* __restrict__ sc_d2,
        float* __restrict__ part3,
        __half* __restrict__ mid2,
        float* __restrict__ part4,
        int Nqkv,
        int K,
        int G,
        float eps,
        float scale,
        int cols1,
        int ks1,
        int gps1,
        int cols2,
        int ks2,
        int gps2,
        int H,
        int Hkv,
        int D,
        int Sk,
        int nsp,
        int Kout,
        int I2,
        int cols3,
        int ks3,
        int gps3,
        int cols4,
        int ks4,
        int gps4,
        __half* __restrict__ peer_inbox,
        const __half* __restrict__ my_inbox,
        int* __restrict__ my_flag,
        int* __restrict__ peer_flag,
        int seq0,
        __half* __restrict__ buf_mlp,
        int kv_pos,
        const __half* __restrict__ dense_qkv,
        const __half* __restrict__ q_norm, const __half* __restrict__ k_norm,
        const float* __restrict__ cs, int rot, __half* __restrict__ dbg,
        int upoff_in, int Ntot_in) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float sx[XT2_];
    attn_layer_body<THREADS, CPT>(grid, sx, x, rw, qw_p, qz_p, sc_p, qw_o, qz_o, sc_o, kc, vc, kp8, vp8, btab, blk_stride, bsz, kscale, vscale, part1, mid, pm, pl, pacc, part2, out, rw2, qw_g2, qz_g2, sc_g2, qw_u2, qz_u2, sc_u2, qw_d2, qz_d2, sc_d2, part3, mid2, part4, Nqkv, K, G, eps, scale, cols1, ks1, gps1, cols2, ks2, gps2, H, Hkv, D, Sk, nsp, Kout, I2, cols3, ks3, gps3, cols4, ks4, gps4, plan_pair(peer_inbox, my_inbox, my_flag, peer_flag, min((int)gridDim.x, FA2SM70_EXCH_W)), seq0, buf_mlp, kv_pos, dense_qkv, q_norm, k_norm, cs, rot, upoff_in, Ntot_in, dbg);
}

torch::Tensor attn_layer(torch::Tensor x, torch::Tensor rw, double eps,
                         torch::Tensor qw_p, torch::Tensor qz_p, torch::Tensor sc_p,
                         torch::Tensor qw_o, torch::Tensor qz_o, torch::Tensor sc_o,
                         torch::Tensor kc, torch::Tensor vc,
                         torch::Tensor rw2,
                         torch::Tensor qw_g2, torch::Tensor qz_g2, torch::Tensor sc_g2,
                         torch::Tensor qw_u2, torch::Tensor qz_u2, torch::Tensor sc_u2,
                         torch::Tensor qw_d2, torch::Tensor qz_d2, torch::Tensor sc_d2,
                         int64_t group_size, double scale, int64_t H, int64_t Hkv,
                         c10::optional<torch::Tensor> kpool, c10::optional<torch::Tensor> vpool,
                         c10::optional<torch::Tensor> block_table,
                         int64_t block_size, int64_t seq_len, double kscale, double vscale,
                         c10::optional<torch::Tensor> dense_qkv,
                         c10::optional<torch::Tensor> q_norm, c10::optional<torch::Tensor> k_norm,
                         c10::optional<torch::Tensor> cos_sin, int64_t rot, int64_t kv_pos,
                         c10::optional<torch::Tensor> dbg,
                         int64_t up_offset, int64_t mlp_width) {
    // СЛИТЫЙ gate_up: подмена ловится в ОБЕ стороны. Без этого ядро считало ширину ВДВОЕ больше
    // и выходило за границу памяти вместо отказа -- это и есть тихая неверность с падением.
    TORCH_CHECK(mlp_width == 0 || (qw_g2.data_ptr() == qw_u2.data_ptr() && up_offset > 0),
                "attn_layer: mlp_width задан, но gate и up -- разные тензоры или нет смещения");
    TORCH_CHECK(mlp_width != 0 || qw_g2.data_ptr() != qw_u2.data_ptr(),
                "attn_layer: gate и up -- ОДИН тензор, но mlp_width не задан (посчиталось бы вдвое шире)");
    const int upoff_ = (int)up_offset, Ntot_ = (int)mlp_width;
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "attn_layer: плотный вектор на карте");
    const bool dq = dense_qkv.has_value();
    // ПЛОТНЫЙ qkv: ширины берутся ИЗ НЕГО, а не из квантованного тензора, которого у боевой сети нет.
    const int K = dq ? (int)dense_qkv->size(1) : (int)qw_p.size(0) * 4;
    const int Nqkv = dq ? (int)dense_qkv->size(0) : (int)qw_p.size(1);
    const int G = (int)group_size;
    TORCH_CHECK(!dq || (q_norm.has_value() && k_norm.has_value() && cos_sin.has_value()),
                "attn_layer: плотный qkv требует q_norm/k_norm/cos_sin -- иначе разбор молча не сделается");
    TORCH_CHECK(kv_pos < 0 || kpool.has_value(), "attn_layer: kv_pos задан без склада");
    const bool paged = kpool.has_value();
    const int Kout = (int)qw_o.size(0) * 4;
    const int D = paged ? (int)kpool->size(3) : (int)kc.size(2);
    const int Sk = paged ? (int)seq_len : (int)kc.size(0);
    long blk_stride = paged ? (long)kpool->stride(0) : 0;
    const int bsz = (int)block_size;
    uint8_t* pk8 = paged ? reinterpret_cast<uint8_t*>(kpool->data_ptr()) : nullptr;
    uint8_t* pv8 = paged ? reinterpret_cast<uint8_t*>(vpool->data_ptr()) : nullptr;
    const int* pbt = paged ? block_table->data_ptr<int>() : nullptr;
    float fks = (float)kscale, fvs = (float)vscale;
    // ОДНА геометрия на оба типа слоя: 128 потоков, по 2 элемента головы на поток.
    const int T = 128, CPT = 4;
    TORCH_CHECK(D == 2 * T, "attn_layer: собрано под d=256 при 128 потоках");
    TORCH_CHECK((int)H * D == Kout, "attn_layer: вход o_proj не равен выходу внимания");
    TORCH_CHECK(Nqkv % (T * CPT) == 0 && K % (T * CPT) == 0, "attn_layer: плитки не сходятся");
    const at::cuda::CUDAGuard guard(x.device());
    auto st = at::cuda::getCurrentCUDAStream();
    auto fo = torch::dtype(torch::kFloat32).device(x.device());
    auto ho = torch::dtype(torch::kFloat16).device(x.device());
    const int cols1 = Nqkv / (T * CPT), cols2 = K / (T * CPT);
    const int ng1 = (K + G - 1) / G, ng2 = (Kout + G - 1) / G;
    const void* kfn = (const void*)attn_layer_kernel<128, 4>;
    int perSM = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kfn, T, XT2_ * sizeof(float)));
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int nblk = max(1, perSM * nsm);
    // ПРОВЕРКА ГИПОТЕЗЫ: если разница цепи и обёртки -- лишь ДРУГОЕ ДРОБЛЕНИЕ суммы, то при
    // одинаковом числе блоков она обязана исчезнуть. Гейт только для замера, умолчание прежнее.
    if (const char* e_ = getenv("FA2SM70_NBLK")) nblk = atoi(e_);
    int ks1 = max(1, min(ng1, (nblk + cols1 - 1) / cols1));
    const int gps1 = (ng1 + ks1 - 1) / ks1; ks1 = (ng1 + gps1 - 1) / gps1;
    int ks2 = max(1, min(ng2, (nblk + cols2 - 1) / cols2));
    const int gps2 = (ng2 + ks2 - 1) / ks2; ks2 = (ng2 + gps2 - 1) / gps2;
    int nsp = max(1, min(Sk, nblk / max(1, (int)H)));
    auto part1 = torch::empty({ks1, Nqkv}, fo), part2 = torch::empty({ks2, K}, fo);
    auto pm = torch::empty({nsp, (int)H}, fo), pl = torch::empty({nsp, (int)H}, fo);
    auto bmlp = torch::empty({K}, ho);
    auto pacc = torch::empty({nsp, (int)H, D}, fo);
    const int I2 = (mlp_width ? (int)(mlp_width/2) : (int)qw_g2.size(1));
    // Плитка 1024 (256 потоков x CPT=4), а I2 = 8704 на неё НЕ делится -- берём с округлением
    // вверх и полагаемся на маску живых столбцов, она уже есть.
    const int cols3 = (I2 + T * CPT - 1) / (T * CPT), cols4 = K / (T * CPT);
    const int ng3 = (K + G - 1) / G, ng4 = (I2 + G - 1) / G;
    int ks3 = max(1, min(ng3, (nblk + cols3 - 1) / cols3));
    const int gps3 = (ng3 + ks3 - 1) / ks3; ks3 = (ng3 + gps3 - 1) / gps3;
    int ks4 = max(1, min(ng4, (nblk + cols4 - 1) / cols4));
    const int gps4 = (ng4 + ks4 - 1) / ks4; ks4 = (ng4 + gps4 - 1) / gps4;
    auto part3 = torch::zeros({(long)ks3 * 2 * I2 + 1024}, fo), part4 = torch::empty({ks4, K}, fo);
    auto mid2 = torch::empty({I2}, ho);
    auto mid = torch::empty({max(Nqkv, Kout)}, ho), out = torch::empty({1, K}, ho);
    auto H2 = [](torch::Tensor t){ return reinterpret_cast<const __half*>(t.data_ptr<at::Half>()); };
    const __half *px=H2(x), *prw=H2(rw), *psp=H2(sc_p), *pso=H2(sc_o);
    const __half *pk = paged ? nullptr : H2(kc), *pv = paged ? nullptr : H2(vc);
    __half *pmid=reinterpret_cast<__half*>(mid.data_ptr<at::Half>());
    __half *po=reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    int *pwp=qw_p.data_ptr<int>(), *pzp=qz_p.data_ptr<int>(), *pwo=qw_o.data_ptr<int>(), *pzo=qz_o.data_ptr<int>();
    float *pp1=part1.data_ptr<float>(), *pp2=part2.data_ptr<float>();
    float *ppm=pm.data_ptr<float>(), *ppl=pl.data_ptr<float>(), *ppa=pacc.data_ptr<float>();
    const __half *prw2=H2(rw2), *psg2=H2(sc_g2), *psu2=H2(sc_u2), *psd2=H2(sc_d2);
    int *pwg2=qw_g2.data_ptr<int>(), *pzg2=qz_g2.data_ptr<int>();
    int *pwu2=qw_u2.data_ptr<int>(), *pzu2=qz_u2.data_ptr<int>();
    int *pwd2=qw_d2.data_ptr<int>(), *pzd2=qz_d2.data_ptr<int>();
    float *pp3=part3.data_ptr<float>(), *pp4=part4.data_ptr<float>();
    __half *pmid2=reinterpret_cast<__half*>(mid2.data_ptr<at::Half>());
    float feps=(float)eps, fsc=(float)scale;
    int hh=(int)H, hkv=(int)Hkv, dd=D;
    __half* nullpx=nullptr; const __half* nullcpx=nullptr;
    int *nullip=nullptr, *nullip2=nullptr; int zseq=0;
    __half* pbmlp=reinterpret_cast<__half*>(bmlp.data_ptr<at::Half>());
    int kvp = (int)kv_pos, irot = (int)rot;
    const __half* pdq = dq ? (const __half*)dense_qkv->data_ptr() : nullptr;
    const __half* pqn = q_norm.has_value() ? (const __half*)q_norm->data_ptr() : nullptr;
    const __half* pkn = k_norm.has_value() ? (const __half*)k_norm->data_ptr() : nullptr;
    const float*  pcs = cos_sin.has_value() ? cos_sin->data_ptr<float>() : nullptr;
    __half* pdbg = dbg.has_value() ? (__half*)dbg->data_ptr() : nullptr;
    void* ka[] = {(void*)&px,(void*)&prw,(void*)&pwp,(void*)&pzp,(void*)&psp,
                  (void*)&pwo,(void*)&pzo,(void*)&pso,(void*)&pk,(void*)&pv,
                  (void*)&pk8,(void*)&pv8,(void*)&pbt,(void*)&blk_stride,(void*)&bsz,
                  (void*)&fks,(void*)&fvs,
                  (void*)&pp1,(void*)&pmid,(void*)&ppm,(void*)&ppl,(void*)&ppa,
                  (void*)&pp2,(void*)&po,
                  (void*)&prw2,(void*)&pwg2,(void*)&pzg2,(void*)&psg2,
                  (void*)&pwu2,(void*)&pzu2,(void*)&psu2,
                  (void*)&pwd2,(void*)&pzd2,(void*)&psd2,
                  (void*)&pp3,(void*)&pmid2,(void*)&pp4,
                  (void*)&Nqkv,(void*)&K,(void*)&G,(void*)&feps,(void*)&fsc,
                  (void*)&cols1,(void*)&ks1,(void*)&gps1,(void*)&cols2,(void*)&ks2,(void*)&gps2,
                  (void*)&hh,(void*)&hkv,(void*)&dd,(void*)&Sk,(void*)&nsp,(void*)&Kout,
                  (void*)&I2,(void*)&cols3,(void*)&ks3,(void*)&gps3,
                  (void*)&cols4,(void*)&ks4,(void*)&gps4,
                  (void*)&nullpx,(void*)&nullcpx,(void*)&nullip,(void*)&nullip2,
                  (void*)&zseq,(void*)&pbmlp,
                  (void*)&kvp,(void*)&pdq,(void*)&pqn,(void*)&pkn,(void*)&pcs,(void*)&irot,(void*)&pdbg,(void*)&upoff_,(void*)&Ntot_};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kfn, dim3(nblk), dim3(T), ka, XT2_ * sizeof(float), st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ================================================================= ЦЕПЬ СЛОЁВ В ОДНОМ ПУСКЕ
// Параметров у тела 59-61, поэтому слои передаются ОПИСАТЕЛЯМИ в памяти карты, а не аргументами.
// Ветвление по типу слоя идёт по номеру -- он одинаков во всех блоках, значит расхождения на
// барьере не возникает по построению.
struct FA2Layer {
    int is_gdn;
    const __half *rw, *rw2;
    const int *qw1, *qz1; const __half *sc1;      // GDN: in_proj_qkv   | ATTN: qkv_proj
    const int *qw2, *qz2; const __half *sc2;      // GDN: in_proj_z     | ATTN: не нужен
    const __half *w_ab;                            // GDN
    const int *qwo, *qzo; const __half *sco;      // out_proj | o_proj
    const __half *conv_w; __half *conv_state;      // GDN
    const float *a_log; const __half *dt_bias; float *hstate;   // GDN
    const __half *gnorm;   // GDN: вес RMSNormGated выхода мешалки (norm_before_gate=True)
    const int *qwg, *qzg; const __half *scg;      // MLP gate
    const int *qwu, *qzu; const __half *scu;      // MLP up
    const int *qwd, *qzd; const __half *scd;      // MLP down
    const __half *kc, *vc;                         // ATTN: плотные K/V
    uint8_t *kp8, *vp8; const int *btab;     // ATTN: постраничный склад
    long blk_stride; int bsz, seq_len; float kscale, vscale;
    int kv_pos;                // позиция текущего токена в складе (-1 = пишет вызывающий)
    const __half* dense_qkv;   // ПЛОТНЫЙ qkv (не квантован) -- nullptr = идти квантованным путём
    const __half *q_norm, *k_norm; const float* cs; int rot;   // внимание: нормы по голове + вращение
    // ЧТО МЕНЯЕТСЯ КАЖДЫЙ ШАГ -- ТОЛЬКО ЧЕРЕЗ УСТРОЙСТВО. Опись копируется на карту ОДИН раз и
    // при захвате в CUDA-граф ЗАМОРАЖИВАЕТСЯ: хостовые `seq_len`, `kv_pos` и указатель на слот
    // состояния после захвата остались бы навсегда теми, что были в момент захвата. Поэтому они
    // читаются из памяти карты: позиция -- из тензора движка, слот -- из его же индекса.
    const long* poz;   // позиция токена (int64, как у vLLM); nullptr = брать хостовые поля
    const int* slot;   // номер слота состояния GDN; nullptr = слот уже учтён в указателе
    long conv_шаг, h_шаг;      // шаг между слотами состояния, в элементах
    long conv_ш_i, conv_ш_j;   // шаги внутри слота свёрточного состояния
    long cs_шаг;               // шаг строки таблицы вращения (0 = дана уже строка)
    int Nqz;           // СЛИТЫЙ in_proj_qkvz: полная ширина плиты (0 = раздельные)
    int upoff, Ntot;   // СЛИТЫЙ gate_up_proj: база up и полная ширина плиты (0/0 = раздельные)
};

// БЛОКОВ НА SM -- ЗАДАЁТСЯ, А НЕ ВЫБИРАЕТСЯ КОМПИЛЯТОРОМ. Замер сеткой (40/80/120/160 блоков:
// 202/340/397/472 ГБ/с) показал, что цепь НЕ насыщает канал -- её связывает параллелизм. При
// 172 регистрах на SM влезает 2 блока по 128 нитей = 256 нитей из возможных 65536/172=381.
// Порог трёх блоков (12 варпов) -- 65536/(32*12)=170.7, то есть 168 регистров: не хватало ЧЕТЫРЁХ.
// Порог четырёх блоков -- 128 регистров.
#ifndef FA2SM70_CHAIN_BPSM
#define FA2SM70_CHAIN_BPSM 3
#endif
#ifndef FA2SM70_CHAIN_CPT
#define FA2SM70_CHAIN_CPT 4
#endif
template <int THREADS, int CPT>
__global__ __launch_bounds__(THREADS, FA2SM70_CHAIN_BPSM)
void decode_chain_kernel(const FA2Layer* __restrict__ lay, int nlayers,
                                    const __half* __restrict__ x0, __half* __restrict__ buf,
                                    float* p1, __half* mid, float* p2, float* p3, __half* mid2, float* p4,
                                    __half* out,
                                    int Nq, int Nz, int Nab, int Nqkv, int K, int G,
                                    float eps, float scale, float scale_gdn,
                                    int gc1, int gk1, int gg1, int gc2, int gk2, int gg2,
                                    int ac1, int ak1, int ag1, int ac2, int ak2, int ag2,
                                    int c3, int k3, int g3, int c4, int k4, int g4,
                                    int Hk, int Hv, int DK, int DV, int KoutG,
                                    int H, int Hkv, int D, int nsp, int KoutA, int I2,
                                    float* pm, float* pl, float* pacc,
                                    PeerPlan plan,
                                    int seq0, __half* bmlp, __half* dbgp) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float sx[XT2_];
    const __half* cur = x0;
    for (int L = 0; L < nlayers; ++L) {
        const FA2Layer& d = lay[L];
        __half* dst = (L & 1) ? out : buf;          // пинг-понг: выход слоя -- вход следующего
        // ПОЗИЦИЯ И СЛОТ -- С УСТРОЙСТВА, если они там даны (условие захвата в граф).
        const int _поз = d.poz ? (int)(*d.poz) : -1;
        const int _Sk = (_поз >= 0) ? (_поз + 1) : d.seq_len;
        const int _kvp = (_поз >= 0) ? _поз : d.kv_pos;
        const int _сл = d.slot ? *d.slot : 0;
        __half* _conv = d.conv_state ? (d.conv_state + (long)_сл * d.conv_шаг) : nullptr;
        float*  _hst  = d.hstate ? (d.hstate + (long)_сл * d.h_шаг) : nullptr;
        // СТРОКА ВРАЩЕНИЯ -- ТОЖЕ ПО ПОЗИЦИИ С КАРТЫ: иначе в графе она осталась бы строкой
        // момента захвата, и все токены вращались бы на один и тот же угол.
        const float* _cs = (d.cs && d.cs_шаг > 0) ? (d.cs + (long)_поз * d.cs_шаг) : d.cs;
        // ОТНЕСЕНИЕ РЕГИСТРОВ ПО ФАЗАМ. Регистровый бюджет ядра -- МАКСИМУМ по телам, значит
        // одна фаза запирает занятость всем остальным. Какая -- узнаётся только исключением:
        // собрать без неё и прочитать REG/STACK. Тело заведомо неверно, ЗАПУСКАТЬ НЕЛЬЗЯ,
        // читается только отчёт компилятора. По умолчанию гейты выключены.
#if !defined(FA2SM70_БЕЗ_GDN)
        if (d.is_gdn) {
            gdn_layer_body<THREADS, CPT>(grid, sx, cur, d.rw, d.qw1, d.qz1, d.sc1,
                d.qw2, d.qz2, d.sc2, d.w_ab, d.qwo, d.qzo, d.sco, d.conv_w, _conv,
                d.conv_ш_i, d.conv_ш_j, d.a_log, d.dt_bias, _hst, d.gnorm,
                (L == 0 ? dbgp : nullptr), p1, mid, p2, dst,
                d.rw2, d.qwg, d.qzg, d.scg, d.qwu, d.qzu, d.scu, d.qwd, d.qzd, d.scd,
                p3, mid2, p4,
                Nq, Nz, Nab, K, G, eps, scale_gdn, gc1, gk1, gg1, gc2, gk2, gg2,
                Hk, Hv, DK, DV, KoutG, I2, c3, k3, g3, c4, k4, g4,
                plan, seq0 + 2 * L, bmlp, d.upoff, d.Ntot, d.Nqz);
        } else
#else
        if (!d.is_gdn)
#endif
        {
#if !defined(FA2SM70_БЕЗ_ВНИМ)
            attn_layer_body<THREADS, CPT>(grid, sx, cur, d.rw, d.qw1, d.qz1, d.sc1,
                d.qwo, d.qzo, d.sco, d.kc, d.vc, d.kp8, d.vp8, d.btab,
                d.blk_stride, d.bsz, d.kscale, d.vscale, p1, mid, pm, pl, pacc, p2, dst,
                d.rw2, d.qwg, d.qzg, d.scg, d.qwu, d.qzu, d.scu, d.qwd, d.qzd, d.scd,
                p3, mid2, p4,
                Nqkv, K, G, eps, scale, ac1, ak1, ag1, ac2, ak2, ag2,
                H, Hkv, D, _Sk, nsp, KoutA, I2, c3, k3, g3, c4, k4, g4,
                plan, seq0 + 2 * L, bmlp, _kvp, d.dense_qkv, d.q_norm, d.k_norm, _cs, d.rot, d.upoff, d.Ntot, nullptr);
#endif
        }
        ГБАР(grid);
        // обнуление хвостового слота БОЛЬШЕ НЕ НУЖНО: сумма собирается не атомарно, а по слоту
        // на блок, и каждый слой переписывает СВОИ слоты целиком.
        cur = dst;
        ГБАР(grid);
    }
}



// ---------------------------------------------------------------- АРЕНА ЦЕПИ
// Рабочая область и опись слоёв ЖИВУТ МЕЖДУ ВЫЗОВАМИ. Две причины, и вторая важнее первой:
//   1) одиннадцать `torch::zeros` на каждый шаг -- это выделения и обнуления на ровном месте;
//   2) опись копировалась из СТРАНИЧНОЙ памяти (`std::vector` на стеке функции). При захвате в
//      CUDA-граф такая копия становится узлом, указывающим на память, которой после возврата
//      уже нет: захват проходит, а ВОСПРОИЗВЕДЕНИЕ падает по недопустимому адресу. Замерено.
// Закреплённый буфер снимает обе.
struct ЦепьАрена {
    torch::Tensor опись_хост, опись_карта;
    torch::Tensor p1, p2, p3, p4, mid, mid2, pm, pl, pacc, buf, out, bmlpT;
};
static std::unordered_map<int, ЦепьАрена> АРЕНЫ_ЦЕПИ;

// Тензор нужного размера: переиспользуем, если подходит, иначе заводим заново.
static inline torch::Tensor& арена_взять(torch::Tensor& т, long n, const torch::TensorOptions& o) {
    if (!т.defined() || т.numel() < n) т = torch::zeros({n}, o);
    return т;
}

// Точка входа цепи. Слои передаются списками тензоров в ФИКСИРОВАННОМ порядке; порядок и длина
// проверяются здесь же -- перепутанный тензор дал бы не падение, а молча неверный ответ.
//   GDN  (26): rw rw2 | qkv(3) z(3) | w_ab | out(3) | conv_w conv_state a_log dt_bias hstate | g(3) u(3) d(3)
//   ATTN (20): rw rw2 | qkv(3)      | out(3) | kc vc | g(3) u(3) d(3)
torch::Tensor decode_chain(torch::Tensor x0,
                           std::vector<std::vector<torch::Tensor>> layers,
                           std::vector<int64_t> is_gdn,
                           int64_t group_size, double eps, double scale,   // scale -- ВНИМАНИЯ (1/sqrt(d))
                           double scale_gdn,               // у GDN СВОЙ (1/sqrt(DK))
                           int64_t Hk, int64_t Hv, int64_t H, int64_t Hkv, int64_t seq_len,
                           int64_t rot, std::vector<int64_t> kv_pos,
                           int64_t qkvz_width, int64_t up_offset, int64_t mlp_width,
                           double kscale, double vscale,
                           int64_t обм_ярусов,
                           std::vector<int64_t> обм_соседей,
                           std::vector<int64_t> обм_тянуть,
                           std::vector<int64_t> обм_чужой,
                           std::vector<torch::Tensor> обм_приход,
                           std::vector<int64_t> обм_флаг_чужой,
                           std::vector<torch::Tensor> обм_флаг_мой,
                           int64_t обм_ширина,
                           c10::optional<torch::Tensor> обм_счёт,
                           c10::optional<torch::Tensor> peer_inbox,
                           c10::optional<torch::Tensor> my_inbox,
                           c10::optional<torch::Tensor> my_flag,
                           c10::optional<torch::Tensor> peer_flag,
                           // ВХОДЫ, ГОДНЫЕ ДЛЯ ГРАФА: позиция и номера слотов состояния лежат на
                           // карте, поэтому повтор графа видит ТЕКУЩИЕ значения, а не значения
                           // момента захвата. Без них шаг в граф не заводится.
                           c10::optional<torch::Tensor> poz_dev = c10::nullopt,
                           c10::optional<std::vector<torch::Tensor>> slot_idx = c10::nullopt,
                           c10::optional<torch::Tensor> dbg_плита = c10::nullopt) {
    const int nl = (int)layers.size();
    TORCH_CHECK(nl > 0 && (int)is_gdn.size() == nl, "decode_chain: списки не сходятся");
    const int K = (int)x0.numel(), G = (int)group_size, DK = 128, DV = 128, D = 256;
    // СТОЛБЦОВ НА НИТЬ -- свободная переменная стенда: она задаёт и регистровое давление, и
    // число заданий. Меняется одним -D, чтобы ранжировать замером, а не рассуждением.
    const int T = 128, CPT = FA2SM70_CHAIN_CPT;
    const at::cuda::CUDAGuard guard(x0.device());
    auto st = at::cuda::getCurrentCUDAStream();
    auto fo = torch::dtype(torch::kFloat32).device(x0.device());
    auto ho = torch::dtype(torch::kFloat16).device(x0.device());
    auto H2 = [](const torch::Tensor& t){ return reinterpret_cast<const __half*>(t.data_ptr<at::Half>()); };

    // геометрия обоих типов -- одна на всю цепь, берётся с ПЕРВОГО слоя каждого типа
    int Nq=0,Nz=0,Nab=0,Nqkv=0,I2=0,KoutG=0,KoutA=0;
    for (int L = 0; L < nl; ++L) {
        const auto& v = layers[L];
        if (is_gdn[L]) {
            TORCH_CHECK(v.size() == 27, "decode_chain: у слоя GDN должно быть 27 тензоров "
                                        "(26 прежних + вес нормы выхода мешалки)");
            // ПРИ СЛИТОЙ ПЛИТЕ форма не источник геометрии: ширины выводим из ГОЛОВ и СВЕРЯЕМ
            // с полной шириной -- проверка контролирует сама себя, а не доверяет одному числу.
            if (qkvz_width) {
                Nq=(int)(Hk*128*2 + Hv*128); Nz=(int)(Hv*128);
                TORCH_CHECK(Nq+Nz == (int)qkvz_width,
                            "decode_chain: геометрия голов не сходится с шириной плиты qkvz");
            } else { Nq=(int)v[2].size(1); Nz=(int)v[5].size(1); }
            Nab=(int)v[8].size(1);
            KoutG=(int)v[9].size(0)*4; I2=mlp_width ? (int)(mlp_width/2) : (int)v[17].size(1);
        } else {
            TORCH_CHECK(v.size() == 19 || v.size() == 23 || v.size() == 26,
                        "decode_chain: у слоя внимания 19, 23 (плотный qkv) или 26 (со складом) тензоров");
            // ПЛОТНЫЙ qkv боевой сети: ширина берётся ИЗ НЕГО, а не из квантованного тензора.
            Nqkv = ((int)v.size() >= 23) ? (int)v[19].size(0) : (int)v[2].size(1);
            KoutA=(int)v[5].size(0)*4; I2=mlp_width ? (int)(mlp_width/2) : (int)v[10].size(1);   // v[10] -- ВЕСА gate,
                                                                     // v[11] это НУЛИ (ширина I/4)
        }
    }
    TORCH_CHECK(I2 > 0 && I2 % (T * CPT) == 0,
                "decode_chain: промежуточная ширина MLP не кратна плитке -- проверьте позиции тензоров");
    const int NQZ=Nq+Nz, NT=NQZ+Nab;
    const void* kfn = (const void*)decode_chain_kernel<128,FA2SM70_CHAIN_CPT>;
    int perSM=0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM,kfn,T,XT2_*sizeof(float)));
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int nblk = max(1, perSM*nsm);
    // ПРОБА ПО ЗАНЯТОСТИ. Сетка -- свободная переменная стенда: без неё «связывает занятость»
    // остаётся догадкой, а не замером. ВВЕРХ поднимать нельзя (кооперативный пуск откажет),
    // поэтому только вниз. По умолчанию не вмешивается.
    if (const char* e_ = getenv("FA2SM70_NBLK")) { int v = atoi(e_); if (v > 0) nblk = min(v, nblk); }
    { static int _раз = 0;
      if (!_раз++) fprintf(stderr, "[fa2_sm70 цепь] блоков на SM=%d SM=%d сетка=%d нитей=%d\n",
                           perSM, nsm, nblk, T); }
    const int ЦПТ_ВЫХ = (FA2SM70_CPT_OUT_HOST);
    auto split=[&](int cols,int ng){ int k=max(1,min(ng,(nblk+cols-1)/cols)); int g=(ng+k-1)/k; k=(ng+g-1)/g; return std::make_pair(k,g); };
    int gc1=NQZ?NQZ/(T*CPT):1, gc2=K/(T*ЦПТ_ВЫХ), ac1=Nqkv?Nqkv/(T*CPT):1, ac2=K/(T*ЦПТ_ВЫХ);
    int c3=(I2+T*CPT-1)/(T*CPT), c4=K/(T*CPT);
    auto [gk1,gg1]=split(gc1,(K+G-1)/G);
    auto [gk2,gg2]=split(gc2,KoutG?(KoutG+G-1)/G:1);
    auto [ak1,ag1]=split(ac1,(K+G-1)/G);
    auto [ak2,ag2]=split(ac2,KoutA?(KoutA+G-1)/G:1);
    auto [k3,g3]=split(c3,(K+G-1)/G);
    auto [k4,g4]=split(c4,(I2+G-1)/G);
    // ЧИСЛО СРЕЗОВ НЕ ДОЛЖНО ЗАВИСЕТЬ ОТ ПОЗИЦИИ, когда позиция живёт на карте: иначе оно
    // замёрзнет при захвате в граф. Пустые срезы обрабатываются штатно (m=-inf, вклад ноль).
    int nsp = poz_dev.has_value() ? max(1, nblk/max(1,(int)H))
                                  : max(1, min((int)seq_len, nblk/max(1,(int)H)));

    // Д2а ДОКАЗАН: грязная рабочая область МЕНЯЕТ ответ. В цепи буферы ОБЩИЕ для обоих типов,
    // поэтому внимание после GDN видит ЕГО остатки -- источник расхождения смешанной цепи.
    // ОБНУЛЕНИЕ ОСТАЁТСЯ: доказано, что грязная рабочая область МЕНЯЕТ ответ (буферы общие для
    // обоих типов слоя). Уходят только ВЫДЕЛЕНИЯ -- сама память переживает вызов.
    ЦепьАрена& ар = АРЕНЫ_ЦЕПИ[(int)x0.device().index()];
    auto& p1 = арена_взять(ар.p1, max((long)gk1*NT,(long)ak1*Nqkv), fo);
    auto& p2 = арена_взять(ар.p2, (long)max(gk2,ak2)*K, fo);
    auto& p3 = арена_взять(ар.p3, (long)k3*2*I2 + 1024, fo);   // хвост: слот НА БЛОК, а не один общий
    auto& p4 = арена_взять(ар.p4, (long)k4*K, fo);
    // МЕСТО ПОД ПЛОСКУЮ РАСКЛАДКУ. Плита qkvz приходит блоками по k-голове, а свёртка и мешалка
    // работают над сплошным [q|k|v] и сплошным z, поэтому за плитой держим ещё Nq+Nz+Hv*DV:
    // плоские q|k|v, плоский z и выход мешалки. Без этого перекладка писала бы в чужое.
    auto& mid = арена_взять(ар.mid, max((long)NT + Nq + Nz + (long)Hv * 128,
                                        (long)max(Nqkv, KoutA)), ho);
    auto& mid2 = арена_взять(ар.mid2, I2, ho);
    auto& pm = арена_взять(ар.pm, (long)nsp*H, fo);
    auto& pl = арена_взять(ар.pl, (long)nsp*H, fo);
    auto& pacc = арена_взять(ар.pacc, (long)nsp*H*D, fo);
    auto& buf = арена_взять(ар.buf, K, ho);
    auto& out = арена_взять(ар.out, K, ho);
    auto& bmlpT = арена_взять(ар.bmlpT, K, ho);
    for (torch::Tensor* т : {&p1,&p2,&p3,&p4,&mid,&mid2,&pm,&pl,&pacc,&buf,&out,&bmlpT}) т->zero_();

    // ОПИСЬ -- В ЗАКРЕПЛЁННОМ буфере: только из него копия годится внутри графа.
    const long опись_байт = (long)nl * (long)sizeof(FA2Layer);
    if (!ар.опись_хост.defined() || ар.опись_хост.numel() < опись_байт)
        ар.опись_хост = torch::empty({опись_байт},
            torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    if (!ар.опись_карта.defined() || ар.опись_карта.numel() < опись_байт)
        ар.опись_карта = torch::empty({опись_байт},
            torch::dtype(torch::kUInt8).device(x0.device()));
    FA2Layer* hl = reinterpret_cast<FA2Layer*>(ар.опись_хост.data_ptr());
    for (int L=0;L<nl;++L){
        const auto& v=layers[L]; FA2Layer d{}; d.is_gdn=(int)is_gdn[L];
        if (poz_dev.has_value()) {
            TORCH_CHECK(poz_dev->scalar_type() == torch::kLong && poz_dev->numel() >= 1,
                        "decode_chain: позиция должна быть int64 на карте (как у vLLM)");
            d.poz = poz_dev->data_ptr<long>();
        }
        // ЯВНЫЙ ПРИЗНАК "НЕ ПИСАТЬ": обнуление структуры даёт kv_pos=0, а это ЗАПИСЬ В СЛОТ 0.
        // Умолчание обязано быть безопасным, а не совпадать с нулём по случайности.
        d.kv_pos = -1;
        d.rw=H2(v[0]); d.rw2=H2(v[1]);
        d.qw1=v[2].data_ptr<int>(); d.qz1=v[3].data_ptr<int>(); d.sc1=H2(v[4]);
        int o = d.is_gdn ? 5 : 5;
        if (d.is_gdn){
            d.qw2=v[5].data_ptr<int>(); d.qz2=v[6].data_ptr<int>(); d.sc2=H2(v[7]);
            d.w_ab=H2(v[8]);
            d.qwo=v[9].data_ptr<int>(); d.qzo=v[10].data_ptr<int>(); d.sco=H2(v[11]);
            d.conv_w=H2(v[12]); d.conv_state=reinterpret_cast<__half*>(v[13].data_ptr<at::Half>());
            // ШАГИ -- ИЗ САМОГО ТЕНЗОРА, а не из предположения о раскладке. Без device-слота
            // остаётся прежний договор: плотный вид [Cqkv, 3], который готовит вызывающий.
            if (slot_idx.has_value()) {
                TORCH_CHECK(v[13].dim() == 3,
                            "decode_chain: при слоте на карте conv_state обязан быть [слоты, 3, C]");
                d.conv_шаг = v[13].stride(0); d.conv_ш_i = v[13].stride(2); d.conv_ш_j = v[13].stride(1);
            } else { d.conv_шаг = 0; d.conv_ш_i = 3; d.conv_ш_j = 1; }
            // ПЛОТНОСТЬ ТРЕБУЕТСЯ ТОЛЬКО СТАРОМУ ПУТИ. При слоте на карте ядро читает
            // свёрточное состояние ПО ШАГАМ самого тензора, поэтому чужая раскладка законна;
            // проверять надо не плотность, а то, что шаги ЕСТЬ и слот плотный (ниже).
            if (!slot_idx.has_value())
                TORCH_CHECK(v[13].is_contiguous() && v[16].is_contiguous(),
                            "decode_chain: состояния слоя GDN обязаны быть ПЛОТНЫМИ");
            TORCH_CHECK(v[13].scalar_type() == torch::kHalf && v[16].scalar_type() == torch::kFloat,
                        "decode_chain: conv_state -- fp16, hstate -- fp32");
            d.a_log=v[14].data_ptr<float>(); d.dt_bias=H2(v[15]); d.hstate=v[16].data_ptr<float>();
            d.h_шаг = slot_idx.has_value() ? v[16].stride(0) : 0;
            if (slot_idx.has_value())
                TORCH_CHECK(v[16].select(0, 0).is_contiguous(),
                            "decode_chain: слот скрытого состояния обязан быть плотным");
            if (slot_idx.has_value()) {
                TORCH_CHECK((int)slot_idx->size() == nl,
                            "decode_chain: индексов слота должно быть по одному на слой цепи");
                const torch::Tensor& s = (*slot_idx)[L];
                d.slot = s.defined() ? s.data_ptr<int>() : nullptr;
            }
            // ВЕС НОРМЫ ВЫХОДА МЕШАЛКИ. Его не было в описателе вовсе, а ядро делало
            // o*silu(z) вместо norm(o)*w*silu(z) -- комментарий обещал норму, код её не делал.
            TORCH_CHECK((int)v.size() >= 27, "decode_chain: у слоя GDN нет веса нормы выхода (позиция 26)");
            d.gnorm=H2(v[26]);
            o=17;
        } else {
            d.qwo=v[5].data_ptr<int>(); d.qzo=v[6].data_ptr<int>(); d.sco=H2(v[7]);
            d.kc=H2(v[8]); d.vc=H2(v[9]);
            // СНЯТИЕ ФАЗЫ: FA2SM70_KEYCAP=N обходит не больше N ключей.
            // Ответ ЗАВЕДОМО НЕВЕРЕН -- читается только время; разность с полным
            // прогоном даёт ДОЛЮ фазы обхода ключей, а не догадку о ней.
            // getenv БЕЗ static: `static const` уже один раз зафиксировал гейт
            // на первом вызове и дал молча неверный вердикт.
            { const char* _c = std::getenv("FA2SM70_KEYCAP");
              const int _cap = _c ? atoi(_c) : 0;
              d.seq_len = (_cap > 0 && (int)seq_len > _cap) ? _cap : (int)seq_len; }
            d.bsz=16; d.kscale=1.f; d.vscale=1.f;
            o=10;
        }
        d.qwg=v[o+0].data_ptr<int>(); d.qzg=v[o+1].data_ptr<int>(); d.scg=H2(v[o+2]);
        d.qwu=v[o+3].data_ptr<int>(); d.qzu=v[o+4].data_ptr<int>(); d.scu=H2(v[o+5]);
        d.qwd=v[o+6].data_ptr<int>(); d.qzd=v[o+7].data_ptr<int>(); d.scd=H2(v[o+8]);
        // ДОБАВКИ БОЕВОЙ СЕТИ (по факту разведки): плотный qkv, нормы по голове, строка вращения.
        // Их наличие определяется ДЛИНОЙ списка -- лишние тензоры нельзя подать молча.
        const int e = o + 9;
        if (!d.is_gdn && (int)v.size() >= e + 7) {
            // ПОСТРАНИЧНЫЙ СКЛАД в цепи: три тензора после добавок. Шаг блока берётся из САМОГО
            // тензора (stride), а не задаётся числом -- число разъезжается с формой молча.
            d.kp8=reinterpret_cast<uint8_t*>(v[e+4].data_ptr());
            d.vp8=reinterpret_cast<uint8_t*>(v[e+5].data_ptr());
            d.btab=v[e+6].data_ptr<int>();
            d.blk_stride=v[e+4].stride(0); d.bsz=(int)v[e+4].size(1);
            d.kscale=(float)kscale; d.vscale=(float)vscale;
            { const char* _c = std::getenv("FA2SM70_KEYCAP");
              const int _cap = _c ? atoi(_c) : 0;
              d.seq_len = (_cap > 0 && (int)seq_len > _cap) ? _cap : (int)seq_len; }
            TORCH_CHECK(v[e+4].scalar_type()==torch::kUInt8 && v[e+5].scalar_type()==torch::kUInt8,
                        "decode_chain: склад KV обязан быть байтовым");
        }
        if (!d.is_gdn && (int)v.size() >= e + 4) {
            d.dense_qkv=H2(v[e]); d.q_norm=H2(v[e+1]); d.k_norm=H2(v[e+2]);
            d.cs=v[e+3].data_ptr<float>(); d.rot=(int)rot;
            // ПРИ ПОЗИЦИИ НА КАРТЕ сюда кладут ВСЮ таблицу вращения, а строку выбирает ядро:
            // строка, выбранная на хосте, в графе замёрзла бы и все токены вращались бы одинаково.
            // Двумерное = ВСЯ таблица (строку берёт ядро), одномерное = уже готовая строка.
            // Различаем по форме, а не по флагу: так оба договора остаются законными.
            if (v[e+3].dim() == 2) d.cs_шаг = v[e+3].stride(0);
            if ((int)kv_pos.size() == nl) d.kv_pos=(int)kv_pos[L];
        }
        d.Nqz=(int)qkvz_width; d.upoff=(int)up_offset; d.Ntot=(int)mlp_width;
        hl[L]=d;
    }
    C10_CUDA_CHECK(cudaMemcpyAsync(ар.опись_карта.data_ptr(), hl, опись_байт,
                                   cudaMemcpyHostToDevice, st));
    const FA2Layer* pl_dev = reinterpret_cast<const FA2Layer*>(ар.опись_карта.data_ptr());
    const __half* px=H2(x0);
    __half *pbuf=reinterpret_cast<__half*>(buf.data_ptr<at::Half>()),
           *pout=reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
           *pmid=reinterpret_cast<__half*>(mid.data_ptr<at::Half>()),
           *pmid2=reinterpret_cast<__half*>(mid2.data_ptr<at::Half>());
    float *q1=p1.data_ptr<float>(),*q2=p2.data_ptr<float>(),*q3=p3.data_ptr<float>(),*q4=p4.data_ptr<float>();
    float *ppm=pm.data_ptr<float>(),*ppl=pl.data_ptr<float>(),*ppa=pacc.data_ptr<float>();
    float feps=(float)eps,fsc=(float)scale;
    // ДВА РАЗНЫХ МАСШТАБА. Один на оба типа НЕ СУЩЕСТВУЕТ: у GDN 1/sqrt(DK), у внимания
    // 1/sqrt(d). Пока параметр был один, половина слоёв молча считалась не тем.
    float fsg=(float)(scale_gdn > 0.0 ? scale_gdn : scale);
    int hk=(int)Hk,hv=(int)Hv,hh=(int)H,hkv=(int)Hkv,dk=DK,dv=DV,dd=D,nlv=nl;
    // ОБМЕН МЕЖДУ КАРТАМИ. Был ЗАШИТ НУЛЯМИ -- ядро молча пропускало сложение половин
    // (`if (peer_inbox == nullptr) return;`), и каждая карта работала со СВОЕЙ половиной.
    // Именно это давало мусор на выходе: out_proj/o_proj у vLLM -- RowParallelLinear.
    // План строится ЗДЕСЬ, из того, что одинаково на всех картах (топология + геометрия). Ветвь,
    // выбранная на шаге и локально, уже вешала ядро: одна карта вошла в обмен, другая нет.
    PeerPlan plan{};
    plan.tiers = 0;
    plan.width = max(1, min(nblk, обм_ширина > 0 ? (int)обм_ширина : 36));
    // СЧЁТЧИК КРУГОВ обязан ПЕРЕЖИВАТЬ пуски: свежий нулевой каждый шаг вернул бы ту же беду,
    // ради которой счёт и перенесён на устройство.
    plan.count = обм_счёт.has_value() ? обм_счёт->data_ptr<int>() : nullptr;
    const bool пара = peer_inbox.has_value();
    const bool общий = обм_ярусов > 0;
    TORCH_CHECK(!(пара && общий), "decode_chain: задан И парный обмен, И общий план -- что-то одно");
    if (пара || общий) {
        TORCH_CHECK(обм_счёт.has_value() && обм_счёт->numel() >= plan.width &&
                    обм_счёт->scalar_type() == torch::kInt,
                    "decode_chain: нужен ПОСТОЯННЫЙ счётчик кругов (int32, длина >= ", plan.width, ")");
    }
    if (пара) {
        TORCH_CHECK(my_inbox.has_value() && my_flag.has_value() && peer_flag.has_value(),
                    "decode_chain: для парного обмена нужны ВСЕ четыре буфера");
        TORCH_CHECK(peer_inbox->numel() >= K && my_inbox->numel() >= K,
                    "decode_chain: почтовый ящик короче скрытой оси");
        TORCH_CHECK(peer_inbox->scalar_type() == torch::kHalf, "decode_chain: ящик обязан быть fp16");
        TORCH_CHECK(my_flag->numel() >= plan.width && peer_flag->numel() >= plan.width,
                    "decode_chain: флагов меньше ширины рандеву (", plan.width,
                    ") -- поблочный барьер прочтёт чужую память");
        plan.tiers = 1; plan.peers[0] = 1; plan.pull[0] = 0;
        plan.far[0][0] = (__half*)peer_inbox->data_ptr();
        plan.inbox[0][0] = (const __half*)my_inbox->data_ptr();
        plan.far_flag[0][0] = peer_flag->data_ptr<int>();
        plan.mine_flag[0][0] = my_flag->data_ptr<int>();
    } else if (общий) {
        TORCH_CHECK(обм_ярусов <= EXCH_MAX_TIERS, "decode_chain: ярусов больше ", EXCH_MAX_TIERS);
        TORCH_CHECK((int)обм_соседей.size() == обм_ярусов && (int)обм_тянуть.size() == обм_ярусов,
                    "decode_chain: длины описи ярусов не совпадают");
        int всего = 0;
        for (int t = 0; t < обм_ярусов; ++t) {
            TORCH_CHECK(обм_соседей[t] >= 1 && обм_соседей[t] <= EXCH_MAX_PEERS,
                        "decode_chain: соседей на ярусе вне 1..", EXCH_MAX_PEERS);
            всего += обм_соседей[t];
        }
        TORCH_CHECK((int)обм_чужой.size() == всего && (int)обм_приход.size() == всего &&
                    (int)обм_флаг_чужой.size() == всего && (int)обм_флаг_мой.size() == всего,
                    "decode_chain: число буферов не равно числу соседей по всем ярусам (", всего, ")");
        plan.tiers = обм_ярусов;
        int c = 0;
        for (int t = 0; t < обм_ярусов; ++t) {
            plan.peers[t] = обм_соседей[t];
            plan.pull[t] = обм_тянуть[t] ? 1 : 0;
            for (int k = 0; k < обм_соседей[t]; ++k, ++c) {
                // ЧУЖАЯ сторона -- сырой указатель: проверить её нечем и не нашей памятью она
                // является. СВОЮ проверяем полностью, чтобы ошибка формы не ушла в ядро молча.
                TORCH_CHECK(обм_чужой[c] != 0 && обм_флаг_чужой[c] != 0,
                            "decode_chain: нулевой указатель на буфер соседа, ярус ", t);
                TORCH_CHECK(обм_приход[c].numel() >= K &&
                            обм_приход[c].scalar_type() == torch::kHalf,
                            "decode_chain: свой ящик короче скрытой оси либо не fp16, ярус ", t);
                TORCH_CHECK(обм_флаг_мой[c].numel() >= plan.width,
                            "decode_chain: своих флагов меньше ширины рандеву ", plan.width);
                plan.far[t][k] = (__half*)(uintptr_t)обм_чужой[c];
                plan.inbox[t][k] = (const __half*)обм_приход[c].data_ptr();
                plan.far_flag[t][k] = (int*)(uintptr_t)обм_флаг_чужой[c];
                plan.mine_flag[t][k] = обм_флаг_мой[c].data_ptr<int>();
            }
        }
    }
    int cseq = 1;
    __half* cbm=reinterpret_cast<__half*>(bmlpT.data_ptr<at::Half>());
    __half* pdbgp = dbg_плита.has_value()
        ? reinterpret_cast<__half*>(dbg_плита->data_ptr<at::Half>()) : nullptr;
    void* ka[]={(void*)&pl_dev,(void*)&nlv,(void*)&px,(void*)&pbuf,
                (void*)&q1,(void*)&pmid,(void*)&q2,(void*)&q3,(void*)&pmid2,(void*)&q4,(void*)&pout,
                (void*)&Nq,(void*)&Nz,(void*)&Nab,(void*)&Nqkv,(void*)&K,(void*)&G,(void*)&feps,(void*)&fsc,(void*)&fsg,
                (void*)&gc1,(void*)&gk1,(void*)&gg1,(void*)&gc2,(void*)&gk2,(void*)&gg2,
                (void*)&ac1,(void*)&ak1,(void*)&ag1,(void*)&ac2,(void*)&ak2,(void*)&ag2,
                (void*)&c3,(void*)&k3,(void*)&g3,(void*)&c4,(void*)&k4,(void*)&g4,
                (void*)&hk,(void*)&hv,(void*)&dk,(void*)&dv,(void*)&KoutG,
                (void*)&hh,(void*)&hkv,(void*)&dd,(void*)&nsp,(void*)&KoutA,(void*)&I2,
                (void*)&ppm,(void*)&ppl,(void*)&ppa,
                (void*)&plan,(void*)&cseq,(void*)&cbm,(void*)&pdbgp};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kfn,dim3(nblk),dim3(T),ka,XT2_*sizeof(float),st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // ПИНГ-ПОНГ: писал ПОСЛЕДНИЙ слой, а его номер nl-1, не nl. На чётном числе слоёв прежняя
    // формула возвращала чужой буфер -- ответ был осмысленный по форме и неверный по существу.
    return ((nl - 1) & 1) ? out : buf;
}

// ================================================================= Я5: ГОЛОВА СЛОВАРЯ + ВЫБОР
// Сейчас: голова считает 124 160 логитов на воркер (1.27 ГБ веса), кладёт их в HBM, потом отдельный
// сэмплер читает их обратно и берёт максимум. Ради ОДНОГО числа.
// Здесь максимум берётся ТАМ ЖЕ, где логит родился, -- в регистре. Наружу выходит пара
// (значение, индекс), а вектор логитов в HBM не появляется ВООБЩЕ. При жадном выборе это точно тот
// же ответ; для температурного пути наружу можно отдать усечённый список, но это отдельный ход.
__global__ void lm_head_argmax_kernel(const __half* __restrict__ x,     // [K]
                                      const __half* __restrict__ w,     // [K, N]
                                      float* __restrict__ pv, int* __restrict__ pi,
                                      int K, int N, int nblk_cols) {
    cg::grid_group grid = cg::this_grid();
    const int nb = gridDim.x, b = blockIdx.x, T = blockDim.x;
    __shared__ float rv[1024]; __shared__ int ri[1024];
    float bestv = -INFINITY; int besti = -1;
    for (long n = (long)b * T + threadIdx.x; n < N; n += (long)nb * T) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc = fmaf(__half2float(x[k]), __half2float(w[(long)k * N + n]), acc);
        if (acc > bestv) { bestv = acc; besti = (int)n; }
    }
    rv[threadIdx.x] = bestv; ri[threadIdx.x] = besti;
    __syncthreads();
    for (int s2 = T >> 1; s2; s2 >>= 1) {
        if (threadIdx.x < s2 && rv[threadIdx.x + s2] > rv[threadIdx.x]) {
            rv[threadIdx.x] = rv[threadIdx.x + s2]; ri[threadIdx.x] = ri[threadIdx.x + s2];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pv[b] = rv[0]; pi[b] = ri[0]; }
    ГБАР(grid);
    if (b == 0 && threadIdx.x == 0) {
        float bv = -INFINITY; int bi = -1;
        for (int i = 0; i < nb; ++i) if (pv[i] > bv) { bv = pv[i]; bi = pi[i]; }
        pv[nb] = bv; pi[nb] = bi;
    }
}

std::vector<torch::Tensor> lm_head_argmax(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.is_contiguous() && w.is_contiguous(),
                "lm_head_argmax: плотные тензоры на карте");
    const int K = (int)w.size(0), N = (int)w.size(1);
    TORCH_CHECK(x.numel() == K, "lm_head_argmax: собрано под M=1");
    const at::cuda::CUDAGuard guard(x.device());
    auto st = at::cuda::getCurrentCUDAStream();
    const int T = 256;
    const void* kfn = (const void*)lm_head_argmax_kernel;
    int perSM = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kfn, T, 0));
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int nblk = max(1, perSM * nsm);
    // ПРОВЕРКА ГИПОТЕЗЫ: если разница цепи и обёртки -- лишь ДРУГОЕ ДРОБЛЕНИЕ суммы, то при
    // одинаковом числе блоков она обязана исчезнуть. Гейт только для замера, умолчание прежнее.
    if (const char* e_ = getenv("FA2SM70_NBLK")) nblk = atoi(e_);
    auto pv = torch::empty({nblk + 1}, torch::dtype(torch::kFloat32).device(x.device()));
    auto pi = torch::empty({nblk + 1}, torch::dtype(torch::kInt32).device(x.device()));
    const __half* px = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const __half* pw = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
    float* ppv = pv.data_ptr<float>(); int* ppi = pi.data_ptr<int>();
    int nbc = nblk;
    void* ka[] = {(void*)&px,(void*)&pw,(void*)&ppv,(void*)&ppi,(void*)&K,(void*)&N,(void*)&nbc};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kfn, dim3(nblk), dim3(T), ka, 0, st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pv.narrow(0, nblk, 1), pi.narrow(0, nblk, 1)};
}

// Открыть прямой доступ между картами. Нужен для обмена ВНУТРИ ядра: без него запись в буфер
// соседа уходит в никуда, причём БЕЗ ошибки -- молча неверный результат.
// ==================================================================================================
// ОБЩАЯ ПАМЯТЬ МЕЖДУ ПРОЦЕССАМИ -- СВОИМИ РУКАМИ, а не механизмом torch.
// У torch этот путь рассчитан на разделение памяти ОДНОЙ карты между процессами: отображение
// создаётся в контексте карты-ВЛАДЕЛЬЦА, и полученный указатель непригоден для ядра, которое
// исполняется на НАШЕЙ карте. Здесь `cudaIpcOpenMemHandle` зовётся при ТЕКУЩЕЙ (нашей) карте
// и с ленивым включением прямого доступа -- ровно как в custom_all_reduce.
// Плюс своя выдача памяти: дескриптор берётся только с НАЧАЛА выделения, а распределитель torch
// отдаёт куски внутри большого блока.
// ==================================================================================================
std::tuple<int64_t, torch::Tensor> shared_alloc(int64_t nbytes) {
    void* p = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&p, (size_t)nbytes));
    C10_CUDA_CHECK(cudaMemset(p, 0, (size_t)nbytes));
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto h = torch::empty({(int64_t)sizeof(cudaIpcMemHandle_t)}, opts);
    C10_CUDA_CHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)h.data_ptr(), p));
    return std::make_tuple((int64_t)(uintptr_t)p, h);
}

int64_t shared_open(torch::Tensor handle) {
    TORCH_CHECK(handle.numel() == (int64_t)sizeof(cudaIpcMemHandle_t) &&
                handle.scalar_type() == torch::kByte && handle.device().is_cpu(),
                "shared_open: дескриптор должен быть байтовым тензором на хосте нужной длины");
    void* p = nullptr;
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&p, *((const cudaIpcMemHandle_t*)handle.data_ptr()),
                                        cudaIpcMemLazyEnablePeerAccess));
    return (int64_t)(uintptr_t)p;
}

// Свой буфер видим как тензор -- локальная сторона обмена остаётся проверяемой по форме и типу.
torch::Tensor shared_view(int64_t ptr, std::vector<int64_t> sizes, bool half) {
    auto opts = torch::TensorOptions()
                    .dtype(half ? torch::kHalf : torch::kInt)
                    .device(torch::kCUDA, at::cuda::current_device());
    return torch::from_blob((void*)(uintptr_t)ptr, sizes, opts);
}

void enable_p2p(int64_t a, int64_t b) {
    int can = 0;
    C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, (int)a, (int)b));
    TORCH_CHECK(can, "между картами нет прямого доступа");
    int cur; C10_CUDA_CHECK(cudaGetDevice(&cur));
    cudaSetDevice((int)a); cudaDeviceEnablePeerAccess((int)b, 0); cudaGetLastError();
    cudaSetDevice((int)b); cudaDeviceEnablePeerAccess((int)a, 0); cudaGetLastError();
    cudaSetDevice(cur);
}


// ==================================================================================================
// ПЛОТНЫЙ ОПЕРАТОР: ВАРП НА СТОЛБЕЦ. Три слота боевой сети НЕ квантованы (слой 0 целиком,
// qkv_proj во всех 16 слоях внимания, in_proj_ba во всех 48 GDN) -- разведка по живой модели.
// Раскладка torch [N, K]: k идёт подряд, n -- с шагом K. Поток-на-столбец (как в int8-пути) дал бы
// 32 отдельных обращения на варп; варп-на-столбец сливает чтение строки в непрерывные слова.
// ==================================================================================================
template <int THREADS>
__device__ __forceinline__ void gemv_dense_warp(const float* __restrict__ sx, int K,
                                                const __half* __restrict__ w, int n0, int nN,
                                                float* __restrict__ acc) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = THREADS >> 5;
    for (int n = n0 + warp; n < nN; n += nwarp) {
        const __half* row = w + (long)n * K;
        float sum = 0.f;
        // по 8 половинок за обращение: широкое чтение -- наш же приём (Ш3)
        for (int k = lane * 8; k < K; k += 32 * 8) {
            const int4 v = *reinterpret_cast<const int4*>(row + k);
            const __half2* h = reinterpret_cast<const __half2*>(&v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(h[j]);
                sum += f.x * sx[k + 2 * j] + f.y * sx[k + 2 * j + 1];
            }
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) acc[n - n0] = sum;
    }
}

// Отдельный вход для сверки против torch на БОЕВЫХ весах (не на случайных: на случайных я уже
// получал inf и делал ложные выводы).
template <int THREADS>
__global__ void dense_gemv_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                  __half* __restrict__ y, int K, int N) {
    extern __shared__ float smem_all[];
    float* sx = smem_all;                       // ОДНА область, а не два псевдонима на один адрес
    for (int i = threadIdx.x; i < K; i += THREADS) sx[i] = __half2float(x[i]);
    __syncthreads();
    const int per = (N + gridDim.x - 1) / gridDim.x;
    const int n0 = blockIdx.x * per, nN = min(N, n0 + per);
    float* acc = smem_all + K;
    gemv_dense_warp<THREADS>(sx, K, w, n0, nN, acc);
    __syncthreads();
    for (int n = n0 + threadIdx.x; n < nN; n += THREADS) y[n] = __float2half(acc[n - n0]);
}

torch::Tensor dense_gemv(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "нужны тензоры на карте");
    TORCH_CHECK(x.scalar_type() == torch::kHalf && w.scalar_type() == torch::kHalf, "нужен fp16");
    const int K = (int)w.size(1), N = (int)w.size(0);
    TORCH_CHECK((int)x.numel() == K, "длина входа не совпала с K веса");
    TORCH_CHECK(K % 256 == 0, "K обязан делиться на 256: варп читает по 8 половинок с шагом 256");
    auto y = torch::empty({N}, x.options());
    const int THREADS = 256;
    const int blocks = 80;                       // по числу SM, а не по геометрии выхода
    const int per = (N + blocks - 1) / blocks;
    const size_t shm = sizeof(float) * ((size_t)K + per);
    TORCH_CHECK(shm <= 96 * 1024, "не влезает в разделяемую: ", shm);
    cudaFuncSetAttribute(dense_gemv_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    dense_gemv_kernel<256><<<blocks, THREADS, shm, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (const __half*)w.data_ptr(), (__half*)y.data_ptr(), K, N);
    return y;
}


// ==================================================================================================
// РАЗБОР qkv У ВНИМАНИЯ Qwen3-Next. Снято с их кода, а не из общих соображений:
//   qkv.split([2*q_size, kv, kv]) -> q_gate.view(Hq, 2*d) -> chunk(2,-1)
//   => q и ГЕЙТ ЧЕРЕДУЮТСЯ ПО ГОЛОВЕ: [q(d) гейт(d)] x Hq, затем k, затем v.
//   q_norm/k_norm -- GemmaRMSNorm: x*rsqrt(mean+eps)*(1+w)   <-- ЕДИНИЦА, её у меня не было
//   RoPE ЧАСТИЧНЫЙ: rotary_dim=64 из d=256, стиль neox (половины 0..31 и 32..63)
//   выход внимания умножается на sigmoid(гейт) -- гейт готовим здесь же
// ==================================================================================================
template <int THREADS>
__global__ void qkv_post_kernel(const __half* __restrict__ qkv,
                                const __half* __restrict__ q_norm,
                                const __half* __restrict__ k_norm,
                                const float* __restrict__ cs,   // [rot/2 cos, rot/2 sin]
                                __half* __restrict__ q, __half* __restrict__ k,
                                __half* __restrict__ v, __half* __restrict__ gate,
                                int Hq, int Hkv, int D, int rot, float eps) {
    __shared__ float red[THREADS / 32];
    const int t = threadIdx.x, blk = blockIdx.x;
    const int kv0 = Hq * 2 * D;
    const bool is_q = blk < Hq;
    const bool is_k = !is_q && blk < Hq + Hkv;
    const int h = is_q ? blk : (is_k ? blk - Hq : blk - Hq - Hkv);

    if (!is_q && !is_k) {                       // ---- V: без нормы и без вращения
        for (int i = t; i < D; i += THREADS) v[h * D + i] = qkv[kv0 + Hkv * D + h * D + i];
        return;
    }
    const long base = is_q ? (long)h * 2 * D : (long)kv0 + (long)h * D;
    float x = (t < D) ? __half2float(qkv[base + t]) : 0.f;
    float ss = x * x;
    #pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    if (t == 0) {
        float s = 0.f;
        for (int i = 0; i < THREADS / 32; ++i) s += red[i];
        red[0] = rsqrtf(s / (float)D + eps);
    }
    __syncthreads();
    const __half* nw = is_q ? q_norm : k_norm;
    float xn = (t < D) ? x * red[0] * (1.f + __half2float(nw[t])) : 0.f;

    // ---- вращение ТОЛЬКО первых rot измерений; остальные проходят как есть
    __shared__ float sh[512];
    if (t < D) sh[t] = xn;
    __syncthreads();
    if (t < rot) {
        const int half = rot >> 1;
        const int j = t % half;
        const float c = cs[j], sn = cs[half + j];
        const float x1 = sh[j], x2 = sh[half + j];
        xn = (t < half) ? (x1 * c - x2 * sn) : (x2 * c + x1 * sn);
    }
    if (t < D) {
        if (is_q) {
            q[(long)h * D + t] = __float2half(xn);
            const float g = __half2float(qkv[(long)h * 2 * D + D + t]);
            gate[(long)h * D + t] = __float2half(1.f / (1.f + __expf(-g)));   // sigmoid СРАЗУ
        } else {
            k[(long)h * D + t] = __float2half(xn);
        }
    }
}

std::vector<torch::Tensor> qkv_post(torch::Tensor qkv, torch::Tensor q_norm, torch::Tensor k_norm,
                                    torch::Tensor cos_sin, int64_t Hq, int64_t Hkv, int64_t D,
                                    int64_t rot, double eps) {
    TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous(), "qkv_post: плотный тензор на карте");
    TORCH_CHECK(qkv.numel() == Hq * 2 * D + 2 * Hkv * D, "qkv_post: ширина не сходится с геометрией");
    TORCH_CHECK(D <= 512 && (rot & 1) == 0 && rot <= D, "qkv_post: D<=512 и чётный rot<=D");
    TORCH_CHECK(cos_sin.numel() == rot && cos_sin.scalar_type() == torch::kFloat,
                "qkv_post: строка cos/sin -- rot чисел fp32");
    auto o = qkv.options();
    auto q = torch::empty({Hq * D}, o), k = torch::empty({Hkv * D}, o);
    auto v = torch::empty({Hkv * D}, o), g = torch::empty({Hq * D}, o);
    qkv_post_kernel<256><<<(int)(Hq + 2 * Hkv), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)qkv.data_ptr(), (const __half*)q_norm.data_ptr(),
        (const __half*)k_norm.data_ptr(), cos_sin.data_ptr<float>(),
        (__half*)q.data_ptr(), (__half*)k.data_ptr(), (__half*)v.data_ptr(), (__half*)g.data_ptr(),
        (int)Hq, (int)Hkv, (int)D, (int)rot, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q, k, v, g};
}

template <int THREADS>
__global__ void qkv_post_grid_kernel(__half* mid, float* tmp, const __half* qn, const __half* kn,
                                     const float* cs, int Hq, int Hkv, int D, int rot, float eps) {
    cg::grid_group grid = cg::this_grid();
    qkv_post_grid<THREADS>(grid, mid, tmp, qn, kn, cs, Hq, Hkv, D, rot, eps, gridDim.x, blockIdx.x);
}

torch::Tensor qkv_post_g(torch::Tensor qkv, torch::Tensor q_norm, torch::Tensor k_norm,
                         torch::Tensor cos_sin, int64_t Hq, int64_t Hkv, int64_t D,
                         int64_t rot, double eps) {
    auto mid = qkv.clone();
    auto tmp = torch::zeros({(long)(2 * Hq * D + 2 * Hkv * D)}, qkv.options().dtype(torch::kFloat));
    void* ka[] = {(void*)mid.data_ptr(), (void*)tmp.data_ptr(), (void*)q_norm.data_ptr(),
                  (void*)k_norm.data_ptr(), (void*)cos_sin.data_ptr(),
                  (void*)&Hq, (void*)&Hkv, (void*)&D, (void*)&rot, (void*)&eps};
    // числа передаются по значению -- их надо взять как int/float, а не int64/double
    int hq = (int)Hq, hkv = (int)Hkv, dd = (int)D, rr = (int)rot; float ee = (float)eps;
    (void)ka;
    __half* pm = (__half*)mid.data_ptr(); float* pt = tmp.data_ptr<float>();
    const __half* pq = (const __half*)q_norm.data_ptr();
    const __half* pk = (const __half*)k_norm.data_ptr();
    const float* pc = cos_sin.data_ptr<float>();
    void* args[] = {(void*)&pm, (void*)&pt, (void*)&pq, (void*)&pk, (void*)&pc,
                    (void*)&hq, (void*)&hkv, (void*)&dd, (void*)&rr, (void*)&ee};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)qkv_post_grid_kernel<128>,
        dim3(40), dim3(128), args, 0, at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mid;
}

template <int THREADS>
__global__ void dense_norm_kernel(const __half* x, const __half* rw, const __half* w,
                                  __half* y, int K, int N, float eps) {
    const float rs = rms_scale2<THREADS>(x, K, eps);
    const int per = (N + gridDim.x - 1) / gridDim.x;
    gemv_dense_norm<THREADS>(x, rw, rs, K, w, blockIdx.x * per, min(N, (int)(blockIdx.x + 1) * per), y);
}

torch::Tensor dense_gemv_norm(torch::Tensor x, torch::Tensor rms_weight, double eps, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.is_contiguous() && w.is_contiguous(), "нужны плотные на карте");
    const int K = (int)w.size(1), N = (int)w.size(0);
    TORCH_CHECK((int)x.numel() == K && (int)rms_weight.numel() == K, "длины не сходятся с K");
    TORCH_CHECK(K % 256 == 0, "K обязан делиться на 256");
    auto y = torch::empty({N}, x.options());
    dense_norm_kernel<256><<<80, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (const __half*)rms_weight.data_ptr(), (const __half*)w.data_ptr(),
        (__half*)y.data_ptr(), K, N, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

__global__ void f2e4m3_kernel(const float* x, uint8_t* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2e4m3(x[i]);
}

torch::Tensor to_e4m3(torch::Tensor x) {
    auto xf = x.to(torch::kFloat).contiguous();
    auto y = torch::empty({xf.numel()}, xf.options().dtype(torch::kUInt8));
    const int n = (int)xf.numel();
    f2e4m3_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        xf.data_ptr<float>(), y.data_ptr<uint8_t>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

__global__ void from_e4m3_kernel(const uint8_t* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = e4m3_fast(x[i]);
}
torch::Tensor from_e4m3(torch::Tensor x) {
    auto y = torch::empty({x.numel()}, x.options().dtype(torch::kFloat));
    const int n = (int)x.numel();
    from_e4m3_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<uint8_t>(), y.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("from_e4m3", &from_e4m3, "развёртка ЯДРА -- для сверки соглашений");
    m.def("to_e4m3", &to_e4m3, "свёртка float -> e4m3, округление к ближайшему");
    m.def("dense_gemv_norm", &dense_gemv_norm, "плотный оператор с нормой Gemma-стиля на лету");
    m.def("qkv_post_g", &qkv_post_g, "тот же разбор, но сеткой (как в теле слоя)");
    m.def("qkv_post", &qkv_post, "разбор qkv внимания: q/гейт по голове, нормы, частичный RoPE");
    m.def("dense_gemv", &dense_gemv, "плотный fp16 оператор: варп на столбец");
    m.def("exch_selftest", &exch_selftest, "только сведение, тем же кодом и планом");
    m.def("shared_alloc", &shared_alloc, "выдать общую память и дескриптор к ней");
    m.def("shared_open", &shared_open, "раскрыть дескриптор соседа ПРИ НАШЕЙ текущей карте");
    m.def("shared_view", &shared_view, "смотреть на свой общий буфер как на тензор");
    m.def("enable_p2p", &enable_p2p, "открыть прямой доступ между двумя картами");
    m.def("decode_chain", &decode_chain, py::arg("x0"), py::arg("layers"), py::arg("is_gdn"),
          py::arg("group_size"), py::arg("eps"), py::arg("scale"), py::arg("scale_gdn") = 0.0,
          py::arg("Hk"), py::arg("Hv"), py::arg("H"), py::arg("Hkv"), py::arg("seq_len"),
          py::arg("rot") = 0, py::arg("kv_pos") = std::vector<int64_t>{},
          py::arg("qkvz_width") = 0, py::arg("up_offset") = 0, py::arg("mlp_width") = 0,
          py::arg("kscale") = 1.0, py::arg("vscale") = 1.0,
          py::arg("exch_tiers") = 0,
          py::arg("exch_peers") = std::vector<int64_t>{},
          py::arg("exch_pull") = std::vector<int64_t>{},
          py::arg("exch_far") = std::vector<int64_t>{},
          py::arg("exch_in") = std::vector<torch::Tensor>{},
          py::arg("exch_far_flag") = std::vector<int64_t>{},
          py::arg("exch_my_flag") = std::vector<torch::Tensor>{},
          py::arg("exch_width") = 0, py::arg("exch_count") = c10::nullopt,
          py::arg("peer_inbox") = c10::nullopt, py::arg("my_inbox") = c10::nullopt,
          py::arg("my_flag") = c10::nullopt, py::arg("peer_flag") = c10::nullopt,
          py::arg("poz_dev") = c10::nullopt, py::arg("slot_idx") = c10::nullopt,
          py::arg("dbg_plita") = c10::nullopt,
          "ЦЕПЬ: несколько слоёв обоих типов в ОДНОМ кооперативном пуске");
    m.def("attn_layer", &attn_layer, py::arg("x"), py::arg("rw"), py::arg("eps"),
          py::arg("qw_p"), py::arg("qz_p"), py::arg("sc_p"), py::arg("qw_o"), py::arg("qz_o"), py::arg("sc_o"),
          py::arg("kc"), py::arg("vc"), py::arg("rw2"),
          py::arg("qw_g2"), py::arg("qz_g2"), py::arg("sc_g2"), py::arg("qw_u2"), py::arg("qz_u2"), py::arg("sc_u2"),
          py::arg("qw_d2"), py::arg("qz_d2"), py::arg("sc_d2"),
          py::arg("group_size"), py::arg("scale"), py::arg("H"), py::arg("Hkv"),
          py::arg("kpool"), py::arg("vpool"), py::arg("block_table"), py::arg("block_size"),
          py::arg("seq_len"), py::arg("kscale"), py::arg("vscale"),
          py::arg("dense_qkv") = c10::nullopt, py::arg("q_norm") = c10::nullopt,
          py::arg("k_norm") = c10::nullopt, py::arg("cos_sin") = c10::nullopt,
          py::arg("rot") = 0, py::arg("kv_pos") = -1, py::arg("dbg") = c10::nullopt,
          py::arg("up_offset") = 0, py::arg("mlp_width") = 0,
          "СБОРКА: слой внимания одним пуском -- норма, qkv, ядро внимания, o_proj");
    m.def("gdn_layer", &gdn_layer, py::arg("x"), py::arg("rw"), py::arg("eps"),
          py::arg("qw_q"), py::arg("qz_q"), py::arg("sc_q"), py::arg("qw_z"), py::arg("qz_z"), py::arg("sc_z"),
          py::arg("w_ab"), py::arg("qw_o"), py::arg("qz_o"), py::arg("sc_o"),
          py::arg("conv_w"), py::arg("conv_state"), py::arg("a_log"), py::arg("dt_bias"), py::arg("hstate"),
          py::arg("rw2"), py::arg("qw_g2"), py::arg("qz_g2"), py::arg("sc_g2"),
          py::arg("qw_u2"), py::arg("qz_u2"), py::arg("sc_u2"),
          py::arg("qw_d2"), py::arg("qz_d2"), py::arg("sc_d2"),
          py::arg("group_size"), py::arg("scale"), py::arg("Hk"), py::arg("Hv"),
          py::arg("total_width") = 0, py::arg("q_width") = 0,
          py::arg("up_offset") = 0, py::arg("mlp_width") = 0,
          "СБОРКА: весь слой GDN одним пуском -- норма, проекции, свёртка, мешалка, out_proj");
    m.def("lm_head_argmax", &lm_head_argmax,
          "Я5: голова словаря + выбор максимума В РЕГИСТРЕ, логиты в HBM не выходят");
    m.def("proj_from", &proj_from, "Я2в: проекция из готового вектора, без нормы");
    m.def("attn_decode", &attn_decode, "Я3: ядро внимания на декоде, онлайн-softmax со срезами");
    m.def("gdn_mixer", &gdn_mixer,
          "Я2б: мешалка GDN на декоде -- l2norm, гейт, дельта-правило, выход");
    m.def("mlp_body", &mlp_body, py::arg("x"), py::arg("rms_weight"), py::arg("eps"),
          py::arg("qw_g"), py::arg("qz_g"), py::arg("sc_g"), py::arg("qw_u"), py::arg("qz_u"), py::arg("sc_u"),
          py::arg("qw_d"), py::arg("qz_d"), py::arg("sc_d"), py::arg("group_size"), py::arg("nlayers"),
          py::arg("peer_inbox"), py::arg("my_inbox"), py::arg("my_flag"), py::arg("peer_flag"),
          py::arg("seq0"), py::arg("up_col_offset") = 0, py::arg("total_width") = 0,
          "Я1: всё тело MLP одним запуском -- норма + gate_up + silu*mul + down");
    m.def("gdn_proj", &gdn_proj, py::arg("x"), py::arg("rms_weight"), py::arg("eps"),
          py::arg("qw_q"), py::arg("qz_q"), py::arg("sc_q"), py::arg("qw_z"), py::arg("qz_z"), py::arg("sc_z"),
          py::arg("w_ab"), py::arg("group_size"), py::arg("total_width") = 0, py::arg("q_width") = 0,
          "Я2а: норма + in_proj_qkv + in_proj_z ОДНИМ проходом по общему входу");
}
