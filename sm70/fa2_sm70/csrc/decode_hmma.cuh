// -------------------------------------------------------------------------------------------
// ХИМЕРНОЕ ДЕКОДНОЕ ВНИМАНИЕ: плитка K/V в разделяемой памяти + обе GEMM на тензорных ядрах.
//
// ЗАЧЕМ. Профиль боевых форм (Sk=262144, int8, ns=120, карта 1), записка 25 §98/§98a:
//   * qGrp=1 -> 835 мкс, qGrp=4 -> 3280 мкс при ОДНОЙ полосе (328/337 ГБ/с): каждая строка
//     спекуляции проходит по KV заново (полезных 268 МБ, читается 1.07 ГБ);
//   * по конвейерам FMA 53.5% инструкций, а FP16 (тензорные) 0.03% -- тензорные ядра в
//     декодном внимании не использовались вообще.
// Половинки цели не берут: одна плитка упирается в FMA (x1.87), один HMMA -- в четырёхкратное
// чтение (x1.03). Вместе: чтение однократно + арифметика на HMMA -> расчётные x3.6-4.1.
//
// РАСКЛАДКА. Блок = (запрос, KV-голова, срез ключей). Строки плитки -- qGrp строк спекуляции
// x GF голов запроса ОДНОЙ KV-головы: ровно те, что раньше читали KV порознь.
//   S = Q*K^T : A=[M][D] k-непрерывно, B=[KT][D] k-непрерывно -> .row.col (kKMajor)
//   O += P*V  : A=[M][KT] k-непрерывно, B=[KT][D] n-непрерывно -> .row.row (kNMajor)
// Обе ложатся на готовый WarpMma без перестановок -- ради этого раскладка и выбрана.
//
// ВАРПЫ. KT=32 -> две плитки по 16 ключей, MT=32 -> две по 16 строк: варп w берёт строки
// (w>>1)*16 и ключи (w&1)*16, так заняты все четыре. Во втором GEMM варп берёт те же 16 строк
// и половину измерения D.
// -------------------------------------------------------------------------------------------
#pragma once
#include "gemm/volta_warp_mma.h"

namespace fa2sm70_dechmma {

using fa2_sm70::WarpMma;
using fa2_sm70::BLayout;
using fa2_sm70::ALayout;
using fa2_sm70::wmma_acc_row;
using fa2_sm70::wmma_acc_col;
using fa2_sm70::wmma_b_col;
using fa2_sm70::wmma_b_krow;
using fa2_sm70::wmma_b_nbase;

// KT=64 и восемь варпов -- не украшение: при KT=32 и четырёх варпах плитка S (32x32) даёт
// ровно 4 плитки 16x16, половина варпов простаивает во втором GEMM, а занятость выходит
// 6.25% (замер первой версии: x0.87 против скалярного ядра при вчетверо меньшем чтении --
// то есть упор был в занятость и барьеры, а не в байты).
// [V3, 11.09] ДВА БЛОКА НА SM. ncu на 262144: warps_active 12.5 % (1 блок из 8 варпов, smem 81 КБ,
// 181->138 регистров), стойла wait/short_scoreboard/barrier -- латентность, не полоса и не тензорные
// ядра (15 %). Лечится только занятостью: KT=32, ОДИН буфер K (предвыборка следующей плитки живёт в
// регистрах и кладётся после барьера S-фазы), S-фаза всеми 8 варпами через РАЗРЕЗ D пополам с двумя
// частичными S в разделяемой. Итог 46 КБ + __launch_bounds__(256, 2). Выключатель: FA2SM70_DECHMMA_V3=0.
#ifndef FA2SM70_DECHMMA_V3
#define FA2SM70_DECHMMA_V3 1
#endif
constexpr bool V3 = FA2SM70_DECHMMA_V3 != 0;
constexpr int KT = V3 ? 32 : 64;  // ключей в плитке
constexpr int MT = 32;            // строк плитки (qGrp*GF, добивается нулями)
constexpr int NW = 8;             // варпов в блоке
constexpr int NT = NW * 32;
constexpr int KBUF = V3 ? 1 : 2;  // буферов K
// [V5, 12.09] S-фаза без дублирования декванта K: варп = (группа 16 ключей, ЧЕТВЕРТЬ D), обе группы
// строк (MB=2); частичные S складываются атомарно (RED.ADD shared) в ОДИН буфер, обнуляемый после
// софтмакса. Поправка строк -- в отдельном массиве popr. Требует V3. Выключатель FA2SM70_DECHMMA_V5=0.
#ifndef FA2SM70_DECHMMA_V5
#define FA2SM70_DECHMMA_V5 0   // замер 12.09: атомики МЕДЛЕННЕЕ (16.2 против 13.3 мс/шаг) -- выключено
#endif
constexpr bool V5 = V3 && (FA2SM70_DECHMMA_V5 != 0);
constexpr int NS = V5 ? 1 : (V3 ? 2 : 1);    // частичных S
constexpr int SMB = V5 ? 2 : 1;              // групп строк на варп в S-фазе
constexpr int LDK = 256 + 16;     // шаг строки плитки K/V в байтах (кратен 16)
constexpr int LDQ = 256 + 8;      // шаг строки Q в half (528 Б, кратно 16)
constexpr int LDP = KT + 4;       // шаг строки P/S (136 Б, кратно 8)

template <int D>
struct Obshchaya {
    // Двойной буфер держит только K: с двумя буферами на обе плитки разделяемая память
    // выходит за 96 КиБ (замер: запуск отвергается с invalid argument). V перекрывается
    // иначе -- её чтение выдаётся в регистры в начале тела и укладывается перед вторым GEMM,
    // то есть идёт параллельно первому.
    int8_t sK[KBUF][KT * LDK];
    int8_t sV[KT * LDK];
    float  шк[2][KT], шv[KT];        // шкалы K всегда в двух буферах (V3 читает их в софтмаксе)         // шкалы ключей плитки
    __half sQ[MT * LDQ];
    float  S[NS][MT * LDP];           // V3: два частичных S (половины D), сумма и маска -- в софтмаксе
    __half P[MT * LDP];
    float  mrow[MT], lrow[MT];
    float  popr[MT];                  // поправка строк софтмакса (раньше жила в S[.][стр*LDP])
};

// Фрагмент из int8 БЕЗ шкалы. Шкала у нас на ТОКЕН, а не на элемент, поэтому она выносится
// за скалярное произведение целиком: <q, k_i*s_i> = s_i*<q, k_i> -- одно умножение на выход
// вместо четырёх на каждый фрагмент. Для значений то же: sum p_i*(v_i*s_i) = sum (p_i*s_i)*v_i,
// то есть шкала вносится в ВЕСА при записи P. Замер до этого шага: Compute 62.9% при DRAM
// 14.5%, то есть лишняя арифметика и была узким местом.
// [ДЕКВАНТ БЕЗ I2F, 11.09] int8 -> fp16 через магическую экспоненту: байт u = x^0x80 (= x+128) кладётся
// PRMT в младшие 8 бит полуслова с показателем 1024 (0x64uu -> 1024+u, ulp = 1), затем HSUB2 на 1152.
// Прежний путь: 4 x __int2half_rn на 4 элемента -- I2F идёт квартерной скоростью (XU), и по счёту
// инструкций деквант фрагментов стоил ~10 тыс. на варп на плитку (химера мерялась 210 ГБ/с при
// полосе 900 -- упор в конвейер выдачи, не в HBM и не в тензорные ядра). Здесь на 4 элемента:
// 1 LOP3 + 2 PRMT + 2 HSUB2, все на полноскоростных конвейерах. Выключатель для A/B: FA2SM70_DECHMMA_PRMT=0.
#ifndef FA2SM70_DECHMMA_PRMT
#define FA2SM70_DECHMMA_PRMT 1
#endif
__device__ __forceinline__ void i8x4_to_h2x2(uint32_t raw, uint32_t& lo, uint32_t& hi) {
#if FA2SM70_DECHMMA_PRMT
    const uint32_t t = raw ^ 0x80808080u;
    uint32_t a = __byte_perm(t, 0x64646464u, 0x5140u);   // [b0,0x64,b1,0x64] -> h0=1024+u0, h1=1024+u1
    uint32_t b = __byte_perm(t, 0x64646464u, 0x5342u);   // [b2,0x64,b3,0x64]
    const uint32_t magic = 0x64806480u;                   // half2(1152, 1152)
    __half2 ha = __hsub2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&magic));
    __half2 hb = __hsub2(*reinterpret_cast<const __half2*>(&b), *reinterpret_cast<const __half2*>(&magic));
    lo = *reinterpret_cast<const uint32_t*>(&ha);
    hi = *reinterpret_cast<const uint32_t*>(&hb);
#else
    const int8_t* bb = reinterpret_cast<const int8_t*>(&raw);
    __half h[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) h[e] = __int2half_rn((int)bb[e]);
    lo = reinterpret_cast<const uint32_t*>(h)[0];
    hi = reinterpret_cast<const uint32_t*>(h)[1];
#endif
}
__device__ __forceinline__ void frag_i8(uint32_t (&fb)[2], const int8_t* p) {
    i8x4_to_h2x2(*reinterpret_cast<const uint32_t*>(p), fb[0], fb[1]);
}
__device__ __forceinline__ void frag_i8x2(uint32_t (&f0)[2], uint32_t (&f1)[2], const int8_t* p) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    i8x4_to_h2x2(raw.x, f0[0], f0[1]);
    i8x4_to_h2x2(raw.y, f1[0], f1[1]);
}
template <int D>
__device__ __forceinline__ void plitka(int8_t* sh, float* шк, const int8_t* __restrict__ src,
                                       const float* __restrict__ shk, int nkey, int tid,
                                       const int* __restrict__ bt, int breq, int hkv,
                                       int bs_log, int mb, int Hkv, int k0, long шаг_блока) {
    for (int i = tid * 16; i < nkey * D; i += NT * 16) {
        const int кл = i / D, эл = i - кл * D;
        const int gk = k0 + кл;
        // Сдвиг вместо деления: block_size -- степень двойки (проверено в точке входа), а
        // деление на ПЕРЕМЕННУЮ компилятор разворачивает в длинную цепочку IMAD. Замер:
        // ALU 80.5 млн инструкций из 229 млн -- главный конвейер после снятия прочих упоров.
        const int blk = bt[(long)breq * mb + (gk >> bs_log)];
        const int wb = gk & ((1 << bs_log) - 1);
        *reinterpret_cast<uint4*>(sh + (long)кл * LDK + эл) =
            *reinterpret_cast<const uint4*>(src + ((long)blk * шаг_блока + (long)wb * Hkv + hkv) * D + эл);
    }
    // ХВОСТ ПЛИТКИ ОБЯЗАН БЫТЬ ОБНУЛЁН, А НЕ ОСТАВЛЕН КАК ЕСТЬ. Вес такого ключа равен нулю
    // (его счёт -inf), но ноль УМНОЖАЕТСЯ на шкалу, а неинициализированная разделяемая память
    // легко содержит битовый узор inf/NaN -- и тогда 0*inf = NaN расходится по всему выходу.
    // Офлайн этого не видно: там память чиста, и ядро совпадало со старым до 1.4e-04, а в
    // сервере разделяемую только что использовали чужие ядра.
    for (int кл = nkey + tid; кл < KT; кл += NT) шк[кл] = 0.f;
    for (int кл = tid; кл < nkey; кл += NT) {
        const int gk = k0 + кл;
        const int blk = bt[(long)breq * mb + (gk >> bs_log)];
        // Таблица шкал ПЛОТНАЯ -- [num_blocks, block_size, Hkv], без удвоения.
        const long слот = (((long)blk << bs_log) + (gk & ((1 << bs_log) - 1))) * Hkv + hkv;
        шк[кл] = shk ? shk[слот] : 1.0f;
    }
}

template <int D>
__global__ __launch_bounds__(NT, V3 ? 2 : 1) void decode_hmma_kernel(
        const __half* __restrict__ Q, const int8_t* __restrict__ K, const int8_t* __restrict__ V,
        const float* __restrict__ sK, const float* __restrict__ sV,
        float* __restrict__ Aout, float* __restrict__ Bout,
        int H, int Hkv, int Sk, float scale, int ns,
        const int* __restrict__ block_table, int bs_log, int max_blocks,
        const int* __restrict__ seqlens, int qGrp, int GF, int qCausal, long шаг_блока,
        const int* __restrict__ selIdx, const int* __restrict__ selCnt, int selStride, int selBlkLog) {
    extern __shared__ char smem_raw[];
    Obshchaya<D>* sh = reinterpret_cast<Obshchaya<D>*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int строк = qGrp * GF;
    const int b = blockIdx.x, hkv = blockIdx.y, si = blockIdx.z;
    const int стр0 = (warp >> 2) * 16;                 // 16 строк этого варпа
    const int кл0  = V5 ? ((warp >> 2) * 16) : (V3 ? (((warp >> 1) & 1) * 16) : ((warp & 3) * 16));   // 16 ключей варпа
    const int kh   = V5 ? (warp & 3) : (V3 ? (warp & 1) : 0);   // V5: четверть D; V3: половина D
    constexpr int KSEG = V5 ? (D / 4) : (V3 ? (D / 2) : D);
    const int k_lo = kh * KSEG, k_hi = k_lo + KSEG;

    // Длина -- по САМОЙ ДЛИННОЙ строке запроса; причинность строк учитывается маской ниже,
    // как и в скалярном ядре (строка r видит на (qGrp-1-r) ключей меньше).
    // Длины подаются НА ЗАПРОС (как и таблица блоков), а строка r видит на (qGrp-1-r) ключей
    // меньше -- ровно формула скалярного ядра. Индексировать длины по СТРОКЕ было бы тихой
    // порчей: при qGrp=4 читалась бы длина чужого запроса.
    const int len = seqlens ? seqlens[b] : Sk;
    const int на_срез = ((len + KT - 1) / KT + ns - 1) / ns * KT;   // срез кратен плитке
    const int k0 = si * на_срез, k1 = min(len, k0 + на_срез);
    // [СПИСОК БЛОКОВ ОТБОРА, 14.09] Профиль шага на 250K: с индексом отбора движок уходил на скалярное ядро
    // (15 % ключей стоили 65 % плотного HMMA). Здесь тот же цикл по плиткам, но плитки берутся из СПИСКА
    // отобранных блоков (блок = 1<<selBlkLog ключей = TPB плиток), срез расщепления -- по ЧИСЛУ плиток.
    // Список -- строки 0 запроса (отбор с группой строк даёт один список на запрос). Без списка -- бит в бит прежнее.
    const int* sel = nullptr; int TPB = 1, T0 = 0, T1 = (k0 < k1) ? ((k1 - k0 + KT - 1) / KT) : 0, lim = k1;
    if (selIdx != nullptr) {
        const long o = (long)(b * qGrp) * Hkv + hkv;
        sel = selIdx + o * (long)selStride;
        TPB = (1 << selBlkLog) / KT;
        const int ntiles = selCnt[o] * TPB;
        const int per = (ntiles + ns - 1) / ns;
        T0 = min(si * per, ntiles); T1 = min(T0 + per, ntiles); lim = len;
    }
    auto tstart = [&](int t) -> int { return sel ? ((sel[t / TPB] << selBlkLog) + (t - (t / TPB) * TPB) * KT) : (k0 + t * KT); };

    // [V4, 11.09] PV: варп держит ОБЕ группы строк (32) и 32 столбца d (warp*32) -- фрагмент V
    // конвертируется одним варпом, а не двумя (прежде варпы (группа строк, четверть d) дублировали
    // деквант V). Регистры накопителя те же: 2x2x8 = 32. Выключатель FA2SM70_DECHMMA_V4=0.
#ifndef FA2SM70_DECHMMA_V4
#define FA2SM70_DECHMMA_V4 1
#endif
    constexpr bool V4 = FA2SM70_DECHMMA_V4 != 0;
    constexpr int OMB = V4 ? 2 : 1, ONB = V4 ? 2 : (D / 64);
    WarpMma<OMB, ONB, BLayout::kNMajor> mo; mo.clear();
    const int d0 = V4 ? (warp * 32) : ((warp & 3) * (D / 4));

    if (T0 < T1) {
        for (int i = tid; i < MT * D; i += NT) {
            const int стр = i / D, эл = i - стр * D;
            __half v = __float2half(0.f);
            if (стр < строк) {
                const int r = стр / GF, g = стр - r * GF;
                v = __hmul(Q[((long)(b * qGrp + r) * H + hkv * GF + g) * D + эл], __float2half(scale));
            }
            sh->sQ[(long)стр * LDQ + эл] = v;
        }
        for (int i = tid; i < MT; i += NT) { sh->mrow[i] = -CUDART_INF_F; sh->lrow[i] = 0.f; }
        if constexpr (V5) for (int i = tid; i < MT * LDP; i += NT) sh->S[0][i] = 0.f;
        __syncthreads();

        // ---- ГЛАВНЫЙ ЦИКЛ: два буфера. Плитка kb+KT выдаётся в РЕГИСТРЫ в начале тела и
        // ложится в разделяемую только в конце -- её чтение из памяти идёт параллельно всей
        // арифметике текущей плитки. Без этого ядро при одном блоке на SM простаивает.
        constexpr int ЗАГР = (KT * D) / NT / 16;        // uint4 на поток за плитку
        uint4 pfK[ЗАГР], pfV[ЗАГР];
        int буф = 0;
        plitka<D>(sh->sK[0], sh->шк[0], K, sK, max(0, min(KT, lim - tstart(T0))), tid, block_table, b, hkv,
                  bs_log, max_blocks, Hkv, tstart(T0), шаг_блока);
        __syncthreads();

        for (int t = T0; t < T1; ++t) {
            const int kb = tstart(t);
            const int nk = max(0, min(KT, lim - kb));
            const bool есть_ещё = (t + 1 < T1);
            const int kn = есть_ещё ? tstart(t + 1) : (kb + KT);
            // Чтения выдаются ДО арифметики: V этой плитки и K следующей.
#pragma unroll
            for (int c = 0; c < ЗАГР; ++c) {
                const int i = (tid + c * NT) * 16;
                if (i < nk * D) {
                    const int кл = i / D, эл = i - кл * D;
                    const int gk = kb + кл;
                    const long blk = block_table[(long)b * max_blocks + (gk >> bs_log)];
                    const long wb = gk & ((1 << bs_log) - 1);
                    pfV[c] = *reinterpret_cast<const uint4*>(V + (blk * шаг_блока + wb * Hkv + hkv) * D + эл);
                }
            }
            if (есть_ещё) {
                const int nk2 = max(0, min(KT, lim - kn));
#pragma unroll
                for (int c = 0; c < ЗАГР; ++c) {
                    const int i = (tid + c * NT) * 16;
                    if (i < nk2 * D) {
                        const int кл = i / D, эл = i - кл * D;
                        const int gk = kn + кл;
                        const long blk = block_table[(long)b * max_blocks + (gk >> bs_log)];
                        const long wb = gk & ((1 << bs_log) - 1);
                        pfK[c] = *reinterpret_cast<const uint4*>(K + (blk * шаг_блока + wb * Hkv + hkv) * D + эл);
                    }
                }
            }
            // S = Q * K^T на тензорных ядрах; B-фрагмент деквантуется из int8 прямо здесь.
            WarpMma<SMB, 1, BLayout::kKMajor> ms; ms.clear();
            uint32_t fa[SMB][2], fb[1][2];
            const int8_t* pk = sh->sK[V3 ? 0 : буф] + (long)(кл0 + wmma_b_col(lane)) * LDK;
            const __half* pq = sh->sQ + (long)((V5 ? 0 : стр0) + fa2_sm70::wmma_a_row(lane)) * LDQ;
            uint32_t fa2[SMB][2], fb2[1][2];
            // [11.09] ncu: стойла wait/short_scoreboard (цепь LDS -> PRMT/HSUB2 -> mma) при 8 варпах/SM.
            // Развёртка даёт планировщику перекрыть деквант шага k+8 с mma шага k. Выключатель для A/B.
#ifndef FA2SM70_DECHMMA_UNROLL
#define FA2SM70_DECHMMA_UNROLL 4
#endif
#pragma unroll 4
            for (int k = k_lo; k < k_hi; k += 8) {
#pragma unroll
                for (int i = 0; i < SMB; ++i) {
                    const uint4 q4 = *reinterpret_cast<const uint4*>(pq + (long)i * 16 * LDQ + k);   // 8 half = два шага
                    fa[i][0] = q4.x; fa[i][1] = q4.y; fa2[i][0] = q4.z; fa2[i][1] = q4.w;
                }
                frag_i8x2(fb[0], fb2[0], pk + k);
                ms.mma(fa, fb);
                ms.mma(fa2, fb2);
            }
#pragma unroll
            for (int i = 0; i < SMB; ++i)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int стр = (V5 ? i * 16 : стр0) + wmma_acc_row(lane, r);
                const int кл  = кл0  + wmma_acc_col(lane, r);
                if constexpr (V5) {
                    atomicAdd(&sh->S[0][(long)стр * LDP + кл], ms.acc[i][0][r]);   // четыре четверти D складываются
                } else if constexpr (V3) {
                    sh->S[kh][(long)стр * LDP + кл] = ms.acc[0][0][r];   // маска и шкала -- в софтмаксе
                } else {
                    bool годен = (кл < nk) && (стр < строк);
                    if (годен && qCausal) {
                        const int r_ = стр / GF;
                        годен = (kb + кл) < (len - (qGrp - 1 - r_));
                    }
                    // Шкала ключа вносится ЗДЕСЬ, один раз на выход.
                    sh->S[0][(long)стр * LDP + кл] = годен ? ms.acc[0][0][r] * sh->шк[буф][кл] : -CUDART_INF_F;
                }
            }
            // Укладка V: её чтение шло параллельно первому GEMM.
#pragma unroll
            for (int c = 0; c < ЗАГР; ++c) {
                const int i = (tid + c * NT) * 16;
                if (i < nk * D) {
                    const int кл_ = i / D, эл_ = i - кл_ * D;
                    *reinterpret_cast<uint4*>(sh->sV + (long)кл_ * LDK + эл_) = pfV[c];
                }
            }
            for (int кл = nk + tid; кл < KT; кл += NT) sh->шv[кл] = 0.f;
            for (int кл = tid; кл < nk; кл += NT) {
                const int gk = kb + кл;
                const long слот = (((long)block_table[(long)b * max_blocks + (gk >> bs_log)] << bs_log)
                                   + (gk & ((1 << bs_log) - 1))) * Hkv + hkv;
                sh->шv[кл] = sV ? sV[слот] : 1.f;
            }
            __syncthreads();
            if constexpr (V3) {                          // ОДИН буфер K: следующая плитка кладётся ПОСЛЕ барьера S-фазы,
                if (есть_ещё) {                          // пока идёт софтмакс; шкалы -- во второй буфер, их читает софтмакс
                    const int nk2 = max(0, min(KT, lim - kn));
#pragma unroll
                    for (int c = 0; c < ЗАГР; ++c) {
                        const int i = (tid + c * NT) * 16;
                        if (i < nk2 * D) {
                            const int кл_ = i / D, эл_ = i - кл_ * D;
                            *reinterpret_cast<uint4*>(sh->sK[0] + (long)кл_ * LDK + эл_) = pfK[c];
                        }
                    }
                    for (int кл = tid; кл < nk2; кл += NT) {
                        const int gk = kn + кл;
                        const long слот = (((long)block_table[(long)b * max_blocks + (gk >> bs_log)] << bs_log)
                                           + (gk & ((1 << bs_log) - 1))) * Hkv + hkv;
                        sh->шк[буф ^ 1][кл] = sK ? sK[слот] : 1.f;
                    }
                }
            }
            // Онлайн-софтмакс: СТРОКА НА ВАРП, редукция через shuffle. [11.09] Развёртка по строкам
            // (MT/NW = 4 витка, счётный цикл): четыре независимые shuffle-цепи вместо одной за другой.
#pragma unroll
            for (int стр = warp; стр < MT; стр += NW) {
                const int c0 = lane, c1 = lane + 32;
                float z0, z1;
                if constexpr (V3) {
                    const int кл = c0;
                    bool годен = (кл < nk) && (стр < строк);
                    if (годен && qCausal) { const int r_ = стр / GF; годен = (kb + кл) < (len - (qGrp - 1 - r_)); }
                    z0 = годен ? (V5 ? sh->S[0][(long)стр * LDP + кл]
                                     : (sh->S[0][(long)стр * LDP + кл] + sh->S[NS - 1][(long)стр * LDP + кл])) * sh->шк[буф][кл]
                               : -CUDART_INF_F;
                    z1 = -CUDART_INF_F;
                } else {
                    z0 = sh->S[0][(long)стр * LDP + c0]; z1 = sh->S[0][(long)стр * LDP + c1];
                }
                float mx = fmaxf(z0, z1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                const float мп = sh->mrow[стр];
                const float M = fmaxf(мп, mx);
                const float поправка = (мп == -CUDART_INF_F) ? 1.f : __expf(мп - M);
                const float e0 = (z0 == -CUDART_INF_F) ? 0.f : __expf(z0 - M);
                const float e1 = (z1 == -CUDART_INF_F) ? 0.f : __expf(z1 - M);
                float sm = e0 + e1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
                // Шкала ЗНАЧЕНИЙ вносится в веса -- тогда второй GEMM берёт V как есть.
                sh->P[(long)стр * LDP + c0] = __float2half(e0 * sh->шv[c0]);
                if constexpr (!V3) sh->P[(long)стр * LDP + c1] = __float2half(e1 * sh->шv[c1]);
                if (lane == 0) {
                    sh->lrow[стр] = sh->lrow[стр] * поправка + sm;
                    sh->mrow[стр] = M;
                    sh->popr[стр] = поправка;
                }
            }
            __syncthreads();
            if constexpr (V5) for (int i = tid; i < MT * LDP; i += NT) sh->S[0][i] = 0.f;   // под следующую плитку; PV S не читает
#pragma unroll
            for (int i = 0; i < OMB; ++i)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float попр = sh->popr[(V4 ? i * 16 : стр0) + wmma_acc_row(lane, r)];
                if (попр == 1.f) continue;
#pragma unroll
                for (int j = 0; j < ONB; ++j) mo.acc[i][j][r] *= попр;
            }
            // O += P * V. V берётся n-мажорно: 4 соседних столбца d одного ключа = 4 байта.
            uint32_t pa[OMB][2], pb[ONB][2];
            const __half* pp = sh->P + (long)((V4 ? 0 : стр0) + fa2_sm70::wmma_a_row(lane)) * LDP;
#pragma unroll 4
            for (int k = 0; k < KT; k += 4) {
#pragma unroll
                for (int i = 0; i < OMB; ++i) {
                    const uint2 p2 = *reinterpret_cast<const uint2*>(pp + (long)i * 16 * LDP + k);
                    pa[i][0] = p2.x; pa[i][1] = p2.y;
                }
                const int8_t* pv = sh->sV + (long)(k + wmma_b_krow(lane)) * LDK + d0 + wmma_b_nbase(lane);
#pragma unroll
                for (int j = 0; j < ONB; ++j) frag_i8(pb[j], pv + j * 16);
                mo.mma(pa, pb);
            }
            __syncthreads();
            if constexpr (V3) { if (есть_ещё) буф ^= 1; }   // V3: K уже уложен после барьера S-фазы; переключаем только шкалы
            if (!V3 && есть_ещё) {                              // укладка предвыбранной плитки (два буфера)
                const int nk2 = max(0, min(KT, lim - kn));
#pragma unroll
                for (int c = 0; c < ЗАГР; ++c) {
                    const int i = (tid + c * NT) * 16;
                    if (i < nk2 * D) {
                        const int кл_ = i / D, эл_ = i - кл_ * D;
                        *reinterpret_cast<uint4*>(sh->sK[буф ^ 1] + (long)кл_ * LDK + эл_) = pfK[c];
                    }
                }
                for (int кл = tid; кл < nk2; кл += NT) {
                    const int gk = kn + кл;
                    const long слот = (((long)block_table[(long)b * max_blocks + (gk >> bs_log)] << bs_log)
                                       + (gk & ((1 << bs_log) - 1))) * Hkv + hkv;
                    sh->шк[буф ^ 1][кл] = sK ? sK[слот] : 1.f;
                }
                __syncthreads();
                буф ^= 1;
            }
        }
    }
    // Выдача среза: A -- накопленный O (без нормировки), B -- (max, sum) на строку.
#pragma unroll
    for (int i = 0; i < OMB; ++i)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int стр = (V4 ? i * 16 : стр0) + wmma_acc_row(lane, r);
        if (стр >= строк) continue;
        const int rq = стр / GF, g = стр - rq * GF;
        const long баз = ((long)(b * qGrp + rq) * H + hkv * GF + g) * (long)ns * D + (long)si * D;
#pragma unroll
        for (int j = 0; j < ONB; ++j) {
            const int col = d0 + j * 16 + wmma_acc_col(lane, r);
            if (col < D) Aout[баз + col] = (T0 < T1) ? mo.acc[i][j][r] : 0.f;
        }
    }
    for (int стр = tid; стр < строк; стр += NT) {
        const int rq = стр / GF, g = стр - rq * GF;
        const long бб = (((long)(b * qGrp + rq) * H + hkv * GF + g) * (long)ns + si) * 2;
        Bout[бб]     = (T0 < T1) ? sh->mrow[стр] : -CUDART_INF_F;
        Bout[бб + 1] = (T0 < T1) ? sh->lrow[стр] : 0.f;
    }
}

}  // namespace fa2sm70_dechmma
