// Volta (sm_70) FMHA backward, block-level hand-written mainloop.
//
// Извлечён из tools/volta_block_mainloop.cu, где он доведён до 2.184 мс при d=256/H=8/S=2048
// против 2.954 мс у ядра cutlass (x1.35) — см. docs/SM70_KERNEL_PLAYBOOK.md §14-16. Отличия от
// прототипа, нужные для отгрузки:
//   * раскладка BHSD: смещение головы вычисляется из задания, шаг строки = head_dim;
//   * dK и dV пишутся СРАЗУ в fp16 (при QSPLIT==1 каждую плитку ключей владеет ровно один блок,
//     поэтому обычная запись, без атомарного сложения);
//   * dQ копится атомарно в fp32-области (её обнуляет и конвертирует обёртка) — по ключам он
//     разложен между блоками, и другого способа нет;
//   * все настройки геометрии зафиксированы на измеренном оптимуме: 8 варпов, плитка варпа 32x16,
//     K/V резидентны в smem, дополнение строк +4, векторная перезаливка глобальная->smem.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "gemm/volta_warp_mma.h"

namespace fa2_sm70 {
namespace block_bwd {

using fa2_sm70::WarpMma;
using fa2_sm70::BLayout;
using fa2_sm70::ALayout;

static __device__ int g_ticket_bwd;   // талон очереди работ (внутреннее связывание: -rdc=false)

template <int BK, int BQ, int D, int DC1, int DC2, int WM, int WN, bool CAUSAL, bool KVRES = true, bool SEQ = false, int DC3P = 64, bool SDMA = false, bool PERSIST = false, int QSPLIT = 1, int PAD = 4, bool SWZ = false>
__global__ __launch_bounds__(WM * WN * 32) void volta_bwd_block(const __half* __restrict__ Q,
                                            const __half* __restrict__ Kk,
                                            const __half* __restrict__ Vv,
                                            const __half* __restrict__ dO,
                                            const float* __restrict__ lse,
                                            const float* __restrict__ Delta,
                                            __half* __restrict__ dV,
                                            __half* __restrict__ dK,
                                            float* __restrict__ dQ, int Sq, int Sk,
                                            float qk_scale, int lse_strideH, int nJobs = 0) {
  constexpr int kWarps = WM * WN;   // [ГЕОМЕТРИЯ] раньше было прибито 16; теперь плитка варпа
                                    // TM x TN = BK/WM x BQ/WN перебирается, и с ней загрузок-на-mma
  constexpr int LD1 = DC1 + PAD, LDP = SWZ ? BQ : (BQ + PAD), LD2 = DC2 + PAD;
  constexpr int DC3 = DC3P;                          // кусок по d для dQ
  constexpr int kChunks = D / DC2;
  constexpr int TM = BK / WM, TN = BQ / WN;          // плитка варпа для S/dP: TM x TN
  constexpr int GM = BK / WM, GN = DC2 / WN;         // плитка варпа для dV/dK
  // [ГЕОМЕТРИЯ] сетка варпов фазы dQ ВЫВОДИТСЯ из числа варпов, а не зашита 8x2: при kWarps != 16
  // прежняя запись оставляла половину плитки dQ несчитанной (сверка на S=512 ловит это как ОТКАЗ).
  constexpr int WNq = 2, WMq = kWarps / WNq;
  constexpr int QM = BQ / WMq, QN = DC3 / WNq;      // плитка варпа для dQ
  static_assert(TM % 16 == 0 && TN % 16 == 0, "плитка варпа для S должна быть кратна 16");
  static_assert(GM % 16 == 0 && GN % 16 == 0, "плитка варпа для dV/dK должна быть кратна 16: увеличьте DC2");
  static_assert(QM % 16 == 0 && QN % 16 == 0, "плитка варпа для dQ должна быть кратна 16");
  // Разделяемая память: sP живёт между фазами, поэтому не пересекается; sQ и sdO — пересекаются.
  extern __shared__ __half smem[];
  // KVRES=true: K и V целиком (минимум их трафика, но BK ограничен smem).
  // KVRES=false: K/V текут кусками -> BK можно удвоить, а значит и плитку варпа. Профиль показал,
  // что узкое место — разделяемая память (short_scoreboard 2.37 при плитке 16x16 = ДВЕ загрузки
  // фрагмента на одну mma), а вовсе не глобальный трафик (DRAM 13%). Плитка важнее трафика.
  // В потоковом варианте один и тот же буфер принимает куски ДВУХ разных ширин: DC1 в фазе 1 и
  // DC3 в фазе dQ. Шаг строки берётся по максимуму — иначе запись куска DC3 уезжает за строку.
  constexpr int LDK = KVRES ? (D + PAD) : ((DC1 > DC3 ? DC1 : DC3) + PAD);
  // Арена служит трём разным нуждам, и размер берётся по МАКСИМУМУ. Пропущенный здесь третий
  // случай (fp32-буфер сброса dQ) раньше переполнял арену молча: он затирал буферы, которые всё
  // равно перезаписывались следующей итерацией. Как только K стала резидентной, та же ошибка стала
  // затирать K и дала NaN. Скрытая ошибка проявляется, когда рядом появляется долгоживущий сосед.
  constexpr int kArenaA = 4 * BQ * LD1 > BQ * LD2 ? 4 * BQ * LD1 : BQ * LD2;   // 4 = 2 буфера x (Q,dO)
  constexpr int kArena  = kArenaA > 2 * BQ * DC3 ? kArenaA : 2 * BQ * DC3;
  __half* sP    = smem;                        // P / dS, живёт между фазами
  __half* arena = sP + BK * LDP;               // Q (фаза 1) -> dO/Q (фазы 2/3)
  // [опкоды] lse и Delta читались ИЗ ГЛОБАЛЬНОЙ памяти на каждый элемент аккумулятора внутри
  // визитора: аудит SASS показал 46 из 50 LDG шириной 32 бита, и 16% из них на расстоянии 0-4
  // инструкций от потребителя, то есть с полностью открытой задержкой. Кладём их в smem один раз
  // на блок запросов (2*BQ float = 1 КБ) — вместо десятков разрозненных глобальных чтений.
  __half* sK    = arena + kArena;              // K целиком [BK][LDK]
  __half* sV    = sK + BK * LDK;               // V целиком [BK][LDK]
  // Фаза 1 живёт в арене двойным буфером; фазы 2-4 берут ту же арену целиком (во времени не пересекаются).
  __half* qbuf[2] = {arena, arena + 2 * BQ * LD1};
  __half* obuf[2] = {arena + BQ * LD1, arena + 3 * BQ * LD1};

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;         // сетка варпов WM x WN
  // [PERSIST] Постоянные блоки: сетка кратна ЧИСЛУ SM (80), а работа берётся динамически по талону.
  // Статическая раскладка «блок = плитка ключей x голова» даёт 64 блока при H=1 -> 16 SM из 80
  // простаивают весь запуск, а при причинной маске ещё и перекошена: время задаёт самый тяжёлый блок.
  // Талон на atomicAdd безопасен по построению — никто никого не ждёт (см. SM70_LATENCY_HIDING §10.1).
  __shared__ int s_job;
  int job = 0;
  if constexpr (PERSIST) {
    if (tid == 0) s_job = atomicAdd(&g_ticket_bwd, 1);
    __syncthreads();
    job = s_job;
  }
  const __half *Q0 = Q, *Kk0 = Kk, *Vv0 = Vv, *dO0 = dO;   // базы: каждое задание смещает от них
  const float *lse0 = lse, *Delta0 = Delta;
  __half *dV0 = dV, *dK0 = dK; float* dQ0 = dQ;
  while (true) {
  if constexpr (PERSIST) { if (job >= nJobs) break; }
  // [QSPLIT] Причинность делает задания РАЗНЫМИ по весу: плитка ключей i обрабатывает Sq/BQ - i*BK/BQ
  // блоков запросов. При Sk=2048, BK=32, BQ=128 это 16..1, всего 544 единицы, а время задаёт самая
  // тяжёлая = 16. Значит машина занята на 544/(16*80) = 42.5%, и никакая раздача 64 крупных заданий
  // этого не исправит. Дробим по ЗАПРОСАМ: срез s берёт блоки запросов с шагом QSPLIT, вес задания
  // падает до ~16/QSPLIT, и жадная раздача по талону выходит на предел 544/80 = 6.8.
  // Цена — dK/dV теперь копятся атомарно (срезы одной плитки ключей складываются).
  const int slice = PERSIST ? (job % QSPLIT) : 0;
  const int kb = (PERSIST ? ((job / QSPLIT) % (Sk / BK)) : blockIdx.x) * BK;   // плитка ключей
  // Голова = blockIdx.y. При причинности вся экономия — в ЧИСЛЕ волн: 64 блока на 80 SM идут
  // одновременно, и время задаёт самый тяжёлый (kb=0). Реальный B*H даёт много волн, и пропуск
  // блоков запросов превращается в реальную экономию.
  {
    const long hoff = PERSIST ? (long)(job / (QSPLIT * (Sk / BK))) : (long)blockIdx.y;
    Q += hoff * (long)Sq * D;  Kk += hoff * (long)Sk * D;  Vv += hoff * (long)Sk * D;
    dO += hoff * (long)Sq * D;
    lse += hoff * (long)lse_strideH;      // у обёртки шаг lse дополнен до кратности 8
    Delta += hoff * (long)Sq;             // delta плотная [B,H,Sq]
    dV += hoff * (long)Sk * D; dK += hoff * (long)Sk * D;  dQ += hoff * (long)Sq * D;
  }

  // K и V читаются из глобальной памяти ОДИН раз на блок, а не на каждый кусок по d и не на каждый
  // блок запросов: для Sq/BQ проходов это во столько же раз меньше их трафика, и главное — они
  // уходят из всех внутренних циклов вместе со своими барьерами.
  if constexpr (KVRES)
    for (int i = tid; i < BK * D / 8; i += kWarps * 32) {      // 16 Б за доступ вместо 2
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      const uint4 gk_ = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + r) * D + c]);
      const uint4 gv_ = *reinterpret_cast<const uint4*>(&Vv[(long)(kb + r) * D + c]);
      __half* pk = &sK[r * LDK + c]; __half* pv = &sV[r * LDK + c];
      if constexpr (PAD % 8 == 0) { *reinterpret_cast<uint4*>(pk) = gk_; *reinterpret_cast<uint4*>(pv) = gv_; }
      else { *reinterpret_cast<uint2*>(pk)     = make_uint2(gk_.x, gk_.y);
             *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk_.z, gk_.w);
             *reinterpret_cast<uint2*>(pv)     = make_uint2(gv_.x, gv_.y);
             *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv_.z, gv_.w); }
    }

  // dV И dK в РЕГИСТРАХ на всю развёртку по блокам запросов — это и есть цель.
  WarpMma<GM / 16, GN / 16, BLayout::kNMajor> gv[kChunks], gk[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) { gv[c].clear(); gk[c].clear(); }

  // Причинность: ключ k влияет на запрос q только при k <= q. Блоки запросов ЦЕЛИКОМ левее
  // диагонали пропускаются — это и есть та половина работы, которую снимает маска; поэлементная
  // маска нужна лишь на диагональном блоке.
  const int qStart = (CAUSAL ? (kb / BQ) * BQ : 0) + slice * BQ;
  for (int q0 = qStart; q0 < Sq; q0 += BQ * QSPLIT) {

  // [АДРЕСАЦИЯ, шаг 2] Перезаливка глобальная->smem шла ПО ОДНОМУ half: BQ*DC2 = 16384 элемента на
  // 512 потоков = 32 прохода, и в каждом полный пересчёт адреса (два IMAD + деление + остаток).
  // Замер состава инструкций дал integer 2.54x и memory 3.96x против cutlass — вот их источник.
  // Читаем из глобальной по 16 Б (строка D=256 half выровнена, смещения кратны 8), кладём в smem
  // двумя по 8 Б: шаг строки LD2 = DC2+4 half не кратен 16 Б, а бесконфликтность важнее ширины.
  auto fill_arena = [&](const __half* __restrict__ src, int chunk) {
    constexpr int kRowVec = DC2 / 8;                   // векторов в строке куска
    constexpr int kIters  = BQ * kRowVec / (WM * WN * 32);
    static_assert(BQ * kRowVec % (WM * WN * 32) == 0, "перезаливка арены не делится на блок");
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int v = tid + it * (WM * WN * 32), r = v / kRowVec, cc = (v % kRowVec) * 8;
      const uint4 g = *reinterpret_cast<const uint4*>(&src[(long)(q0 + r) * D + chunk * DC2 + cc]);
      __half* d = &arena[r * LD2 + cc];
      if constexpr (PAD % 8 == 0) *reinterpret_cast<uint4*>(d) = g;   // шаг кратен 16 Б
      else { *reinterpret_cast<uint2*>(d)     = make_uint2(g.x, g.y);
             *reinterpret_cast<uint2*>(d + 4) = make_uint2(g.z, g.w); }
    }
  };
    // ---- фаза 1: S = K * Q^T, редукция по d кусками DC1
    // SEQ: S и dP^T НЕ живут одновременно. P к моменту dP^T уже лежит в smem, поэтому регистры
    // S можно освободить и посчитать в них dP^T. Пик падает на 32 float — ровно столько нужно,
    // чтобы при BK=BQ=DC2=128 влезли резидентные dV+dK и плитка стала 32x32 во ВСЕХ gemm.
    WarpMma<TM / 16, TN / 16, BLayout::kKMajor> qk, dpt;
    qk.clear();
    if constexpr (!SEQ) dpt.clear();
    // Двойная буферизация глобального этапа: загрузка куска dc+DC1 ВЫДАЁТСЯ до счёта куска dc, и её
    // задержку покрывают mma текущего куска. Без этого цикл строго последователен:
    // барьер -> загрузка -> барьер -> счёт, и задержка глобальной памяти ничем не закрыта.
    constexpr int kPerThr = BQ * DC1 / (kWarps * 32);
    static_assert(kPerThr % 8 == 0, "нужен целый 16-байтовый доступ на поток");
    constexpr int kVec = kPerThr / 8, kChkRow = DC1 / 8;
    uint4 stq[kVec], sto[kVec];
    // [softDMA] Выдача ОДНОЙ порции: слот idx. Порции Q идут первыми, затем dO.
    auto issue_one = [&](int dcx, int idx) {
      if (idx < kVec) {
        const int v = idx, i2 = tid + v * kWarps * 32, r = i2 / kChkRow, c = (i2 % kChkRow) * 8;
        stq[v] = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * D + dcx + c]);
      } else {
        const int v = idx - kVec, i2 = tid + v * kWarps * 32, r = i2 / kChkRow, c = (i2 % kChkRow) * 8;
        sto[v] = *reinterpret_cast<const uint4*>(&dO[(long)(q0 + r) * D + dcx + c]);
      }
    };
    auto ldg_chunk = [&](int dcx) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        stq[v] = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * D + dcx + c]);
        sto[v] = *reinterpret_cast<const uint4*>(&dO[(long)(q0 + r) * D + dcx + c]);
      }
    };
    auto ldg_chunk_o = [&](int dcx) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        sto[v] = *reinterpret_cast<const uint4*>(&dO[(long)(q0 + r) * D + dcx + c]);
      }
    };
    auto sts_chunk_o = [&](int b) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        __half* po = &obuf[b][r * LD1 + c];
        if constexpr (PAD % 8 == 0) *reinterpret_cast<uint4*>(po) = sto[v];
        else { *reinterpret_cast<uint2*>(po)     = make_uint2(sto[v].x, sto[v].y);
               *reinterpret_cast<uint2*>(po + 4) = make_uint2(sto[v].z, sto[v].w); }
      }
    };
    auto sts_chunk = [&](int b) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        // Шаг строки LD1 = DC1+4 половинок = 72 Б: строки ложатся на РАЗНЫЕ банки (бесконфликтно),
        // но 16-байтовая запись для нечётных строк невыровнена. Из глобальной читаем по 16 Б,
        // в smem кладём двумя 8-байтовыми. Дополнение и широкая запись несовместимы — выбираем
        // бесконфликтность, она в этом цикле важнее.
        __half* pq = &qbuf[b][r * LD1 + c];
        __half* po = &obuf[b][r * LD1 + c];
        if constexpr (PAD % 8 == 0) { *reinterpret_cast<uint4*>(pq) = stq[v]; *reinterpret_cast<uint4*>(po) = sto[v]; }
        else { *reinterpret_cast<uint2*>(pq)     = make_uint2(stq[v].x, stq[v].y);
               *reinterpret_cast<uint2*>(pq + 4) = make_uint2(stq[v].z, stq[v].w);
               *reinterpret_cast<uint2*>(po)     = make_uint2(sto[v].x, sto[v].y);
               *reinterpret_cast<uint2*>(po + 4) = make_uint2(sto[v].z, sto[v].w); }
      }
    };
    __syncthreads();
    ldg_chunk(0); sts_chunk(0);
    __syncthreads();
    int cur = 0;
    for (int dc = 0; dc < D; dc += DC1) {
      if (dc + DC1 < D) {
        ldg_chunk(dc + DC1);                          // выдаётся заранее
        // БАРЬЕР ПЛАНИРОВЩИКА. Аудит SASS: ptxas УТОПИЛ эти LDG к самому потребителю — медиана
        // расстояния «загрузка -> первое использование» 5-16 инструкций против 58 у cutlass, и 16%
        // загрузок стоят на расстоянии 0-4, то есть их задержка не спрятана ничем. Планировщик
        // минимизирует время жизни регистров и потому сдвигает загрузку вниз. Пустой asm с
        // клоббером памяти запрещает переставлять обращения к памяти через эту точку: LDG остаётся
        // сверху, а весь блок mma+LDS оказывается между ним и потребляющей STS.
        // asm volatile("" ::: "memory");   // A/B: барьер планировщика
      }
      if constexpr (!KVRES) {
        __syncthreads();
        for (int i = tid; i < BK * DC1; i += kWarps * 32) {
          const int r = i / DC1, c = i % DC1;
          sK[r * LDK + c] = Kk[(long)(kb + r) * D + dc + c];
          sV[r * LDK + c] = Vv[(long)(kb + r) * D + dc + c];
        }
        __syncthreads();
      }
      const int koff = KVRES ? dc : 0;
      qk.template accumulate<DC1>(sK + koff, LDK, wm * TM, qbuf[cur], LD1, wn * TN, lane);
      if constexpr (!SEQ)
        dpt.template accumulate<DC1>(sV + koff, LDK, wm * TM, obuf[cur], LD1, wn * TN, lane);
      if (dc + DC1 < D) {
        __syncthreads();
        sts_chunk(cur ^ 1);
        __syncthreads();
        cur ^= 1;
      }
    }
    // ---- P = exp(S*scale - lse[запрос]) и выгрузка в fp16 для следующего gemm
    const float scale = qk_scale;   // масштаб задаёт обёртка, а не 1/sqrt(d)
    const int r0 = wm * TM, c0 = wn * TN;
    // [опкоды] lse и Delta читались из ГЛОБАЛЬНОЙ памяти на КАЖДЫЙ элемент аккумулятора: аудит SASS
    // насчитал 46 из 50 LDG шириной 32 бита, и 16% из них на расстоянии 0-4 от потребителя.
    // Поток трогает всего ЧЕТЫРЕ различных столбца на 16-широкий блок, поэтому поднимаем их в
    // регистры один раз. Номер слота выводится из столбца: colbase(lane) = (l&2)|((l&4)<<1) имеет
    // нулевые биты 0 и 2, значит c&1 и (c>>2)&1 — это в точности биты номера регистра.
    // NB перенос этих же чтений в SMEM был измерен и оказался ХУЖЕ на 5.5%: ядро упирается в
    // разделяемую память, и переносить туда работу нельзя. Регистры — единственный верный адрес.
    // [ГЕОМЕТРИЯ] Кэш был рассчитан ровно на ЧЕТЫРЕ столбца, то есть на ОДИН столбцовый блок
    // (TN=16). При TN=32 и шире visit выдаёт c = j*16 + wmma_acc_col(lane,r), и номер блока j=c>>4
    // в индексе отсутствовал -> второй блок брал чужие lse/Delta, портилась P, а с ней ВСЕ ТРИ
    // выхода одинаково (почерк: dV/dK/dQ = 3e-01 при 8 варпах). Слотов теперь 4 на каждый блок.
    constexpr int kNB = TN / 16;                       // столбцовых блоков в плитке варпа
    const int cb = (lane & 2) | ((lane & 4) << 1);
    float lseq[4 * kNB], delq[4 * kNB];
#pragma unroll
    for (int b = 0; b < kNB; ++b)
#pragma unroll
      for (int sI = 0; sI < 4; ++sI) {
        const int cc = c0 + b * 16 + cb + (sI & 1) + 4 * (sI >> 1);
        lseq[b * 4 + sI] = lse[q0 + cc];
        delq[b * 4 + sI] = Delta[q0 + cc];
      }
#define FA2_SLOT(c) (((c) >> 4) * 4 + ((c) & 1) + 2 * (((c) >> 2) & 1))
    qk.visit(lane, [&](float& v, int r, int c) {
      const float p = __expf(v * scale - lseq[FA2_SLOT(c)]);
      v = (CAUSAL && (kb + r0 + r) > (q0 + c0 + c)) ? 0.f : p;
    });
    if constexpr (SWZ) qk.store_half_swz(sP, LDP, r0, c0, lane);
    else qk.store_half(sP, LDP, r0, c0, lane);

    // ---- фаза 2: dV += P^T * dO, редукция по запросам (BQ), кусками по d
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      __syncthreads();
      fill_arena(dO, c);
      __syncthreads();
      if constexpr (SWZ) gv[c].template accumulate_swzA<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
      else gv[c].template accumulate<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
    }

    // ---- SEQ: только теперь считаем dP^T, в регистрах, освободившихся от S
    if constexpr (SEQ) {
      dpt.clear();
      __syncthreads();
      ldg_chunk(0); sts_chunk(0);
      __syncthreads();
      int c2 = 0;
      for (int dc = 0; dc < D; dc += DC1) {
        if (dc + DC1 < D) ldg_chunk_o(dc + DC1);
        // При потоковом K/V кусок V нужен и здесь: фаза 1 оставила в sV только ПОСЛЕДНИЙ кусок.
        if constexpr (!KVRES) {
          __syncthreads();
          for (int i = tid; i < BK * DC1; i += kWarps * 32) {
            const int r = i / DC1, c = i % DC1;
            sV[r * LDK + c] = Vv[(long)(kb + r) * D + dc + c];
          }
          __syncthreads();
        }
        dpt.template accumulate<DC1>(sV + (KVRES ? dc : 0), LDK, wm * TM, obuf[c2], LD1, wn * TN, lane);
        if (dc + DC1 < D) { __syncthreads(); sts_chunk_o(c2 ^ 1); __syncthreads(); c2 ^= 1; }
      }
    }
    // ---- dS = P * (dP^T - Delta[запрос])
    __syncthreads();
    if constexpr (SEQ) {
      // P взята из smem: в регистрах её больше нет
      dpt.visit(lane, [&](float& v, int r, int c) {
        const int rr_ = r0 + r;
        const int cc_ = SWZ ? decltype(dpt)::swz(c0 + c, rr_ & 15) : (c0 + c);
        v = scale * __half2float(sP[(long)rr_ * LDP + cc_]) * (v - delq[FA2_SLOT(c)]);
      });
      __syncthreads();
      dpt.store_half(sP, LDP, r0, c0, lane);
    } else {
      const float* dp = &dpt.acc[0][0][0];
      int idx = 0;
      // dS = scale * P * (dP^T - Delta). В стенде множитель `scale` сокращался с обеих сторон
      // (эталон считался по той же формуле), а настоящему обратному проходу он НУЖЕН: без него dK и
      // dQ выходят ровно в 1/scale раз больше (при d=256 это 16), а dV — верным, потому что он
      // берёт P, а не dS. Подпись ошибки именно такая и была: dV 4e-05, dK/dQ 1.8e+01.
      qk.visit(lane, [&](float& p, int r, int c) { p = scale * p * (dp[idx++] - delq[FA2_SLOT(c)]); });
      // dS ложится в sP теми же правилами
#undef FA2_SLOT
      if constexpr (SWZ) qk.store_half_swz(sP, LDP, r0, c0, lane);
      else qk.store_half(sP, LDP, r0, c0, lane);
    }
    // ---- dK += dS * Q  (.row.row: Q[запрос][d] читается как есть)
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      __syncthreads();
      fill_arena(Q, c);
      __syncthreads();
      if constexpr (SWZ) gk[c].template accumulate_swzA<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
      else gk[c].template accumulate<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
    }

    // ---- dQ[запрос][d] += dS^T * K  (.col.row): dS читается КАК ЕСТЬ как [k=ключ][m=запрос],
    // K — как [k=ключ][n=d]. Ни одной транспозы. Накопитель временный (8 float на поток) и
    // сбрасывается атомарно: dQ принадлежит блоку ЗАПРОСОВ, а блок владеет ключами.
    {
      const int wmq = warp / WNq, wnq = warp % WNq;
#pragma unroll
      for (int c = 0; c < D; c += DC3) {
        WarpMma<QM / 16, QN / 16, BLayout::kNMajor, ALayout::kMMajor> gq;
        gq.clear();
        if constexpr (!KVRES) {
          __syncthreads();
          for (int i = tid; i < BK * DC3; i += kWarps * 32)
            sK[(i / DC3) * LDK + i % DC3] = Kk[(long)(kb + i / DC3) * D + c + i % DC3];
          __syncthreads();
        }
        if constexpr (SWZ) gq.template accumulate_swzA_mmajor<BK>(sP, LDP, wmq * QM, sK + (KVRES ? c : 0), LDK, wnq * QN, lane);
        else gq.template accumulate<BK>(sP, LDP, wmq * QM, sK + (KVRES ? c : 0), LDK, wnq * QN, lane);
        // Раскладка аккумулятора разрозненна по (строка, столбец), поэтому прямой atomicAdd на
        // элемент бьёт по многим гранулам. Складываем плитку в smem (там разрозненность дёшева) и
        // сбрасываем ОДНИМ коалесцированным проходом.
        // ИЗМЕРЕНО при DC3=64: с промежуточным хранением 1.190 мс, без него 1.436 -> коалесценция
        // стоит 17%. NB абляция «убрать саму atomicAdd» показывает лишь 2% — и это НЕ значит, что
        // атомарность бесплатна: при уже коалесцированном доступе она действительно дёшева.
        // Сравнивать надо разрозненную атомарность с коалесцированной, а не с её отсутствием.
        float* stage = reinterpret_cast<float*>(arena);
        __syncthreads();
        gq.visit(lane, [&](float v, int r, int cc) {
          stage[(wmq * QM + r) * DC3 + wnq * QN + cc] = v;
        });
        __syncthreads();
        for (int i = tid; i < BQ * DC3; i += kWarps * 32)
          atomicAdd(&dQ[(long)(q0 + i / DC3) * D + c + i % DC3], stage[i]);
      }
    }
  }

  // ---- эпилог: один проход записи вместо круговых рейсов на каждый блок запросов
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    gv[c].visit(lane, [&](float v, int r, int cc) {
      const long a = (long)(kb + wm * GM + r) * D + c * DC2 + wn * GN + cc;
      dV[a] = __float2half(v);            // QSPLIT==1: плиткой ключей владеет один блок
    });
    gk[c].visit(lane, [&](float v, int r, int cc) {
      const long a = (long)(kb + wm * GM + r) * D + c * DC2 + wn * GN + cc;
      dK[a] = __float2half(v);
    });
  }
  if constexpr (!PERSIST) break;
  // следующее задание по талону; указатели — от баз, накопители обнулить
  Q = Q0; Kk = Kk0; Vv = Vv0; dO = dO0; lse = lse0; Delta = Delta0; dV = dV0; dK = dK0; dQ = dQ0;
  __syncthreads();                       // smem переиспользуется — дождаться конца текущего задания
  if (tid == 0) s_job = atomicAdd(&g_ticket_bwd, 1);
  __syncthreads();
  job = s_job;
  }
}

}  // namespace block_bwd
}  // namespace fa2_sm70

// ---------------------------------------------------------------------------------------------
// Запуск рукописного блочного мейнлупа. Условия применимости узкие НАМЕРЕННО: ядро писалось под
// один режим и в нём измеренно быстрее cutlass в 1.35 раза, а на всё остальное есть штатный путь.
namespace fa2_sm70 { namespace block_bwd {

template <int D, bool CAUSAL>
static bool launch(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
                   const at::Tensor& dO, const at::Tensor& LSE, const at::Tensor& delta,
                   at::Tensor& dQ, at::Tensor& dK, at::Tensor& dV,
                   int B, int H, int Sq, int Sk, double scale, at::Tensor& dqf) {
  constexpr int BK = 32, BQ = 128, DC1 = 32, DC2 = 128, WM = 1, WN = 8, DC3P = 64;
  constexpr int kWarps = WM * WN;
  // smem: sP + арена + резидентные K/V — та же формула, что в стенде
  constexpr int LD1 = DC1 + 4, LDP = BQ + 4, LD2 = DC2 + 4, LDK = D + 4;
  constexpr int kArenaA = 4 * BQ * LD1 > BQ * LD2 ? 4 * BQ * LD1 : BQ * LD2;
  constexpr int kArenaB = kArenaA > 2 * BQ * DC3P ? kArenaA : 2 * BQ * DC3P;
  constexpr size_t smem = (size_t)(BK * LDP + kArenaB + 2 * BK * LDK) * sizeof(__half);
  if (smem > 96 * 1024) return false;

  // Обёртка на BHSD-пути даёт плотные тензоры (p.q_strideH = Sq*d и т.д.), но проверка дешёвая.
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous() && dO.is_contiguous(),
              "блочный мейнлуп ожидает плотные BHSD входы");
  const at::Tensor &Qc = Q, &Kc = K, &Vc = V, &dOc = dO, &Lc = LSE, &Dc = delta;
  const int nJobs = (Sk / BK) * B * H;                 // QSPLIT=1: плиткой ключей владеет один блок
  int nCTA  = nJobs < 80 ? nJobs : 80;                  // постоянные блоки: сетка = число SM
  static const char* e_cta = getenv("FA2SM70_BLOCK_CTA");   // читаем один раз: см. §18 плейбука
  if (e_cta) { int v = atoi(e_cta); if (v > 0) nCTA = v < nJobs ? v : nJobs; }
  auto kern = volta_bwd_block<BK, BQ, D, DC1, DC2, WM, WN, CAUSAL, /*KVRES*/true, /*SEQ*/false,
                              DC3P, /*SDMA*/false, /*PERSIST*/true, /*QSPLIT*/1, /*PAD*/4, /*SWZ*/false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  const int zero = 0;
  C10_CUDA_CHECK(cudaMemcpyToSymbol(g_ticket_bwd, &zero, sizeof(int), 0, cudaMemcpyHostToDevice));
  dqf.zero_();                                          // dQ копится атомарно -> старт с нуля
  kern<<<nCTA, kWarps * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(Qc.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(Kc.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(Vc.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(dOc.data_ptr<at::Half>()),
      Lc.data_ptr<float>(), Dc.data_ptr<float>(),
      reinterpret_cast<__half*>(dV.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(dK.data_ptr<at::Half>()),
      dqf.data_ptr<float>(), Sq, Sk, (float)scale, (int)Lc.size(2), nJobs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dQ.copy_(dqf.to(torch::kFloat16));                    // fp32-область -> fp16 выход
  return true;
}

// Годность: причинная маска, выровненные размеры, без GQA, d из набора, под который писалось ядро.
static inline bool eligible(int B, int H, int Hkv, int Sq, int Sk, int d, bool causal, bool bhsd) {
  if (!causal || !bhsd || H != Hkv) return false;
  if (Sq != Sk) return false;
  if (Sq % 128 != 0 || Sk % 32 != 0) return false;
  if (d != 256 && d != 512) return false;              // d<=128 штатный путь уже быстрее
  static const char* e_on = getenv("FA2SM70_BLOCK_BWD");    // читаем один раз
  if (e_on) return atoi(e_on) != 0;
  return true;
}

}}  // namespace
