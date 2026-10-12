// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// МЕХАНИЗМ АДАПТИВНОЙ РАЗРЕЖЕННОСТИ для рукописного forward d=256 (volta_fwd_block.cuh).
//
// ЧТО ЗДЕСЬ. Три вещи, и они СПЕЦИАЛЬНО разнесены, чтобы каждую можно было замерить отдельно:
//   1. подача маски       -- ядро внимания берёт номера оставленных плиток ключей из подачи
//                            (SPARSE=1 список / SPARSE=2 битовая карта в volta_fwd_block.cuh);
//   2. сводки ключей      -- kv_block_summary: средний ключ плитки + максимум отклонения от него;
//   3. отбор плиток       -- select_blocks: Q x сводки НА ТЕНЗОРНЫХ ЯДРАХ тем же оператором
//                            (volta_warp_mma.h), затем точный top-k двухуровневой гистограммой.
//
// ПОЧЕМУ ОТДЕЛЬНАЯ ЕДИНИЦА ТРАНСЛЯЦИИ. attn_fwd_cutlass.cu тянет cutlass и собирается минутами;
// здесь только наш оператор, сборка секунды. Плюс сюда же положен ЗАМОРОЖЕННЫЙ отгруженный кернел
// (volta_fwd_block_ref.cuh, namespace block_fwd_ref) -- чтобы A/B «механизм против его отсутствия»
// шёл в ОДНОМ бинарнике, на одной обвязке, без пересборки между двумя половинами замера.
//
// ЧТО ЗАМЕР СКАЗАЛ ПРО САМ МЕХАНИЗМ (benchmarks/bench_sparse_mech.py, V100, чередующийся протокол):
//   * ПОБИТОВОЕ равенство при плотности 100% -- есть, все три пути подачи, causal и нет, 87 проверок;
//   * подача СПИСКОМ стоит +6.6..7.3% ВСЕГДА, битовой картой +0.4..1.9%, списком С ЧТЕНИЕМ ВПЕРЁД
//     +0.5%. По инструкциям разница ничтожна (SASS: +1..4 LDG на всё ядро) -- платится ЗАВИСИМОСТЬ,
//     см. комментарий у цикла в volta_fwd_block.cuh;
//   * сводки 0.01-0.02% времени внимания, отбор 1.52-1.57%;
//   * НЕРАВНОМЕРНОСТЬ НАГРУЗКИ от поголовной плотности оказалась НЕ проблемой (макс/среднее веса
//     задания = 1.11-1.15, модель волн 96.7-98.7%), и очередь заданий её не лечит, потому что лечить
//     нечего: любой порядок в пределах +0.5..3.5% от сетки. Это ОПРОВЕРГАЕТ ожидание из задания,
//     построенное на причинной маске (там макс/среднее было ~2).
//   * ГЛАВНОЕ, чего в задании не было: выигрыш определяется не числом выброшенных плиток, а тем,
//     СОГЛАСОВАНЫ ли маски одновременно идущих блоков запросов (см. benchmarks/diag_sparse_share.py).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
#include "volta_fwd_block.cuh"
#include "volta_fwd_block_ref.cuh"
#include "gemm/volta_warp_mma.h"

namespace fa2_sm70 {
namespace sparse {

using fa2_sm70::WarpMma;
using fa2_sm70::BLayout;
using fa2_sm70::ALayout;

// ==============================================================================================
// 1. СВОДКИ ПЛИТОК КЛЮЧЕЙ.
//   kbar[j][:]  -- средний ключ плитки j (это и есть «сводка» из задания);
//   rmax[j]     -- максимум по плитке от ||k - kbar||, то есть РАДИУС плитки.
// Радиус ЗАДУМЫВАЛСЯ как честная верхняя граница максимума в плитке (Коши-Буняковский:
// q.k_i = q.kbar + q.(k_i - kbar) <= q.kbar + ||q||*rmax).
// ЗАМЕР ОПРОВЕРГ ЭТО (tools/sparse_selector_vs_oracle.py, настоящий дамп Q/K/V модели, 4 слоя x все
// головы): поправка радиусом СТАБИЛЬНО ХУЖЕ чистого среднего (например слой 43, плотность 0.75:
// 3.21e-2 против 3.02e-2), а ТОЧНАЯ покомпонентная граница sum_d[q+ * kmax + q- * kmin] хуже в 2-4
// раза. Причина понятна задним числом и её стоит запомнить: отбирать надо по МАССЕ плитки, а массу
// задают ТИПИЧНЫЕ ключи. Любая верхняя граница по максимуму продвигает плитки с одним выбросом,
// в которых массы нет. rmax оставлен (стоит один скаляр на плитку) и по умолчанию НЕ ИСПОЛЬЗУЕТСЯ:
// alpha = 0. Это след замера, а не задел.
// Сетка: (nkb, Hkv, B), 256 нитей -- по нити на измерение d (d=256), чтение строки плитки
// коалесцировано (512 Б на строку).
// ЦЕНА: один проход по K. В чанковом префилле сводки НОВЫХ ключей считаются один раз и дописываются
// в конец, поэтому цена пропорциональна ЧАНКУ, а не контексту -- замерено 0.01-0.02% времени внимания.
// ==============================================================================================
// РАСКЛАДКА НИТЕЙ РАЗНАЯ НА ДВУХ ПРОХОДАХ, И ЭТО НЕ ВКУСОВЩИНА.
// Проход 1 (средний ключ): нить на измерение d -- редукция идёт по СТРОКАМ, между нитями обмена нет.
// Проход 2 (радиус): нужна сумма по d для КАЖДОЙ строки, то есть 64 независимые редукции. Если
// оставить раскладку прохода 1, каждая из них станет блочной (обмен + два барьера) -- 64 барьера на
// плитку. Поэтому строка отдаётся ВАРПУ: редукция по d становится варповой (5 обменов, ноль барьеров),
// а 8 варпов берут по 8 строк.
template <int BK, int D>
__global__ __launch_bounds__(D, 4) void kv_block_summary(
    const __half* __restrict__ K, __half* __restrict__ kbar, float* __restrict__ rmax,
    int Sk, int nkb, long k_sM, long k_sH, long k_sB) {
  __shared__ float sMean[D];
  __shared__ float sWmax[D / 32];
  const int j = blockIdx.x, hkv = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int kWarps = D / 32;
  const long base = (long)b * k_sB + (long)hkv * k_sH + (long)(j * BK) * k_sM;
  const int n = (Sk - j * BK < BK) ? (Sk - j * BK) : BK;        // рваный последний блок
  float acc = 0.f;
  for (int r = 0; r < n; ++r) acc += __half2float(K[base + (long)r * k_sM + tid]);
  const float mean = acc / (float)n;
  const long ob = ((long)b * gridDim.y + hkv) * nkb + j;
  kbar[ob * D + tid] = __float2half(mean);
  sMean[tid] = mean;
  __syncthreads();
  float best = 0.f;
  for (int r = warp; r < n; r += kWarps) {                      // строка -> варп
    float s = 0.f;
    for (int c = lane; c < D; c += 32) {
      const float dv = __half2float(K[base + (long)r * k_sM + c]) - sMean[c];
      s += dv * dv;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (s > best) best = s;
  }
  if (lane == 0) sWmax[warp] = best;
  __syncthreads();
  if (tid == 0) {
    float t = 0.f;
    for (int w = 0; w < kWarps; ++w) t = t > sWmax[w] ? t : sWmax[w];
    rmax[ob] = sqrtf(t);
  }
}

// ==============================================================================================
// 2. ОТБОР ПЛИТОК. Один блок нитей на (блок запросов, голова, батч).
//
// ШАГ A -- ОЦЕНКИ. score[j] = max по запросам блока от (q_i . kbar_j * scale + alpha*||q_i||*rmax_j).
// Это ОБЫЧНЫЙ GEMM [BQ x D] x [D x nkb], и он считается ТЕМ ЖЕ оператором WarpMma, что мейнлуп --
// никакого бокового конвейера (на Volta выдача одна, §4i). Стоимость: ровно 1/(2*BK) от плотного
// внимания, то есть 0.78% при BK=64. Максимум по запросам, а не среднее: софтмакс правит максимум,
// и среднее по 64 запросам свернуло бы GEMM в GEMV -- дешевле, но слепее.
// Q лежит в разделяемой ЦЕЛИКОМ (33 КБ) и читается из памяти ОДИН раз; по d кусками идут сводки,
// а они целиком (2 МБ на 250K) живут в L2 и переиспользуются всеми блоками запросов.
//
// ШАГ B -- ТОЧНЫЙ top-k. Двухуровневая гистограмма по 256 корзин на сортируемом виде float
// (u = знак ? ~b : b|0x80000000), затем добор границы. Ровно target плиток, ни больше ни меньше:
// «подгонять порог под результат» запрещено, значит и плотность обязана быть той, что заявлена.
// ШАГ C -- КОМПАКТИРОВАНИЕ В ПОРЯДКЕ ВОЗРАСТАНИЯ. Список обязан быть отсортирован: (а) при
// плотности 100% он тогда совпадает с плотным обходом и результат ПОБИТОВО тот же; (б) возрастающий
// обход ключей сохраняет локальность L2, ради которой в мейнлупе стоит развёрнутый порядок блоков.
// ==============================================================================================
constexpr int kMaxNkb = 4096;              // Sk <= 262144 при BK=64
constexpr int kBmWords = kMaxNkb / 32;

__device__ __forceinline__ uint32_t sortable(float x) {
  const uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Исключительный префиксный скан popcount по 128 словам битовой карты. Делает ОДИН варп подряд:
// 128 сложений против GEMM на десятки миллионов -- искать здесь параллелизм незачем.
__device__ __forceinline__ void scan_bm(const uint32_t* bm, int* off, int nw, int tid) {
  if (tid == 0) {
    int run = 0;
    for (int w = 0; w < nw; ++w) { off[w] = run; run += __popc(bm[w]); }
    off[nw] = run;
  }
  __syncthreads();
}

template <int BQ, int BK, int D, int NKT, int WM, int WN, int DC>
__global__ __launch_bounds__(WM * WN * 32, 1) void select_blocks(
    const __half* __restrict__ Q, const __half* __restrict__ kbar, const float* __restrict__ rmax,
    const float* __restrict__ dens,            // плотность на (батч, голова) -- ПОСЛОЙНО/ПОГОЛОВНО
    int32_t* __restrict__ kmask, int32_t* __restrict__ kcnt,
    int Sq, int Sk, int nkb, int mask_ld, int nqb, int nheads, int kv_ratio,
    float qk_scale, float alpha, int causal_off, int fmt, int red,
    long q_sM, long q_sH, long q_sB) {
  constexpr int kWarps = WM * WN, PAD = 4, SM = BQ / WM, SN = NKT / WN;
  constexpr int MB = SM / 16, NB = SN / 16;
  static_assert(SM % 16 == 0 && SN % 16 == 0, "плитки кратны 16");
  extern __shared__ __half smem[];
  __half* sQ = smem;                                       // [BQ][D+PAD] резидентно
  __half* sKb = sQ + BQ * (D + PAD);                       // [NKT][DC+PAD] кусками по d
  float* sc = reinterpret_cast<float*>(sKb + NKT * (DC + PAD));   // [nkb] оценки
  float* sRed = sc + kMaxNkb;                              // [WM][NKT]
  float* sQn = sRed + WM * NKT;                            // [BQ] нормы запросов
  float* sRm = sQn + BQ;                                   // [NKT] радиусы плитки
  uint32_t* bmA = reinterpret_cast<uint32_t*>(sRm + NKT);   // [kBmWords] заведомо оставленные
  uint32_t* bmB = bmA + kBmWords;                           // [kBmWords] граничные
  int* sOff = reinterpret_cast<int*>(bmB + kBmWords);        // [kBmWords+1] префиксы
  __shared__ int sHist[256];
  __shared__ int sB1, sB2, sAbove, sNeed;

  const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z, q0 = qb * BQ;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int hkv = h / kv_ratio;
  const int kEnd = (q0 + BQ + causal_off < Sk) ? (q0 + BQ + causal_off) : Sk;
  const int nkbC = (kEnd + BK - 1) / BK;                    // причинный предел этого блока запросов
  const long row = ((long)b * nheads + h) * nqb + qb;
  const __half* Qp = Q + (long)b * q_sB + (long)h * q_sH;
  const __half* Kb = kbar + (((long)b * (nheads / kv_ratio) + hkv) * nkb) * D;
  const float* Rm = rmax + ((long)b * (nheads / kv_ratio) + hkv) * nkb;

  // ---- Q целиком в разделяемую + нормы строк
  for (int i = tid; i < BQ * D / 8; i += kWarps * 32) {
    const int r = i / (D / 8), c = (i % (D / 8)) * 8;
    uint4 g = make_uint4(0, 0, 0, 0);
    if (q0 + r < Sq) g = *reinterpret_cast<const uint4*>(&Qp[(long)(q0 + r) * q_sM + c]);
    __half* d = &sQ[r * (D + PAD) + c];
    *reinterpret_cast<uint2*>(d) = make_uint2(g.x, g.y);
    *reinterpret_cast<uint2*>(d + 4) = make_uint2(g.z, g.w);
  }
  __syncthreads();
  for (int r = tid; r < BQ; r += kWarps * 32) {
    float s = 0.f;
    for (int c = 0; c < D; ++c) { const float x = __half2float(sQ[r * (D + PAD) + c]); s += x * x; }
    sQn[r] = sqrtf(s) * qk_scale;
  }

  // ---- ШАГ A: оценки плитками по NKT сводок
  for (int t0 = 0; t0 < nkbC; t0 += NKT) {
    WarpMma<MB, NB, BLayout::kKMajor> g; g.clear();
    for (int dc = 0; dc < D; dc += DC) {
      __syncthreads();
      for (int i = tid; i < NKT * DC / 8; i += kWarps * 32) {
        const int r = i / (DC / 8), c = (i % (DC / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t0 + r < nkbC) v = *reinterpret_cast<const uint4*>(&Kb[(long)(t0 + r) * D + dc + c]);
        __half* p = &sKb[r * (DC + PAD) + c];
        *reinterpret_cast<uint2*>(p) = make_uint2(v.x, v.y);
        *reinterpret_cast<uint2*>(p + 4) = make_uint2(v.z, v.w);
      }
      if (dc == 0) for (int r = tid; r < NKT; r += kWarps * 32) sRm[r] = (t0 + r < nkbC) ? Rm[t0 + r] : 0.f;
      __syncthreads();
      g.template accumulate<DC>(sQ + dc, D + PAD, wm * SM, sKb, DC + PAD, wn * SN, lane);
    }
    // МАКСИМУМ ПО СТРОКАМ (запросам) ДЛЯ КАЖДОГО СТОЛБЦА (плитки ключей) -- это ТРАНСПОНИРОВАННАЯ
    // редукция относительно той, что делает мейнлуп, и карту надо читать внимательно:
    //   acc_row(lane,reg) = rowbase(биты 0,3,4 полосы) + 2*((reg>>1)&1)
    //   acc_col(lane,reg) = (биты 1,2 полосы)          + (reg&1) + 4*((reg>>2)&1)
    // Столбец, стало быть, задают биты 1,2 полосы и биты 0,2 регистра, а СТРОКУ -- биты 0,3,4 полосы
    // И БИТ 1 РЕГИСТРА. Значит один столбец лежит в ДВУХ регистрах (reg и reg^2), и обмены по битам
    // 0,3,4 покрывают только половину строк. Пары регистров надо свести ЯВНО, иначе половина запросов
    // молча теряется: (0,2), (1,3), (4,6), (5,7) -- ровно четыре столбца на полосу.
    float cm[NB][4], vs[MB][NB][8];
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int p = 0; p < 4; ++p) cm[j][p] = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int rowl = i * 16 + fa2_sm70::wmma_acc_row(lane, r);
          const int coll = j * 16 + fa2_sm70::wmma_acc_col(lane, r);
          const float v = g.acc[i][j][r] * qk_scale + alpha * sQn[wm * SM + rowl] * sRm[wn * SN + coll];
          vs[i][j][r] = v;
          const int p = (r & 1) | ((r >> 1) & 2);     // reg -> номер столбца в четвёрке
          if (v > cm[j][p]) cm[j][p] = v;
        }
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        float x = cm[j][p], y;
        y = __shfl_xor_sync(0xffffffffu, x, 1);  x = x > y ? x : y;   // биты СТРОКИ: 0, 3, 4
        y = __shfl_xor_sync(0xffffffffu, x, 8);  x = x > y ? x : y;
        y = __shfl_xor_sync(0xffffffffu, x, 16); x = x > y ? x : y;
        cm[j][p] = x;
      }
    // [СВЁРТКА ПО ЗАПРОСАМ: LSE, А НЕ МАКСИМУМ -- ЭТО ЗАМЕР, А НЕ ВКУСОВЩИНА]
    // Сверено на НАСТОЯЩЕМ внимании модели против оракула (tools/sparse_selector_vs_oracle.py,
    // 4 слоя x все головы x 12 блоков запросов). Порядок ранжировщиков по relL2 выхода:
    //     mean+lse  <  mean  <  mean+rad(Коши-Буняковский)  <  minmax(точная верхняя граница)
    // Оба «умных» проиграли: точная верхняя граница максимума В ПЛИТКЕ хуже среднего в 2-4 раза,
    // потому что отбирать надо по МАССЕ, а массу задают типичные ключи, а не крайние. LSE по 64
    // запросам вместо максимума по ним даёт ещё 5-12%: масса плитки складывается по ВСЕМ запросам
    // блока, а не определяется одним. Максимум оставлен переключателем (red=0) как след замера.
    if (red) {
      float sm[NB][4];
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int p = 0; p < 4; ++p) sm[j][p] = 0.f;
#pragma unroll
      for (int i = 0; i < MB; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int p = (r & 1) | ((r >> 1) & 2);
            sm[j][p] += exp2f(vs[i][j][r] - cm[j][p]);   // qk_scale уже несёт log2(e)
          }
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          float x = sm[j][p];
          x += __shfl_xor_sync(0xffffffffu, x, 1);
          x += __shfl_xor_sync(0xffffffffu, x, 8);
          x += __shfl_xor_sync(0xffffffffu, x, 16);
          cm[j][p] += log2f(x);                          // ранжирование монотонно, основание неважно
        }
    }
    __syncthreads();
    // Столбец держат полосы с одинаковыми битами 1,2; пишет представитель -- полоса с нулями в 0,3,4.
    if ((lane & 0x19) == 0) {
      const int cbase = (lane & 2) | ((lane & 4) << 1);
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int p = 0; p < 4; ++p)
          sRed[wm * NKT + wn * SN + j * 16 + cbase + (p & 1) + 4 * (p >> 1)] = cm[j][p];
    }
    __syncthreads();
    for (int r = tid; r < NKT; r += kWarps * 32) {
      if (t0 + r < nkbC) {
        float x = sRed[r];
#pragma unroll
        for (int w = 1; w < WM; ++w) { const float y = sRed[w * NKT + r]; x = x > y ? x : y; }
        sc[t0 + r] = x;
      }
    }
  }
  __syncthreads();

  // ЯКОРИ. Плитка на диагонали обязана остаться: без неё у части строк не будет НИ ОДНОГО
  // неотброшенного ключа, l_run = 0, и выход строки станет нулём -- это не «неточность», а брак.
  // Нулевая плитка (сток внимания) держится по той же причине, что во всех методах: там сидит
  // масса, не связанная с содержанием.
  if (tid == 0) { sc[nkbC - 1] = CUDART_INF_F; sc[0] = CUDART_INF_F; }
  // Якорей ДВА, значит и цель не может быть меньше двух -- иначе точный top-k выкинул бы один из них,
  // и «оставили диагональ» осталось бы только на словах.
  const int floor_ = nkbC < 2 ? nkbC : 2;
  const int target0 = (int)lrintf(dens[(long)b * nheads + h] * (float)nkbC);
  const int target = target0 < floor_ ? floor_ : (target0 > nkbC ? nkbC : target0);
  __syncthreads();

  // ---- ШАГ B: двухуровневая гистограмма
  const int nw = (nkbC + 31) >> 5;
  for (int i = tid; i < 256; i += kWarps * 32) sHist[i] = 0;
  for (int i = tid; i < nw; i += kWarps * 32) { bmA[i] = 0u; bmB[i] = 0u; }
  __syncthreads();
  for (int j = tid; j < nkbC; j += kWarps * 32) atomicAdd(&sHist[sortable(sc[j]) >> 24], 1);
  __syncthreads();
  if (tid == 0) {
    int run = 0, b1 = 0;
    for (b1 = 255; b1 > 0; --b1) { if (run + sHist[b1] >= target) break; run += sHist[b1]; }
    sB1 = b1; sAbove = run;
  }
  __syncthreads();
  const int b1 = sB1;
  for (int i = tid; i < 256; i += kWarps * 32) sHist[i] = 0;
  __syncthreads();
  for (int j = tid; j < nkbC; j += kWarps * 32) {
    const uint32_t u = sortable(sc[j]);
    if ((u >> 24) == (uint32_t)b1) atomicAdd(&sHist[(u >> 16) & 0xff], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int run = sAbove, b2 = 0;
    for (b2 = 255; b2 > 0; --b2) { if (run + sHist[b2] >= target) break; run += sHist[b2]; }
    sB2 = b2; sAbove = run; sNeed = target - run;    // сколько добрать из ГРАНИЧНОЙ корзины
  }
  __syncthreads();
  const uint32_t bnd = ((uint32_t)b1 << 8) | (uint32_t)sB2;
  for (int j = tid; j < nkbC; j += kWarps * 32) {
    const uint32_t u16 = sortable(sc[j]) >> 16;
    if (u16 > bnd)       atomicOr(&bmA[j >> 5], 1u << (j & 31));
    else if (u16 == bnd) atomicOr(&bmB[j >> 5], 1u << (j & 31));
  }
  __syncthreads();
  // добор границы В ПОРЯДКЕ ВОЗРАСТАНИЯ индекса: ранг граничной плитки < sNeed -> оставляем
  scan_bm(bmB, sOff, nw, tid);
  const int need = sNeed;
  for (int w = tid; w < nw; w += kWarps * 32) {
    uint32_t m = bmB[w]; int rk = sOff[w]; uint32_t add = 0u;
    while (m) { const int bit = __ffs((int)m) - 1; m &= m - 1u; if (rk < need) add |= 1u << bit; ++rk; }
    if (add) atomicOr(&bmA[w], add);
  }
  __syncthreads();

  // ---- ШАГ C: выдача. fmt=1 -- список (возрастающий), fmt=2 -- битовая карта как есть.
  scan_bm(bmA, sOff, nw, tid);
  if (fmt == 1) {
    for (int w = tid; w < nw; w += kWarps * 32) {
      uint32_t m = bmA[w]; int p = sOff[w];
      while (m) { const int bit = __ffs((int)m) - 1; m &= m - 1u; kmask[row * mask_ld + p++] = w * 32 + bit; }
    }
  } else {
    for (int w = tid; w < nw; w += kWarps * 32) kmask[row * mask_ld + w] = (int32_t)bmA[w];
  }
  if (tid == 0) kcnt[row] = sOff[nw];
}

}  // namespace sparse
}  // namespace fa2_sm70

// ==============================================================================================
// ОБВЯЗКА. Все точки входа принимают/возвращают РОВНО ТО ЖЕ, что attn_fwd_volta: Q [B,Sq,H,D] BSHD,
// K/V [B,Hkv,Sk,D] BHSD. Одна обвязка на все варианты -- иначе разница обвязок читается как эффект.
// ==============================================================================================
namespace {
constexpr int D_ = 256, BQ_ = 64, BK_ = 64, DC_ = 64, PIPE_ = 3, REV_ = 1, PAD_ = 4, MINB_ = 1;
constexpr int WM_ = 2, WN_ = 4;

size_t fwd_smem() {
  return (size_t)(BQ_ * (D_ + PAD_) + BQ_ * (BK_ + PAD_) + BK_ * (DC_ + PAD_) + BK_ * (D_ + 16)) * sizeof(__half);
}

struct Shape { int B, Sq, H, Hkv, Sk, nqb, nkb; };

Shape check(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    Shape s;
    s.B = Q.size(0); s.Sq = Q.size(1); s.H = Q.size(2);
    s.Hkv = K.size(1); s.Sk = K.size(2);
    TORCH_CHECK(Q.size(3) == D_ && K.size(3) == D_, "sparse fwd: head_dim 256");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && s.H % s.Hkv == 0, "sparse fwd: fp16 / GQA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "sparse fwd: contiguous");
    TORCH_CHECK(s.Sk >= s.Sq, "sparse fwd: bottom-right causal needs Sk >= Sq");
    s.nqb = (s.Sq + BQ_ - 1) / BQ_;
    s.nkb = (s.Sk + BK_ - 1) / BK_;
    return s;
}
}  // namespace

// ---- плотное внимание, ЗАМОРОЖЕННЫЙ отгруженный кернел (база «механизма нет»)
std::tuple<torch::Tensor, torch::Tensor> fwd_ref(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                 double scale, bool causal) {
    const c10::cuda::CUDAGuard g(Q.device());
    const Shape s = check(Q, K, V);
    auto O = torch::empty({s.B, s.Sq, s.H, D_}, torch::dtype(torch::kFloat16).device(Q.device()));
    const int ld = ((s.Sq + 31) / 32) * 32;
    auto lse = torch::empty({s.B, s.H, ld}, torch::dtype(torch::kFloat32).device(Q.device()));
    void (*kern)(const __half*, const __half*, const __half*, __half*, float*, int, int, float, int, int,
                 long, long, long, long, long, long, long, long, long, long, long) =
        causal ? fa2_sm70::block_fwd_ref::volta_fwd_block<BQ_, BK_, D_, WM_, WN_, true,  DC_, false, PIPE_, false, MINB_, 0, REV_>
               : fa2_sm70::block_fwd_ref::volta_fwd_block<BQ_, BK_, D_, WM_, WN_, false, DC_, false, PIPE_, false, MINB_, 0, REV_>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fwd_smem()));
    kern<<<dim3(s.nqb, s.H, s.B), WM_ * WN_ * 32, fwd_smem(), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()), lse.data_ptr<float>(),
        s.Sq, s.Sk, (float)scale * 1.4426950408889634f, s.H / s.Hkv, causal ? (s.Sk - s.Sq) : 0,
        (long)s.H * D_, (long)D_, (long)s.Sq * s.H * D_,
        (long)D_, (long)s.Sk * D_, (long)s.Hkv * s.Sk * D_,
        (long)s.H * D_, (long)D_, (long)s.Sq * s.H * D_, (long)ld, (long)s.H * ld);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(O, lse.narrow(2, 0, s.Sq).contiguous());
}

// ---- ядро С МЕХАНИЗМОМ: sp=0 плотный обход (тот же шаблон, доказывает, что рефакторинг бесплатен),
// sp=1 список, sp=2 битовая карта. jobq непустой -> очередь заданий вместо трёхмерной сетки.
std::tuple<torch::Tensor, torch::Tensor> fwd_sparse(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                    double scale, bool causal, int64_t sp,
                                                    torch::Tensor kmask, torch::Tensor kcnt,
                                                    torch::Tensor jobq) {
    const c10::cuda::CUDAGuard g(Q.device());
    const Shape s = check(Q, K, V);
    auto O = torch::empty({s.B, s.Sq, s.H, D_}, torch::dtype(torch::kFloat16).device(Q.device()));
    const int ld = ((s.Sq + 31) / 32) * 32;
    auto lse = torch::empty({s.B, s.H, ld}, torch::dtype(torch::kFloat32).device(Q.device()));
    const bool jq = jobq.numel() > 0;
    int mask_ld = 0;
    const int32_t *mp = nullptr, *cp = nullptr, *jp = nullptr;
    if (sp) {
        TORCH_CHECK(kmask.scalar_type() == torch::kInt32 && kcnt.scalar_type() == torch::kInt32, "маска int32");
        TORCH_CHECK(kmask.dim() == 2 && kmask.size(0) == (long)s.B * s.H * s.nqb, "маска [B*H*nqb, ld]");
        mask_ld = (int)kmask.size(1);
        mp = kmask.data_ptr<int32_t>(); cp = kcnt.data_ptr<int32_t>();
    }
    if (jq) { TORCH_CHECK(jobq.scalar_type() == torch::kInt32, "очередь int32"); jp = jobq.data_ptr<int32_t>(); }
    void (*kern)(const __half*, const __half*, const __half*, __half*, float*, int, int, float, int, int,
                 long, long, long, long, long, long, long, long, long, long, long,
                 const int32_t*, const int32_t*, int, int, int, const int32_t*) = nullptr;
#define PICK(SPx, JQx)                                                                               \
    kern = causal ? fa2_sm70::block_fwd::volta_fwd_block<BQ_, BK_, D_, WM_, WN_, true,  DC_, false, PIPE_, false, MINB_, 0, REV_, 0, 0, 0, 0, 0, SPx, JQx> \
                  : fa2_sm70::block_fwd::volta_fwd_block<BQ_, BK_, D_, WM_, WN_, false, DC_, false, PIPE_, false, MINB_, 0, REV_, 0, 0, 0, 0, 0, SPx, JQx>
    if (!jq) { if (sp == 0) { PICK(0, 0); } else if (sp == 1) { PICK(1, 0); }
               else if (sp == 3) { PICK(3, 0); } else { PICK(2, 0); } }
    else     { if (sp == 0) { PICK(0, 1); } else if (sp == 1) { PICK(1, 1); }
               else if (sp == 3) { PICK(3, 1); } else { PICK(2, 1); } }
#undef PICK
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fwd_smem()));
    const dim3 grid = jq ? dim3((unsigned)jobq.numel(), 1, 1) : dim3(s.nqb, s.H, s.B);
    kern<<<grid, WM_ * WN_ * 32, fwd_smem(), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()), lse.data_ptr<float>(),
        s.Sq, s.Sk, (float)scale * 1.4426950408889634f, s.H / s.Hkv, causal ? (s.Sk - s.Sq) : 0,
        (long)s.H * D_, (long)D_, (long)s.Sq * s.H * D_,
        (long)D_, (long)s.Sk * D_, (long)s.Hkv * s.Sk * D_,
        (long)s.H * D_, (long)D_, (long)s.Sq * s.H * D_, (long)ld, (long)s.H * ld,
        mp, cp, mask_ld, s.nqb, s.H, jp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(O, lse.narrow(2, 0, s.Sq).contiguous());
}

// ---- сводки плиток ключей
std::tuple<torch::Tensor, torch::Tensor> block_summary(torch::Tensor K, int64_t k0) {
    const c10::cuda::CUDAGuard g(K.device());
    const int B = K.size(0), Hkv = K.size(1), Sk = K.size(2);
    TORCH_CHECK(K.size(3) == D_ && K.scalar_type() == torch::kFloat16 && K.is_contiguous(), "сводки: fp16 [B,Hkv,Sk,256]");
    const int nkb = (Sk + BK_ - 1) / BK_;
    auto kbar = torch::empty({B, Hkv, nkb, D_}, torch::dtype(torch::kFloat16).device(K.device()));
    auto rmax = torch::empty({B, Hkv, nkb}, torch::dtype(torch::kFloat32).device(K.device()));
    const int n = (int)(nkb - k0);            // k0 -- с какой плитки считать (чанковый префилл)
    if (n > 0)
      fa2_sm70::sparse::kv_block_summary<BK_, D_><<<dim3(n, Hkv, B), D_, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(kbar.data_ptr<at::Half>()) , rmax.data_ptr<float>(),
        Sk, nkb, (long)D_, (long)Sk * D_, (long)Hkv * Sk * D_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(kbar, rmax);
}

// ---- отбор плиток
std::tuple<torch::Tensor, torch::Tensor> select_host(torch::Tensor Q, torch::Tensor kbar, torch::Tensor rmax,
                                               torch::Tensor dens, int64_t Sk, int64_t Hkv,
                                               double scale, bool causal, double alpha, int64_t fmt,
                                               int64_t red) {
    const c10::cuda::CUDAGuard g(Q.device());
    const int B = Q.size(0), Sq = Q.size(1), H = Q.size(2);
    TORCH_CHECK(Q.size(3) == D_ && Q.scalar_type() == torch::kFloat16 && Q.is_contiguous(), "отбор: fp16 BSHD");
    const int nqb = (Sq + BQ_ - 1) / BQ_, nkb = ((int)Sk + BK_ - 1) / BK_;
    TORCH_CHECK(nkb <= fa2_sm70::sparse::kMaxNkb, "отбор: Sk <= 262144 (оценки живут в разделяемой)");
    const int mask_ld = (fmt == 1) ? nkb : ((nkb + 31) / 32);
    // ZEROS, а не empty: при битовой карте блок запросов заполняет только nw слов из mask_ld, и
    // мусор в хвосте -- это лишние «оставленные» плитки при следующем, более длинном контексте.
    auto kmask = torch::zeros({(long)B * H * nqb, mask_ld}, torch::dtype(torch::kInt32).device(Q.device()));
    auto kcnt = torch::empty({(long)B * H * nqb}, torch::dtype(torch::kInt32).device(Q.device()));
    constexpr int NKT = 64;
    const size_t sh = (size_t)(BQ_ * (D_ + PAD_) + NKT * (DC_ + PAD_)) * sizeof(__half)
                    + (size_t)(fa2_sm70::sparse::kMaxNkb + WM_ * NKT + BQ_ + NKT) * sizeof(float)
                    + (size_t)(2 * fa2_sm70::sparse::kBmWords + fa2_sm70::sparse::kBmWords + 1) * sizeof(int);
    auto kern = fa2_sm70::sparse::select_blocks<BQ_, BK_, D_, NKT, WM_, WN_, DC_>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    kern<<<dim3(nqb, H, B), WM_ * WN_ * 32, sh, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kbar.data_ptr<at::Half>()), rmax.data_ptr<float>(),
        dens.data_ptr<float>(), kmask.data_ptr<int32_t>(), kcnt.data_ptr<int32_t>(),
        Sq, (int)Sk, nkb, mask_ld, nqb, H, H / (int)Hkv,
        (float)scale * 1.4426950408889634f, (float)alpha, causal ? ((int)Sk - Sq) : 0, (int)fmt, (int)red,
        (long)H * D_, (long)D_, (long)Sq * H * D_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(kmask, kcnt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd_ref", &fwd_ref, "плотный forward d=256, ЗАМОРОЖЕННЫЙ отгруженный кернел");
    m.def("fwd_sparse", &fwd_sparse, "forward d=256 с механизмом разреженности (sp=0/1/2, очередь заданий)");
    m.def("block_summary", &block_summary, "сводки плиток ключей: средний ключ + радиус");
    m.def("select", &select_host, "отбор плиток ключей: Q x сводки на тензорных ядрах + точный top-k");
}
