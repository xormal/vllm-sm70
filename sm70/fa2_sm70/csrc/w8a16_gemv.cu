// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Fused W8A16 GEMV for GPTQ int8 weights on sm_70: the weight is consumed PACKED and unpacked in the
// shadow of its own load, feeding the fp16 pipeline. No INT8 tensor cores are involved -- Volta has
// none (IMMA arrived with Turing), and it does not need them.
//
// WHY THIS SHAPE OF KERNEL, MEASURED
//   int8 matmul through DP4A on this card is 54.6 TOPS against 93.6 TFLOPS for fp16 HMMA -- 0.58x.
//   So computing in int8 is the wrong move here; the win is in READING less. At M=1 the same
//   comparison, on a 5760x2880 weight:
//       dense fp16 (cublas HMMA)              51.4 us   646 GB/s   1.00x
//       packed 4-bit, unpack fused (ours)     35.8 us   232 GB/s   1.44x   <- 4x fewer bytes
//       int8 dequantised in a SEPARATE pass  283.0 us    59 GB/s   0.18x   <- the trap
//   Unpacking in a separate pass is 5.5x SLOWER than not quantising at all, because it writes the
//   whole dense weight to memory and reads it back. Fusing is not an optimisation here; it is the
//   difference between a win and a large loss.
//
//   Decode is where this pays: the weights are read ONCE PER TOKEN with no reuse, and on the 27B/TP=2
//   deployment that read is 13.5 GB per card, 48.4% of a 34 ms step. Prefill does not benefit -- a
//   weight tile is reused by ~8192 tokens there, so its 16 ms read hides behind ~2400 ms of GEMM.
//
// LAYOUT (GPTQ, bits=8, sym=false, desc_act=false)
//   qweight [K/4, N] int32 -- four consecutive k in one word, so a warp with one COLUMN per lane
//                             reads 32 consecutive words = 128 B, perfectly coalesced
//   qzeros  [K/G, N/4] int32 -- four consecutive n per word
//   scales  [K/G, N] fp16
//   w[k,n] = (q[k,n] - z[k/G,n]) * s[k/G,n]
//
// One column per THREAD also removes the cross-lane reduction entirely: each thread owns a whole
// output element and writes it once.
#include <torch/extension.h>
#include <unordered_map>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "w8a16_hmma.cuh"

// Shared-memory budget for the staged activation, in fp32 elements. The k-tile and the M-tile trade
// against each other inside it: MT rows x XTILE keys. 4096 floats = 16 KB leaves room for 6 blocks/SM.
#define XSMEM 4096
constexpr int xtile_of(int MT) { int t = XSMEM / MT; return t > 1024 ? 1024 : (t < 128 ? 128 : t); }
#define MAXG (1024 / 128)   // groups in the widest k-tile

// SPLIT-K IS NOT AN OPTIMISATION HERE, IT IS THE WHOLE KERNEL.
// One column per thread gives grid = N/THREADS blocks. On the real checkpoint (N=5120, THREADS=256)
// that is TWENTY blocks on an 80-SM card: three quarters of the machine idle, each thread walking all
// of K=17408 alone. Measured 48 GB/s = 0.19x of dense fp16 -- and hoisting the group metadata out of
// the inner loop moved it only 40 -> 48, which is what proves the limiter is not the inner loop.
// The arithmetic was never in question: 89 MB of packed weight is ~6 us of fma against ~174 us of
// unavoidable memory stream. Only the grid was wrong. With the k-slice split it measured 657 GB/s.
//
// MT (rows of x per pass) exists for the same reason. With MT=1 the weight is re-read once per row,
// so batched decode collapsed exactly linearly -- 2.56x at M=1, 0.43x at M=4, 0.11x at M=16. Spending
// one unpacked weight word on MT accumulators is what makes the packed read pay for a batch. The
// arrays below are indexed only by fully-unrolled loops: a runtime index would put them in local
// memory and silently undo the whole thing (ptxas -v confirms 0 bytes spilled at every MT).
//
// THE ACTIVATION IS STAGED AS fp32, NOT fp16, AND THE GROUP SUMS ARE PRECOMPUTED.
// Staging it as fp16 costs a half->float conversion of the SAME value once per output column, i.e. N
// times over, and folding the zero as (q-z)*x needs a running Sum x that costs 3 more adds per element
// per row. Both are loop-invariant in n. Converting once at staging time and reading float4 turns the
// inner step into 1 LDS.128 + 4 FFMA per row per weight word, down from ~13 instructions. ptxas showed
// no spill in either version, so the first MT=16 result (0.28x) was pure instruction issue, not
// occupancy -- the fp32 SIMT pipe is the wall at high MT, which is exactly what tensor cores exist for.

// ============================ ЧЕТЫРЁХСТОЛБЦОВЫЙ ВАРИАНТ ============================
// [ПОЧЕМУ ОН ЕСТЬ -- НАЗВАНО МАШИННЫМ КОДОМ, А НЕ ДОГАДКОЙ]
// Укладчик по телу мейнлупа (tools/kanal_ili_cep.py): у НАС связывает MIO на 100 % при 144
// командах этого канала, у чужого `vllm::gptq::gemm_half_q_half_gptq_8bit` -- FPU, а MIO всего
// 13.8 % при 64 командах. Разбор SASS показал ровно причину: у чужого есть `LDG.E.128`, у нас его
// НЕТ ВОВСЕ -- вес читается 32-разрядными загрузками. Байт столько же, команд вчетверо больше, и
// на теле, связанном этим каналом, это и есть потеря (замер на месте: 74.5 мкс против 66.9).
//
// ХОД. Раскладка `qweight [K/4, N]` по N НЕПРЕРЫВНА, значит один поток может взять ЧЕТЫРЕ соседних
// столбца одним `uint4`. Даром достаётся и второе: `qzeros` пакует РОВНО 4 столбца в слово, то есть
// одна загрузка нуля обслуживает все четыре -- ни сдвига, ни маски. Масштабы (4 подряд `__half`)
// берутся одним 64-разрядным чтением.
//
// ТОЧНОСТЬ НЕ ТРОГАЕТСЯ: та же арифметика в fp32, тот же порядок суммирования по k, та же
// условность нуля. Меняется ТОЛЬКО ширина обращения -- это размещение, а не вычисление.
// [ШИРИНА СТОЛБЦОВ -- ПАРАМЕТР, А НЕ КОНСТАНТА. ЗАМЕР РЕГИСТРОВ 25.08]
// ptxas -v на этом самом ядре: MT=1 -> 63 рег (32 варпа/SM), MT=2 -> 79 (24), MT=4 -> 114 (16),
// MT=8 -> 156 (8 варпов из 64). Занятость обваливается ровно там, где падает полоса:
// down 8704x5120 даёт 687 ГБ/с при M=1, 522 при M=4 и 293 при M=8. То есть при M>1 ядро
// перестаёт прятать латентность HBM -- не из-за счёта, а из-за НЕХВАТКИ ВАРПОВ.
// Регистры считаны поимённо: acc[MT][VW] + qa[MT][VW] + g[VW][4] + s[VW] + z[VW]. Половина из
// них масштабируется ШИРИНОЙ СТОЛБЦОВ, и она была прибита четвёркой. Делаем её параметром:
// при MT>=4 берём VW=2 -- чтение остаётся векторным (LDG.64, 256 нитей * 8 Б = 2 КиБ подряд),
// а регистров вдвое меньше на этой половине.
template <int THREADS, int MT, bool SPLIT, int VW = 4>
__global__ void w8a16_gemv4_kernel(const __half* __restrict__ x,
                                   const int* __restrict__ qweight,
                                   const int* __restrict__ qzeros,
                                   const __half* __restrict__ scales,
                                   __half* __restrict__ y, float* __restrict__ ypart,
                                   int M, int N, int K, int G, int groups_per_split, int m0,
                                   float zbias) {
  constexpr int XT = xtile_of(MT);
  constexpr bool GSUM = (MT > 1);
  __shared__ float sx[MT][XT];
  __shared__ float gs[GSUM ? MT : 1][GSUM ? MAXG : 1];
  const int n0 = (blockIdx.x * THREADS + threadIdx.x) * VW;  // VW столбцов на нить
  const bool live = (n0 + VW - 1 < N);
  const int nw = n0 >> 2;                                    // слово qzeros на ЧЕТЫРЕ столбца
  const int nb = n0 & 3;                                     // наш сдвиг внутри слова
  const int mrows = min(MT, M - m0);

  const int kbeg = SPLIT ? blockIdx.y * groups_per_split * G : 0;
  const int kend = SPLIT ? min(K, kbeg + groups_per_split * G) : K;
  if (kbeg >= kend) return;

  float acc[MT][VW];
  #pragma unroll
  for (int mm = 0; mm < MT; ++mm)
    #pragma unroll
    for (int c = 0; c < VW; ++c) acc[mm][c] = 0.f;

  for (int k0 = kbeg; k0 < kend; k0 += XT) {
    const int len = min(XT, kend - k0);
    const int ng = (len + G - 1) / G;
    __syncthreads();
    for (int t = threadIdx.x; t < len * mrows; t += THREADS) {
      const int mm = t / len, kk = t - mm * len;
      sx[mm][kk] = __half2float(x[(long)(m0 + mm) * K + k0 + kk]);
    }
    __syncthreads();
    if constexpr (GSUM) {
      for (int t = threadIdx.x; t < ng * mrows; t += THREADS) {
        const int mm = t / ng, g = t - mm * ng;
        float a = 0.f;
        for (int i = g * G; i < min((g + 1) * G, len); ++i) a += sx[mm][i];
        gs[mm][g] = a;
      }
      __syncthreads();
    }
    if (!live) continue;
    for (int gk = 0; gk < len; gk += G) {
      const int g = (k0 + gk) / G;
      // масштабы: 4 подряд __half = 8 байт одним чтением
      float s[VW];
      #pragma unroll
      for (int c = 0; c < VW; c += 2) {
        const __half2 h2 = *reinterpret_cast<const __half2*>(scales + (long)g * N + n0 + c);
        s[c] = __low2float(h2); s[c + 1] = __high2float(h2);
      }
      // нули: ОДНО слово qzeros несёт ровно наши четыре столбца
      const unsigned zw = (unsigned)qzeros[(long)g * (N >> 2) + nw];
      // [СМЕЩЕНИЕ РАЗВОРОТА СВОРАЧИВАЕТСЯ В НОЛЬ -- ЦЕНА РОВНО НОЛЬ]
      // Разворот ниже даёт не q, а (1024 + q). Подстановка:
      //     s*(sum x*q - z*sum x) = s*(sum x*(1024+q) - (z + 1024)*sum x)
      // Сумма x у нас УЖЕ есть (её считает та же поправка на ноль), поэтому 1024 просто уезжает
      // в константу нуля и не стоит ни одной команды.
      float z[VW];
      #pragma unroll
      for (int c = 0; c < VW; ++c) z[c] = (float)((zw >> (8 * (nb + c))) & 0xFF) + zbias + 1024.0f;
      const int glen = min(G, len - gk);
      float qa[MT][VW], xa[MT];
      #pragma unroll
      for (int mm = 0; mm < MT; ++mm) {
        xa[mm] = 0.f;
        #pragma unroll
        for (int c = 0; c < VW; ++c) qa[mm][c] = 0.f;
      }
      #pragma unroll 4
      for (int k = 0; k < glen; k += 4) {
        // ОДНА LDG.128 вместо четырёх LDG.32: четыре СТОЛБЦА подряд, у каждого свои 4 k
        unsigned wc[VW];
        if constexpr (VW == 4) {
          const uint4 w4 = *reinterpret_cast<const uint4*>(
              qweight + (long)((k0 + gk + k) >> 2) * N + n0);
          wc[0] = w4.x; wc[1] = w4.y; wc[2] = w4.z; wc[3] = w4.w;
        } else {
          const uint2 w2 = *reinterpret_cast<const uint2*>(
              qweight + (long)((k0 + gk + k) >> 2) * N + n0);
          wc[0] = w2.x; wc[1] = w2.y;
        }
        // [РАЗВОРОТ БАЙТА -- ДВЕ ПЕРЕСТАНОВКИ ВМЕСТО ДВЕНАДЦАТИ КОМАНД]  §1b библиотеки примитивов
        // Было на каждый байт: SHF + LOP3 + I2F. Разбор SASS после vec4 показал, что стена
        // переехала с MIO (100 % -> 18.4 %) на ISSUE/ALU, и в теле сидят I2F.U8 x1076, SHF x556,
        // LOP3 x441 -- то есть связывает именно РАЗВОРОТ, а не умножение и не память.
        // `__byte_perm(w, 0x64646464, sel)` собирает полуслово [байт, 0x64] = fp16-число 1024+u.
        // Байт GPTQ БЕЗЗНАКОВЫЙ, поэтому XOR со знаком (нужный KV-кэшу) здесь не требуется вовсе:
        // две команды на четыре байта. Смещение 1024 уже свёрнуто в ноль выше.
        // ТОЧНОСТЬ: (1024+q) представимо в fp16 ТОЧНО (шаг мантиссы там ровно 1 при q<=255), а
        // произведения растут лишь впятеро против сигнала -- ошибка fp32-суммы ~5e-6 при нашем
        // уровне 2.05e-4, то есть на два порядка ниже. Это тот же приём и та же оценка, что у
        // смещения 1152 в ядре внимания.
        float g[VW][4];
        #pragma unroll
        for (int c = 0; c < VW; ++c) {
          const uint32_t f01 = __byte_perm(wc[c], 0x64646464u, 0x4140);
          const uint32_t f23 = __byte_perm(wc[c], 0x64646464u, 0x4342);
          const float2 a01 = __half22float2(*reinterpret_cast<const __half2*>(&f01));
          const float2 a23 = __half22float2(*reinterpret_cast<const __half2*>(&f23));
          g[c][0] = a01.x; g[c][1] = a01.y; g[c][2] = a23.x; g[c][3] = a23.y;
        }
        #pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
          const float4 a = *reinterpret_cast<const float4*>(&sx[mm][gk + k]);
          #pragma unroll
          for (int c = 0; c < VW; ++c) {
            qa[mm][c] = fmaf(a.x, g[c][0], qa[mm][c]);
            qa[mm][c] = fmaf(a.y, g[c][1], qa[mm][c]);
            qa[mm][c] = fmaf(a.z, g[c][2], qa[mm][c]);
            qa[mm][c] = fmaf(a.w, g[c][3], qa[mm][c]);
          }
          if constexpr (!GSUM) xa[mm] += a.x + a.y + a.z + a.w;
        }
      }
      #pragma unroll
      for (int mm = 0; mm < MT; ++mm) {
        const float sx_sum = GSUM ? gs[mm][gk / G] : xa[mm];
        #pragma unroll
        for (int c = 0; c < VW; ++c)
          acc[mm][c] = fmaf(s[c], qa[mm][c] - z[c] * sx_sum, acc[mm][c]);
      }
    }
  }
  if (!live) return;
  #pragma unroll
  for (int mm = 0; mm < MT; ++mm)
    #pragma unroll
    for (int c = 0; c < VW; ++c) {
      // [ДЕТЕРМИНИРОВАННЫЙ СВОД] Здесь стоял atomicAdd, и это делало ответ НЕПОВТОРЯЕМЫМ:
      // сложение float атомарно идёт в ПОРЯДКЕ ПРИХОДА блоков, а он меняется от пуска к пуску.
      // Отчёт падений §8-кватер меряет это как дрожание логитов +-0.003; на вырожденных
      // выдачах оно переворачивает выбор токена. Пишем в СВОЙ слот расщепления, свод -- в
      // фиксированном порядке эпилогом. Обнуление площадки при этом больше не нужно:
      // все использованные слоты перезаписываются целиком.
      if constexpr (SPLIT) ypart[(long)blockIdx.y * M * N + (long)(m0 + mm) * N + n0 + c] = acc[mm][c];
      else                 y[(long)(m0 + mm) * N + n0 + c] = __float2half(acc[mm][c]);
    }
}

template <int THREADS, int MT, bool SPLIT>
__global__ void w8a16_gemv_kernel(const __half* __restrict__ x,
                                  const int* __restrict__ qweight,
                                  const int* __restrict__ qzeros,
                                  const __half* __restrict__ scales,
                                  __half* __restrict__ y, float* __restrict__ ypart,
                                  int M, int N, int K, int G, int groups_per_split, int m0, float zbias) {
  constexpr int XT = xtile_of(MT);
  // At MT=1 the precomputed group sum does not pay: it saves 3 adds per element for a SINGLE row but
  // costs a barrier and a pass that only ng threads take part in (measured 2.56x -> 2.31x). Above one
  // row it pays, because the saving scales with MT while the barrier does not.
  constexpr bool GSUM = (MT > 1);
  __shared__ float sx[MT][XT];
  __shared__ float gs[GSUM ? MT : 1][GSUM ? MAXG : 1];   // Sum x over each group of the tile
  const int n = blockIdx.x * THREADS + threadIdx.x;      // this thread's output column
  const bool live = (n < N);
  const int nw = n >> 2, nsh = (n & 3) * 8;              // which qzeros word / byte
  const int mrows = min(MT, M - m0);                     // ragged last M-chunk

  const int kbeg = SPLIT ? blockIdx.y * groups_per_split * G : 0;
  const int kend = SPLIT ? min(K, kbeg + groups_per_split * G) : K;
  if (kbeg >= kend) return;

  float acc[MT];
  #pragma unroll
  for (int mm = 0; mm < MT; ++mm) acc[mm] = 0.f;

  for (int k0 = kbeg; k0 < kend; k0 += XT) {
    const int len = min(XT, kend - k0);
    const int ng = (len + G - 1) / G;
    __syncthreads();
    for (int t = threadIdx.x; t < len * mrows; t += THREADS) {
      const int mm = t / len, kk = t - mm * len;
      sx[mm][kk] = __half2float(x[(long)(m0 + mm) * K + k0 + kk]);
    }
    __syncthreads();
    // Reducing the group sums in the SAME pass via shared atomics was tried and is worse everywhere
    // (MT=16: 485 -> 1046 us): the contention on ng*mrows slots costs more than the extra barrier.
    if constexpr (GSUM) {
      for (int t = threadIdx.x; t < ng * mrows; t += THREADS) {   // one thread per (row, group)
        const int mm = t / ng, g = t - mm * ng;
        float a = 0.f;
        for (int i = g * G; i < min((g + 1) * G, len); ++i) a += sx[mm][i];
        gs[mm][g] = a;
      }
      __syncthreads();
    }
    if (!live) continue;
    // Outer loop over GROUPS, inner over the group's k. This is what lets the scale and zero be
    // fetched once per 128 k instead of once per 4, AND lets the inner loop unroll (a group-change
    // branch inside the k-loop blocks unrolling, so the two fixes are one fix).
    for (int gk = 0; gk < len; gk += G) {
      const int g = (k0 + gk) / G;
      const float s = __half2float(scales[(long)g * N + n]);
      // [УСЛОВНОСТЬ НУЛЯ -- НЕ ДЕТАЛЬ ФОРМАТА, А ОДИН УРОВЕНЬ КВАНТА НА КАЖДОМ ВЕСЕ]
      // GPTQ v1 (`checkpoint_format: gptq`) хранит (z-1), v2 (`gptq_v2`) хранит z. Ядро читало
      // СЫРОЕ значение, то есть жило по v2, а боевой слепок Qwen3.8 -- v1. Промах на единицу даёт
      // систематическую ошибку ~1/sigma_q ~ 1.4 %: ЗАМЕРЕНО 16.08 против чужого тела -- 5.55e-3 при
      // v1 и 1.20e-3 (= округление вывода в fp16) при v2. Он НЕ ПАДАЕТ и НЕ ВИДЕН на глаз, поэтому
      // условность обязана приходить СНАРУЖИ, а не подразумеваться. При верной условности наше тело
      // точнее чужого: 2.04e-4 против 1.18e-3 к fp32-эталону.
      const float z = (float)((qzeros[(long)g * (N >> 2) + nw] >> nsh) & 0xFF) + zbias;
      const int glen = min(G, len - gk);
      float qa[MT], xa[MT];
      #pragma unroll
      for (int mm = 0; mm < MT; ++mm) { qa[mm] = 0.f; xa[mm] = 0.f; }
      #pragma unroll 4
      for (int k = 0; k < glen; k += 4) {
        // one int32 of qweight carries four consecutive k for this column; a warp holds 32
        // consecutive columns, so this load is 32 consecutive words = 128 B, fully coalesced
        const unsigned w = (unsigned)qweight[(long)((k0 + gk + k) >> 2) * N + n];
        const float q0 = (float)( w        & 0xFF), q1 = (float)((w >>  8) & 0xFF);
        const float q2 = (float)((w >> 16) & 0xFF), q3 = (float)((w >> 24) & 0xFF);
        #pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
          const float4 a = *reinterpret_cast<const float4*>(&sx[mm][gk + k]);
          qa[mm] = fmaf(a.x, q0, qa[mm]); qa[mm] = fmaf(a.y, q1, qa[mm]);
          qa[mm] = fmaf(a.z, q2, qa[mm]); qa[mm] = fmaf(a.w, q3, qa[mm]);
          if constexpr (!GSUM) xa[mm] += a.x + a.y + a.z + a.w;
        }
      }
      #pragma unroll
      for (int mm = 0; mm < MT; ++mm)                    // (q-z)*s summed = s*(Sum q*x - z*Sum x)
        acc[mm] = fmaf(s, qa[mm] - z * (GSUM ? gs[mm][gk / G] : xa[mm]), acc[mm]);
    }
  }
  if (!live) return;
  #pragma unroll
  for (int mm = 0; mm < MT; ++mm) {
    if (mm >= mrows) break;
    if (SPLIT) ypart[(long)blockIdx.y * M * N + (long)(m0 + mm) * N + n] = acc[mm];   // см. выше
    else       y[(long)(m0 + mm) * N + n] = __float2half(acc[mm]);
  }
}

// [ЭПИЛОГ ОБНУЛЯЕТ ПЛОЩАДКУ САМ -- И ЭТО ДАРОМ]
// Раньше площадка k-расщепления заводилась `torch::zeros` НА КАЖДЫЙ ВЫЗОВ, то есть отдельным
// пуском обнуления. Замер ленты 16.08: `FillFunctor` -- 32272 пуска по 1.7 мкс = 56.5 мс = 1.41 %
// шага декода, и это НАША трата. Расщепление при этом СНЯТЬ НЕЛЬЗЯ: без него ядро в 5 раз
// медленнее (замерено: отношение к чужому 1.060 -> 0.207), потому что блоков выходит 5..17 на
// 80 SM. Значит убирать надо обнуление, а не расщепление: площадка становится ПОСТОЯННОЙ, а
// обнуляет её ЭТОТ эпилог, который те же элементы и так читает. Плата -- одна запись нуля на
// элемент (70 КБ на вызов при N=17408, ~0.08 мкс) против 1.7 мкс отдельного пуска.
__global__ void w8a16_cast_kernel(float* __restrict__ src, __half* __restrict__ dst, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { dst[i] = __float2half(src[i]); src[i] = 0.f; }
}

// [СВОД РАСЩЕПЛЕНИЙ В ФИКСИРОВАННОМ ПОРЯДКЕ] Заменяет atomicAdd в теле: каждое расщепление
// пишет в свой слот, а здесь они складываются ВСЕГДА в одном порядке 0..ks-1. Ответ становится
// побитово повторяемым от пуска к пуску. Обнуление не нужно -- слоты перезаписываются целиком.
__global__ void w8a16_reduce_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                    long n, int ks) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = 0.f;
  for (int k = 0; k < ks; ++k) a += src[(long)k * n + i];
  dst[i] = __float2half(a);
}

// Путь на ТЕНЗОРНЫХ ядрах для M>=8, где SIMT-путь проигрывает плотному cublas: там перелом
// физический (15.7 TFLOPS против 93.6), и снять его можно только выйдя на HMMA.
torch::Tensor w8a16_gemm_hmma(torch::Tensor x, torch::Tensor qweight, torch::Tensor qzeros,
                              torch::Tensor scales, int64_t group_size, bool v2_zero) {
  const float zbias = v2_zero ? 0.0f : 1.0f;   // умолчания НЕТ намеренно -- см. w8a16_hmma.cuh
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda(), "w8a16 hmma: CUDA");
  auto xc = x.contiguous(), qw = qweight.contiguous(), qz = qzeros.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(1), K = (int)qw.size(0) * 4;
  const int M = (int)(xc.numel() / xc.size(-1));
  TORCH_CHECK(group_size == 128, "w8a16 hmma: группа 128");
  TORCH_CHECK(K % 128 == 0 && (N & 3) == 0, "w8a16 hmma: K кратно 128, N кратно 4");
  auto y = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(xc.device()));
  constexpr int BN = 64, WN = 4, G = 128;
  const int nblk = (N + BN - 1) / BN, mblk = (M + 15) / 16;
  const int ngroups = K / G;
  // ЗАПОЛНЕНИЕ МАШИНЫ. Правило то же, что у gemv: целимся примерно в 8 блоков на SM (80 SM).
  // Без этого на узкой форме (N=5120 -> nblk=80, mblk=1) сетка ровно 80 блоков, то есть один на SM.
  static const bool ks_off = (getenv("FA2SM70_HMMA_KSPLIT") != nullptr
                              && atoi(getenv("FA2SM70_HMMA_KSPLIT")) == 1);
  int ksplit = ks_off ? 1 : max(1, min(ngroups, (80 * 8 + nblk * mblk - 1) / (nblk * mblk)));
  const int gps = (ngroups + ksplit - 1) / ksplit;
  ksplit = (ngroups + gps - 1) / gps;                    // обрезать пустые хвостовые куски
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  const auto* sp = reinterpret_cast<const __half*>(sc.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  if (ksplit <= 1) {
    dim3 grid(nblk, mblk);
    fa2_sm70::w8a16::w8a16_hmma<BN, WN, G, false><<<grid, WN * 32, 0, stream>>>(
        xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, yp, M, N, K, zbias);
  } else {
    // ПОСТОЯННАЯ ПЛОЩАДКА НА УСТРОЙСТВО, приходит НУЛЕВОЙ: первый раз её обнуляет `torch::zeros`,
    // дальше -- эпилог предыдущего вызова (тот же приём, что у gemv: адрес стабилен, значит
    // захват в CUDA-граф законен и ни одного пуска обнуления на вызов).
    static std::unordered_map<int, torch::Tensor> кэш_hmma;
    static std::vector<torch::Tensor> кэш_hmma_старые;   // [14.10] адрес запечён в графах -- не отпускать
    // Слот НА КАЖДОЕ расщепление -- см. пояснение у записи в w8a16_hmma.cuh.
    const long need = (long)ksplit * M * N;
    const int dev = (int)xc.device().index();
    auto it = кэш_hmma.find(dev);
    if (it == кэш_hmma.end() || it->second.numel() < need) {
      if (it != кэш_hmma.end()) кэш_hmma_старые.push_back(it->second);
      кэш_hmma[dev] = torch::zeros({max(need, (long)1 << 20)},
                                   torch::dtype(torch::kFloat32).device(xc.device()));
    }
    float* pp = кэш_hmma[dev].data_ptr<float>();
    dim3 grid(nblk, mblk, ksplit);
    fa2_sm70::w8a16::w8a16_hmma<BN, WN, G, true><<<grid, WN * 32, 0, stream>>>(
        xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, nullptr, M, N, K, zbias, pp, gps);
    w8a16_reduce_kernel<<<((long)M * N + 255) / 256, 256, 0, stream>>>(pp, yp, (long)M * N, ksplit);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor w8a16_gemv(torch::Tensor x, torch::Tensor qweight, torch::Tensor qzeros,
                         torch::Tensor scales, int64_t group_size, bool v2_zero) {
  // v2_zero=false -- слепок в формате `gptq` (v1, хранится z-1); true -- `gptq_v2`.
  // Умолчания НЕТ намеренно: молчаливое умолчание здесь и есть тот дефект, что был.
  const float zbias = v2_zero ? 0.0f : 1.0f;
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda(), "w8a16: CUDA tensors required");
  TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2, "w8a16: qweight [K/4,N], scales [K/G,N]");
  auto xc = x.contiguous(), qw = qweight.contiguous(), qz = qzeros.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(1), K = (int)qw.size(0) * 4;
  const int M = (int)(xc.numel() / xc.size(-1));
  const int G = (int)group_size;
  TORCH_CHECK(xc.size(-1) == K, "w8a16: K mismatch");
  TORCH_CHECK((N & 3) == 0, "w8a16: N must be a multiple of 4 (qzeros packs 4 columns per word)");
  auto y = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(xc.device()));
  constexpr int THREADS = 256;
  const int nblk = (N + THREADS - 1) / THREADS;
  const int ngroups = (K + G - 1) / G;
  // Fill the machine: aim for ~8 blocks per SM across 80 SMs. Below that the kernel is latency-bound
  // on a stream it should be bandwidth-bound on (measured: 20 blocks -> 48 GB/s of an achievable 841).
  // [ЦЕНА РАСЩЕПЛЕНИЯ -- НЕ ТОЛЬКО АТОМАРНЫЕ] При ksplit>1 заводится площадка `torch::zeros`, а
  // это ОТДЕЛЬНЫЙ пуск обнуления на КАЖДЫЙ вызов. Замер ленты 16.08: `FillFunctor` -- 32272 пуска
  // по 1.7 мкс = 56.5 мс = 1.41 % шага декода, и это НАША трата, а не движка. Рычаг для замера:
  // FA2SM70_W8_KSPLIT=1 запрещает расщепление совсем.
  static const bool ks_off = (getenv("FA2SM70_W8_KSPLIT") != nullptr
                              && atoi(getenv("FA2SM70_W8_KSPLIT")) == 1);
  int ksplit = ks_off ? 1 : max(1, min(ngroups, (80 * 8 + nblk - 1) / nblk));
  const int gps = (ngroups + ksplit - 1) / ksplit;
  ksplit = (ngroups + gps - 1) / gps;                    // trim empty tail splits
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  const auto* sp = reinterpret_cast<const __half*>(sc.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  // ПОСТОЯННАЯ ПЛОЩАДКА НА УСТРОЙСТВО. Она приходит УЖЕ НУЛЕВОЙ: при первом заведении её обнуляет
  // `torch::zeros`, дальше -- эпилог предыдущего вызова. Адрес стабилен, значит захват в CUDA-граф
  // законен (свежая аллокация на каждый вызов тоже стабильна внутри графа, но тянет за собой пуск
  // обнуления, ради снятия которого всё и делается).
  static std::unordered_map<int, torch::Tensor> кэш;
  // [ВИСЯЧИЙ АДРЕС -- 14.10] Площадка растёт с M (голова цели зовёт без графа с M = 4 x запросов), а её адрес
  // запечён в графах тела MTP (M=4). Старую больше не отпускаем: повтор графа продолжает писать в неё и сам же
  // её обнуляет эпилогом, чужая память не задевается.
  static std::vector<torch::Tensor> кэш_старые;
  float* pp = nullptr;
  if (ksplit > 1) {
    // Слот НА КАЖДОЕ расщепление (было: одна общая площадка под atomicAdd). Размер ограничен
    // сверху самим правилом выбора ksplit ~ 640/nblk при nblk ~ N/256, то есть ksplit*N ~ 164k
    // элементов на строку -- около 0.64 МиБ на строку M.
    const long need = (long)ksplit * M * N;
    const int dev = (int)xc.device().index();
    auto it = кэш.find(dev);
    if (it == кэш.end() || it->second.numel() < need) {
      if (it != кэш.end()) кэш_старые.push_back(it->second);
      кэш[dev] = torch::zeros({max(need, (long)1 << 20)},
                              torch::dtype(torch::kFloat32).device(xc.device()));
    }
    pp = кэш[dev].data_ptr<float>();
  }
  const dim3 grid = (ksplit > 1) ? dim3(nblk, ksplit) : dim3(nblk);
  static const bool vec_off = (getenv("FA2SM70_W8_VEC") != nullptr
                               && atoi(getenv("FA2SM70_W8_VEC")) == 0);
  // [ВОРОТА БЫЛИ ШИРЕ, ЧЕМ НАДО, И ЭТО СТОИЛО СЛОВАРНОЙ ПРОЕКЦИИ]
  // Требовалось N % (THREADS*4) == 0, то есть кратность 1024. У словаря N = 124160 -- кратно 4,
  // но не 1024, и вся проекция падала на СКАЛЯРНЫЙ путь (LDG.32), ради снятия которого vec4 и
  // писался. Достаточное условие -- РОВНО N % 4 == 0: `qzeros` пакует 4 столбца в слово, а
  // краевой блок уже закрыт `live = (n0+3 < N)`, причём нежилая нить проходит ВСЕ барьеры
  // (проверка стоит ПОСЛЕ них), поэтому расширение ворот не заводит расхождения в блоке.
  const int nblk4 = (N + THREADS * 4 - 1) / (THREADS * 4);
  const bool vec4 = !vec_off && (N % 4 == 0) && nblk4 > 0;
  const dim3 grid4 = (ksplit > 1) ? dim3(nblk4, ksplit) : dim3(nblk4);
  // Узкая ширина столбцов для MT>=4: вдвое меньше регистров -> вдвое больше варпов на SM.
  // Сетка вдвое шире по N, чтение остаётся векторным (LDG.64).
  const int nblk2 = (N + THREADS * 2 - 1) / (THREADS * 2);
  const dim3 grid2 = (ksplit > 1) ? dim3(nblk2, ksplit) : dim3(nblk2);
  static const int vw_force = getenv("FA2SM70_W8_VW") ? atoi(getenv("FA2SM70_W8_VW")) : 0;

// [ВЫБОР ШИРИНЫ ОБРАЩЕНИЯ] vec4 читает вес одной LDG.128 вместо четырёх LDG.32 -- см. шапку
// w8a16_gemv4_kernel. Применим, когда столбцов хватает на полную сетку по 4 на нить; иначе
// краевой блок вышел бы за N, а обрезать его значило бы завести молчаливую ветвь на границе.
// Откат: FA2SM70_W8_VEC=0 (нужен, чтобы A/B был возможен и после отгрузки).
#define W8A16_LAUNCH(MT)                                                                     \
  do {                                                                                       \
    /* [ШИРИНА СТОЛБЦОВ ПРИ MT=4 -- ПО ФОРМЕ, А НЕ ОДНА НА ВСЕХ]                            \
       Здесь стояло безусловное VW=2 при MT>=4 ("вдвое меньше регистров -> вдвое больше       \
       варпов"). Замер 29.08 (чистая карта, M=4, мкс VW=2 / VW=4):                            \
         qkv     5120x7168   101.7 /  74.9   -> VW=4 в 1.36                                   \
         gdn_in  5120x8192   105.9 /  79.5   -> VW=4 в 1.33                                   \
         down   17408x5120   241.8 / 157.7   -> VW=4 в 1.53  (КПД крыши 46 % -> 71 %)        \
         out_proj 3072x5120   42.1 /  43.5   -> VW=2                                          \
         gate+up 5120x17408  204.5 / 224.5   -> VW=2                                          \
       Выигрыш там, где столбцов НЕ слишком много (N<=8192) и вход широк (K>=5120): там       \
       занятость и так достаточна, и вдвое более широкое обращение окупается. Это ровно       \
       зона M=k+1 спекулятивного декода, то есть боевой режим. При MT=8 VW=2 лучше везде.     \
       Рычаг FA2SM70_W8_VW=2|4 перекрывает правило для A/B. */                                \
    const bool _vw4_ok = (MT == 4) && (N <= 8192) && (K >= 5120);                             \
    constexpr int VWD_UZ = (MT >= 4) ? 2 : 4;                                                 \
    const int VWD = _vw4_ok ? 4 : VWD_UZ;                                                     \
    const int VWX = vw_force ? vw_force : VWD;                                                \
    if (vec4 && VWX == 2) {                                                                   \
      if (ksplit > 1)                                                                         \
        w8a16_gemv4_kernel<THREADS, MT, true, 2><<<grid2, THREADS, 0, stream>>>(              \
            xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, nullptr, pp, M, N, K, G, gps, m0, zbias); \
      else                                                                                    \
        w8a16_gemv4_kernel<THREADS, MT, false, 2><<<grid2, THREADS, 0, stream>>>(             \
            xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, yp, nullptr, M, N, K, G, 0, m0, zbias);  \
    } else if (vec4) {                                                                        \
      if (ksplit > 1)                                                                        \
        w8a16_gemv4_kernel<THREADS, MT, true><<<grid4, THREADS, 0, stream>>>(                \
            xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, nullptr, pp, M, N, K, G, gps, m0, zbias); \
      else                                                                                   \
        w8a16_gemv4_kernel<THREADS, MT, false><<<grid4, THREADS, 0, stream>>>(               \
            xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, yp, nullptr, M, N, K, G, 0, m0, zbias);  \
    } else if (ksplit > 1)                                                                   \
      w8a16_gemv_kernel<THREADS, MT, true><<<grid, THREADS, 0, stream>>>(                     \
          xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, nullptr, pp, M, N, K, G, gps, m0, zbias);  \
    else                                                                                     \
      w8a16_gemv_kernel<THREADS, MT, false><<<grid, THREADS, 0, stream>>>(                    \
          xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, yp, nullptr, M, N, K, G, 0, m0, zbias);    \
  } while (0)

  // Chunk M by the largest tile that fits. Each chunk costs one full pass over the weight, so the
  // packed read only pays while the batch fits in a few chunks -- above that a dense tensor-core GEMM
  // is the right kernel and the caller should use it (see w8a16_prefers_packed).
  for (int m0 = 0; m0 < M; ) {
    const int rem = M - m0;
    if      (rem >= 16) { W8A16_LAUNCH(16); m0 += 16; }
    else if (rem >=  8) { W8A16_LAUNCH(8);  m0 +=  8; }
    else if (rem >=  4) { W8A16_LAUNCH(4);  m0 +=  4; }
    else if (rem >=  2) { W8A16_LAUNCH(2);  m0 +=  2; }
    else                { W8A16_LAUNCH(1);  m0 +=  1; }
  }
#undef W8A16_LAUNCH
  if (ksplit > 1) {
    const long tot = (long)M * N;
    w8a16_reduce_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(pp, yp, tot, ksplit);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Where the packed path wins, MEASURED on Huihui-Qwen3.6-27B-INT8 (V100, clocks locked):
//   M    down_proj 17408x5120        gate_proj 5120x17408
//    1   133 us vs dense 348  2.62x  140 us vs dense 399  2.84x
//    4   219 us vs dense 226  1.03x  242 us vs dense 225  0.93x
//   16   485 us vs dense 236  0.49x  491 us vs dense 231  0.47x
//   64  1926 us vs dense 279  0.14x 1951 us vs dense 238  0.12x
// The crossover is physical, not a defect. Below it the GEMM is bound by the weight READ and halving
// the bytes wins; above it cublas is on the tensor cores at 93.6 TFLOPS while this kernel is on the
// fp32 SIMT pipe at 15.7 -- at M=16 the FFMA count alone is a 182 us floor. Lifting the crossover
// means feeding the unpacked bytes to HMMA, which is a different (GEMM) kernel, not a tuning knob.
bool w8a16_prefers_packed(int64_t M) { return M <= 4; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_gemv", &w8a16_gemv, "fused GPTQ-int8 W8A16 GEMV (weights consumed packed, sm_70)",
        pybind11::arg("x"), pybind11::arg("qweight"), pybind11::arg("qzeros"),
        pybind11::arg("scales"), pybind11::arg("group_size"), pybind11::arg("v2_zero"));
  m.def("w8a16_gemm_hmma", &w8a16_gemm_hmma, "W8A16 через тензорные ядра (M>=8)",
        pybind11::arg("x"), pybind11::arg("qweight"), pybind11::arg("qzeros"),
        pybind11::arg("scales"), pybind11::arg("group_size"), pybind11::arg("v2_zero"));
  m.def("w8a16_prefers_packed", &w8a16_prefers_packed, "measured crossover vs a dense fp16 GEMM");
}
