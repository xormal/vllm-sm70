// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// РЕКУРРЕНТНЫЙ ШАГ GATED-DELTA-NET ДЛЯ ДЕКОДА (задача 180).
// Замена fused_recurrent_gated_delta_rule_fwd_kernel из fla/Triton (fused_recurrent.py).
//
// ЧТО СЧИТАЕТСЯ (на токен, семантика снята с Triton-ядра дословно, порядок операций тот же):
//     h *= e^g
//     s[v]   = SUM_k h[v][k] * k[k]
//     b_v[v] = (v[v] - s[v]) * beta
//     h[v][k] += b_v[v] * k[k]
//     o[v]   = SUM_k h[v][k] * q[k]
// Всё в fp32; вход q/k/v fp16 (или fp32-скаляры g), выход o fp16, состояние fp32 в пуле.
//
// ПОЧЕМУ СВОЁ ЯДРО, ЕСЛИ РАБОТА ТРИВИАЛЬНА. Ограничитель шага -- НЕ счёт (1.6 MFLOP на голову),
// а ПРОХОД ПО СОСТОЯНИЮ: [HV, V, K] fp32 = 1.5 МБ на слой читается И пишется каждый токен.
// Замерено (ведомость декода): мешалка держит 12.4% цепи за 1.2% байт -- чужое ядро не добирает
// полосу. Здесь состояние СРАЗУ живёт в регистрах (32 fp32 на нить), HBM трогается ровно дважды
// (одно чтение + одна запись), и обе стороны -- сплошными float4.
//
// РАСКЛАДКА. Блок = (последовательность, голова значения, срез BV строк v). 128 нитей:
// варп w владеет строками v = w, w+4, w+8, ... (шаг 4 -- варпов 4); полоса lane владеет
// столбцами k = 4*lane .. 4*lane+3 (K=128 = 32 полосы x 4). Свёртки по k -- варповый shuffle
// (полное дерево, 5 шагов), никакой разделяемой для редукций не нужно.
// При BV=32 и боевой форме (HV=24) сетка = N * 24 * (V/BV=4) = 96 блоков на N=1 -- машина
// занята и на одиночном декоде (у Triton это лечится только дроблением BV с теми же волнами).
//
// CONTINUOUS BATCHING / СПЕКУЛЯЦИЯ -- по контракту Triton-ядра:
//   * ssm_state_indices [N, stride]: начальное состояние берётся по индексу токена
//     (при спекуляции -- num_accepted_tokens[n]-1), запись -- по индексу i_t;
//   * индекс < 0 (PAD_SLOT_ID) -- блок не делает ничего / не пишет;
//   * при inplace-финале запись идёт в ТОТ ЖЕ пул состояний (h0 == ht).
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace fa2_sm70 {
namespace gdn {

// BV -- строк v на блок; GQ = HV/H -- отображение голов значения на головы q/k.
// L2N -- нормировка q,k внутри ядра (как у Triton при use_qk_l2norm_in_kernel).
// SPEC -- присутствует num_accepted_tokens (спекулятивный декод).
// TI -- тип входов q/k/v/beta и выхода o (__half или __nv_bfloat16: боевой декод идёт в bf16);
// IdxT -- тип ssm_state_indices (int32 у vLLM, int64 у образца).
template <typename TI> __device__ __forceinline__ float rec_ldf(const TI* p);
template <> __device__ __forceinline__ float rec_ldf<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ __forceinline__ float rec_ldf<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename TI> __device__ __forceinline__ TI rec_stf(float x);
template <> __device__ __forceinline__ __half rec_stf<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 rec_stf<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ------------------------------------------------------------------------------------------------
// СОСТОЯНИЕ В int16 С МАСШТАБОМ (ход владельца 20.08).
//
// ЗАЧЕМ. Состояние GDN рекуррентно: обновляется КАЖДЫМ токеном, и ошибка не рассасывается, а
// копится. Замер снятием фазы: тот же промпт в жадном режиме при состоянии fp16 вырождается
// (доля повторов 0.94 к середине текста, уход в чужой язык), при fp32 -- 0.00 и осмысленный
// конец. То есть ёмкость 262K, купленная за fp16, оплачена КАЧЕСТВОМ.
//
// РАЗМЕН СНИМАЕТСЯ ФОРМАТОМ: int16 занимает те же 2 байта (полоса и ёмкость не меняются), но
// сетка РАВНОМЕРНАЯ -- 15 бит на величину против 11 бит мантиссы у fp16, то есть шаг ошибки
// в 16 раз мельче на всём диапазоне. Цена -- умножение на масштаб при чтении и деление при
// записи; для GDN это допустимо (тело памяти-ограничено, счётный канал недогружен).
//
// ГДЕ ЖИВЁТ МАСШТАБ. Один на (слот, голову, срез BV) -- ровно на то, что обрабатывает ОДИН
// блок, поэтому редукция amax внутриблочная (варповый shuffle + 4 слова в разделяемой), без
// межблочных барьеров. Таблица копеечная: слоты x HV x (V/BV) x 4 байта.
struct СостI16 { using Тип = short; };

__device__ __forceinline__ float rec_i16_ld(short q, float s) { return (float)q * s; }
// НАСЫЩЕНИЯ ЗДЕСЬ НЕТ, И ЭТО НЕ НЕДОСМОТР. Масштаб берётся как a/32766 по amax того же
// варпа, значит |x|*inv_s <= 32766 по ПОСТРОЕНИЮ, а запас до 32767 покрывает округление
// деления. Пара fminf/fmaxf стоила 64 FMNMX из 101 в горячем цикле (разбор SASS 20.08) --
// это была самая дорогая строка квантования, и она сторожила невозможное.
__device__ __forceinline__ short rec_i16_st(float x, float inv_s) {
  return (short)__float2int_rn(x * inv_s);
}

// GATE=1 -- СЛИТЫЙ СИГМОИДНЫЙ ГЕЙТИНГ (замена fused_sigmoid_gating_delta_rule_update --
// боевой быстрый путь декода qwen3_next/_forward_core). Вместо готовых g/beta подаются
// a,b (активации, тип TI) и A_log,dt_bias (параметры, fp32); в ядре, дословно по Triton:
//     x = a + dt_bias;  g = -exp(A_log) * softplus(x);  beta = sigmoid(b)
// softplus с порогом: x <= 20 ? log1p(exp(x)) : x (beta_softplus=1 -- боевой дефолт).
// ЗАНЯТОСТЬ ПРОТИВ РЕГИСТРОВ -- ЗАМЕРЕНО 20.08, БЫЛО 8, СТАЛО 4.
// Требование 8 блоков на SM режет бюджет до 65536/(128*8) = 64 регистров на нить, и ptxas
// разливал: боевой инстанс нёс 13 LDL + 13 STL при пике живых 58 из 255. Укладчик (задача
// 104) на этом теле показал T1=2743 при доказанном оптимуме 913 -- запас x3.00, и главным
// классом сдвига шёл именно разлив. Замер на боевой форме (H=8, HV=24 на карту при TP=2):
//
//   мин.блоков |  B=8   B=16   B=32  (мкс, состояние fp16) | разлив
//        8     | 36.3   75.2  148.7                        | LDL 13, STL 13
//        4     | 18.0   42.8   79.2                        | НЕТ
//        2     | 19.7   43.2   80.1                        | НЕТ
//        1     | 19.7   43.9   80.1                        | НЕТ
//
// 4 -- вершина размена: разлива уже нет, а занятости ещё вдвое больше, чем у 2 (на B=8
// разница 18.0 против 19.7). Ниже 4 опускаться нечем -- регистров и так хватает.
#ifndef FA2SM70_REC_MINBLK
#define FA2SM70_REC_MINBLK 4
#endif
template <int K, int V, int BV, bool L2N, bool SPEC, typename TI, typename IdxT, int GATE = 0, typename StateT = float>
__global__ __launch_bounds__(128, FA2SM70_REC_MINBLK) void volta_gdn_rec(
    const TI* __restrict__ q,          // [B, T, H, K]
    const TI* __restrict__ k,          // [B, T, H, K]
    const TI* __restrict__ v,          // [B, T, HV, V]
    const float*  __restrict__ g,      // [B, T, HV] (GATE=0) либо не читается (GATE=1)
    const TI* __restrict__ beta,       // [B, T, HV] (GATE=0) либо b [B, T, HV] (GATE=1)
    TI* __restrict__ o,                // [B, T, HV, V]
    StateT* __restrict__ hpool,        // пул состояний (fp32 ИЛИ fp16 -- StateT): [slots, HV, V, K] (h0 и ht -- один пул)
    const IdxT* __restrict__ ssm_idx,  // [N, stride_idx] или nullptr
    const int*  __restrict__ n_acc,    // [N] или nullptr (SPEC)
    const IdxT* __restrict__ cu_seqlens,   // [N+1] или nullptr (тот же целочисленный тип)
    long stride_idx, long stride_slot,     // шаги таблицы индексов и пула (в элементах)
    float scale, int T, int H, int HV,
    const TI* __restrict__ a_in = nullptr,       // GATE=1: a [B, T, HV]
    const float* __restrict__ A_log = nullptr,   // GATE=1: [HV]
    const float* __restrict__ dt_bias = nullptr,    // GATE=1: [HV]
    // РЕЖИМ 'all' (задача 194): состояние ЧИТАЕТСЯ по ssm_idx (блок последнего посчитанного
    // токена), а ПИШЕТСЯ по ssm_idx_out (блок последнего запланированного). На шаге, где
    // токен переходит границу блока, это РАЗНЫЕ блоки. Раньше здесь стоял один индекс, и
    // ради этого декод в режиме 'all' уходил на чужой Triton -- то есть на путь, который мы
    // же и вытесняем. Одномерный: [N], без шага.
    const IdxT* __restrict__ ssm_idx_out = nullptr,
    // МАСШТАБЫ СОСТОЯНИЯ при StateT=short: [slots, HV, V/BV], fp32. Читает и ОБНОВЛЯЕТ
    // их тот же блок, который владеет срезом, -- межблочных зависимостей нет.
    float* __restrict__ scl = nullptr,
    // ШИРИНА РЯДА таблицы слотов (число слотов на последовательность). Ниже по тексту она
    // ограничивает чтение ssm_idx[i_n*stride_idx + it]: ряд короче числа токенов означает,
    // что на эти токены слотов НЕ ВЫДЕЛЕНО, и писать состояние некуда. Без этого предела
    // ядро читало соседний ряд (а на последней последовательности -- за конец тензора) и
    // писало состояние ПО ЧУЖОМУ ИНДЕКСУ: memcheck 20.08 поймал это как illegal access,
    // всплывавший на ДРУГОМ вызове. 0 -- предела нет (прежнее поведение).
    int idx_w = 0,
    // [ДЕРЕВО, 05.09] СТРОКА ВЕТВЛЕНИЯ. Раскладка дерева: [якорь, a0..a(W-1), b0..b(W-1)],
    // и рекуррент цепью протаскивал в строки ветви B историю ветви A -- порча по
    // построению, маской не лечится (маска -- у внимания, у рекуррента её нет).
    // branch_at = W+1 велит строке b0 продолжать НЕ предыдущую строку, а ЯКОРЬ: его
    // состояние этот же варп записал в слот колонки 0 на it=0, адреса нитей совпадают,
    // поэтому чтение собственной записи корректно без единого барьера. Масштаб int16
    // якоря запоминается в регистре на it=0 -- чтение scl из памяти не нужно.
    // 0 -- ветвления нет, путь прежний бит в бит.
    int branch_at = 0) {
  static_assert(K % 128 == 0 && K / 128 == 1, "раскладка написана на K=128");
  constexpr bool сост_целое = sizeof(StateT) == 2 && !std::is_same<StateT, __half>::value;
  constexpr int RV = BV / 4;                 // строк v на варп
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  const int i_v = blockIdx.x;                // срез BV
  const int i_hv = blockIdx.y;               // голова значения
  const int i_n = blockIdx.z;                // последовательность
  const int i_h = i_hv / (HV / H);           // голова q/k

  long bos = (long)i_n * T;
  int Tn = T;
  if (cu_seqlens != nullptr) {
    bos = (long)cu_seqlens[i_n];
    Tn = (int)((long)cu_seqlens[i_n + 1] - bos);
  }
  if (Tn <= 0) return;

  // --- начальное состояние: слот по индексу токена (спекуляция: последний принятый).
  float hreg[RV][4];
#pragma unroll
  for (int j = 0; j < RV; ++j)
#pragma unroll
    for (int c = 0; c < 4; ++c) hreg[j][c] = 0.f;
  if (hpool != nullptr) {
    long slot;
    if (ssm_idx != nullptr) {
      const int it0 = SPEC ? (n_acc[i_n] - 1) : 0;
      slot = ssm_idx[(long)i_n * stride_idx + it0];
      if (slot < 0) return;                              // PAD_SLOT_ID: блоку нечего делать
    } else {
      slot = bos;                                        // без таблицы: слот = начало посл-ти
    }
    const StateT* p_h0 = hpool + slot * stride_slot + ((long)i_hv * V + i_v * BV) * K + 4 * lane;
    // масштаб среза: один на (слот, голова, срез BV); при fp16/fp32 не читается вовсе
    // МАСШТАБ НА ВАРП, а не на блок. Комбинатор (задача 104) на первой версии показал
    // T1=2255 при оптимуме 1058, и главными классами сдвига шли БАРЬЕРЫ (BAR/BSYNC, сдвиг
    // -547..-657) -- то есть блочная редукция amax через разделяемую. Варп владеет своими
    // строками v (warp, warp+4, ...) целиком, поэтому масштаб на варп делает редукцию
    // ЧИСТО ВАРПОВОЙ: пять shfl, ни одного __syncthreads и ни байта разделяемой.
    const float s_rd = (sizeof(StateT) == 2 && scl != nullptr)
                           ? scl[((slot * (long)HV + i_hv) * (V / BV) + i_v) * 4 + warp] : 1.f;
#pragma unroll
    for (int j = 0; j < RV; ++j) {
      const StateT* адр = &p_h0[(long)(warp + 4 * j) * K];
      if constexpr (сост_целое) {
        const short4 q = *reinterpret_cast<const short4*>(адр);
        hreg[j][0] = rec_i16_ld(q.x, s_rd); hreg[j][1] = rec_i16_ld(q.y, s_rd);
        hreg[j][2] = rec_i16_ld(q.z, s_rd); hreg[j][3] = rec_i16_ld(q.w, s_rd);
      } else if constexpr (sizeof(StateT) == 4) {
        const float4 f = *reinterpret_cast<const float4*>(адр);
        hreg[j][0] = f.x; hreg[j][1] = f.y; hreg[j][2] = f.z; hreg[j][3] = f.w;
      } else {
        // 4 значения половинной точности -- ОДНА вставка на 8 байт (не четыре по два)
        const float2 упак = *reinterpret_cast<const float2*>(адр);
        const __half2 a0 = *reinterpret_cast<const __half2*>(&упак.x);
        const __half2 a1 = *reinterpret_cast<const __half2*>(&упак.y);
        hreg[j][0] = __low2float(a0);  hreg[j][1] = __high2float(a0);
        hreg[j][2] = __low2float(a1);  hreg[j][3] = __high2float(a1);
      }
    }
  }

  const TI* p_q = q + (bos * H + i_h) * K + 4 * lane;
  const TI* p_k = k + (bos * H + i_h) * K + 4 * lane;
  const TI* p_v = v + (bos * HV + i_hv) * V + i_v * BV;
  const float*  p_g = g + bos * HV + i_hv;
  const TI* p_b = beta + bos * HV + i_hv;
  const TI* p_a = (GATE && a_in != nullptr) ? a_in + bos * HV + i_hv : nullptr;
  TI* p_o = o + (bos * HV + i_hv) * V + i_v * BV;

  float s_якоря = 1.f;                     // масштаб слота колонки 0 (только int16-пул)
  for (int it = 0; it < Tn; ++it) {
    if (branch_at > 0 && it == branch_at && ssm_idx != nullptr && hpool != nullptr) {
      const long slot0 = ssm_idx[(long)i_n * stride_idx + 0];
      if (slot0 >= 0) {
        const StateT* p_h0 = hpool + slot0 * stride_slot + ((long)i_hv * V + i_v * BV) * K + 4 * lane;
#pragma unroll
        for (int j = 0; j < RV; ++j) {
          const StateT* адр = &p_h0[(long)(warp + 4 * j) * K];
          if constexpr (сост_целое) {
            const short4 qq = *reinterpret_cast<const short4*>(адр);
            hreg[j][0] = rec_i16_ld(qq.x, s_якоря); hreg[j][1] = rec_i16_ld(qq.y, s_якоря);
            hreg[j][2] = rec_i16_ld(qq.z, s_якоря); hreg[j][3] = rec_i16_ld(qq.w, s_якоря);
          } else if constexpr (sizeof(StateT) == 4) {
            const float4 f = *reinterpret_cast<const float4*>(адр);
            hreg[j][0] = f.x; hreg[j][1] = f.y; hreg[j][2] = f.z; hreg[j][3] = f.w;
          } else {
            const float2 упак = *reinterpret_cast<const float2*>(адр);
            const __half2 a0 = *reinterpret_cast<const __half2*>(&упак.x);
            const __half2 a1 = *reinterpret_cast<const __half2*>(&упак.y);
            hreg[j][0] = __low2float(a0);  hreg[j][1] = __high2float(a0);
            hreg[j][2] = __low2float(a1);  hreg[j][3] = __high2float(a1);
          }
        }
      }
    }
    // --- фрагменты q,k этой полосы (4 элемента = одна LDG.64) + скаляры головы
    float qf[4], kf[4];
    {
      const uint2 rq = *reinterpret_cast<const uint2*>(p_q);
      const uint2 rk = *reinterpret_cast<const uint2*>(p_k);
      const TI* hq = reinterpret_cast<const TI*>(&rq);
      const TI* hk = reinterpret_cast<const TI*>(&rk);
#pragma unroll
      for (int m = 0; m < 4; ++m) { qf[m] = rec_ldf<TI>(&hq[m]); kf[m] = rec_ldf<TI>(&hk[m]); }
    }
    if constexpr (L2N) {
      // нормировка по ВСЕМУ вектору K: частичные суммы по полосе + дерево shuffle.
      float sq = qf[0] * qf[0] + qf[1] * qf[1] + qf[2] * qf[2] + qf[3] * qf[3];
      float sk = kf[0] * kf[0] + kf[1] * kf[1] + kf[2] * kf[2] + kf[3] * kf[3];
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, d, 32);
        sk += __shfl_xor_sync(0xffffffffu, sk, d, 32);
      }
      const float rq = rsqrtf(sq + 1e-6f), rk = rsqrtf(sk + 1e-6f);
#pragma unroll
      for (int c = 0; c < 4; ++c) { qf[c] *= rq; kf[c] *= rk; }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) qf[c] *= scale;
    float eg, bb;
    if constexpr (GATE) {
      const float x = rec_ldf<TI>(p_a) + dt_bias[i_hv];
      const float sp = (x <= 20.f) ? log1pf(__expf(x)) : x;
      eg = __expf(-__expf(A_log[i_hv]) * sp);
      bb = 1.f / (1.f + __expf(-rec_ldf<TI>(p_b)));
    } else {
      eg = __expf(p_g[0]);
      bb = rec_ldf<TI>(p_b);
    }

    // --- шаг по RV строкам варпа
#pragma unroll
    for (int j = 0; j < RV; ++j) {
      const int vrow = warp + 4 * j;                       // строка внутри среза BV
      // s = SUM_k h*e^g * k  (гейт применяется до свёртки -- как у Triton: b_h *= exp(g) раньше)
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        hreg[j][c] *= eg;
        s += hreg[j][c] * kf[c];
      }
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d, 32);
      const float bv = (rec_ldf<TI>(&p_v[vrow]) - s) * bb;
      float po = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        hreg[j][c] += bv * kf[c];
        po += hreg[j][c] * qf[c];
      }
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) po += __shfl_xor_sync(0xffffffffu, po, d, 32);
      if (lane == 0) p_o[vrow] = rec_stf<TI>(po);
    }

    // --- запись состояния: по слоту токена it (continuous batching) либо по позиции
    {
      long slot = bos + it;
      bool пиши = true;
      if (ssm_idx_out != nullptr) {
        slot = ssm_idx_out[(long)i_n];   // куда класть (режим 'all')
        пиши = (slot >= 0);
      } else if (ssm_idx != nullptr) {
        if (idx_w > 0 && it >= idx_w) {
          пиши = false;                    // слота на этот токен не выделено -- не пишем
        } else {
          slot = ssm_idx[(long)i_n * stride_idx + it];
          пиши = (slot >= 0);
        }
      }
      if (пиши) {
        StateT* p_ht = hpool + slot * stride_slot + ((long)i_hv * V + i_v * BV) * K + 4 * lane;
        float inv_s = 1.f;
        if constexpr (сост_целое) {
          // amax ПО СТРОКАМ ЭТОГО ВАРПА: по нити, затем дерево shfl. Барьеров нет.
          float a = 0.f;
#pragma unroll
          for (int j = 0; j < RV; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) a = fmaxf(a, fabsf(hreg[j][c]));
#pragma unroll
          for (int d = 16; d > 0; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, d, 32));
          const float s_new = (a > 0.f) ? (a / 32766.f) : 1.f;
          inv_s = (a > 0.f) ? (32766.f / a) : 0.f;
          if (branch_at > 0 && it == 0) s_якоря = s_new;   // якорь: масштаб для строки ветвления
          if (lane == 0 && scl != nullptr)
            scl[((slot * (long)HV + i_hv) * (V / BV) + i_v) * 4 + warp] = s_new;
        }
#pragma unroll
        for (int j = 0; j < RV; ++j) {
          StateT* адр = &p_ht[(long)(warp + 4 * j) * K];
          if constexpr (сост_целое) {
            short4 qv;
            qv.x = rec_i16_st(hreg[j][0], inv_s); qv.y = rec_i16_st(hreg[j][1], inv_s);
            qv.z = rec_i16_st(hreg[j][2], inv_s); qv.w = rec_i16_st(hreg[j][3], inv_s);
            *reinterpret_cast<short4*>(адр) = qv;
          } else if constexpr (sizeof(StateT) == 4) {
            *reinterpret_cast<float4*>(адр) =
                make_float4(hreg[j][0], hreg[j][1], hreg[j][2], hreg[j][3]);
          } else {
            const __half2 a0 = __floats2half2_rn(hreg[j][0], hreg[j][1]);
            const __half2 a1 = __floats2half2_rn(hreg[j][2], hreg[j][3]);
            float2 упак;
            *reinterpret_cast<__half2*>(&упак.x) = a0;
            *reinterpret_cast<__half2*>(&упак.y) = a1;
            *reinterpret_cast<float2*>(адр) = упак;
          }
        }
      }
    }

    p_q += (long)H * K; p_k += (long)H * K;
    p_v += (long)HV * V; p_o += (long)HV * V;
    p_g += HV; p_b += HV;
    if constexpr (GATE) p_a += HV;
  }
}

}  // namespace gdn
}  // namespace fa2_sm70
