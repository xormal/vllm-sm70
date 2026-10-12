// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Volta (sm_70) FMHA FORWARD, int8 KV -- РАЗВЯЗКА РОЛЕЙ S/O (записка 26 §3, 12.09.2026).
//
// ЗАЧЕМ. Карта потолка боевого ядра volta_fwd_ws<32,64,256,2,4> снята фальсификаторами: софтмакс 18 %
// времени при простое тензорной трубы, чтения Q 10 %, все чтения операндов 28 %. В ws все восемь
// считающих варпов идут в ОДНОЙ фазе: QK -> софтмакс -> PV, и пока идёт софтмакс, HMMA не выдаётся.
// Здесь считающие разведены на ДВЕ РОЛИ с ОДНИМ рандеву на плитку:
//   везущие (FA2_SOPROD варпов): K(t+1) -> K[(t+1)&1], V(t) -> V[t&1], масштабы (t+1);
//   S-варпы (4): Q РЕЗИДЕНТНО В РЕГИСТРАХ (16 строк x 256 = 128 рег), QK(t) по K[t&1], софтмакс,
//                P(t) -> P[t&1] вместе с corr и частичными Σ P*vs;
//   O-варпы (4): PV(t-1) по P[(t-1)&1] и V[(t-1)&1], накопитель O 32 строки x 64 столбца (64 рег).
// Шаг t: S(t) || O(t-1) || подача -- софтмакс S-варпов перекрывается HMMA O-варпов на тех же SMSP.
// Плитка S-варпа 16x32 (пара варпов делит строки, обмен максимума строки через разделяемую + именованный
// барьер пары); l и Σ P*vs ведутся ЧАСТИЧНО по половинам ключей и складываются в эпилоге / у O-варпов.
//
// ВЕРСИЯ 2 (12.09 23:40): ВЕЗУЩИХ НЕТ -- подачу K(t+1)/V(t) несут O-ВАРПЫ (у них запас: 128 HMMA на плитку
// против 128 HMMA + софтмакс у S-варпов). Причина: версия 1 с 2 везущими (10 варпов, потолок 204 регистра)
// лила в локальную память 254 операции ВНУТРИ главного цикла (qf 128 + o.acc 64 живут вместе) и шла в 2 раза
// медленнее базы. 8 варпов -> потолок 256 регистров; на каждом SMSP ровно один S- и один O-варп.
// ПРОТОТИП: плотный непостраничный путь, KVFMT=0 (int8 + масштаб на позицию), причинность, REV.
// PAGED/selIdx/kSplit -- после замера (адресация подачи та же, что в ws).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>
#include "gemm/volta_warp_mma.h"
#include "gemm/volta_prims.h"


#ifndef FA2_SOPROD
#define FA2_SOPROD 0      /* >0 = столько ОТДЕЛЬНЫХ везущих варпов; 0 = везут O-варпы */
#endif
#ifndef FA2_SO_KSP
#define FA2_SO_KSP 1      /* 2 = две цепочки накопителя QK -- замерено: ноль */
#endif
#ifndef FA2_SO_COPYLATE
#define FA2_SO_COPYLATE 0 /* 1 = подача K/V ПОСЛЕ PV (LDG-задержка за HMMA) */
#endif

#ifndef FA2_SO_ROLLCP
#define FA2_SO_ROLLCP 0   /* 1 = цикл подачи не развёрнут (код меньше) */
#endif
#ifndef FA2_SO_QSMEM
#define FA2_SO_QSMEM 0    /* 1 = Q остаётся в разделяемой, цикл QK свёрнут (регистры свободны, код мал) */
#endif
#ifndef FA2_SO_QHALF
#define FA2_SO_QHALF 0    /* 1 = половина Q (d<128) в регистрах, вторая -- из разделяемой свёрнутым циклом */
#endif
#ifndef FA2_SO_PREF
#define FA2_SO_PREF 0     /* 1 = prefetch.global.L1 строк K(t+2)/V(t+1) в начале шага подачи */
#endif
#ifndef FA2_SO_PHASE
#define FA2_SO_PHASE 0    /* 1 = счётчики тактов по фазам ролей в g_so_phase (прибор, не боевой путь) */
#endif

namespace fa2_sm70 {
namespace fwd_so {
#if FA2_SO_PHASE
__device__ unsigned long long g_so_phase[8];   // S: 0 QK, 1 барьер пары, 2 софтмакс+P, 3 рандеву; O: 4 подача, 5 PV, 6 рандеву
#define SOPH_DECL long long _pht = clock64(); unsigned long long _pha[8] = {0,0,0,0,0,0,0,0};
#define SOPH(i) do { long long _n = clock64(); _pha[i] += (unsigned long long)(_n - _pht); _pht = _n; } while (0)
#define SOPH_OUT() do { if (lane == 0) for (int _i = 0; _i < 8; ++_i) atomicAdd(&g_so_phase[_i], _pha[_i]); } while (0)
#else
#define SOPH_DECL
#define SOPH(i)
#define SOPH_OUT()
#endif

using fa2_sm70::WarpMma;
using fa2_sm70::BLayout;
using fa2_sm70::prims::seed_rank1;
using fa2_sm70::prims::rescale_fold;

template <int BQ, int BK, int D, bool CAUSAL, int REV = 1, int KVFMT = 0, int PAGED = 0>
__global__ __launch_bounds__((8 + FA2_SOPROD) * 32, 1) void volta_fwd_so(
    const __half* __restrict__ Q,
    const int8_t* __restrict__ Kq, const float* __restrict__ Ks,
    const int8_t* __restrict__ Vq, const float* __restrict__ Vs,
    __half* __restrict__ O, float* __restrict__ Lse,
    int Sq, int Sk, float qk_scale, int kv_ratio, int causal_off,
    long q_sM, long q_sH, long q_sB, long k_sM, long k_sH, long k_sB,
    long o_sM, long o_sH, long o_sB, long lse_sH, long lse_sB, long ks_sH, long ks_sB,
    // хвост доводов -- как у volta_fwd_ws (страницы пула, отбор плиток, расщепление по ключам)
    const long* __restrict__ pageOff = nullptr, const long* __restrict__ pageScl = nullptr, int pageHkv = 0,
    const int* __restrict__ selIdx = nullptr, const int* __restrict__ selCnt = nullptr,
    int selStride = 0, int selGroup = 1, int selSub = 0,
    int kSplit = 1, float* __restrict__ Opart = nullptr, float* __restrict__ LSEpart = nullptr) {
  static_assert(BQ == 32 && BK == 64 && D == 256 && KVFMT == 0, "прототип закреплён на боевой форме");
  constexpr int kS = 4, kO = 4, kP = FA2_SOPROD;
  constexpr int kOThr = kO * 32, kAllThr = (kS + kO + kP) * 32;
  constexpr int kCThr = (kP > 0) ? kP * 32 : kOThr;     // нити подачи: везущие либо O-варпы
  constexpr int PAD = 4, LDQ = D + PAD, LDP = BK + PAD, LDK8 = D + 16, LDV = D + 16;
  static_assert(LDK8 % 16 == 0, "байтовый шаг K кратен 16 под LDS.128");
  constexpr int SM = 16, SN = 32;                 // плитка S-варпа
  constexpr int OM = 32, ON = 64;                 // плитка O-варпа
  constexpr int SMB = SM / 16, SNB = SN / 16, OMB = OM / 16, ONB = ON / 16;
  constexpr float kI8Bias = 1152.0f, kFmtMul = 1.0f;
  const float qkm = qk_scale;

  extern __shared__ __half smem_so[];
  __half* sQ = smem_so;                                        // [BQ][LDQ] -- до предзагрузки Q в регистры
#if FA2_SO_QSMEM || FA2_SO_QHALF
  __half* sP = smem_so + BQ * LDQ;                             // Q живёт всё ядро; P[2] отдельно
  int8_t* sBuf = reinterpret_cast<int8_t*>(sP + 2 * BQ * LDP); // [2][K|V]
#else
  __half* sP = smem_so;                                        // затем [2][BQ][LDP] -- P по буферам
  static_assert(2 * BQ * LDP <= BQ * LDQ, "P[2] обязан поместиться в область Q");
  int8_t* sBuf = reinterpret_cast<int8_t*>(smem_so + BQ * LDQ); // [2][K|V]
#endif
  constexpr int kBuf = BK * (LDK8 + LDV);
  __shared__ float sKs[2 * BK], sVs[2 * BK];
  __shared__ float sQsum[BQ];
  __shared__ float sRedM[2][BQ];        // [половина ключей][строка]: обмен максимума в паре S-варпов
  __shared__ float sCorr[2][BQ];        // [буфер P][строка]: corr2 плитки
  __shared__ float sVb[2][2][BQ];       // [буфер P][половина][строка]: Σ P*vs (fp16-округлённые)
  __shared__ float sLpart[2][BQ];       // [половина][строка]: частичные l в эпилоге
  __shared__ float sInv[BQ];            // 1/l на строку

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool isS = warp < kS;
  const bool isO = !isS && warp < kS + kO;
  const bool isC = (kP > 0) ? (warp >= kS + kO) : isO;     // роль подачи
  const int sw = warp;                          // S: 0..3
  const int ow = warp - kS;                     // O: 0..3
  const int ctid = (kP > 0) ? (tid - (kS + kO) * 32) : (tid - kS * 32);   // номер нити в подаче
  const int half = isS ? (sw >> 1) : 0;         // половина ключей S-варпа (0: 0..31, 1: 32..63)
  const int rs = isS ? ((sw & 1) * SM) : 0;     // строки S-варпа
  const int cs = half * SN;
  const int pairId = sw & 1;                    // пара S-варпов = общие строки
  const int q0 = (REV ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x) * BQ;
  long pgHead = 0;
  {
    const long h = blockIdx.y, hkv = h / kv_ratio;
    const long b = (kSplit > 1) ? ((long)blockIdx.z / kSplit) : (long)blockIdx.z;
    Q += b * q_sB + h * q_sH; O += b * o_sB + h * o_sH;
    if constexpr (!PAGED) {
      Kq += b * k_sB + hkv * k_sH; Vq += b * k_sB + hkv * k_sH;
      Ks += b * ks_sB + hkv * ks_sH; Vs += b * ks_sB + hkv * ks_sH;
    } else {
      pgHead = hkv;
    }
    Lse += b * lse_sB + h * lse_sH;
  }
  const int kEnd = CAUSAL ? (((q0 + BQ + causal_off) < Sk) ? (q0 + BQ + causal_off) : Sk) : Sk;
  // отбор плиток и расщепление по ключам -- те же правила, что в volta_fwd_ws
  const int* _selI = nullptr; int _selN = 0;
  if (selIdx != nullptr) {
    const int _st = q0 / (BQ * selGroup), _hk = (kv_ratio > 0) ? (blockIdx.y / kv_ratio) : 0;
    const int _hkv = (pageHkv > 0) ? pageHkv : 1;
    _selI = selIdx + ((size_t)_st * _hkv + _hk) * (size_t)selStride;
    _selN = selCnt[(size_t)_st * _hkv + _hk];
  }
  const int nTilesAll = (_selI != nullptr) ? (_selN << selSub) : ((kEnd + BK - 1) / BK);
  const int _spl  = (kSplit > 1) ? ((int)blockIdx.z % kSplit) : 0;
  const int _base = (kSplit > 1) ? (nTilesAll / kSplit) : nTilesAll;
  const int _rem  = (kSplit > 1) ? (nTilesAll % kSplit) : 0;
  const int tBeg  = (kSplit > 1) ? (_spl * _base + (_spl < _rem ? _spl : _rem)) : 0;
  const int tEnd  = (kSplit > 1) ? (tBeg + _base + (_spl < _rem ? 1 : 0)) : nTilesAll;
  const int nTiles = tEnd - tBeg;                      // плиток у ЭТОГО блока
  float* __restrict__ Op = nullptr;
  float* __restrict__ Lp = nullptr;
  if (Opart != nullptr) {
    const long nH = (long)gridDim.y;
    const long nB = (long)gridDim.z / (kSplit > 0 ? kSplit : 1);
    const long _hh = (long)blockIdx.y;
    const long _bb = (kSplit > 1) ? ((long)blockIdx.z / kSplit) : (long)blockIdx.z;
    const long bh = _bb * nH + _hh;
    Op = Opart  + (long)_spl * nB * nH * (long)Sq * D + bh * (long)Sq * D;
    Lp = LSEpart + (long)_spl * nB * nH * (long)Sq     + bh * (long)Sq;
  }
#define FA2SO_KB_OF(t) ((_selI != nullptr) \
    ? (((_selI[(t) >> selSub] << selSub) | ((t) & ((1 << selSub) - 1))) * BK) : ((t) * BK))

  // ---- пролог: Q в разделяемую (все нити), суммы строк, K(0) и масштабы (0) -- везущими ----
#pragma unroll
  for (int i = tid; i < BQ * D / 8; i += kAllThr) {
    const int r = i / (D / 8), c = (i % (D / 8)) * 8;
    uint4 g = make_uint4(0, 0, 0, 0);
    if (q0 + r < Sq) g = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * q_sM + c]);
    __half* dq = &sQ[r * LDQ + c];
    *reinterpret_cast<uint2*>(dq)     = make_uint2(g.x, g.y);
    *reinterpret_cast<uint2*>(dq + 4) = make_uint2(g.z, g.w);
  }
  __syncthreads();
  for (int r = warp; r < BQ; r += kS + kO + kP) {
    float acc = 0.f;
    const __half* pr = &sQ[r * LDQ];
    for (int d = lane * 4; d < D; d += 128) {
      const uint2 v = *reinterpret_cast<const uint2*>(pr + d);
      const __half2* h = reinterpret_cast<const __half2*>(&v);
      acc += __half2float(__low2half(h[0])) + __half2float(__high2half(h[0]))
           + __half2float(__low2half(h[1])) + __half2float(__high2half(h[1]));
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) sQsum[r] = acc;
  }
  // подача плитки: K(kb) в буфер bufK (если fillK), V(kb) в буфер bufV (если fillV), масштабы (kb) в bufS
  // подача плитки O-варпами (128 нитей), 16 Б на нить за проход: 8 строк по 256 Б за проход, 8 проходов на K и на V
  auto podacha = [&](int kb, int bufK, bool fillK, int bufV, bool fillV, int bufS, bool fillS) {
    static_assert((BK * D / 16) % kCThr == 0, "широкая перекладка делится на нити нацело");
    constexpr int kRowK = kCThr / (D / 16);
    const int r0 = ctid / (D / 16), c0 = (ctid % (D / 16)) * 16;
    const int rv = (kb + BK <= Sk) ? BK : ((Sk > kb) ? (Sk - kb) : 0);
    if constexpr (PAGED) {
      // страницы пула по 16 токенов: строка r плитки лежит в странице r>>4 со смещением (r&15)*pageHkv*D
      static_assert(kRowK == 8 && BK == 64, "PAGED: 8 строк за проход, 4 страницы на плитку -- индекс страницы статичен (u>>1)");
      long pgBase[BK / 16];
#pragma unroll
      for (int pi = 0; pi < BK / 16; ++pi) pgBase[pi] = (kb + pi * 16 < Sk) ? __ldg(pageOff + (kb >> 4) + pi) : 0;
      const long pgOff = (long)pgHead * D + c0, pgRow = (long)pageHkv * D;
      if (fillK) {
        unsigned qk = (unsigned)__cvta_generic_to_shared(sBuf + bufK * kBuf + r0 * LDK8 + c0);
#pragma unroll
        for (int u = 0; u < (BK * D / 16) / kCThr; ++u) {
          const int r = r0 + kRowK * u;
          uint4 w = make_uint4(0, 0, 0, 0);
          if (r < rv) w = *reinterpret_cast<const uint4*>(Kq + pgBase[u >> 1] + (long)(r & 15) * pgRow + pgOff);
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(qk), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w));
          qk += kRowK * LDK8;
        }
      }
      if (fillV) {
        unsigned qv = (unsigned)__cvta_generic_to_shared(sBuf + bufV * kBuf + BK * LDK8 + r0 * LDV + c0);
#pragma unroll
        for (int u = 0; u < (BK * D / 16) / kCThr; ++u) {
          const int r = r0 + kRowK * u;
          uint4 w = make_uint4(0, 0, 0, 0);
          if (r < rv) w = *reinterpret_cast<const uint4*>(Vq + pgBase[u >> 1] + (long)(r & 15) * pgRow + pgOff);
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(qv), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w));
          qv += kRowK * LDV;
        }
      }
      if (fillS) {
        for (int j = ctid; j < BK; j += kCThr) {
          const int p = kb + j;
          float ks = 0.f, vs = 0.f;
          if (p < Sk) {
            const long si = __ldg(pageScl + (p >> 4)) + (long)(p & 15) * pageHkv + pgHead;
            ks = Ks[si]; vs = Vs[si];
          }
          sKs[bufS * BK + j] = ks; sVs[bufS * BK + j] = vs;
        }
      }
      return;
    }
    if (fillK) {
      const int8_t* pk = Kq + (long)(kb + r0) * k_sM + c0;
      unsigned qk = (unsigned)__cvta_generic_to_shared(sBuf + bufK * kBuf + r0 * LDK8 + c0);
      int rr = r0;
#if FA2_SO_ROLLCP
#pragma unroll 2
#else
#pragma unroll
#endif
      for (int u = 0; u < (BK * D / 16) / kCThr; ++u) {
        uint4 w = make_uint4(0, 0, 0, 0);
        if (rr < rv) w = *reinterpret_cast<const uint4*>(pk);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(qk), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w));
        pk += (long)kRowK * k_sM; qk += kRowK * LDK8; rr += kRowK;
      }
    }
    if (fillV) {
      const int8_t* pv = Vq + (long)(kb + r0) * k_sM + c0;
      unsigned qv = (unsigned)__cvta_generic_to_shared(sBuf + bufV * kBuf + BK * LDK8 + r0 * LDV + c0);
      int rr = r0;
#if FA2_SO_ROLLCP
#pragma unroll 2
#else
#pragma unroll
#endif
      for (int u = 0; u < (BK * D / 16) / kCThr; ++u) {
        uint4 w = make_uint4(0, 0, 0, 0);
        if (rr < rv) w = *reinterpret_cast<const uint4*>(pv);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(qv), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w));
        pv += (long)kRowK * k_sM; qv += kRowK * LDV; rr += kRowK;
      }
    }
    if (fillS) {
      for (int j = ctid; j < BK; j += kCThr) {
        const int p = kb + j;
        sKs[bufS * BK + j] = (p < Sk) ? Ks[p] : 0.f;
        sVs[bufS * BK + j] = (p < Sk) ? Vs[p] : 0.f;
      }
    }
  };
  if (isC && nTiles > 0) podacha(FA2SO_KB_OF(tBeg), 0, true, 0, false, 0, true);
  __syncthreads();

  // ---- состояние ролей ----
  const int rowbase = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
  const int cbase = (lane & 2) | ((lane & 4) << 1);
  WarpMma<SMB, SNB, BLayout::kKMajor> s;
  // СОСТОЯНИЕ РОЛЕЙ -- ОДИН union: у S-варпов здесь Q (64 шага x 2 регистра = 128), у O-варпов -- накопитель O
  // (2x4x8 = 64). Порознь они стоили 192 регистра в одной функции (253 занятых, потолок 255), вместе -- 128.
  static_assert(sizeof(uint32_t[D / 4][SMB][2]) >= sizeof(float[OMB][ONB][8]), "union: Q-фрагменты вмещают накопитель O");
#if FA2_SO_QSMEM
  union Sost { uint32_t qf[1][SMB][2]; float oacc[OMB][ONB][8]; } st;   // Q не резидентно: union держит только O
#elif FA2_SO_QHALF
  union Sost { uint32_t qf[D / 8][SMB][2]; float oacc[OMB][ONB][8]; } st; // половина Q: 32 шага x 2 = 64 регистра
#else
  union Sost { uint32_t qf[D / 4][SMB][2]; float oacc[OMB][ONB][8]; } st;
#endif
  float qs[SMB * 2];
  float m_run[SMB][2], l_run[SMB][2];
  WarpMma<OMB, ONB, BLayout::kNMajor> o;       // только методы (фрагменты/адреса); накопитель -- st.oacc
  if (isS) {
#if FA2_SO_QHALF
    s.template preload_a<D / 2>(st.qf, sQ, LDQ, rs, lane);
#elif !FA2_SO_QSMEM
    s.template preload_a<D>(st.qf, sQ, LDQ, rs, lane);
#endif
#pragma unroll
    for (int t = 0; t < 2; ++t) qs[t] = sQsum[rs + rowbase + 2 * t];
#pragma unroll
    for (int t = 0; t < 2; ++t) { m_run[0][t] = -CUDART_INF_F; l_run[0][t] = 0.f; }
  } else {
#pragma unroll
    for (int i = 0; i < OMB; ++i)
#pragma unroll
      for (int j = 0; j < ONB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r) st.oacc[i][j][r] = 0.f;
    qs[0] = qs[1] = 0.f; m_run[0][0] = m_run[0][1] = -CUDART_INF_F; l_run[0][0] = l_run[0][1] = 0.f;
  }
  __syncthreads();                             // Q в регистрах -- область sQ отдаётся под P[2]

  // ---- главный цикл: шаг t = { подача K(t+1),V(t),масштабы(t+1) | S(t) | O(t-1) } ; одно рандеву ----
  SOPH_DECL
  for (int t = 0; t <= nTiles; ++t) {
    if (isS) {
      if (t < nTiles) {
        const int kb = FA2SO_KB_OF(tBeg + t), buf = t & 1;
        const int8_t* sK = sBuf + buf * kBuf;
        seed_rank1<float, SMB, SNB>(s.acc, kI8Bias, qs, 2);
#if FA2_SO_QSMEM
        s.template accumulate_bi8_roll<D, KVFMT>(sQ, LDQ, rs, sK, LDK8, cs, lane);
#elif FA2_SO_QHALF
        s.template accumulate_bi8_pre<D / 2, KVFMT, FA2_SO_KSP>(st.qf, sK, LDK8, cs, lane);                 // d < 128: Q в регистрах
        s.template accumulate_bi8_roll<D / 2, KVFMT>(sQ + D / 2, LDQ, rs, sK + D / 2, LDK8, cs, lane);   // d >= 128: Q из разделяемой, цикл свёрнут
#else
        s.template accumulate_bi8_pre<D, KVFMT, FA2_SO_KSP>(st.qf, sK, LDK8, cs, lane);
#endif
        SOPH(0);
        // маска, масштаб ключа, максимум строки по своим 32 ключам
        float mloc[2] = {-CUDART_INF_F, -CUDART_INF_F};
        s.visit(lane, [&](float& v, int r, int c) {
          const int kk = kb + cs + c;
          v *= qkm * sKs[buf * BK + cs + c];
          if ((CAUSAL && (q0 + rs + r + causal_off) < kk) || kk >= Sk) v = -CUDART_INF_F;
          const int sub = ((r & 15) - rowbase) >> 1;
          if (v > mloc[sub]) mloc[sub] = v;
        });
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          float x = mloc[tt];
          float y = __shfl_xor_sync(0xffffffffu, x, 2); x = x > y ? x : y;
          y = __shfl_xor_sync(0xffffffffu, x, 4);       x = x > y ? x : y;
          mloc[tt] = x;
          if ((lane & 6) == 0) sRedM[half][rs + rowbase + 2 * tt] = x;
        }
        SOPH(2);
        asm volatile("bar.sync %0, 64;" :: "r"(2 + pairId));           // пара: обмен максимумов
        SOPH(1);
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const float y = sRedM[half ^ 1][rs + rowbase + 2 * tt];
          mloc[tt] = mloc[tt] > y ? mloc[tt] : y;
        }
        float corr2[2], cnew[2], lloc[2] = {0.f, 0.f}, vloc[2] = {0.f, 0.f};
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const float mo = m_run[0][tt], mn = (mo > mloc[tt]) ? mo : mloc[tt];
          corr2[tt] = (mo == -CUDART_INF_F) ? 0.f : exp2f(mo - mn);
          cnew[tt]  = (mloc[tt] == -CUDART_INF_F) ? 0.f : exp2f(mloc[tt] - mn);
          m_run[0][tt] = mn;
        }
        __half* sPb = sP + buf * BQ * LDP;
#pragma unroll
        for (int j = 0; j < SNB; ++j)
#pragma unroll
          for (int pr = 0; pr < 4; ++pr) {
            const int rg = (pr & 1) ? (4 + ((pr >> 1) << 1)) : ((pr >> 1) << 1);
            const int reg0 = rg, reg1 = rg + 1;
            const int row = rowbase + 2 * ((reg0 >> 1) & 1);
            const int col = j * 16 + cbase + 4 * ((reg0 >> 2) & 1);
            const int sub = (reg0 >> 1) & 1;
            const float mm = mloc[sub];
            const float v0 = s.acc[0][j][reg0], v1 = s.acc[0][j][reg1];
            const float p0 = (v0 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v0 - mm);
            const float p1 = (v1 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v1 - mm);
            lloc[sub] += p0 + p1;
            const float sc2 = cnew[sub];
            const int cc = cs + col;
            const float w0 = p0 * sc2 * sVs[buf * BK + cc], w1 = p1 * sc2 * sVs[buf * BK + cc + 1];
            const __half2 ph = __floats2half2_rn(w0, w1);
            vloc[sub] += __half2float(__low2half(ph)) + __half2float(__high2half(ph));
            *reinterpret_cast<__half2*>(&sPb[(rs + row) * LDP + cc]) = ph;
          }
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          float x = lloc[tt], y = vloc[tt];
          x += __shfl_xor_sync(0xffffffffu, x, 2); y += __shfl_xor_sync(0xffffffffu, y, 2);
          x += __shfl_xor_sync(0xffffffffu, x, 4); y += __shfl_xor_sync(0xffffffffu, y, 4);
          l_run[0][tt] = l_run[0][tt] * corr2[tt] + x * cnew[tt];      // частичное l по своей половине
          if ((lane & 6) == 0) {
            sVb[buf][half][rs + rowbase + 2 * tt] = y;
            if (half == 0) sCorr[buf][rs + rowbase + 2 * tt] = corr2[tt];
          }
        }
        SOPH(2);
      }
    } else if (isO || isC) {
      constexpr int kItK = (BK * D / 16) / kCThr;
      uint4 kr[FA2_SO_COPYLATE == 2 ? kItK : 1];
      if (isC && FA2_SO_COPYLATE == 2 && t + 1 < nTiles) {          // [COPYLATE=2] LDG K(t+1) ДО PV -- задержка за HMMA
        constexpr int kRowK = kCThr / (D / 16);
        const int r0 = ctid / (D / 16), c0 = (ctid % (D / 16)) * 16;
        const int kb = FA2SO_KB_OF(tBeg + t + 1), rv = (kb + BK <= Sk) ? BK : ((Sk > kb) ? (Sk - kb) : 0);
        const int8_t* pk = Kq + (long)(kb + r0) * k_sM + c0;
#pragma unroll
        for (int u = 0; u < kItK; ++u) {
          kr[u] = make_uint4(0, 0, 0, 0);
          if (r0 + kRowK * u < rv) kr[u] = *reinterpret_cast<const uint4*>(pk + (long)kRowK * u * k_sM);
        }
      }
      if (isC && FA2_SO_PREF) {                                      // [PREF] строки следующего шага -- в L1 заранее
        constexpr int kRowK = kCThr / (D / 16);
        const int r0 = ctid / (D / 16), c0 = (ctid % (D / 16)) * 16;
        if (t + 2 < nTiles) {
          const int8_t* pk = Kq + (long)(FA2SO_KB_OF(tBeg + t + 2) + r0) * k_sM + c0;
#pragma unroll
          for (int u = 0; u < (BK * D / 16) / kCThr; ++u) asm volatile("prefetch.global.L1 [%0];" :: "l"(pk + (long)kRowK * u * k_sM));
        }
        if (t + 1 < nTiles) {
          const int8_t* pv = Vq + (long)(FA2SO_KB_OF(tBeg + t + 1) + r0) * k_sM + c0;
#pragma unroll
          for (int u = 0; u < (BK * D / 16) / kCThr; ++u) asm volatile("prefetch.global.L1 [%0];" :: "l"(pv + (long)kRowK * u * k_sM));
        }
      }
      if (isC && FA2_SO_COPYLATE == 0) {
        if (t + 1 < nTiles) podacha(FA2SO_KB_OF(tBeg + t + 1), (t + 1) & 1, true, 0, false, (t + 1) & 1, true);   // K(t+1), масштабы(t+1)
        if (t < nTiles)     podacha(FA2SO_KB_OF(tBeg + t), 0, false, t & 1, true, 0, false);                           // V(t)
        SOPH(4);
      }
      if (isO && t >= 1) {
        const int buf = (t - 1) & 1;
        const int8_t* sV = sBuf + buf * kBuf + BK * LDK8;
        const __half* sPb = sP + buf * BQ * LDP;
        float corr[OMB][2], vb[OMB][2];
#pragma unroll
        for (int i = 0; i < OMB; ++i)
#pragma unroll
          for (int sub = 0; sub < 2; ++sub) {
            const int row = i * 16 + rowbase + 2 * sub;
            corr[i][sub] = sCorr[buf][row];
            vb[i][sub] = sVb[buf][0][row] + sVb[buf][1][row];
          }
        rescale_fold<float, OMB, ONB>(st.oacc, corr, kI8Bias, vb);
        o.template accumulate_bi8_into<BK, KVFMT>(st.oacc, sPb, LDP, 0, sV, LDV, ow * ON, lane);
        SOPH(5);
      }
      if (isC && FA2_SO_COPYLATE == 1) {
        if (t + 1 < nTiles) podacha(FA2SO_KB_OF(tBeg + t + 1), (t + 1) & 1, true, 0, false, (t + 1) & 1, true);
        if (t < nTiles)     podacha(FA2SO_KB_OF(tBeg + t), 0, false, t & 1, true, 0, false);
      }
      if (isC && FA2_SO_COPYLATE == 2) {
        if (t + 1 < nTiles) {                                        // укладка K(t+1) из регистров + масштабы
          constexpr int kRowK = kCThr / (D / 16);
          const int r0 = ctid / (D / 16), c0 = (ctid % (D / 16)) * 16;
          unsigned qk = (unsigned)__cvta_generic_to_shared(sBuf + ((t + 1) & 1) * kBuf + r0 * LDK8 + c0);
#pragma unroll
          for (int u = 0; u < kItK; ++u) {
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(qk), "r"(kr[u].x), "r"(kr[u].y), "r"(kr[u].z), "r"(kr[u].w));
            qk += kRowK * LDK8;
          }
          podacha(FA2SO_KB_OF(tBeg + t + 1), 0, false, 0, false, (t + 1) & 1, true);
        }
        if (t < nTiles) podacha(FA2SO_KB_OF(tBeg + t), 0, false, t & 1, true, 0, false);   // V(t) как раньше
      }
    }
    asm volatile("bar.sync 1, %0;" :: "r"(kAllThr));                  // ОДНО рандеву на шаг
    if (isS) SOPH(3); else SOPH(6);
  }
  SOPH_OUT();

  // ---- эпилог: l = l0 + l1 (пара), inv и LSE -- S-варпы; O = acc * inv -- O-варпы ----
  if (isS && (lane & 6) == 0) {
#pragma unroll
    for (int tt = 0; tt < 2; ++tt) sLpart[half][rs + rowbase + 2 * tt] = l_run[0][tt];
  }
  asm volatile("bar.sync 1, %0;" :: "r"(kAllThr));
  if (isS && half == 0 && (lane & 6) == 0) {
#pragma unroll
    for (int tt = 0; tt < 2; ++tt) {
      const int row = rs + rowbase + 2 * tt;
      const float l = sLpart[0][row] + sLpart[1][row];
      sInv[row] = (l > 0.f) ? (kFmtMul * __frcp_rn(l)) : 0.f;
      if (q0 + row < Sq) {
        const float _lse = (l > 0.f) ? (m_run[0][tt] * 0.6931471805599453f + __logf(l)) : -CUDART_INF_F;
        if (Lp != nullptr) Lp[q0 + row] = _lse; else Lse[q0 + row] = _lse;
      }
    }
  }
  asm volatile("bar.sync 1, %0;" :: "r"(kAllThr));
  if (isO) {
#pragma unroll
    for (int i = 0; i < OMB; ++i)
#pragma unroll
      for (int j = 0; j < ONB; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; ++rg) {
          const int sub = (rg >> 1) & 1;
          const int row = i * 16 + rowbase + 2 * sub;
          const int col = j * 16 + cbase + (rg & 1) + 4 * ((rg >> 2) & 1);
          if (q0 + row < Sq) {
            const float _v = st.oacc[i][j][rg] * sInv[row];
            if (Op != nullptr) Op[(long)(q0 + row) * D + ow * ON + col] = _v;
            else O[(long)(q0 + row) * o_sM + ow * ON + col] = __float2half(_v);
          }
        }
  }
}

}  // namespace fwd_so
}  // namespace fa2_sm70
#undef FA2SO_KB_OF
