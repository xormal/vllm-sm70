// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ==================================================================================================
// CLEF ВНУТРИ БОЕВОЙ СЕТИ: поправка весов по строкам запроса и голова классификатора, без PyTorch.
// ==================================================================================================
//
// ЧТО ЭТО. Cloudflare/clef = Qwen3.8-27B + LoRA ранга 256 в слоях 40-63 + голова (128M). Боевая сеть
// = тот же Qwen3.8, но ABL (ранг-1 аблитерация o/out/down в слоях 2-63) и int8. Для строк запроса
// Clef к выходу каждой затронутой проекции добавляется
//     y[строки Clef] += (x · Aᵀ) · Bᵀ        (LoRA Clef и обратная аблитерация)
//     y[строки Clef] += x · Dᵀ               (плотная разница in_proj_a/b)
// Базовые строки не трогаются НИ ОДНИМ байтом: поправка идёт только по сегментам строк Clef.
//
// РАЗРЕЗ TP -- ОБЫЧНЫЙ, без своих коллективных операций:
//   column-parallel: A целиком, B -- свои строки выхода (раскладываются по столбцам слитого выхода);
//   row-parallel:    A -- свои столбцы входа, B целиком; частичная сумма уходит в тот же all-reduce,
//                    что сводит сам слой (вызов стоит ВНУТРИ apply, до сведения).
//
// ПАМЯТЬ (замер 11.10, верхняя оценка): множители слоёв 40-63 живут в закреплённой памяти хоста и
// подкачиваются кольцом на nslot слоёв по своему потоку, прячась за счётом слоя. Обратная аблитерация
// слоёв 2-39 (единицы КиБ на слой) лежит на карте постоянно.
//
// ГОЛОВА (только на ранге 0): во время префилла копится memory = LN(h)·Wmᵀ по всем позициям и суммы
// LN(h) по спанам вопросов/вариантов; в конце запроса голова считается по ступеням, веса ступени
// приходят из закреплённой памяти в один буфер. Внимание к памяти -- кусками с онлайн-софтмаксом
// (целиком стоило бы +0.1 ГБ временного).
//
// ОТКАЗ ПРЕДУСЛОВИЯ НЕ БРОСАЕТ ИСКЛЮЧЕНИЕ НА ПУТИ ШАГА (закон fa2-sm70-check-ubivaet-vorker): шаг
// помечается испорченным, счётчик растёт, базовые строки идут как шли. Бросают только init-функции.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

#define CLEF_CK(x)                                                                              \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    TORCH_CHECK(e_ == cudaSuccess, "clef: ", #x, ": ", cudaGetErrorString(e_));                 \
  } while (0)

// ------------------------------------------------------------------------------------------------
// ОБЩЕЕ: y[M,N] (row-major, ldy) = x[M,K] (row-major, ldx) · W[N,K]ᵀ (row-major, ldw) (+ beta·y)
// ------------------------------------------------------------------------------------------------
inline cublasStatus_t gemm_xwt(const half* x, int ldx, const half* w, int ldw, void* y,
                               cudaDataType yt, int ldy, int M, int N, int K, float beta) {
  const float alpha = 1.f;
  cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();   // поток уже текущий
  return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16F, ldw, x,
                      CUDA_R_16F, ldx, &beta, y, yt, ldy, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__device__ __forceinline__ float blk_sum(float v, float* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
  if (w == 0)
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  if (threadIdx.x == 0) sh[32] = t;
  __syncthreads();
  return sh[32];
}

__device__ __forceinline__ float to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(float v) { return v; }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ half from_f<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

__device__ __forceinline__ float gelu_f(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }

// LayerNorm по строке (nn.LayerNorm, eps 1e-5): два прохода -- среднее, затем дисперсия.
template <typename Ti, typename To>
__global__ void ln_k(const Ti* __restrict__ x, int ldx, const half* __restrict__ w,
                     const half* __restrict__ b, To* __restrict__ y, int ldy, int D, float eps) {
  __shared__ float sh[33];
  const Ti* xr = x + (int64_t)blockIdx.x * ldx;
  To* yr = y + (int64_t)blockIdx.x * ldy;
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += to_f(xr[i]);
  const float mean = blk_sum(s, sh) / D;
  float s2 = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float d = to_f(xr[i]) - mean;
    s2 += d * d;
  }
  const float rstd = rsqrtf(blk_sum(s2, sh) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    yr[i] = from_f<To>((to_f(xr[i]) - mean) * rstd * __half2float(w[i]) + __half2float(b[i]));
}

// Эпилог GEMM в fp32: v = C + bias; act (1 = gelu); resid += v; outh = half(v).
__global__ void epi_k(const float* __restrict__ C, int ldc, const half* __restrict__ bias, int M,
                      int N, int act, float* __restrict__ resid, int ldr, half* __restrict__ outh,
                      int ldo) {
  const int64_t n_all = (int64_t)M * N;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n_all;
       e += (int64_t)gridDim.x * blockDim.x) {
    const int m = (int)(e / N), n = (int)(e % N);
    float v = C[(int64_t)m * ldc + n] + (bias ? __half2float(bias[n]) : 0.f);
    if (act == 1) v = gelu_f(v);
    if (resid) resid[(int64_t)m * ldr + n] += v;
    if (outh) outh[(int64_t)m * ldo + n] = __float2half_rn(v);
  }
}

__global__ void bias_half_k(half* __restrict__ Y, int ldy, const half* __restrict__ b, int M, int N) {
  const int64_t n_all = (int64_t)M * N;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n_all;
       e += (int64_t)gridDim.x * blockDim.x) {
    const int m = (int)(e / N), n = (int)(e % N);
    half* p = Y + (int64_t)m * ldy + n;
    *p = __float2half_rn(__half2float(*p) + __half2float(b[n]));
  }
}

__global__ void fill_k(float* __restrict__ p, int64_t n, float v) {
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n;
       e += (int64_t)gridDim.x * blockDim.x)
    p[e] = v;
}

// ------------------------------------------------------------------------------------------------
// ВНИМАНИЕ ГОЛОВЫ: 16 голов по 64, мало запросов (варианты/поля), много ключей (память N до 16K).
// Нить = (голова, запрос); состояние онлайн-софтмакса (m, l, acc) живёт в глобальной памяти между
// кусками ключей. Плитка ключей 32 в разделяемой памяти, чтение -- широковещательное.
// ------------------------------------------------------------------------------------------------
constexpr int HD = 64, NH = 16, WD = 1024, KT = 32;

__global__ void attn_upd_k(const half* __restrict__ Q, int ldq, const half* __restrict__ K,
                           const half* __restrict__ V, int ldkv, int nk, int nq, float* __restrict__ Mx,
                           float* __restrict__ L, float* __restrict__ ACC, float scale) {
  __shared__ half sK[KT][HD];
  __shared__ half sV[KT][HD];
  const int h = blockIdx.x;
  const int qi = blockIdx.y * blockDim.x + threadIdx.x;
  const bool act = qi < nq;
  float q[HD], acc[HD];
  float m = -INFINITY, l = 0.f;
  if (act) {
    const half* qr = Q + (int64_t)qi * ldq + h * HD;
    float* ar = ACC + (int64_t)qi * WD + h * HD;
#pragma unroll
    for (int d = 0; d < HD; ++d) {
      q[d] = __half2float(qr[d]) * scale;
      acc[d] = ar[d];
    }
    m = Mx[h * nq + qi];
    l = L[h * nq + qi];
  }
  for (int k0 = 0; k0 < nk; k0 += KT) {
    for (int e = threadIdx.x; e < KT * HD; e += blockDim.x) {
      const int kk = e / HD, d = e % HD, key = k0 + kk;
      sK[kk][d] = key < nk ? K[(int64_t)key * ldkv + h * HD + d] : __float2half(0.f);
      sV[kk][d] = key < nk ? V[(int64_t)key * ldkv + h * HD + d] : __float2half(0.f);
    }
    __syncthreads();
    if (act) {
      const int kn = min(KT, nk - k0);
      float s[KT];
      float mx = m;
#pragma unroll
      for (int j = 0; j < KT; ++j) {
        float d = 0.f;
#pragma unroll
        for (int t = 0; t < HD; ++t) d += q[t] * __half2float(sK[j][t]);
        s[j] = j < kn ? d : -INFINITY;
        mx = fmaxf(mx, s[j]);
      }
      const float corr = __expf(m - mx);
      l *= corr;
#pragma unroll
      for (int t = 0; t < HD; ++t) acc[t] *= corr;
#pragma unroll
      for (int j = 0; j < KT; ++j) {
        const float p = __expf(s[j] - mx);
        l += p;
#pragma unroll
        for (int t = 0; t < HD; ++t) acc[t] += p * __half2float(sV[j][t]);
      }
      m = mx;
    }
    __syncthreads();
  }
  if (act) {
    float* ar = ACC + (int64_t)qi * WD + h * HD;
#pragma unroll
    for (int d = 0; d < HD; ++d) ar[d] = acc[d];
    Mx[h * nq + qi] = m;
    L[h * nq + qi] = l;
  }
}

__global__ void attn_fin_k(const float* __restrict__ ACC, const float* __restrict__ L, int nq,
                           half* __restrict__ out) {
  const int64_t n_all = (int64_t)nq * WD;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n_all;
       e += (int64_t)gridDim.x * blockDim.x) {
    const int qi = (int)(e / WD), c = (int)(e % WD), h = c / HD;
    out[e] = __float2half_rn(ACC[e] / L[h * nq + qi]);
  }
}

// ------------------------------------------------------------------------------------------------
// ОТВОД ДЛЯ ГОЛОВЫ И СБОРКА ВХОДА
// ------------------------------------------------------------------------------------------------
// суммы LN(h) по спанам (spanid[pos] = номер спана или -1) и последний токен -> glob
__global__ void span_add_k(const half* __restrict__ hn, const int* __restrict__ spanid,
                           float* __restrict__ span, float* __restrict__ glob, int D, int last_row) {
  const int r = blockIdx.x;
  const int sid = spanid[r];
  const half* hr = hn + (int64_t)r * D;
  if (sid >= 0)
    for (int i = threadIdx.x; i < D; i += blockDim.x) atomicAdd(&span[(int64_t)sid * D + i], __half2float(hr[i]));
  if (r == last_row)
    for (int i = threadIdx.x; i < D; i += blockDim.x) glob[i] = __half2float(hr[i]);
}

__global__ void span_fin_k(float* __restrict__ span, const float* __restrict__ inv_len, int S, int D,
                           half* __restrict__ out16) {
  const int64_t n_all = (int64_t)S * D;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n_all;
       e += (int64_t)gridDim.x * blockDim.x) {
    const float v = span[e] * inv_len[e / D];
    span[e] = v;
    out16[e] = __float2half_rn(v);
  }
}

__global__ void to_half_k(const float* __restrict__ x, half* __restrict__ y, int64_t n) {
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n;
       e += (int64_t)gridDim.x * blockDim.x)
    y[e] = __float2half_rn(x[e]);
}

// X[i] = C[i] + Qp[qidx[i]]
__global__ void add_rep_k(const float* __restrict__ C, const float* __restrict__ Qp,
                          const int* __restrict__ qidx, float* __restrict__ X, int nq) {
  const int64_t n_all = (int64_t)nq * WD;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n_all;
       e += (int64_t)gridDim.x * blockDim.x) {
    const int i = (int)(e / WD), d = (int)(e % WD);
    X[e] = C[e] + Qp[(int64_t)qidx[i] * WD + d];
  }
}

// summ[q] = Σ_j softmax_j(X[j]·BF[q] / sqrt(WD)) · X[j],  j по вариантам вопроса q
__global__ void summary_k(const float* __restrict__ X, const float* __restrict__ BF,
                          const int* __restrict__ qoff, float* __restrict__ summ) {
  __shared__ float sh[33];
  __shared__ float w[256];
  const int q = blockIdx.x, j0 = qoff[q], cnt = qoff[q + 1] - qoff[q];
  const float* f = BF + (int64_t)q * WD;
  float mx = -INFINITY;
  for (int j = 0; j < cnt && j < 256; ++j) {
    float s = 0.f;
    for (int d = threadIdx.x; d < WD; d += blockDim.x) s += X[(int64_t)(j0 + j) * WD + d] * f[d];
    s = blk_sum(s, sh) * 0.03125f;   // 1/sqrt(1024)
    if (threadIdx.x == 0) w[j] = s;
    mx = fmaxf(mx, s);
  }
  __syncthreads();
  float z = 0.f;
  for (int j = 0; j < cnt && j < 256; ++j) z += __expf(w[j] - mx);
  for (int d = threadIdx.x; d < WD; d += blockDim.x) {
    float a = 0.f;
    for (int j = 0; j < cnt && j < 256; ++j) a += __expf(w[j] - mx) * X[(int64_t)(j0 + j) * WD + d];
    summ[(int64_t)q * WD + d] = a / z;
  }
}

// F[q] = BF[q] + SN[q] + gp + temb[type[q]]
__global__ void combine_k(const float* __restrict__ BF, const float* __restrict__ SN,
                          const float* __restrict__ gp, const half* __restrict__ temb,
                          const int* __restrict__ types, float* __restrict__ F, int nq) {
  const int64_t n_all = (int64_t)nq * WD;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n_all;
       e += (int64_t)gridDim.x * blockDim.x) {
    const int q = (int)(e / WD), d = (int)(e % WD);
    F[e] = BF[e] + SN[e] + gp[d] + __half2float(temb[(int64_t)types[q] * WD + d]);
  }
}

// признаки скорера [f, o, f*o, |f-o|] и косинус поля и варианта
__global__ void feat_k(const float* __restrict__ FN, const float* __restrict__ ON,
                       const int* __restrict__ qidx, half* __restrict__ G, float* __restrict__ cosv) {
  __shared__ float sh[33];
  const int j = blockIdx.x, q = qidx[j];
  const float* f = FN + (int64_t)q * WD;
  const float* o = ON + (int64_t)j * WD;
  half* g = G + (int64_t)j * 4 * WD;
  float fo = 0.f, ff = 0.f, oo = 0.f;
  for (int d = threadIdx.x; d < WD; d += blockDim.x) {
    const float a = f[d], b = o[d];
    g[d] = __float2half_rn(a);
    g[WD + d] = __float2half_rn(b);
    g[2 * WD + d] = __float2half_rn(a * b);
    g[3 * WD + d] = __float2half_rn(fabsf(a - b));
    fo += a * b;
    ff += a * a;
    oo += b * b;
  }
  fo = blk_sum(fo, sh);
  ff = blk_sum(ff, sh);
  oo = blk_sum(oo, sh);
  if (threadIdx.x == 0) cosv[j] = fo / sqrtf(fmaxf(ff * oo, 1e-16f));
}

// prior[j] = ps · <normalize(lex_j), normalize(qv_q + glob)>   (5120)
__global__ void prior_k(const half* __restrict__ lex, const float* __restrict__ qv,
                        const float* __restrict__ glob, const int* __restrict__ qidx, int D, float ps,
                        float* __restrict__ prior) {
  __shared__ float sh[33];
  const int j = blockIdx.x, q = qidx[j];
  float la = 0.f, ll = 0.f, aa = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float a = qv[(int64_t)q * D + d] + glob[d];
    const float l = __half2float(lex[(int64_t)j * D + d]);
    la += l * a;
    ll += l * l;
    aa += a * a;
  }
  la = blk_sum(la, sh);
  ll = blk_sum(ll, sh);
  aa = blk_sum(aa, sh);
  if (threadIdx.x == 0) prior[j] = ps * la / (fmaxf(sqrtf(ll), 1e-12f) * fmaxf(sqrtf(aa), 1e-12f));
}

__global__ void final_k(const float* __restrict__ prior, const float* __restrict__ cosv,
                        const float* __restrict__ res, const half* __restrict__ b3, float js, float sg,
                        int n, float* __restrict__ out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) out[j] = prior[j] + sg * (js * cosv[j] + res[j] + __half2float(b3[0]));
}

inline int nblk(int64_t n, int t = 256) { return (int)std::min<int64_t>((n + t - 1) / t, 4096); }

// ------------------------------------------------------------------------------------------------
// СОСТОЯНИЕ ПРОЦЕССА
// ------------------------------------------------------------------------------------------------
struct Op {        // одна операция модуля; порядок важен: сжатие задаёт t для следующих разжатий
  int mod, type;   // type 0 = сжатие t = x·Aᵀ (A [n, in]); 1 = разжатие y[:,col:col+n] += t·Bᵀ (B [n, r]); 2 = плотное y += x·Dᵀ
  int64_t off;     // смещение в байтах в блобе слоя
  int n, in, col;
};
struct HLayer {
  int model_layer = -1;
  bool streamed = false;
  int sidx = -1;
  int64_t bytes = 0;
  at::Tensor host;   // закреплённая память (uint8)
  at::Tensor dev;    // для постоянных
  std::vector<Op> ops;
};
struct Seg { int row0, rows, req, pos0, fin; };

struct ReqSlot {
  bool used = false, ready = false, bad = false;
  int N = 0, Qn = 0, Qo = 0, S = 0;
};

struct Engine {
  bool inited = false;
  int rmax = 264, max_rows = 2048, nslot = 3;
  std::vector<HLayer> layers;
  std::vector<int> lmap;                   // модельный слой -> индекс в layers
  std::vector<int> streamed;               // sidx -> индекс в layers
  std::vector<at::Tensor> slots;
  std::vector<int> slot_layer;
  std::vector<cudaEvent_t> ev_copied, ev_free;
  cudaStream_t cs = nullptr;
  int64_t slot_bytes = 0;
  at::Tensor tbuf;
  // шаг
  bool active = false, step_bad = false;
  int cur = -1;
  std::vector<Seg> segs;
  // счётчики
  int64_t n_delta = 0, n_gemm = 0, n_copy = 0, n_err = 0, n_steps = 0, n_tap = 0, n_head = 0;
  std::string last_err;
  // голова
  bool head_on = false;
  int R = 0, Nmax = 0, Smax = 0, Qomax = 0, Qnmax = 0, chunk = 1024;
  float ps = 1.f, js = 1.f, sg = 0.5f;
  std::vector<at::Tensor> stage_host;
  std::map<std::string, std::tuple<int, int64_t, int64_t>> widx;   // имя -> (ступень, смещение эл., эл.)
  at::Tensor stage_buf, hn_w, hn_b, mp_w;
  at::Tensor mem, span, glob, spanid, inv_len, qidx, qoff, types, lex, result;
  at::Tensor ws_hn, ws_X, ws_F, ws_H, ws_C, ws_Hq, ws_G, ws_KV, ws_MC, ws_m, ws_l, ws_acc;
  at::Tensor ws_ctx16, ws_qv16, ws_gl16, ws_Qp, ws_BF, ws_SN, ws_SM, ws_gp, ws_FN, ws_ON, ws_cos, ws_prior, ws_res;
  std::vector<ReqSlot> req;
  int loaded_stage = -1;
  // указатель тензора веса (scales / pack / weight) -> (слой, модуль): ключ, по которому тело
  // умножения узнаёт свой модуль, не меняя ни сигнатуры операции, ни графа
  std::unordered_map<uintptr_t, std::pair<int, int>> keys;
  int64_t n_keymiss = 0;
};
Engine E;

void fail(const std::string& why) {
  E.n_err++;
  E.step_bad = true;
  E.last_err = why;
}

// ------------------------------------------------------------------------------------------------
// ПОПРАВКА
// ------------------------------------------------------------------------------------------------
void init_delta(int64_t nslot, int64_t max_rows, int64_t rmax) {
  TORCH_CHECK(!E.inited, "clef: init_delta повторно");
  E.nslot = (int)nslot;
  E.max_rows = (int)max_rows;
  E.rmax = (int)rmax;
  E.lmap.assign(256, -1);
  E.inited = true;
}

void add_layer(int64_t model_layer, bool streamed, at::Tensor host, at::Tensor ops) {
  TORCH_CHECK(E.inited, "clef: сначала init_delta");
  TORCH_CHECK(host.device().is_cpu() && host.scalar_type() == at::kByte && host.is_contiguous(), "clef: host -- uint8 cpu");
  TORCH_CHECK(!streamed || host.is_pinned(), "clef: подкачиваемый слой обязан быть в закреплённой памяти");
  TORCH_CHECK(ops.device().is_cpu() && ops.scalar_type() == at::kLong && ops.dim() == 2 && ops.size(1) == 6, "clef: ops [P,6] int64");
  TORCH_CHECK(model_layer >= 0 && model_layer < 256 && E.lmap[model_layer] < 0, "clef: слой ", model_layer);
  HLayer L;
  L.model_layer = (int)model_layer;
  L.streamed = streamed;
  L.bytes = host.numel();
  L.host = host;
  auto a = ops.accessor<int64_t, 2>();
  for (int64_t i = 0; i < ops.size(0); ++i) {
    Op o{(int)a[i][0], (int)a[i][1], a[i][2], (int)a[i][3], (int)a[i][4], (int)a[i][5]};
    TORCH_CHECK(o.type >= 0 && o.type <= 2 && o.off >= 0 && o.off < L.bytes, "clef: плохая операция");
    if (o.type == 0) TORCH_CHECK(o.n <= E.rmax, "clef: ранг ", o.n, " > rmax ", E.rmax);
    L.ops.push_back(o);
  }
  if (streamed) {
    L.sidx = (int)E.streamed.size();
    E.streamed.push_back((int)E.layers.size());
    E.slot_bytes = std::max(E.slot_bytes, L.bytes);
  } else {
    L.dev = host.to(at::Device(at::kCUDA, c10::cuda::current_device()));
  }
  E.lmap[model_layer] = (int)E.layers.size();
  E.layers.push_back(std::move(L));
}

void finalize_delta() {
  TORCH_CHECK(E.inited && E.cs == nullptr, "clef: finalize_delta");
  const auto dev = at::Device(at::kCUDA, c10::cuda::current_device());
  const int ns = std::min<int>(E.nslot, (int)E.streamed.size());
  E.nslot = std::max(ns, 1);
  for (int s = 0; s < ns; ++s) E.slots.push_back(at::empty({E.slot_bytes}, at::TensorOptions().dtype(at::kByte).device(dev)));
  E.slot_layer.assign(E.nslot, -1);
  E.ev_copied.resize(E.nslot);
  E.ev_free.resize(E.nslot);
  for (int s = 0; s < E.nslot; ++s) {
    CLEF_CK(cudaEventCreateWithFlags(&E.ev_copied[s], cudaEventDisableTiming));
    CLEF_CK(cudaEventCreateWithFlags(&E.ev_free[s], cudaEventDisableTiming));
  }
  CLEF_CK(cudaStreamCreateWithFlags(&E.cs, cudaStreamNonBlocking));
  E.tbuf = at::empty({(int64_t)E.max_rows, (int64_t)E.rmax}, at::TensorOptions().dtype(at::kHalf).device(dev));
}

void prefetch(int k) {
  const int s = k % E.nslot;
  if (E.slot_layer[s] == k) return;
  const HLayer& L = E.layers[E.streamed[k]];
  CLEF_CK(cudaStreamWaitEvent(E.cs, E.ev_free[s], 0));
  CLEF_CK(cudaMemcpyAsync(E.slots[s].data_ptr(), L.host.data_ptr(), L.bytes, cudaMemcpyHostToDevice, E.cs));
  CLEF_CK(cudaEventRecord(E.ev_copied[s], E.cs));
  E.slot_layer[s] = k;
  E.n_copy++;
}

void enter(int k, cudaStream_t st) {
  if (E.cur >= 0) CLEF_CK(cudaEventRecord(E.ev_free[E.cur % E.nslot], st));
  const int S = (int)E.streamed.size();
  for (int j = k; j < std::min(S, k + E.nslot); ++j) prefetch(j);
  CLEF_CK(cudaStreamWaitEvent(st, E.ev_copied[k % E.nslot], 0));
  E.cur = k;
}

// сегменты строк Clef на этот шаг: [n,5] int32 (row0, rows, req, pos0, fin); пусто = шаг без Clef
void set_step(at::Tensor segs) {
  E.segs.clear();
  E.step_bad = false;
  if (segs.numel() > 0) {
    TORCH_CHECK(segs.device().is_cpu() && segs.scalar_type() == at::kInt && segs.dim() == 2 && segs.size(1) == 5, "clef: segs [n,5] int32 cpu");
    auto a = segs.accessor<int, 2>();
    for (int64_t i = 0; i < segs.size(0); ++i) E.segs.push_back({a[i][0], a[i][1], a[i][2], a[i][3], a[i][4]});
  }
  const bool was = E.active;
  E.active = !E.segs.empty() && E.cs != nullptr;
  if (!E.active) return;
  E.n_steps++;
  // работа прошлого шага уже поставлена в текущий поток: все ячейки кольца освобождаются за ней
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  for (int s = 0; s < E.nslot; ++s) CLEF_CK(cudaEventRecord(E.ev_free[s], st));
  E.cur = -1;
  (void)was;
  const int S = (int)E.streamed.size();
  for (int j = 0; j < std::min(S, E.nslot); ++j) prefetch(j);
}

void delta(at::Tensor x, at::Tensor y, int64_t model_layer, int64_t mod) {
  if (!E.active) return;
  if (model_layer < 0 || model_layer >= 256 || E.lmap[model_layer] < 0) return;
  if (x.dim() != 2 || y.dim() != 2 || x.scalar_type() != at::kHalf || y.scalar_type() != at::kHalf ||
      x.stride(1) != 1 || y.stride(1) != 1 || x.size(0) != y.size(0) || !x.is_cuda() || !y.is_cuda()) {
    fail("delta: форма/тип x,y");
    return;
  }
  HLayer& L = E.layers[E.lmap[model_layer]];
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const uint8_t* base;
  if (L.streamed) {
    if (L.sidx != E.cur) enter(L.sidx, st);
    base = (const uint8_t*)E.slots[L.sidx % E.nslot].data_ptr();
  } else {
    base = (const uint8_t*)L.dev.data_ptr();
  }
  const int64_t M = x.size(0), ldx = x.stride(0), ldy = y.stride(0), K = x.size(1), N = y.size(1);
  half* tb = (half*)E.tbuf.data_ptr();
  for (const Seg& sg : E.segs) {
    if (sg.rows <= 0) continue;
    if (sg.row0 < 0 || sg.row0 + sg.rows > M) { fail("delta: сегмент за пределами строк"); return; }
    const half* xs = (const half*)x.data_ptr() + sg.row0 * ldx;
    half* ys = (half*)y.data_ptr() + sg.row0 * ldy;
    for (int r0 = 0; r0 < sg.rows; r0 += E.max_rows) {
      const int m = std::min(E.max_rows, sg.rows - r0);
      int rank = 0;
      for (const Op& o : L.ops) {
        if (o.mod != mod) continue;
        const half* w = (const half*)(base + o.off);
        cublasStatus_t s = CUBLAS_STATUS_SUCCESS;
        if (o.type == 0) {
          if (o.in != K) { fail("delta: in != K"); return; }
          rank = o.n;
          s = gemm_xwt(xs + r0 * ldx, (int)ldx, w, o.in, tb, CUDA_R_16F, rank, m, rank, o.in, 0.f);
        } else if (o.type == 1) {
          if (rank == 0 || o.col + o.n > N) { fail("delta: разжатие без сжатия / за краем"); return; }
          s = gemm_xwt(tb, rank, w, rank, ys + r0 * ldy + o.col, CUDA_R_16F, (int)ldy, m, o.n, rank, 1.f);
        } else {
          if (o.in != K || o.col + o.n > N) { fail("delta: плотная форма"); return; }
          s = gemm_xwt(xs + r0 * ldx, (int)ldx, w, o.in, ys + r0 * ldy + o.col, CUDA_R_16F, (int)ldy, m, o.n, o.in, 1.f);
        }
        if (s != CUBLAS_STATUS_SUCCESS) { fail("delta: cublas " + std::to_string((int)s)); return; }
        E.n_gemm++;
      }
    }
  }
  E.n_delta++;
}

void bind(at::Tensor key, int64_t model_layer, int64_t mod) {
  TORCH_CHECK(key.numel() > 0, "clef: пустой ключ");
  E.keys[(uintptr_t)key.data_ptr()] = {(int)model_layer, (int)mod};
}

// вызывается из тела умножения: x -- вход, y -- выход (оба 2D-виды), keys -- тензоры веса слоя
void delta_keyed(at::Tensor x, at::Tensor y, std::vector<at::Tensor> keys) {
  if (!E.active) return;
  for (const auto& k : keys) {
    if (!k.defined() || k.numel() == 0) continue;
    auto it = E.keys.find((uintptr_t)k.data_ptr());
    if (it != E.keys.end()) {
      delta(x, y, it->second.first, it->second.second);
      return;
    }
  }
  E.n_keymiss++;
}

// ------------------------------------------------------------------------------------------------
// ГОЛОВА
// ------------------------------------------------------------------------------------------------
const half* W(const std::string& name) {
  auto it = E.widx.find(name);
  TORCH_CHECK(it != E.widx.end(), "clef: нет веса головы ", name);
  const int st = std::get<0>(it->second);
  TORCH_CHECK(st == E.loaded_stage, "clef: вес ", name, " не из загруженной ступени");
  return (const half*)E.stage_buf.data_ptr() + std::get<1>(it->second);
}

void load_stage(int st, cudaStream_t s) {
  const at::Tensor& h = E.stage_host[st];
  CLEF_CK(cudaMemcpyAsync(E.stage_buf.data_ptr(), h.data_ptr(), h.numel() * 2, cudaMemcpyHostToDevice, s));
  E.loaded_stage = st;
}

void init_head(std::vector<at::Tensor> stage_host, std::map<std::string, std::tuple<int64_t, int64_t, int64_t>> widx,
               at::Tensor hn_w, at::Tensor hn_b, at::Tensor mp_w, double ps, double js, double gate,
               int64_t R, int64_t Nmax, int64_t Smax, int64_t Qomax, int64_t Qnmax, int64_t chunk) {
  TORCH_CHECK(!E.head_on, "clef: init_head повторно");
  const auto dev = at::Device(at::kCUDA, c10::cuda::current_device());
  auto f16 = at::TensorOptions().dtype(at::kHalf).device(dev);
  auto f32 = at::TensorOptions().dtype(at::kFloat).device(dev);
  auto i32 = at::TensorOptions().dtype(at::kInt).device(dev);
  int64_t mx = 0;
  for (auto& t : stage_host) {
    TORCH_CHECK(t.is_pinned() && t.scalar_type() == at::kHalf && t.is_contiguous(), "clef: ступени -- half pinned");
    mx = std::max(mx, t.numel());
  }
  E.stage_host = stage_host;
  for (auto& kv : widx) E.widx[kv.first] = {(int)std::get<0>(kv.second), std::get<1>(kv.second), std::get<2>(kv.second)};
  E.stage_buf = at::empty({mx}, f16);
  E.hn_w = hn_w; E.hn_b = hn_b; E.mp_w = mp_w;
  E.ps = (float)std::exp(std::min(ps, std::log(100.0)));
  E.js = (float)std::exp(std::min(js, std::log(100.0)));
  E.sg = (float)(1.0 / (1.0 + std::exp(-gate)));
  E.R = (int)R; E.Nmax = (int)Nmax; E.Smax = (int)Smax; E.Qomax = (int)Qomax; E.Qnmax = (int)Qnmax; E.chunk = (int)chunk;
  E.mem = at::empty({R, Nmax, WD}, f16);
  E.span = at::empty({R, Smax, 5120}, f32);
  E.glob = at::empty({R, 5120}, f32);
  E.spanid = at::empty({R, Nmax}, i32);
  E.inv_len = at::empty({R, Smax}, f32);
  E.qidx = at::empty({R, Qomax}, i32);
  E.qoff = at::empty({R, Qnmax + 1}, i32);
  E.types = at::empty({R, Qnmax}, i32);
  E.lex = at::empty({R, Qomax, 5120}, f16);
  E.result = at::empty({R, Qomax}, f32);
  const int64_t Q = std::max(Qomax, Qnmax);
  E.ws_hn = at::empty({256, 5120}, f16);
  E.ws_X = at::empty({Qomax, WD}, f32);
  E.ws_F = at::empty({Qnmax, WD}, f32);
  E.ws_H = at::empty({Q, WD}, f16);
  E.ws_C = at::empty({Q, 4 * WD}, f32);
  E.ws_Hq = at::empty({Q, 3 * WD}, f16);
  E.ws_G = at::empty({Q, 4 * WD}, f16);
  E.ws_KV = at::empty({chunk, 2 * WD}, f16);
  E.ws_MC = at::empty({chunk, WD}, f16);
  E.ws_m = at::empty({NH * Q}, f32);
  E.ws_l = at::empty({NH * Q}, f32);
  E.ws_acc = at::empty({Q, WD}, f32);
  E.ws_ctx16 = at::empty({Qomax, 5120}, f16);
  E.ws_qv16 = at::empty({Qnmax, 5120}, f16);
  E.ws_gl16 = at::empty({5120}, f16);
  E.ws_Qp = at::empty({Qnmax, WD}, f32);
  E.ws_BF = at::empty({Qnmax, WD}, f32);
  E.ws_SN = at::empty({Qnmax, WD}, f32);
  E.ws_SM = at::empty({Qnmax, WD}, f32);
  E.ws_gp = at::empty({WD}, f32);
  E.ws_FN = at::empty({Qnmax, WD}, f32);
  E.ws_ON = at::empty({Qomax, WD}, f32);
  E.ws_cos = at::empty({Qomax}, f32);
  E.ws_prior = at::empty({Qomax}, f32);
  E.ws_res = at::empty({Qomax}, f32);
  E.req.assign(R, ReqSlot{});
  E.head_on = true;
}

// запрос в ячейку r: позиции -> спан, 1/длина спана, вопрос каждого варианта, границы, типы, lex
void req_begin(int64_t r, int64_t N, at::Tensor spanid, at::Tensor inv_len, at::Tensor qidx, at::Tensor qoff,
               at::Tensor types, at::Tensor lex) {
  TORCH_CHECK(E.head_on && r >= 0 && r < E.R, "clef: req_begin ячейка");
  const int Qn = (int)types.numel(), Qo = (int)qidx.numel(), S = (int)inv_len.numel();
  TORCH_CHECK(N > 0 && N <= E.Nmax && spanid.numel() == N, "clef: N");
  TORCH_CHECK(Qn > 0 && Qn <= E.Qnmax && Qo > 0 && Qo <= E.Qomax && S == Qn + Qo && S <= E.Smax, "clef: Qn/Qo/S");
  TORCH_CHECK(qoff.numel() == Qn + 1 && lex.size(0) == Qo && lex.size(1) == 5120, "clef: qoff/lex");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto cp = [&](at::Tensor dst, const at::Tensor& src, int64_t bytes) {
    at::Tensor s = src.contiguous();
    CLEF_CK(cudaMemcpyAsync(dst.data_ptr(), s.data_ptr(), bytes, cudaMemcpyHostToDevice, st));
  };
  cp(E.spanid[r], spanid.to(at::kInt), N * 4);
  cp(E.inv_len[r], inv_len.to(at::kFloat), S * 4);
  cp(E.qidx[r], qidx.to(at::kInt), Qo * 4);
  cp(E.qoff[r], qoff.to(at::kInt), (Qn + 1) * 4);
  cp(E.types[r], types.to(at::kInt), Qn * 4);
  cp(E.lex[r], lex.to(at::kHalf), (int64_t)Qo * 5120 * 2);
  CLEF_CK(cudaMemsetAsync(E.span[r].data_ptr(), 0, (int64_t)S * 5120 * 4, st));
  CLEF_CK(cudaMemsetAsync(E.glob[r].data_ptr(), 0, 5120 * 4, st));
  // копии с хоста асинхронны: исходники обязаны дожить до конца копии
  CLEF_CK(cudaStreamSynchronize(st));
  E.req[r] = ReqSlot{true, false, false, (int)N, Qn, Qo, S};
}

void req_end(int64_t r) {
  if (r >= 0 && r < E.R) E.req[r] = ReqSlot{};
}

void attn(const half* Qp, int ldq, int nq, const half* K, const half* V, int ldkv, int nk, bool init_state, bool fin,
          half* out, cudaStream_t st) {
  if (init_state) {
    fill_k<<<nblk(NH * nq), 256, 0, st>>>((float*)E.ws_m.data_ptr(), NH * nq, -INFINITY);
    fill_k<<<nblk(NH * nq), 256, 0, st>>>((float*)E.ws_l.data_ptr(), NH * nq, 0.f);
    fill_k<<<nblk((int64_t)nq * WD), 256, 0, st>>>((float*)E.ws_acc.data_ptr(), (int64_t)nq * WD, 0.f);
  }
  dim3 g(NH, (nq + 63) / 64);
  attn_upd_k<<<g, 64, 0, st>>>(Qp, ldq, K, V, ldkv, nk, nq, (float*)E.ws_m.data_ptr(), (float*)E.ws_l.data_ptr(),
                               (float*)E.ws_acc.data_ptr(), 0.125f);
  if (fin) attn_fin_k<<<nblk((int64_t)nq * WD), 256, 0, st>>>((float*)E.ws_acc.data_ptr(), (float*)E.ws_l.data_ptr(), nq, out);
}

#define GEMM(...) do { if (gemm_xwt(__VA_ARGS__) != CUBLAS_STATUS_SUCCESS) { fail("head: cublas"); return false; } } while (0)

// внимание к памяти запроса: Q [nq, 1024] (fp16), память кусками; mem_norm -- имя нормы или ""
bool attn_mem(const std::string& p, const std::string& mem_norm, const half* Qh, int nq, int r, half* out, cudaStream_t st) {
  const int N = E.req[r].N;
  const half* mem = (const half*)E.mem[r].data_ptr();
  const half* iw = W(p + "in_proj_weight");
  const half* ib = W(p + "in_proj_bias");
  half* KV = (half*)E.ws_KV.data_ptr();
  half* MC = (half*)E.ws_MC.data_ptr();
  for (int c0 = 0; c0 < N; c0 += E.chunk) {
    const int c = std::min(E.chunk, N - c0);
    const half* src = mem + (int64_t)c0 * WD;
    if (!mem_norm.empty()) {
      ln_k<half, half><<<c, 256, 0, st>>>(src, WD, W(mem_norm + ".weight"), W(mem_norm + ".bias"), MC, WD, WD, 1e-5f);
      src = MC;
    }
    GEMM(src, WD, iw + (int64_t)WD * WD, WD, KV, CUDA_R_16F, 2 * WD, c, 2 * WD, WD, 0.f);
    bias_half_k<<<nblk((int64_t)c * 2 * WD), 256, 0, st>>>(KV, 2 * WD, ib + WD, c, 2 * WD);
    attn(Qh, WD, nq, KV, KV + WD, 2 * WD, c, c0 == 0, c0 + c >= N, out, st);
  }
  return true;
}

// x (fp32 [n,1024]) += out_proj(attn(LN(x)->Q, память или сам x))
bool head_run(int64_t r) {
  if (!E.head_on || r < 0 || r >= E.R || !E.req[r].used) { fail("head: ячейка"); return false; }
  ReqSlot& R = E.req[r];
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int Qn = R.Qn, Qo = R.Qo, S = R.S;
  float* X = (float*)E.ws_X.data_ptr();
  float* F = (float*)E.ws_F.data_ptr();
  half* H = (half*)E.ws_H.data_ptr();
  float* C = (float*)E.ws_C.data_ptr();
  half* Hq = (half*)E.ws_Hq.data_ptr();
  half* G = (half*)E.ws_G.data_ptr();
  float* span = (float*)E.span[r].data_ptr();
  float* glob = (float*)E.glob[r].data_ptr();
  const half* lex = (const half*)E.lex[r].data_ptr();
  const int* qidx = (const int*)E.qidx[r].data_ptr();
  const int* qoff = (const int*)E.qoff[r].data_ptr();
  const int* types = (const int*)E.types[r].data_ptr();
  half* qv16 = (half*)E.ws_qv16.data_ptr();
  half* ctx16 = (half*)E.ws_ctx16.data_ptr();
  half* gl16 = (half*)E.ws_gl16.data_ptr();

  // средние по спанам (fp32 на месте) и их fp16-копии; порядок спанов: сначала Qn вопросов, затем Qo вариантов
  span_fin_k<<<nblk((int64_t)Qn * 5120), 256, 0, st>>>(span, (const float*)E.inv_len[r].data_ptr(), Qn, 5120, qv16);
  span_fin_k<<<nblk((int64_t)Qo * 5120), 256, 0, st>>>(span + (int64_t)Qn * 5120,
                                                       (const float*)E.inv_len[r].data_ptr() + Qn, Qo, 5120, ctx16);
  to_half_k<<<nblk(5120), 256, 0, st>>>(glob, gl16, 5120);

  // S0: запросы вариантов
  load_stage(0, st);
  GEMM(ctx16, 5120, W("option_context_projection.weight"), 5120, C, CUDA_R_32F, WD, Qo, WD, 5120, 0.f);
  GEMM(lex, 5120, W("option_lexical_projection.weight"), 5120, C, CUDA_R_32F, WD, Qo, WD, 5120, 1.f);
  float* Qp = (float*)E.ws_Qp.data_ptr();
  GEMM(qv16, 5120, W("option_question_projection.weight"), 5120, Qp, CUDA_R_32F, WD, Qn, WD, 5120, 0.f);
  add_rep_k<<<nblk((int64_t)Qo * WD), 256, 0, st>>>(C, Qp, qidx, X, Qo);

  // S1, S2: маршрутизация доказательств
  for (int li = 0; li < 2; ++li) {
    load_stage(1 + li, st);
    const std::string p = "evidence_layers." + std::to_string(li) + ".";
    ln_k<float, half><<<Qo, 256, 0, st>>>(X, WD, W(p + "query_norm.weight"), W(p + "query_norm.bias"), H, WD, WD, 1e-5f);
    GEMM(H, WD, W(p + "attention.in_proj_weight"), WD, C, CUDA_R_32F, WD, Qo, WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qo * WD), 256, 0, st>>>(C, WD, W(p + "attention.in_proj_bias"), Qo, WD, 0, nullptr, 0, Hq, WD);
    if (!attn_mem(p + "attention.", p + "memory_norm", Hq, Qo, (int)r, H, st)) return false;
    GEMM(H, WD, W(p + "attention.out_proj.weight"), WD, C, CUDA_R_32F, WD, Qo, WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qo * WD), 256, 0, st>>>(C, WD, W(p + "attention.out_proj.bias"), Qo, WD, 0, X, WD, nullptr, 0);
    ln_k<float, half><<<Qo, 256, 0, st>>>(X, WD, W(p + "feedforward_norm.weight"), W(p + "feedforward_norm.bias"), H, WD, WD, 1e-5f);
    GEMM(H, WD, W(p + "feedforward.0.weight"), WD, C, CUDA_R_32F, 4 * WD, Qo, 4 * WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qo * 4 * WD), 256, 0, st>>>(C, 4 * WD, W(p + "feedforward.0.bias"), Qo, 4 * WD, 1, nullptr, 0, G, 4 * WD);
    GEMM(G, 4 * WD, W(p + "feedforward.3.weight"), 4 * WD, C, CUDA_R_32F, WD, Qo, WD, 4 * WD, 0.f);
    epi_k<<<nblk((int64_t)Qo * WD), 256, 0, st>>>(C, WD, W(p + "feedforward.3.bias"), Qo, WD, 0, X, WD, nullptr, 0);
  }

  // S3: поля
  load_stage(3, st);
  float* BF = (float*)E.ws_BF.data_ptr();
  float* SM = (float*)E.ws_SM.data_ptr();
  float* SN = (float*)E.ws_SN.data_ptr();
  float* gp = (float*)E.ws_gp.data_ptr();
  GEMM(qv16, 5120, W("question_projection.weight"), 5120, BF, CUDA_R_32F, WD, Qn, WD, 5120, 0.f);
  summary_k<<<Qn, 256, 0, st>>>(X, BF, qoff, SM);
  ln_k<float, float><<<Qn, 256, 0, st>>>(SM, WD, W("option_summary_norm.weight"), W("option_summary_norm.bias"), SN, WD, WD, 1e-5f);
  GEMM(gl16, 5120, W("global_projection.weight"), 5120, gp, CUDA_R_32F, WD, 1, WD, 5120, 0.f);
  combine_k<<<nblk((int64_t)Qn * WD), 256, 0, st>>>(BF, SN, gp, W("type_embedding.weight"), types, F, Qn);

  // S4..S7: декодер полей (norm_first)
  for (int li = 0; li < 4; ++li) {
    load_stage(4 + li, st);
    const std::string p = "layers." + std::to_string(li) + ".";
    ln_k<float, half><<<Qn, 256, 0, st>>>(F, WD, W(p + "norm1.weight"), W(p + "norm1.bias"), H, WD, WD, 1e-5f);
    GEMM(H, WD, W(p + "self_attn.in_proj_weight"), WD, C, CUDA_R_32F, 3 * WD, Qn, 3 * WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qn * 3 * WD), 256, 0, st>>>(C, 3 * WD, W(p + "self_attn.in_proj_bias"), Qn, 3 * WD, 0, nullptr, 0, Hq, 3 * WD);
    attn(Hq, 3 * WD, Qn, Hq + WD, Hq + 2 * WD, 3 * WD, Qn, true, true, H, st);
    GEMM(H, WD, W(p + "self_attn.out_proj.weight"), WD, C, CUDA_R_32F, WD, Qn, WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qn * WD), 256, 0, st>>>(C, WD, W(p + "self_attn.out_proj.bias"), Qn, WD, 0, F, WD, nullptr, 0);
    ln_k<float, half><<<Qn, 256, 0, st>>>(F, WD, W(p + "norm2.weight"), W(p + "norm2.bias"), H, WD, WD, 1e-5f);
    GEMM(H, WD, W(p + "multihead_attn.in_proj_weight"), WD, C, CUDA_R_32F, WD, Qn, WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qn * WD), 256, 0, st>>>(C, WD, W(p + "multihead_attn.in_proj_bias"), Qn, WD, 0, nullptr, 0, Hq, WD);
    if (!attn_mem(p + "multihead_attn.", "", Hq, Qn, (int)r, H, st)) return false;
    GEMM(H, WD, W(p + "multihead_attn.out_proj.weight"), WD, C, CUDA_R_32F, WD, Qn, WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qn * WD), 256, 0, st>>>(C, WD, W(p + "multihead_attn.out_proj.bias"), Qn, WD, 0, F, WD, nullptr, 0);
    ln_k<float, half><<<Qn, 256, 0, st>>>(F, WD, W(p + "norm3.weight"), W(p + "norm3.bias"), H, WD, WD, 1e-5f);
    GEMM(H, WD, W(p + "linear1.weight"), WD, C, CUDA_R_32F, 4 * WD, Qn, 4 * WD, WD, 0.f);
    epi_k<<<nblk((int64_t)Qn * 4 * WD), 256, 0, st>>>(C, 4 * WD, W(p + "linear1.bias"), Qn, 4 * WD, 1, nullptr, 0, G, 4 * WD);
    GEMM(G, 4 * WD, W(p + "linear2.weight"), 4 * WD, C, CUDA_R_32F, WD, Qn, WD, 4 * WD, 0.f);
    epi_k<<<nblk((int64_t)Qn * WD), 256, 0, st>>>(C, WD, W(p + "linear2.bias"), Qn, WD, 0, F, WD, nullptr, 0);
  }

  // S8: оценка вариантов
  load_stage(8, st);
  float* FN = (float*)E.ws_FN.data_ptr();
  float* ON = (float*)E.ws_ON.data_ptr();
  float* cosv = (float*)E.ws_cos.data_ptr();
  float* prior = (float*)E.ws_prior.data_ptr();
  float* res = (float*)E.ws_res.data_ptr();
  ln_k<float, float><<<Qn, 256, 0, st>>>(F, WD, W("field_norm.weight"), W("field_norm.bias"), FN, WD, WD, 1e-5f);
  ln_k<float, float><<<Qo, 256, 0, st>>>(X, WD, W("option_norm.weight"), W("option_norm.bias"), ON, WD, WD, 1e-5f);
  feat_k<<<Qo, 256, 0, st>>>(FN, ON, qidx, G, cosv);
  prior_k<<<Qo, 256, 0, st>>>(lex, span, glob, qidx, 5120, E.ps, prior);
  GEMM(G, 4 * WD, W("residual_scorer.0.weight"), 4 * WD, C, CUDA_R_32F, WD, Qo, WD, 4 * WD, 0.f);
  epi_k<<<nblk((int64_t)Qo * WD), 256, 0, st>>>(C, WD, W("residual_scorer.0.bias"), Qo, WD, 1, nullptr, 0, H, WD);
  GEMM(H, WD, W("residual_scorer.3.weight"), WD, res, CUDA_R_32F, 1, Qo, 1, WD, 0.f);
  final_k<<<(Qo + 255) / 256, 256, 0, st>>>(prior, cosv, res, W("residual_scorer.3.bias"), E.js, E.sg, Qo,
                                            (float*)E.result[r].data_ptr());
  CLEF_CK(cudaGetLastError());
  R.ready = true;
  E.n_head++;
  return true;
}

// отвод: h [T, 5120] fp16 -- скрытые после финальной нормы всех строк шага (ранг с головой)
void tap(at::Tensor h) {
  if (!E.active || !E.head_on) return;
  if (h.dim() != 2 || h.size(1) != 5120 || h.scalar_type() != at::kHalf || h.stride(1) != 1 || !h.is_cuda()) {
    fail("tap: форма h");
    return;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int64_t ld = h.stride(0);
  half* hn = (half*)E.ws_hn.data_ptr();
  for (const Seg& sg : E.segs) {
    if (sg.req < 0 || sg.req >= E.R || !E.req[sg.req].used) { fail("tap: ячейка"); continue; }
    ReqSlot& R = E.req[sg.req];
    if (sg.row0 < 0 || sg.row0 + sg.rows > h.size(0) || sg.pos0 < 0 || sg.pos0 + sg.rows > R.N) {
      fail("tap: сегмент");
      R.bad = true;
      continue;
    }
    for (int i0 = 0; i0 < sg.rows; i0 += 256) {
      const int n = std::min(256, sg.rows - i0);
      const half* src = (const half*)h.data_ptr() + (int64_t)(sg.row0 + i0) * ld;
      ln_k<half, half><<<n, 256, 0, st>>>(src, (int)ld, (const half*)E.hn_w.data_ptr(), (const half*)E.hn_b.data_ptr(),
                                          hn, 5120, 5120, 1e-5f);
      half* memr = (half*)E.mem[sg.req].data_ptr() + (int64_t)(sg.pos0 + i0) * WD;
      if (gemm_xwt(hn, 5120, (const half*)E.mp_w.data_ptr(), 5120, memr, CUDA_R_16F, WD, n, WD, 5120, 0.f) !=
          CUBLAS_STATUS_SUCCESS) {
        fail("tap: cublas");
        R.bad = true;
        break;
      }
      const int pos = sg.pos0 + i0;
      const int last = (pos + n == R.N) ? n - 1 : -1;
      span_add_k<<<n, 256, 0, st>>>(hn, (const int*)E.spanid[sg.req].data_ptr() + pos, (float*)E.span[sg.req].data_ptr(),
                                    (float*)E.glob[sg.req].data_ptr(), 5120, last);
    }
    E.n_tap++;
    if (sg.fin && !R.bad) {
      if (sg.pos0 + sg.rows != R.N) { fail("tap: конец не на N"); R.bad = true; continue; }
      if (!head_run(sg.req)) R.bad = true;
    }
  }
}

at::Tensor result(int64_t r) {
  TORCH_CHECK(E.head_on && r >= 0 && r < E.R, "clef: result ячейка");
  return E.result[r].narrow(0, 0, std::max(E.req[r].Qo, 0));
}

bool ready(int64_t r) { return E.head_on && r >= 0 && r < E.R && E.req[r].ready && !E.req[r].bad; }
bool step_bad() { return E.step_bad; }

std::map<std::string, int64_t> stats() {
  return {{"delta", E.n_delta}, {"gemm", E.n_gemm}, {"copy", E.n_copy}, {"err", E.n_err},
          {"steps", E.n_steps}, {"tap", E.n_tap}, {"head", E.n_head},
          {"slot_bytes", E.slot_bytes}, {"nslot", E.nslot}, {"layers", (int64_t)E.layers.size()},
          {"streamed", (int64_t)E.streamed.size()}, {"keys", (int64_t)E.keys.size()},
          {"keymiss", E.n_keymiss}, {"active", (int64_t)E.active}};
}
std::string last_error() { return E.last_err; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_delta", &init_delta);
  m.def("add_layer", &add_layer);
  m.def("finalize_delta", &finalize_delta);
  m.def("set_step", &set_step);
  m.def("delta", &delta);
  m.def("bind", &bind);
  m.def("delta_keyed", &delta_keyed);
  m.def("init_head", &init_head);
  m.def("req_begin", &req_begin);
  m.def("req_end", &req_end);
  m.def("tap", &tap);
  m.def("head_run", &head_run);
  m.def("result", &result);
  m.def("ready", &ready);
  m.def("step_bad", &step_bad);
  m.def("stats", &stats);
  m.def("last_error", &last_error);
}
