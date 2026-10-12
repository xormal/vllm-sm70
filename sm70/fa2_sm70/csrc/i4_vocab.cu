// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ================================================================================================
// ДВУХСТУПЕНЧАТЫЙ АРГМАКС СЛОВАРЯ ЧЕРНОВИКА (записка 25 §238-§239).
// ================================================================================================
// ЗАЧЕМ. Черновик MTP идёт цепью из k проходов, и на каждом проецирует скрытое состояние в
// ВЕСЬ словарь: 248320 x 5120 fp16 -- 1.27 ГБ на ранг, 1.41 мс пола чтения. За шаг одна и та
// же матрица читается ЧЕТЫРЕЖДЫ (раз целью, трижды черновиком) -- 7.3 мс из 52.
//
// НО ЧЕРНОВИКУ НУЖЕН НЕ ЛОГИТ, А АРГМАКС. Значит достаточно посчитать ПРИБЛИЖЁННЫЕ счёты по
// дешёвому представлению, взять короткий список и досчитать точно только его.
//
// ПОЧЕМУ int4, А НЕ ЭСКИЗ. Замерено офлайн на 700 НАСТОЯЩИХ скрытых состояниях черновика:
//   случайная проекция ранга 512 (чтение 1/10)  -- истинный аргмакс в top-16 лишь в 21.4 %
//   int8 по строке              (чтение 1/2)    -- 99.86 % уже при N=1
//   int4 по строке              (чтение 1/4)    -- 92.71 % при N=1 и 100.00 % при N=16
// Эскиз отвергнут числом: у логитов по 124 тысячам строк зазор между первым и вторым мал,
// и проекция его не сохраняет. int4 + досчитка top-16 даёт СОВПАДЕНИЕ 100 %.
//
// ЦЕНА ПАМЯТИ РЕШИЛА ВЫБОР: на боевом свободно 651 МиБ на карту; int4 -- 303 МиБ, влезает,
// int8 -- 606 МиБ, нет. То есть более дешёвый по чтению вариант оказался и единственно
// возможным по памяти.
//
// РАСКЛАДКА. Веса пакуются ПО СТРОКЕ: два знаковых 4-битных значения в байт, младшая
// половина -- чётный столбец. Масштаб один на строку (fp16). Хранение: [V, K/2] uint8.
// ================================================================================================
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>

namespace {

// Упаковка: [V, K] fp16 -> коды [V, K/2] uint8 и масштабы [V] fp16.
__global__ void i4_pack_k(const __half* __restrict__ W, uint8_t* __restrict__ Q,
                          __half* __restrict__ S, int V, int K) {
  const int r = blockIdx.x;
  if (r >= V) return;
  const __half* row = W + (long)r * K;
  // 1) максимум модуля по строке -- редукция блоком
  __shared__ float gmax;
  float mx = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) mx = fmaxf(mx, fabsf(__half2float(row[i])));
  for (int shf = 16; shf; shf >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, shf));
  __shared__ float warp_red[32];
  if ((threadIdx.x & 31) == 0) warp_red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot = 0.f;
    for (int i = 0; i < (blockDim.x + 31) / 32; ++i) tot = fmaxf(tot, warp_red[i]);
    gmax = tot / 7.0f;                 // int4 знаковый: -7..7
    S[r] = __float2half(gmax);
  }
  __syncthreads();
  const float scale = gmax > 0.f ? gmax : 1.f;
  // 2) кодирование пар
  uint8_t* dst = Q + (long)r * (K / 2);
  for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
    int a = __float2int_rn(__half2float(row[2 * i]) / scale);
    int b = __float2int_rn(__half2float(row[2 * i + 1]) / scale);
    a = max(-7, min(7, a)); b = max(-7, min(7, b));
    dst[i] = (uint8_t)((a & 0xF) | ((b & 0xF) << 4));
  }
}

// Приближённые счёты: [M, K] fp16 x коды [V, K/2] -> [M, V] fp16.
// ВАРП НА СТРОКУ, а не блок. Первая редакция ставила блок из 256 нитей на одну строку словаря
// и делала внутри цикла по M полную блочную редукцию с __syncthreads -- вышло 22 % крыши
// чтения (1.48 мс против пола 0.33). Варп сводит своё частичное сумму сдвигами, барьеров нет
// вовсе, и восемь строк идут одним блоком.
template <int MFIX>
__global__ void i4_scores_k(const __half* __restrict__ h, const uint8_t* __restrict__ Q,
                            const __half* __restrict__ S, __half* __restrict__ out,
                            int M, int V, int K) {
  extern __shared__ __half sh_in[];                 // [M][K] -- вход, общий на блок
  for (int i = threadIdx.x; i < M * K; i += blockDim.x) sh_in[i] = h[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int npair4 = (K / 2) / 4;
  for (long r = (long)blockIdx.x * warps + warp; r < V; r += (long)gridDim.x * warps) {
    const uchar4* q4 = reinterpret_cast<const uchar4*>(Q + r * (long)(K / 2));
    float acc[MFIX];
#pragma unroll
    for (int m = 0; m < MFIX; ++m) acc[m] = 0.f;
    for (int i = lane; i < npair4; i += 32) {
      uchar4 c = q4[i];
      const uint8_t bb4[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int a = (int)(bb4[j] & 0xF); a = (a > 7) ? a - 16 : a;
        int b2 = (int)(bb4[j] >> 4); b2 = (b2 > 7) ? b2 - 16 : b2;
        const int kk = ((i * 4 + j) << 1);
        // ЧИСЛО СТРОК -- ШАБЛОННОЕ. При переменном M компилятор не разворачивает этот цикл,
        // и на M=1 ядро шло 34 % крыши чтения. Специализация по MFIX разворачивает его в
        // одну-две операции и снимает лишние чтения разделяемой.
#pragma unroll
        for (int m = 0; m < MFIX; ++m) {
          acc[m] += __half2float(sh_in[m * K + kk]) * (float)a
                  + __half2float(sh_in[m * K + kk + 1]) * (float)b2;
        }
      }
    }
    const float sc = __half2float(S[r]);
#pragma unroll
    for (int m = 0; m < MFIX; ++m) {
      float partial = acc[m];
#pragma unroll
      for (int shf = 16; shf; shf >>= 1) partial += __shfl_down_sync(0xffffffff, partial, shf);
      if (lane == 0) out[(long)m * V + r] = __float2half(partial * sc);
    }
  }
}

}  // namespace

std::vector<at::Tensor> i4_pack(at::Tensor W) {
  TORCH_CHECK(W.is_cuda() && W.dim() == 2 && W.scalar_type() == at::kHalf,
              "i4_pack: ждём двумерный fp16 на карте");
  TORCH_CHECK(W.size(1) % 8 == 0, "i4_pack: K обязан быть кратен 8");
  auto W_ = W.contiguous();
  const int V = W_.size(0), K = W_.size(1);
  auto Q = at::empty({V, K / 2}, W_.options().dtype(at::kByte));
  auto S = at::empty({V}, W_.options());
  i4_pack_k<<<V, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)W_.data_ptr(), (uint8_t*)Q.data_ptr(), (__half*)S.data_ptr(), V, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {Q, S};
}

at::Tensor i4_scores(at::Tensor h, at::Tensor Q, at::Tensor S) {
  TORCH_CHECK(h.is_cuda() && h.dim() == 2 && h.scalar_type() == at::kHalf, "i4_scores: h fp16 [M,K]");
  auto h_ = h.contiguous();
  const int M = h_.size(0), K = h_.size(1), V = Q.size(0);
  TORCH_CHECK(Q.size(1) == K / 2, "i4_scores: несогласованные K");
  TORCH_CHECK(M >= 1 && M <= 4, "i4_scores: M от 1 до 4");
  auto out = at::empty({M, V}, h_.options());
  const int thr = 256;
  const size_t shmem = (size_t)M * K * sizeof(__half);
  const int warps = thr >> 5;
  int grid = (V + warps - 1) / warps;
  if (grid > 65535) grid = 65535;
  auto пуск = [&](auto MC) {
    i4_scores_k<decltype(MC)::value><<<grid, thr, shmem, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)h_.data_ptr(), (const uint8_t*)Q.data_ptr(), (const __half*)S.data_ptr(),
        (__half*)out.data_ptr(), M, V, K);
  };
  if (M == 1) пуск(std::integral_constant<int, 1>{});
  else if (M == 2) пуск(std::integral_constant<int, 2>{});
  else if (M == 3) пуск(std::integral_constant<int, 3>{});
  else пуск(std::integral_constant<int, 4>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &i4_pack, "упаковать голову словаря в int4 по строке");
  m.def("scores", &i4_scores, "приближённые счёты по int4-раскладке");
}
