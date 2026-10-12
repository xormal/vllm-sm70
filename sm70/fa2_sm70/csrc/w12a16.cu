// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ==================================================================================================
// W12A16: ВЕС В 12 РАЗРЯДАХ, ПОТРЕБЛЯЕМЫЙ УПАКОВАННЫМ. Пара к квантователю tools/quantize_iN.py.
// ==================================================================================================
//
// ЗАЧЕМ. В боевом слепке Huihui-Qwen3.6-27B-INT8 веса `self_attn.{q,k,v}_proj` шестнадцати слоёв
// полного внимания НЕ КВАНТОВАНЫ -- лежат в BF16, 2.349 ГБ, и читаются каждый шаг декода. Ход в
// int12 здесь НЕ РАЗМЕН: замерено на этих самых тензорах, ошибка 6.10e-04 против собственной
// зернистости BF16 1.95e-03 -- то есть втрое ТОЧНЕЕ хранимого, при 0.75 байт. Причина не в хитрости,
// а в устранении дублирования: BF16 держит экспоненту на каждом весе, тогда как динамический
// диапазон уже вынесен в масштаб группы.
//
// ЧТО ЗДЕСЬ УНАСЛЕДОВАНО ЗАМЕРОМ, А НЕ СКОПИРОВАНО (см. шапку w8a16_gemv.cu):
//   1. РАСПАКОВКА ОБЯЗАНА БЫТЬ СЛИТА С ЧТЕНИЕМ. Отдельный проход распаковки был замерен на int8 и
//      дал 0.18x от НЕквантованного пути: он пишет плотный вес в память и читает обратно. Слияние
//      здесь -- не оптимизация, а разница между выигрышем и крупным проигрышем.
//   2. РАЗРЕЗ ПО k -- ЭТО И ЕСТЬ ЯДРО, а не украшение. Одна колонка на нить даёт N/THREADS блоков =
//      20 на 80 SM: три четверти машины простаивает, замерено 48 ГБ/с из 841.
//   3. MT (строк x за проход) существует потому, что при MT=1 вес перечитывается на каждую строку.
//      Массивы индексируются ТОЛЬКО полностью развёрнутыми циклами -- иначе они уедут в локальную
//      память и молча снимут весь выигрыш.
//
// ЧЕМ ОТЛИЧАЕТСЯ ОТ int8-ПУТИ, И ЭТО В ПОЛЬЗУ int12:
//   * КВАНТОВАНИЕ СИММЕТРИЧНОЕ. У GPTQ эпилог группы обязан считать сумму x по группе, чтобы
//     свернуть нулевую точку: s*(Sum q*x - z*Sum x). Здесь z нет, эпилог = ОДИН множитель. Целая
//     фаза ядра (отдельный проход по sx, барьер, разделяемая под gs) исчезает вместе с ней.
//   * Смещение 2048 (хранимое беззнаковым) снимается ОДНИМ вычитанием на ЗНАЧЕНИЕ ВЕСА, а не на
//     пару (строка, значение) -- то есть его цена не растёт с MT.
//
// РАСКЛАДКА (её задаёт ЧИТАТЕЛЬ, см. шапку quantize_iN.py):
//   qw     [K/8, 3, N] int32 -- 8 подряд идущих k ОДНОГО столбца = 96 разрядов = ровно 3 слова,
//                               разложенные по трём плоскостям. Варп = 32 подряд идущих слова = 128 Б.
//   scales [K/G, N] fp16
//   w[k,n] = (q[k,n]) * s[k/G,n],  q знаковое в [-2047, 2047], хранится как q+2048.
//
// [ВОССТАНОВЛЕНО 16.08.2026 -- ЧИТАТЬ, ПРЕЖДЕ ЧЕМ ДОВЕРЯТЬ]
// Файл был повреждён МОЕЙ ЖЕ правкой: скрипт резал по индексу подстроки `// =====`, а таких
// разделителей в шапке несколько, и срез снёс включения, распаковщик и хвост пускача GEMV. Копий
// не нашлось нигде -- дерево ядер НЕ ПОД ВЕРСИЕЙ. Шапка, `unpack8_i12`, `w12a16_gemv_kernel` и
// `w12_cast_kernel` восстановлены дословно из истории заседания; `w12_dequant_kernel` и хвост
// `w12a16_gemv` ПЕРЕСОБРАНЫ ПО КОНТРАКТУ вызывающего кода и проверены заново -- сверкой с
// независимым развалом упаковки (`tools/quantize_iN.py::разобрать`).
// ВЫВОД ДОРОЖЕ ФАЙЛА: правка исходника скриптом по НЕуникальному якорю -- это резание вслепую.
#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "w12a16_hmma.cuh"
#include "volta_gemm_f16.cuh"

#define XSMEM 4096
constexpr int xtile_of(int MT) { int t = XSMEM / MT; return t > 1024 ? 1024 : (t < 128 ? 128 : t); }

// Разбор трёх слов в восемь значений. Вынесен отдельно ровно потому, что этот же разбор обязан
// стоять и в ядре GEMV, и в распаковщике для большого M: два разошедшихся разбора -- это тихая
// неверность на половине форм, а не падение.
__device__ __forceinline__ void unpack8_i12(unsigned w0, unsigned w1, unsigned w2, float q[8]) {
  q[0] = (float)( w0         & 0xFFFu)                              - 2048.f;
  q[1] = (float)((w0 >> 12)  & 0xFFFu)                              - 2048.f;
  q[2] = (float)(((w0 >> 24) & 0xFFu) | ((w1 & 0xFu) << 8))         - 2048.f;
  q[3] = (float)((w1 >>  4)  & 0xFFFu)                              - 2048.f;
  q[4] = (float)((w1 >> 16)  & 0xFFFu)                              - 2048.f;
  q[5] = (float)(((w1 >> 28) & 0xFu)  | ((w2 & 0xFFu) << 4))        - 2048.f;
  q[6] = (float)((w2 >>  8)  & 0xFFFu)                              - 2048.f;
  q[7] = (float)((w2 >> 20)  & 0xFFFu)                              - 2048.f;
}

template <int THREADS, int MT, bool SPLIT>
__global__ void w12a16_gemv_kernel(const __half* __restrict__ x,
                                   const int* __restrict__ qw,
                                   const __half* __restrict__ scales,
                                   __half* __restrict__ y, float* __restrict__ ypart,
                                   int M, int N, int K, int G, int groups_per_split, int m0) {
  constexpr int XT = xtile_of(MT);
  static_assert(XT % 8 == 0, "плитка x кратна 8: шаг распаковки");
  __shared__ float sx[MT][XT];
  const int n = blockIdx.x * THREADS + threadIdx.x;
  const bool live = (n < N);
  const int mrows = min(MT, M - m0);

  const int kbeg = SPLIT ? blockIdx.y * groups_per_split * G : 0;
  const int kend = SPLIT ? min(K, kbeg + groups_per_split * G) : K;
  if (kbeg >= kend) return;

  float acc[MT];
#pragma unroll
  for (int mm = 0; mm < MT; ++mm) acc[mm] = 0.f;

  for (int k0 = kbeg; k0 < kend; k0 += XT) {
    const int len = min(XT, kend - k0);
    __syncthreads();
    // x кладётся как fp32: конверсия half->float одного и того же значения иначе повторялась бы N раз
    // (по разу на выходной столбец). Замерено на int8-пути: 1 LDS.128 + 4 FFMA вместо ~13 команд.
    for (int t = threadIdx.x; t < len * mrows; t += THREADS) {
      const int mm = t / len, kk = t - mm * len;
      sx[mm][kk] = __half2float(x[(long)(m0 + mm) * K + k0 + kk]);
    }
    __syncthreads();
    if (!live) continue;
    // Внешний цикл по ГРУППАМ, внутренний по её ключам: масштаб читается раз на 128 ключей, и
    // внутренний цикл разворачивается (ветвь смены группы внутри k-цикла разворот запрещает).
    for (int gk = 0; gk < len; gk += G) {
      const int g = (k0 + gk) / G;
      const float s = __half2float(scales[(long)g * N + n]);
      const int glen = min(G, len - gk);
      float qa[MT];
#pragma unroll
      for (int mm = 0; mm < MT; ++mm) qa[mm] = 0.f;
#pragma unroll 2
      for (int k = 0; k < glen; k += 8) {
        const long b = (long)((k0 + gk + k) >> 3) * 3;
        const unsigned w0 = (unsigned)qw[(b + 0) * N + n];
        const unsigned w1 = (unsigned)qw[(b + 1) * N + n];
        const unsigned w2 = (unsigned)qw[(b + 2) * N + n];
        float q[8];
        unpack8_i12(w0, w1, w2, q);
#pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
          const float4 a0 = *reinterpret_cast<const float4*>(&sx[mm][gk + k]);
          const float4 a1 = *reinterpret_cast<const float4*>(&sx[mm][gk + k + 4]);
          qa[mm] = fmaf(a0.x, q[0], qa[mm]); qa[mm] = fmaf(a0.y, q[1], qa[mm]);
          qa[mm] = fmaf(a0.z, q[2], qa[mm]); qa[mm] = fmaf(a0.w, q[3], qa[mm]);
          qa[mm] = fmaf(a1.x, q[4], qa[mm]); qa[mm] = fmaf(a1.y, q[5], qa[mm]);
          qa[mm] = fmaf(a1.z, q[6], qa[mm]); qa[mm] = fmaf(a1.w, q[7], qa[mm]);
        }
      }
#pragma unroll
      for (int mm = 0; mm < MT; ++mm) acc[mm] = fmaf(s, qa[mm], acc[mm]);   // симметрично: без z
    }
  }
  if (!live) return;
#pragma unroll
  for (int mm = 0; mm < MT; ++mm) {
    if (mm >= mrows) break;
    // [ДЕТЕРМИНИРОВАННЫЙ СВОД] Атомарное сложение float идёт в порядке прихода блоков и делает
    // ответ неповторяемым (отчёт падений §8-кватер: дрожание логитов +-0.003, на вырожденных
    // выдачах переворачивает выбор). Пишем в СВОЙ слот, свод -- фиксированным порядком в эпилоге.
    if (SPLIT) ypart[(long)blockIdx.y * M * N + (long)(m0 + mm) * N + n] = acc[mm];
    else       y[(long)(m0 + mm) * N + n] = __float2half(acc[mm]);
  }
}

// Свод расщеплений в ФИКСИРОВАННОМ порядке -- замена atomicAdd (см. тело ядра).
__global__ void w12_reduce_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                  long n, int ks) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = 0.f;
  for (int k = 0; k < ks; ++k) a += src[(long)k * n + i];
  dst[i] = __float2half(a);
}

__global__ void w12_cast_kernel(const float* __restrict__ src, __half* __restrict__ dst, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

// РАСПАКОВЩИК ДЛЯ БОЛЬШОГО M. При префилле плитка веса переиспользуется тысячами строк, поэтому
// связывает СЧЁТ, а не чтение веса, и правильный ход -- отдать работу плотному тензорному пути.
// Отдельный проход здесь ЗАКОНЕН ровно по той причине, по которой он запрещён на декоде: его цена
// (один проход по весу) делится на M строк. Буфер -- ОДИН на процесс, переиспользуемый: экономия
// памяти сохраняется, потому что мы не держим 80 плотных копий, а одну временную.
// Выход -- ПЛОТНЫЙ [K, N] fp16: n остаётся быстрой осью, как и в упаковке, значит запись
// коалесцирована, а читателем может быть обычный плотный GEMM `x[M,K] @ Wd[K,N]`.
__global__ void w12_dequant_kernel(const int* __restrict__ qw, const __half* __restrict__ sc,
                                   __half* __restrict__ out, int N, int K, int G) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  for (int kq = blockIdx.y; kq < K / 8; kq += gridDim.y) {
    const long b = (long)kq * 3;
    const unsigned w0 = (unsigned)qw[(b + 0) * N + n];
    const unsigned w1 = (unsigned)qw[(b + 1) * N + n];
    const unsigned w2 = (unsigned)qw[(b + 2) * N + n];
    float q[8];
    unpack8_i12(w0, w1, w2, q);
    const int k0 = kq * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      out[(long)(k0 + i) * N + n] =
          __float2half(q[i] * __half2float(sc[(long)((k0 + i) / G) * N + n]));
  }
}

torch::Tensor w12a16_gemv(torch::Tensor x, torch::Tensor qw_t, torch::Tensor scales,
                          int64_t group_size) {
  TORCH_CHECK(x.is_cuda() && qw_t.is_cuda(), "w12a16: нужны тензоры CUDA");
  TORCH_CHECK(qw_t.dim() == 3 && qw_t.size(1) == 3, "w12a16: qw [K/8, 3, N]");
  TORCH_CHECK(scales.dim() == 2, "w12a16: scales [K/G, N]");
  auto xc = x.contiguous(), qw = qw_t.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(2), K = (int)qw.size(0) * 8;
  const int M = (int)(xc.numel() / xc.size(-1));
  const int G = (int)group_size;
  TORCH_CHECK(xc.size(-1) == K, "w12a16: K не сходится");
  TORCH_CHECK(G % 8 == 0, "w12a16: группа кратна 8 (шаг распаковки)");
  auto y = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(xc.device()));
  constexpr int THREADS = 256;
  const int nblk = (N + THREADS - 1) / THREADS;
  const int ngroups = (K + G - 1) / G;
  int ksplit = max(1, min(ngroups, (80 * 8 + nblk - 1) / nblk));
  const int gps = (ngroups + ksplit - 1) / ksplit;
  ksplit = (ngroups + gps - 1) / gps;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  const auto* sp = reinterpret_cast<const __half*>(sc.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  torch::Tensor part;
  float* pp = nullptr;
  if (ksplit > 1) {
    // СЛОТ НА КАЖДОЕ расщепление -- см. пояснение у записи в теле ядра.
    part = torch::empty({(long)ksplit, (long)M, (long)N},
                        torch::dtype(torch::kFloat32).device(xc.device()));
    pp = part.data_ptr<float>();
  }
  const dim3 grid = (ksplit > 1) ? dim3(nblk, ksplit) : dim3(nblk);

#define W12_LAUNCH(MT)                                                                       \
  do {                                                                                       \
    if (ksplit > 1)                                                                          \
      w12a16_gemv_kernel<THREADS, MT, true><<<grid, THREADS, 0, stream>>>(                    \
          xp, qw.data_ptr<int>(), sp, nullptr, pp, M, N, K, G, gps, m0);                      \
    else                                                                                     \
      w12a16_gemv_kernel<THREADS, MT, false><<<grid, THREADS, 0, stream>>>(                   \
          xp, qw.data_ptr<int>(), sp, yp, nullptr, M, N, K, G, 0, m0);                        \
  } while (0)

  // Дробим M самой крупной плиткой, что влезает: каждый кусок -- отдельный ПОЛНЫЙ проход по весу,
  // поэтому упакованное чтение окупается лишь пока кусков немного (см. порог в w8a16_gemv.cu).
  for (int m0 = 0; m0 < M; ) {
    const int rem = M - m0;
    if      (rem >= 16) { W12_LAUNCH(16); m0 += 16; }
    else if (rem >=  8) { W12_LAUNCH(8);  m0 +=  8; }
    else if (rem >=  4) { W12_LAUNCH(4);  m0 +=  4; }
    else if (rem >=  2) { W12_LAUNCH(2);  m0 +=  2; }
    else                { W12_LAUNCH(1);  m0 +=  1; }
  }
#undef W12_LAUNCH
  if (ksplit > 1) {
    const long tot = (long)M * N;
    w12_reduce_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(pp, yp, tot, ksplit);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Пишет В ПЕРЕДАННЫЙ буфер, а не выделяет: буфер обязан быть ОДИН на процесс, иначе плотные копии
// съедят ровно то, что мы освободили упаковкой.
void w12_dequant_into(torch::Tensor qw_t, torch::Tensor scales, torch::Tensor out,
                      int64_t group_size) {
  TORCH_CHECK(qw_t.dim() == 3 && qw_t.size(1) == 3, "w12_dequant: qw [K/8, 3, N]");
  auto qw = qw_t.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(2), K = (int)qw.size(0) * 8;
  TORCH_CHECK(out.numel() >= (long)K * N && out.scalar_type() == torch::kFloat16,
              "w12_dequant: буфер мал или не fp16");
  TORCH_CHECK(out.is_contiguous(), "w12_dequant: буфер обязан быть плотным");
  const int G = (int)group_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 blk(256), grid((N + 255) / 256, min(K / 8, 1024));
  w12_dequant_kernel<<<grid, blk, 0, stream>>>(
      qw.data_ptr<int>(), reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), N, K, G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ==================================================================================================
// ТЕНЗОРНЫЙ ПУТЬ: ГЕОМЕТРИЯ КАК ОСЬ ВЫБОРА, А НЕ КОНСТАНТА
// ==================================================================================================
// У turbomind набор воплощений перебирается САМОНАСТРОЙКОЙ (замерено по ленте: 6383 пробных пуска
// в одном вызове, `MMA_Map<8..16, 128..256, 32..64>` и расщепление K 1..16). Мы этот перебор не
// повторяем на подъёме -- он стоил 406 мс, -- но саму ОСЬ забираем: воплощение выбирается снаружи,
// чтобы его ранжировал ЗАМЕР, а не автор. Дважды за один заход выбор автора оказывался неверным
// (конвейер и ширина плитки), поэтому осей здесь пять: варпы, ширина по N, глубина по K, конвейер
// и принуждённая занятость MINB.
//
// Ограничения, отсекающие незаконные точки: BN кратно WN*16; BK кратно 8 и делит группу 128;
// разделяемая (NBUF*(MT*LDA + KD*LDB)*2 + мелочь) не выше 48 КиБ статического потолка sm_70.
struct W12Geom { int bn, wn, bk, pipe, minb; };
static const W12Geom kW12Geoms[] = {
    {128, 4, 32, 1, 1}, {128, 4, 32, 0, 1}, {128, 4, 64, 0, 1}, {128, 4, 16, 1, 1},
    { 64, 4, 32, 1, 1}, { 64, 4, 32, 0, 1}, { 64, 4, 64, 0, 1}, { 64, 4, 16, 1, 1},
    {256, 4, 16, 1, 1}, {256, 4, 16, 0, 1}, {256, 4, 32, 0, 1},
    {128, 8, 32, 1, 1}, {128, 8, 32, 0, 1}, {128, 8, 64, 0, 1}, {128, 8, 16, 1, 1},
    {256, 8, 32, 0, 1}, {256, 8, 16, 1, 1}, {256, 8, 16, 0, 1},
    // ТЕ ЖЕ ТОЧКИ, НО С ПРИНУЖДЁННОЙ ЗАНЯТОСТЬЮ: ptxas обязан урезать регистры.
    {128, 4, 32, 1, 3}, {128, 4, 32, 0, 4}, { 64, 4, 32, 0, 6}, {128, 4, 16, 1, 4},
    {128, 8, 32, 0, 3}, {256, 8, 16, 1, 2}, {256, 4, 16, 1, 2}, { 64, 4, 16, 1, 6},
};
constexpr int kW12GeomCount = (int)(sizeof(kW12Geoms) / sizeof(kW12Geoms[0]));

int64_t w12a16_hmma_geom_count() { return kW12GeomCount; }
std::vector<int64_t> w12a16_hmma_geom(int64_t i) {
  TORCH_CHECK(i >= 0 && i < kW12GeomCount, "w12a16 hmma: нет воплощения ", i);
  return {kW12Geoms[i].bn, kW12Geoms[i].wn, kW12Geoms[i].bk, kW12Geoms[i].pipe,
          kW12Geoms[i].minb};
}

torch::Tensor w12a16_hmma_gemm(torch::Tensor x, torch::Tensor qw_t, torch::Tensor scales,
                               int64_t group_size, int64_t geom, int64_t diag) {
  TORCH_CHECK(x.is_cuda() && qw_t.is_cuda(), "w12a16 hmma: нужны тензоры CUDA");
  TORCH_CHECK(qw_t.dim() == 3 && qw_t.size(1) == 3, "w12a16 hmma: qw [K/8, 3, N]");
  auto xc = x.contiguous(), qw = qw_t.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(2), K = (int)qw.size(0) * 8;
  const int M = (int)(xc.numel() / xc.size(-1));
  const int G = (int)group_size;
  TORCH_CHECK(xc.size(-1) == K, "w12a16 hmma: K не сходится");
  TORCH_CHECK(G == 128, "w12a16 hmma: группа 128");
  TORCH_CHECK(geom >= 0 && geom < kW12GeomCount, "w12a16 hmma: нет воплощения ", geom);
  auto y = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(xc.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  const auto* sp = reinterpret_cast<const __half*>(sc.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  // Суммы x по группам считаются ОДИН раз на вызов, а не в каждом блоке сетки.
  auto sxg = torch::empty({M, K / G}, torch::dtype(torch::kFloat32).device(xc.device()));
  fa2_sm70::w12a16::w12_sx_kernel<<<dim3((K / G + 63) / 64, M), 64, 0, stream>>>(
      xp, sxg.data_ptr<float>(), M, K, G);
  const float* sxp = sxg.data_ptr<float>();

#define W12_ONE(BN, WN, BK, P, MB, D)                                                         \
  fa2_sm70::w12a16::w12a16_hmma<BN, WN, 128, BK, P, MB, D>                                    \
      <<<dim3((N + (BN) - 1) / (BN), (M + 15) / 16), (WN) * 32, 0, stream>>>(                 \
          xp, qw.data_ptr<int>(), sp, sxp, yp, M, N, K)
#define W12_DIAG(BN, WN, BK, P, MB)                                                           \
  switch (diag) {                                                                             \
    case 0: W12_ONE(BN, WN, BK, P, MB, 0); break;                                             \
    case 1: W12_ONE(BN, WN, BK, P, MB, 1); break;                                             \
    case 2: W12_ONE(BN, WN, BK, P, MB, 2); break;                                             \
    case 3: W12_ONE(BN, WN, BK, P, MB, 3); break;                                             \
    case 4: W12_ONE(BN, WN, BK, P, MB, 4); break;                                             \
    case 5: W12_ONE(BN, WN, BK, P, MB, 5); break;                                             \
    default: TORCH_CHECK(false, "w12a16 hmma: неизвестный diag ", diag);                       \
  }
#define W12_CASE(I, BN, WN, BK, P, MB) case I: { W12_DIAG(BN, WN, BK, P, MB); break; }
  switch (geom) {
    W12_CASE( 0, 128, 4, 32, 1, 1) W12_CASE( 1, 128, 4, 32, 0, 1) W12_CASE( 2, 128, 4, 64, 0, 1)
    W12_CASE( 3, 128, 4, 16, 1, 1) W12_CASE( 4,  64, 4, 32, 1, 1) W12_CASE( 5,  64, 4, 32, 0, 1)
    W12_CASE( 6,  64, 4, 64, 0, 1) W12_CASE( 7,  64, 4, 16, 1, 1) W12_CASE( 8, 256, 4, 16, 1, 1)
    W12_CASE( 9, 256, 4, 16, 0, 1) W12_CASE(10, 256, 4, 32, 0, 1) W12_CASE(11, 128, 8, 32, 1, 1)
    W12_CASE(12, 128, 8, 32, 0, 1) W12_CASE(13, 128, 8, 64, 0, 1) W12_CASE(14, 128, 8, 16, 1, 1)
    W12_CASE(15, 256, 8, 32, 0, 1) W12_CASE(16, 256, 8, 16, 1, 1) W12_CASE(17, 256, 8, 16, 0, 1)
    W12_CASE(18, 128, 4, 32, 1, 3) W12_CASE(19, 128, 4, 32, 0, 4) W12_CASE(20,  64, 4, 32, 0, 6)
    W12_CASE(21, 128, 4, 16, 1, 4) W12_CASE(22, 128, 8, 32, 0, 3) W12_CASE(23, 256, 8, 16, 1, 2)
    W12_CASE(24, 256, 4, 16, 1, 2) W12_CASE(25,  64, 4, 16, 1, 6)
    default: TORCH_CHECK(false, "w12a16 hmma: нет воплощения ", geom);
  }
#undef W12_CASE
#undef W12_DIAG
#undef W12_ONE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


// ------------------------------------------------------------------ ПЛОТНЫЙ fp16-GEMM (197)
// Заменяет torch.matmul в пути w12 при больших M. Таблица воплощений -- для свипа: выбор
// геометрии делается ЗАМЕРОМ, а не вкусом (см. research/19: параметрические оси надо
// проходить, а не угадывать).
struct GemmGeom { int BM, BN, BK, WM, WN; };
// ПОТОЛОК РАЗДЕЛЯЕМОЙ -- 48 КиБ НА БЛОК (статический лимит sm_70; 96 КиБ требуют opt-in
// через cudaFuncSetAttribute). Считается как 2*(BM*(BK+4) + BK*(BN+16))*2 байт; первый заход
// таблицы этого не учёл, и ptxas отказал на четырёх воплощениях -- поэтому размер посчитан
// здесь явно, рядом со строками.
static constexpr GemmGeom kGemmGeoms[] = {
    {128, 128, 32, 2, 4},   // 0 -- база: 8 варпов, варп 64x32;      36 КиБ
    {128,  64, 32, 2, 2},   // 1 -- 4 варпа, варп 64x32;             28 КиБ
    { 64, 128, 32, 2, 4},   // 2 -- узкая по M, варп 32x32;          27 КиБ
    {128, 128, 32, 4, 4},   // 3 -- 16 варпов, варп 32x32;           36 КиБ
    {256,  64, 32, 4, 2},   // 4 -- вес читается вдвое реже;         46 КиБ
    {192, 128, 32, 4, 4},   // 5 -- варп 48x32;                      46 КиБ
    { 64,  64, 32, 2, 2},   // 6 -- мелкая, варп 32x32;              19 КиБ
    {128, 128, 16, 2, 4},   // 7 -- мельче по K (короче барьерный шаг); 20 КиБ
    // ВТОРОЙ ЗАХОД -- ВАРП 64x64. Фальсификатор снятием фазы: тензор 12 %, подача 26 %,
    // остальное -- РАСКЛАДКА В РАЗДЕЛЯЕМУЮ. На шаг k варп читает MB+NB фрагментов и делает
    // MB*NB умножений, то есть (MB+NB)/(MB*NB) чтений на HMMA: при 64x32 это 0.75, при
    // 64x64 -- 0.50. Вдвое меньше обращений на ту же тензорную работу.
    {128, 128, 32, 2, 2},   // 8 -- 4 варпа, варп 64x64, acc 128;      36 КиБ
    { 64, 128, 32, 1, 2},   // 9 -- 2 варпа, варп 64x64;               27 КиБ
    {256, 128, 16, 4, 2},   // 10 -- 8 варпов, варп 64x64;             29 КиБ
    {192, 128, 16, 3, 2},   // 11 -- 6 варпов, варп 64x64;             24 КиБ
    // ---- ТРЕТИЙ ЗАХОД: ПРИНУЖДЁННАЯ ЗАНЯТОСТЬ ------------------------------------------
    // Разложение по фазам (20.08) не нашло доминирующей дыры: подача 24 %, барьеры 12 %,
    // умножение 13 %, сумма ~49 % -- фазы ПЕРЕКРЫВАЮТСЯ. Снятие чтения разделяемой дало
    // МИНУС 6.5 % (тело замедлилось), то есть MIO здесь НЕ связывает. Остаётся дефицит
    // теней: при REG=117 и 256 нитях на SM влезает 2 блока = 16 варпов из 64 (25 %).
    // MINB принуждает ptxas уложиться в 65536/(MINB*нити) регистров -- ось, а не решение.
    {128, 128, 32, 2, 4},   // 12 -- то же, что 0, но MINB=2 (бюджет 128 рег)
    {128, 128, 32, 2, 4},   // 13 -- то же, но MINB=3 (бюджет 85 рег)
    { 64, 128, 32, 2, 4},   // 14 -- узкая по M, MINB=3
    {128,  64, 32, 2, 2},   // 15 -- 4 варпа, MINB=4
    // ---- ЧЕТВЁРТЫЙ ЗАХОД: СНИЗИТЬ ТРАФИК, А НЕ ОБРАЩЕНИЯ -------------------------------
    // Арифметика полосы (20.08): плитка 128x128 на qkv M=8192 читает (M/BM)*(N/BN) блоков по
    // (BM+BN)*K*2 байт = 9.4 ГБ; при 878 ГБ/с это 10.7 мс, а замер даёт 9.8 -- то есть тело
    // УЖЕ идёт по полосе HBM. cuBLAS за 6.7 мс укладывается в ~5.9 ГБ, значит он просто
    // ЧИТАЕТ МЕНЬШЕ. Трафик = K*2*(M/BM + N/BN)*BM*BN/(BM*BN) -> растёт при вытянутой
    // плитке и падает при крупной. 256x128 даёт 7.0 ГБ вместо 9.4 (-26 %).
    {256, 128, 16, 2, 4},   // 16 -- варп 128x32, acc 128; 29 КиБ; трафик 7.0 ГБ
    {256, 128, 16, 2, 4},   // 17 -- то же, MINB=2
    {256, 256, 16, 4, 4},   // 18 -- варп 64x64, трафик 4.7 ГБ; 37 КиБ
    {256, 256, 16, 4, 4},   // 19 -- то же, MINB=2
    // ---- ПЯТЫЙ ЗАХОД: ДВА БЛОКА НА SM -------------------------------------------------
    // Комбинатор: горячий цикл 348 команд, из них 256 HMMA, T1=474 при тензорном поле 512 --
    // цикл УЖЕ на полу, резать в нём нечего. Значит потери снаружи, и главная — занятость:
    // лучшее воплощение берёт 29 КиБ разделяемой при лимите 48 КиБ на SM, то есть блок на SM
    // ОДИН, и на каждом барьере простаивает весь SM. Ниже — та же крупная плитка, но
    // ужатая по разделяемой так, чтобы влезали ДВА блока (расход = 2*(BM*(BK+4)+BK*(BN+16))*2).
    {256, 128,  8, 4, 2},   // 20 -- 16.5 КиБ -> 2 блока; барьеров вдвое больше
    {192, 128, 16, 3, 2},   // 21 -- 24.0 КиБ -> 2 блока впритык
    {128, 128, 16, 2, 4},   // 22 -- 19.0 КиБ -> 2 блока, MINB=2
    {256, 128,  8, 4, 2},   // 23 -- то же, что 20, но MINB=2
    // ---- СЛИЯНИЕ БЛОКОВ: персистентный обход задач --------------------------------------
    {256, 128, 16, 4, 2},   // 24 -- geom 10, но PERSIST
    {128, 128, 32, 2, 4},   // 25 -- geom 0, но PERSIST
    {256, 128, 16, 4, 2},   // 26 -- PERSIST + MINB=2
    // ---- ШЕСТОЙ ЗАХОД: ФОРМА ПЛИТКИ ПО ЦЕЛЕВОЙ ФУНКЦИИ (консультация Fable, 20.08) ------
    // Доказано: класс здесь C-РЕЗИДЕНТНЫЙ (частичные суммы нельзя выгружать в 11-битный
    // ярус -- одна выгрузка вносит 2^-11, больше всего бюджета округления), поэтому граница
    // обмена идёт НЕ по ёмкости быстрого яруса, а по числу слотов накопителя R:
    //     Q >= 2*MNK/sqrt(R) + MN,  достигается КВАДРАТНОЙ плиткой m = n = sqrt(R).
    // Целевая функция трафика -- (1/m + 1/n). Наша 256x128 проигрывает не размером, а
    // НЕСБАЛАНСИРОВАННОСТЬЮ: q = 0.011719 против 0.009114 у 256x192 (-22 %).
    // Ключ, который был упущен: такие плитки влезают в разделяемую только при МЕЛКОМ шаге
    // по K (BK=8), а я перебирал BK=16/32, где они не помещались.
    {256, 192,  8, 2, 4},   // 27 -- q 0.778x, acc 192/дорожка, 18.5 КиБ
    {224, 192,  8, 2, 4},   // 28 -- q 0.825x, acc 168
    {256, 160,  8, 4, 2},   // 29 -- q 0.867x, acc 160
    {256, 192, 16, 2, 4},   // 30 -- та же форма, но вдвое реже барьеры (33.0 КиБ)
};
static constexpr int kGemmGeomCount = (int)(sizeof(kGemmGeoms) / sizeof(kGemmGeoms[0]));

int64_t gemm_f16_geom_count() { return kGemmGeomCount; }

std::vector<int64_t> gemm_f16_geom(int64_t g) {
  TORCH_CHECK(g >= 0 && g < kGemmGeomCount, "gemm_f16: нет воплощения ", g);
  const auto& G = kGemmGeoms[g];
  return {G.BM, G.BN, G.BK, G.WM, G.WN};
}

torch::Tensor gemm_f16(torch::Tensor x, torch::Tensor w, int64_t geom, int64_t diag) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "gemm_f16: нужны тензоры CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kHalf && w.scalar_type() == torch::kHalf,
              "gemm_f16: оба операнда fp16");
  TORCH_CHECK(w.dim() == 2, "gemm_f16: вес [K, N]");
  auto xc = x.contiguous(), wc = w.contiguous();
  const int K = (int)wc.size(0), N = (int)wc.size(1);
  const int M = (int)(xc.numel() / xc.size(-1));
  TORCH_CHECK((int)xc.size(-1) == K, "gemm_f16: K не сходится");
  TORCH_CHECK(geom >= 0 && geom < kGemmGeomCount, "gemm_f16: нет воплощения ", geom);
  auto y = torch::empty({M, N}, torch::dtype(torch::kHalf).device(xc.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  const auto* wp = reinterpret_cast<const __half*>(wc.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

#define GEMM_ONE(BM, BN, BK, WM, WN)                                                          \
  switch (diag) {                                                                             \
    case 1: GEMM_D(BM, BN, BK, WM, WN, 1); break;                                             \
    case 2: GEMM_D(BM, BN, BK, WM, WN, 2); break;                                             \
    case 3: GEMM_D(BM, BN, BK, WM, WN, 3); break;                                             \
    case 4: GEMM_D(BM, BN, BK, WM, WN, 4); break;                                             \
    case 5: GEMM_D(BM, BN, BK, WM, WN, 5); break;                                             \
    default: GEMM_D(BM, BN, BK, WM, WN, 0); break;                                            \
  }
#define GEMM_P(BM, BN, BK, WM, WN, MB)                                                        \
  fa2_sm70::gemm_f16::volta_gemm_f16<BM, BN, BK, WM, WN, MB, 0, 1>                            \
      <<<dim3(80 * (MB), 1), (WM) * (WN) * 32, 0, stream>>>(xp, wp, yp, M, N, K)
#define GEMM_MB(BM, BN, BK, WM, WN, MB)                                                       \
  switch (diag) {                                                                             \
    case 1: GEMM_DM(BM, BN, BK, WM, WN, MB, 1); break;                                        \
    case 2: GEMM_DM(BM, BN, BK, WM, WN, MB, 2); break;                                        \
    case 3: GEMM_DM(BM, BN, BK, WM, WN, MB, 3); break;                                        \
    case 4: GEMM_DM(BM, BN, BK, WM, WN, MB, 4); break;                                        \
    case 5: GEMM_DM(BM, BN, BK, WM, WN, MB, 5); break;                                        \
    default: GEMM_DM(BM, BN, BK, WM, WN, MB, 0); break;                                       \
  }
#define GEMM_DM(BM, BN, BK, WM, WN, MB, D)                                                    \
  fa2_sm70::gemm_f16::volta_gemm_f16<BM, BN, BK, WM, WN, MB, D>                               \
      <<<dim3((N + (BN) - 1) / (BN), (M + (BM) - 1) / (BM)), (WM) * (WN) * 32, 0, stream>>>(  \
          xp, wp, yp, M, N, K)
#define GEMM_D(BM, BN, BK, WM, WN, D)                                                         \
  fa2_sm70::gemm_f16::volta_gemm_f16<BM, BN, BK, WM, WN, 1, D>                                \
      <<<dim3((N + (BN) - 1) / (BN), (M + (BM) - 1) / (BM)), (WM) * (WN) * 32, 0, stream>>>(  \
          xp, wp, yp, M, N, K)

  switch (geom) {
    case 0: GEMM_ONE(128, 128, 32, 2, 4); break;
    case 1: GEMM_ONE(128,  64, 32, 2, 2); break;
    case 2: GEMM_ONE( 64, 128, 32, 2, 4); break;
    case 3: GEMM_ONE(128, 128, 32, 4, 4); break;
    case 4: GEMM_ONE(256,  64, 32, 4, 2); break;
    case 5: GEMM_ONE(192, 128, 32, 4, 4); break;
    case 6: GEMM_ONE( 64,  64, 32, 2, 2); break;
    case 7: GEMM_ONE(128, 128, 16, 2, 4); break;
    case 8: GEMM_ONE(128, 128, 32, 2, 2); break;
    case 9: GEMM_ONE( 64, 128, 32, 1, 2); break;
    case 10: GEMM_ONE(256, 128, 16, 4, 2); break;
    case 11: GEMM_ONE(192, 128, 16, 3, 2); break;
    case 12: GEMM_MB(128, 128, 32, 2, 4, 2); break;
    case 13: GEMM_MB(128, 128, 32, 2, 4, 3); break;
    case 14: GEMM_MB( 64, 128, 32, 2, 4, 3); break;
    case 15: GEMM_MB(128,  64, 32, 2, 2, 4); break;
    case 16: GEMM_ONE(256, 128, 16, 2, 4); break;
    case 17: GEMM_MB(256, 128, 16, 2, 4, 2); break;
    case 18: GEMM_ONE(256, 256, 16, 4, 4); break;
    case 19: GEMM_MB(256, 256, 16, 4, 4, 2); break;
    case 20: GEMM_ONE(256, 128,  8, 4, 2); break;
    case 21: GEMM_ONE(192, 128, 16, 3, 2); break;
    case 22: GEMM_MB(128, 128, 16, 2, 4, 2); break;
    case 23: GEMM_MB(256, 128,  8, 4, 2, 2); break;
    case 24: GEMM_P(256, 128, 16, 4, 2, 1); break;
    case 25: GEMM_P(128, 128, 32, 2, 4, 1); break;
    case 26: GEMM_P(256, 128, 16, 4, 2, 2); break;
    case 27: GEMM_ONE(256, 192,  8, 2, 4); break;
    case 28: GEMM_ONE(224, 192,  8, 2, 4); break;
    case 29: GEMM_ONE(256, 160,  8, 4, 2); break;
    case 30: GEMM_ONE(256, 192, 16, 2, 4); break;
    default: TORCH_CHECK(false, "gemm_f16: воплощение вне таблицы");
  }
#undef GEMM_ONE
#undef GEMM_D
#undef GEMM_MB
#undef GEMM_DM
#undef GEMM_P
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w12a16_gemv", &w12a16_gemv, "W12A16 GEMV: вес потребляется упакованным (sm_70)");
  m.def("w12_dequant_into", &w12_dequant_into, "распаковка в ОБЩИЙ буфер для большого M");
  m.def("w12a16_hmma_gemm", &w12a16_hmma_gemm,
        "W12A16 на тензорных ядрах: две шестиразрядные плоскости + удвоенная ось свёртки",
        pybind11::arg("x"), pybind11::arg("qw_t"), pybind11::arg("scales"),
        pybind11::arg("group_size"), pybind11::arg("geom") = 0, pybind11::arg("diag") = 0);
  m.def("w12a16_hmma_geom_count", &w12a16_hmma_geom_count, "сколько воплощений в таблице");
  m.def("gemm_f16", &gemm_f16, "плотный fp16-GEMM (diag: 0 полное, 1 без подачи, 2 без умножения)",
        py::arg("x"), py::arg("w"), py::arg("geom"), py::arg("diag") = 0);
  m.def("gemm_f16_geom_count", &gemm_f16_geom_count, "сколько воплощений плотного GEMM");
  m.def("gemm_f16_geom", &gemm_f16_geom, "параметры воплощения: [BM, BN, BK, WM, WN]");
  m.def("w12a16_hmma_geom", &w12a16_hmma_geom, "параметры воплощения: [BN, WN, BK, конвейер, MINB]");
}
