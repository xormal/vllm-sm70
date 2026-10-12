// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// СЛИТОЕ ТЕЛО: RMSNorm + W8A16 GEMV в ОДНОМ запуске. Скелет резидентной схемы (docs/DESIGN_RESIDENT_BLOCK.md, Ш1).
//
// ЗАЧЕМ ИМЕННО ТАК, А НЕ «ускорить норму».
// Разбор шага декода дал разложение превышения над периодом (13.13 мс из 31.22): самая крупная
// статья -- 3.943 мс = 12.6 % шага в фазах ЧИСТОЙ ВЫДАЧИ, где 877 запусков двигают 0.1 % байт.
// Это не медленные ядра, это ГРАНИЦЫ между ними: зависимость между значениями в 10-35 КБ платится
// границей ЗАПУСКА. Четыре независимых замера (глубина конвейера, события в прологе, геометрия
// блока, формат площадки) дали РОВНО НОЛЬ каждый -- все они меняли устройство ВНУТРИ тел, не трогая
// границ. Пока границы стоят, улучшение внутри тела упирается в них.
//
// РАЗМЕН, НА КОТОРОМ ЭТО ДЕРЖИТСЯ: пересчитать дешевле, чем привезти.
// Счётный канал шага занят на 3.1 % (2 Флоп/байт против гребня 64.4), запас x32. Поэтому норму
// здесь НЕ читают готовой, а КАЖДЫЙ БЛОК СЧИТАЕТ ЕЁ САМ. Цена: лишнее чтение x (K*2 Б) на блок.
// При K=N=5120 и 256 столбцах на блок это 20 блоков x 10 КБ = 200 КБ против 26 МБ веса = 0.8 %.
// Взамен исчезает целое ядро вместе с записью нормализованного вектора в HBM и его чтением обратно.
//
// ЧТО ЗДЕСЬ НЕ ДЕЛАЕТСЯ И ПОЧЕМУ. Арифметика умножения скопирована из w8a16_gemv.cu БЕЗ ИЗМЕНЕНИЙ
// (та же группировка по G, тот же порядок накопления, та же свёртка (q-z)*s). Это условие гейта:
// на скелете проверяется СХЕМА, а не формат и не точность. Единственная разница с эталоном --
// x приходит сырым и нормируется на месте.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdlib>

namespace {

constexpr int MAXG_ = 64;

// Плитка активации в разделяемой: держим её пошире, чтобы внутренний виток читал float4.
constexpr int XT_ = 512;

// ------------------------------------------------------------------ проход 1: норма, своими силами
// Блок читает ВЕСЬ x и сводит сумму квадратов. Читается только x (K значений half), не вес.
// [ВХОД МОЖЕТ БЫТЬ fp32, И ЭТО НЕ ПРИДИРКА]
// Норма здесь -- `GemmaRMSNorm`, и у неё при fp16 НЕВЯЗКА ДЕРЖИТСЯ В fp32:
// `x = x.float() + residual.float(); residual = x`. Снятый машинный код Triton это подтверждает --
// у тела три входа, два fp16 и один fp32, и сумма идёт в fp32. Принять такой вход как __half
// значило бы округлить состояние токена до привязки и внести МОЛЧАЛИВУЮ потерю точности, которую
// потом искали бы в другом месте.
template <typename XT> __device__ __forceinline__ float _kf(XT v);
template <> __device__ __forceinline__ float _kf<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float _kf<float>(float v) { return v; }

template <int THREADS, typename XT = __half>
__device__ __forceinline__ float rms_scale_of(const XT* __restrict__ x, int K, float eps) {
    __shared__ float red[THREADS / 32];
    float s = 0.f;
    for (int k = threadIdx.x; k < K; k += THREADS) {
        const float v = _kf<XT>(x[k]);
        s = fmaf(v, v, s);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; ++w) t += red[w];
        red[0] = rsqrtf(t / (float)K + eps);       // масштаб RMS -- один на весь блок
    }
    __syncthreads();
    return red[0];
}

// ------------------------------------------------------------------ слитое тело
template <int THREADS, bool SPLIT, typename XT = __half>
__global__ void rmsnorm_w8a16_gemv_kernel(const XT* __restrict__ x,
                                          const __half* __restrict__ rw,   // веса нормы, K штук
                                          const int* __restrict__ qweight,
                                          const int* __restrict__ qzeros,
                                          const __half* __restrict__ scales,
                                          __half* __restrict__ y, float* __restrict__ ypart,
                                          int N, int K, int G, float eps, int groups_per_split, float zbias) {
    // [Ш1] Состояние токена ЖИВЁТ в блоке: нормированная плитка x кладётся в разделяемую и
    // потребляется всеми столбцами этого блока, ни разу не уходя в HBM.
    // ВЫРАВНИВАНИЕ ДАЁТСЯ ТИПУ ОБЪЯВЛЕНИЯ, а не тому, во что кастуют: sx читается float4
    // (16 Б), а `float[]` гарантирует только 4. При 256 потоках red[8]=32 Б ставила sx на
    // границу 16 СЛУЧАЙНО; при 64 потоках red[2]=8 Б -- и тот же код едет по невыровненному.
    __shared__ __align__(16) float sx[XT_];

    const float rs = rms_scale_of<THREADS, XT>(x, K, eps);   // проход 1 -- пересчёт вместо привоза

    const int n = blockIdx.x * THREADS + threadIdx.x;
    const bool live = (n < N);
    const int nw = n >> 2, nsh = (n & 3) * 8;
    float acc = 0.f;

    // [ДЕФЕКТ, НАЙДЕННЫЙ ПЕРВЫМ ЖЕ ЗАПУСКОМ 16.08.2026 -- ТЕЛО БЫЛО НАПИСАНО И НИ РАЗУ НЕ ИСПОЛНЕНО]
    // Пускач заводил сетку (nblk, ksplit) и передавал `groups_per_split`, а ядро НЕ ЧИТАЛО ни
    // `blockIdx.y`, ни этот параметр: каждый из ksplit блоков считал свёртку по ВСЕМУ K, и все
    // складывались атомарно. Ответ умножался на число расщеплений -- ровно поэтому relL2 рос с K
    // (3.0 при K=512, 39 при K=5120: это и есть ksplit 4 и 40).
    // ДИАГНОЗ ДАЛА НЕ ЛОГИКА, А ФОРМА ОШИБКИ: она не «шумела», а была ЦЕЛЫМ МНОЖИТЕЛЕМ.
    // Норма (`rs`) при этом обязана считаться по ВСЕМУ K и потому остаётся вне разреза.
    const int kbeg = SPLIT ? blockIdx.y * groups_per_split * G : 0;
    const int kend = SPLIT ? min(K, kbeg + groups_per_split * G) : K;
    if (kbeg >= kend) return;                 // ветвь ОДНОРОДНА по блоку: барьеров впереди нет

    for (int k0 = kbeg; k0 < kend; k0 += XT_) {
        const int len = min(XT_, kend - k0);
        __syncthreads();
        // УКЛАДКА С НОРМОЙ НА МЕСТЕ: элемент и так проходит через регистр, множитель бесплатен.
        for (int t = threadIdx.x; t < len; t += THREADS)
            sx[t] = _kf<XT>(x[k0 + t]) * rs * __half2float(rw[k0 + t]);
        __syncthreads();
        if (!live) continue;

        for (int gk = 0; gk < len; gk += G) {
            const int g = (k0 + gk) / G;
            const float s = __half2float(scales[(long)g * N + n]);
      // [УСЛОВНОСТЬ НУЛЯ GPTQ] v1 (`checkpoint_format: gptq`) хранит (z-1), v2 -- z. Читать СЫРОЕ
      // значение значит жить по v2 и промахиваться на ОДИН УРОВЕНЬ КВАНТА на каждом весе: ~1.4 %
      // систематически, без падения и без внешнего признака. ЗАМЕРЕНО 16.08 на w8a16 против чужого
      // тела: 5.55e-3 при неверной условности против 1.20e-3 при верной. Приходит СНАРУЖИ.
            const float z = (float)((qzeros[(long)g * (N >> 2) + nw] >> nsh) & 0xFF) + zbias;
            const int glen = min(G, len - gk);
            float qa = 0.f, xa = 0.f;
            #pragma unroll 4
            for (int k = 0; k < glen; k += 4) {
                const unsigned w = (unsigned)qweight[(long)((k0 + gk + k) >> 2) * N + n];
                const float q0 = (float)( w        & 0xFF), q1 = (float)((w >>  8) & 0xFF);
                const float q2 = (float)((w >> 16) & 0xFF), q3 = (float)((w >> 24) & 0xFF);
                const float4 a = *reinterpret_cast<const float4*>(&sx[gk + k]);
                qa = fmaf(a.x, q0, qa); qa = fmaf(a.y, q1, qa);
                qa = fmaf(a.z, q2, qa); qa = fmaf(a.w, q3, qa);
                xa += a.x + a.y + a.z + a.w;
            }
            acc = fmaf(s, qa - z * xa, acc);      // та же свёртка (q-z)*s, что и в эталоне
        }
    }
    if (!live) return;
    if (SPLIT) atomicAdd(ypart + n, acc);
    else       y[n] = __float2half(acc);
}


// ------------------------------------------------------------------ ЦЕПЬ MLP: норма + gate_up + silu
// СВЁРТКА «взять - обработать - положить - взять - обработать - положить» в «взять - обработать -
// обработать - положить». Сейчас на слой идёт: норма (чтение x, запись x'), gate_up (чтение x',
// чтение веса, запись u размером 2*I), silu*mul (чтение 2*I, запись I). Здесь всё это ОДНО тело:
// x читается ОДИН раз, u не рождается в HBM вовсе, наружу выходит только I значений.
//
// ПОЧЕМУ БАРЬЕР ПО СЕТКЕ НЕ НУЖЕН. silu(u1)*u2 поэлементна по столбцу промежуточной оси, а обе
// половины столбца считает ОДИН блок: он ведёт два накопителя по двум срезам веса. Редукции между
// блоками нет, значит граница снимается слиянием двух тел, без кооперативного запуска.
// (Свернуть сюда же down нельзя: ему нужен ВЕСЬ промежуточный вектор, это редукция по блокам.)
// РАСЩЕПЛЕНИЕ ПО K -- не оптимизация, а весь смысл (так же, как в отгруженном w8a16_gemv).
// Столбцов ровно I, поток на столбец: меняя ЧИСЛО ПОТОКОВ В БЛОКЕ, я переупаковывал одни и те же
// I потоков и параллелизма не добавлял (лестница 0.42->3.40 волны дала 2.4 %). Байт в полёте растёт
// только когда k-срезы идут РАЗНЫМИ блоками -- это и есть дефицит Литтла из DESIGN_RESIDENT_BLOCK §4.
// ШИРИНА ОБРАЩЕНИЯ (Ш3): CPT столбцов на поток. При CPT=1 варп берёт 32 слова = 128 Б за шаг;
// при CPT=4 -- 512 Б одной LDG.128 на поток. Байт в полёте растёт вчетверо на ту же команду, а
// число независимых потоков в DRAM падает вчетверо. Побочно: xa (сумма активаций группы) одна на
// все столбцы, и при CPT=4 она считается ОДИН раз вместо четырёх; нули четырёх соседних столбцов
// лежат в ОДНОМ слове qzeros, то есть четыре распаковки нуля превращаются в одну загрузку.
template <int THREADS, int CPT, bool SPLIT, typename XT = __half>
__global__ void rmsnorm_gateup_silu_kernel(const XT* __restrict__ x,
                                           const __half* __restrict__ rw,
                                           const int* __restrict__ qw_g, const int* __restrict__ qz_g,
                                           const __half* __restrict__ sc_g,
                                           const int* __restrict__ qw_u, const int* __restrict__ qz_u,
                                           const __half* __restrict__ sc_u,
                                           __half* __restrict__ y, float* __restrict__ ypart,
                                           int I, int K, int G, float eps, int groups_per_split,
                                           float rs_pre, float zbias, float wbias) {
    __shared__ __align__(16) float sx[XT_];
    const float rs = (rs_pre >= 0.f) ? rs_pre : rms_scale_of<THREADS, XT>(x, K, eps);

    const int n0 = (blockIdx.x * THREADS + threadIdx.x) * CPT;    // первый столбец потока
    const bool live = (n0 + CPT <= I);
    float ag[CPT], au[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) { ag[c] = 0.f; au[c] = 0.f; }

    const int kbeg = SPLIT ? blockIdx.y * groups_per_split * G : 0;
    const int kend = SPLIT ? min(K, kbeg + groups_per_split * G) : K;
    if (kbeg >= kend) return;

    for (int k0 = kbeg; k0 < kend; k0 += XT_) {
        const int len = min(XT_, kend - k0);
        __syncthreads();
        for (int t = threadIdx.x; t < len; t += THREADS)
            // [(1+w) У GemmaRMSNorm] Множитель нормы здесь `1 + weight`, а вес
            // инициализирован НУЛЯМИ. Прибавка сделана ЗДЕСЬ, а не в питоне: снаружи
            // она потребовала бы либо лишнего тела на слой, либо мутации модуля под
            // трассировкой -- и то и другое уже роняло подъём. Цена ровно ноль.
            sx[t] = _kf<XT>(x[k0 + t]) * rs * (wbias + __half2float(rw[k0 + t]));
        __syncthreads();
        if (!live) continue;
        for (int gk = 0; gk < len; gk += G) {
            const int g = (k0 + gk) / G;
            float sg[CPT], su[CPT], zg[CPT], zu[CPT];
            const unsigned wzg = (unsigned)qz_g[(long)g * (I >> 2) + (n0 >> 2)];
            const unsigned wzu = (unsigned)qz_u[(long)g * (I >> 2) + (n0 >> 2)];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                sg[c] = __half2float(sc_g[(long)g * I + n0 + c]);
                su[c] = __half2float(sc_u[(long)g * I + n0 + c]);
                // при CPT=4 нули четырёх столбцов -- байты ОДНОГО слова; иначе слово своё у каждого
                const unsigned bg = (CPT == 4) ? wzg : (unsigned)qz_g[(long)g * (I >> 2) + ((n0 + c) >> 2)];
                const unsigned bu = (CPT == 4) ? wzu : (unsigned)qz_u[(long)g * (I >> 2) + ((n0 + c) >> 2)];
                const int sh = ((n0 + c) & 3) * 8;
      // [УСЛОВНОСТЬ НУЛЯ GPTQ] v1 (`checkpoint_format: gptq`) хранит (z-1), v2 -- z. Читать СЫРОЕ
      // значение значит жить по v2 и промахиваться на ОДИН УРОВЕНЬ КВАНТА на каждом весе: ~1.4 %
      // систематически, без падения и без внешнего признака. ЗАМЕРЕНО 16.08 на w8a16 против чужого
      // тела: 5.55e-3 при неверной условности против 1.20e-3 при верной. Приходит СНАРУЖИ.
                zg[c] = (float)((bg >> sh) & 0xFF) + zbias;
                zu[c] = (float)((bu >> sh) & 0xFF) + zbias;
            }
            const int glen = min(G, len - gk);
            float qg[CPT], qu[CPT], xa = 0.f;
            #pragma unroll
            for (int c = 0; c < CPT; ++c) { qg[c] = 0.f; qu[c] = 0.f; }
            #pragma unroll 4
            for (int k = 0; k < glen; k += 4) {
                const long wi = (long)((k0 + gk + k) >> 2) * I + n0;
                const float4 a = *reinterpret_cast<const float4*>(&sx[gk + k]);
                xa += a.x + a.y + a.z + a.w;            // одна на все столбцы -- считаем один раз
                #pragma unroll
                for (int c = 0; c < CPT; ++c) {
                    const unsigned wg = (unsigned)qw_g[wi + c], wu = (unsigned)qw_u[wi + c];
                    qg[c] = fmaf(a.x, (float)( wg        & 0xFF), qg[c]);
                    qg[c] = fmaf(a.y, (float)((wg >>  8) & 0xFF), qg[c]);
                    qg[c] = fmaf(a.z, (float)((wg >> 16) & 0xFF), qg[c]);
                    qg[c] = fmaf(a.w, (float)((wg >> 24) & 0xFF), qg[c]);
                    qu[c] = fmaf(a.x, (float)( wu        & 0xFF), qu[c]);
                    qu[c] = fmaf(a.y, (float)((wu >>  8) & 0xFF), qu[c]);
                    qu[c] = fmaf(a.z, (float)((wu >> 16) & 0xFF), qu[c]);
                    qu[c] = fmaf(a.w, (float)((wu >> 24) & 0xFF), qu[c]);
                }
            }
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                ag[c] = fmaf(sg[c], qg[c] - zg[c] * xa, ag[c]);
                au[c] = fmaf(su[c], qu[c] - zu[c] * xa, au[c]);
            }
        }
    }
    if (!live) return;
    #pragma unroll
    for (int c = 0; c < CPT; ++c) {
        if (SPLIT) { atomicAdd(ypart + n0 + c, ag[c]); atomicAdd(ypart + I + n0 + c, au[c]); }
        else y[n0 + c] = __float2half(ag[c] / (1.f + __expf(-ag[c])) * au[c]);
    }
}

__global__ void part_cast_kernel(const float* __restrict__ part, __half* __restrict__ y, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) y[n] = __float2half(part[n]);
}

__global__ void chain_epilogue_kernel(const float* __restrict__ part, __half* __restrict__ y, int I) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= I) return;
    const float g = part[n], u = part[I + n];
    y[n] = __float2half(g / (1.f + __expf(-g)) * u);
}

}  // namespace

// x: [1, K] half (СЫРОЙ, ненормированный)   rw: [K] half   qweight: [K/4, N] int32
// qzeros: [K/G, N/4] int32                  scales: [K/G, N] half
torch::Tensor rmsnorm_w8a16_gemv(torch::Tensor x, torch::Tensor rms_weight, double eps,
                                 torch::Tensor qweight, torch::Tensor qzeros,
                                 torch::Tensor scales, int64_t group_size, bool v2_zero) {
  // Условность нуля GPTQ приходит СНАРУЖИ и умолчания НЕ ИМЕЕТ: v1 (`checkpoint_format: gptq`)
  // хранит (z-1), v2 -- z. Молчаливое умолчание здесь и было дефектом: тело жило по v2, боевой
  // слепок -- v1, промах на один уровень кванта на КАЖДОМ весе (~1.4 %, без падения).
  const float zbias = v2_zero ? 0.0f : 1.0f;

    TORCH_CHECK(x.is_cuda() && qweight.is_cuda(), "fused: нужны тензоры на карте");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16,
                "fused gemv: вход fp16 (fp32 пока только у слитой цепи MLP)");
    TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2, "fused: qweight [K/4,N], scales [K/G,N]");
    auto xc = x.contiguous(), rwc = rms_weight.contiguous();
    auto qw = qweight.contiguous(), qz = qzeros.contiguous(), sc = scales.contiguous();
    const int N = (int)qw.size(1), K = (int)qw.size(0) * 4;
    const int G = (int)group_size;
    TORCH_CHECK(xc.numel() == K, "fused: скелет собран под M=1 (декодный токен)");
    TORCH_CHECK(rwc.numel() == K, "fused: веса нормы должны быть длины K");
    TORCH_CHECK((N & 3) == 0, "fused: N кратно 4 (qzeros пакует 4 столбца в слово)");
    TORCH_CHECK(G <= MAXG_ * 4, "fused: слишком большая группа");
    TORCH_CHECK(XT_ % G == 0, "fused: плитка обязана делиться на группу без остатка");

    auto y = torch::empty({1, N}, torch::dtype(torch::kFloat16).device(xc.device()));
    constexpr int THREADS = 256;
    const int nblk = (N + THREADS - 1) / THREADS;
    const int ngroups = (K + G - 1) / G;
    int ksplit = max(1, min(ngroups, (80 * 8 + nblk - 1) / nblk));   // 8 блоков на SM, как в отгруженном
    const int gps = (ngroups + ksplit - 1) / ksplit;
    ksplit = (ngroups + gps - 1) / gps;
    torch::Tensor part; float* pp = nullptr;
    if (ksplit > 1) { part = torch::zeros({N}, torch::dtype(torch::kFloat32).device(xc.device()));
                      pp = part.data_ptr<float>(); }
    const at::cuda::CUDAGuard guard(xc.device());
    auto st = at::cuda::getCurrentCUDAStream();
    const dim3 grid = (ksplit > 1) ? dim3(nblk, ksplit) : dim3(nblk);
    if (ksplit > 1)
      rmsnorm_w8a16_gemv_kernel<THREADS, true><<<grid, THREADS, 0, st>>>(
        reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(rwc.data_ptr<at::Half>()),
        qw.data_ptr<int>(), qz.data_ptr<int>(),
        reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
        nullptr, pp, N, K, G, (float)eps, gps, zbias);
    else
      rmsnorm_w8a16_gemv_kernel<THREADS, false><<<grid, THREADS, 0, st>>>(
        reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(rwc.data_ptr<at::Half>()),
        qw.data_ptr<int>(), qz.data_ptr<int>(),
        reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()), nullptr, N, K, G, (float)eps, 0, zbias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (ksplit > 1) {
        part_cast_kernel<<<(N + 255) / 256, 256, 0, st>>>(
            pp, reinterpret_cast<__half*>(y.data_ptr<at::Half>()), N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}



// Цепь MLP: x сырой -> норма -> gate_up (два среза веса) -> silu*mul -> наружу ТОЛЬКО I значений.
torch::Tensor rmsnorm_gateup_silu(torch::Tensor x, torch::Tensor rms_weight, double eps,
                                  bool w_plus1,
                                  torch::Tensor qw_g, torch::Tensor qz_g, torch::Tensor sc_g,
                                  torch::Tensor qw_u, torch::Tensor qz_u, torch::Tensor sc_u,
                                  int64_t group_size, bool v2_zero) {
  // Условность нуля GPTQ приходит СНАРУЖИ и умолчания НЕ ИМЕЕТ: v1 (`checkpoint_format: gptq`)
  // хранит (z-1), v2 -- z. Молчаливое умолчание здесь и было дефектом: тело жило по v2, боевой
  // слепок -- v1, промах на один уровень кванта на КАЖДОМ весе (~1.4 %, без падения).
  const float zbias = v2_zero ? 0.0f : 1.0f;

    TORCH_CHECK(x.is_cuda(), "fused: нужны тензоры на карте");
    const bool xf32 = (x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(xf32 || x.scalar_type() == torch::kFloat16, "fused: вход fp16 либо fp32");
    auto xc = x.contiguous(), rwc = rms_weight.contiguous();
    auto Wg = qw_g.contiguous(), Zg = qz_g.contiguous(), Sg = sc_g.contiguous();
    auto Wu = qw_u.contiguous(), Zu = qz_u.contiguous(), Su = sc_u.contiguous();
    const int I = (int)Wg.size(1), K = (int)Wg.size(0) * 4, G = (int)group_size;
    TORCH_CHECK(Wu.size(1) == I && Wu.size(0) * 4 == K, "fused: срезы веса разной формы");
    TORCH_CHECK(xc.numel() == K, "fused: скелет собран под M=1");
    TORCH_CHECK((I & 3) == 0 && XT_ % G == 0, "fused: I кратно 4, плитка кратна группе");
    auto y = torch::empty({1, I}, torch::dtype(torch::kFloat16).device(xc.device()));
    const at::cuda::CUDAGuard guard(xc.device());
    // ЗАПОЛНЕНИЕ МАШИНЫ -- отдельная ось, и она здесь ТОЧНАЯ: столбцов фиксировано, значит меньше
    // потоков в блоке = больше блоков, а арифметика не меняется ни на бит и сведения между блоками
    // не возникает. При I=8704 и 256 потоках выходит 34 блока на 80 SM = 0.42 ВОЛНЫ, то есть
    // больше половины микроконтроллеров простаивает -- именно это, а не слияние, съело выигрыш цепи.
    const char* e = getenv("FA2SM70_FUSED_THREADS");
    const int T = e ? atoi(e) : 128;   // замерено: 128x2 = 82.4 % крыши против 72.6 % у 256x1
    const char* ec = getenv("FA2SM70_FUSED_CPT");
    const int CPT = ec ? atoi(ec) : 2;
    const char* en = getenv("FA2SM70_FUSED_NORS");
    const float rs_pre = (en && atoi(en)) ? 1.0f : -1.0f;    // фальсификатор: снять фазу нормы
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* px = xf32 ? nullptr
                            : reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
    const __half* pr = reinterpret_cast<const __half*>(rwc.data_ptr<at::Half>());
    const __half* pg = reinterpret_cast<const __half*>(Sg.data_ptr<at::Half>());
    const __half* pu = reinterpret_cast<const __half*>(Su.data_ptr<at::Half>());
    __half* py = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

    // цель -- 8 блоков на SM, как в отгруженном теле; ниже этого тело связано ЗАДЕРЖКОЙ, а не полосой
    const int nblk = (I + T * CPT - 1) / (T * CPT);
    const int ngroups = (K + G - 1) / G;
    const char* ek = getenv("FA2SM70_FUSED_KSPLIT");
    int ksplit = ek ? atoi(ek) : max(1, min(ngroups, (80 * 8 + nblk - 1) / nblk));
    ksplit = max(1, min(ngroups, ksplit));
    const int gps = (ngroups + ksplit - 1) / ksplit;
    ksplit = (ngroups + gps - 1) / gps;                 // отсечь пустые хвостовые срезы

    torch::Tensor part;
    float* pp = nullptr;
    if (ksplit > 1) {
        part = torch::zeros({2, I}, torch::dtype(torch::kFloat32).device(xc.device()));
        pp = part.data_ptr<float>();
    }
    const dim3 grid = (ksplit > 1) ? dim3(nblk, ksplit) : dim3(nblk);
// Диспетчер по ТИПУ ВХОДА: fp32-невязка Gemma идёт в ядро как есть, без округления до fp16.
#define CHAIN_ONE(TT, CC, SP, XTY, PX, PART)                                                    \
    rmsnorm_gateup_silu_kernel<TT, CC, SP, XTY><<<grid, TT, 0, st>>>(                           \
        PX, pr, Wg.data_ptr<int>(), Zg.data_ptr<int>(), pg,                                     \
        Wu.data_ptr<int>(), Zu.data_ptr<int>(), pu, py, PART, I, K, G, (float)eps,              \
        (SP ? gps : 0), rs_pre, zbias, w_plus1 ? 1.0f : 0.0f)
#define CHAIN_LAUNCH(TT, CC)                                                                    \
    do {                                                                                        \
      if (xf32) {                                                                               \
        const float* pxf = xc.data_ptr<float>();                                                \
        if (ksplit > 1) { CHAIN_ONE(TT, CC, true, float, pxf, pp); }                             \
        else            { CHAIN_ONE(TT, CC, false, float, pxf, nullptr); }                       \
      } else {                                                                                  \
        if (ksplit > 1) { CHAIN_ONE(TT, CC, true, __half, px, pp); }                             \
        else            { CHAIN_ONE(TT, CC, false, __half, px, nullptr); }                       \
      }                                                                                         \
    } while (0)
    if      (T == 128 && CPT == 4) { CHAIN_LAUNCH(128, 4); }
    else if (T == 128 && CPT == 2) { CHAIN_LAUNCH(128, 2); }
    else if (T == 128)             { CHAIN_LAUNCH(128, 1); }
    else if (CPT == 4)             { CHAIN_LAUNCH(256, 4); }
    else if (CPT == 2)             { CHAIN_LAUNCH(256, 2); }
    else                           { CHAIN_LAUNCH(256, 1); }
#undef CHAIN_LAUNCH
#undef CHAIN_ONE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (ksplit > 1) {
        chain_epilogue_kernel<<<(I + 255) / 256, 256, 0, st>>>(pp, py, I);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_w8a16_gemv", &rmsnorm_w8a16_gemv,
          "СЛИТОЕ: RMSNorm + W8A16 GEMV одним запуском, норма пересчитывается в блоке");
    m.def("rmsnorm_gateup_silu", &rmsnorm_gateup_silu,
          pybind11::arg("x"), pybind11::arg("rms_weight"), pybind11::arg("eps"),
          pybind11::arg("w_plus1"), pybind11::arg("qw_g"), pybind11::arg("qz_g"),
          pybind11::arg("sc_g"), pybind11::arg("qw_u"), pybind11::arg("qz_u"),
          pybind11::arg("sc_u"), pybind11::arg("group_size"), pybind11::arg("v2_zero"),
          "СЛИТОЕ: норма + gate_up + silu*mul, промежуточный u в HBM не выходит");
}
