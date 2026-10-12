// -------------------------------------------------------------------------------------------
// ГОЛОВА ЦЕПОЧКИ GDN: l2norm(q), l2norm(k) и локальный cumsum(g) ОДНИМ ПУСКОМ.
//
// ЗАЧЕМ. Замер `tools/gdn_vs_triton.py` на боевых формах (T=4, H=24, Hg=8, D=128) показал:
// наш GDN идёт 0.418 мс на слой против 0.916 у Triton (x2.19), но ПЯТЬ шагов конвейера из
// шести переписаны и дают x3.4-4.8, а шестой -- `l2norm+cumsum` -- не тронут вовсе и равен
// образцу (0.273 против 0.273). Он один занимает 65 % нашего GDN, то есть 13.1 мс на шаг
// декода из 58.8 -- 22 % шага (записка 25 §108).
//
// ПОЧЕМУ ОН ДОРОГ НЕ ПО РАБОТЕ. При T=4 работы здесь на единицы микросекунд: 2 x 8 x 4
// нормировок по 128 элементов и 24 x 4 префиксных сумм. 0.273 мс -- это НАКЛАДНЫЕ трёх
// отдельных пусков Triton (q, k, g) с их подготовкой, а не счёт. Отсюда и решение: не
// «ускорять формулу», а слить три пуска в один и убрать подготовку.
//
// РАСКЛАДКА. Блок = (батч, позиция t). Варп 0..Hg-1 нормирует q, следующие Hg варпов -- k,
// оставшиеся потоки считают cumsum по g для своих голов. Нормировка по D=128 -- ровно четыре
// элемента на линию варпа, редукция через __shfl_xor_sync без разделяемой памяти.
// -------------------------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

// Преобразования half<->float ТОЛЬКО явные: расширение собирается с
// -D__CUDA_NO_HALF_CONVERSIONS__, и неявные приведения там не компилируются.
template <int D>
__device__ __forceinline__ void norm_row(const __half* __restrict__ vh, __half* __restrict__ out,
                                         int lane, float eps) {
    constexpr int EPT = D / 32;                       // D=128 -> 4 элемента на линию
    float v[EPT], s = 0.f;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        v[e] = __half2float(vh[lane * EPT + e]);
        s += v[e] * v[e];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float m = rsqrtf(s + eps);
#pragma unroll
    for (int e = 0; e < EPT; ++e) out[lane * EPT + e] = __float2half(v[e] * m);
}

// q,k: [B, T, Hg, D] (нормируются по D); g: [B, T, H] (кумсумма по t внутри чанка).
template <int D>
__global__ void gdn_head_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                                __half* __restrict__ qo, __half* __restrict__ ko,
                                const float* __restrict__ g, float* __restrict__ go,
                                int Tlen, int Hg, int H, int chunk, float eps) {
    const int b = blockIdx.y, t = blockIdx.x;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    // Нормировки: варпы [0, Hg) -- q, [Hg, 2Hg) -- k.
    if (w < Hg) {
        const long off = ((long)(b * Tlen + t) * Hg + w) * D;
        norm_row<D>(q + off, qo + off, lane, eps);
    } else if (w < 2 * Hg) {
        const int h = w - Hg;
        const long off = ((long)(b * Tlen + t) * Hg + h) * D;
        norm_row<D>(k + off, ko + off, lane, eps);
    }
    // Кумсумма внутри чанка: её считает ПЕРВЫЙ блок чанка целиком, иначе пришлось бы
    // синхронизировать блоки. При T<=chunk (декод, T=4) это ровно один блок на всю длину.
    if (t % chunk == 0) {
        const int конец = min(t + chunk, Tlen);
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            float с = 0.f;
            for (int i = t; i < конец; ++i) {
                с += g[(long)(b * Tlen + i) * H + h];
                go[(long)(b * Tlen + i) * H + h] = с;
            }
        }
    }
}

}  // namespace

std::vector<torch::Tensor> gdn_head(torch::Tensor q, torch::Tensor k, torch::Tensor g,
                                    int64_t chunk, double eps) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && g.is_cuda(), "gdn_head: CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && g.dim() == 3, "gdn_head: [B,T,Hg,D] и [B,T,H]");
    TORCH_CHECK(q.size(3) == 128, "gdn_head: пока только D=128");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && g.is_contiguous(), "gdn_head: contiguous");
    TORCH_CHECK(g.scalar_type() == torch::kFloat32, "gdn_head: g в float32");
    const int B = q.size(0), T = q.size(1), Hg = q.size(2), H = g.size(2);
    auto qo = torch::empty_like(q), ko = torch::empty_like(k);
    auto go = torch::empty_like(g);
    const int варпов = std::max(2 * Hg, (H + 31) / 32);
    const dim3 grid(T, B), blk(32 * варпов);
    const at::cuda::OptionalCUDAGuard guard(device_of(q));
    auto st = at::cuda::getCurrentCUDAStream();
    if (q.scalar_type() == torch::kHalf) {
        gdn_head_kernel<128><<<grid, blk, 0, st>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(qo.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(ko.data_ptr<at::Half>()),
            g.data_ptr<float>(), go.data_ptr<float>(), T, Hg, H, (int)chunk, (float)eps);
    } else {
        TORCH_CHECK(false, "gdn_head: пока только fp16");
    }
    return {qo, ko, go};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_head", &gdn_head, "GDN: l2norm(q), l2norm(k) и локальный cumsum(g) одним пуском",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("g"),
          pybind11::arg("chunk") = 64, pybind11::arg("eps") = 1e-6);
}
