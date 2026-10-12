# Проверка рукописного варпового мейнлупа на mma.sync.m8n8k4 (sm_70).
#
# ГЛАВНОЕ В ЭТОМ ТЕСТЕ — ПРОВЕРКА ПОКРЫТИЯ, а не совпадения значений. Первая версия мейнлупа считала
# ЧЕТВЕРТЬ своей плитки и проходила сверку с torch: сверка брала ровно те ячейки, которые называет
# измеренная таблица раскладки, то есть ровно те, которые и вычислялись. Ошибку показал только замер
# скорости. Поэтому здесь каждая ячейка выхода помечается атомарным счётчиком: тест падает, если хоть
# одна ячейка не записана или записана дважды. Совпадение значений проверяется отдельно и после.
#
# Причина той ошибки (измерена, docs/volta_hmma_map.txt): инструкция исполняется ПОКВАДРОПАРНО —
# квадропара p владеет полосами {4p..4p+3} U {4p+16..4p+19}, и четыре квадропары варпа считают четыре
# НЕЗАВИСИМЫХ произведения 8x8x4. Тождественная подача (полоса l даёт строку l и столбец l) поэтому
# считает блочную диагональ 32x32 — 256 ячеек из 1024. Лечится подачей, а не инструкцией.
import os, tempfile, torch
from torch.utils.cpp_extension import load_inline

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "volta_hmma.h"

// Плотная плитка 16x16: считаем её целиком и одновременно помечаем каждую записанную ячейку.
template <int KT>
__global__ void k_dense16(const __half* __restrict__ A, const __half* __restrict__ B,
                          float* __restrict__ D, int* __restrict__ hits) {
    constexpr int LD = KT + 4;
    __shared__ __half sA[16 * LD], sB[16 * LD];
    const int lane = threadIdx.x;
    for (int i = lane; i < 16 * KT; i += 32) { sA[(i / KT) * LD + i % KT] = A[i]; sB[(i / KT) * LD + i % KT] = B[i]; }
    __syncwarp();
    float acc[1][1][8];
    fa2_sm70::hmma_clear<1, 1>(acc);
    fa2_sm70::hmma_warp_tile<1, 1, KT>(acc, sA + fa2_sm70::hmma_a_row(lane) * LD, LD,
                                            sB + fa2_sm70::hmma_b_col(lane) * LD, LD);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int m = fa2_sm70::hmma_acc_row(lane, r), n = fa2_sm70::hmma_acc_col(lane, r);
        D[m * 16 + n] = acc[0][0][r];
        atomicAdd(&hits[m * 16 + n], 1);
    }
}

// Блочная плитка с регистровой блокировкой MB x NB и XOR-перестановкой в разделяемой памяти.
template <int MB, int NB, int BK>
__global__ void k_tile(const __half* __restrict__ A, const __half* __restrict__ B,
                       float* __restrict__ D, int* __restrict__ hits) {
    constexpr int M = MB * 16, N = NB * 16;
    __shared__ __half sA[M * BK], sB[N * BK];
    const int lane = threadIdx.x;
    for (int i = lane; i < M * BK; i += 32)
        sA[fa2_sm70::SwizzleTile<BK>::off(i / BK, (i % BK) / 8) + (i % 8)] = A[i];
    for (int i = lane; i < N * BK; i += 32)
        sB[fa2_sm70::SwizzleTile<BK>::off(i / BK, (i % BK) / 8) + (i % 8)] = B[i];
    __syncwarp();
    float acc[MB][NB][8];
    fa2_sm70::hmma_clear<MB, NB>(acc);
    fa2_sm70::hmma_warp_tile_sw<MB, NB, BK>(acc, sA, fa2_sm70::hmma_a_row(lane),
                                                 sB, fa2_sm70::hmma_b_col(lane));
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int m = i * 16 + fa2_sm70::hmma_acc_row(lane, r), n = j * 16 + fa2_sm70::hmma_acc_col(lane, r);
            D[m * N + n] = acc[i][j][r];
            atomicAdd(&hits[m * N + n], 1);
        }
}

std::vector<torch::Tensor> dense16(torch::Tensor A, torch::Tensor B) {
    auto D = torch::full({16, 16}, std::numeric_limits<float>::quiet_NaN(),
                         torch::dtype(torch::kFloat32).device(A.device()));
    auto H = torch::zeros({16, 16}, torch::dtype(torch::kInt32).device(A.device()));
    k_dense16<32><<<1, 32>>>((const __half*)A.data_ptr<at::Half>(), (const __half*)B.data_ptr<at::Half>(),
                             D.data_ptr<float>(), H.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {D, H};
}

std::vector<torch::Tensor> tile(torch::Tensor A, torch::Tensor B, int64_t MB, int64_t NB) {
    const int M = MB * 16, N = NB * 16;
    auto D = torch::full({M, N}, std::numeric_limits<float>::quiet_NaN(),
                         torch::dtype(torch::kFloat32).device(A.device()));
    auto H = torch::zeros({M, N}, torch::dtype(torch::kInt32).device(A.device()));
    auto a = (const __half*)A.data_ptr<at::Half>(); auto b = (const __half*)B.data_ptr<at::Half>();
    auto d = D.data_ptr<float>(); auto h = H.data_ptr<int>();
#define CASE(mb, nb) if (MB == mb && NB == nb) k_tile<mb, nb, 32><<<1, 32>>>(a, b, d, h); else
    CASE(1, 1) CASE(2, 2) CASE(2, 4) CASE(4, 2) CASE(4, 4) { TORCH_CHECK(false, "нет такой конфигурации"); }
#undef CASE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {D, H};
}
'''

mod = load_inline(name="volta_hmma_check",
                  cpp_sources=("std::vector<torch::Tensor> dense16(torch::Tensor A, torch::Tensor B);\n"
                               "std::vector<torch::Tensor> tile(torch::Tensor A, torch::Tensor B, int64_t MB, int64_t NB);"),
                  cuda_sources=SRC, functions=["dense16", "tile"], verbose=False,
                  extra_include_paths=[root + "/fa2_src/fmha_kernel/gemm"],
                  extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_70,code=sm_70"],
                  build_directory=os.environ.get("PROBE_BUILD", tempfile.mkdtemp()))

dev = "cuda"
torch.manual_seed(0)
fails = []


def check(name, D, H, ref):
    miss = int((H == 0).sum())
    dup = int((H > 1).sum())
    err = ((D - ref).abs() / ref.abs().clamp_min(1e-2)).max().item()
    ok = (miss == 0 and dup == 0 and err < 3e-3)
    print(f"{name:<28} покрытие: пропущено {miss:>4}, дважды {dup:>4} из {H.numel():>5} | "
          f"макс. относит. ошибка {err:.3e} | {'ОК' if ok else 'ОТКАЗ'}")
    if not ok:
        fails.append(name)


KT = 32
A = (torch.randn(16, KT, device=dev, dtype=torch.float16) * 0.4)
B = (torch.randn(16, KT, device=dev, dtype=torch.float16) * 0.4)
D, H = mod.dense16(A, B)
check("плотный 16x16", D.cpu(), H.cpu(), (A.float() @ B.float().T).cpu())

for (MB, NB) in [(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)]:
    A = (torch.randn(MB * 16, KT, device=dev, dtype=torch.float16) * 0.4)
    B = (torch.randn(NB * 16, KT, device=dev, dtype=torch.float16) * 0.4)
    D, H = mod.tile(A, B, MB, NB)
    check(f"плитка {MB*16}x{NB*16} (свиззл)", D.cpu(), H.cpu(), (A.float() @ B.float().T).cpu())

print()
if fails:
    raise SystemExit("ОТКАЗ: " + ", ".join(fails))
print("все проверки покрытия и значений пройдены")
