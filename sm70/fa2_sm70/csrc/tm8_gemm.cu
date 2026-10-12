// tm8_gemm.cu — задача 191: НАШ GPTQ-int8 (v1, g=128, асимметричный) на ДВИГАТЕЛЕ turbomind
// s884 при больших M (префилл). Нулевой объём нового device-кода: конфигурация Config_U8_g —
// клон Config_U4_g с uint8_t (ConvertKvCache<uint8_t,half> в форке УЖЕ ЕСТЬ, quantization.h:665,
// возвращает ровно q), деквант в мейнлупе x*_s[0]+_s[1] с плавлеными парами {s, -z*s}.
//
// Почему прямой пуск, а не Gemm::Run: реестр форка не знает u8-весов, а тащить его диспетчер
// со всеми архитектурами в JIT-расширение — чужая сборка. Идиом пуска списан с ДВУХ доказанных
// мест форка: kernel_impl.h::Launch (эталон, включая transpose(desc_B/V) перед to_param) и
// sm70_f16_lm_head_top1_tc_out (отгруженный прямой пуск с log_tile=4 и offsets_=nullptr;
// группа group_axis=0 с нулевыми индексами вырождается в плотный GEMM — доказано им же).
//
// Хост-конвертеры: та же инстанциация, что у e4m3 (вес: Stype=uint16_t -> Dtype=uint8_t,
// B|1, kRowMajor) и у U4_g (шкалы: uint32_t -> uint32_t, V|1, kColMajor) — Convert_v2_Impl
// зовётся напрямую, минуя реестр GetConverters (он в чужой единице трансляции).
//
// Условность нуля — v1 (z = хранимое + 1): решает ВЫЗЫВАЮЩИЙ (python подаёт zeros уже с +1).
// Гейт — положительный контроль против нашего же reconstruct-пути на БОЕВЫХ весах, сперва он.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "tm8_mainloop_p104.h"
#include "tm8_mainloop_t135.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm::sm70_s884 {

// Клон Config_U4_g (config_sm70_s884.h:133) с байтовым весом. group_axis=0 — как у
// доказанного плотного пуска top1 (Config_F16_Top1<kColMajor,0>): с offsets_=nullptr
// киндексованная A вырождается в плоскую.
template<Order raster_order>
using Config_U8_g = Sm70_s884<Operand_A<half>,           // A
                              Transform_Default,         // transform A
                              VoidOperand,               // U
                              Operand_B_Pack<uint8_t>,   // B: наш q-байт
                              Transform_HMMA_SIMT_B,     // деквант x*_s[0]+_s[1]
                              Operand_V_Pack<uint32_t>,  // V: пары {s, -z*s} в u32
                              kRowMajor,                 // order_C
                              half,                      // Tc
                              raster_order,
                              0>;

// Клон Sm70_s884::Type на ФОРКНУТОМ мейнлупе (Т135: расщеплённый барьер границы стадии).
template<Order raster_order>
struct Config_U8_g_T135 {
    using A = Operand_A<half>;
    using B = Operand_B_Pack<uint8_t>;
    using V = Operand_V_Pack<uint32_t>;
    static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_K = A::SmemCopyAtom::K;
    static constexpr auto MODE_A = Striding::kIndexed;   // group_axis=0, как Config_U8_g
    static constexpr auto MODE_B = Striding::kBlocked;
    static constexpr auto MODE_C = Striding::kBlocked;

    template<int CTA_M, int CTA_N, int CTA_K, int TG_M, int TG_N, int TG_K,
             class PolicyA, class PolicyB, int Stages, bool SplitK,
             int GroupSizeU = 1, int GroupSizeV = 1, int TILE_C_M_ = -1, int TILE_C_N_ = -1>
    struct Type {
        using MMA_Atom = SM70_MMA_884;
        using Partition = Blocked<TG_M, TG_N, kColMajor>;
        using MMA_Map_ = MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;
        using MMA = Tiled_MMA_v2<MMA_Atom, MMA_Map_>;
        using Mainloop = MainloopSm70T135<MMA, A, IteratorSm70<MODE_A, PolicyA>, Transform_Default,
                                          VoidOperand, GroupSizeU, B, IteratorSm70<MODE_B, PolicyB>,
                                          Transform_HMMA_SIMT_B, V, GroupSizeV, Stages, true, 1>;
        static constexpr int CHUNK_K = std::lcm(std::lcm(GroupSizeU, GroupSizeV), CTA_K);
        using Scheduler = SchedulerSm70<raster_order, CTA_M, CTA_N, CTA_K, CHUNK_K, SplitK, 0>;
        static constexpr int TILE_C_M = TILE_C_M_ == -1 ? CTA_M : TILE_C_M_;
        static constexpr int TILE_C_N = TILE_C_N_ == -1 ? CTA_N : TILE_C_N_;
        using Epilogue = gemm::Epilogue_<half, CTA_M, CTA_N, TILE_C_M, TILE_C_N,
                                         MMA::kThreadCount, Rearrange<MMA>,
                                         Operand_C<float, kRowMajor>, MODE_C, SplitK>;
        using Kernel = GemmUniversal<Sm70, Mainloop, Epilogue, Scheduler>;
    };
};

// Клон Sm70_s884::Type на ФОРКНУТОМ мейнлупе (P104: деквант перед HMMA по расписанию переукладчика).
template<Order raster_order>
struct Config_U8_g_P104 {
    using A = Operand_A<half>;
    using B = Operand_B_Pack<uint8_t>;
    using V = Operand_V_Pack<uint32_t>;
    static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_K = A::SmemCopyAtom::K;
    static constexpr auto MODE_A = Striding::kIndexed;   // group_axis=0, как Config_U8_g
    static constexpr auto MODE_B = Striding::kBlocked;
    static constexpr auto MODE_C = Striding::kBlocked;

    template<int CTA_M, int CTA_N, int CTA_K, int TG_M, int TG_N, int TG_K,
             class PolicyA, class PolicyB, int Stages, bool SplitK,
             int GroupSizeU = 1, int GroupSizeV = 1, int TILE_C_M_ = -1, int TILE_C_N_ = -1>
    struct Type {
        using MMA_Atom = SM70_MMA_884;
        using Partition = Blocked<TG_M, TG_N, kColMajor>;
        using MMA_Map_ = MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;
        using MMA = Tiled_MMA_v2<MMA_Atom, MMA_Map_>;
        using Mainloop = MainloopSm70P104<MMA, A, IteratorSm70<MODE_A, PolicyA>, Transform_Default,
                                          VoidOperand, GroupSizeU, B, IteratorSm70<MODE_B, PolicyB>,
                                          Transform_HMMA_SIMT_B, V, GroupSizeV, Stages, true, 1>;
        static constexpr int CHUNK_K = std::lcm(std::lcm(GroupSizeU, GroupSizeV), CTA_K);
        using Scheduler = SchedulerSm70<raster_order, CTA_M, CTA_N, CTA_K, CHUNK_K, SplitK, 0>;
        static constexpr int TILE_C_M = TILE_C_M_ == -1 ? CTA_M : TILE_C_M_;
        static constexpr int TILE_C_N = TILE_C_N_ == -1 ? CTA_N : TILE_C_N_;
        using Epilogue = gemm::Epilogue_<half, CTA_M, CTA_N, TILE_C_M, TILE_C_N,
                                         MMA::kThreadCount, Rearrange<MMA>,
                                         Operand_C<float, kRowMajor>, MODE_C, SplitK>;
        using Kernel = GemmUniversal<Sm70, Mainloop, Epilogue, Scheduler>;
    };
};

}  // namespace turbomind::gemm::sm70_s884

namespace {

namespace tg = turbomind::gemm;

// ---------------------------------------------------------------- дескрипторы (зеркало awq)
// Вес: константы конвертера известны статически (kRowMajor, HMMA_884|OPERAND_B|1, операнд B),
// поэтому ветвления is_A/is_B авторского кода схлопнуты в их B-ветку.
tg::MatrixLayout weight_src_desc(int64_t n, int64_t k) {
  // w_desc из awq_sm70_prepare ПОСЛЕ swap+~order: rows=k, cols=n, kColMajor, ld=k
  return {turbomind::kHalf, tg::kColMajor, static_cast<int>(n), static_cast<int>(k),
          static_cast<int>(k)};
}

tg::MatrixLayout scales_src_desc(int64_t n, int64_t g) {
  // s_desc ПОСЛЕ swap+~order: rows=g, cols=n, kRowMajor, ld=n
  return {turbomind::kUint32, tg::kColMajor, static_cast<int>(n), static_cast<int>(g),
          static_cast<int>(n)};
}

void fixup_bpack_desc(tg::MatrixLayout& d) {
  std::swap(d.rows, d.cols);
  d.order = ~d.order;
}

// ------------------------------------------------------------------------------- prepare
// w_u16: [n, k] uint16 (q-байт, расширенный до u16 — как tmp_u16_conv у автора),
// scales/zeros: [G, n] fp16, zeros УЖЕ с условностью v1 (+1). Возвращает
// {tm_weight u8 [k,n]-байт в упаковке, tm_scales int32 [G,n] в упаковке, meta {k_ld, q_ld}}.
std::vector<torch::Tensor> tm8_prepare(torch::Tensor w_u16, torch::Tensor scales,
                                       torch::Tensor zeros, int64_t group_size) {
  TORCH_CHECK(w_u16.is_cuda() && w_u16.dtype() == torch::kInt16 && w_u16.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kFloat16 && scales.is_contiguous());
  TORCH_CHECK(zeros.is_cuda() && zeros.dtype() == torch::kFloat16 && zeros.is_contiguous());
  const int64_t n = w_u16.size(0), k = w_u16.size(1);
  const int64_t g = k / group_size;
  TORCH_CHECK(g * group_size == k, "tm8_prepare: k % group_size != 0");
  TORCH_CHECK(scales.size(0) == g && scales.size(1) == n, "tm8_prepare: форма шкал");
  TORCH_CHECK(zeros.sizes() == scales.sizes(), "tm8_prepare: форма нулей");

  const at::cuda::OptionalCUDAGuard guard(device_of(w_u16));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // КОМПЕНСАЦИЯ РАСХОЖДЕНИЯ КОНВЕРТЕР<->ЧИТАТЕЛЬ (найдена дельта-зондом 18.08.2026).
  // u8-ветка хост-конвертера s884 (та же инстанциация, что у e4m3 — которую на sm70 никто
  // не гонял) кладёт четвёрки по k в порядке (0,2,1,3) относительно того, как их читает
  // SmemCopy_MMA_884_B + cvt_f16x4_u8. Карта снята прямым зондом (строка k промаркирована
  // своим номером, вход — дельта): период 4, свап средней пары, самообратна. Дефект был
  // НЕВИДИМ при постоянном входе (сумма по k инвариантна к перестановке) — ловится только
  // случайным x. Компенсируем той же перестановкой источника; все читатели формата наши.
  const int64_t n_pre = w_u16.size(0), k_pre = w_u16.size(1);
  auto perm = torch::tensor({0, 2, 1, 3}, torch::TensorOptions()
                                              .dtype(torch::kLong).device(w_u16.device()));
  auto w_fixed = w_u16.view({n_pre, k_pre / 4, 4}).index_select(2, perm)
                     .reshape({n_pre, k_pre}).contiguous();

  // --- вес: (n,k)-источник -> упаковка B|1. Sdesc для Convert_v2_Impl — ТРАНСПОНИРОВАННЫЙ
  // (LayoutConverterImpl::Convert делает это сам для операндов B/V; здесь — вручную).
  auto w_desc = weight_src_desc(n, k);
  fixup_bpack_desc(w_desc);  // rows=k, cols=n, kCol, ld=k
  tg::MatrixLayout w_src = tg::transpose(w_desc);  // rows=n, cols=k, kRow, ld=k — сам буфер

  tg::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::kUint8;
  k_desc.pack = tg::HMMA_884 | tg::OPERAND_B | 1;

  auto tm_weight = torch::empty({k, n}, w_u16.options().dtype(torch::kUInt8));
  {
    using Operand = tg::GetOperand<tg::HMMA_884, tg::OPERAND_B, uint16_t, tg::kRowMajor, false>::Operand;
    tg::Convert_v2_Impl<tg::Config<Operand, uint8_t, 1>>(
        w_fixed.data_ptr(), w_src, tm_weight.data_ptr(), k_desc, stream);
  }
  const int k_ld =
      tg::mk2cs<tg::kRowMajor>(tg::Packing_v2<tg::HMMA_884 | tg::OPERAND_B | 1, tg::kRowMajor>::apply(
                                   {w_src.rows, w_src.cols}))
          .x;

  // --- шкалы: плавленые пары {s, -z*s} (арифметика в half — как fuse_scales_and_zeros автора)
  auto fused = torch::stack({scales, -(zeros * scales)}, -1).reshape({g, 2 * n}).contiguous();

  auto s_desc = scales_src_desc(n, g);
  fixup_bpack_desc(s_desc);  // rows=g, cols=n, kRow, ld=n
  tg::MatrixLayout s_src = tg::transpose(s_desc);  // rows=n, cols=g, kCol, ld=n — сам буфер

  tg::MatrixLayout q_desc = s_desc;
  q_desc.pack = tg::HMMA_884 | tg::OPERAND_V | 1;

  auto tm_scales = torch::empty({g, n}, scales.options().dtype(torch::kInt32));
  {
    using Operand = tg::GetOperand<tg::HMMA_884, tg::OPERAND_V, uint32_t, tg::kColMajor, false>::Operand;
    tg::Convert_v2_Impl<tg::Config<Operand, uint32_t, 1>>(
        fused.data_ptr(), s_src, tm_scales.data_ptr(), q_desc, stream);
  }
  const int q_ld =
      tg::mk2cs<tg::kColMajor>(tg::Packing_v2<tg::HMMA_884 | tg::OPERAND_V | 1, tg::kColMajor>::apply(
                                   {s_src.rows, s_src.cols}))
          .x;

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, static_cast<int64_t>(k_ld));
  meta.index_put_({1}, static_cast<int64_t>(q_ld));
  return {tm_weight, tm_scales, meta};
}

// ------------------------------------------------------------------------------- пуск
template<class Gemm>
void run_variant(const void* A, tg::MatrixLayout dA, const void* B, tg::MatrixLayout dB,
                 const void* V, tg::MatrixLayout dV, void* D, tg::MatrixLayout dD,
                 cudaStream_t stream, int splits, int log_tile,
                 const torch::TensorOptions& opts) {
  using Sched = typename Gemm::Scheduler;

  // СЕРИЙНЫЙ split-k (контракт kernel_impl::GetMaxSplits): воркспейс на ОДИН сплит,
  // splits <= k/CHUNK_K; барьеры ядро оставляет чистыми (воркспейс движка персистентен).
  constexpr int chunk_k = Gemm::kChunkSizeK;
  splits = std::max(1, std::min(splits, (dA.cols + chunk_k - 1) / chunk_k));

  Sched sched{{dA.rows, dD.cols, dA.cols, 1}, log_tile, splits};
  sched.offsets_ = nullptr;

  tg::MatrixLayout dP = dD;
  dP.ld = tg::mk2cs<Gemm::kOrderC>(dP.rows, dP.cols).x;

  torch::Tensor barriers, partials;
  void* bar_ptr = nullptr;
  void* par_ptr = nullptr;
  if (splits > 1) {
    const int64_t tiles = static_cast<int64_t>(sched.tiles_[0]) * sched.tiles_[1];
    barriers = torch::zeros({tiles}, opts.dtype(torch::kInt32));
    partials = torch::empty({tiles * Gemm::CTA_M * Gemm::CTA_N}, opts.dtype(torch::kFloat32));
    bar_ptr = barriers.data_ptr();
    par_ptr = partials.data_ptr();
  }

  tg::MatrixCombination_v3 combin{tg::to_param(nullptr, dD), 1.f, 0.f};
  tg::EpilogueParam epi{tg::to_param(D, dD),
                        tg::to_param(par_ptr, dP),
                        static_cast<int*>(bar_ptr),
                        combin,
                        false,
                        false,
                        nullptr,
                        nullptr,
                        nullptr,
                        false,
                        tg::TileAllReduceParam{}};

  tg::MatrixLayout dU{};
  tg::GemmParam param{
      tg::to_param(const_cast<void*>(A), dA),
      tg::to_param(const_cast<void*>(B), tg::transpose(dB)),
      tg::to_param(nullptr, dU),
      tg::to_param(const_cast<void*>(V), tg::transpose(dV)),
  };

  const auto grid = sched.get_grid_shape();
  const auto block = Gemm::Impl::WARPS * WARP_SIZE;
  constexpr int smem = static_cast<int>(sizeof(typename Gemm::SharedStorage));
  auto kernel = tg::gemm_kernel<Gemm, tg::GemmParam, tg::EpilogueParam, Sched>;
  if constexpr (smem > (48 << 10)) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  kernel<<<grid, block, smem, stream>>>(param, epi, sched);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Плитки — зеркало больших-M строк реестра e4m3 (sm70_884_8.cu), GroupSizeV=128 под наш g.
using C8 = tg::sm70_s884::Config_U8_g<tg::kColMajor>;
using PD = turbomind::cache_policy::Default;
using PS = turbomind::cache_policy::Stream;
using G0 = C8::Type<128, 256, 16, 2, 4, 1, PD, PD, 2, true, 1, 128, 128, 128>::Kernel;
using G1 = C8::Type<128, 128, 16, 2, 2, 1, PD, PD, 2, true, 1, 128, 64, 128>::Kernel;
using G2 = C8::Type<64, 256, 16, 1, 4, 1, PD, PS, 2, true, 1, 128, 64, 128>::Kernel;
using G3 = C8::Type<64, 128, 32, 1, 4, 1, PD, PS, 2, true, 1, 128, 32, 128>::Kernel;
// Малые M (декодная зона): вес стримится (policy S), плитки из реестра e4m3.
using G4 = C8::Type<32, 128, 32, 1, 4, 1, PD, PS, 2, true, 1, 128>::Kernel;
using G5 = C8::Type<16, 128, 32, 1, 4, 1, PD, PS, 2, true, 1, 128>::Kernel;
using G6 = C8::Type<8, 128, 64, 1, 4, 1, PD, PS, 2, true, 1, 128>::Kernel;
using G7 = C8::Type<8, 256, 32, 1, 4, 1, PD, PS, 2, true, 1, 128>::Kernel;
// Свип-оси кукбука (без новых тел): TG-разбиение, Stages=3, политика S на B.
using G8 = C8::Type<64, 256, 16, 2, 2, 1, PD, PS, 2, true, 1, 128, 64, 128>::Kernel;
using G9 = C8::Type<64, 256, 16, 1, 4, 1, PD, PS, 3, true, 1, 128, 64, 128>::Kernel;
using G10 = C8::Type<96, 128, 32, 2, 2, 1, PD, PS, 2, true, 1, 128, 48, 128>::Kernel;
using G11 = C8::Type<128, 256, 16, 2, 4, 1, PD, PS, 2, true, 1, 128, 128, 128>::Kernel;
using G12 = C8::Type<64, 128, 32, 2, 2, 1, PD, PS, 2, true, 1, 128, 32, 128>::Kernel;
// GmemLookahead=2: глубже предвыборка gmem -- топит пролог-пачку адресов в тень HMMA
// (укладчик: канал TENSOR 100%, RecMII=0; дыры фактического расписания 27-44 команды).
using G13 = C8::Type<64, 256, 16, 1, 4, 1, PD, PS, 2, true, 1, 128, 64, 128, 2>::Kernel;
using G14 = C8::Type<128, 128, 16, 2, 2, 1, PD, PD, 2, true, 1, 128, 64, 128, 2>::Kernel;
// Половинный аккумулятор на варп (32x64 -> 64 рег): воздух для укладки при пределе ISA 255.
using G15 = C8::Type<32, 256, 32, 1, 4, 1, PD, PS, 2, true, 1, 128, 32, 128>::Kernel;
using C8T = tg::sm70_s884::Config_U8_g_T135<tg::kColMajor>;
using C8P = tg::sm70_s884::Config_U8_g_P104<tg::kColMajor>;
using G17 = C8P::Type<64, 256, 16, 1, 4, 1, PD, PS, 2, true, 1, 128, 64, 128>::Kernel;
using G16 = C8T::Type<64, 256, 16, 1, 4, 1, PD, PS, 2, true, 1, 128, 64, 128>::Kernel;

// out [m,n] fp16; in [m,k] fp16 (строки плотные по k); variant 0..3.
void tm8_gemm(torch::Tensor out, torch::Tensor in, torch::Tensor tm_weight,
              torch::Tensor tm_scales, int64_t group_size, int64_t k_ld, int64_t q_ld,
              int64_t variant, int64_t splits, int64_t log_tile) {
  TORCH_CHECK(in.is_cuda() && in.dtype() == torch::kFloat16 && in.dim() == 2);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat16 && out.dim() == 2);
  TORCH_CHECK(in.stride(1) == 1 && out.stride(1) == 1, "tm8_gemm: строки должны быть плотными");
  const int64_t m = in.size(0), k = in.size(1);
  const int64_t g = tm_scales.size(0), n = tm_scales.size(1);
  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "tm8_gemm: форма выхода");
  TORCH_CHECK(g * group_size == k, "tm8_gemm: g*group_size != k");
  if (m == 0 || n == 0) return;

  const at::cuda::OptionalCUDAGuard guard(device_of(in));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  tg::MatrixLayout dA{turbomind::kHalf, tg::kRowMajor, static_cast<int>(m),
                      static_cast<int>(k), static_cast<int>(in.stride(0))};

  auto dB = weight_src_desc(n, k);
  fixup_bpack_desc(dB);
  dB.type = turbomind::kUint8;
  dB.pack = tg::HMMA_884 | tg::OPERAND_B | 1;
  dB.ld = static_cast<int>(k_ld);

  auto dV = scales_src_desc(n, g);
  fixup_bpack_desc(dV);
  dV.pack = tg::HMMA_884 | tg::OPERAND_V | 1;
  dV.ld = static_cast<int>(q_ld);

  tg::MatrixLayout dD{turbomind::kHalf, tg::kRowMajor, static_cast<int>(m),
                      static_cast<int>(n), static_cast<int>(out.stride(0))};

  switch (variant) {
    case 0: run_variant<G0>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 1: run_variant<G1>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 2: run_variant<G2>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 3: run_variant<G3>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 4: run_variant<G4>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 5: run_variant<G5>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 6: run_variant<G6>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 7: run_variant<G7>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 8: run_variant<G8>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 9: run_variant<G9>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 10: run_variant<G10>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 11: run_variant<G11>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 12: run_variant<G12>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 13: run_variant<G13>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 14: run_variant<G14>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 15: run_variant<G15>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 16: run_variant<G16>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    case 17: run_variant<G17>(in.data_ptr(), dA, tm_weight.data_ptr(), dB,
                            tm_scales.data_ptr(), dV, out.data_ptr(), dD, stream,
                            static_cast<int>(splits), static_cast<int>(log_tile),
                            in.options()); break;
    default: TORCH_CHECK(false, "tm8_gemm: неизвестный variant ", variant);
  }
}

// ------------------------------------------------------------------- reconstruct-из-tm8
// Обратная распаковка pack -> Wt [n, k] fp16 с деквантом (для M>512: cuBLAS с opB=T).
// Формула раскладки снята ЧИСЛОМ (4 прогона по байтам линейного индекса) и подтверждена
// на 100% позиций двух форм: pos = (n>>5)*32k + (k>>3)*256 + (n&31)*8 + (k&7).
// Шкалы/нули — штатные тензоры слоя (scales fp16 [G,n], qzeros int32 [G,n/4], z=хран+1
// при v1); pack при полной переукладке остаётся ЕДИНСТВЕННОЙ копией веса.
__global__ void tm8_reconstruct_kernel(half* __restrict__ Wt, const uint8_t* __restrict__ P,
                                       const int32_t* __restrict__ qz,
                                       const half* __restrict__ sc, int64_t total, int n,
                                       int khi_cnt, int zplus) {
  const int64_t pos4 = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * 4;
  if (pos4 >= total) return;
  const uchar4 q = *reinterpret_cast<const uchar4*>(P + pos4);
  const int klo = pos4 & 7;                      // 0 или 4: четвёрка внутри фрагмента 8
  const int nlo = (pos4 >> 3) & 31;
  const int64_t rest = pos4 >> 8;
  const int khi = static_cast<int>(rest % khi_cnt);
  const int nn = static_cast<int>(rest / khi_cnt) * 32 + nlo;
  const int kk = khi * 8 + klo;
  const int g = kk >> 7;                          // GroupSize 128
  const float s = __half2float(sc[(int64_t)g * n + nn]);
  const int z = ((qz[(int64_t)g * (n >> 2) + (nn >> 2)] >> (8 * (nn & 3))) & 0xFF) + zplus;
  const float zs = -static_cast<float>(z) * s;
  const int64_t k_total = (int64_t)khi_cnt * 8;
  half2* out = reinterpret_cast<half2*>(Wt + (int64_t)nn * k_total + kk);
  out[0] = __floats2half2_rn(fmaf(q.x, s, zs), fmaf(q.y, s, zs));
  out[1] = __floats2half2_rn(fmaf(q.z, s, zs), fmaf(q.w, s, zs));
}

// [14.09, ПЕРЕБОР УСКОРЕНИЙ, ВАРИАНТ 4] БЫСТРЫЙ reconstruct: старое ядро (выше, оставлено как
// reconstruct_v1) писало Wt кусками по 16 Б на строку n (раскладка pack тайлит 32 n x 8 k) --
// 0.18 ТБ/с при полосе 0.9. Здесь плитка 32(n) x 128(k) = 4 КиБ pack ПОДРЯД (16 фрагментов по 256 Б),
// транспонирование через разделяемую и запись строк Wt[n, k0..k0+127] по 256 Б слитно; один g
// (= k0/128) на плитку -> одна шкала/ноль на n. Побитово совпадает с v1 на всех боевых формах и
// обоих форматах нулей (scratchpad/recon2/test_recon2.py); время x3.7-4.0 (0.69-0.73 ТБ/с):
// gate_up 1.465 -> 0.366 мс, down 0.729 -> 0.186, in_qkv 0.434 -> 0.111.
constexpr int RC_KC = 128, RC_NB = 32, RC_PADH = 8;
__global__ void __launch_bounds__(256) tm8_reconstruct2_kernel(
    half* __restrict__ Wt, const uint8_t* __restrict__ P, const int32_t* __restrict__ qz,
    const half* __restrict__ sc, int n, int k, int zplus) {
  __shared__ __align__(16) half tile[RC_NB][RC_KC + RC_PADH];
  const int kb = blockIdx.x, nb = blockIdx.y, t = threadIdx.x;
  const int k0 = kb * RC_KC, n0 = nb * RC_NB;
  const uint8_t* src = P + (size_t)nb * RC_NB * k + (size_t)(k0 >> 3) * 256;
  const uint4 q = *reinterpret_cast<const uint4*>(src + (size_t)t * 16);   // 16 байт: 2 n x 8 k
  const int f = t >> 4;                 // фрагмент 0..15 -> k = k0 + 8f .. +7
  const int nl = (t & 15) * 2;          // n&31 первой из двух n
  const int g = k0 >> 7;
  const uint8_t* qb = reinterpret_cast<const uint8_t*>(&q);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int nn = n0 + nl + j;
    const float s = __half2float(sc[(size_t)g * n + nn]);
    const int z = ((qz[(size_t)g * (n >> 2) + (nn >> 2)] >> (8 * (nn & 3))) & 0xFF) + zplus;
    const float zs = -static_cast<float>(z) * s;
    half2* row = reinterpret_cast<half2*>(&tile[nl + j][8 * f]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      row[i] = __floats2half2_rn(fmaf(qb[8 * j + 2 * i], s, zs), fmaf(qb[8 * j + 2 * i + 1], s, zs));
  }
  __syncthreads();
#pragma unroll
  for (int p = 0; p < 2; ++p) {        // 32 строк x 256 Б: 16 потоков на строку по 16 Б, два прохода
    const int r = p * 16 + (t >> 4), c = (t & 15) * 8;
    const uint4 v = *reinterpret_cast<const uint4*>(&tile[r][c]);
    *reinterpret_cast<uint4*>(Wt + (size_t)(n0 + r) * k + k0 + c) = v;
  }
}

// Wt [n, k] fp16 (потребитель: x @ Wt.t(), cuBLAS opB=T без копии). Быстрое ядро.
void tm8_reconstruct(torch::Tensor Wt, torch::Tensor tm_weight, torch::Tensor qzeros,
                     torch::Tensor scales, int64_t n, int64_t k, bool v2) {
  TORCH_CHECK(Wt.is_cuda() && Wt.dtype() == torch::kFloat16 && Wt.is_contiguous());
  TORCH_CHECK(Wt.numel() >= k * n && k % 128 == 0 && n % 32 == 0);
  TORCH_CHECK(tm_weight.numel() >= k * n && qzeros.dtype() == torch::kInt32 &&
              scales.dtype() == torch::kFloat16 && scales.numel() >= (k / 128) * n);
  const at::cuda::OptionalCUDAGuard guard(device_of(Wt));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(k / RC_KC), static_cast<unsigned>(n / RC_NB));
  tm8_reconstruct2_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<half*>(Wt.data_ptr<at::Half>()), tm_weight.data_ptr<uint8_t>(),
      qzeros.data_ptr<int32_t>(), reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      static_cast<int>(n), static_cast<int>(k), v2 ? 0 : 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Прежнее ядро (v1) -- оставлено для A/B и как эталон побитового совпадения.
void tm8_reconstruct_v1(torch::Tensor Wt, torch::Tensor tm_weight, torch::Tensor qzeros,
                     torch::Tensor scales, int64_t n, int64_t k, bool v2) {
  TORCH_CHECK(Wt.is_cuda() && Wt.dtype() == torch::kFloat16 && Wt.is_contiguous());
  TORCH_CHECK(Wt.numel() == k * n && k % 128 == 0 && n % 32 == 0);
  const at::cuda::OptionalCUDAGuard guard(device_of(Wt));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = k * n;
  const int64_t blocks = (total / 4 + 255) / 256;
  tm8_reconstruct_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
      reinterpret_cast<half*>(Wt.data_ptr<at::Half>()),
      tm_weight.data_ptr<uint8_t>(), qzeros.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()), total,
      static_cast<int>(n), static_cast<int>(k / 8), v2 ? 0 : 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &tm8_prepare, "GPTQ-8 -> turbomind s884 pack (weight, fused scales, meta)");
  m.def("gemm", &tm8_gemm, "fp16 x u8(g=128, asym) -> fp16 on s884, direct launch",
        py::arg("out"), py::arg("in"), py::arg("tm_weight"), py::arg("tm_scales"),
        py::arg("group_size"), py::arg("k_ld"), py::arg("q_ld"), py::arg("variant"),
        py::arg("splits") = 1, py::arg("log_tile") = 4);
  m.def("reconstruct", &tm8_reconstruct,
        "tm8 pack -> Wt [n,k] fp16 (деквант плиткой 32x128 через разделяемую; для M>512 через cuBLAS opB=T)");
  m.def("reconstruct_v1", &tm8_reconstruct_v1, "прежний reconstruct (16-байтовые записи), эталон/A/B");
  m.attr("num_variants") = 17;
}
