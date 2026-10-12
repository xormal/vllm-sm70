/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#ifdef HAS_PYTORCH
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#endif

#include <curand_kernel.h>
#include <cmath>
#include <cinttypes>
#include <vector>

#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "debug_utils.h"
#include "epilogue/epilogue_pipelined.h"
#include "epilogue/epilogue_rescale_output.h"
#include "gemm/custom_mma.h"
#include "gemm/find_default_mma.h"
#include "gemm/mma_from_smem.h"
#include "gemm_kernel_utils.h"
#include "transform/tile_smem_loader.h"

using namespace gemm_kernel_utils;

namespace {
template <typename scalar_t, typename Arch>
// [ЗАНЯТОСТЬ] Для sm_70 здесь стояла жёсткая цель 12 варпов на SM, откуда kMinBlocksPerSm = 12/4 = 3
// блока и целевые 65536/(128*3) = 170 регистров (измеренные 168). Предел Volta — 64 варпа, то есть
// ограничение чисто искусственное. Профиль forward: занятость варпов 16.4% при tensor% 29.35 и IPC
// 0.45, ни один простой не доминирует — упор именно в занятость (docs/SM70_KERNEL_PLAYBOOK.md §21).
// Цель перебирается через -DFMHA_FWD_WARPS_SM=N; N=16 даёт 4 блока и 128 регистров (+33% занятости).
#ifndef FMHA_FWD_WARPS_SM
#define FMHA_FWD_WARPS_SM 12
#endif
constexpr int getWarpsPerSmFw() {
  return (
      Arch::kMinComputeCapability >= 80 &&
              !cutlass::platform::is_same<scalar_t, float>::value
          ? 16
          : FMHA_FWD_WARPS_SM);
}
static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
  // source: https://stackoverflow.com/a/51549250
  return (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}
} // namespace

// If ToBatchHookType_ is supplied other than this default (which is
// never the case in the xformers library) then the user is
// defining the logic which each block uses to find its data to work on,
// with the advance_to_batch function with the following signature.
// It should return false if there is no work to do for this block.
// In general this will not work with saving for backward due to fixed layout
// for logsumexp and incompatible rngs for dropout, so is likely only useful for
// custom inference.
struct DefaultToBatchHook {
  template <typename Params>
  CUTLASS_DEVICE static bool advance_to_batch(
      Params&,
      int64_t& /* q_start */,
      int64_t& /* k_start */) {
    return true;
  }
};

template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_,
    int kQueriesPerBlock_,
    int kKeysPerBlock_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(),
    // This is quite slower on V100 for some reason
    // Set to false if you know at compile-time you will never need dropout
    bool kSupportsDropout_ = true,
    bool kSupportsBias_ = true,
    typename ToBatchHookType_ = DefaultToBatchHook,
    // Write the final output as BHSD [B,H,Sq,D] instead of the native BSHD [B,Sq,H,D]. Lets the
    // wrapper skip a ~250us post-kernel permute+contiguous (a full copy of O) -- the only difference
    // is the per-(batch,head) base offset of output_ptr; the per-query stride is o_strideM either way.
    bool kOutputBHSD_ = false,
    // Тип элемента выхода. По умолчанию совпадает со входом (half). Для ЧАСТИЧНЫХ результатов
    // split-K нужен float: свод складывает до num_splits слагаемых с весами по LSE, и округление
    // в half на каждом слагаемом удвоило бы ошибку (2.6e-04 -> ~6e-04).
    typename output_t_ = scalar_t_,
    // [Т12 / #129 -- ШИРИНА НАКОПИТЕЛЯ ВЫХОДА] НОВЫЙ ПАРАМЕТР ТОЛЬКО В КОНЕЦ СПИСКА: вставленный в
    // середину, он молча сдвинул бы позиционные аргументы всех существующих инстанцирований.
    // При kMaxK > kKeysPerBlock накопитель выхода НЕ ЖИВЁТ в регистрах и ездит через глобальную
    // память на КАЖДОЙ итерации по ключам. Замер (reports/EV_t4_kv_int12.md §3, d=512, Sq=2048,
    // Sk=16384): 8.02 ГБ записи + 8.09 ГБ чтения = 72 % ВСЕГО трафика DRAM этого ядра, и он же
    // вымывает плиту K/V из L2 (попаданий 50.8 % против 98.5 % там, где выплёскивания нет).
    // false (умолчание) -- накопитель fp32, поведение всех прежних инстанцирований НЕ МЕНЯЕТСЯ
    //                     НИ НА БИТ (тот же тип, тот же буфер, тот же машинный код);
    // true            -- накопитель в ТИПЕ ВЫХОДА (half): вдвое меньше байт, И отдельный буфер
    //                     `output_accum_ptr` пропадает вовсе (kNeedsOutputAccumulatorBuffer=false),
    //                     то есть заодно снимается рабочая область B*Sq*H*d fp32 (67 МБ при
    //                     Sq=2048, 268 МБ при Sq=8192).
    // ЦЕНА -- ТОЧНОСТЬ, и она обязана быть измерена против fp64-эталона ДО секундомера.
    bool kNarrowOutputAccum_ = false>
struct AttentionKernel {
  enum CustomMaskType {
    NoCustomMask = 0,
    CausalFromTopLeft = 1,
    CausalFromBottomRight = 2,
    NumCustomMaskTypes,
  };

  using scalar_t = scalar_t_;
  using accum_t = float;
  using lse_scalar_t = float;
  using output_t = output_t_;
  // Accumulator between 2 iterations
  // Using `accum_t` improves perf on f16 at the cost of
  // numerical errors
  // [Т12] ...и ровно этим разменом мы теперь и распоряжаемся ЯВНО, ключом kNarrowOutputAccum_.
  static constexpr bool kNarrowOutputAccum = kNarrowOutputAccum_;
  using output_accum_t = typename cutlass::platform::
      conditional<kNarrowOutputAccum_, output_t_, accum_t>::type;
  static constexpr bool kSupportsDropout = kSupportsDropout_;
  static constexpr bool kSupportsBias = kSupportsBias_;
  static constexpr bool kOutputBHSD = kOutputBHSD_;
  static constexpr int kKeysPerBlock = kKeysPerBlock_;
  static constexpr int kQueriesPerBlock = kQueriesPerBlock_;
  static constexpr int kMaxK = kMaxK_;
  static constexpr bool kIsAligned = isAligned_;
  static constexpr bool kSingleValueIteration = kMaxK <= kKeysPerBlock;
  static constexpr int32_t kAlignLSE = 32; // block size of backward
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value == 16;
  static constexpr bool kPreloadV =
      ArchTag::kMinComputeCapability >= 80 && kIsHalf;
  static constexpr bool kKeepOutputInRF = kSingleValueIteration;
  static constexpr bool kNeedsOutputAccumulatorBuffer = !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  // [Т12] Узкий накопитель складывает частичный результат ПРЯМО В ВЫХОД, и это законно только
  // если раскладка накопителя совпадает с раскладкой выхода. Итератор накопителя жёстко берёт шаг
  // строки `head_dim_value * num_heads` -- это BSHD. У BHSD-выхода шаг головы другой
  // (Sq*D вместо D), совпадения нет, и молчаливый результат был бы ТИХО НЕВЕРНЫМ, а не падением.
  static_assert(!kNarrowOutputAccum || !kOutputBHSD,
                "narrow output accumulator accumulates in place in O and needs the BSHD layout");
  static_assert(!kNarrowOutputAccum ||
                    cutlass::platform::is_same<output_t_, scalar_t_>::value,
                "narrow output accumulator: accumulator type is the OUTPUT type; with an fp32 "
                "output (split-K partials) narrowing is a no-op and the flag is a lie");
  static_assert(kQueriesPerBlock % 32 == 0, "");
  static_assert(kKeysPerBlock % 32 == 0, "");
  static constexpr int kNumWarpsPerBlock =
      kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  static constexpr int kWarpSize = 32;

  // Launch bounds
  static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int kMinBlocksPerSm =
      getWarpsPerSmFw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  struct Params {
    // Input tensors
    scalar_t* query_ptr = nullptr; // [num_queries, num_heads, head_dim]
    scalar_t* key_ptr = nullptr; // [num_keys, num_heads, head_dim]
    scalar_t* value_ptr = nullptr; // [num_keys, num_heads, head_dim_value]
    scalar_t* attn_bias_ptr = nullptr; // [num_heads, num_queries, num_keys]
    int32_t* seqstart_q_ptr = nullptr;
    int32_t* seqstart_k_ptr = nullptr;

    int32_t* seqlen_k_ptr = nullptr;
    uint32_t causal_diagonal_offset = 0;

    // Output tensors
    output_t* output_ptr = nullptr; // [num_queries, num_heads, head_dim_value]
    // [num_queries, num_heads, head_dim_value]
    output_accum_t* output_accum_ptr = nullptr;
    // [num_heads, num_queries] - can be null
    lse_scalar_t* logsumexp_ptr = nullptr;

    // Scale
    accum_t scale = 0.0;

    // Dimensions/strides
    int32_t head_dim = 0;
    int32_t head_dim_value = 0;
    int32_t num_queries = 0;
    int32_t num_keys = 0;
    int32_t num_keys_absolute = 0;

    uint8_t custom_mask_type = NoCustomMask;
    // Раздавать плитки запросов в ОБРАТНОМ порядке (тяжёлые причинные блоки -- в первую волну).
    // Ставится хостом только при причинной маске; см. advance_to_block().
    bool reverse_blocks = false;
    // [SPLIT-K] Число срезов по ключам. При causal вес плитки запросов растёт с её номером, и
    // расписание не короче САМОГО ДЛИННОГО задания: при S=2048,H=1 максимум 16 плиток ключей против
    // идеала 544/240 = 2.27, отсюда измеренные 4.1x переплаты на голову. Порядок блоков этого не
    // лечит (закон плейбука §406) -- лечит только дробление тяжёлых заданий. Срез s берёт плитки
    // [s*T/S, (s+1)*T/S) причинно-обрезанного диапазона; пустые срезы выходят сразу.
    int32_t num_splits = 1;
    int32_t split_idx = 0;
    int64_t o_split_stride = 0;      // шаг между срезами в частичном O (в элементах)
    int64_t lse_split_stride = 0;    // шаг между срезами в частичном LSE (в элементах)
    int32_t split_key_begin = 0;     // начало среза (в ключах), заполняется в advance_to_block
    // Отображённый индекс плитки запросов. blockIdx.x пересчитывается в теле ядра ещё в четырёх местах
    // (маска, окно, ALiBi), поэтому источник истины должен быть ОДИН: иначе разворот рассогласует
    // маскирование с адресацией. Заполняется в advance_to_block() ДО любого использования.
    int32_t block_x = 0;
    // Local-attention window (relative to the causal/diagonal center abs_q + causal_diagonal_offset):
    // query attends keys in [center - window_left, center + window_right]. -1 on a side = unbounded.
    // Causal past-window (the common case) is window_left = W-1, window_right = -1, causal = true.
    int32_t window_left = -1;
    int32_t window_right = -1;
    float logit_softcap = 0.f; // Gemma-style attention logit soft-cap: logit -> cap*tanh(logit/cap); 0 = off
    const float* alibi_slopes_ptr = nullptr;  // per-head ALiBi slopes [num_heads]; adds slope*(key-query) to logit
    float alibi_slope = 0.f;                   // slope for THIS head (loaded per-block in advance_to_block)

    int32_t q_strideM = 0;
    int32_t k_strideM = 0;
    int32_t v_strideM = 0;
    int32_t bias_strideM = 0;

    int32_t o_strideM = 0;

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int32_t q_strideH = 0;
    int32_t k_strideH = 0;
    int32_t kv_head_ratio = 1;   // H / Hkv: native GQA/MQA (query head -> KV head head_id/ratio)
    int32_t v_strideH = 0;
    int64_t bias_strideH = 0;

    int64_t q_strideB = 0;
    int64_t k_strideB = 0;
    int64_t v_strideB = 0;
    int64_t bias_strideB = 0;

    int32_t num_batches = 0;
    int32_t num_heads = 0;

    // dropout
    bool use_dropout = false;
    unsigned long long dropout_batch_head_rng_offset = 0;
    float dropout_prob = 0.0f;
#ifdef HAS_PYTORCH
    at::PhiloxCudaState rng_engine_inputs = at::PhiloxCudaState(0, 0);
#endif

    // Moves pointers to what we should process
    // Returns "false" if there is no work to do
    CUTLASS_DEVICE bool advance_to_block() {
      auto batch_id = blockIdx.z;
      auto head_id = blockIdx.y;
      // [ПОРЯДОК БЛОКОВ] При причинной маске блок с query_start=0 считает ОДНУ плитку ключей, а
      // последний -- все num_keys/kKeysPerBlock. Диспетчер раздаёт блоки по возрастанию blockIdx,
      // поэтому в ОГРЫЗОК последней волны попадают самые ТЯЖЁЛЫЕ блоки. Профиль прототипа: 256 блоков
      // на 240 слотов -> хвост из 16 блоков с максимальной работой на пустой машине (эфф волн 53%).
      // Разворот кладёт тяжёлые в первую полную волну, а лёгкие -- в огрызок. Это классический
      // longest-processing-time-first, для жадного планировщика он и оптимален.
      // Замер на прототипе: S=4096 +20.2%, S=8192 +29.8%, короткие формы без изменений.
      // Порядок НЕ влияет на результат: блоки независимы, каждый пишет свои строки.
      // Сетка по x = (число плиток запросов) * num_splits. Срез -- младшая компонента, чтобы блоки
      // одной плитки запросов шли подряд и делили Q через L2.
      int32_t qb = int32_t(blockIdx.x), nqb = int32_t(gridDim.x);
      if (num_splits > 1) { split_idx = qb % num_splits; qb /= num_splits; nqb /= num_splits; }
      block_x = reverse_blocks ? (nqb - 1 - qb) : qb;
      auto query_start = block_x * kQueriesPerBlock;
      if (alibi_slopes_ptr != nullptr) {
        alibi_slope = alibi_slopes_ptr[head_id];   // per-head ALiBi slope for this block
      }

      auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

      if (kSupportsDropout) {
        dropout_batch_head_rng_offset =
            batch_id * num_heads * num_queries * num_keys +
            head_id * num_queries * num_keys;
      }

      int64_t q_start = 0, k_start = 0;
      // Advance to current batch - in case of different sequence lengths
      constexpr bool kToBatchHook =
          !cutlass::platform::is_same<ToBatchHookType_, DefaultToBatchHook>::
              value;
      if (kToBatchHook) {
        // Call out to a custom implementation.
        if (!ToBatchHookType_::advance_to_batch(*this, q_start, k_start)) {
          return false;
        }
      } else if (seqstart_q_ptr != nullptr) {
        assert(seqstart_k_ptr != nullptr);
        seqstart_q_ptr += batch_id;

        q_start = seqstart_q_ptr[0];
        int64_t q_next_start = seqstart_q_ptr[1];
        int64_t k_end;
        seqstart_k_ptr += batch_id;

        if (seqlen_k_ptr) {
          k_start = seqstart_k_ptr[0];
          k_end = k_start + seqlen_k_ptr[batch_id];
        } else {
          k_start = seqstart_k_ptr[0];
          k_end = seqstart_k_ptr[1];
        }

        num_queries = q_next_start - q_start;
        num_keys = k_end - k_start;

        if (query_start >= num_queries) {
          return false;
        }
      } else {
        query_ptr += batch_id * q_strideB;
        key_ptr += batch_id * k_strideB;
        value_ptr += batch_id * v_strideB;
        output_ptr += kOutputBHSD
            ? int64_t(batch_id) * num_heads * num_queries * head_dim_value   // BHSD: [B,H,Sq,D] batch stride
            : int64_t(batch_id * num_queries) * o_strideM;                   // BSHD: [B,Sq,H,D]
        if (output_accum_ptr != nullptr) {
          output_accum_ptr +=
              int64_t(batch_id * num_queries) * (head_dim_value * num_heads);
        }
        q_start = 0;
        k_start = 0;
      }

      // Advance to the current batch / head / query_start. Native GQA/MQA: query head `head_id` reads
      // KV head `head_id / kv_head_ratio` (ratio = H/Hkv; 1 = MHA, H = MQA) -- no K/V expansion.
      query_ptr += (q_start + query_start) * q_strideM + head_id * q_strideH;
      key_ptr += k_start * k_strideM + (head_id / kv_head_ratio) * k_strideH;

      value_ptr += k_start * v_strideM + (head_id / kv_head_ratio) * v_strideH;
      output_ptr += int64_t(q_start + query_start) * o_strideM +
          (kOutputBHSD ? int64_t(head_id) * num_queries * head_dim_value      // BHSD: head stride = Sq*D
                       : int64_t(head_id) * head_dim_value);                  // BSHD: head stride = D

      if (kSupportsBias && attn_bias_ptr != nullptr) {
        attn_bias_ptr += (batch_id * bias_strideB) + (head_id * bias_strideH);
      }
      if (output_accum_ptr != nullptr) {
        output_accum_ptr +=
            int64_t(q_start + query_start) * (head_dim_value * num_heads) +
            head_id * head_dim_value;
      } else {
        // Accumulate directly in the destination buffer (eg for f32)
        // [Т12] Приведение было к `accum_t*` НАГЛУХО -- то есть эта ветка молча предполагала, что
        // накопитель всегда fp32. При узком накопителе тип совпадает с выходом, и приведение
        // обязано идти к типу НАКОПИТЕЛЯ, иначе не соберётся. Раскладка при этом совпадает: шаг
        // строки накопителя (head_dim_value * num_heads) равен o_strideM у BSHD-выхода, а
        // смещение на (batch, head) считается той же формулой -- см. static_assert ниже.
        output_accum_ptr = (output_accum_t*)output_ptr;
      }

      if (logsumexp_ptr != nullptr) {
        // lse[batch_id, head_id, query_start]
        logsumexp_ptr +=
            batch_id * lse_dim * num_heads + head_id * lse_dim + query_start;
      }

      // Custom masking
      if (custom_mask_type == CausalFromBottomRight) {
        causal_diagonal_offset = num_keys - num_queries;
      }
      // We use num_keys_absolute to index into the rng_state
      // We need this index to match between forward and backwards
      num_keys_absolute = num_keys;
      if (custom_mask_type == CausalFromTopLeft ||
          custom_mask_type == CausalFromBottomRight) {
        // the bottom row of the current block is query_start + kQueriesPerBlock
        // the last active key is then query_start + causal_diagonal_offset +
        // kQueriesPerBlock so num_keys is the min between actual num_keys and
        // this to avoid extra computations
        num_keys = cutlass::fast_min(
            int32_t(query_start + causal_diagonal_offset + kQueriesPerBlock),
            num_keys);
      }

      if (num_splits > 1) {
        const int32_t ntiles = ceil_div(num_keys, (int32_t)kKeysPerBlock);
        const int32_t lo_t = int32_t((int64_t)split_idx * ntiles / num_splits);
        const int32_t hi_t = int32_t((int64_t)(split_idx + 1) * ntiles / num_splits);
        if (lo_t >= hi_t) {
          return false;                     // пустой срез: LSE остаётся -inf, свод его пропустит
        }
        split_key_begin = lo_t * (int32_t)kKeysPerBlock;
        num_keys = cutlass::fast_min(num_keys, hi_t * (int32_t)kKeysPerBlock);
        output_ptr += split_idx * o_split_stride;
        if (logsumexp_ptr != nullptr) logsumexp_ptr += split_idx * lse_split_stride;
        // НАКОПИТЕЛЬ ТОЖЕ СВОЙ У КАЖДОГО СРЕЗА. Без этой строки все срезы пишут в один буфер
        // накопителя и затирают друг друга -- при d > kKeysPerBlock (выход НЕ живёт в регистрах)
        // это и был источник relL2 1.2e+01..4.8e+01, из-за которого расщепление запретили при
        // d>128 целиком. Шаг тот же o_split_stride: у накопителя раскладка BSHD [ns,B,Sq,H,D],
        // у выхода BHSD [ns,B,H,Sq,D] -- РАЗНЫЙ порядок осей, но одинаковое число элементов среза.
        if (output_accum_ptr != nullptr) output_accum_ptr += split_idx * o_split_stride;
      }

      num_queries -= query_start;
      num_batches = 0; // no longer used after

      // If num_queries == 1, and there is only one key head we're wasting
      // 15/16th of tensor core compute In that case :
      //  - we only launch kernels for head_id % kQueriesPerBlock == 0
      //  - we iterate over heads instead of queries (strideM = strideH)
      if (num_queries == 1 && k_strideH == 0 && v_strideH == 0) {
        if (head_id % kQueriesPerBlock != 0)
          return false;
        q_strideM = q_strideH;
        num_queries = num_heads;
        num_heads = 1; // unused but here for intent
        // remove causal since n_query = 1
        // otherwise, offset would change with head !
        custom_mask_type = NoCustomMask;
        o_strideM = head_dim_value;
      }

      // Make sure the compiler knows these variables are the same on all
      // the threads of the warp.
      // Only worth doing if they could have been modified above.
      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      if (kSupportsBias) {
        attn_bias_ptr = warp_uniform(attn_bias_ptr);
      }
      output_ptr = warp_uniform(output_ptr);
      output_accum_ptr = warp_uniform(output_accum_ptr);
      logsumexp_ptr = warp_uniform(logsumexp_ptr);
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);
      num_heads = warp_uniform(num_heads);
      o_strideM = warp_uniform(o_strideM);
      custom_mask_type = warp_uniform(custom_mask_type);
      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      // split-K: срез -- МЛАДШАЯ компонента x, чтобы блоки одной плитки запросов шли подряд и делили
      // Q через L2 (перечитывание почти бесплатно: ncu показал L2 hit 97.7%, HBM 3.26% от пика).
      return dim3(
          ceil_div(num_queries, (int32_t)kQueriesPerBlock) * (num_splits > 1 ? num_splits : 1),
          num_heads,
          num_batches);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  struct MM0 {
    /*
      In this first matmul, we compute a block of `Q @ K.T`.
      While the calculation result is still hot in registers, we update
      `mi`, `m_prime`, `s_prime` in shared-memory, and then store this value
      into a shared-memory ("AccumulatorSharedStorage") that is used later as
      operand A for the second matmul (see MM1)
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            scalar_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment;
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    // WARP-TILE WIDTH IS A LAW, NOT A CONSTANT (this is the kKeysPerBlock=256 wall).
    // Operand A here is the Q tile, kQueriesPerBlock(M) x ThreadK(=32) elements. In the raked thread
    // map its contiguous extent is 32/8 = 4 vector accesses against a WarpThreadArrangement of 4, so
    // WarpAccessIterations::kContiguous is exactly 1 -- the warps CANNOT be spread along contiguous,
    // they must all fit along strided, which holds kQueriesPerBlock/8 accesses. Hence
    //     warps  =  (kQ/32) * (kK/WarpN)  <=  kQ/8      <=>      WarpN >= kK/4.
    // With WarpN hard-coded to 32 that holds up to kKeysPerBlock=128 and breaks at 256: the map
    // computes kWarpsContiguous=2, Iterations::kContiguous = 1/2 = 0, and cutlass fails with
    // "Number of iterations must be non-zero" -- an integer division reaching zero, the same shape of
    // wall as the narrow-N one in the backward. For every kKeysPerBlock <= 128 the expression below
    // still yields exactly 32, so all existing instantiations are unchanged.
    static constexpr int kWarpN = kKeysPerBlock / 4 > 32 ? kKeysPerBlock / 4 : 32;
    using WarpShape = cutlass::gemm::GemmShape<32, kWarpN, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        kAlignmentB,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        ArchTag, // ArchTag
        ThreadblockShape, // ThreadblockShape
        WarpShape, // WarpShape
        typename GemmType::InstructionShape, // InstructionShape
        ArchTag::kMinComputeCapability >= 80 && kIsHalf
            ? 4
            : DefaultConfig::kStages,
        typename GemmType::Operator // Operator
        >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using DefaultThreadblockMma = typename DefaultMma::ThreadblockMma;
    using Mma = typename cutlass::platform::conditional<
        kSingleValueIteration,
        typename MakeCustomMma<DefaultThreadblockMma, kMaxK>::Mma,
        DefaultThreadblockMma>::type;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    static_assert(
        MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
                MmaCore::WarpCount::kK ==
            kNumWarpsPerBlock,
        "");

    // used for efficient load of bias tile Bij from global to shared memory
    using BiasLoader = TileSmemLoader<
        scalar_t,
        cutlass::MatrixShape<kQueriesPerBlock, kKeysPerBlock>,
        MmaCore::kThreads,
        // input restriction: kv_len has to be a multiple of this value
        128 / cutlass::sizeof_bits<scalar_t>::value>;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MM1 {
    /**
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            output_accum_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA = DefaultConfig::kAlignmentA; // from smem
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using LayoutB = cutlass::layout::RowMajor;
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        LayoutB, // LayoutB,
        kAlignmentB,
        output_accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        ArchTag::kMinComputeCapability >= 80 && kIsHalf
            ? 4
            : DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Policy::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Policy::Operator::InstructionShape,
            typename DefaultGemm::Mma::Policy::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MM0::AccumulatorSharedStorage::Shape::kN, // kMaxK
            WarpIteratorA,
            false>; // kScaleOperandA
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;
    static_assert(
        WarpCount::kM * WarpCount::kN * WarpCount::kK == kNumWarpsPerBlock,
        "");

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_t>;
    using OutputTileIteratorAccum =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_accum_t>;
  };

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  // Shared storage - depends on kernel params
  struct ScalingCoefs {
    cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> s_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
    cutlass::Array<accum_t, kQueriesPerBlock> out_rescale;
    cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>
        addition_storage;
  };

  struct SharedStorageEpilogueAtEnd : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::BiasLoader::SmemTile bias;
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::Mma::SharedStorage mm1;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }
  };

  struct SharedStorageEpilogueInLoop : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::BiasLoader::SmemTile bias;
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::Mma::SharedStorage mm1;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return after_mm0.epilogue;
    }
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kSingleValueIteration || kKeepOutputInRF,
      SharedStorageEpilogueAtEnd,
      SharedStorageEpilogueInLoop>::type;

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kAlignmentQ);
    CHECK_ALIGNED_PTR(p.key_ptr, kAlignmentK);
    CHECK_ALIGNED_PTR(p.value_ptr, kAlignmentV);
    if (kSupportsBias) {
      CHECK_ALIGNED_PTR(p.attn_bias_ptr, kAlignmentQ);
      XFORMERS_CHECK(
          p.num_batches <= 1 || p.bias_strideB % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideB)");
      XFORMERS_CHECK(
          p.num_heads <= 1 || p.bias_strideH % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideH)");
      XFORMERS_CHECK(
          p.bias_strideM % kAlignmentQ == 0,
          "attn_bias is not correctly aligned");
    }
    XFORMERS_CHECK(
        p.q_strideM % kAlignmentQ == 0,
        "query is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.k_strideM % kAlignmentK == 0,
        "key is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.v_strideM % kAlignmentV == 0,
        "value is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.q_strideH % kAlignmentQ == 0,
        "query is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.k_strideH % kAlignmentK == 0,
        "key is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.v_strideH % kAlignmentV == 0,
        "value is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.custom_mask_type < NumCustomMaskTypes,
        "invalid value for `custom_mask_type`");
    return true;
  }

  static void CUTLASS_DEVICE attention_kernel(Params& p) {
    // In this block, we will only ever:
    // - read query[query_start:query_end, :]
    // - write to output[query_start:query_end, :]

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& mi = shared_storage.mi;
    auto& out_rescale = shared_storage.out_rescale;
    const uint32_t query_start = uint32_t(p.block_x) * kQueriesPerBlock;

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      out_rescale[thread_id()] = accum_t(1.0);
      m_prime[thread_id()] =
          -cutlass::platform::numeric_limits<accum_t>::infinity();
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
    }
    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();

    auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
      using OutputTileIterator = typename MM1::OutputTileIterator;
      return OutputTileIterator(
          typename OutputTileIterator::Params{(int32_t)p.o_strideM},
          p.output_ptr,
          typename OutputTileIterator::TensorCoord{
              p.num_queries, p.head_dim_value},
          thread_id(),
          {0, col});
    };

    auto createOutputAccumIter = [&](int col) ->
        typename MM1::OutputTileIteratorAccum {
          using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
          return OutputTileIteratorAccum(
              typename OutputTileIteratorAccum::Params{
                  (int32_t)(p.head_dim_value * p.num_heads)},
              p.output_accum_ptr,
              typename OutputTileIteratorAccum::TensorCoord{
                  p.num_queries, p.head_dim_value},
              thread_id(),
              {0, col});
        };

#ifdef HAS_PYTORCH
    curandStatePhilox4_32_10_t curand_state_init;
    if (kSupportsDropout && p.use_dropout) {
      const auto seeds = at::cuda::philox::unpack(p.rng_engine_inputs);

      // each element of the attention matrix P with shape
      // (batch_sz, n_heads, n_queries, n_keys) is associated with a single
      // offset in RNG sequence. we initialize the RNG state with offset that
      // starts at the beginning of a (n_queries, n_keys) matrix for this
      // block's batch_id and head_id
      // initializing rng state is very expensive, so we run once per kernel,
      // rather than once per iteration. each iteration takes a copy of the
      // initialized RNG state and offsets it as needed.
      curand_init(
          std::get<0>(seeds),
          0,
          std::get<1>(seeds) + p.dropout_batch_head_rng_offset,
          &curand_state_init);
    }
#endif

    // Iterate through keys. Sliding window: skip key blocks entirely older than the window of this
    // query block's FIRST query (abs key < q_block_start - window + 1) -> O(Sk*window) not O(Sk^2),
    // and (critically) never process a leading block that is fully masked for every query (which
    // would make the running max stay -inf and produce nan). The per-element mask above trims the
    // exact upper (causal) and lower (window) edges within the processed blocks.
    int32_t iter_key_begin = p.split_key_begin;    // split-K: срез начинается здесь (0 без дробления)
    int32_t iter_key_end = p.num_keys;
    if (p.window_left >= 0) {   // skip leading blocks older than the FIRST query's window (abs_k < q0+off-wl)
      int32_t lo = p.block_x * kQueriesPerBlock + p.causal_diagonal_offset - p.window_left;
      if (lo > 0) iter_key_begin = (lo / kKeysPerBlock) * kKeysPerBlock;
    }
    if (p.window_right >= 0) {  // skip trailing blocks past the LAST query's window (abs_k > qlast+off+wr)
      int32_t hi = (p.block_x * kQueriesPerBlock + kQueriesPerBlock - 1) +
                   p.causal_diagonal_offset + p.window_right;
      iter_key_end = cutlass::fast_min(iter_key_end, hi + 1);
    }
    for (int32_t iter_key_start = iter_key_begin; iter_key_start < iter_key_end;
         iter_key_start += kKeysPerBlock) {
      int32_t problem_size_0_m =
          cutlass::fast_min((int32_t)kQueriesPerBlock, p.num_queries);
      int32_t problem_size_0_n = cutlass::fast_min(
          int32_t(kKeysPerBlock), p.num_keys - iter_key_start);
      int32_t const& problem_size_0_k = p.head_dim;
      int32_t const& problem_size_1_n = p.head_dim_value;
      int32_t const& problem_size_1_k = problem_size_0_n;

      auto prologueV = [&](int blockN) {
        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{typename MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        MM1::Mma::prologue(
            shared_storage.after_mm0.mm1,
            iterator_V,
            thread_id(),
            problem_size_1_k);
      };

      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter
      //
      // MATMUL: Q.K_t
      //
      // Computes the block-matrix product of:
      // (a) query[query_start:query_end, :]
      // with
      // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
      // and stores that into `shared_storage.si`
      //

      // Compute threadblock location
      cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};

      cutlass::MatrixCoord tb_offset_A{
          tb_tile_offset.m() * MM0::Mma::Shape::kM, tb_tile_offset.k()};

      cutlass::MatrixCoord tb_offset_B{
          tb_tile_offset.k(), tb_tile_offset.n() * MM0::Mma::Shape::kN};

      // Construct iterators to A and B operands
      typename MM0::IteratorA iterator_A(
          typename MM0::IteratorA::Params(
              typename MM0::MmaCore::LayoutA(p.q_strideM)),
          p.query_ptr,
          {problem_size_0_m, problem_size_0_k},
          thread_id(),
          tb_offset_A);

      typename MM0::IteratorB iterator_B(
          typename MM0::IteratorB::Params(
              typename MM0::MmaCore::LayoutB(p.k_strideM)),
          p.key_ptr + iter_key_start * p.k_strideM,
          {problem_size_0_k, problem_size_0_n},
          thread_id(),
          tb_offset_B);

      auto my_warp_id = warp_uniform(warp_id());
      auto my_lane_id = lane_id();

      // Construct thread-scoped matrix multiply
      typename MM0::Mma mma(
          shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

      typename MM0::Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size_0_k + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();

      if (kPreloadV) {
        prologueV(0);
      } else {
        MM1::Mma::drain_cp_asyncs();
      }

      typename MM0::Mma::Operator::IteratorC::TensorCoord
          iteratorC_tile_offset = {
              (tb_tile_offset.m() * MM0::Mma::WarpCount::kM) +
                  (my_warp_id % MM0::Mma::WarpCount::kM),
              (tb_tile_offset.n() * MM0::Mma::WarpCount::kN) +
                  (my_warp_id / MM0::Mma::WarpCount::kM)};

      // multiply by scaling factor
      if (kSupportsBias) {
        accum =
            cutlass::multiplies<typename MM0::Mma::FragmentC>()(p.scale, accum);
      }

      // apply attention bias if applicable
      if (kSupportsBias && p.attn_bias_ptr != nullptr) {
        // load bias tile Bij into shared memory
        typename MM0::BiasLoader::GmemTileIterator bias_iter(
            {cutlass::layout::RowMajor(p.bias_strideM)},
            // attn_bias_pointer points to matrix of size (n_queries, n_keys)
            // for the relevant batch_id and head_id
            p.attn_bias_ptr + query_start * p.bias_strideM + iter_key_start,
            {problem_size_0_m, problem_size_0_n},
            thread_id());
        cutlass::TensorRef<scalar_t, cutlass::layout::RowMajor> bias_tensor_ref(
            shared_storage.after_mm0.bias.data(),
            cutlass::layout::RowMajor(MM0::ThreadblockShape::kN));
        typename MM0::BiasLoader::SmemTileIterator smem_tile_iter(
            bias_tensor_ref, thread_id());
        MM0::BiasLoader::load(bias_iter, smem_tile_iter);

        // Pij += Bij, Pij is in register fragment and Bij is in shared memory
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            my_lane_id, my_warp_id, iteratorC_tile_offset);
        MM0::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              if (accum_m < problem_size_0_m && accum_n < problem_size_0_n) {
                accum[idx] += bias_tensor_ref.at({accum_m, accum_n});
              }
            },
            [&](int accum_m) {});
      }

      // ALiBi: add per-head slope*(key_pos - query_pos) to every (in-range) logit. accum is pre-scale
      // (iterative_softmax multiplies by p.scale), so add the bias divided by scale. Applied on ALL key
      // blocks (unlike the boundary-only mask). The -slope*query part is a per-row constant that cancels
      // under softmax, but the full term is added so the LSE is exact too.
      if (p.alibi_slope != 0.f) {
        auto query_start_ab = p.block_x * kQueriesPerBlock;
        const float alibi_over_scale = p.alibi_slope / p.scale;
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            my_lane_id, my_warp_id, iteratorC_tile_offset);
        MM0::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              if (accum_m < problem_size_0_m && accum_n < problem_size_0_n) {
                accum[idx] += alibi_over_scale *
                    float((iter_key_start + accum_n) - (query_start_ab + accum_m));
              }
            },
            [&](int accum_m) {});
      }

      // Mask out last if causal
      // This is only needed if upper-right corner of current query / key block
      // intersects the mask Coordinates of upper-right corner of current block
      // is y=query_start x=min(iter_key_start + kKeysPerBlock, num_keys)) The
      // first masked element is x = y + offset -> query_start + offset There is
      // intersection (and we need to mask) if min(iter_key_start +
      // kKeysPerBlock, num_keys)) >= query_start + offset
      // Run the per-element mask only on BOUNDARY key blocks: the causal-diagonal edge (upper), or --
      // for a window -- the lower edge (a block holding a key older than the last query's window,
      // iter_key_start < q_block_start + kQueriesPerBlock - window). Fully in-window middle blocks need
      // no masking, so they are skipped (the block-skip already dropped the fully-out-of-window ones).
      const bool is_causal = (p.custom_mask_type != NoCustomMask);
      const bool has_window = (p.window_left >= 0) || (p.window_right >= 0);
      // Run the per-element mask on the causal-diagonal boundary block, OR -- when a window is active --
      // on every processed block (the block-skip already dropped the fully-out-of-window ones; in-window
      // middle blocks no-op). This also runs for a NON-causal two-sided window (custom_mask_type == 0).
      if ((is_causal &&
           cutlass::fast_min(iter_key_start + kKeysPerBlock, p.num_keys) >=
               (query_start + p.causal_diagonal_offset)) ||
          has_window) {
        auto query_start = uint32_t(p.block_x) * kQueriesPerBlock;
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            my_lane_id, my_warp_id, iteratorC_tile_offset);
        int32_t center_col;
        const int32_t wl = p.window_left, wr = p.window_right;
        MM0::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {
              // local column of the diagonal/center for this row: abs_q + offset - iter_key_start
              center_col = query_start + accum_m + p.causal_diagonal_offset -
                  iter_key_start;
            },
            [&](int accum_m, int accum_n, int idx) {
              // mask if: causal upper (key past the diagonal), OR left window (key older than
              // center - window_left), OR right window (key newer than center + window_right).
              if ((is_causal && accum_n > center_col) ||
                  (wl >= 0 && accum_n < center_col - wl) ||
                  (wr >= 0 && accum_n > center_col + wr)) {
                accum[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_m) {});
      }
      // Update `mi` from accum stored in registers
      // Also does accum[i] <- exp(accum[i] - mi)
      iterative_softmax<typename MM0::Mma::Operator::IteratorC>(
          accum_o,
          accum,
          mi,
          m_prime,
          s_prime,
          out_rescale,
          shared_storage.addition_storage,
          my_lane_id,
          thread_id(),
          my_warp_id,
          p.num_keys - iter_key_start,
          iter_key_start == iter_key_begin,
          iteratorC_tile_offset,
          kSupportsBias ? 1.0f : p.scale,
          p.logit_softcap);

      // Output results to shared-memory
      int warp_idx_mn_0 = my_warp_id %
          (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

      MM0::B2bGemm::accumToSmem(
          shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);

      __syncthreads();

#ifdef HAS_PYTORCH
      // apply dropout (if applicable) after we've written Pij to smem.
      // dropout is applied by multiplying each element of Pij by:
      // - 0 with probability dropout_p
      // - 1 / (1 - dropout_p) with probability 1 - dropout_p
      //
      // for backward purposes we want to be able to map each element of the
      // attention matrix to the same random uniform number as the one we used
      // in forward, without needing to use the same iteration order or having
      // to store the dropout matrix. its possible to do this in registers but
      // it ends up being very slow because each thread having noncontiguous
      // strips of the Pij tile means we have to skip around a lot, and also
      // have to generate a single random number at a time
      if (kSupportsDropout && p.use_dropout) {
        auto si = shared_storage.after_mm0.si.accum_ref();
        // each thread handles a contiguous sequence of elements from Sij, all
        // coming from the same row. the reason they have to come from the same
        // row is that the sampling random numbers from a contiguous random
        // number sequence is much more efficient than jumping around, and the
        // linear offset of each element of S (the global matrix) maps to an
        // offset in a random number sequence. for S, the end of a row and the
        // beginning of the next have adjacent offsets, but for Sij, this is not
        // necessarily the case.
        const int num_threads = blockDim.x * blockDim.y * blockDim.z;
        const int threads_per_row =
            cutlass::fast_min(num_threads / problem_size_0_m, problem_size_0_n);
        const int elts_per_thread = cutlass::round_nearest(
            cutlass::ceil_div(problem_size_0_n, threads_per_row), 4);

        const int thread_i = thread_id() / threads_per_row;
        const int thread_start_j =
            (thread_id() % threads_per_row) * elts_per_thread;

        if (thread_i < problem_size_0_m && thread_start_j < problem_size_0_n) {
          curandStatePhilox4_32_10_t curand_state = curand_state_init;
          skipahead(
              static_cast<unsigned long long>(
                  (query_start + thread_i) * p.num_keys_absolute +
                  (iter_key_start + thread_start_j)),
              &curand_state);
          const float dropout_scale = 1.0 / (1.0 - p.dropout_prob);

          // apply dropout scaling to elements this thread is responsible for,
          // in chunks of 4
          for (int sij_start_col_idx = thread_start_j; sij_start_col_idx <
               cutlass::fast_min(thread_start_j + elts_per_thread,
                                 problem_size_0_n);
               sij_start_col_idx += 4) {
            const float4 rand_uniform_quad = curand_uniform4(&curand_state);

            CUTLASS_PRAGMA_UNROLL
            for (int quad_idx = 0; quad_idx < 4; ++quad_idx) {
              si.at({thread_i, sij_start_col_idx + quad_idx}) *=
                  static_cast<scalar_t>(
                      dropout_scale *
                      ((&rand_uniform_quad.x)[quad_idx] > p.dropout_prob));
            }
          }
        }
        __syncthreads(); // p.use_dropout should have same value kernel-wide
      }
#endif

      //
      // MATMUL: Attn . V
      // Run the matmul `attn @ V` for a block of attn and V.
      // `attn` is read from shared memory (in `shared_storage_si`)
      // `V` is read from global memory (with iterator_B)
      //

      const int64_t nBlockN = kSingleValueIteration
          ? 1
          : ceil_div(
                (int64_t)problem_size_1_n, int64_t(MM1::ThreadblockShape::kN));
      for (int blockN = 0; blockN < nBlockN; ++blockN) {
        int gemm_k_iterations =
            (problem_size_1_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add and store it in accum
        // (in registers)
        if (!kPreloadV) {
          __syncthreads(); // we share shmem between mma and epilogue
        }

        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{typename MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        typename MM1::Mma mma_pv(
            // operand A: Pij_dropped in shared memory
            shared_storage.after_mm0.si.accum_ref(),
            // operand B: shared memory staging area for Vj, which is loaded
            // from global memory
            shared_storage.after_mm0.mm1.operand_B_ref(),
            (int)thread_id(),
            (int)my_warp_id,
            (int)my_lane_id);
        mma_pv.set_prologue_done(kPreloadV);
        if (!kKeepOutputInRF) {
          accum_o.clear();
        }
        mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
        __syncthreads();

        if (kPreloadV && !kSingleValueIteration && blockN + 1 < nBlockN) {
          prologueV(blockN + 1);
        }

        if (!kKeepOutputInRF) {
          MM1::Mma::drain_cp_asyncs();
          DISPATCH_BOOL(
              iter_key_start == iter_key_begin, kIsFirst, ([&] {
                DISPATCH_BOOL(
                    // last PROCESSED block: use iter_key_end (a right-window shrinks the loop below
                    // num_keys), else the per-block normalization epilogue never fires -> un-normalized O.
                    (iter_key_start + kKeysPerBlock) >= iter_key_end,
                    kIsLast,
                    ([&] {
                      using DefaultEpilogue = typename MM1::DefaultEpilogue;
                      using DefaultOp =
                          typename MM1::DefaultConfig::EpilogueOutputOp;
                      // [Т12] ТИП СЧЁТА эпилога берётся у accum_t, а НЕ у типа хранения. При широком накопителе
                      // это то же самое (DefaultOp::ElementCompute там и есть float), при узком --
                      // разница между «храним в half» и «СЧИТАЕМ в half»: `s_prime`/`m_prime` лежат
                      // в accum_t, и молчаливое сползание счёта в half было бы ТИХОЙ потерей точности
                      // сверх той, за которую мы платим сознательно.
                      using ElementCompute = accum_t;
                      using EpilogueOutputOp = typename cutlass::epilogue::
                          thread::MemoryEfficientAttentionNormalize<
                              typename cutlass::platform::conditional<
                                  kIsLast::value,
                                  output_t,
                                  output_accum_t>::type,
                              output_accum_t,
                              DefaultOp::kCount,
                              typename DefaultOp::ElementAccumulator,
                              ElementCompute,
                              kIsFirst::value,
                              kIsLast::value,
                              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
                      using Epilogue = typename cutlass::epilogue::threadblock::
                          EpiloguePipelined<
                              typename DefaultEpilogue::Shape,
                              typename MM1::Mma::Operator,
                              DefaultEpilogue::kPartitionsK,
                              typename cutlass::platform::conditional<
                                  kIsLast::value,
                                  typename MM1::OutputTileIterator,
                                  typename MM1::OutputTileIteratorAccum>::type,
                              typename DefaultEpilogue::
                                  AccumulatorFragmentIterator,
                              typename DefaultEpilogue::WarpTileIterator,
                              typename DefaultEpilogue::SharedLoadIterator,
                              EpilogueOutputOp,
                              typename DefaultEpilogue::Padding,
                              DefaultEpilogue::kFragmentsPerIteration,
                              true, // IterationsUnroll
                              typename MM1::OutputTileIteratorAccum // Read
                                                                    // iterator
                              >;

                      int col = blockN * MM1::Mma::Shape::kN;
                      auto source_iter = createOutputAccumIter(col);
                      auto dest_iter = call_conditional<
                          kIsLast::value,
                          decltype(createOutputIter),
                          decltype(createOutputAccumIter)>::
                          apply(createOutputIter, createOutputAccumIter, col);
                      EpilogueOutputOp rescale(s_prime, out_rescale);
                      Epilogue epilogue(
                          shared_storage.epilogue_shared_storage(),
                          thread_id(),
                          my_warp_id,
                          my_lane_id);
                      epilogue(rescale, dest_iter, accum_o, source_iter);
                    }));
              }));
          if (!kSingleValueIteration) {
            __syncthreads();
          }
        }
      }
      __syncthreads(); // we modify `m_prime` after
    }

    if (kKeepOutputInRF) {
      constexpr bool kIsFirst = true;
      constexpr bool kIsLast = true;
      using DefaultEpilogue = typename MM1::DefaultEpilogue;
      using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
      // [Т12] ТИП СЧЁТА эпилога берётся у accum_t, а НЕ у типа хранения. При широком накопителе
      // это то же самое (DefaultOp::ElementCompute там и есть float), при узком -- разница между
      // «храним в half» и «СЧИТАЕМ в half»: `s_prime`/`m_prime` лежат в accum_t, и молчаливое
      // сползание счёта в half было бы ТИХОЙ потерей точности сверх той, за которую мы платим
      // сознательно.
      using ElementCompute = accum_t;
      using EpilogueOutputOp =
          typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
              output_t, // output
              output_accum_t, // source
              DefaultOp::kCount,
              typename DefaultOp::ElementAccumulator, // accum
              ElementCompute, // compute -- [Т12] был output_accum_t, см. выше
              kIsFirst,
              kIsLast,
              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
      using Epilogue =
          typename cutlass::epilogue::threadblock::EpiloguePipelined<
              typename DefaultEpilogue::Shape,
              typename MM1::Mma::Operator,
              DefaultEpilogue::kPartitionsK,
              typename MM1::OutputTileIterator, // destination
              typename DefaultEpilogue::AccumulatorFragmentIterator,
              typename DefaultEpilogue::WarpTileIterator,
              typename DefaultEpilogue::SharedLoadIterator,
              EpilogueOutputOp,
              typename DefaultEpilogue::Padding,
              DefaultEpilogue::kFragmentsPerIteration,
              true, // IterationsUnroll
              typename MM1::OutputTileIteratorAccum // source tile
              >;
      auto dest_iter = createOutputIter(0);
      EpilogueOutputOp rescale(s_prime, out_rescale);
      Epilogue epilogue(
          shared_storage.epilogue_shared_storage(),
          thread_id(),
          warp_id(),
          lane_id());
      MM1::Mma::drain_cp_asyncs();
      epilogue(rescale, dest_iter, accum_o);
    }

    // 7. Calculate logsumexp
    // To make the backward easier, we pad logsumexp with `inf`
    // this avoids a few bound checks, and is not more expensive during fwd
    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
      auto lse_dim = ceil_div((int32_t)p.num_queries, kAlignLSE) * kAlignLSE;
      constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
      if (thread_id() < p.num_queries) {
        p.logsumexp_ptr[thread_id()] = accum_t(mi[thread_id()] / kLog2e) +
            cutlass::fast_log(accum_t(s_prime[thread_id()]));
      } else if (thread_id() < lse_dim) {
        p.logsumexp_ptr[thread_id()] =
            cutlass::platform::numeric_limits<accum_t>::infinity();
      }
    }
  }

  template <typename WarpIteratorC>
  CUTLASS_DEVICE static void iterative_softmax(
      typename WarpIteratorC::Fragment& frag_o, // output so far
      typename WarpIteratorC::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& out_rescale,
      cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>&
          addition_storage,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int max_col,
      bool is_first,
      typename WarpIteratorC::TensorCoord const& tile_offset,
      float scaling,
      float logit_softcap) {
    /* Iterates on the accumulator and corresponding position on result matrix

    (1) Update `mi[r]` to the max value of the row `r`
    (2) In a second iteration do the following:
        (a) accum   <- exp(accum - mi)
        (b) m_prime <- exp(m_prime - mi)
        (c) s_prime <- s_prime * m_prime + sum(accum)

    All of this is done on registers, before we store all of this
    on shared memory for the next matmul with Value.
    */
    using Fragment = typename WarpIteratorC::Fragment;
    using LambdaIterator = typename DefaultMmaAccumLambdaIterator<
        WarpIteratorC,
        accum_t,
        kWarpSize>::Iterator;
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E

    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
    static constexpr int kLinesPerWarp = kQueriesPerBlock / kNumWarpsPerBlock;

    if (logit_softcap > 0.f) {
      // Gemma-style attention logit soft-capping: logit -> cap*tanh(scale*logit/cap), then softmax.
      // Applied per element in the log2 domain; masked entries (-inf) are preserved (tanh(-inf) = -1
      // would otherwise leak finite mass). One-sided masks set exactly -inf, tested below.
      const float inv = 1.0f / logit_softcap, s2 = scaling, capL2 = logit_softcap * kLog2e;
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < Fragment::kElements; ++e) {
        float v = frag[e];
        frag[e] = (v == -cutlass::platform::numeric_limits<accum_t>::infinity())
            ? v : capL2 * tanhf(s2 * v * inv);
      }
    } else {
      frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);
    }

    auto lane_offset =
        LambdaIterator::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (accum_n < max_col) {
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Having 4x atomicMax seems faster than reduce within warp
            // first...
            atomicMaxFloat(&mi[accum_m], max);
          });
    }

    // Make sure we all share the update values for `mi`
    __syncthreads();

    // Doing this `exp` is quite expensive. Let's
    // split it across the warps
    bool restore_mi_to_minus_inf = false;
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      auto m_prime_id = m_prime[id];
      auto mi_id = mi[id];
      bool changed = m_prime_id < mi_id; // `false` if both are -inf
      if (changed) {
        auto m_prime_exp = exp2f(m_prime_id - mi_id);
        out_rescale[id] = m_prime_exp;
        s_prime[id] *= m_prime_exp;
      } else {
        // It's possible that all the first-processed values of a row are masked to `-inf`
        // (with bias, OR sliding-window where a query's first in-range key block starts later
        // than the block's first query). Avoid `nan = exp2f(-inf - (-inf))` by temporarily
        // setting `mi` to 0 (restored below). Never triggers for plain causal/full (their first
        // block always has a valid key), so this is safe for those paths.
        if (mi_id == -cutlass::platform::numeric_limits<accum_t>::infinity()) {
          restore_mi_to_minus_inf = true;
          mi[id] = 0.0f;
        }
        out_rescale[id] = 1.0f;
      }
    }
    __syncthreads(); // Update output fragments
    if (kKeepOutputInRF && !is_first) {
      accum_t line_rescale;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag_o[idx] = frag_o[idx] * line_rescale;
          },
          [&](int accum_m) {});
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] =
                (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0);
          },
          [&](int accum_m) {});
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // NOTE: we could atomically add `total_row` to `s_prime`, but
              // it's faster (and deterministic) to avoid atomics here
              addition_storage
                  [accum_m + kQueriesPerBlock * tile_offset.column()] =
                      total_row;
            }
          });
    }
    __syncthreads();
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      accum_t total_row = s_prime[id];
      if (restore_mi_to_minus_inf) {
        // Restore `mi`, see above when we set `restore_mi_to_minus_inf=true`
        mi[id] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      } else {
        m_prime[id] = mi[id];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0::MmaCore::WarpCount::kN; ++i) {
        total_row += addition_storage[id + kQueriesPerBlock * i];
      }
      s_prime[id] = total_row;
    }
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x;
  }
  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.y;
  }
  static CUTLASS_DEVICE int16_t thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched_impl(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched(typename AK::Params params);
