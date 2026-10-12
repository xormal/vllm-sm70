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

#include <cmath>
#include <cinttypes>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <curand_kernel.h>

#ifdef HAS_PYTORCH
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/fast_math.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "fused_qk_gradk.h" // [CP1] hand-written interleaved QK||GradK mainloop (templated)
#include "cutlass/tensor_ref.h"

#include "debug_utils.h"
#include "gemm_kernel_utils.h"

#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/threadblock/epilogue_smem_accumulator.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "epilogue/epilogue_pipelined.h"
#include "iterators/epilogue_predicated_tile_iterator.h"

#include "gemm/custom_mma.h"
#include "gemm/find_default_mma.h"
#include "gemm/mma_accum_lambda_iterator.h"
#include "gemm/mma_from_smem.h"
#include "transform/tile_smem_loader.h"

using namespace gemm_kernel_utils;

namespace {

// [ACCUM16] Storage element for the dK/dV gmem accumulator round-trip. Accumulation stays fp32 IN
// REGISTERS; only the partial parked in gmem between query blocks narrows, so rounding happens once per
// round-trip instead of compounding inside the accumulator.
// Why: at head_dim > kBlockSizeI those partials are forced through gmem (kOutputInRF false) and re-read
// once per query block. ncu on B1H8 S2048 d256: 831.7 MB read + 413.6 MB written = 1245 MB of DRAM
// against 58.7 MB of actual tensors (21x), and the kernel is memory-LATENCY bound (long_scoreboard 2.93
// vs wait 0.81) at 8 warps/SM, so none of it is hidden. Halving the storage halves ~90% of the traffic.
// Only dK/dV narrow: the dQ tile keeps float because its path uses storeAtomicAdd, and fp16 atomics are
// emulated on Volta (measured 2x slower in an earlier round).
//
// [MEASURED 2026-07-26 -- OFF BY DEFAULT, and the result is more useful than the feature]
//   ratio = torch/ours, causal fp16, 11 shapes d=256/512:  fp32 accum 0.8843  ->  fp16 accum 0.9346
//   gradient error (max over dQ/dK/dV):                    3.74e-04           ->  1.15e-03
// dQ stays exact (3.67e-04) as designed; the error is entirely dK/dV and it GROWS with sequence length
// (8.3e-4 at S=2048, 1.15e-3 at S=4096) because that is the number of round-trips. It therefore fails the
// 1e-3 bar set before the experiment, and +5.7% does not buy a 3x accuracy loss that worsens with S.
//
// THE POINT, though: halving ~90% of the kernel's DRAM traffic bought only +5.7%. If this kernel were
// bandwidth-limited that should have been worth ~40%. It is not bandwidth-limited -- it is LATENCY
// limited (long_scoreboard 2.93 vs wait 0.81, 8 warps/SM, DRAM only 47% of peak), and fewer bytes do not
// shorten the wait for each transaction when there is nothing to overlap it with. This closes the whole
// "cut the accumulator traffic" family (fp16 storage, smem accumulators at kBJ=32, d-split accumulation):
// they all attack volume, and volume is not the binding constraint. The remaining lever is hiding the
// latency -- the off-chip prefetch schedule in research/ -- or more warps, which registers forbid.
#ifdef FMHA_BWD_ACCUM16
using fmha_accum_storage_t = cutlass::half_t;
#else
using fmha_accum_storage_t = float;
#endif

template <typename FragmentType, int32_t kNumThreads, typename StorageT = float>
struct GmemTile {
  /*
    Helper functions to efficient store/load RF to gmem

    GEMM accumulators have a particular format on A100, and
    it takes some compute/shared-memory to rearrange them to
    a RowMajor or ColumnMajor format in global memory through
    an Epilogue. The same complexity goes for loading into RF.

    This class loads/stores RF as they are, and can be used for
    efficient accumulation across gemms for instance:

    ```
    GmemTile tile;
    for (int i = 0; i < N; ++i) {
      // ...

      Fragment accum;
      if (i == 0) {
        accum.clear();
      } else {
        tile.load(accum);
      }
      mma(accum, ...);
      if (i < N-1) {
        // Store for next GEMM
        tile.store(accum);
      } else {
        // Store in tensor (eg RowMajor)
        epilogue(accum);
      }

      // ...
    }
    ```
  */

  // 128 bits per thread whatever the storage width
  using AccessType = cutlass::Array<StorageT, 16 / sizeof(StorageT)>;
  static constexpr int32_t kBytes = sizeof(AccessType);
  static constexpr int32_t kStride = kNumThreads * AccessType::kElements;
  static constexpr int32_t kNumIters =
      FragmentType::kElements / AccessType::kElements;
  static constexpr int32_t kElementsStored =
      kNumThreads * FragmentType::kElements;
  static_assert(
      FragmentType::kElements % AccessType::kElements == 0,
      "fragment not aligned on 128 bits");

  StorageT* ptr;

  CUTLASS_DEVICE void load(FragmentType& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      cutlass::arch::global_load<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        fragment[i * AccessType::kElements + j] = float(sub_fragment[j]);
      }
    }
  }

  CUTLASS_DEVICE void store(FragmentType const& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        sub_fragment[j] = StorageT(fragment[i * AccessType::kElements + j]);
      }
      cutlass::arch::global_store<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
    }
  }

  // dQ only. fp16 atomics are emulated on sm_70 (CAS loop), so this path must stay fp32.
  CUTLASS_DEVICE void storeAtomicAdd(
      FragmentType const& fragment,
      int thread_id) {
    static_assert(cutlass::platform::is_same<StorageT, float>::value,
                  "storeAtomicAdd requires fp32 storage (fp16 atomics are emulated on Volta)");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      float* gmem_ptr = ptr + thread_id * AccessType::kElements + i * kStride;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        float val = fragment[i * AccessType::kElements + j];
        float* ptr = gmem_ptr + j;
        atomicAdd(ptr, val);
      }
    }
  }
};

struct AtomicLock {
  CUTLASS_DEVICE static void acquire(
      int32_t* lock,
      int set_val,
      int thread_id) {
    if (thread_id == 0) {
      while (atomicCAS(lock, 0 /*cmp*/, set_val /*setval*/) != set_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(40);
#endif
      }
    }
    __syncthreads();
  }
  CUTLASS_DEVICE static void release(int32_t* lock, int thread_id) {
    if (thread_id == 0) {
      int status = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(status));
#else
      asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
  }
};

template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmBw() {
  bool is_half = !cutlass::platform::is_same<scalar_t, float>::value;
  if (Arch::kMinComputeCapability >= 80) {
    return is_half ? 12 : 8;
  }
  return 8;
}
} // namespace

// [W16] Width of the WARP tile's N dimension. 32 = shipped (8 warps at 128x64); 16 = the 16-warp
// probe. Compile-time because it changes every Matmul*'s WarpShape and the thread count with it.
#ifndef FMHA_BWD_WARP_N
#define FMHA_BWD_WARP_N 32
#endif

template <
    // which arch we target (eg `cutlass::arch::Sm80`)
    typename ArchTag_,
    // input/output type
    typename scalar_t_,
    // run optimized kernel because memory accesses will be aligned
    bool kIsAligned_,
    // use dropout if enabled
    bool kApplyDropout_,
    // when doing a GEMM, preload the next one (uses more shmem)
    bool kPreload_,
    // block dimensions
    int kBlockSizeI_,
    int kBlockSizeJ_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(),
    // assumes that `cu_seqlen` is None, and
    // (1) `num_queries % kBlockSizeI == 0`
    // (2) `num_keys % kBlockSizeJ == 0`
    bool kKeysQueriesAlignedToBlockSize_ = false,
    // Allows to parallelize across keys
    bool kEnableSplitKeys_ = true,
    // Optional dV-only or dQ+dK split modes for isolated backward experiments:
    //   kComputeVOnly = compute ONLY dV (keep QK + GradV; skip DOIVJ/dS/GradQ/GradK)
    //     -> a charge-2, forward-shaped ~168-reg / 3-blocks-per-SM sweep that never loads V.
    //   kSkipGradV    = compute dQ + dK only (skip the GradV stage; the dV-only sweep owns dV).
    // Both false = the full fused backward (unchanged).
    bool kComputeVOnly_ = false,
    bool kSkipGradV_ = false,
    // [TWO-PASS large-head] sink-homogeneous split (own the OUTPUT, not the axis) so the fill-split needs NO
    // cross-tile atomic sink -> no spill, no s-fold scratch. Pass KV: kSkipGradQ = compute dK+dV, skip GradQ
    // (key-split -> each CTA owns disjoint keys -> dK/dV need no atomic). Pass Q: kComputeQOnly = compute dQ,
    // skip GradV+GradK (query-split -> each CTA owns disjoint queries -> dQ needs no atomic). Two launches.
    bool kSkipGradQ_ = false,
    bool kComputeQOnly_ = false>
struct AttentionBackwardKernel {
  enum CustomMaskType {
    NoCustomMask = 0,
    CausalFromTopLeft = 1,
    CausalFromBottomRight = 2,
    NumCustomMaskTypes,
  };
  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using output_accum_t = float;
  using lse_scalar_t = float;
  using accum_t = float;
  using ArchTag = ArchTag_;
  static constexpr bool kIsAligned = kIsAligned_;
  static constexpr bool kApplyDropout = kApplyDropout_;
  static constexpr bool kPreload = kPreload_;
  static constexpr int kBlockSizeI = kBlockSizeI_;
  static constexpr int kBlockSizeJ = kBlockSizeJ_;
  static constexpr int kMaxK = kMaxK_;
  static constexpr bool kKeysQueriesAlignedToBlockSize =
      kKeysQueriesAlignedToBlockSize_;
  static constexpr bool kComputeVOnly = kComputeVOnly_;
  static constexpr bool kSkipGradV = kSkipGradV_;
  static constexpr bool kSkipGradQ = kSkipGradQ_;       // [TWO-PASS] Pass KV: compute dK+dV, skip GradQ
  static constexpr bool kComputeQOnly = kComputeQOnly_; // [TWO-PASS] Pass Q: compute dQ only, skip GradK+GradV

  static constexpr int64_t kWarpSize = 32;

  // If this is true, we store and accumulate dK/dV in RF
  // rather than going back to gmem everytime
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value <= 16;
  // [RF256 probe] This gate is STRUCTURAL (kMaxK <= kBlockSizeI), not a register-capacity computation,
  // and it is what forces the dK/dV accumulators into global memory at every head_dim > kBlockSizeI --
  // measured as ~90% of this kernel's DRAM traffic and the source of its dominant long_scoreboard stall.
  // Capacity actually says d=256 might fit: at kBJ=64, d=256, 256 threads the accumulator is
  // 64*256/256 = 64 floats/thread for each of dK and dV, i.e. 128 of the 255 registers. d=512 would be
  // 256 registers and cannot. So the ban covers d=256 only by lumping it with d=512.
  // Keeping the accumulator resident removes the round-trip ENTIRELY -- not its bytes but its latency
  // exposure at every query-block boundary -- which is the one thing that helps a latency-bound kernel
  // (halving the bytes was measured at only +5.7%). Gated: FA2SM70_BWD_RF256=1.
  //
  // [FINAL 2026-07-26 -- DEAD. The prize was unpaid work, and completing the work removed it.]
  //   probe, ONE fragment (half the accumulation):  1.372-1.501x, gradients garbage
  //   probe, multi-fragment (correct):              0.399-0.455x, dK/dV EXACT (3.67e-4 / 2.85e-4)
  //   shipped gmem path, for reference:             0.744-1.047x
  // The multi-fragment fix below was the right diagnosis -- dK and dV became exact the moment each
  // column tile got its own fragment. But the 1.4x was never residency paying off; the kernel was fast
  // because it accumulated one fragment instead of two, i.e. did half the job. With the accumulation
  // complete the fragments cost 128 registers/thread, ptxas spills hard, and it lands at 0.43x, half
  // the speed of the gmem path it was meant to beat.
  // My capacity estimate ("128 of 255, it fits") counted the fragments and not the working set, which
  // already uses all 255. The ~128-register refund from dropping the gmem machinery is real -- ptxas
  // showed it at ONE fragment (255 regs, zero spill) -- but it does not cover TWO.
  // This is the second time in this session a probe's speed came from work it was not doing (the other
  // was FA2SM70_FORCE_QSPLIT). Rule: never quote a probe's timing until its output is exact.
  //
  // [EARLIER, superseded -- kept because the reasoning below is what led to the multi-fragment fix]
  //   d=256, 7 shapes:  torch/ours 0.744-1.047  ->  1.313-1.471   (geomean of the set 0.884 -> 1.157)
  //   d=512, 4 shapes:  unchanged (the gate does not fire there) and still exact
  //   gradients at d=256: WRONG (1e+04, nan). Speed only; do not ship.
  // So removing the round-trip outright is worth ~1.4x where halving its bytes was worth 1.057x --
  // confirming the kernel is latency-bound and that this is the right axis.
  // WHY IT IS WRONG, and why the upstream gate is structural rather than a capacity test: my capacity
  // argument (64 floats/thread, 128 of 255 registers) measured the wrong quantity. The register
  // accumulator is a GEMM FRAGMENT, sized by the tile -- kBJ x kBlockSizeI = 64x128 -- while dK/dV need
  // kBJ x head_dim = 64x256. Half the head width is simply not accumulated. Registers were never the
  // constraint; FRAGMENT WIDTH is. `kMaxK <= kBlockSizeI` is exactly the statement "the fragment covers
  // the whole head".
  // ROUTE OUT -- and the arithmetic of the obvious one, so it is not built by mistake.
  // A dispatch-level head-split (run the backward twice, each pass owning half of d, so kMaxK=128 and
  // kOutputInRF is true NATURALLY with no gate hacking) is NOT free, because the split is only partly
  // legal: dV = P^T*dO, dK = dS^T*Q and dQ = dS*K all factor cleanly along d, but dP = dO*V^T SUMS over
  // the whole head, and without the complete dP there is no dS -- which both passes need. So each pass
  // must recompute S and dP at FULL width.
  // Cost in units of n^2*d: baseline is 5 (S, dP, dV, dQ, dK). Split: per pass 1 (S) + 1 (dP) + 3x0.5
  // (the outputs) = 3.5, so **7 against 5** -- the same +40% that already sank the two-pass scheme
  // (measured 0.7225 vs 0.8901). +40% work against a 1.4x prize is a wash, and it must not be built on
  // the strength of the 1.4x alone.
  // [CORRECTED 2026-07-26 -- the block that used to stand here was measured on a kernel that computed
  //  HALF the head, and every number in it was therefore wrong.] It read:
  //     "d=256 resident (RF256): 224/254/255 regs, ZERO spill ... making it resident did not add
  //      register pressure, it REMOVED the spill ... the only blocker is correctness, not capacity."
  // That was taken when OutputFragments held ONE fragment per tensor -- i.e. exactly the bug that made
  // the second column tile accumulate on top of the first. One fragment is half the registers AND half
  // the answer. With the fragment-per-column-tile fix in place (kNumColTilesV/K below), the corrected
  // build says the opposite, from the cubin:
  //     d=256, accumulator in gmem (shipped):   255 regs,   8-80 B stack
  //     d=256, accumulator resident (RF256):    128 regs,  696 B stack   <- ptxas gave up on registers
  //                                             190 regs,  512 B stack
  //     d=512 / kMaxK=65536 (gate off):         170 regs,     0 B        <- control, unchanged
  // 696 B/thread x 256 threads = 178 KB of local memory per block, walked every iteration. And the
  // wall-clock agrees: RF256 at d=256 measures 0.46x against torch where the shipped path measures
  // 0.98x -- 2.2x SLOWER. (dQ also comes out NaN, because kOutputInRF additionally gates the R7
  // whole-B dQ path whose HOST-side workspace is gated on kMaxK<=kBI and is therefore never
  // allocated; that is a second, separate bug, and fixing it would not move the spill.)
  //
  // WHY IT CANNOT FIT, in closed form. Resident cost per thread per tensor is
  //     (warp_M * warp_N / 32) * (head_dim / kBI)
  // = 32 * d/kBI at the shipped 32x32 warp tile, so 64 floats each for dK and dV at d=256, 128 total,
  // on top of a kernel that already needs 255. There are exactly two levers and both are measured shut:
  //   * kBI cannot grow -- at kBI=256 the MatmulQK accumulator alone is kBJ*kBI*2 = 32 KB and the tile
  //     does not fit 96 KB (128x128 already fails to launch);
  //   * the warp tile cannot shrink -- mma_tensor_op_sm70.h:136 asserts Shape % GemmShape<32,32,4>,
  //     because cutlass's Volta operand layouts are ORGANISED in 32x32 interleaved tiles.
  // So within cutlass, residency at d>128 is unreachable, and this is now three independent
  // measurements saying so rather than one derivation. The way through is the hand-written warp op in
  // gemm/volta_hmma.h (which has no interleaving constraint and measures 96.8% of peak) replacing
  // MmaVoltaTensorOp -- a real integration, not a template fix.
  // Keep FMHA_BWD_RF256 reachable ONLY as the A/B that produced these numbers.
#ifdef FMHA_BWD_RF256
  static constexpr bool kOutputInRF =
      kIsHalf && (kMaxK <= kBlockSizeI || (kMaxK <= 2 * kBlockSizeI && kBlockSizeI >= 128));
#else
  static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;
#endif
  // [CP1] fuse the deferred GradK with the next tile's QK into one hand-written interleaved mainloop.
  // Same gate as the deferral (kOutputInRF && !kPreload); toggle with -DFMHA_FUSE_QK_GRADK=0 for the A/B.
#ifndef FMHA_FUSE_QK_GRADK
#define FMHA_FUSE_QK_GRADK 1
#endif
  // The fused QK||GradK loop interleaves ONE k-loop for both gemms, so it is exact ONLY when QK's contraction
  // (head_dim) equals GradK's (num_queries = kBlockSizeI): compile-gate on kMaxK==kBlockSizeI (the d128
  // config) and add a RUNTIME p.head_dim==kBlockSizeI guard at the call (a d128-config model with head_dim<128
  // then falls back to the standalone deferred GradK, which is correct for any head_dim).
  static constexpr bool kFuseQKGradK =
      (kOutputInRF && !kPreload && !kComputeVOnly && (kMaxK == kBlockSizeI)) && (FMHA_FUSE_QK_GRADK);
  static_assert(
      !kPreload ||
          (kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF),
      "preload MMA not supported");
  static constexpr bool kPrologueQK = kPreload;
  static constexpr bool kPrologueGV = kPreload;
  static constexpr bool kPrologueDOV = kPreload;
  static constexpr bool kPrologueGQ = kPreload;
  static constexpr bool kPrologueGK = kPreload;

  // [W16 -- 16 warps per block by halving the WARP tile, not the threadblock tile]
  // *[measured]* occupancy at d>128 is pinned to ONE block per SM by BOTH limits at once:
  // launch__occupancy_limit_registers = 1 (255 regs x 256 threads = 65280 of 65536) and
  // launch__occupancy_limit_shared_mem = 1 (74.26 KB of 96). So "two blocks per SM" is unreachable --
  // it would need <=128 regs AND <=48 KB, and the tile needs 74. But ONE block of SIXTEEN warps is
  // perfectly legal: the 74 KB stays a single block's allocation, while 512 threads sharing the same
  // threadblock tile halve everything held per thread.
  //   warp tile <32,32> -> 8 warps, FragmentC = 32*32/32 = 32 floats/thread
  //   warp tile <32,16> -> 16 warps, FragmentC = 32*16/32 = 16 floats/thread
  // This also revises the [RESIDENCY COST] note below: 32*d/kBI assumed a 32x32 warp tile. In general
  // it is (warp_M*warp_N/32)*(d/kBI), so the warp tile is a SECOND lever on residency -- at <32,16>
  // the resident dK+dV at d=256 costs 64 floats/thread, exactly what the working d=128 path costs.
  //
  // *[measured 2026-07-26]* BLOCKED, and not by the same kind of gap as the narrow-N thread map:
  //   mma_tensor_op_sm70.h:136  static_assert(!(Shape::kM % 32) && !(Shape::kN % 32),
  //                             "Shape must be a multiple of InterleavedTileShape")
  //   with InterleavedTileShape = GemmShape<32,32,4>.
  // cutlass's Volta operand layouts (VoltaTensorOpMultiplicandCrosswise/Congruous) are ORGANISED in
  // 32x32 interleaved tiles and MmaVoltaTensorOpMultiplicandTileIterator is built on that
  // interleaving, so a sub-32 warp tile has no iterator, not merely a mis-derived one. Unlike the
  // narrow-N case there is nothing to relax -- the layout itself would have to change. The hardware
  // is still fine (32x16 is two 16x16x4 quad-pair ops), so the route to it is the hand-written warp
  // op in gemm/volta_hmma.h, which is a much larger integration than a template fix.
  // Kept parameterised at its shipped value: FMHA_BWD_WARP_N defaults to 32 and changes nothing.
  static constexpr int kWarpTileN = FMHA_BWD_WARP_N;
  static constexpr int64_t kNumWarpsPerBlock =
      (kBlockSizeI * kBlockSizeJ) / (32 * kWarpTileN);

  // [narrow-tile alignment -- TRIED AND INSUFFICIENT, kept as the map of where the wall actually is]
  // Why kBlockSizeJ=32 does not build: cutlass's warp-raked thread map puts 8 threads along the
  // contiguous axis and asks each for 8 halves (128-bit access), i.e. 64 contiguous elements. Against a
  // 32-wide tile the iteration count is 32/64 = 0, the B fragment derives to ZERO elements, and the
  // build dies in NumericArrayConverter<..., N=0>. Nothing about the geometry or the hardware forbids
  // it -- a 32-wide tile decomposes into whole m8n8k4 HMMA ops.
  // Lowering the iterator's alignment (this helper, 8 -> 4 at kBJ=32) does NOT fix it, and the failure
  // message says exactly why: the iterator duly becomes `..., 4, false>` while its thread map stays
  // `PitchLinearWarpRakedThreadMap<PitchLinearShape<32,32>, 128, PitchLinearShape<8,4>, 8>`. The two
  // numbers come from different places -- the alignment is the parameter we pass, but the thread map is
  // built by DefaultMmaCore from a hard-wired 128-bit access assumption (128/16 = 8 halves) that never
  // consults either the alignment or the tile width. Our parameter simply does not reach it.
  // So the way through is NOT a parameter: it is defining the Mma type ourselves instead of going
  // through DefaultGemm, or writing the mainloop by hand on mma.sync. Left in place because it is a
  // no-op at kBJ=64 (returns 8) and it documents the constraint the template omits.
  static constexpr int kAlignBForTile(int cfg_align, int tile_contiguous) {
    return cfg_align * 8 <= tile_contiguous ? cfg_align
                                            : (tile_contiguous / 8 > 0 ? tile_contiguous / 8 : 1);
  }

  // Compute delta for the f16 kernels
  // TODO: Figure out why it's slower on the f32 kernels
  // (something due to RF pressure?)
  // TODO: Remove condition on `kOutputInRF` - this is needed to work
  // around a compiler bug on V100, not exactly sure why but I spent
  // too much time on this already. Reproducible with
  // (B, Mq, Mkv, K) = (1, 1, 1, 136) for instance
  // [RF256] delta must follow the ORIGINAL predicate, not the override. The host precomputes delta and
  // passes it in precisely because the kernel's own delta path is not used on sm_70 when
  // kMaxK > kBlockSizeI (see the layout contract at the top of attn_bwd_cutlass.cu). Letting the RF256
  // override flip this to true makes the kernel recompute delta over our BHSD/deferred layout -- the
  // most likely cause of the wrong gradients the probe measured, and cheaper to rule out than fragment
  // width. MEASURED: it is NOT the cause -- the errors are identical to the digit with delta ownership
  // restored (1.79e+04 / 3.97e+03 / nan on the same shapes), so this decoupling is kept only because it
  // is correct in its own right (placement and delta ownership ARE independent questions). Speed with it
  // is 1.372-1.501x. Remaining suspects, in order: the GradV/GradK epilogue path taken under
  // kOutputInRF (it writes the output directly instead of through the workspace, and kNeedsAccumGradK/V
  // go false with it), and only then fragment width -- note Mma is parameterised by kMaxK (:469), not by
  // kBlockSizeI, so the "half the head is never accumulated" story needs checking against the source
  // before it is believed. FURTHER (2026-07-26, by reading, no build): the structure around the resident
  // path looks correct -- output_frags.clear() runs at the top of every key-block (:1654) and
  // writeFragsToGmem runs at its end (:1722) -- and the writer asks for the FULL width,
  // {num_keys_in_block, p.head_dim_value}, so the fragment is expected to span the whole head. That
  // weakens the fragment-width story further. Next place to look is one level down: DefaultMmaFromSmem,
  // where MatmulGradV/MatmulGradK get their Mma (:580) and hence FragmentC's actual extent. Placement of the accumulator and ownership of delta are independent decisions; the upstream
  // code only ties them together because one predicate happened to answer both questions.
  static constexpr bool kOutputInRF_native = kIsHalf && kMaxK <= kBlockSizeI;
  static constexpr bool kKernelComputesDelta =
      kIsHalf && (kOutputInRF_native || ArchTag::kMinComputeCapability != 70);

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      getWarpsPerSmBw<scalar_t, ArchTag>() / kNumWarpsPerBlock > 0
      ? getWarpsPerSmBw<scalar_t, ArchTag>() / kNumWarpsPerBlock
      : 1;   // [W16] 16 warps/block -> 8/16 == 0, and __launch_bounds__(512, 0) is invalid

  using GemmType = DefaultGemmType<ArchTag, scalar_t>;
  using DefaultConfig =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          typename GemmType::OpClass,
          ArchTag,
          scalar_t,
          scalar_t,
          scalar_t, // ElementC
          accum_t // ElementAccumulator
          >;
  static constexpr auto kOptimalAlignement = cutlass::platform::max(
      DefaultConfig::kAlignmentA,
      DefaultConfig::kAlignmentB);
  static constexpr auto kMinimumAlignment = GemmType::kMinimumAlignment;

  struct MatmulQK {
    /*
    attn_T = k_j @ q_i.transpose(-2, -1) # matmul
    attn_T = (attn_T - logsumexp[i_start:i_end].unsqueeze(1).transpose(-2,
    -1)).exp() # epilogue

    with attn_T.shape = (kBlockSizeJ, kBlockSizeI)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, kWarpTileN, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? kAlignBForTile(DefaultConfig::kAlignmentB, kBlockSizeJ)
                   : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor = false,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma =
        typename MakeCustomMma<typename DefaultMma::ThreadblockMma, kMaxK>::Mma;

    // [PREFETCH] whole-B-resident QK mma: same operands as Mma but with kNStages_ == the kblock count, so the
    // whole q (operand B) lives resident in smem and the mainloop skips all global q loads (fed by
    // prologue_whole). Reconstructed from Mma's exposed types (no MakeCustomMma change). MmaQK == Mma when off.
#ifdef FMHA_BWD_QK_WHOLEB
    static constexpr int kQWholeStages =
        (kMaxK + int(Mma::Shape::kK) - 1) / int(Mma::Shape::kK);
    using MmaQK = cutlass::gemm::threadblock::CustomMmaPipelined<
        typename Mma::Shape,
        typename Mma::IteratorA,
        typename Mma::SmemIteratorA,
        typename Mma::IteratorB,
        typename Mma::SmemIteratorB,
        typename Mma::ElementC,
        typename Mma::LayoutC,
        typename Mma::Policy,
        typename Mma::TransformA,
        typename Mma::TransformB,
        kQWholeStages,
        true>; // kWholeResident_
#else
    using MmaQK = Mma;
#endif

    // used for efficient load of bias tile (Bij) from global memory to shared
    // memory
    using BiasLoader = TileSmemLoader<
        scalar_t,
        // Bij is applied to transposed attn matrix tile (Pij.T). Bij is loaded
        // row-major but needs to have transposed shape so we get the same
        // elements.
        cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kM>,
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
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradV {
    /*
    grad_v[j_start:j_end] += attn_T @ do_i # matmul

    Dimensions: (kBlockSizeJ * kNumWarpsPerBlock, kBlockSizeI, K)
    (we might need to iterate multiple times on K)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, kWarpTileN, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? kAlignBForTile(DefaultConfig::kAlignmentB, kBlockSizeJ)
                   : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    // if dropout:
    //   for computing dVj += (Pij.T * Zij) @ dOi
    //   Pij_dropped.T = Pij.T * Zij is computed on the fly as fragments of
    //   Pij.T are loaded in. The reason we do it this way is because Pij.T and
    //   Zij are reused in later steps, while Pij_dropped.T is only needed in
    //   this step. computing Pij_dropped.T on the fly allows us to avoid
    //   keeping all 3 of Pij_dropped.T, Pij.T, and Zij in shared memory at the
    //   same time.
    // if no dropout:
    //   for computing dVj += Pij.T @ dOi
    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Operator::
                InstructionShape, // InstructionShape
            typename DefaultGemm::Mma::Operator::
                IteratorA, // RegularWarpIterator
            typename DefaultGemm::Mma::Policy // Policy
            >::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulQK::AccumulatorSharedStorage::Shape::kN,
            WarpIteratorA,
            kApplyDropout>; // kScaleOperandA

    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem =
        GmemTile<typename Mma::FragmentC, (int)kNumThreads, fmha_accum_storage_t>;
  };

  struct MatmulDOIVJ {
    /*
    doi_t_vj = do_i @ v_j.transpose(-2, -1) # matmul
    tmp = (doi_t_vj - Di.unsqueeze(1)) * attn # inplace / epilogue?
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, kWarpTileN, GemmType::WarpK>;

    using ElementC = output_t;
    using ElementAccum = accum_t;

    // no-op output op - epilogue just stores result to global memory
    using BiasGradEpilogueOutputOp =
        typename cutlass::epilogue::thread::LinearCombination<
            ElementC,
            DefaultConfig::EpilogueOutputOp::kCount,
            typename DefaultConfig::EpilogueOutputOp::ElementAccumulator,
            typename DefaultConfig::EpilogueOutputOp::ElementCompute,
            cutlass::epilogue::thread::ScaleType::Nothing>;

    using DefaultGemm = typename cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? kAlignBForTile(DefaultConfig::kAlignmentB, kBlockSizeJ)
                   : GemmType::kMinimumAlignment,
        ElementC, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        ElementAccum, // ElementAccumulator
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        BiasGradEpilogueOutputOp, // EpilogueOutputOp
        void, // ThreadblockSwizzle (not used)
        // multiple preloads, dropout Zij tile, and 3 stages push us over shared
        // memory capacity on A100. set a ceiling on number of stages to save
        // shared memory if dropout is in use.
        // [PREFETCH A2] cap DOIVJ at 2 stages (from 3): shrinks mm_doivj 24576->16384, part3 40960->32768
        // (-8192 B) to fund the persistent q home slot. DOIVJ then exposes more of its own latency, which the
        // register-transit prefetch (B) is designed to hide anyway. Gate: -DFMHA_BWD_DOIVJ_2STAGE=1.
#ifdef FMHA_BWD_DOIVJ_2STAGE
        cutlass::const_min(2, DefaultConfig::kStages), // Stages
#else
        kPreload && kApplyDropout && (kBlockSizeI * kBlockSizeJ > 64 * 64)
            ? cutlass::const_min(2, DefaultConfig::kStages)
            : DefaultConfig::kStages, // Stages
#endif
        false, // SplitKSerial
        typename GemmType::Operator,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using Mma = typename MakeCustomMma<typename DefaultGemm::Mma, kMaxK>::Mma;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        ElementAccum,
        kWarpSize>::Iterator;

    // epilogue used to write bias gradient, which is just the output of this
    // matmul with some operations applied to the fragment
    using BiasGradEpilogue = typename DefaultGemm::Epilogue;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename DefaultGemm::Mma::Operator::IteratorC,
        typename DefaultGemm::Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradQ {
    // grad_q <- tmp @ k_j
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, kWarpTileN, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? kAlignBForTile(DefaultConfig::kAlignmentB, kBlockSizeJ)
                   : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kN,
            WarpIteratorA,
            false>; // kScaleOperandA
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;

#ifdef FMHA_BWD_R7
    // [R7] measured single-orientation reassociation: dQ^T = k^T @ S^T. A = k^T (read column-major from the
    // key tile), B = S^T [key,query] (the resident p/S^T buffer), C = dQ^T [d,query]. Reads S^T (the same
    // orientation GradK reads) so the dS copy is never stored. DefaultMma pattern like QK (A global, B whole-
    // resident); output is TRANSPOSED [d,query] -> epilogue must store element-addressed.
    using R7_ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeI, GemmType::ThreadK>;
    // [R7] WarpShape <32,64> -> WarpCount 128/32 x 128/64 = 4x2 = 8 warps = kNumWarpsPerBlock (a <32,32> warp
    // would need 16 warps for the <128,128> tile -> only half computed).
    using R7_WarpShape = cutlass::gemm::GemmShape<32, 2 * kWarpTileN, GemmType::WarpK>;
    using R7_DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, cutlass::layout::ColumnMajor, DefaultConfig::kAlignmentA, // A = k^T
        scalar_t, cutlass::layout::RowMajor, DefaultConfig::kAlignmentB,    // B = S^T [key,query]
        accum_t, cutlass::layout::RowMajor,                                 // C = dQ^T [d,query]
        typename GemmType::OpClass, ArchTag,
        R7_ThreadblockShape, R7_WarpShape, typename GemmType::InstructionShape,
        DefaultConfig::kStages, typename GemmType::Operator,
        false, cutlass::gemm::SharedMemoryClearOption::kNone>;
    using R7_Mma =
        typename MakeCustomMma<typename R7_DefaultMma::ThreadblockMma, kBlockSizeJ>::Mma;
    // whole-B variant: S^T (operand B) lives resident in smem (the p/S^T buffer); mainloop reads it with no
    // global loads (pure-compute path), A = k^T pipelined from global. kNStages == the key-kblock count.
    static constexpr int kR7WholeStages =
        (kBlockSizeJ + int(R7_Mma::Shape::kK) - 1) / int(R7_Mma::Shape::kK);
    using R7_MmaWholeB = cutlass::gemm::threadblock::CustomMmaPipelined<
        typename R7_Mma::Shape,
        typename R7_Mma::IteratorA,
        typename R7_Mma::SmemIteratorA,
        typename R7_Mma::IteratorB,
        typename R7_Mma::SmemIteratorB,
        typename R7_Mma::ElementC,
        typename R7_Mma::LayoutC,
        typename R7_Mma::Policy,
        typename R7_Mma::TransformA,
        typename R7_Mma::TransformB,
        kR7WholeStages,
        true>; // kWholeResident_
    using R7_AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename R7_MmaWholeB::Operator::IteratorC, accum_t, kWarpSize>::Iterator;
#endif
  };
  struct MatmulGradK {
    // grad_k <- tmp.transpose(-2, -1) @ q_i
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, kWarpTileN, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? kAlignBForTile(DefaultConfig::kAlignmentB, kBlockSizeJ)
                   : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmemN =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulQK::AccumulatorSharedStorage::Shape::kN, // kMaxK
            WarpIteratorA,
            false>; // kScaleOperandA
    using DefaultMmaFromSmemT =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kM, // kMaxK
            WarpIteratorA,
            false, // kScaleOperandA
            kPreload>; // kTransposeA
    using DefaultMmaFromSmem = typename cutlass::platform::conditional<
        DefaultMmaFromSmemT::kIsTransposedA,
        DefaultMmaFromSmemT,
        DefaultMmaFromSmemN>::type;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem =
        GmemTile<typename Mma::FragmentC, (int)kNumThreads, fmha_accum_storage_t>;
  };

  static constexpr bool kEnableSplitKeys = kEnableSplitKeys_;

  static constexpr bool kNeedsAccumGradQ = kEnableSplitKeys ||
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  static constexpr bool kNeedsAccumGradK = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  static constexpr bool kNeedsAccumGradV = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  struct GradQTempStorage {
    int32_t lock;
    int32_t counter;
    int32_t pad[2]; // pad to 128bits
    output_accum_t buffer[MatmulGradQ::AccumTileGmem::kElementsStored];
  };

  struct Params {
    // Input tensors
    scalar_t* query_ptr = nullptr; // [Mq, nH, K]
    scalar_t* key_ptr = nullptr; // [Mk, nH, K]
    scalar_t* value_ptr = nullptr; // [Mk, nH, Kv]
    scalar_t* bias_ptr = nullptr;
    lse_scalar_t* logsumexp_ptr = nullptr; // [nH, Mq]
    scalar_t* output_ptr = nullptr; // [Mq, nH, Kv]
    scalar_t* grad_output_ptr = nullptr; // [Mq, nH, Kv]
    accum_t* delta_ptr = nullptr; // [nH, Mq]
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;

    // Output tensors
    output_t* grad_query_ptr = nullptr; //  [Mq, nH, K]
    // [R7] global scratch for S^T [B,H,Sk,Sq] -- routes the on-chip S^T through global so the whole-B mma B-fill
    // (prologue_whole via SmemIteratorB) matches its B-read swizzle (proves the reassociation; on-chip fill is the
    // later optimization). Element-addressed store at (key)*num_queries + (query) within the (batch,head) slice.
    scalar_t* r7_st_scratch = nullptr;
    // [R7] float dQ accumulator (atomicAdd on float is hardware-fast + lossless vs fp16 atomicAdd). Converted
    // to grad_query (fp16) after the kernel. [B,H,Sq,d].
    float* r7_dq_f32 = nullptr;
    // [COAL_FO d64] the FO fragorder workspace is kBI-wide in d (Sq*kBI per head), NOT d-wide like dQ -- so its
    // batch/head/query advance needs kBI-based strides, not gQ_stride* (which are d-wide). Equal at d==128 (kBI==d).
    // 0 => fall back to gQ_stride* (scattered path / d==128, where they coincide).
    int64_t r7_dq_strideB = 0, r7_dq_strideH = 0;
    int32_t r7_dq_strideM = 0;
    // [COALESCE] set by the dispatch on UNDER-FILLED shapes: route the dQ fold through the smem transpose slab
    // (coalesced RED, 4x fewer far-store granules) instead of the raw scattered emit. Off for filled shapes.
    bool coalesce_dq = false;
    // [COAL_FO] fragment-order coalescing: r7_dq_f32 is a FRAGMENT-ORDER workspace (addr=warp*kElem*32+idx*32+lane
    // -> coalesced RED by construction) and r7_perm captures fragorder->(query_local*kMaxK+d) on the first tile so
    // the dispatch can transpose fragment-order -> dQ[query,d]. nullptr disables the perm capture.
    float* r7_perm = nullptr;
    // [COAL_FO] runtime shape-gate: emit in fragment-order (coalesced, +10-31% on filled shapes) iff true; the
    // tiniest under-filled shape (B1H8-S512) keeps the scattered emit (the fixed convert-pass overhead loses there).
    bool fo_dq = false;
    output_t* grad_key_ptr = nullptr; //    [Mk, nH, K]
    output_t* grad_value_ptr = nullptr; //  [Mk, nH, Kv]
    output_t* grad_bias_ptr = nullptr;

    // Accumulators
    output_accum_t* workspace = nullptr; // [Mq, Kq] + [Mkv, Kq] + [Mkv, Kv]
    output_accum_t* workspace_gv =
        nullptr; // (will be calculated by the kernel)
    GradQTempStorage* workspace_gq =
        nullptr; // (will be calculated by the kernel)

    // Scale
    accum_t scale = 1.0f;

    // Dimensions/strides
    int32_t head_dim = -1;
    int32_t head_dim_value = -1;
    int32_t num_queries = -1;
    int32_t num_keys = -1;
    int32_t num_heads = -1;
    uint8_t custom_mask_type = NoCustomMask;

    int32_t q_strideM = -1;
    int32_t k_strideM = -1;
    int32_t v_strideM = -1;
    int32_t bias_strideM = 0;
    int32_t gO_strideM = -1;
    int32_t gB_strideM = -1;
    int8_t gQKV_strideM_multiplier = 1; // 3 for packed, 1 otherwise
    // Output row-stride overrides (>=0 wins). Default (-1) keeps the BSHD packed layout
    // (row stride = num_heads*head_dim). Set to head_dim for a BHSD-direct [B,H,S,D] layout
    // so the wrapper can skip the BSHD<->BHSD transposes (o/dQ/dK/dV read/written in place).
    int32_t o_strideM_override = -1;
    int32_t gQ_strideM_override = -1;
    int32_t gK_strideM_override = -1;
    int32_t gV_strideM_override = -1;

#ifdef HAS_PYTORCH
    // dropout
    at::PhiloxCudaState rng_engine_inputs = {0, 0};
#endif
    // RNG sequence offset based on batch_id and head_id
    unsigned long long dropout_batch_head_rng_offset = 0;
    float dropout_prob = 0.0f;

    CUTLASS_HOST_DEVICE int32_t o_strideM() const {
      return o_strideM_override >= 0 ? o_strideM_override : head_dim_value * num_heads;
    }
    CUTLASS_HOST_DEVICE int32_t gQ_strideM() const {
      return gQ_strideM_override >= 0 ? gQ_strideM_override
                                      : gQKV_strideM_multiplier * num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int32_t gK_strideM() const {
      return gK_strideM_override >= 0 ? gK_strideM_override
                                      : gQKV_strideM_multiplier * num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int32_t gV_strideM() const {
      return gV_strideM_override >= 0 ? gV_strideM_override
                                      : gQKV_strideM_multiplier * num_heads * head_dim_value;
    }

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int64_t o_strideH = -1;
    int32_t q_strideH = -1;
    int32_t k_strideH = -1;
    int32_t v_strideH = -1;
    int64_t bias_strideH = 0;
    int64_t o_strideB = -1;
    int64_t q_strideB = -1;
    int64_t k_strideB = -1;
    int64_t v_strideB = -1;
    int64_t bias_strideB = 0;
    int64_t lse_strideB = -1;
    int64_t lse_strideH = -1;
    int64_t delta_strideB = -1;
    int64_t delta_strideH = -1;
    int32_t num_batches = -1;
    int16_t num_splits_key = 1; // We use `gridDim.x` inside kernel
    int16_t num_splits_query = 1; // [Q1 query-split] gridDim.x = num_splits_key * num_splits_query; blockIdx.x
                                  // encodes (split_key = blockIdx.x/q, q_piece = blockIdx.x%q). q=1 => no-op.

    int64_t gO_strideB = 0;
    int64_t gQ_strideB = 0;
    int64_t gK_strideB = 0;
    int64_t gV_strideB = 0;
    int64_t gB_strideB = 0;
    int64_t gO_strideH = 0;
    int64_t gQ_strideH = 0;
    int64_t gK_strideH = 0;
    int64_t gV_strideH = 0;
    int64_t gB_strideH = 0;

    CUTLASS_DEVICE int16_t num_splits_key_device() const {
      // [Q1] key-split count = gridDim.x / num_splits_query (q=1 -> gridDim.x, unchanged).
      return kEnableSplitKeys ? (int16_t)(gridDim.x / num_splits_query) : 1;
    }
    CUTLASS_DEVICE int16_t split_key_device() const {
      return kEnableSplitKeys ? (int16_t)(blockIdx.x / num_splits_query) : 0;
    }
    // [Q1] which query-piece (0..num_splits_query-1) this CTA owns; q=1 -> always 0 (no-op).
    CUTLASS_DEVICE int16_t q_piece_device() const {
      return kEnableSplitKeys ? (int16_t)(blockIdx.x % num_splits_query) : 0;
    }

    CUTLASS_DEVICE bool advance_to_block() {
      int64_t batch_id = blockIdx.z;
      int32_t head_id = blockIdx.y;

      if (kNeedsAccumGradQ || kNeedsAccumGradK || kNeedsAccumGradV) {
        assert(workspace_size() == 0 || workspace != nullptr);

        workspace += (batch_id * num_heads + head_id) * workspace_strideBH();
        workspace = warp_uniform(workspace);
        workspace_gv = workspace + workspace_elements_gk();
        workspace_gq =
            (GradQTempStorage*)(workspace_gv + workspace_elements_gv());
        if (kEnableSplitKeys) {
          workspace_gv += workspace_elements_gv() * split_key_device() /
              num_splits_key_device();
          workspace += workspace_elements_gk() * split_key_device() /
              num_splits_key_device();
        }
      } else {
        workspace = nullptr;
      }

      // Advance pointers that depend on the total concatenated
      // number of queries, as `num_queries` is modified in the block
      // below
      dropout_batch_head_rng_offset =
          batch_id * (num_heads * num_queries * num_keys) +
          head_id * (num_queries * num_keys);
      logsumexp_ptr += batch_id * lse_strideB + head_id * lse_strideH;

      if (cu_seqlens_q_ptr != nullptr) {
        assert(cu_seqlens_k_ptr != nullptr);
        cu_seqlens_q_ptr += batch_id;
        cu_seqlens_k_ptr += batch_id;
        int32_t q_start = cu_seqlens_q_ptr[0];
        int32_t k_start = cu_seqlens_k_ptr[0];
        int64_t q_next_start = cu_seqlens_q_ptr[1];
        int64_t k_next_start = cu_seqlens_k_ptr[1];
        assert(q_next_start - q_start <= num_queries);
        assert(k_next_start - k_start <= num_keys);
        num_queries = q_next_start - q_start;
        num_keys = k_next_start - k_start;

        // Jump manually
        batch_id = 0;

        query_ptr += q_start * q_strideM;
        key_ptr += k_start * k_strideM;
        value_ptr += k_start * v_strideM;
        assert(bias_ptr == nullptr);
        assert(grad_bias_ptr == nullptr);
        output_ptr += q_start * o_strideM();
        grad_output_ptr += q_start * gO_strideM;
        delta_ptr += q_start;

        grad_query_ptr += q_start * gQ_strideM();
        grad_key_ptr += k_start * gK_strideM();
        grad_value_ptr += k_start * gV_strideM();
#ifdef FMHA_BWD_R7
        if (r7_dq_f32 != nullptr) r7_dq_f32 += q_start * (r7_dq_strideM ? (int64_t)r7_dq_strideM : (int64_t)gQ_strideM());
#endif
      }

      query_ptr += batch_id * q_strideB + head_id * q_strideH;
      key_ptr += batch_id * k_strideB + head_id * k_strideH;
      value_ptr += batch_id * v_strideB + head_id * v_strideH;
      if (bias_ptr != nullptr) {
        bias_ptr += batch_id * bias_strideB + head_id * bias_strideH;
      }
      output_ptr += batch_id * o_strideB + head_id * o_strideH;
      grad_output_ptr += batch_id * gO_strideB + head_id * gO_strideH;
      delta_ptr += batch_id * delta_strideB + head_id * delta_strideH;

      grad_query_ptr += batch_id * gQ_strideB + head_id * gQ_strideH;
      grad_key_ptr += batch_id * gK_strideB + head_id * gK_strideH;
      grad_value_ptr += batch_id * gV_strideB + head_id * gV_strideH;
#ifdef FMHA_BWD_R7
      if (r7_st_scratch != nullptr) {
        r7_st_scratch += ((int64_t)batch_id * num_heads + head_id) *
            (int64_t)num_keys * num_queries;
      }
      if (r7_dq_f32 != nullptr) {
        r7_dq_f32 += batch_id * (r7_dq_strideB ? r7_dq_strideB : gQ_strideB) +
            head_id * (r7_dq_strideH ? r7_dq_strideH : gQ_strideH);
      }
#endif
      if (grad_bias_ptr != nullptr) {
        grad_bias_ptr += batch_id * gB_strideB + head_id * gB_strideH;
      }

      // Some values are modified above
      // Signal to the compiler that they are the same in all threads
      // and can be stored in warp-uniform registers (Sm75+)
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);
      custom_mask_type = warp_uniform(custom_mask_type);

      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      bias_ptr = warp_uniform(bias_ptr);
      logsumexp_ptr = warp_uniform(logsumexp_ptr);
      output_ptr = warp_uniform(output_ptr);
      grad_output_ptr = warp_uniform(grad_output_ptr);
      delta_ptr = warp_uniform(delta_ptr);

      grad_query_ptr = warp_uniform(grad_query_ptr);
      grad_key_ptr = warp_uniform(grad_key_ptr);
      grad_value_ptr = warp_uniform(grad_value_ptr);
      grad_bias_ptr = warp_uniform(grad_bias_ptr);

#if 0
      PRINT_T0("[b:%d h:%d] dp[0]:%f Q:%f K:%f V:%f LSE:%f",
        int(blockIdx.z), int(blockIdx.y),
        float(delta_ptr[0]),
        float(query_ptr[0]), float(key_ptr[0]), float(value_ptr[0]),
        float(logsumexp_ptr[0])
      )
#endif
      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(num_splits_key * num_splits_query, num_heads, num_batches);
    }
    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize * kNumWarpsPerBlock, 1, 1);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gk() const {
      if (!kNeedsAccumGradK) {
        return 0;
      }
      return num_splits_key * align_up(num_keys, (int32_t)kBlockSizeJ) *
          align_up(head_dim, (int32_t)kBlockSizeI);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gv() const {
      if (!kNeedsAccumGradV) {
        return 0;
      }
      return num_splits_key * align_up(num_keys, (int32_t)kBlockSizeJ) *
          align_up(head_dim_value, (int32_t)kBlockSizeI);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gq() const {
      if (!kNeedsAccumGradQ) {
        return 0;
      }
      int num_blocks = ceil_div(num_queries, kBlockSizeI);
      int num_cols = ceil_div(head_dim, MatmulGradQ::ThreadblockShape::kN);
      return num_blocks * num_cols * sizeof(GradQTempStorage) /
          sizeof(output_accum_t);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_strideBH() const {
      // Aligned on 128bits
      return align_up(
          workspace_elements_gk() + workspace_elements_gv() +
              workspace_elements_gq(),
          int64_t(4));
    }
    CUTLASS_HOST_DEVICE int64_t workspace_size() const {
      // Returns size of buffer we need to run this kernel
      return num_batches * num_heads * workspace_strideBH() * sizeof(float);
    }
    CUTLASS_HOST_DEVICE bool should_zero_workspace() const {
      return num_splits_key > 1;
    }
  };

  // shared storage for keeping Zij matrix. not needed if we aren't using
  // dropout, in which case we use an empty array to save shared memory
  using ZijSharedStorage = typename cutlass::platform::conditional<
      kApplyDropout,
      typename MatmulQK::AccumulatorSharedStorage,
      // dummy shared storage object that takes up no space.
      typename cutlass::gemm::threadblock::AccumulatorSharedStorage<
#ifdef _WIN32
          // windows builds throw the error:
          // "type containing an unknown-size array is not allowed"
          // if we try to make Zij shared storage zero-sized.
          // To get around this just make it sized 1 on windows.
          typename cutlass::gemm::GemmShape<1, 1, 0>,
#else
          typename cutlass::gemm::GemmShape<0, 0, 0>,
#endif
          typename MatmulQK::AccumulatorSharedStorage::Element,
          typename MatmulQK::AccumulatorSharedStorage::Layout,
          typename cutlass::MatrixShape<0, 0>>>::type;

  struct SharedStoragePrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
      typename MatmulQK::Mma::SharedStorageA mm_qk_k;
    } persistent;
    union {
      struct {
        // part1 - after Q.K / dV / dO.V
        union {
          // 1. efficient load of bias tile Bij, which is then applied to Pij
          typename MatmulQK::BiasLoader::SmemTile bias;
          // 4. store Pij. it is needed:
          // - in dVj += (Pij.T * Zij) @ dOi
          // - in dSij = Pij * (dPij - Di)
          // 6. dVj += (Pij.T * Zij) @ dOi
          // 10. write to fragment
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };
        // 5. store Zij. it is needed in dVj += (Pij.T * Zij) @ dOi
        ZijSharedStorage zij;

        union {
          // 2. prologue for dVj
          // 6. workspace for dVj += (Pij.T * Zij) @ dOi
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          // 7. dVj epilogue
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };

        // 3. prologue for dPij_dropped
        // 8. used in dPij_dropped = dOi @ Vj.T
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      } part1;

      struct {
        // part2 - dQ
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::Mma::SharedStorage mm_gradQ; // (preload)
        union {
          // store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
        };

      } part2;

      struct {
        // part3 - after last iteration on dQ's epilogue / dK
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::DefaultEpilogue::SharedStorage
            gradQ_epilogue_lastIter;

        typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
      } part3;

      struct {
        // part4 - after last iteration on dK's epilogue / preload next K.Q_t
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;

        // If we reach end of current key, dump RF->gmem with "final" epilogues
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part4;
    };
    static void print_size() {
      // Field size
#define FSZ(f) int((sizeof(((SharedStoragePrologue*)0)->f)))

      printf("Total smem: %d bytes\n", int(sizeof(SharedStoragePrologue)));
      printf("  persistent: %db\n", FSZ(persistent));
      printf("    mm_qk_k: %db\n", FSZ(persistent.mm_qk_k));
      printf("  part1: %db\n", FSZ(part1));
      printf("    bias: %db\n", FSZ(part1.bias));
      printf("    attn_shared_storage: %db\n", FSZ(part1.attn_shared_storage));
      printf("    zij: %db\n", FSZ(part1.zij));
      printf("    mm_gradV: %db\n", FSZ(part1.mm_gradV));
      printf("    gradV_epilogue: %db\n", FSZ(part1.gradV_epilogue));
      printf("    mm_doivj: %db\n", FSZ(part1.mm_doivj));
      printf("  part2: %db\n", FSZ(part2));
      printf("    tmpT_shared_storage: %db\n", FSZ(part2.tmpT_shared_storage));
      printf("    tmp_shared_storage: %db\n", FSZ(part2.tmp_shared_storage));
      printf("    mm_gradK: %db\n", FSZ(part2.mm_gradK));
      printf("    mm_gradQ: %db\n", FSZ(part2.mm_gradQ));
      printf("    gradB_epilogue: %db\n", FSZ(part2.gradB_epilogue));
      printf("    gradQ_epilogue: %db\n", FSZ(part2.gradQ_epilogue));
      printf("  part3: %db\n", FSZ(part3));
      printf("    tmpT_shared_storage: %db\n", FSZ(part3.tmpT_shared_storage));
      printf("  part4: %db\n", FSZ(part4));
      printf("    mm_qk_q: %db\n", FSZ(part4.mm_qk_q));
      printf(
          "    gradK_epilogue_final: %db\n", FSZ(part4.gradK_epilogue_final));
      printf(
          "    gradV_epilogue_final: %db\n", FSZ(part4.gradV_epilogue_final));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(persistent, mm_qk_k)
    FIELD(part1, bias)
    FIELD(part1, attn_shared_storage)
    FIELD(part1, zij)
    FIELD(part1, mm_gradV)
    FIELD(part1, gradV_epilogue)
    FIELD(part1, mm_doivj)
    FIELD(part2, mm_gradK)
    FIELD(part2, mm_gradQ)
    FIELD(part2, gradB_epilogue)
    FIELD(part2, gradQ_epilogue)
    FIELD(part2, tmp_shared_storage)
    FIELD(part3, tmpT_shared_storage)
    FIELD(part3, gradQ_epilogue_lastIter)
    FIELD(part3, gradK_epilogue)
    FIELD(part4, mm_qk_q)
    FIELD(part4, gradK_epilogue_final)
    FIELD(part4, gradV_epilogue_final)
  };

    // [PREFETCH A1] empty placeholder so mm_gradK_deferred collapses to ~0 B when the deferral/fusion is
    // gated off (kFuseQKGradK==false): processGradK (its only reader, :1534) is called only under kFuseQKGradK,
    // so the buffer is dead on the fusion-off path -- reclaiming 16384 B toward the q-home-slot prefetch.
    struct EmptySmem {};

    struct SharedStorageNoPrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
      // [CP1 deferred-Kbar] T and T^T persisted OUT of the part-union so tile
      // i-1's T survives into tile i's start (part1's QK no longer clobbers it).
      // Cost: +sizeof(tmpT)+sizeof(tmp) SMEM, funded by the 47KB headroom.
      typename MatmulQK::AccumulatorSharedStorage tmpT_shared_storage;
#ifdef FMHA_BWD_R7
      // [R7] GradQ is reassociated to dQ^T = k^T @ S^T via R7_MmaWholeB (pure-compute, both operands resident).
      // r7_kT lives in the part4 union (filled in-phase by prologue_whole). r7_sT (S^T = B multiplicand) is filled
      // ON-CHIP by the DOIVJ epilogue (part3) via .at() and must SURVIVE into the GradQ mma (part4) -> it is the
      // one cross-phase survivor, so it sits in persistent (+16KB -> ~90.6KB total, fits V100's 98304). This drops
      // the global S^T round-trip (item 2). dS (tmp) is no longer read by GradQ -> dropped.
      // [R7 d-gate] r7_sT (and r7_kT below) only exist for the single-tile dQ path (kOutputInRF == d<=128,
      // where the whole-B dQ^T=k^T@S^T runs). For d>128 (kMaxK>kBlockSizeI) the lock-RMW column-loop path runs
      // instead, which needs the normal DOIVJ accumulator in tmp_shared_storage -> keep it there, r7_sT Empty.
      typename cutlass::platform::conditional<kOutputInRF,
          EmptySmem, typename MatmulDOIVJ::AccumulatorSharedStorage>::type tmp_shared_storage;
      typename cutlass::platform::conditional<kOutputInRF,
          typename MatmulGradQ::R7_MmaWholeB::Base::SharedStorageB, EmptySmem>::type r7_sT;
#else
      typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
#endif
      // [CP1 1c] disjoint Q-operand buffer for the deferred GradK so its contraction can OVERLAP the next
      // tile's QK (whose operands live in part1) without an aliasing collision -- the anchored Phase-1 merge.
      // [PREFETCH A1] present only when we actually fuse/defer; ~0 B otherwise (funds the q-home-slot).
      typename cutlass::platform::conditional<
          kFuseQKGradK,
          typename MatmulGradK::Mma::SharedStorage,
          EmptySmem>::type mm_gradK_deferred;
    } persistent;
    union {
      struct {
        // part1 - Q.K matmul ([PREFETCH] MmaQK == Mma unless FMHA_BWD_QK_WHOLEB, then whole-tile-resident q)
        typename MatmulQK::MmaQK::SharedStorageA mm_qk_k;
        typename MatmulQK::MmaQK::SharedStorageB mm_qk_q;
      } part1;

      struct {
        // part2 - compute gradV
        union {
          // 1. efficient load of bias tile Bij, which is then applied to Pij
          typename MatmulQK::BiasLoader::SmemTile bias;
          // 2. store Pij to shared memory. it is needed:
          // - in this step, where it is used in dVj += (Pij.T * Zij) @ dOi
          // - in next step where it is used in dSij = Pij * (dPij - Di)
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };
        // 3. store Zij. it is needed in this step, where it is used
        // to compute Pij_dropped = Pij * Zij on the fly as fragments of Pij are
        // loaded for the computation of dVj.
        ZijSharedStorage zij;

        union {
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };
      } part2;

      struct {
        // part3 - DO.V matmul
        union {
          // first compute dPij = (dOi @ Vj.T) * Zij
          // and dSij = Pij * (dPij - Di)
          struct {
            // (from part2) - Pij for computing dSij = Pij * (dPij - Di)
            typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
            // matmul to compute dOiVj
            typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
          };
          // then store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
        };
      } part3;

      struct {
        // part4 - compute gradQ  ([CP1] tmpT/tmp moved to persistent)
        union {
#ifdef FMHA_BWD_R7
          // [R7] whole-B GradQ A operand k^T -- filled in-phase by prologue_whole, consumed by the mma, so it
          // lives in this part4 union arm (overlaps parts 1/2/3/5/6). (r7_sT is persistent -- it survives part3.)
          // [R7 d-gate] only for kOutputInRF (d<=128); Empty for d>128 (lock-RMW path uses mm_gradQ).
          typename cutlass::platform::conditional<kOutputInRF,
              typename MatmulGradQ::R7_MmaWholeB::Base::SharedStorageA, EmptySmem>::type r7_kT;
#endif
          typename MatmulGradQ::Mma::SharedStorage mm_gradQ;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage
              gradQ_epilogue_lastIter;
        };
      } part4;

      struct {
        // part5 - compute gradK  ([CP1] tmpT/tmp moved to persistent)
        union {
          typename MatmulGradK::Mma::SharedStorage mm_gradK;
          typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
        };
      } part5;

      struct {
        // part6 - store RF accumulated into gmem
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part6;
    };
    static void print_size() {
#define FIELD_SIZEOF(f) int((sizeof(((SharedStorageNoPrologue*)0)->f)))
      printf("Total smem: %d bytes\n", int(sizeof(SharedStorageNoPrologue)));
      printf("  persistent: %db\n", FIELD_SIZEOF(persistent));
      printf("  part1: %db\n", FIELD_SIZEOF(part1));
      printf("  part2: %db\n", FIELD_SIZEOF(part2));
      printf("  part3: %db\n", FIELD_SIZEOF(part3));
      printf("  part4: %db\n", FIELD_SIZEOF(part4));
      printf("  part5: %db\n", FIELD_SIZEOF(part5));
      printf("  part6: %db\n", FIELD_SIZEOF(part6));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(persistent, tmpT_shared_storage)
    FIELD(persistent, tmp_shared_storage)
#ifdef FMHA_BWD_R7
    FIELD(part4, r7_kT)
    FIELD(persistent, r7_sT)
#endif
    FIELD(persistent, mm_gradK_deferred)
    FIELD(part1, mm_qk_k)
    FIELD(part1, mm_qk_q)
    FIELD(part2, bias)
    FIELD(part2, attn_shared_storage)
    FIELD(part2, zij)
    FIELD(part2, mm_gradV)
    FIELD(part2, gradV_epilogue)
    FIELD(part3, mm_doivj)
    FIELD(part3, gradB_epilogue)
    FIELD(part4, mm_gradQ)
    FIELD(part4, gradQ_epilogue)
    FIELD(part4, gradQ_epilogue_lastIter)
    FIELD(part5, mm_gradK)
    FIELD(part5, gradK_epilogue)
    FIELD(part6, gradK_epilogue_final)
    FIELD(part6, gradV_epilogue_final)
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kPreload,
      SharedStoragePrologue,
      SharedStorageNoPrologue>::type;

  // [RF-multi] The dK/dV output tile is kBJ x ThreadblockShape::kN, and kN == kBlockSizeI -- so at
  // head_dim > kBlockSizeI the epilogue walks the head in COLUMN TILES (`for col ... col += kN`, with
  // `storage_id = col / kN` naming each one). With the accumulator in gmem that is fine: every column
  // tile loads and stores its own workspace slab, so they stay separate. Held in REGISTERS there was
  // only ever ONE fragment, so the second column tile accumulated on top of the first and half the head
  // came out garbage -- that is exactly what the RF256 probe measured (1.372-1.501x and wrong grads).
  // Give the residency one fragment per column tile, indexed by that same storage_id. The non-resident
  // path keeps a single fragment, so it stays byte-identical.
  //
  // [RESIDENCY COST -- derived, and it prunes a whole family of ideas]
  //   kNumThreads   = 32 * (kBI*kBJ)/1024 = kBI*kBJ/32
  //   column tiles  = head_dim / kN,  and kN == kBI
  //   per tile/thread = kBJ*kN / kNumThreads = kBJ*kBI / (kBI*kBJ/32) = 32   (always, exactly)
  //   TOTAL per thread, per tensor = 32 * head_dim / kBI
  // So the resident dK (or dV) accumulator costs 32*d/kBI floats per thread and depends on NOTHING
  // else: not kBJ, not the column-tile width, not the warp count. Halving kBJ halves the tile AND the
  // warp count together, so floats-per-thread do not move. That refutes the premise written next to
  // the kBJ=32 probe in attn_bwd_cutlass.cu ("kBJ=32 gives 32*128/256 = 16 floats/thread") -- that
  // arithmetic used 256 threads for a tile that only ever has 128.
  // Consequences: d=128,kBI=128 -> 32+32 = 64 floats (this is the shipped fast path, and it fits);
  // d=256,kBI=128 -> 64+64 = 128 (the RF256 spill); d=512,kBI=128 -> 256 (hopeless). The ONLY lever on
  // residency is kBI, which is why <256,64,256> was the shape worth wanting -- but its smem is what
  // kills it, not the instantiation (MatmulQK's accumulator alone is kBJ*kBI*2 = 32 KB at kBI=256).
  static constexpr int kNumColTilesV =
      kOutputInRF ? (kMaxK + MatmulGradV::ThreadblockShape::kN - 1) /
                        MatmulGradV::ThreadblockShape::kN
                  : 1;
  static constexpr int kNumColTilesK =
      kOutputInRF ? (kMaxK + MatmulGradK::ThreadblockShape::kN - 1) /
                        MatmulGradK::ThreadblockShape::kN
                  : 1;

  struct OutputFragments {
    typename MatmulGradV::Mma::FragmentC gradV[kNumColTilesV];
    typename MatmulGradK::Mma::FragmentC gradK[kNumColTilesK];

    CUTLASS_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNumColTilesV; ++i) gradV[i].clear();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNumColTilesK; ++i) gradK[i].clear();
    }
  };

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.key_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.value_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.grad_output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.bias_ptr, kMinimumAlignment);
    XFORMERS_CHECK(p.lse_strideH % 8 == 0, "LSE is not correctly aligned");
    XFORMERS_CHECK(p.lse_strideB % 8 == 0, "LSE is not correctly aligned");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.q_strideH % kMinimumAlignment == 0,
        "query is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.k_strideH % kMinimumAlignment == 0,
        "key is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.v_strideH % kMinimumAlignment == 0,
        "value is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_batches <= 1 || p.q_strideB % kMinimumAlignment == 0,
        "query is not correctly aligned (strideB)");
    XFORMERS_CHECK(
        p.num_batches <= 1 || p.k_strideB % kMinimumAlignment == 0,
        "key is not correctly aligned (strideB)");
    XFORMERS_CHECK(
        p.num_batches <= 1 || p.v_strideB % kMinimumAlignment == 0,
        "value is not correctly aligned (strideB)");
    XFORMERS_CHECK(
        p.q_strideM % kMinimumAlignment == 0,
        "query is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.k_strideM % kMinimumAlignment == 0,
        "key is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.v_strideM % kMinimumAlignment == 0,
        "value is not correctly aligned (strideM)");
    if (p.bias_ptr) {
      XFORMERS_CHECK(
          p.num_batches <= 1 || p.bias_strideB % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideB)");
      XFORMERS_CHECK(
          p.num_heads <= 1 || p.bias_strideH % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideH)");
      XFORMERS_CHECK(
          p.bias_strideM % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideM)");
    }
    if (p.grad_bias_ptr) {
      XFORMERS_CHECK(
          p.num_batches <= 1 || p.gB_strideB % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideB)");
      XFORMERS_CHECK(
          p.num_heads <= 1 || p.gB_strideH % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideH)");
      XFORMERS_CHECK(
          p.gB_strideM % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideM)");
    }
    XFORMERS_CHECK(
        !(p.cu_seqlens_q_ptr && p.bias_ptr),
        "CuSeqlen + bias not implemented yet");
    XFORMERS_CHECK(
        p.custom_mask_type < NumCustomMaskTypes,
        "Invalid value for `custom_mask_type`");
    XFORMERS_CHECK(
        p.dropout_prob <= 1.0f && p.dropout_prob >= 0.0f,
        "Invalid value for `dropout_prob`");
    XFORMERS_CHECK(
        kApplyDropout || p.dropout_prob == 0.0f,
        "Set `kApplyDropout`=True to support `dropout_prob > 0`");
    XFORMERS_CHECK(p.head_dim > 0, "Invalid value for `head_dim`");
    XFORMERS_CHECK(p.head_dim_value > 0, "Invalid value for `head_dim_value`");
    XFORMERS_CHECK(p.num_queries > 0, "Invalid value for `num_queries`");
    XFORMERS_CHECK(p.num_keys > 0, "Invalid value for `num_keys`");
    XFORMERS_CHECK(p.num_heads > 0, "Invalid value for `num_heads`");
    XFORMERS_CHECK(p.num_batches > 0, "Invalid value for `num_batches`");
    XFORMERS_CHECK(p.head_dim <= kMaxK, "kMaxK: Expected `head_dim < kMaxK`");
    XFORMERS_CHECK(
        p.head_dim_value <= kMaxK, "kMaxK: Expected `head_dim_value < kMaxK`");
    if (kKeysQueriesAlignedToBlockSize) {
      XFORMERS_CHECK(
          p.cu_seqlens_k_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      XFORMERS_CHECK(
          p.cu_seqlens_q_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      XFORMERS_CHECK(
          p.num_queries % kBlockSizeI == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
      XFORMERS_CHECK(
          p.num_keys % kBlockSizeJ == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
    }
    XFORMERS_CHECK(
        kEnableSplitKeys || p.num_splits_key == 1, "SplitKeys is disabled");
    XFORMERS_CHECK(
        p.num_splits_key > 0, "Invalid `num_splits_key` (expected >0)");
    XFORMERS_CHECK(
        p.num_splits_key <= cutlass::ceil_div(p.num_keys, kBlockSizeJ),
        "Invalid `num_splits_key` (too large)");
    return true;
  }

  static CUTLASS_DEVICE void attention_kernel(Params p) {
    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);

    uint16_t thread_id = threadIdx.x;
    uint8_t warp_id = warp_uniform(thread_id / 32);
    uint8_t lane_id = thread_id % 32;

    int32_t key_start = p.split_key_device() * kBlockSizeJ;
    if (key_start >= p.num_keys) {
      return;
    }
    if (kPrologueQK) {
      int32_t query_start = getQueryStart(p, key_start);
      prologueQkNextIteration<true>(
          shared_storage, p, query_start, key_start, warp_id, lane_id);
    }

    // Computes (dO*out).sum(-1) and writes it to `p.delta_ptr`
    if (kKernelComputesDelta) {
      constexpr int kOptimalElements =
          128 / cutlass::sizeof_bits<scalar_t>::value;
      if (p.head_dim_value % kOptimalElements == 0) {
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<kOptimalElements>(p, query_start, warp_id, lane_id);
        }
      } else {
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<1>(p, query_start, warp_id, lane_id);
        }
      }
      __syncthreads();
    }

    OutputFragments output_frags;

    curandStatePhilox4_32_10_t rng_state_init;
#ifdef HAS_PYTORCH
    if (kApplyDropout) {
      auto seeds = at::cuda::philox::unpack(p.rng_engine_inputs);
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
          &rng_state_init);
    }
#endif
    CUTLASS_PRAGMA_UNROLL
    for (; key_start < p.num_keys;
         key_start += p.num_splits_key_device() * kBlockSizeJ) {
      output_frags.clear();
      // [CP1 deferred-Kbar] the tile whose T is currently in the persistent tmp/tmpT buffer with its GradK
      // pending; issued at the next tile's top or by the post-sweep flush below.
      int32_t prev_gradk_qs = -1;
      // [Q1 query-split] this CTA owns query-piece q_piece of num_splits_query, an equal (b-aligned) cut of the
      // current key-band's live query range [getSmallestQueryForKey(key_start), num_queries). q=1 -> full range.
      int32_t qsplit_lo = 0, qsplit_hi = p.num_queries;
      if (p.num_splits_query > 1) {
        int32_t q0 = getSmallestQueryForKey(p, key_start);
        int32_t m_s = (p.num_queries - q0 + kBlockSizeI - 1) / kBlockSizeI;  // live query-blocks
        int16_t pc = p.q_piece_device();
        qsplit_lo = q0 + (int32_t)((int64_t)pc * m_s / p.num_splits_query) * kBlockSizeI;
        qsplit_hi = q0 + (int32_t)((int64_t)(pc + 1) * m_s / p.num_splits_query) * kBlockSizeI;
      }

      // [Q1] q>1 iterates the piece CONTIGUOUSLY over [qsplit_lo, qsplit_hi) with NO shift/wrap -> the
      // kFuseQKGradK defer/flush pipeline stays intact within the contiguous piece (no tile skipping).
      // [STAGGER — MEASURED NET-NEGATIVE, REMOVED 2026-07-20] measured's service-rate lever (rotate each
      // key-split's query sweep by a per-split phase to de-collide the dQ atomic-fold writers) was built + verified
      // (11/11 exact) but A/B REFUTED it: ncu showed lg_throttle UNCHANGED (2.14->2.16) => the far-store-port
      // stall is NOT collision-serialization but pure arrival-burst (measured's own falsifier branch); the rotation
      // desynced the prologue prefetch -> REGRESSED filled shapes (S2048 1.028->0.983) while the tiny-shape "gain"
      // was noise (baseline B1H8-S512 itself swings 0.98-1.06). Not the lever. See fa2-sm70-bwd-round20-queue.
      CUTLASS_PRAGMA_UNROLL
      for (int32_t query_start_shifted =
               (p.num_splits_query > 1 ? qsplit_lo : getQueryStart(p, key_start));
           query_start_shifted <
               (p.num_splits_query > 1 ? qsplit_hi
                                       : getQueryStartShift(p) + getQueryEnd(p));
           query_start_shifted += kBlockSizeI) {
        // This line here
        // vvvvvvvvvvvvvv
        warp_id = warp_uniform(warp_id);
        // ^^^^^^^^^^^^^^
        // ... makes everything use less RF and be 10% faster. Why?
        // I don't know. My theory is that it forces `nvcc` to
        // re-compute indices, offsets etc... and not keep them
        // from the previous iteration, which prevents MASSIVE
        // register spilling.

        int32_t query_start = query_start_shifted;
        if (p.num_splits_query == 1 && query_start >= p.num_queries) {
          query_start = query_start % getQueryEnd(p);  // shift/wrap only on the non-query-split path
        }

        processBlockIJ<kKeysQueriesAlignedToBlockSize>(
            shared_storage,
            output_frags,
            p,
            query_start,
            key_start,
            prev_gradk_qs,
            rng_state_init,
            warp_id,
            lane_id);
        if constexpr (kFuseQKGradK && kKeysQueriesAlignedToBlockSize) {
          prev_gradk_qs = query_start;
        }
      }
      // [CP1 deferred-Kbar] flush the last tile's pending GradK (fusing path only).
      if constexpr (kFuseQKGradK && kKeysQueriesAlignedToBlockSize) {
        if (prev_gradk_qs >= 0) {
          warp_id = warp_uniform(warp_id);
          processGradK<kKeysQueriesAlignedToBlockSize>(
              shared_storage, output_frags, p, prev_gradk_qs, key_start,
              warp_id, lane_id);
        }
      }
      if (kOutputInRF) {
        writeFragsToGmem<kKeysQueriesAlignedToBlockSize>(
            shared_storage, output_frags, p, key_start, warp_id, lane_id);
      } else if (getQueryStart(p, key_start) >= p.num_queries) {
        zfillGradKV<kKeysQueriesAlignedToBlockSize>(
            p, key_start, warp_id, lane_id);
      }
      __syncthreads();
    }
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void zfillGradKV(
      Params const& p,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    constexpr int kThreadsPerKey = 8;
    constexpr int kParallelKeys = kNumThreads / kThreadsPerKey;
    static_assert(kBlockSizeJ % kParallelKeys == 0, "");
    // This function is not really optimized, but should rarely be used
    // It's only used when some keys are "useless" and don't attend to
    // any query, due to causal masking

    int thread_id = 32 * warp_id + lane_id;
    int k_shift = lane_id % kThreadsPerKey;

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kBlockSizeJ; j += kParallelKeys) {
      int key = key_start + j + (thread_id / kThreadsPerKey);
      if (!skipBoundsChecks && key >= p.num_keys) {
        continue;
      }
      auto gv_ptr = p.grad_value_ptr + key * p.gV_strideM();
      auto gk_ptr = p.grad_key_ptr + key * p.gK_strideM();

      for (int k = k_shift; k < p.head_dim_value; k += kThreadsPerKey) {
        gv_ptr[k] = scalar_t(0);
      }
      for (int k = k_shift; k < p.head_dim; k += kThreadsPerKey) {
        gk_ptr[k] = scalar_t(0);
      }
    }
  }

  // [CP1 deferred-Kbar] Standalone GradK contraction (dK += T^T @ q) for the ANCHORED schedule: it is
  // deferred one query-tile and issued at the top of the next tile, reading the T left
  // in the persistent tmp/tmpT buffer by the previous tile's DOIVJ. Only used on the kOutputInRF && !kPreload
  // path (d128 shipped multibatch); computes its own num_*_in_block for the deferred query_start. The
  // per-tile GradK sum is order-independent, so deferring is EXACT. kPrologue*/!kOutputInRF branches are
  // dead here (kPreload=false) and omitted.
  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void processGradK(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params& p,
      int32_t query_start,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    int16_t thread_id = 32 * warp_id + lane_id;
    cutlass::MatrixCoord no_offset{0, 0};
    int32_t num_queries_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kN
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kN, p.num_queries - query_start));
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start));
    constexpr bool kSingleIterationGradK =
        kMaxK <= MatmulGradK::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradK ? 1 : p.head_dim);
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;
      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block, p.head_dim - col, num_queries_in_block);
      typename Mma::IteratorB iterator_B(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);
      auto getTmp = [&](int) { return &shared_storage.tmp_shared_storage(); };
      auto getTmpT = [&](int) { return &shared_storage.tmpT_shared_storage(); };
      bool constexpr kIsTransposedA =
          MatmulGradK::DefaultMmaFromSmem::kIsTransposedA;
      auto& opA = *call_conditional<
          kIsTransposedA,
          decltype(getTmp),
          decltype(getTmpT)>::apply(getTmp, getTmpT, 0);
      Mma mma(
          opA.accum_ref(),
          shared_storage.mm_gradK_deferred().operand_B_ref(),  // [CP1 1c] disjoint from part1 (QK operands)
          thread_id,
          warp_id,
          lane_id);
      mma.set_prologue_done(false);
      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      __syncthreads();
      // fused QK||GradK requires kMaxK == kBlockSizeI, i.e. exactly one column tile
      mma(gemm_k_iterations, output_frags.gradK[0], iterator_B, output_frags.gradK[0]);
      // [CP1 1c] trailing __syncthreads REMOVED so this GradK's drain overlaps the next tile's QK fill
      // (anchored phase-1 merge). Safe: GradK's operands (tmp/tmpT + mm_gradK_deferred, all
      // persistent) are disjoint from QK's (part1 mm_qk + part2 attn), and tmp/tmpT are not rewritten
      // until DOIVJ, many __syncthreads later.
    }
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void processBlockIJ(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params& p,
      int32_t query_start,
      int32_t key_start,
      int32_t gradk_query_start,
      const curandStatePhilox4_32_10_t& curand_state_init,
      uint8_t warp_id,
      uint8_t lane_id) {
#ifdef FMHA_BWD_R7
    { (void)sizeof(typename MatmulGradQ::R7_MmaWholeB::FragmentC);
      (void)sizeof(typename MatmulGradQ::R7_MmaWholeB::Base::SharedStorageB); }  // validate whole-B config
#endif
    // [CP1 deferred-Kbar] The anchored deferral is ACTIVE only when we actually fuse (kFuseQKGradK && aligned
    // tiles): GradK for tile i-1 is subsumed by the fused QK||GradK call below when head_dim==kBlockSizeI, and
    // the last tile is flushed after the sweep. FALLBACK: if head_dim != kBlockSizeI (a d128-config model with
    // head_dim<128), the fused loop's equal-k assumption fails, so run the standalone deferred GradK here
    // (correct for any head_dim). On the non-fused path (non-aligned/fusion off) there is NO deferral.
    if constexpr (kFuseQKGradK) {
      if (gradk_query_start >= 0 && p.head_dim != kBlockSizeI) {
        processGradK<skipBoundsChecks>(
            shared_storage, output_frags, p, gradk_query_start, key_start,
            warp_id, lane_id);
      }
    }
    cutlass::Array<cutlass::uint1b_t, MatmulDOIVJ::Mma::FragmentC::kElements>
        dropout_keep_mask_doivj;
    dropout_keep_mask_doivj.fill(cutlass::uint1b_t{1});
    const float dropout_scale =
        kApplyDropout ? 1.0 / (1.0 - p.dropout_prob) : 1.0f;

    cutlass::MatrixCoord no_offset{0, 0};
    accum_t scale = p.scale;
    int16_t thread_id = 32 * warp_id + lane_id;

    auto rematerializeThreadIds = [&]() {
      // Prevents `nvcc` from keeping values deduced from
      // `thread_id`, `warp_id`, ... in RF - to reduce register pressure
      warp_id = warp_uniform(thread_id / 32);
      lane_id = thread_id % 32;
      thread_id = 32 * warp_id + lane_id;
    };

    bool isFirstQuery = (query_start == getQueryStart(p, key_start));
    int32_t next_query, next_key;
    incrIteration(p, query_start, key_start, next_query, next_key);
    bool isLastQuery = next_key != key_start;

    accum_t di_rf = accum_t(0);
    if (thread_id < kBlockSizeI) {
      if (query_start + thread_id < p.num_queries) {
        di_rf = p.delta_ptr[query_start + thread_id];
      }
      shared_storage.di()[thread_id] = di_rf;
    }

    int32_t num_queries_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kN
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kN, p.num_queries - query_start));
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start));

    auto prologueGradV = [&](int col) {
      typename MatmulGradV::Mma::IteratorB iterator_dO(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM + col,
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);
      MatmulGradV::Mma::prologue(
          shared_storage.mm_gradV(),
          iterator_dO,
          thread_id,
          num_queries_in_block);
    };
    auto prologueGradQ = [&](int col) {
      typename MatmulGradQ::Mma::IteratorB iterator_K(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM + col,
          {num_keys_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      MatmulGradQ::Mma::prologue(
          shared_storage.mm_gradQ(), iterator_K, thread_id, num_keys_in_block);
    };
    auto prologueGradK = [&](int col) {
      typename MatmulGradK::Mma::IteratorB iterator_Q(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM + col,
          {num_queries_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      MatmulGradK::Mma::prologue(
          shared_storage.mm_gradK(),
          iterator_Q,
          thread_id,
          num_queries_in_block);
    };
    auto prologueDOV = [&]() {
      typename MatmulDOIVJ::Mma::IteratorA iterator_A(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM,
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);
      typename MatmulDOIVJ::Mma::IteratorB iterator_B(
          {int32_t(p.v_strideM)},
          p.value_ptr + key_start * p.v_strideM,
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);
      MatmulDOIVJ::Mma::prologue(
          shared_storage.mm_doivj(),
          iterator_A,
          iterator_B,
          thread_id,
          p.head_dim_value);
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulQK
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulQK::MmaQK;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block,
          num_queries_in_block,
          p.head_dim // k
      );

      // k_j
      typename Mma::IteratorA iterator_A(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM,
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.mm_qk_k(),
          shared_storage.mm_qk_q(),
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
#if defined(FMHA_BWD_QK_WHOLEB)
      // [PREFETCH] whole-B-resident QK: pre-load ALL kblocks of k,q into the (whole-tile) smem buffers, then
      // the kNStages_>2 mainloop runs pure compute (no global q loads). q now lives resident for the whole tile.
      MatmulQK::MmaQK::prologue_whole(
          shared_storage.mm_qk_k(), shared_storage.mm_qk_q(),
          iterator_A, iterator_B, thread_id, problem_size.k());
      mma.set_prologue_done(true);
#elif defined(FMHA_BWD_QK_PROLOGUE)
      // [PREFETCH] exercise the operand-prologue: pre-load kblock 0 of k,q, then skip it inside the mma.
      if (skipBoundsChecks) {
        MatmulQK::Mma::prologue(
            shared_storage.mm_qk_k(), shared_storage.mm_qk_q(),
            iterator_A, iterator_B, thread_id, problem_size.k());
        mma.set_prologue_done(true);
      } else {
        mma.set_prologue_done(kPrologueQK);
      }
#else
      mma.set_prologue_done(kPrologueQK);
#endif
      mma.set_zero_outside_bounds(!skipBoundsChecks);
      // [CP1] fused QK||GradK interleave for full tiles with a pending deferred GradK; else the plain QK mma
      // (the deferred GradK, if any, already ran at the top). GradK is single-column here (kMaxK<=kBI),
      // k=num_queries=head_dim=128 == QK's k, so the two mainloops have equal gemm_k_iterations.
      if constexpr (kFuseQKGradK) {
        if (gradk_query_start >= 0 && skipBoundsChecks && p.head_dim == kBlockSizeI) {
          using GkMma = typename MatmulGradK::Mma;
          auto getTmp = [&](int) { return &shared_storage.tmp_shared_storage(); };
          auto getTmpT = [&](int) { return &shared_storage.tmpT_shared_storage(); };
          constexpr bool kIsTransposedA =
              MatmulGradK::DefaultMmaFromSmem::kIsTransposedA;
          auto& gk_opA = *call_conditional<
              kIsTransposedA, decltype(getTmp), decltype(getTmpT)>::
              apply(getTmp, getTmpT, 0);
          GkMma gk_mma(
              gk_opA.accum_ref(),
              shared_storage.mm_gradK_deferred().operand_B_ref(),
              thread_id, warp_id, lane_id);
          typename GkMma::IteratorB gk_iter_B(
              {int32_t(p.q_strideM)},
              p.query_ptr + gradk_query_start * p.q_strideM,
              {num_queries_in_block, p.head_dim},
              thread_id, no_offset);
          fmha_fused::fused_qk_gradk(
              mma, iterator_A, iterator_B, accum,
              gk_mma, gk_iter_B, output_frags.gradK[0], gemm_k_iterations);
        } else {
          mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
        }
      } else {
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      }
      accum = cutlass::multiplies<typename Mma::FragmentC>()(scale, accum);

      // Epilogue: add LSE + exp and store that to our shared memory buffer
      // shmem <- (matmul_result -
      // logsumexp[i_start:i_end].unsqueeze(1)).exp()
      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};

      // apply bias if applicable
      if (p.bias_ptr != nullptr) {
        // load bias tile Bij into shared memory
        typename MatmulQK::BiasLoader::GmemTileIterator bias_iter(
            {cutlass::layout::RowMajor(p.bias_strideM)},
            p.bias_ptr + query_start * p.bias_strideM + key_start,
            {num_queries_in_block, num_keys_in_block},
            thread_id);
        cutlass::TensorRef<scalar_t, cutlass::layout::RowMajor> bias_tensor_ref(
            shared_storage.bias().data(),
            cutlass::layout::RowMajor(MatmulQK::ThreadblockShape::kM));
        typename MatmulQK::BiasLoader::SmemTileIterator smem_tile_iter(
            bias_tensor_ref, thread_id);
        MatmulQK::BiasLoader::load(bias_iter, smem_tile_iter);

        // Pij += Bij, where Pij is in register fragment and Bij is in shmem
        auto lane_offset = MatmulQK::AccumLambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords);
        MatmulQK::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_n) {},
            [&](int accum_m, int accum_n, int idx) {
              // remember we are transposed
              accum[idx] += bias_tensor_ref.at({accum_n, accum_m});
            },
            [&](int accum_n) {});
      }

      // Apply mask
      if (p.custom_mask_type == CausalFromTopLeft ||
          p.custom_mask_type == CausalFromBottomRight) {
        auto lane_offset = MatmulQK::AccumLambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords);
        int shift = query_start - key_start;
        if (p.custom_mask_type == CausalFromBottomRight) {
          shift += p.num_keys - p.num_queries;
        }
        // current_key = key_start + accum_m
        // current_query = query_start + accum_n
        // mask if: `current_key > current_query`
        MatmulQK::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              if (accum_m > accum_n + shift) {
                accum[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_m) {});
      }

      __syncthreads();
      if (kPrologueGV) {
        prologueGradV(0);
      }
      if (kPrologueDOV) {
        prologueDOV();
      }

      MatmulQK::B2bGemm::accumApplyLSEToSmem(
          shared_storage.attn_shared_storage(),
          accum,
          p.logsumexp_ptr + query_start,
          problem_size.n(),
          thread_id,
          warp_id,
          lane_id,
          output_tile_coords);
#if 0
      auto accum_ref_attnT = shared_storage.attn_shared_storage().accum_ref();
      PRINT_TENSOR4x4_T0_L0("attn_T", accum_ref_attnT);
#endif

      // if we are using dropout, compute Zij, writing it to shared memory.
      // each element of Zij is:
      // - 0 with probability dropout_p
      // - 1 / (1 - dropout_p) with probability 1 - dropout_p
      if (kApplyDropout) {
        auto zij = shared_storage.zij().accum_ref();
        // each thread generates a contiguous sequence of elements in Zij, all
        // in the same row. the reason they have to come from the same row is
        // that sampling random numbers from a contiguous random number sequence
        // is much more efficient than jumping around, and the linear offset of
        // each element of Z (the global matrix) maps to an offset in a random
        // number sequence. for Z, the end of a row and the beginning of the
        // next have adjacent offsets, but for Zij (tile of global matrix), this
        // is not necessarily the case.
        // We must fill the entire `zij` shmem with values (even out of bounds
        // on the K-dimension) otherwise we can get NaNs during the GEMM
        const int kQueriesPerBlock = kBlockSizeI;
        const int threads_per_row = cutlass::fast_min(
            int32_t(kNumThreads / kQueriesPerBlock), num_keys_in_block);
        const int elts_per_thread = cutlass::round_nearest(
            cutlass::ceil_div(num_keys_in_block, threads_per_row), 4);

        const int thread_i = thread_id / threads_per_row;
        const int thread_start_j =
            (thread_id % threads_per_row) * elts_per_thread;

        if (thread_i < kQueriesPerBlock && thread_start_j < num_keys_in_block) {
          curandStatePhilox4_32_10_t curand_state = curand_state_init;
          skipahead(
              (query_start + thread_i) * p.num_keys +
                  (key_start + thread_start_j),
              &curand_state);

          // generate elements of Zij, 4 elements at a time
          for (int zij_start_col_idx = thread_start_j; zij_start_col_idx <
               cutlass::fast_min<int32_t>(thread_start_j + elts_per_thread,
                                          num_keys_in_block);
               zij_start_col_idx += 4) {
            const float4 rand_uniform_quad = curand_uniform4(&curand_state);

            CUTLASS_PRAGMA_UNROLL
            for (int quad_idx = 0; quad_idx < 4; ++quad_idx) {
              // we'll write Zij transposed since attention is also transposed
              // during the matmul to compute dV.
              zij.at({zij_start_col_idx + quad_idx /*k*/, thread_i /*q*/}) =
                  (&rand_uniform_quad.x)[quad_idx] > p.dropout_prob
                  ? scalar_t(dropout_scale)
                  : scalar_t(0);
            }
          }
        }
        __syncthreads();
#if 0
        PRINT_TENSOR4x4_T0_L0("zij", zij);
        PRINT_TENSOR4x4_T0_L0_START("zij", zij, kBlockSizeJ - 4, kBlockSizeI - 4);
#endif

        // Save mask for later DOIVJ matmul

        int warp_idx_mn_0 = warp_id %
            (MatmulDOIVJ::Mma::Base::WarpCount::kM *
             MatmulDOIVJ::Mma::Base::WarpCount::kN);
        auto output_tile_coords_doivj = cutlass::MatrixCoord{
            warp_idx_mn_0 % MatmulDOIVJ::Mma::Base::WarpCount::kM,
            warp_idx_mn_0 / MatmulDOIVJ::Mma::Base::WarpCount::kM};
        auto lane_offset = MatmulDOIVJ::AccumLambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords_doivj);
        MatmulDOIVJ::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m /*q*/, int accum_n /*k*/, int idx) {
              if (zij.at({accum_n, accum_m}) == scalar_t(0)) {
                dropout_keep_mask_doivj[idx] = cutlass::uint1b_t{0};
              }
            },
            [&](int accum_m) {});
      }
      __syncthreads();
    }
    rematerializeThreadIds();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradV matmul
    //
    // grad_v[j_start:j_end] += attn_T @ do_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    constexpr bool kSingleIterationGradV =
        kMaxK <= MatmulGradV::ThreadblockShape::kN;
    // [TWO-PASS] Pass Q (kComputeQOnly) computes dQ only -> the whole GradV stage is compiled out (the Pass KV
    // launch owns dV). Gating the loop condition (rather than brace-wrapping) keeps the #ifdef-interleaved body
    // untouched; nvcc eliminates the loop entirely for the constexpr-false case.
    for (int col = 0; !kComputeQOnly && col < (kSingleIterationGradV ? 1 : p.head_dim_value);
         col += MatmulGradV::ThreadblockShape::kN) {
      using Mma = typename MatmulGradV::Mma;
      // [ACCUM16] dV/dK partials narrow; the fragment type stays MatmulGradQ's, as upstream reuses it
      // for all three tiles -- only the STORAGE width changes here.
      using AccumTileGmem = GmemTile<typename MatmulGradQ::Mma::FragmentC,
                                     (int)kNumThreads, fmha_accum_storage_t>;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block, p.head_dim_value - col, num_queries_in_block);
      auto createEpilogueIter = [&]() {
        return typename MatmulGradV::OutputTileIterator(
            typename MatmulGradV::OutputTileIterator::Params{p.gV_strideM()},
            p.grad_value_ptr + key_start * p.gV_strideM() + col,
            {num_keys_in_block, p.head_dim_value - col},
            thread_id);
      };
      typename Mma::IteratorB iterator_B(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM + col,
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);

      // if dropout: dVj += (Pij.T * Zij) @ dOi
      // otherwise:  dVj += Pij.T @ dOi
      Mma mma(
          // operand A: Pij.T
          shared_storage.attn_shared_storage().accum_ref(),
          // operand A_scale Zij.T:
          // if we're using dropout, operand A is Pij_dropped.T = Pij.T * Zij.T
          // which is computed on the fly as fragments of Pij.T are loaded in
          shared_storage.zij().accum_ref(),
          // operand B: dOi - which was loaded into shared memory previously
          // when we computed dVj
          shared_storage.mm_gradV().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      int storage_id = col / MatmulGradV::ThreadblockShape::kN;
      // [RF-multi] one fragment per column tile when resident; a single one otherwise (kNumColTilesV==1)
      const int fragV = kOutputInRF ? storage_id : 0;
      // Workspace stays allocated in fp32 units (the host sizing is unchanged); a narrow tile simply
      // uses the first half of its region, and offsets are counted in storage elements.
      AccumTileGmem gmem_tile{
          reinterpret_cast<fmha_accum_storage_t*>(p.workspace_gv) +
          storage_id * AccumTileGmem::kElementsStored};
      if (!kOutputInRF) {
        if (isFirstQuery || !kNeedsAccumGradV) {
          output_frags.gradV[fragV].clear();
        } else {
          gmem_tile.load(output_frags.gradV[fragV], thread_id);
        }
      }
      mma.set_prologue_done(kPrologueGV);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();

      mma(gemm_k_iterations,
          output_frags.gradV[fragV],
          iterator_B,
          output_frags.gradV[fragV]);
      __syncthreads();
      if (kPrologueGV && !kSingleIterationGradV &&
          col + MatmulGradV::ThreadblockShape::kN < p.head_dim_value) {
        prologueGradV(col + MatmulGradV::ThreadblockShape::kN);
      }

      if (!kOutputInRF) {
        if (kNeedsAccumGradV && !isLastQuery) {
          gmem_tile.store(output_frags.gradV[fragV], thread_id);
        } else {
          accumulateInGmem<MatmulGradV>(
              shared_storage.gradV_epilogue(),
              output_frags.gradV[fragV],
              createEpilogueIter(),
              isFirstQuery || kNeedsAccumGradV,
              warp_id,
              lane_id);
        }
      }
    }
    __syncthreads();

    // The dV-only sweep stops here -- dV = P^T @ dO needs nothing below
    // (no dO@V^T, no dS, no dQ/dK), so gating this out drops the register footprint to the
    // forward-shaped ~168 (3 blocks/SM) instead of the fused 255 (1 block/SM). Never loads V.
    if constexpr (!kComputeVOnly) {
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulDOIVJ
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulDOIVJ::Mma;
      // do_i
      typename Mma::IteratorA iterator_A(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM,
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);

      // v_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.v_strideM)},
          p.value_ptr + key_start * p.v_strideM,
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);

      Mma mma(shared_storage.mm_doivj(), thread_id, warp_id, lane_id);
      mma.set_prologue_done(kPrologueDOV);
      mma.set_zero_outside_bounds(!skipBoundsChecks);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (p.head_dim_value + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();
      if (kPrologueGQ) {
        prologueGradQ(0);
      }
      if (kPrologueGK) {
        prologueGradK(0);
      }

      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};
      // TODO: This must be terribly inefficient. There must be a better way
      // tmp [RF] <- (accum [RF] - Di [smem] ) * attn_T.T [smem]
      // attn_shared_storage  [smem] <- tmp.T
      // tmp_shared_storage [smem] <- tmp
      {
        using LambdaIterator = typename MatmulDOIVJ::AccumLambdaIterator;
        auto lane_offset = LambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords);
        // if dropout was used, compute dPij = dPij_dropped * Zij
        if (kApplyDropout) {
          LambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {},
              [&](int accum_m, int accum_n, int idx) {
                if (dropout_keep_mask_doivj[idx].get()) {
                  accum[idx] *= dropout_scale;
                } else {
                  accum[idx] = 0;
                }
              },
              [&](int accum_m) {});
        }

        auto attn_T = shared_storage.attn_shared_storage().accum_ref();
#if 0
        PRINT_B0_T0("doivj_dropped");
        print_warp_accum<LambdaIterator>(accum, lane_offset, 4, 4);
        PRINT_TENSOR4x4_T0_L0("attn_T", attn_T)
#endif
        accum_t current_di;
        // dSij = (dPij - Di) * Pij
        LambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) { current_di = shared_storage.di()[accum_m]; },
            [&](int accum_m, int accum_n, int idx) {
              // TODO: Otherwise we can get nans as we
              // might have infs here (only seen on f16 tho)
              if (skipBoundsChecks ||
                  (accum_m < num_queries_in_block &&
                   accum_n < num_keys_in_block)) {
                accum_t attn = attn_T.at({accum_n, accum_m});
                accum[idx] = (accum[idx] - current_di) * attn;
              } else {
                accum[idx] = 0;
              }
            },
            [&](int accum_m) {

            });

        // store bias gradient tile dBij to global memory,
        // where dBij = dSij = Pij * (dPij - Di)
        if (p.grad_bias_ptr != nullptr) {
          typename MatmulDOIVJ::BiasGradEpilogue::OutputTileIterator
              output_iter(
                  typename MatmulDOIVJ::BiasGradEpilogue::OutputTileIterator::
                      Params{p.gB_strideM},
                  // grad_bias_ptr is offset to point at beginning of
                  // matrix of shape (queries, keys) for a given
                  // (batch_id, head_id) the pointer arithmetic here produces
                  // a pointer to the start of the current tile within that
                  // matrix
                  p.grad_bias_ptr + query_start * p.gB_strideM + key_start,
                  {num_queries_in_block, num_keys_in_block},
                  thread_id);

          // no-op epilogue operator - just casting and storing contents of
          // accum to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::OutputOp output_op(
              typename MatmulDOIVJ::BiasGradEpilogue::OutputOp::Params{1, 1});
          typename MatmulDOIVJ::BiasGradEpilogue epilogue(
              shared_storage.gradB_epilogue(), thread_id, warp_id, lane_id);
          epilogue(output_op, output_iter, accum, output_iter);
        }

        accum = accum * scale;

#if 0
        PRINT_B0_T0("(doivj - di) * attn * scale");
        print_warp_accum<LambdaIterator>(accum, lane_offset, 4, 4);
#endif

        __syncthreads();
        if (!MatmulGradK::DefaultMmaFromSmem::kIsTransposedA) {
          auto tmpT = shared_storage.tmpT_shared_storage().accum_ref();
          // attn <- attn_T.T
          LambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {},
              [&](int accum_m, int accum_n, int idx) {
                tmpT.at({accum_n, accum_m}) = scalar_t(accum[idx]);
              },
              [&](int accum_m) {});
        }
#ifdef FMHA_BWD_R7
        // [R7 d-gate] the on-chip S^T fill only exists for the whole-B path (kOutputInRF == d<=128); for d>128
        // r7_sT is Empty and the lock-RMW GradQ reads tmp_shared_storage (filled by accumToSmem below).
        if constexpr (kOutputInRF) {
          // [R7] item-2: store S^T ON-CHIP into the persistent r7_sT (B multiplicand) via .at(), dropping the
          // global round-trip. S^T[key,query] = dS[query,key]; DOIVJ lambda gives accum_m=query, accum_n=key.
          // r7_sT survives part3->part4 (persistent). Tests whether .at() matches the whole-B read swizzle.
          auto r7sT = shared_storage.r7_sT().ref();
          LambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {},
              [&](int accum_m /*query*/, int accum_n /*key*/, int idx) {
                r7sT.at({accum_n, accum_m}) = scalar_t(accum[idx]);
              },
              [&](int accum_m) {});
        }
#endif
      }

#ifndef FMHA_BWD_R7
      MatmulDOIVJ::B2bGemm::accumToSmem(
          shared_storage.tmp_shared_storage(),
          accum,
          lane_id,
          output_tile_coords);
#else
      // [R7 d-gate] d>128 uses the lock-RMW GradQ which reads dS from tmp_shared_storage -> fill it.
      if constexpr (!kOutputInRF) {
        MatmulDOIVJ::B2bGemm::accumToSmem(
            shared_storage.tmp_shared_storage(),
            accum,
            lane_id,
            output_tile_coords);
      }
#endif
      __syncthreads();
    }
    // Force `nvcc` to recompute values that depend on the variables just below
    // to use less RF and prevent some spilling
    p.head_dim = warp_uniform(p.head_dim);
    p.k_strideM = warp_uniform(p.k_strideM);
    rematerializeThreadIds();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradQ matmul
    //
    // grad_q[i_start:i_end] += tmp @ k_j
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Skip the loop & associated branches if we know at compile time the number
    // of iterations
// [R7 d-gate] R7's whole-B dQ^T=k^T@S^T is valid only for single-tile dQ (kOutputInRF == d<=128). For d>128
// (64x64x{256,512}) fall back to the lock-RMW column-loop path even in an R7 build (its tmp_shared_storage +
// r7_sT are d-gated above). In a non-R7 build both #ifdef guards vanish and lock-RMW runs unconditionally.
#ifdef FMHA_BWD_R7
    if constexpr (!kOutputInRF && !kSkipGradQ)   // [TWO-PASS] Pass KV skips GradQ (computes dK+dV only)
#else
    if constexpr (!kSkipGradQ)
#endif
    {
    constexpr bool kSingleIterationGradQ =
        kMaxK <= MatmulGradQ::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradQ ? 1 : p.head_dim);
         col += MatmulGradQ::ThreadblockShape::kN) {
      using Mma = typename MatmulGradQ::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;

      cutlass::gemm::GemmCoord problem_size(
          num_queries_in_block,
          false ? MatmulGradQ::ThreadblockShape::kN : p.head_dim - col,
          num_keys_in_block);

      // k_j
      typename Mma::IteratorB iterator_B(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      auto a = shared_storage.tmp_shared_storage().accum_ref();
      Mma mma(
          // operand A: dSij
          shared_storage.tmp_shared_storage().accum_ref(),
          // operand B: Kj
          shared_storage.mm_gradQ().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      int col_id = col / MatmulGradQ::ThreadblockShape::kN;
      int num_cols = kSingleIterationGradQ
          ? 1
          : ceil_div(p.head_dim, MatmulGradQ::ThreadblockShape::kN);
      int storage_id = (col_id + query_start / kBlockSizeI * num_cols);

      if (p.num_splits_key_device() > 1) {
        AtomicLock::acquire(
            &p.workspace_gq[storage_id].lock,
            p.split_key_device() + 1,
            thread_id);
        // Make sure we can see other block's output
        __threadfence();
      }

      AccumTileGmem gmem_tile{&p.workspace_gq[storage_id].buffer[0]};
      if (!kNeedsAccumGradQ ||
          (p.num_splits_key_device() == 1 && key_start == 0)) {
        // if we know we are the first to access it, we know it's only zeros.
        // Avoids a load from gmem (and gmem init as well)
        accum.clear();
      } else {
        gmem_tile.load(accum, thread_id);
      }

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma.set_prologue_done(kPrologueGQ);
      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();
      bool isLastColumn = kSingleIterationGradQ ||
          (col + MatmulGradQ::ThreadblockShape::kN >= p.head_dim);
      if (kPrologueGQ && !isLastColumn) {
        prologueGradQ(col + MatmulGradQ::ThreadblockShape::kN);
      }

      bool isLast = [&]() {
        int32_t next_key = key_start + p.num_splits_key_device() * kBlockSizeJ;
        if (p.num_keys <= next_key) {
          return true;
        }
        if (query_start < getSmallestQueryForKey(p, next_key)) {
          return true;
        }
        return false;
      }();
      // Output results
      if (p.num_splits_key_device() > 1) {
        int32_t numAddsSoFar = -1;
        if (isLast && thread_id == 0) {
          numAddsSoFar = atomicAdd(&p.workspace_gq[storage_id].counter, 1) +
              1; // `atomicAdd` returns the old value
        }
        isLast = __syncthreads_or(
            numAddsSoFar == getNumParallelBlocksForQuery(p, query_start));
        assert(numAddsSoFar <= getNumParallelBlocksForQuery(p, query_start));
      }
      if (kNeedsAccumGradQ && !isLast) {
        gmem_tile.store(accum, thread_id);
        if (p.num_splits_key_device() > 1) {
          // Make sure everyone wrote before we release the lock
          __threadfence();
          __syncthreads();
          AtomicLock::release(&p.workspace_gq[storage_id].lock, thread_id);
        }
      } else {
        // NOTE: We're not releasing the lock because no one is expected
        // to come after us (we're the last one to write)
        typename MatmulGradQ::OutputTileIterator output_it(
            typename MatmulGradQ::OutputTileIterator::Params{p.gQ_strideM()},
            p.grad_query_ptr + query_start * p.gQ_strideM() + col,
            {problem_size.m(), problem_size.n()},
            thread_id);
        bool storage_contains_zeros = kNeedsAccumGradQ || key_start == 0 ||
            (p.num_splits_key_device() > 1);
        accumulateInGmem<MatmulGradQ>(
            isLastColumn ? shared_storage.gradQ_epilogue_lastIter()
                         : shared_storage.gradQ_epilogue(),
            accum,
            output_it,
            storage_contains_zeros,
            warp_id,
            lane_id);
      }
    }
    }
#ifdef FMHA_BWD_R7
    if constexpr (kOutputInRF && !kSkipGradQ)   // [TWO-PASS] Pass KV skips GradQ
    {
      // [R7] reassociated GradQ: dQ^T = k^T @ S^T. A = k^T (r7_kT, filled here from the key tile column-major),
      // B = S^T (r7_sT, already stored by the DOIVJ epilogue). Pure-compute whole-B mma -> accum = dQ^T [d,query].
      // Transposed epilogue: grad_query[query,d] += accum[d,query] (atomicAdd; grad_query is pre-zeroed by torch).
      using R7Mma = typename MatmulGradQ::R7_MmaWholeB;
      cutlass::gemm::GemmCoord r7_ps(
          p.head_dim, num_queries_in_block, num_keys_in_block);
      typename R7Mma::IteratorA iter_kT(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM,
          {r7_ps.m(), r7_ps.k()},
          thread_id,
          cutlass::MatrixCoord{0, 0});
      typename R7Mma::IteratorB iter_sT(
          {int32_t(p.num_queries)},
          p.r7_st_scratch + (int64_t)key_start * p.num_queries + query_start,
          {r7_ps.k(), r7_ps.n()},
          thread_id,
          cutlass::MatrixCoord{0, 0});
      R7Mma::template prologue_whole<true, false>(  // load ONLY k^T (A); S^T (B) already filled on-chip in DOIVJ
          shared_storage.r7_kT(), shared_storage.r7_sT(),
          iter_kT, iter_sT, thread_id, num_keys_in_block);
      __syncthreads();
      R7Mma mma(
          shared_storage.r7_kT(), shared_storage.r7_sT(),
          thread_id, warp_id, lane_id);
      mma.set_prologue_done(true);
      typename R7Mma::FragmentC accum;
      accum.clear();
      auto r7_k_iters =
          (num_keys_in_block + R7Mma::Shape::kK - 1) / R7Mma::Shape::kK;
      mma(r7_k_iters, accum, iter_kT, iter_sT, accum);
      int r7_warp_mn = warp_id %
          (R7Mma::Base::WarpCount::kM * R7Mma::Base::WarpCount::kN);
      auto r7_coords = cutlass::MatrixCoord{
          r7_warp_mn % R7Mma::Base::WarpCount::kM,
          r7_warp_mn / R7Mma::Base::WarpCount::kM};
      auto r7_lo = MatmulGradQ::R7_AccumLambdaIterator::get_lane_offset(
          lane_id, warp_id, r7_coords);
#if defined(FMHA_BWD_COALESCE)
      // [COALESCE, measured] Transpose accum[d,query] through an fp32 smem slab in [query,d] layout so the
      // fold's per-word global RED is d-CONTIGUOUS (4 transaction-granules/request vs the raw scattered 16 -> the
      // far-store port drains ~4x faster; the fold is the measured #1 stall). fp32 stage is MANDATORY (a half
      // stage would round each of the up-to-8 non-sign-aligned partials before the sum -> real accuracy loss).
      // Row padded to kMaxK+1 to kill the stride-128 write-phase bank conflict. Emission stays PER-WORD atomicAdd
      // (a vectorized load-add-store would be a non-atomic RMW that drops other key-splits' addends = a bug).
      // [SHAPE-GATED, measured] coalesce ONLY when the shape is UNDER-FILLED (p.coalesce_dq set by the dispatch):
      // there the fold-burst is the #1 stall so the 4x port cut > the transpose overhead (B1H8-S512 0.97->1.04).
      // On FILLED shapes the fold is not the bottleneck (MAC-bound) so the transpose is pure add-on -> they keep
      // the raw scattered emit (else branch). measured-Q4iii predicted exactly this gating.
      if (p.coalesce_dq) {
#ifndef FMHA_COAL_QS
#define FMHA_COAL_QS 16
#endif
        constexpr int kQs = FMHA_COAL_QS;   // slab height (query-rows/pass); bigger -> fewer passes/barriers
        constexpr int kPad = kMaxK + 1;     // padded d-run (129 at d=128) -> no write bank conflict
        // Reuse r7_kT smem (the whole-B mma's A operand, DEAD after the mma above) as the transpose slab -> no new
        // smem, room for a bigger kQs (fewer passes). Needs kQs*kPad*4 <= sizeof(r7_kT).
        static_assert(
            kQs * kPad * sizeof(float) <=
                sizeof(typename MatmulGradQ::R7_MmaWholeB::Base::SharedStorageA),
            "coalesce slab exceeds r7_kT smem");
        float* slab = reinterpret_cast<float*>(&shared_storage.r7_kT());
        for (int32_t qb = 0; qb < num_queries_in_block; qb += kQs) {
          __syncthreads();
          // WRITE phase: dump this slab's kQs query-columns of the [d,query] fragment into slab[row*kPad + d].
          MatmulGradQ::R7_AccumLambdaIterator::iterateRows(
              r7_lo,
              [&](int) {},
              [&](int accum_m /*d*/, int accum_n /*query*/, int idx) {
                int row = accum_n - qb;
                if (row >= 0 && row < kQs && accum_m < p.head_dim) {
                  slab[row * kPad + accum_m] = float(accum[idx]);
                }
              },
              [&](int) {});
          __syncthreads();
          // READ phase: coalesced per-word RED -- consecutive threads hit consecutive d of a row (contiguous gmem).
          int rows_here = num_queries_in_block - qb;
          if (rows_here > kQs) rows_here = kQs;
          for (int i = thread_id; i < rows_here * p.head_dim; i += kNumThreads) {
            int row = i / p.head_dim;
            int dd = i - row * p.head_dim;
            int32_t q = query_start + qb + row;
            if (q < p.num_queries) {
              atomicAdd(p.r7_dq_f32 + q * p.gQ_strideM() + dd, slab[row * kPad + dd]);
            }
          }
        }
      } else
#endif
      {
      MatmulGradQ::R7_AccumLambdaIterator::iterateRows(
          r7_lo,
          [&](int accum_m) {},
          [&](int accum_m /*d*/, int accum_n /*query*/, int idx) {
            if (accum_m < p.head_dim &&
                (query_start + accum_n) < p.num_queries) {
#if defined(FMHA_BWD_NODQ)
              // [DE-RISK ONLY, breaks correctness] skip the dQ atomic burst entirely to isolate its share of the
              // far-store-port idle (lg_throttle). If lg_throttle drops ~2.14->~0, the fold-burst IS the bottleneck
              // (PACE has headroom); if it holds, the operand-load floor dominates (PACE dead, right-reason wall).
              (void)idx;
#elif defined(FMHA_BWD_R7_FP16DQ)
              // [Q2] fold dQ directly into the fp16 output via sm_70 half atomicAdd -- no fp32 scratch, no
              // separate down-convert pass. grad_query is pre-zeroed by the dispatch. Precision: fp16 accumulation
              // of <= (num key-blocks) causal terms; gated by correctness.
              atomicAdd(
                  reinterpret_cast<__half*>(p.grad_query_ptr) +
                      (query_start + accum_n) * p.gQ_strideM() + accum_m,
                  __float2half(float(accum[idx])));
#elif defined(FMHA_BWD_COAL_FO)
              // [COAL_FO] emit the RED in FRAGMENT ORDER (coalesced 4 granules by construction, no shuffle/smem)
              // WHEN p.fo_dq (filled shapes); else the raw scattered emit (tiny under-filled shape). For a fixed idx
              // the 32 lanes hit contiguous addresses; the (warp,lane,idx) slot is fixed per (d,q) so the cross-split
              // RED accumulates correctly. The dispatch transposes fragment-order -> dQ[query,d] via r7_perm.
              if (p.fo_dq) {
                constexpr int kR7E = int(R7Mma::FragmentC::kElements);
                int64_t qb = query_start / kBlockSizeI;
                int64_t fo = qb * (int64_t)(kR7E * 32 * kNumWarpsPerBlock) +
                    (int64_t)warp_id * (kR7E * 32) + (int64_t)idx * 32 + lane_id;
                atomicAdd(p.r7_dq_f32 + fo, float(accum[idx]));
                if (query_start == 0 && key_start == 0 && p.r7_perm != nullptr) {
                  // [COAL_FO d64] FORWARD map: out_index (query_local*kMaxK + d) -> fragorder slot. Forward (not the
                  // old fragorder->out) so the convert reads it DIRECTLY (no index_copy) AND so head_dim<kBI (d=64,
                  // where only d<head_dim fragorder slots are written) has no zero-collision in the inverse. out_index
                  // spans [0, kBI*kMaxK) <= 16384; the fragorder slot spans the full [0, kBI*kBI) tile.
                  p.r7_perm[(int)(accum_n * kMaxK + accum_m)] =
                      float(warp_id * (kR7E * 32) + idx * 32 + lane_id);
                }
              } else {
                atomicAdd(
                    p.r7_dq_f32 + (query_start + accum_n) * p.gQ_strideM() + accum_m,
                    float(accum[idx]));
              }
#else
              atomicAdd(
                  p.r7_dq_f32 +
                      (query_start + accum_n) * p.gQ_strideM() + accum_m,
                  float(accum[idx]));
#endif
            }
          },
          [&](int accum_m) {});
      }
#ifdef FMHA_BWD_R7_TICKET
      // [Q2 closing-ticket, measured] PER-ARRIVAL finalizer: threshold = #key-blocks <= this query-band
      // (uncapped num_key_blocks = i+1, ns-invariant). Each R7 tile-add punches the shared per-(batch,head,
      // query-block) counter; the (i+1)-th arriver converts the fp32 r7_dq_f32 row to fp16 grad_query IN PLACE,
      // deleting the separate torch down-convert pass. FENCES (device scope, measured): release (__threadfence)
      // before the counter increment so our fp32 adds are visible; acquire (__threadfence) before the winner's
      // read so it sees ALL c(i) contributions. Counter is RMW-only + pre-zeroed (workspace is torch::zeros).
      {
        // PER-UNIT (not per-tile): punch only on THIS CTA's last key-block for the query-band (schedule isLast,
        // mirror of the lock-RMW), threshold = getNumParallelBlocksForQuery (distinct units, mask-aware, <=ns).
        // By its isLast tile the CTA has atomic-added ALL its owned key-blocks to r7_dq_f32. Far fewer counter
        // atomics + fences than per-tile -> avoids the fp32-counter contention that made per-tile net-negative.
        bool isLast_u = [&]() {
          int32_t nk = key_start + p.num_splits_key_device() * kBlockSizeJ;
          if (p.num_keys <= nk) return true;
          if (query_start < getSmallestQueryForKey(p, nk)) return true;
          return false;
        }();
        if (isLast_u) {
          int32_t sid_tk = query_start / kBlockSizeI;
          __threadfence();  // release: this CTA's fp32 adds visible before its counter increment
          __shared__ int s_win_tk;
          if (thread_id == 0) {
            int32_t n = atomicAdd(&p.workspace_gq[sid_tk].counter, 1) + 1;
            s_win_tk = (n == getNumParallelBlocksForQuery(p, query_start)) ? 1 : 0;
          }
          __syncthreads();
          if (s_win_tk) {
            __threadfence();  // acquire: see all units' adds before reading the fp32 row
            int32_t nq_tk = num_queries_in_block * p.head_dim;
            for (int32_t e_tk = thread_id; e_tk < nq_tk; e_tk += (int32_t)kNumThreads) {
              int32_t r_tk = e_tk / p.head_dim, c_tk = e_tk % p.head_dim;
              int64_t off_tk = (int64_t)(query_start + r_tk) * p.gQ_strideM() + c_tk;
              p.grad_query_ptr[off_tk] = scalar_t(p.r7_dq_f32[off_tk]);
            }
          }
        }
      }
#endif
    }
#endif
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradK matmul
    //
    // grad_k[i_start:i_end] += tmp.transpose(-2, -1) @ q_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // [CP1 deferred-Kbar] the in-tile GradK is compiled out ONLY on the fusing path (kFuseQKGradK && aligned
    // tiles), where GradK is deferred + fused into the next tile's QK (+ a post-sweep flush). Every other
    // instantiation (non-aligned, fusion off, d256/512, ...) keeps the baseline in-tile GradK -- so the
    // non-fused path is byte-for-byte baseline (no deferred-serial regression).
    // [TWO-PASS] Pass Q (kComputeQOnly) computes dQ only -> skip GradK entirely (the Pass KV launch owns dK).
    if constexpr (!kComputeQOnly && !(kFuseQKGradK && kKeysQueriesAlignedToBlockSize)) {
    rematerializeThreadIds();

    constexpr bool kSingleIterationGradK =
        kMaxK <= MatmulGradK::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradK ? 1 : p.head_dim);
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;
      // [ACCUM16] dV/dK partials narrow; the fragment type stays MatmulGradQ's, as upstream reuses it
      // for all three tiles -- only the STORAGE width changes here.
      using AccumTileGmem = GmemTile<typename MatmulGradQ::Mma::FragmentC,
                                     (int)kNumThreads, fmha_accum_storage_t>;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block,
          false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col,
          num_queries_in_block);
      auto createEpilogueIter = [&]() {
        return typename MatmulGradK::OutputTileIterator(
            typename MatmulGradK::OutputTileIterator::Params{p.gK_strideM()},
            p.grad_key_ptr + key_start * p.gK_strideM() + col,
            {num_keys_in_block,
             false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col},
            thread_id);
      };

      // q_i
      typename Mma::IteratorB iterator_B(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      auto getTmp = [&](int) { return &shared_storage.tmp_shared_storage(); };
      auto getTmpT = [&](int) { return &shared_storage.tmpT_shared_storage(); };
      // this is basically:
      // opA = kIsTransposedA ? getTmp() : getTmpT();
      bool constexpr kIsTransposedA =
          MatmulGradK::DefaultMmaFromSmem::kIsTransposedA;
      auto& opA = *call_conditional<
          kIsTransposedA,
          decltype(getTmp),
          decltype(getTmpT)>::apply(getTmp, getTmpT, 0);
      Mma mma(
          // operand A: dSij.T
          opA.accum_ref(),
          // operand B: Qi
          shared_storage.mm_gradK().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      int storage_id = col / MatmulGradK::ThreadblockShape::kN;
      const int fragK = kOutputInRF ? storage_id : 0;
      AccumTileGmem gmem_tile{
          reinterpret_cast<fmha_accum_storage_t*>(p.workspace) +
          storage_id * AccumTileGmem::kElementsStored};
      if (!kOutputInRF) {
        if (isFirstQuery || !kNeedsAccumGradK) {
          output_frags.gradK[fragK].clear();
        } else {
          gmem_tile.load(output_frags.gradK[fragK], thread_id);
        }
      }
      mma.set_prologue_done(kPrologueGK);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();

      mma(gemm_k_iterations,
          output_frags.gradK[fragK],
          iterator_B,
          output_frags.gradK[fragK]);
      __syncthreads();
      bool isLastColumn = kSingleIterationGradK ||
          col + MatmulGradK::ThreadblockShape::kN >= p.head_dim;
      if (kPrologueGK && !isLastColumn) {
        prologueGradK(col + MatmulGradK::ThreadblockShape::kN);
      }

      if (kPrologueQK && isLastColumn) {
        int32_t next_query, next_key;
        incrIteration(p, query_start, key_start, next_query, next_key);
        DISPATCH_BOOL(
            next_key != key_start, kForceReloadK, ([&]() {
              prologueQkNextIteration<kForceReloadK::value>(
                  shared_storage, p, next_query, next_key, warp_id, lane_id);
            }));
      }

      // Output results
      if (!kOutputInRF) {
        if (kNeedsAccumGradK && !isLastQuery) {
          gmem_tile.store(output_frags.gradK[fragK], thread_id);
        } else {
          accumulateInGmem<MatmulGradK>(
              isLastColumn ? shared_storage.gradK_epilogue_final()
                           : shared_storage.gradK_epilogue(),
              output_frags.gradK[fragK],
              createEpilogueIter(),
              isFirstQuery || kNeedsAccumGradK,
              warp_id,
              lane_id);
          __syncthreads();
        }
      }
    }
    }  // [CP1 deferred-Kbar] close the in-tile-GradK guard (non-deferred configs)
    }  // end: if constexpr (!kComputeVOnly)
  }

  static CUTLASS_DEVICE int32_t getQueryStartShift(Params const& p) {
    if (p.custom_mask_type == NoCustomMask && p.num_splits_key_device() > 1) {
      return (p.split_key_device() * kBlockSizeI) % getQueryEnd(p);
    }
    return 0;
  }

  // Iteration order logic
  static CUTLASS_DEVICE int32_t
  getQueryStart(Params const& p, int32_t key_start) {
    return getSmallestQueryForKey(p, key_start) + getQueryStartShift(p);
  };
  static CUTLASS_DEVICE int32_t getQueryEnd(Params const& p) {
    return align_up(p.num_queries, kBlockSizeI);
  };

  static CUTLASS_DEVICE int32_t
  getSmallestQueryForKey(Params const& p, int32_t key_start) {
    if (p.custom_mask_type == CausalFromTopLeft) {
      return (key_start / kBlockSizeI) * kBlockSizeI;
    } else if (p.custom_mask_type == CausalFromBottomRight) {
      int first_query =
          cutlass::fast_max(0, key_start - p.num_keys + p.num_queries);
      return (first_query / kBlockSizeI) * kBlockSizeI;
    }
    return 0;
  };

  // Returns how many kernel blocks will write to a given block in `grad_query`
  // This is usually equal to the number of key splits, but can be different
  // for instance in the causal case, or varying seqlen
  static CUTLASS_DEVICE int32_t
  getNumParallelBlocksForQuery(Params const& p, int32_t query_start) {
    int16_t num_key_blocks = ceil_div(p.num_keys, kBlockSizeJ);
    if (p.custom_mask_type == CausalFromTopLeft) {
      int32_t last_key_for_block = query_start + kBlockSizeI - 1;
      last_key_for_block = cutlass::fast_min(last_key_for_block, p.num_keys);
      num_key_blocks = ceil_div(last_key_for_block, kBlockSizeJ);
    } else if (p.custom_mask_type == CausalFromBottomRight) {
      int32_t last_key_for_block =
          query_start + (kBlockSizeI - 1) + (1 + p.num_keys - p.num_queries);
      last_key_for_block = cutlass::fast_min(last_key_for_block, p.num_keys);
      num_key_blocks = ceil_div(last_key_for_block, kBlockSizeJ);
    }
    return cutlass::fast_min(p.num_splits_key_device(), num_key_blocks);
  };

  // Returns the next block to process
  static CUTLASS_DEVICE void incrIteration(
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      int32_t& next_query,
      int32_t& next_key) {
    next_query = query_start + kBlockSizeI;
    next_key = key_start;
    auto query_shift = getQueryStartShift(p);
    // Wrap around
    if (query_shift) {
      if (next_query >= p.num_queries) {
        next_query = getSmallestQueryForKey(p, key_start);
        return;
      } else if (query_start < query_shift && query_shift <= next_query) {
        // jump to next key
      } else {
        return;
      }
    } else {
      if (next_query < p.num_queries) {
        return;
      }
      // jump to next key
    }
    // Next key
    next_key = key_start + p.num_splits_key_device() * kBlockSizeJ;
    next_query = getQueryStart(p, next_key);
  }

  template <bool kForceReloadK>
  static CUTLASS_DEVICE void prologueQkNextIteration(
      SharedStorage& shared_storage,
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    if (query_start >= p.num_queries || key_start >= p.num_keys) {
      return;
    }

    static constexpr bool kReloadK =
        kForceReloadK || !MatmulQK::MmaQK::kSmemContainsEntireMat;
    int thread_id = 32 * warp_id + lane_id;
    typename MatmulQK::MmaQK::IteratorA iterator_A(
        {int32_t(p.k_strideM)},
        p.key_ptr + key_start * p.k_strideM,
        {p.num_keys - key_start, p.head_dim},
        thread_id,
        cutlass::MatrixCoord{0, 0});

    typename MatmulQK::MmaQK::IteratorB iterator_B(
        {int32_t(p.q_strideM)},
        p.query_ptr + query_start * p.q_strideM,
        {p.head_dim, p.num_queries - query_start},
        thread_id,
        cutlass::MatrixCoord{0, 0});

    MatmulQK::MmaQK::template prologue<kReloadK, true>(
        shared_storage.mm_qk_k(),
        shared_storage.mm_qk_q(),
        iterator_A,
        iterator_B,
        thread_id,
        p.head_dim);
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void writeFragsToGmem(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params const& p,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    uint16_t thread_id = 32 * warp_id + lane_id;
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start);
    if constexpr (!kSkipGradV && !kComputeQOnly) {   // dQ+dK sweep / [TWO-PASS] Pass Q leave dV to their dV owner
      // [RF-multi] walk the same column tiles the accumulate loop used, one fragment each.
      CUTLASS_PRAGMA_UNROLL
      for (int t = 0; t < kNumColTilesV; ++t) {
        const int col = t * MatmulGradV::ThreadblockShape::kN;
        typename MatmulGradV::OutputTileIterator outputV_it(
            typename MatmulGradV::OutputTileIterator::Params{p.gV_strideM()},
            p.grad_value_ptr + key_start * p.gV_strideM() + col,
            {num_keys_in_block,
             cutlass::fast_min((int32_t)MatmulGradV::ThreadblockShape::kN,
                               (int32_t)p.head_dim_value - col)},
            thread_id);

        accumulateInGmem<MatmulGradV>(
            shared_storage.gradV_epilogue_final(),
            output_frags.gradV[t],
            outputV_it,
            true,
            warp_id,
            lane_id);
      }
    }

    if constexpr (!kComputeVOnly && !kComputeQOnly) {   // dV-only sweep / [TWO-PASS] Pass Q never form dK
      CUTLASS_PRAGMA_UNROLL
      for (int t = 0; t < kNumColTilesK; ++t) {
        const int col = t * MatmulGradK::ThreadblockShape::kN;
        typename MatmulGradK::OutputTileIterator outputK_it(
            typename MatmulGradK::OutputTileIterator::Params{p.gK_strideM()},
            p.grad_key_ptr + key_start * p.gK_strideM() + col,
            {num_keys_in_block,
             cutlass::fast_min((int32_t)MatmulGradK::ThreadblockShape::kN,
                               (int32_t)p.head_dim - col)},
            thread_id);
        accumulateInGmem<MatmulGradK>(
            shared_storage.gradK_epilogue_final(),
            output_frags.gradK[t],
            outputK_it,
            true,
            warp_id,
            lane_id);
      }
    }
  }

  template <typename MatmulT>
  static CUTLASS_DEVICE void accumulateInGmem(
      typename MatmulT::DefaultEpilogue::SharedStorage& epilogue_smem,
      typename MatmulT::Mma::FragmentC const& accum,
      typename MatmulT::OutputTileIterator output_it,
      bool first,
      uint8_t warp_id,
      uint8_t lane_id) {
    using DefaultEpilogue = typename MatmulT::DefaultEpilogue;
    using DefaultOutputOp = typename MatmulT::DefaultOutputOp;
    using Mma = typename MatmulT::Mma;
    int thread_id = 32 * warp_id + lane_id;
    DISPATCH_BOOL(
        first, kIsFirst, ([&]() {
          static constexpr auto ScaleType = kIsFirst::value
              ? cutlass::epilogue::thread::ScaleType::Nothing
              : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
          using EpilogueOutputOp =
              typename cutlass::epilogue::thread::LinearCombination<
                  typename DefaultOutputOp::ElementOutput,
                  DefaultOutputOp::kCount,
                  typename DefaultOutputOp::ElementAccumulator,
                  typename DefaultOutputOp::ElementCompute,
                  ScaleType>;
          using Epilogue =
              typename cutlass::epilogue::threadblock::EpiloguePipelined<
                  typename DefaultEpilogue::Shape,
                  typename Mma::Operator,
                  DefaultEpilogue::kPartitionsK,
                  typename MatmulT::OutputTileIterator,
                  typename DefaultEpilogue::AccumulatorFragmentIterator,
                  typename DefaultEpilogue::WarpTileIterator,
                  typename DefaultEpilogue::SharedLoadIterator,
                  EpilogueOutputOp,
                  typename DefaultEpilogue::Padding,
                  DefaultEpilogue::kFragmentsPerIteration,
                  true // IterationsUnroll
                  >;
          EpilogueOutputOp rescale({1, 1});
          Epilogue epilogue(epilogue_smem, thread_id, warp_id, lane_id);
          epilogue(rescale, output_it, accum, output_it);
        }));
  }

  template <int kElementsPerAccess>
  static CUTLASS_DEVICE void computeDelta(
      Params const& p,
      int32_t query_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    // Each thread computes one value for Delta
    // Depending on warp configuration, we might have multiple
    // threads of the same warp working on the same row
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    static_assert(kNumThreads >= kBlockSizeI, "");
    static constexpr int kNumThreadsPerLine = kNumThreads / kBlockSizeI;
    int16_t thread_id = 32 * warp_id + lane_id;

    int16_t laneFirstCol = kElementsPerAccess * (lane_id % kNumThreadsPerLine);
    int16_t laneRow = thread_id / kNumThreadsPerLine;
    bool rowPred = (query_start + laneRow) < p.num_queries;
    bool pred = rowPred;

    // on windows, previous syntax __restrict__ AccessType*
    // resulted in error: "restrict" is not allowed
    const AccessType* __restrict__ grad_output_ptr =
        reinterpret_cast<const AccessType*>(
            p.grad_output_ptr + (query_start + laneRow) * p.gO_strideM +
            laneFirstCol);
    const AccessType* __restrict__ output_ptr =
        reinterpret_cast<const AccessType*>(
            p.output_ptr + (query_start + laneRow) * p.o_strideM() +
            laneFirstCol);

    static constexpr int64_t kMaxIters =
        kMaxK / (kElementsPerAccess * kNumThreadsPerLine);
    constexpr int kPipelineStages = 2;
    accum_t delta_value = accum_t(0);
    using GlobalLoad =
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>;
    AccessType frag_grad_output[kPipelineStages];
    AccessType frag_output[kPipelineStages];

    auto loadAndIncrement = [&](int ld_pos, bool is_valid) {
      frag_grad_output[ld_pos].clear();
      frag_output[ld_pos].clear();
      GlobalLoad(frag_grad_output[ld_pos], grad_output_ptr, is_valid);
      GlobalLoad(frag_output[ld_pos], output_ptr, is_valid);
      grad_output_ptr += kNumThreadsPerLine;
      output_ptr += kNumThreadsPerLine;
    };

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < kPipelineStages - 1; ++iter) {
      int ld_pos = iter % kPipelineStages;
      pred = pred &&
          (laneFirstCol + iter * kElementsPerAccess * kNumThreadsPerLine) <
              p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
    }
    auto columnIteration = [&](int iter) {
      // Load for next iter
      int ld_pos = (iter + kPipelineStages - 1) % kPipelineStages;
      pred = pred &&
          (laneFirstCol +
           (iter + kPipelineStages - 1) * kElementsPerAccess *
               kNumThreadsPerLine) < p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < AccessType::kElements; ++i) {
        delta_value += accum_t(frag_output[iter % kPipelineStages][i]) *
            accum_t(frag_grad_output[iter % kPipelineStages][i]);
      }
    };

    // If we have a small lower-bound for K, we can unroll the loop
    if (kMaxK <= 256) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kMaxIters; ++iter) {
        columnIteration(iter);
      }
    } else {
      int num_iters =
          ceil_div(p.head_dim_value, kElementsPerAccess * kNumThreadsPerLine) *
          (kElementsPerAccess * kNumThreadsPerLine);
      for (int iter = 0; iter < num_iters; ++iter) {
        columnIteration(iter);
      }
    }

    // Reduce between workers
    static_assert(
        kNumThreadsPerLine == 1 || kNumThreadsPerLine == 2 ||
            kNumThreadsPerLine == 4,
        "");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kNumThreadsPerLine; i *= 2) {
      delta_value = delta_value + __shfl_xor_sync(0xffffffff, delta_value, i);
    }

    // Store in gmem
    if (rowPred) {
      p.delta_ptr[query_start + laneRow] = delta_value;
    }
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_backward_batched_impl(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_backward_batched(typename AK::Params params);
