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
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "custom_mma_base.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Transformation applied to A operand
    typename TransformA_ = NumericArrayConverter<
        typename SmemIteratorA_::Element,
        typename IteratorA_::Element,
        IteratorA_::Fragment::kElements>,
    ///
    /// Transformation applied to B operand
    typename TransformB_ = NumericArrayConverter<
        typename SmemIteratorB_::Element,
        typename IteratorB_::Element,
        IteratorB_::Fragment::kElements>,
    /// [PREFETCH] Number of pipeline stages. 2 = classic double-buffer (default, shipped path). When set to
    /// the whole tile's kblock count, the operator can run a whole-operand-resident prologue (all kblocks
    /// pre-loaded, mainloop is pure compute) -- the sm_70 substitute for the sm_80 kSmemContainsEntireB path.
    int kNStages_ = 2,
    /// [PREFETCH] whole-operand-resident pure-compute mainloop (both A and B pre-loaded). Explicit flag rather
    /// than the kNStages_>2 heuristic, so it also works when the whole contraction is only 2 kblocks (R7 GradQ).
    bool kWholeResident_ = false,
    /// Used for partial specialization
    typename Enable = bool>
class CustomMmaPipelined : public CustomMmaBase<Shape_, Policy_, kNStages_> {
 public:
  ///< Base class
  using Base = CustomMmaBase<Shape_, Policy_, kNStages_>;

  using Shape =
      Shape_; ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA =
      IteratorA_; ///< Iterates over tiles of A operand in global memory
  using IteratorB =
      IteratorB_; ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_; ///< Data type of accumulator matrix
  using LayoutC = LayoutC_; ///< Layout of accumulator matrix
  using Policy = Policy_; ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // [PREFETCH] classic pipeline uses 2 stages; whole-operand-resident prologue uses kStages == kblock count.
  // The 2-stage mainloop path (operator() below) still requires exactly 2; the whole-resident path (prologue_
  // done_ + kStages > 2) runs a separate pure-compute mainloop with no global loads.
  // [kBJ=32] The >=2 floor belongs to the CLASSIC path only -- it is the double buffer. The whole-resident
  // path takes a separate `if constexpr` branch with no global loads, no smem stores and no circular wrap,
  // so kStages there means "how many kblocks are resident", and a contraction that IS one kblock (kBJ=32
  // with Shape::kK=32, i.e. R7 GradQ) is a legitimate 1. Blocking it here is what stopped kBlockSizeJ=32
  // from instantiating at all, long before anything about the shape could be measured.
  static_assert(
      (Base::kStages >= 2) || kWholeResident_,
      "CustomMmaPipelined requires kStages >= 2 (except the whole-resident path)");

  static bool const kSmemContainsEntireMat = false;

 private:
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;

 protected:
  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

 public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  CustomMmaPipelined(
      typename Base::SharedStorageA& shared_storageA,
      typename Base::SharedStorageB& shared_storageB,
      int thread_idx, ///< ID within the threadblock
      int warp_idx, ///< ID of warp
      int lane_idx ///< ID of each thread within a warp
      )
      : Base(shared_storageA, shared_storageB, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storageA.ref(), thread_idx),
        smem_iterator_B_(shared_storageB.ref(), thread_idx) {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }
  CUTLASS_DEVICE
  CustomMmaPipelined(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage& st,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : CustomMmaPipelined(
            st.operand_A,
            st.operand_B,
            thread_idx,
            warp_idx,
            lane_idx) {}

  // [PREFETCH] operand-prologue mechanism (was a stub). When set, operator() assumes kblock 0 was already
  // loaded+stored to stage-0 smem by prologue() and skips its own initial load, advancing iterator state to
  // match. Foundation for the whole-B-resident q home slot (measured register-transit prefetch).
  bool prologue_done_ = false;
  CUTLASS_DEVICE
  bool set_prologue_done(bool value) {
    prologue_done_ = value;
    return prologue_done_;
  }

  CUTLASS_DEVICE
  bool set_zero_outside_bounds(bool value) {
    // NOT NEEDED FOR PIPELINED
    // shared memory will always be zero-filled
  }

  // [CP1 fusion] accessors for fused_qk_gradk.h. warp_mma + smem_write_stage_idx + transforms are locals in
  // operator(), so the fused driver owns its own; only the smem/warp iterators need exposing.
  CUTLASS_DEVICE auto& fused_smem_iter_A() { return smem_iterator_A_; }
  CUTLASS_DEVICE auto& fused_smem_iter_B() { return smem_iterator_B_; }
  CUTLASS_DEVICE auto& fused_warp_iter_A() { return this->warp_tile_iterator_A_; }
  CUTLASS_DEVICE auto& fused_warp_iter_B() { return this->warp_tile_iterator_B_; }

  template <bool kLoadA = true, bool kLoadB = true>
  CUTLASS_DEVICE static void prologue(
      typename Base::SharedStorage& shared_storage,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    prologue<kLoadA, kLoadB>(
        shared_storage.operand_A,
        shared_storage.operand_B,
        iterator_A,
        iterator_B,
        thread_idx,
        problem_size_k);
  }

  template <bool kLoadA = true, bool kLoadB = true>
  CUTLASS_DEVICE static void prologue(
      typename Base::SharedStorageA& shared_storageA,
      typename Base::SharedStorageB& shared_storageB,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    // [PREFETCH] pre-load kblock 0 of A/B into stage-0 smem so operator() (with prologue_done_) can skip its
    // own initial load. Mirrors the operator prologue (load -> transform -> store) on fresh smem iterators.
    SmemIteratorA smem_iterator_A(shared_storageA.ref(), thread_idx);
    SmemIteratorB smem_iterator_B(shared_storageB.ref(), thread_idx);
    int gemm_k_iterations = (problem_size_k + Shape::kK - 1) / Shape::kK;
    iterator_A.clear_mask(gemm_k_iterations <= 0);
    iterator_B.clear_mask(gemm_k_iterations <= 0);
    FragmentA tb_frag_A;
    FragmentB tb_frag_B;
    tb_frag_A.clear();
    tb_frag_B.clear();
    if (kLoadA) {
      iterator_A.load(tb_frag_A);
      smem_iterator_A.store(TransformA()(tb_frag_A));
    }
    if (kLoadB) {
      iterator_B.load(tb_frag_B);
      smem_iterator_B.store(TransformB()(tb_frag_B));
    }
  }

  // [PREFETCH] whole-operand-resident prologue: load ALL kNStages_ kblocks of A/B into smem (vs kblock 0 only).
  // Pairs with set_prologue_done(true) + the kNStages_>2 pure-compute mainloop. The whole q (operand B) then
  // lives resident in smem -- the sm_70 substitute for the sm_80 kSmemContainsEntireB path.
  template <bool kLoadA = true, bool kLoadB = true>
  CUTLASS_DEVICE static void prologue_whole(
      typename Base::SharedStorageA& shared_storageA,
      typename Base::SharedStorageB& shared_storageB,
      IteratorA iterator_A,
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    SmemIteratorA smem_iterator_A(shared_storageA.ref(), thread_idx);
    SmemIteratorB smem_iterator_B(shared_storageB.ref(), thread_idx);
    int gemm_k_iterations = (problem_size_k + Shape::kK - 1) / Shape::kK;
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < kNStages_; ++stage) {
      iterator_A.clear_mask(stage >= gemm_k_iterations);
      iterator_B.clear_mask(stage >= gemm_k_iterations);
      FragmentA tb_frag_A;
      FragmentB tb_frag_B;
      tb_frag_A.clear();
      tb_frag_B.clear();
      if (kLoadA) {
        iterator_A.load(tb_frag_A);
        smem_iterator_A.store(TransformA()(tb_frag_A));
      }
      if (kLoadB) {
        iterator_B.load(tb_frag_B);
        smem_iterator_B.store(TransformB()(tb_frag_B));
      }
      ++iterator_A;
      ++iterator_B;
      ++smem_iterator_A;
      ++smem_iterator_B;
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations, ///< number of iterations of the mainloop
      FragmentC& accum, ///< destination accumulator tile
      IteratorA iterator_A, ///< iterator over A operand in global memory
      IteratorB iterator_B, ///< iterator over B operand in global memory
      FragmentC const& src_accum, ///< source accumulator tile
      TransformA transform_A =
          TransformA(), ///< transformation applied to A fragment
      TransformB transform_B =
          TransformB()) { ///< transformation applied to B fragment

    //
    // Prologue
    //

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // [PREFETCH] whole-operand-resident path: prologue() (called externally with set_prologue_done(true)) has
    // pre-loaded ALL kNStages_ kblocks of A and B into smem. The mainloop is then pure compute -- no global
    // loads, no smem stores, no circular wrap -- reading warp tiles linearly through all resident stages.
    if constexpr (kWholeResident_) {
      __syncthreads(); // ensure the external prologue's stores are visible to all warps
      WarpFragmentA warp_frag_A[2];
      WarpFragmentB warp_frag_B[2];
      this->warp_tile_iterator_A_.set_kgroup_index(0);
      this->warp_tile_iterator_B_.set_kgroup_index(0);
      this->warp_tile_iterator_A_.load(warp_frag_A[0]);
      this->warp_tile_iterator_B_.load(warp_frag_B[0]);
      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
      Operator warp_mma;
      CUTLASS_GEMM_LOOP
      for (; gemm_k_iterations > 0; --gemm_k_iterations) {
        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
             ++warp_mma_k) {
          bool is_last = (gemm_k_iterations == 1) &&
              (warp_mma_k == Base::kWarpGemmIterations - 1);
          if (!is_last) {
            this->warp_tile_iterator_A_.set_kgroup_index(
                (warp_mma_k + 1) % Base::kWarpGemmIterations);
            this->warp_tile_iterator_B_.set_kgroup_index(
                (warp_mma_k + 1) % Base::kWarpGemmIterations);
            this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
            this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);
            ++this->warp_tile_iterator_A_;
            ++this->warp_tile_iterator_B_;
          }
          warp_mma(
              accum,
              warp_frag_A[warp_mma_k % 2],
              warp_frag_B[warp_mma_k % 2],
              accum);
        }
      }
      return;
    }

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();

    // The last kblock is loaded in the prolog. [PREFETCH] if prologue_done_, prologue() already loaded+stored
    // kblock 0 into stage-0 smem; skip the load/store here and only advance iterator state to match (tb_frag
    // stays cleared and is overwritten by the mainloop's kblock-1 load before its first store).
    if (!prologue_done_) {
      iterator_A.load(tb_frag_A);
      iterator_B.load(tb_frag_B);
      this->smem_iterator_A_.store(transform_A(tb_frag_A));
      this->smem_iterator_B_.store(transform_B(tb_frag_B));
    }

    ++iterator_A;
    ++iterator_B;

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    // Issue loads during the first warp-level matrix multiply-add *AFTER*
    // issuing shared memory loads (which have the tightest latency
    // requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A(tb_frag_A));

          this->smem_iterator_B_.store(transform_B(tb_frag_B));

          __syncthreads();

          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          } else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0,
                 -Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {
          iterator_A.load(tb_frag_A);
          iterator_B.load(tb_frag_B);

          ++iterator_A;
          ++iterator_B;

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        warp_mma(
            accum,
            warp_frag_A[warp_mma_k % 2],
            warp_frag_B[warp_mma_k % 2],
            accum);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
