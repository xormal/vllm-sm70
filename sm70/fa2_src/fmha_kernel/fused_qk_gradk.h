// [CP1 manual fusion] Interleaved QK || GradK threadblock mainloop for the anchored Phase-1 merge.
// Drives BOTH gemms' warp-mma + double-buffered warp fragments in ONE k-loop so their independent HMMAs
// fill each other's smem-load (Long-Scoreboard) shadows. Reuses cutlass's tuned warp_mma verbatim (only the
// loop orchestration is hand-written -> the slab per-HMMA loss does NOT apply). Numerically identical to
// running the two mmas separately.
//
// BOTH mmas are self-contained (CustomMmaPipelined / MmaPipelinedFromSharedMemory inline their prologue +
// mainloop with LOCAL warp_mma + stage counter + identity transforms), so this driver owns its own warp_mma
// + stage counters for each and replicates both inlined operator()s interleaved.
//   QK  : s      = K @ Q^T ; streams A(K),B(Q) gmem->smem, 2-stage double buffer.
//   GradK: gradK += T^T @ Q; A(=T) FIXED smem tile (warp-iter reads it, no wrap), only B(Q) streamed.
// Transforms are identity (NumericArrayConverter same-type) -> stored raw. GradK's A-scale is disabled.
#pragma once
#include "cutlass/cutlass.h"

namespace fmha_fused {

template <typename MmaQK, typename MmaGK>
CUTLASS_DEVICE void fused_qk_gradk(
    MmaQK& qk,
    typename MmaQK::IteratorA qk_iter_A,   // K (gmem)
    typename MmaQK::IteratorB qk_iter_B,   // Q (gmem)
    typename MmaQK::FragmentC& s_accum,    // pre-cleared; accumulated
    MmaGK& gk,
    typename MmaGK::IteratorB gk_iter_B,   // Q (gmem)
    typename MmaGK::FragmentC& gradK_accum,// read-modify-write (cross-sweep accumulator)
    int gemm_k_iterations) {
  constexpr int kWarpGemmIterations = MmaQK::Base::kWarpGemmIterations;
  constexpr int kStages = MmaQK::Base::kStages;
  constexpr int kPartitionsK = MmaQK::Policy::kPartitionsK;

  using QkWarpA = typename MmaQK::Operator::FragmentA;
  using QkWarpB = typename MmaQK::Operator::FragmentB;
  using QkTbA = typename MmaQK::FragmentA;
  using QkTbB = typename MmaQK::FragmentB;
  using GkWarpA = typename MmaGK::Operator::FragmentA;
  using GkWarpB = typename MmaGK::Operator::FragmentB;
  using GkTbB = typename MmaGK::FragmentB;

  typename MmaQK::Operator qk_warp_mma;
  typename MmaGK::Operator gk_warp_mma;
  int qk_stage = 1;
  int gk_stage = 1;

  // ---------------- Prologue (both), single shared sync ----------------
  QkTbA qk_tb_A;
  QkTbB qk_tb_B;
  qk_tb_A.clear();
  qk_tb_B.clear();
  qk_iter_A.load(qk_tb_A);
  qk_iter_B.load(qk_tb_B);
  ++qk_iter_A;
  ++qk_iter_B;
  qk.fused_smem_iter_A().store(qk_tb_A);
  qk.fused_smem_iter_B().store(qk_tb_B);
  ++qk.fused_smem_iter_A();
  ++qk.fused_smem_iter_B();

  GkTbB gk_tb_B;
  gk_tb_B.clear();
  gk_iter_B.set_residual_tile(gemm_k_iterations == 1);
  gk_iter_B.load(gk_tb_B);
  ++gk_iter_B;
  gk.fused_smem_iter_B().store(gk_tb_B);
  ++gk.fused_smem_iter_B();

  __syncthreads();

  // ---------------- Load warp fragment [0] for both ----------------
  QkWarpA qk_A[2];
  QkWarpB qk_B[2];
  qk.fused_warp_iter_A().set_kgroup_index(0);
  qk.fused_warp_iter_B().set_kgroup_index(0);
  qk.fused_warp_iter_A().load(qk_A[0]);
  qk.fused_warp_iter_B().load(qk_B[0]);
  ++qk.fused_warp_iter_A();
  ++qk.fused_warp_iter_B();

  GkWarpA gk_A[2];
  GkWarpB gk_B[2];
  gk_A[0].clear();
  gk_B[0].clear();
  gk.fused_warp_iter_B().set_kgroup_index(0);
  gk.fused_warp_iter_A().load(gk_A[0]);
  gk.fused_warp_iter_B().load(gk_B[0]);
  ++gk.fused_warp_iter_A();
  ++gk.fused_warp_iter_B();

  qk_iter_A.clear_mask(gemm_k_iterations <= 1);
  qk_iter_B.clear_mask(gemm_k_iterations <= 1);
  gk_iter_B.set_residual_tile(gemm_k_iterations == 2);
  gk_iter_B.clear_mask(gemm_k_iterations <= 1);

  // ---------------- Interleaved mainloop ----------------
  CUTLASS_GEMM_LOOP
  for (; gemm_k_iterations > 0; --gemm_k_iterations) {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {
      bool gk_hasNext = true;

      if (warp_mma_k == kWarpGemmIterations - 1) {
        qk.fused_smem_iter_A().store(qk_tb_A);
        qk.fused_smem_iter_B().store(qk_tb_B);
        if (gemm_k_iterations > 1) {
          gk.fused_smem_iter_B().store(gk_tb_B);
        }
        __syncthreads();   // single shared barrier
        // QK stage advance (A + B stream).
        ++qk.fused_smem_iter_A();
        ++qk.fused_smem_iter_B();
        if (qk_stage == 1) {
          qk.fused_smem_iter_A().add_tile_offset({0, -kStages});
          qk.fused_smem_iter_B().add_tile_offset({-kStages, 0});
        } else {
          qk.fused_warp_iter_A().add_tile_offset(
              {0, -kStages * kPartitionsK * kWarpGemmIterations});
          qk.fused_warp_iter_B().add_tile_offset(
              {-kStages * kPartitionsK * kWarpGemmIterations, 0});
        }
        qk_stage ^= 1;
        // GradK stage advance (B only; A=T fixed).
        ++gk.fused_smem_iter_B();
        if (gk_stage == 1) {
          gk.fused_smem_iter_B().add_tile_offset({-kStages, 0});
        } else {
          gk.fused_warp_iter_B().add_tile_offset(
              {-kStages * kPartitionsK * kWarpGemmIterations, 0});
        }
        gk_stage ^= 1;
        gk_hasNext = gemm_k_iterations > 1;
      }

      qk.fused_warp_iter_A().set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations);
      qk.fused_warp_iter_B().set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations);
      qk.fused_warp_iter_A().load(qk_A[(warp_mma_k + 1) % 2]);
      qk.fused_warp_iter_B().load(qk_B[(warp_mma_k + 1) % 2]);
      ++qk.fused_warp_iter_A();
      ++qk.fused_warp_iter_B();

      if (gk_hasNext) {
        gk.fused_warp_iter_B().set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations);
        gk.fused_warp_iter_A().load(gk_A[(warp_mma_k + 1) % 2]);
        gk.fused_warp_iter_B().load(gk_B[(warp_mma_k + 1) % 2]);
        ++gk.fused_warp_iter_A();
        ++gk.fused_warp_iter_B();
      }

      if (warp_mma_k == 0) {
        qk_iter_A.load(qk_tb_A);
        qk_iter_B.load(qk_tb_B);
        ++qk_iter_A;
        ++qk_iter_B;
        qk_iter_A.clear_mask(gemm_k_iterations <= 2);
        qk_iter_B.clear_mask(gemm_k_iterations <= 2);
        if (gk_hasNext) {
          gk_iter_B.load(gk_tb_B);
          ++gk_iter_B;
          gk_iter_B.set_residual_tile(gemm_k_iterations == 3);
          gk_iter_B.clear_mask(gemm_k_iterations <= 2);
        }
      }

      // The two independent warp-mmas: their HMMAs overlap, filling each other's smem-load shadows.
      qk_warp_mma(s_accum, qk_A[warp_mma_k % 2], qk_B[warp_mma_k % 2], s_accum);
      gk_warp_mma(gradK_accum, gk_A[warp_mma_k % 2], gk_B[warp_mma_k % 2], gradK_accum);
    }
  }
}

} // namespace fmha_fused
