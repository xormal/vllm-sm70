// Hand-written Volta (sm_70) HMMA mainloop.
//
// WHY THIS EXISTS. cutlass cannot derive a valid fragment/thread map for three shapes this project
// needs -- <128,32,*>, <256,64,256> and the 256-wide forward tile -- and all three fail the same way:
// the derivation yields ZERO elements per thread (`NumericArrayConverter<..., N=0>`, "zero-sized
// variable tb_frag_B", "Number of iterations must be non-zero"). Nothing about the hardware forbids
// them: a 32-wide tile decomposes into whole m8n8k4 HMMA operations. It is a metaprogramming limit.
// Configuration cannot route around it either -- lowering the iterator alignment reaches the iterator
// but not the thread map, which DefaultMmaCore builds from a hard-wired 128-bit access assumption.
// So the mainloop gets written by hand, and this header is it.
//
// THE INSTRUCTION, AS MEASURED (tools/hmma_map_probe.py -> docs/volta_hmma_map.txt).
// `mma.sync.aligned.m8n8k4` is issued per QUAD PAIR, not per warp. Quad pair p owns lanes
// {4p..4p+3} U {4p+16..4p+19}, and a warp's four quad pairs execute four INDEPENDENT 8x8x4 products.
// So feeding it "lane l supplies row l of A and column l of B" computes the four DIAGONAL 8x8 blocks
// of a 32x32 -- 256 of 1024 cells. That is a real property of the hardware, and it is exactly what
// made the first version of this header silently compute a quarter of its tile: a value check against
// torch passes, because it compares the cells the map names, which are the cells that get computed.
//
// THE FIX IS THE OPERAND FEED, not the instruction. With
//     A row supplied by lane l:  (l&3) | ((l&16)>>2) | (l&8)
//     B col supplied by lane l:  (l&3) | ((l&16)>>2) | ((l&4)<<1)
// the four quad pairs land on the four 8x8 quadrants of a DENSE 16x16 tile, and the warp's 256
// accumulators cover its 256 cells bijectively. Derived from the measured table, asserted on the GPU
// (tests/test_volta_hmma.py checks BIJECTION, not just values -- a value check cannot see coverage).
//
// Accumulator placement, same derivation:
//     row = ((l&1) | ((l&16)>>2) | (l&8)) + 2*((reg>>1)&1)
//     col = ((l&2) | ((l&4)<<1))          +   (reg&1) + 4*((reg>>2)&1)
//
// THE MAPPING IS THE WHOLE PROBLEM. Which matrix element a lane's register holds is not what it is on
// Ampere, and getting it wrong produces plausible-looking garbage rather than an error. Re-run the
// probe after ANY change here.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace fa2_sm70 {

// ---------------------------------------------------------------------------------------------
// The instruction.
//
// Accumulate IN PLACE via "+f" read-write operands. The separate-C-and-D form costs 8 register
// copies per issue -- ptxas does not always coalesce them, and at 8 mma per k-step that is 64 MOVs
// per step competing for issue slots with the HMMAs themselves.
//
// NOT `asm volatile`: this is arithmetic, and volatile pins it in program order, which is precisely
// what stops ptxas from interleaving the LDS of step k+1 with the HMMAs of step k. Non-volatile asm
// with correct constraints is reordered safely and is what makes software pipelining possible at all.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void hmma_884(float (&d)[8], const uint32_t (&a)[2], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 800)
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
#else
  (void)d; (void)a; (void)b;   // host pass / other arch: keep the symbol valid, never executed
#endif
}

// Legacy three-operand form (D = A*B + C with C != D), kept for the raw probe kernel only.
__device__ __forceinline__ void hmma_16x16x4_f32(
    float (&d)[8], const uint32_t (&a)[2], const uint32_t (&b)[2], const float (&c)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = c[i];
  hmma_884(d, a, b);
}

// ---------------------------------------------------------------------------------------------
// Dense 16x16 feed and accumulator placement (measured; see header comment).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ int hmma_a_row(int lane) { return (lane & 3) | ((lane & 16) >> 2) | (lane & 8); }
__device__ __forceinline__ int hmma_b_col(int lane) { return (lane & 3) | ((lane & 16) >> 2) | ((lane & 4) << 1); }
__device__ __forceinline__ int hmma_acc_row(int lane, int reg) {
  return ((lane & 1) | ((lane & 16) >> 2) | (lane & 8)) + 2 * ((reg >> 1) & 1);
}
__device__ __forceinline__ int hmma_acc_col(int lane, int reg) {
  return ((lane & 2) | ((lane & 4) << 1)) + (reg & 1) + 4 * ((reg >> 2) & 1);
}

// ---------------------------------------------------------------------------------------------
// Shared-memory layout.
//
// Both operands are k-major (A is [M][K] row-major, B is [N][K] row-major = B^T column-major), which
// is what FMHA hands us and what makes a fragment 4 CONSECUTIVE halves -- one 8-byte load, no shuffle,
// no transpose.
//
// The row stride is padded by 4 halves. With stride (K+4) halves the 16 rows of a fragment start on
// banks (18*r) mod 32, all distinct, so the 8-byte-per-lane access is conflict-free. Unpadded (K=32,
// stride 64 B) every second row collides: rows 0,2,4.. all start on bank 0 -- an 8-way conflict on
// every single fragment load.
template <int K> struct SmemPad { static constexpr int kLd = K + 4; };

// One fragment = 4 consecutive k for this lane's row. Issued as a single 64-bit load; two 32-bit
// loads of adjacent words is what the previous version wrote, and ptxas kept them as 2x LDS.32.
__device__ __forceinline__ void hmma_ld_frag(const __half* p, uint32_t (&f)[2]) {
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  f[0] = v.x; f[1] = v.y;
}

// ---------------------------------------------------------------------------------------------
// Warp tile: MB*16 rows by NB*16 columns, K walked in steps of 4.
//
// Register blocking is what makes the ratio work. Per k-step a warp issues MB+NB fragment loads and
// MB*NB mma; at MB=NB=1 that is 2 loads per mma (shared memory saturates and the tensor cores idle),
// at MB=2,NB=4 it is 6 loads per 8 mma, and the smem pipe drops to ~75% of the tensor pipe's time.
// ---------------------------------------------------------------------------------------------
template <int MB, int NB>
__device__ __forceinline__ void hmma_k4(float (&acc)[MB][NB][8],
                                        const uint32_t (&fa)[MB][2],
                                        const uint32_t (&fb)[NB][2]) {
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
      hmma_884(acc[i][j], fa[i], fb[j]);
}

template <int MB>
__device__ __forceinline__ void hmma_ld_a(uint32_t (&fa)[MB][2], const __half* pA, int lda, int k) {
#pragma unroll
  for (int i = 0; i < MB; ++i) hmma_ld_frag(pA + (long)(i * 16) * lda + k, fa[i]);
}
template <int NB>
__device__ __forceinline__ void hmma_ld_b(uint32_t (&fb)[NB][2], const __half* pB, int ldb, int k) {
#pragma unroll
  for (int j = 0; j < NB; ++j) hmma_ld_frag(pB + (long)(j * 16) * ldb + k, fb[j]);
}

// Accumulate a whole KT-deep tile out of shared memory, fragments double buffered: the loads for
// step k+4 are issued BEFORE the mma of step k, so their ~25-cycle shared-memory latency is covered
// by MB*NB tensor operations instead of stalling the warp.
//
// pA / pB are this LANE's row/column base: sA + hmma_a_row(lane)*lda, sB + hmma_b_col(lane)*ldb.
template <int MB, int NB, int KT>
__device__ __forceinline__ void hmma_warp_tile(float (&acc)[MB][NB][8],
                                               const __half* pA, int lda,
                                               const __half* pB, int ldb) {
  uint32_t fa[2][MB][2], fb[2][NB][2];
  hmma_ld_a(fa[0], pA, lda, 0);
  hmma_ld_b(fb[0], pB, ldb, 0);
#pragma unroll
  for (int k = 0; k < KT; k += 4) {
    const int cur = (k >> 2) & 1;
    if (k + 4 < KT) {
      hmma_ld_a(fa[cur ^ 1], pA, lda, k + 4);
      hmma_ld_b(fb[cur ^ 1], pB, ldb, k + 4);
    }
    hmma_k4<MB, NB>(acc, fa[cur], fb[cur]);
  }
}

// ---------------------------------------------------------------------------------------------
// Chunk-swizzled shared layout -- the layout the padded one cannot be.
//
// Padding to LD = K+4 gives conflict-free 8-byte fragment loads, but it FORBIDS 16-byte ones: a row
// stride of 72 bytes leaves every odd row only 8-byte aligned. That forces 2x LDS.64 per 8 halves and
// 2x STS.64 per 16-byte global chunk -- and `mio_throttle` (the shared/global queue backing up) was
// measured as the #3 stall in the full pipeline, at 0.53 cycles per issued instruction.
//
// XOR swizzle removes the padding instead: chunk c of row r is stored at c ^ (r & (kChunks-1)). Rows
// stay 16-byte aligned, the 16 rows of a fragment still land on distinct banks, and both directions
// become single 128-bit accesses. Because r*BK has zero low bits, the swizzle is a plain XOR on the
// byte address -- one LOP3, no table.
template <int BK>
struct SwizzleTile {
  static constexpr int kChunks = BK / 8;                       // 16-byte chunks per row
  static_assert(BK % 8 == 0 && (BK / 8 & (BK / 8 - 1)) == 0, "BK/8 must be a power of two");
  __device__ static __forceinline__ int off(int row, int c) {  // in halves
    return row * BK + ((c ^ (row & (kChunks - 1))) * 8);
  }
};

// One 16-byte chunk = 8 consecutive k = TWO k-steps of the instruction. Halves 0..3 feed step k,
// halves 4..7 feed step k+4, so a single LDS.128 replaces two LDS.64 with identical bytes moved.
__device__ __forceinline__ void hmma_frag_pair(const uint4& v, uint32_t (&lo)[2], uint32_t (&hi)[2]) {
  lo[0] = v.x; lo[1] = v.y; hi[0] = v.z; hi[1] = v.w;
}

// Swizzled warp tile. rowA/rowB are this lane's row/column within the block tile
// (warp offset + hmma_a_row(lane) / hmma_b_col(lane)).
template <int MB, int NB, int BK>
__device__ __forceinline__ void hmma_warp_tile_sw(float (&acc)[MB][NB][8],
                                                  const __half* sA, int rowA,
                                                  const __half* sB, int rowB) {
  constexpr int NC = BK / 8, kMask = NC - 1;
  const __half* pA = sA + rowA * BK;
  const __half* pB = sB + rowB * BK;
  const int xa = rowA & kMask, xb = rowB & kMask;   // 16*i keeps the low bits, so one mask per lane
  uint4 fa[2][MB], fb[2][NB];
#pragma unroll
  for (int i = 0; i < MB; ++i) fa[0][i] = *reinterpret_cast<const uint4*>(pA + (long)(i * 16) * BK + ((0 ^ xa) * 8));
#pragma unroll
  for (int j = 0; j < NB; ++j) fb[0][j] = *reinterpret_cast<const uint4*>(pB + (long)(j * 16) * BK + ((0 ^ xb) * 8));
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int b = c & 1;
    if (c + 1 < NC) {   // chunk c+1 is issued BEFORE the 2*MB*NB mma of chunk c cover its latency
#pragma unroll
      for (int i = 0; i < MB; ++i)
        fa[b ^ 1][i] = *reinterpret_cast<const uint4*>(pA + (long)(i * 16) * BK + (((c + 1) ^ xa) * 8));
#pragma unroll
      for (int j = 0; j < NB; ++j)
        fb[b ^ 1][j] = *reinterpret_cast<const uint4*>(pB + (long)(j * 16) * BK + (((c + 1) ^ xb) * 8));
    }
    uint32_t alo[MB][2], ahi[MB][2], blo[NB][2], bhi[NB][2];
#pragma unroll
    for (int i = 0; i < MB; ++i) hmma_frag_pair(fa[b][i], alo[i], ahi[i]);
#pragma unroll
    for (int j = 0; j < NB; ++j) hmma_frag_pair(fb[b][j], blo[j], bhi[j]);
    hmma_k4<MB, NB>(acc, alo, blo);
    hmma_k4<MB, NB>(acc, ahi, bhi);
  }
}

template <int MB, int NB>
__device__ __forceinline__ void hmma_clear(float (&acc)[MB][NB][8]) {
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.f;
}

}  // namespace fa2_sm70
