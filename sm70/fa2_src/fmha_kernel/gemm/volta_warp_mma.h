// Hand-written Volta warp-level MMA operator -- the thing cutlass's MmaVoltaTensorOp cannot be.
//
// WHY. `mma_tensor_op_sm70.h:136` asserts `Shape % GemmShape<32,32,4>` because cutlass's Volta operand
// layouts (VoltaTensorOpMultiplicandCrosswise/Congruous) and their tile iterators are ORGANISED in
// 32x32 interleaved tiles. That is a property of the layout, not of the machine: 32x16 is simply two
// 16x16x4 quad-pair operations. *[measured]* that constraint is what blocks BOTH remaining routes at
// head_dim > 128 -- 16 warps in one block (which shared memory allows; two blocks per SM does not) and
// a register-resident dK/dV accumulator -- because each needs a sub-32 warp tile. See
// docs/VOLTA_SM70.md §6b/§7.1 and the [RESIDENCY COST] note in kernel_backward.h.
//
// So this operator drops the interleaving: any tile 16*MB x 16*NB, any K multiple of 4.
//
// AND IT PAYS, measured (V100 @ 1530 MHz, peak 125.3, 240 blocks; tools/volta_warp_test.cu):
//     32x32 / 8 warps   (what cutlass gives)   102.3 TFLOP/s   81.6%    60 registers
//     64x64 / 8 warps                          119.2           95.2%   190
//     32x16 / 16 warps  (the point)            115.1           91.9%    68
//     64x32 / 16 warps                         112.2           89.5%   114
// The narrow tile at 16 warps BEATS the cutlass-shaped one by 12.5%, comes within 3.5% of the widest
// tile -- so doubling the warps fully pays for the ILP a narrow tile gives up -- and costs 68
// registers instead of 190. That last column is the one that matters: at d>128 the backward's
// occupancy is pinned by registers and smem together, so a shape that is both fast and cheap in
// registers is exactly what the kernel has never been able to ask for.
//
// Correctness is asserted by COVERAGE, not by values: every output cell is stamped with an atomic
// counter and the test fails on any cell written zero or twice. A value check cannot see coverage --
// it compares the cells the map names, which are exactly the cells a partial walk computes, and that
// is how an earlier version that computed a QUARTER of its tile passed.
//
// WHAT IS MEASURED HERE, not assumed (tools/hmma_map_probe.py, scratch probe over all four variants):
//   * the instruction runs PER QUAD PAIR -- quad pair p owns lanes {4p..4p+3} U {4p+16..4p+19} and the
//     warp's four quad pairs are four INDEPENDENT 8x8x4 products. Feeding "lane l -> row l, column l"
//     therefore computes a block-diagonal QUARTER of a 32x32. The dense feed below is what fixes it.
//   * all four layout variants (row/col x row/col) resolve 256/256 cells, and -- the fact that makes
//     this header simple -- THE ACCUMULATOR MAP IS THE SAME FOR ALL OF THEM. Only the operand feed
//     changes. So one accumulator layout, one epilogue, one visitor, whatever the operand orientation.
//   * `.row.row` reads B as [K][N] with N contiguous: lane l takes B[k0 + (l&3)][nbase(l) .. +3], and
//     those four columns are ADJACENT, so it is still one 64-bit load. This is what lets the backward
//     read dO (for dV = P^T dO) and K (for dQ = dS K) WITHOUT transposing them through shared memory.
#pragma once
#ifndef FA2_DIAGQ
#define FA2_DIAGQ 0
#endif
#ifndef FA2_PRE2
#define FA2_PRE2 1   /* замерено лучшим для volta_fwd_so (только accumulate_bi8_pre) */
#endif
#ifndef FA2_ROLL_PV
#define FA2_ROLL_PV 2 /* замерено лучшим для volta_fwd_so (только accumulate_bi8_into) */
#endif
#ifndef FA2_DIAGK
#define FA2_DIAGK 0
#endif
#ifndef FA2_DIAGV
#define FA2_DIAGV 0
#endif
#ifndef FA2_DIAGP
#define FA2_DIAGP 0
#endif

#include "volta_hmma.h"
#include "volta_prims.h"   // наши примитивы: перестановка байт, свёртка смещения, правила раскладки

namespace fa2_sm70 {

// Source layout of operand B.
//   kKMajor: B is [N][K], k contiguous  -> instruction .row.col  (Q*K^T, dO*V^T)
//   kNMajor: B is [K][N], n contiguous  -> instruction .row.row  (P^T*dO, dS*K -- no transpose needed)
enum class BLayout { kKMajor, kNMajor };
// Source layout of operand A.
//   kKMajor: A is [M][K], k contiguous  -> instruction .row.*
//   kMMajor: A is [K][M], m contiguous  -> instruction .col.*  (dQ = dS^T K reads dS as stored)
enum class ALayout { kKMajor, kMMajor };

#define FA2_HMMA_ASM(sfx)                                                                   \
  asm("mma.sync.aligned.m8n8k4." sfx ".f32.f16.f16.f32 "                                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"         \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),                                     \
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])                                      \
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]))

template <BLayout BL, ALayout AL = ALayout::kKMajor>
__device__ __forceinline__ void hmma_884_b(float (&d)[8], const uint32_t (&a)[2], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 800)
  if constexpr (AL == ALayout::kKMajor) {
    if constexpr (BL == BLayout::kKMajor) { FA2_HMMA_ASM("row.col"); }
    else                                  { FA2_HMMA_ASM("row.row"); }
  } else {
    if constexpr (BL == BLayout::kKMajor) { FA2_HMMA_ASM("col.col"); }
    else                                  { FA2_HMMA_ASM("col.row"); }
  }
#else
  (void)d; (void)a; (void)b;
#endif
}

// ---------------------------------------------------------------------------------------------
// Operand feeds for a DENSE 16x16 per instruction (all measured; see header comment).
// ---------------------------------------------------------------------------------------------
// A, k-major: lane l supplies row a_row(l), four consecutive k.
__device__ __forceinline__ int wmma_a_row(int lane) { return (lane & 3) | ((lane & 16) >> 2) | (lane & 8); }
// A, m-major: lane l supplies row (k) k0 + (l&3) and FOUR ADJACENT rows of the tile from a_mbase(l).
__device__ __forceinline__ int wmma_a_krow(int lane) { return lane & 3; }
__device__ __forceinline__ int wmma_a_mbase(int lane) {
  const int q = lane >> 2;                 // 0..7
  return 4 * ((q >> 2) & 1) + 8 * ((q >> 1) & 1);
}
// B, k-major: lane l supplies column b_col(l), four consecutive k.
__device__ __forceinline__ int wmma_b_col(int lane) { return (lane & 3) | ((lane & 16) >> 2) | ((lane & 4) << 1); }
// B, n-major: lane l supplies row (k) k0 + (l&3) and FOUR ADJACENT columns from b_nbase(l).
__device__ __forceinline__ int wmma_b_krow(int lane) { return lane & 3; }
__device__ __forceinline__ int wmma_b_nbase(int lane) {
  const int q = lane >> 2;                 // 0..7
  return 4 * ((q >> 2) & 1) + 8 * (q & 1);
}
// Accumulator: the SAME map for every variant.
__device__ __forceinline__ int wmma_acc_row(int lane, int reg) {
  return ((lane & 1) | ((lane & 16) >> 2) | (lane & 8)) + 2 * ((reg >> 1) & 1);
}
__device__ __forceinline__ int wmma_acc_col(int lane, int reg) {
  return ((lane & 2) | ((lane & 4) << 1)) + (reg & 1) + 4 * ((reg >> 2) & 1);
}

// ---------------------------------------------------------------------------------------------
// The operator. Tile is (16*MB) x (16*NB); K is walked in steps of 4.
//
// Every shape a multiple of 16 is legal, INCLUDING 32x16 -- which is the whole point: it is what
// halves the per-thread fragment (and so the register cost of a resident accumulator) and what puts
// 16 warps in one threadblock.
// ---------------------------------------------------------------------------------------------
template <int MB, int NB, BLayout BL, ALayout AL = ALayout::kKMajor>
struct WarpMma {
  static constexpr int kM = MB * 16, kN = NB * 16;
  float acc[MB][NB][8];

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.f;
  }

  // A: kKMajor -> [rows][lda] with k contiguous;  kMMajor -> [K][lda] with m contiguous.
  // `row0` is this warp's first row of the tile.
  __device__ __forceinline__ void load_a(uint32_t (&fa)[MB][2], const __half* sA, int lda,
                                         int row0, int lane, int k) const {
    if constexpr (AL == ALayout::kKMajor) {
      const __half* p = sA + (long)(row0 + wmma_a_row(lane)) * lda + k;
#pragma unroll
      for (int i = 0; i < MB; ++i) {
        const uint2 v = *reinterpret_cast<const uint2*>(p + (long)(i * 16) * lda);
        fa[i][0] = v.x; fa[i][1] = v.y;
      }
    } else {
      const __half* p = sA + (long)(k + wmma_a_krow(lane)) * lda + row0 + wmma_a_mbase(lane);
#pragma unroll
      for (int i = 0; i < MB; ++i) {
        const uint2 v = *reinterpret_cast<const uint2*>(p + i * 16);
        fa[i][0] = v.x; fa[i][1] = v.y;
      }
    }
  }

  // B: kKMajor -> [cols][ldb] with k contiguous;  kNMajor -> [K][ldb] with n contiguous.
  // `col0` is this warp's first column of the tile.
  __device__ __forceinline__ void load_b(uint32_t (&fb)[NB][2], const __half* sB, int ldb,
                                         int col0, int lane, int k) const {
    if constexpr (BL == BLayout::kKMajor) {
      const __half* p = sB + (long)(col0 + wmma_b_col(lane)) * ldb + k;
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        const uint2 v = *reinterpret_cast<const uint2*>(p + (long)(j * 16) * ldb);
        fb[j][0] = v.x; fb[j][1] = v.y;
      }
    } else {
      const __half* p = sB + (long)(k + wmma_b_krow(lane)) * ldb + col0 + wmma_b_nbase(lane);
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        const uint2 v = *reinterpret_cast<const uint2*>(p + j * 16);
        fb[j][0] = v.x; fb[j][1] = v.y;
      }
    }
  }

  // Та же выдача, но в ЧУЖОЙ накопитель: нужна для расщепления цепочки по k (см. KSP).
  __device__ __forceinline__ void mma_into(float (&dst)[MB][NB][8], const uint32_t (&fa)[MB][2],
                                           const uint32_t (&fb)[NB][2]) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
        hmma_884_b<BL, AL>(dst[i][j], fa[i], fb[j]);
  }
  __device__ __forceinline__ void mma(const uint32_t (&fa)[MB][2], const uint32_t (&fb)[NB][2]) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
        hmma_884_b<BL, AL>(acc[i][j], fa[i], fb[j]);
  }

  // Accumulate a whole K-deep tile, fragments double buffered so the shared-memory latency of step
  // k+4 is covered by the MB*NB tensor operations of step k.
  // ---- [АДРЕСАЦИЯ, шаг 1] Итератор фрагмента вместо пересчёта адреса на каждом доступе.
  // Замер состава инструкций (d=256 H=8) показал: тензорных инструкций у нас РОВНО столько же, что у
  // cutlass (89 128 960), но всего инструкций в 1.74 раза больше, и главный избыток —
  // ЦЕЛОЧИСЛЕННАЯ АДРЕСАЦИЯ (2.54x, +4036M). У cutlass
  // (`gemm/warp/mma_tensor_op_tile_iterator_sm70.h`) декодирование lane в адрес происходит ОДИН раз в
  // конструкторе итератора, а каждый доступ = база + КОНСТАНТА развёрнутого цикла, что сворачивается
  // в непосредственное поле LDS. Сдвиг по k у них — одно сложение указателя (для 32x32 вообще обмен
  // двух указателей). Здесь то же: база считается один раз на плитку, дальше только константы.
  __device__ __forceinline__ const __half* a_base(const __half* sA, int lda, int row0, int lane) const {
    if constexpr (AL == ALayout::kKMajor) return sA + (long)(row0 + wmma_a_row(lane)) * lda;
    else return sA + (long)wmma_a_krow(lane) * lda + row0 + wmma_a_mbase(lane);
  }
  __device__ __forceinline__ const __half* b_base(const __half* sB, int ldb, int col0, int lane) const {
    if constexpr (BL == BLayout::kKMajor) return sB + (long)(col0 + wmma_b_col(lane)) * ldb;
    else return sB + (long)wmma_b_krow(lane) * ldb + col0 + wmma_b_nbase(lane);
  }
  __device__ __forceinline__ void load_a_p(uint32_t (&fa)[MB][2], const __half* pa, int lda, int k) const {
    const __half* p = (AL == ALayout::kKMajor) ? (pa + k) : (pa + (long)k * lda);
#pragma unroll
    for (int i = 0; i < MB; ++i) {
      const __half* q = (AL == ALayout::kKMajor) ? (p + (long)(i * 16) * lda) : (p + i * 16);
      const uint2 v = *reinterpret_cast<const uint2*>(q);
      fa[i][0] = v.x; fa[i][1] = v.y;
    }
  }
  __device__ __forceinline__ void load_b_p(uint32_t (&fb)[NB][2], const __half* pb, int ldb, int k) const {
    const __half* p = (BL == BLayout::kKMajor) ? (pb + k) : (pb + (long)k * ldb);
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const __half* q = (BL == BLayout::kKMajor) ? (p + (long)(j * 16) * ldb) : (p + j * 16);
      const uint2 v = *reinterpret_cast<const uint2*>(q);
      fb[j][0] = v.x; fb[j][1] = v.y;
    }
  }

  // ---- [XOR-СВИЗЛ вместо дополнения]
  // Дополнение строки на +4 half убирает банковские конфликты, но делает шаг строки некратным 16 Б:
  // ни широкой записи, ни широкой загрузки. Перемешивание даёт и то и другое: шаг строки — степень
  // двойки, а физический номер 8-half вектора = (c>>3) ^ (r & 15). Строки по-прежнему ложатся на
  // разные банки, но адрес кратен 16 Б. Так это устроено и у cutlass
  // (`access_contiguous = (vec_col << 2) | ((lane & 3) ^ vec_row)`), только там перемешивание зашито
  // в конструктор итератора, а здесь — в один XOR на доступ.
  // NB: номер строки фрагмента i равен arow + i*16, а 16 & 15 == 0, значит XOR у всех MB фрагментов
  // ОДИН И ТОТ ЖЕ — его можно вынести из цикла.
  static __device__ __forceinline__ int swz(int c, int xr) { return (((c >> 3) ^ xr) << 3) | (c & 7); }

  __device__ __forceinline__ void store_half_swz(__half* dst, int ld, int row0, int col0, int lane) const {
    visit(lane, [&](float v, int r, int c) {
      const int rr = row0 + r;
      dst[(long)rr * ld + swz(col0 + c, rr & 15)] = __float2half(v);
    });
  }

  // Операнд A в k-мажорной раскладке: k идёт по строке подряд, поэтому ОДНА загрузка 16 Б покрывает
  // ДВА шага k (по 4 half каждый). Это вдвое режет число LDS в подаче A.
  template <int K>
  __device__ __forceinline__ void accumulate_swzA(const __half* sA, int lda, int row0,
                                                  const __half* sB, int ldb, int col0, int lane) {
    static_assert(AL == ALayout::kKMajor, "свизл A реализован для k-мажорной раскладки");
    static_assert(K % 8 == 0, "широкая загрузка A покрывает два шага k");
    const int arow = row0 + wmma_a_row(lane);
    const __half* pa = sA + (long)arow * lda;
    const int xr = arow & 15;
    const __half* pb = b_base(sB, ldb, col0, lane);
    uint4 wa[MB];
    uint32_t fa[MB][2], fb[2][NB][2];
    load_b_p(fb[0], pb, ldb, 0);
#pragma unroll
    for (int k = 0; k < K; k += 8) {
      const int off = ((k >> 3) ^ xr) << 3;
#pragma unroll
      for (int i = 0; i < MB; ++i)
        wa[i] = *reinterpret_cast<const uint4*>(pa + (long)(i * 16) * lda + off);
      if (k + 4 < K) load_b_p(fb[1], pb, ldb, k + 4);
#pragma unroll
      for (int i = 0; i < MB; ++i) { fa[i][0] = wa[i].x; fa[i][1] = wa[i].y; }
      mma(fa, fb[0]);
      if (k + 8 < K) load_b_p(fb[0], pb, ldb, k + 8);
#pragma unroll
      for (int i = 0; i < MB; ++i) { fa[i][0] = wa[i].z; fa[i][1] = wa[i].w; }
      mma(fa, fb[1]);
    }
  }

  // [SWZ ⊕ КОНВЕЙЕР d=256, 11.09] То же свизл-чтение Q, но для КУСКОВОЙ подачи K (DC<D): swizzle
  // считается по ГЛОБАЛЬНОМУ индексу k = kbase + k, поэтому Q читается из той же свизл-раскладки
  // полной строки D, а B берётся из текущего куска sB (ldb = DC+PAD). kbase обязан быть кратен 8
  // (широкая загрузка uint4). Снимает банк-конфликты Q (главный доступ: Q перечитывается на каждой
  // плитке ключей) БЕЗ отказа от конвейера K (он лишь управляет, когда кусок попал в sB).
  template <int K>
  __device__ __forceinline__ void accumulate_swzA_dc(const __half* sA, int lda, int row0, int kbase,
                                                     const __half* sB, int ldb, int col0, int lane) {
    static_assert(AL == ALayout::kKMajor, "свизл A реализован для k-мажорной раскладки");
    static_assert(K % 8 == 0, "широкая загрузка A покрывает два шага k");
    const int arow = row0 + wmma_a_row(lane);
    const __half* pa = sA + (long)arow * lda;
    const int xr = arow & 15;
    const __half* pb = b_base(sB, ldb, col0, lane);
    uint4 wa[MB];
    uint32_t fa[MB][2], fb[2][NB][2];
    load_b_p(fb[0], pb, ldb, 0);
#pragma unroll
    for (int k = 0; k < K; k += 8) {
      const int off = ((((kbase + k) >> 3) ^ xr) << 3);
#pragma unroll
      for (int i = 0; i < MB; ++i)
        wa[i] = *reinterpret_cast<const uint4*>(pa + (long)(i * 16) * lda + off);
      if (k + 4 < K) load_b_p(fb[1], pb, ldb, k + 4);
#pragma unroll
      for (int i = 0; i < MB; ++i) { fa[i][0] = wa[i].x; fa[i][1] = wa[i].y; }
      mma(fa, fb[0]);
      if (k + 8 < K) load_b_p(fb[0], pb, ldb, k + 8);
#pragma unroll
      for (int i = 0; i < MB; ++i) { fa[i][0] = wa[i].z; fa[i][1] = wa[i].w; }
      mma(fa, fb[1]);
    }
  }

  // Операнд A в m-мажорной раскладке (фаза dQ): k индексирует СТРОКИ, поэтому XOR зависит от k и
  // считается на каждом доступе; широкой загрузки здесь не будет, только совместимость со свизлом.
  template <int K>
  __device__ __forceinline__ void accumulate_swzA_mmajor(const __half* sA, int lda, int row0,
                                                         const __half* sB, int ldb, int col0, int lane) {
    static_assert(AL == ALayout::kMMajor, "эта ветка для m-мажорной раскладки A");
    const int krow = wmma_a_krow(lane), mbase = row0 + wmma_a_mbase(lane);
    const __half* pb = b_base(sB, ldb, col0, lane);
    uint32_t fa[MB][2], fb[2][NB][2];
    load_b_p(fb[0], pb, ldb, 0);
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int c = (k >> 2) & 1;
      const int rr = k + krow;
      const __half* p = sA + (long)rr * lda;
      const int xr = rr & 15;
#pragma unroll
      for (int i = 0; i < MB; ++i) {
        const uint2 v = *reinterpret_cast<const uint2*>(p + swz(mbase + i * 16, xr));
        fa[i][0] = v.x; fa[i][1] = v.y;
      }
      if (k + 4 < K) load_b_p(fb[c ^ 1], pb, ldb, k + 4);
      mma(fa, fb[c]);
    }
  }

  // ---- [Q В РЕГИСТРАХ] Операнд A не меняется между итерациями внешнего цикла (в forward это Q),
  // поэтому его фрагменты грузятся ОДИН раз и переиспользуются. Замер forward: инструкций доступа у
  // нас 147M против 76M у cutlass при равном общем числе инструкций — мелкая плитка BK означает
  // вчетверо больше итераций, и Q перечитывалась из smem на каждой. Здесь она читается однажды.
  // Стоимость: MB*2*(K/4) регистров (при MB=1, K=128 это 64), проверяйте бюджет.
  template <int K>
  __device__ __forceinline__ void preload_a(uint32_t (&fa)[K / 4][MB][2], const __half* sA, int lda,
                                            int row0, int lane) const {
    const __half* pa = a_base(sA, lda, row0, lane);
#pragma unroll
    for (int k = 0; k < K; k += 4) load_a_p(fa[k >> 2], pa, lda, k);
  }
  template <int NPRE>
  __device__ __forceinline__ void preload_n(uint32_t (&fa)[NPRE ? NPRE : 1][MB][2], const __half* sA,
                                            int lda, int row0, int lane) const {
    const __half* pa = a_base(sA, lda, row0, lane);
#pragma unroll
    for (int t = 0; t < NPRE; ++t) load_a_p(fa[t], pa, lda, t * 4);
  }
  // Частичная предзагрузка: первые NPRE шагов k берутся из регистров, остальные — из smem как обычно.
  // Так стоимость в регистрах дозируется (NPRE*MB*2) и сравнение остаётся в том же бюджете блоков.
  template <int K, int NPRE>
  __device__ __forceinline__ void accumulate_part(const uint32_t (&fa)[NPRE ? NPRE : 1][MB][2],
                                                  const __half* sA, int lda, int row0,
                                                  const __half* sB, int ldb, int col0, int lane) {
    const __half* pa = a_base(sA, lda, row0, lane);
    const __half* pb = b_base(sB, ldb, col0, lane);
    uint32_t fa2[2][MB][2], fb[2][NB][2];
    load_b_p(fb[0], pb, ldb, 0);
    if constexpr (NPRE == 0) load_a_p(fa2[0], pa, lda, 0);
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int st = k >> 2, c = st & 1;
      if (k + 4 < K) {
        load_b_p(fb[c ^ 1], pb, ldb, k + 4);
        if (st + 1 >= NPRE) load_a_p(fa2[c ^ 1], pa, lda, k + 4);
      }
      if (st < NPRE) mma(fa[st], fb[c]);
      else           mma(fa2[c], fb[c]);
    }
  }
  template <int K>
  __device__ __forceinline__ void accumulate_pre(const uint32_t (&fa)[K / 4][MB][2],
                                                 const __half* sB, int ldb, int col0, int lane) {
    const __half* pb = b_base(sB, ldb, col0, lane);
    uint32_t fb[2][NB][2];
    load_b_p(fb[0], pb, ldb, 0);
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int c = (k >> 2) & 1;
      if (k + 4 < K) load_b_p(fb[c ^ 1], pb, ldb, k + 4);
      mma(fa[k >> 2], fb[c]);
    }
  }

  template <int K>
  __device__ __forceinline__ void accumulate(const __half* sA, int lda, int row0,
                                             const __half* sB, int ldb, int col0, int lane) {
    const __half* pa = a_base(sA, lda, row0, lane);
    const __half* pb = b_base(sB, ldb, col0, lane);
    uint32_t fa[2][MB][2], fb[2][NB][2];
    load_a_p(fa[0], pa, lda, 0);
    load_b_p(fb[0], pb, ldb, 0);
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int c = (k >> 2) & 1;
      if (k + 4 < K) {
        load_a_p(fa[c ^ 1], pa, lda, k + 4);
        load_b_p(fb[c ^ 1], pb, ldb, k + 4);
      }
      mma(fa[c], fb[c]);
    }
  }


  // ---- [ОДНОБАЙТОВЫЙ ОПЕРАНД B] ------------------------------------------------------------
  // Мотив: ncu на forward d=256 даёт l1tex 65.9% против тензорного 34.1% -- лимит в БАЙТАХ,
  // поднимаемых из разделяемой памяти, а K и V дают 128 из 288 загрузок на плитку (44%). Если
  // держать их однобайтовыми, фрагмент читается одной LDS.32 вместо LDS.64: вдвое меньше
  // транзакций. Разворот делается В РЕГИСТРАХ и стоит целочисленных операций, которым есть где
  // спрятаться -- тензорный конвейер простаивает на две трети.
  //
  // Разворот без cvt: fp16 0x6400 == 1024.0, мантисса 10 бит, поэтому 0x6400 | u даёт РОВНО 1024+u
  // для u в 0..255. Для знакового int8: u = v + 128 = (uint8)v ^ 0x80, значит закодированное
  // значение равно 1152 + v. Смещение 1152 сворачивается в ранг-1 поправку эпилога и ничего не
  // стоит: для QK это 1152*(сумма q по d) на строку, для PV -- 1152*(сумма P по ключам).
  // ТРИ ИНСТРУКЦИИ ВМЕСТО ДВАДЦАТИ. Нужный результат -- байты [u, 0x64, u, 0x64], то есть чистая
  // ПЕРЕСТАНОВКА, а не арифметика: у NVIDIA для неё есть prmt.b32 (__byte_perm). Знаковое смещение
  // (v+128 = v^0x80) снимается ОДНИМ XOR сразу на все четыре байта.
  // Прежняя запись стоила 4 сдвига + 4 маски + 4 XOR + 6 OR/сдвигов и сидела на КРИТИЧЕСКОМ ПУТИ
  // между загрузкой и MMA -- ровно это и съело выигрыш от однобайтового операнда.
  // Проверено на границах: -128, -1, 0, 127 дают ровно себя после вычитания 1152.
  // BIASED=1 -- байт уже смещён квантователем (u = v+128), XOR не нужен: ДВЕ инструкции вместо трёх.
  // FMT=0 -- знаковый int8 (смещение 1152 снимается свёрткой в ранг-1 у вызывающего);
  // FMT=1 -- e5m2, старший байт fp16: разворот точен побитово и НИКАКОЙ свёртки не требует (§4nn).
  // FMT=2 -- ФАЛЬСИФИКАТОР, не боевой формат. Это РОВНО e5m2 (результат побитово тот же), но с
  // добавленным LOP3 той же формы и того же места, что XOR знака у int8. Нужен, чтобы развести две
  // причины, которые в паре int8/e5m2 слиты: ЦЕНА ЛИШНЕЙ ИНСТРУКЦИИ в цепи распаковки и ЦЕНА
  // ТАБЛИЦЫ МАСШТАБОВ (лишние LDG у везущих варпов + свёртка смещения). Маска берётся из
  // постоянной памяти и равна нулю, поэтому значение не меняется, а инструкция остаётся: компилятор
  // не вправе свернуть чтение __constant__, которое хост может переписать.
  template <int FMT = 0>
  __device__ __forceinline__ static void expand_i8(uint32_t w, uint32_t& f0, uint32_t& f1) {
    if constexpr (FMT == 2) {
      w ^= fa2_sm70::prims::kFakeXorMask;               // тот же LOP3, что у int8; значение не меняет
      fa2_sm70::prims::expand_e5m2x4(w, f0, f1);
      return;
    }
    // FMT=4 -- int8 БЕЗ таблицы масштабов (масштаб вынесен в столбцовый, свёрнут в Q/эпилог):
    //          разворот тот же, что у FMT=0, свёртка смещения 1152 остаётся.
    // FMT=5 -- e5m2 без машинерии масштабов; ДИАГНОСТИКА, не боевой формат и не эталон: нужен
    //          только чтобы отделить выигрыш от снятия ХОЛОСТОЙ единицы, которую платит и e5m2.
    // FMT=6/7 -- СМЕЩЁННЫЙ БАЙТ (offset binary, u = v+128), с таблицей и без неё: XOR не нужен
    //          вовсе, потому что его работу сделал квантователь ОДИН РАЗ (§1a библиотеки).
    //          Значение побитово то же, что у 0/4: смещение остаётся той же константой 1152.
    if constexpr (FMT == 1 || FMT == 3 || FMT == 5) { fa2_sm70::prims::expand_e5m2x4(w, f0, f1); return; }
    // FMT=11 -- e4m3 ПРЯМО ИЗ ЧУЖОГО ПУЛА (vLLM). Шесть инструкций на четыре байта (§1e примитивов),
    //           значение выходит делённым на 256; множитель сворачивается в масштаб софтмакса (K)
    //           и в нормировку эпилога (V), то есть стоит ноль.
    // FMT=12 -- ФАЛЬСИФИКАТОР: те же байты e4m3, но развёрнутые КАК e5m2 (две инструкции). Ответ
    //           НЕВЕРЕН -- вариант существует ровно чтобы измерить ДОЛЮ ВРЕМЕНИ фазы разворота,
    //           а не только её счёт выдач (§3b, разложение фальсификаторами).
    // FMT=13 -- e4m3 прежней записью, ВОСЕМЬ инструкций. Ответ побитово тот же, что у 11.
    if constexpr (FMT == 11) { fa2_sm70::prims::expand_e4m3x4(w, f0, f1); return; }
    if constexpr (FMT == 12) { fa2_sm70::prims::expand_e5m2x4(w, f0, f1); return; }
    if constexpr (FMT == 13) { fa2_sm70::prims::expand_e4m3x4_naive(w, f0, f1); return; }
    // FMT=8 -- смещённый байт + ЧЕРЕДУЮЩАЯСЯ таблица масштабов (одна широкая загрузка на обе).
    if constexpr (FMT >= 6 && FMT <= 10) { fa2_sm70::prims::expand_u8x4_biased(w, f0, f1); return; }
#if defined(FA2_BIASED_U8) && FA2_BIASED_U8
    fa2_sm70::prims::expand_u8x4_biased(w, f0, f1);
#else
    fa2_sm70::prims::expand_i8x4(w, f0, f1);           // §1a библиотеки примитивов
#endif
  }
  __device__ __forceinline__ const int8_t* b_base_i8(const int8_t* sB, int ldb, int col0, int lane) const {
    if constexpr (BL == BLayout::kKMajor) return sB + (long)(col0 + wmma_b_col(lane)) * ldb;
    else return sB + (long)wmma_b_krow(lane) * ldb + col0 + wmma_b_nbase(lane);
  }
  // ОДНА LDS.64 БАЙТОВ = 8 значений = ДВА шага по k. Узкая LDS.32 бесполезна: конвейер разделяемой
  // памяти считает ИНСТРУКЦИИ, а не байты, поэтому вдвое меньше байт при том же числе инструкций не
  // даёт ничего (замерено: 3.800 против 3.663 у fp16 -- ХУЖЕ на цену разворота). Выигрыш только тогда,
  // когда одна инструкция кроет два шага: 288 загрузок на плитку -> 224.
  template <int FMT = 0>
  __device__ __forceinline__ void load_b_p_i8x2(uint32_t (&fb0)[NB][2], uint32_t (&fb1)[NB][2],
                                                const int8_t* pb, int ldb, int k) const {
    const int8_t* p = (BL == BLayout::kKMajor) ? (pb + k) : (pb + (long)k * ldb);
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const int8_t* q = (BL == BLayout::kKMajor) ? (p + (long)(j * 16) * ldb) : (p + j * 16);
      const uint2 v = *reinterpret_cast<const uint2*>(q);      // 8 байт = 8 значений = 2 шага
      expand_i8<FMT>(v.x, fb0[j][0], fb0[j][1]);
      expand_i8<FMT>(v.y, fb1[j][0], fb1[j][1]);
    }
  }
  template <int FMT = 0>
  __device__ __forceinline__ void load_b_p_i8(uint32_t (&fb)[NB][2], const int8_t* pb, int ldb, int k) const {
    const int8_t* p = (BL == BLayout::kKMajor) ? (pb + k) : (pb + (long)k * ldb);
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const int8_t* q = (BL == BLayout::kKMajor) ? (p + (long)(j * 16) * ldb) : (p + j * 16);
      expand_i8<FMT>(*reinterpret_cast<const uint32_t*>(q), fb[j][0], fb[j][1]);
    }
  }
  // Тот же mainloop, но операнд B однобайтовый.
  // Мейнлуп с ОДНОБАЙТОВЫМ операндом B: одна загрузка кроет два шага по k, разворот идёт в
  // регистрах и прячется в тени MMA (тензорный конвейер занят на 34%, целочисленному есть где встать).
  // [SWZ -- ПЕРЕСТАНОВКА ШЕСТНАДЦАТИБАЙТНЫХ КУСКОВ ПРОТИВ БАНКОВОГО КОНФЛИКТА]
  // Банк = (адрес/4) % 32, адрес фрагмента = строка*ldb + k. Полоса берёт строку
  // col0 + wmma_b_col(lane), а b_col даёт в ОДНОЙ восьмёрке полос строки {0..3, 8..11} -- и
  // строки, различающиеся на 8, попадают в ОДИН банк ровно тогда, когда (8*ldb/4) % 32 == 0, то
  // есть когда ldb кратно 16. Но кратность 16 -- ТРЕБОВАНИЕ самой LDS.128. То есть выравнивание,
  // которое РАЗРЕШАЕТ широкую загрузку, оно же и СОЗДАЁТ двухпутевой конфликт: четыре фазы по
  // восемь полос идут 8 тактов вместо 4. Развязка -- не другой шаг (любой кратный 16 даёт то же),
  // а ПЕРЕСТАНОВКА: кусок 16 Б внутри строки выбирается как k XOR ((строка>>3 & 1)*16). Тогда
  // строки, отличающиеся битом 3, читают разные куски -> разные банки, а кратность 16 сохранена.
  // Пишущая сторона (везущие варпы) обязана переставлять ТЕМ ЖЕ правилом.
  // [KSP -- РАСЩЕПЛЕНИЕ ЦЕПОЧКИ НАКОПИТЕЛЯ ПО k]  Замер фальсификаторами: ПЕРВОЕ умножение стоит
  // 35.1 % времени, ВТОРОЕ -- 19.7 %, при РОВНО ТОЙ ЖЕ тензорной работе (256 HMMA) и том же числе
  // вспомогательных выдач (208). Отличается одно: у первого плитка варпа 16x16, то есть ОДИН
  // блок mma и ЧЕТЫРЕ независимых цепочки накопителя (STEP0..3), а у второго NB=4, то есть
  // ШЕСТНАДЦАТЬ. Значит первое умножение упирается не в выдачи и не в память, а в ЗАДЕРЖКУ
  // зависимой HMMA: четырёх цепочек не хватает, чтобы закрыть её, и планировщик простаивает.
  // KSP=2 ведёт ДВА частичных накопителя (чётные и нечётные шаги по k) и складывает их в конце:
  // независимых цепочек становится восемь, цена -- 8 регистров и 8 FADD на плитку.
  // ВНИМАНИЕ: порядок суммирования меняется, поэтому ответ НЕ побитово тот же (точность при этом
  // не хуже: две частичные суммы -- это шаг попарного суммирования).
  // [РАЗВЯЗКА РОЛЕЙ S/O, 12.09] То же, что accumulate_bi8 (k-мажорный байтовый B), но A (Q) взят из
  // ПРЕДЗАГРУЖЕННЫХ регистров fa[K/4][MB][2] -- ни одного чтения A из разделяемой на плитку.
  // Фальсификатор DIAGQ показал потолок этого хода −10 %; смысл развязки -- не трафик, а перекрытие
  // софтмакса S-варпов с HMMA O-варпов (записка 26 §3).
  template <int K, int FMT = 0, int KSP = 1>
  __device__ __forceinline__ void accumulate_bi8_pre(const uint32_t (&fa)[K / 4][MB][2],
                                                     const int8_t* sB, int ldb, int col0, int lane) {
    static_assert(BL == BLayout::kKMajor, "accumulate_bi8_pre: только k-мажорный B (Q*K^T)");
    static_assert(K % 16 == 0, "широкая загрузка: K кратно 16");
    const int8_t* pb = b_base_i8(sB, ldb, col0, lane);
    uint32_t fbA[NB][2], fbB[NB][2];
    uint4 raw[NB];
    float acc2[MB][NB][8];                 // KSP=2: нечётные k-шаги -- во вторую цепочку (латентность HMMA)
    if constexpr (KSP == 2) {
#pragma unroll
      for (int i = 0; i < MB; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
          for (int r = 0; r < 8; ++r) acc2[i][j][r] = 0.f;
    }
    const int8_t* pbj[NB];
#if FA2_PRE2
    // [PRE2] ДВА буфера uint4 по чётности группы из 16 k: группа g+2 грузится, когда группа g отработала, --
    // задержка LDS.128 укрыта целой группой (8 HMMA), а не одним шагом.
    uint4 rawN[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      pbj[j] = pb + (long)(j * 16) * ldb;
      raw[j]  = *reinterpret_cast<const uint4*>(pbj[j]);
      rawN[j] = (16 < K) ? *reinterpret_cast<const uint4*>(pbj[j] + 16) : make_uint4(0, 0, 0, 0);
    }
#else
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      pbj[j] = pb + (long)(j * 16) * ldb;
      raw[j] = *reinterpret_cast<const uint4*>(pbj[j]);
    }
#endif
#pragma unroll
    for (int j = 0; j < NB; ++j) expand_i8<FMT>(raw[j].x, fbA[j][0], fbA[j][1]);
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int c = (k >> 2) & 1, t = (k >> 2) & 3;
      if constexpr (KSP == 2) { if (c) mma_into(acc2, fa[k >> 2], fbB); else mma(fa[k >> 2], fbA); }
      else mma(fa[k >> 2], c ? fbB : fbA);
      if (k + 4 < K) {
#if FA2_PRE2
        const int g = (k >> 4) & 1;               // чётность текущей группы
        if (t == 3) {                             // группа кончилась: её буфер -> под группу g+2
#pragma unroll
          for (int j = 0; j < NB; ++j) {
            const uint4 nv = (k + 4 + 16 < K) ? *reinterpret_cast<const uint4*>(pbj[j] + (k + 4 + 16)) : make_uint4(0, 0, 0, 0);
            if (g) rawN[j] = nv; else raw[j] = nv;
          }
        }
        const int nt = (t + 1) & 3;
        const int gn = ((k + 4) >> 4) & 1;        // чётность группы следующего шага
#pragma unroll
        for (int j = 0; j < NB; ++j) {
          const uint4 rb = gn ? rawN[j] : raw[j];
          const uint32_t w = (nt == 0) ? rb.x : (nt == 1) ? rb.y : (nt == 2) ? rb.z : rb.w;
          if (c) expand_i8<FMT>(w, fbA[j][0], fbA[j][1]);
          else   expand_i8<FMT>(w, fbB[j][0], fbB[j][1]);
        }
#else
        if (t == 3) {
#pragma unroll
          for (int j = 0; j < NB; ++j) raw[j] = *reinterpret_cast<const uint4*>(pbj[j] + (k + 4));
        }
        const int nt = (t + 1) & 3;
#pragma unroll
        for (int j = 0; j < NB; ++j) {
          const uint32_t w = (nt == 0) ? raw[j].x : (nt == 1) ? raw[j].y : (nt == 2) ? raw[j].z : raw[j].w;
          if (c) expand_i8<FMT>(w, fbA[j][0], fbA[j][1]);
          else   expand_i8<FMT>(w, fbB[j][0], fbB[j][1]);
        }
#endif
      }
    }
    if constexpr (KSP == 2) {
#pragma unroll
      for (int i = 0; i < MB; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
          for (int r = 0; r < 8; ++r) acc[i][j][r] += acc2[i][j][r];
    }
  }
  // [РАЗВЯЗКА РОЛЕЙ S/O] accumulate_bi8 для n-мажорного байтового B (P*V) с ВНЕШНИМ накопителем:
  // накопитель O-варпов живёт в union с Q-фрагментами S-варпов, чтобы две роли не держали регистры порознь.
  template <int K, int FMT = 0>
  __device__ __forceinline__ void accumulate_bi8_into(float (&dst)[MB][NB][8], const __half* sA, int lda, int row0,
                                                      const int8_t* sB, int ldb, int col0, int lane) {
    static_assert(BL == BLayout::kNMajor, "accumulate_bi8_into: только n-мажорный B (P*V)");
    const __half* pa = a_base(sA, lda, row0, lane);
    const int8_t* pb = b_base_i8(sB, ldb, col0, lane);
    uint32_t fa[2][MB][2], fb[2][NB][2];
    load_a_p(fa[0], pa, lda, 0);
    load_b_p_i8<FMT>(fb[0], pb, ldb, 0);
#if FA2_ROLL_PV
    // [ROLL_PV] тело развёрнуто на 16 k (4 шага, чётность буферов статична), внешний цикл по группам НЕ развёрнут:
    // код PV меньше вчетверо -- проба гипотезы «no_instruction» (кэш инструкций делят S- и O-варпы одного SMSP).
    constexpr int kG = (FA2_ROLL_PV == 2) ? 8 : 16;
    static_assert(K % kG == 0, "ROLL_PV: K кратно телу");
#pragma unroll 1
    for (int g = 0; g < K; g += kG) {
#pragma unroll
      for (int kk = 0; kk < kG; kk += 4) {
        const int c = (kk >> 2) & 1, k = g + kk;
        if (k + 4 < K) { load_a_p(fa[c ^ 1], pa, lda, k + 4); load_b_p_i8<FMT>(fb[c ^ 1], pb, ldb, k + 4); }
        mma_into(dst, fa[c], fb[c]);
      }
    }
#else
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int c = (k >> 2) & 1;
      if (k + 4 < K) { load_a_p(fa[c ^ 1], pa, lda, k + 4); load_b_p_i8<FMT>(fb[c ^ 1], pb, ldb, k + 4); }
      mma_into(dst, fa[c], fb[c]);
    }
#endif
  }
  // [РАЗВЯЗКА РОЛЕЙ, вариант QSMEM] Q из разделяемой (как в ws), но цикл по k СВЁРНУТ по группам 16:
  // код QK меньше вчетверо -- против «no_instruction» (S- и O-варпы одного SMSP делят кэш инструкций).
  template <int K, int FMT = 0>
  __device__ __forceinline__ void accumulate_bi8_roll(const __half* sA, int lda, int row0,
                                                      const int8_t* sB, int ldb, int col0, int lane) {
    static_assert(BL == BLayout::kKMajor, "accumulate_bi8_roll: только k-мажорный B");
    static_assert(K % 16 == 0, "K кратно 16");
    const __half* pa = a_base(sA, lda, row0, lane);
    const int8_t* pb = b_base_i8(sB, ldb, col0, lane);
    uint32_t fa[2][MB][2], fbA[NB][2], fbB[NB][2];
    uint4 raw[NB];
    const int8_t* pbj[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) { pbj[j] = pb + (long)(j * 16) * ldb; raw[j] = *reinterpret_cast<const uint4*>(pbj[j]); }
    load_a_p(fa[0], pa, lda, 0);
#pragma unroll
    for (int j = 0; j < NB; ++j) expand_i8<FMT>(raw[j].x, fbA[j][0], fbA[j][1]);
#pragma unroll 1
    for (int g = 0; g < K; g += 16) {
#pragma unroll
      for (int kk = 0; kk < 16; kk += 4) {
        const int k = g + kk, c = (kk >> 2) & 1, t = (kk >> 2) & 3;
        if (k + 4 < K) load_a_p(fa[c ^ 1], pa, lda, k + 4);
        mma(fa[c], c ? fbB : fbA);
        if (k + 4 < K) {
          if (t == 3) {
#pragma unroll
            for (int j = 0; j < NB; ++j) raw[j] = *reinterpret_cast<const uint4*>(pbj[j] + (k + 4));
          }
          const int nt = (t + 1) & 3;
#pragma unroll
          for (int j = 0; j < NB; ++j) {
            const uint32_t w = (nt == 0) ? raw[j].x : (nt == 1) ? raw[j].y : (nt == 2) ? raw[j].z : raw[j].w;
            if (c) expand_i8<FMT>(w, fbA[j][0], fbA[j][1]);
            else   expand_i8<FMT>(w, fbB[j][0], fbB[j][1]);
          }
        }
      }
    }
  }
  template <int K, int FMT = 0, int SWZ = 0, int KSP = 1>
  __device__ __forceinline__ void accumulate_bi8(const __half* sA, int lda, int row0,
                                                 const int8_t* sB, int ldb, int col0, int lane) {
    const __half* pa = a_base(sA, lda, row0, lane);
    const int8_t* pb = b_base_i8(sB, ldb, col0, lane);
    // ПАРНАЯ ЗАГРУЗКА ТОЛЬКО ДЛЯ k-МАЖОРНОГО ОПЕРАНДА. У n-мажорного база полосы шагает по 4
    // (wmma_b_nbase даёт {0,4,8,12}), поэтому 8-байтовое чтение с неё НЕ выровнено -- misaligned
    // address. Это свойство РАСКЛАДКИ инструкции, а не кода: там, где шаг полосы кратен 4, а не 8,
    // широкая загрузка запрещена аппаратно.
    if constexpr (BL == BLayout::kKMajor) {
      // ОДНА LDS.128 НА ЧЕТЫРЕ ШАГА (§1c примитивов): mio_throttle считает ИНСТРУКЦИИ, поэтому
      // ширина загрузки -- прямой рычаг по нему. Замерено по пути: 20-инструкционный разворот
      // 3.797 мс, перестановкой байт 3.315 при базе fp16 3.666.
      static_assert(K % 16 == 0, "широкая загрузка: K кратно 16");
      // ПРОГРАММНЫЙ КОНВЕЙЕР: разворот шага t+1 выдаётся ПОСЛЕ mma шага t и от неё НЕ ЗАВИСИТ.
      // Прежняя запись разворачивала всю группу ПАЧКОЙ перед MMA -- цепочка expand -> mma
      // оставалась последовательной, просто короче. Здесь целочисленная работа встаёт в свободные
      // слоты МЕЖДУ тензорными, на непересекающихся регистрах: тензорный и ALU полезны одновременно.
      // Одна LDS.128 раз в четыре шага держит MIO пустым (mio_throttle был первым простоем, 0.94).
      static_assert(K % 16 == 0, "конвейер байтового операнда: K кратно 16");
      uint32_t fa[2][MB][2], fbA[NB][2], fbB[NB][2];
      uint4 raw[NB];
      float acc2[MB][NB][8];
      if constexpr (KSP == 2) {
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
          for (int j = 0; j < NB; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) acc2[i][j][r] = 0.f;
      }
      // [ПРЯМАЯ АДРЕСАЦИЯ ФРАГМЕНТОВ] Адрес (j*16)*ldb пересчитывался ВНУТРИ цикла по k, а ldb --
      // рантаймовый, поэтому каждый пересчёт это IMAD. Разбор SASS: IMAD был ПЕРВОЙ статьёй избытка
      // байтового пути (1.08 инструкции на HMMA против 0.25 у fp16) даже после того, как ту же
      // ошибку убрали в перекладках. База фрагмента КОНСТАНТНА по k -- выносим её из цикла.
      const int8_t* pbj[NB];
      int swj[NB];        // сдвиг перестановки полосы: 0 или 16 байт, ПОСТОЯНЕН по k
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        // [АУДИТ БАНКОВ] АМПЛИТУДА 64 Б, А НЕ 16. Сдвиг на 16 Б МЁРТВ: он сажает строку 8 на банки
        // строки 1, то есть переставляет внутри той же четвёрки банков и конфликт не разводит.
        // Замерено: амплитуда 16 -> конфликты 19.4 % -> 19.5 % (не сдвинулись) и время 0.9875x,
        // то есть УБЫТОК 1.25 %; амплитуда 64 -> конфликты 5.8 %, время 1.0341x на 6 формах из 6.
        // Читается вместе с записью в volta_fwd_ws.cuh (kcd): обе амплитуды обязаны совпадать.
        swj[j] = SWZ ? ((((col0 + wmma_b_col(lane) + j * 16) >> 3) & 1) * 64) : 0;
        pbj[j] = pb + (long)(j * 16) * ldb;
        raw[j] = *reinterpret_cast<const uint4*>(pbj[j] + swj[j]);
      }
      load_a_p(fa[0], pa, lda, 0);
#pragma unroll
      for (int j = 0; j < NB; ++j) expand_i8<FMT>(raw[j].x, fbA[j][0], fbA[j][1]);
#pragma unroll
      for (int k = 0; k < K; k += 4) {
        const int c = (k >> 2) & 1, t = (k >> 2) & 3;
#if FA2_DIAGQ
        // [ФАЛЬСИФИКАТОР] A (Q) грузится ОДИН раз на плитку и переиспользуется: ответ НЕВЕРЕН,
        // время -- потолок любого «Q в регистрах». Снимает 64 LDS.64 на плитку на варп (40 % волн).
        if (k == 0) { fa[1][0][0] = fa[0][0][0]; fa[1][0][1] = fa[0][0][1]; }
#else
        load_a_p(fa[c ^ 1], pa, lda, k + 4);              // A следующего шага: LSU, независимо
#endif
        if constexpr (KSP == 2) {                         // нечётные шаги -- во ВТОРУЮ цепочку
          if (c) mma_into(acc2, fa[c], fbB); else mma(fa[c], fbA);
        } else
        mma(fa[c], c ? fbB : fbA);                        // тензорная выдача
        if (k + 4 < K) {                                  // МЕЖДУ MMA: готовим B следующего шага
#if FA2_DIAGK
          if (false) {   // [ФАЛЬСИФИКАТОР] K грузится один раз на плитку: ответ неверен, время -- потолок
#else
          if (t == 3) {
#endif
#pragma unroll
            for (int j = 0; j < NB; ++j)   // база уже посчитана; XOR -- перестановка куска (см. SWZ)
              raw[j] = *reinterpret_cast<const uint4*>(pbj[j] + ((k + 4) ^ swj[j]));
          }
          const int nt = (t + 1) & 3;
#pragma unroll
          for (int j = 0; j < NB; ++j) {
            const uint32_t w = (nt == 0) ? raw[j].x : (nt == 1) ? raw[j].y
                             : (nt == 2) ? raw[j].z : raw[j].w;
            if (c) expand_i8<FMT>(w, fbA[j][0], fbA[j][1]);
            else   expand_i8<FMT>(w, fbB[j][0], fbB[j][1]);
          }
        }
      }
      if constexpr (KSP == 2) {
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
          for (int j = 0; j < NB; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) acc[i][j][r] += acc2[i][j][r];
      }
    } else {
      uint32_t fa[2][MB][2], fb[2][NB][2];
      load_a_p(fa[0], pa, lda, 0);
      load_b_p_i8<FMT>(fb[0], pb, ldb, 0);
#pragma unroll
      for (int k = 0; k < K; k += 4) {
        const int c = (k >> 2) & 1;
#if FA2_DIAGV
        if (k + 4 < K) { load_a_p(fa[c ^ 1], pa, lda, k + 4); if (k == 0) { for (int j = 0; j < NB; ++j) { fb[1][j][0] = fb[0][j][0]; fb[1][j][1] = fb[0][j][1]; } } }
#elif FA2_DIAGP
        if (k + 4 < K) { if (k == 0) { fa[1][0][0] = fa[0][0][0]; fa[1][0][1] = fa[0][0][1]; } load_b_p_i8<FMT>(fb[c ^ 1], pb, ldb, k + 4); }
#else
        if (k + 4 < K) { load_a_p(fa[c ^ 1], pa, lda, k + 4); load_b_p_i8<FMT>(fb[c ^ 1], pb, ldb, k + 4); }
#endif
        mma(fa[c], fb[c]);
      }
    }
  }

  // [softDMA] Same as accumulate(), but calls hook(step) before each k-step's mma. The point is to
  // place global-memory issues BETWEEN mma steps instead of bursting them before the loop: the bus
  // slots become deterministic and the issue-to-use distance becomes structural rather than whatever
  // ptxas decides. Measured in isolation (tools/volta_pipe_test.cu): +21..43% over the burst form,
  // and unlike loader-warp specialisation it keeps every thread issuing, so memory-level parallelism
  // and each warp's self-coverage are untouched.
  template <int K, typename Hook>
  __device__ __forceinline__ void accumulate_hooked(const __half* sA, int lda, int row0,
                                                    const __half* sB, int ldb, int col0, int lane,
                                                    Hook&& hook) {
    uint32_t fa[2][MB][2], fb[2][NB][2];
    load_a(fa[0], sA, lda, row0, lane, 0);
    load_b(fb[0], sB, ldb, col0, lane, 0);
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int c = (k >> 2) & 1;
      hook(k >> 2);
      if (k + 4 < K) {
        load_a(fa[c ^ 1], sA, lda, row0, lane, k + 4);
        load_b(fb[c ^ 1], sB, ldb, col0, lane, k + 4);
      }
      mma(fa[c], fb[c]);
    }
  }

  // Visit every accumulator element with its (row, col) INSIDE the warp tile. This is what a softmax
  // or any row/column reduction needs: cutlass hides the same thing behind
  // DefaultMmaAccumLambdaIterator, and here it is four lines because the map was measured.
  template <typename F>
  __device__ __forceinline__ void visit(int lane, F&& f) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r)
          f(acc[i][j][r], i * 16 + wmma_acc_row(lane, r), j * 16 + wmma_acc_col(lane, r));
  }

  template <typename F>
  __device__ __forceinline__ void visit(int lane, F&& f) const {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r)
          f(acc[i][j][r], i * 16 + wmma_acc_row(lane, r), j * 16 + wmma_acc_col(lane, r));
  }

  // Write the tile out as fp16 so the NEXT gemm can take it as an operand: [row][ld] row-major, i.e.
  // k-major when this tile's columns are the next contraction's k (S -> P^T dO), and n-major when
  // they are its n. One store per element; the scatter is inherent to the accumulator layout, and it
  // is what cutlass's B2bGemm does too.
  __device__ __forceinline__ void store_half(__half* dst, int ld, int row0, int col0, int lane) const {
    visit(lane, [&](float v, int r, int c) { dst[(long)(row0 + r) * ld + col0 + c] = __float2half(v); });
  }
};

}  // namespace fa2_sm70
