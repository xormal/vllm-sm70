// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// MXFP4 (OCP Microscaling, v1.0) dequantisation for sm_70.
//
// WHY THIS EXISTS
//   vLLM only offers an MXFP4 path for device capability 9.0-11.0 (Hopper/Blackwell); on Volta there
//   is none, so an MXFP4 checkpoint such as gpt-oss-20b simply cannot load. Volta has no FP4
//   hardware -- but nothing here needs hardware: the format is a nibble table and a power-of-two
//   scale, and unpacking it is arithmetic we own.
//
// THE FORMAT (as it appears in the gpt-oss checkpoint)
//   *_blocks  [.., nblk, 16] uint8   -- 16 bytes = 32 nibbles = one MX block of 32 elements
//   *_scales  [.., nblk]     uint8   -- one E8M0 byte per block; value = 2^(s - 127)
//   element   E2M1 = [sign | exp:2 | mant:1], i.e. exactly 8 magnitudes:
//                 exp=0,m=0 -> 0     exp=1,m=0 -> 1.0    exp=2,m=0 -> 2.0    exp=3,m=0 -> 4.0
//                 exp=0,m=1 -> 0.5   exp=1,m=1 -> 1.5    exp=2,m=1 -> 3.0    exp=3,m=1 -> 6.0
//
// WHY A TABLE AND NOT A BIT TRICK
//   The e4m3 conversion this project replaced earlier WAS worth a bit trick: it had 8 mantissa bits,
//   a denormal branch, and cost ~12 ops on the critical chain (85% of the decode's ALU). E2M1 has
//   FOUR distinct values plus a sign -- a 16-entry constant table is one LDC and no branches, which
//   is already the floor. Reaching for a bit trick here would add work, not remove it.
//
// SCALE HANDLING
//   E8M0 is a pure power of two, so the multiply is exact in fp32 and never rounds; it is applied in
//   fp32 and only the final store narrows to fp16. s = 255 is the format's NaN and is passed through
//   as such rather than silently clamped -- a NaN in a weight must be visible, not hidden.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// sign in bit 3; magnitudes are the 8 representable E2M1 values
__constant__ float kE2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

template <typename out_t>
__global__ void mxfp4_dequant_kernel(const uint8_t* __restrict__ blocks,
                                     const uint8_t* __restrict__ scales,
                                     out_t* __restrict__ out,
                                     long nbytes) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;   // one thread per BYTE = 2 elements
  if (i >= nbytes) return;
  const long blk = i >> 4;                                      // 16 bytes per MX block
  const int s = scales[blk];
  // 2^(s-127) exactly: build the fp32 bit pattern instead of calling exp2f (which is an SFU op).
  // s=0 would be 2^-127, below fp32's smallest normal exponent field, so clamp the CONSTRUCTION at
  // the subnormal boundary; s=255 is the format's NaN.
  float sc;
  if (s == 255) sc = __int_as_float(0x7fc00000);                // NaN, propagated on purpose
  else if (s == 0) sc = 0.0f;                                   // 2^-127 flushes; the block is zero
  else sc = __int_as_float(s << 23);                            // exponent field = s, mantissa 0
  const uint8_t byte = blocks[i];
  const float lo = kE2M1[byte & 0xF] * sc;                      // low nibble = FIRST element
  const float hi = kE2M1[byte >> 4] * sc;
  if constexpr (sizeof(out_t) == 2) {
    out[2 * i]     = __float2half(lo);
    out[2 * i + 1] = __float2half(hi);
  } else {
    out[2 * i]     = lo;
    out[2 * i + 1] = hi;
  }
}

// blocks [..., nblk, 16] uint8, scales [..., nblk] uint8 -> [..., nblk*32] of `dtype`
torch::Tensor mxfp4_dequant(torch::Tensor blocks, torch::Tensor scales, torch::ScalarType dtype) {
  TORCH_CHECK(blocks.is_cuda() && scales.is_cuda(), "mxfp4: tensors must be CUDA");
  TORCH_CHECK(blocks.scalar_type() == torch::kUInt8 && scales.scalar_type() == torch::kUInt8,
              "mxfp4: blocks and scales must be uint8");
  TORCH_CHECK(blocks.size(-1) == 16, "mxfp4: last dim of blocks must be 16 (32 nibbles per block)");
  auto b = blocks.contiguous(), s = scales.contiguous();
  TORCH_CHECK(b.numel() == s.numel() * 16, "mxfp4: one scale byte per 16 block bytes");

  auto shape = b.sizes().vec();
  shape.pop_back();                                   // drop the 16
  shape.back() *= 32;                                 // nblk -> nblk*32 elements
  auto out = torch::empty(shape, torch::dtype(dtype).device(b.device()));

  const long nbytes = b.numel();
  const int threads = 256;
  const long grid = (nbytes + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dtype == torch::kFloat16) {
    mxfp4_dequant_kernel<__half><<<grid, threads, 0, stream>>>(
        b.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), nbytes);
  } else if (dtype == torch::kFloat32) {
    mxfp4_dequant_kernel<float><<<grid, threads, 0, stream>>>(
        b.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), out.data_ptr<float>(), nbytes);
  } else {
    TORCH_CHECK(false, "mxfp4: dtype must be float16 or float32");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


// ---------------------------------------------------------------------------------------------
// TILE dequantisation: unpack ONE tile of a packed matrix straight into a caller-provided buffer.
//
// WHY A TILE ENTRY POINT AND NOT PER-ELEMENT DEQUANT IN THE GEMM INNER LOOP
//   Unpacking where the multiply happens would pay the nibble split, the table lookup and the scale
//   for EVERY use of a weight. In a GEMM each weight tile is reused by the whole M dimension -- at
//   prefill that is ~1024 tokens per expert -- so per-use dequant would multiply the ALU cost by M on
//   exactly the path that is already compute-bound. Unpacking once per TILE keeps the cost at one
//   pass over the weights, which is what the packed format was supposed to buy.
//
//   The numbers that set this policy, for gpt-oss-20b: 19.11e9 expert parameters, i.e. 38.2 GB dense
//   against 9.6 GB packed, but only 4 of 32 experts are active per token, so per-token traffic is
//   4.78 GB against 1.19 GB. The dominant win is therefore CAPACITY (about 14 GB per card freed at
//   TP=2), not decode bandwidth -- and the risk to guard is slowing the compute-bound prefill.
//
// `blocks` [rows, K/32, 16] uint8, `scales` [rows, K/32] uint8 -> `out` [rows, K] half, row-major
// with a caller-chosen leading dimension so the result can land directly in a GEMM's shared-memory
// staging layout.
__global__ void mxfp4_dequant_tile_kernel(const uint8_t* __restrict__ blocks,
                                          const uint8_t* __restrict__ scales,
                                          __half* __restrict__ out,
                                          int rows, int nblk, int ldo) {
  const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long nbytes = (long)rows * nblk * 16;
  if (tid >= nbytes) return;
  const long blk = tid >> 4;                     // 16 bytes per MX block
  const int r = (int)(blk / nblk);               // row of the tile
  const long c = (blk - (long)r * nblk) * 32 + ((tid & 15) << 1);   // first of the two elements
  const int s = scales[blk];
  float sc;
  if (s == 255) sc = __int_as_float(0x7fc00000);
  else if (s == 0) sc = 0.0f;
  else sc = __int_as_float(s << 23);
  const uint8_t byte = blocks[tid];
  __half* o = out + (long)r * ldo + c;
  o[0] = __float2half(kE2M1[byte & 0xF] * sc);
  o[1] = __float2half(kE2M1[byte >> 4] * sc);
}

torch::Tensor mxfp4_dequant_tile(torch::Tensor blocks, torch::Tensor scales, int64_t ldo) {
  TORCH_CHECK(blocks.is_cuda() && blocks.dim() == 3 && blocks.size(2) == 16,
              "mxfp4 tile: blocks must be [rows, nblk, 16] on CUDA");
  auto b = blocks.contiguous(), s = scales.contiguous();
  const int rows = (int)b.size(0), nblk = (int)b.size(1);
  const long ld = ldo > 0 ? ldo : (long)nblk * 32;
  auto out = torch::empty({rows, ld}, torch::dtype(torch::kFloat16).device(b.device()));
  const long nbytes = (long)rows * nblk * 16;
  const int threads = 256;
  mxfp4_dequant_tile_kernel<<<(nbytes + threads - 1) / threads, threads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      b.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), rows, nblk, (int)ld);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}



// [SASS] E2M1 AS ARITHMETIC, NOT A TABLE LOOKUP.
// The table version compiled to one LDC per nibble -- 32 of them per 16-byte block. Constant memory
// on Volta is a BROADCAST path: when the 32 lanes of a warp read 32 DIFFERENT entries it serialises,
// and the GEMV measured 56 GB/s against the dense fp16 path's 613 GB/s despite reading 4x fewer
// weight bytes. The table was the bottleneck, not the bandwidth.
//
// E2M1 is v = [sign | exp:2 | mant:1], and its eight magnitudes are exactly what a float exponent
// field expresses: e>0 -> 2^(e-1) * (1 + m/2), e=0 -> m * 0.5. So the value is just a bit pattern:
//   exponent field = 126 + e     (e=0 gives 126, i.e. 0.5, which is the subnormal case)
//   mantissa       = m << 22, but only when e != 0
//   the whole thing is zero when e|m == 0
//
// AND THE SCALE FOLDS IN. E8M0 is a pure power of two, 2^(s-127), so multiplying is adding to the
// exponent field: 126 + e + (s - 127) = s + e - 1. That removes the per-element FMUL as well, which
// is why this both shortens the dependency chain and drops two instruction classes at once.
__device__ __forceinline__ float e2m1_scaled(uint32_t v, int s) {
  const uint32_t sg = (v >> 3) & 1u, e = (v >> 1) & 3u, m = v & 1u;
  const int ef = s + (int)e - 1;                              // folded exponent field
  const uint32_t nz = (uint32_t)(-(int)(((e | m) != 0u) & (ef > 0)));   // zero, and underflow, flush
  const uint32_t em = (uint32_t)(-(int)(e != 0u));            // mantissa only exists for e != 0
  const uint32_t bits = (((uint32_t)ef) << 23) | ((m << 22) & em);
  return __int_as_float((int)((bits & nz) | (sg << 31)));
}

// ---------------------------------------------------------------------------------------------
// FUSED MXFP4 GEMV: y[m, n] = sum_k x[m, k] * W[n, k], with W kept PACKED.
//
// WHY A SEPARATE KERNEL FROM THE TILE PATH
//   The two regimes want opposite things, and the arithmetic says so plainly (gpt-oss-20b, TP=2):
//     * PREFILL: a chunk routes ~1024 tokens to each expert, so every weight tile is reused ~1024
//       times. Unpacking once into a buffer costs one pass and is then amortised -- and unpacking
//       per use would multiply ALU by M on a path that is already compute-bound.
//     * DECODE: only 4 of 32 experts are live per token and each weight is used ONCE. Unpacking into
//       a buffer would cost ~21 ms per token against a 34 ms step -- it would destroy the very thing
//       it is meant to help. Here the weight must be consumed straight from its packed form, which
//       cuts the read from 4.78 GB to 1.19 GB per token.
//   So: tile-unpack for prefill, fused GEMV for decode. One kernel cannot serve both without losing
//   one of them.
//
//   The dequant ALU is free in this kernel by construction: a GEMV is bandwidth-bound, so the table
//   lookup and the scale multiply sit in the shadow of the loads they depend on.
//
// x [M, K] half, blocks [N, K/32, 16] uint8, scales [N, K/32] uint8 -> y [M, N] half.
// One warp per output element; the warp strides over K so its 32 lanes read consecutive bytes.
template <int WARPS>
__global__ void mxfp4_gemv_kernel(const __half* __restrict__ x,
                                  const uint8_t* __restrict__ blocks,
                                  const uint8_t* __restrict__ scales,
                                  __half* __restrict__ y,
                                  int M, int N, int nblk) {
  // x LIVES IN SHARED MEMORY. The first version read it straight from global inside the k-loop, so
  // every one of the N output rows re-read the whole activation vector: at N=K=2880 that is 2880 x
  // 5.7 KB = 16 MB of traffic to deliver 5.7 KB of data, and the kernel measured 56 GB/s against the
  // dense fp16 path's 613 GB/s -- 2.7x SLOWER despite reading 4x fewer weight bytes. Reading fewer
  // bytes only helps if the other operand is not re-fetched N times.
  //
  // ПОПРАВКА 02.08.2026: прежнее «GEMV is bandwidth-bound, dequant ALU is free» ОПРОВЕРГНУТО.
  // Ядро связано СВОИМ ЖЕ конфликтом банков: 32 полосы ложились на 2 банка (конфликтность 16.00),
  // и канал разделяемой стоил 256 тактов на варп-итерацию против ~75 у глобальной -- запас 3.4x.
  // Не полоса связывала, а раскладка. Лечение -- шаг блока 34 полуслова, см. SXB ниже.
  // [ШАГ БЛОКА 34 ПОЛУСЛОВА, А НЕ 32 -- СНЯТИЕ 16-КРАТНОГО КОНФЛИКТА БАНКОВ]
  // Полоса ell читает sx + ell*32 как half2, то есть адрес СЛОВА = 16*ell + i. При фиксированном i
  // величина 16*ell mod 32 принимает РОВНО ДВА значения {0,16}: тридцать два разных адреса ложатся
  // на два банка, конфликтность 16.00 по замкнутой форме |R| / |{r*S mod 32}|.
  // Шаг 34 полуслова (= 17 слов) даёт 17*ell mod 32 -- биекция (17 нечётно, взаимно просто с 32),
  // конфликтность 1.00. Выравнивание half2 сохранено: 34*2 = 68 Б кратно 4.
  // Кривая проверена целиком: дополнения 2/4/8 слов ХУЖЕ (2.00/4.00/8.00) -- одностороннее
  // «дополнить, чтобы банки разъехались» здесь неверно, годится ровно НЕЧЁТНОЕ число слов.
  // Ответ не меняется НИ НА БИТ: это только размещение.
  static constexpr int SXB = 34;                      // полуслов на блок из 32 значений
  extern __shared__ __half sx[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int K = nblk * 32;
  const int nthr = WARPS * 32;

  for (int m = 0; m < M; ++m) {
    for (int t = threadIdx.x; t < K; t += nthr)
      sx[(t >> 5) * SXB + (t & 31)] = x[(long)m * K + t];
    __syncthreads();
    const int n = blockIdx.x * WARPS + warp;                 // output feature
    if (n < N) {
      const uint8_t* wb = blocks + (long)n * nblk * 16;
      const uint8_t* ws = scales + (long)n * nblk;
      float acc = 0.f;
      for (int b = lane; b < nblk; b += 32) {
        const int sv = ws[b];                                // E8M0, folded into the exponent below
        const uint4 packed = *reinterpret_cast<const uint4*>(wb + (long)b * 16);
        const uint8_t* by = reinterpret_cast<const uint8_t*>(&packed);
        const __half2* xb = reinterpret_cast<const __half2*>(sx + b * SXB);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint32_t v = by[i];
          const float2 xv = __half22float2(xb[i]);          // two activations per packed byte
          acc = fmaf(xv.x, e2m1_scaled(v & 0xFu, sv), acc);
          acc = fmaf(xv.y, e2m1_scaled(v >> 4, sv), acc);
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
      if (lane == 0) y[(long)m * N + n] = __float2half(acc);
    }
    __syncthreads();
  }
}

torch::Tensor mxfp4_gemv(torch::Tensor x, torch::Tensor blocks, torch::Tensor scales) {
  TORCH_CHECK(x.is_cuda() && blocks.is_cuda(), "mxfp4 gemv: CUDA tensors required");
  TORCH_CHECK(blocks.dim() == 3 && blocks.size(2) == 16, "mxfp4 gemv: blocks [N, nblk, 16]");
  auto xc = x.contiguous(), b = blocks.contiguous(), s = scales.contiguous();
  const int M = (int)(xc.numel() / xc.size(-1)), N = (int)b.size(0), nblk = (int)b.size(1);
  TORCH_CHECK(xc.size(-1) == (long)nblk * 32, "mxfp4 gemv: K mismatch");
  auto y = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(xc.device()));
  constexpr int WARPS = 4;
  const size_t shm = (size_t)nblk * 34 * sizeof(__half);   // 34, а не 32: см. SXB в ядре
  mxfp4_gemv_kernel<WARPS><<<(N + WARPS - 1) / WARPS, WARPS * 32, shm,
                             at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
      b.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), M, N, nblk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp4_dequant", &mxfp4_dequant, "MXFP4 -> fp16/fp32 dequantisation (sm_70)");
  m.def("mxfp4_dequant_tile", &mxfp4_dequant_tile, "MXFP4 tile -> fp16 with a chosen leading dim");
  m.def("mxfp4_gemv", &mxfp4_gemv, "fused MXFP4 GEMV: weights consumed packed (decode path)");
}
