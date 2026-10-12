// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Flash-Decoding (split-KV) for sm70, with a kv_dtype flag: fp16 (EXACT), int8, int4 KV cache.
// Decode is HBM-bandwidth-bound, so reading a narrower KV cache is the ONLY way past the exact
// bandwidth ceiling: int8 ~2x, int4 ~4x fewer bytes read. Per-token symmetric
// quantization (one scale per key) -- the propagation identity says a per-key scale is free and
// only the key-to-key error DISPERSION matters. int4 additionally uses the exact top-1 control
// point as an inexpensive correction: the split tracks its argmax key; the reducer re-reads that
// one pair EXACTLY (fp16) and swaps it in, removing the leading Cov_pi(delta,v) term.
#ifndef FA2SM70_DEC_PF
#define FA2SM70_DEC_PF 0
#endif
#ifndef FA2SM70_DEC_BLK
// SM КАК МИКРОКОНТРОЛЛЕР, А НЕ КАК СЕТКА БЛОКОВ.
// Обрыв "168 регистров = 3 блока/SM", об который расшибся конвейер (см. EV_decode_pipeline_pf_uu),
// НЕ свойство машины: он следствие того, что мы сами нарезали SM на три тощих блока.
// Тот же регистровый файл иначе: 3 x 128 x 168 = 64512 против 1 x 256 x 255 = 65280 -- сумма та же,
// но контекст ОДИН, ему достаётся 255 регистров вместо 168 и вся разделяемая (96 КБ) вместо 24.
// Плата -- 8 варпов вместо 12, то есть меньше АППАРАТНОГО сокрытия задержки; его заменяет явный
// конвейер (роли варпов: подвозчики в скретчпад / счётчики).
#define FA2SM70_DEC_BLK 128
#endif
#ifndef FA2SM70_DEC_SAH
// ПЛОЩАДКА СВЕДЕНИЯ В half. Сейчас sA -- fp32, хотя ВЫШЕ fp16 у нас нигде ничего нет: чистая
// инерция формата. Она же 99 % разделяемой блока (24576 из 24768 Б) и крупнейший байтовый хозяин
// шага. СЦЕПЛЕНО С BLK: при BLK=256 версия fp32 даёт 49152 Б -- ровно потолок СТАТИЧЕСКОЙ
// разделяемой на Volta, тело не собирается. То есть жирный блок ОБЯЗЫВАЕТ снять формат.
#define FA2SM70_DEC_SAH 0
#endif
#ifndef FA2SM70_DEC_QW
// ШИРОКОЕ ЧТЕНИЕ Q В ПРОЛОГЕ. Замерено в машинном коде боевого тела <EPT=8,GF=6>: пролог
// содержит GF*EPT = 48 обращений LDG.E.U16, то есть восемь СМЕЖНЫХ half берутся ПОЭЛЕМЕНТНО.
// Те же 96 байт укладываются в 6 обращений LDG.E.128 -- то же число БАЙТ при ВОСЬМИКРАТНО
// меньшем числе СОБЫТИЙ перевозки (мера у нас -- событие, а не байт). Компилятор не
// сворачивает сам: half* обещает выравнивание 2 байта, а load_bytes заявляет 16 явно.
// Выключено по умолчанию: включение -- решение, иначе A/B бессмыслен.
#define FA2SM70_DEC_QW 0
#endif
#ifndef FA2SM70_DEC_VK
// Issue V together with K instead of in a later wave (the change that doubled outstanding loads per
// warp). Default on; the switch exists so the A/B can be reproduced on a live server, where the
// answer can differ from a microbenchmark -- and did.
#define FA2SM70_DEC_VK 1
#endif
#ifndef FA2SM70_DEC_UU
// Unroll of the KV token loop: how many tokens a warp has in flight at once. It sets BOTH the
// outstanding-load count and the register cost, so it cannot be tuned independently of
// FA2SM70_DEC_PF -- the two were swept together.
//
// MEASURED (int8/e4m3 paged pool, B=1, H=32, Hkv=8, d=128, us per decode step):
//    UU     1K     4K     8K    16K    32K    64K   128K   256K
//     2   22.9   55.2   91.9  150.9  257.6  493.2  947.3 1845.7
//     3   23.2   57.2   99.7  146.6  244.4  451.2  876.9 1683.4   <- best on LONG only
//     4   22.4   53.7   92.8  142.9  239.8  457.9  887.3 1732.8   <- best COMPROMISE, default
//     6   25.1   68.1  103.4  167.2  285.0  531.3 1036.6 2049.2
//     8   23.8   70.3  103.0  163.7  282.9  517.6 1023.7 1994.1
//
// 3 is the fastest at 64K-256K but pays -6.5% at 4K and -7.4% at 8K; 4 is within 3% of it there and
// loses nowhere except 8K (-1%), so 4 is the default. Past 4 the register cost overtakes the ILP and
// everything regresses -- the optimum is a peak, not a slope.
//
// THE OPTIMUM IS SHAPE-DEPENDENT. Those numbers are one geometry; GF (fused query heads per KV head)
// and d change the register budget, so a different model may peak elsewhere. That is why this stays
// a switch: FA2SM70_DEC_UU=<n> at build time.
#define FA2SM70_DEC_UU 0        // 0 = derive from EPT (see below)
#endif
#include <cstdlib>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <math_constants.h>
static constexpr int BLK = FA2SM70_DEC_BLK, NW = BLK / 32;
#include "decode_hmma.cuh"

// load this lane's EPT elements of key j as RAW floats (no per-token scale); KVB in {16,8,4}.
// Vectorized: one 32/64-bit transaction per lane. The per-token scale is applied ONCE per key by
// the caller (folded into the QK dot for K, into the softmax weight p for V), saving EPT-1 muls/key.
// OCP fp8 e4m3 (1 sign, 4 exp bias 7, 3 mantissa) -> float, EXACTLY. Volta has no fp8 hardware, but
// e4m3 is a strict subset of fp16's range, so the conversion is pure bit surgery: re-bias the exponent
// (+8) and left-shift the mantissa (<<7). Denormals (exp==0, mant!=0) are renormalised explicitly.
// This lets our paged decode read vLLM's own fp8_e4m3 KV pool in place -- no repacking, no side copy.
__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
    uint16_t s = (uint16_t)(b & 0x80) << 8;
    int e = (b >> 3) & 0xF, m = b & 0x7;
    uint16_t h;
    if (e == 0) {
        if (m == 0) h = s;                             // +/-0
        else {                                         // denormal: m * 2^-9
            int sh = (m >= 4) ? 0 : ((m >= 2) ? 1 : 2);
            h = s | (uint16_t)((15 - 6 - sh) << 10) | (uint16_t)((((m << (sh + 1)) & 0x7) << 7));
        }
    } else {
        h = s | (uint16_t)((e + 8) << 10) | (uint16_t)(m << 7);
    }
    return __half2float(__ushort_as_half(h));
}

// [FAST e4m3] Branch-free OCP-e4m3 -> float. The bit-picking version above costs ~12 ALU ops with a
// denormal BRANCH, and ncu says that is where this kernel lives: 58.9M ALU instructions against 9.9M
// memory instructions at ctx=131072, and the #1 stall is `wait` (fixed-latency execution dependency,
// 1.31/issue) sitting on the load -> convert -> FMA chain. 16 conversions per lane per token (8 K + 8 V)
// x ~12 ops x 262144 token-visits = 50.3M -- i.e. ~85% of all ALU work is this one function.
//
// The trick: fp16 is E5M10 with bias 15, e4m3 is E4M3 with bias 7. Drop the 7 bits EEEEMMM at fp16 bit 7
// and the fields line up as exponent=EEEE (top exponent bit 0), mantissa=MMM<<7, giving 2^(E-15)*1.MMM.
// One multiply by 2^8 fixes the bias to 2^(E-7)*1.MMM. DENORMALS NEED NO SPECIAL CASE: e4m3's E=0 maps to
// fp16's E=0, and an fp16 denormal is 0.MMM * 2^-14, which the same 2^8 turns into 0.MMM * 2^-6 -- exactly
// the e4m3 denormal. Zero maps to zero. The only divergence from the reference is the 0x7F/0xFF NaN
// encoding, which becomes +/-480: our own store never emits it (float_to_e4m3 clamps to 0x7E and sends
// NaN to 0), so no value the pool can hold is affected. Verified exhaustively over all 256 bytes.
__device__ __forceinline__ float e4m3_to_float_fast(uint8_t b) {
    const uint16_t h = (uint16_t)((uint16_t)(b & 0x80) << 8) | (uint16_t)((uint16_t)(b & 0x7F) << 7);
    return __half2float(__ushort_as_half(h)) * 256.0f;
}

// float -> OCP fp8 e4m3, round-to-nearest. Inverse of e4m3_to_float; pure bit work, no fp8 hardware.
__device__ __forceinline__ uint8_t float_to_e4m3(float f) {
    uint32_t x = __float_as_uint(f);
    uint8_t s = (uint8_t)((x >> 24) & 0x80);
    float a = fabsf(f);
    if (!(a > 0.f)) return s;                                  // +/-0 (and NaN -> 0)
    if (a >= 448.f) return (uint8_t)(s | 0x7E);                // clamp to e4m3's largest finite
    int e; float m = frexpf(a, &e);                            // a = m*2^e, m in [0.5,1)
    int E = e - 1 + 7;                                         // e4m3 bias 7
    if (E <= 0) {                                              // denormal: value = M/8 * 2^-6
        int M = __float2int_rn(a * 512.f);                     // a / 2^-6 * 8
        if (M > 7) return (uint8_t)(s | (1 << 3));             // rounded up into the smallest normal
        return (uint8_t)(s | M);
    }
    int M = __float2int_rn((2.f * m - 1.f) * 8.f);
    if (M == 8) { M = 0; ++E; }
    if (E > 15) return (uint8_t)(s | 0x7E);
    return (uint8_t)(s | (E << 3) | M);
}

// [FAST float->e4m3] Mirror of e4m3_to_float_fast, and for the same reason: the reference version below
// calls frexpf and branches on the denormal case (~15-20 ops), and it runs on EVERY stored KV element --
// the whole prompt during prefill, every layer. Measured before this: the store sustains 107-131 GB/s
// against the ~820 GB/s this card gives on a plain read, the same ALU-bound signature the READ side had.
//
// Route the value through fp16, which Volta converts in ONE hardware instruction (F2F.F16.F32): scaling by
// 2^-8 lines e4m3's E4M3/bias-7 field up with fp16's E5M10/bias-15, so the e4m3 byte is just bits [13:7]
// of the fp16 pattern -- exactly the inverse of the read trick, denormals included (an fp16 denormal
// 0.MMM*2^-14 IS the e4m3 denormal 0.MMM*2^-6 after the scale). Rounding is round-to-nearest-EVEN taken on
// bit 6 with a sticky OR of the low bits, matching what the reference computes with __float2int_rn.
//
// Two deliberate deviations, both pre-existing policy rather than new behaviour:
//   * |x| >= 448 SATURATES to 0x7E instead of becoming NaN as OCP/torch would. That saturation is what
//     keeps a single overflow from poisoning the running ||k|| bound (see the NaN-latch fix), so it stays.
//   * NaN maps to 0, likewise pre-existing.
// Double rounding is a non-issue in the hot path: the kernel's input is already `half`, so with the usual
// k_scale/v_scale of 1 the float->half step is exact and there is only one rounding.
__device__ __forceinline__ uint8_t float_to_e4m3_fast(float f) {
    const uint8_t sgn = (uint8_t)((__float_as_uint(f) >> 24) & 0x80);
    const float a = fabsf(f);
    if (!(a > 0.f)) return sgn;                                    // +/-0, and NaN -> 0 (policy)
    if (a >= 448.f) return (uint8_t)(sgn | 0x7E);                  // saturate (policy)
    // DENORMALS EXACTLY, and not through fp16. e4m3's denormals live below 2^-6; scaled by 2^-8 they land
    // in fp16's OWN denormal range, where fp16 no longer has the bits to hold them, so the scale itself
    // rounds and the result is double-rounded. Measured on the first version: 0.08% of elements off by one
    // step (0.00098 -> 0 instead of 0.001953, 0.00683 -> 0.0078 instead of 0.0059). A denormal is just
    // M*2^-9, so round M directly -- one compare, taken by a vanishing fraction of real KV values.
    if (a < 0.015625f) {                                           // < 2^-6 = smallest e4m3 normal
        const int M = __float2int_rn(a * 512.0f);                  // a / 2^-9, round to nearest
        return (uint8_t)(sgn | (uint8_t)(M > 7 ? 8 : M));          // M==8 rounds up into the smallest normal
    }
    const uint16_t h = __half_as_ushort(__float2half_rn(a * (1.0f / 256.0f)));
    const uint16_t lo = (uint16_t)(h & 0x7F);                      // bits below the e4m3 mantissa
    uint16_t t = (uint16_t)(h >> 7);                               // the 7 bits EEEEMMM
    if (lo > 0x40 || (lo == 0x40 && (t & 1))) ++t;                 // round to nearest, ties to even
    if (t > 0x7E) t = 0x7E;                                        // rounding can only reach 448 from below
    return (uint8_t)(sgn | (uint8_t)t);
}

// [e5m2] The other OCP fp8 format, and on this hardware it is the EASY one: e5m2 is E5M2 with bias 15,
// which is fp16's exponent width and fp16's bias. So the byte is literally the top half of an fp16 --
// conversion is a shift, with no scaling multiply of the kind e4m3 needs.
// Everything falls out exactly, in BOTH directions, with no special cases:
//   * normals      S EEEEE MM  ->  S EEEEE MM00000000, same exponent, same bias;
//   * denormals    0.MM*2^-14  ->  0.MM00000000*2^-14, the identical value (fp16 denormals work the same);
//   * inf and NaN  occupy the same encodings in both formats, so they map through untouched.
// Contrast e4m3, whose write direction needs an explicit denormal branch because its denormals land in
// fp16's denormal range after the 2^-8 scale and get double-rounded. e5m2 has no such hazard.
__device__ __forceinline__ float e5m2_to_float(uint8_t b) {
    return __half2float(__ushort_as_half((uint16_t)b << 8));
}

// Round-to-nearest-EVEN on the 8 discarded mantissa bits. Saturating overflow into inf is the correct
// e5m2 behaviour (unlike e4m3, e5m2 HAS an infinity), so no clamp is needed; NaN stays NaN.
__device__ __forceinline__ uint8_t float_to_e5m2(float f) {
    const uint16_t u = __half_as_ushort(__float2half_rn(f));
    const uint16_t lo = (uint16_t)(u & 0xFF);
    uint16_t hi = (uint16_t)(u >> 8);
    if (lo > 0x80 || (lo == 0x80 && (hi & 1))) ++hi;          // ties to even
    return (uint8_t)(hi & 0xFF);
}

__device__ __forceinline__ void atomic_max_pos(float* addr, float v) {
    // v >= 0: the IEEE bit pattern of a non-negative float is monotone, so an integer max works.
    // Defence in depth for that same monotonicity: inf/NaN outrank every finite value, so a single
    // non-finite v would latch the running max forever. Callers must already keep v finite; drop it here
    // regardless, because the failure mode is a silent permanent outage rather than a wrong number.
    if (!isfinite(v)) return;
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

// [serving] OUR e4m3 KV store: quantise + write into the engine's (possibly strided) byte pool AND fold
// the new tokens into the running ||k_raw|| bound IN THE SAME KERNEL. Doing the bound with torch ops cost
// four extra kernels per layer per token; here it is free (the row is already in registers).
// [ДЕРЕВО КАНДИДАТОВ, 06.09] МАСКА МОЖЕТ ВЫЧЕРКНУТЬ ВЕСЬ СРЕЗ РАСЩЕПЛЕНИЯ.
// Онлайн-софтмакс копил exp(part - m). Пока маска была ПРЕФИКСНОЙ (цепь), длина
// строки резала ровно там же, где маска, и вычеркнутый ключ в цикл не попадал.
// У ДЕРЕВА маска дырявая: строка b0 читает ключи a0,a1 (они внутри её длины), но
// НЕ видит их. Если срез расщепления начинается с таких ключей, то part=-inf И
// m=-inf, и exp(-inf - (-inf)) = exp(NaN) = NaN -- Bsum и acc портятся навсегда, а
// за ними весь слой. Замер: строки b выходили ЦЕЛИКОМ NaN (248320 из 248320
// логитов) на первом же шаге дерева, строки a были чисты; с FA2SM70_TREE_NOMASK=1
// (без маски) NaN исчезал и ответ был чист. Вычеркнутый ключ обязан давать РОВНО
// ноль -- это и есть его вес в софтмаксе.
#define FA2SM70_EXP_MASKED(p_, m_) (((p_) == -CUDART_INF_F) ? 0.f : __expf((p_) - (m_)))

__global__ void reshape_and_cache_e4m3_kernel(
    const half* __restrict__ key, const half* __restrict__ value,
    uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const long* __restrict__ slot, int n, int Hkv, int d,
    long blk_stride, int block_size, float inv_ks, float inv_vs,
    float* __restrict__ kbound, float margin) {
    const int t = blockIdx.x, h = blockIdx.y;                  // token, kv-head
    if (t >= n) return;
    const long s = slot[t];
    if (s < 0) return;                                         // padded slot
    const long pb = s / block_size, wb = s - pb * block_size;
    const long dst = pb * blk_stride + ((long)wb * Hkv + h) * d;
    const long src = ((long)t * Hkv + h) * d;
    float acc = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float kf = __half2float(key[src + i]) * inv_ks;  // raw (pre-scale) units
        const float vf = __half2float(value[src + i]) * inv_vs;
        const uint8_t kq = float_to_e4m3_fast(kf);
        kc[dst + i] = kq;
        vc[dst + i] = float_to_e4m3_fast(vf);
        // [NaN-LATCH FIX] Accumulate the norm over the value the pool ACTUALLY now holds, not over the
        // pre-quantisation input. float_to_e4m3 maps NaN->0 and |x|>=448->448, so the bound stays a true
        // upper bound of the stored pool AND is structurally finite (<= 448*sqrt(d)).
        //
        // Why this matters: kbound is a pool-wide RUNNING MAX, folded in by atomic_max_pos, which compares
        // IEEE bit patterns as integers. inf (0x7F800000) and NaN (0x7FC00000) outrank every finite float,
        // so ONE non-finite k element -- a single fp16 overflow in the K projection, anywhere, ever --
        // latched the bound permanently. Decode uses the bound as the certified U in exp(score - U):
        // U=NaN made EVERY later decode return NaN (prefill was unaffected, it never reads the bound), and
        // U=inf zeroed it. That was a permanent, server-wide decode outage curable only by a restart.
        const float ks = e4m3_to_float(kq);
        acc += ks * ks;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __shared__ float warpsum[32];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) warpsum[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int i = 0; i < (blockDim.x + 31) / 32; ++i) tot += warpsum[i];
        atomic_max_pos(kbound + h, sqrtf(tot) * margin);
    }
}

// [serving] НАШ int8-СТОР В ТОТ ЖЕ БАЙТОВЫЙ ПУЛ: квантование симметричное, масштаб НА ПОЗИЦИЮ.
// Близнец reshape_and_cache_e4m3_kernel, и отличий ровно три, все принципиальные:
//   * байт значит int8, а не e4m3: 7 бит мантиссы на строку вместо 3 при широком порядке;
//   * масштаб идёт не скаляром на слой, а в ПОБОЧНУЮ ТАБЛИЦУ по слоту [блоки, размер_блока, Hkv] --
//     ровно ту раскладку, которую обходит s_off() в ядре декода, то есть по ТОМУ ЖЕ block_table;
//   * k_scale/v_scale движка тут НЕ УЧАСТВУЮТ: масштаб на позицию уже несёт истинные единицы fp16
//     (значение = q*s). Скалярный масштаб на слой существовал только чтобы вписать значения в узкий
//     диапазон e4m3; per-position делает это точнее и локально.
//
// ЖИЗНЕННЫЙ ЦИКЛ ТАБЛИЦЫ СОВПАДАЕТ С ЖИЗНЕННЫМ ЦИКЛОМ БАЙТОВ ПО ПОСТРОЕНИЮ: масштаб слота пишется
// ТЕМ ЖЕ вызовом, что и его d байт, поэтому вытеснение и переиспользование блока не могут развести
// их во времени (в отличие от границы ||k|| по блокам, которая только монотонно растёт).
//
// НЕ-КОНЕЧНОЕ ГАСИТСЯ, А НЕ РАСПРОСТРАНЯЕТСЯ. Один inf в строке K дал бы absmax=inf -> s=inf -> вся
// строка в ноль, а NaN -- неопределённое приведение к int8. Урок уже оплачен на границе ||k||
// (единственный inf защёлкивал её навсегда и глушил декод до перезапуска), поэтому не-конечное
// значение заменяется нулём ДО свёртки максимума: портится одно число, а не строка и не сервер.
__global__ void reshape_and_cache_i8_kernel(
    const half* __restrict__ key, const half* __restrict__ value,
    int8_t* __restrict__ kc, int8_t* __restrict__ vc,
    const long* __restrict__ slot, int n, int Hkv, int d,
    long blk_stride, int block_size,
    float* __restrict__ kst, float* __restrict__ vst) {
    const int t = blockIdx.x, h = blockIdx.y;                  // token, kv-head (both block-uniform)
    if (t >= n) return;
    const long s = slot[t];
    if (s < 0) return;                                         // padded slot (profiling / graph capture)
    const long pb = s / block_size, wb = s - pb * block_size;
    const long dst = pb * blk_stride + ((long)wb * Hkv + h) * d;
    const long src = ((long)t * Hkv + h) * d;
    const long sidx = (pb * (long)block_size + wb) * Hkv + h;  // побочная таблица -- ПЛОТНАЯ
    // Строка целиком в регистрах: d <= 4*blockDim (256 нитей -> d <= 1024), поэтому второго ЧТЕНИЯ
    // из памяти для перекантовки не будет -- максимум и квантование идут по уже прочитанному.
    constexpr int EMAX = 4;
    float kr[EMAX], vr[EMAX];
    float ka = 0.f, va = 0.f;
    #pragma unroll
    for (int e = 0; e < EMAX; ++e) {
        const int i = threadIdx.x + e * blockDim.x;
        kr[e] = 0.f; vr[e] = 0.f;
        if (i < d) {
            float kf = __half2float(key[src + i]), vf = __half2float(value[src + i]);
            if (!isfinite(kf)) kf = 0.f;                       // см. «не-конечное гасится» выше
            if (!isfinite(vf)) vf = 0.f;
            kr[e] = kf; vr[e] = vf;
            ka = fmaxf(ka, fabsf(kf)); va = fmaxf(va, fabsf(vf));
        }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        ka = fmaxf(ka, __shfl_xor_sync(0xffffffffu, ka, o));
        va = fmaxf(va, __shfl_xor_sync(0xffffffffu, va, o));
    }
    __shared__ float sm[2][32];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) { sm[0][wid] = ka; sm[1][wid] = va; }
    __syncthreads();
    if (wid == 0) {
        float a = lane < nw ? sm[0][lane] : 0.f, b = lane < nw ? sm[1][lane] : 0.f;
        #pragma unroll
        for (int o = 16; o; o >>= 1) {
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            b = fmaxf(b, __shfl_xor_sync(0xffffffffu, b, o));
        }
        if (lane == 0) { sm[0][0] = a; sm[1][0] = b; }
    }
    __syncthreads();
    // absmax/127, симметрично, к ближайшему -- ТО ЖЕ правило, что у сборщика gather_paged_kv_i8 и у
    // эталона в тестах. Пол в наименьшее нормальное fp16 убирает деление на ноль на нулевой строке.
    const float ks = fmaxf(sm[0][0], 6.1e-5f) / 127.f, vs = fmaxf(sm[1][0], 6.1e-5f) / 127.f;
    if (threadIdx.x == 0) { kst[sidx] = ks; vst[sidx] = vs; }
    // ДЕЛЕНИЕ ЯВНО ТОЧНОЕ (__fdiv_rn): ядро собирается с --use_fast_math, где `/` становится
    // приближённым обратным и уводит результат на ULP -- ровно столько, чтобы перевернуть округление
    // на границе .5 и дать байт, отличный от эталона. Замерено на сборщике, цена нулевая.
    #pragma unroll
    for (int e = 0; e < EMAX; ++e) {
        const int i = threadIdx.x + e * blockDim.x;
        if (i < d) {
            kc[dst + i] = (int8_t)fmaxf(-127.f, fminf(127.f, rintf(__fdiv_rn(kr[e], ks))));
            vc[dst + i] = (int8_t)fmaxf(-127.f, fminf(127.f, rintf(__fdiv_rn(vr[e], vs))));
        }
    }
}

// [serving] Paged -> contiguous gather for the PREFIX/CHUNKED prefill path.
// vLLM's own gather has a fast CUDA path but guards it with `key_cache.dtype != uint8`, and an
// fp8_e4m3 pool IS uint8 -- so with 8-bit KV it silently takes a Python double loop over
// (sequence, block) with a .item() sync per block. At 145K tokens that is ~9000 synchronised
// iterations per layer, which is what collapses long prefill; the attention math there is already
// ours. Emitting straight to [Hkv, T, d] (BHSD) also saves the transpose our kernel would otherwise
// pay for, the same lever that won twice before on this project.
template <int KVB>
__global__ void gather_paged_kv_kernel(
    const void* __restrict__ kc, const void* __restrict__ vc,
    const int* __restrict__ block_table,
    half* __restrict__ ok, half* __restrict__ ov,
    int T, int Hkv, int d, long blk_stride, int block_size, float k_scale, float v_scale) {
    const int t = blockIdx.x, h = blockIdx.y;
    if (t >= T) return;
    const int lb = t / block_size;
    const long src = (long)block_table[lb] * blk_stride + ((long)(t - lb * block_size) * Hkv + h) * d;
    const long dst = ((long)h * T + t) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float kf, vf;
        if (KVB == 87) {                                       // fp8_e4m3 byte pool -> fp16
            kf = e4m3_to_float_fast(reinterpret_cast<const uint8_t*>(kc)[src + i]) * k_scale;
            vf = e4m3_to_float_fast(reinterpret_cast<const uint8_t*>(vc)[src + i]) * v_scale;
        } else {                                               // fp16 pool: straight copy
            kf = __half2float(reinterpret_cast<const half*>(kc)[src + i]);
            vf = __half2float(reinterpret_cast<const half*>(vc)[src + i]);
        }
        ok[dst + i] = __float2half(kf);
        ov[dst + i] = __float2half(vf);
    }
}

// СБОРКА СРАЗУ В БАЙТЫ (для ядра ролей). Отличие от сборщика выше -- ОДНО, и оно принципиальное:
// собранная плитка НЕ разворачивается в fp16, а перекантовывается в int8 с масштабом НА ПОЗИЦИЮ,
// то есть в тот самый формат, который attn_fwd_volta_i8 подаёт на тензорные ядра напрямую.
// Выход вдвое уже (d байт против 2d), и деквантование из ядра внимания исчезает совсем.
//
// ПОЧЕМУ ЭТО ЗАКОННО ПО ТОЧНОСТИ: пул хранит e4m3 -- 3 бита мантиссы при широком порядке. int8 с
// масштабом по строке даёт 7 бит на строку с умеренным размахом, что для K/V-строк внимания и есть
// типичный случай. Перекантовка добавляет свою ошибку поверх e4m3, и она ЗАМЕРЕНА (см. тест
// tests/test_vllm_i8_gather.py), а не предположена: округление здесь -- к ближайшему, не усечение.
//
// Один блок = одна строка (позиция, голова): d значений. Свёртка максимума -- по варпам через shfl,
// затем по варпам через разделяемую; d <= 256, поэтому хватает восьми варпов.
template <int KVB>
__global__ void gather_paged_kv_i8_kernel(
    const void* __restrict__ kc, const void* __restrict__ vc,
    const int* __restrict__ block_table,
    int8_t* __restrict__ okq, float* __restrict__ oks,
    int8_t* __restrict__ ovq, float* __restrict__ ovs,
    int T, int Hkv, int d, long blk_stride, int block_size, float k_scale, float v_scale) {
    extern __shared__ float smax[];                    // [2][32] -- по варпу на каждый из K и V
    const int t = blockIdx.x, h = blockIdx.y;
    if (t >= T) return;
    const int lb = t / block_size;
    const long src = (long)block_table[lb] * blk_stride + ((long)(t - lb * block_size) * Hkv + h) * d;
    const long dst = ((long)h * T + t) * d;

    // Первый проход: читаем строку в регистры и берём максимум модуля. d <= 256 при 256 нитях --
    // не больше одного значения на нить, поэтому строка целиком помещается в регистры и второго
    // ЧТЕНИЯ ИЗ ПАМЯТИ не будет: перекантовка идёт по уже прочитанному.
    float kv = 0.f, vv = 0.f;
    const bool own = threadIdx.x < d;
    if (own) {
        if (KVB == 87) {
            kv = e4m3_to_float_fast(reinterpret_cast<const uint8_t*>(kc)[src + threadIdx.x]) * k_scale;
            vv = e4m3_to_float_fast(reinterpret_cast<const uint8_t*>(vc)[src + threadIdx.x]) * v_scale;
        } else {
            kv = __half2float(reinterpret_cast<const half*>(kc)[src + threadIdx.x]);
            vv = __half2float(reinterpret_cast<const half*>(vc)[src + threadIdx.x]);
        }
    }
    float ka = fabsf(kv), va = fabsf(vv);
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        ka = fmaxf(ka, __shfl_xor_sync(0xffffffff, ka, o));
        va = fmaxf(va, __shfl_xor_sync(0xffffffff, va, o));
    }
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) { smax[wid] = ka; smax[32 + wid] = va; }
    __syncthreads();
    if (wid == 0) {
        float a = lane < nw ? smax[lane] : 0.f, b = lane < nw ? smax[32 + lane] : 0.f;
        #pragma unroll
        for (int o = 16; o; o >>= 1) {
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
            b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, o));
        }
        if (lane == 0) { smax[0] = a; smax[32] = b; }
    }
    __syncthreads();
    // Масштаб -- ровно как в эталонном квантовании обвязки: absmax/127, симметрично, к ближайшему.
    // Нулевая строка дала бы деление на ноль, поэтому пол в наименьшее нормальное fp16.
    const float ks = fmaxf(smax[0], 6.1e-5f) / 127.f, vs = fmaxf(smax[32], 6.1e-5f) / 127.f;
    if (threadIdx.x == 0) {
        oks[(long)h * T + t] = ks;              // ядро ролей ждёт масштабы в fp32
        ovs[(long)h * T + t] = vs;
    }
    // ДЕЛЕНИЕ ЯВНО ТОЧНОЕ. Ядро собирается с --use_fast_math, где `/` превращается в приближённый
    // обратный, и результат уходит на ULP. Само по себе это мелочь, но значения e4m3 имеют мало
    // разрядов мантиссы, поэтому kv/ks ЧАСТО попадает ровно на границу .5 -- и ULP переворачивает
    // округление, давая байт на единицу иной, чем у эталона. Замерено: с fast-math сверка с явным
    // эталоном не проходит, с __fdiv_rn совпадает побитово. Цена нулевая -- ядро упирается в память.
    if (own) {
        okq[dst + threadIdx.x] = (int8_t)max(-127.f, fminf(127.f, rintf(__fdiv_rn(kv, ks))));
        ovq[dst + threadIdx.x] = (int8_t)max(-127.f, fminf(127.f, rintf(__fdiv_rn(vv, vs))));
    }
}

// СБОРКА БАЙТОВ КАК ЕСТЬ -- для e5m2. Ни конверсии, ни свёртки, ни масштабов: байт e5m2 УЖЕ есть
// старший байт fp16 (§4nn), поэтому его надо просто ПЕРЕЛОЖИТЬ из страниц в непрерывное [Hkv,T,d],
// а разворот сделает ядро внимания одной перестановкой байт. Это тот случай, ради которого весь
// байтовый путь и строился: сборщик перестаёт быть вычислением и становится перевозкой.
// Ширина -- uint4 там, где строка кратна 16 Б (d=256 кратно): §1c, MIO считает ИНСТРУКЦИИ.
__global__ void gather_paged_kv_bytes_kernel(
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int* __restrict__ block_table, uint8_t* __restrict__ ok, uint8_t* __restrict__ ov,
    int T, int Hkv, int d, long blk_stride, int block_size) {
    const int t = blockIdx.x, h = blockIdx.y;
    if (t >= T) return;
    const int lb = t / block_size;
    const long src = (long)block_table[lb] * blk_stride + ((long)(t - lb * block_size) * Hkv + h) * d;
    const long dst = ((long)h * T + t) * d;
    if ((d & 15) == 0) {
        const int n = d >> 4;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            reinterpret_cast<uint4*>(ok + dst)[i] = reinterpret_cast<const uint4*>(kc + src)[i];
            reinterpret_cast<uint4*>(ov + dst)[i] = reinterpret_cast<const uint4*>(vc + src)[i];
        }
    } else {
        for (int i = threadIdx.x; i < d; i += blockDim.x) { ok[dst + i] = kc[src + i]; ov[dst + i] = vc[src + i]; }
    }
}

// [serving] СБОРКА ИЗ int8-ПУЛА В ФОРМАТ ЯДРА РОЛЕЙ -- БЕЗ ЕДИНОЙ ПОТЕРИ.
// Когда пул СРАЗУ хранит int8 с масштабом на позицию, префиллу нечего конвертировать: байты
// перекладываются как есть, а масштаб просто переносится из побочной таблицы в [Hkv, T]. Это тот
// самый выигрыш, ради которого меняется формат: у e4m3-пула тот же маршрут требовал ВТОРОГО
// квантования и стоил 12.6-17.9x точности (tests/test_vllm_i8_gather.py), здесь он ТОЖДЕСТВЕНЕН.
__global__ void gather_paged_kv_i8_pool_kernel(
    const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
    const float* __restrict__ kst, const float* __restrict__ vst,
    const int* __restrict__ block_table,
    int8_t* __restrict__ okq, float* __restrict__ oks,
    int8_t* __restrict__ ovq, float* __restrict__ ovs,
    int T, int Hkv, int d, long blk_stride, int block_size) {
    const int t = blockIdx.x, h = blockIdx.y;
    if (t >= T) return;
    const int lb = t / block_size, wb = t - lb * block_size;
    const long pb = block_table[lb];
    const long src = pb * blk_stride + ((long)wb * Hkv + h) * d;
    const long dst = ((long)h * T + t) * d;
    const long sidx = (pb * (long)block_size + wb) * Hkv + h;
    if (threadIdx.x == 0) {
        oks[(long)h * T + t] = kst[sidx];
        ovs[(long)h * T + t] = vst[sidx];
    }
    if ((d & 15) == 0) {                                       // d=256: uint4, MIO считает ИНСТРУКЦИИ
        const int nq = d >> 4;
        for (int i = threadIdx.x; i < nq; i += blockDim.x) {
            reinterpret_cast<uint4*>(okq + dst)[i] = reinterpret_cast<const uint4*>(kc + src)[i];
            reinterpret_cast<uint4*>(ovq + dst)[i] = reinterpret_cast<const uint4*>(vc + src)[i];
        }
    } else {
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            okq[dst + i] = kc[src + i]; ovq[dst + i] = vc[src + i];
        }
    }
}

// [serving] Тот же обход, но выход fp16 -- для форм, где ядро ролей не применимо (d != 256).
// Разворот здесь ТОЧНЫЙ по построению: значение = q * s, одно умножение, никакого повторного
// квантования. Именно поэтому оба маршрута чтения int8-пула дают ОДИН формат: разница только в цене.
__global__ void gather_paged_kv_i8f16_kernel(
    const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
    const float* __restrict__ kst, const float* __restrict__ vst,
    const int* __restrict__ block_table, half* __restrict__ ok, half* __restrict__ ov,
    int T, int Hkv, int d, long blk_stride, int block_size) {
    const int t = blockIdx.x, h = blockIdx.y;
    if (t >= T) return;
    const int lb = t / block_size, wb = t - lb * block_size;
    const long pb = block_table[lb];
    const long src = pb * blk_stride + ((long)wb * Hkv + h) * d;
    const long dst = ((long)h * T + t) * d;
    const long sidx = (pb * (long)block_size + wb) * Hkv + h;
    const float ks = kst[sidx], vs = vst[sidx];
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        ok[dst + i] = __float2half((float)kc[src + i] * ks);
        ov[dst + i] = __float2half((float)vc[src + i] * vs);
    }
}

// [WIDE LOADS] Read the KV bytes through the WIDEST type the lane offset is aligned for, then unpack in
// registers. SASS audit (tools/sass_audit.py) on the shipped decode kernel showed every LDG issued as
// LDG.E.32 -- 107 of them -- with 7.3 address-arithmetic instructions EACH, because the e4m3 path read
// uchar4 (4 B) and the fp16 path half2 (4 B) while a lane actually owns EPT*sizeof(elem) contiguous bytes:
// 8 B at d=256/e4m3, 16 B at d=256/fp16. ptxas does not widen these on its own -- it keeps the access width
// the source asked for. One LDG.E.128 replaces four LDG.E.32 and their address math with it.
// Alignment is by construction: the lane offset is lane*EPT elements, so EPT*sizeof(elem) bytes, and the
// pool base is a torch allocation (256 B aligned) offset by whole (block, token, head) strides.
// NB: the alignment ASSUMPTION is the load-bearing part. Without it ptxas cannot prove the pointer is
// 8/16-byte aligned (it arrives as a void* plus a runtime offset) and conservatively splits every access
// back into 32-bit pieces -- measured: 97 of 104 LDGs stayed .32 on the first attempt. __builtin_assume_
// aligned states what is true by construction: lane offset = lane*EPT elements, pool base is a torch
// allocation (256 B) plus whole block/token/head strides.
template <int NB>          // NB = bytes this lane loads
__device__ __forceinline__ void load_bytes(const uint8_t* praw, uint8_t* dst) {
    if constexpr (NB % 16 == 0) {
        const uint8_t* p = (const uint8_t*)__builtin_assume_aligned(praw, 16);
        #pragma unroll
        for (int c = 0; c < NB / 16; ++c)
            reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(p)[c];
    } else if constexpr (NB % 8 == 0) {
        const uint8_t* p = (const uint8_t*)__builtin_assume_aligned(praw, 8);
        #pragma unroll
        for (int c = 0; c < NB / 8; ++c)
            reinterpret_cast<uint2*>(dst)[c] = reinterpret_cast<const uint2*>(p)[c];
    } else if constexpr (NB % 4 == 0) {
        const uint8_t* p = (const uint8_t*)__builtin_assume_aligned(praw, 4);
        #pragma unroll
        for (int c = 0; c < NB / 4; ++c)
            reinterpret_cast<uint32_t*>(dst)[c] = reinterpret_cast<const uint32_t*>(p)[c];
    } else {
        #pragma unroll
        for (int c = 0; c < NB; ++c) dst[c] = praw[c];
    }
}

template <int EPT, int KVB>
__device__ __forceinline__ void load_raw(const void* base, long eoff, float out[EPT]) {
    if (KVB == 52) {                                   // fp8_e5m2, 1 byte/elem (see e5m2_to_float)
        const uint8_t* p = reinterpret_cast<const uint8_t*>(base) + eoff;
        uint8_t b[EPT];
        load_bytes<EPT>(p, b);
        #pragma unroll
        for (int e = 0; e < EPT; ++e) out[e] = e5m2_to_float(b[e]);
    } else if (KVB == 87) {                            // fp8_e4m3 (vLLM KV pool), 1 byte/elem
        const uint8_t* p = reinterpret_cast<const uint8_t*>(base) + eoff;
        uint8_t b[EPT];
        load_bytes<EPT>(p, b);
        #pragma unroll
        for (int e = 0; e < EPT; ++e) out[e] = e4m3_to_float_fast(b[e]);
    } else if (KVB == 16) {                            // EPT halves via half2 (32-bit) loads
        const half* p = reinterpret_cast<const half*>(base) + eoff;
        half hb[EPT];
        load_bytes<EPT * 2>(reinterpret_cast<const uint8_t*>(p), reinterpret_cast<uint8_t*>(hb));
        #pragma unroll
        for (int e2 = 0; e2 < EPT / 2; ++e2) {
            half2 h = *reinterpret_cast<const half2*>(hb + e2 * 2);
            out[e2 * 2] = __low2float(h); out[e2 * 2 + 1] = __high2float(h); }
    } else if (KVB == 8) {                             // EPT int8 via char4 (32-bit) chunks; EPT=2 via char2
        const int8_t* p = reinterpret_cast<const int8_t*>(base) + eoff;
        int8_t b[EPT];
        load_bytes<EPT>(reinterpret_cast<const uint8_t*>(p), reinterpret_cast<uint8_t*>(b));
        #pragma unroll
        for (int e = 0; e < EPT; ++e) out[e] = b[e];
    } else {                                           // int4 packed 2/byte, zero-point 8; uint16 (4 nibbles) chunks
        const uint8_t* p = reinterpret_cast<const uint8_t*>(base) + (eoff >> 1);
        if (EPT % 4 == 0) {
            uint8_t b[EPT / 2];
            load_bytes<EPT / 2>(p, b);
            #pragma unroll
            for (int c = 0; c < EPT / 2; ++c) {
                out[c * 2] = (int)(b[c] & 0xF) - 8; out[c * 2 + 1] = (int)(b[c] >> 4) - 8; }
        } else { uint8_t byte = *p; out[0] = (int)(byte & 0xF) - 8; out[1] = (int)(byte >> 4) - 8; }
    }
}

template <int EPT, int KVB>
__global__ void split_decode_kernel(const half* __restrict__ Q, const void* __restrict__ K, const void* __restrict__ V,
                                    const float* __restrict__ sK, const float* __restrict__ sV,
                                    float* __restrict__ Op, float* __restrict__ Lp, float* __restrict__ Mp, int* __restrict__ Jp,
                                    int H, int Hkv, int Sk, int d, float scale, int num_splits, int Lc,
                                    float softcap, const float* __restrict__ alibi_slopes) {
    const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    // Score modifiers (online path -> the running max absorbs any bounded bias, no U-bound needed):
    //   softcap>0: s -> cap*tanh(s/cap);   ALiBi: += slope_h*(key - query), query at position Sk-1.
    // Applied inline at each `part` site (SMOD) after the QK reduce; no device lambda (needs a flag).
    const float alibi = alibi_slopes ? alibi_slopes[h] : 0.f;
    const float sc_inv = softcap > 0.f ? 1.0f / softcap : 0.f;
    const int qpos = Sk - 1;
#define SMOD(part, key) do { \
        if (softcap > 0.f) (part) = softcap * tanhf((part) * sc_inv); \
        if (alibi != 0.f) (part) += alibi * (float)((key) - qpos); } while (0)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (H / Hkv);
    const int k0 = c * Lc, k1 = min(k0 + Lc, Sk);
    const int oidx = ((b * H + h) * num_splits + c);
    __shared__ float sm[NW], sl[NW], sacc[NW * 32 * EPT]; __shared__ int sj[NW];

    const half* Qb = Q + (long)(b * H + h) * d;
    float qr[EPT], acc[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) { qr[e] = scale * __half2float(Qb[lane * EPT + e]); acc[e] = 0.f; }
    float m = -CUDART_INF_F, l = 0.f; int jmax = -1;
    const long kvbase = (long)(b * Hkv + hkv) * Sk * d;   // fp16/int8 element base; int4 uses eoff>>1
    const float* sKb = sK ? sK + (long)(b * Hkv + hkv) * Sk : nullptr;
    const float* sVb = sV ? sV + (long)(b * Hkv + hkv) * Sk : nullptr;

    constexpr int U = 4;                               // unroll: U keys in flight per warp step (ILP+MLP)
    int j = k0 + warp;
    for (; j + (U - 1) * NW < k1; j += U * NW) {
        float part[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {                  // U independent dots (loads overlap -> MLP)
            int jj = j + u * NW; float ksl[EPT];
            load_raw<EPT, KVB>(K, kvbase + (long)jj * d + lane * EPT, ksl);
            float pp = 0.f;
            #pragma unroll
            for (int e = 0; e < EPT; ++e) pp += qr[e] * ksl[e];
            part[u] = pp;
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {                  // U independent warp-reduces -> ILP
            for (int o = 16; o > 0; o >>= 1) part[u] += __shfl_xor_sync(0xffffffffu, part[u], o);
            if (sKb) part[u] *= sKb[j + u * NW];        // K per-token scale ONCE, after reduce
            SMOD(part[u], j + u * NW); }                // softcap / ALiBi (online: running max absorbs it)
        float m_old = m;
        #pragma unroll
        for (int u = 0; u < U; ++u) if (part[u] > m) { m = part[u]; jmax = j + u * NW; }   // running max
        float corr = (m_old == -CUDART_INF_F) ? 0.f : __expf(m_old - m);                   // rescale previous ONCE
        l *= corr;
        #pragma unroll
        for (int e = 0; e < EPT; ++e) acc[e] *= corr;
        #pragma unroll
        for (int u = 0; u < U; ++u) {                  // add the U new contributions relative to m
            float p = FA2SM70_EXP_MASKED(part[u], m);
            l += p; int jj = j + u * NW; float pv = sVb ? p * sVb[jj] : p; float vsl[EPT];
            load_raw<EPT, KVB>(V, kvbase + (long)jj * d + lane * EPT, vsl);
            #pragma unroll
            for (int e = 0; e < EPT; ++e) acc[e] += pv * vsl[e];
        }
    }
    for (; j < k1; j += NW) {                           // tail: remaining < U*NW keys, standard online step
        const long eoff = kvbase + (long)j * d + lane * EPT;
        float ksl[EPT]; load_raw<EPT, KVB>(K, eoff, ksl);
        float part = 0.f;
        #pragma unroll
        for (int e = 0; e < EPT; ++e) part += qr[e] * ksl[e];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
        if (sKb) part *= sKb[j];
        SMOD(part, j);                                  // softcap / ALiBi
        float m_old = m; if (part > m) { m = part; jmax = j; }
        float corr = (m_old == -CUDART_INF_F) ? 0.f : __expf(m_old - m);
        float p = FA2SM70_EXP_MASKED(part, m);
        float pv = sVb ? p * sVb[j] : p; float vsl[EPT]; load_raw<EPT, KVB>(V, eoff, vsl);
        l = l * corr + p;
        #pragma unroll
        for (int e = 0; e < EPT; ++e) acc[e] = acc[e] * corr + pv * vsl[e];
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; sj[warp] = jmax; }
    #pragma unroll
    for (int e = 0; e < EPT; ++e) sacc[warp * 32 * EPT + lane * EPT + e] = acc[e];
    __syncthreads();

    if (warp == 0) {
        float gm = -CUDART_INF_F; int gj = -1;
        #pragma unroll
        for (int w = 0; w < NW; ++w) if (sm[w] > gm) { gm = sm[w]; gj = sj[w]; }
        if (gm == -CUDART_INF_F) {
            #pragma unroll
            for (int e = 0; e < EPT; ++e) Op[(long)oidx * d + lane * EPT + e] = 0.f;
            if (lane == 0) { Lp[oidx] = -CUDART_INF_F; if (Mp) Mp[oidx] = -CUDART_INF_F; if (Jp) Jp[oidx] = -1; }
            return;
        }
        float gl = 0.f, ga[EPT];
        #pragma unroll
        for (int e = 0; e < EPT; ++e) ga[e] = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) { float cw = __expf(sm[w] - gm); gl += cw * sl[w];
            #pragma unroll
            for (int e = 0; e < EPT; ++e) ga[e] += cw * sacc[w * 32 * EPT + lane * EPT + e]; }
        #pragma unroll
        for (int e = 0; e < EPT; ++e) Op[(long)oidx * d + lane * EPT + e] = ga[e] / (gl == 0.f ? 1.f : gl);
        if (lane == 0) { Lp[oidx] = gm + logf(gl); if (Mp) Mp[oidx] = gm; if (Jp) Jp[oidx] = gj; }
    }
}
#undef SMOD

// combine splits; optional exact top-1 control point (Kf/Vf = full fp16 KV) removes the spike term.
// Control variate: O_corr = O_blend + (v* - O_blend)(pi* - pi_hat), where j* = global
// argmax key, pi_hat = exp(quant_score - LSE_g)= its quantized weight, pi* = exp(exact_score - LSE_g).
template <int EPT>
__global__ void reduce_splits_kernel(const float* __restrict__ Op, const float* __restrict__ Lp,
                                     const float* __restrict__ Mp, const int* __restrict__ Jp,
                                     const half* __restrict__ Kf, const half* __restrict__ Vf,
                                     const half* __restrict__ Q, float* __restrict__ O,
                                     float* __restrict__ Lse,       // [B,H] final natural-log LSE (null = skip)
                                     int H, int Hkv, int Sk, int d, float scale, int num_splits) {
    const int h = blockIdx.x, b = blockIdx.y, lane = threadIdx.x;   // 32 lanes cover d via EPT
    const int base = (b * H + h) * num_splits;
    float mx = -CUDART_INF_F;
    for (int c = 0; c < num_splits; ++c) mx = fmaxf(mx, Lp[base + c]);
    if (mx == -CUDART_INF_F) {
        for (int e = 0; e < EPT; ++e) O[(long)(b * H + h) * d + lane * EPT + e] = 0.f;
        if (Lse && lane == 0) Lse[b * H + h] = -CUDART_INF_F;
        return;
    }
    float denom = 0.f, o[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) o[e] = 0.f;
    for (int c = 0; c < num_splits; ++c) {
        float w = __expf(Lp[base + c] - mx); denom += w;
        #pragma unroll
        for (int e = 0; e < EPT; ++e) o[e] += w * Op[(long)(base + c) * d + lane * EPT + e];
    }
    if (Lse && lane == 0) Lse[b * H + h] = mx + logf(denom);   // final LSE (free: mx,denom already here)
    float ob[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) ob[e] = o[e] / denom;
    if (Kf && Mp) {                                    // exact top-1 control point
        float gm = -CUDART_INF_F; int gj = -1;
        for (int c = 0; c < num_splits; ++c) if (Mp[base + c] > gm) { gm = Mp[base + c]; gj = Jp[base + c]; }
        if (gj >= 0) {
            const int hkv = h / (H / Hkv);
            const long off = (long)(b * Hkv + hkv) * Sk * d + (long)gj * d + lane * EPT;
            const half* Qb = Q + (long)(b * H + h) * d;
            float sdot = 0.f;
            #pragma unroll
            for (int e = 0; e < EPT; ++e) sdot += (scale * __half2float(Qb[lane * EPT + e])) * __half2float(Kf[off + e]);
            #pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) sdot += __shfl_xor_sync(0xffffffffu, sdot, o2);   // exact score of j*
            float pi_hat = __expf(gm - mx) / denom;                       // quantized weight of j*
            float pi_star = __expf(sdot - mx) / denom;                    // exact weight of j*
            #pragma unroll
            for (int e = 0; e < EPT; ++e) ob[e] += (__half2float(Vf[off + e]) - ob[e]) * (pi_star - pi_hat);
        }
    }
    #pragma unroll
    for (int e = 0; e < EPT; ++e) O[(long)(b * H + h) * d + lane * EPT + e] = ob[e];
}

static void launch(int KVB, const half* q, const void* k, const void* v, const float* sk, const float* sv,
                   const half* kf, const half* vf, float* op, float* lp, float* mp, int* jp, float* o,
                   int B, int H, int Hkv, int Sk, int d, float scale, int ns, int Lc, bool topk, cudaStream_t st,
                   float softcap = 0.f, const float* alibi = nullptr, float* lse = nullptr) {
    dim3 g1(ns, H, B); dim3 g2(H, B);
    const half* qk = topk ? q : nullptr;   // reducer needs Q only for the control point
    // [ОКНО, НИ РАЗУ НЕ ПРОВЕРЕННОЕ] РАЗДЕЛ L1 / РАЗДЕЛЯЕМОЙ. Эти ядра запускаются со СТАТИЧЕСКОЙ
    // разделяемой (нулевой динамический аргумент), поэтому раздел 128 КБ выбирает драйвер сам, а мы
    // в него не вмешивались никогда -- `PreferredSharedMemoryCarveout` не выставлен нигде в дереве.
    // Гипотеза стоит проверки именно на ДЕКОДЕ: он на 90 % упирается в HBM, а при GQA одну и ту же
    // плитку KV читают несколько голов группы -- то есть больший L1 может дать попадания там, где
    // разделяемая простаивает. Обратное тоже возможно, поэтому это ЗАМЕР, а не правка.
    // FA2SM70_DEC_CARVEOUT = доля разделяемой в процентах (0 = максимум L1, 100 = максимум shared).
    // Не задан -- поведение прежнее, ни один атрибут не трогается.
    static const int carve = getenv("FA2SM70_DEC_CARVEOUT") ? atoi(getenv("FA2SM70_DEC_CARVEOUT")) : -1;
// Склейки токенов тут быть НЕ МОЖЕТ: K -- шаблонное выражение с запятой и угловыми скобками.
// Каждое раскрытие и так сидит в своей области видимости do{}while(0), поэтому обычный static
// достаточен и даёт по одному флагу на инстанцирование.
#define SET_CARVE(K) do { if (carve >= 0) { static bool done = false; if (!done) { \
        done = true; cudaFuncSetAttribute((const void*)(K), cudaFuncAttributePreferredSharedMemoryCarveout, carve); } } } while(0)
#define SPL(EPT) do { \
        if (KVB == 16) SET_CARVE((split_decode_kernel<EPT,16>)); \
        else if (KVB == 8) SET_CARVE((split_decode_kernel<EPT,8>)); \
        else SET_CARVE((split_decode_kernel<EPT,4>)); \
        if (KVB == 16) split_decode_kernel<EPT,16><<<g1,BLK,0,st>>>(q,k,v,sk,sv,op,lp,mp,jp,H,Hkv,Sk,d,scale,ns,Lc,softcap,alibi); \
        else if (KVB == 8) split_decode_kernel<EPT,8><<<g1,BLK,0,st>>>(q,k,v,sk,sv,op,lp,mp,jp,H,Hkv,Sk,d,scale,ns,Lc,softcap,alibi); \
        else split_decode_kernel<EPT,4><<<g1,BLK,0,st>>>(q,k,v,sk,sv,op,lp,mp,jp,H,Hkv,Sk,d,scale,ns,Lc,softcap,alibi); \
        reduce_splits_kernel<EPT><<<g2,32,0,st>>>(op,lp,mp,jp,kf,vf,qk,o,lse,H,Hkv,Sk,d,scale,ns); } while(0)
    if (d == 512) SPL(16); else if (d == 256) SPL(8); else if (d == 128) SPL(4); else SPL(2);
#undef SPL
#undef SET_CARVE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// exact fp16 path (drop-in, lossless). Optional score modifiers -- softcap and per-head ALiBi -- ride
// the online running-max (no U-bound assumption), so they compose exactly (the deferred path's
// Cauchy-Schwarz U does not bound modified scores). alibi_slopes: fp32 [H] (empty = off).
std::tuple<torch::Tensor, torch::Tensor> flash_decode(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, int64_t num_splits,
                                                      double softcap, torch::Tensor alibi_slopes) {
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    int B = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Hkv = K.size(1), Sk = K.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Sq == 1 && (d == 64 || d == 128 || d == 256 || d == 512) && H % Hkv == 0);
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    const float* al = nullptr; torch::Tensor als;
    if (alibi_slopes.defined() && alibi_slopes.numel() > 0) { TORCH_CHECK(alibi_slopes.scalar_type() == torch::kFloat32 && alibi_slopes.numel() == H, "alibi_slopes fp32 [H]");
        als = alibi_slopes.contiguous(); al = als.data_ptr<float>(); }
    int ns = std::max<int>(1, std::min<int>(num_splits, (Sk + 31) / 32)); int Lc = (Sk + ns - 1) / ns;
    auto f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto Op = torch::empty({B, H, ns, d}, f32), Lp = torch::empty({B, H, ns}, f32), O = torch::empty({B, H, Sq, d}, f32);
    auto Lse = torch::empty({B, H, Sq}, f32);   // final LSE written by the reduce kernel (no torch logsumexp)
    launch(16, (const half*)Qc.data_ptr<at::Half>(), Kc.data_ptr<at::Half>(), Vc.data_ptr<at::Half>(), nullptr, nullptr,
           nullptr, nullptr, Op.data_ptr<float>(), Lp.data_ptr<float>(), nullptr, nullptr, O.data_ptr<float>(),
           B, H, Hkv, Sk, d, (float)scale, ns, Lc, false, at::cuda::getCurrentCUDAStream(), (float)softcap, al, Lse.data_ptr<float>());
    return std::make_tuple(O, Lse);
}

// quantized path: pre-quantized cache Kq/Vq (int8 or packed-int4) + per-token scales sK/sV.
// Kf/Vf optional full-fp16 KV for the exact top-1 control point (int4).
std::tuple<torch::Tensor, torch::Tensor> flash_decode_q(torch::Tensor Q, torch::Tensor Kq, torch::Tensor sK, torch::Tensor Vq, torch::Tensor sV,
                                                        double scale, int64_t num_splits, int64_t bits,
                                                        c10::optional<torch::Tensor> Kf, c10::optional<torch::Tensor> Vf) {
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    int B = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Hkv = sK.size(1), Sk = sK.size(2);
    TORCH_CHECK(Sq == 1 && (d == 64 || d == 128 || d == 256 || d == 512) && H % Hkv == 0 && (bits == 8 || bits == 4));
    auto Qc = Q.contiguous();
    int ns = std::max<int>(1, std::min<int>(num_splits, (Sk + 31) / 32)); int Lc = (Sk + ns - 1) / ns;
    auto f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto Op = torch::empty({B, H, ns, d}, f32), Lp = torch::empty({B, H, ns}, f32), O = torch::empty({B, H, Sq, d}, f32);
    bool topk = Kf.has_value();
    auto Jp = torch::empty({B, H, ns}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Mp = torch::empty({B, H, ns}, f32);
    torch::Tensor Kfc, Vfc;
    const half *kf = nullptr, *vf = nullptr;
    if (topk) { Kfc = Kf->contiguous(); Vfc = Vf->contiguous(); kf = (const half*)Kfc.data_ptr<at::Half>(); vf = (const half*)Vfc.data_ptr<at::Half>(); }
    auto Lse = torch::empty({B, H, Sq}, f32);   // final LSE written by the reduce kernel (no torch logsumexp)
    launch((int)bits, (const half*)Qc.data_ptr<at::Half>(), Kq.data_ptr(), Vq.data_ptr(), sK.data_ptr<float>(), sV.data_ptr<float>(),
           kf, vf, Op.data_ptr<float>(), Lp.data_ptr<float>(), Mp.data_ptr<float>(), Jp.data_ptr<int>(), O.data_ptr<float>(),
           B, H, Hkv, Sk, d, (float)scale, ns, Lc, topk, at::cuda::getCurrentCUDAStream(), 0.f, nullptr, Lse.data_ptr<float>());
    return std::make_tuple(O, Lse);
}
// ---- Deferred-rescale variant: certified U = ||scale q|| * max||k|| >= max s_j, so
// p_j = exp(s_j - U) sums associatively -- NO running max, NO online rescale, NO argmax. All splits
// share U => the reducer is a plain sum. fp32 range condition (Delta<=~69) makes Cauchy-Schwarz safe.
// Logical key index -> physical element base. Contiguous cache (PAGED=false): (b,hkv) slab, key j at
// ((b*Hkv+hkv)*Sk + j)*d. Paged cache (PAGED=true): a shared block pool [num_blocks, block_size, Hkv, d];
// block_table[b*max_blocks + j/block_size] gives the physical block, j%block_size the slot within it.
template <bool PAGED>
__device__ __forceinline__ long kvb_off(long kvbase, int j, int d, const int* __restrict__ bt,
                                        int b, int hkv, int block_size, int max_blocks, int Hkv) {
    if constexpr (PAGED) {
        int lb = j / block_size, wb = j - lb * block_size;
        long pb = bt[(long)b * max_blocks + lb];
        // `kvbase` carries the BLOCK stride in elements for the paged case. It is block_size*Hkv*d for a
        // densely packed pool, but a serving engine may hand us a strided view: vLLM keeps KV as
        // [num_blocks, 2, block_size, Hkv, D] and passes `kv_cache.unbind(1)`, whose block stride is
        // TWICE the dense one. Taking the stride as a parameter is what lets us read that view in place
        // -- calling .contiguous() on it instead copied the whole 640 MB layer pool on EVERY decode
        // call (measured 3.77 ms/call in-server vs 0.048 ms for the same kernel on a packed cache).
        return (pb * kvbase + (long)wb * Hkv + hkv) * (long)d;
    } else {
        return kvbase + (long)j * d;
    }
}

// [paged int8] per-token scale slot for key j. Same block-table walk as kvb_off but WITHOUT the *d:
// there is one scale per (token, kv-head), so the side table is [num_blocks, block_size, Hkv].
// Non-paged callers pre-offset their base pointer, so the index is just j.
template <bool PAGED>
__device__ __forceinline__ long s_off(int j, const int* __restrict__ bt, int b, int hkv,
                                      int block_size, int max_blocks, int Hkv) {
    if constexpr (PAGED) {
        int lb = j / block_size, wb = j - lb * block_size;
        long pb = bt[(long)b * max_blocks + lb];
        return (pb * block_size + wb) * Hkv + hkv;
    } else {
        return j;
    }
}

template <int EPT, int KVB, bool PAGED = false>
__global__ void split_defer_kernel(const half* __restrict__ Q, const void* __restrict__ K, const void* __restrict__ V,
                                   const float* __restrict__ sK, const float* __restrict__ sV, const float* __restrict__ Kmax,
                                   float* __restrict__ A, float* __restrict__ B, int H, int Hkv, int Sk, int d, float scale, int ns, int Lc, int kv_len,
                                   const int* __restrict__ block_table, int block_size, int max_blocks, const int* __restrict__ seqlens,
                                   const int* __restrict__ kvstart,
                                   // см. MQA-ядро: qGrp -- строк спекуляции на запрос; ядро само
                                   // берёт запрос как b/qGrp. qGrp=1 -- прежнее поведение бит в бит.
                                   int qGrp = 1, int qCausal = 1) {
    const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (H / Hkv);
    const int breq = (qGrp > 1) ? (b / qGrp) : b;
    const int qpos = (qGrp > 1) ? (b - breq * qGrp) : 0;
    const int b_len = seqlens ? (seqlens[breq] - (qCausal ? (qGrp - 1 - qpos) : 0)) : kv_len;                 // per-sequence valid length (paged serving)
    // Split the SEQUENCE, not the block table's CAPACITY. The host cannot size Lc from the live
    // length -- reading it needs a device sync, illegal while the decode graph is captured -- so it
    // passes capacity/ns, and at ctx=4K out of a 164K-token table only the first 2 of 64 splits had
    // any work: 12 CTAs on 80 SMs, latency-bound, and that (not the reduce, and not KV bandwidth)
    // was the decode's fixed floor. `seqlens` is already here on the device, so derive the stride
    // from it: the grid stays static (CUDA-graph safe), only each CTA's range moves.
    // [ОКНО] Скользящий слой отличается от полного ТОЛЬКО левой границей: запрос в позиции P смотрит
    // на ключи [P-W+1, P]. Правая граница -- это уже имеющийся b_len, поэтому окно целиком выражается
    // ОДНИМ числом на последовательность: началом диапазона. Это точно ДО ТОКЕНА, в отличие от
    // подмены строки block_table (та умеет только кратное block_size и тащит лишние ключи), и оно же
    // снимает трафик: ядро читает W ключей вместо всей длины.
    const int b_st = kvstart ? kvstart[breq] : 0;
    const int Lb = seqlens ? ((b_len - b_st + ns - 1) / ns) : Lc;
    const int k0 = b_st + c * Lb, k1 = min(k0 + Lb, b_len);
    const int oidx = ((b * H + h) * ns + c);
    __shared__ float sB[NW], sA[NW * 32 * EPT], sM[NW];
    const half* Qb = Q + (long)(b * H + h) * d;
    float qr[EPT], acc[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) { qr[e] = scale * __half2float(Qb[lane * EPT + e]); acc[e] = 0.f; }
    // [ONLINE MAX] The certified Cauchy-Schwarz bound U = ||scale q|| * max||k|| is gone. It made the
    // result hostage to a number sourced OUTSIDE the kernel, and that number was a monotone running max
    // folded in by the KV store: one non-finite ||k|| latched it forever (U=NaN -> every decode NaN,
    // U=inf -> zeros), and even a finite outlier left it ~50x loose, at which point exp(score - U)
    // underflows to zero and the softmax degenerates to 0/0. A per-split running max is taken from the
    // scores themselves, so it is tight by construction and cannot be poisoned from outside.
    float m = -CUDART_INF_F;
    // PAGED: `Sk` carries the pool's BLOCK stride divided by d (block_size*Hkv for a packed pool,
    // 2*block_size*Hkv for vLLM's [num_blocks,2,block_size,Hkv,D] unbind(1) view). Non-paged keeps
    // the old per-(batch,kv-head) base.
    const long kvbase = PAGED ? (long)Sk : (long)(b * Hkv + hkv) * Sk * d;
    const float* sKb = sK ? (PAGED ? sK : sK + (long)(b * Hkv + hkv) * Sk) : nullptr;
    const float* sVb = sV ? (PAGED ? sV : sV + (long)(b * Hkv + hkv) * Sk) : nullptr;
    float Bsum = 0.f;
    // ГЛУБИНА НЕЗАВИСИМЫХ ЗАГРУЗОК. Замер §80c: ядро в 16 раз медленнее предела памяти и в
    // 38 -- предела вычислений, то есть связывает ЛАТЕНТНОСТЬ, а не канал и не полоса.
    // Глубина выведена под тот же макрос, что у соседнего тела, чтобы её можно было мерить.
    constexpr int UU = FA2SM70_DEC_UU ? FA2SM70_DEC_UU : 4;
    int j = k0 + warp;
    for (; j + (UU - 1) * NW < k1; j += UU * NW) {
        float part[UU];
        #pragma unroll
        for (int u = 0; u < UU; ++u) { int jj = j + u * NW; float ksl[EPT];
            load_raw<EPT, KVB>(K, kvb_off<PAGED>(kvbase, jj, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT, ksl);
            float pp = 0.f;
            #pragma unroll
            for (int e = 0; e < EPT; ++e) pp += qr[e] * ksl[e]; part[u] = pp; }
        #pragma unroll
        for (int u = 0; u < UU; ++u) { for (int o = 16; o > 0; o >>= 1) part[u] += __shfl_xor_sync(0xffffffffu, part[u], o);
            if (sKb) part[u] *= sKb[s_off<PAGED>(j + u * NW, block_table, breq, hkv, block_size, max_blocks, Hkv)]; }            // K per-token scale ONCE, after the reduce
        { float mo = m;                                   // rescale ONCE per unrolled group, if the max grew
            #pragma unroll
            for (int u = 0; u < UU; ++u) m = fmaxf(m, part[u]);
            if (m > mo) { const float corr = (mo == -CUDART_INF_F) ? 0.f : __expf(mo - m);
                Bsum *= corr;
                #pragma unroll
                for (int e = 0; e < EPT; ++e) acc[e] *= corr; } }
        #pragma unroll
        for (int u = 0; u < UU; ++u) {
            float p = FA2SM70_EXP_MASKED(part[u], m); Bsum += p; int jj = j + u * NW;
            float pv = sVb ? p * sVb[s_off<PAGED>(jj, block_table, breq, hkv, block_size, max_blocks, Hkv)] : p; float vsl[EPT]; // V per-token scale folded into p (once)
            load_raw<EPT, KVB>(V, kvb_off<PAGED>(kvbase, jj, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT, vsl);
            #pragma unroll
            for (int e = 0; e < EPT; ++e) acc[e] += pv * vsl[e]; }
    }
    for (; j < k1; j += NW) {
        const long eoff = kvb_off<PAGED>(kvbase, j, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT; float ksl[EPT];
        load_raw<EPT, KVB>(K, eoff, ksl);
        float part = 0.f;
        #pragma unroll
        for (int e = 0; e < EPT; ++e) part += qr[e] * ksl[e];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
        if (sKb) part *= sKb[s_off<PAGED>(j, block_table, breq, hkv, block_size, max_blocks, Hkv)];
        { const float mo = m; m = fmaxf(m, part);
          if (m > mo) { const float corr = (mo == -CUDART_INF_F) ? 0.f : __expf(mo - m);
              Bsum *= corr;
              #pragma unroll
              for (int e = 0; e < EPT; ++e) acc[e] *= corr; } }
        float p = FA2SM70_EXP_MASKED(part, m); Bsum += p;
        float pv = sVb ? p * sVb[s_off<PAGED>(j, block_table, breq, hkv, block_size, max_blocks, Hkv)] : p; float vsl[EPT];
        load_raw<EPT, KVB>(V, eoff, vsl);
        #pragma unroll
        for (int e = 0; e < EPT; ++e) acc[e] += pv * vsl[e];
    }
    if (lane == 0) { sB[warp] = Bsum; sM[warp] = m; }
    #pragma unroll
    for (int e = 0; e < EPT; ++e) sA[warp * 32 * EPT + lane * EPT + e] = acc[e];
    __syncthreads();
    if (warp == 0) {
        // Warps now carry DIFFERENT maxima, so the cross-warp merge is weighted (it used to be a plain
        // sum, valid only while every warp shared one U). Empty warps hold m=-inf: weight them 0
        // explicitly, since -inf minus -inf is NaN and 0*NaN would poison the whole split.
        float gm = -CUDART_INF_F;
        #pragma unroll
        for (int w = 0; w < NW; ++w) gm = fmaxf(gm, sM[w]);
        float gB = 0.f, gA[EPT];
        #pragma unroll
        for (int e = 0; e < EPT; ++e) gA[e] = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) { const float cw = (sM[w] == -CUDART_INF_F) ? 0.f : __expf(sM[w] - gm);
            gB += cw * sB[w];
            #pragma unroll
            for (int e = 0; e < EPT; ++e) gA[e] += cw * sA[w * 32 * EPT + lane * EPT + e]; }
        #pragma unroll
        for (int e = 0; e < EPT; ++e) A[(long)oidx * d + lane * EPT + e] = gA[e];
        // The per-split max rides in the tail of B (gridDim.z == batch), so no signature, no extra
        // allocation and no change to the caller-owned zero-allocation workspace path.
        if (lane == 0) { B[oidx] = gB; B[(long)gridDim.z * H * ns + oidx] = gm; }
    }
}

// [serving] Same reduction, but writing fp16 STRAIGHT into the engine's output buffer (and optionally
// applying the value scale). This removes an fp32 output tensor, an fp32->fp16 cast and a copy from
// every decode call -- the per-call glue that kept us behind an incumbent which writes into `out`
// directly. `Ohalf` may be strided in the row dimension (o_row_stride elements), as vLLM's [tokens,
// H, D] view is.
// Split-combine, parallel over splits. It used to run ONE warp per (head, batch) walking all `ns`
// splits serially -- at H=12, B=1 that is 12 warps for the whole GPU, and once the split count rose
// to 64 it became the decode's fixed floor rather than the KV read: 786 KB of partials pulled by 384
// threads, with nothing to hide the latency behind. Now NR warps split the `ns` loop between them and
// combine once through smem. Exactness is unchanged -- the partials are summed, and addition of the
// same values in a different order is the only thing that differs (all fp32).
// NR defaults so call sites stay `reduce_defer_out_kernel<EPT>`: these launches live inside macros,
// where a comma in the template argument list would be eaten as a macro argument separator.
template <int EPT, int NR = 8>
__global__ void reduce_defer_out_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                        half* __restrict__ Ohalf, int H, int d, int ns,
                                        long o_row_stride, float vscale) {
    const int h = blockIdx.x, b = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int base = (b * H + h) * ns;
    // [ONLINE MAX] Splits no longer share one exponent shift, so combining them is a max-weighted merge
    // instead of a plain sum. The per-split maxima ride in the tail of B (gridDim.y == batch).
    const float* __restrict__ Bmax = B + (long)gridDim.y * H * ns;
    float mx = -CUDART_INF_F;
    for (int c = 0; c < ns; ++c) mx = fmaxf(mx, Bmax[base + c]);
    float Bt = 0.f, o[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) o[e] = 0.f;
    for (int c = warp; c < ns; c += NR) {
        const float mc = Bmax[base + c];
        // SKIP empty splits outright. Their weight is exactly 0, so reading their d floats of A was pure
        // waste -- and it was the reason a large split count punished short contexts: the reducer's cost
        // scales with ns, not with the live length, so at ctx=8K/ns=256 it pulled 3.1 MB of zeros against
        // 8 MB of real KV. Skipping them makes ns a free parallelism knob and lets one static grid (which
        // is all a captured CUDA graph allows) serve both short and long sequences.
        if (mc == -CUDART_INF_F) continue;
        const float w = __expf(mc - mx);
        Bt += w * B[base + c];
        #pragma unroll
        for (int e = 0; e < EPT; ++e) o[e] += w * A[(long)(base + c) * d + lane * EPT + e]; }
    __shared__ float sO[NR * 32 * EPT];
    __shared__ float sBt[NR];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) sO[(warp * 32 + lane) * EPT + e] = o[e];
    if (lane == 0) sBt[warp] = Bt;                       // every lane holds the same partial
    __syncthreads();
    if (warp != 0) return;
    float tot = 0.f, acc[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) acc[e] = 0.f;
    for (int w = 0; w < NR; ++w) { tot += sBt[w];
        #pragma unroll
        for (int e = 0; e < EPT; ++e) acc[e] += sO[(w * 32 + lane) * EPT + e]; }
    const float inv = ((tot == 0.f) ? 1.f : 1.f / tot) * vscale;
    half* dst = Ohalf + (long)b * o_row_stride + (long)h * d;
    #pragma unroll
    for (int e = 0; e < EPT; ++e) dst[lane * EPT + e] = __float2half(acc[e] * inv);
}

template <int EPT>
__global__ void reduce_defer_kernel(const float* __restrict__ A, const float* __restrict__ B, const half* __restrict__ Q,
                                    const float* __restrict__ Kmax, float* __restrict__ O, float* __restrict__ Lse,
                                    int H, int Hkv, int d, int ns, float scale) {
    const int h = blockIdx.x, b = blockIdx.y, lane = threadIdx.x;
    const int base = (b * H + h) * ns;
    const float* __restrict__ Bmax = B + (long)gridDim.y * H * ns;       // per-split maxima (see reduce_defer_out)
    float mx = -CUDART_INF_F;
    for (int c = 0; c < ns; ++c) mx = fmaxf(mx, Bmax[base + c]);
    float Bt = 0.f, o[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) o[e] = 0.f;
    for (int c = 0; c < ns; ++c) {
        const float mc = Bmax[base + c];
        if (mc == -CUDART_INF_F) continue;                               // see reduce_defer_out_kernel
        const float w = __expf(mc - mx);
        Bt += w * B[base + c];
        #pragma unroll
        for (int e = 0; e < EPT; ++e) o[e] += w * A[(long)(base + c) * d + lane * EPT + e]; }
    float inv = (Bt == 0.f) ? 1.f : 1.f / Bt;
    #pragma unroll
    for (int e = 0; e < EPT; ++e) O[(long)(b * H + h) * d + lane * EPT + e] = o[e] * inv;
    // The LSE is now the TRUE one (mx + log sum), not bound + log sum: no Q reload, no Kmax, and it no
    // longer inherits the bound's looseness. Q/Kmax stay in the signature so every call site is unchanged.
    if (Lse && lane == 0) Lse[b * H + h] = (Bt == 0.f) ? -CUDART_INF_F : (mx + logf(Bt));
}

// MQA/GQA-aware deferred kernel: ONE CTA serves a group of GF query heads that share a KV head, so
// each K/V element is read from HBM ONCE and dotted against all GF queries (traffic /GF). Deferred
// (associative, no per-head rescale chain) so fusing GF heads is just GF independent accumulators;
// the certified U-bound is per query head. Output layout == the per-head kernel, so reduce_defer
// combines it unchanged. Requires (H/Hkv) % GF == 0 so all GF heads map to the same KV head.
template <int EPT, int KVB, int GF, bool VEQK = false, bool PAGED = false>
__global__ void split_defer_mqa_kernel(const half* __restrict__ Q, const void* __restrict__ K, const void* __restrict__ V,
                                       const float* __restrict__ sK, const float* __restrict__ sV, const float* __restrict__ Kmax,
                                       float* __restrict__ A, float* __restrict__ B, int H, int Hkv, int Sk, int d, float scale, int ns, int Lc, int kv_len,
                                       const int* __restrict__ block_table, int block_size, int max_blocks, const int* __restrict__ seqlens,
                                       const int* __restrict__ kvstart,
                                       // [МАСКА ХВОСТА -- ДЕРЕВО КАНДИДАТОВ] Ключи до `ctxlen[b]` видит КАЖДАЯ строка
                                       // (общий контекст). Ключи текущего шага -- только те, чей бит стоит в
                                       // `tailmask[b]`. Причинность ЦЕПИ выражалась одним числом (длиной префикса),
                                       // но у ДЕРЕВА брат лежит МЕЖДУ предками и самим узлом, и одним интервалом это
                                       // не выражается. Оба указателя нулевые = прежнее поведение до бита.
                                       const int* __restrict__ ctxlen = nullptr,
                                       const unsigned* __restrict__ tailmask = nullptr,
                                       // [РАЗРЕЖЕННЫЙ ОБХОД -- шаг 2] Без индекса путь БАЙТ В БАЙТ прежний.
                                       const int* __restrict__ selIdx = nullptr,
                                       const int* __restrict__ selCnt = nullptr,
                                       int selStride = 0, int selBlkLog = 7,
                                       // [ВИРТУАЛЬНЫЙ БАТЧ БЕЗ КОПИЙ] qGrp = строк спекуляции на ЗАПРОС.
                                       // Раньше хозяин РАСКЛАДЫВАЛ таблицу блоков и длины на каждую строку
                                       // (две копии + два запуска на шаг). Но строка r запроса p -- это
                                       // p = r / qGrp, а её длина -- sl[p] - (qGrp-1 - r%qGrp) при причинности.
                                       // Ядро считает это само; копии исчезают вместе с запусками.
                                       // qGrp = 1 -- прежнее поведение БИТ В БИТ.
                                       int qGrp = 1, int qCausal = 1) {
    // VEQK: values ARE the keys (Gemma-4 global layers, attention_k_eq_v -> no v_proj). We hold the
    // already-loaded K in registers and reuse it as V, halving the HBM read traffic on those layers.
    const int c = blockIdx.x, hg = blockIdx.y, b = blockIdx.z;   // hg: query-head GROUP (GF heads)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h0 = hg * GF, hkv = h0 / (H / Hkv);                // GF heads h0..h0+GF-1 share this KV head
    const int breq = (qGrp > 1) ? (b / qGrp) : b;                    // ЗАПРОС строки b
    const int qpos = (qGrp > 1) ? (b - breq * qGrp) : 0;             // позиция внутри спекуляции
    const int b_len = seqlens ? (seqlens[breq] - (qCausal ? (qGrp - 1 - qpos) : 0)) : kv_len;
    // Split the SEQUENCE, not the block table's CAPACITY. The host cannot size Lc from the live
    // length -- reading it needs a device sync, illegal while the decode graph is captured -- so it
    // passes capacity/ns, and at ctx=4K out of a 164K-token table only the first 2 of 64 splits had
    // any work: 12 CTAs on 80 SMs, latency-bound, and that (not the reduce, and not KV bandwidth)
    // was the decode's fixed floor. `seqlens` is already here on the device, so derive the stride
    // from it: the grid stays static (CUDA-graph safe), only each CTA's range moves.
    // [ОКНО] см. split_defer_kernel: левая граница на последовательность -- это всё, чем скользящий
    // слой отличается от полного, и она точна ДО ТОКЕНА.
    const int b_st = kvstart ? kvstart[breq] : 0;
    const int ctxb = tailmask ? ctxlen[b] : 0x7fffffff;      // граница «общий контекст | хвост шага»
    const unsigned tmb = tailmask ? tailmask[b] : 0u;
    const int Lb = seqlens ? ((b_len - b_st + ns - 1) / ns) : Lc;
    const int _k0d = b_st + c * Lb, _k1d = min(_k0d + Lb, b_len);
    // [СРЕЗ СПИСКА] Расщепление берёт срез СПИСКА ОТОБРАННЫХ БЛОКОВ по ЧИСЛУ, а не по диапазону
    // позиций: иначе часть расщеплений получила бы пустой срез, а часть -- всю работу.
    const int* _sI = nullptr; int _b0 = 0, _b1 = 1;
    if (selIdx != nullptr) {
        const long _o = (long)(b * Hkv + hkv);
        _sI = selIdx + _o * (long)selStride;
        const int _cnt = selCnt[_o];
        const int _per = (_cnt + ns - 1) / ns;
        _b0 = min(c * _per, _cnt); _b1 = min(_b0 + _per, _cnt);
    }
#if FA2SM70_DEC_SAH
    using SAcc = half;                       // выше fp16 у нас нигде ничего нет
#define SA_PUT(x) __float2half(x)
#define SA_GET(x) __half2float(x)
                       // выше fp16 у нас нигде ничего нет
#else
    using SAcc = float;
#define SA_PUT(x) (x)
#define SA_GET(x) (x)
#endif
    __shared__ float sB[NW * GF], sM[NW * GF];
    __shared__ SAcc  sA[NW * 32 * EPT * GF];
    float qr[GF][EPT], acc[GF][EPT], m[GF], Bsum[GF];
    #pragma unroll
    for (int g = 0; g < GF; ++g) {
        const half* Qb = Q + (long)(b * H + h0 + g) * d;
#if FA2SM70_DEC_QW
        half hb[EPT];                                    // 8 half = 16 B = ОДНО LDG.E.128
        load_bytes<EPT * 2>(reinterpret_cast<const uint8_t*>(Qb + lane * EPT),
                            reinterpret_cast<uint8_t*>(hb));
        #pragma unroll
        for (int e = 0; e < EPT; ++e) { qr[g][e] = scale * __half2float(hb[e]); acc[g][e] = 0.f; }
#else
        #pragma unroll
        for (int e = 0; e < EPT; ++e) { qr[g][e] = scale * __half2float(Qb[lane * EPT + e]); acc[g][e] = 0.f; }
#endif
        // [ONLINE MAX] see split_defer_kernel: the external certified bound is gone, each of the GF heads
        // carries its own running max taken from its own scores.
        m[g] = -CUDART_INF_F; Bsum[g] = 0.f;
    }
    // PAGED: `Sk` carries the pool's BLOCK stride divided by d (block_size*Hkv for a packed pool,
    // 2*block_size*Hkv for vLLM's [num_blocks,2,block_size,Hkv,D] unbind(1) view). Non-paged keeps
    // the old per-(batch,kv-head) base.
    const long kvbase = PAGED ? (long)Sk : (long)(b * Hkv + hkv) * Sk * d;
    const float* sKb = sK ? (PAGED ? sK : sK + (long)(b * Hkv + hkv) * Sk) : nullptr;
    const float* sVb = sV ? (PAGED ? sV : sV + (long)(b * Hkv + hkv) * Sk) : nullptr;
    // The in-flight register cost is UU*EPT, and EPT = d/32 -- so a single constant UU cannot be
    // right for both d=128 (EPT=4) and d=256 (EPT=8). Tuning UU=4 on a d=128 microbenchmark and
    // shipping it REGRESSED a real d=256 model by 10% at 128K on the server, which is exactly the
    // divergence a microbenchmark cannot show. Hold UU*EPT roughly constant instead.
    constexpr int UU = FA2SM70_DEC_UU ? FA2SM70_DEC_UU : (EPT <= 4 ? 4 : 2);
    // ВНЕШНИЙ ЦИКЛ ПО ОТОБРАННЫМ БЛОКАМ. При плотном пути -- РОВНО ОДНА итерация с прежними
    // границами. Накопители (m, Bsum, acc) объявлены ВЫШЕ и переживают смену блока: это тот же
    // онлайн-софтмакс, просто диапазон ключей теперь рваный. Предвыборка перезаряжается на
    // каждом блоке -- иначе она подтянула бы ключи ЗА его границей, то есть чужие.
    for (int _bi = _b0; _bi < _b1; ++_bi) {
    const int k0 = (_sI != nullptr) ? max(b_st, (int)((long)_sI[_bi] << selBlkLog)) : _k0d;
    const int k1 = (_sI != nullptr) ? min((int)(((long)_sI[_bi] + 1) << selBlkLog), b_len) : _k1d;
    int j = k0 + warp;
#if FA2SM70_DEC_PF
    // [MLP, шаг 2] Cross-iteration prefetch: issue the NEXT group's K/V before consuming this one, so
    // the load has a whole loop body of dots, shuffles and exp() to land in instead of the couple of
    // instructions it had. Costs 2*UU*EPT registers for the second buffer. Gated so the A/B is real.
    float kpf[UU][EPT], vpf[VEQK ? 1 : UU][EPT];
    bool have_pf = false;
    if (j + (UU - 1) * NW < k1) {
#pragma unroll
        for (int u = 0; u < UU; ++u) {
            const long eo = kvb_off<PAGED>(kvbase, j + u * NW, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT;
            load_raw<EPT, KVB>(K, eo, kpf[u]);
            if constexpr (!VEQK) load_raw<EPT, KVB>(V, eo, vpf[u]);
        }
        have_pf = true;
    }
#endif
    for (; j + (UU - 1) * NW < k1; j += UU * NW) {
        float part[GF][UU], khold[VEQK ? UU : 1][EPT];
        // [MLP] V IS ISSUED WITH K, not in the later phase where it is consumed.
        // Profiled cause: the decode kernel is memory-LATENCY bound, not bandwidth bound --
        // long_scoreboard 3.39-4.63 per active issue (89% of all stalls) against 0.05 in the forward
        // prefill, with DRAM at only 38-54% of peak. The old shape issued K, consumed it immediately
        // in the dot product, and only afterwards issued V, so at most UU K-loads were ever in
        // flight. V depends on nothing computed here, so hoisting it doubles the outstanding
        // requests per warp at a cost of UU*EPT registers -- and the arithmetic is untouched, so the
        // result stays bit-identical (the accumulation order does not move).
        float vhold[VEQK ? 1 : UU][EPT], kcur[UU][EPT];
#if FA2SM70_DEC_PF
        // take this group from the prefetch buffer, then IMMEDIATELY issue the next group's loads
#pragma unroll
        for (int u = 0; u < UU; ++u) {
#pragma unroll
            for (int e = 0; e < EPT; ++e) kcur[u][e] = kpf[u][e];
            if constexpr (!VEQK) {
#pragma unroll
                for (int e = 0; e < EPT; ++e) vhold[u][e] = vpf[u][e]; } }
        const int jn = j + UU * NW;
        if (jn + (UU - 1) * NW < k1) {
#pragma unroll
            for (int u = 0; u < UU; ++u) {
                const long eo = kvb_off<PAGED>(kvbase, jn + u * NW, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT;
                load_raw<EPT, KVB>(K, eo, kpf[u]);                            // in flight across the whole body
                if constexpr (!VEQK) load_raw<EPT, KVB>(V, eo, vpf[u]); } }
#else
#pragma unroll
        for (int u = 0; u < UU; ++u) { int jj = j + u * NW;
            const long eo = kvb_off<PAGED>(kvbase, jj, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT;
            load_raw<EPT, KVB>(K, eo, kcur[u]);                               // ONE K load -> GF dots
            if constexpr (!VEQK) {
#if FA2SM70_DEC_VK
                load_raw<EPT, KVB>(V, eo, vhold[u]);                          // ... and V alongside it
#endif
            } }
#endif
        #pragma unroll
        for (int u = 0; u < UU; ++u) {
            if constexpr (VEQK) {
                #pragma unroll
                for (int e = 0; e < EPT; ++e) khold[u][e] = kcur[u][e]; }     // keep for reuse as V
            #pragma unroll
            for (int g = 0; g < GF; ++g) { float pp = 0.f;
                #pragma unroll
                for (int e = 0; e < EPT; ++e) pp += qr[g][e] * kcur[u][e]; part[g][u] = pp; } }
        #pragma unroll
        for (int g = 0; g < GF; ++g)
            #pragma unroll
            for (int u = 0; u < UU; ++u) { for (int o = 16; o > 0; o >>= 1) part[g][u] += __shfl_xor_sync(0xffffffffu, part[g][u], o);
                if (sKb) part[g][u] *= sKb[s_off<PAGED>(j + u * NW, block_table, breq, hkv, block_size, max_blocks, Hkv)]; }
        if (tailmask) {                                   // ХВОСТ: ключ вне маски строки не участвует
            #pragma unroll
            for (int u = 0; u < UU; ++u) { const int t_ = (j + u * NW) - ctxb;
                if (t_ >= 0 && (t_ >= 32 || !((tmb >> t_) & 1u))) {
                    #pragma unroll
                    for (int g = 0; g < GF; ++g) part[g][u] = -CUDART_INF_F; } } }
        #pragma unroll
        for (int g = 0; g < GF; ++g) { float mo = m[g];       // rescale ONCE per unrolled group, if the max grew
            #pragma unroll
            for (int u = 0; u < UU; ++u) m[g] = fmaxf(m[g], part[g][u]);
            if (m[g] > mo) { const float corr = (mo == -CUDART_INF_F) ? 0.f : __expf(mo - m[g]);
                Bsum[g] *= corr;
                #pragma unroll
                for (int e = 0; e < EPT; ++e) acc[g][e] *= corr; } }
        #pragma unroll
        for (int u = 0; u < UU; ++u) { int jj = j + u * NW; float vsl[EPT];
            if constexpr (VEQK) {
                #pragma unroll
                for (int e = 0; e < EPT; ++e) vsl[e] = khold[u][e]; }         // V == K: no second load
            else {
#if FA2SM70_DEC_VK
                #pragma unroll
                for (int e = 0; e < EPT; ++e) vsl[e] = vhold[u][e];           // issued back with K, above
#else
                load_raw<EPT, KVB>(V, kvb_off<PAGED>(kvbase, jj, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT, vsl);
#endif
            }
            float vs = sVb ? sVb[s_off<PAGED>(jj, block_table, breq, hkv, block_size, max_blocks, Hkv)] : 1.f;
            #pragma unroll
            for (int g = 0; g < GF; ++g) { float p = FA2SM70_EXP_MASKED(part[g][u], m[g]); Bsum[g] += p;
                float pv = sVb ? p * vs : p;
                #pragma unroll
                for (int e = 0; e < EPT; ++e) acc[g][e] += pv * vsl[e]; } }
    }
    for (; j < k1; j += NW) {
        const long eoff = kvb_off<PAGED>(kvbase, j, d, block_table, breq, hkv, block_size, max_blocks, Hkv) + lane * EPT; float ksl[EPT];
        load_raw<EPT, KVB>(K, eoff, ksl);
        float part[GF];
        #pragma unroll
        for (int g = 0; g < GF; ++g) { float pp = 0.f;
            #pragma unroll
            for (int e = 0; e < EPT; ++e) pp += qr[g][e] * ksl[e]; part[g] = pp; }
        #pragma unroll
        for (int g = 0; g < GF; ++g) { for (int o = 16; o > 0; o >>= 1) part[g] += __shfl_xor_sync(0xffffffffu, part[g], o);
            if (sKb) part[g] *= sKb[s_off<PAGED>(j, block_table, breq, hkv, block_size, max_blocks, Hkv)]; }
        float vsl[EPT];
        if constexpr (VEQK) {
            #pragma unroll
            for (int e = 0; e < EPT; ++e) vsl[e] = ksl[e]; }                  // V == K: reuse the loaded key
        else load_raw<EPT, KVB>(V, eoff, vsl);
        if (tailmask) { const int t_ = j - ctxb;                                  // ХВОСТ, остаточный цикл
            if (t_ >= 0 && (t_ >= 32 || !((tmb >> t_) & 1u))) {
                #pragma unroll
                for (int g = 0; g < GF; ++g) part[g] = -CUDART_INF_F; } }
        float vs = sVb ? sVb[s_off<PAGED>(j, block_table, breq, hkv, block_size, max_blocks, Hkv)] : 1.f;
        #pragma unroll
        for (int g = 0; g < GF; ++g) { const float mo = m[g]; m[g] = fmaxf(m[g], part[g]);
            if (m[g] > mo) { const float corr = (mo == -CUDART_INF_F) ? 0.f : __expf(mo - m[g]);
                Bsum[g] *= corr;
                #pragma unroll
                for (int e = 0; e < EPT; ++e) acc[g][e] *= corr; } }
        #pragma unroll
        for (int g = 0; g < GF; ++g) { float p = FA2SM70_EXP_MASKED(part[g], m[g]); Bsum[g] += p;
            float pv = sVb ? p * vs : p;
            #pragma unroll
            for (int e = 0; e < EPT; ++e) acc[g][e] += pv * vsl[e]; }
    }
    }   // конец внешнего цикла по отобранным блокам
    #pragma unroll
    for (int g = 0; g < GF; ++g) {
        if (lane == 0) { sB[warp * GF + g] = Bsum[g]; sM[warp * GF + g] = m[g]; }
        #pragma unroll
        for (int e = 0; e < EPT; ++e)
            sA[(warp * GF + g) * 32 * EPT + lane * EPT + e] = SA_PUT(acc[g][e]);
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int g = 0; g < GF; ++g) {
            const int oidx = ((b * H + h0 + g) * ns + c);
            // Weighted cross-warp merge (warps hold different maxima now); -inf warps weigh 0 explicitly
            // because -inf minus -inf is NaN.
            float gm = -CUDART_INF_F;
            #pragma unroll
            for (int w = 0; w < NW; ++w) gm = fmaxf(gm, sM[w * GF + g]);
            float gB = 0.f, gA[EPT];
            #pragma unroll
            for (int e = 0; e < EPT; ++e) gA[e] = 0.f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) { const float cw = (sM[w * GF + g] == -CUDART_INF_F) ? 0.f : __expf(sM[w * GF + g] - gm);
                gB += cw * sB[w * GF + g];
                #pragma unroll
                for (int e = 0; e < EPT; ++e)
                    gA[e] += cw * SA_GET(sA[(w * GF + g) * 32 * EPT + lane * EPT + e]); }
            #pragma unroll
            for (int e = 0; e < EPT; ++e) A[(long)oidx * d + lane * EPT + e] = gA[e];
            if (lane == 0) { B[oidx] = gB; B[(long)gridDim.z * H * ns + oidx] = gm; }
        }
    }
}

static void launch_defer(int KVB, const half* q, const void* k, const void* v, const float* sk, const float* sv, const float* kmax,
                         float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d, float scale, int ns, int Lc, int kv_len, cudaStream_t st) {
    dim3 g1(ns, H, Bb); dim3 g2(H, Bb);
#define SPD(EPT) do { \
        if (KVB == 16) split_defer_kernel<EPT,16><<<g1,BLK,0,st>>>(q,k,v,sk,sv,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,nullptr,0,0,nullptr,nullptr); \
        else if (KVB == 8) split_defer_kernel<EPT,8><<<g1,BLK,0,st>>>(q,k,v,sk,sv,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,nullptr,0,0,nullptr,nullptr); \
        else split_defer_kernel<EPT,4><<<g1,BLK,0,st>>>(q,k,v,sk,sv,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,nullptr,0,0,nullptr,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
    if (d == 512) SPD(16); else if (d == 256) SPD(8); else if (d == 128) SPD(4); else SPD(2);
#undef SPD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// GF chosen per d to bound registers (qr/acc are GF*EPT each), then to the largest that divides the
// GQA group size H/Hkv. GF==1 means no fusion possible -> caller uses the per-head path.
static int pick_gf(int d, int group) {
    // Candidates must include NON-powers-of-two: this serving model's GQA group is 6 (H=24/Hkv=4,
    // and 12/2 per GPU at TP=2), which divides by neither 8 nor 4, so a {8,4,2} list fell to GF=2 and
    // read every K/V element THREE times. Cap: static smem is NW*32*EPT*GF floats = 4096*GF bytes at
    // d=256 (24.6 KB at GF=6) and 8192*GF at d=512, which is why d=512 stays low. The d=256 cap is 6
    // rather than 8 so that a group of 8 still picks 4 exactly as before -- one variable at a time.
    // FA2SM70_DEC_GF overrides the choice. GF is a TRAFFIC-vs-OCCUPANCY trade, not just a reuse
    // factor: it costs 2*GF*EPT registers (qr + acc), and at d=256/GF=6 that is 96 of the kernel's
    // 168 registers/thread, which caps residency at 3 blocks/SM and leaves warps active at 17.8%.
    // Halving GF doubles the K/V reads but frees registers -- and at this shape DRAM runs at only
    // 22.2% of peak, so the bandwidth to pay with is there. Swept, not assumed.
    if (const char* e = std::getenv("FA2SM70_DEC_GF")) {
        int g = atoi(e);
        if (g >= 1 && group % g == 0) return g;
    }
    int cap = (d == 512) ? 2 : (d == 256) ? 6 : 8;
    for (int gf : {8, 6, 4, 3, 2}) if (gf <= cap && group % gf == 0) return gf;
    return 1;
}

static void launch_defer_mqa(int KVB, const half* q, const void* k, const void* v, const float* sk, const float* sv, const float* kmax,
                             float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d, float scale, int ns, int Lc, int GF, bool veqk, int kv_len, cudaStream_t st) {
    dim3 g1(ns, H / GF, Bb); dim3 g2(H, Bb);
#define SPDM(EPT, GFV, VQ) do { \
        if (KVB == 16) split_defer_mqa_kernel<EPT,16,GFV,VQ><<<g1,BLK,0,st>>>(q,k,v,sk,sv,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,nullptr,0,0,nullptr,nullptr); \
        else if (KVB == 8) split_defer_mqa_kernel<EPT,8,GFV,VQ><<<g1,BLK,0,st>>>(q,k,v,sk,sv,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,nullptr,0,0,nullptr,nullptr); \
        else split_defer_mqa_kernel<EPT,4,GFV,VQ><<<g1,BLK,0,st>>>(q,k,v,sk,sv,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,nullptr,0,0,nullptr,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
#define SPDMV(EPT, GFV) do { if (veqk) SPDM(EPT,GFV,true); else SPDM(EPT,GFV,false); } while(0)
    // GF set is tied to EPT so static smem (NW*32*EPT*GF floats) stays <= 48KB: EPT16 caps at GF4.
    if (d == 512)      { if (GF == 4) SPDMV(16,4); else SPDMV(16,2); }
    else if (d == 256) { if (GF == 8) SPDMV(8,8); else if (GF == 6) SPDMV(8,6); else if (GF == 4) SPDMV(8,4); else if (GF == 3) SPDMV(8,3); else if (GF == 2) SPDMV(8,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else if (d == 128) { if (GF == 8) SPDMV(4,8); else if (GF == 6) SPDMV(4,6); else if (GF == 4) SPDMV(4,4); else if (GF == 3) SPDMV(4,3); else if (GF == 2) SPDMV(4,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else               { if (GF == 8) SPDMV(2,8); else if (GF == 6) SPDMV(2,6); else if (GF == 4) SPDMV(2,4); else if (GF == 3) SPDMV(2,3); else if (GF == 2) SPDMV(2,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
#undef SPDMV
#undef SPDM
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Paged variants (fp16 / KVB=16 only): identical algorithm, but K/V live in a block pool indexed by
// block_table (PAGED=true). GF>1 uses the fused MQA/GQA kernel; GF<=1 the per-head kernel.
static void launch_defer_mqa_paged(const half* q, const void* k, const void* v, const float* kmax,
                                   float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d,
                                   float scale, int ns, int Lc, int GF, bool veqk, int kv_len,
                                   const int* bt, int bs, int mb, const int* seqlens, cudaStream_t st) {
    dim3 g1(ns, H / GF, Bb); dim3 g2(H, Bb);
#define SPDMP(EPT, GFV, VQ) do { \
        split_defer_mqa_kernel<EPT,16,GFV,VQ,true><<<g1,BLK,0,st>>>(q,k,v,nullptr,nullptr,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,bt,bs,mb,seqlens,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
#define SPDMVP(EPT, GFV) do { if (veqk) SPDMP(EPT,GFV,true); else SPDMP(EPT,GFV,false); } while(0)
    if (d == 512)      { if (GF == 4) SPDMVP(16,4); else SPDMVP(16,2); }
    else if (d == 256) { if (GF == 8) SPDMVP(8,8); else if (GF == 6) SPDMVP(8,6); else if (GF == 4) SPDMVP(8,4); else if (GF == 3) SPDMVP(8,3); else if (GF == 2) SPDMVP(8,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else if (d == 128) { if (GF == 8) SPDMVP(4,8); else if (GF == 6) SPDMVP(4,6); else if (GF == 4) SPDMVP(4,4); else if (GF == 3) SPDMVP(4,3); else if (GF == 2) SPDMVP(4,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else               { if (GF == 8) SPDMVP(2,8); else if (GF == 6) SPDMVP(2,6); else if (GF == 4) SPDMVP(2,4); else if (GF == 3) SPDMVP(2,3); else if (GF == 2) SPDMVP(2,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
#undef SPDMVP
#undef SPDMP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [int8 paged] OUR quantisation in vLLM's byte pool: per-(token,kv-head) symmetric int8 with a side
// scale table [num_blocks, block_size, Hkv]. Strictly more accurate than e4m3 at the same 1 byte/elem
// (e4m3 spends 4 bits on an exponent it barely needs and keeps only 3 mantissa bits with ONE scalar
// per layer; a per-token scale adapts to each token's magnitude and keeps 8 bits of mantissa).
static void launch_defer_mqa_paged_i8(const half* q, const void* k, const void* v, const float* sK,
                                      const float* sV, const float* kmax,
                                      float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d,
                                      float scale, int ns, int Lc, int GF, bool veqk, int kv_len,
                                      const int* bt, int bs, int mb, const int* seqlens, cudaStream_t st) {
    dim3 g1(ns, H / GF, Bb); dim3 g2(H, Bb);
#define SPDMPI(EPT, GFV, VQ) do { \
        split_defer_mqa_kernel<EPT,8,GFV,VQ,true><<<g1,BLK,0,st>>>(q,k,v,sK,sV,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,bt,bs,mb,seqlens,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
#define SPDMVPI(EPT, GFV) do { if (veqk) SPDMPI(EPT,GFV,true); else SPDMPI(EPT,GFV,false); } while(0)
    if (d == 512)      { if (GF == 4) SPDMVPI(16,4); else SPDMVPI(16,2); }
    else if (d == 256) { if (GF == 8) SPDMVPI(8,8); else if (GF == 6) SPDMVPI(8,6); else if (GF == 4) SPDMVPI(8,4); else if (GF == 3) SPDMVPI(8,3); else if (GF == 2) SPDMVPI(8,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else if (d == 128) { if (GF == 8) SPDMVPI(4,8); else if (GF == 6) SPDMVPI(4,6); else if (GF == 4) SPDMVPI(4,4); else if (GF == 3) SPDMVPI(4,3); else if (GF == 2) SPDMVPI(4,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else               { if (GF == 8) SPDMVPI(2,8); else if (GF == 6) SPDMVPI(2,6); else if (GF == 4) SPDMVPI(2,4); else if (GF == 3) SPDMVPI(2,3); else if (GF == 2) SPDMVPI(2,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
#undef SPDMVPI
#undef SPDMPI
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_defer_paged_i8(const half* q, const void* k, const void* v, const float* sK,
                                  const float* sV, const float* kmax,
                                  float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d,
                                  float scale, int ns, int Lc, int kv_len, const int* bt, int bs, int mb,
                                  const int* seqlens, cudaStream_t st) {
    dim3 g1(ns, H, Bb); dim3 g2(H, Bb);
#define SPDPI(EPT) do { \
        split_defer_kernel<EPT,8,true><<<g1,BLK,0,st>>>(q,k,v,sK,sV,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,bt,bs,mb,seqlens,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
    if (d == 512) SPDPI(16); else if (d == 256) SPDPI(8); else if (d == 128) SPDPI(4); else SPDPI(2);
#undef SPDPI
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [fp8 paged] same as launch_defer_mqa_paged but reading an e4m3 (1 byte/elem) pool.
static void launch_defer_mqa_paged_f8(const half* q, const void* k, const void* v, const float* kmax,
                                      float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d,
                                      float scale, int ns, int Lc, int GF, bool veqk, int kv_len,
                                      const int* bt, int bs, int mb, const int* seqlens, cudaStream_t st) {
    dim3 g1(ns, H / GF, Bb); dim3 g2(H, Bb);
#define SPDMP8(EPT, GFV, VQ) do { \
        split_defer_mqa_kernel<EPT,87,GFV,VQ,true><<<g1,BLK,0,st>>>(q,k,v,nullptr,nullptr,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,bt,bs,mb,seqlens,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
#define SPDMVP8(EPT, GFV) do { if (veqk) SPDMP8(EPT,GFV,true); else SPDMP8(EPT,GFV,false); } while(0)
    if (d == 512)      { if (GF == 4) SPDMVP8(16,4); else SPDMVP8(16,2); }
    else if (d == 256) { if (GF == 8) SPDMVP8(8,8); else if (GF == 6) SPDMVP8(8,6); else if (GF == 4) SPDMVP8(8,4); else if (GF == 3) SPDMVP8(8,3); else if (GF == 2) SPDMVP8(8,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else if (d == 128) { if (GF == 8) SPDMVP8(4,8); else if (GF == 6) SPDMVP8(4,6); else if (GF == 4) SPDMVP8(4,4); else if (GF == 3) SPDMVP8(4,3); else if (GF == 2) SPDMVP8(4,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
    else               { if (GF == 8) SPDMVP8(2,8); else if (GF == 6) SPDMVP8(2,6); else if (GF == 4) SPDMVP8(2,4); else if (GF == 3) SPDMVP8(2,3); else if (GF == 2) SPDMVP8(2,2); else TORCH_CHECK(false, "decode: unsupported GF=", GF); }
#undef SPDMVP8
#undef SPDMP8
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_defer_paged_f8(const half* q, const void* k, const void* v, const float* kmax,
                                  float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d,
                                  float scale, int ns, int Lc, int kv_len, const int* bt, int bs, int mb,
                                  const int* seqlens, cudaStream_t st) {
    dim3 g1(ns, H, Bb); dim3 g2(H, Bb);
#define SPDP8(EPT) do { \
        split_defer_kernel<EPT,87,true><<<g1,BLK,0,st>>>(q,k,v,nullptr,nullptr,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,bt,bs,mb,seqlens,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
    if (d == 512) SPDP8(16); else if (d == 256) SPDP8(8); else if (d == 128) SPDP8(4); else SPDP8(2);
#undef SPDP8
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_defer_paged(const half* q, const void* k, const void* v, const float* kmax,
                               float* A, float* B, float* o, float* lse, int Bb, int H, int Hkv, int Sk, int d,
                               float scale, int ns, int Lc, int kv_len, const int* bt, int bs, int mb, const int* seqlens, cudaStream_t st) {
    dim3 g1(ns, H, Bb); dim3 g2(H, Bb);
#define SPDP(EPT) do { \
        split_defer_kernel<EPT,16,true><<<g1,BLK,0,st>>>(q,k,v,nullptr,nullptr,kmax,A,B,H,Hkv,Sk,d,scale,ns,Lc,kv_len,bt,bs,mb,seqlens,nullptr); \
        reduce_defer_kernel<EPT><<<g2,32,0,st>>>(A,B,q,kmax,o,lse,H,Hkv,d,ns,scale); } while(0)
    if (d == 512) SPDP(16); else if (d == 256) SPDP(8); else if (d == 128) SPDP(4); else SPDP(2);
#undef SPDP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Paged decode entry. Kcache/Vcache: block pool [num_blocks, block_size, Hkv, d]; block_table [B,
// max_blocks] int32 maps logical block -> physical block. kv_len = max valid length (Lc/ns sizing);
// seqlens (optional int32 [B]) gives PER-SEQUENCE valid lengths (splits past a sequence's length are
// empty and contribute 0 to the associative sum). VEQK auto-detected (Kcache is Vcache).
std::tuple<torch::Tensor, torch::Tensor> flash_decode_defer_mqa_paged_impl(
    torch::Tensor Q, torch::Tensor Kcache, torch::Tensor Vcache, torch::Tensor Kmax,
    torch::Tensor block_table, int64_t block_size, double scale, int64_t num_splits, int64_t kv_len,
    torch::Tensor seqlens, bool fp8, double k_scale, double v_scale) {
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    int Bb = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Hkv = Kcache.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Sq == 1 && (d == 64 || d == 128 || d == 256 || d == 512) && H % Hkv == 0);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table int32");
    if (fp8) TORCH_CHECK(Kcache.element_size() == 1 && Vcache.element_size() == 1, "fp8 paged: 1-byte KV pool");
    // [fp8] the pool holds k/k_scale, v/v_scale in e4m3. QK is linear in k, so k_scale folds into the
    // softmax scale (exactly); the value side is linear in v, so v_scale is applied once to the output.
    // Kmax must therefore be the bound on the RAW (unscaled) keys -- the caller keeps it in those units.
    if (fp8) scale *= k_scale;
    int mb = block_table.size(1);
    bool veqk = (Kcache.data_ptr() == Vcache.data_ptr() && Kcache.sizes() == Vcache.sizes());
    int valid = (int)kv_len;                                          // = max_j seqlens[j] (or the uniform length)
    // NEVER densify the pool here. A serving engine hands us a STRIDED view (vLLM stores KV as
    // [num_blocks, 2, block_size, Hkv, D] and passes kv_cache.unbind(1)), and .contiguous() on that
    // copies the whole ~640 MB layer pool on EVERY decode call -- measured 3.77 ms/call in-server
    // against 0.048 ms for this same kernel on a packed cache, i.e. the entire end-to-end regression.
    // Instead require the last three dims to be packed (they always are) and pass the BLOCK stride.
    TORCH_CHECK(Kcache.stride(3) == 1 && Kcache.stride(2) == d && Kcache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    TORCH_CHECK(Vcache.stride(0) == Kcache.stride(0), "paged KV: K and V need the same block stride");
    int64_t blk_stride_over_d = Kcache.stride(0) / d;                 // block_size*Hkv, or 2x that for unbind(1)
    auto Qc = Q.contiguous(); auto Kc = Kcache, Vc = Vcache;
    auto Km = Kmax.contiguous().to(torch::kFloat32), bt = block_table.contiguous();
    const int* slp = nullptr; torch::Tensor sl;
    if (seqlens.numel() > 0) { TORCH_CHECK(seqlens.scalar_type() == torch::kInt32 && seqlens.numel() == Bb, "seqlens int32 [B]");
        sl = seqlens.contiguous(); slp = sl.data_ptr<int32_t>(); }
    int ns = std::max<int>(1, std::min<int>(num_splits, (valid + 31) / 32)); int Lc = (valid + ns - 1) / ns;
    auto f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto A = torch::empty({Bb, H, ns, d}, f32), B = torch::empty({2, Bb, H, ns}, f32);   // [0]=sums, [1]=per-split maxima
    auto O = torch::empty({Bb, H, Sq, d}, f32), Lse = torch::empty({Bb, H}, f32);
    int GF = pick_gf(d, H / Hkv);
    const half* q = (const half*)Qc.data_ptr<at::Half>();
    const void* k = Kc.data_ptr(); const void* v = Vc.data_ptr();
    const int* btp = bt.data_ptr<int32_t>();
    // Sk (physical stride) is unused under paging (kvbase is not used); pass block_size as a placeholder.
    if (fp8) {
        if (GF <= 1)
            launch_defer_paged_f8(q, k, v, Km.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
                                  O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, (int)blk_stride_over_d, d,
                                  (float)scale, ns, Lc, valid, btp, (int)block_size, mb, slp, at::cuda::getCurrentCUDAStream());
        else
            launch_defer_mqa_paged_f8(q, k, v, Km.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
                                      O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, (int)blk_stride_over_d, d,
                                      (float)scale, ns, Lc, GF, veqk, valid, btp, (int)block_size, mb, slp, at::cuda::getCurrentCUDAStream());
        if (v_scale != 1.0) O.mul_((float)v_scale);     // value side is linear in v_scale
    } else if (GF <= 1)
        launch_defer_paged(q, k, v, Km.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
                           O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, (int)blk_stride_over_d, d,
                           (float)scale, ns, Lc, valid, btp, (int)block_size, mb, slp, at::cuda::getCurrentCUDAStream());
    else
        launch_defer_mqa_paged(q, k, v, Km.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
                               O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, (int)blk_stride_over_d, d,
                               (float)scale, ns, Lc, GF, veqk, valid, btp, (int)block_size, mb, slp, at::cuda::getCurrentCUDAStream());
    return std::make_tuple(O, Lse.reshape({Bb, H, 1}));
}

// [serving fast path] Zero-allocation decode for an engine: the caller owns the split workspace (A, B)
// and the output buffer, so a decode step allocates nothing and copies nothing. `Out` is fp16 with row
// stride `o_row_stride` (vLLM hands us a [tokens, H, D] view). This is the shape the incumbent's entry
// has -- matching it is what lets our faster kernel actually show up end to end.
void flash_decode_defer_mqa_paged_into(
    torch::Tensor Q, torch::Tensor Kcache, torch::Tensor Vcache, torch::Tensor Kmax,
    torch::Tensor block_table, int64_t block_size, double scale, int64_t num_splits, int64_t kv_len,
    torch::Tensor seqlens, bool fp8, double k_scale, double v_scale,
    torch::Tensor A, torch::Tensor B, torch::Tensor Out, int64_t o_row_stride,
    // ОПЦИОНАЛЬНЫЕ, И ИМЕННО std::optional, А НЕ torch::Tensor СО ЗНАЧЕНИЕМ ПО УМОЛЧАНИЮ.
    // Значение по умолчанию torch::Tensor() пробрасывается в Python как None, а None в этой сборке
    // torch НЕ приводится к torch::Tensor -- то есть все прежние вызовы на 17 аргументов (весь
    // e4m3-декод обвязки!) начали бы падать с TypeError. Поймано замером, а не рассуждением.
    std::optional<torch::Tensor> sK_opt, std::optional<torch::Tensor> sV_opt,
    // [ОКНО] Левая граница диапазона ключей на последовательность, int32 [B]. Отсутствует = ноль,
    // то есть ровно прежнее поведение -- ни один существующий вызов не меняет смысла. Тем же
    // std::optional, что и таблицы масштабов, и по той же причине: None не приводится к
    // torch::Tensor, а значит значение по умолчанию torch::Tensor() сломало бы старые вызовы.
    std::optional<torch::Tensor> kv_start_opt,
    // [ДЕРЕВО] Граница «общий контекст | хвост шага» на строку (int32 [B]) и битовая маска хвоста
    // (uint32 [B]). Обе отсутствуют = прежнее поведение до бита: цепь выражается длиной префикса.
    std::optional<torch::Tensor> ctxlen_opt, std::optional<torch::Tensor> tailmask_opt,
    // [РАЗРЕЖЕННЫЙ ОБХОД -- шаг 2] Индекс отобранных блоков ключей на (строку, KV-голову).
    // Отсутствует = плотный путь, БАЙТ В БАЙТ прежний (проверено побайтовой сверкой выхода).
    std::optional<torch::Tensor> selCnt_opt, std::optional<torch::Tensor> selIdx_opt,
    int64_t selBlk = 128,
    // [ВИРТУАЛЬНЫЙ БАТЧ БЕЗ КОПИЙ] qGrp -- строк спекуляции на ЗАПРОС. Таблицу блоков и длины
    // подаём ИСХОДНЫЕ (на запрос), ядро само берёт p = r/qGrp и длину sl[p]-(qGrp-1-r%qGrp).
    // Раньше хозяин раскладывал их копиями: два запуска и две копии на шаг. qGrp=1 -- прежнее
    // поведение бит в бит. qCausal=0 -- двунаправленный блок черновика (длина у всех строк одна).
    int64_t qGrp = 1, int64_t qCausal = 1) {
    TORCH_CHECK(qGrp >= 1, "qGrp >= 1");
    const torch::Tensor selCnt_t = selCnt_opt.has_value() ? *selCnt_opt : torch::Tensor();
    const torch::Tensor selIdx_t = selIdx_opt.has_value() ? *selIdx_opt : torch::Tensor();
    const int* sCp = nullptr; const int* sIp = nullptr; int sStr = 0, sBlg = 7;
    if (selIdx_t.defined()) {
        TORCH_CHECK(selCnt_t.defined(), "индекс отбора требует счётчиков");
        TORCH_CHECK(selIdx_t.scalar_type() == torch::kInt32 && selCnt_t.scalar_type() == torch::kInt32
                    && selIdx_t.is_contiguous() && selCnt_t.is_contiguous(),
                    "индекс отбора: int32 плотные");
        TORCH_CHECK((selBlk & (selBlk - 1)) == 0, "блок отбора -- степень двойки: ", selBlk);
        sBlg = 0; while ((1LL << sBlg) < selBlk) ++sBlg;   // ядро СДВИГАЕТ, а не делит
        sIp = selIdx_t.data_ptr<int32_t>(); sCp = selCnt_t.data_ptr<int32_t>();
        sStr = (int)selIdx_t.size(-1);
    }
    const torch::Tensor ctxlen_t = ctxlen_opt.has_value() ? *ctxlen_opt : torch::Tensor();
    const torch::Tensor tailmask_t = tailmask_opt.has_value() ? *tailmask_opt : torch::Tensor();
    const int* ctp = ctxlen_t.defined() ? ctxlen_t.data_ptr<int32_t>() : nullptr;
    const unsigned* tmp_ = tailmask_t.defined()
        ? reinterpret_cast<const unsigned*>(tailmask_t.data_ptr<int32_t>()) : nullptr;
    TORCH_CHECK(!tailmask_t.defined() || ctxlen_t.defined(), "маска хвоста требует ctxlen");
    const torch::Tensor sK = sK_opt.has_value() ? *sK_opt : torch::Tensor();
    const torch::Tensor sV = sV_opt.has_value() ? *sV_opt : torch::Tensor();
    const torch::Tensor kv_start = kv_start_opt.has_value() ? *kv_start_opt : torch::Tensor();
    const c10::cuda::CUDAGuard device_guard(Q.device());
    int Bb = Q.size(0), H = Q.size(1), d = Q.size(3), Hkv = Kcache.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Q.size(2) == 1);
    // Диспетчер ниже кончается на `else -> EPT=2`, то есть НЕПОДДЕРЖАННЫЙ d уходил в ядро для d=64
    // и возвращал мусор МОЛЧА. У соседнего входа (`_impl`) эта проверка есть; здесь её не было, и
    // именно этот вход обслуживает бэкенд vLLM -- то есть дыра лежала ровно на боевом пути.
    TORCH_CHECK(d == 64 || d == 128 || d == 256 || d == 512, "paged decode: d in {64,128,256,512}, got ", d);
    TORCH_CHECK(Out.scalar_type() == torch::kFloat16, "Out must be fp16");
    TORCH_CHECK(Kcache.stride(3) == 1 && Kcache.stride(2) == d && Kcache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    // [ФОРМАТ ЯВНЫЙ] Признак формата -- НАЛИЧИЕ таблиц масштабов, а не флаг рядом с данными: пустые
    // таблицы физически невозможно перепутать с заполненными, а два формата одновременно отвергаются
    // ПАДЕНИЕМ. int8-пул НЕ УЧАСТВУЕТ в k_scale/v_scale: масштаб на позицию уже несёт истинные
    // единицы, поэтому ни свёртки k_scale в softmax-масштаб, ни умножения выхода на v_scale здесь
    // быть не должно -- и если бы они остались, ошибка была бы ТИХОЙ.
    const bool i8 = sK.defined() && sK.numel() > 0;
    TORCH_CHECK(!(i8 && fp8), "paged decode: int8 (scale tables) and fp8_e4m3 are DIFFERENT pool formats; pass one");
    if (i8) {
        TORCH_CHECK(Kcache.element_size() == 1, "int8 paged: 1-byte KV pool");
        TORCH_CHECK(sK.scalar_type() == torch::kFloat32 && sV.scalar_type() == torch::kFloat32
                    && sK.is_contiguous() && sV.is_contiguous(), "int8 paged: fp32 contiguous scale tables");
        TORCH_CHECK(sK.numel() >= (long)Kcache.size(0) * block_size * Hkv
                    && sV.numel() >= (long)Kcache.size(0) * block_size * Hkv,
                    "int8 paged: scale table must be [num_blocks, block_size, Hkv]");
    }
    if (fp8) { TORCH_CHECK(Kcache.element_size() == 1, "fp8 paged: 1-byte KV pool"); scale *= k_scale; }
    int mb = block_table.size(1), valid = (int)kv_len;
    // [ВИРТУАЛЬНЫЙ БАТЧ БЕЗ КОПИЙ] Предусловия формы: при qGrp>1 строк ВТРОЕ/ВЧЕТВЕРО больше,
    // чем записей в таблице и длинах. Несоответствие обязано быть ОТКАЗОМ, а не выходом за
    // границу: тихая порча тут выглядит как «незаконный доступ» глубоко в воркере.
    TORCH_CHECK((long)block_table.size(0) * qGrp >= (long)Bb,
                "qGrp=", qGrp, ": строк ", Bb, ", а таблица блоков на ", block_table.size(0),
                " запросов -- не хватает");
    TORCH_CHECK(seqlens.numel() == 0 || (long)seqlens.numel() * qGrp >= (long)Bb,
                "qGrp=", qGrp, ": строк ", Bb, ", а длин ", seqlens.numel(), " -- не хватает");
    int64_t blk = Kcache.stride(0) / d;
    bool veqk = (Kcache.data_ptr() == Vcache.data_ptr() && Kcache.sizes() == Vcache.sizes());
    int ns = std::max<int>(1, std::min<int>(num_splits, (valid + 31) / 32));
    int Lc = (valid + ns - 1) / ns;
    TORCH_CHECK(A.numel() >= (long)Bb * H * ns * d && B.numel() >= 2 * (long)Bb * H * ns,
                "workspace too small (B needs 2*B*H*ns: sums + per-split maxima)");
    const int* slp = seqlens.numel() > 0 ? seqlens.data_ptr<int32_t>() : nullptr;
    // [ОКНО] Начало без КОНЦА бессмысленно: без seqlens ядро берёт длину из kv_len (ёмкости таблицы)
    // и делит [0, kv_len) на срезы, поэтому сдвиг начала дал бы диапазон, не совпадающий ни с одной
    // последовательностью. Отвергаем ПАДЕНИЕМ, а не молча.
    const int* stp = nullptr;
    if (kv_start.defined() && kv_start.numel() > 0) {
        TORCH_CHECK(kv_start.scalar_type() == torch::kInt32 && kv_start.is_contiguous()
                    && (long)kv_start.numel() * qGrp >= (long)Bb,
                    "kv_start: int32 contiguous [B/qGrp]");
        TORCH_CHECK(slp != nullptr, "kv_start requires per-sequence seqlens");
        stp = kv_start.data_ptr<int32_t>();
    }
    int GF = pick_gf(d, H / Hkv);
    const half* q = (const half*)Q.data_ptr<at::Half>();
    half* op = (half*)Out.data_ptr<at::Half>();
    float* ap = A.data_ptr<float>(); float* bp = B.data_ptr<float>();
    auto st = at::cuda::getCurrentCUDAStream();
    dim3 g1(ns, H / GF, Bb), g1s(ns, H, Bb), g2(H, Bb);
    const float* km = Kmax.data_ptr<float>();
    const void* kp = Kcache.data_ptr(); const void* vp = Vcache.data_ptr();
    const float* skp = i8 ? sK.data_ptr<float>() : nullptr;
    const float* svp = i8 ? sV.data_ptr<float>() : nullptr;
#define RUNK(EPT, KVB, GFV, VQ)                                                                       \
    do { if (GF <= 1) split_defer_kernel<EPT,KVB,true><<<g1s,BLK,0,st>>>(                             \
             q,kp,vp,skp,svp,km,ap,bp,H,Hkv,(int)blk,d,(float)scale,ns,Lc,valid,                      \
             block_table.data_ptr<int32_t>(),(int)block_size,mb,slp,stp,(int)qGrp,(int)qCausal);   \
         else split_defer_mqa_kernel<EPT,KVB,GFV,VQ,true><<<g1,BLK,0,st>>>(                           \
             q,kp,vp,skp,svp,km,ap,bp,H,Hkv,(int)blk,d,(float)scale,ns,Lc,valid,                      \
             block_table.data_ptr<int32_t>(),(int)block_size,mb,slp,stp,ctp,tmp_,sIp,sCp,sStr,sBlg,  \
             (int)qGrp,(int)qCausal);                                                                 \
         reduce_defer_out_kernel<EPT><<<g2,256,0,st>>>(ap,bp,op,H,d,ns,(long)o_row_stride,          \
                                                      fp8 ? (float)v_scale : 1.0f); } while (0)
#define RUNV(EPT, KVB, GFV) do { if (veqk) RUNK(EPT,KVB,GFV,true); else RUNK(EPT,KVB,GFV,false); } while (0)
// d=512 (EPT=16) cannot host GF=8: the fused accumulators exceed Volta's 48 KB smem (ptxas: 0x10080 >
// 0xc000). Mirror the shipped launcher's per-d GF menu exactly.
// GF<=1 (MHA: H == Hkv, нечего сплавлять) выбирает ПОЯДЕРНОЕ ядро уже внутри RUNK, поэтому шаблонный
// аргумент GFV там не используется -- отправляем в инстанциацию GFV=2, а не отказываем. Раньше здесь
// стоял TORCH_CHECK(false), то есть этот вход просто не мог обслужить MHA-модель, хотя ядро, в которое
// он диспетчеризует, этот случай умеет; у боевой модели GF=6, поэтому дыра не проявлялась.
#define RUNG(EPT, KVB) do { if (GF == 8) RUNV(EPT,KVB,8); else if (GF == 6) RUNV(EPT,KVB,6); \
    else if (GF == 4) RUNV(EPT,KVB,4); else if (GF == 3) RUNV(EPT,KVB,3); else RUNV(EPT,KVB,2); } while (0)
#define RUNG512(KVB)   do { if (GF == 4) RUNV(16,KVB,4); else RUNV(16,KVB,2); } while (0)
    if (i8)       { if (d == 512) RUNG512(8);  else if (d == 256) RUNG(8,8);  else if (d == 128) RUNG(4,8);  else RUNG(2,8); }
    else if (fp8) { if (d == 512) RUNG512(87); else if (d == 256) RUNG(8,87); else if (d == 128) RUNG(4,87); else RUNG(2,87); }
    else          { if (d == 512) RUNG512(16); else if (d == 256) RUNG(8,16); else if (d == 128) RUNG(4,16); else RUNG(2,16); }
#undef RUNG512
#undef RUNG
#undef RUNV
#undef RUNK
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [serving] OUR e4m3 store, so the shim owns the whole KV path and never delegates.
void reshape_and_cache_e4m3(torch::Tensor key, torch::Tensor value,
                            torch::Tensor key_cache, torch::Tensor value_cache,
                            torch::Tensor slot_mapping, double k_scale, double v_scale,
                            torch::Tensor kbound, double margin) {
    const c10::cuda::CUDAGuard device_guard(key.device());
    TORCH_CHECK(key.scalar_type() == torch::kFloat16 && value.scalar_type() == torch::kFloat16, "K/V fp16");
    TORCH_CHECK(key_cache.element_size() == 1 && value_cache.element_size() == 1, "e4m3 pool is 1 byte/elem");
    auto k = key.contiguous(), v = value.contiguous();
    const int d = (int)key_cache.size(3), Hkv = (int)key_cache.size(2), bs = (int)key_cache.size(1);
    const int n = (int)(k.numel() / (Hkv * d));
    TORCH_CHECK(key_cache.stride(3) == 1 && key_cache.stride(2) == d && key_cache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    auto sl = slot_mapping.to(torch::kLong).contiguous();
    const int thr = d >= 256 ? 256 : 128;
    dim3 g(n, Hkv);
    reshape_and_cache_e4m3_kernel<<<g, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)k.data_ptr<at::Half>(), (const half*)v.data_ptr<at::Half>(),
        (uint8_t*)key_cache.data_ptr(), (uint8_t*)value_cache.data_ptr(),
        sl.data_ptr<long>(), n, Hkv, d, key_cache.stride(0), bs,
        1.0f / (float)k_scale, 1.0f / (float)v_scale,
        kbound.data_ptr<float>(), (float)margin);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [serving] OUR int8 store into the engine's byte pool + the per-(token,kv-head) scale side table.
// Ks/Vs are fp32 [num_blocks, block_size, Hkv] (dense, OUR allocation), which is exactly the layout
// s_off() walks in the decode kernel and gather_paged_kv_i8_pool walks in the prefill gather. The
// engine's k_scale/v_scale are deliberately NOT parameters: on this path the scale IS the table.
void reshape_and_cache_i8(torch::Tensor key, torch::Tensor value,
                          torch::Tensor key_cache, torch::Tensor value_cache,
                          torch::Tensor slot_mapping, torch::Tensor k_scales, torch::Tensor v_scales) {
    const c10::cuda::CUDAGuard device_guard(key.device());
    TORCH_CHECK(key.scalar_type() == torch::kFloat16 && value.scalar_type() == torch::kFloat16, "K/V fp16");
    TORCH_CHECK(key_cache.element_size() == 1 && value_cache.element_size() == 1, "int8 pool is 1 byte/elem");
    TORCH_CHECK(k_scales.scalar_type() == torch::kFloat32 && v_scales.scalar_type() == torch::kFloat32,
                "int8 store: scale tables fp32");
    TORCH_CHECK(k_scales.is_contiguous() && v_scales.is_contiguous(), "int8 store: scale tables contiguous");
    auto k = key.contiguous(), v = value.contiguous();
    const int d = (int)key_cache.size(3), Hkv = (int)key_cache.size(2), bs = (int)key_cache.size(1);
    const int nblk = (int)key_cache.size(0);
    const int n = (int)(k.numel() / (Hkv * d));
    TORCH_CHECK(key_cache.stride(3) == 1 && key_cache.stride(2) == d && key_cache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    TORCH_CHECK(k_scales.numel() >= (long)nblk * bs * Hkv && v_scales.numel() >= (long)nblk * bs * Hkv,
                "int8 store: scale table must be [num_blocks, block_size, Hkv]");
    TORCH_CHECK(d <= 1024, "int8 store: one row per block of 256 threads -> d <= 1024");
    auto sl = slot_mapping.to(torch::kLong).contiguous();
    const int thr = 256;
    dim3 g(n, Hkv);
    reshape_and_cache_i8_kernel<<<g, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)k.data_ptr<at::Half>(), (const half*)v.data_ptr<at::Half>(),
        reinterpret_cast<int8_t*>(key_cache.data_ptr()), reinterpret_cast<int8_t*>(value_cache.data_ptr()),
        sl.data_ptr<long>(), n, Hkv, d, key_cache.stride(0), bs,
        k_scales.data_ptr<float>(), v_scales.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [serving] int8 pool -> int8 [Hkv,T,d] + fp32 [Hkv,T] (the role kernel's own format). EXACT.
void gather_paged_kv_i8_pool(torch::Tensor key_cache, torch::Tensor value_cache,
                             torch::Tensor k_scales, torch::Tensor v_scales,
                             torch::Tensor block_table, int64_t seq_len,
                             torch::Tensor out_kq, torch::Tensor out_ks,
                             torch::Tensor out_vq, torch::Tensor out_vs) {
    const c10::cuda::CUDAGuard device_guard(key_cache.device());
    const int d = (int)key_cache.size(3), Hkv = (int)key_cache.size(2), bs = (int)key_cache.size(1);
    const int nblk = (int)key_cache.size(0), T = (int)seq_len;
    TORCH_CHECK(key_cache.element_size() == 1 && value_cache.element_size() == 1, "int8 pool is 1 byte/elem");
    TORCH_CHECK(key_cache.stride(3) == 1 && key_cache.stride(2) == d && key_cache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    TORCH_CHECK(out_kq.is_contiguous() && out_vq.is_contiguous() && out_ks.is_contiguous() && out_vs.is_contiguous(),
                "gather_i8_pool: dest contiguous");
    TORCH_CHECK(out_kq.numel() >= (long)Hkv * T * d && out_ks.numel() >= (long)Hkv * T, "gather_i8_pool: dest too small");
    TORCH_CHECK(out_ks.scalar_type() == torch::kFloat32 && out_vs.scalar_type() == torch::kFloat32,
                "gather_i8_pool: масштабы в fp32 -- этого формата ждёт attn_fwd_volta_i8");
    TORCH_CHECK(k_scales.numel() >= (long)nblk * bs * Hkv && v_scales.numel() >= (long)nblk * bs * Hkv,
                "gather_i8_pool: scale table too small");
    TORCH_CHECK((long)((T + bs - 1) / bs) <= block_table.numel(), "block table shorter than seq_len");
    auto bt = block_table.to(torch::kInt32).contiguous();
    dim3 g(T, Hkv);
    gather_paged_kv_i8_pool_kernel<<<g, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int8_t*>(key_cache.data_ptr()), reinterpret_cast<const int8_t*>(value_cache.data_ptr()),
        k_scales.data_ptr<float>(), v_scales.data_ptr<float>(), bt.data_ptr<int>(),
        reinterpret_cast<int8_t*>(out_kq.data_ptr()), out_ks.data_ptr<float>(),
        reinterpret_cast<int8_t*>(out_vq.data_ptr()), out_vs.data_ptr<float>(),
        T, Hkv, d, key_cache.stride(0), bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [serving] int8 pool -> fp16 [Hkv,T,d] (fallback for shapes the role kernel does not take).
void gather_paged_kv_i8_pool_f16(torch::Tensor key_cache, torch::Tensor value_cache,
                                 torch::Tensor k_scales, torch::Tensor v_scales,
                                 torch::Tensor block_table, int64_t seq_len,
                                 torch::Tensor out_k, torch::Tensor out_v) {
    const c10::cuda::CUDAGuard device_guard(key_cache.device());
    const int d = (int)key_cache.size(3), Hkv = (int)key_cache.size(2), bs = (int)key_cache.size(1);
    const int nblk = (int)key_cache.size(0), T = (int)seq_len;
    TORCH_CHECK(key_cache.element_size() == 1 && value_cache.element_size() == 1, "int8 pool is 1 byte/elem");
    TORCH_CHECK(key_cache.stride(3) == 1 && key_cache.stride(2) == d && key_cache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    TORCH_CHECK(out_k.is_contiguous() && out_v.is_contiguous(), "gather dest must be contiguous");
    TORCH_CHECK(out_k.numel() >= (long)Hkv * T * d, "gather dest too small");
    TORCH_CHECK(k_scales.numel() >= (long)nblk * bs * Hkv && v_scales.numel() >= (long)nblk * bs * Hkv,
                "gather_i8_pool_f16: scale table too small");
    TORCH_CHECK((long)((T + bs - 1) / bs) <= block_table.numel(), "block table shorter than seq_len");
    auto bt = block_table.to(torch::kInt32).contiguous();
    const int thr = d >= 256 ? 256 : 128;
    dim3 g(T, Hkv);
    gather_paged_kv_i8f16_kernel<<<g, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int8_t*>(key_cache.data_ptr()), reinterpret_cast<const int8_t*>(value_cache.data_ptr()),
        k_scales.data_ptr<float>(), v_scales.data_ptr<float>(), bt.data_ptr<int>(),
        (half*)out_k.data_ptr<at::Half>(), (half*)out_v.data_ptr<at::Half>(),
        T, Hkv, d, key_cache.stride(0), bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static constexpr int KROWS = 64;               // k-rows per block for the dequant kernel
// [serving] GPTQ 8-bit weight -> fp16, for the BATCHED-DECODE window only.
// vLLM's gptq_gemm (exllama) is excellent at M=1, where it is memory-bound and fuses the dequant,
// and fine again once M is large. In between -- exactly the batched-decode range -- it loses badly to
// a plain fp16 GEMM. Measured on this model's MLP shape (K=5120, N=17408, V100, ms):
//     M        1      64    1024    4096   16384
//   gptq   0.124   0.887   2.629   8.343  30.626
//   fp16   0.418   0.245   2.025   8.083  33.339
// Dequantising once and calling cuBLAS is therefore worth it only in a WINDOW. Measured end to end
// with the kernel below (K=5120, N=17408, ms):
//     M         1      8     16     32     64    128    512   1024   4096
//   gptq    0.118  0.266  0.495  0.872  0.883  0.918  1.609  2.497  8.263
//   deq+gemm 0.834  0.647  0.650  0.650  0.651  0.694  1.448  2.402  8.539
//   gain     0.14   0.41   0.76   1.34   1.36   1.32   1.11   1.04   0.97
// So: a real 1.32-1.36x for M in [32, 128], nothing above ~512, and a 2.4-7x LOSS below M=16.
// NOT wired in by default, deliberately: this box serves 4-5 concurrent requests, i.e. M~5 per decode
// step, which is squarely in the region where this path HURTS. It is here for large-batch serving.
// Safe because gptq_shuffle is a VERIFIED NO-OP at 8 bits with desc_act=false (it only permutes for
// act-order), so the stored qweight is still in plain AutoGPTQ packing here.
__global__ void gptq_w8_dequant_kernel(const int32_t* __restrict__ qw, const int32_t* __restrict__ qz,
                                       const half* __restrict__ sc, half* __restrict__ out,
                                       int K, int N, int group) {
    // One thread owns a COLUMN STRIP of 2 outputs and walks k, so the per-group zero/scale lookups
    // amortise over the whole column instead of being redone per element, and the store is a half2.
    // The first version did one element per thread with a (N/256, K) grid -- 348K blocks each doing a
    // single fp16 store -- and reached only 351 GB/s of ~900.
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (n0 >= N) return;
    const int kb = blockIdx.y * KROWS;
    const int kend = min(kb + KROWS, K);
    int gprev = -1; float s0 = 0.f, s1 = 0.f; int z0 = 0, z1 = 0;
    for (int k = kb; k < kend; ++k) {
        const int g = k / group;
        if (g != gprev) {                                   // group boundary: refresh z and scale
            gprev = g;
            const int zw = qz[(long)g * (N >> 2) + (n0 >> 2)];
            z0 = ((zw >> (8 * (n0 & 3))) & 0xFF) + 1;
            z1 = ((zw >> (8 * ((n0 + 1) & 3))) & 0xFF) + 1;
            s0 = __half2float(sc[(long)g * N + n0]);
            s1 = __half2float(sc[(long)g * N + n0 + 1]);
        }
        const int32_t* qrow = qw + (long)(k >> 2) * N;
        const int sh = 8 * (k & 3);
        const int q0 = (qrow[n0] >> sh) & 0xFF;
        const int q1 = (qrow[n0 + 1] >> sh) & 0xFF;
        *reinterpret_cast<half2*>(out + (long)k * N + n0) =
            __halves2half2(__float2half((float)(q0 - z0) * s0), __float2half((float)(q1 - z1) * s1));
    }
}

void gptq_dequant_w8(torch::Tensor qweight, torch::Tensor qzeros, torch::Tensor scales,
                     torch::Tensor out) {
    const c10::cuda::CUDAGuard device_guard(qweight.device());
    const int K = (int)qweight.size(0) * 4, N = (int)qweight.size(1);
    const int ng = (int)scales.size(0);
    TORCH_CHECK(K % ng == 0, "K must be a multiple of the group count");
    TORCH_CHECK(N % 4 == 0, "N must be a multiple of 4 (qzeros packs 4 per int32)");
    TORCH_CHECK(out.numel() >= (long)K * N, "dequant scratch too small");
    const int group = K / ng;
    const int thr = 128;                       // 2 columns per thread -> 256 columns per block
    dim3 g((N / 2 + thr - 1) / thr, (K + KROWS - 1) / KROWS);
    gptq_w8_dequant_kernel<<<g, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        qweight.data_ptr<int32_t>(), qzeros.data_ptr<int32_t>(),
        (const half*)scales.data_ptr<at::Half>(), (half*)out.data_ptr<at::Half>(), K, N, group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [serving] Gather one sequence's paged KV into contiguous fp16 [Hkv, T, d] (BHSD, ready for our
// prefill kernel with no transpose). Caller owns the destination, so a persistent buffer can be
// reused across layers and chunks; nothing here allocates or synchronises.
void gather_paged_kv(torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor block_table,
                     int64_t seq_len, double k_scale, double v_scale,
                     torch::Tensor out_k, torch::Tensor out_v) {
    const c10::cuda::CUDAGuard device_guard(key_cache.device());
    const int d = (int)key_cache.size(3), Hkv = (int)key_cache.size(2), bs = (int)key_cache.size(1);
    const int T = (int)seq_len;
    TORCH_CHECK(key_cache.stride(3) == 1 && key_cache.stride(2) == d && key_cache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    TORCH_CHECK(out_k.is_contiguous() && out_v.is_contiguous(), "gather dest must be contiguous");
    TORCH_CHECK(out_k.numel() >= (long)Hkv * T * d, "gather dest too small");
    TORCH_CHECK((long)((T + bs - 1) / bs) <= block_table.numel(), "block table shorter than seq_len");
    auto bt = block_table.to(torch::kInt32).contiguous();
    const int thr = d >= 256 ? 256 : 128;
    dim3 g(T, Hkv);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (key_cache.element_size() == 1) {
        gather_paged_kv_kernel<87><<<g, thr, 0, stream>>>(
            key_cache.data_ptr(), value_cache.data_ptr(), bt.data_ptr<int>(),
            (half*)out_k.data_ptr<at::Half>(), (half*)out_v.data_ptr<at::Half>(),
            T, Hkv, d, key_cache.stride(0), bs, (float)k_scale, (float)v_scale);
    } else {
        TORCH_CHECK(key_cache.scalar_type() == torch::kFloat16, "gather: fp16 or 1-byte pool");
        gather_paged_kv_kernel<16><<<g, thr, 0, stream>>>(
            key_cache.data_ptr(), value_cache.data_ptr(), bt.data_ptr<int>(),
            (half*)out_k.data_ptr<at::Half>(), (half*)out_v.data_ptr<at::Half>(),
            T, Hkv, d, key_cache.stride(0), bs, 1.0f, 1.0f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Тот же обход страниц, но выход -- байты с масштабом на позицию (формат ядра ролей).
void gather_paged_kv_i8(torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor block_table,
                        int64_t seq_len, double k_scale, double v_scale,
                        torch::Tensor out_kq, torch::Tensor out_ks,
                        torch::Tensor out_vq, torch::Tensor out_vs) {
    const c10::cuda::CUDAGuard device_guard(key_cache.device());
    const int d = (int)key_cache.size(3), Hkv = (int)key_cache.size(2), bs = (int)key_cache.size(1);
    const int T = (int)seq_len;
    TORCH_CHECK(key_cache.stride(3) == 1 && key_cache.stride(2) == d && key_cache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    TORCH_CHECK(d <= 256, "gather_i8: одна строка = один блок нитей, поэтому d <= 256");
    TORCH_CHECK(out_kq.is_contiguous() && out_vq.is_contiguous() && out_ks.is_contiguous() && out_vs.is_contiguous(),
                "gather_i8 dest must be contiguous");
    TORCH_CHECK(out_kq.numel() >= (long)Hkv * T * d && out_ks.numel() >= (long)Hkv * T, "gather_i8 dest too small");
    TORCH_CHECK((long)((T + bs - 1) / bs) <= block_table.numel(), "block table shorter than seq_len");
    auto bt = block_table.to(torch::kInt32).contiguous();
    const int thr = 256;                       // d <= 256: не больше одного значения на нить
    const size_t sm = 64 * sizeof(float);
    dim3 g(T, Hkv);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* kq = reinterpret_cast<int8_t*>(out_kq.data_ptr());
    auto* vq = reinterpret_cast<int8_t*>(out_vq.data_ptr());
    TORCH_CHECK(out_ks.scalar_type() == torch::kFloat32 && out_vs.scalar_type() == torch::kFloat32,
                "gather_i8: масштабы в fp32 -- этого формата ждёт attn_fwd_volta_i8");
    auto* ks = out_ks.data_ptr<float>();
    auto* vs = out_vs.data_ptr<float>();
    if (key_cache.element_size() == 1) {
        gather_paged_kv_i8_kernel<87><<<g, thr, sm, stream>>>(
            key_cache.data_ptr(), value_cache.data_ptr(), bt.data_ptr<int>(), kq, ks, vq, vs,
            T, Hkv, d, key_cache.stride(0), bs, (float)k_scale, (float)v_scale);
    } else {
        TORCH_CHECK(key_cache.scalar_type() == torch::kFloat16, "gather_i8: fp16 or 1-byte pool");
        gather_paged_kv_i8_kernel<16><<<g, thr, sm, stream>>>(
            key_cache.data_ptr(), value_cache.data_ptr(), bt.data_ptr<int>(), kq, ks, vq, vs,
            T, Hkv, d, key_cache.stride(0), bs, 1.0f, 1.0f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_paged_kv_bytes(torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor block_table,
                          int64_t seq_len, torch::Tensor out_k, torch::Tensor out_v) {
    const c10::cuda::CUDAGuard device_guard(key_cache.device());
    const int d = (int)key_cache.size(3), Hkv = (int)key_cache.size(2), bs = (int)key_cache.size(1);
    const int T = (int)seq_len;
    TORCH_CHECK(key_cache.element_size() == 1 && value_cache.element_size() == 1, "gather_bytes: однобайтовый пул");
    TORCH_CHECK(key_cache.stride(3) == 1 && key_cache.stride(2) == d && key_cache.stride(1) == (long)Hkv * d,
                "paged KV: [.., block_size, Hkv, d] must be packed within a block");
    TORCH_CHECK(out_k.is_contiguous() && out_v.is_contiguous() && out_k.element_size() == 1, "gather_bytes: dest");
    TORCH_CHECK(out_k.numel() >= (long)Hkv * T * d, "gather_bytes: dest too small");
    TORCH_CHECK((long)((T + bs - 1) / bs) <= block_table.numel(), "block table shorter than seq_len");
    auto bt = block_table.to(torch::kInt32).contiguous();
    dim3 g(T, Hkv);
    gather_paged_kv_bytes_kernel<<<g, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(key_cache.data_ptr()), reinterpret_cast<const uint8_t*>(value_cache.data_ptr()),
        bt.data_ptr<int>(), reinterpret_cast<uint8_t*>(out_k.data_ptr()), reinterpret_cast<uint8_t*>(out_v.data_ptr()),
        T, Hkv, d, key_cache.stride(0), bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Back-compat wrapper (fp16 pool) + the fp8 entry used by the vLLM shim.
std::tuple<torch::Tensor, torch::Tensor> flash_decode_defer_mqa_paged(
    torch::Tensor Q, torch::Tensor Kcache, torch::Tensor Vcache, torch::Tensor Kmax,
    torch::Tensor block_table, int64_t block_size, double scale, int64_t num_splits, int64_t kv_len,
    torch::Tensor seqlens) {
    return flash_decode_defer_mqa_paged_impl(Q, Kcache, Vcache, Kmax, block_table, block_size, scale,
                                             num_splits, kv_len, seqlens, false, 1.0, 1.0);
}

std::tuple<torch::Tensor, torch::Tensor> flash_decode_defer_mqa_paged_fp8(
    torch::Tensor Q, torch::Tensor Kcache, torch::Tensor Vcache, torch::Tensor Kmax,
    torch::Tensor block_table, int64_t block_size, double scale, int64_t num_splits, int64_t kv_len,
    torch::Tensor seqlens, double k_scale, double v_scale) {
    return flash_decode_defer_mqa_paged_impl(Q, Kcache, Vcache, Kmax, block_table, block_size, scale,
                                             num_splits, kv_len, seqlens, true, k_scale, v_scale);
}

// OUR int8 paged decode: byte pool + per-(token,kv-head) scale tables sK/sV laid out exactly like the
// pool ([num_blocks, block_size, Hkv]), so the block table walks both. Kmax is the bound on the RAW
// (pre-scale) keys, matching what the kernel reads.
std::tuple<torch::Tensor, torch::Tensor> flash_decode_defer_mqa_paged_i8(
    torch::Tensor Q, torch::Tensor Kcache, torch::Tensor sK, torch::Tensor Vcache, torch::Tensor sV,
    torch::Tensor Kmax, torch::Tensor block_table, int64_t block_size, double scale,
    int64_t num_splits, int64_t kv_len, torch::Tensor seqlens) {
    const c10::cuda::CUDAGuard device_guard(Q.device());
    int Bb = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Hkv = Kcache.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Sq == 1 && (d == 64 || d == 128 || d == 256 || d == 512) && H % Hkv == 0);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table int32");
    TORCH_CHECK(Kcache.element_size() == 1 && Vcache.element_size() == 1, "int8 paged: 1-byte KV pool");
    int mb = block_table.size(1);
    bool veqk = (Kcache.data_ptr() == Vcache.data_ptr() && Kcache.sizes() == Vcache.sizes());
    int valid = (int)kv_len;
    auto Qc = Q.contiguous(), Kc = Kcache.contiguous(), Vc = Vcache.contiguous();
    auto Km = Kmax.contiguous().to(torch::kFloat32), bt = block_table.contiguous();
    auto sKc = sK.contiguous().to(torch::kFloat32), sVc = sV.contiguous().to(torch::kFloat32);
    const int* slp = nullptr; torch::Tensor sl;
    if (seqlens.numel() > 0) { TORCH_CHECK(seqlens.scalar_type() == torch::kInt32 && seqlens.numel() == Bb, "seqlens int32 [B]");
        sl = seqlens.contiguous(); slp = sl.data_ptr<int32_t>(); }
    int ns = std::max<int>(1, std::min<int>(num_splits, (valid + 31) / 32)); int Lc = (valid + ns - 1) / ns;
    auto f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto A = torch::empty({Bb, H, ns, d}, f32), B = torch::empty({2, Bb, H, ns}, f32);   // [0]=sums, [1]=per-split maxima
    auto O = torch::empty({Bb, H, Sq, d}, f32), Lse = torch::empty({Bb, H}, f32);
    int GF = pick_gf(d, H / Hkv);
    const half* q = (const half*)Qc.data_ptr<at::Half>();
    if (GF <= 1)
        launch_defer_paged_i8(q, Kc.data_ptr(), Vc.data_ptr(), sKc.data_ptr<float>(), sVc.data_ptr<float>(),
                              Km.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), O.data_ptr<float>(),
                              Lse.data_ptr<float>(), Bb, H, Hkv, block_size, d, (float)scale, ns, Lc, valid,
                              bt.data_ptr<int32_t>(), (int)block_size, mb, slp, at::cuda::getCurrentCUDAStream());
    else
        launch_defer_mqa_paged_i8(q, Kc.data_ptr(), Vc.data_ptr(), sKc.data_ptr<float>(), sVc.data_ptr<float>(),
                                  Km.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), O.data_ptr<float>(),
                                  Lse.data_ptr<float>(), Bb, H, Hkv, block_size, d, (float)scale, ns, Lc, GF, veqk, valid,
                                  bt.data_ptr<int32_t>(), (int)block_size, mb, slp, at::cuda::getCurrentCUDAStream());
    return std::make_tuple(O, Lse.reshape({Bb, H, 1}));
}

// MQA/GQA-aware deferred decode: fuses GF query heads per CTA so shared KV is read from HBM once
// (traffic /GF). Falls back to the per-head path when the group can't be fused (GF==1).
std::tuple<torch::Tensor, torch::Tensor> flash_decode_defer_mqa(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Kmax, double scale, int64_t num_splits, int64_t kv_len) {
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    int Bb = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Hkv = K.size(1), Sk = K.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Sq == 1 && (d == 64 || d == 128 || d == 256 || d == 512) && H % Hkv == 0);
    bool veqk = (K.data_ptr() == V.data_ptr() && K.sizes() == V.sizes());   // V IS K (Gemma-4 global layers)
    int valid = (kv_len > 0) ? (int)kv_len : Sk;   // valid cache length (kv_len<=0 -> full); Sk = physical stride
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous(), Km = Kmax.contiguous().to(torch::kFloat32);
    int ns = std::max<int>(1, std::min<int>(num_splits, (valid + 31) / 32)); int Lc = (valid + ns - 1) / ns;
    auto f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto A = torch::empty({Bb, H, ns, d}, f32), B = torch::empty({2, Bb, H, ns}, f32);   // [0]=sums, [1]=per-split maxima
    auto O = torch::empty({Bb, H, Sq, d}, f32), Lse = torch::empty({Bb, H}, f32);
    int GF = pick_gf(d, H / Hkv);
    if (GF <= 1)
        launch_defer(16, (const half*)Qc.data_ptr<at::Half>(), Kc.data_ptr<at::Half>(), Vc.data_ptr<at::Half>(), nullptr, nullptr, Km.data_ptr<float>(),
                     A.data_ptr<float>(), B.data_ptr<float>(), O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, Sk, d, (float)scale, ns, Lc, valid, at::cuda::getCurrentCUDAStream());
    else
        launch_defer_mqa(16, (const half*)Qc.data_ptr<at::Half>(), Kc.data_ptr<at::Half>(), Vc.data_ptr<at::Half>(), nullptr, nullptr, Km.data_ptr<float>(),
                         A.data_ptr<float>(), B.data_ptr<float>(), O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, Sk, d, (float)scale, ns, Lc, GF, veqk, valid, at::cuda::getCurrentCUDAStream());
    return std::make_tuple(O, Lse.reshape({Bb, H, 1}));
}

std::tuple<torch::Tensor, torch::Tensor> flash_decode_defer(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Kmax, double scale, int64_t num_splits) {
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    int Bb = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Hkv = K.size(1), Sk = K.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Sq == 1 && (d == 64 || d == 128 || d == 256 || d == 512) && H % Hkv == 0);
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous(), Km = Kmax.contiguous().to(torch::kFloat32);
    int ns = std::max<int>(1, std::min<int>(num_splits, (Sk + 31) / 32)); int Lc = (Sk + ns - 1) / ns;
    auto f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto A = torch::empty({Bb, H, ns, d}, f32), B = torch::empty({2, Bb, H, ns}, f32);   // [0]=sums, [1]=per-split maxima
    auto O = torch::empty({Bb, H, Sq, d}, f32), Lse = torch::empty({Bb, H}, f32);
    launch_defer(16, (const half*)Qc.data_ptr<at::Half>(), Kc.data_ptr<at::Half>(), Vc.data_ptr<at::Half>(), nullptr, nullptr, Km.data_ptr<float>(),
                 A.data_ptr<float>(), B.data_ptr<float>(), O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, Sk, d, (float)scale, ns, Lc, Sk, at::cuda::getCurrentCUDAStream());
    return std::make_tuple(O, Lse.reshape({Bb, H, 1}));
}

// int8/int4 + deferred-rescale: the long-context optimum (fewer bytes AND no serial chain).
std::tuple<torch::Tensor, torch::Tensor> flash_decode_defer_q(torch::Tensor Q, torch::Tensor Kq, torch::Tensor sK, torch::Tensor Vq, torch::Tensor sV,
                                                              torch::Tensor Kmax, double scale, int64_t num_splits, int64_t bits, int64_t kv_len) {
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    int Bb = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Hkv = sK.size(1), Sk = sK.size(2);
    TORCH_CHECK(Sq == 1 && (d == 64 || d == 128 || d == 256 || d == 512) && H % Hkv == 0 && (bits == 8 || bits == 4));
    int valid = (kv_len > 0) ? (int)kv_len : Sk;   // valid cache length (kv_len<=0 -> full); Sk = physical stride
    auto Qc = Q.contiguous(), Km = Kmax.contiguous().to(torch::kFloat32);
    int ns = std::max<int>(1, std::min<int>(num_splits, (valid + 31) / 32)); int Lc = (valid + ns - 1) / ns;
    auto f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto A = torch::empty({Bb, H, ns, d}, f32), B = torch::empty({2, Bb, H, ns}, f32);   // [0]=sums, [1]=per-split maxima
    auto O = torch::empty({Bb, H, Sq, d}, f32), Lse = torch::empty({Bb, H}, f32);
    launch_defer((int)bits, (const half*)Qc.data_ptr<at::Half>(), Kq.data_ptr(), Vq.data_ptr(), sK.data_ptr<float>(), sV.data_ptr<float>(), Km.data_ptr<float>(),
                 A.data_ptr<float>(), B.data_ptr<float>(), O.data_ptr<float>(), Lse.data_ptr<float>(), Bb, H, Hkv, Sk, d, (float)scale, ns, Lc, valid, at::cuda::getCurrentCUDAStream());
    return std::make_tuple(O, Lse.reshape({Bb, H, 1}));
}


// ------------------------------------------------------------------------------------------
// ХИМЕРНЫЙ ПУТЬ (тензорные ядра + плитка в разделяемой): точка входа и свод срезов.
// Отдельная функция, а не ветка внутри старой: боевой путь обязан остаться битово прежним,
// пока новый не пройдёт гейт точности и скорости.
// ------------------------------------------------------------------------------------------
__global__ void hmma_svod_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                 half* __restrict__ Out, int H, int D, int ns, int ldo) {
    const int стр = blockIdx.x, h = blockIdx.y;
    const long бб = ((long)стр * H + h) * ns;
    float M = -CUDART_INF_F;
    for (int i = 0; i < ns; ++i) M = fmaxf(M, B[(бб + i) * 2]);
    float L = 0.f;
    for (int i = 0; i < ns; ++i) {
        const float m = B[(бб + i) * 2];
        if (m == -CUDART_INF_F) continue;
        L += __expf(m - M) * B[(бб + i) * 2 + 1];
    }
    const float inv = (L > 0.f) ? 1.0f / L : 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float o = 0.f;
        for (int i = 0; i < ns; ++i) {
            const float m = B[(бб + i) * 2];
            if (m == -CUDART_INF_F) continue;
            o += __expf(m - M) * A[(бб + i) * (long)D + d];
        }
        Out[(long)стр * ldo + h * D + d] = __float2half(o * inv);
    }
}

void flash_decode_hmma_paged(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                             torch::Tensor block_table, int64_t block_size, double scale,
                             int64_t num_splits, torch::Tensor seqlens,
                             torch::Tensor A, torch::Tensor B, torch::Tensor Out, int64_t ldo,
                             torch::Tensor sK, torch::Tensor sV, int64_t qGrp, int64_t qCausal,
                             std::optional<torch::Tensor> selCnt = std::nullopt,
                             std::optional<torch::Tensor> selIdx = std::nullopt, int64_t selBlk = 128) {
    TORCH_CHECK(Q.scalar_type() == torch::kHalf, "hmma decode: Q half");
    // [СПИСОК БЛОКОВ ОТБОРА, 14.09] необязательный индекс отбора (cnt [строк, Hkv], idx [строк, Hkv, NB] int32)
    const int* _sIp = nullptr; const int* _sCp = nullptr; int _sStr = 0, _sBlg = 0;
    if (selIdx.has_value() && selIdx->defined()) {
        TORCH_CHECK(selCnt.has_value() && selCnt->defined(), "hmma decode: индекс отбора требует счётчиков");
        TORCH_CHECK(selIdx->scalar_type() == torch::kInt32 && selCnt->scalar_type() == torch::kInt32
                    && selIdx->is_contiguous() && selCnt->is_contiguous(), "hmma decode: индекс отбора int32 плотный");
        TORCH_CHECK((selBlk & (selBlk - 1)) == 0 && selBlk % fa2sm70_dechmma::KT == 0,
                    "hmma decode: блок отбора -- степень двойки, кратная плитке ", fa2sm70_dechmma::KT);
        _sIp = selIdx->data_ptr<int32_t>(); _sCp = selCnt->data_ptr<int32_t>(); _sStr = (int)selIdx->size(-1);
        while ((1LL << _sBlg) < selBlk) ++_sBlg;
    }
    TORCH_CHECK(K.scalar_type() == torch::kChar && V.scalar_type() == torch::kChar, "hmma decode: KV int8");
    const int строк_всего = Q.size(0), H = Q.size(1), D = Q.size(3);
    TORCH_CHECK(K.dim() >= 3, "hmma decode: пул [.., block, Hkv, D]");
    TORCH_CHECK(D == 256, "hmma decode: пока только head_dim=256");
    const int Hkv = K.size(K.dim() - 2), bs = (int)block_size, mb = block_table.size(1);
    // Шаг блока берём У ТЕНЗОРА, а не выводим из формы: пул может быть строенным видом.
    const long шаг_блока = K.stride(0) / D;
    TORCH_CHECK((bs & (bs - 1)) == 0, "hmma decode: block_size обязан быть степенью двойки");
    int bs_log = 0; while ((1 << bs_log) < bs) ++bs_log;
    const int B_ = строк_всего / (int)qGrp, GF = H / Hkv;
    TORCH_CHECK((int)qGrp * GF <= fa2sm70_dechmma::MT, "hmma decode: qGrp*GF > 32");
    // Строки ОБЯЗАНЫ делиться на qGrp: сетка идёт по запросам, и хвост строк иначе никем не
    // записывается -- свод прочитает в рабочей области остаток прошлого шага. Предусловие
    // ФОРМЫ, а не пожелание.
    TORCH_CHECK(строк_всего % (int)qGrp == 0, "hmma decode: строк ", строк_всего,
                " не делится на qGrp=", qGrp);
    const int ns = (int)num_splits;
    const size_t shmem = sizeof(fa2sm70_dechmma::Obshchaya<256>);
    // ДИНАМИЧЕСКАЯ РАЗДЕЛЯЕМАЯ СВЫШЕ 48 КиБ ТРЕБУЕТ ЯВНОГО СОГЛАСИЯ, И ЕГО НАДО ПРОВЕРЯТЬ.
    // Без проверки отказ виден только как МУСОР НА ВЫХОДЕ: ядро не запускается, буфер остаётся
    // с прошлого шага, а сервер отвечает связным началом и рядами восклицательных знаков.
    // Ровно это и случилось при первом подключении к бэкенду.
    static bool _opt = false;
    if (!_opt) {
        cudaError_t _e = cudaFuncSetAttribute(fa2sm70_dechmma::decode_hmma_kernel<256>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        // [V3] два блока по ~46 КБ на SM требуют полной выкройки разделяемой (96 КБ на V100)
        cudaFuncSetAttribute(fa2sm70_dechmma::decode_hmma_kernel<256>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        TORCH_CHECK(_e == cudaSuccess, "hmma decode: не удалось запросить ", shmem,
                    " Б разделяемой: ", cudaGetErrorString(_e));
        _opt = true;
    }
    dim3 grid(B_, Hkv, ns), blk(fa2sm70_dechmma::NT);
    auto st = at::cuda::getCurrentCUDAStream();
    fa2sm70_dechmma::decode_hmma_kernel<256><<<grid, blk, shmem, st>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        K.data_ptr<int8_t>(), V.data_ptr<int8_t>(),
        sK.numel() ? sK.data_ptr<float>() : nullptr, sV.numel() ? sV.data_ptr<float>() : nullptr,
        A.data_ptr<float>(), B.data_ptr<float>(), H, Hkv, 0, (float)scale, ns,
        block_table.data_ptr<int>(), bs_log, mb, seqlens.data_ptr<int>(), (int)qGrp, GF, (int)qCausal,
        шаг_блока, _sIp, _sCp, _sStr, _sBlg);
    {   // Отказ запуска обязан быть ГРОМКИМ, а не тихой порчей выхода.
        cudaError_t _l = cudaGetLastError();
        TORCH_CHECK(_l == cudaSuccess, "hmma decode: запуск ядра отвергнут: ",
                    cudaGetErrorString(_l), " (shmem=", shmem, ")");
    }
    dim3 g2(строк_всего, H);
    hmma_svod_kernel<<<g2, 128, 0, st>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                         reinterpret_cast<half*>(Out.data_ptr<at::Half>()),
                                         H, D, ns, (int)ldo);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_decode", &flash_decode, "sm70 split-KV flash-decoding (fp16 exact; optional softcap / ALiBi)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("scale"), pybind11::arg("num_splits"),
          pybind11::arg("softcap") = 0.0, pybind11::arg("alibi_slopes") = torch::Tensor());
    m.def("flash_decode_q", &flash_decode_q, "sm70 split-KV flash-decoding (int8/int4 KV + optional top-1 correction)");
    m.def("flash_decode_defer", &flash_decode_defer, "sm70 deferred-rescale flash-decoding (associative, certified U-bound)");
    m.def("gptq_dequant_w8", &gptq_dequant_w8,
          "GPTQ 8-bit qweight/qzeros/scales -> fp16 [K,N] (batched-decode fast path)",
          pybind11::arg("qweight"), pybind11::arg("qzeros"), pybind11::arg("scales"), pybind11::arg("out"));
    m.def("gather_paged_kv", &gather_paged_kv,
          "gather one sequence's paged KV (fp8_e4m3 or fp16 pool) into contiguous fp16 [Hkv, T, d]",
          pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("block_table"),
          pybind11::arg("seq_len"), pybind11::arg("k_scale"), pybind11::arg("v_scale"),
          pybind11::arg("out_k"), pybind11::arg("out_v"));
    m.def("gather_paged_kv_bytes", &gather_paged_kv_bytes,
          "gather one sequence's paged 1-byte KV verbatim into contiguous [Hkv,T,d] (e5m2: no conversion)",
          pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("block_table"),
          pybind11::arg("seq_len"), pybind11::arg("out_k"), pybind11::arg("out_v"));
    m.def("gather_paged_kv_i8", &gather_paged_kv_i8,
          "gather one sequence's paged KV into int8 [Hkv,T,d] + per-position fp16 scales [Hkv,T] "
          "gather one sequence's paged KV into int8 [Hkv,T,d] + per-position fp32 scales [Hkv,T]",
          pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("block_table"),
          pybind11::arg("seq_len"), pybind11::arg("k_scale"), pybind11::arg("v_scale"),
          pybind11::arg("out_kq"), pybind11::arg("out_ks"),
          pybind11::arg("out_vq"), pybind11::arg("out_vs"));
    m.def("gather_paged_kv_i8_pool", &gather_paged_kv_i8_pool,
          "gather one sequence's paged int8 KV VERBATIM into [Hkv,T,d] + per-position fp32 scales [Hkv,T] "
          "(int8 pool: no conversion, no second quantisation, EXACT)",
          pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("k_scales"),
          pybind11::arg("v_scales"), pybind11::arg("block_table"), pybind11::arg("seq_len"),
          pybind11::arg("out_kq"), pybind11::arg("out_ks"), pybind11::arg("out_vq"), pybind11::arg("out_vs"));
    m.def("gather_paged_kv_i8_pool_f16", &gather_paged_kv_i8_pool_f16,
          "gather one sequence's paged int8 KV into contiguous fp16 [Hkv,T,d] (exact dequant q*s)",
          pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("k_scales"),
          pybind11::arg("v_scales"), pybind11::arg("block_table"), pybind11::arg("seq_len"),
          pybind11::arg("out_k"), pybind11::arg("out_v"));
    m.def("reshape_and_cache_i8", &reshape_and_cache_i8,
          "sm70: quantise K/V to int8 + per-(token,kv-head) scale into the engine's (strided) byte pool, "
          "scales into a dense side table [num_blocks, block_size, Hkv]",
          py::arg("key"), py::arg("value"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("slot_mapping"), py::arg("k_scales"), py::arg("v_scales"));
    m.def("reshape_and_cache_e4m3", &reshape_and_cache_e4m3,
          "sm70: quantise K/V to e4m3 into the engine's (strided) paged pool AND fold the new tokens into "
          "the running ||k_raw|| bound in the same kernel",
          py::arg("key"), py::arg("value"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("slot_mapping"), py::arg("k_scale"), py::arg("v_scale"), py::arg("kbound"),
          py::arg("margin") = 1.15);
    m.def("flash_decode_defer_mqa_paged_into", &flash_decode_defer_mqa_paged_into,
          "sm70 paged deferred decode writing fp16 STRAIGHT into a caller-owned output buffer, with a "
          "caller-owned split workspace: zero allocations and zero output copies per decode step",
          py::arg("Q"), py::arg("Kcache"), py::arg("Vcache"), py::arg("Kmax"), py::arg("block_table"),
          py::arg("block_size"), py::arg("scale"), py::arg("num_splits"), py::arg("kv_len"),
          py::arg("seqlens"), py::arg("fp8"), py::arg("k_scale"), py::arg("v_scale"),
          py::arg("A"), py::arg("B"), py::arg("Out"), py::arg("o_row_stride"),
          // Таблицы масштабов int8-пула. Отсутствуют (по умолчанию) = пул НЕ int8, и тогда путь тот
          // же, что был: НИ ОДИН существующий вызов на 17 аргументов не меняется -- это проверено
          // вызовом, а не предположено (см. комментарий у сигнатуры про None -> torch::Tensor).
          py::arg("sK") = std::nullopt, py::arg("sV") = std::nullopt,
          // [ОКНО] Левая граница ключей на последовательность (int32 [B]); None = 0 = прежнее поведение.
          py::arg("kv_start") = std::nullopt,
          // [ДЕРЕВО] Граница контекста и битовая маска хвоста на строку; None = цепь, как было.
          py::arg("ctxlen") = std::nullopt, py::arg("tailmask") = std::nullopt,
          // Индекс отобранных блоков: без него путь плотный и прежний до бита.
          py::arg("selCnt") = std::nullopt, py::arg("selIdx") = std::nullopt,
          py::arg("selBlk") = 128, py::arg("qGrp") = 1, py::arg("qCausal") = 1);
    m.def("flash_decode_defer_mqa_paged_i8", &flash_decode_defer_mqa_paged_i8,
          "sm70 paged deferred decode over OUR int8 KV pool (per-(token,kv-head) scales, block-table indexed)",
          py::arg("Q"), py::arg("Kcache"), py::arg("sK"), py::arg("Vcache"), py::arg("sV"), py::arg("Kmax"),
          py::arg("block_table"), py::arg("block_size"), py::arg("scale"), py::arg("num_splits"),
          py::arg("kv_len"), py::arg("seqlens"));
    m.def("flash_decode_defer_mqa_paged_fp8", &flash_decode_defer_mqa_paged_fp8,
          "sm70 paged deferred decode over an fp8_e4m3 KV pool (vLLM layout; k_scale folded into scale, v_scale on output)",
          py::arg("Q"), py::arg("Kcache"), py::arg("Vcache"), py::arg("Kmax"), py::arg("block_table"),
          py::arg("block_size"), py::arg("scale"), py::arg("num_splits"), py::arg("kv_len"),
          py::arg("seqlens"), py::arg("k_scale") = 1.0, py::arg("v_scale") = 1.0);
    m.def("flash_decode_defer_mqa_paged", &flash_decode_defer_mqa_paged, "sm70 paged deferred decode (block-table KV pool)",
          pybind11::arg("Q"), pybind11::arg("Kcache"), pybind11::arg("Vcache"), pybind11::arg("Kmax"),
          pybind11::arg("block_table"), pybind11::arg("block_size"), pybind11::arg("scale"),
          pybind11::arg("num_splits"), pybind11::arg("kv_len"), pybind11::arg("seqlens"));
    m.def("flash_decode_defer_mqa", &flash_decode_defer_mqa, "sm70 MQA/GQA-aware deferred decode (fuse GF query heads, shared KV read once)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("Kmax"), pybind11::arg("scale"), pybind11::arg("num_splits"), pybind11::arg("kv_len") = 0);
    m.def("flash_decode_hmma_paged", &flash_decode_hmma_paged,
          "sm70 химерный декод: плитка K/V в разделяемой + обе GEMM на тензорных ядрах; selCnt/selIdx/selBlk -- список блоков отбора",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("block_table"),
          pybind11::arg("block_size"), pybind11::arg("scale"), pybind11::arg("num_splits"),
          pybind11::arg("seqlens"), pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("Out"),
          pybind11::arg("ldo"), pybind11::arg("sK"), pybind11::arg("sV"),
          pybind11::arg("qGrp") = 1, pybind11::arg("qCausal") = 1,
          pybind11::arg("selCnt") = std::nullopt, pybind11::arg("selIdx") = std::nullopt, pybind11::arg("selBlk") = 128);
    m.def("flash_decode_defer_q", &flash_decode_defer_q, "sm70 deferred-rescale + int8/int4 KV (long-context optimum)",
          pybind11::arg("Q"), pybind11::arg("Kq"), pybind11::arg("sK"), pybind11::arg("Vq"), pybind11::arg("sV"),
          pybind11::arg("Kmax"), pybind11::arg("scale"), pybind11::arg("num_splits"), pybind11::arg("bits"),
          pybind11::arg("kv_len") = 0);
}
