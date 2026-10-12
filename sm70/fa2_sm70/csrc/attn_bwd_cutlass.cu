// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// FA2-sm70 BACKWARD via cutlass's own FMHA backward machinery (kernel_backward.h, BSD-3).
// Feasibility gate (see TODO.md) proved AttentionBackwardKernel<Sm70, half_t, ...> compiles clean
// on Volta (kPreload/kOutputInRF paths off for cc==70). This wraps it under our BHSD API.
//
// LAYOUT CONTRACT of the cutlass backward kernel (why everything here is BSHD [B, S, H, D]):
//   * output O, grad dQ/dK/dV have a HARD-CODED token stride  o_strideM()=gQ_strideM()=num_heads*head_dim
//     (methods, not fields) -> those tensors MUST be packed [tokens, heads, dim] (BSHD).
//   * Q/K/V and grad_output dO use configurable strideM/H/B fields (any layout) -- we still feed BSHD.
//   * delta = (dO*O).sum(-1): on sm_70 with kMaxK>kBlockSizeI (kOutputInRF=false) the kernel's
//     kKernelComputesDelta is FALSE, so we precompute delta in torch and pass it in (harmless when
//     the kernel does recompute it for the small-head kMaxK=64 path).
//   * GQA/MQA is handled in Python (expand K/V to H heads, reduce dK/dV back to Hkv) -- the backward
//     kernel itself assumes one KV head per query head.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "kernel_backward.h"
#include "volta_bwd_block.cuh"   // рукописный блочный мейнлуп (причинный d=256/512)

#ifdef FMHA_BWD_COAL_FO
// [COAL_FO] bespoke transpose-convert: fragment-order fp32 ws -> output-order fp16 dQ. Consecutive threads write
// consecutive output positions (query_local*d_pad+d, contiguous) -> COALESCED store; the read ws[base+inv[j]] is
// the (permuted) scattered side, but a plain resident-level load. One 2N pass, no torch dispatch/materialisation
// overhead (which swamped the tiny single-wave shape). inv is the cached inverse perm (fragorder-slot -> was here).
// out_span = kBI*kMaxK (per-qblk OUTPUT slice, = kBI*head_dim); ws_span = kBI*kBI (per-qblk FRAGORDER slice, the
// whole <kBI,kBI> R7 tile). Equal at head_dim==kBI (d=128); for d=64 (head_dim=64<kBI=128) the fragorder tile is
// still kBI-wide so ws_span=2*out_span. inv[j] (j = out-index within the qblk) is the fragorder slot for out-index j.
__global__ void fo_convert_kernel(const float* __restrict__ ws, __half* __restrict__ dq,
                                  const int32_t* __restrict__ inv, int64_t out_span, int64_t ws_span, int64_t total) {
  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += (int64_t)gridDim.x * blockDim.x) {
    int64_t j = t % out_span;             // output-order index within a (bh,qblk) block
    int64_t blk = t / out_span;           // which (bh,qblk)
    dq[t] = __float2half(ws[blk * ws_span + (int64_t)inv[j]]);
  }
}
#endif


// [ОБВЯЗКА, малые формы] `getenv` — линейный поиск по окружению, а в обратном проходе их 18 на вызов.
// На малой форме (d=128, H=8, S=512) работы на GPU всего 188 мкс, а настенное время было 234: разница
// целиком хостовая. Ключи настройки читаем ОДИН раз за процесс (они экспериментальные, менять их
// посреди работы процесса незачем — задавайте до запуска).
static inline const char* fa2_env(const char* name) {
  struct Slot { const char* k; const char* v; };
  static Slot slots[24]; static int n = 0;
  for (int i = 0; i < n; ++i) if (slots[i].k == name) return slots[i].v;   // сравнение указателей: имена — литералы
  const char* v = getenv(name);
  if (n < 24) { slots[n].k = name; slots[n].v = v; ++n; }
  return v;
}

template <int kBI, int kBJ, int kMaxK, bool kDrop = false, bool kAligned = false, bool kESK = true,
          bool kCVO = false, bool kSGV = false, bool kSGQ = false, bool kCQO = false>
static std::vector<torch::Tensor> run_bwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,   // BSHD [B, Sq/Sk, H, D] fp16 (K/V already expanded to H)
    torch::Tensor O, torch::Tensor dO,                   // BSHD [B, Sq, H, D] fp16
    torch::Tensor LSE, torch::Tensor delta,              // [B, H, lse_dim] fp32 (natural-log), [B, H, Sq] fp32
    double scale, bool causal, int64_t num_splits,
    double dropout_p = 0.0, int64_t seed = 0, int64_t offset = 0, bool bhsd = false,
    int64_t qsplit = 0) {   // [TWO-PASS] >0 => own disjoint query bands (Pass Q); 0 => env/default
    // bhsd: inputs are native BHSD [B,H,S,D] (Q/K/V/O/dO) and grads are written BHSD in place -- the
    // wrapper then skips the BSHD<->BHSD permute+contiguous transposes (measured ~1.3x of the backward
    // wall is that glue). Enabled by settable output-row-stride overrides in the kernel Params. The
    // kernel is layout-agnostic (strided iterators); only these host-side strides + output shapes change.
    // Block geometry (kBI x kBJ) tuned per shape; all fit Volta's 96KB smem except 128x128 (97KB, excluded).
    // kAligned == kKeysQueriesAlignedToBlockSize: when Sq%kBI==0 && Sk%kBJ==0, the kernel drops the
    // per-tile boundary predication (the `skipBoundsChecks` path: no num_*_in_block min(), no
    // set_zero_outside_bounds masking, no `key >= num_keys` guards). This is exactly torch's
    // `fmha_cutlassB_..._seqaligned_sm70` kernel and is the ~1.2-1.3x backward gap vs our bounds-checked
    // default. The caller MUST guarantee the alignment (asserted inside the kernel); causal masking is
    // orthogonal and still applied.
    // kESK == kEnableSplitKeys: our cutlass-4.6 kernel's 10th template param (absent in torch's 9-param
    // kernel). Its split-key workspace-atomic-dQ machinery costs +15 registers -> 255 (spilling 248 B) vs
    // torch's 240 (no spill). We ship BOTH: kESK=false is byte-for-byte torch's 240-no-spill kernel (7-10%
    // faster) but cannot split the key axis, so it only wins when B*H already fills the 80 SMs; kESK=true
    // splits keys to fill the SMs at small B*H (its spill is the lesser evil there). Routed by B*H in
    // attn_bwd_cutlass() and _bwd_run.
    // NOTE: kPreload (stock cutlass prologue prefetch) was tested on sm_70 (assert relaxed): MEASURED a
    // ~1-2% wash AND incorrect (dK fails on causal Sq!=Sk / non-aligned) -- the prologue does LDG->STS
    // adjacently so the fetch latency is still exposed at the STS (measured failure). The real lever is
    // the CHUNKED tail-station schedule (separate LDG from STS by a contraction span); see prefetch-spec.
    using AK = AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t,
                                       /*kIsAligned*/ true, /*kApplyDropout*/ kDrop, /*kPreload*/ false,
                                       /*kBlockSizeI*/ kBI, /*kBlockSizeJ*/ kBJ, kMaxK,
                                       /*kKeysQueriesAlignedToBlockSize*/ kAligned, /*kEnableSplitKeys*/ kESK,
                                       /*kComputeVOnly*/ kCVO, /*kSkipGradV*/ kSGV,
                                       /*kSkipGradQ*/ kSGQ, /*kComputeQOnly*/ kCQO>;
    // Layout: BSHD [B,Sq/Sk,H,D] (dims 1,2 = seq,head) or BHSD [B,H,Sq/Sk,D] (dims 1,2 = head,seq).
    int B = Q.size(0);
    int Sq = bhsd ? Q.size(2) : Q.size(1);
    int H  = bhsd ? Q.size(1) : Q.size(2);
    int d  = Q.size(3);
    int Sk = bhsd ? K.size(2) : K.size(1);
    auto opt_h = torch::dtype(torch::kFloat16).device(Q.device());
    auto dQ = bhsd ? torch::empty({B, H, Sq, d}, opt_h) : torch::empty({B, Sq, H, d}, opt_h);
    auto dK = bhsd ? torch::empty({B, H, Sk, d}, opt_h) : torch::empty({B, Sk, H, d}, opt_h);
    auto dV = bhsd ? torch::empty({B, H, Sk, d}, opt_h) : torch::empty({B, Sk, H, d}, opt_h);
    // [БЛОЧНЫЙ МЕЙНЛУП] Рукописный путь для ПРИЧИННОГО d=256/512 с выровненными размерами: измеренно
    // 2.184 мс против 2.954 у ядра cutlass на d=256/H=8/S=2048, x1.35 (docs/SM70_KERNEL_PLAYBOOK.md
    // §14-16). Условия применимости узкие намеренно — на всё остальное остаётся штатный путь.
    if constexpr (kMaxK == 256 || kMaxK == 512) {
      if (d == kMaxK && fa2_sm70::block_bwd::eligible(B, H, H, Sq, Sk, d, causal, bhsd)) {
        auto dqf = torch::empty({B, H, Sq, d}, torch::dtype(torch::kFloat32).device(Q.device()));
        if (fa2_sm70::block_bwd::launch<kMaxK, true>(Q, K, V, dO, LSE, delta, dQ, dK, dV,
                                           B, H, Sq, Sk, scale, dqf))
          return {dQ, dK, dV};
      }
    }
    // Declared OUTSIDE the R7 block on purpose: `p.fo_dq = fo_elig` further down is NOT inside it, so with
    // FA2SM70_BWD_R7=0 the documented A/B off-switch did not compile at all ("identifier fo_elig is
    // undefined"). Without R7 there is no fragment-order fold, so it simply stays false.
    bool fo_elig = false;
#ifdef FMHA_BWD_R7
    auto opt_f32 = torch::dtype(torch::kFloat32).device(Q.device());
    // [COAL_FO d64+d128] FO-eligibility, computed HERE because the fragorder workspace is kBI-wide in d (the whole
    // <kBI,kBI> R7 tile), i.e. Sq*kBI slots -- NOT Sq*d (equal only at d==128). Eligible: bhsd, kBI==128, head_dim
    // ==kMaxK (a fully-populated d-tile: d==64 or d==128), full 128-query blocks. (d=64 also uses R7 with kMaxK=64.)
#ifdef FMHA_BWD_COAL_FO
    // [ДЕФЕКТ Sq>Sk] Карта перестановки захватывается кернелом НА ПЕРВОЙ ПЛИТКЕ. При причинной маске
    // с нижне-правым выравниванием и Sq > Sk первые Sq-Sk запросов не видят НИ ОДНОГО ключа, то есть
    // первая плитка пуста: карта не захватывается, остаётся нулевой, и конверсия выдаёт НОЛЬ по всему
    // dQ — молча, при верных dK/dV и верном выходе прямого прохода. Ловится
    // tests/test_backward.py (случаи Sq>Sk causal). Условие годности теперь это исключает.
    fo_elig = bhsd && ((int)kBI == 128) && ((int)kMaxK == d) && (d == 64 || d == 128) && ((Sq % 128) == 0)
              && (!causal || Sq <= Sk);
    if (const char* _f = fa2_env("FA2SM70_FORCE_FO")) { if (atoi(_f) == 0) fo_elig = false; }  // off-switch (A/B)
#endif
    int fo_w = fo_elig ? (int)kBI : d;   // fragorder ws is kBI-wide (<kBI,kBI> tile); scattered/plain dQ is d-wide
    // [ZERO-FILL d-gate] r7_dqf is READ AND WRITTEN ONLY by the R7 whole-B path, which needs kMaxK <= kBI
    // (see the [R7 d-gate] note at the epilogue below). It was nevertheless allocated AND ZEROED for every
    // shape, because the guard was a BUILD flag (#ifdef FMHA_BWD_R7), not the runtime geometry: at d=256
    // that is B*H*Sq*d floats = 16.8 MB of pure waste. Per-kernel profiling put the three FillFunctor
    // launches at 1164 us of a 4291 us backward -- 27% of wall time, against torch's 141 us.
    constexpr bool kUseR7Dqf = (kMaxK <= kBI);
    auto r7_dqf = kUseR7Dqf
        ? (bhsd ? torch::zeros({B, H, Sq, fo_w}, opt_f32) : torch::zeros({B, Sq, H, d}, opt_f32))
        : torch::empty({1}, opt_f32);   // d > kBI: путь R7 не работает, буфер не нужен даже как память
    // [R7 dead-scratch] the global S^T round-trip was replaced by the ON-CHIP r7_sT (DOIVJ .at() fill +
    // prologue_whole<true,false> loads only k^T), so r7_st_scratch is never dereferenced. Keep a size-1 valid
    // pointer (iterators do offset arithmetic but never load) and drop the S^2/head allocation. Confirmed: the
    // old allocation was torch::empty (uninitialized) yet correctness passed -> the buffer is dead.
    auto r7_scratch = torch::empty({1}, opt_h);  // [R7] dead S^T scratch (on-chip r7_sT superseded it)
#ifdef FMHA_BWD_R7_FP16DQ
    dQ.zero_();  // [Q2] the R7 epilogue half-atomicAdds dQ into grad_query directly -> must start at zero
#endif
#ifdef FMHA_BWD_COAL_FO
    // [COAL_FO] r7_dqf is reused as a FRAGMENT-ORDER workspace (same numel as dQ). r7_perm captures the fixed
    // fragorder-slot -> (query_local*d_pad + d) map on the first tile so we can transpose it back to dQ[query,d].
    // Perm size = per-lane fragment elems * 32 lanes * warps = the per-query-block fragment span.
    // [ZERO-FILL d-gate] то же и для карты перестановки: она нужна только пути COAL_FO (d==64/128).
    // Карта перестановки ФИКСИРОВАНА для пары (kBI,kMaxK) и заполняется кернелом на первой плитке.
    // Раньше она выделялась И ОБНУЛЯЛАСЬ на каждый вызов (64 КБ + запуск ядра) — держим одну на процесс.
    static torch::Tensor s_perm;
    if (kUseR7Dqf) {
      if (!s_perm.defined() || s_perm.device() != Q.device())
        s_perm = torch::zeros({8 * 64 * 32}, opt_f32);
    }
    auto r7_perm_t = kUseR7Dqf ? s_perm : torch::empty({1}, opt_f32);
#endif
#endif

    typename AK::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr   = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>());
    p.output_ptr      = reinterpret_cast<cutlass::half_t*>(O.data_ptr<at::Half>());
    p.grad_output_ptr = reinterpret_cast<cutlass::half_t*>(dO.data_ptr<at::Half>());
    p.grad_query_ptr  = reinterpret_cast<cutlass::half_t*>(dQ.data_ptr<at::Half>());
#ifdef FMHA_BWD_R7
    p.r7_st_scratch   = reinterpret_cast<cutlass::half_t*>(r7_scratch.data_ptr<at::Half>());
    p.r7_dq_f32       = r7_dqf.data_ptr<float>();
#ifdef FMHA_BWD_COAL_FO
    p.r7_perm         = r7_perm_t.data_ptr<float>();
#endif
#endif
    p.grad_key_ptr    = reinterpret_cast<cutlass::half_t*>(dK.data_ptr<at::Half>());
    p.grad_value_ptr  = reinterpret_cast<cutlass::half_t*>(dV.data_ptr<at::Half>());
    p.logsumexp_ptr = LSE.data_ptr<float>();
    p.delta_ptr     = delta.data_ptr<float>();
    p.bias_ptr = nullptr; p.grad_bias_ptr = nullptr;
    p.cu_seqlens_q_ptr = nullptr; p.cu_seqlens_k_ptr = nullptr;

    p.scale = (float)scale;
    p.head_dim = d; p.head_dim_value = d;
    p.num_heads = H; p.num_batches = B;
    p.num_queries = Sq; p.num_keys = Sk;
    p.custom_mask_type = causal ? AK::CausalFromBottomRight : AK::NoCustomMask;
    // Split the key axis across CTAs (grid.x) to fill the 80-SM V100 -- B*H CTAs alone underfill it.
    // dQ is reduced across splits via the workspace atomic-accumulate path (workspace is zeroed below).
    int max_splits = (Sk + 64 - 1) / 64;                                   // >=1 key-block (kBlockSizeJ) per split
    int ns = (int)num_splits; if (ns < 1) ns = 1; if (ns > max_splits) ns = max_splits;
    if (const char* _fns = fa2_env("FA2SM70_FORCE_NS")) { ns = atoi(_fns); if (ns < 1) ns = 1; if (ns > max_splits) ns = max_splits; }  // [EXPERIMENT] ns sweep
    if (!kESK) ns = 1;                                                      // the 240-reg kernel forbids key-splitting
    p.num_splits_key = ns;
    // [Q1 query-split] de-risk knob: force num_splits_query to measure the FILL speedup (grids B1H8-S512 from
    // 64 CTAs to 64*q). dK/dV are WRONG until the atomic-fold lands -- timing-only until then.
    if (const char* _q = fa2_env("FA2SM70_FORCE_QSPLIT")) { int q = atoi(_q); if (q >= 1) p.num_splits_query = (int16_t)q; }
    // [TWO-PASS] Pass Q owns disjoint query bands (its dQ needs no cross-tile reduction), so the fill comes from
    // the QUERY axis and the key axis is NOT split (ns=1). Caller passes qsplit = number of query blocks.
    if (qsplit > 0) { p.num_splits_query = (int16_t)qsplit; p.num_splits_key = ns = 1; }
    // [COALESCE] UNDER-FILLED shapes (total grid < ~1 wave of 80 SMs) route the dQ fold through the smem transpose
    // slab (coalesced RED, 4x fewer far-store granules -> lg_throttle 2.14->0.32): there the fold-burst is the #1
    // stall so the port cut > the transpose overhead (B1H8-S512 0.97->1.04). Filled shapes keep the scattered emit
    // (MAC-bound; the transpose would be pure add-on that regresses them 0.84x). Measured shape-gate measured.
    p.coalesce_dq = ((int64_t)ns * (int64_t)H * (int64_t)B * (int64_t)p.num_splits_query < 80);
    if (const char* _c = fa2_env("FA2SM70_FORCE_COALESCE")) p.coalesce_dq = (atoi(_c) != 0);
    // [COAL_FO] fragment-order coalesced fold: measured +6-29% (1.16-1.31x isolated backward vs SDPA-eff) on the
    // d=128 spectrum. GATED to the VERIFIED case ONLY: bhsd (the convert transposes to [B,H,Sq,d]); the 128-tile
    // (kBI=kMaxK=128); d==128 (the R7 fragment layout / fragorder-slot span / perm are validated only there --
    // d=64/256/512 use different tiles whose fragorder->(query,d) map is not validated, and d=64 measured a wrong
    // dQ); and Sq a multiple of 128 (whole-B query-blocks -- a partial last block overruns the Sq*d workspace).
    // Everything else (d!=128, partial Sq, BSHD) falls back to the exact scattered emit. Env FA2SM70_FORCE_FO=0/1.
    p.fo_dq = fo_elig;   // [COAL_FO] d==128 (kMaxK=128) and d==64 (kMaxK=64); gate + workspace width set at alloc
    if (fo_elig) {       // [COAL_FO d64] kBI-wide FO workspace -> advance r7_dq_f32 with kBI strides (gQ_stride* are d-wide)
        p.r7_dq_strideM = (int32_t)kBI;
        p.r7_dq_strideH = (int64_t)Sq * (int64_t)kBI;
        p.r7_dq_strideB = (int64_t)H * (int64_t)Sq * (int64_t)kBI;
    }
    p.dropout_prob = kDrop ? (float)dropout_p : 0.0f;                       // replay fwd's mask (same seed/offset)
    if (kDrop) p.rng_engine_inputs = at::PhiloxCudaState((uint64_t)seed, (uint64_t)offset);

    if (bhsd) {
      // BHSD [B,H,S,D]: token(M)=D, head(H)=S*D, batch(B)=H*S*D. Per-head [S,D] is contiguous.
      // Output row-strides go through the overrides (= D) so dQ/dK/dV/O land in BHSD in place.
      p.q_strideM = d; p.q_strideH = Sq * d; p.q_strideB = (int64_t)H * Sq * d;
      p.k_strideM = d; p.k_strideH = Sk * d; p.k_strideB = (int64_t)H * Sk * d;
      p.v_strideM = d; p.v_strideH = Sk * d; p.v_strideB = (int64_t)H * Sk * d;
      p.o_strideM_override = d; p.o_strideH = Sq * d; p.o_strideB = (int64_t)H * Sq * d;
      p.gO_strideM = d; p.gO_strideH = Sq * d; p.gO_strideB = (int64_t)H * Sq * d;
      p.gQ_strideM_override = d; p.gQ_strideH = Sq * d; p.gQ_strideB = (int64_t)H * Sq * d;
      p.gK_strideM_override = d; p.gK_strideH = Sk * d; p.gK_strideB = (int64_t)H * Sk * d;
      p.gV_strideM_override = d; p.gV_strideH = Sk * d; p.gV_strideB = (int64_t)H * Sk * d;
    } else {
      // BSHD strides: token(M)=H*D, head(H)=D, batch(B)=S*H*D. O/dQ/dK/dV M-stride is fixed = H*D.
      p.q_strideM = H * d; p.q_strideH = d; p.q_strideB = (int64_t)Sq * H * d;
      p.k_strideM = H * d; p.k_strideH = d; p.k_strideB = (int64_t)Sk * H * d;
      p.v_strideM = H * d; p.v_strideH = d; p.v_strideB = (int64_t)Sk * H * d;
      p.o_strideH = d;     p.o_strideB = (int64_t)Sq * H * d;                 // o_strideM() fixed = H*d
      p.gO_strideM = H * d; p.gO_strideH = d; p.gO_strideB = (int64_t)Sq * H * d;
      p.gQ_strideH = d; p.gQ_strideB = (int64_t)Sq * H * d;                   // gQ_strideM() fixed = H*d
      p.gK_strideH = d; p.gK_strideB = (int64_t)Sk * H * d;
      p.gV_strideH = d; p.gV_strideB = (int64_t)Sk * H * d;
    }
    int lse_dim = LSE.size(2);
    p.lse_strideB = (int64_t)H * lse_dim; p.lse_strideH = lse_dim;          // must be % 8 == 0 (padded)
    p.delta_strideB = (int64_t)H * Sq; p.delta_strideH = Sq;

    // Accumulator workspace (dQ across key-blocks; dK/dV when not kept in RF).
    // [ZERO-FILL] Обнулялась ЦЕЛИКОМ: при d=256,H=8,S=2048,ns=10 это 352 МБ и 1142 мкс из 4287 мкс
    // обратного прохода (26.6%), против 141 мкс у torch. Но нужна инициализация только части gq:
    // там в `GradQTempStorage` живут `lock` и `counter`, чередуясь с полезной нагрузкой. Части gk и gv
    // (336 МБ из 352) кернел ЗАПИСЫВАЕТ, а не складывает: `gmem_tile.store(output_frags.gradK/V, ...)`
    // для не-последнего блока запросов, а первый вход помечен флагом `isFirstQuery`. Раскладка на
    // голову: [gk | gv | gq] (см. workspace_strideBH()), поэтому обнуляем ТОЛЬКО хвост gq.
    torch::Tensor wsT;
    int64_t ws = p.workspace_size();
    if (ws > 0) {
        const int64_t strideBH = p.workspace_strideBH();          // во float
        const int64_t gqEl     = p.workspace_elements_gq();       // хвост со счётчиками и замками
        const int64_t headEl   = strideBH - gqEl;                 // gk + gv: инициализация не нужна
        // Полосовое обнуление хвоста выгодно только когда хвост МАЛ: сам по себе полосовой доступ
        // дороже сплошного, и на малых областях (S=512: gq — заметная доля) сплошное `zeros` быстрее.
        // Порог измерен: полосовой путь берём, когда gq меньше четверти шага на голову.
        const bool striped = (gqEl > 0) && (gqEl * 4 < strideBH);
        if (const char* _z = fa2_env("FA2SM70_FORCE_WSZERO")) {   // A/B: 0 = всегда сплошное
          if (atoi(_z) == 0) { wsT = torch::zeros({ws}, torch::dtype(torch::kUInt8).device(Q.device())); }
          else { wsT = torch::empty({ws}, torch::dtype(torch::kUInt8).device(Q.device()));
                 if (gqEl > 0) wsT.view(torch::kFloat32).view({(int64_t)B * H, strideBH}).narrow(1, headEl, gqEl).zero_(); }
        } else if (striped) {
          wsT = torch::empty({ws}, torch::dtype(torch::kUInt8).device(Q.device()));
          wsT.view(torch::kFloat32).view({(int64_t)B * H, strideBH}).narrow(1, headEl, gqEl).zero_();
        } else {
          wsT = torch::zeros({ws}, torch::dtype(torch::kUInt8).device(Q.device()));
        }
        p.workspace = reinterpret_cast<typename AK::output_accum_t*>(wsT.data_ptr());
    } else {
        p.workspace = nullptr;
    }

    TORCH_CHECK(AK::check_supported(p), "cutlass FMHA backward: params unsupported");
    auto kernel_fn = attention_kernel_backward_batched_impl<AK>;
    int smem = int(sizeof(typename AK::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#ifdef FMHA_BWD_R7
#if !defined(FMHA_BWD_R7_FP16DQ) && !defined(FMHA_BWD_R7_TICKET)
    // [R7 d-gate] the atomic fp32 dQ accumulator is written only by the whole-B path (kOutputInRF == kMaxK<=kBI,
    // i.e. d<=128). For d>128 the lock-RMW path wrote dQ directly -> do NOT overwrite it with the zero r7_dqf.
    // (FMHA_BWD_R7_TICKET finalizes fp32->fp16 IN-KERNEL -> no copy here.)
#ifdef FMHA_BWD_COAL_FO
    if constexpr (kMaxK <= kBI) {
      if (p.fo_dq && bhsd) {
        // [COAL_FO] transpose the FRAGMENT-ORDER workspace r7_dqf -> dQ[query,d] via r7_perm (fragorder-slot ->
        // query_local*kMaxK + d). Bespoke ONE-pass convert kernel (coalesced fp16 store, no torch dispatch/
        // materialisation overhead -- the torch index_select cost ~40us and lost the tiny single-wave shape).
        int64_t out_span = (int64_t)kBI * (int64_t)kMaxK;   // per-qblk OUTPUT slice (= kBI*head_dim: 16384@d128, 8192@d64)
        int64_t ws_span  = (int64_t)kBI * (int64_t)kBI;     // per-qblk FRAGORDER slice (whole <kBI,kBI> tile: 16384)
        int64_t nqblk = (int64_t)Sq / (int64_t)kBI;
        int64_t total = (int64_t)B * (int64_t)H * nqblk * out_span;   // = B*H*Sq*d
        // r7_perm is the FORWARD map out_index -> fragorder slot (FIXED per (kBI,kMaxK), captured in-kernel on tile 0).
        // Cast to int32 once and cache (rebuilding every call was ~60us; keyed by out_span to distinguish d128/d64).
        static torch::Tensor s_inv;
        if (!s_inv.defined() || s_inv.numel() != out_span || s_inv.device() != Q.device()) {
          s_inv = r7_perm_t.narrow(0, 0, out_span).to(torch::kInt32);
        }
        int threads = 256;
        int64_t blocks = (total + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        fo_convert_kernel<<<(unsigned)blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            r7_dqf.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr<at::Half>()),
            s_inv.data_ptr<int32_t>(), out_span, ws_span, total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        dQ.copy_(r7_dqf);  // BSHD fragment-order transpose not implemented (not the tested path)
      }
    }
#else
    if constexpr (kMaxK <= kBI) dQ.copy_(r7_dqf);
#endif
#endif
#endif
    return {dQ, dK, dV};
}

// ---- Variable-length / packed backward (cu_seqlens) ---------------------------------------------
// Packed tokens [T, H, D] (K/V already expanded to H heads in Python). The cutlass backward supports
// cu_seqlens_q/k natively: batch_id indexes a sequence, advance jumps to q_start/k_start and sets
// num_queries/num_keys per-sequence. LSE is [nseq, H, lse_dim] (advanced by the ORIGINAL batch_id,
// so it matches the varlen forward output). delta is packed [H, Tq] (advanced by q_start then head).
template <int kBI, int kBJ, int kMaxK>
static std::vector<torch::Tensor> run_bwd_varlen(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,   // packed [Tq/Tk, H, D] fp16
    torch::Tensor O, torch::Tensor dO,                   // packed [Tq, H, D] fp16
    torch::Tensor LSE, torch::Tensor delta,              // [nseq, H, lse_dim] fp32, [H, Tq] fp32
    torch::Tensor cu_q, torch::Tensor cu_k, int64_t max_q, int64_t max_k,
    double scale, bool causal, int64_t num_splits) {
    using AK = AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t,
                                       true, false, false, kBI, kBJ, kMaxK, false, true>;
    int Tq = Q.size(0), H = Q.size(1), d = Q.size(2), Tk = K.size(0);
    int nseq = cu_q.size(0) - 1;
    auto opt_h = torch::dtype(torch::kFloat16).device(Q.device());
    auto dQ = torch::empty({Tq, H, d}, opt_h);
    auto dK = torch::empty({Tk, H, d}, opt_h);
    auto dV = torch::empty({Tk, H, d}, opt_h);
#ifdef FMHA_BWD_R7
    auto opt_f32 = torch::dtype(torch::kFloat32).device(Q.device());
    auto r7_dqf = torch::zeros({Tq, H, d}, opt_f32);
    auto r7_scratch = torch::empty({1}, opt_h);  // [R7] dead S^T scratch (on-chip r7_sT superseded it; size-1)
#ifdef FMHA_BWD_COAL_FO
    auto r7_perm_t = torch::zeros({8 * 64 * 32}, opt_f32);  // [COAL_FO] fragorder->(query,d) map (varlen: unused path)
#endif
#endif

    typename AK::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr   = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>());
    p.output_ptr      = reinterpret_cast<cutlass::half_t*>(O.data_ptr<at::Half>());
    p.grad_output_ptr = reinterpret_cast<cutlass::half_t*>(dO.data_ptr<at::Half>());
    p.grad_query_ptr  = reinterpret_cast<cutlass::half_t*>(dQ.data_ptr<at::Half>());
#ifdef FMHA_BWD_R7
    p.r7_st_scratch   = reinterpret_cast<cutlass::half_t*>(r7_scratch.data_ptr<at::Half>());
    p.r7_dq_f32       = r7_dqf.data_ptr<float>();
#ifdef FMHA_BWD_COAL_FO
    p.r7_perm         = r7_perm_t.data_ptr<float>();
#endif
#endif
    p.grad_key_ptr    = reinterpret_cast<cutlass::half_t*>(dK.data_ptr<at::Half>());
    p.grad_value_ptr  = reinterpret_cast<cutlass::half_t*>(dV.data_ptr<at::Half>());
    p.logsumexp_ptr = LSE.data_ptr<float>();
    p.delta_ptr     = delta.data_ptr<float>();
    p.bias_ptr = nullptr; p.grad_bias_ptr = nullptr;
    p.cu_seqlens_q_ptr = cu_q.data_ptr<int32_t>();
    p.cu_seqlens_k_ptr = cu_k.data_ptr<int32_t>();

    p.scale = (float)scale;
    p.head_dim = d; p.head_dim_value = d;
    p.num_heads = H; p.num_batches = nseq;
    p.num_queries = (int)max_q; p.num_keys = (int)max_k;                    // caps; per-seq set from cu_seqlens
    p.custom_mask_type = causal ? AK::CausalFromBottomRight : AK::NoCustomMask;
    int max_splits = ((int)max_k + 64 - 1) / 64;
    int ns = (int)num_splits; if (ns < 1) ns = 1; if (ns > max_splits) ns = max_splits;
    p.num_splits_key = ns;

    // Packed [T, H, D] strides: token(M)=H*D, head=D, batch unused (cu_seqlens jumps).
    p.q_strideM = H * d; p.q_strideH = d; p.q_strideB = 0;
    p.k_strideM = H * d; p.k_strideH = d; p.k_strideB = 0;
    p.v_strideM = H * d; p.v_strideH = d; p.v_strideB = 0;
    p.o_strideH = d;     p.o_strideB = 0;
    p.gO_strideM = H * d; p.gO_strideH = d; p.gO_strideB = 0;
    p.gQ_strideH = d; p.gQ_strideB = 0;
    p.gK_strideH = d; p.gK_strideB = 0;
    p.gV_strideH = d; p.gV_strideB = 0;
    int lse_dim = LSE.size(2);
    p.lse_strideB = (int64_t)H * lse_dim; p.lse_strideH = lse_dim;          // [nseq, H, lse_dim]
    p.delta_strideB = 0; p.delta_strideH = Tq;                             // packed [H, Tq]

    torch::Tensor wsT;
    int64_t ws = p.workspace_size();
    if (ws > 0) {
        wsT = torch::zeros({ws}, torch::dtype(torch::kUInt8).device(Q.device()));
        p.workspace = reinterpret_cast<typename AK::output_accum_t*>(wsT.data_ptr());
    } else {
        p.workspace = nullptr;
    }

    TORCH_CHECK(AK::check_supported(p), "cutlass FMHA varlen backward: params unsupported");
    auto kernel_fn = attention_kernel_backward_batched_impl<AK>;
    int smem = int(sizeof(typename AK::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#ifdef FMHA_BWD_R7
#if !defined(FMHA_BWD_R7_FP16DQ) && !defined(FMHA_BWD_R7_TICKET)
    // [R7 d-gate] the atomic fp32 dQ accumulator is written only by the whole-B path (kOutputInRF == kMaxK<=kBI,
    // i.e. d<=128). For d>128 the lock-RMW path wrote dQ directly -> do NOT overwrite it with the zero r7_dqf.
    // (FMHA_BWD_R7_TICKET finalizes fp32->fp16 IN-KERNEL -> no copy here.)
    if constexpr (kMaxK <= kBI) dQ.copy_(r7_dqf);
#endif
#endif
    return {dQ, dK, dV};
}

std::vector<torch::Tensor> attn_bwd_cutlass_varlen(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO,
    torch::Tensor LSE, torch::Tensor delta,
    torch::Tensor cu_q, torch::Tensor cu_k, int64_t max_q, int64_t max_k,
    double scale, bool causal, int64_t num_splits) {
    int d = Q.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "varlen backward: fp16 only");
    TORCH_CHECK(d <= 512, "varlen backward: head_dim <= 512");
#ifdef FMHA_BWD_D128_ONLY
    TORCH_CHECK(false, "varlen disabled in FMHA_BWD_D128_ONLY harness"); return {};
#else
    // [R7 partial-block correctness] varlen per-sequence query lengths (cu_seqlens) are arbitrary -> partial 128-
    // query blocks are the norm, and R7's whole-B dQ transpose mis-corresponds slots on a partial high 32-strip.
    // So route ALL varlen d<=128 to <64,64,256>: kMaxK=256 > kBlockSizeI=64 => kOutputInRF=false => the exact
    // width-panelled DIRECT (lock-RMW) GradQ, no S^T transpose (the exercised d>128 workhorse; handles head_dim<=128).
    if (d <= 256) return run_bwd_varlen<64, 64, 256>(Q, K, V, O, dO, LSE, delta, cu_q, cu_k, max_q, max_k, scale, causal, num_splits);
    return run_bwd_varlen<64, 64, 512>(Q, K, V, O, dO, LSE, delta, cu_q, cu_k, max_q, max_k, scale, causal, num_splits);
#endif
}

// Entry: K/V already expanded to H heads, delta precomputed. bhsd=false -> BSHD [B,S,H,D] contiguous
// (legacy); bhsd=true -> native BHSD [B,H,S,D], letting the wrapper skip the transpose glue.
std::vector<torch::Tensor> attn_bwd_cutlass(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO,
    torch::Tensor LSE, torch::Tensor delta,
    double scale, bool causal, int64_t num_splits, int64_t blk, bool bhsd = false) {
    int d = Q.size(3);
    int64_t Sq = bhsd ? Q.size(2) : Q.size(1);
    int64_t Sk = bhsd ? K.size(2) : K.size(1);
    int64_t BH = bhsd ? Q.size(0) * Q.size(1) : Q.size(0) * Q.size(2);  // B*H = no-split grid; 80 SMs
    // Route to the 240-reg no-split kernel wherever B*H fills enough of the 80 SMs. Splitting the key
    // axis does NOT help our kernel the way it helps torch's: our split kernel (kEnableSplitKeys=true,
    // 255-reg, spills) has a high per-block penalty, so measured (A/B sweep, research dir) the no-split
    // 240-reg kernel is fastest once B*H reaches ~half a wave, and stays fastest all the way up (B*H =
    // 48,64,96,128,160,256 all prefer no-split). Splitting only pays under severe under-fill (B*H<40:
    // e.g. B*H=8 is ~3.4x faster split). So the crossover is ~40*waves (was 60 -- that wrongly split
    // B*H=48/96, ~7-10% slower there). Registers are NOT torch's lever (torch is 240-reg but spills
    // 1024B, more than ours, at the same 1-block/SM occupancy); its edge is per-block efficiency. When
    // we do split (B*H<40), the wrapper's num_splits (~256/B*H, filling the SMs) is near-optimal.
    // Sk<=1024: at long key axes the serial no-split CTA loses to finer key-parallelism.
    int64_t waves = (BH + 79) / 80;
    // [R7 re-tune] The OLD heuristic below routed B*H>=~40 to the 240-reg NO-split kernel, because the pre-R7
    // split kernel spilled (255-reg) and lost its key-parallelism edge. R7 REMOVED that spill (238-254 reg,
    // no spill) AND made the dQ cross-split fold cheap concurrent atomics -> the split kernel now DOMINATES the
    // no-split kernel at EVERY B*H measured: e.g. B*H=64,S1024 no-split 0.82x vs split(ns=16) 1.065x; B*H=128
    // 0.807x vs split(ns=8) 1.085x (+30-34%). The no-split kernel under-fills (B*H CTAs < 80 SMs) while the
    // split kernel fills via key-parallelism at no register cost. So: ALWAYS split (nosplit_ok=false); the
    // 240-reg no-split kernel is now reachable only via FA2SM70_FORCE_NOSPLIT=1 (kept for A/B).
    bool nosplit_ok = false;  (void)waves;
    if (const char* _fns = fa2_env("FA2SM70_FORCE_NOSPLIT")) nosplit_ok = (atoi(_fns) != 0);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "backward: fp16 only");
    TORCH_CHECK(d <= 512, "backward: head_dim <= 512 (padded)");
    // Pick the seqaligned kernel (skips per-tile boundary predication -- torch's ~1.2-1.3x-faster
    // `_seqaligned_sm70` variant) whenever the block-alignment precondition holds, else the bounds-checked
    // one. Exact either way; the aligned kernel asserts the precondition internally.
#define RUN_BWD(BI, BJ, MK)                                                                            \
    ((Sq % (BI) == 0 && Sk % (BJ) == 0)                                                                \
         ? run_bwd<BI, BJ, MK, false, true>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd)     \
         : run_bwd<BI, BJ, MK, false, false>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd))
    // No-split (kEnableSplitKeys=false) = torch's 240-register / no-spill kernel. It cannot split the key
    // axis, so we use it only when B*H alone fills the 80 SMs (B*H>=64, measured crossover); there it is
    // 7-10% faster than the 255-reg split kernel. Below that B*H the split kernel's key-parallelism wins.
#define RUN_BWD_NS(BI, BJ, MK)                                                                            \
    ((Sq % (BI) == 0 && Sk % (BJ) == 0)                                                                   \
         ? run_bwd<BI, BJ, MK, false, true,  false>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd)\
         : run_bwd<BI, BJ, MK, false, false, false>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd))
    // blk selects (kBI x kBJ): 0=64x64, 1=64x128, 2=128x64 (all fit 96KB smem; 128x128 excluded at 97KB).
#ifdef FMHA_BWD_D128_ONLY
    // Fast-iteration harness: compile ONLY the d128 no-split kernel (~3 min vs ~15 for all configs), for
    // developing the slab-major prefetch restructure. Build with -DFMHA_BWD_D128_ONLY=1. Off by default.
    // FMHA_BWD_SLAB_KBI sets kBlockSizeI (query tile): 128 (shipped) or 32 (slab-config, to measure the
    // small-tile MMA-efficiency penalty that decides simple-kBI=32 vs hybrid build).
#ifndef FMHA_BWD_SLAB_KBI
#define FMHA_BWD_SLAB_KBI 128
#endif
#ifndef FMHA_BWD_SLAB_KBJ
#define FMHA_BWD_SLAB_KBJ 64
#endif
    (void)nosplit_ok;
    TORCH_CHECK(d <= 128 && blk == 2, "FMHA_BWD_D128_ONLY: d<=128, blk==2 only");
#if defined(FMHA_BWD_MEASURE_DECOMP) && FMHA_BWD_MEASURE_DECOMP == 1
    // [DECOMP] dV-only family (kComputeVOnly): C1,C2 only -> only dV resident. Measure its reg+smem occupancy.
    return run_bwd<FMHA_BWD_SLAB_KBI, 64, 128, false, true, false, /*kCVO*/ true, /*kSGV*/ false>(
        Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd);
#elif defined(FMHA_BWD_MEASURE_DECOMP) && FMHA_BWD_MEASURE_DECOMP == 2
    // [DECOMP] dQ+dK family (kSkipGradV): recompute C1, then C3,C4,C5 -> dK+dQ resident, no dV.
    return run_bwd<FMHA_BWD_SLAB_KBI, 64, 128, false, true, false, /*kCVO*/ false, /*kSGV*/ true>(
        Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd);
#else
    return RUN_BWD_NS(FMHA_BWD_SLAB_KBI, FMHA_BWD_SLAB_KBJ, 128);
#endif
#else
    // [R7 partial-block correctness] R7's reassociated whole-B dQ^T=k^T.S^T mis-corresponds fragment slots on a
    // partially-entered high 32-query sub-strip -> a WRONG dQ on the last ~32 queries of a partial 128-query block
    // (both causal AND non-causal; confirmed vs torch on Sq=447/500/1000; full 128-blocks are exact, which is why
    // the test suite's size-64 partials missed it). measured located it as the
    // transposing S^T materialization's slot-correspondence break and recommends the transpose-free DIRECT GradQ.
    // Route any non-128-multiple Sq at d<=128 to RUN_BWD(64,64,256): kMaxK=256 > kBlockSizeI=64 => kOutputInRF=false
    // => the exact width-panelled DIRECT (lock-RMW) GradQ, no S^T transpose measured; handles head_dim<=128.
    // (The (64,64,128) tile is the never-reached blk!=1,2 fallback -- _bwd_run always passes blk=2 -- and is broken;
    // (64,64,256) is the exercised d>128 workhorse.) Sq%128==0 (=> every query block full) keeps R7 + fragment-order
    // coalescing -- all the perf shapes. A/B: set FA2SM70_FORCE_R7PARTIAL=1 to keep the (buggy) R7 path on partial Sq.
    if (d <= 128 && (Sq % 128 != 0) && fa2_env("FA2SM70_FORCE_R7PARTIAL") == nullptr)
        return RUN_BWD(64, 64, 256);
    if (d <= 64) {
        if (blk == 1) return RUN_BWD(64, 128, 64);
        if (blk == 2) return (nosplit_ok) ? RUN_BWD_NS(128, 64, 64) : RUN_BWD(128, 64, 64);
        return RUN_BWD(64, 64, 64);
    }
    if (d <= 128) {
        if (blk == 1) return RUN_BWD(64, 128, 128);
        // [note] a 32-wide key-band (kBJ=32) to fill small shapes does NOT compile on Volta -- the cutlass
        // pitch-linear thread map requires ShapeInAccesses divisible by WarpThreadArrangement + kStages>=2,
        // which a 128x32 tile violates. So the B1H8-S512 under-fill needs a query-split axis, not a narrow band.
        if (blk == 2) return (nosplit_ok) ? RUN_BWD_NS(128, 64, 128) : RUN_BWD(128, 64, 128);
        return RUN_BWD(64, 64, 128);
    }
    // d > 128: kOutputInRF is false regardless of tile (kMaxK > kBlockSizeI), so dK/dV go through the
    // gmem workspace (kNeedsAccumGradK/V) -- the analog of the forward's output-accumulator path. delta
    // is precomputed in torch (kKernelComputesDelta is false here). 64x64 keeps smem/registers modest.
    // [large-head experiment] ncu showed torch's d256/512 backward is fmha_cutlassB_128x64 (128-query tile, 8 warps,
    // 176 reg no-spill, NO key-split) at 3.09ms vs our 64x64/4-warp/255-reg-spill/ns-split 4.37ms + 9x memory. Mirror
    // torch's shape: a 128x64 NO-SPLIT tile (kESK=false -> ns=1, no gmem split workspace). A/B: FA2SM70_LARGEHEAD_128.
    if (fa2_env("FA2SM70_LARGEHEAD_128") != nullptr) {
        if (d <= 256) return RUN_BWD_NS(128, 64, 256);
        return RUN_BWD_NS(128, 64, 512);
    }
    // [large-head, THE MISSING CONFIG] torch's winning kernel is the 128x64 tile (8 warps, 176 reg, no spill) and
    // its grid (200 blocks @ B1H8S2048) shows it ALSO parallelises over key blocks -- i.e. same fill as our ns-split.
    // So the gap is per-block efficiency (tile 128x64 vs our 64x64, 8 warps vs 4, no-spill vs 255-reg-spill), NOT
    // fill. We had only ever tried 128x64 WITHOUT split (under-fills) and 64x64 WITH split (spills). Try the union:
    // the 128x64 tile WITH key-split. A/B: FA2SM70_LH_SPLIT128.
    if (fa2_env("FA2SM70_LH_SPLIT128") != nullptr) {
        if (d <= 256) return RUN_BWD(128, 64, 256);
        return RUN_BWD(128, 64, 512);
    }
    // [TWO-PASS probe] Pass KV only: 128x64 tile, key-split fill (kESK=true, ns from _bwd_run/FORCE_NS), compute
    // dK+dV, SKIP GradQ -> each key-split owns disjoint keys -> dK/dV need NO atomic sink (no 9x scratch, no spill).
    // dQ output is left GARBAGE (skipped) -- this probe measures only whether dK/dV fill fast + low memory.
    if (fa2_env("FA2SM70_TWOPASS_KV") != nullptr) {
        if (d <= 256) return run_bwd<128, 64, 256, false, false, true, false, false, /*kSGQ*/true>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd);
        return run_bwd<128, 64, 512, false, false, true, false, false, /*kSGQ*/true>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd);
    }
    // [TWO-PASS, the large-head target] Two sink-homogeneous launches, each conflict-free (the
    // decomposition: own the OUTPUT, not the axis), so NEITHER needs a cross-tile atomic sink:
    //   Pass KV -- tiles own disjoint KEY bands, sweep queries, compute dK+dV  (kSkipGradQ, key-split fill)
    //   Pass Q  -- tiles own disjoint QUERY bands, sweep keys, compute dQ      (kComputeQOnly, query-split fill)
    // Cost is 7u vs the fused 5u (Class-A S/P/G is rebuilt in both passes), but it is spread over all 80 SMs
    // instead of being serialised behind an under-filled or spilling single pass.
    //
    // [MEASURED 2026-07-26 -- DEAD, do not retry] The register profile was the encouraging part and it is
    // real: at <128,64,256/512> the two halves are kEnableSplitKeys=1 with NO spill (kSkipGradQ 174 reg,
    // kComputeQOnly 234 reg) against the fused split kernel's 255 reg + 16-40 B spill. It does not help.
    //   ratio = torch/ours, causal fp16, clock-locked 1350 MHz, median of 3x30 iters:
    //                        fused (shipped)   TWO-PASS
    //     B1H8  S2048 d256       0.850           0.686
    //     B1H8  S4096 d256       0.867           0.731
    //     B2H8  S2048 d256       0.917           0.745
    //     B1H16 S2048 d256       0.920           0.746
    //     B4H8  S1024 d256       0.977           0.769
    //     B1H8  S2048 d512       0.835           0.667
    //     B1H8  S4096 d512       0.865           0.723
    //     B2H8  S2048 d512       0.897           0.718
    //   geomean 0.890 -> 0.7225. dK/dV are EXACT (3.7e-4 / 2.9e-4); Pass Q's dQ is WRONG (relL2 1.3e-1 ..
    //   2.6e-1), so the timing above is if anything FLATTERING to two-pass (a broken pass is not a slower
    //   one). Fixing dQ cannot save it: the decomposition pays 7u for 5u of work, i.e. +40%, and it can only
    //   be repaid in fill -- but the fused key-split ALREADY puts 64-80 CTAs on the 80 SMs at B*H=8, so there
    //   is no 40% of idle SMs left to buy. The premise "under-filled single pass" was true for the pre-split
    //   dispatch, not for the shipped one.
    if (fa2_env("FA2SM70_TWOPASS") != nullptr) {
        int64_t qblocks = (Sq + 127) / 128;                  // one query block per Pass-Q tile
#define RUN_TWOPASS(MK)                                                                                   \
        do {                                                                                              \
            auto kv = run_bwd<128, 64, MK, false, false, true, false, false, /*kSGQ*/ true, false>(       \
                Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, 0.0, 0, 0, bhsd, 0);               \
            auto qq = run_bwd<128, 64, MK, false, false, true, false, false, false, /*kCQO*/ true>(       \
                Q, K, V, O, dO, LSE, delta, scale, causal, 1, 0.0, 0, 0, bhsd, qblocks);                  \
            return {qq[0], kv[1], kv[2]};   /* dQ from Pass Q; dK,dV from Pass KV */                      \
        } while (0)
        if (d <= 256) RUN_TWOPASS(256);
        RUN_TWOPASS(512);
#undef RUN_TWOPASS
    }
    // [ncu 2026-07-26 -- what the d>128 backward is ACTUALLY bound by, so nobody re-guesses]
    // B1H8 S2048 d256, the shipped split path:
    //     long_scoreboard (global-memory latency)  2.93   <- dominant stall
    //     wait (arithmetic dependency)             0.81
    //     DRAM throughput                        47.0% of peak
    //     SM throughput                          19.5%
    //     warps active                           12.5%   (exactly 8 warps of 64, at 255 registers)
    //     ALU 79.3M / tensor 89.1M / LSU 23.1M instructions
    // So this kernel is MEMORY-LATENCY bound with too few warps to hide it -- NOT ALU-bound. That matters
    // because the decode kernel next door WAS ALU-bound (85% of its ALU was the emulated e4m3 conversion,
    // fixed in fd5a84e for 1.5-1.8x); the same trick has nothing to bite on here. The two remaining levers
    // are therefore (a) more warps, which registers forbid -- 65536/255 = 8 warps whatever the CTA shape,
    // and torch sits at the same 8 -- or (b) hiding the latency, i.e. the off-chip prefetch schedule
    // (research/, ~x1.33, multi-week). Do not re-run ALU micro-optimisation experiments here.
    // DEFAULT for d>128: the 128-query tile WITH key-split. Measured against torch SDPA (V100, causal,
    // fp16; ratio = torch/ours, >1 = we win), across the three configurations that exist here:
    //                        64x64 split   128x64 no-split   128x64 SPLIT (this)
    //   B1 H8  S2048 d256        0.51           0.28              0.63
    //   B1 H8  S4096 d256        0.45           0.27              0.58
    //   B2 H8  S2048 d256        0.48           0.46              0.58
    //   B1 H16 S2048 d256        0.49           0.46              0.60
    //   B1 H8  S2048 d512        0.49           0.29              0.60
    //   B1 H8  S4096 d512        0.46           0.27              0.58
    //   B2 H8  S2048 d512        0.46           0.46              0.57
    // i.e. +24% over the old default, everywhere, and it needs the UNION of the two ideas: the wide
    // query tile alone under-fills (128x64 no-split is the worst of the three at small B*H), and the
    // key-split alone keeps the narrow tile's 4 warps and 255-register spill. Widening kBI is what cuts
    // the dK/dV gmem-accumulator round-trips, whose count is Sq/kBI -- the same structure that made the
    // forward's query tile the lever at d=256.
    // Exact on every shape tried, including the partial-Sq ones that this tile does not seqalign
    // (S=500/447/320/130), non-causal, and GQA/MQA: dQ/dK/dV relL2 3.0e-4..4.6e-4 vs an fp32 reference.
    // A/B the old default with FA2SM70_LH_64: it is kept only as a fallback, not as a tuned choice.
    // DEAD END, do not retry: kBI=256 would make `kMaxK <= kBI` true at d=256 and open the in-register
    // output path (R7) that makes d<=128 fast -- but <256,64,256> does not instantiate on sm_70
    // ("zero-sized variable tb_frag_B" in CustomMmaPipelined, GemmShape<256,64,32>). Note this is the
    // SAME wall that blocks the forward's 256-key tile (tb_frag_A, <*,256,256>): the Volta pipelined MMA
    // simply has no valid thread map at these shapes, in either direction. And it is not an smem
    // question -- it fails at template instantiation, before any size is computed.
    if (fa2_env("FA2SM70_LH_64") != nullptr) {
        if (d <= 256) return RUN_BWD(64, 64, 256);
        return RUN_BWD(64, 64, 512);
    }
    // [d>128 no-split at scale] The split kernel's key-parallelism costs REGISTERS: measured from the
    // built cubin, kEnableSplitKeys=1 at d=256/512 is 255 registers WITH spill (32 B), while the same
    // tile at kEnableSplitKeys=0 is 173/170 registers and NO spill -- essentially torch's own profile
    // (176, no spill). The split is worth its spill only while B*H alone cannot fill the 80 SMs.
    // Measured (ratio = torch/ours, >1 = we win; d=256 S=1024 unless noted):
    //     B*H=32   split 0.97   nosplit 0.74
    //     B*H=64   split 1.01   nosplit 1.04      (B4H16: 1.03 vs 1.05)
    //     B*H=128  split 0.99   nosplit 1.00      (B16H8 S512: 0.98 vs 1.00)
    //     B*H=64 d512  split 1.00  nosplit 1.01   (B4H16 d512: 0.99 vs 1.00)
    // So hand the GPU to the spill-free kernel once B*H >= 64, and keep the tuned key-split below it.
    // [tile probe -- GEOMETRY IS EXHAUSTED, measured] All four tiles swept at d>128 with the tuned split
    // rule (ratio = torch/ours at B*H = 8 / 16 / 32, d=256):
    //     128x64  (shipped)  0.85 / 0.92 / 0.98
    //     64x64              0.58 / 0.60 / 0.66
    //     64x128             0.70 / 0.75 /  -
    //     128x128            does not launch: "CUDA error: invalid argument" (smem over 96 KB, even
    //                        though d>128 uses only 74 KB in the 128x64 config -- the wider key tile
    //                        adds more than the headroom)
    // 128x64 wins everywhere. Keep the probe so nobody re-runs this sweep.
    // FA2SM70_LH_TILE=1|2|3 -> 64x64 | 128x128 | 64x128.
    if (const char* _t = fa2_env("FA2SM70_LH_TILE")) {
        switch (atoi(_t)) {
            case 1: return (d <= 256) ? RUN_BWD(64, 64, 256)   : RUN_BWD(64, 64, 512);
            case 2: return (d <= 256) ? RUN_BWD(128, 128, 256) : RUN_BWD(128, 128, 512);
            case 3: return (d <= 256) ? RUN_BWD(64, 128, 256)  : RUN_BWD(64, 128, 512);
            default: break;
        }
    }
    // [K65536 probe] torch's sm_70 backward has only TWO kMaxK instantiations -- k128 and k65536 -- and
    // uses the k65536 one for every large head. We instead specialise kMaxK=256/512, believing that
    // helps. But kMaxK feeds MakeCustomMma (kernel_backward.h:504) and the k-block count (:511), i.e. it
    // CONFIGURES THE QK MAINLOOP, so a tight kMaxK can ask for a more resident schedule and cost the
    // registers we are short of. Probe the torch shape directly: FA2SM70_BWD_K65536=1.
    // [kBJ32 -- WALL REMOVED, SHAPE MEASURED DEAD. Do not re-run this sweep.]
    // THE WALL IS GONE. cutlass's warp-raked thread map hard-wired WarpThreadArrangement<8,4>, i.e. it
    // assumed the tile's contiguous extent holds >= 8 vector accesses (kN >= 64 halves at 128-bit). A
    // 32-wide tile has 4, and 4/8 == 0 by integer division, so the fragment came out EMPTY and the
    // build died three headers away in NumericArrayConverter<half_t,half_t,N=0> ("zero-sized
    // variable"). default_mma_core_sm70.h now sizes that span to what the tile actually has and gives
    // the freed threads to the strided axis; shapes that already built keep the exact <8,4> they had.
    // Second blocker after that: CustomMmaPipelined's static_assert(kStages >= 2), which is the classic
    // path's double buffer -- the whole-resident path takes a separate `if constexpr` branch with no
    // global loads and no wrap, where kStages means "kblocks resident" and a one-kblock contraction is
    // legitimate. Errors 5 -> 1 -> 0; <128,32,256> and <128,32,512> now build and run.
    //
    // AND THE PREMISE ABOVE WAS WRONG. "a fragment is kBJ*kN/threads, so kBJ=32 gives 32*128/256 = 16
    // floats/thread" used 256 threads for a tile that only ever has 128: kNumWarpsPerBlock is
    // kBI*kBJ/1024, so halving kBJ halves the warp count with it. The true residency cost is
    // 32*d/kBI floats per thread per tensor and is INDEPENDENT of kBJ -- derivation in
    // kernel_backward.h, [RESIDENCY COST]. kBJ=32 buys no register room at all.
    //
    // MEASURED (clock-locked V100, ratio torch/ours, 7 shapes d=256/512, B*H=8..64):
    //     default (kBJ=64)   geomean 0.979   1 win of 7
    //     kBJ=32             geomean 0.677   0 wins   AND dQ is wrong (relL2 0.88-0.90, dK/dV exact)
    // 31% slower: 4 warps per block halve what the block brings to the SM and double the per-key-block
    // prologue. The dQ break was not chased -- the shape is dead on speed either way.
    // Kept reachable because the cutlass fix is GENERAL (any future narrow-N tile instantiates now),
    // not because this configuration is worth revisiting.
    // COMPILE-TIME gated: a runtime getenv does not stop a template from being instantiated -- my first
    // version used only the getenv and broke every build until that was noticed.
#ifdef FMHA_BWD_KBJ32
    if (fa2_env("FA2SM70_BWD_KBJ32") != nullptr) {
        const bool ns_ok = (BH >= 64);
        if (d <= 256) return ns_ok ? RUN_BWD_NS(128, 32, 256) : RUN_BWD(128, 32, 256);
        return ns_ok ? RUN_BWD_NS(128, 32, 512) : RUN_BWD(128, 32, 512);
    }
#endif
    if (fa2_env("FA2SM70_BWD_K65536") != nullptr) {
        const bool ns_ok = (BH >= 64);
        return ns_ok ? RUN_BWD_NS(128, 64, 65536) : RUN_BWD(128, 64, 65536);
    }
    const bool ns_ok_large = (BH >= 64);   // BH is computed above (bhsd-aware)
    if (const char* _e = fa2_env("FA2SM70_LH_NOSPLIT")) {
        if (atoi(_e) == 0) { if (d <= 256) return RUN_BWD(128, 64, 256); return RUN_BWD(128, 64, 512); }
    }
    if (d <= 256) return ns_ok_large ? RUN_BWD_NS(128, 64, 256) : RUN_BWD(128, 64, 256);
    return ns_ok_large ? RUN_BWD_NS(128, 64, 512) : RUN_BWD(128, 64, 512);
#endif
#undef RUN_BWD
#undef RUN_BWD_NS
}

// Backward with dropout-mask replay (kApplyDropout=true). Uses 64x64 tiles: the 128x64 tile plus the
// dropout mask/rng state needs 99072 B > the 96KB V100 smem limit (launch fails "invalid argument"),
// while 64x64 fits at 49664 B. The Philox mask is position-based (per (i,j)), so the tile size does not
// change the mask values -> grads stay exact. Same BSHD contract; mask reproduced from (seed, offset).
std::vector<torch::Tensor> attn_bwd_cutlass_dropout(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO,
    torch::Tensor LSE, torch::Tensor delta,
    double scale, bool causal, int64_t num_splits,
    double dropout_p, int64_t seed, int64_t offset) {
    int d = Q.size(3);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "dropout backward: fp16 only");
    TORCH_CHECK(d <= 128, "dropout backward: head_dim <= 128");
#ifdef FMHA_BWD_D128_ONLY
    TORCH_CHECK(false, "dropout disabled in FMHA_BWD_D128_ONLY harness"); return {};
#else
    // [R7 partial-block correctness] the d<=64 dropout tile <64,64,64> is R7 (kOutputInRF, kMaxK=64<=kBI=64) and so
    // mis-computes dQ on a partial high 32-strip. Route a non-128-multiple Sq to <64,64,256> (kMaxK=256>kBI=64 =>
    // lock-RMW direct GradQ, exact; handles head_dim<=64). Sq is the BSHD token axis here (Q is [B,S,H,D]).
    int64_t Sq = Q.size(1);
    if (d <= 64 && (Sq % 128 != 0))
        return run_bwd<64, 64, 256, true>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, dropout_p, seed, offset);
    if (d <= 64) return run_bwd<64, 64, 64, true>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, dropout_p, seed, offset);
    return run_bwd<64, 64, 128, true>(Q, K, V, O, dO, LSE, delta, scale, causal, num_splits, dropout_p, seed, offset);
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_bwd_cutlass", &attn_bwd_cutlass, "sm70 cutlass FMHA backward (dQ, dK, dV)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("O"), pybind11::arg("dO"), pybind11::arg("LSE"), pybind11::arg("delta"),
          pybind11::arg("scale"), pybind11::arg("causal"), pybind11::arg("num_splits") = 1,
          pybind11::arg("blk") = 0, pybind11::arg("bhsd") = false);
    m.def("attn_bwd_cutlass_varlen", &attn_bwd_cutlass_varlen, "sm70 cutlass FMHA backward, variable-length (cu_seqlens)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("O"), pybind11::arg("dO"), pybind11::arg("LSE"), pybind11::arg("delta"),
          pybind11::arg("cu_q"), pybind11::arg("cu_k"), pybind11::arg("max_q"), pybind11::arg("max_k"),
          pybind11::arg("scale"), pybind11::arg("causal"), pybind11::arg("num_splits") = 1);
    m.def("attn_bwd_cutlass_dropout", &attn_bwd_cutlass_dropout, "sm70 cutlass FMHA backward with dropout replay",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("O"), pybind11::arg("dO"), pybind11::arg("LSE"), pybind11::arg("delta"),
          pybind11::arg("scale"), pybind11::arg("causal"), pybind11::arg("num_splits"),
          pybind11::arg("dropout_p"), pybind11::arg("seed"), pybind11::arg("offset"));
}
