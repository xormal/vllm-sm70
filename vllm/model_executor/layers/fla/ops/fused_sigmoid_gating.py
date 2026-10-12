# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch

from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        # Режим 'all': читаем из одного блока, пишем в другой (задача 194).
        "HAS_OUT_INDICES": lambda args: args["ssm_state_indices_out"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_indices_out,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    HAS_OUT_INDICES: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all_tokens = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_tokens = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all_tokens + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale

        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            if HAS_OUT_INDICES:
                final_state_idx = tl.load(ssm_state_indices_out + i_n).to(tl.int64)
            else:
                final_state_idx = tl.load(
                    ssm_state_indices + i_n * stride_indices_seq + i_t
                ).to(tl.int64)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    # Режим 'all': куда положить состояние, если это не тот блок, откуда оно прочитано.
    ssm_state_indices_out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    branch_at: int = 0,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """Fuse sigmoid gating with recurrent delta-rule update."""
    # [FA2/SM70, задача 180 -- НАТИВНАЯ ВЕТКА, БЕЗ ПИТОН-ПРОКЛАДОК ПОВЕРХ ИМПОРТОВ]
    # Слитый сигмоидный гейтинг + рекуррент нашим ядром volta_gdn_rec (fa2_sm70/gdn_scan.cu):
    # состояние в регистрах, HBM дважды, гейтинг (softplus/sigmoid) внутри ядра. Сверено с этим
    # же Triton-ядром на 12 случаях (o relL2 <= 5e-05, пул ~5e-08), в бою гейт '391' и декод
    # без потерь. Включение: FA2SM70_GDN_REC=1 (объявлен в envs.py -- иначе компил-кэш не
    # различает REC=0/1 и ветка МОЛЧА не участвует). Откат ТОЛЬКО по форме; нехватка памяти
    # отдаётся движку (запасной Triton-путь тяжелее основного, урок 11.08).
    if os.environ.get("FA2SM70_GDN_REC", "0") == "1":
        try:
            _hv = v.shape[2]
            _a3 = a if a.dim() == 3 else a.view(1, -1, _hv)
            _b3 = b if b.dim() == 3 else b.view(1, -1, _hv)
            if (initial_state is None or not inplace_final_state or is_kda
                    or float(beta) != 1.0 or float(threshold) != 20.0
                    or q.shape[-1] != 128 or v.shape[-1] != 128
                    or _hv % q.shape[2] != 0):
                raise NotImplementedError("форма вне нашего рекуррента")
            _scale = q.shape[-1] ** -0.5 if scale is None else scale
            # ЕДИНЫЙ ЦЕЛОЧИСЛЕННЫЙ ТИП НА ВСЕЙ ТРОЙКЕ. Ядро инстанцируется по ОДНОМУ типу
            # индекса, а приходить могут разные: таблица блоков int32, cu_seqlens int64,
            # gather по .long() -- тоже int64. Прежде приводился только cu, и пара
            # (индексы чтения, индексы записи) могла разъехаться -- ядро отказывало
            # «expected Int but found Long», и декод молча уходил на Triton.
            _cu = cu_seqlens
            _idx_in = ssm_state_indices
            _idx_out = ssm_state_indices_out
            _тип = torch.int32
            # Счётчик принятых токенов ядро читает как int32 (data_ptr<int>): приходит он
            # 64-битным, и это давало тот же тихий уход на Triton, что и индексы.
            _nacc = (num_accepted_tokens.to(_тип)
                     if num_accepted_tokens is not None else None)
            if _idx_in is not None:
                _idx_in = _idx_in.to(_тип)
            if _idx_out is not None:
                _idx_out = _idx_out.to(_тип)
            if _cu is not None:
                _cu = _cu.to(_тип)
            from fa2_sm70 import _ext as _fa2_ext
            _o = _fa2_ext.gdn_ext().gdn_rec_gating(
                A_log.float(), _a3.contiguous(), _b3.contiguous(), dt_bias.float(),
                q.contiguous(), k.contiguous(), v.contiguous(), float(_scale),
                initial_state, _idx_in, _nacc, _cu,
                bool(use_qk_l2norm_in_kernel),
                # Куда писать состояние, если это НЕ тот блок, откуда читали (режим 'all').
                # None в остальных режимах -- ядро тогда пишет по прежнему индексу.
                (_idx_out.contiguous() if _idx_out is not None else None),
                # масштабы состояния: нужны только при int16-пуле (ход 20.08); при fp16/fp32
                # передаём None и ядро их не читает
                # ПРОБНИК «без масштабов» ОТВЕРГНУТ 06.09: при int16-пуле ядро без
                # массива масштабов не запускается вовсе (движок не встаёт), поэтому
                # локализовать падающее обращение так нельзя.
                getattr(initial_state, "fa2sm70_scales", None),
                # [ДЕРЕВО, 05.09] строка ветвления: b0 продолжает ЯКОРЬ, а не хвост ветви A
                int(branch_at))
            global _FA2SM70_REC_OURS
            try:
                _FA2SM70_REC_OURS += 1
            except NameError:
                _FA2SM70_REC_OURS = 1
            if _FA2SM70_REC_OURS & (_FA2SM70_REC_OURS - 1) == 0:
                import sys as _sys
                print(f"[fa2_sm70 gdn] НАТИВНЫЙ РЕКУРРЕНТ декода отработал: {_FA2SM70_REC_OURS}",
                      file=_sys.stderr, flush=True)
            return _o, initial_state
        except torch.OutOfMemoryError:
            raise
        except Exception as _e:  # noqa: BLE001 -- отказ формы: путь образца ниже
            # ОТКАТ ЗАПРЕЩЁН ПРИ int16-ПУЛЕ. Triton-путь читает состояние как ЧИСЛА, а в
            # int16-пуле лежат КОДЫ, масштаб к которым знает только наше ядро. Тихая порча
            # тут страшнее падения: ответ останется правдоподобным (задача 195).
            if (initial_state is not None
                    and getattr(initial_state, "dtype", None) == torch.int16):
                raise RuntimeError(
                    "[fa2_sm70] состояние GDN в int16, а наш рекуррент не пошёл "
                    f"({type(_e).__name__}: {_e}). Откат на Triton запрещён: он прочитает "
                    "коды как числа. Выключите FA2SM70_GDN_I16 или почините форму.") from _e
            if os.environ.get("FA2SM70_GDN_STRICT", "0") == "1":
                raise RuntimeError(
                    f"[fa2_sm70] СТРОГИЙ РЕЖИМ: рекуррент GDN не пошёл нашим ядром: "
                    f"{type(_e).__name__}: {_e}") from _e
            global _FA2SM70_REC_FB
            try:
                _FA2SM70_REC_FB += 1
            except NameError:
                _FA2SM70_REC_FB = 1
            if _FA2SM70_REC_FB == 1:
                import sys as _sys
                print(f"[fa2_sm70 gdn] нативный рекуррент: откат на Triton: "
                      f"{type(_e).__name__}: {_e}", file=_sys.stderr, flush=True)
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # ЗАПОЛНЕНИЕ МАШИНЫ: BV прибит константой, а правило выбора УЖЕ НАПИСАНО -- в соседней ветке.
    #
    # `min(next_power_of_2(V), 32)` -- это не выбор, а константа: при боевой геометрии (V=128, N=1,
    # HV=24) она даёт NV=4 и грид (1, 4, 24) = 96 CTA на 80 SM, то есть 1.2 волны и 4.8 варпа на SM
    # из 64 слотов. Рядом, в fused_recurrent.py:68, лежит `_select_sm70_bv` -- отлаженное правило,
    # которое целится в 2 волны (160 CTA) и на той же геометрии выбрало бы BV=16 -> 192 CTA.
    # Оно просто не вызывается ИЗ ЭТОЙ ветки, а боевой ходит именно сюда.
    #
    # ГЕЙТ С УМОЛЧАНИЕМ «КАК СЕЙЧАС». Этот файл лежит в дереве vLLM, а не в нашем, поэтому его НЕ
    # закрывают две ветки дерева ядер (боевой слепок / рабочее): правка здесь достала бы боевой при
    # ближайшем подъёме. Значит поведение по умолчанию НЕ меняется, пока ход не замерен;
    # включается VLLM_SM70_FLA_GATING_BV=1.
    BK = triton.next_power_of_2(K)
    if os.environ.get("VLLM_SM70_FLA_GATING_BV") == "1":
        from vllm.model_executor.layers.fla.ops.fused_recurrent import _select_sm70_bv
        BV = _select_sm70_bv(V, N, HV, v.device)
    else:
        BV = min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} "
            "when using `cu_seqlens`."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        assert initial_state is not None
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    assert initial_state is not None
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_indices_out=ssm_state_indices_out,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o.squeeze(0), final_state
