# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ОБМЕН TP В ОДИН БАЙТ: вдвое меньше байт на мосту.
#
# ПОЧЕМУ СЮДА. Замерено снятием фазы: обмен TP = 58% префилла на паре PCIe, из них перекос рангов
# 2-3%, остальное -- ПЕРЕВОЗКА. Пол по перевозке 28.7 мс на свёртку [15680,5120] fp16 (5.6 ГБ/с),
# и он линеен по байтам -> один байт вместо двух даёт ~x2 на транспорте.
#
# ЧТО ИМЕННО КВАНТУЕТСЯ -- И ЧТО НЕТ. Свёртка складывает ДВЕ частичные суммы. Своя половина у нас
# уже в регистрах и через мост не едет -- значит квантовать её незачем:
#     out = x_свой (ТОЧНО, fp16)  +  deq(q(x_чужой))
# Это вдвое меньше шума, чем у наивного "проквантовать всё и сложить", и ровно тот же трафик.
# Масштаб -- НА ГРУППУ последней оси (не на строку): у строки 5120 элементов, и один выброс
# растягивает шаг квантования на всю строку. Группа 128 держит динамический диапазон узким.
#
# ПОЧЕМУ all_gather, А НЕ all_reduce. all_reduce обязан складывать НА ЛЕТУ, то есть в общем типе;
# int8 переполнится. all_gather просто везёт байты, а сложение делается локально уже в fp16 --
# и заодно это ОДИН шаг вместо двух (reduce-scatter + all-gather у кольца).
#
# ЦЕНА В ТОЧНОСТИ ИЗМЕРЯЕТСЯ, А НЕ ОБЪЯВЛЯЕТСЯ: самопроверка ниже печатает относительную ошибку на
# боевой форме при импорте, а сквозная -- приёмкой сервера.
#
# Включается FA2SM70_GATES='FA2SM70_ОБМЕН_INT8=1'. Порог по размеру -- чтобы мелкие служебные
# свёртки (и весь декод) шли прежним точным путём: там байты не связывают, а лишние ядра платятся.
import os
import sys

_ГРУППА = int(os.environ.get("FA2SM70_ОБМЕН_ГРУППА", "128"))
_ПОРОГ = int(os.environ.get("FA2SM70_ОБМЕН_ПОРОГ", str(1 << 20)))   # 1M элементов ~ 2 МБ fp16

if os.environ.get("FA2SM70_ОБМЕН_INT8", "0") == "1":
    import torch
    import torch.distributed as dist

    С = {"n": 0, "байт_ушло": 0, "байт_было_бы": 0}

    def _квант(xf, G):
        """xf: [M, N] -> (int8 [M, N], масштабы fp16 [M, N/G]). Группа по последней оси."""
        M, N = xf.shape
        g = xf.view(M, N // G, G)
        s = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / 127.0
        q = (g / s).round_().clamp_(-127, 127).to(torch.int8).view(M, N)
        return q, s.squeeze(2).to(torch.float16)

    def _деквант(q, s, G, dt):
        M, N = q.shape
        return (q.view(M, N // G, G).to(dt) * s.unsqueeze(2).to(dt)).view(M, N)

    def _обёртка(старый):
        def _f(self, input_, *a, **k):
            G = _ГРУППА
            n = input_.numel()
            if (n < _ПОРОГ or input_.dtype not in (torch.float16, torch.bfloat16)
                    or input_.shape[-1] % G != 0 or torch.cuda.is_current_stream_capturing()):
                return старый(self, input_, *a, **k)
            мир = dist.get_world_size(); ранг = dist.get_rank()
            xf = input_.reshape(-1, input_.shape[-1])
            q, s = _квант(xf, G)
            qs = [torch.empty_like(q) for _ in range(мир)]
            ss = [torch.empty_like(s) for _ in range(мир)]
            dist.all_gather(qs, q)
            dist.all_gather(ss, s)
            out = xf.clone()                     # СВОЯ часть -- точная, через мост не ехала
            for r in range(мир):
                if r == ранг:
                    continue
                out += _деквант(qs[r], ss[r], G, input_.dtype)
            С["n"] += 1
            С["байт_ушло"] += q.numel() + s.numel() * 2
            С["байт_было_бы"] += n * input_.element_size() * 2
            if С["n"] & (С["n"] - 1) == 0:
                print(f"[fa2_sm70 ОБМЕН-int8] свёрток {С['n']}: отправлено "
                      f"{С['байт_ушло']/1e9:.2f} ГБ вместо {С['байт_было_бы']/1e9:.2f} ГБ "
                      f"(x{С['байт_было_бы']/max(1,С['байт_ушло']):.2f} меньше)",
                      file=sys.stderr, flush=True)
            return out.view_as(input_)
        return _f

    # --- САМОПРОВЕРКА ТОЧНОСТИ НА БОЕВОЙ ФОРМЕ (без сети): печатается ОДИН раз при импорте.
    try:
        _a = torch.randn(1024, 5120, device="cuda", dtype=torch.float16)
        _b = torch.randn(1024, 5120, device="cuda", dtype=torch.float16)
        _q, _s = _квант(_b.float().view(1024, 5120), _ГРУППА)
        _наш = _a.float() + _деквант(_q, _s, _ГРУППА, torch.float32)
        _эт = _a.float() + _b.float()
        _о = (_наш - _эт).norm() / _эт.norm()
        print(f"[fa2_sm70] ОБМЕН В INT8 включён (группа {_ГРУППА}, порог {_ПОРОГ} элементов). "
              f"Самопроверка на гауссовом входе: relL2 суммы {_о:.2e}", file=sys.stderr, flush=True)
        del _a, _b, _q, _s, _наш, _эт
    except Exception as e:
        print(f"[fa2_sm70] ОБМЕН В INT8: самопроверка не прошла ({e})", file=sys.stderr, flush=True)

    сколько = 0
    for путь, имя in (
        ("vllm.distributed.device_communicators.base_device_communicator", "DeviceCommunicatorBase"),
        ("vllm.distributed.device_communicators.cuda_communicator", "CudaCommunicator"),
    ):
        try:
            мод = __import__(путь, fromlist=[имя])
            кл = getattr(мод, имя)
            if "all_reduce" in кл.__dict__:
                кл.all_reduce = _обёртка(кл.__dict__["all_reduce"])
                сколько += 1
        except Exception:
            pass
    if сколько == 0:
        print("[fa2_sm70] ОБМЕН-int8: НИ ОДИН класс не подменён -- ПРИБОР СЛЕП",
              file=sys.stderr, flush=True)
