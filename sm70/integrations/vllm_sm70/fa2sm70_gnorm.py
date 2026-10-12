# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# GNORM: гейтованная норма GDN одним ядром вместо eager-лесенки из ~6 aten-пусков.
#
# ЗАЧЕМ. Опись стыков префилла (18.08, лента 32K): RMSNormGated (48 GDN-слоёв, вызов сразу
# после нашего сшитого GDN-тела) исполнялась NATIVE-путём — pow+mean+add+rsqrt+mul+silu
# ~650-700 мс на 32K-проход (~5.5% вычислений; на 250K больше — чанков больше). У vLLM есть
# быстрый путь forward_cuda (Triton-ядро fla), микрозамер на боевой форме [65536,128]:
# x2.84 к native и ТОЧНЕЕ (2.1e-4 против 4.7e-4 — внутренности fp32).
#
# ПОЧЕМУ НЕ compilation-config '+rms_norm_gated': включение кастом-опа глобально меняет
# диспетчер ВНУТРИ компилируемой области — dynamo падает на staticmethod.__call__
# (замерено отказом подъёма 18.08 22:24). Законный путь — НАШ же паттерн w8: свой
# зарегистрированный op (для dynamo непрозрачен) + подмена forward у класса.
#
# ВКЛЮЧЕНИЕ: FA2SM70_GNORM=1 (по умолчанию ВЫКЛ до сквозного гейта).
import os
import sys

import torch

_ВКЛ = os.environ.get("FA2SM70_GNORM", "0") == "1"
_С = [0]


def установить():
    if not _ВКЛ:
        return False
    try:
        from vllm.model_executor.layers.layernorm import RMSNormGated
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
        from vllm.utils.torch_utils import direct_register_custom_op
    except Exception as e:  # noqa: BLE001
        print(f"[fa2_sm70 GNORM] движок не отдал RMSNormGated/rmsnorm_fn ({str(e)[:60]}) — "
              f"подмены нет", file=sys.stderr, flush=True)
        return False

    # Покрываем ТОЛЬКО боевую конфигурацию GDN-нормы (group_size=None,
    # norm_before_gate=True, z есть): out = norm(x)*w*silu(z). Остальное — native, как было.
    def _gnorm(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        y = rmsnorm_fn(x, w, None, z=z, eps=eps, group_size=None, norm_before_gate=True)
        _С[0] += 1
        n = _С[0]
        if n == 1 or (n & (n - 1) == 0 and n >= 4096):
            print(f"[fa2_sm70 GNORM] слитая норма сработала, всего {n}", file=sys.stderr,
                  flush=True)
        return y

    def _gnorm_fake(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.empty_like(x)

    direct_register_custom_op("fa2sm70_gnorm", _gnorm, mutates_args=[], fake_impl=_gnorm_fake)

    _чужой = RMSNormGated.forward_native

    def forward_native(self, x, z=None):
        if (z is not None and self.group_size is None and self.norm_before_gate
                and x.dim() == 2 and x.dtype == torch.float16):
            return torch.ops.vllm.fa2sm70_gnorm(x, z, self.weight, float(self.eps))
        return _чужой(self, x, z)

    RMSNormGated.forward_native = forward_native
    # CustomOp при enabled=False зовёт forward_native — этого достаточно; forward_cuda
    # не трогаем (он включается только через compilation-config, который нам запрещён).
    print("[fa2_sm70 GNORM] гейтованная норма GDN переведена на слитое ядро "
          "(native-лесенка ~6 пусков -> 1; вне боевой конфигурации — как было)",
          file=sys.stderr, flush=True)
    return True
