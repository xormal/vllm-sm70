import torch
import os
"""ПЕРЕХОДНИК: веса боевой сети -> описатели слоёв для слитого декодного шага.

Здесь живёт единственное место, где можно перепутать тензор, и ошибка будет ТИХОЙ: формы у многих
проекций совпадают, поэтому перестановка не упадёт, а даст правдоподобный неверный ответ. Поэтому
порядок задан ЯВНО одним списком имён на тип слоя, и каждая форма проверяется против ожидаемой,
выведенной из конфигурации, а не из соседнего тензора.

Порядок обязан совпадать с `decode_chain` в fused_mlp_body.cu:
  GDN  (26): rw rw2 | qkv(3) z(3) | w_ab | out(3) | conv_w conv_state a_log dt_bias hstate | g(3) u(3) d(3)
  ATTN (19): rw rw2 | qkv(3)      | out(3) | kc vc | g(3) u(3) d(3)
"""

GDN_NAMES = [
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "linear_attn.in_proj_qkv.qweight", "linear_attn.in_proj_qkv.qzeros", "linear_attn.in_proj_qkv.scales",
    "linear_attn.in_proj_z.qweight",   "linear_attn.in_proj_z.qzeros",   "linear_attn.in_proj_z.scales",
    "<w_ab>",                                   # in_proj_a и in_proj_b, склеенные по столбцам
    "linear_attn.out_proj.qweight", "linear_attn.out_proj.qzeros", "linear_attn.out_proj.scales",
    "linear_attn.conv1d.weight", "<conv_state>", "linear_attn.A_log", "linear_attn.dt_bias", "<hstate>",
    "mlp.gate_proj.qweight", "mlp.gate_proj.qzeros", "mlp.gate_proj.scales",
    "mlp.up_proj.qweight",   "mlp.up_proj.qzeros",   "mlp.up_proj.scales",
    "mlp.down_proj.qweight", "mlp.down_proj.qzeros", "mlp.down_proj.scales",
    # ВЕС RMSNormGated ВЫХОДА МЕШАЛКИ (позиция 26). Добавлен В КОНЕЦ намеренно: все прежние
    # позиции сохраняются, ядро читает новую отдельно. Без него цепь делала o*silu(z) вместо
    # norm(o)*w*silu(z) -- операции не было вовсе, и обе сверки её отсутствие не ловили,
    # потому что читали один и тот же наш путь.
    "linear_attn.norm.weight",
]

ATTN_NAMES = [
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.qkv_proj.qweight", "self_attn.qkv_proj.qzeros", "self_attn.qkv_proj.scales",
    "self_attn.o_proj.qweight", "self_attn.o_proj.qzeros", "self_attn.o_proj.scales",
    "<kcache>", "<vcache>",
    "mlp.gate_proj.qweight", "mlp.gate_proj.qzeros", "mlp.gate_proj.scales",
    "mlp.up_proj.qweight",   "mlp.up_proj.qzeros",   "mlp.up_proj.scales",
    "mlp.down_proj.qweight", "mlp.down_proj.qzeros", "mlp.down_proj.scales",
]


def expected_shapes(cfg, tp: int):
    """Ожидаемые формы НА ВОРКЕР, выведенные из конфигурации, а не из соседнего тензора."""
    K = cfg["hidden_size"]
    I = cfg["intermediate_size"] // tp
    dk = cfg["linear_key_head_dim"]; dv = cfg["linear_value_head_dim"]
    hk = cfg["linear_num_key_heads"] // tp; hv = cfg["linear_num_value_heads"] // tp
    H = cfg["num_attention_heads"] // tp; Hkv = cfg["num_key_value_heads"] // tp
    d = cfg["head_dim"]
    return {
        "K": K, "I": I,
        "Nq": 2 * hk * dk + hv * dv, "Nz": hv * dv, "Nab": 2 * hv,
        "KoutG": hv * dv, "Nqkv": H * d + 2 * Hkv * d, "KoutA": H * d,
        "Hk": hk, "Hv": hv, "H": H, "Hkv": Hkv, "DK": dk, "DV": dv, "d": d,
    }


def check_layout(cfg, tp: int = 2):
    """Сверка раскладки БЕЗ карты и БЕЗ сервера: кратности плиток и согласованность осей."""
    e = expected_shapes(cfg, tp)
    T, CPT = 128, 4
    tile = T * CPT
    bad = []
    if (e["Nq"] + e["Nz"]) % tile:
        bad.append(f"Nq+Nz={e['Nq']+e['Nz']} не кратно плитке {tile}")
    if e["K"] % tile:
        bad.append(f"K={e['K']} не кратно плитке {tile}")
    if e["Nab"] > 64:
        bad.append(f"Nab={e['Nab']} шире 64 столбцов")
    if e["KoutG"] != e["Hv"] * e["DV"]:
        bad.append("вход out_proj не равен выходу мешалки")
    if e["KoutA"] != e["H"] * e["d"]:
        bad.append("вход o_proj не равен выходу внимания")
    if e["d"] != 2 * T:
        bad.append(f"d={e['d']} не равно 2*{T} (два элемента на поток)")
    if e["DV"] != T:
        bad.append(f"DV={e['DV']} не равно числу потоков {T}")
    if len(GDN_NAMES) != 27:
        bad.append(f"у GDN {len(GDN_NAMES)} имён вместо 27")
    if len(ATTN_NAMES) != 19:
        bad.append(f"у внимания {len(ATTN_NAMES)} имён вместо 19")
    return e, bad


def _get(mod, path):
    """Достать по пути атрибутов. Отсутствие -> ОТКАЗ, а не None: молчаливый None ушёл бы в ядро
    нулевым указателем и дал бы неверный ответ без падения."""
    cur = mod
    for p in path.split("."):
        if not hasattr(cur, p):
            raise KeyError(f"нет пути «{path}» (сломалось на «{p}»)")
        cur = getattr(cur, p)
    return cur


def probe_model(model, tp: int = 2):
    """РАЗВЕДКА ПО ЖИВОЙ МОДЕЛИ: какие слои есть, какого типа, и находятся ли ожидаемые тензоры.
    Ничего не подменяет -- только смотрит и отчитывается. Имена в загруженном объекте могут не
    совпасть с именами в слепке, и узнать это надо ДО подмены пути, а не после.
    """
    root = None
    if hasattr(model, "layers"):            # держатель слоёв ПЕРЕДАН НАПРЯМУЮ -- не ищем
        root = model
        root = model.layers
    for path in ("model.language_model.layers", "language_model.layers", "model.layers", "layers"):
        if root is not None:
            break
        try:
            root = _get(model, path); break
        except KeyError:
            continue
    if root is None:
        return {"ошибка": "не найден список слоёв ни по одному из известных путей"}
    rep = {"слоёв": len(root), "gdn": 0, "attn": 0, "пропало": [], "формы": {}}
    # ЧИТАТЬ РЕАЛЬНОСТЬ, А НЕ СВЕРЯТЬ СО СВОИМ СПИСКОМ. Мой перечень имён -- гипотеза; фактические
    # имена в загруженном объекте есть ФАКТ. Расхождение "пропало"/"формы" -- признак, что слои
    # ОДНОГО типа различаются между собой, и увидеть это можно только перечислив их поимённо.
    seen = {}
    for i, lay in enumerate(root):
        k = "gdn" if hasattr(lay, "linear_attn") else "attn"
        nm = tuple(sorted(n for n, _ in lay.named_parameters()))
        seen.setdefault((k, nm), []).append(i)
    # ТОЧНЫЕ БАЙТЫ, А НЕ ОЦЕНКА ПО "int8 везде": потолок 15.663 ГБ считался в предпосылке сплошного
    # квантования, а плотных тензоров тут три класса. Считаем numel*itemsize по факту.
    tot = {}
    for i, lay in enumerate(root):
        k = "gdn" if hasattr(lay, "linear_attn") else "attn"
        for n, t in lay.named_parameters():
            kind = "плотн" if n.endswith(".weight") and t.dtype.itemsize == 2 and t.dim() > 1 else (
                "int8" if "qweight" in n else "мелочь")
            tot[kind] = tot.get(kind, 0) + t.numel() * t.element_size()
        for n, b in lay.named_buffers():
            tot["мелочь"] = tot.get("мелочь", 0) + b.numel() * b.element_size()
    # НАЗВАТЬ ПОИМЁННО: 2.26 ГБ, которые я не могу назвать, -- это не остаток округления,
    # а 14.7 % трафика весов. Классификатор мой, значит и ошибка может быть моя.
    byname = {}
    for i, lay in enumerate(root):
        for n, t in list(lay.named_parameters()) + list(lay.named_buffers()):
            byname[n] = byname.get(n, 0) + t.numel() * t.element_size()
    # ИДЕНТИЧНОСТЬ ПО ВЛАДЕНИЮ, А НЕ ПО ИМЕНИ: одно и то же хранилище, видное из 16 слоёв, в
    # поимённой описи выглядит как 16 разных тензоров. Различает только data_ptr.
    uniq, seen_ptr = 0, set()
    rope = {}
    for i, lay in enumerate(root):
        for n, t in list(lay.named_parameters()) + list(lay.named_buffers()):
            ptr = t.data_ptr()
            if ptr not in seen_ptr:
                seen_ptr.add(ptr)
                uniq += t.numel() * t.element_size()
            if "cos_sin_cache" in n:
                rope.setdefault(ptr, []).append(i)
    rep["УНИКАЛЬНЫХ_ГБ"] = round(uniq / 1e9, 4)
    rep["РОПЕ"] = {"разных хранилищ": len(rope), "слоёв на хранилище": [len(v) for v in rope.values()],
                   "ГБ за штуку": round(list(rope.keys()) and 0 or 0, 4)}
    # ВЫГРУЗКА НАСТОЯЩИХ ВЕСОВ: на случайных весах я уже получал inf и ложные вердикты. Один слой
    # каждой разновидности на диск -- дальше сверка идёт против torch на БОЕВЫХ числах, без подъёма
    # сервера на каждую итерацию.
    dump = os.environ.get("FA2SM70_MEGA_DUMP")
    if dump:
        import torch
        got = {}
        for i, lay in enumerate(root):
            k = "gdn" if hasattr(lay, "linear_attn") else "attn"
            tag = f"{k}{i}"
            if k == "gdn" and i not in (0, 1):
                continue
            if k == "attn" and i != 3:
                continue
            got[tag] = {n: t.detach().to("cpu")
                        for n, t in list(lay.named_parameters()) + list(lay.named_buffers())
                        if "cos_sin_cache" not in n}
        torch.save(got, dump)
        rep["ВЫГРУЖЕНО"] = {k: len(v) for k, v in got.items()}
    rep["ПОИМЁННО"] = {k: round(v / 1e9, 4)
                       for k, v in sorted(byname.items(), key=lambda x: -x[1])[:12]}
    rep["БАЙТ"] = {k: round(v / 1e9, 4) for k, v in tot.items()}
    rep["БАЙТ"]["итого"] = round(sum(tot.values()) / 1e9, 4)
    rep["РАЗНОВИДНОСТИ"] = [
        {"тип": k, "слоёв": len(idx), "первый": idx[0], "имена": len(nm)}
        for (k, nm), idx in seen.items()
    ]
    for i, lay in enumerate(root):
        is_gdn = hasattr(lay, "linear_attn")
        rep["gdn" if is_gdn else "attn"] += 1
        names = GDN_NAMES if is_gdn else ATTN_NAMES
        for n in names:
            if n.startswith("<"):
                continue
            try:
                t = _get(lay, n)
                if i in (0, 1, 3) and n not in rep["формы"]:
                    rep["формы"][n] = tuple(t.shape)
            except KeyError as e:
                key = f"{'gdn' if is_gdn else 'attn'}:{n}"
                if key not in rep["пропало"]:
                    rep["пропало"].append(key)
    return rep


_MODEL = None


def register_model(m):
    """Точка СОЗДАНИЯ объявляет владельца. Гейт внутри: без него ничего не запоминаем.

    ЗДЕСЬ ЖЕ регистрируем операцию: трассировка идёт РАНЬШЕ первого декода, и если операции нет,
    вызов внутри компилируемого тела поднимает исключение (проверено отказом: в журнале нет ни
    подготовки, ни регистрации -- только отказ трассировки).
    """
    global _MODEL
    if os.environ.get("FA2SM70_MEGA_CHECK") == "1" or os.environ.get("FA2SM70_MEGA") == "1":
        _MODEL = m
        if os.environ.get("FA2SM70_MEGA") == "1":
            зарегистрировать()


def registered_model():
    return _MODEL


def install_probe(logger=None):
    """ЗАЦЕП ТАМ, ГДЕ ВЛАДЕЛЕЦ СЕБЯ ЗНАЕТ.

    Пять попыток выудить модель снаружи (по имени класса, по структуре, по полю, через сборщик
    мусора) дали ноль: из тела внимания держатель слоёв не виден. Правильный ход -- не искать
    владельца, а встать там, где `self` И ЕСТЬ владелец. Обёртка ничего не подменяет в вычислении:
    один раз докладывает и вызывает исходный проход.
    """
    from vllm.model_executor.models.qwen3_next import Qwen3NextModel
    if getattr(Qwen3NextModel, "_fa2sm70_probe", False):
        return "уже стоит"
    orig = Qwen3NextModel.forward

    def wrapped(self, *a, **kw):
        if not getattr(Qwen3NextModel, "_fa2sm70_done", False):
            Qwen3NextModel._fa2sm70_done = True
            try:
                rep = probe_model(self)
                if logger:
                    logger.info("[fa2_sm70 мегашаг] РАЗВЕДКА: %s", rep)
            except Exception as e:
                if logger:
                    logger.info("[fa2_sm70 мегашаг] разведка упала: %s: %s", type(e).__name__, e)
        return orig(self, *a, **kw)

    Qwen3NextModel.forward = wrapped
    Qwen3NextModel._fa2sm70_probe = True
    return "поставлен"


# ============================================================================================
# ФАКТИЧЕСКАЯ ГЕОМЕТРИЯ БОЕВОЙ СЕТИ (снята с ЖИВОЙ модели 09.08.2026, не из моих предположений)
# ============================================================================================
# Прежняя запись "Nqkv=4096, Hkv=2, d=2x128" НЕВЕРНА в части внимания. Проверка check_layout()
# "проходила без расхождений" потому, что сверяла модель с моей же гипотезой, а не с фактом.
FACT = {
    "K": 5120, "I": 17408, "слоёв": 64, "полное внимание каждые": 4, "TP": 2,
    # --- GDN (48 слоёв), на воркер: подтверждено тензорами
    "gdn": {
        "in_proj_qkvz.N": 8192,      # q 1024 + k 1024 + v 3072 + z 3072
        "Hk": 8, "Hv": 24, "DK": 128, "DV": 128,
        "in_proj_ba.N": 48,          # ПЛОТНЫЙ fp16 во всех слоях
        "out_proj.K": 3072,
    },
    # --- полное внимание (16 слоёв), на воркер
    "attn": {
        "qkv.N": 7168,               # q 3072 + ГЕЙТ 3072 + k 512 + v 512
        "Hq": 12, "Hkv": 2, "d": 256,
        "qkv ПЛОТНЫЙ": True,         # квантован только o_proj
        "o_proj.K": 3072,
        # ПОРЯДОК: qkv.split([2*q_size, kv, kv]); q_gate.view(Hq, 2*d); chunk(2, -1)
        # => q и ГЕЙТ ЧЕРЕДУЮТСЯ ПО ГОЛОВЕ: [q(256) гейт(256)] x 12, затем k, затем v.
        "q_gate чередуется по голове": True,
        "q_norm/k_norm": 256,        # RMSNorm по head_dim, ДО RoPE
        "выход": "attn * sigmoid(гейт)",
        # ВРАЩЕНИЕ ЧАСТИЧНОЕ: partial_rotary_factor=0.25 -> вращаются 64 измерения из 256.
        # Таблица cos_sin_cache -- fp32 [262144, 64+64] = 134.2 МБ, ОДНА на все 16 слоёв
        # (см. опись по владению). За токен из неё читаются единицы позиций, а не вся плита.
        "rotary_dim": 64, "head_dim": 256, "таблица": "fp32 [262144, 64cos+64sin]",
        "eps": 1e-6,
    },
    # --- три разновидности слоёв, а не одна
    "разновидности": {
        "слой 0": "НЕ КВАНТОВАН вовсе (все .weight)",
        "слои 1..63 GDN": "int8 GPTQ, кроме in_proj_ba (плотный)",
        "слои внимания": "int8 GPTQ, кроме qkv_proj (плотный)",
    },
    "g_idx": "пустой (0,) -> desc_act=false подтверждён",
    "резидентно на воркер, ГБ": 13.37,   # по ВЛАДЕНИЮ; по именам было 15.383 (таблица RoPE x16)
}


# ==================================================================================================
# ИЗВЛЕЧЕНИЕ ВЕСОВ ДЛЯ ЦЕПИ. Порядок тензоров ПОЗИЦИОННЫЙ -- он же проверен сверкой цепи против
# одиночного слоя, поэтому менять его нельзя без повторной сверки. Статика готовится ОДИН раз;
# состояния и склад подаются на каждом шаге (они меняются, веса нет).
# ==================================================================================================
def головы_gdn(qkvz0, out0, DK: int = 128, DV: int = 128):
    """Hk и Hv НА ВОРКЕР -- из форм весов, а не из полей слоя (те хранят ОБЩЕЕ число, без деления TP).

    Из одной ширины плиты их не вывести: два неизвестных, одно уравнение. Второй источник --
    `out_proj`, у которого вход равен Hv*DV.
        Nq + Nz = ширина плиты qkvz,   Nz = Hv*DV,   Nq = Hk*DK*2 + Hv*DV
    """
    W = int(qkvz0.qweight.shape[1])          # Nq + Nz
    Kout = int(out0.qweight.shape[0]) * 4    # вход out_proj = Hv*DV
    Hv = Kout // DV
    Nz = Hv * DV
    Nq = W - Nz
    Hk = (Nq - Hv * DV) // (2 * DK)
    if Hk <= 0 or Hv <= 0 or Nq != Hk * DK * 2 + Hv * DV:
        raise RuntimeError(f"головы_gdn: формы не сходятся (W={W}, Kout={Kout}, Hk={Hk}, Hv={Hv})")
    return Hk, Hv


def сверить_tp(слой, Hk: int, Hv: int):
    """НЕЗАВИСИМЫЙ контроль. Уравнения форм самосогласованы и для ДРУГИХ троек (проверено: 11/21
    проходит), поэтому одной их сходимости мало. Поля слоя хранят ОБЩЕЕ число голов -- значит
    отношение общего к выведенному обязано быть ОДНИМ И ТЕМ ЖЕ целым, и это степень TP.
    """
    ok, ov = getattr(слой, "num_k_heads", None), getattr(слой, "num_v_heads", None)
    if ok is None or ov is None:
        return None                       # полей нет -- контроль недоступен, молча не выдумываем
    if int(ok) % Hk or int(ov) % Hv or int(ok) // Hk != int(ov) // Hv:
        raise RuntimeError(f"сверить_tp: общее {ok}/{ov} не кратно выведенному {Hk}/{Hv} одним TP")
    return int(ok) // Hk


def prepare_static(model, tp: int = 2):
    import torch
    root = model.layers if hasattr(model, "layers") else _get(model, "model.layers")
    слои, gdn_flag, непокрытые = [], [], []
    корни = []          # НОМЕР В КОРНЕ для каждого покрытого слоя -- не выводить арифметикой
    W = None
    for _номер_в_корне, lay in enumerate(root):
        gdn = hasattr(lay, "linear_attn")
        rw  = lay.input_layernorm.weight
        rw2 = lay.post_attention_layernorm.weight
        mlp = lay.mlp
        gu, dn = mlp.gate_up_proj, mlp.down_proj
        квант = hasattr(gu, "qweight")
        if not квант:
            # СЛОЙ 0 НЕ КВАНТОВАН ВОВСЕ (разведка). Не подставляем чужие тензоры и не молчим:
            # слой возвращается в списке НЕПОКРЫТЫХ, чтобы пробел был виден вызывающему.
            непокрытые.append(len(слои) + len(непокрытые))
            continue
        mlp9 = [gu.qweight, gu.qzeros, gu.scales,      # gate и up -- ОДНА плита, разводится смещением
                gu.qweight, gu.qzeros, gu.scales,
                dn.qweight, dn.qzeros, dn.scales]
        if gdn:
            la = lay.linear_attn
            qkvz = la.in_proj_qkvz
            ab = la.in_proj_ba.weight.t().contiguous()          # ядро ждёт [K, Nab], модель даёт [Nab, K]
            слои.append([rw, rw2,
                         qkvz.qweight, qkvz.qzeros, qkvz.scales,
                         qkvz.qweight, qkvz.qzeros, qkvz.scales,   # та же плита: q и z по смещению
                         ab,
                         la.out_proj.qweight, la.out_proj.qzeros, la.out_proj.scales,
                         la.conv1d.weight.view(la.conv1d.weight.shape[0], -1).contiguous(),
                         None, la.A_log.float(), la.dt_bias, None] + mlp9 + [
                         # ВЕС RMSNormGated ВЫХОДА МЕШАЛКИ (позиция 26, в конце). Ядро делало
                         # o*silu(z) вместо norm(o)*w*silu(z): операции не было вовсе.
                         la.norm.weight])
            gdn_flag.append(1)
        else:
            sa = lay.self_attn
            W = sa.qkv_proj.weight                                # ПЛОТНЫЙ у боевой сети
            o = sa.o_proj
            # ЗАГЛУШКА -- ПУСТОЙ ТЕНЗОР, а не None: None не переводится в тензор и роняет вызов,
            # а места kc/vc не заполняются шагом (у боевой сети склад постраничный).
            пусто = torch.empty(0, dtype=rw.dtype, device=rw.device)
            слои.append([rw, rw2,
                         o.qweight, o.qzeros, o.scales,            # заглушка на месте квантованного qkv
                         o.qweight, o.qzeros, o.scales,
                         пусто, пусто] + mlp9 +
                        [W, sa.q_norm.weight, sa.k_norm.weight, None,
                         None, None, None])
            gdn_flag.append(0)
        корни.append(_номер_в_корне)
    gu0 = root[1].mlp.gate_up_proj
    qkvz0 = root[1].linear_attn.in_proj_qkvz
    import math
    cfg = root[1].linear_attn
    out0 = root[1].linear_attn.out_proj
    _hk, _hv = головы_gdn(qkvz0, out0)
    _tp = сверить_tp(root[1].linear_attn, _hk, _hv)
    мера = {"qkvz_width": int(qkvz0.qweight.shape[1]), "Hk": _hk, "Hv": _hv,
            # ЧИСЛО ГОЛОВ -- ИЗ ФОРМЫ ВЕСА, а не из полей слоя: поля хранят ОБЩЕЕ число, а на
            # воркер приходится доля TP. Сверка с факт-листом поймала ровно это (16/48 против 8/24).
            # Плита qkvz: Nz = Hv*DV (вторая половина), Nq = Hk*DK*2 + Hv*DV (первая).
            "scale_gdn": 1.0 / math.sqrt(128.0),
            "mlp_width":  int(gu0.qweight.shape[1]),
            "up_offset":  int(gu0.qweight.shape[1] // 2),
            "Nqkv":       int(W.shape[0]) if W is not None else 0}
    сам = next(root[i] for i in range(len(root)) if not hasattr(root[i], "linear_attn"))
    d_ = int(сам.self_attn.head_dim)
    мера.update({"Hq": int(сам.self_attn.num_heads), "Hkv": int(сам.self_attn.num_kv_heads),
                 "d": d_, "scale_attn": 1.0 / math.sqrt(float(d_)),
                 "rot": int(сам.self_attn.rotary_emb.rotary_dim),
                 "kscale": 1.0, "vscale": 1.0})
    мера["TP"] = _tp
    мера["непокрытые слои"] = непокрытые
    # СООТВЕТСТВИЕ «СЛОЙ ЦЕПИ -> СЛОЙ КОРНЯ» ЯВНЫМ СПИСКОМ, а не арифметикой `L+1`.
    # Арифметика верна ровно при одном непокрытом слое (нулевом). Второй непокрытый где-то
    # в середине сдвинул бы ВЕСА И СОСТОЯНИЯ всех последующих слоёв и не уронил бы ничего:
    # ответ остался бы «слоеобразным» и неверным -- ровно тот класс, что уже стоил суток.
    мера["корни"] = корни
    if len(корни) != len(слои):
        raise RuntimeError(f"prepare_static: корней {len(корни)}, слоёв {len(слои)}")
    return слои, gdn_flag, мера


# Позиции динамических мест в позиционном списке -- ОДНО определение на весь код.
# Дублировать эти числа нельзя: две копии разъезжаются молча (сегодняшний урок про две реализации).
МЕСТА = {
    "gdn":  {"conv_state": 13, "hstate": 16},
    "attn": {"cos_sin": 22, "kpool": 23, "vpool": 24, "block_table": 25},
}


def fill_step(слои, флаги, *, cos_sin_row=None, conv_states=None, hstates=None,
              kpools=None, vpools=None, block_tables=None):
    """Заполнить динамические места. Веса статичны, состояния и склад меняются каждый шаг.

    Ничего не угадывает: если место требуется, а значение не дано, поднимает ошибку с ИМЕНЕМ места,
    вместо того чтобы оставить None и получить обращение по нулевому указателю в ядре.
    """
    out = []
    gi = ai = 0
    for L, (v, gdn) in enumerate(zip(слои, флаги)):
        v = list(v)
        if gdn:
            for имя, поз in МЕСТА["gdn"].items():
                знач = (conv_states or {}).get(gi) if имя == "conv_state" else (hstates or {}).get(gi)
                if знач is None:
                    raise ValueError(f"слой {L} (GDN #{gi}): не дано '{имя}'")
                v[поз] = знач
            gi += 1
        else:
            # СКЛАД KV -- СВОЙ У КАЖДОГО СЛОЯ. Раньше сюда клался склад ПЕРВОГО слоя внимания
            # для всех шестнадцати: каждый слой смотрел в ЧУЖИЕ ключи. Ответ при этом остаётся
            # связным на вид -- ровно тот класс тихой неверности, который не падает.
            for имя, поз in МЕСТА["attn"].items():
                знач = {"cos_sin": cos_sin_row,
                        "kpool": (kpools or {}).get(ai),
                        "vpool": (vpools or {}).get(ai),
                        "block_table": (block_tables or {}).get(ai)}[имя]
                if знач is None:
                    raise ValueError(f"слой {L} (внимание #{ai}): не дано '{имя}'")
                v[поз] = знач
            ai += 1
        out.append(v)
    return out


def пул_kv(кэш):
    """K и V из склада слоя ТАК ЖЕ, как это делает бэкенд форка.

    У них: `if kv_cache.shape[0] == 2: unbind(0) else: unbind(1)`. Раскладок ДВЕ --
    `[2, блоки, токены, головы, d]` и `[блоки, 2, токены, головы, d]`. Мы брали `кэш[0]`
    всегда: во второй раскладке это ОДИН БЛОК, а шаг блока (stride(0)) оказывается шагом
    между K и V -- чтение уходит в чужие блоки МОЛЧА, тем сильнее, чем длиннее контекст.
    """
    ось = [i for i in (0, 1) if int(кэш.shape[i]) == 2]
    if len(ось) != 1:
        raise RuntimeError(f"пул_kv: не опознать ось K/V у склада формы {tuple(кэш.shape)} -- "
                           "двойка стоит и на 0, и на 1 (или нигде); гадать здесь нельзя")
    return кэш.unbind(ось[0])


def rope_row(cos_sin_cache, position: int):
    """Строка cos/sin для позиции токена. Таблица ОДНА на все слои (см. опись по владению)."""
    r = cos_sin_cache[position]
    return r.float().contiguous() if r.dtype != __import__("torch").float32 else r.contiguous()


# ==================================================================================================
# ОПЕРАЦИЯ ДЛЯ ЦИКЛА СЛОЁВ. Через пользовательскую операцию проходят ТОЛЬКО тензоры и числа,
# поэтому подготовленное берётся из реестра ПО ИМЕНИ -- так же, как у них в `gdn_attention_core`.
# Компилятор считает такую операцию непрозрачной и внутрь не заглядывает.
# ==================================================================================================
_ГОТОВО = {}
_БУФЕРЫ = {}   # постоянные плотные буферы состояний GDN: выделять на КАЖДОМ шаге дорого
СЧЁТ = {}     # СЧЁТЧИК МАРШРУТА: пишет ВОРКЕР ПО ФАКТУ, а не наши намерения


def подготовить(model, имя: str = "по умолчанию"):
    """Один раз на подъём: статические веса + мера. Состояния и склад -- на каждом шаге."""
    слои, флаги, мера = prepare_static(model)
    # ЦЕПЬ ПОКРЫВАЕТ ВСЁ, КРОМЕ СЛОЯ 0. Непокрытый слой ГДЕ-ТО В СЕРЕДИНЕ цепь не считает
    # вовсе, а вызывающий прогоняет слой 0 отдельно и полагает остальное покрытым: ответ
    # вышел бы слоеобразным и неверным, без единого признака. Проверяем ЯВНО.
    непокр = list(мера.get("непокрытые слои", []))
    if непокр != [0]:
        raise RuntimeError(f"подготовить: непокрыты слои {непокр}, а цепь умеет ровно [0] -- "
                           "покрытие надо расширить, а не молча пропустить")
    _ГОТОВО[имя] = {"слои": слои, "флаги": флаги, "мера": мера, "модель": model}
    return мера


def шаг_цепи(x, позиция: int, имя: str = "по умолчанию", слоёв: int = 0,
             поз_тензор=None):
    """Один шаг декода цепью. Возвращает выход ПОСЛЕ последнего слоя цепи (слой 0 идёт мимо).

    Ничего не угадывает: недостающее -> ошибка с именем, а не подстановка нуля.
    """
    import torch
    from fa2_sm70 import _ext as _e
    # РАСКЛАДКА ОБВЯЗКИ. Ядро замерено (события), а 21-25 мс шага лежат ВНЕ его -- без разбиения
    # на сборку описи / пуск / возврат состояний следующий ход выбирается догадкой.
    _проф0 = os.environ.get("FA2SM70_CHAIN_PROF") == "1"
    _tвх = __import__("time").perf_counter() if _проф0 else 0.0
    _ext = _e.mega_ext()
    г = _ГОТОВО.get(имя)
    if г is None:
        raise RuntimeError("шаг_цепи: цепь не подготовлена (позовите подготовить)")
    слои, флаги, мера = г["слои"], г["флаги"], г["мера"]
    if слоёв:
        # СРЕЗ ПО ГЛУБИНЕ. Глубина выбирается так, чтобы слой внимания ПРИСУТСТВОВАЛ -- иначе
        # срабатывает собственная проверка склада, и инструмент ломает контракт (уже проверено).
        # СКЛАД БЕРЁТСЯ ИЗ ПОЛНОГО СПИСКА (ниже), поэтому срез может быть любой глубины --
        # инструмент больше не требует, чтобы в срез попал слой внимания.
        слои, флаги = слои[:слоёв], флаги[:слоёв]
    if слоёв:
        слои, флаги = слои[:слоёв], флаги[:слоёв]   # РАЗРЕЗ ПО ГЛУБИНЕ для отладки
    корень = г["модель"].layers
    # СОСТОЯНИЯ -- ИЗ САМИХ СЛОЁВ (`layer.kv_cache`), склад -- оттуда же у слоёв внимания.
    from vllm.forward_context import get_forward_context
    фк = get_forward_context()
    мета = фк.attn_metadata
    if not isinstance(мета, dict):
        raise RuntimeError("шаг_цепи: метаданные не по именам слоёв -- путь изменился, не угадываем")
    conv, hs, слоты = {}, {}, []
    kps, vps, bts = {}, {}, {}          # склад ПО НОМЕРУ слоя внимания в цепи
    масштабы = []                       # масштабы БАЙТОВОГО пула, по слоям
    gi = ai = 0
    корни = мера["корни"]
    # НОМЕРА СЛОТОВ -- ОДНИМ ЧТЕНИЕМ НА ШАГ, А НЕ ПО СЛОЮ. `int(слот[0])` читает CUDA-тензор,
    # то есть СИНХРОНИЗИРУЕТ поток; сорок семь таких синхронизаций стоили 5.35 мс из 5.7 мс
    # описи -- больше, чем все девяносто четыре копии состояний вместе взятые.
    # СЛОТ СОСТОЯНИЯ -- ТЕНЗОРОМ НА КАРТУ, А НЕ ЧИСЛОМ НА ХОСТ. Число пришлось бы читать
    # (синхронизация), а главное -- оно ЗАМОРОЗИЛОСЬ БЫ при захвате шага в CUDA-граф: граф,
    # снятый на одном запросе, продолжил бы писать в его слот. Отдаём указатель, ядро само
    # адресуется; заодно исчезают 94 копии состояний на шаг (ядро пишет прямо в слот движка).
    слот_тензоры = []
    for L, (v, gdn) in enumerate(zip(слои, флаги)):
        if gdn:
            слот = мета[корень[корни[L]].linear_attn.prefix].non_spec_state_indices_tensor
            if слот is None or слот.numel() != 1:
                raise RuntimeError("шаг_цепи: цепь собрана под ОДИН запрос, слотов %s"
                                   % (None if слот is None else слот.numel()))
            слот_тензоры.append(слот.reshape(1))
        else:
            слот_тензоры.append(None)
    # ГОДНОСТЬ ПРЯМОГО ПУТИ ДОКАЗЫВАЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ: не тот тип индекса или не та
    # размерность склада состояний -- откатываемся на прежний путь с копиями, а не гадаем.
    _пример = next((с for с in слот_тензоры if с is not None), None)
    прямые = (_пример is not None and _пример.dtype is torch.int32
              and поз_тензор is not None and поз_тензор.dtype is torch.int64)
    if прямые:
        for L, (v, gdn) in enumerate(zip(слои, флаги)):
            if gdn and корень[корни[L]].linear_attn.kv_cache[фк.virtual_engine][0].dim() != 3:
                прямые = False; break
    if СЧЁТ.get("путь_слотов") is None:
        СЧЁТ["путь_слотов"] = 1
        import logging
        logging.getLogger("vllm").info(
            "[fa2_sm70 мегашаг] СЛОТЫ: %s (индекс=%s, позиция=%s)",
            "ПРЯМЫЕ (ядро адресуется само)" if прямые else "через копии",
            None if _пример is None else _пример.dtype,
            None if поз_тензор is None else поз_тензор.dtype)
    for L, (v, gdn) in enumerate(zip(слои, флаги)):
        сл = корень[корни[L]]                    # СООТВЕТСТВИЕ СПИСКОМ, а не `L+1`
        if gdn:
            кэш = сл.linear_attn.kv_cache[фк.virtual_engine]
            if прямые:
                conv[gi], hs[gi] = кэш[0], кэш[1]          # ЦЕЛИКОМ: слот выберет ядро
            else:
                i = int(слот_тензоры[L][0])
                слот_h = кэш[1][i]; пр = слот_h.is_contiguous()
                б = _БУФЕРЫ.get(gi)
                нужен = (б is None) or (б[0].shape != кэш[0][i].transpose(-1, -2).shape) \
                        or (not пр and б[1].shape != слот_h.shape)
                if нужен:
                    б = (кэш[0][i].transpose(-1, -2).contiguous(),
                         слот_h if пр else слот_h.contiguous())
                    _БУФЕРЫ[gi] = б
                else:
                    б[0].copy_(кэш[0][i].transpose(-1, -2))
                    if пр:
                        б = (б[0], слот_h); _БУФЕРЫ[gi] = б
                    else:
                        б[1].copy_(слот_h)
                conv[gi], hs[gi] = б
                слоты.append((кэш[0][i], None if пр else слот_h, gi))
            gi += 1
        else:
            кэш = пул_kv(сл.self_attn.attn.kv_cache[фк.virtual_engine])
            м2 = мета[сл.self_attn.attn.layer_name]
            if not hasattr(м2, "block_table"):
                raise RuntimeError("шаг_цепи: в метаданных внимания нет таблицы блоков")
            kps[ai], vps[ai], bts[ai] = кэш[0], кэш[1], м2.block_table
            вн_сл = сл.self_attn.attn
            масштабы.append((float(getattr(вн_сл, "_k_scale_float", 1.0) or 1.0),
                             float(getattr(вн_сл, "_v_scale_float", 1.0) or 1.0)))
            ai += 1
    _tсост = __import__("time").perf_counter() if _проф0 else 0.0
    if not kps and any(not g for g in г["флаги"]):
        # Срез по глубине мог не захватить ни одного слоя внимания -- тогда склад не нужен вовсе.
        pass
    # СТРОКА ВРАЩЕНИЯ: таблица ОДНА на все слои внимания (опись по владению), берём с первого.
    # ТАБЛИЦА ВРАЩЕНИЯ -- ИЗ ПОЛНОГО СПИСКА, а не из среза: при глубине 1-2 слоя внимания в
    # срезе нет вовсе, и поиск падал StopIteration. Таблица одна на все слои (опись по владению).
    первый_вним = next(корень[мера["корни"][L]] for L, g in enumerate(г["флаги"]) if not g)
    таб = первый_вним.self_attn.rotary_emb.cos_sin_cache
    # ПРИ ПРЯМОМ ПУТИ отдаём ВСЮ таблицу: строку выберет ядро по позиции с карты. Это последняя
    # величина, которая иначе замёрзла бы при захвате шага в граф.
    строка = (таб if (таб.dtype is torch.float32 and таб.dim() == 2)
              else rope_row(таб, int(позиция))) if прямые else rope_row(таб, int(позиция))
    if СЧЁТ.get("типы") is None:
        СЧЁТ["типы"] = 1
        import logging
        л = logging.getLogger("vllm")
        л.info("[fa2_sm70 мегашаг] ТИПЫ: x=%s conv=%s h=%s cos=%s складов=%d kp0=%s bt0=%s",
               x.dtype, conv[0].dtype, hs[0].dtype, строка.dtype, len(kps),
               kps[0].dtype if kps else "-", bts[0].dtype if bts else "-")
        # ВСЕ 26 позиций слоя: искать виновника перебором дороже, чем напечатать разом.
        для_печати = [(i, (t.dtype if hasattr(t, "dtype") else type(t).__name__))
                      for i, t in enumerate(слои[0])]
        л.info("[fa2_sm70 мегашаг] ТИПЫ ПОЗИЦИЙ слоя GDN: %s", для_печати)
        л.info("[fa2_sm70 мегашаг] ТИПЫ позиций слоя ВНИМАНИЯ: %s",
               [(i, (t.dtype if hasattr(t, "dtype") else type(t).__name__))
                for i, t in enumerate(next(v for v, g in zip(слои, флаги) if not g))])
        л.info("[fa2_sm70 мегашаг] ТИПЫ весов: rw=%s alog=%s dtb=%s wab=%s conv_w=%s",
               слои[0][0].dtype, слои[0][14].dtype, слои[0][15].dtype,
               слои[0][8].dtype, слои[0][12].dtype)
    # ЯДРО ПРИНИМАЕТ ОДИН масштаб на всю цепь. Значит равенство по слоям -- ПРЕДПОСЫЛКА,
    # и её надо проверять, а не предполагать: разные масштабы дали бы тихо неверный ответ.
    # ИМЯ `м` ЗДЕСЬ -- ЕЩЁ МЕТАДАННЫЕ ИЗ ЦИКЛА, а таблица меры называется `мера`. Столкновение
    # имён дало отказ на первом же шаге; берём явно.
    ks_общ, vs_общ = мера["kscale"], мера["vscale"]
    if масштабы:
        разные = {м2 for м2 in масштабы}
        if len(разные) != 1:
            raise RuntimeError(f"шаг_цепи: масштабы склада РАЗЛИЧАЮТСЯ по слоям {sorted(разные)} -- "
                               "ядро принимает один на цепь, нужен пер-слойный ввод")
        ks_общ, vs_общ = масштабы[0]
    _tцикл = __import__("time").perf_counter() if _проф0 else 0.0
    сл = fill_step(слои, флаги, cos_sin_row=строка, conv_states=conv, hstates=hs,
                   kpools=kps, vpools=vps, block_tables=bts)
    _tзап = __import__("time").perf_counter() if _проф0 else 0.0
    о = подготовить_обмен(int(x.numel()))
    _tобм = __import__("time").perf_counter() if _проф0 else 0.0
    # ПРИ ПОЗИЦИИ С КАРТЫ хостовое Sk не участвует в ответе: ядро считает его само.
    Sk = (int(позиция) + 1) if int(позиция) >= 0 else 1
    м = мера
    # РАЗЛОЖЕНИЕ ШАГА ПО ФАЗАМ: ЯДРО против ОБВЯЗКИ. Без него выбор следующего хода —
    # догадка: 5.24 ток/с на 50K может быть и медленным ядром, и питоном на каждом шаге.
    # События ставим ВОКРУГ пуска; барьер ставится и ДО отсчёта тоже (иначе в ядро
    # приписалось бы всё, что не синхронизировано с прошлого раза).
    # СИНХРОНИЗАЦИИ НА ШАГ ЗДЕСЬ НЕТ НАМЕРЕННО. Она отбирает у шага перекрытие ЦП с ГП, то есть
    # прибор менял бы измеряемое: обвязка, спрятанная за ядром, выглядела бы платной. События
    # пишутся в поток и читаются ОДИН раз на окно, когда они заведомо готовы.
    _проф = _проф0
    if _проф:
        _н, _к = torch.cuda.Event(True), torch.cuda.Event(True)
        _н.record()
    _tзов = __import__("time").perf_counter() if _проф else 0.0
    итог_цепи = _ext.decode_chain(
        x, сл, флаги, 128, 1e-6,
        м["scale_attn"], м["scale_gdn"],
        м["Hk"], м["Hv"], м["Hq"], м["Hkv"], Sk,
        rot=м["rot"], kv_pos=[(Sk - 1) if not g else -1 for g in флаги],
        qkvz_width=м["qkvz_width"], up_offset=м["up_offset"], mlp_width=м["mlp_width"],
        kscale=ks_общ, vscale=vs_общ,
        # ОБЩИЙ ПЛАН (2..4 карты, до двух ярусов). Прежняя парная четвёрка указателей осталась во
        # входе ядра ради одиночных ядер слоёв, но цепь идёт ТОЛЬКО через план.
        exch_tiers=int(о.get("ярусов", 0)) if о.get("нужен") else 0,
        exch_peers=list(о.get("соседей", [])) if о.get("нужен") else [],
        exch_pull=list(о.get("тянуть", [])) if о.get("нужен") else [],
        exch_far=list(о.get("far", [])) if о.get("нужен") else [],
        exch_in=list(о.get("inbox", [])) if о.get("нужен") else [],
        exch_far_flag=list(о.get("far_flag", [])) if о.get("нужен") else [],
        exch_my_flag=list(о.get("my_flag", [])) if о.get("нужен") else [],
        exch_width=int(о.get("ширина", 0)) if о.get("нужен") else 0,
        exch_count=о.get("счёт") if о.get("нужен") else None,
        dbg_plita=СЧЁТ.get("дбг_плита"),
        poz_dev=поз_тензор if прямые else None,
        slot_idx=([(с if с is not None else torch.empty(0, dtype=torch.int32, device=x.device))
                   for с in слот_тензоры] if прямые else None))
    _tпосле = __import__("time").perf_counter() if _проф else 0.0
    if _проф:
        _к.record()
    # ВЕРНУТЬ СОСТОЯНИЯ В СЛОТЫ ДВИЖКА. Без этого рекуррентная память теряется каждый шаг, и вывод
    # распадается после первого токена -- подпись дефекта обвязки, а не арифметики.
    for слот_conv, слот_h, gi2 in слоты:
        слот_conv.copy_(conv[gi2].transpose(-1, -2))
        if слот_h is not None:              # None = ядро писало ПРЯМО в слот, возвращать нечего
            слот_h.copy_(hs[gi2])
    if _проф:
        _tкон = __import__("time").perf_counter()
        # ОКНО, А НЕ НАКОПЛЕНИЕ. Накопительное среднее ПОЛЗЁТ ВВЕРХ при постоянной
        # величине (ранние шаги были дешевле и тянут сумму вниз) -- и это уже дало
        # неверный вывод. Признак вырождения: величина, обязанная стоять на месте, растёт.
        СЧЁТ.setdefault("проф_окно", []).append(
            (_н, _к,
             (_tзов - _tвх) * 1000.0,          # опись целиком
             (_tпосле - _tзов) * 1000.0,       # сам зов pybind (1728 тензоров на шаг)
             (_tкон - _tпосле) * 1000.0,       # возврат состояний в слоты движка
             _tвх,                             # начало шага -- для ПЕРИОДА между шагами
             (_tсост - _tвх) * 1000.0,         # ... из неё: цикл по слоям (96 копий состояний)
             (_tцикл - _tсост) * 1000.0,       # ... строка вращения + проверка масштабов
             (_tзап - _tцикл) * 1000.0,        # ... fill_step
             (_tобм - _tзап) * 1000.0))        # ... подготовить_обмен
        if len(СЧЁТ["проф_окно"]) >= 16:
            о = СЧЁТ.pop("проф_окно")
            torch.cuda.synchronize()           # ОДНА на окно, а не на шаг
            n = len(о)
            ср = lambda k: sum(x[k] for x in о) / n
            ядро = sum(a.elapsed_time(b) for a, b, *_ in о) / n
            # ПЕРИОД шага -- разность входов соседних шагов. Это и есть та единица, в которой
            # считается ток/с; сумма фаз меньше периода ровно на то, что идёт ВНЕ цепи.
            период = ((о[-1][5] - о[0][5]) / max(1, n - 1)) * 1000.0
            import logging
            logging.getLogger("vllm").info(
                "[fa2_sm70 мегашаг] ФАЗЫ окно=%d Sk=%d: ПЕРИОД=%.2f | ЯДРО=%.2f | "
                "ОПИСЬ=%.2f (слои=%.2f строка=%.2f заполн=%.2f обмен=%.2f) ЗОВ=%.2f ВОЗВРАТ=%.2f мс",
                n, Sk, период, ядро, ср(2), ср(6), ср(7), ср(8), ср(9), ср(3), ср(4))
    if СЧЁТ.get("дбг_плита") is not None and СЧЁТ.get("сверка_плиты") is None:
        СЧЁТ["сверка_плиты"] = 1
        try:
            сверить_плиту(г["модель"], СЧЁТ["дбг_плита"], x)
        except Exception as e:
            import logging
            logging.getLogger("vllm").info("[fa2_sm70 сверка ba] ОТКАЗ: %r", e)
    return итог_цепи


# ==================================================================================================
# РЕГИСТРАЦИЯ ОПЕРАЦИИ. Компилятор считает её непрозрачной и внутрь не заглядывает -- это штатный
# путь vLLM (образец: `gdn_attention_core` в том же файле модели).
# ==================================================================================================

# ==================================================================================================
# СНИМОК И ОТКАТ СОСТОЯНИЙ GDN. Слой GDN РЕКУРРЕНТЕН: его вызов пишет свёрточное и скрытое
# состояние В СЛОТЫ ДВИЖКА. Разрез по глубине прогонял эталонные слои 70 раз ДО первого измерения
# и мерил собственный след. Выдало это не правдоподобие чисел, а величина, которая различаться НЕ
# МОЖЕТ: эталон разошёлся между рангами (30.31 против 36.34), хотя сведение внутри строчно-
# параллельного слоя обязано делать карты одинаковыми.
# ==================================================================================================
def снимок_состояний(вн, мета, ve):
    """Пары (живой слот, копия) по всем слоям GDN. Копия -- отдельная память, не вид."""
    сн = []
    for сл in вн.layers:
        if not hasattr(сл, "linear_attn"):
            continue
        м = мета[сл.linear_attn.prefix]
        слот = м.non_spec_state_indices_tensor
        if слот is None or слот.numel() != 1:
            raise RuntimeError("снимок_состояний: ожидался ОДИН слот состояния")
        i = int(слот[0])
        кэш = сл.linear_attn.kv_cache[ve]
        сн.append((кэш[0][i], кэш[0][i].clone(), кэш[1][i], кэш[1][i].clone()))
    return сн


def вернуть_состояния(сн):
    """Откат ровно в снятое. Без него каждое следующее измерение стоит на следе предыдущего."""
    for живой_c, копия_c, живой_h, копия_h in сн:
        живой_c.copy_(копия_c)
        живой_h.copy_(копия_h)


def _мега_шаг(hidden: torch.Tensor, positions: torch.Tensor, out: torch.Tensor) -> None:
    # ИМЕНА ПАРАМЕТРОВ -- ЛАТИНИЦЕЙ: разборщик схемы операции кириллицу не принимает
    # ("expected a valid token but found ' '"). В теле и комментариях -- как удобно.
    # ЛЕНИВАЯ ПОДГОТОВКА: захватывать ссылки на веса при создании модели НЕЛЬЗЯ -- постобработка
    # квантованных слоёв может ПОДМЕНИТЬ тензоры, и мы бы держали чужие (класс "ключ по владению").
    СЧЁТ["цепью"] = СЧЁТ.get("цепью", 0) + 1      # считаем ВНУТРИ операции: она непрозрачна
    if СЧЁТ["цепью"] in (1, 8, 64, 512):
        import logging
        logging.getLogger("vllm").info("[fa2_sm70 мегашаг] ЦЕПЬЮ выполнено шагов: %d", СЧЁТ["цепью"])
    if _ГОТОВО.get("по умолчанию") is None:
        if _MODEL is None:
            raise RuntimeError("мега-шаг: владелец не объявлен")
        подготовить(_MODEL)
    x = hidden.reshape(-1)
    поз = int(positions.reshape(-1)[0].item())
    out.copy_(шаг_цепи(x, поз).reshape(out.shape))


def _мега_шаг_фиктивный(hidden: torch.Tensor, positions: torch.Tensor,
                        out: torch.Tensor) -> None:
    return


_ЗАРЕГИСТРИРОВАНА = False


def зарегистрировать():
    """Идемпотентно. Повторный вызов -- не ошибка, но и не вторая регистрация."""
    global _ЗАРЕГИСТРИРОВАНА
    if _ЗАРЕГИСТРИРОВАНА:
        return "уже"
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="fa2sm70_mega_step", op_func=_мега_шаг,
                              mutates_args=["out"], fake_impl=_мега_шаг_фиктивный)
    _ЗАРЕГИСТРИРОВАНА = True
    return "зарегистрирована"


def _доложить():
    """Печать по РЕДКИМ рубежам: журнал не заливаем, но маршрут виден."""
    n = СЧЁТ.get("вызовов", 0)
    if n and (n & (n - 1)) == 0:            # степени двойки
        import logging
        logging.getLogger("vllm").info(
            "[fa2_sm70 мегашаг] МАРШРУТ: вызовов=%d цепью=%d мимо=%d | первые причины=%s",
            n, СЧЁТ.get("цепью", 0), СЧЁТ.get("мимо", 0), СЧЁТ.get("почему", [])[:3])


def есть_метаданные() -> bool:
    """ПРОГРЕВ И ЗАХВАТ ГРАФОВ идут с ПУСТЫМИ метаданными -- там цепь применять нельзя.
    Условие НЕ тензорное (оно про контекст), поэтому трассировщик берёт его как обычное значение.
    Установлено отказом: пятый по счёту, и это не дефект, а пропущенное состояние.
    """
    try:
        from vllm.forward_context import get_forward_context
        м = get_forward_context().attn_metadata
        return isinstance(м, dict) and len(м) > 0
    except Exception:
        return False


def годен_ли_шаг(hidden, флаги=None) -> bool:
    """Цепь собрана под ОДИН токен одного запроса. Не подходит -- честно False, а не попытка."""
    # НИКАКИХ МУТАЦИЙ ЗДЕСЬ. Возврат первого отказа ("Builtin operator.* comparison with
    # constant self") совпал ровно с добавлением словаря/списка в эту функцию: трассировщик не
    # берёт мутацию питоновских структур внутри компилируемой области. Счёт ведём В ОПЕРАЦИИ --
    # она для компилятора непрозрачна. Инструмент оказался источником первого падения.
    # ТОЛЬКО ЧИСТОЕ ВЫЧИСЛЕНИЕ. Внутри компилируемого тела не проходит НИЧЕГО постороннего:
    # ни мутация словаря/списка, ни установка атрибута функции, ни вызов логгера -- каждое
    # проверено отдельным отказом. Наблюдать можно ТОЛЬКО изнутри операции (она непрозрачна).
    return есть_метаданные() and hidden.dim() == 2 and hidden.shape[0] == 1


# ==================================================================================================
# ПРОХОД ВЫШЕ КОМПИЛЯЦИИ. Здесь обычный питон: ветвление и наблюдение разрешены. Внутри же
# скомпилированного тела не проходит НИЧЕГО постороннего -- проверено шестью отказами.
# Возвращает None = "не мой случай", и вызывающий идёт обычным путём.
# ==================================================================================================
def _отказ(почему):
    """Доклад о ПЕРВЫХ отказах: выше компиляции наблюдение разрешено."""
    n = СЧЁТ.get("отказов", 0) + 1
    СЧЁТ["отказов"] = n
    # ПЕРВЫЙ РАЗ КАЖДОЙ ПРИЧИНЫ, а не первые пять подряд: пять подряд -- это пять раз ОДНА
    # причина, и настоящая (та, что срабатывает при захвате графа) в лог не попадала.
    вид = str(почему).split(":")[0]
    первый = вид not in СЧЁТ.setdefault("виды отказов", set())
    СЧЁТ["виды отказов"].add(вид)
    if первый or n <= 5 or n in (64, 512):
        import logging
        logging.getLogger("vllm").info("[fa2_sm70 мегашаг] ПРОХОД НЕ ВЗЯТ #%d: %s", n, почему)
    return None


def разрез_внимания(вн, h, r, positions, поз, _до, _после):
    """ОДИН слой внимания: наш против эталонного, ДВУМЯ срезами -- вход o_proj и выход слоя.

    Вход o_proj берётся у эталона перехватом (pre-hook), у нас -- слепком `dbg` одиночного
    ядра. Совпал вход o_proj -> дефект в o_proj/MLP; разошёлся -> в qkv/вращении/внимании.
    """
    import torch
    from fa2_sm70 import _ext as _e
    ext = _e.mega_ext()
    г = _ГОТОВО["по умолчанию"]
    слои, флаги, мера = г["слои"], г["флаги"], г["мера"]
    корень = г["модель"].layers
    i = next(L for L, g in enumerate(флаги) if not g)     # первый слой внимания в цепи
    сл = корень[мера["корни"][i]]
    v = слои[i]
    # ВХОД СЛОЯ: поток после всех предыдущих слоёв цепи, посчитанный ЭТАЛОНОМ с общего старта.
    вернуть_состояния(_до)
    hh, rr = h.clone(), r.clone()
    for сл_ in корень[мера["корни"][0]:мера["корни"][i]]:
        hh, rr = сл_(positions=positions, hidden_states=hh, residual=rr)
    вход = (hh + rr).reshape(-1).to(torch.float16).contiguous()

    from vllm.forward_context import get_forward_context
    фк = get_forward_context()
    мета = фк.attn_metadata[сл.self_attn.attn.layer_name]
    кэш = пул_kv(сл.self_attn.attn.kv_cache[фк.virtual_engine])
    вн_сл = сл.self_attn.attn
    ks = float(getattr(вн_сл, "_k_scale_float", 1.0) or 1.0)
    vs = float(getattr(вн_сл, "_v_scale_float", 1.0) or 1.0)
    строка = rope_row(сл.self_attn.rotary_emb.cos_sin_cache, int(поз))
    Sk = int(поз) + 1
    Nqkv = int(v[19].size(0))
    dbg = torch.zeros(Nqkv + 8, dtype=torch.float16, device=вход.device)

    # ЭТАЛОН: перехватываем ВХОД o_proj и, отдельно, q/k/v НА ВХОДЕ ЯДРА ВНИМАНИЯ.
    # Второй перехват делит первую половину слоя ещё раз: проекция+норма+вращение против
    # самого внимания. Слепок `dbg` держит k и v (их ядро не перезаписывает), q -- нет.
    поймано = {}

    def _крюк(_м, вх):
        поймано["o"] = вх[0].detach().reshape(-1).float().clone()

    def _крюк_qkv(_м, вх):
        поймано["q"] = вх[0].detach().reshape(-1).float().clone()
        поймано["k"] = вх[1].detach().reshape(-1).float().clone()
        поймано["v"] = вх[2].detach().reshape(-1).float().clone()

    ручка = сл.self_attn.o_proj.register_forward_pre_hook(_крюк)
    ручка2 = сл.self_attn.attn.register_forward_pre_hook(_крюк_qkv)
    try:
        э_h, э_r = сл(positions=positions, hidden_states=hh.clone(), residual=rr.clone())
    finally:
        ручка.remove(); ручка2.remove()
    эт_вых = (э_h + э_r).reshape(-1).float()

    # НАШ: то же самое одиночным ядром, с тем же входом и тем же складом.
    вернуть_состояния(_до)
    наш_вых = ext.attn_layer(
        вход, v[0], 1e-6, v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[1],
        v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18],
        128, мера["scale_attn"], мера["Hq"], мера["Hkv"],
        кэш[0], кэш[1], мета.block_table, int(кэш[0].size(1)), Sk, ks, vs,
        v[19], v[20], v[21], строка, мера["rot"], Sk - 1, dbg,
        мера["up_offset"], мера["mlp_width"]).reshape(-1).float()
    вернуть_состояния(_после)

    HD = int(мера["Hq"]) * int(мера["d"])
    наш_o = dbg[:HD].float()
    эт_o = поймано["o"][:HD]

    def _пара(a, b):
        na, nb2 = float(a.norm()), float(b.norm())
        кос = float(torch.dot(a, b) / (max(na, 1e-6) * max(nb2, 1e-6)))
        return f"|эт|={na:.3f} |наш|={nb2:.3f} отношение={nb2/max(na,1e-6):.3f} КОСИНУС={кос:.5f}"

    # K И V ПОСЛЕ НОРМЫ И ВРАЩЕНИЯ -- слепок их не перезаписывает (перезаписан только q).
    HKD = int(мера["Hkv"]) * int(мера["d"])
    наш_k = dbg[HD:HD + HKD].float()
    наш_v = dbg[HD + HKD:HD + 2 * HKD].float()
    части = "k: %s | v: %s" % (_пара(поймано["k"][:HKD], наш_k), _пара(поймано["v"][:HKD], наш_v))
    # ФОРМА СКЛАДА -- ПЕЧАТАТЬ, А НЕ ПРЕДПОЛАГАТЬ. Ядро берёт bsz=size(1) и шаг блока
    # stride(0): при раскладке [блоки, головы, токены, d] обе величины были бы верны по
    # отдельности и неверны вместе, и чтение ушло бы в чужие блоки МОЛЧА.
    сырой = сл.self_attn.attn.kv_cache[фк.virtual_engine]
    склад = "склад СЫРОЙ=%s | k: форма=%s шаги=%s тип=%s | bt=%s %s" % (
        tuple(сырой.shape),
        tuple(кэш[0].shape), tuple(кэш[0].stride()), кэш[0].dtype,
        tuple(мета.block_table.shape), мета.block_table.dtype)
    return ("РАЗРЕЗ ВНИМАНИЯ слой=%d Sk=%d | %s | %s | ВХОД o_proj: %s | "
            "ВЫХОД слоя (наш БЕЗ сведения TP): %s | pm=%.3f pl=%.3f nsp=%.0f" %
            (мера["корни"][i], Sk, склад, части, _пара(эт_o, наш_o), _пара(эт_вых, наш_вых),
             float(dbg[Nqkv]), float(dbg[Nqkv + 1]), float(dbg[Nqkv + 2])))



# ==================================================================================================
# СВОЙ CUDA-ГРАФ НА УЧАСТОК ЦЕПИ.
# Захват vLLM идёт через `_dummy_run`, а он зовёт `self.model(...)` НАПРЯМУЮ, мимо
# `_model_forward`, где стоит наш проход. Поэтому в чужой граф цепь не попадает НИКОГДА, а на
# исполнении наш проход возвращается раньше обёртки графа -- и весь шаг идёт энергично.
# Цена этого замерена A/B на ЧУЖОМ пути: 27.03 ток/с с графом против 19.66 без = 13.9 мс шага.
# Значит граф надо снимать самим, вокруг своего же участка [слой 0 + цепь + норма].
#
# ЧТО ЗАПЕКАЕТСЯ В ГРАФ -- то и должно быть постоянным: адреса входа/выхода (свои буферы),
# указатели складов, таблиц и слотов (живут сколько живёт запрос). Всё, что меняется КАЖДЫЙ шаг,
# уже переведено на карту (позиция, слот состояния, строка вращения) -- см. память про граф.
# КЛЮЧ пересъёмки собирается из ВСЕХ запечённых указателей: сменился хоть один -- снимаем заново,
# а не молча считаем по чужой памяти.
# ==================================================================================================
_ГРАФЫ = {}


def _ключ_графа(вн, фк, мета):
    """Всё, что граф запечёт как адрес. Меняется -- граф недействителен."""
    к = [фк.virtual_engine]
    for сл in вн.layers:
        if hasattr(сл, "linear_attn"):
            кэш = сл.linear_attn.kv_cache[фк.virtual_engine]
            м = мета[сл.linear_attn.prefix]
            к.append((кэш[0].data_ptr(), кэш[1].data_ptr(),
                      м.non_spec_state_indices_tensor.data_ptr()))
        elif hasattr(сл, "self_attn"):
            кэш = сл.self_attn.attn.kv_cache[фк.virtual_engine]
            м = мета[сл.self_attn.attn.layer_name]
            к.append((кэш.data_ptr(), м.block_table.data_ptr()))
    return tuple(к)


def в_графе(вн, фк, мета, x, positions, участок):
    """Выполнить `тело()` повтором графа. Возвращает (итог, взят_ли_граф).

    `тело` обязана быть ЧИСТОЙ по хосту: ни чтения с карты, ни выделений вне графа. Всё, что
    ей нужно из переменного, лежит в постоянных буферах и обновляется ДО повтора.
    """
    if os.environ.get("FA2SM70_MEGA_GRAPH", "1") != "1":
        return участок(), False
    ключ = _ключ_графа(вн, фк, мета)
    г = _ГРАФЫ.get("тек")
    if г is not None and г["ключ"] == ключ:
        г["вх_x"].copy_(x); г["вх_поз"].copy_(positions)
        # ВРЕМЯ ПОВТОРА -- событиями СНАРУЖИ графа (внутри они запрещены). Разность
        # «период минус повтор» и есть цена того, что делает движок вокруг нас.
        _сб = г.get("соб")
        if _сб is not None:
            _сб[0].record(); г["граф"].replay(); _сб[1].record()
            г["окно"].append(1)
            if len(г["окно"]) >= 32:
                г["окно"].clear(); torch.cuda.synchronize()
                import logging
                logging.getLogger("vllm").info(
                    "[fa2_sm70 повтор] участок=%.3f мс", _сб[0].elapsed_time(_сб[1]))
        else:
            г["граф"].replay()
        return г["вых"], True
    # ---- пересъёмка. Прогрев ОБЯЗАТЕЛЕН (требование torch), и он же двигает состояния GDN,
    # поэтому снимок/откат вокруг него -- не украшение, а условие верности.
    import logging
    сн = снимок_состояний(вн, мета, фк.virtual_engine)
    try:
        вх_x = x.clone(); вх_поз = positions.clone()
        поток = torch.cuda.Stream(); поток.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(поток):
            for _ in range(2):
                участок()
        torch.cuda.current_stream().wait_stream(поток)
        вернуть_состояния(сн)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            вых = участок()
        вернуть_состояния(сн)
        _ГРАФЫ["тек"] = {"ключ": ключ, "граф": g, "вх_x": вх_x, "вх_поз": вх_поз, "вых": вых,
                         "соб": ([torch.cuda.Event(True), torch.cuda.Event(True)]
                                 if os.environ.get("FA2SM70_CHAIN_PROF") == "1" else None),
                         "окно": []}
    except Exception as e:
        # ОТКАЗ ГРАФА -- НЕ ТИХИЙ: он меняет цену шага втрое, и молчание тут дороже падения.
        вернуть_состояния(сн)
        logging.getLogger("vllm").warning(
            "[fa2_sm70 мегашаг] ГРАФ НЕ СНЯТ (%s: %s) -- идём энергично", type(e).__name__, str(e)[:160])
        os.environ["FA2SM70_MEGA_GRAPH"] = "0"
        return участок(), False
    logging.getLogger("vllm").info("[fa2_sm70 мегашаг] ГРАФ СНЯТ (участок: слой 0 + цепь + норма)")
    # Первый ответ считаем повтором -- чтобы вход графа был ровно тот, что у вызывающего.
    _ГРАФЫ["тек"]["вх_x"].copy_(x); _ГРАФЫ["тек"]["вх_поз"].copy_(positions)
    g.replay()
    return вых, True



_РАННЕР = None


def разрез_gdn(вн):
    # ЧТО ЯДРО РЕАЛЬНО ДЕРЖИТ. Спор о раскладке плиты решается слепком из ядра, а не чтением
    # чужого кода: сравниваем слепок с эталоном модели в ОБОИХ прочтениях -- сплошном и по
    # k-голове. Какое совпадёт, то и правда.
    сл = вн.layers[1].linear_attn
    г = _ГОТОВО.get("по умолчанию")
    if г is None:
        return
    import logging
    л = logging.getLogger("vllm")
    Hk = int(г["мера"]["Hk"]); Hv = int(г["мера"]["Hv"]); DK = DV = 128
    Nq = Hk*DK*2 + Hv*DV; Nz = Hv*DV; Nab = 2*Hv; NT = Nq + Nz + Nab
    сл_дбг = torch.zeros(NT, dtype=torch.float16, device="cuda")
    СЧЁТ["дбг_плита"] = сл_дбг
    л.info("[fa2_sm70 разрез GDN] буфер слепка готов: NT=%d (Hk=%d Hv=%d)", NT, Hk, Hv)


def сверить_плиту(вн, слепок, поток):
    # ba НЕ проходит свёртку, поэтому его кусок слепка сравним с проекцией НАПРЯМУЮ.
    # Нормировку слоя делаем сами (в модели она в input_layernorm, у нас внутри ядра).
    сл = вн.layers[1]; ла = сл.linear_attn
    Hk = int(ла.num_k_heads // ла.tp_size); Hv = int(ла.num_v_heads // ла.tp_size)
    DK = int(ла.head_k_dim); DV = int(ла.head_v_dim); r = Hv // Hk
    Nq = Hk*DK*2 + Hv*DV; NQZ = Nq + Hv*DV; Nab = 2*Hv
    with torch.no_grad():
        x = поток.reshape(1, -1).float()
        w = сл.input_layernorm.weight.float()
        xn = (x * torch.rsqrt((x*x).mean(-1, keepdim=True) + 1e-6) * (1.0 + w)).half()
        ba = ла.in_proj_ba(xn)
        ba = (ba[0] if isinstance(ba, tuple) else ba).reshape(-1).float()
        qkvz = ла.in_proj_qkvz(xn)
        qkvz = (qkvz[0] if isinstance(qkvz, tuple) else qkvz).reshape(-1).float()
    наш = слепок[NQZ:NQZ+Nab].float()
    def rel(a, b): return ((a-b).norm()/b.norm().clamp_min(1e-30)).item()
    # прочтение 1: как есть (сплошное). прочтение 2: перестановка блоками по k-голове.
    пер = torch.empty_like(наш)
    for hk in range(Hk):
        for j in range(r):
            пер[hk*2*r + j]     = наш[hk*r + j]          # b <- первая половина
            пер[hk*2*r + r + j] = наш[Hv + hk*r + j]     # a <- вторая половина
    # z СВЁРТКУ НЕ ПРОХОДИТ -- значит его кусок слепка сравним с проекцией напрямую и
    # проверяет ГРАНИЦУ Nq/Nz. Два прочтения: сплошное и блоками по k-голове.
    z_наш = слепок[Nq:NQZ].float()
    z_эт = qkvz[Nq:NQZ]
    W = 2*DK + 2*r*DV
    z_бл = torch.empty_like(z_наш)
    for hk in range(Hk):
        for j in range(r):
            з = qkvz[hk*W + 2*DK + r*DV + j*DV : hk*W + 2*DK + r*DV + (j+1)*DV]
            z_бл[(hk*r + j)*DV : (hk*r + j + 1)*DV] = з
    import logging
    logging.getLogger("vllm").info(
        "[fa2_sm70 сверка плиты] ba: как_есть=%.3e переставлено=%.3e | "
        "z: сплошное=%.3e блоками_по_k=%.3e",
        rel(наш, ba), rel(пер, ba), rel(z_наш, z_эт), rel(z_наш, z_бл))


def обвесить_сэмплер():
    # ЦЕНА СЭМПЛЕРА -- ЗАМЕРОМ. В остатке шага 6.5 мс, и без разбиения непонятно, что там:
    # ядра сэмплера, синхронизация по токену или питон планировщика.
    р = _РАННЕР
    if р is None or getattr(р, "_fa2_сэмплер_обвешен", False):
        return
    исх = р.sampler
    сост = {"окно": []}

    class Обёртка:
        def __call__(self, *a, **kw):
            н, к = torch.cuda.Event(True), torch.cuda.Event(True)
            _t = __import__("time").perf_counter()
            н.record(); о = исх(*a, **kw); к.record()
            сост["окно"].append((н, к, (__import__("time").perf_counter() - _t) * 1e3))
            if len(сост["окно"]) >= 32:
                сп = сост.pop("окно"); сост["окно"] = []
                torch.cuda.synchronize()
                import logging
                logging.getLogger("vllm").info(
                    "[fa2_sm70 сэмплер] окно=%d: ГП=%.3f мс ЦП=%.3f мс",
                    len(сп), sum(x.elapsed_time(y) for x, y, _ in сп) / len(сп),
                    sum(z for _, _, z in сп) / len(сп))
            return о

        def __getattr__(self, имя):
            return getattr(исх, имя)

    р.sampler = Обёртка()
    р._fa2_сэмплер_обвешен = True


def обвесить_голову(модель):
    """ЦЕНА ГОЛОВЫ И СЭМПЛЕРА -- ЗАМЕРОМ. Хвост (7.9 мс со снятой цепью) я объявил чужим; спека
    же говорит, что это ярус Я5 нашего плана -- весь шаг одним пуском, включая сэмплер. Прежде
    чем строить, надо знать доли: со слоем 0 догадка была вчетверо мимо.

    Обёртка ставится ОДИН раз, события читаются окном (синхронизация одна на 32 шага).
    """
    if getattr(модель, "_fa2_голова_обвешена", False):
        return
    исх = модель.compute_logits
    сост = {"окно": []}

    def обёртка(*a, **kw):
        н, к = torch.cuda.Event(True), torch.cuda.Event(True)
        н.record(); р = исх(*a, **kw); к.record()
        сост["окно"].append((н, к))
        if len(сост["окно"]) >= 32:
            о = сост.pop("окно"); сост["окно"] = []
            torch.cuda.synchronize()
            import logging
            logging.getLogger("vllm").info(
                "[fa2_sm70 голова] окно=%d: compute_logits=%.3f мс, форма выхода=%s",
                len(о), sum(x.elapsed_time(y) for x, y in о) / len(о),
                tuple(р.shape) if hasattr(р, "shape") else "?")
        return р

    модель.compute_logits = обёртка
    модель._fa2_голова_обвешена = True


def попробовать_проход(модель, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    # ВХОД -- ВЛОЖЕНИЯ, А НЕ input_ids. Замер показал: раннер боевой сети ВСЕГДА передаёт готовые
    # вложения (мультимодальная обёртка считает их заранее). Моя предпосылка про input_ids была
    # неверна -- шестой случай за сутки, когда предпосылка о чужом интерфейсе не совпала с фактом.
    if intermediate_tensors is not None:
        return _отказ("промежуточные тензоры (не последний этап конвейера)")
    if _MODEL is None:
        return _отказ("владелец не объявлен")
    if not есть_метаданные():
        return _отказ("нет метаданных (прогрев/захват графов)")
    вн = _MODEL                            # держатель слоёв (объявил себя при создании)
    _пр = os.environ.get("FA2SM70_CHAIN_PROF") == "1"
    _p0 = __import__("time").perf_counter() if _пр else 0.0
    # ЧТЕНИЕ ПОЗИЦИИ -- ЭТО СИНХРОНИЗАЦИЯ ПОТОКА (`.item()` ждёт всю очередь) И ГЛАВНОЕ --
    # хостовое число, которое при захвате шага в граф замёрзло бы. Цепь берёт позицию с карты,
    # поэтому читаем её ТОЛЬКО когда её просит отладочный разрез.
    _нужна_поз = (os.environ.get("FA2SM70_MEGA_STEPDIFF") == "1"
                  or os.environ.get("FA2SM70_MEGA_DIFF") == "1"
                  or os.environ.get("FA2SM70_MEGA_ATTN") == "1")
    поз = int(positions.reshape(-1)[0].item()) if _нужна_поз else -1
    _p1 = __import__("time").perf_counter() if _пр else 0.0
    x = inputs_embeds
    if x is None:
        if input_ids is None:
            return _отказ("нет ни вложений, ни input_ids")
        x = вн.embed_input_ids(input_ids)
    if x.dim() != 2 or x.shape[0] != 1:
        return _отказ(f"не один токен: вложения {tuple(x.shape)}")
    # СЛОЙ 0 -- обычным путём (он не квантован, цепь его не покрывает); соглашение границы у них
    # отложенное, поэтому поток складываем ЯВНО.
    # КЛОН ТОЛЬКО ПОД ОДНОРАЗОВЫЙ ИНВАРИАНТ. Он печатается один раз за подъём, а клон делался
    # КАЖДЫЙ шаг -- лишнее выделение и пуск на ровном месте.
    # ---- ВЕТВЬ ГРАФА. Берётся только когда всё уже проверено энергичным путём (цепь
    # подготовлена, типы сошлись, отказов нет) и не включён ни один отладочный разрез:
    # разрезы делают хостовую работу, которая в граф не входит.
    _отл = (os.environ.get("FA2SM70_MEGA_STEPDIFF") == "1"
            or os.environ.get("FA2SM70_MEGA_DIFF") == "1"
            or os.environ.get("FA2SM70_MEGA_ATTN") == "1")
    if (not _отл) and _ГОТОВО.get("по умолчанию") is not None and СЧЁТ.get("проходов", 0) >= 2:
        from vllm.forward_context import get_forward_context as _гфк
        _фк = _гфк(); _мт = _фк.attn_metadata
        if isinstance(_мт, dict) and _мт:
            _вх = _ГРАФЫ.get("вход")
            if (_вх is None or _вх["x"].shape != x.shape
                    or _вх["поз"].shape != positions.shape or _вх["x"].dtype != x.dtype):
                _вх = {"x": x.detach().clone(), "поз": positions.detach().clone()}
                _ГРАФЫ["вход"] = _вх; _ГРАФЫ.pop("тек", None)
            _вх["x"].copy_(x); _вх["поз"].copy_(positions)

            # СОБЫТИЙ ВНУТРИ ЗАХВАТА БЫТЬ НЕ МОЖЕТ: `cudaEventRecord` с включённым таймингом в
            # режиме захвата отвергается (`invalid argument`), и прибор ломает измеряемое. Долю
            # слоя 0 меряем СНЯТИЕМ ФАЗЫ: гейт ниже выбрасывает его целиком (ответ заведомо
            # неверен, читается только время).
            _без0 = os.environ.get("FA2SM70_БЕЗ_СЛОЯ0") == "1"

            def _участок():
                if _без0:
                    _с = _вх["x"].reshape(-1).float()
                else:
                    _h, _r = вн.layers[0](positions=_вх["поз"], hidden_states=_вх["x"], residual=None)
                    _с = (_h + _r).reshape(-1)
                _п = _с if _с.dtype is torch.float16 else _с.to(torch.float16)
                # СНЯТИЕ ЦЕПИ ЦЕЛИКОМ: остаётся ровно то, что платит движок вокруг нас (голова,
                # сэмплер, планировщик). Ответ мусор, читается ТОЛЬКО период.
                _из = (_п if os.environ.get("FA2SM70_БЕЗ_ЦЕПИ") == "1"
                       else шаг_цепи(_п, -1, поз_тензор=_вх["поз"].reshape(-1)[:1]))
                _ит, _ = вн.norm(_из.reshape(1, -1), torch.zeros_like(_из).reshape(1, -1))
                return _ит

            _т0 = __import__("time").perf_counter() if _пр else 0.0
            _рез, _взят = в_графе(вн, _фк, _мт, _вх["x"], _вх["поз"], _участок)
            if _взят:
                СЧЁТ["проходов"] = СЧЁТ.get("проходов", 0) + 1
                if _пр:
                    о3 = СЧЁТ.setdefault("проф_граф", []); о3.append(_т0)
                    if len(о3) >= 32:
                        СЧЁТ.pop("проф_граф")
                        import logging
                        logging.getLogger("vllm").info(
                            "[fa2_sm70 граф] окно=%d ПЕРИОД=%.2f мс (слой0 %s, цепь %s)",
                            len(о3), (о3[-1] - о3[0]) / (len(о3) - 1) * 1e3,
                            "СНЯТ" if _без0 else "на месте",
                            "СНЯТА" if os.environ.get("FA2SM70_БЕЗ_ЦЕПИ") == "1" else "на месте")
                return _рез
    x0 = x.detach().clone() if СЧЁТ.get("инвар") is None else None
    # ВРЕМЯ СЛОЯ 0 -- СОБЫТИЯМИ, а не секундомером ЦП: он идёт чужим путём и целиком на ГП.
    if _пр:
        _e0, _e1 = torch.cuda.Event(True), torch.cuda.Event(True); _e0.record()
    h, r = вн.layers[0](positions=positions, hidden_states=x, residual=None)
    if _пр:
        _e1.record()
    сырой = (h + r).reshape(-1)
    _p2 = __import__("time").perf_counter() if _пр else 0.0
    # ИНВАРИАНТНОСТЬ ПО РАНГАМ -- ВЕЛИЧИНА, КОТОРАЯ РАЗЛИЧАТЬСЯ НЕ МОЖЕТ. При TP сведение внутри
    # строчно-параллельного слоя делает карты одинаковыми; расхождение здесь означает, что все
    # отношения "цепь/эталон" сравнивают величины НЕИЗВЕСТНОЙ ПОЛНОТЫ. Печатаем ДО цепи, один раз.
    if СЧЁТ.get("инвар") is None:
        СЧЁТ["инвар"] = 1
        import logging
        logging.getLogger("vllm").info(
            "[fa2_sm70 мегашаг] ИНВАРИАНТ: |вход0|=%.4f |вклад слоя|=%.4f |h|=%.4f |r|=%.4f",
            float(x0.reshape(-1).norm()),
            float((r.reshape(-1) - x0.reshape(-1)).norm()),
            float(h.reshape(-1).norm()), float(r.reshape(-1).norm()))
    # ПРИВЕДЕНИЕ ТИПА НА ГРАНИЦЕ, С ПРОВЕРКОЙ В ОБЕ СТОРОНЫ. vLLM ведёт остаточный поток в fp32
    # (точность сложения), цепь собрана под fp16. Молча приводить нельзя: неожиданный тип должен
    # ОТВЕРГАТЬСЯ, иначе это девятый стык подряд, закрытый догадкой.
    if сырой.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return _отказ(f"неожиданный тип потока: {сырой.dtype}")
    исходный = сырой.dtype
    поток = сырой if исходный is torch.float16 else сырой.to(torch.float16)
    # ЛЕНИВАЯ ПОДГОТОВКА ЗДЕСЬ ЖЕ. Она была только внутри зарегистрированной операции, а этот
    # проход идёт МИМО неё -- отсюда отказ «цепь не подготовлена» на первом же шаге декода.
    # Место выбрано после всех отказов: на префилле готовить незачем. Ветвь одинакова на всех
    # картах (обе увидели один и тот же однотокенный шаг), значит и подготовка совпадает.
    if _ГОТОВО.get("по умолчанию") is None:
        подготовить(_MODEL)
        if os.environ.get("FA2SM70_CHAIN_PROF") == "1":
            обвесить_голову(модель); обвесить_сэмплер()
        if os.environ.get("FA2SM70_MEGA_GDN") == "1" and СЧЁТ.get("разрез_gdn") is None:
            СЧЁТ["разрез_gdn"] = 1
            try:
                разрез_gdn(вн)
            except Exception as e:
                import logging
                logging.getLogger("vllm").info("[fa2_sm70 разрез GDN] ОТКАЗ: %r", e)
    # ЕСЛИ ПЛАН ГОВОРИТ «NCCL» -- ЦЕПЬ БРАТЬ НЕЛЬЗЯ. Она сводит половины ВНУТРИ ядра и другого
    # способа не имеет: при выключенном своём обмене она молча посчитала бы по своей половине,
    # то есть выдала бы правдоподобный НЕВЕРНЫЙ ответ. Отказ здесь -- и обычный путь vLLM
    # сделает all-reduce сам. Вызов коллективный, его делают ОБЕ карты, ветвь одна.
    о = подготовить_обмен(int(поток.numel()))
    # СНЯТИЕ ФАЗЫ: FA2SM70_EXCH_OFF=1 отключает обмен В ЦЕПИ, оставляя всё остальное на месте.
    # Ответ заведомо неверен (половины не сложены), читается только ПАДАЕТ ИЛИ НЕТ.
    if os.environ.get("FA2SM70_EXCH_OFF") == "1":
        о = dict(о); о["нужен"] = False
    if о.get("путь") == "nccl":
        return _отказ("план обмена = nccl: цепь не умеет сводить половины вне ядра")
    # СНИМОК ДО ЦЕПИ, А НЕ ПОСЛЕ. Цепь ДВИГАЕТ состояния GDN. Эталон, посчитанный после неё,
    # идёт по ДВАЖДЫ сдвинутому состоянию -- прибор мерит собственный след, и это ровно та
    # ошибка, которая уже была один раз (см. память "прибор меняет измеряемое").
    _стд = os.environ.get("FA2SM70_MEGA_STEPDIFF") == "1" and СЧЁТ.get("проходов", 0) < 3
    # ЛЕСТНИЦА -- НА ПЕРВОМ ШАГЕ КАЖДОГО ИЗ ПЕРВЫХ ДВУХ ЗАПРОСОВ. Один запуск сервера даёт
    # тогда ДВЕ длины контекста, а рост расхождения с длиной отделяет "чтение постраничного
    # склада" от "потокенной арифметики". Признак начала запроса -- позиция НЕ следующая.
    _перв = (СЧЁТ.get("посл_поз") is None) or (поз != СЧЁТ["посл_поз"] + 1)
    СЧЁТ["посл_поз"] = поз
    _лестница = (os.environ.get("FA2SM70_MEGA_DIFF") == "1"
                 and _перв and СЧЁТ.get("лестниц", 0) < 2)
    _разрез = (os.environ.get("FA2SM70_MEGA_ATTN") == "1" and СЧЁТ.get("вним_разрез") is None)
    _до = _после = None
    if _стд or _лестница or _разрез:
        from vllm.forward_context import get_forward_context as _фк0
        _до = снимок_состояний(вн, _фк0().attn_metadata, _фк0().virtual_engine)
    _p3 = __import__("time").perf_counter() if _пр else 0.0
    из_цепи = шаг_цепи(поток, поз, поз_тензор=positions.reshape(-1)[:1])
    _p4 = __import__("time").perf_counter() if _пр else 0.0
    if _стд or _лестница or _разрез:
        from vllm.forward_context import get_forward_context as _фк1
        _после = снимок_состояний(вн, _фк1().attn_metadata, _фк1().virtual_engine)
    if из_цепи.dtype is not torch.float16:
        return _отказ(f"цепь вернула не fp16: {из_цепи.dtype}")
    # ВОЗВРАЩАТЬ В ТИПЕ ТЕЛА МОДЕЛИ, а не входа. Вход (вложения) пришёл fp32, но голова словаря
    # ждёт fp16 -- обратное приведение к типу входа дало отказ `sm70_f16_gemm: input must be float16`.
    тело = вн.norm.weight.dtype
    из_цепи = из_цепи if из_цепи.dtype is тело else из_цепи.to(тело)
    n = СЧЁТ.get("проходов", 0) + 1
    СЧЁТ["проходов"] = n
    if n in (1, 8, 64, 512):
        import logging
        logging.getLogger("vllm").info("[fa2_sm70 мегашаг] ПРОХОДОВ ЦЕПЬЮ: %d", n)
    # РАЗРЕЗ В СЕРВЕРЕ: тот же вход прогнать ОБЫЧНЫМ путём и сравнить. Делает своё дело один раз,
    # по гейту -- иначе это удвоение работы на каждом шаге.
    # СРАВНЕНИЕ ПО ШАГАМ, НА ПОЛНОЙ ГЛУБИНЕ. Первый токен верен, дальше распад -- подпись
    # ОБВЯЗКИ СОСТОЯНИЙ. Отношение на шаге 1 против шага 2 разделяет "цепь считает неверно"
    # и "цепь неверно передаёт состояние следующему шагу".
    if _стд:
        import logging
        _л = logging.getLogger("vllm")
        # ТРИ ТОЧКИ: _до (старт цепи), _после (то, что цепь оставила -- его и вернуть в конце),
        # между ними эталон считается С ТОГО ЖЕ СТАРТА, что и цепь. Иначе сравниваются два
        # РАЗНЫХ шага, и любое отношение недействительно.
        вернуть_состояния(_до)
        _hh, _rr = h.clone(), r.clone()
        for _сл in вн.layers[1:]:
            _hh, _rr = _сл(positions=positions, hidden_states=_hh, residual=_rr)
        _э = (_hh + _rr).reshape(-1).float()
        вернуть_состояния(_после)
        _ц_в = из_цепи.reshape(-1).float()
        _эт, _ц = float(_э.norm()), float(_ц_в.norm())
        # УГОЛ, А НЕ ДЛИНА. Отношение норм не различает "масштаб" и "неверно": финальная норма
        # масштаб гасит. Косинус отвечает на вопрос, верно ли НАПРАВЛЕНИЕ.
        _кос = float(torch.dot(_э, _ц_в) / (max(_эт, 1e-6) * max(_ц, 1e-6)))
        _л.info("[fa2_sm70 мегашаг] ШАГ %d: обычный=%.3f цепь=%.3f отношение=%.4f КОСИНУС=%.6f",
                СЧЁТ.get("проходов", 0), _эт, _ц, _ц / max(_эт, 1e-6), _кос)

    if _лестница:
        СЧЁТ["лестниц"] = СЧЁТ.get("лестниц", 0) + 1
        import logging
        л = logging.getLogger("vllm")
        л.info("[fa2_sm70 мегашаг] ЛЕСТНИЦА #%d при Sk=%d", СЧЁТ["лестниц"], поз + 1)
        # ЛЕСТНИЦА ПО ГЛУБИНЕ: где расхождение начинается и как растёт -- это отличит
        # "лишнее сложение на каждом слое" от "одна грубая ошибка".
        # ОБЕ ветви стартуют С ОДНОГО состояния `_до` (снятого ДО настоящего вызова цепи),
        # а в конце движок возвращается в `_после`. Прежняя версия снимала снимок ПОСЛЕ цепи
        # и потому сравнивала два РАЗНЫХ шага.
        # ОГОВОРКА: склад KV здесь НЕ откатывается -- цепь пишет по одной позиции на слой
        # внимания, и глубины со вниманием остаются приблизительными. Это названо, а не скрыто.
        for k in (1, 2, 3, 4, 7, 8, 15, 31, len(вн.layers) - 1):
            вернуть_состояния(_до)
            hh, rr = h.clone(), r.clone()
            for сл_ in вн.layers[1:1 + k]:
                hh, rr = сл_(positions=positions, hidden_states=hh, residual=rr)
            эв = (hh + rr).reshape(-1).float()
            вернуть_состояния(_до)
            цв = шаг_цепи(поток, поз, слоёв=k).reshape(-1).float()
            эт, ц = float(эв.norm()), float(цв.norm())
            кос = float(torch.dot(эв, цв) / (max(эт, 1e-6) * max(ц, 1e-6)))
            л.info("[fa2_sm70 мегашаг] ГЛУБИНА %2d: обычный=%.2f цепь=%.2f отношение=%.3f КОСИНУС=%.5f",
                   k, эт, ц, ц / max(эт, 1e-6), кос)
        вернуть_состояния(_после)
        л.info("[fa2_sm70 мегашаг] ЦЕПЬ ЦЕЛИКОМ: |цепь|=%.3f |вход|=%.3f",
               float(из_цепи.norm()), float(поток.norm()))

    # РАЗРЕЗ ВНУТРИ СЛОЯ ВНИМАНИЯ. Лестница показала: мешалка верна (косинус 0.9995), ломает
    # ПЕРВЫЙ ЖЕ слой внимания. Делим слой пополам ОДНИМ замером: сравниваем ВХОД o_proj
    # (выход внимания, умноженный на гейт) -- он же слепок `dbg` одиночного ядра. Величина
    # локальная по головам, значит от TP не зависит и сравнима без сведения.
    if _разрез:
        СЧЁТ["вним_разрез"] = 1
        import logging
        л2 = logging.getLogger("vllm")
        try:
            л2.info("[fa2_sm70 мегашаг] %s", разрез_внимания(вн, h, r, positions, поз, _до, _после))
        except Exception as e:                    # инструмент не имеет права ронять сервер
            л2.info("[fa2_sm70 мегашаг] РАЗРЕЗ ВНИМАНИЯ ОТКАЗ: %r", e)
            вернуть_состояния(_после)
    _p5 = __import__("time").perf_counter() if _пр else 0.0
    итог, _ = вн.norm(из_цепи.reshape(1, -1), torch.zeros_like(из_цепи).reshape(1, -1))
    if итог.dtype is not тело:
        return _отказ(f"норма вернула {итог.dtype}, тело модели {тело}")
    if _пр:
        _p6 = __import__("time").perf_counter()
        # ПЕРИОД БЕРЁТСЯ ОТ ВХОДА ДО ВХОДА: разность даёт то, что происходит ВНЕ прохода
        # (сэмплер, планировщик, голова) -- иначе эта доля остаётся безымянной.
        СЧЁТ.setdefault("проф_проход", []).append(
            (_p0, (_p1 - _p0) * 1e3, (_p2 - _p1) * 1e3, (_p3 - _p2) * 1e3,
             (_p4 - _p3) * 1e3, (_p5 - _p4) * 1e3, (_p6 - _p5) * 1e3, _e0, _e1))
        о2 = СЧЁТ["проф_проход"]
        if len(о2) >= 16:
            СЧЁТ.pop("проф_проход")
            n2 = len(о2); ср2 = lambda k: sum(z[k] for z in о2) / n2
            torch.cuda.synchronize()                      # ОДНА на окно
            гп0 = sum(z[7].elapsed_time(z[8]) for z in о2) / n2
            пер = (о2[-1][0] - о2[0][0]) / max(1, n2 - 1) * 1e3
            вне = пер - sum(ср2(k) for k in range(1, 7))
            import logging
            logging.getLogger("vllm").info(
                "[fa2_sm70 проход] окно=%d ПЕРИОД=%.2f | ЦП: позиция=%.2f слой0=%.2f цепь=%.2f "
                "норма=%.2f вне=%.2f | ГП: слой0=%.2f мс",
                n2, пер, ср2(1), ср2(2), ср2(4), ср2(6), вне, гп0)
    return итог


# ==================================================================================================
# ОБМЕН МЕЖДУ КАРТАМИ. `out_proj`/`o_proj` у vLLM -- RowParallelLinear: каждая карта даёт ЧАСТИЧНУЮ
# сумму. Ядро умеет сложить половины само, но ему нужен указатель на ящик СОСЕДА, а воркеры -- разные
# процессы. Дескриптор памяти передаём через ту же группу процессов, что уже есть у движка.
# ==================================================================================================
_ОБМЕН = {}


ШИРИНА_РАНДЕВУ = int(os.environ.get("FA2SM70_EXCH_WIDTH", "36"))
"""Сколько блоков сетки участвует в рандеву. ЗАМЕРЕНО (карты 0-1, мост, 10 КБ): 8 блоков -- 9.84 мкс,
36 -- 8.35, 80 -- 17.82. Оптимум не на краю: одновременные рандеву через мост толкаются друг с
другом. 36 совпало с `defaultBlockLimit` чужого ядра, сошлось независимо."""


def подготовить_обмен(K: int):
    """Один раз на подъём. Строит ПЛАН по топологии и раскрывает буферы всех соседей.

    Ветвь (свой путь либо NCCL, плоский план либо ярусный) выводится ТОЛЬКО из величин, одинаковых
    на всех картах: матрица `nvidia-smi topo -m` и размер посылки. Локального в решении нет ничего --
    расхождение ветви между картами вешает ядро насмерть, это уже случалось.
    """
    if _ОБМЕН.get("готов"):
        return _ОБМЕН
    import torch.distributed as dist
    from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
    from . import topology as топ
    мир = get_tensor_model_parallel_world_size()
    if мир < 2:
        _ОБМЕН.update({"готов": True, "нужен": False, "путь": "нет"})
        return _ОБМЕН
    гр = get_tp_group()
    ранг = гр.rank_in_group
    W = ШИРИНА_РАНДЕВУ

    план = топ.построить(ранг, мир, байт=K * 2)
    ок, почему = топ.согласован(мир, байт=K * 2)
    if not ок:
        # НЕ входим в обмен, о котором соседи не знают. Отказ громкий: молчаливый откат
        # выглядел бы как «работает», а половины не складывались бы.
        raise RuntimeError(f"fa2_sm70: план обмена НЕ согласован по рангам -- {почему}")
    if план.путь != "свой":
        _ОБМЕН.update({"готов": True, "нужен": False, "путь": план.путь, "план": план})
        return _ОБМЕН

    # Ящики и флаги индексируем ПО НОМЕРУ ОТПРАВИТЕЛЯ -- тогда ни одна ячейка не спорна:
    # в [t, s] лежит то, что прислал ранг s на ярусе t.
    # ПРЯМОЙ ДОСТУП МЕЖДУ КАРТАМИ -- ДО раскрытия чужих буферов. Раскрытый дескриптор даёт
    # указатель на чужую карту, но без включённого доступа ЯДРО по нему падает недопустимым
    # обращением: рукопожатие проходит, а первое же касание из ядра нет. Вход в расширении был,
    # и никто его не звал -- ровно поэтому отказ выглядел как дефект ядра, а не обвязки.
    from fa2_sm70 import _ext as _e
    м = _e.mega_ext()                       # тот же модуль, что исполняет цепь
    видимых = torch.cuda.device_count()
    for a in range(видимых):
        for b in range(видимых):
            if a != b:
                м.enable_p2p(a, b)

    # ОБЩАЯ ПАМЯТЬ -- СВОЯ, А НЕ ЧЕРЕЗ TORCH. Механизм torch рассчитан на разделение памяти ОДНОЙ
    # карты между процессами: отображение создаётся в контексте карты-ВЛАДЕЛЬЦА, и указатель
    # непригоден для ядра, работающего на НАШЕЙ карте. Плюс дескриптор берётся только с НАЧАЛА
    # выделения, а распределитель torch отдаёт куски внутри большого блока. Оба свойства делают
    # чужой указатель тихо неверным: рукопожатие проходит, падает первое касание из ядра.
    Я = план.рандеву
    байт_ящиков, байт_флагов = Я * мир * K * 2, Я * мир * W * 4
    птр_ящиков, оп_ящиков = м.shared_alloc(байт_ящиков)
    птр_флагов, оп_флагов = м.shared_alloc(байт_флагов)
    ящики = м.shared_view(птр_ящиков, [Я, мир, K], True)
    флаги = м.shared_view(птр_флагов, [Я, мир, W], False)

    все = [None] * гр.world_size
    dist.all_gather_object(все, (оп_ящиков, оп_флагов), group=гр.device_group)

    # ЧУЖУЮ ПАМЯТЬ ДЕРЖИМ СЫРЫМ УКАЗАТЕЛЕМ. Оборачивать её в тензор нельзя: распределитель torch
    # начал бы вести учёт памяти, которой не владеет, а проверить форму всё равно нечем.
    чужие = {}
    for r in range(мир):
        if r == ранг:
            continue
        чужие[r] = (м.shared_open(все[r][0]), м.shared_open(все[r][1]))

    # Разворачиваем план в плоские списки в порядке ЯРУС -> СОСЕД: ядро читает их так же.
    # СРЕЗ ДЛЯ ОТЛАДКИ: FA2SM70_EXCH_SELF=1 направляет обмен на СВОИ ЖЕ буферы. Ответ заведомо
    # неверен, читается только ПАДАЕТ ИЛИ НЕТ. Если с ним не падает -- виноват ЧУЖОЙ УКАЗАТЕЛЬ
    # (раскрытие дескриптора), а не логика ядра; если падает -- наоборот. Одно плечо, один ответ.
    срез_свои = os.environ.get("FA2SM70_EXCH_SELF") == "1"

    far, inbox, far_flag, my_flag, соседей = [], [], [], [], []
    for t, соседи in enumerate(план.ярусы):
        соседей.append(len(соседи))
        for s in соседи:
            чя, чф = (птр_ящиков, птр_флагов) if срез_свои else чужие[s]
            # В СРЕЗЕ ждать надо ТОТ ЖЕ слот, в который сам и пишешь, иначе ожидание сторожит
            # ячейку, которую никто не заполнит, и вместо ответа получается зависание.
            откуда = ранг if срез_свои else s
            # ЧУЖАЯ сторона -- АДРЕС СЛОТА, посчитанный от начала чужого выделения. Раскладка
            # [ярус][ОТПРАВИТЕЛЬ][длина] одинакова у всех, поэтому смещение считается, а не
            # запрашивается: спорной ячейки не существует по построению.
            far.append(чя + ((t * мир + ранг) * K) * 2)          # КУДА кладу свою долю у соседа s
            far_flag.append(чф + ((t * мир + ранг) * W) * 4)     # флаг «от меня» у соседа s
            inbox.append(ящики[t, откуда])                      # ЧТО прислал мне сосед s
            my_flag.append(флаги[t, откуда])                    # мой флаг «от соседа s»

    _ОБМЕН.update({
        "готов": True, "нужен": True, "путь": "свой", "ранг": ранг, "план": план,
        "ярусов": Я, "соседей": соседей,
        # ТЯНУЩИЙ режим в ядре есть, но здесь НЕ разведён: тянуть надо не из ящика, а из рабочего
        # буфера соседа, а он не расшарен. На мосту это и не нужно -- замер даёт толкающий дешевле
        # в 1.34-1.54 раза. Разводить под NVLink, когда будет чем померить боевую пару.
        "тянуть": [0] * Я,
        "far": far, "inbox": inbox, "far_flag": far_flag, "my_flag": my_flag,
        # СЧЁТЧИК КРУГОВ -- ПОСТОЯННЫЙ и ПРИВАТНЫЙ (не расшарен). Живёт столько же, сколько ящики:
        # он монотонен между шагами, оттого ожидание не проскакивает по старым флагам, и он
        # безопасен в CUDA-графе, где запечённый с хоста номер никогда бы не менялся.
        "счёт": torch.zeros(W, dtype=torch.int32, device="cuda"),
        "ширина": W, "ящики": ящики, "флаги": флаги, "чужие": чужие,
        # ДЕРЖАТЬ ССЫЛКИ НА ВЫДЕЛЕНИЯ: вид `from_blob` памятью не владеет, и без этих полей
        # ничто не мешало бы освободить её раньше, чем ядро перестанет по ней ходить.
        "птр": (птр_ящиков, птр_флагов), "дескрипторы": (оп_ящиков, оп_флагов),
    })
    import logging
    logging.getLogger("vllm").info(
        "[fa2_sm70 мегашаг] ОБМЕН: ранг %d/%d, %s, ширина рандеву %d, ящик %d половинок",
        ранг, мир, план.описание(тянуть=[0] * Я), W, K)
    return _ОБМЕН
