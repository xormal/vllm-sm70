"""fp16 checkpoint -> GPTQ-format int8 (RTN, per-group asymmetric).

Why RTN and not real GPTQ: at 8 bits the Hessian-based error compensation buys very little over
round-to-nearest (the quantisation grid is already fine), while costing hours of calibration and a
new dependency. The packing format is the part that must be exact, and it was validated against
vLLM's own gptq_gemm before this script was written (relL2 1.09e-3 on a toy layer).
"""
import json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1]; DST = sys.argv[2]
BITS, GROUP = 8, 128
PF = 32 // BITS
DEV = "cuda"

# Mirror the AWQ checkpoint's modules_to_not_convert exactly, so vLLM builds the same module
# structure it already builds today -- one variable at a time.
SKIP = [r".*visual.*", r".*\.self_attn\.(q|k|v)_proj$", r".*\.linear_attn\.in_proj_(a|b)$",
        r".*\.layers\.0\..*", r".*mtp.*", r"^lm_head$"]
SKIP_RE = [re.compile(p) for p in SKIP]
QUANT_SUFFIX = ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.o_proj",
                "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")

def should_quant(mod: str) -> bool:
    if any(r.match(mod) for r in SKIP_RE): return False
    return any(mod.endswith(s) for s in QUANT_SUFFIX)

def quant_pack(W):
    """W [out, in] fp16 -> GPTQ v1 tensors. Quantisation runs along the INPUT dim."""
    Wt = W.t().contiguous().to(DEV).float()          # [in, out]
    inf, outf = Wt.shape
    if inf % GROUP or outf % PF: return None
    ng = inf // GROUP
    Wg = Wt.view(ng, GROUP, outf)
    mx, mn = Wg.amax(1), Wg.amin(1)
    maxq = 2 ** BITS - 1
    scale = ((mx - mn).clamp(min=1e-8) / maxq)
    zero = torch.round(-mn / scale).clamp(0, maxq)
    q = torch.clamp(torch.round(Wg / scale[:, None, :]) + zero[:, None, :], 0, maxq).to(torch.int32)
    q = q.view(inf, outf)
    qweight = torch.zeros((inf // PF, outf), dtype=torch.int32, device=DEV)
    for j in range(PF): qweight |= (q[j::PF] & 0xFF) << (BITS * j)
    z = (zero.to(torch.int32) - 1).clamp(min=0)      # v1 convention: stored zero is zero-1
    qzeros = torch.zeros((ng, outf // PF), dtype=torch.int32, device=DEV)
    for j in range(PF): qzeros |= (z[:, j::PF] & 0xFF) << (BITS * j)
    g_idx = (torch.arange(inf, device=DEV) // GROUP).to(torch.int32)
    return {"qweight": qweight.cpu(), "qzeros": qzeros.cpu(),
            "scales": scale.half().cpu(), "g_idx": g_idx.cpu()}

os.makedirs(DST, exist_ok=True)
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
new_map, total_q, total_kept, t0 = {}, 0, 0, time.perf_counter()
for si, sh in enumerate(shards):
    out = {}
    with safe_open(os.path.join(SRC, sh), framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            mod = name[:-len(".weight")] if name.endswith(".weight") else None
            if mod is not None and t.dim() == 2 and should_quant(mod):
                packed = quant_pack(t)
                if packed is None:
                    out[name] = t; total_kept += 1
                else:
                    for k, v in packed.items(): out[f"{mod}.{k}"] = v
                    total_q += 1
            else:
                out[name] = t; total_kept += 1
    newname = sh.replace(".safetensors", "-int8.safetensors")
    save_file(out, os.path.join(DST, newname), metadata={"format": "pt"})
    for k in out: new_map[k] = newname
    del out; torch.cuda.empty_cache()
    print(f"[{si+1}/{len(shards)}] {sh} -> {newname}  quantised={total_q} kept={total_kept} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(DST, f))
                                          for f in set(new_map.values()))},
           "weight_map": new_map},
          open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)
cfg = json.load(open(os.path.join(SRC, "config.json")))
# [ЭТОТ БЛОК СТОИЛ ЗАБОРА ИЗ ВОСКЛИЦАНИЙ. Первая редакция писала ТОЛЬКО `dynamic`, и притом по
#  ИМЕНАМ ИЗ СЛЕПКА -- а движок к моменту вопроса "квантован ли модуль" уже СЛИЛ q/k/v в
#  `qkv_proj`, а in_proj_a/b в `in_proj_ba`. Оба шаблона мимо, `modules_to_not_convert` нет вовсе:
#  плотные модули строятся как квантованные, сервер ПОДНИМАЕТСЯ, ОБСЛУЖИВАЕТ и выдаёт `!!!!!!`.
#  Нужны ОБА поля и `dynamic` -- по СЛИТЫМ именам. Проверка -- генерацией, не подъёмом.]
SKIP_FUSED = [r".*visual.*", r".*\.self_attn\.qkv_proj$", r".*\.linear_attn\.in_proj_ba$",
              r".*\.layers\.0\..*", r".*mtp.*", r"^lm_head$"]
NOT_CONVERT = ["visual", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
               "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "model.layers.0.", "model.language_model.layers.0.", "mtp"]
cfg["quantization_config"] = {
    "quant_method": "gptq", "bits": BITS, "group_size": GROUP,
    "desc_act": False, "sym": False, "lm_head": False, "checkpoint_format": "gptq",
    "dynamic": {f"-:{p}": {} for p in SKIP_FUSED},
    "modules_to_not_convert": NOT_CONVERT,
}
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1)
for f in os.listdir(SRC):
    if f.endswith((".json", ".txt", ".py", ".jinja", ".model")) and f not in (
            "config.json", "model.safetensors.index.json"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
print(f"DONE quantised={total_q} kept={total_kept} in {time.perf_counter()-t0:.0f}s")
print("INT8_DONE")
