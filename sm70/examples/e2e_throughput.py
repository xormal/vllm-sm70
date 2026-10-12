#!/usr/bin/env python
"""Whole-model prefill(pp)/decode throughput for Gemma-4 on one GPU.
  ATTN=fa2_sm70        -> our sm70 kernel, every layer  (V100)
  ATTN=sdpa            -> torch SDPA, every layer        (A100 fallback; FA2 can't do d512 global)
  ATTN=hybrid          -> real FlashAttention-2 on the d<=256 sliding layers + SDPA on the d512 global
                          layers = the *fastest a reference stack can serve this model* (A100)
Prints prompt-processing (pp) tok/s and decode tok/s. MODEL, PPLEN env-tunable."""
import os, time, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
MODEL = os.environ["MODEL"]; ATTN = os.environ.get("ATTN", "fa2_sm70")
PPLEN = int(os.environ.get("PPLEN", "1024")); NGEN = int(os.environ.get("NGEN", "128"))
torch.manual_seed(0)

impl = ATTN
if ATTN == "fa2_sm70":
    import fa2_sm70
    def fn(module, q, k, v, am, scaling=None, dropout=0.0, **kw):
        D, Sq, Sk = q.shape[-1], q.shape[2], k.shape[2]; sc = float(scaling) if scaling is not None else D**-0.5
        win = int(kw.get("sliding_window") or getattr(module, "sliding_window", None) or 0)
        if Sq == 1 and Sk > 1:
            K, V = (k, v) if not (win and Sk > win) else (k[:, :, -win:].contiguous(), v[:, :, -win:].contiguous())
            o, _ = fa2_sm70.decode(q.contiguous(), K.contiguous(), V.contiguous(), scale=sc)
        else:
            o, _ = fa2_sm70.attention(q, k, v, causal=True, scale=sc, window=win)
        return o.transpose(1, 2).contiguous().to(q.dtype), None
    try: ALL_ATTENTION_FUNCTIONS.register("fa2_sm70", fn)
    except Exception: ALL_ATTENTION_FUNCTIONS["fa2_sm70"] = fn
elif ATTN == "hybrid":
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    def fn(module, q, k, v, am, scaling=None, dropout=0.0, **kw):
        D = q.shape[-1]; sc = float(scaling) if scaling is not None else D**-0.5
        win = int(kw.get("sliding_window") or getattr(module, "sliding_window", None) or 0)
        if D <= 256:                                             # sliding layers -> REAL FlashAttention-2
            ws = (win - 1, 0) if win else (-1, -1)
            o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                causal=True, window_size=ws, softmax_scale=sc)
            return o.contiguous().to(q.dtype), None              # [B,S,H,D]
        g = q.shape[1] // k.shape[1]                             # d512 global -> SDPA (FA2 can't)
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                                           is_causal=True, scale=sc)
        return o.transpose(1, 2).contiguous().to(q.dtype), None
    try: ALL_ATTENTION_FUNCTIONS.register("hybrid", fn)
    except Exception: ALL_ATTENTION_FUNCTIONS["hybrid"] = fn

print(f"[{ATTN}] transformers {transformers.__version__} | {torch.cuda.get_device_name(0)} cc{torch.cuda.get_device_capability(0)}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation=impl).eval()
dev = next(model.parameters()).device
# a real ~PPLEN-token prompt (repeat a paragraph, then trim to PPLEN)
base = ("Relativity changes how we measure time and space; explain it carefully. ") * 200
ids = tok(base, return_tensors="pt").input_ids[:, :PPLEN].to(dev)
P = ids.shape[1]
# warmup
with torch.no_grad(): model(ids[:, :64], use_cache=True)
torch.cuda.synchronize()
# prefill (pp)
torch.cuda.synchronize(); t = time.time()
with torch.no_grad(): out = model(ids, use_cache=True)
torch.cuda.synchronize(); tpf = time.time() - t
past = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
# decode
torch.cuda.synchronize(); t = time.time()
with torch.no_grad():
    for _ in range(NGEN):
        o = model(nxt, past_key_values=past, use_cache=True); past = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
torch.cuda.synchronize(); tdc = time.time() - t
print(f"[{ATTN}]  pp({P} tok): {tpf*1e3:7.1f} ms -> {P/tpf:8.1f} tok/s   |   decode({NGEN} tok): {tdc*1e3:7.1f} ms -> {NGEN/tdc:6.1f} tok/s ({tdc/NGEN*1e3:.1f} ms/tok)", flush=True)
