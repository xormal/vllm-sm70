#!/usr/bin/env python
"""End-to-end demo: run a Gemma model on V100 (sm_70) with THIS package's FlashAttention-2 kernel
as the attention backend, and print prefill/decode tok/s plus explicit Q&A.

Gemma exercises exactly the features this kernel was built for: GQA, per-layer sliding-window
attention, large head_dim (256 for Gemma-4), and (Gemma-2) logit soft-capping.

Usage:
    # env: torch + this package (fa2_sm70) + transformers>=5.10 (Gemma-4) / >=4.42 (Gemma-2/3)
    MODEL=/path/to/gemma-4-12B-it python examples/run_gemma_fa2_sm70.py
    # single GPU (fits ~12B in bf16 on one 32 GB V100):  CUDA_VISIBLE_DEVICES=0 MODEL=... python ...
    # multi-GPU (device_map=auto):                        CUDA_VISIBLE_DEVICES=0,1 MODEL=... python ...

The kernel is registered as a transformers attention backend ("fa2_sm70"); every prefill and every
decoded token flows through it. Verified: bit-close to SDPA per layer (relL2 ~1e-3 vs fp32).
"""
import os, time, torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import fa2_sm70

MODEL = os.environ.get("MODEL")
if not MODEL:
    raise SystemExit("Set MODEL to a local Gemma model directory")
torch.manual_seed(0)

# ---- register this package as a transformers attention backend --------------------------------
_calls = {"prefill": 0, "decode": 0}
def fa2_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    # query [B,Hq,Sq,D]; key/value [B,Hkv,Sk,D] -- already RoPE'd + q/k-normed by the module.
    D, Sq, Sk = query.shape[-1], query.shape[2], key.shape[2]
    scale = float(scaling) if scaling is not None else D ** -0.5
    win = int(kw.get("sliding_window") or getattr(module, "sliding_window", None) or 0)
    sc = float(kw.get("softcap") or kw.get("attn_logit_softcapping") or 0.0)   # Gemma-2 attn softcap
    if Sq == 1 and Sk > 1:                                   # decode
        K, V = key, value
        if win and Sk > win:                                 # sliding layer: attend only the last `win` keys
            K, V = key[:, :, -win:].contiguous(), value[:, :, -win:].contiguous()
        o, _ = fa2_sm70.decode(query.contiguous(), K.contiguous(), V.contiguous(), scale=scale, softcap=sc)
        _calls["decode"] += 1
    else:                                                    # prefill (causal, optional sliding window)
        o, _ = fa2_sm70.attention(query, key, value, causal=True, scale=scale, window=win, softcap=sc)
        _calls["prefill"] += 1
    return o.transpose(1, 2).contiguous().to(query.dtype), None   # -> [B,Sq,Hq,D]

try:    ALL_ATTENTION_FUNCTIONS.register("fa2_sm70", fa2_attn)      # transformers >= 5
except Exception: ALL_ATTENTION_FUNCTIONS["fa2_sm70"] = fa2_attn   # transformers 4.x

# ---- load -------------------------------------------------------------------------------------
print(f"transformers {transformers.__version__} | loading {MODEL} (bf16) with attn=fa2_sm70 ...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="fa2_sm70").eval()
dev = next(model.parameters()).device
print(f"loaded in {time.time()-t0:.0f}s; attn = {model.config._attn_implementation}", flush=True)

def chat(prompt, max_new=96):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True, tokenize=True)
    enc = {k: v.to(dev) for k, v in enc.items() if hasattr(v, "to")}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ---- WARM UP before any timing ----------------------------------------------------------------
print("\n=== WARMUP ===", flush=True)
_ = chat("Say hello in one word.", max_new=8); torch.cuda.synchronize()
print("warmup done.", flush=True)

# ---- prefill / decode timing (manual, clean separation) --------------------------------------
print("\n=== BENCHMARK (attention = fa2_sm70 on sm_70) ===", flush=True)
enc = tok.apply_chat_template([{"role": "user", "content":
        "Explain, step by step, why moving clocks and clocks deep in gravity wells run slow, "
        "and how GPS corrects for both effects. Be thorough."}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True)
ids = enc["input_ids"].to(dev); P = ids.shape[1]; N = 128
torch.cuda.synchronize(); t = time.time()
with torch.no_grad(): out = model(ids, use_cache=True)
torch.cuda.synchronize(); t_prefill = time.time() - t
past = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
torch.cuda.synchronize(); t = time.time()
with torch.no_grad():
    for _ in range(N):
        o = model(nxt, past_key_values=past, use_cache=True)
        past = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
torch.cuda.synchronize(); t_decode = time.time() - t
print(f"  prompt tokens (prefill) = {P}")
print(f"  PREFILL : {t_prefill*1e3:8.1f} ms  ->  {P/t_prefill:8.1f} tok/s")
print(f"  DECODE  : {t_decode*1e3:8.1f} ms / {N} tok  ->  {N/t_decode:8.1f} tok/s  ({t_decode/N*1e3:.1f} ms/tok)")
print(f"  fa2_sm70 kernel calls so far: prefill={_calls['prefill']}, decode={_calls['decode']}", flush=True)

# ---- explicit Q&A (proves coherent output, not byte-salad) ------------------------------------
print("\n=== Q&A (greedy, every token through fa2_sm70) ===", flush=True)
for q in [
    "What is the capital of France, and what river runs through it?",
    "Write a one-line Python function that returns the n-th Fibonacci number.",
    "In two sentences, explain why the sky is blue.",
    "What is 17 * 24? Show the steps.",
]:
    print(f"\nQ: {q}\nA: {chat(q)}", flush=True)
print("\n=== DONE ===")
