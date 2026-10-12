#!/usr/bin/env python
"""One script to A/B a Gemma model across attention backends on different GPUs, 1:1.
  ATTN=fa2_sm70  MODEL=/path CUDA_VISIBLE_DEVICES=0 python examples/compare_run.py   # our sm70 kernel (V100)
  ATTN=flash_attention_2 MODEL=/path CUDA_VISIBLE_DEVICES=3 python examples/compare_run.py  # real FA2 (A100)
Prints prefill/decode tok/s and, for each fixed prompt, the greedy answer text + the first 24 generated
token ids -- so the two runs can be diffed token-for-token (identical ids => bit-identical decisions)."""
import os, time, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ["MODEL"]
ATTN = os.environ.get("ATTN", "fa2_sm70")
torch.manual_seed(0)

if ATTN == "fa2_sm70":
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import fa2_sm70
    def fa2_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        D, Sq, Sk = query.shape[-1], query.shape[2], key.shape[2]
        scale = float(scaling) if scaling is not None else D ** -0.5
        win = int(kw.get("sliding_window") or getattr(module, "sliding_window", None) or 0)
        sc = float(kw.get("softcap") or kw.get("attn_logit_softcapping") or 0.0)
        if Sq == 1 and Sk > 1:
            K, V = (key, value)
            if win and Sk > win: K, V = key[:, :, -win:].contiguous(), value[:, :, -win:].contiguous()
            o, _ = fa2_sm70.decode(query.contiguous(), K.contiguous(), V.contiguous(), scale=scale, softcap=sc)
        else:
            o, _ = fa2_sm70.attention(query, key, value, causal=True, scale=scale, window=win, softcap=sc)
        return o.transpose(1, 2).contiguous().to(query.dtype), None
    try: ALL_ATTENTION_FUNCTIONS.register("fa2_sm70", fa2_attn)
    except Exception: ALL_ATTENTION_FUNCTIONS["fa2_sm70"] = fa2_attn

dev_name = torch.cuda.get_device_name(0)
print(f"[{ATTN}] transformers {transformers.__version__} | GPU {dev_name} cc{torch.cuda.get_device_capability(0)}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation=ATTN).eval()
dev = next(model.parameters()).device
print(f"[{ATTN}] loaded in {time.time()-t0:.0f}s; attn={model.config._attn_implementation}", flush=True)

def enc_of(prompt):
    e = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                return_tensors="pt", return_dict=True, tokenize=True)
    return {k: v.to(dev) for k, v in e.items() if hasattr(v, "to")}

def gen(prompt, n=90):
    e = enc_of(prompt); plen = e["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**e, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    ids = out[0, plen:]
    return tok.decode(ids, skip_special_tokens=True).strip(), ids[:24].tolist()

# warmup
_ = gen("Say hello in one word.", n=8); torch.cuda.synchronize()

# prefill / decode timing (manual)
e = enc_of("Explain, step by step, why moving clocks and clocks deep in gravity wells run slow, and how "
           "GPS corrects for both effects. Be thorough."); ids = e["input_ids"]; P = ids.shape[1]; N = 128
torch.cuda.synchronize(); t = time.time()
with torch.no_grad(): o = model(ids, use_cache=True)
torch.cuda.synchronize(); tpf = time.time() - t
past = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
torch.cuda.synchronize(); t = time.time()
with torch.no_grad():
    for _ in range(N):
        oo = model(nxt, past_key_values=past, use_cache=True); past = oo.past_key_values; nxt = oo.logits[:, -1:].argmax(-1)
torch.cuda.synchronize(); tdc = time.time() - t
print(f"[{ATTN}] PREFILL {P} tok: {tpf*1e3:.1f} ms -> {P/tpf:.1f} tok/s   |   DECODE {N} tok: {tdc*1e3:.1f} ms -> {N/tdc:.1f} tok/s ({tdc/N*1e3:.1f} ms/tok)", flush=True)

print(f"[{ATTN}] === Q&A (greedy) ===", flush=True)
for q in ["What is the capital of France, and what river runs through it?",
          "Write a one-line Python function that returns the n-th Fibonacci number.",
          "In two sentences, explain why the sky is blue.",
          "What is 17 * 24? Show the steps."]:
    a, ids24 = gen(q)
    print(f"\nQ: {q}\nA: {a}\n   first24_token_ids: {ids24}", flush=True)
print(f"\n[{ATTN}] DONE")
