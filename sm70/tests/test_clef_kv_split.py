"""Clef's per-rank split of the attention q/k/v LoRA must match vLLM's sharding at every TP (CPU, torch).

Qwen3.8-27B has 4 KV heads. At TP=8 vLLM replicates each KV head on 2 ranks (QKVParallelLinear:
num_kv_head_replicas = tp // n_kv, shard_rank = tp_rank // num_kv_head_replicas); a plain 1/tp row slice
would give every rank half a head and shift the columns of everything after it.
"""
import importlib.util
import os

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "clef_sborka", os.path.join(HERE, "..", "integrations", "vllm_sm70", "clef", "clef_sborka.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

NQ, NKV, HD, R, HID = 24, 4, 16, 8, 32          # head counts of the real model, small dims


def vllm_rows(t, rank, tp, n_heads):
    """rows of a q/k/v output that vLLM's weight loader gives `rank`"""
    replicas = max(1, tp // n_heads)
    per_rank = max(1, n_heads // tp)
    shard = rank // replicas
    hd = t.shape[0] // n_heads
    return t[shard * per_rank * hd:(shard + 1) * per_rank * hd]


def test_kv_rows_matches_vllm():
    k = torch.arange(NKV * HD, dtype=torch.float32).reshape(NKV * HD, 1)
    for tp in (1, 2, 4, 8):
        for r in range(tp):
            assert torch.equal(C.kv_rows(k, r, tp, NKV), vllm_rows(k, r, tp, NKV)), (tp, r)
    assert not torch.equal(C.half_rows(k, 1, 8), vllm_rows(k, 1, 8, NKV))   # the old split was wrong here
    assert torch.equal(C.kv_rows(k, 3, 8, None), C.half_rows(k, 3, 8))        # no head count: old behaviour


def test_sobrat_attention_columns(tmp_path):
    torch.manual_seed(0)
    pfx = C.PFX + "0.self_attn."
    full = {"q": torch.randn(NQ * HD, R), "k": torch.randn(NKV * HD, R), "v": torch.randn(NKV * HD, R)}
    t = {}
    for n, B in full.items():
        t[pfx + f"{n}_proj.lora_A"] = torch.randn(R, HID)
        t[pfx + f"{n}_proj.lora_B"] = B
    save_file(t, str(tmp_path / "lora.safetensors"))
    save_file({"x": torch.zeros(1)}, str(tmp_path / "deabl.safetensors"))
    orig = C.Blob.pack
    C.Blob.pack = lambda self, pinned: orig(self, False)                       # no CUDA needed
    try:
        for tp in (2, 4, 8):
            for r in range(tp):
                (L, _st, host, ops), = C.sobrat(str(tmp_path / "lora.safetensors"), str(tmp_path / "deabl.safetensors"),
                                                r, tp, layer_types=["full_attention"], num_kv_heads=NKV)
                exp = [o for o in ops.tolist() if o[1] == 1]                    # expand ops: (mod, 1, off, n, r, col)
                want = [vllm_rows(full["q"], r, tp, NQ), vllm_rows(full["k"], r, tp, NKV), vllm_rows(full["v"], r, tp, NKV)]
                col = 0
                for (_m, _k, off, n, rr, c), B in zip(exp, want):
                    assert (n, c) == (B.shape[0], col), (tp, r, n, c, B.shape[0], col)
                    got = host[off:off + n * rr * 2].view(torch.float16).reshape(n, rr)
                    assert torch.equal(got, B.to(torch.float16)), (tp, r)
                    col += n
    finally:
        C.Blob.pack = orig
