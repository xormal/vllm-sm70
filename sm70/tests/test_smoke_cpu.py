# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# CPU-level API smoke test: no GPU / no CUDA build required. Verifies the package imports, the public
# API surface exists with the expected signatures, and the flash-attn-compatible layer is wired. Safe
# to run in a syntax/CI stage on a machine without an sm_70 device. Run:  python tests/test_smoke_cpu.py
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
from fa2_sm70 import flash_attn_interface as fa

ok = True


def want(cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")


# Public API surface
for name in ["attention", "attention_varlen", "flash_attn_func", "flash_attn_varlen_func",
             "decode", "decode_paged", "quantize_kv", "decode_quantized", "KVCache", "SM70_PEAK_TFLOPS",
             "SM70_SPEC_TFLOPS"]:
    want(hasattr(fa2_sm70, name), f"fa2_sm70.{name} exists")

# Signatures carry the modifier knobs we shipped
sig_attn = inspect.signature(fa2_sm70.attention).parameters
for p in ["causal", "scale", "window", "softcap", "alibi_slopes"]:
    want(p in sig_attn, f"attention(...) has '{p}'")
want("dropout_p" in inspect.signature(fa2_sm70.flash_attn_func).parameters, "flash_attn_func(...) has 'dropout_p'")
want("seqlens" in inspect.signature(fa2_sm70.decode_paged).parameters, "decode_paged(...) has 'seqlens'")

# flash-attn-compatible layer
for name in ["flash_attn_func", "flash_attn_qkvpacked_func", "flash_attn_kvpacked_func", "flash_attn_varlen_func"]:
    want(hasattr(fa, name), f"flash_attn_interface.{name} exists")
sig_fa = inspect.signature(fa.flash_attn_func).parameters
for p in ["dropout_p", "softmax_scale", "causal", "window_size", "softcap", "alibi_slopes"]:
    want(p in sig_fa, f"fa.flash_attn_func(...) has '{p}'")

# window mapping is pure-python (no CUDA): (left,right) -> (window, window_right)
want(fa._win_map((-1, -1)) == (0, -1), "window (-1,-1) -> (0,-1) off")
want(fa._win_map((63, 0)) == (64, 0), "window (63,0) -> (64,0) causal past-window")
want(fa._win_map((32, 32)) == (33, 32), "window (32,32) -> (33,32) two-sided (supported)")
want("window_right" in inspect.signature(fa2_sm70.attention).parameters, "attention(...) has 'window_right'")

want(fa2_sm70.SM70_PEAK_TFLOPS == 93.6, "SM70_PEAK_TFLOPS == 93.6 (measured HMMA ceiling)")
want(fa2_sm70.SM70_SPEC_TFLOPS == 125.3, "SM70_SPEC_TFLOPS == 125.3 (data-sheet peak)")

print("SMOKE_CPU:", "ALL OK" if ok else "FAILED")
assert ok
