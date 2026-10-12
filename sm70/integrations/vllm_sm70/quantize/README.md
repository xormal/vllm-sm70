# int8 weights on Volta: fp16 -> GPTQ-format W8A16

## Why GPTQ and not compressed-tensors

On `sm_70` (capability 70) most int8 paths are closed. Measured from this vLLM build's own
capability gates:

| path | `get_min_capability` | usable on V100 |
|---|---|---|
| `compressed-tensors` wNa16 (W8A16) | 75 | no |
| `compressed-tensors` W8A8-int8 | 75 | no |
| `bitsandbytes` | 70 | yes, but not installed and slow at inference |
| **`gptq`** | **60** | **yes** — and `torch.ops._C.gptq_gemm` is present in the build |
| `awq` | 70 | yes (what ships today, 4-bit) |

So GPTQ 8-bit is the route.

## Why RTN and not real GPTQ

At 8 bits the Hessian-based error compensation buys very little over round-to-nearest -- the grid is
already fine -- while costing hours of calibration and a new dependency. Measured weight fidelity of
plain per-group RTN at 8 bits / group 128: **relL2 6.0e-3**, and the round trip through the written
checkpoint reproduces exactly that, so nothing is lost in the packing.

## The packing format was validated BEFORE converting 52 GB

`gptq_pack_test.py` builds one small layer, packs it, and runs vLLM's own `gptq_gemm` against a
reference dequantisation, sweeping the conventions that differ between GPTQ variants. Only one
combination is right, and the first two guesses were off by relL2 1.3 (pure noise) -- which is why
this is a separate, ten-second test rather than something discovered three hours into a conversion:

| zero-1 | shuffle | exllama | v2 | relL2 |
|---|---|---|---|---|
| **yes** | no | no | **no** | **1.09e-3** |
| no | no | no | yes | 1.09e-3 |
| yes | no | no | yes | 1.76e-2 |
| any | yes | any | any | >= 1.3 |

i.e. pack `zero-1` and declare classic `gptq` (or pack `zero` and declare `gptq_v2`); never
pre-shuffle, and let vLLM choose exllama itself.

## Usage

```bash
python to_int8_gptq.py <fp16-model-dir> <output-dir>
```
The module list mirrors the existing AWQ checkpoint's `modules_to_not_convert` exactly (visual tower,
`self_attn.{q,k,v}_proj`, `linear_attn.in_proj_{a,b}`, layer 0, mtp, lm_head are left fp16), so vLLM
builds the same module structure it already builds -- one variable at a time. Skips are expressed as
GPTQ `dynamic` negative patterns (`-:<regex>`), which is what this vLLM reads.

## Two vLLM patches are required

See `../patches/`. Without them the server boots, serves, and emits pure garbage. Both failures are
about vLLM FUSING modules before asking whether they are quantised, so the checkpoint's skip list
must name the fused modules (`self_attn.qkv_proj`, `linear_attn.in_proj_ba`) and ALSO carry
`modules_to_not_convert` in the AWQ spelling, which a model-level heuristic reads.

## Measured, end to end

Qwen3.6-27B-abliterated: fp16 52 GB -> **int8 32 GB** (346 modules) -> for reference AWQ 4-bit 21 GB.
Loaded footprint per GPU at TP=2: **17.7 GB** (int8) vs **12.43 GB** (AWQ 4-bit).

Same server, same benchmark, unique prompts, TP=2 on 2xV100, fp8_e4m3 KV:

| prompt tok | prefill int8 | prefill AWQ4 | ratio | decode int8 | decode AWQ4 | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4 119 | 877 | 860 | 1.02 | 32.2 | 41.8 | 0.77 |
| 16 368 | 872 | 800 | 1.09 | 30.4 | 36.8 | 0.83 |
| 32 743 | 824 | 793 | 1.04 | 30.1 | 26.5 | 1.14 |
| 65 456 | 747 | 687 | 1.09 | 26.9 | 23.7 | 1.14 |
| 131 027 | 613 | 562 | 1.09 | 20.9 | 19.2 | 1.09 |

This corrects the prediction that was written here first ("decode should drop ~15%"). **Prefill is
slightly FASTER on int8** (1.02-1.09x) -- prefill is compute-bound, not weight-bandwidth-bound, so
the larger weights cost nothing and the GPTQ 8-bit dequant path is a little cheaper than AWQ's on
Volta. **Decode is worse only at SHORT context** (0.77x at 4K), where reading the weights dominates;
past ~32K, where KV traffic dominates instead, int8 is ahead (1.09-1.14x).

The real cost is memory, and it is a context-length cost. int8 weights take 5.3 GB/GPU more, which
must come out of the KV pool: 6 GiB of KV (vs 12 GiB for AWQ) after an OOM at 9 GiB during a 32K
prefill. That caps context at **~377K tokens** (concurrency 1.44x for 256K), where AWQ 4-bit reaches
**512K+** (verified: a real 508,497-token request answered correctly). So the choice is
int8 + <=256K, or AWQ 4-bit + 512K.
