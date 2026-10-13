# Memory profiles

`scripts/serve_qwen38_clef.sh` picks `tp<TP>-<card class>.env` from the number of cards and the memory of
the smallest one (`PROFILE=` overrides). A profile is applied on top of `../qwen38-27b-int8-clef.env`;
values from the caller's environment win over both.

| Profile | Status | KV pool / card | Max context |
|---|---|---|---|
| `tp2-32g` | **validated** in production (2026-09 … 10) | 9.9 GiB | 262,144 |
| `tp4-32g` | experimental | 9.9 GiB | 262,144 |
| `tp4-16g` | experimental, sizes are estimates | 2.5 GiB | 98,304 |
| `tp8-32g` | experimental | 9.9 GiB | 262,144 |
| `tp8-16g` | experimental, sizes are estimates | 6 GiB | 262,144 |

**How the experimental numbers were derived.** In production (TP=2, 32 GB) everything except the KV pool
takes ~21.3 GiB per card: ~2 GiB that does not shrink with TP (CUDA context, NCCL buffers, graphs,
activation workspace, Clef head on rank 0) and ~19.3 GiB of weight shards. So at TP=4 the non-KV part is
~11.7 GiB and at TP=8 ~6.8 GiB. A token costs ~39.6 KiB of KV per card at TP=2 (9.9 GiB / 262,144),
about half at TP=4 (one KV head per rank instead of two, half the GDN state), and less again at TP=8,
where the KV head is replicated but GDN state halves. With ~1 GiB of headroom, a 16 GB V100 (~15.75 GiB
usable) leaves ~2.5–3 GiB of KV at TP=4.

**Calibrating a profile** (do this before serving one in production):
1. start it; vLLM refuses at start-up if one `MAX_MODEL_LEN` request does not fit the pool;
2. run the gates of docs/sm70/VALIDATION.md, at least `igla` at the profile's maximum context, `smes`
   and `kartinki ... clef`, while sampling `nvidia-smi --query-gpu=memory.free` every second;
3. keep >= 500 MiB free at the peak; raise `FA2SM70_KV_BYTES` / `MAX_MODEL_LEN` while that holds;
4. record the result here and flip the status.
A 16 GB card can be emulated on a 32 GB one by holding 16 GiB on it with another process.
