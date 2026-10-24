# Benchmarking contract

Every number published for this stack is a record in `docs/sm70/results/*.json`. The format is
[`docs/sm70/benchmark-result.schema.json`](docs/sm70/benchmark-result.schema.json). A new result is
compared automatically against the record it names as `baseline`:

```bash
python tools/sm70_bench_compare.py my_results.json                   # validate and compare
python tools/sm70_bench_compare.py docs/sm70/results/*.json          # validate the published records
```

The tool uses the standard library only. It exits with 1 on an invalid record, on a contract mismatch with
the baseline, or on a regression of a `*_median` metric beyond `--tolerance` (default 2 %).

## What a record must carry

| Group | Fields |
|---|---|
| Hardware | GPU model and count, GPU indices, topology (`NVLink (NVn)`, `PCIe switch (PIX/PXB)`, `PCIe host bridge (PHB)`, `SYS`), P2P on/off, clocks (locked or not, locked MHz; observed SM clock and temperature during the run), driver |
| Software | CUDA, torch, Python, commit SHA, lever file and any lever overrides, extra environment (e.g. `NCCL_P2P_DISABLE`) |
| Model | checkpoint name, weight format (e.g. `GPTQ int8 g128 + W12 q/k/v`), KV dtype, `max_model_len` |
| Workload | description, **exact** prompt tokens and output tokens (null only for production traffic), concurrency, prefix-cache state (`disabled`, `enabled, cold`, `enabled, warm`), speculation, optional prompt SHA-256 |
| Method | warmup count, number of runs, the tool and its arguments, path or URL of the raw log |
| Results | metric → value; timed metrics with more than one run give `*_median` and `*_best` |

## Rules

* **Contract.** Two records are comparable only if all of these are equal:
  * kind;
  * GPU model and count, topology, P2P, clock lock and locked MHz;
  * weight format, KV dtype;
  * prompt and output tokens, concurrency, prefix-cache state.
  
  The tool refuses the comparison otherwise.
* **Metric names state their unit and direction.** Rates contain `tok_s` or `_per_s` and are
  higher-is-better. Times contain `_sec_` or `_ms_`; they, latencies, errors and deltas are
  lower-is-better. Example: `prefill_tok_s_median`, `prefill_sec_best`, `clef_max_dp`.
* **Warm, long, honest.**
  * Measure under sustained load: a short run catches the boost clock and overstates by up to 24 % on V100.
  * Record clocks and temperature next to every timing.
  * Keep the prefix cache state explicit: a "second call hits the cache" assumption already produced a
    wrong decode curve once.
* **Production traffic** (`kind: production_traffic`) is wall-clock counters of a live server. Its averages
  measure delivered load, not capacity, and are never compared with a single-request benchmark.
* **Accuracy before speed.** A speed record of a configuration that has not passed the gates in
  [docs/sm70/VALIDATION.md](docs/sm70/VALIDATION.md) is marked as such in `workload.description`.
* **Comparisons with other runtimes.**
  * Only records that meet this contract on both sides go into a comparison.
  * 4-bit and FP4 formats are not compared with int8/int12 in the accuracy-oriented class.

## Baselines

| id | What |
|---|---|
| `prod-v100-nvlink-tp2-2026-10-24` | 40 h of production traffic, 2× V100 NVLink |
| `prod-v100-pcie-tp2-2026-10-24` | 40 h of production traffic, 2× V100 PCIe, P2P off |

Report: [docs/qwen38-27b-sm70-performance.md](docs/qwen38-27b-sm70-performance.md).
