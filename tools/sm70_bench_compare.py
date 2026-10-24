#!/usr/bin/env python3
"""Check vllm-sm70 benchmark records and compare them with their baseline.

    python tools/sm70_bench_compare.py docs/sm70/results/*.json            # validate every record
    python tools/sm70_bench_compare.py new.json --baseline-dir docs/sm70/results

Validation covers the required fields of docs/sm70/benchmark-result.schema.json (standard library only, no
jsonschema dependency). A record with a "baseline" field is compared against the record of that id. The
comparison is refused when the benchmark contract differs. The contract is: GPU model and count, topology,
P2P, clock lock, model weights, KV dtype, prompt and output tokens, concurrency, prefix-cache state, kind.
Exit code 1 on invalid records, contract mismatch, or a regression beyond --tolerance on a *_median metric.
"""
import argparse
import glob
import json
import os
import sys

REQ = {
    "": ["schema_version", "id", "kind", "date", "hardware", "software", "model", "workload", "method", "results"],
    "hardware": ["gpu_model", "gpu_count", "topology", "p2p", "clocks", "driver"],
    "software": ["cuda", "torch", "commit", "lever_file"],
    "model": ["name", "weights", "kv_dtype"],
    "workload": ["description", "prompt_tokens", "output_tokens", "concurrency", "prefix_cache"],
    "method": ["warmup", "runs", "source"],
}
CONTRACT = [("kind",), ("hardware", "gpu_model"), ("hardware", "gpu_count"), ("hardware", "topology"),
            ("hardware", "p2p"), ("hardware", "clocks", "locked"), ("hardware", "clocks", "locked_mhz"),
            ("model", "weights"), ("model", "kv_dtype"), ("workload", "prompt_tokens"),
            ("workload", "output_tokens"), ("workload", "concurrency"), ("workload", "prefix_cache")]
# Direction by name: rates (*tok_s*, *_per_s*) are higher-is-better; times (*_sec_*, *_ms_*), latencies,
# errors and deltas are lower-is-better; everything else is higher-is-better.
LOWER_IS_BETTER = ("_sec_", "_ms_", "latency", "ttft", "error", "delta")


def lower_is_better(name):
    if "tok_s" in name or "_per_s" in name:
        return False
    return any(s in name for s in LOWER_IS_BETTER)


def get(rec, path):
    for k in path:
        if not isinstance(rec, dict) or k not in rec:
            return None
        rec = rec[k]
    return rec


def validate(rec):
    errs = []
    for sect, keys in REQ.items():
        obj = rec if not sect else rec.get(sect)
        if not isinstance(obj, dict):
            errs.append(f"missing section {sect}")
            continue
        errs += [f"missing {sect + '.' if sect else ''}{k}" for k in keys if k not in obj]
    if rec.get("schema_version") != 1:
        errs.append("schema_version must be 1")
    if rec.get("kind") != "production_traffic":
        for k in ("prompt_tokens", "output_tokens"):
            if not isinstance(get(rec, ("workload", k)), int):
                errs.append(f"workload.{k} must be an exact integer for kind {rec.get('kind')}")
        if get(rec, ("method", "runs")) and get(rec, ("method", "runs")) > 1:
            if not any(k.endswith("_median") for k in rec.get("results", {})):
                errs.append("runs > 1 but no *_median metric")
    return errs


def load(paths):
    recs = {}
    for p in paths:
        for r in json.load(open(p))["records"]:
            r["_file"] = p
            if r.get("id") in recs:
                print(f"duplicate id {r['id']} in {p} and {recs[r['id']]['_file']}")
            recs[r.get("id")] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline-dir", default=os.path.join(os.path.dirname(__file__), "..", "docs", "sm70", "results"))
    ap.add_argument("--tolerance", type=float, default=0.02, help="allowed relative regression of *_median")
    a = ap.parse_args()
    new = load(a.files)
    pool = load(sorted(glob.glob(os.path.join(a.baseline_dir, "*.json"))))
    pool.update(new)
    bad = 0
    for rid, r in new.items():
        errs = validate(r)
        if errs:
            bad += 1
            print(f"INVALID {rid}: " + "; ".join(errs))
            continue
        base_id = r.get("baseline")
        if not base_id:
            print(f"ok {rid} (no baseline)")
            continue
        b = pool.get(base_id)
        if b is None:
            bad += 1
            print(f"FAIL {rid}: baseline {base_id} not found")
            continue
        diff = [".".join(p) for p in CONTRACT if get(r, p) != get(b, p)]
        if diff:
            bad += 1
            print(f"FAIL {rid}: contract differs from {base_id}: {', '.join(diff)}")
            continue
        for k, v in r["results"].items():
            bv = b["results"].get(k)
            if not k.endswith("_median") or not isinstance(v, (int, float)) or not isinstance(bv, (int, float)) or not bv:
                continue
            rel = (v - bv) / abs(bv)
            worse = rel > a.tolerance if lower_is_better(k) else rel < -a.tolerance
            print(f"{'REGRESSION' if worse else 'ok'} {rid} {k}: {bv} -> {v} ({rel:+.1%})")
            bad += worse
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
