#!/usr/bin/env python3
"""Plan vllm-sm70 servers on the GPUs of a host: how many replicas, which cards go together, which
tensor-parallel exchange each group uses, and a docker compose file that pins every replica to its
cards by UUID.

    scripts/plan_gpus.py                                  # all GPUs, one replica per group (default)
    scripts/plan_gpus.py --gpus 2,3,5,6                   # only these cards (indices or GPU-UUIDs)
    scripts/plan_gpus.py --single                         # ONE server on all allowed cards (TP = their count)
    scripts/plan_gpus.py --tp 4 --out deploy/             # force TP, write compose.yaml, nginx.conf, plan.json
    scripts/plan_gpus.py --query q.csv --topo t.txt       # plan from saved nvidia-smi output (other hosts, tests)

Rules (docs/sm70/DOCKER.md explains them):
* Group size = the smallest TP whose weight shard fits the SMALLEST allowed card: 32 GB -> 2, 16 GB -> 4.
  --tp raises it (2, 4 or 8); --single uses all allowed cards as one group.
* Cards are split into groups that maximise the WEAKEST link inside a group first (a TP all-reduce runs
  at the pace of its slowest hop), then the total link quality, then NUMA locality. Leftover cards that
  do not fill a group stay unused.
* Exchange: a pair on NVLink -> fp16 NCCL; a pair without NVLink -> int12-compressed exchange
  (FA2SM70_EXCH12=1, measured gain on PCIe, loss on NVLink); groups of 4 or 8 -> fp16 NCCL, which routes
  over NVLink where it exists.
* Only tp2-32g is validated end to end (production). Other profiles are marked experimental.
Standard library only.
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys

TP_CHOICES = (2, 4, 8)
VALIDATED = {"tp2-32g"}


# ------------------------------------------------------------------ reading nvidia-smi
def _run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def parse_query(text):
    """`nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits`"""
    gpus = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        idx, uuid, name, mem = [x.strip() for x in line.split(",")[:4]]
        gpus.append({"index": int(idx), "uuid": uuid, "name": name, "mem_mib": int(float(mem))})
    return gpus


def parse_topo(text):
    """`nvidia-smi topo -m` -> ({(i, j): link}, {i: cpu_affinity}). NIC rows/columns are ignored."""
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", ln).rstrip("\n") for ln in text.splitlines()]
    head = next(ln for ln in lines if re.search(r"\bGPU0\b", ln) and not re.match(r"GPU\d+\s", ln))
    cols = [c.strip() for c in head.split("\t")]
    gpu_cols = [c for c in cols if re.fullmatch(r"GPU\d+", c)]
    links, aff = {}, {}
    for ln in lines:
        f = ln.split("\t")
        if not f or not re.fullmatch(r"GPU\d+", f[0].strip()):
            continue
        i = int(f[0].strip()[3:])
        vals = [x.strip() for x in f[1:]]
        for c, v in zip(gpu_cols, vals[:len(gpu_cols)]):
            j = int(c[3:])
            if i != j:
                links[(i, j)] = v
        rest = [v for v in vals[len(gpu_cols):]]
        aff[i] = rest[0] if rest else ""
    return links, aff


def link_score(v):
    """NVLink outranks every PCIe path; within PCIe, closer is better."""
    m = re.fullmatch(r"NV(\d+)", v or "")
    if m:
        return 10 + int(m.group(1))
    return {"PIX": 4, "PXB": 3, "PHB": 2, "NODE": 1, "SYS": 0, "SOC": 0}.get(v, 0)


def is_nvlink(v):
    return bool(re.fullmatch(r"NV\d+", v or ""))


# ------------------------------------------------------------------ planning
def mem_class(mib):
    if mib >= 30000:
        return "32g"
    if mib >= 15000:
        return "16g"
    raise SystemExit(f"plan: a {mib} MiB card is too small for the int8 27B model (needs >= 16 GB cards)")


def min_tp(cls):
    return {"32g": 2, "16g": 4}[cls]


def group_key(g, links, aff):
    pairs = list(itertools.combinations(g, 2))
    scores = [link_score(links.get((a, b))) for a, b in pairs]
    same_numa = len({aff.get(x, "") for x in g}) == 1
    return (min(scores), sum(scores), int(same_numa))


def best_partition(cards, tp, links, aff):
    """Exhaustive over set partitions (<= 8 cards: at most 105 pairings / 35 quad splits).
    Objective: lexicographic on the sorted group keys, worst group first (max-min)."""
    n_groups = len(cards) // tp
    best = None

    def rec(rest, groups, skips_left):
        nonlocal best
        if len(groups) == n_groups:
            key = tuple(sorted(group_key(g, links, aff) for g in groups))
            if best is None or key > best[0]:
                best = (key, [tuple(g) for g in groups])
            return
        if len(rest) < tp:
            return
        first, others = rest[0], rest[1:]
        for comb in itertools.combinations(others, tp - 1):
            g = (first,) + comb
            rec([c for c in others if c not in comb], groups + [g], skips_left)
        if skips_left > 0:                              # leave `first` unused
            rec(others, groups, skips_left - 1)

    rec(sorted(cards), [], len(cards) - n_groups * tp)
    return best[1] if best else []


def exchange(g, links):
    pairs = list(itertools.combinations(g, 2))
    nv = [is_nvlink(links.get(p)) for p in pairs]
    if len(g) == 2:
        if nv[0]:
            return {"FA2SM70_EXCH12": "0"}, "NVLink pair: fp16 NCCL all-reduce"
        return {"FA2SM70_EXCH12": "1"}, f"no NVLink ({links.get(pairs[0])}): int12-compressed exchange"
    kind = "all pairs NVLink" if all(nv) else ("partial NVLink" if any(nv) else "no NVLink")
    return {"FA2SM70_EXCH12": "0"}, f"{kind}: fp16 NCCL all-reduce (int12 exchange loses beyond 2 ranks)"


def resolve_allowed(spec, gpus):
    if not spec:
        return [g["index"] for g in gpus]
    by_uuid = {g["uuid"]: g["index"] for g in gpus}
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok.startswith("GPU-"):
            if tok not in by_uuid:
                raise SystemExit(f"plan: no GPU with UUID {tok}")
            out.append(by_uuid[tok])
        else:
            i = int(tok)
            if i not in {g["index"] for g in gpus}:
                raise SystemExit(f"plan: no GPU with index {i}")
            out.append(i)
    if len(set(out)) != len(out):
        raise SystemExit("plan: --gpus lists a card twice")
    return out


def plan(gpus, links, aff, allowed=None, tp=None, single=False):
    by_idx = {g["index"]: g for g in gpus}
    cards = resolve_allowed(allowed, gpus)
    warn = []
    if not all("V100" in by_idx[c]["name"] for c in cards):
        warn.append("not every card is a V100: the sm70 kernels target Volta only")
    cls_set = {mem_class(by_idx[c]["mem_mib"]) for c in cards}
    cls = "16g" if "16g" in cls_set else "32g"
    if len(cls_set) > 1:
        warn.append("mixed 16 GB and 32 GB cards: everything is sized for 16 GB")
    need = min_tp(cls)
    if single:
        tp = len(cards)
    tp = tp or need
    if tp not in TP_CHOICES:
        raise SystemExit(f"plan: TP must be one of {TP_CHOICES} (got {tp}; with --single it is the card count)")
    if tp < need:
        raise SystemExit(f"plan: TP={tp} does not fit {cls} cards (minimum TP={need})")
    if len(cards) < tp:
        raise SystemExit(f"plan: {len(cards)} allowed card(s), TP={tp} needs {tp}")
    groups = best_partition(cards, tp, links, aff)
    unused = sorted(set(cards) - {c for g in groups for c in g})
    if unused:
        warn.append(f"cards {unused} stay unused: {len(cards)} cards do not split into groups of {tp}")
    profile = f"tp{tp}-{cls}"
    out = []
    for k, g in enumerate(groups):
        env, why = exchange(g, links)
        mn, tot, numa = group_key(g, links, aff)
        out.append({
            "replica": k, "cards": list(g), "uuids": [by_idx[c]["uuid"] for c in g], "tp": tp,
            "profile": profile, "validated": profile in VALIDATED, "env": env, "exchange": why,
            "links": {f"{a}-{b}": links.get((a, b), "?") for a, b in itertools.combinations(g, 2)},
            "same_numa": bool(numa), "cpuset": aff.get(g[0], "") if numa else "",
        })
    return {"tp": tp, "profile": profile, "replicas": out, "unused": unused, "warnings": warn}


# ------------------------------------------------------------------ output
def render_text(p):
    lines = [f"profile {p['profile']} ({'validated' if p['profile'] in VALIDATED else 'EXPERIMENTAL: not validated on hardware'}), "
             f"{len(p['replicas'])} replica(s) x TP={p['tp']}"]
    for r in p["replicas"]:
        if len(r["cards"]) <= 4:
            ln = "links " + " ".join(f"{k}:{v}" for k, v in r["links"].items())
        else:
            kinds = {}
            for v in r["links"].values():
                k = "NVLink" if is_nvlink(v) else v
                kinds[k] = kinds.get(k, 0) + 1
            ln = f"{len(r['links'])} pairs: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda x: -x[1]))
        lines.append(f"  replica {r['replica']}: GPUs {','.join(map(str, r['cards']))}  {ln}  -> {r['exchange']}"
                     + ("" if r["same_numa"] else "  (spans NUMA nodes)"))
    for w in p["warnings"]:
        lines.append(f"  warning: {w}")
    return "\n".join(lines)


def render_compose(p, base_port, runtime):
    reps = p["replicas"]
    y = ["# Generated by scripts/plan_gpus.py -- one vllm-sm70 replica per GPU group, nginx in front.",
         f"# Plan: profile {p['profile']}, {len(reps)} replica(s) x TP={p['tp']}.",
         "# Run:  MODEL_DIR=/path/to/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef docker compose up -d",
         "name: vllm-sm70", "services:"]
    for r in reps:
        name = f"sm70-{r['replica']}"
        y += [f"  {name}:",
              "    image: ${SM70_IMAGE:-vllm-sm70:latest}",
              "    restart: unless-stopped",
              "    shm_size: 8g",
              "    ulimits: {memlock: -1, stack: 67108864}"]
        if runtime == "cdi":
            y.append("    devices:")
            y += [f"      - nvidia.com/gpu={u}" for u in r["uuids"]]
        else:
            y += ["    deploy:", "      resources:", "        reservations:", "          devices:",
                  "            - driver: nvidia",
                  f"              device_ids: [{', '.join(repr(u) for u in r['uuids'])}]",
                  "              capabilities: [gpu]"]
        if r["cpuset"]:
            y.append(f"    cpuset: \"{r['cpuset']}\"")
        env = dict(r["env"], PROFILE=r["profile"], SERVED_NAMES="${SERVED_NAMES:-Qwen3.8-27B-ABLITERATED}",
                   VLLM_API_KEY="${VLLM_API_KEY:-}", ALLOWED_MEDIA_DOMAINS="${ALLOWED_MEDIA_DOMAINS:-}")
        y.append("    environment:")
        y += [f"      {k}: \"{v}\"" for k, v in env.items()]
        y += ["    volumes:",
              "      - ${MODEL_DIR:?set MODEL_DIR}:/models/model:ro",
              f"      - {name}-state:/state",
              "    ports:",
              f"      - \"127.0.0.1:{base_port + 1 + r['replica']}:8000\""]
    y += ["  router:",
          "    image: nginx:1.27-alpine",
          "    restart: unless-stopped",
          "    depends_on: [" + ", ".join(f"sm70-{r['replica']}" for r in reps) + "]",
          "    volumes:", "      - ./nginx.conf:/etc/nginx/nginx.conf:ro",
          "    ports:", f"      - \"${{HOST_PORT:-{base_port}}}:8000\"",
          "volumes:"]
    y += [f"  sm70-{r['replica']}-state:" for r in reps]
    return "\n".join(y) + "\n"


def render_nginx(p):
    ups = "\n".join(f"    server sm70-{r['replica']}:8000 max_fails=0;" for r in p["replicas"])
    return f"""# Generated by scripts/plan_gpus.py. least_conn spreads load; long prompts make requests very uneven.
events {{}}
http {{
  client_max_body_size 64m;
  upstream sm70 {{
    least_conn;
{ups}
  }}
  server {{
    listen 8000;
    location / {{
      proxy_pass http://sm70;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_buffering off;              # streaming
      proxy_read_timeout 3600s;         # a 250K-token prefill takes minutes
      proxy_send_timeout 3600s;
    }}
  }}
}}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpus", default=os.environ.get("ALLOWED_GPUS", ""), help="allowed cards: indices or GPU-UUIDs, comma-separated")
    ap.add_argument("--tp", type=int, choices=TP_CHOICES)
    ap.add_argument("--single", action="store_true", help="one server on all allowed cards (experimental for 8)")
    ap.add_argument("--query", help="saved output of nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits")
    ap.add_argument("--topo", help="saved output of nvidia-smi topo -m")
    ap.add_argument("--out", help="write compose.yaml, nginx.conf and plan.json here")
    ap.add_argument("--port", type=int, default=8000, help="router port; replicas get 127.0.0.1:port+1..")
    ap.add_argument("--runtime", choices=("cdi", "nvidia"), default="cdi", help="how compose requests GPUs")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    q = open(a.query).read() if a.query else _run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                                                   "--format=csv,noheader,nounits"])
    t = open(a.topo).read() if a.topo else _run(["nvidia-smi", "topo", "-m"])
    gpus = parse_query(q)
    links, aff = parse_topo(t)
    p = plan(gpus, links, aff, a.gpus, a.tp, a.single)
    print(json.dumps(p, indent=2) if a.json else render_text(p))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "compose.yaml"), "w") as f:
            f.write(render_compose(p, a.port, a.runtime))
        with open(os.path.join(a.out, "nginx.conf"), "w") as f:
            f.write(render_nginx(p))
        with open(os.path.join(a.out, "plan.json"), "w") as f:
            json.dump(p, f, indent=2)
        print(f"wrote {a.out}/compose.yaml, nginx.conf, plan.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
