"""A stand-in for `nvidia-smi` that answers from a recorded topology (tests and GPU-less dry runs).

    FAKE_Q=topologies/dgx1_8x_32g.query.txt FAKE_T=topologies/dgx1_8x_32g.topo.txt FAKE_SEL=0,3 \\
        python -I fake_nvidia_smi.py topo -m

FAKE_SEL lists host indices of the cards a container was given; they are renumbered 0..n-1, as CDI
does. Only the queries of scripts/serve_qwen38_clef.sh are answered: --query-gpu (index, uuid, name,
memory.total, memory.used = 0), -i <index|uuid>, topo -m.
"""
import os
import re
import sys


def main(argv):
    rows = [ln.split(", ") for ln in open(os.environ["FAKE_Q"]).read().splitlines() if ln.strip()]
    sel = [int(x) for x in os.environ.get("FAKE_SEL", ",".join(r[0] for r in rows)).split(",")]
    gpus = [{"index": str(n), "uuid": rows[h][1], "name": rows[h][2], "memory.total": rows[h][3],
             "memory.used": "0"} for n, h in enumerate(sel)]
    if argv[:2] == ["topo", "-m"]:
        lines = open(os.environ["FAKE_T"]).read().splitlines()
        hdr = re.sub(r"\x1b\[[0-9;]*m", "", lines[0]).split("\t")
        by_name = {ln.split("\t")[0]: ln.split("\t") for ln in lines[1:] if ln.startswith("GPU")}
        tail = [i for i, h in enumerate(hdr) if h and not re.fullmatch(r"GPU\d+", h)]
        print("\t" + "\t".join([f"GPU{n}" for n in range(len(sel))] + [hdr[i] for i in tail]))
        for n, h in enumerate(sel):
            r = by_name[f"GPU{h}"]
            print("\t".join([f"GPU{n}"] + [r[1 + hh] for hh in sel] + [r[i] for i in tail]))
        return 0
    pick = gpus
    if "-i" in argv:
        key = argv[argv.index("-i") + 1]
        pick = [g for g in gpus if key in (g["index"], g["uuid"])]
        if not pick:
            print(f"No devices were found matching {key}", file=sys.stderr)
            return 6
    fields = next(a.split("=", 1)[1] for a in argv if a.startswith("--query-gpu=")).split(",")
    for g in pick:
        print(", ".join(g[f] for f in fields))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
