"""scripts/serve_qwen38_clef.sh on recorded topologies, as a container would see them (DRYRUN=1, no GPU).

nvidia-smi is replaced by fake_nvidia_smi.py; the cards a container was given are renumbered 0..n-1.
Needs bash and a CUDA toolkit (the launcher probes nvcc before it prints the plan).
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "..", "..", "scripts", "serve_qwen38_clef.sh")
TOPO = os.path.join(HERE, "topologies")
CUDA_HOME = os.environ.get("CUDA_HOME") or (
    os.path.dirname(os.path.dirname(shutil.which("nvcc"))) if shutil.which("nvcc") else None)

pytestmark = pytest.mark.skipif(not CUDA_HOME or not os.path.isfile(f"{CUDA_HOME}/bin/nvcc") or not shutil.which("bash"),
                                reason="needs bash and a CUDA toolkit (CUDA_HOME or nvcc on PATH)")


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    d = tmp_path_factory.mktemp("dryrun")
    (d / "bin").mkdir()
    smi = d / "bin" / "nvidia-smi"
    smi.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" -I "{HERE}/fake_nvidia_smi.py" "$@"\n')
    smi.chmod(0o755)
    model = d / "model"
    (model / "clef").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    for f in ("clef_lora_r256_fp16.safetensors", "deabl_abl_fp16.safetensors", "joint_head_fp16.safetensors"):
        (model / "clef" / f).write_bytes(b"")
    return d


def launch(sandbox, fixture, sel=None, **env):
    e = {"PATH": f"{sandbox}/bin:{CUDA_HOME}/bin:/usr/bin:/bin", "HOME": str(sandbox), "PY": sys.executable,
         "CUDA_HOME": CUDA_HOME, "MODEL_DIR": str(sandbox / "model"), "STATE_DIR": str(sandbox / "state"),
         "DRYRUN": "1", "SM70_IN_CONTAINER": "1",
         "FAKE_Q": f"{TOPO}/{fixture}.query.txt", "FAKE_T": f"{TOPO}/{fixture}.topo.txt"}
    if sel:
        e["FAKE_SEL"] = sel
    e.update(env)
    p = subprocess.run(["bash", LAUNCHER], env=e, capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    if p.returncode:
        return {"error": out}
    m = re.search(r"cards (\S+), TP=(\d+), profile (\S+), max context (\d+)", out)
    assert m, out
    lev = dict(re.findall(r"^\s+(FA2SM70_\w+)=(.*)$", out, re.M))
    return {"cards": m[1], "tp": int(m[2]), "profile": m[3], "ctx": int(m[4]),
            "exch12": lev["FA2SM70_EXCH12"], "cvd": lev["FA2SM70_CVD"]}


@pytest.mark.parametrize("fixture,sel,tp,profile,exch12", [
    ("host4_sxm2_32g", "2,3", 2, "tp2-32g", "0"),       # the production pair, NVLink
    ("host4_sxm2_32g", "0,1", 2, "tp2-32g", "1"),       # PCIe pair: int12 exchange
    ("pcie8_32g_2socket", "0,1", 2, "tp2-32g", "1"),
    ("dgx1_8x_32g", "0,3", 2, "tp2-32g", "0"),
    ("station4_16g", None, 4, "tp4-16g", "0"),
    ("dgx1_8x_32g", None, 8, "tp8-32g", "0"),
    ("dgx1_8x_16g", None, 8, "tp8-16g", "0"),
])
def test_cards_profile_exchange(sandbox, fixture, sel, tp, profile, exch12):
    r = launch(sandbox, fixture, sel)
    assert "error" not in r, r.get("error")
    assert (r["tp"], r["profile"], r["exch12"]) == (tp, profile, exch12)
    assert r["cards"] == r["cvd"] == ",".join(map(str, range(tp)))


def test_cards_by_uuid(sandbox):
    u4, u7 = "GPU-00000000-0000-0000-0000-000000000004", "GPU-00000000-0000-0000-0000-000000000007"
    r = launch(sandbox, "dgx1_8x_32g", CARDS=f"{u4},{u7}")
    assert (r["tp"], r["exch12"], r["cvd"]) == (2, "0", f"{u4},{u7}")


def test_int12_refused_beyond_two_ranks(sandbox):
    assert launch(sandbox, "station4_16g", FA2SM70_EXCH12="1")["exch12"] == "0"


def test_profile_limits(sandbox):
    assert launch(sandbox, "station4_16g")["ctx"] == 98304
    assert launch(sandbox, "host4_sxm2_32g", "2,3", MAX_MODEL_LEN="65536")["ctx"] == 65536


@pytest.mark.parametrize("fixture,sel,msg", [
    ("station4_16g", "0,1", "16 GB cards need at least 4 GPUs"),
    ("dgx1_8x_32g", "0,1,2", "CARDS must name 2, 4 or 8 GPUs"),
])
def test_refusals(sandbox, fixture, sel, msg):
    assert msg in launch(sandbox, fixture, sel).get("error", "")
