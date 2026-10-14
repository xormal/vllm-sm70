"""Sparse-decode selection buffers must outlive a growth step when FA2SM70_KEEP_BUFS asks for it (CPU, torch).

`_sel_bufs` / `_sel_bufs_h` in vllm/v1/attention/backends/fa2_sm70.py are shared by every attention layer and
replaced when the virtual batch grows. Captured CUDA graphs keep the old address, so the old tuple must be kept,
and FA2SM70_SELBUF_PROBE must report writes to it (VALIDATION.md, incident 2026-10-14). The functions are taken
from the source file itself, so the test needs neither a GPU nor an importable vLLM.
"""
import ast
import contextlib
import gc
import io
import os
import weakref

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "vllm", "v1", "attention", "backends", "fa2_sm70.py")
NAMES = {"_КАКИЕ_ДЕРЖАТЬ", "_ДЕРЖАТЬ_БУФЕРЫ", "_ДЕРЖАТЬ_МОД", "_ПЕНСИЯ_МОДУЛЯ", "_отставить_мод",
         "_SELBUF_PROBE", "_SELBUF_ZNAK", "_SELBUF_SLED", "_SELBUF_SCH", "_selbuf_otstavit",
         "_selbuf_proverit", "_selbuf_shag", "_SELBUF", "_sel_bufs", "_SELBUF_H", "_sel_bufs_h"}


def load(monkeypatch, keep, probe):
    monkeypatch.setenv("FA2SM70_KEEP_BUFS", keep)
    monkeypatch.setenv("FA2SM70_SELBUF_PROBE", probe)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    nodes = []
    for n in ast.parse(open(SRC, encoding="utf-8").read()).body:
        name = (n.name if isinstance(n, ast.FunctionDef) else
                n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else
                n.target.id if isinstance(n, ast.AnnAssign) else None)
        if name in NAMES:
            nodes.append(n)
    mod = {"os": os, "torch": torch}
    exec(compile(ast.Module(nodes, []), SRC, "exec"), mod)
    assert NAMES <= set(mod), NAMES - set(mod)
    return mod


def grow(mod, stale_write):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mod["_sel_bufs"](8, 2, 600, "cpu")
        mod["_sel_bufs_h"](8, 2, 3600, "cpu")
        old = weakref.ref(mod["_SELBUF"]["cpu"][0])
        old_h = weakref.ref(mod["_SELBUF_H"]["cpu"][0])
        old_idx = mod["_SELBUF"]["cpu"][1]
        mod["_sel_bufs"](12, 2, 600, "cpu")            # growth: the tuple is replaced
        mod["_sel_bufs_h"](12, 2, 3600, "cpu")
        if stale_write:
            old_idx[3, 1, 7] = 20                      # what a replay of a graph captured before the growth does
        del old_idx
        for _ in range(4):
            mod["_sel_bufs"](12, 2, 600, "cpu")
    gc.collect()
    return old() is not None, old_h() is not None, "ИЗМЕНЁН" in out.getvalue()


@pytest.mark.parametrize("keep,probe,stale,expect", [
    ("0", "0", False, (False, False, False)),        # default: released as before
    ("4", "0", False, (True, True, False)),          # module caches kept
    ("1", "2", False, (True, True, False)),          # probe silent without a stale write
    ("1", "2", True, (True, True, True)),            # probe reports a stale write
    ("2", "2", True, (False, False, False)),         # fields-only mask does not keep module caches
])
def test_selbuf_keep(monkeypatch, keep, probe, stale, expect):
    assert grow(load(monkeypatch, keep, probe), stale) == expect
