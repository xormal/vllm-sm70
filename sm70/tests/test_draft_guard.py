"""FA2SM70_DRAFT_GUARD: a draft token id outside [0, vocabulary) must not reach the scheduler (CPU).

On 2026-10-14 production died with `OverflowError: Python integer 85899345940 out of bounds for int32` when
a garbage draft id was written into `token_ids_cpu` (VALIDATION.md, incident 2026-10-14). With the guard the
row is dropped (that request runs the step without speculation) and logged. The methods are taken from the
source file itself, so the test needs neither a GPU nor an importable vLLM.
"""
import ast
import logging
import os
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "vllm", "v1", "worker", "gpu_model_runner.py")
VOCAB = 248320


def runner(guard, buf, req_ids, lists=None):
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
    meths = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_get_draft_token_ids_cpu", "_черновик_в_словаре")]
    logs = []
    log = logging.getLogger("draft_guard_test")
    log.warning = lambda *a: logs.append(a[0] % a[1:])
    mod = {"torch": torch, "logger": log, "_ПРОВЕРКА_ЧЕРНОВИКА": guard}
    exec(compile(ast.Module(meths, []), SRC, "exec"), mod)
    r = types.SimpleNamespace()
    for name in ("_get_draft_token_ids_cpu", "_черновик_в_словаре"):
        setattr(r, name, types.MethodType(mod[name], r))
    r._гран_пропуск = lambda rows, ids: rows
    r.model_config = types.SimpleNamespace(get_vocab_size=lambda: VOCAB)
    r.draft_token_ids_event = types.SimpleNamespace(synchronize=lambda: None)
    r.draft_token_ids_cpu = torch.tensor(buf, dtype=torch.int64)
    r._draft_token_req_ids = req_ids
    r._draft_token_ids = lists if lists is not None else torch.zeros(1)
    r.input_batch = types.SimpleNamespace(req_ids=req_ids)
    return r, logs


@pytest.mark.parametrize("guard,buf,expect", [
    (True, [[1, 2, 3], [VOCAB - 1, 0, 5]], [[1, 2, 3], [VOCAB - 1, 0, 5]]),
    (True, [[1, 2, 3], [85899345940, 20, 20], [7, 8, 9]], [[1, 2, 3], [], [7, 8, 9]]),
    (True, [[-1, 2, 3], [VOCAB, 1, 1]], [[], []]),
    (False, [[1, 2, 3], [85899345940, 20, 20]], [[1, 2, 3], [85899345940, 20, 20]]),
])
def test_tensor_path(guard, buf, expect):
    r, logs = runner(guard, buf, [f"r{i}" for i in range(len(buf))])
    rows, _ = r._get_draft_token_ids_cpu()
    assert rows == expect
    assert bool(logs) == (rows != buf)


def test_list_path():
    r, logs = runner(True, [[0, 0, 0]], ["a", "b"], lists=[[1, 2, 3], [300000, 1, 2]])
    rows, _ = r._get_draft_token_ids_cpu()
    assert rows == [[1, 2, 3], []] and "0x493e0" in logs[0]
