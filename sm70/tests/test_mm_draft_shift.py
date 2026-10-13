"""The scheduler must run an image's encoder when the MTP/eagle drafter needs it (CPU, needs vllm).

The drafter reads the target's window shifted by one position. When a prefill chunk ends exactly at an
image's first token, the target does not touch the image but the drafter does; without
FA2SM70_MM_DRAFT_SHIFT=1 the scheduler skipped the encoder and the drafter hit `Encoder cache miss`
(EngineDeadError in production, 2026-10-13).
"""
import importlib
import os
import types

import pytest

pytest.importorskip("vllm")


def _scheduler(lever):
    os.environ["FA2SM70_MM_DRAFT_SHIFT"] = lever
    import vllm.v1.core.sched.scheduler as m
    return importlib.reload(m).Scheduler


def _run(Scheduler, nc, nn, shift, start=300, length=1200):
    from vllm.multimodal.inputs import PlaceholderRange
    ecm = types.SimpleNamespace(check_and_update_cache=lambda r, i: False, can_allocate=lambda r, i, b, n: True)
    me = types.SimpleNamespace(is_encoder_decoder=False, encoder_cache_manager=ecm, ec_connector=None,
                               scheduler_config=types.SimpleNamespace(disable_chunked_mm_input=False))
    f = types.SimpleNamespace(mm_position=PlaceholderRange(offset=start, length=length), identifier="img")
    req = types.SimpleNamespace(has_encoder_inputs=True, mm_features=[f])
    return Scheduler._try_schedule_encoder_inputs(me, req, nc, nn, 10 ** 6, shift)[0] == [0]


@pytest.fixture
def restore_env():
    old = os.environ.get("FA2SM70_MM_DRAFT_SHIFT")
    yield
    if old is None:
        os.environ.pop("FA2SM70_MM_DRAFT_SHIFT", None)
    else:
        os.environ["FA2SM70_MM_DRAFT_SHIFT"] = old
    import vllm.v1.core.sched.scheduler as m
    importlib.reload(m)


def test_chunk_ending_at_image_start(restore_env):
    off, on = _scheduler("0"), None
    assert not _run(off, 0, 300, 1)                 # the defect: drafter needs position 300, encoder skipped
    on = _scheduler("1")
    assert _run(on, 0, 300, 1)                      # fixed
    assert not _run(on, 0, 299, 1)                  # drafter window ends at 299: no extra encoder work
    assert not _run(on, 0, 300, 0)                  # no drafter: unchanged
    assert _run(on, 0, 305, 1) and _run(off, 0, 305, 1)
