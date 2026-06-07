"""Tests for the auto batch-size finder."""

import pytest

from newsrec.training.batch_finder import find_max_batch_size, search_largest_fit


def test_search_finds_threshold():
    # can_fit is monotonic: fits iff b <= 10
    assert search_largest_fit(lambda b: b <= 10, 1, 256) == 10


def test_search_refines_non_power_of_two():
    # doubling brackets 4(ok)/8(fail), binary refine -> 7
    assert search_largest_fit(lambda b: b <= 7, 1, 256) == 7


def test_search_caps_at_max_batch():
    # everything fits -> capped at max_batch
    assert search_largest_fit(lambda b: True, 1, 64) == 64


def test_search_min_only():
    assert search_largest_fit(lambda b: b <= 1, 1, 256) == 1


def test_search_raises_when_nothing_fits():
    with pytest.raises(RuntimeError):
        search_largest_fit(lambda b: False, 1, 256)


def test_find_max_batch_size_none_on_cpu():
    # auto-sizing is GPU-only; on CPU it returns None so callers fall back
    called = {"build": 0, "probe": 0}

    def build(bs):
        called["build"] += 1
        return {}

    def probe(batch):
        called["probe"] += 1

    assert find_max_batch_size(build, probe, device="cpu") is None
    assert called == {"build": 0, "probe": 0}  # never probed on CPU
