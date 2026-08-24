"""llmcls.text 的截斷邏輯：純 Python id 操作，不需要真正的 tokenizer。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.text import NUM_SPECIAL_TOKENS, build_input_ids, split_budget, swap_ab_label, truncate_ids


def test_truncate_ids_noop_when_within_budget():
    assert truncate_ids([1, 2, 3], budget=10) == [1, 2, 3]


def test_truncate_ids_keeps_head_and_tail():
    ids = list(range(10))
    out = truncate_ids(ids, budget=4, head_ratio=0.5)
    assert len(out) == 4
    assert out == ids[:2] + ids[-2:]


def test_truncate_ids_zero_budget():
    assert truncate_ids([1, 2, 3], budget=0) == []


def test_truncate_ids_budget_of_one_keeps_head_only():
    assert truncate_ids([1, 2, 3], budget=1) == [1]


def test_split_budget_sums_to_total():
    for total in [0, 1, 2, 10, 511, 1000]:
        a, b, c = split_budget(total)
        assert a + b + c == total
        assert min(a, b, c) >= 0


def test_split_budget_matches_default_ratios_roughly():
    a, b, c = split_budget(1000)
    assert a == pytest.approx(200, abs=1)
    assert b == pytest.approx(400, abs=1)
    assert c == pytest.approx(400, abs=1)


def test_build_input_ids_respects_max_len():
    long_prompt = list(range(1000))
    long_a = list(range(1000))
    long_b = list(range(1000))
    max_len = 128
    p, a, b = build_input_ids(long_prompt, long_a, long_b, max_len)
    assert len(p) + len(a) + len(b) + NUM_SPECIAL_TOKENS <= max_len


def test_build_input_ids_no_truncation_when_short():
    p, a, b = build_input_ids([1, 2], [3, 4], [5, 6], max_len=512)
    assert (p, a, b) == ([1, 2], [3, 4], [5, 6])


def test_build_input_ids_rejects_too_small_max_len():
    with pytest.raises(ValueError, match="太小"):
        build_input_ids([1], [2], [3], max_len=NUM_SPECIAL_TOKENS)


def test_swap_ab_label_flips_a_and_b():
    assert swap_ab_label(0) == 1
    assert swap_ab_label(1) == 0


def test_swap_ab_label_keeps_tie():
    assert swap_ab_label(2) == 2


def test_swap_ab_label_is_its_own_inverse():
    for label in (0, 1, 2):
        assert swap_ab_label(swap_ab_label(label)) == label
