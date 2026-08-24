"""llmcls.training_safety 的煞車判斷邏輯：純 Python，不需要 torch/GPU。"""

from __future__ import annotations

import sys
from pathlib import Path

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.training_safety import should_stop_for_nonfinite


def test_healthy_log_does_not_stop():
    should_stop, count = should_stop_for_nonfinite({"loss": 1.09, "grad_norm": 2.5}, 0)
    assert should_stop is False
    assert count == 0


def test_nonfinite_loss_stops_immediately():
    should_stop, _ = should_stop_for_nonfinite({"loss": float("nan"), "grad_norm": 3.0}, 0)
    assert should_stop is True


def test_single_nonfinite_grad_norm_does_not_stop():
    # 這是 group_by_length 那次的誤判場景：loss 健康、只有 grad_norm 是 inf。
    should_stop, count = should_stop_for_nonfinite({"loss": 1.079, "grad_norm": float("inf")}, 0)
    assert should_stop is False
    assert count == 1


def test_grad_norm_recovers_resets_counter():
    _, count = should_stop_for_nonfinite({"loss": 1.08, "grad_norm": float("inf")}, 0)
    assert count == 1
    should_stop, count = should_stop_for_nonfinite({"loss": 1.08, "grad_norm": 4.0}, count)
    assert should_stop is False
    assert count == 0


def test_consecutive_nonfinite_grad_norm_stops():
    count = 0
    should_stop = False
    for _ in range(3):
        should_stop, count = should_stop_for_nonfinite({"loss": 1.08, "grad_norm": float("inf")}, count)
    assert should_stop is True
    assert count == 3


def test_custom_patience():
    count = 0
    should_stop = False
    for _ in range(2):
        should_stop, count = should_stop_for_nonfinite(
            {"loss": 1.08, "grad_norm": float("inf")}, count, grad_norm_patience=2
        )
    assert should_stop is True
