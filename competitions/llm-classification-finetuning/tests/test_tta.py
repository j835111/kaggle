"""llmcls.tta 的欄位對齊邏輯：純 numpy 運算，不需要真正的模型。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.tta import average_swapped


def test_average_swapped_realigns_ab_columns():
    """交換後模型強烈偏好『目前放在 A 位置』的那個——但那其實是原本的
    response_b。對齊後應該變成 winner_model_b 的機率較高，不能誤判成
    winner_model_a 贏（這正是忘記換欄位順序會踩到的 bug）。
    """
    probs_orig = np.array([[0.34, 0.33, 0.33]])  # 原始順序幾乎持平
    probs_swapped = np.array([[0.9, 0.05, 0.05]])  # 交換後強烈偏好「目前 A 位置」= 原本的 response_b
    result = average_swapped(probs_orig, probs_swapped)
    assert result[0, 1] > result[0, 0], "對調後的欄位沒有換回來，錯把 B 贏判成 A 贏"


def test_average_swapped_is_plain_mean_after_realignment():
    probs_orig = np.array([[0.5, 0.3, 0.2]])
    probs_swapped = np.array([[0.4, 0.4, 0.2]])  # 對齊後變成 [0.4, 0.4, 0.2]（前兩欄相等）
    result = average_swapped(probs_orig, probs_swapped)
    np.testing.assert_allclose(result, [[0.45, 0.35, 0.2]])


def test_average_swapped_preserves_row_sum():
    probs_orig = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    probs_swapped = np.array([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])
    result = average_swapped(probs_orig, probs_swapped)
    np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])


def test_average_swapped_works_on_logits_too():
    """同一個函式對 logits（不必和為 1、可以是負值）也適用——欄位重排 + 平均
    這個操作本身跟輸入是機率還是 logits 無關。
    """
    logits_orig = np.array([[1.0, 0.0, -1.0]])
    logits_swapped = np.array([[5.0, 0.0, -1.0]])  # 強烈偏好「目前 A 位置」= 原本的 response_b
    result = average_swapped(logits_orig, logits_swapped)
    assert result[0, 1] > result[0, 0]


def test_average_swapped_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        average_swapped(np.zeros((2, 3)), np.zeros((3, 3)))
