"""llmcls.calibration 的 temperature scaling：純 numpy 運算，不需要真正的模型。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.calibration import apply_temperature, fit_temperature
from llmcls.metrics import log_loss, softmax


def test_apply_temperature_one_is_identity():
    logits = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    np.testing.assert_allclose(apply_temperature(logits, 1.0), softmax(logits))


def test_apply_temperature_large_t_flattens_toward_uniform():
    logits = np.array([[10.0, 0.0, 0.0]])
    probs = apply_temperature(logits, temperature=1e6)
    np.testing.assert_allclose(probs, [[1 / 3, 1 / 3, 1 / 3]], atol=1e-3)


def test_fit_temperature_reduces_overconfident_and_partly_wrong_logits():
    """overconfident 且有部分預測錯誤的模型，配溫度（T > 1，撫平過度自信）應該能
    降低 log loss —— 這是 temperature scaling 在這題上的核心價值：log loss 在意
    機率校準，不是準確率。
    """
    rng = np.random.default_rng(0)
    n = 300
    labels = rng.integers(0, 3, size=n)
    correct = rng.random(n) < 0.6
    wrong_labels = (labels + 1) % 3
    logits = np.full((n, 3), -2.0)
    logits[np.arange(n)[correct], labels[correct]] = 4.0
    logits[np.arange(n)[~correct], wrong_labels[~correct]] = 4.0

    baseline_loss = log_loss(labels, softmax(logits))
    t = fit_temperature(logits, labels, hi=15.0)
    calibrated_loss = log_loss(labels, apply_temperature(logits, t))

    assert t > 1.0, "overconfident 的 logits 應該要配出 T > 1 才能撫平"
    assert calibrated_loss < baseline_loss


def test_fit_temperature_stays_near_one_when_labels_match_model_confidence():
    """label 直接照模型自己算出的機率分佈抽樣——這樣模型的機率『就是』真正的資料
    生成機率，定義上已經完美校準，配出來的 T 應該落在 1 附近（而不是被拉去 0 或
    邊界，那代表模型過度自信或不夠自信，需要校準）。
    """
    rng = np.random.default_rng(1)
    n = 2000
    logits = rng.normal(scale=1.0, size=(n, 3))
    probs = softmax(logits)
    labels = np.array([rng.choice(3, p=probs[i]) for i in range(n)])
    t = fit_temperature(logits, labels)
    assert 0.7 < t < 1.4
